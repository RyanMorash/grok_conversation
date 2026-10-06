"""Shared xAI API helpers (chat + live search via xai-sdk)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

import grpc
from grpc import aio as grpc_aio
from xai_sdk import AsyncClient
from xai_sdk.chat import assistant, image, system, tool, tool_result, user
from xai_sdk.proto import chat_pb2
from xai_sdk.search import SearchParameters, web_source, x_source

from .const import (
    LIVE_SEARCH_FULL,
    LIVE_SEARCH_OFF,
    LIVE_SEARCH_WEB,
    LIVE_SEARCH_X,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_FALLBACK_MODEL,
    RECOMMENDED_FAST_MODEL,
    RETIRED_MODELS,
)

CLIENT_TIMEOUT_SECONDS = 120.0
PROBE_TIMEOUT_SECONDS = 10.0

# Substrings that mark non-chat models returned by the models API
_NON_CHAT_MODEL_MARKERS: tuple[str, ...] = (
    "imagine",
    "image",
    "video",
    "tts",
    "stt",
    "voice",
    "embedding",
    "embed",
    "whisper",
    "moderation",
    "realtime",
    "audio",
    "speech",
)

# Known-good fallbacks if the models API is unreachable (current models only).
_FALLBACK_CHAT_MODELS: tuple[str, ...] = (
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_FAST_MODEL,
    RECOMMENDED_FALLBACK_MODEL,
    "grok-4.7",
    "grok-4.6",
    "grok-4.5",
    "grok-4.5-latest",
    "grok-4.3",
    "grok-4-latest",
    "grok-4",
)


class XAIError(Exception):
    """Base error talking to the xAI API."""


class XAIAuthError(XAIError):
    """API key is missing, invalid, or lacks permission."""


class XAIConnectionError(XAIError):
    """Could not reach the xAI API (network / timeout)."""


class XAIRateLimitError(XAIError):
    """Rate limited or quota exhausted."""


class XAIInvalidArgumentError(XAIError):
    """The request was rejected as invalid or unsupported."""


class CombinedSearchRejected(Exception):
    """Tools and live search were rejected together on one chat.create."""


@dataclass(slots=True)
class ChatToolCall:
    """Normalized client-side tool call from a chat response."""

    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class ChatResult:
    """Normalized chat sample result (SDK-agnostic)."""

    content: str
    tool_calls: list[ChatToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    citations: list[Any] = field(default_factory=list)


def create_xai_client(
    api_key: str, *, timeout: float = CLIENT_TIMEOUT_SECONDS
) -> AsyncClient:
    """Create an async xAI gRPC client."""
    return AsyncClient(api_key=api_key, timeout=timeout)


async def close_xai_client(client: Any) -> None:
    """Close a gRPC client if it exposes ``close``."""
    close = getattr(client, "close", None)
    if close is None:
        return
    result = close()
    if hasattr(result, "__await__"):
        await result


def map_xai_error(err: BaseException) -> XAIError:
    """Map a gRPC / SDK exception onto the integration's error types."""
    if isinstance(err, XAIError):
        return err
    details = str(err)
    code = None
    if isinstance(err, (grpc.RpcError, grpc_aio.AioRpcError)):
        try:
            code = err.code()
        except Exception:  # noqa: BLE001
            code = None
        try:
            details = err.details() or details
        except Exception:  # noqa: BLE001
            pass
        if code in (
            grpc.StatusCode.UNAUTHENTICATED,
            grpc.StatusCode.PERMISSION_DENIED,
        ):
            return XAIAuthError(details)
        if code in (
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
        ):
            return XAIConnectionError(details)
        if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
            return XAIRateLimitError(details)
        if code in (
            grpc.StatusCode.INVALID_ARGUMENT,
            grpc.StatusCode.UNIMPLEMENTED,
        ):
            return XAIInvalidArgumentError(details)
    return XAIError(details)


def is_unsupported_tools_search(err: BaseException) -> bool:
    """Return True when xAI rejected the tools-plus-search combination."""
    return isinstance(err, XAIInvalidArgumentError)


def is_chat_model_id(model_id: str) -> bool:
    """Return True if model_id looks like a text/chat LLM (not image/voice/etc.)."""
    mid = (model_id or "").strip().lower()
    if not mid:
        return False
    if any(marker in mid for marker in _NON_CHAT_MODEL_MARKERS):
        return False
    # xAI chat models are grok-* (and occasionally bare aliases)
    if mid.startswith("grok"):
        return True
    # Allow unknown future text models that don't match exclude list
    # but skip obvious non-ids
    if mid.startswith(("ft:", "text-", "code-")):
        return True
    return False


def filter_chat_model_ids(model_ids: list[str]) -> list[str]:
    """Filter + de-dupe + sort chat-capable model ids."""
    seen: set[str] = set()
    out: list[str] = []
    for mid in model_ids:
        if not isinstance(mid, str):
            continue
        name = mid.strip()
        if (
            not name
            or name in seen
            or name in RETIRED_MODELS
            or not is_chat_model_id(name)
        ):
            continue
        seen.add(name)
        out.append(name)

    def _sort_key(name: str) -> tuple:
        # Prefer "latest" / higher major versions first-ish, then alpha
        lower = name.lower()
        latest = 0 if "latest" in lower else 1
        return (latest, lower)

    out.sort(key=_sort_key)
    return out


def _model_ids_from_language_model(item: Any) -> list[str]:
    """Collect name + aliases from a LanguageModel proto/object."""
    ids: list[str] = []
    name = getattr(item, "name", None)
    if name:
        ids.append(str(name))
    aliases = getattr(item, "aliases", None) or []
    for alias in aliases:
        if alias:
            ids.append(str(alias))
    return ids


async def async_list_chat_models(client: Any) -> list[str]:
    """Fetch chat-capable model ids from xAI language-model listing.

    Filters out image/video/voice/embedding models. Falls back to a static
    known list if the API call fails so Options still works offline.
    """
    try:
        page = await client.models.list_language_models()
        raw_ids: list[str] = []
        for item in page or []:
            raw_ids.extend(_model_ids_from_language_model(item))
        models = filter_chat_model_ids(raw_ids)
        if models:
            LOGGER.debug("xAI chat models: %s", models)
            return models
        LOGGER.warning("xAI models list returned no chat models; using fallbacks")
    except Exception as err:  # noqa: BLE001
        LOGGER.warning("Could not list xAI models (%s); using fallbacks", err)

    return filter_chat_model_ids(list(_FALLBACK_CHAT_MODELS))


def build_search_parameters(
    live_search: str, *, return_citations: bool = True
) -> SearchParameters | None:
    """Return xai-sdk SearchParameters for the given live-search mode."""
    mode = (live_search or LIVE_SEARCH_OFF).lower().strip()
    if not mode or mode == LIVE_SEARCH_OFF:
        return None
    sources: list[Any] = []
    if mode in (LIVE_SEARCH_WEB, LIVE_SEARCH_FULL, "web search", "on", "auto"):
        sources.append(web_source())
    if mode in (LIVE_SEARCH_X, LIVE_SEARCH_FULL, "x search", "on", "auto"):
        sources.append(x_source())
    if not sources:
        return None
    return SearchParameters(
        mode="on",
        sources=sources,
        return_citations=return_citations,
    )


def format_citations(citations: Any) -> str:
    """Format citation URLs into a readable footer."""
    if not citations:
        return ""
    urls: list[str] = []
    if isinstance(citations, (list, tuple)):
        for item in citations:
            if isinstance(item, str) and item.startswith("http"):
                urls.append(item)
            elif isinstance(item, dict):
                url = item.get("url") or item.get("uri") or item.get("id")
                if url:
                    urls.append(str(url))
            else:
                url = getattr(item, "url", None) or getattr(item, "uri", None)
                if url:
                    urls.append(str(url))
    elif isinstance(citations, str):
        urls = [citations]
    # Dedupe preserve order
    seen: set[str] = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    if not unique:
        return ""
    lines = "\n".join(f"- {u}" for u in unique[:12])
    return f"\n\nSources:\n{lines}"


def extract_usage(response: Any) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) from a chat result."""
    if isinstance(response, ChatResult):
        return response.prompt_tokens, response.completion_tokens
    usage = getattr(response, "usage", None)
    if not usage:
        return 0, 0
    prompt = getattr(usage, "prompt_tokens", None)
    if prompt is None:
        prompt = getattr(usage, "input_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", None)
    if completion is None:
        completion = getattr(usage, "output_tokens", 0) or 0
    return int(prompt or 0), int(completion or 0)


def _content_parts(content: Any) -> list[Any]:
    """Convert OpenAI-style content to xai-sdk user/system/assistant parts."""
    if content is None:
        return []
    if isinstance(content, str):
        return [content] if content else []
    if isinstance(content, list):
        parts: list[Any] = []
        for item in content:
            if isinstance(item, str):
                if item:
                    parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype in (None, "text"):
                text = item.get("text")
                if text:
                    parts.append(str(text))
                continue
            if itype == "image_url":
                image_url = item.get("image_url")
                detail = "auto"
                url = ""
                if isinstance(image_url, dict):
                    url = str(image_url.get("url") or "")
                    detail = str(image_url.get("detail") or "auto")
                elif image_url:
                    url = str(image_url)
                if url:
                    if detail not in ("auto", "low", "high"):
                        detail = "auto"
                    parts.append(image(url, detail=detail))
        return parts
    return [str(content)]


def convert_messages(messages: list[dict[str, Any]]) -> list[Any]:
    """Convert HA/OpenAI-shaped message dicts to xai-sdk Message protos."""
    out: list[Any] = []
    for msg in messages:
        role = str(msg.get("role") or "").lower()
        content = msg.get("content")
        if role == "developer":
            role = "system"
        if role == "tool":
            if isinstance(content, str):
                text = content
            elif content is None:
                text = ""
            else:
                text = json.dumps(content, default=str)
            out.append(
                tool_result(text, tool_call_id=msg.get("tool_call_id") or None)
            )
            continue
        parts = _content_parts(content)
        if role == "system":
            out.append(system(*parts) if parts else system(""))
            continue
        if role == "assistant":
            msg_pb = assistant(*parts) if parts else assistant("")
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    call_id = str(tc.get("id") or "")
                    name = str(fn.get("name") or "")
                    arguments = fn.get("arguments") or "{}"
                else:
                    call_id = str(getattr(tc, "id", "") or "")
                    fn_obj = getattr(tc, "function", None)
                    name = str(getattr(fn_obj, "name", "") or "")
                    arguments = getattr(fn_obj, "arguments", None) or "{}"
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, default=str)
                msg_pb.tool_calls.append(
                    chat_pb2.ToolCall(
                        id=call_id,
                        function=chat_pb2.FunctionCall(
                            name=name,
                            arguments=arguments,
                        ),
                    )
                )
            out.append(msg_pb)
            continue
        out.append(user(*parts) if parts else user(""))
    return out


def convert_tools(tools: list[dict[str, Any]] | None) -> list[Any] | None:
    """Convert OpenAI-style function tools to xai-sdk Tool protos."""
    if not tools:
        return None
    out: list[Any] = []
    for item in tools:
        if not isinstance(item, dict):
            out.append(item)
            continue
        fn = item.get("function") if "function" in item else item
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "")
        if not name:
            continue
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        out.append(
            tool(
                name=name,
                description=str(fn.get("description") or ""),
                parameters=params,
            )
        )
    return out or None


def convert_response_format(response_format: dict[str, Any] | Any | None) -> Any | None:
    """Convert OpenAI-style json_schema dict to xai-sdk ResponseFormat proto."""
    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        return response_format
    rtype = response_format.get("type")
    if rtype == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema") or {}
        return chat_pb2.ResponseFormat(
            format_type=chat_pb2.FORMAT_TYPE_JSON_SCHEMA,
            schema=json.dumps(schema),
        )
    if rtype == "json_object":
        return chat_pb2.ResponseFormat(
            format_type=chat_pb2.FORMAT_TYPE_JSON_OBJECT
        )
    return None


def _normalize_finish_reason(raw: Any) -> str:
    """Map xai-sdk finish reasons onto the integration's stop/length values."""
    text = str(raw or "")
    upper = text.upper()
    if (
        "MAX_LEN" in upper
        or "MAX_CONTEXT" in upper
        or text.lower() == "length"
    ):
        return "length"
    return "stop"


def _tool_calls_from_response(response: Any) -> list[ChatToolCall]:
    """Extract client-side function tool calls from a sample response."""
    out: list[ChatToolCall] = []
    for tc in getattr(response, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        name = str(getattr(fn, "name", "") or "")
        arguments = getattr(fn, "arguments", None) or "{}"
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, default=str)
        out.append(
            ChatToolCall(
                id=str(getattr(tc, "id", "") or ""),
                name=name,
                arguments=arguments,
            )
        )
    return out


def chat_result_from_response(response: Any) -> ChatResult:
    """Normalize an xai-sdk chat Response into ChatResult."""
    p_tok, c_tok = extract_usage(response)
    content = getattr(response, "content", None)
    citations = getattr(response, "citations", None)
    return ChatResult(
        content=str(content or ""),
        tool_calls=_tool_calls_from_response(response),
        finish_reason=_normalize_finish_reason(
            getattr(response, "finish_reason", None)
        ),
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        citations=list(citations) if citations else [],
    )


# Models that rejected reasoning_effort. Logged once, then omitted.
_REASONING_EFFORT_REJECTED: set[str] = set()


def _model_sends_reasoning_effort(model: str) -> bool:
    """Return True when this id should receive reasoning_effort."""
    if model in _REASONING_EFFORT_REJECTED:
        return False
    lowered = model.lower()
    if "reasoning" in lowered:
        return True
    # grok-4.3 supports effort but the id does not contain "reasoning".
    return lowered == "grok-4.3" or lowered.startswith("grok-4.3-")


def _maybe_reasoning_effort(model: str, reasoning_effort: str | None) -> str | None:
    if not reasoning_effort or reasoning_effort == "none":
        return None
    if not _model_sends_reasoning_effort(model):
        return None
    return reasoning_effort


def _reasoning_effort_rejected(err: BaseException) -> bool:
    """Return True when the API refused the reasoning_effort field."""
    text = str(err).lower().replace("_", " ")
    return "reasoning effort" in text


def _drop_reasoning_effort(model: str) -> None:
    """Stop sending reasoning_effort for this id, and log that once."""
    if model in _REASONING_EFFORT_REJECTED:
        return
    _REASONING_EFFORT_REJECTED.add(model)
    LOGGER.warning(
        "Model '%s' rejected reasoning_effort; omitting it for this id",
        model,
    )


async def _sample_chat(client: Any, kwargs: dict[str, Any]) -> Any:
    """Create a chat and sample it, retrying once if effort is rejected."""
    model = str(kwargs.get("model") or "")
    try:
        chat = client.chat.create(**kwargs)
        return await chat.sample()
    except Exception as err:  # noqa: BLE001
        mapped = map_xai_error(err)
        if not (kwargs.get("reasoning_effort") and _reasoning_effort_rejected(mapped)):
            raise mapped from err
        _drop_reasoning_effort(model)
        retry = {key: value for key, value in kwargs.items() if key != "reasoning_effort"}
        try:
            chat = client.chat.create(**retry)
            return await chat.sample()
        except Exception as retry_err:  # noqa: BLE001
            raise map_xai_error(retry_err) from retry_err


def _chat_create_kwargs(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    reasoning_effort: str | None = None,
    user: str | None = None,
    response_format: dict[str, Any] | None = None,
    search_parameters: SearchParameters | None = None,
) -> dict[str, Any]:
    """Build chat.create kwargs shared by sample() and stream()."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": convert_messages(messages),
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if user:
        kwargs["user"] = user
    converted_tools = convert_tools(tools)
    if converted_tools:
        kwargs["tools"] = converted_tools
        kwargs["tool_choice"] = tool_choice or "auto"
    converted_format = convert_response_format(response_format)
    if converted_format is not None:
        kwargs["response_format"] = converted_format
    effort = _maybe_reasoning_effort(model, reasoning_effort)
    if effort:
        kwargs["reasoning_effort"] = effort
    if search_parameters is not None:
        kwargs["search_parameters"] = search_parameters
    LOGGER.debug(
        "chat.create model=%s tools=%s response_format=%s search=%s",
        model,
        bool(converted_tools),
        bool(converted_format),
        search_parameters is not None,
    )
    return kwargs


async def async_chat_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    reasoning_effort: str | None = None,
    user: str | None = None,
    response_format: dict[str, Any] | None = None,
    search_parameters: SearchParameters | None = None,
) -> ChatResult:
    """Sample a chat completion via xai-sdk."""
    kwargs = _chat_create_kwargs(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=reasoning_effort,
        user=user,
        response_format=response_format,
        search_parameters=search_parameters,
    )
    response = await _sample_chat(client, kwargs)
    return chat_result_from_response(response)


async def async_chat_stream(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    reasoning_effort: str | None = None,
    user: str | None = None,
    response_format: dict[str, Any] | None = None,
    search_parameters: SearchParameters | None = None,
):
    """Yield ``(response, chunk)`` pairs from ``chat.stream()``."""
    kwargs = _chat_create_kwargs(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=reasoning_effort,
        user=user,
        response_format=response_format,
        search_parameters=search_parameters,
    )
    yielded = False

    async def _open(request: dict[str, Any]) -> Any:
        try:
            return client.chat.create(**request)
        except Exception as err:  # noqa: BLE001
            raise map_xai_error(err) from err

    try:
        chat = await _open(kwargs)
    except Exception as err:  # noqa: BLE001
        mapped = map_xai_error(err) if not isinstance(err, XAIError) else err
        if not (kwargs.get("reasoning_effort") and _reasoning_effort_rejected(mapped)):
            raise mapped from err
        _drop_reasoning_effort(str(kwargs.get("model") or ""))
        kwargs = {
            key: value for key, value in kwargs.items() if key != "reasoning_effort"
        }
        chat = await _open(kwargs)

    try:
        async for response, chunk in chat.stream():
            yielded = True
            yield response, chunk
    except Exception as err:  # noqa: BLE001
        mapped = map_xai_error(err)
        if (
            yielded
            or not kwargs.get("reasoning_effort")
            or not _reasoning_effort_rejected(mapped)
        ):
            raise mapped from err
        _drop_reasoning_effort(str(kwargs.get("model") or ""))
        retry = {
            key: value for key, value in kwargs.items() if key != "reasoning_effort"
        }
        try:
            chat = await _open(retry)
            async for response, chunk in chat.stream():
                yield response, chunk
        except Exception as retry_err:  # noqa: BLE001
            raise map_xai_error(retry_err) from retry_err


async def async_responses_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    live_search: str = LIVE_SEARCH_OFF,
    show_citations: bool = True,
    reasoning_effort: str | None = None,
) -> tuple[str, int, int]:
    """Run a live-search chat sample. Returns text, prompt_tok, completion_tok."""
    search_parameters = build_search_parameters(
        live_search, return_citations=show_citations
    )
    request_messages: list[dict[str, Any]] = []
    if system_prompt:
        request_messages.append({"role": "system", "content": system_prompt})
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            content = msg.get("content")
            if isinstance(content, str) and content:
                request_messages.append({"role": "system", "content": content})
            continue
        if role not in ("user", "assistant"):
            continue
        request_messages.append(msg)

    LOGGER.debug(
        "chat.create live_search model=%s live_search=%s",
        model,
        live_search,
    )
    try:
        result = await async_chat_completion(
            client,
            model=model,
            messages=request_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            search_parameters=search_parameters,
        )
    except Exception as err:  # noqa: BLE001
        LOGGER.warning("Live search chat failed (%s); caller may fall back", err)
        raise

    text = result.content
    if show_citations:
        text = text + format_citations(result.citations)
    return text, result.prompt_tokens, result.completion_tokens


async def async_generate_images(client: Any, **kwargs: Any) -> list[Any]:
    """Generate images via xai-sdk image.sample / sample_batch."""
    n = int(kwargs.pop("n", 1) or 1)
    try:
        if n <= 1:
            response = await client.image.sample(**kwargs)
            return [response]
        return list(await client.image.sample_batch(n=n, **kwargs))
    except Exception as err:  # noqa: BLE001
        raise map_xai_error(err) from err


# Allow-list phrases. Matched on word boundaries so "on x" does not hit
# "turn on xbox", and "current" does not hit "currently".
_SEARCH_PHRASES: tuple[str, ...] = (
    "latest",
    "news",
    "headline",
    "headlines",
    "today",
    "tonight",
    "tomorrow",
    "right now",
    "current",
    "final score",
    "box score",
    "score",
    "scores",
    "stock",
    "stocks",
    "price of",
    "weather",
    "forecast",
    "who won",
    "who is winning",
    "standings",
    "trending",
    "on x",
    "on twitter",
    "search the web",
    "look up",
    "google",
    "what happened",
    "who is playing",
    "near me",
    "nearest",
    "closest",
    "open now",
    "around here",
    "in my area",
    "nearby",
)

_SEARCH_QUERY = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(phrase)
        for phrase in sorted(_SEARCH_PHRASES, key=len, reverse=True)
    )
    + r")\b",
    re.IGNORECASE,
)

# Verbs that start a device command even when the sentence also has a
# lookup word such as "today". "open" and "close" are not in this list:
# "open restaurants near me" is a lookup.
_DEVICE_PREFIXES: tuple[str, ...] = (
    "turn ",
    "set ",
    "play ",
    "lock ",
    "unlock ",
    "pause ",
    "stop ",
    "switch ",
    "dim ",
    "brighten ",
    "activate ",
    "deactivate ",
    "toggle ",
)

_REQUEST_PREFIX = re.compile(
    r"^(?:(?:please|can you|could you|would you|will you|hey|ok|okay)\b[, ]*)+",
    re.IGNORECASE,
)

_DEVICE_LIGHT = re.compile(
    r"\b(?:lights? on|lights? off)\b",
    re.IGNORECASE,
)

# Home targets for the ambiguous verbs "open" and "close".
_OPEN_CLOSE_TARGET = re.compile(
    r"\b(?:doors?|garages?|blinds?|shades?|covers?|curtains?|gates?|windows?|"
    r"locks?|lights?|lamps?|fans?|valves?|switches?|tvs?|televisions?|"
    r"speakers?|thermostats?|outlets?|plugs?)\b",
    re.IGNORECASE,
)


def _command_text(text: str) -> str:
    """Drop a leading politeness phrase so the verb can be recognized."""
    stripped = _REQUEST_PREFIX.sub("", (text or "").strip())
    return stripped.strip().lower()


def looks_like_search_query(text: str) -> bool:
    """Allow-list heuristic: user clearly wants fresh/web/X info."""
    return _SEARCH_QUERY.search(text or "") is not None


def looks_like_device_command(text: str) -> bool:
    """Return True for ordinary device commands, not web lookups."""
    t = _command_text(text)
    if not t:
        return False
    if any(t.startswith(prefix) for prefix in _DEVICE_PREFIXES):
        return True
    if t.startswith(("open ", "close ")):
        return _OPEN_CLOSE_TARGET.search(t) is not None
    return _DEVICE_LIGHT.search(t) is not None


def looks_like_non_search_query(text: str) -> bool:
    """Deny-list for pipeline mode: greetings, jokes, timers, recipes, devices.

    In pipeline mode HA already handled device intents before Grok runs, so
    anything that still reaches Grok usually wants fresh data — except these.
    """
    t = (text or "").strip().lower()
    if not t:
        return True

    if t in ("hi", "hey", "hello", "thanks", "thank you", "bye", "goodbye"):
        return True

    greeting_prefixes = (
        "hello",
        "hi ",
        "hi,",
        "hey ",
        "hey,",
        "good morning",
        "good night",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "how are you",
        "what's up",
        "whats up",
    )
    if any(t.startswith(p) for p in greeting_prefixes):
        return True

    if looks_like_device_command(t):
        return True

    non_search_phrases = (
        "tell me a joke",
        "say a joke",
        "make me laugh",
        "set a timer",
        "start a timer",
        "timer for",
        "remind me",
        "set an alarm",
        "wake me",
        "recipe for",
        "how to cook",
        "how do i cook",
        "how do i make",
        "ingredients for",
    )
    return any(p in t for p in non_search_phrases)


def should_use_live_search(
    text: str, *, interaction_mode: str, live_search: str
) -> bool:
    """Decide whether to attach live search to this utterance.

    Device commands never search. ``chat_only`` and ``tools`` search only
    when the allow-list matches. Pipeline mode still searches by default
    except for the deny-list.
    """
    if not live_search or live_search == LIVE_SEARCH_OFF:
        return False
    if looks_like_device_command(text):
        return False
    if interaction_mode == "chat_only":
        return looks_like_search_query(text)
    if looks_like_search_query(text):
        return True
    if interaction_mode == "pipeline":
        return not looks_like_non_search_query(text)
    return False


def looks_like_simple_query(text: str) -> bool:
    """Heuristic for auto-routing to a fast model."""
    t = (text or "").strip()
    if len(t) > 160:
        return False
    simple_starts = (
        "turn ",
        "switch ",
        "set ",
        "open ",
        "close ",
        "lock ",
        "unlock ",
        "play ",
        "pause ",
        "stop ",
        "what's the",
        "what is the",
        "is the ",
        "are the ",
        "how warm",
        "how cold",
        "temperature",
        "lights",
        "good morning",
        "good night",
        "hello",
        "hi ",
        "thanks",
        "thank you",
    )
    lower = t.lower()
    if any(lower.startswith(s) for s in simple_starts):
        return True
    # Short yes/no or single command
    return len(t.split()) <= 8 and "?" not in t[20:]
