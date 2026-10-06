"""Shared LLM entity base for Grok Conversation and AI Task."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
from pathlib import Path
from typing import Any, Literal

from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity
from homeassistant.util import slugify
from homeassistant.util.json import json_loads

try:
    import probatio
except ImportError:  # pragma: no cover - HA < 2026.8 may lack probatio
    probatio = None  # type: ignore[assignment]

from .api_helpers import (
    ChatResult,
    ChatToolCall,
    CombinedSearchRejected,
    XAIAuthError,
    XAIConnectionError,
    XAIError,
    XAIInvalidArgumentError,
    XAIRateLimitError,
    async_chat_completion,
    async_chat_stream,
    chat_result_from_response,
    format_citations,
    is_unsupported_tools_search,
)
from .const import (
    CONF_BUDGET_WARN_USD,
    CONF_CHAT_MODEL,
    CONF_FALLBACK_MODEL,
    CONF_MAX_TOKENS,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_VISION_MODEL,
    DOMAIN,
    LOGGER,
    RECOMMENDED_AI_TASK_MAX_TOKENS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_FALLBACK_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_VISION_MODEL,
    RETIRED_VISION_MODELS,
    remap_retired_chat_model,
)
from .exceptions import TokenLengthExceededError
from .usage import UsageTracker

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


def _strip_json_from_response(response: str) -> str:
    """Strip JSON objects from the end of LLM responses."""
    if not response:
        return response
    last_brace_index = response.rfind("{")
    if last_brace_index == -1:
        return response
    potential_json = response[last_brace_index:]
    try:
        json.loads(potential_json)
        return response[:last_brace_index].strip()
    except json.JSONDecodeError:
        return response


def sanitize_tool_schema(schema: Any) -> dict[str, Any]:
    """Normalize tool JSON schema for xAI function calling.

    xAI rejects roots that are anyOf/oneOf unions (e.g. HA HassStartTimer).
    Merge properties from every object branch so alternate arg shapes
    (name vs hours/minutes/seconds) are all still available to the model.
    """
    if not isinstance(schema, dict):
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }

    def _merge_union_branches(branches: list[Any]) -> dict[str, Any] | None:
        """Collapse anyOf/oneOf/allOf into one object schema."""
        props: dict[str, Any] = {}
        objectish = 0
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            branch = _clean(branch)
            if not isinstance(branch, dict):
                continue
            if branch.get("type") == "object" or "properties" in branch:
                objectish += 1
                nested = branch.get("properties")
                if isinstance(nested, dict):
                    props.update(nested)
        if objectish:
            return {
                "type": "object",
                "properties": props,
                "additionalProperties": True,
            }
        for branch in branches:
            if isinstance(branch, dict):
                return _clean(branch)
        return None

    def _clean(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        node = dict(node)

        for union_key in ("anyOf", "oneOf", "allOf"):
            if union_key not in node or not isinstance(node[union_key], list):
                continue
            branches = node[union_key]
            merged = _merge_union_branches(branches)
            rest = {
                k: v
                for k, v in node.items()
                if k not in (union_key, "required")
            }
            if isinstance(merged, dict):
                node = {**rest, **merged}
            else:
                node = rest
            break

        t = node.get("type")
        if isinstance(t, list):
            if "object" in t:
                node["type"] = "object"
            elif "array" in t:
                node["type"] = "array"
            elif t:
                node["type"] = t[0]

        if "properties" in node and isinstance(node["properties"], dict):
            node["properties"] = {
                key: _clean(value) for key, value in node["properties"].items()
            }
        if "items" in node:
            node["items"] = _clean(node["items"])
        if "additionalProperties" in node and isinstance(
            node["additionalProperties"], dict
        ):
            node["additionalProperties"] = _clean(node["additionalProperties"])

        node.pop("not", None)
        return node

    cleaned = _clean(schema)

    if not isinstance(cleaned, dict):
        cleaned = {}
    if cleaned.get("type") != "object" and "properties" not in cleaned:
        cleaned = {
            "type": "object",
            "properties": {"value": cleaned} if cleaned else {},
            "additionalProperties": True,
        }
    cleaned.setdefault("type", "object")
    if cleaned["type"] != "object":
        cleaned = {
            "type": "object",
            "properties": {"value": cleaned},
            "additionalProperties": True,
        }
    cleaned.setdefault("properties", {})
    if not isinstance(cleaned["properties"], dict):
        cleaned["properties"] = {}

    for bad in ("oneOf", "anyOf", "allOf", "not", "enum"):
        cleaned.pop(bad, None)
    if "required" in cleaned and not cleaned.get("properties"):
        cleaned.pop("required", None)

    return cleaned


# Backwards-compatible alias used by conversation imports during refactor
_sanitize_tool_schema = sanitize_tool_schema


def format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format tool specification for xAI / OpenAI-compatible function calling."""
    serializer = custom_serializer or llm.selector_serializer
    try:
        if probatio is not None:
            raw_schema = probatio.to_openapi(
                tool.parameters,
                custom_serializer=serializer,
                openapi_version="3.1.0",
            )
        else:
            raw_schema = convert(tool.parameters, custom_serializer=serializer)
    except Exception as err:  # noqa: BLE001
        LOGGER.warning(
            "Failed to convert schema for tool %s (%s); using empty object schema",
            tool.name,
            err,
        )
        raw_schema = {"type": "object", "properties": {}}

    schema = sanitize_tool_schema(raw_schema)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema,
        },
    }


def _adjust_strict_schema(schema: dict[str, Any]) -> None:
    """Make a JSON schema valid for strict structured outputs.

    xAI / OpenAI strict mode requires every property to appear in ``required``.
    Optional fields are made nullable (``[type, "null"]``) and then required,
    matching core ``openai_conversation`` behavior.
    """
    node_type = schema.get("type")
    if node_type == "object" or "properties" in schema:
        schema.setdefault("additionalProperties", False)
        props = schema.get("properties")
        if not isinstance(props, dict):
            return
        required = list(schema.get("required") or [])
        for prop, prop_info in props.items():
            if isinstance(prop_info, dict):
                _adjust_strict_schema(prop_info)
            if prop not in required:
                if isinstance(prop_info, dict):
                    ptype = prop_info.get("type")
                    if isinstance(ptype, list):
                        if "null" not in ptype:
                            prop_info["type"] = [*ptype, "null"]
                    elif ptype is not None:
                        prop_info["type"] = [ptype, "null"]
                    else:
                        prop_info["type"] = ["string", "null"]
                required.append(prop)
        schema["required"] = required
    elif node_type == "array" and isinstance(schema.get("items"), dict):
        _adjust_strict_schema(schema["items"])


def format_structured_output(
    structure: Any, llm_api: llm.APIInstance | None
) -> dict[str, Any]:
    """Convert a structure schema for xAI json_schema response_format.

    HA 2026.7 used voluptuous (+ voluptuous_openapi). HA 2026.8+ uses
    probatio.Schema. Support both so we stay compatible with the declared
    min version and current core.
    """
    serializer = (
        llm_api.custom_serializer if llm_api else llm.selector_serializer
    )
    raw_schema: Any
    if probatio is not None and isinstance(structure, probatio.Schema):
        raw_schema = probatio.to_openapi(
            structure,
            custom_serializer=serializer,
            openapi_version="3.1.0",
        )
    else:
        raw_schema = convert(structure, custom_serializer=serializer)

    if not isinstance(raw_schema, dict):
        raise HomeAssistantError(
            f"Could not convert structure schema to JSON Schema: {raw_schema!r}"
        )

    schema = sanitize_tool_schema(raw_schema)
    # Structured outputs default additionalProperties to false on xAI
    schema["additionalProperties"] = False

    def _force_no_additional(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
        props = node.get("properties")
        if isinstance(props, dict):
            for child in props.values():
                _force_no_additional(child)
        if "items" in node:
            _force_no_additional(node["items"])

    _force_no_additional(schema)
    _adjust_strict_schema(schema)
    return schema


def _tool_result_payload(tool_result: Any) -> str:
    """Serialize tool results for the model."""
    if hasattr(tool_result, "data"):
        payload = {
            "data": tool_result.data,
            "error": getattr(tool_result, "error", False),
        }
        try:
            return json.dumps(payload, default=str)
        except TypeError:
            return json.dumps({"result": str(tool_result)})
    try:
        return json.dumps(tool_result, default=str)
    except TypeError:
        return json.dumps({"result": str(tool_result)})


def convert_content_to_param(
    content: conversation.Content,
) -> list[dict[str, Any]]:
    """Convert any native chat message for this agent to the native format."""
    messages: list[dict[str, Any]] = []

    if isinstance(content, conversation.ToolResultContent):
        result = getattr(content, "result", None)
        if result is None:
            result = getattr(content, "tool_result", None)
        tool_message = {
            "role": "tool",
            "content": _tool_result_payload(result),
            "tool_call_id": content.tool_call_id,
        }
        messages.append(tool_message)  # type: ignore[arg-type]
        return messages

    if isinstance(content, conversation.AssistantContent) and content.tool_calls:
        tool_calls_list = []
        for tool_call in content.tool_calls:
            if hasattr(tool_call, "function"):
                tool_calls_list.append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )
            elif hasattr(tool_call, "tool_name"):
                args = getattr(tool_call, "tool_args", {})
                tool_calls_list.append(
                    {
                        "id": tool_call.id
                        if hasattr(tool_call, "id")
                        else str(hash(tool_call)),
                        "type": "function",
                        "function": {
                            "name": tool_call.tool_name,
                            "arguments": json.dumps(args)
                            if not isinstance(args, str)
                            else args,
                        },
                    }
                )
            elif isinstance(tool_call, dict):
                tool_calls_list.append(
                    {
                        "id": tool_call.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tool_call.get(
                                "tool_name", tool_call.get("name", "")
                            ),
                            "arguments": json.dumps(
                                tool_call.get(
                                    "tool_args", tool_call.get("arguments", {})
                                )
                            ),
                        },
                    }
                )
            else:
                tool_calls_list.append(
                    {
                        "id": getattr(tool_call, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(
                                tool_call,
                                "tool_name",
                                getattr(tool_call, "name", ""),
                            ),
                            "arguments": json.dumps(
                                getattr(
                                    tool_call,
                                    "tool_args",
                                    getattr(tool_call, "arguments", {}),
                                )
                            ),
                        },
                    }
                )

        messages.append(
            {
                "role": "assistant",
                "content": content.content or "",
                "tool_calls": tool_calls_list,
            }
        )  # type: ignore[arg-type]
        return messages

    if isinstance(content, conversation.UserContent):
        role: Literal["user", "assistant", "system", "developer"] = content.role
        text = content.content or ""
        attachments = getattr(content, "attachments", None) or ()
        if attachments:
            # Multimodal parts are attached later in _async_handle_chat_log
            # once files are read off the event loop. Here keep text only;
            # attachment expansion happens when building the request.
            messages.append({"role": role, "content": text})
        elif text:
            messages.append({"role": role, "content": text})
        return messages

    if hasattr(content, "content") and content.content:
        role = content.role
        if role == "developer":
            role = "system"
        messages.append({"role": role, "content": content.content})
    return messages


def model_supports_vision(model: str) -> bool:
    """Return True if the model accepts image input.

    Current Grok 4.x chat models accept images. Retired ``grok-2-vision-*``
    ids are not usable. Image-generation / voice / embedding models do not
    accept chat image input.
    """
    mid = (model or "").strip().lower()
    if not mid:
        return False
    if any(
        marker in mid
        for marker in ("imagine", "tts", "stt", "voice", "embedding", "embed")
    ):
        return False
    # Retired dedicated vision models (xAI removed grok-2-vision-*)
    if mid.startswith("grok-2-vision") or mid.startswith("grok-2-image"):
        return False
    if "vision" in mid:
        return True
    # Grok 4+ chat models accept image input per xAI docs (2026)
    if mid.startswith("grok-4") or mid.startswith("grok-build"):
        return True
    if mid in {"grok-3", "grok-3-mini", "grok-3-mini-fast", "grok-3-fast"}:
        # Grok 3 family: text-only unless vision variant
        return False
    # Default chat model aliases (latest) track current multimodal Grok
    if mid in {"grok-4.3-latest", "grok-4-latest", "grok-4.5-latest", "grok-4.6-latest"}:
        return True
    return mid.startswith("grok-4")


def resolve_vision_model(options: dict[str, Any], chat_model: str) -> str:
    """Pick a vision-capable model for attachment requests.

    Prefer the configured chat model when it accepts images so AI Task
    subentries (and Assist) keep the user's chosen model/cost. Only fall
    back to ``vision_model`` / ``RECOMMENDED_VISION_MODEL`` when the chat
    model cannot take images. Retired ``grok-2-vision-*`` values are skipped.
    """
    chat = remap_retired_chat_model(chat_model, RECOMMENDED_CHAT_MODEL)
    if model_supports_vision(chat):
        return chat
    configured = options.get(CONF_VISION_MODEL)
    if configured and configured not in RETIRED_VISION_MODELS:
        configured = remap_retired_chat_model(str(configured), RECOMMENDED_VISION_MODEL)
        if model_supports_vision(str(configured)):
            return str(configured)
    return RECOMMENDED_VISION_MODEL


class GrokBaseLLMEntity(Entity):
    """Shared Grok LLM helpers for conversation and AI Task entities."""

    _attr_has_entity_name = True
    _attr_name: str | None = None

    def __init__(
        self,
        entry: ConfigEntry,
        subentry: ConfigSubentry | None = None,
    ) -> None:
        """Initialize the base entity."""
        self.entry = entry
        self.subentry = subentry
        if subentry is not None:
            # Use the device name only (has_entity_name). Setting both the
            # entity name and device name to the subentry title produced
            # ai_task.grok_ai_task_grok_ai_task / "Grok AI Task Grok AI Task".
            self._attr_name = None
            self._attr_unique_id = subentry.subentry_id
            self._attr_device_info = dr.DeviceInfo(
                identifiers={(DOMAIN, subentry.subentry_id)},
                name=subentry.title,
                manufacturer="xAI",
                model="Grok",
                entry_type=dr.DeviceEntryType.SERVICE,
            )

    def _llm_options(self) -> dict[str, Any]:
        """Return options for this entity (subentry data or entry options)."""
        if self.subentry is not None:
            return dict(self.subentry.data)
        return dict(self.entry.options)

    def _usage_tracker(self) -> UsageTracker | None:
        data = self.hass.data.get(DOMAIN, {}).get(self.entry.entry_id)
        if not data:
            return None
        return data.get("usage")

    async def _record_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        service: str,
    ) -> None:
        tracker = self._usage_tracker()
        if tracker:
            await tracker.async_record(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                service=service,
                budget_warn_usd=float(
                    self.entry.options.get(CONF_BUDGET_WARN_USD, 0) or 0
                ),
            )

    async def _async_prepare_attachment_parts(
        self,
        attachments: list[Any],
    ) -> list[dict[str, Any]]:
        """Read image attachments off the event loop as base64 image_url parts."""
        from . import encode_file

        files: list[tuple[Path, str | None]] = []
        for attachment in attachments:
            mime = getattr(attachment, "mime_type", None) or ""
            path = getattr(attachment, "path", None)
            if not mime.startswith("image/"):
                raise HomeAssistantError(
                    "Only images are supported by the xAI API"
                )
            if path is None:
                raise HomeAssistantError("Attachment path is missing")
            files.append((Path(path), mime))

        def _read() -> list[dict[str, Any]]:
            parts: list[dict[str, Any]] = []
            for path, mime_type in files:
                if not path.exists():
                    raise HomeAssistantError(f"`{path}` does not exist")
                encoded_mime, b64 = encode_file(str(path))
                use_mime = mime_type or encoded_mime
                if "image/" not in use_mime:
                    raise HomeAssistantError(
                        "Only images are supported by the xAI API"
                    )
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{use_mime};base64,{b64}",
                            "detail": "auto",
                        },
                    }
                )
            return parts

        return await self.hass.async_add_executor_job(_read)

    def _messages_from_chat_log(
        self,
        chat_log: conversation.ChatLog,
        *,
        attachment_parts: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build OpenAI-compatible messages from the chat log."""
        messages: list[dict[str, Any]] = []
        last_user_idx: int | None = None
        for content in chat_log.content:
            converted = convert_content_to_param(content)
            if not converted:
                continue
            start = len(messages)
            messages.extend(converted)  # type: ignore[arg-type]
            if isinstance(content, conversation.UserContent):
                last_user_idx = start

        if attachment_parts and last_user_idx is not None:
            msg = messages[last_user_idx]
            text = msg.get("content") or ""
            if isinstance(text, str):
                content_parts: list[dict[str, Any]] = [
                    {"type": "text", "text": text}
                ]
            elif isinstance(text, list):
                content_parts = list(text)
            else:
                content_parts = [{"type": "text", "text": str(text)}]
            content_parts.extend(attachment_parts)
            messages[last_user_idx] = {
                **msg,
                "content": content_parts,
            }
        return messages

    def _tool_calls_to_ha(
        self, tool_calls: list[ChatToolCall] | Any
    ) -> tuple[list[llm.ToolInput], dict[str, dict[str, Any]]]:
        """Convert SDK tool calls to HA ToolInputs.

        Returns ``(tool_inputs, parse_errors)`` where ``parse_errors`` maps
        tool_call id → error payload for malformed argument JSON. Those
        calls must not execute the tool; feed the error back to the model.
        """
        ha_calls: list[llm.ToolInput] = []
        parse_errors: dict[str, dict[str, Any]] = {}
        for tc in tool_calls:
            call_id = getattr(tc, "id", "")
            name = getattr(tc, "name", None)
            raw_args = getattr(tc, "arguments", None)
            if name is None:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", "") if fn is not None else ""
                raw_args = getattr(fn, "arguments", None) if fn is not None else None
            raw_args = raw_args or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as err:
                parse_errors[call_id] = {
                    "error": f"Invalid tool arguments JSON: {err}",
                    "raw_arguments": raw_args,
                }
                args = {}
            if not isinstance(args, dict):
                parse_errors[call_id] = {
                    "error": "Invalid tool arguments JSON: expected object",
                    "raw_arguments": raw_args,
                }
                args = {}
            ha_calls.append(
                llm.ToolInput(
                    tool_name=str(name or ""),
                    tool_args=args,
                    id=call_id,
                )
            )
        return ha_calls, parse_errors

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        *,
        structure: Any | None = None,
        structure_name: str | None = None,
        max_iterations: int = MAX_TOOL_ITERATIONS,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        agent_id: str | None = None,
        service: str = "conversation",
        fallback_model: str | None = None,
        search_parameters: Any | None = None,
        stream_final: bool = False,
        stream_timeout: float | None = None,
        append_citations: bool = False,
    ) -> None:
        """Run chat completions + tool loop. Does not depend on ConversationInput."""
        opts = options if options is not None else self._llm_options()
        active_model = remap_retired_chat_model(
            model or opts.get(CONF_CHAT_MODEL),
            RECOMMENDED_CHAT_MODEL,
        )
        agent = agent_id or self.entity_id
        client = self.entry.runtime_data

        attachment_parts: list[dict[str, Any]] | None = None
        last_content = chat_log.content[-1] if chat_log.content else None
        if (
            isinstance(last_content, conversation.UserContent)
            and last_content.attachments
        ):
            attachment_parts = await self._async_prepare_attachment_parts(
                list(last_content.attachments)
            )
            active_model = resolve_vision_model(opts, active_model)

        response_format: dict[str, Any] | None = None
        if structure is not None:
            name = slugify(structure_name or "response")
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": format_structured_output(structure, chat_log.llm_api),
                    "strict": True,
                },
            }

        models_to_try = [active_model]
        raw_fb = (
            fallback_model
            if fallback_model is not None
            else opts.get(CONF_FALLBACK_MODEL, RECOMMENDED_FALLBACK_MODEL)
        )
        fb = (
            remap_retired_chat_model(str(raw_fb), RECOMMENDED_FALLBACK_MODEL)
            if raw_fb
            else ""
        )
        # AI Task subentries may omit fallback; conversation keeps entry fallback
        if fb and fb != active_model and service == "conversation":
            models_to_try.append(str(fb))

        last_error: Exception | None = None
        for try_model in models_to_try:
            # Fresh message copy + chat_log checkpoint so a failed primary
            # attempt does not leak partial tool turns into the fallback.
            attempt_messages = (
                [dict(m) for m in messages] if messages is not None else None
            )
            content_checkpoint = len(chat_log.content)
            try:
                await self._async_tool_loop(
                    chat_log=chat_log,
                    messages=attempt_messages,
                    model=try_model,
                    options=opts,
                    client=client,
                    agent_id=agent,
                    service=service,
                    response_format=response_format,
                    attachment_parts=attachment_parts,
                    max_iterations=max_iterations,
                    skip_json_strip=structure is not None,
                    search_parameters=search_parameters,
                    stream_final=stream_final,
                    stream_timeout=stream_timeout,
                    append_citations=append_citations,
                )
                return
            except CombinedSearchRejected:
                raise
            except XAIRateLimitError as err:
                last_error = err
                LOGGER.error("Rate limited by xAI on %s: %s", try_model, err)
                del chat_log.content[content_checkpoint:]
                break
            except XAIError as err:
                last_error = err
                LOGGER.warning(
                    "Model %s failed (%s); trying fallback if available",
                    try_model,
                    err,
                )
                del chat_log.content[content_checkpoint:]
                continue
            except TokenLengthExceededError:
                raise
            except HomeAssistantError:
                raise
            except Exception as err:  # noqa: BLE001
                last_error = err
                LOGGER.warning("Unexpected error on %s: %s", try_model, err)
                del chat_log.content[content_checkpoint:]
                continue

        if isinstance(last_error, XAIRateLimitError):
            raise HomeAssistantError(
                "Rate limited or insufficient funds"
            ) from last_error
        raise HomeAssistantError(f"Error talking to xAI: {last_error}") from last_error

    async def _async_stream_completion(
        self,
        client: Any,
        *,
        stream_timeout: float | None,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        top_p: float | None,
        temperature: float | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
        reasoning_effort: str | None,
        user: str | None,
        response_format: dict[str, Any] | None,
        search_parameters: Any | None,
    ) -> tuple[ChatResult, list[str], bool]:
        """Buffer ``chat.stream()`` until it ends or the deadline passes."""
        deltas: list[str] = []
        response: Any = None
        timed_out = False
        stream = async_chat_stream(
            client,
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
        try:
            loop = asyncio.get_running_loop()
            deadline = (
                None if stream_timeout is None else loop.time() + stream_timeout
            )
            while True:
                if deadline is None:
                    try:
                        response, chunk = await anext(stream)
                    except StopAsyncIteration:
                        break
                else:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        timed_out = True
                        break
                    try:
                        response, chunk = await asyncio.wait_for(
                            anext(stream), remaining
                        )
                    except TimeoutError:
                        timed_out = True
                        break
                    except StopAsyncIteration:
                        break
                piece = getattr(chunk, "content", None) or ""
                if piece:
                    deltas.append(str(piece))
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Closing chat stream failed", exc_info=True)

        if response is None:
            result = ChatResult(content="".join(deltas))
        else:
            result = chat_result_from_response(response)
            if not result.content and deltas:
                result = ChatResult(
                    content="".join(deltas),
                    tool_calls=result.tool_calls,
                    finish_reason=result.finish_reason,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    citations=list(result.citations),
                )
        return result, deltas, timed_out

    async def _async_add_streamed_text(
        self,
        chat_log: conversation.ChatLog,
        *,
        agent_id: str,
        raw_text: str,
        deltas: list[str],
        citation_text: str,
        skip_json_strip: bool,
    ) -> str:
        """Replay buffered text deltas. Tool-call turns never reach this."""
        text = raw_text if skip_json_strip else _strip_json_from_response(raw_text)
        if deltas and "".join(deltas) == raw_text and text == raw_text:
            pieces = [piece for piece in deltas if piece]
        else:
            pieces = [text] if text else []
        if citation_text:
            pieces.append(citation_text)
        full = "".join(pieces)
        if not pieces:
            if skip_json_strip:
                async for _ in chat_log.async_add_assistant_content(
                    conversation.AssistantContent(agent_id=agent_id, content=full)
                ):
                    pass
            return full

        async def _deltas():
            started = False
            for piece in pieces:
                if not piece:
                    continue
                if not started:
                    yield {"role": "assistant", "content": piece}
                    started = True
                else:
                    yield {"content": piece}

        async for _ in chat_log.async_add_delta_content_stream(agent_id, _deltas()):
            pass
        return full

    async def _async_tool_loop(
        self,
        *,
        chat_log: conversation.ChatLog,
        messages: list[dict[str, Any]] | None,
        model: str,
        options: dict[str, Any],
        client: Any,
        agent_id: str,
        service: str,
        response_format: dict[str, Any] | None,
        attachment_parts: list[dict[str, Any]] | None,
        max_iterations: int,
        skip_json_strip: bool,
        search_parameters: Any | None = None,
        stream_final: bool = False,
        stream_timeout: float | None = None,
        append_citations: bool = False,
    ) -> None:
        """Run chat completion tool iterations for one model."""
        working_messages = messages
        saved_citations: list[Any] = []
        default_max_tokens = (
            RECOMMENDED_AI_TASK_MAX_TOKENS
            if service == "ai_task"
            else RECOMMENDED_MAX_TOKENS
        )

        for _iteration in range(max_iterations):
            if working_messages is None:
                request_messages = self._messages_from_chat_log(
                    chat_log, attachment_parts=attachment_parts
                )
            else:
                request_messages = list(working_messages)
                if attachment_parts and _iteration == 0:
                    # Attach images to the last user message once
                    for i in range(len(request_messages) - 1, -1, -1):
                        if request_messages[i].get("role") == "user":
                            content = request_messages[i].get("content")
                            if isinstance(content, str):
                                request_messages[i] = {
                                    **request_messages[i],
                                    "content": [
                                        {"type": "text", "text": content},
                                        *attachment_parts,
                                    ],
                                }
                            break
                    attachment_parts = None  # only once

            tools: list[dict[str, Any]] | None = None
            if chat_log.llm_api:
                tools = []
                for tool in chat_log.llm_api.tools:
                    try:
                        tools.append(
                            format_tool(tool, chat_log.llm_api.custom_serializer)
                        )
                    except Exception as err:  # noqa: BLE001
                        LOGGER.warning(
                            "Skipping tool %s due to schema error: %s",
                            getattr(tool, "name", "?"),
                            err,
                        )
                if not tools:
                    tools = None

            # xAI Grok 4 supports response_format together with function tools.
            # After tool results, drop tools so the model emits the structured
            # final answer rather than another tool call.
            post_tool = bool(
                chat_log.content
                and getattr(chat_log.content[-1], "role", None) == "tool_result"
            )
            if response_format and post_tool:
                use_tools = None
            else:
                use_tools = tools
            use_response_format = response_format

            # Search stays on later rounds. The provider's results are not in
            # the tool messages, so a follow-up without search_parameters would
            # answer from the tool payload alone.
            call_search = search_parameters
            completion_kwargs = {
                "model": model,
                "messages": request_messages,
                "max_tokens": options.get(CONF_MAX_TOKENS, default_max_tokens),
                "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                "tools": use_tools,
                "tool_choice": "auto" if use_tools else None,
                "reasoning_effort": options.get(
                    CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
                ),
                "user": chat_log.conversation_id,
                "response_format": use_response_format,
                "search_parameters": call_search,
            }
            try:
                if stream_final:
                    result, deltas, timed_out = await self._async_stream_completion(
                        client,
                        stream_timeout=stream_timeout,
                        **completion_kwargs,
                    )
                else:
                    result = await async_chat_completion(client, **completion_kwargs)
                    deltas = []
                    timed_out = False
            except XAIRateLimitError:
                raise
            except (XAIAuthError, XAIConnectionError):
                raise
            except XAIInvalidArgumentError as err:
                if (
                    _iteration == 0
                    and call_search is not None
                    and use_tools
                    and is_unsupported_tools_search(err)
                ):
                    raise CombinedSearchRejected(str(err)) from err
                raise
            except XAIError:
                raise

            if timed_out:
                LOGGER.debug(
                    "Stopped chat stream after %.0f seconds",
                    stream_timeout or 0,
                )
                result = ChatResult(
                    content=result.content,
                    finish_reason="stop",
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                )
                deltas = [result.content] if result.content else []

            p_tok, c_tok = result.prompt_tokens, result.completion_tokens

            if result.tool_calls:
                if result.citations:
                    saved_citations = list(result.citations)
                ha_tool_calls, parse_errors = self._tool_calls_to_ha(
                    result.tool_calls
                )
                assistant_content = conversation.AssistantContent(
                    agent_id=agent_id,
                    content=result.content or "",
                    tool_calls=ha_tool_calls,
                )
                tool_messages: list[dict[str, Any]] = [
                    {
                        "role": "assistant",
                        "content": result.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                },
                            }
                            for tc in result.tool_calls
                        ],
                    }
                ]

                # Pre-seed error results for malformed arg JSON so the tool
                # is never executed with empty/guessed args (restores v1.8
                # behavior).
                tool_call_tasks: dict[str, Any] | None = None
                if parse_errors:

                    async def _const_result(payload: dict[str, Any]) -> dict[str, Any]:
                        return payload

                    tool_call_tasks = {
                        call_id: self.hass.async_create_task(
                            _const_result(payload),
                            name=f"llm_tool_parse_error_{call_id}",
                        )
                        for call_id, payload in parse_errors.items()
                    }

                async for tool_response in chat_log.async_add_assistant_content(
                    assistant_content,
                    tool_call_tasks=tool_call_tasks,
                ):
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_response.tool_call_id,
                            "content": _tool_result_payload(
                                getattr(tool_response, "tool_result", None)
                                if getattr(tool_response, "tool_result", None)
                                is not None
                                else getattr(tool_response, "result", None)
                            ),
                        }
                    )

                if working_messages is not None:
                    working_messages.extend(tool_messages)
                await self._record_usage(model, p_tok, c_tok, service=service)
                continue

            raw_text = result.content or ""
            if stream_final:
                citation_text = ""
                if append_citations:
                    citation_text = format_citations(
                        saved_citations or result.citations
                    )
                full_response = await self._async_add_streamed_text(
                    chat_log,
                    agent_id=agent_id,
                    raw_text=raw_text,
                    deltas=deltas,
                    citation_text=citation_text,
                    skip_json_strip=skip_json_strip,
                )
            else:
                full_response = (
                    raw_text
                    if skip_json_strip
                    else _strip_json_from_response(raw_text)
                )
                if full_response or skip_json_strip:
                    async for _ in chat_log.async_add_assistant_content(
                        conversation.AssistantContent(
                            agent_id=agent_id, content=full_response
                        )
                    ):
                        pass
            if full_response or skip_json_strip:
                if working_messages is not None:
                    working_messages.append(
                        {"role": "assistant", "content": full_response}
                    )

            chat_log.async_trace(
                {
                    "stats": {
                        "input_tokens": result.prompt_tokens,
                        "output_tokens": result.completion_tokens,
                    }
                }
            )
            await self._record_usage(model, p_tok, c_tok, service=service)

            if result.finish_reason == "length":
                raise TokenLengthExceededError(
                    options.get(CONF_MAX_TOKENS, default_max_tokens)
                )
            break
