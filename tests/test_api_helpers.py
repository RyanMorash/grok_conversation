"""Unit tests for xai-sdk adapter helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from xai_sdk.proto import chat_pb2
from xai_sdk.search import SearchParameters

from custom_components.grok_conversation.api_helpers import (
    XAIAuthError,
    XAIConnectionError,
    XAIError,
    XAIInvalidArgumentError,
    XAIRateLimitError,
    _REASONING_EFFORT_REJECTED,
    _chat_create_kwargs,
    _normalize_finish_reason,
    async_chat_completion,
    async_chat_stream,
    build_search_parameters,
    convert_messages,
    convert_response_format,
    convert_tools,
    format_citations,
    is_unsupported_tools_search,
    map_xai_error,
)


def test_convert_messages_roles_and_images() -> None:
    """HA/OpenAI-shaped dicts become xai-sdk Message protos."""
    messages = convert_messages(
        [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/a.png",
                            "detail": "auto",
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "test_light",
                            "arguments": '{"action":"on"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"ok": true}',
            },
        ]
    )
    roles = [
        chat_pb2.MessageRole.Name(m.role).removeprefix("ROLE_").lower()
        for m in messages
    ]
    assert roles == ["system", "user", "assistant", "tool"]
    assert any(part.HasField("image_url") for part in messages[1].content)
    assert messages[2].tool_calls[0].function.name == "test_light"
    assert messages[3].tool_call_id == "call_1"


def test_convert_tools_and_json_schema() -> None:
    """Function tools and json_schema response_format convert to protos."""
    tools = convert_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
    )
    assert tools is not None
    assert tools[0].function.name == "get_weather"

    rf = convert_response_format(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "driveway_check",
                "strict": True,
                "schema": {"type": "object", "properties": {"cars": {"type": "integer"}}},
            },
        }
    )
    assert rf.format_type == chat_pb2.FORMAT_TYPE_JSON_SCHEMA
    assert "cars" in rf.schema


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("off", None),
        ("web", ["web"]),
        ("x", ["x"]),
        ("full", ["web", "x"]),
    ],
)
def test_build_search_parameters_sources(mode: str, expected: list[str] | None) -> None:
    """Live-search modes select web / X sources."""
    params = build_search_parameters(mode)
    if expected is None:
        assert params is None
        return
    assert isinstance(params, SearchParameters)
    kinds = []
    for source in params.sources or []:
        if source.HasField("web"):
            kinds.append("web")
        if source.HasField("x"):
            kinds.append("x")
    assert kinds == expected


def test_format_citations_dedupes() -> None:
    """Citation URLs are formatted as a Sources footer."""
    text = format_citations(
        ["https://a.example/", "https://a.example/", "https://b.example/"]
    )
    assert text.startswith("\n\nSources:")
    assert text.count("https://a.example/") == 1


def test_map_xai_error_passthrough() -> None:
    """Existing XAIError subclasses are returned unchanged."""
    err = XAIAuthError("nope")
    assert map_xai_error(err) is err
    assert isinstance(map_xai_error(RuntimeError("boom")), XAIError)


def test_map_grpc_status_codes() -> None:
    """gRPC status codes map to auth / connection / rate-limit errors."""
    grpc = pytest.importorskip("grpc")

    class _Rpc(grpc.RpcError):
        def __init__(self, code) -> None:
            super().__init__()
            self._code = code

        def code(self):
            return self._code

        def details(self):
            return "detail"

    assert isinstance(map_xai_error(_Rpc(grpc.StatusCode.UNAUTHENTICATED)), XAIAuthError)
    assert isinstance(
        map_xai_error(_Rpc(grpc.StatusCode.UNAVAILABLE)), XAIConnectionError
    )
    assert isinstance(
        map_xai_error(_Rpc(grpc.StatusCode.RESOURCE_EXHAUSTED)), XAIRateLimitError
    )
    invalid = map_xai_error(_Rpc(grpc.StatusCode.INVALID_ARGUMENT))
    assert isinstance(invalid, XAIInvalidArgumentError)
    assert is_unsupported_tools_search(invalid)
    assert not is_unsupported_tools_search(XAIAuthError("revoked"))
    assert not is_unsupported_tools_search(XAIConnectionError("down"))
    assert not is_unsupported_tools_search(XAIError("primary down"))


def test_normalize_finish_reason_max_context() -> None:
    """MAX_LEN and MAX_CONTEXT both map to the token-length finish reason."""
    assert _normalize_finish_reason("REASON_MAX_LEN") == "length"
    assert _normalize_finish_reason("REASON_MAX_CONTEXT") == "length"
    assert _normalize_finish_reason("REASON_STOP") == "stop"
    assert _normalize_finish_reason("REASON_TOOL_CALLS") == "stop"


def _kwargs_for(model: str, effort: str | None) -> dict:
    _REASONING_EFFORT_REJECTED.clear()
    return _chat_create_kwargs(
        model=model,
        messages=[{"role": "user", "content": "Hi"}],
        reasoning_effort=effort,
    )


def test_reasoning_effort_sent_for_grok_43_ids() -> None:
    """Current grok-4.3 ids receive reasoning_effort even without the word."""
    assert _kwargs_for("grok-4.3", "low")["reasoning_effort"] == "low"
    assert _kwargs_for("grok-4.3-latest", "high")["reasoning_effort"] == "high"
    assert _kwargs_for("grok-4-1-fast-reasoning", "medium")["reasoning_effort"] == "medium"
    assert "reasoning_effort" not in _kwargs_for("grok-4.6", "low")
    assert "reasoning_effort" not in _kwargs_for("grok-4.3", "none")
    assert "reasoning_effort" not in _kwargs_for("grok-4.3", None)


def _sample_response() -> MagicMock:
    response = MagicMock()
    response.content = "ok"
    response.tool_calls = []
    response.finish_reason = "REASON_STOP"
    response.usage = None
    response.citations = []
    return response


@pytest.mark.asyncio
async def test_rejected_reasoning_effort_is_dropped_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A model that refuses reasoning_effort is retried without it, once."""
    _REASONING_EFFORT_REJECTED.clear()
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        chat = MagicMock()
        if "reasoning_effort" in kwargs:
            chat.sample = AsyncMock(
                side_effect=XAIInvalidArgumentError("unknown field reasoning_effort")
            )
        else:
            chat.sample = AsyncMock(return_value=_sample_response())
        return chat

    client = MagicMock()
    client.chat.create = MagicMock(side_effect=create)
    messages = [{"role": "user", "content": "Hi"}]

    with caplog.at_level("WARNING"):
        result = await async_chat_completion(
            client,
            model="grok-4.3",
            messages=messages,
            reasoning_effort="low",
        )
        assert result.content == "ok"
        calls.clear()
        await async_chat_completion(
            client,
            model="grok-4.3",
            messages=messages,
            reasoning_effort="low",
        )

    assert calls and "reasoning_effort" not in calls[0]
    assert caplog.text.count("omitting it for this id") == 1


@pytest.mark.asyncio
async def test_stream_retries_without_rejected_reasoning_effort() -> None:
    """chat.stream drops reasoning_effort when the model rejects the field."""
    _REASONING_EFFORT_REJECTED.clear()
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(dict(kwargs))
        chat = MagicMock()

        async def _stream():
            if "reasoning_effort" in kwargs:
                raise XAIInvalidArgumentError("reasoning_effort is not supported")
            yield _sample_response(), MagicMock()

        chat.stream = _stream
        return chat

    client = MagicMock()
    client.chat.create = MagicMock(side_effect=create)
    chunks = [
        pair
        async for pair in async_chat_stream(
            client,
            model="grok-4.3-latest",
            messages=[{"role": "user", "content": "Hi"}],
            reasoning_effort="medium",
        )
    ]
    assert len(chunks) == 1
    assert "reasoning_effort" in calls[0]
    assert "reasoning_effort" not in calls[1]
    assert "grok-4.3-latest" in _REASONING_EFFORT_REJECTED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        XAIAuthError("reasoning_effort rejected for this key"),
        XAIRateLimitError("rate limited while sending reasoning_effort"),
        XAIInvalidArgumentError(
            "invalid reasoning_effort value 'extreme'; must be one of low, medium, high"
        ),
    ],
)
async def test_other_errors_do_not_drop_reasoning_effort(error: Exception) -> None:
    """Only an unsupported-field error permanently omits reasoning_effort."""
    _REASONING_EFFORT_REJECTED.clear()
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        chat = MagicMock()
        chat.sample = AsyncMock(side_effect=error)
        return chat

    client = MagicMock()
    client.chat.create = MagicMock(side_effect=create)
    with pytest.raises(type(error)):
        await async_chat_completion(
            client,
            model="grok-4.3",
            messages=[{"role": "user", "content": "Hi"}],
            reasoning_effort="low",
        )

    assert len(calls) == 1
    assert "reasoning_effort" in calls[0]
    assert "grok-4.3" not in _REASONING_EFFORT_REJECTED
    follow_up = _kwargs_for("grok-4.3", "low")
    assert follow_up["reasoning_effort"] == "low"
