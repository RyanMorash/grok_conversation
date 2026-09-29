"""Unit tests for xai-sdk adapter helpers."""

from __future__ import annotations

import pytest
from xai_sdk.proto import chat_pb2
from xai_sdk.search import SearchParameters

from custom_components.grok_conversation.api_helpers import (
    XAIAuthError,
    XAIConnectionError,
    XAIError,
    XAIRateLimitError,
    build_search_parameters,
    convert_messages,
    convert_response_format,
    convert_tools,
    format_citations,
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
