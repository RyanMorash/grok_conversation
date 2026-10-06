"""One streamed chat.create for Assist, including tools plus live search."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.components.conversation.chat_log import ChatLog
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry
from xai_sdk.proto import chat_pb2

from custom_components.grok_conversation.api_helpers import (
    CLIENT_TIMEOUT_SECONDS,
    XAIAuthError,
    XAIConnectionError,
    XAIError,
    XAIInvalidArgumentError,
    XAIRateLimitError,
)
from custom_components.grok_conversation.const import (
    CONF_AUTO_MODEL_ROUTING,
    CONF_CHAT_MODEL,
    CONF_FALLBACK_MODEL,
    CONF_LIVE_SEARCH,
    DOMAIN,
    LIVE_SEARCH_WEB,
)
from custom_components.grok_conversation.conversation import (
    SATELLITE_STREAM_TIMEOUT_SECONDS,
)
from custom_components.grok_conversation.exceptions import TokenLengthExceededError


def test_client_timeout_stays_at_two_minutes() -> None:
    """AI Task keeps the shared client timeout; satellite cuts only the stream."""
    assert CLIENT_TIMEOUT_SECONDS == 120.0
    assert SATELLITE_STREAM_TIMEOUT_SECONDS == 45.0


def _chat_response(
    text: str = "",
    *,
    tool_calls=None,
    finish_reason: str = "REASON_STOP",
    citations=None,
):
    result = MagicMock()
    result.content = text
    result.tool_calls = tool_calls or []
    result.finish_reason = finish_reason
    result.usage.prompt_tokens = 11
    result.usage.completion_tokens = 7
    result.citations = citations or []
    return result


def _tool_call(name: str, arguments: str, call_id: str = "call_1"):
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _pair(
    text: str,
    *,
    content: str | None = None,
    tool_calls=None,
    finish_reason: str = "REASON_STOP",
    citations=None,
):
    chunk = MagicMock()
    chunk.content = text
    chunk.tool_calls = []
    response = _chat_response(
        content if content is not None else text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        citations=citations,
    )
    return response, chunk


def _gen(pairs, *, pause_after: int | None = None, delay: float = 0.0):
    async def _stream():
        for index, pair in enumerate(pairs):
            if pause_after is not None and index == pause_after:
                await asyncio.sleep(delay)
            yield pair

    return _stream


def _raise_gen(exc: BaseException):
    async def _stream():
        raise exc
        yield None  # pragma: no cover - keeps this an async generator

    return _stream


def _install_raising_stream(client: MagicMock, exc: BaseException) -> None:
    async def _stream():
        raise exc
        yield None  # pragma: no cover - keeps this an async generator

    client.chat.create.return_value.stream = _stream


def _install_streams(client: MagicMock, factories: list) -> None:
    pending = list(factories)

    def stream():
        factory = pending.pop(0)
        return factory()

    client.chat.create.return_value.stream = stream


def _message_role(msg) -> str:
    return chat_pb2.MessageRole.Name(msg.role).removeprefix("ROLE_").lower()


def _message_text(msg) -> str:
    return "".join(part.text for part in msg.content)


def _speech(result) -> str:
    return result.response.speech["plain"]["speech"]


class _RecordingTool(llm.Tool):
    """Test tool that records calls."""

    name = "test_light"
    description = "Control a test light"
    parameters = vol.Schema({vol.Required("action"): str}, extra=vol.ALLOW_EXTRA)

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> llm.ToolResult:
        self.calls.append(dict(tool_input.tool_args))
        return llm.ToolResult(
            data={"ok": True, "action": tool_input.tool_args.get("action")}
        )


def _mock_llm_api(hass: HomeAssistant, tool: llm.Tool) -> llm.APIInstance:
    api = MagicMock(spec=llm.API)
    api.hass = hass
    api.id = "test"
    api.name = "Test"
    return llm.APIInstance(
        api=api,
        api_prompt="Test tools available.",
        llm_context=llm.LLMContext(
            platform=DOMAIN,
            context=Context(),
            language="en",
            assistant="conversation",
            device_id=None,
        ),
        tools=[tool],
    )


async def _use_options(
    hass: HomeAssistant, entry: MockConfigEntry, **updates: Any
) -> None:
    hass.config_entries.async_update_entry(
        entry,
        options={
            **dict(entry.options),
            CONF_AUTO_MODEL_ROUTING: False,
            **updates,
        },
    )
    await hass.async_block_till_done()


async def _turn(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    text: str,
    *,
    device_id: str | None = None,
    tool: _RecordingTool | None = None,
):
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    chat_log = ChatLog(hass=hass, conversation_id=entry.entry_id)
    deltas: list[dict[str, Any]] = []
    chat_log.delta_listener = lambda _log, delta: deltas.append(dict(delta))
    chat_log.async_add_user_content(conversation.UserContent(content=text))

    async def _provide(*_args, **_kwargs) -> None:
        if tool is not None:
            chat_log.llm_api = _mock_llm_api(hass, tool)

    user_input = conversation.ConversationInput(
        text=text,
        context=Context(),
        conversation_id=entry.entry_id,
        device_id=device_id,
        satellite_id=None,
        language="en",
        agent_id=agent.entity_id,
    )
    chat_log.async_provide_llm_data = _provide  # type: ignore[method-assign]
    result = await agent._async_handle_message_inner(user_input, chat_log)  # noqa: SLF001
    return agent, chat_log, result, deltas


async def test_conversation_entity_supports_streaming(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Assist is told this agent can stream."""
    agent = conversation.async_get_agent(hass, mock_config_entry.entry_id)
    assert agent is not None
    assert agent.supports_streaming is True


async def test_chat_log_without_stream_flag_still_samples(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """AI Task and existing chat-log callers keep chat.sample()."""
    agent = conversation.async_get_agent(hass, mock_config_entry.entry_id)
    assert agent is not None
    mock_xai_client.chat.create.return_value.sample = AsyncMock(
        return_value=_chat_response("Sampled.")
    )
    chat_log = ChatLog(hass=hass, conversation_id="sample-path")
    chat_log.async_add_user_content(conversation.UserContent(content="Hi"))
    await agent._async_handle_chat_log(  # noqa: SLF001
        chat_log,
        model="grok-4.3-latest",
        options={
            CONF_CHAT_MODEL: "grok-4.3-latest",
            CONF_FALLBACK_MODEL: "grok-4.6",
        },
        messages=[{"role": "user", "content": "Hi"}],
        agent_id=agent.entity_id,
        service="ai_task",
        fallback_model="",
    )
    assert mock_xai_client.chat.create.return_value.sample.await_count == 1
    assert mock_xai_client.chat.create.return_value.stream.call_count == 0


async def test_final_text_is_replayed_as_deltas_not_tool_payloads(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """Tool turns stay on the non-streaming path; the final reply is deltas."""
    tool = _RecordingTool()
    agent = conversation.async_get_agent(hass, mock_config_entry.entry_id)
    assert agent is not None
    _install_streams(
        mock_xai_client,
        [
            _gen(
                [
                    _pair(
                        "preamble",
                        tool_calls=[_tool_call("test_light", '{"action":"turn_on"}')],
                    )
                ]
            ),
            _gen([_pair("The light"), _pair(" is on.", content="The light is on.")]),
        ],
    )
    chat_log = ChatLog(hass=hass, conversation_id="stream-tools")
    deltas: list[dict[str, Any]] = []
    chat_log.delta_listener = lambda _log, delta: deltas.append(dict(delta))
    chat_log.llm_api = _mock_llm_api(hass, tool)
    chat_log.async_add_user_content(conversation.UserContent(content="Turn on the light"))
    await agent._async_handle_chat_log(  # noqa: SLF001
        chat_log,
        model="grok-4.3-latest",
        options={CONF_CHAT_MODEL: "grok-4.3-latest", CONF_FALLBACK_MODEL: ""},
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Turn on the light"},
        ],
        agent_id=agent.entity_id,
        service="conversation",
        fallback_model="",
        stream_final=True,
    )

    assert tool.calls == [{"action": "turn_on"}]
    assert mock_xai_client.chat.create.return_value.sample.await_count == 0
    heard = "".join(str(delta.get("content") or "") for delta in deltas)
    assert "preamble" not in heard
    assert "test_light" not in heard
    assert all("tool_calls" not in delta for delta in deltas)
    assert heard == "The light is on."
    assert any(
        isinstance(item, conversation.AssistantContent)
        and item.content == "The light is on."
        for item in chat_log.content
    )


async def test_search_and_tools_share_one_streamed_create(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """Live search parameters ride on the same create as Home Assistant tools."""
    await _use_options(hass, mock_config_entry, **{CONF_LIVE_SEARCH: LIVE_SEARCH_WEB})
    tool = _RecordingTool()
    _install_streams(
        mock_xai_client,
        [
            _gen(
                [
                    _pair(
                        "Score ",
                        citations=["https://example.com/game"],
                    ),
                    _pair(
                        "is 2-1.",
                        content="Score is 2-1.",
                        citations=["https://example.com/game"],
                    ),
                ]
            )
        ],
    )
    _agent, chat_log, result, deltas = await _turn(
        hass,
        mock_config_entry,
        "latest score",
        tool=tool,
    )

    assert mock_xai_client.chat.create.call_count == 1
    kwargs = mock_xai_client.chat.create.call_args.kwargs
    assert kwargs["search_parameters"] is not None
    assert kwargs["tools"]
    texts = [
        _message_text(message)
        for message in kwargs["messages"]
        if _message_role(message) == "system"
    ]
    assert any("live web/X search" in text for text in texts)
    assert mock_xai_client.chat.create.return_value.sample.await_count == 0
    speech = _speech(result)
    assert "Score is 2-1." in speech
    assert "https://example.com/game" in speech
    assert "Sources:" in speech
    assert any("https://example.com/game" in str(delta.get("content")) for delta in deltas)
    assert tool.calls == []
    assert chat_log.content


async def test_search_only_streams_without_a_second_sample(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """Search without tools uses one streamed create, not the old sample pass."""
    await _use_options(hass, mock_config_entry, **{CONF_LIVE_SEARCH: LIVE_SEARCH_WEB})
    _install_streams(
        mock_xai_client,
        [_gen([_pair("Headline"), _pair(" today.", content="Headline today.")])],
    )
    _agent, _log, result, _deltas = await _turn(
        hass, mock_config_entry, "latest score"
    )
    assert mock_xai_client.chat.create.call_count == 1
    kwargs = mock_xai_client.chat.create.call_args.kwargs
    assert kwargs["search_parameters"] is not None
    assert "tools" not in kwargs
    assert mock_xai_client.chat.create.return_value.sample.await_count == 0
    assert _speech(result) == "Headline today."


async def test_combined_rejection_uses_two_pass_sample(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """A rejected tools+search create falls back to the sample two-pass path."""
    await _use_options(hass, mock_config_entry, **{CONF_LIVE_SEARCH: LIVE_SEARCH_WEB})
    tool = _RecordingTool()
    _install_streams(
        mock_xai_client,
        [
            _raise_gen(
                XAIInvalidArgumentError(
                    "tools and search_parameters are not supported together"
                )
            )
        ],
    )
    mock_xai_client.chat.create.return_value.sample = AsyncMock(
        side_effect=[
            _chat_response(
                "Live fact",
                citations=["https://example.com/fact"],
            ),
            _chat_response("Using the live fact."),
        ]
    )
    _agent, _log, result, _deltas = await _turn(
        hass,
        mock_config_entry,
        "latest score",
        tool=tool,
    )

    assert _speech(result) == "Using the live fact."
    assert mock_xai_client.chat.create.call_count == 3
    first, search, tools = mock_xai_client.chat.create.call_args_list
    assert first.kwargs["search_parameters"] is not None
    assert first.kwargs["tools"]
    assert search.kwargs["search_parameters"] is not None
    assert "tools" not in search.kwargs
    assert "search_parameters" not in tools.kwargs
    assert tools.kwargs["tools"]
    assert mock_xai_client.chat.create.return_value.sample.await_count == 2


async def test_post_tool_request_keeps_search_parameters(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """A follow-up after a Home Assistant tool still sends live search."""
    await _use_options(hass, mock_config_entry, **{CONF_LIVE_SEARCH: LIVE_SEARCH_WEB})
    tool = _RecordingTool()
    _install_streams(
        mock_xai_client,
        [
            _gen(
                [
                    _pair(
                        "checking",
                        tool_calls=[_tool_call("test_light", '{"action":"turn_on"}')],
                    )
                ]
            ),
            _gen([_pair("The score is 2-1.")]),
        ],
    )
    _agent, _log, result, _deltas = await _turn(
        hass,
        mock_config_entry,
        "latest score",
        tool=tool,
    )

    assert tool.calls == [{"action": "turn_on"}]
    assert _speech(result) == "The score is 2-1."
    assert mock_xai_client.chat.create.call_count == 2
    first, follow_up = mock_xai_client.chat.create.call_args_list
    assert first.kwargs["search_parameters"] is not None
    assert follow_up.kwargs["search_parameters"] is not None
    assert follow_up.kwargs["tools"]
    roles = [_message_role(message) for message in follow_up.kwargs["messages"]]
    assert "tool" in roles
    assert mock_xai_client.chat.create.return_value.sample.await_count == 0


async def test_auth_and_connection_errors_do_not_two_pass(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """Revoked keys and outages stay on normal error handling."""
    await _use_options(hass, mock_config_entry, **{CONF_LIVE_SEARCH: LIVE_SEARCH_WEB})
    tool = _RecordingTool()
    for exc, match in (
        (XAIAuthError("revoked"), "revoked"),
        (XAIConnectionError("unreachable"), "unreachable"),
        (XAIError("primary down"), "primary down"),
    ):
        mock_xai_client.chat.create.reset_mock()
        mock_xai_client.chat.create.return_value.sample = AsyncMock()
        _install_raising_stream(mock_xai_client, exc)
        with pytest.raises(HomeAssistantError, match=match):
            await _turn(hass, mock_config_entry, "latest score", tool=tool)
        assert mock_xai_client.chat.create.return_value.sample.await_count == 0
        assert mock_xai_client.chat.create.call_count >= 1
        for call in mock_xai_client.chat.create.call_args_list:
            assert call.kwargs["search_parameters"] is not None
            assert call.kwargs["tools"]


async def test_combined_rate_limit_does_not_two_pass(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """Rate limits stay on the streamed call."""
    await _use_options(hass, mock_config_entry, **{CONF_LIVE_SEARCH: LIVE_SEARCH_WEB})
    tool = _RecordingTool()
    _install_streams(mock_xai_client, [_raise_gen(XAIRateLimitError("quota"))])
    with pytest.raises(HomeAssistantError, match="Rate limited or insufficient funds"):
        await _turn(hass, mock_config_entry, "latest score", tool=tool)
    assert mock_xai_client.chat.create.return_value.sample.await_count == 0
    assert mock_xai_client.chat.create.call_count == 1


async def test_combined_token_length_does_not_two_pass(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """A length finish is not treated as a rejected combined call."""
    await _use_options(hass, mock_config_entry, **{CONF_LIVE_SEARCH: LIVE_SEARCH_WEB})
    tool = _RecordingTool()
    _install_streams(
        mock_xai_client,
        [_gen([_pair("Partial", finish_reason="REASON_MAX_LEN")])],
    )
    with pytest.raises(TokenLengthExceededError):
        await _turn(hass, mock_config_entry, "latest score", tool=tool)
    assert mock_xai_client.chat.create.return_value.sample.await_count == 0
    assert mock_xai_client.chat.create.call_count == 1


async def test_satellite_stream_stops_at_patched_timeout(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A satellite turn stops the stream; other turns wait for the rest."""
    monkeypatch.setattr(
        "custom_components.grok_conversation.conversation.SATELLITE_STREAM_TIMEOUT_SECONDS",
        0.05,
    )
    _install_streams(
        mock_xai_client,
        [
            _gen(
                [_pair("Hello"), _pair(" world", content="Hello world")],
                pause_after=1,
                delay=0.4,
            )
        ],
    )
    _agent, _log, result, _deltas = await _turn(
        hass,
        mock_config_entry,
        "hello there friend",
        device_id="satellite-1",
    )
    assert _speech(result) == "Hello"
    assert "world" not in _speech(result)

    _install_streams(
        mock_xai_client,
        [
            _gen(
                [_pair("Hello"), _pair(" world", content="Hello world")],
                pause_after=1,
                delay=0.4,
            )
        ],
    )
    _agent, _log, result, _deltas = await _turn(
        hass,
        mock_config_entry,
        "hello there friend",
    )
    assert _speech(result) == "Hello world"


async def test_satellite_reply_omits_citations(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """Spoken Assist does not append citation footnotes."""
    await _use_options(hass, mock_config_entry, **{CONF_LIVE_SEARCH: LIVE_SEARCH_WEB})
    _install_streams(
        mock_xai_client,
        [
            _gen(
                [
                    _pair(
                        "Sunny.",
                        citations=["https://example.com/weather"],
                    )
                ]
            )
        ],
    )
    _agent, _log, result, deltas = await _turn(
        hass,
        mock_config_entry,
        "latest score",
        device_id="satellite-1",
    )
    speech = _speech(result)
    assert speech == "Sunny."
    assert "Sources:" not in speech
    assert all("Sources:" not in str(delta.get("content")) for delta in deltas)
