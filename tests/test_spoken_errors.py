"""Spoken Assist errors for rate limit, auth failure, and token length."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from homeassistant.components import conversation
from homeassistant.components.conversation.chat_log import ChatLog
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.grok_conversation.api_helpers import (
    XAIAuthError,
    XAIRateLimitError,
)
from custom_components.grok_conversation.const import DOMAIN
from custom_components.grok_conversation.conversation import spoken_error_translation_key
from custom_components.grok_conversation.exceptions import TokenLengthExceededError

_STRINGS = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "grok_conversation"
    / "strings.json"
)


def _chat_response(text: str = "", *, finish_reason: str = "REASON_STOP"):
    from unittest.mock import MagicMock

    result = MagicMock()
    result.content = text
    result.tool_calls = []
    result.finish_reason = finish_reason
    result.usage.prompt_tokens = 11
    result.usage.completion_tokens = 7
    result.citations = []
    return result


def _install_raising_stream(client, exc: BaseException) -> None:
    async def _stream():
        raise exc
        yield None  # pragma: no cover

    client.chat.create.return_value.stream = _stream


def _install_length_stream(client) -> None:
    response = _chat_response("Partial", finish_reason="REASON_MAX_LEN")
    chunk = type("Chunk", (), {"content": "Partial", "tool_calls": []})()

    async def _stream():
        yield response, chunk

    client.chat.create.return_value.stream = _stream


def _user_input(agent, text: str) -> conversation.ConversationInput:
    return conversation.ConversationInput(
        text=text,
        context=Context(),
        conversation_id="spoken-errors",
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id=agent.entity_id,
    )


async def _raise_from_handler(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    client,
    text: str,
) -> HomeAssistantError:
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    chat_log = ChatLog(hass=hass, conversation_id="spoken-errors")
    chat_log.async_add_user_content(conversation.UserContent(content=text))
    with pytest.raises(HomeAssistantError) as caught:
        await agent._async_handle_message(  # noqa: SLF001
            _user_input(agent, text), chat_log
        )
    return caught.value


def test_spoken_error_strings_are_translated() -> None:
    """The three short speeches live in strings and the locale files."""
    catalog = json.loads(_STRINGS.read_text())
    messages = catalog["exceptions"]
    assert messages["rate_limit"]["message"].startswith("xAI is busy")
    assert "API key" in messages["auth_failed"]["message"]
    assert "token limit" in messages["token_length"]["message"]
    root = _STRINGS.parent / "translations"
    for path in root.glob("*.json"):
        data = json.loads(path.read_text())
        for key in ("rate_limit", "auth_failed", "token_length"):
            assert data["exceptions"][key]["message"]


def test_mapper_ignores_unrelated_errors() -> None:
    """Only the three Assist failures get a speech key."""
    assert spoken_error_translation_key(TokenLengthExceededError(600)) == "token_length"
    assert (
        spoken_error_translation_key(
            HomeAssistantError("Rate limited or insufficient funds")
        )
        == "rate_limit"
    )
    assert (
        spoken_error_translation_key(
            HomeAssistantError("Error talking to xAI: revoked")
        )
        is None
    )
    wrapped = HomeAssistantError("Error talking to xAI: revoked")
    wrapped.__cause__ = XAIAuthError("revoked")
    assert spoken_error_translation_key(wrapped) == "auth_failed"


async def test_rate_limit_auth_and_token_length_reach_assist(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client,
) -> None:
    """Those failures leave the handler as translated HomeAssistantError."""
    _install_raising_stream(mock_xai_client, XAIRateLimitError("quota"))
    rate = await _raise_from_handler(
        hass, mock_config_entry, mock_xai_client, "turn on the lights"
    )
    assert rate.translation_domain == DOMAIN
    assert rate.translation_key == "rate_limit"

    _install_raising_stream(mock_xai_client, XAIAuthError("revoked"))
    auth = await _raise_from_handler(
        hass, mock_config_entry, mock_xai_client, "hello"
    )
    assert auth.translation_key == "auth_failed"
    assert isinstance(auth.__cause__, HomeAssistantError)

    _install_length_stream(mock_xai_client)
    length = await _raise_from_handler(
        hass, mock_config_entry, mock_xai_client, "explain in detail"
    )
    assert length.translation_key == "token_length"
    assert isinstance(length.__cause__, TokenLengthExceededError)


async def test_other_home_assistant_errors_reach_assist(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """A HomeAssistantError that is not one of the three is not swallowed."""
    agent = conversation.async_get_agent(hass, mock_config_entry.entry_id)
    assert agent is not None

    async def _images(*_args, **_kwargs):
        raise HomeAssistantError("Only images are supported")

    agent._async_handle_message_inner = _images  # type: ignore[method-assign]
    chat_log = ChatLog(hass=hass, conversation_id="spoken-errors")
    with pytest.raises(HomeAssistantError, match="Only images are supported") as caught:
        await agent._async_handle_message(  # noqa: SLF001
            _user_input(agent, "look at this"), chat_log
        )
    assert caught.value.translation_key is None


async def test_unexpected_errors_stay_local_speech(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Bugs that are not HomeAssistantError still get the generic spoken reply."""
    agent = conversation.async_get_agent(hass, mock_config_entry.entry_id)
    assert agent is not None

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("handler bug")

    agent._async_handle_message_inner = _boom  # type: ignore[method-assign]
    chat_log = ChatLog(hass=hass, conversation_id="spoken-errors")
    result = await agent._async_handle_message(  # noqa: SLF001
        _user_input(agent, "hello"), chat_log
    )
    speech = result.response.speech["plain"]["speech"]
    assert "unexpected error" in speech.lower()
