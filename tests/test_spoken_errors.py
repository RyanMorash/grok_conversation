"""Translated Assist errors for rate limit, auth, token length, and connect."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components import conversation
from homeassistant.components.conversation.chat_log import ChatLog
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er, intent
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.grok_conversation.api_helpers import (
    XAIAuthError,
    XAIConnectionError,
    XAIError,
    XAIRateLimitError,
)
from custom_components.grok_conversation.const import DOMAIN
from custom_components.grok_conversation.exceptions import (
    TokenLengthExceededError,
    xai_error_translation_key,
)
from custom_components.grok_conversation.voice_api import XAIVoiceError

_ROOT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "grok_conversation"
)
_STRINGS = _ROOT / "strings.json"
_ENGLISH = _ROOT / "translations" / "en.json"

_KEPT = {
    "rate_limit": "xAI is busy right now. Please try again in a moment.",
    "auth_failed": "The xAI API key was rejected. Update it in settings.",
    "token_length": "That reply was cut off. Raise the token limit and try again.",
    "cannot_connect": "Could not reach xAI. Please try again.",
}


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


def _speech(result: conversation.ConversationResult) -> str:
    return result.response.speech["plain"]["speech"]


async def _converse(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    text: str,
    language: str,
) -> conversation.ConversationResult:
    return await conversation.async_converse(
        hass,
        text,
        conversation_id=None,
        context=Context(),
        language=language,
        agent_id=entry.entry_id,
    )


def test_exception_strings_are_plain_english() -> None:
    """English catalogs share the stable sentences, with no key references."""
    strings = json.loads(_STRINGS.read_text())["exceptions"]
    english = json.loads(_ENGLISH.read_text())["exceptions"]
    assert {key: english[key]["message"] for key in _KEPT} == _KEPT
    assert {key: strings[key]["message"] for key in _KEPT} == _KEPT
    for message in _KEPT.values():
        assert "[%key" not in message
        assert "{" not in message
    for path in (_ROOT / "translations").glob("*.json"):
        data = json.loads(path.read_text())
        for key in ("rate_limit", "auth_failed", "token_length"):
            message = data["exceptions"][key]["message"]
            assert message
            assert "[%key" not in message


def test_translation_key_follows_exception_type() -> None:
    """Rate limit, auth, and token length stay distinct. Everything else connects."""
    assert xai_error_translation_key(TokenLengthExceededError(600)) == "token_length"
    assert xai_error_translation_key(XAIRateLimitError("quota")) == "rate_limit"
    assert xai_error_translation_key(XAIAuthError("revoked")) == "auth_failed"
    assert xai_error_translation_key(XAIConnectionError("down")) == "cannot_connect"
    assert xai_error_translation_key(XAIError("boom")) == "cannot_connect"
    assert xai_error_translation_key(XAIVoiceError("nope", status=401)) == "auth_failed"
    assert xai_error_translation_key(XAIVoiceError("nope", status=403)) == "auth_failed"
    assert xai_error_translation_key(XAIVoiceError("slow", status=429)) == "rate_limit"
    assert (
        xai_error_translation_key(XAIVoiceError("down", status=500))
        == "cannot_connect"
    )
    assert xai_error_translation_key(XAIVoiceError("network")) == "cannot_connect"
    wrapped = RuntimeError("wrapped")
    wrapped.__cause__ = XAIAuthError("revoked")
    assert xai_error_translation_key(wrapped) == "auth_failed"
    plain = HomeAssistantError("Rate limited or insufficient funds")
    assert xai_error_translation_key(plain) == "cannot_connect"


async def test_assist_sees_translated_failures(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Assist records a failure whose speech is the catalog sentence."""
    cases = (
        (XAIRateLimitError("quota-secret"), "rate_limit", "quota-secret"),
        (XAIAuthError("revoked-secret"), "auth_failed", "revoked-secret"),
        (XAIError("provider-secret"), "cannot_connect", "provider-secret"),
    )
    for exc, key, hidden in cases:
        caplog.clear()
        _install_raising_stream(mock_xai_client, exc)
        with caplog.at_level("ERROR"):
            result = await _converse(hass, mock_config_entry, "hello", "fr")
        speech = _speech(result)
        assert result.response.error_code == intent.IntentResponseErrorCode.UNKNOWN
        assert speech == _KEPT[key].rstrip(".")
        assert hidden not in speech
        assert hidden in caplog.text

    caplog.clear()
    _install_length_stream(mock_xai_client)
    with caplog.at_level("ERROR"):
        length = await _converse(hass, mock_config_entry, "explain in detail", "de")
    speech = _speech(length)
    assert length.response.error_code == intent.IntentResponseErrorCode.UNKNOWN
    assert speech == _KEPT["token_length"].rstrip(".")
    assert "600" not in speech
    assert "token limit" in caplog.text.lower() or "tokens" in caplog.text.lower()


async def test_other_home_assistant_errors_reach_assist(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """A HomeAssistantError that is not an xAI failure keeps its own message."""
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


async def test_unexpected_errors_propagate(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Bugs are logged and leave the handler instead of a spoken apology."""
    agent = conversation.async_get_agent(hass, mock_config_entry.entry_id)
    assert agent is not None

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("handler bug")

    agent._async_handle_message_inner = _boom  # type: ignore[method-assign]
    chat_log = ChatLog(hass=hass, conversation_id="spoken-errors")
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="handler bug"):
        await agent._async_handle_message(  # noqa: SLF001
            _user_input(agent, "hello"), chat_log
        )
    assert "handler bug" in caplog.text
    assert "Sorry, I encountered an unexpected error" not in caplog.text


async def test_tts_errors_use_translation_keys(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """TTS failures use the same keys and keep the provider text in the log."""
    registry = er.async_get(hass)
    entity_id = next(
        entity.entity_id
        for entity in registry.entities.values()
        if entity.domain == "tts"
        and entity.config_entry_id == mock_config_entry.entry_id
    )
    entity = hass.data["tts"].get_entity(entity_id)
    assert entity is not None
    cases = (
        (XAIVoiceError("tts-secret", status=401), "auth_failed"),
        (XAIVoiceError("tts-secret", status=429), "rate_limit"),
        (XAIVoiceError("tts-secret"), "cannot_connect"),
    )
    for exc, key in cases:
        caplog.clear()
        with (
            caplog.at_level("ERROR"),
            patch(
                "custom_components.grok_conversation.tts.async_tts",
                new=AsyncMock(side_effect=exc),
            ),
            pytest.raises(HomeAssistantError) as caught,
        ):
            await entity.async_get_tts_audio("hello", "en", {})
        assert caught.value.translation_domain == DOMAIN
        assert caught.value.translation_key == key
        assert caught.value.translation_placeholders is None
        assert "tts-secret" not in str(caught.value)
        assert "tts-secret" in caplog.text
