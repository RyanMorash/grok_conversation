"""Voice probe cache, chat-only repair, and satellite pipeline prewarm."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import ClientError
import homeassistant.components as ha_components
from homeassistant.components import conversation

# The Assist pipeline package init imports pymicro_vad, which is not installed
# in this test environment. Register the package path so the store models can
# load without that audio dependency. A normal Home Assistant install imports
# the real package before prewarm runs.
_ASSIST_PKG = "homeassistant.components.assist_pipeline"
if _ASSIST_PKG not in sys.modules:
    _assist = types.ModuleType(_ASSIST_PKG)
    _assist.__path__ = [str(Path(ha_components.__file__).parent / "assist_pipeline")]  # type: ignore[attr-defined]
    _assist.__package__ = _ASSIST_PKG
    sys.modules[_ASSIST_PKG] = _assist

from homeassistant.components.assist_pipeline.models import Pipeline  # noqa: E402
from homeassistant.components.assist_pipeline.runtime import (  # noqa: E402
    KEY_ASSIST_PIPELINE,
    AssistDevice,
)
from homeassistant.components.conversation.chat_log import AssistantContent, ChatLog
from homeassistant.config_entries import SOURCE_USER, ConfigEntryState
from homeassistant.const import CONF_API_KEY
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.grok_conversation.config_flow import RECOMMENDED_OPTIONS
from custom_components.grok_conversation.const import (
    CONF_RECOMMENDED,
    DOMAIN,
    SERVICE_CLEAR_MEMORY,
    SERVICE_RECHECK_VOICE,
)
from custom_components.grok_conversation.repairs import VoiceRecheckFlow
from custom_components.grok_conversation.voice_api import async_validate_voice_access
from custom_components.grok_conversation.voice_const import (
    CONF_RECHECK_VOICE,
    CONF_VOICE_ACCESS,
)


class _Response:
    def __init__(self, status: int, payload=None, text: str = "") -> None:
        self.status = status
        self._payload = payload
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def text(self) -> str:
        return self._text

    async def json(self, content_type=None):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Session:
    def __init__(self, response: _Response | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.posts: list[str] = []
        self.gets: list[str] = []

    def get(self, url, **kwargs):
        self.gets.append(url)
        if self.error:
            raise self.error
        return self.response

    def post(self, url, **kwargs):
        self.posts.append(url)
        raise AssertionError("voice check must not synthesize speech")


def _entry(**data) -> MockConfigEntry:
    return MockConfigEntry(
        domain=DOMAIN,
        title="xAI Grok",
        data={CONF_API_KEY: "sk-test", **data},
        options=dict(RECOMMENDED_OPTIONS),
        version=1,
        minor_version=4,
    )


def _pipeline(pipeline_id: str, engine: str) -> Pipeline:
    return Pipeline(
        conversation_engine="conversation.grok",
        conversation_language="en",
        language="en",
        name=pipeline_id,
        stt_engine=None,
        stt_language=None,
        tts_engine=engine,
        tts_language="en",
        tts_voice="ara",
        wake_word_entity=None,
        wake_word_id=None,
        id=pipeline_id,
    )


class _Store:
    def __init__(self, pipelines: list[Pipeline], preferred: str) -> None:
        self.data = {pipeline.id: pipeline for pipeline in pipelines}
        self._preferred = preferred

    def async_get_preferred_item(self) -> str:
        return self._preferred

    def async_items(self):
        return list(self.data.values())


def _install_pipelines(hass: HomeAssistant, devices: dict[str, AssistDevice]) -> None:
    device = _pipeline("device-pipe", "tts.satellite")
    preferred = _pipeline("preferred-pipe", "tts.preferred")
    hass.data[KEY_ASSIST_PIPELINE] = MagicMock(
        pipeline_store=_Store([device, preferred], "preferred-pipe"),
        pipeline_devices=devices,
    )


def _conversation_entity(hass: HomeAssistant, entry: MockConfigEntry):
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    return agent


async def test_unusable_voices_list_does_not_synthesize() -> None:
    """401, 404, empty, and network errors never POST a greeting."""
    denied = _Session(_Response(401, text="no"))
    ok, detail = await async_validate_voice_access(denied, "sk")  # type: ignore[arg-type]
    assert ok is False
    assert "API key rejected" in detail
    assert denied.posts == []

    missing = _Session(_Response(404))
    ok, detail = await async_validate_voice_access(missing, "sk")  # type: ignore[arg-type]
    assert ok is False
    assert detail == "Voices list was not usable (404)"
    assert missing.posts == []

    empty = _Session(_Response(200, payload={"voices": []}))
    ok, detail = await async_validate_voice_access(empty, "sk")  # type: ignore[arg-type]
    assert ok is False
    assert detail == "Voices list was not usable"
    assert empty.posts == []

    down = _Session(error=ClientError("timeout"))
    ok, detail = await async_validate_voice_access(down, "sk")  # type: ignore[arg-type]
    assert ok is False
    assert detail.startswith("Could not reach")
    assert down.posts == []

    timed_out = _Session(error=TimeoutError())
    ok, detail = await async_validate_voice_access(timed_out, "sk")  # type: ignore[arg-type]
    assert ok is False
    assert detail.startswith("Could not reach")
    assert timed_out.posts == []


async def test_usable_voices_list_does_not_post() -> None:
    """A voices payload with an id is enough; no speech sample is sent."""
    session = _Session(
        _Response(200, payload={"voices": [{"voice_id": "eve", "name": "Eve"}]})
    )
    ok, detail = await async_validate_voice_access(session, "sk")  # type: ignore[arg-type]
    assert ok is True
    assert "voices list" in detail
    assert session.posts == []
    assert session.gets


async def test_setup_caches_probe_and_reload_reuses_it(hass: HomeAssistant) -> None:
    """Setup stores the voices-list result and a later reload does not probe."""
    probe = AsyncMock(return_value=(False, "Voices list was not usable"))
    entry = _entry()
    entry.add_to_hass(hass)
    with patch(
        "custom_components.grok_conversation.async_validate_voice_access",
        probe,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        assert probe.await_count == 1
        assert entry.data[CONF_VOICE_ACCESS] == {
            "ok": False,
            "detail": "Voices list was not usable",
        }
        issue = ir.async_get(hass).async_get_issue(
            DOMAIN, f"voice_chat_only_{entry.entry_id}"
        )
        assert issue is not None
        assert issue.translation_key == "voice_chat_only"

        assert await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()
        assert probe.await_count == 1


async def test_connection_failure_does_not_raise_repair(hass: HomeAssistant) -> None:
    """A network failure is not a chat-only key."""
    probe = AsyncMock(return_value=(False, "Could not reach xAI Voice API: down"))
    entry = _entry()
    entry.add_to_hass(hass)
    with patch(
        "custom_components.grok_conversation.async_validate_voice_access",
        probe,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    assert (
        ir.async_get(hass).async_get_issue(DOMAIN, f"voice_chat_only_{entry.entry_id}")
        is None
    )


async def test_reauth_replaces_voice_cache(hass: HomeAssistant) -> None:
    """Reauth stores the new key's probe and drops the previous cache."""
    entry = _entry(
        **{
            CONF_VOICE_ACCESS: {
                "ok": False,
                "detail": "Voices list was not usable",
            }
        }
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    with patch(
        "custom_components.grok_conversation.config_flow.async_validate_voice_access",
        return_value=(True, "Voice API accessible (voices list OK)"),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": "reauth", "entry_id": entry.entry_id},
            data=dict(entry.data),
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_API_KEY: "sk-new"}
        )
        await hass.async_block_till_done()

    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert entry.data[CONF_API_KEY] == "sk-new"
    assert entry.data[CONF_VOICE_ACCESS]["ok"] is True
    assert "not usable" not in entry.data[CONF_VOICE_ACCESS]["detail"]


async def test_clear_memory_does_not_reload(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """clear_memory leaves the loaded entry and its voice cache alone."""
    with patch.object(
        hass.config_entries,
        "async_reload",
        side_effect=AssertionError("reloaded"),
    ):
        result = await hass.services.async_call(
            DOMAIN,
            SERVICE_CLEAR_MEMORY,
            {"config_entry": mock_config_entry.entry_id},
            blocking=True,
            return_response=True,
        )
    assert result["status"] == "ok"
    assert mock_config_entry.state is ConfigEntryState.LOADED


async def test_recheck_reachability_keeps_existing_repair(hass: HomeAssistant) -> None:
    """A recheck that cannot reach Voice does not delete the chat-only repair."""
    probe = AsyncMock(
        side_effect=[
            (False, "Voices list was not usable"),
            (False, "Could not reach xAI Voice API: down"),
        ]
    )
    entry = _entry()
    entry.add_to_hass(hass)
    with patch(
        "custom_components.grok_conversation.async_validate_voice_access",
        probe,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        await hass.services.async_call(
            DOMAIN,
            SERVICE_RECHECK_VOICE,
            {"config_entry": entry.entry_id},
            blocking=True,
            return_response=True,
        )
        await hass.async_block_till_done()
    issue = ir.async_get(hass).async_get_issue(
        DOMAIN, f"voice_chat_only_{entry.entry_id}"
    )
    assert issue is not None
    assert entry.data[CONF_VOICE_ACCESS]["detail"].startswith("Could not reach")


async def test_user_flow_connectivity_is_not_chat_only(hass: HomeAssistant) -> None:
    """A Voice timeout does not ask the user to confirm a chat-only key."""
    with patch(
        "custom_components.grok_conversation.config_flow.async_validate_voice_access",
        return_value=(False, "Could not reach xAI Voice API: timed out"),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_API_KEY: "sk-chat"}
        )
        await hass.async_block_till_done()

    assert result["type"] == FlowResultType.CREATE_ENTRY
    created = result["result"]
    assert created.data[CONF_VOICE_ACCESS]["ok"] is False
    assert (
        ir.async_get(hass).async_get_issue(
            DOMAIN, f"voice_chat_only_{created.entry_id}"
        )
        is None
    )


async def test_recheck_service_clears_chat_only_issue(hass: HomeAssistant) -> None:
    """An explicit recheck stores a new probe and removes the repair."""
    probe = AsyncMock(
        side_effect=[
            (False, "Voices list was not usable"),
            (True, "Voice API accessible (voices list OK)"),
        ]
    )
    entry = _entry()
    entry.add_to_hass(hass)
    with patch(
        "custom_components.grok_conversation.async_validate_voice_access",
        probe,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        result = await hass.services.async_call(
            DOMAIN,
            SERVICE_RECHECK_VOICE,
            {"config_entry": entry.entry_id},
            blocking=True,
            return_response=True,
        )
        await hass.async_block_till_done()
    assert result["voice_ok"] is True
    assert entry.data[CONF_VOICE_ACCESS]["ok"] is True
    assert (
        ir.async_get(hass).async_get_issue(DOMAIN, f"voice_chat_only_{entry.entry_id}")
        is None
    )


async def test_user_flow_voice_note_then_creates(hass: HomeAssistant) -> None:
    """The first chat-only result is shown on the form; the next submit saves it."""
    with patch(
        "custom_components.grok_conversation.config_flow.async_validate_voice_access",
        return_value=(False, "Voices list was not usable (404)"),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": SOURCE_USER}
        )
        assert result["description_placeholders"]["voice_note"] == ""
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_API_KEY: "sk-chat"}
        )
        assert result["type"] == FlowResultType.FORM
        assert result["errors"]["base"] == "voice_chat_only"
        assert "xAI Voice is not available" in result["description_placeholders"]["voice_note"]
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_API_KEY: "sk-chat"}
        )
        await hass.async_block_till_done()

    assert result["type"] == FlowResultType.CREATE_ENTRY
    created = result["result"]
    assert created.data[CONF_VOICE_ACCESS]["ok"] is False
    issue = ir.async_get(hass).async_get_issue(
        DOMAIN, f"voice_chat_only_{created.entry_id}"
    )
    assert issue is not None


async def test_repair_flow_rechecks_voice(hass: HomeAssistant) -> None:
    """The repair confirm step probes again and reports the new result."""
    probe = AsyncMock(
        side_effect=[
            (False, "Voices list was not usable"),
            (False, "Voices list was not usable"),
            (True, "Voice API accessible (voices list OK)"),
        ]
    )
    entry = _entry()
    entry.add_to_hass(hass)
    with patch(
        "custom_components.grok_conversation.async_validate_voice_access",
        probe,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        flow = VoiceRecheckFlow()
        flow.hass = hass
        flow.data = {"entry_id": entry.entry_id}
        shown = await flow.async_step_init(None)
        assert shown["type"] == FlowResultType.FORM
        still = await flow.async_step_init({})
        assert still["type"] == FlowResultType.ABORT
        assert still["reason"] == "voice_still_unavailable"
        assert (
            ir.async_get(hass).async_get_issue(
                DOMAIN, f"voice_chat_only_{entry.entry_id}"
            )
            is not None
        )
        fixed = await flow.async_step_init({})
        await hass.async_block_till_done()
    assert fixed["reason"] == "voice_ok"
    assert (
        ir.async_get(hass).async_get_issue(DOMAIN, f"voice_chat_only_{entry.entry_id}")
        is None
    )


async def test_options_recheck_refreshes_cache(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """The options checkbox probes again and is not stored as an option."""
    hass.config_entries.async_update_entry(
        mock_config_entry,
        data={
            **mock_config_entry.data,
            CONF_VOICE_ACCESS: {"ok": False, "detail": "Voices list was not usable"},
        },
    )
    await hass.async_block_till_done()
    probe = AsyncMock(return_value=(True, "Voice API accessible (voices list OK)"))
    with patch(
        "custom_components.grok_conversation.async_validate_voice_access",
        probe,
    ):
        result = await hass.config_entries.options.async_init(
            mock_config_entry.entry_id
        )
        assert "xAI Voice is not available" in result["description_placeholders"]["voice_note"]
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            {CONF_RECOMMENDED: True, CONF_RECHECK_VOICE: True},
        )
        await hass.async_block_till_done()
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert probe.await_count == 1
    assert mock_config_entry.data[CONF_VOICE_ACCESS]["ok"] is True
    assert CONF_RECHECK_VOICE not in mock_config_entry.options


async def test_prewarm_uses_device_pipeline(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Prewarm fills the TTS cache for the satellite's pipeline, not preferred."""
    _install_pipelines(
        hass, {"sat-1": AssistDevice("assist_satellite", "sat-1")}
    )
    entity = _conversation_entity(hass, mock_config_entry)
    user_input = conversation.ConversationInput(
        text="hello",
        context=Context(),
        conversation_id="c1",
        device_id="sat-1",
        satellite_id=None,
        language="en",
        agent_id=mock_config_entry.entry_id,
    )
    with (
        patch(
            "homeassistant.components.assist_pipeline.select.get_chosen_pipeline",
            return_value="device-pipe",
        ),
        patch(
            "homeassistant.components.tts.media_source.generate_media_source_id",
            return_value="media-id",
        ) as generate,
        patch(
            "homeassistant.components.tts.async_get_media_source_audio",
            new_callable=AsyncMock,
        ) as audio,
    ):
        assert await entity._prewarm_pipeline_tts("Hello there", user_input) is True
    assert generate.call_args.kwargs["engine"] == "tts.satellite"
    audio.assert_awaited()


async def test_unresolved_device_skips_prewarm_for_long_reply(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """A device without a pipeline does not prewarm, and a long reply stays quiet."""
    _install_pipelines(hass, {})
    entity = _conversation_entity(hass, mock_config_entry)
    speech = ("status " * 50).strip() + "?"
    assert len(speech) >= 280
    chat_log = ChatLog(hass, "c1")
    chat_log.content.append(AssistantContent(agent_id="grok", content=speech))
    user_input = conversation.ConversationInput(
        text="what is going on",
        context=Context(),
        conversation_id="c1",
        device_id="missing",
        satellite_id=None,
        language="en",
        agent_id=mock_config_entry.entry_id,
    )
    with patch(
        "homeassistant.components.tts.media_source.generate_media_source_id",
        side_effect=AssertionError("prewarmed"),
    ):
        assert (
            await entity._resolve_continue_conversation(user_input, chat_log, speech)
            is False
        )

    text_input = conversation.ConversationInput(
        text="what is going on",
        context=Context(),
        conversation_id="c1",
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id=mock_config_entry.entry_id,
    )
    with patch.object(
        entity,
        "_prewarm_pipeline_tts",
        side_effect=AssertionError("prewarmed"),
    ):
        assert (
            await entity._resolve_continue_conversation(text_input, chat_log, speech)
            is True
        )
