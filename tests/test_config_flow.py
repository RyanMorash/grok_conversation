"""Tests for config flow and AI Task subentry flow."""

from __future__ import annotations

from asyncio import CancelledError
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components import conversation
from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.grok_conversation import async_setup_entry
from custom_components.grok_conversation.api_helpers import (
    XAIAuthError,
    XAIConnectionError,
)
from custom_components.grok_conversation.config_flow import (
    RECOMMENDED_OPTIONS,
    validate_input,
)
from custom_components.grok_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_INTERACTION_MODE,
    CONF_IMAGE_MODEL,
    CONF_RECOMMENDED,
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
    SUBENTRY_CONVERSATION,
    RECOMMENDED_IMAGE_GENERATION_MODEL,
)


async def test_user_flow_creates_ai_task_subentry(
    hass: HomeAssistant, mock_xai_client: MagicMock
) -> None:
    """User config flow creates an entry with a default ai_task_data subentry."""
    assert await async_setup_component(hass, "homeassistant", {})
    await hass.async_block_till_done()

    with (
        patch(
            "custom_components.grok_conversation.config_flow.validate_input",
            return_value={"voice_ok": True, "voice_detail": "ok"},
        ),
        patch(
            "custom_components.grok_conversation.create_xai_client",
            return_value=mock_xai_client,
        ),
        patch(
            "custom_components.grok_conversation.async_validate_voice_access",
            return_value=(True, "ok"),
        ),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": SOURCE_USER}
        )
        assert result["type"] == FlowResultType.FORM

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_API_KEY: "sk-test"}
        )
        await hass.async_block_till_done()

    assert result["type"] == FlowResultType.CREATE_ENTRY
    entry = result["result"]
    kinds = {s.subentry_type for s in entry.subentries.values()}
    assert {"ai_task_data", "conversation", "tts", "stt"} <= kinds
    conversation_sub = next(
        s for s in entry.subentries.values() if s.subentry_type == "conversation"
    )
    assert conversation_sub.title == "xAI Grok"
    assert conversation_sub.data["inherit_entry_options"] is True


async def test_ai_task_subentry_create_and_reconfigure(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """Subentry flow can create and reconfigure an ai_task_data subentry."""
    with (
        patch(
            "custom_components.grok_conversation.config_flow.async_list_chat_models",
            return_value=[RECOMMENDED_CHAT_MODEL],
        ),
        patch(
            "custom_components.grok_conversation.create_xai_client",
            return_value=mock_xai_client,
        ),
        patch(
            "custom_components.grok_conversation.async_validate_voice_access",
            return_value=(True, "ok"),
        ),
    ):
        result = await hass.config_entries.subentries.async_init(
            (mock_config_entry.entry_id, "ai_task_data"),
            context={"source": SOURCE_USER},
        )
        assert result["type"] == FlowResultType.FORM

        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"],
            {
                CONF_NAME: "Custom Task",
                CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
                CONF_IMAGE_MODEL: RECOMMENDED_IMAGE_GENERATION_MODEL,
                CONF_RECOMMENDED: True,
            },
        )
        await hass.async_block_till_done()

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "Custom Task"

    subentry = next(
        s
        for s in mock_config_entry.subentries.values()
        if s.title == "Custom Task"
    )

    with (
        patch(
            "custom_components.grok_conversation.config_flow.async_list_chat_models",
            return_value=[RECOMMENDED_CHAT_MODEL, "grok-4.5"],
        ),
        patch(
            "custom_components.grok_conversation.create_xai_client",
            return_value=mock_xai_client,
        ),
        patch(
            "custom_components.grok_conversation.async_validate_voice_access",
            return_value=(True, "ok"),
        ),
    ):
        result = await hass.config_entries.subentries.async_init(
            (mock_config_entry.entry_id, "ai_task_data"),
            context={
                "source": "reconfigure",
                "subentry_id": subentry.subentry_id,
            },
        )
        assert result["type"] == FlowResultType.FORM

        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"],
            {
                CONF_CHAT_MODEL: "grok-4.5",
                CONF_IMAGE_MODEL: RECOMMENDED_IMAGE_GENERATION_MODEL,
                CONF_RECOMMENDED: True,
            },
        )
        await hass.async_block_till_done()

    assert result["type"] in (FlowResultType.ABORT, FlowResultType.CREATE_ENTRY)
    updated = mock_config_entry.subentries[subentry.subentry_id]
    assert updated.data[CONF_CHAT_MODEL] == "grok-4.5"


async def test_user_flow_invalid_auth(
    hass: HomeAssistant, mock_xai_client: MagicMock
) -> None:
    """UNAUTHENTICATED gRPC errors become invalid_auth."""
    mock_xai_client.models.list_language_models = AsyncMock(
        side_effect=XAIAuthError("bad key")
    )
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-bad"}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["errors"]["base"] == "invalid_auth"


async def test_user_flow_cannot_connect(
    hass: HomeAssistant, mock_xai_client: MagicMock
) -> None:
    """UNAVAILABLE gRPC errors become cannot_connect."""
    mock_xai_client.models.list_language_models = AsyncMock(
        side_effect=XAIConnectionError("down")
    )
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-test"}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["errors"]["base"] == "cannot_connect"


def _grpc_client() -> MagicMock:
    """Return a distinct mocked xAI client with awaitable close()."""
    client = MagicMock()
    client.close = AsyncMock()
    client.models.list_language_models = AsyncMock(return_value=[])
    return client


async def test_setup_closes_runtime_client_on_later_failure(
    hass: HomeAssistant,
) -> None:
    """Probe and runtime gRPC clients are both closed if later setup fails."""
    probe = _grpc_client()
    runtime = _grpc_client()
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="xAI Grok",
        data={CONF_API_KEY: "test-key"},
        options=dict(RECOMMENDED_OPTIONS),
        version=1,
        minor_version=4,
    )
    entry.add_to_hass(hass)
    with (
        patch(
            "custom_components.grok_conversation.create_xai_client",
            side_effect=[probe, runtime],
        ),
        patch("custom_components.grok_conversation.UsageTracker") as tracker_cls,
    ):
        tracker_cls.return_value.async_load = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        assert not await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    assert probe.close.await_count == 1
    assert runtime.close.await_count == 1
    assert getattr(entry, "runtime_data", None) is None


async def test_setup_closes_probe_on_cancellation(hass: HomeAssistant) -> None:
    """Cancelled model-list probe still closes the temporary gRPC channel."""
    probe = _grpc_client()
    probe.models.list_language_models = AsyncMock(side_effect=CancelledError)
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="xAI Grok",
        data={CONF_API_KEY: "test-key"},
        options=dict(RECOMMENDED_OPTIONS),
        version=1,
        minor_version=4,
    )
    with patch(
        "custom_components.grok_conversation.create_xai_client",
        return_value=probe,
    ):
        with pytest.raises(CancelledError):
            await async_setup_entry(hass, entry)
    assert probe.close.await_count == 1
    assert getattr(entry, "runtime_data", None) is None


async def test_validate_input_closes_client_on_cancellation(
    hass: HomeAssistant,
) -> None:
    """Cancelled config-flow validation still closes the probe channel."""
    client = _grpc_client()
    client.models.list_language_models = AsyncMock(side_effect=CancelledError)
    with patch(
        "custom_components.grok_conversation.config_flow.create_xai_client",
        return_value=client,
    ):
        with pytest.raises(CancelledError):
            await validate_input(hass, {CONF_API_KEY: "sk-test"})
    assert client.close.await_count == 1


async def test_migrated_agents_do_not_share_a_device(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Conversation, TTS, and STT each get a device. Sensors stay on the account."""
    dev_reg = dr.async_get(hass)
    ent_reg = er.async_get(hass)
    sensors = [
        entity
        for entity in ent_reg.entities.values()
        if entity.domain == "sensor" and entity.config_entry_id == mock_config_entry.entry_id
    ]
    assert sensors
    account = dev_reg.async_get(sensors[0].device_id)
    assert account is not None
    assert (DOMAIN, mock_config_entry.entry_id) in account.identifiers
    assert account.model == "Grok"
    assert all(entity.device_id == account.id for entity in sensors)
    assert hass.states.get("sensor.xai_grok_estimated_cost") is not None
    assert hass.states.get("sensor.xai_grok_last_model") is not None

    by_type = {
        subentry.subentry_type: subentry
        for subentry in mock_config_entry.subentries.values()
    }
    device_ids = {account.id}
    for platform, kind in (
        ("conversation", "conversation"),
        ("tts", "tts"),
        ("stt", "stt"),
    ):
        subentry = by_type[kind]
        entity_id = ent_reg.async_get_entity_id(platform, DOMAIN, subentry.subentry_id)
        assert entity_id is not None
        entity = ent_reg.async_get(entity_id)
        assert entity is not None
        assert entity.config_subentry_id == subentry.subentry_id
        device = dev_reg.async_get(entity.device_id)
        assert device is not None
        assert (DOMAIN, subentry.subentry_id) in device.identifiers
        assert device.id not in device_ids
        device_ids.add(device.id)
        if platform != "conversation":
            assert entity.original_name == subentry.title

    legacy = conversation.async_get_agent(hass, mock_config_entry.entry_id)
    assert legacy is not None
    assert legacy.unique_id == by_type["conversation"].subentry_id


async def test_second_conversation_agent_uses_its_own_model(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """A second conversation subentry is another agent on the same API key."""
    with patch(
        "custom_components.grok_conversation.config_flow.async_list_chat_models",
        return_value=[RECOMMENDED_CHAT_MODEL, "grok-4.5"],
    ):
        result = await hass.config_entries.subentries.async_init(
            (mock_config_entry.entry_id, SUBENTRY_CONVERSATION),
            context={"source": SOURCE_USER},
        )
        assert result["type"] == FlowResultType.FORM
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"],
            {
                CONF_NAME: "Kitchen Grok",
                CONF_CHAT_MODEL: "grok-4.5",
                CONF_INTERACTION_MODE: "chat_only",
            },
        )
        await hass.async_block_till_done()

    assert result["type"] == FlowResultType.CREATE_ENTRY
    ent_reg = er.async_get(hass)
    agents = [
        entity
        for entity in ent_reg.entities.values()
        if entity.domain == "conversation"
        and entity.config_entry_id == mock_config_entry.entry_id
    ]
    assert len(agents) == 2
    assert any(
        sub.title == "Kitchen Grok" for sub in mock_config_entry.subentries.values()
    )
    kitchen_sub = next(
        sub
        for sub in mock_config_entry.subentries.values()
        if sub.title == "Kitchen Grok"
    )
    assert kitchen_sub.data[CONF_CHAT_MODEL] == "grok-4.5"
    assert kitchen_sub.data["inherit_entry_options"] is False
    kitchen_entity = ent_reg.async_get_entity_id(
        "conversation", DOMAIN, kitchen_sub.subentry_id
    )
    assert kitchen_entity is not None
    device = dr.async_get(hass).async_get(ent_reg.async_get(kitchen_entity).device_id)
    assert device is not None
    assert device.model == "grok-4.5"
    # The original agent remains the one registered for the config entry id.
    legacy = conversation.async_get_agent(hass, mock_config_entry.entry_id)
    original = next(
        sub
        for sub in mock_config_entry.subentries.values()
        if sub.subentry_type == "conversation" and sub.title != "Kitchen Grok"
    )
    assert legacy.unique_id == original.subentry_id
