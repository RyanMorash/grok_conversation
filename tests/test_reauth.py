"""Reauth when xAI rejects the API key, and unique ids that are not the key."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.config_entries import SOURCE_REAUTH, SOURCE_USER, ConfigEntryState
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.grok_conversation import async_setup_entry
from custom_components.grok_conversation.api_helpers import (
    XAIAuthError,
    XAIConnectionError,
)
from custom_components.grok_conversation.config_flow import (
    RECOMMENDED_OPTIONS,
    api_key_unique_id,
)
from custom_components.grok_conversation.const import DOMAIN


@pytest.fixture(autouse=True)
def _patch_voice_probe():
    """Keep validate_input off the network; the model list mock still runs."""
    with patch(
        "custom_components.grok_conversation.config_flow.async_validate_voice_access",
        return_value=(True, "ok"),
    ):
        yield


def test_api_key_unique_id_is_a_hash() -> None:
    """The unique id is the SHA-256 of the key, not the key itself."""
    key = "sk-secret-value"
    digest = api_key_unique_id(key)
    assert digest == hashlib.sha256(key.encode()).hexdigest()
    assert key not in digest


def _entry(api_key: str, *, unique_id: str | None = None) -> MockConfigEntry:
    return MockConfigEntry(
        domain=DOMAIN,
        title="xAI Grok",
        data={CONF_API_KEY: api_key},
        options=dict(RECOMMENDED_OPTIONS),
        version=1,
        minor_version=4,
        unique_id=unique_id,
    )


async def test_setup_auth_failure_raises_and_starts_reauth(
    hass: HomeAssistant, mock_xai_client
) -> None:
    """A revoked key raises ConfigEntryAuthFailed and opens reauth."""
    mock_xai_client.models.list_language_models = AsyncMock(
        side_effect=XAIAuthError("revoked")
    )
    entry = _entry("sk-old")
    entry.add_to_hass(hass)

    with pytest.raises(ConfigEntryAuthFailed):
        await async_setup_entry(hass, entry)

    assert not await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.SETUP_ERROR
    assert entry.unique_id is None
    assert hass.config_entries.async_get_entry(entry.entry_id) is entry
    flows = hass.config_entries.flow.async_progress()
    assert any(flow["context"]["source"] == SOURCE_REAUTH for flow in flows)


async def test_setup_connection_error_is_not_ready(
    hass: HomeAssistant, mock_xai_client
) -> None:
    """A connection error retries; it does not start reauth."""
    mock_xai_client.models.list_language_models = AsyncMock(
        side_effect=XAIConnectionError("down")
    )
    entry = _entry("sk-old")
    entry.add_to_hass(hass)

    with pytest.raises(ConfigEntryNotReady):
        await async_setup_entry(hass, entry)

    assert not await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.SETUP_RETRY
    flows = hass.config_entries.flow.async_progress()
    assert not any(flow["context"]["source"] == SOURCE_REAUTH for flow in flows)


async def test_second_add_of_same_key_aborts(hass: HomeAssistant) -> None:
    """Adding the same key again aborts, without putting the key in the unique id."""
    first = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    first = await hass.config_entries.flow.async_configure(
        first["flow_id"], {CONF_API_KEY: "sk-same"}
    )
    await hass.async_block_till_done()
    assert first["type"] == FlowResultType.CREATE_ENTRY
    created = first["result"]
    assert created.unique_id == api_key_unique_id("sk-same")
    assert "sk-same" not in created.unique_id

    second = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    second = await hass.config_entries.flow.async_configure(
        second["flow_id"], {CONF_API_KEY: "sk-same"}
    )
    assert second["type"] == FlowResultType.ABORT
    assert second["reason"] == "already_configured"
    assert len(hass.config_entries.async_entries(DOMAIN)) == 1


async def test_reauth_confirm_replaces_key_and_unique_id(
    hass: HomeAssistant,
) -> None:
    """Reauth confirm validates, stores the new key, and updates the hash."""
    entry = _entry("sk-old", unique_id=api_key_unique_id("sk-old"))
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={
            "source": SOURCE_REAUTH,
            "entry_id": entry.entry_id,
            "unique_id": entry.unique_id,
        },
        data=dict(entry.data),
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-new"}
    )
    await hass.async_block_till_done()

    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert entry.data[CONF_API_KEY] == "sk-new"
    assert entry.unique_id == api_key_unique_id("sk-new")
    assert "sk-new" not in entry.unique_id


async def test_reauth_confirm_invalid_key(hass: HomeAssistant, mock_xai_client) -> None:
    """A rejected replacement key stays on the confirm form."""
    entry = _entry("sk-old", unique_id=api_key_unique_id("sk-old"))
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    mock_xai_client.models.list_language_models = AsyncMock(
        side_effect=XAIAuthError("still bad")
    )

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": SOURCE_REAUTH, "entry_id": entry.entry_id},
        data=dict(entry.data),
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-bad"}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["errors"]["base"] == "invalid_auth"
    assert entry.data[CONF_API_KEY] == "sk-old"


async def test_create_rejects_key_stored_on_legacy_entry(hass: HomeAssistant) -> None:
    """A second add of a key matches legacy entries that have no unique id."""
    legacy = _entry("sk-legacy")
    legacy.add_to_hass(hass)
    assert await hass.config_entries.async_setup(legacy.entry_id)
    await hass.async_block_till_done()
    assert legacy.unique_id is None

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-legacy"}
    )
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "already_configured"
    assert legacy.unique_id is None
    assert len(hass.config_entries.async_entries(DOMAIN)) == 1


async def test_reauth_rejects_key_stored_on_legacy_entry(hass: HomeAssistant) -> None:
    """Reauth will not take a key that a unique_id-less entry already stores."""
    legacy = _entry("sk-legacy")
    current = _entry("sk-current", unique_id=api_key_unique_id("sk-current"))
    legacy.add_to_hass(hass)
    assert await hass.config_entries.async_setup(legacy.entry_id)
    current.add_to_hass(hass)
    assert await hass.config_entries.async_setup(current.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": SOURCE_REAUTH, "entry_id": current.entry_id},
        data=dict(current.data),
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-legacy"}
    )
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "already_configured"
    assert current.data[CONF_API_KEY] == "sk-current"
    assert legacy.data[CONF_API_KEY] == "sk-legacy"
    assert legacy.unique_id is None


async def test_reauth_does_not_take_another_entrys_key(hass: HomeAssistant) -> None:
    """Reauth aborts when the new key hash already belongs to another entry."""
    first = _entry("sk-first", unique_id=api_key_unique_id("sk-first"))
    second = _entry("sk-second", unique_id=api_key_unique_id("sk-second"))
    first.add_to_hass(hass)
    assert await hass.config_entries.async_setup(first.entry_id)
    second.add_to_hass(hass)
    assert await hass.config_entries.async_setup(second.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": SOURCE_REAUTH, "entry_id": first.entry_id},
        data=dict(first.data),
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-second"}
    )
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "already_configured"
    assert first.data[CONF_API_KEY] == "sk-first"
    assert second.unique_id == api_key_unique_id("sk-second")
