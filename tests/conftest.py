"""Shared fixtures for Grok Conversation tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.grok_conversation.config_flow import RECOMMENDED_OPTIONS
from custom_components.grok_conversation.const import (
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
)

pytest_plugins = "pytest_homeassistant_custom_component"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations for all tests."""
    return


@pytest.fixture(autouse=True)
async def setup_ha(hass: HomeAssistant) -> None:
    """Set up the homeassistant component (exposed entities, etc.)."""
    assert await async_setup_component(hass, "homeassistant", {})


@pytest.fixture
def mock_xai_client():
    """Mock xai_sdk.AsyncClient used by the integration."""
    client = MagicMock()
    client.close = AsyncMock()
    language_model = MagicMock()
    language_model.name = "grok-4.3-latest"
    language_model.aliases = []
    client.models.list_language_models = AsyncMock(return_value=[language_model])
    chat = MagicMock()
    chat.sample = AsyncMock()
    client.chat.create = MagicMock(return_value=chat)
    client.image.sample = AsyncMock()
    client.image.sample_batch = AsyncMock()
    return client


@pytest.fixture(autouse=True)
def patch_xai_client(mock_xai_client: MagicMock):
    """Keep the gRPC factory mocked for setup, options reload, and teardown."""
    with (
        patch(
            "custom_components.grok_conversation.create_xai_client",
            return_value=mock_xai_client,
        ),
        patch(
            "custom_components.grok_conversation.config_flow.create_xai_client",
            return_value=mock_xai_client,
        ),
        patch(
            "custom_components.grok_conversation.api_helpers.create_xai_client",
            return_value=mock_xai_client,
        ),
        patch(
            "custom_components.grok_conversation.async_validate_voice_access",
            return_value=(True, "ok"),
        ),
    ):
        yield mock_xai_client


@pytest.fixture
def mock_openai_client(mock_xai_client):
    """Backward-compatible alias used by older test names."""
    return mock_xai_client


@pytest.fixture
async def mock_config_entry(
    hass: HomeAssistant,
    mock_xai_client: MagicMock,
) -> MockConfigEntry:
    """Create a loaded config entry with default AI Task subentry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="xAI Grok",
        data={CONF_API_KEY: "test-key"},
        options=dict(RECOMMENDED_OPTIONS),
        version=1,
        minor_version=3,
        subentries_data=[
            {
                "subentry_type": "ai_task_data",
                "title": "Grok AI Task",
                "data": dict(RECOMMENDED_AI_TASK_OPTIONS),
                "unique_id": None,
            }
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    entry.runtime_data = mock_xai_client
    return entry
