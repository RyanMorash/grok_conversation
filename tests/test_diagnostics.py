"""Tests for the config-entry diagnostics download."""

from __future__ import annotations

from http import HTTPStatus
from typing import cast

from syrupy.assertion import SnapshotAssertion

from homeassistant.components.diagnostics import REDACTED
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_ACCESS_TOKEN,
    CONF_API_KEY,
    CONF_API_TOKEN,
    CONF_EMAIL,
    CONF_LATITUDE,
    CONF_LOCATION,
    CONF_LONGITUDE,
    CONF_TOKEN,
    CONF_USERNAME,
)
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from homeassistant.util.json import JsonObjectType
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.typing import ClientSessionGenerator

from custom_components.grok_conversation.config_flow import (
    RECOMMENDED_OPTIONS,
    api_key_unique_id,
)
from custom_components.grok_conversation.const import (
    CONF_LOCATION_CONTEXT,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
)
from custom_components.grok_conversation.diagnostics import (
    async_get_config_entry_diagnostics,
)
from custom_components.grok_conversation.voice_const import CONF_VOICE_ACCESS

API_KEY = "xai-secret-key-do-not-leak"
PROMPT = "You know Ada lives at 1 Secret Lane."
LOCATION = "1 Secret Lane, Springfield"
TOKEN = "tok-personal-access"
ACCESS_TOKEN = "access-should-hide"
API_TOKEN = "api-token-should-hide"
EMAIL = "ada@example.com"
USERNAME = "ada-lovelace"


async def get_diagnostics_for_config_entry(
    hass: HomeAssistant,
    hass_client: ClientSessionGenerator,
    config_entry: ConfigEntry,
) -> JsonObjectType:
    """Return the diagnostics config entry for the specified domain."""
    assert await async_setup_component(hass, "diagnostics", {})
    await hass.async_block_till_done()

    client = await hass_client()
    response = await client.get(
        f"/api/diagnostics/config_entry/{config_entry.entry_id}"
    )
    assert response.status == HTTPStatus.OK
    data = cast(JsonObjectType, await response.json())
    return cast(JsonObjectType, data["data"])


async def test_config_entry_diagnostics(
    hass: HomeAssistant,
    hass_client: ClientSessionGenerator,
    snapshot: SnapshotAssertion,
) -> None:
    """The download redacts the API key, prompt, location, and tokens."""
    # Latitude, email, and token keys are not stored by a normal setup.
    # They are included so the download proves personal data and tokens redact.
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="xAI Grok",
        unique_id=api_key_unique_id(API_KEY),
        data={
            CONF_API_KEY: API_KEY,
            CONF_TOKEN: TOKEN,
            CONF_ACCESS_TOKEN: ACCESS_TOKEN,
            CONF_API_TOKEN: API_TOKEN,
            "refresh_token": "refresh-should-hide",
            CONF_LATITUDE: 42.123,
            CONF_LONGITUDE: -71.456,
            CONF_LOCATION: "42.123,-71.456",
            CONF_VOICE_ACCESS: {"ok": True, "detail": "ok"},
        },
        options={
            **RECOMMENDED_OPTIONS,
            CONF_PROMPT: PROMPT,
            CONF_LOCATION_CONTEXT: LOCATION,
            CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
            CONF_EMAIL: EMAIL,
            CONF_USERNAME: USERNAME,
        },
        version=1,
        minor_version=4,
        subentries_data=[
            {
                "subentry_type": "ai_task_data",
                "title": "Grok AI Task",
                "data": {
                    **RECOMMENDED_AI_TASK_OPTIONS,
                    CONF_PROMPT: "Subentry prompt with a home address",
                },
                "unique_id": None,
            }
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    diagnostics = await get_diagnostics_for_config_entry(hass, hass_client, entry)

    rendered = str(diagnostics)
    for secret in (
        API_KEY,
        PROMPT,
        LOCATION,
        TOKEN,
        ACCESS_TOKEN,
        API_TOKEN,
        "refresh-should-hide",
        EMAIL,
        USERNAME,
        "42.123",
        "-71.456",
        "Subentry prompt with a home address",
        api_key_unique_id(API_KEY),
    ):
        assert secret not in rendered

    assert diagnostics["data"][CONF_API_KEY] == REDACTED
    assert diagnostics["data"][CONF_TOKEN] == REDACTED
    assert diagnostics["data"][CONF_VOICE_ACCESS] == {"ok": True, "detail": "ok"}
    assert diagnostics["options"][CONF_PROMPT] == REDACTED
    assert diagnostics["options"][CONF_LOCATION_CONTEXT] == REDACTED
    assert diagnostics["options"][CONF_EMAIL] == REDACTED
    assert diagnostics["options"]["chat_model"] == RECOMMENDED_CHAT_MODEL
    assert diagnostics["options"][CONF_MAX_TOKENS] == RECOMMENDED_MAX_TOKENS
    assert diagnostics["subentries"][0]["data"][CONF_PROMPT] == REDACTED
    assert diagnostics == snapshot


async def test_empty_location_and_prompt_stay_visible(
    hass: HomeAssistant,
) -> None:
    """Unset location and prompt stay empty so a missing value is still visible."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: API_KEY, CONF_TOKEN: ""},
        options={CONF_PROMPT: "", CONF_LOCATION_CONTEXT: ""},
    )
    entry.add_to_hass(hass)

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    assert diagnostics["data"][CONF_API_KEY] == REDACTED
    assert diagnostics["data"][CONF_TOKEN] == ""
    assert diagnostics["options"][CONF_PROMPT] == ""
    assert diagnostics["options"][CONF_LOCATION_CONTEXT] == ""
    assert API_KEY not in str(diagnostics)
