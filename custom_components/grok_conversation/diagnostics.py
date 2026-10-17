"""Diagnostics support for Grok Conversation.

The config-entry download redacts the API key, prompt, location, and tokens.
"""

from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_ACCESS_TOKEN,
    CONF_API_KEY,
    CONF_API_TOKEN,
    CONF_CLIENT_SECRET,
    CONF_EMAIL,
    CONF_LATITUDE,
    CONF_LOCATION,
    CONF_LONGITUDE,
    CONF_PASSWORD,
    CONF_PROMPT,
    CONF_TOKEN,
    CONF_USERNAME,
)
from homeassistant.core import HomeAssistant

from .const import CONF_LOCATION_CONTEXT

TO_REDACT = {
    CONF_ACCESS_TOKEN,
    CONF_API_KEY,
    CONF_API_TOKEN,
    CONF_CLIENT_SECRET,
    CONF_EMAIL,
    CONF_LATITUDE,
    CONF_LOCATION,
    CONF_LOCATION_CONTEXT,
    CONF_LONGITUDE,
    CONF_PASSWORD,
    CONF_PROMPT,
    CONF_TOKEN,
    CONF_USERNAME,
    "refresh_token",
}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    return async_redact_data(
        {
            "title": entry.title,
            "entry_version": f"{entry.version}.{entry.minor_version}",
            "state": entry.state.value,
            "data": entry.data,
            "options": entry.options,
            "subentries": [
                {
                    "subentry_type": subentry.subentry_type,
                    "title": subentry.title,
                    "data": subentry.data,
                }
                for subentry in entry.subentries.values()
            ],
        },
        TO_REDACT,
    )
