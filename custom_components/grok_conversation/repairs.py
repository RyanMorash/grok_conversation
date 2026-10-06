"""Repair flow that rechecks xAI Voice access for a chat-only key."""

from __future__ import annotations

from typing import Any

from homeassistant.components.repairs import RepairsFlow
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.exceptions import HomeAssistantError

from .const import DOMAIN


async def async_create_fix_flow(
    hass: HomeAssistant,
    issue_id: str,
    data: dict[str, str | int | float | None] | None,
) -> RepairsFlow:
    """Create the voice recheck flow for a chat-only key."""
    if str(issue_id).startswith("voice_chat_only_"):
        return VoiceRecheckFlow()
    raise HomeAssistantError(f"Unknown repair {issue_id}")


class VoiceRecheckFlow(RepairsFlow):
    """Probe the voices list again and clear the repair when Voice works."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Confirm, then recheck the voices list for this config entry."""
        entry_id = self.data.get("entry_id") if isinstance(self.data, dict) else None
        entry = (
            self.hass.config_entries.async_get_entry(str(entry_id))
            if entry_id
            else None
        )
        if entry is None or entry.domain != DOMAIN:
            return self.async_abort(reason="entry_missing")
        if user_input is None:
            return self.async_show_form(step_id="init")

        from . import async_recheck_voice_access

        voice_ok, _detail = await async_recheck_voice_access(self.hass, entry)
        if voice_ok:
            return self.async_abort(reason="voice_ok")
        return self.async_abort(reason="voice_still_unavailable")
