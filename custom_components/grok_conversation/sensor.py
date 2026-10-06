"""Usage sensors for Grok Conversation."""

from __future__ import annotations

from datetime import datetime

from homeassistant.components.sensor import (
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .usage import UsageTracker


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Grok usage sensors."""
    data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if not data:
        return
    tracker: UsageTracker = data["usage"]
    device = DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name=entry.title or "Grok",
        manufacturer="xAI",
        model="Grok Conversation",
        entry_type=DeviceEntryType.SERVICE,
    )
    async_add_entities(
        [
            GrokUsageSensor(entry, tracker, device, "total_tokens", "Total tokens", "tokens"),
            GrokUsageSensor(entry, tracker, device, "prompt_tokens", "Prompt tokens", "tokens"),
            GrokUsageSensor(
                entry, tracker, device, "completion_tokens", "Completion tokens", "tokens"
            ),
            GrokUsageSensor(entry, tracker, device, "request_count", "API requests", "requests"),
            GrokUsageSensor(
                entry,
                tracker,
                device,
                "estimated_cost_usd",
                "Estimated cost",
                "USD",
                precision=4,
            ),
            GrokLastModelSensor(entry, tracker, device),
        ]
    )


class GrokUsageSensor(SensorEntity):
    """Numeric usage counter."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    # Totals can return to zero when stats are cleared, so last_reset marks the cycle.
    _attr_state_class = SensorStateClass.TOTAL

    def __init__(
        self,
        entry: ConfigEntry,
        tracker: UsageTracker,
        device: DeviceInfo,
        key: str,
        name: str,
        unit: str,
        precision: int | None = None,
    ) -> None:
        self._entry = entry
        self._tracker = tracker
        self._key = key
        self._attr_name = name
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_native_unit_of_measurement = unit
        self._attr_device_info = device
        self._attr_suggested_display_precision = precision
        self._estimated = key == "estimated_cost_usd"
        self._unsub = None

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self._unsub = self._tracker.async_add_listener(self._handle_update)
        self._handle_update()

    async def async_will_remove_from_hass(self) -> None:
        if self._unsub:
            self._unsub()
        await super().async_will_remove_from_hass()

    @callback
    def _handle_update(self) -> None:
        snap = self._tracker.snapshot
        self._attr_native_value = getattr(snap, self._key, 0)
        self._attr_last_reset = _parse_last_reset(snap.last_reset)
        if self._estimated:
            self._attr_extra_state_attributes = {"estimated": True}
        self.async_write_ha_state()


class GrokLastModelSensor(SensorEntity):
    """Last model used."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_name = "Last model"

    def __init__(
        self, entry: ConfigEntry, tracker: UsageTracker, device: DeviceInfo
    ) -> None:
        self._entry = entry
        self._tracker = tracker
        self._attr_unique_id = f"{entry.entry_id}_last_model"
        self._attr_device_info = device
        self._unsub = None

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self._unsub = self._tracker.async_add_listener(self._handle_update)
        self._handle_update()

    async def async_will_remove_from_hass(self) -> None:
        if self._unsub:
            self._unsub()
        await super().async_will_remove_from_hass()

    @callback
    def _handle_update(self) -> None:
        snap = self._tracker.snapshot
        self._attr_native_value = snap.last_model or "unknown"
        # Per-model history stays in storage. The entity keeps the last model only.
        self._attr_extra_state_attributes = {
            "last_request_at": snap.last_request_at,
        }
        self.async_write_ha_state()


def _parse_last_reset(value: str | None) -> datetime | None:
    """Parse the stored reset timestamp for the sensor statistics cycle."""
    if not value:
        return None
    parsed = dt_util.parse_datetime(value)
    if parsed is None:
        return None
    return dt_util.as_utc(parsed)
