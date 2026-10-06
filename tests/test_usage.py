"""Cost estimates, budget cooldown, and usage sensors."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from homeassistant.components.sensor import SensorStateClass
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.grok_conversation.const import (
    CONF_PROMPT,
    DEFAULT_INPUT_PRICE_PER_M,
    DEFAULT_OUTPUT_PRICE_PER_M,
    DOMAIN,
    RECOMMENDED_IMAGE_GENERATION_MODEL,
    SERVICE_GENERATE_IMAGE,
)
from custom_components.grok_conversation.sensor import GrokUsageSensor
from custom_components.grok_conversation.usage import (
    imagine_estimate_usd,
    token_prices_for_model,
)


def test_usage_sensors_use_total_state_class() -> None:
    """Cleared totals need TOTAL plus last_reset, not TOTAL_INCREASING."""
    # SensorEntity wraps _attr_state_class in a cached property.
    assert GrokUsageSensor.__dict__["_attr_state_class"] is SensorStateClass.TOTAL


def test_model_price_table_and_imagine_estimate() -> None:
    """Prices follow the model id, and Imagine is not billed as zero tokens."""
    assert token_prices_for_model("grok-4.3") == (1.25, 2.50)
    assert token_prices_for_model("grok-4.3-latest") == (1.25, 2.50)
    assert token_prices_for_model("grok-4.6") == (2.0, 6.0)
    assert token_prices_for_model("grok-3-mini") == (0.30, 0.50)
    assert token_prices_for_model("not-a-grok") == (
        DEFAULT_INPUT_PRICE_PER_M,
        DEFAULT_OUTPUT_PRICE_PER_M,
    )
    assert imagine_estimate_usd(RECOMMENDED_IMAGE_GENERATION_MODEL, 1) == 0.02
    assert imagine_estimate_usd("grok-imagine-image-2.0", 2) == 0.08
    assert imagine_estimate_usd("grok-imagine-image-quality") == 0.05
    # The shorter imagine-image prefix must not steal the 2.0 rate.
    assert imagine_estimate_usd("grok-imagine-image") == 0.02


async def test_record_debits_table_price_without_waiting_on_disk(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """The reply updates memory immediately and defers the store write."""
    tracker = hass.data[DOMAIN][mock_config_entry.entry_id]["usage"]
    saves: list[int] = []

    async def _save() -> None:
        saves.append(1)

    tracker.async_save = _save  # type: ignore[method-assign]

    await tracker.async_record(
        model="grok-4.3",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        service="conversation",
    )

    assert saves == []
    assert tracker.snapshot.estimated_cost_usd == 1.25 + 2.50
    assert tracker.snapshot.by_model["grok-4.3"]["estimated_cost_usd"] == 3.75
    assert tracker._dirty is True  # noqa: SLF001

    await tracker.async_flush()
    assert saves == [1]
    assert tracker._dirty is False  # noqa: SLF001


async def test_budget_warning_fires_on_cross_then_cooldown(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """The budget event fires when spend crosses, then only after the cooldown."""
    tracker = hass.data[DOMAIN][mock_config_entry.entry_id]["usage"]
    events: list[float] = []

    hass.bus.async_listen(
        f"{DOMAIN}_budget_warning",
        lambda event: events.append(event.data["estimated_cost_usd"]),
    )

    await tracker.async_record(
        model="grok-4.3",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        budget_warn_usd=1.0,
    )
    await tracker.async_record(
        model="grok-4.3",
        prompt_tokens=100,
        completion_tokens=0,
        budget_warn_usd=1.0,
    )
    await hass.async_block_till_done()
    assert len(events) == 1
    assert events[0] == 1.25

    tracker.snapshot.budget_warned_at = (
        datetime.now(timezone.utc) - timedelta(hours=7)
    ).isoformat()
    await tracker.async_record(
        model="grok-4.3",
        prompt_tokens=100,
        completion_tokens=0,
        budget_warn_usd=1.0,
    )
    await hass.async_block_till_done()
    assert len(events) == 2


async def test_reset_sets_last_reset_and_sensors_hide_model_history(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Reset marks last_reset, and the entity keeps totals plus the last model."""
    tracker = hass.data[DOMAIN][mock_config_entry.entry_id]["usage"]
    await tracker.async_record(
        model="grok-4.6",
        prompt_tokens=10,
        completion_tokens=5,
        service="conversation",
    )
    await hass.async_block_till_done()

    cost = hass.states.get("sensor.xai_grok_estimated_cost")
    assert cost is not None
    assert cost.attributes["estimated"] is True
    assert "by_model" not in cost.attributes

    last_model = hass.states.get("sensor.xai_grok_last_model")
    assert last_model is not None
    assert last_model.state == "grok-4.6"
    assert "by_model" not in last_model.attributes
    assert "by_service" not in last_model.attributes
    assert "grok-4.6" in tracker.snapshot.by_model

    await tracker.async_reset()
    await hass.async_block_till_done()

    cost = hass.states.get("sensor.xai_grok_estimated_cost")
    assert cost is not None
    assert float(cost.state) == 0.0
    assert cost.attributes["estimated"] is True
    assert cost.attributes["last_reset"]
    assert tracker.snapshot.prompt_tokens == 0
    assert tracker.snapshot.by_model == {}
    assert tracker.snapshot.last_reset


async def test_generate_image_records_imagine_estimate(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client,
) -> None:
    """Image generation adds the Imagine price instead of a zero-token call."""
    mock_xai_client.image.sample_batch = AsyncMock(
        return_value=[
            SimpleNamespace(url="https://cdn.example/a.png"),
            SimpleNamespace(url="https://cdn.example/b.png"),
        ]
    )
    tracker = hass.data[DOMAIN][mock_config_entry.entry_id]["usage"]
    await hass.services.async_call(
        DOMAIN,
        SERVICE_GENERATE_IMAGE,
        {
            "config_entry": mock_config_entry.entry_id,
            CONF_PROMPT: "two dogs",
            "n": 2,
            "model": "grok-imagine-image-2.0",
        },
        blocking=True,
        return_response=True,
    )

    assert tracker.snapshot.prompt_tokens == 0
    assert tracker.snapshot.completion_tokens == 0
    assert tracker.snapshot.estimated_cost_usd == 0.08
    assert tracker.snapshot.by_service["generate_image"]["estimated_cost_usd"] == 0.08
    assert tracker.snapshot.last_model == "grok-imagine-image-2.0"
