"""Token usage tracking for Grok Conversation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.storage import Store

from .const import (
    DEFAULT_INPUT_PRICE_PER_M,
    DEFAULT_OUTPUT_PRICE_PER_M,
    DOMAIN,
    EVENT_USAGE_UPDATED,
    LOGGER,
)

STORAGE_KEY = f"{DOMAIN}.usage"
STORAGE_VERSION = 1

# Reply path must not wait on disk. Listeners still see the in-memory totals.
USAGE_SAVE_DELAY_SECONDS = 15.0
# Re-fire the budget event while spend stays over the threshold, but not every turn.
BUDGET_WARNING_COOLDOWN = timedelta(hours=6)

# Prompt size at which xAI bills the long-context rate for every token.
# https://docs.x.ai/developers/pricing — estimates, not an invoice.
LONG_CONTEXT_PROMPT_TOKENS = 200_000

# Standard list rates below the long-context tier, USD per 1M tokens.
# Models in _LONG_CONTEXT_PREFIXES double both rates at that tier.
_MODEL_TOKEN_PRICES: tuple[tuple[str, float, float], ...] = (
    ("grok-4.7", 2.0, 6.0),
    ("grok-4.6", 2.0, 6.0),
    ("grok-4.5", 2.0, 6.0),
    ("grok-4.3", 1.25, 2.50),
    ("grok-4.20", 1.25, 2.50),
    ("grok-build-0.1", 1.0, 2.0),
    ("grok-4", 3.0, 15.0),
    ("grok-3-mini", 0.30, 0.50),
    ("grok-3", 3.0, 15.0),
    ("grok-2", 2.0, 10.0),
)

# Models whose published table has a doubled rate at LONG_CONTEXT_PROMPT_TOKENS.
_LONG_CONTEXT_PREFIXES: tuple[str, ...] = (
    "grok-4.7",
    "grok-4.6",
    "grok-4.5",
    "grok-4.3",
    "grok-4.20",
    "grok-build-0.1",
)

# Flat USD per image. grok-imagine-image-2.0 is priced separately by tier.
_IMAGINE_PRICES: tuple[tuple[str, float], ...] = (
    ("grok-imagine-image-quality", 0.05),
    ("grok-imagine-image", 0.02),
)
DEFAULT_IMAGINE_USD = 0.02

# grok-imagine-image-2.0: 1k/low $0.04, 1k/medium $0.06, 2k/low $0.06, 2k/medium $0.08.
# Omitted resolution is 1k. Omitted quality is auto, which bills generation at low.
_IMAGINE_2_0_PREFIX = "grok-imagine-image-2.0"
_IMAGINE_2_0_PRICES: dict[tuple[str, str], float] = {
    ("1k", "low"): 0.04,
    ("1k", "medium"): 0.06,
    ("2k", "low"): 0.06,
    ("2k", "medium"): 0.08,
}


def _matches_model_prefix(model_id: str, prefix: str) -> bool:
    """Match an id or a dated / latest alias without crossing sibling ids."""
    return model_id == prefix or model_id.startswith(f"{prefix}-")


def token_prices_for_model(
    model: str, prompt_tokens: int = 0
) -> tuple[float, float]:
    """Return (input, output) USD per 1M tokens for a model id.

    A prompt at or above the long-context tier uses that model's doubled rate
    for every input and output token in the request.
    """
    mid = (model or "").strip().lower()
    input_price = DEFAULT_INPUT_PRICE_PER_M
    output_price = DEFAULT_OUTPUT_PRICE_PER_M
    for prefix, listed_input, listed_output in sorted(
        _MODEL_TOKEN_PRICES, key=lambda row: len(row[0]), reverse=True
    ):
        if _matches_model_prefix(mid, prefix):
            input_price = listed_input
            output_price = listed_output
            break
    long_context = int(prompt_tokens or 0) >= LONG_CONTEXT_PROMPT_TOKENS and any(
        _matches_model_prefix(mid, prefix) for prefix in _LONG_CONTEXT_PREFIXES
    )
    if long_context:
        return input_price * 2, output_price * 2
    return input_price, output_price


def imagine_estimate_usd(
    model: str,
    count: int = 1,
    *,
    resolution: str | None = None,
    quality: str | None = None,
) -> float:
    """Estimated USD for ``count`` Imagine images of ``model``.

    ``grok-imagine-image-2.0`` follows resolution and quality. Other Imagine
    models stay on their flat per-image rate.
    """
    mid = (model or "").strip().lower()
    images = max(int(count or 1), 1)
    if _matches_model_prefix(mid, _IMAGINE_2_0_PREFIX):
        tier_resolution = (resolution or "1k").strip().lower()
        tier_quality = (quality or "low").strip().lower()
        if tier_quality == "auto":
            tier_quality = "low"
        price = _IMAGINE_2_0_PRICES.get(
            (tier_resolution, tier_quality), _IMAGINE_2_0_PRICES[("1k", "low")]
        )
        return round(price * images, 6)
    for prefix, price in sorted(
        _IMAGINE_PRICES, key=lambda row: len(row[0]), reverse=True
    ):
        if _matches_model_prefix(mid, prefix):
            return round(price * images, 6)
    return round(DEFAULT_IMAGINE_USD * images, 6)


@dataclass
class UsageSnapshot:
    """Aggregated usage counters."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    estimated_cost_usd: float = 0.0
    last_model: str = ""
    last_request_at: str | None = None
    last_reset: str | None = None
    budget_warned_at: str | None = None
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_service: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "last_model": self.last_model,
            "last_request_at": self.last_request_at,
            "last_reset": self.last_reset,
            "budget_warned_at": self.budget_warned_at,
            "by_model": self.by_model,
            "by_service": self.by_service,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> UsageSnapshot:
        """Deserialize."""
        if not data:
            return cls()
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            request_count=int(data.get("request_count", 0)),
            estimated_cost_usd=float(data.get("estimated_cost_usd", 0.0)),
            last_model=str(data.get("last_model", "")),
            last_request_at=data.get("last_request_at"),
            last_reset=data.get("last_reset"),
            budget_warned_at=data.get("budget_warned_at"),
            by_model=dict(data.get("by_model") or {}),
            by_service=dict(data.get("by_service") or {}),
        )


class UsageTracker:
    """Persist and expose token usage statistics."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        self.hass = hass
        self.entry_id = entry_id
        self._store = Store(hass, STORAGE_VERSION, f"{STORAGE_KEY}_{entry_id}")
        self.snapshot = UsageSnapshot()
        self._listeners: list[callback] = []
        self._unsub_save: Callable[[], None] | None = None
        self._dirty = False
        self._revision = 0
        self._saved_revision = 0
        self._save_lock = asyncio.Lock()

    async def async_load(self) -> None:
        """Load from disk."""
        data = await self._store.async_load()
        self.snapshot = UsageSnapshot.from_dict(data)
        LOGGER.debug("Loaded usage stats for %s: %s", self.entry_id, self.snapshot)

    async def async_save(self) -> None:
        """Persist to disk."""
        await self._store.async_save(self.snapshot.to_dict())

    def _mark_dirty(self) -> None:
        """Remember that memory is ahead of the last successful write."""
        self._revision += 1
        self._dirty = True

    def _schedule_save(self) -> None:
        """Persist soon, without blocking the reply that recorded usage."""
        self._mark_dirty()
        if self._unsub_save is not None:
            return
        self._unsub_save = async_call_later(
            self.hass, USAGE_SAVE_DELAY_SECONDS, self._async_debounced_save
        )

    async def _async_debounced_save(self, _now: datetime) -> None:
        """Write the snapshot after the debounce delay."""
        self._unsub_save = None
        await self._save_dirty()

    async def _save_dirty(self) -> None:
        """Write until the snapshot that finished last is on disk.

        Dirty stays set until that write returns. A flush during the write
        waits for it, then writes again if a newer record arrived.
        """
        async with self._save_lock:
            while self._saved_revision != self._revision:
                revision = self._revision
                await self.async_save()
                if self._revision == revision:
                    self._saved_revision = revision
                    self._dirty = False
                else:
                    self._saved_revision = revision

    async def async_flush(self) -> None:
        """Cancel a pending debounce and wait for the snapshot to hit disk."""
        if self._unsub_save is not None:
            self._unsub_save()
            self._unsub_save = None
        await self._save_dirty()

    def _notify(self) -> None:
        """Publish the in-memory snapshot to listeners and the event bus."""
        self.hass.bus.async_fire(
            EVENT_USAGE_UPDATED,
            {"entry_id": self.entry_id, **self.snapshot.to_dict()},
        )
        for listener in list(self._listeners):
            listener()

    def _maybe_budget_warning(self, previous_cost: float, budget: float | None) -> None:
        """Fire once when spend crosses the threshold, then on a cooldown."""
        if budget is None:
            return
        limit = float(budget or 0)
        if limit <= 0:
            return
        cost = self.snapshot.estimated_cost_usd
        if cost < limit:
            return
        crossed = previous_cost < limit <= cost
        now = datetime.now(timezone.utc)
        last: datetime | None = None
        if self.snapshot.budget_warned_at:
            parsed = datetime.fromisoformat(self.snapshot.budget_warned_at)
            last = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        cooled = last is not None and (now - last) >= BUDGET_WARNING_COOLDOWN
        if not crossed and not cooled:
            return
        self.snapshot.budget_warned_at = now.isoformat()
        LOGGER.warning(
            "Grok estimated spend $%.4f exceeded budget warn $%.2f",
            cost,
            limit,
        )
        self.hass.bus.async_fire(
            f"{DOMAIN}_budget_warning",
            {
                "entry_id": self.entry_id,
                "estimated_cost_usd": cost,
                "budget_warn_usd": limit,
            },
        )

    def async_add_listener(self, listener: callback) -> callback:
        """Register a listener fired after each record."""
        self._listeners.append(listener)

        def _remove() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return _remove

    async def async_record(
        self,
        *,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        service: str = "conversation",
        input_price_per_m: float | None = None,
        output_price_per_m: float | None = None,
        extra_cost_usd: float = 0.0,
        budget_warn_usd: float | None = None,
    ) -> None:
        """Record a completed API call.

        Token prices come from the model table unless a caller overrides them.
        ``extra_cost_usd`` is the Imagine estimate (image calls have no tokens).
        The store write is debounced so this returns before disk I/O.
        """
        prompt_tokens = max(int(prompt_tokens or 0), 0)
        completion_tokens = max(int(completion_tokens or 0), 0)
        if input_price_per_m is None or output_price_per_m is None:
            table_input, table_output = token_prices_for_model(model, prompt_tokens)
            if input_price_per_m is None:
                input_price_per_m = table_input
            if output_price_per_m is None:
                output_price_per_m = table_output
        total = prompt_tokens + completion_tokens
        cost = (prompt_tokens / 1_000_000) * input_price_per_m + (
            completion_tokens / 1_000_000
        ) * output_price_per_m
        cost += max(float(extra_cost_usd or 0), 0.0)

        snap = self.snapshot
        previous_cost = snap.estimated_cost_usd
        snap.prompt_tokens += prompt_tokens
        snap.completion_tokens += completion_tokens
        snap.total_tokens += total
        snap.request_count += 1
        snap.estimated_cost_usd += cost
        snap.last_model = model or snap.last_model
        snap.last_request_at = datetime.now(timezone.utc).isoformat()

        model_key = model or "unknown"
        model_bucket = snap.by_model.setdefault(
            model_key,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "request_count": 0,
                "estimated_cost_usd": 0.0,
            },
        )
        model_bucket["prompt_tokens"] += prompt_tokens
        model_bucket["completion_tokens"] += completion_tokens
        model_bucket["total_tokens"] += total
        model_bucket["request_count"] += 1
        model_bucket["estimated_cost_usd"] = round(
            float(model_bucket["estimated_cost_usd"]) + cost, 6
        )

        svc_bucket = snap.by_service.setdefault(
            service,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "request_count": 0,
                "estimated_cost_usd": 0.0,
            },
        )
        svc_bucket["prompt_tokens"] += prompt_tokens
        svc_bucket["completion_tokens"] += completion_tokens
        svc_bucket["total_tokens"] += total
        svc_bucket["request_count"] += 1
        svc_bucket["estimated_cost_usd"] = round(
            float(svc_bucket["estimated_cost_usd"]) + cost, 6
        )

        self._maybe_budget_warning(previous_cost, budget_warn_usd)
        self._schedule_save()
        self._notify()

    async def async_reset(self) -> None:
        """Zero all counters and mark the statistics cycle."""
        self.snapshot = UsageSnapshot(
            last_reset=datetime.now(timezone.utc).isoformat()
        )
        if self._unsub_save is not None:
            self._unsub_save()
            self._unsub_save = None
        async with self._save_lock:
            self._revision += 1
            revision = self._revision
            await self.async_save()
            self._saved_revision = revision
            self._dirty = False
        self._notify()
