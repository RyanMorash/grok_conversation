"""Household data stays off unless the user asked, and briefings use Assist."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

from homeassistant.components import conversation
from homeassistant.components.conversation.chat_log import ChatLog
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.const import CONF_API_KEY
from homeassistant.core import Context, HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.grok_conversation.api_helpers import ChatResult
from custom_components.grok_conversation.config_flow import RECOMMENDED_OPTIONS
from custom_components.grok_conversation.const import (
    CONF_HOME_CONTEXT,
    CONF_LOCATION_CONTEXT,
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_HOME_CONTEXT,
)
from custom_components.grok_conversation.conversation import (
    utterance_requests_household,
)
from custom_components.grok_conversation import (
    HOME_BRIEFING_GROUP_CAP,
    collect_home_briefing_lines,
)


def test_new_entries_default_home_context_off() -> None:
    """New recommended options do not turn home context on."""
    assert RECOMMENDED_HOME_CONTEXT is False
    assert RECOMMENDED_OPTIONS[CONF_HOME_CONTEXT] is False


def test_utterance_household_matcher() -> None:
    """Only people, presence, and weather questions match."""
    assert utterance_requests_household("who is home")
    assert utterance_requests_household("what's the weather")
    assert utterance_requests_household("is it raining")
    assert utterance_requests_household("is the family home")
    assert utterance_requests_household("any guests home")
    assert utterance_requests_household("how cold is it outside")
    assert not utterance_requests_household("turn on the lights")
    assert not utterance_requests_household("personal note")
    assert not utterance_requests_household("train the model")
    assert not utterance_requests_household("turn on the family room lights")
    assert not utterance_requests_household("turn off the guest bedroom light")
    assert not utterance_requests_household("add milk to the family shopping list")
    assert not utterance_requests_household("How cold is the freezer?")
    assert not utterance_requests_household("How hot should I set the oven?")
    assert not utterance_requests_household("Forecast our revenue for next year")


def _entry(options: dict) -> MockConfigEntry:
    return MockConfigEntry(
        domain=DOMAIN,
        title="xAI Grok",
        data={CONF_API_KEY: "test-key"},
        options=options,
        version=1,
        minor_version=4,
        subentries_data=[
            {
                "subentry_type": "ai_task_data",
                "title": "Grok AI Task",
                "data": dict(RECOMMENDED_AI_TASK_OPTIONS),
                "unique_id": None,
            }
        ],
    )


def _input(text: str) -> conversation.ConversationInput:
    return conversation.ConversationInput(
        text=text,
        context=Context(),
        conversation_id="conv",
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id="conversation.xai_grok",
    )


async def test_stored_home_context_is_not_turned_off(hass: HomeAssistant) -> None:
    """An entry that already stored home context keeps that value."""
    entry = _entry({**dict(RECOMMENDED_OPTIONS), CONF_HOME_CONTEXT: True})
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.options[CONF_HOME_CONTEXT] is True


async def test_presence_and_weather_follow_the_utterance(
    hass: HomeAssistant,
) -> None:
    """People and weather are sent only for a matching question with context on."""
    hass.states.async_set("person.alex", "home", {"friendly_name": "Alex"})
    hass.states.async_set(
        "weather.local",
        "sunny",
        {"temperature": 20, "temperature_unit": "°C"},
    )
    hass.config.time_zone = "America/Edmonton"

    off = _entry(
        {
            **dict(RECOMMENDED_OPTIONS),
            CONF_HOME_CONTEXT: False,
            CONF_LOCATION_CONTEXT: "Calgary",
        }
    )
    on = _entry({**dict(RECOMMENDED_OPTIONS), CONF_HOME_CONTEXT: True})
    off.add_to_hass(hass)
    assert await hass.config_entries.async_setup(off.entry_id)
    on.add_to_hass(hass)
    assert await hass.config_entries.async_setup(on.entry_id)
    await hass.async_block_till_done()

    off_agent = conversation.async_get_agent(hass, off.entry_id)
    on_agent = conversation.async_get_agent(hass, on.entry_id)
    assert off_agent is not None and on_agent is not None

    quiet = off_agent._build_factual_context(_input("who is home"))  # noqa: SLF001
    assert "Calgary" in quiet
    assert "Person presence" not in quiet
    assert "Weather entity" not in quiet

    lights = on_agent._build_factual_context(_input("turn on the lights"))  # noqa: SLF001
    assert "America/Edmonton" in lights
    assert "Person presence" not in lights
    assert "Weather entity" not in lights

    people = on_agent._build_factual_context(_input("who is home"))  # noqa: SLF001
    assert "Person presence: Alex=home." in people
    assert "weather.local" not in people

    forecast = on_agent._build_factual_context(_input("what's the weather"))  # noqa: SLF001
    assert "weather.local" in forecast
    assert "sunny" in forecast
    assert "Person presence" not in forecast


async def test_utterance_is_logged_at_debug(
    hass: HomeAssistant, caplog
) -> None:
    """The spoken text is a debug log, not an info log."""
    entry = _entry(dict(RECOMMENDED_OPTIONS))
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    chat_log = ChatLog(hass=hass, conversation_id="conv-log")
    chat_log.async_add_user_content(
        conversation.UserContent(content="secret utterance")
    )
    caplog.set_level(logging.DEBUG, logger="custom_components.grok_conversation")
    with patch.object(agent, "_async_handle_chat_log", new=AsyncMock()):
        await agent._async_handle_message_inner(  # noqa: SLF001
            _input("secret utterance"), chat_log
        )

    info_hits = [
        rec
        for rec in caplog.records
        if rec.levelno == logging.INFO and "secret utterance" in rec.message
    ]
    debug_hits = [
        rec
        for rec in caplog.records
        if rec.levelno == logging.DEBUG and "secret utterance" in rec.message
    ]
    assert info_hits == []
    assert debug_hits


def _expose(hass: HomeAssistant, entity_id: str) -> None:
    async_expose_entity(hass, "conversation", entity_id, True)


async def test_briefing_uses_exposed_groups_in_order(hass: HomeAssistant) -> None:
    """Briefing snapshot is exposed alarm, lock, door, climate, then person."""
    hass.states.async_set("person.hidden", "home", {"friendly_name": "Hidden"})
    hass.states.async_set("light.kitchen", "on", {"friendly_name": "Kitchen"})
    hass.states.async_set("person.alex", "home", {"friendly_name": "Alex"})
    hass.states.async_set("climate.hall", "heat", {"friendly_name": "Hall"})
    hass.states.async_set(
        "binary_sensor.front_door",
        "off",
        {"device_class": "door", "friendly_name": "Front door"},
    )
    hass.states.async_set("lock.front", "locked", {"friendly_name": "Front lock"})
    hass.states.async_set(
        "alarm_control_panel.home",
        "armed_away",
        {"friendly_name": "Alarm"},
    )
    for entity_id in (
        "person.alex",
        "climate.hall",
        "binary_sensor.front_door",
        "lock.front",
        "alarm_control_panel.home",
        "light.kitchen",
    ):
        _expose(hass, entity_id)

    lines = collect_home_briefing_lines(hass, domains=None)
    text = "\n".join(lines)
    assert "light.kitchen" not in text
    assert "person.hidden" not in text
    order = [
        text.index("alarm_control_panel.home"),
        text.index("lock.front"),
        text.index("binary_sensor.front_door"),
        text.index("climate.hall"),
        text.index("person.alex"),
    ]
    assert order == sorted(order)

    for index in range(HOME_BRIEFING_GROUP_CAP + 1):
        entity_id = f"lock.extra_{index:02d}"
        hass.states.async_set(entity_id, "locked")
        _expose(hass, entity_id)
    capped = collect_home_briefing_lines(hass, per_group_cap=2, max_entities=80)
    lock_lines = [line for line in capped if "(lock." in line]
    assert len(lock_lines) == 2


async def test_briefing_filters_each_entity_domain(hass: HomeAssistant) -> None:
    """A domain filter uses the entity domain, not every domain in the group."""
    hass.states.async_set(
        "binary_sensor.front_door",
        "off",
        {"device_class": "door", "friendly_name": "Front door"},
    )
    hass.states.async_set(
        "binary_sensor.garage_door",
        "on",
        {"device_class": "garage_door", "friendly_name": "Garage door sensor"},
    )
    hass.states.async_set(
        "cover.garage",
        "closed",
        {"device_class": "garage", "friendly_name": "Garage door"},
    )
    hass.states.async_set(
        "cover.front",
        "closed",
        {"device_class": "door", "friendly_name": "Front door cover"},
    )
    for entity_id in (
        "binary_sensor.front_door",
        "binary_sensor.garage_door",
        "cover.garage",
        "cover.front",
    ):
        _expose(hass, entity_id)

    binary = "\n".join(
        collect_home_briefing_lines(hass, domains={"binary_sensor"})
    )
    assert "binary_sensor.front_door" in binary
    assert "binary_sensor.garage_door" in binary
    assert "cover.garage" not in binary
    assert "cover.front" not in binary

    covers = "\n".join(collect_home_briefing_lines(hass, domains={"cover"}))
    assert "cover.garage" in covers
    assert "cover.front" in covers
    assert "binary_sensor.front_door" not in covers
    assert "binary_sensor.garage_door" not in covers


async def test_home_briefing_service_sends_exposed_snapshot(
    hass: HomeAssistant,
) -> None:
    """The home_briefing service sends the exposed snapshot, not every state."""
    entry = _entry(dict(RECOMMENDED_OPTIONS))
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    hass.states.async_set("light.kitchen", "on", {"friendly_name": "Kitchen"})
    hass.states.async_set("lock.front", "locked", {"friendly_name": "Front lock"})
    _expose(hass, "light.kitchen")
    _expose(hass, "lock.front")

    completion = AsyncMock(return_value=ChatResult(content="Front lock is locked."))
    with patch(
        "custom_components.grok_conversation.async_chat_completion",
        completion,
    ):
        result = await hass.services.async_call(
            DOMAIN,
            "home_briefing",
            {"config_entry": entry.entry_id},
            blocking=True,
            return_response=True,
        )

    payload = completion.await_args.kwargs["messages"][1]["content"]
    assert "lock.front" in payload
    assert "light.kitchen" not in payload
    assert result["entities_included"] == 1
    assert result["response_text"] == "Front lock is locked."
