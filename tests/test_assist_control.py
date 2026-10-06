"""Default Assist control: store the LLM API and drop tool demands in chat-only."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from homeassistant.components import conversation
from homeassistant.components.conversation.chat_log import ChatLog
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.config_entries import SOURCE_USER
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.grok_conversation.config_flow import RECOMMENDED_OPTIONS
from custom_components.grok_conversation.const import (
    CONF_INTERACTION_MODE,
    CONF_PROMPT,
    DOMAIN,
    GROK_CHAT_ONLY_PROMPT,
    GROK_SYSTEM_PROMPT,
    MODE_CHAT_ONLY,
    MODE_PIPELINE,
    MODE_TOOLS,
    RECOMMENDED_AI_TASK_OPTIONS,
    pick_default_llm_api,
    prompt_for_interaction,
)


class _Api:
    def __init__(self, api_id: str, name: str) -> None:
        self.id = api_id
        self.name = name


def test_pick_default_llm_api_prefers_assist_then_homeassistant() -> None:
    """Assist wins over an earlier homeassistant id; otherwise homeassistant, else first."""
    assert pick_default_llm_api([]) is None
    apis = [
        _Api("homeassistant", "Home Assistant"),
        _Api("calendar", "Calendar"),
        _Api("custom", "My Assist"),
    ]
    assert pick_default_llm_api(apis) == "custom"
    assert (
        pick_default_llm_api(
            [_Api("calendar", "Calendar"), _Api("homeassistant", "Home Assistant")]
        )
        == "homeassistant"
    )
    assert pick_default_llm_api([_Api("calendar", "Calendar")]) == "calendar"


def test_prompt_for_interaction_replaces_only_stock_chat_only_prompt() -> None:
    """Chat-only drops the stock tool prompt and leaves edits and other modes."""
    assert prompt_for_interaction(MODE_CHAT_ONLY, GROK_SYSTEM_PROMPT) == (
        GROK_CHAT_ONLY_PROMPT
    )
    assert prompt_for_interaction(MODE_CHAT_ONLY, f"  {GROK_SYSTEM_PROMPT}  ") == (
        GROK_CHAT_ONLY_PROMPT
    )
    assert "Home Assistant tools" not in GROK_CHAT_ONLY_PROMPT
    custom = "Answer in haiku."
    assert prompt_for_interaction(MODE_CHAT_ONLY, custom) == custom
    assert prompt_for_interaction(MODE_TOOLS, GROK_SYSTEM_PROMPT) == GROK_SYSTEM_PROMPT
    assert prompt_for_interaction(MODE_PIPELINE, GROK_SYSTEM_PROMPT) == (
        GROK_SYSTEM_PROMPT
    )
    assert prompt_for_interaction(MODE_CHAT_ONLY, None) is None


def _entry(
    options: dict,
    *,
    minor_version: int = 3,
) -> MockConfigEntry:
    return MockConfigEntry(
        domain=DOMAIN,
        title="xAI Grok",
        data={CONF_API_KEY: "test-key"},
        options=options,
        version=1,
        minor_version=minor_version,
        subentries_data=[
            {
                "subentry_type": "ai_task_data",
                "title": "Grok AI Task",
                "data": dict(RECOMMENDED_AI_TASK_OPTIONS),
                "unique_id": None,
            }
        ],
    )


async def test_user_flow_stores_assist_api(hass: HomeAssistant) -> None:
    """A new entry keeps tool mode and stores the Assist API id."""
    apis = [_Api("calendar", "Calendar"), _Api("assist", "Assist")]
    with (
        patch(
            "custom_components.grok_conversation.config_flow.validate_input",
            return_value={"voice_ok": True, "voice_detail": "ok"},
        ),
        patch(
            "custom_components.grok_conversation.config_flow.llm.async_get_apis",
            return_value=apis,
        ),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_API_KEY: "sk-test"}
        )
        await hass.async_block_till_done()

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["result"].options[CONF_LLM_HASS_API] == ["assist"]
    assert result["result"].options[CONF_INTERACTION_MODE] == MODE_TOOLS


async def test_user_flow_without_apis_leaves_llm_unset(hass: HomeAssistant) -> None:
    """No registered LLM API means the new entry does not invent one."""
    with (
        patch(
            "custom_components.grok_conversation.config_flow.validate_input",
            return_value={"voice_ok": True, "voice_detail": "ok"},
        ),
        patch(
            "custom_components.grok_conversation.config_flow.llm.async_get_apis",
            return_value=[],
        ),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_API_KEY: "sk-test"}
        )
        await hass.async_block_till_done()

    assert CONF_LLM_HASS_API not in result["result"].options


async def test_migration_leaves_other_modes_and_stored_apis(hass: HomeAssistant) -> None:
    """Migration keeps chat-only, pipeline, and an API the user already stored."""
    chat_only = _entry(
        {
            **dict(RECOMMENDED_OPTIONS),
            CONF_INTERACTION_MODE: MODE_CHAT_ONLY,
        }
    )
    pipeline = _entry(
        {
            **dict(RECOMMENDED_OPTIONS),
            CONF_INTERACTION_MODE: MODE_PIPELINE,
        }
    )
    already = _entry(
        {
            **dict(RECOMMENDED_OPTIONS),
            CONF_LLM_HASS_API: ["calendar"],
        }
    )
    for entry in (chat_only, pipeline, already):
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert chat_only.minor_version == 4
    assert pipeline.minor_version == 4
    assert already.minor_version == 4
    assert CONF_LLM_HASS_API not in chat_only.options
    assert CONF_LLM_HASS_API not in pipeline.options
    assert already.options[CONF_LLM_HASS_API] == ["calendar"]


async def test_pre_v4_cleared_tools_entry_is_not_restored(hass: HomeAssistant) -> None:
    """No control on a pre-v4 tools entry has no API and must stay cleared."""
    entry = _entry(
        {
            **dict(RECOMMENDED_OPTIONS),
            CONF_INTERACTION_MODE: MODE_TOOLS,
        },
        minor_version=3,
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.minor_version == 4
    assert CONF_LLM_HASS_API not in entry.options
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    assert not (
        agent.supported_features & conversation.ConversationEntityFeature.CONTROL
    )


async def test_no_control_is_stored_as_explicit_opt_out(hass: HomeAssistant) -> None:
    """Selecting No control keeps an explicit opt-out instead of deleting the key."""
    entry = _entry(
        {
            **dict(RECOMMENDED_OPTIONS),
            CONF_INTERACTION_MODE: MODE_TOOLS,
            CONF_LLM_HASS_API: ["assist"],
        },
        minor_version=4,
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] == FlowResultType.FORM
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "recommended": True,
            CONF_LLM_HASS_API: ["none"],
        },
    )
    await hass.async_block_till_done()

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_LLM_HASS_API] == ["none"]
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    assert not (
        agent.supported_features & conversation.ConversationEntityFeature.CONTROL
    )


async def test_current_tools_entry_without_api_is_not_backfilled(
    hass: HomeAssistant,
) -> None:
    """A v4 tool-mode entry with no API stays that way (user cleared it)."""
    entry = _entry(
        {
            **dict(RECOMMENDED_OPTIONS),
            CONF_INTERACTION_MODE: MODE_TOOLS,
        },
        minor_version=4,
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert CONF_LLM_HASS_API not in entry.options


async def test_tools_entry_exposes_control_feature(hass: HomeAssistant) -> None:
    """A stored Assist API turns the conversation control feature on."""
    entry = _entry(
        {**dict(RECOMMENDED_OPTIONS), CONF_LLM_HASS_API: ["assist"]},
        minor_version=4,
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    assert agent.supported_features & conversation.ConversationEntityFeature.CONTROL


async def test_chat_only_does_not_expose_control_feature(hass: HomeAssistant) -> None:
    """Chat-only stays uncontrolled even if an API id is stored."""
    entry = _entry(
        {
            **dict(RECOMMENDED_OPTIONS),
            CONF_INTERACTION_MODE: MODE_CHAT_ONLY,
            CONF_LLM_HASS_API: ["assist"],
            CONF_PROMPT: GROK_SYSTEM_PROMPT,
        },
        minor_version=4,
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    assert not (
        agent.supported_features & conversation.ConversationEntityFeature.CONTROL
    )


async def _run_turn(hass: HomeAssistant, entry: MockConfigEntry, text: str) -> str:
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    chat_log = ChatLog(hass=hass, conversation_id=entry.entry_id)
    chat_log.async_add_user_content(conversation.UserContent(content=text))
    user_input = conversation.ConversationInput(
        text=text,
        context=Context(),
        conversation_id=entry.entry_id,
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id=agent.entity_id,
    )
    with patch.object(agent, "_async_handle_chat_log", new=AsyncMock()):
        await agent._async_handle_message_inner(user_input, chat_log)  # noqa: SLF001
    return str(chat_log.content[0].content or "")


async def test_chat_only_stock_prompt_is_replaced_at_runtime(
    hass: HomeAssistant,
) -> None:
    """The stored stock prompt stays stored, but chat-only does not send it."""
    entry = _entry(
        {
            **dict(RECOMMENDED_OPTIONS),
            CONF_INTERACTION_MODE: MODE_CHAT_ONLY,
            CONF_PROMPT: GROK_SYSTEM_PROMPT,
        },
        minor_version=4,
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    system = await _run_turn(hass, entry, "Hello")
    assert entry.options[CONF_PROMPT] == GROK_SYSTEM_PROMPT
    assert "ALWAYS use Home Assistant tools" not in system
    assert "Do not call tools" in system


async def test_chat_only_custom_prompt_is_kept(hass: HomeAssistant) -> None:
    """A user-edited chat-only prompt is sent as stored."""
    entry = _entry(
        {
            **dict(RECOMMENDED_OPTIONS),
            CONF_INTERACTION_MODE: MODE_CHAT_ONLY,
            CONF_PROMPT: "Answer in haiku.",
        },
        minor_version=4,
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    system = await _run_turn(hass, entry, "Hello")
    assert "Answer in haiku." in system
    assert "Do not call tools" not in system
