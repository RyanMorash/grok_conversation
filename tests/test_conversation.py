"""Conversation regression tests for the shared LLM tool loop (#34)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import probatio
import pytest
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.components.conversation.chat_log import ChatLog
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry
from xai_sdk.proto import chat_pb2

from custom_components.grok_conversation.api_helpers import XAIError
from custom_components.grok_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_FALLBACK_MODEL,
    DOMAIN,
)
from custom_components.grok_conversation.entity import format_tool


def _chat_response(text: str = "", *, tool_calls=None, model: str | None = None):
    """Build a minimal xai-sdk chat.sample-like response."""
    result = MagicMock()
    result.content = text
    result.tool_calls = tool_calls or []
    result.finish_reason = "REASON_STOP"
    result.usage.prompt_tokens = 11
    result.usage.completion_tokens = 7
    result.citations = []
    result.model = model
    return result


def _tool_call(name: str, arguments: str, call_id: str = "call_1"):
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _message_role(msg) -> str:
    return chat_pb2.MessageRole.Name(msg.role).removeprefix("ROLE_").lower()


def _message_text(msg) -> str:
    return "".join(part.text for part in msg.content)


def _set_sample(client: MagicMock, side_effect) -> None:
    client.chat.create.return_value.sample = AsyncMock(side_effect=side_effect)


class _RecordingTool(llm.Tool):
    """Test tool that records calls."""

    name = "test_light"
    description = "Control a test light"
    parameters = vol.Schema({vol.Required("action"): str}, extra=vol.ALLOW_EXTRA)

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        self.calls.append(dict(tool_input.tool_args))
        return {"ok": True, "action": tool_input.tool_args.get("action")}


class _UnconvertibleTool(llm.Tool):
    """Tool whose parameter schema cannot be converted to OpenAPI."""

    name = "broken_schema"
    description = "Schema cannot be converted"

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        return {"ok": False}


def _cyclic_schema() -> dict[str, Any]:
    """Return a schema that probatio.to_openapi refuses to render."""
    cycle: dict[str, Any] = {}
    cycle["child"] = cycle
    return cycle


def _mock_llm_api(hass: HomeAssistant, *tools: llm.Tool) -> llm.APIInstance:
    api = MagicMock(spec=llm.API)
    api.hass = hass
    api.id = "test"
    api.name = "Test"
    return llm.APIInstance(
        api=api,
        api_prompt="Test tools available.",
        llm_context=llm.LLMContext(
            platform=DOMAIN,
            context=Context(),
            language="en",
            assistant="conversation",
            device_id=None,
        ),
        tools=list(tools),
    )


def _conversation_entity(hass: HomeAssistant, entry: MockConfigEntry):
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    return agent


async def test_conversation_tool_loop_executes_tool(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """Tool calls from the model execute through the shared chat-log loop."""
    tool = _RecordingTool()
    entity = _conversation_entity(hass, mock_config_entry)

    _set_sample(
        mock_xai_client,
        [
            _chat_response(
                tool_calls=[_tool_call("test_light", '{"action":"turn_on"}')]
            ),
            _chat_response("The light is on."),
        ],
    )

    chat_log = ChatLog(hass=hass, conversation_id="conv-tools")
    chat_log.llm_api = _mock_llm_api(hass, tool)
    chat_log.async_add_user_content(
        conversation.UserContent(content="Turn on the light")
    )

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Turn on the light"},
    ]
    await entity._async_handle_chat_log(  # noqa: SLF001
        chat_log,
        model="grok-4.3-latest",
        options={
            CONF_CHAT_MODEL: "grok-4.3-latest",
            CONF_FALLBACK_MODEL: "grok-4.6",
        },
        messages=messages,
        agent_id=entity.entity_id,
        service="conversation",
        fallback_model="grok-4.6",
    )

    assert tool.calls == [{"action": "turn_on"}]
    assert mock_xai_client.chat.create.call_count == 2
    assert any(
        isinstance(c, conversation.AssistantContent) and c.content == "The light is on."
        for c in chat_log.content
    )


async def test_conversation_fallback_model_on_primary_error(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """Primary XAIError falls through to the fallback model."""
    entity = _conversation_entity(hass, mock_config_entry)

    _set_sample(
        mock_xai_client,
        [
            XAIError("primary down"),
            _chat_response("Fallback answered."),
        ],
    )

    chat_log = ChatLog(hass=hass, conversation_id="conv-fallback")
    chat_log.async_add_user_content(conversation.UserContent(content="Hi"))

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    await entity._async_handle_chat_log(  # noqa: SLF001
        chat_log,
        model="grok-4.3-latest",
        options={
            CONF_CHAT_MODEL: "grok-4.3-latest",
            CONF_FALLBACK_MODEL: "grok-4.6",
        },
        messages=messages,
        agent_id=entity.entity_id,
        service="conversation",
        fallback_model="grok-4.6",
    )

    assert mock_xai_client.chat.create.call_count == 2
    first_kwargs = mock_xai_client.chat.create.call_args_list[0].kwargs
    second_kwargs = mock_xai_client.chat.create.call_args_list[1].kwargs
    assert first_kwargs["model"] == "grok-4.3-latest"
    assert second_kwargs["model"] == "grok-4.6"
    assert any(
        isinstance(c, conversation.AssistantContent)
        and c.content == "Fallback answered."
        for c in chat_log.content
    )


async def test_rate_limit_does_not_use_fallback(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """RESOURCE_EXHAUSTED / rate-limit errors skip fallback and fail closed."""
    grpc = pytest.importorskip("grpc")

    class _Rpc(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.RESOURCE_EXHAUSTED

        def details(self):
            return "quota"

    entity = _conversation_entity(hass, mock_config_entry)
    _set_sample(mock_xai_client, _Rpc())

    chat_log = ChatLog(hass=hass, conversation_id="conv-rate-limit")
    chat_log.async_add_user_content(conversation.UserContent(content="Hi"))
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    with pytest.raises(HomeAssistantError) as caught:
        await entity._async_handle_chat_log(  # noqa: SLF001
            chat_log,
            model="grok-4.3-latest",
            options={
                CONF_CHAT_MODEL: "grok-4.3-latest",
                CONF_FALLBACK_MODEL: "grok-4.6",
            },
            messages=messages,
            agent_id=entity.entity_id,
            service="conversation",
            fallback_model="grok-4.6",
        )
    assert caught.value.translation_key == "rate_limit"
    assert "quota" not in str(caught.value)

    assert mock_xai_client.chat.create.call_count == 1


async def test_conversation_fallback_does_not_inherit_partial_tools(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """Fallback request must not include the failed model's partial tool turns."""
    tool = _RecordingTool()
    entity = _conversation_entity(hass, mock_config_entry)

    _set_sample(
        mock_xai_client,
        [
            _chat_response(
                tool_calls=[_tool_call("test_light", '{"action":"turn_on"}')]
            ),
            XAIError("boom after tool"),
            _chat_response("Recovered without prior tools."),
        ],
    )

    chat_log = ChatLog(hass=hass, conversation_id="conv-partial")
    chat_log.llm_api = _mock_llm_api(hass, tool)
    chat_log.async_add_user_content(
        conversation.UserContent(content="Turn on the light")
    )

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Turn on the light"},
    ]
    await entity._async_handle_chat_log(  # noqa: SLF001
        chat_log,
        model="grok-4.3-latest",
        options={
            CONF_CHAT_MODEL: "grok-4.3-latest",
            CONF_FALLBACK_MODEL: "grok-4.6",
        },
        messages=messages,
        agent_id=entity.entity_id,
        service="conversation",
        fallback_model="grok-4.6",
    )

    assert tool.calls == [{"action": "turn_on"}]
    fallback_kwargs = mock_xai_client.chat.create.call_args_list[2].kwargs
    assert fallback_kwargs["model"] == "grok-4.6"
    roles = [_message_role(m) for m in fallback_kwargs["messages"]]
    assert "tool" not in roles
    assert roles.count("assistant") == 0 or all(
        not list(m.tool_calls)
        for m in fallback_kwargs["messages"]
        if _message_role(m) == "assistant"
    )
    assert roles == ["system", "user"]


async def test_invalid_tool_argument_json_returns_error(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
) -> None:
    """Malformed tool-argument JSON is returned as a tool error, not executed."""
    tool = _RecordingTool()
    entity = _conversation_entity(hass, mock_config_entry)

    _set_sample(
        mock_xai_client,
        [
            _chat_response(tool_calls=[_tool_call("test_light", "{not-json")]),
            _chat_response("I could not parse that tool call."),
        ],
    )

    chat_log = ChatLog(hass=hass, conversation_id="conv-bad-json")
    chat_log.llm_api = _mock_llm_api(hass, tool)
    chat_log.async_add_user_content(conversation.UserContent(content="Toggle light"))

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Toggle light"},
    ]
    await entity._async_handle_chat_log(  # noqa: SLF001
        chat_log,
        model="grok-4.3-latest",
        options={CONF_CHAT_MODEL: "grok-4.3-latest"},
        messages=messages,
        agent_id=entity.entity_id,
        service="conversation",
        fallback_model="",
    )

    assert tool.calls == []
    second_kwargs = mock_xai_client.chat.create.call_args_list[1].kwargs
    tool_msgs = [m for m in second_kwargs["messages"] if _message_role(m) == "tool"]
    assert tool_msgs
    assert "Invalid tool arguments JSON" in _message_text(tool_msgs[0])


def test_format_tool_reraises_when_schema_does_not_convert(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A schema that cannot be converted is not replaced with an empty object."""
    tool = _UnconvertibleTool()
    tool.parameters = _cyclic_schema()

    with (
        caplog.at_level("WARNING", logger="custom_components.grok_conversation"),
        pytest.raises(probatio.SchemaError, match="references itself"),
    ):
        format_tool(tool, None)

    assert "broken_schema" in caplog.text
    assert "empty object schema" not in caplog.text


def test_format_tool_keeps_converted_parameters() -> None:
    """A convertible schema is still sent with its parameters."""
    formatted = format_tool(_RecordingTool(), None)
    parameters = formatted["function"]["parameters"]
    assert parameters["properties"]["action"]["type"] == "string"
    assert parameters["properties"] != {}


async def test_unconvertible_tool_schema_is_omitted_from_request(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tool whose schema fails conversion is left out; the rest are sent."""
    good = _RecordingTool()
    broken = _UnconvertibleTool()
    broken.parameters = _cyclic_schema()
    entity = _conversation_entity(hass, mock_config_entry)

    _set_sample(mock_xai_client, [_chat_response("Done.")])

    chat_log = ChatLog(hass=hass, conversation_id="conv-skip-schema")
    chat_log.llm_api = _mock_llm_api(hass, broken, good)
    chat_log.async_add_user_content(
        conversation.UserContent(content="Turn on the light")
    )

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Turn on the light"},
    ]
    with caplog.at_level("WARNING", logger="custom_components.grok_conversation"):
        await entity._async_handle_chat_log(  # noqa: SLF001
            chat_log,
            model="grok-4.3-latest",
            options={CONF_CHAT_MODEL: "grok-4.3-latest"},
            messages=messages,
            agent_id=entity.entity_id,
            service="conversation",
            fallback_model="",
        )

    sent = mock_xai_client.chat.create.call_args_list[0].kwargs["tools"]
    names = [item.function.name for item in sent]
    assert names == ["test_light"]
    parameters = json.loads(sent[0].function.parameters)
    assert parameters["properties"]["action"]["type"] == "string"
    assert parameters != {"type": "object", "properties": {}}
    assert "Skipping tool broken_schema" in caplog.text
