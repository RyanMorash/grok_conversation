"""Tests for AI Task platform."""

from __future__ import annotations

import base64
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import voluptuous as vol
from homeassistant.components import ai_task, conversation
from homeassistant.components.ai_task.const import DATA_COMPONENT
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er, llm, selector
from homeassistant.helpers.chat_session import async_get_chat_session
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

try:
    import probatio
except ImportError:  # pragma: no cover
    probatio = None

from xai_sdk.proto import chat_pb2

from custom_components.grok_conversation.ai_task import imagine_edit_kwargs
from custom_components.grok_conversation.api_helpers import XAIError
from custom_components.grok_conversation.config_flow import RECOMMENDED_OPTIONS
from custom_components.grok_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_IMAGE_MODEL,
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_IMAGE_GENERATION_MODEL,
    RECOMMENDED_VISION_MODEL,
)
from custom_components.grok_conversation.entity import (
    MAX_TOOL_ITERATIONS,
    GrokBaseLLMEntity,
)


def _structure_schema(fields: dict):
    """Build a structure schema compatible with the installed HA version."""
    if probatio is not None:
        mapped = {}
        for key, value in fields.items():
            if isinstance(key, vol.Required):
                mapped[probatio.Required(key.schema)] = value
            elif isinstance(key, vol.Optional):
                mapped[probatio.Optional(key.schema)] = value
            else:
                mapped[key] = value
        return probatio.Schema(mapped)
    return vol.Schema(fields)


def _chat_response(text: str, *, tool_calls=None):
    """Build a minimal xai-sdk chat.sample-like response."""
    result = MagicMock()
    result.content = text
    result.tool_calls = tool_calls
    result.finish_reason = "REASON_STOP"
    result.usage.prompt_tokens = 10
    result.usage.completion_tokens = 5
    result.citations = []
    return result


def _set_sample(client: MagicMock, value) -> None:
    client.chat.create.return_value.sample = AsyncMock(return_value=value)


@pytest.fixture
def ai_task_entity_id(hass: HomeAssistant, mock_config_entry: MockConfigEntry) -> str:
    """Return the AI Task entity id created for the default subentry."""
    registry = er.async_get(hass)
    entities = [
        e
        for e in registry.entities.values()
        if e.domain == "ai_task" and e.config_entry_id == mock_config_entry.entry_id
    ]
    assert entities, "Expected an ai_task entity from the default subentry"
    return entities[0].entity_id


async def test_setup_creates_ai_task_entity(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    ai_task_entity_id: str,
) -> None:
    """Setting up the entry creates the ai_task entity from the default subentry."""
    assert ai_task_entity_id == "ai_task.grok_ai_task"

    state = hass.states.get(ai_task_entity_id)
    assert state is not None
    # has_entity_name + _attr_name=None → friendly name is the device title only
    assert state.name == "Grok AI Task"
    features = ai_task.AITaskEntityFeature(state.attributes["supported_features"])
    assert ai_task.AITaskEntityFeature.GENERATE_DATA in features
    assert ai_task.AITaskEntityFeature.GENERATE_IMAGE in features
    assert ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS in features

    registry = er.async_get(hass)
    entry = registry.async_get(ai_task_entity_id)
    assert entry is not None
    assert entry.config_entry_id == mock_config_entry.entry_id
    assert entry.config_subentry_id is not None
    assert entry.unique_id == entry.config_subentry_id
    assert entry.entity_id == "ai_task.grok_ai_task"


async def test_migrate_adds_ai_task_subentry(
    hass: HomeAssistant, mock_xai_client: MagicMock
) -> None:
    """Migrating a v1.1 entry adds the ai_task_data subentry."""
    assert await async_setup_component(hass, "homeassistant", {})
    await hass.async_block_till_done()

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="xAI Grok",
        data={CONF_API_KEY: "test-key"},
        options=dict(RECOMMENDED_OPTIONS),
        version=1,
        minor_version=1,
    )
    entry.add_to_hass(hass)

    # Preserve a conversation entity unique_id across migration
    registry = er.async_get(hass)
    registry.async_get_or_create(
        "conversation",
        DOMAIN,
        entry.entry_id,
        config_entry=entry,
        suggested_object_id="grok",
    )
    conv_before = registry.async_get_entity_id(
        "conversation", DOMAIN, entry.entry_id
    )
    options_before = dict(entry.options)

    with (
        patch(
            "custom_components.grok_conversation.create_xai_client",
            return_value=mock_xai_client,
        ),
        patch(
            "custom_components.grok_conversation.async_validate_voice_access",
            return_value=(True, "ok"),
        ),
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    assert entry.minor_version == 4
    assert any(
        s.subentry_type == "ai_task_data" for s in entry.subentries.values()
    )
    assert dict(entry.options) == options_before
    assert (
        registry.async_get_entity_id("conversation", DOMAIN, entry.entry_id)
        == conv_before
    )


async def test_generate_data_free_text(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """Free-text generate_data returns a string."""
    _set_sample(mock_xai_client, _chat_response("The driveway is clear."))

    result = await ai_task.async_generate_data(
        hass,
        task_name="Driveway",
        entity_id=ai_task_entity_id,
        instructions="Describe the driveway",
    )

    assert result.data == "The driveway is clear."
    call_kwargs = mock_xai_client.chat.create.call_args.kwargs
    assert "response_format" not in call_kwargs


async def test_generate_data_structured(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """Structured generate_data sends json_schema response_format and returns dict."""
    _set_sample(mock_xai_client, _chat_response('{"cars": 2}'))

    result = await ai_task.async_generate_data(
        hass,
        task_name="Driveway check",
        entity_id=ai_task_entity_id,
        instructions="How many cars?",
        structure=_structure_schema(
            {
                vol.Required("cars"): selector.selector({"number": {}}),
                vol.Optional("notes"): selector.selector({"text": {}}),
            }
        ),
    )

    assert result.data == {"cars": 2}
    call_kwargs = mock_xai_client.chat.create.call_args.kwargs
    rf = call_kwargs["response_format"]
    assert rf.format_type == chat_pb2.FORMAT_TYPE_JSON_SCHEMA
    schema = json.loads(rf.schema)
    # Strict mode: every property (including optional) must be required;
    # optional fields are nullable.
    assert set(schema["required"]) == {"cars", "notes"}
    assert "null" in schema["properties"]["notes"]["type"]
    assert call_kwargs["max_tokens"] >= 2500


async def test_generate_data_records_usage(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """AI Task token usage lands in UsageTracker / usage sensors."""
    from custom_components.grok_conversation.const import DOMAIN

    _set_sample(mock_xai_client, _chat_response("Counted."))

    tracker = hass.data[DOMAIN][mock_config_entry.entry_id]["usage"]
    before = tracker.snapshot.request_count

    await ai_task.async_generate_data(
        hass,
        task_name="Usage",
        entity_id=ai_task_entity_id,
        instructions="Count",
    )
    await hass.async_block_till_done()

    assert tracker.snapshot.request_count == before + 1
    assert tracker.snapshot.prompt_tokens >= 10
    assert tracker.snapshot.completion_tokens >= 5
    assert "ai_task" in tracker.snapshot.by_service
    assert tracker.snapshot.by_service["ai_task"]["request_count"] >= 1

    # Sensor entity_id is derived from device + name; find via registry.
    registry = er.async_get(hass)
    prompt_entities = [
        e
        for e in registry.entities.values()
        if e.domain == "sensor"
        and e.config_entry_id == mock_config_entry.entry_id
        and e.unique_id.endswith("_prompt_tokens")
    ]
    assert prompt_entities
    prompt_state = hass.states.get(prompt_entities[0].entity_id)
    assert prompt_state is not None
    assert float(prompt_state.state) >= 10


async def test_generate_data_invalid_json(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """Invalid JSON raises HomeAssistantError."""
    _set_sample(mock_xai_client, _chat_response("NOT JSON"))

    with pytest.raises(HomeAssistantError, match="Error with Grok structured response"):
        await ai_task.async_generate_data(
            hass,
            task_name="Driveway check",
            entity_id=ai_task_entity_id,
            instructions="How many cars?",
            structure=_structure_schema(
                {vol.Required("cars"): selector.selector({"number": {}})}
            ),
        )


async def test_generate_data_image_attachment(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    ai_task_entity_id: str,
    tmp_path: Path,
) -> None:
    """Image attachments become image_url content parts."""
    image_path = tmp_path / "driveway.jpg"
    image_path.write_bytes(b"fake-image-bytes")

    _set_sample(mock_xai_client, _chat_response("Two cars"))

    with (
        patch(
            "homeassistant.components.media_source.async_resolve_media",
            return_value=MagicMock(
                url="http://example.com/driveway.jpg",
                mime_type="image/jpeg",
                path=image_path,
            ),
        ),
    ):
        result = await ai_task.async_generate_data(
            hass,
            task_name="Driveway",
            entity_id=ai_task_entity_id,
            instructions="How many cars?",
            attachments=[
                {
                    "media_content_id": "media-source://media/driveway.jpg",
                    "media_content_type": "image/jpeg",
                }
            ],
        )

    assert result.data == "Two cars"
    kwargs = mock_xai_client.chat.create.call_args.kwargs
    assert kwargs["model"] == RECOMMENDED_CHAT_MODEL
    messages = kwargs["messages"]
    user_msg = next(
        m
        for m in messages
        if chat_pb2.MessageRole.Name(m.role).removeprefix("ROLE_").lower() == "user"
    )
    assert any(part.HasField("image_url") for part in user_msg.content)


async def test_generate_data_non_image_attachment(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    ai_task_entity_id: str,
    tmp_path: Path,
) -> None:
    """Non-image attachments raise HomeAssistantError."""
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    with (
        patch(
            "homeassistant.components.media_source.async_resolve_media",
            return_value=MagicMock(
                url="http://example.com/notes.pdf",
                mime_type="application/pdf",
                path=pdf_path,
            ),
        ),
        pytest.raises(HomeAssistantError, match="Only images are supported"),
    ):
        await ai_task.async_generate_data(
            hass,
            task_name="Docs",
            entity_id=ai_task_entity_id,
            instructions="Summarize",
            attachments=[
                {
                    "media_content_id": "media-source://media/notes.pdf",
                    "media_content_type": "application/pdf",
                }
            ],
        )


async def test_generate_data_api_error(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """xAI API errors map to HomeAssistantError."""
    mock_xai_client.chat.create.return_value.sample = AsyncMock(
        side_effect=XAIError("boom")
    )

    with pytest.raises(HomeAssistantError, match="Error talking to xAI"):
        await ai_task.async_generate_data(
            hass,
            task_name="Fail",
            entity_id=ai_task_entity_id,
            instructions="Anything",
        )


def test_conversation_tool_cap_stays_at_ten() -> None:
    """Conversation still stops after 10 tool rounds."""
    assert MAX_TOOL_ITERATIONS == 10
    default = inspect.signature(
        GrokBaseLLMEntity._async_handle_chat_log  # noqa: SLF001
    ).parameters["max_iterations"].default
    assert default == MAX_TOOL_ITERATIONS


def test_installed_imagine_client_can_edit_an_image() -> None:
    """The installed xAI client edits through image.sample, not a separate API."""
    from xai_sdk.aio.image import Client

    one = imagine_edit_kwargs(Client.sample, ["data:image/png;base64,YQ=="])
    assert one == {"image_url": "data:image/png;base64,YQ=="}
    two = imagine_edit_kwargs(
        Client.sample,
        ["data:image/png;base64,YQ==", "data:image/jpeg;base64,Yg=="],
    )
    assert two["image_urls"] == [
        "data:image/png;base64,YQ==",
        "data:image/jpeg;base64,Yg==",
    ]


def test_imagine_edit_without_image_input_uses_translation_key() -> None:
    """A client that cannot take a source image raises a translated error."""

    async def _sample(prompt: str, model: str) -> None:
        return None

    with pytest.raises(HomeAssistantError) as err:
        imagine_edit_kwargs(_sample, ["data:image/png;base64,YQ=="])
    assert err.value.translation_domain == DOMAIN
    assert err.value.translation_key == "image_edit_unsupported"


def _ai_task_entity(hass: HomeAssistant, entity_id: str):
    entity = hass.data[DATA_COMPONENT].get_entity(entity_id)
    assert entity is not None
    return entity


def _image_response(
    *,
    url: str | None = None,
    b64: str | None = None,
    mime_type: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> SimpleNamespace:
    payload: dict[str, Any] = {"respect_moderation": True}
    if url is not None:
        payload["url"] = url
    if b64 is not None:
        payload["base64"] = b64
    if mime_type is not None:
        payload["mime_type"] = mime_type
    if width is not None:
        payload["width"] = width
    if height is not None:
        payload["height"] = height
    return SimpleNamespace(**payload)


async def _run_generate_image(hass: HomeAssistant, entity, instructions: str, **task_kwargs):
    with async_get_chat_session(hass) as session:
        return await entity.internal_async_generate_image(
            session,
            ai_task.GenImageTask(
                name="Picture",
                instructions=instructions,
                **task_kwargs,
            ),
        )


async def _setup_task_entry(
    hass: HomeAssistant,
    mock_xai_client: MagicMock,
    *,
    title: str,
    data: dict[str, Any],
) -> str:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=title,
        data={CONF_API_KEY: "test-key"},
        options=dict(RECOMMENDED_OPTIONS),
        version=1,
        minor_version=4,
        subentries_data=[
            {
                "subentry_type": "ai_task_data",
                "title": title,
                "data": data,
                "unique_id": None,
            }
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    entry.runtime_data = mock_xai_client
    registry = er.async_get(hass)
    entities = [
        item
        for item in registry.entities.values()
        if item.domain == "ai_task" and item.config_entry_id == entry.entry_id
    ]
    assert len(entities) == 1
    return entities[0].entity_id


class _RecordingTool(llm.Tool):
    """Test tool that records calls."""

    name = "test_light"
    description = "Control a test light"
    parameters = vol.Schema({vol.Required("action"): str})

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> llm.ToolResult:
        self.calls.append(dict(tool_input.tool_args))
        return llm.ToolResult(data={"ok": True})


@dataclass(slots=True, kw_only=True)
class _ToolAPI(llm.API):
    """LLM API exposing one recording tool."""

    tool: _RecordingTool

    async def async_get_api_instance(
        self, llm_context: llm.LLMContext
    ) -> llm.APIInstance:
        return llm.APIInstance(
            api=self,
            api_prompt="Tools are available.",
            llm_context=llm_context,
            tools=[self.tool],
        )


def _tool_call(name: str, arguments: str, call_id: str) -> MagicMock:
    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function = MagicMock()
    tool_call.function.name = name
    tool_call.function.arguments = arguments
    return tool_call


async def test_generate_data_runs_more_than_ten_tool_rounds(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """generate_data keeps going after the conversation cap of 10 rounds."""
    tool = _RecordingTool()
    rounds = 11
    responses = [
        _chat_response(
            "",
            tool_calls=[
                _tool_call("test_light", '{"action":"tick"}', f"call_{index}")
            ],
        )
        for index in range(rounds)
    ]
    responses.append(_chat_response("finished the chain"))
    mock_xai_client.chat.create.return_value.sample = AsyncMock(side_effect=responses)

    result = await ai_task.async_generate_data(
        hass,
        task_name="Long chain",
        entity_id=ai_task_entity_id,
        instructions="Keep going",
        llm_api=_ToolAPI(hass=hass, id="test_ai_task", name="Test", tool=tool),
    )

    assert result.data == "finished the chain"
    assert len(tool.calls) == rounds
    assert mock_xai_client.chat.create.call_count == rounds + 1


async def test_text_only_chat_model_still_accepts_attachments(
    hass: HomeAssistant,
    mock_xai_client: MagicMock,
    tmp_path: Path,
) -> None:
    """Attachment support does not follow the chat model id."""
    entity_id = await _setup_task_entry(
        hass,
        mock_xai_client,
        title="Text Only Task",
        data={
            **RECOMMENDED_AI_TASK_OPTIONS,
            CONF_CHAT_MODEL: "text-only-custom",
        },
    )
    state = hass.states.get(entity_id)
    assert state is not None
    features = ai_task.AITaskEntityFeature(state.attributes["supported_features"])
    assert ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS in features
    assert ai_task.AITaskEntityFeature.GENERATE_IMAGE in features

    image_path = tmp_path / "driveway.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    _set_sample(mock_xai_client, _chat_response("Two cars"))

    with patch(
        "homeassistant.components.media_source.async_resolve_media",
        return_value=MagicMock(
            url="http://example.com/driveway.jpg",
            mime_type="image/jpeg",
            path=image_path,
        ),
    ):
        result = await ai_task.async_generate_data(
            hass,
            task_name="Driveway",
            entity_id=entity_id,
            instructions="How many cars?",
            attachments=[
                {
                    "media_content_id": "media-source://media/driveway.jpg",
                    "media_content_type": "image/jpeg",
                }
            ],
        )

    assert result.data == "Two cars"
    assert (
        mock_xai_client.chat.create.call_args.kwargs["model"]
        == RECOMMENDED_VISION_MODEL
    )


async def test_generate_image_returns_bytes_and_subentry_model(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """generate_image returns one image from the subentry Imagine model."""
    raw = b"png-bytes"
    encoded = base64.b64encode(raw).decode()
    mock_xai_client.image.sample = AsyncMock(
        return_value=_image_response(
            b64=f"data:image/png;base64,{encoded}",
            mime_type="image/png",
            width=32,
            height=48,
        )
    )
    entity = _ai_task_entity(hass, ai_task_entity_id)

    result = await _run_generate_image(hass, entity, "a lighthouse")

    assert result.image_data == raw
    assert result.mime_type == "image/png"
    assert result.model == RECOMMENDED_IMAGE_GENERATION_MODEL
    assert result.model != RECOMMENDED_CHAT_MODEL
    assert result.width == 32
    assert result.height == 48
    assert result.conversation_id
    kwargs = mock_xai_client.image.sample.await_args.kwargs
    assert kwargs["model"] == RECOMMENDED_IMAGE_GENERATION_MODEL
    assert kwargs["prompt"] == "a lighthouse"
    assert kwargs["image_format"] == "base64"
    assert "aspect_ratio" not in kwargs
    assert "resolution" not in kwargs
    assert "quality" not in kwargs
    assert "image_url" not in kwargs
    assert "n" not in kwargs


async def test_generate_image_uses_subentry_image_model(
    hass: HomeAssistant,
    mock_xai_client: MagicMock,
) -> None:
    """The image model is the subentry field, not the chat model."""
    entity_id = await _setup_task_entry(
        hass,
        mock_xai_client,
        title="Imagine Two",
        data={
            **RECOMMENDED_AI_TASK_OPTIONS,
            CONF_CHAT_MODEL: "text-only-custom",
            CONF_IMAGE_MODEL: "grok-imagine-image-2.0",
        },
    )
    raw = b"jpeg-bytes"
    mock_xai_client.image.sample = AsyncMock(
        return_value=_image_response(
            b64=base64.b64encode(raw).decode(),
            mime_type="image/jpeg",
        )
    )

    result = await _run_generate_image(
        hass, _ai_task_entity(hass, entity_id), "a red barn"
    )

    assert result.image_data == raw
    assert result.model == "grok-imagine-image-2.0"
    assert result.width is None
    assert result.height is None
    assert mock_xai_client.image.sample.await_args.kwargs["model"] == (
        "grok-imagine-image-2.0"
    )


async def test_generate_image_downloads_url(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """A URL response is downloaded with the Home Assistant session."""
    png = b"\x89PNG-cat"
    response = MagicMock()
    response.read = AsyncMock(return_value=png)
    response.content_type = "image/png"
    response.raise_for_status = MagicMock()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.get = MagicMock(return_value=response)
    mock_xai_client.image.sample = AsyncMock(
        return_value=_image_response(url="https://cdn.example/cat.png")
    )

    with patch(
        "custom_components.grok_conversation.ai_task.async_get_clientsession",
        return_value=session,
    ):
        result = await _run_generate_image(
            hass, _ai_task_entity(hass, ai_task_entity_id), "a cat"
        )

    assert result.image_data == png
    assert result.mime_type == "image/png"
    assert result.width is None
    assert result.height is None
    session.get.assert_called_once_with("https://cdn.example/cat.png")


async def test_generate_image_edits_attachment(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    ai_task_entity_id: str,
    tmp_path: Path,
) -> None:
    """An image attachment is sent to Imagine as an edit source."""
    image_path = tmp_path / "source.jpg"
    image_path.write_bytes(b"source-jpeg")
    raw = b"edited"
    mock_xai_client.image.sample = AsyncMock(
        return_value=_image_response(
            b64=base64.b64encode(raw).decode(),
            mime_type="image/png",
        )
    )

    result = await _run_generate_image(
        hass,
        _ai_task_entity(hass, ai_task_entity_id),
        "make it night",
        attachments=[
            conversation.Attachment(
                media_content_id="media-source://media/source.jpg",
                mime_type="image/jpeg",
                path=image_path,
            )
        ],
    )

    assert result.image_data == raw
    image_url = mock_xai_client.image.sample.await_args.kwargs["image_url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(image_url.split("base64,", 1)[1]) == b"source-jpeg"


async def test_generate_image_edits_multiple_attachments(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    ai_task_entity_id: str,
    tmp_path: Path,
) -> None:
    """Several images use the client's multi-image edit input."""
    first = tmp_path / "one.png"
    second = tmp_path / "two.png"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    mock_xai_client.image.sample = AsyncMock(
        return_value=_image_response(
            b64=base64.b64encode(b"out").decode(),
            mime_type="image/png",
        )
    )

    await _run_generate_image(
        hass,
        _ai_task_entity(hass, ai_task_entity_id),
        "combine them",
        attachments=[
            conversation.Attachment(
                media_content_id="media-source://media/one.png",
                mime_type="image/png",
                path=first,
            ),
            conversation.Attachment(
                media_content_id="media-source://media/two.png",
                mime_type="image/png",
                path=second,
            ),
        ],
    )

    kwargs = mock_xai_client.image.sample.await_args.kwargs
    assert "image_url" not in kwargs
    assert len(kwargs["image_urls"]) == 2
    assert all(url.startswith("data:image/png;base64,") for url in kwargs["image_urls"])


async def test_generate_image_rejects_non_image_attachment(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    ai_task_entity_id: str,
    tmp_path: Path,
) -> None:
    """A non-image attachment cannot be used as an Imagine edit source."""
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(b"%PDF-secret-provider")

    with pytest.raises(HomeAssistantError) as err:
        await _run_generate_image(
            hass,
            _ai_task_entity(hass, ai_task_entity_id),
            "edit this",
            attachments=[
                conversation.Attachment(
                    media_content_id="media-source://media/notes.pdf",
                    mime_type="application/pdf",
                    path=pdf_path,
                )
            ],
        )

    assert err.value.translation_key == "image_edit_unsupported"
    assert str(err.value) == "Grok Imagine cannot edit this attachment"
    assert "secret-provider" not in str(err.value)
    mock_xai_client.image.sample.assert_not_awaited()


async def test_generate_image_api_error_hides_provider_text(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """Imagine failures do not include the provider exception text."""
    mock_xai_client.image.sample = AsyncMock(
        side_effect=XAIError("secret-provider-text")
    )

    with pytest.raises(HomeAssistantError, match="^Error generating image$") as err:
        await _run_generate_image(
            hass, _ai_task_entity(hass, ai_task_entity_id), "a cat"
        )

    assert "secret-provider-text" not in str(err.value)


async def test_generate_image_service_returns_media(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_xai_client: MagicMock,
    ai_task_entity_id: str,
) -> None:
    """An automation can call ai_task.generate_image and receive one image."""
    media = Path(next(iter(hass.config.media_dirs.values())))
    raw = b"service-png"
    mock_xai_client.image.sample = AsyncMock(
        return_value=_image_response(
            b64=base64.b64encode(raw).decode(),
            mime_type="image/png",
            width=8,
            height=8,
        )
    )

    result = await ai_task.async_generate_image(
        hass,
        task_name="Lighthouse",
        entity_id=ai_task_entity_id,
        instructions="a lighthouse",
    )

    assert result["model"] == RECOMMENDED_IMAGE_GENERATION_MODEL
    assert result["mime_type"] == "image/png"
    assert result["width"] == 8
    assert result["height"] == 8
    assert result["media_source_id"].startswith("media-source://ai_task/")
    relative = result["media_source_id"].split("://", 1)[1].split("/", 1)[1]
    saved = media / "ai_task" / relative
    assert saved.read_bytes() == raw
