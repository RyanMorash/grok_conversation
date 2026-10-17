"""AI Task platform for xAI Grok."""

from __future__ import annotations

import base64
import inspect
from json import JSONDecodeError
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import ClientError

from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from . import (
    _safe_image_attr,
    build_image_generate_kwargs,
    encode_file,
    format_images_response,
)
from .api_helpers import XAIError, async_generate_images
from .const import (
    CONF_IMAGE_MODEL,
    CONF_PROMPT,
    DOMAIN,
    LOGGER,
    RECOMMENDED_IMAGE_GENERATION_MODEL,
)
from .entity import GrokBaseLLMEntity

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigSubentry

    from . import GrokConfigEntry

PARALLEL_UPDATES = 0

# openai_conversation uses 1000 for generate_data. Conversation stays at 10.
_AI_TASK_MAX_TOOL_ITERATIONS = 1000


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: GrokConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "ai_task_data":
            continue

        async_add_entities(
            [GrokAITaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


def imagine_edit_kwargs(sample: Any, image_inputs: list[str]) -> dict[str, Any]:
    """Return Imagine edit kwargs when this client can edit an image.

    xAI Imagine takes the source image on ``image.sample`` (``image_url`` /
    ``image_urls``). There is no separate edit method.
    """
    if not image_inputs:
        return {}
    try:
        parameters = inspect.signature(sample).parameters
    except (TypeError, ValueError) as err:
        raise HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="image_edit_unsupported",
        ) from err

    accepts_any = any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )
    if len(image_inputs) == 1 and (accepts_any or "image_url" in parameters):
        return {"image_url": image_inputs[0]}
    if accepts_any or "image_urls" in parameters:
        return {"image_urls": list(image_inputs)}
    raise HomeAssistantError(
        translation_domain=DOMAIN,
        translation_key="image_edit_unsupported",
    )


def _positive_dimension(item: Any, name: str) -> int | None:
    """Return a positive integer dimension when the image response has one."""
    try:
        value = getattr(item, name, None)
    except (ValueError, AttributeError):
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _mime_from_image(item: Any, raw_b64: str | None) -> str | None:
    """Read an image MIME type from the response or a data-URI prefix."""
    mime = _safe_image_attr(item, "mime_type")
    if isinstance(mime, str):
        declared = mime.split(";", 1)[0].strip().lower()
        if declared.startswith("image/"):
            return declared
    if isinstance(raw_b64, str) and "base64," in raw_b64 and "data:" in raw_b64:
        header = raw_b64.split("base64,", 1)[0]
        declared = header.split("data:", 1)[-1].split(";", 1)[0].strip().lower()
        if declared.startswith("image/"):
            return declared
    return None


class GrokAITaskEntity(ai_task.AITaskEntity, GrokBaseLLMEntity):
    """Grok AI Task entity."""

    def __init__(
        self, entry: GrokConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)
        # Attachments stay on even when the chat model is not a vision model.
        # resolve_vision_model still picks the model for a turn that has them.
        # GENERATE_IMAGE follows Cloud: the image model is not the chat model.
        self._attr_supported_features = (
            ai_task.AITaskEntityFeature.GENERATE_DATA
            | ai_task.AITaskEntityFeature.GENERATE_IMAGE
            | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
        )

    def _image_model(self) -> str:
        """Return the subentry Imagine model."""
        data = self.subentry.data if self.subentry is not None else {}
        model = data.get(CONF_IMAGE_MODEL) or RECOMMENDED_IMAGE_GENERATION_MODEL
        return str(model)

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        try:
            await self._async_handle_chat_log(
                chat_log,
                structure=task.structure,
                structure_name=task.name,
                service="ai_task",
                fallback_model="",  # no conversation fallback for AI Task
                max_iterations=_AI_TASK_MAX_TOOL_ITERATIONS,
            )
        except HomeAssistantError:
            raise
        except Exception as err:
            LOGGER.error("Error in AI Task generate_data: %s", err, exc_info=True)
            raise HomeAssistantError(f"Error talking to xAI: {err}") from err

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise HomeAssistantError(
                "Last content in chat log is not an AssistantContent"
            )

        text = chat_log.content[-1].content or ""

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )

        try:
            data = json_loads(text)
        except JSONDecodeError as err:
            LOGGER.error(
                "Failed to parse JSON response: %s. Response: %s",
                err,
                text,
            )
            raise HomeAssistantError(
                "Error with Grok structured response"
            ) from err

        if not isinstance(data, dict):
            LOGGER.error(
                "Structured response was not a dict: %s. Response: %s",
                type(data).__name__,
                text,
            )
            raise HomeAssistantError("Error with Grok structured response")

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )

    async def _async_generate_image(
        self,
        task: ai_task.GenImageTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenImageTaskResult:
        """Handle a generate image task."""
        model = self._image_model()
        # Aspect ratio, resolution, quality, and count stay on the
        # generate_image service. Ask Imagine for bytes; still accept a URL.
        kwargs = build_image_generate_kwargs(
            {
                CONF_PROMPT: task.instructions,
                "n": 1,
                "response_format": "b64_json",
            },
            model,
        )
        if task.attachments:
            image_inputs = await self._async_image_edit_inputs(task.attachments)
            kwargs.update(
                imagine_edit_kwargs(self.entry.runtime_data.image.sample, image_inputs)
            )

        client = self.entry.runtime_data
        try:
            images = await async_generate_images(client, **kwargs)
        except XAIError as err:
            LOGGER.error("Error generating image: %s", err)
            raise HomeAssistantError("Error generating image") from err

        if not images:
            raise HomeAssistantError("Error generating image")

        item = images[0]
        raw_b64 = _safe_image_attr(item, "base64", "b64_json")
        mime_type = _mime_from_image(
            item, raw_b64 if isinstance(raw_b64, str) else None
        )
        formatted = format_images_response(
            item, model=model, response_format="b64_json"
        )
        image_entry = formatted["images"][0]
        if image_entry.get("b64_json"):
            try:
                image_data = base64.b64decode(image_entry["b64_json"])
            except ValueError as err:
                LOGGER.error("Image generation returned invalid base64: %s", err)
                raise HomeAssistantError("Error generating image") from err
            if not image_data:
                raise HomeAssistantError("Error generating image")
        elif image_entry.get("url"):
            image_data, downloaded_mime = await self._async_download_image(
                str(image_entry["url"])
            )
            mime_type = mime_type or downloaded_mime
        else:
            raise HomeAssistantError("Error generating image")

        return ai_task.GenImageTaskResult(
            image_data=image_data,
            conversation_id=chat_log.conversation_id,
            mime_type=mime_type or "image/png",
            model=model,
            width=_positive_dimension(item, "width"),
            height=_positive_dimension(item, "height"),
        )

    async def _async_image_edit_inputs(
        self, attachments: list[conversation.Attachment]
    ) -> list[str]:
        """Read image attachments as data URIs for Imagine edit."""
        files: list[tuple[Path, str]] = []
        for attachment in attachments:
            mime = (attachment.mime_type or "").split(";", 1)[0].strip().lower()
            path = attachment.path
            if (
                not mime.startswith("image/")
                or path is None
                or not Path(path).exists()
            ):
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="image_edit_unsupported",
                )
            files.append((Path(path), mime))

        def _read() -> list[str]:
            urls: list[str] = []
            for path, mime in files:
                try:
                    encoded_mime, b64 = encode_file(str(path))
                except HomeAssistantError as err:
                    raise HomeAssistantError(
                        translation_domain=DOMAIN,
                        translation_key="image_edit_unsupported",
                    ) from err
                use_mime = mime if mime.startswith("image/") else encoded_mime
                if not str(use_mime).startswith("image/"):
                    raise HomeAssistantError(
                        translation_domain=DOMAIN,
                        translation_key="image_edit_unsupported",
                    )
                urls.append(f"data:{use_mime};base64,{b64}")
            return urls

        return await self.hass.async_add_executor_job(_read)

    async def _async_download_image(self, url: str) -> tuple[bytes, str | None]:
        """Download image bytes with the Home Assistant session."""
        session = async_get_clientsession(self.hass)
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.read()
                content_type = response.content_type
        except (ClientError, TimeoutError) as err:
            LOGGER.error("Error downloading generated image: %s", err)
            raise HomeAssistantError("Error generating image") from err
        if not data:
            raise HomeAssistantError("Error generating image")
        mime_type = None
        if isinstance(content_type, str):
            declared = content_type.split(";", 1)[0].strip().lower()
            if declared.startswith("image/"):
                mime_type = declared
        return data, mime_type
