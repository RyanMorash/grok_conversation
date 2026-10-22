"""xAI Grok Text-to-Speech entity for Home Assistant Assist."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.tts import (
    ATTR_PREFERRED_BITRATE,
    ATTR_PREFERRED_FORMAT,
    ATTR_PREFERRED_SAMPLE_RATE,
    ATTR_VOICE,
    TextToSpeechEntity,
    TtsAudioType,
    Voice,
)
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    DOMAIN,
    SUBENTRY_TTS,
    TTS_DEVICE_MODEL,
    effective_agent_options,
)
from .exceptions import raise_translated_xai_error
from .voice_api import XAIVoiceError, async_list_voices, async_tts
from .voice_const import (
    CONF_TTS_LANGUAGE,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    RECOMMENDED_TTS_LANGUAGE,
    RECOMMENDED_TTS_SPEED,
    RECOMMENDED_TTS_VOICE,
    TTS_LANGUAGES,
    XAI_VOICES,
    normalize_tts_language,
)

_LOGGER = logging.getLogger(__name__)
PARALLEL_UPDATES = 4


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up xAI TTS entities from TTS subentries."""
    if not config_entry.options.get("enable_tts", True):
        _LOGGER.debug("xAI TTS disabled in options")
        return

    subentries = [
        subentry
        for subentry in config_entry.subentries.values()
        if subentry.subentry_type == SUBENTRY_TTS
    ]
    if not subentries:
        return

    session = async_get_clientsession(hass)
    api_key = config_entry.data.get("api_key", "")
    voices = dict(XAI_VOICES)
    try:
        voices = await async_list_voices(session, api_key)
    except Exception as err:  # noqa: BLE001
        _LOGGER.debug("Using built-in voice catalog: %s", err)

    # Cache on hass data for options flow
    hass.data.setdefault(DOMAIN, {}).setdefault(config_entry.entry_id, {})[
        "tts_voices"
    ] = voices

    for subentry in subentries:
        async_add_entities(
            [XAIGrokTTSEntity(config_entry, subentry, voices)],
            config_subentry_id=subentry.subentry_id,
        )


class XAIGrokTTSEntity(TextToSpeechEntity):
    """xAI TTS engine (appears in Assist → Text-to-speech)."""

    # OpenAI TTS keeps has_entity_name off and uses the subentry title,
    # so the voice picker shows the name the user chose.
    _attr_has_entity_name = False
    # Preferred format options reach the engine only when listed here.
    # Otherwise Home Assistant strips them and converts with FFmpeg.
    _attr_supported_options = [
        ATTR_VOICE,
        ATTR_PREFERRED_FORMAT,
        ATTR_PREFERRED_SAMPLE_RATE,
        ATTR_PREFERRED_BITRATE,
    ]

    def __init__(
        self,
        config_entry: ConfigEntry,
        subentry: ConfigSubentry,
        voices: dict[str, str],
    ) -> None:
        self._entry = config_entry
        self._subentry = subentry
        self._voices_map = voices
        self._attr_name = subentry.title
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="xAI",
            model=TTS_DEVICE_MODEL,
            entry_type=DeviceEntryType.SERVICE,
        )
        options = self._options()
        # Assist language dropdown
        self._attr_supported_languages = list(TTS_LANGUAGES)
        default_lang = options.get(CONF_TTS_LANGUAGE, RECOMMENDED_TTS_LANGUAGE)
        if default_lang not in self._attr_supported_languages:
            default_lang = RECOMMENDED_TTS_LANGUAGE
        self._attr_default_language = default_lang

    def _options(self) -> dict[str, Any]:
        """Voice settings for this engine."""
        return effective_agent_options(self._entry.options, self._subentry.data)

    @property
    def default_options(self) -> dict[str, Any]:
        """Default voice for Assist UI / Try voice."""
        voice = self._options().get(CONF_TTS_VOICE, RECOMMENDED_TTS_VOICE)
        if voice not in self._voices_map:
            voice = next(iter(self._voices_map), RECOMMENDED_TTS_VOICE)
        return {ATTR_VOICE: voice}

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice]:
        """Voices for the Assist voice dropdown (all xAI voices are multilingual)."""
        default = self._options().get(CONF_TTS_VOICE, RECOMMENDED_TTS_VOICE)
        items = [
            Voice(vid, label) for vid, label in sorted(self._voices_map.items(), key=lambda x: x[1].lower())
        ]
        # Put configured default first
        items.sort(key=lambda v: (0 if v.voice_id == default else 1, v.name.lower()))
        return items

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Generate audio via xAI /v1/tts in the caller's preferred format."""
        stored = self._options()
        voice_id = options.get(
            ATTR_VOICE,
            stored.get(CONF_TTS_VOICE, RECOMMENDED_TTS_VOICE),
        )
        speed = float(stored.get(CONF_TTS_SPEED, RECOMMENDED_TTS_SPEED))
        # Prefer Assist-selected language; fall back to integration default
        lang = normalize_tts_language(
            language or stored.get(CONF_TTS_LANGUAGE, RECOMMENDED_TTS_LANGUAGE)
        )
        api_key = self._entry.data.get("api_key", "")
        session = async_get_clientsession(self.hass)

        _LOGGER.debug(
            "xAI TTS: voice=%s lang=%s format=%s chars=%s",
            voice_id,
            lang,
            options.get(ATTR_PREFERRED_FORMAT) or "mp3",
            len(message or ""),
        )
        try:
            extension, audio = await async_tts(
                session,
                api_key,
                text=message,
                voice_id=str(voice_id),
                language=lang,
                speed=speed,
                preferred_format=options.get(ATTR_PREFERRED_FORMAT),
                preferred_sample_rate=options.get(ATTR_PREFERRED_SAMPLE_RATE),
                preferred_bitrate=options.get(ATTR_PREFERRED_BITRATE),
            )
        except XAIVoiceError as err:
            raise_translated_xai_error(err)

        return extension, audio
