"""xAI Grok Speech-to-Text entity for Home Assistant Assist."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable

from homeassistant.components import stt
from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    DOMAIN,
    STT_DEVICE_MODEL,
    SUBENTRY_STT,
    effective_agent_options,
)
from .voice_api import XAIVoiceError, async_stt
from .voice_const import (
    CONF_STT_LANGUAGE,
    RECOMMENDED_STT_LANGUAGE,
    STT_LANGUAGES,
)

_LOGGER = logging.getLogger(__name__)
PARALLEL_UPDATES = 4


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up xAI STT entities from STT subentries."""
    if not config_entry.options.get("enable_stt", True):
        _LOGGER.debug("xAI STT disabled in options")
        return
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_STT:
            continue
        async_add_entities(
            [XAIGrokSTTEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class XAIGrokSTTEntity(SpeechToTextEntity):
    """xAI STT engine (appears in Assist → Speech-to-text)."""

    # Match OpenAI STT: the entity name is the subentry title.
    _attr_has_entity_name = False

    def __init__(self, config_entry: ConfigEntry, subentry: ConfigSubentry) -> None:
        self._entry = config_entry
        self._subentry = subentry
        self._attr_name = subentry.title
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="xAI",
            model=STT_DEVICE_MODEL,
            entry_type=DeviceEntryType.SERVICE,
        )

    def _options(self) -> dict:
        """Language settings for this engine."""
        return effective_agent_options(self._entry.options, self._subentry.data)

    @property
    def supported_languages(self) -> list[str]:
        """Languages shown in Assist STT language dropdown."""
        return list(STT_LANGUAGES)

    @property
    def supported_formats(self) -> list[AudioFormats]:
        return [AudioFormats.WAV, AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        return [AudioCodecs.PCM, AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        # Must match homeassistant.components.stt.AudioSampleRates exactly
        return [
            AudioSampleRates.SAMPLERATE_8000,
            AudioSampleRates.SAMPLERATE_16000,
            AudioSampleRates.SAMPLERATE_22000,
            AudioSampleRates.SAMPLERATE_44100,
            AudioSampleRates.SAMPLERATE_48000,
        ]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        return [AudioChannels.CHANNEL_MONO, AudioChannels.CHANNEL_STEREO]

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Collect Assist audio stream and transcribe via xAI /v1/stt."""
        audio = b""
        async for chunk in stream:
            audio += chunk

        if not audio:
            _LOGGER.warning("xAI STT: empty audio stream")
            return SpeechResult(None, SpeechResultState.ERROR)

        language = metadata.language or self._options().get(
            CONF_STT_LANGUAGE, RECOMMENDED_STT_LANGUAGE
        )
        sample_rate = int(metadata.sample_rate or 16000)
        channels = 1 if metadata.channel == AudioChannels.CHANNEL_MONO else 2
        is_pcm = metadata.codec == AudioCodecs.PCM

        _LOGGER.debug(
            "xAI STT: bytes=%s lang=%s rate=%s codec=%s channels=%s",
            len(audio),
            language,
            sample_rate,
            metadata.codec,
            channels,
        )

        api_key = self._entry.data.get("api_key", "")
        session = async_get_clientsession(self.hass)

        try:
            if is_pcm:
                text = await async_stt(
                    session,
                    api_key,
                    audio=audio,
                    language=language,
                    sample_rate=sample_rate,
                    is_raw_pcm=True,
                    channels=channels,
                )
            else:
                # OGG/Opus or other container — send as-is
                ext = "ogg" if metadata.format == AudioFormats.OGG else "wav"
                ctype = "audio/ogg" if ext == "ogg" else "audio/wav"
                text = await async_stt(
                    session,
                    api_key,
                    audio=audio,
                    language=language,
                    is_raw_pcm=False,
                    filename=f"audio.{ext}",
                    content_type=ctype,
                )
        except XAIVoiceError as err:
            _LOGGER.error("xAI STT error: %s", err)
            return SpeechResult(None, SpeechResultState.ERROR)

        if not text:
            return SpeechResult("", SpeechResultState.SUCCESS)

        _LOGGER.debug("xAI STT transcript: %s", text[:200])
        return SpeechResult(text, SpeechResultState.SUCCESS)
