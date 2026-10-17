"""Async helpers for xAI TTS and STT HTTP APIs."""

from __future__ import annotations

import logging
import struct
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout, FormData

from .voice_const import (
    XAI_STT_URL,
    XAI_TTS_URL,
    XAI_VOICES,
    XAI_VOICES_URL,
    normalize_stt_language,
    normalize_tts_language,
)

_LOGGER = logging.getLogger(__name__)

_TIMEOUT_SHORT = ClientTimeout(total=20)
_TIMEOUT_TTS = ClientTimeout(total=60)
_TIMEOUT_STT = ClientTimeout(total=120)

# https://docs.x.ai/developers/model-capabilities/audio/text-to-speech
# sample_rate is shared. bit_rate is MP3 only.
_TTS_SAMPLE_RATES = frozenset({8000, 16000, 22050, 24000, 44100, 48000})
_TTS_MP3_BIT_RATES = frozenset({32000, 64000, 96000, 128000, 192000})
_DEFAULT_TTS_SAMPLE_RATE = 24000
_DEFAULT_TTS_BIT_RATE = 128000

# Assist preferred format → xAI codec. ``raw`` is PCM, matching OpenAI.
# ogg, oga, and opus are not codecs on the xAI page.
_PREFERRED_TO_CODEC = {
    "mp3": "mp3",
    "wav": "wav",
    "pcm": "pcm",
    "raw": "pcm",
    "mulaw": "mulaw",
    "ulaw": "ulaw",
    "alaw": "alaw",
}


class XAIVoiceError(Exception):
    """Raised when an xAI voice API call fails."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _pcm16_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw s16le PCM in a minimal WAV header."""
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,  # PCM fmt chunk size
        1,  # audio format PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm


def _voices_list_usable(data: Any) -> bool:
    """Return True when the voices payload contains at least one voice id."""
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        raw = data.get("voices") or data.get("data") or []
        items = raw if isinstance(raw, list) else []
    else:
        return False
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("voice_id") or item.get("id") or item.get("name"):
            return True
    return False


async def async_validate_voice_access(
    session: ClientSession, api_key: str
) -> tuple[bool, str]:
    """Check whether the API key can use xAI voice endpoints.

    Uses the voices list only. A missing or unusable list does not synthesize
    speech. Returns (ok, detail_message).
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with session.get(
            XAI_VOICES_URL, headers=headers, timeout=_TIMEOUT_SHORT
        ) as resp:
            if resp.status in (401, 403):
                text = await resp.text()
                return (
                    False,
                    f"API key rejected for voice ({resp.status}): {text[:200]}",
                )
            if resp.status != 200:
                return False, f"Voices list was not usable ({resp.status})"
            try:
                data = await resp.json(content_type=None)
            except (ClientError, ValueError, TypeError):
                return False, "Voices list was not usable"
    except (ClientError, TimeoutError) as err:
        return False, f"Could not reach xAI Voice API: {err}"

    if _voices_list_usable(data):
        return True, "Voice API accessible (voices list OK)"
    return False, "Voices list was not usable"


async def async_list_voices(
    session: ClientSession, api_key: str
) -> dict[str, str]:
    """Return voice_id → name mapping (falls back to built-in catalog)."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with session.get(
            XAI_VOICES_URL, headers=headers, timeout=_TIMEOUT_SHORT
        ) as resp:
            if resp.status != 200:
                return dict(XAI_VOICES)
            data = await resp.json(content_type=None)
    except (ClientError, ValueError):
        return dict(XAI_VOICES)

    voices: dict[str, str] = {}
    items = data if isinstance(data, list) else data.get("voices") or data.get("data") or []
    for item in items:
        if not isinstance(item, dict):
            continue
        vid = item.get("voice_id") or item.get("id") or item.get("name")
        name = item.get("name") or item.get("display_name") or vid
        if vid:
            voices[str(vid)] = str(name)
    return voices or dict(XAI_VOICES)


def _exact_int(value: Any) -> int | None:
    """Return an int when ``value`` is an integer, or None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
    return None


def tts_output_format(
    preferred_format: str | None = None,
    preferred_sample_rate: Any = None,
    preferred_bitrate: Any = None,
) -> tuple[str, dict[str, Any]]:
    """Map a preferred format onto an xAI ``output_format`` and a file extension.

    Codecs xAI documents are ``mp3``, ``wav``, ``pcm``, ``mulaw`` (or ``ulaw``),
    and ``alaw``. ``raw`` is sent as ``pcm`` and reported back as ``raw``.
    ``ogg``, ``oga``, ``opus``, and any other name are requested as MP3.

    A preferred sample rate is sent only when it is one of 8000, 16000, 22050,
    24000, 44100, or 48000. A preferred bit rate is sent only for MP3, and only
    when it is one of 32000, 64000, 96000, 128000, or 192000. An out-of-range
    MP3 pair stays at 24 kHz / 128 kbps. Other codecs omit ``bit_rate`` and use
    24 kHz when the sample rate is missing or not in that list.
    """
    name = (
        preferred_format.strip().lower()
        if isinstance(preferred_format, str)
        else ""
    )
    codec = _PREFERRED_TO_CODEC.get(name)
    sample_rate = _exact_int(preferred_sample_rate)
    bit_rate = _exact_int(preferred_bitrate)
    if codec is None:
        codec = "mp3"
        extension = "mp3"
    else:
        extension = name

    if codec == "mp3":
        rate_allowed = sample_rate is None or sample_rate in _TTS_SAMPLE_RATES
        bit_allowed = bit_rate is None or bit_rate in _TTS_MP3_BIT_RATES
        if rate_allowed and bit_allowed:
            output_rate = (
                sample_rate
                if sample_rate in _TTS_SAMPLE_RATES
                else _DEFAULT_TTS_SAMPLE_RATE
            )
            output_bit_rate = (
                bit_rate if bit_rate in _TTS_MP3_BIT_RATES else _DEFAULT_TTS_BIT_RATE
            )
        else:
            output_rate = _DEFAULT_TTS_SAMPLE_RATE
            output_bit_rate = _DEFAULT_TTS_BIT_RATE
        return extension, {
            "codec": "mp3",
            "sample_rate": output_rate,
            "bit_rate": output_bit_rate,
        }

    output_rate = (
        sample_rate if sample_rate in _TTS_SAMPLE_RATES else _DEFAULT_TTS_SAMPLE_RATE
    )
    return extension, {"codec": codec, "sample_rate": output_rate}


async def async_tts(
    session: ClientSession,
    api_key: str,
    *,
    text: str,
    voice_id: str = "eve",
    language: str = "en",
    speed: float = 1.0,
    preferred_format: str | None = None,
    preferred_sample_rate: Any = None,
    preferred_bitrate: Any = None,
) -> tuple[str, bytes]:
    """Synthesize speech; returns the file extension and audio bytes."""
    if not text or not text.strip():
        raise XAIVoiceError("Empty TTS text")
    # xAI max 15_000 chars per request
    message = text.strip()
    if len(message) > 15000:
        message = message[:15000]

    lang = normalize_tts_language(language)
    extension, output_format = tts_output_format(
        preferred_format,
        preferred_sample_rate,
        preferred_bitrate,
    )
    payload: dict[str, Any] = {
        "text": message,
        "voice_id": voice_id or "eve",
        "language": lang,
        "output_format": output_format,
    }
    if speed and abs(speed - 1.0) > 0.01:
        payload["speed"] = max(0.7, min(1.5, float(speed)))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with session.post(
            XAI_TTS_URL, headers=headers, json=payload, timeout=_TIMEOUT_TTS
        ) as resp:
            body = await resp.read()
            if resp.status != 200:
                detail = body.decode("utf-8", errors="replace")[:300]
                raise XAIVoiceError(
                    f"xAI TTS failed ({resp.status}): {detail}",
                    status=resp.status,
                )
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "application/json" in content_type:
                # Timestamped envelope
                import base64
                import json

                data = json.loads(body)
                audio_b64 = data.get("audio")
                if not audio_b64:
                    raise XAIVoiceError("TTS JSON response missing audio")
                return extension, base64.b64decode(audio_b64)
            return extension, body
    except ClientError as err:
        raise XAIVoiceError(f"xAI TTS network error: {err}") from err


async def async_stt(
    session: ClientSession,
    api_key: str,
    *,
    audio: bytes,
    language: str | None = "en",
    sample_rate: int = 16000,
    is_raw_pcm: bool = False,
    channels: int = 1,
    filename: str = "audio.wav",
    content_type: str = "audio/wav",
) -> str:
    """Transcribe audio bytes; returns transcript text."""
    if not audio:
        raise XAIVoiceError("Empty audio for STT")

    if is_raw_pcm:
        # Prefer WAV wrapper for reliability
        audio = _pcm16_to_wav(audio, sample_rate=sample_rate, channels=channels)
        filename = "audio.wav"
        content_type = "audio/wav"

    lang = normalize_stt_language(language)
    headers = {"Authorization": f"Bearer {api_key}"}

    form = FormData()
    # Fields before file
    if lang:
        form.add_field("language", lang)
        form.add_field("format", "true")
    form.add_field(
        "file",
        audio,
        filename=filename,
        content_type=content_type,
    )

    try:
        async with session.post(
            XAI_STT_URL, headers=headers, data=form, timeout=_TIMEOUT_STT
        ) as resp:
            if resp.status != 200:
                detail = (await resp.text())[:300]
                raise XAIVoiceError(
                    f"xAI STT failed ({resp.status}): {detail}",
                    status=resp.status,
                )
            data = await resp.json(content_type=None)
    except ClientError as err:
        raise XAIVoiceError(f"xAI STT network error: {err}") from err

    text = data.get("text") if isinstance(data, dict) else None
    if text is None:
        raise XAIVoiceError(f"Unexpected STT response: {str(data)[:200]}")
    return str(text).strip()
