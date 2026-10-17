"""The exceptions used by Grok Conversation."""

from typing import NoReturn

from homeassistant.exceptions import HomeAssistantError

from .api_helpers import XAIAuthError, XAIRateLimitError
from .const import DOMAIN, LOGGER
from .voice_api import XAIVoiceError

RATE_LIMIT = "rate_limit"
AUTH_FAILED = "auth_failed"
TOKEN_LENGTH = "token_length"
CANNOT_CONNECT = "cannot_connect"


class FunctionLoadFailed(HomeAssistantError):
    """When function load failed."""

    def __init__(self) -> None:
        """Initialize error."""
        super().__init__(
            "failed to load functions. Verify functions are valid in a yaml format"
        )


class ParseArgumentsFailed(HomeAssistantError):
    """When parse arguments failed."""

    def __init__(self, arguments: str) -> None:
        """Initialize error."""
        super().__init__(
            f"failed to parse arguments `{arguments}`. Increase maximum token to avoid the issue."
        )
        self.arguments = arguments


class TokenLengthExceededError(HomeAssistantError):
    """When the model returns 'length' as finish_reason."""

    def __init__(self, token: int) -> None:
        """Initialize error."""
        super().__init__(
            translation_domain=DOMAIN,
            translation_key=TOKEN_LENGTH,
        )
        self.token = token


def xai_error_translation_key(err: BaseException) -> str:
    """Pick a stable exception key from the failure type.

    Voice failures are one exception type. The HTTP status selects the same
    auth and rate-limit keys the chat client uses subclasses for.
    """
    candidates = [err]
    cause = err.__cause__
    if isinstance(cause, BaseException):
        candidates.append(cause)
    for candidate in candidates:
        if isinstance(candidate, TokenLengthExceededError):
            return TOKEN_LENGTH
        if isinstance(candidate, XAIRateLimitError):
            return RATE_LIMIT
        if isinstance(candidate, XAIAuthError):
            return AUTH_FAILED
        if isinstance(candidate, XAIVoiceError):
            if candidate.status in (401, 403):
                return AUTH_FAILED
            if candidate.status == 429:
                return RATE_LIMIT
    return CANNOT_CONNECT


def raise_translated_xai_error(err: BaseException) -> NoReturn:
    """Log the provider exception and raise a translated Home Assistant error."""
    key = xai_error_translation_key(err)
    LOGGER.error("xAI request failed (%s): %s", key, err, exc_info=err)
    raise HomeAssistantError(
        translation_domain=DOMAIN,
        translation_key=key,
    ) from err
