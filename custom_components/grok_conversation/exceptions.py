"""The exceptions used by Grok Conversation."""
from homeassistant.exceptions import HomeAssistantError


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
            f"token length(`{token}`) exceeded. Increase maximum token to avoid the issue."
        )
        self.token = token
