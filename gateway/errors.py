"""Error types and retry classification for Mesh API calls."""

from typing import Literal


class MeshError(RuntimeError):
    """A Mesh API call failed after the allowed attempts, or failed permanently."""


class StructuredOutputError(MeshError):
    """No structured-output method produced a valid object."""


class EmbeddingDimensionError(MeshError):
    """Mesh returned embeddings with an unexpected dimension."""


Decision = Literal["retry", "fatal", "spend_limit"]

_RETRY_STATUS = {408, 409, 429}


def classify(exc: BaseException) -> Decision:
    """Decide whether a failed Mesh attempt is retried."""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status == 402:
        return "spend_limit"
    if isinstance(status, int):
        return "retry" if status in _RETRY_STATUS or status >= 500 else "fatal"
    name = type(exc).__name__
    if isinstance(exc, (TimeoutError, ConnectionError)) or name in (
        "APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError",
    ):
        return "retry"
    return "fatal"
