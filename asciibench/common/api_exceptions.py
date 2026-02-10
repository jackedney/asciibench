"""Shared API error classification helpers.

Provides reusable functions to classify HTTP errors into semantic categories,
reducing duplication across httpx-based API clients.
"""

from typing import NoReturn

import httpx


def raise_for_httpx_status_error(
    exc: httpx.HTTPStatusError,
    error_map: dict[str, type[Exception]],
    base_error: type[Exception],
    model_context: str = "",
) -> NoReturn:
    """Classify an httpx HTTPStatusError and raise the appropriate exception.

    Args:
        exc: The httpx HTTPStatusError to classify
        error_map: Mapping of category names to exception types.
            Supported keys: "rate_limit", "transient", "auth", "model_not_found"
        base_error: Fallback exception type for unclassified errors
        model_context: Optional model ID for richer error messages
    """
    status_code = exc.response.status_code
    error_text = exc.response.text.lower() if exc.response.text else ""

    if status_code == 429 and "rate_limit" in error_map:
        raise error_map["rate_limit"](f"Rate limit exceeded: {exc}") from exc

    if status_code in (502, 503, 504) and "transient" in error_map:
        raise error_map["transient"](f"Transient error ({status_code}): {exc}") from exc

    if "auth" in error_map and (
        status_code == 401 or "unauthorized" in error_text or "invalid api key" in error_text
    ):
        raise error_map["auth"](f"Authentication failed: {exc}") from exc

    if "model_not_found" in error_map and (
        status_code == 404 or "not found" in error_text or "model" in error_text
    ):
        msg = f"Invalid model ID '{model_context}': {exc}" if model_context else f"Not found: {exc}"
        raise error_map["model_not_found"](msg) from exc

    raise base_error(f"API error: {exc}") from exc


def raise_for_error_message(
    exc: Exception,
    error_map: dict[str, type[Exception]],
    base_error: type[Exception],
    model_context: str = "",
) -> NoReturn:
    """Classify a generic exception by its message and raise the appropriate exception.

    Args:
        exc: The exception to classify
        error_map: Mapping of category names to exception types.
            Supported keys: "rate_limit", "auth", "model_not_found"
        base_error: Fallback exception type for unclassified errors
        model_context: Optional model ID for richer error messages
    """
    msg = str(exc).lower()

    if ("429" in msg or "rate limit" in msg) and "rate_limit" in error_map:
        raise error_map["rate_limit"](f"Rate limit exceeded: {exc}") from exc

    if "model_not_found" in error_map and ("404" in msg or "not found" in msg):
        if model_context:
            err_msg = f"Invalid model ID '{model_context}': {exc}"
        else:
            err_msg = f"Not found: {exc}"
        raise error_map["model_not_found"](err_msg) from exc

    if ("unauthorized" in msg or "authentication" in msg) and "auth" in error_map:
        raise error_map["auth"](f"Authentication failed: {exc}") from exc

    raise base_error(f"Client error: {exc}") from exc
