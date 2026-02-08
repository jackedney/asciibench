"""Retry decorator with exponential backoff for transient failures."""

import asyncio
import functools
import inspect
import time
from collections.abc import Awaitable, Callable

from asciibench.common.logging import get_logger

logger = get_logger(__name__)

# Type alias for sleep functions - accepts any callable that takes a float
# and returns either None (sync) or an Awaitable[None] (async)
SleepFunc = Callable[[float], None | Awaitable[None]]


def retry(
    max_retries: int = 3,
    base_delay_seconds: float = 1,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    sleep_func: SleepFunc | None = None,
):
    """Decorator to retry function calls on specified exceptions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default:3)
        base_delay_seconds: Base delay in seconds before first retry (default:1)
        retryable_exceptions: Tuple of exception types to retry (default: Exception)
        sleep_func: Optional sleep function for testability. If None, uses time.sleep for sync
                   functions and asyncio.sleep for async functions.

    Returns:
        Decorated function that retries on specified exceptions

    Example:
        @retry(max_retries=3, base_delay_seconds=1, retryable_exceptions=(RateLimitError,))
        def api_call():
            ...

        On first failure: retry after 1s
        On second failure: retry after 2s
        On third failure: retry after 4s
        After max retries: re-raise exception

    Example (async):
        @retry(max_retries=3, base_delay_seconds=1)
        async def async_api_call():
            ...

        On first failure: retry after 1s
        On second failure: retry after 2s
        On third failure: retry after 4s
        After max retries: re-raise exception

    Note:
        The sleep_func parameter accepts any callable that returns either None (sync)
        or an Awaitable[None] (async). This includes lambdas, functools.partial,
        and mock objects. The awaitability is detected at call time.
    """
    if not isinstance(max_retries, int):
        raise ValueError(f"max_retries must be an integer, got {type(max_retries).__name__}")
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")
    if not isinstance(base_delay_seconds, (int, float)):
        raise ValueError(
            f"base_delay_seconds must be a number, got {type(base_delay_seconds).__name__}"
        )
    if base_delay_seconds < 0:
        raise ValueError(f"base_delay_seconds must be >= 0, got {base_delay_seconds}")
    if not isinstance(retryable_exceptions, tuple):
        raise TypeError(
            f"retryable_exceptions must be a tuple, got {type(retryable_exceptions).__name__}"
        )
    if len(retryable_exceptions) == 0:
        raise TypeError("retryable_exceptions must not be empty")
    for exc_type in retryable_exceptions:
        if not (isinstance(exc_type, type) and issubclass(exc_type, Exception)):
            raise TypeError(f"retryable_exceptions must contain Exception types, got {exc_type}")

    async def _async_sleep(delay: float, custom_sleep: SleepFunc | None) -> None:
        """Execute sleep, handling both sync and async sleep functions."""
        if custom_sleep is None:
            await asyncio.sleep(delay)
        else:
            result = custom_sleep(delay)
            if inspect.isawaitable(result):
                await result

    def _sync_sleep(delay: float, custom_sleep: SleepFunc | None) -> None:
        """Execute sleep for sync context."""
        if custom_sleep is None:
            time.sleep(delay)
        else:
            custom_sleep(delay)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.warning(
                            f"Function {func.__name__} failed after {max_retries} retries"
                        )
                        raise

                    delay = base_delay_seconds * (2**attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {delay}s delay: {e}"
                    )
                    _sync_sleep(delay, sleep_func)

            # This should never be reached, but type checkers need it
            if last_exception is None:
                raise RuntimeError("Unexpected retry loop exit without exception")
            raise last_exception

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.warning(
                            f"Async function {func.__name__} failed after {max_retries} retries"
                        )
                        raise

                    delay = base_delay_seconds * (2**attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {delay}s delay: {e}"
                    )
                    await _async_sleep(delay, sleep_func)

            # This should never be reached, but type checkers need it
            if last_exception is None:
                raise RuntimeError("Unexpected retry loop exit without exception")
            raise last_exception

        return async_wrapper if inspect.iscoroutinefunction(func) else wrapper

    return decorator
