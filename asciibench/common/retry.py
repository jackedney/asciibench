"""Retry decorator with exponential backoff for transient failures."""

import asyncio
import functools
import inspect
import time
from typing import Any

from asciibench.common.logging import get_logger

logger = get_logger(__name__)


def retry(
    max_retries: int = 3,
    base_delay_seconds: float = 1,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    sleep_func: Any = None,
):
    """Decorator to retry function calls on specified exceptions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay_seconds: Base delay in seconds before first retry (default: 1)
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

    def decorator(func):
        is_async = inspect.iscoroutinefunction(func)

        # Validate sleep_func matches async/sync nature of decorated function
        if sleep_func is not None:
            sleep_func_is_async = inspect.iscoroutinefunction(sleep_func)
            if is_async and not sleep_func_is_async:
                raise TypeError("sleep_func must be async when decorating async functions")
            if not is_async and sleep_func_is_async:
                raise TypeError("sleep_func must be sync when decorating sync functions")

        if is_async:
            if sleep_func is None:
                actual_sleep_func = asyncio.sleep
            else:
                actual_sleep_func = sleep_func

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None

                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except retryable_exceptions as e:
                        last_exception = e

                        if attempt < max_retries:
                            delay = base_delay_seconds * (2**attempt)
                            logger.warning(
                                f"Retry attempt {attempt + 1}/{max_retries} after {delay:.2f}s",
                                {
                                    "function": func.__name__,
                                    "attempt": attempt + 1,
                                    "delay_seconds": delay,
                                    "exception": str(e),
                                },
                            )
                            await actual_sleep_func(delay)  # type: ignore[misc]
                        else:
                            logger.error(
                                f"Max retries ({max_retries}) exceeded for {func.__name__}",
                                {"function": func.__name__, "exception": str(e)},
                            )
                            raise
                    except Exception as e:
                        logger.error(
                            f"Non-retryable exception in {func.__name__}",
                            {"function": func.__name__, "exception": str(e)},
                        )
                        raise

                if last_exception:
                    raise last_exception

            return async_wrapper
        else:
            if sleep_func is None:
                actual_sleep_func = time.sleep
            else:
                actual_sleep_func = sleep_func

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None

                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except retryable_exceptions as e:
                        last_exception = e

                        if attempt < max_retries:
                            delay = base_delay_seconds * (2**attempt)
                            logger.warning(
                                f"Retry attempt {attempt + 1}/{max_retries} after {delay:.2f}s",
                                {
                                    "function": func.__name__,
                                    "attempt": attempt + 1,
                                    "delay_seconds": delay,
                                    "exception": str(e),
                                },
                            )
                            actual_sleep_func(delay)
                        else:
                            logger.error(
                                f"Max retries ({max_retries}) exceeded for {func.__name__}",
                                {"function": func.__name__, "exception": str(e)},
                            )
                            raise
                    except Exception as e:
                        logger.error(
                            f"Non-retryable exception in {func.__name__}",
                            {"function": func.__name__, "exception": str(e)},
                        )
                        raise

                if last_exception:
                    raise last_exception

            return wrapper

    return decorator
