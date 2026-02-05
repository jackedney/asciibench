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

    def decorator(func):
        is_async = inspect.iscoroutinefunction(func)

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
