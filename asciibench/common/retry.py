"""Retry decorator with exponential backoff for transient failures."""

import asyncio
import functools
import inspect
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, cast

from asciibench.common.logging import get_logger

logger = get_logger(__name__)

# Type alias for sleep functions - accepts any callable that takes a float
# and returns either None (sync) or an Awaitable[None] (async)
SleepFunc = Callable[[float], None | Awaitable[None]]


@dataclass
class AttemptHistory:
    """History of a single retry attempt."""

    attempt_number: int
    exception: Exception
    delay_seconds: float | None


class MaxRetriesError(Exception):
    """Raised when all retry attempts are exhausted.

    Attributes:
        max_attempts: Maximum number of attempts that were tried
        last_exception: The exception that caused the final failure
        attempt_history: List of AttemptHistory objects for all attempts
    """

    def __init__(
        self,
        max_attempts: int,
        last_exception: Exception,
        attempt_history: list[AttemptHistory],
    ) -> None:
        self.max_attempts = max_attempts
        self.last_exception = last_exception
        self.attempt_history = attempt_history
        super().__init__(
            f"Max retries ({max_attempts}) exceeded. "
            f"Last error: {type(last_exception).__name__}: {last_exception}"
        )


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
        """Execute sleep for sync context, handling async sleep functions."""
        if custom_sleep is None:
            time.sleep(delay)
        else:
            result = custom_sleep(delay)
            if inspect.isawaitable(result):
                # Handle async sleep function in sync context
                coro = cast(Coroutine[Any, Any, None], result)
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None:
                    # We're in an async context but called synchronously - run in executor
                    loop.run_until_complete(coro)
                else:
                    # No running loop, create one to run the coroutine
                    asyncio.run(coro)

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
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}",
                            metadata={"function": func.__name__, "max_retries": max_retries},
                        )
                        raise

                    delay = base_delay_seconds * (2**attempt)
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{max_retries} for {func.__name__}",
                        metadata={"attempt": attempt + 1, "delay": delay, "error": str(e)},
                    )
                    _sync_sleep(delay, sleep_func)
                except Exception as e:
                    # Non-retryable exception
                    logger.error(
                        f"Non-retryable exception in {func.__name__}: {type(e).__name__}",
                        metadata={"function": func.__name__, "error": str(e)},
                    )
                    raise

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
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}",
                            metadata={"function": func.__name__, "max_retries": max_retries},
                        )
                        raise

                    delay = base_delay_seconds * (2**attempt)
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{max_retries} for {func.__name__}",
                        metadata={"attempt": attempt + 1, "delay": delay, "error": str(e)},
                    )
                    await _async_sleep(delay, sleep_func)
                except Exception as e:
                    # Non-retryable exception
                    logger.error(
                        f"Non-retryable exception in {func.__name__}: {type(e).__name__}",
                        metadata={"function": func.__name__, "error": str(e)},
                    )
                    raise

            # This should never be reached, but type checkers need it
            if last_exception is None:
                raise RuntimeError("Unexpected retry loop exit without exception")
            raise last_exception

        return async_wrapper if inspect.iscoroutinefunction(func) else wrapper

    return decorator


class RetryableTaskExecutor:
    """Class-based retry executor with exponential backoff.

    Provides an alternative to the @retry decorator for scenarios where
    a class-based approach is preferred (e.g., dynamic retry configuration,
    retry state tracking, or dependency injection).

    Example:
        executor = RetryableTaskExecutor(
            max_attempts=3,
            base_delay_seconds=1,
            retryable_exceptions=(RateLimitError, TransientError),
        )
        result = executor.execute(api_call, arg1, arg2)

    Example (async):
        executor = RetryableTaskExecutor(max_attempts=3)
        result = await executor.execute_async(async_api_call, arg1, arg2)

    Attributes:
        max_attempts: Maximum number of execution attempts (including first)
        base_delay_seconds: Base delay in seconds before first retry
        retryable_exceptions: Tuple of exception types to retry
        sleep_func: Optional custom sleep function for testability
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay_seconds: float = 1,
        retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
        sleep_func: SleepFunc | None = None,
    ) -> None:
        """Initialize the retry executor.

        Args:
            max_attempts: Maximum number of attempts (default: 3)
            base_delay_seconds: Base delay before first retry in seconds (default: 1)
            retryable_exceptions: Exception types to retry (default: Exception)
            sleep_func: Optional sleep function for testability

        Raises:
            ValueError: If max_attempts < 1 or base_delay_seconds < 0
            TypeError: If retryable_exceptions is not a tuple of Exception types
        """
        if not isinstance(max_attempts, int):
            raise ValueError(f"max_attempts must be an integer, got {type(max_attempts).__name__}")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
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
                raise TypeError(
                    f"retryable_exceptions must contain Exception types, got {exc_type}"
                )

        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.retryable_exceptions = retryable_exceptions
        self.sleep_func = sleep_func

    def execute(self, func: Callable, *args, **kwargs):
        """Execute a synchronous function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            Result of func(*args, **kwargs) on success

        Raises:
            MaxRetriesError: If all retry attempts are exhausted
            Exception: If an exception not in retryable_exceptions is raised
        """
        attempt_history: list[AttemptHistory] = []

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except self.retryable_exceptions as e:
                is_final_attempt = attempt == self.max_attempts - 1

                if is_final_attempt:
                    # Final attempt - no delay, record and raise
                    attempt_history.append(
                        AttemptHistory(
                            attempt_number=attempt + 1,
                            exception=e,  # type: ignore[arg-type]
                            delay_seconds=None,
                        )
                    )
                    logger.warning(f"Function failed after {self.max_attempts} attempts")
                    raise MaxRetriesError(
                        max_attempts=self.max_attempts,
                        last_exception=e,  # type: ignore[arg-type]
                        attempt_history=attempt_history,
                    ) from e

                # Non-final attempt - calculate delay, record, sleep, and retry
                delay = self.base_delay_seconds * (2**attempt)
                attempt_history.append(
                    AttemptHistory(
                        attempt_number=attempt + 1,
                        exception=e,  # type: ignore[arg-type]
                        delay_seconds=delay,
                    )
                )

                logger.debug(
                    f"Attempt {attempt + 1}/{self.max_attempts} failed, "
                    f"retrying after {delay}s: {e}"
                )

                if self.sleep_func is None:
                    time.sleep(delay)
                else:
                    result = self.sleep_func(delay)
                    if inspect.isawaitable(result):
                        # Handle async sleep function in sync context
                        coro = cast(Coroutine[Any, Any, None], result)
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            loop = None
                        if loop is not None:
                            loop.run_until_complete(coro)
                        else:
                            asyncio.run(coro)

        raise RuntimeError("Unexpected retry loop exit without exception")

    async def execute_async(self, func: Callable, *args, **kwargs):
        """Execute an async function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            Result of await func(*args, **kwargs) on success

        Raises:
            MaxRetriesError: If all retry attempts are exhausted
            Exception: If an exception not in retryable_exceptions is raised
        """
        attempt_history: list[AttemptHistory] = []

        for attempt in range(self.max_attempts):
            try:
                return await func(*args, **kwargs)
            except self.retryable_exceptions as e:
                is_final_attempt = attempt == self.max_attempts - 1

                if is_final_attempt:
                    # Final attempt - no delay, record and raise
                    attempt_history.append(
                        AttemptHistory(
                            attempt_number=attempt + 1,
                            exception=e,  # type: ignore[arg-type]
                            delay_seconds=None,
                        )
                    )
                    logger.warning(f"Async function failed after {self.max_attempts} attempts")
                    raise MaxRetriesError(
                        max_attempts=self.max_attempts,
                        last_exception=e,  # type: ignore[arg-type]
                        attempt_history=attempt_history,
                    ) from e

                # Non-final attempt - calculate delay, record, sleep, and retry
                delay = self.base_delay_seconds * (2**attempt)
                attempt_history.append(
                    AttemptHistory(
                        attempt_number=attempt + 1,
                        exception=e,  # type: ignore[arg-type]
                        delay_seconds=delay,
                    )
                )

                logger.debug(
                    f"Async attempt {attempt + 1}/{self.max_attempts} failed, "
                    f"retrying after {delay}s: {e}"
                )

                if self.sleep_func is None:
                    await asyncio.sleep(delay)
                else:
                    result = self.sleep_func(delay)
                    if inspect.isawaitable(result):
                        await result

        raise RuntimeError("Unexpected retry loop exit without exception")
