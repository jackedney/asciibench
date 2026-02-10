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


def _validate_retry_config(
    max_retries: int,
    base_delay_seconds: float,
    retryable_exceptions: tuple[type[Exception], ...],
    min_max_retries: int = 0,
    param_name: str = "max_retries",
) -> None:
    """Validate retry configuration parameters.

    Args:
        max_retries: Maximum number of retries
        base_delay_seconds: Base delay before first retry in seconds
        retryable_exceptions: Tuple of exception types to retry
        min_max_retries: Minimum allowed value for max_retries (default: 0 for decorator)
        param_name: Name of the parameter for error messages (default: "max_retries")

    Raises:
        ValueError: If max_retries or base_delay_seconds are invalid
        TypeError: If retryable_exceptions is invalid
    """
    if not isinstance(max_retries, int):
        raise ValueError(f"{param_name} must be an integer, got {type(max_retries).__name__}")
    if max_retries < min_max_retries:
        raise ValueError(f"{param_name} must be >= {min_max_retries}, got {max_retries}")
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
    """Execute sleep, handling both sync and async sleep functions.

    Args:
        delay: Delay in seconds
        custom_sleep: Optional custom sleep function

    Raises:
        RuntimeError: If a non-awaitable sleep function is called in async context
    """
    if custom_sleep is None:
        await asyncio.sleep(delay)
    else:
        result = custom_sleep(delay)
        if inspect.isawaitable(result):
            await result


def _sync_sleep(delay: float, custom_sleep: SleepFunc | None) -> None:
    """Execute sleep for sync context, handling async sleep functions.

    Args:
        delay: Delay in seconds
        custom_sleep: Optional custom sleep function

    Raises:
        RuntimeError: If called from async context with coroutine sleep function or
                    if a non-coroutine awaitable is provided
    """
    if custom_sleep is None:
        time.sleep(delay)
    else:
        result = custom_sleep(delay)
        if inspect.iscoroutine(result):
            coro = cast(Coroutine[Any, Any, None], result)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                coro.close()
                raise RuntimeError(
                    "Sync retry path was called from an async context with a coroutine "
                    "sleep function. Use the async retry wrapper or execute_async method instead."
                )
            else:
                asyncio.run(coro)
        elif inspect.isawaitable(result):
            raise RuntimeError(
                "Sync retry path received a non-coroutine awaitable (e.g., "
                "Future, Task) which is not supported. Use async retry wrapper "
                "or execute_async method instead."
            )


def _calculate_delay(attempt: int, base_delay_seconds: float) -> float:
    """Calculate delay for a retry attempt using exponential backoff.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay_seconds: Base delay before first retry in seconds

    Returns:
        Delay in seconds for this retry
    """
    return base_delay_seconds * (2**attempt)


def _retry_sync_core(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    max_attempts: int,
    base_delay_seconds: float,
    retryable_exceptions: tuple[type[Exception], ...],
    sleep_func: SleepFunc | None,
    func_name: str | None = None,
    raise_max_retries_error: bool = False,
) -> Any:
    """Core sync retry logic shared by sync decorator and executor.

    Args:
        func: Function to execute
        args: Positional arguments to pass to func
        kwargs: Keyword arguments to pass to func
        max_attempts: Maximum number of execution attempts (including first)
        base_delay_seconds: Base delay before first retry in seconds
        retryable_exceptions: Exception types to retry
        sleep_func: Optional custom sleep function
        func_name: Optional function name for logging (decorator use case)
        raise_max_retries_error: True to raise MaxRetriesError, False to raise original exception

    Returns:
        Result of func(*args, **kwargs) on success

    Raises:
        MaxRetriesError: If raise_max_retries_error is True and all attempts exhausted
        Exception: If raise_max_retries_error is False and all attempts exhausted,
                   or if a non-retryable exception is raised
    """
    attempt_history: list[AttemptHistory] = []

    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except retryable_exceptions as e:
            is_final_attempt = attempt == max_attempts - 1

            if is_final_attempt:
                attempt_history.append(
                    AttemptHistory(
                        attempt_number=attempt + 1,
                        exception=e,
                        delay_seconds=None,
                    )
                )
                if raise_max_retries_error:
                    logger.warning(f"Function failed after {max_attempts} attempts")
                    raise MaxRetriesError(
                        max_attempts=max_attempts,
                        last_exception=e,
                        attempt_history=attempt_history,
                    ) from e
                else:
                    logger.error(
                        f"Max retries ({max_attempts - 1}) exceeded for {func_name or 'function'}",
                        metadata={
                            "function": func_name or "function",
                            "max_retries": max_attempts - 1,
                        },
                    )
                    raise

            delay = _calculate_delay(attempt, base_delay_seconds)
            attempt_history.append(
                AttemptHistory(
                    attempt_number=attempt + 1,
                    exception=e,
                    delay_seconds=delay,
                )
            )

            if func_name:
                logger.warning(
                    f"Retry attempt {attempt + 1}/{max_attempts - 1} for {func_name}",
                    metadata={"attempt": attempt + 1, "delay": delay, "error": str(e)},
                )
            else:
                logger.debug(
                    f"Attempt {attempt + 1}/{max_attempts} failed, retrying after {delay}s: {e}"
                )

            _sync_sleep(delay, sleep_func)
        except Exception as e:
            logger.error(
                f"Non-retryable exception in {func_name or 'function'}: {type(e).__name__}",
                metadata={"function": func_name or "function", "error": str(e)},
            )
            raise

    raise RuntimeError("Unexpected retry loop exit without exception")


async def _retry_async_core(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    max_attempts: int,
    base_delay_seconds: float,
    retryable_exceptions: tuple[type[Exception], ...],
    sleep_func: SleepFunc | None,
    func_name: str | None = None,
    raise_max_retries_error: bool = False,
) -> Any:
    """Core async retry logic shared by async decorator and executor.

    Args:
        func: Async function to execute
        args: Positional arguments to pass to func
        kwargs: Keyword arguments to pass to func
        max_attempts: Maximum number of execution attempts (including first)
        base_delay_seconds: Base delay before first retry in seconds
        retryable_exceptions: Exception types to retry
        sleep_func: Optional custom sleep function
        func_name: Optional function name for logging (decorator use case)
        raise_max_retries_error: True to raise MaxRetriesError, False to raise original exception

    Returns:
        Result of await func(*args, **kwargs) on success

    Raises:
        MaxRetriesError: If raise_max_retries_error is True and all attempts exhausted
        Exception: If raise_max_retries_error is False and all attempts exhausted,
                   or if a non-retryable exception is raised
    """
    attempt_history: list[AttemptHistory] = []

    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            is_final_attempt = attempt == max_attempts - 1

            if is_final_attempt:
                attempt_history.append(
                    AttemptHistory(
                        attempt_number=attempt + 1,
                        exception=e,
                        delay_seconds=None,
                    )
                )
                if raise_max_retries_error:
                    logger.warning(f"Async function failed after {max_attempts} attempts")
                    raise MaxRetriesError(
                        max_attempts=max_attempts,
                        last_exception=e,
                        attempt_history=attempt_history,
                    ) from e
                else:
                    logger.error(
                        f"Max retries ({max_attempts - 1}) exceeded for {func_name or 'function'}",
                        metadata={
                            "function": func_name or "function",
                            "max_retries": max_attempts - 1,
                        },
                    )
                    raise

            delay = _calculate_delay(attempt, base_delay_seconds)
            attempt_history.append(
                AttemptHistory(
                    attempt_number=attempt + 1,
                    exception=e,
                    delay_seconds=delay,
                )
            )

            if func_name:
                logger.warning(
                    f"Retry attempt {attempt + 1}/{max_attempts - 1} for {func_name}",
                    metadata={"attempt": attempt + 1, "delay": delay, "error": str(e)},
                )
            else:
                logger.debug(
                    f"Async attempt {attempt + 1}/{max_attempts} failed, "
                    f"retrying after {delay}s: {e}"
                )

            await _async_sleep(delay, sleep_func)
        except Exception as e:
            logger.error(
                f"Non-retryable exception in {func_name or 'function'}: {type(e).__name__}",
                metadata={"function": func_name or "function", "error": str(e)},
            )
            raise

    raise RuntimeError("Unexpected retry loop exit without exception")


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
    _validate_retry_config(
        max_retries,
        base_delay_seconds,
        retryable_exceptions,
        min_max_retries=0,
        param_name="max_retries",
    )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _retry_sync_core(
                func=func,
                args=args,
                kwargs=kwargs,
                max_attempts=max_retries + 1,
                base_delay_seconds=base_delay_seconds,
                retryable_exceptions=retryable_exceptions,
                sleep_func=sleep_func,
                func_name=func.__name__,
                raise_max_retries_error=False,
            )

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await _retry_async_core(
                func=func,
                args=args,
                kwargs=kwargs,
                max_attempts=max_retries + 1,
                base_delay_seconds=base_delay_seconds,
                retryable_exceptions=retryable_exceptions,
                sleep_func=sleep_func,
                func_name=func.__name__,
                raise_max_retries_error=False,
            )

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
        _validate_retry_config(
            max_attempts,
            base_delay_seconds,
            retryable_exceptions,
            min_max_retries=1,
            param_name="max_attempts",
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
        return _retry_sync_core(
            func=func,
            args=args,
            kwargs=kwargs,
            max_attempts=self.max_attempts,
            base_delay_seconds=self.base_delay_seconds,
            retryable_exceptions=self.retryable_exceptions,
            sleep_func=self.sleep_func,
            func_name=None,
            raise_max_retries_error=True,
        )

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
        return await _retry_async_core(
            func=func,
            args=args,
            kwargs=kwargs,
            max_attempts=self.max_attempts,
            base_delay_seconds=self.base_delay_seconds,
            retryable_exceptions=self.retryable_exceptions,
            sleep_func=self.sleep_func,
            func_name=None,
            raise_max_retries_error=True,
        )
