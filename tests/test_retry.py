"""Tests for retry decorator with exponential backoff."""

import asyncio
import json
import time
from pathlib import Path

import pytest

from asciibench.common.logging import JSONLogger
from asciibench.common.retry import retry


class CustomRetryableError(Exception):
    """Custom exception for testing retry logic."""

    pass


class NonRetryableError(Exception):
    """Custom exception for testing non-retryable scenarios."""

    pass


class TestRetryDecorator:
    """Tests for retry decorator."""

    def test_succeeds_on_first_attempt(self) -> None:
        """Function succeeds on first attempt without retries."""
        attempt_count = 0

        @retry(max_retries=3, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        def successful_call():
            nonlocal attempt_count
            attempt_count += 1
            return "success"

        result = successful_call()

        assert result == "success"
        assert attempt_count == 1

    def test_retries_on_specified_exception(self) -> None:
        """Function retries when specified exception is raised."""
        attempt_count = 0

        @retry(max_retries=3, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        def failing_call():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise CustomRetryableError("Temporary error")
            return "success"

        result = failing_call()

        assert result == "success"
        assert attempt_count == 3

    def test_raises_after_max_retries(self) -> None:
        """Function raises exception after max retries exceeded."""
        attempt_count = 0

        @retry(max_retries=2, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        def always_failing_call():
            nonlocal attempt_count
            attempt_count += 1
            raise CustomRetryableError("Persistent error")

        with pytest.raises(CustomRetryableError, match="Persistent error"):
            always_failing_call()

        assert attempt_count == 3  # Initial attempt + 2 retries

    def test_uses_exponential_backoff(self) -> None:
        """Retry delays follow exponential backoff pattern."""
        attempt_count = 0
        timestamps = []

        @retry(max_retries=3, base_delay_seconds=0.1, retryable_exceptions=(CustomRetryableError,))
        def failing_call():
            nonlocal attempt_count
            attempt_count += 1
            timestamps.append(time.time())
            if attempt_count < 4:
                raise CustomRetryableError("Temporary error")
            return "success"

        failing_call()

        assert len(timestamps) == 4
        delays = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

        assert delays[0] >= 0.1 * 0.8  # Allow some tolerance
        assert delays[1] >= 0.2 * 0.8  # 0.1 * 2^1
        assert delays[2] >= 0.4 * 0.8  # 0.1 * 2^2

    def test_default_max_retries(self) -> None:
        """Default max_retries is 3."""
        attempt_count = 0

        @retry(retryable_exceptions=(CustomRetryableError,))
        def failing_call():
            nonlocal attempt_count
            attempt_count += 1
            raise CustomRetryableError("Error")

        with pytest.raises(CustomRetryableError):
            failing_call()

        assert attempt_count == 4  # Initial attempt + 3 retries

    def test_default_base_delay_seconds(self) -> None:
        """Default base_delay_seconds is 1."""
        attempt_count = 0
        timestamps = []

        @retry(max_retries=2, retryable_exceptions=(CustomRetryableError,))
        def failing_call():
            nonlocal attempt_count
            attempt_count += 1
            timestamps.append(time.time())
            if attempt_count < 3:
                raise CustomRetryableError("Error")
            return "success"

        failing_call()

        assert len(timestamps) == 3
        delays = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

        assert delays[0] >= 1.0 * 0.8
        assert delays[1] >= 2.0 * 0.8  # 1.0 * 2^1

    def test_non_retryable_exception_raises_immediately(self) -> None:
        """Non-retryable exception raises immediately without retry."""
        attempt_count = 0

        @retry(max_retries=3, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        def non_retryable_call():
            nonlocal attempt_count
            attempt_count += 1
            raise NonRetryableError("Non-retryable error")

        with pytest.raises(NonRetryableError, match="Non-retryable error"):
            non_retryable_call()

        assert attempt_count == 1  # Only initial attempt, no retries

    def test_multiple_retryable_exception_types(self) -> None:
        """Retry works with multiple exception types."""
        attempt_count = 0

        @retry(
            max_retries=3,
            base_delay_seconds=0.01,
            retryable_exceptions=(CustomRetryableError, ValueError),
        )
        def multi_exception_call():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise CustomRetryableError("First error")
            if attempt_count == 2:
                raise ValueError("Second error")
            return "success"

        result = multi_exception_call()

        assert result == "success"
        assert attempt_count == 3

    def test_preserves_function_name_and_docstring(self) -> None:
        """Decorator preserves original function's metadata."""

        @retry(max_retries=3, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        def my_function():
            """My function docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My function docstring."

    def test_passes_arguments_and_returns_result(self) -> None:
        """Decorator correctly passes arguments and returns result."""

        @retry(max_retries=3, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)

        assert result == 5

    def test_passes_keyword_arguments(self) -> None:
        """Decorator correctly passes keyword arguments."""

        @retry(max_retries=3, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = greet("World", greeting="Hi")

        assert result == "Hi, World!"

    def test_logs_retry_attempts(self, tmp_path: Path, monkeypatch) -> None:
        """Retry attempts are logged with attempt number and delay."""
        log_path = tmp_path / "retry_test.jsonl"
        test_logger = JSONLogger("asciibench.common.retry", log_path)
        attempt_count = 0

        monkeypatch.setattr("asciibench.common.retry.logger", test_logger)

        @retry(max_retries=3, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        def failing_call():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise CustomRetryableError("Temporary error")
            return "success"

        failing_call()

        assert attempt_count == 3

        log_content = log_path.read_text()
        log_lines = [json.loads(line) for line in log_content.strip().split("\n") if line]

        retry_logs = [entry for entry in log_lines if "Retry attempt" in entry["message"]]
        assert len(retry_logs) == 2
        assert retry_logs[0]["level"] == "warning"
        assert "attempt" in retry_logs[0]["metadata"]
        assert retry_logs[0]["metadata"]["attempt"] == 1

    def test_logs_max_retries_exceeded(self, tmp_path: Path, monkeypatch) -> None:
        """Max retries exceeded is logged."""
        log_path = tmp_path / "retry_max_exceeded.jsonl"
        test_logger = JSONLogger("asciibench.common.retry", log_path)

        monkeypatch.setattr("asciibench.common.retry.logger", test_logger)

        @retry(max_retries=2, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        def always_failing_call():
            raise CustomRetryableError("Persistent error")

        with pytest.raises(CustomRetryableError):
            always_failing_call()

        log_content = log_path.read_text()
        log_lines = [json.loads(line) for line in log_content.strip().split("\n") if line]

        max_retries_logs = [
            entry
            for entry in log_lines
            if "Max retries" in entry["message"] and "exceeded" in entry["message"]
        ]
        assert len(max_retries_logs) == 1
        assert max_retries_logs[0]["level"] == "error"

    def test_logs_non_retryable_exception(self, tmp_path: Path, monkeypatch) -> None:
        """Non-retryable exceptions are logged."""
        log_path = tmp_path / "retry_non_retryable.jsonl"
        test_logger = JSONLogger("asciibench.common.retry", log_path)

        monkeypatch.setattr("asciibench.common.retry.logger", test_logger)

        @retry(max_retries=3, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        def non_retryable_call():
            raise NonRetryableError("Non-retryable")

        with pytest.raises(NonRetryableError):
            non_retryable_call()

        log_content = log_path.read_text()
        log_lines = [json.loads(line) for line in log_content.strip().split("\n") if line]

        non_retryable_logs = [
            entry for entry in log_lines if "Non-retryable exception" in entry["message"]
        ]
        assert len(non_retryable_logs) == 1
        assert non_retryable_logs[0]["level"] == "error"

    def test_zero_retries(self) -> None:
        """Zero retries means function only runs once."""
        attempt_count = 0

        @retry(max_retries=0, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        def failing_call():
            nonlocal attempt_count
            attempt_count += 1
            raise CustomRetryableError("Error")

        with pytest.raises(CustomRetryableError):
            failing_call()

        assert attempt_count == 1

    def test_zero_base_delay(self) -> None:
        """Zero base delay means no waiting between retries."""
        attempt_count = 0

        @retry(max_retries=3, base_delay_seconds=0, retryable_exceptions=(CustomRetryableError,))
        def failing_call():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 4:
                raise CustomRetryableError("Error")
            return "success"

        start_time = time.time()
        failing_call()
        elapsed_time = time.time() - start_time

        assert attempt_count == 4
        assert elapsed_time < 0.5  # Should complete quickly without delays


class TestAsyncRetryDecorator:
    """Tests for retry decorator with async functions."""

    @pytest.mark.asyncio
    async def test_async_succeeds_on_first_attempt(self) -> None:
        """Async function succeeds on first attempt without retries."""
        attempt_count = 0

        @retry(max_retries=3, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        async def successful_call():
            nonlocal attempt_count
            attempt_count += 1
            return "success"

        result = await successful_call()

        assert result == "success"
        assert attempt_count == 1

    @pytest.mark.asyncio
    async def test_async_retries_on_specified_exception(self) -> None:
        """Async function retries when specified exception is raised."""
        attempt_count = 0

        @retry(max_retries=3, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        async def failing_call():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise CustomRetryableError("Temporary error")
            return "success"

        result = await failing_call()

        assert result == "success"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_async_raises_after_max_retries(self) -> None:
        """Async function raises exception after max retries exceeded."""
        attempt_count = 0

        @retry(max_retries=2, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        async def always_failing_call():
            nonlocal attempt_count
            attempt_count += 1
            raise CustomRetryableError("Persistent error")

        with pytest.raises(CustomRetryableError, match="Persistent error"):
            await always_failing_call()

        assert attempt_count == 3  # Initial attempt + 2 retries

    @pytest.mark.asyncio
    async def test_async_uses_exponential_backoff(self) -> None:
        """Async retry delays follow exponential backoff pattern."""
        attempt_count = 0
        timestamps = []

        @retry(max_retries=3, base_delay_seconds=0.1, retryable_exceptions=(CustomRetryableError,))
        async def failing_call():
            nonlocal attempt_count
            attempt_count += 1
            timestamps.append(time.time())
            if attempt_count < 4:
                raise CustomRetryableError("Temporary error")
            return "success"

        await failing_call()

        assert len(timestamps) == 4
        delays = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

        assert delays[0] >= 0.1 * 0.8  # Allow some tolerance
        assert delays[1] >= 0.2 * 0.8  # 0.1 * 2^1
        assert delays[2] >= 0.4 * 0.8  # 0.1 * 2^2

    @pytest.mark.asyncio
    async def test_async_non_retryable_exception_raises_immediately(self) -> None:
        """Non-retryable exception raises immediately without retry."""
        attempt_count = 0

        @retry(max_retries=3, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        async def non_retryable_call():
            nonlocal attempt_count
            attempt_count += 1
            raise NonRetryableError("Non-retryable error")

        with pytest.raises(NonRetryableError, match="Non-retryable error"):
            await non_retryable_call()

        assert attempt_count == 1  # Only initial attempt, no retries

    @pytest.mark.asyncio
    async def test_async_preserves_function_name_and_docstring(self) -> None:
        """Decorator preserves original async function's metadata."""

        @retry(max_retries=3, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        async def my_function():
            """My function docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My function docstring."

    @pytest.mark.asyncio
    async def test_async_passes_arguments_and_returns_result(self) -> None:
        """Decorator correctly passes arguments and returns result."""

        @retry(max_retries=3, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        async def add(a: int, b: int) -> int:
            return a + b

        result = await add(2, 3)

        assert result == 5

    @pytest.mark.asyncio
    async def test_async_passes_keyword_arguments(self) -> None:
        """Decorator correctly passes keyword arguments."""

        @retry(max_retries=3, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        async def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = await greet("World", greeting="Hi")

        assert result == "Hi, World!"

    @pytest.mark.asyncio
    async def test_async_with_custom_sleep_func(self) -> None:
        """Async decorator uses custom sleep function when provided."""
        sleep_calls = []

        async def custom_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        attempt_count = 0

        @retry(
            max_retries=2,
            base_delay_seconds=0.1,
            retryable_exceptions=(CustomRetryableError,),
            sleep_func=custom_sleep,
        )
        async def failing_call():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise CustomRetryableError("Temporary error")
            return "success"

        await failing_call()

        assert attempt_count == 3
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == 0.1  # First retry delay
        assert sleep_calls[1] == 0.2  # Second retry delay (exponential backoff)

    @pytest.mark.asyncio
    async def test_async_zero_retries(self) -> None:
        """Zero retries means async function only runs once."""
        attempt_count = 0

        @retry(max_retries=0, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        async def failing_call():
            nonlocal attempt_count
            attempt_count += 1
            raise CustomRetryableError("Error")

        with pytest.raises(CustomRetryableError):
            await failing_call()

        assert attempt_count == 1

    @pytest.mark.asyncio
    async def test_async_zero_base_delay(self) -> None:
        """Zero base delay means no waiting between retries."""
        attempt_count = 0

        @retry(max_retries=3, base_delay_seconds=0, retryable_exceptions=(CustomRetryableError,))
        async def failing_call():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 4:
                raise CustomRetryableError("Error")
            return "success"

        start_time = time.time()
        await failing_call()
        elapsed_time = time.time() - start_time

        assert attempt_count == 4
        assert elapsed_time < 0.5  # Should complete quickly without delays

    @pytest.mark.asyncio
    async def test_sync_decorator_on_sync_function_still_works(self) -> None:
        """Negative case: @retry() on sync def bar() still works as before."""
        attempt_count = 0

        @retry(max_retries=2, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        def sync_call():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise CustomRetryableError("Temporary error")
            return "success"

        result = sync_call()

        assert result == "success"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_async_awaits_func_correctly(self) -> None:
        """Async wrapper correctly awaits func(*args, **kwargs)."""

        @retry(max_retries=1, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        async def async_call():
            await asyncio.sleep(0)
            return "awaited"

        result = await async_call()

        assert result == "awaited"

    @pytest.mark.asyncio
    async def test_async_multiple_retryable_exception_types(self) -> None:
        """Async retry works with multiple exception types."""
        attempt_count = 0

        @retry(
            max_retries=3,
            base_delay_seconds=0.01,
            retryable_exceptions=(CustomRetryableError, ValueError),
        )
        async def multi_exception_call():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise CustomRetryableError("First error")
            if attempt_count == 2:
                raise ValueError("Second error")
            return "success"

        result = await multi_exception_call()

        assert result == "success"
        assert attempt_count == 3


class TestRetryValidation:
    """Tests for input validation of retry decorator parameters."""

    def test_negative_max_retries_raises_value_error(self) -> None:
        """Example: retry(max_retries=-1) raises ValueError with clear message."""
        with pytest.raises(ValueError, match="max_retries must be >= 0"):

            @retry(max_retries=-1)
            def dummy_func():
                pass

    def test_string_max_retries_raises_value_error(self) -> None:
        """Invalid max_retries type raises ValueError."""
        with pytest.raises(ValueError, match="max_retries must be an integer"):

            @retry(max_retries="3")  # type: ignore[invalid-argument-type]
            def dummy_func():
                pass

    def test_float_max_retries_raises_value_error(self) -> None:
        """Float max_retries raises ValueError."""
        with pytest.raises(ValueError, match="max_retries must be an integer"):

            @retry(max_retries=3.5)  # type: ignore[invalid-argument-type]
            def dummy_func():
                pass

    def test_negative_base_delay_raises_value_error(self) -> None:
        """Negative base_delay_seconds raises ValueError."""
        with pytest.raises(ValueError, match="base_delay_seconds must be >= 0"):

            @retry(base_delay_seconds=-1)
            def dummy_func():
                pass

    def test_string_base_delay_raises_value_error(self) -> None:
        """Example: retry(base_delay_seconds='invalid') raises ValueError."""
        with pytest.raises(ValueError, match="base_delay_seconds must be a number"):

            @retry(base_delay_seconds="invalid")  # type: ignore[invalid-argument-type]
            def dummy_func():
                pass

    def test_zero_max_retries_is_valid(self) -> None:
        """Negative case: retry(max_retries=0) is valid (no retries)."""

        @retry(max_retries=0, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        def failing_call():
            raise CustomRetryableError("Error")

        with pytest.raises(CustomRetryableError):
            failing_call()

    def test_zero_base_delay_is_valid(self) -> None:
        """Zero base delay is valid."""

        @retry(max_retries=0, base_delay_seconds=0, retryable_exceptions=(CustomRetryableError,))
        def dummy_func():
            return "success"

        result = dummy_func()
        assert result == "success"

    def test_empty_retryable_exceptions_raises_type_error(self) -> None:
        """Empty retryable_exceptions tuple raises TypeError."""
        with pytest.raises(TypeError, match="retryable_exceptions must not be empty"):

            @retry(retryable_exceptions=())
            def dummy_func():
                pass

    def test_list_retryable_exceptions_raises_type_error(self) -> None:
        """List instead of tuple raises TypeError."""
        with pytest.raises(TypeError, match="retryable_exceptions must be a tuple"):

            @retry(retryable_exceptions=[Exception])  # type: ignore[invalid-argument-type]
            def dummy_func():
                pass

    def test_non_exception_type_in_retryable_exceptions_raises_type_error(self) -> None:
        """Non-Exception type in retryable_exceptions raises TypeError."""
        with pytest.raises(TypeError, match="retryable_exceptions must contain Exception types"):

            @retry(retryable_exceptions=(Exception, "not_an_exception"))  # type: ignore[invalid-argument-type]
            def dummy_func():
                pass

    def test_non_class_in_retryable_exceptions_raises_type_error(self) -> None:
        """Non-class in retryable_exceptions raises TypeError."""
        with pytest.raises(TypeError, match="retryable_exceptions must contain Exception types"):

            @retry(retryable_exceptions=(Exception, 123))  # type: ignore[invalid-argument-type]
            def dummy_func():
                pass

    def test_valid_retryable_exceptions_tuple(self) -> None:
        """Valid tuple of exception types works correctly."""

        @retry(retryable_exceptions=(CustomRetryableError, ValueError))
        def dummy_func():
            return "success"

        result = dummy_func()
        assert result == "success"

    def test_sync_sleep_func_with_sync_function_works(self) -> None:
        """Example: decorating sync function with time.sleep works."""
        sleep_calls = []

        def custom_sync_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        attempt_count = 0

        @retry(
            max_retries=2,
            base_delay_seconds=0.1,
            retryable_exceptions=(CustomRetryableError,),
            sleep_func=custom_sync_sleep,
        )
        def sync_call():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise CustomRetryableError("Temporary error")
            return "success"

        result = sync_call()

        assert result == "success"
        assert attempt_count == 3
        assert len(sleep_calls) == 2

    def test_async_sleep_func_with_sync_function_raises_type_error(self) -> None:
        """Negative case: decorating sync function with asyncio.sleep raises TypeError."""
        with pytest.raises(
            TypeError, match="sleep_func must be sync when decorating sync functions"
        ):

            @retry(
                max_retries=2,
                base_delay_seconds=0.1,
                retryable_exceptions=(CustomRetryableError,),
                sleep_func=asyncio.sleep,
            )
            def sync_call():
                return "success"

    def test_sync_sleep_func_with_async_function_raises_type_error(self) -> None:
        """Negative case: decorating async function with time.sleep raises TypeError."""
        with pytest.raises(
            TypeError, match="sleep_func must be async when decorating async functions"
        ):

            @retry(
                max_retries=2,
                base_delay_seconds=0.1,
                retryable_exceptions=(CustomRetryableError,),
                sleep_func=time.sleep,
            )
            async def async_call():
                return "success"

    @pytest.mark.asyncio
    async def test_async_sleep_func_with_async_function_works(self) -> None:
        """Example: decorating async function with asyncio.sleep works."""
        sleep_calls = []

        async def custom_async_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        attempt_count = 0

        @retry(
            max_retries=2,
            base_delay_seconds=0.1,
            retryable_exceptions=(CustomRetryableError,),
            sleep_func=custom_async_sleep,
        )
        async def async_call():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise CustomRetryableError("Temporary error")
            return "success"

        result = await async_call()

        assert result == "success"
        assert attempt_count == 3
        assert len(sleep_calls) == 2


class TestRetryExamples:
    """Example tests from acceptance criteria."""

    def test_429_response_example(self) -> None:
        """Example: 429 response triggers retry after 1s, then 2s, then 4s."""
        sleep_calls = []

        def instant_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        attempt_count = 0

        @retry(
            max_retries=3,
            base_delay_seconds=1,
            retryable_exceptions=(CustomRetryableError,),
            sleep_func=instant_sleep,
        )
        def api_call():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count <= 3:
                raise CustomRetryableError("429 Rate limit exceeded")
            return "response"

        start_time = time.time()
        result = api_call()
        elapsed_time = time.time() - start_time

        assert result == "response"
        assert attempt_count == 4
        assert len(sleep_calls) == 3
        assert sleep_calls[0] == 1.0  # First retry delay
        assert sleep_calls[1] == 2.0  # Second retry delay
        assert sleep_calls[2] == 4.0  # Third retry delay
        assert elapsed_time < 0.5  # Completes quickly without real sleeps

    def test_authentication_error_raises_immediately(self) -> None:
        """Negative case: Non-retryable exception raises immediately without retry."""
        attempt_count = 0

        @retry(max_retries=3, base_delay_seconds=1, retryable_exceptions=(CustomRetryableError,))
        def auth_call():
            nonlocal attempt_count
            attempt_count += 1
            raise NonRetryableError("401 Unauthorized")

        with pytest.raises(NonRetryableError):
            auth_call()

        assert attempt_count == 1
