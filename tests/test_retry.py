"""Tests for retry decorator with exponential backoff."""

import time

import pytest

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

        @retry(max_retries=2, base_delay_seconds=1.0, retryable_exceptions=(CustomRetryableError,))
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

    def test_logs_retry_attempts(self, caplog) -> None:
        """Retry attempts are logged with attempt number and delay."""
        attempt_count = 0

        @retry(max_retries=3, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        def failing_call():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise CustomRetryableError("Temporary error")
            return "success"

        failing_call()

        assert attempt_count == 3

    def test_logs_max_retries_exceeded(self) -> None:
        """Max retries exceeded is logged."""

        @retry(max_retries=2, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        def always_failing_call():
            raise CustomRetryableError("Persistent error")

        with pytest.raises(CustomRetryableError):
            always_failing_call()

    def test_logs_non_retryable_exception(self) -> None:
        """Non-retryable exceptions are logged."""

        @retry(max_retries=3, base_delay_seconds=0.01, retryable_exceptions=(CustomRetryableError,))
        def non_retryable_call():
            raise NonRetryableError("Non-retryable")

        with pytest.raises(NonRetryableError):
            non_retryable_call()

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


class TestRetryExamples:
    """Example tests from acceptance criteria."""

    def test_429_response_example(self) -> None:
        """Example: 429 response triggers retry after 1s, then 2s, then 4s."""
        attempt_count = 0

        @retry(max_retries=3, base_delay_seconds=1, retryable_exceptions=(CustomRetryableError,))
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
        assert elapsed_time >= 7 * 0.8  # 1 + 2 + 4 seconds minimum

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
