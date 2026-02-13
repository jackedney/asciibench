"""Tests for the concurrent generation module.

This module tests the shared concurrent generation primitive that is used
by both batch CLI (sampler.py) and Judge UI (generation_service.py).

Tests cover:
- Concurrent generation with semaphore control
- Idempotency checks against existing samples
- Error handling that creates is_valid=False samples
- Callback invocation after each sample generation
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from asciibench.common.config import GenerationConfig
from asciibench.common.models import ArtSample, OpenRouterResponse
from asciibench.generator.client import (
    AuthenticationError,
    ModelError,
    OpenRouterClient,
    OpenRouterClientError,
    RateLimitError,
    TransientError,
)
from asciibench.generator.concurrent import (
    GenerationTask,
    _create_error_sample,
    _handle_generation_error,
    _validate_output,
    generate_samples_concurrent,
)


class TestGenerationTask:
    """Tests for GenerationTask dataclass."""

    def test_generation_task_creation(self) -> None:
        """GenerationTask can be created with required fields."""
        task = GenerationTask(
            model_id="openai/gpt-4o",
            prompt_text="Draw a cat",
            category="animals",
            attempt=1,
        )

        assert task.model_id == "openai/gpt-4o"
        assert task.prompt_text == "Draw a cat"
        assert task.category == "animals"
        assert task.attempt == 1


class TestCreateErrorSample:
    """Tests for _create_error_sample helper function."""

    def test_create_error_sample_returns_invalid_sample(self) -> None:
        """_create_error_sample returns ArtSample with is_valid=False."""
        task = GenerationTask(
            model_id="model-a",
            prompt_text="Draw a cat",
            category="animals",
            attempt=1,
        )

        sample = _create_error_sample(task)

        assert sample.model_id == "model-a"
        assert sample.prompt_text == "Draw a cat"
        assert sample.category == "animals"
        assert sample.attempt_number == 1
        assert sample.raw_output == ""
        assert sample.sanitized_output == ""
        assert sample.is_valid is False
        assert sample.output_tokens is None
        assert sample.cost is None


class TestHandleGenerationError:
    """Tests for _handle_generation_error helper function."""

    def test_handle_rate_limit_error(self) -> None:
        """_handle_generation_error handles RateLimitError."""
        task = GenerationTask(
            model_id="model-a",
            prompt_text="Draw a cat",
            category="animals",
            attempt=1,
        )

        sample = _handle_generation_error(task, RateLimitError("Rate limited"))

        assert sample.is_valid is False
        assert sample.raw_output == ""

    def test_handle_authentication_error(self) -> None:
        """_handle_generation_error handles AuthenticationError."""
        task = GenerationTask(
            model_id="model-a",
            prompt_text="Draw a cat",
            category="animals",
            attempt=1,
        )

        sample = _handle_generation_error(task, AuthenticationError("Auth failed"))

        assert sample.is_valid is False

    def test_handle_model_error(self) -> None:
        """_handle_generation_error handles ModelError."""
        task = GenerationTask(
            model_id="model-a",
            prompt_text="Draw a cat",
            category="animals",
            attempt=1,
        )

        sample = _handle_generation_error(task, ModelError("Model error"))

        assert sample.is_valid is False

    def test_handle_transient_error(self) -> None:
        """_handle_generation_error handles TransientError."""
        task = GenerationTask(
            model_id="model-a",
            prompt_text="Draw a cat",
            category="animals",
            attempt=1,
        )

        sample = _handle_generation_error(task, TransientError("Transient"))

        assert sample.is_valid is False

    def test_handle_generic_client_error(self) -> None:
        """_handle_generation_error handles generic OpenRouterClientError."""
        task = GenerationTask(
            model_id="model-a",
            prompt_text="Draw a cat",
            category="animals",
            attempt=1,
        )

        sample = _handle_generation_error(task, OpenRouterClientError("Generic error"))

        assert sample.is_valid is False

    def test_handle_unexpected_exception(self) -> None:
        """_handle_generation_error handles unexpected exception types."""
        task = GenerationTask(
            model_id="model-a",
            prompt_text="Draw a cat",
            category="animals",
            attempt=1,
        )

        sample = _handle_generation_error(task, ValueError("Unexpected"))

        assert sample.is_valid is False


class TestValidateOutput:
    """Tests for _validate_output helper function."""

    def test_validate_output_valid(self) -> None:
        """_validate_output returns True for valid output."""
        raw_output = "```text\n/\\_/\\\n```"
        sanitized_output = "/\\_/\\"
        max_tokens = 1000

        result = _validate_output(raw_output, sanitized_output, max_tokens)

        assert result is True

    def test_validate_output_empty_sanitized(self) -> None:
        """_validate_output returns False for empty sanitized output."""
        raw_output = "no code block"
        sanitized_output = ""
        max_tokens = 1000

        result = _validate_output(raw_output, sanitized_output, max_tokens)

        assert result is False

    def test_validate_output_exceeds_max_tokens(self) -> None:
        """_validate_output returns False when output exceeds max_tokens * 3."""
        raw_output = "x" * 4000
        sanitized_output = "x" * 4000
        max_tokens = 1000

        result = _validate_output(raw_output, sanitized_output, max_tokens)

        assert result is False

    def test_validate_output_at_limit(self) -> None:
        """_validate_output returns True at exactly max_tokens * 3."""
        raw_output = "x" * 2999
        sanitized_output = "x" * 2999
        max_tokens = 1000

        result = _validate_output(raw_output, sanitized_output, max_tokens)

        assert result is True


class TestGenerateSamplesConcurrent:
    """Tests for generate_samples_concurrent function."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock OpenRouterClient."""
        client = MagicMock(spec=OpenRouterClient)
        mock_response = OpenRouterResponse(
            text="```\n/\\_/\\\n( o.o )\n > ^ <\n```",
            prompt_tokens=10,
            completion_tokens=50,
            total_tokens=60,
            cost=0.0001,
        )
        client.generate_async = AsyncMock(return_value=mock_response)
        return client

    @pytest.fixture
    def sample_config(self) -> GenerationConfig:
        """Create sample generation config."""
        return GenerationConfig(
            attempts_per_prompt=1,
            temperature=0.0,
            max_tokens=1000,
            max_concurrent_requests=10,
        )

    @pytest.mark.asyncio
    async def test_generates_samples_concurrently(
        self,
        mock_client: MagicMock,
        sample_config: GenerationConfig,
        tmp_path: Path,
    ) -> None:
        """generate_samples_concurrent generates all tasks."""
        db_path = tmp_path / "database.jsonl"

        tasks = [
            GenerationTask(
                model_id="model-a",
                prompt_text="Draw a cat",
                category="animals",
                attempt=1,
            ),
            GenerationTask(
                model_id="model-b",
                prompt_text="Draw a dog",
                category="animals",
                attempt=1,
            ),
        ]

        samples = await generate_samples_concurrent(
            tasks=tasks,
            client=mock_client,
            config=sample_config,
            database_path=db_path,
            existing_keys=set(),
            max_concurrent=10,
        )

        assert len(samples) == 2
        assert mock_client.generate_async.call_count == 2

        for sample in samples:
            assert sample.is_valid is True

    @pytest.mark.asyncio
    async def test_skips_existing_samples_via_idempotency(
        self,
        mock_client: MagicMock,
        sample_config: GenerationConfig,
        tmp_path: Path,
    ) -> None:
        """generate_samples_concurrent skips tasks that exist in existing_keys."""
        db_path = tmp_path / "database.jsonl"

        tasks = [
            GenerationTask(
                model_id="model-a",
                prompt_text="Draw a cat",
                category="animals",
                attempt=1,
            ),
            GenerationTask(
                model_id="model-b",
                prompt_text="Draw a dog",
                category="animals",
                attempt=1,
            ),
        ]

        existing_keys = {("model-a", "Draw a cat", 1)}

        samples = await generate_samples_concurrent(
            tasks=tasks,
            client=mock_client,
            config=sample_config,
            database_path=db_path,
            existing_keys=existing_keys,
            max_concurrent=10,
        )

        assert len(samples) == 1
        assert samples[0].model_id == "model-b"
        assert mock_client.generate_async.call_count == 1

    @pytest.mark.asyncio
    async def test_creates_error_sample_on_api_failure(
        self,
        sample_config: GenerationConfig,
        tmp_path: Path,
    ) -> None:
        """generate_samples_concurrent creates is_valid=False sample on error."""
        db_path = tmp_path / "database.jsonl"

        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.generate_async = AsyncMock(side_effect=OpenRouterClientError("API error"))

        tasks = [
            GenerationTask(
                model_id="model-a",
                prompt_text="Draw a cat",
                category="animals",
                attempt=1,
            ),
        ]

        samples = await generate_samples_concurrent(
            tasks=tasks,
            client=mock_client,
            config=sample_config,
            database_path=db_path,
            existing_keys=set(),
            max_concurrent=10,
        )

        assert len(samples) == 1
        assert samples[0].is_valid is False
        assert samples[0].raw_output == ""

    @pytest.mark.asyncio
    async def test_invokes_callback_after_each_sample(
        self,
        mock_client: MagicMock,
        sample_config: GenerationConfig,
        tmp_path: Path,
    ) -> None:
        """generate_samples_concurrent invokes on_generated callback."""
        db_path = tmp_path / "database.jsonl"

        callback_samples: list[ArtSample] = []

        def on_generated(sample: ArtSample) -> None:
            callback_samples.append(sample)

        tasks = [
            GenerationTask(
                model_id="model-a",
                prompt_text="Draw a cat",
                category="animals",
                attempt=1,
            ),
            GenerationTask(
                model_id="model-b",
                prompt_text="Draw a dog",
                category="animals",
                attempt=1,
            ),
        ]

        await generate_samples_concurrent(
            tasks=tasks,
            client=mock_client,
            config=sample_config,
            database_path=db_path,
            existing_keys=set(),
            max_concurrent=10,
            on_generated=on_generated,
        )

        assert len(callback_samples) == 2

    @pytest.mark.asyncio
    async def test_respects_max_concurrent_limit(
        self,
        sample_config: GenerationConfig,
        tmp_path: Path,
    ) -> None:
        """generate_samples_concurrent respects max_concurrent semaphore."""
        import asyncio

        db_path = tmp_path / "database.jsonl"

        concurrent_calls = 0
        max_concurrent_calls = 0
        lock = asyncio.Lock()

        async def tracked_generate(*args, **kwargs):
            nonlocal concurrent_calls, max_concurrent_calls

            async with lock:
                concurrent_calls += 1
                if concurrent_calls > max_concurrent_calls:
                    max_concurrent_calls = concurrent_calls

            await asyncio.sleep(0.05)

            async with lock:
                concurrent_calls -= 1

            return OpenRouterResponse(
                text="```\nart\n```",
                prompt_tokens=10,
                completion_tokens=50,
                total_tokens=60,
                cost=0.0001,
            )

        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.generate_async = AsyncMock(wraps=tracked_generate)

        tasks = [
            GenerationTask(
                model_id=f"model-{i}",
                prompt_text=f"Draw {i}",
                category="test",
                attempt=1,
            )
            for i in range(6)
        ]

        await generate_samples_concurrent(
            tasks=tasks,
            client=mock_client,
            config=sample_config,
            database_path=db_path,
            existing_keys=set(),
            max_concurrent=2,
        )

        assert max_concurrent_calls <= 2

    @pytest.mark.asyncio
    async def test_persists_samples_to_database(
        self,
        mock_client: MagicMock,
        sample_config: GenerationConfig,
        tmp_path: Path,
    ) -> None:
        """generate_samples_concurrent persists samples to JSONL file."""
        db_path = tmp_path / "database.jsonl"

        tasks = [
            GenerationTask(
                model_id="model-a",
                prompt_text="Draw a cat",
                category="animals",
                attempt=1,
            ),
        ]

        await generate_samples_concurrent(
            tasks=tasks,
            client=mock_client,
            config=sample_config,
            database_path=db_path,
            existing_keys=set(),
            max_concurrent=10,
        )

        assert db_path.exists()
        content = db_path.read_text()
        assert "model-a" in content
        assert "Draw a cat" in content

    @pytest.mark.asyncio
    async def test_empty_tasks_returns_empty_list(
        self,
        mock_client: MagicMock,
        sample_config: GenerationConfig,
        tmp_path: Path,
    ) -> None:
        """generate_samples_concurrent returns empty list for empty tasks."""
        db_path = tmp_path / "database.jsonl"

        samples = await generate_samples_concurrent(
            tasks=[],
            client=mock_client,
            config=sample_config,
            database_path=db_path,
            existing_keys=set(),
            max_concurrent=10,
        )

        assert samples == []
        assert mock_client.generate_async.call_count == 0

    @pytest.mark.asyncio
    async def test_continues_on_individual_task_failure(
        self,
        sample_config: GenerationConfig,
        tmp_path: Path,
    ) -> None:
        """generate_samples_concurrent continues after individual task failure."""
        db_path = tmp_path / "database.jsonl"

        call_count = 0

        async def failing_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OpenRouterClientError("API error")
            return OpenRouterResponse(
                text="```\nart\n```",
                prompt_tokens=10,
                completion_tokens=50,
                total_tokens=60,
                cost=0.0001,
            )

        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.generate_async = AsyncMock(wraps=failing_generate)

        tasks = [
            GenerationTask(
                model_id="model-a",
                prompt_text="Draw a cat",
                category="animals",
                attempt=1,
            ),
            GenerationTask(
                model_id="model-b",
                prompt_text="Draw a dog",
                category="animals",
                attempt=1,
            ),
        ]

        samples = await generate_samples_concurrent(
            tasks=tasks,
            client=mock_client,
            config=sample_config,
            database_path=db_path,
            existing_keys=set(),
            max_concurrent=10,
        )

        assert len(samples) == 2
        valid_samples = [s for s in samples if s.is_valid]
        invalid_samples = [s for s in samples if not s.is_valid]
        assert len(valid_samples) == 1
        assert len(invalid_samples) == 1

    @pytest.mark.asyncio
    async def test_callback_invoked_for_error_samples(
        self,
        sample_config: GenerationConfig,
        tmp_path: Path,
    ) -> None:
        """generate_samples_concurrent invokes callback even for error samples."""
        db_path = tmp_path / "database.jsonl"

        callback_samples: list[ArtSample] = []

        def on_generated(sample: ArtSample) -> None:
            callback_samples.append(sample)

        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.generate_async = AsyncMock(side_effect=OpenRouterClientError("API error"))

        tasks = [
            GenerationTask(
                model_id="model-a",
                prompt_text="Draw a cat",
                category="animals",
                attempt=1,
            ),
        ]

        await generate_samples_concurrent(
            tasks=tasks,
            client=mock_client,
            config=sample_config,
            database_path=db_path,
            existing_keys=set(),
            max_concurrent=10,
            on_generated=on_generated,
        )

        assert len(callback_samples) == 1
        assert callback_samples[0].is_valid is False

    @pytest.mark.asyncio
    async def test_sample_includes_cost_and_tokens(
        self,
        mock_client: MagicMock,
        sample_config: GenerationConfig,
        tmp_path: Path,
    ) -> None:
        """generate_samples_concurrent includes cost and tokens from API response."""
        db_path = tmp_path / "database.jsonl"

        tasks = [
            GenerationTask(
                model_id="model-a",
                prompt_text="Draw a cat",
                category="animals",
                attempt=1,
            ),
        ]

        samples = await generate_samples_concurrent(
            tasks=tasks,
            client=mock_client,
            config=sample_config,
            database_path=db_path,
            existing_keys=set(),
            max_concurrent=10,
        )

        assert len(samples) == 1
        assert samples[0].output_tokens == 50
        assert samples[0].cost == 0.0001
