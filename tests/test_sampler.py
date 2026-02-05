"""Tests for the sampler module."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from asciibench.common.config import GenerationConfig
from asciibench.common.models import ArtSample, Model, OpenRouterResponse, Prompt
from asciibench.generator.client import OpenRouterClient, OpenRouterClientError
from asciibench.generator.sampler import (
    _build_existing_sample_keys,
    _sample_exists,
    generate_samples,
)


class TestBuildExistingSampleKeys:
    """Tests for _build_existing_sample_keys helper function."""

    def test_empty_list_returns_empty_set(self) -> None:
        """Empty sample list returns empty set."""
        result = _build_existing_sample_keys([])
        assert result == set()

    def test_builds_keys_from_samples(self) -> None:
        """Builds correct keys from sample list."""
        samples = [
            ArtSample(
                model_id="openai/gpt-4o",
                prompt_text="Draw a cat",
                category="single_animal",
                attempt_number=1,
                raw_output="```\ncat\n```",
                sanitized_output="cat",
                is_valid=True,
            ),
            ArtSample(
                model_id="openai/gpt-4o",
                prompt_text="Draw a cat",
                category="single_animal",
                attempt_number=2,
                raw_output="```\ncat2\n```",
                sanitized_output="cat2",
                is_valid=True,
            ),
            ArtSample(
                model_id="anthropic/claude-3-opus",
                prompt_text="Draw a dog",
                category="single_animal",
                attempt_number=1,
                raw_output="```\ndog\n```",
                sanitized_output="dog",
                is_valid=True,
            ),
        ]

        result = _build_existing_sample_keys(samples)

        assert len(result) == 3
        assert ("openai/gpt-4o", "Draw a cat", 1) in result
        assert ("openai/gpt-4o", "Draw a cat", 2) in result
        assert ("anthropic/claude-3-opus", "Draw a dog", 1) in result


class TestSampleExists:
    """Tests for _sample_exists helper function."""

    def test_returns_true_when_key_exists(self) -> None:
        """Returns True when sample key exists in set."""
        existing_keys = {
            ("openai/gpt-4o", "Draw a cat", 1),
            ("openai/gpt-4o", "Draw a cat", 2),
        }

        assert _sample_exists("openai/gpt-4o", "Draw a cat", 1, existing_keys) is True
        assert _sample_exists("openai/gpt-4o", "Draw a cat", 2, existing_keys) is True

    def test_returns_false_when_key_does_not_exist(self) -> None:
        """Returns False when sample key doesn't exist in set."""
        existing_keys = {
            ("openai/gpt-4o", "Draw a cat", 1),
        }

        assert _sample_exists("openai/gpt-4o", "Draw a cat", 2, existing_keys) is False
        assert _sample_exists("openai/gpt-4o", "Draw a dog", 1, existing_keys) is False
        assert _sample_exists("anthropic/claude", "Draw a cat", 1, existing_keys) is False

    def test_returns_false_for_empty_set(self) -> None:
        """Returns False when set is empty."""
        assert _sample_exists("openai/gpt-4o", "Draw a cat", 1, set()) is False


class TestGenerateSamples:
    """Tests for generate_samples function."""

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
        client.generate.return_value = mock_response
        client.generate_async = AsyncMock(return_value=mock_response)
        return client

    @pytest.fixture
    def sample_models(self) -> list[Model]:
        """Create sample model list."""
        return [
            Model(id="openai/gpt-4o", name="GPT-4o"),
        ]

    @pytest.fixture
    def sample_prompts(self) -> list[Prompt]:
        """Create sample prompt list."""
        return [
            Prompt(text="Draw a cat", category="single_animal", template_type="animal"),
        ]

    @pytest.fixture
    def sample_config(self) -> GenerationConfig:
        """Create sample generation config."""
        return GenerationConfig(
            attempts_per_prompt=2,
            temperature=0.0,
            max_tokens=1000,
        )

    def test_generates_samples_for_all_combinations(
        self,
        mock_client: MagicMock,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        sample_config: GenerationConfig,
        tmp_path: Path,
    ) -> None:
        """Generates samples for all model x prompt x attempt combinations."""
        db_path = tmp_path / "database.jsonl"

        result = generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=sample_config,
            database_path=db_path,
            client=mock_client,
        )

        # Should generate 1 model x 1 prompt x 2 attempts = 2 samples
        assert len(result) == 2
        assert mock_client.generate_async.call_count == 2

        # Verify sample properties
        for i, sample in enumerate(result, start=1):
            assert sample.model_id == "openai/gpt-4o"
            assert sample.prompt_text == "Draw a cat"
            assert sample.category == "single_animal"
            assert sample.attempt_number == i
            assert sample.is_valid is True
            assert sample.sanitized_output == "/\\_/\\\n( o.o )\n > ^ <"

    def test_idempotency_skips_existing_samples(
        self,
        mock_client: MagicMock,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        sample_config: GenerationConfig,
        tmp_path: Path,
    ) -> None:
        """Running twice produces same database (idempotent)."""
        db_path = tmp_path / "database.jsonl"

        # First run - generates all samples
        result1 = generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=sample_config,
            database_path=db_path,
            client=mock_client,
        )

        assert len(result1) == 2
        assert mock_client.generate_async.call_count == 2

        # Second run - should skip all samples (idempotent)
        mock_client.reset_mock()
        result2 = generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=sample_config,
            database_path=db_path,
            client=mock_client,
        )

        assert len(result2) == 0  # No new samples generated
        assert mock_client.generate_async.call_count == 0  # No API calls made

        # Database should still have 2 entries
        with open(db_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 2

    def test_resume_capability_continues_from_last_sample(
        self,
        mock_client: MagicMock,
        sample_config: GenerationConfig,
        tmp_path: Path,
    ) -> None:
        """Can resume generation after interruption."""
        db_path = tmp_path / "database.jsonl"

        models = [Model(id="openai/gpt-4o", name="GPT-4o")]
        prompts = [
            Prompt(text="Draw a cat", category="single_animal", template_type="animal"),
            Prompt(text="Draw a dog", category="single_animal", template_type="animal"),
        ]

        # Simulate first run that only completes first prompt
        existing_sample = ArtSample(
            model_id="openai/gpt-4o",
            prompt_text="Draw a cat",
            category="single_animal",
            attempt_number=1,
            raw_output="```\ncat\n```",
            sanitized_output="cat",
            is_valid=True,
        )
        with open(db_path, "w") as f:
            f.write(existing_sample.model_dump_json() + "\n")

        # Resume - should only generate missing samples
        config = GenerationConfig(attempts_per_prompt=1)
        result = generate_samples(
            models=models,
            prompts=prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        # Should only generate 1 new sample (dog prompt)
        assert len(result) == 1
        assert result[0].prompt_text == "Draw a dog"
        assert mock_client.generate_async.call_count == 1

    def test_api_error_does_not_prevent_other_samples(
        self,
        sample_models: list[Model],
        sample_config: GenerationConfig,
        tmp_path: Path,
    ) -> None:
        """API error on one sample doesn't prevent others from being saved."""
        db_path = tmp_path / "database.jsonl"

        prompts = [
            Prompt(text="Draw a cat", category="single_animal", template_type="animal"),
            Prompt(text="Draw a dog", category="single_animal", template_type="animal"),
        ]

        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.generate_async = AsyncMock(
            side_effect=[
                OpenRouterClientError("API error"),
                OpenRouterResponse(text="```\ndog\n```"),
            ]
        )

        config = GenerationConfig(attempts_per_prompt=1)
        result = generate_samples(
            models=sample_models,
            prompts=prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        assert len(result) == 2

        cat_sample = next(s for s in result if s.prompt_text == "Draw a cat")
        dog_sample = next(s for s in result if s.prompt_text == "Draw a dog")

        assert cat_sample.is_valid is False
        assert cat_sample.raw_output == ""

        assert dog_sample.is_valid is True
        assert dog_sample.sanitized_output == "dog"

        # Database should have 2 entries
        with open(db_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 2

    def test_invalid_when_no_code_block_found(
        self,
        mock_client: MagicMock,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        tmp_path: Path,
    ) -> None:
        """Sample is marked invalid when no code block is found."""
        db_path = tmp_path / "database.jsonl"

        mock_client.generate_async = AsyncMock(
            return_value=OpenRouterResponse(text="Here is a cat: /\\_/\\")
        )

        config = GenerationConfig(attempts_per_prompt=1)
        result = generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        assert len(result) == 1
        assert result[0].is_valid is False
        assert result[0].sanitized_output == ""
        assert result[0].raw_output == "Here is a cat: /\\_/\\"

    def test_invalid_when_output_exceeds_max_tokens(
        self,
        mock_client: MagicMock,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        tmp_path: Path,
    ) -> None:
        """Sample is marked invalid when output exceeds max_tokens limit."""
        db_path = tmp_path / "database.jsonl"

        long_content = "x" * 100
        mock_client.generate_async = AsyncMock(
            return_value=OpenRouterResponse(text=f"```\n{long_content}\n```")
        )

        config = GenerationConfig(attempts_per_prompt=1, max_tokens=10)
        result = generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        assert len(result) == 1
        assert result[0].is_valid is False
        assert result[0].sanitized_output == long_content

    def test_multiple_models_and_prompts(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Correctly iterates through multiple models and prompts."""
        db_path = tmp_path / "database.jsonl"

        models = [
            Model(id="openai/gpt-4o", name="GPT-4o"),
            Model(id="anthropic/claude-3-opus", name="Claude 3 Opus"),
        ]
        prompts = [
            Prompt(text="Draw a cat", category="single_animal", template_type="animal"),
            Prompt(text="Draw a tree", category="single_object", template_type="object"),
        ]
        config = GenerationConfig(attempts_per_prompt=2)

        result = generate_samples(
            models=models,
            prompts=prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        # 2 models x 2 prompts x 2 attempts = 8 samples
        assert len(result) == 8
        assert mock_client.generate_async.call_count == 8

        # Verify all combinations are present
        combinations = {(s.model_id, s.prompt_text, s.attempt_number) for s in result}
        expected = {
            ("openai/gpt-4o", "Draw a cat", 1),
            ("openai/gpt-4o", "Draw a cat", 2),
            ("openai/gpt-4o", "Draw a tree", 1),
            ("openai/gpt-4o", "Draw a tree", 2),
            ("anthropic/claude-3-opus", "Draw a cat", 1),
            ("anthropic/claude-3-opus", "Draw a cat", 2),
            ("anthropic/claude-3-opus", "Draw a tree", 1),
            ("anthropic/claude-3-opus", "Draw a tree", 2),
        }
        assert combinations == expected

    def test_persists_samples_immediately(
        self,
        mock_client: MagicMock,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        tmp_path: Path,
    ) -> None:
        """Each sample is persisted immediately after generation."""
        db_path = tmp_path / "database.jsonl"

        mock_client.generate_async = AsyncMock(
            return_value=OpenRouterResponse(text="```\nart\n```")
        )

        config = GenerationConfig(attempts_per_prompt=3)
        generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        # Verify all 3 calls were made
        assert mock_client.generate_async.call_count == 3

        # Verify database has all 3 samples persisted
        with open(db_path) as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 3

    def test_creates_client_from_settings_if_not_provided(
        self,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        tmp_path: Path,
    ) -> None:
        """Creates OpenRouterClient from Settings if client not provided."""
        db_path = tmp_path / "database.jsonl"
        config = GenerationConfig(attempts_per_prompt=1)

        with patch("asciibench.generator.sampler.OpenRouterClient") as mock_client_class:
            mock_instance = MagicMock()
            mock_instance.generate_async = AsyncMock(
                return_value=OpenRouterResponse(text="```\nart\n```")
            )
            mock_client_class.return_value = mock_instance

            from asciibench.common.config import Settings

            settings = Settings(openrouter_api_key="test-key", base_url="https://test.api")

            generate_samples(
                models=sample_models,
                prompts=sample_prompts,
                config=config,
                database_path=db_path,
                settings=settings,
            )

            mock_client_class.assert_called_once_with(
                api_key="test-key",
                base_url="https://test.api",
                timeout=120,
            )

    def test_empty_models_returns_empty_list(
        self,
        mock_client: MagicMock,
        sample_prompts: list[Prompt],
        sample_config: GenerationConfig,
        tmp_path: Path,
    ) -> None:
        """Empty models list returns empty result."""
        db_path = tmp_path / "database.jsonl"

        result = generate_samples(
            models=[],
            prompts=sample_prompts,
            config=sample_config,
            database_path=db_path,
            client=mock_client,
        )

        assert result == []
        assert mock_client.generate_async.call_count == 0

    def test_empty_prompts_returns_empty_list(
        self,
        mock_client: MagicMock,
        sample_models: list[Model],
        sample_config: GenerationConfig,
        tmp_path: Path,
    ) -> None:
        """Empty prompts list returns empty result."""
        db_path = tmp_path / "database.jsonl"

        result = generate_samples(
            models=sample_models,
            prompts=[],
            config=sample_config,
            database_path=db_path,
            client=mock_client,
        )

        assert result == []
        assert mock_client.generate_async.call_count == 0

    def test_sample_has_uuid_and_timestamp(
        self,
        mock_client: MagicMock,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        tmp_path: Path,
    ) -> None:
        """Generated samples have UUID and timestamp fields."""
        db_path = tmp_path / "database.jsonl"
        config = GenerationConfig(attempts_per_prompt=1)

        result = generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        assert len(result) == 1
        sample = result[0]

        # Verify UUID is set
        assert sample.id is not None
        assert len(str(sample.id)) == 36  # UUID format

        # Verify timestamp is set
        assert sample.timestamp is not None

    def test_sample_includes_cost_and_tokens(
        self,
        mock_client: MagicMock,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        tmp_path: Path,
    ) -> None:
        """Generated samples include cost and token metadata from API response."""
        db_path = tmp_path / "database.jsonl"
        config = GenerationConfig(attempts_per_prompt=1)

        result = generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        assert len(result) == 1
        sample = result[0]

        # Verify cost and tokens are set from mock response
        assert sample.output_tokens == 50
        assert sample.cost == 0.0001

    def test_sample_handles_missing_cost_and_tokens(
        self,
        tmp_path: Path,
    ) -> None:
        """Generated samples handle None cost and tokens from API response."""
        db_path = tmp_path / "database.jsonl"
        config = GenerationConfig(attempts_per_prompt=1)

        # Create mock with missing cost/tokens
        mock_client = MagicMock(spec=OpenRouterClient)
        mock_response = OpenRouterResponse(
            text="```\n/\\_/\\\n( o.o )\n > ^ <\n```",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            cost=None,
        )
        mock_client.generate.return_value = mock_response
        mock_client.generate_async = AsyncMock(return_value=mock_response)

        models = [Model(id="openai/gpt-4o", name="GPT-4o")]
        prompts = [Prompt(text="Draw a cat", category="single_animal", template_type="animal")]

        result = generate_samples(
            models=models,
            prompts=prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        assert len(result) == 1
        sample = result[0]

        # Verify cost and tokens are None when missing from API response
        assert sample.output_tokens is None
        assert sample.cost is None

    def test_failed_sample_has_none_cost_and_tokens(
        self,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        tmp_path: Path,
    ) -> None:
        """Failed samples have None cost and tokens."""
        db_path = tmp_path / "database.jsonl"
        config = GenerationConfig(attempts_per_prompt=1)

        # Create mock that raises error
        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.generate_async = AsyncMock(side_effect=OpenRouterClientError("API error"))

        result = generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        assert len(result) == 1
        sample = result[0]

        # Failed samples should have None cost and tokens
        assert sample.is_valid is False
        assert sample.output_tokens is None
        assert sample.cost is None

    def test_generate_samples_sets_run_id(
        self,
        mock_client: MagicMock,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        tmp_path: Path,
    ) -> None:
        """generate_samples sets run_id for entire batch."""
        db_path = tmp_path / "database.jsonl"
        log_path = tmp_path / "logs.jsonl"

        from asciibench.common.logging import get_logger, get_run_id, set_run_id

        # Clear any previous run_id
        set_run_id(None)

        # Configure logger to use custom log_path
        logger = get_logger("generator.sampler")
        logger.log_path = log_path

        config = GenerationConfig(attempts_per_prompt=2)
        generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        # Verify run_id was set during generation
        run_id = get_run_id()
        # run_id must be a non-empty string
        assert isinstance(run_id, str)
        assert len(run_id) > 0

        # Log file must exist
        assert log_path.exists()

        # Read log file and verify run_id is present in logs
        with open(log_path) as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) > 0, "Log file should have entries"

        import json

        # Check that at least some logs have run_id
        log_entries = [json.loads(line) for line in lines]
        entries_with_run_id = [e for e in log_entries if "run_id" in e]
        # Should have at least some entries with run_id
        assert len(entries_with_run_id) > 0, "At least one log entry should have run_id"

        # Clear run_id
        set_run_id(None)

    def test_each_sample_has_unique_request_id_in_context(
        self,
        mock_client: MagicMock,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        tmp_path: Path,
    ) -> None:
        """Each sample generation sets a unique request_id."""
        db_path = tmp_path / "database.jsonl"
        log_path = tmp_path / "logs.jsonl"

        from asciibench.common.logging import get_logger, set_request_id

        # Clear any previous request_id
        set_request_id(None)

        # Configure logger to use custom log_path
        logger = get_logger("generator.sampler")
        logger.log_path = log_path

        attempts_per_prompt = 2
        config = GenerationConfig(attempts_per_prompt=attempts_per_prompt)
        generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        # Clear request_id
        set_request_id(None)

        # Log file must exist
        assert log_path.exists()

        # Read log file and verify request_ids are present and unique
        with open(log_path) as f:
            lines = [line for line in f if line.strip()]

        assert len(lines) > 0, "Log file should have entries"

        import json

        # Extract all request_ids from log entries
        log_entries = [json.loads(line) for line in lines]
        request_ids = [e["request_id"] for e in log_entries if "request_id" in e]

        # Verify we have the expected number of unique request_ids
        expected_count = len(sample_models) * len(sample_prompts) * attempts_per_prompt
        unique_request_ids = set(request_ids)

        assert len(unique_request_ids) == expected_count, (
            f"Expected {expected_count} unique request_ids, got {len(unique_request_ids)}"
        )
        assert all(rid is not None for rid in request_ids), "No request_id should be None"


class TestSemaphoreConcurrencyLimit:
    """Tests for semaphore-based concurrency limit."""

    @pytest.fixture
    def sample_models(self) -> list[Model]:
        """Create sample model list."""
        return [
            Model(id="openai/gpt-4o", name="GPT-4o"),
            Model(id="anthropic/claude-3-opus", name="Claude 3 Opus"),
        ]

    @pytest.fixture
    def sample_prompts(self) -> list[Prompt]:
        """Create sample prompt list."""
        return [
            Prompt(text="Draw a cat", category="single_animal", template_type="animal"),
            Prompt(text="Draw a dog", category="single_animal", template_type="animal"),
            Prompt(text="Draw a tree", category="single_object", template_type="object"),
        ]

    def test_concurrent_generation_respects_semaphore_limit(
        self,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        tmp_path: Path,
    ) -> None:
        """Verify max concurrent calls never exceeds config limit."""
        import asyncio
        from unittest.mock import AsyncMock

        db_path = tmp_path / "database.jsonl"

        # Track concurrent calls
        concurrent_calls = 0
        max_concurrent_calls = 0
        call_lock = asyncio.Lock()

        async def delayed_generate(*args, **kwargs):
            """Mock generate with delay to allow concurrency tracking."""
            nonlocal concurrent_calls, max_concurrent_calls

            async with call_lock:
                concurrent_calls += 1
                if concurrent_calls > max_concurrent_calls:
                    max_concurrent_calls = concurrent_calls

            # Simulate API delay
            await asyncio.sleep(0.1)

            async with call_lock:
                concurrent_calls -= 1

            return OpenRouterResponse(
                text="```\n/\\_/\\\n( o.o )\n > ^ <\n```",
                prompt_tokens=10,
                completion_tokens=50,
                total_tokens=60,
                cost=0.0001,
            )

        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.generate_async = AsyncMock(wraps=delayed_generate)

        # Test with limit=2 and 10 tasks
        config = GenerationConfig(
            attempts_per_prompt=1,
            max_concurrent_requests=2,
        )

        # Create 10 tasks (2 models x 3 prompts x 1 attempt = 6, but we can adjust)
        # Let's create more prompts to get closer to 10 tasks
        additional_prompts = [
            *sample_prompts,
            Prompt(text="Draw a bird", category="single_animal", template_type="animal"),
            Prompt(text="Draw a fish", category="single_animal", template_type="animal"),
            Prompt(text="Draw a house", category="single_object", template_type="object"),
        ]

        result = generate_samples(
            models=sample_models,
            prompts=additional_prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        # Verify tasks completed
        assert len(result) == len(sample_models) * len(additional_prompts) * 1

        # Verify max concurrent calls never exceeded limit of 2
        assert max_concurrent_calls <= 2
        assert 1 <= max_concurrent_calls <= 2, (
            "Should have at least 1 concurrent call to test concurrency"
        )

    def test_sequential_mode_with_limit_one(
        self,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        tmp_path: Path,
    ) -> None:
        """Setting limit=1 results in sequential execution."""
        import asyncio
        from unittest.mock import AsyncMock

        db_path = tmp_path / "database.jsonl"

        # Track concurrent calls and order
        concurrent_calls = 0
        max_concurrent_calls = 0
        call_order = []
        call_lock = asyncio.Lock()

        async def delayed_generate(*args, **kwargs):
            """Mock generate with delay to verify sequential execution."""
            nonlocal concurrent_calls, max_concurrent_calls

            async with call_lock:
                concurrent_calls += 1
                if concurrent_calls > max_concurrent_calls:
                    max_concurrent_calls = concurrent_calls

            # Simulate API delay
            await asyncio.sleep(0.05)

            async with call_lock:
                concurrent_calls -= 1

            # Extract model_id and prompt_text from kwargs to track order
            model_id = kwargs.get("model_id", "")
            prompt_text = kwargs.get("prompt_text", "")
            call_order.append((model_id, prompt_text))

            return OpenRouterResponse(
                text="```\n/\\_/\\\n( o.o )\n > ^ <\n```",
                prompt_tokens=10,
                completion_tokens=50,
                total_tokens=60,
                cost=0.0001,
            )

        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.generate_async = AsyncMock(wraps=delayed_generate)

        # Test with limit=1 (sequential execution)
        config = GenerationConfig(
            attempts_per_prompt=1,
            max_concurrent_requests=1,
        )

        result = generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        # Verify tasks completed
        expected_tasks = len(sample_models) * len(sample_prompts) * 1
        assert len(result) == expected_tasks

        # Verify max concurrent calls was always 1 (sequential)
        assert max_concurrent_calls <= 1

        # Verify results match expected sequential order
        # Tasks are built in order: for model in models, for prompt in prompts,
        # for attempt in attempts
        expected_order = []
        for model in sample_models:
            for prompt in sample_prompts:
                expected_order.append((model.id, prompt.text))

        # Results should be in the same order as submitted tasks
        result_order = [(s.model_id, s.prompt_text) for s in result]
        assert result_order == expected_order, (
            f"Result order {result_order} does not match expected order {expected_order}"
        )

    def test_concurrent_mode_with_limit_greater_than_one(
        self,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        tmp_path: Path,
    ) -> None:
        """Negative case: with limit>1, tasks overlap (not sequential)."""
        import asyncio
        from unittest.mock import AsyncMock

        db_path = tmp_path / "database.jsonl"

        # Track concurrent calls
        concurrent_calls = 0
        max_concurrent_calls = 0
        call_lock = asyncio.Lock()

        async def delayed_generate(*args, **kwargs):
            """Mock generate with delay to allow concurrent execution."""
            nonlocal concurrent_calls, max_concurrent_calls

            async with call_lock:
                concurrent_calls += 1
                if concurrent_calls > max_concurrent_calls:
                    max_concurrent_calls = concurrent_calls

            # Simulate API delay to allow overlap
            await asyncio.sleep(0.05)

            async with call_lock:
                concurrent_calls -= 1

            return OpenRouterResponse(
                text="```\n/\\_/\\\n( o.o )\n > ^ <\n```",
                prompt_tokens=10,
                completion_tokens=50,
                total_tokens=60,
                cost=0.0001,
            )

        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.generate_async = AsyncMock(wraps=delayed_generate)

        # Test with limit=2 (allows 2 concurrent tasks, not sequential)
        config = GenerationConfig(
            attempts_per_prompt=1,
            max_concurrent_requests=2,
        )

        result = generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        # Verify tasks completed
        expected_tasks = len(sample_models) * len(sample_prompts) * 1
        assert len(result) == expected_tasks

        # Verify max concurrent calls exceeded 1 (not sequential)
        assert max_concurrent_calls > 1, (
            "Expected tasks to overlap with limit=2, "
            f"but max_concurrent_calls={max_concurrent_calls}"
        )


class TestConcurrentIdempotency:
    """Tests for atomic idempotency checks under concurrent execution."""

    @pytest.fixture
    def sample_models(self) -> list[Model]:
        """Create sample model list."""
        return [
            Model(id="openai/gpt-4o", name="GPT-4o"),
        ]

    @pytest.fixture
    def sample_prompts(self) -> list[Prompt]:
        """Create sample prompt list."""
        return [
            Prompt(text="Draw a cat", category="single_animal", template_type="animal"),
        ]

    def test_concurrent_idempotency_race_condition(
        self,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        tmp_path: Path,
    ) -> None:
        """Verify atomic idempotency checks prevent race conditions."""
        import asyncio
        from unittest.mock import AsyncMock

        db_path = tmp_path / "database.jsonl"

        # Track API calls to verify only one is made for duplicate keys
        api_call_count = 0
        call_lock = asyncio.Lock()

        async def tracked_generate(*args, **kwargs):
            """Mock generate with delay to allow concurrent race conditions."""
            nonlocal api_call_count

            # Simulate API delay to increase race condition window
            await asyncio.sleep(0.1)

            async with call_lock:
                api_call_count += 1

            return OpenRouterResponse(
                text="```\n/\\_/\\\n( o.o )\n > ^ <\n```",
                prompt_tokens=10,
                completion_tokens=50,
                total_tokens=60,
                cost=0.0001,
            )

        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.generate_async = AsyncMock(wraps=tracked_generate)

        # Create 5 identical tasks (same model, prompt, attempt)
        # This creates a race condition where multiple concurrent tasks
        # try to add the same key to SharedState
        num_tasks = 5
        identical_models = [sample_models[0]] * num_tasks
        identical_prompts = [sample_prompts[0]] * num_tasks

        config = GenerationConfig(
            attempts_per_prompt=1,
            max_concurrent_requests=num_tasks,
        )

        result = generate_samples(
            models=identical_models,
            prompts=identical_prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        # Verify only one task succeeded in generating the sample
        assert len(result) == 1, "Only one sample should be generated for duplicate keys"

        # Verify only one API call was made
        assert api_call_count == 1, "Only one API call should be made for duplicate keys"

        # Verify database has exactly one entry
        with open(db_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 1, "Database should have exactly one entry"

        # Verify the generated sample has correct properties
        sample = result[0]
        assert sample.model_id == "openai/gpt-4o"
        assert sample.prompt_text == "Draw a cat"
        assert sample.attempt_number == 1


class TestConcurrentMetricsAccuracy:
    """Tests for metrics accuracy under concurrent execution."""

    @pytest.fixture
    def sample_models(self) -> list[Model]:
        """Create sample model list."""
        return [
            Model(id="openai/gpt-4o", name="GPT-4o"),
        ]

    @pytest.fixture
    def sample_prompts(self) -> list[Prompt]:
        """Create sample prompt list."""
        return [
            Prompt(text="Draw a cat", category="single_animal", template_type="animal"),
            Prompt(text="Draw a dog", category="single_animal", template_type="animal"),
            Prompt(text="Draw a tree", category="single_object", template_type="object"),
            Prompt(text="Draw a bird", category="single_animal", template_type="animal"),
            Prompt(text="Draw a fish", category="single_animal", template_type="animal"),
        ]

    def test_concurrent_metrics_accuracy(
        self,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        tmp_path: Path,
    ) -> None:
        """Verify metrics are accurate under concurrent execution."""
        from unittest.mock import AsyncMock

        db_path = tmp_path / "database.jsonl"

        # Track metrics via stats_callback
        successful_count = 0
        failed_count = 0
        total_cost = 0.0

        def stats_callback(is_valid: bool, cost: float | None) -> None:
            """Track metrics from each sample generation."""
            nonlocal successful_count, failed_count, total_cost
            if is_valid:
                successful_count += 1
            else:
                failed_count += 1
            if cost is not None:
                total_cost += cost

        # Create 10 samples: 5 successful, 5 failed
        num_samples = 10
        successful_samples = 5
        failed_samples = 5

        # Create mock responses
        success_response = OpenRouterResponse(
            text="```\n/\\_/\\\n( o.o )\n > ^ <\n```",
            prompt_tokens=10,
            completion_tokens=50,
            total_tokens=60,
            cost=0.0001,
        )

        # Mix of successful and failed responses in a thread-safe queue
        mock_queue = asyncio.Queue()
        for i in range(num_samples):
            if i < successful_samples:
                mock_queue.put_nowait(success_response)
            else:
                mock_queue.put_nowait(OpenRouterClientError("API error"))

        async def mock_generate_async(*args, **kwargs):
            """Mock generate that returns mixed responses."""
            # Get the next response from the queue (thread-safe)
            response = await mock_queue.get()
            if isinstance(response, OpenRouterResponse):
                return response
            raise response

        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.generate_async = AsyncMock(side_effect=mock_generate_async)

        config = GenerationConfig(
            attempts_per_prompt=2,
            max_concurrent_requests=10,
        )

        result = generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
            stats_callback=stats_callback,
        )

        # Verify samples were generated
        assert len(result) == num_samples, f"Expected {num_samples} samples, got {len(result)}"

        # Verify successful count from callback
        assert successful_count == successful_samples, (
            f"Expected {successful_samples} successful samples, got {successful_count}"
        )

        # Verify failed count from callback
        assert failed_count == failed_samples, (
            f"Expected {failed_samples} failed samples, got {failed_count}"
        )

        # Verify cost from callback
        expected_cost = successful_samples * 0.0001
        assert total_cost == expected_cost, f"Expected cost {expected_cost}, got {total_cost}"

        # Verify samples in result
        result_successful = sum(1 for s in result if s.is_valid)
        result_failed = sum(1 for s in result if not s.is_valid)
        assert result_successful == successful_samples
        assert result_failed == failed_samples


class TestConcurrentPersistenceNoDuplicates:
    """Tests for ensuring no duplicate entries in persistence under concurrent execution."""

    @pytest.fixture
    def sample_models(self) -> list[Model]:
        """Create sample model list."""
        return [
            Model(id="openai/gpt-4o", name="GPT-4o"),
            Model(id="anthropic/claude-3-opus", name="Claude 3 Opus"),
            Model(id="google/gemini-pro", name="Gemini Pro"),
            Model(id="meta/llama-3-70b", name="Llama 3 70B"),
            Model(id="mistral/mistral-large", name="Mistral Large"),
        ]

    @pytest.fixture
    def sample_prompts(self) -> list[Prompt]:
        """Create sample prompt list."""
        return [
            Prompt(text="Draw a cat", category="single_animal", template_type="animal"),
            Prompt(text="Draw a dog", category="single_animal", template_type="animal"),
            Prompt(text="Draw a tree", category="single_object", template_type="object"),
        ]

    def test_concurrent_persistence_no_duplicates(
        self,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        tmp_path: Path,
    ) -> None:
        """Verify no duplicate entries in JSONL file under concurrent execution."""
        import json
        from unittest.mock import AsyncMock

        db_path = tmp_path / "database.jsonl"

        # Mock client with successful response
        mock_client = MagicMock(spec=OpenRouterClient)
        mock_client.generate_async = AsyncMock(
            return_value=OpenRouterResponse(
                text="```\n/\\_/\\\n( o.o )\n > ^ <\n```",
                prompt_tokens=10,
                completion_tokens=50,
                total_tokens=60,
                cost=0.0001,
            )
        )

        # Create config with attempts_per_prompt=3
        # This gives: 5 models x 3 prompts x 3 attempts = 45 tasks
        config = GenerationConfig(
            attempts_per_prompt=3,
            max_concurrent_requests=10,
        )

        # Run generation
        result = generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=config,
            database_path=db_path,
            client=mock_client,
        )

        # Verify all samples were generated
        expected_tasks = len(sample_models) * len(sample_prompts) * 3
        assert len(result) == expected_tasks, (
            f"Expected {expected_tasks} samples, got {len(result)}"
        )

        # Verify API calls were made
        assert mock_client.generate_async.call_count == expected_tasks

        # Read database file
        with open(db_path) as f:
            lines = [line.strip() for line in f if line.strip()]

        # Verify database has exactly 45 entries
        assert len(lines) == expected_tasks, (
            f"Expected {expected_tasks} entries in database, got {len(lines)}"
        )

        # Extract all (model_id, prompt_text, attempt_number) tuples
        sample_keys = []
        for line in lines:
            sample = json.loads(line)
            sample_keys.append(
                (sample["model_id"], sample["prompt_text"], sample["attempt_number"])
            )

        # Verify no duplicates
        unique_keys = set(sample_keys)
        assert len(unique_keys) == len(sample_keys), "Found duplicate entries in database"

        # Verify all expected combinations are present
        expected_keys = set()
        for model in sample_models:
            for prompt in sample_prompts:
                for attempt in range(1, 4):
                    expected_keys.add((model.id, prompt.text, attempt))

        assert unique_keys == expected_keys, "Database does not contain all expected combinations"

        # Negative case: if there were duplicates, the assertion above would fail
        # To verify this, we could artificially add a duplicate and check it fails
        # But this is already covered by the assertion len(unique_keys) == len(sample_keys)
