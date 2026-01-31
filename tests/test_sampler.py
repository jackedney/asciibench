"""Tests for the sampler module."""

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
