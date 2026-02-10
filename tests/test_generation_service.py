"""Tests for GenerationService on-demand sample generation."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from asciibench.common.config import GenerationConfig
from asciibench.common.models import ArtSample, Matchup, Prompt, RoundState
from asciibench.generator.client import OpenRouterClient, OpenRouterResponse
from asciibench.judge_ui.generation_service import GenerationService


class TestFindExistingSample:
    """Tests for find_existing_sample method."""

    @pytest.fixture
    def service(self, tmp_path: Path) -> GenerationService:
        """Create a GenerationService instance for tests."""
        mock_client = MagicMock(spec=OpenRouterClient)
        config = GenerationConfig()
        return GenerationService(mock_client, config, tmp_path / "database.jsonl")

    @pytest.fixture
    def sample_samples(self) -> list[ArtSample]:
        """Create a sample list of ArtSample objects."""
        return [
            ArtSample(
                model_id="model-a",
                prompt_text="Draw a cat",
                category="animal",
                attempt_number=1,
                raw_output="```text\n/\\_/\\\n( o.o )\n > ^ <\n```",
                sanitized_output="/\\_/\\\n( o.o )\n > ^ <",
                is_valid=True,
            ),
            ArtSample(
                model_id="model-b",
                prompt_text="Draw a dog",
                category="animal",
                attempt_number=1,
                raw_output=(
                    "```text\n  / \\__\n (    @\\___\n /         O\n /   (_____/\n/_____/   U\n```"
                ),
                sanitized_output=(
                    "  / \\__\n (    @\\___\n /         O\n /   (_____/\n/_____/   U"
                ),
                is_valid=True,
            ),
            ArtSample(
                model_id="model-a",
                prompt_text="Draw a bird",
                category="animal",
                attempt_number=1,
                raw_output="```text\n   (v)\n  ((  \n  / \\\n```",
                sanitized_output="   (v)\n  ((  \n  / \\\n",
                is_valid=True,
            ),
        ]

    def test_find_existing_sample_returns_matching_sample(
        self, service: GenerationService, sample_samples: list[ArtSample]
    ) -> None:
        """Test find_existing_sample returns matching valid sample when one exists."""
        result = service.find_existing_sample("model-a", "Draw a cat", sample_samples)

        assert result is not None
        assert result.model_id == "model-a"
        assert result.prompt_text == "Draw a cat"

    def test_find_existing_sample_returns_none_no_match(
        self, service: GenerationService, sample_samples: list[ArtSample]
    ) -> None:
        """Test find_existing_sample returns None when no match exists."""
        result = service.find_existing_sample("model-c", "Draw a fish", sample_samples)

        assert result is None

    def test_find_existing_sample_returns_first_match(
        self, service: GenerationService, sample_samples: list[ArtSample]
    ) -> None:
        """Test find_existing_sample returns first matching sample when multiple exist."""
        result = service.find_existing_sample("model-a", "Draw a cat", sample_samples)

        assert result is not None
        assert result.model_id == "model-a"
        assert result.prompt_text == "Draw a cat"
        assert result.id == sample_samples[0].id

    def test_find_existing_sample_different_model_same_prompt(
        self, service: GenerationService, sample_samples: list[ArtSample]
    ) -> None:
        """Test find_existing_sample with different model but same prompt."""
        result = service.find_existing_sample("model-b", "Draw a cat", sample_samples)

        assert result is None

    def test_find_existing_sample_same_model_different_prompt(
        self, service: GenerationService, sample_samples: list[ArtSample]
    ) -> None:
        """Test find_existing_sample with same model but different prompt."""
        result = service.find_existing_sample("model-a", "Draw a fish", sample_samples)

        assert result is None

    def test_find_existing_sample_empty_list(self, service: GenerationService) -> None:
        """Test find_existing_sample with empty samples list."""
        result = service.find_existing_sample("model-a", "Draw a cat", [])

        assert result is None


class TestGenerateSample:
    """Tests for generate_sample method."""

    @pytest.fixture
    def service(self, tmp_path: Path) -> GenerationService:
        """Create a GenerationService instance for tests."""
        mock_client = MagicMock()
        config = GenerationConfig()
        return GenerationService(mock_client, config, tmp_path / "database.jsonl")

    @pytest.fixture
    def sample_prompt(self) -> Prompt:
        """Create a sample Prompt object."""
        return Prompt(text="Draw a cat", category="animal", template_type="simple")

    @pytest.mark.asyncio
    async def test_generate_sample_calls_dependencies(
        self, service: GenerationService, sample_prompt: Prompt
    ) -> None:
        """Test generate_sample calls API client and dependencies with correct args."""
        mock_response = OpenRouterResponse(
            text="```text\n/\\_/\\\n( o.o )\n > ^ <\n```",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost=0.0001,
        )
        service.client.generate_async = AsyncMock(return_value=mock_response)  # type: ignore[assignment]

        with (
            patch(
                "asciibench.judge_ui.generation_service.extract_ascii_from_markdown",
                return_value="/\\_/\\\n( o.o )\n > ^ <",
            ) as mock_extract,
            patch("asciibench.judge_ui.generation_service.append_jsonl") as mock_append,
        ):
            result = await service.generate_sample("model-a", sample_prompt)

        service.client.generate_async.assert_called_once_with(
            model_id="model-a",
            prompt="Draw a cat",
            config=service.config,
        )

        mock_extract.assert_called_once_with(mock_response.text)

        mock_append.assert_called_once_with(
            service.database_path,
            result,
        )

        assert result.model_id == "model-a"
        assert result.prompt_text == "Draw a cat"
        assert result.sanitized_output == "/\\_/\\\n( o.o )\n > ^ <"
        assert result.is_valid is True
        assert result.output_tokens == 20
        assert result.cost == 0.0001

    @pytest.mark.asyncio
    async def test_generate_sample_empty_sanitized_output(
        self, service: GenerationService, sample_prompt: Prompt
    ) -> None:
        """Test generate_sample with empty sanitized output marks as invalid."""
        mock_response = OpenRouterResponse(
            text="No code block here",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost=0.00005,
        )
        service.client.generate_async = AsyncMock(return_value=mock_response)  # type: ignore[assignment]

        with (
            patch(
                "asciibench.judge_ui.generation_service.extract_ascii_from_markdown",
                return_value="",
            ),
            patch("asciibench.judge_ui.generation_service.append_jsonl"),
        ):
            result = await service.generate_sample("model-a", sample_prompt)

        assert result.is_valid is False
        assert result.sanitized_output == ""

    @pytest.mark.asyncio
    async def test_generate_sample_with_none_tokens(
        self, service: GenerationService, sample_prompt: Prompt
    ) -> None:
        """Test generate_sample handles None token counts."""
        mock_response = OpenRouterResponse(
            text="```text\n/\\_/\\\n```",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            cost=None,
        )
        service.client.generate_async = AsyncMock(return_value=mock_response)  # type: ignore[assignment]

        with (
            patch(
                "asciibench.judge_ui.generation_service.extract_ascii_from_markdown",
                return_value="/\\_/\\",
            ),
            patch("asciibench.judge_ui.generation_service.append_jsonl"),
        ):
            result = await service.generate_sample("model-a", sample_prompt)

        assert result.output_tokens is None
        assert result.cost is None


class TestEnsureSamplesForRound:
    """Tests for ensure_samples_for_round method."""

    @pytest.fixture
    def service(self, tmp_path: Path) -> GenerationService:
        """Create a GenerationService instance for tests."""
        mock_client = MagicMock()
        config = GenerationConfig()
        return GenerationService(mock_client, config, tmp_path / "database.jsonl")

    @pytest.fixture
    def sample_existing_samples(self) -> list[ArtSample]:
        """Create a sample list of existing samples."""
        return [
            ArtSample(
                model_id="model-a",
                prompt_text="Draw a cat",
                category="animal",
                attempt_number=1,
                raw_output="```text\n/\\_/\\\n```",
                sanitized_output="/\\_/\\",
                is_valid=True,
            ),
            ArtSample(
                model_id="model-b",
                prompt_text="Draw a cat",
                category="animal",
                attempt_number=1,
                raw_output="```text\n  / \\__\n```",
                sanitized_output="  / \\__",
                is_valid=True,
            ),
        ]

    @pytest.fixture
    def sample_round_state(self) -> RoundState:
        """Create a sample RoundState with matchups."""
        return RoundState(
            round_number=1,
            matchups=[
                Matchup(
                    model_a_id="model-a",
                    model_b_id="model-b",
                    prompt_text="Draw a cat",
                    prompt_category="animal",
                ),
                Matchup(
                    model_a_id="model-a",
                    model_b_id="model-c",
                    prompt_text="Draw a dog",
                    prompt_category="animal",
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_ensure_samples_for_round_skips_generation(
        self,
        service: GenerationService,
        sample_round_state: RoundState,
        sample_existing_samples: list[ArtSample],
    ) -> None:
        """Test ensure_samples_for_round skips generation for matchups where both samples exist."""
        round_state_with_single_matchup = RoundState(
            round_number=1,
            matchups=[
                Matchup(
                    model_a_id="model-a",
                    model_b_id="model-b",
                    prompt_text="Draw a cat",
                    prompt_category="animal",
                ),
            ],
        )

        with patch.object(service, "generate_sample", new_callable=AsyncMock) as mock_generate:
            result = await service.ensure_samples_for_round(
                round_state_with_single_matchup, sample_existing_samples
            )

        mock_generate.assert_not_called()

        assert len(result.matchups) == 1
        assert result.matchups[0].sample_a_id is not None
        assert result.matchups[0].sample_b_id is not None
        assert result.generation_complete is True

    @pytest.mark.asyncio
    async def test_ensure_samples_for_round_generates_missing_samples(
        self,
        service: GenerationService,
        sample_round_state: RoundState,
        sample_existing_samples: list[ArtSample],
    ) -> None:
        """Test ensure_samples_for_round generates missing samples and fills in IDs."""
        mock_response = OpenRouterResponse(
            text="```text\n  O\n /|\\\n  |\\\n```",
            prompt_tokens=10,
            completion_tokens=15,
            total_tokens=25,
            cost=0.0001,
        )
        service.client.generate_async = AsyncMock(return_value=mock_response)  # type: ignore[assignment]

        with (
            patch(
                "asciibench.judge_ui.generation_service.extract_ascii_from_markdown",
                return_value="  O\n /|\\\n  |\\",
            ),
            patch("asciibench.judge_ui.generation_service.append_jsonl"),
        ):
            result = await service.ensure_samples_for_round(
                sample_round_state, sample_existing_samples
            )

        assert len(result.matchups) == 2

        first_matchup = result.matchups[0]
        assert first_matchup.sample_a_id is not None
        assert first_matchup.sample_b_id is not None

        second_matchup = result.matchups[1]
        assert second_matchup.sample_a_id is not None
        assert second_matchup.sample_b_id is not None

        assert result.generation_complete is True

    @pytest.mark.asyncio
    async def test_ensure_samples_for_round_sets_generation_complete(
        self,
        service: GenerationService,
        sample_round_state: RoundState,
        sample_existing_samples: list[ArtSample],
    ) -> None:
        """Test ensure_samples_for_round sets generation_complete=True."""
        mock_response = OpenRouterResponse(
            text="```text\n  O\n /|\\\n```",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost=0.0001,
        )
        service.client.generate_async = AsyncMock(return_value=mock_response)  # type: ignore[assignment]

        with (
            patch(
                "asciibench.judge_ui.generation_service.extract_ascii_from_markdown",
                return_value="  O\n /|\\",
            ),
            patch("asciibench.judge_ui.generation_service.append_jsonl"),
        ):
            result = await service.ensure_samples_for_round(
                sample_round_state, sample_existing_samples
            )

        assert result.generation_complete is True

    @pytest.mark.asyncio
    async def test_ensure_samples_for_round_updates_existing_samples_list(
        self, service: GenerationService, sample_round_state: RoundState
    ) -> None:
        """Test ensure_samples_for_round adds generated samples to existing_samples list."""
        mock_response = OpenRouterResponse(
            text="```text\n  O\n /|\\\n```",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost=0.0001,
        )
        service.client.generate_async = AsyncMock(return_value=mock_response)  # type: ignore[assignment]

        existing_samples = []

        with (
            patch(
                "asciibench.judge_ui.generation_service.extract_ascii_from_markdown",
                return_value="  O\n /|\\",
            ),
            patch("asciibench.judge_ui.generation_service.append_jsonl"),
        ):
            await service.ensure_samples_for_round(sample_round_state, existing_samples)

        assert len(existing_samples) == 4

    @pytest.mark.asyncio
    async def test_ensure_samples_for_round_generates_both_samples_when_none_exist(
        self, service: GenerationService, sample_round_state: RoundState
    ) -> None:
        """Test ensure_samples_for_round generates both samples when neither exists."""
        mock_response = OpenRouterResponse(
            text="```text\n  O\n /|\\\n```",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            cost=0.0001,
        )
        service.client.generate_async = AsyncMock(return_value=mock_response)  # type: ignore[assignment]

        existing_samples = []

        with (
            patch(
                "asciibench.judge_ui.generation_service.extract_ascii_from_markdown",
                return_value="  O\n /|\\",
            ),
            patch("asciibench.judge_ui.generation_service.append_jsonl"),
        ):
            result = await service.ensure_samples_for_round(sample_round_state, existing_samples)

        first_matchup = result.matchups[0]
        assert first_matchup.sample_a_id is not None
        assert first_matchup.sample_b_id is not None

        second_matchup = result.matchups[1]
        assert second_matchup.sample_a_id is not None
        assert second_matchup.sample_b_id is not None

        assert result.generation_complete is True
