"""Tests for GenerationService on-demand sample generation."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from asciibench.common.config import GenerationConfig
from asciibench.common.models import ArtSample, Matchup, RoundState
from asciibench.generator.client import OpenRouterClient
from asciibench.generator.concurrent import GenerationTask
from asciibench.judge_ui.generation_service import GenerationService


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

        with patch(
            "asciibench.judge_ui.generation_service.generate_samples_concurrent",
            new_callable=AsyncMock,
        ) as mock_concurrent:
            result = await service.ensure_samples_for_round(
                round_state_with_single_matchup, sample_existing_samples
            )

            mock_concurrent.assert_not_called()

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
        new_sample_a = ArtSample(
            model_id="model-a",
            prompt_text="Draw a dog",
            category="animal",
            attempt_number=1,
            raw_output="```text\n  O\n```",
            sanitized_output="  O",
            is_valid=True,
        )
        new_sample_c = ArtSample(
            model_id="model-c",
            prompt_text="Draw a dog",
            category="animal",
            attempt_number=1,
            raw_output="```text\n /|\\\n```",
            sanitized_output=" /|\\",
            is_valid=True,
        )

        with patch(
            "asciibench.judge_ui.generation_service.generate_samples_concurrent",
            new_callable=AsyncMock,
            return_value=[new_sample_a, new_sample_c],
        ) as mock_concurrent:
            result = await service.ensure_samples_for_round(
                sample_round_state, sample_existing_samples
            )

            mock_concurrent.assert_called_once()
            call_args = mock_concurrent.call_args
            tasks = call_args.kwargs["tasks"]
            assert len(tasks) == 2
            task_keys = {(t.model_id, t.prompt_text) for t in tasks}
            assert ("model-a", "Draw a dog") in task_keys
            assert ("model-c", "Draw a dog") in task_keys

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
        new_sample_a = ArtSample(
            model_id="model-a",
            prompt_text="Draw a dog",
            category="animal",
            attempt_number=1,
            raw_output="```text\n  O\n```",
            sanitized_output="  O",
            is_valid=True,
        )
        new_sample_c = ArtSample(
            model_id="model-c",
            prompt_text="Draw a dog",
            category="animal",
            attempt_number=1,
            raw_output="```text\n /|\\\n```",
            sanitized_output=" /|\\",
            is_valid=True,
        )

        with patch(
            "asciibench.judge_ui.generation_service.generate_samples_concurrent",
            new_callable=AsyncMock,
            return_value=[new_sample_a, new_sample_c],
        ):
            result = await service.ensure_samples_for_round(
                sample_round_state, sample_existing_samples
            )

        assert result.generation_complete is True

    @pytest.mark.asyncio
    async def test_ensure_samples_for_round_generates_both_samples_when_none_exist(
        self, service: GenerationService, sample_round_state: RoundState
    ) -> None:
        """Test ensure_samples_for_round generates both samples when neither exists."""
        new_samples = [
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
            ArtSample(
                model_id="model-a",
                prompt_text="Draw a dog",
                category="animal",
                attempt_number=1,
                raw_output="```text\n  O\n```",
                sanitized_output="  O",
                is_valid=True,
            ),
            ArtSample(
                model_id="model-c",
                prompt_text="Draw a dog",
                category="animal",
                attempt_number=1,
                raw_output="```text\n /|\\\n```",
                sanitized_output=" /|\\",
                is_valid=True,
            ),
        ]

        existing_samples: list[ArtSample] = []

        with patch(
            "asciibench.judge_ui.generation_service.generate_samples_concurrent",
            new_callable=AsyncMock,
            return_value=new_samples,
        ) as mock_concurrent:
            result = await service.ensure_samples_for_round(sample_round_state, existing_samples)

            call_args = mock_concurrent.call_args
            tasks = call_args.kwargs["tasks"]
            assert len(tasks) == 4

        first_matchup = result.matchups[0]
        assert first_matchup.sample_a_id is not None
        assert first_matchup.sample_b_id is not None

        second_matchup = result.matchups[1]
        assert second_matchup.sample_a_id is not None
        assert second_matchup.sample_b_id is not None

        assert result.generation_complete is True

    @pytest.mark.asyncio
    async def test_ensure_samples_for_round_calls_callback(
        self,
        service: GenerationService,
        sample_round_state: RoundState,
        sample_existing_samples: list[ArtSample],
    ) -> None:
        """Test ensure_samples_for_round calls callback for each matchup with correct index."""
        new_sample_a = ArtSample(
            model_id="model-a",
            prompt_text="Draw a dog",
            category="animal",
            attempt_number=1,
            raw_output="```text\n  O\n```",
            sanitized_output="  O",
            is_valid=True,
        )
        new_sample_c = ArtSample(
            model_id="model-c",
            prompt_text="Draw a dog",
            category="animal",
            attempt_number=1,
            raw_output="```text\n /|\\\n```",
            sanitized_output=" /|\\",
            is_valid=True,
        )

        round_state_3_matchups = RoundState(
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
                Matchup(
                    model_a_id="model-d",
                    model_b_id="model-e",
                    prompt_text="Draw a bird",
                    prompt_category="animal",
                ),
            ],
        )

        new_samples_for_round = [
            ArtSample(
                model_id="model-d",
                prompt_text="Draw a bird",
                category="animal",
                attempt_number=1,
                raw_output="```text\n (v)\n```",
                sanitized_output=" (v)",
                is_valid=True,
            ),
            ArtSample(
                model_id="model-e",
                prompt_text="Draw a bird",
                category="animal",
                attempt_number=1,
                raw_output="```text\n (V)\n```",
                sanitized_output=" (V)",
                is_valid=True,
            ),
        ]

        callback_calls: list[tuple[int, Matchup]] = []

        def callback(index: int, matchup: Matchup) -> None:
            callback_calls.append((index, matchup))

        with patch(
            "asciibench.judge_ui.generation_service.generate_samples_concurrent",
            new_callable=AsyncMock,
            return_value=[new_sample_a, new_sample_c, *new_samples_for_round],
        ):
            await service.ensure_samples_for_round(
                round_state_3_matchups, sample_existing_samples, on_matchup_ready=callback
            )

        assert len(callback_calls) == 3

        for i in range(3):
            index, matchup = callback_calls[i]
            assert index == i
            assert matchup.sample_a_id is not None
            assert matchup.sample_b_id is not None

    @pytest.mark.asyncio
    async def test_ensure_samples_for_round_no_callback_when_none(
        self,
        service: GenerationService,
        sample_round_state: RoundState,
        sample_existing_samples: list[ArtSample],
    ) -> None:
        """Test ensure_samples_for_round does not call callback when parameter is None."""
        new_sample_a = ArtSample(
            model_id="model-a",
            prompt_text="Draw a dog",
            category="animal",
            attempt_number=1,
            raw_output="```text\n  O\n```",
            sanitized_output="  O",
            is_valid=True,
        )
        new_sample_c = ArtSample(
            model_id="model-c",
            prompt_text="Draw a dog",
            category="animal",
            attempt_number=1,
            raw_output="```text\n /|\\\n```",
            sanitized_output=" /|\\",
            is_valid=True,
        )

        with patch(
            "asciibench.judge_ui.generation_service.generate_samples_concurrent",
            new_callable=AsyncMock,
            return_value=[new_sample_a, new_sample_c],
        ):
            result = await service.ensure_samples_for_round(
                sample_round_state, sample_existing_samples, on_matchup_ready=None
            )

        assert result.generation_complete is True

    @pytest.mark.asyncio
    async def test_ensure_samples_for_round_deduplicates_pairs(
        self,
        service: GenerationService,
        sample_existing_samples: list[ArtSample],
    ) -> None:
        """Test ensure_samples_for_round generates only missing unique (model, prompt) pairs."""
        round_state_duplicate_prompt = RoundState(
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
                    prompt_text="Draw a cat",
                    prompt_category="animal",
                ),
            ],
        )

        new_sample_c = ArtSample(
            model_id="model-c",
            prompt_text="Draw a cat",
            category="animal",
            attempt_number=1,
            raw_output="```text\n  O\n```",
            sanitized_output="  O",
            is_valid=True,
        )

        with patch(
            "asciibench.judge_ui.generation_service.generate_samples_concurrent",
            new_callable=AsyncMock,
            return_value=[new_sample_c],
        ) as mock_concurrent:
            result = await service.ensure_samples_for_round(
                round_state_duplicate_prompt, sample_existing_samples
            )

            mock_concurrent.assert_called_once()
            call_args = mock_concurrent.call_args
            tasks: list[GenerationTask] = call_args.kwargs["tasks"]
            assert len(tasks) == 1
            assert tasks[0].model_id == "model-c"
            assert tasks[0].prompt_text == "Draw a cat"

        for matchup in result.matchups:
            assert matchup.sample_a_id is not None
            assert matchup.sample_b_id is not None

    @pytest.mark.asyncio
    async def test_ensure_samples_for_round_concurrent_with_missing_model_c(
        self,
        service: GenerationService,
        sample_existing_samples: list[ArtSample],
    ) -> None:
        """Test ensure_samples_for_round generates for model-c when it appears in matchups."""
        round_state = RoundState(
            round_number=1,
            matchups=[
                Matchup(
                    model_a_id="model-c",
                    model_b_id="model-d",
                    prompt_text="Draw a dog",
                    prompt_category="animal",
                ),
            ],
        )

        new_sample_c = ArtSample(
            model_id="model-c",
            prompt_text="Draw a dog",
            category="animal",
            attempt_number=1,
            raw_output="```text\n  O\n```",
            sanitized_output="  O",
            is_valid=True,
        )
        new_sample_d = ArtSample(
            model_id="model-d",
            prompt_text="Draw a dog",
            category="animal",
            attempt_number=1,
            raw_output="```text\n /|\\\n```",
            sanitized_output=" /|\\",
            is_valid=True,
        )

        with patch(
            "asciibench.judge_ui.generation_service.generate_samples_concurrent",
            new_callable=AsyncMock,
            return_value=[new_sample_c, new_sample_d],
        ) as mock_concurrent:
            result = await service.ensure_samples_for_round(round_state, sample_existing_samples)

            call_args = mock_concurrent.call_args
            tasks: list[GenerationTask] = call_args.kwargs["tasks"]
            task_keys = {(t.model_id, t.prompt_text) for t in tasks}
            assert ("model-c", "Draw a dog") in task_keys
            assert ("model-d", "Draw a dog") in task_keys
            assert len(tasks) == 2

        assert result.matchups[0].sample_a_id is not None
        assert result.matchups[0].sample_b_id is not None
