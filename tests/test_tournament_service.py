"""Tests for TournamentService round orchestration."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest

from asciibench.common.models import Matchup, Model, Prompt, RoundState, Vote
from asciibench.judge_ui.generation_service import GenerationService
from asciibench.judge_ui.tournament_service import TournamentService


class TestInitialize:
    """Tests for initialize method."""

    @pytest.fixture
    def mock_generation_service(self) -> MagicMock:
        """Create a mock GenerationService."""
        return MagicMock(spec=GenerationService)

    @pytest.fixture
    def mock_config_service(self) -> MagicMock:
        """Create a mock ConfigService."""
        config = MagicMock()
        config.get_models.return_value = [
            Model(id="model-a", name="Model A"),
            Model(id="model-b", name="Model B"),
            Model(id="model-c", name="Model C"),
        ]
        config.get_prompts.return_value = [
            Prompt(text="Draw a cat", category="animal", template_type="simple"),
            Prompt(text="Draw a dog", category="animal", template_type="simple"),
            Prompt(text="Draw a bird", category="animal", template_type="simple"),
        ]
        return config

    @pytest.fixture
    def mock_repo(self, tmp_path: Path) -> MagicMock:
        """Create a mock DataRepository."""
        repo = MagicMock()
        repo.get_all_samples.return_value = []
        repo.get_votes.return_value = []
        return repo

    @pytest.fixture
    def service(
        self,
        mock_generation_service: MagicMock,
        mock_config_service: MagicMock,
        mock_repo: MagicMock,
    ) -> TournamentService:
        """Create a TournamentService instance for tests."""
        return TournamentService(
            generation_service=mock_generation_service,
            config_service=mock_config_service,
            repo=mock_repo,
            n=1,
        )

    @pytest.mark.asyncio
    async def test_initialize_creates_round_1_without_existing_rounds(
        self,
        service: TournamentService,
        mock_generation_service: MagicMock,
        mock_config_service: MagicMock,
        mock_repo: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test initialize() with no existing rounds.jsonl creates round 1 with all random pairs."""
        service._rounds_path = tmp_path / "rounds.jsonl"

        def ensure_samples_for_round_impl(round_state, samples, on_matchup_ready=None):
            """Mock implementation that marks generation complete."""
            updated_matchups = []
            for matchup in round_state.matchups:
                updated_matchup = matchup.model_copy(
                    update={
                        "sample_a_id": str(uuid4()),
                        "sample_b_id": str(uuid4()),
                    }
                )
                updated_matchups.append(updated_matchup)
            return round_state.model_copy(
                update={
                    "matchups": updated_matchups,
                    "generation_complete": True,
                }
            )

        mock_generation_service.ensure_samples_for_round = AsyncMock(
            side_effect=ensure_samples_for_round_impl
        )

        with patch.object(service, "_start_background_generation"):
            await service.initialize()

        if service._initial_generation_task is not None:
            await service._initial_generation_task

        assert service._current_round is not None
        assert service._current_round.round_number == 1
        assert len(service._current_round.matchups) > 0
        assert service._current_round.elo_snapshot == {}
        assert service._current_round.generation_complete is True
        mock_generation_service.ensure_samples_for_round.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_recovers_last_round_state(
        self,
        service: TournamentService,
        mock_generation_service: MagicMock,
        mock_config_service: MagicMock,
        mock_repo: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test initialize() with existing rounds.jsonl recovers last round state."""
        rounds_path = tmp_path / "rounds.jsonl"
        service._rounds_path = rounds_path

        vote_id = str(uuid4())
        existing_round = RoundState(
            id=uuid4(),
            round_number=1,
            matchups=[
                Matchup(
                    id=uuid4(),
                    model_a_id="model-a",
                    model_b_id="model-b",
                    prompt_text="Draw a cat",
                    prompt_category="animal",
                    is_judged=True,
                    vote_id=vote_id,
                ),
            ],
            generation_complete=True,
        )

        from asciibench.common.persistence import append_jsonl

        append_jsonl(rounds_path, existing_round)

        mock_vote = Vote(
            id=uuid4(),
            sample_a_id=str(uuid4()),
            sample_b_id=str(uuid4()),
            winner="A",
        )
        mock_repo.get_votes.return_value = [mock_vote]

        with patch.object(service, "_start_background_generation"):
            await service.initialize()

        assert service._current_round is not None
        assert service._current_round.round_number == 1
        assert len(service._current_round.matchups) == 1

    @pytest.mark.asyncio
    async def test_initialize_non_blocking(
        self,
        service: TournamentService,
        mock_generation_service: MagicMock,
        mock_config_service: MagicMock,
        mock_repo: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test initialize() returns before _initial_generation_task completes."""
        service._rounds_path = tmp_path / "rounds.jsonl"

        generation_started = asyncio.Event()
        generation_can_complete = asyncio.Event()

        async def slow_ensure_samples_for_round(round_state, samples, on_matchup_ready=None):
            generation_started.set()
            await generation_can_complete.wait()
            updated_matchups = []
            for matchup in round_state.matchups:
                updated_matchup = matchup.model_copy(
                    update={
                        "sample_a_id": str(uuid4()),
                        "sample_b_id": str(uuid4()),
                    }
                )
                updated_matchups.append(updated_matchup)
            return round_state.model_copy(
                update={
                    "matchups": updated_matchups,
                    "generation_complete": True,
                }
            )

        mock_generation_service.ensure_samples_for_round = AsyncMock(
            side_effect=slow_ensure_samples_for_round
        )

        with patch.object(service, "_start_background_generation"):
            initialize_task = asyncio.create_task(service.initialize())

        await generation_started.wait()

        assert service._current_round is not None
        assert service._initial_generation_task is not None
        assert not service._initial_generation_task.done()
        assert service._current_round.generation_complete is False

        generation_can_complete.set()
        await initialize_task

        if service._initial_generation_task is not None:
            await service._initial_generation_task

        assert service._current_round.generation_complete is True


class TestGetNextMatchup:
    """Tests for get_next_matchup method."""

    @pytest.fixture
    def service(self, tmp_path: Path) -> TournamentService:
        """Create a TournamentService instance for tests."""
        mock_generation_service = MagicMock(spec=GenerationService)
        mock_config_service = MagicMock()
        mock_config_service.get_models.return_value = []
        mock_config_service.get_prompts.return_value = []
        mock_repo = MagicMock()
        mock_repo.get_all_samples.return_value = []
        mock_repo.get_votes.return_value = []

        service = TournamentService(
            generation_service=mock_generation_service,
            config_service=mock_config_service,
            repo=mock_repo,
            n=1,
        )
        service._rounds_path = tmp_path / "rounds.jsonl"
        return service

    def test_get_next_matchup_returns_unjudged_matchup(self, service: TournamentService) -> None:
        """Test get_next_matchup() returns unjudged matchup from current round."""
        unjudged_matchup = Matchup(
            id=uuid4(),
            model_a_id="model-a",
            model_b_id="model-b",
            prompt_text="Draw a cat",
            prompt_category="animal",
            is_judged=False,
            sample_a_id=str(uuid4()),
            sample_b_id=str(uuid4()),
        )
        judged_matchup = Matchup(
            id=uuid4(),
            model_a_id="model-a",
            model_b_id="model-c",
            prompt_text="Draw a dog",
            prompt_category="animal",
            is_judged=True,
            sample_a_id=str(uuid4()),
            sample_b_id=str(uuid4()),
        )

        service._current_round = RoundState(
            id=uuid4(),
            round_number=1,
            matchups=[judged_matchup, unjudged_matchup],
        )

        result = service.get_next_matchup()

        assert result is not None
        assert result.id == unjudged_matchup.id
        assert result.is_judged is False

    def test_get_next_matchup_returns_none_all_judged(self, service: TournamentService) -> None:
        """Test get_next_matchup() returns None when all matchups judged."""
        judged_matchup = Matchup(
            id=uuid4(),
            model_a_id="model-a",
            model_b_id="model-b",
            prompt_text="Draw a cat",
            prompt_category="animal",
            is_judged=True,
            sample_a_id=str(uuid4()),
            sample_b_id=str(uuid4()),
        )

        service._current_round = RoundState(
            id=uuid4(),
            round_number=1,
            matchups=[judged_matchup],
        )

        result = service.get_next_matchup()

        assert result is None

    def test_get_next_matchup_returns_none_no_current_round(
        self, service: TournamentService
    ) -> None:
        """Test get_next_matchup() returns None when no current round."""
        service._current_round = None

        result = service.get_next_matchup()

        assert result is None

    def test_get_next_matchup_returns_none_without_both_samples(
        self, service: TournamentService
    ) -> None:
        """Test get_next_matchup() returns None when matchup lacks both samples."""
        matchup_with_only_sample_a = Matchup(
            id=uuid4(),
            model_a_id="model-a",
            model_b_id="model-b",
            prompt_text="Draw a cat",
            prompt_category="animal",
            is_judged=False,
            sample_a_id=str(uuid4()),
            sample_b_id=None,
        )
        matchup_with_only_sample_b = Matchup(
            id=uuid4(),
            model_a_id="model-a",
            model_b_id="model-c",
            prompt_text="Draw a dog",
            prompt_category="animal",
            is_judged=False,
            sample_a_id=None,
            sample_b_id=str(uuid4()),
        )
        matchup_with_no_samples = Matchup(
            id=uuid4(),
            model_a_id="model-a",
            model_b_id="model-d",
            prompt_text="Draw a bird",
            prompt_category="animal",
            is_judged=False,
            sample_a_id=None,
            sample_b_id=None,
        )

        service._current_round = RoundState(
            id=uuid4(),
            round_number=1,
            matchups=[
                matchup_with_only_sample_a,
                matchup_with_only_sample_b,
                matchup_with_no_samples,
            ],
        )

        result = service.get_next_matchup()

        assert result is None

    def test_get_next_matchup_filters_incomplete_matchups(self, service: TournamentService) -> None:
        """Test get_next_matchup() only returns matchups with both samples."""
        incomplete_matchup = Matchup(
            id=uuid4(),
            model_a_id="model-a",
            model_b_id="model-b",
            prompt_text="Draw a cat",
            prompt_category="animal",
            is_judged=False,
            sample_a_id=str(uuid4()),
            sample_b_id=None,
        )
        complete_matchup = Matchup(
            id=uuid4(),
            model_a_id="model-a",
            model_b_id="model-c",
            prompt_text="Draw a dog",
            prompt_category="animal",
            is_judged=False,
            sample_a_id=str(uuid4()),
            sample_b_id=str(uuid4()),
        )

        service._current_round = RoundState(
            id=uuid4(),
            round_number=1,
            matchups=[incomplete_matchup, complete_matchup],
        )

        result = service.get_next_matchup()

        assert result is not None
        assert result.id == complete_matchup.id


class TestRecordVote:
    """Tests for record_vote method."""

    @pytest.fixture
    def service(self, tmp_path: Path) -> TournamentService:
        """Create a TournamentService instance for tests."""
        mock_generation_service = Mock(spec=GenerationService)
        mock_generation_service.ensure_samples_for_round = AsyncMock(
            side_effect=lambda rs, s, on_matchup_ready=None: rs.model_copy(
                update={"generation_complete": True}
            )
        )
        mock_config_service = MagicMock()
        mock_config_service.get_models.return_value = []
        mock_config_service.get_prompts.return_value = []
        mock_repo = MagicMock()
        mock_repo.get_all_samples.return_value = []
        mock_repo.get_votes.return_value = []

        service = TournamentService(
            generation_service=mock_generation_service,
            config_service=mock_config_service,
            repo=mock_repo,
            n=1,
        )
        service._rounds_path = tmp_path / "rounds.jsonl"
        return service

    @pytest.mark.asyncio
    async def test_record_vote_marks_judged_and_persists(
        self, service: TournamentService, tmp_path: Path
    ) -> None:
        """Test record_vote() marks matchup as judged and persists state."""
        matchup_id = uuid4()
        vote_id = str(uuid4())
        matchup = Matchup(
            id=matchup_id,
            model_a_id="model-a",
            model_b_id="model-b",
            prompt_text="Draw a cat",
            prompt_category="animal",
            is_judged=False,
        )

        another_matchup = Matchup(
            id=uuid4(),
            model_a_id="model-a",
            model_b_id="model-c",
            prompt_text="Draw a dog",
            prompt_category="animal",
            is_judged=False,
        )

        service._current_round = RoundState(
            id=uuid4(),
            round_number=1,
            matchups=[matchup, another_matchup],
            all_judged=False,
        )

        with patch("asciibench.judge_ui.tournament_service.append_jsonl") as mock_append:
            await service.record_vote(matchup_id, vote_id)

        assert service._current_round.all_judged is False
        updated_matchup = next(
            (m for m in service._current_round.matchups if m.id == matchup_id), None
        )
        assert updated_matchup is not None
        assert updated_matchup.is_judged is True
        assert updated_matchup.vote_id == vote_id
        mock_append.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_vote_triggers_round_completion_on_last_matchup(
        self, service: TournamentService, tmp_path: Path
    ) -> None:
        """Test record_vote() on last unjudged matchup triggers round completion."""
        matchup_id = uuid4()
        vote_id = str(uuid4())
        matchup = Matchup(
            id=matchup_id,
            model_a_id="model-a",
            model_b_id="model-b",
            prompt_text="Draw a cat",
            prompt_category="animal",
            is_judged=False,
        )

        service._current_round = RoundState(
            id=uuid4(),
            round_number=1,
            matchups=[matchup],
        )

        with (
            patch("asciibench.judge_ui.tournament_service.append_jsonl"),
            patch("asciibench.judge_ui.tournament_service.calculate_elo") as mock_elo,
            patch.object(service, "_complete_round", new_callable=AsyncMock) as mock_complete,
        ):
            mock_elo.return_value = {"model-a": 1500.0, "model-b": 1500.0}
            await service.record_vote(matchup_id, vote_id)

        assert service._current_round.all_judged is True
        mock_complete.assert_called_once()


class TestUndoLastVote:
    """Tests for undo_last_vote method."""

    @pytest.fixture
    def service(self, tmp_path: Path) -> TournamentService:
        """Create a TournamentService instance for tests."""
        mock_generation_service = MagicMock(spec=GenerationService)
        mock_config_service = MagicMock()
        mock_config_service.get_models.return_value = []
        mock_config_service.get_prompts.return_value = []
        mock_repo = MagicMock()
        mock_repo.get_all_samples.return_value = []
        mock_repo.get_votes.return_value = []

        service = TournamentService(
            generation_service=mock_generation_service,
            config_service=mock_config_service,
            repo=mock_repo,
            n=1,
        )
        service._rounds_path = tmp_path / "rounds.jsonl"
        return service

    @pytest.mark.asyncio
    async def test_undo_last_vote_unmarks_matchup_and_clears_vote_id(
        self, service: TournamentService, tmp_path: Path
    ) -> None:
        """Test undo_last_vote() unmarks matchup and clears vote_id."""
        matchup_id = uuid4()
        vote_id = str(uuid4())
        matchup = Matchup(
            id=matchup_id,
            model_a_id="model-a",
            model_b_id="model-b",
            prompt_text="Draw a cat",
            prompt_category="animal",
            is_judged=True,
            vote_id=vote_id,
        )

        service._current_round = RoundState(
            id=uuid4(),
            round_number=1,
            matchups=[matchup],
        )

        with patch("asciibench.judge_ui.tournament_service.append_jsonl") as mock_append:
            await service.undo_last_vote(matchup_id)

        assert service._current_round.matchups[0].is_judged is False
        assert service._current_round.matchups[0].vote_id is None
        mock_append.assert_called_once()


class TestGetRoundProgress:
    """Tests for get_round_progress method."""

    @pytest.fixture
    def service(self, tmp_path: Path) -> TournamentService:
        """Create a TournamentService instance for tests."""
        mock_generation_service = MagicMock(spec=GenerationService)
        mock_config_service = MagicMock()
        mock_config_service.get_models.return_value = []
        mock_config_service.get_prompts.return_value = []
        mock_repo = MagicMock()
        mock_repo.get_all_samples.return_value = []
        mock_repo.get_votes.return_value = []

        service = TournamentService(
            generation_service=mock_generation_service,
            config_service=mock_config_service,
            repo=mock_repo,
            n=1,
        )
        service._rounds_path = tmp_path / "rounds.jsonl"
        return service

    def test_get_round_progress_returns_correct_counts(self, service: TournamentService) -> None:
        """Test get_round_progress() returns correct counts."""
        judged_matchup = Matchup(
            id=uuid4(),
            model_a_id="model-a",
            model_b_id="model-b",
            prompt_text="Draw a cat",
            prompt_category="animal",
            is_judged=True,
        )
        unjudged_matchup = Matchup(
            id=uuid4(),
            model_a_id="model-a",
            model_b_id="model-c",
            prompt_text="Draw a dog",
            prompt_category="animal",
            is_judged=False,
        )

        service._current_round = RoundState(
            id=uuid4(),
            round_number=1,
            matchups=[judged_matchup, unjudged_matchup],
        )

        result = service.get_round_progress()

        assert result["round_number"] == 1
        assert result["judged_count"] == 1
        assert result["total_count"] == 2
        assert result["next_round_ready"] is False

    def test_get_round_progress_with_next_round_ready(self, service: TournamentService) -> None:
        """Test get_round_progress() with next round ready."""
        judged_matchup = Matchup(
            id=uuid4(),
            model_a_id="model-a",
            model_b_id="model-b",
            prompt_text="Draw a cat",
            prompt_category="animal",
            is_judged=True,
        )

        service._current_round = RoundState(
            id=uuid4(),
            round_number=1,
            matchups=[judged_matchup],
        )

        service._next_round = RoundState(
            id=uuid4(),
            round_number=2,
            matchups=[],
            generation_complete=True,
        )

        result = service.get_round_progress()

        assert result["next_round_ready"] is True

    def test_get_round_progress_no_current_round(self, service: TournamentService) -> None:
        """Test get_round_progress() with no current round."""
        service._current_round = None

        result = service.get_round_progress()

        assert result["round_number"] == 0
        assert result["judged_count"] == 0
        assert result["total_count"] == 0
        assert result["next_round_ready"] is False


class TestCompleteRound:
    """Tests for _complete_round method."""

    @pytest.fixture
    def service(self, tmp_path: Path) -> TournamentService:
        """Create a TournamentService instance for tests."""
        mock_generation_service = MagicMock(spec=GenerationService)
        mock_config_service = MagicMock()
        mock_config_service.get_models.return_value = []
        mock_config_service.get_prompts.return_value = []
        mock_repo = MagicMock()
        mock_repo.get_all_samples.return_value = []

        vote = Vote(
            id=uuid4(),
            sample_a_id=str(uuid4()),
            sample_b_id=str(uuid4()),
            winner="A",
        )
        mock_repo.get_votes.return_value = [vote]

        service = TournamentService(
            generation_service=mock_generation_service,
            config_service=mock_config_service,
            repo=mock_repo,
            n=1,
        )
        service._rounds_path = tmp_path / "rounds.jsonl"
        return service

    @pytest.mark.asyncio
    async def test_complete_round_recomputes_elo_using_calculate_elo(
        self, service: TournamentService, tmp_path: Path
    ) -> None:
        """Test round completion recomputes Elo using analyst calculate_elo()."""
        matchup = Matchup(
            id=uuid4(),
            model_a_id="model-a",
            model_b_id="model-b",
            prompt_text="Draw a cat",
            prompt_category="animal",
            is_judged=True,
        )

        service._current_round = RoundState(
            id=uuid4(),
            round_number=1,
            matchups=[matchup],
        )

        with (
            patch("asciibench.judge_ui.tournament_service.calculate_elo") as mock_elo,
            patch("asciibench.judge_ui.tournament_service.append_jsonl"),
            patch.object(service, "_create_round", new_callable=AsyncMock) as mock_create,
            patch.object(
                service.generation_service, "ensure_samples_for_round", new_callable=AsyncMock
            ) as mock_ensure,
        ):
            mock_elo.return_value = {"model-a": 1510.0, "model-b": 1490.0}
            mock_create.return_value = RoundState(
                id=uuid4(),
                round_number=2,
                matchups=[],
            )
            mock_ensure.return_value = RoundState(
                id=uuid4(),
                round_number=2,
                matchups=[],
                generation_complete=True,
            )
            await service._complete_round()

        mock_elo.assert_called_once()


class TestGetGenerationStatus:
    """Tests for get_generation_status method."""

    @pytest.fixture
    def service(self, tmp_path: Path) -> TournamentService:
        """Create a TournamentService instance for tests."""
        mock_generation_service = MagicMock(spec=GenerationService)
        mock_config_service = MagicMock()
        mock_config_service.get_models.return_value = []
        mock_config_service.get_prompts.return_value = []
        mock_repo = MagicMock()
        mock_repo.get_all_samples.return_value = []
        mock_repo.get_votes.return_value = []

        service = TournamentService(
            generation_service=mock_generation_service,
            config_service=mock_config_service,
            repo=mock_repo,
            n=1,
        )
        service._rounds_path = tmp_path / "rounds.jsonl"
        return service

    def test_get_generation_status_defaults_to_not_generating(
        self, service: TournamentService
    ) -> None:
        """Test get_generation_status() returns defaults when no generation."""
        result = service.get_generation_status()

        assert result["generating"] is False
        assert result["completed"] == 0
        assert result["total"] == 0

    def test_get_generation_status_with_generation_complete(
        self, service: TournamentService
    ) -> None:
        """Test get_generation_status() returns completed state after generation."""
        service._generation_total = 10
        service._generation_completed = 10
        service._initial_generation_task = None

        result = service.get_generation_status()

        assert result["generating"] is False
        assert result["completed"] == 10
        assert result["total"] == 10

    def test_get_generation_status_during_generation(self, service: TournamentService) -> None:
        """Test get_generation_status() returns generating true during generation."""
        service._generation_total = 10
        service._generation_completed = 3

        mock_task = MagicMock()
        mock_task.done.return_value = False
        service._initial_generation_task = mock_task

        result = service.get_generation_status()

        assert result["generating"] is True
        assert result["completed"] == 3
        assert result["total"] == 10

    def test_get_generation_status_with_partial_completion(
        self, service: TournamentService
    ) -> None:
        """Test get_generation_status() returns partial progress."""
        service._generation_total = 5
        service._generation_completed = 2
        service._initial_generation_task = None

        result = service.get_generation_status()

        assert result["generating"] is False
        assert result["completed"] == 2
        assert result["total"] == 5
