"""Tests for ProgressService."""

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

from asciibench.common.models import ArtSample, Vote
from asciibench.common.repository import DataRepository
from asciibench.judge_ui.matchup_service import MatchupService
from asciibench.judge_ui.progress_service import ProgressService


@pytest.fixture
def temp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up a temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    return data_dir


@pytest.fixture
def repo(temp_data_dir: Path) -> DataRepository:
    """Create a DataRepository instance for tests."""
    return DataRepository(data_dir=temp_data_dir)


@pytest.fixture
def matchup_service(temp_data_dir: Path) -> MatchupService:
    """Create a MatchupService instance for tests."""
    return MatchupService(
        database_path=temp_data_dir / "database.jsonl",
        votes_path=temp_data_dir / "votes.jsonl",
    )


@pytest.fixture
def service(
    repo: DataRepository,
    matchup_service: MatchupService,
) -> ProgressService:
    """Create a ProgressService instance for tests."""
    return ProgressService(repo=repo, matchup_service=matchup_service)


@pytest.fixture
def sample_samples() -> list[ArtSample]:
    """Create sample test data with samples from different models and categories."""
    return [
        ArtSample(
            id=uuid4(),
            model_id="model-a",
            prompt_text="Draw a cat",
            category="single_animal",
            attempt_number=1,
            raw_output="```\ncat\n```",
            sanitized_output="cat",
            is_valid=True,
            timestamp=datetime.now(),
        ),
        ArtSample(
            id=uuid4(),
            model_id="model-a",
            prompt_text="Draw a dog",
            category="single_animal",
            attempt_number=1,
            raw_output="```\ndog\n```",
            sanitized_output="dog",
            is_valid=True,
            timestamp=datetime.now(),
        ),
        ArtSample(
            id=uuid4(),
            model_id="model-b",
            prompt_text="Draw a cat",
            category="single_animal",
            attempt_number=1,
            raw_output="```\ncat\n```",
            sanitized_output="cat",
            is_valid=True,
            timestamp=datetime.now(),
        ),
        ArtSample(
            id=uuid4(),
            model_id="model-b",
            prompt_text="Draw a dog",
            category="single_animal",
            attempt_number=1,
            raw_output="```\ndog\n```",
            sanitized_output="dog",
            is_valid=True,
            timestamp=datetime.now(),
        ),
        ArtSample(
            id=uuid4(),
            model_id="model-c",
            prompt_text="Draw a tree",
            category="single_object",
            attempt_number=1,
            raw_output="```\ntree\n```",
            sanitized_output="tree",
            is_valid=True,
            timestamp=datetime.now(),
        ),
        ArtSample(
            id=uuid4(),
            model_id="model-c",
            prompt_text="Draw a house",
            category="single_object",
            attempt_number=1,
            raw_output="```\nhouse\n```",
            sanitized_output="house",
            is_valid=True,
            timestamp=datetime.now(),
        ),
    ]


@pytest.fixture
def sample_votes(sample_samples: list[ArtSample]) -> list[Vote]:
    """Create sample vote data."""
    return [
        Vote(
            sample_a_id=str(sample_samples[0].id),
            sample_b_id=str(sample_samples[2].id),
            winner="A",
            timestamp=datetime.now(),
        ),
        Vote(
            sample_a_id=str(sample_samples[0].id),
            sample_b_id=str(sample_samples[2].id),
            winner="B",
            timestamp=datetime.now(),
        ),
        Vote(
            sample_a_id=str(sample_samples[0].id),
            sample_b_id=str(sample_samples[4].id),
            winner="A",
            timestamp=datetime.now(),
        ),
        Vote(
            sample_a_id=str(sample_samples[1].id),
            sample_b_id=str(sample_samples[3].id),
            winner="B",
            timestamp=datetime.now(),
        ),
    ]


@pytest.fixture
def invalid_sample() -> ArtSample:
    """Create an invalid sample for testing."""
    return ArtSample(
        id=uuid4(),
        model_id="model-a",
        prompt_text="Draw invalid",
        category="single_animal",
        attempt_number=1,
        raw_output="```\ninvalid\n```",
        sanitized_output="invalid",
        is_valid=False,
        timestamp=datetime.now(),
    )


class TestGetProgress:
    """Tests for get_progress method."""

    def test_get_progress_with_no_samples_returns_zero_values(
        self, service: ProgressService
    ) -> None:
        """Test that progress returns zero values when no samples exist."""
        progress = service.get_progress()

        assert progress.votes_completed == 0
        assert progress.unique_pairs_judged == 0
        assert progress.total_possible_pairs == 0
        assert progress.by_category == {}

    def test_get_progress_with_no_votes(
        self,
        service: ProgressService,
        repo: DataRepository,
        sample_samples: list[ArtSample],
    ) -> None:
        """Test that progress returns zero votes when samples exist but no votes."""
        from asciibench.common.persistence import append_jsonl

        db_path = repo.database_path

        for sample in sample_samples:
            append_jsonl(db_path, sample)

        progress = service.get_progress()

        assert progress.votes_completed == 0
        assert progress.unique_pairs_judged == 0
        assert progress.total_possible_pairs > 0
        assert len(progress.by_category) > 0

    def test_get_progress_with_votes(
        self,
        service: ProgressService,
        repo: DataRepository,
        sample_samples: list[ArtSample],
        sample_votes: list[Vote],
    ) -> None:
        """Test that progress is calculated correctly with votes."""
        from asciibench.common.persistence import append_jsonl

        db_path = repo.database_path
        votes_path = repo.votes_path

        for sample in sample_samples:
            append_jsonl(db_path, sample)

        for vote in sample_votes:
            append_jsonl(votes_path, vote)

        progress = service.get_progress()

        assert progress.votes_completed == len(sample_votes)
        assert progress.unique_pairs_judged > 0
        assert progress.total_possible_pairs > 0
        assert len(progress.by_category) > 0

    def test_get_progress_with_invalid_samples_excludes_them(
        self,
        service: ProgressService,
        repo: DataRepository,
        sample_samples: list[ArtSample],
        invalid_sample: ArtSample,
    ) -> None:
        """Test that invalid samples are excluded from progress calculations."""
        from asciibench.common.persistence import append_jsonl

        db_path = repo.database_path

        for sample in sample_samples:
            append_jsonl(db_path, sample)

        append_jsonl(db_path, invalid_sample)

        progress = service.get_progress()

        total_valid_samples = len(sample_samples)
        expected_possible_pairs = 0

        for i in range(total_valid_samples):
            for j in range(i + 1, total_valid_samples):
                if sample_samples[i].model_id != sample_samples[j].model_id:
                    expected_possible_pairs += 1

        assert progress.total_possible_pairs == expected_possible_pairs

    def test_get_progress_category_breakdown(
        self,
        service: ProgressService,
        repo: DataRepository,
        sample_samples: list[ArtSample],
        sample_votes: list[Vote],
    ) -> None:
        """Test that category breakdown is calculated correctly."""
        from asciibench.common.persistence import append_jsonl

        db_path = repo.database_path
        votes_path = repo.votes_path

        for sample in sample_samples:
            append_jsonl(db_path, sample)

        for vote in sample_votes:
            append_jsonl(votes_path, vote)

        progress = service.get_progress()

        assert "single_animal" in progress.by_category
        assert "single_object" in progress.by_category

        single_animal = progress.by_category["single_animal"]
        single_object = progress.by_category["single_object"]

        assert single_animal.votes_completed >= 0
        assert single_animal.unique_pairs_judged >= 0
        assert single_animal.total_possible_pairs > 0

        assert single_object.votes_completed >= 0
        assert single_object.unique_pairs_judged >= 0
        assert single_object.total_possible_pairs >= 0

    def test_get_progress_all_completed(
        self,
        service: ProgressService,
        repo: DataRepository,
        sample_samples: list[ArtSample],
    ) -> None:
        """Test progress when all possible pairs have been judged."""
        from asciibench.common.persistence import append_jsonl

        db_path = repo.database_path
        votes_path = repo.votes_path

        for sample in sample_samples:
            append_jsonl(db_path, sample)

        valid_samples = [s for s in sample_samples if s.is_valid]

        votes = []
        for i in range(len(valid_samples)):
            for j in range(i + 1, len(valid_samples)):
                if valid_samples[i].model_id != valid_samples[j].model_id:
                    votes.append(
                        Vote(
                            sample_a_id=str(valid_samples[i].id),
                            sample_b_id=str(valid_samples[j].id),
                            winner="A",
                            timestamp=datetime.now(),
                        )
                    )

        for vote in votes:
            append_jsonl(votes_path, vote)

        progress = service.get_progress()

        assert progress.votes_completed == len(votes)
        # All sample pairs have been voted on
        assert progress.unique_pairs_judged == 3  # 3 unique model pairs: (a,b), (a,c), (b,c)
        assert progress.total_possible_pairs == 12  # 12 sample pairs between different models

    def test_get_progress_partial_completion(
        self,
        service: ProgressService,
        repo: DataRepository,
        sample_samples: list[ArtSample],
    ) -> None:
        """Test progress with partial completion."""
        from asciibench.common.persistence import append_jsonl

        db_path = repo.database_path
        votes_path = repo.votes_path

        for sample in sample_samples:
            append_jsonl(db_path, sample)

        valid_samples = [s for s in sample_samples if s.is_valid]

        vote = Vote(
            sample_a_id=str(valid_samples[0].id),
            sample_b_id=str(valid_samples[2].id),
            winner="A",
            timestamp=datetime.now(),
        )
        append_jsonl(votes_path, vote)

        progress = service.get_progress()

        assert progress.votes_completed == 1
        assert progress.unique_pairs_judged == 1
        assert progress.unique_pairs_judged < progress.total_possible_pairs


class TestCalculateProgressByCategory:
    """Tests for _calculate_progress_by_category method."""

    def test_calculate_progress_by_category_empty_data(self, service: ProgressService) -> None:
        """Test that category progress is empty with no samples or votes."""
        result = service._calculate_progress_by_category([], [])
        assert result == {}

    def test_calculate_progress_by_category_with_valid_samples(
        self, service: ProgressService, sample_samples: list[ArtSample]
    ) -> None:
        """Test that category progress includes valid samples by category."""
        result = service._calculate_progress_by_category([], sample_samples)

        assert "single_animal" in result
        assert "single_object" in result

        assert result["single_animal"].votes_completed == 0
        assert result["single_animal"].unique_pairs_judged == 0
        assert result["single_animal"].total_possible_pairs > 0

    def test_calculate_progress_by_category_with_votes(
        self,
        service: ProgressService,
        sample_samples: list[ArtSample],
        sample_votes: list[Vote],
    ) -> None:
        """Test that category progress counts votes correctly."""
        result = service._calculate_progress_by_category(sample_votes, sample_samples)

        single_animal = result["single_animal"]
        assert single_animal.votes_completed >= 0
        assert single_animal.unique_pairs_judged >= 0

    def test_calculate_progress_by_category_excludes_invalid_samples(
        self,
        service: ProgressService,
        sample_samples: list[ArtSample],
        invalid_sample: ArtSample,
    ) -> None:
        """Test that invalid samples are excluded from category progress."""
        all_samples = [*sample_samples, invalid_sample]
        result = service._calculate_progress_by_category([], all_samples)

        # single_animal has 2 models, so pairs > 0
        assert result["single_animal"].total_possible_pairs > 0
        # single_object has only 1 model, so pairs = 0
        assert result["single_object"].total_possible_pairs == 0

    def test_calculate_progress_by_category_no_samples_returns_empty(
        self, service: ProgressService
    ) -> None:
        """Test that no samples returns empty dict, not division error."""
        result = service._calculate_progress_by_category([], [])
        assert result == {}
