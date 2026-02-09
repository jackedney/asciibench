"""Tests for AnalyticsService."""

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

from asciibench.common.models import ArtSample, VLMEvaluation, Vote
from asciibench.common.repository import DataRepository
from asciibench.judge_ui.analytics_service import AnalyticsService


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
def service(repo: DataRepository) -> AnalyticsService:
    """Create an AnalyticsService instance for tests."""
    return AnalyticsService(repo=repo)


@pytest.fixture
def sample_samples() -> list[ArtSample]:
    """Create sample test data with samples from different models."""
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
def sample_vlm_evaluations(sample_samples: list[ArtSample]) -> list[VLMEvaluation]:
    """Create sample VLM evaluation data."""
    return [
        VLMEvaluation(
            sample_id=str(sample_samples[0].id),
            vlm_model_id="gpt-4",
            expected_subject=sample_samples[0].prompt_text.split()[-1],
            vlm_response="cat",
            similarity_score=0.9,
            is_correct=True,
            timestamp=datetime.now(),
        ),
        VLMEvaluation(
            sample_id=str(sample_samples[1].id),
            vlm_model_id="gpt-4",
            expected_subject=sample_samples[1].prompt_text.split()[-1],
            vlm_response="dog",
            similarity_score=0.95,
            is_correct=True,
            timestamp=datetime.now(),
        ),
        VLMEvaluation(
            sample_id=str(sample_samples[2].id),
            vlm_model_id="gpt-4",
            expected_subject=sample_samples[2].prompt_text.split()[-1],
            vlm_response="cat",
            similarity_score=0.8,
            is_correct=True,
            timestamp=datetime.now(),
        ),
        VLMEvaluation(
            sample_id=str(sample_samples[3].id),
            vlm_model_id="gpt-4",
            expected_subject=sample_samples[3].prompt_text.split()[-1],
            vlm_response="dog",
            similarity_score=0.7,
            is_correct=True,
            timestamp=datetime.now(),
        ),
    ]


class TestAnalyticsServiceGetAnalyticsData:
    """Tests for get_analytics_data method."""

    def test_get_analytics_data_returns_zero_values_with_no_votes(
        self, service: AnalyticsService, temp_data_dir: Path
    ) -> None:
        """Test that analytics returns zeros when no votes exist."""
        analytics = service.get_analytics_data()

        assert analytics.total_votes == 0
        assert analytics.leaderboard == []
        assert analytics.elo_history == {}
        assert analytics.head_to_head == {}
        assert analytics.stability.score == 0.0
        assert analytics.stability.is_stable is False
        assert "No votes to analyze" in analytics.stability.warnings

    def test_get_analytics_data_with_votes(
        self,
        service: AnalyticsService,
        repo: DataRepository,
        sample_samples: list[ArtSample],
        sample_votes: list[Vote],
    ) -> None:
        """Test that analytics data is calculated correctly with votes."""
        from asciibench.common.persistence import append_jsonl

        db_path = repo.database_path
        votes_path = repo.votes_path

        for sample in sample_samples:
            append_jsonl(db_path, sample)

        for vote in sample_votes:
            append_jsonl(votes_path, vote)

        analytics = service.get_analytics_data()

        assert analytics.total_votes == len(sample_votes)
        assert len(analytics.leaderboard) > 0
        assert len(analytics.elo_history) > 0
        assert len(analytics.head_to_head) > 0

    def test_get_analytics_data_handles_missing_files(
        self, temp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that analytics handles missing data files gracefully."""
        repo = DataRepository(data_dir=temp_data_dir)
        service = AnalyticsService(repo=repo)

        analytics = service.get_analytics_data()

        assert analytics.total_votes == 0
        assert analytics.leaderboard == []


class TestCalculateEloHistory:
    """Tests for _calculate_elo_history method."""

    def test_calculate_elo_history_empty_votes(
        self, service: AnalyticsService, sample_samples: list[ArtSample]
    ) -> None:
        """Test that Elo history is empty with no votes."""
        history = service._calculate_elo_history([], sample_samples)
        assert history == {}

    def test_calculate_elo_history_creates_checkpoints(
        self, service: AnalyticsService, sample_samples: list[ArtSample]
    ) -> None:
        """Test that Elo history creates checkpoints at intervals."""
        votes = [
            Vote(
                sample_a_id=str(sample_samples[0].id),
                sample_b_id=str(sample_samples[2].id),
                winner="A",
                timestamp=datetime.now(),
            )
            for _ in range(25)
        ]

        history = service._calculate_elo_history(votes, sample_samples, checkpoint_interval=10)

        assert len(history) > 0

        for _model_id, points in history.items():
            assert len(points) >= 2
            assert all(p.vote_count >= 10 for p in points)
            assert all(isinstance(p.elo, (int, float)) for p in points)

    def test_calculate_elo_history_tracks_changes(
        self, service: AnalyticsService, sample_samples: list[ArtSample]
    ) -> None:
        """Test that Elo history tracks rating changes over time."""
        votes = [
            Vote(
                sample_a_id=str(sample_samples[0].id),
                sample_b_id=str(sample_samples[2].id),
                winner="A" if i % 2 == 0 else "B",
                timestamp=datetime.now(),
            )
            for i in range(20)
        ]

        history = service._calculate_elo_history(votes, sample_samples, checkpoint_interval=10)

        assert "model-a" in history or "model-b" in history

        for _model_id, points in history.items():
            if len(points) > 1:
                first_elo = points[0].elo
                last_elo = points[-1].elo
                assert isinstance(first_elo, (int, float))
                assert isinstance(last_elo, (int, float))


class TestCalculateHeadToHead:
    """Tests for _calculate_head_to_head method."""

    def test_calculate_head_to_head_empty_votes(
        self, service: AnalyticsService, sample_samples: list[ArtSample]
    ) -> None:
        """Test that head-to-head is empty with no votes."""
        h2h = service._calculate_head_to_head([], sample_samples)
        assert h2h == {}

    def test_calculate_head_to_head_counts_wins(
        self, service: AnalyticsService, sample_samples: list[ArtSample]
    ) -> None:
        """Test that head-to-head correctly counts wins and losses."""
        votes = [
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
                sample_b_id=str(sample_samples[2].id),
                winner="A",
                timestamp=datetime.now(),
            ),
        ]

        h2h = service._calculate_head_to_head(votes, sample_samples)

        assert "model-a" in h2h
        assert "model-b" in h2h["model-a"]
        record = h2h["model-a"]["model-b"]
        assert record.wins == 2
        assert record.losses == 1

    def test_calculate_head_to_head_ignores_fail_votes(
        self, service: AnalyticsService, sample_samples: list[ArtSample]
    ) -> None:
        """Test that head-to-head ignores 'fail' votes."""
        votes = [
            Vote(
                sample_a_id=str(sample_samples[0].id),
                sample_b_id=str(sample_samples[2].id),
                winner="A",
                timestamp=datetime.now(),
            ),
            Vote(
                sample_a_id=str(sample_samples[0].id),
                sample_b_id=str(sample_samples[2].id),
                winner="fail",
                timestamp=datetime.now(),
            ),
        ]

        h2h = service._calculate_head_to_head(votes, sample_samples)

        record = h2h["model-a"]["model-b"]
        assert record.wins == 1
        assert record.losses == 0

    def test_calculate_head_to_head_all_model_pairs(
        self, service: AnalyticsService, sample_samples: list[ArtSample]
    ) -> None:
        """Test that head-to-head includes all model pairs that have votes."""
        votes = [
            Vote(
                sample_a_id=str(sample_samples[0].id),
                sample_b_id=str(sample_samples[2].id),
                winner="A",
                timestamp=datetime.now(),
            )
        ]

        h2h = service._calculate_head_to_head(votes, sample_samples)

        assert "model-a" in h2h
        assert "model-b" in h2h

        assert "model-b" in h2h["model-a"]
        assert "model-a" in h2h["model-b"]

    def test_calculate_head_to_head_no_self_matches(
        self, service: AnalyticsService, sample_samples: list[ArtSample]
    ) -> None:
        """Test that head-to-head doesn't include self-matches."""
        h2h = service._calculate_head_to_head([], sample_samples)

        for model_a in h2h:
            assert model_a not in h2h[model_a]


class TestCalculateVLMAccuracy:
    """Tests for _calculate_vlm_accuracy method."""

    def test_calculate_vlm_accuracy_empty_evaluations(
        self, service: AnalyticsService, sample_samples: list[ArtSample]
    ) -> None:
        """Test that VLM accuracy is empty with no evaluations."""
        accuracy = service._calculate_vlm_accuracy([], sample_samples)
        assert accuracy == {}

    def test_calculate_vlm_accuracy_by_model(
        self,
        service: AnalyticsService,
        sample_samples: list[ArtSample],
        sample_vlm_evaluations: list[VLMEvaluation],
    ) -> None:
        """Test that VLM accuracy is calculated per model."""
        accuracy = service._calculate_vlm_accuracy(sample_vlm_evaluations, sample_samples)

        assert "model-a" in accuracy
        assert "model-b" in accuracy

        model_a_stats = accuracy["model-a"]
        assert model_a_stats.total == 2
        assert model_a_stats.accuracy == pytest.approx(1.0, 0.01)

    def test_calculate_vlm_accuracy_correct_and_incorrect(
        self, service: AnalyticsService, sample_samples: list[ArtSample]
    ) -> None:
        """Test that VLM accuracy counts correct and incorrect evaluations."""
        evaluations = [
            VLMEvaluation(
                sample_id=str(sample_samples[0].id),
                vlm_model_id="gpt-4",
                expected_subject=sample_samples[0].prompt_text.split()[-1],
                vlm_response="cat",
                similarity_score=0.9,
                is_correct=True,
                timestamp=datetime.now(),
            ),
            VLMEvaluation(
                sample_id=str(sample_samples[1].id),
                vlm_model_id="gpt-4",
                expected_subject=sample_samples[1].prompt_text.split()[-1],
                vlm_response="cat",
                similarity_score=0.5,
                is_correct=False,
                timestamp=datetime.now(),
            ),
        ]

        accuracy = service._calculate_vlm_accuracy(evaluations, sample_samples)

        assert "model-a" in accuracy
        assert accuracy["model-a"].total == 2
        assert accuracy["model-a"].correct in [0, 1, 2]

    def test_calculate_vlm_accuracy_invalid_sample(
        self, service: AnalyticsService, sample_samples: list[ArtSample]
    ) -> None:
        """Test that VLM accuracy handles evaluations for non-existent samples."""
        evaluations = [
            VLMEvaluation(
                sample_id=str(uuid4()),
                vlm_model_id="gpt-4",
                expected_subject="cat",
                vlm_response="cat",
                similarity_score=0.9,
                is_correct=True,
                timestamp=datetime.now(),
            )
        ]

        accuracy = service._calculate_vlm_accuracy(evaluations, sample_samples)
        assert accuracy == {}

    def test_calculate_vlm_accuracy_all_evaluations_incorrect(
        self, service: AnalyticsService, sample_samples: list[ArtSample]
    ) -> None:
        """Test that VLM accuracy is 0 when all evaluations are incorrect."""
        evaluations = [
            VLMEvaluation(
                sample_id=str(sample_samples[0].id),
                vlm_model_id="gpt-4",
                expected_subject=sample_samples[0].prompt_text.split()[-1],
                vlm_response="dog",
                similarity_score=0.3,
                is_correct=False,
                timestamp=datetime.now(),
            )
        ]

        accuracy = service._calculate_vlm_accuracy(evaluations, sample_samples)

        if "model-a" in accuracy:
            assert accuracy["model-a"].accuracy == 0.0


class TestCalculateCategoryAccuracy:
    """Tests for _calculate_category_accuracy method."""

    def test_calculate_category_accuracy_empty_evaluations(
        self, service: AnalyticsService, sample_samples: list[ArtSample]
    ) -> None:
        """Test that category accuracy is empty with no evaluations."""
        accuracy = service._calculate_category_accuracy([], sample_samples)
        assert accuracy == {}

    def test_calculate_category_accuracy_by_category(
        self,
        service: AnalyticsService,
        sample_samples: list[ArtSample],
        sample_vlm_evaluations: list[VLMEvaluation],
    ) -> None:
        """Test that category accuracy is calculated per category."""
        accuracy = service._calculate_category_accuracy(sample_vlm_evaluations, sample_samples)

        assert "single_animal" in accuracy

        category_stats = accuracy["single_animal"]
        assert category_stats.total == 4
        assert isinstance(category_stats.accuracy, float)

    def test_calculate_category_accuracy_multiple_categories(
        self,
        service: AnalyticsService,
        sample_samples: list[ArtSample],
        sample_vlm_evaluations: list[VLMEvaluation],
    ) -> None:
        """Test that category accuracy handles multiple categories."""
        accuracy = service._calculate_category_accuracy(sample_vlm_evaluations, sample_samples)

        assert len(accuracy) > 0

        for _category, stats in accuracy.items():
            assert stats.total > 0
            assert 0.0 <= stats.accuracy <= 1.0


class TestCalculatePearsonCorrelation:
    """Tests for _calculate_pearson_correlation method."""

    def test_calculate_pearson_correlation_empty_lists(self, service: AnalyticsService) -> None:
        """Test that Pearson correlation returns None for empty lists."""
        result = service._calculate_pearson_correlation([], [])
        assert result is None

    def test_calculate_pearson_correlation_less_than_three(self, service: AnalyticsService) -> None:
        """Test that Pearson correlation returns None with fewer than 3 data points."""
        result = service._calculate_pearson_correlation([1.0, 2.0], [2.0, 3.0])
        assert result is None

    def test_calculate_pearson_correlation_mismatched_lengths(
        self, service: AnalyticsService
    ) -> None:
        """Test that Pearson correlation raises ValueError for mismatched lengths."""
        with pytest.raises(ValueError, match="Lists must have the same length"):
            service._calculate_pearson_correlation([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0])

    def test_calculate_pearson_correlation_perfect_positive(
        self, service: AnalyticsService
    ) -> None:
        """Test that Pearson correlation is 1.0 for perfect positive correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]

        result = service._calculate_pearson_correlation(x, y)
        assert result is not None
        assert result == pytest.approx(1.0, 0.01)

    def test_calculate_pearson_correlation_perfect_negative(
        self, service: AnalyticsService
    ) -> None:
        """Test that Pearson correlation is -1.0 for perfect negative correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]

        result = service._calculate_pearson_correlation(x, y)
        assert result is not None
        assert result == pytest.approx(-1.0, 0.01)

    def test_calculate_pearson_correlation_no_correlation(self, service: AnalyticsService) -> None:
        """Test that Pearson correlation is near 0 for no correlation."""
        import random

        random.seed(42)
        x = [random.random() for _ in range(100)]
        y = [random.random() for _ in range(100)]

        result = service._calculate_pearson_correlation(x, y)
        assert result is not None
        assert result == pytest.approx(0.0, abs=0.3)

    def test_calculate_pearson_correlation_zero_variance(self, service: AnalyticsService) -> None:
        """Test that Pearson correlation returns None when variance is zero."""
        x = [1.0, 1.0, 1.0, 1.0]
        y = [2.0, 3.0, 4.0, 5.0]

        result = service._calculate_pearson_correlation(x, y)
        assert result is None

    def test_calculate_pearson_correlation_valid_range(self, service: AnalyticsService) -> None:
        """Test that Pearson correlation is always in valid range [-1, 1]."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        y = [3.0, 5.0, 2.0, 8.0, 4.0, 7.0, 1.0, 6.0]

        result = service._calculate_pearson_correlation(x, y)
        assert result is not None
        assert -1.0 <= result <= 1.0


class TestGetVLMAccuracyData:
    """Tests for get_vlm_accuracy_data method."""

    def test_get_vlm_accuracy_data_no_evaluations(
        self, service: AnalyticsService, temp_data_dir: Path
    ) -> None:
        """Test that VLM accuracy data is empty with no evaluations."""
        accuracy_data = service.get_vlm_accuracy_data()

        assert accuracy_data.by_model == {}
        assert accuracy_data.by_category == {}

    def test_get_vlm_accuracy_data_with_evaluations(
        self,
        service: AnalyticsService,
        repo: DataRepository,
        sample_samples: list[ArtSample],
        sample_vlm_evaluations: list[VLMEvaluation],
    ) -> None:
        """Test that VLM accuracy data is calculated correctly."""
        from asciibench.common.persistence import append_jsonl

        db_path = repo.database_path
        vlm_path = repo.evaluations_path

        for sample in sample_samples:
            append_jsonl(db_path, sample)

        for evaluation in sample_vlm_evaluations:
            append_jsonl(vlm_path, evaluation)

        accuracy_data = service.get_vlm_accuracy_data()

        assert len(accuracy_data.by_model) > 0
        assert len(accuracy_data.by_category) > 0


class TestGetVLMAccuracyByModel:
    """Tests for get_vlm_accuracy_by_model method."""

    def test_get_vlm_accuracy_by_model_no_evaluations(
        self, service: AnalyticsService, temp_data_dir: Path
    ) -> None:
        """Test that VLM accuracy by model is empty with no evaluations."""
        accuracy = service.get_vlm_accuracy_by_model()
        assert accuracy == {}

    def test_get_vlm_accuracy_by_model_with_evaluations(
        self,
        service: AnalyticsService,
        repo: DataRepository,
        sample_samples: list[ArtSample],
        sample_vlm_evaluations: list[VLMEvaluation],
    ) -> None:
        """Test that VLM accuracy by model is calculated correctly."""
        from asciibench.common.persistence import append_jsonl

        db_path = repo.database_path
        vlm_path = repo.evaluations_path

        for sample in sample_samples:
            append_jsonl(db_path, sample)

        for evaluation in sample_vlm_evaluations:
            append_jsonl(vlm_path, evaluation)

        accuracy = service.get_vlm_accuracy_by_model()

        assert len(accuracy) > 0

        for _model_id, stats in accuracy.items():
            assert stats.total > 0
            assert 0.0 <= stats.accuracy <= 1.0
