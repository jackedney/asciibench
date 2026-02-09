"""Tests for RankingStabilityAnalyzer class."""

from uuid import uuid4

import pytest

from asciibench.analyst.errors import InsufficientDataError
from asciibench.analyst.ranking_stability_analyzer import RankingStabilityAnalyzer
from asciibench.common.models import ArtSample, Vote


@pytest.fixture
def two_model_samples() -> list[ArtSample]:
    """Create two model samples for testing."""
    sample_a = ArtSample(
        id=uuid4(),
        model_id="model1",
        prompt_text="test",
        category="test",
        attempt_number=1,
        raw_output="test",
        sanitized_output="test",
        is_valid=True,
    )
    sample_b = ArtSample(
        id=uuid4(),
        model_id="model2",
        prompt_text="test",
        category="test",
        attempt_number=1,
        raw_output="test",
        sanitized_output="test",
        is_valid=True,
    )
    return [sample_a, sample_b]


@pytest.fixture
def three_model_samples() -> list[ArtSample]:
    """Create three model samples for testing."""
    return [
        ArtSample(
            id=uuid4(),
            model_id=f"model{i}",
            prompt_text="test",
            category="test",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        for i in range(1, 4)
    ]


@pytest.fixture
def balanced_votes(two_model_samples: list[ArtSample]) -> list[Vote]:
    """Create 50 balanced votes (25 wins each)."""
    from datetime import datetime, timedelta

    samples = two_model_samples
    base_time = datetime.now()
    votes = []
    for i in range(50):
        winner = "A" if i % 2 == 0 else "B"
        votes.append(
            Vote(
                sample_a_id=str(samples[0].id),
                sample_b_id=str(samples[1].id),
                winner=winner,
                timestamp=base_time + timedelta(seconds=i),
            )
        )
    return votes


@pytest.fixture
def dominant_votes(two_model_samples: list[ArtSample]) -> list[Vote]:
    """Create 100 votes where model1 wins 90%."""
    from datetime import datetime, timedelta

    samples = two_model_samples
    base_time = datetime.now()
    votes = []
    for i in range(100):
        winner = "A" if i < 90 else "B"
        votes.append(
            Vote(
                sample_a_id=str(samples[0].id),
                sample_b_id=str(samples[1].id),
                winner=winner,
                timestamp=base_time + timedelta(seconds=i),
            )
        )
    return votes


@pytest.fixture
def three_model_votes(three_model_samples: list[ArtSample]) -> list[Vote]:
    """Create votes for 3 models with clear ranking: model1 > model2 > model3."""
    from datetime import datetime, timedelta

    samples = three_model_samples
    base_time = datetime.now()
    votes = []
    i = 0

    for j in range(50):
        winner = "A" if j < 40 else "B"
        votes.append(
            Vote(
                sample_a_id=str(samples[0].id),
                sample_b_id=str(samples[1].id),
                winner=winner,
                timestamp=base_time + timedelta(seconds=i),
            )
        )
        i += 1

    for j in range(50):
        winner = "A" if j < 40 else "B"
        votes.append(
            Vote(
                sample_a_id=str(samples[1].id),
                sample_b_id=str(samples[2].id),
                winner=winner,
                timestamp=base_time + timedelta(seconds=i),
            )
        )
        i += 1

    for j in range(50):
        winner = "A" if j < 45 else "B"
        votes.append(
            Vote(
                sample_a_id=str(samples[0].id),
                sample_b_id=str(samples[2].id),
                winner=winner,
                timestamp=base_time + timedelta(seconds=i),
            )
        )
        i += 1

    return votes


class TestRankingStabilityAnalyzer:
    """Tests for RankingStabilityAnalyzer class."""

    def test_initialization_with_seed(self) -> None:
        """Analyzer can be initialized with seed."""
        analyzer = RankingStabilityAnalyzer(seed=42)
        assert analyzer.seed == 42

    def test_initialization_without_seed(self) -> None:
        """Analyzer can be initialized without seed."""
        analyzer = RankingStabilityAnalyzer()
        assert analyzer.seed is None

    def test_calculate_stability_with_insufficient_iterations(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Raises InsufficientDataError when n_iterations < MIN_ITERATIONS."""
        analyzer = RankingStabilityAnalyzer()
        with pytest.raises(InsufficientDataError) as exc_info:
            analyzer.calculate_stability(balanced_votes, two_model_samples, n_iterations=1)
        assert "must be at least 2" in str(exc_info.value)

    def test_calculate_stability_with_empty_votes(self, two_model_samples: list[ArtSample]) -> None:
        """Returns empty dict when no votes provided."""
        analyzer = RankingStabilityAnalyzer()
        result = analyzer.calculate_stability([], two_model_samples, n_iterations=100)
        assert result == {}

    def test_calculate_stability_dominant_model_has_high_stability(
        self, dominant_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Dominant model should usually be rank 1."""
        analyzer = RankingStabilityAnalyzer(seed=42)
        result = analyzer.calculate_stability(dominant_votes, two_model_samples, n_iterations=100)
        assert result["model1"].modal_rank == 1
        assert result["model2"].modal_rank == 2
        assert result["model1"].rank_stability_pct > 0.5

    def test_calculate_stability_balanced_models_have_lower_stability(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Balanced models should have lower rank stability."""
        analyzer = RankingStabilityAnalyzer(seed=42)
        result = analyzer.calculate_stability(balanced_votes, two_model_samples, n_iterations=100)
        assert result["model1"].rank_stability_pct < 0.95
        assert result["model2"].rank_stability_pct < 0.95

    def test_calculate_stability_rank_distribution_sums_to_one(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Rank distribution percentages should sum to approximately 1."""
        analyzer = RankingStabilityAnalyzer(seed=42)
        result = analyzer.calculate_stability(balanced_votes, two_model_samples, n_iterations=100)
        for rs in result.values():
            total = sum(rs.rank_distribution.values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_calculate_stability_modal_rank_is_most_frequent(
        self, dominant_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Modal rank should be most frequent rank."""
        analyzer = RankingStabilityAnalyzer(seed=42)
        result = analyzer.calculate_stability(dominant_votes, two_model_samples, n_iterations=100)
        for rs in result.values():
            modal_freq = rs.rank_distribution.get(rs.modal_rank, 0)
            for _rank, freq in rs.rank_distribution.items():
                assert modal_freq >= freq

    def test_calculate_stability_three_models(
        self, three_model_votes: list[Vote], three_model_samples: list[ArtSample]
    ) -> None:
        """Should handle three models correctly."""
        analyzer = RankingStabilityAnalyzer(seed=42)
        result = analyzer.calculate_stability(
            three_model_votes, three_model_samples, n_iterations=100
        )
        assert len(result) == 3
        assert all(model in result for model in ["model1", "model2", "model3"])

    def test_calculate_stability_seed_produces_reproducible_results(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Same seed produces same results."""
        analyzer1 = RankingStabilityAnalyzer(seed=42)
        analyzer2 = RankingStabilityAnalyzer(seed=42)

        result1 = analyzer1.calculate_stability(balanced_votes, two_model_samples, n_iterations=100)
        result2 = analyzer2.calculate_stability(balanced_votes, two_model_samples, n_iterations=100)

        assert result1["model1"].modal_rank == result2["model1"].modal_rank
        assert result1["model1"].rank_stability_pct == result2["model1"].rank_stability_pct

    def test_calculate_stability_with_progress_callback(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Progress callback is called during calculation."""
        analyzer = RankingStabilityAnalyzer(seed=42)
        calls = []

        def callback(current: int, total: int) -> None:
            calls.append((current, total))

        analyzer.calculate_stability(
            balanced_votes, two_model_samples, n_iterations=10, progress_callback=callback
        )

        assert len(calls) == 10
        assert calls[0] == (1, 10)
        assert calls[-1] == (10, 10)

    def test_ratings_to_ranks_handles_ties(self) -> None:
        """Ties should get different ranks (arbitrary but consistent)."""
        analyzer = RankingStabilityAnalyzer()
        ratings = {"a": 1500.0, "b": 1500.0, "c": 1400.0}
        ranks = analyzer._ratings_to_ranks(ratings)
        assert set(ranks.values()) == {1, 2, 3}
        assert ranks["c"] == 3

    def test_ratings_to_ranks_correct_order(self) -> None:
        """Ratings should convert to correct ranks."""
        analyzer = RankingStabilityAnalyzer()
        ratings = {"a": 1600.0, "b": 1500.0, "c": 1400.0}
        ranks = analyzer._ratings_to_ranks(ratings)
        assert ranks["a"] == 1
        assert ranks["b"] == 2
        assert ranks["c"] == 3
