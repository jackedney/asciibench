"""Tests for BootstrapAnalyzer class."""

from uuid import uuid4

import pytest

from asciibench.analyst.bootstrap_analyzer import BootstrapAnalyzer
from asciibench.analyst.errors import InsufficientDataError
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


class TestBootstrapAnalyzer:
    """Tests for BootstrapAnalyzer class."""

    def test_initialization_with_seed(self) -> None:
        """Analyzer can be initialized with seed."""
        analyzer = BootstrapAnalyzer(seed=42)
        assert analyzer.seed == 42

    def test_initialization_without_seed(self) -> None:
        """Analyzer can be initialized without seed."""
        analyzer = BootstrapAnalyzer()
        assert analyzer.seed is None

    def test_calculate_ci_with_insufficient_iterations(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Raises InsufficientDataError when n_samples < MIN_ITERATIONS."""
        analyzer = BootstrapAnalyzer()
        with pytest.raises(InsufficientDataError) as exc_info:
            analyzer.calculate_ci(balanced_votes, two_model_samples, n_samples=1)
        assert "must be at least 2" in str(exc_info.value)

    def test_calculate_ci_with_invalid_confidence_level(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Raises ValueError when confidence_level not in (0, 1]."""
        analyzer = BootstrapAnalyzer()
        with pytest.raises(ValueError, match="confidence_level must be in \\(0, 1\\]"):
            analyzer.calculate_ci(balanced_votes, two_model_samples, confidence_level=0.0)
        with pytest.raises(ValueError, match="confidence_level must be in \\(0, 1\\]"):
            analyzer.calculate_ci(balanced_votes, two_model_samples, confidence_level=1.5)

    def test_calculate_ci_with_empty_votes(self, two_model_samples: list[ArtSample]) -> None:
        """Returns empty dict when no votes provided."""
        analyzer = BootstrapAnalyzer()
        result = analyzer.calculate_ci([], two_model_samples, n_samples=100)
        assert result == {}

    def test_calculate_ci_returns_ci_for_each_model(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Returns confidence interval for each model with votes."""
        analyzer = BootstrapAnalyzer(seed=42)
        result = analyzer.calculate_ci(balanced_votes, two_model_samples, n_samples=100)
        assert "model1" in result
        assert "model2" in result

    def test_calculate_ci_contains_point_estimate(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """CI should contain the point estimate."""
        analyzer = BootstrapAnalyzer(seed=42)
        result = analyzer.calculate_ci(balanced_votes, two_model_samples, n_samples=100)
        for ci in result.values():
            assert ci.ci_lower <= ci.point_estimate <= ci.ci_upper

    def test_calculate_ci_dominant_model_has_higher_ci(
        self, dominant_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Model with more wins has higher point estimate."""
        analyzer = BootstrapAnalyzer(seed=42)
        result = analyzer.calculate_ci(dominant_votes, two_model_samples, n_samples=100)
        assert result["model1"].point_estimate > result["model2"].point_estimate
        assert result["model1"].ci_lower > result["model2"].ci_lower

    def test_calculate_ci_balanced_votes_have_overlapping_cis(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Balanced votes should have overlapping CIs."""
        analyzer = BootstrapAnalyzer(seed=42)
        result = analyzer.calculate_ci(balanced_votes, two_model_samples, n_samples=100)
        model1_ci = result["model1"]
        model2_ci = result["model2"]
        overlaps = (
            model1_ci.ci_lower <= model2_ci.ci_upper and model2_ci.ci_lower <= model1_ci.ci_upper
        )
        assert overlaps

    def test_calculate_ci_seed_produces_reproducible_results(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Same seed produces same results."""
        analyzer1 = BootstrapAnalyzer(seed=42)
        analyzer2 = BootstrapAnalyzer(seed=42)

        result1 = analyzer1.calculate_ci(balanced_votes, two_model_samples, n_samples=100)
        result2 = analyzer2.calculate_ci(balanced_votes, two_model_samples, n_samples=100)

        assert result1["model1"].ci_lower == result2["model1"].ci_lower
        assert result1["model1"].ci_upper == result2["model1"].ci_upper

    def test_calculate_ci_ci_width_is_correct(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """CI width should equal upper - lower."""
        analyzer = BootstrapAnalyzer(seed=42)
        result = analyzer.calculate_ci(balanced_votes, two_model_samples, n_samples=100)
        for ci in result.values():
            assert ci.ci_width == pytest.approx(ci.ci_upper - ci.ci_lower)

    def test_calculate_ci_with_progress_callback(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Progress callback is called during calculation."""
        analyzer = BootstrapAnalyzer(seed=42)
        calls = []

        def callback(current: int, total: int) -> None:
            calls.append((current, total))

        analyzer.calculate_ci(
            balanced_votes, two_model_samples, n_samples=10, progress_callback=callback
        )

        assert len(calls) == 10
        assert calls[0] == (1, 10)
        assert calls[-1] == (10, 10)

    def test_calculate_ci_custom_confidence_level(
        self, dominant_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Custom confidence level produces different CI width."""
        analyzer_95 = BootstrapAnalyzer(seed=42)
        result_95 = analyzer_95.calculate_ci(
            dominant_votes, two_model_samples, n_samples=100, confidence_level=0.95
        )

        analyzer_80 = BootstrapAnalyzer(seed=42)
        result_80 = analyzer_80.calculate_ci(
            dominant_votes, two_model_samples, n_samples=100, confidence_level=0.80
        )

        assert result_80["model1"].ci_width < result_95["model1"].ci_width
