"""Tests for Elo stability metrics."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from asciibench.analyst.stability import (
    _binomial_pmf,
    _binomial_test_two_tailed,
    _calculate_rating_history,
    _calculate_trend_slope,
    _ratings_to_ranks,
    _resample_votes,
    bootstrap_confidence_intervals,
    bradley_terry_significance,
    calculate_convergence,
    calculate_ranking_stability,
    generate_stability_report,
)
from asciibench.common.models import ArtSample, Vote

# ============================================================================
# Test Fixtures
# ============================================================================


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
    samples = three_model_samples
    base_time = datetime.now()
    votes = []
    i = 0

    # model1 vs model2: model1 wins 80%
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

    # model2 vs model3: model2 wins 80%
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

    # model1 vs model3: model1 wins 90%
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


# ============================================================================
# Bootstrap Confidence Intervals Tests
# ============================================================================


class TestBootstrapConfidenceIntervals:
    """Tests for bootstrap_confidence_intervals function."""

    def test_empty_votes_returns_empty_dict(self, two_model_samples: list[ArtSample]) -> None:
        """Empty votes returns empty confidence intervals."""
        result = bootstrap_confidence_intervals([], two_model_samples)
        assert result == {}

    def test_returns_ci_for_each_model(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Returns confidence interval for each model with votes."""
        result = bootstrap_confidence_intervals(
            balanced_votes, two_model_samples, n_iterations=100, seed=42
        )
        assert "model1" in result
        assert "model2" in result

    def test_ci_contains_point_estimate(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """CI should contain the point estimate."""
        result = bootstrap_confidence_intervals(
            balanced_votes, two_model_samples, n_iterations=100, seed=42
        )
        for ci in result.values():
            assert ci.ci_lower <= ci.point_estimate <= ci.ci_upper

    def test_dominant_model_has_higher_ci(
        self, dominant_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Model with more wins has higher point estimate."""
        result = bootstrap_confidence_intervals(
            dominant_votes, two_model_samples, n_iterations=100, seed=42
        )
        # With 90% win rate, model1 should have higher point estimate
        assert result["model1"].point_estimate > result["model2"].point_estimate
        # And model1's CI should be mostly above model2's
        assert result["model1"].ci_lower > result["model2"].ci_lower

    def test_balanced_votes_have_overlapping_cis(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Balanced votes should have overlapping CIs."""
        result = bootstrap_confidence_intervals(
            balanced_votes, two_model_samples, n_iterations=100, seed=42
        )
        model1_ci = result["model1"]
        model2_ci = result["model2"]
        overlaps = (
            model1_ci.ci_lower <= model2_ci.ci_upper and model2_ci.ci_lower <= model1_ci.ci_upper
        )
        assert overlaps

    def test_seed_produces_reproducible_results(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Same seed produces same results."""
        result1 = bootstrap_confidence_intervals(
            balanced_votes, two_model_samples, n_iterations=100, seed=42
        )
        result2 = bootstrap_confidence_intervals(
            balanced_votes, two_model_samples, n_iterations=100, seed=42
        )
        assert result1["model1"].ci_lower == result2["model1"].ci_lower
        assert result1["model1"].ci_upper == result2["model1"].ci_upper

    def test_ci_width_is_correct(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """CI width should equal upper - lower."""
        result = bootstrap_confidence_intervals(
            balanced_votes, two_model_samples, n_iterations=100, seed=42
        )
        for ci in result.values():
            assert ci.ci_width == pytest.approx(ci.ci_upper - ci.ci_lower)


# ============================================================================
# Ranking Stability Tests
# ============================================================================


class TestRankingStability:
    """Tests for calculate_ranking_stability function."""

    def test_empty_votes_returns_empty(self, two_model_samples: list[ArtSample]) -> None:
        """Empty votes returns empty stability metrics."""
        result = calculate_ranking_stability([], two_model_samples)
        assert result == {}

    def test_dominant_model_has_high_stability(
        self, dominant_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Dominant model should usually be rank 1."""
        result = calculate_ranking_stability(
            dominant_votes, two_model_samples, n_iterations=100, seed=42
        )
        # With 90% win rate, model1 should usually be rank 1
        assert result["model1"].modal_rank == 1
        assert result["model2"].modal_rank == 2
        # With 2 models, both have same stability (complementary ranks)
        assert result["model1"].rank_stability_pct > 0.5

    def test_balanced_models_have_lower_stability(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Balanced models should have lower rank stability."""
        result = calculate_ranking_stability(
            balanced_votes, two_model_samples, n_iterations=100, seed=42
        )
        # Neither should be >90% stable with balanced votes
        assert result["model1"].rank_stability_pct < 0.95
        assert result["model2"].rank_stability_pct < 0.95

    def test_rank_distribution_sums_to_one(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Rank distribution percentages should sum to approximately 1."""
        result = calculate_ranking_stability(
            balanced_votes, two_model_samples, n_iterations=100, seed=42
        )
        for rs in result.values():
            total = sum(rs.rank_distribution.values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_modal_rank_is_most_frequent(
        self, dominant_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Modal rank should be the most frequent rank."""
        result = calculate_ranking_stability(
            dominant_votes, two_model_samples, n_iterations=100, seed=42
        )
        for rs in result.values():
            modal_freq = rs.rank_distribution.get(rs.modal_rank, 0)
            for _rank, freq in rs.rank_distribution.items():
                assert modal_freq >= freq


# ============================================================================
# Bradley-Terry Significance Tests
# ============================================================================


class TestBradleyTerrySignificance:
    """Tests for bradley_terry_significance function."""

    def test_empty_votes_returns_empty(self, two_model_samples: list[ArtSample]) -> None:
        """Empty votes returns empty list."""
        result = bradley_terry_significance([], two_model_samples)
        assert result == []

    def test_dominant_comparison_is_significant(
        self, dominant_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """90% win rate should be statistically significant."""
        result = bradley_terry_significance(dominant_votes, two_model_samples)
        assert len(result) == 1  # One adjacent pair
        assert result[0].is_significant
        assert result[0].p_value < 0.05

    def test_balanced_comparison_not_significant(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """50% win rate should not be significant."""
        result = bradley_terry_significance(balanced_votes, two_model_samples)
        assert len(result) == 1
        assert not result[0].is_significant
        assert result[0].p_value > 0.05

    def test_returns_adjacent_pairs_only(
        self, three_model_votes: list[Vote], three_model_samples: list[ArtSample]
    ) -> None:
        """Should return significance for adjacent model pairs by Elo rank."""
        result = bradley_terry_significance(three_model_votes, three_model_samples)
        # 3 models = 2 adjacent pairs
        assert len(result) == 2

    def test_win_counts_are_correct(
        self, dominant_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Win counts should match the input data."""
        result = bradley_terry_significance(dominant_votes, two_model_samples)
        # model1 won 90, model2 won 10
        ps = result[0]
        # Higher ranked model (model1) should be model_a
        assert ps.wins_a == 90
        assert ps.wins_b == 10


# ============================================================================
# Convergence Tests
# ============================================================================


class TestConvergence:
    """Tests for calculate_convergence function."""

    def test_empty_votes_returns_empty(self, two_model_samples: list[ArtSample]) -> None:
        """Empty votes returns empty convergence metrics."""
        result = calculate_convergence([], two_model_samples)
        assert result == {}

    def test_returns_metrics_for_each_model(
        self, dominant_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Should return convergence metrics for each model."""
        result = calculate_convergence(dominant_votes, two_model_samples)
        assert "model1" in result
        assert "model2" in result

    def test_final_rating_matches_elo(
        self, dominant_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Final rating should match calculated Elo."""
        from asciibench.analyst.elo import calculate_elo

        elo_ratings = calculate_elo(dominant_votes, two_model_samples)
        convergence = calculate_convergence(dominant_votes, two_model_samples)

        for model_id, metrics in convergence.items():
            assert metrics.final_rating == pytest.approx(elo_ratings[model_id], abs=0.01)

    def test_stable_ratings_are_converged(
        self, dominant_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Convergence metrics should be calculated for all models."""
        result = calculate_convergence(
            dominant_votes, two_model_samples, window_size=50, change_threshold=100.0
        )
        # All models should have convergence metrics
        assert len(result) == 2
        # Trend slopes should exist
        for metrics in result.values():
            assert isinstance(metrics.trend_slope, float)
            assert isinstance(metrics.max_change_last_n, float)


# ============================================================================
# Stability Report Tests
# ============================================================================


class TestGenerateStabilityReport:
    """Tests for generate_stability_report function."""

    def test_empty_votes_returns_unstable(self, two_model_samples: list[ArtSample]) -> None:
        """Empty votes should return unstable report."""
        result = generate_stability_report([], two_model_samples)
        assert not result.is_stable_for_publication
        assert len(result.stability_warnings) > 0

    def test_includes_all_metrics(
        self, dominant_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Report should include all four metric types."""
        result = generate_stability_report(
            dominant_votes, two_model_samples, n_bootstrap=100, seed=42
        )
        assert len(result.confidence_intervals) > 0
        assert len(result.ranking_stability) > 0
        assert len(result.pairwise_significance) > 0
        assert len(result.convergence) > 0

    def test_stability_score_in_range(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Stability score should be 0-100."""
        result = generate_stability_report(
            balanced_votes, two_model_samples, n_bootstrap=100, seed=42
        )
        assert 0 <= result.stability_score <= 100

    def test_dominant_model_has_high_score(
        self, dominant_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Clear winner with enough votes should have high stability score."""
        result = generate_stability_report(
            dominant_votes, two_model_samples, n_bootstrap=100, seed=42
        )
        # With 90% win rate and 100 votes, should score well
        assert result.stability_score >= 50

    def test_seed_produces_reproducible_report(
        self, balanced_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Same seed produces same stability score."""
        result1 = generate_stability_report(
            balanced_votes, two_model_samples, n_bootstrap=100, seed=42
        )
        result2 = generate_stability_report(
            balanced_votes, two_model_samples, n_bootstrap=100, seed=42
        )
        assert result1.stability_score == result2.stability_score


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_resample_votes_same_length(self, balanced_votes: list[Vote]) -> None:
        """Resampled votes should have same length."""
        import random

        rng = random.Random(42)
        resampled = _resample_votes(balanced_votes, rng)
        assert len(resampled) == len(balanced_votes)

    def test_resample_votes_with_replacement(self, balanced_votes: list[Vote]) -> None:
        """Resampling should produce duplicates (with high probability)."""
        import random

        rng = random.Random(42)
        resampled = _resample_votes(balanced_votes, rng)
        # With replacement, we expect some duplicates
        ids = [v.id for v in resampled]
        # Very unlikely to have all unique with replacement
        assert len(ids) != len(set(ids))

    def test_ratings_to_ranks_correct_order(self) -> None:
        """Ratings should convert to correct ranks."""
        ratings = {"a": 1600.0, "b": 1500.0, "c": 1400.0}
        ranks = _ratings_to_ranks(ratings)
        assert ranks["a"] == 1
        assert ranks["b"] == 2
        assert ranks["c"] == 3

    def test_ratings_to_ranks_handles_ties(self) -> None:
        """Ties should get different ranks (arbitrary but consistent)."""
        ratings = {"a": 1500.0, "b": 1500.0, "c": 1400.0}
        ranks = _ratings_to_ranks(ratings)
        # Both a and b have 1500, one gets rank 1, one gets rank 2
        assert set(ranks.values()) == {1, 2, 3}
        assert ranks["c"] == 3

    def test_trend_slope_positive_for_increasing(self) -> None:
        """Positive slope for increasing ratings."""
        history = [(10, 1500.0), (20, 1510.0), (30, 1520.0), (40, 1530.0)]
        slope = _calculate_trend_slope(history)
        assert slope > 0

    def test_trend_slope_negative_for_decreasing(self) -> None:
        """Negative slope for decreasing ratings."""
        history = [(10, 1530.0), (20, 1520.0), (30, 1510.0), (40, 1500.0)]
        slope = _calculate_trend_slope(history)
        assert slope < 0

    def test_trend_slope_near_zero_for_stable(self) -> None:
        """Near-zero slope for stable ratings."""
        history = [(10, 1500.0), (20, 1501.0), (30, 1499.0), (40, 1500.0)]
        slope = _calculate_trend_slope(history)
        assert abs(slope) < 0.1

    def test_trend_slope_empty_history(self) -> None:
        """Empty or single-item history returns 0."""
        assert _calculate_trend_slope([]) == 0.0
        assert _calculate_trend_slope([(10, 1500.0)]) == 0.0


# ============================================================================
# Binomial Test Tests
# ============================================================================


class TestBinomialTest:
    """Tests for binomial test helper functions."""

    def test_binomial_pmf_sums_to_one(self) -> None:
        """PMF over all k should sum to 1."""
        n = 10
        p = 0.5
        total = sum(_binomial_pmf(k, n, p) for k in range(n + 1))
        assert total == pytest.approx(1.0, abs=0.0001)

    def test_binomial_pmf_fair_coin(self) -> None:
        """Fair coin PMF should be symmetric."""
        n = 10
        p = 0.5
        # P(X=3) should equal P(X=7) for fair coin
        assert _binomial_pmf(3, n, p) == pytest.approx(_binomial_pmf(7, n, p), abs=0.0001)

    def test_binomial_test_fair_result(self) -> None:
        """50/50 split should have high p-value."""
        p_value = _binomial_test_two_tailed(25, 50, 0.5)
        assert p_value > 0.9  # Very close to expected

    def test_binomial_test_extreme_result(self) -> None:
        """90/10 split should have low p-value."""
        p_value = _binomial_test_two_tailed(90, 100, 0.5)
        assert p_value < 0.001  # Very unlikely under H0

    def test_binomial_test_zero_trials(self) -> None:
        """Zero trials should return p-value of 1."""
        p_value = _binomial_test_two_tailed(0, 0, 0.5)
        assert p_value == 1.0


# ============================================================================
# Rating History Tests
# ============================================================================


class TestRatingHistory:
    """Tests for rating history calculation."""

    def test_history_has_checkpoints(
        self, dominant_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """History should have multiple checkpoints."""
        history = _calculate_rating_history(
            dominant_votes, two_model_samples, checkpoint_interval=10
        )
        assert len(history) > 0
        for model_history in history.values():
            assert len(model_history) >= 2

    def test_history_vote_counts_increase(
        self, dominant_votes: list[Vote], two_model_samples: list[ArtSample]
    ) -> None:
        """Vote counts in history should be increasing."""
        history = _calculate_rating_history(
            dominant_votes, two_model_samples, checkpoint_interval=10
        )
        for model_history in history.values():
            vote_counts = [vc for vc, _ in model_history]
            assert vote_counts == sorted(vote_counts)

    def test_empty_votes_returns_empty(self, two_model_samples: list[ArtSample]) -> None:
        """Empty votes returns empty history."""
        history = _calculate_rating_history([], two_model_samples, checkpoint_interval=10)
        assert history == {}
