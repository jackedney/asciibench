"""Elo stability metrics for statistical validation.

This module provides functions to assess whether Elo ratings are
statistically stable enough for publication in leaderboards and
research papers.

Metrics implemented:
1. Bootstrap Confidence Intervals - 95% CIs for each model's rating
2. Ranking Stability - How often ranks hold across bootstrap samples
3. Bradley-Terry Significance - Pairwise p-values for ranking claims
4. Convergence Tracking - Rating change over time

Dependencies:
    - asciibench.common.models: Vote, ArtSample models
    - asciibench.analyst.elo: calculate_elo function
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from asciibench.analyst.elo import calculate_elo

if TYPE_CHECKING:
    from asciibench.common.models import ArtSample, Vote


# ============================================================================
# Data Classes for Structured Results
# ============================================================================


@dataclass
class ConfidenceInterval:
    """95% confidence interval for a model's Elo rating."""

    model_id: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_width: float


@dataclass
class RankingStability:
    """Stability metrics for a model's rank position."""

    model_id: str
    modal_rank: int  # Most frequent rank
    rank_stability_pct: float  # How often it holds this rank
    rank_distribution: dict[int, float] = field(default_factory=dict)


@dataclass
class PairwiseSignificance:
    """Bradley-Terry significance test for pairwise comparison."""

    model_a: str
    model_b: str
    wins_a: float  # Ties count as 0.5 for each side
    wins_b: float  # Ties count as 0.5 for each side
    p_value: float
    is_significant: bool  # p < 0.05


@dataclass
class ConvergenceMetrics:
    """Convergence tracking for rating stability over time."""

    model_id: str
    final_rating: float
    max_change_last_n: float  # Max rating change in last N votes
    trend_slope: float  # Approaching zero = converged
    is_converged: bool  # Changed <20 points in last 100 votes


@dataclass
class StabilityReport:
    """Complete stability report for all models."""

    confidence_intervals: dict[str, ConfidenceInterval]
    ranking_stability: dict[str, RankingStability]
    pairwise_significance: list[PairwiseSignificance]
    convergence: dict[str, ConvergenceMetrics]

    # Summary thresholds
    is_stable_for_publication: bool
    stability_warnings: list[str]
    stability_score: float  # 0-100


# ============================================================================
# 1. Bootstrap Confidence Intervals
# ============================================================================


def bootstrap_confidence_intervals(
    votes: list[Vote],
    samples: list[ArtSample],
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, ConfidenceInterval]:
    """Calculate bootstrap 95% confidence intervals for Elo ratings.

    Resamples votes with replacement and calculates Elo for each
    bootstrap sample to estimate rating uncertainty.

    Args:
        votes: List of Vote objects
        samples: List of ArtSample objects for model lookup
        n_iterations: Number of bootstrap iterations (default 1000)
        confidence_level: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        Dict mapping model_id to ConfidenceInterval dataclass

    Thresholds:
        - CIs shouldn't overlap for models claimed to be "different"
        - CI width < 150 points suggests reasonable stability
    """
    if n_iterations < 2:
        raise ValueError("n_iterations must be at least 2 for CI calculation")

    if not (0 < confidence_level <= 1.0):
        raise ValueError("confidence_level must be in (0, 1]")

    if not votes:
        return {}

    rng = random.Random(seed)

    # Calculate point estimates
    point_estimates = calculate_elo(votes, samples)
    if not point_estimates:
        return {}

    # Collect bootstrap distributions
    model_ids = list(point_estimates.keys())
    distributions: dict[str, list[float]] = {m: [] for m in model_ids}

    for i in range(n_iterations):
        resampled = _resample_votes(votes, rng)
        bootstrap_ratings = calculate_elo(resampled, samples)

        for model_id in model_ids:
            rating = bootstrap_ratings.get(model_id, point_estimates[model_id])
            distributions[model_id].append(rating)

        if progress_callback:
            progress_callback(i + 1, n_iterations)

    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_pct = alpha / 2
    upper_pct = 1 - alpha / 2

    results: dict[str, ConfidenceInterval] = {}
    for model_id in model_ids:
        dist = sorted(distributions[model_id])
        n = len(dist)
        lower_idx = max(0, int(lower_pct * n))
        upper_idx = min(n - 1, int(upper_pct * n) - 1)

        ci_lower = dist[lower_idx]
        ci_upper = dist[upper_idx]

        results[model_id] = ConfidenceInterval(
            model_id=model_id,
            point_estimate=point_estimates[model_id],
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_width=ci_upper - ci_lower,
        )

    return results


def _resample_votes(votes: list[Vote], rng: random.Random) -> list[Vote]:
    """Resample votes with replacement."""
    n = len(votes)
    return [votes[rng.randint(0, n - 1)] for _ in range(n)]


# ============================================================================
# 2. Ranking Stability
# ============================================================================


def calculate_ranking_stability(
    votes: list[Vote],
    samples: list[ArtSample],
    n_iterations: int = 1000,
    seed: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, RankingStability]:
    """Calculate ranking stability from bootstrap samples.

    Measures how often each model maintains its rank position across
    bootstrap resamples.

    Args:
        votes: List of Vote objects
        samples: List of ArtSample objects
        n_iterations: Bootstrap iterations
        seed: Random seed
        progress_callback: Optional callback(current, total) for progress

    Returns:
        Dict mapping model_id to RankingStability

    Thresholds:
        - Top-3 should maintain ranks in >90% of bootstraps
    """
    if not votes:
        return {}

    rng = random.Random(seed)

    point_estimates = calculate_elo(votes, samples)
    if not point_estimates:
        return {}

    model_ids = list(point_estimates.keys())
    rank_counts: dict[str, dict[int, int]] = {m: {} for m in model_ids}

    for i in range(n_iterations):
        resampled = _resample_votes(votes, rng)
        bootstrap_ratings = calculate_elo(resampled, samples)

        # Fill missing models with point estimates
        for model_id in model_ids:
            if model_id not in bootstrap_ratings:
                bootstrap_ratings[model_id] = point_estimates[model_id]

        ranks = _ratings_to_ranks(bootstrap_ratings)

        for model_id, rank in ranks.items():
            if model_id in rank_counts:
                rank_counts[model_id][rank] = rank_counts[model_id].get(rank, 0) + 1

        if progress_callback:
            progress_callback(i + 1, n_iterations)

    results: dict[str, RankingStability] = {}
    for model_id in model_ids:
        counts = rank_counts[model_id]
        if not counts:
            continue

        modal_rank = max(counts.keys(), key=lambda r: counts[r])
        modal_count = counts[modal_rank]
        rank_stability_pct = modal_count / n_iterations

        # Convert counts to percentages
        rank_distribution = {r: c / n_iterations for r, c in counts.items()}

        results[model_id] = RankingStability(
            model_id=model_id,
            modal_rank=modal_rank,
            rank_stability_pct=rank_stability_pct,
            rank_distribution=rank_distribution,
        )

    return results


def _ratings_to_ranks(ratings: dict[str, float]) -> dict[str, int]:
    """Convert rating dict to rank dict (1 = highest rating)."""
    sorted_models = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    return {model_id: rank for rank, (model_id, _) in enumerate(sorted_models, 1)}


# ============================================================================
# 3. Bradley-Terry Statistical Significance
# ============================================================================


def bradley_terry_significance(
    votes: list[Vote],
    samples: list[ArtSample],
) -> list[PairwiseSignificance]:
    """Calculate statistical significance for pairwise comparisons.

    Uses a binomial test to determine if one model is significantly
    better than another based on head-to-head win counts.

    Args:
        votes: List of Vote objects
        samples: List of ArtSample objects

    Returns:
        List of PairwiseSignificance for adjacent model pairs (by Elo rank)

    Thresholds:
        - p < 0.05 for ranking claims between adjacent models
    """
    if not votes:
        return []

    # Build win matrix
    sample_to_model: dict[str, str] = {str(s.id): s.model_id for s in samples}
    win_matrix: dict[str, dict[str, float]] = {}

    for vote in votes:
        if vote.winner == "fail":
            continue

        model_a = sample_to_model.get(vote.sample_a_id)
        model_b = sample_to_model.get(vote.sample_b_id)

        if not model_a or not model_b or model_a == model_b:
            continue

        if model_a not in win_matrix:
            win_matrix[model_a] = {}
        if model_b not in win_matrix:
            win_matrix[model_b] = {}

        if vote.winner == "A":
            win_matrix[model_a][model_b] = win_matrix[model_a].get(model_b, 0.0) + 1.0
        elif vote.winner == "B":
            win_matrix[model_b][model_a] = win_matrix[model_b].get(model_a, 0.0) + 1.0
        elif vote.winner == "tie":
            # Tie counts as 0.5 win for each side
            win_matrix[model_a][model_b] = win_matrix[model_a].get(model_b, 0.0) + 0.5
            win_matrix[model_b][model_a] = win_matrix[model_b].get(model_a, 0.0) + 0.5

    # Get Elo ratings to determine adjacent pairs
    elo_ratings = calculate_elo(votes, samples)
    if not elo_ratings:
        return []

    sorted_models = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)

    results: list[PairwiseSignificance] = []
    for i in range(len(sorted_models) - 1):
        model_a, _ = sorted_models[i]
        model_b, _ = sorted_models[i + 1]

        wins_a = win_matrix.get(model_a, {}).get(model_b, 0.0)
        wins_b = win_matrix.get(model_b, {}).get(model_a, 0.0)
        total = wins_a + wins_b

        if total == 0:
            p_value = 1.0
        else:
            # Two-tailed binomial test: H0 is p=0.5
            # Note: total is always an integer since each vote (including ties)
            # contributes 1 to the total (0.5 + 0.5 for ties)
            p_value = _binomial_test_two_tailed(wins_a, int(total), 0.5)

        results.append(
            PairwiseSignificance(
                model_a=model_a,
                model_b=model_b,
                wins_a=wins_a,
                wins_b=wins_b,
                p_value=p_value,
                is_significant=p_value < 0.05,
            )
        )

    return results


def _binomial_test_two_tailed(k: float, n: int, p: float = 0.5) -> float:
    """Two-tailed binomial test p-value.

    Tests H0: probability of success = p
    against H1: probability of success != p

    Uses exact binomial calculation (pure Python, no scipy).

    Args:
        k: Number of successes (can be fractional due to ties counting as 0.5)
        n: Number of trials
        p: Null hypothesis probability (default 0.5)

    Returns:
        Two-tailed p-value
    """
    if n == 0:
        return 1.0

    # Calculate probability of observing k or more extreme values
    # For two-tailed: sum probabilities of outcomes as extreme or more extreme

    expected = n * p
    observed_deviation = abs(k - expected)

    # Sum probabilities of all outcomes at least as extreme
    p_value = 0.0
    for i in range(n + 1):
        if abs(i - expected) >= observed_deviation:
            p_value += _binomial_pmf(i, n, p)

    return min(p_value, 1.0)


def _binomial_pmf(k: int, n: int, p: float) -> float:
    """Binomial probability mass function: P(X = k)."""
    if k < 0 or k > n:
        return 0.0

    # Use log-space to avoid overflow for large n
    log_coef = _log_binomial_coefficient(n, k)
    log_prob = log_coef + k * math.log(p) + (n - k) * math.log(1 - p)
    return math.exp(log_prob)


def _log_binomial_coefficient(n: int, k: int) -> float:
    """Log of binomial coefficient C(n, k) = n! / (k! * (n-k)!)."""
    if k > n or k < 0:
        return float("-inf")
    if k == 0 or k == n:
        return 0.0

    # Use log-gamma for numerical stability
    return _log_factorial(n) - _log_factorial(k) - _log_factorial(n - k)


def _log_factorial(n: int) -> float:
    """Log factorial using Stirling's approximation for large n."""
    if n <= 1:
        return 0.0
    if n < 20:
        # Direct calculation for small n
        return sum(math.log(i) for i in range(2, n + 1))
    # Stirling's approximation for large n
    return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)


# ============================================================================
# 4. Convergence Tracking
# ============================================================================


def calculate_convergence(
    votes: list[Vote],
    samples: list[ArtSample],
    window_size: int = 100,
    change_threshold: float = 20.0,
) -> dict[str, ConvergenceMetrics]:
    """Track Elo convergence over time.

    Calculates ratings incrementally and measures stability in a
    trailing window.

    Args:
        votes: List of Vote objects (will be sorted by timestamp)
        samples: List of ArtSample objects
        window_size: Number of recent votes to consider for convergence
        change_threshold: Max acceptable change for "converged" status

    Returns:
        Dict mapping model_id to ConvergenceMetrics

    Thresholds:
        - Ratings changed <20 points in last 100 votes = converged
        - Trend slope approaching zero indicates stability
    """
    if not votes:
        return {}

    # Sort votes by timestamp
    sorted_votes = sorted(votes, key=lambda v: v.timestamp)

    # Calculate rating history at checkpoints
    checkpoint_interval = max(1, len(sorted_votes) // 50)  # ~50 checkpoints
    history = _calculate_rating_history(sorted_votes, samples, checkpoint_interval)

    if not history:
        return {}

    # Final ratings
    final_ratings = calculate_elo(sorted_votes, samples)

    results: dict[str, ConvergenceMetrics] = {}
    for model_id, checkpoints in history.items():
        if len(checkpoints) < 2:
            results[model_id] = ConvergenceMetrics(
                model_id=model_id,
                final_rating=final_ratings.get(model_id, 1500.0),
                max_change_last_n=0.0,
                trend_slope=0.0,
                is_converged=True,
            )
            continue

        # Calculate max change in recent window based on window_size
        cutoff_vote_idx = max(0, len(sorted_votes) - window_size)
        recent_checkpoints = [
            (idx, rating) for idx, rating in checkpoints if idx >= cutoff_vote_idx
        ]
        if len(recent_checkpoints) < 2:
            recent_checkpoints = checkpoints[-2:] if len(checkpoints) >= 2 else checkpoints
        if len(recent_checkpoints) >= 2:
            recent_ratings = [r for _, r in recent_checkpoints]
            max_change = max(recent_ratings) - min(recent_ratings)
        else:
            max_change = 0.0

        # Calculate trend slope
        trend_slope = _calculate_trend_slope(checkpoints)

        # Determine convergence
        is_converged = max_change < change_threshold and abs(trend_slope) < 1.0

        results[model_id] = ConvergenceMetrics(
            model_id=model_id,
            final_rating=final_ratings.get(model_id, 1500.0),
            max_change_last_n=max_change,
            trend_slope=trend_slope,
            is_converged=is_converged,
        )

    return results


def _calculate_rating_history(
    votes: list[Vote],
    samples: list[ArtSample],
    checkpoint_interval: int = 10,
) -> dict[str, list[tuple[int, float]]]:
    """Calculate rating at checkpoints during vote processing.

    Returns dict mapping model_id to list of (vote_count, rating) tuples.
    """
    if not votes:
        return {}

    history: dict[str, list[tuple[int, float]]] = {}
    n_votes = len(votes)

    for i in range(checkpoint_interval, n_votes + 1, checkpoint_interval):
        partial_votes = votes[:i]
        ratings = calculate_elo(partial_votes, samples)

        for model_id, rating in ratings.items():
            if model_id not in history:
                history[model_id] = []
            history[model_id].append((i, rating))

    # Add final checkpoint if not already included
    if n_votes % checkpoint_interval != 0:
        final_ratings = calculate_elo(votes, samples)
        for model_id, rating in final_ratings.items():
            if model_id not in history:
                history[model_id] = []
            history[model_id].append((n_votes, rating))

    return history


def _calculate_trend_slope(history: list[tuple[int, float]]) -> float:
    """Calculate linear regression slope of rating history.

    Pure Python implementation of simple linear regression.

    Args:
        history: List of (vote_count, rating) tuples

    Returns:
        Slope in rating points per vote
    """
    if len(history) < 2:
        return 0.0

    n = len(history)
    sum_x = sum(x for x, _ in history)
    sum_y = sum(y for _, y in history)
    sum_xy = sum(x * y for x, y in history)
    sum_x2 = sum(x * x for x, _ in history)

    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope


# ============================================================================
# Combined Stability Report
# ============================================================================


def generate_stability_report(
    votes: list[Vote],
    samples: list[ArtSample],
    n_bootstrap: int = 1000,
    convergence_window: int = 100,
    seed: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> StabilityReport:
    """Generate comprehensive stability report.

    Combines all four stability metrics and provides overall
    assessment of whether ratings are stable enough for publication.

    Args:
        votes: List of Vote objects
        samples: List of ArtSample objects
        n_bootstrap: Bootstrap iterations
        convergence_window: Window for convergence tracking
        seed: Random seed for reproducibility
        progress_callback: Optional callback(current, total) for progress

    Returns:
        StabilityReport with all metrics and overall assessment

    Publication Thresholds:
        - All top-3 models maintain rank in >90% of bootstraps
        - Adjacent models have p < 0.05 or overlapping CIs (for ties)
        - All models converged (changed <20 points in last 100 votes)
        - CI widths < 150 points for all models
    """
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2 for CI calculation")

    if not votes:
        return StabilityReport(
            confidence_intervals={},
            ranking_stability={},
            pairwise_significance=[],
            convergence={},
            is_stable_for_publication=False,
            stability_warnings=["No votes to analyze"],
            stability_score=0.0,
        )

    # Calculate all metrics
    # Bootstrap is shared between CI and ranking stability
    rng = random.Random(seed)

    point_estimates = calculate_elo(votes, samples)
    if not point_estimates:
        return StabilityReport(
            confidence_intervals={},
            ranking_stability={},
            pairwise_significance=[],
            convergence={},
            is_stable_for_publication=False,
            stability_warnings=["No valid votes for Elo calculation"],
            stability_score=0.0,
        )

    model_ids = list(point_estimates.keys())
    distributions: dict[str, list[float]] = {m: [] for m in model_ids}
    rank_counts: dict[str, dict[int, int]] = {m: {} for m in model_ids}

    # Single bootstrap loop for both CI and ranking stability
    for i in range(n_bootstrap):
        resampled = _resample_votes(votes, rng)
        bootstrap_ratings = calculate_elo(resampled, samples)

        # Fill missing models with point estimates
        for model_id in model_ids:
            if model_id not in bootstrap_ratings:
                bootstrap_ratings[model_id] = point_estimates[model_id]

        # Collect for CI
        for model_id in model_ids:
            distributions[model_id].append(bootstrap_ratings[model_id])

        # Collect for ranking stability
        ranks = _ratings_to_ranks(bootstrap_ratings)
        for model_id, rank in ranks.items():
            if model_id in rank_counts:
                rank_counts[model_id][rank] = rank_counts[model_id].get(rank, 0) + 1

        if progress_callback:
            progress_callback(i + 1, n_bootstrap)

    # Build confidence intervals
    confidence_intervals: dict[str, ConfidenceInterval] = {}
    for model_id in model_ids:
        dist = sorted(distributions[model_id])
        n = len(dist)
        ci_lower = dist[int(0.025 * n)]
        ci_upper = dist[int(0.975 * n) - 1]

        confidence_intervals[model_id] = ConfidenceInterval(
            model_id=model_id,
            point_estimate=point_estimates[model_id],
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_width=ci_upper - ci_lower,
        )

    # Build ranking stability
    ranking_stability: dict[str, RankingStability] = {}
    for model_id in model_ids:
        counts = rank_counts[model_id]
        if not counts:
            continue

        modal_rank = max(counts.keys(), key=lambda r: counts[r])
        modal_count = counts[modal_rank]
        rank_stability_pct = modal_count / n_bootstrap
        rank_distribution = {r: c / n_bootstrap for r, c in counts.items()}

        ranking_stability[model_id] = RankingStability(
            model_id=model_id,
            modal_rank=modal_rank,
            rank_stability_pct=rank_stability_pct,
            rank_distribution=rank_distribution,
        )

    # Bradley-Terry significance
    pairwise_significance = bradley_terry_significance(votes, samples)

    # Convergence tracking
    convergence = calculate_convergence(votes, samples, convergence_window)

    # Evaluate stability for publication
    warnings: list[str] = []
    score_components: list[float] = []

    # Check 1: Top-3 ranking stability > 90%
    sorted_by_elo = sorted(point_estimates.items(), key=lambda x: x[1], reverse=True)
    top_3_models = [m for m, _ in sorted_by_elo[:3]]
    top_3_stability = [
        ranking_stability[m].rank_stability_pct for m in top_3_models if m in ranking_stability
    ]
    if top_3_stability:
        min_top3_stability = min(top_3_stability)
        if min_top3_stability < 0.9:
            warnings.append(
                f"Top-3 rank stability is {min_top3_stability:.0%}, below 90% threshold"
            )
        score_components.append(min(min_top3_stability / 0.9, 1.0) * 25)
    else:
        score_components.append(0)

    # Check 2: CI widths < 150
    ci_widths = [ci.ci_width for ci in confidence_intervals.values()]
    if ci_widths:
        max_ci_width = max(ci_widths)
        if max_ci_width > 150:
            warnings.append(f"Max CI width is {max_ci_width:.0f}, above 150 threshold")
        if max_ci_width <= 0:
            # All CI widths are 0, treat as fully stable
            score_components.append(25)
        else:
            score_components.append(min(150 / max_ci_width, 1.0) * 25)
    else:
        score_components.append(0)

    # Check 3: Adjacent models have significance or acknowledged ties
    non_significant_adjacent = [ps for ps in pairwise_significance if not ps.is_significant]
    # Check if non-significant pairs have overlapping CIs (acceptable tie)
    unclear_rankings = 0
    for ps in non_significant_adjacent:
        ci_a = confidence_intervals.get(ps.model_a)
        ci_b = confidence_intervals.get(ps.model_b)
        if ci_a and ci_b:
            # Check if CIs overlap (acceptable for non-significant)
            overlaps = ci_a.ci_lower <= ci_b.ci_upper and ci_b.ci_lower <= ci_a.ci_upper
            if not overlaps:
                unclear_rankings += 1

    if unclear_rankings > 0:
        warnings.append(f"{unclear_rankings} adjacent model pair(s) have unclear ranking")
    sig_ratio = 1.0 - (unclear_rankings / max(len(pairwise_significance), 1))
    score_components.append(sig_ratio * 25)

    # Check 4: Convergence
    non_converged = [c for c in convergence.values() if not c.is_converged]
    if non_converged:
        warnings.append(
            f"{len(non_converged)} model(s) have not converged "
            f"({', '.join(c.model_id for c in non_converged[:3])})"
        )
    conv_ratio = 1.0 - (len(non_converged) / max(len(convergence), 1))
    score_components.append(conv_ratio * 25)

    # Overall score and stability determination
    stability_score = sum(score_components)
    is_stable = len(warnings) == 0

    return StabilityReport(
        confidence_intervals=confidence_intervals,
        ranking_stability=ranking_stability,
        pairwise_significance=pairwise_significance,
        convergence=convergence,
        is_stable_for_publication=is_stable,
        stability_warnings=warnings,
        stability_score=stability_score,
    )
