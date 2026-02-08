"""Elo stability metrics for statistical validation.

This module provides functions to assess whether Elo ratings are
statistically stable enough for publication in leaderboards and
research papers.

This module acts as an orchestrator, composing specialized analyzers:
- BootstrapAnalyzer: Handles bootstrap CI calculations
- RankingStabilityAnalyzer: Handles ranking stability analysis
- PairwiseSignificanceAnalyzer: Handles Bradley-Terry significance tests
- ConvergenceAnalyzer: Handles convergence tracking

Dependencies:
    - asciibench.common.models: Vote, ArtSample models
    - asciibench.analyst.elo: calculate_elo function
"""

from __future__ import annotations

from collections.abc import Callable
import random

from asciibench.analyst.bootstrap_analyzer import BootstrapAnalyzer
from asciibench.analyst.convergence_analyzer import ConvergenceAnalyzer
from asciibench.analyst.elo import calculate_elo
from asciibench.analyst.pairwise_significance_analyzer import (
    PairwiseSignificanceAnalyzer,
)
from asciibench.analyst.ranking_stability_analyzer import RankingStabilityAnalyzer
from asciibench.analyst.stability_models import (
    ConfidenceInterval,
    ConvergenceMetrics,
    PairwiseSignificance,
    RankingStability,
    StabilityReport,
)
from asciibench.common.models import ArtSample, Vote


def bootstrap_confidence_intervals(
    votes: list[Vote],
    samples: list[ArtSample],
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, ConfidenceInterval]:
    """Calculate bootstrap 95% confidence intervals for Elo ratings."""
    if not votes:
        return {}
    analyzer = BootstrapAnalyzer(seed=seed)
    return analyzer.calculate_ci(
        votes,
        samples,
        n_samples=n_iterations,
        confidence_level=confidence_level,
        progress_callback=progress_callback,
    )


def calculate_ranking_stability(
    votes: list[Vote],
    samples: list[ArtSample],
    n_iterations: int = 1000,
    seed: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, RankingStability]:
    """Calculate ranking stability from bootstrap samples."""
    if not votes:
        return {}
    analyzer = RankingStabilityAnalyzer(seed=seed)
    return analyzer.calculate_stability(
        votes, samples, n_iterations=n_iterations, progress_callback=progress_callback
    )


def bradley_terry_significance(
    votes: list[Vote],
    samples: list[ArtSample],
) -> list[PairwiseSignificance]:
    """Calculate statistical significance for pairwise comparisons."""
    analyzer = PairwiseSignificanceAnalyzer()
    return analyzer.calculate_significance(votes, samples)


def calculate_convergence(
    votes: list[Vote],
    samples: list[ArtSample],
    window_size: int = 100,
    change_threshold: float = 20.0,
) -> dict[str, ConvergenceMetrics]:
    """Track Elo convergence over time."""
    analyzer = ConvergenceAnalyzer()
    return analyzer.calculate_convergence(votes, samples, window_size, change_threshold)


def _ratings_to_ranks(ratings: dict[str, float]) -> dict[str, int]:
    """Convert rating dict to rank dict (1 = highest rating)."""
    sorted_models = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    return {model_id: rank for rank, (model_id, _) in enumerate(sorted_models, 1)}


def _resample_votes(votes: list[Vote], rng: random.Random) -> list[Vote]:
    """Resample votes with replacement.

    Kept for backward compatibility with tests.
    """
    n = len(votes)
    return [votes[rng.randint(0, n - 1)] for _ in range(n)]


def generate_stability_report(
    votes: list[Vote],
    samples: list[ArtSample],
    n_bootstrap: int = 1000,
    convergence_window: int = 100,
    seed: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> StabilityReport:
    """Generate comprehensive stability report."""
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

    confidence_intervals = bootstrap_confidence_intervals(
        votes, samples, n_iterations=n_bootstrap, seed=seed, progress_callback=progress_callback
    )

    ranking_stability = calculate_ranking_stability(
        votes, samples, n_iterations=n_bootstrap, seed=seed
    )

    pairwise_significance = bradley_terry_significance(votes, samples)

    convergence = calculate_convergence(votes, samples, convergence_window)

    return _evaluate_publication_readiness(
        point_estimates, confidence_intervals, ranking_stability, pairwise_significance, convergence
    )


def _evaluate_publication_readiness(
    point_estimates: dict[str, float],
    confidence_intervals: dict[str, ConfidenceInterval],
    ranking_stability: dict[str, RankingStability],
    pairwise_significance: list[PairwiseSignificance],
    convergence: dict[str, ConvergenceMetrics],
) -> StabilityReport:
    """Evaluate whether ratings are ready for publication."""
    warnings: list[str] = []
    score_components: list[float] = []

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

    ci_widths = [ci.ci_width for ci in confidence_intervals.values()]
    if ci_widths:
        max_ci_width = max(ci_widths)
        if max_ci_width > 150:
            warnings.append(f"Max CI width is {max_ci_width:.0f}, above 150 threshold")
        if max_ci_width <= 0:
            score_components.append(25)
        else:
            score_components.append(min(150 / max_ci_width, 1.0) * 25)
    else:
        score_components.append(0)

    non_significant_adjacent = [ps for ps in pairwise_significance if not ps.is_significant]
    unclear_rankings = 0
    for ps in non_significant_adjacent:
        ci_a = confidence_intervals.get(ps.model_a)
        ci_b = confidence_intervals.get(ps.model_b)
        if ci_a and ci_b:
            overlaps = ci_a.ci_lower <= ci_b.ci_upper and ci_b.ci_lower <= ci_a.ci_upper
            if not overlaps:
                unclear_rankings += 1

    if unclear_rankings > 0:
        warnings.append(f"{unclear_rankings} adjacent model pair(s) have unclear ranking")
    sig_ratio = 1.0 - (unclear_rankings / max(len(pairwise_significance), 1))
    score_components.append(sig_ratio * 25)

    non_converged = [c for c in convergence.values() if not c.is_converged]
    if non_converged:
        warnings.append(
            f"{len(non_converged)} model(s) have not converged "
            f"({', '.join(c.model_id for c in non_converged[:3])})"
        )
    conv_ratio = 1.0 - (len(non_converged) / max(len(convergence), 1))
    score_components.append(conv_ratio * 25)

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
