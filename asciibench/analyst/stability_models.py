"""Data classes for stability analysis results."""

from dataclasses import dataclass, field


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
    modal_rank: int
    rank_stability_pct: float
    rank_distribution: dict[int, float] = field(default_factory=dict)


@dataclass
class PairwiseSignificance:
    """Bradley-Terry significance test for pairwise comparison."""

    model_a: str
    model_b: str
    wins_a: float
    wins_b: float
    p_value: float
    is_significant: bool


@dataclass
class ConvergenceMetrics:
    """Convergence tracking for rating stability over time."""

    model_id: str
    final_rating: float
    max_change_last_n: float
    trend_slope: float
    is_converged: bool


@dataclass
class StabilityReport:
    """Complete stability report for all models."""

    confidence_intervals: dict[str, ConfidenceInterval]
    ranking_stability: dict[str, RankingStability]
    pairwise_significance: list[PairwiseSignificance]
    convergence: dict[str, ConvergenceMetrics]

    is_stable_for_publication: bool
    stability_warnings: list[str]
    stability_score: float
