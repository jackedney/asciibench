"""Analytics service for calculating leaderboard and VLM statistics.

This module provides AnalyticsService class that encapsulates all analytics
calculations including Elo ratings, stability metrics, head-to-head records,
and VLM accuracy statistics.

The service uses DataRepository for data access through dependency injection,
making it easy to test and mock.
"""

import statistics

from asciibench.analyst.elo import calculate_elo
from asciibench.analyst.stability import generate_stability_report
from asciibench.common.models import ArtSample, VLMEvaluation, Vote
from asciibench.judge_ui.api_models import (
    AnalyticsResponse,
    CategoryAccuracyStats,
    ConfidenceIntervalData,
    EloHistoryPoint,
    HeadToHeadRecord,
    LeaderboardEntry,
    ModelAccuracyStats,
    ModelStabilityData,
    StabilityData,
    VLMAccuracyResponse,
)
from asciibench.common.repository import DataRepository


class AnalyticsService:
    """Service for calculating analytics data from votes and samples.

    This service provides methods for computing Elo ratings, stability metrics,
    head-to-head comparisons, and VLM accuracy statistics. All calculations
    are based on data retrieved through the DataRepository.

    Args:
        repo: DataRepository instance for accessing votes, samples, and evaluations.

    Example:
        >>> repo = DataRepository()
        >>> service = AnalyticsService(repo)
        >>> analytics = service.get_analytics_data()
        >>> print(f"Total votes: {analytics.total_votes}")
    """

    def __init__(self, repo: DataRepository) -> None:
        """Initialize the AnalyticsService.

        Args:
            repo: DataRepository instance for data access.
        """
        self._repo = repo

    def get_analytics_data(self, n_bootstrap: int = 200) -> AnalyticsResponse:
        """Build complete analytics data from votes and samples.

        Args:
            n_bootstrap: Number of bootstrap iterations for stability analysis.

        Returns:
            AnalyticsResponse with leaderboard, stability, elo_history, and head_to_head.

        Example:
            >>> repo = DataRepository()
            >>> service = AnalyticsService(repo)
            >>> analytics = service.get_analytics_data(n_bootstrap=100)
            >>> print(f"Leaderboard has {len(analytics.leaderboard)} entries")
        """
        try:
            votes = self._repo.get_votes()
        except FileNotFoundError:
            votes = []

        try:
            samples = self._repo.get_valid_samples()
        except FileNotFoundError:
            samples = []

        if not votes:
            return AnalyticsResponse(
                leaderboard=[],
                stability=StabilityData(
                    score=0.0,
                    is_stable=False,
                    warnings=["No votes to analyze"],
                    models={},
                ),
                elo_history={},
                head_to_head={},
                total_votes=0,
            )

        return self._build_analytics_data(votes, samples, n_bootstrap)

    def get_vlm_accuracy_data(self) -> VLMAccuracyResponse:
        """Get VLM accuracy statistics per model and category.

        Returns:
            VLMAccuracyResponse with by_model and by_category accuracy stats.

        Example:
            >>> repo = DataRepository()
            >>> service = AnalyticsService(repo)
            >>> vlm_data = service.get_vlm_accuracy_data()
            >>> print(f"Accuracy data for {len(vlm_data.by_model)} models")
        """
        try:
            evaluations = self._repo.get_evaluations()
        except FileNotFoundError:
            evaluations = []

        try:
            samples = self._repo.get_all_samples()
        except FileNotFoundError:
            samples = []

        if not evaluations:
            return VLMAccuracyResponse(by_model={}, by_category={})

        by_model = self._calculate_vlm_accuracy(evaluations, samples)
        by_category = self._calculate_category_accuracy(evaluations, samples)

        return VLMAccuracyResponse(by_model=by_model, by_category=by_category)

    def _calculate_elo_history(
        self,
        votes: list[Vote],
        samples: list[ArtSample],
        checkpoint_interval: int = 10,
    ) -> dict[str, list[EloHistoryPoint]]:
        """Calculate ELO rating history at checkpoints during vote processing."""
        if not votes:
            return {}

        sorted_votes = sorted(votes, key=lambda v: v.timestamp)
        history: dict[str, list[EloHistoryPoint]] = {}
        n_votes = len(sorted_votes)

        for i in range(checkpoint_interval, n_votes + 1, checkpoint_interval):
            partial_votes = sorted_votes[:i]
            ratings = calculate_elo(partial_votes, samples)

            for model_id, rating in ratings.items():
                if model_id not in history:
                    history[model_id] = []
                history[model_id].append(EloHistoryPoint(vote_count=i, elo=rating))

        if n_votes % checkpoint_interval != 0:
            final_ratings = calculate_elo(sorted_votes, samples)
            for model_id, rating in final_ratings.items():
                if model_id not in history:
                    history[model_id] = []
                history[model_id].append(EloHistoryPoint(vote_count=n_votes, elo=rating))

        return history

    def _calculate_head_to_head(
        self,
        votes: list[Vote],
        samples: list[ArtSample],
    ) -> dict[str, dict[str, HeadToHeadRecord]]:
        """Calculate head-to-head win/loss records between all model pairs."""
        sample_to_model: dict[str, str] = {str(s.id): s.model_id for s in samples}

        win_counts: dict[str, dict[str, int]] = {}

        for vote in votes:
            if vote.winner == "fail":
                continue

            model_a = sample_to_model.get(vote.sample_a_id)
            model_b = sample_to_model.get(vote.sample_b_id)

            if not model_a or not model_b or model_a == model_b:
                continue

            if model_a not in win_counts:
                win_counts[model_a] = {}
            if model_b not in win_counts:
                win_counts[model_b] = {}

            if vote.winner == "A":
                win_counts[model_a][model_b] = win_counts[model_a].get(model_b, 0) + 1
            elif vote.winner == "B":
                win_counts[model_b][model_a] = win_counts[model_b].get(model_a, 0) + 1

        result: dict[str, dict[str, HeadToHeadRecord]] = {}
        all_models = set(win_counts.keys())

        for model_a in all_models:
            result[model_a] = {}
            for model_b in all_models:
                if model_a == model_b:
                    continue
                wins = win_counts.get(model_a, {}).get(model_b, 0)
                losses = win_counts.get(model_b, {}).get(model_a, 0)
                result[model_a][model_b] = HeadToHeadRecord(wins=wins, losses=losses)

        return result

    def _build_analytics_data(
        self,
        votes: list[Vote],
        samples: list[ArtSample],
        n_bootstrap: int = 200,
    ) -> AnalyticsResponse:
        """Build complete analytics data from votes and samples."""
        elo_ratings = calculate_elo(votes, samples)

        sorted_ratings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
        leaderboard = [
            LeaderboardEntry(rank=i + 1, model_id=model_id, elo=round(elo, 1))
            for i, (model_id, elo) in enumerate(sorted_ratings)
        ]

        stability_report = generate_stability_report(
            votes, samples, n_bootstrap=n_bootstrap, seed=42
        )

        models_stability: dict[str, ModelStabilityData] = {}
        for model_id, elo in elo_ratings.items():
            ci = stability_report.confidence_intervals.get(model_id)
            rs = stability_report.ranking_stability.get(model_id)
            conv = stability_report.convergence.get(model_id)

            models_stability[model_id] = ModelStabilityData(
                elo=round(elo, 1),
                confidence_interval=ConfidenceIntervalData(
                    ci_lower=round(ci.ci_lower, 1),
                    ci_upper=round(ci.ci_upper, 1),
                    ci_width=round(ci.ci_width, 1),
                )
                if ci
                else None,
                rank_stability_pct=round(rs.rank_stability_pct * 100, 1) if rs else None,
                is_converged=conv.is_converged if conv else None,
            )

        stability = StabilityData(
            score=round(stability_report.stability_score, 1),
            is_stable=stability_report.is_stable_for_publication,
            warnings=stability_report.stability_warnings,
            models=models_stability,
        )

        elo_history = self._calculate_elo_history(votes, samples)
        head_to_head = self._calculate_head_to_head(votes, samples)

        return AnalyticsResponse(
            leaderboard=leaderboard,
            stability=stability,
            elo_history=elo_history,
            head_to_head=head_to_head,
            total_votes=len(votes),
        )

    def _calculate_vlm_accuracy(
        self,
        evaluations: list[VLMEvaluation],
        samples: list[ArtSample],
    ) -> dict[str, ModelAccuracyStats]:
        """Calculate VLM accuracy statistics per ASCII-generating model.

        Args:
            evaluations: List of VLM evaluations
            samples: List of all samples from database

        Returns:
            Dictionary mapping model_id to accuracy statistics
        """
        sample_lookup: dict[str, ArtSample] = {str(s.id): s for s in samples}

        by_model: dict[str, dict[str, int]] = {}

        for evaluation in evaluations:
            sample = sample_lookup.get(evaluation.sample_id)
            if not sample:
                continue

            model_id = sample.model_id
            if model_id not in by_model:
                by_model[model_id] = {"total": 0, "correct": 0}

            by_model[model_id]["total"] += 1
            if evaluation.is_correct:
                by_model[model_id]["correct"] += 1

        result: dict[str, ModelAccuracyStats] = {}
        for model_id, stats in by_model.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            result[model_id] = ModelAccuracyStats(
                total=stats["total"],
                correct=stats["correct"],
                accuracy=round(accuracy, 2),
            )

        return result

    def _calculate_category_accuracy(
        self,
        evaluations: list[VLMEvaluation],
        samples: list[ArtSample],
    ) -> dict[str, CategoryAccuracyStats]:
        """Calculate VLM accuracy statistics per category.

        Args:
            evaluations: List of VLM evaluations
            samples: List of all samples from database

        Returns:
            Dictionary mapping category to accuracy statistics
        """
        sample_lookup: dict[str, ArtSample] = {str(s.id): s for s in samples}

        by_category: dict[str, dict[str, int]] = {}

        for evaluation in evaluations:
            sample = sample_lookup.get(evaluation.sample_id)
            if not sample:
                continue

            category = sample.category
            if category not in by_category:
                by_category[category] = {"total": 0, "correct": 0}

            by_category[category]["total"] += 1
            if evaluation.is_correct:
                by_category[category]["correct"] += 1

        result: dict[str, CategoryAccuracyStats] = {}
        for category, stats in by_category.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            result[category] = CategoryAccuracyStats(
                total=stats["total"],
                correct=stats["correct"],
                accuracy=round(accuracy, 2),
            )

        return result

    def calculate_pearson_correlation(self, x: list[float], y: list[float]) -> float | None:
        """Calculate Pearson correlation coefficient.

        Args:
            x: First list of values
            y: Second list of values

        Returns:
            Pearson correlation coefficient, or None if fewer than 3 data points
        """
        return self._calculate_pearson_correlation(x, y)

    def _calculate_pearson_correlation(self, x: list[float], y: list[float]) -> float | None:
        """Calculate Pearson correlation coefficient.

        Args:
            x: First list of values
            y: Second list of values

        Returns:
            Pearson correlation coefficient, or None if fewer than 3 data points
        """
        if len(x) < 3 or len(y) < 3:
            return None

        n = len(x)

        if n != len(y):
            raise ValueError("Lists must have the same length")

        if n < 2:
            return None

        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True))

        sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
        sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)

        denominator = (sum_sq_x * sum_sq_y) ** 0.5

        if denominator < 1e-10:
            return None

        return numerator / denominator

    def get_vlm_accuracy_by_model(self) -> dict[str, ModelAccuracyStats]:
        """Get VLM accuracy statistics by model.

        Returns:
            Dictionary mapping model_id to accuracy statistics
        """
        try:
            evaluations = self._repo.get_evaluations()
        except FileNotFoundError:
            evaluations = []

        try:
            samples = self._repo.get_all_samples()
        except FileNotFoundError:
            samples = []

        return self._calculate_vlm_accuracy(evaluations, samples)
