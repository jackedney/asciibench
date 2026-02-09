"""Convergence analyzer for Elo ratings.

This module provides ConvergenceAnalyzer class for tracking
Elo rating convergence over time.

Dependencies:
    - asciibench.analyst.elo: calculate_elo function
"""

from __future__ import annotations

from asciibench.analyst.elo import calculate_elo
from asciibench.analyst.stability_models import ConvergenceMetrics
from asciibench.common.models import ArtSample, Vote


class ConvergenceAnalyzer:
    """Analyzer for tracking rating convergence over time."""

    DEFAULT_WINDOW_SIZE = 100
    DEFAULT_CHANGE_THRESHOLD = 20.0

    def calculate_convergence(
        self,
        votes: list[Vote],
        samples: list[ArtSample],
        window_size: int = DEFAULT_WINDOW_SIZE,
        change_threshold: float = DEFAULT_CHANGE_THRESHOLD,
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

        sorted_votes = sorted(votes, key=lambda v: v.timestamp)
        checkpoint_interval = max(1, len(sorted_votes) // 50)
        history = self._calculate_rating_history(sorted_votes, samples, checkpoint_interval)

        if not history:
            return {}

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

            trend_slope = self._calculate_trend_slope(checkpoints)
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
        self,
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

        if n_votes % checkpoint_interval != 0:
            final_ratings = calculate_elo(votes, samples)
            for model_id, rating in final_ratings.items():
                if model_id not in history:
                    history[model_id] = []
                history[model_id].append((n_votes, rating))

        return history

    def _calculate_trend_slope(self, history: list[tuple[int, float]]) -> float:
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
