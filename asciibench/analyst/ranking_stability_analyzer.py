"""Ranking stability analyzer for Elo ratings.

This module provides RankingStabilityAnalyzer class for measuring
how stable model rankings are across bootstrap resamples.

Dependencies:
    - asciibench.analyst.elo: calculate_elo function
    - asciibench.analyst.errors: StatisticalError, InsufficientDataError
"""

from __future__ import annotations

import random
from collections.abc import Callable

from asciibench.analyst.elo import calculate_elo
from asciibench.analyst.errors import InsufficientDataError
from asciibench.analyst.stability_models import RankingStability
from asciibench.common.models import ArtSample, Vote


class RankingStabilityAnalyzer:
    """Analyzer for calculating ranking stability across bootstrap samples.

    This class measures how often each model maintains its rank position
    across bootstrap resamples to assess ranking reliability.
    """

    MIN_ITERATIONS = 2
    DEFAULT_ITERATIONS = 1000

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the analyzer with an optional random seed.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = random.Random(seed)

    def calculate_stability(
        self,
        votes: list[Vote],
        samples: list[ArtSample],
        n_iterations: int = DEFAULT_ITERATIONS,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, RankingStability]:
        """Calculate ranking stability from bootstrap samples.

        Args:
            votes: List of Vote objects
            samples: List of ArtSample objects
            n_iterations: Number of bootstrap iterations
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Dict mapping model_id to RankingStability

        Raises:
            InsufficientDataError: If n_iterations < MIN_ITERATIONS

        Example:
            >>> analyzer = RankingStabilityAnalyzer(seed=42)
            >>> stability = analyzer.calculate_stability(
            ...     votes, samples, n_iterations=1000
            ... )
            >>> model1_stab = stability["model1"]
            >>> print(
            ...     f"Model1 rank {model1_stab.modal_rank} is stable "
            ...     f"{model1_stab.rank_stability_pct:.0%}"
            ... )
        """
        self._validate_inputs(votes, n_iterations)

        if not votes:
            return {}

        point_estimates = calculate_elo(votes, samples)
        if not point_estimates:
            return {}

        return self._bootstrap_stability(
            votes, samples, point_estimates, n_iterations, progress_callback
        )

    def _validate_inputs(self, votes: list[Vote], n_iterations: int) -> None:
        """Validate input parameters.

        Args:
            votes: List of Vote objects
            n_iterations: Number of bootstrap iterations

        Raises:
            InsufficientDataError: If n_iterations < MIN_ITERATIONS
        """
        if n_iterations < self.MIN_ITERATIONS:
            msg = f"n_iterations must be at least {self.MIN_ITERATIONS} for stability calculation"
            raise InsufficientDataError(msg, min_required=self.MIN_ITERATIONS)

    def _bootstrap_stability(
        self,
        votes: list[Vote],
        samples: list[ArtSample],
        point_estimates: dict[str, float],
        n_iterations: int,
        progress_callback: Callable[[int, int], None] | None,
    ) -> dict[str, RankingStability]:
        """Perform bootstrap resampling and calculate rank stability.

        Args:
            votes: List of Vote objects
            samples: List of ArtSample objects
            point_estimates: Point estimates from Elo calculation
            n_iterations: Number of bootstrap iterations
            progress_callback: Optional progress callback

        Returns:
            Dict mapping model_id to RankingStability
        """
        model_ids = list(point_estimates.keys())
        rank_counts: dict[str, dict[int, int]] = {m: {} for m in model_ids}

        for i in range(n_iterations):
            resampled = self._resample_votes(votes)
            bootstrap_ratings = calculate_elo(resampled, samples)

            for model_id in model_ids:
                if model_id not in bootstrap_ratings:
                    bootstrap_ratings[model_id] = point_estimates[model_id]

            ranks = self._ratings_to_ranks(bootstrap_ratings)

            for model_id, rank in ranks.items():
                if model_id in rank_counts:
                    rank_counts[model_id][rank] = rank_counts[model_id].get(rank, 0) + 1

            if progress_callback:
                progress_callback(i + 1, n_iterations)

        return self._calculate_stability_metrics(rank_counts, model_ids, n_iterations)

    def _resample_votes(self, votes: list[Vote]) -> list[Vote]:
        """Resample votes with replacement.

        Args:
            votes: List of Vote objects to resample

        Returns:
            Resampled list of votes (same length)
        """
        n = len(votes)
        return [votes[self.rng.randint(0, n - 1)] for _ in range(n)]

    def _ratings_to_ranks(self, ratings: dict[str, float]) -> dict[str, int]:
        """Convert rating dict to rank dict (1 = highest rating).

        Args:
            ratings: Dict mapping model_id to rating

        Returns:
            Dict mapping model_id to rank (1 = best)
        """
        sorted_models = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
        return {model_id: rank for rank, (model_id, _) in enumerate(sorted_models, 1)}

    def _calculate_stability_metrics(
        self,
        rank_counts: dict[str, dict[int, int]],
        model_ids: list[str],
        n_iterations: int,
    ) -> dict[str, RankingStability]:
        """Calculate stability metrics from rank count data.

        Args:
            rank_counts: Dict mapping model_id to rank count dict
            model_ids: List of model IDs
            n_iterations: Total number of bootstrap iterations

        Returns:
            Dict mapping model_id to RankingStability
        """
        results: dict[str, RankingStability] = {}
        for model_id in model_ids:
            counts = rank_counts[model_id]
            if not counts:
                continue

            modal_rank = max(counts.keys(), key=lambda r: counts[r])
            modal_count = counts[modal_rank]
            rank_stability_pct = modal_count / n_iterations

            rank_distribution = {r: c / n_iterations for r, c in counts.items()}

            results[model_id] = RankingStability(
                model_id=model_id,
                modal_rank=modal_rank,
                rank_stability_pct=rank_stability_pct,
                rank_distribution=rank_distribution,
            )

        return results
