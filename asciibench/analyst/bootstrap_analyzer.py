"""Bootstrap confidence interval analyzer for Elo ratings.

This module provides BootstrapAnalyzer class for calculating
confidence intervals for Elo ratings using bootstrap resampling.

Dependencies:
    - asciibench.analyst.elo: calculate_elo function
    - asciibench.analyst.errors: StatisticalError, InsufficientDataError
"""

from __future__ import annotations

import random
from collections.abc import Callable

from asciibench.analyst.elo import calculate_elo
from asciibench.analyst.errors import InsufficientDataError
from asciibench.analyst.stability_models import ConfidenceInterval
from asciibench.common.models import ArtSample, Vote


class BootstrapAnalyzer:
    """Analyzer for calculating bootstrap confidence intervals.

    This class handles bootstrap resampling to estimate uncertainty
    in Elo ratings through confidence intervals.
    """

    MIN_ITERATIONS = 2
    DEFAULT_ITERATIONS = 1000
    DEFAULT_CONFIDENCE_LEVEL = 0.95

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the analyzer with an optional random seed.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = random.Random(seed)

    def calculate_ci(
        self,
        votes: list[Vote],
        samples: list[ArtSample],
        n_samples: int = DEFAULT_ITERATIONS,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, ConfidenceInterval]:
        """Calculate bootstrap confidence intervals for Elo ratings.

        Args:
            votes: List of Vote objects
            samples: List of ArtSample objects for model lookup
            n_samples: Number of bootstrap samples (default 1000)
            confidence_level: Confidence level (default 0.95 for 95% CI)
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Dict mapping model_id to ConfidenceInterval dataclass

        Raises:
            InsufficientDataError: If n_samples < MIN_ITERATIONS
            ValueError: If confidence_level is not in (0, 1]

        Example:
            >>> analyzer = BootstrapAnalyzer(seed=42)
            >>> ci_dict = analyzer.calculate_ci(votes, samples, n_samples=1000)
            >>> ci = ci_dict["model1"]
            >>> print(f"Rating: {ci.point_estimate} ± {ci.ci_width/2}")
        """
        self._validate_inputs(votes, n_samples, confidence_level)

        if not votes:
            return {}

        point_estimates = calculate_elo(votes, samples)
        if not point_estimates:
            return {}

        return self._bootstrap_ci(
            votes, samples, point_estimates, n_samples, confidence_level, progress_callback
        )

    def _validate_inputs(self, votes: list[Vote], n_samples: int, confidence_level: float) -> None:
        """Validate input parameters.

        Args:
            votes: List of Vote objects
            n_samples: Number of bootstrap samples
            confidence_level: Confidence level

        Raises:
            InsufficientDataError: If n_samples < MIN_ITERATIONS or no votes
            ValueError: If confidence_level is not in (0, 1]
        """
        if n_samples < self.MIN_ITERATIONS:
            msg = f"n_samples must be at least {self.MIN_ITERATIONS} for CI calculation"
            raise InsufficientDataError(msg, min_required=self.MIN_ITERATIONS)

        if not (0 < confidence_level <= 1.0):
            msg = "confidence_level must be in (0, 1]"
            raise ValueError(msg)

    def _bootstrap_ci(
        self,
        votes: list[Vote],
        samples: list[ArtSample],
        point_estimates: dict[str, float],
        n_samples: int,
        confidence_level: float,
        progress_callback: Callable[[int, int], None] | None,
    ) -> dict[str, ConfidenceInterval]:
        """Perform bootstrap resampling and calculate CIs.

        Args:
            votes: List of Vote objects
            samples: List of ArtSample objects
            point_estimates: Point estimates from Elo calculation
            n_samples: Number of bootstrap samples
            confidence_level: Confidence level
            progress_callback: Optional progress callback

        Returns:
            Dict mapping model_id to ConfidenceInterval
        """
        model_ids = list(point_estimates.keys())
        distributions: dict[str, list[float]] = {m: [] for m in model_ids}

        for i in range(n_samples):
            resampled = self._resample_votes(votes)
            bootstrap_ratings = calculate_elo(resampled, samples)

            for model_id in model_ids:
                rating = bootstrap_ratings.get(model_id, point_estimates[model_id])
                distributions[model_id].append(rating)

            if progress_callback:
                progress_callback(i + 1, n_samples)

        return self._calculate_cis(point_estimates, distributions, model_ids, confidence_level)

    def _resample_votes(self, votes: list[Vote]) -> list[Vote]:
        """Resample votes with replacement.

        Args:
            votes: List of Vote objects to resample

        Returns:
            Resampled list of votes (same length)
        """
        n = len(votes)
        return [votes[self.rng.randint(0, n - 1)] for _ in range(n)]

    def _calculate_cis(
        self,
        point_estimates: dict[str, float],
        distributions: dict[str, list[float]],
        model_ids: list[str],
        confidence_level: float,
    ) -> dict[str, ConfidenceInterval]:
        """Calculate confidence intervals from bootstrap distributions.

        Args:
            point_estimates: Point estimates for each model
            distributions: Bootstrap distributions for each model
            model_ids: List of model IDs
            confidence_level: Confidence level

        Returns:
            Dict mapping model_id to ConfidenceInterval
        """
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
