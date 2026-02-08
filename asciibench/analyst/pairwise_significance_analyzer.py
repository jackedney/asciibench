"""Pairwise significance analyzer for Elo ratings.

This module provides PairwiseSignificanceAnalyzer class for calculating
statistical significance of pairwise model comparisons using
Bradley-Terry model.

Dependencies:
    - asciibench.analyst.elo: calculate_elo function
"""

from __future__ import annotations

import math

from asciibench.analyst.elo import calculate_elo
from asciibench.analyst.stability_models import PairwiseSignificance
from asciibench.common.models import ArtSample, Vote


class PairwiseSignificanceAnalyzer:
    """Analyzer for Bradley-Terry significance tests."""

    def calculate_significance(
        self, votes: list[Vote], samples: list[ArtSample]
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
                win_matrix[model_a][model_b] = win_matrix[model_a].get(model_b, 0.0) + 0.5
                win_matrix[model_b][model_a] = win_matrix[model_b].get(model_a, 0.0) + 0.5

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

            p_value = 1.0 if total == 0 else self._binomial_test_two_tailed(wins_a, int(total), 0.5)

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

    def _binomial_test_two_tailed(self, k: float, n: int, p: float = 0.5) -> float:
        """Two-tailed binomial test p-value."""
        if n == 0:
            return 1.0

        expected = n * p
        observed_deviation = abs(k - expected)

        p_value = 0.0
        for i in range(n + 1):
            if abs(i - expected) >= observed_deviation:
                p_value += self._binomial_pmf(i, n, p)

        return min(p_value, 1.0)

    def _binomial_pmf(self, k: int, n: int, p: float) -> float:
        """Binomial probability mass function: P(X = k)."""
        if k < 0 or k > n:
            return 0.0

        log_coef = self._log_binomial_coefficient(n, k)
        log_prob = log_coef + k * math.log(p) + (n - k) * math.log(1 - p)
        return math.exp(log_prob)

    def _log_binomial_coefficient(self, n: int, k: int) -> float:
        """Log of binomial coefficient C(n, k) = n! / (k! * (n-k)!)."""
        if k > n or k < 0:
            return float("-inf")
        if k == 0 or k == n:
            return 0.0

        return self._log_factorial(n) - self._log_factorial(k) - self._log_factorial(n - k)

    def _log_factorial(self, n: int) -> float:
        """Log factorial using Stirling's approximation for large n."""
        if n <= 1:
            return 0.0
        if n < 20:
            return sum(math.log(i) for i in range(2, n + 1))
        return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)
