"""Swiss tournament pair selection logic.

This module provides a selector for Swiss tournament pair selection,
choosing model pairs based on Elo ratings to ensure informative comparisons.
"""

import random
from itertools import combinations


class SwissPairSelector:
    """Selector for Swiss tournament pair selection.

    This class implements the logic for selecting model pairs in a Swiss
    tournament format, mixing closest-Elo pairs with random pairs to
    ensure both competitive balance and exploration.
    """

    def select_pairs(
        self, model_ids: list[str], elo_ratings: dict[str, float], n: int
    ) -> list[tuple[str, str]]:
        """Select 2N model pairs for a tournament round.

        When Elo ratings are empty (round 1), all pairs are selected randomly.
        When Elo ratings are populated, N pairs are the closest by Elo rating
        and N pairs are selected randomly from the remaining options.

        Args:
            model_ids: List of model IDs to select pairs from.
            elo_ratings: Dictionary mapping model IDs to Elo ratings.
                Empty dict indicates round 1 (all random selection).
            n: Number of closest pairs to select (total pairs = 2*n).

        Returns:
            List of 2*n model pairs as tuples, normalized with sorted order.
            Returns fewer pairs if not enough unique pairs available.
            Returns empty list if fewer than 2 models provided.
        """
        if len(model_ids) < 2:
            return []

        all_pairs = list(combinations(model_ids, 2))

        max_pairs = len(all_pairs)
        requested_pairs = 2 * n

        if max_pairs <= requested_pairs:
            return all_pairs

        if not elo_ratings:
            return self._select_random_pairs(all_pairs, requested_pairs)

        closest_pairs = self._select_closest_elo_pairs(all_pairs, elo_ratings, n)
        remaining_pairs = [p for p in all_pairs if p not in closest_pairs]
        random_pairs = self._select_random_pairs(remaining_pairs, n)

        return closest_pairs + random_pairs

    def _select_random_pairs(
        self, all_pairs: list[tuple[str, str]], count: int
    ) -> list[tuple[str, str]]:
        """Select random pairs from the available pairs.

        Args:
            all_pairs: List of all possible model pairs.
            count: Number of pairs to select.

        Returns:
            List of randomly selected pairs.
        """
        available = min(count, len(all_pairs))
        return random.sample(all_pairs, available)

    def _select_closest_elo_pairs(
        self,
        all_pairs: list[tuple[str, str]],
        elo_ratings: dict[str, float],
        n: int,
    ) -> list[tuple[str, str]]:
        """Select N pairs with the smallest Elo difference.

        Args:
            all_pairs: List of all possible model pairs.
            elo_ratings: Dictionary mapping model IDs to Elo ratings.
            n: Number of closest pairs to select.

        Returns:
            List of N pairs with smallest Elo differences.
        """
        pairs_with_diffs = []
        for model_a, model_b in all_pairs:
            elo_a = elo_ratings.get(model_a, 1500.0)
            elo_b = elo_ratings.get(model_b, 1500.0)
            diff = abs(elo_a - elo_b)
            pairs_with_diffs.append((diff, (model_a, model_b)))

        pairs_with_diffs.sort(key=lambda x: x[0])
        available = min(n, len(pairs_with_diffs))
        return [pair for _, pair in pairs_with_diffs[:available]]
