"""Selectors for matchup selection logic.

This module provides composition-based selectors for model pair selection
and sample selection, allowing these concerns to be tested independently.
"""

import random
from collections import Counter
from collections.abc import Callable

from asciibench.common.models import ArtSample, Vote


class ModelPairSelector:
    """Selector for choosing model pairs based on comparison counts.

    This class encapsulates the logic for finding model pairs that have
    had fewer comparisons, promoting balanced coverage across models.
    """

    def __init__(
        self,
        make_sorted_pair: Callable[[str, str], tuple[str, str]] | None = None,
    ):
        """Initialize ModelPairSelector.

        Args:
            make_sorted_pair: Optional function to create sorted pairs.
                If not provided, uses default implementation.
        """
        self._make_sorted_pair = make_sorted_pair or self._default_make_sorted_pair

    @staticmethod
    def _default_make_sorted_pair(a: str, b: str) -> tuple[str, str]:
        """Create a sorted pair of strings for consistent ordering."""
        return (a, b) if a <= b else (b, a)

    def get_model_pair_comparison_counts(
        self, votes: list[Vote], samples: list[ArtSample]
    ) -> Counter[tuple[str, str]]:
        """Count comparisons between each pair of models.

        Returns a Counter where keys are (model_a_id, model_b_id) tuples (sorted)
        and values are the number of comparisons between those models.

        Args:
            votes: List of votes to analyze.
            samples: List of samples for model mapping.

        Returns:
            Counter mapping model pairs to comparison counts.
        """
        sample_to_model: dict[str, str] = {str(s.id): s.model_id for s in samples}

        counts: Counter[tuple[str, str]] = Counter()
        for vote in votes:
            model_a = sample_to_model.get(vote.sample_a_id)
            model_b = sample_to_model.get(vote.sample_b_id)
            if model_a and model_b and model_a != model_b:
                pair = self._make_sorted_pair(model_a, model_b)
                counts[pair] += 1
        return counts

    def get_least_compared_pairs(
        self, model_ids: list[str], votes: list[Vote], samples: list[ArtSample]
    ) -> list[tuple[str, str]]:
        """Get model pairs with the fewest comparisons.

        Args:
            model_ids: List of model IDs to consider.
            votes: List of votes to analyze.
            samples: List of samples for model mapping.

        Returns:
            List of model pairs (as tuples) with minimum comparison count.
        """
        model_pair_counts = self.get_model_pair_comparison_counts(votes, samples)

        model_pairs = [
            (model_ids[i], model_ids[j])
            for i in range(len(model_ids))
            for j in range(i + 1, len(model_ids))
        ]

        min_count = float("inf")
        for pair in model_pairs:
            sorted_pair = self._make_sorted_pair(pair[0], pair[1])
            count = model_pair_counts.get(sorted_pair, 0)
            if count < min_count:
                min_count = count

        least_compared_pairs = []
        for pair in model_pairs:
            sorted_pair = self._make_sorted_pair(pair[0], pair[1])
            if model_pair_counts.get(sorted_pair, 0) == min_count:
                least_compared_pairs.append(pair)

        return least_compared_pairs


class SampleSelector:
    """Selector for choosing samples from model pairs.

    This class encapsulates the logic for selecting samples from given models,
    handling edge cases like single-model scenarios.
    """

    def select_pair_from_models(
        self,
        model_a_id: str,
        model_b_id: str,
        samples_by_model: dict[str, list[ArtSample]],
    ) -> tuple[ArtSample, ArtSample]:
        """Select one sample from each of the two specified models.

        Args:
            model_a_id: First model ID.
            model_b_id: Second model ID.
            samples_by_model: Dictionary mapping model IDs to lists of samples.

        Returns:
            Tuple of two ArtSample objects (one from each model).

        Raises:
            ValueError: If either model has no samples.
        """
        samples_a = samples_by_model.get(model_a_id, [])
        samples_b = samples_by_model.get(model_b_id, [])

        if not samples_a:
            raise ValueError(f"No samples available for model: {model_a_id}")
        if not samples_b:
            raise ValueError(f"No samples available for model: {model_b_id}")

        sample_a = random.choice(samples_a)
        sample_b = random.choice(samples_b)

        return sample_a, sample_b

    def select_pair_from_single_model(
        self, samples: list[ArtSample]
    ) -> tuple[ArtSample, ArtSample]:
        """Select two samples when only one model is available.

        Args:
            samples: List of samples (all from the same model).

        Returns:
            Tuple of two ArtSample objects.

        Raises:
            ValueError: If fewer than 2 samples are provided.
        """
        if len(samples) < 2:
            raise ValueError("Not enough valid samples for a matchup")

        sample_a, sample_b = random.sample(samples, 2)
        return sample_a, sample_b
