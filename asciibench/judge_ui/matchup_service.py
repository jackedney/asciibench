"""Matchup service for selecting samples for comparison.

This service encapsulates logic for selecting random matchups between
samples from different models, prioritizing pairs with fewer comparisons
to ensure balanced coverage.

Dependencies:
    - asciibench.common.models: ArtSample, Vote models
    - asciibench.common.persistence: JSONL persistence utilities
    - asciibench.judge_ui.selectors: ModelPairSelector, SampleSelector
"""

import random
from collections import Counter
from pathlib import Path

from asciibench.common.models import ArtSample, Vote
from asciibench.common.persistence import read_jsonl
from asciibench.common.repository import DataRepository
from asciibench.judge_ui.selectors import ModelPairSelector, SampleSelector


class MatchupService:
    """Service for selecting matchups between samples from different models.

    This class provides methods to select random matchups between samples,
    prioritizing model pairs that have had fewer comparisons.
    """

    def __init__(
        self,
        database_path: Path | None = None,
        votes_path: Path | None = None,
        model_pair_selector: ModelPairSelector | None = None,
        sample_selector: SampleSelector | None = None,
    ):
        """Initialize MatchupService with paths to data files and selectors.

        Args:
            database_path: Path to database.jsonl file (default: data/database.jsonl)
            votes_path: Path to votes.jsonl file (default: data/votes.jsonl)
            model_pair_selector: Optional ModelPairSelector instance (creates default if None)
            sample_selector: Optional SampleSelector instance (creates default if None)
        """
        self._database_path = database_path or Path("data/database.jsonl")
        self._votes_path = votes_path or Path("data/votes.jsonl")
        self._model_pair_selector = model_pair_selector or ModelPairSelector()
        self._sample_selector = sample_selector or SampleSelector()

    _make_sorted_pair = staticmethod(ModelPairSelector._default_make_sorted_pair)

    def _get_pair_comparison_counts(self, votes: list[Vote]) -> Counter[tuple[str, str]]:
        """Count comparisons for each ordered pair of samples.

        Returns a Counter where keys are (sample_a_id, sample_b_id) tuples
        and values are the number of times that pair has been compared.
        The pair is stored in sorted order to ensure (A, B) and (B, A) are treated the same.
        """
        counts: Counter[tuple[str, str]] = Counter()
        for vote in votes:
            pair = self._make_sorted_pair(vote.sample_a_id, vote.sample_b_id)
            counts[pair] += 1
        return counts

    def _get_model_pair_comparison_counts(
        self, votes: list[Vote], samples: list[ArtSample]
    ) -> Counter[tuple[str, str]]:
        """Count comparisons between each pair of models.

        Returns a Counter where keys are (model_a_id, model_b_id) tuples (sorted)
        and values are the number of comparisons between those models.
        """
        return self._model_pair_selector.get_model_pair_comparison_counts(votes, samples)

    def _calculate_total_possible_pairs(self, valid_samples: list[ArtSample]) -> int:
        """Calculate total possible unique matchups between samples from different models.

        A valid matchup requires two samples from different models. This counts
        all unique sample pairs where the samples have different model_ids.
        """
        total = 0
        n = len(valid_samples)
        for i in range(n):
            for j in range(i + 1, n):
                if valid_samples[i].model_id != valid_samples[j].model_id:
                    total += 1
        return total

    def _calculate_total_possible_pairs_for_samples(self, samples: list[ArtSample]) -> int:
        """Calculate total possible unique matchups for a list of samples.

        Helper that filters to valid samples and calculates pairs.
        """
        valid_samples = [s for s in samples if s.is_valid]
        return self._calculate_total_possible_pairs(valid_samples)

    def get_matchup(
        self, valid_samples: list[ArtSample] | None = None
    ) -> tuple[ArtSample, ArtSample]:
        """Select two samples for a matchup, prioritizing pairs with fewer comparisons.

        Samples must be from different models. The selection prioritizes model pairs
        that have had fewer comparisons to ensure balanced coverage.

        Args:
            valid_samples: Optional list of valid samples (loaded from database if None)

        Returns:
            Tuple of two ArtSample objects for comparison

        Raises:
            ValueError: If not enough valid samples for a matchup
        """
        if valid_samples is None:
            repo = DataRepository(data_dir=self._database_path.parent)
            if self._database_path == repo.database_path:
                valid_samples = repo.get_valid_samples_or_empty()
            else:
                try:
                    all_samples = read_jsonl(self._database_path, ArtSample)
                except FileNotFoundError:
                    all_samples = []
                valid_samples = [s for s in all_samples if s.is_valid]

        try:
            votes = read_jsonl(self._votes_path, Vote)
        except FileNotFoundError:
            votes = []

        return self._select_matchup(valid_samples, votes)

    def _select_matchup(
        self, valid_samples: list[ArtSample], votes: list[Vote]
    ) -> tuple[ArtSample, ArtSample]:
        """Select two samples for a matchup, prioritizing pairs with fewer comparisons.

        Samples must be from different models. The selection prioritizes model pairs
        that have had fewer comparisons to ensure balanced coverage.
        """
        samples_by_model: dict[str, list[ArtSample]] = {}
        for sample in valid_samples:
            if sample.model_id not in samples_by_model:
                samples_by_model[sample.model_id] = []
            samples_by_model[sample.model_id].append(sample)

        model_ids = list(samples_by_model.keys())
        if len(model_ids) < 2:
            return self._sample_selector.select_pair_from_single_model(valid_samples)

        least_compared_pairs = self._model_pair_selector.get_least_compared_pairs(
            model_ids, votes, valid_samples
        )

        model_a_id, model_b_id = random.choice(least_compared_pairs)

        return self._sample_selector.select_pair_from_models(
            model_a_id, model_b_id, samples_by_model
        )

    def get_unique_model_pairs_judged(
        self, votes: list[Vote] | None = None, samples: list[ArtSample] | None = None
    ) -> int:
        """Count unique model pairs that have been compared at least once.

        Returns the number of distinct (model_a, model_b) pairs found in votes
        where the two models are different.

        Args:
            votes: Optional list of votes (loaded from votes_path if None)
            samples: Optional list of samples (loaded from database_path if None)

        Returns:
            Count of unique model pairs judged
        """
        if votes is None:
            try:
                votes = read_jsonl(self._votes_path, Vote)
            except FileNotFoundError:
                votes = []
        if samples is None:
            repo = DataRepository(data_dir=self._database_path.parent)
            if self._database_path == repo.database_path:
                samples = repo.get_all_samples_or_empty()
            else:
                try:
                    samples = read_jsonl(self._database_path, ArtSample)
                except FileNotFoundError:
                    samples = []

        model_pair_counts = self._get_model_pair_comparison_counts(votes, samples)
        return len(model_pair_counts)
