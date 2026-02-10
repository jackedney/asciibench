"""Progress service for calculating judging progress statistics.

This service encapsulates logic for calculating progress statistics,
including total votes, unique model pairs judged, and category breakdowns.

Dependencies:
    - asciibench.common.repository: DataRepository for data access
    - asciibench.common.models: ArtSample, Vote models
    - asciibench.judge_ui.matchup_service: MatchupService for pair calculations
"""

from typing import TYPE_CHECKING

from asciibench.common.models import ArtSample, Vote
from asciibench.common.repository import DataRepository
from asciibench.judge_ui.api_models import CategoryProgress, ProgressResponse

if TYPE_CHECKING:
    from asciibench.judge_ui.matchup_service import MatchupService


class ProgressService:
    """Service for calculating progress statistics.

    This class provides methods to calculate progress statistics for the judging
    process, including overall statistics and category-based breakdowns.

    The service uses dependency injection for MatchupService to allow for
    flexible testing and mockability.
    """

    def __init__(
        self,
        repo: DataRepository,
        matchup_service: "MatchupService",
    ) -> None:
        """Initialize ProgressService.

        Args:
            repo: DataRepository instance for data access
            matchup_service: MatchupService instance for pair calculations
        """
        self._repo = repo
        self._matchup_service = matchup_service

    def get_progress(self) -> ProgressResponse:
        """Get progress statistics for judging.

        Returns:
            ProgressResponse with votes_completed, unique_pairs_judged,
            total_possible_pairs, and by_category breakdown.
        """
        votes = self._repo.get_votes_or_empty()
        all_samples = self._repo.get_all_samples_or_empty()
        valid_samples = [s for s in all_samples if s.is_valid]

        votes_completed = len(votes)
        unique_pairs_judged = self._matchup_service.get_unique_model_pairs_judged(
            votes, valid_samples
        )
        total_possible_pairs = self._matchup_service._calculate_total_possible_pairs(valid_samples)

        by_category = self._calculate_progress_by_category(votes, all_samples)

        return ProgressResponse(
            votes_completed=votes_completed,
            unique_pairs_judged=unique_pairs_judged,
            total_possible_pairs=total_possible_pairs,
            by_category=by_category,
        )

    def _calculate_progress_by_category(
        self, votes: list[Vote], samples: list[ArtSample]
    ) -> dict[str, CategoryProgress]:
        """Calculate progress statistics broken down by category.

        Args:
            votes: List of all votes
            samples: List of all samples (including invalid)

        Returns:
            Dictionary mapping category names to CategoryProgress objects
        """
        sample_lookup = DataRepository.build_sample_lookup(samples)

        samples_by_category: dict[str, list[ArtSample]] = {}
        for sample in samples:
            if sample.is_valid:
                if sample.category not in samples_by_category:
                    samples_by_category[sample.category] = []
                samples_by_category[sample.category].append(sample)

        votes_by_category: dict[str, list[Vote]] = {}
        for vote in votes:
            sample_a = sample_lookup.get(vote.sample_a_id)
            if sample_a and sample_a.is_valid:
                category = sample_a.category
                if category not in votes_by_category:
                    votes_by_category[category] = []
                votes_by_category[category].append(vote)

        result: dict[str, CategoryProgress] = {}
        for category, category_samples in samples_by_category.items():
            category_votes = votes_by_category.get(category, [])

            unique_pairs = self._matchup_service.get_unique_model_pairs_judged(
                category_votes, category_samples
            )

            total_pairs = self._matchup_service._calculate_total_possible_pairs(category_samples)

            result[category] = CategoryProgress(
                votes_completed=len(category_votes),
                unique_pairs_judged=unique_pairs,
                total_possible_pairs=total_pairs,
            )

        return result
