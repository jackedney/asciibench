"""Elo rating calculations for model comparisons.

This module implements the Elo rating system for comparing LLM models
based on pairwise vote data from the Judge UI module.

Elo Rating System:
    The Elo rating system is a method for calculating the relative skill
    levels of players in competitive games. In the context of ASCII art
    generation, we treat each model as a "player" and use vote outcomes
    as "game results."

    Key formulas:
    - Expected score: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    - Rating change: R_A' = R_A + K * (S_A - E_A)

    Where:
    - R_A, R_B: Current ratings of model A and model B
    - E_A: Expected score for model A (probability of winning)
    - S_A: Actual score (1 for win, 0.5 for tie, 0 for loss)
    - K: K-factor controlling rating volatility (typically 10-32)

Dependencies:
    - asciibench.common.models: Vote, ArtSample models
"""

from asciibench.common.models import ArtSample, Vote
from asciibench.common.repository import DataRepository

BASE_RATING = 1500
K_FACTOR = 32


def calculate_elo(votes: list[Vote], samples: list[ArtSample]) -> dict[str, float]:
    """Calculate Elo ratings from a list of votes.

    This function processes judge votes to compute Elo ratings for each
    model. The algorithm iterates through votes and updates model ratings
    based on actual outcomes versus expected outcomes.

    Args:
        votes: A list of Vote objects representing pairwise comparisons.
        samples: A list of ArtSample objects for looking up model IDs.

    Returns:
        A dictionary mapping model IDs to their calculated Elo ratings.

    Example:
        >>> from uuid import uuid4
        >>> from asciibench.common.models import ArtSample
        >>> sample_a = ArtSample(id=uuid4(), model_id="model1", prompt_text="test",
        ...     category="test", attempt_number=1, raw_output="test", sanitized_output="test",
        ...     is_valid=True)
        >>> sample_b = ArtSample(id=uuid4(), model_id="model2", prompt_text="test",
        ...     category="test", attempt_number=1, raw_output="test", sanitized_output="test",
        ...     is_valid=True)
        >>> votes = [
        ...     Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
        ... ]
        >>> ratings = calculate_elo(votes, [sample_a, sample_b])
        >>> ratings["model1"] > 1500
        True
        >>> ratings["model2"] < 1500
        True

    Negative case:
        >>> calculate_elo([], [])
        {}
    """
    if not votes:
        return {}

    sample_to_model = DataRepository.build_sample_model_lookup(samples)

    votes_sorted = sorted(votes, key=lambda v: v.timestamp)

    ratings: dict[str, float] = {}

    for vote in votes_sorted:
        model_a = sample_to_model.get(vote.sample_a_id)
        model_b = sample_to_model.get(vote.sample_b_id)

        if not model_a or not model_b:
            continue

        if model_a == model_b:
            continue

        if model_a not in ratings:
            ratings[model_a] = float(BASE_RATING)
        if model_b not in ratings:
            ratings[model_b] = float(BASE_RATING)

        rating_a = ratings[model_a]
        rating_b = ratings[model_b]

        if vote.winner == "fail":
            continue

        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))

        if vote.winner == "A":
            score_a = 1.0
            score_b = 0.0
        elif vote.winner == "B":
            score_a = 0.0
            score_b = 1.0
        else:
            score_a = 0.5
            score_b = 0.5

        ratings[model_a] = rating_a + K_FACTOR * (score_a - expected_a)
        ratings[model_b] = rating_b + K_FACTOR * (score_b - expected_b)

    return ratings


def calculate_elo_by_category(
    votes: list[Vote], samples: list[ArtSample]
) -> dict[str, dict[str, float]]:
    """Calculate Elo ratings for each category separately.

    This function computes Elo ratings grouped by sample category, allowing
    analysts to identify which models excel at specific tasks (e.g., single
    objects vs. animals vs. spatial relationships).

    Args:
        votes: A list of Vote objects representing pairwise comparisons.
        samples: A list of ArtSample objects for looking up model IDs and
                 categories.

    Returns:
        A dictionary mapping category names to nested dictionaries of model
        ratings. Categories with no votes will have empty model dicts.

        Example:
            >>> ratings = calculate_elo_by_category(votes, samples)
            >>> ratings["single_object"]["gpt4o"]
            1520.0

    Negative case:
        >>> calculate_elo_by_category([], [])
        {}
    """
    if not votes:
        return {}

    sample_to_category: dict[str, str] = {str(s.id): s.category for s in samples}

    categories = sorted(set(sample_to_category.values()))
    result: dict[str, dict[str, float]] = {}

    for category in categories:
        category_votes = [
            vote
            for vote in votes
            if sample_to_category.get(vote.sample_a_id) == category
            and sample_to_category.get(vote.sample_b_id) == category
        ]

        result[category] = calculate_elo(category_votes, samples)

    return result
