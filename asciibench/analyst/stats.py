"""Statistical metrics and consistency calculations.

This module provides statistical analysis functions for model
performance, including consistency metrics and win rate calculations.

Consistency Metrics:
    Consistency measures how stable a model's performance is across
    different prompts and comparisons. High consistency means a model
    performs reliably across various challenges.

    Metrics computed:
    - Win rate: Percentage of comparisons won by the model
    - Standard deviation: Variation in Elo ratings over time
    - Confidence interval: Statistical confidence in rating accuracy

Dependencies:
    - asciibench.common.models: Vote model for computing statistics
"""

from asciibench.common.models import ArtSample, Vote
from asciibench.common.repository import DataRepository


def calculate_consistency(
    votes: list[Vote],
    samples: list[ArtSample],
    model_id: str,
) -> dict[str, float]:
    """Calculate consistency metrics for a specific model.

    This function analyzes a model's vote history to compute consistency
    metrics including win rate, rating volatility, and other statistical
    measures.

    Args:
        votes: A list of Vote objects to analyze.
        samples: A list of ArtSample objects for looking up model IDs.
        model_id: The ID of the model to analyze.

    Returns:
        A dictionary containing consistency metrics:
        - win_rate: Percentage of comparisons won
        - std_dev: Standard deviation of performance
        - comparisons: Total number of comparisons

    Example:
        >>> from uuid import uuid4
        >>> sample_a = ArtSample(id=uuid4(), model_id="model1", prompt_text="test",
        ...     category="test", attempt_number=1, raw_output="test", sanitized_output="test",
        ...     is_valid=True)
        >>> sample_b = ArtSample(id=uuid4(), model_id="model2", prompt_text="test",
        ...     category="test", attempt_number=1, raw_output="test", sanitized_output="test",
        ...     is_valid=True)
        >>> votes = [
        ...     Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
        ...     Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
        ...     Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="B"),
        ... ]
        >>> metrics = calculate_consistency(votes, [sample_a, sample_b], "model1")
        >>> metrics["win_rate"]
        0.6666666666666666
        >>> metrics["comparisons"]
        3.0

    Negative case:
        >>> calculate_consistency([], [], "model1")
        {'win_rate': 0.0, 'std_dev': 0.0, 'comparisons': 0.0}
    """
    if not votes:
        return {"win_rate": 0.0, "std_dev": 0.0, "comparisons": 0.0}

    sample_to_model = DataRepository.build_sample_model_lookup(samples)

    outcomes: list[float] = []

    for vote in votes:
        if vote.winner == "fail":
            continue

        model_a = sample_to_model.get(vote.sample_a_id)
        model_b = sample_to_model.get(vote.sample_b_id)

        if not model_a or not model_b:
            continue

        if model_a != model_id and model_b != model_id:
            continue

        if model_a == model_id:
            if vote.winner == "A":
                outcomes.append(1.0)
            elif vote.winner == "B":
                outcomes.append(0.0)
            else:
                outcomes.append(0.5)
        else:
            if vote.winner == "A":
                outcomes.append(0.0)
            elif vote.winner == "B":
                outcomes.append(1.0)
            else:
                outcomes.append(0.5)

    if not outcomes:
        return {"win_rate": 0.0, "std_dev": 0.0, "comparisons": 0.0}

    comparisons = float(len(outcomes))
    win_rate = sum(outcomes) / comparisons

    if comparisons > 1:
        mean = win_rate
        variance = sum((x - mean) ** 2 for x in outcomes) / comparisons
        std_dev = variance**0.5
    else:
        std_dev = 0.0

    return {"win_rate": win_rate, "std_dev": std_dev, "comparisons": comparisons}
