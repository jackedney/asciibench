"""Leaderboard generation for model rankings.

This module generates formatted leaderboards based on calculated Elo
ratings, providing clear visual rankings of model performance.

Leaderboard Format:
    The leaderboard will display models ranked by Elo rating with the
    following information:
    - Rank position (1, 2, 3, ...)
    - Model name
    - Elo rating
    - Number of comparisons
    - Win rate (percentage)

Dependencies:
    - asciibench.analyst.elo: Elo rating calculation results
    - asciibench.analyst.stats: Consistency metrics calculation
"""

from datetime import datetime
from typing import TYPE_CHECKING

from asciibench.analyst.elo import calculate_elo_by_category
from asciibench.analyst.stats import calculate_consistency

if TYPE_CHECKING:
    from asciibench.common.models import ArtSample, Vote


def generate_leaderboard(
    votes: list["Vote"],
    samples: list["ArtSample"],
    elo_ratings: dict[str, float],
) -> str:
    """Generate a formatted leaderboard from Elo ratings and vote data.

    This function creates a human-readable leaderboard showing model
    rankings based on their calculated Elo ratings and vote history.

    Args:
        votes: A list of Vote objects for computing statistics.
        samples: A list of ArtSample objects for model ID lookup.
        elo_ratings: A dictionary mapping model IDs to Elo ratings.

    Returns:
        A formatted string representing the leaderboard with columns
        for rank, model name, rating, comparisons, and win rate.

    Example:
        >>> from uuid import uuid4
        >>> from asciibench.common.models import ArtSample, Vote
        >>> sample_a = ArtSample(id=uuid4(), model_id="model1", prompt_text="test",
        ...     category="single_object", attempt_number=1, raw_output="test",
        ...     sanitized_output="test", is_valid=True)
        >>> sample_b = ArtSample(id=uuid4(), model_id="model2", prompt_text="test",
        ...     category="single_object", attempt_number=1, raw_output="test",
        ...     sanitized_output="test", is_valid=True)
        >>> votes = [
        ...     Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
        ...     Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="B"),
        ... ]
        >>> ratings = {"model1": 1500.0, "model2": 1500.0}
        >>> board = generate_leaderboard(votes, [sample_a, sample_b], ratings)
        >>> "Rank" in board
        True
        >>> "model1" in board
        True
        >>> "model2" in board
        True

    Negative case:
        >>> generate_leaderboard([], [], {})
        '| Rank | Model | Elo Rating | Comparisons | Win Rate | Last Updated |\\n' \\
        '|------|-------|------------|-------------|----------|--------------|\\n' \\
        '| - | - | - | - | - | - |'
    """
    if not votes or not elo_ratings:
        placeholder = (
            "| Rank | Model | Elo Rating | Comparisons | Win Rate | Last Updated |\n"
            "|------|-------|------------|-------------|----------|--------------|\n"
            "| - | - | - | - | - | - |"
        )
        return placeholder

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _build_table(model_ids: list[str], title: str | None = None) -> str:
        rows = []
        ranked_models = sorted(
            [
                (model_id, elo_ratings[model_id])
                for model_id in model_ids
                if model_id in elo_ratings
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        for rank, (model_id, rating) in enumerate(ranked_models, start=1):
            metrics = calculate_consistency(votes, samples, model_id)
            win_rate_pct = metrics["win_rate"] * 100
            comparisons = int(metrics["comparisons"])
            row = (
                f"| {rank} | {model_id} | {int(rating)} | {comparisons} | "
                f"{win_rate_pct:.1f}% | {timestamp} |"
            )
            rows.append(row)

        if title:
            header = f"\n### {title}\n\n"
        else:
            header = ""
        table_header = "| Rank | Model | Elo Rating | Comparisons | Win Rate | Last Updated |\n"
        table_separator = "|------|-------|------------|-------------|----------|--------------|\n"
        return header + table_header + table_separator + "\n".join(rows)

    output = []

    output.append("# ASCIIBench Leaderboard\n")
    output.append("Model rankings based on Elo ratings from pairwise comparisons.\n")

    output.append("## Rankings\n")
    output.append(_build_table(list(elo_ratings.keys())))

    category_ratings = calculate_elo_by_category(votes, samples)
    for category in sorted(category_ratings.keys()):
        output.append("\n## Rankings by Category\n")
        output.append(f"### {category.replace('_', ' ').title()}\n")
        if category_ratings[category]:
            output.append(_build_table(list(category_ratings[category].keys()), None))
        else:
            output.append(
                "| Rank | Model | Elo Rating | Comparisons | Win Rate | Last Updated |\n"
                "|------|-------|------------|-------------|----------|--------------|\n"
                "| - | - | - | - | - | - |"
            )

    output.append("\n## Methodology\n")
    output.append(
        "This leaderboard uses the Elo rating system to rank models based on\n"
        "pairwise vote data from the Judge UI. Higher ratings indicate better\n"
        "performance in ASCII art generation tasks.\n"
    )

    output.append("### Elo Rating Details\n\n")
    output.append("- **Base Rating**: 1500\n")
    output.append("- **K-Factor**: 32\n")
    output.append("- **Minimum Comparisons**: 10 (to appear on leaderboard)\n\n")

    output.append("For more details on the Elo calculation approach, see\n")
    output.append("`asciibench/analyst/elo.py`.\n")

    return "\n".join(output)
