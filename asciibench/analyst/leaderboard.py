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

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def export_rankings_json(
    votes: list["Vote"],
    samples: list["ArtSample"],
    elo_ratings: dict[str, float],
    output_path: str | Path = "data/rankings.json",
) -> None:
    """Export rankings to JSON file.

    This function exports model ratings and metrics to a JSON file,
    including overall ratings, category-specific ratings, and
    consistency metrics.

    Args:
        votes: A list of Vote objects for computing statistics.
        samples: A list of ArtSample objects for model ID lookup.
        elo_ratings: A dictionary mapping model IDs to Elo ratings.
        output_path: Path to write the JSON file (default: data/rankings.json).

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
        ... ]
        >>> ratings = {"model1": 1516.0, "model2": 1484.0}
        >>> export_rankings_json(votes, [sample_a, sample_b], ratings, "/tmp/test.json")
        >>> import json
        >>> data = json.loads(Path("/tmp/test.json").read_text())
        >>> "overall_ratings" in data
        True
        >>> "category_ratings" in data
        True
        >>> "consistency_metrics" in data
        True

    Negative case:
        >>> export_rankings_json([], [], {}, "/tmp/test_empty.json")
        >>> data = json.loads(Path("/tmp/test_empty.json").read_text())
        >>> data["overall_ratings"] == {}
        True
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    category_ratings = calculate_elo_by_category(votes, samples)

    consistency_metrics: dict[str, dict[str, float]] = {}
    for model_id in elo_ratings:
        consistency_metrics[model_id] = calculate_consistency(votes, samples, model_id)

    export_data: dict[str, Any] = {
        "last_updated": datetime.now().isoformat(),
        "overall_ratings": {k: int(v) for k, v in sorted(elo_ratings.items())},
        "category_ratings": {},
        "consistency_metrics": {},
    }

    for category in sorted(category_ratings.keys()):
        export_data["category_ratings"][category] = {
            k: int(v) for k, v in sorted(category_ratings[category].items())
        }

    for model_id in sorted(consistency_metrics.keys()):
        export_data["consistency_metrics"][model_id] = {
            "win_rate": round(consistency_metrics[model_id]["win_rate"], 6),
            "std_dev": round(consistency_metrics[model_id]["std_dev"], 6),
            "comparisons": int(consistency_metrics[model_id]["comparisons"]),
        }

    output_path.write_text(json.dumps(export_data, indent=2))


def export_rankings_csv(
    votes: list["Vote"],
    samples: list["ArtSample"],
    elo_ratings: dict[str, float],
    output_path: str | Path = "data/rankings.csv",
) -> None:
    """Export rankings to CSV file.

    This function exports model ratings and metrics to a flat CSV file,
    with one row per model and columns for overall rating, category ratings,
    win rate, and comparisons.

    Args:
        votes: A list of Vote objects for computing statistics.
        samples: A list of ArtSample objects for model ID lookup.
        elo_ratings: A dictionary mapping model IDs to Elo ratings.
        output_path: Path to write the CSV file (default: data/rankings.csv).

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
        ... ]
        >>> ratings = {"model1": 1516.0, "model2": 1484.0}
        >>> export_rankings_csv(votes, [sample_a, sample_b], ratings, "/tmp/test.csv")
        >>> import csv
        >>> rows = list(csv.reader(Path("/tmp/test.csv").read_text().splitlines()))
        >>> len(rows)
        3
        >>> rows[0][0]
        'model'
        >>> rows[0][1]
        'overall_elo'

    Negative case:
        >>> export_rankings_csv([], [], {}, "/tmp/test_empty.csv")
        >>> rows = list(csv.reader(Path("/tmp/test_empty.csv").read_text().splitlines()))
        >>> len(rows)
        2
        >>> rows[0][0]
        'model'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    category_ratings = calculate_elo_by_category(votes, samples)

    all_categories = sorted(set(category_ratings.keys()))

    fieldnames = (
        ["model", "overall_elo"]
        + [f"{cat}_elo" for cat in all_categories]
        + ["win_rate", "comparisons"]
    )

    rows = []
    for model_id in sorted(elo_ratings.keys()):
        row = {
            "model": model_id,
            "overall_elo": int(elo_ratings[model_id]),
        }

        for category in all_categories:
            cat_ratings = category_ratings.get(category, {})
            row[f"{category}_elo"] = int(cat_ratings[model_id]) if model_id in cat_ratings else ""

        metrics = calculate_consistency(votes, samples, model_id)
        row["win_rate"] = round(metrics["win_rate"], 6)
        row["comparisons"] = int(metrics["comparisons"])

        rows.append(row)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
