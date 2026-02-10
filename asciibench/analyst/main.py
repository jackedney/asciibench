"""Analyst module main entry point.

This module provides the main entry point for the Analyst module,
which calculates Elo ratings, generates leaderboards, and computes
consistency metrics from judge votes.

Elo Calculation Approach:
    The Analyst will implement Elo rating calculations based on the
    standard Elo system adapted for multi-model comparisons. Key
    concepts include:

    - Base rating: All models start with a default Elo rating (e.g., 1500)
    - Expected score: Based on the rating difference between two models
    - K-factor: Determines how much ratings change after each comparison
    - Update rule: Winner gains points, loser loses points based on
      the difference between actual and expected outcomes

    For double-blind 1v1 comparisons:
    - Winner=A: Model A gains points, Model B loses points
    - Winner=B: Model B gains points, Model A loses points
    - Winner=tie: Ratings move closer together (small adjustment)
    - Winner=fail: No rating change (invalid comparison)

Dependencies:
    - asciibench.common.models: Vote model for judge submissions
    - asciibench.analyst.elo: Elo rating calculation logic
    - asciibench.analyst.leaderboard: Leaderboard generation
    - asciibench.analyst.stats: Consistency and statistical metrics
"""

from pathlib import Path

from rich.panel import Panel
from rich.text import Text

from asciibench.analyst.elo import calculate_elo, calculate_elo_by_category
from asciibench.analyst.leaderboard import (
    export_rankings_csv,
    export_rankings_json,
    generate_leaderboard,
)
from asciibench.analyst.stats import calculate_consistency
from asciibench.common.display import (
    create_leaderboard_table,
    get_console,
    print_banner,
    success_badge,
)
from asciibench.common.repository import DataRepository


def main() -> None:
    """Main entry point for the Analyst module.

    This function coordinates the analysis of judge votes,
    calculation of Elo ratings, and generation of leaderboards.

    It performs the following steps:
    1. Load votes and samples using DataRepository
    2. Calculate overall Elo ratings
    3. Calculate category-specific ratings
    4. Calculate consistency metrics for each model
    5. Generate and write LEADERBOARD.md
    6. Export rankings to JSON and CSV
    7. Print summary to stdout with Rich components
    """
    console = get_console()

    print_banner()
    console.print()

    repo = DataRepository()

    votes = repo.get_votes_or_empty()
    samples = repo.get_all_samples_or_empty()

    load_panel = Panel(
        Text.assemble(
            ("Votes loaded: ", "info"),
            (f"{len(votes)}", "bold accent"),
            (" | ", "default"),
            ("Samples loaded: ", "info"),
            (f"{len(samples)}", "bold accent"),
        ),
        title="Data Load",
        border_style="accent",
    )
    console.print(load_panel)
    console.print()

    if not votes:
        no_votes_panel = Panel(
            Text.assemble(
                ("No votes to analyze", "error"),
            ),
            title="Status",
            border_style="error",
        )
        console.print(no_votes_panel)
        console.print()
        elo_ratings = {}
    else:
        console.print(success_badge(), "[info]Calculating Elo ratings...[/info]")
        console.print()
        elo_ratings = calculate_elo(votes, samples)
        console.print(
            f"[info]Calculated ratings for [/info][bold accent]{len(elo_ratings)}[/bold "
            f"accent][info] models[/info]"
        )
        console.print()

    if elo_ratings:
        rankings_data = []
        for rank, (model_id, rating) in enumerate(
            sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True), start=1
        ):
            metrics = calculate_consistency(votes, samples, model_id)
            rankings_data.append(
                {
                    "rank": rank,
                    "model": model_id,
                    "elo": int(rating),
                    "comparisons": int(metrics["comparisons"]),
                    "win_rate": metrics["win_rate"],
                }
            )

        leaderboard_table = create_leaderboard_table(rankings_data)
        console.print(leaderboard_table)
        console.print()

        category_ratings = calculate_elo_by_category(votes, samples)
        if category_ratings:
            for category in sorted(category_ratings.keys()):
                cat_data = []
                cat_elo = category_ratings[category]
                if cat_elo:
                    for rank, (model_id, rating) in enumerate(
                        sorted(cat_elo.items(), key=lambda x: x[1], reverse=True), start=1
                    ):
                        metrics = calculate_consistency(
                            [
                                v
                                for v in votes
                                if any(
                                    s.category == category
                                    and str(s.id) in [v.sample_a_id, v.sample_b_id]
                                    for s in samples
                                )
                            ],
                            [s for s in samples if s.category == category],
                            model_id,
                        )
                        cat_data.append(
                            {
                                "rank": rank,
                                "model": model_id,
                                "elo": int(rating),
                                "comparisons": int(metrics["comparisons"]),
                                "win_rate": metrics["win_rate"],
                            }
                        )

                    cat_table = create_leaderboard_table(cat_data)
                    cat_table.title = f"Elo Leaderboard - {category.replace('_', ' ').title()}"
                    console.print(cat_table)
                    console.print()

    console.print(success_badge(), "Generating leaderboard markdown...", style="info")
    console.print()
    leaderboard = generate_leaderboard(votes, samples, elo_ratings)

    leaderboard_path = Path("LEADERBOARD.md")
    leaderboard_path.write_text(leaderboard)
    console.print(success_badge(), "LEADERBOARD.md written", style="success")
    console.print()

    console.print(success_badge(), "Exporting rankings to JSON...", style="info")
    console.print()
    json_path = Path("data/rankings.json")
    export_rankings_json(votes, samples, elo_ratings, json_path)
    console.print(success_badge(), "rankings.json written", style="success")
    console.print()

    console.print(success_badge(), "Exporting rankings to CSV...", style="info")
    console.print()
    csv_path = Path("data/rankings.csv")
    export_rankings_csv(votes, samples, elo_ratings, csv_path)
    console.print(success_badge(), "rankings.csv written", style="success")
    console.print()

    summary_text = Text.assemble(
        ("Votes analyzed: ", "info"),
        (f"{len(votes)}", "bold accent"),
        ("\n", "default"),
        ("Samples used: ", "info"),
        (f"{len(samples)}", "bold accent"),
        ("\n", "default"),
        ("Models rated: ", "info"),
        (f"{len(elo_ratings)}", "bold accent"),
        ("\n\n", "default"),
        ("Output files:\n", "bold"),
        (f"  - {leaderboard_path}\n", "default"),
        (f"  - {json_path}\n", "default"),
        (f"  - {csv_path}", "default"),
    )

    summary_panel = Panel(
        summary_text,
        title="Analysis Complete!",
        border_style="accent",
    )
    console.print(summary_panel)


if __name__ == "__main__":
    main()
