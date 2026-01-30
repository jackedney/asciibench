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

from asciibench.analyst.elo import calculate_elo
from asciibench.analyst.leaderboard import (
    export_rankings_csv,
    export_rankings_json,
    generate_leaderboard,
)
from asciibench.common.models import ArtSample, Vote
from asciibench.common.persistence import read_jsonl


def main() -> None:
    """Main entry point for the Analyst module.

    This function coordinates the analysis of judge votes,
    calculation of Elo ratings, and generation of leaderboards.

    It performs the following steps:
    1. Load votes from data/votes.jsonl
    2. Load samples from data/database.jsonl for model_id lookup
    3. Calculate overall Elo ratings
    4. Calculate category-specific ratings
    5. Calculate consistency metrics for each model
    6. Generate and write LEADERBOARD.md
    7. Export rankings to JSON and CSV
    8. Print summary to stdout
    """
    print("ASCIIBench Analyst")
    print("=" * 50)

    votes_path = Path("data/votes.jsonl")
    database_path = Path("data/database.jsonl")

    print(f"\nLoading votes from {votes_path}...")
    votes = read_jsonl(votes_path, Vote)
    print(f"Loaded {len(votes)} votes")

    print(f"\nLoading samples from {database_path}...")
    samples = read_jsonl(database_path, ArtSample)
    print(f"Loaded {len(samples)} samples")

    if not votes:
        print("\nNo votes found. Generating empty leaderboard...")
        elo_ratings = {}
    else:
        print("\nCalculating Elo ratings...")
        elo_ratings = calculate_elo(votes, samples)
        print(f"Calculated ratings for {len(elo_ratings)} models")

        for model_id, rating in sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model_id}: {int(rating)}")

    print("\nGenerating leaderboard...")
    leaderboard = generate_leaderboard(votes, samples, elo_ratings)

    leaderboard_path = Path("LEADERBOARD.md")
    leaderboard_path.write_text(leaderboard)
    print(f"Leaderboard written to {leaderboard_path}")

    print("\nExporting rankings to JSON...")
    json_path = Path("data/rankings.json")
    export_rankings_json(votes, samples, elo_ratings, json_path)
    print(f"Rankings exported to {json_path}")

    print("\nExporting rankings to CSV...")
    csv_path = Path("data/rankings.csv")
    export_rankings_csv(votes, samples, elo_ratings, csv_path)
    print(f"Rankings exported to {csv_path}")

    print("\n" + "=" * 50)
    print("Analysis Complete!")
    print("=" * 50)

    print("\nSummary:")
    print(f"  Votes analyzed: {len(votes)}")
    print(f"  Samples used: {len(samples)}")
    print(f"  Models rated: {len(elo_ratings)}")
    print("\nOutput files:")
    print(f"  - {leaderboard_path}")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")


if __name__ == "__main__":
    main()
