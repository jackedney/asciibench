"""Tests for leaderboard generation."""

import csv
import json
from pathlib import Path
from uuid import uuid4

from asciibench.analyst.elo import calculate_elo
from asciibench.analyst.leaderboard import (
    export_rankings_csv,
    export_rankings_json,
    generate_leaderboard,
)
from asciibench.common.models import ArtSample, Vote


class TestGenerateLeaderboard:
    """Tests for generate_leaderboard function."""

    def test_empty_votes_returns_placeholder_table(self):
        """Empty votes list returns table with placeholder row."""
        board = generate_leaderboard([], [], {})
        assert "| Rank | Model | Elo Rating | Comparisons | Win Rate | Last Updated |" in board
        assert "| - | - | - | - | - | - |" in board
        assert "model1" not in board

    def test_empty_ratings_returns_placeholder_table(self):
        """Empty ratings dict returns table with placeholder row."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="test",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="test",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        vote = Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A")

        board = generate_leaderboard([vote], [sample_a, sample_b], {})
        assert "| Rank | Model | Elo Rating | Comparisons | Win Rate | Last Updated |" in board
        assert "| - | - | - | - | - | - |" in board

    def test_generates_leaderboard_with_two_models(self):
        """Generates leaderboard with two models ranked by Elo."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="B"),
        ]

        ratings = calculate_elo(votes, [sample_a, sample_b])

        board = generate_leaderboard(votes, [sample_a, sample_b], ratings)

        assert "# ASCIIBench Leaderboard" in board
        assert "| Rank | Model | Elo Rating | Comparisons | Win Rate | Last Updated |" in board
        assert "| 1 | model1" in board
        assert "| 2 | model2" in board
        assert "model1" in board.split("| 2 |")[0]
        assert "model2" in board.split("| 2 |")[1]

    def test_sorts_models_by_elo_rating_descending(self):
        """Sorts models by Elo rating in descending order."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A")
            for _ in range(10)
        ]

        ratings = calculate_elo(votes, [sample_a, sample_b])

        board = generate_leaderboard(votes, [sample_a, sample_b], ratings)

        assert ratings["model1"] > ratings["model2"]
        model1_pos = board.index("| 1 | model1")
        model2_pos = board.index("| 2 | model2")
        assert model1_pos < model2_pos

    def test_includes_category_specific_rankings(self):
        """Includes category-specific ranking sections."""
        sample_a1 = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b1 = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_a2 = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_animal",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b2 = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_animal",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [
            Vote(sample_a_id=str(sample_a1.id), sample_b_id=str(sample_b1.id), winner="A"),
            Vote(sample_a_id=str(sample_a2.id), sample_b_id=str(sample_b2.id), winner="B"),
        ]

        ratings = calculate_elo(votes, [sample_a1, sample_b1, sample_a2, sample_b2])

        board = generate_leaderboard(votes, [sample_a1, sample_b1, sample_a2, sample_b2], ratings)

        assert "## Rankings by Category" in board
        assert "### Single Object" in board
        assert "### Single Animal" in board

    def test_includes_last_updated_timestamp(self):
        """Includes last updated timestamp in leaderboard."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A")]

        ratings = {"model1": 1520.0, "model2": 1480.0}

        board = generate_leaderboard(votes, [sample_a, sample_b], ratings)

        assert "| 1 | model1 | 1520 |" in board
        assert "| 2 | model2 | 1480 |" in board
        assert "| Last Updated |" in board

    def test_formats_elo_ratings_as_integers(self):
        """Formats Elo ratings as integers."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A")]

        ratings = {"model1": 1516.5, "model2": 1483.5}

        board = generate_leaderboard(votes, [sample_a, sample_b], ratings)

        assert "| 1 | model1 | 1517 |" in board or "| 1 | model1 | 1516 |" in board
        assert "| 2 | model2 | 1484 |" in board or "| 2 | model2 | 1483 |" in board
        assert ".5" not in board

    def test_formats_win_rate_as_percentage(self):
        """Formats win rate as percentage."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="B"),
        ]

        ratings = {"model1": 1500.0, "model2": 1500.0}

        board = generate_leaderboard(votes, [sample_a, sample_b], ratings)

        assert "66.7%" in board or "66.6%" in board
        assert "33.3%" in board or "33.4%" in board

    def test_includes_methodology_section(self):
        """Includes methodology section with Elo details."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A")]

        ratings = {"model1": 1500.0, "model2": 1500.0}

        board = generate_leaderboard(votes, [sample_a, sample_b], ratings)

        assert "## Methodology" in board
        assert "### Elo Rating Details" in board
        assert "**Base Rating**: 1500" in board
        assert "**K-Factor**: 32" in board
        assert "**Minimum Comparisons**: 10" in board
        assert "asciibench/analyst/elo.py" in board

    def test_four_categories_displayed(self):
        """All four categories are displayed in rankings."""
        categories = [
            "single_object",
            "single_animal",
            "animal_action",
            "spatial_relationship",
        ]

        samples = []
        for cat in categories:
            for model_id in ["model1", "model2"]:
                sample = ArtSample(
                    id=uuid4(),
                    model_id=model_id,
                    prompt_text="test",
                    category=cat,
                    attempt_number=1,
                    raw_output="test",
                    sanitized_output="test",
                    is_valid=True,
                )
                samples.append(sample)

        votes = []
        for i, _ in enumerate(categories):
            if i % 2 == 0:
                votes.append(
                    Vote(
                        sample_a_id=str(samples[i * 2].id),
                        sample_b_id=str(samples[i * 2 + 1].id),
                        winner="A",
                    )
                )
            else:
                votes.append(
                    Vote(
                        sample_a_id=str(samples[i * 2].id),
                        sample_b_id=str(samples[i * 2 + 1].id),
                        winner="B",
                    )
                )

        ratings = {"model1": 1500.0, "model2": 1500.0}

        board = generate_leaderboard(votes, samples, ratings)

        assert "### Single Object" in board
        assert "### Single Animal" in board
        assert "### Animal Action" in board
        assert "### Spatial Relationship" in board

    def test_tie_votes_count_half_in_win_rate(self):
        """Tie votes count as 0.5 wins in win rate calculation."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="tie"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="B"),
        ]

        ratings = {"model1": 1500.0, "model2": 1500.0}

        board = generate_leaderboard(votes, [sample_a, sample_b], ratings)

        assert "50.0%" in board

    def test_fail_votes_excluded_from_win_rate(self):
        """Fail votes are excluded from win rate calculation."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="fail"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="B"),
        ]

        ratings = {"model1": 1500.0, "model2": 1500.0}

        board = generate_leaderboard(votes, [sample_a, sample_b], ratings)

        assert "50.0%" in board

    def test_multiple_models_ranked_correctly(self):
        """Multiple models are ranked correctly by Elo."""
        models = ["model1", "model2", "model3"]
        samples = []

        for model_id in models:
            sample = ArtSample(
                id=uuid4(),
                model_id=model_id,
                prompt_text="test",
                category="single_object",
                attempt_number=1,
                raw_output="test",
                sanitized_output="test",
                is_valid=True,
            )
            samples.append(sample)

        votes = []
        for i in range(len(models) - 1):
            for j in range(i + 1, len(models)):
                for _ in range(3 if i == 0 else 1):
                    votes.append(
                        Vote(
                            sample_a_id=str(samples[i].id),
                            sample_b_id=str(samples[j].id),
                            winner="A" if i == 0 else "B",
                        )
                    )

        ratings = calculate_elo(votes, samples)

        board = generate_leaderboard(votes, samples, ratings)

        assert "| 1 | model1 |" in board
        assert "| 2 | model2" in board or "| 3 | model2" in board
        assert "| 2 | model3" in board or "| 3 | model3" in board
        model1_pos = board.index("| 1 | model1")
        model2_pos = (
            board.index("| 2 | model2") if "| 2 | model2" in board else board.index("| 3 | model2")
        )
        assert model1_pos < model2_pos

    def test_comparisons_count_displayed(self):
        """Comparisons count is displayed correctly."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A")
            for _ in range(10)
        ]

        ratings = calculate_elo(votes, [sample_a, sample_b])

        board = generate_leaderboard(votes, [sample_a, sample_b], ratings)

        assert "| 1 | model1 |" in board
        assert "| 2 | model2 |" in board

    def test_header_matches_format_in_leaderboard_md(self):
        """Leaderboard header matches format in existing LEADERBOARD.md."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A")]

        ratings = {"model1": 1500.0, "model2": 1500.0}

        board = generate_leaderboard(votes, [sample_a, sample_b], ratings)

        assert "# ASCIIBench Leaderboard" in board
        assert "Model rankings based on Elo ratings from pairwise comparisons." in board
        assert "## Rankings" in board
        assert "| Rank | Model | Elo Rating | Comparisons | Win Rate | Last Updated |" in board
        assert "|------|-------|------------|-------------|----------|--------------|" in board


class TestExportRankingsJson:
    """Tests for export_rankings_json function."""

    def test_empty_data_creates_valid_empty_structures(self):
        """Empty data produces valid JSON with empty structures."""
        export_rankings_json([], [], {}, "/tmp/test_empty.json")

        data = json.loads(Path("/tmp/test_empty.json").read_text())

        assert "overall_ratings" in data
        assert "category_ratings" in data
        assert "consistency_metrics" in data
        assert "last_updated" in data
        assert data["overall_ratings"] == {}
        assert data["category_ratings"] == {}
        assert data["consistency_metrics"] == {}

        Path("/tmp/test_empty.json").unlink()

    def test_json_export_includes_overall_ratings(self):
        """JSON export includes overall Elo ratings."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A")
            for _ in range(5)
        ]

        ratings = calculate_elo(votes, [sample_a, sample_b])
        export_rankings_json(votes, [sample_a, sample_b], ratings, "/tmp/test_overall.json")

        data = json.loads(Path("/tmp/test_overall.json").read_text())

        assert "overall_ratings" in data
        assert "model1" in data["overall_ratings"]
        assert "model2" in data["overall_ratings"]
        assert data["overall_ratings"]["model1"] > 1500
        assert data["overall_ratings"]["model2"] < 1500

        Path("/tmp/test_overall.json").unlink()

    def test_json_export_includes_category_ratings(self):
        """JSON export includes category-specific ratings."""
        sample_a1 = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b1 = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_a2 = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_animal",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b2 = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_animal",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [
            Vote(sample_a_id=str(sample_a1.id), sample_b_id=str(sample_b1.id), winner="A"),
            Vote(sample_a_id=str(sample_a2.id), sample_b_id=str(sample_b2.id), winner="B"),
        ]

        ratings = {"model1": 1500.0, "model2": 1500.0}
        export_rankings_json(
            votes, [sample_a1, sample_b1, sample_a2, sample_b2], ratings, "/tmp/test_category.json"
        )

        data = json.loads(Path("/tmp/test_category.json").read_text())

        assert "category_ratings" in data
        assert "single_object" in data["category_ratings"]
        assert "single_animal" in data["category_ratings"]

        Path("/tmp/test_category.json").unlink()

    def test_json_export_includes_consistency_metrics(self):
        """JSON export includes consistency metrics."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="B"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
        ]

        ratings = {"model1": 1500.0, "model2": 1500.0}
        export_rankings_json(votes, [sample_a, sample_b], ratings, "/tmp/test_metrics.json")

        data = json.loads(Path("/tmp/test_metrics.json").read_text())

        assert "consistency_metrics" in data
        assert "model1" in data["consistency_metrics"]
        assert "model2" in data["consistency_metrics"]
        assert "win_rate" in data["consistency_metrics"]["model1"]
        assert "std_dev" in data["consistency_metrics"]["model1"]
        assert "comparisons" in data["consistency_metrics"]["model1"]

        Path("/tmp/test_metrics.json").unlink()

    def test_json_export_creates_valid_json_file(self):
        """export_rankings_json creates a valid JSON file."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A")]

        ratings = {"model1": 1516.0, "model2": 1484.0}
        export_rankings_json(votes, [sample_a, sample_b], ratings, "/tmp/test_valid.json")

        json.loads(Path("/tmp/test_valid.json").read_text())

        Path("/tmp/test_valid.json").unlink()

    def test_json_export_includes_last_updated_timestamp(self):
        """JSON export includes last updated timestamp."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A")]

        ratings = {"model1": 1500.0, "model2": 1500.0}
        export_rankings_json(votes, [sample_a, sample_b], ratings, "/tmp/test_timestamp.json")

        data = json.loads(Path("/tmp/test_timestamp.json").read_text())

        assert "last_updated" in data
        assert isinstance(data["last_updated"], str)

        Path("/tmp/test_timestamp.json").unlink()


class TestExportRankingsCsv:
    """Tests for export_rankings_csv function."""

    def test_empty_data_creates_valid_empty_csv(self):
        """Empty data produces valid CSV with only header."""
        export_rankings_csv([], [], {}, "/tmp/test_empty.csv")

        rows = list(csv.reader(Path("/tmp/test_empty.csv").read_text().splitlines()))

        assert len(rows) == 1
        assert rows[0][0] == "model"
        assert rows[0][1] == "overall_elo"

        Path("/tmp/test_empty.csv").unlink()

    def test_csv_export_has_flat_table_structure(self):
        """CSV export has flat table with one row per model."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A")
            for _ in range(5)
        ]

        ratings = calculate_elo(votes, [sample_a, sample_b])
        export_rankings_csv(votes, [sample_a, sample_b], ratings, "/tmp/test_flat.csv")

        rows = list(csv.reader(Path("/tmp/test_flat.csv").read_text().splitlines()))

        assert len(rows) == 3
        assert rows[0][0] == "model"
        assert rows[1][0] == "model1" or rows[2][0] == "model1"
        assert rows[1][0] == "model2" or rows[2][0] == "model2"

        Path("/tmp/test_flat.csv").unlink()

    def test_csv_export_includes_overall_elo(self):
        """CSV export includes overall Elo rating column."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A")
            for _ in range(5)
        ]

        ratings = calculate_elo(votes, [sample_a, sample_b])
        export_rankings_csv(votes, [sample_a, sample_b], ratings, "/tmp/test_overall_elo.csv")

        rows = list(csv.reader(Path("/tmp/test_overall_elo.csv").read_text().splitlines()))

        assert rows[0][1] == "overall_elo"

        Path("/tmp/test_overall_elo.csv").unlink()

    def test_csv_export_includes_category_elo_columns(self):
        """CSV export includes category-specific Elo columns."""
        sample_a1 = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b1 = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_a2 = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_animal",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b2 = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_animal",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [
            Vote(sample_a_id=str(sample_a1.id), sample_b_id=str(sample_b1.id), winner="A"),
            Vote(sample_a_id=str(sample_a2.id), sample_b_id=str(sample_b2.id), winner="B"),
        ]

        ratings = {"model1": 1500.0, "model2": 1500.0}
        export_rankings_csv(
            votes, [sample_a1, sample_b1, sample_a2, sample_b2], ratings, "/tmp/test_cat_elo.csv"
        )

        rows = list(csv.reader(Path("/tmp/test_cat_elo.csv").read_text().splitlines()))

        assert "single_object_elo" in rows[0]
        assert "single_animal_elo" in rows[0]

        Path("/tmp/test_cat_elo.csv").unlink()

    def test_csv_export_includes_win_rate_and_comparisons(self):
        """CSV export includes win_rate and comparisons columns."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="B"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
        ]

        ratings = {"model1": 1500.0, "model2": 1500.0}
        export_rankings_csv(votes, [sample_a, sample_b], ratings, "/tmp/test_metrics.csv")

        rows = list(csv.reader(Path("/tmp/test_metrics.csv").read_text().splitlines()))

        assert "win_rate" in rows[0]
        assert "comparisons" in rows[0]

        Path("/tmp/test_metrics.csv").unlink()

    def test_csv_export_handles_missing_category_ratings(self):
        """CSV export handles models with no ratings in some categories."""
        sample_a1 = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b1 = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [Vote(sample_a_id=str(sample_a1.id), sample_b_id=str(sample_b1.id), winner="A")]

        ratings = {"model1": 1500.0, "model2": 1500.0}
        export_rankings_csv(votes, [sample_a1, sample_b1], ratings, "/tmp/test_missing_cat.csv")

        rows = list(csv.reader(Path("/tmp/test_missing_cat.csv").read_text().splitlines()))

        assert "single_object_elo" in rows[0]

        Path("/tmp/test_missing_cat.csv").unlink()

    def test_csv_export_creates_valid_csv_file(self):
        """export_rankings_csv creates a valid CSV file."""
        sample_a = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="single_object",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A")]

        ratings = {"model1": 1516.0, "model2": 1484.0}
        export_rankings_csv(votes, [sample_a, sample_b], ratings, "/tmp/test_valid.csv")

        rows = list(csv.reader(Path("/tmp/test_valid.csv").read_text().splitlines()))

        assert len(rows) >= 2
        assert rows[0][0] == "model"

        Path("/tmp/test_valid.csv").unlink()
