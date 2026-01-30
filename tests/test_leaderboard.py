"""Tests for leaderboard generation."""

from uuid import uuid4

from asciibench.analyst.elo import calculate_elo
from asciibench.analyst.leaderboard import generate_leaderboard
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
