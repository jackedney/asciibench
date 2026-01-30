"""Tests for statistical metrics calculations."""

from uuid import uuid4

from asciibench.analyst.stats import calculate_consistency
from asciibench.common.models import ArtSample, Vote


class TestCalculateConsistency:
    """Tests for calculate_consistency function."""

    def test_empty_votes_returns_zero_metrics(self):
        """Empty votes list returns zero metrics."""
        metrics = calculate_consistency([], [], "model1")
        assert metrics == {"win_rate": 0.0, "std_dev": 0.0, "comparisons": 0.0}

    def test_model_with_no_votes_returns_zero_metrics(self):
        """Model with no votes returns zero metrics."""
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

        metrics = calculate_consistency([vote], [sample_a, sample_b], "model3")
        assert metrics == {"win_rate": 0.0, "std_dev": 0.0, "comparisons": 0.0}

    def test_eight_wins_two_losses(self):
        """Model with 8 wins, 2 losses has win_rate=0.8."""
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

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A")
            for _ in range(8)
        ] + [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="B")
            for _ in range(2)
        ]

        metrics = calculate_consistency(votes, [sample_a, sample_b], "model1")
        assert metrics["win_rate"] == 0.8
        assert metrics["comparisons"] == 10.0
        assert metrics["std_dev"] > 0

    def test_tie_counts_as_half_win(self):
        """Tie votes count as 0.5 wins for win rate calculation."""
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

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="tie"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="B"),
        ]

        metrics = calculate_consistency(votes, [sample_a, sample_b], "model1")
        assert metrics["win_rate"] == 0.5
        assert metrics["comparisons"] == 3.0

    def test_fail_votes_excluded_from_calculations(self):
        """Fail votes are excluded from calculations."""
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

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="fail"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="B"),
        ]

        metrics = calculate_consistency(votes, [sample_a, sample_b], "model1")
        assert metrics["comparisons"] == 2.0
        assert metrics["win_rate"] == 0.5

    def test_model_b_as_winning_model(self):
        """Calculates correctly when model is in position B."""
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

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="B"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="B"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
        ]

        metrics = calculate_consistency(votes, [sample_a, sample_b], "model2")
        assert metrics["win_rate"] == 0.6666666666666666
        assert metrics["comparisons"] == 3.0

    def test_std_dev_with_single_comparison(self):
        """Standard deviation is 0 with single comparison."""
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

        metrics = calculate_consistency([vote], [sample_a, sample_b], "model1")
        assert metrics["std_dev"] == 0.0
        assert metrics["win_rate"] == 1.0
        assert metrics["comparisons"] == 1.0

    def test_std_dev_with_varying_outcomes(self):
        """Standard deviation is calculated correctly with varying outcomes."""
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

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="B"),
        ]

        metrics = calculate_consistency(votes, [sample_a, sample_b], "model1")
        assert metrics["std_dev"] > 0
        assert metrics["win_rate"] == 0.5

    def test_only_fail_votes_returns_zero_metrics(self):
        """Only fail votes returns zero metrics."""
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

        votes = [
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="fail"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="fail"),
        ]

        metrics = calculate_consistency(votes, [sample_a, sample_b], "model1")
        assert metrics == {"win_rate": 0.0, "std_dev": 0.0, "comparisons": 0.0}

    def test_missing_sample_id_skipped(self):
        """Votes with missing sample IDs are skipped."""
        vote = Vote(sample_a_id=str(uuid4()), sample_b_id=str(uuid4()), winner="A")

        metrics = calculate_consistency([vote], [], "model1")
        assert metrics == {"win_rate": 0.0, "std_dev": 0.0, "comparisons": 0.0}

    def test_multiple_models_correctly_filtered(self):
        """Correctly filters votes for specific model when multiple models exist."""
        model1_sample = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="test",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        model2_sample = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="test",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        model3_sample = ArtSample(
            id=uuid4(),
            model_id="model3",
            prompt_text="test",
            category="test",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        votes = [
            Vote(sample_a_id=str(model1_sample.id), sample_b_id=str(model2_sample.id), winner="A"),
            Vote(sample_a_id=str(model1_sample.id), sample_b_id=str(model3_sample.id), winner="A"),
            Vote(sample_a_id=str(model2_sample.id), sample_b_id=str(model3_sample.id), winner="A"),
        ]

        model1_metrics = calculate_consistency(
            votes, [model1_sample, model2_sample, model3_sample], "model1"
        )
        model2_metrics = calculate_consistency(
            votes, [model1_sample, model2_sample, model3_sample], "model2"
        )
        model3_metrics = calculate_consistency(
            votes, [model1_sample, model2_sample, model3_sample], "model3"
        )

        assert model1_metrics["comparisons"] == 2.0
        assert model2_metrics["comparisons"] == 2.0
        assert model3_metrics["comparisons"] == 2.0
        assert model1_metrics["win_rate"] == 1.0
        assert model2_metrics["win_rate"] == 0.5
        assert model3_metrics["win_rate"] == 0.0
