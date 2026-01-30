"""Tests for Elo rating calculations."""

from datetime import datetime, timedelta
from uuid import uuid4

from asciibench.analyst.elo import BASE_RATING, K_FACTOR, calculate_elo
from asciibench.common.models import ArtSample, Vote


class TestCalculateElo:
    """Tests for calculate_elo function."""

    def test_empty_votes_returns_empty_dict(self):
        """Empty votes list returns empty dict."""
        ratings = calculate_elo([], [])
        assert ratings == {}

    def test_single_win_for_model_a(self):
        """Model A wins against Model B."""
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

        ratings = calculate_elo([vote], [sample_a, sample_b])

        assert ratings["model1"] > BASE_RATING
        assert ratings["model2"] < BASE_RATING
        assert abs(ratings["model1"] - BASE_RATING - K_FACTOR * (1 - 0.5)) < 0.01
        assert abs(ratings["model2"] - BASE_RATING - K_FACTOR * (0 - 0.5)) < 0.01

    def test_single_win_for_model_b(self):
        """Model B wins against Model A."""
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
        vote = Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="B")

        ratings = calculate_elo([vote], [sample_a, sample_b])

        assert ratings["model1"] < BASE_RATING
        assert ratings["model2"] > BASE_RATING

    def test_tie_moves_ratings_closer(self):
        """Tie vote moves both ratings toward each other slightly."""
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
        vote = Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="tie")

        ratings = calculate_elo([vote], [sample_a, sample_b])

        assert ratings["model1"] == BASE_RATING
        assert ratings["model2"] == BASE_RATING

    def test_fail_vote_no_rating_change(self):
        """Fail vote causes no rating change for models."""
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
        ]

        ratings = calculate_elo(votes, [sample_a, sample_b])

        assert ratings["model1"] > BASE_RATING
        assert ratings["model2"] < BASE_RATING

    def test_ten_consecutive_wins_for_model_a(self):
        """10 votes of A winning -> Model A rating increases, Model B decreases."""
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
            for _ in range(10)
        ]

        ratings = calculate_elo(votes, [sample_a, sample_b])

        assert ratings["model1"] > BASE_RATING
        assert ratings["model2"] < BASE_RATING
        assert ratings["model1"] > ratings["model2"]

    def test_votes_with_same_model_skipped(self):
        """Votes with same model on both sides are skipped."""
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
            model_id="model1",
            prompt_text="test",
            category="test",
            attempt_number=2,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        vote = Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A")

        ratings = calculate_elo([vote], [sample_a, sample_b])

        assert ratings == {}

    def test_missing_sample_id_skipped(self):
        """Votes with missing sample IDs are skipped."""
        vote = Vote(sample_a_id=str(uuid4()), sample_b_id=str(uuid4()), winner="A")

        ratings = calculate_elo([vote], [])

        assert ratings == {}

    def test_processes_votes_chronologically(self):
        """Votes are processed chronologically by timestamp."""
        sample_a1 = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="test",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b1 = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="test",
            attempt_number=1,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_a2 = ArtSample(
            id=uuid4(),
            model_id="model1",
            prompt_text="test",
            category="test",
            attempt_number=2,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )
        sample_b2 = ArtSample(
            id=uuid4(),
            model_id="model2",
            prompt_text="test",
            category="test",
            attempt_number=2,
            raw_output="test",
            sanitized_output="test",
            is_valid=True,
        )

        earlier = datetime.now() - timedelta(hours=1)
        later = datetime.now()

        votes_out_of_order = [
            Vote(
                sample_a_id=str(sample_a2.id),
                sample_b_id=str(sample_b2.id),
                winner="B",
                timestamp=later,
            ),
            Vote(
                sample_a_id=str(sample_a1.id),
                sample_b_id=str(sample_b1.id),
                winner="A",
                timestamp=earlier,
            ),
        ]

        ratings = calculate_elo(votes_out_of_order, [sample_a1, sample_b1, sample_a2, sample_b2])

        assert ratings["model1"] != ratings["model2"]

    def test_multiple_models(self):
        """Handles multiple models correctly."""
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
            Vote(sample_a_id=str(model2_sample.id), sample_b_id=str(model3_sample.id), winner="A"),
            Vote(sample_a_id=str(model1_sample.id), sample_b_id=str(model3_sample.id), winner="B"),
        ]

        ratings = calculate_elo(votes, [model1_sample, model2_sample, model3_sample])

        assert "model1" in ratings
        assert "model2" in ratings
        assert "model3" in ratings
        assert all(r != BASE_RATING for r in ratings.values())

    def test_k_factor_32(self):
        """Uses K-factor of 32 for rating updates."""
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

        ratings = calculate_elo([vote], [sample_a, sample_b])

        expected_a_gain = K_FACTOR * (1 - 0.5)
        assert abs(ratings["model1"] - BASE_RATING - expected_a_gain) < 0.01

    def test_base_rating_1500(self):
        """Initializes all models at base rating 1500."""
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
        sample_c = ArtSample(
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
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
        ]

        ratings = calculate_elo(votes, [sample_a, sample_b, sample_c])

        assert sample_c.model_id not in ratings

        ratings_with_all = calculate_elo(
            [
                Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="A"),
                Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_c.id), winner="A"),
            ],
            [sample_a, sample_b, sample_c],
        )

        assert sample_c.model_id in ratings_with_all
        assert all(rating >= 1400 for rating in ratings_with_all.values())

    def test_mixed_winners(self):
        """Handles mixed win/loss/tie/fail votes."""
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
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="tie"),
            Vote(sample_a_id=str(sample_a.id), sample_b_id=str(sample_b.id), winner="fail"),
        ]

        ratings = calculate_elo(votes, [sample_a, sample_b])

        assert "model1" in ratings
        assert "model2" in ratings
        assert ratings["model1"] != ratings["model2"]
