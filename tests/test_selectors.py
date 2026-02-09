"""Tests for matchup selector classes."""

from uuid import uuid4

import pytest

from asciibench.common.models import ArtSample, Vote
from asciibench.judge_ui.selectors import ModelPairSelector, SampleSelector


class TestModelPairSelector:
    """Tests for ModelPairSelector class."""

    @pytest.fixture
    def selector(self) -> ModelPairSelector:
        """Create a ModelPairSelector instance for tests."""
        return ModelPairSelector()

    def sample_data_multi_model(self) -> list[ArtSample]:
        """Create test data with samples from multiple models."""
        samples = []
        for model_id in ["model-a", "model-b", "model-c"]:
            for i in range(2):
                samples.append(
                    ArtSample(
                        id=uuid4(),
                        model_id=model_id,
                        prompt_text=f"test-{i}",
                        category="test",
                        attempt_number=1,
                        raw_output="",
                        sanitized_output="",
                        is_valid=True,
                    )
                )
        return samples

    def sample_data_single_model(self) -> list[ArtSample]:
        """Create test data with samples from one model."""
        samples = []
        for i in range(3):
            samples.append(
                ArtSample(
                    id=uuid4(),
                    model_id="model-a",
                    prompt_text=f"test-{i}",
                    category="test",
                    attempt_number=1,
                    raw_output="",
                    sanitized_output="",
                    is_valid=True,
                )
            )
        return samples

    def test_get_model_pair_comparison_counts_empty(self, selector: ModelPairSelector) -> None:
        """Test counting model comparisons with no votes."""
        samples = self.sample_data_multi_model()
        counts = selector.get_model_pair_comparison_counts([], samples)
        assert len(counts) == 0

    def test_get_model_pair_comparison_counts_single_vote(
        self, selector: ModelPairSelector
    ) -> None:
        """Test counting model comparisons with a single vote."""
        samples = self.sample_data_multi_model()
        vote = Vote(
            sample_a_id=str(samples[0].id),
            sample_b_id=str(samples[2].id),
            winner="A",
        )
        counts = selector.get_model_pair_comparison_counts([vote], samples)
        expected_pair = ("model-a", "model-b")
        assert counts[expected_pair] == 1
        assert len(counts) == 1

    def test_get_model_pair_comparison_counts_excludes_same_model(
        self, selector: ModelPairSelector
    ) -> None:
        """Test that votes between samples from same model are excluded."""
        samples = self.sample_data_single_model()
        vote = Vote(
            sample_a_id=str(samples[0].id),
            sample_b_id=str(samples[1].id),
            winner="A",
        )
        counts = selector.get_model_pair_comparison_counts([vote], samples)
        assert len(counts) == 0

    def test_get_model_pair_comparison_counts_normalizes_order(
        self, selector: ModelPairSelector
    ) -> None:
        """Test that (A,B) and (B,A) are counted as same model pair."""
        samples = self.sample_data_multi_model()
        votes = [
            Vote(sample_a_id=str(samples[0].id), sample_b_id=str(samples[2].id), winner="A"),
            Vote(sample_a_id=str(samples[2].id), sample_b_id=str(samples[0].id), winner="B"),
        ]
        counts = selector.get_model_pair_comparison_counts(votes, samples)
        expected_pair = ("model-a", "model-b")
        assert counts[expected_pair] == 2
        assert len(counts) == 1

    def test_get_least_compared_pairs_no_votes(self, selector: ModelPairSelector) -> None:
        """Test finding least compared pairs when no votes exist."""
        samples = self.sample_data_multi_model()
        model_ids = ["model-a", "model-b", "model-c"]
        least_compared = selector.get_least_compared_pairs(model_ids, [], samples)
        assert len(least_compared) == 3

    def test_get_least_compared_pairs_with_imbalance(self, selector: ModelPairSelector) -> None:
        """Test finding least compared pairs with imbalanced comparison counts."""
        samples = self.sample_data_multi_model()
        model_ids = ["model-a", "model-b", "model-c"]

        votes = []
        for _ in range(10):
            votes.append(
                Vote(
                    sample_a_id=str(samples[0].id),
                    sample_b_id=str(samples[4].id),
                    winner="A",
                )
            )

        least_compared = selector.get_least_compared_pairs(model_ids, votes, samples)

        least_compared_set = set(least_compared)
        assert ("model-a", "model-b") in least_compared_set
        assert ("model-b", "model-c") in least_compared_set

    def test_get_least_compared_pairs_tie(self, selector: ModelPairSelector) -> None:
        """Test finding least compared pairs when there's a tie."""
        samples = self.sample_data_multi_model()
        model_ids = ["model-a", "model-b", "model-c"]

        votes = [
            Vote(sample_a_id=str(samples[0].id), sample_b_id=str(samples[2].id), winner="A"),
            Vote(sample_a_id=str(samples[2].id), sample_b_id=str(samples[4].id), winner="B"),
        ]

        least_compared = selector.get_least_compared_pairs(model_ids, votes, samples)
        assert len(least_compared) >= 1

    def test_get_least_compared_pairs_two_models(self, selector: ModelPairSelector) -> None:
        """Test finding least compared pairs with only two models."""
        samples = [
            ArtSample(
                id=uuid4(),
                model_id="model-a",
                prompt_text="test",
                category="test",
                attempt_number=1,
                raw_output="",
                sanitized_output="",
                is_valid=True,
            ),
            ArtSample(
                id=uuid4(),
                model_id="model-b",
                prompt_text="test",
                category="test",
                attempt_number=1,
                raw_output="",
                sanitized_output="",
                is_valid=True,
            ),
        ]

        model_ids = ["model-a", "model-b"]
        least_compared = selector.get_least_compared_pairs(model_ids, [], samples)
        assert len(least_compared) == 1
        assert least_compared[0] == ("model-a", "model-b")


class TestSampleSelector:
    """Tests for SampleSelector class."""

    @pytest.fixture
    def selector(self) -> SampleSelector:
        """Create a SampleSelector instance for tests."""
        return SampleSelector()

    def samples_by_model_multi(self) -> dict[str, list[ArtSample]]:
        """Create test data with samples from multiple models."""
        return {
            "model-a": [
                ArtSample(
                    id=uuid4(),
                    model_id="model-a",
                    prompt_text=f"test-{i}",
                    category="test",
                    attempt_number=1,
                    raw_output="",
                    sanitized_output=f"output-{i}",
                    is_valid=True,
                )
                for i in range(2)
            ],
            "model-b": [
                ArtSample(
                    id=uuid4(),
                    model_id="model-b",
                    prompt_text=f"test-{i}",
                    category="test",
                    attempt_number=1,
                    raw_output="",
                    sanitized_output=f"output-{i}",
                    is_valid=True,
                )
                for i in range(2)
            ],
        }

    def samples_single_model(self) -> list[ArtSample]:
        """Create test data with samples from one model."""
        return [
            ArtSample(
                id=uuid4(),
                model_id="model-a",
                prompt_text=f"test-{i}",
                category="test",
                attempt_number=1,
                raw_output="",
                sanitized_output=f"output-{i}",
                is_valid=True,
            )
            for i in range(3)
        ]

    def test_select_pair_from_models_success(self, selector: SampleSelector) -> None:
        """Test selecting samples from two different models."""
        samples_by_model = self.samples_by_model_multi()
        sample_a, sample_b = selector.select_pair_from_models(
            "model-a", "model-b", samples_by_model
        )

        assert sample_a.model_id == "model-a"
        assert sample_b.model_id == "model-b"
        assert sample_a.id != sample_b.id

    def test_select_pair_from_models_missing_model_a(self, selector: SampleSelector) -> None:
        """Test that missing model_a raises ValueError."""
        samples_by_model = self.samples_by_model_multi()
        with pytest.raises(ValueError, match="No samples available for model: model-x"):
            selector.select_pair_from_models("model-x", "model-b", samples_by_model)

    def test_select_pair_from_models_missing_model_b(self, selector: SampleSelector) -> None:
        """Test that missing model_b raises ValueError."""
        samples_by_model = self.samples_by_model_multi()
        with pytest.raises(ValueError, match="No samples available for model: model-x"):
            selector.select_pair_from_models("model-a", "model-x", samples_by_model)

    def test_select_pair_from_single_model_success(self, selector: SampleSelector) -> None:
        """Test selecting samples when only one model is available."""
        samples = self.samples_single_model()
        sample_a, sample_b = selector.select_pair_from_single_model(samples)

        assert sample_a.model_id == "model-a"
        assert sample_b.model_id == "model-a"
        assert sample_a.id != sample_b.id

    def test_select_pair_from_single_model_insufficient_samples(
        self, selector: SampleSelector
    ) -> None:
        """Test that insufficient samples raise ValueError."""
        samples = [
            ArtSample(
                id=uuid4(),
                model_id="model-a",
                prompt_text="test",
                category="test",
                attempt_number=1,
                raw_output="",
                sanitized_output="",
                is_valid=True,
            )
        ]

        with pytest.raises(ValueError, match="Not enough valid samples"):
            selector.select_pair_from_single_model(samples)

    def test_select_pair_from_single_model_exactly_two(self, selector: SampleSelector) -> None:
        """Test selecting when exactly two samples are available."""
        samples = [
            ArtSample(
                id=uuid4(),
                model_id="model-a",
                prompt_text=f"test-{i}",
                category="test",
                attempt_number=1,
                raw_output="",
                sanitized_output=f"output-{i}",
                is_valid=True,
            )
            for i in range(2)
        ]

        sample_a, sample_b = selector.select_pair_from_single_model(samples)
        assert sample_a.id != sample_b.id

    def test_select_pair_from_models_empty_list(self, selector: SampleSelector) -> None:
        """Test that empty model list raises ValueError."""
        with pytest.raises(ValueError, match="No samples available for model: model-a"):
            selector.select_pair_from_models("model-a", "model-b", {})

    def test_select_pair_from_models_returns_different_samples(
        self, selector: SampleSelector
    ) -> None:
        """Test that selecting from models returns different sample IDs."""
        samples_by_model = self.samples_by_model_multi()
        selected_pairs = set()
        for _ in range(20):
            sample_a, sample_b = selector.select_pair_from_models(
                "model-a", "model-b", samples_by_model
            )
            pair = (str(sample_a.id), str(sample_b.id))
            selected_pairs.add(pair)

        assert len(selected_pairs) >= 2
