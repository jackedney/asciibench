"""Tests for Swiss tournament selector classes."""

import random
from unittest.mock import patch

import pytest

from asciibench.common.models import Prompt
from asciibench.judge_ui.swiss_selector import PromptSelector, SwissPairSelector


class TestSwissPairSelector:
    """Tests for SwissPairSelector class."""

    @pytest.fixture
    def selector(self) -> SwissPairSelector:
        """Create a SwissPairSelector instance for tests."""
        return SwissPairSelector()

    @pytest.fixture
    def sample_models(self) -> list[str]:
        """Create a sample list of model IDs."""
        return ["model-a", "model-b", "model-c", "model-d", "model-e"]

    def test_select_pairs_with_empty_elo_round_1(
        self, selector: SwissPairSelector, sample_models: list[str]
    ) -> None:
        """Test select_pairs with empty Elo (round 1): random, correct count, no duplicates."""
        n = 2
        pairs = selector.select_pairs(sample_models, {}, n)

        assert len(pairs) == 2 * n

        assert len(pairs) == len(set(pairs))

        for pair in pairs:
            assert pair[0] in sample_models
            assert pair[1] in sample_models
            assert pair[0] != pair[1]

    def test_select_pairs_with_empty_elo_correct_model_count(
        self, selector: SwissPairSelector
    ) -> None:
        """Test select_pairs with empty Elo returns exactly 2*N pairs."""
        models = ["m1", "m2", "m3", "m4", "m5", "m6"]
        n = 3

        pairs = selector.select_pairs(models, {}, n)

        assert len(pairs) == 2 * n

    def test_select_pairs_with_elo_ratings_closest_and_random(
        self, selector: SwissPairSelector, sample_models: list[str]
    ) -> None:
        """Test select_pairs with Elo: N closest + N random, verify closest pairs are correct."""
        elo_ratings = {
            "model-a": 1500.0,
            "model-b": 1505.0,
            "model-c": 1510.0,
            "model-d": 1600.0,
            "model-e": 1700.0,
        }
        n = 2

        pairs = selector.select_pairs(sample_models, elo_ratings, n)

        assert len(pairs) == 2 * n

        expected_closest_pairs = [
            ("model-a", "model-b"),
            ("model-b", "model-c"),
        ]

        closest_pairs = pairs[:n]
        for expected_pair in expected_closest_pairs:
            assert expected_pair in closest_pairs

        all_pairs_unique = len(pairs) == len(set(pairs))
        assert all_pairs_unique

    def test_select_pairs_fewer_models_than_needed(self, selector: SwissPairSelector) -> None:
        """Test edge case: fewer models than needed for 2N pairs → returns all available pairs."""
        models = ["m1", "m2", "m3"]
        n = 3

        pairs = selector.select_pairs(models, {}, n)

        assert len(pairs) == 3

        all_possible_pairs = {
            ("m1", "m2"),
            ("m1", "m3"),
            ("m2", "m3"),
        }
        assert set(pairs) == all_possible_pairs

    def test_select_pairs_two_models_returns_one_pair(self, selector: SwissPairSelector) -> None:
        """Test edge case: 2 models → returns 1 pair maximum."""
        models = ["m1", "m2"]
        n = 5

        pairs = selector.select_pairs(models, {}, n)

        assert len(pairs) == 1
        assert pairs[0] == ("m1", "m2")

    def test_select_pairs_less_than_two_models_returns_empty(
        self, selector: SwissPairSelector
    ) -> None:
        """Test edge case: less than 2 models → returns empty list."""
        assert selector.select_pairs(["m1"], {}, 1) == []
        assert selector.select_pairs([], {}, 1) == []

    def test_select_pairs_with_elo_all_equal(self, selector: SwissPairSelector) -> None:
        """Test select_pairs with all Elo ratings equal - should handle gracefully."""
        models = ["m1", "m2", "m3", "m4"]
        elo_ratings: dict[str, float] = dict.fromkeys(models, 1500.0)
        n = 2

        pairs = selector.select_pairs(models, elo_ratings, n)

        assert len(pairs) == 2 * n
        assert len(pairs) == len(set(pairs))

    def test_select_pairs_with_missing_elo_ratings(self, selector: SwissPairSelector) -> None:
        """Test select_pairs handles missing Elo ratings by using default 1500.0."""
        models = ["m1", "m2", "m3", "m4"]
        elo_ratings = {"m1": 1500.0, "m2": 1501.0}
        n = 2

        pairs = selector.select_pairs(models, elo_ratings, n)

        assert len(pairs) == 2 * n

    def test_select_pairs_no_duplicate_models_in_pair(self, selector: SwissPairSelector) -> None:
        """Test that pairs never contain the same model twice."""
        models = ["model-a", "model-b", "model-c"]
        n = 2

        pairs = selector.select_pairs(models, {}, n)

        for pair in pairs:
            assert pair[0] != pair[1], f"Pair {pair} contains duplicate model"

    def test_select_pairs_deterministic_with_seed(self, selector: SwissPairSelector) -> None:
        """Test that select_pairs is deterministic when random is seeded."""
        models = ["m1", "m2", "m3", "m4", "m5", "m6"]

        with patch.object(random, "sample") as mock_sample:
            mock_sample.return_value = [("m1", "m2"), ("m3", "m4")]
            pairs1 = selector.select_pairs(models, {}, 2)
            pairs2 = selector.select_pairs(models, {}, 2)

        assert pairs1 == pairs2


class TestPromptSelector:
    """Tests for PromptSelector class."""

    @pytest.fixture
    def selector(self) -> PromptSelector:
        """Create a PromptSelector instance for tests."""
        return PromptSelector()

    @pytest.fixture
    def sample_prompts(self) -> list[Prompt]:
        """Create a sample list of prompts."""
        return [
            Prompt(text="Draw a cat", category="animal", template_type="simple"),
            Prompt(text="Draw a dog", category="animal", template_type="simple"),
            Prompt(text="Draw a bird", category="animal", template_type="simple"),
        ]

    def test_select_prompt_returns_unused_prompt(
        self, selector: PromptSelector, sample_prompts: list[Prompt]
    ) -> None:
        """Test select_prompt returns unused prompt when available."""
        used_prompts = {"Draw a cat"}

        with patch.object(random, "choice") as mock_choice:
            mock_choice.return_value = Prompt(
                text="Draw a dog", category="animal", template_type="simple"
            )
            result = selector.select_prompt("model-a", "model-b", sample_prompts, used_prompts)

        assert result is not None
        assert result.text == "Draw a dog"
        assert result.text not in used_prompts

    def test_select_prompt_all_prompts_unused(
        self, selector: PromptSelector, sample_prompts: list[Prompt]
    ) -> None:
        """Test select_prompt works when all prompts are unused."""
        used_prompts = set()

        result = selector.select_prompt("model-a", "model-b", sample_prompts, used_prompts)

        assert result is not None
        assert result in sample_prompts

    def test_select_prompt_fallback_all_prompts_used(
        self, selector: PromptSelector, sample_prompts: list[Prompt]
    ) -> None:
        """Test select_prompt fallback when all prompts used."""
        used_prompts = {p.text for p in sample_prompts}

        result = selector.select_prompt("model-a", "model-b", sample_prompts, used_prompts)

        assert result is not None
        assert result in sample_prompts

    def test_select_prompt_with_empty_prompt_list(self, selector: PromptSelector) -> None:
        """Test select_prompt with empty prompt list → returns None."""
        used_prompts = set()

        result = selector.select_prompt("model-a", "model-b", [], used_prompts)

        assert result is None

    def test_select_prompt_some_prompts_unused(
        self, selector: PromptSelector, sample_prompts: list[Prompt]
    ) -> None:
        """Test select_prompt returns one of the unused prompts when only some are used."""
        used_prompts = {"Draw a cat"}

        result = selector.select_prompt("model-a", "model-b", sample_prompts, used_prompts)

        assert result is not None
        assert result.text in {"Draw a dog", "Draw a bird"}

    def test_select_prompt_single_unused_prompt(self, selector: PromptSelector) -> None:
        """Test select_prompt returns the only unused prompt."""
        prompts = [
            Prompt(text="Draw a cat", category="animal", template_type="simple"),
            Prompt(text="Draw a dog", category="animal", template_type="simple"),
        ]
        used_prompts = {"Draw a cat"}

        with patch.object(random, "choice") as mock_choice:
            mock_choice.return_value = prompts[1]
            result = selector.select_prompt("model-a", "model-b", prompts, used_prompts)

        assert result is not None
        assert result.text == "Draw a dog"

    def test_select_prompt_with_duplicate_prompt_texts(self, selector: PromptSelector) -> None:
        """Test select_prompt handles prompts with duplicate text."""
        prompts = [
            Prompt(text="Draw a cat", category="animal", template_type="simple"),
            Prompt(text="Draw a cat", category="animal", template_type="simple"),
            Prompt(text="Draw a dog", category="animal", template_type="simple"),
        ]
        used_prompts = {"Draw a cat"}

        result = selector.select_prompt("model-a", "model-b", prompts, used_prompts)

        assert result is not None
        assert result.text == "Draw a dog"

    def test_select_prompt_model_args_unused(
        self, selector: PromptSelector, sample_prompts: list[Prompt]
    ) -> None:
        """Test that model_a and model_b args are accepted but not currently used."""
        used_prompts = set()

        result = selector.select_prompt("model-x", "model-y", sample_prompts, used_prompts)

        assert result is not None
        assert result in sample_prompts

    def test_select_prompt_deterministic_with_seed(
        self, selector: PromptSelector, sample_prompts: list[Prompt]
    ) -> None:
        """Test that select_prompt is deterministic when random is seeded."""
        used_prompts = set()

        with patch.object(random, "choice") as mock_choice:
            mock_choice.return_value = sample_prompts[1]
            result1 = selector.select_prompt("model-a", "model-b", sample_prompts, used_prompts)
            result2 = selector.select_prompt("model-a", "model-b", sample_prompts, used_prompts)

        assert result1 == result2
