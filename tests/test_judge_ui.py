"""Tests for the Judge UI module."""

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from asciibench.common.models import ArtSample, Vote
from asciibench.common.persistence import append_jsonl
from asciibench.judge_ui.main import (
    MatchupResponse,
    SampleResponse,
    _get_model_pair_comparison_counts,
    _get_pair_comparison_counts,
    _make_sorted_pair,
    _select_matchup,
    app,
)


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def temp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up a temporary data directory for tests."""
    import asciibench.judge_ui.main as main_module

    monkeypatch.setattr(main_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main_module, "DATABASE_PATH", tmp_path / "database.jsonl")
    monkeypatch.setattr(main_module, "VOTES_PATH", tmp_path / "votes.jsonl")
    return tmp_path


@pytest.fixture
def sample_data() -> list[ArtSample]:
    """Create sample test data with samples from different models."""
    return [
        ArtSample(
            id=uuid4(),
            model_id="model-a",
            prompt_text="Draw a cat",
            category="single_animal",
            attempt_number=1,
            raw_output="```\n /\\_/\\\n( o.o )\n```",
            sanitized_output="/\\_/\\\n( o.o )",
            is_valid=True,
            timestamp=datetime.now(),
        ),
        ArtSample(
            id=uuid4(),
            model_id="model-a",
            prompt_text="Draw a dog",
            category="single_animal",
            attempt_number=1,
            raw_output="```\n/ \\__\n(    @\\___\n```",
            sanitized_output="/ \\__\n(    @\\___",
            is_valid=True,
            timestamp=datetime.now(),
        ),
        ArtSample(
            id=uuid4(),
            model_id="model-b",
            prompt_text="Draw a cat",
            category="single_animal",
            attempt_number=1,
            raw_output="```\n=^.^=\n```",
            sanitized_output="=^.^=",
            is_valid=True,
            timestamp=datetime.now(),
        ),
        ArtSample(
            id=uuid4(),
            model_id="model-b",
            prompt_text="Draw a dog",
            category="single_animal",
            attempt_number=1,
            raw_output="```\nU・ᴥ・U\n```",
            sanitized_output="U・ᴥ・U",
            is_valid=True,
            timestamp=datetime.now(),
        ),
        ArtSample(
            id=uuid4(),
            model_id="model-c",
            prompt_text="Draw a cat",
            category="single_animal",
            attempt_number=1,
            raw_output="```\n(^・ω・^)\n```",
            sanitized_output="(^・ω・^)",
            is_valid=True,
            timestamp=datetime.now(),
        ),
        # Invalid sample - should be excluded from matchups
        ArtSample(
            id=uuid4(),
            model_id="model-c",
            prompt_text="Draw a tree",
            category="single_object",
            attempt_number=1,
            raw_output="This is not valid",
            sanitized_output="",
            is_valid=False,
            timestamp=datetime.now(),
        ),
    ]


def populate_database(path: Path, samples: list[ArtSample]) -> None:
    """Populate database.jsonl with test samples."""
    db_path = path / "database.jsonl"
    for sample in samples:
        append_jsonl(db_path, sample)


class TestMatchupEndpoint:
    """Tests for the GET /api/matchup endpoint."""

    def test_matchup_returns_two_samples(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that matchup returns two samples."""
        populate_database(temp_data_dir, sample_data)

        response = client.get("/api/matchup")
        assert response.status_code == 200

        data = response.json()
        assert "sample_a" in data
        assert "sample_b" in data
        assert "prompt" in data

    def test_matchup_excludes_model_id(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that matchup response does NOT include model_id (double-blind)."""
        populate_database(temp_data_dir, sample_data)

        response = client.get("/api/matchup")
        assert response.status_code == 200

        data = response.json()
        # Check that model_id is NOT in the sample responses
        assert "model_id" not in data["sample_a"]
        assert "model_id" not in data["sample_b"]

    def test_matchup_includes_required_fields(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that matchup response includes required fields."""
        populate_database(temp_data_dir, sample_data)

        response = client.get("/api/matchup")
        assert response.status_code == 200

        data = response.json()
        # Check sample_a fields
        assert "id" in data["sample_a"]
        assert "sanitized_output" in data["sample_a"]
        assert "prompt_text" in data["sample_a"]

        # Check sample_b fields
        assert "id" in data["sample_b"]
        assert "sanitized_output" in data["sample_b"]
        assert "prompt_text" in data["sample_b"]

    def test_matchup_selects_from_different_models(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that matchup selects samples from different models."""
        populate_database(temp_data_dir, sample_data)

        # Run multiple matchups and verify samples come from different models
        sample_id_to_model = {str(s.id): s.model_id for s in sample_data}

        for _ in range(10):
            response = client.get("/api/matchup")
            assert response.status_code == 200

            data = response.json()
            sample_a_id = data["sample_a"]["id"]
            sample_b_id = data["sample_b"]["id"]

            model_a = sample_id_to_model.get(sample_a_id)
            model_b = sample_id_to_model.get(sample_b_id)

            # With multiple models available, they should be from different models
            assert model_a != model_b, "Samples should be from different models"

    def test_matchup_excludes_invalid_samples(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that matchup only uses valid samples."""
        populate_database(temp_data_dir, sample_data)

        # Find the invalid sample
        invalid_sample_ids = {str(s.id) for s in sample_data if not s.is_valid}

        for _ in range(10):
            response = client.get("/api/matchup")
            assert response.status_code == 200

            data = response.json()
            assert data["sample_a"]["id"] not in invalid_sample_ids
            assert data["sample_b"]["id"] not in invalid_sample_ids

    def test_matchup_error_with_no_samples(
        self,
        client: TestClient,
        temp_data_dir: Path,
    ) -> None:
        """Test that matchup returns error when no samples exist."""
        # Don't populate any samples
        response = client.get("/api/matchup")
        assert response.status_code == 400

        data = response.json()
        assert "Not enough valid samples" in data["detail"]

    def test_matchup_error_with_one_valid_sample(
        self,
        client: TestClient,
        temp_data_dir: Path,
    ) -> None:
        """Test that matchup returns error with only one valid sample."""
        single_sample = ArtSample(
            id=uuid4(),
            model_id="model-a",
            prompt_text="Draw a cat",
            category="single_animal",
            attempt_number=1,
            raw_output="```\ncat\n```",
            sanitized_output="cat",
            is_valid=True,
            timestamp=datetime.now(),
        )
        populate_database(temp_data_dir, [single_sample])

        response = client.get("/api/matchup")
        assert response.status_code == 400

        data = response.json()
        assert "Not enough valid samples" in data["detail"]
        assert "1 valid sample" in data["detail"]

    def test_matchup_works_with_only_invalid_except_two(
        self,
        client: TestClient,
        temp_data_dir: Path,
    ) -> None:
        """Test matchup works when only 2 valid samples exist among many invalid."""
        samples = [
            ArtSample(
                id=uuid4(),
                model_id="model-a",
                prompt_text="Draw a cat",
                category="single_animal",
                attempt_number=1,
                raw_output="```\ncat-a\n```",
                sanitized_output="cat-a",
                is_valid=True,
                timestamp=datetime.now(),
            ),
            ArtSample(
                id=uuid4(),
                model_id="model-b",
                prompt_text="Draw a cat",
                category="single_animal",
                attempt_number=1,
                raw_output="```\ncat-b\n```",
                sanitized_output="cat-b",
                is_valid=True,
                timestamp=datetime.now(),
            ),
            # Invalid samples
            ArtSample(
                id=uuid4(),
                model_id="model-c",
                prompt_text="Draw a dog",
                category="single_animal",
                attempt_number=1,
                raw_output="invalid",
                sanitized_output="",
                is_valid=False,
                timestamp=datetime.now(),
            ),
            ArtSample(
                id=uuid4(),
                model_id="model-d",
                prompt_text="Draw a tree",
                category="single_object",
                attempt_number=1,
                raw_output="invalid",
                sanitized_output="",
                is_valid=False,
                timestamp=datetime.now(),
            ),
        ]
        populate_database(temp_data_dir, samples)

        response = client.get("/api/matchup")
        assert response.status_code == 200


class TestComparisonPrioritization:
    """Tests for the comparison count prioritization logic."""

    def test_get_pair_comparison_counts_empty(self) -> None:
        """Test counting comparisons with no votes."""
        counts = _get_pair_comparison_counts([])
        assert len(counts) == 0

    def test_get_pair_comparison_counts_single_vote(self) -> None:
        """Test counting comparisons with a single vote."""
        vote = Vote(
            sample_a_id="sample-1",
            sample_b_id="sample-2",
            winner="A",
        )
        counts = _get_pair_comparison_counts([vote])
        # Pair should be stored in sorted order
        expected_pair = _make_sorted_pair("sample-1", "sample-2")
        assert counts[expected_pair] == 1

    def test_get_pair_comparison_counts_normalizes_order(self) -> None:
        """Test that (A,B) and (B,A) are counted as the same pair."""
        votes = [
            Vote(sample_a_id="sample-1", sample_b_id="sample-2", winner="A"),
            Vote(sample_a_id="sample-2", sample_b_id="sample-1", winner="B"),
        ]
        counts = _get_pair_comparison_counts(votes)
        expected_pair = _make_sorted_pair("sample-1", "sample-2")
        assert counts[expected_pair] == 2
        assert len(counts) == 1  # Only one unique pair

    def test_get_model_pair_comparison_counts(self) -> None:
        """Test counting comparisons by model pairs."""
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
        vote = Vote(
            sample_a_id=str(samples[0].id),
            sample_b_id=str(samples[1].id),
            winner="A",
        )
        counts = _get_model_pair_comparison_counts([vote], samples)
        expected_pair = _make_sorted_pair("model-a", "model-b")
        assert counts[expected_pair] == 1

    def test_select_matchup_prioritizes_less_compared(self) -> None:
        """Test that matchup selection prioritizes model pairs with fewer comparisons."""
        # Create samples from 3 models
        samples = []
        for model in ["model-a", "model-b", "model-c"]:
            samples.append(
                ArtSample(
                    id=uuid4(),
                    model_id=model,
                    prompt_text="test",
                    category="test",
                    attempt_number=1,
                    raw_output="",
                    sanitized_output="",
                    is_valid=True,
                )
            )

        # Create votes that compare model-a vs model-b multiple times
        votes = []
        for _ in range(10):
            votes.append(
                Vote(
                    sample_a_id=str(samples[0].id),  # model-a
                    sample_b_id=str(samples[1].id),  # model-b
                    winner="A",
                )
            )

        # Run many selections and track which model pairs are selected
        pair_counts: dict[tuple[str, str], int] = {}
        for _ in range(100):
            sample_a, sample_b = _select_matchup(samples, votes)
            pair = _make_sorted_pair(sample_a.model_id, sample_b.model_id)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # model-a vs model-b should be selected less often than other pairs
        # because they already have 10 comparisons
        ab_pair = _make_sorted_pair("model-a", "model-b")
        ac_pair = _make_sorted_pair("model-a", "model-c")
        bc_pair = _make_sorted_pair("model-b", "model-c")

        # The less-compared pairs should dominate
        assert ab_pair not in pair_counts or pair_counts.get(ab_pair, 0) == 0
        # At least one of the less-compared pairs should be selected frequently
        assert pair_counts.get(ac_pair, 0) > 0 or pair_counts.get(bc_pair, 0) > 0

    def test_select_matchup_with_single_model(self) -> None:
        """Test matchup selection when all samples are from one model."""
        samples = [
            ArtSample(
                id=uuid4(),
                model_id="model-a",
                prompt_text="test1",
                category="test",
                attempt_number=1,
                raw_output="",
                sanitized_output="output1",
                is_valid=True,
            ),
            ArtSample(
                id=uuid4(),
                model_id="model-a",
                prompt_text="test2",
                category="test",
                attempt_number=1,
                raw_output="",
                sanitized_output="output2",
                is_valid=True,
            ),
        ]

        # Should still work, returning two different samples from same model
        sample_a, sample_b = _select_matchup(samples, [])
        assert sample_a.id != sample_b.id

    def test_select_matchup_raises_with_insufficient_samples(self) -> None:
        """Test that matchup selection raises error with fewer than 2 samples."""
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
        ]

        with pytest.raises(ValueError, match="Not enough valid samples"):
            _select_matchup(samples, [])


class TestPositionRandomization:
    """Tests for A/B position randomization."""

    def test_matchup_randomizes_positions(
        self,
        client: TestClient,
        temp_data_dir: Path,
    ) -> None:
        """Test that sample A and B positions are randomized over many calls."""
        # Create exactly 2 samples from different models
        sample1 = ArtSample(
            id=uuid4(),
            model_id="model-a",
            prompt_text="Draw a cat",
            category="test",
            attempt_number=1,
            raw_output="```\ncat-a\n```",
            sanitized_output="cat-a",
            is_valid=True,
        )
        sample2 = ArtSample(
            id=uuid4(),
            model_id="model-b",
            prompt_text="Draw a cat",
            category="test",
            attempt_number=1,
            raw_output="```\ncat-b\n```",
            sanitized_output="cat-b",
            is_valid=True,
        )
        populate_database(temp_data_dir, [sample1, sample2])

        # Track which sample appears in position A
        sample1_as_a = 0
        sample2_as_a = 0

        for _ in range(100):
            response = client.get("/api/matchup")
            assert response.status_code == 200
            data = response.json()

            if data["sample_a"]["id"] == str(sample1.id):
                sample1_as_a += 1
            else:
                sample2_as_a += 1

        # Both samples should appear in position A roughly equally
        # With 100 trials, we expect roughly 50 each, allow for some variance
        assert sample1_as_a > 20, f"Sample 1 appeared as A only {sample1_as_a} times"
        assert sample2_as_a > 20, f"Sample 2 appeared as A only {sample2_as_a} times"


class TestResponseModels:
    """Tests for response model validation."""

    def test_sample_response_model(self) -> None:
        """Test SampleResponse model creation."""
        sample = SampleResponse(
            id="test-id",
            sanitized_output="test output",
            prompt_text="Draw a cat",
        )
        assert sample.id == "test-id"
        assert sample.sanitized_output == "test output"
        assert sample.prompt_text == "Draw a cat"

    def test_matchup_response_model(self) -> None:
        """Test MatchupResponse model creation."""
        sample_a = SampleResponse(
            id="id-a",
            sanitized_output="output-a",
            prompt_text="Draw a cat",
        )
        sample_b = SampleResponse(
            id="id-b",
            sanitized_output="output-b",
            prompt_text="Draw a cat",
        )
        matchup = MatchupResponse(
            sample_a=sample_a,
            sample_b=sample_b,
            prompt="Draw a cat",
        )
        assert matchup.sample_a.id == "id-a"
        assert matchup.sample_b.id == "id-b"
        assert matchup.prompt == "Draw a cat"
