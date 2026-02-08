"""Tests for the Judge UI module."""

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from asciibench.common.models import ArtSample, VLMEvaluation, Vote
from asciibench.common.persistence import append_jsonl, read_jsonl
from asciibench.judge_ui.api_models import (
    CategoryProgress,
    MatchupResponse,
    ProgressResponse,
    SampleResponse,
    VoteRequest,
    VoteResponse,
)
from asciibench.judge_ui.main import app
from asciibench.judge_ui.matchup_service import MatchupService
from asciibench.judge_ui.undo_service import UndoService


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
    monkeypatch.setattr(main_module, "VLM_EVALUATIONS_PATH", tmp_path / "vlm_evaluations.jsonl")
    # Create MatchupService and UndoService instances for tests
    matchup_service = MatchupService(
        database_path=tmp_path / "database.jsonl", votes_path=tmp_path / "votes.jsonl"
    )
    undo_service = UndoService(votes_path=tmp_path / "votes.jsonl")
    monkeypatch.setattr(main_module, "matchup_service", matchup_service)
    monkeypatch.setattr(main_module, "undo_service", undo_service)
    return tmp_path


@pytest.fixture
def matchup_service(temp_data_dir: Path) -> MatchupService:
    """Create a MatchupService instance for tests."""
    return MatchupService(
        database_path=temp_data_dir / "database.jsonl",
        votes_path=temp_data_dir / "votes.jsonl",
    )


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

    def test_get_pair_comparison_counts_empty(self, matchup_service: MatchupService) -> None:
        """Test counting comparisons with no votes."""
        counts = matchup_service._get_pair_comparison_counts([])
        assert len(counts) == 0

    def test_get_pair_comparison_counts_single_vote(self, matchup_service: MatchupService) -> None:
        """Test counting comparisons with a single vote."""
        vote = Vote(
            sample_a_id="sample-1",
            sample_b_id="sample-2",
            winner="A",
        )
        counts = matchup_service._get_pair_comparison_counts([vote])
        # Pair should be stored in sorted order
        expected_pair = matchup_service._make_sorted_pair("sample-1", "sample-2")
        assert counts[expected_pair] == 1

    def test_get_pair_comparison_counts_normalizes_order(
        self, matchup_service: MatchupService
    ) -> None:
        """Test that (A,B) and (B,A) are counted as the same pair."""
        votes = [
            Vote(sample_a_id="sample-1", sample_b_id="sample-2", winner="A"),
            Vote(sample_a_id="sample-2", sample_b_id="sample-1", winner="B"),
        ]
        counts = matchup_service._get_pair_comparison_counts(votes)
        expected_pair = matchup_service._make_sorted_pair("sample-1", "sample-2")
        assert counts[expected_pair] == 2
        assert len(counts) == 1  # Only one unique pair

    def test_get_model_pair_comparison_counts(self, matchup_service: MatchupService) -> None:
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
        counts = matchup_service._get_model_pair_comparison_counts([vote], samples)
        expected_pair = matchup_service._make_sorted_pair("model-a", "model-b")
        assert counts[expected_pair] == 1

    def test_select_matchup_prioritizes_less_compared(
        self, matchup_service: MatchupService
    ) -> None:
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
            sample_a, sample_b = matchup_service._select_matchup(samples, votes)
            pair = matchup_service._make_sorted_pair(sample_a.model_id, sample_b.model_id)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # model-a vs model-b should be selected less often than other pairs
        # because they already have 10 comparisons
        ab_pair = matchup_service._make_sorted_pair("model-a", "model-b")
        ac_pair = matchup_service._make_sorted_pair("model-a", "model-c")
        bc_pair = matchup_service._make_sorted_pair("model-b", "model-c")

        # The less-compared pairs should dominate
        assert ab_pair not in pair_counts or pair_counts.get(ab_pair, 0) == 0
        # At least one of the less-compared pairs should be selected frequently
        assert pair_counts.get(ac_pair, 0) > 0 or pair_counts.get(bc_pair, 0) > 0

    def test_select_matchup_with_single_model(self, matchup_service: MatchupService) -> None:
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
        sample_a, sample_b = matchup_service._select_matchup(samples, [])
        assert sample_a.id != sample_b.id

    def test_select_matchup_raises_with_insufficient_samples(
        self, matchup_service: MatchupService
    ) -> None:
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
            matchup_service._select_matchup(samples, [])


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

    def test_vote_request_model(self) -> None:
        """Test VoteRequest model creation."""
        vote_request = VoteRequest(
            sample_a_id="sample-1",
            sample_b_id="sample-2",
            winner="A",
        )
        assert vote_request.sample_a_id == "sample-1"
        assert vote_request.sample_b_id == "sample-2"
        assert vote_request.winner == "A"

    def test_vote_response_model(self) -> None:
        """Test VoteResponse model creation."""
        vote_response = VoteResponse(
            id="vote-id",
            sample_a_id="sample-1",
            sample_b_id="sample-2",
            winner="B",
            timestamp="2026-01-30T12:00:00",
        )
        assert vote_response.id == "vote-id"
        assert vote_response.sample_a_id == "sample-1"
        assert vote_response.sample_b_id == "sample-2"
        assert vote_response.winner == "B"


class TestVoteSubmission:
    """Tests for the POST /api/votes endpoint."""

    def test_submit_vote_success(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test successful vote submission with persistence."""
        populate_database(temp_data_dir, sample_data)

        # Get two sample IDs from the test data
        sample_a_id = str(sample_data[0].id)
        sample_b_id = str(sample_data[2].id)  # From different model

        response = client.post(
            "/api/votes",
            json={
                "sample_a_id": sample_a_id,
                "sample_b_id": sample_b_id,
                "winner": "A",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response contains expected fields
        assert "id" in data
        assert data["sample_a_id"] == sample_a_id
        assert data["sample_b_id"] == sample_b_id
        assert data["winner"] == "A"
        assert "timestamp" in data

        # Verify vote was persisted to votes.jsonl
        votes = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes) == 1
        assert str(votes[0].id) == data["id"]
        assert votes[0].sample_a_id == sample_a_id
        assert votes[0].sample_b_id == sample_b_id
        assert votes[0].winner == "A"

    def test_submit_vote_all_winner_types(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that all valid winner types (A, B, tie, fail) are accepted."""
        populate_database(temp_data_dir, sample_data)

        sample_a_id = str(sample_data[0].id)
        sample_b_id = str(sample_data[2].id)

        for winner in ["A", "B", "tie", "fail"]:
            response = client.post(
                "/api/votes",
                json={
                    "sample_a_id": sample_a_id,
                    "sample_b_id": sample_b_id,
                    "winner": winner,
                },
            )
            assert response.status_code == 200, f"Failed for winner={winner}"
            assert response.json()["winner"] == winner

    def test_submit_vote_generates_uuid(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that vote submission generates a unique UUID."""
        populate_database(temp_data_dir, sample_data)

        sample_a_id = str(sample_data[0].id)
        sample_b_id = str(sample_data[2].id)

        # Submit two votes
        response1 = client.post(
            "/api/votes",
            json={
                "sample_a_id": sample_a_id,
                "sample_b_id": sample_b_id,
                "winner": "A",
            },
        )
        response2 = client.post(
            "/api/votes",
            json={
                "sample_a_id": sample_a_id,
                "sample_b_id": sample_b_id,
                "winner": "B",
            },
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Verify different UUIDs
        id1 = response1.json()["id"]
        id2 = response2.json()["id"]
        assert id1 != id2

    def test_submit_vote_generates_timestamp(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that vote submission generates a valid timestamp."""
        populate_database(temp_data_dir, sample_data)

        sample_a_id = str(sample_data[0].id)
        sample_b_id = str(sample_data[2].id)

        response = client.post(
            "/api/votes",
            json={
                "sample_a_id": sample_a_id,
                "sample_b_id": sample_b_id,
                "winner": "tie",
            },
        )

        assert response.status_code == 200
        timestamp_str = response.json()["timestamp"]

        # Verify it's a valid ISO format timestamp
        timestamp = datetime.fromisoformat(timestamp_str)
        assert timestamp is not None

    def test_submit_vote_invalid_sample_a_id_404(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that invalid sample_a_id returns 404 error."""
        populate_database(temp_data_dir, sample_data)

        # Use a non-existent UUID for sample_a_id
        response = client.post(
            "/api/votes",
            json={
                "sample_a_id": str(uuid4()),  # Non-existent
                "sample_b_id": str(sample_data[0].id),
                "winner": "A",
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_submit_vote_invalid_sample_b_id_404(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that invalid sample_b_id returns 404 error."""
        populate_database(temp_data_dir, sample_data)

        # Use a non-existent UUID for sample_b_id
        response = client.post(
            "/api/votes",
            json={
                "sample_a_id": str(sample_data[0].id),
                "sample_b_id": str(uuid4()),  # Non-existent
                "winner": "B",
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_submit_vote_invalid_winner_422(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that invalid winner value returns 422 validation error."""
        populate_database(temp_data_dir, sample_data)

        sample_a_id = str(sample_data[0].id)
        sample_b_id = str(sample_data[2].id)

        response = client.post(
            "/api/votes",
            json={
                "sample_a_id": sample_a_id,
                "sample_b_id": sample_b_id,
                "winner": "invalid_winner",  # Invalid value
            },
        )

        # Pydantic validation returns 422 Unprocessable Entity
        assert response.status_code == 422

    def test_submit_vote_missing_fields_422(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that missing required fields returns 422 validation error."""
        populate_database(temp_data_dir, sample_data)

        # Missing winner field
        response = client.post(
            "/api/votes",
            json={
                "sample_a_id": str(sample_data[0].id),
                "sample_b_id": str(sample_data[2].id),
            },
        )
        assert response.status_code == 422

        # Missing sample_a_id
        response = client.post(
            "/api/votes",
            json={
                "sample_b_id": str(sample_data[2].id),
                "winner": "A",
            },
        )
        assert response.status_code == 422

        # Missing sample_b_id
        response = client.post(
            "/api/votes",
            json={
                "sample_a_id": str(sample_data[0].id),
                "winner": "A",
            },
        )
        assert response.status_code == 422

    def test_submit_vote_persists_multiple_votes(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that multiple votes are persisted correctly."""
        populate_database(temp_data_dir, sample_data)

        # Submit multiple votes
        for i, winner in enumerate(["A", "B", "tie", "fail"]):
            sample_a_idx = i % 3
            sample_b_idx = (i + 2) % 5
            if sample_b_idx == sample_a_idx:
                sample_b_idx = (sample_b_idx + 1) % 5

            response = client.post(
                "/api/votes",
                json={
                    "sample_a_id": str(sample_data[sample_a_idx].id),
                    "sample_b_id": str(sample_data[sample_b_idx].id),
                    "winner": winner,
                },
            )
            assert response.status_code == 200

        # Verify all votes were persisted
        votes = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes) == 4

    def test_submit_vote_no_database_file(
        self,
        client: TestClient,
        temp_data_dir: Path,
    ) -> None:
        """Test that vote submission fails if database.jsonl doesn't exist."""
        # Don't populate any samples - database.jsonl won't exist

        response = client.post(
            "/api/votes",
            json={
                "sample_a_id": str(uuid4()),
                "sample_b_id": str(uuid4()),
                "winner": "A",
            },
        )

        assert response.status_code == 404


class TestUndoEndpoint:
    """Tests for the POST /api/undo endpoint."""

    def test_undo_removes_last_vote(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that undo removes the most recent vote."""
        populate_database(temp_data_dir, sample_data)

        sample_a_id = str(sample_data[0].id)
        sample_b_id = str(sample_data[2].id)

        # Submit a vote
        vote_response = client.post(
            "/api/votes",
            json={
                "sample_a_id": sample_a_id,
                "sample_b_id": sample_b_id,
                "winner": "A",
            },
        )
        assert vote_response.status_code == 200
        vote_id = vote_response.json()["id"]

        # Verify vote was persisted
        votes_before = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes_before) == 1

        # Call undo
        undo_response = client.post("/api/undo")
        assert undo_response.status_code == 200

        # Verify the undone vote is returned
        undo_data = undo_response.json()
        assert undo_data["id"] == vote_id
        assert undo_data["sample_a_id"] == sample_a_id
        assert undo_data["sample_b_id"] == sample_b_id
        assert undo_data["winner"] == "A"

        # Verify vote was removed from file
        votes_after = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes_after) == 0

    def test_undo_only_removes_last_vote(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that undo only removes the most recent vote, not earlier ones."""
        populate_database(temp_data_dir, sample_data)

        sample_a_id = str(sample_data[0].id)
        sample_b_id = str(sample_data[2].id)

        # Submit multiple votes
        for winner in ["A", "B", "tie"]:
            client.post(
                "/api/votes",
                json={
                    "sample_a_id": sample_a_id,
                    "sample_b_id": sample_b_id,
                    "winner": winner,
                },
            )

        # Verify 3 votes were persisted
        votes_before = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes_before) == 3

        # Call undo
        undo_response = client.post("/api/undo")
        assert undo_response.status_code == 200

        # Verify only the last vote (tie) was removed
        undo_data = undo_response.json()
        assert undo_data["winner"] == "tie"

        # Verify 2 votes remain
        votes_after = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes_after) == 2
        assert votes_after[0].winner == "A"
        assert votes_after[1].winner == "B"

    def test_undo_with_no_votes_returns_404(
        self,
        client: TestClient,
        temp_data_dir: Path,
    ) -> None:
        """Test that undo returns 404 when no votes exist."""
        # Don't submit any votes - votes.jsonl won't exist

        response = client.post("/api/undo")
        assert response.status_code == 404

        data = response.json()
        assert "No votes to undo" in data["detail"]

    def test_undo_twice_in_row_returns_error(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that calling undo twice in a row without new vote returns error."""
        populate_database(temp_data_dir, sample_data)

        sample_a_id = str(sample_data[0].id)
        sample_b_id = str(sample_data[2].id)

        # Submit two votes
        for winner in ["A", "B"]:
            client.post(
                "/api/votes",
                json={
                    "sample_a_id": sample_a_id,
                    "sample_b_id": sample_b_id,
                    "winner": winner,
                },
            )

        # First undo should succeed
        first_undo = client.post("/api/undo")
        assert first_undo.status_code == 200

        # Second undo should fail (can't undo the same vote twice)
        second_undo = client.post("/api/undo")
        assert second_undo.status_code == 400

        data = second_undo.json()
        assert "Cannot undo twice in a row" in data["detail"]

        # Verify only one vote was undone (one remains)
        votes_after = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes_after) == 1

    def test_undo_then_vote_then_undo_works(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that undo works again after submitting a new vote."""
        populate_database(temp_data_dir, sample_data)

        sample_a_id = str(sample_data[0].id)
        sample_b_id = str(sample_data[2].id)

        # Submit first vote
        client.post(
            "/api/votes",
            json={
                "sample_a_id": sample_a_id,
                "sample_b_id": sample_b_id,
                "winner": "A",
            },
        )

        # Undo first vote
        first_undo = client.post("/api/undo")
        assert first_undo.status_code == 200

        # Submit second vote
        client.post(
            "/api/votes",
            json={
                "sample_a_id": sample_a_id,
                "sample_b_id": sample_b_id,
                "winner": "B",
            },
        )

        # Undo second vote should work
        second_undo = client.post("/api/undo")
        assert second_undo.status_code == 200
        assert second_undo.json()["winner"] == "B"

        # Verify no votes remain
        votes_after = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes_after) == 0

    def test_undo_returns_correct_vote_details(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that undo returns all the correct vote details."""
        populate_database(temp_data_dir, sample_data)

        sample_a_id = str(sample_data[0].id)
        sample_b_id = str(sample_data[2].id)

        # Submit a vote
        vote_response = client.post(
            "/api/votes",
            json={
                "sample_a_id": sample_a_id,
                "sample_b_id": sample_b_id,
                "winner": "fail",
            },
        )
        vote_data = vote_response.json()

        # Call undo
        undo_response = client.post("/api/undo")
        assert undo_response.status_code == 200

        undo_data = undo_response.json()

        # Verify all fields match
        assert undo_data["id"] == vote_data["id"]
        assert undo_data["sample_a_id"] == vote_data["sample_a_id"]
        assert undo_data["sample_b_id"] == vote_data["sample_b_id"]
        assert undo_data["winner"] == vote_data["winner"]
        assert undo_data["timestamp"] == vote_data["timestamp"]

    def test_undo_atomic_operation(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that undo rewrites votes.jsonl atomically."""
        populate_database(temp_data_dir, sample_data)

        sample_a_id = str(sample_data[0].id)
        sample_b_id = str(sample_data[2].id)

        # Submit several votes
        for winner in ["A", "B", "tie", "fail"]:
            client.post(
                "/api/votes",
                json={
                    "sample_a_id": sample_a_id,
                    "sample_b_id": sample_b_id,
                    "winner": winner,
                },
            )

        # Undo the last vote
        undo_response = client.post("/api/undo")
        assert undo_response.status_code == 200

        # Read the file and verify it's valid JSONL with correct content
        votes = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes) == 3
        assert votes[0].winner == "A"
        assert votes[1].winner == "B"
        assert votes[2].winner == "tie"


class TestProgressEndpoint:
    """Tests for the GET /api/progress endpoint."""

    def test_progress_returns_zeros_with_no_votes(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that progress returns zeros when no votes exist."""
        populate_database(temp_data_dir, sample_data)

        response = client.get("/api/progress")
        assert response.status_code == 200

        data = response.json()
        assert data["votes_completed"] == 0
        assert data["unique_pairs_judged"] == 0
        assert data["total_possible_pairs"] > 0
        assert "by_category" in data

    def test_progress_returns_zeros_with_no_samples(
        self,
        client: TestClient,
        temp_data_dir: Path,
    ) -> None:
        """Test that progress returns zeros when no samples exist."""
        # Don't populate any samples

        response = client.get("/api/progress")
        assert response.status_code == 200

        data = response.json()
        assert data["votes_completed"] == 0
        assert data["unique_pairs_judged"] == 0
        assert data["total_possible_pairs"] == 0
        assert data["by_category"] == {}

    def test_progress_counts_votes(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that progress correctly counts total votes."""
        populate_database(temp_data_dir, sample_data)

        sample_a_id = str(sample_data[0].id)
        sample_b_id = str(sample_data[2].id)

        # Submit multiple votes
        for winner in ["A", "B", "tie"]:
            client.post(
                "/api/votes",
                json={
                    "sample_a_id": sample_a_id,
                    "sample_b_id": sample_b_id,
                    "winner": winner,
                },
            )

        response = client.get("/api/progress")
        assert response.status_code == 200

        data = response.json()
        assert data["votes_completed"] == 3

    def test_progress_counts_unique_pairs(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that progress correctly counts unique model pairs judged."""
        populate_database(temp_data_dir, sample_data)

        # Submit votes between model-a and model-b
        sample_a_id = str(sample_data[0].id)  # model-a
        sample_b_id = str(sample_data[2].id)  # model-b

        # Multiple votes between same model pair count as 1 unique pair
        for _ in range(3):
            client.post(
                "/api/votes",
                json={
                    "sample_a_id": sample_a_id,
                    "sample_b_id": sample_b_id,
                    "winner": "A",
                },
            )

        response = client.get("/api/progress")
        assert response.status_code == 200

        data = response.json()
        assert data["votes_completed"] == 3
        assert data["unique_pairs_judged"] == 1

    def test_progress_counts_multiple_unique_pairs(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that progress correctly counts multiple unique model pairs."""
        populate_database(temp_data_dir, sample_data)

        # Get samples from different models
        model_a_sample = str(sample_data[0].id)  # model-a
        model_b_sample = str(sample_data[2].id)  # model-b
        model_c_sample = str(sample_data[4].id)  # model-c

        # Vote between model-a and model-b
        client.post(
            "/api/votes",
            json={
                "sample_a_id": model_a_sample,
                "sample_b_id": model_b_sample,
                "winner": "A",
            },
        )

        # Vote between model-a and model-c
        client.post(
            "/api/votes",
            json={
                "sample_a_id": model_a_sample,
                "sample_b_id": model_c_sample,
                "winner": "B",
            },
        )

        # Vote between model-b and model-c
        client.post(
            "/api/votes",
            json={
                "sample_a_id": model_b_sample,
                "sample_b_id": model_c_sample,
                "winner": "tie",
            },
        )

        response = client.get("/api/progress")
        assert response.status_code == 200

        data = response.json()
        assert data["votes_completed"] == 3
        assert data["unique_pairs_judged"] == 3

    def test_progress_calculates_total_possible_pairs(
        self,
        client: TestClient,
        temp_data_dir: Path,
    ) -> None:
        """Test that total_possible_pairs is calculated correctly."""
        # Create samples from 2 models with 2 valid samples each
        samples = [
            ArtSample(
                id=uuid4(),
                model_id="model-a",
                prompt_text="test1",
                category="test",
                attempt_number=1,
                raw_output="```\ntest\n```",
                sanitized_output="test",
                is_valid=True,
            ),
            ArtSample(
                id=uuid4(),
                model_id="model-a",
                prompt_text="test2",
                category="test",
                attempt_number=1,
                raw_output="```\ntest\n```",
                sanitized_output="test",
                is_valid=True,
            ),
            ArtSample(
                id=uuid4(),
                model_id="model-b",
                prompt_text="test1",
                category="test",
                attempt_number=1,
                raw_output="```\ntest\n```",
                sanitized_output="test",
                is_valid=True,
            ),
            ArtSample(
                id=uuid4(),
                model_id="model-b",
                prompt_text="test2",
                category="test",
                attempt_number=1,
                raw_output="```\ntest\n```",
                sanitized_output="test",
                is_valid=True,
            ),
        ]
        populate_database(temp_data_dir, samples)

        response = client.get("/api/progress")
        assert response.status_code == 200

        data = response.json()
        # 4 samples total, but only pairs from different models count
        # model-a samples can pair with model-b samples: 2 * 2 = 4 pairs
        assert data["total_possible_pairs"] == 4

    def test_progress_excludes_invalid_samples_from_total(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that invalid samples are excluded from total_possible_pairs."""
        populate_database(temp_data_dir, sample_data)

        # sample_data has 5 valid samples and 1 invalid
        # The invalid sample (model-c, single_object category) should be excluded
        response = client.get("/api/progress")
        assert response.status_code == 200

        # Verify that total_possible_pairs doesn't count pairs with invalid samples
        # Valid samples: 2 from model-a, 2 from model-b, 1 from model-c
        # Cross-model pairs: (2*2) + (2*1) + (2*1) = 4 + 2 + 2 = 8
        data = response.json()
        assert data["total_possible_pairs"] == 8

    def test_progress_includes_category_breakdown(
        self,
        client: TestClient,
        temp_data_dir: Path,
    ) -> None:
        """Test that progress includes breakdown by category."""
        # Create samples in multiple categories
        samples = [
            ArtSample(
                id=uuid4(),
                model_id="model-a",
                prompt_text="Draw a cat",
                category="single_animal",
                attempt_number=1,
                raw_output="```\ncat\n```",
                sanitized_output="cat",
                is_valid=True,
            ),
            ArtSample(
                id=uuid4(),
                model_id="model-b",
                prompt_text="Draw a cat",
                category="single_animal",
                attempt_number=1,
                raw_output="```\ncat\n```",
                sanitized_output="cat",
                is_valid=True,
            ),
            ArtSample(
                id=uuid4(),
                model_id="model-a",
                prompt_text="Draw a tree",
                category="single_object",
                attempt_number=1,
                raw_output="```\ntree\n```",
                sanitized_output="tree",
                is_valid=True,
            ),
            ArtSample(
                id=uuid4(),
                model_id="model-b",
                prompt_text="Draw a tree",
                category="single_object",
                attempt_number=1,
                raw_output="```\ntree\n```",
                sanitized_output="tree",
                is_valid=True,
            ),
        ]
        populate_database(temp_data_dir, samples)

        response = client.get("/api/progress")
        assert response.status_code == 200

        data = response.json()
        assert "by_category" in data
        assert "single_animal" in data["by_category"]
        assert "single_object" in data["by_category"]

        # Each category has 1 sample per model, so 1 cross-model pair each
        assert data["by_category"]["single_animal"]["total_possible_pairs"] == 1
        assert data["by_category"]["single_object"]["total_possible_pairs"] == 1

    def test_progress_category_breakdown_counts_votes(
        self,
        client: TestClient,
        temp_data_dir: Path,
    ) -> None:
        """Test that category breakdown correctly counts votes per category."""
        # Create samples in multiple categories
        sample_animal_a = ArtSample(
            id=uuid4(),
            model_id="model-a",
            prompt_text="Draw a cat",
            category="single_animal",
            attempt_number=1,
            raw_output="```\ncat\n```",
            sanitized_output="cat",
            is_valid=True,
        )
        sample_animal_b = ArtSample(
            id=uuid4(),
            model_id="model-b",
            prompt_text="Draw a cat",
            category="single_animal",
            attempt_number=1,
            raw_output="```\ncat\n```",
            sanitized_output="cat",
            is_valid=True,
        )
        sample_object_a = ArtSample(
            id=uuid4(),
            model_id="model-a",
            prompt_text="Draw a tree",
            category="single_object",
            attempt_number=1,
            raw_output="```\ntree\n```",
            sanitized_output="tree",
            is_valid=True,
        )
        sample_object_b = ArtSample(
            id=uuid4(),
            model_id="model-b",
            prompt_text="Draw a tree",
            category="single_object",
            attempt_number=1,
            raw_output="```\ntree\n```",
            sanitized_output="tree",
            is_valid=True,
        )
        samples = [sample_animal_a, sample_animal_b, sample_object_a, sample_object_b]
        populate_database(temp_data_dir, samples)

        # Submit votes in single_animal category
        client.post(
            "/api/votes",
            json={
                "sample_a_id": str(sample_animal_a.id),
                "sample_b_id": str(sample_animal_b.id),
                "winner": "A",
            },
        )
        client.post(
            "/api/votes",
            json={
                "sample_a_id": str(sample_animal_a.id),
                "sample_b_id": str(sample_animal_b.id),
                "winner": "B",
            },
        )

        # Submit one vote in single_object category
        client.post(
            "/api/votes",
            json={
                "sample_a_id": str(sample_object_a.id),
                "sample_b_id": str(sample_object_b.id),
                "winner": "tie",
            },
        )

        response = client.get("/api/progress")
        assert response.status_code == 200

        data = response.json()
        assert data["votes_completed"] == 3
        assert data["by_category"]["single_animal"]["votes_completed"] == 2
        assert data["by_category"]["single_animal"]["unique_pairs_judged"] == 1
        assert data["by_category"]["single_object"]["votes_completed"] == 1
        assert data["by_category"]["single_object"]["unique_pairs_judged"] == 1

    def test_progress_response_model(self) -> None:
        """Test ProgressResponse model creation."""
        progress = ProgressResponse(
            votes_completed=50,
            unique_pairs_judged=45,
            total_possible_pairs=1000,
            by_category={
                "single_animal": CategoryProgress(
                    votes_completed=25,
                    unique_pairs_judged=20,
                    total_possible_pairs=500,
                )
            },
        )
        assert progress.votes_completed == 50
        assert progress.unique_pairs_judged == 45
        assert progress.total_possible_pairs == 1000
        assert "single_animal" in progress.by_category
        assert progress.by_category["single_animal"].votes_completed == 25

    def test_category_progress_model(self) -> None:
        """Test CategoryProgress model creation."""
        cat_progress = CategoryProgress(
            votes_completed=10,
            unique_pairs_judged=8,
            total_possible_pairs=100,
        )
        assert cat_progress.votes_completed == 10
        assert cat_progress.unique_pairs_judged == 8
        assert cat_progress.total_possible_pairs == 100


class TestProgressHelperFunctions:
    """Tests for progress calculation helper functions."""

    def test_calculate_total_possible_pairs_empty(self, matchup_service: MatchupService) -> None:
        """Test total pairs calculation with no samples."""
        assert matchup_service._calculate_total_possible_pairs([]) == 0

    def test_calculate_total_possible_pairs_single_model(
        self, matchup_service: MatchupService
    ) -> None:
        """Test total pairs calculation with single model (no cross-model pairs)."""
        samples = [
            ArtSample(
                id=uuid4(),
                model_id="model-a",
                prompt_text="test",
                category="test",
                attempt_number=i,
                raw_output="",
                sanitized_output="",
                is_valid=True,
            )
            for i in range(5)
        ]
        # All samples from same model, so no valid cross-model pairs
        assert matchup_service._calculate_total_possible_pairs(samples) == 0

    def test_calculate_total_possible_pairs_two_models(
        self, matchup_service: MatchupService
    ) -> None:
        """Test total pairs calculation with two models."""
        samples = []
        # 3 samples from model-a, 2 samples from model-b
        for i in range(3):
            samples.append(
                ArtSample(
                    id=uuid4(),
                    model_id="model-a",
                    prompt_text="test",
                    category="test",
                    attempt_number=i,
                    raw_output="",
                    sanitized_output="",
                    is_valid=True,
                )
            )
        for i in range(2):
            samples.append(
                ArtSample(
                    id=uuid4(),
                    model_id="model-b",
                    prompt_text="test",
                    category="test",
                    attempt_number=i,
                    raw_output="",
                    sanitized_output="",
                    is_valid=True,
                )
            )
        # Cross-model pairs: 3 * 2 = 6
        assert matchup_service._calculate_total_possible_pairs(samples) == 6

    def test_calculate_total_possible_pairs_three_models(
        self, matchup_service: MatchupService
    ) -> None:
        """Test total pairs calculation with three models."""
        samples = []
        # 2 samples each from 3 models
        for model in ["model-a", "model-b", "model-c"]:
            for i in range(2):
                samples.append(
                    ArtSample(
                        id=uuid4(),
                        model_id=model,
                        prompt_text="test",
                        category="test",
                        attempt_number=i,
                        raw_output="",
                        sanitized_output="",
                        is_valid=True,
                    )
                )
        # Cross-model pairs: (2*2) + (2*2) + (2*2) = 4 + 4 + 4 = 12
        assert matchup_service._calculate_total_possible_pairs(samples) == 12

    def test_get_unique_model_pairs_judged_empty(self, matchup_service: MatchupService) -> None:
        """Test unique pairs judged with no votes."""
        assert matchup_service.get_unique_model_pairs_judged([], []) == 0

    def test_get_unique_model_pairs_judged_single_pair(
        self, matchup_service: MatchupService
    ) -> None:
        """Test unique pairs judged with one model pair compared."""
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
        votes = [
            Vote(
                sample_a_id=str(samples[0].id),
                sample_b_id=str(samples[1].id),
                winner="A",
            ),
        ]
        assert matchup_service.get_unique_model_pairs_judged(votes, samples) == 1

    def test_get_unique_model_pairs_judged_duplicate_pair(
        self, matchup_service: MatchupService
    ) -> None:
        """Test unique pairs judged counts each model pair once."""
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
        votes = [
            Vote(
                sample_a_id=str(samples[0].id),
                sample_b_id=str(samples[1].id),
                winner="A",
            ),
            Vote(
                sample_a_id=str(samples[0].id),
                sample_b_id=str(samples[1].id),
                winner="B",
            ),
        ]
        assert matchup_service.get_unique_model_pairs_judged(votes, samples) == 1


class TestHTMXEndpoints:
    """Tests for the HTMX HTML fragment endpoints."""

    def test_htmx_matchup_returns_html(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that HTMX matchup endpoint returns HTML."""
        populate_database(temp_data_dir, sample_data)

        response = client.get("/htmx/matchup")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        # Should contain sample display content
        assert "comparison-container" in response.text or "error" in response.text

    def test_htmx_matchup_contains_hidden_ids(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that HTMX matchup includes hidden sample ID inputs."""
        populate_database(temp_data_dir, sample_data)

        response = client.get("/htmx/matchup")
        assert response.status_code == 200
        # Should contain hidden inputs with sample IDs
        assert 'id="sample-a-id"' in response.text
        assert 'id="sample-b-id"' in response.text

    def test_htmx_matchup_error_with_insufficient_samples(
        self,
        client: TestClient,
        temp_data_dir: Path,
    ) -> None:
        """Test that HTMX matchup returns error HTML with insufficient samples."""
        # Don't populate any samples

        response = client.get("/htmx/matchup")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Not enough valid samples" in response.text

    def test_htmx_progress_returns_html(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that HTMX progress endpoint returns HTML."""
        populate_database(temp_data_dir, sample_data)

        response = client.get("/htmx/progress")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Votes:" in response.text

    def test_htmx_progress_updates_after_vote(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that HTMX progress shows updated counts."""
        populate_database(temp_data_dir, sample_data)

        # Get initial progress
        initial_response = client.get("/htmx/progress")
        assert "Votes: 0" in initial_response.text

        # Submit a vote via JSON API
        sample_a_id = str(sample_data[0].id)
        sample_b_id = str(sample_data[2].id)
        client.post(
            "/api/votes",
            json={
                "sample_a_id": sample_a_id,
                "sample_b_id": sample_b_id,
                "winner": "A",
            },
        )

        # Get updated progress
        updated_response = client.get("/htmx/progress")
        assert "Votes: 1" in updated_response.text

    def test_htmx_vote_submits_and_returns_new_matchup(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that HTMX vote endpoint submits vote and returns new matchup."""
        populate_database(temp_data_dir, sample_data)

        sample_a_id = str(sample_data[0].id)
        sample_b_id = str(sample_data[2].id)

        response = client.post(
            "/htmx/vote",
            data={
                "sample_a_id": sample_a_id,
                "sample_b_id": sample_b_id,
                "winner": "A",
            },
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Verify vote was persisted
        votes = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes) == 1
        assert votes[0].winner == "A"

    def test_htmx_vote_error_with_missing_fields(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that HTMX vote returns error HTML with missing fields."""
        populate_database(temp_data_dir, sample_data)

        response = client.post(
            "/htmx/vote",
            data={
                "sample_a_id": str(sample_data[0].id),
                # Missing sample_b_id and winner
            },
        )
        assert response.status_code == 200
        assert "Missing required fields" in response.text

    def test_htmx_vote_error_with_invalid_winner(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that HTMX vote returns error HTML with invalid winner."""
        populate_database(temp_data_dir, sample_data)

        response = client.post(
            "/htmx/vote",
            data={
                "sample_a_id": str(sample_data[0].id),
                "sample_b_id": str(sample_data[2].id),
                "winner": "invalid",
            },
        )
        assert response.status_code == 200
        assert "Invalid winner value" in response.text

    def test_htmx_undo_removes_last_vote(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that HTMX undo endpoint removes last vote and returns matchup."""
        populate_database(temp_data_dir, sample_data)

        sample_a_id = str(sample_data[0].id)
        sample_b_id = str(sample_data[2].id)

        # Submit a vote
        client.post(
            "/api/votes",
            json={
                "sample_a_id": sample_a_id,
                "sample_b_id": sample_b_id,
                "winner": "A",
            },
        )

        # Verify vote exists
        votes_before = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes_before) == 1

        # Call HTMX undo
        response = client.post("/htmx/undo")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Verify vote was removed
        votes_after = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes_after) == 0

    def test_htmx_undo_error_with_no_votes(
        self,
        client: TestClient,
        temp_data_dir: Path,
    ) -> None:
        """Test that HTMX undo returns error HTML with no votes."""
        response = client.post("/htmx/undo")
        assert response.status_code == 200
        assert "No votes to undo" in response.text

    def test_htmx_undo_error_twice_in_row(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that HTMX undo returns error when called twice in a row."""
        populate_database(temp_data_dir, sample_data)

        sample_a_id = str(sample_data[0].id)
        sample_b_id = str(sample_data[2].id)

        # Submit two votes
        for _ in range(2):
            client.post(
                "/api/votes",
                json={
                    "sample_a_id": sample_a_id,
                    "sample_b_id": sample_b_id,
                    "winner": "A",
                },
            )

        # First undo should succeed
        first_undo = client.post("/htmx/undo")
        assert first_undo.status_code == 200
        assert "error" not in first_undo.text.lower() or "Cannot undo" not in first_undo.text

        # Second undo should return error
        second_undo = client.post("/htmx/undo")
        assert second_undo.status_code == 200
        assert "Cannot undo twice in a row" in second_undo.text

    def test_htmx_prompt_returns_html(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that HTMX prompt endpoint returns HTML."""
        populate_database(temp_data_dir, sample_data)

        response = client.get("/htmx/prompt")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Prompt:" in response.text

    def test_htmx_prompt_error_with_no_samples(
        self,
        client: TestClient,
        temp_data_dir: Path,
    ) -> None:
        """Test that HTMX prompt returns message with no samples."""
        response = client.get("/htmx/prompt")
        assert response.status_code == 200
        assert "No samples available" in response.text


@pytest.fixture
def vlm_evaluation_data() -> list[VLMEvaluation]:
    """Create sample VLM evaluation data for tests."""
    sample_id_1 = uuid4()
    sample_id_2 = uuid4()
    sample_id_3 = uuid4()
    sample_id_4 = uuid4()

    return [
        VLMEvaluation(
            id=uuid4(),
            sample_id=str(sample_id_1),
            vlm_model_id="openai/gpt-4o",
            expected_subject="cat",
            vlm_response="A cat",
            similarity_score=0.95,
            is_correct=True,
        ),
        VLMEvaluation(
            id=uuid4(),
            sample_id=str(sample_id_1),
            vlm_model_id="openai/gpt-4o",
            expected_subject="dog",
            vlm_response="A cat",
            similarity_score=0.3,
            is_correct=False,
        ),
        VLMEvaluation(
            id=uuid4(),
            sample_id=str(sample_id_2),
            vlm_model_id="openai/gpt-4o",
            expected_subject="cat",
            vlm_response="A cat",
            similarity_score=0.92,
            is_correct=True,
        ),
        VLMEvaluation(
            id=uuid4(),
            sample_id=str(sample_id_3),
            vlm_model_id="anthropic/claude-3-5-sonnet",
            expected_subject="cat",
            vlm_response="A cat",
            similarity_score=0.88,
            is_correct=True,
        ),
        VLMEvaluation(
            id=uuid4(),
            sample_id=str(sample_id_4),
            vlm_model_id="anthropic/claude-3-5-sonnet",
            expected_subject="cat",
            vlm_response="A dog",
            similarity_score=0.2,
            is_correct=False,
        ),
    ]


def populate_vlm_evaluations(path: Path, evaluations: list[VLMEvaluation]) -> None:
    """Populate vlm_evaluations.jsonl with test evaluations."""
    eval_path = path / "vlm_evaluations.jsonl"
    for evaluation in evaluations:
        append_jsonl(eval_path, evaluation)


class TestVLMAccuracyEndpoint:
    """Tests for GET /api/vlm-accuracy endpoint."""

    def test_vlm_accuracy_returns_empty_with_no_evaluations(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
    ) -> None:
        """Test that VLM accuracy returns empty objects with no evaluations."""
        populate_database(temp_data_dir, sample_data)

        response = client.get("/api/vlm-accuracy")
        assert response.status_code == 200

        data = response.json()
        assert "by_model" in data
        assert "by_category" in data
        assert data["by_model"] == {}
        assert data["by_category"] == {}

    def test_vlm_accuracy_calculates_per_model_stats(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
        vlm_evaluation_data: list[VLMEvaluation],
    ) -> None:
        """Test that VLM accuracy calculates per-model statistics."""
        populate_database(temp_data_dir, sample_data)

        # Link evaluations to first 4 valid samples
        # sample_data[0,1]: model-a, sample_data[2,3]: model-b
        vlm_evaluation_data[0].sample_id = str(sample_data[0].id)
        vlm_evaluation_data[1].sample_id = str(sample_data[0].id)
        vlm_evaluation_data[2].sample_id = str(sample_data[1].id)
        vlm_evaluation_data[3].sample_id = str(sample_data[2].id)
        vlm_evaluation_data[4].sample_id = str(sample_data[3].id)

        populate_vlm_evaluations(temp_data_dir, vlm_evaluation_data)

        response = client.get("/api/vlm-accuracy")
        assert response.status_code == 200

        data = response.json()
        assert "by_model" in data

        # model-a: 3 evaluations (2 correct, 1 incorrect) = 0.67 accuracy
        assert "model-a" in data["by_model"]
        model_a_stats = data["by_model"]["model-a"]
        assert model_a_stats["total"] == 3
        assert model_a_stats["correct"] == 2
        assert model_a_stats["accuracy"] == 0.67

        # model-b: 2 evaluations (1 correct, 1 incorrect) = 0.5 accuracy
        assert "model-b" in data["by_model"]
        model_b_stats = data["by_model"]["model-b"]
        assert model_b_stats["total"] == 2
        assert model_b_stats["correct"] == 1
        assert model_b_stats["accuracy"] == 0.5

    def test_vlm_accuracy_calculates_per_category_stats(
        self,
        client: TestClient,
        temp_data_dir: Path,
        sample_data: list[ArtSample],
        vlm_evaluation_data: list[VLMEvaluation],
    ) -> None:
        """Test that VLM accuracy calculates per-category statistics."""
        populate_database(temp_data_dir, sample_data)

        # Link evaluations to first 4 valid samples
        vlm_evaluation_data[0].sample_id = str(sample_data[0].id)
        vlm_evaluation_data[1].sample_id = str(sample_data[0].id)
        vlm_evaluation_data[2].sample_id = str(sample_data[1].id)
        vlm_evaluation_data[3].sample_id = str(sample_data[2].id)
        vlm_evaluation_data[4].sample_id = str(sample_data[3].id)

        populate_vlm_evaluations(temp_data_dir, vlm_evaluation_data)

        response = client.get("/api/vlm-accuracy")
        assert response.status_code == 200

        data = response.json()
        assert "by_category" in data

        # single_animal: 5 evaluations (3 correct, 2 incorrect) = 0.6 accuracy
        assert "single_animal" in data["by_category"]
        category_stats = data["by_category"]["single_animal"]
        assert category_stats["total"] == 5
        assert category_stats["correct"] == 3
        assert category_stats["accuracy"] == 0.6

    def test_vlm_accuracy_handles_orphan_evaluations(
        self,
        client: TestClient,
        temp_data_dir: Path,
        vlm_evaluation_data: list[VLMEvaluation],
    ) -> None:
        """Test that VLM accuracy handles evaluations for missing samples."""
        # Don't populate database, but add evaluations
        populate_vlm_evaluations(temp_data_dir, vlm_evaluation_data)

        response = client.get("/api/vlm-accuracy")
        assert response.status_code == 200

        data = response.json()
        # Should return empty since no matching samples exist
        assert data["by_model"] == {}
        assert data["by_category"] == {}

    def test_vlm_accuracy_response_model(self, client: TestClient) -> None:
        """Test that VLM accuracy endpoint response model is valid."""
        response = client.get("/api/vlm-accuracy")
        assert response.status_code == 200

        data = response.json()
        assert "by_model" in data
        assert "by_category" in data
        assert isinstance(data["by_model"], dict)
        assert isinstance(data["by_category"], dict)
