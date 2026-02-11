"""End-to-end integration tests for the full ASCIIBench pipeline.

Tests the complete workflow from sample generation through judging to analysis.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from asciibench.analyst.elo import calculate_elo
from asciibench.analyst.leaderboard import (
    export_rankings_csv,
    export_rankings_json,
    generate_leaderboard,
)
from asciibench.common.config import GenerationConfig
from asciibench.common.models import ArtSample, Model, OpenRouterResponse, Prompt, Vote
from asciibench.common.persistence import (
    append_jsonl,
    read_jsonl,
    read_jsonl_by_id,
)
from asciibench.generator.client import OpenRouterClient
from asciibench.generator.sampler import generate_samples
from asciibench.judge_ui.main import app


def _mock_generate_response(
    model_id: str, prompt: str, config: object = None
) -> OpenRouterResponse:
    """Generate ASCII art based on prompt content."""
    if "cat" in prompt.lower():
        return OpenRouterResponse(text="```\n/\\_/\\\n( o.o )\n > ^ <\n```")
    elif "dog" in prompt.lower():
        return OpenRouterResponse(text="```\n/ \\__\n(    @\\___\n```\n")
    elif "tree" in prompt.lower():
        return OpenRouterResponse(text="```\n  /\\  \n /  \\ \n/____\\\n```")
    else:
        return OpenRouterResponse(text="```\n? ? ?\n```")


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock OpenRouterClient with predefined responses."""
    client = MagicMock(spec=OpenRouterClient)

    client.generate.side_effect = _mock_generate_response
    client.generate_async = AsyncMock(side_effect=_mock_generate_response)
    return client


@pytest.fixture
def sample_models() -> list[Model]:
    """Create sample model configuration."""
    return [
        Model(id="openai/gpt-4o", name="GPT-4o"),
        Model(id="anthropic/claude-3-opus", name="Claude 3 Opus"),
    ]


@pytest.fixture
def sample_prompts() -> list[Prompt]:
    """Create sample prompt configuration."""
    return [
        Prompt(text="Draw a cat", category="single_animal", template_type="animal"),
        Prompt(text="Draw a dog", category="single_animal", template_type="animal"),
    ]


@pytest.fixture
def temp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up temporary data directory for all modules."""
    import asciibench.judge_ui.main as judge_main
    from asciibench.common.repository import DataRepository
    from asciibench.judge_ui.analytics_service import AnalyticsService
    from asciibench.judge_ui.matchup_service import MatchupService
    from asciibench.judge_ui.progress_service import ProgressService
    from asciibench.judge_ui.undo_service import UndoService

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(judge_main, "DATA_DIR", data_dir)
    monkeypatch.setattr(judge_main, "DATABASE_PATH", data_dir / "database.jsonl")
    monkeypatch.setattr(judge_main, "VOTES_PATH", data_dir / "votes.jsonl")

    repo = DataRepository(data_dir=data_dir)
    matchup_service = MatchupService(
        database_path=data_dir / "database.jsonl", votes_path=data_dir / "votes.jsonl"
    )
    undo_service = UndoService(votes_path=data_dir / "votes.jsonl")
    progress_service = ProgressService(repo=repo, matchup_service=matchup_service)
    analytics_service = AnalyticsService(repo=repo)

    monkeypatch.setattr(judge_main, "repo", repo)
    monkeypatch.setattr(judge_main, "matchup_service", matchup_service)
    monkeypatch.setattr(judge_main, "undo_service", undo_service)
    monkeypatch.setattr(judge_main, "progress_service", progress_service)
    monkeypatch.setattr(judge_main, "analytics_service", analytics_service)
    monkeypatch.setattr(judge_main, "VLM_EVALUATIONS_PATH", data_dir / "vlm_evaluations.jsonl")
    monkeypatch.setattr(judge_main, "_vlm_evaluation_service", None)
    monkeypatch.setattr(judge_main, "_vlm_init_attempted", False)

    return data_dir


class TestGeneratorIntegration:
    """Integration tests for the Generator module."""

    @pytest.fixture
    def sample_config(self) -> GenerationConfig:
        """Create sample generation config."""
        return GenerationConfig(
            attempts_per_prompt=2,
            temperature=0.0,
            max_tokens=1000,
        )

    def test_full_generation_pipeline(
        self,
        mock_client: MagicMock,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        temp_data_dir: Path,
        sample_config: GenerationConfig,
    ) -> None:
        """Test complete sample generation pipeline with mocked API."""
        db_path = temp_data_dir / "database.jsonl"

        result = generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=sample_config,
            database_path=db_path,
            client=mock_client,
        )

        # Verify correct number of samples generated
        assert len(result) == 8  # 2 models * 2 prompts * 2 attempts

        # Verify all samples persisted to database.jsonl
        samples = read_jsonl(db_path, ArtSample)
        assert len(samples) == 8

        # Verify sample structure
        for sample in samples:
            assert sample.id is not None
            assert sample.model_id in ["openai/gpt-4o", "anthropic/claude-3-opus"]
            assert sample.prompt_text in ["Draw a cat", "Draw a dog"]
            assert sample.category == "single_animal"
            assert sample.attempt_number in [1, 2]
            assert sample.is_valid is True
            assert sample.timestamp is not None
            assert len(sample.raw_output) > 0
            assert len(sample.sanitized_output) > 0

    def test_idempotency_running_twice(
        self,
        mock_client: MagicMock,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        temp_data_dir: Path,
        sample_config: GenerationConfig,
    ) -> None:
        """Test that running generator twice produces same results (idempotent)."""
        db_path = temp_data_dir / "database.jsonl"

        # First run
        result1 = generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=sample_config,
            database_path=db_path,
            client=mock_client,
        )
        assert len(result1) == 8

        samples_after_first = read_jsonl(db_path, ArtSample)
        assert len(samples_after_first) == 8

        # Reset mock to track calls
        mock_client.reset_mock()

        # Second run - should be idempotent
        result2 = generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=sample_config,
            database_path=db_path,
            client=mock_client,
        )

        # No new samples should be generated
        assert len(result2) == 0
        assert mock_client.generate_async.call_count == 0

        # Database should still have exactly 8 samples
        samples_after_second = read_jsonl(db_path, ArtSample)
        assert len(samples_after_second) == 8

        # Verify content is identical
        assert samples_after_first == samples_after_second


class TestJudgeUIIntegration:
    """Integration tests for the Judge UI module."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client for FastAPI app."""
        return TestClient(app)

    def test_complete_voting_workflow(
        self,
        client: TestClient,
        temp_data_dir: Path,
    ) -> None:
        """Test complete workflow: get matchup, submit vote, verify persistence."""
        # Create sample data
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
        ]
        db_path = temp_data_dir / "database.jsonl"
        for sample in samples:
            append_jsonl(db_path, sample)

        # Get matchup
        matchup_response = client.get("/api/matchup")
        assert matchup_response.status_code == 200
        matchup_data = matchup_response.json()

        assert "sample_a" in matchup_data
        assert "sample_b" in matchup_data
        assert "prompt" in matchup_data

        sample_a_id = matchup_data["sample_a"]["id"]
        sample_b_id = matchup_data["sample_b"]["id"]

        # Submit vote
        vote_response = client.post(
            "/api/votes",
            json={
                "sample_a_id": sample_a_id,
                "sample_b_id": sample_b_id,
                "winner": "A",
            },
        )
        assert vote_response.status_code == 200
        vote_data = vote_response.json()

        assert "id" in vote_data
        assert vote_data["sample_a_id"] == sample_a_id
        assert vote_data["sample_b_id"] == sample_b_id
        assert vote_data["winner"] == "A"
        assert "timestamp" in vote_data

        # Verify vote persisted to votes.jsonl
        votes = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes) == 1
        assert str(votes[0].id) == vote_data["id"]
        assert votes[0].sample_a_id == sample_a_id
        assert votes[0].sample_b_id == sample_b_id
        assert votes[0].winner == "A"

    def test_undo_removes_last_vote(
        self,
        client: TestClient,
        temp_data_dir: Path,
    ) -> None:
        """Test undo workflow: vote, undo, verify vote removed."""
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
                timestamp=datetime.now(),
            ),
            ArtSample(
                id=uuid4(),
                model_id="model-b",
                prompt_text="Draw a cat",
                category="single_animal",
                attempt_number=1,
                raw_output="```\ncat2\n```",
                sanitized_output="cat2",
                is_valid=True,
                timestamp=datetime.now(),
            ),
        ]
        db_path = temp_data_dir / "database.jsonl"
        for sample in samples:
            append_jsonl(db_path, sample)

        # Submit a vote
        matchup = client.get("/api/matchup").json()
        vote_response = client.post(
            "/api/votes",
            json={
                "sample_a_id": matchup["sample_a"]["id"],
                "sample_b_id": matchup["sample_b"]["id"],
                "winner": "B",
            },
        )
        assert vote_response.status_code == 200
        vote_id = vote_response.json()["id"]

        # Verify vote exists
        votes_before = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes_before) == 1

        # Undo the vote
        undo_response = client.post("/api/undo")
        assert undo_response.status_code == 200
        undo_data = undo_response.json()

        assert undo_data["id"] == vote_id
        assert undo_data["winner"] == "B"

        # Verify vote was removed
        votes_after = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes_after) == 0

    def test_multiple_votes_and_undo(
        self,
        client: TestClient,
        temp_data_dir: Path,
    ) -> None:
        """Test submitting multiple votes and undoing only the last one."""
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
        ]
        db_path = temp_data_dir / "database.jsonl"
        for sample in samples:
            append_jsonl(db_path, sample)

        # Submit three votes
        for winner in ["A", "B", "tie"]:
            matchup = client.get("/api/matchup").json()
            response = client.post(
                "/api/votes",
                json={
                    "sample_a_id": matchup["sample_a"]["id"],
                    "sample_b_id": matchup["sample_b"]["id"],
                    "winner": winner,
                },
            )
            assert response.status_code == 200

        # Verify 3 votes
        votes_before = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes_before) == 3

        # Undo last vote
        undo_response = client.post("/api/undo")
        assert undo_response.status_code == 200
        assert undo_response.json()["winner"] == "tie"

        # Verify only 2 votes remain
        votes_after = read_jsonl(temp_data_dir / "votes.jsonl", Vote)
        assert len(votes_after) == 2
        assert votes_after[0].winner == "A"
        assert votes_after[1].winner == "B"


class TestAnalystIntegration:
    """Integration tests for the Analyst module."""

    def test_full_analysis_pipeline(
        self,
        temp_data_dir: Path,
    ) -> None:
        """Test complete analysis pipeline with samples and votes."""
        db_path = temp_data_dir / "database.jsonl"
        votes_path = temp_data_dir / "votes.jsonl"

        # Create sample data
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
        ]
        for sample in samples:
            append_jsonl(db_path, sample)

        # Create votes (model-a wins 3 times)
        votes = []
        for _ in range(3):
            vote = Vote(
                sample_a_id=str(samples[0].id),
                sample_b_id=str(samples[1].id),
                winner="A",
            )
            votes.append(vote)
            append_jsonl(votes_path, vote)

        # Load and calculate ratings
        loaded_votes = read_jsonl(votes_path, Vote)
        loaded_samples = read_jsonl(db_path, ArtSample)

        ratings = calculate_elo(loaded_votes, loaded_samples)

        # Verify ratings calculated correctly
        assert "model-a" in ratings
        assert "model-b" in ratings
        assert ratings["model-a"] > 1500  # model-a won, should have higher rating
        assert ratings["model-b"] < 1500  # model-b lost, should have lower rating

        # Generate leaderboard
        leaderboard = generate_leaderboard(loaded_votes, loaded_samples, ratings)
        assert "model-a" in leaderboard
        assert "model-b" in leaderboard
        assert ratings["model-a"] > ratings["model-b"]

        # Export to JSON
        json_path = temp_data_dir / "rankings.json"
        export_rankings_json(loaded_votes, loaded_samples, ratings, json_path)
        assert json_path.exists()
        json_content = json_path.read_text()
        assert "model-a" in json_content
        assert "model-b" in json_content

        # Export to CSV
        csv_path = temp_data_dir / "rankings.csv"
        export_rankings_csv(loaded_votes, loaded_samples, ratings, csv_path)
        assert csv_path.exists()
        csv_content = csv_path.read_text()
        assert "model" in csv_content
        assert "model-a" in csv_content
        assert "model-b" in csv_content

    def test_empty_votes_generates_valid_leaderboard(
        self,
        temp_data_dir: Path,
    ) -> None:
        """Test that analyst handles empty votes gracefully."""
        db_path = temp_data_dir / "database.jsonl"
        votes_path = temp_data_dir / "votes.jsonl"

        # Create samples but no votes
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
                timestamp=datetime.now(),
            ),
        ]
        for sample in samples:
            append_jsonl(db_path, sample)

        # votes.jsonl doesn't exist - should get empty list
        try:
            loaded_votes = read_jsonl(votes_path, Vote)
        except FileNotFoundError:
            loaded_votes = []
        loaded_samples = read_jsonl(db_path, ArtSample)

        ratings = calculate_elo(loaded_votes, loaded_samples)
        assert ratings == {}

        # Should still generate valid leaderboard
        leaderboard = generate_leaderboard(loaded_votes, loaded_samples, ratings)
        assert isinstance(leaderboard, str)
        assert len(leaderboard) > 0


class TestFullPipelineIntegration:
    """End-to-end tests for the complete pipeline."""

    @pytest.fixture
    def sample_config(self) -> GenerationConfig:
        """Create sample generation config."""
        return GenerationConfig(
            attempts_per_prompt=2,
            temperature=0.0,
            max_tokens=1000,
        )

    def test_full_pipeline_with_mocked_responses(
        self,
        mock_client: MagicMock,
        sample_models: list[Model],
        sample_prompts: list[Prompt],
        temp_data_dir: Path,
        sample_config: GenerationConfig,
    ) -> None:
        """Test complete pipeline from generation to analysis with mocked LLM responses."""
        db_path = temp_data_dir / "database.jsonl"
        votes_path = temp_data_dir / "votes.jsonl"

        # Step 1: Generate samples
        generated_samples = generate_samples(
            models=sample_models,
            prompts=sample_prompts,
            config=sample_config,
            database_path=db_path,
            client=mock_client,
        )

        assert len(generated_samples) == 8
        loaded_samples = read_jsonl(db_path, ArtSample)
        assert len(loaded_samples) == 8

        # Step 2: Judge samples (simulate voting)
        # Vote for model-a (first sample) vs model-b (second sample)
        vote_a = Vote(
            sample_a_id=str(loaded_samples[0].id),
            sample_b_id=str(loaded_samples[4].id),  # Different model
            winner="A",
        )
        append_jsonl(votes_path, vote_a)

        # Vote for model-b
        vote_b = Vote(
            sample_a_id=str(loaded_samples[1].id),
            sample_b_id=str(loaded_samples[5].id),  # Different model
            winner="B",
        )
        append_jsonl(votes_path, vote_b)

        # Step 3: Analyze votes
        loaded_votes = read_jsonl(votes_path, Vote)
        ratings = calculate_elo(loaded_votes, loaded_samples)

        # Verify both models have ratings
        assert "openai/gpt-4o" in ratings
        assert "anthropic/claude-3-opus" in ratings

        # Generate outputs
        leaderboard = generate_leaderboard(loaded_votes, loaded_samples, ratings)
        assert "Leaderboard" in leaderboard

        json_path = temp_data_dir / "rankings.json"
        export_rankings_json(loaded_votes, loaded_samples, ratings, json_path)
        assert json_path.exists()

        csv_path = temp_data_dir / "rankings.csv"
        export_rankings_csv(loaded_votes, loaded_samples, ratings, csv_path)
        assert csv_path.exists()

        # Verify all data persisted correctly
        assert len(read_jsonl(db_path, ArtSample)) == 8
        assert len(read_jsonl(votes_path, Vote)) == 2


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_corrupted_jsonl_raises_validation_error(
        self,
        temp_data_dir: Path,
    ) -> None:
        """Test that corrupted JSONL raises a clear validation error."""
        from pydantic_core import ValidationError

        db_path = temp_data_dir / "database.jsonl"

        # Write valid sample
        valid_sample = ArtSample(
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
        append_jsonl(db_path, valid_sample)

        # Corrupt the file by appending invalid JSON
        with open(db_path, "a") as f:
            f.write("{invalid json}\n")

        # Read should raise ValidationError with clear message
        with pytest.raises(ValidationError, match="Invalid JSON"):
            read_jsonl(db_path, ArtSample)

    def test_sample_by_id_with_corrupted_file(
        self,
        temp_data_dir: Path,
    ) -> None:
        """Test read_jsonl_by_id with corrupted file."""
        db_path = temp_data_dir / "database.jsonl"

        # Write valid sample
        valid_sample = ArtSample(
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
        append_jsonl(db_path, valid_sample)
        target_id = valid_sample.id

        # Corrupt the file
        with open(db_path, "a") as f:
            f.write("{invalid json}\n")

        # Should still find the valid sample
        found = read_jsonl_by_id(db_path, target_id, ArtSample)
        assert found is not None
        assert found.id == target_id

    def test_vote_with_invalid_sample_id(
        self,
        temp_data_dir: Path,
    ) -> None:
        """Test that voting with invalid sample ID is rejected."""
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
                timestamp=datetime.now(),
            ),
        ]
        db_path = temp_data_dir / "database.jsonl"
        for sample in samples:
            append_jsonl(db_path, sample)

        client = TestClient(app)

        # Try to vote with non-existent sample ID
        response = client.post(
            "/api/votes",
            json={
                "sample_a_id": str(uuid4()),  # Non-existent
                "sample_b_id": str(samples[0].id),
                "winner": "A",
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
