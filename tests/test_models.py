from datetime import UTC, datetime
from typing import Literal, cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from asciibench.common.models import (
    ArtSample,
    DemoResult,
    Matchup,
    Model,
    OpenRouterResponse,
    Prompt,
    RoundState,
    VLMEvaluation,
    Vote,
)


def test_art_sample_valid():
    sample = ArtSample(
        model_id="openai/gpt-4o",
        prompt_text="Draw a cat",
        category="animal",
        attempt_number=1,
        raw_output="```\n/_/\n( o.o )\n > ^ <\n```",
        sanitized_output="/_/\n( o.o )\n > ^ <",
        is_valid=True,
    )
    assert sample.model_id == "openai/gpt-4o"
    assert sample.prompt_text == "Draw a cat"
    assert sample.category == "animal"
    assert sample.attempt_number == 1
    assert sample.is_valid is True


def test_vote_valid_winners():
    for winner in ["A", "B", "tie", "fail"]:
        vote = Vote(
            sample_a_id="sample1",
            sample_b_id="sample2",
            winner=cast(Literal["A", "B", "tie", "fail"], winner),
        )
        assert vote.winner == winner


def test_vote_invalid_winner():
    with pytest.raises(ValidationError) as exc_info:
        Vote(
            sample_a_id="sample1",
            sample_b_id="sample2",
            winner="X",  # type: ignore[arg-type]
        )
    exc = exc_info.value
    assert isinstance(exc, ValidationError)
    errors = exc.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("winner",)
    assert errors[0]["type"] == "literal_error"


def test_vote_timestamp_default():
    before = datetime.now(tz=UTC)
    vote = Vote(sample_a_id="sample1", sample_b_id="sample2", winner="A")
    after = datetime.now(tz=UTC)
    assert before <= vote.timestamp <= after


def test_model_valid():
    model = Model(id="openai/gpt-4o", name="GPT-4o")
    assert model.id == "openai/gpt-4o"
    assert model.name == "GPT-4o"


def test_prompt_valid():
    prompt = Prompt(text="Draw a cat", category="animal", template_type="simple")
    assert prompt.text == "Draw a cat"
    assert prompt.category == "animal"
    assert prompt.template_type == "simple"


def test_art_sample_uuid_generation():
    """ArtSample generates a unique UUID automatically."""
    sample = ArtSample(
        model_id="openai/gpt-4o",
        prompt_text="Draw a cat",
        category="animal",
        attempt_number=1,
        raw_output="```\n/_/\n```",
        sanitized_output="/_/",
        is_valid=True,
    )
    assert isinstance(sample.id, UUID)


def test_art_sample_timestamp_default():
    """ArtSample generates a timestamp automatically."""
    before = datetime.now(tz=UTC)
    sample = ArtSample(
        model_id="openai/gpt-4o",
        prompt_text="Draw a cat",
        category="animal",
        attempt_number=1,
        raw_output="```\n/_/\n```",
        sanitized_output="/_/",
        is_valid=True,
    )
    after = datetime.now(tz=UTC)
    assert before <= sample.timestamp <= after


def test_two_samples_have_different_uuids():
    """Two samples created sequentially have different UUIDs."""
    sample1 = ArtSample(
        model_id="openai/gpt-4o",
        prompt_text="Draw a cat",
        category="animal",
        attempt_number=1,
        raw_output="```\n/_/\n```",
        sanitized_output="/_/",
        is_valid=True,
    )
    sample2 = ArtSample(
        model_id="openai/gpt-4o",
        prompt_text="Draw a cat",
        category="animal",
        attempt_number=2,
        raw_output="```\n/_/\n```",
        sanitized_output="/_/",
        is_valid=True,
    )
    assert sample1.id != sample2.id


def test_vote_uuid_generation():
    """Vote generates a unique UUID automatically."""
    vote = Vote(sample_a_id="sample1", sample_b_id="sample2", winner="A")
    assert isinstance(vote.id, UUID)


def test_two_votes_have_different_uuids():
    """Two votes created sequentially have different UUIDs."""
    vote1 = Vote(sample_a_id="sample1", sample_b_id="sample2", winner="A")
    vote2 = Vote(sample_a_id="sample3", sample_b_id="sample4", winner="B")
    assert vote1.id != vote2.id


def test_demo_result_with_cost_and_tokens():
    """DemoResult with output_tokens and cost fields."""
    result = DemoResult(
        model_id="openai/gpt-4o-mini",
        model_name="GPT-4o Mini",
        ascii_output="skeleton",
        is_valid=True,
        timestamp=datetime(2026, 1, 30, 20, 0, 0),
        output_tokens=1234,
        cost=0.001234,
    )
    assert result.model_id == "openai/gpt-4o-mini"
    assert result.model_name == "GPT-4o Mini"
    assert result.output_tokens == 1234
    assert result.cost == 0.001234


def test_demo_result_without_cost_and_tokens():
    """DemoResult without output_tokens and cost fields defaults to None."""
    result = DemoResult(
        model_id="openai/gpt-4o-mini",
        model_name="GPT-4o Mini",
        ascii_output="skeleton",
        is_valid=True,
        timestamp=datetime(2026, 1, 30, 20, 0, 0),
    )
    assert result.output_tokens is None
    assert result.cost is None


def test_demo_result_serialization_includes_cost_and_tokens():
    """DemoResult serialization includes output_tokens and cost fields."""
    result = DemoResult(
        model_id="openai/gpt-4o-mini",
        model_name="GPT-4o Mini",
        ascii_output="skeleton",
        is_valid=True,
        timestamp=datetime(2026, 1, 30, 20, 0, 0),
        output_tokens=1234,
        cost=0.001234,
    )
    data = result.model_dump(mode="json")
    assert "output_tokens" in data
    assert data["output_tokens"] == 1234
    assert "cost" in data
    assert data["cost"] == 0.001234


def test_demo_result_serialization_with_none_values():
    """DemoResult serialization handles None values for cost/tokens."""
    result = DemoResult(
        model_id="openai/gpt-4o-mini",
        model_name="GPT-4o Mini",
        ascii_output="skeleton",
        is_valid=True,
        timestamp=datetime(2026, 1, 30, 20, 0, 0),
        output_tokens=None,
        cost=None,
    )
    data = result.model_dump(mode="json")
    assert "output_tokens" in data
    assert data["output_tokens"] is None
    assert "cost" in data
    assert data["cost"] is None


def test_art_sample_with_cost_and_tokens():
    """ArtSample with output_tokens and cost fields."""
    sample = ArtSample(
        model_id="openai/gpt-4o",
        prompt_text="Draw a cat",
        category="animal",
        attempt_number=1,
        raw_output="```\ncat\n```",
        sanitized_output="cat",
        is_valid=True,
        output_tokens=100,
        cost=0.0001,
    )
    assert sample.model_id == "openai/gpt-4o"
    assert sample.output_tokens == 100
    assert sample.cost == 0.0001


def test_art_sample_without_cost_and_tokens():
    """ArtSample without output_tokens and cost fields defaults to None."""
    sample = ArtSample(
        model_id="openai/gpt-4o",
        prompt_text="Draw a cat",
        category="animal",
        attempt_number=1,
        raw_output="```\ncat\n```",
        sanitized_output="cat",
        is_valid=True,
    )
    assert sample.output_tokens is None
    assert sample.cost is None


def test_art_sample_serialization_includes_cost_and_tokens():
    """ArtSample serialization includes output_tokens and cost fields."""
    sample = ArtSample(
        model_id="openai/gpt-4o",
        prompt_text="Draw a cat",
        category="animal",
        attempt_number=1,
        raw_output="```\ncat\n```",
        sanitized_output="cat",
        is_valid=True,
        output_tokens=200,
        cost=0.0002,
    )
    data = sample.model_dump(mode="json")
    assert "output_tokens" in data
    assert data["output_tokens"] == 200
    assert "cost" in data
    assert data["cost"] == 0.0002


def test_open_router_response_with_all_fields():
    """OpenRouterResponse with all fields populated."""
    response = OpenRouterResponse(
        text="Generated text",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        cost=0.001234,
    )
    assert response.text == "Generated text"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 20
    assert response.total_tokens == 30
    assert response.cost == 0.001234


def test_open_router_response_with_none_values():
    """OpenRouterResponse with None values for optional fields."""
    response = OpenRouterResponse(
        text="Generated text",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cost=None,
    )
    assert response.text == "Generated text"
    assert response.prompt_tokens is None
    assert response.completion_tokens is None
    assert response.total_tokens is None
    assert response.cost is None


def test_open_router_response_text_only():
    """OpenRouterResponse with only text field."""
    response = OpenRouterResponse(text="Generated text")
    assert response.text == "Generated text"
    assert response.prompt_tokens is None
    assert response.completion_tokens is None
    assert response.total_tokens is None
    assert response.cost is None


def test_vlm_evaluation_with_all_fields():
    """VLMEvaluation with all required fields."""
    evaluation = VLMEvaluation(
        sample_id="abc",
        vlm_model_id="gpt-4",
        expected_subject="cat",
        vlm_response="a cat",
        similarity_score=0.95,
        is_correct=True,
        cost=0.001,
    )
    assert evaluation.sample_id == "abc"
    assert evaluation.vlm_model_id == "gpt-4"
    assert evaluation.expected_subject == "cat"
    assert evaluation.vlm_response == "a cat"
    assert evaluation.similarity_score == 0.95
    assert evaluation.is_correct is True
    assert evaluation.cost == 0.001


def test_vlm_evaluation_serialization():
    """VLMEvaluation serializes to JSON correctly."""
    evaluation = VLMEvaluation(
        sample_id="abc",
        vlm_model_id="gpt-4",
        expected_subject="cat",
        vlm_response="a cat",
        similarity_score=0.95,
        is_correct=True,
    )
    data = evaluation.model_dump(mode="json")
    assert data["sample_id"] == "abc"
    assert data["vlm_model_id"] == "gpt-4"
    assert data["expected_subject"] == "cat"
    assert data["vlm_response"] == "a cat"
    assert data["similarity_score"] == 0.95
    assert data["is_correct"] is True
    assert "id" in data
    assert "timestamp" in data


def test_vlm_evaluation_similarity_score_invalid_high():
    """VLMEvaluation with similarity_score > 1.0 raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        VLMEvaluation(
            sample_id="abc",
            vlm_model_id="gpt-4",
            expected_subject="cat",
            vlm_response="a cat",
            similarity_score=1.5,
            is_correct=True,
        )
    exc = exc_info.value
    assert isinstance(exc, ValidationError)
    errors = exc.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("similarity_score",)
    assert "must be between 0 and 1" in errors[0]["msg"]


def test_vlm_evaluation_similarity_score_invalid_low():
    """VLMEvaluation with similarity_score < 0.0 raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        VLMEvaluation(
            sample_id="abc",
            vlm_model_id="gpt-4",
            expected_subject="cat",
            vlm_response="a cat",
            similarity_score=-0.1,
            is_correct=False,
        )
    exc = exc_info.value
    assert isinstance(exc, ValidationError)
    errors = exc.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("similarity_score",)
    assert "must be between 0 and 1" in errors[0]["msg"]


def test_vlm_evaluation_similarity_score_boundary_values():
    """VLMEvaluation accepts similarity_score of exactly 0.0 and 1.0."""
    evaluation_low = VLMEvaluation(
        sample_id="abc",
        vlm_model_id="gpt-4",
        expected_subject="cat",
        vlm_response="dog",
        similarity_score=0.0,
        is_correct=False,
    )
    assert evaluation_low.similarity_score == 0.0

    evaluation_high = VLMEvaluation(
        sample_id="abc",
        vlm_model_id="gpt-4",
        expected_subject="cat",
        vlm_response="a cat",
        similarity_score=1.0,
        is_correct=True,
    )
    assert evaluation_high.similarity_score == 1.0


def test_vlm_evaluation_uuid_generation():
    """VLMEvaluation generates a unique UUID automatically."""
    evaluation = VLMEvaluation(
        sample_id="abc",
        vlm_model_id="gpt-4",
        expected_subject="cat",
        vlm_response="a cat",
        similarity_score=0.95,
        is_correct=True,
    )
    assert isinstance(evaluation.id, UUID)


def test_vlm_evaluation_timestamp_default():
    """VLMEvaluation generates a timestamp automatically."""
    before = datetime.now(tz=UTC)
    evaluation = VLMEvaluation(
        sample_id="abc",
        vlm_model_id="gpt-4",
        expected_subject="cat",
        vlm_response="a cat",
        similarity_score=0.95,
        is_correct=True,
    )
    after = datetime.now(tz=UTC)
    assert before <= evaluation.timestamp <= after


def test_vlm_evaluation_optional_cost():
    """VLMEvaluation cost field is optional."""
    evaluation_with_cost = VLMEvaluation(
        sample_id="abc",
        vlm_model_id="gpt-4",
        expected_subject="cat",
        vlm_response="a cat",
        similarity_score=0.95,
        is_correct=True,
        cost=0.001,
    )
    assert evaluation_with_cost.cost == 0.001

    evaluation_without_cost = VLMEvaluation(
        sample_id="abc",
        vlm_model_id="gpt-4",
        expected_subject="cat",
        vlm_response="a cat",
        similarity_score=0.95,
        is_correct=True,
    )
    assert evaluation_without_cost.cost is None


def test_vlm_evaluation_serialization_with_optional_cost():
    """VLMEvaluation serialization includes cost field when present."""
    evaluation_with_cost = VLMEvaluation(
        sample_id="abc",
        vlm_model_id="gpt-4",
        expected_subject="cat",
        vlm_response="a cat",
        similarity_score=0.95,
        is_correct=True,
        cost=0.001,
    )
    data = evaluation_with_cost.model_dump(mode="json")
    assert "cost" in data
    assert data["cost"] == 0.001

    evaluation_without_cost = VLMEvaluation(
        sample_id="abc",
        vlm_model_id="gpt-4",
        expected_subject="cat",
        vlm_response="a cat",
        similarity_score=0.95,
        is_correct=True,
    )
    data = evaluation_without_cost.model_dump(mode="json")
    assert "cost" in data
    assert data["cost"] is None


def test_matchup_valid():
    """Matchup with required fields creates valid instance with auto-generated UUID."""
    matchup = Matchup(
        model_a_id="openai/gpt-4",
        model_b_id="anthropic/claude-3",
        prompt_text="Draw a cat",
        prompt_category="animal",
    )
    assert matchup.model_a_id == "openai/gpt-4"
    assert matchup.model_b_id == "anthropic/claude-3"
    assert matchup.prompt_text == "Draw a cat"
    assert matchup.prompt_category == "animal"
    assert isinstance(matchup.id, UUID)
    assert matchup.sample_a_id is None
    assert matchup.sample_b_id is None
    assert matchup.is_judged is False
    assert matchup.vote_id is None


def test_matchup_uuid_generation():
    """Matchup generates a unique UUID automatically."""
    matchup = Matchup(
        model_a_id="openai/gpt-4",
        model_b_id="anthropic/claude-3",
        prompt_text="Draw a cat",
        prompt_category="animal",
    )
    assert isinstance(matchup.id, UUID)


def test_two_matchups_have_different_uuids():
    """Two matchups created sequentially have different UUIDs."""
    matchup1 = Matchup(
        model_a_id="openai/gpt-4",
        model_b_id="anthropic/claude-3",
        prompt_text="Draw a cat",
        prompt_category="animal",
    )
    matchup2 = Matchup(
        model_a_id="openai/gpt-4",
        model_b_id="anthropic/claude-3",
        prompt_text="Draw a dog",
        prompt_category="animal",
    )
    assert matchup1.id != matchup2.id


def test_matchup_with_sample_ids():
    """Matchup with sample_a_id and sample_b_id fields."""
    matchup = Matchup(
        model_a_id="openai/gpt-4",
        model_b_id="anthropic/claude-3",
        prompt_text="Draw a cat",
        prompt_category="animal",
        sample_a_id="sample-1",
        sample_b_id="sample-2",
    )
    assert matchup.sample_a_id == "sample-1"
    assert matchup.sample_b_id == "sample-2"


def test_matchup_with_judged_status():
    """Matchup with is_judged and vote_id fields."""
    matchup = Matchup(
        model_a_id="openai/gpt-4",
        model_b_id="anthropic/claude-3",
        prompt_text="Draw a cat",
        prompt_category="animal",
        is_judged=True,
        vote_id="vote-1",
    )
    assert matchup.is_judged is True
    assert matchup.vote_id == "vote-1"


def test_matchup_serialization_round_trip():
    """Matchup serializes to/from JSON correctly for JSONL persistence."""
    matchup = Matchup(
        model_a_id="openai/gpt-4",
        model_b_id="anthropic/claude-3",
        prompt_text="Draw a cat",
        prompt_category="animal",
        sample_a_id="sample-1",
        sample_b_id="sample-2",
        is_judged=True,
        vote_id="vote-1",
    )
    json_data = matchup.model_dump_json()
    restored_matchup = Matchup.model_validate_json(json_data)
    assert restored_matchup.model_a_id == matchup.model_a_id
    assert restored_matchup.model_b_id == matchup.model_b_id
    assert restored_matchup.prompt_text == matchup.prompt_text
    assert restored_matchup.prompt_category == matchup.prompt_category
    assert restored_matchup.sample_a_id == matchup.sample_a_id
    assert restored_matchup.sample_b_id == matchup.sample_b_id
    assert restored_matchup.is_judged == matchup.is_judged
    assert restored_matchup.vote_id == matchup.vote_id
    assert restored_matchup.id == matchup.id


def test_round_state_valid():
    """RoundState with required fields creates valid instance with defaults."""
    matchup1 = Matchup(
        model_a_id="openai/gpt-4",
        model_b_id="anthropic/claude-3",
        prompt_text="Draw a cat",
        prompt_category="animal",
    )
    matchup2 = Matchup(
        model_a_id="openai/gpt-4",
        model_b_id="google/gemini-pro",
        prompt_text="Draw a dog",
        prompt_category="animal",
    )
    round_state = RoundState(
        round_number=1,
        matchups=[matchup1, matchup2],
    )
    assert round_state.round_number == 1
    assert len(round_state.matchups) == 2
    assert isinstance(round_state.id, UUID)
    assert round_state.elo_snapshot == {}
    assert round_state.generation_complete is False
    assert round_state.all_judged is False
    assert isinstance(round_state.created_at, datetime)


def test_round_state_uuid_generation():
    """RoundState generates a unique UUID automatically."""
    matchup = Matchup(
        model_a_id="openai/gpt-4",
        model_b_id="anthropic/claude-3",
        prompt_text="Draw a cat",
        prompt_category="animal",
    )
    round_state = RoundState(round_number=1, matchups=[matchup])
    assert isinstance(round_state.id, UUID)


def test_round_state_with_elo_snapshot():
    """RoundState with elo_snapshot field."""
    matchup = Matchup(
        model_a_id="openai/gpt-4",
        model_b_id="anthropic/claude-3",
        prompt_text="Draw a cat",
        prompt_category="animal",
    )
    round_state = RoundState(
        round_number=1,
        matchups=[matchup],
        elo_snapshot={"openai/gpt-4": 1500.0, "anthropic/claude-3": 1480.0},
    )
    assert round_state.elo_snapshot == {"openai/gpt-4": 1500.0, "anthropic/claude-3": 1480.0}


def test_round_state_with_generation_complete():
    """RoundState with generation_complete and all_judged fields."""
    matchup = Matchup(
        model_a_id="openai/gpt-4",
        model_b_id="anthropic/claude-3",
        prompt_text="Draw a cat",
        prompt_category="animal",
    )
    round_state = RoundState(
        round_number=1,
        matchups=[matchup],
        generation_complete=True,
        all_judged=True,
    )
    assert round_state.generation_complete is True
    assert round_state.all_judged is True


def test_round_state_created_at_default():
    """RoundState generates a timestamp automatically."""
    before = datetime.now(tz=UTC)
    matchup = Matchup(
        model_a_id="openai/gpt-4",
        model_b_id="anthropic/claude-3",
        prompt_text="Draw a cat",
        prompt_category="animal",
    )
    round_state = RoundState(round_number=1, matchups=[matchup])
    after = datetime.now(tz=UTC)
    assert before <= round_state.created_at <= after


def test_round_state_without_required_matchups_raises_validation_error():
    """RoundState without required matchups raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        RoundState(round_number=1)  # type: ignore[call-arg]
    exc = exc_info.value
    assert isinstance(exc, ValidationError)
    errors = exc.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("matchups",)
    assert errors[0]["type"] == "missing"


def test_round_state_serialization_round_trip():
    """RoundState serializes to/from JSON correctly for JSONL persistence."""
    matchup1 = Matchup(
        model_a_id="openai/gpt-4",
        model_b_id="anthropic/claude-3",
        prompt_text="Draw a cat",
        prompt_category="animal",
    )
    matchup2 = Matchup(
        model_a_id="openai/gpt-4",
        model_b_id="google/gemini-pro",
        prompt_text="Draw a dog",
        prompt_category="animal",
    )
    round_state = RoundState(
        round_number=1,
        matchups=[matchup1, matchup2],
        elo_snapshot={
            "openai/gpt-4": 1500.0,
            "anthropic/claude-3": 1480.0,
            "google/gemini-pro": 1490.0,
        },
        generation_complete=True,
        all_judged=False,
    )
    json_data = round_state.model_dump_json()
    restored_round_state = RoundState.model_validate_json(json_data)
    assert restored_round_state.round_number == round_state.round_number
    assert len(restored_round_state.matchups) == len(round_state.matchups)
    assert restored_round_state.elo_snapshot == round_state.elo_snapshot
    assert restored_round_state.generation_complete == round_state.generation_complete
    assert restored_round_state.all_judged == round_state.all_judged
    assert restored_round_state.id == round_state.id
    assert restored_round_state.created_at == round_state.created_at


def test_two_round_states_have_different_uuids():
    """Two round states created sequentially have different UUIDs."""
    matchup = Matchup(
        model_a_id="openai/gpt-4",
        model_b_id="anthropic/claude-3",
        prompt_text="Draw a cat",
        prompt_category="animal",
    )
    round_state1 = RoundState(round_number=1, matchups=[matchup])
    round_state2 = RoundState(round_number=2, matchups=[matchup])
    assert round_state1.id != round_state2.id
