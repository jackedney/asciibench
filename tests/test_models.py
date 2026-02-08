from datetime import UTC, datetime
from typing import Literal, cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from asciibench.common.models import (
    ArtSample,
    DemoResult,
    Model,
    OpenRouterResponse,
    Prompt,
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
