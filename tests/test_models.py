from datetime import datetime
from typing import Literal, cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from asciibench.common.models import ArtSample, Model, Prompt, Vote


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
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("winner",)
    assert errors[0]["type"] == "literal_error"


def test_vote_timestamp_default():
    before = datetime.now()
    vote = Vote(sample_a_id="sample1", sample_b_id="sample2", winner="A")
    after = datetime.now()
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
    before = datetime.now()
    sample = ArtSample(
        model_id="openai/gpt-4o",
        prompt_text="Draw a cat",
        category="animal",
        attempt_number=1,
        raw_output="```\n/_/\n```",
        sanitized_output="/_/",
        is_valid=True,
    )
    after = datetime.now()
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
