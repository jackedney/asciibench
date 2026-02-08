from datetime import UTC, datetime
from functools import partial
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

# Helper for timezone-aware timestamps
_utc_now = partial(datetime.now, tz=UTC)


class ArtSample(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    model_id: str
    prompt_text: str
    category: str
    attempt_number: int
    raw_output: str
    sanitized_output: str
    is_valid: bool
    timestamp: datetime = Field(default_factory=_utc_now)
    output_tokens: int | None = None
    cost: float | None = None


class Vote(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    sample_a_id: str
    sample_b_id: str
    winner: Literal["A", "B", "tie", "fail"]
    timestamp: datetime = Field(default_factory=_utc_now)


class Model(BaseModel):
    id: str
    name: str


class Prompt(BaseModel):
    text: str
    category: str
    template_type: str


class DemoResult(BaseModel):
    model_id: str
    model_name: str
    ascii_output: str
    is_valid: bool
    timestamp: datetime
    error_reason: str | None = None
    raw_output: str | None = None
    output_tokens: int | None = None
    cost: float | None = None


class OpenRouterResponse:
    """Response from OpenRouter API including usage and cost metadata."""

    def __init__(
        self,
        text: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        cost: float | None = None,
    ):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.cost = cost


class VLMEvaluation(BaseModel):
    """Evaluation result from VLM analyzing an ASCII art sample."""

    id: UUID = Field(default_factory=uuid4)
    sample_id: str
    vlm_model_id: str
    expected_subject: str
    vlm_response: str
    similarity_score: float
    is_correct: bool
    timestamp: datetime = Field(default_factory=_utc_now)
    cost: float | None = None

    @field_validator("similarity_score")
    @classmethod
    def validate_similarity_score(cls, v: float) -> float:
        """Validate similarity_score is between 0 and 1."""
        if not 0 <= v <= 1:
            msg = "similarity_score must be between 0 and 1"
            raise ValueError(msg)
        return v
