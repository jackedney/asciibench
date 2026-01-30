from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ArtSample(BaseModel):
    model_id: str
    prompt_text: str
    category: str
    attempt_number: int
    raw_output: str
    sanitized_output: str
    is_valid: bool


class Vote(BaseModel):
    sample_a_id: str
    sample_b_id: str
    winner: Literal["A", "B", "tie", "fail"]
    timestamp: datetime = Field(default_factory=datetime.now)


class Model(BaseModel):
    id: str
    name: str


class Prompt(BaseModel):
    text: str
    category: str
    template_type: str
