"""Pydantic API models for Judge UI endpoints."""

from typing import Literal

from pydantic import BaseModel


class SampleResponse(BaseModel):
    """Response model for a sample in a matchup (excludes model_id for double-blind)."""

    id: str
    sanitized_output: str
    prompt_text: str


class MatchupResponse(BaseModel):
    """Response model for matchup endpoint."""

    sample_a: SampleResponse
    sample_b: SampleResponse
    prompt: str


class VoteRequest(BaseModel):
    """Request model for submitting a vote."""

    sample_a_id: str
    sample_b_id: str
    winner: Literal["A", "B", "tie", "fail"]


class VoteResponse(BaseModel):
    """Response model for a submitted vote."""

    id: str
    sample_a_id: str
    sample_b_id: str
    winner: Literal["A", "B", "tie", "fail"]
    timestamp: str


class CategoryProgress(BaseModel):
    """Progress statistics for a single category."""

    votes_completed: int
    unique_pairs_judged: int
    total_possible_pairs: int


class ProgressResponse(BaseModel):
    """Response model for progress tracking."""

    votes_completed: int
    unique_pairs_judged: int
    total_possible_pairs: int
    by_category: dict[str, CategoryProgress]


class LeaderboardEntry(BaseModel):
    """Single entry in leaderboard."""

    rank: int
    model_id: str
    elo: float


class ConfidenceIntervalData(BaseModel):
    """Confidence interval data for a model."""

    ci_lower: float
    ci_upper: float
    ci_width: float


class ModelStabilityData(BaseModel):
    """Stability data for a single model."""

    elo: float
    confidence_interval: ConfidenceIntervalData | None
    rank_stability_pct: float | None
    is_converged: bool | None


class StabilityData(BaseModel):
    """Overall stability metrics."""

    score: float
    is_stable: bool
    warnings: list[str]
    models: dict[str, ModelStabilityData]


class EloHistoryPoint(BaseModel):
    """Single point in ELO history."""

    vote_count: int
    elo: float


class HeadToHeadRecord(BaseModel):
    """Win/loss record between two models."""

    wins: int
    losses: int


class AnalyticsResponse(BaseModel):
    """Complete analytics data response."""

    leaderboard: list[LeaderboardEntry]
    stability: StabilityData
    elo_history: dict[str, list[EloHistoryPoint]]
    head_to_head: dict[str, dict[str, HeadToHeadRecord]]
    total_votes: int


class ModelAccuracyStats(BaseModel):
    """Accuracy statistics for a single model."""

    total: int
    correct: int
    accuracy: float


class CategoryAccuracyStats(BaseModel):
    """Accuracy statistics for a single category."""

    total: int
    correct: int
    accuracy: float


class VLMAccuracyResponse(BaseModel):
    """VLM accuracy statistics response."""

    by_model: dict[str, ModelAccuracyStats]
    by_category: dict[str, CategoryAccuracyStats]


class CorrelationDataPoint(BaseModel):
    """Single data point for Elo-VLM correlation."""

    model_id: str
    model_name: str
    elo_rating: float
    vlm_accuracy: float


class EloVLMCorrelationResponse(BaseModel):
    """Elo-VLM correlation response."""

    correlation_coefficient: float | None
    data: list[CorrelationDataPoint]


__all__ = [
    "AnalyticsResponse",
    "CategoryAccuracyStats",
    "CategoryProgress",
    "ConfidenceIntervalData",
    "CorrelationDataPoint",
    "EloHistoryPoint",
    "EloVLMCorrelationResponse",
    "HeadToHeadRecord",
    "LeaderboardEntry",
    "MatchupResponse",
    "ModelAccuracyStats",
    "ModelStabilityData",
    "ProgressResponse",
    "SampleResponse",
    "StabilityData",
    "VLMAccuracyResponse",
    "VoteRequest",
    "VoteResponse",
]
