from datetime import UTC, datetime
from functools import partial
from typing import Literal, Protocol, TypedDict
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

# Helper for timezone-aware timestamps
_utc_now = partial(datetime.now, tz=UTC)


class ProgressCallback(Protocol):
    """Protocol for progress callbacks during sample generation.

    Callback is invoked before each sample generation with details about
    the model, prompt, attempt, and remaining work.

    Example:
        >>> def on_progress(model_id: str, prompt_text: str, attempt: int, remaining: int) -> None:
        ...     print(f"Generating {model_id} attempt {attempt}, {remaining} remaining")
        >>>
        >>> generate_samples(
        ...     models, prompts, config,
        ...     progress_callback=on_progress
        ... )
    """

    def __call__(
        self, model_id: str, prompt_text: str, attempt_number: int, total_remaining: int
    ) -> None:
        """Called before each sample generation.

        Args:
            model_id: ID of the model being used
            prompt_text: Text of the prompt being generated
            attempt_number: Current attempt number (1-indexed)
            total_remaining: Number of tasks remaining
        """
        ...


class StatsCallback(Protocol):
    """Protocol for statistics callbacks after sample generation.

    Callback is invoked after each sample generation completes with
    validity status and cost information.

    Example:
        >>> def on_stats(is_valid: bool, cost: float | None) -> None:
        ...     print(f"Sample valid: {is_valid}, cost: ${cost or 0:.4f}")
        >>>
        >>> generate_samples(
        ...     models, prompts, config,
        ...     stats_callback=on_stats
        ... )
    """

    def __call__(self, is_valid: bool, cost: float | None) -> None:
        """Called after each sample generation completes.

        Args:
            is_valid: Whether the generated sample is valid
            cost: Cost of the API call (None if not available)
        """
        ...


class EvaluationProgressCallback(Protocol):
    """Protocol for progress callbacks during VLM evaluation.

    Callback is invoked after each evaluation completes with details about
    progress and current VLM model.

    Example:
        >>> def on_eval_progress(processed: int, total: int, model_id: str) -> None:
        ...     print(f"{processed}/{total} evaluated with {model_id}")
        >>>
        >>> await run_evaluation(
        ...     progress_callback=on_eval_progress
        ... )
    """

    def __call__(self, total_processed: int, total_tasks: int, vlm_model_id: str) -> None:
        """Called after each evaluation completes.

        Args:
            total_processed: Number of evaluations completed
            total_tasks: Total number of evaluation tasks
            vlm_model_id: ID of the VLM model used for this evaluation
        """
        ...


class ModelPairCounts(TypedDict):
    """Type definition for model pair comparison counts.

    Maps sorted model pair tuples to the number of comparisons between them.
    Used in MatchupService and ModelPairSelector to track which model pairs
    have been compared how many times.
    """

    # Key is a tuple of (model_a_id, model_b_id) in sorted order
    # Value is the count of comparisons between the two models
    # Example: {("model_a", "model_b"): 5}


class LeaderboardEntryData(TypedDict):
    """Type definition for a single entry in the leaderboard rankings.

    Contains rank position, model information, and computed metrics.
    Used in analyst/main.py for displaying leaderboard tables.
    """

    rank: int
    model: str
    elo: int
    comparisons: int
    win_rate: float


class RankingsData(TypedDict):
    """Type definition for rankings data structure.

    Contains overall Elo ratings for all models.
    Used in judge_ui/main.py for correlation analysis.
    """

    overall_ratings: dict[str, float]


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

    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost: float | None

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


class Matchup(BaseModel):
    """Represents a single matchup between two models on a specific prompt."""

    id: UUID = Field(default_factory=uuid4)
    model_a_id: str
    model_b_id: str
    prompt_text: str
    prompt_category: str
    sample_a_id: str | None = None
    sample_b_id: str | None = None
    is_judged: bool = False
    vote_id: str | None = None


class RoundState(BaseModel):
    """Represents the state of a tournament round including matchups and Elo ratings."""

    id: UUID = Field(default_factory=uuid4)
    round_number: int
    matchups: list[Matchup]
    elo_snapshot: dict[str, float] = Field(default_factory=dict)
    generation_complete: bool = False
    all_judged: bool = False
    created_at: datetime = Field(default_factory=_utc_now)
