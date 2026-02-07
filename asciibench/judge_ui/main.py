import json
import logging
import random
import statistics
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from asciibench.analyst.elo import calculate_elo
from asciibench.analyst.stability import generate_stability_report
from asciibench.common.models import ArtSample, Model, VLMEvaluation, Vote
from asciibench.common.persistence import (
    append_jsonl,
    read_jsonl,
    read_jsonl_by_id,
)
from asciibench.common.yaml_config import load_models
from asciibench.judge_ui.matchup_service import MatchupService
from asciibench.judge_ui.undo_service import UndoService

app = FastAPI(title="ASCIIBench Judge UI")
templates = Jinja2Templates(directory="templates")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Data file paths
DATA_DIR = Path("data")
DATABASE_PATH = DATA_DIR / "database.jsonl"
VOTES_PATH = DATA_DIR / "votes.jsonl"
VLM_EVALUATIONS_PATH = DATA_DIR / "vlm_evaluations.jsonl"
RANKINGS_PATH = DATA_DIR / "rankings.json"

# Service instances
matchup_service = MatchupService(database_path=DATABASE_PATH, votes_path=VOTES_PATH)
undo_service = UndoService(votes_path=VOTES_PATH)


class SampleResponse(BaseModel):
    """Response model for a sample in a matchup (excludes model_id for double-blind)."""

    id: str
    sanitized_output: str
    prompt_text: str


class MatchupResponse(BaseModel):
    """Response model for the matchup endpoint."""

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


def _calculate_progress_by_category(
    votes: list[Vote], samples: list[ArtSample]
) -> dict[str, CategoryProgress]:
    """Calculate progress statistics broken down by category."""
    sample_lookup: dict[str, ArtSample] = {str(s.id): s for s in samples}

    samples_by_category: dict[str, list[ArtSample]] = {}
    for sample in samples:
        if sample.is_valid:
            if sample.category not in samples_by_category:
                samples_by_category[sample.category] = []
            samples_by_category[sample.category].append(sample)

    votes_by_category: dict[str, list[Vote]] = {}
    for vote in votes:
        sample_a = sample_lookup.get(vote.sample_a_id)
        if sample_a and sample_a.is_valid:
            category = sample_a.category
            if category not in votes_by_category:
                votes_by_category[category] = []
            votes_by_category[category].append(vote)

    result: dict[str, CategoryProgress] = {}
    for category, category_samples in samples_by_category.items():
        category_votes = votes_by_category.get(category, [])

        unique_pairs = matchup_service.get_unique_model_pairs_judged(
            category_votes, category_samples
        )

        total_pairs = matchup_service._calculate_total_possible_pairs(category_samples)

        result[category] = CategoryProgress(
            votes_completed=len(category_votes),
            unique_pairs_judged=unique_pairs,
            total_possible_pairs=total_pairs,
        )

    return result


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/judge", response_class=HTMLResponse)
async def judge(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("judge.html", {"request": request})


@app.get("/api/matchup", response_model=MatchupResponse)
async def get_matchup() -> MatchupResponse:
    """Get a random matchup of two samples for comparison.

    Returns two samples from different models for blind comparison.
    Prioritizes model pairs that have had fewer comparisons.
    The sample_a and sample_b positions are randomized to prevent position bias.
    Model IDs are excluded from the response to maintain double-blind judging.
    """
    # Load valid samples from database
    all_samples = read_jsonl(DATABASE_PATH, ArtSample)
    valid_samples = [s for s in all_samples if s.is_valid]

    # Check if we have enough samples
    if len(valid_samples) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough valid samples for comparison. "
            f"Found {len(valid_samples)} valid sample(s), need at least 2. "
            "Please run the generator to create more samples.",
        )

    # Select a matchup
    sample_a, sample_b = matchup_service.get_matchup(valid_samples)

    # Randomize which sample is A vs B to prevent position bias
    if random.random() < 0.5:
        sample_a, sample_b = sample_b, sample_a

    # Build response (excluding model_id for double-blind)
    return MatchupResponse(
        sample_a=SampleResponse(
            id=str(sample_a.id),
            sanitized_output=sample_a.sanitized_output,
            prompt_text=sample_a.prompt_text,
        ),
        sample_b=SampleResponse(
            id=str(sample_b.id),
            sanitized_output=sample_b.sanitized_output,
            prompt_text=sample_b.prompt_text,
        ),
        prompt=sample_a.prompt_text,  # Both samples should have the same prompt ideally
    )


@app.post("/api/votes", response_model=VoteResponse)
async def submit_vote(vote_request: VoteRequest) -> VoteResponse:
    """Submit a vote for a matchup comparison.

    Validates that both sample IDs exist in the database and persists
    the vote to data/votes.jsonl.

    Args:
        vote_request: The vote request containing sample IDs and winner

    Returns:
        The saved vote with its generated ID and timestamp

    Raises:
        HTTPException: 404 if either sample ID doesn't exist in database
        HTTPException: 400 if winner value is invalid (handled by Pydantic)
    """
    # Validate that sample_a_id exists in database
    sample_a = read_jsonl_by_id(DATABASE_PATH, vote_request.sample_a_id, ArtSample)
    if sample_a is None:
        raise HTTPException(
            status_code=404,
            detail=f"Sample with ID '{vote_request.sample_a_id}' not found in database",
        )

    # Validate that sample_b_id exists in database
    sample_b = read_jsonl_by_id(DATABASE_PATH, vote_request.sample_b_id, ArtSample)
    if sample_b is None:
        raise HTTPException(
            status_code=404,
            detail=f"Sample with ID '{vote_request.sample_b_id}' not found in database",
        )

    # Create the Vote model (UUID and timestamp are generated automatically)
    vote = Vote(
        sample_a_id=vote_request.sample_a_id,
        sample_b_id=vote_request.sample_b_id,
        winner=vote_request.winner,
    )

    # Persist to votes.jsonl
    append_jsonl(VOTES_PATH, vote)

    # Clear the undo state since a new vote was submitted
    undo_service.record_vote_submitted()

    # Return the saved vote with generated ID and timestamp
    return VoteResponse(
        id=str(vote.id),
        sample_a_id=vote.sample_a_id,
        sample_b_id=vote.sample_b_id,
        winner=vote.winner,
        timestamp=vote.timestamp.isoformat(),
    )


@app.post("/api/undo", response_model=VoteResponse)
async def undo_vote() -> VoteResponse:
    """Undo the most recent vote.

    Removes the last vote from votes.jsonl and returns it for confirmation.
    The operation is atomic - either the entire undo succeeds or fails.

    Undo can only be called once after each vote. Calling undo twice in a
    row without submitting a new vote will return an error.

    Returns:
        The removed vote for confirmation

    Raises:
        HTTPException: 404 if no votes exist
        HTTPException: 400 if undo was already called without a new vote in between
    """
    last_vote = undo_service.undo_vote()

    if last_vote is None:
        if undo_service.last_action_was_undo:
            raise HTTPException(
                status_code=400,
                detail="Cannot undo twice in a row. Please submit a new vote before undoing again.",
            )
        raise HTTPException(
            status_code=404,
            detail="No votes to undo. The vote history is empty.",
        )

    last_vote_id = str(last_vote.id)

    # Return the removed vote for confirmation
    return VoteResponse(
        id=last_vote_id,
        sample_a_id=last_vote.sample_a_id,
        sample_b_id=last_vote.sample_b_id,
        winner=last_vote.winner,
        timestamp=last_vote.timestamp.isoformat(),
    )


@app.get("/api/progress", response_model=ProgressResponse)
async def get_progress() -> ProgressResponse:
    """Get progress statistics for judging.

    Returns the number of votes completed, unique model pairs judged,
    total possible pairs, and a breakdown by category.

    Returns:
        ProgressResponse with votes_completed, unique_pairs_judged,
        total_possible_pairs, and by_category breakdown.
    """
    # Load all samples from database
    all_samples = read_jsonl(DATABASE_PATH, ArtSample)
    valid_samples = [s for s in all_samples if s.is_valid]

    # Load all votes
    votes = read_jsonl(VOTES_PATH, Vote)

    # Calculate overall statistics
    votes_completed = len(votes)
    unique_pairs_judged = matchup_service.get_unique_model_pairs_judged(votes, valid_samples)
    total_possible_pairs = matchup_service._calculate_total_possible_pairs(valid_samples)

    # Calculate per-category breakdown
    by_category = _calculate_progress_by_category(votes, all_samples)

    return ProgressResponse(
        votes_completed=votes_completed,
        unique_pairs_judged=unique_pairs_judged,
        total_possible_pairs=total_possible_pairs,
        by_category=by_category,
    )


# =============================================================================
# HTMX Endpoints - Return HTML fragments for dynamic UI updates
# =============================================================================


@app.get("/htmx/matchup", response_class=HTMLResponse)
async def htmx_get_matchup(request: Request) -> HTMLResponse:
    """HTMX endpoint to get a matchup as HTML fragment.

    Returns rendered HTML for the matchup comparison area.
    On error, returns an error message HTML fragment.
    """
    try:
        # Load valid samples from database
        all_samples = read_jsonl(DATABASE_PATH, ArtSample)
        valid_samples = [s for s in all_samples if s.is_valid]

        # Check if we have enough samples
        if len(valid_samples) < 2:
            return templates.TemplateResponse(
                request,
                "partials/matchup.html",
                {
                    "error": f"Not enough valid samples for comparison. "
                    f"Found {len(valid_samples)} valid sample(s), need at least 2. "
                    "Please run the generator to create more samples.",
                },
            )

        # Select a matchup
        sample_a, sample_b = matchup_service.get_matchup(valid_samples)

        # Randomize which sample is A vs B to prevent position bias
        if random.random() < 0.5:
            sample_a, sample_b = sample_b, sample_a

        return templates.TemplateResponse(
            request,
            "partials/matchup.html",
            {
                "sample_a": {
                    "id": str(sample_a.id),
                    "sanitized_output": sample_a.sanitized_output,
                },
                "sample_b": {
                    "id": str(sample_b.id),
                    "sanitized_output": sample_b.sanitized_output,
                },
                "prompt": sample_a.prompt_text,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            request,
            "partials/matchup.html",
            {"error": str(e)},
        )


@app.get("/htmx/prompt", response_class=HTMLResponse)
async def htmx_get_prompt(request: Request) -> HTMLResponse:
    """HTMX endpoint to get the current prompt as HTML fragment.

    Returns rendered HTML for the prompt display area.
    """
    try:
        # Load valid samples from database
        all_samples = read_jsonl(DATABASE_PATH, ArtSample)
        valid_samples = [s for s in all_samples if s.is_valid]

        if len(valid_samples) < 2:
            return templates.TemplateResponse(
                request,
                "partials/prompt.html",
                {"prompt": "No samples available"},
            )

        # Select a matchup to get the prompt
        sample_a, _ = matchup_service.get_matchup(valid_samples)

        return templates.TemplateResponse(
            request,
            "partials/prompt.html",
            {"prompt": sample_a.prompt_text},
        )
    except Exception as e:
        return templates.TemplateResponse(
            request,
            "partials/prompt.html",
            {"prompt": f"Error: {e}"},
        )


@app.get("/htmx/progress", response_class=HTMLResponse)
async def htmx_get_progress(request: Request) -> HTMLResponse:
    """HTMX endpoint to get progress as HTML fragment.

    Returns rendered HTML for the progress display area.
    """
    # Load all samples from database
    all_samples = read_jsonl(DATABASE_PATH, ArtSample)
    valid_samples = [s for s in all_samples if s.is_valid]

    # Load all votes
    votes = read_jsonl(VOTES_PATH, Vote)

    # Calculate overall statistics
    votes_completed = len(votes)
    unique_pairs_judged = matchup_service.get_unique_model_pairs_judged(votes, valid_samples)
    total_possible_pairs = matchup_service._calculate_total_possible_pairs(valid_samples)

    return templates.TemplateResponse(
        request,
        "partials/progress.html",
        {
            "votes_completed": votes_completed,
            "unique_pairs_judged": unique_pairs_judged,
            "total_possible_pairs": total_possible_pairs,
        },
    )


@app.post("/htmx/vote", response_class=HTMLResponse)
async def htmx_submit_vote(request: Request) -> HTMLResponse:
    """HTMX endpoint to submit a vote and get new matchup as HTML.

    Accepts form data with sample_a_id, sample_b_id, and winner.
    Returns a new matchup HTML fragment after saving the vote.
    """
    try:
        # Parse form data
        form_data = await request.form()
        sample_a_id = form_data.get("sample_a_id")
        sample_b_id = form_data.get("sample_b_id")
        winner = form_data.get("winner")

        if not all([sample_a_id, sample_b_id, winner]):
            return templates.TemplateResponse(
                request,
                "partials/matchup.html",
                {"error": "Missing required fields"},
            )

        # Validate winner value
        if winner not in ["A", "B", "tie", "fail"]:
            return templates.TemplateResponse(
                request,
                "partials/matchup.html",
                {"error": f"Invalid winner value: {winner}"},
            )

        # Validate that sample_a_id exists in database
        sample_a = read_jsonl_by_id(DATABASE_PATH, str(sample_a_id), ArtSample)
        if sample_a is None:
            return templates.TemplateResponse(
                request,
                "partials/matchup.html",
                {"error": f"Sample A not found: {sample_a_id}"},
            )

        # Validate that sample_b_id exists in database
        sample_b = read_jsonl_by_id(DATABASE_PATH, str(sample_b_id), ArtSample)
        if sample_b is None:
            return templates.TemplateResponse(
                request,
                "partials/matchup.html",
                {"error": f"Sample B not found: {sample_b_id}"},
            )

        # Create and save the vote
        vote = Vote(
            sample_a_id=str(sample_a_id),
            sample_b_id=str(sample_b_id),
            winner=winner,  # type: ignore[arg-type]
        )
        append_jsonl(VOTES_PATH, vote)

        # Clear the undo state since a new vote was submitted
        undo_service.record_vote_submitted()

        # Return a new matchup
        return await htmx_get_matchup(request)

    except Exception as e:
        return templates.TemplateResponse(
            request,
            "partials/matchup.html",
            {"error": f"Error submitting vote: {e}"},
        )


@app.post("/htmx/undo", response_class=HTMLResponse)
async def htmx_undo_vote(request: Request) -> HTMLResponse:
    """HTMX endpoint to undo the last vote and return status message.

    Returns the current matchup (unchanged) with an optional message,
    or an error message if undo fails.
    """
    try:
        # Check if last action was already an undo (prevent double-undo)
        if undo_service.last_action_was_undo:
            return templates.TemplateResponse(
                request,
                "partials/matchup.html",
                {
                    "error": "Cannot undo twice in a row. Submit a new vote first.",
                },
            )

        last_vote = undo_service.undo_vote()

        # Check if there was a vote to undo
        if last_vote is None:
            return templates.TemplateResponse(
                request,
                "partials/matchup.html",
                {"error": "No votes to undo."},
            )

        # Return the current matchup (don't load a new one)
        return await htmx_get_matchup(request)

    except Exception as e:
        return templates.TemplateResponse(
            request,
            "partials/matchup.html",
            {"error": f"Error undoing vote: {e}"},
        )


# =============================================================================
# Analytics Endpoints
# =============================================================================


class LeaderboardEntry(BaseModel):
    """Single entry in the leaderboard."""

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


def _calculate_elo_history(
    votes: list[Vote],
    samples: list[ArtSample],
    checkpoint_interval: int = 10,
) -> dict[str, list[EloHistoryPoint]]:
    """Calculate ELO rating history at checkpoints during vote processing."""
    if not votes:
        return {}

    sorted_votes = sorted(votes, key=lambda v: v.timestamp)
    history: dict[str, list[EloHistoryPoint]] = {}
    n_votes = len(sorted_votes)

    for i in range(checkpoint_interval, n_votes + 1, checkpoint_interval):
        partial_votes = sorted_votes[:i]
        ratings = calculate_elo(partial_votes, samples)

        for model_id, rating in ratings.items():
            if model_id not in history:
                history[model_id] = []
            history[model_id].append(EloHistoryPoint(vote_count=i, elo=rating))

    # Add final checkpoint if not already included
    if n_votes % checkpoint_interval != 0:
        final_ratings = calculate_elo(sorted_votes, samples)
        for model_id, rating in final_ratings.items():
            if model_id not in history:
                history[model_id] = []
            history[model_id].append(EloHistoryPoint(vote_count=n_votes, elo=rating))

    return history


def _calculate_head_to_head(
    votes: list[Vote],
    samples: list[ArtSample],
) -> dict[str, dict[str, HeadToHeadRecord]]:
    """Calculate head-to-head win/loss records between all model pairs."""
    sample_to_model: dict[str, str] = {str(s.id): s.model_id for s in samples}

    # Initialize win matrix
    win_counts: dict[str, dict[str, int]] = {}

    for vote in votes:
        if vote.winner == "fail":
            continue

        model_a = sample_to_model.get(vote.sample_a_id)
        model_b = sample_to_model.get(vote.sample_b_id)

        if not model_a or not model_b or model_a == model_b:
            continue

        # Initialize nested dicts if needed
        if model_a not in win_counts:
            win_counts[model_a] = {}
        if model_b not in win_counts:
            win_counts[model_b] = {}

        # Count wins
        if vote.winner == "A":
            win_counts[model_a][model_b] = win_counts[model_a].get(model_b, 0) + 1
        elif vote.winner == "B":
            win_counts[model_b][model_a] = win_counts[model_b].get(model_a, 0) + 1
        # Ties don't count as wins for either

    # Convert to HeadToHeadRecord format
    result: dict[str, dict[str, HeadToHeadRecord]] = {}
    all_models = set(win_counts.keys())

    for model_a in all_models:
        result[model_a] = {}
        for model_b in all_models:
            if model_a == model_b:
                continue
            wins = win_counts.get(model_a, {}).get(model_b, 0)
            losses = win_counts.get(model_b, {}).get(model_a, 0)
            result[model_a][model_b] = HeadToHeadRecord(wins=wins, losses=losses)

    return result


def _build_analytics_data(
    votes: list[Vote],
    samples: list[ArtSample],
    n_bootstrap: int = 200,
) -> AnalyticsResponse:
    """Build complete analytics data from votes and samples."""
    # Calculate current ELO ratings
    elo_ratings = calculate_elo(votes, samples)

    # Build leaderboard
    sorted_ratings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    leaderboard = [
        LeaderboardEntry(rank=i + 1, model_id=model_id, elo=round(elo, 1))
        for i, (model_id, elo) in enumerate(sorted_ratings)
    ]

    # Generate stability report (use fewer iterations for UI responsiveness)
    stability_report = generate_stability_report(votes, samples, n_bootstrap=n_bootstrap, seed=42)

    # Build per-model stability data
    models_stability: dict[str, ModelStabilityData] = {}
    for model_id, elo in elo_ratings.items():
        ci = stability_report.confidence_intervals.get(model_id)
        rs = stability_report.ranking_stability.get(model_id)
        conv = stability_report.convergence.get(model_id)

        models_stability[model_id] = ModelStabilityData(
            elo=round(elo, 1),
            confidence_interval=ConfidenceIntervalData(
                ci_lower=round(ci.ci_lower, 1),
                ci_upper=round(ci.ci_upper, 1),
                ci_width=round(ci.ci_width, 1),
            )
            if ci
            else None,
            rank_stability_pct=round(rs.rank_stability_pct * 100, 1) if rs else None,
            is_converged=conv.is_converged if conv else None,
        )

    stability = StabilityData(
        score=round(stability_report.stability_score, 1),
        is_stable=stability_report.is_stable_for_publication,
        warnings=stability_report.stability_warnings,
        models=models_stability,
    )

    # Calculate ELO history
    elo_history = _calculate_elo_history(votes, samples)

    # Calculate head-to-head matrix
    head_to_head = _calculate_head_to_head(votes, samples)

    return AnalyticsResponse(
        leaderboard=leaderboard,
        stability=stability,
        elo_history=elo_history,
        head_to_head=head_to_head,
        total_votes=len(votes),
    )


@app.get("/api/analytics", response_model=AnalyticsResponse)
async def get_analytics() -> AnalyticsResponse:
    """Get complete analytics data for the leaderboard and stability metrics.

    Returns leaderboard rankings, stability metrics, ELO history over time,
    and head-to-head comparison matrix.
    """
    # Load all samples and votes
    all_samples = read_jsonl(DATABASE_PATH, ArtSample)
    valid_samples = [s for s in all_samples if s.is_valid]
    votes = read_jsonl(VOTES_PATH, Vote)

    if not votes:
        # Return empty analytics if no votes
        return AnalyticsResponse(
            leaderboard=[],
            stability=StabilityData(
                score=0.0,
                is_stable=False,
                warnings=["No votes to analyze"],
                models={},
            ),
            elo_history={},
            head_to_head={},
            total_votes=0,
        )

    return _build_analytics_data(votes, valid_samples)


def _calculate_vlm_accuracy(
    evaluations: list[VLMEvaluation],
    samples: list[ArtSample],
) -> dict[str, ModelAccuracyStats]:
    """Calculate VLM accuracy statistics per ASCII-generating model.

    Args:
        evaluations: List of VLM evaluations
        samples: List of all samples from database

    Returns:
        Dictionary mapping model_id to accuracy statistics
    """
    sample_lookup: dict[str, ArtSample] = {str(s.id): s for s in samples}

    by_model: dict[str, dict[str, int]] = {}

    for evaluation in evaluations:
        sample = sample_lookup.get(evaluation.sample_id)
        if not sample:
            continue

        model_id = sample.model_id
        if model_id not in by_model:
            by_model[model_id] = {"total": 0, "correct": 0}

        by_model[model_id]["total"] += 1
        if evaluation.is_correct:
            by_model[model_id]["correct"] += 1

    result: dict[str, ModelAccuracyStats] = {}
    for model_id, stats in by_model.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        result[model_id] = ModelAccuracyStats(
            total=stats["total"],
            correct=stats["correct"],
            accuracy=round(accuracy, 2),
        )

    return result


def _calculate_category_accuracy(
    evaluations: list[VLMEvaluation],
    samples: list[ArtSample],
) -> dict[str, CategoryAccuracyStats]:
    """Calculate VLM accuracy statistics per category.

    Args:
        evaluations: List of VLM evaluations
        samples: List of all samples from database

    Returns:
        Dictionary mapping category to accuracy statistics
    """
    sample_lookup: dict[str, ArtSample] = {str(s.id): s for s in samples}

    by_category: dict[str, dict[str, int]] = {}

    for evaluation in evaluations:
        sample = sample_lookup.get(evaluation.sample_id)
        if not sample:
            continue

        category = sample.category
        if category not in by_category:
            by_category[category] = {"total": 0, "correct": 0}

        by_category[category]["total"] += 1
        if evaluation.is_correct:
            by_category[category]["correct"] += 1

    result: dict[str, CategoryAccuracyStats] = {}
    for category, stats in by_category.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        result[category] = CategoryAccuracyStats(
            total=stats["total"],
            correct=stats["correct"],
            accuracy=round(accuracy, 2),
        )

    return result


def _calculate_pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """Calculate Pearson correlation coefficient.

    Args:
        x: First list of values
        y: Second list of values

    Returns:
        Pearson correlation coefficient, or None if fewer than 3 data points
    """
    if len(x) < 3 or len(y) < 3:
        return None

    n = len(x)

    if n != len(y):
        raise ValueError("Lists must have the same length")

    if n < 2:
        return None

    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True))

    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)

    denominator = (sum_sq_x * sum_sq_y) ** 0.5

    if denominator == 0:
        return None

    return numerator / denominator


@app.get("/api/vlm-accuracy", response_model=VLMAccuracyResponse)
async def get_vlm_accuracy() -> VLMAccuracyResponse:
    """Get VLM accuracy statistics per model and category.

    Returns accuracy metrics for ASCII-generating models based on VLM evaluations.
    Reads from vlm_evaluations.jsonl joined with database.jsonl.

    Returns:
        VLMAccuracyResponse with by_model and by_category accuracy stats

    Raises:
        HTTPException: 500 if vlm_evaluations.jsonl is malformed
    """
    try:
        evaluations = read_jsonl(VLM_EVALUATIONS_PATH, VLMEvaluation)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading vlm_evaluations.jsonl: {e}",
        ) from e

    samples = read_jsonl(DATABASE_PATH, ArtSample)

    if not evaluations:
        return VLMAccuracyResponse(by_model={}, by_category={})

    by_model = _calculate_vlm_accuracy(evaluations, samples)
    by_category = _calculate_category_accuracy(evaluations, samples)

    return VLMAccuracyResponse(by_model=by_model, by_category=by_category)


@app.get("/api/elo-vlm-correlation", response_model=EloVLMCorrelationResponse)
async def get_elo_vlm_correlation() -> EloVLMCorrelationResponse:
    """Get Elo ratings and VLM accuracy per model for correlation analysis.

    Returns correlation data between human Elo ratings and VLM recognition rates.
    Joins data from rankings.json and vlm_evaluations.jsonl.

    Returns:
        EloVLMCorrelationResponse with correlation coefficient and data array

    Raises:
        HTTPException: 500 if data files are malformed
    """
    try:
        evaluations = read_jsonl(VLM_EVALUATIONS_PATH, VLMEvaluation)
    except FileNotFoundError:
        evaluations = []
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading vlm_evaluations.jsonl: {e}",
        ) from e

    samples = read_jsonl(DATABASE_PATH, ArtSample)

    try:
        with open(RANKINGS_PATH) as f:
            rankings_data = json.load(f)
    except FileNotFoundError:
        return EloVLMCorrelationResponse(correlation_coefficient=None, data=[])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading rankings.json: {e}",
        ) from e

    try:
        models = load_models()
    except FileNotFoundError:
        models = []
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading models.yaml: {e}",
        ) from e

    if not evaluations or not rankings_data.get("overall_ratings"):
        return EloVLMCorrelationResponse(correlation_coefficient=None, data=[])

    elo_ratings = rankings_data["overall_ratings"]
    vlm_accuracy = _calculate_vlm_accuracy(evaluations, samples)

    model_lookup: dict[str, Model] = {m.id: m for m in models}

    correlation_data: list[CorrelationDataPoint] = []
    elo_values: list[float] = []
    accuracy_values: list[float] = []

    for model_id, elo_rating in elo_ratings.items():
        if model_id in vlm_accuracy:
            model = model_lookup.get(model_id)
            model_name = model.name if model else model_id
            accuracy = vlm_accuracy[model_id].accuracy

            correlation_data.append(
                CorrelationDataPoint(
                    model_id=model_id,
                    model_name=model_name,
                    elo_rating=elo_rating,
                    vlm_accuracy=accuracy,
                )
            )
            elo_values.append(elo_rating)
            accuracy_values.append(accuracy)

    correlation_coefficient = _calculate_pearson_correlation(elo_values, accuracy_values)

    return EloVLMCorrelationResponse(
        correlation_coefficient=correlation_coefficient,
        data=correlation_data,
    )


@app.get("/htmx/analytics", response_class=HTMLResponse)
async def htmx_get_analytics(request: Request) -> HTMLResponse:
    """HTMX endpoint to get analytics dashboard as HTML fragment."""
    try:
        # Load all samples and votes
        all_samples = read_jsonl(DATABASE_PATH, ArtSample)
        valid_samples = [s for s in all_samples if s.is_valid]
        votes = read_jsonl(VOTES_PATH, Vote)

        if not votes:
            return templates.TemplateResponse(
                request,
                "partials/analytics.html",
                {
                    "error": None,
                    "leaderboard": [],
                    "stability": {
                        "score": 0.0,
                        "is_stable": False,
                        "warnings": ["No votes to analyze yet"],
                        "models": {},
                    },
                    "elo_history": {},
                    "elo_history_json": "{}",
                    "head_to_head": {},
                    "models": [],
                    "total_votes": 0,
                },
            )

        analytics = _build_analytics_data(votes, valid_samples)

        # Get sorted model list for head-to-head matrix
        models = [entry.model_id for entry in analytics.leaderboard]

        # Convert elo_history to JSON for Chart.js
        elo_history_json = json.dumps(
            {
                model_id: [{"x": p.vote_count, "y": p.elo} for p in points]
                for model_id, points in analytics.elo_history.items()
            }
        )

        return templates.TemplateResponse(
            request,
            "partials/analytics.html",
            {
                "error": None,
                "leaderboard": analytics.leaderboard,
                "stability": analytics.stability,
                "elo_history": analytics.elo_history,
                "elo_history_json": elo_history_json,
                "head_to_head": analytics.head_to_head,
                "models": models,
                "total_votes": analytics.total_votes,
            },
        )
    except Exception:
        logging.exception("Error generating analytics")
        return templates.TemplateResponse(
            request,
            "partials/analytics.html",
            {
                "error": "An error occurred while generating analytics. Please try again.",
                "leaderboard": [],
                "stability": {"score": 0, "is_stable": False, "warnings": [], "models": {}},
                "elo_history": {},
                "elo_history_json": "{}",
                "head_to_head": {},
                "models": [],
                "total_votes": 0,
            },
        )


@app.get("/htmx/vlm-accuracy", response_class=HTMLResponse)
async def htmx_get_vlm_accuracy(request: Request) -> HTMLResponse:
    """HTMX endpoint to get VLM accuracy table as HTML fragment."""
    try:
        # Load evaluations and samples
        evaluations = read_jsonl(VLM_EVALUATIONS_PATH, VLMEvaluation)
        samples = read_jsonl(DATABASE_PATH, ArtSample)

        # Load models for display names
        try:
            models = load_models()
        except (FileNotFoundError, Exception):
            models = []

        model_lookup: dict[str, Model] = {m.id: m for m in models}

        if not evaluations:
            return templates.TemplateResponse(
                request,
                "partials/vlm_accuracy.html",
                {"vlm_accuracy_data": []},
            )

        # Calculate accuracy per model
        by_model = _calculate_vlm_accuracy(evaluations, samples)

        # Sort by accuracy descending and convert to list for template
        sorted_models = sorted(by_model.items(), key=lambda x: x[1].accuracy, reverse=True)

        vlm_accuracy_data = [
            {
                "model_id": model_id,
                "model_name": model_lookup.get(model_id, Model(id=model_id, name=model_id)).name,
                "total": stats.total,
                "correct": stats.correct,
                "accuracy": stats.accuracy,
            }
            for model_id, stats in sorted_models
        ]

        return templates.TemplateResponse(
            request,
            "partials/vlm_accuracy.html",
            {"vlm_accuracy_data": vlm_accuracy_data},
        )
    except Exception:
        logging.exception("Error generating VLM accuracy")
        return templates.TemplateResponse(
            request,
            "partials/vlm_accuracy.html",
            {"vlm_accuracy_data": [], "error": "Failed to load VLM accuracy data."},
        )
