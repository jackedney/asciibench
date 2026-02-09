"""FastAPI application for ASCIIBench Judge UI.

This module provides the main web application for the judging interface,
including endpoints for:

- Matchup selection and voting (REST and HTMX endpoints)
- Progress tracking and statistics
- Analytics dashboard with Elo ratings and stability metrics
- VLM accuracy statistics
- Undo functionality for correcting mistakes

Architecture:
    The application uses a service-oriented architecture with dependency injection:
    - AnalyticsService: Computes Elo ratings, stability metrics, and head-to-head stats
    - ProgressService: Calculates progress statistics for judging
    - MatchupService: Selects random matchups prioritizing under-compared model pairs
    - UndoService: Handles vote undo functionality
    - DataRepository: Unified data access for samples, votes, and evaluations

Data flow:
    1. User views two ASCII art samples in a blinded comparison
    2. User votes for winner, which is persisted to data/votes.jsonl
    3. Progress stats update automatically to show completion
    4. Analytics are computed on-demand from votes and samples

HTMX integration:
    The UI uses HTMX for dynamic updates without page reloads.
    HTMX endpoints return HTML fragments for partial page updates.
"""

import json
import logging
import random
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from asciibench.common.models import ArtSample, Model, VLMEvaluation, Vote
from asciibench.common.persistence import (
    append_jsonl,
    read_jsonl,
    read_jsonl_by_id,
)
from asciibench.common.repository import DataRepository
from asciibench.common.yaml_config import load_models
from asciibench.judge_ui.analytics_service import AnalyticsService
from asciibench.judge_ui.api_models import (
    AnalyticsResponse,
    CorrelationDataPoint,
    EloVLMCorrelationResponse,
    MatchupResponse,
    ProgressResponse,
    SampleResponse,
    VLMAccuracyResponse,
    VoteRequest,
    VoteResponse,
)
from asciibench.judge_ui.matchup_service import MatchupService
from asciibench.judge_ui.progress_service import ProgressService
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


@lru_cache(maxsize=1)
def _get_database_indexed(path: Path, mtime: int) -> tuple[list[ArtSample], dict[UUID, ArtSample]]:
    """Load database and return both list and ID-indexed dictionary.

    Cached based on file path and modification time (nanoseconds).
    """
    samples = read_jsonl(path, ArtSample)
    indexed = {s.id: s for s in samples}
    return samples, indexed


def _get_all_samples() -> list[ArtSample]:
    """Get all samples from database with in-memory caching."""
    if not DATABASE_PATH.exists():
        return []
    try:
        mtime = DATABASE_PATH.stat().st_mtime_ns
        samples, _ = _get_database_indexed(DATABASE_PATH, mtime)
        return samples
    except Exception:
        # Fallback to direct read on error (e.g. file deleted between check and read)
        try:
            return read_jsonl(DATABASE_PATH, ArtSample)
        except FileNotFoundError:
            return []


def _get_sample_by_id(sample_id: str | UUID) -> ArtSample | None:
    """Find a sample by ID using the cached database content."""
    if not DATABASE_PATH.exists():
        return None
    try:
        target_id = UUID(str(sample_id)) if not isinstance(sample_id, UUID) else sample_id
        mtime = DATABASE_PATH.stat().st_mtime_ns
        _, indexed = _get_database_indexed(DATABASE_PATH, mtime)
        return indexed.get(target_id)
    except Exception:
        # Fallback to direct read on error
        return read_jsonl_by_id(DATABASE_PATH, sample_id, ArtSample)


# Service instances
# Use cache_ttl=0 to disable caching - votes are written directly to file
# and need to be immediately visible in subsequent reads
repo = DataRepository(data_dir=DATA_DIR, cache_ttl=0)
analytics_service = AnalyticsService(repo=repo)
matchup_service = MatchupService(database_path=DATABASE_PATH, votes_path=VOTES_PATH)
progress_service = ProgressService(repo=repo, matchup_service=matchup_service)
undo_service = UndoService(votes_path=VOTES_PATH)


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
    # Check if database file exists
    if not DATABASE_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail="Database file not found. Please run the generator to create samples.",
        )

    # Load valid samples from database
    all_samples = _get_all_samples()
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
    sample_a = _get_sample_by_id(vote_request.sample_a_id)
    if sample_a is None:
        raise HTTPException(
            status_code=404,
            detail=f"Sample with ID '{vote_request.sample_a_id}' not found in database",
        )

    # Validate that sample_b_id exists in database
    sample_b = _get_sample_by_id(vote_request.sample_b_id)
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
    return progress_service.get_progress()


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
        # Check if database file exists
        if not DATABASE_PATH.exists():
            return templates.TemplateResponse(
                request,
                "partials/matchup.html",
                {"error": "File not found: database.jsonl"},
            )

        # Load valid samples from database
        all_samples = _get_all_samples()
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
        # Check if database file exists
        if not DATABASE_PATH.exists():
            return templates.TemplateResponse(
                request,
                "partials/prompt.html",
                {"prompt": "Error: Database file not found"},
            )

        # Load valid samples from database
        all_samples = _get_all_samples()
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
    progress = progress_service.get_progress()

    return templates.TemplateResponse(
        request,
        "partials/progress.html",
        {
            "votes_completed": progress.votes_completed,
            "unique_pairs_judged": progress.unique_pairs_judged,
            "total_possible_pairs": progress.total_possible_pairs,
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
        sample_a = _get_sample_by_id(str(sample_a_id))
        if sample_a is None:
            return templates.TemplateResponse(
                request,
                "partials/matchup.html",
                {"error": f"Sample A not found: {sample_a_id}"},
            )

        # Validate that sample_b_id exists in database
        sample_b = _get_sample_by_id(str(sample_b_id))
        if sample_b is None:
            return templates.TemplateResponse(
                request,
                "partials/matchup.html",
                {"error": f"Sample B not found: {sample_b_id}"},
            )

        # Create and save the vote
        # winner is validated above to be one of "A", "B", "tie", "fail"
        vote = Vote(
            sample_a_id=str(sample_a_id),
            sample_b_id=str(sample_b_id),
            winner=cast(Literal["A", "B", "tie", "fail"], winner),
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


@app.get("/api/analytics", response_model=AnalyticsResponse)
async def get_analytics() -> AnalyticsResponse:
    """Get complete analytics data for the leaderboard and stability metrics.

    Returns leaderboard rankings, stability metrics, ELO history over time,
    and head-to-head comparison matrix.
    """
    return analytics_service.get_analytics_data()


@app.get("/api/vlm-accuracy", response_model=VLMAccuracyResponse)
async def get_vlm_accuracy() -> VLMAccuracyResponse:
    """Get VLM accuracy statistics per model and category.

    Returns accuracy statistics from vlm_evaluations.jsonl,
    broken down by model and category.
    """
    return analytics_service.get_vlm_accuracy_data()


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
    vlm_accuracy = analytics_service.get_vlm_accuracy_by_model()

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

    correlation_coefficient = analytics_service.calculate_pearson_correlation(
        elo_values, accuracy_values
    )

    return EloVLMCorrelationResponse(
        correlation_coefficient=correlation_coefficient,
        data=correlation_data,
    )


@app.get("/htmx/analytics", response_class=HTMLResponse)
async def htmx_get_analytics(request: Request) -> HTMLResponse:
    """HTMX endpoint to get analytics dashboard as HTML fragment."""
    try:
        analytics = analytics_service.get_analytics_data()

        if not analytics.leaderboard:
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
        # Load models for display names
        try:
            models = load_models()
        except (FileNotFoundError, Exception):
            models = []

        model_lookup: dict[str, Model] = {m.id: m for m in models}

        # Get VLM accuracy data from service
        vlm_response = analytics_service.get_vlm_accuracy_data()
        by_model = vlm_response.by_model

        if not by_model:
            return templates.TemplateResponse(
                request,
                "partials/vlm_accuracy.html",
                {"vlm_accuracy_data": []},
            )

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


@app.get("/htmx/elo-vlm-correlation", response_class=HTMLResponse)
async def htmx_get_elo_vlm_correlation(request: Request) -> HTMLResponse:
    """HTMX endpoint to get Elo-VLM correlation chart as HTML fragment."""
    try:
        # Get correlation data from API endpoint
        correlation_response = await get_elo_vlm_correlation()

        # Convert correlation data to JSON for Chart.js
        correlation_json = json.dumps(
            {
                "correlation_coefficient": correlation_response.correlation_coefficient,
                "data": [
                    {
                        "x": point.elo_rating,
                        "y": point.vlm_accuracy,
                        "model_id": point.model_id,
                        "model_name": point.model_name,
                    }
                    for point in correlation_response.data
                ],
            }
        )

        return templates.TemplateResponse(
            request,
            "partials/elo_vlm_correlation.html",
            {
                "correlation_json": correlation_json,
                "correlation_coefficient": correlation_response.correlation_coefficient,
                "data": correlation_response.data,
            },
        )
    except Exception:
        logging.exception("Error generating Elo-VLM correlation chart")
        return templates.TemplateResponse(
            request,
            "partials/elo_vlm_correlation.html",
            {"error": "Failed to load correlation data."},
        )
