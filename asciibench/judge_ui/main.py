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
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from asciibench.common.config import Settings
from asciibench.common.config_service import ConfigService
from asciibench.common.models import ArtSample, Model, VLMEvaluation, Vote
from asciibench.common.persistence import (
    append_jsonl,
    read_jsonl,
)
from asciibench.common.repository import DataRepository
from asciibench.common.yaml_config import load_models
from asciibench.generator.client import OpenRouterClient
from asciibench.judge_ui.analytics_service import AnalyticsService
from asciibench.judge_ui.api_models import (
    AnalyticsResponse,
    EloVLMCorrelationResponse,
    MatchupResponse,
    ProgressResponse,
    SampleResponse,
    VLMAccuracyResponse,
    VoteRequest,
    VoteResponse,
)
from asciibench.judge_ui.generation_service import GenerationService
from asciibench.judge_ui.htmx_error_handling import (
    htmx_error_handler,
    htmx_error_handler_with_context,
)
from asciibench.judge_ui.matchup_service import MatchupService
from asciibench.judge_ui.progress_service import ProgressService
from asciibench.judge_ui.tournament_service import TournamentService
from asciibench.judge_ui.undo_service import UndoService

templates = Jinja2Templates(directory="templates")

# Data file paths (constants)
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
    except (OSError, json.JSONDecodeError, ValueError):
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
    except ValueError:
        # Invalid UUID format - return None so caller can return 404
        return None
    mtime = DATABASE_PATH.stat().st_mtime_ns
    _, indexed = _get_database_indexed(DATABASE_PATH, mtime)
    return indexed.get(target_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application.

    Initializes all services on startup.
    """
    # Initialize settings and config
    app.state.settings = Settings()
    app.state.config_service = ConfigService()
    app.state.tournament_config = app.state.config_service.get_tournament_config()
    app.state.generation_config = app.state.config_service.get_app_config()

    # Initialize repository
    app.state.repo = DataRepository(data_dir=DATA_DIR, cache_ttl=0)

    # Initialize core services
    app.state.analytics_service = AnalyticsService(repo=app.state.repo)
    app.state.matchup_service = MatchupService(database_path=DATABASE_PATH, votes_path=VOTES_PATH)
    app.state.progress_service = ProgressService(
        repo=app.state.repo, matchup_service=app.state.matchup_service
    )
    app.state.undo_service = UndoService(votes_path=VOTES_PATH)

    # Initialize tournament-related services
    app.state.openrouter_client = OpenRouterClient(
        api_key=app.state.settings.openrouter_api_key,
        base_url=app.state.settings.base_url,
        timeout=app.state.settings.openrouter_timeout_seconds,
    )
    app.state.generation_service = GenerationService(
        client=app.state.openrouter_client,
        config=app.state.generation_config,
        database_path=DATABASE_PATH,
    )

    # Initialize tournament service
    app.state.tournament_service = TournamentService(
        generation_service=app.state.generation_service,
        config_service=app.state.config_service,
        repo=app.state.repo,
        n=app.state.tournament_config.round_size,
    )
    await app.state.tournament_service.initialize()

    # Initialize VLM evaluation service state
    # Type: VLMEvaluationService | None
    app.state.vlm_evaluation_service = None
    app.state.vlm_init_attempted = False

    yield

    # Cleanup on shutdown
    await app.state.tournament_service.shutdown()


app = FastAPI(title="ASCIIBench Judge UI", lifespan=lifespan)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/judge", response_class=HTMLResponse)
async def judge(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("judge.html", {"request": request})


@app.get("/api/matchup", response_model=MatchupResponse)
async def get_matchup(request: Request) -> MatchupResponse:
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
    sample_a, sample_b = request.app.state.matchup_service.get_matchup(valid_samples)

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
async def submit_vote(vote_request: VoteRequest, request: Request) -> VoteResponse:
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
    request.app.state.undo_service.record_vote_submitted()

    # Return the saved vote with generated ID and timestamp
    return VoteResponse(
        id=str(vote.id),
        sample_a_id=vote.sample_a_id,
        sample_b_id=vote.sample_b_id,
        winner=vote.winner,
        timestamp=vote.timestamp.isoformat(),
    )


@app.post("/api/undo", response_model=VoteResponse)
async def undo_vote(request: Request) -> VoteResponse:
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
    last_vote = request.app.state.undo_service.undo_vote()

    if last_vote is None:
        if request.app.state.undo_service.last_action_was_undo:
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
async def get_progress(request: Request) -> ProgressResponse:
    """Get progress statistics for judging.

    Returns the number of votes completed, unique model pairs judged,
    total possible pairs, and a breakdown by category.

    Returns:
        ProgressResponse with votes_completed, unique_pairs_judged,
        total_possible_pairs, and by_category breakdown.
    """
    return request.app.state.progress_service.get_progress()


# =============================================================================
# HTMX Endpoints - Return HTML fragments for dynamic UI updates
# =============================================================================


@app.get("/htmx/matchup", response_class=HTMLResponse)
@htmx_error_handler("partials/matchup.html")
async def htmx_get_matchup(request: Request) -> HTMLResponse:
    """HTMX endpoint to get a matchup as HTML fragment.

    Returns rendered HTML for the matchup comparison area.
    On error, returns an error message HTML fragment.
    """
    # Get next matchup from tournament service
    if request.app.state.tournament_service is None:
        return templates.TemplateResponse(
            request,
            "partials/matchup.html",
            {"error": "Tournament service not initialized"},
        )

    matchup = request.app.state.tournament_service.get_next_matchup()

    if matchup is None:
        return templates.TemplateResponse(
            request,
            "partials/matchup.html",
            {
                "error": "No matchups available. "
                "All rounds complete or next round is being generated.",
            },
        )

    # Check if sample IDs are populated
    if matchup.sample_a_id is None or matchup.sample_b_id is None:
        return templates.TemplateResponse(
            request,
            "partials/matchup.html",
            {"error": "Sample data not ready for this matchup"},
        )

    # Load samples using cached lookup
    sample_a = _get_sample_by_id(matchup.sample_a_id)
    sample_b = _get_sample_by_id(matchup.sample_b_id)

    if sample_a is None or sample_b is None:
        return templates.TemplateResponse(
            request,
            "partials/matchup.html",
            {"error": "Sample data not found for this matchup"},
        )

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
            "matchup_id": str(matchup.id),
        },
    )


@app.get("/htmx/prompt", response_class=HTMLResponse)
@htmx_error_handler(
    "partials/prompt.html", error_context_key="prompt", error_message_prefix="Error: "
)
async def htmx_get_prompt(request: Request) -> HTMLResponse:
    """HTMX endpoint to get the current prompt as HTML fragment.

    Returns rendered HTML for the prompt display area.
    """
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
    sample_a, _ = request.app.state.matchup_service.get_matchup(valid_samples)

    return templates.TemplateResponse(
        request,
        "partials/prompt.html",
        {"prompt": sample_a.prompt_text},
    )


@app.get("/htmx/progress", response_class=HTMLResponse)
async def htmx_get_progress(request: Request) -> HTMLResponse:
    """HTMX endpoint to get progress as HTML fragment.

    Returns rendered HTML for the progress display area.
    """
    progress = request.app.state.progress_service.get_progress()

    # Get round progress from tournament service
    round_progress = {}
    if request.app.state.tournament_service is not None:
        round_progress = request.app.state.tournament_service.get_round_progress()

    return templates.TemplateResponse(
        request,
        "partials/progress.html",
        {
            "votes_completed": progress.votes_completed,
            "unique_pairs_judged": progress.unique_pairs_judged,
            "total_possible_pairs": progress.total_possible_pairs,
            "round_number": round_progress.get("round_number", 0),
            "round_judged": round_progress.get("judged_count", 0),
            "round_total": round_progress.get("total_count", 0),
        },
    )


@app.post("/htmx/vote", response_class=HTMLResponse)
@htmx_error_handler("partials/matchup.html", error_message_prefix="Error submitting vote: ")
async def htmx_submit_vote(request: Request) -> HTMLResponse:
    """HTMX endpoint to submit a vote and get new matchup as HTML.

    Accepts form data with sample_a_id, sample_b_id, winner, and matchup_id.
    Returns a new matchup HTML fragment after saving the vote.
    """
    # Parse form data
    form_data = await request.form()
    sample_a_id = form_data.get("sample_a_id")
    sample_b_id = form_data.get("sample_b_id")
    winner = form_data.get("winner")
    matchup_id = form_data.get("matchup_id")

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

    # Validate matchup_id as UUID if provided
    validated_matchup_id: UUID | None = None
    if matchup_id is not None:
        try:
            validated_matchup_id = UUID(str(matchup_id))
        except ValueError:
            return templates.TemplateResponse(
                request,
                "partials/matchup.html",
                {"error": f"Invalid matchup ID: {matchup_id}"},
            )

    # Create the vote
    # winner is validated above to be one of "A", "B", "tie", "fail"
    vote = Vote(
        sample_a_id=str(sample_a_id),
        sample_b_id=str(sample_b_id),
        winner=cast(Literal["A", "B", "tie", "fail"], winner),
    )

    # Persist to votes.jsonl first (source of truth)
    append_jsonl(VOTES_PATH, vote)

    # Then update tournament state (reconstructed from votes on restart if this fails)
    if request.app.state.tournament_service is not None and validated_matchup_id is not None:
        await request.app.state.tournament_service.record_vote(validated_matchup_id, str(vote.id))

    # Clear the undo state since a new vote was submitted
    request.app.state.undo_service.record_vote_submitted()

    # Return reveal panel with model identities (instead of next matchup)
    return templates.TemplateResponse(
        request,
        "partials/vote_reveal.html",
        {
            "sample_a": {
                "id": str(sample_a.id),
                "sanitized_output": sample_a.sanitized_output,
                "model_id": sample_a.model_id,
            },
            "sample_b": {
                "id": str(sample_b.id),
                "sanitized_output": sample_b.sanitized_output,
                "model_id": sample_b.model_id,
            },
            "prompt": sample_a.prompt_text,
            "winner": winner,
        },
    )


@app.post("/htmx/undo", response_class=HTMLResponse)
@htmx_error_handler("partials/matchup.html", error_message_prefix="Error undoing vote: ")
async def htmx_undo_vote(request: Request) -> HTMLResponse:
    """HTMX endpoint to undo the last vote and return status message.

    Returns the current matchup (unchanged) with an optional message,
    or an error message if undo fails.
    """
    # Check if last action was already an undo (prevent double-undo)
    if request.app.state.undo_service.last_action_was_undo:
        return templates.TemplateResponse(
            request,
            "partials/matchup.html",
            {
                "error": "Cannot undo twice in a row. Submit a new vote first.",
            },
        )

    last_vote = request.app.state.undo_service.undo_vote()

    # Check if there was a vote to undo
    if last_vote is None:
        return templates.TemplateResponse(
            request,
            "partials/matchup.html",
            {"error": "No votes to undo."},
        )

    # Find the matchup for this vote and tell tournament service to undo
    if request.app.state.tournament_service is not None:
        matchup = request.app.state.tournament_service.find_matchup_by_samples(
            last_vote.sample_a_id, last_vote.sample_b_id
        )
        if matchup is not None:
            await request.app.state.tournament_service.undo_last_vote(matchup.id)

    # Return the current matchup (don't load a new one)
    return await htmx_get_matchup(request)


def _deduplicate_evaluations(
    existing_evaluations: list[VLMEvaluation],
    new_evaluations: list[VLMEvaluation],
    sample_id: str,
) -> list[VLMEvaluation]:
    """Deduplicate evaluations by vlm_model_id for a specific sample.

    Args:
        existing_evaluations: All existing evaluations from the repository
        new_evaluations: New evaluations from the current request
        sample_id: The sample ID to filter existing evaluations by

    Returns:
        Deduplicated list of evaluations with unique vlm_model_id values.
        Existing evaluations are preferred over new ones.
    """
    seen: set[str] = set()
    results: list[VLMEvaluation] = []

    for e in [
        *[e for e in existing_evaluations if e.sample_id == sample_id],
        *new_evaluations,
    ]:
        if e.vlm_model_id not in seen:
            seen.add(e.vlm_model_id)
            results.append(e)

    return results


@app.get("/htmx/vlm-eval", response_class=HTMLResponse)
@htmx_error_handler("partials/vlm_results.html")
async def htmx_vlm_eval(request: Request) -> HTMLResponse:
    """HTMX endpoint to evaluate samples with VLM after a vote.

    Lazily initializes VLMEvaluationService on first call.
    Evaluates both samples with all configured VLM models (idempotent).
    Returns graceful fallback if VLM evaluation is not configured.
    """

    sample_a_id = request.query_params.get("sample_a_id")
    sample_b_id = request.query_params.get("sample_b_id")

    if not sample_a_id or not sample_b_id:
        return templates.TemplateResponse(
            request,
            "partials/vlm_results.html",
            {"error": "Missing sample IDs"},
        )

    # Lazy-init VLM evaluation service
    if not request.app.state.vlm_init_attempted:
        request.app.state.vlm_init_attempted = True
        try:
            from asciibench.evaluator.vlm_evaluation_service import (
                VLMEvaluationService,
            )

            request.app.state.vlm_evaluation_service = VLMEvaluationService(VLM_EVALUATIONS_PATH)
        except (ImportError, FileNotFoundError) as e:
            logging.info(f"VLM evaluation service not available: {e.__class__.__name__}: {e}")
            request.app.state.vlm_evaluation_service = None

    if request.app.state.vlm_evaluation_service is None:
        return templates.TemplateResponse(
            request,
            "partials/vlm_results.html",
            {"error": "VLM evaluation not configured"},
        )

    # Load samples
    sample_a = _get_sample_by_id(sample_a_id)
    sample_b = _get_sample_by_id(sample_b_id)

    if not sample_a or not sample_b:
        return templates.TemplateResponse(
            request,
            "partials/vlm_results.html",
            {"error": "Sample not found"},
        )

    # Load existing evaluations for idempotency
    existing_evaluations = request.app.state.repo.get_evaluations_or_empty()
    existing_keys = {(e.sample_id, e.vlm_model_id) for e in existing_evaluations}

    # Evaluate both samples (skips already-evaluated pairs)
    new_results_a = await request.app.state.vlm_evaluation_service.evaluate_sample_all_models(
        sample_a, existing_keys
    )
    new_results_b = await request.app.state.vlm_evaluation_service.evaluate_sample_all_models(
        sample_b, existing_keys
    )

    # Collect all results (pre-existing + new) for these samples, deduplicated
    results_a = _deduplicate_evaluations(existing_evaluations, new_results_a, sample_a_id)
    results_b = _deduplicate_evaluations(existing_evaluations, new_results_b, sample_b_id)

    return templates.TemplateResponse(
        request,
        "partials/vlm_results.html",
        {"results_a": results_a, "results_b": results_b},
    )


# =============================================================================
# Analytics Endpoints
# =============================================================================


@app.get("/api/analytics", response_model=AnalyticsResponse)
async def get_analytics(request: Request) -> AnalyticsResponse:
    """Get complete analytics data for the leaderboard and stability metrics.

    Returns leaderboard rankings, stability metrics, ELO history over time,
    and head-to-head comparison matrix.
    """
    return request.app.state.analytics_service.get_analytics_data()


@app.get("/api/vlm-accuracy", response_model=VLMAccuracyResponse)
async def get_vlm_accuracy(request: Request) -> VLMAccuracyResponse:
    """Get VLM accuracy statistics per model and category.

    Returns accuracy statistics from vlm_evaluations.jsonl,
    broken down by model and category.
    """
    return request.app.state.analytics_service.get_vlm_accuracy_data()


@app.get("/api/elo-vlm-correlation", response_model=EloVLMCorrelationResponse)
async def get_elo_vlm_correlation(request: Request) -> EloVLMCorrelationResponse:
    """Get Elo ratings and VLM accuracy per model for correlation analysis.

    Returns correlation data between human Elo ratings and VLM recognition rates.
    Joins data from rankings.json and vlm_evaluations.jsonl.

    Returns:
        EloVLMCorrelationResponse with correlation coefficient and data array

    Raises:
        HTTPException: 500 if data files are malformed
    """
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

    return request.app.state.analytics_service.compute_elo_vlm_correlation(rankings_data, models)


@app.get("/htmx/analytics", response_class=HTMLResponse)
@htmx_error_handler_with_context(
    "partials/analytics.html",
    {
        "leaderboard": [],
        "stability": {"score": 0, "is_stable": False, "warnings": [], "models": {}},
        "elo_history": {},
        "elo_history_json": "{}",
        "head_to_head": {},
        "models": [],
        "total_votes": 0,
    },
    custom_error_message="An error occurred while generating analytics. Please try again.",
)
async def htmx_get_analytics(request: Request) -> HTMLResponse:
    """HTMX endpoint to get analytics dashboard as HTML fragment."""
    analytics = request.app.state.analytics_service.get_analytics_data()

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


@app.get("/htmx/vlm-accuracy", response_class=HTMLResponse)
@htmx_error_handler_with_context(
    "partials/vlm_accuracy.html",
    {"vlm_accuracy_data": []},
    custom_error_message="Failed to load VLM accuracy data.",
)
async def htmx_get_vlm_accuracy(request: Request) -> HTMLResponse:
    """HTMX endpoint to get VLM accuracy table as HTML fragment."""
    # Load models for display names
    try:
        models = load_models()
    except (FileNotFoundError, yaml.YAMLError):
        models = []

    model_lookup: dict[str, Model] = {m.id: m for m in models}

    # Get VLM accuracy data from service
    vlm_response = request.app.state.analytics_service.get_vlm_accuracy_data()
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


@app.get("/htmx/elo-vlm-correlation", response_class=HTMLResponse)
@htmx_error_handler(
    "partials/elo_vlm_correlation.html", custom_error_message="Failed to load correlation data."
)
async def htmx_get_elo_vlm_correlation(request: Request) -> HTMLResponse:
    """HTMX endpoint to get Elo-VLM correlation chart as HTML fragment."""
    # Get correlation data from API endpoint
    correlation_response = await get_elo_vlm_correlation(request)

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
