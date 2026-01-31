import random
from collections import Counter
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from asciibench.common.models import ArtSample, Vote
from asciibench.common.persistence import (
    append_jsonl,
    read_jsonl,
    read_jsonl_by_id,
)
from asciibench.judge_ui.matchup_service import MatchupService
from asciibench.judge_ui.undo_service import UndoService

app = FastAPI(title="ASCIIBench Judge UI")
templates = Jinja2Templates(directory="templates")

# Data file paths
DATA_DIR = Path("data")
DATABASE_PATH = DATA_DIR / "database.jsonl"
VOTES_PATH = DATA_DIR / "votes.jsonl"

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


# Backward compatibility wrappers for tests
def _make_sorted_pair(a: str, b: str) -> tuple[str, str]:
    """Create a sorted pair of strings for consistent ordering."""
    return matchup_service._make_sorted_pair(a, b)


def _get_pair_comparison_counts(votes: list[Vote]) -> Counter[tuple[str, str]]:
    """Count comparisons for each ordered pair of samples."""
    return matchup_service._get_pair_comparison_counts(votes)


def _get_model_pair_comparison_counts(
    votes: list[Vote], samples: list[ArtSample]
) -> Counter[tuple[str, str]]:
    """Count comparisons between each pair of models."""
    return matchup_service._get_model_pair_comparison_counts(votes, samples)


def _select_matchup(
    valid_samples: list[ArtSample], votes: list[Vote]
) -> tuple[ArtSample, ArtSample]:
    """Select two samples for a matchup."""
    return matchup_service._select_matchup(valid_samples, votes)


def _calculate_total_possible_pairs(valid_samples: list[ArtSample]) -> int:
    """Calculate total possible unique matchups between samples."""
    return matchup_service._calculate_total_possible_pairs(valid_samples)


def _get_unique_model_pairs_judged(votes: list[Vote], samples: list[ArtSample]) -> int:
    """Count unique model pairs that have been compared."""
    return matchup_service.get_unique_model_pairs_judged(votes, samples)


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
