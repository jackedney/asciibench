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
    write_jsonl,
)

app = FastAPI(title="ASCIIBench Judge UI")
templates = Jinja2Templates(directory="templates")

# Data file paths
DATA_DIR = Path("data")
DATABASE_PATH = DATA_DIR / "database.jsonl"
VOTES_PATH = DATA_DIR / "votes.jsonl"

# Session state tracking for undo functionality
# Tracks whether the last action was an undo (prevents calling undo twice in a row)
_last_action_was_undo: bool = False


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


def _make_sorted_pair(a: str, b: str) -> tuple[str, str]:
    """Create a sorted pair of strings for consistent ordering."""
    return (a, b) if a <= b else (b, a)


def _get_pair_comparison_counts(votes: list[Vote]) -> Counter[tuple[str, str]]:
    """Count comparisons for each ordered pair of samples.

    Returns a Counter where keys are (sample_a_id, sample_b_id) tuples
    and values are the number of times that pair has been compared.
    The pair is stored in sorted order to ensure (A, B) and (B, A) are treated the same.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for vote in votes:
        # Sort IDs to normalize the pair ordering
        pair = _make_sorted_pair(vote.sample_a_id, vote.sample_b_id)
        counts[pair] += 1
    return counts


def _get_model_pair_comparison_counts(
    votes: list[Vote], samples: list[ArtSample]
) -> Counter[tuple[str, str]]:
    """Count comparisons between each pair of models.

    Returns a Counter where keys are (model_a_id, model_b_id) tuples (sorted)
    and values are the number of comparisons between those models.
    """
    # Build a lookup from sample_id to model_id
    sample_to_model: dict[str, str] = {str(s.id): s.model_id for s in samples}

    counts: Counter[tuple[str, str]] = Counter()
    for vote in votes:
        model_a = sample_to_model.get(vote.sample_a_id)
        model_b = sample_to_model.get(vote.sample_b_id)
        if model_a and model_b and model_a != model_b:
            pair = _make_sorted_pair(model_a, model_b)
            counts[pair] += 1
    return counts


def _select_matchup(
    valid_samples: list[ArtSample], votes: list[Vote]
) -> tuple[ArtSample, ArtSample]:
    """Select two samples for a matchup, prioritizing pairs with fewer comparisons.

    Samples must be from different models. The selection prioritizes model pairs
    that have had fewer comparisons to ensure balanced coverage.
    """
    # Group samples by model
    samples_by_model: dict[str, list[ArtSample]] = {}
    for sample in valid_samples:
        if sample.model_id not in samples_by_model:
            samples_by_model[sample.model_id] = []
        samples_by_model[sample.model_id].append(sample)

    model_ids = list(samples_by_model.keys())
    if len(model_ids) < 2:
        # Need at least 2 different models to create a matchup
        # If only one model, just pick two random samples from it
        if len(valid_samples) >= 2:
            sample_a, sample_b = random.sample(valid_samples, 2)
            return sample_a, sample_b
        raise ValueError("Not enough valid samples for a matchup")

    # Get comparison counts between model pairs
    model_pair_counts = _get_model_pair_comparison_counts(votes, valid_samples)

    # Create all possible model pairs and sort by comparison count
    model_pairs = [
        (model_ids[i], model_ids[j])
        for i in range(len(model_ids))
        for j in range(i + 1, len(model_ids))
    ]

    # Find the minimum comparison count among all pairs
    min_count = float("inf")
    for pair in model_pairs:
        sorted_pair = tuple(sorted(pair))
        count = model_pair_counts.get(sorted_pair, 0)
        if count < min_count:
            min_count = count

    # Filter to only pairs with minimum comparisons
    least_compared_pairs = []
    for pair in model_pairs:
        sorted_pair = tuple(sorted(pair))
        if model_pair_counts.get(sorted_pair, 0) == min_count:
            least_compared_pairs.append(pair)

    # Pick a random pair from the least compared
    model_a_id, model_b_id = random.choice(least_compared_pairs)

    # Pick random samples from each model
    sample_a = random.choice(samples_by_model[model_a_id])
    sample_b = random.choice(samples_by_model[model_b_id])

    return sample_a, sample_b


def _calculate_total_possible_pairs(valid_samples: list[ArtSample]) -> int:
    """Calculate total possible unique matchups between samples from different models.

    A valid matchup requires two samples from different models. This counts
    all unique sample pairs where the samples have different model_ids.
    """
    total = 0
    n = len(valid_samples)
    for i in range(n):
        for j in range(i + 1, n):
            if valid_samples[i].model_id != valid_samples[j].model_id:
                total += 1
    return total


def _calculate_total_possible_pairs_for_samples(
    samples: list[ArtSample],
) -> int:
    """Calculate total possible unique matchups for a list of samples.

    Helper that filters to valid samples and calculates pairs.
    """
    valid_samples = [s for s in samples if s.is_valid]
    return _calculate_total_possible_pairs(valid_samples)


def _get_unique_model_pairs_judged(votes: list[Vote], samples: list[ArtSample]) -> int:
    """Count unique model pairs that have been compared at least once.

    Returns the number of distinct (model_a, model_b) pairs found in votes
    where the two models are different.
    """
    model_pair_counts = _get_model_pair_comparison_counts(votes, samples)
    return len(model_pair_counts)


def _calculate_progress_by_category(
    votes: list[Vote], samples: list[ArtSample]
) -> dict[str, CategoryProgress]:
    """Calculate progress statistics broken down by category.

    Returns a dict mapping category name to CategoryProgress with
    votes_completed, unique_pairs_judged, and total_possible_pairs.
    """
    # Build lookup from sample_id to sample
    sample_lookup: dict[str, ArtSample] = {str(s.id): s for s in samples}

    # Group samples by category
    samples_by_category: dict[str, list[ArtSample]] = {}
    for sample in samples:
        if sample.is_valid:
            if sample.category not in samples_by_category:
                samples_by_category[sample.category] = []
            samples_by_category[sample.category].append(sample)

    # Group votes by category (based on sample_a's category)
    votes_by_category: dict[str, list[Vote]] = {}
    for vote in votes:
        sample_a = sample_lookup.get(vote.sample_a_id)
        if sample_a and sample_a.is_valid:
            category = sample_a.category
            if category not in votes_by_category:
                votes_by_category[category] = []
            votes_by_category[category].append(vote)

    # Calculate progress for each category
    result: dict[str, CategoryProgress] = {}
    for category, category_samples in samples_by_category.items():
        category_votes = votes_by_category.get(category, [])

        # Count unique model pairs judged in this category
        unique_pairs = _get_unique_model_pairs_judged(category_votes, category_samples)

        # Calculate total possible pairs in this category
        total_pairs = _calculate_total_possible_pairs(category_samples)

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

    # Load existing votes for comparison count prioritization
    votes = read_jsonl(VOTES_PATH, Vote)

    # Select a matchup
    sample_a, sample_b = _select_matchup(valid_samples, votes)

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
    global _last_action_was_undo
    _last_action_was_undo = False

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
    global _last_action_was_undo

    # Check if last action was already an undo (prevent double-undo)
    if _last_action_was_undo:
        raise HTTPException(
            status_code=400,
            detail="Cannot undo twice in a row. Please submit a new vote before undoing again.",
        )

    # Read all current votes
    votes = read_jsonl(VOTES_PATH, Vote)

    # Check if there are any votes to undo
    if not votes:
        raise HTTPException(
            status_code=404,
            detail="No votes to undo. The vote history is empty.",
        )

    # Get the last vote
    last_vote = votes[-1]
    last_vote_id = str(last_vote.id)

    # Remove the last vote and rewrite the file atomically
    votes_without_last = votes[:-1]
    write_jsonl(VOTES_PATH, votes_without_last)

    # Track that this action was an undo
    _last_action_was_undo = True

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
    unique_pairs_judged = _get_unique_model_pairs_judged(votes, valid_samples)
    total_possible_pairs = _calculate_total_possible_pairs(valid_samples)

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

        # Load existing votes for comparison count prioritization
        votes = read_jsonl(VOTES_PATH, Vote)

        # Select a matchup
        sample_a, sample_b = _select_matchup(valid_samples, votes)

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

        # Load existing votes for comparison count prioritization
        votes = read_jsonl(VOTES_PATH, Vote)

        # Select a matchup to get the prompt
        sample_a, _ = _select_matchup(valid_samples, votes)

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
    unique_pairs_judged = _get_unique_model_pairs_judged(votes, valid_samples)
    total_possible_pairs = _calculate_total_possible_pairs(valid_samples)

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
    global _last_action_was_undo

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
        _last_action_was_undo = False

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
    global _last_action_was_undo

    try:
        # Check if last action was already an undo (prevent double-undo)
        if _last_action_was_undo:
            return templates.TemplateResponse(
                request,
                "partials/matchup.html",
                {
                    "error": "Cannot undo twice in a row. Submit a new vote first.",
                },
            )

        # Read all current votes
        votes = read_jsonl(VOTES_PATH, Vote)

        # Check if there are any votes to undo
        if not votes:
            return templates.TemplateResponse(
                request,
                "partials/matchup.html",
                {"error": "No votes to undo."},
            )

        # Remove the last vote and rewrite the file atomically
        votes_without_last = votes[:-1]
        write_jsonl(VOTES_PATH, votes_without_last)

        # Track that this action was an undo
        _last_action_was_undo = True

        # Return the current matchup (don't load a new one)
        return await htmx_get_matchup(request)

    except Exception as e:
        return templates.TemplateResponse(
            request,
            "partials/matchup.html",
            {"error": f"Error undoing vote: {e}"},
        )
