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
