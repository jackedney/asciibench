import random
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from asciibench.common.models import ArtSample, Vote
from asciibench.common.persistence import read_jsonl

app = FastAPI(title="ASCIIBench Judge UI")
templates = Jinja2Templates(directory="templates")

# Data file paths
DATA_DIR = Path("data")
DATABASE_PATH = DATA_DIR / "database.jsonl"
VOTES_PATH = DATA_DIR / "votes.jsonl"


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


@app.post("/api/votes")
async def submit_vote(vote: Vote) -> dict:
    return {"status": "received", "vote": vote.model_dump()}
