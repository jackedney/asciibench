# Swiss-Tournament Lazy Generation

## Context

Currently ASCIIBench generates ALL samples upfront (19 models x 113 prompts x 3 attempts = 6,441 API calls) before any judging begins. This is expensive and wasteful -- most model pairs may not need many comparisons to establish ranking. We want to generate samples **on-demand** as the judge works, focusing generation budget on comparisons that actually inform rankings.

The new approach: Swiss-tournament style rounds where each round selects 2N model pairs (N closest Elo + N random), assigns each pair an unused prompt, generates only the missing samples, and serves them for judging. The next round's samples generate in the background while the current round is being judged.

## Design

### Round Lifecycle

```
Server starts → initialize()
  ├─ Load or create round 1 (all random pairs, no Elo yet)
  ├─ Generate missing samples for round 1 (blocking)
  └─ Start background generation for round 2

Judge works through round 1 matchups
  ├─ get_next_matchup() → random unjudged matchup from current round
  ├─ record_vote() → mark matchup judged
  └─ When all 2N judged → complete_round()
       ├─ Recompute Elo from ALL votes
       ├─ Swap to round 2 (already generated in background)
       └─ Start background generation for round 3
```

### Pair Selection (per round)

- **N closest**: Sort all C(M,2) model pairs by `abs(elo_a - elo_b)`, take N smallest
- **N random**: From remaining pairs, pick N at random
- **Round 1**: All 2N are random (no Elo data yet)
- **Dedup**: No repeated pair within a round

### Prompt Selection (per pair)

- Build set of prompts this pair has already been compared on (derived from votes + samples)
- Pick a random prompt NOT in that set
- Fallback if all 113 exhausted: pick least-used prompt for that pair

### Sample Generation

- For each matchup, check if each model already has a valid sample for that prompt in `database.jsonl`
- Only generate what's missing (1 attempt per model/prompt)
- Reuse existing `OpenRouterClient.generate_async()` + `extract_ascii_from_markdown()`
- Persist each new sample via `append_jsonl()` immediately

### Same-Prompt Matching

Both models in a matchup are always compared on the **same prompt**. This is a change from the current system (which picks random samples from each model regardless of prompt).

## New Files

### 1. `asciibench/judge_ui/tournament_service.py` — Central orchestrator

```python
class TournamentService:
    """Orchestrates Swiss-tournament rounds."""

    def __init__(self, generation_service, config_service, repo, n=10):
        self._current_round: RoundState | None
        self._next_round: RoundState | None
        self._background_task: asyncio.Task | None
        self._lock: asyncio.Lock  # serialize round transitions

    async def initialize()          # Load/create round on startup
    async def get_next_matchup()    # Random unjudged matchup from current round
    async def record_vote(matchup_id, vote)  # Mark matchup judged, trigger round completion if all done
    async def undo_last_vote()      # Unmark last matchup, delegate to undo_service
    def get_round_progress()        # {round_number, judged, total, next_round_ready}
```

Round state is kept in-memory but persisted to `data/rounds.jsonl` for crash recovery. On startup, `initialize()` reconstructs state by cross-referencing round matchups with actual votes in `votes.jsonl`.

Background generation uses `asyncio.create_task()` on the FastAPI event loop (not `BackgroundTasks` which are request-scoped).

### 2. `asciibench/judge_ui/swiss_selector.py` — Pair + prompt selection

```python
class SwissPairSelector:
    def select_pairs(model_ids, elo_ratings, n=10) -> list[tuple[str, str]]
        # N closest + N random, deduped

class PromptSelector:
    def select_prompt(model_a, model_b, all_prompts, used_prompts) -> Prompt | None
        # Random unused prompt for this pair
```

### 3. `asciibench/judge_ui/generation_service.py` — On-demand sample generation

```python
class GenerationService:
    def __init__(self, client: OpenRouterClient, config: GenerationConfig, database_path):

    def find_existing_sample(model_id, prompt_text, samples) -> ArtSample | None
    async def generate_sample(model_id, prompt) -> ArtSample  # single sample, persist immediately
    async def ensure_samples_for_round(round_state, existing_samples) -> RoundState  # fill in all sample IDs
```

Reuses:
- `OpenRouterClient.generate_async()` from `generator/client.py`
- `extract_ascii_from_markdown()` from `generator/sanitizer.py`
- `append_jsonl()` from `common/persistence.py`

### 4. New models in `asciibench/common/models.py`

```python
class Matchup(BaseModel):
    id: UUID
    model_a_id: str
    model_b_id: str
    prompt_text: str
    prompt_category: str
    sample_a_id: str | None = None  # filled after generation
    sample_b_id: str | None = None
    is_judged: bool = False
    vote_id: str | None = None

class RoundState(BaseModel):
    id: UUID
    round_number: int
    matchups: list[Matchup]
    elo_snapshot: dict[str, float]  # Elo at round start (empty for round 1)
    generation_complete: bool = False
    all_judged: bool = False
    created_at: datetime
```

## Modified Files

### 5. `asciibench/judge_ui/main.py` — Wire in tournament service

**Add lifespan handler:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    await tournament_service.initialize()
    yield

app = FastAPI(title="ASCIIBench Judge UI", lifespan=lifespan)
```

**Add service instantiation** (alongside existing services):
```python
settings = Settings()
config_service = ConfigService()
gen_config = config_service.get_app_config()
client = OpenRouterClient(api_key=settings.openrouter_api_key, ...)
generation_service = GenerationService(client, gen_config, DATABASE_PATH)
tournament_service = TournamentService(generation_service, config_service, repo)
```

**Modify `GET /htmx/matchup`** (lines 267-318):
- Replace `matchup_service.get_matchup()` with `tournament_service.get_next_matchup()`
- Load samples by ID from the matchup's `sample_a_id`/`sample_b_id`
- Pass `matchup_id` to template

**Modify `POST /htmx/vote`** (lines 374-441):
- After persisting vote, call `await tournament_service.record_vote(matchup_id, vote)`
- Read `matchup_id` from form data

**Modify `POST /htmx/undo`** (lines 444-480):
- After undo_service removes vote, call `await tournament_service.undo_last_vote()`

**Modify `GET /htmx/progress`** (lines 355-371):
- Add round progress from `tournament_service.get_round_progress()`

### 6. `templates/partials/matchup.html`

Add hidden input for matchup_id:
```html
<input type="hidden" id="matchup-id" value="{{ matchup_id }}" />
```

### 7. `templates/judge.html`

- Add `matchup_id` hidden field to the vote form
- Update `updateFormFields()` to read `matchup-id` and set it on the form
- The form submission already goes through HTMX which will include all hidden fields

### 8. `templates/partials/progress.html`

Show round progress:
```
Round {{ round_number }} · {{ round_judged }}/{{ round_total }} this round | Votes: {{ votes_completed }}
```

## What's NOT Changed

- `generator/sampler.py` and `generator/main.py` — `uv run generate` still works for bulk generation
- `analyst/` — `uv run analyze` still works, reads same `votes.jsonl` and `database.jsonl`
- `MatchupService` — kept for `ProgressService` counting methods
- `selectors.py` — kept for `ModelPairSelector.get_model_pair_comparison_counts()`
- Double-blind judging — model IDs still excluded from responses
- JSONL persistence format — same files, same formats
- Keyboard shortcuts (A/D/S/F/Z) — unchanged

## Implementation Order

1. Add `Matchup` and `RoundState` to `common/models.py`
2. Create `judge_ui/swiss_selector.py` (pair + prompt selection)
3. Create `judge_ui/generation_service.py` (on-demand generation)
4. Create `judge_ui/tournament_service.py` (round orchestration)
5. Modify `judge_ui/main.py` (wire in tournament, lifespan, modify endpoints)
6. Modify templates (`matchup.html`, `judge.html`, `progress.html`)
7. End-to-end test: start server, verify round 1 generates, judge matchups, verify round 2 starts

## Verification

1. `uv run dev` — server starts, round 1 is planned and samples generate (check logs)
2. Open judge UI — matchups appear with same-prompt comparisons
3. Judge all 20 matchups — round completes, Elo recalculates, round 2 starts
4. Round 2 should have N closest + N random pairs based on Elo
5. `uv run generate` — still works independently for bulk generation
6. `uv run analyze` — still works, picks up all votes
7. Kill and restart server mid-round — round state recovers from `rounds.jsonl`
8. Run existing tests — no regressions
