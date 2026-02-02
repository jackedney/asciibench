# Async Queue Implementation Plan

## Problem Statement

The generator currently processes models **sequentially** (`sampler.py:478`), creating a significant performance bottleneck:

```
Current flow:
  Model 1: [9 concurrent tasks] → wait →
  Model 2: [9 concurrent tasks] → wait →
  Model 3: [9 concurrent tasks] → wait → ...
```

**Example timing** (5 models × 3 prompts × 3 attempts = 45 samples):
- Current: ~50 seconds (5 models × ~10s each)
- Optimal: ~10 seconds (all 45 tasks concurrent)
- **Potential speedup: 5x**

## Current Architecture

### Execution Flow

```
generate_samples()           # Sync wrapper
    └── asyncio.run(generate_samples_async())
        └── for model in models:                    # SEQUENTIAL (bottleneck)
            └── _generate_batch_for_model()
                └── asyncio.gather(*tasks)          # Concurrent within model
                    └── process_task() → _generate_single_sample()
```

### Key Components

| Component | Location | Thread-Safety |
|-----------|----------|---------------|
| `existing_keys` set | sampler.py:468 | Mutable, needs lock |
| `samples_processed_ref` | sampler.py:475 | Mutable list, needs lock |
| `BatchMetrics` | sampler.py:52-84 | Not protected, needs lock |
| `append_jsonl()` | persistence.py:13-31 | FileLock ✓ |
| `ContextVar` (run_id/request_id) | logging.py | Async-safe ✓ |

## Solution: Semaphore + Gather All

### Why This Approach

| Approach | Pros | Cons |
|----------|------|------|
| **Semaphore + gather** | Simple, minimal changes, native asyncio | Less queue ordering control |
| asyncio.Queue + workers | Fine-grained control | Overkill, more complex |
| External library | Feature-rich | New dependency |

**Recommendation**: Semaphore + gather is optimal because:
1. Already using `asyncio.gather()` — minimal code change
2. All tasks are independent — no ordering requirements
3. Native asyncio, no new dependencies

---

## Implementation Steps

### Step 1: Add Config Option

**File**: `asciibench/common/config.py`

```python
class GenerationConfig(BaseModel):
    attempts_per_prompt: int = 5
    temperature: float = 0.0
    max_tokens: int = 1000
    provider: str = "openrouter"
    system_prompt: str = ""
    reasoning_effort: str | None = None
    reasoning: bool = False
    include_reasoning: bool = False
    max_concurrent_requests: int = 10  # NEW: Configurable concurrency limit
```

**Rationale**: OpenRouter typically allows 10-20 concurrent requests. Default of 10 is conservative.

---

### Step 2: Add Thread-Safe State Class

**File**: `asciibench/generator/sampler.py`

Add after `SampleTask` class (~line 94):

```python
@dataclass
class SharedState:
    """Thread-safe container for shared mutable state during concurrent generation."""

    existing_keys: set[tuple[str, str, int]]
    samples_processed: int = 0
    metrics: BatchMetrics = field(default_factory=BatchMetrics)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def check_and_add_key(self, key: tuple[str, str, int]) -> bool:
        """Atomically check if key exists and add if not.

        Returns:
            True if key was added (didn't exist), False if already exists.
        """
        async with self._lock:
            if key in self.existing_keys:
                return False
            self.existing_keys.add(key)
            return True

    async def increment_processed(self) -> int:
        """Atomically increment and return the new samples_processed count."""
        async with self._lock:
            self.samples_processed += 1
            return self.samples_processed

    async def record_sample(self, success: bool, duration_ms: float, cost: float | None) -> None:
        """Thread-safe metrics recording."""
        async with self._lock:
            self.metrics.record_sample(success, duration_ms, cost)
```

---

### Step 3: Create Independent Task Processor

**File**: `asciibench/generator/sampler.py`

Add after `_generate_single_sample()` (~line 324):

```python
async def _process_single_task(
    client: OpenRouterClient,
    task: SampleTask,
    config: GenerationConfig,
    database_path: Path,
    state: SharedState,
    progress_callback: ProgressCallback | None,
    stats_callback: StatsCallback | None,
    total_combinations: int,
    semaphore: asyncio.Semaphore,
) -> ArtSample | None:
    """Process a single sample task with semaphore-limited concurrency.

    Args:
        client: OpenRouter client instance
        task: Sample task with model, prompt, and attempt info
        config: Generation configuration
        database_path: Path to database file
        state: Shared state for thread-safe access
        progress_callback: Optional progress callback
        stats_callback: Optional stats callback
        total_combinations: Total number of combinations for progress tracking
        semaphore: Semaphore to limit concurrent API calls

    Returns:
        ArtSample if generated, None if skipped (already exists)
    """
    async with semaphore:
        # Atomic idempotency check - prevents race conditions
        key = (task.model.id, task.prompt.text, task.attempt)
        if not await state.check_and_add_key(key):
            return None  # Sample already exists, skip

        current_count = await state.increment_processed()

        # Call progress callback before generation
        if progress_callback is not None:
            remaining = total_combinations - current_count + 1
            progress_callback(task.model.id, task.prompt.text, task.attempt, remaining)

        # Generate sample
        sample, duration_ms, error_message = await _generate_single_sample(
            client, task, config
        )

        # Log metrics for this sample
        logger.info(
            "Sample generated",
            {
                "model": task.model.id,
                "prompt_id": f"{task.model.id}_{task.prompt.text}_{task.attempt}",
                "duration_ms": round(duration_ms, 2),
                "success": sample.is_valid,
                "cost": sample.cost,
                "error": error_message if error_message else None,
            },
        )

        # Record batch metrics (thread-safe)
        await state.record_sample(sample.is_valid, duration_ms, sample.cost)

        # Call stats callback after generation
        if stats_callback is not None:
            stats_callback(task.model.id, task.prompt.text, sample.is_valid, sample.cost)

        # Persist immediately (FileLock handles concurrent writes)
        append_jsonl(database_path, sample)

        return sample
```

---

### Step 4: Refactor Main Async Function

**File**: `asciibench/generator/sampler.py`

Replace `generate_samples_async()` (lines 414-506):

```python
async def generate_samples_async(
    models: list[Model],
    prompts: list[Prompt],
    config: GenerationConfig,
    database_path: str | Path = "data/database.jsonl",
    client: OpenRouterClient | None = None,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
    stats_callback: StatsCallback | None = None,
) -> list[ArtSample]:
    """Generate ASCII art samples with concurrent processing across all models.

    This function coordinates generation by:
    1. Building all tasks upfront (all models × prompts × attempts)
    2. Processing all tasks concurrently with semaphore-limited parallelism
    3. Checking for existing samples to support idempotent resume capability
    4. Persisting each sample immediately for resume capability

    Args:
        models: List of Model objects to generate samples from
        prompts: List of Prompt objects with ASCII art generation prompts
        config: GenerationConfig with settings (attempts_per_prompt, temperature, etc.)
        database_path: Path to the database JSONL file (default: data/database.jsonl)
        client: Optional OpenRouterClient instance (created from settings if not provided)
        settings: Optional Settings instance (required if client not provided)
        progress_callback: Optional callback called before each sample generation
        stats_callback: Optional callback called after each sample generation

    Returns:
        List of newly generated ArtSample objects (excludes existing samples)
    """
    database_path = Path(database_path)

    # Generate and set run_id for this generation batch
    run_id = generate_id()
    set_run_id(run_id)

    # Initialize client if not provided
    if client is None:
        if settings is None:
            settings = Settings()
        client = OpenRouterClient(
            api_key=settings.openrouter_api_key,
            base_url=settings.base_url,
            timeout=settings.openrouter_timeout_seconds,
        )

    # Load existing samples for idempotency check
    existing_samples = read_jsonl(database_path, ArtSample)
    existing_keys = _build_existing_sample_keys(existing_samples)

    # Initialize shared state with thread-safe access
    state = SharedState(existing_keys=existing_keys)

    # Create semaphore for concurrency limiting
    semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    # Build ALL tasks upfront (all models × prompts × attempts)
    all_tasks = [
        SampleTask(model=model, prompt=prompt, attempt=attempt)
        for model in models
        for prompt in prompts
        for attempt in range(1, config.attempts_per_prompt + 1)
    ]

    total_combinations = len(all_tasks)

    # Process ALL tasks concurrently with semaphore limiting
    results = await asyncio.gather(
        *[
            _process_single_task(
                client=client,
                task=task,
                config=config,
                database_path=database_path,
                state=state,
                progress_callback=progress_callback,
                stats_callback=stats_callback,
                total_combinations=total_combinations,
                semaphore=semaphore,
            )
            for task in all_tasks
        ],
        return_exceptions=True,  # Don't fail fast on individual errors
    )

    # Filter results: remove None (skipped) and log exceptions
    newly_generated: list[ArtSample] = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(
                "Task failed with unhandled exception",
                {"error": str(result), "type": type(result).__name__},
            )
        elif result is not None:
            newly_generated.append(result)

    # Log batch summary
    state.metrics.log_summary()

    return newly_generated
```

---

### Step 5: Remove Obsolete Function

**File**: `asciibench/generator/sampler.py`

Delete `_generate_batch_for_model()` function (lines 326-411) — it's no longer needed.

---

## Files Summary

| File | Action |
|------|--------|
| `asciibench/common/config.py` | Add `max_concurrent_requests: int = 10` |
| `asciibench/generator/sampler.py` | Add `SharedState` class |
| `asciibench/generator/sampler.py` | Add `_process_single_task()` function |
| `asciibench/generator/sampler.py` | Rewrite `generate_samples_async()` |
| `asciibench/generator/sampler.py` | Delete `_generate_batch_for_model()` |
| `tests/test_sampler.py` | Add concurrent behavior tests |

---

## Testing Strategy

### Unit Tests to Add

1. **`test_concurrent_generation_respects_semaphore_limit`**
   - Mock client with delay
   - Verify max concurrent calls equals config limit

2. **`test_concurrent_idempotency_race_condition`**
   - Submit same sample key from multiple tasks
   - Verify only one succeeds

3. **`test_concurrent_metrics_accuracy`**
   - Generate samples concurrently
   - Verify final counts match expected

4. **`test_concurrent_persistence_no_duplicates`**
   - Run concurrent generation
   - Verify JSONL has no duplicate entries

### Manual Verification

```bash
# Run generator and observe timing
python -m asciibench.generator.main

# Check for duplicates in output
cat data/database.jsonl | jq -c '[.model_id, .prompt_text, .attempt_number]' | sort | uniq -d
```

---

## Backwards Compatibility

- Default `max_concurrent_requests=10` provides immediate benefit
- Set `max_concurrent_requests=1` to restore sequential behavior
- All existing function signatures unchanged
- All existing tests should pass without modification

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Rate limit errors increase | Medium | Medium | Default conservative limit (10), exponential backoff in client |
| Race condition in idempotency | Low | High | Atomic check-and-add with asyncio.Lock |
| Progress callback ordering | Low | Low | UI already handles out-of-order updates |
| Memory spike with many tasks | Low | Low | Tasks are lightweight dataclasses |

---

## Expected Performance

| Scenario | Sequential | Concurrent (limit=10) | Speedup |
|----------|-----------|----------------------|---------|
| 5 models × 3 prompts × 3 attempts | ~50s | ~10-15s | ~4-5x |
| 10 models × 5 prompts × 5 attempts | ~500s | ~50-75s | ~7-10x |
