"""Sampler module for generating ASCII art samples.

This module provides functionality to coordinate the generation of
multiple samples from configured models and prompts.

Dependencies:
    - asciibench.common.models: Data models for ArtSample, Model, Prompt
    - asciibench.common.config: GenerationConfig for settings
    - asciibench.common.persistence: JSONL persistence utilities
    - asciibench.generator.client: OpenRouter client for LLM API calls
    - asciibench.generator.sanitizer: ASCII art extraction utilities
"""

import asyncio
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from asciibench.common.config import GenerationConfig, Settings
from asciibench.common.logging import (
    generate_id,
    get_logger,
    set_request_id,
    set_run_id,
)
from asciibench.common.models import ArtSample, Model, Prompt
from asciibench.common.persistence import append_jsonl, read_jsonl
from asciibench.generator.client import (
    AuthenticationError,
    ModelError,
    OpenRouterClient,
    OpenRouterClientError,
    RateLimitError,
    TransientError,
)
from asciibench.generator.sanitizer import extract_ascii_from_markdown
from asciibench.generator.state import BatchMetrics, SharedState

logger = get_logger("generator.sampler")

# Type alias for progress callback: (model_id, prompt_text, attempt_number, total_remaining) -> None
ProgressCallback = Callable[[str, str, int, int], None]

# Type alias for stats callback: (is_valid, cost) -> None
StatsCallback = Callable[[bool, float | None], None]

# Default number of retries for invalid outputs or API errors
DEFAULT_MAX_RETRIES = 3


@dataclass
class SampleTask:
    """Represents a single sample generation task."""

    model: Model
    prompt: Prompt
    attempt: int


def _build_existing_sample_keys(samples: list[ArtSample]) -> set[tuple[str, str, int]]:
    """Build a set of keys for existing samples for O(1) lookup.

    Args:
        samples: List of existing ArtSample objects

    Returns:
        Set of (model_id, prompt_text, attempt_number) tuples
    """
    return {(s.model_id, s.prompt_text, s.attempt_number) for s in samples}


def _sample_exists(
    model_id: str,
    prompt_text: str,
    attempt_number: int,
    existing_keys: set[tuple[str, str, int]],
) -> bool:
    """Check if a sample already exists in the database.

    Args:
        model_id: Model identifier
        prompt_text: Prompt text
        attempt_number: Attempt number (1-indexed)
        existing_keys: Set of existing sample keys

    Returns:
        True if sample already exists, False otherwise
    """
    return (model_id, prompt_text, attempt_number) in existing_keys


def _validate_output(raw_output: str, sanitized_output: str, max_tokens: int) -> bool:
    """Validate the output from an LLM call.

    Args:
        raw_output: Raw output from the model
        sanitized_output: Sanitized/extracted ASCII art
        max_tokens: Maximum tokens configured

    Returns:
        True if output is valid, False otherwise
    """
    # Invalid if no code block found (empty sanitized output)
    if not sanitized_output:
        return False
    # Invalid if output exceeds max_tokens (rough estimate: 3 chars per token for ASCII art)
    if len(raw_output) > max_tokens * 3:
        return False
    return True


async def _generate_single_sample(
    client: OpenRouterClient,
    task: SampleTask,
    config: GenerationConfig,
) -> tuple[ArtSample, float, str | None]:
    """Generate a single sample without retry logic.

    Args:
        client: OpenRouter client instance
        task: Sample task with model, prompt, and attempt info
        config: Generation configuration

    Returns:
        Tuple of (ArtSample, duration_ms, error_message)
    """
    # Generate and set request_id for this sample generation
    request_id = generate_id()
    set_request_id(request_id)

    start_time = time.perf_counter()
    error_message = None

    try:
        response = await client.generate_async(
            model_id=task.model.id,
            prompt=task.prompt.text,
            config=config,
        )
        raw_output = response.text
        sanitized_output = extract_ascii_from_markdown(raw_output)
        is_valid = _validate_output(raw_output, sanitized_output, config.max_tokens)

        sample = ArtSample(
            model_id=task.model.id,
            prompt_text=task.prompt.text,
            category=task.prompt.category,
            attempt_number=task.attempt,
            raw_output=raw_output,
            sanitized_output=sanitized_output,
            is_valid=is_valid,
            output_tokens=response.completion_tokens,
            cost=response.cost,
        )
    except RateLimitError as e:
        sample = ArtSample(
            model_id=task.model.id,
            prompt_text=task.prompt.text,
            category=task.prompt.category,
            attempt_number=task.attempt,
            raw_output="",
            sanitized_output="",
            is_valid=False,
            output_tokens=None,
            cost=None,
        )
        logger.error(
            "Rate limited after retries",
            {
                "model": task.model.id,
                "attempt": task.attempt,
                "error": str(e),
            },
        )
        error_message = f"RateLimitError: {e}"
    except AuthenticationError as e:
        sample = ArtSample(
            model_id=task.model.id,
            prompt_text=task.prompt.text,
            category=task.prompt.category,
            attempt_number=task.attempt,
            raw_output="",
            sanitized_output="",
            is_valid=False,
            output_tokens=None,
            cost=None,
        )
        logger.error(
            "Authentication failed",
            {
                "model": task.model.id,
                "attempt": task.attempt,
                "error": str(e),
            },
        )
        error_message = f"AuthenticationError: {e}"
    except TransientError as e:
        sample = ArtSample(
            model_id=task.model.id,
            prompt_text=task.prompt.text,
            category=task.prompt.category,
            attempt_number=task.attempt,
            raw_output="",
            sanitized_output="",
            is_valid=False,
            output_tokens=None,
            cost=None,
        )
        logger.error(
            "Transient error encountered",
            {
                "model": task.model.id,
                "attempt": task.attempt,
                "error": str(e),
            },
        )
        error_message = f"TransientError: {e}"
    except ModelError as e:
        sample = ArtSample(
            model_id=task.model.id,
            prompt_text=task.prompt.text,
            category=task.prompt.category,
            attempt_number=task.attempt,
            raw_output="",
            sanitized_output="",
            is_valid=False,
            output_tokens=None,
            cost=None,
        )
        logger.error(
            "Model error encountered",
            {
                "model": task.model.id,
                "attempt": task.attempt,
                "error": str(e),
            },
        )
        error_message = f"ModelError: {e}"
    except OpenRouterClientError as e:
        sample = ArtSample(
            model_id=task.model.id,
            prompt_text=task.prompt.text,
            category=task.prompt.category,
            attempt_number=task.attempt,
            raw_output="",
            sanitized_output="",
            is_valid=False,
            output_tokens=None,
            cost=None,
        )
        logger.error(
            "OpenRouter client error",
            {
                "model": task.model.id,
                "attempt": task.attempt,
                "error": str(e),
            },
        )
        error_message = f"OpenRouterClientError: {e}"
    except Exception as e:
        sample = ArtSample(
            model_id=task.model.id,
            prompt_text=task.prompt.text,
            category=task.prompt.category,
            attempt_number=task.attempt,
            raw_output="",
            sanitized_output="",
            is_valid=False,
            output_tokens=None,
            cost=None,
        )
        logger.error(
            "Unexpected exception",
            {
                "model": task.model.id,
                "attempt": task.attempt,
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )
        error_message = f"Unexpected {type(e).__name__}: {e}"

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000

    return sample, duration_ms, error_message


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
    """Process a single sample generation task with semaphore-limited concurrency.

    This function handles idempotency checks, generation, metrics recording,
    and persistence for a single task, with concurrency controlled by the
    provided semaphore.

    Args:
        client: OpenRouter client instance
        task: Sample task with model, prompt, and attempt info
        config: Generation configuration
        database_path: Path to database file for persistence
        state: SharedState for thread-safe idempotency and metrics
        progress_callback: Optional progress callback before generation
        stats_callback: Optional stats callback after generation
        total_combinations: Total number of combinations for progress tracking
        semaphore: Async semaphore to limit concurrent operations

    Returns:
        ArtSample if generated (including failed samples), None if skipped
    """
    # Use semaphore to limit concurrent executions
    async with semaphore:
        # Increment concurrent task counter
        await state.increment_concurrent()

        try:
            # Build sample key for idempotency check
            sample_key = (task.model.id, task.prompt.text, task.attempt)

            # Atomically check if key already exists, add if not
            key_added = await state.check_and_add_key(sample_key)
            if not key_added:
                # Sample already exists, skip generation
                return None

            # Call progress callback before generation if provided
            if progress_callback is not None:
                processed = await state.increment_processed()
                remaining = total_combinations - processed + 1
                progress_callback(task.model.id, task.prompt.text, task.attempt, remaining)

                # Log concurrency metrics every 10 tasks
                await state.maybe_log_concurrency()

            # Generate the sample
            sample, duration_ms, error_message = await _generate_single_sample(client, task, config)

            # Record sample in thread-safe metrics
            await state.record_sample(sample.is_valid, duration_ms, sample.cost)

            # Log sample generation
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

            # Call stats callback after generation if provided
            if stats_callback is not None:
                stats_callback(sample.is_valid, sample.cost)

            # Persist immediately for resume capability
            append_jsonl(database_path, sample)

            return sample
        finally:
            # Decrement concurrent task counter
            await state.decrement_concurrent()


async def _generate_batch_for_model(
    client: OpenRouterClient,
    model: Model,
    tasks: list[SampleTask],
    config: GenerationConfig,
    database_path: Path,
    existing_keys: set[tuple[str, str, int]],
    progress_callback: ProgressCallback | None,
    stats_callback: StatsCallback | None,
    total_combinations: int,
    samples_processed_ref: list[int],
    metrics: BatchMetrics,
) -> list[ArtSample]:
    """Generate all samples for a single model concurrently.

    Args:
        client: OpenRouter client instance
        model: Model to generate samples for
        tasks: List of sample tasks for this model
        config: Generation configuration
        database_path: Path to database file
        existing_keys: Set of existing sample keys (will be mutated)
        progress_callback: Optional progress callback
        stats_callback: Optional stats callback called after each sample
        total_combinations: Total number of combinations for progress tracking
        samples_processed_ref: Mutable reference to track samples processed
        metrics: BatchMetrics to track generation statistics

    Returns:
        List of newly generated samples
    """
    newly_generated: list[ArtSample] = []

    async def process_task(task: SampleTask) -> ArtSample | None:
        samples_processed_ref[0] += 1

        # Check idempotency - skip if sample already exists
        if _sample_exists(task.model.id, task.prompt.text, task.attempt, existing_keys):
            return None

        # Call progress callback before generation
        if progress_callback is not None:
            remaining = total_combinations - samples_processed_ref[0] + 1
            progress_callback(task.model.id, task.prompt.text, task.attempt, remaining)

        # Generate and set request_id for this sample (also set in _generate_single_sample)
        request_id = generate_id()
        set_request_id(request_id)

        sample, duration_ms, error_message = await _generate_single_sample(client, task, config)

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

        # Record batch metrics
        metrics.record_sample(sample.is_valid, duration_ms, sample.cost)

        # Call stats callback after generation
        if stats_callback is not None:
            stats_callback(sample.is_valid, sample.cost)

        # Persist immediately for resume capability
        append_jsonl(database_path, sample)

        # Update tracking for idempotency
        existing_keys.add((task.model.id, task.prompt.text, task.attempt))

        return sample

    # Run all tasks for this model concurrently
    results = await asyncio.gather(*[process_task(task) for task in tasks])

    # Filter out None results (skipped samples)
    newly_generated = [r for r in results if r is not None]

    return newly_generated


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
    """Generate ASCII art samples from configured models and prompts asynchronously.

    This function coordinates the generation of multiple samples by:
    1. Building all tasks upfront (all models x prompts x attempts)
    2. Processing all tasks concurrently with semaphore-limited parallelism
    3. Checking for existing samples via SharedState for idempotent resume
    4. Persisting each sample immediately for resume capability

    Args:
        models: List of Model objects to generate samples from
        prompts: List of Prompt objects with ASCII art generation prompts
        config: GenerationConfig with settings (attempts_per_prompt, temperature, etc.)
        database_path: Path to the database JSONL file (default: data/database.jsonl)
        client: Optional OpenRouterClient instance (created from settings if not provided)
        settings: Optional Settings instance (required if client not provided)
        progress_callback: Optional callback called before each sample generation
            with (model_id, prompt_text, attempt_number, total_remaining)
        stats_callback: Optional callback called after each sample generation
            with (is_valid, cost)

    Returns:
        List of newly generated ArtSample objects (excludes existing samples)

    Raises:
        ValueError: If neither client nor settings are provided
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

    # Initialize SharedState with existing keys and concurrency limit
    state = SharedState(
        existing_keys=existing_keys,
        max_concurrent=config.max_concurrent_requests,
        metrics=BatchMetrics(),
    )

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    # Calculate total samples to generate for progress tracking
    total_combinations = len(models) * len(prompts) * config.attempts_per_prompt

    # Build all tasks upfront (all models x prompts x attempts)
    tasks = [
        SampleTask(model=model, prompt=prompt, attempt=attempt)
        for model in models
        for prompt in prompts
        for attempt in range(1, config.attempts_per_prompt + 1)
    ]

    # Process all tasks concurrently with semaphore limit
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
            for task in tasks
        ],
        return_exceptions=True,
    )

    # Filter results to remove None (skipped) and log exceptions
    newly_generated: list[ArtSample] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(
                "Task failed with exception",
                {
                    "task_index": i,
                    "task": tasks[i],
                    "error_type": type(result).__name__,
                    "error": str(result),
                },
            )
        elif isinstance(result, ArtSample):
            newly_generated.append(result)

    # Log batch summary
    state.metrics.log_summary()

    return newly_generated


def generate_samples(
    models: list[Model],
    prompts: list[Prompt],
    config: GenerationConfig,
    database_path: str | Path = "data/database.jsonl",
    client: OpenRouterClient | None = None,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
    stats_callback: StatsCallback | None = None,
) -> list[ArtSample]:
    """Generate ASCII art samples from configured models and prompts.

    This is a synchronous wrapper around generate_samples_async for backwards
    compatibility. See generate_samples_async for full documentation.

    Args:
        models: List of Model objects to generate samples from
        prompts: List of Prompt objects with ASCII art generation prompts
        config: GenerationConfig with settings (attempts_per_prompt, temperature, etc.)
        database_path: Path to the database JSONL file (default: data/database.jsonl)
        client: Optional OpenRouterClient instance (created from settings if not provided)
        settings: Optional Settings instance (required if client not provided)
        progress_callback: Optional callback called before each sample generation
            with (model_id, prompt_text, attempt_number, total_remaining)
        stats_callback: Optional callback called after each sample generation
            with (is_valid, cost)

    Returns:
        List of newly generated ArtSample objects (excludes existing samples)
    """
    return asyncio.run(
        generate_samples_async(
            models=models,
            prompts=prompts,
            config=config,
            database_path=database_path,
            client=client,
            settings=settings,
            progress_callback=progress_callback,
            stats_callback=stats_callback,
        )
    )
