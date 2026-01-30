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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from asciibench.common.config import GenerationConfig, Settings
from asciibench.common.models import ArtSample, Model, Prompt
from asciibench.common.persistence import append_jsonl, read_jsonl
from asciibench.generator.client import OpenRouterClient, OpenRouterClientError
from asciibench.generator.sanitizer import extract_ascii_from_markdown

# Type alias for progress callback: (model_id, prompt_text, attempt_number, total_remaining) -> None
ProgressCallback = Callable[[str, str, int, int], None]

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
    # Invalid if output exceeds max_tokens (rough estimate: 4 chars per token)
    if len(raw_output) > max_tokens * 4:
        return False
    return True


async def _generate_single_sample(
    client: OpenRouterClient,
    task: SampleTask,
    config: GenerationConfig,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ArtSample:
    """Generate a single sample with retry logic.

    Retries on:
    - API errors (OpenRouterClientError)
    - Invalid output (no code block found or output too long)

    Args:
        client: OpenRouter client instance
        task: Sample task with model, prompt, and attempt info
        config: Generation configuration
        max_retries: Maximum number of retry attempts

    Returns:
        Generated ArtSample (may be invalid if all retries exhausted)
    """
    last_raw_output = ""
    last_sanitized_output = ""

    for _retry in range(max_retries):
        try:
            raw_output = await client.generate_async(
                model_id=task.model.id,
                prompt=task.prompt.text,
                config=config,
            )

            sanitized_output = extract_ascii_from_markdown(raw_output)
            is_valid = _validate_output(raw_output, sanitized_output, config.max_tokens)

            if is_valid:
                return ArtSample(
                    model_id=task.model.id,
                    prompt_text=task.prompt.text,
                    category=task.prompt.category,
                    attempt_number=task.attempt,
                    raw_output=raw_output,
                    sanitized_output=sanitized_output,
                    is_valid=True,
                )

            # Invalid output - save for potential final sample and retry
            last_raw_output = raw_output
            last_sanitized_output = sanitized_output

        except OpenRouterClientError:
            pass  # Continue to retry

    # All retries exhausted - return invalid sample
    return ArtSample(
        model_id=task.model.id,
        prompt_text=task.prompt.text,
        category=task.prompt.category,
        attempt_number=task.attempt,
        raw_output=last_raw_output,
        sanitized_output=last_sanitized_output,
        is_valid=False,
    )


async def _generate_batch_for_model(
    client: OpenRouterClient,
    model: Model,
    tasks: list[SampleTask],
    config: GenerationConfig,
    database_path: Path,
    existing_keys: set[tuple[str, str, int]],
    progress_callback: ProgressCallback | None,
    total_combinations: int,
    samples_processed_ref: list[int],
    max_retries: int = DEFAULT_MAX_RETRIES,
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
        total_combinations: Total number of combinations for progress tracking
        samples_processed_ref: Mutable reference to track samples processed
        max_retries: Maximum number of retry attempts

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

        # Generate sample with retries
        sample = await _generate_single_sample(client, task, config, max_retries)

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
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[ArtSample]:
    """Generate ASCII art samples from configured models and prompts asynchronously.

    This function coordinates the generation of multiple samples by:
    1. Processing one model at a time
    2. Running all prompts for that model concurrently
    3. Retrying on API errors or invalid outputs (up to max_retries times)
    4. Checking for existing samples to support idempotent resume capability
    5. Persisting each sample immediately for resume capability

    Args:
        models: List of Model objects to generate samples from
        prompts: List of Prompt objects with ASCII art generation prompts
        config: GenerationConfig with settings (attempts_per_prompt, temperature, etc.)
        database_path: Path to the database JSONL file (default: data/database.jsonl)
        client: Optional OpenRouterClient instance (created from settings if not provided)
        settings: Optional Settings instance (required if client not provided)
        progress_callback: Optional callback called before each sample generation
            with (model_id, prompt_text, attempt_number, total_remaining)
        max_retries: Maximum number of retries for invalid outputs or API errors

    Returns:
        List of newly generated ArtSample objects (excludes existing samples)

    Raises:
        ValueError: If neither client nor settings are provided
    """
    database_path = Path(database_path)

    # Initialize client if not provided
    if client is None:
        if settings is None:
            settings = Settings()
        client = OpenRouterClient(
            api_key=settings.openrouter_api_key,
            base_url=settings.base_url,
        )

    # Load existing samples for idempotency check
    existing_samples = read_jsonl(database_path, ArtSample)
    existing_keys = _build_existing_sample_keys(existing_samples)

    newly_generated: list[ArtSample] = []

    # Calculate total samples to generate for progress tracking
    total_combinations = len(models) * len(prompts) * config.attempts_per_prompt
    samples_processed_ref = [0]  # Use list as mutable reference

    # Process one model at a time, running all prompts concurrently for each model
    for model in models:
        # Build tasks for this model
        tasks = [
            SampleTask(model=model, prompt=prompt, attempt=attempt)
            for prompt in prompts
            for attempt in range(1, config.attempts_per_prompt + 1)
        ]

        # Generate all samples for this model concurrently
        model_samples = await _generate_batch_for_model(
            client=client,
            model=model,
            tasks=tasks,
            config=config,
            database_path=database_path,
            existing_keys=existing_keys,
            progress_callback=progress_callback,
            total_combinations=total_combinations,
            samples_processed_ref=samples_processed_ref,
            max_retries=max_retries,
        )

        newly_generated.extend(model_samples)

    return newly_generated


def generate_samples(
    models: list[Model],
    prompts: list[Prompt],
    config: GenerationConfig,
    database_path: str | Path = "data/database.jsonl",
    client: OpenRouterClient | None = None,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
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
        max_retries: Maximum number of retries for invalid outputs or API errors

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
            max_retries=max_retries,
        )
    )
