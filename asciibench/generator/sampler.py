"""Sampler module for generating ASCII art samples.

This module provides functionality to coordinate generation of
multiple samples from configured models and prompts.

Dependencies:
    - asciibench.common.models: Data models for ArtSample, Model, Prompt
    - asciibench.common.config: GenerationConfig for settings
    - asciibench.common.repository: DataRepository for loading existing samples
    - asciibench.generator.client: OpenRouter client for LLM API calls
    - asciibench.generator.concurrent: Concurrent generation primitive
    - asciibench.generator.state: BatchMetrics for tracking statistics
"""

import asyncio
from collections.abc import Callable
from pathlib import Path

from asciibench.common.config import GenerationConfig, Settings
from asciibench.common.logging import generate_id, get_logger, get_run_id, set_run_id
from asciibench.common.models import ArtSample, Model, Prompt
from asciibench.common.observability import is_logfire_enabled
from asciibench.common.repository import DataRepository
from asciibench.generator.client import OpenRouterClient
from asciibench.generator.concurrent import GenerationTask, generate_samples_concurrent
from asciibench.generator.state import BatchMetrics

logger = get_logger("generator.sampler")

ProgressCallbackType = Callable[[str, str, int, int], None]
StatsCallbackType = Callable[[bool, float | None], None]


def _build_existing_sample_keys(samples: list[ArtSample]) -> set[tuple[str, str, int]]:
    """Build a set of keys for existing samples for O(1) lookup.

    Args:
        samples: List of existing ArtSample objects

    Returns:
        Set of (model_id, prompt_text, attempt_number) tuples
    """
    return {(s.model_id, s.prompt_text, s.attempt_number) for s in samples}


async def generate_samples_async(
    models: list[Model],
    prompts: list[Prompt],
    config: GenerationConfig,
    database_path: str | Path = "data/database.jsonl",
    client: OpenRouterClient | None = None,
    settings: Settings | None = None,
    progress_callback: ProgressCallbackType | None = None,
    stats_callback: StatsCallbackType | None = None,
) -> list[ArtSample]:
    """Generate ASCII art samples from configured models and prompts asynchronously.

    This function coordinates the generation of multiple samples by:
    1. Building GenerationTask list from models x prompts x attempts
    2. Loading existing samples via DataRepository for idempotency
    3. Delegating to generate_samples_concurrent() for concurrent execution
    4. Using on_generated callback to feed progress and stats callbacks

    Args:
        models: List of Model objects to generate samples from
        prompts: List of Prompt objects with ASCII art generation prompts
        config: GenerationConfig with settings (attempts_per_prompt, temperature, etc.)
        database_path: Path to the database JSONL file (default: data/database.jsonl)
        client: Optional OpenRouterClient instance (created from settings if not provided)
        settings: Optional Settings instance (required if client not provided)
        progress_callback: Optional callback called after each sample generation
            with (model_id, prompt_text, attempt_number, total_remaining)
        stats_callback: Optional callback called after each sample generation
            with (is_valid, cost)

    Returns:
        List of newly generated ArtSample objects (excludes existing samples)

    Raises:
        ValueError: If neither client nor settings are provided
    """
    database_path = Path(database_path)
    data_dir = database_path.parent

    if get_run_id() is None:
        run_id = generate_id()
        set_run_id(run_id)

    if client is None:
        if settings is None:
            settings = Settings()
        client = OpenRouterClient(
            api_key=settings.openrouter_api_key,
            base_url=settings.base_url,
            timeout=settings.openrouter_timeout_seconds,
        )

    repo = DataRepository(data_dir=data_dir)
    try:
        existing_samples = repo.get_all_samples()
    except FileNotFoundError:
        existing_samples = []
    existing_keys = _build_existing_sample_keys(existing_samples)

    all_tasks = [
        GenerationTask(
            model_id=model.id,
            prompt_text=prompt.text,
            category=prompt.category,
            attempt=attempt,
        )
        for model in models
        for prompt in prompts
        for attempt in range(1, config.attempts_per_prompt + 1)
    ]

    seen_keys: set[tuple[str, str, int]] = set()
    generation_tasks: list[GenerationTask] = []
    for task in all_tasks:
        key = (task.model_id, task.prompt_text, task.attempt)
        if key not in seen_keys:
            seen_keys.add(key)
            generation_tasks.append(task)

    total_combinations = sum(
        1 for t in generation_tasks
        if (t.model_id, t.prompt_text, t.attempt) not in existing_keys
    )
    generated_count = 0
    metrics = BatchMetrics()

    def on_generated(sample: ArtSample) -> None:
        nonlocal generated_count
        generated_count += 1

        metrics.record_sample(sample.is_valid, sample.cost)

        if progress_callback is not None:
            remaining = total_combinations - generated_count
            progress_callback(sample.model_id, sample.prompt_text, sample.attempt_number, remaining)

        if stats_callback is not None:
            stats_callback(sample.is_valid, sample.cost)

    samples = await generate_samples_concurrent(
        tasks=generation_tasks,
        client=client,
        config=config,
        database_path=database_path,
        existing_keys=existing_keys,
        max_concurrent=config.max_concurrent_requests,
        on_generated=on_generated,
    )

    metrics.log_summary()

    return samples


def generate_samples(
    models: list[Model],
    prompts: list[Prompt],
    config: GenerationConfig,
    database_path: str | Path = "data/database.jsonl",
    client: OpenRouterClient | None = None,
    settings: Settings | None = None,
    progress_callback: ProgressCallbackType | None = None,
    stats_callback: StatsCallbackType | None = None,
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
        progress_callback: Optional callback called after each sample generation
            with (model_id, prompt_text, attempt_number, total_remaining)
        stats_callback: Optional callback called after each sample generation
            with (is_valid, cost)

    Returns:
        List of newly generated ArtSample objects (excludes existing samples)
    """
    if get_run_id() is None:
        run_id = generate_id()
        set_run_id(run_id)

    if is_logfire_enabled():
        import logfire

        total_tasks = len(models) * len(prompts) * config.attempts_per_prompt
        model_ids = [m.id for m in models]

        with logfire.span(
            "batch.generate",
            total_tasks=total_tasks,
            max_concurrent_requests=config.max_concurrent_requests,
            model_ids=model_ids,
            run_id=get_run_id(),
        ):
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
    else:
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
