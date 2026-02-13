"""Concurrent generation primitive for unified sample generation.

This module provides a shared concurrent generation function that can be used
by both batch CLI (sampler.py) and Judge UI (generation_service.py).

Dependencies:
    - asciibench.common.models: Data models for ArtSample
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

from asciibench.common.config import GenerationConfig
from asciibench.common.logging import generate_id, get_logger, get_run_id, set_request_id
from asciibench.common.models import ArtSample
from asciibench.common.observability import is_logfire_enabled
from asciibench.common.persistence import append_jsonl
from asciibench.generator.client import (
    AuthenticationError,
    ModelError,
    OpenRouterClient,
    OpenRouterClientError,
    RateLimitError,
    TransientError,
)
from asciibench.generator.sanitizer import extract_ascii_from_markdown

logger = get_logger("generator.concurrent")

OnGeneratedCallback = Callable[[ArtSample], None]


@dataclass
class GenerationTask:
    """Represents a single sample generation task.

    Attributes:
        model_id: Identifier of the model to use for generation
        prompt_text: The prompt text to send to the model
        category: Category of the prompt (for metadata)
        attempt: Attempt number (1-indexed)
    """

    model_id: str
    prompt_text: str
    category: str
    attempt: int


def _create_error_sample(task: GenerationTask) -> ArtSample:
    """Create an ArtSample configured for error conditions.

    Args:
        task: GenerationTask with model_id, prompt_text, category, and attempt info

    Returns:
        ArtSample with error fields set (empty output, is_valid=False)
    """
    return ArtSample(
        model_id=task.model_id,
        prompt_text=task.prompt_text,
        category=task.category,
        attempt_number=task.attempt,
        raw_output="",
        sanitized_output="",
        is_valid=False,
        output_tokens=None,
        cost=None,
    )


def _handle_generation_error(
    task: GenerationTask,
    exception: Exception,
) -> ArtSample:
    """Handle generation errors with logging.

    Args:
        task: GenerationTask with model_id, prompt_text, category, and attempt info
        exception: The exception that was raised

    Returns:
        ArtSample with is_valid=False
    """
    exception_type = type(exception)
    error_info_map: dict[type[Exception], tuple[str, str]] = {
        RateLimitError: ("Rate limited after retries", "RateLimitError"),
        AuthenticationError: ("Authentication failed", "AuthenticationError"),
        TransientError: ("Transient error encountered", "TransientError"),
        ModelError: ("Model error encountered", "ModelError"),
        OpenRouterClientError: ("OpenRouter client error", "OpenRouterClientError"),
    }

    log_message, _error_prefix = error_info_map.get(
        exception_type,
        ("Unexpected exception", f"Unexpected {exception_type.__name__}"),
    )

    sample = _create_error_sample(task)
    log_context = {
        "model": task.model_id,
        "attempt": task.attempt,
        "error": str(exception),
    }
    if log_message == "Unexpected exception":
        log_context["error_type"] = exception_type.__name__
        log_context["traceback"] = traceback.format_exc()

    logger.error(log_message, log_context)

    return sample


def _validate_output(raw_output: str, sanitized_output: str, max_tokens: int) -> bool:
    """Validate the output from an LLM call.

    Args:
        raw_output: Raw output from the model
        sanitized_output: Sanitized/extracted ASCII art
        max_tokens: Maximum tokens configured

    Returns:
        True if output is valid, False otherwise
    """
    if not sanitized_output:
        return False
    if len(raw_output) > max_tokens * 3:
        return False
    return True


async def _generate_one(
    task: GenerationTask,
    client: OpenRouterClient,
    config: GenerationConfig,
    database_path: Path,
    existing_keys: set[tuple[str, str, int]],
    semaphore: asyncio.Semaphore,
    on_generated: OnGeneratedCallback | None,
) -> ArtSample | None:
    """Generate a single sample with semaphore-limited concurrency.

    This function handles:
    - Semaphore acquire for concurrency control
    - Idempotency check against existing_keys
    - API call via OpenRouterClient
    - ASCII extraction via sanitizer
    - ArtSample creation
    - Persistence via append_jsonl
    - Optional callback invocation

    Args:
        task: GenerationTask to process
        client: OpenRouterClient instance for API calls
        config: GenerationConfig for generation parameters
        database_path: Path to the JSONL database file
        existing_keys: Set of (model_id, prompt_text, attempt) tuples for idempotency
        semaphore: Async semaphore to limit concurrent operations
        on_generated: Optional callback invoked after successful generation

    Returns:
        ArtSample if generated (including error samples), None if skipped due to idempotency
    """
    async with semaphore:
        sample_key = (task.model_id, task.prompt_text, task.attempt)

        if sample_key in existing_keys:
            logger.debug(
                "Skipping existing sample",
                {"model": task.model_id, "attempt": task.attempt},
            )
            return None

        request_id = generate_id()
        set_request_id(request_id)

        start_time = time.perf_counter()

        if is_logfire_enabled():
            import logfire

            prompt_id = f"{task.model_id}_{task.prompt_text}_{task.attempt}"
            with logfire.span(
                "sample.generate",
                prompt_id=prompt_id,
                model_id=task.model_id,
                attempt_number=task.attempt,
                run_id=get_run_id(),
                request_id=request_id,
            ) as span:
                try:
                    response = await client.generate_async(
                        model_id=task.model_id,
                        prompt=task.prompt_text,
                        config=config,
                    )
                    raw_output = response.text
                    sanitized_output = extract_ascii_from_markdown(raw_output)
                    is_valid = _validate_output(raw_output, sanitized_output, config.max_tokens)

                    sample = ArtSample(
                        model_id=task.model_id,
                        prompt_text=task.prompt_text,
                        category=task.category,
                        attempt_number=task.attempt,
                        raw_output=raw_output,
                        sanitized_output=sanitized_output,
                        is_valid=is_valid,
                        output_tokens=response.completion_tokens,
                        cost=response.cost,
                    )
                except Exception as e:
                    sample = _handle_generation_error(task, e)
                    span.record_exception(e)
        else:
            try:
                response = await client.generate_async(
                    model_id=task.model_id,
                    prompt=task.prompt_text,
                    config=config,
                )
                raw_output = response.text
                sanitized_output = extract_ascii_from_markdown(raw_output)
                is_valid = _validate_output(raw_output, sanitized_output, config.max_tokens)

                sample = ArtSample(
                    model_id=task.model_id,
                    prompt_text=task.prompt_text,
                    category=task.category,
                    attempt_number=task.attempt,
                    raw_output=raw_output,
                    sanitized_output=sanitized_output,
                    is_valid=is_valid,
                    output_tokens=response.completion_tokens,
                    cost=response.cost,
                )
            except Exception as e:
                sample = _handle_generation_error(task, e)

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        logger.info(
            "Sample generated",
            {
                "model": task.model_id,
                "prompt_id": f"{task.model_id}_{task.prompt_text}_{task.attempt}",
                "duration_ms": round(duration_ms, 2),
                "success": sample.is_valid,
                "cost": sample.cost,
            },
        )

        append_jsonl(database_path, sample)

        if on_generated is not None:
            on_generated(sample)

        return sample


async def generate_samples_concurrent(
    tasks: list[GenerationTask],
    client: OpenRouterClient,
    config: GenerationConfig,
    database_path: Path,
    existing_keys: set[tuple[str, str, int]],
    max_concurrent: int = 10,
    on_generated: OnGeneratedCallback | None = None,
) -> list[ArtSample]:
    """Generate ASCII art samples concurrently.

    This function coordinates concurrent generation of multiple samples by:
    1. Using a semaphore to limit concurrent API calls
    2. Checking existing_keys for idempotency (skip if already exists)
    3. Calling the API, extracting ASCII art, creating ArtSample
    4. Persisting each sample immediately via append_jsonl
    5. Invoking on_generated callback after each sample

    Args:
        tasks: List of GenerationTask objects to process
        client: OpenRouterClient instance for API calls
        config: GenerationConfig for generation parameters
        database_path: Path to the JSONL database file for persistence
        existing_keys: Set of (model_id, prompt_text, attempt) tuples for idempotency
        max_concurrent: Maximum number of concurrent API calls (default: 10)
        on_generated: Optional callback invoked after each sample is generated

    Returns:
        List of newly generated ArtSample objects (excludes samples skipped due to idempotency).
        Includes error samples with is_valid=False for failed generations.

    Example:
        >>> tasks = [
        ...     GenerationTask(
        ...         model_id="gpt-4", prompt_text="draw a cat", category="animals", attempt=1
        ...     ),
        ...     GenerationTask(
        ...         model_id="gpt-4", prompt_text="draw a dog", category="animals", attempt=1
        ...     ),
        ... ]
        >>> samples = await generate_samples_concurrent(
        ...     tasks=tasks,
        ...     client=client,
        ...     config=config,
        ...     database_path=Path("data/database.jsonl"),
        ...     existing_keys=set(),
        ...     max_concurrent=10,
        ... )
        >>> len(samples)
        2

    Negative case (API failure returns is_valid=False sample):
        >>> # If the second task fails, it returns a sample with is_valid=False
        >>> samples[1].is_valid
        False
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    results = await asyncio.gather(
        *[
            _generate_one(
                task=task,
                client=client,
                config=config,
                database_path=database_path,
                existing_keys=existing_keys,
                semaphore=semaphore,
                on_generated=on_generated,
            )
            for task in tasks
        ],
        return_exceptions=True,
    )

    generated_samples: list[ArtSample] = []
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
            generated_samples.append(result)

    return generated_samples
