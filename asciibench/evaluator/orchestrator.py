"""Evaluation orchestrator for VLM evaluation pipeline.

This module coordinates the full evaluation pipeline, including loading samples,
rendering ASCII art, calling VLMs, computing similarity, and persisting results.
It supports concurrent processing with idempotent resume capability.

Dependencies:
    - asyncio: Async coordination and concurrency control
    - rich: Progress bar display
"""

import asyncio
from collections.abc import AsyncGenerator, Callable, Coroutine
from pathlib import Path
from typing import Any

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)

from asciibench.common.config import EvaluatorConfig, RendererConfig, Settings
from asciibench.common.logging import get_logger
from asciibench.common.models import ArtSample, VLMEvaluation
from asciibench.common.persistence import append_jsonl, read_jsonl
from asciibench.evaluator.renderer import render_ascii_to_image
from asciibench.evaluator.similarity import compute_similarity
from asciibench.evaluator.subject_extractor import extract_subject
from asciibench.evaluator.vlm_client import VLMClient

logger = get_logger(__name__)


async def run_evaluation(
    database_path: str | Path = "data/database.jsonl",
    evaluations_path: str | Path = "data/vlm_evaluations.jsonl",
    config: EvaluatorConfig | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    limit: int | None = None,
) -> list[VLMEvaluation]:
    """Run the full evaluation pipeline asynchronously.

    This function coordinates the evaluation of ASCII art samples by:
    1. Loading valid samples from database.jsonl
    2. Loading existing evaluations for idempotency
    3. Building tasks for samples not yet evaluated by each VLM model
    4. Processing tasks concurrently with semaphore-limited parallelism
    5. For each task: render image, call VLM, extract subject,
       compute similarity, determine correctness
    6. Persisting each VLMEvaluation immediately to vlm_evaluations.jsonl
    7. Displaying progress with Rich progress bar

    Args:
        database_path: Path to database.jsonl file (default: data/database.jsonl)
        evaluations_path: Path to vlm_evaluations.jsonl file
            (default: data/vlm_evaluations.jsonl)
        config: EvaluatorConfig with settings (vlm_models, similarity_threshold,
            max_concurrency, font). If None, loads from evaluator_config.yaml
        progress_callback: Optional callback called after each evaluation with
            (total_processed, total_tasks, model_id)
        limit: Optional limit on the number of evaluation tasks to process.
            If provided, only the first N tasks will be processed.

    Returns:
        List of newly created VLMEvaluation objects (excludes existing evaluations)

    Examples:
        >>> evaluations = await run_evaluation()
        >>> len(evaluations)
        100

        >>> evaluations = await run_evaluation(limit=10)
        >>> len(evaluations) <= 10
        True

    Negative case:
        Network failure on one sample logs error and continues with others
    """
    database_path = Path(database_path)
    evaluations_path = Path(evaluations_path)

    if config is None:
        from asciibench.common.yaml_config import load_evaluator_config

        config = load_evaluator_config()

    settings = Settings()
    renderer_config = RendererConfig(font=config.font)

    logger.info("Loading valid samples from database", {"database_path": database_path})
    all_samples = read_jsonl(database_path, ArtSample)
    valid_samples = [s for s in all_samples if s.is_valid]

    if not valid_samples:
        logger.warning("No valid samples found in database")
        return []

    logger.info(
        "Found valid samples",
        {"total": len(all_samples), "valid": len(valid_samples)},
    )

    logger.info(
        "Loading existing evaluations for idempotency", {"evaluations_path": evaluations_path}
    )
    existing_evaluations = read_jsonl(evaluations_path, VLMEvaluation)

    logger.info(
        "Loaded existing evaluations",
        {"count": len(existing_evaluations)},
    )

    existing_eval_keys = {(str(ev.sample_id), ev.vlm_model_id) for ev in existing_evaluations}

    tasks_to_process = []
    for sample in valid_samples:
        for vlm_model_id in config.vlm_models:
            key = (str(sample.id), vlm_model_id)
            if key not in existing_eval_keys:
                tasks_to_process.append((sample, vlm_model_id))

    logger.info(
        "Tasks to process after idempotency check",
        {"tasks": len(tasks_to_process)},
    )

    if limit is not None and limit > 0:
        original_count = len(tasks_to_process)
        tasks_to_process = tasks_to_process[:limit]
        logger.info(
            "Applied limit to tasks",
            {"original": original_count, "limited_to": len(tasks_to_process)},
        )

    if not tasks_to_process:
        logger.info("All samples already evaluated")
        return []

    results: list[VLMEvaluation] = []

    async def process_task(
        sample: ArtSample,
        vlm_model_id: str,
        vlm_client: VLMClient,
    ) -> VLMEvaluation | None:
        """Process a single sample evaluation task.

        Args:
            sample: ArtSample to evaluate
            vlm_model_id: VLM model to use for evaluation
            vlm_client: VLMClient instance for API calls

        Returns:
            VLMEvaluation if successful, None if error occurs
        """
        try:
            expected_subject = extract_subject(sample.prompt_text)

            image_bytes = render_ascii_to_image(sample.sanitized_output, renderer_config)

            vlm_response, cost = await vlm_client.analyze_image(image_bytes, vlm_model_id)

            similarity_score = await compute_similarity(expected_subject, vlm_response)

            is_correct = similarity_score >= config.similarity_threshold

            evaluation = VLMEvaluation(
                sample_id=str(sample.id),
                vlm_model_id=vlm_model_id,
                expected_subject=expected_subject,
                vlm_response=vlm_response,
                similarity_score=similarity_score,
                is_correct=is_correct,
                cost=cost,
            )

            append_jsonl(evaluations_path, evaluation)

            logger.info(
                "Evaluation completed",
                {
                    "sample_id": str(sample.id),
                    "vlm_model_id": vlm_model_id,
                    "expected_subject": expected_subject,
                    "vlm_response": vlm_response[:50],
                    "similarity_score": similarity_score,
                    "is_correct": is_correct,
                    "cost": cost,
                },
            )

            return evaluation

        except Exception as e:
            logger.error(
                "Evaluation failed",
                {
                    "sample_id": str(sample.id),
                    "vlm_model_id": vlm_model_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return None

    vlm_client = VLMClient(settings)

    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def process_with_semaphore(
        sample: ArtSample,
        vlm_model_id: str,
    ) -> VLMEvaluation | None:
        async with semaphore:
            return await process_task(sample, vlm_model_id, vlm_client)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Evaluating samples with {len(config.vlm_models)} VLM model(s)...",
            total=len(tasks_to_process),
        )

        async for result in _async_generator_with_progress(
            [
                process_with_semaphore(sample, vlm_model_id)
                for sample, vlm_model_id in tasks_to_process
            ],
            progress,
            task,
        ):
            if result is not None:
                results.append(result)
                if progress_callback:
                    progress_callback(len(results), len(tasks_to_process), result.vlm_model_id)

    logger.info(
        "Evaluation pipeline completed",
        {
            "total_tasks": len(tasks_to_process),
            "successful": len(results),
            "failed": len(tasks_to_process) - len(results),
        },
    )

    return results


async def _async_generator_with_progress(
    coroutines: list[Coroutine[Any, Any, VLMEvaluation | None]],
    progress: Progress,
    task_id: TaskID,
) -> AsyncGenerator[VLMEvaluation | None, None]:
    """Execute coroutines concurrently and yield results as they complete.

    Args:
        coroutines: List of coroutines to execute
        progress: Rich Progress object
        task_id: Progress task ID to update

    Yields:
        Results from coroutines as they complete
    """
    for future in asyncio.as_completed(coroutines):
        result = await future
        progress.update(task_id, advance=1)
        yield result
