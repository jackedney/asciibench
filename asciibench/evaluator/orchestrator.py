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


class ImageRenderer:
    """Service for rendering ASCII art to PNG images.

    This class encapsulates the ASCII-to-image conversion logic,
    making it testable and reusable.

    Args:
        config: RendererConfig with font and color settings.

    Example:
        >>> renderer = ImageRenderer(RendererConfig())
        >>> image_bytes = renderer.render(" /\\_/\\ \\n( o.o )")
    """

    def __init__(self, config: RendererConfig) -> None:
        """Initialize ImageRenderer.

        Args:
            config: RendererConfig with font and color settings.
        """
        self._config = config

    def render(self, ascii_text: str | None) -> bytes:
        """Render ASCII art text to a PNG image.

        Args:
            ascii_text: ASCII art text to render (can be multi-line)

        Returns:
            PNG image bytes suitable for base64 encoding

        Example:
            >>> renderer = ImageRenderer(RendererConfig())
            >>> png_bytes = renderer.render(' /\\_/\\ \\n( o.o )')
            >>> isinstance(png_bytes, bytes)
            True
        """
        return render_ascii_to_image(ascii_text, self._config)


class VLMAnalyzer:
    """Service for analyzing images with Vision Language Models.

    This class encapsulates VLM API calls and similarity computation,
    making it testable and reusable.

    Args:
        vlm_client: VLMClient instance for API calls
        similarity_threshold: Threshold for considering a VLM response correct

    Example:
        >>> client = VLMClient(Settings())
        >>> analyzer = VLMAnalyzer(client, similarity_threshold=0.7)
        >>> is_correct, response, cost = await analyzer.analyze(image_bytes, "cat")
    """

    def __init__(self, vlm_client: VLMClient, similarity_threshold: float = 0.7) -> None:
        """Initialize VLMAnalyzer.

        Args:
            vlm_client: VLMClient instance for API calls
            similarity_threshold: Threshold for considering a VLM response correct
        """
        self._vlm_client = vlm_client
        self._similarity_threshold = similarity_threshold

    async def analyze(
        self, image_bytes: bytes, expected_subject: str, vlm_model_id: str
    ) -> tuple[str, float, bool]:
        """Analyze an image and compare with expected subject.

        Args:
            image_bytes: PNG image bytes to analyze
            expected_subject: Expected subject from the prompt
            vlm_model_id: VLM model ID to use for analysis

        Returns:
            Tuple of (vlm_response, cost, is_correct)

        Example:
            >>> analyzer = VLMAnalyzer(client, similarity_threshold=0.7)
            >>> response, cost, is_correct = await analyzer.analyze(
            ...     image_bytes, "cat", "openai/gpt-4o"
            ... )
        """
        vlm_response, cost = await self._vlm_client.analyze_image(image_bytes, vlm_model_id)

        similarity_score = await compute_similarity(expected_subject, vlm_response)
        is_correct = similarity_score >= self._similarity_threshold

        return vlm_response, cost, is_correct


class EvaluationWriter:
    """Service for persisting VLM evaluations to JSONL.

    This class encapsulates JSONL persistence for evaluations,
    making it testable and reusable.

    Args:
        evaluations_path: Path to vlm_evaluations.jsonl file

    Example:
        >>> writer = EvaluationWriter(Path("data/vlm_evaluations.jsonl"))
        >>> writer.write(evaluation)
    """

    def __init__(self, evaluations_path: Path) -> None:
        """Initialize EvaluationWriter.

        Args:
            evaluations_path: Path to vlm_evaluations.jsonl file
        """
        self._evaluations_path = Path(evaluations_path)

    def write(self, evaluation: VLMEvaluation) -> None:
        """Write a single evaluation to JSONL file.

        Args:
            evaluation: VLMEvaluation instance to write

        Example:
            >>> writer = EvaluationWriter(Path("data/vlm_evaluations.jsonl"))
            >>> writer.write(VLMEvaluation(
            ...     sample_id="123",
            ...     vlm_model_id="openai/gpt-4o",
            ...     expected_subject="cat",
            ...     vlm_response="A cat",
            ...     similarity_score=0.95,
            ...     is_correct=True
            ... ))
        """
        append_jsonl(self._evaluations_path, evaluation)


class EvaluationOrchestrator:
    """Orchestrator for the VLM evaluation pipeline.

    This class composes ImageRenderer, VLMAnalyzer, and EvaluationWriter
    to coordinate the full evaluation workflow. It handles concurrent
    processing with idempotent resume capability.

    Args:
        renderer: ImageRenderer instance for ASCII-to-image conversion
        vlm_analyzer: VLMAnalyzer instance for VLM API calls
        evaluation_writer: EvaluationWriter instance for JSONL persistence
        config: EvaluatorConfig with settings

    Example:
        >>> repo = DataRepository()
        >>> config = load_evaluator_config()
        >>> renderer = ImageRenderer(RendererConfig(font=config.font))
        >>> analyzer = VLMAnalyzer(VLMClient(Settings()), config.similarity_threshold)
        >>> writer = EvaluationWriter(Path("data/vlm_evaluations.jsonl"))
        >>> orchestrator = EvaluationOrchestrator(renderer, analyzer, writer, config)
        >>> results = await orchestrator.run(samples, existing_evaluations)
    """

    def __init__(
        self,
        renderer: ImageRenderer,
        vlm_analyzer: VLMAnalyzer,
        evaluation_writer: EvaluationWriter,
        config: EvaluatorConfig,
    ) -> None:
        """Initialize EvaluationOrchestrator.

        Args:
            renderer: ImageRenderer instance for ASCII-to-image conversion
            vlm_analyzer: VLMAnalyzer instance for VLM API calls
            evaluation_writer: EvaluationWriter instance for JSONL persistence
            config: EvaluatorConfig with settings
        """
        self._renderer = renderer
        self._vlm_analyzer = vlm_analyzer
        self._evaluation_writer = evaluation_writer
        self._config = config

    async def run(
        self,
        samples: list[ArtSample],
        existing_evaluations: list[VLMEvaluation],
        progress_callback: Callable[[int, int, str], None] | None = None,
        limit: int | None = None,
    ) -> list[VLMEvaluation]:
        """Run evaluation pipeline on samples.

        Args:
            samples: List of ArtSample objects to evaluate
            existing_evaluations: List of existing VLMEvaluation objects for idempotency
            progress_callback: Optional callback called after each evaluation
            limit: Optional limit on the number of evaluation tasks to process

        Returns:
            List of newly created VLMEvaluation objects

        Example:
            >>> orchestrator = EvaluationOrchestrator(renderer, analyzer, writer, config)
            >>> results = await orchestrator.run(samples, existing_evaluations)
        """
        existing_eval_keys = {(str(ev.sample_id), ev.vlm_model_id) for ev in existing_evaluations}

        tasks_to_process = []
        for sample in samples:
            for vlm_model_id in self._config.vlm_models:
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
        ) -> VLMEvaluation | None:
            """Process a single sample evaluation task.

            Args:
                sample: ArtSample to evaluate
                vlm_model_id: VLM model to use for evaluation

            Returns:
                VLMEvaluation if successful, None if error occurs
            """
            try:
                expected_subject = extract_subject(sample.prompt_text)

                image_bytes = self._renderer.render(sample.sanitized_output)

                vlm_response, cost, is_correct = await self._vlm_analyzer.analyze(
                    image_bytes, expected_subject, vlm_model_id
                )

                evaluation = VLMEvaluation(
                    sample_id=str(sample.id),
                    vlm_model_id=vlm_model_id,
                    expected_subject=expected_subject,
                    vlm_response=vlm_response,
                    similarity_score=0.0,
                    is_correct=is_correct,
                    cost=cost,
                )

                self._evaluation_writer.write(evaluation)

                logger.info(
                    "Evaluation completed",
                    {
                        "sample_id": str(sample.id),
                        "vlm_model_id": vlm_model_id,
                        "expected_subject": expected_subject,
                        "vlm_response": vlm_response[:50],
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

        semaphore = asyncio.Semaphore(self._config.max_concurrency)

        async def process_with_semaphore(
            sample: ArtSample,
            vlm_model_id: str,
        ) -> VLMEvaluation | None:
            async with semaphore:
                return await process_task(sample, vlm_model_id)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            expand=True,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Evaluating samples with {len(self._config.vlm_models)} VLM model(s)...",
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

    renderer = ImageRenderer(renderer_config)
    vlm_client = VLMClient(settings)
    vlm_analyzer = VLMAnalyzer(vlm_client, config.similarity_threshold)
    evaluation_writer = EvaluationWriter(evaluations_path)

    orchestrator = EvaluationOrchestrator(renderer, vlm_analyzer, evaluation_writer, config)

    return await orchestrator.run(valid_samples, existing_evaluations, progress_callback, limit)


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
