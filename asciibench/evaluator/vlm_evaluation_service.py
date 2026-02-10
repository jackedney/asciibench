"""VLM evaluation service for single-sample evaluation.

This service provides a high-level interface for evaluating individual
ASCII art samples with VLMs. It is designed to be used by both the
judge UI (inline evaluation after voting) and potentially the batch
evaluator.
"""

from pathlib import Path

from asciibench.common.config import EvaluatorConfig, RendererConfig, Settings
from asciibench.common.logging import get_logger
from asciibench.common.models import ArtSample, VLMEvaluation
from asciibench.evaluator.orchestrator import (
    EvaluationWriter,
    ImageRenderer,
    VLMAnalyzer,
)
from asciibench.evaluator.subject_extractor import extract_subject
from asciibench.evaluator.vlm_client import VLMClient

logger = get_logger(__name__)


class VLMEvaluationService:
    """Service for evaluating individual samples with VLMs.

    Unlike EvaluationOrchestrator which processes batches with concurrency
    control, this service evaluates a single sample at a time. It handles:
    - Rendering ASCII art to an image
    - Calling VLM to analyze the image
    - Computing correctness
    - Persisting the result
    """

    def __init__(
        self,
        evaluations_path: Path,
        config: EvaluatorConfig | None = None,
    ) -> None:
        if config is None:
            from asciibench.common.yaml_config import load_evaluator_config

            config = load_evaluator_config()

        self._config = config
        renderer_config = RendererConfig(font=config.font)
        self._renderer = ImageRenderer(renderer_config)

        settings = Settings()
        vlm_client = VLMClient(settings)
        self._analyzer = VLMAnalyzer(vlm_client, config.similarity_threshold)
        self._writer = EvaluationWriter(evaluations_path)

    @property
    def vlm_models(self) -> list[str]:
        """VLM model IDs configured for evaluation."""
        return self._config.vlm_models

    async def evaluate_sample(
        self,
        sample: ArtSample,
        vlm_model_id: str,
    ) -> VLMEvaluation | None:
        """Evaluate a single sample with a single VLM model.

        Returns None on failure (logged, not raised).
        The evaluation is persisted immediately to the JSONL file.
        """
        try:
            expected_subject = extract_subject(sample.prompt_text)
            image_bytes = self._renderer.render(sample.sanitized_output)
            vlm_response, cost, is_correct = await self._analyzer.analyze(
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
            self._writer.write(evaluation)
            return evaluation
        except Exception:
            logger.error(
                "Single-sample evaluation failed",
                {
                    "sample_id": str(sample.id),
                    "vlm_model_id": vlm_model_id,
                },
            )
            return None

    async def evaluate_sample_all_models(
        self,
        sample: ArtSample,
        existing_keys: set[tuple[str, str]] | None = None,
    ) -> list[VLMEvaluation]:
        """Evaluate a sample with all configured VLM models.

        Skips models for which an evaluation already exists (idempotency).
        """
        results = []
        for vlm_model_id in self._config.vlm_models:
            if existing_keys and (str(sample.id), vlm_model_id) in existing_keys:
                continue
            result = await self.evaluate_sample(sample, vlm_model_id)
            if result:
                results.append(result)
        return results
