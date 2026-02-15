"""GenerationService for on-demand sample generation.

This module provides a service for generating samples on demand
and persisting them immediately for tournament rounds.

Dependencies:
    - asciibench.generator.concurrent: Concurrent generation primitive
    - asciibench.common.config: GenerationConfig for settings
    - asciibench.common.models: Data models for ArtSample, Matchup, RoundState
"""

import logging
from collections.abc import Callable

from asciibench.common.config import GenerationConfig
from asciibench.common.models import ArtSample, Matchup, RoundState
from asciibench.generator.client import OpenRouterClient
from asciibench.generator.concurrent import GenerationTask, generate_samples_concurrent

logger = logging.getLogger(__name__)


class GenerationService:
    """Service for generating samples on demand and persisting them.

    This service is used by the tournament system to generate missing samples
    for matchups using concurrent API calls.
    """

    def __init__(
        self,
        client: OpenRouterClient,
        config: GenerationConfig,
        database_path,
    ) -> None:
        """Initialize the GenerationService.

        Args:
            client: OpenRouterClient instance for API calls
            config: GenerationConfig for generation parameters
            database_path: Path to the JSONL database file for persisting samples
        """
        self.client = client
        self.config = config
        self.database_path = database_path

    async def ensure_samples_for_round(
        self,
        round_state: RoundState,
        existing_samples: list[ArtSample],
        on_matchup_ready: Callable[[int, Matchup], None] | None = None,
    ) -> RoundState:
        """Ensure all samples for matchups in the round exist.

        Collects all unique (model_id, prompt_text) pairs from matchups,
        generates missing samples concurrently, builds a lookup dict,
        fills sample_a_id/sample_b_id on all matchups, and fires callbacks.

        Args:
            round_state: RoundState with matchups to populate
            existing_samples: List of existing samples to search for matches
            on_matchup_ready: Optional callback called after each matchup is updated
                with (index, updated_matchup) where index is 0-based

        Returns:
            Updated RoundState with sample IDs filled and generation_complete=True
        """
        pairs_to_generate: set[tuple[str, str, str]] = set()

        for matchup in round_state.matchups:
            pairs_to_generate.add(
                (matchup.model_a_id, matchup.prompt_text, matchup.prompt_category)
            )
            pairs_to_generate.add(
                (matchup.model_b_id, matchup.prompt_text, matchup.prompt_category)
            )

        existing_keys: set[tuple[str, str, int]] = {
            (s.model_id, s.prompt_text, s.attempt_number) for s in existing_samples
        }

        tasks: list[GenerationTask] = []
        for model_id, prompt_text, category in pairs_to_generate:
            if (model_id, prompt_text, 1) not in existing_keys:
                tasks.append(
                    GenerationTask(
                        model_id=model_id,
                        prompt_text=prompt_text,
                        category=category,
                        attempt=1,
                    )
                )

        generated_samples: list[ArtSample] = []
        if tasks:
            models = {t.model_id for t in tasks}
            logger.info(
                "Dispatching %d LLM call(s) across %d model(s)",
                len(tasks),
                len(models),
            )
        else:
            logger.info("All samples already exist, skipping generation")

        if tasks:
            generated_samples = await generate_samples_concurrent(
                tasks=tasks,
                client=self.client,
                config=self.config,
                database_path=self.database_path,
                existing_keys=existing_keys,
                max_concurrent=self.config.max_concurrent_requests,
            )
            logger.info("All %d LLM calls completed", len(tasks))

        sample_lookup: dict[tuple[str, str], ArtSample] = {}
        for sample in existing_samples:
            if not sample.is_valid:
                continue
            key = (sample.model_id, sample.prompt_text)
            if key not in sample_lookup:
                sample_lookup[key] = sample
        for sample in generated_samples:
            if not sample.is_valid:
                continue
            key = (sample.model_id, sample.prompt_text)
            if key not in sample_lookup:
                sample_lookup[key] = sample

        updated_matchups: list[Matchup] = []
        for index, matchup in enumerate(round_state.matchups):
            sample_a = sample_lookup.get((matchup.model_a_id, matchup.prompt_text))
            sample_b = sample_lookup.get((matchup.model_b_id, matchup.prompt_text))

            updated_matchup = matchup.model_copy(
                update={
                    "sample_a_id": str(sample_a.id) if sample_a else None,
                    "sample_b_id": str(sample_b.id) if sample_b else None,
                }
            )
            updated_matchups.append(updated_matchup)

            if on_matchup_ready is not None:
                on_matchup_ready(index, updated_matchup)

        updated_round_state = round_state.model_copy(
            update={
                "matchups": updated_matchups,
                "generation_complete": True,
            }
        )

        return updated_round_state
