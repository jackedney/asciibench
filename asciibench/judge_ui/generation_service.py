"""GenerationService for on-demand sample generation.

This module provides a service for generating individual samples on demand
and persisting them immediately for tournament rounds.

Dependencies:
    - OpenRouterClient: API client for generating samples
    - GenerationConfig: Configuration for sample generation
    - extract_ascii_from_markdown: Function to extract ASCII from markdown
    - append_jsonl: Function to persist samples
"""

from pathlib import Path

from asciibench.common.config import GenerationConfig
from asciibench.common.models import ArtSample, Prompt, RoundState
from asciibench.common.persistence import append_jsonl
from asciibench.generator.client import OpenRouterClient
from asciibench.generator.sanitizer import extract_ascii_from_markdown


class GenerationService:
    """Service for generating individual samples on demand and persisting them.

    This service is used by the tournament system to generate missing samples
    for matchups as needed, rather than pre-generating all samples upfront.
    """

    def __init__(
        self,
        client: OpenRouterClient,
        config: GenerationConfig,
        database_path: Path,
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

    def find_existing_sample(
        self, model_id: str, prompt_text: str, samples: list[ArtSample]
    ) -> ArtSample | None:
        """Find an existing sample matching the model and prompt.

        Args:
            model_id: ID of the model to match
            prompt_text: Text of the prompt to match
            samples: List of existing samples to search

        Returns:
            First matching valid sample, or None if no match found
        """
        for sample in samples:
            if sample.model_id == model_id and sample.prompt_text == prompt_text:
                return sample
        return None

    async def generate_sample(self, model_id: str, prompt: Prompt) -> ArtSample:
        """Generate a single sample for the given model and prompt.

        This method calls the OpenRouter API, extracts ASCII art from markdown,
        creates an ArtSample, persists it, and returns it.

        Args:
            model_id: ID of the model to generate from
            prompt: Prompt object with text and category

        Returns:
            Generated ArtSample instance

        Raises:
            Exception: If the API call fails (caller handles retry/skip)
        """
        response = await self.client.generate_async(
            model_id=model_id,
            prompt=prompt.text,
            config=self.config,
        )

        sanitized_output = extract_ascii_from_markdown(response.text)

        sample = ArtSample(
            model_id=model_id,
            prompt_text=prompt.text,
            category=prompt.category,
            attempt_number=1,
            raw_output=response.text,
            sanitized_output=sanitized_output,
            is_valid=bool(sanitized_output),
            output_tokens=response.completion_tokens,
            cost=response.cost,
        )

        append_jsonl(self.database_path, sample)

        return sample

    async def ensure_samples_for_round(
        self, round_state: RoundState, existing_samples: list[ArtSample]
    ) -> RoundState:
        """Ensure all samples for matchups in the round exist.

        Iterates through matchups, finds or generates samples as needed,
        fills in sample_a_id and sample_b_id, sets generation_complete=True,
        and returns the updated RoundState.

        Args:
            round_state: RoundState with matchups to populate
            existing_samples: List of existing samples to search for matches

        Returns:
            Updated RoundState with sample IDs filled and generation_complete=True
        """
        updated_matchups = []

        for matchup in round_state.matchups:
            sample_a = self.find_existing_sample(
                matchup.model_a_id, matchup.prompt_text, existing_samples
            )

            sample_b = self.find_existing_sample(
                matchup.model_b_id, matchup.prompt_text, existing_samples
            )

            if sample_a is None:
                prompt = Prompt(
                    text=matchup.prompt_text,
                    category=matchup.prompt_category,
                    template_type="unknown",
                )
                sample_a = await self.generate_sample(matchup.model_a_id, prompt)
                existing_samples.append(sample_a)

            if sample_b is None:
                prompt = Prompt(
                    text=matchup.prompt_text,
                    category=matchup.prompt_category,
                    template_type="unknown",
                )
                sample_b = await self.generate_sample(matchup.model_b_id, prompt)
                existing_samples.append(sample_b)

            updated_matchup = matchup.model_copy(
                update={
                    "sample_a_id": str(sample_a.id),
                    "sample_b_id": str(sample_b.id),
                }
            )
            updated_matchups.append(updated_matchup)

        updated_round_state = round_state.model_copy(
            update={
                "matchups": updated_matchups,
                "generation_complete": True,
            }
        )

        return updated_round_state
