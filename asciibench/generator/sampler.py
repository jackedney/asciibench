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

from collections.abc import Callable
from pathlib import Path

from asciibench.common.config import GenerationConfig, Settings
from asciibench.common.models import ArtSample, Model, Prompt
from asciibench.common.persistence import append_jsonl, read_jsonl
from asciibench.generator.client import OpenRouterClient, OpenRouterClientError
from asciibench.generator.sanitizer import extract_ascii_from_markdown

# Type alias for progress callback: (model_id, prompt_text, attempt_number, total_remaining) -> None
ProgressCallback = Callable[[str, str, int, int], None]


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


def generate_samples(
    models: list[Model],
    prompts: list[Prompt],
    config: GenerationConfig,
    database_path: str | Path = "data/database.jsonl",
    client: OpenRouterClient | None = None,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[ArtSample]:
    """Generate ASCII art samples from configured models and prompts.

    This function coordinates the generation of multiple samples by:
    1. Iterating through each configured model and prompt combination
    2. Generating multiple attempts per prompt based on config.attempts_per_prompt
    3. Checking for existing samples to support idempotent resume capability
    4. Applying generation settings (temperature, max_tokens, etc.)
    5. Sanitizing outputs and validating them
    6. Persisting each sample immediately for resume capability

    Args:
        models: List of Model objects to generate samples from
        prompts: List of Prompt objects with ASCII art generation prompts
        config: GenerationConfig with settings (attempts_per_prompt, temperature, etc.)
        database_path: Path to the database JSONL file (default: data/database.jsonl)
        client: Optional OpenRouterClient instance (created from settings if not provided)
        settings: Optional Settings instance (required if client not provided)
        progress_callback: Optional callback called before each sample generation
            with (model_id, prompt_text, attempt_number, total_remaining)

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
    samples_processed = 0

    # Iterate through all combinations: models x prompts x attempts
    for model in models:
        for prompt in prompts:
            for attempt in range(1, config.attempts_per_prompt + 1):
                samples_processed += 1

                # Check idempotency - skip if sample already exists
                if _sample_exists(model.id, prompt.text, attempt, existing_keys):
                    continue

                # Call progress callback before generation
                if progress_callback is not None:
                    remaining = total_combinations - samples_processed + 1
                    progress_callback(model.id, prompt.text, attempt, remaining)

                # Generate new sample
                try:
                    raw_output = client.generate(
                        model_id=model.id,
                        prompt=prompt.text,
                        config=config,
                    )
                except OpenRouterClientError:
                    # API error - skip this sample but continue with others
                    # Create a failed sample record for tracking
                    sample = ArtSample(
                        model_id=model.id,
                        prompt_text=prompt.text,
                        category=prompt.category,
                        attempt_number=attempt,
                        raw_output="",
                        sanitized_output="",
                        is_valid=False,
                    )
                    append_jsonl(database_path, sample)
                    existing_keys.add((model.id, prompt.text, attempt))
                    newly_generated.append(sample)
                    continue

                # Sanitize output
                sanitized_output = extract_ascii_from_markdown(raw_output)

                # Validate output:
                # - Invalid if no code block found (empty sanitized output)
                # - Invalid if output exceeds max_tokens (rough estimate: 4 chars per token)
                is_valid = bool(sanitized_output) and len(raw_output) <= config.max_tokens * 4

                # Create sample
                sample = ArtSample(
                    model_id=model.id,
                    prompt_text=prompt.text,
                    category=prompt.category,
                    attempt_number=attempt,
                    raw_output=raw_output,
                    sanitized_output=sanitized_output,
                    is_valid=is_valid,
                )

                # Persist immediately for resume capability
                append_jsonl(database_path, sample)

                # Update tracking for idempotency
                existing_keys.add((model.id, prompt.text, attempt))
                newly_generated.append(sample)

    return newly_generated
