"""Generator module main entry point.

This module provides the main entry point for the Generator module,
which coordinates the sampling and sanitization of ASCII art samples
from language models.

Dependencies:
    - asciibench.common.models: Data models for ArtSample, Model, Prompt
    - asciibench.common.config: Configuration settings
    - asciibench.generator.client: OpenRouter API client using smolagents
    - asciibench.generator.sampler: Sample generation logic
    - asciibench.generator.sanitizer: ASCII art extraction and validation
"""

import sys

from asciibench.common.config import Settings
from asciibench.common.yaml_config import load_generation_config, load_models, load_prompts
from asciibench.generator.sampler import generate_samples


def _print_progress(model_id: str, prompt_text: str, attempt: int, remaining: int) -> None:
    """Print progress information to stdout.

    Args:
        model_id: Model identifier being used
        prompt_text: Prompt text being processed
        attempt: Current attempt number
        remaining: Number of samples remaining to process
    """
    # Truncate prompt for display
    prompt_display = prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text
    print(f"[{remaining} remaining] {model_id} | Attempt {attempt} | {prompt_display}")


def main() -> None:
    """Main entry point for the Generator module.

    This function coordinates the generation of ASCII art samples
    from configured language models by:
    1. Loading settings from .env and config.yaml
    2. Loading models from models.yaml
    3. Loading and expanding prompts from prompts.yaml
    4. Calling generate_samples() with loaded configuration
    5. Printing progress to stdout
    6. Printing summary on completion
    """
    print("ASCIIBench Generator")
    print("=" * 50)

    # Load settings from .env
    try:
        settings = Settings()
    except Exception as e:
        print(f"Error loading settings: {e}", file=sys.stderr)
        sys.exit(1)

    # Check for API key
    if not settings.openrouter_api_key:
        print(
            "Error: Missing OpenRouter API key.\n\n"
            "Please set the OPENROUTER_API_KEY environment variable or add it to your .env file:\n"
            "  OPENROUTER_API_KEY=your-api-key-here\n\n"
            "You can get an API key from: https://openrouter.ai/keys",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load generation config from config.yaml
    try:
        config = load_generation_config()
        print(
            f"Config loaded: {config.attempts_per_prompt} attempts per prompt, "
            f"temperature={config.temperature}, max_tokens={config.max_tokens}"
        )
    except Exception as e:
        print(f"Error loading config.yaml: {e}", file=sys.stderr)
        sys.exit(1)

    # Load models from models.yaml
    try:
        models = load_models()
        print(f"Models loaded: {len(models)} models")
        for model in models:
            print(f"  - {model.name} ({model.id})")
    except FileNotFoundError:
        print("Error: models.yaml not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading models.yaml: {e}", file=sys.stderr)
        sys.exit(1)

    # Load and expand prompts from prompts.yaml
    try:
        prompts = load_prompts()
        print(f"Prompts loaded: {len(prompts)} prompts")

        # Show category breakdown
        categories: dict[str, int] = {}
        for prompt in prompts:
            categories[prompt.category] = categories.get(prompt.category, 0) + 1
        for category, count in sorted(categories.items()):
            print(f"  - {category}: {count} prompts")
    except FileNotFoundError:
        print("Error: prompts.yaml not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading prompts.yaml: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate expected samples
    total_samples = len(models) * len(prompts) * config.attempts_per_prompt
    print(
        f"\nExpected samples: {len(models)} models x {len(prompts)} prompts x "
        f"{config.attempts_per_prompt} attempts = {total_samples} samples"
    )
    print("=" * 50)
    print("\nStarting generation...\n")

    # Generate samples with progress callback
    try:
        new_samples = generate_samples(
            models=models,
            prompts=prompts,
            config=config,
            settings=settings,
            progress_callback=_print_progress,
        )
    except Exception as e:
        print(f"\nError during generation: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 50)
    print("Generation Complete!")
    print("=" * 50)

    valid_count = sum(1 for s in new_samples if s.is_valid)
    invalid_count = len(new_samples) - valid_count

    print("\nSummary:")
    print(f"  Total new samples generated: {len(new_samples)}")
    print(f"  Valid samples: {valid_count}")
    print(f"  Invalid samples: {invalid_count}")

    if len(new_samples) == 0:
        print("\n  (No new samples generated - all combinations already exist in database)")

    print("\nSamples saved to: data/database.jsonl")


if __name__ == "__main__":
    main()
