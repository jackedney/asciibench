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

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from asciibench.common.config import Settings
from asciibench.common.display import (
    create_live_stats,
    get_console,
    print_banner,
    success_badge,
    update_live_stats,
)
from asciibench.common.models import ArtSample
from asciibench.common.yaml_config import load_generation_config, load_models, load_prompts
from asciibench.generator.sampler import generate_samples


def _print_progress(model_id: str, prompt_text: str, attempt: int, remaining: int) -> None:
    """Print progress information to stdout (deprecated, kept for test compatibility).

    Args:
        model_id: Model identifier being used
        prompt_text: Prompt text being processed
        attempt: Current attempt number
        remaining: Number of samples remaining to process
    """
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
    5. Displaying progress with Rich components
    6. Displaying summary with Rich components
    """
    console = get_console()

    print_banner()

    console.print("\n[info]Loading configuration...[/info]\n")

    # Load settings from .env
    try:
        settings = Settings()
    except Exception as e:
        console.print(f"[error]Error loading settings: {e}[/error]")
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
        config_text = Text.assemble(
            ("Attempts per prompt: ", "info"),
            (f"{config.attempts_per_prompt}", "accent bold"),
            (" | Temperature: ", "info"),
            (f"{config.temperature}", "accent bold"),
            (" | Max tokens: ", "info"),
            (f"{config.max_tokens}", "accent bold"),
        )
        console.print(
            Panel(config_text, title="[primary]Generation Config[/primary]", border_style="accent")
        )
    except Exception as e:
        console.print(f"[error]Error loading config.yaml: {e}[/error]")
        sys.exit(1)

    console.print()

    # Load models from models.yaml
    try:
        models = load_models()
        models_table = Table(title=f"Models ({len(models)} loaded)", border_style="accent")
        models_table.add_column("Name", style="bold")
        models_table.add_column("ID")

        for model in models:
            models_table.add_row(model.name, model.id)

        console.print(models_table)
    except FileNotFoundError:
        console.print("[error]Error: models.yaml not found[/error]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[error]Error loading models.yaml: {e}[/error]")
        sys.exit(1)

    console.print()

    # Load and expand prompts from prompts.yaml
    try:
        prompts = load_prompts()
        console.print(f"[info]Prompts loaded: {len(prompts)}[/info]")

        # Show category breakdown
        categories: dict[str, int] = {}
        for prompt in prompts:
            categories[prompt.category] = categories.get(prompt.category, 0) + 1

        for category, count in sorted(categories.items()):
            console.print(f"  - {category}: {count} prompts")
    except FileNotFoundError:
        console.print("[error]Error: prompts.yaml not found[/error]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[error]Error loading prompts.yaml: {e}[/error]")
        sys.exit(1)

    # Calculate expected samples
    total_expected = len(models) * len(prompts) * config.attempts_per_prompt
    console.print(
        f"\n[info]Expected samples: {len(models)} models x {len(prompts)} prompts x "
        f"{config.attempts_per_prompt} attempts = {total_expected} samples[/info]\n"
    )

    if total_expected == 0:
        console.print(
            Panel(
                "[warning]Nothing to generate![/warning]\n\nNo models or prompts configured.",
                border_style="warning",
            )
        )
        return

    # Track generation progress
    samples_processed = 0

    def _rich_progress_callback(
        model_id: str, prompt_text: str, attempt: int, remaining: int
    ) -> None:
        """Progress callback using Rich display."""
        nonlocal samples_processed

        # Track total samples processed
        total = total_expected - remaining + 1
        samples_processed = total

        # Update live stats with running total
        update_live_stats(live_stats, live, total=total, valid=0, invalid=0)

    # Initialize live stats
    live_stats, live = create_live_stats()
    samples_generated: list[ArtSample] = []

    # Generate samples with progress callback
    try:
        if live is not None:
            with live:
                new_samples = generate_samples(
                    models=models,
                    prompts=prompts,
                    config=config,
                    settings=settings,
                    progress_callback=_rich_progress_callback,
                )
        else:
            new_samples = generate_samples(
                models=models,
                prompts=prompts,
                config=config,
                settings=settings,
                progress_callback=_rich_progress_callback,
            )
    except Exception as e:
        console.print(f"\n[error]Error during generation: {e}[/error]")
        sys.exit(1)

    samples_generated = new_samples

    # Calculate final stats from newly generated samples
    valid_count = sum(1 for s in samples_generated if s.is_valid)
    invalid_count = len(samples_generated) - valid_count

    # Update final stats display
    if live is not None:
        update_live_stats(
            live_stats, live, total=len(samples_generated), valid=valid_count, invalid=invalid_count
        )

    # Final summary panel
    console.print()

    if len(samples_generated) == 0:
        console.print(
            Panel(
                f"{success_badge()} All samples already exist in database.\n\n"
                "No new samples generated.",
                title="[info]Generation Complete[/info]",
                border_style="info",
            )
        )
    else:
        summary_text = Text.assemble(
            ("Total new samples generated: ", "info"),
            (f"{len(samples_generated)}", "accent bold"),
            ("\nValid samples: ", "success"),
            (f"{valid_count}", "success bold"),
            ("\nInvalid samples: ", "error"),
            (f"{invalid_count}", "error bold"),
        )
        console.print(
            Panel(
                summary_text,
                title="[success]Generation Complete![/success]",
                border_style="success",
            )
        )

    console.print("\n[info]Samples saved to: data/database.jsonl[/info]")


if __name__ == "__main__":
    main()
