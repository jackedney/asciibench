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
from rich.text import Text

from asciibench.common.config import Settings
from asciibench.common.display import get_console, success_badge
from asciibench.common.loader import RuneScapeLoader
from asciibench.common.models import ArtSample
from asciibench.common.simple_display import create_loader, show_banner, show_prompt
from asciibench.common.yaml_config import load_generation_config, load_models, load_prompts
from asciibench.generator.sampler import generate_samples


def show_stats(success_count: int, failure_count: int, running_cost: float) -> None:
    """Display current stats line with colored success/failure counts.

    Args:
        success_count: Number of successfully generated samples
        failure_count: Number of failed generations
        running_cost: Total cost of all successful generations

    Example:
        >>> show_stats(3, 1, 0.001234)
        # Prints: ✓ 3 | ✗ 1 | Cost: $0.001234 (with colors)
    """
    console = get_console()
    stats_text = Text.assemble(
        ("✓ ", "green bold"),
        (f"{success_count}", "green bold"),
        (" | ", "info"),
        ("✗ ", "red bold"),
        (f"{failure_count}", "red bold"),
        (" | Cost: $", "info"),
        (f"{running_cost:.6f}", "accent bold"),
    )
    console.print(stats_text)


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
    5. Displaying progress with RuneScape-style loader
    6. Displaying summary with Rich components
    """
    console = get_console()

    # Show ASCII banner at startup
    show_banner()

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
    except Exception as e:
        console.print(f"[error]Error loading config.yaml: {e}[/error]")
        sys.exit(1)

    # Load models from models.yaml
    try:
        models = load_models()
    except FileNotFoundError:
        console.print("[error]Error: models.yaml not found[/error]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[error]Error loading models.yaml: {e}[/error]")
        sys.exit(1)

    # Load and expand prompts from prompts.yaml
    try:
        prompts = load_prompts()
    except FileNotFoundError:
        console.print("[error]Error: prompts.yaml not found[/error]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[error]Error loading prompts.yaml: {e}[/error]")
        sys.exit(1)

    # Calculate expected samples
    total_expected = len(models) * len(prompts) * config.attempts_per_prompt

    if total_expected == 0:
        console.print(
            Panel(
                "[warning]Nothing to generate![/warning]\n\nNo models or prompts configured.",
                border_style="warning",
            )
        )
        return

    # Track state for the RuneScape loader
    current_model_id: str | None = None
    current_prompt_text: str | None = None
    samples_for_current_model = 0
    total_samples_per_model = len(prompts) * config.attempts_per_prompt
    loader: RuneScapeLoader | None = None
    samples_generated: list[ArtSample] = []

    # Track stats across all samples
    success_count = 0
    failure_count = 0
    running_cost = 0.0
    current_model_cost = 0.0

    def _loader_progress_callback(
        model_id: str, prompt_text: str, attempt: int, remaining: int
    ) -> None:
        """Progress callback using RuneScape loader."""
        nonlocal \
            current_model_id, \
            current_prompt_text, \
            samples_for_current_model, \
            loader, \
            current_model_cost

        # Detect model change
        if model_id != current_model_id:
            # Complete previous loader if exists
            if loader is not None:
                loader.complete(success=True, cost=current_model_cost)
                loader.stop()
                current_model_cost = 0.0

            # Reset for new model
            current_model_id = model_id
            samples_for_current_model = 0

            # Show first prompt for this model batch
            show_prompt(prompt_text)
            current_prompt_text = prompt_text

            # Display stats above progress bar before each model batch starts
            show_stats(success_count, failure_count, running_cost)
            console.print()

            # Create and start new loader for this model
            loader = create_loader(model_id, total_samples_per_model)
            loader.start()

        # Update progress
        samples_for_current_model += 1
        if loader is not None:
            loader.update(samples_for_current_model)

    def _stats_callback(is_valid: bool, cost: float | None) -> None:
        """Stats callback called after each sample generation."""
        nonlocal success_count, failure_count, running_cost, current_model_cost

        actual_cost = cost if cost is not None else 0.0
        if is_valid:
            success_count += 1
            running_cost += actual_cost
            current_model_cost += actual_cost
        else:
            failure_count += 1

    # Generate samples with progress and stats callbacks
    try:
        samples_generated = generate_samples(
            models=models,
            prompts=prompts,
            config=config,
            settings=settings,
            progress_callback=_loader_progress_callback,
            stats_callback=_stats_callback,
        )

        # Complete the final loader
        if loader is not None:
            loader.complete(success=True, cost=current_model_cost)
            loader.stop()

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        if loader is not None:
            loader.complete(success=False, cost=current_model_cost)
            loader.stop()
        console.print("\n[warning]Generation interrupted by user.[/warning]")
        sys.exit(0)
    except Exception as e:
        # Handle other errors
        if loader is not None:
            loader.complete(success=False, cost=current_model_cost)
            loader.stop()
        console.print(f"\n[error]Error during generation: {e}[/error]")
        sys.exit(1)

    # Calculate final stats from newly generated samples
    valid_count = sum(1 for s in samples_generated if s.is_valid)
    invalid_count = len(samples_generated) - valid_count

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
