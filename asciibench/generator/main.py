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
from asciibench.common.display import (
    create_loader,
    get_console,
    get_stderr_console,
    show_banner,
    success_badge,
)
from asciibench.common.models import ArtSample
from asciibench.common.observability import init_logfire
from asciibench.common.persistence import read_jsonl
from asciibench.common.yaml_config import load_generation_config, load_models, load_prompts
from asciibench.generator.sampler import generate_samples

# Default database path (same as sampler)
DEFAULT_DATABASE_PATH = "data/database.jsonl"


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
    console = get_console()
    prompt_display = prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text
    console.print(
        f"[info][{remaining} remaining] {model_id} | Attempt {attempt} | {prompt_display}[/info]"
    )


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

    # Load settings from .env
    try:
        settings = Settings()
    except Exception as e:
        console.print(f"[error]Error loading settings: {e}[/error]")
        sys.exit(1)

    # Initialize Logfire if enabled
    init_logfire(settings)

    # Check for API key
    if not settings.openrouter_api_key:
        console = get_stderr_console()
        console.print(
            "[error]Error: Missing OpenRouter API key.\n\n"
            "Please set the OPENROUTER_API_KEY environment variable or add it to your .env file:\n"
            "  OPENROUTER_API_KEY=your-api-key-here\n\n"
            "You can get an API key from: https://openrouter.ai/keys[/error]"
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

    # Calculate total combinations
    total_combinations = len(models) * len(prompts) * config.attempts_per_prompt

    if total_combinations == 0:
        console.print(
            Panel(
                "[warning]Nothing to generate![/warning]\n\nNo models or prompts configured.",
                border_style="warning",
            )
        )
        return

    # Load existing samples to count how many will be skipped when resuming
    existing_samples = read_jsonl(DEFAULT_DATABASE_PATH, ArtSample)
    existing_keys = {(s.model_id, s.prompt_text, s.attempt_number) for s in existing_samples}

    # Count how many planned combinations already exist in the database
    existing_count = sum(
        1
        for model in models
        for prompt in prompts
        for attempt in range(1, config.attempts_per_prompt + 1)
        if (model.id, prompt.text, attempt) in existing_keys
    )

    # Calculate samples that actually need to be generated
    total_expected = total_combinations - existing_count

    # Display loaded configuration like demo.py
    console.print()
    console.print(
        f"  [dim]•[/dim] [bold cyan]{len(models)}[/bold cyan] "
        f"[dim]models loaded from[/dim] [white]models.yaml[/white]"
    )
    console.print(
        f"  [dim]•[/dim] [bold cyan]{len(prompts)}[/bold cyan] "
        f"[dim]prompts loaded from[/dim] [white]prompts.yaml[/white]"
    )
    if existing_count > 0:
        console.print(
            f"  [dim]•[/dim] [bold yellow]{total_expected}[/bold yellow] "
            f"[dim]samples to generate[/dim] [dim]({existing_count} existing, skipped)[/dim]"
        )
    else:
        console.print(
            f"  [dim]•[/dim] [bold yellow]{total_expected}[/bold yellow] "
            f"[dim]total samples to generate[/dim]"
        )
    console.print()

    # Track samples generated
    samples_generated: list[ArtSample] = []

    # Build model_id -> model.name mapping for display
    model_names = {m.id: m.name for m in models}

    # Track total progress across all samples
    total_completed = 0

    # Create a single loader for total progress
    loader = create_loader("Generating", total_expected)

    def _loader_progress_callback(
        model_id: str,
        prompt_text: str,
        attempt: int,
        remaining: int,
    ) -> None:
        """Progress callback - updates model name and prompt display.

        Args:
            model_id: The model identifier
            prompt_text: The prompt text being processed
            attempt: Current attempt number (unused, but part of callback signature)
            remaining: Remaining attempts (unused, but part of callback signature)
        """
        # Update model name (use display name from mapping, without resetting progress)
        display_name = model_names.get(model_id, model_id)
        loader.set_model_name(display_name)

        # Update the current prompt in the loader (in-place update)
        loader.set_prompt(prompt_text)

    def _stats_callback(is_valid: bool, cost: float | None) -> None:
        """Stats callback - updates success/failure/cost counters and progress."""
        nonlocal total_completed
        actual_cost = cost if cost is not None else 0.0
        loader.record_result(is_valid, actual_cost)

        # Increment progress for every completed attempt (both success and failure)
        total_completed += 1
        loader.update(total_completed)

    # Generate samples with progress and stats callbacks
    try:
        loader.start()
        samples_generated = generate_samples(
            models=models,
            prompts=prompts,
            config=config,
            settings=settings,
            progress_callback=_loader_progress_callback,
            stats_callback=_stats_callback,
        )
        loader.stop()

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        loader.stop()
        console.print("\n[warning]Generation interrupted by user.[/warning]")
        sys.exit(0)
    except Exception as e:
        # Handle other errors
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
