"""Demo module for ASCII art generation.

This module provides a demo mode for generating ASCII art samples
from language models and displaying results in HTML format.

Demo generates ASCII art using a fixed prompt and saves outputs
to .demo_outputs/demo.html with incremental/resumable generation support.
"""

import html
import json
import traceback
from datetime import datetime
from pathlib import Path
from string import Template

from rich.panel import Panel
from rich.text import Text

from asciibench.common.config import Settings
from asciibench.common.config_service import ConfigService, ConfigServiceError
from asciibench.common.display import (
    create_loader,
    get_console,
    show_banner,
    show_prompt,
)
from asciibench.common.logging import generate_id, get_logger, set_request_id, set_run_id
from asciibench.common.models import DemoResult, OpenRouterResponse
from asciibench.generator.client import (
    AuthenticationError,
    ModelError,
    OpenRouterClient,
    OpenRouterClientError,
)
from asciibench.generator.sanitizer import extract_ascii_from_markdown

DEMO_OUTPUTS_DIR = Path(".demo_outputs")
RESULTS_JSON_PATH = DEMO_OUTPUTS_DIR / "results.json"
DEMO_HTML_PATH = DEMO_OUTPUTS_DIR / "demo.html"
ERRORS_LOG_PATH = DEMO_OUTPUTS_DIR / "errors.log"
TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates" / "demo"

logger = get_logger("generator.demo")


def log_generation_error(
    model_id: str,
    model_name: str,
    error_reason: str,
    raw_output: str | None = None,
    exception: Exception | None = None,
) -> None:
    """Log a generation error with full context using structured logger.

    Args:
        model_id: Model identifier (e.g., 'openai/gpt-4o-mini')
        model_name: Human-readable model name
        error_reason: Brief description of why generation failed
        raw_output: The raw output received from the model (if any)
        exception: The exception that was raised (if any)
    """
    metadata = {
        "model": model_name,
        "model_id": model_id,
        "error_reason": error_reason,
    }

    if exception:
        metadata["exception_type"] = type(exception).__name__
        metadata["exception_message"] = str(exception)
        if exception.__traceback__:
            tb_lines = traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
            metadata["traceback"] = "".join(tb_lines)
        else:
            metadata["traceback"] = ""

    if raw_output:
        # Truncate very long outputs but keep enough for debugging
        truncated = raw_output[:2000] + "..." if len(raw_output) > 2000 else raw_output
        metadata["raw_output_length"] = str(len(raw_output))
        metadata["raw_output_preview"] = truncated

    logger.error("Demo generation failed", metadata)


def load_demo_results() -> list[DemoResult]:
    """Load existing demo results from results.json.

    Creates .demo_outputs directory if it doesn't exist.
    Returns empty list if file doesn't exist or is corrupted.

    Returns:
        List of DemoResult objects loaded from results.json

    Example:
        >>> results = load_demo_results()
        >>> len(results)
        5
        >>> results[0].model_id
        'openai/gpt-4o-mini'

    Negative case:
        >>> if not RESULTS_JSON_PATH.exists():
        ...     results = load_demo_results()
        ...     len(results) == 0
        True
    """
    DEMO_OUTPUTS_DIR.mkdir(exist_ok=True)

    if not RESULTS_JSON_PATH.exists():
        return []

    try:
        with RESULTS_JSON_PATH.open("r") as f:
            data = json.load(f)
        return [DemoResult(**item) for item in data]
    except (json.JSONDecodeError, ValueError) as e:
        console = get_console()
        console.print(f"[warning]Warning: Corrupted results.json: {e}[/warning]")
        console.print("[info]Starting with empty results.[/info]")
        return []


def save_demo_results(results: list[DemoResult]) -> None:
    """Save demo results to results.json.

    Args:
        results: List of DemoResult objects to save

    Example:
        >>> from datetime import datetime
        >>> result = DemoResult(
        ...     model_id="openai/gpt-4o-mini",
        ...     model_name="GPT-4o-mini",
        ...     ascii_output="...ascii art...",
        ...     is_valid=True,
        ...     timestamp=datetime.now()
        ... )
        >>> save_demo_results([result])
    """
    DEMO_OUTPUTS_DIR.mkdir(exist_ok=True)

    data = [result.model_dump(mode="json") for result in results]
    with RESULTS_JSON_PATH.open("w") as f:
        json.dump(data, f, indent=2)


def get_completed_model_ids() -> set[str]:
    """Get set of model IDs that already have completed results.

    Returns:
        Set of model_id strings that have been generated

    Example:
        >>> ids = get_completed_model_ids()
        >>> 'openai/gpt-4o-mini' in ids
        True
    """
    results = load_demo_results()
    return {result.model_id for result in results}


def generate_demo_sample(
    model_id: str,
    model_name: str,
) -> DemoResult:
    """Generate a demo ASCII art sample from a single model.

    Uses hardcoded prompt 'Draw a skeleton in ASCII art' and loads
    generation configuration from config.yaml.

    Args:
        model_id: Model identifier (e.g., 'openai/gpt-4o-mini')
        model_name: Human-readable model name (e.g., 'GPT-4o Mini')

    Returns:
        DemoResult with model info, ascii_output, is_valid, timestamp,
        and error details if failed

    Example:
        >>> result = generate_demo_sample('openai/gpt-4o-mini', 'GPT-4o Mini')
        >>> result.model_id
        'openai/gpt-4o-mini'
        >>> result.model_name
        'GPT-4o Mini'
        >>> result.is_valid
        True
        >>> len(result.ascii_output) > 0
        True

    Negative case:
        >>> # With missing API key, returns invalid result with error message
        >>> result = generate_demo_sample('openai/gpt-4o-mini', 'GPT-4o Mini')
        >>> result.is_valid
        False
        >>> 'Error' in result.ascii_output
        True
    """
    # Generate and set run_id for this demo run
    run_id = generate_id()
    set_run_id(run_id)

    # Generate and set request_id for this sample
    request_id = generate_id()
    set_request_id(request_id)

    settings = Settings()
    config = ConfigService().get_app_config()
    demo_prompt = "Draw a skeleton in ASCII art"

    client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        base_url=settings.base_url,
        timeout=settings.openrouter_timeout_seconds,
    )

    response_raw_output: str | None = None
    error_reason: str | None = None
    last_exception: Exception | None = None
    response: OpenRouterResponse | None = None

    try:
        response = client.generate(model_id, demo_prompt, config=config)
        response_raw_output = response.text
        ascii_output = extract_ascii_from_markdown(response_raw_output)
        is_valid = bool(ascii_output)

        if is_valid:
            return DemoResult(
                model_id=model_id,
                model_name=model_name,
                ascii_output=ascii_output,
                is_valid=True,
                timestamp=datetime.now(),
                error_reason=None,
                raw_output=None,
                output_tokens=response.completion_tokens,
                cost=response.cost,
            )

        error_reason = "No valid ASCII art block found in output"

    except (AuthenticationError, ModelError, OpenRouterClientError) as e:
        error_reason = f"API error: {type(e).__name__}"
        last_exception = e
    except Exception as e:
        error_reason = f"Unexpected error: {type(e).__name__}"
        last_exception = e

    if error_reason:
        log_generation_error(
            model_id=model_id,
            model_name=model_name,
            error_reason=error_reason,
            raw_output=response_raw_output,
            exception=last_exception,
        )

    ascii_output = f"Error: {error_reason}"
    if last_exception:
        ascii_output += f" ({last_exception})"

    return DemoResult(
        model_id=model_id,
        model_name=model_name,
        ascii_output=ascii_output,
        is_valid=False,
        timestamp=datetime.now(),
        error_reason=error_reason,
        raw_output=response_raw_output,
        output_tokens=None,
        cost=None,
    )


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


def _read_template(template_name: str) -> str:
    """Read a template file from the templates directory.

    Args:
        template_name: Name of the template file (e.g., 'demo.html')

    Returns:
        Template file contents as string

    Raises:
        FileNotFoundError: If template file does not exist
    """
    template_path = TEMPLATE_DIR / template_name
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template file not found: {template_path}. "
            f"Please ensure the template directory exists at {TEMPLATE_DIR}"
        )
    return template_path.read_text(encoding="utf-8")


def generate_html() -> None:
    """Generate HTML output from results.json.

    Creates .demo_outputs/demo.html with all model outputs displayed
    in a scrollable page with modern styling and filter controls.

    Features:
    - Summary stats bar with total, valid, invalid counts and cost
    - Filter buttons (All/Valid/Invalid) with minimal JavaScript
    - Modern card design with hover effects
    - Improved badge styling and typography
    - Responsive layout for mobile

    Empty results show 'No results yet' message.

    Raises:
        FileNotFoundError: If template files are missing
    """
    results = load_demo_results()

    DEMO_OUTPUTS_DIR.mkdir(exist_ok=True)

    # Calculate stats for summary bar
    total_count = len(results)
    valid_count = sum(1 for r in results if r.is_valid)
    invalid_count = total_count - valid_count
    total_cost = sum(r.cost or 0 for r in results)
    valid_pct = (valid_count / total_count * 100) if total_count > 0 else 0

    # Read base template
    base_template_str = _read_template("demo.html")
    base_template = Template(base_template_str)

    # Build content section
    if results:
        # Build stats bar
        stats_bar_template_str = _read_template("stats_bar.html")
        stats_bar_template = Template(stats_bar_template_str)
        stats_bar_html = stats_bar_template.substitute(
            total_count=str(total_count),
            valid_count=str(valid_count),
            invalid_count=str(invalid_count),
            total_cost=f"${total_cost:.4f}",
            valid_pct=f"{valid_pct:.1f}%",
            progress_width=f"{valid_pct}%",
        )

        # Build filter controls
        filter_controls_template_str = _read_template("filter_controls.html")
        filter_controls_template = Template(filter_controls_template_str)
        filter_controls_html = filter_controls_template.substitute(
            total_count=str(total_count),
            valid_count=str(valid_count),
            invalid_count=str(invalid_count),
        )

        # Build model cards
        model_card_template_str = _read_template("model_card.html")
        model_card_template = Template(model_card_template_str)

        model_cards_html = ""
        for result in results:
            timestamp_str = result.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            valid_class = "valid-model" if result.is_valid else "invalid"
            badge_class = "valid" if result.is_valid else "invalid"
            badge_icon = "✓" if result.is_valid else "✗"
            badge_text = "Valid" if result.is_valid else "Invalid"
            escaped_output = html.escape(result.ascii_output)
            cost_str = f"${result.cost:.6f}" if result.cost is not None else "$0.00"
            tokens_str = f"{result.output_tokens}" if result.output_tokens is not None else "N/A"
            data_valid = "true" if result.is_valid else "false"

            model_card_html = model_card_template.substitute(
                valid_class=valid_class,
                data_valid=data_valid,
                model_name=result.model_name,
                model_id=result.model_id,
                cost_str=cost_str,
                tokens_str=tokens_str,
                badge_class=badge_class,
                badge_icon=badge_icon,
                badge_text=badge_text,
                escaped_output=escaped_output,
                timestamp_str=timestamp_str,
            )
            model_cards_html += model_card_html
    else:
        # No results - show empty state
        stats_bar_html = ""
        filter_controls_html = ""
        model_cards_html = _read_template("no_results.html")

    # Substitute content into base template
    html_content = base_template.substitute(content=model_cards_html)

    # Insert stats bar and filter controls before models container
    if results:
        # Find the position of models-container div and insert stats/filter before it
        models_container_pos = html_content.find('<div class="models-container">')
        if models_container_pos != -1:
            html_content = (
                html_content[:models_container_pos]
                + stats_bar_html
                + "\n"
                + filter_controls_html
                + "\n"
                + html_content[models_container_pos:]
            )

    with DEMO_HTML_PATH.open("w", encoding="utf-8") as f:
        f.write(html_content)


def main() -> None:
    """Main entry point for the Demo module.

    This function runs the demo generator that:
    1. Displays ASCII banner
    2. Sets up error logging to .demo_outputs/errors.log
    3. Loads models from models.yaml
    4. Shows demo prompt above loading bar
    5. Generates ASCII art for each model using animated loader
    6. Saves results to .demo_outputs/results.json
    7. Generates HTML output to .demo_outputs/demo.html
    """
    console = get_console()

    # Show ASCII banner at startup
    show_banner()

    # Load models using ConfigService
    try:
        config_service = ConfigService()
        models = config_service.get_models()
    except ConfigServiceError as e:
        console.print(
            Panel(
                f"[error]Error loading configuration: {e}[/error]",
                title="[error]Error[/error]",
                border_style="error",
            )
        )
        return

    if not models:
        console.print(
            Panel(
                "[warning]No models found in models.yaml[/warning]",
                title="[warning]Warning[/warning]",
                border_style="warning",
            )
        )
        return

    console.print()
    console.print(
        f"  [dim]•[/dim] [bold cyan]{len(models)}[/bold cyan] "
        f"[dim]models loaded from[/dim] [white]models.yaml[/white]"
    )

    results = load_demo_results()
    completed_ids = get_completed_model_ids()
    remaining_models = [m for m in models if m.id not in completed_ids]

    console.print(
        f"  [dim]•[/dim] [bold yellow]{len(remaining_models)}[/bold yellow] "
        "[dim]remaining to generate[/dim]"
    )
    console.print()

    if not remaining_models:
        console.print(
            Panel(
                f"All models already have results.\n\nTotal results: {len(results)}",
                title="[success]Demo Complete![/success]",
                border_style="success",
            )
        )
        generate_html()
        console.print(f"\n[info]HTML output generated: {DEMO_HTML_PATH}[/info]")
        return

    # Show the demo prompt above the loading bar
    demo_prompt = "Draw a skeleton in ASCII art"
    show_prompt(demo_prompt)

    # Track completion stats
    completed_count = len(completed_ids)
    failed_count = 0
    running_cost = 0.0

    # Create loader for the generation process
    # Each model is one step in the loader
    loader = create_loader(
        remaining_models[0].name if remaining_models else "Loading",
        total=len(remaining_models),
    )

    try:
        with loader:
            # Generate for remaining models
            for i, model in enumerate(remaining_models, start=1):
                # Update loader to show current model name
                loader.set_model(model.name)
                # Set progress to current step (1 to total) so the bar shows activity
                # Each model is one full step in the total progress
                loader.update(i)  # Show progress for current model being processed

                result = generate_demo_sample(model.id, model.name)

                results.append(result)
                save_demo_results(results)

                if result.is_valid:
                    completed_count += 1
                    cost = result.cost if result.cost is not None else 0.0
                    running_cost += cost
                    # Show success flash and continue to next model
                    loader.complete(success=True, cost=cost)
                else:
                    failed_count += 1
                    # Show failure flash and continue to next model
                    loader.complete(success=False, cost=0.0)

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        console.print("\n[warning]Generation interrupted by user.[/warning]")

    # Final summary panel
    console.print()

    summary_text = Text.assemble(
        ("Total results: ", "info"),
        (f"{len(results)}", "accent bold"),
        ("\nCompleted: ", "success"),
        (f"{completed_count}", "success bold"),
        ("\nFailed: ", "error"),
        (f"{failed_count}", "error bold"),
    )

    console.print(
        Panel(
            summary_text,
            title="[success]Demo Complete![/success]",
            border_style="success",
        )
    )

    if failed_count > 0:
        console.print(f"[info]See {ERRORS_LOG_PATH} for detailed error information[/info]")

    generate_html()
    console.print(f"\n[info]HTML output generated: {DEMO_HTML_PATH}[/info]")


if __name__ == "__main__":
    main()
