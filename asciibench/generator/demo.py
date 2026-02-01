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

from rich.panel import Panel
from rich.text import Text

from asciibench.common.config import Settings
from asciibench.common.display import get_console
from asciibench.common.logging import get_logger
from asciibench.common.models import DemoResult, OpenRouterResponse
from asciibench.common.simple_display import create_loader, show_banner, show_prompt
from asciibench.common.yaml_config import load_generation_config, load_models
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
        metadata["traceback"] = traceback.format_exc()

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
    settings = Settings()
    config = load_generation_config()
    demo_prompt = "Draw a skeleton in ASCII art"

    client = OpenRouterClient(api_key=settings.openrouter_api_key, base_url=settings.base_url)

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
    """
    results = load_demo_results()

    DEMO_OUTPUTS_DIR.mkdir(exist_ok=True)

    # Calculate stats for summary bar
    total_count = len(results)
    valid_count = sum(1 for r in results if r.is_valid)
    invalid_count = total_count - valid_count
    total_cost = sum(r.cost or 0 for r in results)
    valid_pct = (valid_count / total_count * 100) if total_count > 0 else 0

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASCIIBench Demo - Skeleton ASCII Art</title>
    <style>
        :root {
            --bg-primary: #f8fafc;
            --bg-card: #ffffff;
            --bg-code: #1e293b;
            --text-primary: #0f172a;
            --text-secondary: #64748b;
            --text-muted: #94a3b8;
            --border-color: #e2e8f0;
            --accent-green: #10b981;
            --accent-green-bg: #d1fae5;
            --accent-green-text: #065f46;
            --accent-red: #ef4444;
            --accent-red-bg: #fee2e2;
            --accent-red-text: #991b1b;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 16px;
            --transition: all 0.2s ease;
        }

        * {
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                "Helvetica Neue", Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px;
            line-height: 1.6;
            background-color: var(--bg-primary);
            color: var(--text-primary);
        }

        header {
            text-align: center;
            margin-bottom: 32px;
            padding: 40px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-lg);
        }

        h1 {
            color: #fff;
            margin: 0 0 8px 0;
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }

        .subtitle {
            color: rgba(255, 255, 255, 0.85);
            font-size: 1rem;
            margin: 0;
        }

        .stats-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            justify-content: center;
            margin-bottom: 24px;
            padding: 20px;
            background: var(--bg-card);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-color);
        }

        .stat {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--bg-primary);
            border-radius: var(--radius-sm);
            font-size: 0.9rem;
            font-weight: 500;
        }

        .stat-icon {
            font-size: 1.1rem;
        }

        .stat-value {
            font-weight: 700;
            font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
        }

        .stat.valid .stat-value { color: var(--accent-green); }
        .stat.invalid .stat-value { color: var(--accent-red); }
        .stat.cost .stat-value { color: var(--accent-purple); }

        .progress-container {
            width: 100%;
            margin-top: 12px;
            padding-top: 16px;
            border-top: 1px solid var(--border-color);
        }

        .progress-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }

        .progress-bar {
            height: 8px;
            background: var(--accent-red-bg);
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-green) 0%, #34d399 100%);
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        .filter-controls {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-bottom: 24px;
        }

        .filter-btn {
            padding: 10px 20px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            border: 2px solid var(--border-color);
            border-radius: var(--radius-sm);
            background: var(--bg-card);
            color: var(--text-secondary);
            transition: var(--transition);
        }

        .filter-btn:hover {
            border-color: var(--accent-blue);
            color: var(--accent-blue);
        }

        .filter-btn.active {
            background: var(--accent-blue);
            border-color: var(--accent-blue);
            color: #fff;
        }

        .models-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .no-results {
            text-align: center;
            padding: 60px 24px;
            color: var(--text-secondary);
            font-size: 1.1rem;
            background: var(--bg-card);
            border-radius: var(--radius-md);
            border: 2px dashed var(--border-color);
        }

        .model-section {
            background: var(--bg-card);
            border-radius: var(--radius-md);
            padding: 24px;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-color);
            transition: var(--transition);
        }

        .model-section:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }

        .model-section.invalid {
            border-left: 4px solid var(--accent-red);
        }

        .model-section.valid-model {
            border-left: 4px solid var(--accent-green);
        }

        .model-section.hidden {
            display: none;
        }

        .model-header {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-color);
        }

        .model-name {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: 0;
        }

        .model-id {
            color: var(--text-muted);
            font-size: 0.85rem;
            font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
        }

        .model-meta {
            display: flex;
            gap: 16px;
            margin-left: auto;
            align-items: center;
        }

        .meta-item {
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
        }

        .valid-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .valid-badge.valid {
            background: var(--accent-green-bg);
            color: var(--accent-green-text);
        }

        .valid-badge.invalid {
            background: var(--accent-red-bg);
            color: var(--accent-red-text);
        }

        pre {
            font-family: 'SF Mono', 'Consolas', 'Monaco', 'Courier New', monospace;
            background: var(--bg-code);
            color: #e2e8f0;
            padding: 20px;
            border-radius: var(--radius-sm);
            overflow-x: auto;
            white-space: pre;
            font-size: 13px;
            line-height: 1.4;
            margin: 0;
        }

        .invalid pre {
            background: #7f1d1d;
            color: #fecaca;
        }

        .timestamp {
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-top: 12px;
            text-align: right;
        }

        @media (max-width: 768px) {
            body {
                padding: 16px;
            }

            header {
                padding: 24px 16px;
            }

            h1 {
                font-size: 1.5rem;
            }

            .stats-bar {
                flex-direction: column;
                align-items: stretch;
            }

            .stat {
                justify-content: space-between;
            }

            .model-header {
                flex-direction: column;
                align-items: flex-start;
            }

            .model-meta {
                margin-left: 0;
                flex-wrap: wrap;
            }

            .filter-controls {
                flex-wrap: wrap;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>ASCIIBench Demo</h1>
        <p class="subtitle">Skeleton ASCII Art Generation</p>
    </header>
"""

    if results:
        html_content += f"""
    <div class="stats-bar">
        <div class="stat">
            <span class="stat-icon">📊</span>
            <span>Total:</span>
            <span class="stat-value">{total_count}</span>
        </div>
        <div class="stat valid">
            <span class="stat-icon">✓</span>
            <span>Valid:</span>
            <span class="stat-value">{valid_count}</span>
        </div>
        <div class="stat invalid">
            <span class="stat-icon">✗</span>
            <span>Invalid:</span>
            <span class="stat-value">{invalid_count}</span>
        </div>
        <div class="stat cost">
            <span class="stat-icon">💰</span>
            <span>Total Cost:</span>
            <span class="stat-value">${total_cost:.4f}</span>
        </div>
        <div class="progress-container">
            <div class="progress-label">
                <span>Success Rate</span>
                <span>{valid_pct:.1f}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {valid_pct}%"></div>
            </div>
        </div>
    </div>

    <div class="filter-controls">
        <button class="filter-btn active" onclick="filterModels('all')">All ({total_count})</button>
        <button class="filter-btn" onclick="filterModels('valid')">Valid ({valid_count})</button>
        <button class="filter-btn" onclick="filterModels('invalid')">Invalid ({invalid_count})</button>
    </div>

    <div class="models-container">
"""

    if not results:
        html_content += """
    <div class="no-results">
        No results yet. Run <code>task demo</code> to generate ASCII art samples.
    </div>
"""
    else:
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

            html_content += f"""
        <div class="model-section {valid_class}" data-valid="{data_valid}">
            <div class="model-header">
                <h2 class="model-name">{result.model_name}</h2>
                <span class="model-id">{result.model_id}</span>
                <div class="model-meta">
                    <span class="meta-item">💰 {cost_str}</span>
                    <span class="meta-item">🔤 {tokens_str} tokens</span>
                    <span class="valid-badge {badge_class}">{badge_icon} {badge_text}</span>
                </div>
            </div>
            <pre>{escaped_output}</pre>
            <div class="timestamp">Generated: {timestamp_str}</div>
        </div>
"""

    if results:
        html_content += """
    </div>
"""

    html_content += """
    <script>
        function filterModels(status) {
            const sections = document.querySelectorAll('.model-section');
            const buttons = document.querySelectorAll('.filter-btn');

            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            sections.forEach(section => {
                const isValid = section.dataset.valid === 'true';
                if (status === 'all') {
                    section.classList.remove('hidden');
                } else if (status === 'valid') {
                    section.classList.toggle('hidden', !isValid);
                } else if (status === 'invalid') {
                    section.classList.toggle('hidden', isValid);
                }
            });
        }
    </script>
</body>
</html>
"""

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

    # Load models
    try:
        models = load_models()
    except FileNotFoundError:
        console.print(
            Panel(
                "[error]models.yaml not found[/error]",
                title="[error]Error[/error]",
                border_style="error",
            )
        )
        return
    except Exception as e:
        console.print(
            Panel(
                f"[error]Error loading models.yaml: {e}[/error]",
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
        f"  [dim]•[/dim] [bold cyan]{len(models)}[/bold cyan] [dim]models loaded from[/dim] [white]models.yaml[/white]"
    )

    results = load_demo_results()
    completed_ids = get_completed_model_ids()
    remaining_models = [m for m in models if m.id not in completed_ids]

    console.print(
        f"  [dim]•[/dim] [bold yellow]{len(remaining_models)}[/bold yellow] [dim]remaining to generate[/dim]"
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
