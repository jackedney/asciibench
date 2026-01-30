"""Demo module for ASCII art generation.

This module provides a demo mode for generating ASCII art samples
from language models and displaying results in HTML format.

Demo generates ASCII art using a fixed prompt and saves outputs
to .demo_outputs/demo.html with incremental/resumable generation support.
"""

import html
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

from asciibench.common.config import Settings
from asciibench.common.models import DemoResult
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


def setup_error_logger() -> logging.Logger:
    """Set up a dedicated logger for generation errors.

    Creates .demo_outputs directory if needed and configures
    file handler for errors.log with detailed formatting.

    Returns:
        Configured Logger instance for error logging
    """
    DEMO_OUTPUTS_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger("asciibench.demo.errors")
    logger.setLevel(logging.ERROR)

    # Avoid adding duplicate handlers on repeated calls
    if not logger.handlers:
        file_handler = logging.FileHandler(ERRORS_LOG_PATH, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_generation_error(
    logger: logging.Logger,
    model_id: str,
    model_name: str,
    error_reason: str,
    raw_output: str | None = None,
    exception: Exception | None = None,
) -> None:
    """Log a generation error with full context.

    Args:
        logger: The error logger instance
        model_id: Model identifier (e.g., 'openai/gpt-4o-mini')
        model_name: Human-readable model name
        error_reason: Brief description of why generation failed
        raw_output: The raw output received from the model (if any)
        exception: The exception that was raised (if any)
    """
    separator = "=" * 80
    log_lines = [
        "",
        separator,
        f"MODEL: {model_name} ({model_id})",
        f"REASON: {error_reason}",
    ]

    if exception:
        log_lines.append(f"EXCEPTION: {type(exception).__name__}: {exception}")
        log_lines.append(f"TRACEBACK:\n{traceback.format_exc()}")

    if raw_output:
        # Truncate very long outputs but keep enough for debugging
        truncated = raw_output[:2000] + "..." if len(raw_output) > 2000 else raw_output
        log_lines.append(f"RAW OUTPUT ({len(raw_output)} chars):\n{truncated}")

    log_lines.append(separator)

    logger.error("\n".join(log_lines))


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
        print(f"Warning: Corrupted results.json: {e}")
        print("Starting with empty results.")
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
    logger: logging.Logger | None = None,
) -> DemoResult:
    """Generate a demo ASCII art sample from a single model.

    Uses hardcoded prompt 'Draw a skeleton in ASCII art' and loads
    generation configuration from config.yaml.

    Args:
        model_id: Model identifier (e.g., 'openai/gpt-4o-mini')
        model_name: Human-readable model name (e.g., 'GPT-4o Mini')
        logger: Optional error logger for detailed failure tracking

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

    raw_output: str | None = None
    error_reason: str | None = None

    try:
        raw_output = client.generate(model_id, demo_prompt, config=config)
        ascii_output = extract_ascii_from_markdown(raw_output)
        is_valid = bool(ascii_output)

        if not is_valid:
            error_reason = "No valid ASCII art block found in output"
            if logger:
                log_generation_error(
                    logger,
                    model_id,
                    model_name,
                    error_reason,
                    raw_output=raw_output,
                )

    except (AuthenticationError, ModelError, OpenRouterClientError) as e:
        error_reason = f"API error: {type(e).__name__}"
        ascii_output = f"Error: {e!s}"
        is_valid = False
        if logger:
            log_generation_error(
                logger,
                model_id,
                model_name,
                error_reason,
                raw_output=raw_output,
                exception=e,
            )

    except Exception as e:
        error_reason = f"Unexpected error: {type(e).__name__}"
        ascii_output = f"Unexpected error: {e!s}"
        is_valid = False
        if logger:
            log_generation_error(
                logger,
                model_id,
                model_name,
                error_reason,
                raw_output=raw_output,
                exception=e,
            )

    return DemoResult(
        model_id=model_id,
        model_name=model_name,
        ascii_output=ascii_output,
        is_valid=is_valid,
        timestamp=datetime.now(),
        error_reason=error_reason,
        raw_output=raw_output if not is_valid else None,
    )


def generate_html() -> None:
    """Generate HTML output from results.json.

    Creates .demo_outputs/demo.html with all model outputs displayed
    in a scrollable page with clean inline CSS styling.

    Each model section includes:
    - h2 header with model name
    - pre block with ASCII art (monospace font)
    - timestamp showing generation time
    - red border styling for invalid outputs

    Empty results show 'No results yet' message.
    """
    results = load_demo_results()

    DEMO_OUTPUTS_DIR.mkdir(exist_ok=True)

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASCIIBench Demo - Skeleton ASCII Art</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            background-color: #f9fafb;
        }
        header {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px 0;
            border-bottom: 2px solid #e5e7eb;
        }
        h1 {
            color: #111827;
            margin: 0;
        }
        .no-results {
            text-align: center;
            padding: 60px 20px;
            color: #6b7280;
            font-size: 1.2em;
            background-color: #fff;
            border-radius: 8px;
            margin: 40px 0;
        }
        .model-section {
            background-color: #fff;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        .model-section.invalid {
            border: 2px solid #ef4444;
        }
        h2 {
            color: #1f2937;
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 1.5em;
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 10px;
        }
        .model-id {
            color: #6b7280;
            font-size: 0.9em;
            margin-left: 10px;
            font-family: 'Courier New', monospace;
        }
        pre {
            font-family: 'Courier New', Courier, monospace;
            background-color: #f3f4f6;
            padding: 20px;
            border-radius: 6px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 14px;
            line-height: 1.5;
            border: 1px solid #e5e7eb;
        }
        .invalid pre {
            background-color: #fef2f2;
            border-color: #fca5a5;
        }
        .timestamp {
            color: #9ca3af;
            font-size: 0.85em;
            margin-top: 15px;
            text-align: right;
        }
        .valid-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }
        .valid-badge.valid {
            background-color: #d1fae5;
            color: #065f46;
        }
        .valid-badge.invalid {
            background-color: #fee2e2;
            color: #991b1b;
        }
        .error-message {
            color: #dc2626;
            font-style: italic;
        }
    </style>
</head>
<body>
    <header>
        <h1>ASCIIBench Demo - Skeleton ASCII Art</h1>
    </header>
"""

    if not results:
        html_content += """
    <div class="no-results">
        No results yet. Run 'task demo' to generate ASCII art samples.
    </div>
"""
    else:
        for result in results:
            timestamp_str = result.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            section_class = "invalid" if not result.is_valid else ""
            badge_class = "valid" if result.is_valid else "invalid"
            badge_text = "Valid" if result.is_valid else "Invalid"
            escaped_output = html.escape(result.ascii_output)

            html_content += f"""
    <div class="model-section {section_class}">
        <h2>
            {result.model_name}
            <span class="model-id">({result.model_id})</span>
            <span class="valid-badge {badge_class}">{badge_text}</span>
        </h2>
        <pre>{escaped_output}</pre>
        <div class="timestamp">Generated: {timestamp_str}</div>
    </div>
"""

    html_content += """
</body>
</html>
"""

    with DEMO_HTML_PATH.open("w", encoding="utf-8") as f:
        f.write(html_content)


def main() -> None:
    """Main entry point for the Demo module.

    This function runs the demo generator that:
    1. Prints header banner
    2. Sets up error logging to .demo_outputs/errors.log
    3. Loads models from models.yaml
    4. Generates ASCII art for each model using fixed prompt
    5. Saves results to .demo_outputs/results.json
    6. Generates HTML output to .demo_outputs/demo.html
    """
    print("ASCIIBench Demo")
    print("=" * 50)

    # Set up error logging
    error_logger = setup_error_logger()

    models = load_models()

    if not models:
        print("\nWarning: No models found in models.yaml")
        print("Exiting gracefully.")
        return

    print(f"\nLoaded {len(models)} models from models.yaml")

    results = load_demo_results()
    print(f"Loaded {len(results)} existing results")

    completed_ids = get_completed_model_ids()
    print(f"Completed models: {len(completed_ids)}")

    if completed_ids:
        print("  Models already done:")
        for model_id in sorted(completed_ids):
            print(f"    - {model_id}")

    remaining_models = [m for m in models if m.id not in completed_ids]
    print(f"\nRemaining models to generate: {len(remaining_models)}")

    if not remaining_models:
        print("\nAll models already have results. Nothing to do.")
        print("\n" + "=" * 50)
        print("Demo Complete!")
        print(f"Total results: {len(results)}")
        generate_html()
        print(f"\nHTML output generated: {DEMO_HTML_PATH}")
        return

    print("\nStarting generation...")
    print(f"Errors will be logged to: {ERRORS_LOG_PATH}")
    print("-" * 50)

    failed_count = 0
    for i, model in enumerate(remaining_models, start=1):
        print(f"[{i}/{len(remaining_models)}] Generating for {model.name} ({model.id})...")

        result = generate_demo_sample(model.id, model.name, logger=error_logger)

        results.append(result)
        save_demo_results(results)

        if result.is_valid:
            print(f"  ✓ Generated successfully ({len(result.ascii_output)} characters)")
        else:
            failed_count += 1
            error_msg = result.error_reason or "Unknown error"
            print(f"  ✗ Failed: {error_msg}")

    print("\n" + "=" * 50)
    print("Demo Complete!")
    print(f"Total results: {len(results)}")

    if failed_count > 0:
        print(f"Failed generations: {failed_count}")
        print(f"See {ERRORS_LOG_PATH} for detailed error information")

    generate_html()
    print(f"\nHTML output generated: {DEMO_HTML_PATH}")


if __name__ == "__main__":
    main()
