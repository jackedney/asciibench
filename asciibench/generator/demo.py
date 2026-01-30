"""Demo module for ASCII art generation.

This module provides a demo mode for generating ASCII art samples
from language models and displaying results in HTML format.

Demo generates ASCII art using a fixed prompt and saves outputs
to .demo_outputs/demo.html with incremental/resumable generation support.
"""

import json
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


def generate_demo_sample(model_id: str, model_name: str) -> DemoResult:
    """Generate a demo ASCII art sample from a single model.

    Uses hardcoded prompt 'Draw a skeleton in ASCII art' and loads
    generation configuration from config.yaml.

    Args:
        model_id: Model identifier (e.g., 'openai/gpt-4o-mini')
        model_name: Human-readable model name (e.g., 'GPT-4o Mini')

    Returns:
        DemoResult with model info, ascii_output, is_valid, and timestamp

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

    try:
        raw_output = client.generate(model_id, demo_prompt, config=config)
        ascii_output = extract_ascii_from_markdown(raw_output)
        is_valid = bool(ascii_output)
    except (AuthenticationError, ModelError, OpenRouterClientError) as e:
        ascii_output = f"Error: {e!s}"
        is_valid = False
    except Exception as e:
        ascii_output = f"Unexpected error: {e!s}"
        is_valid = False

    return DemoResult(
        model_id=model_id,
        model_name=model_name,
        ascii_output=ascii_output,
        is_valid=is_valid,
        timestamp=datetime.now(),
    )


def main() -> None:
    """Main entry point for the Demo module.

    This function runs the demo generator that:
    1. Prints header banner
    2. Loads models from models.yaml
    3. Generates ASCII art for each model using fixed prompt
    4. Saves results to .demo_outputs/results.json
    5. Generates HTML output to .demo_outputs/demo.html
    """
    print("ASCIIBench Demo")
    print("=" * 50)

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
        return

    print("\nStarting generation...")
    print("-" * 50)

    for i, model in enumerate(remaining_models, start=1):
        print(f"[{i}/{len(remaining_models)}] Generating for {model.name} ({model.id})...")

        result = generate_demo_sample(model.id, model.name)

        results.append(result)
        save_demo_results(results)

        if result.is_valid:
            print(f"  ✓ Generated successfully ({len(result.ascii_output)} characters)")
        else:
            print(f"  ✗ Failed: {result.ascii_output[:100]}")

    print("\n" + "=" * 50)
    print("Demo Complete!")
    print(f"Total results: {len(results)}")


if __name__ == "__main__":
    main()
