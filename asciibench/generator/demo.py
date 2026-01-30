"""Demo module for ASCII art generation.

This module provides a demo mode for generating ASCII art samples
from language models and displaying results in HTML format.

Demo generates ASCII art using a fixed prompt and saves outputs
to .demo_outputs/demo.html with incremental/resumable generation support.
"""

import json
from pathlib import Path

from asciibench.common.models import DemoResult

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

    results = load_demo_results()
    print(f"\nLoaded {len(results)} existing results")

    completed_ids = get_completed_model_ids()
    print(f"Completed models: {len(completed_ids)}")

    if completed_ids:
        print("  Models already done:")
        for model_id in sorted(completed_ids):
            print(f"    - {model_id}")

    print("\nDemo mode coming soon!")
    print("This will generate skeleton ASCII art from all configured models.")
    print("Outputs will be saved to .demo_outputs/demo.html")

    print("\n" + "=" * 50)
    print("Demo Complete!")


if __name__ == "__main__":
    main()
