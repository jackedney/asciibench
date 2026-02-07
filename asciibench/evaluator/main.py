"""Evaluator module main entry point.

This module provides the main entry point for the Evaluator module,
which renders ASCII art to images and evaluates them with Vision Language Models (VLMs).

Dependencies:
    - asciibench.common.yaml_config: Evaluator configuration loading
    - asciibench.common.display: Console output and formatting
"""

import sys

from asciibench.common.display import get_console
from asciibench.common.yaml_config import load_evaluator_config


def main() -> None:
    """Main entry point for the Evaluator module.

    This function coordinates the VLM evaluation of ASCII art samples:
    1. Load evaluator configuration from evaluator_config.yaml
    2. Validate configuration
    3. Display startup message

    Future iterations will:
    - Load valid samples from database.jsonl
    - Render ASCII art to images
    - Send images to VLMs for analysis
    - Compute similarity scores
    - Persist evaluation results
    """
    console = get_console()

    console.print("[info]VLM Evaluator starting...[/info]")

    try:
        config = load_evaluator_config()
        console.print(
            f"[dim]Loaded evaluator config with {len(config.vlm_models)} VLM models[/dim]"
        )
    except Exception as e:
        console.print(f"[error]Error loading evaluator config: {e}[/error]")
        sys.exit(1)


if __name__ == "__main__":
    main()
