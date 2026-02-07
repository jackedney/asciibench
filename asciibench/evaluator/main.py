"""Evaluator module main entry point.

This module provides the main entry point for the Evaluator module,
which renders ASCII art to images and evaluates them with Vision Language Models (VLMs).

Dependencies:
    - asciibench.common.yaml_config: Evaluator configuration loading
    - asciibench.common.display: Console output and formatting
"""

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import typer

from asciibench.common.display import get_console
from asciibench.common.models import ArtSample, VLMEvaluation
from asciibench.common.persistence import read_jsonl
from asciibench.common.yaml_config import load_evaluator_config
from asciibench.evaluator.orchestrator import run_evaluation

app = typer.Typer()

DEFAULT_DATABASE_PATH = Path("data/database.jsonl")
DEFAULT_EVALUATIONS_PATH = Path("data/vlm_evaluations.jsonl")


@app.command()
def main(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be evaluated without making API calls"),
    ] = False,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Limit to first N samples (for testing)")
    ] = None,
    database_path: Annotated[
        Path, typer.Option("--database-path", help="Path to database.jsonl file")
    ] = DEFAULT_DATABASE_PATH,
    evaluations_path: Annotated[
        Path, typer.Option("--evaluations-path", help="Path to vlm_evaluations.jsonl file")
    ] = DEFAULT_EVALUATIONS_PATH,
) -> None:
    """Run VLM evaluator on ASCII art samples.

    Evaluates valid ASCII art samples by rendering them to images and
    analyzing them with Vision Language Models to determine if the
    intended subject is correctly identified.
    """
    console = get_console()

    console.print("[info]VLM Evaluator starting...[/info]")

    try:
        config = load_evaluator_config()
        console.print(
            f"[dim]Loaded evaluator config with {len(config.vlm_models)} VLM model(s)[/dim]"
        )
    except Exception as e:
        console.print(f"[error]Error loading evaluator config: {e}[/error]")
        sys.exit(1)

    if not database_path.exists():
        console.print(f"[error]Database file not found: {database_path}[/error]")
        sys.exit(1)

    logger = console
    logger.print("[dim]Loading valid samples from database...[/dim]")
    all_samples = read_jsonl(database_path, ArtSample)
    valid_samples = [s for s in all_samples if s.is_valid]

    if not valid_samples:
        console.print(
            "[warning]No valid samples found in database. Run 'task generate' first.[/warning]"
        )
        sys.exit(0)

    console.print(f"[dim]Found {len(valid_samples)} valid samples[/dim]")

    existing_evaluations = []
    if evaluations_path.exists():
        existing_evaluations = read_jsonl(evaluations_path, VLMEvaluation)

    existing_eval_keys = {(str(ev.sample_id), ev.vlm_model_id) for ev in existing_evaluations}

    tasks_to_process = []
    for sample in valid_samples:
        for vlm_model_id in config.vlm_models:
            key = (str(sample.id), vlm_model_id)
            if key not in existing_eval_keys:
                tasks_to_process.append((sample, vlm_model_id))

    if limit is not None and limit > 0:
        original_count = len(tasks_to_process)
        tasks_to_process = tasks_to_process[:limit]
        console.print(
            f"[dim]Limited to first {len(tasks_to_process)} of {original_count} tasks[/dim]"
        )

    if not tasks_to_process:
        console.print("[success]All samples already evaluated by configured VLM models[/success]")
        sys.exit(0)

    if dry_run:
        vlm_count = len(config.vlm_models)
        console.print(
            f"[info]Would evaluate {len(tasks_to_process)} samples "
            f"with {vlm_count} VLM model(s)[/info]"
        )
        for sample, vlm_model_id in tasks_to_process[:5]:
            prompt_preview = f"'{sample.prompt_text[:50]}...'"
            console.print(f"  [dim]- {sample.model_id}: {prompt_preview} -> {vlm_model_id}[/dim]")
        if len(tasks_to_process) > 5:
            console.print(f"  [dim]... and {len(tasks_to_process) - 5} more[/dim]")
        sys.exit(0)

    console.print(f"[info]Evaluating {len(tasks_to_process)} samples...[/info]")

    results = asyncio.run(
        run_evaluation(
            database_path=database_path,
            evaluations_path=evaluations_path,
            config=config,
            limit=limit,
        )
    )

    if not results:
        console.print("[warning]No evaluations completed (all may have failed)[/warning]")
        sys.exit(1)

    total_evaluated = len(results)
    correct_count = sum(1 for r in results if r.is_correct)
    accuracy = (correct_count / total_evaluated) * 100 if total_evaluated > 0 else 0.0
    total_cost = sum(r.cost for r in results if r.cost is not None)

    console.print()
    console.print("[success]Evaluation completed[/success]")
    console.print(
        f"  Evaluated {total_evaluated} samples, {correct_count} correct ({accuracy:.1f}%)"
    )

    if total_cost > 0:
        console.print(f"  Total cost: ${total_cost:.4f}")

    console.print(f"  Results saved to {evaluations_path}")
    console.print()


if __name__ == "__main__":
    app()
