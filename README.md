# ASCIIBench

![CI](https://github.com/YOUR_USERNAME/asciibench/workflows/CI/badge.svg)

Elo-based LLM benchmark system with Generator, Judge UI, and Analyst modules.

## Installation

```bash
uv sync
```

## Development

Run the dev server:
```bash
uv run dev
```

## Quality Gates

Run all quality checks:
```bash
uv run pytest --cov
uv run ruff check
uv run ruff format --check
uv run ty check
```

## Project Structure

```
asciibench/
├── generator/    # Batch generation module
├── judge_ui/     # FastAPI + HTMX comparison interface
├── analyst/      # Elo rating and leaderboard
└── common/       # Shared models and configuration
```

See [SPECIFICATION.md](SPECIFICATION.md) for detailed requirements.
