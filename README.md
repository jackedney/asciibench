# ASCIIBench

![CI](https://github.com/YOUR_USERNAME/asciibench/workflows/CI/badge.svg)

ASCIIBench is an Elo-based benchmark system designed to rank Large Language Models (LLMs) based on their spatial reasoning and artistic capability within the medium of ASCII art. The system uses a human-in-the-loop evaluation approach with three core modules:

- **Generator**: Batch processor that generates ASCII art samples from LLMs
- **Judge UI**: Double-blind web interface for 1v1 human comparisons
- **Analyst**: Elo rating calculator and leaderboard generator

## Installation

Prerequisites: Python 3.12+ and [uv](https://github.com/astral-sh/uv) package manager

```bash
# Install dependencies
uv sync

# Copy example environment file and add your API key
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

## Development

Run the dev server:
```bash
uv run dev
```

This starts the FastAPI server with hot-reload at `http://localhost:8000`

## Project Structure

```
asciibench/
├── generator/         # Batch generation module
│   ├── main.py        # Entry point for generation scripts
│   ├── client.py      # OpenRouter API client using smolagents
│   ├── sampler.py     # Sample generation logic
│   └── sanitizer.py   # ASCII extraction from markdown
├── judge_ui/          # FastAPI + HTMX comparison interface
│   ├── main.py        # FastAPI app instance and routes
│   └── templates/     # HTML templates for judge interface
├── analyst/           # Elo rating and leaderboard
│   ├── main.py        # Entry point for analysis scripts
│   ├── elo.py         # Elo rating calculation
│   ├── leaderboard.py # Leaderboard generation
│   └── stats.py       # Consistency metrics
├── common/            # Shared models and configuration
│   ├── models.py      # Pydantic data models
│   ├── config.py      # Pydantic-settings configuration
│   └── yaml_config.py # YAML config loaders
data/                  # JSONL data files
├── database.jsonl     # Generated art samples
└── votes.jsonl        # Judge comparison votes
```

## Modules

### Generator
The Generator module handles batch generation of ASCII art samples from configured LLMs. It:
- Expands prompt templates into unique test prompts
- Calls the OpenRouter API for each model and prompt combination
- Extracts and validates ASCII art from markdown code blocks
- Saves results to `data/database.jsonl`

### Judge UI
The Judge UI provides a double-blind 1v1 comparison interface. It:
- Presents two random ASCII art samples side-by-side
- Hides model identities during comparison
- Renders each sample in three fonts for visual robustness
- Accepts keyboard input for judgments (A/D/S/F)
- Saves votes to `data/votes.jsonl`

### Analyst
The Analyst module calculates rankings from comparison data. It:
- Reads votes from `data/votes.jsonl`
- Applies Bradley-Terry/Elo rating models
- Generates `LEADERBOARD.md` with:
  - Overall Elo ratings
  - Category-specific rankings
  - Consistency scores across attempts

## Quality Gates

Run all quality checks:

```bash
# Tests with coverage
uv run pytest --cov

# Linting
uv run ruff check

# Formatting check
uv run ruff format --check

# Type checking
uv run ty check
```

## Configuration

Configure the system via YAML files:

- `models.yaml`: List of OpenRouter model identifiers
- `prompts.yaml`: Prompt templates organized by category
- `config.yaml`: Generation settings (temperature, max_tokens, etc.)

See [SPECIFICATION.md](SPECIFICATION.md) for detailed requirements and pipeline documentation.
