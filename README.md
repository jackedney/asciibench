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

## Observability

ASCIIBench integrates with [Logfire](https://logfire.pydantic.dev) for observability and monitoring of LLM API calls. Logfire provides a dashboard to view traces, metrics, and performance data.

### Setup

Logfire is optional and disabled by default. To enable it:

1. Sign up for a Logfire account at [https://logfire.pydantic.dev](https://logfire.pydantic.dev)
2. Create a new project and copy your API token
3. Add the following to your `.env` file:

```bash
LOGFIRE_TOKEN=your_logfire_token_here
LOGFIRE_SERVICE_NAME=asciibench
LOGFIRE_ENVIRONMENT=development
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LOGFIRE_TOKEN` | Optional | - | Your Logfire API token from the dashboard. Presence enables Logfire monitoring |
| `LOGFIRE_SERVICE_NAME` | Optional | `asciibench` | Name of the service in Logfire |
| `LOGFIRE_ENVIRONMENT` | Optional | `development` | Environment identifier (e.g., `development`, `staging`, `production`) |

### Data Captured

When Logfire is enabled, it captures the following data for each LLM API call:

- **Request details**: model ID, temperature, max_tokens, timeout, reasoning_enabled
- **Full prompt messages**: Complete prompt sent to the LLM (including system prompt if configured)
- **Full response content**: Complete text response from the LLM
- **Token usage**: prompt_tokens, completion_tokens, total_tokens
- **Cost**: Cost of the API call in USD (provided by LiteLLM)
- **Latency**: Response time in milliseconds
- **Error details**: Full exception trace if an API call fails

### Traces

Logfire automatically creates nested spans for batch operations:

- **Batch span** (`batch.generate`): Wraps an entire batch generation job
  - Attributes: total_tasks, max_concurrent_requests, model_ids
- **Sample span** (`sample.generate`): Wraps individual sample generation
  - Attributes: prompt_id, model_id, attempt_number, run_id, request_id
- **LLM span** (`llm.generate`): Wraps the actual LLM API call
  - Attributes: model_id, temperature, max_tokens, timeout, reasoning_enabled
  - Captures all request/response data, tokens, and cost

This nested structure allows you to drill down from batch operations to individual API calls for debugging and analysis.

### Privacy Note

Logfire captures full prompt and response content. Ensure you only enable Logfire with data that complies with your privacy and security requirements.

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
