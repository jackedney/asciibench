# Contributing to ASCIIBench

Thank you for your interest in contributing to ASCIIBench! This document provides guidelines for contributing to the project.

## Setting Up Your Development Environment

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/asciibench.git
   cd asciibench
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

4. Add your OpenRouter API key to `.env`:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Development Workflow

### Running the Development Server

```bash
uv run dev
```

This starts the FastAPI server with hot-reload at `http://localhost:8000`

### Running Tests

Run the test suite:
```bash
uv run pytest
```

Run tests with coverage:
```bash
uv run pytest --cov
```

### Pre-Commit Hooks

Pre-commit hooks automatically check code quality before each commit. Install them once:

```bash
uv run pre-commit install
```

Hooks will run automatically on staged files when you commit. They check:
- Ruff linting with auto-fix
- Ruff formatting
- Type checking with ty

If you need to skip hooks for emergencies:
```bash
git commit --no-verify -m "message"
```

### Code Quality

Before committing, ensure all quality gates pass:

```bash
uv run ruff check          # Linting
uv run ruff format --check # Formatting
uv run ty check            # Type checking
```

Auto-fix linting and formatting issues:
```bash
uv run ruff check --fix
uv run ruff format
```

## Code Style

- Use Python 3.12+ type hints
- Follow PEP 8 conventions (enforced by ruff)
- Maximum line length: 100 characters
- Use double quotes for strings
- Use pydantic v2 for all data models
- Follow DRY principles - use shared utilities in the `common/` package

## Module Guidelines

### Generator Module
- Implement batch processing logic in `generator/sampler.py`
- API calls go in `generator/client.py`
- ASCII sanitization in `generator/sanitizer.py`
- All generated samples must be valid pydantic `ArtSample` models

### Judge UI Module
- Routes defined in `judge_ui/main.py`
- Templates in `templates/` directory
- Use HTMX for dynamic updates
- Keyboard shortcuts: A (left), D (right), S (tie), F (fail)

### Analyst Module
- Elo calculations in `analyst/elo.py`
- Leaderboard generation in `analyst/leaderboard.py`
- Consistency metrics in `analyst/stats.py`
- Read from `data/votes.jsonl` for input

## Commit Messages

Follow conventional commit format:
- `feat: add feature description`
- `fix: bug fix description`
- `docs: documentation update`
- `refactor: code refactoring`
- `test: add/update tests`
- `chore: maintenance tasks`

## Pull Request Process

1. Create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure:
   - All tests pass
   - All quality gates pass
   - New features include tests
   - Documentation is updated if needed

3. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add your feature"
   ```

4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Create a pull request with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots for UI changes (if applicable)

## Questions?

- Open an issue for bugs or feature requests
- Check [SPECIFICATION.md](SPECIFICATION.md) for detailed requirements
- Review existing code for patterns and conventions
