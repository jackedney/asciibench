# ASCIIBench Code Quality Analysis Report

**Date:** 2026-02-10
**Scope:** Full codebase analysis (12,062 LOC across 52 Python files)

## Overview

The codebase is **well-structured overall** - it uses proper service-oriented architecture, dependency injection, comprehensive tests (1.93x test-to-code ratio), and Pydantic models. However, there are several areas where code quality can be meaningfully improved.

---

## HIGH PRIORITY

### 1. Massive inline HTML in `generator/demo.py:generate_html()` (lines 302-776)

**Issue:** 475 lines of inline HTML/CSS/JS built via string concatenation inside a Python function. This is the single largest readability problem in the codebase.

**Impact:** Impossible to maintain CSS/HTML separately, no IDE support for the embedded languages, no template caching.

**Suggestion:** Extract the HTML to a Jinja2 template file (similar to how `judge_ui` uses templates). Move CSS to a static file or reuse the existing Tailwind setup.

---

### 2. Repetitive exception handling in `generator/sampler.py:_generate_single_sample()` (lines 194-273)

**Issue:** 6 nearly identical `except` blocks that differ only in the log message string and error prefix:

```python
except RateLimitError as e:
    sample = _create_error_sample(task)
    logger.error("Rate limited after retries", {...})
    error_message = f"RateLimitError: {e}"
    if span is not None: span.record_exception(e)
except AuthenticationError as e:
    sample = _create_error_sample(task)
    logger.error("Authentication failed", {...})
    error_message = f"AuthenticationError: {e}"
    if span is not None: span.record_exception(e)
# ...4 more identical blocks
```

**Suggestion:** Collapse into a single `except (RateLimitError, AuthenticationError, ...) as e` block, or use a mapping of exception type to log message.

---

### 3. Module-level mutable state in `judge_ui/main.py` (lines 74-154)

**Issue:** Service instances are created at module import time as globals, and `tournament_service` uses the `global` keyword in the lifespan handler. This makes testing harder and creates implicit coupling.

```python
_vlm_evaluation_service: object = None  # global mutable state
_vlm_init_attempted = False              # global mutable state
settings = Settings()                    # created at import time
config_service = ConfigService()         # created at import time
repo = DataRepository(...)               # created at import time
tournament_service: TournamentService | None = None  # mutated via global
```

**Suggestion:** Use FastAPI's dependency injection (`Depends()`) or an application state object (`app.state`) instead of module-level globals. This is the standard FastAPI pattern and would improve testability.

---

### 4. Repetitive config loading in `config_service.py` (lines 221-517)

**Issue:** Five methods (`get_models`, `get_prompts`, `get_evaluator_config`, `get_app_config`, `get_tournament_config`) share an identical structure:

```python
def get_X(self, path="X.yaml"):
    if not self._cache.X_loaded or self._cache.X_loaded_path != path:
        try:
            with Path(path).open() as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                data = {}
            # parse data...
            self._cache.X_loaded = True
            self._cache.X_loaded_path = path
        except FileNotFoundError as e:
            raise ConfigServiceError(f"X.yaml not found: {path}") from e
        except ValidationError as e:
            raise ConfigServiceError(f"Invalid X.yaml structure: {e}") from e
    return self._cache.X
```

The `ConfigCache` dataclass also has 10 redundant `*_loaded`/`*_loaded_path` fields.

**Suggestion:** Extract a generic `_load_yaml_config(path, key, parser_fn)` method. Replace the boolean/path tracking fields with a single `dict[str, Any]` cache keyed by `(config_type, path)`.

---

### 5. Duplicated sync/async retry logic in `common/retry.py`

**Issue:** The retry logic is implemented **four times** with nearly identical code:
- `retry` decorator `wrapper` (sync, lines 162-193)
- `retry` decorator `async_wrapper` (async, lines 196-227)
- `RetryableTaskExecutor.execute` (sync, lines 306-392)
- `RetryableTaskExecutor.execute_async` (async, lines 394-455)

Additionally, the input validation in the decorator (lines 96-114) is duplicated in the class constructor (lines 279-299).

**Suggestion:** Extract the shared retry loop logic into a common helper, or have the decorator delegate to `RetryableTaskExecutor` internally. Share validation logic.

---

## MEDIUM PRIORITY

### 6. Duplicated deduplication logic in `judge_ui/main.py:htmx_vlm_eval()` (lines 722-734)

**Issue:** Two identical loops for deduplicating `results_a` and `results_b`:

```python
seen_a: set[str] = set()
results_a = []
for e in [*[e for e in existing_evaluations if e.sample_id == sample_a_id], *new_results_a]:
    if e.vlm_model_id not in seen_a:
        seen_a.add(e.vlm_model_id)
        results_a.append(e)
# ...exact same pattern for results_b
```

**Suggestion:** Extract to a helper function like `_deduplicate_evaluations(existing, new, sample_id)`.

---

### 7. Repetitive HTMX error handling in `judge_ui/main.py`

**Issue:** Nearly every HTMX endpoint follows the same pattern:

```python
try:
    # ...business logic...
except Exception as e:
    return templates.TemplateResponse(request, "partials/X.html", {"error": str(e)})
```

This is repeated across `htmx_get_matchup`, `htmx_get_prompt`, `htmx_submit_vote`, `htmx_undo_vote`, `htmx_vlm_eval`, `htmx_get_analytics`, `htmx_get_vlm_accuracy`, `htmx_get_elo_vlm_correlation`.

**Suggestion:** Create a decorator or middleware that catches exceptions and renders the appropriate error template, reducing boilerplate in each endpoint.

---

### 8. Manual span management in `sampler.py` (lines 154-283)

**Issue:** Logfire spans are managed manually with `span.__enter__()` / `span.__exit__()` instead of using Python's context manager protocol:

```python
span = logfire.span(...)
span.__enter__()
try:
    ...
finally:
    span.__exit__(None, None, None)
```

**Suggestion:** Use `with logfire.span(...) as span:` which is cleaner and handles exceptions properly. The current pattern doesn't pass exception info to `__exit__`.

---

### 9. Long `get_elo_vlm_correlation` endpoint (lines 775-855)

**Issue:** 80 lines of inline data loading, joining, and correlation computation directly in an endpoint handler. This includes 3 separate try/except blocks for loading different data sources.

**Suggestion:** Move the correlation computation logic into `AnalyticsService`, keeping the endpoint thin.

---

## LOW PRIORITY

### 10. Overly defensive `_read_jsonl` fallback in `repository.py` (lines 30-104)

**Issue:** The module uses a try/except on import with a fallback local implementation of `_read_jsonl`, plus a `_read_jsonl_fn` wrapper and a `_get_read_jsonl_fn` factory. This adds indirection for an edge case that shouldn't occur in normal operation.

---

### 11. `ConfigCache` dataclass bloat in `config_service.py` (lines 128-164)

**Issue:** 15 fields where 5 would suffice with a better data structure. Each config type requires 3 fields: `value`, `loaded` boolean, and `loaded_path`.

---

### 12. Broad exception handling patterns

**Issue:** A few places use overly broad exception handling:
- `judge_ui/main.py:935` - `except (FileNotFoundError, Exception)` which catches everything
- Several `except Exception` blocks that silently swallow errors

---

## Summary Table

| # | File | Issue | Priority | Type |
|---|------|-------|----------|------|
| 1 | `generator/demo.py` | 475-line inline HTML/CSS/JS | HIGH | Structure |
| 2 | `generator/sampler.py` | 6 identical except blocks | HIGH | Duplication |
| 3 | `judge_ui/main.py` | Module-level mutable globals | HIGH | Architecture |
| 4 | `common/config_service.py` | 5 identical load-cache methods | HIGH | Duplication |
| 5 | `common/retry.py` | 4x duplicated retry loops | HIGH | Duplication |
| 6 | `judge_ui/main.py` | Duplicated dedup loops | MEDIUM | Duplication |
| 7 | `judge_ui/main.py` | Repetitive HTMX error handling | MEDIUM | Duplication |
| 8 | `generator/sampler.py` | Manual span `__enter__`/`__exit__` | MEDIUM | Anti-pattern |
| 9 | `judge_ui/main.py` | Long correlation endpoint | MEDIUM | Complexity |
| 10 | `common/repository.py` | Overly defensive import fallback | LOW | Over-engineering |
| 11 | `common/config_service.py` | ConfigCache field bloat | LOW | Structure |
| 12 | Various | Broad exception handling | LOW | Error handling |

---

## What's Already Good

- Clean service-oriented architecture with dependency injection (evaluator, analytics, tournament)
- Proper use of Pydantic models for data validation
- Comprehensive test suite with 1.93x test-to-code ratio
- Consistent use of `dataclass`, `typing`, and `pathlib`
- Well-documented modules with docstrings
- Idempotent resume patterns throughout (generator, evaluator)
- Thread-safe `SharedState` for concurrent generation

---

## Recommended Action Order

The highest-impact improvements would be addressing items **1-5** - they account for the bulk of duplicated/complex code and would significantly reduce maintenance burden.

1. **Start with #2 and #5** (duplication) - lowest risk, highest code reduction
2. **Then #4** (config service) - self-contained refactor
3. **Then #1** (demo HTML) - improves developer experience significantly
4. **Then #3** (FastAPI globals) - architectural improvement, requires test updates
5. **Then #6-9** (medium priority) - incremental improvements
