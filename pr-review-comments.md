# PR #1 Review Comments

## Review 1 (ID: 3757459035)
**Submitted:** 2026-02-05T14:26:07Z
**Reviewer:** coderabbitai[bot]

### Actionable Comments Summary (18 items)

#### asciibench/common/loader.py
- **Line 247-261:** `set_prompt` currently assigns prompt_text directly (after truncation) and can retain embedded newlines which break single-line carriage-return rendering; sanitize the input by normalizing to a single-line string (e.g., replace all '\r' and '\n' with a single space and collapse repeated whitespace) before applying max_len and assigning to `self._current_prompt` inside the `with self._lock` block so set_prompt, self._current_prompt and the truncation logic always work with a safe single-line prompt.

#### asciibench/common/logging.py
- **Line 10:** The project imports FileLock via "from filelock import FileLock" in asciibench/common/logging.py but filelock is not declared as an explicit dependency; update pyproject.toml by adding "filelock" to the dependencies array (ensure version specifier compatible with the project, e.g., a minimum version or caret constraint) so that FileLock is guaranteed to be installed transitively and CI/builds remain reproducible.

#### asciibench/common/retry.py
- **Line 37-72:** The decorator currently assumes sync functions (decorator -> wrapper) so applying it to async functions returns a coroutine unawaited and misses exceptions; update the decorator to detect async functions using `inspect.iscoroutinefunction(func)` and provide an async_wrapper that mirrors the existing retry logic (respecting retryable_exceptions, max_retries, base_delay_seconds and logging via logger) and awaits `func(*args, **kwargs)`, while keeping the existing sync wrapper for regular functions; ensure both branches raise the last exception when max retries are exceeded and preserve `functools.wraps` for the chosen wrapper.
- **Line 11-15:** The retry decorator's signature (function retry) needs input validation to avoid silent no-ops or runtime errors: check max_retries is an int >= 0 and base_delay_seconds is a float/number >= 0 at the start of retry (or in the outer decorator factory) and raise a ValueError with a clear message if not; also optionally validate retryable_exceptions is a non-empty tuple of Exception types and raise TypeError/ValueError for invalid types to fail fast rather than letting the decorated call behave unexpectedly.

#### asciibench/generator/client.py
- **Line 153-157:** The retry decorator currently retries on transient errors but will also retry when run_with_timeout raises a timeout, which can leave the original worker thread running and cause duplicate in-flight requests; update the retry logic so timeouts are non-retryable by ensuring TimeoutError (or the timeout exception type raised by run_with_timeout) is not included in retryable_exceptions on the `@retry` decorator and/or explicitly catch TimeoutError in the wrapped function (the method using run_with_timeout) and do not re-raise it for retry; adjust all occurrences (the `@retry` usages around the functions that call run_with_timeout, referenced by the retry decorator and the run_with_timeout call sites) accordingly so timeouts are not retried.

#### asciibench/generator/demo.py
- **Line 41-75:** The traceback is being captured with `traceback.format_exc()` outside the except context which returns None; update log_generation_error to build the traceback from the preserved exception by using the exception's `__traceback__` (e.g., via `traceback.format_exception` or `format_tb`) and store that string into `metadata["traceback"]`; retain the other metadata keys ("exception_type", "exception_message", "raw_output_preview", etc.) and then call `logger.error("Demo generation failed", metadata)`.

#### asciibench/generator/main.py
- **Line 175-190:** The progress callback _loader_progress_callback currently increments total_completed and calls loader.update() before generation starts, causing the progress bar to advance on failures; modify the logic so the callback only updates display name and prompt (use model_names.get, loader.set_model_name, loader.set_prompt) and do NOT increment total_completed or call loader.update there—instead, move the increment of total_completed and the loader.update(...) call to occur after a successful return from _generate_single_sample (or where generation success is determined) so the progress only advances when a sample actually completes.

#### asciibench/generator/sampler.py
- **Line 441-444:** The span in generate_samples records run_id as None because generate_samples creates the tracing span using get_run_id() before generate_samples_async sets a new run_id; move the run_id initialization so generate_samples ensures a run_id is set before creating the span (call get_run_id() and if None call generate_id() and set_run_id(run_id) prior to span creation), and modify generate_samples_async so it only calls generate_id() and set_run_id(run_id) when get_run_id() returns None (i.e., do not unconditionally overwrite an existing run_id).

#### input.css
- **Line 1-3:** Add a Biome linter configuration to ignore Tailwind at-rules instead of sprinkling local ignore comments: in your biome.json under "linter.rules.suspicious.noUnknownAtRules.options.ignore" add the Tailwind at-rule names (at minimum "tailwind","apply","layer","config"; for Tailwind v4 also include "theme","utility","variant","plugin") so Biome will not flag the `@tailwind`, `@apply`, `@layer`, etc. directives found in input.css.

#### README.md
- **Line 106-112:** Update the README table to use the official env var name LOGFIRE_SEND_TO_LOGFIRE in place of LOGFIRE_ENABLED (set Required, Default `false`, and description "Set to `true` to send spans/logs to Logfire"), and verify/update the entries for LOGFIRE_SERVICE_NAME and LOGFIRE_ENVIRONMENT to reflect whether they are custom app variables or should map to official Logfire configuration names; adjust their descriptions/defaults accordingly so the table accurately matches the official Logfire configuration.

#### tailwind.config.js
- **Line 3-6:** The tailwind.config.js file's content array contains a duplicated glob entry ("./templates/**/*.html"); remove the redundant entry so the content array only lists each template glob once (edit the content property in tailwind.config.js to keep a single "./templates/**/*.html" entry).

#### tests/test_generator_main.py
- **Line 126-132:** The test reads captured.out and asserts raw strings but must first remove ANSI color codes to avoid flakiness; update the assertions in tests/test_generator_main.py to normalize output by stripping ANSI sequences (e.g. assign `cleaned = re.sub(r'\x1B\\[[0-?]*[ -/]*[\`@-\`~]', '', captured.out)`) and then check `assert "___" in cleaned`, `assert "models loaded from" in cleaned`, and `assert "models.yaml" in cleaned`; add an import for re at the top of the test if not present.

#### tests/test_logging.py
- **Line 212-217:** The test_default_log_path currently passes an explicit path to JSONLogger so it doesn't validate the logger's default location; update the test to change working dir to the tmp_path (use `monkeypatch.chdir(tmp_path)`) and instantiate `JSONLogger("test")` with no path argument, call `logger.info("Default path test")`, then assert the default file (e.g., default name used by JSONLogger) exists in tmp_path to verify default behavior; keep the test function name test_default_log_path and use the JSONLogger class to locate the relevant code under test.

#### tests/test_observability.py
- **Line 56-67:** The is_enabled logic in LogfireConfig is treating empty strings as valid tokens; update `LogfireConfig.is_enabled` to require a non-empty token (e.g., `return self.enabled and bool(self.token)`) so empty "" is considered invalid, and then update the test test_init_logfire_with_empty_token in tests/test_observability.py to stop mocking logfire.configure and assert `observability.init_logfire` returns False and that configure is not called (or keep the mock but assert no call), ensuring init_logfire respects the new validation.

#### tests/test_sampler.py
- **Line 812-814:** The strict equality assertion on max_concurrent_calls is timing-sensitive and can be flaky; instead of asserting `max_concurrent_calls == 2` in tests/test_sampler.py, relax the check to ensure concurrency happened but not exceeded the limit (e.g., `assert max_concurrent_calls <= 2` and `assert max_concurrent_calls >= 1`) or alternatively increase the asyncio.sleep used in the test to guarantee overlap — update the assertions around the max_concurrent_calls check (and/or extend the test's sleep duration) to make the test stable.
- **Line 1122-1135:** The test's mock_generate_async uses a shared list mock_responses and calls `mock_responses.pop(0)`, which can race under concurrent async tasks; replace the list with a thread-safe asynchronous queue (e.g., an `asyncio.Queue`) or a synchronized deque and update mock_generate_async to await/get from that queue instead of pop(0) so each concurrent caller consumes a single response safely; update the setup where mock_responses is populated (and any references) to use queue.put_nowait or put so the sequence of OpenRouterResponse and OpenRouterClientError is preserved for the mock_generate_async function.
- **Line 629-673:** The test is too permissive and may skip verification; change it to assert a real run_id was set and that logs contain it: call `set_run_id(None)` before invoking generate_samples, then after generate_samples assert `get_run_id()` is a non-empty str (not None) and/or capture the run_id returned or exposed by generate_samples; ensure logging is enabled by passing a log path or configuring GenerationConfig/logging so log_path.exists() is true, then read the log file unconditionally, parse JSON lines and assert at least one entry has the same run_id; update references to generate_samples, get_run_id, set_run_id, log_path, and GenerationConfig accordingly so the test fails when run_id is not propagated.

#### tests/test_state.py
- **Line 7-8:** Update the imports in tests/test_state.py to import BatchMetrics directly from its defining module: replace the current indirect import from asciibench.generator.sampler with a direct import from asciibench.generator.state so the file imports BatchMetrics and SharedState from state (use the symbols BatchMetrics and SharedState to locate the change).

---

### Nitpick Comments (16 items)

#### tests/test_judge_ui.py (1)
- **Line 338-520:** Optional: factor a `matchup_service` fixture to DRY setup. Multiple tests repeat the same `MatchupService` construction; a fixture would centralize setup and make future changes easier.

<details>
<summary>Example pattern (apply similarly across tests)</summary>

```diff
+@pytest.fixture
+def matchup_service(temp_data_dir: Path) -> MatchupService:
+    return MatchupService(
+        database_path=temp_data_dir / "database.jsonl",
+        votes_path=temp_data_dir / "votes.jsonl",
+    )

-    def test_get_pair_comparison_counts_empty(self, temp_data_dir: Path) -> None:
+    def test_get_pair_comparison_counts_empty(
+        self, matchup_service: MatchupService
+    ) -> None:
         """Test counting comparisons with no votes."""
-        matchup_service = MatchupService(
-            database_path=temp_data_dir / "database.jsonl",
-            votes_path=temp_data_dir / "votes.jsonl",
-        )
         counts = matchup_service._get_pair_comparison_counts([])
```
</details>

Also applies to: 1594-1788

#### .pre-commit-config.yaml (1)
- **Line 17-22:** Enable `pass_filenames: true` for faster commits. With `pass_filenames: false`, `ty` checks the entire project on every commit. Since `ty` supports file-scoped checks (e.g., `ty check path/to/file.py`), consider enabling `pass_filenames: true` to limit checks to staged files only.

#### scripts/verify_no_duplicates.sh (1)
- **Line 28-35:** Avoid O(n²) duplicate counting on large databases. `list.count` inside a loop over unique keys is quadratic. Use `collections.Counter` to count in one pass.

<details>
<summary>Suggested change</summary>

```diff
-import json
-import sys
+import json
+import sys
+from collections import Counter
@@
-sample_keys = []
-for line in lines:
-    sample = json.loads(line)
-    sample_keys.append((sample['model_id'], sample['prompt_text'], sample['attempt_number']))
-
-unique_keys = set(sample_keys)
-duplicates = [(key, sample_keys.count(key)) for key in unique_keys if sample_keys.count(key) > 1]
+sample_keys = []
+for line in lines:
+    sample = json.loads(line)
+    sample_keys.append((sample['model_id'], sample['prompt_text'], sample['attempt_number']))
+
+counts = Counter(sample_keys)
+duplicates = [(key, count) for key, count in counts.items() if count > 1]
```
</details>

#### asciibench/generator/state.py (1)
- **Line 135-146:** Guard against `current_tasks` underflow. A defensive check helps detect mismatched increment/decrement calls.

<details>
<summary>Optional guard</summary>

```diff
     async def decrement_concurrent(self) -> int:
         """Atomically decrement the current concurrent tasks counter.
@@
         """
         async with self._lock:
-            self.current_tasks -= 1
-            return self.current_tasks
+            if self.current_tasks == 0:
+                logger.warning("decrement_concurrent called when current_tasks is 0")
+                return 0
+            self.current_tasks -= 1
+            return self.current_tasks
```
</details>

#### templates/base.html (1)
- **Line 28-33:** Consider avoiding `!important` declarations. The `!important` declarations on `.vote-btn.active` styles can make future style overrides difficult. If the specificity battle is with htmx or JavaScript-added classes, consider increasing selector specificity instead (e.g., `button.vote-btn.active` or `.container .vote-btn.active`).

#### tests/test_config.py (2)
- **Line 64-86:** Use `tmp_path` fixture instead of writing to project directories. The test writes temporary env files to `Path(__file__).parent.parent`, which could cause issues in parallel test runs or CI environments. Use pytest's `tmp_path` fixture for test isolation.

<details>
<summary>Proposed refactor using tmp_path</summary>

```diff
-def test_settings_timeout_seconds_from_env():
+def test_settings_timeout_seconds_from_env(tmp_path):
     """Test that timeout_seconds can be loaded from env file."""
     import os

-    env_file = Path(__file__).parent.parent / "test_env_timeout.env"
+    env_file = tmp_path / "test_env_timeout.env"
     env_file.write_text("OPENROUTER_API_KEY=test_key\nOPENROUTER_TIMEOUT_SECONDS=60")
```
</details>

This applies similarly to `test_settings_timeout_seconds_negative_uses_default` (line 93) and `test_settings_timeout_seconds_non_numeric_uses_default` (line 118).

- **Line 174-191:** Move imports to module level. The `pytest` and `ValidationError` imports are duplicated inside multiple test functions. Move them to the module-level imports for clarity and to follow Python conventions.

#### templates/partials/matchup.html (1)
- **Line 5-40:** Consider extracting repeated markup into a Jinja macro. The Sample A and Sample B sections have identical structure with only the variable bindings (`sample_a` vs `sample_b`) and heading text differing. A Jinja macro could reduce duplication.

<details>
<summary>Example refactor using Jinja macro</summary>

```jinja
{% macro render_sample(sample, label) %}
<div class="sample bg-white border border-slate-200 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
    <h2 class="text-xl font-semibold text-slate-800 mb-4">{{ label }}</h2>
    <div class="font-variants">
        {% for font_name, font_class in [('Classic', 'font-classic'), ('Modern', 'font-modern'), ('Condensed', 'font-condensed')] %}
        <div class="font-variant">
            <span class="font-label bg-slate-100 text-slate-600 px-3 py-1.5 text-sm font-medium rounded-t-lg block mb-0">{{ font_name }}</span>
            <div class="art-display bg-slate-50 p-4 rounded-b-lg min-h-32 overflow-x-auto font-mono text-sm leading-relaxed text-slate-700 {{ font_class }}">{{ sample.sanitized_output }}</div>
        </div>
        {% endfor %}
    </div>
</div>
{% endmacro %}

{{ render_sample(sample_a, "Sample A") }}
{{ render_sample(sample_b, "Sample B") }}
```
</details>

The current explicit approach is also acceptable for readability and simplicity in a smaller template.

#### tests/test_retry.py (2)
- **Line 203-218:** Logging test does not assert on log content. The `caplog` fixture is injected but no assertions verify that retry attempts are actually logged. Consider asserting on `caplog.records` or `caplog.text` to validate log messages.

<details>
<summary>Suggested assertion</summary>

```diff
         failing_call()

         assert attempt_count == 3
+        # Verify retry attempts were logged
+        assert any("Retrying" in record.message or "retry" in record.message.lower() for record in caplog.records)
```
</details>

- **Line 219-237:** Logging tests lack log content verification. `test_logs_max_retries_exceeded` and `test_logs_non_retryable_exception` only verify exceptions are raised but don't assert that the expected log messages were produced. Consider adding `caplog` fixture and assertions to validate logging behavior.

#### asciibench/common/observability.py (1)
- **Line 7-8:** Consider thread safety for global initialization flag. `_LOGFIRE_INITIALIZED` is a global flag that could be accessed concurrently if `init_logfire` is called from multiple threads. While the early return pattern provides some protection, a race condition could cause duplicate initialization attempts. For an initialization function typically called once at startup, this is likely acceptable. However, if this becomes problematic, consider using a threading lock.

#### asciibench/common/config.py (2)
- **Line 49-54:** Validator logs warning but doesn't disable Logfire. When `enabled=True` but `token is None`, the validator logs a warning but returns the config unchanged. The `is_enabled` property will return `False`, but `v.enabled` remains `True`. Consider setting `v.enabled = False` to make the state consistent.

<details>
<summary>Suggested fix</summary>

```diff
     @field_validator("logfire", mode="after")
     @classmethod
     def validate_logfire(cls, v: LogfireConfig) -> LogfireConfig:
         if v.enabled and v.token is None:
             logger.warning("LOGFIRE_ENABLED is true but LOGFIRE_TOKEN missing; Logfire disabled.")
+            # Create new config with enabled=False for consistency
+            return LogfireConfig(
+                token=v.token,
+                service_name=v.service_name,
+                environment=v.environment,
+                enabled=False,
+            )
         return v
```
</details>

- **Line 61-82:** Debug logging in validator may be verbose. The `validate_openrouter_timeout_seconds` validator has multiple `logger.debug()` calls that execute on every Settings instantiation. While useful during development, this could add noise in production logs if debug level is enabled.

#### tests/test_state.py (2)
- **Line 11-16:** Custom event_loop fixture may be unnecessary. The custom `event_loop` fixture creates a new event loop but isn't used by tests (tests use `asyncio.run()` directly). If you're not using `pytest-asyncio` with `@pytest.mark.asyncio` decorators, this fixture is unused and can be removed.

<details>
<summary>Suggested removal</summary>

```diff
-@pytest.fixture
-def event_loop():
-    """Create an event loop for async tests."""
-    loop = asyncio.new_event_loop()
-    yield loop
-    loop.close()
-
-
```
</details>

- **Line 265-277:** Direct state mutation in test may bypass thread safety. Line 268 directly sets `state.current_tasks = 10` outside the async context, bypassing the lock. While this works for test setup, it doesn't follow the thread-safe API. Consider using a loop of `increment_concurrent()` calls for consistency.

#### templates/judge.html (1)
- **Line 64-69:** Consider adding ARIA labels for accessibility. The vote buttons use `data-winner` for functionality but lack explicit ARIA labels. Screen reader users would benefit from more descriptive labels.

<details>
<summary>Accessibility improvement</summary>

```diff
-        <button id="btn-a" class="vote-btn px-6 py-3 text-base font-semibold border-2 border-slate-300 rounded-lg bg-white text-slate-700 hover:bg-slate-50 hover:border-slate-400 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed" data-winner="A">A wins</button>
+        <button id="btn-a" class="vote-btn px-6 py-3 text-base font-semibold border-2 border-slate-300 rounded-lg bg-white text-slate-700 hover:bg-slate-50 hover:border-slate-400 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed" data-winner="A" aria-label="Vote for Sample A as winner">A wins</button>
```
</details>

---

## Review 2 (ID: 3757505140)
**Submitted:** 2026-02-05T14:34:31Z
**Reviewer:** coderabbitai[bot]

### Actionable Comments Summary (1 item)

#### tests/test_sampler.py
- **Line 675-721:** The test_each_sample_has_unique_request_id_in_context currently neither ensures the log_path is used nor asserts uniqueness per generated sample; update the test to configure and use log_path (or capture request IDs directly) and collect request_ids during generate_samples (e.g., by mocking the client or hooking into the logging/callback used by generate_samples) instead of relying on an unread log file; then compute `expected_count = len(sample_models) * len(sample_prompts) * GenerationConfig(attempts_per_prompt=...).attempts_per_prompt` and assert `len(set(request_ids)) == expected_count` and that no request_id is None, while still clearing request_id via `set_request_id(None)` before/after the test.

### Outside Diff Range Comments (1 item)

#### asciibench/generator/sanitizer.py (1)
- **Line 17-18:** Docstring inconsistency: claims "without any manipulation" but function now strips content. Line 18 states the content is extracted "without any manipulation," but the updated return description (line 30) and implementation now strip leading/trailing blank lines and trailing newlines. Update line 18 to reflect the actual behavior.

<details>
<summary>Suggested fix</summary>

```diff
     This function searches for code blocks in markdown format
-    (e.g., ```text...``` or ```...```) and extracts the content exactly
-    as it appears between the backticks without any manipulation.
+    (e.g., ```text...``` or ```...```) and extracts the content,
+    stripping leading/trailing blank lines and normalizing whitespace.
```
</details>

### Nitpick Comments (1 item)

#### asciibench/generator/sanitizer.py (1)
- **Line 42-44:** Consider case-insensitive matching for language specifiers. The regex only matches lowercase `text`, `ascii`, `plaintext`. If inputs may contain `TEXT`, `Ascii`, etc., consider adding `re.IGNORECASE` to the `re.search` call on line 46.

<details>
<summary>Optional fix for case-insensitive matching</summary>

```diff
-    match = re.search(pattern, markdown, re.DOTALL)
+    match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
```
</details>

---

## Inline Comments (from review threads)

### asciibench/common/loader.py:261

_⚠️ Potential issue_ | _🟡 Minor_

**Sanitize prompts to avoid embedded newlines breaking terminal output.**

The `set_prompt` method directly assigns potentially multi-line text. If a prompt contains `\n` or `\r`, it will break the single-line carriage-return based progress display.

<details>
<summary>Suggested fix</summary>

```diff
     def set_prompt(self, prompt_text: str) -> None:
         """Update the current prompt text displayed in the loader.

         The prompt is updated in-place without creating new lines.

         Args:
             prompt_text: The prompt text to display.
         """
         with self._lock:
+            # Normalize to single line for terminal display
+            clean_prompt = " ".join(prompt_text.replace("\r", " ").replace("\n", " ").split())
             # Truncate long prompts for display
             max_len = 80
-            if len(prompt_text) > max_len:
-                self._current_prompt = prompt_text[:max_len] + "..."
+            if len(clean_prompt) > max_len:
+                self._current_prompt = clean_prompt[:max_len] + "..."
             else:
-                self._current_prompt = prompt_text
+                self._current_prompt = clean_prompt
```
</details>

---

### asciibench/common/logging.py:10

_⚠️ Potential issue_ | _🟡 Minor_

**Add filelock as an explicit dependency.**

The code imports `FileLock` from `filelock`, but this package isn't declared in `pyproject.toml`.

Add to pyproject.toml dependencies:
```toml
"filelock>=3.0.0",
```

---

### asciibench/common/retry.py:72

_⚠️ Potential issue_ | _🟡 Minor_

**Decorator silently breaks async functions.**

The `@retry` decorator uses a synchronous wrapper. If applied to an async function, it will return a coroutine object without executing it, and exceptions won't be caught.

<details>
<summary>Add async support</summary>

```python
import inspect

def retry(
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    max_retries: int = 3,
    base_delay_seconds: float = 1.0,
    logger: logging.Logger | None = None,
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exception: Exception | None = None
                for attempt in range(max_retries):
                    try:
                        return await func(*args, **kwargs)
                    except retryable_exceptions as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            delay = base_delay_seconds * (2 ** attempt)
                            if logger:
                                logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                            await asyncio.sleep(delay)
                raise last_exception
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # ... existing sync logic
            return cast(F, wrapper)
    return decorator
```
</details>

---

### asciibench/common/retry.py:15

_⚠️ Potential issue_ | _🟡 Minor_

**Add input validation to retry decorator parameters.**

The decorator accepts parameters without validation. Invalid values (negative retries, non-numeric delays) would cause confusing runtime errors.

<details>
<summary>Add validation</summary>

```python
def retry(
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    max_retries: int = 3,
    base_delay_seconds: float = 1.0,
    logger: logging.Logger | None = None,
) -> Callable[[F], F]:
    # Validate inputs
    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError(f"max_retries must be a non-negative integer, got {max_retries}")
    if not isinstance(base_delay_seconds, (int, float)) or base_delay_seconds < 0:
        raise ValueError(f"base_delay_seconds must be a non-negative number, got {base_delay_seconds}")
    if not retryable_exceptions or not all(isinstance(e, type) and issubclass(e, Exception) for e in retryable_exceptions):
        raise TypeError("retryable_exceptions must be a non-empty tuple of Exception types")

    def decorator(func: F) -> F:
        # ... rest of implementation
```
</details>

---

### asciibench/generator/client.py:157

_⚠️ Potential issue_ | _🟡 Minor_

**TimeoutError should not be retried.**

When `run_with_timeout` raises a timeout, the original thread may still be running. Retrying creates duplicate in-flight requests, potentially leading to resource exhaustion or duplicate side effects.

Ensure `TimeoutError` is excluded from retryable exceptions:

```python
@retry(
    retryable_exceptions=(OpenRouterClientError, ConnectionError),  # NOT TimeoutError
    max_retries=3,
    base_delay_seconds=1.0,
)
def generate(self, ...) -> ...:
    ...
```

---

### asciibench/generator/demo.py:75

_⚠️ Potential issue_ | _🟡 Minor_

**traceback.format_exc() returns None outside except block.**

The `log_generation_error` function is called with an exception object, but `traceback.format_exc()` on line 75 only works inside an except block. When called elsewhere, it returns "NoneType: None".

<details>
<summary>Fix using exception's __traceback__</summary>

```diff
 def log_generation_error(exception: Exception, raw_output: str | None = None) -> None:
     metadata = {
         "exception_type": type(exception).__name__,
         "exception_message": str(exception),
-        "traceback": traceback.format_exc(),
+        "traceback": "".join(traceback.format_exception(type(exception), exception, exception.__traceback__)),
     }
```
</details>

---

### asciibench/generator/main.py:190

_⚠️ Potential issue_ | _🟡 Minor_

**Progress bar advances before generation completes.**

The `_loader_progress_callback` increments `total_completed` and calls `loader.update()` when generation *starts*, not when it *completes*. This means failed generations still advance the progress bar.

Move the increment to occur only after successful completion:

```python
# In callback - only update display, don't increment
def _loader_progress_callback(model_id: str, prompt_text: str) -> None:
    model_name = model_names.get(model_id, model_id)
    loader.set_model_name(model_name)
    loader.set_prompt(prompt_text)

# After successful generation
result = await _generate_single_sample(...)
if result is not None:
    nonlocal total_completed
    total_completed += 1
    loader.update(total_completed)
```

---

### asciibench/generator/sampler.py:444

_⚠️ Potential issue_ | _🟡 Minor_

**run_id is None in tracing span.**

`generate_samples` creates a span with `get_run_id()` before `generate_samples_async` sets a new run_id. The span always records `run_id=None`.

<details>
<summary>Initialize run_id before span creation</summary>

```diff
 def generate_samples(...):
+    # Ensure run_id is set before creating span
+    if get_run_id() is None:
+        set_run_id(generate_id())
+
     with logfire.span(
         "generate_samples",
         run_id=get_run_id(),
         ...
     ):
```

And in `generate_samples_async`:
```diff
-    run_id = generate_id()
-    set_run_id(run_id)
+    # Only set if not already set
+    run_id = get_run_id()
+    if run_id is None:
+        run_id = generate_id()
+        set_run_id(run_id)
```
</details>

---

### input.css:3

_🛠️ Refactor_ | _🟡 Minor_

**Configure Biome to ignore Tailwind at-rules globally.**

Instead of inline comments, configure biome.json to ignore Tailwind directives:

```json
{
  "linter": {
    "rules": {
      "suspicious": {
        "noUnknownAtRules": {
          "options": {
            "ignore": ["tailwind", "apply", "layer", "config", "theme", "utility", "variant", "plugin"]
          }
        }
      }
    }
  }
}
```

---

### README.md:112

_📚 Documentation_ | _🟡 Minor_

**Use official Logfire environment variable name.**

The README references `LOGFIRE_ENABLED`, but the official Logfire SDK uses `LOGFIRE_SEND_TO_LOGFIRE`. Update for consistency:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LOGFIRE_SEND_TO_LOGFIRE` | No | `false` | Set to `true` to send spans/logs to Logfire |

---

### tailwind.config.js:6

_🛠️ Refactor_ | _🟡 Minor_

**Remove duplicate content glob.**

The content array has `"./templates/**/*.html"` listed twice.

```diff
 content: [
     "./templates/**/*.html",
-    "./templates/**/*.html",
     "./static/**/*.js",
 ],
```

---

### tests/test_generator_main.py:132

_🧪 Test_ | _🟡 Minor_

**Strip ANSI codes before assertions.**

The test asserts on raw terminal output which may contain ANSI color codes, making tests flaky across environments.

```python
import re

def strip_ansi(text: str) -> str:
    return re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', text)

# In test
cleaned = strip_ansi(captured.out)
assert "models loaded from" in cleaned
assert "models.yaml" in cleaned
```

---

### tests/test_logging.py:217

_🧪 Test_ | _🟡 Minor_

**Test doesn't validate default path behavior.**

`test_default_log_path` passes an explicit path, so it doesn't actually test the default path logic.

```python
def test_default_log_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger = JSONLogger("test")  # No path argument
    logger.info("Default path test")
    # Assert default file exists in tmp_path
    assert (tmp_path / "test.jsonl").exists()  # or whatever the default name is
```

---

### tests/test_observability.py:67

_🧪 Test_ | _🟡 Minor_

**Empty token string treated as valid.**

`LogfireConfig.is_enabled` should return False for empty strings:

```python
@property
def is_enabled(self) -> bool:
    return self.enabled and bool(self.token)
```

Update test to verify empty token disables Logfire without calling configure.

---

### tests/test_sampler.py:814

_🧪 Test_ | _🟡 Minor_

**Flaky concurrency assertion.**

`assert max_concurrent_calls == 2` is timing-dependent. Relax to bounds check:

```python
assert 1 <= max_concurrent_calls <= 2
```

Or increase sleep duration to guarantee overlap.

---

### tests/test_sampler.py:1135

_🧪 Test_ | _🟡 Minor_

**Race condition in mock responses.**

`mock_responses.pop(0)` on a shared list can race under concurrent async tasks.

Replace with `asyncio.Queue`:

```python
mock_queue = asyncio.Queue()
for response in responses:
    mock_queue.put_nowait(response)

async def mock_generate_async(*args, **kwargs):
    return await mock_queue.get()
```

---

### tests/test_sampler.py:673

_🧪 Test_ | _🟡 Minor_

**Test may skip verification if log file doesn't exist.**

Ensure logging is configured, then unconditionally verify run_id in logs:

```python
set_run_id(None)
# ... run generate_samples with logging enabled ...
assert get_run_id() is not None
assert log_path.exists()
with open(log_path) as f:
    for line in f:
        entry = json.loads(line)
        assert entry.get("run_id") == get_run_id()
```

---

### tests/test_state.py:8

_🛠️ Refactor_ | _🟡 Minor_

**Import BatchMetrics from its defining module.**

```diff
-from asciibench.generator.sampler import BatchMetrics
+from asciibench.generator.state import BatchMetrics, SharedState
```

---

### asciibench/generator/sanitizer.py:46

_💡 Suggestion_ | _🟢 Minor_

**Consider case-insensitive matching for language specifiers.**

The regex only matches lowercase `text`, `ascii`, `plaintext`. If inputs may contain `TEXT`, `Ascii`, etc., consider adding `re.IGNORECASE`:

```diff
-    match = re.search(pattern, markdown, re.DOTALL)
+    match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
```

---

### tests/test_sampler.py:721

_🧪 Test_ | _🟡 Minor_

**test_each_sample_has_unique_request_id_in_context doesn't verify uniqueness.**

The test should:
1. Configure logging with a log_path
2. Collect request_ids during generation (via mock or log parsing)
3. Assert `len(set(request_ids)) == expected_count`
4. Assert no request_id is None

---

### asciibench/generator/sanitizer.py:18

_📚 Documentation_ | _🟡 Minor_

**Docstring claims "without any manipulation" but function strips content.**

Update docstring to reflect actual behavior:

```diff
     This function searches for code blocks in markdown format
-    (e.g., ```text...``` or ```...```) and extracts the content exactly
-    as it appears between the backticks without any manipulation.
+    (e.g., ```text...``` or ```...```) and extracts the content,
+    stripping leading/trailing blank lines and normalizing whitespace.
```
