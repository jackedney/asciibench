"""RuneScape-style animated loading bar component using Rich Live."""

import os
import re
import shutil
import threading
import time
from types import TracebackType

from rich.console import Console
from rich.live import Live
from rich.text import Text

from asciibench.common.wave_text import (
    render_cycling_text,
    render_gradient_text,
    render_status_line,
)

# Color constants for completion effects
SUCCESS_COLOR = "#00FF00"  # Green
FAILURE_COLOR = "#FF0000"  # Red
FLASH_DURATION_MS = 300  # Duration of flash effect in milliseconds

# Fallback progress bar constants
FALLBACK_BAR_WIDTH = 20  # Width of the progress bar in characters


def detect_terminal_capabilities(console: Console | None = None) -> dict[str, bool]:
    """Detect terminal capabilities for the loader.

    Checks for:
    - is_terminal: Whether output is going to a terminal (not piped)
    - has_color: Whether the terminal supports colors
    - supports_live: Whether Rich Live updates are supported

    The NO_COLOR environment variable (https://no-color.org/) is respected
    and disables color support when set.

    Args:
        console: Optional Rich Console to check. Creates one if not provided.

    Returns:
        Dictionary with capability flags:
        - is_terminal: True if stdout is a terminal
        - has_color: True if color output is supported
        - supports_live: True if Live updates are supported (requires terminal)

    Example:
        >>> caps = detect_terminal_capabilities()
        >>> if not caps['has_color']:
        ...     print("Using fallback mode")
    """
    if console is None:
        console = Console()

    # Check if NO_COLOR environment variable is set
    no_color = os.environ.get("NO_COLOR") is not None

    # Check if we're outputting to a terminal
    is_terminal = console.is_terminal

    # Check for color support (Rich's color_system returns None if no color)
    # Also respect NO_COLOR environment variable
    has_color = console.color_system is not None and not no_color

    # Live updates require a terminal - piped output cannot do in-place updates
    supports_live = is_terminal

    return {
        "is_terminal": is_terminal,
        "has_color": has_color,
        "supports_live": supports_live,
    }


def format_simple_progress(
    model_name: str,
    progress: float,
    success_count: int = 0,
    failure_count: int = 0,
    total_cost: float = 0.0,
    width: int = FALLBACK_BAR_WIDTH,
) -> str:
    """Format a simple text-based progress bar with status totals.

    Creates a simple progress display like:
    'Model: GPT-4o [=====>    ] 50% | ✓ 5 ✗ 2 $0.01'

    Args:
        model_name: The model name to display.
        progress: Progress value from 0.0 to 1.0.
        success_count: Number of successful operations (default 0).
        failure_count: Number of failed operations (default 0).
        total_cost: Total accumulated cost (default 0.0).
        width: Width of the progress bar (default 20).

    Returns:
        Formatted progress string with status totals.

    Example:
        >>> format_simple_progress('GPT-4o', 0.5, 5, 2, 0.01)
        'Model: GPT-4o [==========>          ]  50% | ✓ 5 ✗ 2 $0.0100'
        >>> format_simple_progress('GPT-4o', 0.0)
        'Model: GPT-4o [                    ]   0% | ✓ 0 ✗ 0 $0.0000'
        >>> format_simple_progress('GPT-4o', 1.0)
        'Model: GPT-4o [====================] 100% | ✓ 0 ✗ 0 $0.0000'
    """
    # Clamp progress to valid range
    progress = max(0.0, min(1.0, progress))

    # Calculate filled portion
    filled = int(width * progress)
    remaining = width - filled

    # Build the bar - use '=' for filled, '>' for the leading edge (if not at start/end)
    if filled == 0:
        bar = " " * width
    elif filled == width:
        bar = "=" * width
    else:
        bar = "=" * (filled - 1) + ">" + " " * remaining

    # Format percentage with consistent width
    percent = int(progress * 100)

    # Format status totals
    status_totals = f" | ✓ {success_count} ✗ {failure_count} ${total_cost:.4f}"

    return f"Model: {model_name} [{bar}] {percent:3d}%{status_totals}"


class RuneScapeLoader:
    """Animated loading bar with RuneScape-style wave text and rainbow colors.

    Uses Rich Live for non-blocking animation updates. Combines wave_text rendering
    with progress fill calculation to show animated model name filling across the
    terminal as progress increases.

    Automatically detects terminal capabilities and falls back to simple text
    progress if the terminal doesn't support colors or animations:
    - If NO_COLOR environment variable is set: uses simple text progress
    - If output is piped (not a terminal): uses simple line-based updates
    - If terminal doesn't support colors: uses simple text progress

    Args:
        model_name: The model name to display (e.g., 'GPT-4o').
        total_steps: Total number of steps for the loading process.
        refresh_rate: Animation refresh rate in FPS (default: 12).
        console: Optional Rich Console instance (uses default if not provided).

    Example:
        >>> with RuneScapeLoader('Claude-4o', total_steps=100) as loader:
        ...     for i in range(100):
        ...         loader.update(i + 1)
        ...         time.sleep(0.1)

    Fallback mode example (when NO_COLOR=1 or piped output):
        Model: Claude-4o [==========>          ]  50%
    """

    def __init__(
        self,
        model_name: str,
        total_steps: int,
        refresh_rate: float = 12.0,
        console: Console | None = None,
    ) -> None:
        """Initialize the loader with model name and total steps."""
        self._model_name = model_name
        self._total_steps = max(1, total_steps)  # Avoid division by zero
        self._current_step = 0
        self._refresh_rate = max(1.0, min(30.0, refresh_rate))  # Clamp to 1-30 FPS
        self._console = console or Console()
        self._frame = 0
        self._live: Live | None = None
        self._animation_thread: threading.Thread | None = None
        self._running = False
        self._completed = False
        self._flashing = False  # Guards against stacking flash effects
        self._flash_color: str | None = None  # Current flash color if flashing
        self._lock = threading.Lock()

        # Running totals for status line
        self._success_count = 0
        self._failure_count = 0
        self._total_cost = 0.0

        # Current prompt text (updated in-place)
        self._current_prompt: str = ""

        # Detect terminal capabilities for fallback mode
        self._capabilities = detect_terminal_capabilities(self._console)
        has_color = self._capabilities["has_color"]
        supports_live = self._capabilities["supports_live"]
        self._use_fallback = not has_color or not supports_live
        self._last_printed_progress: float | None = None  # Track last progress

    @property
    def use_fallback(self) -> bool:
        """Check if the loader is using fallback mode."""
        return self._use_fallback

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self._model_name

    @property
    def progress(self) -> float:
        """Get the current progress as a value between 0.0 and 1.0."""
        with self._lock:
            return self._current_step / self._total_steps

    @property
    def is_complete(self) -> bool:
        """Check if the loader has completed."""
        with self._lock:
            return self._completed

    def set_model(self, model_name: str) -> None:
        """Change the displayed model name and reset progress.

        Use this when starting a new model batch. For updating the display
        name without resetting progress, use set_model_name() instead.

        Args:
            model_name: New model name to display.
        """
        with self._lock:
            self._model_name = model_name
            self._current_step = 0
            self._completed = False
            self._last_printed_progress = None

    def set_model_name(self, model_name: str) -> None:
        """Update the displayed model name without resetting progress.

        Use this to update the model name in the display while preserving
        the current progress. For resetting progress, use set_model() instead.

        Args:
            model_name: New model name to display.
        """
        with self._lock:
            self._model_name = model_name

    def set_prompt(self, prompt_text: str) -> None:
        """Update the current prompt text displayed in the loader.

        The prompt is updated in-place without creating new lines.

        Args:
            prompt_text: The prompt text to display.
        """
        with self._lock:
            # Sanitize prompt: replace all \r and \n with single space, collapse repeated whitespace
            if prompt_text:
                sanitized = re.sub(r"[\r\n]+", " ", prompt_text)
                sanitized = re.sub(r"[ \t]+", " ", sanitized).strip()
            else:
                sanitized = ""

            # Truncate long prompts for display
            max_len = 80
            if len(sanitized) > max_len:
                self._current_prompt = sanitized[:max_len] + "..."
            else:
                self._current_prompt = sanitized

    def record_result(self, success: bool, cost: float = 0.0) -> None:
        """Record a generation result and update the status counters.

        Thread-safe method to update success/failure counts and cost.

        Note: This method is mutually exclusive with complete() for the same
        result. Do not call both record_result() and complete() for the same
        operation, as this would double-increment counters.

        Args:
            success: True if generation was successful, False otherwise.
            cost: Cost to add to total (default 0.0).
        """
        with self._lock:
            if success:
                self._success_count += 1
            else:
                self._failure_count += 1
            self._total_cost += cost

    def update(self, current_step: int) -> None:
        """Update the progress to the given step.

        Safe to call after completion - will not error.

        In fallback mode, prints a new line with the progress bar when progress
        changes significantly (to avoid excessive output).

        Args:
            current_step: The current step number (0 to total_steps).
        """
        with self._lock:
            # Clamp to valid range
            self._current_step = max(0, min(self._total_steps, current_step))
            # Check if we've completed
            if self._current_step >= self._total_steps:
                self._completed = True
            progress = self._current_step / self._total_steps
            model_name = self._model_name

        # In fallback mode, print progress updates
        if self._use_fallback and self._running:
            self._print_fallback_progress(progress, model_name)

    def _print_fallback_progress(self, progress: float, model_name: str) -> None:
        """Print fallback progress bar to console.

        For terminals: uses carriage return to update in place.
        For piped output: prints new lines only when progress changes significantly.

        Args:
            progress: Current progress from 0.0 to 1.0.
            model_name: The model name to display.
        """
        # Only print if progress has changed enough (5% increments or completion)
        if self._last_printed_progress is not None:
            progress_diff = abs(progress - self._last_printed_progress)
            if progress_diff < 0.05 and progress < 1.0:
                return

        self._last_printed_progress = progress

        # Get current status totals
        with self._lock:
            success_count = self._success_count
            failure_count = self._failure_count
            total_cost = self._total_cost
            current_prompt = self._current_prompt

        progress_text = format_simple_progress(
            model_name, progress, success_count, failure_count, total_cost
        )

        # Include prompt info for fallback
        if current_prompt:
            prompt_display = (
                current_prompt[:40] + "..." if len(current_prompt) > 40 else current_prompt
            )
            progress_text = f"[{prompt_display}] {progress_text}"

        if self._capabilities["is_terminal"]:
            # Terminal mode: use carriage return to update in place
            self._console.print(f"\r{progress_text}", end="")
        else:
            # Piped mode: print new lines
            self._console.print(progress_text)

    def complete(self, success: bool, cost: float = 0.0) -> None:
        """Flash the bar to indicate completion status and update counters.

        Shows a brief color flash (green for success, red for failure) for ~300ms,
        then clears and prepares the loader for the next model. Increments the
        success/failure counters and total cost based on the result.

        In fallback mode, prints a completion message instead of flashing.

        Calling complete multiple times does not stack flashes - if a flash is
        already in progress, subsequent calls are ignored.

        Note: This method is mutually exclusive with record_result() for the
        same result. Do not call both complete() and record_result() for the same
        operation, as this would double-increment counters.

        Args:
            success: True for green (success) flash, False for red (failure) flash.
            cost: Cost to add to total_cost (default 0.0).

        Example:
            >>> loader.complete(success=True, cost=0.005)  # Shows green flash, adds cost
            >>> loader.complete(success=False, cost=0.0)  # Shows red flash, no cost
        """
        # Increment counters before flash
        with self._lock:
            if success:
                self._success_count += 1
            else:
                self._failure_count += 1
            self._total_cost += cost

        # Handle fallback mode separately
        if self._use_fallback:
            with self._lock:
                model_name = self._model_name
                self._current_step = 0
                self._completed = False
                self._last_printed_progress = None

            if self._running:
                status = "DONE" if success else "FAILED"
                with self._lock:
                    success_count = self._success_count
                    failure_count = self._failure_count
                    total_cost = self._total_cost

                progress_text = format_simple_progress(
                    model_name, 1.0, success_count, failure_count, total_cost
                )
                if self._capabilities["is_terminal"]:
                    # Clear the line and print final status
                    self._console.print(f"\r{progress_text} [{status}]")
                else:
                    self._console.print(f"{progress_text} [{status}]")
            return

        with self._lock:
            # Guard against stacking flashes
            if self._flashing:
                return
            self._flashing = True
            self._flash_color = SUCCESS_COLOR if success else FAILURE_COLOR

        # Update display to show flash
        if self._live is not None:
            try:
                self._live.update(self._render_frame())
            except Exception:
                pass

        # Hold the flash for the specified duration
        time.sleep(FLASH_DURATION_MS / 1000.0)

        # Clear the flash state and reset for next model
        with self._lock:
            self._flashing = False
            self._flash_color = None
            self._current_step = 0
            self._completed = False

        # Update display to clear the flash
        if self._live is not None:
            try:
                self._live.update(self._render_frame())
            except Exception:
                pass

    def _get_terminal_width(self) -> int:
        """Get the current terminal width."""
        try:
            return shutil.get_terminal_size().columns
        except Exception:
            return 80  # Fallback width

    def _render_frame(self) -> Text:
        """Render the current animation frame with 4-layer layout.

        Layout:
        - Line 1: Current prompt text (truncated)
        - Line 2: Model name with rainbow cycling
        - Line 3: Progress bar with gradient
        - Line 4: Status line (success/failure/cost)

        Returns:
            Rich Text object with the current loader state.
        """
        with self._lock:
            model_name = self._model_name
            progress = self._current_step / self._total_steps
            frame = self._frame
            flashing = self._flashing
            flash_color = self._flash_color
            success_count = self._success_count
            failure_count = self._failure_count
            total_cost = self._total_cost
            current_prompt = self._current_prompt

        width = self._get_terminal_width()

        # If flashing, render entire display in flash color
        if flashing and flash_color is not None:
            result = Text()
            result.append("━" * width, style=flash_color)
            return result

        result = Text()

        # Line 1: Current prompt (if set)
        if current_prompt:
            result.append("Prompt: ", style="bold cyan")
            result.append(current_prompt, style="white")
            result.append("\n")

        # Line 2: Model name with color cycling
        model_name_text = render_cycling_text(model_name, frame, shift_interval=2)
        result.append(model_name_text)
        result.append("\n")

        # Line 3: Progress bar with gradient
        # Use ━ (filled) and ─ (empty) for progress bar
        bar_width = min(80, width - 4)  # Limit bar width to 80 or available width
        filled_count = int(bar_width * progress)
        empty_count = bar_width - filled_count
        progress_bar_text = "━" * filled_count + "─" * empty_count
        progress_bar = render_gradient_text(progress_bar_text, frame, filled_ratio=progress)
        result.append(progress_bar)
        result.append("\n")

        # Line 4: Status line with success/failure counts and cost
        status_line = render_status_line(success_count, failure_count, total_cost)
        result.append(status_line)

        return result

    def _animation_loop(self) -> None:
        """Background thread that updates the animation frames."""
        interval = 1.0 / self._refresh_rate

        while self._running:
            with self._lock:
                self._frame += 1

            if self._live is not None:
                try:
                    self._live.update(self._render_frame())
                except Exception:
                    # Handle case where Live context is closed
                    break

            time.sleep(interval)

    def start(self) -> None:
        """Start the animated loader.

        In fallback mode, no animation thread or Live display is started.
        Progress updates are printed directly in the update() method.
        """
        if self._running:
            return

        self._running = True
        self._frame = 0
        self._completed = False
        self._last_printed_progress = None

        # In fallback mode, don't start animation or Live display
        if self._use_fallback:
            return

        # Create and start the Live display
        self._live = Live(
            self._render_frame(),
            console=self._console,
            refresh_per_second=self._refresh_rate,
            transient=True,
        )
        self._live.start()

        # Start the animation thread
        self._animation_thread = threading.Thread(target=self._animation_loop, daemon=True)
        self._animation_thread.start()

    def stop(self) -> None:
        """Stop the animated loader.

        In fallback mode, ensures a newline is printed after progress output.
        """
        self._running = False

        # In fallback mode, ensure we end on a new line if we printed anything
        if self._use_fallback:
            if self._last_printed_progress is not None and self._capabilities["is_terminal"]:
                self._console.print()  # End the line after carriage return updates
            return

        # Wait for animation thread to finish
        if self._animation_thread is not None:
            self._animation_thread.join(timeout=0.5)
            self._animation_thread = None

        # Stop the Live display
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def __enter__(self) -> "RuneScapeLoader":
        """Enter context manager - starts the loader."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager - stops the loader."""
        self.stop()
