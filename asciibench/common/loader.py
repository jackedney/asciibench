"""RuneScape-style animated loading bar component using Rich Live."""

import shutil
import threading
import time
from types import TracebackType

from rich.console import Console
from rich.live import Live
from rich.text import Text

from asciibench.common.wave_text import fill_progress, render_wave_text


class RuneScapeLoader:
    """Animated loading bar with RuneScape-style wave text and rainbow colors.

    Uses Rich Live for non-blocking animation updates. Combines wave_text rendering
    with progress fill calculation to show animated model name filling across the
    terminal as progress increases.

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
        self._lock = threading.Lock()

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
        """Change the displayed model name.

        Args:
            model_name: New model name to display.
        """
        with self._lock:
            self._model_name = model_name
            self._current_step = 0
            self._completed = False

    def update(self, current_step: int) -> None:
        """Update the progress to the given step.

        Safe to call after completion - will not error.

        Args:
            current_step: The current step number (0 to total_steps).
        """
        with self._lock:
            # Clamp to valid range
            self._current_step = max(0, min(self._total_steps, current_step))
            # Check if we've completed
            if self._current_step >= self._total_steps:
                self._completed = True

    def _get_terminal_width(self) -> int:
        """Get the current terminal width."""
        try:
            return shutil.get_terminal_size().columns
        except Exception:
            return 80  # Fallback width

    def _render_frame(self) -> Text:
        """Render the current animation frame.

        Returns:
            Rich Text object with the current loader state.
        """
        with self._lock:
            model_name = self._model_name
            progress = self._current_step / self._total_steps
            frame = self._frame

        width = self._get_terminal_width()

        # Get the filled text based on progress
        filled_text = fill_progress(model_name, width, progress)

        if not filled_text:
            return Text("")

        # Apply wave animation to the filled text
        return render_wave_text(filled_text, frame)

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
        """Start the animated loader."""
        if self._running:
            return

        self._running = True
        self._frame = 0
        self._completed = False

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
        """Stop the animated loader."""
        self._running = False

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
