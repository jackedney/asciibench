"""Tests for the RuneScape-style animated loading bar component."""

import os
import threading
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.text import Text

from asciibench.common.loader import (
    FAILURE_COLOR,
    FALLBACK_BAR_WIDTH,
    FLASH_DURATION_MS,
    SUCCESS_COLOR,
    RuneScapeLoader,
    detect_terminal_capabilities,
    format_simple_progress,
)


class TestRuneScapeLoaderInit:
    """Tests for RuneScapeLoader initialization."""

    def test_init_with_model_name_and_steps(self):
        """Constructor takes model_name and total_steps."""
        loader = RuneScapeLoader("GPT-4o", total_steps=100)
        assert loader.model_name == "GPT-4o"
        assert loader._total_steps == 100

    def test_init_default_refresh_rate(self):
        """Default refresh rate is 12 FPS."""
        loader = RuneScapeLoader("Model", total_steps=10)
        assert loader._refresh_rate == 12.0

    def test_init_custom_refresh_rate(self):
        """Custom refresh rate can be set."""
        loader = RuneScapeLoader("Model", total_steps=10, refresh_rate=15.0)
        assert loader._refresh_rate == 15.0

    def test_init_refresh_rate_clamped_low(self):
        """Refresh rate below 1 is clamped to 1."""
        loader = RuneScapeLoader("Model", total_steps=10, refresh_rate=0.5)
        assert loader._refresh_rate == 1.0

    def test_init_refresh_rate_clamped_high(self):
        """Refresh rate above 30 is clamped to 30."""
        loader = RuneScapeLoader("Model", total_steps=10, refresh_rate=60.0)
        assert loader._refresh_rate == 30.0

    def test_init_custom_console(self):
        """Custom console can be provided."""
        console = Console()
        loader = RuneScapeLoader("Model", total_steps=10, console=console)
        assert loader._console is console

    def test_init_total_steps_zero_becomes_one(self):
        """Total steps of 0 becomes 1 to avoid division by zero."""
        loader = RuneScapeLoader("Model", total_steps=0)
        assert loader._total_steps == 1

    def test_init_total_steps_negative_becomes_one(self):
        """Negative total steps becomes 1."""
        loader = RuneScapeLoader("Model", total_steps=-5)
        assert loader._total_steps == 1

    def test_initial_progress_is_zero(self):
        """Initial progress is 0.0."""
        loader = RuneScapeLoader("Model", total_steps=100)
        assert loader.progress == 0.0

    def test_initial_is_complete_false(self):
        """Initial is_complete is False."""
        loader = RuneScapeLoader("Model", total_steps=100)
        assert loader.is_complete is False


class TestRuneScapeLoaderUpdate:
    """Tests for update method."""

    def test_update_changes_progress(self):
        """update() changes the progress value."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(50)
        assert loader.progress == 0.5

    def test_update_to_full_progress(self):
        """update() to total_steps sets progress to 1.0."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(100)
        assert loader.progress == 1.0

    def test_update_after_completion_does_not_error(self):
        """Calling update after completion does not error."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(100)
        assert loader.is_complete is True
        # Should not raise
        loader.update(200)
        loader.update(50)

    def test_update_clamps_to_max(self):
        """update() with step > total clamps to total."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(150)
        assert loader.progress == 1.0
        assert loader._current_step == 100

    def test_update_clamps_to_zero(self):
        """update() with negative step clamps to 0."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(-10)
        assert loader.progress == 0.0
        assert loader._current_step == 0

    def test_update_sets_completed_at_full(self):
        """update() to full progress sets is_complete to True."""
        loader = RuneScapeLoader("Model", total_steps=100)
        assert loader.is_complete is False
        loader.update(100)
        assert loader.is_complete is True


class TestRuneScapeLoaderSetModel:
    """Tests for set_model method."""

    def test_set_model_changes_model_name(self):
        """set_model() changes the displayed model name."""
        loader = RuneScapeLoader("GPT-4o", total_steps=100)
        loader.set_model("Claude-3")
        assert loader.model_name == "Claude-3"

    def test_set_model_resets_progress(self):
        """set_model() resets progress to 0."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(50)
        assert loader.progress == 0.5
        loader.set_model("NewModel")
        assert loader.progress == 0.0

    def test_set_model_resets_completed(self):
        """set_model() resets is_complete to False."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(100)
        assert loader.is_complete is True
        loader.set_model("NewModel")
        assert loader.is_complete is False


class TestRuneScapeLoaderRendering:
    """Tests for rendering functionality."""

    def test_render_frame_returns_text(self):
        """_render_frame returns a Rich Text object."""
        loader = RuneScapeLoader("Model", total_steps=100)
        result = loader._render_frame()
        assert isinstance(result, Text)

    def test_render_frame_at_zero_progress_is_empty(self):
        """At 0% progress, rendered frame shows all 4 layers with empty progress bar."""
        loader = RuneScapeLoader("Model", total_steps=100)
        result = loader._render_frame()
        # With new 4-layer layout, always shows status line, model name, empty
        # progress bar, domino row
        result_str = str(result)
        assert "✓" in result_str  # Status line
        assert "Model" in result_str  # Model name
        assert "─" in result_str  # Empty progress bar
        assert "▌" in result_str  # Dominoes

    def test_render_frame_at_full_progress(self):
        """At 100% progress, rendered frame shows filled progress bar in 4-layer layout."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(100)
        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            result_str = str(result)
            # Should have 4 layers (3 newlines)
            assert result_str.count("\n") == 3
            # Progress bar should be filled (━━━)
            assert "━━" in result_str

    def test_render_frame_at_partial_progress(self):
        """At 50% progress, rendered frame shows half-filled progress bar."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(50)
        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            result_str = str(result)
            # Should have 4 layers (3 newlines)
            assert result_str.count("\n") == 3
            # Progress bar should be half filled
            lines = result_str.split("\n")
            progress_bar = None
            for line in lines:
                if "━" in line or "─" in line:
                    progress_bar = line
                    break
            assert progress_bar is not None
            # Should have both filled and empty characters
            assert "━" in progress_bar
            assert "─" in progress_bar

    def test_render_frame_contains_model_name(self):
        """Rendered frame contains the model name."""
        loader = RuneScapeLoader("TestModel", total_steps=100)
        loader.update(100)
        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            assert "TestModel" in str(result)


class TestRuneScapeLoaderContextManager:
    """Tests for context manager support."""

    def test_context_manager_enter_returns_loader(self):
        """__enter__ returns the loader instance."""
        loader = RuneScapeLoader("Model", total_steps=100)
        with patch.object(loader, "start"):
            with loader as ctx:
                assert ctx is loader

    def test_context_manager_calls_start(self):
        """Entering context manager calls start()."""
        loader = RuneScapeLoader("Model", total_steps=100)
        with patch.object(loader, "start") as mock_start:
            with patch.object(loader, "stop"):
                with loader:
                    mock_start.assert_called_once()

    def test_context_manager_calls_stop_on_exit(self):
        """Exiting context manager calls stop()."""
        loader = RuneScapeLoader("Model", total_steps=100)
        with patch.object(loader, "start"):
            with patch.object(loader, "stop") as mock_stop:
                with loader:
                    pass
                mock_stop.assert_called_once()

    def test_context_manager_calls_stop_on_exception(self):
        """stop() is called even if exception occurs inside context."""
        loader = RuneScapeLoader("Model", total_steps=100)
        with patch.object(loader, "start"):
            with patch.object(loader, "stop") as mock_stop:
                with pytest.raises(ValueError, match="test error"):
                    with loader:
                        raise ValueError("test error")
                mock_stop.assert_called_once()


class TestRuneScapeLoaderStartStop:
    """Tests for start and stop methods."""

    def test_start_sets_running_true(self):
        """start() sets _running to True."""
        loader = RuneScapeLoader("Model", total_steps=100)
        with patch("asciibench.common.loader.Live"):
            loader.start()
            try:
                assert loader._running is True
            finally:
                loader._running = False

    def test_start_resets_frame_counter(self):
        """start() resets frame counter to 0 before animation begins."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader._frame = 100
        with patch("asciibench.common.loader.Live"):
            # Immediately stop after start to prevent thread from running
            loader.start()
            loader.stop()
            # Frame should have been reset from 100 to a low value (0 or small increment)
            assert loader._frame < 100

    def test_stop_sets_running_false(self):
        """stop() sets _running to False."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader._running = True
        loader.stop()
        assert loader._running is False

    def test_double_start_is_safe(self):
        """Calling start() twice does not error."""
        loader = RuneScapeLoader("Model", total_steps=100)
        with patch("asciibench.common.loader.Live"):
            loader.start()
            try:
                loader.start()  # Should not raise
            finally:
                loader._running = False

    def test_double_stop_is_safe(self):
        """Calling stop() twice does not error."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.stop()
        loader.stop()  # Should not raise


class TestRuneScapeLoaderProperties:
    """Tests for loader properties."""

    def test_model_name_property(self):
        """model_name property returns the current model name."""
        loader = RuneScapeLoader("TestModel", total_steps=100)
        assert loader.model_name == "TestModel"

    def test_progress_property_at_zero(self):
        """progress property returns 0.0 at start."""
        loader = RuneScapeLoader("Model", total_steps=100)
        assert loader.progress == 0.0

    def test_progress_property_at_half(self):
        """progress property returns 0.5 at half progress."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(50)
        assert loader.progress == 0.5

    def test_progress_property_at_full(self):
        """progress property returns 1.0 at full progress."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(100)
        assert loader.progress == 1.0

    def test_is_complete_property_false_initially(self):
        """is_complete property is False initially."""
        loader = RuneScapeLoader("Model", total_steps=100)
        assert loader.is_complete is False

    def test_is_complete_property_true_after_completion(self):
        """is_complete property is True after reaching total steps."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(100)
        assert loader.is_complete is True


class TestRuneScapeLoaderTerminalWidth:
    """Tests for terminal width handling."""

    def test_get_terminal_width_returns_int(self):
        """_get_terminal_width returns an integer."""
        loader = RuneScapeLoader("Model", total_steps=100)
        width = loader._get_terminal_width()
        assert isinstance(width, int)

    def test_get_terminal_width_fallback(self):
        """Falls back to 80 if terminal size cannot be determined."""
        loader = RuneScapeLoader("Model", total_steps=100)
        with patch("shutil.get_terminal_size", side_effect=Exception("no terminal")):
            width = loader._get_terminal_width()
            assert width == 80


class TestRuneScapeLoaderThreadSafety:
    """Tests for thread safety."""

    def test_update_is_thread_safe(self):
        """Multiple threads can call update() safely."""
        loader = RuneScapeLoader("Model", total_steps=1000)

        def update_many():
            for i in range(100):
                loader.update(i)

        threads = [threading.Thread(target=update_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors

    def test_set_model_is_thread_safe(self):
        """set_model() is thread safe."""
        loader = RuneScapeLoader("Model", total_steps=100)

        def change_model():
            for i in range(50):
                loader.set_model(f"Model{i}")

        threads = [threading.Thread(target=change_model) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors


class TestRuneScapeLoaderExampleUsage:
    """Tests demonstrating expected usage patterns."""

    def test_example_30_percent_progress(self):
        """Example: loader shows partial fill at 30% progress."""
        loader = RuneScapeLoader("Claude-4o", total_steps=100)
        loader.update(30)
        with patch.object(loader, "_get_terminal_width", return_value=100):
            result = loader._render_frame()
            result_str = str(result)
            # Should contain model name
            assert "Claude-4o" in result_str
            # Progress bar should have filled portion
            assert "━" in result_str

    def test_example_update_after_completion(self):
        """Example: calling update after completion does not error."""
        loader = RuneScapeLoader("Model", total_steps=10)
        for i in range(15):  # Intentionally go beyond total_steps
            loader.update(i)  # Should not raise

    def test_example_context_manager_usage(self):
        """Example: context manager for clean start/stop."""
        with patch("asciibench.common.loader.Live"):
            with RuneScapeLoader("GPT-4o", total_steps=100) as loader:
                loader.update(50)
                assert loader.progress == 0.5


class TestRuneScapeLoaderComplete:
    """Tests for completion and failure color effects (US-004)."""

    def test_complete_method_exists(self):
        """complete(success: bool) method exists on RuneScapeLoader."""
        loader = RuneScapeLoader("Model", total_steps=100)
        assert hasattr(loader, "complete")
        assert callable(loader.complete)

    def test_complete_success_sets_flash_color_green(self):
        """complete(success=True) sets flash color to green (#00FF00)."""
        # Force terminal mode to use the flash code path (not fallback)
        env = os.environ.copy()
        env.pop("NO_COLOR", None)
        with patch.dict(os.environ, env, clear=True):
            console = Console(force_terminal=True)
            loader = RuneScapeLoader("Model", total_steps=100, console=console)
            loader.update(100)  # Complete the progress

            # Capture flash color during the flash
            captured_color = []

            def capture_during_flash(duration):
                # Capture state during the flash (while sleeping)
                with loader._lock:
                    if loader._flashing:
                        captured_color.append(loader._flash_color)
                # Don't actually sleep to speed up test

            with patch("asciibench.common.loader.time.sleep", side_effect=capture_during_flash):
                loader.complete(success=True)

            # Verify green color was used during flash
            assert SUCCESS_COLOR in captured_color

    def test_complete_failure_sets_flash_color_red(self):
        """complete(success=False) sets flash color to red (#FF0000)."""
        # Force terminal mode to use the flash code path (not fallback)
        env = os.environ.copy()
        env.pop("NO_COLOR", None)
        with patch.dict(os.environ, env, clear=True):
            console = Console(force_terminal=True)
            loader = RuneScapeLoader("Model", total_steps=100, console=console)
            loader.update(100)  # Complete the progress

            captured_color = []

            def capture_during_flash(duration):
                with loader._lock:
                    if loader._flashing:
                        captured_color.append(loader._flash_color)

            with patch("asciibench.common.loader.time.sleep", side_effect=capture_during_flash):
                loader.complete(success=False)

            assert FAILURE_COLOR in captured_color

    def test_complete_resets_progress_after_flash(self):
        """After flash, loader resets progress for next model."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(100)
        assert loader.progress == 1.0

        with patch("asciibench.common.loader.time.sleep"):
            loader.complete(success=True)

        # After complete, progress should be reset
        assert loader.progress == 0.0
        assert loader.is_complete is False

    def test_complete_clears_flash_state_after_duration(self):
        """Flash state is cleared after the flash duration."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(100)

        with patch("asciibench.common.loader.time.sleep"):
            loader.complete(success=True)

        # After complete, flash state should be cleared
        assert loader._flashing is False
        assert loader._flash_color is None

    def test_complete_multiple_times_does_not_stack(self):
        """Calling complete multiple times does not stack flashes."""
        # Force terminal mode to use the flash code path (not fallback)
        env = os.environ.copy()
        env.pop("NO_COLOR", None)
        with patch.dict(os.environ, env, clear=True):
            console = Console(force_terminal=True)
            loader = RuneScapeLoader("Model", total_steps=100, console=console)
            loader.update(100)

            flash_count = 0

            def count_flash(duration):
                nonlocal flash_count
                flash_count += 1

            with patch("asciibench.common.loader.time.sleep", side_effect=count_flash):
                # First call should flash
                loader.complete(success=True)
                assert flash_count == 1

                # Second call should also flash (since first is done)
                loader.update(100)
                loader.complete(success=False)
                assert flash_count == 2

    def test_complete_during_flash_is_ignored(self):
        """If complete is called while already flashing, it's ignored."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(100)

        # Manually set flashing state
        with loader._lock:
            loader._flashing = True
            loader._flash_color = SUCCESS_COLOR

        # This should be ignored since we're already flashing
        with patch("asciibench.common.loader.time.sleep") as mock_sleep:
            loader.complete(success=False)
            mock_sleep.assert_not_called()

    def test_complete_example_success_shows_green_flash(self):
        """Example: loader.complete(success=True) shows green flash then clears."""
        loader = RuneScapeLoader("GPT-4o", total_steps=100)
        loader.update(100)

        with patch("asciibench.common.loader.time.sleep"):
            loader.complete(success=True)

        # After complete, loader should be ready for next model
        assert loader.progress == 0.0
        assert loader.is_complete is False
        assert loader._flashing is False

    def test_complete_example_failure_shows_red_flash(self):
        """Example: loader.complete(success=False) shows red flash then clears."""
        loader = RuneScapeLoader("GPT-4o", total_steps=100)
        loader.update(50)  # Partial progress before failure

        with patch("asciibench.common.loader.time.sleep"):
            loader.complete(success=False)

        # After complete, loader should be ready for next model
        assert loader.progress == 0.0
        assert loader.is_complete is False
        assert loader._flashing is False

    def test_flash_renders_full_width(self):
        """During flash, bar renders at full width with flash color."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(50)  # 50% progress

        with loader._lock:
            loader._flashing = True
            loader._flash_color = SUCCESS_COLOR

        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            # During flash, should render full width with filled bars
            result_str = str(result)
            # Should have filled characters spanning the width
            assert "━" in result_str
            # The flash bar should be the full width (no empty characters in progress bar)
            lines = result_str.split("\n")
            flash_line = None
            for line in lines:
                if "━" in line:
                    flash_line = line
                    break
            assert flash_line is not None
            # Flash line should be all filled (no ─ characters)
            assert "─" not in flash_line

    def test_flash_renders_with_failure_color(self):
        """Flash renders with red color for failure."""
        loader = RuneScapeLoader("Model", total_steps=100)

        with loader._lock:
            loader._flashing = True
            loader._flash_color = FAILURE_COLOR

        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            # Flash renders filled bars in the flash color
            result_str = str(result)
            # Should have filled characters spanning the width
            assert "━" in result_str

    def test_complete_with_live_display_updates(self):
        """complete() updates the Live display during flash."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(100)

        mock_live = patch.object(loader, "_live")
        with patch("asciibench.common.loader.time.sleep"):
            with mock_live as live_mock:
                live_mock.update = lambda x: None  # Mock update method
                loader._live = live_mock
                loader.complete(success=True)

    def test_constants_have_correct_values(self):
        """Verify color constants have correct hex values."""
        assert SUCCESS_COLOR == "#00FF00"
        assert FAILURE_COLOR == "#FF0000"
        assert FLASH_DURATION_MS == 300


class TestDetectTerminalCapabilities:
    """Tests for terminal capability detection (US-008)."""

    def test_returns_dict_with_expected_keys(self):
        """detect_terminal_capabilities returns dict with required keys."""
        caps = detect_terminal_capabilities()
        assert "is_terminal" in caps
        assert "has_color" in caps
        assert "supports_live" in caps

    def test_all_values_are_booleans(self):
        """All capability values are booleans."""
        caps = detect_terminal_capabilities()
        assert isinstance(caps["is_terminal"], bool)
        assert isinstance(caps["has_color"], bool)
        assert isinstance(caps["supports_live"], bool)

    def test_accepts_console_argument(self):
        """Can pass a custom Console instance."""
        console = Console()
        caps = detect_terminal_capabilities(console)
        assert isinstance(caps, dict)

    def test_no_color_env_disables_color(self):
        """NO_COLOR environment variable disables color support."""
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            # Create a fresh console to pick up the environment
            console = Console(force_terminal=True)
            caps = detect_terminal_capabilities(console)
            assert caps["has_color"] is False

    def test_no_color_env_not_set_allows_color(self):
        """Without NO_COLOR, color support depends on terminal."""
        env = os.environ.copy()
        env.pop("NO_COLOR", None)  # Ensure NO_COLOR is not set
        with patch.dict(os.environ, env, clear=True):
            console = Console(force_terminal=True)
            caps = detect_terminal_capabilities(console)
            # In a forced terminal, should have color support
            assert caps["has_color"] is True

    def test_non_terminal_has_no_live_support(self):
        """Non-terminal output (piped) does not support Live updates."""
        console = Console(force_terminal=False)
        caps = detect_terminal_capabilities(console)
        assert caps["supports_live"] is False
        assert caps["is_terminal"] is False

    def test_terminal_has_live_support(self):
        """Terminal output supports Live updates."""
        console = Console(force_terminal=True)
        caps = detect_terminal_capabilities(console)
        assert caps["supports_live"] is True
        assert caps["is_terminal"] is True


class TestFormatSimpleProgress:
    """Tests for simple text progress formatting (US-008)."""

    def test_zero_progress(self):
        """0% progress shows empty bar."""
        result = format_simple_progress("GPT-4o", 0.0)
        assert "Model: GPT-4o" in result
        assert "[" in result
        assert "]" in result
        assert "0%" in result
        # Bar should be all spaces
        bar_content = result.split("[")[1].split("]")[0]
        assert bar_content.strip() == ""

    def test_50_percent_progress(self):
        """50% progress shows half-filled bar."""
        result = format_simple_progress("GPT-4o", 0.5)
        assert "Model: GPT-4o" in result
        assert "50%" in result
        # Check bar has some fill characters
        bar_content = result.split("[")[1].split("]")[0]
        assert "=" in bar_content or ">" in bar_content

    def test_100_percent_progress(self):
        """100% progress shows full bar."""
        result = format_simple_progress("GPT-4o", 1.0)
        assert "Model: GPT-4o" in result
        assert "100%" in result
        # Bar should be all equals
        bar_content = result.split("[")[1].split("]")[0]
        assert bar_content == "=" * FALLBACK_BAR_WIDTH

    def test_progress_clamped_above_1(self):
        """Progress > 1.0 is clamped to 1.0."""
        result = format_simple_progress("Model", 1.5)
        assert "100%" in result

    def test_progress_clamped_below_0(self):
        """Progress < 0.0 is clamped to 0.0."""
        result = format_simple_progress("Model", -0.5)
        assert "0%" in result

    def test_custom_bar_width(self):
        """Can specify custom bar width."""
        result = format_simple_progress("Model", 1.0, width=10)
        bar_content = result.split("[")[1].split("]")[0]
        assert len(bar_content) == 10

    def test_percentage_format_consistent(self):
        """Percentage format is consistent width."""
        result_0 = format_simple_progress("M", 0.0)
        result_50 = format_simple_progress("M", 0.5)
        result_100 = format_simple_progress("M", 1.0)
        # All should have same length with consistent percent formatting
        assert "  0%" in result_0
        assert " 50%" in result_50
        assert "100%" in result_100


class TestRuneScapeLoaderFallbackMode:
    """Tests for fallback mode functionality (US-008)."""

    def test_use_fallback_property_exists(self):
        """RuneScapeLoader has use_fallback property."""
        loader = RuneScapeLoader("Model", total_steps=100)
        assert hasattr(loader, "use_fallback")

    def test_fallback_enabled_when_no_color(self):
        """Fallback mode is enabled when NO_COLOR is set."""
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            console = Console(force_terminal=True)
            loader = RuneScapeLoader("Model", total_steps=100, console=console)
            assert loader.use_fallback is True

    def test_fallback_enabled_when_piped(self):
        """Fallback mode is enabled when output is piped."""
        console = Console(force_terminal=False)
        loader = RuneScapeLoader("Model", total_steps=100, console=console)
        assert loader.use_fallback is True

    def test_fallback_disabled_in_color_terminal(self):
        """Fallback mode is disabled in color terminal."""
        env = os.environ.copy()
        env.pop("NO_COLOR", None)
        with patch.dict(os.environ, env, clear=True):
            console = Console(force_terminal=True)
            loader = RuneScapeLoader("Model", total_steps=100, console=console)
            assert loader.use_fallback is False

    def test_fallback_start_does_not_create_live(self):
        """In fallback mode, start() does not create Live display."""
        console = Console(force_terminal=False)
        loader = RuneScapeLoader("Model", total_steps=100, console=console)
        assert loader.use_fallback is True
        loader.start()
        assert loader._live is None
        assert loader._animation_thread is None
        loader.stop()

    def test_fallback_start_sets_running(self):
        """In fallback mode, start() still sets _running to True."""
        console = Console(force_terminal=False)
        loader = RuneScapeLoader("Model", total_steps=100, console=console)
        loader.start()
        assert loader._running is True
        loader.stop()

    def test_fallback_update_prints_progress(self, capsys):
        """In fallback mode, update() prints progress to stdout."""
        console = Console(force_terminal=False)
        loader = RuneScapeLoader("GPT-4o", total_steps=100, console=console)
        loader.start()
        loader.update(50)
        loader.stop()
        captured = capsys.readouterr()
        assert "GPT-4o" in captured.out
        assert "50%" in captured.out

    def test_fallback_complete_prints_status(self, capsys):
        """In fallback mode, complete() prints completion status."""
        console = Console(force_terminal=False)
        loader = RuneScapeLoader("GPT-4o", total_steps=100, console=console)
        loader.start()
        loader.update(100)
        loader.complete(success=True)
        loader.stop()
        captured = capsys.readouterr()
        assert "DONE" in captured.out

    def test_fallback_complete_failure_prints_failed(self, capsys):
        """In fallback mode, failure shows FAILED status."""
        console = Console(force_terminal=False)
        loader = RuneScapeLoader("GPT-4o", total_steps=100, console=console)
        loader.start()
        loader.update(50)
        loader.complete(success=False)
        loader.stop()
        captured = capsys.readouterr()
        assert "FAILED" in captured.out

    def test_fallback_progress_percentage_correct(self, capsys):
        """Fallback mode shows correct progress percentage."""
        console = Console(force_terminal=False)
        loader = RuneScapeLoader("Model", total_steps=100, console=console)
        loader.start()
        loader.update(25)
        loader.stop()
        captured = capsys.readouterr()
        assert "25%" in captured.out

    def test_fallback_context_manager_works(self, capsys):
        """Context manager works in fallback mode."""
        console = Console(force_terminal=False)
        with RuneScapeLoader("Model", total_steps=100, console=console) as loader:
            loader.update(50)
        captured = capsys.readouterr()
        assert "Model" in captured.out


class TestRuneScapeLoaderFallbackExamples:
    """Tests demonstrating fallback mode examples from acceptance criteria."""

    def test_example_no_color_env_uses_fallback(self):
        """Example: running with NO_COLOR=1 uses fallback."""
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            console = Console(force_terminal=True)
            loader = RuneScapeLoader("GPT-4o", total_steps=100, console=console)
            assert loader.use_fallback is True

    def test_example_piped_output_uses_simple_progress(self, capsys):
        """Example: piping output to file uses simple text progress."""
        console = Console(force_terminal=False)
        loader = RuneScapeLoader("Claude-3", total_steps=100, console=console)
        loader.start()
        loader.update(75)
        loader.stop()
        captured = capsys.readouterr()
        # Should have simple format: 'Model: Claude-3 [=====>    ] 75%'
        assert "Model: Claude-3" in captured.out
        assert "[" in captured.out
        assert "]" in captured.out
        assert "75%" in captured.out

    def test_negative_case_fallback_shows_correct_percentage(self, capsys):
        """Negative case: fallback mode still shows progress percentage correctly."""
        console = Console(force_terminal=False)
        loader = RuneScapeLoader("Model", total_steps=200, console=console)
        loader.start()
        # Test various percentages
        loader.update(0)
        loader.update(100)  # 50%
        loader.update(200)  # 100%
        loader.stop()
        captured = capsys.readouterr()
        # Should show correct percentages
        assert "0%" in captured.out or "50%" in captured.out or "100%" in captured.out


class TestRuneScapeLoaderFallbackProgressTracking:
    """Tests for fallback progress update throttling."""

    def test_fallback_throttles_updates(self, capsys):
        """Fallback mode throttles updates to avoid excessive output."""
        console = Console(force_terminal=False)
        loader = RuneScapeLoader("Model", total_steps=100, console=console)
        loader.start()
        # Many small updates should be throttled
        for i in range(10):
            loader.update(i)
        loader.stop()
        captured = capsys.readouterr()
        # Should not have 10 lines of output due to throttling
        lines = [line for line in captured.out.strip().split("\n") if line]
        assert len(lines) < 10

    def test_set_model_resets_progress_tracking(self, capsys):
        """set_model() resets progress tracking for new model."""
        console = Console(force_terminal=False)
        loader = RuneScapeLoader("Model1", total_steps=100, console=console)
        loader.start()
        loader.update(50)
        loader.set_model("Model2")
        loader.update(10)  # Should print since it's a new model
        loader.stop()
        captured = capsys.readouterr()
        assert "Model1" in captured.out
        assert "Model2" in captured.out


class TestRuneScapeLoaderFourLayerLayout:
    """Tests for 4-layer layout (US-006)."""

    def test_loader_has_counters(self):
        """Loader has success_count, failure_count, total_cost attributes."""
        loader = RuneScapeLoader("Model", total_steps=100)
        assert hasattr(loader, "_success_count")
        assert hasattr(loader, "_failure_count")
        assert hasattr(loader, "_total_cost")
        assert loader._success_count == 0
        assert loader._failure_count == 0
        assert loader._total_cost == 0.0


class TestRuneScapeLoaderCompleteCounters:
    """Tests for complete() counter increments (US-007)."""

    def test_complete_success_increments_success_count(self):
        """complete(success=True) increments success_count by 1."""
        loader = RuneScapeLoader("Model", total_steps=100)
        assert loader._success_count == 0

        with patch("asciibench.common.loader.time.sleep"):
            loader.complete(success=True)

        assert loader._success_count == 1
        assert loader._failure_count == 0

    def test_complete_failure_increments_failure_count(self):
        """complete(success=False) increments failure_count by 1."""
        loader = RuneScapeLoader("Model", total_steps=100)
        assert loader._failure_count == 0

        with patch("asciibench.common.loader.time.sleep"):
            loader.complete(success=False)

        assert loader._success_count == 0
        assert loader._failure_count == 1

    def test_complete_with_cost_increments_total_cost(self):
        """complete(success=True, cost=0.005) increments total_cost by cost."""
        loader = RuneScapeLoader("Model", total_steps=100)
        assert loader._total_cost == 0.0

        with patch("asciibench.common.loader.time.sleep"):
            loader.complete(success=True, cost=0.005)

        assert loader._total_cost == 0.005

    def test_complete_success_with_cost(self):
        """Example: after complete(True, 0.005), success_count increases by 1, total_cost increases by 0.005."""
        loader = RuneScapeLoader("Model", total_steps=100)
        initial_success = loader._success_count
        initial_cost = loader._total_cost

        with patch("asciibench.common.loader.time.sleep"):
            loader.complete(success=True, cost=0.005)

        assert loader._success_count == initial_success + 1
        assert loader._total_cost == initial_cost + 0.005

    def test_complete_failure_no_cost(self):
        """Negative case: complete(False, 0.0) increments failure_count, cost unchanged."""
        loader = RuneScapeLoader("Model", total_steps=100)
        initial_failure = loader._failure_count
        initial_cost = loader._total_cost

        with patch("asciibench.common.loader.time.sleep"):
            loader.complete(success=False, cost=0.0)

        assert loader._failure_count == initial_failure + 1
        assert loader._total_cost == initial_cost

    def test_complete_multiple_successes_accumulate(self):
        """Multiple complete(success=True) calls accumulate success_count."""
        loader = RuneScapeLoader("Model", total_steps=100)

        with patch("asciibench.common.loader.time.sleep"):
            loader.complete(success=True)
            loader.complete(success=True)
            loader.complete(success=True)

        assert loader._success_count == 3
        assert loader._failure_count == 0

    def test_complete_multiple_failures_accumulate(self):
        """Multiple complete(success=False) calls accumulate failure_count."""
        loader = RuneScapeLoader("Model", total_steps=100)

        with patch("asciibench.common.loader.time.sleep"):
            loader.complete(success=False)
            loader.complete(success=False)

        assert loader._success_count == 0
        assert loader._failure_count == 2

    def test_complete_mixed_results_accumulate_correctly(self):
        """Mixed success/failure calls accumulate counters correctly."""
        loader = RuneScapeLoader("Model", total_steps=100)

        with patch("asciibench.common.loader.time.sleep"):
            loader.complete(success=True, cost=0.001)
            loader.complete(success=False, cost=0.0)
            loader.complete(success=True, cost=0.002)
            loader.complete(success=False, cost=0.0)

        assert loader._success_count == 2
        assert loader._failure_count == 2
        assert loader._total_cost == 0.003

    def test_complete_preserves_counters_after_reset(self):
        """After flash and reset, counters remain updated for next model."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(100)

        with patch("asciibench.common.loader.time.sleep"):
            loader.complete(success=True, cost=0.005)

        # Progress is reset
        assert loader.progress == 0.0
        assert loader.is_complete is False

        # But counters are preserved
        assert loader._success_count == 1
        assert loader._total_cost == 0.005

    def test_complete_cost_default_zero(self):
        """complete(success) without cost parameter defaults cost to 0.0."""
        loader = RuneScapeLoader("Model", total_steps=100)
        initial_cost = loader._total_cost

        with patch("asciibench.common.loader.time.sleep"):
            loader.complete(success=True)

        assert loader._total_cost == initial_cost

    def test_complete_fallback_mode_increments_counters(self):
        """In fallback mode, complete() still increments counters."""
        console = Console(force_terminal=False)
        loader = RuneScapeLoader("Model", total_steps=100, console=console)
        loader.start()

        with patch("asciibench.common.loader.time.sleep"):
            loader.complete(success=True, cost=0.005)

        loader.stop()

        assert loader._success_count == 1
        assert loader._total_cost == 0.005

    def test_complete_flash_behavior_preserved_success(self):
        """Flash behavior preserved: green for success, ~300ms duration."""
        # Force terminal mode to use the flash code path (not fallback)
        env = os.environ.copy()
        env.pop("NO_COLOR", None)
        with patch.dict(os.environ, env, clear=True):
            console = Console(force_terminal=True)
            loader = RuneScapeLoader("Model", total_steps=100, console=console)
            loader.update(100)

            flash_duration_ms = None

            def capture_flash_duration(duration):
                nonlocal flash_duration_ms
                flash_duration_ms = duration

            with patch("asciibench.common.loader.time.sleep", side_effect=capture_flash_duration):
                loader.complete(success=True)

            # Verify flash duration is correct
            assert flash_duration_ms == FLASH_DURATION_MS / 1000.0

            # Verify counters were incremented
            assert loader._success_count == 1

    def test_complete_flash_behavior_preserved_failure(self):
        """Flash behavior preserved: red for failure, ~300ms duration."""
        # Force terminal mode to use the flash code path (not fallback)
        env = os.environ.copy()
        env.pop("NO_COLOR", None)
        with patch.dict(os.environ, env, clear=True):
            console = Console(force_terminal=True)
            loader = RuneScapeLoader("Model", total_steps=100, console=console)
            loader.update(100)

            flash_duration_ms = None
            captured_color = []

            def capture_flash(duration):
                nonlocal flash_duration_ms
                flash_duration_ms = duration
                with loader._lock:
                    if loader._flashing:
                        captured_color.append(loader._flash_color)

            with patch("asciibench.common.loader.time.sleep", side_effect=capture_flash):
                loader.complete(success=False)

            # Verify flash duration is correct
            assert flash_duration_ms == FLASH_DURATION_MS / 1000.0

            # Verify red color was used during flash
            assert FAILURE_COLOR in captured_color

            # Verify counters were incremented
            assert loader._failure_count == 1

    def test_counters_appear_in_status_line(self):
        """Counters appear in status line after complete()."""
        loader = RuneScapeLoader("Model", total_steps=100)

        with patch("asciibench.common.loader.time.sleep"):
            loader.complete(success=True, cost=0.005)
            loader.complete(success=False, cost=0.0)

        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            result_str = str(result)

            # Status line should show the updated totals
            lines = result_str.split("\n")
            status_line = lines[0]

            assert "✓ 1" in status_line
            assert "✗ 1" in status_line
            assert "$0.0050" in status_line

    def test_loader_has_domino_state(self):
        """Loader has domino_state attribute."""
        loader = RuneScapeLoader("Model", total_steps=100)
        assert hasattr(loader, "_domino_state")
        assert hasattr(loader, "_domino_width")

    def test_render_frame_returns_four_layers(self):
        """_render_frame returns 4-layer layout."""
        loader = RuneScapeLoader("TestModel", total_steps=100)
        loader.update(50)  # 50% progress
        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            result_str = str(result)
            # Should have 3 newlines (4 layers)
            assert result_str.count("\n") == 3

    def test_render_frame_status_line_shows_totals(self):
        """Status line shows success, failure, cost totals."""
        loader = RuneScapeLoader("Model", total_steps=100)
        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            result_str = str(result)
            # Should have checkmark, cross, and dollar sign
            assert "✓" in result_str
            assert "✗" in result_str
            assert "$" in result_str
            # Should show zeros initially
            assert "0" in result_str

    def test_render_frame_model_name_present(self):
        """Model name appears in second layer."""
        loader = RuneScapeLoader("TestModel", total_steps=100)
        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            result_str = str(result)
            assert "TestModel" in result_str

    def test_render_frame_progress_bar_present(self):
        """Progress bar appears with filled and empty characters."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(50)
        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            result_str = str(result)
            # Should have filled and empty progress characters
            assert "━" in result_str
            assert "─" in result_str

    def test_render_frame_domino_row_present(self):
        """Domino row appears in fourth layer."""
        loader = RuneScapeLoader("Model", total_steps=100)
        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            result_str = str(result)
            # Should have domino characters
            assert "▌" in result_str or "▞" in result_str or "▚" in result_str or "▄" in result_str

    def test_domino_animation_updates_each_frame(self):
        """Domino animation updates on each frame."""
        loader = RuneScapeLoader("Model", total_steps=100)
        with patch.object(loader, "_get_terminal_width", return_value=80):
            initial_state = loader._domino_state
            loader._render_frame()
            after_one_frame = loader._domino_state
            loader._render_frame()
            after_two_frames = loader._domino_state

            # Domino state should change after each render
            assert initial_state != after_one_frame
            assert after_one_frame != after_two_frames

    def test_fifty_percent_progress_half_filled_bar(self):
        """At 50% progress, progress bar is half-filled."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(50)
        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            result_str = str(result)
            # Find the progress bar line (should have ━ and ─)
            lines = result_str.split("\n")
            progress_bar = None
            for line in lines:
                if "━" in line or "─" in line:
                    progress_bar = line
                    break

            assert progress_bar is not None
            # Count filled vs empty characters
            filled_count = progress_bar.count("━")
            empty_count = progress_bar.count("─")
            # At 50%, should have approximately equal counts
            assert abs(filled_count - empty_count) <= 2

    def test_zero_progress_empty_bar(self):
        """At 0% progress, progress bar is empty but dominos still animate."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(0)
        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            result_str = str(result)
            lines = result_str.split("\n")

            # Find progress bar line
            progress_bar = None
            for line in lines:
                if "━" in line or "─" in line:
                    progress_bar = line
                    break

            assert progress_bar is not None
            # Progress bar should be all empty (─) or minimal filled
            assert progress_bar.count("━") <= 2

            # Domino row should still be present
            has_domino = any(
                "▌" in line or "▞" in line or "▚" in line or "▄" in line for line in lines
            )
            assert has_domino

    def test_full_progress_completely_filled_bar(self):
        """At 100% progress, progress bar is completely filled."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(100)
        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            result_str = str(result)
            lines = result_str.split("\n")

            # Find progress bar line
            progress_bar = None
            for line in lines:
                if "━" in line or "─" in line:
                    progress_bar = line
                    break

            assert progress_bar is not None
            # Progress bar should be all filled (━) or minimal empty
            assert progress_bar.count("─") <= 2

    def test_status_line_format(self):
        """Status line has correct format: ✓ {success} | ✗ {failed} | ${cost}"""
        loader = RuneScapeLoader("Model", total_steps=100)
        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            result_str = str(result)
            lines = result_str.split("\n")
            status_line = lines[0]

            # Check format elements
            assert "✓" in status_line
            assert "✗" in status_line
            assert "|" in status_line
            assert "$" in status_line
            assert "0.0000" in status_line  # Initial cost formatted to 4 decimals

    def test_integration_example_four_layers(self):
        """Example: at 50% progress, all 4 layers are visible and correct."""
        loader = RuneScapeLoader("TestModel", total_steps=100)
        loader.update(50)

        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            result_str = str(result)
            lines = result_str.split("\n")

            # Should have 4 layers
            assert len(lines) == 4

            # Layer 1: Status line
            assert "✓" in lines[0]
            assert "✗" in lines[0]
            assert "$" in lines[0]

            # Layer 2: Model name
            assert "TestModel" in lines[1]

            # Layer 3: Progress bar
            assert "━" in lines[2]
            assert "─" in lines[2]

            # Layer 4: Domino row
            domino_chars = {"▌", "▞", "▚", "▄"}
            has_domino = any(char in lines[3] for char in domino_chars)
            assert has_domino
