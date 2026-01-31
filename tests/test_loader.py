"""Tests for the RuneScape-style animated loading bar component."""

import threading
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.text import Text

from asciibench.common.loader import RuneScapeLoader


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
        """At 0% progress, rendered frame is empty."""
        loader = RuneScapeLoader("Model", total_steps=100)
        result = loader._render_frame()
        assert str(result) == ""

    def test_render_frame_at_full_progress(self):
        """At 100% progress, rendered frame fills terminal width."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(100)
        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            assert len(str(result)) == 80

    def test_render_frame_at_partial_progress(self):
        """At 50% progress, rendered frame is approximately half width."""
        loader = RuneScapeLoader("Model", total_steps=100)
        loader.update(50)
        with patch.object(loader, "_get_terminal_width", return_value=80):
            result = loader._render_frame()
            assert len(str(result)) == 40

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
            # At 30% progress with 100 width, should have 30 chars
            assert len(str(result)) == 30
            # Should contain model name
            assert "Claude-4o" in str(result)

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
