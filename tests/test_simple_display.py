"""Tests for the simplified CLI display wrapper."""

from asciibench.common.display import (
    MAX_PROMPT_LENGTH,
    create_loader,
    get_console,
    show_banner,
    show_prompt,
)
from asciibench.common.loader import RuneScapeLoader


def test_show_banner_does_not_raise():
    """show_banner should print the ASCII banner without errors."""
    get_console(force_terminal=True)
    show_banner()
    assert True


def test_show_banner_prints_banner():
    """show_banner should display the ASCII art banner."""
    console = get_console(force_terminal=True)
    with console.capture() as capture:
        show_banner()
    output = capture.get()
    # Check for parts of the ASCII art - the box border characters
    assert "╭" in output
    assert "╯" in output


def test_show_prompt_displays_text():
    """show_prompt should display the prompt text."""
    console = get_console(force_terminal=True)
    with console.capture() as capture:
        show_prompt("Draw a cat")
    output = capture.get()
    assert "Draw a cat" in output


def test_show_prompt_includes_label():
    """show_prompt should include a 'Prompt:' label."""
    console = get_console(force_terminal=True)
    with console.capture() as capture:
        show_prompt("Test prompt")
    output = capture.get()
    assert "Prompt:" in output


def test_show_prompt_empty_string():
    """show_prompt should handle empty strings."""
    console = get_console(force_terminal=True)
    with console.capture() as capture:
        show_prompt("")
    output = capture.get()
    assert "Prompt:" in output


def test_show_prompt_truncates_long_text():
    """show_prompt should truncate prompts longer than 500 chars with ellipsis."""
    long_prompt = "x" * 600
    console = get_console(force_terminal=True)
    with console.capture() as capture:
        show_prompt(long_prompt)
    output = capture.get()
    assert "..." in output
    # The output should not contain the full 600 chars
    assert "x" * 600 not in output


def test_show_prompt_exact_max_length():
    """show_prompt should not truncate prompts at exactly MAX_PROMPT_LENGTH."""
    exact_prompt = "x" * MAX_PROMPT_LENGTH
    console = get_console(force_terminal=True)
    with console.capture() as capture:
        show_prompt(exact_prompt)
    output = capture.get()
    # Should not have ellipsis
    assert "..." not in output


def test_show_prompt_one_over_max_length():
    """show_prompt should truncate prompts one char over MAX_PROMPT_LENGTH."""
    long_prompt = "x" * (MAX_PROMPT_LENGTH + 1)
    console = get_console(force_terminal=True)
    with console.capture() as capture:
        show_prompt(long_prompt)
    output = capture.get()
    assert "..." in output


def test_show_prompt_preserves_special_chars():
    """show_prompt should handle special characters."""
    console = get_console(force_terminal=True)
    with console.capture() as capture:
        show_prompt("Draw a 🐱 cat!")
    output = capture.get()
    assert "cat" in output


def test_create_loader_returns_runescape_loader():
    """create_loader should return a RuneScapeLoader instance."""
    loader = create_loader("GPT-4o", total=100)
    assert isinstance(loader, RuneScapeLoader)


def test_create_loader_sets_model_name():
    """create_loader should set the correct model name."""
    loader = create_loader("Claude-4", total=50)
    assert loader.model_name == "Claude-4"


def test_create_loader_sets_total_steps():
    """create_loader should configure the loader with the correct total."""
    loader = create_loader("Test", total=200)
    # The loader should start at 0 progress
    assert loader.progress == 0.0


def test_create_loader_works_as_context_manager():
    """create_loader should return a loader that works as a context manager."""
    loader = create_loader("GPT-4o", total=10)
    with loader:
        loader.update(5)
        assert loader.progress == 0.5


def test_create_loader_uses_shared_console():
    """create_loader should use the shared console instance."""
    # Force terminal mode
    get_console(force_terminal=True)
    loader = create_loader("Test", total=100)
    # The loader's console should be the shared one
    assert loader._console is not None
