"""Simplified CLI display wrapper for the RuneScape-style loader.

Provides a minimal display interface that shows only the ASCII banner,
prompt text, and loading bar - removing verbose tables and panels.
"""

from asciibench.common.display import get_console, print_banner
from asciibench.common.loader import RuneScapeLoader

# Maximum prompt length before truncation
MAX_PROMPT_LENGTH = 500


def show_banner() -> None:
    """Display the ASCII art banner.

    Reuses the existing banner from the display module.

    Example:
        >>> show_banner()
        # Prints the ASCII art banner to the console
    """
    print_banner()


def show_prompt(prompt_text: str) -> None:
    """Display the current prompt text above the loader.

    The prompt is displayed statically (not animated) using the existing
    theme colors. Very long prompts (>500 chars) are truncated with ellipsis.

    Args:
        prompt_text: The prompt text to display.

    Example:
        >>> show_prompt('Draw a cat')
        # Displays: "Prompt: Draw a cat" styled appropriately

        >>> show_prompt('x' * 600)
        # Displays truncated text with "..." at the end
    """
    console = get_console()

    # Truncate long prompts
    if len(prompt_text) > MAX_PROMPT_LENGTH:
        prompt_text = prompt_text[:MAX_PROMPT_LENGTH] + "..."

    console.print(f"[accent]Prompt:[/accent] [primary]{prompt_text}[/primary]")


def create_loader(model_name: str, total: int) -> RuneScapeLoader:
    """Factory function to create a RuneScapeLoader instance.

    Creates a loader configured with the shared console instance
    for consistent display.

    Args:
        model_name: The model name to display in the loader.
        total: Total number of steps for the loading process.

    Returns:
        A configured RuneScapeLoader instance.

    Example:
        >>> loader = create_loader('GPT-4o', total=100)
        >>> with loader:
        ...     for i in range(100):
        ...         loader.update(i + 1)
    """
    console = get_console()
    return RuneScapeLoader(model_name=model_name, total_steps=total, console=console)
