
from rich.console import Console
from rich.theme import Theme

LILAC = "#C8A2C8"
GOLD = "#FFD700"


_neobrutalist_theme = Theme(
    {
        "primary": LILAC,
        "accent": GOLD,
        "bold": "bold",
        "success": "green bold",
        "error": "red bold",
        "warning": "yellow bold",
        "info": "blue bold",
    }
)


_console: Console | None = None


def get_theme() -> Theme:
    return _neobrutalist_theme


def get_console(force_terminal: bool | None = None) -> Console:
    global _console

    if _console is None:
        _console = Console(
            theme=_neobrutalist_theme,
            force_terminal=force_terminal,
            legacy_windows=False,
        )

    return _console
