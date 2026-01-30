from rich.console import Console
from rich.panel import Panel
from rich.text import Text
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


def print_banner() -> None:
    console = get_console()

    ascii_art = [
        "    _    _  ____   ____  ____  _______",
        "   / \\  | |/ __ \\ / __ \\/ __ \\/ ____/",
        "  / _ \\ | | |  | | |  | | |  | |     ",
        " / ___ \\| | |  | | |  | | |  | |___ ",
        "/_/   \\_\\_|_|  |_|_|  |_|_|  |_|____/",
        "  _       ____  _______ ____   _____ ",
        " | |     / __ \\/ ____/ __ \\ / ___/ ",
        " | |    / /_/ / __/ / /_/ / \\__ \\ ",
        " | |___/ _, _/ /___/ _, _/ ___/ / ",
        " |_____/_/ |_/_____/_/ |_| /____/  ",
    ]

    banner_text = Text()
    for line in ascii_art:
        banner_text.append(line + "\n", style=f"bold {LILAC}")

    panel = Panel(
        banner_text,
        border_style=GOLD,
        padding=(0, 1),
    )

    console.print(panel)
