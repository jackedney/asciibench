from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
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


def create_generation_progress(total: int = 100):
    console = get_console()

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="primary", finished_style="success"),
        TaskProgressColumn(),
        TextColumn("[info]ETA: {task.fields[eta]}", justify="right"),
        TextColumn("|"),
        TextColumn("[accent]{task.fields[model]}", justify="right"),
        TextColumn("|"),
        TextColumn("[info]{task.fields[prompt]}", justify="right"),
        TextColumn("|"),
        TextColumn("[accent]Attempt {task.fields[attempt]}", justify="right"),
        TextColumn("|"),
        TextColumn("[success]✓{task.fields[success_count]}", justify="right"),
        TextColumn("[error]✗{task.fields[fail_count]}", justify="right"),
        console=console,
    )

    return progress
