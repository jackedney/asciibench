from rich.console import Console
from rich.live import Live
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

    if _console is None or force_terminal is not None:
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


class LiveStatsDisplay:
    def __init__(self, total: int = 0, valid: int = 0, invalid: int = 0):
        self.total = total
        self.valid = valid
        self.invalid = invalid

    def update(
        self, total: int | None = None, valid: int | None = None, invalid: int | None = None
    ):
        if total is not None:
            self.total = total
        if valid is not None:
            self.valid = valid
        if invalid is not None:
            self.invalid = invalid

    def render(self) -> Panel:
        stats_text = Text.assemble(
            ("Generated: ", "info"),
            (f"{self.total}", "accent bold"),
            (" | ", "default"),
            ("Valid: ", "success"),
            (f"{self.valid}", "success bold"),
            (" | ", "default"),
            ("Invalid: ", "error"),
            (f"{self.invalid}", "error bold"),
        )

        panel = Panel(
            stats_text,
            border_style=GOLD,
            padding=(0, 2),
        )
        return panel


def create_live_stats():
    console = get_console()
    display = LiveStatsDisplay()

    if console.is_terminal:
        live = Live(
            display.render(),
            console=console,
            refresh_per_second=1,
        )
    else:
        live = None

    return display, live


def update_live_stats(
    display: LiveStatsDisplay,
    live: Live | None,
    total: int | None = None,
    valid: int | None = None,
    invalid: int | None = None,
):
    display.update(total=total, valid=valid, invalid=invalid)

    if live is not None:
        live.update(display.render())
    else:
        console = get_console()
        console.print(
            f"Generated: {display.total} | Valid: {display.valid} | Invalid: {display.invalid}"
        )


def success_badge() -> Text:
    return Text("[SUCCESS]", style="success")


def failed_badge() -> Text:
    return Text("[FAILED]", style="error")


def pending_badge() -> Text:
    return Text("[PENDING]", style=f"bold {GOLD}")
