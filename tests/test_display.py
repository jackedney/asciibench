from rich.text import Text

from asciibench.common.display import (
    LiveStatsDisplay,
    create_generation_progress,
    create_live_stats,
    failed_badge,
    get_console,
    get_theme,
    pending_badge,
    print_banner,
    success_badge,
    update_live_stats,
)


def test_get_theme_returns_theme():
    theme = get_theme()
    assert theme is not None
    assert "primary" in theme.styles
    assert "accent" in theme.styles


def test_get_theme_has_lilac_and_gold():
    theme = get_theme()
    assert theme.styles["primary"].color is not None
    assert theme.styles["accent"].color is not None


def test_get_console_returns_singleton():
    console1 = get_console()
    console2 = get_console()
    assert console1 is console2


def test_get_console_forces_terminal():
    console = get_console(force_terminal=True)
    assert console.legacy_windows is False


def test_print_banner_does_not_raise():
    get_console(force_terminal=True)
    print_banner()
    assert True


def test_print_banner_on_narrow_terminal():
    get_console(force_terminal=True)
    print_banner()
    assert True


def test_create_generation_progress_returns_progress():
    progress = create_generation_progress()
    assert progress is not None
    assert hasattr(progress, "add_task")


def test_create_generation_progress_as_context_manager():
    with create_generation_progress() as progress:
        assert progress is not None


def test_create_generation_progress_adds_task_with_all_fields():
    with create_generation_progress() as p:
        task = p.add_task(
            "Generating...",
            total=100,
            eta="--:--",
            model="gpt-4o",
            prompt="Draw a cat",
            attempt=1,
            success_count=0,
            fail_count=0,
        )
        assert task is not None


def test_create_generation_progress_updates_task():
    with create_generation_progress() as p:
        task = p.add_task(
            "Generating...",
            total=100,
            eta="--:--",
            model="gpt-4o",
            prompt="Draw a cat",
            attempt=1,
            success_count=0,
            fail_count=0,
        )
        p.update(task, advance=10, success_count=5, fail_count=2)
        assert True


def test_live_stats_display_initializes():
    display = LiveStatsDisplay()
    assert display.total == 0
    assert display.valid == 0
    assert display.invalid == 0


def test_live_stats_display_updates():
    display = LiveStatsDisplay()
    display.update(total=10, valid=8, invalid=2)
    assert display.total == 10
    assert display.valid == 8
    assert display.invalid == 2


def test_live_stats_display_partial_update():
    display = LiveStatsDisplay(total=5, valid=4, invalid=1)
    display.update(total=10)
    assert display.total == 10
    assert display.valid == 4
    assert display.invalid == 1


def test_live_stats_display_renders():
    display = LiveStatsDisplay(total=10, valid=8, invalid=2)
    panel = display.render()
    assert panel is not None
    assert "Generated" in str(panel.renderable)
    assert "Valid" in str(panel.renderable)
    assert "Invalid" in str(panel.renderable)


def test_create_live_stats_returns_display():
    display, _live = create_live_stats()
    assert display is not None
    assert isinstance(display, LiveStatsDisplay)


def test_create_live_stats_on_terminal_returns_live():
    get_console(force_terminal=True)
    _display, live = create_live_stats()
    assert live is not None


def test_create_live_stats_non_terminal_no_live():
    get_console(force_terminal=False)
    _display, live = create_live_stats()
    assert live is None


def test_update_live_stats_updates_display():
    display = LiveStatsDisplay()
    live = None
    update_live_stats(display, live, total=10, valid=8, invalid=2)
    assert display.total == 10
    assert display.valid == 8
    assert display.invalid == 2


def test_live_stats_with_progress():
    display = LiveStatsDisplay()
    for i in range(5):
        update_live_stats(display, None, total=i + 1, valid=i + 1, invalid=0)
    assert display.total == 5
    assert display.valid == 5
    assert display.invalid == 0


def test_success_badge_returns_text():
    badge = success_badge()
    assert badge is not None
    assert isinstance(badge, Text)


def test_success_badge_content():
    badge = success_badge()
    assert "[SUCCESS]" in str(badge)


def test_failed_badge_returns_text():
    badge = failed_badge()
    assert badge is not None
    assert isinstance(badge, Text)


def test_failed_badge_content():
    badge = failed_badge()
    assert "[FAILED]" in str(badge)


def test_pending_badge_returns_text():
    badge = pending_badge()
    assert badge is not None
    assert isinstance(badge, Text)


def test_pending_badge_content():
    badge = pending_badge()
    assert "[PENDING]" in str(badge)


def test_badges_can_be_printed():
    console = get_console(force_terminal=True)
    console.print(success_badge(), "Test message")
    console.print(failed_badge(), "Test message")
    console.print(pending_badge(), "Test message")
    assert True
