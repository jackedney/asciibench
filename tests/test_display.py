from asciibench.common.display import (
    get_console,
    get_theme,
    print_banner,
    create_generation_progress,
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
