from asciibench.common.display import get_console, get_theme, print_banner


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
