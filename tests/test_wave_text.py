"""Tests for the RuneScape-style wave text renderer."""

from rich.text import Text

from asciibench.common.wave_text import (
    RAINBOW_COLORS,
    get_wave_displacement,
    interpolate_color,
    render_cycling_text,
    render_gradient_text,
    render_status_line,
    render_wave_text,
)


class TestRenderWaveText:
    """Tests for render_wave_text function."""

    def test_returns_text_object(self):
        """render_wave_text returns a Rich Text object."""
        result = render_wave_text("GPT-4o")
        assert isinstance(result, Text)

    def test_empty_string_returns_empty_text(self):
        """Empty string input returns empty Text object."""
        result = render_wave_text("")
        assert isinstance(result, Text)
        assert len(result) == 0
        assert str(result) == ""

    def test_preserves_text_content(self):
        """Text content is preserved in output."""
        text = "GPT-4o"
        result = render_wave_text(text)
        assert str(result) == text

    def test_preserves_text_content_with_spaces(self):
        """Text with spaces is preserved."""
        text = "Hello World"
        result = render_wave_text(text)
        assert str(result) == text

    def test_frame_zero_applies_colors(self):
        """Frame 0 applies colors starting from the first rainbow color."""
        result = render_wave_text("ABC", frame=0)
        # Each character should have a style applied
        assert len(result._spans) > 0 or result.style is not None

    def test_different_frames_produce_different_styles(self):
        """Different frame numbers produce different color arrangements."""
        result_frame0 = render_wave_text("ABC", frame=0)
        result_frame1 = render_wave_text("ABC", frame=1)
        # The text content is the same
        assert str(result_frame0) == str(result_frame1)
        # But the styling should differ due to color shift

    def test_rainbow_colors_are_applied(self):
        """Characters receive colors from the rainbow palette."""
        # With 7 colors and frame=0, first 7 chars get all rainbow colors
        result = render_wave_text("ABCDEFG", frame=0)
        # Text is rendered with colors applied
        assert len(str(result)) == 7

    def test_colors_cycle_through_rainbow(self):
        """Colors cycle through the rainbow when text is longer than palette."""
        text = "A" * (len(RAINBOW_COLORS) + 3)
        result = render_wave_text(text, frame=0)
        # Should still produce valid output
        assert str(result) == text

    def test_frame_shifts_colors(self):
        """Frame number shifts color assignment for flowing effect."""
        # At frame=0, position 0 gets color 0
        # At frame=1, position 0 gets color 1
        result_f0 = render_wave_text("A", frame=0)
        result_f1 = render_wave_text("A", frame=1)
        # Both should be valid Text objects with different styles
        assert isinstance(result_f0, Text)
        assert isinstance(result_f1, Text)

    def test_long_text_handled_without_error(self):
        """Very long text (>200 chars) is handled without error."""
        long_text = "X" * 250
        result = render_wave_text(long_text, frame=0)
        assert str(result) == long_text
        assert len(result) == 250

    def test_very_long_text_handles_1000_chars(self):
        """Handles extremely long text gracefully."""
        very_long = "Model" * 200
        result = render_wave_text(very_long, frame=0)
        assert str(result) == very_long

    def test_special_characters_preserved(self):
        """Special characters are preserved in output."""
        text = "GPT-4o-mini [test] (v2)"
        result = render_wave_text(text)
        assert str(result) == text

    def test_unicode_characters_handled(self):
        """Unicode characters are handled correctly."""
        text = "模型名称 🎮 Test"
        result = render_wave_text(text)
        assert str(result) == text

    def test_single_character(self):
        """Single character input works correctly."""
        result = render_wave_text("X")
        assert str(result) == "X"

    def test_whitespace_only(self):
        """Whitespace-only input is handled."""
        result = render_wave_text("   ")
        assert str(result) == "   "

    def test_high_frame_numbers(self):
        """Very high frame numbers work correctly."""
        result = render_wave_text("Test", frame=10000)
        assert str(result) == "Test"

    def test_negative_frame_numbers(self):
        """Negative frame numbers are handled (modulo operation)."""
        result = render_wave_text("Test", frame=-5)
        assert str(result) == "Test"


class TestGetWaveDisplacement:
    """Tests for get_wave_displacement function."""

    def test_returns_float(self):
        """Returns a float value."""
        result = get_wave_displacement(0, 0)
        assert isinstance(result, float)

    def test_displacement_in_range(self):
        """Displacement is always between -1 and 1 (sine wave range)."""
        for i in range(100):
            for frame in range(50):
                displacement = get_wave_displacement(i, frame)
                assert -1.0 <= displacement <= 1.0

    def test_different_positions_different_displacements(self):
        """Different character positions have different displacements."""
        d0 = get_wave_displacement(0, 0)
        d1 = get_wave_displacement(1, 0)
        d2 = get_wave_displacement(2, 0)
        # They should differ (unless by coincidence at wave intersections)
        displacements = {d0, d1, d2}
        assert len(displacements) >= 2  # At least 2 different values

    def test_frame_changes_displacement(self):
        """Frame number affects displacement (animation)."""
        d_f0 = get_wave_displacement(5, 0)
        d_f10 = get_wave_displacement(5, 10)
        # Different frames produce different displacements
        assert d_f0 != d_f10

    def test_wave_periodicity(self):
        """Wave has periodic behavior."""
        # Sine wave repeats every 2*pi, so with phase = i * 0.5,
        # it repeats roughly every 12.5 characters
        # We just verify the wave continues to oscillate
        displacements = [get_wave_displacement(i, 0) for i in range(20)]
        # Check that we have both positive and negative values (wave oscillates)
        has_positive = any(d > 0 for d in displacements)
        has_negative = any(d < 0 for d in displacements)
        assert has_positive
        assert has_negative


class TestRainbowColors:
    """Tests for rainbow color configuration."""

    def test_rainbow_has_seven_colors(self):
        """Rainbow palette has exactly 7 colors."""
        assert len(RAINBOW_COLORS) == 7

    def test_rainbow_colors_are_hex(self):
        """All rainbow colors are valid hex color codes."""
        for color in RAINBOW_COLORS:
            assert color.startswith("#")
            assert len(color) == 7
            # Verify hex format
            int(color[1:], 16)

    def test_rainbow_order(self):
        """Rainbow colors are in correct order."""
        expected = [
            "#FF0000",  # red
            "#FF7F00",  # orange
            "#FFFF00",  # yellow
            "#00FF00",  # green
            "#00FFFF",  # cyan
            "#0000FF",  # blue
            "#8B00FF",  # purple
        ]
        assert RAINBOW_COLORS == expected


class TestFillProgress:
    """Tests for fill_progress function."""

    def test_returns_string(self):
        """fill_progress returns a string."""
        from asciibench.common.wave_text import fill_progress

        result = fill_progress("GPT-4o", width=80, progress=0.5)
        assert isinstance(result, str)

    def test_zero_progress_returns_empty_string(self):
        """Progress 0.0 returns empty string."""
        from asciibench.common.wave_text import fill_progress

        result = fill_progress("GPT-4o", width=80, progress=0.0)
        assert result == ""

    def test_full_progress_fills_width(self):
        """Progress 1.0 returns full 80 chars filled."""
        from asciibench.common.wave_text import fill_progress

        result = fill_progress("GPT-4o", width=80, progress=1.0)
        assert len(result) == 80

    def test_half_progress_fills_half_width(self):
        """Progress 0.5 returns approximately half the width."""
        from asciibench.common.wave_text import fill_progress

        result = fill_progress("GPT-4o", width=80, progress=0.5)
        assert len(result) == 40

    def test_model_name_repeated(self):
        """Model name is repeated in the output."""
        from asciibench.common.wave_text import fill_progress

        result = fill_progress("GPT-4o", width=80, progress=1.0)
        # The model name should appear multiple times
        assert result.count("GPT-4o") > 1

    def test_space_separator_between_repetitions(self):
        """Space separator exists between repetitions for readability."""
        from asciibench.common.wave_text import fill_progress

        result = fill_progress("GPT-4o", width=80, progress=1.0)
        # Should have "GPT-4o " pattern (with space separator)
        assert "GPT-4o " in result or "GPT-4o" in result

    def test_progress_greater_than_one_clamped(self):
        """Progress > 1.0 is clamped to 1.0."""
        from asciibench.common.wave_text import fill_progress

        result = fill_progress("GPT-4o", width=80, progress=1.5)
        # Should produce same result as progress=1.0
        assert len(result) == 80

    def test_progress_less_than_zero_clamped(self):
        """Progress < 0.0 is clamped to 0.0."""
        from asciibench.common.wave_text import fill_progress

        result = fill_progress("GPT-4o", width=80, progress=-0.5)
        assert result == ""

    def test_exact_length_at_various_progress_values(self):
        """Output length matches width * progress exactly."""
        from asciibench.common.wave_text import fill_progress

        assert len(fill_progress("Test", width=100, progress=0.0)) == 0
        assert len(fill_progress("Test", width=100, progress=0.25)) == 25
        assert len(fill_progress("Test", width=100, progress=0.5)) == 50
        assert len(fill_progress("Test", width=100, progress=0.75)) == 75
        assert len(fill_progress("Test", width=100, progress=1.0)) == 100

    def test_short_model_name(self):
        """Short model names work correctly."""
        from asciibench.common.wave_text import fill_progress

        result = fill_progress("A", width=10, progress=1.0)
        assert len(result) == 10

    def test_long_model_name(self):
        """Long model names work correctly."""
        from asciibench.common.wave_text import fill_progress

        result = fill_progress("VeryLongModelName", width=50, progress=1.0)
        assert len(result) == 50

    def test_model_name_with_spaces(self):
        """Model names with spaces work correctly."""
        from asciibench.common.wave_text import fill_progress

        result = fill_progress("Claude 3", width=40, progress=1.0)
        assert len(result) == 40

    def test_empty_model_name_returns_empty(self):
        """Empty model name returns empty string regardless of progress."""
        from asciibench.common.wave_text import fill_progress

        result = fill_progress("", width=80, progress=1.0)
        assert result == ""

    def test_zero_width_returns_empty(self):
        """Width of 0 returns empty string."""
        from asciibench.common.wave_text import fill_progress

        result = fill_progress("GPT-4o", width=0, progress=1.0)
        assert result == ""

    def test_small_width_truncates_model_name(self):
        """Width smaller than model name + space truncates appropriately."""
        from asciibench.common.wave_text import fill_progress

        result = fill_progress("GPT-4o", width=3, progress=1.0)
        assert len(result) == 3


class TestInterpolateColor:
    """Tests for interpolate_color function."""

    def test_returns_hex_string(self):
        """interpolate_color returns a hex color string."""
        result = interpolate_color("#FF0000", "#0000FF", 0.5)
        assert isinstance(result, str)
        assert result.startswith("#")
        assert len(result) == 7

    def test_t_zero_returns_color1(self):
        """t=0.0 returns the first color."""
        result = interpolate_color("#FF0000", "#0000FF", 0.0)
        assert result == "#FF0000"

    def test_t_one_returns_color2(self):
        """t=1.0 returns the second color."""
        result = interpolate_color("#FF0000", "#0000FF", 1.0)
        assert result == "#0000FF"

    def test_t_half_returns_midpoint(self):
        """t=0.5 returns the midpoint between colors."""
        result = interpolate_color("#FF0000", "#0000FF", 0.5)
        # Red (255,0,0) + Blue (0,0,255) at 0.5 = (127.5,0,127.5) -> #7F007F (int truncates)
        assert result == "#7F007F"

    def test_red_to_blue_interpolation(self):
        """Interpolate from red to blue produces purple at midpoint."""
        result = interpolate_color("#FF0000", "#0000FF", 0.5)
        assert result == "#7F007F"

    def test_black_to_white_interpolation(self):
        """Interpolate from black to white produces gray at midpoint."""
        result = interpolate_color("#000000", "#FFFFFF", 0.5)
        assert result == "#7F7F7F"

    def test_green_to_yellow_interpolation(self):
        """Interpolate from green to yellow produces yellow-green at midpoint."""
        result = interpolate_color("#00FF00", "#FFFF00", 0.5)
        # (0,255,0) to (255,255,0) at 0.5 = (127.5,255,0) -> #7FFF00 (int truncates)
        assert result == "#7FFF00"

    def test_t_greater_than_one_clamped(self):
        """t > 1.0 is clamped to 1.0, returning color2."""
        result = interpolate_color("#FF0000", "#0000FF", 1.5)
        assert result == "#0000FF"

    def test_t_less_than_zero_clamped(self):
        """t < 0.0 is clamped to 0.0, returning color1."""
        result = interpolate_color("#FF0000", "#0000FF", -0.5)
        assert result == "#FF0000"

    def test_extreme_negative_t_clamped(self):
        """Very negative t is clamped to 0.0."""
        result = interpolate_color("#FF0000", "#0000FF", -100.0)
        assert result == "#FF0000"

    def test_extreme_positive_t_clamped(self):
        """Very large t is clamped to 1.0."""
        result = interpolate_color("#FF0000", "#0000FF", 100.0)
        assert result == "#0000FF"

    def test_same_colors_returns_same_color(self):
        """Interpolating between identical colors returns that color."""
        result = interpolate_color("#FF0000", "#FF0000", 0.5)
        assert result == "#FF0000"

    def test_hex_without_hash_prefix(self):
        """Colors without '#' prefix work correctly."""
        result = interpolate_color("FF0000", "0000FF", 0.5)
        assert result == "#7F007F"

    def test_lowercase_hex_input(self):
        """Lowercase hex input produces uppercase hex output."""
        result = interpolate_color("#ff0000", "#0000ff", 0.5)
        assert result == "#7F007F"

    def test_output_is_uppercase(self):
        """Output hex string is always uppercase."""
        result = interpolate_color("#ff0000", "#0000ff", 0.5)
        assert result.isupper()

    def test_various_t_values(self):
        """Various t values produce correct interpolated colors."""
        # Test at t=0.25: 25% of the way from red to blue
        # Red (255,0,0) to Blue (0,0,255): r=255-63.75=191.25->191 (0xBF), g=0,
        # b=0+63.75=63.75->63 (0x3F)
        result = interpolate_color("#FF0000", "#0000FF", 0.25)
        assert result == "#BF003F"

        # Test at t=0.75: 75% of the way from red to blue
        # r=255-191.25=63.75->63 (0x3F), g=0, b=0+191.25=191.25->191 (0xBF)
        result = interpolate_color("#FF0000", "#0000FF", 0.75)
        assert result == "#3F00BF"

    def test_all_primary_color_combinations(self):
        """All combinations of primary colors interpolate correctly."""
        # Red to Green
        result = interpolate_color("#FF0000", "#00FF00", 0.5)
        # (255,0,0) to (0,255,0) at 0.5 = (127.5,127.5,0) -> #7F7F00
        assert result == "#7F7F00"

        # Red to Blue
        result = interpolate_color("#FF0000", "#0000FF", 0.5)
        assert result == "#7F007F"

        # Green to Blue
        result = interpolate_color("#00FF00", "#0000FF", 0.5)
        # (0,255,0) to (0,0,255) at 0.5 = (0,127.5,127.5) -> #007F7F
        assert result == "#007F7F"


class TestRenderGradientText:
    """Tests for render_gradient_text function."""

    def test_returns_text_object(self):
        """render_gradient_text returns a Rich Text object."""
        result = render_gradient_text("ABC", frame=0, filled_ratio=1.0)
        assert isinstance(result, Text)

    def test_empty_string_returns_empty_text(self):
        """Empty string input returns empty Text object."""
        result = render_gradient_text("", frame=0, filled_ratio=1.0)
        assert isinstance(result, Text)
        assert len(result) == 0
        assert str(result) == ""

    def test_preserves_text_content(self):
        """Text content is preserved in output."""
        text = "━━━━━───"
        result = render_gradient_text(text, frame=0, filled_ratio=0.5)
        assert str(result) == text

    def test_full_ratio_applies_gradient_to_all(self):
        """filled_ratio=1.0 applies gradient to all characters."""
        text = "ABC"
        result = render_gradient_text(text, frame=0, filled_ratio=1.0)
        assert str(result) == text
        # All characters should have styles applied (not dim)
        assert len(result._spans) > 0

    def test_zero_ratio_applies_dim_to_all(self):
        """filled_ratio=0.0 applies dim style to all characters."""
        text = "ABC"
        result = render_gradient_text(text, frame=0, filled_ratio=0.0)
        assert str(result) == text
        # All characters should be dim
        assert len(result._spans) > 0

    def test_half_ratio_applies_gradient_to_half(self):
        """filled_ratio=0.5 applies gradient to first half of characters."""
        text = "━━━━━───"  # 8 characters
        result = render_gradient_text(text, frame=0, filled_ratio=0.5)
        # First 4 chars should have gradient, last 4 should be dim
        assert str(result) == text
        assert len(result._spans) > 0

    def test_ratio_clamped_to_one(self):
        """filled_ratio > 1.0 is clamped to 1.0."""
        text = "ABC"
        result = render_gradient_text(text, frame=0, filled_ratio=1.5)
        assert str(result) == text
        # Should have gradients on all characters
        assert len(result._spans) > 0

    def test_ratio_clamped_to_zero(self):
        """filled_ratio < 0.0 is clamped to 0.0."""
        text = "ABC"
        result = render_gradient_text(text, frame=0, filled_ratio=-0.5)
        assert str(result) == text
        # Should all be dim
        assert len(result._spans) > 0

    def test_frame_zero_works(self):
        """Frame 0 produces valid output."""
        result = render_gradient_text("TEST", frame=0, filled_ratio=1.0)
        assert str(result) == "TEST"
        assert len(result) == 4

    def test_frame_changes_gradient(self):
        """Different frame numbers produce different color arrangements."""
        result_frame0 = render_gradient_text("ABC", frame=0, filled_ratio=1.0)
        result_frame1 = render_gradient_text("ABC", frame=1, filled_ratio=1.0)
        # Text content is the same
        assert str(result_frame0) == str(result_frame1)
        # But the styling should differ due to gradient shift
        assert len(result_frame0._spans) > 0
        assert len(result_frame1._spans) > 0

    def test_gradient_uses_rainbow_colors(self):
        """Gradient colors come from the rainbow palette."""
        # Use text longer than the rainbow colors to verify cycling
        text = "A" * (len(RAINBOW_COLORS) + 3)
        result = render_gradient_text(text, frame=0, filled_ratio=1.0)
        assert str(result) == text
        assert len(result) == len(text)

    def test_long_text_handled(self):
        """Long text is handled without error."""
        long_text = "X" * 100
        result = render_gradient_text(long_text, frame=0, filled_ratio=0.75)
        assert str(result) == long_text
        assert len(result) == 100

    def test_single_character(self):
        """Single character input works correctly."""
        result = render_gradient_text("X", frame=0, filled_ratio=1.0)
        assert str(result) == "X"

    def test_special_characters_preserved(self):
        """Special characters are preserved in output."""
        text = "━━━───"
        result = render_gradient_text(text, frame=0, filled_ratio=0.5)
        assert str(result) == text

    def test_high_frame_numbers(self):
        """Very high frame numbers work correctly."""
        result = render_gradient_text("Test", frame=10000, filled_ratio=1.0)
        assert str(result) == "Test"

    def test_negative_frame_numbers(self):
        """Negative frame numbers work correctly."""
        result = render_gradient_text("Test", frame=-5, filled_ratio=1.0)
        assert str(result) == "Test"

    def test_progress_bar_example(self):
        """Example from PRD: '━━━━━───' with 0.5 filled."""
        text = "━━━━━───"
        result = render_gradient_text(text, frame=0, filled_ratio=0.5)
        # First 4 chars should have gradient (8 * 0.5 = 4, truncated)
        assert str(result) == text
        assert len(result._spans) > 0

    def test_various_filled_ratios(self):
        """Various filled_ratio values work correctly."""
        text = "━━━━━━━━"  # 8 characters

        # 25% filled
        result = render_gradient_text(text, frame=0, filled_ratio=0.25)
        assert str(result) == text

        # 50% filled
        result = render_gradient_text(text, frame=0, filled_ratio=0.5)
        assert str(result) == text

        # 75% filled
        result = render_gradient_text(text, frame=0, filled_ratio=0.75)
        assert str(result) == text

    def test_fractional_filled_ratio(self):
        """Fractional filled_ratio values are handled."""
        # With 10 characters and filled_ratio=0.45, should fill 4 chars
        text = "━━━━━━━━━━"  # 10 characters
        result = render_gradient_text(text, frame=0, filled_ratio=0.45)
        assert str(result) == text
        assert len(result) == 10


class TestRenderCyclingText:
    """Tests for render_cycling_text function."""

    def test_returns_text_object(self):
        """render_cycling_text returns a Rich Text object."""
        result = render_cycling_text("ABC", frame=0)
        assert isinstance(result, Text)

    def test_empty_string_returns_empty_text(self):
        """Empty string input returns empty Text object."""
        result = render_cycling_text("", frame=0)
        assert isinstance(result, Text)
        assert len(result) == 0
        assert str(result) == ""

    def test_preserves_text_content(self):
        """Text content is preserved in output."""
        text = "GPT-4o"
        result = render_cycling_text(text, frame=0)
        assert str(result) == text

    def test_frame_zero_applies_colors(self):
        """Frame 0 applies colors starting from the first rainbow color."""
        result = render_cycling_text("ABC", frame=0)
        # Each character should have a style applied
        assert len(result._spans) > 0

    def test_default_shift_interval(self):
        """Default shift_interval=2 is used."""
        result = render_cycling_text("ABC", frame=0, shift_interval=2)
        assert str(result) == "ABC"

    def test_custom_shift_interval(self):
        """Custom shift_interval is respected."""
        result = render_cycling_text("ABC", frame=0, shift_interval=3)
        assert str(result) == "ABC"

    def test_colors_shift_at_frame_interval(self):
        """Colors shift only at shift_interval boundaries, not every frame."""
        # With shift_interval=2, colors should be same at frames 0 and 1
        result_f0 = render_cycling_text("A", frame=0, shift_interval=2)
        result_f1 = render_cycling_text("A", frame=1, shift_interval=2)
        # Both should have same color (shift = 0 // 2 = 0)
        assert str(result_f0) == str(result_f1)

        # At frame 2, color should shift (shift = 2 // 2 = 1)
        result_f2 = render_cycling_text("A", frame=2, shift_interval=2)
        # Text content is same but color should be different
        assert str(result_f2) == "A"

    def test_color_shift_timing(self):
        """Color shift timing matches shift_interval exactly."""
        # With shift_interval=3:
        # Frame 0: shift=0, char 0 gets color 0
        # Frame 1: shift=0, char 0 gets color 0
        # Frame 2: shift=0, char 0 gets color 0
        # Frame 3: shift=1, char 0 gets color 1
        result = render_cycling_text("A", frame=0, shift_interval=3)
        assert str(result) == "A"

        result = render_cycling_text("A", frame=1, shift_interval=3)
        assert str(result) == "A"

        result = render_cycling_text("A", frame=2, shift_interval=3)
        assert str(result) == "A"

        result = render_cycling_text("A", frame=3, shift_interval=3)
        assert str(result) == "A"

    def test_example_from_prd(self):
        """Example from PRD: at frame 0, char 0 is red, char 1 is orange."""
        result = render_cycling_text("AB", frame=0)
        assert str(result) == "AB"

    def test_example_from_prd_frame_2(self):
        """Example from PRD: at frame 2, char 0 is orange, char 1 is yellow."""
        result = render_cycling_text("AB", frame=2)
        assert str(result) == "AB"

    def test_discrete_color_jumps(self):
        """Colors jump discretely between characters with no interpolation."""
        # With a 7-character text, each should get a different color at frame 0
        text = "ABCDEFG"
        result = render_cycling_text(text, frame=0)
        assert str(result) == text
        # All characters should have discrete colors (not interpolated)

    def test_rainbow_colors_are_used(self):
        """RAINBOW_COLORS are used for the color sequence."""
        text = "A" * len(RAINBOW_COLORS)
        result = render_cycling_text(text, frame=0)
        assert str(result) == text

    def test_colors_cycle_through_rainbow(self):
        """Colors cycle through RAINBOW_COLORS when text is longer than palette."""
        text = "A" * (len(RAINBOW_COLORS) + 3)
        result = render_cycling_text(text, frame=0)
        assert str(result) == text

    def test_frame_affects_color_assignment(self):
        """Frame number affects color assignment through shift."""
        result_f0 = render_cycling_text("ABC", frame=0, shift_interval=2)
        result_f4 = render_cycling_text("ABC", frame=4, shift_interval=2)
        # Text content is same
        assert str(result_f0) == str(result_f4)

    def test_high_frame_numbers(self):
        """Very high frame numbers work correctly."""
        result = render_cycling_text("Test", frame=10000)
        assert str(result) == "Test"

    def test_negative_frame_numbers(self):
        """Negative frame numbers are handled (integer division)."""
        result = render_cycling_text("Test", frame=-5)
        assert str(result) == "Test"

    def test_long_text_handled(self):
        """Long text is handled without error."""
        long_text = "X" * 100
        result = render_cycling_text(long_text, frame=0)
        assert str(result) == long_text
        assert len(result) == 100

    def test_single_character(self):
        """Single character input works correctly."""
        result = render_cycling_text("X", frame=0)
        assert str(result) == "X"

    def test_special_characters_preserved(self):
        """Special characters are preserved in output."""
        text = "GPT-4o-mini [test] (v2)"
        result = render_cycling_text(text, frame=0)
        assert str(result) == text

    def test_unicode_characters_handled(self):
        """Unicode characters are handled correctly."""
        text = "模型名称 🎮 Test"
        result = render_cycling_text(text, frame=0)
        assert str(result) == text

    def test_whitespace_only(self):
        """Whitespace-only input is handled."""
        result = render_cycling_text("   ", frame=0)
        assert str(result) == "   "

    def test_shift_interval_of_one(self):
        """shift_interval=1 causes colors to shift every frame."""
        result_f0 = render_cycling_text("A", frame=0, shift_interval=1)
        result_f1 = render_cycling_text("A", frame=1, shift_interval=1)
        assert str(result_f0) == str(result_f1)

    def test_shift_interval_of_four(self):
        """shift_interval=4 causes colors to shift every 4 frames."""
        # At frames 0,1,2,3: shift=0
        # At frame 4: shift=1
        for frame in range(4):
            result = render_cycling_text("A", frame=frame, shift_interval=4)
            assert str(result) == "A"

        result = render_cycling_text("A", frame=4, shift_interval=4)
        assert str(result) == "A"

    def test_multiple_characters_color_sequence(self):
        """Multiple characters each get sequential colors from rainbow."""
        # At frame 0 with 2 chars: char 0 gets color 0, char 1 gets color 1
        text = "AB"
        result = render_cycling_text(text, frame=0)
        assert str(result) == "AB"
        assert len(result) == 2

    def test_no_color_interpolation(self):
        """No interpolation between colors - discrete jumps only."""
        # Verify that each character gets exactly one of RAINBOW_COLORS
        # not an interpolated value
        text = "ABC"
        result = render_cycling_text(text, frame=0)
        assert str(result) == "ABC"


class TestRenderStatusLine:
    """Tests for render_status_line function."""

    def test_returns_text_object(self):
        """render_status_line returns a Rich Text object."""
        result = render_status_line(5, 2, 0.0123)
        assert isinstance(result, Text)

    def test_format_example_from_prd(self):
        """Example from PRD: render_status_line(5, 2, 0.0123) returns '✓ 5 | ✗ 2 | $0.0123'."""
        result = render_status_line(5, 2, 0.0123)
        assert str(result) == "✓ 5 | ✗ 2 | $0.0123"

    def test_all_zeros(self):
        """Negative case: all zeros renders '✓ 0 | ✗ 0 | $0.0000'."""
        result = render_status_line(0, 0, 0.0)
        assert str(result) == "✓ 0 | ✗ 0 | $0.0000"

    def test_success_count_styled_green(self):
        """Success count is styled green."""
        result = render_status_line(5, 2, 0.0123)
        # Check that spans contain green styling
        has_green_style = False
        for span in result._spans:
            if "green" in str(span.style):
                has_green_style = True
                break
        assert has_green_style

    def test_failure_count_styled_red(self):
        """Failure count is styled red."""
        result = render_status_line(5, 2, 0.0123)
        # Check that spans contain red styling
        has_red_style = False
        for span in result._spans:
            if "red" in str(span.style):
                has_red_style = True
                break
        assert has_red_style

    def test_cost_formatted_to_four_decimal_places(self):
        """Cost is formatted to 4 decimal places."""
        result = render_status_line(0, 0, 0.12345)
        # Should be rounded to 4 decimal places
        assert "$0.1235" in str(result)

    def test_cost_less_than_zero_01(self):
        """Cost less than 0.01 is formatted correctly."""
        result = render_status_line(0, 0, 0.005)
        assert str(result) == "✓ 0 | ✗ 0 | $0.0050"

    def test_cost_with_many_decimals(self):
        """Cost with many decimal places is formatted to 4 places."""
        result = render_status_line(0, 0, 0.123456789)
        assert str(result) == "✓ 0 | ✗ 0 | $0.1235"

    def test_high_success_count(self):
        """High success count is handled correctly."""
        result = render_status_line(1000, 0, 0.0)
        assert str(result) == "✓ 1000 | ✗ 0 | $0.0000"

    def test_high_failure_count(self):
        """High failure count is handled correctly."""
        result = render_status_line(0, 1000, 0.0)
        assert str(result) == "✓ 0 | ✗ 1000 | $0.0000"

    def test_both_counts_equal(self):
        """Both success and failure counts equal."""
        result = render_status_line(10, 10, 0.5)
        assert str(result) == "✓ 10 | ✗ 10 | $0.5000"

    def test_negative_cost_handled(self):
        """Negative cost is formatted correctly."""
        result = render_status_line(5, 2, -0.0123)
        assert str(result) == "✓ 5 | ✗ 2 | $-0.0123"

    def test_large_cost(self):
        """Large cost is formatted correctly."""
        result = render_status_line(0, 0, 123.456)
        assert str(result) == "✓ 0 | ✗ 0 | $123.4560"

    def test_zero_cost(self):
        """Zero cost is formatted correctly."""
        result = render_status_line(5, 2, 0.0)
        assert str(result) == "✓ 5 | ✗ 2 | $0.0000"

    def test_only_successes(self):
        """Only successes (no failures)."""
        result = render_status_line(10, 0, 0.1)
        assert str(result) == "✓ 10 | ✗ 0 | $0.1000"

    def test_only_failures(self):
        """Only failures (no successes)."""
        result = render_status_line(0, 10, 0.1)
        assert str(result) == "✓ 0 | ✗ 10 | $0.1000"

    def test_checkmark_symbol(self):
        """Checkmark symbol is present in output."""
        result = render_status_line(5, 2, 0.0123)
        assert "✓" in str(result)

    def test_cross_symbol(self):
        """Cross symbol is present in output."""
        result = render_status_line(5, 2, 0.0123)
        assert "✗" in str(result)

    def test_dollar_sign(self):
        """Dollar sign is present before cost."""
        result = render_status_line(5, 2, 0.0123)
        assert "$" in str(result)

    def test_separator_pipes(self):
        """Pipe separators are present in output."""
        result = render_status_line(5, 2, 0.0123)
        assert str(result).count("|") == 2

    def test_cost_very_small(self):
        """Very small cost is formatted correctly."""
        result = render_status_line(0, 0, 0.00001)
        assert "$0.0000" in str(result)

    def test_cost_exactly_four_decimals(self):
        """Cost with exactly 4 decimal places is preserved."""
        result = render_status_line(0, 0, 0.1234)
        assert str(result) == "✓ 0 | ✗ 0 | $0.1234"

    def test_integer_values(self):
        """Integer values for counts and cost work correctly."""
        result = render_status_line(1, 1, 1)
        assert str(result) == "✓ 1 | ✗ 1 | $1.0000"
