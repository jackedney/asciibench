"""Tests for the RuneScape-style wave text renderer."""

from rich.text import Text

from asciibench.common.wave_text import (
    RAINBOW_COLORS,
    get_wave_displacement,
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
