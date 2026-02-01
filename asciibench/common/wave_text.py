"""RuneScape-style wave text renderer with rainbow colors and sine wave animation."""

import math

from rich.text import Text

# Rainbow color sequence: red -> orange -> yellow -> green -> cyan -> blue -> purple
RAINBOW_COLORS = [
    "#FF0000",  # red
    "#FF7F00",  # orange
    "#FFFF00",  # yellow
    "#00FF00",  # green
    "#00FFFF",  # cyan
    "#0000FF",  # blue
    "#8B00FF",  # purple
]


def render_wave_text(text: str, frame: int = 0) -> Text:
    """Render text with RuneScape-style wave animation and rainbow colors.

    Characters are vertically displaced using a sine wave based on character
    position and frame number. Colors cycle through the rainbow spectrum
    and shift based on the frame number for a flowing effect.

    Args:
        text: The text to render with wave effect.
        frame: Animation frame number (affects wave position and color shift).

    Returns:
        Rich Text object with styled characters. Empty Text if input is empty.
    """
    if not text:
        return Text()

    result = Text()
    num_colors = len(RAINBOW_COLORS)

    for i, char in enumerate(text):
        # Calculate color index with frame-based shift for flowing effect
        # The color shifts based on both character position and frame
        color_index = (i + frame) % num_colors
        color = RAINBOW_COLORS[color_index]

        # Calculate vertical displacement using sine wave
        # Wave parameters: amplitude determines visual "height" of wave
        # Phase shifts based on character position and frame for animation
        wave_phase = (i * 0.5) + (frame * 0.3)
        displacement = math.sin(wave_phase)

        # For Rich Text, we can't actually displace vertically in terminal,
        # but we can simulate the wave effect through style variations.
        # The displacement value is stored in case future implementations
        # need it for multi-line wave effects.
        # For now, we apply the color and could add bold for "peak" positions.
        style = color
        if displacement > 0.7:
            # At wave peaks, add bold for visual emphasis
            style = f"bold {color}"

        result.append(char, style=style)

    return result


def get_wave_displacement(char_index: int, frame: int) -> float:
    """Calculate the vertical displacement for a character in the wave.

    Args:
        char_index: Position of the character in the text.
        frame: Current animation frame.

    Returns:
        Displacement value between -1.0 and 1.0.
    """
    wave_phase = (char_index * 0.5) + (frame * 0.3)
    return math.sin(wave_phase)


def fill_progress(model_name: str, width: int, progress: float) -> str:
    """Generate a string with model name repeated to fill width based on progress.

    The model name is repeated with space separators to fill (width * progress)
    characters. This is used for the loading bar to grow as generation progresses.

    Args:
        model_name: The model name to repeat (e.g., 'GPT-4o').
        width: The total terminal width to fill at 100% progress.
        progress: Progress percentage from 0.0 to 1.0 (clamped if out of range).

    Returns:
        String with model name repeated to fill the target length.
        Empty string if model_name is empty, width is 0, or progress is 0.
    """
    # Handle edge cases
    if not model_name or width <= 0:
        return ""

    # Clamp progress to [0.0, 1.0]
    progress = max(0.0, min(1.0, progress))

    # Calculate target length
    target_length = int(width * progress)

    if target_length == 0:
        return ""

    # Build repeated pattern with space separator
    pattern = model_name + " "
    pattern_len = len(pattern)

    # Calculate how many full repetitions we need
    if pattern_len == 0:
        return ""

    # Build the result by repeating the pattern
    repetitions = (target_length // pattern_len) + 2  # +2 to ensure we have enough
    full_text = pattern * repetitions

    # Truncate to exact target length
    return full_text[:target_length]


def interpolate_color(color1: str, color2: str, t: float) -> str:
    """Interpolate between two hex colors.

    Linearly interpolates between color1 and color2 based on parameter t.
    At t=0.0, returns color1. At t=1.0, returns color2. Values outside
    [0.0, 1.0] are clamped to the nearest bound.

    Args:
        color1: Starting hex color string (e.g., '#FF0000').
        color2: Ending hex color string (e.g., '#0000FF').
        t: Interpolation parameter (0.0 = color1, 1.0 = color2).

    Returns:
        Hex color string representing the interpolated color.

    Example:
        interpolate_color('#FF0000', '#0000FF', 0.5) returns '#800080' (purple).
    """
    # Clamp t to [0.0, 1.0]
    t = max(0.0, min(1.0, t))

    # Parse hex colors (remove '#' if present)
    r1 = int(color1.lstrip("#")[0:2], 16)
    g1 = int(color1.lstrip("#")[2:4], 16)
    b1 = int(color1.lstrip("#")[4:6], 16)

    r2 = int(color2.lstrip("#")[0:2], 16)
    g2 = int(color2.lstrip("#")[2:4], 16)
    b2 = int(color2.lstrip("#")[4:6], 16)

    # Linear interpolation
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)

    # Format back to hex
    return f"#{r:02X}{g:02X}{b:02X}"
