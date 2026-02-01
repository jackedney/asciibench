"""RuneScape-style wave text renderer with rainbow colors and sine wave animation."""

import math

from rich.text import Text

# Rainbow color sequence with smooth red-to-violet transition:
# red -> orange -> yellow -> green -> cyan -> blue -> blue-magenta ->
# magenta -> magenta-violet -> violet
RAINBOW_COLORS = [
    "#FF0000",  # red
    "#FF7F00",  # orange
    "#FFFF00",  # yellow
    "#00FF00",  # green
    "#00FFFF",  # cyan
    "#0000FF",  # blue
    "#4000FF",  # blue-magenta
    "#8000FF",  # blue-magenta (intermediate)
    "#FF00FF",  # magenta
    "#FF00BF",  # magenta-violet
    "#FF007F",  # magenta-violet (intermediate)
    "#9400D3",  # violet
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


def render_gradient_text(text: str, frame: int, filled_ratio: float) -> Text:
    """Render text with a rainbow gradient that flows and shifts based on frame.

    Only the filled portion of the text (based on filled_ratio) displays the
    gradient colors. The remaining portion uses a dim/empty style for unfilled
    characters.

    Args:
        text: The text to render with gradient effect.
        frame: Animation frame number (affects gradient position/shift).
        filled_ratio: Ratio of text to fill with gradient (0.0 to 1.0).

    Returns:
        Rich Text object with gradient-styled characters for filled portion
        and dim style for unfilled portion. Empty Text if input is empty.

    Example:
        render_gradient_text('━━━━━───', 0, 0.5) shows gradient on first 4 chars.
    """
    if not text:
        return Text()

    # Clamp filled_ratio to [0.0, 1.0]
    filled_ratio = max(0.0, min(1.0, filled_ratio))

    result = Text()
    num_chars = len(text)
    filled_chars = int(num_chars * filled_ratio)

    for i, char in enumerate(text):
        if i < filled_chars:
            # Calculate gradient color for this character
            # Gradient position is based on character index and frame offset
            # This creates a flowing effect across the text
            gradient_pos = (i + frame * 0.5) % num_chars / num_chars

            # Find which two rainbow colors we're between
            num_colors = len(RAINBOW_COLORS)
            color_index = gradient_pos * (num_colors - 1)
            color_idx_floor = int(color_index)
            color_idx_ceil = min(color_idx_floor + 1, num_colors - 1)
            t = color_index - color_idx_floor

            # Interpolate between the two adjacent rainbow colors
            color1 = RAINBOW_COLORS[color_idx_floor]
            color2 = RAINBOW_COLORS[color_idx_ceil]
            gradient_color = interpolate_color(color1, color2, t)

            result.append(char, style=gradient_color)
        else:
            # Unfilled portion gets dim style
            result.append(char, style="dim")

    return result


def render_cycling_text(text: str, frame: int, shift_interval: int = 2) -> Text:
    """Render text with Runescape-style discrete color cycling animation.

    Each character gets a color from the rainbow sequence. The color pattern
    shifts every shift_interval frames (not every frame), creating a classic
    discrete cycling effect. Colors jump discretely between characters with
    no interpolation.

    Args:
        text: The text to render with color cycling effect.
        frame: Animation frame number (affects color shift timing).
        shift_interval: Number of frames between color shifts (default 2).

    Returns:
        Rich Text object with rainbow-styled characters. Empty Text if input is empty.

    Example:
        At frame 0: char 0 is red, char 1 is orange
        At frame 2: char 0 is orange, char 1 is yellow (with shift_interval=2)
    """
    if not text:
        return Text()

    result = Text()
    num_colors = len(RAINBOW_COLORS)

    # Calculate discrete shift based on shift_interval
    # The color pattern only shifts every shift_interval frames
    shift = frame // shift_interval

    for i, char in enumerate(text):
        # Calculate color index with discrete shift
        # Colors jump discretely between characters (no interpolation)
        color_index = (i + shift) % num_colors
        color = RAINBOW_COLORS[color_index]

        result.append(char, style=color)

    return result


def render_status_line(success_count: int, failure_count: int, total_cost: float) -> Text:
    """Render a status line showing running totals for success, failure, and cost.

    The status line displays the number of successful and failed operations,
    along with the total accumulated cost. Success count is styled green,
    failure count is styled red, and cost uses default styling.

    Args:
        success_count: Number of successful operations.
        failure_count: Number of failed operations.
        total_cost: Total accumulated cost.

    Returns:
        Rich Text object with styled status line. Format: '✓ {success} | ✗ {failed} | ${cost:.4f}'

    Example:
        render_status_line(5, 2, 0.0123) returns '✓ 5 | ✗ 2 | $0.0123'
    """
    result = Text()

    # Success checkmark and count (green)
    result.append("✓ ", style="green")
    result.append(str(success_count), style="green")

    # Separator
    result.append(" | ")

    # Failure cross and count (red)
    result.append("✗ ", style="red")
    result.append(str(failure_count), style="red")

    # Separator
    result.append(" | ")

    # Cost (default style)
    result.append(f"${total_cost:.4f}")

    return result
