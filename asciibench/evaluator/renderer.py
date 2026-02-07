"""ASCII-to-image renderer for VLM evaluation.

This module provides functionality to render ASCII art text as PNG images
suitable for processing by Vision Language Models (VLMs).

Dependencies:
    - PIL/Pillow: Image rendering and manipulation
"""

from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from asciibench.common.config import RendererConfig


def render_ascii_to_image(text: str | None, config: RendererConfig) -> bytes:
    """Render ASCII art text to a PNG image.

    Args:
        text: ASCII art text to render (can be multi-line)
        config: RendererConfig with font and color settings

    Returns:
        PNG image bytes suitable for base64 encoding

    Examples:
        >>> config = RendererConfig()
        >>> png_bytes = render_ascii_to_image(' /\\_/\\ \\n( o.o )', config)
        >>> isinstance(png_bytes, bytes)
        True

        >>> empty_png = render_ascii_to_image('', config)
        >>> isinstance(empty_png, bytes)
        True
    """
    if not text:
        text = ""

    lines = text.split("\n")

    try:
        font = ImageFont.truetype(config.font.family, config.font.size)
    except OSError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    line_sizes = [draw.textbbox((0, 0), line, font=font) for line in lines]

    max_width = int(max((size[2] for size in line_sizes), default=1))
    line_heights = [int(size[3] - size[1]) for size in line_sizes]
    total_height = int(
        sum(max(h, config.font.size) for h in line_heights) if line_heights else config.font.size
    )

    padding = 10
    width = int(max_width + padding * 2)
    height = int(total_height + padding * 2)

    image = Image.new("RGB", (width, height), config.background_color)
    draw = ImageDraw.Draw(image)

    y_offset = padding
    for line, height in zip(lines, line_heights, strict=True):
        draw.text(
            (padding, y_offset),
            line,
            fill=config.text_color,
            font=font,
        )
        y_offset += height if height else config.font.size

    output = BytesIO()
    image.save(output, format="PNG")
    output.seek(0)

    return output.getvalue()
