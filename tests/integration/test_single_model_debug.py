"""Single model whitespace debug test.

Usage:
    uv run python -m pytest tests/test_single_model_debug.py -v -s --no-cov
"""

import pytest

from asciibench.common.config import Settings
from asciibench.common.yaml_config import load_generation_config
from asciibench.generator.client import OpenRouterClient
from asciibench.generator.sanitizer import extract_ascii_from_markdown


def _has_api_key() -> bool:
    try:
        settings = Settings()
        return bool(settings.openrouter_api_key)
    except Exception:
        return False


@pytest.mark.skipif(not _has_api_key(), reason="No API key")
def test_single_model_whitespace():
    """Test whitespace preservation with real API call."""
    settings = Settings()
    config = load_generation_config()

    client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        base_url=settings.base_url,
    )

    response = client.generate("openai/gpt-oss-120b", "Draw a skeleton in ASCII art", config=config)

    print(f"\n{'=' * 60}")
    print("RAW OUTPUT (first 200 chars):")
    print(repr(response.text[:200]))

    extracted = extract_ascii_from_markdown(response.text)
    lines = extracted.split("\n")

    print(f"\n{'=' * 60}")
    print("EXTRACTED (first 10 lines):")
    print(f"{'Line':>4} | {'Spaces':>6} | Content")
    print("-" * 60)

    for i, line in enumerate(lines[:10]):
        spaces = len(line) - len(line.lstrip(" "))
        print(f"{i:>4} | {spaces:>6} | {line[:40]!r}")

    first_line_spaces = len(lines[0]) - len(lines[0].lstrip(" "))
    print(f"\nFirst line spaces: {first_line_spaces}")

    assert extracted.strip(), f"Should extract non-empty ASCII art, got: {extracted!r}"
    print(
        f"\n>>> PASS: Extracted ASCII art ({len(lines)} lines,"
        f" first line has {first_line_spaces} leading spaces)"
    )
