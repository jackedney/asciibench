"""Sanitizer module for extracting and validating ASCII art.

This module provides utilities for extracting ASCII art content
from markdown code blocks and validating the output.

Dependencies:
    - re: Python standard library for regex operations
"""

import re


def extract_ascii_from_markdown(markdown: str) -> str:
    """Extract ASCII art content from markdown code blocks.

    This function searches for code blocks in markdown format
    (e.g., ```text...``` or ```...```) and extracts the content
    with stripping leading/trailing blank lines and normalizing whitespace.

    Supported code block formats:
        - ```text...```
        - ```ascii...```
        - ```plaintext...```
        - ```...``` (no language specifier)

    Args:
        markdown: Markdown text potentially containing ASCII art in code blocks

    Returns:
        Extracted ASCII art content with leading/trailing blank lines stripped,
        or empty string if no code blocks found.

    Examples:
        >>> markdown = "```text\\n/\\_/\\\\n( o.o )\\n > ^ <\\n```"
        >>> extract_ascii_from_markdown(markdown)
        '/\\_/\\\\n( o.o )\\n > ^ <'

        >>> markdown = "No code block here"
        >>> extract_ascii_from_markdown(markdown)
        ''
    """
    # Match code blocks with optional language specifier (text, ascii, plaintext, or none)
    # Allow optional whitespace/tabs after backticks and before/after language specifier
    pattern = r"```[ \t]*(?:(?:text|ascii|plaintext)[ \t]*)?\n(.*?)```"

    match = re.search(pattern, markdown, re.DOTALL)

    if match:
        content = match.group(1)

        # Strip trailing newline (the one right before closing ```)
        if content.endswith("\n"):
            content = content[:-1]

        # Strip leading and trailing blank lines (empty or whitespace-only)
        lines = content.split("\n")

        # Remove leading blank lines
        while lines and not lines[0].strip():
            lines.pop(0)

        # Remove trailing blank lines
        while lines and not lines[-1].strip():
            lines.pop()

        result = "\n".join(lines)

        # If result is only whitespace, return empty string
        if not result.strip():
            return ""

        return result

    return ""
