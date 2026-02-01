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
    (e.g., ```text...``` or ```...```) and extracts the content exactly
    as it appears between the backticks without any manipulation.

    Supported code block formats:
        - ```text...```
        - ```ascii...```
        - ```plaintext...```
        - ```...``` (no language specifier)

    Args:
        markdown: Markdown text potentially containing ASCII art in code blocks

    Returns:
        Extracted ASCII art content exactly as it appears between backticks,
        or empty string if no code blocks found.

    Examples:
        >>> markdown = "```text\\n/\\_/\\\\n( o.o )\\n > ^ <\\n```"
        >>> extract_ascii_from_markdown(markdown)
        '/\\_/\\\\n( o.o )\\n > ^ <'

        >>> markdown = "No code block here"
        >>> extract_ascii_from_markdown(markdown)
        ''
    """
    # Match everything between ``` delimiters
    # Allow optional language specifier after opening backticks
    # Use \n? to only consume the newline after language specifier, not leading spaces
    pattern = r"```(?:[a-z]*\n)?(.*?)```"

    match = re.search(pattern, markdown, re.DOTALL)

    if match:
        # Return content exactly as is, without any whitespace manipulation
        return match.group(1)

    return ""
