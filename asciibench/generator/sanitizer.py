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
    (e.g., ```text...``` or ```...```) and extracts the content.

    Supported code block formats:
        - ```text...```
        - ```ascii...```
        - ```plaintext...```
        - ```...``` (no language specifier)

    Args:
        markdown: Markdown text potentially containing ASCII art in code blocks

    Returns:
        Extracted ASCII art content with leading/trailing whitespace stripped,
        or empty string if no code blocks found or code block is empty.

    Examples:
        >>> markdown = "```text\\n/\\_/\\\\n( o.o )\\n > ^ <\\n```"
        >>> extract_ascii_from_markdown(markdown)
        '/\\_/\\\\n( o.o )\\n > ^ <'

        >>> markdown = "No code block here"
        >>> extract_ascii_from_markdown(markdown)
        ''
    """
    # Match code blocks with specific language specifiers (text, ascii, plaintext)
    # or no language specifier at all
    # The pattern uses alternation to handle:
    # 1. ``` followed by text/ascii/plaintext and then content
    # 2. ``` followed immediately by newline (no language) and then content
    pattern = r"```(?:(?:text|ascii|plaintext)\s*\n|\n)(.*?)```"

    match = re.search(pattern, markdown, re.DOTALL)

    if match:
        content = match.group(1).strip()
        return content

    return ""
