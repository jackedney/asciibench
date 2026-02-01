"""Diagnostic tests for ASCII art whitespace preservation.

This module provides tests to help diagnose where leading whitespace
is being lost in the ASCII art pipeline.
"""

import pytest

from asciibench.generator.sanitizer import extract_ascii_from_markdown


class TestWhitespacePreservation:
    """Tests to diagnose whitespace issues in ASCII art extraction."""

    def test_first_line_leading_spaces_preserved(self):
        """Verify leading spaces on first line are preserved after extraction."""
        # Simulate model output with leading spaces on first line
        markdown = "```\n      ___\n     /   \\\n    | o o |\n```"
        result = extract_ascii_from_markdown(markdown)

        # The first line should start with 6 spaces
        first_line = result.split("\n")[0]
        assert first_line == "      ___", f"Expected '      ___' but got '{first_line}'"
        assert first_line.startswith("      "), f"First line should start with 6 spaces: '{first_line}'"

    def test_first_line_leading_spaces_with_text_lang(self):
        """Verify leading spaces preserved with ```text code block."""
        markdown = "```text\n      ___\n     /   \\\n```"
        result = extract_ascii_from_markdown(markdown)

        first_line = result.split("\n")[0]
        assert first_line == "      ___", f"Expected '      ___' but got '{first_line}'"

    def test_first_line_leading_spaces_with_ascii_lang(self):
        """Verify leading spaces preserved with ```ascii code block."""
        markdown = "```ascii\n      ___\n     /   \\\n```"
        result = extract_ascii_from_markdown(markdown)

        first_line = result.split("\n")[0]
        assert first_line == "      ___", f"Expected '      ___' but got '{first_line}'"

    def test_blank_lines_before_content_removed_but_first_line_spaces_kept(self):
        """Ensure blank lines are removed but first content line keeps its spaces."""
        # Blank lines at start should be removed, but first content line
        # should preserve its leading whitespace
        markdown = "```\n\n\n      ___\n     /   \\\n```"
        result = extract_ascii_from_markdown(markdown)

        first_line = result.split("\n")[0]
        assert first_line == "      ___", f"Expected '      ___' but got '{first_line}'"
        # Verify blank lines were removed (should only have 2 lines now)
        lines = result.split("\n")
        assert len(lines) == 2, f"Expected 2 lines after stripping blanks, got {len(lines)}"

    def test_skeleton_ascii_art_from_real_output(self):
        """Test with realistic skeleton ASCII art output."""
        # This simulates what Kimi K2.5 might output
        markdown = """Here is a skeleton in ASCII art:

```
      ___
      /     \\
     | o   o |
     |   >   |
     |  ===  |
      \\_____/
        |||
    ~==|||||==~
      |||||
      |||||
       |||
      _/ \\_
     /     \\
    '       '
```

This skeleton features a simple head with eyes and a nose."""

        result = extract_ascii_from_markdown(markdown)
        lines = result.split("\n")

        # First line should have 6 leading spaces
        assert lines[0] == "      ___", f"First line should be '      ___', got '{lines[0]}'"
        assert lines[0].startswith("      "), f"First line should start with spaces"

    def test_whitespace_only_first_line_treated_as_blank(self):
        """Line with only whitespace should be treated as blank and removed."""
        # First line is just spaces (blank), second line has actual content
        markdown = "```\n   \n      ___\n```"
        result = extract_ascii_from_markdown(markdown)

        first_line = result.split("\n")[0]
        # The whitespace-only line should be stripped, leaving "      ___" as first line
        assert first_line == "      ___", f"Expected '      ___' but got '{first_line}'"

    def test_tab_characters_preserved(self):
        """Ensure tab characters in ASCII art are preserved."""
        markdown = "```\n\t\tHead\n\t/|\\\n```"
        result = extract_ascii_from_markdown(markdown)

        assert result.startswith("\t\t"), f"Tabs should be preserved: '{result}'"

    def test_mixed_spaces_and_content(self):
        """Test that lines with both leading spaces and content are preserved."""
        markdown = "```\n    A\n   B B\n  C   C\n D     D\n```"
        result = extract_ascii_from_markdown(markdown)
        lines = result.split("\n")

        assert lines[0] == "    A"
        assert lines[1] == "   B B"
        assert lines[2] == "  C   C"
        assert lines[3] == " D     D"


class TestModelOutputSimulation:
    """Tests simulating actual model outputs to diagnose whitespace issues."""

    def test_model_output_with_no_leading_spaces_on_first_line(self):
        """Test what happens when model doesn't put spaces on first line.

        This tests the hypothesis that the MODEL is outputting without
        leading spaces on the first line, not that our code is stripping them.
        """
        # This is what Kimi K2.5 might actually output - no leading spaces on first line
        model_raw_output = """```
___
      /     \\
     | o   o |
     |   >   |
     |  ===  |
      \\_____/
        |||
    ~==|||||==~
      |||||
      |||||
       |||
      _/ \\_
     /     \\
    '       '
```"""

        result = extract_ascii_from_markdown(model_raw_output)
        lines = result.split("\n")

        # If the model doesn't output leading spaces, we can't magically add them
        # This test documents the actual behavior - first line has no leading spaces
        assert lines[0] == "___", f"First line: '{lines[0]}'"
        # But second line DOES have leading spaces
        assert lines[1].startswith("      "), f"Second line should have spaces: '{lines[1]}'"

    def test_model_output_with_leading_spaces_on_first_line(self):
        """Test that if model DOES output leading spaces, they are preserved."""
        # This is what we WANT the model to output
        model_raw_output = """```
      ___
      /     \\
     | o   o |
```"""

        result = extract_ascii_from_markdown(model_raw_output)
        lines = result.split("\n")

        # If model outputs leading spaces, they should be preserved
        assert lines[0] == "      ___", f"First line: '{lines[0]}'"


class TestHTMLEscapingPreservation:
    """Tests for HTML escaping that verify whitespace isn't affected."""

    def test_html_escape_preserves_leading_spaces(self):
        """Ensure html.escape doesn't affect leading whitespace."""
        import html

        ascii_art = "      ___\n     /   \\\n    | o o |"
        escaped = html.escape(ascii_art)

        first_line = escaped.split("\n")[0]
        assert first_line == "      ___", f"html.escape should preserve spaces: '{first_line}'"

    def test_pre_tag_with_leading_whitespace(self):
        """Test that a <pre> tag should preserve leading whitespace."""
        import html

        ascii_art = "      ___\n     /   \\\n    | o o |"
        escaped = html.escape(ascii_art)

        # Construct what would go in the HTML
        pre_content = f"<pre>{escaped}</pre>"

        # Verify the content still has leading spaces after the <pre> tag
        assert "<pre>      ___" in pre_content, f"Leading spaces should be after <pre>: {pre_content[:50]}"
