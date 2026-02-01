"""Tests for the sanitizer module."""

from asciibench.generator.sanitizer import extract_ascii_from_markdown


class TestExtractAsciiFromMarkdown:
    """Tests for extract_ascii_from_markdown function."""

    def test_extract_from_text_code_block(self):
        """Extract content from ```text code block."""
        markdown = "```text\n/_/\\\n( o.o )\n > ^ <\n```"
        result = extract_ascii_from_markdown(markdown)
        assert result == "/_/\\\n( o.o )\n > ^ <"

    def test_extract_from_ascii_code_block(self):
        """Extract content from ```ascii code block."""
        markdown = "```ascii\n/_/\\\n( o.o )\n```"
        result = extract_ascii_from_markdown(markdown)
        assert result == "/_/\\\n( o.o )"

    def test_extract_from_plaintext_code_block(self):
        """Extract content from ```plaintext code block."""
        markdown = "```plaintext\nHello World\n```"
        result = extract_ascii_from_markdown(markdown)
        assert result == "Hello World"

    def test_extract_from_no_language_code_block(self):
        """Extract content from ``` code block (no language specifier)."""
        markdown = "```\n/_/\\\n( o.o )\n > ^ <\n```"
        result = extract_ascii_from_markdown(markdown)
        assert result == "/_/\\\n( o.o )\n > ^ <"

    def test_no_code_block_returns_empty_string(self):
        """Return empty string when no code block is found."""
        markdown = "No code block here"
        result = extract_ascii_from_markdown(markdown)
        assert result == ""

    def test_empty_code_block_returns_empty_string(self):
        """Return empty string when code block is empty."""
        markdown = "```\n```"
        result = extract_ascii_from_markdown(markdown)
        assert result == ""

    def test_empty_text_code_block_returns_empty_string(self):
        """Return empty string when text code block is empty."""
        markdown = "```text\n```"
        result = extract_ascii_from_markdown(markdown)
        assert result == ""

    def test_multiple_code_blocks_returns_first(self):
        """Return content from the first code block when multiple exist."""
        markdown = "```\nFirst block\n```\n\n```\nSecond block\n```"
        result = extract_ascii_from_markdown(markdown)
        assert result == "First block"

    def test_strips_leading_trailing_blank_lines(self):
        """Strip leading and trailing blank lines but preserve line whitespace."""
        markdown = "```\n\n  content  \n\n```"
        result = extract_ascii_from_markdown(markdown)
        # Empty lines at start/end are stripped, but whitespace on content lines is preserved
        assert result == "  content  "

    def test_preserves_internal_whitespace(self):
        """Preserve internal whitespace and newlines in content."""
        markdown = "```\nline1\n  indented\nline3\n```"
        result = extract_ascii_from_markdown(markdown)
        assert result == "line1\n  indented\nline3"

    def test_code_block_with_surrounding_text(self):
        """Extract code block when surrounded by other text."""
        markdown = "Here is some art:\n```\ncat\n```\nIsn't it nice?"
        result = extract_ascii_from_markdown(markdown)
        assert result == "cat"

    def test_example_from_acceptance_criteria(self):
        """Test the example from the acceptance criteria."""
        # The example: '```\n/_/\\n( o.o )\n```' returns '/_/\\n( o.o )'
        markdown = "```\n/_/\\\n( o.o )\n```"
        result = extract_ascii_from_markdown(markdown)
        assert result == "/_/\\\n( o.o )"

    def test_empty_input_returns_empty_string(self):
        """Return empty string for empty input."""
        result = extract_ascii_from_markdown("")
        assert result == ""

    def test_only_backticks_no_closing(self):
        """Return empty string when code block is not properly closed."""
        markdown = "```\nunclosed content"
        result = extract_ascii_from_markdown(markdown)
        assert result == ""

    def test_code_block_with_special_characters(self):
        """Handle special characters in content."""
        markdown = "```\n( ^_^ )  (>_<)  \\o/\n```"
        result = extract_ascii_from_markdown(markdown)
        assert result == "( ^_^ )  (>_<)  \\o/"

    def test_ignores_other_language_specifiers(self):
        """Ignore code blocks with unsupported language specifiers."""
        # Should not match ```python code blocks
        markdown = "```python\nprint('hello')\n```"
        result = extract_ascii_from_markdown(markdown)
        assert result == ""

    def test_code_block_with_only_whitespace(self):
        """Return empty string when code block contains only whitespace."""
        markdown = "```\n   \n  \n```"
        result = extract_ascii_from_markdown(markdown)
        assert result == ""

    def test_preserves_ascii_art_alignment(self):
        """Preserve leading whitespace on first line for ASCII art alignment."""
        # This is critical for ASCII art that starts with leading spaces
        markdown = "```\n      ___\n     /   \\\n    | o o |\n```"
        result = extract_ascii_from_markdown(markdown)
        # All leading spaces must be preserved for proper alignment
        assert result == "      ___\n     /   \\\n    | o o |"
        # Verify the first line starts with spaces
        assert result.startswith("      ")

    def test_tolerates_spaces_after_opening_backticks(self):
        """Handle code blocks with spaces after opening backticks.

        Some models (e.g., GPT-4.1 Mini) output ```  instead of ```.
        The sanitizer should tolerate this whitespace variation.
        """
        # Two spaces after backticks
        markdown = "```  \n/_/\\\n( o.o )\n```"
        result = extract_ascii_from_markdown(markdown)
        assert result == "/_/\\\n( o.o )"

    def test_tolerates_tabs_after_opening_backticks(self):
        """Handle code blocks with tabs after opening backticks."""
        markdown = "```\t\n/_/\\\n( o.o )\n```"
        result = extract_ascii_from_markdown(markdown)
        assert result == "/_/\\\n( o.o )"

    def test_tolerates_spaces_before_language_specifier(self):
        """Handle code blocks with spaces between backticks and language."""
        markdown = "```  text\n/_/\\\n( o.o )\n```"
        result = extract_ascii_from_markdown(markdown)
        assert result == "/_/\\\n( o.o )"
