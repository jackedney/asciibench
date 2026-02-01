"""Single model test for diagnosing ASCII art whitespace issues.

This test can make a real API call to diagnose where leading whitespace
on the first line is being lost. Run with:

    python -m pytest tests/test_single_model_whitespace.py -v -s

Use -s to see the actual output for debugging.

API tests require OPENROUTER_API_KEY to be set.
"""

import os
import re

import pytest


# Mark for API tests only
requires_api_key = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping real API tests",
)


@pytest.fixture
def api_key():
    """Get API key from environment."""
    return os.getenv("OPENROUTER_API_KEY")


@requires_api_key
class TestSingleModelWhitespaceDiagnosis:
    """Real API call test to diagnose whitespace issues."""

    # Use a cheap, fast model for testing
    MODEL_ID = "openai/gpt-4.1-mini"
    MODEL_NAME = "GPT-4.1 Mini"

    # Simple prompt that should produce centered ASCII art
    PROMPT = "Draw a simple smiley face in ASCII art. Center it with leading spaces."

    def test_raw_model_output_whitespace(self, api_key):
        """Test what the model actually outputs, including whitespace.

        This test:
        1. Makes a real API call
        2. Prints the raw output for inspection
        3. Shows exactly what whitespace is in the first line
        """
        from asciibench.common.config import Settings
        from asciibench.common.yaml_config import load_generation_config
        from asciibench.generator.client import OpenRouterClient
        from asciibench.generator.sanitizer import extract_ascii_from_markdown

        # Use real settings with API key override
        settings = Settings()
        config = load_generation_config()

        client = OpenRouterClient(api_key=api_key, base_url=settings.base_url)

        # Make the API call
        response = client.generate(self.MODEL_ID, self.PROMPT, config=config)
        raw_output = response.text

        print("\n" + "=" * 60)
        print("RAW MODEL OUTPUT (repr to show whitespace):")
        print("=" * 60)
        # Print first 500 chars with repr to see exact characters
        print(repr(raw_output[:500]))

        print("\n" + "=" * 60)
        print("RAW MODEL OUTPUT (visual):")
        print("=" * 60)
        print(raw_output)

        # Extract ASCII from the output
        ascii_art = extract_ascii_from_markdown(raw_output)

        print("\n" + "=" * 60)
        print("EXTRACTED ASCII (repr to show whitespace):")
        print("=" * 60)
        print(repr(ascii_art[:300] if ascii_art else "EMPTY"))

        print("\n" + "=" * 60)
        print("EXTRACTED ASCII (visual):")
        print("=" * 60)
        print(ascii_art if ascii_art else "NO ASCII EXTRACTED")

        if ascii_art:
            lines = ascii_art.split("\n")
            print("\n" + "=" * 60)
            print("LINE-BY-LINE ANALYSIS:")
            print("=" * 60)
            for i, line in enumerate(lines[:5]):  # First 5 lines
                leading_spaces = len(line) - len(line.lstrip(" "))
                print(f"Line {i}: {leading_spaces} leading spaces | repr: {repr(line)}")

            # The actual assertion - document what's happening
            first_line = lines[0]
            leading_spaces = len(first_line) - len(first_line.lstrip(" "))
            print(f"\nFirst line has {leading_spaces} leading space(s)")

            # This is the key finding:
            # If this fails, the model IS outputting leading spaces (good)
            # If this passes, the model is NOT outputting leading spaces (the problem)
            if leading_spaces == 0:
                print("\n>>> DIAGNOSIS: Model is NOT outputting leading spaces on first line <<<")
            else:
                print(f"\n>>> DIAGNOSIS: Model IS outputting {leading_spaces} leading spaces on first line <<<")

        assert ascii_art, "Should extract some ASCII art"

    def test_specific_skeleton_prompt(self, api_key):
        """Test with the exact skeleton prompt from demo mode.

        This uses the same prompt as the demo to reproduce the issue exactly.
        """
        from asciibench.common.config import Settings
        from asciibench.common.yaml_config import load_generation_config
        from asciibench.generator.client import OpenRouterClient
        from asciibench.generator.sanitizer import extract_ascii_from_markdown

        settings = Settings()
        config = load_generation_config()

        client = OpenRouterClient(api_key=api_key, base_url=settings.base_url)

        # Use exact demo prompt
        prompt = "Draw a skeleton in ASCII art"
        response = client.generate(self.MODEL_ID, prompt, config=config)
        raw_output = response.text

        print("\n" + "=" * 60)
        print(f"SKELETON PROMPT TEST - Model: {self.MODEL_ID}")
        print("=" * 60)
        print("\nRAW OUTPUT:")
        print(raw_output)

        ascii_art = extract_ascii_from_markdown(raw_output)
        print("\nEXTRACTED ASCII:")
        print(ascii_art if ascii_art else "NO ASCII EXTRACTED")

        if ascii_art:
            lines = ascii_art.split("\n")
            first_line = lines[0]
            leading_spaces = len(first_line) - len(first_line.lstrip(" "))
            print(f"\n>>> First line: '{first_line}'")
            print(f">>> Leading spaces on first line: {leading_spaces}")

        assert ascii_art, "Should extract ASCII art from skeleton prompt"


class TestCodeBlockPatternMatching:
    """Test regex pattern matching on code blocks."""

    def test_code_block_immediately_after_backticks(self):
        """Test what happens when content starts right after opening backticks.

        Some models might output:
        ```___        (content on same line as backticks)
        rather than:
        ```
        ___          (content on next line)
        """
        from asciibench.generator.sanitizer import extract_ascii_from_markdown

        # Content on same line as backticks - currently NOT matched by pattern
        markdown1 = "```___\n  /   \\\n```"
        result1 = extract_ascii_from_markdown(markdown1)
        print(f"\nSame-line content: '{markdown1}'")
        print(f"Extracted: '{result1}'")

        # Content on next line (standard) - this SHOULD work
        markdown2 = "```\n___\n  /   \\\n```"
        result2 = extract_ascii_from_markdown(markdown2)
        print(f"\nNext-line content: '{markdown2}'")
        print(f"Extracted: '{result2}'")

        # The standard format should work
        assert result2 == "___\n  /   \\", f"Standard format should work: '{result2}'"

    def test_regex_pattern_captures_all_content(self):
        """Verify the regex pattern correctly captures content with leading spaces."""
        pattern = r"```(?:(?:text|ascii|plaintext)\s*\n|\n)(.*?)```"

        # Test case: content has leading spaces on all lines
        markdown = "```\n      HEAD\n     /    \\\n```"
        match = re.search(pattern, markdown, re.DOTALL)

        assert match is not None, "Pattern should match"
        captured = match.group(1)
        print(f"\nCaptured content: {repr(captured)}")

        # The captured content should include leading spaces
        assert captured.startswith("      HEAD"), f"Should capture leading spaces: {repr(captured)}"
