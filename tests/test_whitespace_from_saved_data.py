"""Analyze whitespace in saved demo results without making API calls.

This test examines the existing results.json to diagnose whitespace issues.

Usage:
    uv run python -m pytest tests/test_whitespace_from_saved_data.py -v -s --no-cov
"""

import json
from pathlib import Path

import pytest


class TestWhitespaceFromSavedData:
    """Analyze whitespace issues from existing saved data."""

    RESULTS_FILE = Path("demo_outputs/results.json")

    @pytest.fixture
    def results(self):
        """Load existing demo results."""
        if not self.RESULTS_FILE.exists():
            pytest.skip("demo_outputs/results.json not found")
        with open(self.RESULTS_FILE) as f:
            return json.load(f)

    def test_analyze_first_line_whitespace_all_models(self, results):
        """Analyze first line whitespace for all valid models."""
        print("\n" + "=" * 80)
        print("WHITESPACE ANALYSIS OF SAVED RESULTS")
        print("=" * 80)

        valid_results = [r for r in results if r.get("is_valid", False)]

        print(f"\nAnalyzing {len(valid_results)} valid results...\n")
        print(f"{'Model':<40} | {'1st Line Spaces':>15} | {'Max Spaces':>10} | First Line Content")
        print("-" * 100)

        problem_models = []

        for result in valid_results:
            model_name = result.get("model_name", "Unknown")
            ascii_output = result.get("ascii_output", "")

            if not ascii_output:
                continue

            lines = ascii_output.split("\n")
            first_line = lines[0] if lines else ""

            # Count leading spaces on first line
            first_line_spaces = len(first_line) - len(first_line.lstrip(" "))

            # Find max leading spaces on any line
            max_spaces = max(
                (len(line) - len(line.lstrip(" ")) for line in lines if line.strip()),
                default=0,
            )

            # Truncate first line for display
            first_line_display = first_line[:30] + "..." if len(first_line) > 30 else first_line

            print(
                f"{model_name:<40} | {first_line_spaces:>15} | "
                f"{max_spaces:>10} | {first_line_display!r}"
            )

            # Track models with the problem
            if first_line_spaces == 0 and max_spaces > 0:
                problem_models.append(model_name)

        print("\n" + "=" * 80)
        print("DIAGNOSIS SUMMARY")
        print("=" * 80)

        if problem_models:
            print(f"\n{len(problem_models)} models have the whitespace issue:")
            for model in problem_models:
                print(f"  - {model}")
            print("\n>>> These models output 0 leading spaces on first line")
            print(">>> but have leading spaces on other lines.")
            print(">>> This is a MODEL OUTPUT issue, not a parsing issue.")
        else:
            print("\nNo models show the first-line whitespace issue.")

    def test_detailed_single_model(self, results):
        """Detailed analysis of first model with the issue."""
        valid_results = [r for r in results if r.get("is_valid", False)]

        # Find first model with the issue
        problem_result = None
        for result in valid_results:
            ascii_output = result.get("ascii_output", "")
            if not ascii_output:
                continue
            lines = ascii_output.split("\n")
            first_line = lines[0] if lines else ""
            first_line_spaces = len(first_line) - len(first_line.lstrip(" "))
            max_spaces = max(
                (len(line) - len(line.lstrip(" ")) for line in lines if line.strip()),
                default=0,
            )
            if first_line_spaces == 0 and max_spaces > 0:
                problem_result = result
                break

        if not problem_result:
            print("\nNo model with the whitespace issue found in saved data.")
            return

        print(f"\n{'=' * 80}")
        print(f"DETAILED ANALYSIS: {problem_result.get('model_name', 'Unknown')}")
        print(f"Model ID: {problem_result.get('model_id', 'Unknown')}")
        print(f"{'=' * 80}")

        ascii_output = problem_result.get("ascii_output", "")
        lines = ascii_output.split("\n")

        print("\nLine-by-line analysis (first 15 lines):")
        print(f"{'Line':>4} | {'Spaces':>6} | Content (repr)")
        print("-" * 70)

        for i, line in enumerate(lines[:15]):
            spaces = len(line) - len(line.lstrip(" "))
            display = line[:45] + "..." if len(line) > 45 else line
            print(f"{i:>4} | {spaces:>6} | {display!r}")

        print("\n" + "=" * 80)
        print("RAW ASCII OUTPUT (repr - showing exact characters):")
        print("=" * 80)
        print(repr(ascii_output[:500]))

        print("\n" + "=" * 80)
        print("VISUAL ASCII OUTPUT:")
        print("=" * 80)
        print(ascii_output)

    def test_check_raw_output_if_available(self, results):
        """Check if raw_output field shows different whitespace than ascii_output."""
        print("\n" + "=" * 80)
        print("COMPARING RAW OUTPUT VS EXTRACTED ASCII")
        print("=" * 80)

        for result in results[:5]:  # Check first 5
            model_name = result.get("model_name", "Unknown")
            raw_output = result.get("raw_output")
            ascii_output = result.get("ascii_output", "")

            if not raw_output or not result.get("is_valid"):
                continue

            print(f"\n--- {model_name} ---")

            # Find the code block in raw output
            import re

            pattern = r"```(?:(?:text|ascii|plaintext)\s*\n|\n)(.*?)```"
            match = re.search(pattern, raw_output, re.DOTALL)

            if match:
                raw_content = match.group(1)
                raw_first_line = raw_content.split("\n")[0] if raw_content else ""
                extracted_first_line = ascii_output.split("\n")[0] if ascii_output else ""

                print(f"  Raw code block first line: {raw_first_line[:50]!r}")
                print(f"  Extracted first line:      {extracted_first_line[:50]!r}")

                if raw_first_line != extracted_first_line:
                    print("  >>> DIFFERENCE FOUND - extraction is changing the content!")
                else:
                    print("  (Same content - model output is the source of missing spaces)")
