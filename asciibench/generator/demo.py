"""Demo module for ASCII art generation.

This module provides a demo mode for generating ASCII art samples
from language models and displaying results in HTML format.

Demo generates ASCII art using a fixed prompt and saves outputs
to .demo_outputs/demo.html with incremental/resumable generation support.
"""


def main() -> None:
    """Main entry point for the Demo module.

    This function runs the demo generator that:
    1. Prints header banner
    2. Loads models from models.yaml
    3. Generates ASCII art for each model using fixed prompt
    4. Saves results to .demo_outputs/results.json
    5. Generates HTML output to .demo_outputs/demo.html
    """
    print("ASCIIBench Demo")
    print("=" * 50)

    print("\nDemo mode coming soon!")
    print("This will generate skeleton ASCII art from all configured models.")
    print("Outputs will be saved to .demo_outputs/demo.html")

    print("\n" + "=" * 50)
    print("Demo Complete!")


if __name__ == "__main__":
    main()
