"""Integration tests for OpenRouter API calls."""

from os import environ
from unittest.mock import MagicMock, patch

import pytest

from asciibench.generator.client import OpenRouterClient

# Skip test if OPENROUTER_API_KEY is not set
pytestmark = pytest.mark.skipif(
    "OPENROUTER_API_KEY" not in environ,
    reason="OPENROUTER_API_KEY environment variable is not set",
)


class TestCostTracking:
    """Tests for cost tracking with real API calls."""

    def test_generate_returns_cost_greater_than_zero(self):
        """Generate returns cost greater than 0 when calling real OpenRouter API."""
        api_key = environ["OPENROUTER_API_KEY"]
        client = OpenRouterClient(api_key=api_key)

        response = client.generate(
            model_id="openai/gpt-oss-120b",
            prompt="Say hello in exactly 3 words",
        )

        assert response.text is not None
        assert isinstance(response.cost, (int, float))
        assert response.cost > 0.0


class TestLoaderCostPassing:
    """Tests for passing cost from generation to loader.complete()."""

    @patch("asciibench.generator.demo.generate_demo_sample")
    def test_demo_passes_cost_to_loader_complete(self, mock_generate):
        """Demo passes result.cost to loader.complete() on success."""
        from datetime import datetime

        from asciibench.common.models import DemoResult
        from asciibench.generator.demo import main as demo_main

        mock_result = DemoResult(
            model_id="openai/gpt-4o-mini",
            model_name="GPT-4o Mini",
            ascii_output="test",
            is_valid=True,
            cost=0.005,
            timestamp=datetime.now(),
        )
        mock_generate.return_value = mock_result

        with patch("asciibench.generator.demo.load_models") as mock_load:
            from asciibench.common.models import Model

            mock_load.return_value = [
                Model(
                    id="openai/gpt-4o-mini",
                    name="GPT-4o Mini",
                )
            ]

            with patch("asciibench.generator.demo.load_demo_results") as mock_load_results:
                mock_load_results.return_value = []
                with patch("asciibench.generator.demo.get_completed_model_ids") as mock_ids:
                    mock_ids.return_value = set()
                    with patch("asciibench.generator.demo.create_loader") as mock_create_loader:
                        mock_loader = MagicMock()
                        mock_create_loader.return_value = mock_loader

                        demo_main()

                        mock_loader.complete.assert_called_once_with(success=True, cost=0.005)

    @patch("asciibench.generator.demo.generate_demo_sample")
    def test_demo_passes_zero_cost_on_failure(self, mock_generate):
        """Demo passes cost=0.0 to loader.complete() on failure."""
        from datetime import datetime

        from asciibench.common.models import DemoResult
        from asciibench.generator.demo import main as demo_main

        mock_result = DemoResult(
            model_id="openai/gpt-4o-mini",
            model_name="GPT-4o Mini",
            ascii_output="Error: test",
            is_valid=False,
            cost=None,
            timestamp=datetime.now(),
        )
        mock_generate.return_value = mock_result

        with patch("asciibench.generator.demo.load_models") as mock_load:
            from asciibench.common.models import Model

            mock_load.return_value = [
                Model(
                    id="openai/gpt-4o-mini",
                    name="GPT-4o Mini",
                )
            ]

            with patch("asciibench.generator.demo.load_demo_results") as mock_load_results:
                mock_load_results.return_value = []
                with patch("asciibench.generator.demo.get_completed_model_ids") as mock_ids:
                    mock_ids.return_value = set()
                    with patch("asciibench.generator.demo.create_loader") as mock_create_loader:
                        mock_loader = MagicMock()
                        mock_create_loader.return_value = mock_loader

                        demo_main()

                        mock_loader.complete.assert_called_once_with(success=False, cost=0.0)
