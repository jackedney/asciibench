"""Integration tests for OpenRouter API calls."""

from os import environ

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
