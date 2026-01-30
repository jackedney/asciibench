"""OpenRouter API client using smolagents.

This module provides a client for interacting with the OpenRouter API
to generate ASCII art samples from language models.

Dependencies:
    - smolagents: Python agent library for LLM interactions
    - asciibench.common.config: Settings for API configuration
"""

from smolagents import OpenAIModel

from asciibench.common.config import GenerationConfig


class OpenRouterClientError(Exception):
    """Base exception for OpenRouter client errors."""

    pass


class AuthenticationError(OpenRouterClientError):
    """Raised when API authentication fails."""

    pass


class ModelError(OpenRouterClientError):
    """Raised when there's an issue with the specified model."""

    pass


class OpenRouterClient:
    """Client for interacting with OpenRouter API using smolagents OpenAIModel."""

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> None:
        """Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key for authentication
            base_url: Base URL for the OpenRouter API
        """
        self.api_key = api_key
        self.base_url = base_url

    def generate(
        self,
        model_id: str,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate text response from the specified model.

        Args:
            model_id: Model identifier (e.g., 'openai/gpt-4o')
            prompt: Text prompt to send to the model
            config: Optional generation configuration with temperature, max_tokens, system_prompt

        Returns:
            Generated text response from the model

        Raises:
            AuthenticationError: When API authentication fails (invalid API key)
            ModelError: When the specified model is invalid or unavailable
            OpenRouterClientError: For other API errors
        """
        if config is None:
            config = GenerationConfig()

        try:
            # Initialize the model with OpenRouter configuration
            model = OpenAIModel(
                model_id=model_id,
                api_base=self.base_url,
                api_key=self.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            # Build messages list
            messages: list[dict] = []

            # Add system prompt if provided
            if config.system_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": config.system_prompt}],
                    }
                )

            # Add user prompt
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            )

            # Call the model and get response
            response = model(messages)

            # The response from smolagents model call is a ChatMessage object
            # We need to extract the text content
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)

        except Exception as e:
            error_message = str(e).lower()

            # Check for authentication errors
            auth_keywords = ["401", "unauthorized", "invalid api key", "authentication"]
            if any(keyword in error_message for keyword in auth_keywords):
                raise AuthenticationError(f"Authentication failed: {e}") from e

            # Check for model-related errors
            model_keywords = ["not found", "invalid", "does not exist"]
            is_model_error = "404" in error_message or (
                "model" in error_message
                and any(keyword in error_message for keyword in model_keywords)
            )
            if is_model_error:
                raise ModelError(f"Invalid model ID '{model_id}': {e}") from e

            # Re-raise as generic client error
            raise OpenRouterClientError(f"API error: {e}") from e
