"""OpenRouter API client using smolagents.

This module provides a client for interacting with the OpenRouter API
to generate ASCII art samples from language models.

Dependencies:
    - smolagents: Python agent library for LLM interactions
    - asciibench.common.config: Settings for API configuration
"""

import asyncio
import threading
from typing import Any

from smolagents import OpenAIModel

from asciibench.common.config import GenerationConfig
from asciibench.common.models import OpenRouterResponse


class APITimeoutError(Exception):
    """Raised when API call times out."""

    pass


def run_with_timeout(func, timeout_seconds: int) -> Any:
    """Run a function with a timeout.

    Args:
        func: Function to execute
        timeout_seconds: Maximum time to wait in seconds

    Returns:
        Result of the function

    Raises:
        APITimeoutError: If function doesn't complete within timeout
    """
    result = None
    exception = None

    def target():
        nonlocal result, exception
        try:
            result = func()
        except Exception as e:
            exception = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise APITimeoutError(f"API call timed out after {timeout_seconds} seconds")

    if exception is not None:
        raise exception

    return result


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

    def __init__(
        self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", timeout: int = 120
    ) -> None:
        """Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key for authentication
            base_url: Base URL for the OpenRouter API
            timeout: Timeout in seconds for API calls (default: 120)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    def generate(
        self,
        model_id: str,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> OpenRouterResponse:
        """Generate text response from the specified model.

        Args:
            model_id: Model identifier (e.g., 'openai/gpt-4o')
            prompt: Text prompt to send to the model
            config: Optional generation configuration with temperature, max_tokens, system_prompt

        Returns:
            OpenRouterResponse containing text and usage metadata
            (prompt_tokens, completion_tokens, total_tokens, cost)

        Raises:
            AuthenticationError: When API authentication fails (invalid API key)
            ModelError: When the specified model is invalid or unavailable
            OpenRouterClientError: For other API errors
        """
        if config is None:
            config = GenerationConfig()

        try:
            # Initialize the model with OpenRouter configuration
            # Disable retries to prevent indefinite retrying on rate limits
            model = OpenAIModel(
                model_id=model_id,
                api_base=self.base_url,
                api_key=self.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                client_kwargs={"max_retries": 0, "timeout": self.timeout},
                retry=False,  # Disable smolagents' built-in retry logic
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

            # Call the model with timeout to prevent indefinite hanging
            response = run_with_timeout(lambda: model(messages), self.timeout)

            # The response from smolagents model call is a ChatMessage object
            # Extract text content and usage metadata
            if hasattr(response, "content"):
                text = str(response.content)
            else:
                text = str(response)

            # Extract usage metadata from response
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            cost = None

            # Extract token usage from response.token_usage
            if hasattr(response, "token_usage") and response.token_usage is not None:
                token_usage = response.token_usage
                prompt_tokens = getattr(token_usage, "input_tokens", None)
                completion_tokens = getattr(token_usage, "output_tokens", None)
                total_tokens = getattr(token_usage, "total_tokens", None)

            # Extract cost from raw response (OpenRouter-specific field)
            if hasattr(response, "raw") and response.raw is not None:
                raw = response.raw
                if hasattr(raw, "usage") and raw.usage is not None:
                    cost = getattr(raw.usage, "total_cost", None)

            return OpenRouterResponse(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=cost,
            )

        except APITimeoutError as e:
            raise OpenRouterClientError(f"API call timed out after {self.timeout} seconds") from e
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

    async def generate_async(
        self,
        model_id: str,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> OpenRouterResponse:
        """Async version of generate that runs the sync call in a thread pool.

        Args:
            model_id: Model identifier (e.g., 'openai/gpt-4o')
            prompt: Text prompt to send to the model
            config: Optional generation configuration with temperature, max_tokens, system_prompt

        Returns:
            OpenRouterResponse containing text and usage metadata
            (prompt_tokens, completion_tokens, total_tokens, cost)

        Raises:
            AuthenticationError: When API authentication fails (invalid API key)
            ModelError: When the specified model is invalid or unavailable
            OpenRouterClientError: For other API errors
        """
        return await asyncio.to_thread(self.generate, model_id, prompt, config)
