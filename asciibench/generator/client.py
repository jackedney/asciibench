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

import litellm
from smolagents import ChatMessage, LiteLLMModel

from asciibench.common.config import GenerationConfig
from asciibench.common.models import OpenRouterResponse
from asciibench.common.retry import retry

# Suppress LiteLLM debug info (e.g., "Provider List: https://docs.litellm.ai/docs/providers")
litellm.suppress_debug_info = True  # type: ignore[invalid-assignment]


class LiteLLMModelWithCost(LiteLLMModel):
    """Custom LiteLLMModel subclass that extracts cost from LiteLLM responses.

    LiteLLM provides cost data via response._hidden_params['response_cost'].
    This subclass extracts that cost and makes it available via _litellm_cost attribute.
    """

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """Generate and extract cost from LiteLLM response."""
        chat_message = super().generate(
            messages, stop_sequences, response_format, tools_to_call_from, **kwargs
        )

        # Extract cost from LiteLLM's _hidden_params if available
        if chat_message.raw is not None:
            raw_response = chat_message.raw

            # LiteLLM stores cost in _hidden_params['response_cost']
            if hasattr(raw_response, "_hidden_params"):
                hidden_params = raw_response._hidden_params
                if hidden_params and "response_cost" in hidden_params:
                    cost = hidden_params["response_cost"]
                    # Store cost as an attribute on the raw response
                    raw_response._litellm_cost = cost

        return chat_message


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


class RateLimitError(OpenRouterClientError):
    """Raised when API rate limit is exceeded."""

    pass


class TransientError(OpenRouterClientError):
    """Raised for transient API errors that can be retried (5xx, connection errors)."""

    pass


class OpenRouterClient:
    """Client for interacting with OpenRouter API using smolagents LiteLLMModel."""

    def __init__(
        self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", timeout: int = 180
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

    @retry(
        max_retries=3,
        base_delay_seconds=1,
        retryable_exceptions=(RateLimitError, TransientError),
    )
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
            RateLimitError: When API rate limit is exceeded (retries 3 times)
            TransientError: For transient errors (502, 503, connection, timeout)
            AuthenticationError: When API authentication fails (invalid API key)
            ModelError: When the specified model is invalid or unavailable
            OpenRouterClientError: For other API errors
        """
        if config is None:
            config = GenerationConfig()

        try:
            # Initialize the model with LiteLLMModelWithCost for cost tracking
            # Disable retries to prevent indefinite retrying on rate limits
            # Auto-prepend 'openrouter/' prefix for LiteLLM
            litellm_model_id = f"openrouter/{model_id}"
            # Build model kwargs
            model_kwargs = {
                "model_id": litellm_model_id,
                "api_base": self.base_url,
                "api_key": self.api_key,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "client_kwargs": {"max_retries": 0, "timeout": self.timeout},
                "retry": False,
            }

            # Build extra_body for OpenRouter-specific parameters
            extra_body = {}

            # Add reasoning parameters if enabled (supported by o1, deepseek, kimi-k2)
            if config.reasoning:
                extra_body["reasoning"] = True
                if config.include_reasoning:
                    extra_body["include_reasoning"] = True

            # Add extra_body to model kwargs if not empty
            if extra_body:
                model_kwargs["extra_body"] = extra_body

            model = LiteLLMModelWithCost(**model_kwargs)

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
            text = ""
            if hasattr(response, "content") and response.content:
                text = str(response.content)
            elif hasattr(response, "raw") and response.raw is not None:
                # For reasoning models, content may be in reasoning_content
                raw = response.raw
                if hasattr(raw, "choices") and raw.choices:
                    message = raw.choices[0].message
                    # Try reasoning_content if content is empty
                    if hasattr(message, "reasoning_content") and message.reasoning_content:
                        text = str(message.reasoning_content)
                    elif hasattr(message, "content") and message.content:
                        text = str(message.content)

            if not text:
                text = str(response) if response else ""

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

            # Extract cost from raw response (set by LiteLLMModelWithCost)
            if hasattr(response, "raw") and response.raw is not None:
                raw = response.raw
                cost = getattr(raw, "_litellm_cost", None)

            return OpenRouterResponse(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=cost,
            )

        except APITimeoutError as e:
            raise TransientError(f"API call timed out after {self.timeout} seconds") from e
        except Exception as e:
            error_message = str(e).lower()

            # Check for rate limit errors (429)
            rate_limit_keywords = ["429", "rate limit", "too many requests"]
            if any(keyword in error_message for keyword in rate_limit_keywords):
                raise RateLimitError(f"Rate limit exceeded: {e}") from e

            # Check for transient errors (502, 503, connection errors)
            transient_keywords = [
                "502",
                "503",
                "bad gateway",
                "service unavailable",
                "connection",
                "timeout",
            ]
            if any(keyword in error_message for keyword in transient_keywords):
                raise TransientError(f"Transient error: {e}") from e

            # Check for authentication errors
            auth_keywords = ["401", "unauthorized", "invalid api key", "authentication"]
            if any(keyword in error_message for keyword in auth_keywords):
                raise AuthenticationError(f"Authentication failed: {e}") from e

            # Check for bad request errors (400)
            bad_request_keywords = ["400", "bad request"]
            if any(keyword in error_message for keyword in bad_request_keywords):
                raise OpenRouterClientError(f"Bad request: {e}") from e

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
            RateLimitError: When API rate limit is exceeded (retries 3 times)
            TransientError: For transient errors (502, 503, connection, timeout)
            AuthenticationError: When API authentication fails (invalid API key)
            ModelError: When the specified model is invalid or unavailable
            OpenRouterClientError: For other API errors
        """
        return await asyncio.to_thread(self.generate, model_id, prompt, config)
