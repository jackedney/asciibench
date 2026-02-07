"""VLM client for image analysis using OpenRouter API.

This module provides functionality to send images to Vision Language Models (VLMs)
and extract their responses, including cost tracking.

Dependencies:
    - httpx: Async HTTP client for API requests
"""

import base64

import httpx

from asciibench.common.config import Settings
from asciibench.common.logging import get_logger
from asciibench.common.retry import retry

logger = get_logger(__name__)


class ModelNotFoundError(Exception):
    """Raised when the specified VLM model ID is not found or invalid."""

    pass


class VLMClientError(Exception):
    """Base exception for VLM client errors."""

    pass


class AuthenticationError(VLMClientError):
    """Raised when API authentication fails."""

    pass


class RateLimitError(VLMClientError):
    """Raised when API rate limit is exceeded."""

    pass


class TransientError(VLMClientError):
    """Raised for transient API errors that can be retried (5xx, connection errors)."""

    pass


class VLMClient:
    """Client for interacting with OpenRouter Vision Language Models."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the VLM client.

        Args:
            settings: Settings object with API key and base URL configuration
        """
        self.api_key = settings.openrouter_api_key
        self.base_url = settings.base_url
        self.timeout = settings.openrouter_timeout_seconds

    @retry(
        max_retries=3,
        base_delay_seconds=1,
        retryable_exceptions=(RateLimitError, TransientError),
    )
    async def analyze_image(self, image_bytes: bytes, model_id: str) -> tuple[str, float]:
        """Analyze an image using a Vision Language Model.

        Args:
            image_bytes: PNG image bytes to analyze
            model_id: Model identifier (e.g., 'openai/gpt-4o')

        Returns:
            Tuple of (response_text, cost)

        Raises:
            ModelNotFoundError: When the specified model ID is invalid or unavailable
            RateLimitError: When API rate limit is exceeded (retries 3 times)
            TransientError: For transient errors (502, 503, connection, timeout)
            AuthenticationError: When API authentication fails (invalid API key)
            VLMClientError: For other API errors
        """
        prompt_text = "What does this image show? Describe the main subject in a few words."

        try:
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:image/png;base64,{base64_image}"

            payload = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )

            response.raise_for_status()
            data = response.json()

            response_text = ""
            cost = 0.0

            if data.get("choices") and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if choice.get("message") and choice["message"].get("content"):
                    response_text = choice["message"]["content"]

            if "usage" in data:
                usage = data["usage"]
                if "prompt_tokens" in usage and "completion_tokens" in usage:
                    prompt_tokens = usage["prompt_tokens"]
                    completion_tokens = usage["completion_tokens"]
                    if data.get("prompt_tokens_details") and data["prompt_tokens_details"].get(
                        "image_tokens"
                    ):
                        image_tokens = data["prompt_tokens_details"]["image_tokens"]
                        prompt_tokens += image_tokens
                    if data.get("completion_tokens_details"):
                        completion_details = data.get("completion_tokens_details")
                        if completion_details and "reasoning_tokens" in completion_details:
                            completion_tokens += completion_details["reasoning_tokens"]
                    cost = (prompt_tokens * 0.0000025 + completion_tokens * 0.00001) * 0.001

            logger.info(
                "VLM analysis complete",
                {
                    "model_id": model_id,
                    "response_text": response_text,
                    "cost": cost,
                },
            )

            return response_text, cost

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_text = e.response.text.lower() if e.response.text else ""

            if status_code == 429:
                raise RateLimitError(f"Rate limit exceeded: {e}") from e

            if status_code in (502, 503, 504):
                raise TransientError(f"Transient error ({status_code}): {e}") from e

            if (
                status_code == 401
                or "unauthorized" in error_text
                or "invalid api key" in error_text
            ):
                raise AuthenticationError(f"Authentication failed: {e}") from e

            if status_code == 404 or "not found" in error_text or "model" in error_text:
                raise ModelNotFoundError(f"Invalid model ID '{model_id}': {e}") from e

            raise VLMClientError(f"API error: {e}") from e

        except httpx.RequestError as e:
            raise TransientError(f"Request error: {e}") from e

        except Exception as e:
            error_message = str(e).lower()

            if "429" in error_message or "rate limit" in error_message:
                raise RateLimitError(f"Rate limit exceeded: {e}") from e

            if "404" in error_message or "not found" in error_message:
                raise ModelNotFoundError(f"Invalid model ID '{model_id}': {e}") from e

            if "unauthorized" in error_message or "authentication" in error_message:
                raise AuthenticationError(f"Authentication failed: {e}") from e

            raise VLMClientError(f"VLM client error: {e}") from e


def analyze_image(image_bytes: bytes, model_id: str) -> tuple[str, float]:
    """Convenience function to analyze an image using the default VLM client.

    This function creates a VLM client with default settings and analyzes the image.

    Args:
        image_bytes: PNG image bytes to analyze
        model_id: Model identifier (e.g., 'openai/gpt-4o')

    Returns:
        Tuple of (response_text, cost)

    Raises:
        ModelNotFoundError: When the specified model ID is invalid or unavailable
        RateLimitError: When API rate limit is exceeded (retries 3 times)
        TransientError: For transient errors (502, 503, connection, timeout)
        AuthenticationError: When API authentication fails (invalid API key)
        VLMClientError: For other API errors
    """
    import asyncio

    async def _analyze() -> tuple[str, float]:
        settings = Settings()
        client = VLMClient(settings)
        return await client.analyze_image(image_bytes, model_id)

    return asyncio.run(_analyze())
