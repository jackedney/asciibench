"""Tests for the OpenRouter client module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from asciibench.common.config import GenerationConfig
from asciibench.common.models import OpenRouterResponse
from asciibench.generator.client import (
    AuthenticationError,
    ModelError,
    OpenRouterClient,
    OpenRouterClientError,
    RateLimitError,
    TransientError,
)


class TestOpenRouterClientInit:
    """Tests for OpenRouterClient initialization."""

    def test_init_with_defaults(self):
        """Initialize client with default base URL."""
        client = OpenRouterClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.base_url == "https://openrouter.ai/api/v1"

    def test_init_with_custom_base_url(self):
        """Initialize client with custom base URL."""
        client = OpenRouterClient(api_key="test-key", base_url="https://custom.api/v1")
        assert client.api_key == "test-key"
        assert client.base_url == "https://custom.api/v1"


class TestOpenRouterClientGenerate:
    """Tests for OpenRouterClient.generate method."""

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_returns_text_response(self, mock_model_class):
        """Generate returns OpenRouterResponse with text from model."""
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "```\n/_/\\\n( o.o )\n```"
        mock_response.token_usage = None
        mock_response.raw = None
        mock_model_instance.return_value = mock_response
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")
        result = client.generate("openai/gpt-4o", "Draw a cat")

        assert isinstance(result, OpenRouterResponse)
        assert result.text == "```\n/_/\\\n( o.o )\n```"
        assert result.prompt_tokens is None
        assert result.completion_tokens is None
        assert result.total_tokens is None
        assert result.cost is None
        mock_model_class.assert_called_once_with(
            model_id="openrouter/openai/gpt-4o",
            api_base="https://openrouter.ai/api/v1",
            api_key="test-key",
            temperature=0.0,
            max_tokens=1000,
            client_kwargs={"max_retries": 0, "timeout": 180},
            retry=False,
        )

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_with_custom_config(self, mock_model_class):
        """Generate uses custom configuration settings."""
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Generated art"
        mock_response.token_usage = None
        mock_response.raw = None
        mock_model_instance.return_value = mock_response
        mock_model_class.return_value = mock_model_instance

        config = GenerationConfig(
            temperature=0.7, max_tokens=500, system_prompt="You are an artist"
        )
        client = OpenRouterClient(api_key="test-key")
        result = client.generate("anthropic/claude-3-opus", "Draw a tree", config=config)

        assert isinstance(result, OpenRouterResponse)
        assert result.text == "Generated art"
        mock_model_class.assert_called_once_with(
            model_id="openrouter/anthropic/claude-3-opus",
            api_base="https://openrouter.ai/api/v1",
            api_key="test-key",
            temperature=0.7,
            max_tokens=500,
            client_kwargs={"max_retries": 0, "timeout": 180},
            retry=False,
        )

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_includes_system_prompt(self, mock_model_class):
        """Generate includes system prompt in messages when provided."""
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Art output"
        mock_response.token_usage = None
        mock_response.raw = None
        mock_model_instance.return_value = mock_response
        mock_model_class.return_value = mock_model_instance

        config = GenerationConfig(system_prompt="Create ASCII art only")
        client = OpenRouterClient(api_key="test-key")
        client.generate("openai/gpt-4o", "Draw a dog", config=config)

        call_args = mock_model_instance.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"][0]["text"] == "Create ASCII art only"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"][0]["text"] == "Draw a dog"

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_without_system_prompt(self, mock_model_class):
        """Generate sends only user message when no system prompt."""
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Art output"
        mock_response.token_usage = None
        mock_response.raw = None
        mock_model_instance.return_value = mock_response
        mock_model_class.return_value = mock_model_instance

        config = GenerationConfig(system_prompt="")
        client = OpenRouterClient(api_key="test-key")
        client.generate("openai/gpt-4o", "Draw a dog", config=config)

        call_args = mock_model_instance.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_handles_string_response(self, mock_model_class):
        """Generate handles response that is already a string."""
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = "Direct string response"
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")
        result = client.generate("openai/gpt-4o", "Draw a cat")

        assert isinstance(result, OpenRouterResponse)
        assert result.text == "Direct string response"


class TestOpenRouterClientErrors:
    """Tests for error handling in OpenRouterClient."""

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_raises_authentication_error_on_401(self, mock_model_class):
        """Generate raises AuthenticationError on 401 response."""
        mock_model_instance = MagicMock()
        mock_model_instance.side_effect = Exception("401 Unauthorized: Invalid API key")
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="invalid-key")
        with pytest.raises(AuthenticationError) as exc_info:
            client.generate("openai/gpt-4o", "Draw a cat")

        assert "Authentication failed" in str(exc_info.value)

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_raises_authentication_error_on_unauthorized(self, mock_model_class):
        """Generate raises AuthenticationError when unauthorized."""
        mock_model_instance = MagicMock()
        mock_model_instance.side_effect = Exception("Unauthorized access")
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="bad-key")
        with pytest.raises(AuthenticationError):
            client.generate("openai/gpt-4o", "Draw a cat")

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_raises_model_error_on_404(self, mock_model_class):
        """Generate raises ModelError on 404 response."""
        mock_model_instance = MagicMock()
        mock_model_instance.side_effect = Exception("404 Not Found: Model not found")
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")
        with pytest.raises(ModelError) as exc_info:
            client.generate("invalid/model-id", "Draw a cat")

        assert "Invalid model ID" in str(exc_info.value)
        assert "invalid/model-id" in str(exc_info.value)

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_raises_model_error_on_model_not_found(self, mock_model_class):
        """Generate raises ModelError when model does not exist."""
        mock_model_instance = MagicMock()
        mock_model_instance.side_effect = Exception("Model does not exist: fake-model")
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")
        with pytest.raises(ModelError):
            client.generate("fake/model", "Draw a cat")

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_raises_transient_error_on_connection_timeout(self, mock_model_class):
        """Generate raises TransientError on connection timeout (retryable)."""
        mock_model_instance = MagicMock()
        mock_model_instance.side_effect = Exception("Connection timeout")
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")
        from asciibench.generator.client import TransientError

        with pytest.raises(TransientError) as exc_info:
            client.generate("openai/gpt-4o", "Draw a cat")

        assert "Transient error" in str(exc_info.value)
        assert "Connection timeout" in str(exc_info.value)

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_raises_rate_limit_error_on_429(self, mock_model_class):
        """Generate raises RateLimitError on 429 response (retryable)."""
        mock_model_instance = MagicMock()
        mock_model_instance.side_effect = Exception("429 Too Many Requests")
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")
        from asciibench.generator.client import RateLimitError

        with pytest.raises(RateLimitError) as exc_info:
            client.generate("openai/gpt-4o", "Draw a cat")

        assert "Rate limit exceeded" in str(exc_info.value)

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_raises_transient_error_on_503(self, mock_model_class):
        """Generate raises TransientError on 503 response (service unavailable)."""
        mock_model_instance = MagicMock()
        mock_model_instance.side_effect = Exception("503 Service Unavailable")
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")
        with pytest.raises(TransientError) as exc_info:
            client.generate("openai/gpt-4o", "Draw a cat")

        assert "Transient error" in str(exc_info.value)
        assert "503" in str(exc_info.value)

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_raises_transient_error_on_502(self, mock_model_class):
        """Generate raises TransientError on 502 response (bad gateway)."""
        mock_model_instance = MagicMock()
        mock_model_instance.side_effect = Exception("502 Bad Gateway")
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")
        with pytest.raises(TransientError) as exc_info:
            client.generate("openai/gpt-4o", "Draw a cat")

        assert "Transient error" in str(exc_info.value)
        assert "502" in str(exc_info.value)

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_raises_client_error_on_400(self, mock_model_class):
        """Generate raises OpenRouterClientError on 400 response (not retryable)."""
        mock_model_instance = MagicMock()
        mock_model_instance.side_effect = Exception("400 Bad Request")
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")
        with pytest.raises(OpenRouterClientError) as exc_info:
            client.generate("openai/gpt-4o", "Draw a cat")

        assert "Bad request" in str(exc_info.value)
        assert "400" in str(exc_info.value)


class TestOpenRouterClientRetryBehavior:
    """Tests for retry behavior with rate limit and transient errors."""

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_rate_limit_retries_3_times_then_raises(self, mock_model_class):
        """429 response retries 3 times before raising RateLimitError."""
        mock_model_instance = MagicMock()
        mock_model_instance.side_effect = Exception("429 Too Many Requests")
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")

        start_time = time.time()
        with pytest.raises(RateLimitError):
            client.generate("openai/gpt-4o", "Draw a cat")
        elapsed_time = time.time() - start_time

        # Should have been called 4 times (initial + 3 retries)
        assert mock_model_instance.call_count == 4

        # Should have waited at least 1 + 2 + 4 = 7 seconds (with some tolerance)
        assert elapsed_time >= 7 * 0.8

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_transient_error_retries_3_times_then_raises(self, mock_model_class):
        """503 response retries 3 times before raising TransientError."""
        mock_model_instance = MagicMock()
        mock_model_instance.side_effect = Exception("503 Service Unavailable")
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")

        with pytest.raises(TransientError):
            client.generate("openai/gpt-4o", "Draw a cat")

        # Should have been called 4 times (initial + 3 retries)
        assert mock_model_instance.call_count == 4

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_authentication_error_does_not_retry(self, mock_model_class):
        """401 raises AuthenticationError immediately without retry."""
        mock_model_instance = MagicMock()
        mock_model_instance.side_effect = Exception("401 Unauthorized")
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")

        with pytest.raises(AuthenticationError):
            client.generate("openai/gpt-4o", "Draw a cat")

        # Should have been called only once (no retries)
        assert mock_model_instance.call_count == 1

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_model_error_does_not_retry(self, mock_model_class):
        """404 raises ModelError immediately without retry."""
        mock_model_instance = MagicMock()
        mock_model_instance.side_effect = Exception("404 Model not found")
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")

        with pytest.raises(ModelError):
            client.generate("openai/gpt-4o", "Draw a cat")

        # Should have been called only once (no retries)
        assert mock_model_instance.call_count == 1

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_bad_request_error_does_not_retry(self, mock_model_class):
        """400 raises OpenRouterClientError immediately without retry."""
        mock_model_instance = MagicMock()
        mock_model_instance.side_effect = Exception("400 Bad Request")
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")

        with pytest.raises(OpenRouterClientError):
            client.generate("openai/gpt-4o", "Draw a cat")

        # Should have been called only once (no retries)
        assert mock_model_instance.call_count == 1

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_rate_limit_succeeds_on_third_retry(self, mock_model_class):
        """429 response succeeds after 2 retries (3rd attempt)."""
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Generated art"
        mock_response.token_usage = None
        mock_response.raw = None

        # Fail first 2 times, succeed on 3rd
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("429 Too Many Requests")
            return mock_response

        mock_model_instance.side_effect = side_effect
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")
        result = client.generate("openai/gpt-4o", "Draw a cat")

        # Should have succeeded on 3rd attempt
        assert result.text == "Generated art"
        assert mock_model_instance.call_count == 3


class TestOpenRouterClientUsageMetadata:
    """Tests for usage and cost metadata extraction."""

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_returns_usage_metadata_with_cost(self, mock_model_class):
        """Generate returns OpenRouterResponse with usage metadata and cost."""
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "ASCII art"
        mock_token_usage = MagicMock()
        mock_token_usage.input_tokens = 10
        mock_token_usage.output_tokens = 20
        mock_token_usage.total_tokens = 30
        mock_response.token_usage = mock_token_usage
        mock_raw = MagicMock()
        mock_raw._litellm_cost = 0.001234
        mock_response.raw = mock_raw
        mock_model_instance.return_value = mock_response
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")
        result = client.generate("openai/gpt-4o", "Draw a cat")

        assert isinstance(result, OpenRouterResponse)
        assert result.text == "ASCII art"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.total_tokens == 30
        assert result.cost == 0.001234

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_returns_none_for_missing_usage(self, mock_model_class):
        """Generate returns OpenRouterResponse with None values when usage metadata is missing."""
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "ASCII art"
        mock_response.token_usage = None
        mock_response.raw = None
        mock_model_instance.return_value = mock_response
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")
        result = client.generate("openai/gpt-4o", "Draw a cat")

        assert isinstance(result, OpenRouterResponse)
        assert result.text == "ASCII art"
        assert result.prompt_tokens is None
        assert result.completion_tokens is None
        assert result.total_tokens is None
        assert result.cost is None

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_returns_none_for_missing_cost(self, mock_model_class):
        """Generate returns OpenRouterResponse with None cost when _litellm_cost is missing."""
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "ASCII art"
        mock_token_usage = MagicMock()
        mock_token_usage.input_tokens = 10
        mock_token_usage.output_tokens = 20
        mock_token_usage.total_tokens = 30
        mock_response.token_usage = mock_token_usage
        mock_raw = MagicMock()
        del mock_raw._litellm_cost
        mock_response.raw = mock_raw
        mock_model_instance.return_value = mock_response
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")
        result = client.generate("openai/gpt-4o", "Draw a cat")

        assert isinstance(result, OpenRouterResponse)
        assert result.text == "ASCII art"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.total_tokens == 30
        assert result.cost is None

    @patch("asciibench.generator.client.LiteLLMModelWithCost")
    def test_generate_returns_none_when_raw_is_none(self, mock_model_class):
        """Generate returns OpenRouterResponse with None cost when raw is None."""
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "ASCII art"
        mock_token_usage = MagicMock()
        mock_token_usage.input_tokens = 10
        mock_token_usage.output_tokens = 20
        mock_token_usage.total_tokens = 30
        mock_response.token_usage = mock_token_usage
        mock_response.raw = None
        mock_model_instance.return_value = mock_response
        mock_model_class.return_value = mock_model_instance

        client = OpenRouterClient(api_key="test-key")
        result = client.generate("openai/gpt-4o", "Draw a cat")

        assert isinstance(result, OpenRouterResponse)
        assert result.text == "ASCII art"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.total_tokens == 30
        assert result.cost is None


class TestOpenRouterClientExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_authentication_error_is_client_error(self):
        """AuthenticationError inherits from OpenRouterClientError."""
        assert issubclass(AuthenticationError, OpenRouterClientError)

    def test_model_error_is_client_error(self):
        """ModelError inherits from OpenRouterClientError."""
        assert issubclass(ModelError, OpenRouterClientError)

    def test_rate_limit_error_is_client_error(self):
        """RateLimitError inherits from OpenRouterClientError."""
        assert issubclass(RateLimitError, OpenRouterClientError)

    def test_transient_error_is_client_error(self):
        """TransientError inherits from OpenRouterClientError."""
        assert issubclass(TransientError, OpenRouterClientError)

    def test_client_error_is_exception(self):
        """OpenRouterClientError inherits from Exception."""
        assert issubclass(OpenRouterClientError, Exception)
