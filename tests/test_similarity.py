"""Tests for the similarity module."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from asciibench.evaluator.similarity import (
    AuthenticationError,
    EmbeddingClient,
    RateLimitError,
    TransientError,
    compute_similarity,
)


class TestComputeCosineSimilarity:
    """Tests for _compute_cosine_similarity method."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1.0."""
        client = EmbeddingClient(MagicMock())
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0, 3.0]

        result = client._compute_cosine_similarity(vec1, vec2)

        assert result == pytest.approx(1.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity of -1.0."""
        client = EmbeddingClient(MagicMock())
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]

        result = client._compute_cosine_similarity(vec1, vec2)

        assert result == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity of 0.0."""
        client = EmbeddingClient(MagicMock())
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]

        result = client._compute_cosine_similarity(vec1, vec2)

        assert result == pytest.approx(0.0)

    def test_zero_vector_first(self):
        """Zero vector in first argument returns 0.0."""
        client = EmbeddingClient(MagicMock())
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]

        result = client._compute_cosine_similarity(vec1, vec2)

        assert result == 0.0

    def test_zero_vector_second(self):
        """Zero vector in second argument returns 0.0."""
        client = EmbeddingClient(MagicMock())
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [0.0, 0.0, 0.0]

        result = client._compute_cosine_similarity(vec1, vec2)

        assert result == 0.0

    def test_high_similarity_vectors(self):
        """Vectors with similar direction should have high similarity."""
        client = EmbeddingClient(MagicMock())
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.1, 2.1, 3.1]

        result = client._compute_cosine_similarity(vec1, vec2)

        assert result > 0.99

    def test_low_similarity_vectors(self):
        """Vectors with different direction should have low similarity."""
        client = EmbeddingClient(MagicMock())
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 1.0]

        result = client._compute_cosine_similarity(vec1, vec2)

        assert result == pytest.approx(0.0)


class TestGetEmbedding:
    """Tests for _get_embedding method."""

    @pytest.mark.asyncio
    async def test_successful_embedding_request(self):
        """Successful API request returns embedding vector."""
        settings_mock = MagicMock()
        settings_mock.openrouter_api_key = "test-key"
        settings_mock.base_url = "https://api.test.com"
        settings_mock.openrouter_timeout_seconds = 120

        client = EmbeddingClient(settings_mock)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client._get_embedding("test text")

            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_rate_limit_error(self):
        """Rate limit error raises RateLimitError."""
        import httpx

        settings_mock = MagicMock()
        settings_mock.openrouter_api_key = "test-key"
        settings_mock.base_url = "https://api.test.com"
        settings_mock.openrouter_timeout_seconds = 120

        client = EmbeddingClient(settings_mock)

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate limit exceeded",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(RateLimitError, match="Rate limit exceeded"):
                await client._get_embedding("test text")

    @pytest.mark.asyncio
    async def test_transient_error_502(self):
        """502 error raises TransientError."""
        settings_mock = MagicMock()
        settings_mock.openrouter_api_key = "test-key"
        settings_mock.base_url = "https://api.test.com"
        settings_mock.openrouter_timeout_seconds = 120

        client = EmbeddingClient(settings_mock)

        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.text = "Bad gateway"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad gateway",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(TransientError, match="Transient error"):
                await client._get_embedding("test text")

    @pytest.mark.asyncio
    async def test_authentication_error(self):
        """Authentication error raises AuthenticationError."""
        import httpx

        settings_mock = MagicMock()
        settings_mock.openrouter_api_key = "test-key"
        settings_mock.base_url = "https://api.test.com"
        settings_mock.openrouter_timeout_seconds = 120

        client = EmbeddingClient(settings_mock)

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(AuthenticationError, match="Authentication failed"):
                await client._get_embedding("test text")


class TestComputeSimilarity:
    """Tests for compute_similarity method."""

    @pytest.mark.asyncio
    async def test_empty_text1(self):
        """Empty string in text1 returns 0.0 similarity."""
        settings_mock = MagicMock()
        client = EmbeddingClient(settings_mock)

        result = await client.compute_similarity("", "cat")

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_empty_text2(self):
        """Empty string in text2 returns 0.0 similarity."""
        settings_mock = MagicMock()
        client = EmbeddingClient(settings_mock)

        result = await client.compute_similarity("cat", "")

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_both_empty_strings(self):
        """Both empty strings return 0.0 similarity."""
        settings_mock = MagicMock()
        client = EmbeddingClient(settings_mock)

        result = await client.compute_similarity("", "")

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_identical_strings(self):
        """Identical strings return 1.0 similarity without API call."""
        settings_mock = MagicMock()
        client = EmbeddingClient(settings_mock)

        result = await client.compute_similarity("cat", "cat")

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_caches_embedding_for_text1(self):
        """Embedding for text1 is cached after first call."""
        settings_mock = MagicMock()
        settings_mock.openrouter_api_key = "test-key"
        settings_mock.base_url = "https://api.test.com"
        settings_mock.openrouter_timeout_seconds = 120

        client = EmbeddingClient(settings_mock)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            await client.compute_similarity("cat", "kitty")

            assert "cat" in client._cache

    @pytest.mark.asyncio
    async def test_caches_embedding_for_text2(self):
        """Embedding for text2 is cached after first call."""
        settings_mock = MagicMock()
        settings_mock.openrouter_api_key = "test-key"
        settings_mock.base_url = "https://api.test.com"
        settings_mock.openrouter_timeout_seconds = 120

        client = EmbeddingClient(settings_mock)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            await client.compute_similarity("cat", "kitty")

            assert "kitty" in client._cache

    @pytest.mark.asyncio
    async def test_uses_cached_embedding(self):
        """Cached embedding is reused for repeated text."""
        settings_mock = MagicMock()
        settings_mock.openrouter_api_key = "test-key"
        settings_mock.base_url = "https://api.test.com"
        settings_mock.openrouter_timeout_seconds = 120

        client = EmbeddingClient(settings_mock)
        client._cache["cat"] = [0.1, 0.2, 0.3]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.2, 0.3, 0.4]}]}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            await client.compute_similarity("cat", "kitty")

            assert "cat" in client._cache
            assert "kitty" in client._cache

    @pytest.mark.asyncio
    async def test_computes_similarity_between_different_texts(self):
        """Computes similarity between two different texts."""
        settings_mock = MagicMock()
        settings_mock.openrouter_api_key = "test-key"
        settings_mock.base_url = "https://api.test.com"
        settings_mock.openrouter_timeout_seconds = 120

        client = EmbeddingClient(settings_mock)

        mock_response = MagicMock()
        mock_response.status_code = 200

        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]

        mock_response.json.side_effect = [
            {"data": [{"embedding": embedding1}]},
            {"data": [{"embedding": embedding2}]},
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.compute_similarity("text1", "text2")

            assert result == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_high_similarity_case(self):
        """Similar texts produce high similarity scores."""
        settings_mock = MagicMock()
        settings_mock.openrouter_api_key = "test-key"
        settings_mock.base_url = "https://api.test.com"
        settings_mock.openrouter_timeout_seconds = 120

        client = EmbeddingClient(settings_mock)

        mock_response = MagicMock()
        mock_response.status_code = 200

        embedding1 = [1.0, 2.0, 3.0]
        embedding2 = [1.1, 2.1, 3.1]

        mock_response.json.side_effect = [
            {"data": [{"embedding": embedding1}]},
            {"data": [{"embedding": embedding2}]},
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.compute_similarity("cat", "kitty")

            assert result > 0.99


class TestComputeSimilarityConvenience:
    """Tests for compute_similarity convenience function."""

    @pytest.mark.asyncio
    async def test_convenience_function_with_empty_string(self):
        """Convenience function handles empty string input."""
        with patch("asciibench.evaluator.similarity.EmbeddingClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.compute_similarity = AsyncMock(return_value=0.0)
            mock_client_class.return_value = mock_client

            result = await compute_similarity("", "cat")

            assert result == 0.0

    @pytest.mark.asyncio
    async def test_convenience_function_creates_client(self):
        """Convenience function creates EmbeddingClient with default settings."""
        with patch("asciibench.evaluator.similarity.Settings"):
            with patch("asciibench.evaluator.similarity.EmbeddingClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.compute_similarity = AsyncMock(return_value=0.8)
                mock_client_class.return_value = mock_client

                result = await compute_similarity("cat", "kitty")

                mock_client_class.assert_called_once()
                mock_client.compute_similarity.assert_called_once_with("cat", "kitty")
                assert result == 0.8


class TestSimilarityExamples:
    """Example tests from acceptance criteria."""

    @pytest.mark.asyncio
    async def test_cat_kitty_high_similarity(self):
        """Example: compute_similarity('cat', 'kitty') returns >= 0.8."""
        settings_mock = MagicMock()
        settings_mock.openrouter_api_key = "test-key"
        settings_mock.base_url = "https://api.test.com"
        settings_mock.openrouter_timeout_seconds = 120

        client = EmbeddingClient(settings_mock)

        mock_response = MagicMock()
        mock_response.status_code = 200

        embedding1 = [1.0, 1.0, 1.0]
        embedding2 = [1.0, 1.0, 1.0]

        mock_response.json.side_effect = [
            {"data": [{"embedding": embedding1}]},
            {"data": [{"embedding": embedding2}]},
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.compute_similarity("cat", "kitty")

            assert result >= 0.8

    @pytest.mark.asyncio
    async def test_cat_dog_low_similarity(self):
        """Example: compute_similarity('cat', 'dog') returns <= 0.5."""
        settings_mock = MagicMock()
        settings_mock.openrouter_api_key = "test-key"
        settings_mock.base_url = "https://api.test.com"
        settings_mock.openrouter_timeout_seconds = 120

        client = EmbeddingClient(settings_mock)

        mock_response = MagicMock()
        mock_response.status_code = 200

        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]

        mock_response.json.side_effect = [
            {"data": [{"embedding": embedding1}]},
            {"data": [{"embedding": embedding2}]},
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.compute_similarity("cat", "dog")

            assert result <= 0.5

    @pytest.mark.asyncio
    async def test_empty_string_returns_zero(self):
        """Negative case: empty string input returns 0.0 similarity."""
        settings_mock = MagicMock()
        client = EmbeddingClient(settings_mock)

        result = await client.compute_similarity("", "cat")

        assert result == 0.0
