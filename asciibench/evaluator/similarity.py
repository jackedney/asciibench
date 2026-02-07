"""Semantic similarity matching using OpenAI embeddings via OpenRouter.

This module provides functionality to compute semantic similarity between two text strings
using OpenAI's text-embedding-3-small model through OpenRouter API.

Dependencies:
    - httpx: Async HTTP client for API requests
"""

from typing import ClassVar

import httpx

from asciibench.common.config import Settings
from asciibench.common.logging import get_logger
from asciibench.common.retry import retry

logger = get_logger(__name__)


class EmbeddingClientError(Exception):
    """Base exception for embedding client errors."""

    pass


class AuthenticationError(EmbeddingClientError):
    """Raised when API authentication fails."""

    pass


class RateLimitError(EmbeddingClientError):
    """Raised when API rate limit is exceeded."""

    pass


class TransientError(EmbeddingClientError):
    """Raised for transient API errors that can be retried (5xx, connection errors)."""

    pass


class EmbeddingClient:
    """Client for computing text embeddings using OpenAI via OpenRouter."""

    EMBEDDING_MODEL: ClassVar[str] = "openai/text-embedding-3-small"

    def __init__(self, settings: Settings) -> None:
        """Initialize the embedding client.

        Args:
            settings: Settings object with API key and base URL configuration
        """
        self.api_key = settings.openrouter_api_key
        self.base_url = settings.base_url
        self.timeout = settings.openrouter_timeout_seconds
        self._cache: dict[str, list[float]] = {}

    def _compute_cosine_similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score in range [0, 1]
        """
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2, strict=True))

        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @retry(
        max_retries=3,
        base_delay_seconds=1,
        retryable_exceptions=(RateLimitError, TransientError),
    )
    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            RateLimitError: When API rate limit is exceeded (retries 3 times)
            TransientError: For transient errors (502, 503, connection, timeout)
            AuthenticationError: When API authentication fails (invalid API key)
            EmbeddingClientError: For other API errors
        """
        response = None
        try:
            payload = {
                "model": self.EMBEDDING_MODEL,
                "input": text,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=payload,
                )

            response.raise_for_status()
            data = response.json()

            if data.get("data") and len(data["data"]) > 0:
                embedding = data["data"][0].get("embedding", [])
                return embedding

            raise EmbeddingClientError("No embedding returned from API")

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

            raise EmbeddingClientError(f"API error: {e}") from e

        except httpx.RequestError as e:
            raise TransientError(f"Request error: {e}") from e

        except Exception as e:
            error_message = str(e).lower()

            if "429" in error_message or "rate limit" in error_message:
                raise RateLimitError(f"Rate limit exceeded: {e}") from e

            if "unauthorized" in error_message or "authentication" in error_message:
                raise AuthenticationError(f"Authentication failed: {e}") from e

            raise EmbeddingClientError(f"Embedding client error: {e}") from e

    async def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two text strings.

        Uses cosine similarity between embeddings. Caches embeddings to reduce
        API costs for repeated texts.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Similarity score in range [0, 1], where 1 means identical meaning

        Examples:
            >>> similarity = await client.compute_similarity('cat', 'kitty')
            >>> similarity >= 0.8
            True
            >>> similarity = await client.compute_similarity('cat', 'dog')
            >>> similarity <= 0.5
            True

        Negative case:
            Empty string input returns 0.0 similarity
        """
        if not text1 or not text2:
            logger.info(
                "Empty string input, returning 0.0 similarity",
                {"text1": text1, "text2": text2},
            )
            return 0.0

        if text1 == text2:
            logger.debug(
                "Identical strings, returning 1.0 similarity",
                {"text": text1},
            )
            return 1.0

        if text1 in self._cache:
            embedding1 = self._cache[text1]
            logger.debug(
                "Using cached embedding for text1",
                {"text": text1[:50] if len(text1) > 50 else text1},
            )
        else:
            embedding1 = await self._get_embedding(text1)
            self._cache[text1] = embedding1
            logger.debug(
                "Cached new embedding for text1",
                {"text": text1[:50] if len(text1) > 50 else text1},
            )

        if text2 in self._cache:
            embedding2 = self._cache[text2]
            logger.debug(
                "Using cached embedding for text2",
                {"text": text2[:50] if len(text2) > 50 else text2},
            )
        else:
            embedding2 = await self._get_embedding(text2)
            self._cache[text2] = embedding2
            logger.debug(
                "Cached new embedding for text2",
                {"text": text2[:50] if len(text2) > 50 else text2},
            )

        similarity = self._compute_cosine_similarity(embedding1, embedding2)

        logger.info(
            "Similarity computed",
            {
                "text1": text1[:50] if len(text1) > 50 else text1,
                "text2": text2[:50] if len(text2) > 50 else text2,
                "similarity": similarity,
            },
        )

        return similarity


async def compute_similarity(text1: str, text2: str) -> float:
    """Convenience function to compute semantic similarity.

    This function creates an EmbeddingClient with default settings and
    computes the similarity between two texts.

    Args:
        text1: First text string
        text2: Second text string

    Returns:
        Similarity score in range [0, 1]

    Examples:
        >>> similarity = await compute_similarity('cat', 'kitty')
        >>> similarity >= 0.8
        True

        >>> similarity = await compute_similarity('cat', 'dog')
        >>> similarity <= 0.5
        True

        >>> similarity = await compute_similarity('', 'cat')
        >>> similarity == 0.0
        True
    """
    settings = Settings()
    client = EmbeddingClient(settings)
    return await client.compute_similarity(text1, text2)
