"""Tests for the evaluator module.

This file contains tests for all evaluator components:
- renderer: ASCII-to-image rendering
- subject_extractor: prompt subject extraction
- similarity: semantic similarity computation

External APIs (VLM, embeddings) are mocked in tests.
"""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from asciibench.common.config import FontConfig, RendererConfig
from asciibench.evaluator.renderer import render_ascii_to_image
from asciibench.evaluator.similarity import EmbeddingClient
from asciibench.evaluator.subject_extractor import extract_subject


class TestRenderer:
    """Tests for render_ascii_to_image function."""

    def test_valid_ascii_produces_png(self):
        """Valid ASCII art produces valid PNG bytes."""
        config = RendererConfig()
        ascii_text = " /\\_/\\ \n( o.o )\n  > ^ <"

        result = render_ascii_to_image(ascii_text, config)

        assert isinstance(result, bytes)
        assert len(result) > 0

        result_io = BytesIO(result)
        with Image.open(result_io) as img:
            assert img.format == "PNG"
            assert img.width > 0
            assert img.height > 0

    def test_empty_input_handled(self):
        """Empty string input returns minimal valid PNG."""
        config = RendererConfig()

        result = render_ascii_to_image("", config)

        assert isinstance(result, bytes)
        assert len(result) > 0

        result_io = BytesIO(result)
        with Image.open(result_io) as img:
            assert img.format == "PNG"

    def test_none_input_handled(self):
        """None input returns minimal valid PNG."""
        config = RendererConfig()

        result = render_ascii_to_image(None, config)

        assert isinstance(result, bytes)
        assert len(result) > 0

        result_io = BytesIO(result)
        with Image.open(result_io) as img:
            assert img.format == "PNG"

    def test_single_line_ascii(self):
        """Single line ASCII art is rendered correctly."""
        config = RendererConfig()
        ascii_text = "Hello World"

        result = render_ascii_to_image(ascii_text, config)

        assert isinstance(result, bytes)
        assert len(result) > 0

        result_io = BytesIO(result)
        with Image.open(result_io) as img:
            assert img.format == "PNG"

    def test_multiline_ascii_preserves_spacing(self):
        """Multi-line ASCII art preserves line breaks and spacing."""
        config = RendererConfig()
        multiline_text = "  /\\_/\\  \n ( o.o ) \n  > ^ <"
        baseline_text = "X"

        baseline_result = render_ascii_to_image(baseline_text, config)
        baseline_io = BytesIO(baseline_result)
        with Image.open(baseline_io) as baseline_img:
            baseline_height = baseline_img.height

        result = render_ascii_to_image(multiline_text, config)

        assert isinstance(result, bytes)
        result_io = BytesIO(result)
        with Image.open(result_io) as img:
            assert img.format == "PNG"
            assert img.height > baseline_height

    def test_custom_font_config(self):
        """Custom font configuration is respected."""
        config = RendererConfig(font=FontConfig(family="Courier", size=20))
        ascii_text = "Test"

        result = render_ascii_to_image(ascii_text, config)

        assert isinstance(result, bytes)
        result_io = BytesIO(result)
        with Image.open(result_io) as img:
            assert img.format == "PNG"

    def test_custom_colors(self):
        """Custom foreground and background colors are applied."""
        config = RendererConfig(background_color="black", text_color="white")
        ascii_text = "X"

        result = render_ascii_to_image(ascii_text, config)

        assert isinstance(result, bytes)
        result_io = BytesIO(result)
        with Image.open(result_io) as img:
            assert img.format == "PNG"
            pixel = img.getpixel((0, 0))
            assert pixel[0] == 0

    def test_wide_ascii_art(self):
        """Wide ASCII art is rendered correctly."""
        config = RendererConfig()
        ascii_text = "A" * 100 + "\n" + "B" * 100

        result = render_ascii_to_image(ascii_text, config)

        assert isinstance(result, bytes)
        result_io = BytesIO(result)
        with Image.open(result_io) as img:
            assert img.format == "PNG"
            assert img.width > 100

    def test_tall_ascii_art(self):
        """Tall ASCII art is rendered correctly."""
        config = RendererConfig()
        ascii_text = "\n".join([f"Line {i}" for i in range(50)])

        result = render_ascii_to_image(ascii_text, config)

        assert isinstance(result, bytes)
        result_io = BytesIO(result)
        with Image.open(result_io) as img:
            assert img.format == "PNG"
            assert img.height > 100

    def test_special_characters_in_ascii(self):
        """Special characters in ASCII art are rendered correctly."""
        config = RendererConfig()
        ascii_text = "@#$%^&*()\n~`-_=+[]{}|;':\",./<>?"

        result = render_ascii_to_image(ascii_text, config)

        assert isinstance(result, bytes)
        result_io = BytesIO(result)
        with Image.open(result_io) as img:
            assert img.format == "PNG"


class TestSubjectExtractor:
    """Tests for extract_subject function."""

    def test_extract_subject_single_object(self):
        """Example: extract_subject('Draw a house') == 'house'."""
        result = extract_subject("Draw a house")
        assert result == "house"

    def test_extract_subject_with_preposition_in_ascii(self):
        """Extract subject when prompt includes 'in ASCII art'."""
        result = extract_subject("Draw a cat in ASCII art")
        assert result == "cat"

    def test_extract_subject_spatial_sitting_on(self):
        """Extract both subjects with 'sitting on' preposition."""
        result = extract_subject("Draw a cat sitting on a fence")
        assert result == "cat, fence"

    def test_extract_subject_spatial_above(self):
        """Extract both subjects with 'above' preposition."""
        result = extract_subject("Draw a bird above a tree")
        assert result == "bird, tree"

    def test_extract_subject_spatial_under(self):
        """Extract both subjects with 'under' preposition."""
        result = extract_subject("Draw a dog under a table")
        assert result == "dog, table"

    def test_extract_subject_with_an_article(self):
        """Handle 'Draw an' instead of 'Draw a'."""
        result = extract_subject("Draw an apple")
        assert result == "apple"

    def test_extract_subject_with_adjective(self):
        """Extract subject with adjective."""
        result = extract_subject("Draw a happy cat")
        assert result == "happy cat"

    def test_extract_subject_with_action(self):
        """Extract subject when there's an action verb."""
        result = extract_subject("Draw a cat running fast")
        assert result == "cat"

    def test_extract_subject_spatial_multi_word_secondary(self):
        """Extract multi-word secondary subject."""
        result = extract_subject("Draw a cat sitting on a wooden fence")
        assert result == "cat, wooden fence"

    def test_extract_subject_unrecognized_format(self):
        """Return full prompt text for unrecognized format."""
        result = extract_subject("Some unrecognized prompt format")
        assert result == "Some unrecognized prompt format"

    def test_extract_subject_empty_string(self):
        """Return empty string for empty input."""
        result = extract_subject("")
        assert result == ""


class TestSimilarity:
    """Tests for semantic similarity computation."""

    def test_cosine_similarity_identical_vectors(self):
        """Cosine similarity of identical vectors is 1.0."""
        settings_mock = MagicMock()
        client = EmbeddingClient(settings_mock)
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0, 3.0]

        result = client._compute_cosine_similarity(vec1, vec2)

        assert result == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Cosine similarity of orthogonal vectors is 0.0."""
        settings_mock = MagicMock()
        client = EmbeddingClient(settings_mock)
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]

        result = client._compute_cosine_similarity(vec1, vec2)

        assert result == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self):
        """Cosine similarity with zero vector returns 0.0."""
        settings_mock = MagicMock()
        client = EmbeddingClient(settings_mock)
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]

        result = client._compute_cosine_similarity(vec1, vec2)

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_similarity_empty_input(self):
        """Negative case: empty string input returns 0.0 similarity."""
        settings_mock = MagicMock()
        client = EmbeddingClient(settings_mock)

        result = await client.compute_similarity("", "cat")

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_similarity_identical_strings(self):
        """Identical strings return 1.0 similarity."""
        settings_mock = MagicMock()
        client = EmbeddingClient(settings_mock)

        result = await client.compute_similarity("cat", "cat")

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_similarity_known_similar_pair(self):
        """Known similar pair (cat, kitty) returns >= 0.8."""
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
    async def test_similarity_known_dissimilar_pair(self):
        """Known dissimilar pair (cat, dog) returns <= 0.5."""
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
    async def test_similarity_caches_embeddings(self):
        """Embeddings are cached to reduce API calls."""
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
            assert "kitty" in client._cache
