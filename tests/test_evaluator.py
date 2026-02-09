"""Tests for the evaluator module.

This file contains tests for all evaluator components:
- renderer: ASCII-to-image rendering
- subject_extractor: prompt subject extraction
- similarity: semantic similarity computation
- orchestrator: evaluation orchestration with composition pattern

External APIs (VLM, embeddings) are mocked in tests.
"""

from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from PIL import Image

from asciibench.common.config import (
    EvaluatorConfig,
    FontConfig,
    RendererConfig,
)
from asciibench.common.models import ArtSample, VLMEvaluation
from asciibench.evaluator.orchestrator import (
    EvaluationOrchestrator,
    EvaluationWriter,
    ImageRenderer,
    VLMAnalyzer,
)
from asciibench.evaluator.renderer import render_ascii_to_image
from asciibench.evaluator.similarity import EmbeddingClient
from asciibench.evaluator.subject_extractor import extract_subject
from asciibench.evaluator.vlm_client import VLMClient


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
            assert isinstance(pixel, tuple)
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


class TestImageRenderer:
    """Tests for ImageRenderer class."""

    def test_render_valid_ascii(self):
        """ImageRenderer renders valid ASCII to PNG bytes."""
        config = RendererConfig()
        renderer = ImageRenderer(config)
        ascii_text = " /\\_/\\ \n( o.o )\n  > ^ <"

        result = renderer.render(ascii_text)

        assert isinstance(result, bytes)
        assert len(result) > 0

        result_io = BytesIO(result)
        with Image.open(result_io) as img:
            assert img.format == "PNG"
            assert img.width > 0
            assert img.height > 0

    def test_render_empty_string(self):
        """ImageRenderer handles empty string input."""
        config = RendererConfig()
        renderer = ImageRenderer(config)

        result = renderer.render("")

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_render_none_input(self):
        """ImageRenderer handles None input."""
        config = RendererConfig()
        renderer = ImageRenderer(config)

        result = renderer.render(None)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_render_with_custom_config(self):
        """ImageRenderer respects custom config."""
        config = RendererConfig(
            font=FontConfig(family="Courier", size=20),
            background_color="black",
            text_color="white",
        )
        renderer = ImageRenderer(config)

        result = renderer.render("Test")

        assert isinstance(result, bytes)
        assert len(result) > 0


class TestVLMAnalyzer:
    """Tests for VLMAnalyzer class."""

    @pytest.mark.asyncio
    async def test_analyze_with_mocked_client(self):
        """VLMAnalyzer correctly calls VLM client and computes similarity."""
        mock_client = MagicMock(spec=VLMClient)
        mock_client.api_key = "test-model"
        mock_client.analyze_image = AsyncMock(return_value=("A cat", 0.001))

        with patch(
            "asciibench.evaluator.orchestrator.compute_similarity",
            return_value=0.95,
        ):
            analyzer = VLMAnalyzer(mock_client, similarity_threshold=0.7)
            response, cost, is_correct = await analyzer.analyze(
                b"fake_image_bytes", "cat", "openai/gpt-4o"
            )

        mock_client.analyze_image.assert_called_once_with(b"fake_image_bytes", "openai/gpt-4o")
        assert response == "A cat"
        assert cost == 0.001
        assert is_correct is True

    @pytest.mark.asyncio
    async def test_analyze_below_threshold(self):
        """VLMAnalyzer marks as incorrect when similarity below threshold."""
        mock_client = MagicMock(spec=VLMClient)
        mock_client.api_key = "test-model"
        mock_client.analyze_image = AsyncMock(return_value=("A dog", 0.001))

        with patch(
            "asciibench.evaluator.orchestrator.compute_similarity",
            return_value=0.5,
        ):
            analyzer = VLMAnalyzer(mock_client, similarity_threshold=0.7)
            _response, _cost, is_correct = await analyzer.analyze(
                b"fake_image_bytes", "cat", "openai/gpt-4o"
            )

        assert is_correct is False

    @pytest.mark.asyncio
    async def test_analyze_at_threshold(self):
        """VLMAnalyzer marks as correct when similarity equals threshold."""
        mock_client = MagicMock(spec=VLMClient)
        mock_client.api_key = "test-model"
        mock_client.analyze_image = AsyncMock(return_value=("A cat", 0.001))

        with patch(
            "asciibench.evaluator.orchestrator.compute_similarity",
            return_value=0.7,
        ):
            analyzer = VLMAnalyzer(mock_client, similarity_threshold=0.7)
            _, _, is_correct = await analyzer.analyze(b"fake_image_bytes", "cat", "openai/gpt-4o")

        assert is_correct is True


class TestEvaluationWriter:
    """Tests for EvaluationWriter class."""

    def test_write_evaluation(self, tmp_path: Path):
        """EvaluationWriter correctly writes evaluation to JSONL."""
        evaluations_path = tmp_path / "vlm_evaluations.jsonl"
        writer = EvaluationWriter(evaluations_path)

        evaluation = VLMEvaluation(
            sample_id=str(uuid4()),
            vlm_model_id="openai/gpt-4o",
            expected_subject="cat",
            vlm_response="A cat",
            similarity_score=0.95,
            is_correct=True,
            cost=0.001,
        )

        writer.write(evaluation)

        assert evaluations_path.exists()
        content = evaluations_path.read_text()
        assert "cat" in content
        assert "openai/gpt-4o" in content

    def test_write_multiple_evaluations(self, tmp_path: Path):
        """EvaluationWriter appends multiple evaluations to JSONL."""
        evaluations_path = tmp_path / "vlm_evaluations.jsonl"
        writer = EvaluationWriter(evaluations_path)

        for i in range(3):
            evaluation = VLMEvaluation(
                sample_id=str(uuid4()),
                vlm_model_id="openai/gpt-4o",
                expected_subject=f"subject_{i}",
                vlm_response=f"response_{i}",
                similarity_score=0.9,
                is_correct=True,
                cost=0.001,
            )
            writer.write(evaluation)

        content = evaluations_path.read_text()
        lines = [line for line in content.strip().split("\n") if line]
        assert len(lines) == 3


class TestEvaluationOrchestrator:
    """Tests for EvaluationOrchestrator class."""

    def test_orchestrator_initialization(self):
        """EvaluationOrchestrator initializes with required dependencies."""
        renderer = MagicMock(spec=ImageRenderer)
        vlm_analyzer = MagicMock(spec=VLMAnalyzer)
        evaluation_writer = MagicMock(spec=EvaluationWriter)
        config = EvaluatorConfig(
            vlm_models=["openai/gpt-4o"],
            similarity_threshold=0.7,
            max_concurrency=5,
        )

        orchestrator = EvaluationOrchestrator(renderer, vlm_analyzer, evaluation_writer, config)

        assert orchestrator._renderer is renderer
        assert orchestrator._vlm_analyzer is vlm_analyzer
        assert orchestrator._evaluation_writer is evaluation_writer
        assert orchestrator._config is config

    @pytest.mark.asyncio
    async def test_orchestrator_processes_samples(self, tmp_path: Path):
        """EvaluationOrchestrator processes samples successfully."""
        mock_renderer = MagicMock(spec=ImageRenderer)
        mock_renderer.render.return_value = b"fake_image"

        mock_analyzer = MagicMock(spec=VLMAnalyzer)
        mock_analyzer.analyze = AsyncMock(return_value=("A cat", 0.001, True))

        mock_writer = MagicMock(spec=EvaluationWriter)

        config = EvaluatorConfig(
            vlm_models=["openai/gpt-4o"],
            similarity_threshold=0.7,
            max_concurrency=2,
        )

        orchestrator = EvaluationOrchestrator(mock_renderer, mock_analyzer, mock_writer, config)

        samples = [
            ArtSample(
                model_id="model1",
                prompt_text="Draw a cat",
                category="animals",
                attempt_number=1,
                raw_output="output",
                sanitized_output="output",
                is_valid=True,
            )
        ]

        results = await orchestrator.run(samples, [])

        assert len(results) == 1
        mock_renderer.render.assert_called_once()
        mock_analyzer.analyze.assert_called_once()
        mock_writer.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrator_skips_existing_evaluations(self):
        """EvaluationOrchestrator skips samples with existing evaluations."""
        mock_renderer = MagicMock(spec=ImageRenderer)
        mock_analyzer = MagicMock(spec=VLMAnalyzer)
        mock_analyzer.analyze = AsyncMock(return_value=("A cat", 0.001, True))
        mock_writer = MagicMock(spec=EvaluationWriter)

        config = EvaluatorConfig(
            vlm_models=["openai/gpt-4o"],
            similarity_threshold=0.7,
            max_concurrency=2,
        )

        orchestrator = EvaluationOrchestrator(mock_renderer, mock_analyzer, mock_writer, config)

        samples = [
            ArtSample(
                model_id="model1",
                prompt_text="Draw a cat",
                category="animals",
                attempt_number=1,
                raw_output="output",
                sanitized_output="output",
                is_valid=True,
            )
        ]

        existing_evaluations = [
            VLMEvaluation(
                sample_id=str(samples[0].id),
                vlm_model_id="openai/gpt-4o",
                expected_subject="cat",
                vlm_response="A cat",
                similarity_score=0.95,
                is_correct=True,
                cost=0.001,
            )
        ]

        results = await orchestrator.run(samples, existing_evaluations)

        assert len(results) == 0
        mock_renderer.render.assert_not_called()
        mock_analyzer.analyze.assert_not_called()
        mock_writer.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_orchestrator_handles_error_gracefully(self):
        """EvaluationOrchestrator handles errors and continues processing."""
        mock_renderer = MagicMock(spec=ImageRenderer)
        mock_renderer.render.side_effect = [b"image1", Exception("Render error"), b"image3"]

        mock_analyzer = MagicMock(spec=VLMAnalyzer)
        mock_analyzer.analyze = AsyncMock(return_value=("A cat", 0.001, True))

        mock_writer = MagicMock(spec=EvaluationWriter)

        config = EvaluatorConfig(
            vlm_models=["openai/gpt-4o"],
            similarity_threshold=0.7,
            max_concurrency=2,
        )

        orchestrator = EvaluationOrchestrator(mock_renderer, mock_analyzer, mock_writer, config)

        samples = [
            ArtSample(
                model_id=f"model{i}",
                prompt_text=f"Draw a cat{i}",
                category="animals",
                attempt_number=1,
                raw_output="output",
                sanitized_output="output",
                is_valid=True,
            )
            for i in range(3)
        ]

        results = await orchestrator.run(samples, [])

        assert len(results) == 2
        assert mock_writer.write.call_count == 2

    @pytest.mark.asyncio
    async def test_orchestrator_respects_limit(self):
        """EvaluationOrchestrator respects the limit parameter."""
        mock_renderer = MagicMock(spec=ImageRenderer)
        mock_renderer.render.return_value = b"fake_image"

        mock_analyzer = MagicMock(spec=VLMAnalyzer)
        mock_analyzer.analyze = AsyncMock(return_value=("A cat", 0.001, True))

        mock_writer = MagicMock(spec=EvaluationWriter)

        config = EvaluatorConfig(
            vlm_models=["openai/gpt-4o"],
            similarity_threshold=0.7,
            max_concurrency=2,
        )

        orchestrator = EvaluationOrchestrator(mock_renderer, mock_analyzer, mock_writer, config)

        samples = [
            ArtSample(
                model_id=f"model{i}",
                prompt_text=f"Draw a cat{i}",
                category="animals",
                attempt_number=1,
                raw_output="output",
                sanitized_output="output",
                is_valid=True,
            )
            for i in range(10)
        ]

        results = await orchestrator.run(samples, [], limit=3)

        assert len(results) == 3
        assert mock_renderer.render.call_count == 3

    @pytest.mark.asyncio
    async def test_orchestrator_calls_progress_callback(self):
        """EvaluationOrchestrator calls progress callback after each evaluation."""
        mock_renderer = MagicMock(spec=ImageRenderer)
        mock_renderer.render.return_value = b"fake_image"

        mock_analyzer = MagicMock(spec=VLMAnalyzer)
        mock_analyzer.analyze = AsyncMock(return_value=("A cat", 0.001, True))

        mock_writer = MagicMock(spec=EvaluationWriter)

        config = EvaluatorConfig(
            vlm_models=["openai/gpt-4o"],
            similarity_threshold=0.7,
            max_concurrency=2,
        )

        orchestrator = EvaluationOrchestrator(mock_renderer, mock_analyzer, mock_writer, config)

        samples = [
            ArtSample(
                model_id="model1",
                prompt_text="Draw a cat",
                category="animals",
                attempt_number=1,
                raw_output="output",
                sanitized_output="output",
                is_valid=True,
            )
        ]

        callback_calls = []

        def mock_callback(processed: int, total: int, model_id: str):
            callback_calls.append((processed, total, model_id))

        await orchestrator.run(samples, [], progress_callback=mock_callback)

        assert len(callback_calls) == 1
        assert callback_calls[0] == (1, 1, "openai/gpt-4o")

    @pytest.mark.asyncio
    async def test_orchestrator_multi_model(self):
        """EvaluationOrchestrator processes samples for multiple VLM models."""
        mock_renderer = MagicMock(spec=ImageRenderer)
        mock_renderer.render.return_value = b"fake_image"

        mock_analyzer = MagicMock(spec=VLMAnalyzer)
        mock_analyzer.analyze = AsyncMock(return_value=("A cat", 0.001, True))

        mock_writer = MagicMock(spec=EvaluationWriter)

        config = EvaluatorConfig(
            vlm_models=["openai/gpt-4o", "anthropic/claude-3-opus"],
            similarity_threshold=0.7,
            max_concurrency=2,
        )

        orchestrator = EvaluationOrchestrator(mock_renderer, mock_analyzer, mock_writer, config)

        samples = [
            ArtSample(
                model_id="model1",
                prompt_text="Draw a cat",
                category="animals",
                attempt_number=1,
                raw_output="output",
                sanitized_output="output",
                is_valid=True,
            )
        ]

        results = await orchestrator.run(samples, [])

        assert len(results) == 2
        assert mock_renderer.render.call_count == 2
        assert mock_analyzer.analyze.call_count == 2
        assert mock_writer.write.call_count == 2

        vlm_model_ids = {r.vlm_model_id for r in results}
        assert vlm_model_ids == {"openai/gpt-4o", "anthropic/claude-3-opus"}
