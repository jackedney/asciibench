"""Tests for demo module."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from asciibench.common.models import DemoResult, OpenRouterResponse
from asciibench.generator.demo import (
    generate_demo_sample,
    generate_html,
    get_completed_model_ids,
    load_demo_results,
    save_demo_results,
)


class TestLoadDemoResults:
    """Tests for load_demo_results function."""

    def test_load_nonexistent_file_returns_empty_list(self, tmp_path: Path) -> None:
        """Test loading when results.json doesn't exist."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        results = load_demo_results()
        assert results == []

    def test_load_creates_directory_if_not_exists(self, tmp_path: Path) -> None:
        """Test that loading creates .demo_outputs directory."""
        import asciibench.generator.demo as demo_module

        demo_dir = tmp_path / ".demo_outputs"
        demo_module.DEMO_OUTPUTS_DIR = demo_dir
        demo_module.RESULTS_JSON_PATH = demo_dir / "results.json"

        assert not demo_dir.exists()
        load_demo_results()
        assert demo_dir.exists()

    def test_load_valid_json_returns_demo_results(self, tmp_path: Path) -> None:
        """Test loading valid results.json returns DemoResult objects."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        data = [
            {
                "model_id": "openai/gpt-4o-mini",
                "model_name": "GPT-4o-mini",
                "ascii_output": "skeleton",
                "is_valid": True,
                "timestamp": "2026-01-30T20:00:00",
            }
        ]
        with demo_module.RESULTS_JSON_PATH.open("w") as f:
            json.dump(data, f)

        results = load_demo_results()
        assert len(results) == 1
        assert results[0].model_id == "openai/gpt-4o-mini"
        assert results[0].model_name == "GPT-4o-mini"
        assert results[0].is_valid is True

    def test_load_multiple_results(self, tmp_path: Path) -> None:
        """Test loading multiple results."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        data = [
            {
                "model_id": "openai/gpt-4o-mini",
                "model_name": "GPT-4o-mini",
                "ascii_output": "skeleton1",
                "is_valid": True,
                "timestamp": "2026-01-30T20:00:00",
            },
            {
                "model_id": "anthropic/claude-sonnet-4.5",
                "model_name": "Claude 4.5 Sonnet",
                "ascii_output": "skeleton2",
                "is_valid": False,
                "timestamp": "2026-01-30T20:00:01",
            },
        ]
        with demo_module.RESULTS_JSON_PATH.open("w") as f:
            json.dump(data, f)

        results = load_demo_results()
        assert len(results) == 2

    def test_load_corrupted_json_returns_empty_list(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that corrupted JSON returns empty list and warns user."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        with demo_module.RESULTS_JSON_PATH.open("w") as f:
            f.write("{corrupted json here}")

        results = load_demo_results()
        assert results == []

        captured = capsys.readouterr()
        assert "Warning: Corrupted results.json" in captured.out
        assert "Starting with empty results" in captured.out


class TestSaveDemoResults:
    """Tests for save_demo_results function."""

    def test_save_creates_directory_if_not_exists(self, tmp_path: Path) -> None:
        """Test that saving creates .demo_outputs directory."""
        import asciibench.generator.demo as demo_module

        demo_dir = tmp_path / ".demo_outputs"
        demo_module.DEMO_OUTPUTS_DIR = demo_dir
        demo_module.RESULTS_JSON_PATH = demo_dir / "results.json"

        assert not demo_dir.exists()

        result = DemoResult(
            model_id="openai/gpt-4o-mini",
            model_name="GPT-4o-mini",
            ascii_output="skeleton",
            is_valid=True,
            timestamp=datetime.now(),
        )
        save_demo_results([result])

        assert demo_dir.exists()

    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        """Test that saving writes valid JSON to file."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        result = DemoResult(
            model_id="openai/gpt-4o-mini",
            model_name="GPT-4o-mini",
            ascii_output="skeleton",
            is_valid=True,
            timestamp=datetime(2026, 1, 30, 20, 0, 0),
        )
        save_demo_results([result])

        with demo_module.RESULTS_JSON_PATH.open("r") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["model_id"] == "openai/gpt-4o-mini"
        assert data[0]["model_name"] == "GPT-4o-mini"
        assert data[0]["ascii_output"] == "skeleton"
        assert data[0]["is_valid"] is True
        assert "timestamp" in data[0]

    def test_save_multiple_results(self, tmp_path: Path) -> None:
        """Test saving multiple results."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        results = [
            DemoResult(
                model_id="openai/gpt-4o-mini",
                model_name="GPT-4o-mini",
                ascii_output="skeleton1",
                is_valid=True,
                timestamp=datetime(2026, 1, 30, 20, 0, 0),
            ),
            DemoResult(
                model_id="anthropic/claude-sonnet-4.5",
                model_name="Claude 4.5 Sonnet",
                ascii_output="skeleton2",
                is_valid=False,
                timestamp=datetime(2026, 1, 30, 20, 0, 1),
            ),
        ]
        save_demo_results(results)

        with demo_module.RESULTS_JSON_PATH.open("r") as f:
            data = json.load(f)

        assert len(data) == 2

    def test_save_overwrites_existing_file(self, tmp_path: Path) -> None:
        """Test that saving overwrites existing results.json."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        result1 = DemoResult(
            model_id="openai/gpt-4o-mini",
            model_name="GPT-4o-mini",
            ascii_output="skeleton1",
            is_valid=True,
            timestamp=datetime(2026, 1, 30, 20, 0, 0),
        )
        save_demo_results([result1])

        result2 = DemoResult(
            model_id="anthropic/claude-sonnet-4.5",
            model_name="Claude 4.5 Sonnet",
            ascii_output="skeleton2",
            is_valid=True,
            timestamp=datetime(2026, 1, 30, 20, 0, 1),
        )
        save_demo_results([result2])

        with demo_module.RESULTS_JSON_PATH.open("r") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["model_id"] == "anthropic/claude-sonnet-4.5"


class TestGetCompletedModelIds:
    """Tests for get_completed_model_ids function."""

    def test_get_completed_ids_empty(self, tmp_path: Path) -> None:
        """Test getting completed IDs when no results exist."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        ids = get_completed_model_ids()
        assert ids == set()

    def test_get_completed_ids_with_results(self, tmp_path: Path) -> None:
        """Test getting completed IDs from existing results."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        results = [
            DemoResult(
                model_id="openai/gpt-4o-mini",
                model_name="GPT-4o-mini",
                ascii_output="skeleton1",
                is_valid=True,
                timestamp=datetime(2026, 1, 30, 20, 0, 0),
            ),
            DemoResult(
                model_id="anthropic/claude-sonnet-4.5",
                model_name="Claude 4.5 Sonnet",
                ascii_output="skeleton2",
                is_valid=True,
                timestamp=datetime(2026, 1, 30, 20, 0, 1),
            ),
        ]
        save_demo_results(results)

        ids = get_completed_model_ids()
        assert len(ids) == 2
        assert "openai/gpt-4o-mini" in ids
        assert "anthropic/claude-sonnet-4.5" in ids


class TestRoundtrip:
    """Integration tests for save/load roundtrip."""

    def test_roundtrip_preserves_data(self, tmp_path: Path) -> None:
        """Test that saving and loading preserves all data."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        original_results = [
            DemoResult(
                model_id="openai/gpt-4o-mini",
                model_name="GPT-4o-mini",
                ascii_output="multiline\nascii\nart",
                is_valid=True,
                timestamp=datetime(2026, 1, 30, 20, 0, 0),
            ),
            DemoResult(
                model_id="anthropic/claude-sonnet-4.5",
                model_name="Claude 4.5 Sonnet",
                ascii_output="skeleton",
                is_valid=False,
                timestamp=datetime(2026, 1, 30, 20, 0, 1),
            ),
        ]

        save_demo_results(original_results)
        loaded_results = load_demo_results()

        assert len(loaded_results) == len(original_results)
        for original, loaded in zip(original_results, loaded_results, strict=True):
            assert loaded.model_id == original.model_id
            assert loaded.model_name == original.model_name
            assert loaded.ascii_output == original.ascii_output
            assert loaded.is_valid == original.is_valid

    def test_roundtrip_with_corrupted_file(self, tmp_path: Path) -> None:
        """Test that corrupted file doesn't prevent subsequent saves."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        with demo_module.RESULTS_JSON_PATH.open("w") as f:
            f.write("{corrupted}")

        results = load_demo_results()
        assert results == []

        result = DemoResult(
            model_id="openai/gpt-4o-mini",
            model_name="GPT-4o-mini",
            ascii_output="skeleton",
            is_valid=True,
            timestamp=datetime(2026, 1, 30, 20, 0, 0),
        )
        save_demo_results([result])

        loaded = load_demo_results()
        assert len(loaded) == 1


class TestGenerateDemoSample:
    """Tests for generate_demo_sample function."""

    @patch("asciibench.generator.demo.load_generation_config")
    @patch("asciibench.generator.demo.Settings")
    @patch("asciibench.generator.demo.OpenRouterClient")
    def test_generate_demo_sample_valid_output(
        self, mock_client_class, mock_settings_class, mock_load_config
    ):
        """Test generate_demo_sample returns valid DemoResult with ASCII art."""
        mock_config = MagicMock()
        mock_config.temperature = 0.0
        mock_config.max_tokens = 2000
        mock_config.system_prompt = "Draw the requested ASCII art."
        mock_load_config.return_value = mock_config

        mock_settings = MagicMock()
        mock_settings.openrouter_api_key = "test-api-key"
        mock_settings.base_url = "https://openrouter.ai/api/v1"
        mock_settings_class.return_value = mock_settings

        mock_client_instance = MagicMock()
        mock_client_instance.generate.return_value = OpenRouterResponse(
            text="```\n/_/\\\n( o.o )\n > ^ <\n```"
        )
        mock_client_class.return_value = mock_client_instance

        result = generate_demo_sample("openai/gpt-4o-mini", "GPT-4o Mini")

        assert result.model_id == "openai/gpt-4o-mini"
        assert result.model_name == "GPT-4o Mini"
        assert result.is_valid is True
        assert result.ascii_output == "/_/\\\n( o.o )\n > ^ <"
        assert isinstance(result.timestamp, datetime)

        mock_client_class.assert_called_once_with(
            api_key="test-api-key", base_url="https://openrouter.ai/api/v1"
        )
        mock_client_instance.generate.assert_called_once()

    @patch("asciibench.generator.demo.load_generation_config")
    @patch("asciibench.generator.demo.Settings")
    @patch("asciibench.generator.demo.OpenRouterClient")
    def test_generate_demo_sample_uses_config(
        self, mock_client_class, mock_settings_class, mock_load_config
    ):
        """Test generate_demo_sample loads config from config.yaml."""
        mock_config = MagicMock()
        mock_config.temperature = 0.5
        mock_config.max_tokens = 3000
        mock_config.system_prompt = "You are an ASCII art expert."
        mock_load_config.return_value = mock_config

        mock_settings = MagicMock()
        mock_settings.openrouter_api_key = "test-key"
        mock_settings.base_url = "https://openrouter.ai/api/v1"
        mock_settings_class.return_value = mock_settings

        mock_client_instance = MagicMock()
        mock_client_instance.generate.return_value = OpenRouterResponse(text="```\ntest\n```")
        mock_client_class.return_value = mock_client_instance

        result = generate_demo_sample("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet")

        assert result.is_valid is True

        call_args = mock_client_instance.generate.call_args
        assert call_args[0][0] == "anthropic/claude-3.5-sonnet"
        assert call_args[0][1] == "Draw a skeleton in ASCII art"
        assert call_args[1]["config"] == mock_config

    @patch("asciibench.generator.demo.load_generation_config")
    @patch("asciibench.generator.demo.Settings")
    @patch("asciibench.generator.demo.OpenRouterClient")
    def test_generate_demo_sample_handles_auth_error(
        self, mock_client_class, mock_settings_class, mock_load_config
    ):
        """Test generate_demo_sample handles authentication error gracefully."""
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        mock_settings = MagicMock()
        mock_settings.openrouter_api_key = "invalid-key"
        mock_settings.base_url = "https://openrouter.ai/api/v1"
        mock_settings_class.return_value = mock_settings

        from asciibench.generator.client import AuthenticationError

        mock_client_instance = MagicMock()
        mock_client_instance.generate.side_effect = AuthenticationError("Authentication failed")
        mock_client_class.return_value = mock_client_instance

        result = generate_demo_sample("openai/gpt-4o-mini", "GPT-4o Mini")

        assert result.model_id == "openai/gpt-4o-mini"
        assert result.model_name == "GPT-4o Mini"
        assert result.is_valid is False
        assert "Error" in result.ascii_output
        assert "Authentication failed" in result.ascii_output

    @patch("asciibench.generator.demo.load_generation_config")
    @patch("asciibench.generator.demo.Settings")
    @patch("asciibench.generator.demo.OpenRouterClient")
    def test_generate_demo_sample_handles_model_error(
        self, mock_client_class, mock_settings_class, mock_load_config
    ):
        """Test generate_demo_sample handles model error gracefully."""
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        mock_settings = MagicMock()
        mock_settings.openrouter_api_key = "test-key"
        mock_settings.base_url = "https://openrouter.ai/api/v1"
        mock_settings_class.return_value = mock_settings

        from asciibench.generator.client import ModelError

        mock_client_instance = MagicMock()
        mock_client_instance.generate.side_effect = ModelError("Model not found")
        mock_client_class.return_value = mock_client_instance

        result = generate_demo_sample("invalid/model", "Invalid Model")

        assert result.is_valid is False
        assert "Error" in result.ascii_output
        assert "Model not found" in result.ascii_output

    @patch("asciibench.generator.demo.load_generation_config")
    @patch("asciibench.generator.demo.Settings")
    @patch("asciibench.generator.demo.OpenRouterClient")
    def test_generate_demo_sample_handles_client_error(
        self, mock_client_class, mock_settings_class, mock_load_config
    ):
        """Test generate_demo_sample handles generic client error gracefully."""
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        mock_settings = MagicMock()
        mock_settings.openrouter_api_key = "test-key"
        mock_settings.base_url = "https://openrouter.ai/api/v1"
        mock_settings_class.return_value = mock_settings

        from asciibench.generator.client import OpenRouterClientError

        mock_client_instance = MagicMock()
        mock_client_instance.generate.side_effect = OpenRouterClientError("API error")
        mock_client_class.return_value = mock_client_instance

        result = generate_demo_sample("openai/gpt-4o-mini", "GPT-4o Mini")

        assert result.is_valid is False
        assert "Error" in result.ascii_output
        assert "API error" in result.ascii_output

    @patch("asciibench.generator.demo.load_generation_config")
    @patch("asciibench.generator.demo.Settings")
    @patch("asciibench.generator.demo.OpenRouterClient")
    def test_generate_demo_sample_handles_unexpected_error(
        self, mock_client_class, mock_settings_class, mock_load_config
    ):
        """Test generate_demo_sample handles unexpected errors gracefully."""
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        mock_settings = MagicMock()
        mock_settings.openrouter_api_key = "test-key"
        mock_settings.base_url = "https://openrouter.ai/api/v1"
        mock_settings_class.return_value = mock_settings

        mock_client_instance = MagicMock()
        mock_client_instance.generate.side_effect = RuntimeError("Unexpected error")
        mock_client_class.return_value = mock_client_instance

        result = generate_demo_sample("openai/gpt-4o-mini", "GPT-4o Mini")

        assert result.is_valid is False
        assert "Unexpected error" in result.ascii_output

    @patch("asciibench.generator.demo.load_generation_config")
    @patch("asciibench.generator.demo.Settings")
    @patch("asciibench.generator.demo.OpenRouterClient")
    def test_generate_demo_sample_empty_output_invalid(
        self, mock_client_class, mock_settings_class, mock_load_config
    ):
        """Test generate_demo_sample returns is_valid=False for empty output."""
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        mock_settings = MagicMock()
        mock_settings.openrouter_api_key = "test-key"
        mock_settings.base_url = "https://openrouter.ai/api/v1"
        mock_settings_class.return_value = mock_settings

        mock_client_instance = MagicMock()
        mock_client_instance.generate.return_value = OpenRouterResponse(text="No code block here")
        mock_client_class.return_value = mock_client_instance

        result = generate_demo_sample("openai/gpt-4o-mini", "GPT-4o Mini")

        assert result.is_valid is False
        assert "Error:" in result.ascii_output


class TestGenerateHtml:
    """Tests for generate_html function."""

    def test_generate_html_creates_directory(self, tmp_path: Path) -> None:
        """Test that HTML generation creates .demo_outputs directory."""
        import asciibench.generator.demo as demo_module

        demo_dir = tmp_path / ".demo_outputs"
        demo_module.DEMO_OUTPUTS_DIR = demo_dir
        demo_module.DEMO_HTML_PATH = demo_dir / "demo.html"
        demo_module.RESULTS_JSON_PATH = demo_dir / "results.json"

        assert not demo_dir.exists()
        generate_html()
        assert demo_dir.exists()

    def test_generate_html_empty_results(self, tmp_path: Path) -> None:
        """Test HTML generation with empty results shows 'No results yet' message."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.DEMO_HTML_PATH = tmp_path / "demo.html"
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        generate_html()

        assert demo_module.DEMO_HTML_PATH.exists()
        html_content = demo_module.DEMO_HTML_PATH.read_text(encoding="utf-8")

        assert "ASCIIBench Demo - Skeleton ASCII Art" in html_content
        assert "No results yet" in html_content
        assert "Run 'task demo' to generate ASCII art samples" in html_content

    def test_generate_html_with_single_result(self, tmp_path: Path) -> None:
        """Test HTML generation with a single valid result."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.DEMO_HTML_PATH = tmp_path / "demo.html"
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        result = DemoResult(
            model_id="openai/gpt-4o-mini",
            model_name="GPT-4o-mini",
            ascii_output="skeleton\nart",
            is_valid=True,
            timestamp=datetime(2026, 1, 30, 20, 0, 0),
        )
        save_demo_results([result])

        generate_html()

        assert demo_module.DEMO_HTML_PATH.exists()
        html_content = demo_module.DEMO_HTML_PATH.read_text(encoding="utf-8")

        assert "ASCIIBench Demo - Skeleton ASCII Art" in html_content
        assert "GPT-4o-mini" in html_content
        assert "openai/gpt-4o-mini" in html_content
        assert "skeleton\nart" in html_content
        assert "Generated: 2026-01-30 20:00:00" in html_content
        assert "Valid" in html_content

    def test_generate_html_with_multiple_results(self, tmp_path: Path) -> None:
        """Test HTML generation with multiple results."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.DEMO_HTML_PATH = tmp_path / "demo.html"
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        results = [
            DemoResult(
                model_id="openai/gpt-4o-mini",
                model_name="GPT-4o-mini",
                ascii_output="skeleton1",
                is_valid=True,
                timestamp=datetime(2026, 1, 30, 20, 0, 0),
            ),
            DemoResult(
                model_id="anthropic/claude-sonnet-4.5",
                model_name="Claude 4.5 Sonnet",
                ascii_output="skeleton2",
                is_valid=True,
                timestamp=datetime(2026, 1, 30, 20, 0, 1),
            ),
        ]
        save_demo_results(results)

        generate_html()

        assert demo_module.DEMO_HTML_PATH.exists()
        html_content = demo_module.DEMO_HTML_PATH.read_text(encoding="utf-8")

        assert "GPT-4o-mini" in html_content
        assert "Claude 4.5 Sonnet" in html_content
        assert "skeleton1" in html_content
        assert "skeleton2" in html_content

    def test_generate_html_invalid_result_styling(self, tmp_path: Path) -> None:
        """Test HTML generation shows red border for invalid results."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.DEMO_HTML_PATH = tmp_path / "demo.html"
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        result = DemoResult(
            model_id="openai/gpt-4o-mini",
            model_name="GPT-4o-mini",
            ascii_output="Error: API error",
            is_valid=False,
            timestamp=datetime(2026, 1, 30, 20, 0, 0),
        )
        save_demo_results([result])

        generate_html()

        assert demo_module.DEMO_HTML_PATH.exists()
        html_content = demo_module.DEMO_HTML_PATH.read_text(encoding="utf-8")

        assert 'class="model-section invalid"' in html_content
        assert "Invalid" in html_content
        assert 'class="valid-badge invalid"' in html_content

    def test_generate_html_includes_inline_css(self, tmp_path: Path) -> None:
        """Test HTML generation includes inline CSS styling."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.DEMO_HTML_PATH = tmp_path / "demo.html"
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        result = DemoResult(
            model_id="openai/gpt-4o-mini",
            model_name="GPT-4o-mini",
            ascii_output="skeleton",
            is_valid=True,
            timestamp=datetime(2026, 1, 30, 20, 0, 0),
        )
        save_demo_results([result])

        generate_html()

        html_content = demo_module.DEMO_HTML_PATH.read_text(encoding="utf-8")

        assert "<style>" in html_content
        assert "Courier New" in html_content
        assert "monospace" in html_content
        assert "<!DOCTYPE html>" in html_content
        assert "<title>ASCIIBench Demo - Skeleton ASCII Art</title>" in html_content

    def test_generate_html_monospace_font(self, tmp_path: Path) -> None:
        """Test HTML generation uses monospace font for ASCII art."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.DEMO_HTML_PATH = tmp_path / "demo.html"
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        result = DemoResult(
            model_id="openai/gpt-4o-mini",
            model_name="GPT-4o-mini",
            ascii_output="skeleton",
            is_valid=True,
            timestamp=datetime(2026, 1, 30, 20, 0, 0),
        )
        save_demo_results([result])

        generate_html()

        html_content = demo_module.DEMO_HTML_PATH.read_text(encoding="utf-8")

        assert "font-family: 'Courier New', Courier, monospace" in html_content

    def test_generate_html_uses_utf8_encoding(self, tmp_path: Path) -> None:
        """Test HTML generation uses UTF-8 encoding."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.DEMO_HTML_PATH = tmp_path / "demo.html"
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        result = DemoResult(
            model_id="openai/gpt-4o-mini",
            model_name="GPT-4o-mini",
            ascii_output="skeleton with émojis",
            is_valid=True,
            timestamp=datetime(2026, 1, 30, 20, 0, 0),
        )
        save_demo_results([result])

        generate_html()

        assert demo_module.DEMO_HTML_PATH.exists()
        html_content = demo_module.DEMO_HTML_PATH.read_text(encoding="utf-8")

        assert "skeleton with émojis" in html_content

    def test_generate_html_escapes_special_characters(self, tmp_path: Path) -> None:
        """Test HTML generation escapes special characters properly."""
        import asciibench.generator.demo as demo_module

        demo_module.DEMO_OUTPUTS_DIR = tmp_path
        demo_module.DEMO_HTML_PATH = tmp_path / "demo.html"
        demo_module.RESULTS_JSON_PATH = tmp_path / "results.json"

        result = DemoResult(
            model_id="openai/gpt-4o-mini",
            model_name="GPT-4o-mini",
            ascii_output='<div>test</div> & "quotes"',
            is_valid=True,
            timestamp=datetime(2026, 1, 30, 20, 0, 0),
        )
        save_demo_results([result])

        generate_html()

        assert demo_module.DEMO_HTML_PATH.exists()
        html_content = demo_module.DEMO_HTML_PATH.read_text(encoding="utf-8")

        assert "&lt;div&gt;test&lt;/div&gt;" in html_content
        assert "&amp;" in html_content
        assert "&quot;" in html_content
        assert "<div>" not in html_content
        assert "&quot;quotes&quot;" in html_content
