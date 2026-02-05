"""Tests for the generator main module."""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from asciibench.common.config import GenerationConfig
from asciibench.common.models import ArtSample, Model, Prompt
from asciibench.generator.main import _print_progress, main


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


class TestPrintProgress:
    """Tests for the _print_progress helper function."""

    def test_prints_progress_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Progress info is printed to stdout."""
        _print_progress("openai/gpt-4o", "Draw a cat", 1, 100)

        captured = capsys.readouterr()
        stripped_output = strip_ansi(captured.out)
        assert "[100 remaining]" in stripped_output
        assert "openai/gpt-4o" in stripped_output
        assert "Attempt 1" in stripped_output
        assert "Draw a cat" in stripped_output

    def test_truncates_long_prompt(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Long prompts are truncated."""
        long_prompt = "A" * 100
        _print_progress("model", long_prompt, 1, 50)

        captured = capsys.readouterr()
        stripped_output = strip_ansi(captured.out)
        assert "..." in stripped_output
        # Should be truncated to 50 chars + "..."
        assert "A" * 50 in stripped_output
        assert "A" * 51 not in stripped_output


class TestMain:
    """Tests for the main() entry point."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock Settings with API key."""
        mock = MagicMock()
        mock.openrouter_api_key = "test-api-key"
        mock.base_url = "https://openrouter.ai/api/v1"
        return mock

    @pytest.fixture
    def mock_config(self) -> GenerationConfig:
        """Create mock GenerationConfig."""
        return GenerationConfig(attempts_per_prompt=2)

    @pytest.fixture
    def mock_models(self) -> list[Model]:
        """Create mock models list."""
        return [Model(id="openai/gpt-4o", name="GPT-4o")]

    @pytest.fixture
    def mock_prompts(self) -> list[Prompt]:
        """Create mock prompts list."""
        return [Prompt(text="Draw a cat", category="animal", template_type="test")]

    def test_main_missing_api_key_exits(self) -> None:
        """Missing API key prints helpful error and exits."""
        mock_settings = MagicMock()
        mock_settings.openrouter_api_key = ""

        with (
            patch("asciibench.generator.main.Settings", return_value=mock_settings),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        exc = exc_info.value
        assert isinstance(exc, SystemExit)
        assert exc.code == 1

    def test_main_missing_api_key_error_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Missing API key shows helpful error message."""
        mock_settings = MagicMock()
        mock_settings.openrouter_api_key = ""

        with (
            patch("asciibench.generator.main.Settings", return_value=mock_settings),
            pytest.raises(SystemExit),
        ):
            main()

        captured = capsys.readouterr()
        assert "Missing OpenRouter API key" in captured.err
        assert "OPENROUTER_API_KEY" in captured.err
        assert "openrouter.ai" in captured.err

    def test_main_shows_banner_and_config_summary(
        self,
        mock_settings: MagicMock,
        mock_config: GenerationConfig,
        mock_models: list[Model],
        mock_prompts: list[Prompt],
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        """Main shows banner and configuration summary."""
        with (
            patch("asciibench.generator.main.Settings", return_value=mock_settings),
            patch(
                "asciibench.generator.main.load_generation_config",
                return_value=mock_config,
            ),
            patch("asciibench.generator.main.load_models", return_value=mock_models),
            patch("asciibench.generator.main.load_prompts", return_value=mock_prompts),
            patch("asciibench.generator.main.generate_samples", return_value=[]),
        ):
            main()

        captured = capsys.readouterr()
        stripped_output = strip_ansi(captured.out)
        # Check for banner (contains ASCII art with these patterns)
        assert "___" in stripped_output  # Part of the ASCII art banner
        # Check for config summary (like demo.py)
        assert "models loaded from" in stripped_output
        assert "models.yaml" in stripped_output

    def test_main_calls_generate_samples_with_correct_args(
        self,
        mock_settings: MagicMock,
        mock_config: GenerationConfig,
        mock_models: list[Model],
        mock_prompts: list[Prompt],
        tmp_path: Path,
    ) -> None:
        """Main calls generate_samples with loaded configuration."""
        with (
            patch("asciibench.generator.main.Settings", return_value=mock_settings),
            patch(
                "asciibench.generator.main.load_generation_config",
                return_value=mock_config,
            ),
            patch("asciibench.generator.main.load_models", return_value=mock_models),
            patch("asciibench.generator.main.load_prompts", return_value=mock_prompts),
            patch("asciibench.generator.main.generate_samples", return_value=[]) as mock_generate,
        ):
            main()

        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["models"] == mock_models
        assert call_kwargs["prompts"] == mock_prompts
        assert call_kwargs["config"] == mock_config
        assert call_kwargs["settings"] == mock_settings
        assert call_kwargs["progress_callback"] is not None

    def test_main_prints_summary_with_valid_invalid_counts(
        self,
        mock_settings: MagicMock,
        mock_config: GenerationConfig,
        mock_models: list[Model],
        mock_prompts: list[Prompt],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Main prints summary with valid/invalid counts."""
        samples = [
            ArtSample(
                model_id="m1",
                prompt_text="p1",
                category="c1",
                attempt_number=1,
                raw_output="out",
                sanitized_output="out",
                is_valid=True,
            ),
            ArtSample(
                model_id="m1",
                prompt_text="p2",
                category="c1",
                attempt_number=1,
                raw_output="",
                sanitized_output="",
                is_valid=False,
            ),
            ArtSample(
                model_id="m1",
                prompt_text="p3",
                category="c1",
                attempt_number=1,
                raw_output="out",
                sanitized_output="out",
                is_valid=True,
            ),
        ]

        with (
            patch("asciibench.generator.main.Settings", return_value=mock_settings),
            patch(
                "asciibench.generator.main.load_generation_config",
                return_value=mock_config,
            ),
            patch("asciibench.generator.main.load_models", return_value=mock_models),
            patch("asciibench.generator.main.load_prompts", return_value=mock_prompts),
            patch("asciibench.generator.main.generate_samples", return_value=samples),
        ):
            main()

        captured = capsys.readouterr()
        assert "Generation Complete!" in captured.out
        assert "3" in captured.out  # Total samples
        assert "2" in captured.out  # Valid samples
        assert "1" in captured.out  # Invalid samples

    def test_main_prints_no_new_samples_message(
        self,
        mock_settings: MagicMock,
        mock_config: GenerationConfig,
        mock_models: list[Model],
        mock_prompts: list[Prompt],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Main prints message when no new samples generated (idempotent run)."""
        with (
            patch("asciibench.generator.main.Settings", return_value=mock_settings),
            patch(
                "asciibench.generator.main.load_generation_config",
                return_value=mock_config,
            ),
            patch("asciibench.generator.main.load_models", return_value=mock_models),
            patch("asciibench.generator.main.load_prompts", return_value=mock_prompts),
            patch("asciibench.generator.main.generate_samples", return_value=[]),
        ):
            main()

        captured = capsys.readouterr()
        assert "No new samples generated" in captured.out
        assert "already exist in database" in captured.out

    def test_main_missing_models_file_exits(self) -> None:
        """Missing models.yaml prints error and exits."""
        mock_settings = MagicMock()
        mock_settings.openrouter_api_key = "test-key"

        with (
            patch("asciibench.generator.main.Settings", return_value=mock_settings),
            patch(
                "asciibench.generator.main.load_generation_config",
                return_value=GenerationConfig(),
            ),
            patch(
                "asciibench.generator.main.load_models",
                side_effect=FileNotFoundError("models.yaml not found"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        exc = exc_info.value
        assert isinstance(exc, SystemExit)
        assert exc.code == 1

    def test_main_missing_prompts_file_exits(self) -> None:
        """Missing prompts.yaml prints error and exits."""
        mock_settings = MagicMock()
        mock_settings.openrouter_api_key = "test-key"

        with (
            patch("asciibench.generator.main.Settings", return_value=mock_settings),
            patch(
                "asciibench.generator.main.load_generation_config",
                return_value=GenerationConfig(),
            ),
            patch("asciibench.generator.main.load_models", return_value=[]),
            patch(
                "asciibench.generator.main.load_prompts",
                side_effect=FileNotFoundError("prompts.yaml not found"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        exc = exc_info.value
        assert isinstance(exc, SystemExit)
        assert exc.code == 1

    def test_main_nothing_to_generate_shows_warning(
        self,
        mock_settings: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Main shows warning when no models or prompts are configured."""
        config = GenerationConfig(attempts_per_prompt=5)
        # Empty models or prompts result in 0 expected samples
        models: list[Model] = []
        prompts: list[Prompt] = []

        with (
            patch("asciibench.generator.main.Settings", return_value=mock_settings),
            patch("asciibench.generator.main.load_generation_config", return_value=config),
            patch("asciibench.generator.main.load_models", return_value=models),
            patch("asciibench.generator.main.load_prompts", return_value=prompts),
        ):
            main()

        captured = capsys.readouterr()
        assert "Nothing to generate" in captured.out


class TestMainModuleEntry:
    """Test that the module can be run via python -m."""

    def test_module_has_main_guard(self) -> None:
        """Module has if __name__ == '__main__' guard."""
        import inspect

        import asciibench.generator.main as main_module

        source = inspect.getsource(main_module)
        assert 'if __name__ == "__main__":' in source
        assert "main()" in source
