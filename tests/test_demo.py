"""Tests for demo module."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from asciibench.common.models import DemoResult
from asciibench.generator.demo import (
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
