"""Tests for structured JSON logging module."""

import json
from pathlib import Path

from asciibench.common.logging import (
    JSONLogger,
    generate_id,
    get_logger,
    get_request_id,
    get_run_id,
    set_request_id,
    set_run_id,
)


class TestJSONLogger:
    """Tests for JSONLogger class."""

    def test_creates_log_file_if_not_exists(self, tmp_path: Path) -> None:
        """JSONLogger creates log file if it doesn't exist."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)
        logger.info("Test message")

        assert log_path.exists()

    def test_writes_valid_json_line(self, tmp_path: Path) -> None:
        """Log entry is a valid JSON line."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)
        logger.info("Sample generated", {"model": "gpt-4", "duration_ms": 1234})

        content = log_path.read_text()
        line = content.strip()
        entry = json.loads(line)

        assert entry["level"] == "info"
        assert entry["message"] == "Sample generated"
        assert entry["logger"] == "test"
        assert "timestamp" in entry

    def test_includes_timestamp_iso8601(self, tmp_path: Path) -> None:
        """Log entry includes ISO8601 timestamp."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)
        logger.info("Test message")

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert "timestamp" in entry
        assert isinstance(entry["timestamp"], str)
        assert "T" in entry["timestamp"] or "-" in entry["timestamp"]

    def test_includes_level(self, tmp_path: Path) -> None:
        """Log entry includes level field."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)

        logger.info("Info message")
        logger.error("Error message")
        logger.warning("Warning message")
        logger.debug("Debug message")

        lines = log_path.read_text().strip().split("\n")
        assert json.loads(lines[0])["level"] == "info"
        assert json.loads(lines[1])["level"] == "error"
        assert json.loads(lines[2])["level"] == "warning"
        assert json.loads(lines[3])["level"] == "debug"

    def test_includes_logger_name(self, tmp_path: Path) -> None:
        """Log entry includes logger name."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("my.module.name", log_path)
        logger.info("Test")

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert entry["logger"] == "my.module.name"

    def test_includes_metadata(self, tmp_path: Path) -> None:
        """Log entry includes optional metadata dict."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)
        logger.info("Sample generated", {"model": "gpt-4", "duration_ms": 1234, "tokens": 500})

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert "metadata" in entry
        assert entry["metadata"]["model"] == "gpt-4"
        assert entry["metadata"]["duration_ms"] == 1234
        assert entry["metadata"]["tokens"] == 500

    def test_handles_non_serializable_objects(self, tmp_path: Path) -> None:
        """Non-serializable objects are converted to string representation."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)

        class CustomObject:
            def __str__(self) -> str:
                return "CustomObject()"

        obj = CustomObject()
        logger.info("Test with object", {"custom": obj})

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert "metadata" in entry
        assert entry["metadata"]["custom"] == "CustomObject()"

    def test_handles_nested_non_serializable_objects(self, tmp_path: Path) -> None:
        """Nested non-serializable objects are converted to string representation."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)

        class Inner:
            def __str__(self) -> str:
                return "Inner()"

        logger.info("Test nested", {"outer": {"inner": Inner()}})

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert "metadata" in entry
        assert entry["metadata"]["outer"]["inner"] == "Inner()"

    def test_handles_non_serializable_in_list(self, tmp_path: Path) -> None:
        """Non-serializable objects in lists are converted to string representation."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)

        class Item:
            def __str__(self) -> str:
                return "Item()"

        logger.info("Test list", {"items": [Item(), 1, 2]})

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert "metadata" in entry
        assert entry["metadata"]["items"] == ["Item()", 1, 2]

    def test_metadata_optional(self, tmp_path: Path) -> None:
        """Metadata dict is optional."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)

        logger.info("Message without metadata")

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert "metadata" not in entry
        assert entry["message"] == "Message without metadata"

    def test_empty_metadata(self, tmp_path: Path) -> None:
        """Empty metadata dict is not included."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)

        logger.info("Message", {})

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert "metadata" not in entry

    def test_multiple_log_entries(self, tmp_path: Path) -> None:
        """Multiple log entries are written correctly."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)

        logger.info("First message", {"id": 1})
        logger.error("Error occurred", {"error_type": "ValueError"})
        logger.warning("Warning", {"count": 3})

        content = log_path.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 3
        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])
        entry3 = json.loads(lines[2])

        assert entry1["message"] == "First message"
        assert entry1["metadata"]["id"] == 1
        assert entry2["level"] == "error"
        assert entry3["level"] == "warning"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Logger creates parent directories if needed."""
        log_path = tmp_path / "nested" / "dir" / "logs.jsonl"
        logger = JSONLogger("test", log_path)
        logger.info("Test")

        assert log_path.exists()

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """Logger accepts string paths."""
        log_path = str(tmp_path / "logs.jsonl")
        logger = JSONLogger("test", log_path)
        logger.info("Test")

        assert Path(log_path).exists()

    def test_default_log_path(self, tmp_path: Path) -> None:
        """Logger uses default log path when not specified."""
        logger = JSONLogger("test", tmp_path / "default_logs.jsonl")
        logger.info("Default path test")

        assert (tmp_path / "default_logs.jsonl").exists()

    def test_info_method(self, tmp_path: Path) -> None:
        """info() method logs at info level."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)
        logger.info("Info message")

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert entry["level"] == "info"
        assert entry["message"] == "Info message"

    def test_error_method(self, tmp_path: Path) -> None:
        """error() method logs at error level."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)
        logger.error("Error message")

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert entry["level"] == "error"
        assert entry["message"] == "Error message"

    def test_warning_method(self, tmp_path: Path) -> None:
        """warning() method logs at warning level."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)
        logger.warning("Warning message")

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert entry["level"] == "warning"
        assert entry["message"] == "Warning message"

    def test_debug_method(self, tmp_path: Path) -> None:
        """debug() method logs at debug level."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)
        logger.debug("Debug message")

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert entry["level"] == "debug"
        assert entry["message"] == "Debug message"


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger_instance(self, tmp_path: Path) -> None:
        """get_logger returns a JSONLogger instance."""
        log_path = tmp_path / "logs.jsonl"
        logger = get_logger("test")
        logger.log_path = log_path
        logger.info("Test")

        assert isinstance(logger, JSONLogger)

    def test_same_name_returns_same_instance(self, tmp_path: Path) -> None:
        """get_logger returns same instance for same name."""
        log_path = tmp_path / "logs.jsonl"
        logger1 = get_logger("test")
        logger1.log_path = log_path

        logger2 = get_logger("test")
        logger2.log_path = log_path

        assert logger1 is logger2

    def test_different_name_returns_different_instance(self, tmp_path: Path) -> None:
        """get_logger returns different instance for different names."""
        log_path1 = tmp_path / "logs1.jsonl"
        log_path2 = tmp_path / "logs2.jsonl"

        logger1 = get_logger("module1")
        logger1.log_path = log_path1

        logger2 = get_logger("module2")
        logger2.log_path = log_path2

        assert logger1 is not logger2
        assert logger1.name == "module1"
        assert logger2.name == "module2"


class TestIDGeneration:
    """Tests for ID generation functions."""

    def test_generate_id_returns_string(self) -> None:
        """generate_id returns a string."""
        id_str = generate_id()
        assert isinstance(id_str, str)
        assert len(id_str) > 0

    def test_generate_id_unique(self) -> None:
        """generate_id returns unique IDs."""
        id1 = generate_id()
        id2 = generate_id()
        assert id1 != id2

    def test_generate_id_is_uuid_or_timestamp(self) -> None:
        """generate_id returns UUID or timestamp-based ID."""
        id_str = generate_id()
        # Either a UUID with dashes or a timestamp with T or -
        assert "-" in id_str or "T" in id_str


class TestRunIDContext:
    """Tests for run_id context management."""

    def test_set_and_get_run_id(self) -> None:
        """set_run_id and get_run_id work correctly."""
        set_run_id("test-run-123")
        assert get_run_id() == "test-run-123"

    def test_get_run_id_none_by_default(self) -> None:
        """get_run_id returns None when not set."""
        set_run_id(None)
        assert get_run_id() is None

    def test_clear_run_id(self) -> None:
        """set_run_id(None) clears the run_id."""
        set_run_id("test-run")
        assert get_run_id() == "test-run"
        set_run_id(None)
        assert get_run_id() is None


class TestRequestIDContext:
    """Tests for request_id context management."""

    def test_set_and_get_request_id(self) -> None:
        """set_request_id and get_request_id work correctly."""
        set_request_id("test-req-456")
        assert get_request_id() == "test-req-456"

    def test_get_request_id_none_by_default(self) -> None:
        """get_request_id returns None when not set."""
        set_request_id(None)
        assert get_request_id() is None

    def test_clear_request_id(self) -> None:
        """set_request_id(None) clears the request_id."""
        set_request_id("test-req")
        assert get_request_id() == "test-req"
        set_request_id(None)
        assert get_request_id() is None


class TestLoggingWithIDs:
    """Tests for logging with run_id and request_id."""

    def test_log_includes_run_id_when_set(self, tmp_path: Path) -> None:
        """Log entry includes run_id when set."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)

        set_run_id("run-123")
        logger.info("Test message")
        set_run_id(None)

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert "run_id" in entry
        assert entry["run_id"] == "run-123"

    def test_log_includes_request_id_when_set(self, tmp_path: Path) -> None:
        """Log entry includes request_id when set."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)

        set_request_id("req-456")
        logger.info("Test message")
        set_request_id(None)

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert "request_id" in entry
        assert entry["request_id"] == "req-456"

    def test_log_includes_both_ids_when_set(self, tmp_path: Path) -> None:
        """Log entry includes both run_id and request_id when set."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)

        set_run_id("run-123")
        set_request_id("req-456")
        logger.info("Test message")
        set_run_id(None)
        set_request_id(None)

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert entry["run_id"] == "run-123"
        assert entry["request_id"] == "req-456"

    def test_log_omits_run_id_when_not_set(self, tmp_path: Path) -> None:
        """Log entry omits run_id when not set."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)

        set_run_id(None)
        logger.info("Test message")

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert "run_id" not in entry

    def test_log_omits_request_id_when_not_set(self, tmp_path: Path) -> None:
        """Log entry omits request_id when not set."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)

        set_request_id(None)
        logger.info("Test message")

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert "request_id" not in entry

    def test_multiple_logs_same_run_id(self, tmp_path: Path) -> None:
        """Multiple logs share the same run_id when not changed."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)

        set_run_id("run-123")
        logger.info("First message")
        logger.info("Second message")
        logger.info("Third message")
        set_run_id(None)

        content = log_path.read_text()
        lines = content.strip().split("\n")

        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])
        entry3 = json.loads(lines[2])

        assert entry1["run_id"] == "run-123"
        assert entry2["run_id"] == "run-123"
        assert entry3["run_id"] == "run-123"

    def test_multiple_logs_different_request_ids(self, tmp_path: Path) -> None:
        """Logs can have different request_ids."""
        log_path = tmp_path / "logs.jsonl"
        logger = JSONLogger("test", log_path)

        set_run_id("run-123")

        set_request_id("req-1")
        logger.info("First request")

        set_request_id("req-2")
        logger.info("Second request")

        set_request_id("req-3")
        logger.info("Third request")

        set_run_id(None)
        set_request_id(None)

        content = log_path.read_text()
        lines = content.strip().split("\n")

        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])
        entry3 = json.loads(lines[2])

        assert entry1["run_id"] == "run-123"
        assert entry2["run_id"] == "run-123"
        assert entry3["run_id"] == "run-123"
        assert entry1["request_id"] == "req-1"
        assert entry2["request_id"] == "req-2"
        assert entry3["request_id"] == "req-3"


class TestIntegration:
    """Integration tests for logging module."""

    def test_full_logging_workflow(self, tmp_path: Path) -> None:
        """Complete logging workflow as specified in requirements."""
        log_path = tmp_path / "asciibench.jsonl"
        logger = JSONLogger("generator.sampler", log_path)

        logger.info("Sample generated", {"model": "gpt-4", "duration_ms": 1234})

        content = log_path.read_text()
        entry = json.loads(content.strip())

        assert entry["timestamp"] is not None
        assert entry["level"] == "info"
        assert entry["logger"] == "generator.sampler"
        assert entry["message"] == "Sample generated"
        assert entry["metadata"]["model"] == "gpt-4"
        assert entry["metadata"]["duration_ms"] == 1234

    def test_multiple_loggers_same_file(self, tmp_path: Path) -> None:
        """Multiple loggers can write to the same file."""
        log_path = tmp_path / "shared_logs.jsonl"
        logger1 = JSONLogger("module1", log_path)
        logger2 = JSONLogger("module2", log_path)

        logger1.info("Message from module1", {"id": 1})
        logger2.error("Error from module2", {"id": 2})

        content = log_path.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 2
        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])

        assert entry1["logger"] == "module1"
        assert entry2["logger"] == "module2"
