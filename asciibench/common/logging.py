"""Structured JSON logging module for asciibench."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from filelock import FileLock


class JSONLogger:
    """Logger that writes JSON Lines to a file with consistent metadata."""

    def __init__(self, name: str, log_path: str | Path = "data/logs/asciibench.jsonl"):
        self.name = name
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._log_path.with_suffix(self._log_path.suffix + ".lock")

    @property
    def log_path(self) -> Path:
        """Get the log file path."""
        return self._log_path

    @log_path.setter
    def log_path(self, value: str | Path) -> None:
        """Set the log file path."""
        self._log_path = Path(value)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._log_path.with_suffix(self._log_path.suffix + ".lock")

    def _serialize_value(self, value: Any) -> Any:
        """Convert non-serializable values to string representation.

        Args:
            value: Value to serialize

        Returns:
            JSON-serializable value
        """
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._serialize_value(v) for k, v in value.items()}
        return str(value)

    def _log(self, level: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        """Write a log entry as a JSON line.

        Args:
            level: Log level (e.g., "info", "error", "warning", "debug")
            message: Log message
            metadata: Optional metadata dict to include in log entry
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
        }

        if metadata:
            entry["metadata"] = self._serialize_value(metadata)

        json_line = json.dumps(entry, ensure_ascii=False)

        with FileLock(self._lock_path):
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json_line + "\n")

    def info(self, message: str, metadata: dict[str, Any] | None = None) -> None:
        """Log an info message.

        Args:
            message: Log message
            metadata: Optional metadata dict
        """
        self._log("info", message, metadata)

    def error(self, message: str, metadata: dict[str, Any] | None = None) -> None:
        """Log an error message.

        Args:
            message: Log message
            metadata: Optional metadata dict
        """
        self._log("error", message, metadata)

    def warning(self, message: str, metadata: dict[str, Any] | None = None) -> None:
        """Log a warning message.

        Args:
            message: Log message
            metadata: Optional metadata dict
        """
        self._log("warning", message, metadata)

    def debug(self, message: str, metadata: dict[str, Any] | None = None) -> None:
        """Log a debug message.

        Args:
            message: Log message
            metadata: Optional metadata dict
        """
        self._log("debug", message, metadata)


_loggers: dict[str, JSONLogger] = {}


def get_logger(name: str) -> JSONLogger:
    """Get or create a logger with the given name.

    Args:
        name: Logger name (typically module name, e.g., "generator.sampler")

    Returns:
        JSONLogger instance
    """
    if name not in _loggers:
        _loggers[name] = JSONLogger(name)
    return _loggers[name]
