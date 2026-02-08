"""Structured JSON logging module for asciibench."""

import json
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def generate_id() -> str:
    """Generate a UUID with timestamp-based fallback.

    Returns:
        UUID string as fallback: ISO8601 timestamp with microseconds

    Example:
        >>> id = generate_id()
        >>> isinstance(id, str)
        True
        >>> len(id) > 0
        True

    Negative case:
        >>> If UUID generation somehow fails, returns timestamp-based ID
        >>> id = generate_id()
        >>> 'T' in id or '-' in id  # Timestamp contains date/time separators
        True
    """
    try:
        return str(uuid.uuid4())
    except OSError:
        # Fallback to timestamp if UUID generation fails (e.g., no entropy source)
        return datetime.now(tz=UTC).isoformat()


def set_run_id(run_id: str | None) -> None:
    """Set the run ID for the current context.

    Args:
        run_id: Run ID string or None to clear

    Example:
        >>> set_run_id("abc123")
        >>> get_run_id()
        'abc123'
        >>> set_run_id(None)
        >>> get_run_id() is None
        True
    """
    _run_id.set(run_id)


def get_run_id() -> str | None:
    """Get the current run ID from context.

    Returns:
        Run ID string or None if not set

    Example:
        >>> get_run_id() is None or isinstance(get_run_id(), str)
        True
    """
    return _run_id.get()


def set_request_id(request_id: str | None) -> None:
    """Set the request ID for the current context.

    Args:
        request_id: Request ID string or None to clear

    Example:
        >>> set_request_id("req-456")
        >>> get_request_id()
        'req-456'
        >>> set_request_id(None)
        >>> get_request_id() is None
        True
    """
    _request_id.set(request_id)


def get_request_id() -> str | None:
    """Get the current request ID from context.

    Returns:
        Request ID string or None if not set

    Example:
        >>> get_request_id() is None or isinstance(get_request_id(), str)
        True
    """
    return _request_id.get()


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
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
        }

        run_id = get_run_id()
        if run_id:
            entry["run_id"] = run_id

        request_id = get_request_id()
        if request_id:
            entry["request_id"] = request_id

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
