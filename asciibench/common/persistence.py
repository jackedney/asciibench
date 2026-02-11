"""JSONL persistence utilities for reading and writing Pydantic models."""

import tempfile
from pathlib import Path
from typing import TypeVar
from uuid import UUID

from filelock import FileLock
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def append_jsonl(path: str | Path, obj: BaseModel) -> None:
    """Append a Pydantic model as a JSON line to a JSONL file.

    Uses file locking to prevent corruption from concurrent writes.
    Creates the file if it doesn't exist.

    Args:
        path: Path to the JSONL file
        obj: Pydantic model instance to append
    """
    path = Path(path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = path.with_suffix(path.suffix + ".lock")
    with FileLock(lock_path):
        with path.open("a", encoding="utf-8") as f:
            f.write(obj.model_dump_json() + "\n")


def read_jsonl(path: str | Path, model_class: type[T]) -> list[T]:
    """Read all lines from a JSONL file as model instances.

    Args:
        path: Path to the JSONL file
        model_class: Pydantic model class to parse each line as

    Returns:
        List of model instances

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    results: list[T] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(model_class.model_validate_json(line))

    return results


def read_jsonl_by_id(path: str | Path, id: UUID | str, model_class: type[T]) -> T | None:
    """Find a single record by UUID from a JSONL file.

    Args:
        path: Path to the JSONL file
        id: UUID to search for (can be UUID object or string)
        model_class: Pydantic model class to parse each line as

    Returns:
        The matching model instance, or None if not found
    """
    path = Path(path)

    if not path.exists():
        return None

    # Convert string to UUID if needed for consistent comparison
    target_id = UUID(str(id)) if not isinstance(id, UUID) else id

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = model_class.model_validate_json(line)
                # Access id field - assumes model has an 'id' field
                if hasattr(obj, "id") and obj.id == target_id:
                    return obj

    return None


def write_jsonl(path: str | Path, objects: list[T]) -> None:
    """Write a list of Pydantic models to a JSONL file atomically.

    Uses file locking and atomic write (write to temp, then rename) to
    prevent corruption from concurrent writes or interrupted operations.

    Args:
        path: Path to the JSONL file
        objects: List of Pydantic model instances to write
    """
    path = Path(path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = path.with_suffix(path.suffix + ".lock")
    with FileLock(lock_path):
        # Write to a temporary file in same directory for atomic rename
        fd, temp_path_str = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temp_path = Path(temp_path_str)
        try:
            with open(fd, "w", encoding="utf-8") as f:
                for obj in objects:
                    f.write(obj.model_dump_json() + "\n")
            # Atomic rename using pathlib
            temp_path.replace(path)
        finally:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
