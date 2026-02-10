"""Data repository for unified data access with caching support.

This module provides a DataRepository class that encapsulates data loading
patterns for JSONL files, with optional caching to reduce redundant file I/O.

Caching behavior:
    - Caching is optional and can be disabled by setting cache_ttl to 0 or None
    - When enabled, data is cached for the specified TTL (time-to-live) in seconds
    - Each data type (samples, votes, evaluations) has its own cache entry
    - Cache entries expire after the TTL and are reloaded on next access
    - Cache is instance-based, not global (different repository instances have different caches)

Example:
    >>> repo = DataRepository(cache_ttl=60)  # Cache for 60 seconds
    >>> samples = repo.get_valid_samples()
    >>> # Second call within 60 seconds returns cached data
    >>> samples_again = repo.get_valid_samples()
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from asciibench.common.models import ArtSample, VLMEvaluation, Vote

try:
    from asciibench.common.persistence import read_jsonl as _persistence_read_jsonl
except (SyntaxError, ImportError):
    _persistence_read_jsonl: Callable[[Path, type[Any]], list[Any]] | None = None


T = TypeVar("T", bound=BaseModel)

_ReadJsonlFn = Callable[[Path, type[Any]], list[Any]]


def _read_jsonl_fn(path: Path, model_class: type[Any]) -> list[Any]:
    return _read_jsonl(path, model_class)


@dataclass
class CacheEntry(Generic[T]):
    """A cached data entry with timestamp for TTL-based expiration."""

    data: list[T]
    cached_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def is_expired(self, ttl_seconds: int | None) -> bool:
        """Check if the cache entry has expired based on TTL.

        Args:
            ttl_seconds: TTL in seconds. If None, the cache never expires.

        Returns:
            True if expired, False otherwise.
        """
        if ttl_seconds is None:
            return False
        if ttl_seconds <= 0:
            return True
        expiry = self.cached_at + timedelta(seconds=ttl_seconds)
        return datetime.now(UTC) > expiry


def _read_jsonl(path: Path, model_class: type[T]) -> list[T]:
    """Read all lines from a JSONL file as model instances.

    Args:
        path: Path to the JSONL file
        model_class: Pydantic model class to parse each line as

    Returns:
        List of model instances

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    results: list[T] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(model_class.model_validate_json(line))

    return results


def _get_read_jsonl_fn():
    """Get the read_jsonl function to use.

    Returns:
        The read_jsonl function from persistence module if available,
        otherwise the local implementation.
    """
    if _persistence_read_jsonl is not None:
        return _persistence_read_jsonl
    return _read_jsonl


class DataRepository:
    """Unified data repository for accessing JSONL data with optional caching.

    This class provides a single point of access for loading data from JSONL files,
    reducing code duplication across the codebase. Caching can be enabled to improve
    performance for frequently accessed data.

    The repository is instance-based, not a singleton. Each instance maintains its
    own cache, allowing different caching strategies for different use cases.

    Args:
        data_dir: Directory containing data files (default: "data")
        cache_ttl: Cache time-to-live in seconds. None = never expire, 0 = disable.

    Example:
        >>> # Basic usage without caching
        >>> repo = DataRepository()
        >>> samples = repo.get_all_samples()
        >>> valid_samples = repo.get_valid_samples()

        >>> # With caching (60 second TTL)
        >>> repo = DataRepository(cache_ttl=60)
        >>> samples = repo.get_all_samples()  # Loads from file
        >>> samples = repo.get_all_samples()  # Returns cached data

        >>> # Disable caching explicitly
        >>> repo = DataRepository(cache_ttl=0)

    Raises:
        FileNotFoundError: If a data file path is invalid or doesn't exist when
            attempting to read from it. The error message includes the specific
            file path that couldn't be found.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        cache_ttl: int | None = None,
    ) -> None:
        """Initialize the DataRepository.

        Args:
            data_dir: Path to data directory containing JSONL files.
            cache_ttl: Cache time-to-live in seconds. None = never expire, 0 = disable.
        """
        self._data_dir = Path(data_dir)
        self._cache_ttl = cache_ttl
        self._samples_cache: CacheEntry[ArtSample] | None = None
        self._votes_cache: CacheEntry[Vote] | None = None
        self._evaluations_cache: CacheEntry[VLMEvaluation] | None = None

    @property
    def database_path(self) -> Path:
        """Get the path to the database.jsonl file."""
        return self._data_dir / "database.jsonl"

    @property
    def votes_path(self) -> Path:
        """Get the path to the votes.jsonl file."""
        return self._data_dir / "votes.jsonl"

    @property
    def evaluations_path(self) -> Path:
        """Get the path to the vlm_evaluations.jsonl file."""
        return self._data_dir / "vlm_evaluations.jsonl"

    def _get_cached_or_load(
        self,
        path: Path,
        model_class: type[T],
        cache_attr_name: str,
    ) -> list[T]:
        """Load data from cache if valid, otherwise from file.

        Args:
            path: Path to JSONL file.
            model_class: Pydantic model class to parse data.
            cache_attr_name: Name of cache attribute on this instance.

        Returns:
            List of parsed model instances.

        Raises:
            FileNotFoundError: If data directory is invalid or doesn't exist.
        """
        cache_entry = getattr(self, cache_attr_name)

        if cache_entry is not None and not cache_entry.is_expired(self._cache_ttl):
            return cache_entry.data

        if not self._data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self._data_dir}. "
                f"Ensure that data directory exists and contains the required JSONL files."
            )

        read_fn = _get_read_jsonl_fn()
        data = read_fn(path, model_class)
        setattr(
            self,
            cache_attr_name,
            CacheEntry(data=data),
        )
        return data

    def get_all_samples(self) -> list[ArtSample]:
        """Get all samples from the database.

        Returns:
            List of all ArtSample objects from database.jsonl.

        Raises:
            FileNotFoundError: If database.jsonl doesn't exist or the data directory is invalid.

        Example:
            >>> repo = DataRepository()
            >>> samples = repo.get_all_samples()
            >>> print(f"Loaded {len(samples)} samples")
        """
        return self._get_cached_or_load(
            self.database_path,
            ArtSample,
            "_samples_cache",
        )

    def get_valid_samples(self) -> list[ArtSample]:
        """Get only valid samples from the database.

        This is equivalent to filtering get_all_samples() for is_valid=True,
        but uses caching independently for better performance when this is
        the primary access pattern.

        Returns:
            List of valid ArtSample objects (where is_valid=True).

        Raises:
            FileNotFoundError: If database.jsonl doesn't exist or the data directory is invalid.

        Example:
            >>> repo = DataRepository()
            >>> valid = repo.get_valid_samples()
            >>> print(f"Loaded {len(valid)} valid samples")
        """
        all_samples = self.get_all_samples()
        return [s for s in all_samples if s.is_valid]

    def get_votes(self) -> list[Vote]:
        """Get all votes from the votes file.

        Returns:
            List of all Vote objects from votes.jsonl.

        Raises:
            FileNotFoundError: If votes.jsonl doesn't exist or the data directory is invalid.

        Example:
            >>> repo = DataRepository()
            >>> votes = repo.get_votes()
            >>> print(f"Loaded {len(votes)} votes")
        """
        return self._get_cached_or_load(
            self.votes_path,
            Vote,
            "_votes_cache",
        )

    def get_evaluations(self) -> list[VLMEvaluation]:
        """Get all VLM evaluations from the evaluations file.

        Returns:
            List of all VLMEvaluation objects from vlm_evaluations.jsonl.

        Raises:
            FileNotFoundError: If vlm_evaluations.jsonl doesn't exist or data directory is invalid.

        Example:
            >>> repo = DataRepository()
            >>> evaluations = repo.get_evaluations()
            >>> print(f"Loaded {len(evaluations)} evaluations")
        """
        return self._get_cached_or_load(
            self.evaluations_path,
            VLMEvaluation,
            "_evaluations_cache",
        )

    def clear_cache(self) -> None:
        """Clear all cached data.

        Forces the next data access to reload from files. This is useful when
        external processes may have modified the data files.

        Example:
            >>> repo = DataRepository(cache_ttl=60)
            >>> samples = repo.get_all_samples()  # Cached
            >>> # External process modifies database.jsonl
            >>> repo.clear_cache()
            >>> samples = repo.get_all_samples()  # Reloads from file
        """
        self._samples_cache = None
        self._votes_cache = None
        self._evaluations_cache = None

    def get_all_samples_or_empty(self) -> list[ArtSample]:
        """Get all samples, returning empty list if file not found."""
        try:
            return self.get_all_samples()
        except FileNotFoundError:
            return []

    def get_valid_samples_or_empty(self) -> list[ArtSample]:
        """Get valid samples, returning empty list if file not found."""
        try:
            return self.get_valid_samples()
        except FileNotFoundError:
            return []

    def get_votes_or_empty(self) -> list[Vote]:
        """Get votes, returning empty list if file not found."""
        try:
            return self.get_votes()
        except FileNotFoundError:
            return []

    def get_evaluations_or_empty(self) -> list[VLMEvaluation]:
        """Get evaluations, returning empty list if file not found."""
        try:
            return self.get_evaluations()
        except FileNotFoundError:
            return []

    @staticmethod
    def build_sample_lookup(samples: list[ArtSample]) -> dict[str, ArtSample]:
        """Build {sample_id_str: ArtSample} mapping."""
        return {str(s.id): s for s in samples}

    @staticmethod
    def build_sample_model_lookup(samples: list[ArtSample]) -> dict[str, str]:
        """Build {sample_id_str: model_id} mapping."""
        return {str(s.id): s.model_id for s in samples}

    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled for this repository instance.

        Returns:
            True if caching is enabled (cache_ttl > 0), False otherwise.
        """
        return self._cache_ttl is not None and self._cache_ttl > 0
