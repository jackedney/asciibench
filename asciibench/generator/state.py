"""Thread-safe state container for concurrent sample generation.

This module provides a SharedState dataclass that allows concurrent tasks
to safely share and modify mutable state using asyncio.Lock for atomic
operations.
"""

import asyncio
from dataclasses import dataclass, field

from asciibench.generator.sampler import BatchMetrics


@dataclass
class SharedState:
    """Thread-safe state container for concurrent sample generation.

    This class provides atomic operations for checking sample existence,
    tracking progress, and recording metrics across concurrent tasks.

    Attributes:
        existing_keys: Set of (model_id, prompt_text, attempt_number) tuples
            for existing samples
        samples_processed: Count of samples that have been processed
        metrics: BatchMetrics instance for tracking generation statistics
        _lock: asyncio.Lock for thread-safe operations
    """

    existing_keys: set[tuple[str, str, int]] = field(default_factory=set)
    samples_processed: int = 0
    metrics: BatchMetrics = field(default_factory=BatchMetrics)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def check_and_add_key(self, key: tuple[str, str, int]) -> bool:
        """Atomically check if key exists and add it if not.

        This method is thread-safe and ensures that only one concurrent
        task can successfully add a given key. Multiple concurrent calls
        with the same key will result in only one returning True.

        Args:
            key: Tuple of (model_id, prompt_text, attempt_number)

        Returns:
            True if the key was added (did not previously exist),
            False if the key already existed
        """
        async with self._lock:
            if key in self.existing_keys:
                return False
            self.existing_keys.add(key)
            return True

    async def increment_processed(self) -> int:
        """Atomically increment the samples processed counter.

        This method is thread-safe and ensures that concurrent calls
        to increment_processed will correctly count all samples.

        Returns:
            The new value of samples_processed after incrementing
        """
        async with self._lock:
            self.samples_processed += 1
            return self.samples_processed

    async def record_sample(self, success: bool, duration_ms: float, cost: float | None) -> None:
        """Record a sample generation in a thread-safe manner.

        This method forwards the record to the underlying BatchMetrics
        instance in a thread-safe way.

        Args:
            success: Whether the sample generation was successful
            duration_ms: Duration of the generation in milliseconds
            cost: Cost of the generation (may be None)
        """
        async with self._lock:
            self.metrics.record_sample(success, duration_ms, cost)
