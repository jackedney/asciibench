"""Thread-safe state container for concurrent sample generation.

This module provides a SharedState dataclass that allows concurrent tasks
to safely share and modify mutable state using asyncio.Lock for atomic
operations.
"""

import asyncio
from dataclasses import dataclass, field

from asciibench.common.logging import get_logger

logger = get_logger("generator.state")


@dataclass
class BatchMetrics:
    """Tracks metrics for a batch of sample generations."""

    total_samples: int = 0
    successful: int = 0
    failed: int = 0
    total_duration_ms: float = 0.0
    total_cost: float = 0.0

    def record_sample(self, success: bool, duration_ms: float, cost: float | None) -> None:
        """Record metrics for a single sample generation."""
        self.total_samples += 1
        if success:
            self.successful += 1
        else:
            self.failed += 1
        self.total_duration_ms += duration_ms
        if cost is not None:
            self.total_cost += cost

    def log_summary(self) -> None:
        """Log batch summary metrics."""
        logger.info(
            "Batch generation complete",
            {
                "total_samples": self.total_samples,
                "successful": self.successful,
                "failed": self.failed,
                "total_duration_ms": round(self.total_duration_ms, 2),
                "total_cost": round(self.total_cost, 6),
            },
        )


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
        current_tasks: Count of currently active concurrent tasks
        max_concurrent: Maximum concurrent tasks limit from config
        _lock: asyncio.Lock for thread-safe operations
    """

    existing_keys: set[tuple[str, str, int]] = field(default_factory=set)
    samples_processed: int = 0
    metrics: BatchMetrics = field(default_factory=BatchMetrics)
    current_tasks: int = 0
    max_concurrent: int = 10
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

    async def increment_concurrent(self) -> int:
        """Atomically increment the current concurrent tasks counter.

        This method is thread-safe and ensures that concurrent calls
        to increment_concurrent will correctly track active tasks.

        Returns:
            The new value of current_tasks after incrementing
        """
        async with self._lock:
            self.current_tasks += 1
            return self.current_tasks

    async def decrement_concurrent(self) -> int:
        """Atomically decrement the current concurrent tasks counter.

        This method is thread-safe and ensures that concurrent calls
        to decrement_concurrent will correctly track active tasks.

        Returns:
            The new value of current_tasks after decrementing
        """
        async with self._lock:
            if self.current_tasks == 0:
                logger.warning(
                    "Attempted to decrement concurrent tasks counter when already at 0",
                )
                return 0
            self.current_tasks -= 1
            return self.current_tasks

    async def maybe_log_concurrency(self) -> None:
        """Log concurrency metrics every 10 tasks processed.

        This method checks if the samples_processed count is a multiple of 10
        and logs concurrency metrics if so. Logging includes:
        - current_concurrent: Number of currently active tasks
        - max_concurrent: Maximum concurrent task limit
        - tasks_completed: Total tasks processed so far
        """
        async with self._lock:
            if self.samples_processed % 10 == 0 and self.samples_processed > 0:
                logger.info(
                    "Concurrency metrics",
                    {
                        "current_concurrent": self.current_tasks,
                        "max_concurrent": self.max_concurrent,
                        "tasks_completed": self.samples_processed,
                    },
                )
