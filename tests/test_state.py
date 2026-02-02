"""Tests for SharedState thread-safe state container."""

import asyncio

import pytest

from asciibench.generator.sampler import BatchMetrics
from asciibench.generator.state import SharedState


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestSharedState:
    """Test SharedState class initialization and basic functionality."""

    def test_initialization(self):
        """Test SharedState initializes with default values."""
        state = SharedState()
        assert state.existing_keys == set()
        assert state.samples_processed == 0
        assert isinstance(state.metrics, BatchMetrics)
        assert state.metrics.total_samples == 0
        assert state.metrics.successful == 0
        assert state.metrics.failed == 0

    def test_initialization_with_existing_keys(self):
        """Test SharedState can be initialized with existing keys."""
        existing = {("model1", "prompt1", 1), ("model2", "prompt2", 2)}
        state = SharedState(existing_keys=existing)
        assert state.existing_keys == existing
        assert len(state.existing_keys) == 2


class TestCheckAndAddKey:
    """Test check_and_add_key method for atomic idempotency."""

    def test_check_and_add_new_key(self):
        """Test adding a new key returns True."""
        state = SharedState()
        key = ("model1", "prompt1", 1)
        result = asyncio.run(state.check_and_add_key(key))
        assert result is True
        assert key in state.existing_keys

    def test_check_and_add_existing_key(self):
        """Test adding an existing key returns False."""
        state = SharedState(existing_keys={("model1", "prompt1", 1)})
        key = ("model1", "prompt1", 1)
        result = asyncio.run(state.check_and_add_key(key))
        assert result is False

    def test_concurrent_check_and_add_same_key(self):
        """Test that concurrent calls with same key result in only one success."""
        state = SharedState()
        key = ("model1", "prompt1", 1)

        async def try_add():
            return await state.check_and_add_key(key)

        async def run_test():
            results = await asyncio.gather(*[try_add() for _ in range(10)])
            success_count = sum(1 for r in results if r)
            assert success_count == 1
            assert len(state.existing_keys) == 1

        asyncio.run(run_test())

    def test_concurrent_check_and_add_different_keys(self):
        """Test that concurrent calls with different keys all succeed."""
        state = SharedState()
        keys = [(f"model{i}", f"prompt{i}", i) for i in range(10)]

        async def try_add(key):
            return await state.check_and_add_key(key)

        async def run_test():
            results = await asyncio.gather(*[try_add(k) for k in keys])
            assert all(results)
            assert len(state.existing_keys) == 10

        asyncio.run(run_test())


class TestIncrementProcessed:
    """Test increment_processed method for atomic counter."""

    def test_increment_processed_returns_one(self):
        """Test first increment returns 1."""
        state = SharedState()
        result = asyncio.run(state.increment_processed())
        assert result == 1
        assert state.samples_processed == 1

    def test_increment_processed_multiple_times(self):
        """Test multiple increments accumulate correctly."""
        state = SharedState()
        for expected in range(1, 6):
            result = asyncio.run(state.increment_processed())
            assert result == expected
        assert state.samples_processed == 5

    def test_concurrent_increment_processed(self):
        """Test 10 concurrent increments result in final count of 10."""
        state = SharedState()

        async def increment():
            await state.increment_processed()

        async def run_test():
            await asyncio.gather(*[increment() for _ in range(10)])
            assert state.samples_processed == 10

        asyncio.run(run_test())

    def test_concurrent_increment_processed_all_distinct(self):
        """Test that concurrent increments produce distinct return values."""
        state = SharedState()

        async def run_test():
            results = await asyncio.gather(*[state.increment_processed() for _ in range(10)])
            assert sorted(results) == list(range(1, 11))

        asyncio.run(run_test())


class TestRecordSample:
    """Test record_sample method for thread-safe metrics."""

    def test_record_sample_success(self):
        """Test recording a successful sample."""
        state = SharedState()
        asyncio.run(state.record_sample(success=True, duration_ms=100.0, cost=0.001))
        assert state.metrics.total_samples == 1
        assert state.metrics.successful == 1
        assert state.metrics.failed == 0
        assert state.metrics.total_duration_ms == 100.0
        assert state.metrics.total_cost == 0.001

    def test_record_sample_failure(self):
        """Test recording a failed sample."""
        state = SharedState()
        asyncio.run(state.record_sample(success=False, duration_ms=50.0, cost=None))
        assert state.metrics.total_samples == 1
        assert state.metrics.successful == 0
        assert state.metrics.failed == 1
        assert state.metrics.total_duration_ms == 50.0
        assert state.metrics.total_cost == 0.0

    def test_record_sample_multiple(self):
        """Test recording multiple samples accumulates correctly."""
        state = SharedState()
        asyncio.run(state.record_sample(success=True, duration_ms=100.0, cost=0.001))
        asyncio.run(state.record_sample(success=True, duration_ms=150.0, cost=0.002))
        asyncio.run(state.record_sample(success=False, duration_ms=50.0, cost=None))
        assert state.metrics.total_samples == 3
        assert state.metrics.successful == 2
        assert state.metrics.failed == 1
        assert state.metrics.total_duration_ms == 300.0
        assert state.metrics.total_cost == 0.003

    def test_concurrent_record_sample(self):
        """Test concurrent recording of samples is thread-safe."""
        state = SharedState()

        async def record(i):
            success = i % 2 == 0
            await state.record_sample(success=success, duration_ms=float(i), cost=0.001)

        async def run_test():
            await asyncio.gather(*[record(i) for i in range(10)])
            assert state.metrics.total_samples == 10
            assert state.metrics.successful == 5
            assert state.metrics.failed == 5
            assert state.metrics.total_duration_ms == sum(range(10))
            assert state.metrics.total_cost == pytest.approx(0.01)

        asyncio.run(run_test())


class TestIntegration:
    """Integration tests for SharedState with mixed operations."""

    def test_mixed_concurrent_operations(self):
        """Test that mixed concurrent operations work correctly."""
        state = SharedState()

        async def task(i):
            key = (f"model{i}", f"prompt{i}", 1)
            added = await state.check_and_add_key(key)
            if added:
                await state.increment_processed()
                await state.record_sample(success=True, duration_ms=100.0, cost=0.001)

        async def run_test():
            await asyncio.gather(*[task(i) for i in range(10)])
            assert len(state.existing_keys) == 10
            assert state.samples_processed == 10
            assert state.metrics.total_samples == 10
            assert state.metrics.successful == 10

        asyncio.run(run_test())
