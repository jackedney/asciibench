"""Tests for DataRepository class."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import sleep
from uuid import uuid4

import pytest

from asciibench.common.models import ArtSample, VLMEvaluation, Vote
from asciibench.common.repository import CacheEntry, DataRepository


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self) -> None:
        """CacheEntry stores data and timestamp correctly."""
        data = [
            Vote(
                id=uuid4(),
                sample_a_id="test-a",
                sample_b_id="test-b",
                winner="A",
                timestamp=datetime.now(UTC),
            ),
            Vote(
                id=uuid4(),
                sample_a_id="test-a",
                sample_b_id="test-b",
                winner="B",
                timestamp=datetime.now(UTC),
            ),
            Vote(
                id=uuid4(),
                sample_a_id="test-a",
                sample_b_id="test-b",
                winner="tie",
                timestamp=datetime.now(UTC),
            ),
        ]
        entry = CacheEntry(data=data)

        assert entry.data == data
        assert isinstance(entry.cached_at, datetime)
        assert entry.cached_at.tzinfo is not None

    def test_is_expired_with_none_ttl(self) -> None:
        """CacheEntry never expires when TTL is None."""

        entry = CacheEntry(data=[])

        assert not entry.is_expired(None)

    def test_is_expired_with_zero_ttl(self) -> None:
        """CacheEntry expires immediately when TTL is 0."""
        entry = CacheEntry(data=[])

        assert entry.is_expired(0)

    def test_is_expired_with_negative_ttl(self) -> None:
        """CacheEntry expires immediately when TTL is negative."""
        entry = CacheEntry(data=[])

        assert entry.is_expired(-1)

    def test_is_expired_when_fresh(self) -> None:
        """CacheEntry is not expired when within TTL."""
        entry = CacheEntry(data=[])

        assert not entry.is_expired(60)

    def test_is_expired_when_old(self) -> None:
        """CacheEntry is expired when past TTL."""
        entry = CacheEntry(data=[])
        entry.cached_at = datetime.now(UTC) - timedelta(seconds=120)

        assert entry.is_expired(60)

    def test_is_expired_at_exact_boundary(self) -> None:
        """CacheEntry is not expired at exact TTL boundary."""
        entry = CacheEntry(data=[])
        now = datetime.now(UTC)
        entry.cached_at = now - timedelta(seconds=59)

        is_expired = entry.is_expired(60)
        assert not is_expired, (
            f"Expected not expired but got is_expired={is_expired}, "
            f"cached_at={entry.cached_at}, now={now}"
        )


class TestDataRepositoryInit:
    """Tests for DataRepository initialization."""

    def test_default_initialization(self, tmp_path: Path) -> None:
        """DataRepository initializes with default values."""
        repo = DataRepository(data_dir=tmp_path)

        assert repo._data_dir == tmp_path
        assert repo._cache_ttl is None
        assert repo._samples_cache is None
        assert repo._votes_cache is None
        assert repo._evaluations_cache is None

    def test_custom_data_dir(self, tmp_path: Path) -> None:
        """DataRepository accepts custom data directory."""
        custom_dir = tmp_path / "custom"
        repo = DataRepository(data_dir=custom_dir)

        assert repo._data_dir == custom_dir

    def test_cache_ttl_enabled(self, tmp_path: Path) -> None:
        """DataRepository accepts cache TTL."""
        repo = DataRepository(data_dir=tmp_path, cache_ttl=60)

        assert repo._cache_ttl == 60

    def test_cache_ttl_zero_disables_cache(self, tmp_path: Path) -> None:
        """Cache TTL of 0 disables caching."""
        repo = DataRepository(data_dir=tmp_path, cache_ttl=0)

        assert not repo.is_cache_enabled()

    def test_is_cache_enabled(self, tmp_path: Path) -> None:
        """is_cache_enabled returns correct state."""
        repo_with_cache = DataRepository(data_dir=tmp_path, cache_ttl=60)
        repo_without_cache = DataRepository(data_dir=tmp_path, cache_ttl=0)
        repo_none_cache = DataRepository(data_dir=tmp_path, cache_ttl=None)

        assert repo_with_cache.is_cache_enabled() is True
        assert repo_without_cache.is_cache_enabled() is False
        assert repo_none_cache.is_cache_enabled() is False


class TestDataRepositoryPaths:
    """Tests for DataRepository path properties."""

    def test_database_path(self, tmp_path: Path) -> None:
        """database_path returns correct path."""
        repo = DataRepository(data_dir=tmp_path)

        assert repo.database_path == tmp_path / "database.jsonl"

    def test_votes_path(self, tmp_path: Path) -> None:
        """votes_path returns correct path."""
        repo = DataRepository(data_dir=tmp_path)

        assert repo.votes_path == tmp_path / "votes.jsonl"

    def test_evaluations_path(self, tmp_path: Path) -> None:
        """evaluations_path returns correct path."""
        repo = DataRepository(data_dir=tmp_path)

        assert repo.evaluations_path == tmp_path / "vlm_evaluations.jsonl"


class TestDataRepositoryGetAllSamples:
    """Tests for get_all_samples method."""

    def test_returns_all_samples(self, tmp_path: Path) -> None:
        """get_all_samples returns all samples from database."""
        sample1 = ArtSample(
            model_id="model1",
            prompt_text="prompt1",
            category="cat1",
            attempt_number=1,
            raw_output="raw1",
            sanitized_output="clean1",
            is_valid=True,
        )
        sample2 = ArtSample(
            model_id="model2",
            prompt_text="prompt2",
            category="cat2",
            attempt_number=1,
            raw_output="raw2",
            sanitized_output="clean2",
            is_valid=False,
        )

        database_path = tmp_path / "database.jsonl"
        with database_path.open("a") as f:
            f.write(sample1.model_dump_json() + "\n")
            f.write(sample2.model_dump_json() + "\n")

        repo = DataRepository(data_dir=tmp_path)
        samples = repo.get_all_samples()

        assert len(samples) == 2
        assert samples[0].model_id == "model1"
        assert samples[1].model_id == "model2"

    def test_returns_empty_list_when_no_samples(self, tmp_path: Path) -> None:
        """get_all_samples returns empty list for empty database."""
        database_path = tmp_path / "database.jsonl"
        database_path.touch()

        repo = DataRepository(data_dir=tmp_path)
        samples = repo.get_all_samples()

        assert samples == []

    def test_raises_file_not_found_when_database_missing(self, tmp_path: Path) -> None:
        """get_all_samples raises FileNotFoundError when database.jsonl doesn't exist."""
        repo = DataRepository(data_dir=tmp_path)

        with pytest.raises(FileNotFoundError) as exc_info:
            repo.get_all_samples()

        error_msg = str(exc_info.value)
        assert "database.jsonl" in error_msg
        assert "not found" in error_msg.lower()

    def test_raises_file_not_found_when_data_dir_missing(self, tmp_path: Path) -> None:
        """get_all_samples raises FileNotFoundError when data directory doesn't exist."""
        missing_dir = tmp_path / "nonexistent"
        repo = DataRepository(data_dir=missing_dir)

        with pytest.raises(FileNotFoundError) as exc_info:
            repo.get_all_samples()

        error_msg = str(exc_info.value)
        assert "Data directory not found" in error_msg
        assert str(missing_dir) in error_msg


class TestDataRepositoryGetValidSamples:
    """Tests for get_valid_samples method."""

    def test_filters_to_valid_samples_only(self, tmp_path: Path) -> None:
        """get_valid_samples returns only samples where is_valid=True."""
        valid_sample = ArtSample(
            model_id="model1",
            prompt_text="prompt1",
            category="cat1",
            attempt_number=1,
            raw_output="raw1",
            sanitized_output="clean1",
            is_valid=True,
        )
        invalid_sample = ArtSample(
            model_id="model2",
            prompt_text="prompt2",
            category="cat2",
            attempt_number=1,
            raw_output="raw2",
            sanitized_output="clean2",
            is_valid=False,
        )

        database_path = tmp_path / "database.jsonl"
        with database_path.open("a") as f:
            f.write(valid_sample.model_dump_json() + "\n")
            f.write(invalid_sample.model_dump_json() + "\n")

        repo = DataRepository(data_dir=tmp_path)
        samples = repo.get_valid_samples()

        assert len(samples) == 1
        assert samples[0].model_id == "model1"
        assert samples[0].is_valid is True

    def test_returns_all_when_all_valid(self, tmp_path: Path) -> None:
        """get_valid_samples returns all samples when all are valid."""
        sample1 = ArtSample(
            model_id="model1",
            prompt_text="prompt1",
            category="cat1",
            attempt_number=1,
            raw_output="raw1",
            sanitized_output="clean1",
            is_valid=True,
        )
        sample2 = ArtSample(
            model_id="model2",
            prompt_text="prompt2",
            category="cat2",
            attempt_number=1,
            raw_output="raw2",
            sanitized_output="clean2",
            is_valid=True,
        )

        database_path = tmp_path / "database.jsonl"
        with database_path.open("a") as f:
            f.write(sample1.model_dump_json() + "\n")
            f.write(sample2.model_dump_json() + "\n")

        repo = DataRepository(data_dir=tmp_path)
        samples = repo.get_valid_samples()

        assert len(samples) == 2

    def test_returns_empty_when_all_invalid(self, tmp_path: Path) -> None:
        """get_valid_samples returns empty list when all samples are invalid."""
        invalid_sample = ArtSample(
            model_id="model1",
            prompt_text="prompt1",
            category="cat1",
            attempt_number=1,
            raw_output="raw1",
            sanitized_output="clean1",
            is_valid=False,
        )

        database_path = tmp_path / "database.jsonl"
        with database_path.open("a") as f:
            f.write(invalid_sample.model_dump_json() + "\n")

        repo = DataRepository(data_dir=tmp_path)
        samples = repo.get_valid_samples()

        assert samples == []


class TestDataRepositoryGetVotes:
    """Tests for get_votes method."""

    def test_returns_all_votes(self, tmp_path: Path) -> None:
        """get_votes returns all votes from votes file."""
        vote1 = Vote(sample_a_id="a1", sample_b_id="b1", winner="A")
        vote2 = Vote(sample_a_id="a2", sample_b_id="b2", winner="B")

        votes_path = tmp_path / "votes.jsonl"
        with votes_path.open("a") as f:
            f.write(vote1.model_dump_json() + "\n")
            f.write(vote2.model_dump_json() + "\n")

        repo = DataRepository(data_dir=tmp_path)
        votes = repo.get_votes()

        assert len(votes) == 2
        assert votes[0].sample_a_id == "a1"
        assert votes[1].sample_a_id == "a2"

    def test_returns_empty_list_when_no_votes(self, tmp_path: Path) -> None:
        """get_votes returns empty list for empty votes file."""
        votes_path = tmp_path / "votes.jsonl"
        votes_path.touch()

        repo = DataRepository(data_dir=tmp_path)
        votes = repo.get_votes()

        assert votes == []

    def test_raises_file_not_found_when_votes_missing(self, tmp_path: Path) -> None:
        """get_votes raises FileNotFoundError when votes.jsonl doesn't exist."""
        repo = DataRepository(data_dir=tmp_path)

        with pytest.raises(FileNotFoundError) as exc_info:
            repo.get_votes()

        error_msg = str(exc_info.value)
        assert "votes.jsonl" in error_msg
        assert "not found" in error_msg.lower()


class TestDataRepositoryGetEvaluations:
    """Tests for get_evaluations method."""

    def test_returns_all_evaluations(self, tmp_path: Path) -> None:
        """get_evaluations returns all evaluations from file."""
        eval1 = VLMEvaluation(
            sample_id="sample1",
            vlm_model_id="vlm1",
            expected_subject="cat",
            vlm_response="It's a cat",
            similarity_score=0.9,
            is_correct=True,
        )
        eval2 = VLMEvaluation(
            sample_id="sample2",
            vlm_model_id="vlm2",
            expected_subject="dog",
            vlm_response="It's a dog",
            similarity_score=0.8,
            is_correct=True,
        )

        eval_path = tmp_path / "vlm_evaluations.jsonl"
        with eval_path.open("a") as f:
            f.write(eval1.model_dump_json() + "\n")
            f.write(eval2.model_dump_json() + "\n")

        repo = DataRepository(data_dir=tmp_path)
        evaluations = repo.get_evaluations()

        assert len(evaluations) == 2
        assert evaluations[0].sample_id == "sample1"
        assert evaluations[1].sample_id == "sample2"

    def test_returns_empty_list_when_no_evaluations(self, tmp_path: Path) -> None:
        """get_evaluations returns empty list for empty evaluations file."""
        eval_path = tmp_path / "vlm_evaluations.jsonl"
        eval_path.touch()

        repo = DataRepository(data_dir=tmp_path)
        evaluations = repo.get_evaluations()

        assert evaluations == []

    def test_raises_file_not_found_when_evaluations_missing(self, tmp_path: Path) -> None:
        """get_evaluations raises FileNotFoundError when vlm_evaluations.jsonl doesn't exist."""
        repo = DataRepository(data_dir=tmp_path)

        with pytest.raises(FileNotFoundError) as exc_info:
            repo.get_evaluations()

        error_msg = str(exc_info.value)
        assert "vlm_evaluations.jsonl" in error_msg
        assert "not found" in error_msg.lower()


class TestDataRepositoryCaching:
    """Tests for caching behavior."""

    def test_caching_disabled_with_zero_ttl(self, tmp_path: Path) -> None:
        """Repository doesn't cache when TTL is 0."""
        sample = ArtSample(
            model_id="model1",
            prompt_text="prompt",
            category="cat",
            attempt_number=1,
            raw_output="raw",
            sanitized_output="clean",
            is_valid=True,
        )

        database_path = tmp_path / "database.jsonl"
        with database_path.open("a") as f:
            f.write(sample.model_dump_json() + "\n")

        repo = DataRepository(data_dir=tmp_path, cache_ttl=0)
        _ = repo.get_all_samples()

        # Add another sample to the file
        sample2 = ArtSample(
            model_id="model2",
            prompt_text="prompt2",
            category="cat2",
            attempt_number=1,
            raw_output="raw2",
            sanitized_output="clean2",
            is_valid=True,
        )
        with database_path.open("a") as f:
            f.write(sample2.model_dump_json() + "\n")

        samples2 = repo.get_all_samples()

        # With caching disabled, we should see the new sample
        assert len(samples2) == 2

    def test_caching_enabled_returns_same_object(self, tmp_path: Path) -> None:
        """Repository returns cached data when within TTL."""
        sample = ArtSample(
            model_id="model1",
            prompt_text="prompt",
            category="cat",
            attempt_number=1,
            raw_output="raw",
            sanitized_output="clean",
            is_valid=True,
        )

        database_path = tmp_path / "database.jsonl"
        with database_path.open("a") as f:
            f.write(sample.model_dump_json() + "\n")

        repo = DataRepository(data_dir=tmp_path, cache_ttl=60)
        samples1 = repo.get_all_samples()
        samples2 = repo.get_all_samples()

        # Should return the same cached list (same object identity)
        assert samples1 is samples2

    def test_cache_expires_after_ttl(self, tmp_path: Path) -> None:
        """Repository reloads data after cache expires."""
        sample = ArtSample(
            model_id="model1",
            prompt_text="prompt",
            category="cat",
            attempt_number=1,
            raw_output="raw",
            sanitized_output="clean",
            is_valid=True,
        )

        database_path = tmp_path / "database.jsonl"
        with database_path.open("a") as f:
            f.write(sample.model_dump_json() + "\n")

        repo = DataRepository(data_dir=tmp_path, cache_ttl=1)
        _ = repo.get_all_samples()

        # Add another sample to the file
        sample2 = ArtSample(
            model_id="model2",
            prompt_text="prompt2",
            category="cat2",
            attempt_number=1,
            raw_output="raw2",
            sanitized_output="clean2",
            is_valid=True,
        )
        with database_path.open("a") as f:
            f.write(sample2.model_dump_json() + "\n")

        # Wait for cache to expire
        sleep(1.5)

        samples2 = repo.get_all_samples()

        # Should reload and see the new sample
        assert len(samples2) == 2

    def test_cache_never_expires_with_none_ttl(self, tmp_path: Path) -> None:
        """Repository caches indefinitely when TTL is None."""
        sample = ArtSample(
            model_id="model1",
            prompt_text="prompt",
            category="cat",
            attempt_number=1,
            raw_output="raw",
            sanitized_output="clean",
            is_valid=True,
        )

        database_path = tmp_path / "database.jsonl"
        with database_path.open("a") as f:
            f.write(sample.model_dump_json() + "\n")

        repo = DataRepository(data_dir=tmp_path, cache_ttl=None)
        samples1 = repo.get_all_samples()

        # Add another sample to the file
        sample2 = ArtSample(
            model_id="model2",
            prompt_text="prompt2",
            category="cat2",
            attempt_number=1,
            raw_output="raw2",
            sanitized_output="clean2",
            is_valid=True,
        )
        with database_path.open("a") as f:
            f.write(sample2.model_dump_json() + "\n")

        # Even after waiting, cache should persist
        sleep(0.5)
        samples2 = repo.get_all_samples()

        # Should still return cached data (1 sample)
        assert samples1 is samples2
        assert len(samples2) == 1

    def test_clear_cache_clears_all_caches(self, tmp_path: Path) -> None:
        """clear_cache clears all cached data."""
        sample = ArtSample(
            model_id="model1",
            prompt_text="prompt",
            category="cat",
            attempt_number=1,
            raw_output="raw",
            sanitized_output="clean",
            is_valid=True,
        )
        vote = Vote(sample_a_id="a1", sample_b_id="b1", winner="A")

        database_path = tmp_path / "database.jsonl"
        with database_path.open("a") as f:
            f.write(sample.model_dump_json() + "\n")

        votes_path = tmp_path / "votes.jsonl"
        with votes_path.open("a") as f:
            f.write(vote.model_dump_json() + "\n")

        repo = DataRepository(data_dir=tmp_path, cache_ttl=60)
        _ = repo.get_all_samples()
        _ = repo.get_votes()

        # Verify caches are populated
        assert repo._samples_cache is not None
        assert repo._votes_cache is not None

        # Clear cache
        repo.clear_cache()

        # Verify caches are cleared
        assert repo._samples_cache is None
        assert repo._votes_cache is None
        assert repo._evaluations_cache is None

    def test_clear_cache_forces_reload(self, tmp_path: Path) -> None:
        """clear_cache forces next call to reload from file."""
        sample = ArtSample(
            model_id="model1",
            prompt_text="prompt",
            category="cat",
            attempt_number=1,
            raw_output="raw",
            sanitized_output="clean",
            is_valid=True,
        )

        database_path = tmp_path / "database.jsonl"
        with database_path.open("a") as f:
            f.write(sample.model_dump_json() + "\n")

        repo = DataRepository(data_dir=tmp_path, cache_ttl=60)
        samples1 = repo.get_all_samples()

        # Add another sample to the file
        sample2 = ArtSample(
            model_id="model2",
            prompt_text="prompt2",
            category="cat2",
            attempt_number=1,
            raw_output="raw2",
            sanitized_output="clean2",
            is_valid=True,
        )
        with database_path.open("a") as f:
            f.write(sample2.model_dump_json() + "\n")

        # Clear cache
        repo.clear_cache()

        samples2 = repo.get_all_samples()

        # Should reload and see the new sample
        assert len(samples2) == 2
        assert samples1 is not samples2

    def test_each_data_type_has_independent_cache(self, tmp_path: Path) -> None:
        """Samples, votes, and evaluations have separate caches."""
        sample = ArtSample(
            model_id="model1",
            prompt_text="prompt",
            category="cat",
            attempt_number=1,
            raw_output="raw",
            sanitized_output="clean",
            is_valid=True,
        )
        vote = Vote(sample_a_id="a1", sample_b_id="b1", winner="A")
        evaluation = VLMEvaluation(
            sample_id="sample1",
            vlm_model_id="vlm1",
            expected_subject="cat",
            vlm_response="cat",
            similarity_score=1.0,
            is_correct=True,
        )

        database_path = tmp_path / "database.jsonl"
        with database_path.open("a") as f:
            f.write(sample.model_dump_json() + "\n")

        votes_path = tmp_path / "votes.jsonl"
        with votes_path.open("a") as f:
            f.write(vote.model_dump_json() + "\n")

        eval_path = tmp_path / "vlm_evaluations.jsonl"
        with eval_path.open("a") as f:
            f.write(evaluation.model_dump_json() + "\n")

        repo = DataRepository(data_dir=tmp_path, cache_ttl=60)
        samples = repo.get_all_samples()
        votes = repo.get_votes()
        evaluations = repo.get_evaluations()

        # All caches should be populated independently
        assert repo._samples_cache is not None
        assert repo._votes_cache is not None
        assert repo._evaluations_cache is not None
        assert repo._samples_cache.data is samples
        assert repo._votes_cache.data is votes
        assert repo._evaluations_cache.data is evaluations

    def test_cache_timestamp_is_utc(self, tmp_path: Path) -> None:
        """Cache timestamp is in UTC timezone."""
        sample = ArtSample(
            model_id="model1",
            prompt_text="prompt",
            category="cat",
            attempt_number=1,
            raw_output="raw",
            sanitized_output="clean",
            is_valid=True,
        )

        database_path = tmp_path / "database.jsonl"
        with database_path.open("a") as f:
            f.write(sample.model_dump_json() + "\n")

        repo = DataRepository(data_dir=tmp_path, cache_ttl=60)
        repo.get_all_samples()

        # Cache timestamp should be in UTC
        assert repo._samples_cache is not None
        assert repo._samples_cache.cached_at.tzinfo is not None


class TestDataRepositoryStringPath:
    """Tests for DataRepository with string paths."""

    def test_accepts_string_data_dir(self, tmp_path: Path) -> None:
        """DataRepository accepts string path for data_dir."""
        sample = ArtSample(
            model_id="model1",
            prompt_text="prompt",
            category="cat",
            attempt_number=1,
            raw_output="raw",
            sanitized_output="clean",
            is_valid=True,
        )

        database_path = tmp_path / "database.jsonl"
        with database_path.open("a") as f:
            f.write(sample.model_dump_json() + "\n")

        # Use string path instead of Path
        repo = DataRepository(data_dir=str(tmp_path))
        samples = repo.get_all_samples()

        assert len(samples) == 1
