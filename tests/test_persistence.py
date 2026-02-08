"""Tests for JSONL persistence utilities."""

from pathlib import Path
from uuid import uuid4

import pytest

from asciibench.common.models import ArtSample, Vote
from asciibench.common.persistence import (
    append_jsonl,
    read_jsonl,
    read_jsonl_by_id,
    write_jsonl,
)


class TestAppendJsonl:
    """Tests for append_jsonl function."""

    def test_creates_file_if_not_exists(self, tmp_path: Path) -> None:
        """append_jsonl creates the file if it doesn't exist."""
        file_path = tmp_path / "new_file.jsonl"
        vote = Vote(
            sample_a_id="sample-a",
            sample_b_id="sample-b",
            winner="A",
        )

        append_jsonl(file_path, vote)

        assert file_path.exists()
        content = file_path.read_text()
        assert "sample-a" in content
        assert "sample-b" in content

    def test_appends_to_existing_file(self, tmp_path: Path) -> None:
        """append_jsonl appends to an existing file."""
        file_path = tmp_path / "votes.jsonl"
        vote1 = Vote(sample_a_id="a1", sample_b_id="b1", winner="A")
        vote2 = Vote(sample_a_id="a2", sample_b_id="b2", winner="B")

        append_jsonl(file_path, vote1)
        append_jsonl(file_path, vote2)

        lines = file_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert "a1" in lines[0]
        assert "a2" in lines[1]

    def test_appends_art_sample(self, tmp_path: Path) -> None:
        """append_jsonl works with ArtSample models."""
        file_path = tmp_path / "samples.jsonl"
        sample = ArtSample(
            model_id="openai/gpt-4o",
            prompt_text="Draw a cat",
            category="single_object",
            attempt_number=1,
            raw_output="```\n/_\\\n```",
            sanitized_output="/_\\",
            is_valid=True,
        )

        append_jsonl(file_path, sample)

        content = file_path.read_text()
        assert "openai/gpt-4o" in content
        assert "Draw a cat" in content
        assert "single_object" in content

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """append_jsonl creates parent directories if needed."""
        file_path = tmp_path / "subdir" / "nested" / "file.jsonl"
        vote = Vote(sample_a_id="a", sample_b_id="b", winner="tie")

        append_jsonl(file_path, vote)

        assert file_path.exists()
        assert "tie" in file_path.read_text()

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """append_jsonl accepts string paths."""
        file_path = str(tmp_path / "test.jsonl")
        vote = Vote(sample_a_id="a", sample_b_id="b", winner="fail")

        append_jsonl(file_path, vote)

        assert Path(file_path).exists()


class TestReadJsonl:
    """Tests for read_jsonl function."""

    def test_raises_file_not_found_for_nonexistent_file(self, tmp_path: Path) -> None:
        """read_jsonl raises FileNotFoundError if file doesn't exist."""
        file_path = tmp_path / "nonexistent.jsonl"

        with pytest.raises(FileNotFoundError) as exc_info:
            read_jsonl(file_path, Vote)

        assert str(file_path) in str(exc_info.value)

    def test_reads_all_lines(self, tmp_path: Path) -> None:
        """read_jsonl reads all lines from the file."""
        file_path = tmp_path / "votes.jsonl"
        votes = [
            Vote(sample_a_id="a1", sample_b_id="b1", winner="A"),
            Vote(sample_a_id="a2", sample_b_id="b2", winner="B"),
            Vote(sample_a_id="a3", sample_b_id="b3", winner="tie"),
        ]
        for vote in votes:
            append_jsonl(file_path, vote)

        result = read_jsonl(file_path, Vote)

        assert len(result) == 3
        assert result[0].sample_a_id == "a1"
        assert result[1].sample_a_id == "a2"
        assert result[2].sample_a_id == "a3"

    def test_reads_art_samples(self, tmp_path: Path) -> None:
        """read_jsonl correctly parses ArtSample models."""
        file_path = tmp_path / "samples.jsonl"
        sample = ArtSample(
            model_id="anthropic/claude-3",
            prompt_text="Draw a tree",
            category="single_object",
            attempt_number=2,
            raw_output="```\n/|\\\n```",
            sanitized_output="/|\\",
            is_valid=True,
        )
        append_jsonl(file_path, sample)

        result = read_jsonl(file_path, ArtSample)

        assert len(result) == 1
        assert result[0].model_id == "anthropic/claude-3"
        assert result[0].prompt_text == "Draw a tree"
        assert result[0].attempt_number == 2
        assert result[0].is_valid is True

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        """read_jsonl skips empty lines in the file."""
        file_path = tmp_path / "votes.jsonl"
        vote = Vote(sample_a_id="a", sample_b_id="b", winner="A")
        append_jsonl(file_path, vote)

        # Manually add empty lines
        with file_path.open("a") as f:
            f.write("\n\n")

        append_jsonl(file_path, vote)

        result = read_jsonl(file_path, Vote)
        assert len(result) == 2

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """read_jsonl accepts string paths."""
        file_path = str(tmp_path / "test.jsonl")
        vote = Vote(sample_a_id="a", sample_b_id="b", winner="B")
        append_jsonl(file_path, vote)

        result = read_jsonl(file_path, Vote)

        assert len(result) == 1


class TestReadJsonlById:
    """Tests for read_jsonl_by_id function."""

    def test_returns_none_for_nonexistent_file(self, tmp_path: Path) -> None:
        """read_jsonl_by_id returns None if file doesn't exist."""
        file_path = tmp_path / "nonexistent.jsonl"
        some_id = uuid4()

        result = read_jsonl_by_id(file_path, some_id, Vote)

        assert result is None

    def test_finds_record_by_uuid(self, tmp_path: Path) -> None:
        """read_jsonl_by_id finds a record by its UUID."""
        file_path = tmp_path / "votes.jsonl"
        target_id = uuid4()
        votes = [
            Vote(sample_a_id="a1", sample_b_id="b1", winner="A"),
            Vote(id=target_id, sample_a_id="a2", sample_b_id="b2", winner="B"),
            Vote(sample_a_id="a3", sample_b_id="b3", winner="tie"),
        ]
        for vote in votes:
            append_jsonl(file_path, vote)

        result = read_jsonl_by_id(file_path, target_id, Vote)

        assert result is not None
        assert result.id == target_id
        assert result.sample_a_id == "a2"
        assert result.winner == "B"

    def test_finds_record_by_uuid_string(self, tmp_path: Path) -> None:
        """read_jsonl_by_id accepts UUID as string."""
        file_path = tmp_path / "votes.jsonl"
        target_id = uuid4()
        vote = Vote(id=target_id, sample_a_id="a", sample_b_id="b", winner="A")
        append_jsonl(file_path, vote)

        result = read_jsonl_by_id(file_path, str(target_id), Vote)

        assert result is not None
        assert result.id == target_id

    def test_returns_none_if_not_found(self, tmp_path: Path) -> None:
        """read_jsonl_by_id returns None if ID not found."""
        file_path = tmp_path / "votes.jsonl"
        vote = Vote(sample_a_id="a", sample_b_id="b", winner="A")
        append_jsonl(file_path, vote)

        result = read_jsonl_by_id(file_path, uuid4(), Vote)

        assert result is None

    def test_finds_art_sample_by_id(self, tmp_path: Path) -> None:
        """read_jsonl_by_id works with ArtSample models."""
        file_path = tmp_path / "samples.jsonl"
        target_id = uuid4()
        sample = ArtSample(
            id=target_id,
            model_id="test-model",
            prompt_text="Draw something",
            category="test",
            attempt_number=1,
            raw_output="raw",
            sanitized_output="clean",
            is_valid=True,
        )
        append_jsonl(file_path, sample)

        result = read_jsonl_by_id(file_path, target_id, ArtSample)

        assert result is not None
        assert result.id == target_id
        assert result.model_id == "test-model"

    def test_returns_first_match(self, tmp_path: Path) -> None:
        """read_jsonl_by_id returns the matching record (only one should exist)."""
        file_path = tmp_path / "votes.jsonl"
        target_id = uuid4()
        vote1 = Vote(id=target_id, sample_a_id="first", sample_b_id="b", winner="A")
        vote2 = Vote(sample_a_id="other", sample_b_id="b", winner="B")
        append_jsonl(file_path, vote1)
        append_jsonl(file_path, vote2)

        result = read_jsonl_by_id(file_path, target_id, Vote)

        assert result is not None
        assert result.sample_a_id == "first"

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """read_jsonl_by_id accepts string paths."""
        file_path = str(tmp_path / "test.jsonl")
        target_id = uuid4()
        vote = Vote(id=target_id, sample_a_id="a", sample_b_id="b", winner="tie")
        append_jsonl(file_path, vote)

        result = read_jsonl_by_id(file_path, target_id, Vote)

        assert result is not None
        assert result.id == target_id


class TestFileLocking:
    """Tests for file locking behavior."""

    def test_lock_file_created(self, tmp_path: Path) -> None:
        """append_jsonl creates a lock file during operation."""
        file_path = tmp_path / "test.jsonl"
        lock_path = tmp_path / "test.jsonl.lock"
        vote = Vote(sample_a_id="a", sample_b_id="b", winner="A")

        append_jsonl(file_path, vote)

        # Lock file should exist after operation (filelock keeps it)
        assert lock_path.exists()


class TestIntegration:
    """Integration tests for persistence utilities."""

    def test_roundtrip_vote(self, tmp_path: Path) -> None:
        """Vote can be written and read back correctly."""
        file_path = tmp_path / "votes.jsonl"
        original = Vote(sample_a_id="sample-a", sample_b_id="sample-b", winner="A")

        append_jsonl(file_path, original)
        result = read_jsonl(file_path, Vote)

        assert len(result) == 1
        assert result[0].id == original.id
        assert result[0].sample_a_id == original.sample_a_id
        assert result[0].sample_b_id == original.sample_b_id
        assert result[0].winner == original.winner
        assert result[0].timestamp == original.timestamp

    def test_roundtrip_art_sample(self, tmp_path: Path) -> None:
        """ArtSample can be written and read back correctly."""
        file_path = tmp_path / "samples.jsonl"
        original = ArtSample(
            model_id="test/model",
            prompt_text="Test prompt",
            category="test_category",
            attempt_number=3,
            raw_output="```\nraw\n```",
            sanitized_output="raw",
            is_valid=False,
        )

        append_jsonl(file_path, original)
        result = read_jsonl(file_path, ArtSample)

        assert len(result) == 1
        assert result[0].id == original.id
        assert result[0].model_id == original.model_id
        assert result[0].prompt_text == original.prompt_text
        assert result[0].category == original.category
        assert result[0].attempt_number == original.attempt_number
        assert result[0].raw_output == original.raw_output
        assert result[0].sanitized_output == original.sanitized_output
        assert result[0].is_valid == original.is_valid
        assert result[0].timestamp == original.timestamp

    def test_multiple_appends_and_reads(self, tmp_path: Path) -> None:
        """Multiple appends followed by read returns all records."""
        file_path = tmp_path / "votes.jsonl"
        ids = [uuid4() for _ in range(5)]

        for i, vote_id in enumerate(ids):
            vote = Vote(
                id=vote_id,
                sample_a_id=f"a{i}",
                sample_b_id=f"b{i}",
                winner="A" if i % 2 == 0 else "B",
            )
            append_jsonl(file_path, vote)

        result = read_jsonl(file_path, Vote)

        assert len(result) == 5
        for i, vote in enumerate(result):
            assert vote.id == ids[i]
            assert vote.sample_a_id == f"a{i}"

    def test_append_then_find_by_id(self, tmp_path: Path) -> None:
        """Can append and then find by ID."""
        file_path = tmp_path / "votes.jsonl"
        target_id = uuid4()

        # Add several votes
        for _ in range(3):
            vote = Vote(sample_a_id="a", sample_b_id="b", winner="A")
            append_jsonl(file_path, vote)

        # Add the target vote
        target_vote = Vote(id=target_id, sample_a_id="target", sample_b_id="b", winner="B")
        append_jsonl(file_path, target_vote)

        # Add more votes
        for _ in range(3):
            vote = Vote(sample_a_id="a", sample_b_id="b", winner="tie")
            append_jsonl(file_path, vote)

        result = read_jsonl_by_id(file_path, target_id, Vote)

        assert result is not None
        assert result.id == target_id
        assert result.sample_a_id == "target"


class TestWriteJsonl:
    """Tests for write_jsonl function (atomic rewrite)."""

    def test_writes_empty_list(self, tmp_path: Path) -> None:
        """write_jsonl can write an empty list."""
        file_path = tmp_path / "test.jsonl"
        # First add a vote
        vote = Vote(sample_a_id="a", sample_b_id="b", winner="A")
        append_jsonl(file_path, vote)
        assert len(read_jsonl(file_path, Vote)) == 1

        # Now write empty list
        write_jsonl(file_path, [])

        assert file_path.exists()
        result = read_jsonl(file_path, Vote)
        assert result == []

    def test_writes_single_item(self, tmp_path: Path) -> None:
        """write_jsonl writes a single item correctly."""
        file_path = tmp_path / "test.jsonl"
        vote = Vote(sample_a_id="a", sample_b_id="b", winner="A")

        write_jsonl(file_path, [vote])

        result = read_jsonl(file_path, Vote)
        assert len(result) == 1
        assert result[0].id == vote.id
        assert result[0].winner == "A"

    def test_writes_multiple_items(self, tmp_path: Path) -> None:
        """write_jsonl writes multiple items correctly."""
        file_path = tmp_path / "test.jsonl"
        votes = [
            Vote(sample_a_id="a1", sample_b_id="b1", winner="A"),
            Vote(sample_a_id="a2", sample_b_id="b2", winner="B"),
            Vote(sample_a_id="a3", sample_b_id="b3", winner="tie"),
        ]

        write_jsonl(file_path, votes)

        result = read_jsonl(file_path, Vote)
        assert len(result) == 3
        assert result[0].winner == "A"
        assert result[1].winner == "B"
        assert result[2].winner == "tie"

    def test_overwrites_existing_content(self, tmp_path: Path) -> None:
        """write_jsonl completely replaces existing file content."""
        file_path = tmp_path / "test.jsonl"

        # First write 5 votes
        initial_votes = [Vote(sample_a_id=f"a{i}", sample_b_id="b", winner="A") for i in range(5)]
        write_jsonl(file_path, initial_votes)
        assert len(read_jsonl(file_path, Vote)) == 5

        # Now write only 2 votes
        new_votes = [
            Vote(sample_a_id="new1", sample_b_id="b", winner="B"),
            Vote(sample_a_id="new2", sample_b_id="b", winner="tie"),
        ]
        write_jsonl(file_path, new_votes)

        result = read_jsonl(file_path, Vote)
        assert len(result) == 2
        assert result[0].sample_a_id == "new1"
        assert result[1].sample_a_id == "new2"

    def test_creates_file_if_not_exists(self, tmp_path: Path) -> None:
        """write_jsonl creates the file if it doesn't exist."""
        file_path = tmp_path / "new_file.jsonl"
        assert not file_path.exists()

        vote = Vote(sample_a_id="a", sample_b_id="b", winner="A")
        write_jsonl(file_path, [vote])

        assert file_path.exists()
        result = read_jsonl(file_path, Vote)
        assert len(result) == 1

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """write_jsonl creates parent directories if needed."""
        file_path = tmp_path / "subdir" / "nested" / "file.jsonl"

        vote = Vote(sample_a_id="a", sample_b_id="b", winner="A")
        write_jsonl(file_path, [vote])

        assert file_path.exists()

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """write_jsonl accepts string paths."""
        file_path = str(tmp_path / "test.jsonl")
        vote = Vote(sample_a_id="a", sample_b_id="b", winner="A")

        write_jsonl(file_path, [vote])

        result = read_jsonl(file_path, Vote)
        assert len(result) == 1

    def test_writes_art_samples(self, tmp_path: Path) -> None:
        """write_jsonl works with ArtSample models."""
        file_path = tmp_path / "samples.jsonl"
        samples = [
            ArtSample(
                model_id="model1",
                prompt_text="test1",
                category="cat1",
                attempt_number=1,
                raw_output="raw1",
                sanitized_output="clean1",
                is_valid=True,
            ),
            ArtSample(
                model_id="model2",
                prompt_text="test2",
                category="cat2",
                attempt_number=2,
                raw_output="raw2",
                sanitized_output="clean2",
                is_valid=False,
            ),
        ]

        write_jsonl(file_path, samples)

        result = read_jsonl(file_path, ArtSample)
        assert len(result) == 2
        assert result[0].model_id == "model1"
        assert result[1].model_id == "model2"

    def test_atomic_write_no_temp_file_left(self, tmp_path: Path) -> None:
        """write_jsonl doesn't leave temp files after successful write."""
        file_path = tmp_path / "test.jsonl"
        vote = Vote(sample_a_id="a", sample_b_id="b", winner="A")

        write_jsonl(file_path, [vote])

        # Check that no .tmp files are left
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_roundtrip_preserves_data(self, tmp_path: Path) -> None:
        """Data written with write_jsonl can be read back correctly."""
        file_path = tmp_path / "test.jsonl"
        original_id = uuid4()
        original_vote = Vote(
            id=original_id, sample_a_id="original-a", sample_b_id="original-b", winner="fail"
        )

        write_jsonl(file_path, [original_vote])
        result = read_jsonl(file_path, Vote)

        assert len(result) == 1
        assert result[0].id == original_id
        assert result[0].sample_a_id == "original-a"
        assert result[0].sample_b_id == "original-b"
        assert result[0].winner == "fail"
