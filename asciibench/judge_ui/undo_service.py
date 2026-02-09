"""Undo service for managing undo state in the Judge UI.

This service encapsulates the undo state to prevent calling undo
twice in a row, which is a user interface concern.

Dependencies:
    - asciibench.common.persistence: JSONL persistence utilities
"""

from pathlib import Path

from asciibench.common.models import Vote
from asciibench.common.persistence import read_jsonl, write_jsonl


class UndoService:
    """Service for managing vote undo operations.

    Tracks whether the last action was an undo to prevent
    calling undo twice in a row without a vote in between.
    """

    def __init__(self, votes_path: Path | None = None):
        """Initialize UndoService with path to votes file.

        Args:
            votes_path: Path to votes.jsonl file (default: data/votes.jsonl)
        """
        self._votes_path = votes_path or Path("data/votes.jsonl")
        self._last_action_was_undo: bool = False

    def undo_vote(self) -> Vote | None:
        """Undo the most recent vote.

        Removes the last vote from votes.jsonl and returns it for confirmation.
        The operation is atomic - either the entire undo succeeds or fails.

        Undo can only be called once after each vote. Calling undo twice in a
        row without submitting a new vote will return None.

        Returns:
            The removed vote for confirmation, or None if no vote to undo
        """
        if self._last_action_was_undo:
            return None

        try:
            votes = read_jsonl(self._votes_path, Vote)
        except FileNotFoundError:
            return None

        if not votes:
            return None

        last_vote = votes[-1]

        votes_without_last = votes[:-1]
        write_jsonl(self._votes_path, votes_without_last)

        self._last_action_was_undo = True

        return last_vote

    def record_vote_submitted(self) -> None:
        """Record that a new vote was submitted.

        Resets the undo state so undo can be called again.
        """
        self._last_action_was_undo = False

    @property
    def last_action_was_undo(self) -> bool:
        """Check if the last action was an undo."""
        return self._last_action_was_undo
