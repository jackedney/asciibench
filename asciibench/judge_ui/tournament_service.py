"""TournamentService for orchestrating Swiss-tournament rounds.

This service manages the complete tournament lifecycle including round creation,
matchup serving, vote recording, round transitions, and background sample
generation.

Dependencies:
    - GenerationService: For on-demand sample generation
    - ConfigService: For tournament configuration
    - DataRepository: For data access
    - SwissPairSelector: For Swiss pair selection
    - PromptSelector: For prompt selection
    - analyst.calculate_elo: For Elo rating calculation
"""

import asyncio
import logging
import random
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from asciibench.analyst.elo import calculate_elo
from asciibench.common.models import ArtSample, Matchup, RoundState
from asciibench.common.persistence import append_jsonl, read_jsonl
from asciibench.judge_ui.generation_service import GenerationService
from asciibench.judge_ui.swiss_selector import PromptSelector, SwissPairSelector

if TYPE_CHECKING:
    from asciibench.common.config_service import ConfigService
    from asciibench.common.repository import DataRepository
    from asciibench.evaluator.orchestrator import EvaluationOrchestrator

    VLMMetadata = tuple[EvaluationOrchestrator, "DataRepository", Path]
    VLMOrchestratorFactory = Callable[[], VLMMetadata | None]

logger = logging.getLogger(__name__)


class TournamentService:
    """Service for orchestrating Swiss-tournament rounds.

    Manages tournament state including current and next rounds, background
    generation, and round transitions. Uses Swiss pairing (N closest Elo + N random)
    for model selection and ensures samples are generated on demand.
    """

    def __init__(
        self,
        generation_service: GenerationService,
        config_service: "ConfigService",
        repo: "DataRepository",
        n: int,
        vlm_orchestrator_factory: "VLMOrchestratorFactory | None" = None,
    ) -> None:
        """Initialize the TournamentService.

        Args:
            generation_service: Service for generating samples on demand
            config_service: Service for configuration access
            repo: Data repository for accessing samples and votes
            n: Round size from tournament config (number of closest pairs)
            vlm_orchestrator_factory: Optional factory callable that creates VLM
                evaluation orchestrator. Returns (orchestrator, repo, path) tuple
                or None if VLM evaluation is not available.
        """
        self.generation_service = generation_service
        self.config_service = config_service
        self.repo = repo
        self.n = n
        self._vlm_orchestrator_factory = vlm_orchestrator_factory

        self._current_round: RoundState | None = None
        self._next_round: RoundState | None = None
        self._background_task: asyncio.Task | None = None
        self._initial_generation_task: asyncio.Task | None = None
        self._vlm_task: asyncio.Task | None = None
        self._vlm_pending_samples: list[ArtSample] | None = None
        self._generation_total: int = 0
        self._generation_completed: int = 0
        self._lock = asyncio.Lock()

        self._rounds_path: Path = Path("data/rounds.jsonl")
        self._pair_selector = SwissPairSelector()
        self._prompt_selector = PromptSelector()

    async def initialize(self) -> None:
        """Initialize the tournament service.

        Loads existing round state from rounds.jsonl or creates round 1 with
        all random pairs. Generates missing samples (blocking) and starts
        background generation for round 2.
        """
        async with self._lock:
            await self._initialize_internal()

    async def _initialize_internal(self) -> None:
        """Internal initialization logic (called within lock)."""
        existing_round = self._load_latest_round()

        if existing_round is not None:
            self._current_round = await self._reconstruct_round_state(existing_round)
            logger.info(f"Loaded existing round {self._current_round.round_number}")
        else:
            self._current_round = await self._create_round(1)
            logger.info(f"Created round 1 with {len(self._current_round.matchups)} matchups")

        if self._current_round is not None and not self._current_round.generation_complete:
            self._start_initial_generation()

        if self._next_round is None and self._current_round is not None:
            self._start_background_generation()

    def _start_initial_generation(self) -> None:
        """Start background initial generation for the current round."""
        if self._initial_generation_task is not None and not self._initial_generation_task.done():
            logger.debug("Initial generation already in progress")
            return

        async def generate_initial_round() -> None:
            try:
                if self._current_round is None:
                    return

                async with self._lock:
                    self._generation_total = len(self._current_round.matchups)
                    self._generation_completed = 0

                logger.info(
                    "Starting initial generation for round %d (%d matchups)",
                    self._current_round.round_number,
                    len(self._current_round.matchups),
                )

                samples = self.repo.get_all_samples_or_empty()

                updated_round = await self.generation_service.ensure_samples_for_round(
                    self._current_round, samples
                )

                async with self._lock:
                    if self._current_round is None:
                        return

                    self._current_round = updated_round
                    self._generation_completed = self._generation_total
                    self._persist_round_state(self._current_round)

                logger.info(f"Initial generation complete for round {updated_round.round_number}")

                round_samples = self._get_samples_for_round(updated_round)
                if round_samples:
                    self._start_vlm_evaluation(round_samples)
            except Exception as e:
                logger.error(f"Error in initial generation: {e}")

        self._initial_generation_task = asyncio.create_task(generate_initial_round())

    def _load_latest_round(self) -> RoundState | None:
        """Load the latest round from rounds.jsonl.

        Returns:
            The latest RoundState or None if file doesn't exist or is empty
        """
        if not self._rounds_path.exists():
            return None

        try:
            rounds = read_jsonl(self._rounds_path, RoundState)
            if rounds:
                return rounds[-1]
        except Exception as e:
            logger.error(f"Error loading rounds from {self._rounds_path}: {e}")

        return None

    async def _reconstruct_round_state(self, round_state: RoundState) -> RoundState:
        """Reconstruct round state by cross-referencing with actual votes.

        Args:
            round_state: RoundState loaded from rounds.jsonl

        Returns:
            Reconstructed RoundState with updated is_jged and vote_id fields
        """
        try:
            votes = self.repo.get_votes()
            vote_id_to_winner: dict[str, str] = {str(v.id): v.winner for v in votes}
        except FileNotFoundError:
            votes = []
            vote_id_to_winner = {}

        updated_matchups = []
        for matchup in round_state.matchups:
            if matchup.vote_id:
                winner = vote_id_to_winner.get(matchup.vote_id)
                if winner:
                    updated_matchup = matchup.model_copy(
                        update={"is_judged": True, "vote_id": matchup.vote_id}
                    )
                else:
                    updated_matchup = matchup.model_copy(
                        update={"is_judged": False, "vote_id": None}
                    )
            else:
                updated_matchup = matchup.model_copy(update={"is_judged": False, "vote_id": None})
            updated_matchups.append(updated_matchup)

        return round_state.model_copy(update={"matchups": updated_matchups})

    async def _create_round(self, round_number: int) -> RoundState:
        """Create a new tournament round.

        Args:
            round_number: The round number to create

        Returns:
            New RoundState with matchups created using Swiss pairing
        """
        models = self.config_service.get_models()
        model_ids = [model.id for model in models]
        prompts = self.config_service.get_prompts()

        if round_number == 1:
            elo_ratings: dict[str, float] = {}
        else:
            elo_ratings = await self._calculate_current_elo()

        pairs = self._pair_selector.select_pairs(model_ids, elo_ratings, self.n)

        matchups = []
        for model_a_id, model_b_id in pairs:
            used_prompts = self._get_used_prompts_for_pair(model_a_id, model_b_id)
            prompt = self._prompt_selector.select_prompt(
                model_a_id, model_b_id, prompts, used_prompts
            )

            if prompt:
                matchup = Matchup(
                    model_a_id=model_a_id,
                    model_b_id=model_b_id,
                    prompt_text=prompt.text,
                    prompt_category=prompt.category,
                )
                matchups.append(matchup)

        return RoundState(
            id=uuid4(),
            round_number=round_number,
            matchups=matchups,
            elo_snapshot=elo_ratings,
        )

    async def _calculate_current_elo(self) -> dict[str, float]:
        """Calculate current Elo ratings from all votes.

        Returns:
            Dictionary mapping model IDs to Elo ratings
        """
        try:
            votes = self.repo.get_votes()
            samples = self.repo.get_all_samples()
            return calculate_elo(votes, samples)
        except FileNotFoundError:
            return {}

    def _get_used_prompts_for_pair(self, model_a: str, model_b: str) -> set[str]:
        """Get set of prompt texts already used for a model pair.

        Args:
            model_a: First model ID
            model_b: Second model ID

        Returns:
            Set of prompt texts that have been used for this pair
        """
        if self._current_round is None:
            return set()

        used_prompts: set[str] = set()
        for matchup in self._current_round.matchups:
            if (matchup.model_a_id == model_a and matchup.model_b_id == model_b) or (
                matchup.model_a_id == model_b and matchup.model_b_id == model_a
            ):
                used_prompts.add(matchup.prompt_text)

        return used_prompts

    def _persist_round_state(self, round_state: RoundState) -> None:
        """Persist round state to rounds.jsonl.

        Args:
            round_state: RoundState to persist
        """
        append_jsonl(self._rounds_path, round_state)

    def _start_background_generation(self) -> None:
        """Start background generation for the next round."""
        if self._background_task is not None and not self._background_task.done():
            logger.debug("Background generation already in progress")
            return

        async def generate_next_round() -> None:
            try:
                if self._current_round is None:
                    return

                current_round = self._current_round
                next_round_number = current_round.round_number + 1
                logger.info("Starting background generation for round %d", next_round_number)
                self._next_round = await self._create_round(next_round_number)

                samples = self.repo.get_all_samples_or_empty()
                self._next_round = await self.generation_service.ensure_samples_for_round(
                    self._next_round, samples
                )

                logger.info(f"Background generation complete for round {next_round_number}")

                if self._next_round is not None:
                    round_samples = self._get_samples_for_round(self._next_round)
                    if round_samples:
                        self._start_vlm_evaluation(round_samples)
            except Exception as e:
                logger.error(f"Error in background generation: {e}")

        self._background_task = asyncio.create_task(generate_next_round())

    def _get_samples_for_round(self, round_state: RoundState) -> list[ArtSample]:
        """Get ArtSample objects for all matchups in a round.

        Args:
            round_state: RoundState to get samples for

        Returns:
            List of ArtSample objects referenced by the round's matchups
        """
        samples = self.repo.get_all_samples_or_empty()
        sample_by_id = {str(s.id): s for s in samples}

        round_samples: list[ArtSample] = []
        for matchup in round_state.matchups:
            if matchup.sample_a_id and matchup.sample_a_id in sample_by_id:
                round_samples.append(sample_by_id[matchup.sample_a_id])
            if matchup.sample_b_id and matchup.sample_b_id in sample_by_id:
                round_samples.append(sample_by_id[matchup.sample_b_id])

        return round_samples

    def _start_vlm_evaluation(self, samples: list[ArtSample]) -> None:
        """Start background VLM evaluation for the given samples.

        Runs EvaluationOrchestrator.run() in a background asyncio task.
        VLM evaluation is fire-and-forget: failures are logged but do not
        block tournament progress.

        Args:
            samples: List of ArtSample objects to evaluate with VLM
        """
        if self._vlm_orchestrator_factory is None:
            logger.debug("VLM orchestrator factory not configured, skipping VLM evaluation")
            return

        if not samples:
            logger.debug("No samples to evaluate with VLM")
            return

        if self._vlm_task is not None and not self._vlm_task.done():
            logger.warning("VLM evaluation already in progress, queueing for later")
            self._vlm_pending_samples = samples
            return

        async def run_vlm_evaluation() -> None:
            try:
                factory = self._vlm_orchestrator_factory
                if factory is None:
                    logger.warning("VLM orchestrator factory became None, skipping evaluation")
                    return

                result = factory()
                if result is None:
                    logger.warning("VLM orchestrator factory returned None, skipping evaluation")
                    return

                orchestrator, repo, _ = result
                valid_samples = [s for s in samples if s.is_valid]
                if not valid_samples:
                    logger.debug("No valid samples for VLM evaluation")
                    return

                existing_evaluations = repo.get_evaluations_or_empty()
                logger.info(
                    "Starting VLM evaluation for %d samples (%d valid)",
                    len(samples),
                    len(valid_samples),
                )

                await orchestrator.run(
                    samples=valid_samples,
                    existing_evaluations=existing_evaluations,
                    show_progress=False,
                )

                logger.info("VLM evaluation completed for %d samples", len(valid_samples))
            except Exception as e:
                logger.warning("VLM evaluation failed (tournament continues): %s", e)
            finally:
                pending = self._vlm_pending_samples
                self._vlm_pending_samples = None
                self._vlm_task = None
                if pending:
                    logger.info("Running queued VLM evaluation for %d samples", len(pending))
                    self._start_vlm_evaluation(pending)

        self._vlm_task = asyncio.create_task(run_vlm_evaluation())

    def get_next_matchup(self) -> Matchup | None:
        """Get a random unjudged matchup from the current round with both samples ready.

        Returns:
            A random unjudged Matchup with sample_a_id and sample_b_id set,
            or None if all judged or no current round or no ready matchups
        """
        if self._current_round is None:
            return None

        ready_matchups = [
            m
            for m in self._current_round.matchups
            if not m.is_judged and m.sample_a_id is not None and m.sample_b_id is not None
        ]

        if not ready_matchups:
            return None

        return random.choice(ready_matchups)

    async def record_vote(self, matchup_id: UUID, vote_id: str) -> None:
        """Record a vote for a matchup.

        Marks the matchup as judged, persists round state, and triggers
        round completion if all matchups are judged.

        Args:
            matchup_id: ID of the matchup being voted on
            vote_id: ID of the vote that was recorded
        """
        async with self._lock:
            await self._record_vote_internal(matchup_id, vote_id)

    async def _record_vote_internal(self, matchup_id: UUID, vote_id: str) -> None:
        """Internal vote recording logic (called within lock)."""
        if self._current_round is None:
            logger.warning("No current round to record vote")
            return

        updated_matchups = []
        all_judged = True

        for matchup in self._current_round.matchups:
            if matchup.id == matchup_id:
                updated_matchup = matchup.model_copy(update={"is_judged": True, "vote_id": vote_id})
            else:
                updated_matchup = matchup.model_copy()

            updated_matchups.append(updated_matchup)

            if not updated_matchup.is_judged:
                all_judged = False

        self._current_round = self._current_round.model_copy(
            update={"matchups": updated_matchups, "all_judged": all_judged}
        )

        self._persist_round_state(self._current_round)

        if all_judged:
            await self._complete_round()

    async def _complete_round(self) -> None:
        """Complete the current round.

        Recomputes Elo ratings from all votes, swaps to pre-generated next round,
        and starts background generation for the round after that.
        """
        if self._current_round is None:
            logger.warning("No current round to complete")
            return

        logger.info(f"Completing round {self._current_round.round_number}")

        await self._calculate_current_elo()

        if self._next_round is not None:
            self._current_round = self._next_round
            self._next_round = None
            logger.info(f"Swapped to round {self._current_round.round_number}")
        else:
            next_round_number = self._current_round.round_number + 1
            self._current_round = await self._create_round(next_round_number)
            samples = self.repo.get_all_samples_or_empty()
            self._current_round = await self.generation_service.ensure_samples_for_round(
                self._current_round, samples
            )
            logger.info(f"Created round {self._current_round.round_number}")

        self._persist_round_state(self._current_round)
        self._start_background_generation()

    async def undo_last_vote(self, matchup_id: UUID) -> None:
        """Undo the last vote for a matchup.

        Unmarks the matchup as judged and clears its vote_id.

        Args:
            matchup_id: ID of the matchup to undo
        """
        async with self._lock:
            await self._undo_last_vote_internal(matchup_id)

    async def _undo_last_vote_internal(self, matchup_id: UUID) -> None:
        """Internal vote undo logic (called within lock)."""
        if self._current_round is None:
            logger.warning("No current round to undo vote")
            return

        updated_matchups = []

        for matchup in self._current_round.matchups:
            if matchup.id == matchup_id:
                updated_matchup = matchup.model_copy(update={"is_judged": False, "vote_id": None})
            else:
                updated_matchup = matchup.model_copy()

            updated_matchups.append(updated_matchup)

        self._current_round = self._current_round.model_copy(
            update={"matchups": updated_matchups, "all_judged": False}
        )

        self._persist_round_state(self._current_round)

    async def shutdown(self) -> None:
        """Shut down the tournament service and cancel background tasks."""
        if self._initial_generation_task is not None and not self._initial_generation_task.done():
            self._initial_generation_task.cancel()
            try:
                await self._initial_generation_task
            except asyncio.CancelledError:
                pass
            logger.info("Initial generation task cancelled")

        if self._background_task is not None and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            logger.info("Background generation task cancelled")

        if self._vlm_task is not None and not self._vlm_task.done():
            self._vlm_task.cancel()
            try:
                await self._vlm_task
            except asyncio.CancelledError:
                pass
            logger.info("VLM evaluation task cancelled")

    def find_matchup_by_samples(self, sample_a_id: str, sample_b_id: str) -> Matchup | None:
        """Find a matchup in the current round by sample IDs.

        Checks both orderings since samples may have been swapped for position bias.

        Args:
            sample_a_id: First sample ID
            sample_b_id: Second sample ID

        Returns:
            The matching Matchup, or None if not found or no current round
        """
        if self._current_round is None:
            return None

        for matchup in self._current_round.matchups:
            if (matchup.sample_a_id == sample_a_id and matchup.sample_b_id == sample_b_id) or (
                matchup.sample_a_id == sample_b_id and matchup.sample_b_id == sample_a_id
            ):
                return matchup

        return None

    def get_round_progress(self) -> dict:
        """Get progress information for the current round.

        Returns:
            Dictionary with round_number, judged_count, total_count, next_round_ready
        """
        if self._current_round is None:
            return {
                "round_number": 0,
                "judged_count": 0,
                "total_count": 0,
                "next_round_ready": False,
            }

        judged_count = sum(1 for m in self._current_round.matchups if m.is_judged)
        total_count = len(self._current_round.matchups)
        next_round_ready = self._next_round is not None and self._next_round.generation_complete

        return {
            "round_number": self._current_round.round_number,
            "judged_count": judged_count,
            "total_count": total_count,
            "next_round_ready": next_round_ready,
        }

    def get_generation_status(self) -> dict:
        """Get generation progress status.

        Returns:
            Dictionary with generating: bool, completed: int, total: int
        """
        generating = (
            self._initial_generation_task is not None and not self._initial_generation_task.done()
        )
        return {
            "generating": generating,
            "completed": self._generation_completed,
            "total": self._generation_total,
        }
