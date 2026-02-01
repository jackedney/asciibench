"""Tests for the domino animation module."""

from asciibench.common.domino import (
    FALLEN,
    STANDING,
    TIPPING_LEFT,
    TIPPING_RIGHT,
    DominoState,
    get_domino_frame,
)


class TestDominoState:
    """Tests for DominoState dataclass."""

    def test_initial_state(self):
        """DominoState can be created with initial values."""
        state = DominoState("left_to_right", 0, 0, 5)
        assert state.direction == "left_to_right"
        assert state.phase == 0
        assert state.position == 0
        assert state.width == 5

    def test_right_to_left_direction(self):
        """DominoState can use right_to_left direction."""
        state = DominoState("right_to_left", 0, 4, 5)
        assert state.direction == "right_to_left"

    def test_rising_direction(self):
        """DominoState can use rising direction."""
        state = DominoState("rising", 0, 0, 5)
        assert state.direction == "rising"

    def test_phase_values(self):
        """Phase can be 0, 1, or 2."""
        state = DominoState("left_to_right", 1, 2, 5)
        assert state.phase == 1

        state = DominoState("left_to_right", 2, 2, 5)
        assert state.phase == 2


class TestGetDominoFrame:
    """Tests for get_domino_frame function."""

    def test_returns_tuple(self):
        """get_domino_frame returns a tuple of (string, state)."""
        state = DominoState("left_to_right", 0, 0, 5)
        result = get_domino_frame(state)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], DominoState)

    def test_initial_state_shows_all_standing(self):
        """Initial state shows all dominoes standing."""
        state = DominoState("left_to_right", 0, 0, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == STANDING * 5

    def test_width_zero_returns_empty_string(self):
        """Width of 0 returns empty string."""
        state = DominoState("left_to_right", 0, 0, 0)
        rendered, _ = get_domino_frame(state)
        assert rendered == ""

    def test_width_zero_preserves_state(self):
        """Width of 0 preserves the state."""
        state = DominoState("left_to_right", 0, 0, 0)
        _, next_state = get_domino_frame(state)
        assert next_state == state

    def test_negative_width_returns_empty_string(self):
        """Negative width returns empty string."""
        state = DominoState("left_to_right", 0, 0, -5)
        rendered, _ = get_domino_frame(state)
        assert rendered == ""


class TestLeftToRightFall:
    """Tests for left-to-right falling animation."""

    def test_phase_0_position_0_shows_all_standing(self):
        """Phase 0 at position 0 shows all standing dominoes."""
        state = DominoState("left_to_right", 0, 0, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == STANDING * 5

    def test_phase_1_position_0_shows_first_tipping(self):
        """Phase 1 at position 0 shows first domino tipping right."""
        state = DominoState("left_to_right", 1, 0, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == TIPPING_RIGHT + STANDING * 4

    def test_phase_2_position_0_shows_first_fallen(self):
        """Phase 2 at position 0 shows first domino fallen."""
        state = DominoState("left_to_right", 2, 0, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == FALLEN + STANDING * 4

    def test_phase_0_position_1_shows_first_fallen(self):
        """Phase 0 at position 1 shows first domino fallen, second standing."""
        state = DominoState("left_to_right", 0, 1, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == FALLEN + STANDING * 4

    def test_phase_1_position_1_shows_second_tipping(self):
        """Phase 1 at position 1 shows second domino tipping right."""
        state = DominoState("left_to_right", 1, 1, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == FALLEN + TIPPING_RIGHT + STANDING * 3

    def test_phase_2_position_1_shows_second_fallen(self):
        """Phase 2 at position 1 shows second domino fallen."""
        state = DominoState("left_to_right", 2, 1, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == FALLEN * 2 + STANDING * 3

    def test_multiple_fallen_dominoes(self):
        """Multiple fallen dominoes display correctly."""
        state = DominoState("left_to_right", 0, 2, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == FALLEN * 2 + STANDING * 3

    def test_all_fallen(self):
        """All dominoes fallen shows all fallen characters."""
        state = DominoState("left_to_right", 2, 4, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == FALLEN * 5

    def test_example_from_prd(self):
        """Example from PRD: width=5, after several frames shows '▄▞▌▌▌'."""
        state = DominoState("left_to_right", 1, 1, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == FALLEN + TIPPING_RIGHT + STANDING * 3

    def test_single_domino_falls_correctly(self):
        """Single domino width falls correctly."""
        state = DominoState("left_to_right", 0, 0, 1)
        rendered, _ = get_domino_frame(state)
        assert rendered == STANDING

        state = DominoState("left_to_right", 1, 0, 1)
        rendered, _ = get_domino_frame(state)
        assert rendered == TIPPING_RIGHT

        state = DominoState("left_to_right", 2, 0, 1)
        rendered, _ = get_domino_frame(state)
        assert rendered == FALLEN


class TestRightToLeftFall:
    """Tests for right-to-left falling animation."""

    def test_phase_0_position_4_shows_all_standing(self):
        """Phase 0 at position 4 shows all standing dominoes."""
        state = DominoState("right_to_left", 0, 4, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == STANDING * 5

    def test_phase_1_position_4_shows_last_tipping(self):
        """Phase 1 at position 4 shows last domino tipping left."""
        state = DominoState("right_to_left", 1, 4, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == STANDING * 4 + TIPPING_LEFT

    def test_phase_2_position_4_shows_last_fallen(self):
        """Phase 2 at position 4 shows last domino fallen."""
        state = DominoState("right_to_left", 2, 4, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == STANDING * 4 + FALLEN

    def test_phase_0_position_3_shows_last_fallen(self):
        """Phase 0 at position 3 shows last domino fallen."""
        state = DominoState("right_to_left", 0, 3, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == STANDING * 4 + FALLEN

    def test_phase_1_position_3_shows_second_last_tipping(self):
        """Phase 1 at position 3 shows second last domino tipping left."""
        state = DominoState("right_to_left", 1, 3, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == STANDING * 3 + TIPPING_LEFT + FALLEN

    def test_phase_2_position_3_shows_second_last_fallen(self):
        """Phase 2 at position 3 shows second last domino fallen."""
        state = DominoState("right_to_left", 2, 3, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == STANDING * 3 + FALLEN * 2

    def test_multiple_fallen_dominoes(self):
        """Multiple fallen dominoes display correctly."""
        state = DominoState("right_to_left", 0, 2, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == STANDING * 3 + FALLEN * 2

    def test_all_fallen(self):
        """All dominoes fallen shows all fallen characters."""
        state = DominoState("right_to_left", 2, 0, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == FALLEN * 5


class TestRisingAnimation:
    """Tests for rising animation."""

    def test_phase_0_shows_all_fallen(self):
        """Rising phase 0 shows all fallen dominoes."""
        state = DominoState("rising", 0, 0, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == FALLEN * 5

    def test_phase_1_shows_all_tipping_right(self):
        """Rising phase 1 shows all tipping right dominoes."""
        state = DominoState("rising", 1, 0, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == TIPPING_RIGHT * 5

    def test_phase_2_shows_all_standing(self):
        """Rising phase 2 shows all standing dominoes."""
        state = DominoState("rising", 2, 0, 5)
        rendered, _ = get_domino_frame(state)
        assert rendered == STANDING * 5

    def test_single_domino_rises_correctly(self):
        """Single domino width rises correctly."""
        state = DominoState("rising", 0, 0, 1)
        rendered, _ = get_domino_frame(state)
        assert rendered == FALLEN

        state = DominoState("rising", 1, 0, 1)
        rendered, _ = get_domino_frame(state)
        assert rendered == TIPPING_RIGHT

        state = DominoState("rising", 2, 0, 1)
        rendered, _ = get_domino_frame(state)
        assert rendered == STANDING


class TestAnimationCycle:
    """Tests for complete animation cycle."""

    def test_left_to_right_to_rising_transition(self):
        """Left-to-right fall transitions to rising after all dominoes fall."""
        state = DominoState("left_to_right", 2, 4, 5)
        _, next_state = get_domino_frame(state)
        assert next_state.direction == "rising"
        assert next_state.phase == 0

    def test_rising_to_left_to_right_transition(self):
        """Rising transitions to left-to-right after completing."""
        state = DominoState("rising", 2, 0, 5)
        _, next_state = get_domino_frame(state)
        assert next_state.direction == "left_to_right"
        assert next_state.phase == 0

    def test_right_to_left_to_rising_transition(self):
        """Right-to-left fall transitions to rising after all dominoes fall."""
        state = DominoState("right_to_left", 2, 0, 5)
        _, next_state = get_domino_frame(state)
        assert next_state.direction == "rising"

    def test_full_cycle_sequence(self):
        """Full animation cycle follows: L->R -> Rise -> R->L -> Rise -> L->R."""
        state = DominoState("left_to_right", 0, 0, 3)

        left_to_right_states = []
        while state.direction == "left_to_right":
            left_to_right_states.append(state)
            _, state = get_domino_frame(state)

        rising_states = []
        while state.direction == "rising":
            rising_states.append(state)
            _, state = get_domino_frame(state)

        right_to_left_states = []
        while state.direction == "right_to_left":
            right_to_left_states.append(state)
            _, state = get_domino_frame(state)

        rising_states_2 = []
        while state.direction == "rising":
            rising_states_2.append(state)
            _, state = get_domino_frame(state)

        assert len(left_to_right_states) > 0
        assert len(rising_states) > 0
        assert len(right_to_left_states) > 0
        assert len(rising_states_2) > 0
        assert state.direction == "left_to_right"

    def test_cycle_repeats_indefinitely(self):
        """Animation cycle repeats indefinitely."""
        state = DominoState("left_to_right", 0, 0, 3)

        directions = []
        for _ in range(100):
            directions.append(state.direction)
            _, state = get_domino_frame(state)

        assert directions[0] == "left_to_right"
        assert directions.count("left_to_right") > 1
        assert directions.count("rising") > 1
        assert directions.count("right_to_left") > 1


class TestStateTransitions:
    """Tests for state transitions."""

    def test_phase_increments(self):
        """Phase increments from 0 to 1 to 2."""
        state = DominoState("left_to_right", 0, 0, 5)

        _, state = get_domino_frame(state)
        assert state.phase == 1

        _, state = get_domino_frame(state)
        assert state.phase == 2

    def test_position_increments_left_to_right(self):
        """Position increments after phase completes in left-to-right."""
        state = DominoState("left_to_right", 2, 0, 5)

        _, state = get_domino_frame(state)
        assert state.position == 1
        assert state.phase == 0

        _, state = get_domino_frame(state)
        _, state = get_domino_frame(state)
        _, state = get_domino_frame(state)
        assert state.position == 2
        assert state.phase == 0

    def test_position_decrements_right_to_left(self):
        """Position decrements after phase completes in right-to-left."""
        state = DominoState("right_to_left", 2, 4, 5)

        _, state = get_domino_frame(state)
        assert state.position == 3
        assert state.phase == 0

        _, state = get_domino_frame(state)
        _, state = get_domino_frame(state)
        _, state = get_domino_frame(state)
        assert state.position == 2
        assert state.phase == 0


class TestDominoCharacters:
    """Tests for domino character constants."""

    def test_standing_character(self):
        """STANDING is the correct character."""
        assert STANDING == "▌"

    def test_tipping_right_character(self):
        """TIPPING_RIGHT is the correct character."""
        assert TIPPING_RIGHT == "▞"

    def test_tipping_left_character(self):
        """TIPPING_LEFT is the correct character."""
        assert TIPPING_LEFT == "▚"

    def test_fallen_character(self):
        """FALLEN is the correct character."""
        assert FALLEN == "▄"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_width_one_cycles_correctly(self):
        """Width of 1 cycles through all phases correctly."""
        state = DominoState("left_to_right", 0, 0, 1)

        rendered, state = get_domino_frame(state)
        assert rendered == STANDING

        rendered, state = get_domino_frame(state)
        assert rendered == TIPPING_RIGHT

        rendered, state = get_domino_frame(state)
        assert rendered == FALLEN

        rendered, state = get_domino_frame(state)
        assert rendered == FALLEN
        assert state.direction == "rising"

    def test_large_width_handled(self):
        """Large width is handled without error."""
        state = DominoState("left_to_right", 0, 0, 100)
        rendered, _ = get_domino_frame(state)
        assert len(rendered) == 100
        assert rendered == STANDING * 100

    def test_small_width(self):
        """Small width (2) works correctly."""
        state = DominoState("left_to_right", 0, 0, 2)
        rendered, _ = get_domino_frame(state)
        assert rendered == STANDING * 2
        assert len(rendered) == 2
