"""Domino animation module for ambient falling/rising domino effect."""

from dataclasses import dataclass
from typing import Literal

# Domino characters
STANDING = "▌"  # U+258C
TIPPING_RIGHT = "▞"  # U+259E
TIPPING_LEFT = "▚"  # U+259A
FALLEN = "▄"  # U+2584


@dataclass
class DominoState:
    """State for the domino animation.

    Attributes:
        direction: Current animation direction.
        phase: Current animation phase (0-2 for falling/rising).
        position: Current position for falling animations.
        width: Number of dominoes in the animation.
    """

    direction: Literal["left_to_right", "right_to_left", "rising"]
    phase: int
    position: int
    width: int


def _render_left_to_right(state: DominoState) -> str:
    """Render dominoes for left-to-right falling animation.

    Args:
        state: Current domino state.

    Returns:
        String of domino characters showing current animation state.
    """
    result = []
    for i in range(state.width):
        if i < state.position:
            result.append(FALLEN)
        elif i == state.position:
            if state.phase == 0:
                result.append(STANDING)
            elif state.phase == 1:
                result.append(TIPPING_RIGHT)
            else:
                result.append(FALLEN)
        else:
            result.append(STANDING)
    return "".join(result)


def _render_right_to_left(state: DominoState) -> str:
    """Render dominoes for right-to-left falling animation.

    Args:
        state: Current domino state.

    Returns:
        String of domino characters showing current animation state.
    """
    result = []
    for i in range(state.width):
        if i > state.position:
            result.append(FALLEN)
        elif i == state.position:
            if state.phase == 0:
                result.append(STANDING)
            elif state.phase == 1:
                result.append(TIPPING_LEFT)
            else:
                result.append(FALLEN)
        else:
            result.append(STANDING)
    return "".join(result)


def _render_rising(state: DominoState) -> str:
    """Render dominoes for rising animation.

    Args:
        state: Current domino state.

    Returns:
        String of domino characters showing current animation state.
    """
    if state.phase == 0:
        return FALLEN * state.width
    elif state.phase == 1:
        return TIPPING_RIGHT * state.width
    else:
        return STANDING * state.width


def get_domino_frame(state: DominoState) -> tuple[str, DominoState]:
    """Get the next frame of the domino animation.

    Args:
        state: Current domino state.

    Returns:
        Tuple of (rendered_string, next_state).

    Example:
        >>> state = DominoState("left_to_right", 0, 0, 5)
        >>> frame, next_state = get_domino_frame(state)
        >>> frame
        '▌▌▌▌▌'
    """
    if state.width <= 0:
        return ("", state)

    if state.direction == "left_to_right":
        return _get_left_to_right_next(state)
    elif state.direction == "right_to_left":
        return _get_right_to_left_next(state)
    else:
        return _get_rising_next(state)


def _get_left_to_right_next(state: DominoState) -> tuple[str, DominoState]:
    """Get next frame for left-to-right falling animation.

    Args:
        state: Current domino state.

    Returns:
        Tuple of (rendered_string, next_state).
    """
    rendered = _render_left_to_right(state)

    if state.phase < 2:
        next_state = DominoState(state.direction, state.phase + 1, state.position, state.width)
    elif state.position < state.width - 1:
        next_state = DominoState(state.direction, 0, state.position + 1, state.width)
    else:
        next_state = DominoState("rising", 0, state.width - 1, state.width)

    return (rendered, next_state)


def _get_right_to_left_next(state: DominoState) -> tuple[str, DominoState]:
    """Get next frame for right-to-left falling animation.

    Args:
        state: Current domino state.

    Returns:
        Tuple of (rendered_string, next_state).
    """
    rendered = _render_right_to_left(state)

    if state.phase < 2:
        next_state = DominoState(state.direction, state.phase + 1, state.position, state.width)
    elif state.position > 0:
        next_state = DominoState(state.direction, 0, state.position - 1, state.width)
    else:
        next_state = DominoState("rising", 0, 0, state.width)

    return (rendered, next_state)


def _get_rising_next(state: DominoState) -> tuple[str, DominoState]:
    """Get next frame for rising animation.

    Args:
        state: Current domino state.

    Returns:
        Tuple of (rendered_string, next_state).
    """
    rendered = _render_rising(state)

    if state.phase < 2:
        next_state = DominoState(state.direction, state.phase + 1, state.position, state.width)
    else:
        if state.position == state.width - 1:
            next_state = DominoState("right_to_left", 0, state.width - 1, state.width)
        else:
            next_state = DominoState("left_to_right", 0, 0, state.width)

    return (rendered, next_state)
