"""History-dependent standard draw adjudication for Nine Men's Morris.

The board model deliberately stores only an instantaneous position.  Draw
rules need additional history, so this module keeps that state separately and
can be snapshotted when a rollout branches or retries from an earlier board.

One observed action is one complete logical ply: a primary place/move plus
any compulsory removal encoded in the same move dictionary.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TypeAlias

from .board import POSITIONS, BoardState


REPETITION_DRAW = "repetition"
NO_PROGRESS_DRAW = "50-move rule"
THREEFOLD_COUNT = 3
NO_PROGRESS_LIMIT = 100

PositionKey: TypeAlias = tuple[tuple[str, ...], str]


def _position_key(board: BoardState) -> PositionKey:
    """Return the stable-moving repetition identity used by this project."""
    return tuple(board.positions[pos] for pos in POSITIONS), board.turn


@dataclass(frozen=True)
class StandardDrawState:
    """Serializable-in-memory state for an exact branch or retry."""

    repetition_counts: tuple[tuple[PositionKey, int], ...]
    no_progress_plies: int


class StandardDrawTracker:
    """Apply automatic threefold and 100-ply no-capture draw rules.

    The semantics match the tracked MRS contract:

    * only stable movement/flying positions are repetition observations;
    * placement and removal reset the active repetition window;
    * the current stable position is observed after a reset;
    * only movement plies count toward no-progress;
    * placement and removal reset the no-progress counter; and
    * repetition has priority when both rules become true at one boundary.
    """

    def __init__(
        self,
        board: BoardState,
        *,
        state: StandardDrawState | None = None,
    ) -> None:
        if state is None:
            self._repetition_counts: Counter[PositionKey] = Counter()
            self.no_progress_plies = 0
            self._observe_if_stable_moving(board)
        else:
            if state.no_progress_plies < 0:
                raise ValueError("no_progress_plies must be non-negative")
            self._repetition_counts = Counter(dict(state.repetition_counts))
            if any(count <= 0 for count in self._repetition_counts.values()):
                raise ValueError("repetition counts must be positive")
            self.no_progress_plies = state.no_progress_plies

    def snapshot(self) -> StandardDrawState:
        """Return an immutable copy suitable for a branch or retry."""
        return StandardDrawState(
            repetition_counts=tuple(
                sorted(self._repetition_counts.items(), key=lambda item: item[0])
            ),
            no_progress_plies=self.no_progress_plies,
        )

    def observe(
        self,
        before: BoardState,
        move: dict,
        after: BoardState,
    ) -> str | None:
        """Record one completed logical ply and return a draw reason, if any."""
        if before.turn == after.turn:
            raise ValueError("a complete logical ply must change side to move")

        is_placement = move.get("from") is None
        is_removal = bool(move.get("capture"))

        if is_placement:
            self.no_progress_plies = 0
        else:
            self.no_progress_plies += 1
        if is_removal:
            self.no_progress_plies = 0

        if is_placement or is_removal:
            self._repetition_counts.clear()

        occurrence = self._observe_if_stable_moving(after)
        if occurrence >= THREEFOLD_COUNT:
            return REPETITION_DRAW
        if self.no_progress_plies >= NO_PROGRESS_LIMIT:
            return NO_PROGRESS_DRAW
        return None

    def _observe_if_stable_moving(self, board: BoardState) -> int:
        if board.phase != "move":
            return 0
        key = _position_key(board)
        self._repetition_counts[key] += 1
        return self._repetition_counts[key]
