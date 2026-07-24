"""learned_ai/training/termination.py — Termination classification for training rollouts.

Distinguishes rules-based endings (material, blocking, max-ply truncation) from
infrastructure failures (learner encoder / policy or opponent engine produced no
action or raised). Infrastructure failures must never enter W/D/L or advancement
statistics; they are logged as their own class for observability.

Reserved-but-not-yet-detected: repetition and 50-move rule. They live in the
enum so they can be wired in later without changing any downstream consumer.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from game.board import BoardState
from game.rules import get_game_phase, is_blocked


class TerminationReason(str, Enum):
    WIN_FEWER_THAN_THREE   = "win_lt3"          # opponent placed 9 and dropped below 3
    WIN_NO_LEGAL_MOVE      = "win_blocked"      # opponent has no legal move
    LOSS_FEWER_THAN_THREE  = "loss_lt3"
    LOSS_NO_LEGAL_MOVE     = "loss_blocked"
    DRAW_MAX_PLY_TRUNCATED = "draw_trunc"       # rollout hit max_ply without a terminal state
    DRAW_REPETITION        = "draw_rep"         # reserved
    DRAW_50_MOVE           = "draw_50"          # reserved
    INFRA_LEARNER_FAILURE  = "infra_learner"    # learner encoder / policy raised or returned nothing
    INFRA_OPPONENT_FAILURE = "infra_opponent"   # opponent raised or returned no move on non-terminal


VALID_OUTCOMES = frozenset({
    TerminationReason.WIN_FEWER_THAN_THREE,
    TerminationReason.WIN_NO_LEGAL_MOVE,
    TerminationReason.LOSS_FEWER_THAN_THREE,
    TerminationReason.LOSS_NO_LEGAL_MOVE,
    TerminationReason.DRAW_MAX_PLY_TRUNCATED,
    TerminationReason.DRAW_REPETITION,
    TerminationReason.DRAW_50_MOVE,
})

INFRA_REASONS = frozenset({
    TerminationReason.INFRA_LEARNER_FAILURE,
    TerminationReason.INFRA_OPPONENT_FAILURE,
})

# Ordered list used for stacked-area plotting so colour semantics stay consistent.
PLOT_ORDER = [
    TerminationReason.WIN_FEWER_THAN_THREE,
    TerminationReason.WIN_NO_LEGAL_MOVE,
    TerminationReason.LOSS_FEWER_THAN_THREE,
    TerminationReason.LOSS_NO_LEGAL_MOVE,
    TerminationReason.DRAW_REPETITION,
    TerminationReason.DRAW_50_MOVE,
    TerminationReason.DRAW_MAX_PLY_TRUNCATED,
]


def classify_terminal(board: BoardState, learner_color: str) -> TerminationReason:
    """Classify a board on which `is_terminal(board)` returned True.

    Material first (either color placed all 9 and has <3), then blocking.
    """
    for color in ("W", "B"):
        if board.pieces_placed[color] == 9 and board.pieces_on_board[color] < 3:
            if color == learner_color:
                return TerminationReason.LOSS_FEWER_THAN_THREE
            return TerminationReason.WIN_FEWER_THAN_THREE
    current = board.turn
    if get_game_phase(board, current) == "move" and is_blocked(board, current):
        if current == learner_color:
            return TerminationReason.LOSS_NO_LEGAL_MOVE
        return TerminationReason.WIN_NO_LEGAL_MOVE
    # Fallback — is_terminal said True but neither branch matched. Shouldn't happen
    # in practice; fall through to the safest classification for the recorded winner.
    return TerminationReason.LOSS_NO_LEGAL_MOVE


def rolling_percentages(recent: list[TerminationReason]) -> dict[str, float]:
    """Return per-reason percentage over a window, ignoring infra failures.

    infra_learner and infra_opponent are returned as absolute counts (not %), so a
    burst of infrastructure failure is visible independent of the outcome mix.
    """
    valid = [r for r in recent if r in VALID_OUTCOMES]
    total = len(valid)
    out: dict[str, float] = {}
    for reason in PLOT_ORDER:
        pct = (sum(1 for r in valid if r == reason) / total * 100.0) if total else 0.0
        out[reason.value] = round(pct, 2)
    out[TerminationReason.INFRA_LEARNER_FAILURE.value] = float(
        sum(1 for r in recent if r == TerminationReason.INFRA_LEARNER_FAILURE)
    )
    out[TerminationReason.INFRA_OPPONENT_FAILURE.value] = float(
        sum(1 for r in recent if r == TerminationReason.INFRA_OPPONENT_FAILURE)
    )
    return out
