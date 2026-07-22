"""Compatibility rollout evaluator used by Generalist v2 training.

This module deliberately preserves the historical empty-square checks from
``train_s_gen_v2.py``. ``BoardState`` stores an empty square as ``""``, while
the evaluator checks for ``None``. As a result, its mobility terms remain
zero and occupied opponent pieces are treated as blocked. Changing that
behaviour would define a corrected experiment route, not reproduce training.
"""

from __future__ import annotations

import math

from game.board import ADJACENCY, MILLS, BoardState
from game.rules import is_terminal


def training_rollout_evaluate(board: BoardState, color: str) -> float:
    """Return the exact historical Generalist v2 rollout score."""
    terminal, winner = is_terminal(board)
    if terminal:
        return 1.0 if winner == color else -1.0

    opponent = "B" if color == "W" else "W"
    our_mills = sum(
        1
        for mill in MILLS
        if all(board.positions.get(position) == color for position in mill)
    )
    opponent_mills = sum(
        1
        for mill in MILLS
        if all(board.positions.get(position) == opponent for position in mill)
    )
    our_mobility = sum(
        1
        for position, piece in board.positions.items()
        if piece == color
        for adjacent in ADJACENCY.get(position, [])
        if board.positions.get(adjacent) is None
    )
    opponent_mobility = sum(
        1
        for position, piece in board.positions.items()
        if piece == opponent
        for adjacent in ADJACENCY.get(position, [])
        if board.positions.get(adjacent) is None
    )
    blocked_opponents = sum(
        1
        for position, piece in board.positions.items()
        if piece == opponent
        and all(
            board.positions.get(adjacent) is not None
            for adjacent in ADJACENCY.get(position, [])
        )
    )
    raw = float(
        500 * (our_mills - opponent_mills)
        + 10 * (our_mobility - opponent_mobility)
        + 50 * blocked_opponents
    )
    return math.tanh(raw / 1500.0)
