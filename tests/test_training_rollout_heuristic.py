from __future__ import annotations

import math

import pytest

from game.board import BoardState
from learned_ai.models.training_rollout_heuristic import (
    training_rollout_evaluate,
)


def test_training_rollout_empty_board_is_neutral() -> None:
    assert training_rollout_evaluate(BoardState.new_game(), "W") == 0.0


def test_training_rollout_preserves_historical_empty_square_semantics() -> None:
    board = BoardState.from_setup({"a1": "B"}, turn="W", phase="place")

    assert training_rollout_evaluate(board, "W") == pytest.approx(
        math.tanh(50.0 / 1500.0)
    )


def test_training_rollout_resolves_rules_terminal_before_features() -> None:
    board = BoardState.from_setup(
        {"a1": "W", "a4": "W", "b2": "B", "b4": "B", "b6": "B"},
        turn="W",
        phase="move",
    )

    assert training_rollout_evaluate(board, "W") == -1.0
    assert training_rollout_evaluate(board, "B") == 1.0
