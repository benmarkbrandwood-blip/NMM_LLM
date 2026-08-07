"""Trainer integration tests for rules draws versus rollout truncation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from game.board import POSITIONS, BoardState
from game.draw_rules import StandardDrawState
from scripts import train_s_gen_v2 as trainer


class _OneMoveModel:
    def policy_logits(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(features.shape[0], dtype=torch.float32)


class _UnexpectedOpponent:
    def choose_move(self, board: BoardState) -> dict:
        raise AssertionError("the rollout should stop before the opponent moves")


def _moving_board() -> BoardState:
    positions = {point: "" for point in POSITIONS}
    for point in ("d5", "a7", "b6", "g1"):
        positions[point] = "W"
    for point in ("e4", "g7", "b2", "a1"):
        positions[point] = "B"
    return BoardState.from_setup(positions, "W", "move")


def _encoded(move: dict) -> SimpleNamespace:
    return SimpleNamespace(
        feat_matrix=np.zeros(
            (1, trainer.MOVE_FEAT_DIM_WITH_LOOKAHEAD), dtype=np.float32
        ),
        value_input=np.zeros(trainer.VALUE_INPUT_DIM, dtype=np.float32),
        legal_moves=[move],
        sentinel_scores=[],
        h_scores_abs=[0.0],
        h_before=0.0,
        vn_scores_abs=[0.0],
        vn_before=0.0,
    )


def test_rollout_records_rules_draw_and_closes_last_learner_step(monkeypatch) -> None:
    board = _moving_board()
    move = {"from": "d5", "to": "e5", "capture": None}
    after = board.apply_move(move)
    target_key = trainer.StandardDrawTracker(after).snapshot().repetition_counts[0][0]
    draw_state = StandardDrawState(((target_key, 2),), 0)

    monkeypatch.setattr(
        trainer,
        "encode_position_with_lookahead",
        lambda current, *args, **kwargs: _encoded(move),
    )

    result = trainer._rollout(
        model=_OneMoveModel(),
        device=torch.device("cpu"),
        start_board=board,
        learner_color="W",
        opponent=_UnexpectedOpponent(),
        opp_color="B",
        sentinel=None,
        value_net=None,
        temperature=1.0,
        max_ply=4,
        record_branches=False,
        branch_every=0,
        retry_ply=0,
        draw_state=draw_state,
    )

    assert result.outcome == trainer.DRAW_SHORT
    assert result.termination_reason == "repetition"
    assert result.ply == 1
    assert len(result.trajectory) == 1
    assert result.trajectory[-1].done is True


def test_rollout_distinguishes_max_ply_truncation(monkeypatch) -> None:
    board = BoardState.new_game()
    move = {"from": None, "to": "a7", "capture": None}
    monkeypatch.setattr(
        trainer,
        "encode_position_with_lookahead",
        lambda current, *args, **kwargs: _encoded(move),
    )

    result = trainer._rollout(
        model=_OneMoveModel(),
        device=torch.device("cpu"),
        start_board=board,
        learner_color="W",
        opponent=_UnexpectedOpponent(),
        opp_color="B",
        sentinel=None,
        value_net=None,
        temperature=1.0,
        max_ply=1,
        record_branches=False,
        branch_every=0,
        retry_ply=0,
    )

    assert result.outcome == trainer.DRAW_LONG
    assert result.termination_reason == "max-ply-truncation"
    assert result.ply == 1


def test_terminal_move_at_max_ply_is_not_reported_as_truncation(monkeypatch) -> None:
    positions = {point: "" for point in POSITIONS}
    for point in ("d5", "a7", "b6", "g1"):
        positions[point] = "W"
    for point in ("g7", "b2", "a1"):
        positions[point] = "B"
    board = BoardState.from_setup(positions, "W", "move")
    move = {"from": "d5", "to": "e5", "capture": "g7"}
    monkeypatch.setattr(
        trainer,
        "encode_position_with_lookahead",
        lambda current, *args, **kwargs: _encoded(move),
    )

    result = trainer._rollout(
        model=_OneMoveModel(),
        device=torch.device("cpu"),
        start_board=board,
        learner_color="W",
        opponent=_UnexpectedOpponent(),
        opp_color="B",
        sentinel=None,
        value_net=None,
        temperature=1.0,
        max_ply=1,
        record_branches=False,
        branch_every=0,
        retry_ply=0,
    )

    assert result.outcome == trainer.WIN_REWARD
    assert result.termination_reason == "fewer-than-three"
    assert result.ply == 1
    assert result.trajectory[-1].done is True
