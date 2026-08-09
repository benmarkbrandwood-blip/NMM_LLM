"""Focused tests for the held-out mill-bonus no-update probe."""

from __future__ import annotations

import pytest

from game.board import BoardState
from learned_ai.validation.mill_bonus_no_update_probe import (
    MillBonusProbeError,
    probe_reward_transition,
)
from scripts import train_s_gen_v2 as trainer


def _placement_mill_board() -> BoardState:
    return BoardState.from_setup(
        {"a7": "W", "d7": "W", "a4": "B", "b4": "B"},
        turn="W",
        phase="place",
    )


def test_probe_changes_only_bonus_for_a_frozen_downgrade_turn() -> None:
    board = _placement_mill_board()
    move = {"from": None, "to": "g7", "capture": "a4"}
    after = board.apply_move(move)

    result = probe_reward_transition(
        board=board,
        move=move,
        malom_quality=-1.0,
        expected_after_fen=after.to_fen_string(),
    )

    assert result["before_fen"] == board.to_fen_string()
    assert result["after_fen"] == after.to_fen_string()
    assert result["move"] == move
    assert result["mills_formed"] == 1
    assert result["rewards"] == {
        "disabled": 0.0,
        "legacy-unconditional": trainer.MILL_BONUS,
        "malom-preserving-only": 0.0,
    }


def test_probe_retains_bonus_for_a_value_preserving_mill() -> None:
    board = _placement_mill_board()
    move = {"from": None, "to": "g7", "capture": "a4"}

    result = probe_reward_transition(
        board=board,
        move=move,
        malom_quality=0.0,
        expected_after_fen=board.apply_move(move).to_fen_string(),
    )

    assert result["rewards"]["legacy-unconditional"] == trainer.MILL_BONUS
    assert result["rewards"]["malom-preserving-only"] == trainer.MILL_BONUS
    assert result["rewards"]["disabled"] == 0.0


def test_probe_reports_no_difference_for_a_non_mill_turn() -> None:
    board = BoardState.new_game()
    move = {"from": None, "to": "a7", "capture": None}

    result = probe_reward_transition(
        board=board,
        move=move,
        malom_quality=0.0,
        expected_after_fen=board.apply_move(move).to_fen_string(),
    )

    assert result["mills_formed"] == 0
    assert set(result["rewards"].values()) == {0.0}


def test_probe_rejects_a_changed_successor() -> None:
    board = BoardState.new_game()
    move = {"from": None, "to": "a7", "capture": None}

    with pytest.raises(MillBonusProbeError, match="successor FEN differs"):
        probe_reward_transition(
            board=board,
            move=move,
            malom_quality=0.0,
            expected_after_fen=board.to_fen_string(),
        )


def test_probe_rejects_an_incomplete_mill_turn() -> None:
    board = _placement_mill_board()

    with pytest.raises(MillBonusProbeError, match="complete legal turn"):
        probe_reward_transition(
            board=board,
            move={"from": None, "to": "g7", "capture": None},
            malom_quality=-1.0,
            expected_after_fen=board.to_fen_string(),
        )
