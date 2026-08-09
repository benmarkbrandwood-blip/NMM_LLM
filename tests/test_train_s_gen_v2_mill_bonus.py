"""Focused contracts for Generalist v2 mill-bonus shaping."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from game.board import BoardState
from scripts import train_s_gen_v2 as trainer


@pytest.mark.parametrize(
    ("mode", "malom_quality", "expected"),
    [
        ("legacy-unconditional", -2.0, trainer.MILL_BONUS),
        ("malom-preserving-only", 0.0, trainer.MILL_BONUS),
        ("malom-preserving-only", -1.0, 0.0),
        ("malom-preserving-only", -2.0, 0.0),
        (
            "malom-preserving-plus-downgrade-penalty",
            0.0,
            trainer.MILL_BONUS,
        ),
        (
            "malom-preserving-plus-downgrade-penalty",
            -1.0,
            0.0,
        ),
        ("disabled", None, 0.0),
    ],
)
def test_mill_bonus_mode_contract(
    mode: str,
    malom_quality: float | None,
    expected: float,
) -> None:
    assert trainer._mill_formation_reward(
        mills_formed=1,
        malom_quality=malom_quality,
        mode=mode,
    ) == pytest.approx(expected)


def test_preserving_bonus_scales_for_double_mill() -> None:
    assert trainer._mill_formation_reward(
        mills_formed=2,
        malom_quality=0.0,
        mode="malom-preserving-only",
    ) == pytest.approx(2 * trainer.MILL_BONUS)


@pytest.mark.parametrize("malom_quality", [None, float("nan"), 1.0, -3.0])
def test_preserving_bonus_fails_closed_without_valid_exact_quality(
    malom_quality: float | None,
) -> None:
    with pytest.raises(RuntimeError, match="Malom move quality"):
        trainer._mill_formation_reward(
            mills_formed=1,
            malom_quality=malom_quality,
            mode="malom-preserving-only",
        )


def test_non_mill_needs_no_malom_query_for_bonus() -> None:
    assert trainer._mill_formation_reward(
        mills_formed=0,
        malom_quality=None,
        mode="malom-preserving-only",
    ) == 0.0


@pytest.mark.parametrize(
    ("quality", "expected"),
    [
        (0.0, 0.0),
        (-1.0, -trainer.MALOM_DOWNGRADE_PENALTY),
        (-2.0, -2 * trainer.MALOM_DOWNGRADE_PENALTY),
    ],
)
def test_downgrade_penalty_is_asymmetric_and_rank_weighted(
    quality: float,
    expected: float,
) -> None:
    assert trainer._malom_downgrade_reward(
        malom_quality=quality,
        mode="malom-preserving-plus-downgrade-penalty",
    ) == pytest.approx(expected)


@pytest.mark.parametrize("quality", [None, True, float("nan"), 1.0, -3.0])
def test_downgrade_penalty_fails_closed_without_exact_quality(
    quality: float | bool | None,
) -> None:
    with pytest.raises(RuntimeError, match="Malom move quality"):
        trainer._malom_downgrade_reward(
            malom_quality=quality,
            mode="malom-preserving-plus-downgrade-penalty",
        )


def test_existing_modes_retain_zero_per_move_malom_reward() -> None:
    for mode in (
        "legacy-unconditional",
        "malom-preserving-only",
        "disabled",
    ):
        assert trainer._malom_downgrade_reward(
            malom_quality=None,
            mode=mode,
        ) == 0.0


def test_parser_preserves_legacy_default_and_accepts_successor_mode() -> None:
    parser = trainer._build_argument_parser()

    legacy = parser.parse_args(["--preflight", "smoke"])
    successor = parser.parse_args(
        [
            "--preflight",
            "smoke",
            "--mill-bonus-mode",
            "malom-preserving-only",
        ]
    )
    penalty = parser.parse_args(
        [
            "--preflight",
            "smoke",
            "--mill-bonus-mode",
            "malom-preserving-plus-downgrade-penalty",
        ]
    )

    assert legacy.mill_bonus_mode == "legacy-unconditional"
    assert successor.mill_bonus_mode == "malom-preserving-only"
    assert (
        penalty.mill_bonus_mode
        == "malom-preserving-plus-downgrade-penalty"
    )


class _OneMoveModel:
    def policy_logits(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(features.shape[0], dtype=torch.float32)


class _UnexpectedOpponent:
    def choose_move(self, _board: BoardState) -> dict:
        raise AssertionError("the one-ply rollout must not ask the opponent")


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


class _FixedMalomQuality:
    def __init__(self, quality: float) -> None:
        self.quality = quality

    def query_move_quality(self, _board: BoardState, _move: dict) -> float:
        return self.quality


def test_rollout_suppresses_only_the_contradictory_mill_bonus(monkeypatch) -> None:
    board = BoardState.from_setup(
        {"a7": "W", "d7": "W", "a4": "B", "b4": "B"},
        turn="W",
        phase="place",
    )
    move = {"from": None, "to": "g7", "capture": "a4"}
    monkeypatch.setattr(
        trainer,
        "encode_position_with_lookahead",
        lambda *_args, **_kwargs: _encoded(move),
    )
    common = {
        "model": _OneMoveModel(),
        "device": torch.device("cpu"),
        "start_board": board,
        "learner_color": "W",
        "opponent": _UnexpectedOpponent(),
        "opp_color": "B",
        "sentinel": None,
        "value_net": None,
        "temperature": 0.9,
        "max_ply": 1,
        "record_branches": False,
        "branch_every": 0,
        "retry_ply": 0,
        "malom_db": _FixedMalomQuality(-1.0),
        "persist_rollout_evidence": False,
    }

    legacy = trainer._rollout(
        **common,
        mill_bonus_mode="legacy-unconditional",
    )
    corrected = trainer._rollout(
        **common,
        mill_bonus_mode="malom-preserving-only",
    )
    penalty = trainer._rollout(
        **common,
        mill_bonus_mode="malom-preserving-plus-downgrade-penalty",
    )

    assert legacy.trajectory[0].reward == pytest.approx(trainer.MILL_BONUS)
    assert corrected.trajectory[0].reward == 0.0
    assert penalty.trajectory[0].reward == pytest.approx(
        -trainer.MALOM_DOWNGRADE_PENALTY
    )
    assert corrected.step_diags[0].mills_formed == 1
    assert corrected.step_diags[0].malom_quality == -1.0
    assert corrected.step_diags[0].mill_bonus_awarded == 0.0
    assert corrected.step_diags[0].board_phase == "place"
    assert penalty.step_diags[0].reward.malom == pytest.approx(
        -trainer.MALOM_DOWNGRADE_PENALTY
    )


def test_non_mill_downgrade_receives_the_new_penalty() -> None:
    enc = _encoded({"from": None, "to": "a7", "capture": None})

    reward, breakdown = trainer._compute_per_move_reward(
        enc,
        0,
        None,
        board_phase="place",
        malom_q=-2.0,
        mill_bonus_mode="malom-preserving-plus-downgrade-penalty",
    )

    assert reward == pytest.approx(-2 * trainer.MALOM_DOWNGRADE_PENALTY)
    assert breakdown.malom == reward
    assert breakdown.mill_formed == 0.0
