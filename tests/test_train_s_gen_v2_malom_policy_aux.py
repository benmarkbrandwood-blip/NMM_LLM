"""Focused Generalist wiring for exact-WDL preserving-set supervision."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from game.board import BoardState
from learned_ai.training.generalist_preflight import PreflightConfigurationError
from scripts import train_s_gen_v2 as trainer


class _ZeroModel:
    def policy_logits(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(features.shape[0], dtype=torch.float32)


class _UnexpectedOpponent:
    def choose_move(self, _board: BoardState) -> dict:
        raise AssertionError("one-ply rollout must not ask the opponent")


class _ExactActionTeacher:
    def __init__(self, legal_moves: list[dict]) -> None:
        self.legal_moves = legal_moves
        self.query_move_quality_calls = 0

    def is_available(self) -> bool:
        return True

    def query_state(self, _board: BoardState) -> str:
        return "D"

    def query_all_moves(self, _board: BoardState, _player: str) -> list[dict]:
        return [
            {"move": dict(self.legal_moves[2]), "wdl": "draw"},
            {"move": dict(self.legal_moves[0]), "wdl": "draw"},
            {"move": dict(self.legal_moves[1]), "wdl": "loss"},
        ]

    def query_move_quality(self, _board: BoardState, _move: dict) -> float:
        self.query_move_quality_calls += 1
        raise AssertionError("all-action labels must replace the duplicate query")


def _encoded(legal_moves: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        feat_matrix=np.zeros(
            (len(legal_moves), trainer.MOVE_FEAT_DIM_WITH_LOOKAHEAD),
            dtype=np.float32,
        ),
        value_input=np.zeros(trainer.VALUE_INPUT_DIM, dtype=np.float32),
        legal_moves=legal_moves,
        sentinel_scores=[],
        h_scores_abs=[0.0] * len(legal_moves),
        h_before=0.0,
        vn_scores_abs=[0.0] * len(legal_moves),
        vn_before=0.0,
    )


def test_rollout_attaches_complete_exact_action_labels(monkeypatch) -> None:
    legal_moves = [
        {"from": None, "to": "a7", "capture": None},
        {"from": None, "to": "d7", "capture": None},
        {"from": None, "to": "g7", "capture": None},
    ]
    teacher = _ExactActionTeacher(legal_moves)
    monkeypatch.setattr(
        trainer,
        "encode_position_with_lookahead",
        lambda *_args, **_kwargs: _encoded(legal_moves),
    )

    result = trainer._rollout(
        model=_ZeroModel(),
        device=torch.device("cpu"),
        start_board=BoardState.new_game(),
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
        forced_placements=["a7"],
        malom_db=teacher,
        persist_rollout_evidence=False,
        mill_bonus_mode="malom-preserving-only",
        malom_policy_aux_coef=0.25,
    )

    assert teacher.query_move_quality_calls == 0
    assert len(result.trajectory) == 1
    assert result.trajectory[0].malom_preserving_mask.tolist() == [True, False, True]
    assert result.step_diags[0].malom_quality == 0.0
    assert result.step_diags[0].malom_preserving_action_count == 2
    assert result.step_diags[0].malom_downgrading_action_count == 1
    assert result.step_diags[0].malom_preserving_probability == pytest.approx(2 / 3)


def test_update_wrapper_passes_auxiliary_configuration_and_diagnostics() -> None:
    calls = []

    def update(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["diagnostics"]["malom_policy_aux_loss"] = 0.125
        return 1.0, 2.0, 3.0

    diagnostics: dict[str, float | int] = {}
    result = trainer._update_if_ready(
        update_fn=update,
        model=object(),
        optimizer=object(),
        steps=[object()] * trainer.MIN_UPDATE_STEPS,
        device=torch.device("cpu"),
        gamma=0.99,
        entropy_coef=0.01,
        malom_policy_aux_coef=0.25,
        diagnostics=diagnostics,
    )

    assert result == (1.0, 2.0, 3.0)
    assert calls[0][1]["malom_policy_aux_coef"] == 0.25
    assert calls[0][1]["diagnostics"] is diagnostics
    assert diagnostics["malom_policy_aux_loss"] == 0.125


def test_parser_defaults_auxiliary_off_and_accepts_nonnegative_value() -> None:
    parser = trainer._build_argument_parser()

    default = parser.parse_args(["--preflight", "smoke"])
    enabled = parser.parse_args(
        ["--preflight", "smoke", "--malom-policy-aux-coef", "0.25"]
    )

    assert default.malom_policy_aux_coef == 0.0
    assert enabled.malom_policy_aux_coef == 0.25


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf"])
def test_parser_rejects_invalid_auxiliary_coefficient(value: str) -> None:
    parser = trainer._build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--preflight", "smoke", "--malom-policy-aux-coef", value]
        )


def test_auxiliary_requires_a2c_and_preserving_only_reward() -> None:
    parser = trainer._build_argument_parser()
    common = [
        "--preflight",
        "smoke",
        "--no-opening-forcing",
        "--malom-policy-aux-coef",
        "0.25",
    ]
    ppo = parser.parse_args(
        [*common, "--mill-bonus-mode", "malom-preserving-only", "--ppo"]
    )
    wrong_reward = parser.parse_args(common)

    with pytest.raises(PreflightConfigurationError, match="requires A2C"):
        trainer.validate_generalist_configuration(ppo)
    with pytest.raises(
        PreflightConfigurationError,
        match="malom-preserving-only",
    ):
        trainer.validate_generalist_configuration(wrong_reward)
