"""Regression tests for additive no-update rollout probe controls."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from game.board import BoardState
from scripts import train_s_gen_v2 as trainer


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


def _one_ply_rollout(**overrides):
    arguments = {
        "model": _OneMoveModel(),
        "device": torch.device("cpu"),
        "start_board": BoardState.new_game(),
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
    }
    arguments.update(overrides)
    return trainer._rollout(**arguments)


def test_probe_suppresses_persistence_but_preserves_specialist_reads(
    monkeypatch,
) -> None:
    move = {"from": None, "to": "a7", "capture": None}
    specialist = object()
    observed: list[object] = []

    def encode(_board, *_args, **kwargs):
        observed.append(kwargs.get("specialist_db"))
        return _encoded(move)

    monkeypatch.setattr(trainer, "encode_position_with_lookahead", encode)
    monkeypatch.setattr(
        trainer,
        "_persist_rollout_evidence",
        lambda **_kwargs: pytest.fail("probe attempted rollout persistence"),
    )

    result = _one_ply_rollout(
        specialist_db=specialist,
        persist_rollout_evidence=False,
    )

    assert observed[0] is specialist
    assert result.termination_reason == "max-ply-truncation"


def test_rollout_persistence_remains_enabled_by_default(monkeypatch) -> None:
    move = {"from": None, "to": "a7", "capture": None}
    calls: list[dict] = []
    monkeypatch.setattr(
        trainer,
        "encode_position_with_lookahead",
        lambda *_args, **_kwargs: _encoded(move),
    )
    monkeypatch.setattr(
        trainer,
        "_persist_rollout_evidence",
        lambda **kwargs: calls.append(kwargs),
    )

    _one_ply_rollout(specialist_db=object())

    assert len(calls) == 1
    assert calls[0]["learner_color"] == "W"


def test_timing_observer_does_not_change_rollout_semantics(monkeypatch) -> None:
    move = {"from": None, "to": "a7", "capture": None}
    monkeypatch.setattr(
        trainer,
        "encode_position_with_lookahead",
        lambda *_args, **_kwargs: _encoded(move),
    )
    baseline = _one_ply_rollout(persist_rollout_evidence=False)
    timings: list[tuple[str, float]] = []
    observed = _one_ply_rollout(
        persist_rollout_evidence=False,
        timing_observer=lambda stage, seconds: timings.append((stage, seconds)),
    )

    assert (observed.outcome, observed.ply, observed.termination_reason) == (
        baseline.outcome,
        baseline.ply,
        baseline.termination_reason,
    )
    assert observed.trajectory[0].chosen_idx == baseline.trajectory[0].chosen_idx
    assert observed.trajectory[0].reward == baseline.trajectory[0].reward
    assert observed.phase_ply_counts == {"place": 1}
    assert {stage for stage, _seconds in timings} >= {
        "learner_encode",
        "learner_policy",
        "successor_encode",
        "malom_move_quality",
    }
    assert all(seconds >= 0.0 for _stage, seconds in timings)


def test_deep_route_restores_simulation_depth_after_exception(monkeypatch) -> None:
    advisor = SimpleNamespace(_sim_ply_depth=5, _ply_depth=12)

    def fail_encode(*_args, **_kwargs):
        assert advisor._sim_ply_depth == 12
        raise RuntimeError("synthetic encoder failure")

    monkeypatch.setattr(trainer, "encode_position_with_lookahead", fail_encode)

    with pytest.raises(RuntimeError, match="synthetic encoder failure"):
        _one_ply_rollout(
            lookahead_advisor=advisor,
            deep_game=True,
            persist_rollout_evidence=False,
        )

    assert advisor._sim_ply_depth == 5
