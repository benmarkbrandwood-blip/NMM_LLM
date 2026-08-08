"""Focused route-parity tests for the frozen Generalist opponent."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from game.board import BoardState
from scripts import train_s_gen_v2 as trainer


class _FixedLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def policy_logits(self, features: torch.Tensor) -> torch.Tensor:
        assert features.shape[0] == 2
        return torch.tensor([0.0, 1.0], device=features.device) + self.anchor


def test_frozen_opponent_uses_the_production_feature_route(monkeypatch) -> None:
    move = {"from": None, "to": "a7", "capture": None}
    advisor = object()
    specialist = object()
    observed: list[dict] = []

    def encode(_board, *_args, **kwargs):
        observed.append(kwargs)
        return SimpleNamespace(
            feat_matrix=np.zeros(
                (1, trainer.MOVE_FEAT_DIM_WITH_LOOKAHEAD), dtype=np.float32
            ),
            legal_moves=[move],
        )

    monkeypatch.setattr(trainer, "encode_position_with_lookahead", encode)
    model = trainer.ScaffoldedPolicyNet(
        move_feat_dim=trainer.MOVE_FEAT_DIM_WITH_LOOKAHEAD
    )
    opponent = trainer.FrozenModelOpponent(
        model,
        torch.device("cpu"),
        lookahead_advisor=advisor,
        specialist_db=specialist,
    )

    assert opponent.choose_move(BoardState.new_game()) == move
    assert observed == [
        {
            "sentinel_advisor": None,
            "db": None,
            "value_net": None,
            "lookahead_advisor": advisor,
            "specialist_db": specialist,
            "sdb_min_samples": 3,
        }
    ]


def test_frozen_opponent_keeps_deterministic_argmax_selection(monkeypatch) -> None:
    moves = [
        {"from": None, "to": "a7", "capture": None},
        {"from": None, "to": "d7", "capture": None},
    ]
    monkeypatch.setattr(
        trainer,
        "encode_position_with_lookahead",
        lambda *_args, **_kwargs: SimpleNamespace(
            feat_matrix=np.zeros(
                (2, trainer.MOVE_FEAT_DIM_WITH_LOOKAHEAD), dtype=np.float32
            ),
            legal_moves=moves,
        ),
    )
    opponent = trainer.FrozenModelOpponent(
        _FixedLogitModel(),
        torch.device("cpu"),
        lookahead_advisor=object(),
        specialist_db=object(),
    )

    assert [opponent.choose_move(BoardState.new_game()) for _ in range(4)] == [
        moves[1]
    ] * 4
