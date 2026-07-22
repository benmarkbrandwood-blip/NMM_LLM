from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest
import torch

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.sentinel.config import SentinelConfig
from learned_ai.sentinel.model import SentinelNet
from scripts import build_gap_dataset as gap_builder


def _empty_db(path) -> None:
    sqlite3.connect(path).close()


def test_build_dataset_rejects_missing_required_sentinel(
    tmp_path, monkeypatch
) -> None:
    db_path = tmp_path / "human.sqlite"
    _empty_db(db_path)
    sentinel_path = tmp_path / "missing.pt"

    monkeypatch.setattr(
        gap_builder,
        "require_current_human_db_malom_labels",
        lambda conn, path: None,
    )
    monkeypatch.setattr(
        gap_builder,
        "_query_categories",
        lambda *args, **kwargs: pytest.fail(
            "database queries must not start without the required Sentinel"
        ),
    )

    with pytest.raises(FileNotFoundError, match="required Sentinel checkpoint"):
        gap_builder.build_dataset(
            db_path,
            sentinel_path,
            tmp_path / "unused-value-net.npz",
            n_per_category=1,
            dtw_threshold=15,
        )


def test_build_dataset_rejects_incompatible_required_sentinel(
    tmp_path, monkeypatch
) -> None:
    db_path = tmp_path / "human.sqlite"
    _empty_db(db_path)
    sentinel_path = tmp_path / "broken.pt"
    sentinel_path.write_bytes(b"not a checkpoint")

    class BrokenAdvisor:
        def __init__(self, *args, **kwargs) -> None:
            raise ValueError("incompatible checkpoint")

    import learned_ai.sentinel.infer as sentinel_infer

    monkeypatch.setattr(sentinel_infer, "SentinelAdvisor", BrokenAdvisor)
    monkeypatch.setattr(
        gap_builder,
        "require_current_human_db_malom_labels",
        lambda conn, path: None,
    )
    monkeypatch.setattr(
        gap_builder,
        "_query_categories",
        lambda *args, **kwargs: pytest.fail(
            "database queries must not start with an incompatible Sentinel"
        ),
    )

    with pytest.raises(RuntimeError, match="required Sentinel checkpoint"):
        gap_builder.build_dataset(
            db_path,
            sentinel_path,
            tmp_path / "unused-value-net.npz",
            n_per_category=1,
            dtw_threshold=15,
        )


def test_load_required_sentinel_accepts_compatible_checkpoint(tmp_path) -> None:
    config = SentinelConfig()
    model = SentinelNet(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    sentinel_path = tmp_path / "sentinel.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.to_dict(),
        },
        sentinel_path,
    )

    advisor = gap_builder._load_required_sentinel(sentinel_path)

    assert advisor.is_loaded()


@pytest.mark.parametrize(
    "advice",
    [
        None,
        SimpleNamespace(move_scores=[]),
        SimpleNamespace(move_scores=[float("nan")]),
    ],
)
def test_score_moves_rejects_invalid_required_sentinel_advice(advice) -> None:
    board = BoardState.new_game()
    legal = list(get_all_legal_moves(board))[:1]

    class InvalidAdvisor:
        def advise(self, *args, **kwargs):
            return advice

    with pytest.raises(RuntimeError, match="required Sentinel"):
        gap_builder._score_moves(board, legal, InvalidAdvisor())


def test_score_moves_rejects_required_sentinel_inference_failure() -> None:
    board = BoardState.new_game()
    legal = list(get_all_legal_moves(board))[:1]

    class FailingAdvisor:
        def advise(self, *args, **kwargs):
            raise ValueError("inference failed")

    with pytest.raises(
        gap_builder.RequiredSentinelError,
        match="failed during GapNet scoring",
    ):
        gap_builder._score_moves(board, legal, FailingAdvisor())


def test_score_moves_uses_valid_required_sentinel_advice() -> None:
    board = BoardState.new_game()
    legal = list(get_all_legal_moves(board))[:2]

    class ValidAdvisor:
        def advise(self, *args, **kwargs):
            return SimpleNamespace(move_scores=[0.25, 0.75])

    scores = gap_builder._score_moves(board, legal, ValidAdvisor())

    assert len(scores) == 2
    assert all(0.0 <= score <= 1.0 for score in scores.values())
