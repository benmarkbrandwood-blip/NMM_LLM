"""Tests for learned_ai/sentinel/dataset.py (loads from data/games)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.sentinel.dataset import (
    SentinelDataset,
    examples_from_game,
)
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.sentinel.feature_builder import FEATURE_DIM


@pytest.fixture
def game_logs_dir(tmp_path):
    """Create one deterministic six-ply JSONL game for loader tests."""
    board = BoardState.new_game()
    moves = []
    malom_labels = ("win", "loss", "draw")
    for ply in range(6):
        move = get_all_legal_moves(board)[0]
        moves.append(
            {
                "board_fen_before": board.to_fen_string(),
                "color": board.turn,
                **move,
                "malom_move_wdl": malom_labels[ply % len(malom_labels)],
                "malom_dtw": 20 + ply,
            }
        )
        board = board.apply_move(move)

    game_dir = tmp_path / "games"
    game_dir.mkdir()
    record = {"winner": "W", "moves": moves}
    (game_dir / "fixture.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
    return game_dir


def test_load_from_games_no_crash(game_logs_dir):
    ds = SentinelDataset.load_from_games(
        str(game_logs_dir), db=ExternalSolvedDB(""), limit=20
    )
    assert len(ds) > 0


def test_item_shape_and_targets(game_logs_dir):
    ds = SentinelDataset.load_from_games(
        str(game_logs_dir), db=ExternalSolvedDB(""), limit=10
    )
    feat, label = ds[0]
    assert tuple(feat.shape) == (FEATURE_DIM,)
    # label is (quality: float, weight: float, wdl_cls: int)
    quality, weight, wdl_cls = label
    assert 0.0 <= float(quality) <= 1.0
    assert float(weight) > 0.0
    assert wdl_cls in (-1, 0, 1, 2)


def test_dataset_length_positive_for_fixture_positions(game_logs_dir):
    ds = SentinelDataset.load_from_games(
        str(game_logs_dir), db=ExternalSolvedDB(""), limit=5
    )
    assert len(ds) >= 6


def test_save_load_roundtrip(game_logs_dir, tmp_path):
    ds = SentinelDataset.load_from_games(
        str(game_logs_dir), db=ExternalSolvedDB(""), limit=10
    )
    path = str(tmp_path / "ds.npz")
    ds.save_to_disk(path)
    ds2 = SentinelDataset.load_from_disk(path)
    assert len(ds2) == len(ds)
    f1, (q1, w1, _) = ds[0]
    f2, (q2, w2, _) = ds2[0]
    assert np.allclose(np.asarray(f1), np.asarray(f2), atol=1e-6)
    assert abs(float(q1) - float(q2)) < 1e-5
    assert abs(float(w1) - float(w2)) < 1e-5


def test_quality_distribution_has_multiple_types(game_logs_dir):
    ds = SentinelDataset.load_from_games(
        str(game_logs_dir), db=ExternalSolvedDB(""), limit=60
    )
    dist = ds.quality_distribution()
    assert sum(count > 0 for count in dist.values()) >= 2
    assert sum(dist.values()) == len(ds)


def test_examples_from_game_handles_empty():
    assert examples_from_game({"moves": []}) == []
    assert examples_from_game({}) == []
