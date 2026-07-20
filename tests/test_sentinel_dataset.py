"""Tests for learned_ai/sentinel/dataset.py (loads from data/games)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from learned_ai.sentinel.dataset import (
    SentinelDataset,
    examples_from_game,
)
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.sentinel.feature_builder import FEATURE_DIM

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GAME_DIR = _REPO_ROOT / "data" / "games"


def _have_games():
    return _GAME_DIR.is_dir() and next(_GAME_DIR.rglob("*.jsonl"), None) is not None


requires_game_logs = pytest.mark.skipif(
    not _have_games(),
    reason="requires local JSONL game logs under data/games",
)


@requires_game_logs
def test_load_from_games_no_crash():
    ds = SentinelDataset.load_from_games(str(_GAME_DIR), db=ExternalSolvedDB(""), limit=20)
    assert len(ds) > 0


@requires_game_logs
def test_item_shape_and_targets():
    ds = SentinelDataset.load_from_games(str(_GAME_DIR), db=ExternalSolvedDB(""), limit=10)
    feat, label = ds[0]
    assert tuple(feat.shape) == (FEATURE_DIM,)
    # label is (quality: float, weight: float, wdl_cls: int)
    quality, weight, wdl_cls = label
    assert 0.0 <= float(quality) <= 1.0
    assert float(weight) > 0.0
    assert wdl_cls in (-1, 0, 1, 2)


@requires_game_logs
def test_dataset_length_positive_per_game():
    # A handful of games should each yield at least one example.
    ds = SentinelDataset.load_from_games(str(_GAME_DIR), db=ExternalSolvedDB(""), limit=5)
    assert len(ds) >= 5


@requires_game_logs
def test_save_load_roundtrip(tmp_path):
    ds = SentinelDataset.load_from_games(str(_GAME_DIR), db=ExternalSolvedDB(""), limit=10)
    path = str(tmp_path / "ds.npz")
    ds.save_to_disk(path)
    ds2 = SentinelDataset.load_from_disk(path)
    assert len(ds2) == len(ds)
    f1, (q1, w1, _) = ds[0]
    f2, (q2, w2, _) = ds2[0]
    assert np.allclose(np.asarray(f1), np.asarray(f2), atol=1e-6)
    assert abs(float(q1) - float(q2)) < 1e-5
    assert abs(float(w1) - float(w2)) < 1e-5


@requires_game_logs
def test_quality_distribution_has_multiple_types():
    ds = SentinelDataset.load_from_games(str(_GAME_DIR), db=ExternalSolvedDB(""), limit=60)
    dist = ds.quality_distribution()
    assert len(dist) >= 2        # at least win + loss buckets
    assert sum(dist.values()) == len(ds)


def test_examples_from_game_handles_empty():
    assert examples_from_game({"moves": []}) == []
    assert examples_from_game({}) == []
