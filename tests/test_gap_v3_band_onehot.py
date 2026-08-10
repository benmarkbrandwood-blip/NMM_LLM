"""tests/test_gap_v3_band_onehot.py — 82-feature band one-hot construction.

Decision 2A (2026-08-06): input = 79 board + 3-way band one-hot.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_gap_net_v3", _ROOT / "tools" / "train_gap_net_v3.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trainer():
    return _load_trainer()


def test_input_dim_is_82(trainer):
    assert trainer._BOARD_DIM == 79
    assert trainer._N_BANDS == 3
    assert trainer._INPUT_DIM == 82


def test_band_onehot_placement(trainer):
    board = torch.zeros(3, 79)
    bands = torch.tensor([0, 1, 2])
    feats = trainer._build_features(board, bands)
    assert feats.shape == (3, 82)
    # Band block sits at indices 79..81 (last three columns)
    assert feats[0, 79] == 1.0 and feats[0, 80] == 0.0 and feats[0, 81] == 0.0
    assert feats[1, 79] == 0.0 and feats[1, 80] == 1.0 and feats[1, 81] == 0.0
    assert feats[2, 79] == 0.0 and feats[2, 80] == 0.0 and feats[2, 81] == 1.0


def test_board_block_preserved(trainer):
    torch.manual_seed(0)
    board = torch.randn(5, 79)
    bands = torch.tensor([0, 1, 2, 1, 0])
    feats = trainer._build_features(board, bands)
    assert torch.equal(feats[:, :79], board)
    # Each row has exactly one 1.0 in the band block
    assert (feats[:, 79:82].sum(dim=1) == 1.0).all()


def test_model_accepts_82(trainer):
    model = trainer.GapNetV3()
    x = torch.randn(4, 82)
    y = model(x)
    assert y.shape == (4, 3)
