"""tests/test_gap_v3_three_head.py — 3-head architecture + save/load round-trip.

Decision 4A (2026-08-06): Component D dropped for regret_v1.  Model has three heads.
"""
from __future__ import annotations

import importlib.util
import json
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


def test_three_heads_configured(trainer):
    assert trainer._N_HEADS == 3
    assert len(trainer._COMP_NAMES) == 3
    assert trainer._COMP_NAMES == (
        "class_downgrade", "wdl_utility_loss", "ordinal_rank_loss",
    )


def test_model_output_shape(trainer):
    model = trainer.GapNetV3()
    x = torch.randn(7, 82)
    y = model(x)
    assert y.shape == (7, 3)


def test_save_load_roundtrip(trainer, tmp_path):
    torch.manual_seed(123)
    model = trainer.GapNetV3()
    out = tmp_path / "cand.npz"

    prov = {"seed": 123, "d4_augmentation": "off"}
    trainer._save(model, out, prov)
    assert out.exists()

    z = np.load(str(out), allow_pickle=True)

    arch = json.loads(str(z["architecture"]))
    assert arch["input"]     == 82
    assert arch["heads"]     == 3
    assert arch["board_dim"] == 79
    assert arch["n_bands"]   == 3
    assert arch["components"] == ["class_downgrade", "wdl_utility_loss", "ordinal_rank_loss"]
    assert arch["bands"]     == ["lower", "middle", "upper"]

    loaded_prov = json.loads(str(z["provenance"]))
    assert loaded_prov["seed"] == 123
    assert loaded_prov["d4_augmentation"] == "off"

    # Weights round-trip exactly
    state = model.state_dict()
    for k, v in state.items():
        np.testing.assert_array_equal(z[k], v.numpy())
