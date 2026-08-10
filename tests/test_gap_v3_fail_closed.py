"""tests/test_gap_v3_fail_closed.py — trainer refuses to load non-finite targets.

Decision from Codex review 2026-08-06: fail-closed validation of A/B/C targets.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

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


def _write_synthetic_dataset(
    dir_path: Path,
    n_rows: int = 8,
    corrupt_file: str | None = None,
    corrupt_value: float = float("inf"),
) -> None:
    """Write a minimal synthetic dataset in the v2 layout under dir_path."""
    board_dim = 79
    n_heads   = 3

    rng    = np.random.default_rng(42)
    feats  = rng.standard_normal((n_rows, board_dim)).astype(np.float32)
    tgts   = rng.standard_normal((n_rows, n_heads)).astype(np.float32)
    unif   = rng.standard_normal((n_rows, n_heads)).astype(np.float32)
    emp    = rng.standard_normal((n_rows, n_heads)).astype(np.float32)
    band   = np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int8)
    split  = np.array([0] * n_rows, dtype=np.int8)   # all-train for the test

    if corrupt_file == "targets":
        tgts[0, 0] = corrupt_value
    elif corrupt_file == "targets_uniform":
        unif[1, 1] = corrupt_value
    elif corrupt_file == "targets_empirical":
        emp[2, 2] = corrupt_value

    dir_path.mkdir(parents=True, exist_ok=True)
    feats.tofile(dir_path / "parent_feats.f32.bin")
    tgts.tofile(dir_path  / "targets.f32.bin")
    unif.tofile(dir_path  / "targets_uniform.f32.bin")
    emp.tofile(dir_path   / "targets_empirical.f32.bin")

    np.savez(
        str(dir_path / "metadata.npz"),
        state_keys=np.array([f"sk{i}" for i in range(n_rows)], dtype=object),
        band_idx=band,
        split=split,
        phase=np.array(["place"] * n_rows, dtype=object),
        mover_color=np.array(["W"] * n_rows, dtype=object),
        n_legal=np.array([5] * n_rows, dtype=np.int16),
        ph_source=np.array(["model"] * n_rows, dtype=object),
        provenance=np.array("{}", dtype=object),
    )


def test_clean_dataset_loads(trainer, tmp_path):
    _write_synthetic_dataset(tmp_path)
    result = trainer._load_split(tmp_path, split_val=0)
    assert result["board"].shape[0] == 8
    assert result["y_model"].shape == (8, 3)


@pytest.mark.parametrize("corrupt_file", ["targets", "targets_uniform", "targets_empirical"])
def test_inf_rejected(trainer, tmp_path, corrupt_file):
    _write_synthetic_dataset(tmp_path, corrupt_file=corrupt_file, corrupt_value=float("inf"))
    with pytest.raises(ValueError, match=corrupt_file):
        trainer._load_split(tmp_path, split_val=0)


def test_neg_inf_rejected(trainer, tmp_path):
    _write_synthetic_dataset(tmp_path, corrupt_file="targets", corrupt_value=float("-inf"))
    with pytest.raises(ValueError):
        trainer._load_split(tmp_path, split_val=0)


def test_nan_targets_allowed(trainer, tmp_path):
    """NaN is the sentinel for 'unavailable' and MUST be allowed."""
    _write_synthetic_dataset(tmp_path)
    # Directly patch some NaNs into targets after clean write
    tgts = np.memmap(
        str(tmp_path / "targets.f32.bin"),
        dtype="float32", mode="r+", shape=(8, 3),
    )
    tgts[3, 1] = np.nan
    tgts.flush()
    result = trainer._load_split(tmp_path, split_val=0)
    assert bool(np.isnan(result["y_model"][3, 1].item()))
