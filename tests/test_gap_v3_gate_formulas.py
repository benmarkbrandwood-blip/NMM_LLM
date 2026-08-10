"""tests/test_gap_v3_gate_formulas.py — hand-verified Stage E gate formulas.

Decision 3A (2026-08-06): reference framing —
  - High-support rows: empirical G_v is the reference; candidate/teacher/uniform MSE
    are each computed against it.
  - Model-only rows: teacher-fidelity MSE reported separately (not empirical validation).
  - Mean-predictor baseline reported separately.
"""
from __future__ import annotations

import importlib.util
import math
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


def test_mse_where_basic(trainer):
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    ref  = np.array([1.5, 2.5, 2.5, 5.0])
    mask = np.array([True, True, False, True])
    # (0.25 + 0.25 + 1.00) / 3 = 0.5
    assert math.isclose(trainer._mse_where(pred, ref, mask), 0.5, rel_tol=1e-9)


def test_mse_where_empty_mask(trainer):
    pred = np.array([1.0, 2.0])
    ref  = np.array([1.0, 2.0])
    mask = np.array([False, False])
    assert math.isnan(trainer._mse_where(pred, ref, mask))


def test_report_gate_shape_and_labels(trainer):
    rng = np.random.default_rng(0)
    n = 300
    val_pred = rng.standard_normal((n, 3)).astype(np.float32)
    y_model  = rng.standard_normal((n, 3)).astype(np.float32)
    y_unif   = rng.standard_normal((n, 3)).astype(np.float32)
    y_emp    = rng.standard_normal((n, 3)).astype(np.float32)
    # 40% of empirical rows unavailable
    emp_missing = rng.random(n) < 0.4
    y_emp[emp_missing, :] = np.nan
    band_idx = rng.integers(0, 3, size=n)
    tr_means = np.array([0.1, 0.2, 0.3])

    results = trainer._report_gate(val_pred, y_model, y_unif, y_emp, band_idx, tr_means)
    assert len(results) == 3
    for comp in results:
        assert "component" in comp
        assert set(comp["per_band"].keys()) == {"lower", "middle", "upper"}
        for band_row in comp["per_band"].values():
            assert "candidate_teacher_fidelity_mse" in band_row
            assert "mean_predictor_vs_teacher_mse" in band_row


def test_hand_computed_high_support_gate(trainer):
    # One band, one component — deterministic small input.
    val_pred = np.array([[0.10], [0.20], [0.30], [0.40]])
    y_model  = np.array([[0.15], [0.25], [0.28], [0.42]])   # teacher target
    y_unif   = np.array([[0.20], [0.30], [0.30], [0.50]])   # uniform-P_h G_v
    y_emp    = np.array([[0.12], [0.22], [np.nan], [0.44]]) # empirical G_v
    band_idx = np.array([0, 0, 0, 0])
    tr_means = np.array([0.25, 0.0, 0.0])   # only comp 0 is exercised; other slots ignored

    # Adapt shapes to (N, 3) with the other components NaN so they contribute nothing
    def _pad(a):
        out = np.full((len(a), 3), np.nan, dtype=np.float32)
        out[:, 0] = a[:, 0]
        return out

    val_pred_p = _pad(val_pred)
    y_model_p  = _pad(y_model)
    y_unif_p   = _pad(y_unif)
    y_emp_p    = _pad(y_emp)

    results = trainer._report_gate(
        val_pred_p, y_model_p, y_unif_p, y_emp_p, band_idx, tr_means,
    )
    row = results[0]["per_band"]["lower"]

    # High-support rows: indices 0, 1, 3
    hi = [0, 1, 3]
    exp_cand = float(np.mean((val_pred[hi, 0] - y_emp[hi, 0]) ** 2))
    exp_teach = float(np.mean((y_model[hi, 0] - y_emp[hi, 0]) ** 2))
    exp_unif = float(np.mean((y_unif[hi, 0] - y_emp[hi, 0]) ** 2))
    assert math.isclose(row["candidate_vs_empirical_mse"], exp_cand, rel_tol=1e-6)
    assert math.isclose(row["teacher_vs_empirical_mse"], exp_teach, rel_tol=1e-6)
    assert math.isclose(row["uniform_vs_empirical_mse"], exp_unif, rel_tol=1e-6)

    # Teacher-fidelity uses ALL four rows (all valid on y_model side)
    exp_tf = float(np.mean((val_pred[:, 0] - y_model[:, 0]) ** 2))
    assert math.isclose(row["candidate_teacher_fidelity_mse"], exp_tf, rel_tol=1e-6)

    # Mean predictor MSE = mean((tr_mean - y_model)**2) over valid_model
    exp_mp = float(np.mean((0.25 - y_model[:, 0]) ** 2))
    assert math.isclose(row["mean_predictor_vs_teacher_mse"], exp_mp, rel_tol=1e-6)


def test_no_high_support_rows_reports_teacher_fidelity_only(trainer):
    n = 20
    val_pred = np.random.default_rng(1).standard_normal((n, 3)).astype(np.float32)
    y_model  = np.random.default_rng(2).standard_normal((n, 3)).astype(np.float32)
    y_unif   = np.zeros((n, 3), dtype=np.float32)
    y_emp    = np.full((n, 3), np.nan, dtype=np.float32)  # no empirical anywhere
    band_idx = np.zeros(n, dtype=np.int64)
    tr_means = np.array([0.0, 0.0, 0.0])

    results = trainer._report_gate(val_pred, y_model, y_unif, y_emp, band_idx, tr_means)
    row = results[0]["per_band"]["lower"]
    assert row["n_high_support"] == 0
    assert "candidate_vs_empirical_mse" not in row
    assert not math.isnan(row["candidate_teacher_fidelity_mse"])
