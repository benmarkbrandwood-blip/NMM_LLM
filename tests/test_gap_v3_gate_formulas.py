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

    # P2-A: min_n_high_support=0 to force all cells to attempt evaluation
    results, summary = trainer._report_gate(
        val_pred, y_model, y_unif, y_emp, band_idx, tr_means,
        min_n_high_support=0,
    )
    assert len(results) == 3
    for comp in results:
        assert "component" in comp
        assert set(comp["per_band"].keys()) == {"lower", "middle", "upper"}
        for band_row in comp["per_band"].values():
            assert "candidate_teacher_fidelity_mse" in band_row
            assert "mean_predictor_vs_teacher_mse" in band_row
            assert "gate_1_verdict" in band_row
            assert "gate_2_verdict" in band_row
            assert "cell_verdict"   in band_row
    assert summary["overall_verdict"] in ("PASS", "FAIL", "INSUFFICIENT_SUPPORT")
    assert summary["n_cells_total"] == 9
    assert "thresholds" in summary


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

    results, _ = trainer._report_gate(
        val_pred_p, y_model_p, y_unif_p, y_emp_p, band_idx, tr_means,
        min_n_high_support=0,
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


def test_no_high_support_rows_skips_gate(trainer):
    """P2-A: cells with n_high_support=0 report SKIP_INSUFFICIENT_SUPPORT,
    not a pass/fail verdict.  Teacher-fidelity is still emitted (informational)."""
    n = 20
    val_pred = np.random.default_rng(1).standard_normal((n, 3)).astype(np.float32)
    y_model  = np.random.default_rng(2).standard_normal((n, 3)).astype(np.float32)
    y_unif   = np.zeros((n, 3), dtype=np.float32)
    y_emp    = np.full((n, 3), np.nan, dtype=np.float32)  # no empirical anywhere
    band_idx = np.zeros(n, dtype=np.int64)
    tr_means = np.array([0.0, 0.0, 0.0])

    results, summary = trainer._report_gate(
        val_pred, y_model, y_unif, y_emp, band_idx, tr_means,
    )
    row = results[0]["per_band"]["lower"]
    assert row["n_high_support"] == 0
    assert row["gate_1_verdict"] == "SKIP_INSUFFICIENT_SUPPORT"
    assert row["gate_2_verdict"] == "SKIP_INSUFFICIENT_SUPPORT"
    assert row["cell_verdict"]   == "SKIP_INSUFFICIENT_SUPPORT"
    assert not math.isnan(row["candidate_teacher_fidelity_mse"])
    # No cells passed since all skipped → overall INSUFFICIENT_SUPPORT
    assert summary["overall_verdict"] == "INSUFFICIENT_SUPPORT"
    assert summary["n_cells_skip_insufficient"] >= 1


# ── P2-A executable-gate coverage (Codex 2026-08-12) ─────────────────────────

def _build_synthetic_cell(
    n_rows: int,
    candidate_mse: float,
    teacher_mse: float,
    uniform_mse: float,
    band: int = 0,
    component: int = 0,
) -> dict:
    """Return kwargs for _report_gate that produce known MSE values on cell (band, c).

    Trick: set y_emp = 0 for the target cell; then MSE(pred, y_emp) = mean(pred**2).
    Choose pred = sqrt(mse) constant for each of candidate/teacher/uniform.
    """
    val_pred = np.zeros((n_rows, 3), dtype=np.float32)
    y_model  = np.zeros((n_rows, 3), dtype=np.float32)
    y_unif   = np.zeros((n_rows, 3), dtype=np.float32)
    y_emp    = np.full((n_rows, 3), np.nan, dtype=np.float32)
    y_emp[:, component] = 0.0
    val_pred[:, component] = np.sqrt(candidate_mse)
    y_model[:,  component] = np.sqrt(teacher_mse)
    y_unif[:,   component] = np.sqrt(uniform_mse)
    band_idx = np.full(n_rows, band, dtype=np.int64)
    tr_means = np.zeros(3, dtype=np.float64)
    return {
        "val_pred": val_pred, "y_model": y_model, "y_unif": y_unif,
        "y_emp": y_emp, "band_idx": band_idx, "tr_means": tr_means,
    }


def test_gate1_pass_when_candidate_beats_uniform_by_threshold(trainer):
    """Gate 1: candidate MSE 0.5, uniform 1.0, X_A=0.30 → 0.5 ≤ (1-0.30)*1.0=0.70 → PASS."""
    kw = _build_synthetic_cell(n_rows=200,
                               candidate_mse=0.5, teacher_mse=0.5,
                               uniform_mse=1.0)
    results, summary = trainer._report_gate(x_a=0.30, x_b=0.20, **kw)
    row = results[0]["per_band"]["lower"]
    assert row["gate_1_verdict"] == "PASS"


def test_gate1_fail_when_candidate_within_threshold_of_uniform(trainer):
    """Gate 1: candidate 0.75, uniform 1.0, X_A=0.30 → 0.75 > 0.70 → FAIL."""
    kw = _build_synthetic_cell(n_rows=200,
                               candidate_mse=0.75, teacher_mse=0.75,
                               uniform_mse=1.0)
    results, summary = trainer._report_gate(x_a=0.30, x_b=0.20, **kw)
    row = results[0]["per_band"]["lower"]
    assert row["gate_1_verdict"] == "FAIL"
    assert summary["overall_verdict"] == "FAIL"


def test_gate2_pass_when_candidate_within_tolerance_of_teacher(trainer):
    """Gate 2: candidate 1.15, teacher 1.0, X_B=0.20 → 1.15 ≤ 1.20 → PASS."""
    kw = _build_synthetic_cell(n_rows=200,
                               candidate_mse=1.15, teacher_mse=1.0,
                               uniform_mse=10.0)   # gate 1 trivially passes
    results, summary = trainer._report_gate(x_a=0.30, x_b=0.20, **kw)
    row = results[0]["per_band"]["lower"]
    assert row["gate_2_verdict"] == "PASS"


def test_gate2_fail_when_candidate_exceeds_teacher_tolerance(trainer):
    """Gate 2: candidate 1.5, teacher 1.0, X_B=0.20 → 1.5 > 1.20 → FAIL."""
    kw = _build_synthetic_cell(n_rows=200,
                               candidate_mse=1.5, teacher_mse=1.0,
                               uniform_mse=10.0)
    results, summary = trainer._report_gate(x_a=0.30, x_b=0.20, **kw)
    row = results[0]["per_band"]["lower"]
    assert row["gate_2_verdict"] == "FAIL"


def test_insufficient_support_skips_cell(trainer):
    """min_n_high_support enforced: 50 rows with default 100 → SKIP."""
    kw = _build_synthetic_cell(n_rows=50,
                               candidate_mse=0.5, teacher_mse=0.5,
                               uniform_mse=1.0)
    results, _ = trainer._report_gate(min_n_high_support=100, **kw)
    row = results[0]["per_band"]["lower"]
    assert row["gate_1_verdict"] == "SKIP_INSUFFICIENT_SUPPORT"
    assert row["gate_2_verdict"] == "SKIP_INSUFFICIENT_SUPPORT"
    assert row["cell_verdict"]   == "SKIP_INSUFFICIENT_SUPPORT"


def test_degenerate_denominator_skips_cell(trainer):
    """Reference MSE near zero → SKIP_DEGENERATE_DENOMINATOR."""
    kw = _build_synthetic_cell(n_rows=200,
                               candidate_mse=0.0, teacher_mse=0.0,
                               uniform_mse=0.0)
    results, _ = trainer._report_gate(min_denominator=1e-6, **kw)
    row = results[0]["per_band"]["lower"]
    assert row["gate_1_verdict"] == "SKIP_DEGENERATE_DENOMINATOR"
    assert row["gate_2_verdict"] == "SKIP_DEGENERATE_DENOMINATOR"


def test_diverged_reference_fails_closed(trainer):
    """+inf reference MSE → FAIL_DIVERGED (training bug marker)."""
    from tools.train_gap_net_v3 import _evaluate_cell_verdict
    verdict, detail = _evaluate_cell_verdict(
        candidate_mse=1.0, reference_mse=float("inf"),
        n_high_support=1000, x=0.30, direction="min_improvement",
        min_n_high_support=100, min_denominator=1e-9,
    )
    assert verdict == "FAIL_DIVERGED"


def test_overall_verdict_pass_requires_all_cells_pass(trainer):
    """A single failing cell must flip overall to FAIL."""
    # Two bands, but only band 0 has data; band 1 will SKIP_INSUFFICIENT
    # candidate 0.75, uniform 1.0 → FAIL under X_A=0.30 (0.75 > 0.70)
    kw = _build_synthetic_cell(n_rows=200,
                               candidate_mse=0.75, teacher_mse=0.75,
                               uniform_mse=1.0)
    _, summary = trainer._report_gate(x_a=0.30, x_b=0.20, **kw)
    assert summary["overall_verdict"] == "FAIL"
    assert summary["n_cells_fail"] >= 1
    assert len(summary["failing_cells"]) >= 1


def test_summary_records_thresholds(trainer):
    kw = _build_synthetic_cell(n_rows=200,
                               candidate_mse=0.5, teacher_mse=0.5,
                               uniform_mse=1.0)
    _, summary = trainer._report_gate(
        x_a=0.25, x_b=0.15,
        min_n_high_support=50, min_denominator=1e-8, **kw,
    )
    th = summary["thresholds"]
    assert th["x_a"] == 0.25
    assert th["x_b"] == 0.15
    assert th["min_n_high_support"] == 50
    assert th["min_denominator"] == 1e-8
    assert "gate_1_formula" in th
    assert "gate_2_formula" in th
