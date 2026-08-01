"""tests/test_human_move_policy_eval.py

Tests for tools/eval_human_move_policy_net.py.

Unit tests (TestStratumECE, TestDegradeCal):
    Hand-crafted inputs with known analytic outputs.  No DB, no torch.

Integration test (TestEvalEndToEnd):
    Extract a tiny slice, train 1 epoch, run full eval.
    Requires PyTorch + candidate DB.
"""
from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

from tools.eval_human_move_policy_net import DegradeCal, Stratum  # noqa: E402

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

_CANDIDATE_DB = _ROOT / "data" / "human_db_candidate.sqlite"
_CANDIDATE_AVAILABLE = _CANDIDATE_DB.exists()


# ── Unit: Stratum ECE ────────────────────────────────────────────────────────

class TestStratumECE(unittest.TestCase):
    """Unit tests for Stratum.finalize() top-label ECE computation."""

    def _make(self, probs, targets, notations=None):
        """Build a Stratum from a single sample with given probs/targets."""
        k = len(probs)
        if notations is None:
            notations = [str(j) for j in range(k)]
        p = np.array(probs, dtype=np.float32)
        t = np.array(targets, dtype=np.int64)
        top_order = np.argsort(-p)
        top1 = notations[top_order[0]]
        top3 = {notations[j] for j in top_order[:3]}
        top5 = {notations[j] for j in top_order[:5]}
        s = Stratum("test")
        s.add_sample(p, t, notations, top1, top3, top5)
        return s.finalize()

    def test_perfect_calibration_uniform(self):
        """Uniform model + uniform human choices → ECE = 0.

        All 4 events land in the same bin (top_conf=0.25).
        Bin confidence = 0.25, bin accuracy = 1/4 = 0.25 → gap = 0.
        """
        probs   = [0.25, 0.25, 0.25, 0.25]
        targets = [1, 1, 1, 1]
        result  = self._make(probs, targets)
        self.assertAlmostEqual(result["ece"], 0.0, places=6)

    def test_perfect_calibration_confident_correct(self):
        """Model confident in move 0 (0.9), human plays move 0 90% of the time → ECE = 0.

        top_conf=0.9 for all 10 events, bin accuracy=9/10=0.9 → gap=0.
        """
        probs   = [0.9, 0.1]
        targets = [9, 1]
        result  = self._make(probs, targets)
        self.assertAlmostEqual(result["ece"], 0.0, places=6)

    def test_known_miscalibration(self):
        """Model says 0.9 confidence but only right 1/2 the time → ECE = 0.4.

        top_conf=0.9 for both events: conf=0.9, acc=0.5, gap=0.4.
        """
        probs   = [0.9, 0.1]
        targets = [1, 1]
        result  = self._make(probs, targets)
        self.assertAlmostEqual(result["ece"], 0.4, places=6)

    def test_ece_in_unit_interval(self):
        """ECE must always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            k = rng.integers(2, 8)
            raw = rng.exponential(1.0, k).astype(np.float32)
            p   = raw / raw.sum()
            t   = rng.integers(0, 5, k).astype(np.int64)
            if t.sum() == 0:
                continue
            result = self._make(p.tolist(), t.tolist())
            self.assertGreaterEqual(result["ece"], 0.0)
            self.assertLessEqual(result["ece"], 1.0)

    def test_bins_by_top_conf_not_played_prob(self):
        """Events from the same position must all land in the same ECE bin.

        If the wrong move was played (move 1, prob=0.1) and the model is
        confident (top_conf=0.9), both events must go into the 0.9 bin, not
        the 0.1 bin.  We verify this indirectly: single-position ECE must be
        |top_conf - accuracy|, where accuracy = fraction of events where the
        played move = model's top-1.
        """
        probs   = [0.9, 0.1]
        targets = [0, 1]   # human played move 1 (not model's top choice)
        result  = self._make(probs, targets)
        # top_conf=0.9, 1 event, was_top1=False → acc=0.0 → ECE=|0.9-0.0|=0.9
        self.assertAlmostEqual(result["ece"], 0.9, places=6)


# ── Unit: DegradeCal ─────────────────────────────────────────────────────────

class TestDegradeCal(unittest.TestCase):
    """Unit tests for DegradeCal.finalize()."""

    def test_no_degrade_positions(self):
        """All positions have zero degrading probability → both ECEs = 0."""
        dc = DegradeCal()
        for _ in range(5):
            dc.add_position(0.0, 0.0, 0, 0.0, 10)
        result = dc.finalize()
        self.assertAlmostEqual(result["mean_pred_prob_degrade"], 0.0)
        self.assertAlmostEqual(result["degrade_ece"], 0.0)

    def test_perfect_degrade_calibration(self):
        """pred=obs for every position → both ECEs = 0."""
        dc = DegradeCal()
        # pred_degrade = 0.1, observed 10% degrade events
        for _ in range(10):
            dc.add_position(0.1, 0.05, 1, 0.5, 10)
        result = dc.finalize()
        self.assertAlmostEqual(result["degrade_ece"], 0.0, places=6)
        self.assertAlmostEqual(result["regret_ece"],  0.0, places=6)

    def test_known_degrade_miscalibration(self):
        """pred=0.4 but obs=0.0 for all positions → degrade_ece = 0.4."""
        dc = DegradeCal()
        for _ in range(10):
            dc.add_position(0.4, 0.0, 0, 0.0, 10)
        result = dc.finalize()
        self.assertAlmostEqual(result["degrade_ece"], 0.4, places=6)

    def test_obs_degrade_freq_correct(self):
        """Aggregate obs_degrade_freq = total_obs_degrade / total_events."""
        dc = DegradeCal()
        dc.add_position(0.1, 0.05, 2, 1.0, 10)   # 2 degrade, 10 total
        dc.add_position(0.2, 0.10, 4, 2.0, 20)   # 4 degrade, 20 total
        result = dc.finalize()
        expected_freq = 6 / 30
        self.assertAlmostEqual(result["obs_degrade_freq"], expected_freq, places=6)

    def test_ece_in_unit_interval(self):
        rng = np.random.default_rng(7)
        dc  = DegradeCal()
        for _ in range(30):
            pd = float(rng.uniform(0, 0.5))
            pr = float(rng.uniform(0, pd))
            od = int(rng.integers(0, 6))
            ot = int(rng.integers(6, 20))
            dc.add_position(pd, pr, od, float(od * 0.5), ot)
        r = dc.finalize()
        self.assertGreaterEqual(r["degrade_ece"], 0.0)
        self.assertLessEqual(r["degrade_ece"], 1.0)
        self.assertGreaterEqual(r["regret_ece"], 0.0)
        self.assertLessEqual(r["regret_ece"], 1.0)


# ── Integration: end-to-end eval ─────────────────────────────────────────────

@unittest.skipUnless(_TORCH_AVAILABLE and _CANDIDATE_AVAILABLE,
                     "Requires PyTorch + candidate DB")
class TestEvalEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scratch = _ROOT / "data" / "_test_hbn_eval"
        if cls.scratch.exists():
            shutil.rmtree(cls.scratch)
        cls.scratch.mkdir(parents=True)
        cls.dataset_dir = cls.scratch / "dataset"
        cls.model       = cls.scratch / "hbn.npz"
        cls.report      = cls.scratch / "eval.json"

        import extract_human_move_policy_dataset as ext
        ext.extract(_CANDIDATE_DB, cls.dataset_dir, limit_state_keys=400)

        import argparse
        from tools.train_human_move_policy_net import train
        train_args = argparse.Namespace(
            dataset_dir=cls.dataset_dir,
            output=cls.model,
            epochs=1, patience=0, lr=3e-4, dropout=0.2,
            batch_positions=64, grad_clip=1.0, seed=42,
        )
        train(train_args)

        from tools.eval_human_move_policy_net import evaluate
        cls.full_report = evaluate(cls.dataset_dir, cls.model, _CANDIDATE_DB,
                                   min_support=10)
        cls.val  = cls.full_report["val"]["model"]
        cls.val_root = cls.full_report["val"]

    @classmethod
    def tearDownClass(cls):
        if cls.scratch.exists():
            shutil.rmtree(cls.scratch)

    # ── Report structure ──

    def test_top_level_keys(self):
        for k in ("meta", "val", "provenance"):
            self.assertIn(k, self.full_report)

    def test_val_model_keys(self):
        for k in ("overall", "by_band", "by_phase", "by_transition",
                  "by_lmc", "ood", "abstention"):
            self.assertIn(k, self.val)

    def test_degrade_calibration_key_present(self):
        self.assertIn("degrade_calibration", self.val_root)

    # ── Overall metrics ──

    def test_overall_has_all_metrics(self):
        overall = self.val["overall"]
        for k in ("n_events", "event_nll", "brier", "top1", "top3", "top5", "ece"):
            self.assertIn(k, overall)
        self.assertGreater(overall["n_events"], 0)
        self.assertGreaterEqual(overall["event_nll"], 0.0)
        self.assertLessEqual(overall["top1"], 1.0)
        self.assertLessEqual(overall["top3"], 1.0)
        self.assertLessEqual(overall["top5"], 1.0)

    def test_ece_in_unit_interval(self):
        ece = self.val["overall"]["ece"]
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_ece_not_trivially_zero(self):
        """ECE should not be near 0 — a near-zero ECE would indicate the old
        broken metric (binning by played-move probability) is still in use."""
        ece = self.val["overall"]["ece"]
        self.assertGreater(ece, 0.01,
            f"ECE={ece:.4f} is suspiciously low — check top-label binning")

    # ── Band / phase / transition structure ──

    def test_all_three_bands_reported(self):
        by_band = self.val["by_band"]
        for b in ("lower", "middle", "upper"):
            self.assertIn(b, by_band)

    def test_both_phases_reported(self):
        by_phase = self.val["by_phase"]
        for p in ("place", "move"):
            self.assertIn(p, by_phase)

    def test_transitions_include_expected_categories(self):
        by_trans = self.val["by_transition"]
        self.assertIn("draw_preserved", by_trans)

    # ── Degrade calibration ──

    def test_degrade_calibration_has_expected_keys(self):
        dc = self.val_root["degrade_calibration"]
        for k in ("n_positions_labelled", "mean_pred_prob_degrade",
                  "obs_degrade_freq", "degrade_ece",
                  "mean_pred_regret", "obs_regret", "regret_ece"):
            self.assertIn(k, dc)

    def test_degrade_calibration_n_positions_positive(self):
        dc = self.val_root["degrade_calibration"]
        self.assertGreater(dc["n_positions_labelled"], 0)

    def test_degrade_ece_in_unit_interval(self):
        dc = self.val_root["degrade_calibration"]
        self.assertGreaterEqual(dc["degrade_ece"], 0.0)
        self.assertLessEqual(dc["degrade_ece"], 1.0)
        self.assertGreaterEqual(dc["regret_ece"], 0.0)
        self.assertLessEqual(dc["regret_ece"], 1.0)

    # ── Baselines / provenance ──

    def test_provenance_passthrough(self):
        prov = self.full_report["provenance"]
        self.assertIsNotNone(prov)
        self.assertIn("training_objective", prov)

    def test_empirical_kl_supported(self):
        ek = self.val_root["empirical_kl_supported"]
        self.assertIn("mean_kl", ek)
        self.assertIn("n_samples", ek)
        self.assertGreaterEqual(ek["mean_kl"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
