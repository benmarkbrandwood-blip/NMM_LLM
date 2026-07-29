"""tests/test_human_blunder_eval.py

Smoke test for tools/eval_human_blunder_net.py.  Extracts a tiny slice,
trains 1 epoch, runs eval, and asserts the report contains every
promised metric and stratum.
"""
from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

_CANDIDATE_DB = _ROOT / "data" / "human_db_candidate.sqlite"
_CANDIDATE_AVAILABLE = _CANDIDATE_DB.exists()


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

        sys.path.insert(0, str(_ROOT / "tools"))
        import extract_human_blunder_dataset as ext
        ext.extract(_CANDIDATE_DB, cls.dataset_dir, limit_state_keys=400)

        import argparse
        from tools.train_human_blunder_net import train
        train_args = argparse.Namespace(
            dataset_dir=cls.dataset_dir,
            output=cls.model,
            epochs=1, patience=0, lr=3e-4, dropout=0.2,
            batch_positions=64, grad_clip=1.0, seed=42,
        )
        train(train_args)

        from tools.eval_human_blunder_net import evaluate
        cls.report_data = evaluate(cls.dataset_dir, cls.model, _CANDIDATE_DB,
                                   min_support=10)

    @classmethod
    def tearDownClass(cls):
        if cls.scratch.exists():
            shutil.rmtree(cls.scratch)

    def test_overall_has_all_metrics(self):
        overall = self.report_data["overall"]
        for k in ("n_events", "event_nll", "top1", "top3", "top5", "ece"):
            self.assertIn(k, overall)
        self.assertGreater(overall["n_events"], 0)
        # NLL is bounded below by 0 (probs ≤ 1 → -log ≥ 0).
        self.assertGreaterEqual(overall["event_nll"], 0.0)
        self.assertLessEqual(overall["top1"], 1.0)
        self.assertLessEqual(overall["top3"], 1.0)
        self.assertLessEqual(overall["top5"], 1.0)

    def test_all_three_bands_reported(self):
        by_band = self.report_data["by_band"]
        for b in ("lower", "middle", "upper"):
            self.assertIn(b, by_band)

    def test_both_phases_reported(self):
        by_phase = self.report_data["by_phase"]
        # place OR move must both appear across a 400-state_key slice.
        for p in ("place", "move"):
            self.assertIn(p, by_phase)

    def test_transitions_include_expected_categories(self):
        by_trans = self.report_data["by_transition"]
        # At minimum the classifier must have produced draw_preserved
        # events from this corpus.
        self.assertIn("draw_preserved", by_trans)

    def test_provenance_passthrough(self):
        # Advisor's provenance must survive into the eval report.
        prov = self.report_data["provenance"]
        self.assertIsNotNone(prov)
        self.assertIn("training_objective", prov)

    def test_empirical_kl_supported(self):
        ek = self.report_data["empirical_kl_supported"]
        self.assertIn("mean_kl", ek)
        self.assertIn("n_samples", ek)
        # KL is non-negative.
        self.assertGreaterEqual(ek["mean_kl"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
