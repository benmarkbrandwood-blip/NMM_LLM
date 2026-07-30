"""tests/test_human_move_policy_trainer.py

End-to-end smoke: extract a tiny slice → train 2 epochs → verify the
.npz artefact is well-formed with the expected shape, provenance, and
inference-time properties (each layer's shape matches the plan).
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

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


_CANDIDATE_DB = _ROOT / "data" / "human_db_candidate.sqlite"
_CANDIDATE_AVAILABLE = _CANDIDATE_DB.exists()


@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch not available")
@unittest.skipUnless(_CANDIDATE_AVAILABLE, "candidate DB not built")
class TestTrainerEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Isolate all artefacts under a scratch directory that we
        # delete afterwards so the test doesn't accumulate state.
        cls.scratch = _ROOT / "data" / "_test_hbn_train"
        if cls.scratch.exists():
            shutil.rmtree(cls.scratch)
        cls.scratch.mkdir(parents=True)
        cls.dataset_dir = cls.scratch / "dataset"
        cls.output_npz  = cls.scratch / "hbn.npz"

        # Extract a small slice.
        sys.path.insert(0, str(_ROOT / "tools"))
        import extract_human_move_policy_dataset as ext
        ext.extract(_CANDIDATE_DB, cls.dataset_dir, limit_state_keys=200)

        # Train for 2 epochs.
        import argparse
        from tools.train_human_move_policy_net import train
        args = argparse.Namespace(
            dataset_dir=cls.dataset_dir,
            output=cls.output_npz,
            epochs=2, patience=0, lr=3e-4, dropout=0.2,
            batch_positions=64, grad_clip=1.0, seed=42,
        )
        cls.provenance = train(args)

    @classmethod
    def tearDownClass(cls):
        if cls.scratch.exists():
            shutil.rmtree(cls.scratch)

    def test_output_created(self):
        self.assertTrue(self.output_npz.exists())

    def test_npz_layout(self):
        d = np.load(self.output_npz, allow_pickle=True)
        for k in ("w0", "b0", "w1", "b1", "w2", "b2", "w3", "b3",
                  "input_dim", "layer_count", "board_feature_dim",
                  "n_bands", "provenance_json"):
            self.assertIn(k, d.files, f"missing key {k!r} in npz")
        self.assertEqual(int(d["layer_count"][0]), 4)
        # Feature dim = board (79) + 3 band bits = 82
        self.assertEqual(int(d["input_dim"][0]),         82)
        self.assertEqual(int(d["board_feature_dim"][0]), 79)
        self.assertEqual(int(d["n_bands"][0]),            3)
        # Plan §3.5: 82 → 128 → 64 → 32 → 1.
        self.assertEqual(d["w0"].shape, (128, 82))
        self.assertEqual(d["w1"].shape, (64, 128))
        self.assertEqual(d["w2"].shape, (32, 64))
        self.assertEqual(d["w3"].shape, (1, 32))

    def test_provenance_recorded(self):
        d = np.load(self.output_npz, allow_pickle=True)
        p = json.loads(str(d["provenance_json"].item()))
        for k in ("trainer_version", "training_objective",
                  "elo_band_config_name", "hparams",
                  "best_val_event_nll", "final_epochs_run",
                  "dataset_provenance"):
            self.assertIn(k, p, f"provenance missing {k!r}")
        self.assertEqual(p["training_objective"], "count_weighted_ce")
        self.assertIn("candidate_db_sha256", p["dataset_provenance"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
