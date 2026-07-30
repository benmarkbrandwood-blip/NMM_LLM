"""tests/test_human_move_policy_dataset_extractor.py

End-to-end tests for tools/extract_human_move_policy_dataset.py.  Skipped if
the v3 candidate DB isn't present.  Uses --limit-state-keys to keep
each test under 5 seconds.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

from ai.value_net import _INPUT_DIM  # noqa: E402
from learned_ai.data.elo_binning import OPTION_A_NAME  # noqa: E402
from learned_ai.data.human_db_split import in_val_bucket, three_way_split  # noqa: E402
import extract_human_move_policy_dataset as ext  # noqa: E402


_CANDIDATE_DB = _ROOT / "data" / "human_db_candidate.sqlite"
_CANDIDATE_AVAILABLE = _CANDIDATE_DB.exists()


@unittest.skipUnless(_CANDIDATE_AVAILABLE,
                     f"Candidate DB not found at {_CANDIDATE_DB}")
class TestExtractorEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.out_dir = _ROOT / "data" / "_test_hbn_dataset"
        if cls.out_dir.exists():
            shutil.rmtree(cls.out_dir)
        cls.provenance = ext.extract(
            _CANDIDATE_DB, cls.out_dir,
            limit_state_keys=200,
            val_fraction=0.20,
        )

    @classmethod
    def tearDownClass(cls):
        # Leave artefacts on disk for debugging; extractor is idempotent.
        pass

    def _load_meta(self) -> dict:
        return dict(np.load(self.out_dir / "metadata.npz", allow_pickle=True))

    def _load_memmap(self, n_rows: int) -> np.memmap:
        return np.memmap(
            self.out_dir / "succ_feats.f32.bin",
            dtype=np.float32, mode="r",
            shape=(n_rows, _INPUT_DIM),
        )

    def test_files_created(self):
        self.assertTrue((self.out_dir / "metadata.npz").exists())
        self.assertTrue((self.out_dir / "succ_feats.f32.bin").exists())

    def test_provenance_recorded(self):
        required = {
            "extract_version", "candidate_db_sha256", "feature_dim",
            "feature_version", "elo_band_config_name", "split_manifest_version",
            "val_fraction", "extract_git_commit",
            "successor_bank_bin", "successor_bank_rows",
            "n_state_keys", "n_samples", "n_samples_by_band",
            "n_samples_train", "n_samples_val", "n_samples_test",
        }
        missing = required - set(self.provenance.keys())
        self.assertFalse(missing, f"provenance missing: {missing}")
        self.assertEqual(self.provenance["elo_band_config_name"], OPTION_A_NAME)
        self.assertEqual(int(self.provenance["feature_dim"]), _INPUT_DIM)
        self.assertEqual(self.provenance["extract_version"], "2")

    def test_memmap_shape_matches_provenance(self):
        n = int(self.provenance["successor_bank_rows"])
        mm = self._load_memmap(n)
        self.assertEqual(mm.shape, (n, _INPUT_DIM))
        # Non-degenerate: some encodings are non-zero.
        self.assertGreater(float(np.abs(mm).sum()), 0.0)

    def test_sample_offsets_align_with_state_offsets(self):
        meta = self._load_meta()
        n_samples = int(meta["sample_state_idx"].shape[0])
        sample_offsets = meta["sample_offsets"]
        state_succ_offsets = meta["state_succ_offsets"]
        sample_state_idx  = meta["sample_state_idx"]
        for i in range(min(20, n_samples)):
            state_idx = int(sample_state_idx[i])
            state_legal = int(state_succ_offsets[state_idx + 1] - state_succ_offsets[state_idx])
            sample_legal = int(sample_offsets[i + 1] - sample_offsets[i])
            self.assertEqual(
                state_legal, sample_legal,
                f"sample {i}: state legal-count {state_legal} vs sample legal-count {sample_legal}",
            )

    def test_targets_are_non_negative(self):
        meta = self._load_meta()
        targets = meta["sample_targets"]
        self.assertGreater(int(targets.sum()), 0)
        self.assertEqual(int((targets < 0).sum()), 0)

    def test_split_flag_matches_three_way_split(self):
        meta = self._load_meta()
        state_keys       = meta["state_keys"]
        sample_state_idx = meta["sample_state_idx"]
        sample_split     = meta["sample_split"]
        sample_is_val    = meta["sample_is_val"]

        _split_int = {"train": 0, "val": 1, "test": 2}
        for i in range(min(50, len(sample_split))):
            sk = str(state_keys[int(sample_state_idx[i])])
            expected_int = _split_int[three_way_split(sk)]
            self.assertEqual(
                int(sample_split[i]), expected_int,
                f"sample {i}: sample_split={sample_split[i]} != {expected_int} for sk={sk!r}",
            )
            # sample_is_val must be False only for train (int8==0).
            self.assertEqual(
                bool(sample_is_val[i]),
                int(sample_split[i]) != 0,
                f"sample {i}: sample_is_val backward-compat mismatch",
            )

    def test_split_counts_sum_to_n_samples(self):
        meta = self._load_meta()
        sample_split = meta["sample_split"]
        n_train = int((sample_split == 0).sum())
        n_val   = int((sample_split == 1).sum())
        n_test  = int((sample_split == 2).sum())
        self.assertEqual(n_train + n_val + n_test, len(sample_split))
        # All three tiers should be non-empty in a 200-state_key slice.
        self.assertGreater(n_train, 0)
        self.assertGreater(n_val, 0)
        self.assertGreater(n_test, 0)

    def test_all_bands_populated(self):
        n_by_band = json.loads(self.provenance["n_samples_by_band"])
        for band in ("lower", "middle", "upper"):
            self.assertGreater(
                int(n_by_band[band]), 0,
                f"band {band} has no samples in the 200-state_key smoke slice",
            )

    def test_band_indices_valid_range(self):
        meta = self._load_meta()
        b = meta["sample_band_idx"]
        self.assertGreaterEqual(int(b.min()), 0)
        self.assertLessEqual(int(b.max()), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
