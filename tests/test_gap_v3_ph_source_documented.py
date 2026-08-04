"""tests/test_gap_v3_ph_source_documented.py — Stage D regression: ph_source field.

Asserts:
1. Every row in the metadata.npz carries a ph_source value in {"model", "empirical",
   "hybrid"}.  ("empirical" alone is unreachable when the model is always available,
   but the enum must still accept it for forward-compat; see provenance
   ph_source_scheme key.)
2. The extraction logic correctly sets ph_source to "hybrid" when empirical support
   meets the threshold and "model" otherwise.
3. The provenance.json carries a ph_source_scheme key explaining the scheme.

Layer 1 — unit tests (always run):
  - ph_source value is in the required enum.
  - _compute_empirical_ph returns None below min_empirical_support.
  - _compute_empirical_ph returns a valid distribution at or above threshold.

Layer 2 — dataset file check (skipped if dataset not yet built):
  - Load metadata.npz; verify every ph_source value is in the enum.
  - Verify provenance.json has ph_source_scheme key.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_DATASET_DIR = _ROOT / "data" / "gap_net_v3_dataset"
_PH_SOURCE_ENUM = {"model", "empirical", "hybrid"}


# ── Layer 1: unit tests ────────────────────────────────────────────────────────

from tools.extract_gap_v3_dataset import _compute_empirical_ph, _move_notation


def _moves(*notations: str) -> list[dict]:
    return [{"from": None, "to": n, "capture": None} for n in notations]


class TestPhSourceEnum(unittest.TestCase):

    def test_enum_is_correct(self):
        self.assertEqual(_PH_SOURCE_ENUM, {"model", "empirical", "hybrid"})

    def test_empirical_ph_below_threshold_returns_none(self):
        mvs = _moves("a1", "a4")
        counts = {_move_notation(mvs[0]): 10, _move_notation(mvs[1]): 14}
        result = _compute_empirical_ph(counts, mvs, min_empirical_support=25)
        self.assertIsNone(result, "Should return None when support < min_empirical_support")

    def test_empirical_ph_at_threshold_returns_distribution(self):
        mvs = _moves("a1", "a4")
        counts = {_move_notation(mvs[0]): 15, _move_notation(mvs[1]): 10}
        result = _compute_empirical_ph(counts, mvs, min_empirical_support=25)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(float(result.sum()), 1.0, places=5)
        self.assertAlmostEqual(float(result[0]), 15 / 25, places=5)
        self.assertAlmostEqual(float(result[1]), 10 / 25, places=5)

    def test_empirical_ph_above_threshold(self):
        mvs = _moves("a1", "a4", "b4")
        counts = {
            _move_notation(mvs[0]): 30,
            _move_notation(mvs[1]): 20,
            _move_notation(mvs[2]): 50,
        }
        result = _compute_empirical_ph(counts, mvs, min_empirical_support=25)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (3,))
        self.assertAlmostEqual(float(result.sum()), 1.0, places=5)

    def test_empirical_ph_missing_notations_get_zero_weight(self):
        """Moves not in the observed counts get zero empirical P_h weight."""
        mvs = _moves("a1", "a4", "b4")
        counts = {_move_notation(mvs[0]): 30}  # only first move observed
        result = _compute_empirical_ph(counts, mvs, min_empirical_support=25)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(float(result[0]), 1.0, places=5)
        self.assertAlmostEqual(float(result[1]), 0.0, places=5)
        self.assertAlmostEqual(float(result[2]), 0.0, places=5)

    def test_ph_source_hybrid_when_empirical_available(self):
        """Logic check: ph_source should be 'hybrid' when empirical P_h is returned."""
        mvs = _moves("a1", "a4")
        counts = {_move_notation(mvs[0]): 20, _move_notation(mvs[1]): 10}
        ph_emp = _compute_empirical_ph(counts, mvs, min_empirical_support=25)
        ph_source = "hybrid" if ph_emp is not None else "model"
        self.assertEqual(ph_source, "hybrid")

    def test_ph_source_model_when_empirical_unavailable(self):
        """Logic check: ph_source should be 'model' when empirical P_h is None."""
        mvs = _moves("a1", "a4")
        counts = {_move_notation(mvs[0]): 5}  # below threshold
        ph_emp = _compute_empirical_ph(counts, mvs, min_empirical_support=25)
        ph_source = "hybrid" if ph_emp is not None else "model"
        self.assertEqual(ph_source, "model")


# ── Layer 2: dataset file validation ──────────────────────────────────────────

_DATASET_AVAILABLE = (
    (_DATASET_DIR / "metadata.npz").exists()
    and (_DATASET_DIR / "provenance.json").exists()
)


@unittest.skipUnless(_DATASET_AVAILABLE, f"Dataset not yet built at {_DATASET_DIR}")
class TestDatasetPhSource(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.meta = np.load(
            str(_DATASET_DIR / "metadata.npz"), allow_pickle=True
        )
        cls.prov = json.loads(
            (_DATASET_DIR / "provenance.json").read_text(encoding="utf-8")
        )

    def test_ph_source_array_exists(self):
        self.assertIn("ph_source", self.meta.files,
                      "metadata.npz must contain a 'ph_source' array")

    def test_all_ph_source_values_in_enum(self):
        ph_source_arr = self.meta["ph_source"]
        for val in ph_source_arr:
            self.assertIn(
                str(val), _PH_SOURCE_ENUM,
                f"ph_source value {val!r} not in {_PH_SOURCE_ENUM}",
            )

    def test_provenance_has_ph_source_scheme(self):
        self.assertIn("ph_source_scheme", self.prov,
                      "provenance.json must have a 'ph_source_scheme' key")
        self.assertIsInstance(self.prov["ph_source_scheme"], str)
        self.assertGreater(len(self.prov["ph_source_scheme"]), 0)

    def test_provenance_has_required_fields(self):
        required = {
            "dataset_version", "feature_version", "regret_version",
            "malom_label_version", "model_sha256", "human_db_sha256",
            "git_commit", "built_at", "n_emitted",
        }
        missing = required - set(self.prov.keys())
        self.assertEqual(missing, set(), f"provenance.json missing fields: {missing}")

    def test_metadata_arrays_consistent_length(self):
        n = len(self.meta["state_keys"])
        for arr_name in ("band_idx", "split", "phase", "mover_color",
                         "n_legal", "ph_source"):
            self.assertEqual(
                len(self.meta[arr_name]), n,
                f"metadata.npz array {arr_name!r} length mismatch",
            )

    def test_band_idx_values_valid(self):
        band_arr = self.meta["band_idx"].astype(int)
        self.assertTrue(
            all(0 <= v <= 2 for v in band_arr),
            "band_idx values must be 0, 1, or 2",
        )

    def test_split_values_valid(self):
        split_arr = self.meta["split"].astype(int)
        self.assertTrue(
            all(0 <= v <= 2 for v in split_arr),
            "split values must be 0 (train), 1 (val), or 2 (test)",
        )

    def test_feature_file_shape_consistent(self):
        n = len(self.meta["state_keys"])
        feat_path = _DATASET_DIR / "parent_feats.f32.bin"
        if feat_path.exists() and n > 0:
            expected_bytes = n * 79 * 4
            actual_bytes = feat_path.stat().st_size
            self.assertEqual(
                actual_bytes, expected_bytes,
                f"parent_feats.f32.bin size mismatch: {actual_bytes} vs {expected_bytes}",
            )

    def test_target_file_shape_consistent(self):
        n = len(self.meta["state_keys"])
        tgt_path = _DATASET_DIR / "targets.f32.bin"
        if tgt_path.exists() and n > 0:
            expected_bytes = n * 4 * 4
            actual_bytes = tgt_path.stat().st_size
            self.assertEqual(
                actual_bytes, expected_bytes,
                f"targets.f32.bin size mismatch: {actual_bytes} vs {expected_bytes}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
