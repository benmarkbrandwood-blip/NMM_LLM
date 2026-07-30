"""tests/test_human_db_split_v2.py — v2 additions to human_db_split.py.

Tests `three_way_split` and `game_level_split` added in MANIFEST_VERSION v2.
Does NOT alter the existing `in_val_bucket` / `partition` contract (other
consumers rely on those being unchanged).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.data.human_db_split import (  # noqa: E402
    MANIFEST_VERSION,
    three_way_split,
    game_level_split,
    state_key_bucket,
    in_val_bucket,
)


class TestManifestVersion(unittest.TestCase):
    def test_version_is_v2(self):
        self.assertEqual(MANIFEST_VERSION, "v2")


class TestThreeWaySplit(unittest.TestCase):

    def test_returns_valid_label(self):
        for sk in ["abc", "xyz", "pos123", "a1b2c3d4"]:
            label = three_way_split(sk)
            self.assertIn(label, ("train", "val", "test"),
                          f"Unexpected label {label!r} for {sk!r}")

    def test_bucket_boundaries(self):
        # Build synthetic state_keys that fall in each bucket range.
        # We can verify by checking state_key_bucket() directly.
        results = {"train": 0, "val": 0, "test": 0}
        import hashlib
        # Generate keys until we find at least one in each tier.
        for i in range(10000):
            sk = f"synthetic_{i}"
            b = state_key_bucket(sk)
            label = three_way_split(sk)
            if b < 5:
                self.assertEqual(label, "test",
                                 f"bucket={b} should be test, got {label!r}")
            elif b < 20:
                self.assertEqual(label, "val",
                                 f"bucket={b} should be val, got {label!r}")
            else:
                self.assertEqual(label, "train",
                                 f"bucket={b} should be train, got {label!r}")
            results[label] += 1

        for tier in ("train", "val", "test"):
            self.assertGreater(results[tier], 0, f"No keys landed in {tier!r} tier")

    def test_approximately_correct_fractions(self):
        # Over 10 000 keys the fractions should be within 5 % of 5/15/80.
        counts = {"train": 0, "val": 0, "test": 0}
        n = 10_000
        for i in range(n):
            counts[three_way_split(f"key_{i}")] += 1
        self.assertAlmostEqual(counts["test"]  / n, 0.05, delta=0.03)
        self.assertAlmostEqual(counts["val"]   / n, 0.15, delta=0.04)
        self.assertAlmostEqual(counts["train"] / n, 0.80, delta=0.04)

    def test_deterministic(self):
        sk = "some_position_key"
        r1 = three_way_split(sk)
        r2 = three_way_split(sk)
        self.assertEqual(r1, r2)

    def test_backward_compat_with_in_val_bucket(self):
        # Any key that three_way_split puts in "test" or "val" must also
        # satisfy in_val_bucket (val_fraction=0.20), because the old val
        # slice was buckets 0..19 — a superset of test (0..4) + val (5..19).
        for i in range(2000):
            sk = f"compat_key_{i}"
            label = three_way_split(sk)
            is_val = in_val_bucket(sk, val_fraction=0.20)
            if label in ("test", "val"):
                self.assertTrue(is_val,
                                f"key={sk!r}: label={label!r} but in_val_bucket=False")
            else:
                self.assertFalse(is_val,
                                 f"key={sk!r}: label=train but in_val_bucket=True")


class TestGameLevelSplit(unittest.TestCase):

    def test_returns_valid_label(self):
        for sid in ["ml11732493", "ml12345678", "game_001", "session_xyz"]:
            label = game_level_split(sid)
            self.assertIn(label, ("train", "val", "test"),
                          f"Unexpected label {label!r} for session_id {sid!r}")

    def test_deterministic(self):
        sid = "ml11732493"
        self.assertEqual(game_level_split(sid), game_level_split(sid))

    def test_different_sessions_can_differ(self):
        labels = {game_level_split(f"session_{i}") for i in range(200)}
        # Should hit at least two tiers in 200 sessions.
        self.assertGreater(len(labels), 1)

    def test_approximately_correct_fractions(self):
        counts = {"train": 0, "val": 0, "test": 0}
        n = 5_000
        for i in range(n):
            counts[game_level_split(f"ml_{i}")] += 1
        self.assertAlmostEqual(counts["test"]  / n, 0.05, delta=0.04)
        self.assertAlmostEqual(counts["val"]   / n, 0.15, delta=0.05)
        self.assertAlmostEqual(counts["train"] / n, 0.80, delta=0.05)

    def test_same_hash_space_as_state_key(self):
        # game_level_split uses SHA-256 % 100 like three_way_split —
        # confirm the bucket assignment matches the same boundary logic.
        import hashlib
        sid = "ml11732493"
        h = hashlib.sha256(sid.encode("utf-8")).digest()
        b = int.from_bytes(h[:4], "big") % 100
        expected = "test" if b < 5 else ("val" if b < 20 else "train")
        self.assertEqual(game_level_split(sid), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
