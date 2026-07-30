"""tests/test_human_move_policy_advisor.py

Unit + smoke tests for `ai/human_move_policy_advisor.py`.  Trains a tiny
HumanMovePolicyNet under a scratch dir, loads it via the advisor, and
asserts:
  - probs sums to 1 across all legal moves
  - probs shape matches legal-move count
  - conditioning on band actually changes the output
  - degenerate / missing-file paths fall back gracefully
"""
from __future__ import annotations

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


class TestTryLoadMissing(unittest.TestCase):
    def test_returns_none_for_missing_file(self):
        from ai.human_move_policy_advisor import try_load
        self.assertIsNone(try_load("/tmp/no_such_hbn.npz"))


@unittest.skipUnless(_TORCH_AVAILABLE and _CANDIDATE_AVAILABLE,
                     "Requires PyTorch + candidate DB")
class TestAdvisorEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scratch = _ROOT / "data" / "_test_hbn_advisor"
        if cls.scratch.exists():
            shutil.rmtree(cls.scratch)
        cls.scratch.mkdir(parents=True)
        cls.dataset_dir = cls.scratch / "dataset"
        cls.output_npz  = cls.scratch / "hbn.npz"

        sys.path.insert(0, str(_ROOT / "tools"))
        import extract_human_move_policy_dataset as ext
        ext.extract(_CANDIDATE_DB, cls.dataset_dir, limit_state_keys=200)

        import argparse
        from tools.train_human_move_policy_net import train
        args = argparse.Namespace(
            dataset_dir=cls.dataset_dir,
            output=cls.output_npz,
            epochs=1, patience=0, lr=3e-4, dropout=0.2,
            batch_positions=64, grad_clip=1.0, seed=42,
        )
        train(args)

    @classmethod
    def tearDownClass(cls):
        if cls.scratch.exists():
            shutil.rmtree(cls.scratch)

    def _fresh_board(self):
        from game.board import BoardState
        return BoardState.new_game()

    def _legal_moves(self, board):
        from game.rules import get_all_legal_moves
        return get_all_legal_moves(board)

    def test_load_and_infer(self):
        from ai.human_move_policy_advisor import HumanMovePolicyAdvisor
        adv = HumanMovePolicyAdvisor(self.output_npz)
        # 82 = 79 board features + 3 elo band bits
        self.assertEqual(adv.input_dim, 82)
        self.assertEqual(adv.board_feature_dim, 79)
        self.assertEqual(adv.n_bands, 3)
        self.assertIsNotNone(adv.provenance)

    def test_probs_sum_to_one_per_band(self):
        from ai.human_move_policy_advisor import HumanMovePolicyAdvisor
        adv = HumanMovePolicyAdvisor(self.output_npz)
        board = self._fresh_board()
        legal = self._legal_moves(board)
        self.assertGreater(len(legal), 0)
        for band in ("lower", "middle", "upper"):
            p = adv.probs(board, legal, band)
            self.assertEqual(p.shape, (len(legal),))
            self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
            # All non-negative.
            self.assertGreaterEqual(float(p.min()), 0.0)

    def test_band_changes_distribution(self):
        """Conditioning on band must produce different distributions
        (assuming training was non-trivial — 1 epoch on a real dataset
        will not saturate the band bits to zero effect)."""
        from ai.human_move_policy_advisor import HumanMovePolicyAdvisor
        adv = HumanMovePolicyAdvisor(self.output_npz)
        board = self._fresh_board()
        legal = self._legal_moves(board)
        p_lower  = adv.probs(board, legal, "lower")
        p_upper  = adv.probs(board, legal, "upper")
        # Any non-trivial band conditioning should produce different
        # distributions.  If bands were ignored, the two would match to
        # within numerical precision.
        max_diff = float(np.abs(p_lower - p_upper).max())
        self.assertGreater(
            max_diff, 1e-4,
            "band conditioning produced identical distributions — model "
            "is not consuming the elo band feature",
        )

    def test_rejects_unknown_band(self):
        from ai.human_move_policy_advisor import HumanMovePolicyAdvisor
        adv = HumanMovePolicyAdvisor(self.output_npz)
        board = self._fresh_board()
        legal = self._legal_moves(board)
        with self.assertRaises(ValueError):
            adv.probs(board, legal, "expert")

    def test_empty_legal_moves(self):
        from ai.human_move_policy_advisor import HumanMovePolicyAdvisor
        adv = HumanMovePolicyAdvisor(self.output_npz)
        board = self._fresh_board()
        p = adv.probs(board, [], "lower")
        self.assertEqual(p.shape, (0,))
        r = adv.rank(board, [], "lower")
        self.assertEqual(r, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
