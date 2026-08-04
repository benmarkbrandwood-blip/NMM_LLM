"""tests/test_gap_v3_dataset_faillclosed.py — Stage D regression: strict fail-closed.

Asserts that extract_gap_v3_dataset.py abstains from any (state_key, band) where:
  - query_regret returns available=False for any legal move
  - component A is None for any legal move (label_inconsistency / unlabelled)
  - component B or C is None for any legal move (best_omv unavailable)

Component D (within_class_distance) is excluded from the strict fail-closed check
per gap_net_v3_plan.md §1 line 11 (deferred; always None in regret_v1).

Tests use in-process mocks so no Malom DB is required.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.malom_db import RegretResult, _MALOM_LABEL_VERSION, _REGRET_VERSION
from tools.extract_gap_v3_dataset import (
    _check_emittable,
    _check_malom_eligible,
    _compute_gv,
)


def _mock_omv(outcome="W") -> MagicMock:
    omv = MagicMock()
    omv.outcome = outcome
    omv.ordering_key.return_value = (0, 0)
    return omv


def _rr_ok(outcome_omv="W") -> RegretResult:
    """A fully available RegretResult with non-None A/B/C."""
    omv = _mock_omv(outcome_omv)
    return RegretResult(
        available=True,
        omv=omv,
        wdl_transition="win_preserved",
        best_omv=omv,
        components={
            "class_downgrade_prob": 0.0,
            "wdl_utility_loss": 0.0,
            "ordinal_rank_loss": 0.0,
            "within_class_distance": None,
        },
        regret_version=_REGRET_VERSION,
        malom_label_version=_MALOM_LABEL_VERSION,
    )


def _rr_unavailable(reason: str = "move_value_unavailable") -> RegretResult:
    return RegretResult(
        available=False,
        omv=None,
        wdl_transition=None,
        best_omv=None,
        components={
            "class_downgrade_prob": None,
            "wdl_utility_loss": None,
            "ordinal_rank_loss": None,
            "within_class_distance": None,
        },
        regret_version=_REGRET_VERSION,
        malom_label_version=_MALOM_LABEL_VERSION,
        unavailable_reason=reason,
    )


def _rr_comp_a_none() -> RegretResult:
    """available=True but component A is None (label_inconsistency)."""
    omv = _mock_omv("W")
    return RegretResult(
        available=True,
        omv=omv,
        wdl_transition="label_inconsistency",
        best_omv=omv,
        components={
            "class_downgrade_prob": None,
            "wdl_utility_loss": 0.0,
            "ordinal_rank_loss": 0.0,
            "within_class_distance": None,
        },
        regret_version=_REGRET_VERSION,
        malom_label_version=_MALOM_LABEL_VERSION,
    )


def _rr_best_omv_none() -> RegretResult:
    """available=True but best_omv is None (some other child not covered).

    Comp A is computed from wdl_transition independent of best_omv;
    comp B/C are None because best_omv is required for them.
    """
    omv = _mock_omv("W")
    return RegretResult(
        available=True,
        omv=omv,
        wdl_transition="win_preserved",
        best_omv=None,
        components={
            "class_downgrade_prob": 0.0,   # A is set from wdl_transition regardless of best_omv
            "wdl_utility_loss": None,       # B requires best_omv
            "ordinal_rank_loss": None,      # C requires best_omv
            "within_class_distance": None,
        },
        regret_version=_REGRET_VERSION,
        malom_label_version=_MALOM_LABEL_VERSION,
    )


def _moves(*notations: str) -> list[dict]:
    return [{"from": None, "to": n, "capture": None} for n in notations]


def _regrets_map(moves: list[dict], results: list) -> dict:
    from tools.extract_gap_v3_dataset import _move_notation
    return {_move_notation(m): r for m, r in zip(moves, results)}


class TestCheckMalomEligible(unittest.TestCase):

    def test_all_ok(self):
        mvs = _moves("a1", "a4")
        reg = _regrets_map(mvs, [_rr_ok(), _rr_ok()])
        ok, reason = _check_malom_eligible(reg, mvs)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_one_unavailable(self):
        mvs = _moves("a1", "a4")
        reg = _regrets_map(mvs, [_rr_ok(), _rr_unavailable()])
        ok, reason = _check_malom_eligible(reg, mvs)
        self.assertFalse(ok)
        self.assertIn("move_unavailable", reason)

    def test_best_omv_none(self):
        mvs = _moves("a1", "a4")
        reg = _regrets_map(mvs, [_rr_ok(), _rr_best_omv_none()])
        ok, reason = _check_malom_eligible(reg, mvs)
        self.assertFalse(ok)
        self.assertEqual(reason, "best_omv_unavailable")


class TestCheckEmittable(unittest.TestCase):

    def test_all_ok_is_emittable(self):
        mvs = _moves("a1", "a4")
        reg = _regrets_map(mvs, [_rr_ok(), _rr_ok()])
        ok, reason = _check_emittable(reg, mvs)
        self.assertTrue(ok)

    def test_one_unavailable_is_not_emittable(self):
        mvs = _moves("a1", "a4")
        reg = _regrets_map(mvs, [_rr_ok(), _rr_unavailable()])
        ok, reason = _check_emittable(reg, mvs)
        self.assertFalse(ok)
        self.assertIn("move_unavailable", reason)

    def test_comp_a_none_is_not_emittable(self):
        mvs = _moves("a1", "a4")
        reg = _regrets_map(mvs, [_rr_ok(), _rr_comp_a_none()])
        ok, reason = _check_emittable(reg, mvs)
        self.assertFalse(ok)
        self.assertEqual(reason, "comp_a_none")

    def test_best_omv_none_is_not_emittable(self):
        """best_omv=None means comp B/C unavailable — strict fail-closed blocks emit."""
        mvs = _moves("a1", "a4")
        reg = _regrets_map(mvs, [_rr_ok(), _rr_best_omv_none()])
        ok, reason = _check_emittable(reg, mvs)
        self.assertFalse(ok)
        self.assertEqual(reason, "comp_bc_none")

    def test_comp_d_none_does_not_block_emit(self):
        """Component D=None must not block emission (deferred per plan §1 line 11)."""
        mvs = _moves("a1", "a4")
        rr1 = _rr_ok()
        rr2 = _rr_ok()
        # Explicitly set D to None (it already is, but be explicit)
        rr1.components["within_class_distance"] = None
        rr2.components["within_class_distance"] = None
        reg = _regrets_map(mvs, [rr1, rr2])
        ok, reason = _check_emittable(reg, mvs)
        self.assertTrue(ok, f"D=None should not block emit; reason={reason!r}")

    def test_two_ok_moves(self):
        mvs = _moves("a1", "a4", "b4")
        reg = _regrets_map(mvs, [_rr_ok(), _rr_ok(), _rr_ok()])
        ok, reason = _check_emittable(reg, mvs)
        self.assertTrue(ok)

    def test_mixed_ok_and_comp_a_none(self):
        """Even if only one move has A=None, the whole position must be abstained."""
        mvs = _moves("a1", "a4", "b4")
        reg = _regrets_map(mvs, [_rr_ok(), _rr_ok(), _rr_comp_a_none()])
        ok, reason = _check_emittable(reg, mvs)
        self.assertFalse(ok)
        self.assertEqual(reason, "comp_a_none")


class TestComputeGv(unittest.TestCase):

    def test_uniform_ph(self):
        """G_v = uniform P_h over 2 moves with A=0 and A=1 → G_v_A = 0.5."""
        import numpy as np
        mvs = _moves("a1", "a4")

        rr_downgrade = _rr_ok()
        rr_downgrade.components["class_downgrade_prob"] = 1.0
        rr_downgrade.components["wdl_utility_loss"] = 2.0
        rr_downgrade.components["ordinal_rank_loss"] = 0.5

        rr_preserve = _rr_ok()
        rr_preserve.components["class_downgrade_prob"] = 0.0
        rr_preserve.components["wdl_utility_loss"] = 0.0
        rr_preserve.components["ordinal_rank_loss"] = 0.0

        reg = _regrets_map(mvs, [rr_downgrade, rr_preserve])
        ph = np.array([0.5, 0.5], dtype=np.float32)
        g_v_A, g_v_B, g_v_C = _compute_gv(ph, mvs, reg)

        self.assertAlmostEqual(g_v_A, 0.5, places=5)
        self.assertAlmostEqual(g_v_B, 1.0, places=5)
        self.assertAlmostEqual(g_v_C, 0.25, places=5)

    def test_concentrated_ph(self):
        """All P_h on first move → G_v = first move's component values."""
        import numpy as np
        mvs = _moves("a1", "a4")

        rr0 = _rr_ok()
        rr0.components["class_downgrade_prob"] = 0.0
        rr0.components["wdl_utility_loss"] = 3.0
        rr0.components["ordinal_rank_loss"] = 0.75

        rr1 = _rr_ok()
        rr1.components["class_downgrade_prob"] = 1.0
        rr1.components["wdl_utility_loss"] = 0.0
        rr1.components["ordinal_rank_loss"] = 0.0

        reg = _regrets_map(mvs, [rr0, rr1])
        ph = np.array([1.0, 0.0], dtype=np.float32)
        g_v_A, g_v_B, g_v_C = _compute_gv(ph, mvs, reg)

        self.assertAlmostEqual(g_v_A, 0.0, places=5)
        self.assertAlmostEqual(g_v_B, 3.0, places=5)
        self.assertAlmostEqual(g_v_C, 0.75, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
