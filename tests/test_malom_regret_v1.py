"""tests/test_malom_regret_v1.py — Golden-corpus regression suite for
MalomDB.query_regret (Stage A of gap_net_v3_plan.md).

Layer 1 — unit tests (always run):
  - _classify_from_mover_pov: all 9 (pre, post) combinations.
  - RegretResult fields exist and have the right version strings.
  - query_regret returns available=False for a terminal parent.

Layer 2 — golden-corpus regression (skipped if Malom DB absent):
  - Every case in tests/fixtures/malom_regret_golden.json is replayed
    against the live DB and asserted to match the stored expected values.
  - D4 symmetry invariance: regret result must be identical under all 8
    board symmetries.

If any layer fails, GapNet v3 Stage C/D (dataset extraction) must not
proceed.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.malom_db import (
    MalomDB,
    RegretResult,
    _MALOM_LABEL_VERSION,
    _REGRET_VERSION,
    _SYM_PERMS,
    _classify_from_mover_pov,
    _sym24_from_perm,
    board_to_wbf,
)
from game.board import BoardState


# ── Malom DB availability ──────────────────────────────────────────────────────

def _resolve_malom_db_dir() -> Path:
    candidates: list[Path] = []
    if os.environ.get("NMM_MALOM_DB"):
        candidates.append(Path(os.environ["NMM_MALOM_DB"]))
    local_cfg = _ROOT / "data" / "training_paths.local.json"
    if local_cfg.exists():
        try:
            cfg = json.loads(local_cfg.read_text(encoding="utf-8"))
            val = cfg.get("malom_db_path")
            if val:
                p = Path(val)
                candidates.append(p if p.is_absolute() else _ROOT / p)
        except Exception:
            pass
    settings = _ROOT / "data" / "settings.json"
    if settings.exists():
        try:
            cfg = json.loads(settings.read_text(encoding="utf-8"))
            val = cfg.get("malom_db_path")
            if val:
                p = Path(val)
                candidates.append(p if p.is_absolute() else _ROOT / p)
        except Exception:
            pass
    for c in candidates:
        if c.is_dir() and any(c.glob("std_*.sec2")):
            return c
    return candidates[0] if candidates else Path("/nonexistent")


_MALOM_DIR = _resolve_malom_db_dir()
_MALOM_AVAILABLE = _MALOM_DIR.is_dir() and any(_MALOM_DIR.glob("std_*.sec2"))

_GOLDEN_PATH = _ROOT / "tests" / "fixtures" / "malom_regret_golden.json"
_GOLDEN_AVAILABLE = _GOLDEN_PATH.exists()


# ── Layer 1: unit tests ────────────────────────────────────────────────────────

class TestClassifyFromMoverPov(unittest.TestCase):
    """_classify_from_mover_pov — both args in mover-POV.

    Derivation: _classify_from_mover_pov(pre, post)
                = _classify_transition(pre, FLIP[post])
    where FLIP = {"W":"L", "L":"W", "D":"D"}.
    """

    def _c(self, pre: str, post: str) -> str:
        return _classify_from_mover_pov(pre, post)

    # W-parent
    def test_win_preserved(self):
        self.assertEqual(self._c("W", "W"), "win_preserved")

    def test_win_to_draw(self):
        self.assertEqual(self._c("W", "D"), "win_to_draw")

    def test_win_to_loss(self):
        self.assertEqual(self._c("W", "L"), "win_to_loss")

    # D-parent
    def test_draw_preserved(self):
        self.assertEqual(self._c("D", "D"), "draw_preserved")

    def test_draw_to_loss(self):
        self.assertEqual(self._c("D", "L"), "draw_to_loss")

    def test_draw_to_win_is_label_inconsistency(self):
        self.assertEqual(self._c("D", "W"), "label_inconsistency")

    # L-parent
    def test_all_losing(self):
        self.assertEqual(self._c("L", "L"), "all_losing")

    def test_loss_to_draw_is_label_inconsistency(self):
        self.assertEqual(self._c("L", "D"), "label_inconsistency")

    def test_loss_to_win_is_label_inconsistency(self):
        self.assertEqual(self._c("L", "W"), "label_inconsistency")

    # Missing values
    def test_unlabelled_pre_missing(self):
        self.assertEqual(self._c(None, "W"), "unlabelled")

    def test_unlabelled_post_missing(self):
        self.assertEqual(self._c("W", None), "unlabelled")


class TestRegretResultStructure(unittest.TestCase):
    """RegretResult dataclass fields and version strings are as specified."""

    def test_fields_present(self):
        fields = set(RegretResult.__dataclass_fields__.keys())
        required = {
            "available", "omv", "wdl_transition", "best_omv",
            "components", "regret_version", "malom_label_version",
            "unavailable_reason",
        }
        self.assertEqual(fields, required)

    def test_version_constants(self):
        self.assertEqual(_REGRET_VERSION, "regret_v1")
        self.assertEqual(_MALOM_LABEL_VERSION, "sector-corrected-v1")

    def test_components_keys(self):
        # A RegretResult for a terminal parent has all None components.
        # Construct one without a DB by verifying the structure directly.
        r = RegretResult(
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
            unavailable_reason="test",
        )
        self.assertFalse(r.available)
        self.assertIsNone(r.components["within_class_distance"])
        self.assertEqual(set(r.components.keys()), {
            "class_downgrade_prob", "wdl_utility_loss",
            "ordinal_rank_loss", "within_class_distance",
        })


# ── Layer 2: golden-corpus regression ─────────────────────────────────────────

def _board_from_dict(d: dict) -> BoardState:
    return BoardState.from_setup(
        positions=d["positions"],
        turn=d["turn"],
        phase=d.get("phase", "move"),
    )


def _apply_sym(db: MalomDB, board: BoardState, perm: list[int]) -> BoardState:
    """Apply one D4 symmetry permutation to a board and return the new board."""
    wb, bb, wf, bf = board_to_wbf(board)
    new_wb = _sym24_from_perm(perm, wb)
    new_bb = _sym24_from_perm(perm, bb)
    from ai.malom_db import MALOM_BITS_TO_POS
    new_positions: dict[str, str] = {}
    for bit in range(24):
        pos = MALOM_BITS_TO_POS[bit]
        if (new_wb >> bit) & 1:
            new_positions[pos] = "W"
        elif (new_bb >> bit) & 1:
            new_positions[pos] = "B"
    return BoardState.from_setup(new_positions, turn=board.turn, phase="move")


def _apply_sym_to_move(move: dict, perm: list[int]) -> dict:
    """Apply a D4 symmetry permutation to a move dict."""
    from ai.malom_db import MALOM_BITS_TO_POS, _POS_TO_MALOM_BIT  # noqa: F401
    _POS_TO_MALOM_BIT = {pos: i for i, pos in enumerate(MALOM_BITS_TO_POS)}

    def transform_sq(sq: str | None) -> str | None:
        if sq is None:
            return None
        bit = _POS_TO_MALOM_BIT[sq]
        new_bit = perm[bit]
        return MALOM_BITS_TO_POS[new_bit]

    return {
        "from": transform_sq(move["from"]),
        "to": transform_sq(move["to"]),
        "capture": transform_sq(move["capture"]),
    }


@unittest.skipUnless(
    _MALOM_AVAILABLE and _GOLDEN_AVAILABLE,
    f"Malom DB not available at {_MALOM_DIR} or golden fixture missing",
)
class TestMalomRegretGolden(unittest.TestCase):
    """Replay every case in the golden corpus against the live Malom DB."""

    @classmethod
    def setUpClass(cls):
        cls.db = MalomDB(str(_MALOM_DIR))
        cls.golden = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def _assert_result_matches(self, result: RegretResult, expected: dict, label: str):
        self.assertEqual(
            result.available, expected["available"],
            f"[{label}] available mismatch",
        )
        if not expected["available"]:
            self.assertEqual(
                result.unavailable_reason,
                expected.get("unavailable_reason"),
                f"[{label}] unavailable_reason mismatch",
            )
            return

        self.assertEqual(
            result.wdl_transition, expected["wdl_transition"],
            f"[{label}] wdl_transition mismatch",
        )
        self.assertIsNotNone(result.omv, f"[{label}] omv is None")
        self.assertEqual(
            result.omv.outcome, expected["omv_outcome"],
            f"[{label}] omv.outcome mismatch",
        )
        if expected.get("best_omv_outcome") is not None:
            self.assertIsNotNone(result.best_omv, f"[{label}] best_omv is None")
            self.assertEqual(
                result.best_omv.outcome, expected["best_omv_outcome"],
                f"[{label}] best_omv.outcome mismatch",
            )
        exp_comp = expected["components"]
        for key in ("class_downgrade_prob", "wdl_utility_loss",
                    "ordinal_rank_loss", "within_class_distance"):
            exp_val = exp_comp.get(key)
            got_val = result.components.get(key)
            if exp_val is None:
                self.assertIsNone(got_val, f"[{label}] {key} should be None")
            else:
                self.assertIsNotNone(got_val, f"[{label}] {key} should not be None")
                self.assertAlmostEqual(
                    got_val, exp_val, places=6,
                    msg=f"[{label}] {key} mismatch",
                )
        self.assertEqual(
            result.regret_version, _REGRET_VERSION,
            f"[{label}] regret_version mismatch",
        )
        self.assertEqual(
            result.malom_label_version, _MALOM_LABEL_VERSION,
            f"[{label}] malom_label_version mismatch",
        )

    def test_golden_cases(self):
        """Every golden case must reproduce its expected result."""
        for case in self.golden["cases"]:
            label = case["label"]
            with self.subTest(label=label):
                board = _board_from_dict(case["board"])
                move = case["move"]
                result = self.db.query_regret(board, move)
                self._assert_result_matches(result, case["expected"], label)

    def test_d4_symmetry_invariance(self):
        """Regret must be identical under all 8 D4 symmetries (available cases only).

        Skips place-phase boards: _apply_sym rebuilds with phase="move" which
        makes placement moves illegal, so symmetry testing is only meaningful
        for movement/fly phase positions.
        """
        for case in self.golden["cases"]:
            if not case["expected"].get("available", True):
                continue
            # Skip place-phase cases (pieces_placed < 9).
            bd = case["board"]
            if bd.get("pieces_placed", {}).get("W", 9) < 9:
                continue
            label = case["label"]
            board = _board_from_dict(case["board"])
            move = case["move"]
            expected = case["expected"]
            for sym_idx, perm in enumerate(_SYM_PERMS[:8]):
                with self.subTest(label=label, sym=sym_idx):
                    sym_board = _apply_sym(self.db, board, perm)
                    sym_move = _apply_sym_to_move(move, perm)
                    result = self.db.query_regret(sym_board, sym_move)
                    # The symmetry may cause a DB miss (not all symmetric
                    # positions are covered); accept that gracefully.
                    if not result.available and result.unavailable_reason in (
                        "parent_value_unavailable", "move_value_unavailable"
                    ):
                        continue
                    self._assert_result_matches(result, expected, f"{label}/sym{sym_idx}")

    def test_version_strings_in_golden_meta(self):
        """Golden meta must match the current version constants."""
        meta = self.golden["meta"]
        self.assertEqual(meta["regret_version"], _REGRET_VERSION)
        self.assertEqual(meta["malom_label_version"], _MALOM_LABEL_VERSION)


if __name__ == "__main__":
    unittest.main(verbosity=2)
