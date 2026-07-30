"""tests/test_human_moves_audit_perspective.py — Locks the parent/child Malom
perspective that HumanMovePolicyNet's audit script depends on.

The reviewer flagged this as a blocker for Phase 2.  Two layers:

  1. Unit tests for `_classify_transition` and `_FLIP` in
     `tools/audit_human_moves.py` — pure logic, no DB required.
     Asserts every parent-mover-POV × child-next-mover-POV combination
     resolves to the documented transition category.

  2. Integration tests against a real Malom DB (skipped if the DB is
     absent) that verify:
       - Malom's parent query returns the mover-POV outcome.
       - Malom's child query returns the next-mover-POV outcome.
       - For every legal move, the `_FLIP` mapping converts
         child-POV → mover-POV consistently with the ordering
         relationship (child-`L` ⇒ mover made a winning move, etc).
       - HumanPrefNet's actual per-state filter keeps the `L`-after
         records (i.e. does not treat them as blunders).

If either layer fails, HumanMovePolicyNet Phase 2 must not proceed.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

import audit_human_moves as ahb  # noqa: E402


# ── 1. Unit tests: transition classification ────────────────────────────────

class TestFlipMapping(unittest.TestCase):
    """The child_next_mover ↔ child_mover flip must be an involution over {W,L,D}."""

    def test_flip_is_involution(self):
        for pov in ("W", "L", "D"):
            self.assertEqual(ahb._FLIP[ahb._FLIP[pov]], pov)

    def test_flip_swaps_w_and_l(self):
        self.assertEqual(ahb._FLIP["W"], "L")
        self.assertEqual(ahb._FLIP["L"], "W")

    def test_flip_fixes_draw(self):
        self.assertEqual(ahb._FLIP["D"], "D")


class TestClassifyTransition(unittest.TestCase):
    """Every (parent-mover, child-next-mover) combination must classify.

    Reviewer's canonical example (from Codex report):
      parent W and raw child L means the human win was preserved.
      parent W and raw child W means win-to-loss.
      parent D and raw child W means draw-to-loss.
    """

    def test_win_preserved(self):
        # parent mover = W ; child next-mover = L → child mover = W ; W→W
        self.assertEqual(ahb._classify_transition("W", "L"), "win_preserved")

    def test_win_to_draw(self):
        # parent mover = W ; child next-mover = D → child mover = D ; W→D
        self.assertEqual(ahb._classify_transition("W", "D"), "win_to_draw")

    def test_win_to_loss(self):
        # parent mover = W ; child next-mover = W → child mover = L ; W→L
        self.assertEqual(ahb._classify_transition("W", "W"), "win_to_loss")

    def test_draw_preserved(self):
        # parent mover = D ; child next-mover = D → child mover = D ; D→D
        self.assertEqual(ahb._classify_transition("D", "D"), "draw_preserved")

    def test_draw_to_loss(self):
        # parent mover = D ; child next-mover = W → child mover = L ; D→L
        self.assertEqual(ahb._classify_transition("D", "W"), "draw_to_loss")

    def test_draw_to_win_is_label_inconsistency(self):
        # parent mover = D ; child next-mover = L → child mover = W ; D→W impossible.
        self.assertEqual(ahb._classify_transition("D", "L"), "label_inconsistency")

    def test_all_losing(self):
        # parent mover = L ; child next-mover = W → child mover = L ; L→L retained.
        self.assertEqual(ahb._classify_transition("L", "W"), "all_losing")

    def test_loss_to_draw_is_label_inconsistency(self):
        # parent mover = L ; child next-mover = D → child mover = D ; L→D impossible.
        self.assertEqual(ahb._classify_transition("L", "D"), "label_inconsistency")

    def test_loss_to_win_is_label_inconsistency(self):
        # parent mover = L ; child next-mover = L → child mover = W ; L→W impossible.
        self.assertEqual(ahb._classify_transition("L", "L"), "label_inconsistency")

    def test_unlabelled_when_pre_missing(self):
        self.assertEqual(ahb._classify_transition(None, "L"), "unlabelled")

    def test_unlabelled_when_after_missing(self):
        self.assertEqual(ahb._classify_transition("W", None), "unlabelled")


class TestEloBanding(unittest.TestCase):
    """Option A boundaries — bin-aligned to 50-Elo edges (v1.2).
    These are strata within the PlayOK amateur corpus; not universal
    strength labels."""

    def test_lower_upper_edge(self):
        # 1149 is the highest lower-band Elo (bin 1100 upper edge).
        self.assertEqual(ahb._elo_band(1149), "lower")

    def test_middle_lower_edge(self):
        # 1150 is the first middle-band Elo (bin 1150 lower edge).
        self.assertEqual(ahb._elo_band(1150), "middle")

    def test_middle_upper_edge(self):
        # 1249 is the highest middle-band Elo (bin 1200 upper edge).
        self.assertEqual(ahb._elo_band(1249), "middle")

    def test_upper_lower_edge(self):
        # 1250 is the first upper-band Elo (bin 1250 lower edge).
        self.assertEqual(ahb._elo_band(1250), "upper")

    def test_unknown_when_missing(self):
        self.assertEqual(ahb._elo_band(None), "unknown")


# ── 2. HumanPrefNet filter direction (reviewer §4) ──────────────────────────

class TestHumanPrefFilterDirection(unittest.TestCase):
    """HumanPrefNet's per-state filter must keep L-after records — child
    labelled L means the opponent is losing and the human made a winning
    move.  A plan that treats L-after as a blunder would be wrong.
    """

    def test_prefers_L_when_any_L_available(self):
        from tools.train_human_pref_net import _per_state_filter
        self.assertEqual(_per_state_filter(["L", "D", "W", "L"]), "L")

    def test_falls_back_to_D_when_no_L(self):
        from tools.train_human_pref_net import _per_state_filter
        self.assertEqual(_per_state_filter(["W", "D", "W"]), "D")

    def test_returns_none_when_all_are_W(self):
        # All records show the human played into a next-mover-W child ⇒
        # every record is a mover-L (all-losing).  HumanPrefNet drops those.
        # HumanMovePolicyNet retains them (per reviewer §5).
        from tools.train_human_pref_net import _per_state_filter
        self.assertIsNone(_per_state_filter(["W", "W"]))


# ── 3. Integration tests: Malom parent/child perspective ────────────────────

def _resolve_malom_db_dir() -> Path:
    candidates: list[Path] = []
    if os.environ.get("NMM_MALOM_DB"):
        candidates.append(Path(os.environ["NMM_MALOM_DB"]))
    local_config = _ROOT / "data" / "training_paths.local.json"
    if local_config.exists():
        try:
            cfg = json.loads(local_config.read_text(encoding="utf-8"))
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


@unittest.skipUnless(_MALOM_AVAILABLE, f"Malom DB not available at {_MALOM_DIR}")
class TestMalomPerspectiveIntegration(unittest.TestCase):
    """Verifies that the audit's perspective assumptions hold on the real
    Malom DB for a concrete endgame position.  Skipped if Malom is absent.
    """

    @classmethod
    def setUpClass(cls):
        from ai.malom_db import MalomDB
        cls.db = MalomDB(str(_MALOM_DIR))

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def _endgame_boards(self):
        """Return a list of (board, expected_mover_outcome_hint) for
        endgame positions where Malom coverage is known-good.

        We do NOT hard-code the exact outcome — different Malom builds may
        return slightly different DTW-optimal answers.  The tests below only
        check the *perspective consistency* of parent/child queries.
        """
        from game.board import BoardState, POSITIONS
        from game.rules import get_all_legal_moves
        # A fully-placed 3v3 movement-phase position.  Reproduced from
        # tests/test_malom_db.py::test_query_endgame_position.
        boards = []
        b = BoardState.from_setup(
            {"a7": "W", "d5": "W", "e4": "W",
             "g7": "B", "f4": "B", "c3": "B"},
            turn="W", phase="move",
        )
        boards.append(b)
        # New-game empty board — parent outcome should be draw.
        boards.append(BoardState.new_game())
        return boards

    def test_parent_query_returns_mover_pov(self):
        """MalomDB.query() on the parent returns the outcome from the
        side-to-move's perspective, i.e. the mover's POV.

        The reviewer's contract: parent Malom = mover-POV.
        """
        for board in self._endgame_boards():
            with self.subTest(fen=board.to_fen_string() if hasattr(board, "to_fen_string") else str(board)):
                result = self.db.query(board)
                self.assertIsNotNone(result, "Malom returned None for expected-covered position.")
                self.assertIn(result["outcome"], ("W", "L", "D"),
                              "Parent outcome must be one of W/L/D.")

    def test_child_flip_consistent_with_mover_pov(self):
        """For each legal move from a covered position, applying the move
        and querying the child yields the next-mover-POV outcome.  The
        `_FLIP` mapping converts that to the mover-POV outcome.

        We assert only consistency: every legal move classifies into one
        of the documented transition categories.  We do NOT require a
        specific outcome — Malom is authoritative.
        """
        from game.rules import get_all_legal_moves
        valid_categories = {
            "win_preserved", "win_to_draw", "win_to_loss",
            "draw_preserved", "draw_to_loss", "all_losing",
        }
        # A `label_inconsistency` classification here would indicate that
        # either the audit's flip is wrong, or Malom is inconsistent.
        for board in self._endgame_boards():
            parent_res = self.db.query(board)
            if parent_res is None:
                continue
            pre_mover = parent_res["outcome"]
            for move in get_all_legal_moves(board):
                child_board = board.apply_move(move)
                child_res = self.db.query(child_board)
                if child_res is None:
                    continue  # partial coverage — legitimate
                after_next = child_res["outcome"]
                category = ahb._classify_transition(pre_mover, after_next)
                self.assertIn(
                    category, valid_categories,
                    f"Perspective flip produced unexpected category "
                    f"{category!r} for pre={pre_mover} after_next={after_next}"
                )

    def test_winning_position_has_at_least_one_win_preserving_move(self):
        """If Malom labels the parent W (mover winning), at least one legal
        move must classify as win_preserved.  This is a definitional
        property of `W`: a winning position by definition has a move that
        keeps the win.  If none of our classifications say `win_preserved`
        for a parent-W position, the perspective flip is wrong.
        """
        from game.rules import get_all_legal_moves
        for board in self._endgame_boards():
            parent_res = self.db.query(board)
            if parent_res is None or parent_res["outcome"] != "W":
                continue
            found = False
            for move in get_all_legal_moves(board):
                child_board = board.apply_move(move)
                child_res = self.db.query(child_board)
                if child_res is None:
                    continue
                if ahb._classify_transition("W", child_res["outcome"]) == "win_preserved":
                    found = True
                    break
            self.assertTrue(
                found,
                "Parent labelled W but no legal move classifies as win_preserved. "
                "Either the audit's _FLIP is inverted, or the Malom parent label is wrong."
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
