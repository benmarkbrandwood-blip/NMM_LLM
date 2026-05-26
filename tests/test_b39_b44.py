"""
tests/test_b39_b44.py — Diagnostic + regression tests for B-39 through B-44.

Each test documents the bug contract; PASS means the behaviour is already
correct, FAIL means the bug is still present and a fix is required.

B-39: disrupted 2-config penalty (placing into a W,W,B dead pattern)
B-40: self_cycle_lost — losing own cycling setup is not penalised
B-41: safe mill opening — no score difference between safe/unsafe piece choice
B-43: dual-threat capture — prefer capture that removes opponent mill feeder
B-44: smart capture — prefer capture that unblocks own 2-config (ALREADY FIXED)
"""
from __future__ import annotations
import unittest
from game.board import BoardState, POSITIONS
from ai.heuristics import (
    tactical_move_bonus, DEFAULT_WEIGHTS,
    _two_configs, _cycling_mill_setup, _closeable_mills, _mill_cycle_ready,
)
from ai.game_ai import GameAI


# ── helpers ──────────────────────────────────────────────────────────────────

def _board(white: list[str], black: list[str], turn: str = "W",
           phase: str = "place",
           w_placed: int | None = None,
           b_placed: int | None = None) -> BoardState:
    pos = {p: "" for p in POSITIONS}
    for p in white:
        pos[p] = "W"
    for p in black:
        pos[p] = "B"
    wp = w_placed if w_placed is not None else (9 if phase != "place" else len(white))
    bp = b_placed if b_placed is not None else (9 if phase != "place" else len(black))
    return BoardState(
        positions=pos, turn=turn,
        pieces_on_board={"W": len(white), "B": len(black)},
        pieces_placed={"W": wp, "B": bp},
        pieces_captured={"W": 0, "B": 0},
    )


def _place(before: BoardState, sq: str) -> BoardState:
    pos = dict(before.positions)
    pos[sq] = "W"
    return BoardState(
        positions=pos, turn="B",
        pieces_on_board={**before.pieces_on_board, "W": before.pieces_on_board["W"] + 1},
        pieces_placed={**before.pieces_placed, "W": before.pieces_placed["W"] + 1},
        pieces_captured=before.pieces_captured,
    )


def _move(before: BoardState, frm: str, to: str,
          capture: str | None = None, color: str = "W") -> BoardState:
    """Apply frm→to for color, with optional capture of opponent piece."""
    pos = dict(before.positions)
    pos[frm] = ""
    pos[to] = color
    opp = "B" if color == "W" else "W"
    b_on = dict(before.pieces_on_board)
    cap_count = dict(before.pieces_captured)
    if capture:
        pos[capture] = ""
        b_on[opp] -= 1
        cap_count[color] = cap_count.get(color, 0) + 1
    return BoardState(
        positions=pos, turn="B" if color == "W" else "W",
        pieces_on_board=b_on,
        pieces_placed=before.pieces_placed,
        pieces_captured=cap_count,
    )


def _score(before: BoardState, after: BoardState, color: str = "W") -> int:
    return tactical_move_bonus(before, after, color)


# ── B-39: disrupted 2-config penalty ─────────────────────────────────────────

class TestB39DisruptedTwoConfig(unittest.TestCase):
    """
    B-39: placing into a (W,W,B) pattern in a mill creates dead structure.
    In move phase, no scatter bonus applies; the only signal is cross-node
    value.  There should be a penalty for creating a blocked dead formation.
    """

    def test_placement_phase_scatter_avoids_dead_structure(self):
        """
        In placement phase (index<6), scatter already discourages placing
        adjacent to own pieces, which covers the main dead-structure case.
        This test verifies the existing mitigation is active.
        """
        # B at d7, W at d6 already placed.
        # d5 is adjacent to d6 (W) → no scatter bonus; a4 is non-adjacent → scatter.
        before = _board(["d6", "f4", "b6", "f6"],
                        ["d7", "b4", "a1", "g1"],
                        phase="place", w_placed=4, b_placed=4)
        score_d5 = _score(before, _place(before, "d5"))
        score_a4 = _score(before, _place(before, "a4"))
        # Scatter bonus (400) should ensure a4 beats d5 substantially
        self.assertGreater(score_a4, score_d5,
            f"Scatter must prefer a4 ({score_a4}) over dead-structure d5 ({score_d5})")

    def test_move_phase_dead_structure_not_rewarded(self):
        """
        In move phase, the dead (W,W,B) pattern in d7-d6-d5 must not give
        setup_mill_bonus since _two_configs requires an empty closing square.
        Confirmed: d5 should not outscore a neutral same-ring move.
        """
        # W: d6, a7, g1, e5.  B: d7, b4, f2, a4.  Move phase.
        before = _board(
            ["d6", "a7", "g1", "e5"],
            ["d7", "b4", "f2", "a4"],
            phase="move", w_placed=9, b_placed=9,
        )
        # Moving e5→d5: creates (B,W,W) dead pattern in d7-d6-d5
        after_dead = _move(before, "e5", "d5")
        # Moving e5→c5: neutral inner-ring square, no dead structure
        after_safe = _move(before, "e5", "c5")

        sd = _score(before, after_dead)
        ss = _score(before, after_safe)

        # Without a disrupted-2-config penalty, d5 (cross-node) may beat c5.
        # The test documents this gap — a failing result here confirms B-39 is present.
        # Currently expected to FAIL if no penalty exists.
        # Both are inner-ring squares; d5 is a cross-node. c5 is a corner.
        # If no dead-structure penalty: sd ≥ ss (d5 wins on cross-node merit).
        # If penalty exists: ss ≥ sd.
        self.assertGreaterEqual(
            ss, sd,
            f"Neutral move c5 ({ss}) should score ≥ dead-structure d5 ({sd}) "
            "in move phase — disrupted-2-config penalty missing",
        )


# ── B-40: self_cycle_lost — no penalty for breaking own cycling setup ─────────

class TestB40PreserveDualMill(unittest.TestCase):
    """
    B-40: _cycling_mill_setup drops when a move breaks an own cycling pair,
    but tactical_move_bonus has no 'self_cycle_lost' negative term.
    Breaking a cycling setup should be penalised relative to a neutral move
    that preserves it.
    """

    def _cycling_board(self) -> BoardState:
        # Two Type-A cycling 2-configs:
        #   a7-d7-g7: W at a7+g7, closing d7
        #   b6-d6-f6: W at b6+f6, closing d6
        # d7 and d6 are adjacent (mill d7-d6-d5) → _cycling_mill_setup = 1.
        return _board(
            ["a7", "g7", "b6", "f6", "e5"],
            ["g4", "d1", "c3", "e3", "b4"],
            phase="move", w_placed=9, b_placed=9,
        )

    def test_cycling_setup_is_present(self):
        """Verify the board actually has a cycling setup before testing."""
        b = self._cycling_board()
        self.assertGreater(_cycling_mill_setup(b, "W"), 0,
            "_cycling_mill_setup must be > 0 for this test board")

    def test_cycling_setup_drops_when_2config_broken(self):
        """
        Moving a7 away breaks the a7-d7-g7 2-config, dropping cycling to 0.
        """
        before = self._cycling_board()
        after_break = _move(before, "a7", "a1")  # a1 is empty outer corner
        cyc_before = _cycling_mill_setup(before, "W")
        cyc_after  = _cycling_mill_setup(after_break, "W")
        self.assertLess(cyc_after, cyc_before,
            f"Cycling setup must drop when a7 moves away "
            f"(before={cyc_before}, after={cyc_after})")

    def test_breaking_cycling_setup_is_penalised(self):
        """
        Moving a7→a1 (breaks cycling) should score LOWER than moving
        e5→d5 (preserves both 2-configs, also gains cross-node).
        """
        before = self._cycling_board()
        after_break    = _move(before, "a7", "a1")   # cycling drops 1→0
        after_preserve = _move(before, "e5", "d5")   # cycling stays at 1

        score_break    = _score(before, after_break)
        score_preserve = _score(before, after_preserve)

        self.assertGreater(
            score_preserve, score_break,
            f"Preserving cycling (e5→d5, {score_preserve}) should outscore "
            f"breaking it (a7→a1, {score_break})",
        )

    def test_breaking_cycling_for_cross_node_is_penalised(self):
        """
        Controlled test: break cycling by moving to a cross-node (g7→d5)
        vs preserve cycling by moving a neutral piece to a corner (g1→a1).
        Without B-40 fix: break scores much higher due to cross-node gain.
        With fix: self_cycle_lost penalty should narrow the gap.
        """
        before = _board(
            ["a7", "g7", "b6", "f6", "g1"],
            ["g4", "d1", "c3", "e3", "b4"],
            phase="move", w_placed=9, b_placed=9,
        )
        self.assertGreater(_cycling_mill_setup(before, "W"), 0,
            "Board must have a cycling setup for this test")
        after_break    = _move(before, "g7", "d5")   # breaks a7-d7-g7 2-config; gains cross-node
        after_preserve = _move(before, "g1", "a1")   # neutral corner move; cycling intact

        cyc_break    = _cycling_mill_setup(after_break, "W")
        cyc_preserve = _cycling_mill_setup(after_preserve, "W")
        score_break    = _score(before, after_break)
        score_preserve = _score(before, after_preserve)

        self.assertEqual(cyc_break, 0, "Break move must drop cycling to 0")
        self.assertGreater(cyc_preserve, 0, "Preserve move must keep cycling > 0")
        # With self_cycle_lost = -300 and cross-node = +200, net = -100 → break < preserve
        self.assertGreater(
            score_preserve, score_break,
            f"Preserving cycling (g1→a1, {score_preserve}) should outscore "
            f"cross-node cycling break (g7→d5, {score_break}). "
            "self_cycle_lost penalty must exceed cross-node gain.",
        )


# ── B-41: safe mill opening ───────────────────────────────────────────────────

class TestB41SafeMillOpening(unittest.TestCase):
    """
    B-41: when opening a closed mill, moving the piece whose departure leaves
    a contested re-closing route should score lower than the safe option.
    Currently mill_open_bonus fires regardless of which piece is moved.
    """

    def _mill_board(self) -> BoardState:
        # W has closed mill a7-d7-g7.
        # a7 is adjacent to a4 (B piece) — so if W opens by moving a7 out,
        #   the closing square (a7) is immediately contestable by B at a4 adj to a7.
        # g7 is adjacent to g4 (empty) — safe opening direction.
        # Opening g7: closing sq a7-d7 → need to put g7 back; g7 is adjacent to g4 (empty).
        # Opening a7: closing sq d7-g7 → need to put a7 back; a7 is adjacent to a4 (B).
        return _board(
            ["a7", "d7", "g7", "f4", "b2"],
            ["a4", "b4", "d1", "e3", "c3"],
            phase="move", w_placed=9, b_placed=9,
        )

    def test_safe_mill_opening_scores_higher_or_equal(self):
        """
        Opening with g7 (safe — B has no piece adjacent to the re-closing
        square g7) should score ≥ opening with a7 (unsafe — B at a4 is
        adjacent to a7, immediately contesting re-closure).
        """
        before = self._mill_board()
        # g7 → f6: opens a7-d7 2-config with closing g7; no B piece next to g7
        after_safe   = _move(before, "g7", "f6")
        # a7 → b6: opens d7-g7 2-config with closing a7; B at a4 is adjacent to a7
        after_unsafe = _move(before, "a7", "b6")

        score_safe   = _score(before, after_safe)
        score_unsafe = _score(before, after_unsafe)

        self.assertGreaterEqual(
            score_safe, score_unsafe,
            f"Safe opening g7→f6 ({score_safe}) should score ≥ "
            f"unsafe opening a7→b6 ({score_unsafe}). "
            "B at a4 can contest re-closure of a7 — no safe-opening penalty.",
        )


# ── B-43: dual-threat mill closure ───────────────────────────────────────────

class TestB43DualThreatMillClosure(unittest.TestCase):
    """
    B-43: when W can close a mill and both a 'feeder-removing capture' (stops
    B's immediate mill threat) and a 'neutral capture' are available, W should
    prefer the feeder-removing capture.

    Design: B has an open 2-config b6-d6-f6 with a feeder piece at f4 (adjacent
    to f6, outside the mill) that can close it next turn.  W closes its own mill
    and chooses between capturing f4 (stops B's threat) vs e3 (neutral).
    """

    def _b43_board(self) -> BoardState:
        # W: a7, d7 (2-config in a7-d7-g7, closing g7); g4, b2, d3 (fillers)
        # B: b6, d6 (2-config in b6-d6-f6, closing f6); f4 (feeder adj to f6);
        #    a4, e3 (neutral pieces)
        return _board(
            ["a7", "d7", "g4", "b2", "d3"],
            ["b6", "d6", "f4", "a4", "e3"],
            phase="move", w_placed=9, b_placed=9,
        )

    def test_b_has_closeable_mill_with_outside_feeder(self):
        """Verify B's b6-d6-f6 is closeable via f4 (outside feeder)."""
        b = self._b43_board()
        # f4 is adjacent to f6? f6-f4-f2 is a mill, so YES f4 adj f6.
        # B has b6+d6 in b6-d6-f6 (closing f6), and f4 is B piece adj to f6.
        cl = _closeable_mills(b, "B")
        self.assertGreater(cl, 0,
            "B must have at least one closeable mill for the B-43 test")

    def test_prefer_feeder_capture_over_neutral(self):
        """
        W closes a7-d7-g7 by placing g4 at g7 (mill closure via the move
        g4→g7). W can then capture f4 (B's feeder, stops B's closeable mill)
        or e3 (neutral).
        """
        before = self._b43_board()
        # Close W's mill a7-d7-g7: move g4→g7
        after_feeder  = _move(before, "g4", "g7", capture="f4")  # stops B's mill
        after_neutral = _move(before, "g4", "g7", capture="e3")  # neutral

        score_feeder  = _score(before, after_feeder)
        score_neutral = _score(before, after_neutral)

        self.assertGreater(
            score_feeder, score_neutral,
            f"Feeder capture (f4, {score_feeder}) should outscore "
            f"neutral capture (e3, {score_neutral}) — "
            "stopping B's closeable mill is high-priority",
        )

    def test_safe_capture_bonus_fires_when_only_threat_removed(self):
        """
        Existing safe_capture_bonus (180) should fire when W's capture
        removes B's only closeable mill threat.
        """
        # Simpler board: B has exactly one closeable mill threat.
        # B has a1+g1 in a1-d1-g1 (closing d1), with feeder b2 adj to...
        # d1 adjacency: d1 adj a1 (in mill), g1 (in mill), d2.
        # Feeder piece outside mill adjacent to closing sq d1: d2 is adj to d1.
        # B has d2 as feeder.
        before = _board(
            ["b6", "f6", "d7", "a7", "e5"],
            ["a1", "g1", "d2", "b4", "f4"],
            phase="move", w_placed=9, b_placed=9,
        )
        # W closes b6-d6-f6 (move f6→d6), captures d2 (B's only feeder) vs b4 (neutral)
        after_feeder  = _move(before, "f6", "d6", capture="d2")
        after_neutral = _move(before, "f6", "d6", capture="b4")

        # Check closeable drops to 0 after capturing d2
        cl_before = _closeable_mills(before, "B")
        cl_after  = _closeable_mills(after_feeder, "B")

        score_feeder  = _score(before, after_feeder)
        score_neutral = _score(before, after_neutral)

        self.assertGreater(
            score_feeder, score_neutral,
            f"Removing B's only closeable threat (d2→gone, cl {cl_before}→{cl_after}, "
            f"score={score_feeder}) should outscore neutral capture b4 ({score_neutral})",
        )


# ── B-44: smart capture (ALREADY FIXED — regression guard) ───────────────────

class TestB44SmartCapture(unittest.TestCase):
    """
    B-44: mill captures should prefer the piece that unblocks an own 2-config
    or removes the opponent's critical feeder.  This is already handled by
    setup_mill_bonus (2-config gained) and capture_disrupt_feeder.
    These tests serve as regression guards.
    """

    def test_prefer_capture_that_unblocks_own_two_config(self):
        """
        Closing b6-f6 mill by moving to d6, then choosing between:
          - capturing d7 (removes B piece blocking W's a7-d7-g7 2-config)
          - capturing a4 (neutral)
        After capturing d7, W gains the a7-d7-g7 2-config → setup_mill_bonus fires.
        """
        before = _board(
            ["b6", "f6", "a7", "g7", "e5"],
            ["d7", "a4", "c3", "e3", "g1"],
            phase="move", w_placed=9, b_placed=9,
        )
        after_smart   = _move(before, "f6", "d6", capture="d7")
        after_neutral = _move(before, "f6", "d6", capture="a4")

        score_smart   = _score(before, after_smart)
        score_neutral = _score(before, after_neutral)

        self.assertGreater(score_smart, score_neutral,
            f"Capture unblocking own 2-config (d7, {score_smart}) should "
            f"outscore neutral capture (a4, {score_neutral})")

    def test_prefer_capture_removing_opponent_cycling_feeder(self):
        """
        Closing a7-d7-g7 mill by moving d7→g7, then choosing between:
          - capturing f4 (B feeder adjacent to B's closed middle-ring mill)
          - capturing g1 (isolated corner)
        capture_disrupt_feeder should reward taking f4.
        """
        before = _board(
            ["a7", "b2", "d5", "e4", "g4"],
            ["b6", "d6", "f6", "f4", "g1"],
            phase="move", w_placed=9, b_placed=9,
        )
        # B has closed mill b6-d6-f6, with f4 as potential cycling feeder (adj to f6)
        after_feeder   = _move(before, "g4", "g7", capture="f4")
        after_isolated = _move(before, "g4", "g7", capture="g1")

        score_feeder   = _score(before, after_feeder)
        score_isolated = _score(before, after_isolated)

        self.assertGreater(score_feeder, score_isolated,
            f"Feeder capture (f4, {score_feeder}) should outscore "
            f"isolated capture (g1, {score_isolated})")


if __name__ == "__main__":
    unittest.main()
