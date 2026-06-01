"""
tests/test_blocking.py — Unit tests for placement-phase threat blocking.

_immediate_mill_threats() restricts the current player to the opponent's
closing squares when:
  - Fork (≥2 simultaneous opponent 2-configs): always restrict.
  - Single threat: restrict unless STM can close their own mill this turn.
"""
from __future__ import annotations

import unittest

from game.board import BoardState
from ai.game_ai import GameAI, _immediate_mill_threats


def _place(positions: dict, turn: str = "W") -> BoardState:
    return BoardState.from_setup(positions, turn=turn, phase="place")


class TestImmediateMillThreatsPlacement(unittest.TestCase):

    # ── Single 2-config: mandatory block (unless STM has own 2-config) ──────────

    def test_single_two_config_is_threat_when_stm_has_no_own_config(self):
        # Black has one 2-config (a7+d7 → closing g7); White has g4 only (no own 2-config).
        # Single threat with no carveout → g7 must be blocked.
        b = _place({"a7": "B", "d7": "B", "g4": "W"})
        self.assertEqual(_immediate_mill_threats(b), {"g7"})

    def test_zero_two_configs_no_threat(self):
        # No opponent 2-config at all.
        b = _place({"a7": "B", "b2": "B", "g4": "W"})
        self.assertEqual(_immediate_mill_threats(b), set())

    # ── Two 2-configs: fork triggers block ─────────────────────────────────────

    def test_two_configs_both_closing_squares_returned(self):
        # Black: a7+d7 (→ g7) and a1+d1 (→ g1).
        b = _place({"a7": "B", "d7": "B", "a1": "B", "d1": "B", "g4": "W", "e4": "W"})
        threats = _immediate_mill_threats(b)
        self.assertIn("g7", threats)
        self.assertIn("g1", threats)

    def test_three_configs_all_closing_squares_returned(self):
        # Black: a7+d7 (→ g7), a1+d1 (→ g1), b6+d6 (→ f6).
        b = _place({
            "a7": "B", "d7": "B",
            "a1": "B", "d1": "B",
            "b6": "B", "d6": "B",
            "g4": "W",
        })
        threats = _immediate_mill_threats(b)
        self.assertIn("g7", threats)
        self.assertIn("g1", threats)
        self.assertIn("f6", threats)

    def test_shared_pivot_fork_both_closing_squares_returned(self):
        # a7 is pivot: sits in mill a7-d7-g7 AND mill a1-a4-a7.
        # Black at a7+d7 → g7; Black at a7+a4 → a1.
        b = _place({"a7": "B", "d7": "B", "a4": "B", "g4": "W", "e4": "W"})
        threats = _immediate_mill_threats(b)
        self.assertIn("g7", threats)
        self.assertIn("a1", threats)

    # ── One of two closing squares already occupied ────────────────────────────

    def test_single_two_config_is_now_a_threat(self):
        # a7+g7 → closing d7 is the one real 2-config.
        # f2+b2 would close d2, but d2 is White → not a 2-config.
        # W (e4 only) has no own 2-config so carveout doesn't fire.
        # Single threat → d7 is mandatory block.
        b = _place({
            "a7": "B", "g7": "B",              # → d7
            "f2": "B", "b2": "B", "d2": "W",   # blocked by White → not counted
            "e4": "W",
        })
        threats = _immediate_mill_threats(b)
        self.assertEqual(threats, {"d7"})

    # ── Phase guard: move-phase board must not use placement-fork logic ─────────

    def test_move_phase_single_two_config_fires_normally(self):
        # In move phase, single 2-config with adjacent opp piece IS a threat.
        b = BoardState.from_setup(
            {"a7": "B", "d7": "B", "a4": "B",   # a4 adjacent to a7 → move-phase threat at g7
             "g1": "W", "d1": "W", "a1": "W",
             "b6": "W", "b4": "W", "b2": "W",
             "f6": "W", "f4": "W", "f2": "W"},
            turn="W", phase="move",
        )
        threats = _immediate_mill_threats(b)
        self.assertIn("g7", threats)

    def test_move_phase_single_threat_carveout_when_stm_closes_mill(self):
        # White threatens b6 (b2-b4-b6).  Black can close c3-c4-c5 — no block-only filter.
        b = BoardState.from_setup(
            {
                "a7": "B", "g4": "B", "g1": "B", "a1": "B", "d6": "B",
                "f2": "B", "c5": "B", "d3": "B", "c4": "B",
                "d7": "W", "g7": "W", "d1": "W", "a4": "W", "f6": "W",
                "f4": "W", "b2": "W", "b4": "W", "d5": "W",
            },
            turn="B", phase="move",
        )
        self.assertEqual(_immediate_mill_threats(b), set())

    # ── choose_move integration ────────────────────────────────────────────────

    def test_choose_move_restricted_to_closing_squares_on_fork(self):
        # Black: b6+d6 (→ f6) and f2+d2 (→ b2) — exactly two 2-configs, no overlap.
        # White must land on f6 or b2.
        b = _place({
            "b6": "B", "d6": "B",   # → f6
            "f2": "B", "d2": "B",   # → b2
            "g4": "W", "e4": "W",
        })
        threats = _immediate_mill_threats(b)
        self.assertEqual(threats, {"f6", "b2"})
        ai = GameAI(color="W", difficulty=3)
        move = ai.choose_move(b)
        self.assertIn(move["to"], threats)

    def test_choose_move_unrestricted_when_stm_has_own_two_config(self):
        # Black has one 2-config (a7+d7 → g7); White has e4+g4 (own 2-config e4-f4-g4).
        # Carveout fires: stm_can_close → threats empty → AI free to choose any square.
        b = _place({
            "a7": "B", "d7": "B",
            "g4": "W", "e4": "W",
        })
        self.assertEqual(_immediate_mill_threats(b), set())
        ai = GameAI(color="W", difficulty=3)
        move = ai.choose_move(b)
        self.assertIsNotNone(move)
        self.assertIn("to", move)


class TestPlacementForkSurplus(unittest.TestCase):
    """B-81 — _placement_fork_surplus correctly counts extra unblockable threats."""

    def setUp(self):
        from ai.heuristics import _placement_fork_surplus
        self._pfs = _placement_fork_surplus

    def _board(self, white, black, turn="W", w_placed=None, b_placed=None):
        pos = {p: "" for p in __import__("game.board", fromlist=["POSITIONS"]).POSITIONS}
        for p in white:
            pos[p] = "W"
        for p in black:
            pos[p] = "B"
        return BoardState.from_setup(pos, turn=turn, phase="place")

    def test_zero_when_no_two_configs(self):
        b = self._board(["d6"], ["f4"])
        self.assertEqual(self._pfs(b, "W"), 0)
        self.assertEqual(self._pfs(b, "B"), 0)

    def test_zero_when_one_two_config(self):
        # Black has (f6,f4,f2) → only f6 closing → surplus 0
        b = self._board(["d6", "g4"], ["f4", "f2"])
        self.assertEqual(self._pfs(b, "B"), 0)

    def test_one_when_two_distinct_closing_squares(self):
        # Black's 6th move f2 position: (f6,f4,f2) closing f6; (f2,d2,b2) closing b2
        # White cannot block both.
        from game.board import POSITIONS
        pos = {p: "" for p in POSITIONS}
        for p in ["b4", "d6", "d3", "a1", "g4", "e4"]:
            pos[p] = "W"
        for p in ["f4", "d2", "d5", "a4", "c4", "f2"]:
            pos[p] = "B"
        b = BoardState.from_setup(pos, turn="W", phase="place")
        self.assertEqual(self._pfs(b, "B"), 1)  # 2 distinct closing → surplus 1
        self.assertEqual(self._pfs(b, "W"), 0)  # White has no 2-configs here

    def test_two_when_three_distinct_closing_squares(self):
        # Construct Black with 3 independent 2-configs: f6, b2, and g7 as closing squares.
        # f6: (f6,f4,f2) — f4=B, f2=B
        # b2: (f2,d2,b2) — f2=B, d2=B (shares f2 piece but different closing sq)
        # g7: (a7,d7,g7) — a7=B, d7=B
        from game.board import POSITIONS
        pos = {p: "" for p in POSITIONS}
        for p in ["g4", "e4", "c5"]:
            pos[p] = "W"
        for p in ["f4", "f2", "d2", "a7", "d7"]:
            pos[p] = "B"
        b = BoardState.from_setup(pos, turn="W", phase="place")
        self.assertGreaterEqual(self._pfs(b, "B"), 2)

    def test_zero_when_all_placed(self):
        # After all 9 pieces placed, function returns 0 (placement phase over).
        from game.board import POSITIONS
        pos = {p: "" for p in POSITIONS}
        white = ["a7", "d7", "g7", "g4", "g1", "d1", "a1", "a4", "b4"]
        black = ["b6", "d6", "f6", "f4", "f2", "d2", "b2", "c5", "e5"]
        for p in white:
            pos[p] = "W"
        for p in black:
            pos[p] = "B"
        b = BoardState(
            positions=pos, turn="W",
            pieces_on_board={"W": 9, "B": 9},
            pieces_placed={"W": 9, "B": 9},
            pieces_captured={"W": 0, "B": 0},
            hash_key=0,
        )
        self.assertEqual(self._pfs(b, "B"), 0)


class TestB81PlacementForkHeuristic(unittest.TestCase):
    """Integration: the evaluate() term penalises positions where the opponent
    has an independent dual 2-config fork in placement phase (B-81)."""

    def test_fork_scores_worse_than_single_threat(self):
        """After Black's fork move, evaluate(board, 'W') must score lower
        (more negative from White's perspective) than the same position with
        only one of Black's two 2-configs present."""
        from game.board import POSITIONS
        from ai.heuristics import evaluate

        # Two-threat fork: Black has (f6,f4,f2) and (f2,d2,b2)
        pos_fork = {p: "" for p in POSITIONS}
        for p in ["b4", "d6", "d3", "a1", "g4", "e4"]:
            pos_fork[p] = "W"
        for p in ["f4", "d2", "d5", "a4", "c4", "f2"]:
            pos_fork[p] = "B"
        board_fork = BoardState.from_setup(pos_fork, turn="W", phase="place")

        # Single threat: remove d2 (breaks (f2,d2,b2)); only (f6,f4,f2) remains
        pos_single = dict(pos_fork)
        pos_single["d2"] = ""
        board_single = BoardState.from_setup(pos_single, turn="W", phase="place")

        score_fork   = evaluate(board_fork,   "W")
        score_single = evaluate(board_single, "W")
        self.assertLess(
            score_fork, score_single,
            f"Fork position must score worse for White than single-threat: "
            f"fork={score_fork}, single={score_single}",
        )


if __name__ == "__main__":
    unittest.main()
