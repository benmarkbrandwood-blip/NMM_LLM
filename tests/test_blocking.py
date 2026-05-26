"""
tests/test_blocking.py — Unit tests for B-50: placement-phase fork blocking.

_immediate_mill_threats() must return non-empty during placement when the
opponent holds ≥2 simultaneous 2-configs (a fork), restricting the
current player to the closing squares of those mills.
"""
from __future__ import annotations

import unittest

from game.board import BoardState
from ai.game_ai import GameAI, _immediate_mill_threats


def _place(positions: dict, turn: str = "W") -> BoardState:
    return BoardState.from_setup(positions, turn=turn, phase="place")


class TestImmediateMillThreatsPlacement(unittest.TestCase):

    # ── Single 2-config: no mandatory block ────────────────────────────────────

    def test_single_two_config_no_threat(self):
        # Black has one 2-config (a7+d7 → closing g7); White to place.
        b = _place({"a7": "B", "d7": "B", "g4": "W"})
        self.assertEqual(_immediate_mill_threats(b), set())

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

    def test_one_of_two_intended_configs_blocked_gives_single_config(self):
        # a7+g7 → closing d7 is the one real 2-config.
        # f2+b2 would close d2, but d2 is White → not a 2-config.
        # a7 and g7 share no other mills with each other; f2 and b2 share none.
        # Count = 1 → no fork → empty set.
        b = _place({
            "a7": "B", "g7": "B",              # → d7
            "f2": "B", "b2": "B", "d2": "W",   # blocked by White → not counted
            "e4": "W",
        })
        threats = _immediate_mill_threats(b)
        self.assertEqual(threats, set())

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

    def test_choose_move_unrestricted_with_single_two_config(self):
        # Only one Black 2-config → no mandatory block → AI free to choose any square.
        b = _place({
            "a7": "B", "d7": "B",
            "g4": "W", "e4": "W",
        })
        ai = GameAI(color="W", difficulty=3)
        move = ai.choose_move(b)
        # Move is legal (not checking destination — just verifying it doesn't crash
        # and is not artificially restricted to g7 alone).
        self.assertIsNotNone(move)
        self.assertIn("to", move)


if __name__ == "__main__":
    unittest.main()
