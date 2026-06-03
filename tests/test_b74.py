"""
tests/test_b74.py — Regression for B-74: cross-mill cycling fork static bonus.

Scenario: the game 1.d6 d2 / 2.f4 b4 / 3.f6 f2 / 4.b6×f2 f2 / 5.c4 b2×f4 /
6.d3 e3 / 7.g4 d7 / 8.c5 c3 / 9.e5 d5 / 10.f6→f4 e3→e4 / 11.f4→f6×d5.

After move 11 White has:
  W: b6, d6, f6, c4, d3, g4, c5, e5   (mill b6-d6-f6 closed; c5-?-e5 needs d5)
  B: d2, b4, f2, b2, d7, c3, e4        (d5 empty — just captured)

d6 is in the closed mill b6-d6-f6 and is adjacent to d5 (empty), which is the
missing piece of c5-d5-e5 (White has c5 and e5).  Moving d6→d5 closes c5-d5-e5
then d5→d6 closes b6-d6-f6 again — capturing every two turns indefinitely.

_cross_mill_cycling should return 1 for White and 0 for Black.
The static eval should score higher for White than for a board where the same
pieces are present but no cycling fork exists.
"""
from __future__ import annotations

import unittest

from game.board import BoardState
from ai.heuristics import _cross_mill_cycling, evaluate, HeuristicWeights


def _make_post_move11_board() -> BoardState:
    """Return the board after 11.f4→f6×d5 using BoardState.from_setup."""
    pos = {
        "b6": "W", "d6": "W", "f6": "W",
        "c4": "W", "d3": "W", "g4": "W", "c5": "W", "e5": "W",
        "d2": "B", "b4": "B", "f2": "B", "b2": "B",
        "d7": "B", "c3": "B", "e4": "B",
    }
    return BoardState.from_setup(pos, turn="W", phase="move")


class TestCrossMillCycling(unittest.TestCase):

    def test_detects_fork_for_white(self):
        board = _make_post_move11_board()
        self.assertEqual(_cross_mill_cycling(board, "W"), 1,
                         "Should detect exactly one two-mill cycling fork for White")

    def test_no_fork_for_black(self):
        board = _make_post_move11_board()
        self.assertEqual(_cross_mill_cycling(board, "B"), 0,
                         "Black has no closed mill adjacent to a near-mill closing square")

    def test_eval_favours_fork_holder(self):
        board = _make_post_move11_board()
        w_score = evaluate(board, "W")
        b_score = evaluate(board, "B")
        self.assertGreater(w_score, 0,
                           "Static eval should be positive for White (holds the fork)")
        self.assertLess(b_score, 0,
                        "Static eval should be negative for Black (facing forced losses)")

    def test_no_fork_without_near_mill(self):
        """If the second mill (c5-d5-e5) is broken, the fork disappears."""
        pos = {
            "b6": "W", "d6": "W", "f6": "W",
            "c4": "W", "d3": "W", "g4": "W", "e5": "W",  # c5 removed
            "d2": "B", "b4": "B", "f2": "B", "b2": "B",
            "d7": "B", "c3": "B", "e4": "B",
        }
        board = BoardState.from_setup(pos, turn="W", phase="move")
        self.assertEqual(_cross_mill_cycling(board, "W"), 0,
                         "Fork disappears when the near-mill is broken")

    def test_no_fork_when_closing_square_occupied(self):
        """If d5 is occupied by an opponent, the pivot can't reach it."""
        pos = {
            "b6": "W", "d6": "W", "f6": "W",
            "c4": "W", "d3": "W", "g4": "W", "c5": "W", "e5": "W",
            "d2": "B", "b4": "B", "f2": "B", "b2": "B",
            "d7": "B", "c3": "B", "e4": "B", "d5": "B",  # d5 blocked
        }
        board = BoardState.from_setup(pos, turn="W", phase="move")
        self.assertEqual(_cross_mill_cycling(board, "W"), 0,
                         "Fork disappears when the near-mill closing square is occupied")


if __name__ == "__main__":
    unittest.main()
