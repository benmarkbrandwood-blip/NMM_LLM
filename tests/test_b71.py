"""
tests/test_b71.py — Regression for B-71: capture reform difficulty.

Problem: After Black closes mill a1-a4-a7 via d1→a1, the AI captures f4
instead of f2.  f4 is a cross-node (cardinal bonus +200) which inflates its
apparent value.  But capturing f4 leaves f2+g4 (two White pieces) adjacent to
the vacated f4 square; White can slide g4→f4 to immediately recreate the
f6-f4-f2 two-config.  Capturing f2 instead leaves only f4 adjacent to the
vacated f2 square, and no slide recreates a two-config.

Root cause: `cardinal_block` bonus rewards capturing cross-node f4 (+200) but
no bonus penalises the ease of re-occupation.

Fix: `capture_reform_difficulty` bonus — simulates each adjacent opponent
slide into the captured square and checks whether any mill would gain a
two-config.  If no such slide exists, the bonus fires (hard-to-reform capture).

Game record: 1.d6 d2/2.c4 b4/3.g4 d5/4.e3 d1/5.d3 c3/6.f4 e4/7.f6 b6/
8.f2xb6 b6/9.b2 a7/10.d6-d7 b4-a4/11.f6-d6 [Black: d1→a1×? pending]
"""
from __future__ import annotations

import unittest

from game.board import BoardState
from ai.game_ai import GameAI
from ai.heuristics import HeuristicWeights, tactical_move_bonus


def _b71_board() -> BoardState:
    """Board before Black's move 11 (d1→a1 pending, capturing White piece)."""
    pos = {
        # White after move 11 (f6→d6)
        "d7": "W", "c4": "W", "g4": "W", "e3": "W", "d3": "W",
        "f4": "W", "d6": "W", "f2": "W", "b2": "W",
        # Black before d1→a1
        "d2": "B", "a4": "B", "d5": "B", "d1": "B",
        "c3": "B", "e4": "B", "b6": "B", "a7": "B",
    }
    return BoardState.from_setup(pos, turn="B", phase="move")


class TestReformDifficultyBonus(unittest.TestCase):
    """Unit tests for the capture reform difficulty heuristic."""

    def test_f2_capture_scores_higher_than_f4(self):
        """After d1→a1 (closing a1-a4-a7), capturing f2 must outscore capturing f4."""
        board = _b71_board()
        weights = HeuristicWeights()

        after_f4 = board.apply_move({"from": "d1", "to": "a1", "capture": "f4"})
        after_f2 = board.apply_move({"from": "d1", "to": "a1", "capture": "f2"})

        score_f4 = tactical_move_bonus(board, after_f4, "B", weights)
        score_f2 = tactical_move_bonus(board, after_f2, "B", weights)

        self.assertGreater(
            score_f2, score_f4,
            f"f2 capture (score={score_f2}) should outscore f4 (score={score_f4}): "
            "f4 can be re-occupied by g4 to reform f6-f4-f2 two-config; f2 cannot.",
        )

    def test_reform_bonus_fires_for_f2_not_f4(self):
        """Reform bonus must fire for f2 (no re-forming slide) but not f4 (g4→f4 reforms)."""
        board = _b71_board()
        weights = HeuristicWeights()
        zero_w = HeuristicWeights(capture_reform_difficulty=0)

        after_f4 = board.apply_move({"from": "d1", "to": "a1", "capture": "f4"})
        after_f2 = board.apply_move({"from": "d1", "to": "a1", "capture": "f2"})

        # With bonus zeroed, check the base difference (cardinal advantage for f4)
        base_f4 = tactical_move_bonus(board, after_f4, "B", zero_w)
        base_f2 = tactical_move_bonus(board, after_f2, "B", zero_w)
        # f4 has cardinal bonus (+200) so base_f4 > base_f2
        self.assertGreater(base_f4, base_f2, "f4 should have higher base score (cardinal bonus)")

        # With bonus active, f2 should win
        bonus_f4 = tactical_move_bonus(board, after_f4, "B", weights)
        bonus_f2 = tactical_move_bonus(board, after_f2, "B", weights)
        self.assertGreater(bonus_f2, bonus_f4, "reform bonus must flip the preference to f2")

        # The bonus increased f2's score but not f4's
        self.assertGreater(bonus_f2 - base_f2, bonus_f4 - base_f4,
                           "reform bonus must add more to f2 than to f4")


class TestReformDifficultyIntegration(unittest.TestCase):
    """Integration: AI must capture f2 (not f4) when closing a1-a4-a7."""

    def test_ai_captures_f2_not_f4_shallow_difficulties(self):
        """
        Regression: after d1→a1 closes a1-a4-a7, shallow AI (difficulties 1-3)
        must not capture f4.  f4 appears attractive (cardinal +200) but g4 can
        immediately re-occupy it, reforming the f6-f4-f2 two-config.

        Note: difficulty=4 (deep search) may legitimately prefer f4 because
        removing a cardinal cross-node piece (blocking 4 lines) has deep tactical
        consequences that outweigh the static reform-difficulty heuristic.
        """
        board = _b71_board()
        for diff in (1, 2, 3):
            ai = GameAI(color="B", difficulty=diff)
            move = ai.choose_move(board)
            self.assertIsNotNone(move)
            if move.get("from") == "d1" and move.get("to") == "a1":
                self.assertNotEqual(
                    move.get("capture"), "f4",
                    f"difficulty={diff}: AI captured f4 after d1→a1; "
                    "f2 is the correct capture (harder for White to re-form).",
                )

    def test_non_reformable_capture_preferred_in_minimal_case(self):
        """Minimal case: two White pieces adjacent to closing mill capture target."""
        # Black closes a7-a4-a1 and chooses between capturing:
        # - f4 (g4=W and f2=W adjacent → easy reform: g4→f4 recreates 2-config)
        # - f2 (f4=W adjacent → no reform slide creates 2-config)
        board = _b71_board()
        weights = HeuristicWeights()

        for cap, can_reform in (("f4", True), ("f2", False)):
            after = board.apply_move({"from": "d1", "to": "a1", "capture": cap})
            score = tactical_move_bonus(board, after, "B", weights)
            # Both must be positive (mill closure bonus dominates)
            self.assertGreater(score, 0, f"capture {cap} should still score positively")

        # f2 specifically must score more
        score_f4 = tactical_move_bonus(
            board, board.apply_move({"from": "d1", "to": "a1", "capture": "f4"}),
            "B", weights,
        )
        score_f2 = tactical_move_bonus(
            board, board.apply_move({"from": "d1", "to": "a1", "capture": "f2"}),
            "B", weights,
        )
        self.assertGreater(score_f2, score_f4)


if __name__ == "__main__":
    unittest.main()
