"""Integration between the scaffolded encoder and the move generator.

The learned-AI subsystem must never enumerate legal moves itself; the only
authority is :func:`game.rules.get_all_legal_moves`. These checks make sure
the current per-candidate encoder preserves that enumeration across phases.
"""

import unittest

from game.board import BoardState, POSITIONS
from game.rules import get_all_legal_moves
from learned_ai.models.scaffolded_encoder import encode_position


def _move_key(move):
    return move.get("from"), move.get("to"), move.get("capture")


def _assert_encoder_matches_engine(testcase, board):
    engine_moves = get_all_legal_moves(board)
    encoded = encode_position(board, board.turn, sentinel_advisor=None, db=None)

    testcase.assertIsNotNone(encoded)
    testcase.assertEqual(
        [_move_key(move) for move in encoded.legal_moves],
        [_move_key(move) for move in engine_moves],
    )
    testcase.assertEqual(encoded.feat_matrix.shape[0], len(engine_moves))


class TestLegalMoves(unittest.TestCase):
    def test_opening_encoder_matches_authoritative_moves(self):
        board = BoardState.new_game()
        _assert_encoder_matches_engine(self, board)

    def test_movement_encoder_matches_authoritative_moves(self):
        board = BoardState.from_setup(
            positions={
                "a7": "W", "d7": "W", "g7": "B", "g4": "B",
                "a1": "W", "g1": "B", "d1": "W",
            },
            turn="W",
            phase="move",
        )
        _assert_encoder_matches_engine(self, board)

    def test_fly_phase_allows_non_adjacent(self):
        # White flying: 3 pieces only, all placed.
        positions = {"a7": "W", "d7": "W", "g7": "W"}
        for sq in ("c5", "d5", "e5", "c4"):
            positions[sq] = "B"
        board = BoardState(
            positions={p: positions.get(p, "") for p in POSITIONS},
            turn="W",
            pieces_on_board={"W": 3, "B": 4},
            pieces_placed={"W": 9, "B": 9},
            pieces_captured={"W": 0, "B": 0},
        )
        legal = get_all_legal_moves(board)
        # Flying white can move any of its 3 pieces to any empty square.
        empties = sum(1 for value in board.positions.values() if value == "")
        self.assertEqual(len(legal), 3 * empties)
        _assert_encoder_matches_engine(self, board)


if __name__ == "__main__":
    unittest.main()
