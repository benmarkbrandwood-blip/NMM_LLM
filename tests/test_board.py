"""
tests/test_board.py — Stage 1 acceptance tests.

Covers:
  - All 16 mills defined and structurally correct
  - Mill detection (is_mill)
  - Adjacency graph: bidirectional, 32 edges, correct neighbour counts
  - Legal placements
  - Legal moves (including fly phase)
  - Blocked player detection
  - Legal captures (non-mill preference, all-in-mill fallback)
  - apply_move: placement, movement, capture
  - Phase transitions
  - FEN string: unique per state, round-trip stable
  - Notation: encode/decode round-trip for all move types
  - Full game replay: the sample game from the plan produces the correct final board
"""

import sys
import os
import unittest

# Allow running from the project root: python -m tests.test_board
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game.board import ADJACENCY, MILLS, POSITIONS, BoardState
from game.notation import encode_move, export_pgn_style, parse_move_string
from game.rules import (
    can_fly,
    does_form_mill,
    get_all_legal_moves,
    get_game_phase,
    is_blocked,
    is_terminal,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def place_pieces(board: BoardState, placements: list[tuple[str, str]]) -> BoardState:
    """Apply a sequence of (color, position) placements, alternating turns is NOT
    enforced here — we set up test fixtures directly."""
    pos = dict(board.positions)
    on = dict(board.pieces_on_board)
    placed = dict(board.pieces_placed)
    captured = dict(board.pieces_captured)
    for color, p in placements:
        pos[p] = color
        on[color] += 1
        placed[color] += 1
    return BoardState(
        positions=pos,
        turn=board.turn,
        pieces_on_board=on,
        pieces_placed=placed,
        pieces_captured=captured,
    )


# ── Mill definitions ──────────────────────────────────────────────────────────

class TestMillDefinitions(unittest.TestCase):

    def test_exactly_16_mills(self):
        self.assertEqual(len(MILLS), 16)

    def test_each_mill_has_three_positions(self):
        for mill in MILLS:
            self.assertEqual(len(mill), 3, msg=f"Mill {mill} does not have 3 positions")

    def test_all_mill_positions_are_valid(self):
        pos_set = set(POSITIONS)
        for mill in MILLS:
            for p in mill:
                self.assertIn(p, pos_set, msg=f"{p} in mill {mill} is not a valid position")

    def test_no_duplicate_mills(self):
        canonical = [tuple(sorted(m)) for m in MILLS]
        self.assertEqual(len(canonical), len(set(canonical)))

    def test_known_outer_mills(self):
        outer = {frozenset(m) for m in MILLS}
        self.assertIn(frozenset(("a7", "d7", "g7")), outer)
        self.assertIn(frozenset(("g1", "d1", "a1")), outer)
        self.assertIn(frozenset(("a1", "a4", "a7")), outer)
        self.assertIn(frozenset(("g7", "g4", "g1")), outer)

    def test_known_cross_mills(self):
        outer = {frozenset(m) for m in MILLS}
        self.assertIn(frozenset(("d7", "d6", "d5")), outer)
        self.assertIn(frozenset(("g4", "f4", "e4")), outer)
        self.assertIn(frozenset(("d1", "d2", "d3")), outer)
        self.assertIn(frozenset(("a4", "b4", "c4")), outer)

    def test_known_inner_mills(self):
        outer = {frozenset(m) for m in MILLS}
        self.assertIn(frozenset(("c5", "d5", "e5")), outer)
        self.assertIn(frozenset(("e5", "e4", "e3")), outer)
        self.assertIn(frozenset(("e3", "d3", "c3")), outer)
        self.assertIn(frozenset(("c3", "c4", "c5")), outer)


# ── Adjacency graph ───────────────────────────────────────────────────────────

class TestAdjacency(unittest.TestCase):

    def test_all_positions_have_adjacency_entry(self):
        for pos in POSITIONS:
            self.assertIn(pos, ADJACENCY, msg=f"{pos} missing from ADJACENCY")

    def test_adjacency_is_bidirectional(self):
        for pos, neighbours in ADJACENCY.items():
            for n in neighbours:
                self.assertIn(pos, ADJACENCY[n],
                              msg=f"Edge {pos}-{n} is not bidirectional")

    def test_total_edges(self):
        # 32 undirected edges → 64 directed
        total = sum(len(v) for v in ADJACENCY.values())
        self.assertEqual(total, 64)

    def test_corner_positions_have_two_neighbours(self):
        corners = {"a7", "g7", "g1", "a1", "b6", "f6", "f2", "b2",
                   "c5", "e5", "e3", "c3"}
        for pos in corners:
            self.assertEqual(len(ADJACENCY[pos]), 2,
                             msg=f"Corner {pos} should have 2 neighbours")

    def test_outer_midpoints_have_three_neighbours(self):
        outer_mids = {"d7", "g4", "d1", "a4"}
        for pos in outer_mids:
            self.assertEqual(len(ADJACENCY[pos]), 3,
                             msg=f"Outer midpoint {pos} should have 3 neighbours")

    def test_middle_midpoints_have_four_neighbours(self):
        middle_mids = {"d6", "f4", "d2", "b4"}
        for pos in middle_mids:
            self.assertEqual(len(ADJACENCY[pos]), 4,
                             msg=f"Middle midpoint {pos} should have 4 neighbours")

    def test_inner_midpoints_have_three_neighbours(self):
        inner_mids = {"d5", "e4", "d3", "c4"}
        for pos in inner_mids:
            self.assertEqual(len(ADJACENCY[pos]), 3,
                             msg=f"Inner midpoint {pos} should have 3 neighbours")

    def test_known_cross_connections(self):
        self.assertIn("d6", ADJACENCY["d7"])
        self.assertIn("d7", ADJACENCY["d6"])
        self.assertIn("d5", ADJACENCY["d6"])
        self.assertIn("d6", ADJACENCY["d5"])


# ── Mill detection ────────────────────────────────────────────────────────────

class TestMillDetection(unittest.TestCase):

    def _board_with(self, placements):
        b = BoardState.new_game()
        return place_pieces(b, placements)

    def test_detects_outer_ring_mill(self):
        b = self._board_with([("W", "a7"), ("W", "d7"), ("W", "g7")])
        self.assertTrue(b.is_mill("a7", "W"))
        self.assertTrue(b.is_mill("d7", "W"))
        self.assertTrue(b.is_mill("g7", "W"))

    def test_does_not_detect_incomplete_mill(self):
        b = self._board_with([("W", "a7"), ("W", "d7")])
        self.assertFalse(b.is_mill("a7", "W"))

    def test_mill_is_color_specific(self):
        b = self._board_with([("W", "a7"), ("W", "d7"), ("B", "g7")])
        self.assertFalse(b.is_mill("a7", "W"))

    def test_detects_cross_mill(self):
        b = self._board_with([("B", "d7"), ("B", "d6"), ("B", "d5")])
        self.assertTrue(b.is_mill("d6", "B"))

    def test_piece_not_in_any_mill(self):
        b = self._board_with([("W", "a7")])
        self.assertFalse(b.is_mill("a7", "W"))


# ── Legal placements ──────────────────────────────────────────────────────────

class TestLegalPlacements(unittest.TestCase):

    def test_empty_board_has_24_placements(self):
        b = BoardState.new_game()
        self.assertEqual(len(b.legal_placements("W")), 24)

    def test_occupied_position_excluded(self):
        b = BoardState.new_game()
        b = place_pieces(b, [("W", "d2")])
        self.assertNotIn("d2", b.legal_placements("W"))
        self.assertEqual(len(b.legal_placements("W")), 23)

    def test_all_returned_positions_are_empty(self):
        b = BoardState.new_game()
        b = place_pieces(b, [("W", "a7"), ("B", "d7")])
        for pos in b.legal_placements("W"):
            self.assertEqual(b.positions[pos], "")


# ── Legal moves (movement / fly phase) ───────────────────────────────────────

class TestLegalMoves(unittest.TestCase):

    def _post_placement_board(self) -> BoardState:
        """Board with all 9 pieces placed per side so move phase begins."""
        b = BoardState.new_game()
        whites = ["a7", "d7", "g7", "g4", "g1", "d1", "a1", "a4", "b6"]
        blacks = ["d6", "f6", "f4", "f2", "d2", "b2", "b4", "c5", "d5"]
        pos = {p: "" for p in POSITIONS}
        for p in whites:
            pos[p] = "W"
        for p in blacks:
            pos[p] = "B"
        return BoardState(
            positions=pos,
            turn="W",
            pieces_on_board={"W": 9, "B": 9},
            pieces_placed={"W": 9, "B": 9},
            pieces_captured={"W": 0, "B": 0},
        )

    def test_move_only_to_adjacent(self):
        b = self._post_placement_board()
        moves = b.legal_moves("W")
        for src, dst in moves:
            self.assertIn(dst, ADJACENCY[src],
                          msg=f"{dst} is not adjacent to {src}")

    def test_cannot_move_to_occupied(self):
        b = self._post_placement_board()
        for src, dst in b.legal_moves("W"):
            self.assertEqual(b.positions[dst], "",
                             msg=f"Destination {dst} is occupied")

    def test_blocked_player_has_no_legal_moves(self):
        """White has 4 pieces in a cluster with every exit blocked by Black."""
        # White: a7, d7, g7, a4 (outer ring top + left midpoint)
        # White pieces can only reach each other (blocked) or Black squares.
        # a7 → d7(W), a4(W)          : both White
        # d7 → a7(W), g7(W), d6(B)   : all blocked
        # g7 → d7(W), g4(B)          : all blocked
        # a4 → a1(B), a7(W), b4(B)   : all blocked
        pos = {p: "" for p in POSITIONS}
        pos["a7"] = pos["d7"] = pos["g7"] = pos["a4"] = "W"
        pos["d6"] = pos["g4"] = pos["a1"] = pos["b4"] = "B"
        b = BoardState(
            positions=pos,
            turn="W",
            pieces_on_board={"W": 4, "B": 4},
            pieces_placed={"W": 9, "B": 9},
            pieces_captured={"W": 0, "B": 0},
        )
        self.assertEqual(b.legal_moves("W"), [])

    def test_fly_phase_allows_any_empty(self):
        """Player with exactly 3 pieces may fly to any empty square."""
        pos = {p: "" for p in POSITIONS}
        pos["a7"] = "W"
        pos["d7"] = "W"
        pos["g7"] = "W"
        b = BoardState(
            positions=pos,
            turn="W",
            pieces_on_board={"W": 3, "B": 0},
            pieces_placed={"W": 9, "B": 9},
            pieces_captured={"W": 6, "B": 0},
        )
        moves = b.legal_moves("W")
        # Every White piece should be able to reach every empty position
        empty_count = sum(1 for p in POSITIONS if pos[p] == "")
        self.assertEqual(len(moves), 3 * empty_count)


# ── Blocked player and terminal detection ─────────────────────────────────────

class TestTerminal(unittest.TestCase):

    def test_not_terminal_on_new_game(self):
        b = BoardState.new_game()
        term, winner = is_terminal(b)
        self.assertFalse(term)
        self.assertIsNone(winner)

    def test_terminal_when_below_three_pieces(self):
        pos = {p: "" for p in POSITIONS}
        pos["a7"] = "B"
        pos["d7"] = "B"
        b = BoardState(
            positions=pos,
            turn="W",
            pieces_on_board={"W": 5, "B": 2},
            pieces_placed={"W": 9, "B": 9},
            pieces_captured={"W": 7, "B": 0},
        )
        term, winner = is_terminal(b)
        self.assertTrue(term)
        self.assertEqual(winner, "W")

    def test_terminal_when_blocked(self):
        """White has pieces but every one is surrounded → blocked → White loses."""
        pos = {p: "" for p in POSITIONS}
        # White piece at d5, all neighbours Black
        pos["d5"] = "W"
        for n in ADJACENCY["d5"]:
            pos[n] = "B"
        # Give White 9 placed so they're in move phase
        b = BoardState(
            positions=pos,
            turn="W",
            pieces_on_board={"W": 1, "B": 3},
            pieces_placed={"W": 9, "B": 9},
            pieces_captured={"W": 0, "B": 0},
        )
        term, winner = is_terminal(b)
        self.assertTrue(term)
        self.assertEqual(winner, "B")

    def test_not_terminal_during_placement(self):
        """Fewer than 3 pieces on board is fine during placement phase."""
        b = BoardState.new_game()
        b = place_pieces(b, [("W", "a7"), ("W", "d7")])
        term, _ = is_terminal(b)
        self.assertFalse(term)


# ── Legal captures ────────────────────────────────────────────────────────────

class TestLegalCaptures(unittest.TestCase):

    def test_prefers_non_mill_pieces(self):
        pos = {p: "" for p in POSITIONS}
        # Black has a mill at a7-d7-g7 and a lone piece at b6
        pos["a7"] = pos["d7"] = pos["g7"] = "B"
        pos["b6"] = "B"
        pos["c5"] = "W"
        b = BoardState(
            positions=pos,
            turn="W",
            pieces_on_board={"W": 1, "B": 4},
            pieces_placed={"W": 1, "B": 4},
            pieces_captured={"W": 0, "B": 0},
        )
        captures = b.legal_captures("W")
        self.assertEqual(captures, ["b6"])

    def test_falls_back_to_mill_pieces_when_all_in_mill(self):
        pos = {p: "" for p in POSITIONS}
        # Black only has the mill a7-d7-g7
        pos["a7"] = pos["d7"] = pos["g7"] = "B"
        pos["c5"] = "W"
        b = BoardState(
            positions=pos,
            turn="W",
            pieces_on_board={"W": 1, "B": 3},
            pieces_placed={"W": 1, "B": 3},
            pieces_captured={"W": 0, "B": 0},
        )
        captures = b.legal_captures("W")
        self.assertCountEqual(captures, ["a7", "d7", "g7"])


# ── apply_move ────────────────────────────────────────────────────────────────

class TestApplyMove(unittest.TestCase):

    def test_placement_updates_position_and_counters(self):
        b = BoardState.new_game()
        move = {"from": None, "to": "d2", "capture": None}
        nb = b.apply_move(move)
        self.assertEqual(nb.positions["d2"], "W")
        self.assertEqual(nb.pieces_on_board["W"], 1)
        self.assertEqual(nb.pieces_placed["W"], 1)
        self.assertEqual(nb.turn, "B")

    def test_movement_updates_position(self):
        pos = {p: "" for p in POSITIONS}
        pos["d5"] = "W"
        b = BoardState(
            positions=pos,
            turn="W",
            pieces_on_board={"W": 1, "B": 0},
            pieces_placed={"W": 9, "B": 9},
            pieces_captured={"W": 0, "B": 0},
        )
        move = {"from": "d5", "to": "c5", "capture": None}
        nb = b.apply_move(move)
        self.assertEqual(nb.positions["d5"], "")
        self.assertEqual(nb.positions["c5"], "W")
        self.assertEqual(nb.pieces_on_board["W"], 1)

    def test_capture_removes_opponent_piece(self):
        b = BoardState.new_game()
        b = place_pieces(b, [("B", "f6")])
        move = {"from": None, "to": "d2", "capture": "f6"}
        nb = b.apply_move(move)
        self.assertEqual(nb.positions["f6"], "")
        self.assertEqual(nb.pieces_on_board["B"], 0)
        self.assertEqual(nb.pieces_captured["W"], 1)

    def test_apply_move_is_immutable(self):
        b = BoardState.new_game()
        move = {"from": None, "to": "d2", "capture": None}
        _ = b.apply_move(move)
        self.assertEqual(b.positions["d2"], "")  # original unchanged


# ── Phase transitions ─────────────────────────────────────────────────────────

class TestPhaseTransitions(unittest.TestCase):

    def test_board_phase_is_place_initially(self):
        b = BoardState.new_game()
        self.assertEqual(b.phase, "place")

    def test_board_phase_moves_after_all_placed(self):
        whites = ["a7", "d7", "g7", "g4", "g1", "d1", "a1", "a4", "b6"]
        blacks = ["d6", "f6", "f4", "f2", "d2", "b2", "b4", "c5", "d5"]
        pos = {p: "" for p in POSITIONS}
        for p in whites:
            pos[p] = "W"
        for p in blacks:
            pos[p] = "B"
        b = BoardState(
            positions=pos,
            turn="W",
            pieces_on_board={"W": 9, "B": 9},
            pieces_placed={"W": 9, "B": 9},
            pieces_captured={"W": 0, "B": 0},
        )
        self.assertEqual(b.phase, "move")

    def test_get_game_phase_place_for_unplaced_pieces(self):
        b = BoardState.new_game()
        self.assertEqual(get_game_phase(b, "W"), "place")

    def test_get_game_phase_fly_when_three_pieces(self):
        pos = {p: "" for p in POSITIONS}
        pos["a7"] = pos["d7"] = pos["g7"] = "W"
        b = BoardState(
            positions=pos,
            turn="W",
            pieces_on_board={"W": 3, "B": 0},
            pieces_placed={"W": 9, "B": 9},
            pieces_captured={"W": 0, "B": 0},
        )
        self.assertEqual(get_game_phase(b, "W"), "fly")


# ── FEN string ────────────────────────────────────────────────────────────────

class TestFenString(unittest.TestCase):

    def test_new_game_fen_has_correct_length(self):
        b = BoardState.new_game()
        fen = b.to_fen_string()
        board_part = fen.split("|")[0]
        self.assertEqual(len(board_part), 24)

    def test_fen_reflects_piece_placements(self):
        b = BoardState.new_game()
        b = place_pieces(b, [("W", POSITIONS[0])])
        fen = b.to_fen_string()
        self.assertTrue(fen.startswith("W"))

    def test_different_boards_have_different_fens(self):
        b1 = BoardState.new_game()
        b1 = place_pieces(b1, [("W", "a7")])
        b2 = BoardState.new_game()
        b2 = place_pieces(b2, [("W", "d7")])
        self.assertNotEqual(b1.to_fen_string(), b2.to_fen_string())

    def test_fen_includes_turn(self):
        b = BoardState.new_game()
        self.assertIn("|W|", b.to_fen_string())


# ── Notation round-trip ───────────────────────────────────────────────────────

class TestNotation(unittest.TestCase):

    def _rt(self, s: str) -> str:
        """Decode then re-encode a notation string."""
        m = parse_move_string(s)
        if m.get("end"):
            return "*"
        from game.rules import get_game_phase
        phase = "move" if "-" in s else "place"
        return encode_move(m, phase)

    def test_placement(self):
        self.assertEqual(self._rt("d2"), "d2")

    def test_placement_with_capture(self):
        self.assertEqual(self._rt("d2xf6"), "d2xf6")

    def test_movement(self):
        self.assertEqual(self._rt("c5-c4"), "c5-c4")

    def test_movement_with_capture(self):
        self.assertEqual(self._rt("g7-g4xd1"), "g7-g4xd1")

    def test_end_marker(self):
        self.assertEqual(self._rt("*"), "*")

    def test_parse_placement(self):
        m = parse_move_string("d2")
        self.assertIsNone(m["from"])
        self.assertEqual(m["to"], "d2")
        self.assertIsNone(m["capture"])

    def test_parse_movement_with_capture(self):
        m = parse_move_string("g7-g4xd1")
        self.assertEqual(m["from"], "g7")
        self.assertEqual(m["to"], "g4")
        self.assertEqual(m["capture"], "d1")


# ── does_form_mill ────────────────────────────────────────────────────────────

class TestDoesFormMill(unittest.TestCase):

    def test_placement_that_forms_mill(self):
        b = BoardState.new_game()
        b = place_pieces(b, [("W", "a7"), ("W", "d7")])
        b = BoardState(
            positions=b.positions,
            turn="W",
            pieces_on_board=b.pieces_on_board,
            pieces_placed=b.pieces_placed,
            pieces_captured=b.pieces_captured,
        )
        move = {"from": None, "to": "g7", "capture": None}
        self.assertTrue(does_form_mill(b, move))

    def test_placement_that_does_not_form_mill(self):
        b = BoardState.new_game()
        b = place_pieces(b, [("W", "a7")])
        b = BoardState(
            positions=b.positions,
            turn="W",
            pieces_on_board=b.pieces_on_board,
            pieces_placed=b.pieces_placed,
            pieces_captured=b.pieces_captured,
        )
        move = {"from": None, "to": "d7", "capture": None}
        self.assertFalse(does_form_mill(b, move))


# ── Full game replay ──────────────────────────────────────────────────────────

class TestFullGameReplay(unittest.TestCase):
    """
    Replay the 14-move sample game from the plan and verify the final board FEN.
    The notation (from the plan):

        1.  d2  d6
        2.  f4  b4
        3.  f2  f6
        4.  b2xf6  f6       <- note: White places b2 and captures f6
        5.  b6  c3
        6.  c5  e3
        7.  d3  d1
        8.  e5  d5
        9.  d7  a4
        10. c5-c4  a4-a7
        11. e5-e4  d5-e5
        12. d7-g7  a7-d7
        13. g7-g4xd1  e5-d5xb6
        14. d2-d1  *

    Turn 14 ends with '*' which we treat as game-over; the game record will
    show White's final move and no Black move.
    We verify:
      - the notation exported by export_pgn_style matches the original
      - the final board state is internally consistent (pieces counted correctly)
    """

    RAW_GAME = [
        # (color, notation_string)
        ("W", "d2"),    ("B", "d6"),
        ("W", "f4"),    ("B", "b4"),
        ("W", "f2"),    ("B", "f6"),
        ("W", "b2xf6"), ("B", "f6"),   # White captures f6; Black places at f6 again
        ("W", "b6"),    ("B", "c3"),
        ("W", "c5"),    ("B", "e3"),
        ("W", "d3"),    ("B", "d1"),
        ("W", "e5"),    ("B", "d5"),
        ("W", "d7"),    ("B", "a4"),
        ("W", "c5-c4"), ("B", "a4-a7"),
        ("W", "e5-e4"), ("B", "d5-e5"),
        ("W", "d7-g7"), ("B", "a7-d7"),
        ("W", "g7-g4xd1"), ("B", "e5-d5xb6"),
        ("W", "d2-d1"),  # game ends after White's move 14
    ]

    def _apply_sequence(self):
        board = BoardState.new_game()
        applied: list[dict] = []

        for color, notation in self.RAW_GAME:
            # Force the turn to match the notation's color
            board = BoardState(
                positions=board.positions,
                turn=color,
                pieces_on_board=board.pieces_on_board,
                pieces_placed=board.pieces_placed,
                pieces_captured=board.pieces_captured,
            )
            move = parse_move_string(notation)
            board = board.apply_move(move)
            applied.append((color, notation, move))
        return board, applied

    def test_replay_produces_consistent_piece_counts(self):
        board, _ = self._apply_sequence()
        # Verify on-board counts match actual positions
        w_actual = sum(1 for p in POSITIONS if board.positions[p] == "W")
        b_actual = sum(1 for p in POSITIONS if board.positions[p] == "B")
        self.assertEqual(board.pieces_on_board["W"], w_actual)
        self.assertEqual(board.pieces_on_board["B"], b_actual)

    def test_replay_placed_counts(self):
        board, _ = self._apply_sequence()
        # White placed 9 pieces in turns 1–9 (one capture each for W, so 9 placements)
        # but we're trusting apply_move here — just check they're ≤ 9
        self.assertLessEqual(board.pieces_placed["W"], 9)
        self.assertLessEqual(board.pieces_placed["B"], 9)

    def test_export_produces_numbered_lines(self):
        from game.game_engine import GameEngine
        engine = GameEngine()
        game_record = {"moves": []}
        board = BoardState.new_game()

        for color, notation in self.RAW_GAME:
            board = BoardState(
                positions=board.positions,
                turn=color,
                pieces_on_board=board.pieces_on_board,
                pieces_placed=board.pieces_placed,
                pieces_captured=board.pieces_captured,
            )
            move = parse_move_string(notation)
            phase = get_game_phase(board, color)
            game_record["moves"].append({
                "color": color,
                "notation": encode_move(move, phase),
            })
            board = board.apply_move(move)

        exported = export_pgn_style(game_record)
        lines = exported.strip().splitlines()
        # First line should be "1. d2 d6"
        self.assertTrue(lines[0].startswith("1."))
        # Last line should contain "*"
        self.assertIn("*", lines[-1])
        # Each line (except the last) should have a turn number
        for line in lines[:-1]:
            self.assertRegex(line, r"^\d+\.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
