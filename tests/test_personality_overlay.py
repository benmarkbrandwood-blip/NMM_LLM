"""tests/test_personality_overlay.py — Hybrid 2+4 personality overlay.

Tests:
  1. personality_blend=0 leaves scored list unchanged.
  2. With personality_blend=50, a move with a strong H1 signal is preferred
     over a v2-equal move when both are within the personality window.
  3. A move clearly outside the window is NOT influenced by H1.
  4. Terminal/DB scores (abs >= INF//2) are never modified.
"""
from __future__ import annotations

import pytest
from ai.game_ai import GameAI
from ai.heuristics import HeuristicWeights, INF, DEFAULT_WEIGHTS
from game.board import BoardState, POSITIONS


def _make_ai(personality_blend: int = 0, personality_window: int = 8,
             use_v2: bool = True) -> GameAI:
    w = HeuristicWeights(
        personality_blend=personality_blend,
        personality_window=personality_window,
    )
    ai = GameAI("W", weights=w)
    ai.use_v2_heuristics = use_v2
    ai._opp_last_weak = False
    return ai


def _start_board() -> BoardState:
    return BoardState.new_game()


# ── helpers ──────────────────────────────────────────────────────────────────

def _fake_scored(scores: list[int]) -> list[tuple[dict, int]]:
    """Build a dummy scored list; moves are placement dicts (from=None) with distinct 'to' squares."""
    squares = POSITIONS[:len(scores)]
    return [({"from": None, "to": sq}, sc) for sq, sc in zip(squares, scores)]


# ── Test 1: blend=0 → no change ──────────────────────────────────────────────

def test_blend_zero_is_noop():
    ai = _make_ai(personality_blend=0)
    board = _start_board()
    scored = _fake_scored([100, 80, 60])
    result = ai._apply_personality_overlay(scored, board)
    assert result == scored, "personality_blend=0 must leave scores unchanged"


# ── Test 2: blend=50 selects H1-preferred move within window ─────────────────

def test_blend_50_prefers_mill_close():
    """Set up a board where White can close a mill to one square and not to
    another.  When both moves have equal v2 scores the overlay should prefer
    the mill-closing one."""
    from game.board import MILLS

    # Build a position where White has two pieces in a mill and can close it.
    # Use the outer ring: a1-d1-g1.  Place W on a1, d1; candidate close is g1.
    positions = {p: "" for p in POSITIONS}
    positions["a1"] = "W"
    positions["d1"] = "W"
    # b2, b4 — unrelated white pieces so we have 9+ placements (avoid piece count guard)
    positions["b2"] = "W"
    positions["b4"] = "W"
    positions["d2"] = "W"
    # Black pieces: scattered, no immediate threats
    positions["g4"] = "B"
    positions["a4"] = "B"
    positions["d6"] = "B"
    positions["g7"] = "B"

    board = BoardState(
        positions=positions,
        pieces_placed={"W": 9, "B": 9},
        pieces_on_board={"W": 5, "B": 4},
        pieces_captured={"W": 0, "B": 0},
        turn="W",
    )

    ai = _make_ai(personality_blend=50, personality_window=100)  # wide window so both moves qualify

    # Two candidate moves: close mill (d1→g1 would need from, use placement context)
    # Simplest: use moves that board.apply_move can handle in move phase.
    # Move W d2→g1 closes mill (a1,d1,g1) vs W d2→d3 (neutral).
    move_close = {"from": "d2", "to": "g1"}
    move_neutral = {"from": "d2", "to": "d3"}

    # Give them equal v2 scores so pure v2 is indifferent.
    scored = [(move_close, 200), (move_neutral, 200)]
    result = ai._apply_personality_overlay(scored, board)

    best = max(result, key=lambda x: x[1])
    assert best[0]["to"] == "g1", (
        "With equal v2 scores, mill-close move should be preferred by H1 overlay"
    )


# ── Test 3: move outside window is not promoted above best v2 move ────────────

def test_out_of_window_move_unaffected():
    ai = _make_ai(personality_blend=80, personality_window=10)
    board = _start_board()

    # Scores: best=1000, in-window=950 (5% gap), out-of-window=100 (90% gap)
    scored = [
        ({"from": None, "to": POSITIONS[0]}, 1000),
        ({"from": None, "to": POSITIONS[1]}, 950),
        ({"from": None, "to": POSITIONS[2]}, 100),   # clearly outside window
    ]
    result = ai._apply_personality_overlay(scored, board)

    result_by_sq = {m["to"]: s for m, s in result}
    # The out-of-window move must not change score
    assert result_by_sq[POSITIONS[2]] == 100, "Out-of-window move must not be modified"


# ── Test 4: terminal scores are never modified ────────────────────────────────

def test_terminal_scores_unmodified():
    ai = _make_ai(personality_blend=100, personality_window=100)
    board = _start_board()

    terminal_score = INF - 1
    scored = [
        ({"from": None, "to": POSITIONS[0]}, terminal_score),
        ({"from": None, "to": POSITIONS[1]}, 50),
    ]
    result = ai._apply_personality_overlay(scored, board)
    result_by_sq = {m["to"]: s for m, s in result}
    assert result_by_sq[POSITIONS[0]] == terminal_score, "Terminal scores must not be modified"


# ── Test 5: blend=0 with v1 heuristics is a no-op ────────────────────────────

def test_v1_heuristics_bypass():
    ai = _make_ai(personality_blend=80, personality_window=20, use_v2=False)
    board = _start_board()

    scored = [({"to": POSITIONS[0]}, 100), ({"to": POSITIONS[1]}, 100)]
    result = ai._apply_personality_overlay(scored, board)
    assert result == scored, "_apply_personality_overlay must be no-op when use_v2_heuristics=False"
