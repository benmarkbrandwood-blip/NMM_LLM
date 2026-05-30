"""tests/test_heuristics_parity.py — Phase 2 parity: tactics + mill helpers.

The full Python heuristic (`ai/heuristics.py::evaluate`) blends ~50 float-scaled
phase-conditional terms and is intentionally NOT reproduced byte-for-byte in
Rust (see docs/RUST_INTEGRATION_PLAN.md risk register). What we CAN assert with
exact parity are the deterministic integer/geometric primitives the Rust search
relies on:

  * py_count_mills / py_forms_mill  vs the Python mill geometry
  * py_immediate_threats            vs the RAW geometric threat-square set
    (the opponent 2-config closing squares reachable in one move — i.e. the
    `_immediate_mill_threats` body BEFORE its B-66 / placement-fork carveouts,
    which are AI-policy heuristics layered on top in game_ai.py).

Skipped when the `nmm_core` extension is not built.
"""

from __future__ import annotations

import random

import pytest

nmm_core = pytest.importorskip("nmm_core")

from game.board import POSITIONS, MILLS, ADJACENCY, BoardState
from game.rules import get_game_phase

_IDX = {p: i for i, p in enumerate(POSITIONS)}


def _bits(positions: dict, color: str) -> int:
    return sum(1 << _IDX[p] for p, v in positions.items() if v == color)


def _random_board(rng: random.Random) -> BoardState:
    phase_choice = rng.choice(["place", "move"])
    positions = {p: "" for p in POSITIONS}
    squares = POSITIONS[:]
    rng.shuffle(squares)
    idx = 0
    if phase_choice == "place":
        wp = rng.randint(0, 8)
        bp = rng.randint(0, 8)
        for _ in range(wp):
            positions[squares[idx]] = "W"; idx += 1
        for _ in range(bp):
            positions[squares[idx]] = "B"; idx += 1
        placed = {"W": wp, "B": bp}
        w_on, b_on = wp, bp
    else:
        w_on = rng.randint(3, 9)
        b_on = rng.randint(3, 9)
        for _ in range(w_on):
            positions[squares[idx]] = "W"; idx += 1
        for _ in range(b_on):
            positions[squares[idx]] = "B"; idx += 1
        placed = {"W": 9, "B": 9}
    return BoardState(
        positions=positions,
        turn=rng.choice(["W", "B"]),
        pieces_on_board={"W": w_on, "B": b_on},
        pieces_placed=placed,
        pieces_captured={"W": 0, "B": 0},
        hash_key=0,
    )


def _raw_immediate_threats(board: BoardState) -> set[str]:
    """The geometric core of game_ai._immediate_mill_threats: closing squares of
    opponent 2-configs reachable in one move, WITHOUT the policy carveouts.

    Matches Rust tactics::immediate_mill_threats:
      place / fly -> any empty closing square; move -> needs an opponent piece
      adjacent to the closing square that is not itself inside the mill."""
    opp = "B" if board.turn == "W" else "W"
    opp_phase = get_game_phase(board, opp)
    threats: set[str] = set()
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(opp) == 2 and vals.count("") == 1:
            empty = next(p for p in mill if board.positions[p] == "")
            if opp_phase in ("place", "fly"):
                threats.add(empty)
            else:  # move
                mill_set = set(mill)
                if any(
                    board.positions[nb] == opp
                    for nb in ADJACENCY[empty]
                    if nb not in mill_set
                ):
                    threats.add(empty)
    return threats


def test_immediate_threats_parity():
    rng = random.Random(808)
    for _ in range(500):
        b = _random_board(rng)
        white = _bits(b.positions, "W")
        black = _bits(b.positions, "B")
        stm = 0 if b.turn == "W" else 1

        py_set = {_IDX[p] for p in _raw_immediate_threats(b)}
        mask = nmm_core.py_immediate_threats(
            white, black, b.pieces_placed["W"], b.pieces_placed["B"], stm
        )
        rust_set = {i for i in range(24) if mask & (1 << i)}
        assert rust_set == py_set, (
            f"threat mismatch board={b.to_fen_string()} py={py_set} rust={rust_set}"
        )


def test_count_mills_parity():
    rng = random.Random(303)
    for _ in range(400):
        b = _random_board(rng)
        white = _bits(b.positions, "W")
        black = _bits(b.positions, "B")
        for color, cidx in (("W", 0), ("B", 1)):
            py = sum(1 for m in MILLS if all(b.positions[p] == color for p in m))
            assert nmm_core.py_count_mills(white, black, cidx) == py


def test_forms_mill_parity():
    """py_forms_mill(white, black, square, color): is there a mill through
    `square` that is FULLY owned by `color` in the current bitboard? (Static
    check over the existing bits — it does not add `square` to color.)"""
    rng = random.Random(123)
    for _ in range(400):
        b = _random_board(rng)
        white = _bits(b.positions, "W")
        black = _bits(b.positions, "B")
        for color, cidx in (("W", 0), ("B", 1)):
            sq_name = rng.choice(POSITIONS)
            sq = _IDX[sq_name]
            # Reference matches Rust: a mill containing `square`, all three owned
            # by `color` in the CURRENT positions.
            py = any(
                sq_name in m and all(b.positions[p] == color for p in m)
                for m in MILLS
            )
            rust = nmm_core.py_forms_mill(white, black, sq, cidx)
            assert rust == py, (
                f"forms_mill mismatch board={b.to_fen_string()} sq={sq_name} "
                f"color={color} py={py} rust={rust}"
            )
