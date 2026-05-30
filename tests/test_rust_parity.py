"""tests/test_rust_parity.py — Parity tests: Rust nmm_core vs Python reference.

These tests are skipped automatically if the compiled `nmm_core` extension is
not importable, so the Python-only test suite is unaffected. When the extension
is present, they assert that the Rust primitives produce results identical to
the Python engine for board parsing, legal-move generation, mill detection,
phase logic, canonicalization, and DB key encoding.

Build the extension first:
    cd native/nmm_core && maturin build --release
    pip install target/wheels/nmm_core-*.whl
"""

from __future__ import annotations

import random

import pytest

nmm_core = pytest.importorskip("nmm_core")

from game.board import POSITIONS, BoardState
from game.rules import get_all_legal_moves, get_game_phase

# The `ai` package __init__ imports chromadb (a heavy optional dep). Load the
# two pure-Python reference modules we need directly by path to avoid that.
import importlib.util as _ilu
import os as _os
import sys as _sys
import types as _types

_AI_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "ai")


def _load_module(name: str, filename: str):
    spec = _ilu.spec_from_file_location(name, _os.path.join(_AI_DIR, filename))
    mod = _ilu.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Build a lightweight `ai` package shell so the pure-Python reference modules'
# relative imports (e.g. fullgame_db's `from .board_symmetry import ...`)
# resolve without triggering ai/__init__.py's chromadb import.
if "ai" not in _sys.modules:
    _ai_pkg = _types.ModuleType("ai")
    _ai_pkg.__path__ = [_AI_DIR]
    _sys.modules["ai"] = _ai_pkg

_board_symmetry = _load_module("ai.board_symmetry", "board_symmetry.py")
_fullgame_db = _load_module("ai.fullgame_db", "fullgame_db.py")
canonical_board_str = _board_symmetry.canonical_board_str

_IDX = {p: i for i, p in enumerate(POSITIONS)}


# ── conversion helpers ────────────────────────────────────────────────────────

def _bits(positions: dict, color: str) -> int:
    bits = 0
    for p, v in positions.items():
        if v == color:
            bits |= 1 << _IDX[p]
    return bits


def _board_to_rust_args(b: BoardState):
    """(white, black, white_placed, black_placed, stm) for the Rust binding."""
    white = _bits(b.positions, "W")
    black = _bits(b.positions, "B")
    stm = 0 if b.turn == "W" else 1
    return white, black, b.pieces_placed["W"], b.pieces_placed["B"], stm


def _py_move_to_tuple(mv: dict):
    """Python move dict -> (from_idx|None, to_idx, cap_idx|None)."""
    f = None if mv["from"] is None else _IDX[mv["from"]]
    t = _IDX[mv["to"]]
    c = None if mv.get("capture") is None else _IDX[mv["capture"]]
    return (f, t, c)


# ── position generators ───────────────────────────────────────────────────────

def _random_positions(rng: random.Random):
    """Generate a plausible random BoardState across all phases."""
    phase_choice = rng.choice(["place", "move"])
    positions = {p: "" for p in POSITIONS}

    if phase_choice == "place":
        # Partial placement: 0..8 placed each, on board == placed (no captures yet).
        wp = rng.randint(0, 8)
        bp = rng.randint(0, 8)
        squares = POSITIONS[:]
        rng.shuffle(squares)
        idx = 0
        for _ in range(wp):
            positions[squares[idx]] = "W"; idx += 1
        for _ in range(bp):
            positions[squares[idx]] = "B"; idx += 1
        placed = {"W": wp, "B": bp}
        w_on, b_on = wp, bp
    else:
        # Movement / fly: both placed 9; 3..9 pieces each on the board.
        w_on = rng.randint(3, 9)
        b_on = rng.randint(3, 9)
        squares = POSITIONS[:]
        rng.shuffle(squares)
        idx = 0
        for _ in range(w_on):
            positions[squares[idx]] = "W"; idx += 1
        for _ in range(b_on):
            positions[squares[idx]] = "B"; idx += 1
        placed = {"W": 9, "B": 9}

    turn = rng.choice(["W", "B"])
    b = BoardState(
        positions=positions,
        turn=turn,
        pieces_on_board={"W": w_on, "B": b_on},
        pieces_placed=placed,
        pieces_captured={"W": 0, "B": 0},
        hash_key=0,
    )
    return b


# ── tests ─────────────────────────────────────────────────────────────────────

def test_legal_moves_parity():
    rng = random.Random(1234)
    for _ in range(400):
        b = _random_positions(rng)
        # Skip terminal/blocked degenerate states where neither side can act:
        white, black, wp, bp, stm = _board_to_rust_args(b)

        py_moves = {_py_move_to_tuple(m) for m in get_all_legal_moves(b)}
        rust_moves = set(nmm_core.py_legal_moves(white, black, wp, bp, stm))

        assert rust_moves == py_moves, (
            f"legal-move mismatch\nboard={b.to_fen_string()}\n"
            f"py-only={py_moves - rust_moves}\nrust-only={rust_moves - py_moves}"
        )


def test_phase_parity():
    rng = random.Random(99)
    for _ in range(300):
        b = _random_positions(rng)
        won = b.pieces_on_board["W"]
        bon = b.pieces_on_board["B"]
        for color, cidx in (("W", 0), ("B", 1)):
            py_phase = get_game_phase(b, color)
            py_code = {"place": 0, "move": 1, "fly": 2}[py_phase]
            rust_code = nmm_core.py_detect_phase(
                b.pieces_placed["W"], b.pieces_placed["B"], won, bon, cidx
            )
            assert rust_code == py_code, (
                f"phase mismatch color={color} board={b.to_fen_string()} "
                f"py={py_code} rust={rust_code}"
            )


def test_mill_count_parity():
    rng = random.Random(7)
    from game.board import MILLS

    def py_count_mills(b, color):
        n = 0
        for mill in MILLS:
            if all(b.positions[p] == color for p in mill):
                n += 1
        return n

    for _ in range(300):
        b = _random_positions(rng)
        white = _bits(b.positions, "W")
        black = _bits(b.positions, "B")
        for color, cidx in (("W", 0), ("B", 1)):
            assert nmm_core.py_count_mills(white, black, cidx) == py_count_mills(b, color)


def test_canonical_board_str_parity():
    rng = random.Random(2024)
    for _ in range(300):
        b = _random_positions(rng)
        board24 = "".join(
            b.positions[p] if b.positions[p] else "." for p in POSITIONS
        )
        py_str, py_idx = canonical_board_str(board24)
        rust_str, rust_idx = nmm_core.py_canonical_board_str(board24)
        assert rust_str == py_str, f"canonical str mismatch {board24}: py={py_str} rust={rust_str}"
        assert rust_idx == py_idx, f"sym_idx mismatch {board24}: py={py_idx} rust={rust_idx}"


def test_db_key_parity():
    _encode_canonical = _fullgame_db._encode_canonical

    rng = random.Random(555)
    for _ in range(300):
        b = _random_positions(rng)
        board24 = "".join(
            b.positions[p] if b.positions[p] else "." for p in POSITIONS
        )
        canon, _ = canonical_board_str(board24)
        turn = b.turn
        pw = b.pieces_placed["W"]
        pb = b.pieces_placed["B"]
        py_key = _encode_canonical(canon, turn, pw, pb)

        white = _bits(b.positions, "W")
        black = _bits(b.positions, "B")
        stm = 0 if turn == "W" else 1
        rust_key = bytes(nmm_core.py_db_key(white, black, stm, pw, pb))
        assert rust_key == py_key, (
            f"db key mismatch board={board24} py={py_key.hex()} rust={rust_key.hex()}"
        )
