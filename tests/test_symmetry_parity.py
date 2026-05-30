"""tests/test_symmetry_parity.py — D4 symmetry parity: Rust vs Python.

Validates that the Rust D4 transform tables, canonicalization, and move-notation
transforms match the Python reference (`ai/board_symmetry.py`) exactly. Skipped
when the `nmm_core` extension is not built.
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os
import random
import sys as _sys
import types as _types

import pytest

nmm_core = pytest.importorskip("nmm_core")

from game.board import POSITIONS

_AI_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "ai")

if "ai" not in _sys.modules:
    _ai_pkg = _types.ModuleType("ai")
    _ai_pkg.__path__ = [_AI_DIR]
    _sys.modules["ai"] = _ai_pkg


def _load(name: str, filename: str):
    spec = _ilu.spec_from_file_location(name, _os.path.join(_AI_DIR, filename))
    mod = _ilu.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_bs = _load("ai.board_symmetry", "board_symmetry.py")

_IDX = {p: i for i, p in enumerate(POSITIONS)}


def test_apply_transform_parity():
    """Rust py_apply_transform(bits, idx) must permute bits the same way the
    Python _BOARD_PERM does (bit at old idx -> bit at perm[old idx])."""
    rng = random.Random(11)
    perms = _bs._BOARD_PERM
    for _ in range(500):
        bits = rng.getrandbits(24)
        for idx in range(8):
            perm = perms[idx]
            expected = 0
            for old in range(24):
                if bits & (1 << old):
                    expected |= 1 << perm[old]
            got = nmm_core.py_apply_transform(bits, idx)
            assert got == expected, f"transform {idx} bits={bits:#08x} got={got:#08x} exp={expected:#08x}"


def test_canonical_board_str_all_transforms():
    """For boards that are themselves a transform of a base, canonicalization
    must agree between Rust and Python including the chosen sym_idx."""
    rng = random.Random(42)
    for _ in range(500):
        chars = [rng.choice(".WB") for _ in range(24)]
        board24 = "".join(chars)
        py_str, py_idx = _bs.canonical_board_str(board24)
        rust_str, rust_idx = nmm_core.py_canonical_board_str(board24)
        assert (rust_str, rust_idx) == (py_str, py_idx), (
            f"board={board24} py=({py_str},{py_idx}) rust=({rust_str},{rust_idx})"
        )


def test_transform_notation_parity():
    rng = random.Random(2026)
    pos = POSITIONS
    notations = []
    # placements
    notations += list(pos)
    # movements + captures (random adjacency-free pairs are fine for the table test)
    for _ in range(200):
        a, b, c = rng.choice(pos), rng.choice(pos), rng.choice(pos)
        notations.append(f"{a}-{b}")
        notations.append(f"{a}-{b}x{c}")
        notations.append(f"{a}x{c}")

    for note in notations:
        for idx in range(8):
            py = _bs.transform_notation(note, idx)
            rust = nmm_core.py_transform_notation(note, idx)
            assert rust == py, f"notation={note} idx={idx} py={py} rust={rust}"


def test_canonical_key_invariant_under_d4():
    """py_canonical_key is a bitboard-pair lex-min canonicalization (distinct
    from the string-based canonical_board_str). Its defining property: every D4
    transform of a board must collapse to the same canonical key."""
    rng = random.Random(321)
    for _ in range(400):
        chars = [rng.choice(".WB") for _ in range(24)]
        white = sum(1 << i for i, ch in enumerate(chars) if ch == "W")
        black = sum(1 << i for i, ch in enumerate(chars) if ch == "B")

        base_key = nmm_core.py_canonical_key(white, black)
        for idx in range(8):
            tw = nmm_core.py_apply_transform(white, idx)
            tb = nmm_core.py_apply_transform(black, idx)
            assert nmm_core.py_canonical_key(tw, tb) == base_key, (
                f"canonical_key not D4-invariant at transform {idx}: "
                f"base={base_key} got={nmm_core.py_canonical_key(tw, tb)}"
            )
