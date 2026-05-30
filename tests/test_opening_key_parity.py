"""tests/test_opening_key_parity.py — Phase 4 parity: opening/trajectory keys.

Asserts the Rust opening-key generation (py_opening_key, py_transform_notation)
matches the Python reference (board_symmetry.canonical_sequence / .transform_
notation) for realistic move-notation sequences. Skipped when nmm_core absent.
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os
import random
import sys as _sys
import types as _types

import pytest

nmm_core = pytest.importorskip("nmm_core")

from game.board import POSITIONS, ADJACENCY

_AI_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "ai")

if "ai" not in _sys.modules:
    _pkg = _types.ModuleType("ai")
    _pkg.__path__ = [_AI_DIR]
    _sys.modules["ai"] = _pkg


def _load(name, filename):
    spec = _ilu.spec_from_file_location(name, _os.path.join(_AI_DIR, filename))
    mod = _ilu.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_bs = _load("ai.board_symmetry", "board_symmetry.py")


def _random_notation_sequence(rng: random.Random, length: int):
    """A plausible move sequence: placements first, then adjacency moves +
    optional captures (using real position labels and adjacency)."""
    seq = []
    for _ in range(length):
        kind = rng.random()
        if kind < 0.5:  # placement
            seq.append(rng.choice(POSITIONS))
        elif kind < 0.85:  # movement along a real edge
            src = rng.choice(POSITIONS)
            dst = rng.choice(ADJACENCY[src])
            seq.append(f"{src}-{dst}")
        else:  # movement + capture
            src = rng.choice(POSITIONS)
            dst = rng.choice(ADJACENCY[src])
            cap = rng.choice(POSITIONS)
            seq.append(f"{src}-{dst}x{cap}")
    return seq


def test_opening_key_parity():
    rng = random.Random(4040)
    for _ in range(400):
        length = rng.randint(1, 10)
        seq = _random_notation_sequence(rng, length)

        py_seq, py_idx = _bs.canonical_sequence(seq)
        py_key = "|".join(py_seq)
        rust_key, rust_idx = nmm_core.py_opening_key(seq, len(seq))
        assert (rust_key, rust_idx) == (py_key, py_idx), (
            f"opening key mismatch seq={seq} py=({py_key},{py_idx}) "
            f"rust=({rust_key},{rust_idx})"
        )


def test_opening_key_depth_prefix():
    """Passing depth < len must canonicalize only the prefix."""
    rng = random.Random(5050)
    for _ in range(200):
        seq = _random_notation_sequence(rng, rng.randint(3, 12))
        depth = rng.randint(1, len(seq))
        prefix = seq[:depth]
        py_seq, py_idx = _bs.canonical_sequence(prefix)
        py_key = "|".join(py_seq)
        rust_key, rust_idx = nmm_core.py_opening_key(seq, depth)
        assert (rust_key, rust_idx) == (py_key, py_idx), (
            f"prefix key mismatch seq={seq} depth={depth} "
            f"py=({py_key},{py_idx}) rust=({rust_key},{rust_idx})"
        )


def test_transform_notation_formats():
    rng = random.Random(6060)
    samples = []
    for _ in range(150):
        src = rng.choice(POSITIONS)
        dst = rng.choice(ADJACENCY[src])
        cap = rng.choice(POSITIONS)
        samples += [src, f"{src}-{dst}", f"{src}-{dst}x{cap}", f"{src}x{cap}"]
    for note in samples:
        for idx in range(8):
            assert nmm_core.py_transform_notation(note, idx) == _bs.transform_notation(note, idx)
