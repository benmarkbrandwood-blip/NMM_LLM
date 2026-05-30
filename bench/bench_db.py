"""bench/bench_db.py — Key-generation throughput: Rust vs Python.

Benchmarks the two hot key-generation paths used by the DB layer:
  1. FullGame DB 9-byte canonical key (canonicalize board + pack bits)
  2. Opening/trajectory canonical-sequence key

Both are computed many times over randomized inputs; the Rust path is compared
to the Python reference (board_symmetry + fullgame_db._encode_canonical) for
keys/sec. Verifies output equality on a sample before timing.

Run:
    python bench/bench_db.py
    python bench/bench_db.py --iters 50000
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import os
import random
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.board import POSITIONS, ADJACENCY  # noqa: E402

_AI_DIR = os.path.join(str(Path(__file__).resolve().parent.parent), "ai")

if "ai" not in sys.modules:
    _pkg = types.ModuleType("ai")
    _pkg.__path__ = [_AI_DIR]
    sys.modules["ai"] = _pkg


def _load(name, filename):
    spec = _ilu.spec_from_file_location(name, os.path.join(_AI_DIR, filename))
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_bs = _load("ai.board_symmetry", "board_symmetry.py")
_fg = _load("ai.fullgame_db", "fullgame_db.py")

try:
    import nmm_core  # type: ignore

    RUST = True
except Exception:
    nmm_core = None
    RUST = False

_IDX = {p: i for i, p in enumerate(POSITIONS)}


def _random_board24(rng: random.Random) -> str:
    return "".join(rng.choice(".WB") for _ in range(24))


def _py_fullgame_key(board24: str, turn: str, pw: int, pb: int) -> bytes:
    canon, _ = _bs.canonical_board_str(board24)
    return _fg._encode_canonical(canon, turn, pw, pb)


def _rust_fullgame_key(board24: str, turn: str, pw: int, pb: int) -> bytes:
    white = sum(1 << i for i, ch in enumerate(board24) if ch == "W")
    black = sum(1 << i for i, ch in enumerate(board24) if ch == "B")
    stm = 0 if turn == "W" else 1
    return bytes(nmm_core.py_db_key(white, black, stm, pw, pb))


def _random_sequence(rng: random.Random, length: int):
    seq = []
    for _ in range(length):
        src = rng.choice(POSITIONS)
        dst = rng.choice(ADJACENCY[src])
        seq.append(rng.choice([src, f"{src}-{dst}"]))
    return seq


def _time(fn, data) -> float:
    t0 = time.perf_counter()
    for args in data:
        fn(*args)
    return time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=1234567)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"NMM key-generation benchmark — iters={args.iters}")
    print(f"Rust extension available: {RUST}")

    # ── FullGame DB key ───────────────────────────────────────────────────────
    boards = [
        (_random_board24(rng), rng.choice("WB"), rng.randint(0, 9), rng.randint(0, 9))
        for _ in range(args.iters)
    ]
    if RUST:
        # verify equality on the first 200 samples
        for b24, t, pw, pb in boards[:200]:
            assert _py_fullgame_key(b24, t, pw, pb) == _rust_fullgame_key(b24, t, pw, pb)

    print("-" * 60)
    py_dt = _time(_py_fullgame_key, boards)
    print(f"FullGame key | Python: {args.iters/py_dt:>12,.0f} keys/sec  ({py_dt:6.3f}s)")
    if RUST:
        rust_dt = _time(_rust_fullgame_key, boards)
        print(f"FullGame key | Rust  : {args.iters/rust_dt:>12,.0f} keys/sec  ({rust_dt:6.3f}s)")
        if rust_dt > 0:
            print(f"FullGame key | speedup: {py_dt/rust_dt:5.1f}x")

    # ── Opening / trajectory key ──────────────────────────────────────────────
    seqs = [(_random_sequence(rng, rng.randint(2, 10)),) for _ in range(args.iters)]

    def _py_open(seq):
        s, _ = _bs.canonical_sequence(seq)
        return "|".join(s)

    def _rust_open(seq):
        return nmm_core.py_opening_key(seq, len(seq))

    if RUST:
        for (seq,) in seqs[:200]:
            ps, _ = _bs.canonical_sequence(seq)
            rk, _ = nmm_core.py_opening_key(seq, len(seq))
            assert "|".join(ps) == rk

    print("-" * 60)
    py_dt = _time(lambda s: _py_open(s), seqs)
    print(f"Opening  key | Python: {args.iters/py_dt:>12,.0f} keys/sec  ({py_dt:6.3f}s)")
    if RUST:
        rust_dt = _time(lambda s: _rust_open(s), seqs)
        print(f"Opening  key | Rust  : {args.iters/rust_dt:>12,.0f} keys/sec  ({rust_dt:6.3f}s)")
        if rust_dt > 0:
            print(f"Opening  key | speedup: {py_dt/rust_dt:5.1f}x")
    else:
        print("Rust core not built — run scripts/build_rust.sh to enable.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
