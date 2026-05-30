"""bench/bench_search.py — Search throughput: Rust core vs pure-Python baseline.

Reports nodes/sec for the Rust negamax (nmm_core.py_get_best_move) and a
reference pure-Python negamax over the same set of positions, using the SAME
legal-move generator (game.rules) and a comparable integer evaluation so the
ratio reflects the engine speedup rather than algorithmic differences.

Run:
    python bench/bench_search.py
    python bench/bench_search.py --depth 5 --positions 6

The Python baseline is deliberately minimal (no TT / move ordering) — it exists
to give a representative "before" number for the hot loop, not to mirror the
production AI (which is coupled to optional DB/LLM infrastructure).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.board import POSITIONS, BoardState  # noqa: E402
from game.rules import get_all_legal_moves, is_terminal  # noqa: E402

try:
    import nmm_core  # type: ignore

    RUST = True
except Exception:
    nmm_core = None
    RUST = False

_IDX = {p: i for i, p in enumerate(POSITIONS)}


# ── pure-Python reference negamax (counts nodes) ──────────────────────────────

_NODES = 0


def _evaluate(board: BoardState) -> int:
    """Tiny material+mobility eval from board.turn's perspective (integer)."""
    me = board.turn
    opp = "B" if me == "W" else "W"
    diff = board.pieces_on_board[me] - board.pieces_on_board[opp]
    return 12 * diff


def _negamax(board: BoardState, depth: int, alpha: int, beta: int) -> int:
    global _NODES
    _NODES += 1
    terminal, winner = is_terminal(board)
    if terminal:
        if winner is None:
            return 0
        return 1_000_000 if winner == board.turn else -1_000_000
    if depth == 0:
        return _evaluate(board)
    moves = get_all_legal_moves(board)
    if not moves:
        return -1_000_000
    best = -10_000_000
    for mv in moves:
        score = -_negamax(board.apply_move(mv), depth - 1, -beta, -alpha)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


# ── position generation ───────────────────────────────────────────────────────

def _random_midgame(rng: random.Random) -> BoardState:
    """A movement-phase position with both sides placed and 4-8 pieces each."""
    positions = {p: "" for p in POSITIONS}
    squares = POSITIONS[:]
    rng.shuffle(squares)
    idx = 0
    w_on = rng.randint(4, 8)
    b_on = rng.randint(4, 8)
    for _ in range(w_on):
        positions[squares[idx]] = "W"; idx += 1
    for _ in range(b_on):
        positions[squares[idx]] = "B"; idx += 1
    return BoardState(
        positions=positions,
        turn="W",
        pieces_on_board={"W": w_on, "B": b_on},
        pieces_placed={"W": 9, "B": 9},
        pieces_captured={"W": 0, "B": 0},
        hash_key=0,
    )


def _bits(board: BoardState):
    white = sum(1 << _IDX[p] for p, v in board.positions.items() if v == "W")
    black = sum(1 << _IDX[p] for p, v in board.positions.items() if v == "B")
    stm = 0 if board.turn == "W" else 1
    return white, black, board.pieces_placed["W"], board.pieces_placed["B"], stm


# ── benchmark ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--positions", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260530)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    boards = [_random_midgame(rng) for _ in range(args.positions)]

    print(f"NMM search benchmark — depth={args.depth}, positions={args.positions}")
    print(f"Rust extension available: {RUST}")
    print("-" * 60)

    # ── Pure-Python baseline ──────────────────────────────────────────────────
    global _NODES
    _NODES = 0
    t0 = time.perf_counter()
    for b in boards:
        _negamax(b, args.depth, -10_000_000, 10_000_000)
    py_dt = time.perf_counter() - t0
    py_nodes = _NODES
    py_nps = py_nodes / py_dt if py_dt > 0 else 0.0
    print(f"Python negamax : {py_nodes:>10,} nodes  in {py_dt:7.3f}s  "
          f"= {py_nps:>12,.0f} nodes/sec")

    # ── Rust core ─────────────────────────────────────────────────────────────
    if RUST:
        # Rust runs its own iterative-deepening to the same max depth, with a
        # generous time budget so depth (not the clock) is the limiter.
        t0 = time.perf_counter()
        rust_nodes = 0
        for b in boards:
            white, black, wp, bp, stm = _bits(b)
            *_mv, nodes, _depth = nmm_core.py_search_stats(
                white, black, wp, bp, stm, args.depth, 60_000
            )
            rust_nodes += nodes
        rust_dt = time.perf_counter() - t0
        rust_nps = rust_nodes / rust_dt if rust_dt > 0 else 0.0
        print(f"Rust  negamax  : {rust_nodes:>10,} nodes  in {rust_dt:7.3f}s  "
              f"= {rust_nps:>12,.0f} nodes/sec")
        print("-" * 60)
        if py_nps > 0 and rust_nps > 0:
            print(f"Throughput speedup (nodes/sec): {rust_nps / py_nps:6.1f}x")
        if py_dt > 0 and rust_dt > 0:
            print(f"Wall-clock speedup (same depth): {py_dt / rust_dt:6.1f}x  "
                  f"(note: Rust has TT + move ordering; Python baseline does not)")
    else:
        print("Rust core not built — run scripts/build_rust.sh to enable.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
