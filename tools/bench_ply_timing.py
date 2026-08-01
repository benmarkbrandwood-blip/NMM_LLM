"""tools/bench_ply_timing.py — Profile Rust iterative deepening per-ply timing.

Runs py_search_root_scored at each individual depth 1..MAX_DEPTH on several
representative positions, then shows incremental time and nodes per depth so
bottlenecks in evaluate_v2 / move ordering are visible.

Usage:
    .venv/bin/python tools/bench_ply_timing.py
    .venv/bin/python tools/bench_ply_timing.py --max-depth 10 --position midgame
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nmm_core as _rc
from ai.native_core import RUST_AVAILABLE

if not RUST_AVAILABLE:
    sys.exit("nmm_core not available — run scripts/build_rust.sh first")

# ---------------------------------------------------------------------------
# Representative positions (white, black, white_placed, black_placed, stm)
# ---------------------------------------------------------------------------
POSITIONS = {
    "empty": (0, 0, 0, 0, 0),  # placement, board empty
    "early": (
        # White: a7,d7,g4 (0,1,3)  Black: g7,d1,a1 (2,5,6)  — 3 placed each
        (1 << 0) | (1 << 1) | (1 << 3),
        (1 << 2) | (1 << 5) | (1 << 6),
        3, 3, 0,
    ),
    "midgame": (
        # White: a7,d7,g7,g4,d5,b4 (0,1,2,3,17,15)
        # Black: g1,d1,a1,a4,d3,f4 (4,5,6,7,21,11)
        # 6 placed each, placement phase
        (1<<0)|(1<<1)|(1<<2)|(1<<3)|(1<<17)|(1<<15),
        (1<<4)|(1<<5)|(1<<6)|(1<<7)|(1<<21)|(1<<11),
        6, 6, 0,
    ),
    "movephase": (
        # 9 placed each — move phase
        # White: a7,d7,g7,g4,g1,d1,b6,d5,f4 (0,1,2,3,4,5,8,17,11)
        # Black: a1,a4,b4,b2,d2,f2,d6,f6,e4 (6,7,15,14,13,12,9,10,19)
        (1<<0)|(1<<1)|(1<<2)|(1<<3)|(1<<4)|(1<<5)|(1<<8)|(1<<17)|(1<<11),
        (1<<6)|(1<<7)|(1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<9)|(1<<10)|(1<<19),
        9, 9, 0,
    ),
}


def bench_position(name: str, pos: tuple, max_depth: int, warmup: int = 1) -> None:
    white, black, wp, bp, stm = pos
    print(f"\n{'='*60}")
    print(f"Position: {name}  wp={wp} bp={bp} stm={'W' if stm==0 else 'B'}")
    print(f"{'Depth':>5}  {'Nodes':>10}  {'Time(s)':>8}  {'kN/s':>8}  {'ΔNodes':>10}  {'ΔTime':>8}")
    print(f"{'-'*65}")

    prev_nodes = 0
    prev_time  = 0.0

    for d in range(1, max_depth + 1):
        # Warmup pass to prime TT/caches
        for _ in range(warmup):
            _rc.py_search_root_scored(
                white, black, wp, bp, stm,
                max_depth=d,
                time_limit_ms=300_000,
            )

        # Timed pass (fresh TT each time for fair comparison)
        t0 = time.perf_counter()
        nodes, depth_reached, moves = _rc.py_search_root_scored(
            white, black, wp, bp, stm,
            max_depth=d,
            time_limit_ms=300_000,
        )
        elapsed = time.perf_counter() - t0

        knps      = nodes / elapsed / 1000 if elapsed > 0 else 0
        d_nodes   = nodes - prev_nodes
        d_time    = elapsed - prev_time
        bf_est    = (d_nodes / max(1, prev_nodes - (prev_nodes - (nodes - prev_nodes)))) if prev_nodes > 0 else 0

        print(f"{d:>5}  {nodes:>10,}  {elapsed:>8.3f}  {knps:>8.0f}  {d_nodes:>10,}  {d_time:>8.3f}")

        prev_nodes = nodes
        prev_time  = elapsed

        if depth_reached < d:
            print(f"       [aborted at depth {depth_reached} — position too complex for this budget]")
            break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--position", choices=list(POSITIONS.keys()) + ["all"], default="all")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup passes per depth (TT pre-fill)")
    args = ap.parse_args()

    positions = POSITIONS if args.position == "all" else {args.position: POSITIONS[args.position]}

    print(f"Rust ply timing benchmark  max_depth={args.max_depth}  warmup={args.warmup}")
    for name, pos in positions.items():
        bench_position(name, pos, args.max_depth, args.warmup)

    print()


if __name__ == "__main__":
    main()
