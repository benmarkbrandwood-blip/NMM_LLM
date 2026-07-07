#!/usr/bin/env python3
"""tools/puzzle_generator.py — CLI for generating NMM endgame puzzles.

Usage examples:

  # Generate forever from random databases (all sizes), 1–2 winning moves, depth 3–5
  .venv/bin/python tools/puzzle_generator.py --random-db

  # Same, but stop after 100 puzzles
  .venv/bin/python tools/puzzle_generator.py --random-db --count 100

  # Specific database, random side and depth
  .venv/bin/python tools/puzzle_generator.py --db endgame_5_4.wdl

  # All random, unique-move only (max 1 winning first move)
  .venv/bin/python tools/puzzle_generator.py --random-db --max-winning-moves 1

  # White wins in 5, from a specific large table, with more sampling effort
  .venv/bin/python tools/puzzle_generator.py --db endgame_6_4.wdl --side W --depth 5 --attempts 20000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from ai.puzzle_search import generate_puzzle, load_puzzle_db

_DB_DIR = _ROOT / "data" / "endgame"
_OUT_DIR = _ROOT / "data" / "puzzles" / "endgame"


def _parse_nw_nb(db_name: str) -> tuple[int, int]:
    stem = db_name.replace("endgame_", "").replace(".wdl", "")
    nW, nB = map(int, stem.split("_"))
    return nW, nB


def _all_db_files() -> list[str]:
    return sorted(f.name for f in _DB_DIR.glob("endgame_*.wdl"))


def main() -> None:
    db_choices = _all_db_files()
    if not db_choices:
        print(f"ERROR: no endgame_*.wdl files found in {_DB_DIR}", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="NMM endgame puzzle generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db", choices=db_choices + ["random"], default="random",
        metavar="FILE",
        help="WDL database file from data/endgame/ (default: random each puzzle)",
    )
    parser.add_argument(
        "--random-db", action="store_true",
        help="Pick a new random database (all sizes) for every puzzle attempt "
             "(overrides --db, enables continuous cross-table generation)",
    )
    parser.add_argument(
        "--side", choices=["W", "B", "random"], default="random",
        help="Winning side (default: random)",
    )
    parser.add_argument(
        "--depth", choices=["3", "4", "5", "6", "7", "random"], default="random",
        help="Target win-depth in winning-side moves — 3–7, or random (default: random 3–7)",
    )
    parser.add_argument(
        "--max-winning-moves", type=int, default=2, metavar="N",
        help="Reject positions with more than N winning first moves (default: 2)",
    )
    parser.add_argument(
        "--count", type=int, default=0,
        help="Puzzles to generate; 0 = run forever (default: 0)",
    )
    parser.add_argument(
        "--attempts", type=int, default=5000,
        help="Positions sampled per puzzle attempt (default: 5000); "
             "increase for large or sparse tables",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help=f"Output directory (default: {_OUT_DIR.relative_to(_ROOT)})",
    )
    parser.add_argument(
        "--print", action="store_true", dest="print_json",
        help="Print each puzzle JSON to stdout",
    )
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else _OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # DB cache: (nW, nB, depth) -> EndgameSolvedDB
    # Avoids reloading the same table set for repeated (db, depth) combinations.
    db_cache: dict[tuple[int, int, int], object] = {}

    def get_db(nW: int, nB: int, depth: int):
        key = (nW, nB, depth)
        if key not in db_cache:
            print(f"  [load] endgame_{nW}_{nB}.wdl + adjacent tables for depth={depth} ...")
            db_cache[key] = load_puzzle_db(_DB_DIR, nW, nB, max_depth=depth)
        return db_cache[key]

    generated = 0
    attempts_total = 0
    run_forever = (args.count == 0)
    t_start = time.time()

    print(f"Puzzle generator started — saving to {out_dir.relative_to(_ROOT)}")
    print(f"Settings: max-winning-moves={args.max_winning_moves}  "
          f"attempts-per-puzzle={args.attempts}  "
          f"count={'∞' if run_forever else args.count}")
    print()

    puzzle_num = 0
    while run_forever or puzzle_num < args.count:
        puzzle_num += 1

        # Resolve db, side, depth for this puzzle
        if args.random_db or args.db == "random":
            db_file = random.choice(db_choices)
        else:
            db_file = args.db

        side  = random.choice(["W", "B"]) if args.side  == "random" else args.side
        depth = random.choice([3, 4, 5, 6, 7])  if args.depth == "random" else int(args.depth)
        nW, nB = _parse_nw_nb(db_file)

        label = f"[{'∞' if run_forever else f'{puzzle_num}/{args.count}'}]"
        print(f"{label}  db=endgame_{nW}_{nB}.wdl  side={side}  depth={depth}  "
              f"max-winning={args.max_winning_moves}")

        db = get_db(nW, nB, depth)

        if db._tables.get((nW, nB)) is None:
            print(f"  SKIP — table ({nW},{nB}) not found in {_DB_DIR}")
            continue

        t0 = time.time()
        puzzle = generate_puzzle(
            db, nW, nB, side, depth,
            max_attempts=args.attempts,
            max_winning_moves=args.max_winning_moves,
        )
        elapsed = time.time() - t0
        attempts_total += args.attempts

        if puzzle is None:
            print(f"  No qualifying puzzle found in {elapsed:.1f}s "
                  f"(tried {args.attempts} positions) — skipping to next")
            # Don't count this toward --count
            puzzle_num -= 1
            continue

        d = puzzle.to_dict()
        d["created_at"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
        n_winning = len(puzzle.legal_moves) - len(puzzle.drawing_moves) - len(puzzle.losing_moves)
        print(f"  FOUND  id={puzzle.puzzle_id}  score={puzzle.hardness_score}  "
              f"time={elapsed:.1f}s")
        print(f"  goal='{puzzle.goal}'")
        print(f"  best={puzzle.best_move}  line={' '.join(puzzle.solution_line)}")
        print(f"  legal={len(puzzle.legal_moves)}  "
              f"winning={n_winning}  "
              f"drawing={len(puzzle.drawing_moves)}  "
              f"losing={len(puzzle.losing_moves)}")
        print(f"  tags={puzzle.tags}")

        if args.print_json:
            print(json.dumps(d, indent=2))

        out_file = out_dir / f"{puzzle.puzzle_id}.json"
        out_file.write_text(json.dumps(d, indent=2))
        print(f"  Saved → {out_file.relative_to(_ROOT)}")
        generated += 1

        total_time = time.time() - t_start
        print(f"  Total: {generated} puzzles in {total_time/60:.1f} min  "
              f"({attempts_total:,} positions sampled)")
        print()

    print(f"Done. {generated} puzzles generated.")


if __name__ == "__main__":
    main()
