#!/usr/bin/env python3
"""tools/malom_puzzle_generator.py — CLI for generating NMM midgame puzzles.

Uses the Malom ultra-strong perfect database to find movement-phase positions
with a forced win in 4–7 moves (distinctly deeper than the endgame puzzles).

Malom DB path is read from data/settings.json (malom_db_path key).

Usage examples:

  # Generate forever — random depth 4–7, random side
  .venv/bin/python tools/malom_puzzle_generator.py

  # Stop after 30 puzzles
  .venv/bin/python tools/malom_puzzle_generator.py --count 30

  # Win-in-5, Black to win
  .venv/bin/python tools/malom_puzzle_generator.py --depth 5 --side B --count 20

  # Unique-move only (max 1 winning first move), any depth
  .venv/bin/python tools/malom_puzzle_generator.py --max-winning-moves 1 --count 20

  # Allow larger piece counts (slower per puzzle due to hash init)
  .venv/bin/python tools/malom_puzzle_generator.py --max-pieces 9 --count 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from ai.malom_db import MalomDB
from ai.malom_puzzle_search import generate_malom_puzzle, prewarm_hash_cache

_OUT_DIR = _ROOT / "data" / "puzzles" / "malom"


def _load_settings() -> dict:
    p = _ROOT / "data" / "settings.json"
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NMM midgame puzzle generator (Malom perfect database)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--side", choices=["W", "B", "random"], default="random",
        help="Winning side (default: random)",
    )
    parser.add_argument(
        "--depth", type=int, choices=[0, 4, 5, 6, 7], default=0,
        metavar="N",
        help="Target win depth in winner moves: 4/5/6/7, or 0 = random (default: 0)",
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
        "--attempts", type=int, default=3000,
        help="Positions sampled per puzzle attempt (default: 3000)",
    )
    parser.add_argument(
        "--min-pieces", type=int, default=4, metavar="N",
        help="Minimum pieces per side (default: 4)",
    )
    parser.add_argument(
        "--max-pieces", type=int, default=7, metavar="N",
        help="Maximum pieces per side (default: 7; set higher for richer midgame)",
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

    settings = _load_settings()
    malom_path = settings.get("malom_db_path", "")
    if not malom_path:
        print("ERROR: malom_db_path not configured in data/settings.json", file=sys.stderr)
        print("  Configure it via the web UI Settings page.", file=sys.stderr)
        sys.exit(1)

    db = MalomDB(malom_path)
    if not db.is_available():
        print(f"ERROR: Malom DB unavailable at {malom_path!r}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out) if args.out else _OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Malom DB: {malom_path}")
    print(f"Settings: depth={'any 4-7' if args.depth==0 else args.depth}  "
          f"side={args.side}  max-winning={args.max_winning_moves}  "
          f"pieces={args.min_pieces}–{args.max_pieces}  "
          f"count={'∞' if args.count==0 else args.count}")
    print(f"Output: {out_dir.relative_to(_ROOT)}")
    print()

    # Pre-warm hash states
    print(f"Pre-warming Malom hash states for pieces 3–{args.max_pieces}…", end=" ", flush=True)
    t_warm = time.time()
    prewarm_hash_cache(args.max_pieces)
    print(f"done ({time.time()-t_warm:.1f}s)")
    print()

    generated = 0
    run_forever = (args.count == 0)
    puzzle_num = 0
    t_start = time.time()

    while run_forever or puzzle_num < args.count:
        puzzle_num += 1
        label = f"[{'∞' if run_forever else f'{puzzle_num}/{args.count}'}]"
        print(f"{label}  side={args.side}  depth={'any' if args.depth==0 else args.depth}  "
              f"max-winning={args.max_winning_moves}")

        t0 = time.time()
        puzzle = generate_malom_puzzle(
            db,
            winning_side=args.side,
            target_win_in=args.depth,
            max_attempts=args.attempts,
            max_winning_moves=args.max_winning_moves,
            min_pieces_per_side=args.min_pieces,
            max_pieces_per_side=args.max_pieces,
        )
        elapsed = time.time() - t0

        if puzzle is None:
            print(f"  No qualifying puzzle in {elapsed:.1f}s — retrying")
            puzzle_num -= 1
            continue

        d = puzzle.to_dict()
        d["created_at"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
        print(f"  FOUND  id={puzzle.puzzle_id}  goal={puzzle.goal}  "
              f"score={puzzle.hardness_score}  time={elapsed:.1f}s")
        print(f"  best={puzzle.best_move}  line={' '.join(puzzle.solution_line)}")
        print(f"  winning={puzzle.winning_moves}  tags={puzzle.tags}")

        if args.print_json:
            print(json.dumps(d, indent=2))

        out_file = out_dir / f"{puzzle.puzzle_id}.json"
        out_file.write_text(json.dumps(d, indent=2))
        print(f"  Saved → {out_file.relative_to(_ROOT)}")
        generated += 1

        total_time = time.time() - t_start
        print(f"  Total: {generated} puzzles in {total_time/60:.1f} min")
        print()

    print(f"Done. {generated} puzzles generated.")


if __name__ == "__main__":
    main()
