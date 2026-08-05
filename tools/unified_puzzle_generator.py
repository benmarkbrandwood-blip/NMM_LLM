#!/usr/bin/env python3
"""tools/unified_puzzle_generator.py — Unified CLI for generating NMM puzzles.

Generates endgame, midgame (Malom), and placement-phase puzzles using
multiprocessing for parallelism.  Supports single-run and batch modes.

Usage examples:

  # Generate 20 midgame puzzles at depth 5 using 4 workers
  .venv/bin/python tools/unified_puzzle_generator.py --type midgame --depth 5 --count 20

  # Generate 50 endgame puzzles, random depth, default workers
  .venv/bin/python tools/unified_puzzle_generator.py --type endgame --depth 0 --count 50

  # Run from a batch config (resume-aware)
  .venv/bin/python tools/unified_puzzle_generator.py --batch data/puzzles/batch_config.json --workers 6

  # Generate forever, White only, win-in-1 constraint
  .venv/bin/python tools/unified_puzzle_generator.py --type midgame --side W --max-winning-moves 1
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

_OUT_DIRS = {
    "endgame":   _ROOT / "data" / "puzzles" / "endgame",
    "midgame":   _ROOT / "data" / "puzzles" / "malom",
    "placement": _ROOT / "data" / "puzzles" / "placement",
}

_DEFAULT_BATCH_CONFIG = _ROOT / "data" / "puzzles" / "batch_config.json"

_DEFAULT_BATCH = {
    "cells": [
        {"type": "midgame",   "side": "random", "depth": 0, "count": 100},
        {"type": "placement", "side": "random", "depth": 0, "count": 50},
        {"type": "endgame",   "side": "random", "depth": 0, "count": 50},
    ],
    "max_winning_moves": 1,
    "min_hardness": 3.0,
}


# ── Settings loader ────────────────────────────────────────────────────────────

def _load_settings() -> dict:
    p = _ROOT / "data" / "settings.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ── Worker initialiser (pre-warms hash cache once per worker process) ─────────

def _worker_init(settings_path: str) -> None:
    """Called once per worker process at pool startup."""
    import sys
    sys.path.insert(0, str(_ROOT))
    global _worker_settings
    try:
        _worker_settings = json.loads(Path(settings_path).read_text()) if Path(settings_path).exists() else {}
    except Exception:
        _worker_settings = {}

    # Pre-warm Malom hash cache (slow once; free afterwards)
    try:
        from ai.malom_puzzle_search import prewarm_hash_cache
        print(f"[worker {os.getpid()}] pre-warming Malom hash states (max_pieces=9)…", flush=True)
        prewarm_hash_cache(9)
        print(f"[worker {os.getpid()}] hash cache ready.", flush=True)
    except Exception as exc:
        print(f"[worker {os.getpid()}] prewarm failed: {exc}", flush=True)


# ── Worker function ────────────────────────────────────────────────────────────

def _worker_generate(args: tuple) -> dict | None:
    """Worker: generate one puzzle and return its dict, or None on failure."""
    puzzle_type, depth, side, max_winning_moves, min_hardness, attempts, settings_path = args

    try:
        settings = json.loads(Path(settings_path).read_text()) if Path(settings_path).exists() else {}
    except Exception:
        settings = {}

    if puzzle_type == "endgame":
        return _gen_endgame(depth, side, max_winning_moves, min_hardness, attempts, settings)
    elif puzzle_type == "midgame":
        return _gen_midgame(depth, side, max_winning_moves, min_hardness, attempts, settings)
    elif puzzle_type == "placement":
        return _gen_placement(depth, side, max_winning_moves, min_hardness, attempts, settings)
    return None


def _gen_endgame(depth, side, max_winning_moves, min_hardness, attempts, settings) -> dict | None:
    import random as _rand
    from ai.puzzle_search import generate_puzzle, load_puzzle_db

    db_dir = _ROOT / "data" / "endgame"
    db_files = sorted(db_dir.glob("endgame_*.wdl"))
    if not db_files:
        return None

    def _parse_nw_nb(name):
        stem = name.stem.replace("endgame_", "")
        nW, nB = map(int, stem.split("_"))
        return nW, nB

    db_file = _rand.choice(db_files)
    nW, nB = _parse_nw_nb(db_file)

    eff_side  = _rand.choice(["W", "B"]) if side == "random" else side
    eff_depth = _rand.choice([3, 4, 5, 6, 7, 8, 9, 10]) if depth == 0 else depth

    db = load_puzzle_db(str(db_dir), nW, nB, max_depth=max(eff_depth, 10))
    if db._tables.get((nW, nB)) is None:
        return None

    puzzle = generate_puzzle(
        db, nW, nB, eff_side, eff_depth,
        max_attempts=attempts,
        max_winning_moves=max_winning_moves,
    )
    if puzzle is None:
        return None
    if puzzle.hardness_score < min_hardness:
        return None

    d = puzzle.to_dict()
    d["created_at"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
    return d


def _gen_midgame(depth, side, max_winning_moves, min_hardness, attempts, settings) -> dict | None:
    from ai.malom_db import MalomDB
    from ai.malom_puzzle_search import generate_malom_puzzle

    malom_path = settings.get("malom_db_path", "")
    if not malom_path:
        return None

    db = MalomDB(malom_path)
    if not db.is_available():
        return None

    puzzle = generate_malom_puzzle(
        db,
        winning_side=side,
        target_win_in=depth,
        max_attempts=attempts,
        max_winning_moves=max_winning_moves,
    )
    if puzzle is None:
        return None
    if puzzle.hardness_score < min_hardness:
        return None

    d = puzzle.to_dict()
    d["created_at"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
    return d


def _gen_placement(depth, side, max_winning_moves, min_hardness, attempts, settings) -> dict | None:
    from ai.malom_db import MalomDB
    from ai.malom_puzzle_search import generate_malom_placement_puzzle

    malom_path = settings.get("malom_db_path", "")
    if not malom_path:
        return None

    db = MalomDB(malom_path)
    if not db.is_available():
        return None

    puzzle = generate_malom_placement_puzzle(
        db,
        winning_side=side,
        target_win_in=depth,
        max_attempts=attempts,
        max_winning_moves=max_winning_moves,
    )
    if puzzle is None:
        return None
    if puzzle.hardness_score < min_hardness:
        return None

    d = puzzle.to_dict()
    d["created_at"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
    return d


# ── Atomic JSON write ─────────────────────────────────────────────────────────

def _write_puzzle_json(out_dir: Path, puzzle_dict: dict) -> Path:
    """Write puzzle JSON atomically (tmp file + rename)."""
    pid = puzzle_dict["id"]
    out_file = out_dir / f"{pid}.json"
    tmp_file = out_dir / f"{pid}.tmp"
    tmp_file.write_text(json.dumps(puzzle_dict, indent=2))
    os.rename(tmp_file, out_file)
    return out_file


# ── Count existing puzzles (for batch quota) ───────────────────────────────────

def _count_existing(out_dir: Path, puzzle_type: str, side: str, depth: int) -> int:
    """Count existing cached puzzles matching (type, side, depth)."""
    if not out_dir.exists():
        return 0
    count = 0
    # Prefix filters by type
    prefixes = {"endgame": "egdb_", "midgame": "mlm_", "placement": "plc_"}
    prefix = prefixes.get(puzzle_type, "")
    for jf in out_dir.glob("*.json"):
        if prefix and not jf.stem.startswith(prefix):
            continue
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        if side != "random" and data.get("winning_side") != side:
            continue
        if depth != 0 and data.get("target_win_in") != depth:
            continue
        count += 1
    return count


# ── Main generation loop ───────────────────────────────────────────────────────

def run_generation(
    puzzle_type: str,
    depth: int,
    side: str,
    max_winning_moves: int,
    min_hardness: float,
    count: int,
    attempts: int,
    workers: int,
    out_dir: Path,
    settings_path: str,
) -> None:
    """Run the generation loop (possibly forever if count==0)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    run_forever = (count == 0)
    generated = 0
    dispatched = 0
    t_start = time.time()

    label = f"[{puzzle_type}/{side}/depth={depth or 'any'}]"

    print(f"{label} Starting — workers={workers}  count={'∞' if run_forever else count}  "
          f"attempts={attempts}  min-hardness={min_hardness}  max-winning={max_winning_moves}")
    print(f"{label} Output: {out_dir}")
    print()

    worker_args = (puzzle_type, depth, side, max_winning_moves, min_hardness, attempts, settings_path)

    with multiprocessing.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(settings_path,),
    ) as pool:
        # Use imap_unordered so we process results as they come in
        while run_forever or generated < count:
            batch_size = workers * 2
            tasks = [worker_args] * batch_size
            dispatched += batch_size

            t0 = time.time()
            for result in pool.imap_unordered(_worker_generate, tasks):
                if result is None:
                    continue
                if not run_forever and generated >= count:
                    break

                elapsed = time.time() - t0
                out_file = _write_puzzle_json(out_dir, result)
                generated += 1
                total_time = time.time() - t_start
                print(
                    f"{label} FOUND id={result['id']}  goal={result.get('goal', '?')}  "
                    f"score={result.get('hardness_score', '?')}  time={elapsed:.1f}s | "
                    f"total={generated} in {total_time/60:.1f} min"
                )
                print(f"  Saved → {out_file.relative_to(_ROOT)}")
                t0 = time.time()

                if not run_forever and generated >= count:
                    break

    total_time = time.time() - t_start
    print(f"\n{label} Done. {generated} puzzles generated in {total_time/60:.1f} min.")


# ── Batch mode ────────────────────────────────────────────────────────────────

def run_batch(
    batch_cfg: dict,
    workers: int,
    out_dir_override: Path | None,
    settings_path: str,
) -> None:
    """Process batch cells sequentially (each cell can use all workers)."""
    cells = batch_cfg.get("cells", [])
    global_mwm = batch_cfg.get("max_winning_moves", 1)
    global_minh = batch_cfg.get("min_hardness", 3.0)

    print(f"Batch mode: {len(cells)} cell(s)  workers={workers}")
    print()

    for i, cell in enumerate(cells):
        puzzle_type = cell["type"]
        side        = cell.get("side", "random")
        depth       = cell.get("depth", 0)
        count       = cell.get("count", 0)
        mwm         = cell.get("max_winning_moves", global_mwm)
        minh        = cell.get("min_hardness", global_minh)
        attempts_default = 5000 if puzzle_type == "endgame" else 3000
        attempts    = cell.get("attempts", attempts_default)

        out_dir = out_dir_override if out_dir_override else _OUT_DIRS.get(puzzle_type, _OUT_DIRS["endgame"])

        # Check existing quota
        existing = _count_existing(out_dir, puzzle_type, side, depth)
        label = f"[cell {i+1}/{len(cells)} {puzzle_type}/{side}/depth={depth or 'any'}]"
        print(f"{label} quota={count}  existing={existing}")

        if count > 0 and existing >= count:
            print(f"{label} Quota already met — skipping.\n")
            continue

        needed = count - existing if count > 0 else 0

        run_generation(
            puzzle_type=puzzle_type,
            depth=depth,
            side=side,
            max_winning_moves=mwm,
            min_hardness=minh,
            count=needed,
            attempts=attempts,
            workers=workers,
            out_dir=out_dir,
            settings_path=settings_path,
        )
        print()


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NMM unified puzzle generator (endgame / midgame / placement)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--type", choices=["endgame", "midgame", "placement"], default=None,
        help="Puzzle type (required unless --batch)",
    )
    parser.add_argument(
        "--depth", type=int, default=0, metavar="N",
        help="Target win depth (0=random within type range; endgame: 3–10; malom types: 4–10)",
    )
    parser.add_argument(
        "--side", choices=["W", "B", "random"], default="random",
        help="Winning side (default: random)",
    )
    parser.add_argument(
        "--max-winning-moves", type=int, default=1, metavar="N",
        help="Reject positions with more than N winning first moves (default: 1)",
    )
    parser.add_argument(
        "--min-hardness", type=float, default=3.0, metavar="F",
        help="Minimum hardness_score to accept (default: 3.0)",
    )
    parser.add_argument(
        "--count", type=int, default=0,
        help="Number of puzzles to generate; 0 = run forever (default: 0)",
    )
    parser.add_argument(
        "--attempts", type=int, default=None, metavar="N",
        help="Positions sampled per puzzle attempt (default: 5000 endgame, 3000 malom)",
    )
    parser.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="Number of parallel workers (default: max(1, cpu_count-1))",
    )
    parser.add_argument(
        "--batch", type=str, default=None, metavar="PATH",
        help="Batch config JSON path (overrides --type/--depth/--side)",
    )
    parser.add_argument(
        "--out", type=str, default=None, metavar="DIR",
        help="Output directory override",
    )
    args = parser.parse_args()

    settings_path = str(_ROOT / "data" / "settings.json")
    workers = args.workers if args.workers else max(1, (multiprocessing.cpu_count() or 2) - 1)
    out_dir_override = Path(args.out) if args.out else None

    if args.batch:
        # Batch mode
        batch_path = Path(args.batch)
        if not batch_path.exists():
            print(f"ERROR: batch config not found: {batch_path}", file=sys.stderr)
            sys.exit(1)
        batch_cfg = json.loads(batch_path.read_text())
        run_batch(batch_cfg, workers, out_dir_override, settings_path)
        return

    # Single-type mode
    if not args.type:
        parser.error("--type is required unless --batch is specified")

    puzzle_type = args.type
    attempts = args.attempts if args.attempts else (5000 if puzzle_type == "endgame" else 3000)

    # Validate depth range
    if puzzle_type == "endgame":
        valid_depths = {0, 3, 4, 5, 6, 7, 8, 9, 10}
    else:
        valid_depths = {0, 4, 5, 6, 7, 8, 9, 10}
    if args.depth not in valid_depths:
        parser.error(f"--depth {args.depth} is not valid for type={puzzle_type}. "
                     f"Valid: {sorted(valid_depths)}")

    out_dir = out_dir_override if out_dir_override else _OUT_DIRS[puzzle_type]

    # Write default batch config if it doesn't exist
    if not _DEFAULT_BATCH_CONFIG.exists():
        _DEFAULT_BATCH_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        _DEFAULT_BATCH_CONFIG.write_text(json.dumps(_DEFAULT_BATCH, indent=2))
        print(f"Created default batch config: {_DEFAULT_BATCH_CONFIG.relative_to(_ROOT)}")

    print(f"NMM Unified Puzzle Generator")
    print(f"Type: {puzzle_type}  Depth: {args.depth or 'any'}  Side: {args.side}")
    print(f"Workers: {workers}  Attempts: {attempts}  Min-hardness: {args.min_hardness}")
    print()

    run_generation(
        puzzle_type=puzzle_type,
        depth=args.depth,
        side=args.side,
        max_winning_moves=args.max_winning_moves,
        min_hardness=args.min_hardness,
        count=args.count,
        attempts=attempts,
        workers=workers,
        out_dir=out_dir,
        settings_path=settings_path,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
