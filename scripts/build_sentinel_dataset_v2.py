#!/usr/bin/env python3
"""scripts/build_sentinel_dataset_v2.py — combined JSONL + Malom-sampled sentinel dataset.

Produces a single `.npz` that `train_sentinel.py --dataset PATH` can consume.
Composition per user spec:
  * 60% Malom-sampled positions, equal splits of placement / midgame / fly.
    Positions are sampled from `human_db.sqlite` state_keys that carry a
    Malom label (covers the reachable-in-play state space with correct
    ground truth); for each, all legal moves are labelled via the same
    ExternalSolvedDB path the trainer already uses.
  * 40% classic JSONL replay from `data/games` + `data/human_games`.
  * "double the training set sizes at each training level" is expressed
    as a target `--total-examples` — dataset is one file, all three
    v2-stage trainings consume it via --dataset.
  * Malom-sampled positions must contain **at least one W-or-L legal
    move** (not all-draw), so the model is trained on genuinely decisive
    states rather than dead-flat draws.

Usage:
    .venv/bin/python scripts/build_sentinel_dataset_v2.py \\
        --out learned_ai/sentinel/datasets/v2_combined.npz \\
        --total-examples 4000000 \\
        --malom-fraction 0.6

    # smoke:
    .venv/bin/python scripts/build_sentinel_dataset_v2.py \\
        --out /tmp/sentinel_combined_smoke.npz \\
        --total-examples 40000

Output NPZ layout is `SentinelDataset.save_to_disk` — the trainer already
knows how to load it via `--dataset`.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState, POSITIONS
from learned_ai.sentinel.dataset import (
    SentinelDataset,
    examples_from_position,
)
from learned_ai.sentinel.labels import MoveExample
from learned_ai.sentinel.db_teacher import ExternalSolvedDB


# ── Board reconstruction (matches ai.trajectory_db.make_board_state_key) ─────

def _board_from_state_key(state_key: str) -> "BoardState | None":
    parts = state_key.split("|")
    if len(parts) != 7:
        return None
    canon24, turn, _phase, pw, pb, ow, ob = parts
    if len(canon24) != 24:
        return None
    try:
        positions = {p: (canon24[i] if canon24[i] != "." else "") for i, p in enumerate(POSITIONS)}
        return BoardState(
            positions=positions,
            turn=turn,
            pieces_on_board={"W": int(ow), "B": int(ob)},
            pieces_placed={"W": int(pw), "B": int(pb)},
            pieces_captured={"W": max(0, int(pb) - int(ob)),
                             "B": max(0, int(pw) - int(ow))},
        )
    except Exception:
        return None


# ── Per-phase state_key sampler ──────────────────────────────────────────────

def _sample_state_keys_by_phase(
    human_db: Path, phase: str, target: int, rng: np.random.Generator,
) -> list[str]:
    """Sample up to `target` state_keys of the given phase that have a Malom label."""
    conn = sqlite3.connect(str(human_db))
    # LIMIT * 4 gives us headroom for later legal-move-count filtering.
    rows = conn.execute(
        "SELECT state_key FROM positions "
        "WHERE malom_wdl IS NOT NULL "
        f"AND state_key LIKE '%|{phase}|%' "
        "LIMIT ?",
        (target * 4,),
    ).fetchall()
    conn.close()
    keys = [r[0] for r in rows]
    if len(keys) <= target:
        return keys
    idx = rng.choice(len(keys), size=target, replace=False)
    return [keys[i] for i in idx]


# ── Malom-sampled example builder ────────────────────────────────────────────

def _malom_examples_for_state_keys(
    state_keys: list[str],
    db_teacher: "ExternalSolvedDB",
    label: str,
) -> tuple[list[MoveExample], dict]:
    """For each key: reconstruct board, generate MoveExamples via the DB teacher.

    Returns (examples, stats).  A position is dropped when:
      * its board fails to reconstruct
      * it has fewer than two legal moves (no ranking signal)
      * every legal move's Malom WDL is "draw" (the "not all draws" filter)
    """
    examples: list[MoveExample] = []
    stats = {"seen": 0, "bad_board": 0, "few_moves": 0, "all_draw": 0, "kept": 0}
    t0 = time.time()

    for i, sk in enumerate(state_keys):
        stats["seen"] += 1
        board = _board_from_state_key(sk)
        if board is None:
            stats["bad_board"] += 1
            continue

        # Generate per-move MoveExamples using the trainer's live-DB path.
        try:
            per_move = examples_from_position(
                board, board.turn, ply=0, db=db_teacher,
                played_move_key=None, trajectory_boost=1.0,
            )
        except Exception:
            stats["bad_board"] += 1
            continue

        if len(per_move) < 2:
            stats["few_moves"] += 1
            continue

        # "Not all draws": require at least one solved_db-labelled W or L.
        # move_quality > 0.5 means W (or DTM-graded win), < 0.5 means L,
        # exactly 0.5 (with solved_db source) means D.
        decisive = any(
            ex.supervision_source.startswith("solved_db")
            and abs(ex.move_quality - 0.5) > 1e-3
            for ex in per_move
        )
        if not decisive:
            stats["all_draw"] += 1
            continue

        for ex in per_move:
            ex.meta = {**(ex.meta or {}), "malom_sample_label": label}
        examples.extend(per_move)
        stats["kept"] += 1

        if (i + 1) % 5000 == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"  [{label}] {i+1:,}/{len(state_keys):,} keys  "
                  f"kept={stats['kept']:,} examples={len(examples):,}  "
                  f"[{rate:.0f}/s]", flush=True)

    return examples, stats


# ── Main pipeline ────────────────────────────────────────────────────────────

def build(
    out: Path,
    total_examples: int,
    malom_fraction: float,
    human_db: Path,
    malom_db_dir: Path,
    game_dir: Path,
    human_game_dir: "Path | None",
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    target_malom = int(round(total_examples * malom_fraction))
    target_jsonl = total_examples - target_malom
    target_per_phase = max(1, target_malom // 3)

    print(f"[v2] Target composition:")
    print(f"     total examples     : {total_examples:,}")
    print(f"     malom-sampled      : {target_malom:,}  (60%)")
    print(f"     per phase (malom)  : {target_per_phase:,}  (place/move/fly)")
    print(f"     jsonl-replayed     : {target_jsonl:,}  (40%)")
    print()

    # ── DB teacher ───────────────────────────────────────────────────────────
    print(f"[v2] Loading Malom DB teacher from {malom_db_dir}")
    db_teacher = ExternalSolvedDB(db_path=str(malom_db_dir))
    if not db_teacher.is_available():
        raise SystemExit(f"Malom DB not usable at {malom_db_dir}")

    # ── Phase-balanced Malom sampling ────────────────────────────────────────
    # Each position produces ~5-30 MoveExamples (one per legal move).  We
    # sample enough state_keys so kept*mean_moves ≈ target_per_phase.
    _MOVES_PER_POSITION = 8   # empirical average across all phases
    keys_per_phase = max(1, target_per_phase // _MOVES_PER_POSITION)

    malom_examples: list[MoveExample] = []
    for phase in ("place", "move", "fly"):
        print(f"[v2] Sampling ~{keys_per_phase:,} '{phase}' state_keys from {human_db}")
        keys = _sample_state_keys_by_phase(human_db, phase, keys_per_phase, rng)
        print(f"     got {len(keys):,} candidate keys")
        exs, stats = _malom_examples_for_state_keys(keys, db_teacher, phase)
        print(f"     [{phase}] stats: {stats}  → {len(exs):,} MoveExamples")
        malom_examples.extend(exs)

    print(f"[v2] Total Malom-sampled MoveExamples: {len(malom_examples):,}")
    if len(malom_examples) > target_malom:
        idx = rng.choice(len(malom_examples), size=target_malom, replace=False)
        malom_examples = [malom_examples[i] for i in idx]
        print(f"     down-sampled to {len(malom_examples):,} (target {target_malom:,})")
    print()

    # ── JSONL replay ─────────────────────────────────────────────────────────
    # Rough calibration: ~10–20 examples per game file after dedup.  A per-file
    # limit proportional to `target_jsonl` keeps small smoke runs fast without
    # capping full runs (where target_jsonl is in the millions).
    _jsonl_file_limit = None
    if target_jsonl < 250_000:
        _jsonl_file_limit = max(50, target_jsonl // 5)
        print(f"[v2] Small target — capping JSONL files at {_jsonl_file_limit:,} for speed")
    print(f"[v2] Replaying JSONL games from {game_dir}"
          + (f" + {human_game_dir}" if human_game_dir else ""))
    jsonl_ds = SentinelDataset.load_from_games(
        game_dir=str(game_dir),
        db=db_teacher,
        extra_dirs=[str(human_game_dir)] if human_game_dir else None,
        limit=_jsonl_file_limit,
    )
    jsonl_examples = list(jsonl_ds.examples)
    print(f"     got {len(jsonl_examples):,} JSONL MoveExamples")
    if len(jsonl_examples) > target_jsonl:
        idx = rng.choice(len(jsonl_examples), size=target_jsonl, replace=False)
        jsonl_examples = [jsonl_examples[i] for i in idx]
        print(f"     down-sampled to {len(jsonl_examples):,} (target {target_jsonl:,})")
    print()

    # ── Combine + write ──────────────────────────────────────────────────────
    combined = SentinelDataset(malom_examples + jsonl_examples)
    print(f"[v2] Combined dataset: {len(combined):,} examples")
    print(f"     malom : {len(malom_examples):,}")
    print(f"     jsonl : {len(jsonl_examples):,}")

    # Shuffle once before writing so per-batch samples come from both sources.
    rng.shuffle(combined.examples)

    out.parent.mkdir(parents=True, exist_ok=True)
    combined.save_to_disk(str(out))
    size_mb = round(out.stat().st_size / (1024 * 1024), 1)
    print(f"[v2] Wrote {out}  ({size_mb} MB)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out",              type=Path, required=True)
    p.add_argument("--total-examples",   type=int,   default=4_000_000,
                   help="Target combined dataset size.  Default 4M (~2× the "
                        "typical JSONL-only pipeline).")
    p.add_argument("--malom-fraction",   type=float, default=0.60,
                   help="Fraction of total examples drawn from Malom sampling.")
    p.add_argument("--human-db",         type=Path,
                   default=_ROOT / "data" / "human_db.sqlite",
                   help="Source of Malom-labelled state_keys.")
    p.add_argument("--malom-db",         type=Path,
                   default=Path("/mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted"),
                   help="Malom DB directory (per-move WDL+DTM teacher).")
    p.add_argument("--game-dir",         type=Path, default=_ROOT / "data" / "games")
    p.add_argument("--human-game-dir",   type=Path, default=_ROOT / "data" / "human_games")
    p.add_argument("--seed",             type=int,   default=42)
    args = p.parse_args()

    if not (0.0 < args.malom_fraction < 1.0):
        raise SystemExit("--malom-fraction must be in (0, 1).")
    if not args.human_db.exists():
        raise SystemExit(f"human_db not found: {args.human_db}")
    if not args.malom_db.exists():
        raise SystemExit(f"malom_db not found: {args.malom_db}")
    if not args.game_dir.exists():
        raise SystemExit(f"game_dir not found: {args.game_dir}")
    if args.human_game_dir and not args.human_game_dir.exists():
        args.human_game_dir = None

    build(
        out=args.out,
        total_examples=args.total_examples,
        malom_fraction=args.malom_fraction,
        human_db=args.human_db,
        malom_db_dir=args.malom_db,
        game_dir=args.game_dir,
        human_game_dir=args.human_game_dir,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
