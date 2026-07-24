#!/usr/bin/env python3
"""scripts/eval_sentinel_db.py — Sentinel evaluation on DB-sampled positions (Step 6a).

Samples positions directly from human_db.sqlite (reversible state_key), reconstructs
each board, enumerates its legal moves, queries the Malom DB for per-move WDL + DTM,
runs the sentinel with DB feature slots ZEROED (matches live inference), and reports
alignment metrics.

specialist_db is NOT sampled — its `pos_hash` primary key is a non-reversible SHA-1,
so board reconstruction from the DB alone is not possible.  Per the plan's "Known
issues" section, that data source would require replaying the JSONL games that fed
the DB.

Metrics (matches scripts/eval_sentinel.py naming):
    win_acc         fraction of DB-win moves the sentinel scores > 0.5
    loss_acc        fraction of DB-loss moves the sentinel scores < 0.5
    top1_win_rate   positions with a DB-win available where sentinel #1 is a win
    spearman_r      Spearman rank correlation between sentinel and DB quality
    dtm_pearson_r   Pearson correlation between sentinel and DB DTM

Run for v1 and v2 sentinels to produce a direct comparison table.

Usage:
    .venv/bin/python scripts/eval_sentinel_db.py \\
        --checkpoint learned_ai/sentinel/checkpoints/best.pt \\
        --output eval_sentinel_db_v1.json --n-samples 1000

    .venv/bin/python scripts/eval_sentinel_db.py \\
        --checkpoint learned_ai/sentinel/checkpoints/v2/best.pt \\
        --output eval_sentinel_db_v2.json --n-samples 1000
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from game.rules import get_all_legal_moves, get_game_phase


# ── Board reconstruction (mirrors ai.trajectory_db.make_board_state_key) ──────

def _board_from_state_key(state_key: str) -> BoardState | None:
    parts = state_key.split("|")
    if len(parts) != 7:
        return None
    canon24, turn, _phase, pw, pb, ow, ob = parts
    if len(canon24) != 24:
        return None
    try:
        from game.board import POSITIONS
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


# ── Statistical helpers ──────────────────────────────────────────────────────

def _spearman_r(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    rank_x = _rank(xs)
    rank_y = _rank(ys)
    return _pearson(rank_x, rank_y)


def _rank(vs: list[float]) -> list[float]:
    idx    = sorted(range(len(vs)), key=lambda i: vs[i])
    ranks  = [0.0] * len(vs)
    for r, i in enumerate(idx):
        ranks[i] = float(r)
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx  = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy  = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-12 or dy < 1e-12:
        return 0.0
    return num / (dx * dy)


# ── Per-move DB quality signal ───────────────────────────────────────────────

_WDL_QUALITY = {"L": 1.0, "D": 0.5, "W": 0.0}   # child perspective: L = we caused opponent loss = good


def _malom_move_signals(board: BoardState, legal: list[dict], malom_db) -> list[tuple[str, int]]:
    """Return [(wdl, dtw)] for each legal move by applying → querying Malom."""
    out: list[tuple[str, int]] = []
    for m in legal:
        try:
            succ = board.apply_move(m)
            v    = malom_db.query(succ)
            if v is None:
                out.append(("?", 0))
            else:
                out.append((v["outcome"], v["dtw"]))
        except Exception:
            out.append(("?", 0))
    return out


# ── Main eval ────────────────────────────────────────────────────────────────

def evaluate(
    checkpoint: Path,
    human_db: Path,
    malom_db_path: Path,
    n_samples: int,
    seed: int,
) -> dict:
    import numpy as np
    from ai.malom_db import MalomDB
    from learned_ai.sentinel.infer import SentinelAdvisor

    # Sample state_keys, stratified by phase where possible.
    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(str(human_db))

    def _sample(phase: str, n: int) -> list[str]:
        rows = conn.execute(
            f"SELECT state_key FROM positions WHERE state_key LIKE '%|{phase}|%'"
        ).fetchall()
        if not rows:
            return []
        keys = [r[0] for r in rows]
        if len(keys) <= n:
            return keys
        idx = rng.choice(len(keys), size=n, replace=False)
        return [keys[i] for i in idx]

    per_phase = max(1, n_samples // 3)
    state_keys = (
        _sample("place", per_phase)
        + _sample("move",  per_phase)
        + _sample("fly",   per_phase)
    )
    conn.close()
    print(f"Sampled {len(state_keys)} positions from human_db "
          f"(target {n_samples} across place / move / fly).")

    # Load sentinel + malom.
    advisor = SentinelAdvisor(checkpoint_path=str(checkpoint))
    malom   = MalomDB(str(malom_db_path))
    # Warm sentinel lazy state.
    board_tmp = BoardState.new_game()
    advisor.advise(board_tmp, [{"from": None, "to": "a1", "capture": None}],
                   board_tmp.turn, played_move_idx=0)

    # Per-phase counters.
    counters = {
        "overall": defaultdict(int),
        "place":   defaultdict(int),
        "move":    defaultdict(int),
        "fly":     defaultdict(int),
    }
    quality_pairs:  list[tuple[float, float]] = []   # (sentinel_score, DB quality)
    dtm_pairs:      list[tuple[float, int]]   = []   # (sentinel_score, DB DTM)
    top1_positions       = 0
    top1_win_positions   = 0
    top1_win_correct     = 0
    skipped              = 0

    for i, sk in enumerate(state_keys):
        board = _board_from_state_key(sk)
        if board is None:
            skipped += 1
            continue
        try:
            phase = get_game_phase(board, board.turn)
        except Exception:
            skipped += 1
            continue
        legal = get_all_legal_moves(board)
        if not legal:
            skipped += 1
            continue
        try:
            advice = advisor.advise(board, legal, board.turn, played_move_idx=0)
        except Exception:
            skipped += 1
            continue
        if advice is None or len(advice.move_scores) != len(legal):
            skipped += 1
            continue
        sent_scores = list(advice.move_scores)
        malom_move  = _malom_move_signals(board, legal, malom)
        if not any(w != "?" for w, _ in malom_move):
            skipped += 1
            continue

        for (wdl, dtw), score in zip(malom_move, sent_scores):
            if wdl not in _WDL_QUALITY:
                continue
            q = _WDL_QUALITY[wdl]
            quality_pairs.append((score, q))
            dtm_pairs.append((score, int(dtw)))
            for cell in ("overall", phase):
                c = counters[cell]
                if wdl == "L":
                    c["win_total"] += 1
                    if score > 0.5:
                        c["win_correct"] += 1
                elif wdl == "W":
                    c["loss_total"] += 1
                    if score < 0.5:
                        c["loss_correct"] += 1

        # top1_win_rate: positions where a DB-win exists, does sentinel top-1 pick one?
        db_win_indices = [j for j, (w, _) in enumerate(malom_move) if w == "L"]
        if db_win_indices:
            top1_win_positions += 1
            top1_idx = max(range(len(sent_scores)), key=lambda k: sent_scores[k])
            if malom_move[top1_idx][0] == "L":
                top1_win_correct += 1
        top1_positions += 1

        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{len(state_keys)}  skipped={skipped}")

    def _safe_div(a: int, b: int) -> float:
        return round(a / b, 4) if b > 0 else 0.0

    def _phase_summary(bucket) -> dict:
        return {
            "win_acc":  _safe_div(bucket["win_correct"],  bucket["win_total"]),
            "loss_acc": _safe_div(bucket["loss_correct"], bucket["loss_total"]),
            "n_win":    bucket["win_total"],
            "n_loss":   bucket["loss_total"],
        }

    result = {
        "checkpoint":     str(checkpoint),
        "n_positions":    top1_positions,
        "n_skipped":      skipped,
        "win_acc":        _phase_summary(counters["overall"])["win_acc"],
        "loss_acc":       _phase_summary(counters["overall"])["loss_acc"],
        "top1_win_rate":  _safe_div(top1_win_correct, top1_win_positions),
        "spearman_r":     round(_spearman_r([p[0] for p in quality_pairs],
                                            [p[1] for p in quality_pairs]), 4),
        "dtm_pearson_r":  round(_pearson([p[0] for p in dtm_pairs],
                                         [float(p[1]) for p in dtm_pairs]), 4),
        "phase_breakdown": {
            phase: _phase_summary(counters[phase]) for phase in ("place", "move", "fly")
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Sentinel checkpoint to evaluate.")
    parser.add_argument("--human-db",  type=Path, default=Path("data/human_db.sqlite"))
    parser.add_argument("--malom-db",  type=Path,
                        default=Path("/mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted"))
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Positions to sample across place / move / fly.")
    parser.add_argument("--output",    type=Path, default=None,
                        help="Optional JSON path for the result summary.")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not args.human_db.exists():
        raise SystemExit(f"human_db not found: {args.human_db}")
    if not args.malom_db.exists():
        raise SystemExit(f"malom_db not found: {args.malom_db}")

    result = evaluate(
        checkpoint=args.checkpoint,
        human_db=args.human_db,
        malom_db_path=args.malom_db,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    print()
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
