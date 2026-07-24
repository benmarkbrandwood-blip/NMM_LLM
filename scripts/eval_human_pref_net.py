#!/usr/bin/env python3
"""scripts/eval_human_pref_net.py — HumanPrefNet held-out evaluation (Step 6d).

Uses the same per-position filter as `tools/train_human_pref_net.py` so the
val set matches the training distribution:
  * keep 'L' records if any 'L' exists at this state_key, else
  * keep 'D' records, else skip.
  * always skip 'W' (human blunder) records.

Split: deterministic state_key hash (SHA-256, first 32 bits mod 100) into
val_fraction% held-out.  Matches the semantics of
`tools/train_value_net_v2.py`; running training + eval back-to-back on the
same seed produces a proper held-out val.

Metrics:
    top1_acc / top3_acc / top5_acc
        For each held-out qualifying position, does the human's actual move
        appear in the top-K ranked successors?
    spearman_multi
        For positions with more than one recorded human move, Spearman r
        between the HP-net ranking and the observed play-frequency ranking.

Reserved for a follow-up (needs Elo-strata data & game bench respectively):
    Elo-strata top-1 accuracy on high-Elo games.
    Move-prune bench: unpruned vs pruned AI win rate.

Usage:
    .venv/bin/python scripts/eval_human_pref_net.py \\
        --net data/human_pref_net.npz \\
        --output eval_human_pref_net.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from game.rules import get_all_legal_moves
from ai.value_net import board_to_features, _INPUT_DIM

# Reuse loader + reconstruction + notation formatter from Step 2.
from tools.train_value_net_v2 import board_from_state_key
from tools.train_human_pref_net import _move_notation, _per_state_filter
from scripts.build_gap_dataset import HumanPrefLoader


def _val_bucket(state_key: str) -> int:
    h = hashlib.sha256(state_key.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % 100


def _spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    def _rank(vs):
        idx  = sorted(range(len(vs)), key=lambda i: vs[i])
        r    = [0.0] * len(vs)
        for k, i in enumerate(idx):
            r[i] = float(k)
        return r
    xr, yr = _rank(xs), _rank(ys)
    mx, my = sum(xr) / n, sum(yr) / n
    num = sum((xr[i] - mx) * (yr[i] - my) for i in range(n))
    dx  = (sum((v - mx) ** 2 for v in xr)) ** 0.5
    dy  = (sum((v - my) ** 2 for v in yr)) ** 0.5
    return num / (dx * dy) if dx > 1e-12 and dy > 1e-12 else 0.0


def evaluate(net_path: Path, human_db: Path, val_fraction: float, limit: int | None) -> dict:
    hp = HumanPrefLoader(net_path)

    upper = int(round(val_fraction * 100))
    conn  = sqlite3.connect(str(human_db))
    rows  = conn.execute(
        "SELECT state_key, notation, total, malom_wdl_after FROM moves "
        "WHERE malom_wdl_after IS NOT NULL ORDER BY state_key"
    )

    n_top1 = n_top3 = n_top5 = n_positions = 0
    spearman_values: list[float] = []
    n_bad_state = n_notation_miss = n_skip_all_losing = n_position_apply_error = 0
    t0 = time.time()

    def _flush(state_key: str, buffered: list[tuple[str, int, str]]) -> None:
        nonlocal n_top1, n_top3, n_top5, n_positions
        nonlocal n_bad_state, n_notation_miss, n_skip_all_losing, n_position_apply_error

        if _val_bucket(state_key) >= upper:
            return   # not in the held-out slice
        wdls = [r[2] for r in buffered]
        keep = _per_state_filter(wdls)
        if keep is None:
            n_skip_all_losing += 1
            return
        chosen = [(not_, tot) for (not_, tot, w) in buffered if w == keep]
        if not chosen:
            return
        board = board_from_state_key(state_key)
        if board is None:
            n_bad_state += 1
            return
        legal = get_all_legal_moves(board)
        if len(legal) < 2:
            return
        notation_to_move: dict[str, dict] = {_move_notation(m): m for m in legal}
        # Compute successor features & hp scores for every legal move.
        succ_feats: list[np.ndarray] = []
        keys_in_order:  list[str]    = []
        for m in legal:
            try:
                succ = board.apply_move(m)
                succ_feats.append(np.asarray(board_to_features(succ, succ.turn), dtype=np.float32))
                keys_in_order.append(_move_notation(m))
            except Exception:
                n_position_apply_error += 1
                return
        if not succ_feats:
            return
        arr = np.stack(succ_feats)
        scores = hp.score_batch(arr)
        # Rank (higher score = higher rank).
        rank_order = np.argsort(-scores)
        ranked_notations = [keys_in_order[i] for i in rank_order]

        # top-K accuracy: chosen move (highest-freq entry) matches ranked_notations[:K].
        best_chosen, _ = max(chosen, key=lambda t: t[1])
        if best_chosen not in notation_to_move:
            n_notation_miss += 1
            return
        n_positions += 1
        if best_chosen == ranked_notations[0]:
            n_top1 += 1
        if best_chosen in ranked_notations[:3]:
            n_top3 += 1
        if best_chosen in ranked_notations[:5]:
            n_top5 += 1

        # Spearman for multi-move positions: HP score vs observed play frequency.
        if len(chosen) >= 2:
            freq_by_notation = {n: t for (n, t) in chosen}
            legal_scores = []
            legal_freqs  = []
            for n, s in zip(keys_in_order, scores):
                if n in freq_by_notation:
                    legal_scores.append(float(s))
                    legal_freqs.append(float(freq_by_notation[n]))
            if len(legal_scores) >= 2:
                spearman_values.append(_spearman(legal_scores, legal_freqs))

    current_key: str | None = None
    buffered: list[tuple[str, int, str]] = []
    for state_key, notation, total, wdl_after in rows:
        if current_key is None:
            current_key = state_key
        if state_key != current_key:
            _flush(current_key, buffered)
            buffered = []
            current_key = state_key
            if limit is not None and n_positions >= limit:
                break
        buffered.append((notation, int(total or 0), wdl_after))
    else:
        if current_key is not None and buffered:
            _flush(current_key, buffered)
    conn.close()

    def _safe(a: int, b: int) -> float:
        return round(a / b, 4) if b > 0 else 0.0

    return {
        "net":                str(net_path),
        "val_fraction":       val_fraction,
        "n_positions":        n_positions,
        "top1_acc":           _safe(n_top1, n_positions),
        "top3_acc":           _safe(n_top3, n_positions),
        "top5_acc":           _safe(n_top5, n_positions),
        "spearman_multi":     round(sum(spearman_values) / len(spearman_values), 4)
                              if spearman_values else 0.0,
        "n_spearman_samples": len(spearman_values),
        "diagnostics": {
            "bad_state":            n_bad_state,
            "notation_miss":        n_notation_miss,
            "skip_all_losing":      n_skip_all_losing,
            "position_apply_error": n_position_apply_error,
        },
        "eval_seconds": round(time.time() - t0, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--net",          type=Path, default=Path("data/human_pref_net.npz"))
    p.add_argument("--human-db",     type=Path, default=Path("data/human_db.sqlite"))
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--limit",        type=int,   default=None)
    p.add_argument("--output",       type=Path,  default=None)
    args = p.parse_args()

    if not args.net.exists():
        raise SystemExit(f"net not found: {args.net}")
    if not args.human_db.exists():
        raise SystemExit(f"human_db not found: {args.human_db}")

    print(f"[hp_eval] Evaluating {args.net} on the {int(args.val_fraction * 100)}% "
          f"held-out slice of {args.human_db}")
    result = evaluate(args.net, args.human_db, args.val_fraction, args.limit)
    print()
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\n[hp_eval] Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
