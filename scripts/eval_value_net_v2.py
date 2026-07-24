#!/usr/bin/env python3
"""scripts/eval_value_net_v2.py — Held-out ValueNet comparison (Step 6c).

Loads every human_db position with a Malom WDL label, deterministically hash-
splits by state_key into train/val, and reports MSE + sign accuracy on the
val portion for **each** ValueNet checkpoint provided.  This is the primary
promotion metric (plan Step 4c held-out MSE).

For a side-by-side comparison of v1 vs v2:

    .venv/bin/python scripts/eval_value_net_v2.py \\
        --net data/value_net.npz --net-name v1 \\
        --net data/value_net_v2.npz --net-name v2 \\
        --output eval_vn_v2_holdout.json

The split is the same across nets (same seed / hash function), so metrics are
directly comparable.  Predictions are clipped to [-1, 1] to match training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.value_net import ValueNet, board_to_features, _INPUT_DIM

# Reuse the reconstruction helper from the v2 training script for exact parity.
from tools.train_value_net_v2 import board_from_state_key


_WDL_TO_LABEL = {"W": 1.0, "D": 0.0, "L": -1.0}


def _val_hash_bucket(state_key: str) -> int:
    """Deterministic 0-99 bucket per state_key so multiple runs share the split."""
    h = hashlib.sha256(state_key.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % 100


def load_val(human_db: Path, val_fraction: float, limit: int | None) -> tuple[np.ndarray, np.ndarray, dict]:
    upper = int(round(val_fraction * 100))
    conn  = sqlite3.connect(str(human_db))
    q     = "SELECT state_key, malom_wdl FROM positions WHERE malom_wdl IS NOT NULL"
    if limit is not None:
        q += f" LIMIT {int(limit)}"

    X: list[np.ndarray] = []
    y: list[float]      = []
    n_seen = n_val_skipped = n_bad = 0
    t0 = time.time()

    for state_key, wdl in conn.execute(q):
        n_seen += 1
        if _val_hash_bucket(state_key) >= upper:
            n_val_skipped += 1
            continue
        label = _WDL_TO_LABEL.get(wdl)
        if label is None:
            n_bad += 1
            continue
        board = board_from_state_key(state_key)
        if board is None:
            n_bad += 1
            continue
        try:
            feat = np.asarray(board_to_features(board, board.turn), dtype=np.float32)
        except Exception:
            n_bad += 1
            continue
        if feat.shape[0] != _INPUT_DIM:
            n_bad += 1
            continue
        X.append(feat)
        y.append(label)

    conn.close()
    return np.stack(X), np.asarray(y, dtype=np.float32), {
        "n_seen":         n_seen,
        "n_val":          len(y),
        "n_val_skipped":  n_val_skipped,
        "n_bad":          n_bad,
        "load_seconds":   round(time.time() - t0, 2),
    }


def evaluate_net(net_path: Path, X: np.ndarray, y: np.ndarray) -> dict:
    net  = ValueNet()
    net.load(str(net_path))
    pred = net.predict_batch(X).ravel().astype(np.float32)
    pred = np.clip(pred, -1.0, 1.0)
    mse       = float(np.mean((pred - y) ** 2))
    sign_acc  = float(np.mean(np.sign(pred) == np.sign(y)))
    # Bucket accuracy by label
    def _acc_where(mask: np.ndarray) -> float:
        if not mask.any():
            return 0.0
        return float(np.mean(np.sign(pred[mask]) == np.sign(y[mask])))
    return {
        "path":            str(net_path),
        "n":               int(y.shape[0]),
        "mse":             round(mse, 5),
        "sign_accuracy":   round(sign_acc, 4),
        "sign_acc_W":      round(_acc_where(y > 0.5),  4),
        "sign_acc_D":      round(_acc_where(np.abs(y) < 0.5), 4),
        "sign_acc_L":      round(_acc_where(y < -0.5), 4),
        "pred_mean":       round(float(pred.mean()), 4),
        "pred_std":        round(float(pred.std()),  4),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--human-db",     type=Path, default=Path("data/human_db.sqlite"))
    p.add_argument("--net",          type=Path, action="append", required=True,
                   help="Value net .npz to evaluate.  Pass multiple times for a comparison.")
    p.add_argument("--net-name",     type=str,  action="append", default=None,
                   help="Optional labels for each --net, in matching order.")
    p.add_argument("--output",       type=Path, default=None)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--limit",        type=int,   default=None,
                   help="Cap positions read from the DB (smoke tests).")
    args = p.parse_args()

    names = args.net_name or [n.stem for n in args.net]
    if len(names) != len(args.net):
        raise SystemExit("Number of --net-name entries must match --net entries.")

    print(f"[vn_eval] Loading val split from {args.human_db}")
    X, y, stats = load_val(args.human_db, args.val_fraction, args.limit)
    print(f"[vn_eval] load stats: {stats}")
    print(f"[vn_eval] label dist: W={int((y > 0.5).sum()):,}  "
          f"D={int((np.abs(y) < 0.5).sum()):,}  L={int((y < -0.5).sum()):,}")
    print()

    per_net = {}
    for net_path, name in zip(args.net, names):
        if not net_path.exists():
            print(f"[vn_eval] SKIP {name}: {net_path} not found")
            continue
        result = evaluate_net(net_path, X, y)
        result["name"] = name
        per_net[name] = result
        print(f"[vn_eval] {name}: mse={result['mse']:.5f}  "
              f"sign_acc={result['sign_accuracy']:.4f}  "
              f"(W={result['sign_acc_W']:.3f}  D={result['sign_acc_D']:.3f}  L={result['sign_acc_L']:.3f})")

    summary = {"val_stats": stats, "nets": per_net}
    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\n[vn_eval] Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
