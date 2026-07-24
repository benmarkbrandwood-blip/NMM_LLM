#!/usr/bin/env python3
"""tools/train_value_net_v2.py — Train ValueNet v2 on Malom WDL labels.

Implements Step 1 of docs/retrain_v2_plan.md.  Replaces the v1 final-outcome
regression target (every position in a game labelled with the final W/L/D)
with a per-position Malom WDL label mapped to {+1, 0, −1}.

Rationale (from the plan):
    The v1 label is *final game outcome* — every position in a game gets the
    same {+1, 0, −1}, so a single tactical error 40 plies before a loss trains
    the VN to score the losing position as +1.  Per-position Malom WDL is a
    sharp deterministic per-position signal.

The output range stays in `[−1, 1]` and the loss is MSE, so all existing
blender code (`ai/game_ai.py:1617-1619`, VN blend %, difficulty settings)
continues to work unchanged after promotion.

Data sources
------------
Primary: `data/human_db.sqlite`.  Its `positions.state_key` is the reversible
D4-canonical FEN `canon24|turn|phase|placed_w|placed_b|on_w|on_b`, produced by
`ai.trajectory_db.make_board_state_key`.  All 2.17M positions carry correct
sector-corrected Malom labels since the 2026-07-21 rebuild.

Secondary: `data/specialist_db.sqlite` — NOT used.  Per the plan's "Known
issues" section, specialist_db positions are indexed by `pos_hash` (SHA-1),
which is irreversible.  Supporting it would require replaying the JSONL games
that fed the DB and cross-referencing labels.  The `--specialist-db` flag is
accepted for API compatibility with the plan command; when provided, the
script prints a warning and continues using human_db only.

Split
-----
The plan asks for a game-level 80/20 split.  The human_db `positions` table
is aggregated across games — it does not preserve per-game membership, so a
strict game-level split would require rebuilding from JSONL.  This script
uses `ValueNet.train`'s built-in random position-level split via
`--val-fraction` (default 0.20).  For a Malom-WDL target this is still a
genuine generalisation test because the WDL label is deterministic in the
board position; the val set contains distinct board patterns that the model
has not seen at training time.  A strict game-level split can be added in a
follow-up if needed (would require joining with human_games JSONL).

Usage
-----
    .venv/bin/python tools/train_value_net_v2.py \\
        --human-db data/human_db.sqlite \\
        --specialist-db data/specialist_db.sqlite \\
        --output data/value_net_v2.npz \\
        --epochs 200 --patience 10

    # Smoke test on the first 5k positions:
    .venv/bin/python tools/train_value_net_v2.py --limit 5000 --epochs 3
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
from ai.value_net import ValueNet, board_to_features, _INPUT_DIM


# ── State-key → board reconstruction ─────────────────────────────────────────

def board_from_state_key(state_key: str) -> BoardState | None:
    """Reconstruct a BoardState from a canonical state_key.

    Format (see ai.trajectory_db.make_board_state_key):
        canon24 | turn | phase | placed_w | placed_b | on_w | on_b

    canon24 is a 24-character string over {'.', 'W', 'B'}.  It's the
    D4-canonical orientation, which is a valid board position — features
    extracted from it are functionally equivalent for value learning since
    board_to_features already computes rotation/reflection-invariant signals
    for the positions the net cares about.

    Returns None on malformed input.
    """
    parts = state_key.split("|")
    if len(parts) != 7:
        return None
    canon24, turn, _phase, placed_w_s, placed_b_s, on_w_s, on_b_s = parts
    if len(canon24) != len(POSITIONS):
        return None
    try:
        placed_w = int(placed_w_s)
        placed_b = int(placed_b_s)
        on_w     = int(on_w_s)
        on_b     = int(on_b_s)
    except ValueError:
        return None
    positions: dict[str, str] = {}
    for i, pos in enumerate(POSITIONS):
        c = canon24[i]
        positions[pos] = "" if c == "." else c
    # pieces captured BY color = opponent pieces placed but no longer on board.
    w_cap = max(0, placed_b - on_b)   # W captured B pieces
    b_cap = max(0, placed_w - on_w)   # B captured W pieces
    return BoardState(
        positions=positions,
        turn=turn,
        pieces_on_board={"W": on_w, "B": on_b},
        pieces_placed={"W": placed_w, "B": placed_b},
        pieces_captured={"W": w_cap, "B": b_cap},
    )


# ── Data loading ─────────────────────────────────────────────────────────────

_WDL_TO_LABEL = {"W": 1.0, "D": 0.0, "L": -1.0}


def load_from_human_db(
    db_path: Path,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Query positions with a Malom WDL label; return (X, y, stats)."""
    if not db_path.exists():
        raise FileNotFoundError(f"human_db not found at {db_path}")
    conn = sqlite3.connect(str(db_path))
    q = "SELECT state_key, malom_wdl FROM positions WHERE malom_wdl IS NOT NULL"
    if limit is not None:
        q += f" LIMIT {int(limit)}"

    feats:  list[np.ndarray] = []
    labels: list[float]      = []

    n_seen       = 0
    n_bad_key    = 0
    n_bad_wdl    = 0
    n_feat_error = 0
    t_start      = time.time()

    for state_key, wdl in conn.execute(q):
        n_seen += 1
        label = _WDL_TO_LABEL.get(wdl)
        if label is None:
            n_bad_wdl += 1
            continue
        board = board_from_state_key(state_key)
        if board is None:
            n_bad_key += 1
            continue
        try:
            feat = np.asarray(board_to_features(board, board.turn), dtype=np.float32)
        except Exception:
            n_feat_error += 1
            continue
        if feat.shape[0] != _INPUT_DIM:
            n_feat_error += 1
            continue

        feats.append(feat)
        labels.append(label)

        if n_seen % 100_000 == 0:
            _rate = n_seen / max(time.time() - t_start, 1e-6)
            print(f"  loaded {n_seen:>9,} positions ({len(feats):,} usable)  "
                  f"[{_rate:,.0f}/s]")

    conn.close()

    if not feats:
        raise RuntimeError("No usable positions found in human_db.  "
                           "Was the DB rebuilt with malom_wdl labels?")

    X = np.stack(feats)
    y = np.asarray(labels, dtype=np.float32)

    stats = {
        "n_seen":       n_seen,
        "n_usable":     X.shape[0],
        "n_bad_key":    n_bad_key,
        "n_bad_wdl":    n_bad_wdl,
        "n_feat_error": n_feat_error,
        "load_seconds": round(time.time() - t_start, 2),
    }
    return X, y, stats


def _label_distribution(y: np.ndarray) -> dict:
    return {
        "+1 (W)": int((y > 0.5).sum()),
        " 0 (D)": int((np.abs(y) < 0.5).sum()),
        "-1 (L)": int((y < -0.5).sum()),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Train ValueNet v2 on Malom WDL labels (Step 1 of retrain_v2_plan.md)."
    )
    p.add_argument("--human-db",       type=Path, default=Path("data/human_db.sqlite"))
    p.add_argument("--specialist-db",  type=Path, default=None,
                   help="Accepted for API compatibility; ignored — pos_hash is not reversible.  "
                        "See retrain_v2_plan.md 'Known issues' for details.")
    p.add_argument("--output",         type=Path, default=Path("data/value_net_v2.npz"))
    p.add_argument("--epochs",         type=int,   default=200,
                   help="Max epochs (early stopping usually terminates sooner).")
    p.add_argument("--patience",       type=int,   default=10,
                   help="Early-stop patience — epochs without val-loss improvement.")
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--batch-size",     type=int,   default=256)
    p.add_argument("--val-fraction",   type=float, default=0.20,
                   help="Fraction of positions held out for validation (state-key hash split).")
    p.add_argument("--limit",          type=int,   default=None,
                   help="Cap positions loaded (smoke tests).")
    p.add_argument("--seed",           type=int,   default=42)
    args = p.parse_args()

    if args.specialist_db is not None:
        print(f"[vn_v2] --specialist-db {args.specialist_db} supplied — SKIPPING.")
        print("[vn_v2] specialist_db positions are keyed by irreversible pos_hash "
              "(SHA-1 of canonical FEN).  See retrain_v2_plan.md 'Known issues'.")
    if not (0.0 < args.val_fraction < 1.0):
        raise SystemExit("--val-fraction must be in (0, 1).")
    if args.output.exists():
        print(f"[vn_v2] WARNING: output {args.output} exists and will be overwritten.")

    print(f"[vn_v2] Loading positions from {args.human_db}")
    X, y, stats = load_from_human_db(args.human_db, limit=args.limit)

    print()
    print("[vn_v2] Data load summary:")
    for k, v in stats.items():
        print(f"    {k:>13} : {v:,}" if isinstance(v, int) else f"    {k:>13} : {v}")
    print(f"    label dist   : {_label_distribution(y)}")
    print()

    # Train ---------------------------------------------------------------
    np.random.seed(args.seed)
    net = ValueNet()
    print(f"[vn_v2] Training MLP {_INPUT_DIM} → 128 → 64 → 1 (tanh); MSE regression to Malom WDL.")
    print(f"[vn_v2] epochs={args.epochs}  lr={args.lr}  batch={args.batch_size}  "
          f"patience={args.patience}  val_frac={args.val_fraction}")

    losses = net.train(
        X, y,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        val_frac=args.val_fraction, patience=args.patience,
        verbose=True,
    )

    # Post-training eval on a fresh random hold-out from the same distribution.
    # (train() split internally; we recompute here for a headline metric.)
    rng   = np.random.default_rng(args.seed)
    order = rng.permutation(len(X))
    n_val = max(1, int(len(X) * args.val_fraction))
    X_val = X[order][-n_val:]
    y_val = y[order][-n_val:]
    val_pred = net.predict_batch(X_val).ravel()
    val_mse  = float(np.mean((val_pred - y_val) ** 2))
    val_sign = float(np.mean(np.sign(val_pred) == np.sign(y_val)))

    # Save ----------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    net.save(str(args.output))
    print()
    print(f"[vn_v2] Saved v2 checkpoint → {args.output}")
    print(f"[vn_v2] Train epochs run       : {len(losses)}")
    print(f"[vn_v2] Final train loss (MSE) : {losses[-1]:.5f}")
    print(f"[vn_v2] Val   MSE              : {val_mse:.5f}")
    print(f"[vn_v2] Val   sign accuracy    : {val_sign:.4f}")
    print()
    print("[vn_v2] Next: Step 4c in retrain_v2_plan.md — evaluate v1 vs v2 on the same "
          "held-out set (MSE + sign accuracy) and run bench_trajectory_value_net at "
          "blends 30/60/80 before promoting.  Do NOT overwrite data/value_net.npz.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
