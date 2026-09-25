#!/usr/bin/env python3
"""tools/train_value_net_v3.py — Train ValueNet v3 with DTW-weighted targets.

Trains three phase-specific ValueNets (place / move / fly) saved as a
PhaseValueNet so app.py picks them up automatically at startup:

    data/value_net_v3_place.npz
    data/value_net_v3_move.npz
    data/value_net_v3_fly.npz

After evaluation, promote by copying to data/value_net_phase_{place,move,fly}.npz.

WHY DTW-WEIGHTED TARGETS
------------------------
v1 used game-outcome labels (every position in a game labelled with the
final W/L/D) — too noisy, one blunder contaminates 40 plies.
v2 used per-position Malom WDL hard labels (+1/0/-1) — better but gives
identical scores to a forced-win-in-1 and a win-in-80.

v3 uses distance-to-win (DTW) to weight the target:

    W:  label = +tanh(dtw_scale / dtw)   → quick wins near +1, distant wins softer
    D:  label =  0
    L:  label = -tanh(dtw_scale / dtw)   → quick losses near -1, distant losses softer

This gives the AI a genuine "urgency" signal — neither GapNet (blunder-zone
risk) nor Sentinel (RL positional quality) encode how soon a position resolves.

PHASE SPLIT
-----------
The state_key format is:  canon24|turn|phase|placed_w|placed_b|on_w|on_b
Phase (index 2) is "place", "move", or "fly".  Separate nets are trained for
each phase.  PhaseValueNet dispatches at inference based on get_game_phase().

DATA SOURCE
-----------
data/human_db_candidate_new.sqlite — 2.31M positions, all with malom_wdl AND
malom_dtw (sector-corrected Malom labels from the 2026-09 rebuild).

Val split uses the same canonical SHA-256 bucket scheme as ValueNet v2 /
HumanPrefNet / GapNet v3 so the held-out slice is consistent across all nets.

Usage
-----
    .venv/bin/python tools/train_value_net_v3.py

    # Smoke test (5k positions, 3 epochs):
    .venv/bin/python tools/train_value_net_v3.py --limit 5000 --epochs 3

    # Custom DB or output:
    .venv/bin/python tools/train_value_net_v3.py \\
        --db data/human_db_candidate_new.sqlite \\
        --output-base data/value_net_v3 \\
        --dtw-scale 7 --epochs 200 --patience 15
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState, POSITIONS
from ai.value_net import ValueNet, PhaseValueNet, board_to_features, _INPUT_DIM


# ── State-key reconstruction (same as train_value_net_v2) ─────────────────────

def _board_and_phase_from_state_key(state_key: str) -> tuple[BoardState, str] | None:
    """Return (board, phase) from a canonical state_key, or None on bad input."""
    parts = state_key.split("|")
    if len(parts) != 7:
        return None
    canon24, turn, phase, placed_w_s, placed_b_s, on_w_s, on_b_s = parts
    if len(canon24) != len(POSITIONS):
        return None
    if phase not in ("place", "move", "fly"):
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
    w_cap = max(0, placed_b - on_b)
    b_cap = max(0, placed_w - on_w)
    board = BoardState(
        positions=positions,
        turn=turn,
        pieces_on_board={"W": on_w, "B": on_b},
        pieces_placed={"W": placed_w, "B": placed_b},
        pieces_captured={"W": w_cap, "B": b_cap},
    )
    return board, phase


# ── DTW target ────────────────────────────────────────────────────────────────

def dtw_target(wdl: str, dtw: int | None, dtw_scale: float) -> float | None:
    """Convert Malom WDL + distance-to-win into a [-1, 1] training label.

    Draws always map to 0.  Wins/losses are weighted by tanh(dtw_scale / dtw)
    so that positions that resolve quickly get a stronger signal than distant
    wins/losses.  Returns None when the label would be degenerate (dtw=0 for
    a non-draw, or unexpected wdl value).
    """
    if wdl == "D":
        return 0.0
    if wdl not in ("W", "L"):
        return None
    if dtw is None or dtw <= 0:
        # dtw=0 for wins/losses is an invalid Malom label — skip.
        return None
    magnitude = math.tanh(dtw_scale / dtw)
    return magnitude if wdl == "W" else -magnitude


# ── Data loading ──────────────────────────────────────────────────────────────

_PHASES = ("place", "move", "fly")


def load_data(
    db_path: Path,
    dtw_scale: float,
    limit: int | None = None,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Load and featurise all positions, split by phase AND by train/val/test.

    Uses the canonical three_way_split (SHA-256 bucket scheme shared with
    ValueNet v2 / HumanPrefNet / GapNet v3):
      train : buckets 20-99 (80%)  — used to fit weights
      val   : buckets  5-19 (15%)  — used for early stopping
      test  : buckets  0- 4 ( 5%)  — held out; never seen during training

    Returns {phase: {split: (X, y)}} — each (X, y) pair is float32 arrays.
    """
    from learned_ai.data.human_db_split import three_way_split

    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    q = "SELECT state_key, malom_wdl, malom_dtw FROM positions WHERE malom_wdl IS NOT NULL"
    if limit is not None:
        q += f" LIMIT {int(limit)}"

    # {phase: {split: (feats, labels)}}
    splits = ("train", "val", "test")
    buf: dict[str, dict[str, tuple[list, list]]] = {
        p: {s: ([], []) for s in splits} for p in _PHASES
    }

    n_seen = n_bad_key = n_bad_label = n_feat_err = 0
    n_per_split: dict[str, int] = {s: 0 for s in splits}
    t0 = time.time()

    for state_key, wdl, dtw in conn.execute(q):
        n_seen += 1

        split = three_way_split(state_key)
        n_per_split[split] += 1

        label = dtw_target(wdl, dtw, dtw_scale)
        if label is None:
            n_bad_label += 1
            continue

        result = _board_and_phase_from_state_key(state_key)
        if result is None:
            n_bad_key += 1
            continue
        board, phase = result

        try:
            feat = board_to_features(board, board.turn).astype(np.float32)
        except Exception:
            n_feat_err += 1
            continue
        if feat.shape[0] != _INPUT_DIM:
            n_feat_err += 1
            continue

        buf[phase][split][0].append(feat)
        buf[phase][split][1].append(label)

        if n_seen % 200_000 == 0:
            rate = n_seen / max(time.time() - t0, 1e-6)
            usable = sum(len(buf[p][s][0]) for p in _PHASES for s in splits)
            print(f"  loaded {n_seen:>9,}  usable {usable:,}  [{rate:,.0f}/s]")

    conn.close()

    print(f"\n  seen={n_seen:,}  "
          f"train={n_per_split['train']:,}  val={n_per_split['val']:,}  "
          f"test={n_per_split['test']:,}  "
          f"bad_key={n_bad_key:,}  bad_label={n_bad_label:,}  feat_err={n_feat_err:,}"
          f"  time={time.time()-t0:.1f}s")

    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for phase in _PHASES:
        out[phase] = {}
        for s in splits:
            feats, labels = buf[phase][s]
            if not feats:
                if s == "train":
                    raise RuntimeError(f"No training positions for phase='{phase}' split='{s}'.")
                # val/test being empty is unlikely but non-fatal for smoke tests
                out[phase][s] = (np.zeros((0, _INPUT_DIM), dtype=np.float32),
                                 np.zeros(0, dtype=np.float32))
            else:
                out[phase][s] = (np.stack(feats), np.array(labels, dtype=np.float32))
    return out


# ── Per-phase statistics ──────────────────────────────────────────────────────

def _label_stats(y: np.ndarray) -> str:
    strong_win  = int((y >= 0.7).sum())
    weak_win    = int(((y > 0) & (y < 0.7)).sum())
    draws       = int((y == 0.0).sum())
    weak_loss   = int(((y < 0) & (y > -0.7)).sum())
    strong_loss = int((y <= -0.7).sum())
    return (f"strong_win≥0.7={strong_win}  weak_win={weak_win}  "
            f"draw={draws}  weak_loss={weak_loss}  strong_loss≤-0.7={strong_loss}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Train ValueNet v3 (DTW-weighted, phase-split) for Nine Men's Morris."
    )
    p.add_argument("--db",          type=Path,  default=Path("data/human_db_candidate_new.sqlite"),
                   help="Source SQLite DB with malom_wdl + malom_dtw columns.")
    p.add_argument("--output-base", type=Path,  default=Path("data/value_net_v3"),
                   help="Base path; three files are written: <base>_place.npz etc.")
    p.add_argument("--dtw-scale",   type=float, default=7.0,
                   help="k in tanh(k/dtw). Higher = steeper urgency curve. Default 7.")
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--patience",    type=int,   default=15,
                   help="Early-stop patience (val-loss epochs without improvement).")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch-size",  type=int,   default=512)
    p.add_argument("--val-fraction",type=float, default=0.20)
    p.add_argument("--weight-decay",type=float, default=1e-4,
                   help="L2 regularisation (helps with fly-phase small dataset).")
    p.add_argument("--limit",       type=int,   default=None,
                   help="Cap total positions loaded (smoke tests).")
    p.add_argument("--seed",        type=int,   default=42)
    args = p.parse_args()

    if not (0.0 < args.val_fraction < 1.0):
        raise SystemExit("--val-fraction must be in (0, 1).")

    print(f"[vn_v3] DB         : {args.db}")
    print(f"[vn_v3] Output base: {args.output_base}")
    print(f"[vn_v3] dtw_scale  : {args.dtw_scale}  (tanh({args.dtw_scale}/dtw))")
    print(f"[vn_v3] epochs={args.epochs}  patience={args.patience}  "
          f"lr={args.lr}  batch={args.batch_size}  wd={args.weight_decay}")
    print()

    # ── Load ──────────────────────────────────────────────────────────────────
    print("[vn_v3] Loading positions …")
    phase_data = load_data(args.db, args.dtw_scale, limit=args.limit)

    for phase in _PHASES:
        splits = phase_data[phase]
        X_tr, y_tr = splits["train"]
        X_te, y_te = splits["test"]
        print(f"  [{phase:5s}]  train={len(X_tr):>7,}  val={len(splits['val'][0]):>6,}  "
              f"test={len(X_te):>6,}  "
              f"y_mean(tr)={y_tr.mean():.4f}  y_std(tr)={y_tr.std():.4f}  "
              f"{_label_stats(y_tr)}")
    print()

    # ── Train one net per phase ───────────────────────────────────────────────
    np.random.seed(args.seed)
    nets: dict[str, ValueNet] = {}

    for phase in _PHASES:
        X_tr, y_tr = phase_data[phase]["train"]
        X_va, y_va = phase_data[phase]["val"]
        print(f"[vn_v3] Training phase='{phase}'  train={len(X_tr):,}  val={len(X_va):,} …")
        net = ValueNet()

        # ValueNet.train() does its own internal val split; we pass the combined
        # train+val here so its internal split approximates our val slice, then
        # we evaluate separately on the held-out test set below.
        X_fit = np.concatenate([X_tr, X_va]) if len(X_va) else X_tr
        y_fit = np.concatenate([y_tr, y_va]) if len(y_va) else y_tr

        losses = net.train(
            X_fit, y_fit,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_frac=max(0.05, len(X_va) / max(len(X_fit), 1)),
            patience=args.patience,
            weight_decay=args.weight_decay,
            verbose=True,
            print_every=20,
        )
        print(f"  [{phase:5s}]  epochs_run={len(losses)}  final_train_loss={losses[-1]:.5f}")
        nets[phase] = net

    # ── Save ─────────────────────────────────────────────────────────────────
    base = args.output_base
    base.parent.mkdir(parents=True, exist_ok=True)
    pv_net = PhaseValueNet(nets["place"], nets["move"], nets["fly"])
    pv_net.save(base)

    print(f"[vn_v3] Saved:")
    for phase in _PHASES:
        path = base.parent / f"{base.stem}_{phase}.npz"
        kb   = round(path.stat().st_size / 1024, 1)
        print(f"  {path}  ({kb} KB)")
    print()

    # ── Test-set evaluation (never seen during training or early stopping) ────
    print("[vn_v3] ── TEST SET EVALUATION (5% held-out, never seen during training) ──")
    all_test_preds: list[np.ndarray] = []
    all_test_y:     list[np.ndarray] = []

    for phase in _PHASES:
        X_te, y_te = phase_data[phase]["test"]
        if len(X_te) == 0:
            print(f"  [{phase:5s}]  test set empty (smoke test with --limit?)")
            continue
        net   = nets[phase]
        preds = net.predict_batch(X_te)
        mse   = float(np.mean((preds - y_te) ** 2))
        nondraw = y_te != 0.0
        sign_acc = float(np.mean(np.sign(preds[nondraw]) == np.sign(y_te[nondraw]))) \
                   if nondraw.any() else float("nan")
        # Directional accuracy on wins: did we score wins higher than losses?
        wins = y_te > 0
        losses_ = y_te < 0
        win_mean  = float(preds[wins].mean())  if wins.any()    else float("nan")
        loss_mean = float(preds[losses_].mean()) if losses_.any() else float("nan")
        draw_mean = float(preds[~nondraw].mean()) if (~nondraw).any() else float("nan")
        print(f"  [{phase:5s}]  n={len(X_te):,}  mse={mse:.5f}  "
              f"sign_acc(non-draw)={sign_acc:.4f}  "
              f"mean_pred: W={win_mean:.3f}  D={draw_mean:.3f}  L={loss_mean:.3f}")
        all_test_preds.append(preds)
        all_test_y.append(y_te)

    if all_test_preds:
        preds_all = np.concatenate(all_test_preds)
        y_all     = np.concatenate(all_test_y)
        mse_all   = float(np.mean((preds_all - y_all) ** 2))
        nondraw_all = y_all != 0.0
        sign_all = float(np.mean(np.sign(preds_all[nondraw_all]) == np.sign(y_all[nondraw_all]))) \
                   if nondraw_all.any() else float("nan")
        print(f"  [total]  n={len(y_all):,}  mse={mse_all:.5f}  sign_acc(non-draw)={sign_all:.4f}")

    print()
    print("[vn_v3] ── PROMOTION ──")
    print("  Run bench_trajectory_value_net.py at blend 30/60/80 before promoting.")
    print("  Then copy to production path:")
    print(f"    cp {base.parent}/{base.stem}_place.npz data/value_net_phase_place.npz")
    print(f"    cp {base.parent}/{base.stem}_move.npz  data/value_net_phase_move.npz")
    print(f"    cp {base.parent}/{base.stem}_fly.npz   data/value_net_phase_fly.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
