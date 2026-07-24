#!/usr/bin/env python3
"""
tools/train_gap_net.py — Train the GapNet (blunder density network).

Loads a dataset built by scripts/build_gap_dataset.py.  The v2 dataset
(gap_net_training_v2.npz) additionally carries a `y_hp` auxiliary label
(HumanPrefNet disagreement — malom_top_q − malom_q_of_hp_top).  When
`--hp-blend > 0`, this label is added into the training target with the
given weight and the sum is clipped to [-1, 1].  Positions with NaN y_hp
(older datasets or synthetic-fallback samples) fall back to the plain
gap target.

Usage:
    # v1 backwards compat (uses only y):
    .venv/bin/python tools/train_gap_net.py --epochs 80

    # v2 (retrain_v2_plan.md Step 5):
    .venv/bin/python tools/train_gap_net.py \\
        --data data/gap_net_training_v2.npz \\
        --out  data/gap_net_v2.npz \\
        --epochs 80 --hp-blend 0.3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
from ai.value_net import ValueNet  # GapNet IS ValueNet architecture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data", default="data/gap_net_training.npz")
    parser.add_argument("--out", default="data/gap_net.npz")
    parser.add_argument("--hp-blend", type=float, default=0.0,
                        help="Weight for hp_disagreement auxiliary label added "
                             "into y (v2 datasets).  0 = ignore.")
    args = parser.parse_args()

    data_path = _ROOT / args.data
    out_path  = _ROOT / args.out

    if not data_path.exists():
        print(f"ERROR: Training data not found at {data_path}")
        print("Run: .venv/bin/python scripts/build_gap_dataset.py")
        sys.exit(1)

    data = np.load(str(data_path))
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    print(f"Loaded {len(X)} samples from {data_path}")
    print(f"y stats: min={y.min():.3f} max={y.max():.3f} mean={y.mean():.3f}")

    # v2 auxiliary label enrichment (Step 4 of retrain_v2_plan.md)
    if args.hp_blend != 0.0 and "y_hp" in data.files:
        y_hp = data["y_hp"].astype(np.float32)
        valid_mask = ~np.isnan(y_hp)
        n_valid = int(valid_mask.sum())
        if n_valid > 0:
            # y_hp in [0, 1] (malom_top_q - malom_q_of_hp_top).  Add scaled hp
            # signal to y so positions with strong HP-vs-Malom disagreement are
            # pushed toward the blunder-zone end of the target range.
            y_blended = y.copy()
            y_blended[valid_mask] = np.clip(
                y[valid_mask] + args.hp_blend * y_hp[valid_mask], -1.0, 1.0
            )
            print(f"y_hp blend {args.hp_blend}: applied to {n_valid}/{len(y)} samples "
                  f"({100*n_valid/len(y):.1f}%)")
            print(f"y (blended) stats: min={y_blended.min():.3f} "
                  f"max={y_blended.max():.3f} mean={y_blended.mean():.3f}")
            y = y_blended
        else:
            print("y_hp present but all NaN — training on plain y.")
    elif args.hp_blend != 0.0:
        print("--hp-blend requested but dataset has no y_hp array — training on plain y.")

    net = ValueNet()
    print(f"Training GapNet for {args.epochs} epochs (lr={args.lr})...")
    losses = net.train(X, y, epochs=args.epochs, lr=args.lr, verbose=True)
    print(f"Final loss: {losses[-1]:.5f}")

    net.save(out_path)
    size_kb = round(out_path.stat().st_size / 1024, 1)
    print(f"Saved gap_net to {out_path} ({size_kb} KB)")

    # Quick validation
    preds = net.predict_batch(X[:500])
    print(f"Prediction range on first 500 samples: [{preds.min():.3f}, {preds.max():.3f}]")


if __name__ == "__main__":
    main()
