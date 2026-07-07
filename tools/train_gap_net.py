#!/usr/bin/env python3
"""
tools/train_gap_net.py — Train the GapNet (blunder density network).

Loads data/gap_net_training.npz (built by scripts/build_gap_dataset.py).
Saves the trained network to data/gap_net.npz.
Does NOT touch data/value_net.npz.

Usage:
    .venv/bin/python tools/train_gap_net.py [--epochs N] [--lr F] [--data PATH]
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
    args = parser.parse_args()

    data_path = _ROOT / args.data
    out_path  = _ROOT / args.out

    if not data_path.exists():
        print(f"ERROR: Training data not found at {data_path}")
        print("Run: .venv/bin/python scripts/build_gap_dataset.py")
        sys.exit(1)

    data = np.load(str(data_path))
    X, y = data["X"].astype(np.float32), data["y"].astype(np.float32)
    print(f"Loaded {len(X)} samples from {data_path}")
    print(f"y stats: min={y.min():.3f} max={y.max():.3f} mean={y.mean():.3f}")

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
