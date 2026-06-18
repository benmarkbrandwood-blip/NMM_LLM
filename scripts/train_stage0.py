"""scripts/train_stage0.py — Stage 0: supervised value-head pre-training.

Trains NMMNet's value head to predict the existing value_net's output.
Two phases:
  Phase 1 — backbone frozen, only value_head trained (high LR, fast convergence)
  Phase 2 — full network unfrozen, low LR, until val MSE plateaus

Exit criterion: val MSE doesn't improve for PATIENCE epochs.

Usage:
    .venv/bin/python scripts/train_stage0.py [--data PATH] [--resume PATH]
                                              [--epochs-p1 N] [--epochs-p2 N]
                                              [--batch-size N] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.models.backbone import NMMNet
from learned_ai.models.gnn_backbone import NMMGNNNet

PATIENCE = 3
DEFAULT_DATA = str(_ROOT / "learned_ai" / "data" / "stage0_positions.npz")
DEFAULT_OUT   = str(_ROOT / "learned_ai" / "checkpoints" / "stage0")


def _load_data(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    d = np.load(path)
    states = torch.from_numpy(d["states"])   # (N, 84) float32
    values = torch.from_numpy(d["values"])   # (N,)    float32
    print(f"Loaded {len(states)} positions from {path}")
    print(f"  Value range: [{values.min():.3f}, {values.max():.3f}]  "
          f"mean={values.mean():.3f}  std={values.std():.3f}")
    return states, values


def _make_loaders(states: torch.Tensor, values: torch.Tensor,
                  batch_size: int, val_frac: float = 0.15
                  ) -> tuple[DataLoader, DataLoader]:
    ds = TensorDataset(states, values)
    n_val = max(1, int(len(ds) * val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=False)
    print(f"Train: {n_train}  Val: {n_val}  batch={batch_size}")
    return train_dl, val_dl


def _run_epoch(model: NMMNet, loader: DataLoader, criterion: nn.Module,
               optimiser: torch.optim.Optimizer | None,
               device: torch.device) -> float:
    training = optimiser is not None
    model.train(training)
    total_loss = 0.0
    n = 0
    with torch.set_grad_enabled(training):
        for states_b, values_b in loader:
            states_b = states_b.to(device)
            values_b = values_b.to(device)
            # Only the value head is needed — use phase 0 but ignore logits
            feats = model.backbone(states_b)
            preds = model.value_head(feats).squeeze(-1)
            loss = criterion(preds, values_b)
            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total_loss += loss.item() * len(states_b)
            n += len(states_b)
    return total_loss / n if n else float("inf")


def _save_checkpoint(model: nn.Module, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    is_gnn = isinstance(model, NMMGNNNet)
    if is_gnn:
        cfg = {"head_hidden": [64], "dropout": 0.0}
    else:
        cfg = {
            "backbone_hidden": [l.out_features for l in model.backbone
                                if isinstance(l, nn.Linear)],
            "head_hidden": [64],
            "dropout": 0.0,
        }
    torch.save({
        "model": model.state_dict(),
        "model_config": cfg,
        "model_type": "gnn" if is_gnn else "mlp",
    }, path)
    print(f"  → saved {path}")


def train_phase(label: str, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader,
                lr: float, max_epochs: int, device: torch.device,
                frozen_backbone: bool, out_dir: Path) -> float:
    criterion = nn.MSELoss()

    if frozen_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        for p in model.value_head.parameters():
            p.requires_grad_(True)
        opt_params = model.value_head.parameters()
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        opt_params = model.parameters()

    optimiser = torch.optim.Adam(opt_params, lr=lr)

    best_val = float("inf")
    no_improve = 0
    print(f"\n── {label}  (lr={lr}  frozen_backbone={frozen_backbone}) ──")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = _run_epoch(model, train_dl, criterion, optimiser, device)
        val_loss   = _run_epoch(model, val_dl,   criterion, None,      device)
        elapsed = time.time() - t0
        improved = val_loss < best_val - 1e-6
        marker = "✓" if improved else " "
        print(f"  epoch {epoch:>2}/{max_epochs}  train={train_loss:.5f}  "
              f"val={val_loss:.5f}  {marker}  ({elapsed:.1f}s)")

        if improved:
            best_val = val_loss
            no_improve = 0
            _save_checkpoint(model, out_dir, "best.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop — no improvement for {PATIENCE} epochs.")
                break

    _save_checkpoint(model, out_dir, "latest.pt")
    return best_val


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 0 value head pre-training")
    ap.add_argument("--data",       default=DEFAULT_DATA, help="Path to stage0_positions.npz")
    ap.add_argument("--resume",     default=None,         help="Resume from NMMNet checkpoint")
    ap.add_argument("--epochs-p1",  type=int, default=20, help="Max epochs phase 1 (frozen backbone)")
    ap.add_argument("--epochs-p2",  type=int, default=30, help="Max epochs phase 2 (full network)")
    ap.add_argument("--lr-p1",      type=float, default=3e-3, help="Phase 1 learning rate")
    ap.add_argument("--lr-p2",      type=float, default=5e-4, help="Phase 2 learning rate")
    ap.add_argument("--batch-size", type=int, default=512,    help="Batch size")
    ap.add_argument("--out-dir",    default=DEFAULT_OUT,      help="Checkpoint output directory")
    ap.add_argument("--device",     default="auto")
    ap.add_argument("--gnn",        action="store_true", help="Use GNN backbone (NMMGNNNet)")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load or build model
    use_gnn = args.gnn
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        mc = ckpt.get("model_config", {})
        ckpt_type = ckpt.get("model_type", "mlp") if isinstance(ckpt, dict) else "mlp"
        if ckpt_type == "gnn" or use_gnn:
            model: nn.Module = NMMGNNNet(
                head_hidden=tuple(mc.get("head_hidden", [64])),
                dropout=float(mc.get("dropout", 0.0)),
            )
            use_gnn = True
        else:
            model = NMMNet(
                backbone_hidden=tuple(mc.get("backbone_hidden", [256, 256, 128])),
                head_hidden=tuple(mc.get("head_hidden", [64])),
                dropout=float(mc.get("dropout", 0.0)),
            )
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"Resumed from {args.resume}  (model_type={'gnn' if use_gnn else 'mlp'})")
    else:
        if use_gnn:
            model = NMMGNNNet()
            print("Initialised fresh NMMGNNNet (GCN backbone, 64 head)")
        else:
            model = NMMNet()
            print("Initialised fresh NMMNet (256-256-128 backbone, 64 head)")
    model = model.to(device)

    states, values = _load_data(args.data)
    train_dl, val_dl = _make_loaders(states, values, args.batch_size)
    out_dir = Path(args.out_dir)

    # Phase 1: frozen backbone
    best1 = train_phase(
        "Phase 1 — frozen backbone",
        model, train_dl, val_dl,
        lr=args.lr_p1, max_epochs=args.epochs_p1,
        device=device, frozen_backbone=True, out_dir=out_dir,
    )

    # Load best from phase 1 before phase 2
    p1_best = out_dir / "best.pt"
    if p1_best.exists():
        ckpt = torch.load(str(p1_best), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"Loaded phase 1 best (val={best1:.5f}) for phase 2")

    # Phase 2: full network
    best2 = train_phase(
        "Phase 2 — full network",
        model, train_dl, val_dl,
        lr=args.lr_p2, max_epochs=args.epochs_p2,
        device=device, frozen_backbone=False, out_dir=out_dir,
    )

    print(f"\nStage 0 complete.  Best val MSE: phase1={best1:.5f}  phase2={best2:.5f}")
    print(f"Checkpoints: {out_dir}/best.pt  (use as --resume for Stage 1)")


if __name__ == "__main__":
    main()
