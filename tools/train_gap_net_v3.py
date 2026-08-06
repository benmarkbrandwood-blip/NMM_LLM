#!/usr/bin/env python3
"""tools/train_gap_net_v3.py — Train GapNet v3 four-head MLP (§11, gap_net_v3_plan.md).

Architecture (§11.1):  79 → 128 → 64 → 32 → 4
Targets: four G_v regret components from data/gap_net_v3_dataset/.
Component D (index 3) is always NaN in regret_v1 and receives no gradient.
NaN masking is applied per-sample so partial future coverage degrades gracefully.
D4 symmetry augmentation (§11.2): each training sample is randomly rotated/reflected.

Split encoding in metadata.npz:  0 = train, 1 = val, 2 = test (never touched here).

Usage:
    .venv/bin/python tools/train_gap_net_v3.py
    .venv/bin/python tools/train_gap_net_v3.py --epochs 100 --lr 3e-4 --out data/gap_net_v3_candidate.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ai.board_symmetry import _BOARD_PERM as _BD_PERM, SYM_INVERSE as _SYM_INV

# ── Constants ────────────────────────────────────────────────────────────────

_INPUT_DIM = 79
_H1, _H2, _H3, _N_HEADS = 128, 64, 32, 4


# ── D4 feature permutations (§11.2) ──────────────────────────────────────────

def _build_feat_perms() -> list[np.ndarray]:
    """Build 79-dim permutation arrays for all 8 D4 symmetries.

    Features 0–71 are 24 × 3 board one-hots (own/opp/empty per position, in
    POSITIONS order).  Features 72–78 are metadata scalars invariant under board
    symmetry.  perm[j] is the source index that provides the value for output
    index j after the transform.
    """
    perms = []
    for sym_idx in range(8):
        perm = np.arange(79, dtype=np.int64)
        inv_bp = _BD_PERM[_SYM_INV[sym_idx]]  # maps new_pos → old_pos
        if inv_bp is not None:
            for new_pos, old_pos in enumerate(inv_bp):
                for c in range(3):
                    perm[new_pos * 3 + c] = old_pos * 3 + c
        perms.append(perm)
    return perms

_FEAT_PERMS: list[np.ndarray] = _build_feat_perms()


# ── Model ────────────────────────────────────────────────────────────────────

class GapNetV3(nn.Module):
    """79 → 128 → 64 → 32 → 4 MLP, linear output heads."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_INPUT_DIM, _H1), nn.ReLU(),
            nn.Linear(_H1, _H2),        nn.ReLU(),
            nn.Linear(_H2, _H3),        nn.ReLU(),
            nn.Linear(_H3, _N_HEADS),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── NaN-masked MSE loss ───────────────────────────────────────────────────────

def _nan_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over non-NaN (pred, target) pairs; returns scalar 0 if all NaN."""
    mask = ~torch.isnan(target)
    if not mask.any():
        return pred.sum() * 0.0
    return ((pred[mask] - target[mask]) ** 2).mean()


# ── Dataset loading ───────────────────────────────────────────────────────────

def _load_split(
    dataset_dir: Path, split_val: int, load_empirical: bool = False
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load features and targets for one split as float32 tensors.

    When load_empirical=True, also returns empirical G_v targets (NaN where absent).
    """
    meta = np.load(str(dataset_dir / "metadata.npz"), allow_pickle=True)
    idx = np.where(meta["split"] == split_val)[0]

    feats_mm = np.memmap(
        str(dataset_dir / "parent_feats.f32.bin"),
        dtype="float32", mode="r", shape=(len(meta["split"]), _INPUT_DIM),
    )
    tgts_mm = np.memmap(
        str(dataset_dir / "targets.f32.bin"),
        dtype="float32", mode="r", shape=(len(meta["split"]), _N_HEADS),
    )

    X = torch.from_numpy(feats_mm[idx].copy())
    y = torch.from_numpy(tgts_mm[idx].copy())

    if load_empirical:
        emp_mm = np.memmap(
            str(dataset_dir / "targets_empirical.f32.bin"),
            dtype="float32", mode="r", shape=(len(meta["split"]), _N_HEADS),
        )
        y_emp = torch.from_numpy(emp_mm[idx].copy())
        return X, y, y_emp

    return X, y


# ── Chunked forward pass ──────────────────────────────────────────────────────

def _forward_chunked(model: GapNetV3, X: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
    """Forward pass in chunks to avoid OOM on large val/test sets."""
    parts = []
    for i in range(0, len(X), chunk):
        parts.append(model(X[i:i + chunk]))
    return torch.cat(parts, dim=0)


# ── Provenance ────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Save ──────────────────────────────────────────────────────────────────────

def _save(model: GapNetV3, path: Path, provenance: dict) -> None:
    weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(
        str(path),
        **weights,
        provenance=np.array(json.dumps(provenance), dtype=object),
        architecture=np.array(
            json.dumps({"input": _INPUT_DIM, "hidden": [_H1, _H2, _H3], "heads": _N_HEADS}),
            dtype=object,
        ),
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default=str(_ROOT / "data" / "gap_net_v3_dataset"))
    p.add_argument("--out", default=str(_ROOT / "data" / "gap_net_v3_candidate.npz"))
    p.add_argument("--epochs",     type=int,   default=80)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--batch-size", type=int,   default=4096)
    p.add_argument("--patience",   type=int,   default=15,
                   help="Early-stop after N epochs without val improvement")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gap_net_v3] device={device}")

    dataset_dir = Path(args.dataset_dir)
    out_path    = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load train / val ─────────────────────────────────────────────────────
    print("[gap_net_v3] Loading train split …")
    t0 = time.time()
    X_tr, y_tr = _load_split(dataset_dir, split_val=0)
    print(f"[gap_net_v3] Train: {len(X_tr):,} rows  ({time.time()-t0:.1f}s)")

    print("[gap_net_v3] Loading val split (+ empirical targets) …")
    t0 = time.time()
    X_val, y_val, y_val_emp = _load_split(dataset_dir, split_val=1, load_empirical=True)
    print(f"[gap_net_v3] Val:   {len(X_val):,} rows  ({time.time()-t0:.1f}s)")

    # Report target coverage per component
    for i in range(_N_HEADS):
        n_valid = int((~torch.isnan(y_tr[:, i])).sum())
        print(f"[gap_net_v3]   comp {i}: {n_valid:,}/{len(y_tr):,} valid train targets "
              f"({'active' if n_valid > 0 else 'NaN-only — no gradient'})")

    # ── Uniform baseline: training-set mean per component ────────────────────
    tr_means: list[float] = []
    for i in range(_N_HEADS):
        mask = ~torch.isnan(y_tr[:, i])
        tr_means.append(float(y_tr[mask, i].mean()) if mask.any() else float("nan"))
    print(f"[gap_net_v3] Train component means (uniform baseline): "
          f"{[f'{v:.6f}' for v in tr_means]}")

    # ── Build DataLoader ──────────────────────────────────────────────────────
    tr_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # ── D4 permutation tensor (§11.2) ─────────────────────────────────────────
    perm_t = torch.from_numpy(np.stack(_FEAT_PERMS)).long().to(device)  # (8, 79)

    # ── Model + optimiser ─────────────────────────────────────────────────────
    model = GapNetV3().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[gap_net_v3] Model: {n_params:,} parameters")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    X_val_dev = X_val.to(device)
    y_val_dev = y_val.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_steps = 0.0, 0
        t_ep = time.time()

        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            # D4 symmetry augmentation (§11.2): pick one of 8 symmetries per sample
            sym_choice = torch.randint(0, 8, (xb.size(0),), device=device)
            gather_idx = perm_t[sym_choice]          # (B, 79)
            xb = torch.gather(xb, 1, gather_idx)
            pred = model(xb)
            loss = _nan_mse(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss  += loss.item()
            ep_steps += 1

        tr_loss = ep_loss / max(ep_steps, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = _forward_chunked(model, X_val_dev)
            val_loss = _nan_mse(val_pred, y_val_dev).item()

        elapsed = time.time() - t_ep
        print(f"[gap_net_v3] epoch {epoch:3d}/{args.epochs}  "
              f"tr={tr_loss:.6f}  val={val_loss:.6f}  ({elapsed:.1f}s)")

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"[gap_net_v3] Early stop at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    # ── Restore best weights ──────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[gap_net_v3] Restored best checkpoint (val_loss={best_val_loss:.6f})")

    # ── Per-component val MSE + §16 Stage E gate baselines ───────────────────
    model.eval()
    with torch.no_grad():
        val_pred_np = _forward_chunked(model, X_val_dev).cpu().numpy()
    y_val_np     = y_val.numpy()
    y_val_emp_np = y_val_emp.numpy()

    print("[gap_net_v3] Per-component val MSE (§16 Stage E gate):")
    gate_results: list[dict | None] = []
    for i in range(_N_HEADS):
        mask = ~np.isnan(y_val_np[:, i])
        if not mask.any():
            print(f"  comp {i}: NaN-only — skipped")
            gate_results.append(None)
            continue

        model_mse = float(np.mean((val_pred_np[mask, i] - y_val_np[mask, i]) ** 2))
        unif_mse  = float(np.mean((tr_means[i] - y_val_np[mask, i]) ** 2)) \
                    if not np.isnan(tr_means[i]) else float("nan")

        # Empirical baseline: val rows where empirical G_v is also available
        emp_mask = mask & ~np.isnan(y_val_emp_np[:, i])
        row: dict = {"model_mse": model_mse, "uniform_mse": unif_mse,
                     "n_val": int(mask.sum())}

        if emp_mask.any():
            emp_mse       = float(np.mean((y_val_emp_np[emp_mask, i] - y_val_np[emp_mask, i]) ** 2))
            model_mse_emp = float(np.mean((val_pred_np[emp_mask, i]  - y_val_np[emp_mask, i]) ** 2))
            gate_a = (unif_mse - model_mse)       / unif_mse * 100 if unif_mse > 0 else float("nan")
            gate_b = (emp_mse  - model_mse_emp)   / emp_mse  * 100 if emp_mse  > 0 else float("nan")
            row.update({"empirical_mse": emp_mse, "model_mse_on_emp_subset": model_mse_emp,
                        "n_emp": int(emp_mask.sum()), "gate_a_pct": gate_a, "gate_b_pct": gate_b})
            pass_str = "✓ PASS" if gate_a >= 30 and gate_b >= 10 else "✗ FAIL"
            print(f"  comp {i}: model={model_mse:.6f}  uniform={unif_mse:.6f}  "
                  f"empirical={emp_mse:.6f}  (n={mask.sum():,}, n_emp={emp_mask.sum():,})")
            print(f"           gate_a(≥30%): {gate_a:+.1f}%  gate_b(≥10%): {gate_b:+.1f}%  {pass_str}")
        else:
            gate_a = (unif_mse - model_mse) / unif_mse * 100 if unif_mse > 0 else float("nan")
            row.update({"empirical_mse": None, "gate_a_pct": gate_a, "gate_b_pct": None})
            pass_str = "✓" if gate_a >= 30 else "✗"
            print(f"  comp {i}: model={model_mse:.6f}  uniform={unif_mse:.6f}  "
                  f"empirical=N/A  (n={mask.sum():,})")
            print(f"           gate_a(≥30%): {gate_a:+.1f}% {pass_str}  gate_b: N/A")

        gate_results.append(row)

    provenance = {
        "model":            "gap_net_v3_candidate",
        "architecture":     f"{_INPUT_DIM}→{_H1}→{_H2}→{_H3}→{_N_HEADS}",
        "dataset_dir":      str(dataset_dir),
        "dataset_sha256":   _sha256(dataset_dir / "parent_feats.f32.bin"),
        "n_train":          len(X_tr),
        "n_val":            len(X_val),
        "best_val_loss":    best_val_loss,
        "epochs_trained":   epoch,
        "lr":               args.lr,
        "batch_size":       args.batch_size,
        "weight_decay":     args.weight_decay,
        "seed":             args.seed,
        "d4_augmentation":  True,
        "tr_means":         tr_means,
        "gate_results":     gate_results,
        "git_commit":       _git_commit(),
        "built_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    model_cpu = model.cpu()
    _save(model_cpu, out_path, provenance)
    size_kb = round(out_path.stat().st_size / 1024, 1)
    print(f"[gap_net_v3] Saved → {out_path}  ({size_kb} KB)")
    print(f"[gap_net_v3] Provenance: {json.dumps(provenance, indent=2)}")


if __name__ == "__main__":
    main()
