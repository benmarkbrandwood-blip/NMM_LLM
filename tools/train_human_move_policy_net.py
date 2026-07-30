#!/usr/bin/env python3
"""tools/train_human_move_policy_net.py — HumanMovePolicyNet Phase 3 trainer.

Reads the extracted dataset produced by
`tools/extract_human_move_policy_dataset.py` and trains an 82 → 128 → 64 →
32 → 1 MLP that outputs a per-legal-move score.  Softmax over every
legal move at the position produces the calibrated human move
distribution `p(m | position, elo_band)`.

Loss (per plan §3.4): ordinary count-weighted cross-entropy over
observed move events.  No focal loss, no per-category reweighting.

    L = − Σ (position, band, move)  count_band(m) · log p_pred(m)
       normalised by total event count in the batch.

Saved as `.npz` mirroring `data/human_pref_net.npz` layout so
`ai/human_move_policy_advisor.py` can be a pure-numpy sibling to
`ai/human_pref_advisor.py`.

Usage
-----
    .venv/bin/python tools/train_human_move_policy_net.py \\
        --dataset-dir data/human_move_policy_dataset \\
        --output data/human_move_policy_net.npz
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.value_net import _INPUT_DIM  # noqa: E402


N_BANDS = 3
FEAT_DIM_WITH_BAND = _INPUT_DIM + N_BANDS


# ── Provenance ──────────────────────────────────────────────────────────────

def _git_head() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except Exception:
        return None


# ── Dataset loader ──────────────────────────────────────────────────────────

class MovePolicyDataset:
    """Reads the memmap + metadata produced by extract_human_move_policy_dataset.

    Presents `samples_train` / `samples_val` — each an object exposing
    `state_idx`, `band_idx`, `targets_flat`, `offsets` — for the loop.
    Successor features live in `succ_feats` (memmap) shared across bands
    via `state_succ_offsets`.
    """
    def __init__(self, dataset_dir: Path):
        meta_path = dataset_dir / "metadata.npz"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.npz not found in {dataset_dir}")
        d = np.load(meta_path, allow_pickle=True)
        self.state_keys         = d["state_keys"]
        self.mover_colors       = d["mover_colors"]
        self.state_succ_offsets = d["state_succ_offsets"].astype(np.int64)
        self.sample_state_idx   = d["sample_state_idx"].astype(np.int64)
        self.sample_band_idx    = d["sample_band_idx"].astype(np.int8)
        self.sample_targets     = d["sample_targets"].astype(np.int32)
        self.sample_offsets     = d["sample_offsets"].astype(np.int64)
        self.sample_is_val      = d["sample_is_val"].astype(bool)
        provenance_raw          = d["provenance"].item()
        if isinstance(provenance_raw, bytes):
            provenance_raw = provenance_raw.decode("utf-8")
        self.provenance = json.loads(provenance_raw)

        succ_bin = dataset_dir / self.provenance["successor_bank_bin"]
        n_rows   = int(self.provenance["successor_bank_rows"])
        feat_dim = int(self.provenance["feature_dim"])
        self.feat_dim = feat_dim
        self.succ_feats = np.memmap(
            succ_bin, dtype=np.float32, mode="r",
            shape=(n_rows, feat_dim),
        )

    @property
    def n_samples(self) -> int:
        return int(self.sample_state_idx.shape[0])

    def train_idx(self) -> np.ndarray:
        return np.where(~self.sample_is_val)[0]

    def val_idx(self) -> np.ndarray:
        return np.where(self.sample_is_val)[0]

    def sample_slice(self, sample_id: int) -> tuple[np.ndarray, np.ndarray, int]:
        """Return (succ_feats_matrix (legal, feat_dim), target_counts (legal,),
        band_idx) for one sample."""
        state_idx = int(self.sample_state_idx[sample_id])
        band      = int(self.sample_band_idx[sample_id])
        succ_start = int(self.state_succ_offsets[state_idx])
        succ_end   = int(self.state_succ_offsets[state_idx + 1])
        feats = self.succ_feats[succ_start:succ_end]
        tgt_start = int(self.sample_offsets[sample_id])
        tgt_end   = int(self.sample_offsets[sample_id + 1])
        targets   = self.sample_targets[tgt_start:tgt_end]
        return feats, targets, band


# ── Model + training loop ───────────────────────────────────────────────────

def _build_model(input_dim: int, dropout: float):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(64, 32),         nn.ReLU(),
        nn.Linear(32, 1),
    )


def _band_onehot(band_idx: int) -> np.ndarray:
    v = np.zeros(N_BANDS, dtype=np.float32)
    v[band_idx] = 1.0
    return v


def _save_npz(model, output: Path, provenance: dict) -> None:
    import torch.nn as nn
    arrays: dict[str, np.ndarray] = {}
    idx = 0
    for layer in model:
        if isinstance(layer, nn.Linear):
            arrays[f"w{idx}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
            arrays[f"b{idx}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
            idx += 1
    arrays["input_dim"]           = np.array([FEAT_DIM_WITH_BAND], dtype=np.int64)
    arrays["layer_count"]         = np.array([idx], dtype=np.int64)
    arrays["board_feature_dim"]   = np.array([_INPUT_DIM], dtype=np.int64)
    arrays["n_bands"]             = np.array([N_BANDS], dtype=np.int64)
    arrays["provenance_json"]     = np.array(json.dumps(provenance), dtype=object)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)


def _batches(indices: np.ndarray, batch_positions: int, rng: np.random.Generator):
    """Yield batches of sample-indices (position-level batching).  Each
    sample is a full (position, band) — legal-move count varies."""
    order = rng.permutation(indices)
    for start in range(0, len(order), batch_positions):
        end = min(start + batch_positions, len(order))
        yield order[start:end]


def train(args: argparse.Namespace) -> dict:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[hbn] device={device.type}")

    ds = MovePolicyDataset(args.dataset_dir)
    tr = ds.train_idx()
    va = ds.val_idx()
    print(f"[hbn] samples: train={len(tr):,}  val={len(va):,}  total={ds.n_samples:,}")

    model = _build_model(FEAT_DIM_WITH_BAND, dropout=args.dropout).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Precompute the 3 band one-hot tensors on-device.
    band_bits = torch.eye(N_BANDS, dtype=torch.float32, device=device)

    def _forward_and_loss_batch(sample_ids: np.ndarray):
        """Return (mean_event_loss_t, total_events, unnormalised_sum_t).
        `mean_event_loss_t` is the per-event NLL — divide the summed
        cross-entropy by total_events.  When the batch has no observed
        events (all sample targets are zero), returns (None, 0, None)."""
        losses:  list = []
        weights: list[int] = []
        total_events = 0
        for sid in sample_ids:
            feats_np, targets_np, band = ds.sample_slice(int(sid))
            if targets_np.sum() == 0:
                continue
            feats   = torch.from_numpy(np.array(feats_np,  dtype=np.float32)).to(device)
            targets = torch.from_numpy(np.array(targets_np, dtype=np.float32)).to(device)
            bhot    = band_bits[band].unsqueeze(0).expand(feats.shape[0], -1)
            x       = torch.cat([feats, bhot], dim=1)
            logits  = model(x).squeeze(-1)                   # (legal,)
            log_p   = F.log_softmax(logits, dim=0)
            sample_loss = -(targets * log_p).sum()
            sample_ev   = int(targets.sum().item())
            losses.append(sample_loss)
            weights.append(sample_ev)
            total_events += sample_ev
        if not losses:
            return None, 0, None
        stacked      = torch.stack(losses)
        unnormalised = stacked.sum()
        mean_event   = unnormalised / max(total_events, 1)
        return mean_event, total_events, unnormalised

    def _epoch_train(epoch: int) -> float:
        model.train()
        loss_sum   = 0.0
        event_sum  = 0
        n_batches  = 0
        rng = np.random.default_rng(args.seed + epoch)
        for sample_ids in _batches(tr, args.batch_positions, rng):
            mean_event_t, total_events, unnormalised = _forward_and_loss_batch(sample_ids)
            if total_events == 0:
                continue
            opt.zero_grad()
            mean_event_t.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            loss_sum  += float(unnormalised.item())
            event_sum += total_events
            n_batches += 1
        return loss_sum / max(event_sum, 1)

    def _epoch_val() -> float:
        model.eval()
        loss_sum  = 0.0
        event_sum = 0
        with torch.no_grad():
            for sample_ids in _batches(va, args.batch_positions, np.random.default_rng(0)):
                _, total_events, unnormalised = _forward_and_loss_batch(sample_ids)
                if total_events == 0:
                    continue
                loss_sum  += float(unnormalised.item())
                event_sum += total_events
        return loss_sum / max(event_sum, 1)

    best_val = float("inf")
    best_state: Optional[dict] = None
    stall = 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        train_loss = _epoch_train(ep)
        val_loss   = _epoch_val()
        marker = ""
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
            stall = 0
            marker = "  * new best"
        else:
            stall += 1
            marker = f"  (stall={stall})"
        print(f"[hbn] epoch {ep:3d}/{args.epochs}  train_nll={train_loss:.4f}  val_nll={val_loss:.4f}{marker}")
        if args.patience > 0 and stall >= args.patience:
            print(f"[hbn] Early stop after {ep} epochs (patience {args.patience}).")
            break

    if best_state is not None:
        import torch
        model.load_state_dict({k: torch.from_numpy(v) for k, v in best_state.items()})

    provenance = {
        "trainer_version":            "1",
        "trainer_git_commit":         _git_head() or "",
        "dataset_provenance":         ds.provenance,
        "feature_dim":                int(_INPUT_DIM),
        "n_bands":                    int(N_BANDS),
        "elo_band_config_name":       ds.provenance["elo_band_config_name"],
        "training_objective":         "count_weighted_ce",
        "hparams": {
            "lr":               args.lr,
            "dropout":          args.dropout,
            "batch_positions":  args.batch_positions,
            "epochs":           args.epochs,
            "patience":         args.patience,
            "grad_clip":        args.grad_clip,
            "seed":             args.seed,
        },
        "best_val_event_nll":         float(best_val),
        "final_epochs_run":           int(ep),
        "elapsed_seconds":            round(time.time() - t0, 1),
    }
    _save_npz(model, args.output, provenance)
    print(f"[hbn] Saved → {args.output}")
    print(f"[hbn] Best val event NLL: {best_val:.5f}")
    return provenance


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, default=Path("data/human_move_policy_dataset"))
    p.add_argument("--output",      type=Path, default=Path("data/human_move_policy_net.npz"))
    p.add_argument("--epochs",      type=int,   default=40)
    p.add_argument("--patience",    type=int,   default=6)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--batch-positions", type=int, default=512,
                   help="Number of (position, band) samples per gradient step.")
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--seed",        type=int,   default=42)
    args = p.parse_args()

    if args.output.exists():
        print(f"[hbn] WARNING: {args.output} exists and will be overwritten.")
    prov = train(args)
    (args.output.with_suffix(args.output.suffix + ".provenance.json")).write_text(
        json.dumps(prov, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
