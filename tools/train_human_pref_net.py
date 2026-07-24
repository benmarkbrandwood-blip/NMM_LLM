#!/usr/bin/env python3
"""tools/train_human_pref_net.py — Train HumanPrefNet on human_db moves.

Implements Step 2 of docs/retrain_v2_plan.md.  Trains a move-ranking network
on human game data so that we can later:
  * use it as an auxiliary signal for GapNet training / eval, and
  * offer a "human-like play" inference mode blending its scores with the
    heuristic engine.

Semantics
---------
Target: a scalar score per candidate successor board.  Higher score = more
"human-likely" to reach from the current position.  Only relative ordering
is trained (pairwise Bradley-Terry loss) — absolute magnitudes are
unconstrained.

Loss (Bradley-Terry / pairwise BCE):
    L = − log σ ( h(chosen_successor) − h(other_successor) )

per (chosen, other) pair, where `other` is any other legal successor from
the same position.

Data selection (per plan)
-------------------------
Query `human_db.moves` for rows with a Malom `malom_wdl_after` label.
`malom_wdl_after` is the WDL of the position *after* the human plays that
move, from the next player's perspective:
  L = next player loses → the human's move was winning
  D = draw               → the human's move was safe
  W = next player wins   → the human played a losing move

Per-position filter (avoids training on positions the human blundered):
  * If any recorded move from this state_key has malom_wdl_after='L',
    keep only records where the chosen move is also 'L'.
  * Else if any 'D' move recorded, keep records where the chosen is 'D'.
  * Skip any record where malom_wdl_after='W' (human played a losing move).

For each qualifying (position, chosen_move), enumerate all legal successor
boards, then pair (chosen_successor) against each other legal successor
(sampled up to `--pairs-per-position`) to form training pairs.

Board reconstruction: uses the D4-canonical state_key format
`canon24|turn|phase|placed_w|placed_b|on_w|on_b` (see
`ai.trajectory_db.make_board_state_key`).  Reconstruction is via the same
helper as ValueNet v2 (`board_from_state_key`).  Notation matching is a
plain string equality between the DB's `notation` column and
`_move_notation(move_dict)` on the canonical board.

Model + output
--------------
PyTorch MLP: 79 → 128 → 64 → 32 → 1 (ReLU + dropout).  After training,
weights are saved as a `.npz` (arrays `w0, b0, w1, b1, w2, b2, w3, b3`) so
inference does not require PyTorch.  A `data/human_pref_net.npz` companion
loader will live in Step 3 (`ai/human_pref_advisor.py`).

Usage
-----
    .venv/bin/python tools/train_human_pref_net.py \\
        --db data/human_db.sqlite \\
        --output data/human_pref_net.npz \\
        --patience 10

    # Smoke test (5000 positions, 3 epochs)
    .venv/bin/python tools/train_human_pref_net.py --limit 5000 --epochs 3
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from game.rules import get_all_legal_moves
from ai.value_net import board_to_features, _INPUT_DIM

# board_from_state_key lives in the ValueNet v2 script — reuse it verbatim so
# the two v2 targets share exactly the same reconstruction contract.
from tools.train_value_net_v2 import board_from_state_key


_UNICODE_X = "×"


def _move_notation(mv: dict) -> str:
    """Format a legal-move dict as the string stored in human_db.moves.notation.

    Formats:
      place              → to
      place + capture    → to  + 'x' + capture
      move               → from + '-' + to
      move  + capture    → from + '-' + to + 'x' + capture
    """
    frm = mv.get("from")
    to  = mv.get("to") or ""
    cap = mv.get("capture") or ""
    base = f"{frm}-{to}" if frm else to
    return f"{base}x{cap}" if cap else base


# ── Data pipeline ────────────────────────────────────────────────────────────

def _per_state_filter(malom_after_list: list[str]) -> str | None:
    """Return the WDL label to keep for this state, or None if the whole state is skipped.

    * If any 'L' → return 'L' (winning-move records only)
    * Elif any 'D' → return 'D' (draw records)
    * Else → None (all records were losing)
    """
    unique = set(malom_after_list)
    if "L" in unique:
        return "L"
    if "D" in unique:
        return "D"
    return None


def build_pairs(
    db_path: Path,
    limit_positions: int | None = None,
    pairs_per_position: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (X_pos, X_neg, stats) where each row is a chosen/other successor pair."""
    if not db_path.exists():
        raise FileNotFoundError(f"human_db not found at {db_path}")

    rng = random.Random(seed)
    conn = sqlite3.connect(str(db_path))
    # Grouping by state_key requires deterministic order.
    rows = conn.execute(
        "SELECT state_key, notation, malom_wdl_after FROM moves "
        "WHERE malom_wdl_after IS NOT NULL ORDER BY state_key"
    )

    pos_feats: list[np.ndarray] = []
    neg_feats: list[np.ndarray] = []
    stats = defaultdict(int)
    t0 = time.time()

    current_key: str | None = None
    buffered: list[tuple[str, str]] = []   # (notation, malom_wdl_after)
    n_positions_seen  = 0

    def _flush(key: str, records: list[tuple[str, str]]) -> None:
        nonlocal n_positions_seen
        n_positions_seen += 1
        stats["positions_seen"] += 1
        wdls = [r[1] for r in records]
        keep = _per_state_filter(wdls)
        if keep is None:
            stats["positions_skip_all_losing"] += 1
            return
        chosen_records = [r for r in records if r[1] == keep]
        if not chosen_records:
            stats["positions_skip_after_filter"] += 1
            return
        board = board_from_state_key(key)
        if board is None:
            stats["positions_bad_state_key"] += 1
            return
        legal = get_all_legal_moves(board)
        if len(legal) < 2:
            stats["positions_too_few_moves"] += 1
            return
        notation_to_move: dict[str, dict] = {}
        for m in legal:
            notation_to_move[_move_notation(m)] = m
        for chosen_notation, _ in chosen_records:
            move = notation_to_move.get(chosen_notation)
            if move is None:
                stats["notation_match_miss"] += 1
                continue
            try:
                chosen_board = board.apply_move(move)
                chosen_feat  = np.asarray(
                    board_to_features(chosen_board, chosen_board.turn),
                    dtype=np.float32,
                )
            except Exception:
                stats["chosen_apply_error"] += 1
                continue
            if chosen_feat.shape[0] != _INPUT_DIM:
                stats["chosen_feat_shape_error"] += 1
                continue
            others = [m for m in legal if _move_notation(m) != chosen_notation]
            if not others:
                continue
            sampled = others if len(others) <= pairs_per_position else rng.sample(others, pairs_per_position)
            for other in sampled:
                try:
                    other_board = board.apply_move(other)
                    other_feat  = np.asarray(
                        board_to_features(other_board, other_board.turn),
                        dtype=np.float32,
                    )
                except Exception:
                    stats["other_apply_error"] += 1
                    continue
                if other_feat.shape[0] != _INPUT_DIM:
                    stats["other_feat_shape_error"] += 1
                    continue
                pos_feats.append(chosen_feat)
                neg_feats.append(other_feat)
                stats["pairs"] += 1

    for state_key, notation, wdl_after in rows:
        if current_key is None:
            current_key = state_key
        if state_key != current_key:
            _flush(current_key, buffered)
            buffered = []
            current_key = state_key
            if limit_positions is not None and n_positions_seen >= limit_positions:
                break
        buffered.append((notation, wdl_after))
    else:
        # Flush the final group only if we exited normally.
        if current_key is not None and buffered:
            _flush(current_key, buffered)
    conn.close()

    stats["load_seconds"] = round(time.time() - t0, 2)

    if not pos_feats:
        raise RuntimeError(
            "No qualifying pairs produced.  Verify malom_wdl_after labels and "
            "notation canonicalisation."
        )

    X_pos = np.stack(pos_feats)
    X_neg = np.stack(neg_feats)
    return X_pos, X_neg, dict(stats)


# ── Model + training (PyTorch) ───────────────────────────────────────────────

def _build_model(input_dim: int = _INPUT_DIM, dropout: float = 0.2):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def _save_npz(model, output: Path) -> None:
    """Save PyTorch Sequential weights as an .npz mirroring the layer order."""
    import torch.nn as nn
    arrays: dict[str, np.ndarray] = {}
    idx = 0
    for layer in model:
        if isinstance(layer, nn.Linear):
            arrays[f"w{idx}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
            arrays[f"b{idx}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
            idx += 1
    # Provenance metadata (readable via np.load(...)['input_dim'], etc.)
    arrays["input_dim"]  = np.array([_INPUT_DIM], dtype=np.int64)
    arrays["layer_count"] = np.array([idx], dtype=np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)


def train_model(
    X_pos: np.ndarray, X_neg: np.ndarray,
    epochs: int, batch_size: int, lr: float, val_frac: float,
    patience: int, seed: int, output: Path,
) -> dict:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _build_model().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)

    N = X_pos.shape[0]
    n_val = max(1, int(N * val_frac))
    rng   = np.random.default_rng(seed)
    order = rng.permutation(N)
    val_idx = order[:n_val]
    tr_idx  = order[n_val:]

    X_pos_tr = torch.from_numpy(X_pos[tr_idx]).to(device)
    X_neg_tr = torch.from_numpy(X_neg[tr_idx]).to(device)
    X_pos_va = torch.from_numpy(X_pos[val_idx]).to(device)
    X_neg_va = torch.from_numpy(X_neg[val_idx]).to(device)

    def _epoch_loss(X_p, X_n) -> float:
        # Bradley-Terry: minimise -log σ(pos - neg) = softplus(-(pos - neg))
        with_grad = model.training
        n = X_p.shape[0]
        total = 0.0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            s_p = model(X_p[start:end]).squeeze(-1)
            s_n = model(X_n[start:end]).squeeze(-1)
            loss = F.softplus(-(s_p - s_n)).mean()
            total += float(loss.item()) * (end - start)
        return total / max(n, 1)

    best_val = float("inf")
    best_state: dict[str, np.ndarray] | None = None
    stall = 0

    print(f"[hp] device={device.type} train_pairs={len(tr_idx):,} val_pairs={len(val_idx):,}")
    for ep in range(1, epochs + 1):
        # Shuffle within train each epoch
        perm = rng.permutation(len(tr_idx))
        X_p_e = X_pos_tr[perm]
        X_n_e = X_neg_tr[perm]

        model.train()
        train_loss_sum = 0.0
        for start in range(0, len(tr_idx), batch_size):
            end = min(start + batch_size, len(tr_idx))
            s_p = model(X_p_e[start:end]).squeeze(-1)
            s_n = model(X_n_e[start:end]).squeeze(-1)
            loss = F.softplus(-(s_p - s_n)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.item()) * (end - start)
        train_loss = train_loss_sum / max(len(tr_idx), 1)

        model.eval()
        with torch.no_grad():
            val_loss = _epoch_loss(X_pos_va, X_neg_va)
            # Ranking accuracy: fraction of (pos, neg) with s(pos) > s(neg).
            s_p = model(X_pos_va).squeeze(-1)
            s_n = model(X_neg_va).squeeze(-1)
            val_acc = float((s_p > s_n).float().mean().item())

        marker = ""
        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
            stall = 0
            marker = "  * new best"
        else:
            stall += 1
            marker = f"  (stall={stall})"
        print(f"[hp] epoch {ep:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"acc={val_acc:.4f}{marker}")
        if patience > 0 and stall >= patience:
            print(f"[hp] Early stop after {ep} epochs (patience {patience}).")
            break

    if best_state is not None:
        model.load_state_dict({k: __import__('torch').from_numpy(v) for k, v in best_state.items()})

    # Final val accuracy on best weights
    import torch as _t
    model.eval()
    with _t.no_grad():
        s_p = model(X_pos_va).squeeze(-1)
        s_n = model(X_neg_va).squeeze(-1)
        final_val_acc = float((s_p > s_n).float().mean().item())

    _save_npz(model, output)
    return {"best_val_loss": best_val, "final_val_acc": final_val_acc}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Train HumanPrefNet on human_db moves (Step 2 of retrain_v2_plan.md)."
    )
    p.add_argument("--db",                type=Path, default=Path("data/human_db.sqlite"))
    p.add_argument("--output",            type=Path, default=Path("data/human_pref_net.npz"))
    p.add_argument("--epochs",            type=int,   default=100)
    p.add_argument("--patience",          type=int,   default=10)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--batch-size",        type=int,   default=512)
    p.add_argument("--val-fraction",      type=float, default=0.20)
    p.add_argument("--pairs-per-position", type=int,   default=4,
                   help="Max (chosen, other) pairs sampled per qualifying position.")
    p.add_argument("--limit",             type=int,   default=None,
                   help="Cap positions loaded (smoke tests).")
    p.add_argument("--seed",              type=int,   default=42)
    args = p.parse_args()

    if not (0.0 < args.val_fraction < 1.0):
        raise SystemExit("--val-fraction must be in (0, 1).")
    if args.output.exists():
        print(f"[hp] WARNING: output {args.output} exists and will be overwritten.")

    print(f"[hp] Building pairs from {args.db}")
    X_pos, X_neg, stats = build_pairs(
        args.db,
        limit_positions=args.limit,
        pairs_per_position=args.pairs_per_position,
        seed=args.seed,
    )
    print()
    print("[hp] Data build summary:")
    for k, v in stats.items():
        print(f"    {k:>30} : {v:,}" if isinstance(v, int) else f"    {k:>30} : {v}")
    print(f"    X_pos shape: {X_pos.shape}   X_neg shape: {X_neg.shape}")
    print()

    print(f"[hp] Training MLP {_INPUT_DIM} → 128 → 64 → 32 → 1 (Bradley-Terry pairwise BCE).")
    print(f"[hp] epochs={args.epochs}  lr={args.lr}  batch={args.batch_size}  "
          f"patience={args.patience}  val_frac={args.val_fraction}")

    result = train_model(
        X_pos, X_neg,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        val_frac=args.val_fraction, patience=args.patience, seed=args.seed,
        output=args.output,
    )

    print()
    print(f"[hp] Saved v1 HumanPrefNet → {args.output}")
    print(f"[hp] Best val loss        : {result['best_val_loss']:.5f}")
    print(f"[hp] Final val rank-acc   : {result['final_val_acc']:.4f}")
    print()
    print("[hp] Next: Step 3 in retrain_v2_plan.md — wire HumanPrefAdvisor into "
          "ai/game_ai.py so this checkpoint can be used at inference (both for the "
          "'human-like play' mode and as the human-proxy opponent in the Step 4 "
          "GapNet dataset build).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
