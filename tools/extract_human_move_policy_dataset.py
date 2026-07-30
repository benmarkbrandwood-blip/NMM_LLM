#!/usr/bin/env python3
"""tools/extract_human_move_policy_dataset.py — one-shot dataset extractor.

Phase 3 preamble.  Reads the v3 candidate HumanDB, enumerates every
legal move at each unique state_key, encodes successor features from
the mover's perspective, and joins with `moves_elo_bins` to produce
per-(state_key, band) training samples where every legal move (observed
or not) participates in the softmax denominator.

Design (per advisor guidance):
  * Deduplicate: one 79-float successor feature per unique
    (state_key, notation).  A single state_key is shared across bands.
  * Store the successor feature bank as a numpy memmap on disk so the
    trainer can index into it without paying 10 GB of RAM.
  * Store per-sample records + offsets in a companion .npz.
  * Tag each sample with `in_val_bucket(state_key)` for the split
    contract shared with HumanPrefNet / ValueNet.

Output layout at `--output-dir`:
    succ_feats.f32.bin        # memmap float32 (n_succ_encodings, 79)
    metadata.npz              # everything else + provenance

Usage
-----
    .venv/bin/python tools/extract_human_move_policy_dataset.py \\
        --db data/human_db_candidate.sqlite \\
        --output-dir data/human_move_policy_dataset

    # Smoke run:
    .venv/bin/python tools/extract_human_move_policy_dataset.py \\
        --limit-state-keys 500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.trajectory_db import make_board_state_key, _norm  # noqa: E402  (make_board_state_key kept for verification tests)
from ai.value_net import board_to_features, _INPUT_DIM   # noqa: E402
from game.board import BoardState                        # noqa: E402
from game.rules import get_all_legal_moves               # noqa: E402
from learned_ai.data.elo_binning import option_a_band_from_bin, OPTION_A_NAME  # noqa: E402
from learned_ai.data.human_db_split import (             # noqa: E402
    DEFAULT_VAL_FRACTION,
    MANIFEST_VERSION as SPLIT_VERSION,
    in_val_bucket,
    three_way_split,
)
from tools.train_value_net_v2 import board_from_state_key  # noqa: E402


EXTRACT_VERSION = "2"

# sample_split int8 encoding: 0=train, 1=val, 2=test (v2 three-way split).
_SPLIT_TO_INT8: dict[str, int] = {"train": 0, "val": 1, "test": 2}


# ── Move-notation helper (matches build_human_db_sha.py convention) ─────────

def _move_notation(mv: dict) -> str:
    frm = mv.get("from")
    to  = mv.get("to") or ""
    cap = mv.get("capture") or ""
    base = f"{frm}-{to}" if frm else to
    return f"{base}x{cap}" if cap else base


# ── Provenance ──────────────────────────────────────────────────────────────

def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git_head() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except Exception:
        return None


# ── DB fetch helpers ────────────────────────────────────────────────────────

def _fetch_state_keys(
    conn: sqlite3.Connection, limit: Optional[int]
) -> list[str]:
    """Return the set of state_keys that have ≥1 row in moves_elo_bins."""
    q = "SELECT DISTINCT state_key FROM moves_elo_bins ORDER BY state_key"
    if limit is not None:
        q += f" LIMIT {int(limit)}"
    return [r[0] for r in conn.execute(q)]


def _fetch_band_counts_for_state(
    conn: sqlite3.Connection, state_key: str
) -> dict[tuple[str, str], int]:
    """Return {(band, notation): total_count} for a state_key, applying
    Option A membership on the 50-Elo bin.  Multiple bins that map to
    the same band are summed."""
    rows = conn.execute(
        "SELECT notation, elo_bin, total FROM moves_elo_bins WHERE state_key = ?",
        (state_key,),
    ).fetchall()
    out: dict[tuple[str, str], int] = defaultdict(int)
    for notation, elo_bin, total in rows:
        band = option_a_band_from_bin(int(elo_bin))
        out[(band, notation)] += int(total)
    return dict(out)


# ── Extraction ──────────────────────────────────────────────────────────────

_BAND_TO_IDX = {"lower": 0, "middle": 1, "upper": 2}


def extract(
    db_path: Path,
    output_dir: Path,
    limit_state_keys: Optional[int] = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> dict:
    if not db_path.exists():
        raise FileNotFoundError(f"Candidate DB not found: {db_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA cache_size = -262144")   # 256 MB page cache

    # ── First pass: enumerate state_keys, size the successor bank.
    print(f"[extract] Enumerating state_keys in {db_path}")
    state_keys = _fetch_state_keys(conn, limit_state_keys)
    print(f"[extract] {len(state_keys):,} candidate state_keys")

    # Reconstruct boards, enumerate legal moves, compute successor features
    # in memory (feature bank is memmap'd so we write incrementally).
    # We first size the bank by summing legal-move counts, then allocate.

    # Data streams captured during pass 1:
    #   state_key -> {"legal_notations": [...], "legal_count": int, "mover": str}
    print(f"[extract] Pass 1: enumerating legal moves per state_key …")
    per_state: list[dict] = []
    skipped_bad_state_key = 0
    skipped_no_legal_moves = 0
    total_succ_encodings = 0
    for sk in state_keys:
        board = board_from_state_key(sk)
        if board is None:
            skipped_bad_state_key += 1
            continue
        legal_moves = get_all_legal_moves(board)
        if not legal_moves:
            skipped_no_legal_moves += 1
            continue
        notations = [_move_notation(m) for m in legal_moves]
        per_state.append({
            "state_key":     sk,
            "board":         board,
            "legal_moves":   legal_moves,
            "notations":     notations,
            "mover":         board.turn,
            "succ_offset":   total_succ_encodings,
            "legal_count":   len(legal_moves),
        })
        total_succ_encodings += len(legal_moves)

    print(f"[extract] {len(per_state):,} usable state_keys, "
          f"{total_succ_encodings:,} successor encodings to compute")
    print(f"[extract] Skipped bad-state-key: {skipped_bad_state_key:,}   "
          f"no-legal-moves: {skipped_no_legal_moves:,}")

    # ── Allocate memmap for successor features.
    feat_bin_path = output_dir / "succ_feats.f32.bin"
    feat_dim      = int(_INPUT_DIM)
    succ_feats    = np.memmap(
        feat_bin_path,
        dtype=np.float32,
        mode="w+",
        shape=(total_succ_encodings, feat_dim),
    )

    # ── Pass 2: encode successor features into the memmap; join bands.
    print(f"[extract] Pass 2: encoding {total_succ_encodings:,} successor features + joining band counts …")
    sample_state_idx:   list[int] = []
    sample_band_idx:    list[int] = []
    sample_targets:     list[int] = []      # flat
    sample_offsets:     list[int] = [0]     # per-sample end offset into sample_targets
    sample_split_vals:  list[int] = []      # int8: 0=train 1=val 2=test

    n_samples_by_band = {"lower": 0, "middle": 0, "upper": 0}
    unmatched_notation_events = 0
    unmatched_notation_states = 0

    for state_idx, block in enumerate(per_state):
        legal_moves  = block["legal_moves"]
        board        = block["board"]
        mover        = block["mover"]
        notations    = block["notations"]
        offset       = block["succ_offset"]
        legal_count  = block["legal_count"]

        # Encode successor features
        for j, move in enumerate(legal_moves):
            try:
                succ = board.apply_move(move)
                succ_feats[offset + j] = board_to_features(succ, mover)
            except Exception:
                succ_feats[offset + j] = 0.0

        # Join per-band counts
        state_key = block["state_key"]
        band_counts = _fetch_band_counts_for_state(conn, state_key)
        if not band_counts:
            continue
        notation_to_idx = {nt: i for i, nt in enumerate(notations)}

        # Group per band
        per_band: dict[str, dict[str, int]] = defaultdict(dict)
        state_unmatched_hit = False
        for (band, notation), cnt in band_counts.items():
            if band == "unknown":
                continue
            idx = notation_to_idx.get(notation)
            if idx is None:
                unmatched_notation_events += cnt
                state_unmatched_hit = True
                continue
            per_band[band][notation] = per_band[band].get(notation, 0) + cnt
        if state_unmatched_hit:
            unmatched_notation_states += 1

        split_label = three_way_split(state_key)
        for band, obs in per_band.items():
            targets = np.zeros(legal_count, dtype=np.int32)
            for nt, cnt in obs.items():
                idx = notation_to_idx[nt]
                targets[idx] = cnt
            sample_state_idx.append(state_idx)
            sample_band_idx.append(_BAND_TO_IDX[band])
            sample_targets.extend(int(x) for x in targets)
            sample_offsets.append(sample_offsets[-1] + legal_count)
            sample_split_vals.append(_SPLIT_TO_INT8[split_label])
            n_samples_by_band[band] += 1

    succ_feats.flush()
    del succ_feats

    # ── Assemble metadata + state-key offsets for the successor bank.
    state_key_arr   = np.array([b["state_key"] for b in per_state], dtype=object)
    mover_color_arr = np.array([b["mover"]     for b in per_state], dtype=object)
    state_succ_offsets = np.zeros(len(per_state) + 1, dtype=np.int64)
    for i, b in enumerate(per_state):
        state_succ_offsets[i + 1] = state_succ_offsets[i] + b["legal_count"]

    sample_state_idx_a = np.asarray(sample_state_idx,  dtype=np.int64)
    sample_band_idx_a  = np.asarray(sample_band_idx,   dtype=np.int8)
    sample_targets_a   = np.asarray(sample_targets,    dtype=np.int32)
    sample_offsets_a   = np.asarray(sample_offsets,    dtype=np.int64)
    sample_split_a     = np.asarray(sample_split_vals, dtype=np.int8)
    # Backward-compat alias: "val" in v2 three-way split maps to is_val=True.
    # Old v1 callers (ValueNet, HumanPrefNet) treat buckets 0..19 as val;
    # v2 "test" (buckets 0..4) appears here as is_val=True so old code
    # remains conservative (test positions are never used for training).
    sample_is_val_a    = sample_split_a != 0  # False only for train (0)

    # Count samples per split tier.
    n_train = int((sample_split_a == 0).sum())
    n_val   = int((sample_split_a == 1).sum())
    n_test  = int((sample_split_a == 2).sum())

    provenance = {
        "extract_version":                  EXTRACT_VERSION,
        "candidate_db_path":                str(db_path),
        "candidate_db_sha256":              _sha256_file(db_path),
        "feature_dim":                      int(feat_dim),
        "feature_version":                  "value_net_v2_input_v1",
        "elo_band_config_name":             OPTION_A_NAME,
        "split_manifest_version":           SPLIT_VERSION,
        "val_fraction":                     float(val_fraction),
        "extract_git_commit":               _git_head() or "",
        "successor_bank_bin":               "succ_feats.f32.bin",
        "successor_bank_dtype":             "float32",
        "successor_bank_rows":              int(total_succ_encodings),
        "n_state_keys":                     int(len(per_state)),
        "n_samples":                        int(len(sample_state_idx)),
        "n_samples_by_band":                json.dumps(n_samples_by_band),
        "n_samples_train":                  n_train,
        "n_samples_val":                    n_val,
        "n_samples_test":                   n_test,
        "unmatched_notation_events":        int(unmatched_notation_events),
        "unmatched_notation_states":        int(unmatched_notation_states),
        "skipped_bad_state_key":            int(skipped_bad_state_key),
        "skipped_no_legal_moves":           int(skipped_no_legal_moves),
        "elapsed_seconds":                  round(time.time() - t0, 1),
    }

    metadata_path = output_dir / "metadata.npz"
    np.savez(
        metadata_path,
        state_keys=state_key_arr,
        mover_colors=mover_color_arr,
        state_succ_offsets=state_succ_offsets,
        sample_state_idx=sample_state_idx_a,
        sample_band_idx=sample_band_idx_a,
        sample_targets=sample_targets_a,
        sample_offsets=sample_offsets_a,
        sample_split=sample_split_a,
        sample_is_val=sample_is_val_a,
        provenance=np.array(json.dumps(provenance), dtype=object),
    )
    print(f"[extract] Wrote {metadata_path}  and  {feat_bin_path}")
    print(f"[extract] Samples: {len(sample_state_idx):,}   "
          f"lower={n_samples_by_band['lower']:,}   "
          f"middle={n_samples_by_band['middle']:,}   "
          f"upper={n_samples_by_band['upper']:,}")
    print(f"[extract] Split: train={n_train:,}  val={n_val:,}  test={n_test:,}")
    print(f"[extract] Unmatched notation events: {unmatched_notation_events:,}   "
          f"states with any: {unmatched_notation_states:,}")
    print(f"[extract] Elapsed: {time.time() - t0:.1f} s")
    conn.close()
    return provenance


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db",         type=Path, default=Path("data/human_db_candidate.sqlite"))
    p.add_argument("--output-dir", type=Path, default=Path("data/human_move_policy_dataset"))
    p.add_argument("--limit-state-keys", type=int, default=None,
                   help="Cap the number of state_keys processed (smoke tests).")
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
                   help="Legacy parameter kept for backward compat; v2 uses "
                        "three_way_split (5%% test / 15%% val / 80%% train) "
                        "regardless of this value.")
    args = p.parse_args()

    if not (0.0 < args.val_fraction < 1.0):
        raise SystemExit("--val-fraction must be in (0, 1).")
    prov = extract(args.db, args.output_dir,
                   limit_state_keys=args.limit_state_keys,
                   val_fraction=args.val_fraction)
    (args.output_dir / "provenance.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
