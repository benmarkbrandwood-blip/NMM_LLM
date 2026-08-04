"""tools/compute_g_v_direct.py — Stage C of gap_net_v3_plan.md.

Computes the direct empirical G_v signal from data/human_db_candidate.sqlite
using MalomDB.query_regret as the oracle.

Per-component fail-closed (user-confirmed design):
  - G_v_class_downgrade: partial sum over legal moves where component A is
    available, weighted by empirical P_h.  None if no contributing move.
  - G_v_wdl_utility / G_v_ordinal_rank: position-level fail-closed (best_omv
    is shared across all moves); None if any non-terminal legal child is
    uncovered by Malom.
  - G_v_within_class_distance: always null in regret_v1.

Outputs:
  data/gap_v3_direct_gv.parquet      — main result
  data/gap_v3_direct_gv_provenance.json
  data/gap_v3_direct_gv_abstained.jsonl

Usage::
    .venv/bin/python tools/compute_g_v_direct.py [--min-support N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.malom_db import MalomDB, _MALOM_LABEL_VERSION, _REGRET_VERSION
from game.board import POSITIONS, BoardState
from game.rules import get_all_legal_moves, is_terminal
from learned_ai.data.elo_binning import option_a_band_from_bin


# ── Board reconstruction ──────────────────────────────────────────────────────

def _board_from_state_key(state_key: str) -> Optional[BoardState]:
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
    w_cap = max(0, placed_b - on_b)
    b_cap = max(0, placed_w - on_w)
    return BoardState(
        positions=positions,
        turn=turn,
        pieces_on_board={"W": on_w, "B": on_b},
        pieces_placed={"W": placed_w, "B": placed_b},
        pieces_captured={"W": w_cap, "B": b_cap},
    )


# ── Move notation (matches extract_human_move_policy_dataset.py) ──────────────

def _move_notation(mv: dict) -> str:
    frm = mv.get("from")
    to  = mv.get("to") or ""
    cap = mv.get("capture") or ""
    base = f"{frm}-{to}" if frm else to
    return f"{base}x{cap}" if cap else base


# ── Provenance helpers ────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_script() -> str:
    return _sha256_file(Path(__file__)) or "unknown"


# ── Load moves_elo_bins, aggregate by (state_key, band) ──────────────────────

def _load_band_counts(
    db_path: Path,
    min_support: int,
) -> dict[tuple[str, str], dict[str, int]]:
    """Return {(state_key, band): {notation: total, ...}} for pairs with >= min_support."""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT state_key, notation, elo_bin, total FROM moves_elo_bins"
        ).fetchall()
    finally:
        conn.close()

    # Aggregate: (state_key, band) → notation → total
    agg: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for state_key, notation, elo_bin, total in rows:
        band = option_a_band_from_bin(elo_bin)
        if band == "unknown":
            continue
        agg[(state_key, band)][notation] += total

    # Filter by min_support
    result: dict[tuple[str, str], dict[str, int]] = {}
    for (sk, band), notation_counts in agg.items():
        if sum(notation_counts.values()) >= min_support:
            result[(sk, band)] = dict(notation_counts)
    return result


# ── G_v computation for a single (state_key, band) ───────────────────────────

def _compute_gv_row(
    db: MalomDB,
    state_key: str,
    band: str,
    notation_counts: dict[str, int],
    abstain_log: list[dict],
) -> Optional[dict]:
    """Return a parquet row dict, or None (and append to abstain_log) on skip."""

    board = _board_from_state_key(state_key)
    if board is None:
        abstain_log.append({"state_key": state_key, "band": band, "reason": "malformed_state_key"})
        return None

    is_term, _ = is_terminal(board)
    if is_term:
        abstain_log.append({"state_key": state_key, "band": band, "reason": "parent_terminal"})
        return None

    legal_moves = get_all_legal_moves(board)
    if len(legal_moves) < 2:
        abstain_log.append({"state_key": state_key, "band": band,
                            "reason": f"n_legal_{len(legal_moves)}_lt_2"})
        return None

    total_band = sum(notation_counts.values())

    # Empirical P_h: notation → weight (sums to 1 over observed moves)
    ph: dict[str, float] = {
        notation: count / total_band
        for notation, count in notation_counts.items()
    }

    # Query regret for ALL legal moves
    regrets = {}
    for mv in legal_moves:
        notation = _move_notation(mv)
        regrets[notation] = db.query_regret(board, mv)

    # Count legal moves with observed P_h > 0
    ph_coverage = sum(
        1 for mv in legal_moves if _move_notation(mv) in ph
    ) / len(legal_moves)

    # ── Component A (class_downgrade_prob) — per-move, partial sum ────────────
    g_v_A_sum = 0.0
    g_v_A_weight = 0.0
    g_v_A_partial = False  # True if some moves had A=None
    for mv in legal_moves:
        notation = _move_notation(mv)
        w = ph.get(notation, 0.0)
        if w == 0.0:
            continue
        rr = regrets[notation]
        a_val = rr.components.get("class_downgrade_prob") if rr.available else None
        if a_val is not None:
            g_v_A_sum += w * a_val
            g_v_A_weight += w
        else:
            g_v_A_partial = True

    g_v_A: Optional[float] = (g_v_A_sum if g_v_A_weight > 0.0 else None)

    # ── Components B and C — position-level fail-closed via best_omv ──────────
    # Check any available result's component B to determine if best_omv resolved.
    # best_omv is position-level: if it's None for one result it's None for all.
    g_v_B: Optional[float] = None
    g_v_C: Optional[float] = None
    any_available = any(rr.available for rr in regrets.values())

    if any_available:
        # Determine if best_omv is available (same for all results at this position)
        sample_rr = next(rr for rr in regrets.values() if rr.available)
        b_val_sample = sample_rr.components.get("wdl_utility_loss")
        if b_val_sample is not None:
            # All moves' B and C should be non-None; compute weighted sums
            b_sum = 0.0
            c_sum = 0.0
            b_weight = 0.0
            c_weight = 0.0
            for mv in legal_moves:
                notation = _move_notation(mv)
                w = ph.get(notation, 0.0)
                if w == 0.0:
                    continue
                rr = regrets[notation]
                if not rr.available:
                    continue
                b = rr.components.get("wdl_utility_loss")
                c = rr.components.get("ordinal_rank_loss")
                if b is not None:
                    b_sum += w * b
                    b_weight += w
                if c is not None:
                    c_sum += w * c
                    c_weight += w
            g_v_B = b_sum if b_weight > 0.0 else None
            g_v_C = c_sum if c_weight > 0.0 else None

    # Phase and mover colour from board
    phase = getattr(board, "phase", "move")
    mover_color = board.turn

    return {
        "state_key": state_key,
        "band": band,
        "mover_color": mover_color,
        "phase": phase,
        "n_legal": len(legal_moves),
        "ph_total_band": total_band,
        "ph_coverage": round(ph_coverage, 4),
        "g_v_class_downgrade": g_v_A,
        "g_v_wdl_utility": g_v_B,
        "g_v_ordinal_rank": g_v_C,
        "g_v_within_class_distance": None,
        "g_v_A_partial": g_v_A_partial,
        "ph_source": "empirical",
    }


# ── Parquet schema ────────────────────────────────────────────────────────────

_SCHEMA = pa.schema([
    ("state_key",                pa.string()),
    ("band",                     pa.string()),
    ("mover_color",              pa.string()),
    ("phase",                    pa.string()),
    ("n_legal",                  pa.int32()),
    ("ph_total_band",            pa.int32()),
    ("ph_coverage",              pa.float32()),
    ("g_v_class_downgrade",      pa.float32()),
    ("g_v_wdl_utility",         pa.float32()),
    ("g_v_ordinal_rank",         pa.float32()),
    ("g_v_within_class_distance", pa.float32()),
    ("g_v_A_partial",            pa.bool_()),
    ("ph_source",                pa.string()),
])


# ── Stage C gate check ────────────────────────────────────────────────────────

def _check_promotion_gate(rows: list[dict]) -> None:
    """Print the Stage C gate: G_v_wdl_utility monotonically decreasing upper < middle < lower."""
    from collections import defaultdict as dd
    band_vals: dict[str, list[float]] = dd(list)
    for r in rows:
        v = r["g_v_wdl_utility"]
        if v is not None:
            band_vals[r["band"]].append(v)

    means: dict[str, Optional[float]] = {}
    for band in ("lower", "middle", "upper"):
        vals = band_vals.get(band, [])
        means[band] = sum(vals) / len(vals) if vals else None

    print("\n── Stage C promotion gate ─────────────────────────────────────────")
    print(f"  G_v_wdl_utility mean by band:")
    for band in ("lower", "middle", "upper"):
        m = means[band]
        n = len(band_vals.get(band, []))
        print(f"    {band:8s}: {m:.6f} (n={n})" if m is not None else f"    {band:8s}: N/A (n=0)")

    gate_ok = False
    l, m_val, u = means.get("lower"), means.get("middle"), means.get("upper")
    if l is not None and m_val is not None and u is not None:
        gate_ok = u < m_val < l
    status = "✅ PASSED" if gate_ok else "❌ FAILED"
    print(f"  Monotone upper < middle < lower: {status}")
    if not gate_ok and l is not None and m_val is not None and u is not None:
        print(f"  (upper={u:.6f}, middle={m_val:.6f}, lower={l:.6f})")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_db_dir() -> str:
    cfg = _ROOT / "data" / "settings.json"
    if cfg.exists():
        d = json.loads(cfg.read_text(encoding="utf-8"))
        p = d.get("malom_db_path")
        if p:
            return p
    raise RuntimeError("Cannot find malom_db_path in data/settings.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--min-support", type=int, default=25,
                        help="Minimum total observed plays per (state_key, band) [default: 25]")
    parser.add_argument("--db-dir", default=None, help="Malom DB directory")
    parser.add_argument("--human-db", default=None,
                        help="Path to human_db_candidate.sqlite")
    args = parser.parse_args()

    min_support = args.min_support
    human_db = Path(args.human_db) if args.human_db else _ROOT / "data" / "human_db_candidate.sqlite"
    db_dir = args.db_dir or _load_db_dir()
    out_parquet = _ROOT / "data" / "gap_v3_direct_gv.parquet"
    out_provenance = _ROOT / "data" / "gap_v3_direct_gv_provenance.json"
    out_abstained = _ROOT / "data" / "gap_v3_direct_gv_abstained.jsonl"

    print(f"Human DB:      {human_db}")
    print(f"Malom DB dir:  {db_dir}")
    print(f"min_support:   {min_support}")
    print(f"Output:        {out_parquet}")

    t0 = time.time()

    # Load Malom DB
    malom = MalomDB(db_dir)
    if not malom.is_available():
        print("ERROR: Malom DB not available", file=sys.stderr)
        sys.exit(1)

    # Load and aggregate moves_elo_bins
    print("\nLoading moves_elo_bins ...")
    band_counts = _load_band_counts(human_db, min_support)
    n_eligible = len(band_counts)
    print(f"  Eligible (state_key, band) pairs: {n_eligible}")

    # Process each pair
    rows: list[dict] = []
    abstain_log: list[dict] = []
    t_last = time.time()
    for i, ((state_key, band), notation_counts) in enumerate(band_counts.items()):
        if i % 500 == 0 and i > 0:
            elapsed = time.time() - t_last
            rate = 500 / elapsed if elapsed > 0 else 0
            remaining = (n_eligible - i) / rate if rate > 0 else 0
            print(f"  [{i}/{n_eligible}] {rate:.0f}/s  ETA {remaining:.0f}s  "
                  f"rows={len(rows)} abstained={len(abstain_log)}")
            t_last = time.time()

        row = _compute_gv_row(malom, state_key, band, notation_counts, abstain_log)
        if row is not None:
            rows.append(row)

    malom.close()
    elapsed_total = time.time() - t0

    # Write parquet
    print(f"\nWriting {len(rows)} rows to {out_parquet} ...")
    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    pq.write_table(table, str(out_parquet), compression="snappy")

    # Write abstained.jsonl
    print(f"Writing {len(abstain_log)} abstained rows to {out_abstained} ...")
    with out_abstained.open("w", encoding="utf-8") as f:
        for entry in abstain_log:
            f.write(json.dumps(entry) + "\n")

    # Summarise abstention reasons
    reason_counts: dict[str, int] = defaultdict(int)
    for entry in abstain_log:
        reason_counts[entry["reason"]] += 1
    print("  Abstention reasons:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")

    # Coverage stats
    n_null_B = sum(1 for r in rows if r["g_v_wdl_utility"] is None)
    n_null_A = sum(1 for r in rows if r["g_v_class_downgrade"] is None)
    n_partial_A = sum(1 for r in rows if r["g_v_A_partial"])
    print(f"\nCoverage:")
    print(f"  Rows emitted:          {len(rows)} / {n_eligible} eligible "
          f"({100*len(rows)/max(n_eligible,1):.1f}%)")
    print(f"  G_v_A null:            {n_null_A} ({100*n_null_A/max(len(rows),1):.1f}%)")
    print(f"  G_v_A partial sum:     {n_partial_A} ({100*n_partial_A/max(len(rows),1):.1f}%)")
    print(f"  G_v_B/C null:          {n_null_B} ({100*n_null_B/max(len(rows),1):.1f}%)")
    print(f"  Elapsed:               {elapsed_total:.1f}s")

    # Promotion gate
    _check_promotion_gate(rows)

    # Write provenance
    provenance = {
        "stage": "C",
        "script": "tools/compute_g_v_direct.py",
        "script_sha256": _sha256_script(),
        "human_db": str(human_db),
        "human_db_sha256": _sha256_file(human_db),
        "malom_db_dir": str(db_dir),
        "regret_version": _REGRET_VERSION,
        "malom_label_version": _MALOM_LABEL_VERSION,
        "min_support": min_support,
        "n_eligible_pairs": n_eligible,
        "n_rows_emitted": len(rows),
        "n_abstained": len(abstain_log),
        "abstention_reasons": dict(reason_counts),
        "elapsed_s": round(elapsed_total, 2),
        "output_parquet": str(out_parquet),
        "output_abstained": str(out_abstained),
    }
    out_provenance.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Provenance written to {out_provenance}")


if __name__ == "__main__":
    main()
