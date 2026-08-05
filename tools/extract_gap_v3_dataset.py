"""tools/extract_gap_v3_dataset.py — Stage D of gap_net_v3_plan.md.

Extracts the GapNet v3 training dataset from data/human_db_candidate.sqlite,
combining Malom objective regret (R_v) with model-based human move policy (P_h)
to produce G_v targets for all four components.

Strict fail-closed (§5.5 / §8.3):
  Any R_v unavailable for any legal move, or any component A/B/C is None for any
  legal move → abstain the entire (state_key, band).
  Component D excluded from strict fail-closed per plan §1 line 11 (regret_v1
  defers D; it is always None and targets are stored as NaN).

P_h source:
  Primary: HumanMovePolicyAdvisor.probs(board, legal_moves, elo_band).
  Secondary (also stored): empirical P_h from moves_elo_bins where
    sum_band_total >= min_empirical_support.
  ph_source per row: "model" (only model available) or "hybrid" (both available).

Splits: deterministic hash on state_key → 80 % train / 10 % val / 10 % test.

Output files (data/gap_net_v3_dataset/):
  parent_feats.f32.bin          — (N, 79) float32, row-major binary
  targets.f32.bin               — (N, 4) float32 [g_v_A, g_v_B, g_v_C, NaN]
  targets_empirical.f32.bin     — (N, 4) float32 (empirical G_v, NaN where absent)
  metadata.npz                  — state_keys, band_idx, split, phase, mover_color,
                                   n_legal, ph_source, provenance JSON
  provenance.json               — full provenance record
  abstained.jsonl               — one JSON line per abstained (state_key, band)
  checkpoint.json               — resume checkpoint (last_state_key, n_emitted, …)
  metadata_checkpoint.npz       — accumulated metadata arrays for resume

Usage::
    .venv/bin/python tools/extract_gap_v3_dataset.py [options]
    .venv/bin/python tools/extract_gap_v3_dataset.py --limit 500  # smoke test
    .venv/bin/python tools/extract_gap_v3_dataset.py --resume      # continue interrupted run
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

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.human_move_policy_advisor import HumanMovePolicyAdvisor
from ai.malom_db import MalomDB, RegretResult, _MALOM_LABEL_VERSION, _REGRET_VERSION
from ai.value_net import board_to_features, _INPUT_DIM
from game.board import POSITIONS, BoardState
from game.rules import get_all_legal_moves, is_terminal
from learned_ai.data.elo_binning import option_a_band_from_bin

import numpy as np

_BAND_TO_IDX = {"lower": 0, "middle": 1, "upper": 2}
_SPLIT_SEED = b"gap_v3_split_v1"
_FEATURE_VERSION = "value_net_v2_input_v1"
_DATASET_VERSION = "1"
_FLUSH_EVERY = 50_000  # Flush binary accumulators and write checkpoint every N emitted rows


# ── Board helpers ─────────────────────────────────────────────────────────────

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


def _move_notation(mv: dict) -> str:
    frm = mv.get("from")
    to  = mv.get("to") or ""
    cap = mv.get("capture") or ""
    base = f"{frm}-{to}" if frm else to
    return f"{base}x{cap}" if cap else base


def _split_from_state_key(state_key: str) -> str:
    h = int.from_bytes(
        hashlib.md5(_SPLIT_SEED + state_key.encode()).digest(), "little"
    ) % 10
    return "train" if h < 8 else ("val" if h == 8 else "test")


def _split_idx(split: str) -> int:
    return {"train": 0, "val": 1, "test": 2}[split]


# ── Provenance helpers ────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ── DB loading (streaming) ────────────────────────────────────────────────────

def _count_distinct_state_keys(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT COUNT(DISTINCT state_key) FROM moves_elo_bins"
        ).fetchone()[0]
    finally:
        conn.close()


def _iter_state_key_groups(
    db_path: Path,
    min_support: int = 1,
):
    """Yield (state_key, {band: {notation: total}}) in state_key sorted order.

    Streams rows from SQLite to avoid loading 3M+ rows into memory at once.
    Only one state_key's data is held in memory at a time.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA read_uncommitted = true")  # faster reads on pre-existing data
    cur = conn.execute(
        "SELECT state_key, notation, elo_bin, total "
        "FROM moves_elo_bins ORDER BY state_key"
    )

    current_sk: Optional[str] = None
    current_bands: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for state_key, notation, elo_bin, total in cur:
        band = option_a_band_from_bin(elo_bin)
        if band == "unknown":
            continue

        if state_key != current_sk:
            if current_sk is not None:
                filtered = {
                    b: dict(nc)
                    for b, nc in current_bands.items()
                    if sum(nc.values()) >= min_support
                }
                if filtered:
                    yield current_sk, filtered
            current_sk = state_key
            current_bands = defaultdict(lambda: defaultdict(int))

        current_bands[band][notation] += total

    # Yield the final group
    if current_sk is not None:
        filtered = {
            b: dict(nc)
            for b, nc in current_bands.items()
            if sum(nc.values()) >= min_support
        }
        if filtered:
            yield current_sk, filtered

    conn.close()


# ── Fail-closed helpers ───────────────────────────────────────────────────────

def _check_malom_eligible(
    regrets: dict[str, RegretResult],
    legal_moves: list[dict],
) -> tuple[bool, str]:
    """Full Malom coverage: all moves available=True AND best_omv not None.

    'full coverage' per §8.3 step 1: parent covered (caller already checked)
    AND every legal successor covered (best_omv resolves iff all non-terminal
    children are Malom-covered).
    """
    for mv in legal_moves:
        rr = regrets[_move_notation(mv)]
        if not rr.available:
            return False, f"move_unavailable:{rr.unavailable_reason}"
        if rr.best_omv is None:
            return False, "best_omv_unavailable"
    return True, ""


def _check_emittable(
    regrets: dict[str, RegretResult],
    legal_moves: list[dict],
) -> tuple[bool, str]:
    """Strict fail-closed §5.5: all A/B/C components must be non-None for all moves.

    Component D excluded per plan §1 line 11 (regret_v1 defers D; it is always None).
    """
    for mv in legal_moves:
        rr = regrets[_move_notation(mv)]
        if not rr.available:
            return False, f"move_unavailable:{rr.unavailable_reason}"
        if rr.components["class_downgrade_prob"] is None:
            return False, "comp_a_none"
        if rr.best_omv is None:
            return False, "comp_bc_none"
    return True, ""


# ── G_v computation ───────────────────────────────────────────────────────────

def _compute_gv(
    ph: np.ndarray,
    legal_moves: list[dict],
    regrets: dict[str, RegretResult],
) -> tuple[float, float, float]:
    """Weighted sum of R_v components under given P_h probabilities.

    Assumes caller has verified emittability (all components non-None).
    Returns (g_v_A, g_v_B, g_v_C).
    """
    g_v_A = g_v_B = g_v_C = 0.0
    for i, mv in enumerate(legal_moves):
        w = float(ph[i])
        rr = regrets[_move_notation(mv)]
        g_v_A += w * rr.components["class_downgrade_prob"]
        g_v_B += w * rr.components["wdl_utility_loss"]
        g_v_C += w * rr.components["ordinal_rank_loss"]
    return g_v_A, g_v_B, g_v_C


def _compute_empirical_ph(
    notation_counts: dict[str, int],
    legal_moves: list[dict],
    min_empirical_support: int,
) -> Optional[np.ndarray]:
    """Empirical P_h from observed counts. Returns None if support < threshold."""
    total = sum(notation_counts.values())
    if total < min_empirical_support:
        return None
    ph = np.zeros(len(legal_moves), dtype=np.float64)
    for i, mv in enumerate(legal_moves):
        ph[i] = notation_counts.get(_move_notation(mv), 0) / total
    return ph.astype(np.float32)


# ── Checkpoint / resume helpers ───────────────────────────────────────────────

def _verify_file_sizes(out_dir: Path, n_emitted: int) -> None:
    """Raise if existing output files don't match n_emitted rows from checkpoint."""
    for fname, row_bytes in [
        ("parent_feats.f32.bin", 79 * 4),
        ("targets.f32.bin", 4 * 4),
        ("targets_empirical.f32.bin", 4 * 4),
    ]:
        p = out_dir / fname
        if not p.exists():
            raise RuntimeError(f"Resume: missing output file {fname}")
        actual = p.stat().st_size
        expected = n_emitted * row_bytes
        if actual != expected:
            raise RuntimeError(
                f"Resume: {fname} has {actual} bytes but checkpoint says "
                f"n_emitted={n_emitted} (expected {expected} bytes)"
            )


def _load_resume_state(out_dir: Path) -> Optional[dict]:
    """Load checkpoint state for resume. Returns None if no checkpoint found."""
    cp = out_dir / "checkpoint.json"
    meta_cp = out_dir / "metadata_checkpoint.npz"
    if not cp.exists() or not meta_cp.exists():
        return None
    state = json.loads(cp.read_text(encoding="utf-8"))
    meta = np.load(str(meta_cp), allow_pickle=True)
    state["meta_state_keys"] = list(meta["state_keys"])
    state["meta_band_idx"] = [int(x) for x in meta["band_idx"]]
    state["meta_split"] = [int(x) for x in meta["split"]]
    state["meta_phase"] = [str(x) for x in meta["phase"]]
    state["meta_mover_color"] = [str(x) for x in meta["mover_color"]]
    state["meta_n_legal"] = [int(x) for x in meta["n_legal"]]
    state["meta_ph_source"] = [str(x) for x in meta["ph_source"]]
    return state


def _flush_binary_batch(
    out_dir: Path,
    feats: list[np.ndarray],
    tgts: list[np.ndarray],
    tgts_emp: list[np.ndarray],
) -> None:
    """Append batch binary data to output files."""
    if not feats:
        return
    with open(out_dir / "parent_feats.f32.bin", "ab") as f:
        np.stack(feats).astype(np.float32).tofile(f)
    with open(out_dir / "targets.f32.bin", "ab") as f:
        np.stack(tgts).astype(np.float32).tofile(f)
    with open(out_dir / "targets_empirical.f32.bin", "ab") as f:
        np.stack(tgts_emp).astype(np.float32).tofile(f)


def _flush_abstained_batch(out_dir: Path, lines: list[str]) -> None:
    """Append abstained JSON lines."""
    if not lines:
        return
    with open(out_dir / "abstained.jsonl", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _save_checkpoint(
    out_dir: Path,
    last_state_key: str,
    n_emitted: int,
    meta_state_keys: list,
    meta_band_idx: list,
    meta_split: list,
    meta_phase: list,
    meta_mover_color: list,
    meta_n_legal: list,
    meta_ph_source: list,
    counters: dict,
) -> None:
    """Overwrite metadata checkpoint .npz, then write checkpoint.json (written last)."""
    np.savez(
        str(out_dir / "metadata_checkpoint.npz"),
        state_keys=np.array(meta_state_keys, dtype=object),
        band_idx=np.array(meta_band_idx, dtype=np.int8),
        split=np.array(meta_split, dtype=np.int8),
        phase=np.array(meta_phase, dtype=object),
        mover_color=np.array(meta_mover_color, dtype=object),
        n_legal=np.array(meta_n_legal, dtype=np.int16),
        ph_source=np.array(meta_ph_source, dtype=object),
    )
    cp = {"last_state_key": last_state_key, "n_emitted": n_emitted, **counters}
    (out_dir / "checkpoint.json").write_text(json.dumps(cp), encoding="utf-8")


# ── Main extraction ───────────────────────────────────────────────────────────

def run_extraction(
    human_db: Path,
    malom_db_dir: str,
    model_path: Path,
    out_dir: Path,
    min_support: int = 1,
    min_empirical_support: int = 25,
    temperature: float = 0.7674,
    limit: Optional[int] = None,
    resume: bool = False,
) -> dict:
    """Run the full extraction pipeline. Returns a coverage summary dict."""
    import datetime

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load advisor and Malom DB
    advisor = HumanMovePolicyAdvisor(model_path, temperature=temperature)
    malom = MalomDB(malom_db_dir)
    if not malom.is_available():
        raise RuntimeError(f"Malom DB not available at {malom_db_dir}")

    # Pre-count distinct state_keys for progress reporting (cheap query)
    print(f"Counting state_keys (min_support={min_support})...")
    t0 = time.time()
    n_total_state_keys = _count_distinct_state_keys(human_db)
    print(f"  {n_total_state_keys:,} distinct state_keys ({time.time()-t0:.1f}s)")

    # === Resume state ===
    skip_until_after: Optional[str] = None  # Skip state_keys <= this value

    if resume:
        prior = _load_resume_state(out_dir)
        if prior is None:
            print("No checkpoint found — starting fresh.")
            resume = False
        else:
            _verify_file_sizes(out_dir, prior["n_emitted"])
            skip_until_after = prior["last_state_key"]
            n_emitted              = prior["n_emitted"]
            n_processed_state_keys = prior.get("n_processed_state_keys", 0)
            n_processed_pairs      = prior.get("n_processed_pairs", 0)
            n_bad_board            = prior.get("n_bad_board", 0)
            n_terminal             = prior.get("n_terminal", 0)
            n_n_legal_lt2          = prior.get("n_n_legal_lt2", 0)
            n_parent_no_malom      = prior.get("n_parent_no_malom", 0)
            n_not_malom_eligible   = prior.get("n_not_malom_eligible", 0)
            n_malom_eligible       = prior.get("n_malom_eligible", 0)
            n_not_emittable        = prior.get("n_not_emittable", 0)
            meta_state_keys        = prior["meta_state_keys"]
            meta_band_idx          = prior["meta_band_idx"]
            meta_split             = prior["meta_split"]
            meta_phase             = prior["meta_phase"]
            meta_mover_color       = prior["meta_mover_color"]
            meta_n_legal           = prior["meta_n_legal"]
            meta_ph_source         = prior["meta_ph_source"]
            print(
                f"Resuming: n_emitted={n_emitted:,}, "
                f"n_processed_state_keys={n_processed_state_keys:,}, "
                f"last_state_key={skip_until_after!r}"
            )

    if not resume:
        # Fresh start: zero counters and create empty output files
        n_emitted = n_processed_state_keys = n_processed_pairs = 0
        n_bad_board = n_terminal = n_n_legal_lt2 = 0
        n_parent_no_malom = n_not_malom_eligible = n_malom_eligible = n_not_emittable = 0
        meta_state_keys: list[str] = []
        meta_band_idx:   list[int] = []
        meta_split:      list[int] = []
        meta_phase:      list[str] = []
        meta_mover_color:list[str] = []
        meta_n_legal:    list[int] = []
        meta_ph_source:  list[str] = []
        for fname in ["parent_feats.f32.bin", "targets.f32.bin", "targets_empirical.f32.bin"]:
            (out_dir / fname).write_bytes(b"")
        (out_dir / "abstained.jsonl").write_text("", encoding="utf-8")

    print("Streaming moves_elo_bins (sorted by state_key — constant memory)...")

    # Batch accumulators (cleared after each flush)
    batch_feats:     list[np.ndarray] = []
    batch_tgts:      list[np.ndarray] = []
    batch_tgts_emp:  list[np.ndarray] = []
    batch_abstained: list[str] = []
    last_flush_emitted = n_emitted
    last_state_key_seen = ""

    _NAN4 = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)

    t_start = time.time()
    sk_gen = _iter_state_key_groups(human_db, min_support=min_support)
    if limit is not None:
        from itertools import islice
        sk_gen = islice(sk_gen, limit)

    sk_idx_this_run = 0

    for state_key, bands_for_sk in sk_gen:
        # Skip already-processed state_keys on resume
        if skip_until_after is not None:
            if state_key <= skip_until_after:
                continue
            skip_until_after = None  # Done skipping

        sk_idx_this_run += 1
        last_state_key_seen = state_key
        n_processed_state_keys += 1

        if n_processed_state_keys % 10_000 == 0:
            elapsed = time.time() - t_start
            rate = max(sk_idx_this_run / elapsed, 1e-6)
            eta = (n_total_state_keys - n_processed_state_keys) / rate
            print(
                f"  {n_processed_state_keys:,}/{n_total_state_keys:,} state_keys "
                f"| emitted={n_emitted:,} | elapsed={elapsed:.0f}s | eta={eta:.0f}s",
                flush=True,
            )

        n_bands_here = len(bands_for_sk)
        n_processed_pairs += n_bands_here

        board = _board_from_state_key(state_key)
        if board is None:
            for band in bands_for_sk:
                batch_abstained.append(json.dumps({
                    "state_key": state_key, "band": band,
                    "reason": "malformed_state_key",
                }))
            n_bad_board += n_bands_here
            continue

        is_term, _ = is_terminal(board)
        if is_term:
            for band in bands_for_sk:
                batch_abstained.append(json.dumps({
                    "state_key": state_key, "band": band,
                    "reason": "parent_terminal",
                }))
            n_terminal += n_bands_here
            continue

        legal_moves = get_all_legal_moves(board)
        if len(legal_moves) < 2:
            for band in bands_for_sk:
                batch_abstained.append(json.dumps({
                    "state_key": state_key, "band": band,
                    "reason": f"n_legal_{len(legal_moves)}_lt_2",
                }))
            n_n_legal_lt2 += n_bands_here
            continue

        # Quick parent-coverage check before querying all moves
        if malom.query_value(board) is None:
            for band in bands_for_sk:
                batch_abstained.append(json.dumps({
                    "state_key": state_key, "band": band,
                    "reason": "parent_value_unavailable",
                }))
            n_parent_no_malom += n_bands_here
            continue

        # Query R_v for all legal moves (shared across bands)
        regrets: dict[str, RegretResult] = {}
        for mv in legal_moves:
            regrets[_move_notation(mv)] = malom.query_regret(board, mv)

        # Malom eligibility: parent + all children covered
        malom_ok, malom_reason = _check_malom_eligible(regrets, legal_moves)
        if not malom_ok:
            for band in bands_for_sk:
                batch_abstained.append(json.dumps({
                    "state_key": state_key, "band": band,
                    "reason": malom_reason,
                }))
            n_not_malom_eligible += n_bands_here
            continue

        n_malom_eligible += n_bands_here

        # Strict fail-closed: A/B/C must be non-None for all moves
        emit_ok, emit_reason = _check_emittable(regrets, legal_moves)
        if not emit_ok:
            for band in bands_for_sk:
                batch_abstained.append(json.dumps({
                    "state_key": state_key, "band": band,
                    "reason": emit_reason,
                }))
            n_not_emittable += n_bands_here
            continue

        # Board features (parent, from original mover POV)
        parent_feats = np.asarray(board_to_features(board, board.turn), dtype=np.float32)
        phase = getattr(board, "phase", "move")
        split = _split_from_state_key(state_key)

        for band, notation_counts in bands_for_sk.items():
            # Model P_h (always available)
            ph_model = advisor.probs(board, legal_moves, band)
            g_v_A_m, g_v_B_m, g_v_C_m = _compute_gv(ph_model, legal_moves, regrets)

            # Empirical P_h (optional)
            ph_emp = _compute_empirical_ph(notation_counts, legal_moves, min_empirical_support)
            if ph_emp is not None:
                g_v_A_e, g_v_B_e, g_v_C_e = _compute_gv(ph_emp, legal_moves, regrets)
                tgt_emp = np.array([g_v_A_e, g_v_B_e, g_v_C_e, np.nan], dtype=np.float32)
                ph_source = "hybrid"
            else:
                tgt_emp = _NAN4.copy()
                ph_source = "model"

            # Batch accumulators (binary data — flushed and cleared periodically)
            batch_feats.append(parent_feats)
            batch_tgts.append(np.array([g_v_A_m, g_v_B_m, g_v_C_m, np.nan], dtype=np.float32))
            batch_tgts_emp.append(tgt_emp)

            # All-metadata accumulators (kept in memory for final npz write)
            meta_state_keys.append(state_key)
            meta_band_idx.append(_BAND_TO_IDX[band])
            meta_split.append(_split_idx(split))
            meta_phase.append(phase)
            meta_mover_color.append(board.turn)
            meta_n_legal.append(len(legal_moves))
            meta_ph_source.append(ph_source)
            n_emitted += 1

        # Flush after completing a state_key if batch is large enough
        if (n_emitted - last_flush_emitted) >= _FLUSH_EVERY:
            _flush_binary_batch(out_dir, batch_feats, batch_tgts, batch_tgts_emp)
            _flush_abstained_batch(out_dir, batch_abstained)
            counters = dict(
                n_processed_state_keys=n_processed_state_keys,
                n_processed_pairs=n_processed_pairs,
                n_bad_board=n_bad_board,
                n_terminal=n_terminal,
                n_n_legal_lt2=n_n_legal_lt2,
                n_parent_no_malom=n_parent_no_malom,
                n_not_malom_eligible=n_not_malom_eligible,
                n_malom_eligible=n_malom_eligible,
                n_not_emittable=n_not_emittable,
            )
            _save_checkpoint(
                out_dir, state_key, n_emitted,
                meta_state_keys, meta_band_idx, meta_split,
                meta_phase, meta_mover_color, meta_n_legal, meta_ph_source,
                counters,
            )
            batch_feats = batch_tgts = batch_tgts_emp = batch_abstained = []
            last_flush_emitted = n_emitted
            print(f"  [checkpoint] n_emitted={n_emitted:,} last_state_key={state_key!r}",
                  flush=True)

    elapsed = time.time() - t_start
    print(f"\nExtraction complete: {n_emitted:,} rows emitted in {elapsed:.1f}s")

    # Final flush of remaining batch
    _flush_binary_batch(out_dir, batch_feats, batch_tgts, batch_tgts_emp)
    _flush_abstained_batch(out_dir, batch_abstained)

    # Write final metadata.npz
    print(f"Writing output files to {out_dir} ...")
    n = n_emitted
    meta_path = out_dir / "metadata.npz"
    provenance_for_meta = {
        "dataset_version": _DATASET_VERSION,
        "feature_version": _FEATURE_VERSION,
        "regret_version": _REGRET_VERSION,
        "malom_label_version": _MALOM_LABEL_VERSION,
        "n_emitted": n_emitted,
        "split_seed": _SPLIT_SEED.decode(),
    }
    np.savez(
        str(meta_path),
        state_keys=np.array(meta_state_keys, dtype=object),
        band_idx=np.array(meta_band_idx, dtype=np.int8),
        split=np.array(meta_split, dtype=np.int8),
        phase=np.array(meta_phase, dtype=object),
        mover_color=np.array(meta_mover_color, dtype=object),
        n_legal=np.array(meta_n_legal, dtype=np.int16),
        ph_source=np.array(meta_ph_source, dtype=object),
        provenance=np.array([json.dumps(provenance_for_meta)], dtype=object),
    )

    # Coverage summary
    if n_malom_eligible > 0:
        coverage_of_malom_eligible = n_emitted / n_malom_eligible
    else:
        coverage_of_malom_eligible = 0.0
    if n_processed_pairs > 0:
        coverage_of_total = n_emitted / n_processed_pairs
    else:
        coverage_of_total = 0.0

    n_by_band = {"lower": 0, "middle": 0, "upper": 0}
    n_by_split = {"train": 0, "val": 0, "test": 0}
    split_names = ["train", "val", "test"]
    band_names = ["lower", "middle", "upper"]
    for bi, si in zip(meta_band_idx, meta_split):
        n_by_band[band_names[bi]] += 1
        n_by_split[split_names[si]] += 1

    summary = {
        "dataset_version": _DATASET_VERSION,
        "n_total_state_keys_in_db": n_total_state_keys,
        "n_processed_state_keys": n_processed_state_keys,
        "n_processed_pairs": n_processed_pairs,
        "n_bad_board": n_bad_board,
        "n_terminal": n_terminal,
        "n_n_legal_lt2": n_n_legal_lt2,
        "n_parent_no_malom": n_parent_no_malom,
        "n_not_malom_eligible": n_not_malom_eligible,
        "n_malom_eligible": n_malom_eligible,
        "n_not_emittable": n_not_emittable,
        "n_emitted": n_emitted,
        "n_abstained": n_processed_pairs - n_emitted,
        "coverage_of_malom_eligible": round(coverage_of_malom_eligible, 6),
        "coverage_of_total_pairs": round(coverage_of_total, 6),
        "n_by_band": n_by_band,
        "n_by_split": n_by_split,
        "n_hybrid_rows": sum(1 for s in meta_ph_source if s == "hybrid"),
        "n_model_only_rows": sum(1 for s in meta_ph_source if s == "model"),
        "elapsed_s": round(elapsed, 1),
    }

    # Full provenance JSON
    prov = {
        "stage": "D",
        "script": "tools/extract_gap_v3_dataset.py",
        "dataset_version": _DATASET_VERSION,
        "script_sha256": _sha256_file(Path(__file__)) or "unknown",
        "human_db_path": str(human_db),
        "human_db_sha256": _sha256_file(human_db) or "unknown",
        "malom_db_dir": str(malom_db_dir),
        "model_path": str(model_path),
        "model_sha256": _sha256_file(model_path) or "unknown",
        "feature_version": _FEATURE_VERSION,
        "feature_dim": _INPUT_DIM,
        "regret_version": _REGRET_VERSION,
        "malom_label_version": _MALOM_LABEL_VERSION,
        "temperature": temperature,
        "min_support": min_support,
        "min_empirical_support": min_empirical_support,
        "split_seed": _SPLIT_SEED.decode(),
        "ph_source_scheme": (
            "model_always_available; hybrid iff empirical_support >= min_empirical_support"
        ),
        "git_commit": _git_commit(),
        "built_at": datetime.datetime.utcnow().isoformat() + "Z",
        "output_files": {
            "parent_feats": "parent_feats.f32.bin  — (N, 79) float32 row-major",
            "targets": "targets.f32.bin  — (N, 4) float32 [g_v_A, g_v_B, g_v_C, NaN]",
            "targets_empirical": "targets_empirical.f32.bin  — (N, 4) float32 (empirical G_v, NaN where absent)",
        },
        **summary,
    }

    prov_path = out_dir / "provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")

    # Final checkpoint (marks completion)
    counters = dict(
        n_processed_state_keys=n_processed_state_keys,
        n_processed_pairs=n_processed_pairs,
        n_bad_board=n_bad_board,
        n_terminal=n_terminal,
        n_n_legal_lt2=n_n_legal_lt2,
        n_parent_no_malom=n_parent_no_malom,
        n_not_malom_eligible=n_not_malom_eligible,
        n_malom_eligible=n_malom_eligible,
        n_not_emittable=n_not_emittable,
    )
    _save_checkpoint(
        out_dir, last_state_key_seen or "", n_emitted,
        meta_state_keys, meta_band_idx, meta_split,
        meta_phase, meta_mover_color, meta_n_legal, meta_ph_source,
        counters,
    )

    return summary


# ── Stage D promotion gate ────────────────────────────────────────────────────

def _print_gate(summary: dict) -> None:
    print("\n── Stage D promotion gate ──────────────────────────────────────────")
    cov = summary["coverage_of_malom_eligible"]
    threshold = 0.60
    status = "✅ PASSED" if cov >= threshold else "❌ FAILED"
    print(f"  Coverage of Malom-eligible pairs: {cov:.4f} ({cov*100:.2f}%)")
    print(f"  Threshold: {threshold*100:.0f}%   {status}")
    print(f"  n_emitted:        {summary['n_emitted']:,}")
    print(f"  n_malom_eligible: {summary['n_malom_eligible']:,}")
    print(f"  coverage_of_total_pairs: {summary['coverage_of_total_pairs']*100:.4f}%")
    print(f"  n_by_band:  {summary['n_by_band']}")
    print(f"  n_by_split: {summary['n_by_split']}")
    print(f"  n_hybrid:   {summary['n_hybrid_rows']:,}  n_model_only: {summary['n_model_only_rows']:,}")
    print()


# ── Settings helpers ──────────────────────────────────────────────────────────

def _load_malom_db_dir() -> str:
    cfg = _ROOT / "data" / "settings.json"
    if cfg.exists():
        d = json.loads(cfg.read_text(encoding="utf-8"))
        p = d.get("malom_db_path")
        if p:
            return p
    raise RuntimeError("Cannot find malom_db_path in data/settings.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--human-db", default=None,
                        help="Path to human_db_candidate.sqlite")
    parser.add_argument("--malom-db-dir", default=None, help="Malom DB directory")
    parser.add_argument("--model", default=None,
                        help="Path to human_move_policy_net_v2_candidate.npz")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory [default: data/gap_net_v3_dataset/]")
    parser.add_argument("--min-support", type=int, default=1,
                        help="Minimum total plays per (state_key, band) [default: 1]")
    parser.add_argument("--min-empirical-support", type=int, default=25,
                        help="Min support for empirical P_h (hybrid) [default: 25]")
    parser.add_argument("--temperature", type=float, default=0.7674,
                        help="Model temperature for P_h [default: 0.7674 = T* from eval]")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N state_keys (smoke test)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint.json in --out-dir")
    args = parser.parse_args()

    human_db = Path(args.human_db) if args.human_db else _ROOT / "data" / "human_db_candidate.sqlite"
    malom_db_dir = args.malom_db_dir or _load_malom_db_dir()
    model_path = Path(args.model) if args.model else _ROOT / "data" / "human_move_policy_net_v2_candidate.npz"
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "data" / "gap_net_v3_dataset"

    print(f"Human DB:      {human_db}")
    print(f"Malom DB:      {malom_db_dir}")
    print(f"Model:         {model_path}")
    print(f"Output dir:    {out_dir}")
    print(f"min_support:   {args.min_support}")
    print(f"min_empirical: {args.min_empirical_support}")
    print(f"temperature:   {args.temperature}")
    if args.limit:
        print(f"*** SMOKE TEST MODE: limit={args.limit} state_keys ***")
    if args.resume:
        print("*** RESUME MODE ***")

    if not human_db.exists():
        print(f"ERROR: human_db not found at {human_db}", file=sys.stderr)
        sys.exit(1)
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    summary = run_extraction(
        human_db=human_db,
        malom_db_dir=malom_db_dir,
        model_path=model_path,
        out_dir=out_dir,
        min_support=args.min_support,
        min_empirical_support=args.min_empirical_support,
        temperature=args.temperature,
        limit=args.limit,
        resume=args.resume,
    )

    _print_gate(summary)

    print(f"Abstained rows: {summary['n_abstained']:,}")
    print(f"Output:         {out_dir}")


if __name__ == "__main__":
    main()
