#!/usr/bin/env python3
"""tools/audit_human_blunders.py — Phase 1 baseline for HumanBlunderNet.

Reads raw human game JSONL + existing Malom labels in data/human_db.sqlite,
classifies every recorded human move by (Elo band × WDL transition × phase),
and prints a coverage / class-balance report.  No network training, no DB
mutation — this is the auditable baseline described in
docs/human_blunder_net_plan.md before we commit to the schema rebuild.

Usage
-----
    .venv/bin/python tools/audit_human_blunders.py \\
        --db data/human_db.sqlite \\
        --games-dir data/human_games \\
        --output data/human_blunder_audit.json

Optional: `--limit-files N` to smoke-test on the first N JSONL files.

Transition categories (mover's perspective, using flip on malom_wdl_after)
-------------------------------------------------------------------------
    W→W  win_preserved       (correct)
    W→D  win_to_draw         (blunder)
    W→L  win_to_loss         (blunder)
    D→D  draw_preserved      (correct)
    D→L  draw_to_loss        (blunder)
    L→L  all_losing          (excluded from BlunderNet training; audited only)
    label_inconsistency      (D→W, L→D, L→W: Malom pre/after disagree)
    unlabelled               (Malom lookup missing on pre or after)

Elo band cut-offs (initial guess — subject to revision after this audit)
------------------------------------------------------------------------
    low   ≤ 1200
    mid   1201–1600
    high  1601+
    unknown  (missing elo or None)

Coverage
--------
For each Elo band we also report the number of unique state_keys with ≥5
human plays in that band — that is the practical floor for computing a
meaningful frequency label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.trajectory_db import make_board_state_key, _norm
from ai.board_symmetry import transform_notation
from game.board import BoardState


# ── Audit provenance version ────────────────────────────────────────────────
# Bumped whenever the audit's sample-flow gates, cell-key definitions, or
# reported columns change.  Written into the JSON output so downstream
# consumers can key on it.
AUDIT_VERSION = "1.1"


# ── WDL helpers ─────────────────────────────────────────────────────────────

_FLIP = {"W": "L", "L": "W", "D": "D"}


def _elo_band(elo: Optional[int]) -> str:
    """Option A cut-offs (PlayOK amateur corpus):
        lower  ≤ 1150
        middle 1151-1250
        upper  ≥ 1251
    'lower/middle/upper' phrasing follows the reviewer's guidance:
    these are strata within this specific corpus, NOT universal strength labels.
    """
    if elo is None:
        return "unknown"
    if elo <= 1150:
        return "lower"
    if elo <= 1250:
        return "middle"
    return "upper"


def _classify_transition(pre_mover: Optional[str], after_next: Optional[str]) -> str:
    """Categorise the WDL transition of a single move.

    pre_mover   : Malom WDL of the pre-move position, from mover's perspective.
    after_next  : Malom WDL of the post-move position, from next-mover's POV
                  (as stored in moves.malom_wdl_after).

    Returns one of: win_preserved / win_to_draw / win_to_loss /
    draw_preserved / draw_to_loss / all_losing /
    label_inconsistency / unlabelled.
    """
    if pre_mover is None or after_next is None:
        return "unlabelled"
    after_mover = _FLIP.get(after_next)
    if after_mover is None:
        return "unlabelled"
    if pre_mover == "W":
        if after_mover == "W":
            return "win_preserved"
        if after_mover == "D":
            return "win_to_draw"
        return "win_to_loss"
    if pre_mover == "D":
        if after_mover == "D":
            return "draw_preserved"
        if after_mover == "L":
            return "draw_to_loss"
        return "label_inconsistency"   # D→W impossible under perfect play
    # pre_mover == "L"
    if after_mover == "L":
        return "all_losing"
    return "label_inconsistency"       # L→W, L→D impossible under perfect play


# ── Provenance helpers ──────────────────────────────────────────────────────

def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    """Return the SHA-256 hex digest of `path`, or None if unreadable."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
    except Exception:
        return None
    return h.hexdigest()


def _games_manifest_hash(paths: list[Path]) -> str:
    """Deterministic hash of the ordered (name, size, mtime) tuples of the
    JSONL files scanned.  Avoids reading every file byte (which would take
    minutes) while still surfacing if the source set changes."""
    h = hashlib.sha256()
    for p in paths:
        try:
            st = p.stat()
        except Exception:
            continue
        h.update(p.name.encode())
        h.update(b"|")
        h.update(str(st.st_size).encode())
        h.update(b"|")
        h.update(str(int(st.st_mtime)).encode())
        h.update(b"\n")
    return h.hexdigest()


def _load_db_provenance(db_path: Path) -> dict:
    """Read schema_version, malom_label_version and other meta rows."""
    if not db_path.exists():
        return {"error": f"HumanDB not found at {db_path}"}
    conn = sqlite3.connect(str(db_path))
    meta = {}
    try:
        for key, value in conn.execute("SELECT key, value FROM meta"):
            meta[key] = value
    except Exception as e:
        meta["error"] = f"failed to read meta: {e}"
    conn.close()
    return meta


def _git_head() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except Exception:
        return None


# ── DB Malom-label preload ──────────────────────────────────────────────────

def _load_malom_labels(db_path: Path) -> tuple[dict, dict]:
    """Return (pos_wdl, move_wdl_after) dicts from human_db.sqlite."""
    if not db_path.exists():
        raise FileNotFoundError(f"HumanDB not found at {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA cache_size = -262144")   # 256 MB page cache
    pos_wdl: dict[str, str] = {}
    for state_key, wdl in conn.execute(
        "SELECT state_key, malom_wdl FROM positions WHERE malom_wdl IS NOT NULL"
    ):
        pos_wdl[state_key] = wdl
    move_wdl: dict[tuple, str] = {}
    for state_key, notation, wdl in conn.execute(
        "SELECT state_key, notation, malom_wdl_after FROM moves WHERE malom_wdl_after IS NOT NULL"
    ):
        move_wdl[(state_key, notation)] = wdl
    conn.close()
    return pos_wdl, move_wdl


# ── JSONL replay ────────────────────────────────────────────────────────────

def _list_game_files(games_dir: Path, limit_files: Optional[int]) -> list[Path]:
    files = sorted(games_dir.glob("*.jsonl"))
    if limit_files is not None:
        files = files[:limit_files]
    return files


def _iter_games(game_files: list[Path]):
    for path in game_files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            yield record


def _skip_record(record: dict) -> bool:
    if record.get("adaptive_softened"):
        return True
    src_type = record.get("source_type", "")
    if src_type not in ("human_vs_human", "human_involved", ""):
        return True
    if record.get("self_play"):
        return True
    if not record.get("moves"):
        return True
    return False


def _phase_bucket(board: BoardState) -> str:
    """Coarse phase label matching the plan's audit dimension."""
    try:
        p = board.phase
    except Exception:
        return "unknown"
    return p if p in ("place", "move") else "unknown"


def audit(
    db_path: Path, games_dir: Path, limit_files: Optional[int] = None,
    top_players: int = 20,
) -> dict:
    """Full-corpus audit.  Tracks every gate in the sample flow so a
    downstream reader can reconcile totals without re-running the script.

    Sample flow (each gate has its own counter):
      raw records
        → non-skip records                    (source_type OK, has moves)
        → mover Elo available (per ply)
        → FEN + notation available
        → BoardState.from_fen_string succeeded
        → make_board_state_key + transform_notation succeeded
        → Malom parent label present
        → Malom child (move-after) label present
        → both present → transition classified into one of the audited categories
    """
    t0 = time.time()
    print(f"[audit] Loading Malom labels from {db_path}")
    pos_wdl, move_wdl = _load_malom_labels(db_path)
    print(f"[audit] {len(pos_wdl):,} position labels · "
          f"{len(move_wdl):,} move-after labels loaded "
          f"({time.time() - t0:.1f}s)")

    # ── Provenance
    db_provenance = _load_db_provenance(db_path)
    db_sha256 = _sha256_file(db_path)
    game_files = _list_game_files(games_dir, limit_files)
    games_manifest_sha256 = _games_manifest_hash(game_files)

    # ── Accumulators
    cell_counts: dict[tuple, int] = defaultdict(int)             # (band, category, phase) → move count
    cell_positions: dict[tuple, set] = defaultdict(set)          # (band, category, phase) → set of state_keys
    band_move_totals: dict[str, int] = defaultdict(int)          # band → total moves
    elo_hist: dict[int, int] = defaultdict(int)                  # 50-Elo bucket → move count
    per_state_band_counts: dict[tuple, int] = defaultdict(int)   # (state_key, band) → play count
    per_state_band_positions_seen: dict[str, set] = defaultdict(set)  # band → set of state_keys

    # Sample-flow gates
    records_seen             = 0     # every JSONL record, including skipped
    records_after_skip       = 0     # passed _skip_record
    games_kept               = 0     # both Elos present or at least one usable per move
    plies_raw                = 0     # all plies in kept games (before any gate)
    plies_missing_fen        = 0
    plies_from_fen_failed    = 0
    plies_state_key_error    = 0
    plies_canon_notation_missing = 0
    plies_mover_elo_missing  = 0
    plies_replayed           = 0     # passed all mechanical gates
    plies_with_parent_label  = 0
    plies_with_after_label   = 0
    plies_with_both_labels   = 0     # === classified into one of the audited transition categories
    plies_missing_parent_only = 0
    plies_missing_after_only  = 0
    plies_missing_both        = 0
    plies_classified         = 0     # duplicate of plies_with_both_labels for clarity

    # Elo-side attrition
    plies_white_side          = 0
    plies_black_side          = 0
    plies_missing_white_elo   = 0    # counted when the mover is White and white_elo is None
    plies_missing_black_elo   = 0

    # Player concentration
    player_move_counts: dict[str, int] = defaultdict(int)   # player_name → plies contributed as mover
    games_per_player: dict[str, int] = defaultdict(int)     # player_name → distinct games as mover
    session_per_player: dict[str, set] = defaultdict(set)

    for record in _iter_games(game_files):
        records_seen += 1
        if _skip_record(record):
            continue
        records_after_skip += 1

        moves = record.get("moves") or []
        w_elo = record.get("white_elo")
        b_elo = record.get("black_elo")
        w_player = record.get("white_player")
        b_player = record.get("black_player")
        session_id = record.get("session_id")

        # Games kept: at least one side has Elo.  Games with neither Elo
        # can still contribute audit-corpus stats but are skipped from
        # Elo-conditioned counters.
        if w_elo is None and b_elo is None:
            continue
        games_kept += 1

        for move in moves:
            plies_raw += 1
            color = move.get("color", "W")
            if color == "W":
                plies_white_side += 1
            else:
                plies_black_side += 1

            notation_raw = _norm(move.get("notation", ""))
            fen = move.get("board_fen_before", "")
            if not notation_raw or not fen:
                plies_missing_fen += 1
                continue
            try:
                board = BoardState.from_fen_string(fen)
            except Exception:
                plies_from_fen_failed += 1
                continue
            try:
                state_key, sym_idx = make_board_state_key(board)
                canon_notation = transform_notation(notation_raw, sym_idx)
            except Exception:
                plies_state_key_error += 1
                continue
            if canon_notation is None:
                plies_canon_notation_missing += 1
                continue

            mover_elo = w_elo if color == "W" else b_elo
            if mover_elo is None:
                if color == "W":
                    plies_missing_white_elo += 1
                else:
                    plies_missing_black_elo += 1
                plies_mover_elo_missing += 1
                # Still count these into plies_replayed so the flow adds up,
                # then place them in the "unknown" band.
            plies_replayed += 1

            band = _elo_band(mover_elo)
            if mover_elo is not None:
                elo_hist[int(mover_elo) // 50 * 50] += 1

            pre = pos_wdl.get(state_key)
            after = move_wdl.get((state_key, canon_notation))
            if pre is not None:
                plies_with_parent_label += 1
            if after is not None:
                plies_with_after_label += 1
            if pre is None and after is None:
                plies_missing_both += 1
            elif pre is None:
                plies_missing_parent_only += 1
            elif after is None:
                plies_missing_after_only += 1
            else:
                plies_with_both_labels += 1

            cat = _classify_transition(pre, after)
            phase = _phase_bucket(board)

            cell_counts[(band, cat, phase)] += 1
            cell_positions[(band, cat, phase)].add(state_key)
            band_move_totals[band] += 1
            per_state_band_counts[(state_key, band)] += 1
            per_state_band_positions_seen[band].add(state_key)

            # Player attribution — count the mover's plies.
            mover_name = w_player if color == "W" else b_player
            if mover_name:
                player_move_counts[mover_name] += 1
                if session_id is not None:
                    session_per_player[mover_name].add(session_id)

    plies_classified = plies_with_both_labels
    for name, sessions in session_per_player.items():
        games_per_player[name] = len(sessions)

    # ── Coverage: unique state_keys per band with ≥N plays.
    coverage_thresholds = (1, 5, 10, 25, 100)
    coverage: dict[str, dict[int, int]] = defaultdict(
        lambda: {t: 0 for t in coverage_thresholds}
    )
    for (state_key, band), n in per_state_band_counts.items():
        for t in coverage_thresholds:
            if n >= t:
                coverage[band][t] += 1
    band_positions_total = {b: len(s) for b, s in per_state_band_positions_seen.items()}

    # ── Cells: (band × category × phase) with n_positions in addition to n_moves.
    cells = []
    all_categories = sorted({c for _, c, _ in cell_counts.keys()})
    all_bands = sorted({b for b, _, _ in cell_counts.keys()})
    all_phases = sorted({p for _, _, p in cell_counts.keys()})
    for band in all_bands:
        for cat in all_categories:
            for phase in all_phases:
                n = cell_counts.get((band, cat, phase), 0)
                if n == 0:
                    continue
                cells.append({
                    "elo_band": band,
                    "transition": cat,
                    "phase": phase,
                    "n_moves": n,
                    "n_positions": len(cell_positions[(band, cat, phase)]),
                })

    # ── Band summary.
    band_summary = []
    for band in sorted(band_move_totals.keys()):
        total_moves = band_move_totals[band]
        blunders = sum(cell_counts.get((band, c, p), 0)
                       for c in ("win_to_draw", "win_to_loss", "draw_to_loss")
                       for p in all_phases)
        correct  = sum(cell_counts.get((band, c, p), 0)
                       for c in ("win_preserved", "draw_preserved")
                       for p in all_phases)
        all_losing = sum(cell_counts.get((band, "all_losing", p), 0)
                         for p in all_phases)
        unlabelled = sum(cell_counts.get((band, "unlabelled", p), 0)
                         for p in all_phases)
        inconsistent = sum(cell_counts.get((band, "label_inconsistency", p), 0)
                           for p in all_phases)
        classified = total_moves - unlabelled
        band_summary.append({
            "elo_band": band,
            "total_moves": total_moves,
            "classified_moves": classified,
            "blunders": blunders,
            "correct_moves": correct,
            "all_losing": all_losing,
            "label_inconsistency": inconsistent,
            "unlabelled": unlabelled,
            "blunder_rate_of_classified": (blunders / classified) if classified else None,
            "distinct_positions": band_positions_total.get(band, 0),
        })

    coverage_out = {}
    for band, counts in coverage.items():
        total_pos = band_positions_total.get(band, 0)
        coverage_out[band] = {
            "counts_at_min_plays": dict(counts),
            "total_positions": total_pos,
            "share_at_min_plays": {
                str(t): (counts[t] / total_pos) if total_pos else 0.0
                for t in coverage_thresholds
            },
        }

    # ── Elo percentiles for band re-cutting decisions.
    elo_hist_sorted = sorted(elo_hist.items())
    total_elo = sum(elo_hist.values())
    cum = 0
    elo_percentiles: dict[str, int] = {}
    _next_percentile = 5
    for bucket, n in elo_hist_sorted:
        cum += n
        pct = 100.0 * cum / total_elo if total_elo else 0.0
        while _next_percentile <= 100 and pct >= _next_percentile:
            elo_percentiles[f"p{_next_percentile}"] = bucket
            _next_percentile += 5

    # ── Player concentration.
    total_player_plies = sum(player_move_counts.values())
    top = sorted(player_move_counts.items(), key=lambda kv: -kv[1])[:top_players]
    top_players_report = [
        {
            "player": name,
            "plies": plies,
            "share_of_all_player_attributed_plies":
                (plies / total_player_plies) if total_player_plies else 0.0,
            "distinct_games": games_per_player.get(name, 0),
        }
        for name, plies in top
    ]
    unique_players = len(player_move_counts)
    top10_share = 0.0
    if total_player_plies and unique_players:
        top10 = sum(v for _, v in sorted(player_move_counts.items(), key=lambda kv: -kv[1])[:10])
        top10_share = top10 / total_player_plies

    # ── Assemble the report.
    report = {
        "meta": {
            "audit_version": AUDIT_VERSION,
            "audit_script": "tools/audit_human_blunders.py",
            "git_head": _git_head(),
            "cwd": str(_ROOT),
            "elapsed_seconds": round(time.time() - t0, 1),
            "elo_bands": {
                "lower":   "≤1150",
                "middle":  "1151-1250",
                "upper":   "1251+",
                "unknown": "no mover Elo recorded on that side",
                "note": "Option A boundaries within this PlayOK amateur corpus. "
                        "Not universal strength labels.",
            },
        },
        "sources": {
            "db_path": str(db_path),
            "db_sha256": db_sha256,
            "db_meta": db_provenance,
            "games_dir": str(games_dir),
            "n_game_files_scanned": len(game_files),
            "games_manifest_sha256": games_manifest_sha256,
            "games_manifest_note":
                "SHA-256 of the sorted (filename, size, mtime) tuples of the "
                "scanned JSONL files.  Content-hashing every game file was "
                "considered but rejected on cost.",
        },
        "sample_flow": {
            "records_seen":               records_seen,
            "records_after_skip":         records_after_skip,
            "games_kept_with_any_elo":    games_kept,
            "plies_raw":                  plies_raw,
            "plies_missing_fen_or_notation": plies_missing_fen,
            "plies_from_fen_failed":      plies_from_fen_failed,
            "plies_state_key_error":      plies_state_key_error,
            "plies_canon_notation_missing": plies_canon_notation_missing,
            "plies_mover_elo_missing":    plies_mover_elo_missing,
            "plies_replayed":             plies_replayed,
            "plies_with_parent_label":    plies_with_parent_label,
            "plies_with_after_label":     plies_with_after_label,
            "plies_missing_parent_only":  plies_missing_parent_only,
            "plies_missing_after_only":   plies_missing_after_only,
            "plies_missing_both":         plies_missing_both,
            "plies_classified":           plies_classified,
            "plies_white_side":           plies_white_side,
            "plies_black_side":           plies_black_side,
            "plies_missing_white_elo":    plies_missing_white_elo,
            "plies_missing_black_elo":    plies_missing_black_elo,
        },
        "band_summary": band_summary,
        "cells": cells,
        "coverage_by_band": coverage_out,
        "elo_histogram_50bucket": {str(k): v for k, v in elo_hist_sorted},
        "elo_percentiles_5pct": elo_percentiles,
        "player_concentration": {
            "unique_players_seen":              unique_players,
            "total_player_attributed_plies":    total_player_plies,
            "top10_share_of_plies":             top10_share,
            "top_players":                      top_players_report,
        },
    }
    return report


# ── CLI ─────────────────────────────────────────────────────────────────────

def _print_summary(report: dict) -> None:
    print()
    print("=" * 78)
    print("Provenance")
    print("=" * 78)
    src = report["sources"]
    print(f"  db path              : {src['db_path']}")
    print(f"  db sha256            : {src['db_sha256']}")
    print(f"  db schema_version    : {src['db_meta'].get('schema_version', '<missing>')}")
    print(f"  db malom_label_ver   : {src['db_meta'].get('malom_label_version', '<missing>')}")
    print(f"  games dir            : {src['games_dir']}")
    print(f"  n game files         : {src['n_game_files_scanned']:,}")
    print(f"  games manifest sha   : {src['games_manifest_sha256']}")
    print(f"  audit version        : {report['meta']['audit_version']}")
    print(f"  git HEAD             : {report['meta']['git_head']}")
    print()
    print("=" * 78)
    print("Sample flow (gate counts)")
    print("=" * 78)
    sf = report["sample_flow"]
    print(f"  records seen                    : {sf['records_seen']:,}")
    print(f"  records after skip filter       : {sf['records_after_skip']:,}")
    print(f"  games kept with any Elo         : {sf['games_kept_with_any_elo']:,}")
    print(f"  plies raw                       : {sf['plies_raw']:,}")
    print(f"  plies missing fen / notation    : {sf['plies_missing_fen_or_notation']:,}")
    print(f"  plies from-fen failed           : {sf['plies_from_fen_failed']:,}")
    print(f"  plies state_key error           : {sf['plies_state_key_error']:,}")
    print(f"  plies canon-notation missing    : {sf['plies_canon_notation_missing']:,}")
    print(f"  plies mover Elo missing         : {sf['plies_mover_elo_missing']:,}")
    print(f"    ├─ white side (elo=None, mover=W) : {sf['plies_missing_white_elo']:,}")
    print(f"    └─ black side (elo=None, mover=B) : {sf['plies_missing_black_elo']:,}")
    print(f"  plies replayed (all mechanical gates OK)     : {sf['plies_replayed']:,}")
    print(f"    ├─ with Malom parent label                  : {sf['plies_with_parent_label']:,}")
    print(f"    ├─ with Malom after-label                   : {sf['plies_with_after_label']:,}")
    print(f"    ├─ missing parent only                      : {sf['plies_missing_parent_only']:,}")
    print(f"    ├─ missing after only                       : {sf['plies_missing_after_only']:,}")
    print(f"    ├─ missing both                             : {sf['plies_missing_both']:,}")
    print(f"    └─ classified (both labels present)         : {sf['plies_classified']:,}")
    print(f"  plies white-side moves          : {sf['plies_white_side']:,}")
    print(f"  plies black-side moves          : {sf['plies_black_side']:,}")
    print()
    print("=" * 78)
    print("Per-Elo-band summary")
    print("=" * 78)
    print(f"  {'band':<8}{'moves':>12}{'classified':>13}{'blunders':>11}"
          f"{'correct':>10}{'unlabelled':>13}{'positions':>12}{'blunder %':>12}")
    for row in report["band_summary"]:
        rate = row["blunder_rate_of_classified"]
        rate_s = "n/a" if rate is None else f"{rate*100:.2f}"
        print(f"  {row['elo_band']:<8}{row['total_moves']:>12,}"
              f"{row['classified_moves']:>13,}{row['blunders']:>11,}"
              f"{row['correct_moves']:>10,}{row['unlabelled']:>13,}"
              f"{row['distinct_positions']:>12,}{rate_s:>12}")
    print()
    print("=" * 78)
    print("Coverage — unique state_keys per band at ≥N plays (count / share)")
    print("=" * 78)
    thresholds = (1, 5, 10, 25, 100)
    for band in sorted(report["coverage_by_band"].keys()):
        row = report["coverage_by_band"][band]
        counts = row["counts_at_min_plays"]
        shares = row["share_at_min_plays"]
        total  = row["total_positions"]
        print(f"  {band:<8}  (total distinct positions = {total:,})")
        for t in thresholds:
            n = counts.get(t, 0)
            s = shares.get(str(t), 0.0)
            print(f"    ≥{t:<4}   {n:>10,}   ({s*100:5.2f} % of positions)")
        print()
    print("=" * 78)
    print("Mover-Elo 5th-percentile ladder (for band re-bucketing)")
    print("=" * 78)
    print(f"  {'percentile':<12}{'Elo bucket':>12}")
    _percentiles = report.get("elo_percentiles_5pct", {})
    for key in sorted(_percentiles.keys(), key=lambda k: int(k.lstrip("p"))):
        print(f"  {key:<12}{_percentiles[key]:>12,}")
    print()
    print("=" * 78)
    print("Cells (band × transition × phase; n_moves and n_positions)")
    print("=" * 78)
    for band in sorted({c["elo_band"] for c in report["cells"]}):
        print(f"  band = {band}")
        for cell in report["cells"]:
            if cell["elo_band"] != band:
                continue
            print(f"    {cell['transition']:<22}{cell['phase']:<7}"
                  f"moves={cell['n_moves']:>10,}  positions={cell['n_positions']:>8,}")
        print()
    print("=" * 78)
    print("Player concentration (top 10 movers)")
    print("=" * 78)
    pc = report["player_concentration"]
    print(f"  unique players           : {pc['unique_players_seen']:,}")
    print(f"  total attributed plies   : {pc['total_player_attributed_plies']:,}")
    print(f"  top-10 share of plies    : {pc['top10_share_of_plies']*100:.2f} %")
    for row in pc["top_players"][:10]:
        print(f"    {row['player']:<20}  plies={row['plies']:>8,}   "
              f"games={row['distinct_games']:>6,}   "
              f"share={row['share_of_all_player_attributed_plies']*100:5.2f} %")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db",         type=Path, default=Path("data/human_db.sqlite"))
    p.add_argument("--games-dir",  type=Path, default=Path("data/human_games"))
    p.add_argument("--output",     type=Path, default=Path("data/human_blunder_audit.json"))
    p.add_argument("--top-players", type=int, default=20,
                   help="How many top movers to list in player concentration report.")
    p.add_argument("--limit-files", type=int, default=None,
                   help="Cap number of JSONL files scanned (smoke tests).")
    args = p.parse_args()

    report = audit(args.db, args.games_dir,
                   limit_files=args.limit_files,
                   top_players=args.top_players)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[audit] Report written → {args.output}")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
