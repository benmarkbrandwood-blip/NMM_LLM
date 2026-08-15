#!/usr/bin/env python3
"""tools/extract_gap_v3_dataset_v2.py — GapNet v3 Stage D extractor (Batch 4).

Rewrite of tools/extract_gap_v3_dataset.py per
docs/gap_net_v3_stage_e_rebuild_checklist.md.  Reads raw
data/human_games/*.jsonl directly (not the aggregated moves_elo_bins),
applies the session-ledger strict owning-tier rule so no train sample
corresponds to a state_key that was reached by a val/test session,
and emits three parallel targets files (model / uniform / empirical G_v).

Owning-tier rule (per checklist):
  1. For each state_key, enumerate the sessions that reached it.
  2. The session with the smallest SHA-256 hash is the owning session;
     its split_tier is the state_key's owning tier.
  3. Aggregate counts using events from the owning tier only.
  4. Discard events for that state from all other tiers.
  5. Never combine counts across tiers before or during tier assignment.
  6. Record per-(band, phase) discarded state and event counts.

Fail-closed A/B/C target discipline (Codex 2026-08-11):
  - targets.f32.bin and targets_uniform.f32.bin are fully finite.
    Unavailable R_v abstains the entire row; never emits NaN.
  - targets_empirical.f32.bin allows NaN as the 'support < min_support'
    row-level sentinel.

Output: data/gap_net_v3_dataset_v2/
  - parent_feats.f32.bin        (N, 79) float32
  - targets.f32.bin             (N, 3)  float32 [g_v_A, g_v_B, g_v_C]  finite
  - targets_uniform.f32.bin     (N, 3)  float32                        finite
  - targets_empirical.f32.bin   (N, 3)  float32   NaN row where support < min
  - metadata.npz                state_keys, band_idx, split, phase,
                                mover_color, n_legal, ph_source,
                                owning_session_min_hash, provenance
  - abstained.jsonl             per-row abstention reasons
  - provenance.json             separate copy of provenance

Coverage floor (per user decision 2026-08-12): 1,275,400 rows (50 % of v1
state-key-split output).  Fewer rows → halt-and-report; do not silently
fall back to mixed aggregates.

Usage
-----
    .venv/bin/python tools/extract_gap_v3_dataset_v2.py \\
        --session-ledger data/gap_v3_session_ledger.json \\
        --teacher-net    data/human_move_policy_net_v3_teacher_candidate.npz

Smoke:
    .venv/bin/python tools/extract_gap_v3_dataset_v2.py \\
        --session-ledger data/gap_v3_session_ledger.json \\
        --teacher-net    data/human_move_policy_net_v2_candidate.npz \\
        --limit-files 500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.human_move_policy_advisor import HumanMovePolicyAdvisor  # noqa: E402
from ai.malom_db import (                                        # noqa: E402
    MalomDB, RegretResult,
    _MALOM_LABEL_VERSION, _REGRET_VERSION,
)
from ai.trajectory_db import make_board_state_key                # noqa: E402
from ai.value_net import board_to_features, _INPUT_DIM           # noqa: E402
from game.board import BoardState                                # noqa: E402
from game.rules import get_all_legal_moves                       # noqa: E402
from learned_ai.data.elo_binning import (                        # noqa: E402
    option_a_band_from_bin, OPTION_A_NAME,
)


# ── Constants ────────────────────────────────────────────────────────────────

EXTRACT_VERSION      = "v2"           # Batch 4 / session-ledger output
_N_HEADS             = 3               # A, B, C (Component D dropped for regret_v1)
_BAND_TO_IDX         = {"lower": 0, "middle": 1, "upper": 2}
_SPLIT_TO_INT8       = {"train": 0, "val": 1, "test": 2}

# Coverage floor: 50 % of v1's 2,550,799-row output (user decision 2026-08-12).
_DEFAULT_COVERAGE_FLOOR_ROWS = 1_275_400

_DEFAULT_MIN_EMPIRICAL_SUPPORT = 25
_DEFAULT_TEMPERATURE           = 0.7674   # inherited from v1 provenance


# ── Provenance helpers ───────────────────────────────────────────────────────

def _sha256_file(path: Path, chunk: int = 1 << 20) -> Optional[str]:
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


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ── Move notation (matches build_human_db_sha.py) ────────────────────────────

def _move_notation(mv: dict) -> str:
    frm = mv.get("from")
    to  = mv.get("to") or ""
    cap = mv.get("capture") or ""
    base = f"{frm}-{to}" if frm else to
    return f"{base}x{cap}" if cap else base


def _elo_to_band(elo: int | float) -> str:
    """Bin the elo to a 50-Elo bucket then map to Option A band."""
    try:
        bucket = int(elo) // 50 * 50
    except Exception:
        return "unknown"
    return option_a_band_from_bin(bucket)


def _phase_from_board(board: BoardState) -> str:
    """Return placement/movement/fly phase label."""
    # ai.value_net already exposes phase logic via metadata features but the
    # explicit labels come from board.phase.  Kept simple; adjust if needed.
    if hasattr(board, "phase"):
        p = board.phase
        if isinstance(p, str):
            return p
    # Fallback: use pieces_placed
    tot_placed = sum(board.pieces_placed.values())
    if tot_placed < 18:
        return "place"
    stm_on_board = board.pieces_on_board[board.turn]
    if stm_on_board <= 3:
        return "fly"
    return "move"


# ── Manifest and teacher lineage checks (Codex P1 2026-08-14) ────────────────

def _verify_files_match_manifest(
    games_dir: Path,
    ledger_file_shas: dict[str, str],
    limit_files: Optional[int],
) -> None:
    """Codex P1(2) 2026-08-14: prior implementation only verified files
    that were PRESENT on disk against the ledger's file manifest — deleted
    files (present in manifest, missing from disk) went undetected.  For a
    full run we now require exact set equality between disk and manifest.
    Smoke runs (limit_files set) may see a subset of the manifest, but any
    disk file NOT in manifest is still reported.
    """
    disk_files = {
        p.relative_to(games_dir).as_posix()
        for p in games_dir.glob("*.jsonl")
    }
    manifest_files = set(ledger_file_shas.keys())
    extra_on_disk = disk_files - manifest_files
    missing_from_disk = manifest_files - disk_files
    if limit_files is not None:
        if extra_on_disk:
            print(f"[extract v2] Smoke: {len(extra_on_disk)} files on disk not "
                  f"in ledger manifest (skipped): "
                  f"{sorted(extra_on_disk)[:5]}")
        return
    if missing_from_disk or extra_on_disk:
        raise RuntimeError(
            f"[extract v2] Files-manifest disagreement (full run):\n"
            f"  missing from disk (in ledger, not found): "
            f"{len(missing_from_disk)} files, sample "
            f"{sorted(missing_from_disk)[:5]}\n"
            f"  extra on disk (not in ledger):            "
            f"{len(extra_on_disk)} files, sample "
            f"{sorted(extra_on_disk)[:5]}\n"
            f"Rebuild the ledger against the current games_dir before extracting."
        )


def _verify_teacher_lineage(
    teacher_net: Path,
    ledger_path: Path,
    ledger_info: dict,
) -> dict:
    """Verify teacher .npz was trained on the SAME session-ledger-isolated
    dataset that this extraction consumes.

    Codex P1(3) 2026-08-14: prior implementation accepted any teacher NPZ
    that existed, so the leaky v2 candidate or a v3 teacher trained against
    a different ledger could reach gate_status="ok".  We now enforce:
      - Teacher provenance carries
        dataset_split_scheme == session_ledger_strict_single_tier.
      - Teacher's recorded session_ledger_sha256 matches sha256(ledger_path).
      - Teacher's recorded session_ledger_files_manifest_sha256 matches
        this ledger's files_manifest_sha256.
    Missing provenance field → fail closed.

    NB (self-review 2026-08-14): the checklist's Decision 6B also requires
    the same "split seed" between teacher and extractor.  Split seed lives
    in the ledger's `split_manifest_version` (and `split_function`).  It is
    NOT verified as a separate field here because ledger_sha256 already
    covers it transitively: any change to split_manifest_version changes
    the ledger's content and therefore its SHA-256, which our sha check
    catches.  The files_manifest_sha256 check adds defence-in-depth against
    downstream drift.  Do NOT weaken the ledger_sha256 check without
    replacing this transitive guarantee.
    """
    z = np.load(str(teacher_net), allow_pickle=True)
    if "provenance_json" not in z.files:
        raise RuntimeError(
            f"[extract v2] teacher {teacher_net} has no provenance_json "
            f"field; a v3 session-ledger-isolated teacher is required "
            f"(Codex P1 2026-08-14)"
        )
    prov = json.loads(str(z["provenance_json"]))
    scheme = prov.get("dataset_split_scheme")
    if scheme != "session_ledger_strict_single_tier":
        raise RuntimeError(
            f"[extract v2] teacher {teacher_net} was trained under "
            f"split_scheme={scheme!r}; required "
            f"'session_ledger_strict_single_tier'.  This rejects the leaky "
            f"v2 candidate and any other non-session-ledger teacher."
        )
    teacher_ledger_sha = prov.get("session_ledger_sha256")
    if not teacher_ledger_sha:
        raise RuntimeError(
            f"[extract v2] teacher {teacher_net} has no session_ledger_sha256 "
            f"in provenance; cannot verify ledger lineage"
        )
    actual_sha = _sha256_file(ledger_path)
    if teacher_ledger_sha != actual_sha:
        raise RuntimeError(
            f"[extract v2] teacher was trained against ledger sha "
            f"{teacher_ledger_sha}, but current --session-ledger at "
            f"{ledger_path} has sha {actual_sha}; teacher and extractor "
            f"must consume the SAME ledger"
        )
    teacher_manifest_sha = prov.get("session_ledger_files_manifest_sha256")
    ledger_manifest_sha  = ledger_info["provenance"].get("files_manifest_sha256")
    if teacher_manifest_sha != ledger_manifest_sha:
        raise RuntimeError(
            f"[extract v2] teacher's session_ledger_files_manifest_sha256 "
            f"({teacher_manifest_sha}) does not match this ledger's "
            f"({ledger_manifest_sha})"
        )
    return prov


# ── Teacher P_h with fail-closed abstain ────────────────────────────────────

def _compute_teacher_ph(
    advisor: "HumanMovePolicyAdvisor",
    board: "BoardState",
    legal_moves: list,
    band: str,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Return (ph_model, None) on success or (None, reason_string) on failure.

    Codex P1 2026-08-14: `advisor.probs()` silently returned a uniform vector
    on degenerate teacher output.  We use `probs_strict()` which raises, then
    additionally enforce shape, finite/non-negative, and sum≈1 so no bad
    teacher output can silently enter G_v.  Callers should abstain the row
    and write the returned reason to `abstained.jsonl`.

    Extracted from the emit loop 2026-08-15 (Codex P2) so the abstain path
    can be exercised directly by unit tests.
    """
    try:
        ph_model = advisor.probs_strict(board, legal_moves, elo_band=band)
        ph_model = np.asarray(ph_model, dtype=np.float32)
        if ph_model.shape != (len(legal_moves),):
            raise ValueError(
                f"ph_model shape {ph_model.shape} != ({len(legal_moves)},)"
            )
        if not np.isfinite(ph_model).all():
            raise ValueError("ph_model contains non-finite values")
        if (ph_model < 0.0).any():
            raise ValueError("ph_model contains negative values")
        if abs(float(ph_model.sum()) - 1.0) > 1e-4:
            raise ValueError(
                f"ph_model does not sum to 1: sum={float(ph_model.sum())!r}"
            )
    except Exception as e:
        return None, f"teacher_probs_failed:{e!s}"
    return ph_model, None


# ── Session ledger ───────────────────────────────────────────────────────────

def _load_session_ledger(path: Path, allow_partial: bool = False) -> dict:
    """Load and index the session ledger by session_id.

    Codex hardening 2026-08-14 (matches P1-B pattern): the ledger is loaded
    via `_verify_ledger_complete()`, which fail-closes on:
      - partial ledgers (built with --limit-files) unless allow_partial=True
      - non-strict ledgers (built with --allow-malformed)
      - ledgers that tolerated malformed JSON lines
      - empty ledgers (n_sessions == 0)

    Returns dict:
        session_meta:     session_id → {'tier', 'session_hash', 'source_file'}
        ledger_file_shas: rel_path   → sha256          (for JSONL drift check)
        provenance:       ledger provenance sub-dict
        n_by_split:       ledger's own count
    """
    if not path.exists():
        raise FileNotFoundError(f"session ledger not found: {path}")
    # Import lazily to avoid a hard module-level dep in unit tests that only
    # exercise the pure scan functions.
    from tools.build_gap_v3_session_ledger import (           # noqa: E402
        LedgerBuildError, _verify_ledger_complete,
    )
    try:
        ledger = _verify_ledger_complete(path, allow_partial=allow_partial)
    except LedgerBuildError as e:
        raise RuntimeError(f"session_ledger verification failed: {e}") from e

    session_meta: dict[str, dict] = {}
    for entry in ledger["sessions"]:
        session_meta[entry["session_id"]] = {
            "tier":         entry["split"],
            "session_hash": entry["session_hash"],
            "source_file":  entry["source_file"],
        }
    ledger_file_shas = {f["rel_path"]: f["sha256"] for f in ledger.get("files", [])}
    provenance = {
        k: ledger.get(k) for k in (
            "ledger_version", "split_function", "split_manifest_version",
            "git_commit", "built_at", "games_dir", "n_jsonl_files",
            "n_sessions", "files_manifest_sha256", "is_partial", "strict",
            "n_malformed_lines",
        )
    }
    return {
        "session_meta":     session_meta,
        "ledger_file_shas": ledger_file_shas,
        "provenance":       provenance,
        "n_by_split":       ledger.get("n_by_split", {}),
    }


# ── JSONL scanning: pass 1 (owning tier) + pass 2 (aggregation) ──────────────

def _iter_jsonl_events(
    games_dir: Path,
    session_meta: dict[str, dict],
    limit_files: Optional[int] = None,
    ledger_file_shas: Optional[dict[str, str]] = None,
    strict: bool = True,
):
    """Stream (state_key, session_id, band, notation, phase, mover_color)
    events across all JSONL files under games_dir.

    Codex hardening 2026-08-14 (matches P1-A/P1-B pattern):
    - When `ledger_file_shas` is provided, each scanned file's SHA-256 is
      verified against the recorded value.  File-not-in-manifest is a
      fatal error under a full run (limit_files is None); tolerated (skip)
      under a smoke run.  SHA drift is always fatal.
    - When `strict` is True (default), malformed JSON lines are fatal.
      When False, they are counted but skipped.
    - Single-pass file read (SHA + parse from the same bytes) to avoid
      read/hash drift.

    Codex P1 2026-08-15 (between-pass file-deletion detection):
    When `ledger_file_shas` is provided, iteration is driven from the frozen
    manifest keys, NOT a fresh disk glob.  A file listed in the manifest but
    missing from disk (deleted between the run_extraction precheck and this
    pass, or between passes) is fatal.  A disk file missing from the
    manifest is fatal under full runs, tolerated under smoke (skipped —
    never iterated because it is not in the manifest set).

    Sessions absent from the ledger are skipped (drift indicator).
    """
    disk_files = {
        p.relative_to(games_dir).as_posix()
        for p in games_dir.glob("*.jsonl")
    }
    if ledger_file_shas is not None:
        manifest_files = set(ledger_file_shas.keys())
        extra_on_disk = disk_files - manifest_files
        if extra_on_disk and limit_files is None:
            raise RuntimeError(
                f"[extract v2] disk file(s) not in ledger file manifest "
                f"(full run): {sorted(extra_on_disk)[:5]}"
            )
        iter_rels = sorted(manifest_files)
    else:
        iter_rels = sorted(disk_files)
    if limit_files is not None:
        iter_rels = iter_rels[:limit_files]

    for rel in iter_rels:
        fpath = games_dir / rel
        try:
            raw_bytes = fpath.read_bytes()
        except FileNotFoundError:
            if ledger_file_shas is not None:
                raise RuntimeError(
                    f"[extract v2] ledger-listed file missing from disk: "
                    f"{rel} (possible between-pass deletion; ledger claims "
                    f"sha={ledger_file_shas.get(rel, '?')[:16]}…)"
                )
            continue
        if ledger_file_shas is not None:
            expected = ledger_file_shas[rel]   # rel came from manifest
            actual = hashlib.sha256(raw_bytes).hexdigest()
            if actual != expected:
                raise RuntimeError(
                    f"[extract v2] SHA-256 drift for {rel}: "
                    f"ledger={expected} actual={actual}"
                )
        for raw_line_b in raw_bytes.splitlines():
            try:
                raw = raw_line_b.decode("utf-8").strip()
            except UnicodeDecodeError:
                if strict:
                    raise RuntimeError(
                        f"[extract v2] non-UTF-8 bytes in {fpath} (strict)"
                    )
                continue
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception as e:
                if strict:
                    raise RuntimeError(
                        f"[extract v2] malformed JSON in {fpath}: {e!s} (strict)"
                    )
                continue

            session_id = str(rec.get("session_id") or fpath.stem)
            sm = session_meta.get(session_id)
            if sm is None:
                continue                     # unknown session (ledger mismatch)

            white_elo = rec.get("white_elo")
            black_elo = rec.get("black_elo")

            moves = rec.get("moves") or []
            for mv in moves:
                fen = mv.get("board_fen_before")
                if not fen:
                    continue
                try:
                    board = BoardState.from_fen_string(fen)
                    sk, _ = make_board_state_key(board)
                except Exception:
                    continue

                color = (mv.get("color") or "").lower()
                if color.startswith("w"):
                    elo = white_elo
                    mover_color = "W"
                elif color.startswith("b"):
                    elo = black_elo
                    mover_color = "B"
                else:
                    elo = white_elo if board.turn == "W" else black_elo
                    mover_color = board.turn

                band = _elo_to_band(elo)
                if band == "unknown":
                    continue

                notation = _move_notation(mv)
                yield sk, session_id, band, notation, _phase_from_board(board), mover_color


def _scan_pass1_owning_tier(
    games_dir: Path,
    session_meta: dict[str, dict],
    limit_files: Optional[int] = None,
    log_every: int = 1_000_000,
    ledger_file_shas: Optional[dict[str, str]] = None,
    strict: bool = True,
) -> tuple[dict[str, dict], dict]:
    """First JSONL pass.  Determine each state_key's owning tier.

    Owning session = session with smallest sha256(session_id) hash among the
    sessions that reached this state_key.  Owning tier = that session's tier.

    Returns:
        state_key_owning: dict[state_key, {
            "tier": str,
            "session_id": str,
            "session_hash": str,
            "tier_event_counts": {tier: int},   # events per tier that reached this sk
        }]
        stats: {"n_events_total", "n_state_keys_seen", "elapsed_seconds"}
    """
    t0 = time.time()
    owning: dict[str, dict] = {}
    n_events = 0

    for sk, session_id, _band, _notation, _phase, _mover in _iter_jsonl_events(
        games_dir, session_meta, limit_files=limit_files,
        ledger_file_shas=ledger_file_shas, strict=strict,
    ):
        n_events += 1
        sm = session_meta[session_id]
        session_hash = sm["session_hash"]
        tier = sm["tier"]

        cur = owning.get(sk)
        if cur is None:
            owning[sk] = {
                "tier":              tier,
                "session_id":        session_id,
                "session_hash":      session_hash,
                "tier_event_counts": {tier: 1},
            }
        else:
            cur["tier_event_counts"][tier] = cur["tier_event_counts"].get(tier, 0) + 1
            if session_hash < cur["session_hash"]:
                cur["tier"]         = tier
                cur["session_id"]   = session_id
                cur["session_hash"] = session_hash

        if log_every and n_events % log_every == 0:
            print(f"[extract v2] pass1 events={n_events:,}  "
                  f"state_keys={len(owning):,}  ({time.time()-t0:.0f}s)")

    return owning, {
        "n_events_total":      n_events,
        "n_state_keys_seen":   len(owning),
        "elapsed_seconds":     round(time.time() - t0, 1),
    }


def _scan_pass2_aggregate(
    games_dir: Path,
    session_meta: dict[str, dict],
    state_key_owning: dict[str, dict],
    limit_files: Optional[int] = None,
    log_every: int = 1_000_000,
    ledger_file_shas: Optional[dict[str, str]] = None,
    strict: bool = True,
) -> tuple[dict, dict]:
    """Second JSONL pass.  Aggregate counts for owning-tier events only.

    Returns:
        counts: dict[(state_key, band), {"notation_counts": dict[str, int],
                                          "phase": str,
                                          "mover_color": str}]
        disposition: {
            "events_kept_by_tier":       {tier: int},
            "events_discarded_by_tier":  {tier: int},   # event's tier ≠ owning tier
            "events_dropped_uncovered":  int,           # state_key had no owning entry
            "states_dropped_uncovered":  int,
            "states_kept_by_band_phase": nested dict for reporting,
            "events_kept_by_band_phase": nested dict for reporting,
        }
    """
    t0 = time.time()
    counts: dict[tuple[str, str], dict] = {}
    events_kept_by_tier      = defaultdict(int)
    events_discarded_by_tier = defaultdict(int)
    events_dropped_uncovered = 0
    kept_by_band_phase = defaultdict(int)
    disc_by_band_phase = defaultdict(int)
    # Codex P2 2026-08-14: per-(band, phase) state counts alongside events.
    # A state_key can appear in multiple (b, p) buckets; we count it once per
    # bucket if any owning-tier / non-owning-tier event landed in that bucket.
    states_kept_by_band_phase: dict[tuple[str, str], set[str]] = defaultdict(set)
    states_discarded_by_band_phase: dict[tuple[str, str], set[str]] = defaultdict(set)

    n_events = 0
    for sk, session_id, band, notation, phase, mover_color in _iter_jsonl_events(
        games_dir, session_meta, limit_files=limit_files,
        ledger_file_shas=ledger_file_shas, strict=strict,
    ):
        n_events += 1
        owning_entry = state_key_owning.get(sk)
        if owning_entry is None:
            events_dropped_uncovered += 1
            continue
        owning_tier   = owning_entry["tier"]
        session_tier  = session_meta[session_id]["tier"]
        key           = (sk, band)
        bp            = (band, phase)
        if session_tier != owning_tier:
            events_discarded_by_tier[session_tier] += 1
            disc_by_band_phase[bp] += 1
            states_discarded_by_band_phase[bp].add(sk)
            continue

        events_kept_by_tier[session_tier] += 1
        kept_by_band_phase[bp] += 1
        states_kept_by_band_phase[bp].add(sk)

        entry = counts.get(key)
        if entry is None:
            counts[key] = {
                "notation_counts": {notation: 1},
                "phase":           phase,
                "mover_color":     mover_color,
            }
        else:
            entry["notation_counts"][notation] = entry["notation_counts"].get(notation, 0) + 1

        if log_every and n_events % log_every == 0:
            print(f"[extract v2] pass2 events={n_events:,}  "
                  f"aggregated_keys={len(counts):,}  ({time.time()-t0:.0f}s)")

    disposition = {
        "events_kept_by_tier":              dict(events_kept_by_tier),
        "events_discarded_by_tier":         dict(events_discarded_by_tier),
        "events_dropped_uncovered":         events_dropped_uncovered,
        "kept_by_band_phase":               {f"{b}|{p}": n for (b, p), n in kept_by_band_phase.items()},
        "disc_by_band_phase":               {f"{b}|{p}": n for (b, p), n in disc_by_band_phase.items()},
        # Codex P2 2026-08-14: per-(band, phase) state counts.
        "states_kept_by_band_phase":        {f"{b}|{p}": len(s) for (b, p), s in states_kept_by_band_phase.items()},
        "states_discarded_other_tier_by_band_phase":
                                            {f"{b}|{p}": len(s) for (b, p), s in states_discarded_by_band_phase.items()},
        "events_discarded_other_tier_by_band_phase":
                                            {f"{b}|{p}": n for (b, p), n in disc_by_band_phase.items()},
        "elapsed_seconds":                  round(time.time() - t0, 1),
    }
    return counts, disposition


# ── Fail-closed helpers (borrowed / adapted from v1 extractor) ───────────────

def _check_emittable(
    regrets: dict[str, RegretResult],
    legal_moves: list[dict],
) -> tuple[bool, str]:
    """Strict fail-closed §5.5: all A/B/C components must be non-None for all moves."""
    for mv in legal_moves:
        rr = regrets[_move_notation(mv)]
        if not rr.available:
            return False, f"move_unavailable:{rr.unavailable_reason}"
        if rr.best_omv is None:
            return False, "best_omv_none"
        if rr.components["class_downgrade_prob"] is None:
            return False, "comp_a_none"
        if rr.components["wdl_utility_loss"] is None:
            return False, "comp_b_none"
        if rr.components["ordinal_rank_loss"] is None:
            return False, "comp_c_none"
    return True, ""


# ── G_v computation ─────────────────────────────────────────────────────────

def _compute_gv(
    ph: np.ndarray,
    legal_moves: list[dict],
    regrets: dict[str, RegretResult],
) -> tuple[float, float, float]:
    """Weighted G_v over three components (A/B/C)."""
    g_a = g_b = g_c = 0.0
    for i, mv in enumerate(legal_moves):
        w = float(ph[i])
        rr = regrets[_move_notation(mv)]
        g_a += w * rr.components["class_downgrade_prob"]
        g_b += w * rr.components["wdl_utility_loss"]
        g_c += w * rr.components["ordinal_rank_loss"]
    return g_a, g_b, g_c


def _uniform_ph(n_legal: int) -> np.ndarray:
    if n_legal <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.full(n_legal, 1.0 / n_legal, dtype=np.float32)


def _empirical_ph(
    notation_counts: dict[str, int],
    legal_moves: list[dict],
    min_support: int,
) -> tuple[Optional[np.ndarray], int]:
    """Empirical P_h over legal moves.

    Codex P1 2026-08-14: previously the denominator was the sum of ALL
    observed notation counts, including notations that were not in
    `legal_moves`.  That silently produced sub-normalised P_h (e.g. legal 25
    + illegal 25 → legal-move vector summing to 0.5) and multiplicatively
    scaled the resulting empirical G_v.  Fix: only legal-notation counts
    enter the denominator; illegal-notation counts are dropped and reported
    separately so the caller can attribute them in provenance / abstain.

    Returns (ph, n_illegal_events):
      ph:               length-|legal_moves| float32 that sums to 1 (within eps),
                        or None if legal_total < min_support.
      n_illegal_events: sum of counts for notations NOT in legal_moves.
    """
    legal_notation_set = {_move_notation(mv) for mv in legal_moves}
    legal_total   = 0
    illegal_total = 0
    for nt, cnt in notation_counts.items():
        if nt in legal_notation_set:
            legal_total += cnt
        else:
            illegal_total += cnt
    if legal_total < min_support:
        return None, illegal_total
    ph = np.zeros(len(legal_moves), dtype=np.float64)
    for i, mv in enumerate(legal_moves):
        ph[i] = notation_counts.get(_move_notation(mv), 0) / legal_total
    result = ph.astype(np.float32)
    # Normalisation sanity — legal-only denominator should always yield sum≈1.
    assert abs(float(result.sum()) - 1.0) < 1e-4, (
        f"empirical_ph not normalised: sum={float(result.sum())!r}"
    )
    return result, illegal_total


# ── Board reconstruction ─────────────────────────────────────────────────────

def _board_from_state_key(state_key: str) -> Optional[BoardState]:
    """Reconstruct a BoardState from make_board_state_key output.  Returns None on failure."""
    try:
        # state_key format is opaque; we go via the trainer helper which knows
        # the canonical encoding.  Import lazily to avoid import cycles at
        # module load.
        from tools.train_value_net_v2 import board_from_state_key
        return board_from_state_key(state_key)
    except Exception:
        return None


# ── Extraction main ─────────────────────────────────────────────────────────

def run_extraction(
    games_dir:    Path,
    ledger_path:  Path,
    teacher_net:  Path,
    malom_db_dir: str,
    out_dir:      Path,
    min_empirical_support: int   = _DEFAULT_MIN_EMPIRICAL_SUPPORT,
    temperature:  float          = _DEFAULT_TEMPERATURE,
    coverage_floor_rows: int     = _DEFAULT_COVERAGE_FLOOR_ROWS,
    limit_files:  Optional[int]  = None,
    require_ready: bool          = True,
    force:        bool           = False,
    strict:       bool           = True,
    allow_partial_ledger: bool   = False,
) -> dict:
    """Run the Stage D v2 extraction end-to-end.  Returns provenance dict.

    Codex hardening 2026-08-14 (P1-B / P1-A pattern):
    - Ledger loaded via _verify_ledger_complete: partial / non-strict /
      malformed-tolerating ledgers are refused unless allow_partial_ledger.
    - Each scanned JSONL file's SHA-256 is verified against the ledger's
      file manifest.  Drift is fatal.  Missing files are fatal under a
      full run; skipped under limit_files (smoke).
    - Malformed JSON lines are fatal under strict=True (default).
    - No-clobber: refuse to write into an existing --out-dir unless
      force=True.  A pre-existing (but empty) directory is allowed.
    """
    t_wall_start = time.time()

    # ── No-clobber ─────────────────────────────────────────────────────────
    def _dir_has_output(d: Path) -> bool:
        if not d.exists():
            return False
        for name in ("parent_feats.f32.bin", "targets.f32.bin",
                     "targets_uniform.f32.bin", "targets_empirical.f32.bin",
                     "metadata.npz"):
            if (d / name).exists():
                return True
        return False
    if _dir_has_output(out_dir) and not force:
        raise RuntimeError(
            f"[extract v2] Refusing to overwrite existing outputs in "
            f"{out_dir}.  Delete them or pass force=True."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load session ledger (verified) ─────────────────────────────────────
    print(f"[extract v2] Loading session ledger: {ledger_path}")
    ledger_info  = _load_session_ledger(ledger_path, allow_partial=allow_partial_ledger)
    session_meta = ledger_info["session_meta"]
    ledger_file_shas = ledger_info["ledger_file_shas"]
    print(f"[extract v2]   sessions in ledger: {len(session_meta):,}   "
          f"tiers: {ledger_info['n_by_split']}   "
          f"files in manifest: {len(ledger_file_shas):,}")

    # ── Manifest set equality (Codex P1(2) 2026-08-14) ─────────────────────
    _verify_files_match_manifest(games_dir, ledger_file_shas, limit_files)

    # ── Teacher net + Malom ────────────────────────────────────────────────
    print(f"[extract v2] Loading teacher net: {teacher_net}")
    if not teacher_net.exists():
        raise FileNotFoundError(f"teacher net not found: {teacher_net}")
    # Codex P1(3) 2026-08-14: teacher provenance must reference THIS ledger.
    teacher_prov = _verify_teacher_lineage(teacher_net, ledger_path, ledger_info)
    print(f"[extract v2]   teacher lineage OK "
          f"(split_scheme={teacher_prov['dataset_split_scheme']!r}, "
          f"ledger_sha256={teacher_prov['session_ledger_sha256'][:16]}…)")
    advisor = HumanMovePolicyAdvisor(teacher_net, temperature=temperature)

    malom = MalomDB(malom_db_dir)
    if not malom.available:
        raise RuntimeError(f"Malom DB not available at {malom_db_dir}")

    # ── Pass 1: owning tier per state_key ──────────────────────────────────
    print(f"[extract v2] Pass 1 — owning tier assignment …")
    state_key_owning, pass1_stats = _scan_pass1_owning_tier(
        games_dir, session_meta, limit_files=limit_files,
        ledger_file_shas=ledger_file_shas, strict=strict,
    )
    print(f"[extract v2]   pass1 events={pass1_stats['n_events_total']:,}  "
          f"state_keys={pass1_stats['n_state_keys_seen']:,}  "
          f"({pass1_stats['elapsed_seconds']:.1f}s)")

    # ── Pass 2: owning-tier-only aggregation ───────────────────────────────
    print(f"[extract v2] Pass 2 — owning-tier-only aggregation …")
    counts, disposition = _scan_pass2_aggregate(
        games_dir, session_meta, state_key_owning, limit_files=limit_files,
        ledger_file_shas=ledger_file_shas, strict=strict,
    )
    print(f"[extract v2]   pass2 aggregated (state_key, band) pairs: {len(counts):,}")
    print(f"[extract v2]   events kept by tier: {disposition['events_kept_by_tier']}")
    print(f"[extract v2]   events discarded by tier: {disposition['events_discarded_by_tier']}")

    # ── Emit loop ──────────────────────────────────────────────────────────
    print(f"[extract v2] Building per-(state_key, band) targets …")

    feat_bin_path = out_dir / "parent_feats.f32.bin"
    tgt_bin_path  = out_dir / "targets.f32.bin"
    unif_bin_path = out_dir / "targets_uniform.f32.bin"
    emp_bin_path  = out_dir / "targets_empirical.f32.bin"
    abst_path     = out_dir / "abstained.jsonl"

    # We don't know the final N yet — first walk to compute; then allocate memmaps.
    # Simpler: buffer to lists (small memory since only 3-float rows), then dump at end.
    feats_rows:   list[np.ndarray] = []
    tgt_rows:     list[np.ndarray] = []
    unif_rows:    list[np.ndarray] = []
    emp_rows:     list[np.ndarray] = []
    state_keys_arr:  list[str]     = []
    band_idx_arr:    list[int]     = []
    split_arr:       list[int]     = []
    phase_arr:       list[str]     = []
    mover_arr:       list[str]     = []
    n_legal_arr:     list[int]     = []
    ph_source_arr:   list[str]     = []
    owning_hash_arr: list[str]     = []

    abstained_lines: list[str] = []
    counters = {
        "n_pairs_seen":                len(counts),
        "n_emitted":                   0,
        "n_abstained_bad_board":       0,
        "n_abstained_no_legal_moves":  0,
        "n_abstained_n_legal_lt2":     0,
        "n_abstained_parent_no_malom": 0,
        "n_abstained_not_emittable":   0,
        "n_hybrid_rows":               0,
        "n_model_only_rows":           0,
        # Codex P2 2026-08-14 self-review: per-(band, phase) counts of actually
        # emitted rows.  Populated during emit loop so it is meaningful on a
        # coverage-floor halt too (halt fires AFTER the emit loop runs).
        "emitted_by_band_phase":       {},
    }

    for (state_key, band), entry in counts.items():
        board = _board_from_state_key(state_key)
        if board is None:
            counters["n_abstained_bad_board"] += 1
            abstained_lines.append(json.dumps({
                "state_key": state_key, "band": band, "reason": "bad_state_key",
            }))
            continue

        legal_moves = get_all_legal_moves(board)
        if not legal_moves:
            counters["n_abstained_no_legal_moves"] += 1
            abstained_lines.append(json.dumps({
                "state_key": state_key, "band": band, "reason": "no_legal_moves",
            }))
            continue
        if len(legal_moves) < 2:
            counters["n_abstained_n_legal_lt2"] += 1
            abstained_lines.append(json.dumps({
                "state_key": state_key, "band": band, "reason": "n_legal_lt2",
            }))
            continue

        parent_omv = malom.query_value(board)
        if parent_omv is None:
            counters["n_abstained_parent_no_malom"] += 1
            abstained_lines.append(json.dumps({
                "state_key": state_key, "band": band, "reason": "parent_no_malom",
            }))
            continue

        regrets = {}
        for mv in legal_moves:
            regrets[_move_notation(mv)] = malom.query_regret(board, mv)

        emittable, reason = _check_emittable(regrets, legal_moves)
        if not emittable:
            counters["n_abstained_not_emittable"] += 1
            abstained_lines.append(json.dumps({
                "state_key": state_key, "band": band, "reason": reason,
            }))
            continue

        # Model P_h from teacher net.  Delegated to _compute_teacher_ph so the
        # abstain path is directly unit-testable (Codex 2026-08-15 P2).
        ph_model, teacher_reason = _compute_teacher_ph(
            advisor, board, legal_moves, band,
        )
        if teacher_reason is not None:
            counters["n_abstained_not_emittable"] += 1
            abstained_lines.append(json.dumps({
                "state_key": state_key, "band": band,
                "reason": teacher_reason,
            }))
            continue

        # Uniform P_h
        ph_uniform = _uniform_ph(len(legal_moves))

        # Empirical P_h — legal-only denominator (Codex P1 2026-08-14).
        ph_emp, illegal_events = _empirical_ph(
            entry["notation_counts"], legal_moves, min_empirical_support,
        )
        if illegal_events:
            counters["empirical_illegal_events_seen"] = (
                counters.get("empirical_illegal_events_seen", 0) + illegal_events
            )

        # G_v triples per source
        g_a_m, g_b_m, g_c_m = _compute_gv(ph_model,   legal_moves, regrets)
        g_a_u, g_b_u, g_c_u = _compute_gv(ph_uniform, legal_moves, regrets)
        if ph_emp is not None:
            g_a_e, g_b_e, g_c_e = _compute_gv(ph_emp, legal_moves, regrets)
            emp_row = np.array([g_a_e, g_b_e, g_c_e], dtype=np.float32)
            ph_src = "hybrid"
            counters["n_hybrid_rows"] += 1
        else:
            emp_row = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
            ph_src = "model"
            counters["n_model_only_rows"] += 1

        # Parent features from mover POV
        try:
            feats = board_to_features(board, board.turn)
        except Exception as e:
            counters["n_abstained_not_emittable"] += 1
            abstained_lines.append(json.dumps({
                "state_key": state_key, "band": band, "reason": f"features_failed:{e!s}",
            }))
            continue

        # Sanity: A/B/C rows must be finite (fail-closed at emit time too)
        tgt_row  = np.array([g_a_m, g_b_m, g_c_m], dtype=np.float32)
        unif_row = np.array([g_a_u, g_b_u, g_c_u], dtype=np.float32)
        if not np.isfinite(tgt_row).all() or not np.isfinite(unif_row).all():
            counters["n_abstained_not_emittable"] += 1
            abstained_lines.append(json.dumps({
                "state_key": state_key, "band": band, "reason": "non_finite_target",
            }))
            continue

        feats_rows.append(np.asarray(feats, dtype=np.float32))
        tgt_rows.append(tgt_row)
        unif_rows.append(unif_row)
        emp_rows.append(emp_row)
        state_keys_arr.append(state_key)
        band_idx_arr.append(_BAND_TO_IDX[band])
        split_arr.append(_SPLIT_TO_INT8[state_key_owning[state_key]["tier"]])
        phase_arr.append(entry["phase"])
        mover_arr.append(entry["mover_color"])
        n_legal_arr.append(len(legal_moves))
        ph_source_arr.append(ph_src)
        owning_hash_arr.append(state_key_owning[state_key]["session_hash"])
        counters["n_emitted"] += 1
        # Per-(band, phase) emitted-row counter (Codex P2 2026-08-14 self-review).
        bp_key = f"{band}|{entry['phase']}"
        counters["emitted_by_band_phase"][bp_key] = (
            counters["emitted_by_band_phase"].get(bp_key, 0) + 1
        )

    print(f"[extract v2] Emitted rows: {counters['n_emitted']:,}")

    # ── Coverage floor ─────────────────────────────────────────────────────
    if require_ready and counters["n_emitted"] < coverage_floor_rows:
        print(f"[extract v2] COVERAGE FLOOR NOT MET: "
              f"{counters['n_emitted']:,} < {coverage_floor_rows:,}. "
              f"Halting per checklist §Coverage floor.")
        # Emit provenance so we have a durable record of the failed run.
        provenance = _make_provenance(
            counters, disposition, ledger_info, pass1_stats,
            teacher_net, malom_db_dir, games_dir, out_dir,
            min_empirical_support, temperature,
            coverage_floor_rows, limit_files,
            gate_status="halt_coverage_floor",
            elapsed_wall=round(time.time() - t_wall_start, 1),
            require_ready=require_ready, strict=strict,
            allow_partial_ledger=allow_partial_ledger,
            teacher_prov=teacher_prov,
        )
        (out_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8",
        )
        return provenance

    # ── Write outputs ──────────────────────────────────────────────────────
    n_emitted = counters["n_emitted"]

    if n_emitted == 0:
        raise RuntimeError("No rows emitted — cannot write empty dataset.  Check inputs.")

    print(f"[extract v2] Writing binary output files …")
    np.stack(feats_rows).astype(np.float32).tofile(feat_bin_path)
    np.stack(tgt_rows).astype(np.float32).tofile(tgt_bin_path)
    np.stack(unif_rows).astype(np.float32).tofile(unif_bin_path)
    np.stack(emp_rows).astype(np.float32).tofile(emp_bin_path)

    if abstained_lines:
        with abst_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(abstained_lines) + "\n")

    metadata_path = out_dir / "metadata.npz"
    np.savez(
        metadata_path,
        state_keys=np.array(state_keys_arr,  dtype=object),
        band_idx=np.array(band_idx_arr,       dtype=np.int8),
        split=np.array(split_arr,             dtype=np.int8),
        phase=np.array(phase_arr,             dtype=object),
        mover_color=np.array(mover_arr,       dtype=object),
        n_legal=np.array(n_legal_arr,         dtype=np.int16),
        ph_source=np.array(ph_source_arr,     dtype=object),
        owning_session_min_hash=np.array(owning_hash_arr, dtype=object),
        provenance=np.array(json.dumps({}), dtype=object),   # rewritten below
    )

    provenance = _make_provenance(
        counters, disposition, ledger_info, pass1_stats,
        teacher_net, malom_db_dir, games_dir, out_dir,
        min_empirical_support, temperature,
        coverage_floor_rows, limit_files,
        gate_status="ok",
        elapsed_wall=round(time.time() - t_wall_start, 1),
        require_ready=require_ready, strict=strict,
        allow_partial_ledger=allow_partial_ledger,
        teacher_prov=teacher_prov,
    )
    # Re-save metadata with real provenance
    md = dict(np.load(metadata_path, allow_pickle=True))
    md["provenance"] = np.array(json.dumps(provenance), dtype=object)
    np.savez(metadata_path, **md)

    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8",
    )

    print(f"[extract v2] Done.  Emitted {n_emitted:,} rows to {out_dir}")
    return provenance


def _make_provenance(
    counters, disposition, ledger_info, pass1_stats,
    teacher_net: Path, malom_db_dir: str,
    games_dir: Path, out_dir: Path,
    min_empirical_support: int, temperature: float,
    coverage_floor_rows: int, limit_files: Optional[int],
    gate_status: str, elapsed_wall: float,
    require_ready: bool = True,
    strict: bool = True,
    allow_partial_ledger: bool = False,
    teacher_prov: Optional[dict] = None,
) -> dict:
    """Assemble the provenance dict for the emitted (or halt-reported) dataset.

    Codex P1(4) 2026-08-14: previously any run — including smoke, under-floor,
    or hardening-bypassed — could reach gate_status="ok" and be published as
    a normal Stage-E-ready output.  We now compute `production_ready` and a
    `non_ready_reasons` list so Stage E can hard-reject them.
    """
    n_emitted = counters["n_emitted"]
    coverage_floor_met = n_emitted >= coverage_floor_rows

    non_ready_reasons: list[str] = []
    if limit_files is not None:
        non_ready_reasons.append(f"limit_files={limit_files}")
    if not strict:
        non_ready_reasons.append("strict=False (tolerated malformed JSON)")
    if allow_partial_ledger:
        non_ready_reasons.append("allow_partial_ledger=True (partial ledger accepted)")
    if not require_ready:
        non_ready_reasons.append("require_ready=False (--no-coverage-gate)")
    if not coverage_floor_met:
        non_ready_reasons.append(
            f"coverage_floor not met (n_emitted={n_emitted} < "
            f"coverage_floor_rows={coverage_floor_rows})"
        )
    if gate_status != "ok":
        non_ready_reasons.append(f"gate_status={gate_status!r}")
    production_ready = (not non_ready_reasons)

    return {
        "extract_version":                  EXTRACT_VERSION,
        "gate_status":                      gate_status,
        # Codex P1(4) 2026-08-14: ready-flag surfaced for Stage E gate.
        "production_ready":                 production_ready,
        "non_ready_reasons":                non_ready_reasons,
        "coverage_floor_rows":              coverage_floor_rows,
        "coverage_floor_met":               coverage_floor_met,
        "extract_run_flags": {
            "require_ready":        require_ready,
            "strict":               strict,
            "allow_partial_ledger": allow_partial_ledger,
            "limit_files":          limit_files,
        },
        "extract_git_commit":               _git_commit(),
        "built_at":                         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds":                  elapsed_wall,
        "games_dir":                        str(games_dir),
        "out_dir":                          str(out_dir),
        "session_ledger":                   ledger_info["provenance"],
        "session_ledger_files_manifest_sha256":
            ledger_info["provenance"].get("files_manifest_sha256"),
        "teacher_net_path":                 str(teacher_net),
        "teacher_net_sha256":               _sha256_file(teacher_net),
        # Codex P1(3) 2026-08-14: teacher lineage recorded from teacher provenance.
        "teacher_provenance":               teacher_prov,
        "teacher_split_scheme":             (teacher_prov or {}).get("dataset_split_scheme"),
        "teacher_session_ledger_sha256":    (teacher_prov or {}).get("session_ledger_sha256"),
        "temperature":                      float(temperature),
        "malom_db_dir":                     str(malom_db_dir),
        "malom_label_version":              _MALOM_LABEL_VERSION,
        "regret_version":                   _REGRET_VERSION,
        "feature_dim":                      int(_INPUT_DIM),
        "n_heads":                          _N_HEADS,
        "components":                       ["class_downgrade", "wdl_utility_loss", "ordinal_rank_loss"],
        "elo_band_config_name":             OPTION_A_NAME,
        "min_empirical_support":            min_empirical_support,
        "limit_files":                      limit_files,
        "pass1_stats":                      pass1_stats,
        "session_split_disposition":        disposition,
        "extract_counters":                 counters,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def _load_malom_db_dir() -> str:
    p = _ROOT / "data" / "settings.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("malom_db_path"):
            return d["malom_db_path"]
    raise RuntimeError("Cannot find malom_db_path in data/settings.json; use --malom-db-dir")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--games-dir",       type=Path, default=_ROOT / "data" / "human_games")
    p.add_argument("--session-ledger",  type=Path, required=True,
                   help="Path to data/gap_v3_session_ledger.json (Batch 3a output).")
    p.add_argument("--teacher-net",     type=Path, required=True,
                   help="Path to HumanMovePolicyNet .npz used as the P_h teacher.  "
                        "Explicit choice required per user decision 2026-08-12.")
    p.add_argument("--malom-db-dir",    type=str, default=None,
                   help="Malom DB directory (default: from data/settings.json).")
    p.add_argument("--out-dir",         type=Path,
                   default=_ROOT / "data" / "gap_net_v3_dataset_v2")
    p.add_argument("--min-empirical-support", type=int,
                   default=_DEFAULT_MIN_EMPIRICAL_SUPPORT)
    p.add_argument("--temperature",     type=float, default=_DEFAULT_TEMPERATURE)
    p.add_argument("--coverage-floor-rows", type=int,
                   default=_DEFAULT_COVERAGE_FLOOR_ROWS,
                   help="Halt-and-report threshold (50 %% of v1's 2,550,799 rows).")
    p.add_argument("--limit-files",     type=int, default=None,
                   help="Cap number of JSONL files scanned (smoke test).")
    p.add_argument("--no-coverage-gate", action="store_true",
                   help="Emit output even if below coverage floor (for smoke tests only).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing --out-dir output files.  Codex hardening "
                        "2026-08-14: default refuses no-clobber.")
    p.add_argument("--allow-malformed", action="store_true",
                   help="Tolerate malformed JSON lines (strict=False).  Default fails "
                        "closed on the first malformed line.")
    p.add_argument("--allow-partial-ledger", action="store_true",
                   help="Accept a ledger built with --limit-files (is_partial=True).  "
                        "For smoke tests only.")
    args = p.parse_args()

    malom_db_dir = args.malom_db_dir or _load_malom_db_dir()

    prov = run_extraction(
        games_dir=args.games_dir,
        ledger_path=args.session_ledger,
        teacher_net=args.teacher_net,
        malom_db_dir=malom_db_dir,
        out_dir=args.out_dir,
        min_empirical_support=args.min_empirical_support,
        temperature=args.temperature,
        coverage_floor_rows=args.coverage_floor_rows,
        limit_files=args.limit_files,
        require_ready=not args.no_coverage_gate,
        force=args.force,
        strict=not args.allow_malformed,
        allow_partial_ledger=args.allow_partial_ledger,
    )
    if prov["gate_status"] == "halt_coverage_floor":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
