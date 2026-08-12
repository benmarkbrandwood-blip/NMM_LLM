#!/usr/bin/env python3
"""tools/build_gap_v3_session_ledger.py — frozen session ledger for GapNet v3 rebuild.

Compiled from data/human_games/*.jsonl. Records, deterministically:
- Every JSONL file's SHA-256, size, mtime, and game count.
- Every session_id present in those files, with:
    session_hash   = sha256(session_id) — used for the owning-tier tie-break rule
    split          = game_level_split(session_id) → train|val|test
    source_file    = which JSONL contained the first occurrence
    session_source = "record" | "file_stem" (fallback when no session_id in JSON)

Consumers (Batch 3 and Batch 4 of gap_net_v3_stage_e_rebuild_checklist.md):
- tools/extract_human_move_policy_dataset.py (HMPN teacher retrain, Batch 3b)
- tools/extract_gap_v3_dataset_v2.py (Stage D rebuild, Batch 4)

Both extractors MUST read splits from this ledger — neither may compute its own
split independently.  The ledger's files_manifest_sha256 must be recorded in
downstream artefacts' provenance so a mismatch is detectable.

Codex P1-B hardening (2026-08-12): production-safe build
- strict=True (default) fails-closed on malformed JSON.
- Empty corpus (no JSONL) and empty ledger (0 sessions) fail-closed.
- No-clobber: existing output refused unless --force.
- Atomic publish: write to <output>.tmp, fsync, atomic rename.
- Single-pass file read: SHA-256 and JSONL parsing consume the same bytes.
- --limit-files marks the ledger is_partial=True; _verify_ledger_complete
  rejects partial ledgers for production consumers.

Usage:
    .venv/bin/python tools/build_gap_v3_session_ledger.py
    .venv/bin/python tools/build_gap_v3_session_ledger.py --limit-files 500  # smoke
    .venv/bin/python tools/build_gap_v3_session_ledger.py --force            # re-run
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

from learned_ai.data.human_db_split import (               # noqa: E402
    MANIFEST_VERSION as _SPLIT_VERSION,
    game_level_split,
)

LEDGER_VERSION = "gap_v3_session_ledger_v1"


class LedgerBuildError(Exception):
    """Raised when the ledger builder refuses to write under P1-B fail-closed rules."""


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _read_file_bytes_and_sha(path: Path) -> tuple[bytes, str]:
    """Single-pass read: return (raw_bytes, sha256_hex).

    Codex P1-B (2026-08-12): the previous implementation opened each file
    twice — once for SHA and once for JSONL parsing — so hash and consumed
    bytes could drift if the file changed between opens.  We now read once
    and compute both from the same bytes.
    """
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def _session_hash(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def build(
    games_dir: Path,
    output: Path,
    limit_files: int | None = None,
    strict: bool = True,
    force: bool = False,
) -> dict:
    """Scan games_dir, write ledger JSON to output.  Returns provenance dict.

    Codex P1-B hardening (2026-08-12):
    - strict=True (default): fail-closed on malformed JSON lines.  Set
      strict=False to tolerate and count them (recorded in provenance).
    - No-clobber: refuse to overwrite existing output unless force=True.
    - Atomic publish: write to `<output>.tmp`, fsync, rename.
    - Empty-corpus fail-closed: zero JSONL files → LedgerBuildError.
    - Empty-ledger fail-closed: zero valid sessions → LedgerBuildError.
    - `limit_files is not None` marks the ledger as `is_partial=True`;
      downstream consumers refuse partial ledgers via
      `_verify_ledger_complete()`.
    - Single-pass file read: SHA-256 and JSONL parsing consume the same
      bytes (no read/hash drift).
    """
    if not games_dir.exists():
        raise FileNotFoundError(f"games_dir not found: {games_dir}")

    if output.exists() and not force:
        raise LedgerBuildError(
            f"[session_ledger] Refusing to overwrite existing output: {output}.  "
            f"Delete the file or pass --force to overwrite."
        )

    t0 = time.time()

    jsonl_files = sorted(games_dir.glob("*.jsonl"))
    is_partial = limit_files is not None
    if limit_files is not None:
        jsonl_files = jsonl_files[:limit_files]
    print(f"[session_ledger] Scanning {len(jsonl_files):,} JSONL files "
          f"(strict={strict}, is_partial={is_partial}) …")

    if not jsonl_files:
        raise LedgerBuildError(
            f"[session_ledger] No *.jsonl files found in {games_dir}.  "
            f"Refusing to write empty ledger (Codex P1-B fail-closed)."
        )

    files_entries: list[dict] = []
    sessions_entries: list[dict] = []
    session_ids_seen: set[str] = set()
    n_sessions_from_stem = 0
    n_malformed_lines    = 0
    n_by_split = {"train": 0, "val": 0, "test": 0}

    for idx, fpath in enumerate(jsonl_files):
        stat = fpath.stat()
        raw_bytes, sha = _read_file_bytes_and_sha(fpath)
        n_games_in_file = 0

        for raw_line_b in raw_bytes.splitlines():
            try:
                raw = raw_line_b.decode("utf-8").strip()
            except UnicodeDecodeError:
                n_malformed_lines += 1
                if strict:
                    raise LedgerBuildError(
                        f"[session_ledger] Non-UTF-8 bytes in {fpath}: refusing "
                        f"to continue under strict mode (Codex P1-B)."
                    )
                continue
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception as e:
                n_malformed_lines += 1
                if strict:
                    raise LedgerBuildError(
                        f"[session_ledger] Malformed JSON in {fpath}: {e!s}.  "
                        f"Refusing to continue under strict mode (Codex P1-B)."
                    )
                continue
            n_games_in_file += 1

            sid_raw = rec.get("session_id")
            if sid_raw:
                session_id = str(sid_raw)
                source = "record"
            else:
                session_id = fpath.stem
                source = "file_stem"
                n_sessions_from_stem += 1

            if session_id in session_ids_seen:
                # First occurrence wins.  Sorted file iteration makes this deterministic.
                continue
            session_ids_seen.add(session_id)

            split = game_level_split(session_id)
            n_by_split[split] += 1
            sessions_entries.append({
                "session_id":     session_id,
                "session_hash":   _session_hash(session_id),
                "split":          split,
                "source_file":    fpath.name,
                "session_source": source,
            })

        files_entries.append({
            "rel_path":   fpath.relative_to(games_dir).as_posix(),
            "sha256":     sha,
            "size_bytes": stat.st_size,
            "mtime":      stat.st_mtime,
            "n_games":    n_games_in_file,
        })

        if (idx + 1) % 5000 == 0:
            print(f"[session_ledger]  … {idx+1:,}/{len(jsonl_files):,} files "
                  f"(sessions={len(sessions_entries):,}  t={time.time()-t0:.0f}s)")

    if not sessions_entries:
        raise LedgerBuildError(
            f"[session_ledger] No valid sessions found across "
            f"{len(jsonl_files):,} JSONL files.  Refusing to write empty ledger "
            f"(Codex P1-B fail-closed)."
        )

    manifest_str = json.dumps(
        [(e["rel_path"], e["sha256"], e["size_bytes"]) for e in files_entries],
        sort_keys=True,
    )
    files_manifest_sha256 = hashlib.sha256(manifest_str.encode()).hexdigest()

    provenance = {
        "ledger_version":         LEDGER_VERSION,
        "is_partial":             is_partial,
        "limit_files_arg":        limit_files,
        "strict":                 strict,
        "n_malformed_lines":      n_malformed_lines,
        "split_function":         "learned_ai.data.human_db_split.game_level_split",
        "split_manifest_version": _SPLIT_VERSION,
        "git_commit":             _git_commit(),
        "built_at":               time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "games_dir":              str(games_dir),
        "n_jsonl_files":          len(jsonl_files),
        "n_sessions":             len(sessions_entries),
        "n_sessions_from_stem":   n_sessions_from_stem,
        "n_by_split":             n_by_split,
        "files_manifest_sha256":  files_manifest_sha256,
        "elapsed_seconds":        round(time.time() - t0, 1),
    }

    ledger = {
        **provenance,
        "files":    files_entries,
        "sessions": sessions_entries,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publish (Codex P1-B, 2026-08-12): write to sibling .tmp, fsync,
    # then atomic rename.  Interrupts leave a .tmp behind, never a partially
    # written canonical output.
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(ledger, fh, indent=2, sort_keys=False)
        fh.flush()
        os.fsync(fh.fileno())
    tmp_path.replace(output)

    print(f"[session_ledger] Wrote → {output}")
    print(f"[session_ledger]  {len(jsonl_files):,} files, "
          f"{len(sessions_entries):,} sessions  "
          f"(train={n_by_split['train']:,}  val={n_by_split['val']:,}  "
          f"test={n_by_split['test']:,})")
    if is_partial:
        print(f"[session_ledger]  ⚠️  is_partial=True (--limit-files={limit_files}); "
              f"downstream consumers refuse this ledger for production runs.")
    if n_malformed_lines:
        print(f"[session_ledger]  ⚠️  n_malformed_lines={n_malformed_lines} "
              f"(tolerated under strict=False)")
    print(f"[session_ledger]  files_manifest_sha256={files_manifest_sha256}")
    print(f"[session_ledger]  elapsed={provenance['elapsed_seconds']}s")
    return provenance


def _verify_ledger_complete(ledger_path: Path, allow_partial: bool = False) -> dict:
    """Load a ledger and verify it is production-safe.

    Downstream consumers (HMPN v3 extractor, GapNet Stage D extractor) call
    this before proceeding.  Codex P1-B fail-closed enforcement (2026-08-12).

    Raises LedgerBuildError if:
      - Ledger is partial (is_partial == True) and allow_partial is False.
      - Ledger was built in non-strict mode.
      - Ledger tolerated malformed lines.
      - Ledger is empty (n_sessions == 0).

    Returns the loaded ledger dict on success.
    """
    with ledger_path.open("r", encoding="utf-8") as f:
        ledger = json.load(f)
    if ledger.get("is_partial") and not allow_partial:
        raise LedgerBuildError(
            f"Ledger at {ledger_path} is partial "
            f"(built with --limit-files={ledger.get('limit_files_arg')}).  "
            f"Refusing to use for production runs.  Pass allow_partial=True "
            f"to override for smoke tests only."
        )
    if not ledger.get("strict", True):
        raise LedgerBuildError(
            f"Ledger at {ledger_path} was built with strict=False.  "
            f"Refusing to use for production runs."
        )
    if ledger.get("n_malformed_lines", 0) > 0:
        raise LedgerBuildError(
            f"Ledger at {ledger_path} tolerated "
            f"{ledger['n_malformed_lines']} malformed JSON lines.  "
            f"Refusing to use for production runs."
        )
    if ledger.get("n_sessions", 0) == 0:
        raise LedgerBuildError(
            f"Ledger at {ledger_path} contains zero sessions.  "
            f"Refusing to use."
        )
    return ledger


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--games-dir",   type=Path,
                   default=_ROOT / "data" / "human_games")
    p.add_argument("--output",      type=Path,
                   default=_ROOT / "data" / "gap_v3_session_ledger.json")
    p.add_argument("--limit-files", type=int, default=None,
                   help="Cap number of JSONL files scanned (smoke test).  "
                        "Sets is_partial=True in the ledger; production "
                        "consumers refuse partial ledgers.")
    p.add_argument("--allow-malformed", action="store_true",
                   help="Tolerate malformed JSON lines and record the count "
                        "in provenance (strict=False).  Production consumers "
                        "refuse ledgers built this way.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing --output.  Default refuses "
                        "no-clobber.")
    args = p.parse_args()

    build(
        args.games_dir, args.output,
        limit_files=args.limit_files,
        strict=not args.allow_malformed,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
