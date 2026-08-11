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
- tools/extract_human_move_policy_dataset_v3.py (HMPN teacher retrain, Batch 3b)
- tools/extract_gap_v3_dataset.py (Stage D rebuild, Batch 4)

Both extractors MUST read splits from this ledger — neither may compute its own
split independently.  The ledger's files_manifest_sha256 must be recorded in
downstream artefacts' provenance so a mismatch is detectable.

Usage:
    .venv/bin/python tools/build_gap_v3_session_ledger.py
    .venv/bin/python tools/build_gap_v3_session_ledger.py --limit-files 500  # smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
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


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _session_hash(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def build(
    games_dir: Path,
    output: Path,
    limit_files: int | None = None,
) -> dict:
    """Scan games_dir, write ledger JSON to output.  Returns provenance dict."""
    if not games_dir.exists():
        raise FileNotFoundError(f"games_dir not found: {games_dir}")

    t0 = time.time()

    jsonl_files = sorted(games_dir.glob("*.jsonl"))
    if limit_files is not None:
        jsonl_files = jsonl_files[:limit_files]
    print(f"[session_ledger] Scanning {len(jsonl_files):,} JSONL files …")

    files_entries: list[dict] = []
    sessions_entries: list[dict] = []
    session_ids_seen: set[str] = set()
    n_sessions_from_stem = 0
    n_by_split = {"train": 0, "val": 0, "test": 0}

    for idx, fpath in enumerate(jsonl_files):
        stat = fpath.stat()
        sha  = _sha256_file(fpath)
        n_games_in_file = 0

        with fpath.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    rec = json.loads(raw_line)
                except Exception:
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

    manifest_str = json.dumps(
        [(e["rel_path"], e["sha256"], e["size_bytes"]) for e in files_entries],
        sort_keys=True,
    )
    files_manifest_sha256 = hashlib.sha256(manifest_str.encode()).hexdigest()

    provenance = {
        "ledger_version":         LEDGER_VERSION,
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
    with output.open("w", encoding="utf-8") as fh:
        json.dump(ledger, fh, indent=2, sort_keys=False)

    print(f"[session_ledger] Wrote → {output}")
    print(f"[session_ledger]  {len(jsonl_files):,} files, "
          f"{len(sessions_entries):,} sessions  "
          f"(train={n_by_split['train']:,}  val={n_by_split['val']:,}  "
          f"test={n_by_split['test']:,})")
    print(f"[session_ledger]  files_manifest_sha256={files_manifest_sha256}")
    print(f"[session_ledger]  elapsed={provenance['elapsed_seconds']}s")
    return provenance


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
                   help="Cap number of JSONL files scanned (smoke test).")
    args = p.parse_args()

    build(args.games_dir, args.output, limit_files=args.limit_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
