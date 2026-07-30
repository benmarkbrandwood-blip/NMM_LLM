#!/usr/bin/env python3
"""tools/validate_human_db_candidate.py — Phase 2 candidate DB validator.

Reviewer requirement (docs/human_move_policy_net_plan.md §2.2): before a
Phase-2 candidate HumanDB may be considered for activation, it must
pass:

    - SQLite `PRAGMA quick_check`
    - Row-count reconciliation: `sum(positions_elo_bins.total_games)`
      per state_key equals `positions.total_games` for every state_key
      that carries Elo-tagged events.
    - Move-count reconciliation: `sum(moves_elo_bins.total)` per
      (state_key, notation) equals `moves.total` for every
      Elo-tagged (state_key, notation).
    - Provenance meta present: schema_version=3, elo_bin_size,
      feature_canonicalisation_version, source_manifest_sha256,
      builder_git_commit, built_at.
    - Semantic probe: for the new-game starting position, Malom (if
      the DB was built with annotation) returns `outcome='D'` per
      the golden vector locked in tests/test_malom_db.py.

Emits a signed report at `<candidate>.validation.json` containing the
candidate's SHA-256, every provenance row, every reconciliation
outcome, and a top-level `ok: true/false` flag.

Activation of the candidate over `data/human_db.sqlite` is a
**separate later decision** — this script never renames or moves any
DB file.

Usage
-----
    .venv/bin/python tools/validate_human_db_candidate.py \\
        --candidate data/human_db_candidate.sqlite
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

REQUIRED_META = (
    "schema_version",
    "elo_bin_size",
    "feature_canonicalisation_version",
    "source_manifest_sha256",
    "builder_git_commit",
    "built_at",
)

REQUIRED_TABLES = (
    "positions",
    "moves",
    "positions_elo_bins",
    "moves_elo_bins",
    "processed_files",
    "meta",
)


def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _quick_check(conn: sqlite3.Connection) -> str:
    row = conn.execute("PRAGMA quick_check").fetchone()
    return row[0] if row else "<no rows>"


def _tables_present(conn: sqlite3.Connection) -> list[str]:
    existing = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    return [t for t in REQUIRED_TABLES if t not in existing]


def _meta_rows(conn: sqlite3.Connection) -> dict:
    return {r[0]: r[1] for r in conn.execute("SELECT key, value FROM meta")}


def _reconcile_positions(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """
        WITH bin_totals AS (
            SELECT state_key, SUM(total_games) AS s
            FROM positions_elo_bins GROUP BY state_key
        )
        SELECT COUNT(*)
        FROM positions p JOIN bin_totals bt USING (state_key)
        WHERE p.total_games != bt.s
        """
    ).fetchone()
    mismatches = int(row[0]) if row else -1
    sample = conn.execute(
        """
        WITH bin_totals AS (
            SELECT state_key, SUM(total_games) AS s
            FROM positions_elo_bins GROUP BY state_key
        )
        SELECT p.state_key, p.total_games, bt.s
        FROM positions p JOIN bin_totals bt USING (state_key)
        WHERE p.total_games != bt.s
        LIMIT 5
        """
    ).fetchall()
    return {"mismatch_count": mismatches, "sample": sample}


def _reconcile_moves(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """
        WITH bin_totals AS (
            SELECT state_key, notation, SUM(total) AS s
            FROM moves_elo_bins GROUP BY state_key, notation
        )
        SELECT COUNT(*)
        FROM moves m JOIN bin_totals bt USING (state_key, notation)
        WHERE m.total != bt.s
        """
    ).fetchone()
    mismatches = int(row[0]) if row else -1
    sample = conn.execute(
        """
        WITH bin_totals AS (
            SELECT state_key, notation, SUM(total) AS s
            FROM moves_elo_bins GROUP BY state_key, notation
        )
        SELECT m.state_key, m.notation, m.total, bt.s
        FROM moves m JOIN bin_totals bt USING (state_key, notation)
        WHERE m.total != bt.s
        LIMIT 5
        """
    ).fetchall()
    return {"mismatch_count": mismatches, "sample": sample}


def _semantic_probe_starting_position(conn: sqlite3.Connection) -> dict:
    """Assert Malom's parent label on the new-game position is 'D'.
    Golden vector locked in tests/test_malom_db.py::test_empty_board.
    Only meaningful for DBs built with Malom annotation; on a --no-malom
    build the row will simply be missing.
    """
    try:
        from ai.trajectory_db import make_board_state_key
        from game.board import BoardState
        board = BoardState.new_game()
        state_key, _ = make_board_state_key(board)
    except Exception as e:
        return {"ok": False, "reason": f"could not construct state_key: {e}"}
    row = conn.execute(
        "SELECT malom_wdl FROM positions WHERE state_key = ?", (state_key,)
    ).fetchone()
    if row is None:
        return {"ok": None, "reason": "starting position absent from DB (--no-malom or corpus lacked)"}
    if row[0] is None:
        return {"ok": None, "reason": "starting position has no Malom label (--no-malom build)"}
    ok = row[0] == "D"
    return {
        "ok":      ok,
        "state_key": state_key,
        "expected": "D",
        "actual":   row[0],
    }


def validate(candidate: Path) -> dict:
    if not candidate.exists():
        return {"ok": False, "reason": f"candidate not found at {candidate}"}
    t0 = time.time()

    conn = sqlite3.connect(str(candidate))
    conn.execute("PRAGMA cache_size = -65536")
    try:
        quick = _quick_check(conn)
        missing_tables = _tables_present(conn)
        meta = _meta_rows(conn)
        missing_meta = [k for k in REQUIRED_META if k not in meta or meta[k] in (None, "")]
        row_counts = {
            t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
            for t in REQUIRED_TABLES if t not in missing_tables
        }
        pos_reconcile  = _reconcile_positions(conn) if "positions_elo_bins" not in missing_tables else {"mismatch_count": -1}
        move_reconcile = _reconcile_moves(conn)     if "moves_elo_bins"     not in missing_tables else {"mismatch_count": -1}
        probe = _semantic_probe_starting_position(conn)
    finally:
        conn.close()

    schema_version = meta.get("schema_version")
    ok = (
        quick == "ok"
        and not missing_tables
        and not missing_meta
        and schema_version == "3"
        and pos_reconcile.get("mismatch_count") == 0
        and move_reconcile.get("mismatch_count") == 0
        # probe.ok may be None (no Malom) — treat as pass-with-warning
        and (probe.get("ok") is not False)
    )
    return {
        "ok":                    ok,
        "candidate_path":        str(candidate),
        "candidate_sha256":      _sha256(candidate),
        "candidate_size_bytes":  candidate.stat().st_size,
        "elapsed_seconds":       round(time.time() - t0, 1),
        "quick_check":           quick,
        "schema_version":        schema_version,
        "missing_tables":        missing_tables,
        "missing_meta_rows":     missing_meta,
        "meta":                  meta,
        "row_counts":            row_counts,
        "positions_reconcile":   pos_reconcile,
        "moves_reconcile":       move_reconcile,
        "semantic_probe":        probe,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--candidate", type=Path, required=True,
                   help="Candidate HumanDB file to validate.")
    p.add_argument("--report", type=Path, default=None,
                   help="Where to write the JSON report.  Defaults to "
                        "<candidate>.validation.json.")
    args = p.parse_args()

    result = validate(args.candidate)
    report_path = args.report or args.candidate.with_suffix(args.candidate.suffix + ".validation.json")
    report_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(f"[validate] Report written → {report_path}")
    print(f"[validate] ok:                {result['ok']}")
    print(f"[validate] schema_version:    {result.get('schema_version')}")
    print(f"[validate] quick_check:       {result.get('quick_check')}")
    print(f"[validate] candidate_sha256:  {result.get('candidate_sha256')}")
    print(f"[validate] missing_tables:    {result.get('missing_tables')}")
    print(f"[validate] missing_meta_rows: {result.get('missing_meta_rows')}")
    print(f"[validate] positions bins mismatches: {result['positions_reconcile'].get('mismatch_count')}")
    print(f"[validate] moves     bins mismatches: {result['moves_reconcile'].get('mismatch_count')}")
    print(f"[validate] semantic probe:    {result.get('semantic_probe')}")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
