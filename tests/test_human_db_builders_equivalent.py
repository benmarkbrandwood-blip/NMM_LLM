"""tests/test_human_db_builders_equivalent.py

Locks the invariant that `tools/build_human_db.py` and
`tools/build_human_db_sha.py` produce **content-equivalent** SQLite
databases from the same JSONL input.  Byte-identical would be too
strong (SQLite files carry rowid ordering, page state, WAL settings in
the header) — instead we hash `SELECT * FROM <table> ORDER BY <pk>` for
each user-visible table.

Also verifies that the `_sha` variant additionally emits the
`.sha256` sidecar file, which is the sole intended difference between
the two entry points.

Skipped if there is no fixture JSONL available.  The fixture path is
`data/human_games/*.jsonl` (a small `--limit-files N` slice); this
matches how the audit script runs — no separate checked-in fixture is
needed because the corpus is already on disk in every dev environment.

If your dev checkout does not carry `data/human_games/`, set env var
`NMM_HUMAN_DB_FIXTURE_DIR` to a directory containing at least one
JSONL file with valid records.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import sqlite3
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ── Fixture resolution ──────────────────────────────────────────────────────

def _resolve_fixture_dir() -> Path:
    if os.environ.get("NMM_HUMAN_DB_FIXTURE_DIR"):
        return Path(os.environ["NMM_HUMAN_DB_FIXTURE_DIR"])
    return _ROOT / "data" / "human_games"


_FIXTURE_DIR = _resolve_fixture_dir()
_FIXTURE_AVAILABLE = _FIXTURE_DIR.exists() and any(_FIXTURE_DIR.glob("*.jsonl"))
_LIMIT_FILES = 5    # small enough to run in seconds


# ── Content-equivalence hashing (advisor's guidance) ────────────────────────

_TABLES_TO_COMPARE = (
    # (table_name, ORDER BY clause) — all user-visible tables.
    ("positions",       "state_key"),
    ("moves",           "state_key, notation"),
    ("processed_files", "file_path"),
    ("meta",            "key"),
)


def _hash_table(conn: sqlite3.Connection, table: str, order_by: str) -> str:
    """Content-hash a table by concatenating stringified rows in PK order.

    We deliberately exclude non-content fields that legitimately differ
    between two runs — `meta.build_date` (timestamp) and
    `processed_files.mtime` (mtime bleeds through) are handled by
    _EXCLUDED_META_KEYS + special handling for processed_files below.
    """
    h = hashlib.sha256()
    if table == "meta":
        # Skip timestamp / build-time keys that are not content-derived.
        excluded = {"build_date"}
        rows = conn.execute(
            f"SELECT key, value FROM meta ORDER BY {order_by}"
        ).fetchall()
        for row in rows:
            if row[0] in excluded:
                continue
            h.update(repr(tuple(row)).encode())
            h.update(b"\n")
        return h.hexdigest()
    if table == "processed_files":
        # `mtime` is a stat of the source file; can differ subtly between
        # runs due to filesystem timestamp resolution.  Compare everything
        # else.
        rows = conn.execute(
            f"SELECT file_path, sha256, games_found FROM processed_files ORDER BY {order_by}"
        ).fetchall()
        for row in rows:
            h.update(repr(tuple(row)).encode())
            h.update(b"\n")
        return h.hexdigest()
    for row in conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}"):
        h.update(repr(tuple(row)).encode())
        h.update(b"\n")
    return h.hexdigest()


def _hash_full_db(db_path: Path) -> dict:
    """Return {table_name: content_hash} for every user table."""
    conn = sqlite3.connect(str(db_path))
    try:
        return {name: _hash_table(conn, name, ob) for name, ob in _TABLES_TO_COMPARE}
    finally:
        conn.close()


# ── Test runners ────────────────────────────────────────────────────────────

def _run_builder(script: str, output_db: Path, games_dir: Path) -> None:
    """Invoke a builder script as a subprocess with --no-malom (fast) and
    --limit-files, into a fresh DB."""
    if output_db.exists():
        output_db.unlink()
    sha_sidecar = output_db.with_suffix(output_db.suffix + ".sha256")
    if sha_sidecar.exists():
        sha_sidecar.unlink()
    cmd = [
        sys.executable, str(_ROOT / "tools" / script),
        "--games-dir", str(games_dir),
        "--output",    str(output_db.relative_to(_ROOT)) if output_db.is_relative_to(_ROOT) else str(output_db),
        "--no-malom",
        "--rebuild",
        "--limit-files", str(_LIMIT_FILES),
    ]
    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{script} failed (rc={proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


@unittest.skipUnless(_FIXTURE_AVAILABLE,
                     f"No JSONL fixture at {_FIXTURE_DIR}; set NMM_HUMAN_DB_FIXTURE_DIR")
class TestBuildersProduceEquivalentContent(unittest.TestCase):
    """End-to-end regression: run both builders on the same fixture slice
    and prove the resulting DBs match on user-visible content."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_root = _ROOT / "data" / "_test_human_db_equiv"
        cls.tmp_root.mkdir(parents=True, exist_ok=True)
        cls.db_plain = cls.tmp_root / "plain.sqlite"
        cls.db_sha   = cls.tmp_root / "sha.sqlite"
        _run_builder("build_human_db.py",     cls.db_plain, _FIXTURE_DIR)
        _run_builder("build_human_db_sha.py", cls.db_sha,   _FIXTURE_DIR)

    @classmethod
    def tearDownClass(cls):
        # Leave artefacts on disk so a failing run is easy to inspect;
        # subsequent runs overwrite via `--rebuild`.
        pass

    def test_positions_content_matches(self):
        h1 = _hash_full_db(self.db_plain)["positions"]
        h2 = _hash_full_db(self.db_sha)["positions"]
        self.assertEqual(h1, h2, "positions table diverged between builders")

    def test_moves_content_matches(self):
        h1 = _hash_full_db(self.db_plain)["moves"]
        h2 = _hash_full_db(self.db_sha)["moves"]
        self.assertEqual(h1, h2, "moves table diverged between builders")

    def test_processed_files_content_matches(self):
        h1 = _hash_full_db(self.db_plain)["processed_files"]
        h2 = _hash_full_db(self.db_sha)["processed_files"]
        self.assertEqual(h1, h2, "processed_files table diverged between builders")

    def test_meta_content_matches(self):
        h1 = _hash_full_db(self.db_plain)["meta"]
        h2 = _hash_full_db(self.db_sha)["meta"]
        self.assertEqual(h1, h2, "meta table diverged between builders")

    def test_row_counts_match(self):
        for db in (self.db_plain, self.db_sha):
            conn = sqlite3.connect(str(db))
            try:
                pos = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
                mv  = conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0]
            finally:
                conn.close()
            self.assertGreater(pos, 0, f"no positions in {db}")
            self.assertGreater(mv,  0, f"no moves in {db}")

    def test_schema_version_stamped(self):
        for db in (self.db_plain, self.db_sha):
            conn = sqlite3.connect(str(db))
            try:
                v = conn.execute(
                    "SELECT value FROM meta WHERE key = 'schema_version'"
                ).fetchone()
            finally:
                conn.close()
            self.assertIsNotNone(v, f"schema_version missing in {db}")
            self.assertEqual(v[0], "2", f"schema_version wrong in {db}")

    def test_sha_variant_produces_sidecar_and_plain_does_not(self):
        sidecar_plain = self.db_plain.with_suffix(self.db_plain.suffix + ".sha256")
        sidecar_sha   = self.db_sha.with_suffix(self.db_sha.suffix + ".sha256")
        self.assertFalse(
            sidecar_plain.exists(),
            f"plain builder should NOT emit sidecar but found {sidecar_plain}",
        )
        self.assertTrue(
            sidecar_sha.exists(),
            f"sha builder should emit sidecar but did not: {sidecar_sha}",
        )
        # sanity-check the sidecar format
        line = sidecar_sha.read_text(encoding="utf-8").strip()
        self.assertRegex(line, r"^[0-9a-f]{64}\s+" + self.db_sha.name + "$")


if __name__ == "__main__":
    unittest.main(verbosity=2)
