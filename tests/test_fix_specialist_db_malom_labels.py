from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fix_specialist_db_malom_labels.py"


def _create_legacy_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE positions (
            pos_hash TEXT PRIMARY KEY,
            wins INTEGER NOT NULL,
            draws INTEGER NOT NULL,
            losses INTEGER NOT NULL,
            malom_label TEXT,
            last_seen TEXT
        );
        CREATE TABLE winning_lines (id INTEGER PRIMARY KEY, payload TEXT);
        CREATE TABLE preferred_plays (id INTEGER PRIMARY KEY, payload TEXT);
        INSERT INTO positions VALUES ('one', 3, 2, 1, 'W', 'now');
        INSERT INTO positions VALUES ('two', 5, 4, 3, 'D', 'now');
        INSERT INTO winning_lines VALUES (1, 'line');
        INSERT INTO preferred_plays VALUES (1, 'play');
        """
    )
    connection.commit()
    connection.close()


def _label_count(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM positions WHERE malom_label IS NOT NULL"
            ).fetchone()[0]
        )
    finally:
        connection.close()


def test_migration_requires_explicit_paths_and_never_uses_cwd_default(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "specialist_db.sqlite"
    database.parent.mkdir()
    _create_legacy_db(database)

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert _label_count(database) == 2


def test_migration_writes_verified_copy_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "legacy.sqlite"
    output = tmp_path / "corrected.sqlite"
    _create_legacy_db(source)
    expected_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(source),
            "--output",
            str(output),
            "--expected-sha256",
            expected_sha256,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert _label_count(source) == 2
    assert _label_count(output) == 0

    connection = sqlite3.connect(output)
    try:
        version = connection.execute(
            "SELECT value FROM meta WHERE key = 'malom_label_version'"
        ).fetchone()
        assert version == ("sector-corrected-v1",)
        assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert connection.execute("SELECT COUNT(*) FROM positions").fetchone() == (2,)
        assert connection.execute("SELECT COUNT(*) FROM winning_lines").fetchone() == (
            1,
        )
        assert connection.execute("SELECT COUNT(*) FROM preferred_plays").fetchone() == (
            1,
        )
    finally:
        connection.close()


def test_migration_rejects_wrong_source_identity(tmp_path: Path) -> None:
    source = tmp_path / "legacy.sqlite"
    output = tmp_path / "corrected.sqlite"
    _create_legacy_db(source)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(source),
            "--output",
            str(output),
            "--expected-sha256",
            "0" * 64,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "SHA-256" in (result.stdout + result.stderr)
    assert not output.exists()
    assert _label_count(source) == 2
