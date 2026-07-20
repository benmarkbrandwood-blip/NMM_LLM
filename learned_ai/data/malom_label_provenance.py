"""Version metadata for persisted labels produced by the Malom decoder.

The version identifies the projection from a Nine Men's Morris position to a
Malom sector/key.  Labels written before the sector-offset fix are not safe to
mix with labels produced by the corrected decoder, even though both use the
same ``W``/``D``/``L`` representation.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


MALOM_LABEL_VERSION_KEY = "malom_label_version"
CURRENT_MALOM_LABEL_VERSION = "sector-corrected-v1"


def read_malom_label_version(conn: sqlite3.Connection) -> Optional[str]:
    """Return the persisted label version, or ``None`` for legacy databases."""
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?",
            (MALOM_LABEL_VERSION_KEY,),
        ).fetchone()
    except sqlite3.Error:
        return None
    return str(row[0]) if row and row[0] else None


def write_current_malom_label_version(conn: sqlite3.Connection) -> None:
    """Mark labels in ``conn`` as produced by the current decoder semantics."""
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        (MALOM_LABEL_VERSION_KEY, CURRENT_MALOM_LABEL_VERSION),
    )


def human_db_has_malom_labels(conn: sqlite3.Connection) -> bool:
    """Return whether either HumanDB table contains at least one Malom label."""
    position = conn.execute(
        "SELECT 1 FROM positions WHERE malom_wdl IS NOT NULL LIMIT 1"
    ).fetchone()
    if position:
        return True
    move = conn.execute(
        "SELECT 1 FROM moves WHERE malom_wdl_after IS NOT NULL LIMIT 1"
    ).fetchone()
    return move is not None


def ensure_human_db_can_be_annotated(
    conn: sqlite3.Connection,
    db_path: Path | str,
) -> None:
    """Reject adding corrected labels to an unversioned, already-labelled DB."""
    version = read_malom_label_version(conn)
    if version == CURRENT_MALOM_LABEL_VERSION:
        return
    if not human_db_has_malom_labels(conn):
        return

    shown_version = version or "<missing>"
    raise RuntimeError(
        f"HumanDB {db_path} already contains Malom labels with version "
        f"{shown_version!r}; corrected labels require "
        f"{CURRENT_MALOM_LABEL_VERSION!r}. Re-run with --rebuild to replace "
        "all persisted labels, or use --no-malom to update only human-game "
        "statistics."
    )


def require_current_human_db_malom_labels(
    conn: sqlite3.Connection,
    db_path: Path | str,
) -> None:
    """Require a HumanDB whose Malom columns have current provenance."""
    version = read_malom_label_version(conn)
    if version == CURRENT_MALOM_LABEL_VERSION:
        return

    shown_version = version or "<missing>"
    raise RuntimeError(
        f"HumanDB {db_path} has Malom label version {shown_version!r}; "
        f"{CURRENT_MALOM_LABEL_VERSION!r} is required. Rebuild the database "
        "with the corrected Malom decoder before consuming its label columns."
    )
