"""Create a corrected-label copy of a legacy SpecialistDB.

The source database is opened read-only and is never modified.  The caller
must name a new output path and provide the expected SHA-256 of the source so
that a stale path or the isolated legacy snapshot cannot be altered by
accident.  Empirical self-play tables are copied unchanged; only persisted
Malom labels are cleared and the output receives current decoder provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.data.malom_label_provenance import (  # noqa: E402
    CURRENT_MALOM_LABEL_VERSION,
    read_malom_label_version,
    write_current_malom_label_version,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_expected_sha256(value: str) -> str:
    normalised = value.strip().lower()
    if len(normalised) != 64 or any(c not in "0123456789abcdef" for c in normalised):
        raise argparse.ArgumentTypeError("expected SHA-256 must be 64 hex characters")
    return normalised


def _open_source_read_only(path: Path) -> sqlite3.Connection:
    uri = "file:" + path.resolve().as_posix() + "?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def migrate_copy(
    source: Path,
    output: Path,
    expected_sha256: str,
) -> dict[str, int | str | None]:
    """Copy ``source`` to ``output`` and clear only the copied Malom labels."""
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise RuntimeError("source and output must be different paths")
    if not source.is_file():
        raise RuntimeError(f"source database does not exist: {source}")
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise RuntimeError(f"output parent directory does not exist: {output.parent}")

    observed_sha256 = _sha256(source)
    if observed_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            "source SHA-256 mismatch: "
            f"expected {expected_sha256.lower()}, observed {observed_sha256.lower()}"
        )

    source_connection = _open_source_read_only(source)
    output_connection: sqlite3.Connection | None = None
    try:
        if source_connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise RuntimeError("source database failed SQLite quick_check")
        tables = {
            str(row[0])
            for row in source_connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "positions" not in tables:
            raise RuntimeError("source database has no positions table")

        existing_version = read_malom_label_version(source_connection)
        if existing_version == CURRENT_MALOM_LABEL_VERSION:
            raise RuntimeError(
                "source already has current Malom provenance; no migration is needed"
            )
        if existing_version is not None:
            raise RuntimeError(
                f"unsupported source Malom label version: {existing_version!r}"
            )

        positions_before = int(
            source_connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        )
        labels_before = int(
            source_connection.execute(
                "SELECT COUNT(*) FROM positions WHERE malom_label IS NOT NULL"
            ).fetchone()[0]
        )

        output_connection = sqlite3.connect(output)
        source_connection.backup(output_connection)
        output_connection.execute(
            "CREATE TABLE IF NOT EXISTS meta "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        with output_connection:
            output_connection.execute(
                "UPDATE positions SET malom_label = NULL "
                "WHERE malom_label IS NOT NULL"
            )
            output_connection.execute(
                "DELETE FROM meta WHERE key = 'malom_label_version'"
            )
            write_current_malom_label_version(output_connection)

        positions_after = int(
            output_connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        )
        labels_after = int(
            output_connection.execute(
                "SELECT COUNT(*) FROM positions WHERE malom_label IS NOT NULL"
            ).fetchone()[0]
        )
        if positions_after != positions_before:
            raise RuntimeError("position count changed during migration")
        if labels_after != 0:
            raise RuntimeError("output still contains Malom labels")
        if read_malom_label_version(output_connection) != CURRENT_MALOM_LABEL_VERSION:
            raise RuntimeError("output Malom label version was not written")
        if output_connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise RuntimeError("output database failed SQLite quick_check")

        return {
            "source_sha256": observed_sha256.lower(),
            "source_label_version": existing_version,
            "positions": positions_after,
            "labels_cleared": labels_before,
            "output_label_version": CURRENT_MALOM_LABEL_VERSION,
        }
    except Exception:
        if output_connection is not None:
            output_connection.close()
            output_connection = None
        if output.exists():
            output.unlink()
        raise
    finally:
        if output_connection is not None:
            output_connection.close()
        source_connection.close()


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new sector-corrected-v1 SpecialistDB copy while "
            "preserving the source database unchanged."
        )
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-sha256",
        required=True,
        type=_normalise_expected_sha256,
        help="Expected identity of the read-only source database",
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    try:
        report = migrate_copy(args.source, args.output, args.expected_sha256)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(
        "Created corrected copy: "
        f"positions={report['positions']}, "
        f"labels_cleared={report['labels_cleared']}, "
        f"version={report['output_label_version']}"
    )


if __name__ == "__main__":
    main()
