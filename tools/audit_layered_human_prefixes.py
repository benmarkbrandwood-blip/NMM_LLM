"""Create an immutable HumanDB snapshot and twelve-ply history audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.layered_human_audit import (
    audit_human_prefix_histories,
    build_layered_human_audit,
    create_human_db_snapshot,
    verify_layered_human_audit,
)
from learned_ai.training.run_contract import canonical_json_bytes


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths-config",
        default="data/training_paths.local.json",
    )
    parser.add_argument(
        "--games-dir",
        default="data/human_games",
    )
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--book-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    if _git("status", "--porcelain"):
        raise SystemExit("refusing to audit HumanDB from a dirty tree")
    commit = _git("rev-parse", "HEAD")
    config = json.loads(Path(args.paths_config).read_text(encoding="utf-8"))
    source_path = Path(config["human_db_path"])
    if not source_path.is_absolute():
        source_path = (ROOT / source_path).resolve()

    snapshot_directory = Path(args.snapshot_dir).resolve()
    if snapshot_directory.exists():
        raise SystemExit(f"refusing to reuse {snapshot_directory}")
    snapshot_directory.mkdir(parents=True)
    snapshot_path = snapshot_directory / "human_db.sqlite"
    sqlite_snapshot = create_human_db_snapshot(source_path, snapshot_path)

    source_audit = audit_human_prefix_histories(
        args.games_dir,
        manifest_path=snapshot_directory / "human-games-manifest.jsonl",
        ledger_path=snapshot_directory / "human-prefixes-12ply.jsonl",
        book_audit_path=args.book_audit,
        worker_count=args.workers,
    )
    imported_path = Path(args.games_dir) / "imported.json"
    imported = json.loads(imported_path.read_text(encoding="utf-8"))
    imported_manifest = {
        "path_lookup_key": "human_games_imported_manifest_path",
        "entry_count": len(imported),
        "byte_length": imported_path.stat().st_size,
        "sha256": _sha256(imported_path),
    }
    audit = build_layered_human_audit(
        generator_commit=commit,
        sqlite_snapshot=sqlite_snapshot,
        source_audit=source_audit,
        imported_manifest=imported_manifest,
    )
    summary = verify_layered_human_audit(audit)

    local_manifest = snapshot_directory / "snapshot-manifest.json"
    local_manifest.write_bytes(canonical_json_bytes(audit) + b"\n")
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(audit) + b"\n")
    print(f"wrote {output}")
    print(f"snapshot={snapshot_path}")
    print(f"local_manifest={local_manifest}")
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
