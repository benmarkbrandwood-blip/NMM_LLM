#!/usr/bin/env python3
"""Run the read-only F0-D0 human-game reconstructability audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.human_raw_reconstructability import (  # noqa: E402
    F0D0AuditError,
    build_f0d0_manifest,
    write_manifest,
)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _preflight(source_commit: str, output: Path) -> None:
    if _git("rev-parse", "--show-toplevel").replace("\\", "/") != str(
        ROOT
    ).replace("\\", "/"):
        raise F0D0AuditError("repository root differs")
    if _git("branch", "--show-current") != "dev":
        raise F0D0AuditError("F0-D0 audit requires branch dev")
    head = _git("rev-parse", "HEAD")
    if head != source_commit:
        raise F0D0AuditError("source commit differs")
    if _git("rev-parse", "origin/dev") != head:
        raise F0D0AuditError("dev and origin/dev differ")
    if _git("status", "--porcelain"):
        raise F0D0AuditError("worktree is not clean before evidence generation")
    if output.exists():
        raise F0D0AuditError("manifest output already exists")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--games-dir", default="data/human_games")
    parser.add_argument(
        "--imported-manifest",
        default="data/human_games/imported.json",
    )
    parser.add_argument("--active-human-db", default="data/human_db.sqlite")
    parser.add_argument(
        "--archived-human-db",
        default="data/backups/maintainer_upload_20260721/human_db.sqlite",
    )
    parser.add_argument(
        "--ruleset",
        default="data/rulesets/nmm-training-core@2.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    output = (ROOT / args.output).resolve()

    def report_progress(completed: int, total: int) -> None:
        print(
            json.dumps(
                {
                    "status": "auditing_raw_games",
                    "completed": completed,
                    "total": total,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    try:
        _preflight(args.source_commit, output)
        payload = build_f0d0_manifest(
            repository_root=ROOT,
            games_directory=ROOT / args.games_dir,
            imported_manifest_path=ROOT / args.imported_manifest,
            active_human_db_path=ROOT / args.active_human_db,
            archived_human_db_path=ROOT / args.archived_human_db,
            ruleset_path=ROOT / args.ruleset,
            source_commit=args.source_commit,
            worker_count=args.workers,
            progress=report_progress,
        )
        write_manifest(payload, output)
    except (F0D0AuditError, OSError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "fatal_stop", "error": str(exc)}))
        return 1

    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "manifest_identity": payload["manifest_identity"],
                "corpus_identity": payload["identities"]["corpus_identity"],
                "counts": payload["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
