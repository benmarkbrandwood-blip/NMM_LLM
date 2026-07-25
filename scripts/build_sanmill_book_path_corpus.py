#!/usr/bin/env python3
"""Build the complete, inventory-only eight-ply Sanmill book corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.sanmill_book_paths import (
    enumerate_complete_book_paths,
    freeze_book_path_corpus,
    load_book_path_corpus,
)
from learned_ai.evaluation.sanmill_data_query import SanmillDataQuerySession
from learned_ai.evaluation.sanmill_uci import inspect_sanmill_installation


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(_ROOT), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Git inspection failed: {detail}")
    return result.stdout.strip()


def _clean_generator_commit() -> str:
    top = Path(_git("rev-parse", "--show-toplevel")).resolve()
    if top != _ROOT.resolve():
        raise RuntimeError(f"unexpected Git top-level: {top}")
    dirty = _git("status", "--short", "--untracked-files=all")
    if dirty:
        raise RuntimeError(
            "book corpus generation requires a clean Git worktree:\n"
            f"{dirty}"
        )
    return _git("rev-parse", "HEAD")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=_ROOT / "data" / "training_paths.local.json",
        help="ignored local path registry containing sanmill_checkout",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            _ROOT
            / "docs"
            / "experiments"
            / "sanmill-book-path-corpus-v1.json"
        ),
    )
    args = parser.parse_args()

    generator_commit = _clean_generator_commit()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {_display_path(output)}")
    installation = inspect_sanmill_installation(args.paths_config)

    records = []
    corpora = []
    for _ in range(2):
        with SanmillDataQuerySession(installation) as session:
            corpus = enumerate_complete_book_paths(
                session,
                installation,
                generator_commit=generator_commit,
            )
        corpora.append(corpus)
        records.append(corpus.to_dict())
    if records[0] != records[1]:
        raise RuntimeError(
            "fresh Sanmill data-query processes produced different corpora"
        )
    if _git("status", "--short", "--untracked-files=all"):
        raise RuntimeError("Git worktree changed during corpus enumeration")

    freeze_book_path_corpus(output, corpora[0])
    loaded = load_book_path_corpus(output)
    if loaded.to_dict() != records[0]:
        raise RuntimeError("frozen book corpus failed strict round-trip")

    payload = loaded.to_dict()
    print(
        json.dumps(
            {
                "status": "built-inventory-only",
                "evaluation_policy_frozen": False,
                "cross_process_equal": True,
                "output": _display_path(output),
                "generator_commit": generator_commit,
                "corpus_identity": loaded.corpus_identity,
                "file_sha256": _sha256_file(output),
                "source_identity_sha256": loaded.source_identity[
                    "identity_sha256"
                ],
                "summary": payload["summary"],
                "depth_audit": payload["depth_audit"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
