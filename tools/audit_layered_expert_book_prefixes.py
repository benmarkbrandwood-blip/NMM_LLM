"""Freeze the maintainer expert-Book twelve-ply source audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.layered_expert_book_audit import (
    build_layered_expert_book_audit,
    load_expert_book_source,
    load_expert_source_overlap_index,
    prepare_expert_book_candidates,
    verify_layered_expert_book_audit,
)
from learned_ai.evaluation.sanmill_data_query import SanmillDataQuerySession
from learned_ai.evaluation.sanmill_uci import inspect_sanmill_installation
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


def _resolve_config_path(config: Path, key: str) -> Path:
    payload = json.loads(config.read_text(encoding="utf-8"))
    raw = payload.get(key)
    if not isinstance(raw, str) or not raw:
        raise SystemExit(f"paths config lacks {key}")
    path = Path(raw)
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths-config",
        default="data/training_paths.local.json",
    )
    parser.add_argument(
        "--source",
        default=(
            "docs/evidence/"
            "maintainer-book-opening-plays-source-2026-07-26.json"
        ),
    )
    parser.add_argument(
        "--book-audit",
        default=(
            "docs/evidence/"
            "sanmill-layered-book-source-audit-2026-07-25.json"
        ),
    )
    parser.add_argument(
        "--human-audit",
        default=(
            "docs/evidence/"
            "sanmill-layered-human-source-audit-2026-07-25.json"
        ),
    )
    parser.add_argument(
        "--perfect-audit",
        default=(
            "docs/evidence/"
            "sanmill-layered-perfect-source-audit-2026-07-25.json"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if _git("status", "--porcelain"):
        raise SystemExit(
            "refusing to freeze expert Book evidence from a dirty tree"
        )
    commit = _git("rev-parse", "HEAD")
    config = Path(args.paths_config)
    source = load_expert_book_source(args.source)
    candidates = prepare_expert_book_candidates(source)
    overlap = load_expert_source_overlap_index(
        book_audit_path=args.book_audit,
        human_audit_path=args.human_audit,
        human_ledger_path=_resolve_config_path(
            config,
            "human_db_prefix12_history_ledger_path",
        ),
        perfect_audit_path=args.perfect_audit,
        expert_exact_history_ids=[
            candidate.exact_history_sha256 for candidate in candidates
        ],
    )
    installation = inspect_sanmill_installation(args.paths_config)
    audits = []
    for _ in range(2):
        with SanmillDataQuerySession(installation) as session:
            audits.append(
                build_layered_expert_book_audit(
                    session,
                    installation,
                    source=source,
                    candidates=candidates,
                    overlap=overlap,
                    generator_commit=commit,
                    fresh_processes=2,
                )
            )
    encoded = [canonical_json_bytes(audit) for audit in audits]
    if encoded[0] != encoded[1]:
        raise SystemExit(
            "fresh expert Book audit processes produced different bytes"
        )
    summary = verify_layered_expert_book_audit(audits[0])

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded[0] + b"\n")
    print(f"wrote {output}")
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
