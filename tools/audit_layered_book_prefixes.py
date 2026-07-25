"""Freeze the deterministic twelve-ply Book source audit."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.layered_book_audit import (
    build_layered_book_audit,
    verify_layered_book_audit,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths-config",
        default="data/training_paths.local.json",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if _git("status", "--porcelain"):
        raise SystemExit("refusing to freeze Book evidence from a dirty tree")
    commit = _git("rev-parse", "HEAD")
    installation = inspect_sanmill_installation(args.paths_config)
    audits = []
    for _ in range(2):
        with SanmillDataQuerySession(installation) as session:
            audits.append(
                build_layered_book_audit(
                    session,
                    installation,
                    generator_commit=commit,
                    fresh_processes=2,
                )
            )
    encoded = [canonical_json_bytes(audit) for audit in audits]
    if encoded[0] != encoded[1]:
        raise SystemExit("fresh Book audit processes produced different bytes")
    summary = verify_layered_book_audit(audits[0])

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
