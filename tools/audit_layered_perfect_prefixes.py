"""Freeze the deterministic twelve-ply Perfect DB source audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.layered_perfect_audit import (
    PERFECT_AUDIT_BASE_SEED,
    PERFECT_AUDIT_ROUTE_COUNT,
    build_layered_perfect_audit,
    load_source_overlap_index,
    verify_layered_perfect_audit,
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


def _resolve_config_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{field} is absent from the local path registry")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths-config",
        default="data/training_paths.local.json",
    )
    parser.add_argument("--book-audit", required=True)
    parser.add_argument("--human-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--routes",
        type=int,
        default=PERFECT_AUDIT_ROUTE_COUNT,
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=PERFECT_AUDIT_BASE_SEED,
    )
    args = parser.parse_args()

    if _git("status", "--porcelain"):
        raise SystemExit(
            "refusing to freeze Perfect DB evidence from a dirty tree"
        )
    commit = _git("rev-parse", "HEAD")
    config_path = Path(args.paths_config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    database = _resolve_config_path(
        config.get("malom_db_path"),
        field="malom_db_path",
    )
    ledger = _resolve_config_path(
        config.get("human_db_prefix12_history_ledger_path"),
        field="human_db_prefix12_history_ledger_path",
    )
    installation = inspect_sanmill_installation(config_path)
    overlap = load_source_overlap_index(
        book_audit_path=args.book_audit,
        human_audit_path=args.human_audit,
        human_ledger_path=ledger,
    )

    audits = []
    for _ in range(2):
        with SanmillDataQuerySession(installation, timeout=300.0) as session:
            audits.append(
                build_layered_perfect_audit(
                    session,
                    installation,
                    database_path=database,
                    generator_commit=commit,
                    overlap=overlap,
                    route_count=args.routes,
                    base_seed=args.base_seed,
                    fresh_processes=2,
                )
            )
    encoded = [canonical_json_bytes(audit) for audit in audits]
    if encoded[0] != encoded[1]:
        raise SystemExit(
            "fresh Perfect DB audit processes produced different bytes"
        )
    summary = verify_layered_perfect_audit(audits[0])

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
