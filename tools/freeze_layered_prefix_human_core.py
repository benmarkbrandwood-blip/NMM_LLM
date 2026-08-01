"""Freeze the source-only 21-member HumanDB stratum selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.layered_core_selection import derive_human_core
from learned_ai.training.run_contract import canonical_json_bytes


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{field} is absent from the local path registry")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    experiments = ROOT / "docs" / "experiments"
    human_audit = _load(
        ROOT
        / "docs"
        / "evidence"
        / "sanmill-layered-human-source-audit-2026-07-25.json"
    )
    book_decision = _load(
        experiments
        / "sanmill-layered-opening-prefix-v2-book-core-2026-08-01.json"
    )
    config = _load(ROOT / args.paths_config)
    ledger = _resolve(
        config.get("human_db_prefix12_history_ledger_path"),
        field="human_db_prefix12_history_ledger_path",
    )
    expected = human_audit["raw_game_source"]["history_ledger"]
    if ledger.stat().st_size != expected["byte_length"]:
        raise SystemExit("HumanDB ledger byte length drifted")
    if _sha256(ledger) != expected["sha256"]:
        raise SystemExit("HumanDB ledger SHA-256 drifted")

    with ledger.open("r", encoding="utf-8") as handle:
        header = json.loads(next(handle))
        records = (json.loads(line) for line in handle)
        selection = derive_human_core(
            human_audit=human_audit,
            book_selection=book_decision["selection"],
            ledger_header=header,
            ledger_records=records,
        )

    payload = {
        "schema_version": "nmm.layered-opening-prefix-human-core-decision.v1",
        "status": "human_membership_frozen_perfect_pending",
        "decision_date": "2026-08-01",
        "candidate_loaded": False,
        "games_played": 0,
        "fallback": "none",
        "composition_decision": (
            "sanmill-layered-opening-prefix-v2-composition-decision-"
            "2026-08-01.json"
        ),
        "book_core_decision": (
            "sanmill-layered-opening-prefix-v2-book-core-2026-08-01.json"
        ),
        "selection": selection,
        "decision": {
            "human_db_membership_frozen": True,
            "human_execution_records_frozen": False,
            "perfect_db_membership_frozen": False,
            "final_64_frozen": False,
            "evaluation_authorized": False,
            "training_authorized": False,
        },
    }

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(payload) + b"\n")
    print(f"wrote {output}")
    print(f"members={len(selection['members'])}")
    print(f"membership_identity={selection['membership_identity']}")
    print(
        "minimum_distinct_game_count="
        f"{selection['summary']['minimum_distinct_game_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
