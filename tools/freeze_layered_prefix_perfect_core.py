"""Freeze the source-only 21-member Perfect DB stratum selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.layered_core_selection import derive_perfect_core
from learned_ai.training.run_contract import canonical_json_bytes


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    experiments = ROOT / "docs" / "experiments"
    perfect_audit = _load(
        ROOT
        / "docs"
        / "evidence"
        / "sanmill-layered-perfect-source-audit-2026-07-25.json"
    )
    book = _load(
        experiments
        / "sanmill-layered-opening-prefix-v2-book-core-2026-08-01.json"
    )
    human = _load(
        experiments
        / "sanmill-layered-opening-prefix-v2-human-core-2026-08-01.json"
    )
    selection = derive_perfect_core(
        perfect_audit=perfect_audit,
        book_selection=book["selection"],
        human_selection=human["selection"],
    )
    payload = {
        "schema_version": "nmm.layered-opening-prefix-perfect-core-decision.v1",
        "status": "all_source_membership_frozen_execution_corpus_pending",
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
        "human_core_decision": (
            "sanmill-layered-opening-prefix-v2-human-core-2026-08-01.json"
        ),
        "selection": selection,
        "decision": {
            "perfect_db_membership_frozen": True,
            "all_source_membership_frozen": True,
            "human_execution_records_frozen": False,
            "final_execution_corpus_frozen": False,
            "review_package_frozen": False,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
