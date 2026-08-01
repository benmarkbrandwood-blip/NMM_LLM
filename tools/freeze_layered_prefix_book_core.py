"""Freeze the source-only 22-member Book stratum selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.layered_core_selection import derive_book_core
from learned_ai.training.run_contract import canonical_json_bytes


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    evidence = ROOT / "docs" / "evidence"
    experiments = ROOT / "docs" / "experiments"
    selection = derive_book_core(
        sanmill_book_audit=_load(
            evidence / "sanmill-layered-book-source-audit-2026-07-25.json"
        ),
        expert_book_audit=_load(
            evidence
            / "sanmill-layered-expert-book-reviewed-source-audit-2026-07-26.json"
        ),
        expert_coverage=_load(
            experiments
            / "sanmill-layered-expert-book-coverage-decision-2026-08-01.json"
        ),
        expert_shortlist=_load(
            experiments
            / "sanmill-layered-expert-book-shortlist-proposal-2026-07-31.json"
        ),
    )
    payload = {
        "schema_version": "nmm.layered-opening-prefix-book-core-decision.v1",
        "status": "book_membership_frozen_other_strata_pending",
        "decision_date": "2026-08-01",
        "candidate_loaded": False,
        "games_played": 0,
        "fallback": "none",
        "composition_decision": (
            "sanmill-layered-opening-prefix-v2-composition-decision-"
            "2026-08-01.json"
        ),
        "selection": selection,
        "decision": {
            "book_subtype_allocation_frozen": True,
            "book_membership_frozen": True,
            "human_db_membership_frozen": False,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
