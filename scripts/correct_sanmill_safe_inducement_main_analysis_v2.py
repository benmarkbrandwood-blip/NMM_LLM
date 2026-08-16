#!/usr/bin/env python3
"""Correct the zero-query budget-decomposition denominator in the main result."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    sha256_file,
    write_sealed_json,
)
from learned_ai.evaluation.sanmill_safe_inducement import (
    MAIN_RESULT_SCHEMA,
    classify_main,
    decompose_budget_stability,
    frequency_weighted_gain,
    load_main_plan,
)


CORRECTION_SCHEMA = "nmm.sanmill-safe-inducement-main-analysis-correction.v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="docs/experiments/sanmill-safe-inducement-mechanism-v2.json",
    )
    parser.add_argument(
        "--source-result",
        default=(
            "docs/evidence/sanmill-safe-inducement-main-v2-"
            "manifest-2026-08-16.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "docs/evidence/sanmill-safe-inducement-main-v2-"
            "analysis-correction-2026-08-16.json"
        ),
    )
    args = parser.parse_args()

    plan, plan_file_sha = load_main_plan(_ROOT / args.plan)
    source_path = _ROOT / args.source_result
    raw = source_path.read_bytes()
    source = json.loads(raw)
    source_identity = source.pop("result_identity", None)
    if (
        source.get("schema_version") != MAIN_RESULT_SCHEMA
        or source_identity != canonical_sha256(source)
    ):
        parser.error("source main result identity differs")
    if source.get("plan_identity") != plan["plan_identity"]:
        parser.error("source result and v2 plan differ")
    analysis = source["analysis"]
    rows = analysis["measurements"]
    corrected = decompose_budget_stability(rows, plan=plan)
    if corrected["overall"]["states"] != 360:
        parser.error("corrected decomposition does not cover all states")
    corrected_decision = classify_main(
        analysis["summaries"],
        plan=plan,
        determinism_passed=True,
    )
    if corrected_decision != analysis["decision"]:
        parser.error("primary decision changed during decomposition correction")
    corrected_weighted = frequency_weighted_gain(
        analysis["summaries"],
        plan=plan,
    )
    if corrected_weighted != analysis["frequency_weighted_secondary"]:
        parser.error("frequency-weighted secondary changed during correction")

    payload = {
        "schema_version": CORRECTION_SCHEMA,
        "status": "technical_analysis_corrected_no_new_measurement",
        "source_result_identity": source_identity,
        "source_result_file_sha256": hashlib.sha256(raw).hexdigest(),
        "source_result_tracked_file": args.source_result,
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": plan_file_sha,
        "measurement_rows": len(rows),
        "measurement_identity": canonical_sha256(rows),
        "defect": {
            "scope": "budget_decomposition_state_denominator_only",
            "cause": (
                "states with no inducing action at any tested budget were not "
                "materialized in the state-flag map"
            ),
            "affected_fields": ["o_inv", "o_sens", "o_union", "states"],
            "unaffected": [
                "all per-cell observations",
                "all per-budget and per-phase summaries",
                "100000-node primary main decision",
                "invariant share among union-induced states",
                "frequency-weighted secondary metric",
                "resource and access ledgers",
            ],
            "original_erroneous_overall": analysis["budget_decomposition"][
                "overall"
            ],
        },
        "corrected_budget_decomposition": corrected,
        "primary_decision_reproduced_unchanged": corrected_decision,
        "frequency_weighted_secondary_reproduced_unchanged": corrected_weighted,
        "additional_resource_use": {
            "engine_single_step_queries": 0,
            "malom_queries": 0,
            "complete_games": 0,
            "model_loads": 0,
            "training_updates": 0,
            "database_writes": 0,
        },
        "access_audit": source["access_audit"],
        "claim_boundary": source["claim_boundary"],
        "implementation_files": {
            path: sha256_file(_ROOT / path)
            for path in (
                "learned_ai/evaluation/sanmill_safe_inducement.py",
                "scripts/correct_sanmill_safe_inducement_main_analysis_v2.py",
                "tests/test_sanmill_safe_inducement.py",
            )
        },
    }
    sealed = write_sealed_json(
        _ROOT / args.output,
        payload,
        identity_field="correction_identity",
    )
    print(sealed["correction_identity"])
    print(corrected["overall"])
    print(corrected["interpretation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
