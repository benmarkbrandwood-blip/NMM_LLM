from __future__ import annotations

import json
from pathlib import Path

from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "docs"
    / "experiments"
    / "sanmill-corrected-retained-v2-heldout-eval-v1.json"
)
AUDIT = (
    ROOT
    / "docs"
    / "evidence"
    / "sanmill-corrected-retained-v2-heldout-exposure-2026-08-09.json"
)


def test_frozen_plan_binds_operational_and_strict_analysis() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan_identity = plan.pop("plan_identity")
    assert canonical_sha256(plan) == plan_identity
    assert plan["workload"] == {
        "games": 128,
        "max_active_hours": 6.0,
        "pairs": 64,
        "safe_exact_resume_same_spec": True,
        "unique_starts": 64,
    }
    assert plan["baseline"]["fixed_node_ceiling_per_logical_turn"] == 500_000
    assert plan["protocol"]["max_post_prefix_logical_plies"] == 1536
    assert plan["protocol"]["max_ply_disposition"] == "incomplete-invalid-not-draw"
    assert plan["analysis"]["strict_independence_is_sensitivity_analysis"] is True
    assert plan["claim_boundary"]["multiple_training_seeds_available"] is False

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    audit_identity = audit.pop("audit_identity")
    assert canonical_sha256(audit) == audit_identity
    assert audit["summary"]["record_count"] == 64
    assert audit["summary"]["strict_independence_count"] == 34
    assert audit["summary"]["strict_independence_stratum_counts"] == {
        "book": 13,
        "perfect_db": 21,
    }
