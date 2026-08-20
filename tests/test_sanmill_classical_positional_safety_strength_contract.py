from __future__ import annotations

import json
from pathlib import Path

from learned_ai.evaluation.human_f0h0_feasibility import canonical_sha256
from learned_ai.evaluation.sanmill_classical_positional_safety_strength import (
    AUTHORIZATION_SCHEMA,
    PLAN_SCHEMA,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import sha256_file


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT
    / "docs/experiments/sanmill-classical-positional-safety-strength-v1.json"
)
AUTHORIZATION_PATH = (
    ROOT
    / "docs/experiments/sanmill-classical-positional-safety-strength-v1/"
    "authorization.json"
)


def test_frozen_plan_and_authorization_are_bound_and_within_envelope() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    plan_body = dict(plan)
    plan_identity = plan_body.pop("plan_identity")
    assert plan["schema_version"] == PLAN_SCHEMA
    assert canonical_sha256(plan_body) == plan_identity
    assert plan["experiment"]["planned_complete_games"] == 480
    assert plan["resource_envelope"]["planned_complete_games"] == 480
    assert plan["resource_envelope"]["maximum_complete_games"] == 600
    assert plan["resource_envelope"]["maximum_active_seconds"] == 18_000
    assert plan["precision_design"]["maximum_half_width"] == 0.045
    assert len(plan["start_subset"]["state_ids"]) == 48
    assert (
        canonical_sha256(plan["start_subset"]["state_ids"])
        == plan["start_subset"]["membership_identity"]
    )

    authorization = json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))
    auth_body = dict(authorization)
    auth_identity = auth_body.pop("authorization_identity")
    assert authorization["schema_version"] == AUTHORIZATION_SCHEMA
    assert canonical_sha256(auth_body) == auth_identity
    assert authorization["plan_identity"] == plan_identity
    assert authorization["plan_file_sha256"] == sha256_file(PLAN_PATH)
    assert authorization["resource_envelope"] == plan["resource_envelope"]
    assert authorization["one_execution_only"] is True
    assert authorization["automatic_retry_or_resume"] is False


def test_frozen_product_increment_is_only_the_final_gate() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    contract = plan["product_contract"]
    assert contract["actual_delivery"]["final_gate"].endswith(
        "ProductPositionalSafetyGate"
    )
    assert contract["actual_delivery"]["root_move_restriction_inside_primary_search"] is False
    assert "identical" in contract["increment_isolated_by_primary_contrast"]
    assert plan["malom_contract"]["safe_set"] == "A_pos"
    assert plan["malom_contract"]["A_allow_claim"] is False
