from __future__ import annotations

import pytest

from learned_ai.training.run_contract import canonical_sha256
from learned_ai.validation.target_refresh_mature_fork_diagnostic import (
    READINESS_SCHEMA,
)
from learned_ai.validation.target_refresh_mature_fork_sequence import (
    MatureTargetRefreshSequenceError,
    build_sequence_authorization,
    build_sequence_steps,
    execute_sequence_steps,
    validate_sequence_authorization,
)


def _contract() -> dict:
    arms = []
    order = 0
    for seed in (67, 68, 69):
        for condition in ("refresh-mature", "stale-control"):
            order += 1
            arms.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "launch_order": order,
                    "arm_id": f"seed{seed}-{condition}",
                }
            )
    return {
        "objective": "test one mature refresh",
        "plan_identity": "a" * 64,
        "arms": arms,
        "resources": {
            "maximum_training_games_total": 3600,
            "maximum_training_games_per_arm": 600,
            "maximum_optimizer_consumed_transitions_total": 49152,
            "maximum_active_wall_hours_total": 4,
            "maximum_active_wall_hours_per_arm": 0.6,
            "maximum_no_update_games_total": 288,
            "maximum_requested_sanmill_node_ceilings": 172800000,
        },
        "claim_boundary": "development mechanism only",
    }


def _readiness(contract: dict) -> dict:
    body = {
        "schema_version": READINESS_SCHEMA,
        "state": "six_arm_plans_ready_for_one_parent_product_authorization",
        "verdict": "needs_decision",
        "launch_authorized": False,
        "contract": {"plan_identity": contract["plan_identity"]},
        "source": {"head": "b" * 40},
    }
    return {**body, "readiness_identity": canonical_sha256(body)}


def test_sequence_order_is_frozen() -> None:
    steps = build_sequence_steps(_contract())

    assert [(step.seed, step.condition) for step in steps[:-1]] == [
        (67, "refresh-mature"),
        (67, "stale-control"),
        (68, "refresh-mature"),
        (68, "stale-control"),
        (69, "refresh-mature"),
        (69, "stale-control"),
    ]
    assert steps[-1].kind == "publish-development-result"


def test_parent_authorization_binds_readiness_and_full_resource_envelope() -> None:
    contract = _contract()
    readiness = _readiness(contract)
    authorization = build_sequence_authorization(
        contract=contract,
        readiness=readiness,
        expected_readiness_identity=readiness["readiness_identity"],
        decision_note="Owner approved this bounded sequence.",
        authorized_at_utc="2026-08-12T00:00:00Z",
    )

    assert authorization["readiness_identity"] == readiness["readiness_identity"]
    assert authorization["resource_envelope"] == contract["resources"]
    assert (
        validate_sequence_authorization(
            authorization,
            contract=contract,
            readiness=readiness,
            expected_readiness_identity=readiness["readiness_identity"],
        )
        == authorization["authorization_identity"]
    )


def test_sequence_stops_on_first_arm_failure() -> None:
    calls = []

    def run_arm(step):
        calls.append(step.arm_id)
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        execute_sequence_steps(
            build_sequence_steps(_contract()),
            run_arm=run_arm,
            publish_result=lambda step: step,
        )

    assert calls == ["seed67-refresh-mature"]


def test_authorization_rejects_stale_readiness_identity() -> None:
    contract = _contract()
    readiness = _readiness(contract)

    with pytest.raises(
        MatureTargetRefreshSequenceError,
        match="expected readiness identity differs",
    ):
        build_sequence_authorization(
            contract=contract,
            readiness=readiness,
            expected_readiness_identity="c" * 64,
            decision_note="Owner approved this bounded sequence.",
            authorized_at_utc="2026-08-12T00:00:00Z",
        )
