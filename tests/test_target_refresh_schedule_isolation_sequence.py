"""Focused tests for the one-shot schedule-isolation parent sequence."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from learned_ai.training.run_contract import canonical_sha256
from learned_ai.validation.target_refresh_equal_transition_diagnostic import (
    load_equal_transition_contract,
)
from learned_ai.validation.target_refresh_schedule_isolation_sequence import (
    AUTHORIZATION_SCHEMA,
    ScheduleIsolationSequenceError,
    build_sequence_authorization,
    build_sequence_steps,
    execute_sequence_steps,
    validate_sequence_authorization,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/experiments/"
    "sanmill-target-refresh-schedule-isolation-diagnostic-v2.json"
)


def _contract() -> dict:
    return load_equal_transition_contract(CONTRACT_PATH)


def _readiness(contract: dict) -> dict:
    body = {
        "schema_version": "nmm.target-refresh-equal-transition-readiness.v1",
        "state": "prefix_plans_ready_for_product_authorization",
        "launch_authorized": False,
        "contract": {"plan_identity": contract["plan_identity"]},
        "source": {"head": "a" * 40},
    }
    return {**body, "readiness_identity": canonical_sha256(body)}


def test_sequence_order_is_prefix_then_same_seed_arms() -> None:
    steps = [step.to_dict() for step in build_sequence_steps(_contract())]

    assert [(step["kind"], step.get("seed"), step.get("condition")) for step in steps] == [
        ("run-prefix", 67, None),
        ("prepare-seed-arms", 67, None),
        ("run-arm", 67, "refresh-once"),
        ("run-arm", 67, "no-refresh"),
        ("run-prefix", 68, None),
        ("prepare-seed-arms", 68, None),
        ("run-arm", 68, "refresh-once"),
        ("run-arm", 68, "no-refresh"),
        ("run-prefix", 69, None),
        ("prepare-seed-arms", 69, None),
        ("run-arm", 69, "refresh-once"),
        ("run-arm", 69, "no-refresh"),
        ("publish-development-result", None, None),
    ]


def test_parent_authorization_binds_exact_aggregate_envelope() -> None:
    contract = _contract()
    readiness = _readiness(contract)

    authorization = build_sequence_authorization(
        contract=contract,
        readiness=readiness,
        decision_note="Owner approved this exact bounded sequence.",
        authorized_at_utc="2026-08-11T01:00:00.000000Z",
    )

    assert authorization["schema_version"] == AUTHORIZATION_SCHEMA
    assert authorization["resource_envelope"] == {
        "maximum_active_wall_hours_total": 6.0,
        "maximum_actual_training_games_total": 3450,
        "maximum_contract_training_games_total": 3600,
        "maximum_development_measurement_games_total": 288,
        "scientific_post_fork_transitions_total": 49152,
    }
    assert authorization["one_parent_launch_attempt"] is True
    assert "automatic-retry" in authorization["prohibited_operations"]
    assert "long-training-launch" in authorization["prohibited_operations"]
    assert (
        validate_sequence_authorization(
            authorization,
            contract=contract,
            readiness=readiness,
        )
        == authorization["authorization_identity"]
    )


def test_parent_authorization_rejects_resource_expansion() -> None:
    contract = _contract()
    readiness = _readiness(contract)
    authorization = build_sequence_authorization(
        contract=contract,
        readiness=readiness,
        decision_note="Owner approved this exact bounded sequence.",
        authorized_at_utc="2026-08-11T01:00:00.000000Z",
    )
    changed = deepcopy(authorization)
    changed["resource_envelope"]["maximum_active_wall_hours_total"] = 7.0
    body = dict(changed)
    body.pop("authorization_identity")
    changed["authorization_identity"] = canonical_sha256(body)

    with pytest.raises(
        ScheduleIsolationSequenceError,
        match="sequence authorization differs",
    ):
        validate_sequence_authorization(
            changed,
            contract=contract,
            readiness=readiness,
        )


def test_first_failure_stops_without_retry_or_later_steps() -> None:
    steps = build_sequence_steps(_contract())
    calls: list[tuple[str, int | None]] = []

    def run_prefix(step):
        calls.append((step.kind, step.seed))
        raise RuntimeError("synthetic failure")

    def forbidden(step):  # pragma: no cover - assertion is the call list
        calls.append((step.kind, step.seed))

    with pytest.raises(RuntimeError, match="synthetic failure"):
        execute_sequence_steps(
            steps,
            run_prefix=run_prefix,
            prepare_seed_arms=forbidden,
            run_arm=forbidden,
            publish_result=forbidden,
        )

    assert calls == [("run-prefix", 67)]


def test_success_dispatches_each_frozen_step_once() -> None:
    steps = build_sequence_steps(_contract())
    calls: list[tuple[str, int | None, str | None]] = []

    def record(step):
        item = (step.kind, step.seed, step.condition)
        calls.append(item)
        return item

    outputs = execute_sequence_steps(
        steps,
        run_prefix=record,
        prepare_seed_arms=record,
        run_arm=record,
        publish_result=record,
    )

    assert outputs == calls
    assert len(calls) == 13
    assert len(set(calls)) == 13
