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
    DEVELOPMENT_ANALYSIS_DEVICE,
    ScheduleIsolationSequenceError,
    build_sequence_authorization,
    build_sequence_steps,
    execute_sequence_steps,
    validate_sequence_authorization,
)
from scripts import run_target_refresh_schedule_isolation_sequence as runner


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/experiments/sanmill-target-refresh-schedule-isolation-diagnostic-v2.json"
)
SEQUENCE_READINESS_IDENTITY = "b" * 64


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

    assert [
        (step["kind"], step.get("seed"), step.get("condition")) for step in steps
    ] == [
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
        sequence_readiness_identity=SEQUENCE_READINESS_IDENTITY,
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
    assert authorization["sequence_readiness_identity"] == SEQUENCE_READINESS_IDENTITY
    assert authorization["development_analysis_device"] == DEVELOPMENT_ANALYSIS_DEVICE
    assert "automatic-retry" in authorization["prohibited_operations"]
    assert "long-training-launch" in authorization["prohibited_operations"]
    assert (
        validate_sequence_authorization(
            authorization,
            contract=contract,
            readiness=readiness,
            sequence_readiness_identity=SEQUENCE_READINESS_IDENTITY,
        )
        == authorization["authorization_identity"]
    )


def test_parent_authorization_rejects_resource_expansion() -> None:
    contract = _contract()
    readiness = _readiness(contract)
    authorization = build_sequence_authorization(
        contract=contract,
        readiness=readiness,
        sequence_readiness_identity=SEQUENCE_READINESS_IDENTITY,
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
            sequence_readiness_identity=SEQUENCE_READINESS_IDENTITY,
        )


def test_parent_authorization_rejects_another_sequence_readiness() -> None:
    contract = _contract()
    readiness = _readiness(contract)
    authorization = build_sequence_authorization(
        contract=contract,
        readiness=readiness,
        sequence_readiness_identity=SEQUENCE_READINESS_IDENTITY,
        decision_note="Owner approved this exact bounded sequence.",
        authorized_at_utc="2026-08-11T01:00:00.000000Z",
    )

    with pytest.raises(
        ScheduleIsolationSequenceError,
        match="sequence authorization differs",
    ):
        validate_sequence_authorization(
            authorization,
            contract=contract,
            readiness=readiness,
            sequence_readiness_identity="c" * 64,
        )


def test_sequence_readiness_identity_survives_authorization_recording() -> None:
    technical = {"schema_version": "test", "source": {"head": "a" * 40}}

    before = runner._sequence_readiness_result(
        technical,
        authorization_present=False,
    )
    after = runner._sequence_readiness_result(
        technical,
        authorization_present=True,
    )

    assert before["sequence_readiness_identity"] == after[
        "sequence_readiness_identity"
    ]
    assert before["launch_authorized"] is False
    assert after["launch_authorized"] is True


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


def _completed_training_steps() -> list[dict]:
    completed = [
        {
            "kind": "run-prefix",
            "seed": seed,
            "result": {
                "completed_games": 50,
                "completed_post_fork_transitions": None,
                "elapsed_hours": 0.25,
            },
        }
        for seed in (67, 68, 69)
    ]
    completed.extend(
        {
            "kind": "run-arm",
            "seed": seed,
            "condition": condition,
            "result": {
                "completed_games": 600,
                "completed_post_fork_transitions": 8192,
                "elapsed_hours": 0.875,
            },
        }
        for seed in (67, 68, 69)
        for condition in ("refresh-once", "no-refresh")
    )
    return completed


def test_training_resource_audit_reconciles_all_parent_limits() -> None:
    audit = runner._training_resource_audit(
        _completed_training_steps(),
        contract=_contract(),
    )

    assert audit["prefix_training_games"] == 150
    assert audit["arm_contract_games_total"] == 3600
    assert audit["actual_training_games_total"] == 3450
    assert audit["post_fork_transitions_total"] == 49152
    assert audit["managed_active_hours_total"] == 6.0


def test_training_resource_audit_rejects_one_game_overrun() -> None:
    completed = _completed_training_steps()
    completed[-1]["result"]["completed_games"] = 601

    with pytest.raises(
        ScheduleIsolationSequenceError,
        match="arm game count differs",
    ):
        runner._training_resource_audit(completed, contract=_contract())


def test_exclusive_publication_never_overwrites(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    target = tmp_path / "one-shot.json"
    runner._publish_exclusive(target, {"value": 1})
    original = target.read_bytes()

    with pytest.raises(
        ScheduleIsolationSequenceError,
        match="output already exists",
    ):
        runner._publish_exclusive(target, {"value": 2})

    assert target.read_bytes() == original
