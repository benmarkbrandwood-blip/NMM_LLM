from __future__ import annotations

import copy
import inspect
import json
import pickle
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

import learned_ai.training.classical_a_pos_governance as governance
from learned_ai.training.classical_a_pos_governance import (
    ConsumedAuthorizationPermit,
    GovernanceContractError,
    GovernanceEvent,
    GovernanceReplay,
    PendingAuthorizationConsumption,
    PendingOperationReservation,
    ProductionAuthorizationPermit,
    ProductionOperationPermit,
    ProductionRuntimePlanPermit,
    ResourceLedger,
    RuntimePlanRecord,
    SingleUseAuthorizationRecord,
    build_runtime_plan_draft,
    confirm_authorization_consumption,
    confirm_operation_reservation,
    decode_governance_ledger,
    encode_governance_ledger,
    prepare_authorization_consumption,
    prepare_operation_reservation,
    replay_governance_ledger,
    require_production_operation_permit,
    verify_runtime_plan,
    verify_single_use_authorization,
)
from learned_ai.training.run_contract import canonical_sha256


EXPECTED_PUBLIC_API = (
    "GovernanceContractError",
    "RuntimePlanRecord",
    "SingleUseAuthorizationRecord",
    "GovernanceEvent",
    "ResourceLedger",
    "GovernanceReplay",
    "ProductionRuntimePlanPermit",
    "ProductionAuthorizationPermit",
    "PendingAuthorizationConsumption",
    "ConsumedAuthorizationPermit",
    "PendingOperationReservation",
    "ProductionOperationPermit",
    "build_runtime_plan_draft",
    "verify_runtime_plan",
    "verify_single_use_authorization",
    "encode_governance_ledger",
    "decode_governance_ledger",
    "replay_governance_ledger",
    "prepare_authorization_consumption",
    "confirm_authorization_consumption",
    "prepare_operation_reservation",
    "confirm_operation_reservation",
    "require_production_operation_permit",
    "build_state_generation_completed_event",
    "build_state_frozen_event",
    "prepared_governance_event_bytes",
)


def _mutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable(item) for item in value]
    return value


def _resign(value: dict[str, Any], identity_field: str) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != identity_field}
    value[identity_field] = canonical_sha256(body)
    return value


def _test_fixture():
    return governance._issue_test_governance_fixture()


def _event_index(events, event_type: str, *, occurrence: int = 1) -> int:
    matches = [
        index for index, event in enumerate(events) if event["event_type"] == event_type
    ]
    return matches[occurrence - 1]


def _rehash_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remapped: dict[str, str] = {}
    previous: str | None = None
    for sequence, event in enumerate(events):
        old_identity = event["event_identity"]
        prerequisite = event["prerequisite_event_identity"]
        if prerequisite in remapped:
            event["prerequisite_event_identity"] = remapped[prerequisite]
        event["sequence"] = sequence
        event["previous_event_identity"] = previous
        body = {key: item for key, item in event.items() if key != "event_identity"}
        event["event_identity"] = canonical_sha256(body)
        remapped[old_identity] = event["event_identity"]
        previous = event["event_identity"]
    return events


def _replay_semantics(replay: GovernanceReplay) -> dict[str, Any]:
    return {
        "state": replay.state,
        "events": tuple(event.to_dict() for event in replay.events),
        "head_event_identity": replay.head_event_identity,
        "experiment_id": replay.experiment_id,
        "proposal_identity": replay.proposal_identity,
        "plan_identity": replay.plan_identity,
        "readiness_identity": replay.readiness_identity,
        "authorization_identity": replay.authorization_identity,
        "authorization_consumption_identity": (
            replay.authorization_consumption_identity
        ),
        "charged_totals": dict(replay.resource_ledger.charged_totals),
        "observed_totals": dict(replay.resource_ledger.observed_totals),
        "open_operation_id": replay.resource_ledger.open_operation_id,
        "restart_requires_failure_closure": (
            replay.resource_ledger.restart_requires_failure_closure
        ),
        "violation": replay.resource_ledger.violation,
        "terminal": replay.terminal,
        "completed_operations": replay.completed_operations,
    }


def test_public_surface_and_zero_argument_draft_builder() -> None:
    import learned_ai.training.classical_a_pos_governance as module

    assert module.__all__ == EXPECTED_PUBLIC_API
    assert tuple(inspect.signature(build_runtime_plan_draft).parameters) == ()
    assert not any(
        forbidden in name.lower()
        for name in module.__all__
        for forbidden in (
            "issuer",
            "publish",
            "write",
            "append",
            "launch",
            "execute",
        )
    )


def test_runtime_plan_draft_is_exact_nonissuable_and_deeply_immutable() -> None:
    plan = build_runtime_plan_draft()

    assert isinstance(plan, RuntimePlanRecord)
    assert plan["schema_version"] == "nmm.classical-a-pos-runtime-plan.v1"
    assert plan["experiment_id"] == "classical-a-pos-offline-distillation-v1"
    assert plan["proposal_commit"] == "948eb5a6352ce5427173ab76ff2886190f5e6886"
    assert (
        plan["proposal_identity"]
        == "edf3e1031ee4bd46b6c567891fcaab26fd288a49997410141e6e19745d070e5c"
    )
    assert (
        plan["profile_identity"]
        == "bfa8d2f8e19b1c24641e24e4765844f678cb9288838e5eda3f8782d7ace9cbe0"
    )
    assert plan["plan_status"] == "nonissuable-draft"
    assert plan["issuable"] is False
    assert plan["executable"] is False
    assert plan["authorization_required"] is True
    assert tuple(plan["technical_bindings"]) == plan["unresolved_bindings"]
    assert set(plan["technical_bindings"].values()) == {None}
    body = {key: item for key, item in plan.items() if key != "plan_identity"}
    assert plan["plan_identity"] == canonical_sha256(body)
    with pytest.raises(TypeError):
        plan["issuable"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        plan["technical_bindings"]["managed_git_state"] = "a" * 64  # type: ignore[index]


def test_runtime_plan_draft_exact_operations_caps_and_policies() -> None:
    plan = build_runtime_plan_draft()

    assert plan["operation_order"] == (
        "runtime-plan-freeze",
        "technical-readiness-freeze",
        "single-use-authorization-consumption",
        "state-generation",
        "state-freeze",
        "offline-d9-teacher-labeling",
        "corpus-freeze",
        "supervised-smoke",
        "seed-2026083001",
        "seed-2026083002",
        "seed-2026083003",
    )
    assert tuple(item["operation_id"] for item in plan["operations"]) == (
        "state-generation",
        "state-freeze",
        "offline-d9-teacher-labeling",
        "corpus-freeze",
        "supervised-smoke",
        "seed-2026083001",
        "seed-2026083002",
        "seed-2026083003",
    )
    assert plan["planned_resource_caps"] == {
        "schema_version": "nmm.classical-a-pos-planned-resource-caps.v1",
        "counts_as_authorization": False,
        "state_generation_games": 1_024,
        "teacher_nodes": 4_000_000_000,
        "teacher_positive_search_labels": 1_280,
        "active_seconds": 86_400,
        "seed_active_seconds_aggregate": 64_800,
        "seed_active_seconds_each": 21_600,
        "supervised_updates": 6_721,
        "evaluation_games": 0,
    }
    assert plan["checkpoint_policy"] == {
        "state0_role": "supervised_seed_latest",
        "interval_updates": 112,
        "smoke_role": "supervised_smoke_disposable",
        "latest_role": "supervised_seed_latest",
        "complete_role": "supervised_seed_complete",
    }
    assert plan["monitoring_policy"] == {
        "update_event_interval": 1,
        "heartbeat_seconds": 60,
        "maximum_concurrency": 1,
    }
    assert plan["retry_policy"] == {
        "automatic_retry": False,
        "retry_limit": 0,
        "exact_resume": "manual-same-attempt-latest-only",
    }


def test_private_complete_plan_is_structurally_valid_but_test_domain_only() -> None:
    fixture = _test_fixture()
    checked = verify_runtime_plan(fixture.plan)

    assert checked["plan_status"] == "frozen"
    assert checked["issuable"] is True
    assert checked["executable"] is False
    assert checked["unresolved_bindings"] == ()
    identities = tuple(checked["technical_bindings"].values())
    assert len(identities) == len(set(identities)) == 9
    assert type(fixture.plan_permit).__name__ == "_TestRuntimePlanPermit"
    assert not isinstance(fixture.plan_permit, ProductionRuntimePlanPermit)


def test_runtime_plan_technical_binding_object_order_is_not_semantic() -> None:
    fixture = _test_fixture()
    reordered = _mutable(fixture.plan)
    reordered["technical_bindings"] = {
        key: reordered["technical_bindings"][key]
        for key in reversed(tuple(reordered["technical_bindings"]))
    }

    checked = verify_runtime_plan(_resign(reordered, "plan_identity"))

    assert checked["technical_bindings"] == fixture.plan["technical_bindings"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("plan_status",), "frozen"),
        (("issuable",), True),
        (("executable",), True),
        (("authorization_required",), 1),
        (("proposal_commit",), "f" * 40),
        (("proposal_identity",), "f" * 64),
        (("planned_resource_caps", "active_seconds"), 86_399),
        (("planned_resource_caps", "evaluation_games"), 1),
        (("retry_policy", "automatic_retry"), True),
        (("checkpoint_policy", "interval_updates"), 111),
    ],
)
def test_draft_plan_drift_is_rejected_even_when_resigned(
    path: tuple[str, ...],
    value: Any,
) -> None:
    plan = _mutable(build_runtime_plan_draft())
    target = plan
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(GovernanceContractError):
        verify_runtime_plan(_resign(plan, "plan_identity"))


def test_plan_unknown_missing_binding_duplicate_and_bool_spoofs_fail() -> None:
    fixture = _test_fixture()
    attacks: list[dict[str, Any]] = []

    unknown = _mutable(fixture.plan)
    unknown["runtime_path"] = "forbidden"
    attacks.append(unknown)
    missing = _mutable(fixture.plan)
    del missing["monitoring_policy"]
    attacks.append(missing)
    null_binding = _mutable(fixture.plan)
    first_binding = next(iter(null_binding["technical_bindings"]))
    null_binding["technical_bindings"][first_binding] = None
    attacks.append(null_binding)
    duplicate_binding = _mutable(fixture.plan)
    keys = tuple(duplicate_binding["technical_bindings"])
    duplicate_binding["technical_bindings"][keys[1]] = duplicate_binding[
        "technical_bindings"
    ][keys[0]]
    attacks.append(duplicate_binding)
    bool_cap = _mutable(fixture.plan)
    bool_cap["planned_resource_caps"]["state_generation_games"] = True
    attacks.append(bool_cap)

    for attack in attacks:
        with pytest.raises(GovernanceContractError):
            verify_runtime_plan(_resign(attack, "plan_identity"))


def test_structurally_valid_caller_plan_cannot_create_production_permit() -> None:
    fixture = _test_fixture()
    caller_record = verify_runtime_plan(_mutable(fixture.plan))

    assert caller_record == fixture.plan
    for constructor in (ProductionRuntimePlanPermit, ProductionAuthorizationPermit):
        with pytest.raises((GovernanceContractError, TypeError)):
            constructor(None)  # type: ignore[call-arg]


def test_single_use_authorization_exact_semantics_and_deep_immutability() -> None:
    fixture = _test_fixture()
    authorization = verify_single_use_authorization(
        fixture.authorization,
        plan=fixture.plan,
    )

    assert isinstance(authorization, SingleUseAuthorizationRecord)
    assert authorization["authorization_status"] == "issued-single-use"
    assert authorization["consumption_limit"] == 1
    assert authorization["retry_limit"] == 0
    assert authorization["automatic_retry"] is False
    assert authorization["allow_exact_resume_same_attempt"] is True
    assert authorization["ordered_operations"] == tuple(
        item["operation_id"] for item in fixture.plan["operations"]
    )
    assert authorization["resource_caps"]["counts_as_authorization"] is True
    assert authorization["resource_caps"]["supervised_updates"] == 6_721
    body = {
        key: item
        for key, item in authorization.items()
        if key != "authorization_identity"
    }
    assert authorization["authorization_identity"] == canonical_sha256(body)
    with pytest.raises(TypeError):
        authorization["consumption_limit"] = 2  # type: ignore[index]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("authorization_status",), "authorized"),
        (("plan_identity",), "f" * 64),
        (("readiness_identity",), "f" * 64),
        (("consumption_limit",), 0),
        (("consumption_limit",), True),
        (("retry_limit",), 1),
        (("automatic_retry",), True),
        (("allow_exact_resume_same_attempt",), False),
        (("resource_caps", "counts_as_authorization"), False),
        (("resource_caps", "teacher_nodes"), 3_999_999_999),
        (("resource_caps", "teacher_nodes"), 4_000_000_001),
        (("ordered_operations",), ["state-generation"]),
        (("claim_boundaries_identity",), "f" * 64),
        (("prohibited_actions_identity",), "f" * 64),
    ],
)
def test_authorization_drift_is_rejected_even_when_resigned(
    path: tuple[str, ...],
    value: Any,
) -> None:
    fixture = _test_fixture()
    authorization = _mutable(fixture.authorization)
    target = authorization
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(GovernanceContractError):
        verify_single_use_authorization(
            _resign(authorization, "authorization_identity"),
            plan=fixture.plan,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("issued_at_utc", "2026-09-01T00:00:00.0Z"),
        ("not_before_utc", "2026-09-02T00:00:00Z"),
        ("consume_by_utc", "2026-09-01T00:00:01Z"),
        ("expires_at_utc", "2026-09-02T00:00:00Z"),
    ],
)
def test_authorization_timestamp_format_and_order_fail(
    field: str,
    value: str,
) -> None:
    fixture = _test_fixture()
    authorization = _mutable(fixture.authorization)
    authorization[field] = value

    with pytest.raises(GovernanceContractError):
        verify_single_use_authorization(
            _resign(authorization, "authorization_identity"),
            plan=fixture.plan,
        )


def test_authority_direct_and_delegation_identity_rules_are_strict() -> None:
    fixture = _test_fixture()
    direct = _mutable(fixture.authorization)
    direct["authority"]["standing_delegation_identity"] = "a" * 64
    with pytest.raises(GovernanceContractError):
        verify_single_use_authorization(
            _resign(direct, "authorization_identity"),
            plan=fixture.plan,
        )

    delegation = _mutable(fixture.authorization)
    delegation["authority"]["kind"] = "standing-delegation"
    delegation["authority"]["standing_delegation_identity"] = None
    with pytest.raises(GovernanceContractError):
        verify_single_use_authorization(
            _resign(delegation, "authorization_identity"),
            plan=fixture.plan,
        )
    delegation["authority"]["standing_delegation_identity"] = "a" * 64
    verify_single_use_authorization(
        _resign(delegation, "authorization_identity"),
        plan=fixture.plan,
    )


def test_c1_resource_and_zero_authorization_are_not_single_use_authorizations() -> None:
    from learned_ai.validation.classical_a_pos_distillation_readiness import (
        REQUESTED_RESOURCE_PACKAGE,
        ZERO_AUTHORIZATION_ENVELOPE,
    )

    fixture = _test_fixture()
    for wrong in (REQUESTED_RESOURCE_PACKAGE, ZERO_AUTHORIZATION_ENVELOPE):
        with pytest.raises(GovernanceContractError):
            verify_single_use_authorization(wrong, plan=fixture.plan)
    with pytest.raises(GovernanceContractError):
        verify_single_use_authorization(
            fixture.authorization,
            plan=build_runtime_plan_draft(),
        )


def test_governance_records_and_permit_classes_are_exposed() -> None:
    assert issubclass(GovernanceContractError, RuntimeError)
    assert all(
        isinstance(candidate, type)
        for candidate in (
            RuntimePlanRecord,
            SingleUseAuthorizationRecord,
            GovernanceEvent,
            ResourceLedger,
            GovernanceReplay,
            ProductionRuntimePlanPermit,
            ProductionAuthorizationPermit,
            PendingAuthorizationConsumption,
            ConsumedAuthorizationPermit,
            PendingOperationReservation,
            ProductionOperationPermit,
        )
    )


def test_imported_public_functions_are_callable() -> None:
    assert all(
        callable(candidate)
        for candidate in (
            verify_runtime_plan,
            verify_single_use_authorization,
            encode_governance_ledger,
            decode_governance_ledger,
            replay_governance_ledger,
            prepare_authorization_consumption,
            confirm_authorization_consumption,
            prepare_operation_reservation,
            confirm_operation_reservation,
            require_production_operation_permit,
        )
    )


def test_complete_governance_ledger_replays_exact_sequence_and_resources() -> None:
    fixture = _test_fixture()
    replay = replay_governance_ledger(
        fixture.events,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )

    assert len(fixture.events) == 81
    assert replay.state == "sequence_complete"
    assert replay.terminal is True
    assert replay.completed_operations == tuple(
        item["operation_id"] for item in fixture.plan["operations"]
    )
    assert replay.resource_ledger.charged_totals == {
        "state_generation_games": 1_024,
        "active_seconds": 86_400,
        "teacher_nodes": 4_000_000_000,
        "teacher_positive_search_labels": 1_280,
        "supervised_updates": 6_721,
        "evaluation_games": 0,
    }
    assert replay.resource_ledger.observed_totals["supervised_updates"] == 6_721
    assert replay.resource_ledger.open_operation_id is None
    assert replay.resource_ledger.restart_requires_failure_closure is False
    assert replay.resource_ledger.violation is None


def test_replay_completed_operations_rejects_mutable_or_duplicate_values() -> None:
    fixture = _test_fixture()
    replay = replay_governance_ledger(
        fixture.events,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )

    with pytest.raises(GovernanceContractError):
        replace(
            replay,
            completed_operations=list(replay.completed_operations),  # type: ignore[arg-type]
        )
    with pytest.raises(GovernanceContractError):
        replace(
            replay,
            completed_operations=(
                replay.completed_operations[0],
                replay.completed_operations[0],
            ),
        )
    assert (
        sum(event["event_type"] == "seed_checkpointed" for event in fixture.events)
        == 57
    )
    assert sum(event["event_type"] == "seed_paused" for event in fixture.events) == 1
    assert sum(event["event_type"] == "seed_resumed" for event in fixture.events) == 1


def test_canonical_ledger_round_trip_and_replay_are_exact() -> None:
    fixture = _test_fixture()
    continuous = replay_governance_ledger(
        fixture.events,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )

    payload = encode_governance_ledger(fixture.events)
    decoded = decode_governance_ledger(payload)
    restored = replay_governance_ledger(
        decoded,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )

    assert payload.endswith(b"\n")
    assert b"\r" not in payload
    assert decoded == fixture.events
    assert _replay_semantics(restored) == _replay_semantics(continuous)
    assert encode_governance_ledger(decoded) == payload


@pytest.mark.parametrize(
    "transform",
    [
        lambda payload: b"\xef\xbb\xbf" + payload,
        lambda payload: payload.removesuffix(b"\n"),
        lambda payload: payload.replace(b"\n", b"\r\n", 1),
        lambda payload: b"\n" + payload,
        lambda payload: b" " + payload,
        lambda payload: payload + b"\n",
        lambda payload: payload.replace(b'":', b'": ', 1),
    ],
)
def test_ledger_decoder_rejects_noncanonical_framing(transform: Any) -> None:
    fixture = _test_fixture()
    payload = encode_governance_ledger(fixture.events[:1])

    with pytest.raises(GovernanceContractError):
        decode_governance_ledger(transform(payload))


def test_ledger_decoder_rejects_duplicate_keys_constants_nonobjects_and_events() -> (
    None
):
    fixture = _test_fixture()
    payload = encode_governance_ledger(fixture.events[:1])
    duplicate_key = payload.replace(
        b"{",
        b'{"schema_version":"duplicate",',
        1,
    )
    nan = payload.replace(b'"sequence":0', b'"sequence":NaN')
    infinity = payload.replace(b'"sequence":0', b'"sequence":Infinity')

    for invalid in (duplicate_key, nan, infinity, b"[]\n", payload + payload):
        with pytest.raises(GovernanceContractError):
            decode_governance_ledger(invalid)


def test_ledger_encoder_requires_verified_event_records() -> None:
    fixture = _test_fixture()
    with pytest.raises(GovernanceContractError):
        encode_governance_ledger([fixture.events[0].to_dict()])  # type: ignore[list-item]
    with pytest.raises(GovernanceContractError):
        encode_governance_ledger([fixture.events[0], fixture.events[0]])
    with pytest.raises(GovernanceContractError):
        decode_governance_ledger(bytearray(encode_governance_ledger(())))  # type: ignore[arg-type]


def test_replay_rejects_event_hash_previous_sequence_delete_and_reorder_attacks() -> (
    None
):
    fixture = _test_fixture()
    originals = [_mutable(event) for event in fixture.events]

    bad_hash = copy.deepcopy(originals)
    bad_hash[4]["timestamp_utc"] = "2026-09-01T00:10:00Z"
    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            bad_hash,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )

    bad_previous = copy.deepcopy(originals)
    bad_previous[5]["previous_event_identity"] = "f" * 64
    _resign(bad_previous[5], "event_identity")
    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            bad_previous,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )

    bad_sequence = copy.deepcopy(originals)
    bad_sequence[5]["sequence"] = 6
    _resign(bad_sequence[5], "event_identity")
    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            bad_sequence,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )

    deleted = _rehash_events(copy.deepcopy(originals[:6] + originals[7:]))
    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            deleted,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )

    reordered = copy.deepcopy(originals)
    reordered[5], reordered[6] = reordered[6], reordered[5]
    _rehash_events(reordered)
    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            reordered,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_replay_rejects_time_reversal_and_cross_plan_events() -> None:
    fixture = _test_fixture()
    time_attack = [_mutable(event) for event in fixture.events]
    time_attack[5]["timestamp_utc"] = "2026-09-01T00:00:00Z"
    _rehash_events(time_attack)
    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            time_attack,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )

    other_plan = _mutable(fixture.plan)
    binding_key = next(iter(other_plan["technical_bindings"]))
    other_plan["technical_bindings"][binding_key] = "e" * 64
    other_plan = verify_runtime_plan(_resign(other_plan, "plan_identity"))
    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            fixture.events,
            plan=other_plan,
            authorization=fixture.authorization,
        )


@pytest.mark.parametrize(
    "removed_event_type",
    [
        "authorization_consumed",
        "state_frozen",
        "corpus_frozen",
        "seed_fresh_init_committed",
        "seed_checkpointed",
        "seed_completed",
    ],
)
def test_replay_rejects_skipped_prerequisite_or_seed_cursor_events(
    removed_event_type: str,
) -> None:
    fixture = _test_fixture()
    events = [_mutable(event) for event in fixture.events]
    del events[_event_index(events, removed_event_type)]
    _rehash_events(events)

    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            events,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_seed_pause_resume_requires_same_complete_checkpoint_ref() -> None:
    fixture = _test_fixture()
    events = [_mutable(event) for event in fixture.events]
    pause_index = _event_index(events, "seed_paused")
    resume_index = _event_index(events, "seed_resumed")
    assert (
        events[pause_index]["evidence"]["checkpoint"]
        == events[resume_index]["evidence"]["checkpoint"]
    )

    events[resume_index]["evidence"]["checkpoint"]["identity"] = "f" * 64
    _rehash_events(events)
    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            events,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_checkpoint_cursor_is_implicit_and_nineteen_intervals_are_required() -> None:
    fixture = _test_fixture()
    events = [_mutable(event) for event in fixture.events]
    seed1_checkpoints = [
        event
        for event in events
        if event["event_type"] == "seed_checkpointed" and event["seed"] == 2026083001
    ]
    assert len(seed1_checkpoints) == 19
    assert fixture.plan["checkpoint_policy"]["interval_updates"] == 112
    assert 19 * 112 == 2_128
    assert 20 * 112 == 2_240
    assert set(seed1_checkpoints[0]["evidence"]["checkpoint"]) == {
        "role",
        "identity",
        "file_sha256",
        "size_bytes",
    }
    # C3 derives cursor progress from event order only.  Binding this opaque ref
    # to checkpoint payload cursor/RNG is intentionally left to the future
    # supervised production issuer and is not claimed by this replay core.

    second_index = next(
        index for index, event in enumerate(events) if event is seed1_checkpoints[1]
    )
    events[second_index]["evidence"]["checkpoint"] = copy.deepcopy(
        seed1_checkpoints[0]["evidence"]["checkpoint"]
    )
    _rehash_events(events)
    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            events,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


@pytest.mark.parametrize(
    "attack",
    ["complete_reuses_latest", "second_seed_reuses_state0", "new_identity_reuses_file"],
)
def test_checkpoint_references_are_new_across_progress_and_seed_scopes(
    attack: str,
) -> None:
    fixture = _test_fixture()
    events = [_mutable(event) for event in fixture.events]
    if attack == "complete_reuses_latest":
        completed_index = _event_index(events, "seed_completed")
        latest_index = max(
            index
            for index, event in enumerate(events[:completed_index])
            if event["event_type"] == "seed_checkpointed"
            and event["seed"] == 2026083001
        )
        reused = copy.deepcopy(events[latest_index]["evidence"]["checkpoint"])
        reused["role"] = fixture.plan["checkpoint_policy"]["complete_role"]
        events[completed_index]["evidence"]["checkpoint"] = reused
    elif attack == "second_seed_reuses_state0":
        first_index = _event_index(events, "seed_fresh_init_committed", occurrence=1)
        second_index = _event_index(events, "seed_fresh_init_committed", occurrence=2)
        events[second_index]["evidence"]["checkpoint"] = copy.deepcopy(
            events[first_index]["evidence"]["checkpoint"]
        )
    else:
        first_index = _event_index(events, "seed_checkpointed", occurrence=1)
        second_index = _event_index(events, "seed_checkpointed", occurrence=2)
        events[second_index]["evidence"]["checkpoint"]["file_sha256"] = events[
            first_index
        ]["evidence"]["checkpoint"]["file_sha256"]
    _rehash_events(events)

    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            events,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


@pytest.mark.parametrize(
    ("event_type", "role"),
    [
        ("smoke_passed", "supervised_seed_latest"),
        ("seed_fresh_init_committed", "supervised_seed_complete"),
        ("seed_checkpointed", "supervised_smoke_disposable"),
        ("seed_completed", "supervised_seed_latest"),
    ],
)
def test_checkpoint_role_cross_contamination_fails(
    event_type: str,
    role: str,
) -> None:
    fixture = _test_fixture()
    events = [_mutable(event) for event in fixture.events]
    index = _event_index(events, event_type)
    events[index]["evidence"]["checkpoint"]["role"] = role
    _rehash_events(events)

    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            events,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


@pytest.mark.parametrize(
    ("event_type", "resource", "value"),
    [
        ("state_generation_reserved", "state_generation_games", 1_023),
        ("state_generation_reserved", "state_generation_games", 1_025),
        ("state_generation_reserved", "state_generation_games", True),
        ("teacher_reserved", "teacher_nodes", 3_999_999_999),
        ("teacher_reserved", "teacher_nodes", 4_000_000_001),
        ("smoke_reserved", "supervised_updates", 0),
        ("smoke_reserved", "evaluation_games", 1),
        ("seed_reserved", "active_seconds", 21_599),
        ("seed_reserved", "active_seconds", 21_601),
    ],
)
def test_reservation_resource_drift_fails_even_when_rehashed(
    event_type: str,
    resource: str,
    value: Any,
) -> None:
    fixture = _test_fixture()
    events = [_mutable(event) for event in fixture.events]
    index = _event_index(events, event_type)
    events[index]["resource_reservation"][resource] = value
    _rehash_events(events)

    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            events,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_negative_observation_and_over_reservation_fail() -> None:
    fixture = _test_fixture()
    for value in (-1, 4_000_000_001):
        events = [_mutable(event) for event in fixture.events]
        index = _event_index(events, "teacher_completed")
        events[index]["resource_observation"]["teacher_nodes"] = value
        _rehash_events(events)
        with pytest.raises(GovernanceContractError):
            replay_governance_ledger(
                events,
                plan=fixture.plan,
                authorization=fixture.authorization,
            )


def test_reservation_prefix_charges_full_cap_and_cannot_reconstruct_permit() -> None:
    fixture = _test_fixture()
    index = _event_index(fixture.events, "state_generation_reserved")
    replay = replay_governance_ledger(
        fixture.events[: index + 1],
        plan=fixture.plan,
        authorization=fixture.authorization,
    )

    assert replay.state == "state_generation_running"
    assert replay.resource_ledger.open_operation_id == "state-generation"
    assert replay.resource_ledger.restart_requires_failure_closure is True
    assert replay.resource_ledger.charged_totals["state_generation_games"] == 1_024
    assert replay.resource_ledger.charged_totals["active_seconds"] == 3_600
    rogue_pending = object.__new__(PendingOperationReservation)
    with pytest.raises(GovernanceContractError):
        confirm_operation_reservation(rogue_pending, replay)


def _terminal_event(
    fixture,
    replay: GovernanceReplay,
    *,
    event_type: str,
    timestamp_utc: str,
    observation: Mapping[str, int] | None = None,
) -> GovernanceEvent:
    governance._validate_replay_capability(replay)
    operation = (
        None
        if replay.resource_ledger.open_operation_id is None
        else governance._operation_by_id(
            fixture.plan,
            replay.resource_ledger.open_operation_id,
        )
    )
    return governance._make_event(
        sequence=len(replay.events),
        timestamp_utc=timestamp_utc,
        event_type=event_type,
        from_state=replay.state,
        to_state={
            "fatal_stop": "failed_closed",
            "authorization_revoked": "revoked",
            "authorization_expired": "expired",
        }[event_type],
        plan=fixture.plan,
        readiness_identity=fixture.authorization["readiness_identity"],
        authorization_identity=replay.authorization_identity,
        authorization_consumption_identity=replay.authorization_consumption_identity,
        operation=operation,
        prerequisite_event_identity=replay.head_event_identity,
        evidence=None,
        resource_reservation=None,
        resource_observation=observation,
        reason_code=f"test-{event_type}",
        previous_event_identity=replay.head_event_identity,
    )


def test_fatal_stop_after_open_reservation_is_terminal_and_preserves_charge() -> None:
    fixture = _test_fixture()
    index = _event_index(fixture.events, "state_generation_reserved")
    prefix = fixture.events[: index + 1]
    replay = replay_governance_ledger(
        prefix,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    terminal = _terminal_event(
        fixture,
        replay,
        event_type="fatal_stop",
        timestamp_utc="2026-09-01T00:01:00Z",
    )
    stopped = replay_governance_ledger(
        (*prefix, terminal),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )

    assert stopped.state == "failed_closed"
    assert stopped.terminal is True
    assert stopped.resource_ledger.charged_totals["state_generation_games"] == 1_024
    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            (*prefix, terminal, fixture.events[index + 1]),
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_smoke_and_seed_completion_record_variable_active_seconds() -> None:
    fixture = _test_fixture()
    events = [_mutable(event) for event in fixture.events]
    smoke_index = _event_index(events, "smoke_passed")
    seed_index = _event_index(events, "seed_completed")
    events[smoke_index]["resource_observation"]["active_seconds"] = 30
    events[seed_index]["resource_observation"]["active_seconds"] = 20
    _rehash_events(events)

    replay = replay_governance_ledger(
        events,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )

    assert replay.resource_ledger.observed_totals["active_seconds"] == 50
    assert replay.resource_ledger.observed_totals["supervised_updates"] == 6_721


def test_terminal_over_cap_observation_is_recorded_as_violation() -> None:
    fixture = _test_fixture()
    index = _event_index(fixture.events, "teacher_reserved")
    prefix = fixture.events[: index + 1]
    replay = replay_governance_ledger(
        prefix,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    terminal = _terminal_event(
        fixture,
        replay,
        event_type="fatal_stop",
        timestamp_utc="2026-09-01T00:01:00Z",
        observation={
            **governance._zero_resource_vector(),
            "teacher_nodes": 4_000_000_001,
        },
    )
    stopped = replay_governance_ledger(
        (*prefix, terminal),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )

    assert stopped.state == "failed_closed"
    assert stopped.resource_ledger.violation in {
        "resource_observation_exceeds_reservation",
        "authorized_resource_cap_exceeded",
    }


def test_fatal_stop_cannot_observe_resources_without_an_open_reservation() -> None:
    fixture = _test_fixture()
    consumed_index = _event_index(fixture.events, "authorization_consumed")
    prefix = fixture.events[: consumed_index + 1]
    replay = replay_governance_ledger(
        prefix,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    terminal = _terminal_event(
        fixture,
        replay,
        event_type="fatal_stop",
        timestamp_utc="2026-09-01T00:01:00Z",
        observation={**governance._zero_resource_vector(), "active_seconds": 1},
    )

    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            (*prefix, terminal),
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_revocation_and_expiry_enter_terminal_states_at_exact_boundaries() -> None:
    fixture = _test_fixture()
    registered_index = _event_index(fixture.events, "authorization_registered")
    prefix = fixture.events[: registered_index + 1]
    registered = replay_governance_ledger(
        prefix,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    revoked_event = _terminal_event(
        fixture,
        registered,
        event_type="authorization_revoked",
        timestamp_utc="2026-09-01T00:01:00Z",
    )
    revoked = replay_governance_ledger(
        (*prefix, revoked_event),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    assert revoked.state == "revoked"

    expired_event = _terminal_event(
        fixture,
        registered,
        event_type="authorization_expired",
        timestamp_utc="2026-09-02T00:00:01Z",
    )
    expired = replay_governance_ledger(
        (*prefix, expired_event),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    assert expired.state == "expired"

    too_early = _terminal_event(
        fixture,
        registered,
        event_type="authorization_expired",
        timestamp_utc="2026-09-02T00:00:00Z",
    )
    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            (*prefix, too_early),
            plan=fixture.plan,
            authorization=fixture.authorization,
        )

    consumed_index = _event_index(fixture.events, "authorization_consumed")
    consumed_prefix = fixture.events[: consumed_index + 1]
    consumed = replay_governance_ledger(
        consumed_prefix,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    sequence_expired_event = _terminal_event(
        fixture,
        consumed,
        event_type="authorization_expired",
        timestamp_utc="2026-09-03T00:00:01Z",
    )
    assert (
        replay_governance_ledger(
            (*consumed_prefix, sequence_expired_event),
            plan=fixture.plan,
            authorization=fixture.authorization,
        ).state
        == "expired"
    )
    sequence_too_early = _terminal_event(
        fixture,
        consumed,
        event_type="authorization_expired",
        timestamp_utc="2026-09-03T00:00:00Z",
    )
    with pytest.raises(GovernanceContractError):
        replay_governance_ledger(
            (*consumed_prefix, sequence_too_early),
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def _replay_prefix(
    fixture, event_type: str, *, occurrence: int = 1
) -> GovernanceReplay:
    index = _event_index(fixture.events, event_type, occurrence=occurrence)
    return replay_governance_ledger(
        fixture.events[: index + 1],
        plan=fixture.plan,
        authorization=fixture.authorization,
    )


def _nondecoded_pending_candidate(
    kind: str,
    *,
    payload: bytes,
    in_memory_event: GovernanceEvent,
) -> GovernanceEvent | dict[str, Any]:
    if kind == "pending_context_event":
        return in_memory_event
    if kind == "json_mapping":
        parsed = json.loads(payload)
        assert isinstance(parsed, dict)
        return parsed
    decoded = decode_governance_ledger(payload)[0]
    if kind == "decoded_to_dict":
        return decoded.to_dict()
    if kind == "reconstructed_event":
        return GovernanceEvent(decoded.to_dict())
    raise AssertionError(f"unknown candidate kind: {kind}")


@pytest.mark.parametrize(
    "candidate_kind",
    [
        "pending_context_event",
        "json_mapping",
        "decoded_to_dict",
        "reconstructed_event",
    ],
)
def test_authorization_confirm_requires_strict_decoder_object_provenance(
    candidate_kind: str,
) -> None:
    fixture = _test_fixture()
    unconsumed = _replay_prefix(fixture, "authorization_registered")
    pending, payload = governance._test_prepare_authorization_consumption(
        fixture.plan_permit,
        fixture.authorization_permit,
        unconsumed,
        timestamp_utc="2026-09-01T00:00:03Z",
    )
    in_memory_event = governance._TEST_PENDING_AUTH_CONTEXTS[pending].event
    candidate = _nondecoded_pending_candidate(
        candidate_kind,
        payload=payload,
        in_memory_event=in_memory_event,
    )
    replay = replay_governance_ledger(
        (*unconsumed.events, candidate),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )

    with pytest.raises(GovernanceContractError):
        governance._test_confirm_authorization_consumption(pending, replay)


@pytest.mark.parametrize(
    "candidate_kind",
    [
        "pending_context_event",
        "json_mapping",
        "decoded_to_dict",
        "reconstructed_event",
    ],
)
def test_operation_confirm_requires_strict_decoder_object_provenance(
    candidate_kind: str,
) -> None:
    fixture = _test_fixture()
    unconsumed = _replay_prefix(fixture, "authorization_registered")
    pending_auth, auth_payload = governance._test_prepare_authorization_consumption(
        fixture.plan_permit,
        fixture.authorization_permit,
        unconsumed,
        timestamp_utc="2026-09-01T00:00:03Z",
    )
    decoded_auth = decode_governance_ledger(auth_payload)
    consumed_replay = replay_governance_ledger(
        (*unconsumed.events, *decoded_auth),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    consumed = governance._test_confirm_authorization_consumption(
        pending_auth,
        consumed_replay,
    )
    pending, payload = governance._test_prepare_operation_reservation(
        consumed,
        consumed_replay,
        operation_id="state-generation",
        timestamp_utc="2026-09-01T00:00:04Z",
    )
    in_memory_event = governance._TEST_PENDING_OPERATION_CONTEXTS[pending].event
    candidate = _nondecoded_pending_candidate(
        candidate_kind,
        payload=payload,
        in_memory_event=in_memory_event,
    )
    replay = replay_governance_ledger(
        (*consumed_replay.events, candidate),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )

    with pytest.raises(GovernanceContractError):
        governance._test_confirm_operation_reservation(pending, replay)


def test_test_domain_authorization_write_ahead_requires_exact_replay() -> None:
    fixture = _test_fixture()
    unconsumed = _replay_prefix(fixture, "authorization_registered")

    with pytest.raises(GovernanceContractError):
        prepare_authorization_consumption(
            fixture.plan_permit,  # type: ignore[arg-type]
            fixture.authorization_permit,  # type: ignore[arg-type]
            unconsumed,
            timestamp_utc="2026-09-01T00:00:03Z",
        )
    pending, payload = governance._test_prepare_authorization_consumption(
        fixture.plan_permit,
        fixture.authorization_permit,
        unconsumed,
        timestamp_utc="2026-09-01T00:00:03Z",
    )
    persisted = decode_governance_ledger(payload)

    assert len(persisted) == 1
    assert persisted[0]["event_type"] == "authorization_consumed"
    assert persisted[0]["previous_event_identity"] == unconsumed.head_event_identity
    with pytest.raises(GovernanceContractError):
        governance._test_prepare_authorization_consumption(
            fixture.plan_permit,
            fixture.authorization_permit,
            unconsumed,
            timestamp_utc="2026-09-01T00:00:03Z",
        )
    with pytest.raises(GovernanceContractError):
        governance._test_confirm_authorization_consumption(pending, unconsumed)
    with pytest.raises(GovernanceContractError):
        confirm_authorization_consumption(pending, unconsumed)  # type: ignore[arg-type]

    consumed_replay = replay_governance_ledger(
        (*unconsumed.events, *persisted),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    consumed = governance._test_confirm_authorization_consumption(
        pending,
        consumed_replay,
    )

    assert type(consumed).__name__ == "_TestConsumedAuthorizationPermit"
    assert not isinstance(consumed, ConsumedAuthorizationPermit)
    with pytest.raises(GovernanceContractError):
        governance._test_confirm_authorization_consumption(pending, consumed_replay)


def test_test_domain_operation_write_ahead_and_one_use_executor_permit() -> None:
    fixture = _test_fixture()
    unconsumed = _replay_prefix(fixture, "authorization_registered")
    pending_auth, auth_payload = governance._test_prepare_authorization_consumption(
        fixture.plan_permit,
        fixture.authorization_permit,
        unconsumed,
        timestamp_utc="2026-09-01T00:00:03Z",
    )
    consumed_replay = replay_governance_ledger(
        (*unconsumed.events, *decode_governance_ledger(auth_payload)),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    consumed = governance._test_confirm_authorization_consumption(
        pending_auth,
        consumed_replay,
    )

    with pytest.raises(GovernanceContractError):
        prepare_operation_reservation(
            consumed,  # type: ignore[arg-type]
            consumed_replay,
            operation_id="state-generation",
            timestamp_utc="2026-09-01T00:00:04Z",
        )
    pending, payload = governance._test_prepare_operation_reservation(
        consumed,
        consumed_replay,
        operation_id="state-generation",
        timestamp_utc="2026-09-01T00:00:04Z",
    )
    reserved_event = decode_governance_ledger(payload)
    reserved_replay = replay_governance_ledger(
        (*consumed_replay.events, *reserved_event),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )

    assert (
        reserved_replay.resource_ledger.charged_totals["state_generation_games"]
        == 1_024
    )
    with pytest.raises(GovernanceContractError):
        governance._test_prepare_operation_reservation(
            consumed,
            consumed_replay,
            operation_id="state-generation",
            timestamp_utc="2026-09-01T00:00:04Z",
        )
    with pytest.raises(GovernanceContractError):
        governance._test_confirm_operation_reservation(pending, consumed_replay)
    with pytest.raises(GovernanceContractError):
        confirm_operation_reservation(pending, reserved_replay)  # type: ignore[arg-type]

    permit = governance._test_confirm_operation_reservation(
        pending,
        reserved_replay,
    )
    with pytest.raises(GovernanceContractError):
        require_production_operation_permit(
            permit,  # type: ignore[arg-type]
            reserved_replay,
            operation_id="state-generation",
            purpose="state",
            seed=None,
        )
    attempt_identity = governance._test_require_operation_permit(
        permit,
        reserved_replay,
        operation_id="state-generation",
        purpose="state",
        seed=None,
    )

    assert len(attempt_identity) == 64
    with pytest.raises(GovernanceContractError):
        governance._test_require_operation_permit(
            permit,
            reserved_replay,
            operation_id="state-generation",
            purpose="state",
            seed=None,
        )


def test_confirmed_operation_permit_must_be_consumed_before_next_reservation() -> None:
    fixture = _test_fixture()
    unconsumed = _replay_prefix(fixture, "authorization_registered")
    pending_auth, auth_payload = governance._test_prepare_authorization_consumption(
        fixture.plan_permit,
        fixture.authorization_permit,
        unconsumed,
        timestamp_utc="2026-09-01T00:00:03Z",
    )
    consumed_replay = replay_governance_ledger(
        (*unconsumed.events, *decode_governance_ledger(auth_payload)),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    consumed = governance._test_confirm_authorization_consumption(
        pending_auth,
        consumed_replay,
    )
    pending, reservation_payload = governance._test_prepare_operation_reservation(
        consumed,
        consumed_replay,
        operation_id="state-generation",
        timestamp_utc="2026-09-01T00:00:04Z",
    )
    reserved_replay = replay_governance_ledger(
        (*consumed_replay.events, *decode_governance_ledger(reservation_payload)),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    permit = governance._test_confirm_operation_reservation(pending, reserved_replay)
    state_frozen = _replay_prefix(fixture, "state_frozen")

    with pytest.raises(GovernanceContractError):
        governance._test_prepare_operation_reservation(
            consumed,
            state_frozen,
            operation_id="offline-d9-teacher-labeling",
            timestamp_utc="2026-09-01T00:00:07Z",
        )

    governance._test_require_operation_permit(
        permit,
        reserved_replay,
        operation_id="state-generation",
        purpose="state",
        seed=None,
    )
    teacher_pending, teacher_payload = governance._test_prepare_operation_reservation(
        consumed,
        state_frozen,
        operation_id="offline-d9-teacher-labeling",
        timestamp_utc="2026-09-01T00:00:07Z",
    )
    assert type(teacher_pending).__name__ == "_TestPendingOperationReservation"
    assert (
        decode_governance_ledger(teacher_payload)[0]["event_type"] == "teacher_reserved"
    )


@pytest.mark.parametrize(
    ("operation_id", "purpose", "seed"),
    [
        ("state-generation", "teacher", None),
        ("offline-d9-teacher-labeling", "state", None),
        ("state-generation", "state", 2026083001),
    ],
)
def test_operation_permit_executor_binding_mismatch_does_not_consume_valid_use(
    operation_id: str,
    purpose: str,
    seed: int | None,
) -> None:
    fixture = _test_fixture()
    unconsumed = _replay_prefix(fixture, "authorization_registered")
    pending_auth, auth_payload = governance._test_prepare_authorization_consumption(
        fixture.plan_permit,
        fixture.authorization_permit,
        unconsumed,
        timestamp_utc="2026-09-01T00:00:03Z",
    )
    consumed_replay = replay_governance_ledger(
        (*unconsumed.events, *decode_governance_ledger(auth_payload)),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    consumed = governance._test_confirm_authorization_consumption(
        pending_auth,
        consumed_replay,
    )
    pending, operation_payload = governance._test_prepare_operation_reservation(
        consumed,
        consumed_replay,
        operation_id="state-generation",
        timestamp_utc="2026-09-01T00:00:04Z",
    )
    reserved_replay = replay_governance_ledger(
        (*consumed_replay.events, *decode_governance_ledger(operation_payload)),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    permit = governance._test_confirm_operation_reservation(pending, reserved_replay)

    with pytest.raises(GovernanceContractError):
        governance._test_require_operation_permit(
            permit,
            reserved_replay,
            operation_id=operation_id,
            purpose=purpose,
            seed=seed,
        )
    governance._test_require_operation_permit(
        permit,
        reserved_replay,
        operation_id="state-generation",
        purpose="state",
        seed=None,
    )


def test_pending_and_permit_confirmation_rejects_wrong_event_or_replay_head() -> None:
    fixture = _test_fixture()
    unconsumed = _replay_prefix(fixture, "authorization_registered")
    pending, payload = governance._test_prepare_authorization_consumption(
        fixture.plan_permit,
        fixture.authorization_permit,
        unconsumed,
        timestamp_utc="2026-09-01T00:00:03Z",
    )
    fixture_consumed = _replay_prefix(fixture, "authorization_consumed")

    assert (
        fixture_consumed.events[-1]["event_identity"]
        == fixture.events[_event_index(fixture.events, "authorization_consumed")][
            "event_identity"
        ]
    )
    with pytest.raises(GovernanceContractError):
        governance._test_confirm_authorization_consumption(pending, fixture_consumed)
    decoded = decode_governance_ledger(payload)
    persisted_replay = replay_governance_ledger(
        (*unconsumed.events, *decoded),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    assert persisted_replay.events[-1] is decoded[0]
    governance._test_confirm_authorization_consumption(pending, persisted_replay)
    with pytest.raises(GovernanceContractError):
        governance._test_confirm_authorization_consumption(pending, persisted_replay)


def test_persistence_failure_leaves_authorization_and_consumed_context_spent() -> None:
    fixture = _test_fixture()
    unconsumed = _replay_prefix(fixture, "authorization_registered")
    governance._test_prepare_authorization_consumption(
        fixture.plan_permit,
        fixture.authorization_permit,
        unconsumed,
        timestamp_utc="2026-09-01T00:00:03Z",
    )
    with pytest.raises(GovernanceContractError):
        governance._test_prepare_authorization_consumption(
            fixture.plan_permit,
            fixture.authorization_permit,
            unconsumed,
            timestamp_utc="2026-09-01T00:00:03Z",
        )

    second = _test_fixture()
    second_unconsumed = _replay_prefix(second, "authorization_registered")
    pending_auth, second_payload = governance._test_prepare_authorization_consumption(
        second.plan_permit,
        second.authorization_permit,
        second_unconsumed,
        timestamp_utc="2026-09-01T00:00:03Z",
    )
    consumed_replay = replay_governance_ledger(
        (*second_unconsumed.events, *decode_governance_ledger(second_payload)),
        plan=second.plan,
        authorization=second.authorization,
    )
    consumed = governance._test_confirm_authorization_consumption(
        pending_auth,
        consumed_replay,
    )
    governance._test_prepare_operation_reservation(
        consumed,
        consumed_replay,
        operation_id="state-generation",
        timestamp_utc="2026-09-01T00:00:04Z",
    )
    with pytest.raises(GovernanceContractError):
        governance._test_prepare_operation_reservation(
            consumed,
            consumed_replay,
            operation_id="state-generation",
            timestamp_utc="2026-09-01T00:00:04Z",
        )


def test_all_production_permits_reject_direct_rogue_and_subclass_construction() -> None:
    production_types = (
        ProductionRuntimePlanPermit,
        ProductionAuthorizationPermit,
        PendingAuthorizationConsumption,
        ConsumedAuthorizationPermit,
        PendingOperationReservation,
        ProductionOperationPermit,
    )
    for permit_type in production_types:
        with pytest.raises((GovernanceContractError, TypeError)):
            permit_type(None)  # type: ignore[call-arg]
        rogue = object.__new__(permit_type)
        assert not hasattr(rogue, "__dict__")
        with pytest.raises(AttributeError):
            rogue.injected = True  # type: ignore[attr-defined]
        with pytest.raises(TypeError):
            copy.copy(rogue)
        with pytest.raises(TypeError):
            copy.deepcopy(rogue)
        with pytest.raises(TypeError):
            pickle.dumps(rogue)

    for permit_type in production_types:
        with pytest.raises(TypeError):

            class _ForbiddenSubclass(permit_type):  # type: ignore[misc, valid-type]
                pass


def test_rogue_production_capabilities_and_replay_are_rejected_by_every_api() -> None:
    fixture = _test_fixture()
    unconsumed = _replay_prefix(fixture, "authorization_registered")
    consumed_replay = _replay_prefix(fixture, "authorization_consumed")
    reserved_replay = _replay_prefix(fixture, "state_generation_reserved")
    rogue_plan = object.__new__(ProductionRuntimePlanPermit)
    rogue_auth = object.__new__(ProductionAuthorizationPermit)
    rogue_pending_auth = object.__new__(PendingAuthorizationConsumption)
    rogue_consumed = object.__new__(ConsumedAuthorizationPermit)
    rogue_pending_operation = object.__new__(PendingOperationReservation)
    rogue_operation = object.__new__(ProductionOperationPermit)

    with pytest.raises(GovernanceContractError):
        prepare_authorization_consumption(
            rogue_plan,
            rogue_auth,
            unconsumed,
            timestamp_utc="2026-09-01T00:00:03Z",
        )
    with pytest.raises(GovernanceContractError):
        confirm_authorization_consumption(rogue_pending_auth, consumed_replay)
    with pytest.raises(GovernanceContractError):
        prepare_operation_reservation(
            rogue_consumed,
            consumed_replay,
            operation_id="state-generation",
            timestamp_utc="2026-09-01T00:00:04Z",
        )
    with pytest.raises(GovernanceContractError):
        confirm_operation_reservation(rogue_pending_operation, reserved_replay)
    with pytest.raises(GovernanceContractError):
        require_production_operation_permit(
            rogue_operation,
            reserved_replay,
            operation_id="state-generation",
            purpose="state",
            seed=None,
        )
    rogue_replay = object.__new__(GovernanceReplay)
    with pytest.raises(GovernanceContractError):
        prepare_authorization_consumption(
            rogue_plan,
            rogue_auth,
            rogue_replay,
            timestamp_utc="2026-09-01T00:00:03Z",
        )


def test_test_domain_handles_never_enter_production_apis_or_gain_attributes() -> None:
    fixture = _test_fixture()
    for handle in (fixture.plan_permit, fixture.authorization_permit):
        assert not isinstance(
            handle,
            (ProductionRuntimePlanPermit, ProductionAuthorizationPermit),
        )
        assert not hasattr(handle, "__dict__")
        with pytest.raises(AttributeError):
            handle.context = fixture.plan  # type: ignore[attr-defined]
        with pytest.raises(TypeError):
            copy.copy(handle)
        with pytest.raises(TypeError):
            copy.deepcopy(handle)
        with pytest.raises(TypeError):
            pickle.dumps(handle)


def test_governance_module_has_no_file_process_or_mutating_public_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _test_fixture()
    payload = encode_governance_ledger(fixture.events[:3])

    def reject_io(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("governance core attempted external I/O")

    monkeypatch.setattr("builtins.open", reject_io)
    assert "Path" not in governance.__dict__
    assert "subprocess" not in governance.__dict__
    assert build_runtime_plan_draft()["issuable"] is False
    verify_runtime_plan(fixture.plan)
    verify_single_use_authorization(fixture.authorization, plan=fixture.plan)
    assert encode_governance_ledger(decode_governance_ledger(payload)) == payload
    assert (
        replay_governance_ledger(
            fixture.events[:3],
            plan=fixture.plan,
            authorization=fixture.authorization,
        ).state
        == "authorized_unconsumed"
    )


def test_state_generation_and_freeze_builders_derive_registered_replay_scope() -> None:
    fixture = _test_fixture()
    reserved_index = _event_index(fixture.events, "state_generation_reserved")
    decoded_prefix = decode_governance_ledger(
        encode_governance_ledger(fixture.events[: reserved_index + 1])
    )
    running = replay_governance_ledger(
        decoded_prefix,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )

    completed, completed_bytes = governance.build_state_generation_completed_event(
        running,
        timestamp_utc="2026-09-01T00:00:05Z",
        evidence={"inputs": [], "outputs": [], "checkpoint": None},
        state_generation_games=1_024,
        active_seconds=17,
    )
    assert completed_bytes == encode_governance_ledger((completed,))
    completed_decoded = decode_governance_ledger(completed_bytes)[0]
    generated = replay_governance_ledger(
        (*decoded_prefix, completed_decoded),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    assert generated.state == "state_generated"
    assert completed["sequence"] == len(decoded_prefix)
    assert completed["operation_id"] == "state-generation"
    assert completed["attempt_identity"] == running.events[-1]["attempt_identity"]
    assert completed["previous_event_identity"] == running.head_event_identity

    frozen, frozen_bytes = governance.build_state_frozen_event(
        generated,
        timestamp_utc="2026-09-01T00:00:06Z",
        evidence={"inputs": [], "outputs": [], "checkpoint": None},
    )
    assert frozen_bytes == encode_governance_ledger((frozen,))
    frozen_decoded = decode_governance_ledger(frozen_bytes)[0]
    replay = replay_governance_ledger(
        (*decoded_prefix, completed_decoded, frozen_decoded),
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    assert replay.state == "state_frozen"
    assert frozen["operation_id"] == "state-freeze"
    assert frozen["attempt_identity"] is not None
    assert frozen["prerequisite_event_identity"] == completed["event_identity"]


def test_prepared_event_bytes_are_registry_bound_and_domain_separated() -> None:
    fixture = _test_fixture()
    replay = _replay_prefix(fixture, "authorization_registered")
    pending, payload = governance._test_prepare_authorization_consumption(
        fixture.plan_permit,
        fixture.authorization_permit,
        replay,
        timestamp_utc="2026-09-01T00:00:03Z",
    )
    assert governance._test_prepared_governance_event_bytes(pending) == payload
    with pytest.raises(GovernanceContractError):
        governance.prepared_governance_event_bytes(pending)
    with pytest.raises(GovernanceContractError):
        governance.prepared_governance_event_bytes(payload)
