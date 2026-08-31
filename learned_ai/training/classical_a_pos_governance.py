"""Pure in-memory governance core for classical ``A_pos`` distillation.

The module validates an immutable runtime plan, a single-use authorization,
and a canonical append-only event ledger.  It deliberately has no filesystem,
subprocess, CLI, training, smoke, data-generation, or production issuer seam.
Production-shaped capabilities therefore cannot be created in this slice.
"""

from __future__ import annotations

import json
import math
import weakref
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Any

from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.validation import (
    classical_a_pos_distillation_readiness as _readiness,
)
from learned_ai.validation.classical_a_pos_experiment_proposal import (
    build_classical_a_pos_experiment_proposal,
)

__all__ = (
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
)

_PLAN_SCHEMA = "nmm.classical-a-pos-runtime-plan.v1"
_AUTHORIZATION_SCHEMA = "nmm.classical-a-pos-single-use-authorization.v1"
_AUTHORIZED_RESOURCE_SCHEMA = "nmm.classical-a-pos-authorized-resource-caps.v1"
_PLANNED_RESOURCE_SCHEMA = "nmm.classical-a-pos-planned-resource-caps.v1"
_EVENT_SCHEMA = "nmm.classical-a-pos-governance-event.v1"
_CONSUMPTION_SCHEMA = "nmm.classical-a-pos-authorization-consumption.v1"
_ATTEMPT_SCHEMA = "nmm.classical-a-pos-operation-attempt.v1"
_EXPERIMENT_ID = "classical-a-pos-offline-distillation-v1"
_PROPOSAL_COMMIT = "948eb5a6352ce5427173ab76ff2886190f5e6886"
_PROPOSAL_IDENTITY = "edf3e1031ee4bd46b6c567891fcaab26fd288a49997410141e6e19745d070e5c"
_PROFILE_IDENTITY = "bfa8d2f8e19b1c24641e24e4765844f678cb9288838e5eda3f8782d7ace9cbe0"
_HEX = frozenset("0123456789abcdef")
_SEEDS = (2026083001, 2026083002, 2026083003)

_TECHNICAL_BINDING_KEYS = (
    "production_d9_teacher_issuer",
    "supervised_plan_issuer",
    "state_freeze_teacher_order_event_chain",
    "supervised_controller",
    "supervised_smoke_authority",
    "production_corpus_loader_binding",
    "device_throughput_measurement",
    "launch_path_binding",
    "managed_git_state",
)
_PLAN_KEYS = {
    "schema_version",
    "experiment_id",
    "proposal_commit",
    "proposal_identity",
    "profile_identity",
    "plan_status",
    "issuable",
    "executable",
    "authorization_required",
    "technical_bindings",
    "unresolved_bindings",
    "operation_order",
    "operations",
    "planned_resource_caps",
    "checkpoint_policy",
    "monitoring_policy",
    "retry_policy",
    "plan_identity",
}
_AUTHORIZATION_KEYS = {
    "schema_version",
    "authorization_status",
    "experiment_id",
    "proposal_identity",
    "plan_identity",
    "readiness_identity",
    "authority",
    "issued_at_utc",
    "not_before_utc",
    "consume_by_utc",
    "expires_at_utc",
    "expiry_policy",
    "revocation_policy",
    "consumption_limit",
    "retry_limit",
    "automatic_retry",
    "allow_exact_resume_same_attempt",
    "ordered_operations",
    "resource_caps",
    "claim_boundaries_identity",
    "prohibited_actions_identity",
    "authorization_identity",
}
_EVENT_KEYS = {
    "schema_version",
    "sequence",
    "timestamp_utc",
    "event_type",
    "from_state",
    "to_state",
    "experiment_id",
    "proposal_identity",
    "plan_identity",
    "readiness_identity",
    "authorization_identity",
    "authorization_consumption_identity",
    "operation_id",
    "attempt_identity",
    "purpose",
    "seed",
    "prerequisite_event_identity",
    "evidence",
    "resource_reservation",
    "resource_observation",
    "reason_code",
    "previous_event_identity",
    "event_identity",
}
_RESOURCE_KEYS = (
    "state_generation_games",
    "active_seconds",
    "teacher_nodes",
    "teacher_positive_search_labels",
    "supervised_updates",
    "evaluation_games",
)
_EVIDENCE_KEYS = {"inputs", "outputs", "checkpoint"}
_EVIDENCE_REF_KEYS = {"role", "identity", "file_sha256", "size_bytes"}


class GovernanceContractError(RuntimeError):
    """A governance record, replay, or capability failed closed."""


def _freeze(value: Any, *, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GovernanceContractError(f"{field_name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise GovernanceContractError(f"{field_name} contains a non-string key")
            copied[key] = _freeze(item, field_name=f"{field_name}.{key}")
        return MappingProxyType(copied)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(
            _freeze(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(value)
        )
    raise GovernanceContractError(
        f"{field_name} contains unsupported {type(value).__name__} data"
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _snapshot(value: Any, *, field_name: str) -> Any:
    return _thaw(_freeze(value, field_name=field_name))


def _require_exact_keys(
    value: Any,
    expected: set[str],
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GovernanceContractError(f"{field_name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise GovernanceContractError(f"{field_name} contains a non-string key")
    actual = set(value)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise GovernanceContractError(
            f"{field_name} has unknown keys: {', '.join(unknown)}"
        )
    if missing:
        raise GovernanceContractError(
            f"{field_name} has missing keys: {', '.join(missing)}"
        )
    return value


def _require_exact(value: Any, expected: Any, *, field_name: str) -> None:
    if isinstance(expected, Mapping):
        checked = _require_exact_keys(value, set(expected), field_name=field_name)
        for key in expected:
            _require_exact(
                checked[key],
                expected[key],
                field_name=f"{field_name}.{key}",
            )
        return
    if isinstance(expected, Sequence) and not isinstance(
        expected,
        (str, bytes, bytearray),
    ):
        if not isinstance(value, Sequence) or isinstance(
            value,
            (str, bytes, bytearray),
        ):
            raise GovernanceContractError(f"{field_name} must be an array")
        if len(value) != len(expected):
            raise GovernanceContractError(f"{field_name} length differs")
        for index, (actual_item, expected_item) in enumerate(
            zip(value, expected, strict=True)
        ):
            _require_exact(
                actual_item,
                expected_item,
                field_name=f"{field_name}[{index}]",
            )
        return
    if type(value) is not type(expected) or value != expected:
        raise GovernanceContractError(f"{field_name} differs from the frozen contract")


def _require_sha256(value: Any, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise GovernanceContractError(
            f"{field_name} must be a 64-character lowercase SHA-256"
        )
    return value


def _require_nonempty_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise GovernanceContractError(f"{field_name} must be non-empty exact text")
    return value


def _require_nonnegative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GovernanceContractError(f"{field_name} must be a non-negative integer")
    return value


def _parse_timestamp(value: Any, *, field_name: str) -> datetime:
    if (
        not isinstance(value, str)
        or len(value) != 20
        or value[4] != "-"
        or value[7] != "-"
        or value[10] != "T"
        or value[13] != ":"
        or value[16] != ":"
        or not value.endswith("Z")
    ):
        raise GovernanceContractError(f"{field_name} must be whole-second RFC3339 UTC")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise GovernanceContractError(
            f"{field_name} must be whole-second RFC3339 UTC"
        ) from exc


@dataclass(frozen=True, slots=True)
class _FrozenMappingRecord(Mapping[str, Any]):
    _data: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_data", _freeze(self._data, field_name="record"))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self._data)


class RuntimePlanRecord(_FrozenMappingRecord):
    """Deeply immutable structural runtime-plan record."""

    __slots__ = ()


class SingleUseAuthorizationRecord(_FrozenMappingRecord):
    """Deeply immutable structural single-use authorization record."""

    __slots__ = ()


class GovernanceEvent(_FrozenMappingRecord):
    """Deeply immutable governance event with a self-authenticating body."""

    __slots__ = ("__weakref__",)


_STRICT_DECODED_EVENTS: dict[
    int,
    tuple[weakref.ReferenceType[GovernanceEvent], bytes],
] = {}


def _register_strict_decoded_event(event: GovernanceEvent, event_bytes: bytes) -> None:
    key = id(event)

    def discard(reference: weakref.ReferenceType[GovernanceEvent]) -> None:
        current = _STRICT_DECODED_EVENTS.get(key)
        if current is not None and current[0] is reference:
            del _STRICT_DECODED_EVENTS[key]

    reference = weakref.ref(event, discard)
    _STRICT_DECODED_EVENTS[key] = (reference, bytes(event_bytes))


def _strict_decoded_event_bytes(event: GovernanceEvent) -> bytes | None:
    registered = _STRICT_DECODED_EVENTS.get(id(event))
    if registered is None or registered[0]() is not event:
        return None
    return registered[1]


def _zero_resource_vector() -> dict[str, int]:
    return {key: 0 for key in _RESOURCE_KEYS}


def _resource_vector(**updates: int) -> dict[str, int]:
    value = _zero_resource_vector()
    unknown = sorted(set(updates) - set(_RESOURCE_KEYS))
    if unknown:
        raise RuntimeError(f"unknown internal resource keys: {', '.join(unknown)}")
    value.update(updates)
    return value


def _validate_resource_vector(value: Any, *, field_name: str) -> dict[str, int]:
    checked = _require_exact_keys(value, set(_RESOURCE_KEYS), field_name=field_name)
    return {
        key: _require_nonnegative_int(
            checked[key],
            field_name=f"{field_name}.{key}",
        )
        for key in _RESOURCE_KEYS
    }


@dataclass(frozen=True, slots=True)
class ResourceLedger:
    """Immutable charged and observed resource totals from one replay."""

    charged_totals: Mapping[str, int]
    observed_totals: Mapping[str, int]
    open_operation_id: str | None
    restart_requires_failure_closure: bool
    violation: str | None

    def __post_init__(self) -> None:
        charged = _validate_resource_vector(
            self.charged_totals,
            field_name="resource ledger charged_totals",
        )
        observed = _validate_resource_vector(
            self.observed_totals,
            field_name="resource ledger observed_totals",
        )
        if self.open_operation_id is not None:
            _require_nonempty_text(
                self.open_operation_id,
                field_name="resource ledger open_operation_id",
            )
        if type(self.restart_requires_failure_closure) is not bool:
            raise GovernanceContractError(
                "resource ledger restart flag must be a boolean"
            )
        if self.violation is not None:
            _require_nonempty_text(
                self.violation,
                field_name="resource ledger violation",
            )
        object.__setattr__(
            self,
            "charged_totals",
            _freeze(charged, field_name="resource ledger charged_totals"),
        )
        object.__setattr__(
            self,
            "observed_totals",
            _freeze(observed, field_name="resource ledger observed_totals"),
        )


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class GovernanceReplay:
    """Immutable result of a full genesis-to-head governance replay."""

    state: str
    events: tuple[GovernanceEvent, ...]
    head_event_identity: str | None
    experiment_id: str
    proposal_identity: str
    plan_identity: str
    readiness_identity: str
    authorization_identity: str | None
    authorization_consumption_identity: str | None
    resource_ledger: ResourceLedger
    terminal: bool
    completed_operations: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonempty_text(self.state, field_name="replay state")
        if not isinstance(self.events, tuple) or any(
            type(event) is not GovernanceEvent for event in self.events
        ):
            raise GovernanceContractError(
                "replay events must be GovernanceEvent records"
            )
        if self.head_event_identity is not None:
            _require_sha256(
                self.head_event_identity,
                field_name="replay head_event_identity",
            )
        _require_nonempty_text(self.experiment_id, field_name="replay experiment_id")
        _require_sha256(self.proposal_identity, field_name="replay proposal_identity")
        _require_sha256(self.plan_identity, field_name="replay plan_identity")
        _require_sha256(self.readiness_identity, field_name="replay readiness_identity")
        if self.authorization_identity is not None:
            _require_sha256(
                self.authorization_identity,
                field_name="replay authorization_identity",
            )
        if self.authorization_consumption_identity is not None:
            _require_sha256(
                self.authorization_consumption_identity,
                field_name="replay authorization_consumption_identity",
            )
        if type(self.resource_ledger) is not ResourceLedger:
            raise GovernanceContractError("replay resource ledger type differs")
        if type(self.terminal) is not bool:
            raise GovernanceContractError("replay terminal flag must be boolean")
        if not isinstance(self.completed_operations, tuple) or any(
            not isinstance(operation_id, str) or not operation_id
            for operation_id in self.completed_operations
        ):
            raise GovernanceContractError(
                "replay completed_operations must be a tuple of operation ids"
            )
        if len(set(self.completed_operations)) != len(self.completed_operations):
            raise GovernanceContractError("replay completed_operations must not repeat")


def _validated_c2_proposal() -> Mapping[str, Any]:
    proposal = build_classical_a_pos_experiment_proposal()
    if (
        proposal["experiment_id"] != _EXPERIMENT_ID
        or proposal["proposal_identity"] != _PROPOSAL_IDENTITY
        or proposal["profile_identity"] != _PROFILE_IDENTITY
        or proposal["executable"] is not False
        or proposal["authorization_status"] != "unauthorized"
    ):
        raise GovernanceContractError("the frozen C2 proposal identity drifted")
    return proposal


def _planned_resource_caps() -> dict[str, Any]:
    requested = _readiness.validate_requested_resource_package(
        _readiness.REQUESTED_RESOURCE_PACKAGE
    )
    profile = _readiness.validate_training_profile(_readiness.FROZEN_TRAINING_PROFILE)
    if canonical_sha256(profile) != _PROFILE_IDENTITY:
        raise GovernanceContractError("the frozen C1 profile identity drifted")
    seed_each = requested["seed"]["per_seed_active_seconds"]
    seed_aggregate = requested["seed"]["aggregate_active_seconds"]
    updates_per_seed = profile["optimizer_updates_per_seed"]
    return {
        "schema_version": _PLANNED_RESOURCE_SCHEMA,
        "counts_as_authorization": False,
        "state_generation_games": requested["state_generation"]["games"],
        "teacher_nodes": requested["teacher"]["nodes"],
        "teacher_positive_search_labels": requested["teacher"][
            "positive_search_labels"
        ],
        "active_seconds": requested["sequence_active_seconds"],
        "seed_active_seconds_aggregate": seed_aggregate,
        "seed_active_seconds_each": seed_each,
        "supervised_updates": 1 + len(_SEEDS) * updates_per_seed,
        "evaluation_games": requested["evaluation_games"],
    }


def _operation_records() -> list[dict[str, Any]]:
    requested = _readiness.validate_requested_resource_package(
        _readiness.REQUESTED_RESOURCE_PACKAGE
    )
    profile = _readiness.validate_training_profile(_readiness.FROZEN_TRAINING_PROFILE)
    zero = _zero_resource_vector
    operations = [
        {
            "operation_id": "state-generation",
            "purpose": "state",
            "prerequisite_operation_id": None,
            "seed": None,
            "resource_reservation": _resource_vector(
                state_generation_games=requested["state_generation"]["games"],
                active_seconds=requested["state_generation"]["active_seconds"],
            ),
        },
        {
            "operation_id": "state-freeze",
            "purpose": "governance",
            "prerequisite_operation_id": "state-generation",
            "seed": None,
            "resource_reservation": zero(),
        },
        {
            "operation_id": "offline-d9-teacher-labeling",
            "purpose": "teacher",
            "prerequisite_operation_id": "state-freeze",
            "seed": None,
            "resource_reservation": _resource_vector(
                active_seconds=requested["teacher"]["active_seconds"],
                teacher_nodes=requested["teacher"]["nodes"],
                teacher_positive_search_labels=requested["teacher"][
                    "positive_search_labels"
                ],
            ),
        },
        {
            "operation_id": "corpus-freeze",
            "purpose": "governance",
            "prerequisite_operation_id": "offline-d9-teacher-labeling",
            "seed": None,
            "resource_reservation": zero(),
        },
        {
            "operation_id": "supervised-smoke",
            "purpose": "smoke",
            "prerequisite_operation_id": "corpus-freeze",
            "seed": None,
            "resource_reservation": _resource_vector(
                active_seconds=requested["smoke"]["active_seconds"],
                supervised_updates=1,
            ),
        },
    ]
    prior = "supervised-smoke"
    for seed in _SEEDS:
        operation_id = f"seed-{seed}"
        operations.append(
            {
                "operation_id": operation_id,
                "purpose": "seed",
                "prerequisite_operation_id": prior,
                "seed": seed,
                "resource_reservation": _resource_vector(
                    active_seconds=requested["seed"]["per_seed_active_seconds"],
                    supervised_updates=profile["optimizer_updates_per_seed"],
                ),
            }
        )
        prior = operation_id
    return operations


def _plan_body(
    *,
    status: str,
    bindings: Mapping[str, str | None],
) -> dict[str, Any]:
    proposal = _validated_c2_proposal()
    if status == "nonissuable-draft":
        issuable = False
        unresolved = list(_TECHNICAL_BINDING_KEYS)
    elif status == "frozen":
        issuable = True
        unresolved = []
    else:
        raise GovernanceContractError("runtime plan status is not supported")
    return {
        "schema_version": _PLAN_SCHEMA,
        "experiment_id": _EXPERIMENT_ID,
        "proposal_commit": _PROPOSAL_COMMIT,
        "proposal_identity": _PROPOSAL_IDENTITY,
        "profile_identity": _PROFILE_IDENTITY,
        "plan_status": status,
        "issuable": issuable,
        "executable": False,
        "authorization_required": True,
        "technical_bindings": dict(bindings),
        "unresolved_bindings": unresolved,
        "operation_order": list(proposal["operation_order"]),
        "operations": _operation_records(),
        "planned_resource_caps": _planned_resource_caps(),
        "checkpoint_policy": {
            "state0_role": "supervised_seed_latest",
            "interval_updates": 112,
            "smoke_role": "supervised_smoke_disposable",
            "latest_role": "supervised_seed_latest",
            "complete_role": "supervised_seed_complete",
        },
        "monitoring_policy": {
            "update_event_interval": 1,
            "heartbeat_seconds": 60,
            "maximum_concurrency": 1,
        },
        "retry_policy": {
            "automatic_retry": False,
            "retry_limit": 0,
            "exact_resume": "manual-same-attempt-latest-only",
        },
    }


def build_runtime_plan_draft() -> RuntimePlanRecord:
    """Build the only public plan: an immutable, nonissuable draft."""
    bindings = {key: None for key in _TECHNICAL_BINDING_KEYS}
    body = _plan_body(status="nonissuable-draft", bindings=bindings)
    return verify_runtime_plan({**body, "plan_identity": canonical_sha256(body)})


def _build_test_complete_runtime_plan_record() -> RuntimePlanRecord:
    bindings = {
        key: canonical_sha256(
            {
                "schema_version": "nmm.classical-a-pos-test-binding.v1",
                "binding": key,
            }
        )
        for key in _TECHNICAL_BINDING_KEYS
    }
    body = _plan_body(status="frozen", bindings=bindings)
    return verify_runtime_plan({**body, "plan_identity": canonical_sha256(body)})


def verify_runtime_plan(record: Mapping[str, Any]) -> RuntimePlanRecord:
    """Verify a draft or structurally complete plan without issuing authority."""
    if isinstance(record, RuntimePlanRecord):
        raw = record.to_dict()
    elif isinstance(record, Mapping):
        raw = _snapshot(record, field_name="runtime plan")
    else:
        raise GovernanceContractError("runtime plan must be an object")
    checked = _require_exact_keys(raw, _PLAN_KEYS, field_name="runtime plan")
    status = checked["plan_status"]
    if status not in {"nonissuable-draft", "frozen"}:
        raise GovernanceContractError("runtime plan status is not supported")
    bindings = _require_exact_keys(
        checked["technical_bindings"],
        set(_TECHNICAL_BINDING_KEYS),
        field_name="runtime plan technical_bindings",
    )
    if status == "nonissuable-draft":
        if any(value is not None for value in bindings.values()):
            raise GovernanceContractError("draft technical bindings must be null")
    else:
        identities = [
            _require_sha256(
                value,
                field_name=f"runtime plan technical_bindings.{key}",
            )
            for key, value in bindings.items()
        ]
        if len(set(identities)) != len(identities):
            raise GovernanceContractError(
                "frozen plan technical binding identities must be distinct"
            )
    expected_body = _plan_body(status=status, bindings=bindings)
    observed_body = {
        key: value for key, value in checked.items() if key != "plan_identity"
    }
    _require_exact(observed_body, expected_body, field_name="runtime plan")
    identity = _require_sha256(
        checked["plan_identity"],
        field_name="runtime plan plan_identity",
    )
    if identity != canonical_sha256(observed_body):
        raise GovernanceContractError("runtime plan identity does not match its body")
    return RuntimePlanRecord(checked)


def _require_complete_plan(plan: Mapping[str, Any]) -> RuntimePlanRecord:
    checked = verify_runtime_plan(plan)
    if checked["plan_status"] != "frozen" or checked["issuable"] is not True:
        raise GovernanceContractError("authorization requires a frozen issuable plan")
    return checked


def _operation_ids(plan: RuntimePlanRecord) -> tuple[str, ...]:
    return tuple(item["operation_id"] for item in plan["operations"])


def _operation_by_id(
    plan: RuntimePlanRecord,
    operation_id: str,
) -> Mapping[str, Any]:
    matches = [
        operation
        for operation in plan["operations"]
        if operation["operation_id"] == operation_id
    ]
    if len(matches) != 1:
        raise GovernanceContractError("runtime plan operation lookup differs")
    return matches[0]


def _expected_readiness_identity(plan: RuntimePlanRecord) -> str:
    return canonical_sha256(
        {
            "schema_version": "nmm.classical-a-pos-technical-readiness-binding.v1",
            "plan_identity": plan["plan_identity"],
            "technical_bindings": plan["technical_bindings"],
        }
    )


def _authorized_resource_caps(plan: RuntimePlanRecord) -> dict[str, Any]:
    caps = plan["planned_resource_caps"]
    return {
        **{
            key: _thaw(value)
            for key, value in caps.items()
            if key not in {"schema_version", "counts_as_authorization"}
        },
        "schema_version": _AUTHORIZED_RESOURCE_SCHEMA,
        "counts_as_authorization": True,
    }


def _proposal_boundary_identities() -> tuple[str, str]:
    proposal = _validated_c2_proposal()
    return (
        canonical_sha256(proposal["claim_boundaries"]),
        canonical_sha256(proposal["prohibited_actions"]),
    )


def _validate_authority(value: Any) -> dict[str, Any]:
    checked = _require_exact_keys(
        value,
        {
            "kind",
            "authorized_by_identity",
            "decision_identity",
            "standing_delegation_identity",
        },
        field_name="authorization authority",
    )
    kind = checked["kind"]
    if kind not in {"product-owner-direct", "standing-delegation"}:
        raise GovernanceContractError("authorization authority kind differs")
    _require_sha256(
        checked["authorized_by_identity"],
        field_name="authorization authority authorized_by_identity",
    )
    _require_sha256(
        checked["decision_identity"],
        field_name="authorization authority decision_identity",
    )
    delegation = checked["standing_delegation_identity"]
    if kind == "product-owner-direct":
        if delegation is not None:
            raise GovernanceContractError(
                "direct authority must not claim a standing delegation"
            )
    else:
        _require_sha256(
            delegation,
            field_name="authorization authority standing_delegation_identity",
        )
    return dict(checked)


def verify_single_use_authorization(
    record: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> SingleUseAuthorizationRecord:
    """Audit one exact authorization without creating a production permit."""
    checked_plan = _require_complete_plan(plan)
    if isinstance(record, SingleUseAuthorizationRecord):
        raw = record.to_dict()
    elif isinstance(record, Mapping):
        raw = _snapshot(record, field_name="single-use authorization")
    else:
        raise GovernanceContractError("single-use authorization must be an object")
    checked = _require_exact_keys(
        raw,
        _AUTHORIZATION_KEYS,
        field_name="single-use authorization",
    )
    fixed = {
        "schema_version": _AUTHORIZATION_SCHEMA,
        "authorization_status": "issued-single-use",
        "experiment_id": _EXPERIMENT_ID,
        "proposal_identity": _PROPOSAL_IDENTITY,
        "plan_identity": checked_plan["plan_identity"],
        "readiness_identity": _expected_readiness_identity(checked_plan),
        "expiry_policy": "consume-by-and-sequence-expiry-fail-closed",
        "revocation_policy": {
            "revocable": True,
            "check_before_every_operation": True,
            "immutable-grant-cannot-clear-revocation": True,
        },
        "consumption_limit": 1,
        "retry_limit": 0,
        "automatic_retry": False,
        "allow_exact_resume_same_attempt": True,
        "ordered_operations": list(_operation_ids(checked_plan)),
        "resource_caps": _authorized_resource_caps(checked_plan),
    }
    for key, expected in fixed.items():
        _require_exact(
            checked[key],
            expected,
            field_name=f"single-use authorization.{key}",
        )
    _validate_authority(checked["authority"])
    issued = _parse_timestamp(
        checked["issued_at_utc"],
        field_name="single-use authorization issued_at_utc",
    )
    not_before = _parse_timestamp(
        checked["not_before_utc"],
        field_name="single-use authorization not_before_utc",
    )
    consume_by = _parse_timestamp(
        checked["consume_by_utc"],
        field_name="single-use authorization consume_by_utc",
    )
    expires = _parse_timestamp(
        checked["expires_at_utc"],
        field_name="single-use authorization expires_at_utc",
    )
    if not issued <= not_before < consume_by < expires:
        raise GovernanceContractError("single-use authorization time order differs")
    claim_identity, prohibited_identity = _proposal_boundary_identities()
    _require_exact(
        checked["claim_boundaries_identity"],
        claim_identity,
        field_name="single-use authorization.claim_boundaries_identity",
    )
    _require_exact(
        checked["prohibited_actions_identity"],
        prohibited_identity,
        field_name="single-use authorization.prohibited_actions_identity",
    )
    observed_body = {
        key: value for key, value in checked.items() if key != "authorization_identity"
    }
    identity = _require_sha256(
        checked["authorization_identity"],
        field_name="single-use authorization authorization_identity",
    )
    if identity != canonical_sha256(observed_body):
        raise GovernanceContractError(
            "single-use authorization identity does not match its body"
        )
    return SingleUseAuthorizationRecord(checked)


def _build_test_authorization_record(
    plan: RuntimePlanRecord,
) -> SingleUseAuthorizationRecord:
    claim_identity, prohibited_identity = _proposal_boundary_identities()
    body = {
        "schema_version": _AUTHORIZATION_SCHEMA,
        "authorization_status": "issued-single-use",
        "experiment_id": _EXPERIMENT_ID,
        "proposal_identity": _PROPOSAL_IDENTITY,
        "plan_identity": plan["plan_identity"],
        "readiness_identity": _expected_readiness_identity(plan),
        "authority": {
            "kind": "product-owner-direct",
            "authorized_by_identity": canonical_sha256(
                {"schema_version": "nmm.test-authority.v1", "role": "owner"}
            ),
            "decision_identity": canonical_sha256(
                {"schema_version": "nmm.test-decision.v1", "decision": "authorize"}
            ),
            "standing_delegation_identity": None,
        },
        "issued_at_utc": "2026-09-01T00:00:00Z",
        "not_before_utc": "2026-09-01T00:00:01Z",
        "consume_by_utc": "2026-09-02T00:00:00Z",
        "expires_at_utc": "2026-09-03T00:00:00Z",
        "expiry_policy": "consume-by-and-sequence-expiry-fail-closed",
        "revocation_policy": {
            "revocable": True,
            "check_before_every_operation": True,
            "immutable-grant-cannot-clear-revocation": True,
        },
        "consumption_limit": 1,
        "retry_limit": 0,
        "automatic_retry": False,
        "allow_exact_resume_same_attempt": True,
        "ordered_operations": list(_operation_ids(plan)),
        "resource_caps": _authorized_resource_caps(plan),
        "claim_boundaries_identity": claim_identity,
        "prohibited_actions_identity": prohibited_identity,
    }
    return verify_single_use_authorization(
        {**body, "authorization_identity": canonical_sha256(body)},
        plan=plan,
    )


def _validate_evidence_ref(value: Any, *, field_name: str) -> dict[str, Any]:
    checked = _require_exact_keys(value, _EVIDENCE_REF_KEYS, field_name=field_name)
    _require_nonempty_text(checked["role"], field_name=f"{field_name}.role")
    _require_sha256(checked["identity"], field_name=f"{field_name}.identity")
    _require_sha256(
        checked["file_sha256"],
        field_name=f"{field_name}.file_sha256",
    )
    _require_nonnegative_int(
        checked["size_bytes"],
        field_name=f"{field_name}.size_bytes",
    )
    return dict(checked)


def _validate_evidence(value: Any) -> dict[str, Any]:
    checked = _require_exact_keys(value, _EVIDENCE_KEYS, field_name="event evidence")
    result: dict[str, Any] = {"inputs": [], "outputs": [], "checkpoint": None}
    for collection_name in ("inputs", "outputs"):
        collection = checked[collection_name]
        if not isinstance(collection, Sequence) or isinstance(
            collection,
            (str, bytes, bytearray),
        ):
            raise GovernanceContractError(
                f"event evidence.{collection_name} must be an array"
            )
        result[collection_name] = [
            _validate_evidence_ref(
                item,
                field_name=f"event evidence.{collection_name}[{index}]",
            )
            for index, item in enumerate(collection)
        ]
    checkpoint = checked["checkpoint"]
    if checkpoint is not None:
        result["checkpoint"] = _validate_evidence_ref(
            checkpoint,
            field_name="event evidence.checkpoint",
        )
    return result


def _verify_event_record(value: Mapping[str, Any]) -> GovernanceEvent:
    if isinstance(value, GovernanceEvent):
        raw = value.to_dict()
    elif isinstance(value, Mapping):
        raw = _snapshot(value, field_name="governance event")
    else:
        raise GovernanceContractError("governance event must be an object")
    checked = _require_exact_keys(raw, _EVENT_KEYS, field_name="governance event")
    _require_exact(
        checked["schema_version"],
        _EVENT_SCHEMA,
        field_name="governance event.schema_version",
    )
    _require_nonnegative_int(
        checked["sequence"], field_name="governance event.sequence"
    )
    _parse_timestamp(
        checked["timestamp_utc"],
        field_name="governance event.timestamp_utc",
    )
    for field_name in ("event_type", "from_state", "to_state", "experiment_id"):
        _require_nonempty_text(
            checked[field_name],
            field_name=f"governance event.{field_name}",
        )
    for field_name in ("proposal_identity", "plan_identity", "readiness_identity"):
        _require_sha256(
            checked[field_name],
            field_name=f"governance event.{field_name}",
        )
    for field_name in (
        "authorization_identity",
        "authorization_consumption_identity",
        "attempt_identity",
        "prerequisite_event_identity",
        "previous_event_identity",
    ):
        if checked[field_name] is not None:
            _require_sha256(
                checked[field_name],
                field_name=f"governance event.{field_name}",
            )
    for field_name in ("operation_id", "purpose", "reason_code"):
        if checked[field_name] is not None:
            _require_nonempty_text(
                checked[field_name],
                field_name=f"governance event.{field_name}",
            )
    if checked["seed"] is not None and (
        isinstance(checked["seed"], bool)
        or not isinstance(checked["seed"], int)
        or checked["seed"] not in _SEEDS
    ):
        raise GovernanceContractError("governance event.seed differs")
    checked["evidence"] = _validate_evidence(checked["evidence"])
    checked["resource_reservation"] = _validate_resource_vector(
        checked["resource_reservation"],
        field_name="governance event.resource_reservation",
    )
    checked["resource_observation"] = _validate_resource_vector(
        checked["resource_observation"],
        field_name="governance event.resource_observation",
    )
    observed_body = {
        key: item for key, item in checked.items() if key != "event_identity"
    }
    identity = _require_sha256(
        checked["event_identity"],
        field_name="governance event.event_identity",
    )
    if identity != canonical_sha256(observed_body):
        raise GovernanceContractError("governance event identity differs")
    return GovernanceEvent(checked)


def encode_governance_ledger(events: Sequence[GovernanceEvent]) -> bytes:
    """Encode validated events as canonical JSONL bytes with one LF per event."""
    if not isinstance(events, Sequence) or isinstance(
        events,
        (str, bytes, bytearray),
    ):
        raise GovernanceContractError("governance ledger events must be an array")
    verified: list[GovernanceEvent] = []
    seen: set[str] = set()
    for item in events:
        if type(item) is not GovernanceEvent:
            raise GovernanceContractError(
                "governance ledger encoding requires GovernanceEvent records"
            )
        event = _verify_event_record(item)
        if event["event_identity"] in seen:
            raise GovernanceContractError("governance ledger event identity repeats")
        seen.add(event["event_identity"])
        verified.append(event)
    return b"".join(canonical_json_bytes(event) + b"\n" for event in verified)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise GovernanceContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(token: str) -> None:
    raise GovernanceContractError(f"non-finite JSON constant is forbidden: {token}")


def decode_governance_ledger(payload: bytes) -> tuple[GovernanceEvent, ...]:
    """Decode strict canonical JSONL bytes without touching a filesystem."""
    if type(payload) is not bytes:
        raise GovernanceContractError("governance ledger payload must be bytes")
    if payload == b"":
        return ()
    if payload.startswith(b"\xef\xbb\xbf"):
        raise GovernanceContractError("governance ledger forbids a UTF-8 BOM")
    if b"\r" in payload:
        raise GovernanceContractError("governance ledger forbids CRLF or CR bytes")
    if not payload.endswith(b"\n"):
        raise GovernanceContractError("governance ledger requires a final LF")
    raw_lines = payload.split(b"\n")[:-1]
    if any(line == b"" for line in raw_lines):
        raise GovernanceContractError("governance ledger forbids blank lines")
    events: list[GovernanceEvent] = []
    seen: set[str] = set()
    for index, raw_line in enumerate(raw_lines):
        try:
            decoded = raw_line.decode("utf-8")
            parsed = json.loads(
                decoded,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
            )
        except GovernanceContractError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GovernanceContractError(
                f"governance ledger line {index} is not strict UTF-8 JSON"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise GovernanceContractError(
                f"governance ledger line {index} must be an object"
            )
        event = _verify_event_record(parsed)
        if raw_line != canonical_json_bytes(event):
            raise GovernanceContractError(
                f"governance ledger line {index} is not canonical JSON"
            )
        if event["event_identity"] in seen:
            raise GovernanceContractError("governance ledger event identity repeats")
        seen.add(event["event_identity"])
        _register_strict_decoded_event(event, raw_line + b"\n")
        events.append(event)
    return tuple(events)


_TERMINAL_STATES = frozenset(
    {"sequence_complete", "failed_closed", "revoked", "expired"}
)
_TERMINAL_EVENTS = {
    "fatal_stop": "failed_closed",
    "authorization_revoked": "revoked",
    "authorization_expired": "expired",
}


def _authorization_consumption_identity(
    authorization: SingleUseAuthorizationRecord,
    plan: RuntimePlanRecord,
) -> str:
    return canonical_sha256(
        {
            "schema_version": _CONSUMPTION_SCHEMA,
            "authorization_identity": authorization["authorization_identity"],
            "plan_identity": plan["plan_identity"],
            "consumption_index": 1,
        }
    )


def _operation_attempt_identity(
    plan: RuntimePlanRecord,
    operation_id: str,
) -> str:
    _operation_by_id(plan, operation_id)
    return canonical_sha256(
        {
            "schema_version": _ATTEMPT_SCHEMA,
            "plan_identity": plan["plan_identity"],
            "operation_id": operation_id,
            "attempt_index": 1,
        }
    )


def _authorization_resource_vector(
    authorization: SingleUseAuthorizationRecord,
) -> dict[str, int]:
    caps = authorization["resource_caps"]
    return {
        "state_generation_games": caps["state_generation_games"],
        "active_seconds": caps["active_seconds"],
        "teacher_nodes": caps["teacher_nodes"],
        "teacher_positive_search_labels": caps["teacher_positive_search_labels"],
        "supervised_updates": caps["supervised_updates"],
        "evaluation_games": caps["evaluation_games"],
    }


def _add_resources(
    left: Mapping[str, int],
    right: Mapping[str, int],
) -> dict[str, int]:
    return {key: left[key] + right[key] for key in _RESOURCE_KEYS}


def _resource_excess(
    value: Mapping[str, int],
    cap: Mapping[str, int],
) -> tuple[str, ...]:
    return tuple(key for key in _RESOURCE_KEYS if value[key] > cap[key])


def _event_scope(
    event: GovernanceEvent,
    *,
    operation: Mapping[str, Any] | None,
    plan: RuntimePlanRecord,
) -> None:
    if operation is None:
        expected = {
            "operation_id": None,
            "attempt_identity": None,
            "purpose": None,
            "seed": None,
        }
    else:
        expected = {
            "operation_id": operation["operation_id"],
            "attempt_identity": _operation_attempt_identity(
                plan,
                operation["operation_id"],
            ),
            "purpose": operation["purpose"],
            "seed": operation["seed"],
        }
    for key, expected_value in expected.items():
        _require_exact(
            event[key],
            expected_value,
            field_name=f"governance event.{key}",
        )


def _require_checkpoint_role(
    event: GovernanceEvent,
    role: str | None,
) -> Mapping[str, Any] | None:
    checkpoint = event["evidence"]["checkpoint"]
    if role is None:
        if checkpoint is not None:
            raise GovernanceContractError(
                "governance event unexpectedly carries a checkpoint"
            )
        return None
    if not isinstance(checkpoint, Mapping) or checkpoint["role"] != role:
        raise GovernanceContractError(
            f"governance event checkpoint role must be {role}"
        )
    return checkpoint


def _require_event_resources(
    event: GovernanceEvent,
    *,
    reservation: Mapping[str, int],
    observation_cap: Mapping[str, int],
    exact_observation: Mapping[str, int] | None = None,
) -> dict[str, int]:
    _require_exact(
        event["resource_reservation"],
        reservation,
        field_name="governance event.resource_reservation",
    )
    observation = _validate_resource_vector(
        event["resource_observation"],
        field_name="governance event.resource_observation",
    )
    if exact_observation is not None:
        _require_exact(
            observation,
            exact_observation,
            field_name="governance event.resource_observation",
        )
    excess = _resource_excess(observation, observation_cap)
    if excess:
        raise GovernanceContractError(
            "governance event observation exceeds its reservation: " + ", ".join(excess)
        )
    if observation["evaluation_games"] != 0:
        raise GovernanceContractError("evaluation games are prohibited")
    return observation


@dataclass(frozen=True, slots=True)
class _ReplayContext:
    plan: RuntimePlanRecord
    authorization: SingleUseAuthorizationRecord | None
    operation_status: Mapping[str, str]
    reservation_event_identities: Mapping[str, str]
    completion_event_identities: Mapping[str, str]
    open_reservation: Mapping[str, int] | None
    open_attempt_identity: str | None
    seed_cursors: Mapping[str, int]
    seed_checkpoints: Mapping[str, Mapping[str, Any]]
    event_identities: frozenset[str]
    authorization_registered: bool
    authorization_consumed: bool
    consumption_event_identity: str | None
    strict_decoded_head_event: GovernanceEvent | None
    strict_decoded_head_bytes: bytes | None


_REPLAY_CONTEXTS: weakref.WeakKeyDictionary[GovernanceReplay, _ReplayContext] = (
    weakref.WeakKeyDictionary()
)


def _validate_replay_capability(replay: GovernanceReplay) -> _ReplayContext:
    if type(replay) is not GovernanceReplay:
        raise GovernanceContractError("governance replay type differs")
    context = _REPLAY_CONTEXTS.get(replay)
    if context is None:
        raise GovernanceContractError(
            "governance replay was not produced by full genesis replay"
        )
    if replay.events:
        if replay.head_event_identity != replay.events[-1]["event_identity"]:
            raise GovernanceContractError("governance replay head differs")
    elif replay.head_event_identity is not None or replay.state != "no_events":
        raise GovernanceContractError("empty governance replay differs")
    if context.strict_decoded_head_event is None:
        if context.strict_decoded_head_bytes is not None:
            raise GovernanceContractError(
                "governance replay strict-decoder provenance differs"
            )
    elif (
        not replay.events
        or replay.events[-1] is not context.strict_decoded_head_event
        or _strict_decoded_event_bytes(context.strict_decoded_head_event)
        != context.strict_decoded_head_bytes
    ):
        raise GovernanceContractError(
            "governance replay strict-decoder head provenance differs"
        )
    return context


def _normal_event_common(
    event: GovernanceEvent,
    *,
    previous_identity: str | None,
) -> None:
    _require_exact(
        event["previous_event_identity"],
        previous_identity,
        field_name="governance event.previous_event_identity",
    )


def _event_expected_transition(
    event: GovernanceEvent,
    *,
    from_state: str,
    to_state: str,
) -> None:
    _require_exact(
        event["from_state"],
        from_state,
        field_name="governance event.from_state",
    )
    _require_exact(
        event["to_state"],
        to_state,
        field_name="governance event.to_state",
    )


def replay_governance_ledger(
    events: Sequence[GovernanceEvent | Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any] | None = None,
) -> GovernanceReplay:
    """Replay the complete ledger from genesis and fail closed on any drift."""
    checked_plan = _require_complete_plan(plan)
    checked_authorization = (
        None
        if authorization is None
        else verify_single_use_authorization(authorization, plan=checked_plan)
    )
    if not isinstance(events, Sequence) or isinstance(
        events,
        (str, bytes, bytearray),
    ):
        raise GovernanceContractError("governance replay events must be an array")

    verified_events: list[GovernanceEvent] = []
    state = "no_events"
    previous_identity: str | None = None
    previous_timestamp: datetime | None = None
    seen: set[str] = set()
    readiness_identity = _expected_readiness_identity(checked_plan)
    authorization_identity: str | None = None
    consumption_identity: str | None = None
    authorization_registered = False
    authorization_consumed = False
    consumption_event_identity: str | None = None
    operation_status = {
        operation_id: "not_started" for operation_id in _operation_ids(checked_plan)
    }
    reservation_event_identities: dict[str, str] = {}
    completion_event_identities: dict[str, str] = {}
    open_operation_id: str | None = None
    open_reservation: dict[str, int] | None = None
    open_attempt_identity: str | None = None
    charged = _zero_resource_vector()
    observed = _zero_resource_vector()
    violation: str | None = None
    seed_cursors = {seed: 0 for seed in _SEEDS}
    seed_checkpoints: dict[int, Mapping[str, Any]] = {}
    checkpoint_identities: set[str] = set()
    checkpoint_file_identities: set[str] = set()
    authorization_cap = (
        _zero_resource_vector()
        if checked_authorization is None
        else _authorization_resource_vector(checked_authorization)
    )
    authorization_times = (
        None
        if checked_authorization is None
        else {
            key: _parse_timestamp(
                checked_authorization[key],
                field_name=f"authorization.{key}",
            )
            for key in (
                "issued_at_utc",
                "not_before_utc",
                "consume_by_utc",
                "expires_at_utc",
            )
        }
    )
    strict_decoded_head_event: GovernanceEvent | None = None
    strict_decoded_head_bytes: bytes | None = None

    def claim_new_checkpoint(checkpoint: Mapping[str, Any]) -> None:
        identity = checkpoint["identity"]
        file_identity = checkpoint["file_sha256"]
        if (
            identity in checkpoint_identities
            or file_identity in checkpoint_file_identities
        ):
            raise GovernanceContractError(
                "checkpoint reference was reused across progress or operation scope"
            )
        checkpoint_identities.add(identity)
        checkpoint_file_identities.add(file_identity)

    for expected_sequence, candidate in enumerate(events):
        strict_candidate_bytes = (
            _strict_decoded_event_bytes(candidate)
            if type(candidate) is GovernanceEvent
            else None
        )
        verified_event = _verify_event_record(candidate)
        if strict_candidate_bytes is None:
            event = verified_event
            strict_decoded_head_event = None
            strict_decoded_head_bytes = None
        else:
            assert type(candidate) is GovernanceEvent
            if candidate.to_dict() != verified_event.to_dict():
                raise GovernanceContractError(
                    "strict-decoded governance event changed during verification"
                )
            event = candidate
            strict_decoded_head_event = event
            strict_decoded_head_bytes = strict_candidate_bytes
        if state in _TERMINAL_STATES:
            raise GovernanceContractError(
                "terminal governance state has no outgoing events"
            )
        if event["sequence"] != expected_sequence:
            raise GovernanceContractError("governance event sequence is not contiguous")
        if event["event_identity"] in seen:
            raise GovernanceContractError("governance event identity repeats")
        seen.add(event["event_identity"])
        _normal_event_common(event, previous_identity=previous_identity)
        timestamp = _parse_timestamp(
            event["timestamp_utc"],
            field_name="governance event.timestamp_utc",
        )
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise GovernanceContractError("governance event time moved backwards")
        previous_timestamp = timestamp
        for field_name, expected_value in (
            ("experiment_id", _EXPERIMENT_ID),
            ("proposal_identity", _PROPOSAL_IDENTITY),
            ("plan_identity", checked_plan["plan_identity"]),
            ("readiness_identity", readiness_identity),
        ):
            _require_exact(
                event[field_name],
                expected_value,
                field_name=f"governance event.{field_name}",
            )
        if event["from_state"] != state:
            raise GovernanceContractError(
                "governance event from_state differs from replay"
            )

        event_type = event["event_type"]
        authorization_deadline = (
            None
            if authorization_times is None
            else authorization_times[
                "expires_at_utc" if authorization_consumed else "consume_by_utc"
            ]
        )
        if event_type in _TERMINAL_EVENTS:
            target = _TERMINAL_EVENTS[event_type]
            _event_expected_transition(event, from_state=state, to_state=target)
            if event["reason_code"] is None:
                raise GovernanceContractError(
                    "terminal governance event needs a reason_code"
                )
            expected_auth = (
                checked_authorization["authorization_identity"]
                if authorization_registered and checked_authorization is not None
                else None
            )
            _require_exact(
                event["authorization_identity"],
                expected_auth,
                field_name="governance event.authorization_identity",
            )
            _require_exact(
                event["authorization_consumption_identity"],
                consumption_identity,
                field_name="governance event.authorization_consumption_identity",
            )
            operation = (
                None
                if open_operation_id is None
                else _operation_by_id(checked_plan, open_operation_id)
            )
            _event_scope(event, operation=operation, plan=checked_plan)
            _require_exact(
                event["prerequisite_event_identity"],
                previous_identity,
                field_name="governance event.prerequisite_event_identity",
            )
            _require_exact(
                event["resource_reservation"],
                _zero_resource_vector(),
                field_name="governance event.resource_reservation",
            )
            terminal_observation = _validate_resource_vector(
                event["resource_observation"],
                field_name="governance event.resource_observation",
            )
            if open_reservation is None and any(terminal_observation.values()):
                raise GovernanceContractError(
                    "terminal resource observation requires an open reservation"
                )
            if event_type != "fatal_stop" and any(terminal_observation.values()):
                raise GovernanceContractError(
                    "revocation or expiry cannot carry resource observations"
                )
            if event_type == "authorization_revoked" and not authorization_registered:
                raise GovernanceContractError(
                    "authorization cannot be revoked before registration"
                )
            if event_type == "authorization_expired":
                if (
                    not authorization_registered
                    or authorization_deadline is None
                    or timestamp <= authorization_deadline
                ):
                    raise GovernanceContractError(
                        "authorization_expired must follow its active deadline"
                    )
            if (
                event_type != "authorization_expired"
                and authorization_deadline is not None
                and timestamp > authorization_deadline
            ):
                raise GovernanceContractError(
                    "only authorization_expired is allowed after its active deadline"
                )
            observed = _add_resources(observed, terminal_observation)
            operation_excess = (
                ()
                if open_reservation is None
                else _resource_excess(terminal_observation, open_reservation)
            )
            total_excess = _resource_excess(observed, authorization_cap)
            if terminal_observation["evaluation_games"] != 0:
                violation = "evaluation_games_nonzero"
            elif operation_excess:
                violation = "resource_observation_exceeds_reservation"
            elif total_excess:
                violation = "authorized_resource_cap_exceeded"
            _require_checkpoint_role(event, None)
            state = target
            previous_identity = event["event_identity"]
            verified_events.append(event)
            continue

        if event["reason_code"] is not None:
            raise GovernanceContractError(
                "normal governance event cannot have a reason_code"
            )
        if authorization_deadline is not None and timestamp > authorization_deadline:
            raise GovernanceContractError(
                "only authorization_expired is allowed after its active deadline"
            )

        expected_event_auth = (
            checked_authorization["authorization_identity"]
            if authorization_registered and checked_authorization is not None
            else None
        )
        if event_type == "authorization_registered":
            if checked_authorization is None:
                raise GovernanceContractError(
                    "authorization_registered requires an authorization record"
                )
            expected_event_auth = checked_authorization["authorization_identity"]
        _require_exact(
            event["authorization_identity"],
            expected_event_auth,
            field_name="governance event.authorization_identity",
        )

        if event_type == "plan_frozen":
            _event_expected_transition(
                event, from_state="no_events", to_state="plan_frozen"
            )
            _event_scope(event, operation=None, plan=checked_plan)
            _require_exact(
                event["authorization_consumption_identity"],
                None,
                field_name="governance event.authorization_consumption_identity",
            )
            _require_exact(
                event["prerequisite_event_identity"],
                None,
                field_name="governance event.prerequisite_event_identity",
            )
            _require_event_resources(
                event,
                reservation=_zero_resource_vector(),
                observation_cap=_zero_resource_vector(),
                exact_observation=_zero_resource_vector(),
            )
            _require_checkpoint_role(event, None)
            state = "plan_frozen"
        elif event_type == "readiness_frozen":
            _event_expected_transition(
                event, from_state="plan_frozen", to_state="ready_unauthorized"
            )
            _event_scope(event, operation=None, plan=checked_plan)
            _require_exact(
                event["authorization_consumption_identity"],
                None,
                field_name="governance event.authorization_consumption_identity",
            )
            _require_exact(
                event["prerequisite_event_identity"],
                previous_identity,
                field_name="governance event.prerequisite_event_identity",
            )
            _require_event_resources(
                event,
                reservation=_zero_resource_vector(),
                observation_cap=_zero_resource_vector(),
                exact_observation=_zero_resource_vector(),
            )
            _require_checkpoint_role(event, None)
            state = "ready_unauthorized"
        elif event_type == "authorization_registered":
            _event_expected_transition(
                event, from_state="ready_unauthorized", to_state="authorized_unconsumed"
            )
            if authorization_registered or checked_authorization is None:
                raise GovernanceContractError("authorization registration repeats")
            _event_scope(event, operation=None, plan=checked_plan)
            _require_exact(
                event["authorization_consumption_identity"],
                None,
                field_name="governance event.authorization_consumption_identity",
            )
            _require_exact(
                event["prerequisite_event_identity"],
                previous_identity,
                field_name="governance event.prerequisite_event_identity",
            )
            _require_event_resources(
                event,
                reservation=_zero_resource_vector(),
                observation_cap=_zero_resource_vector(),
                exact_observation=_zero_resource_vector(),
            )
            _require_checkpoint_role(event, None)
            authorization_registered = True
            authorization_identity = checked_authorization["authorization_identity"]
            state = "authorized_unconsumed"
        elif event_type == "authorization_consumed":
            _event_expected_transition(
                event,
                from_state="authorized_unconsumed",
                to_state="authorization_consumed",
            )
            if (
                not authorization_registered
                or authorization_consumed
                or checked_authorization is None
                or authorization_times is None
            ):
                raise GovernanceContractError(
                    "authorization consumption is out of order"
                )
            if (
                not authorization_times["not_before_utc"]
                <= timestamp
                <= authorization_times["consume_by_utc"]
            ):
                raise GovernanceContractError(
                    "authorization consumption time is outside its window"
                )
            expected_consumption = _authorization_consumption_identity(
                checked_authorization, checked_plan
            )
            _require_exact(
                event["authorization_consumption_identity"],
                expected_consumption,
                field_name="governance event.authorization_consumption_identity",
            )
            _event_scope(event, operation=None, plan=checked_plan)
            _require_exact(
                event["prerequisite_event_identity"],
                previous_identity,
                field_name="governance event.prerequisite_event_identity",
            )
            _require_event_resources(
                event,
                reservation=_zero_resource_vector(),
                observation_cap=_zero_resource_vector(),
                exact_observation=_zero_resource_vector(),
            )
            _require_checkpoint_role(event, None)
            authorization_consumed = True
            consumption_identity = expected_consumption
            consumption_event_identity = event["event_identity"]
            state = "authorization_consumed"
        else:
            if not authorization_consumed or checked_authorization is None:
                raise GovernanceContractError(
                    "operation event requires prior authorization consumption"
                )
            _require_exact(
                event["authorization_consumption_identity"],
                consumption_identity,
                field_name="governance event.authorization_consumption_identity",
            )
            if event_type == "sequence_completed":
                _event_expected_transition(
                    event,
                    from_state=f"seed_{_SEEDS[-1]}_complete",
                    to_state="sequence_complete",
                )
                _event_scope(event, operation=None, plan=checked_plan)
                _require_exact(
                    event["prerequisite_event_identity"],
                    completion_event_identities.get(f"seed-{_SEEDS[-1]}"),
                    field_name="governance event.prerequisite_event_identity",
                )
                _require_event_resources(
                    event,
                    reservation=_zero_resource_vector(),
                    observation_cap=_zero_resource_vector(),
                    exact_observation=_zero_resource_vector(),
                )
                _require_checkpoint_role(event, None)
                if (
                    any(status != "complete" for status in operation_status.values())
                    or open_operation_id is not None
                ):
                    raise GovernanceContractError(
                        "sequence completed before all operations"
                    )
                if (
                    observed["supervised_updates"] != 6_721
                    or observed["evaluation_games"] != 0
                ):
                    raise GovernanceContractError(
                        "sequence supervised update totals differ"
                    )
                state = "sequence_complete"
                previous_identity = event["event_identity"]
                verified_events.append(event)
                continue
            operation_id = event["operation_id"]
            if not isinstance(operation_id, str):
                raise GovernanceContractError("operation event lacks operation_id")
            operation = _operation_by_id(checked_plan, operation_id)
            _event_scope(event, operation=operation, plan=checked_plan)
            reservation = _thaw(operation["resource_reservation"])

            if event_type in {
                "state_generation_reserved",
                "teacher_reserved",
                "smoke_reserved",
                "seed_reserved",
            }:
                expected_by_type = {
                    "state_generation_reserved": (
                        "state-generation",
                        "authorization_consumed",
                        "state_generation_running",
                    ),
                    "teacher_reserved": (
                        "offline-d9-teacher-labeling",
                        "state_frozen",
                        "teacher_running",
                    ),
                    "smoke_reserved": (
                        "supervised-smoke",
                        "corpus_frozen",
                        "smoke_running",
                    ),
                    "seed_reserved": (
                        operation_id,
                        f"seed_{operation['seed']}_initialized",
                        f"seed_{operation['seed']}_running",
                    ),
                }
                expected_operation, expected_from, expected_to = expected_by_type[
                    event_type
                ]
                if operation_id != expected_operation:
                    raise GovernanceContractError("reservation event operation differs")
                _event_expected_transition(
                    event, from_state=expected_from, to_state=expected_to
                )
                if (
                    operation_status[operation_id] != "not_started"
                    or open_operation_id is not None
                ):
                    raise GovernanceContractError(
                        "operation reservation repeats or overlaps"
                    )
                if event_type == "state_generation_reserved":
                    prerequisite = consumption_event_identity
                elif event_type == "teacher_reserved":
                    prerequisite = completion_event_identities.get("state-freeze")
                elif event_type == "smoke_reserved":
                    prerequisite = completion_event_identities.get("corpus-freeze")
                else:
                    prerequisite = completion_event_identities.get(
                        f"{operation_id}:init"
                    )
                _require_exact(
                    event["prerequisite_event_identity"],
                    prerequisite,
                    field_name="governance event.prerequisite_event_identity",
                )
                _require_event_resources(
                    event,
                    reservation=reservation,
                    observation_cap=_zero_resource_vector(),
                    exact_observation=_zero_resource_vector(),
                )
                _require_checkpoint_role(event, None)
                charged_after = _add_resources(charged, reservation)
                excess = _resource_excess(charged_after, authorization_cap)
                if excess:
                    raise GovernanceContractError(
                        "operation reservation exceeds authorization: "
                        + ", ".join(excess)
                    )
                charged = charged_after
                open_operation_id = operation_id
                open_reservation = reservation
                open_attempt_identity = event["attempt_identity"]
                operation_status[operation_id] = "reserved"
                reservation_event_identities[operation_id] = event["event_identity"]
                state = expected_to
            elif event_type in {
                "state_generation_completed",
                "teacher_completed",
                "smoke_passed",
                "seed_completed",
            }:
                expected_by_type = {
                    "state_generation_completed": (
                        "state-generation",
                        "state_generation_running",
                        "state_generated",
                    ),
                    "teacher_completed": (
                        "offline-d9-teacher-labeling",
                        "teacher_running",
                        "teacher_labeled",
                    ),
                    "smoke_passed": (
                        "supervised-smoke",
                        "smoke_running",
                        "smoke_passed",
                    ),
                    "seed_completed": (
                        operation_id,
                        f"seed_{operation['seed']}_running",
                        f"seed_{operation['seed']}_complete",
                    ),
                }
                expected_operation, expected_from, expected_to = expected_by_type[
                    event_type
                ]
                if operation_id != expected_operation:
                    raise GovernanceContractError("completion event operation differs")
                _event_expected_transition(
                    event, from_state=expected_from, to_state=expected_to
                )
                if (
                    open_operation_id != operation_id
                    or operation_status[operation_id] != "reserved"
                    or open_reservation is None
                ):
                    raise GovernanceContractError(
                        "operation completion lacks its reservation"
                    )
                _require_exact(
                    event["prerequisite_event_identity"],
                    reservation_event_identities[operation_id]
                    if event_type != "seed_completed"
                    else previous_identity,
                    field_name="governance event.prerequisite_event_identity",
                )
                checkpoint_role: str | None = None
                required_supervised_updates: int | None = None
                if event_type == "smoke_passed":
                    required_supervised_updates = 1
                    checkpoint_role = checked_plan["checkpoint_policy"]["smoke_role"]
                elif event_type == "seed_completed":
                    seed = operation["seed"]
                    if seed_cursors[seed] != 2_128:
                        raise GovernanceContractError(
                            "seed completion requires nineteen 112-update checkpoints"
                        )
                    required_supervised_updates = 2_240
                    checkpoint_role = checked_plan["checkpoint_policy"]["complete_role"]
                    seed_cursors[seed] = 2_240
                observation = _require_event_resources(
                    event,
                    reservation=_zero_resource_vector(),
                    observation_cap=open_reservation,
                    exact_observation=None,
                )
                if required_supervised_updates is not None:
                    _require_exact(
                        observation,
                        _resource_vector(
                            active_seconds=observation["active_seconds"],
                            supervised_updates=required_supervised_updates,
                        ),
                        field_name="governance event.resource_observation",
                    )
                checkpoint = _require_checkpoint_role(event, checkpoint_role)
                if checkpoint is not None:
                    claim_new_checkpoint(checkpoint)
                observed_after = _add_resources(observed, observation)
                excess = _resource_excess(observed_after, authorization_cap)
                if excess:
                    raise GovernanceContractError(
                        "observed resources exceed authorization: " + ", ".join(excess)
                    )
                observed = observed_after
                operation_status[operation_id] = "complete"
                completion_event_identities[operation_id] = event["event_identity"]
                open_operation_id = None
                open_reservation = None
                open_attempt_identity = None
                state = expected_to
            elif event_type in {"state_frozen", "corpus_frozen"}:
                expected_operation = (
                    "state-freeze" if event_type == "state_frozen" else "corpus-freeze"
                )
                expected_from = (
                    "state_generated"
                    if event_type == "state_frozen"
                    else "teacher_labeled"
                )
                expected_to = (
                    "state_frozen" if event_type == "state_frozen" else "corpus_frozen"
                )
                if operation_id != expected_operation:
                    raise GovernanceContractError("governance freeze operation differs")
                _event_expected_transition(
                    event, from_state=expected_from, to_state=expected_to
                )
                prerequisite_operation = operation["prerequisite_operation_id"]
                prerequisite = completion_event_identities.get(prerequisite_operation)
                _require_exact(
                    event["prerequisite_event_identity"],
                    prerequisite,
                    field_name="governance event.prerequisite_event_identity",
                )
                if operation_status[operation_id] != "not_started" or any(
                    reservation.values()
                ):
                    raise GovernanceContractError(
                        "governance freeze reservation differs"
                    )
                _require_event_resources(
                    event,
                    reservation=_zero_resource_vector(),
                    observation_cap=_zero_resource_vector(),
                    exact_observation=_zero_resource_vector(),
                )
                _require_checkpoint_role(event, None)
                operation_status[operation_id] = "complete"
                completion_event_identities[operation_id] = event["event_identity"]
                state = expected_to
            elif event_type == "seed_fresh_init_committed":
                seed = operation["seed"]
                if seed is None or operation_id != f"seed-{seed}":
                    raise GovernanceContractError("seed initialization scope differs")
                seed_index = _SEEDS.index(seed)
                expected_from = (
                    "smoke_passed"
                    if seed_index == 0
                    else f"seed_{_SEEDS[seed_index - 1]}_complete"
                )
                _event_expected_transition(
                    event, from_state=expected_from, to_state=f"seed_{seed}_initialized"
                )
                prerequisite_operation = operation["prerequisite_operation_id"]
                prerequisite = completion_event_identities.get(prerequisite_operation)
                _require_exact(
                    event["prerequisite_event_identity"],
                    prerequisite,
                    field_name="governance event.prerequisite_event_identity",
                )
                if (
                    operation_status[operation_id] != "not_started"
                    or seed_cursors[seed] != 0
                ):
                    raise GovernanceContractError("seed initialization repeats")
                _require_event_resources(
                    event,
                    reservation=_zero_resource_vector(),
                    observation_cap=_zero_resource_vector(),
                    exact_observation=_zero_resource_vector(),
                )
                checkpoint = _require_checkpoint_role(
                    event, checked_plan["checkpoint_policy"]["state0_role"]
                )
                assert checkpoint is not None
                claim_new_checkpoint(checkpoint)
                seed_checkpoints[seed] = checkpoint
                completion_event_identities[f"{operation_id}:init"] = event[
                    "event_identity"
                ]
                state = f"seed_{seed}_initialized"
            elif event_type == "seed_checkpointed":
                seed = operation["seed"]
                _event_expected_transition(
                    event,
                    from_state=f"seed_{seed}_running",
                    to_state=f"seed_{seed}_running",
                )
                if (
                    open_operation_id != operation_id
                    or operation_status[operation_id] != "reserved"
                ):
                    raise GovernanceContractError(
                        "seed checkpoint lacks active reservation"
                    )
                next_cursor = (
                    seed_cursors[seed]
                    + checked_plan["checkpoint_policy"]["interval_updates"]
                )
                if next_cursor > 2_128:
                    raise GovernanceContractError("seed checkpoint cursor exceeds 2128")
                _require_exact(
                    event["prerequisite_event_identity"],
                    previous_identity,
                    field_name="governance event.prerequisite_event_identity",
                )
                _require_event_resources(
                    event,
                    reservation=_zero_resource_vector(),
                    observation_cap=_zero_resource_vector(),
                    exact_observation=_zero_resource_vector(),
                )
                checkpoint = _require_checkpoint_role(
                    event, checked_plan["checkpoint_policy"]["latest_role"]
                )
                assert checkpoint is not None
                claim_new_checkpoint(checkpoint)
                prior_checkpoint = seed_checkpoints.get(seed)
                if (
                    prior_checkpoint is not None
                    and checkpoint["identity"] == prior_checkpoint["identity"]
                ):
                    raise GovernanceContractError(
                        "seed checkpoint identity did not advance"
                    )
                seed_cursors[seed] = next_cursor
                seed_checkpoints[seed] = checkpoint
                state = f"seed_{seed}_running"
            elif event_type == "seed_paused":
                seed = operation["seed"]
                _event_expected_transition(
                    event,
                    from_state=f"seed_{seed}_running",
                    to_state=f"seed_{seed}_paused",
                )
                if (
                    open_operation_id != operation_id
                    or operation_status[operation_id] != "reserved"
                ):
                    raise GovernanceContractError("seed pause lacks active reservation")
                _require_exact(
                    event["prerequisite_event_identity"],
                    previous_identity,
                    field_name="governance event.prerequisite_event_identity",
                )
                _require_event_resources(
                    event,
                    reservation=_zero_resource_vector(),
                    observation_cap=_zero_resource_vector(),
                    exact_observation=_zero_resource_vector(),
                )
                checkpoint = _require_checkpoint_role(
                    event, checked_plan["checkpoint_policy"]["latest_role"]
                )
                if checkpoint != seed_checkpoints.get(seed):
                    raise GovernanceContractError(
                        "seed pause must reuse the exact latest checkpoint ref"
                    )
                state = f"seed_{seed}_paused"
            elif event_type == "seed_resumed":
                seed = operation["seed"]
                _event_expected_transition(
                    event,
                    from_state=f"seed_{seed}_paused",
                    to_state=f"seed_{seed}_running",
                )
                if (
                    open_operation_id != operation_id
                    or operation_status[operation_id] != "reserved"
                ):
                    raise GovernanceContractError(
                        "seed resume lacks active reservation"
                    )
                _require_exact(
                    event["prerequisite_event_identity"],
                    previous_identity,
                    field_name="governance event.prerequisite_event_identity",
                )
                _require_event_resources(
                    event,
                    reservation=_zero_resource_vector(),
                    observation_cap=_zero_resource_vector(),
                    exact_observation=_zero_resource_vector(),
                )
                checkpoint = _require_checkpoint_role(
                    event, checked_plan["checkpoint_policy"]["latest_role"]
                )
                if checkpoint != seed_checkpoints.get(seed):
                    raise GovernanceContractError(
                        "seed resume must reuse the exact paused checkpoint ref"
                    )
                state = f"seed_{seed}_running"
            else:
                raise GovernanceContractError(
                    f"governance transition is not allowed: {event_type}"
                )

        previous_identity = event["event_identity"]
        verified_events.append(event)

    terminal = state in _TERMINAL_STATES
    restart_requires_failure_closure = (
        open_operation_id is not None and state.endswith("_running") and not terminal
    )
    ledger = ResourceLedger(
        charged_totals=charged,
        observed_totals=observed,
        open_operation_id=open_operation_id,
        restart_requires_failure_closure=restart_requires_failure_closure,
        violation=violation,
    )
    completed_operations = tuple(
        operation_id
        for operation_id in _operation_ids(checked_plan)
        if operation_status[operation_id] == "complete"
    )
    replay = GovernanceReplay(
        state=state,
        events=tuple(verified_events),
        head_event_identity=previous_identity,
        experiment_id=_EXPERIMENT_ID,
        proposal_identity=_PROPOSAL_IDENTITY,
        plan_identity=checked_plan["plan_identity"],
        readiness_identity=readiness_identity,
        authorization_identity=authorization_identity,
        authorization_consumption_identity=consumption_identity,
        resource_ledger=ledger,
        terminal=terminal,
        completed_operations=completed_operations,
    )
    _REPLAY_CONTEXTS[replay] = _ReplayContext(
        plan=checked_plan,
        authorization=checked_authorization,
        operation_status=_freeze(operation_status, field_name="operation status"),
        reservation_event_identities=_freeze(
            reservation_event_identities,
            field_name="reservation event identities",
        ),
        completion_event_identities=_freeze(
            completion_event_identities,
            field_name="completion event identities",
        ),
        open_reservation=(
            None
            if open_reservation is None
            else _freeze(open_reservation, field_name="open reservation")
        ),
        open_attempt_identity=open_attempt_identity,
        seed_cursors=_freeze(
            {str(seed): cursor for seed, cursor in seed_cursors.items()},
            field_name="seed cursors",
        ),
        seed_checkpoints=_freeze(
            {str(seed): checkpoint for seed, checkpoint in seed_checkpoints.items()},
            field_name="seed checkpoints",
        ),
        event_identities=frozenset(seen),
        authorization_registered=authorization_registered,
        authorization_consumed=authorization_consumed,
        consumption_event_identity=consumption_event_identity,
        strict_decoded_head_event=strict_decoded_head_event,
        strict_decoded_head_bytes=strict_decoded_head_bytes,
    )
    return replay


_PRODUCTION_PLAN_TOKEN = object()
_PRODUCTION_AUTHORIZATION_TOKEN = object()
_PRODUCTION_PENDING_AUTH_TOKEN = object()
_PRODUCTION_CONSUMED_AUTH_TOKEN = object()
_PRODUCTION_PENDING_OPERATION_TOKEN = object()
_PRODUCTION_OPERATION_TOKEN = object()
_TEST_PLAN_TOKEN = object()
_TEST_AUTHORIZATION_TOKEN = object()
_TEST_PENDING_AUTH_TOKEN = object()
_TEST_CONSUMED_AUTH_TOKEN = object()
_TEST_PENDING_OPERATION_TOKEN = object()
_TEST_OPERATION_TOKEN = object()


class _OpaquePermit:
    __slots__ = ("__weakref__",)

    _creation_token: object
    _description: str

    def __init__(self, token: object) -> None:
        if token is not self._creation_token:
            raise GovernanceContractError(
                f"{self._description} must be issued by its controlled issuer"
            )

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        raise TypeError(f"{self._description} cannot be serialized")

    def __copy__(self) -> Any:
        raise TypeError(f"{self._description} cannot be copied")

    def __deepcopy__(self, memo: Any) -> Any:
        del memo
        raise TypeError(f"{self._description} cannot be copied")


class ProductionRuntimePlanPermit(_OpaquePermit):
    """Opaque production plan capability; C3 has no issuer for it."""

    __slots__ = ()
    _creation_token = _PRODUCTION_PLAN_TOKEN
    _description = "production runtime plan permit"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("production runtime plan permit cannot be subclassed")


class ProductionAuthorizationPermit(_OpaquePermit):
    """Opaque production authorization capability; C3 has no issuer for it."""

    __slots__ = ()
    _creation_token = _PRODUCTION_AUTHORIZATION_TOKEN
    _description = "production authorization permit"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("production authorization permit cannot be subclassed")


class PendingAuthorizationConsumption(_OpaquePermit):
    """Write-ahead authorization consumption awaiting exact persisted replay."""

    __slots__ = ()
    _creation_token = _PRODUCTION_PENDING_AUTH_TOKEN
    _description = "pending authorization consumption"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("pending authorization consumption cannot be subclassed")


class ConsumedAuthorizationPermit(_OpaquePermit):
    """Opaque capability representing one replay-proven authorization spend."""

    __slots__ = ()
    _creation_token = _PRODUCTION_CONSUMED_AUTH_TOKEN
    _description = "consumed authorization permit"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("consumed authorization permit cannot be subclassed")


class PendingOperationReservation(_OpaquePermit):
    """Write-ahead operation reservation awaiting exact persisted replay."""

    __slots__ = ()
    _creation_token = _PRODUCTION_PENDING_OPERATION_TOKEN
    _description = "pending operation reservation"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("pending operation reservation cannot be subclassed")


class ProductionOperationPermit(_OpaquePermit):
    """One-use operation capability; C3 has no production issuer for it."""

    __slots__ = ()
    _creation_token = _PRODUCTION_OPERATION_TOKEN
    _description = "production operation permit"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("production operation permit cannot be subclassed")


class _TestRuntimePlanPermit(_OpaquePermit):
    __slots__ = ()
    _creation_token = _TEST_PLAN_TOKEN
    _description = "test runtime plan permit"


class _TestAuthorizationPermit(_OpaquePermit):
    __slots__ = ()
    _creation_token = _TEST_AUTHORIZATION_TOKEN
    _description = "test authorization permit"


class _TestPendingAuthorizationConsumption(_OpaquePermit):
    __slots__ = ()
    _creation_token = _TEST_PENDING_AUTH_TOKEN
    _description = "test pending authorization consumption"


class _TestConsumedAuthorizationPermit(_OpaquePermit):
    __slots__ = ()
    _creation_token = _TEST_CONSUMED_AUTH_TOKEN
    _description = "test consumed authorization permit"


class _TestPendingOperationReservation(_OpaquePermit):
    __slots__ = ()
    _creation_token = _TEST_PENDING_OPERATION_TOKEN
    _description = "test pending operation reservation"


class _TestOperationPermit(_OpaquePermit):
    __slots__ = ()
    _creation_token = _TEST_OPERATION_TOKEN
    _description = "test operation permit"


@dataclass(slots=True)
class _PlanPermitContext:
    plan: RuntimePlanRecord
    spent: bool = False


@dataclass(slots=True)
class _AuthorizationPermitContext:
    plan_identity: str
    authorization: SingleUseAuthorizationRecord
    spent: bool = False


@dataclass(frozen=True, slots=True)
class _PendingAuthorizationContext:
    plan: RuntimePlanRecord
    authorization: SingleUseAuthorizationRecord
    predecessor_identity: str
    event: GovernanceEvent
    event_bytes: bytes


@dataclass(slots=True)
class _ConsumedAuthorizationContext:
    plan: RuntimePlanRecord
    authorization: SingleUseAuthorizationRecord
    consumption_event_identity: str
    pending_reservation_identity: str | None = None
    last_reservation_identity: str | None = None
    unconsumed_operation_permit_identity: str | None = None


@dataclass(frozen=True, slots=True)
class _PendingOperationContext:
    consumed_permit: object
    plan: RuntimePlanRecord
    authorization: SingleUseAuthorizationRecord
    operation_id: str
    attempt_identity: str
    predecessor_identity: str
    event: GovernanceEvent
    event_bytes: bytes


@dataclass(slots=True)
class _OperationPermitContext:
    consumed_permit: object
    plan_identity: str
    authorization_identity: str
    consumption_identity: str
    operation_id: str
    attempt_identity: str
    purpose: str
    seed: int | None
    reservation_event_identity: str
    replay_head_identity: str
    spent: bool = False


_PRODUCTION_PLAN_CONTEXTS: weakref.WeakKeyDictionary[
    ProductionRuntimePlanPermit,
    _PlanPermitContext,
] = weakref.WeakKeyDictionary()
_PRODUCTION_AUTH_CONTEXTS: weakref.WeakKeyDictionary[
    ProductionAuthorizationPermit,
    _AuthorizationPermitContext,
] = weakref.WeakKeyDictionary()
_PRODUCTION_PENDING_AUTH_CONTEXTS: weakref.WeakKeyDictionary[
    PendingAuthorizationConsumption,
    _PendingAuthorizationContext,
] = weakref.WeakKeyDictionary()
_PRODUCTION_CONSUMED_CONTEXTS: weakref.WeakKeyDictionary[
    ConsumedAuthorizationPermit,
    _ConsumedAuthorizationContext,
] = weakref.WeakKeyDictionary()
_PRODUCTION_PENDING_OPERATION_CONTEXTS: weakref.WeakKeyDictionary[
    PendingOperationReservation,
    _PendingOperationContext,
] = weakref.WeakKeyDictionary()
_PRODUCTION_OPERATION_CONTEXTS: weakref.WeakKeyDictionary[
    ProductionOperationPermit,
    _OperationPermitContext,
] = weakref.WeakKeyDictionary()

_TEST_PLAN_CONTEXTS: weakref.WeakKeyDictionary[
    _TestRuntimePlanPermit,
    _PlanPermitContext,
] = weakref.WeakKeyDictionary()
_TEST_AUTH_CONTEXTS: weakref.WeakKeyDictionary[
    _TestAuthorizationPermit,
    _AuthorizationPermitContext,
] = weakref.WeakKeyDictionary()
_TEST_PENDING_AUTH_CONTEXTS: weakref.WeakKeyDictionary[
    _TestPendingAuthorizationConsumption,
    _PendingAuthorizationContext,
] = weakref.WeakKeyDictionary()
_TEST_CONSUMED_CONTEXTS: weakref.WeakKeyDictionary[
    _TestConsumedAuthorizationPermit,
    _ConsumedAuthorizationContext,
] = weakref.WeakKeyDictionary()
_TEST_PENDING_OPERATION_CONTEXTS: weakref.WeakKeyDictionary[
    _TestPendingOperationReservation,
    _PendingOperationContext,
] = weakref.WeakKeyDictionary()
_TEST_OPERATION_CONTEXTS: weakref.WeakKeyDictionary[
    _TestOperationPermit,
    _OperationPermitContext,
] = weakref.WeakKeyDictionary()


def _make_event(
    *,
    sequence: int,
    timestamp_utc: str,
    event_type: str,
    from_state: str,
    to_state: str,
    plan: RuntimePlanRecord,
    readiness_identity: str,
    authorization_identity: str | None,
    authorization_consumption_identity: str | None,
    operation: Mapping[str, Any] | None,
    prerequisite_event_identity: str | None,
    evidence: Mapping[str, Any] | None,
    resource_reservation: Mapping[str, int] | None,
    resource_observation: Mapping[str, int] | None,
    reason_code: str | None,
    previous_event_identity: str | None,
) -> GovernanceEvent:
    body = {
        "schema_version": _EVENT_SCHEMA,
        "sequence": sequence,
        "timestamp_utc": timestamp_utc,
        "event_type": event_type,
        "from_state": from_state,
        "to_state": to_state,
        "experiment_id": _EXPERIMENT_ID,
        "proposal_identity": _PROPOSAL_IDENTITY,
        "plan_identity": plan["plan_identity"],
        "readiness_identity": readiness_identity,
        "authorization_identity": authorization_identity,
        "authorization_consumption_identity": authorization_consumption_identity,
        "operation_id": None if operation is None else operation["operation_id"],
        "attempt_identity": (
            None
            if operation is None
            else _operation_attempt_identity(plan, operation["operation_id"])
        ),
        "purpose": None if operation is None else operation["purpose"],
        "seed": None if operation is None else operation["seed"],
        "prerequisite_event_identity": prerequisite_event_identity,
        "evidence": (
            {"inputs": [], "outputs": [], "checkpoint": None}
            if evidence is None
            else _snapshot(evidence, field_name="event evidence")
        ),
        "resource_reservation": (
            _zero_resource_vector()
            if resource_reservation is None
            else dict(resource_reservation)
        ),
        "resource_observation": (
            _zero_resource_vector()
            if resource_observation is None
            else dict(resource_observation)
        ),
        "reason_code": reason_code,
        "previous_event_identity": previous_event_identity,
    }
    return _verify_event_record({**body, "event_identity": canonical_sha256(body)})


def _event_bytes(event: GovernanceEvent) -> bytes:
    return encode_governance_ledger((event,))


def _prepare_authorization_consumption_core(
    plan_permit: object,
    authorization_permit: object,
    replay: GovernanceReplay,
    *,
    timestamp_utc: str,
    plan_type: type,
    authorization_type: type,
    pending_type: type,
    pending_token: object,
    plan_contexts: Mapping[object, _PlanPermitContext],
    authorization_contexts: Mapping[object, _AuthorizationPermitContext],
    pending_contexts: weakref.WeakKeyDictionary,
) -> tuple[object, bytes]:
    if (
        type(plan_permit) is not plan_type
        or type(authorization_permit) is not authorization_type
    ):
        raise GovernanceContractError(
            "authorization consumption requires exact issued permit types"
        )
    plan_context = plan_contexts.get(plan_permit)
    authorization_context = authorization_contexts.get(authorization_permit)
    if plan_context is None or authorization_context is None:
        raise GovernanceContractError(
            "authorization consumption lacks issued permit registry context"
        )
    replay_context = _validate_replay_capability(replay)
    if (
        plan_context.spent
        or authorization_context.spent
        or replay.state != "authorized_unconsumed"
        or replay.terminal
        or replay.resource_ledger.open_operation_id is not None
        or replay_context.plan != plan_context.plan
        or replay_context.authorization != authorization_context.authorization
        or authorization_context.plan_identity != plan_context.plan["plan_identity"]
        or replay.head_event_identity is None
    ):
        raise GovernanceContractError(
            "authorization consumption permit/replay binding differs"
        )
    authorization = authorization_context.authorization
    consumption_identity = _authorization_consumption_identity(
        authorization,
        plan_context.plan,
    )
    event = _make_event(
        sequence=len(replay.events),
        timestamp_utc=timestamp_utc,
        event_type="authorization_consumed",
        from_state="authorized_unconsumed",
        to_state="authorization_consumed",
        plan=plan_context.plan,
        readiness_identity=authorization["readiness_identity"],
        authorization_identity=authorization["authorization_identity"],
        authorization_consumption_identity=consumption_identity,
        operation=None,
        prerequisite_event_identity=replay.head_event_identity,
        evidence=None,
        resource_reservation=None,
        resource_observation=None,
        reason_code=None,
        previous_event_identity=replay.head_event_identity,
    )
    replay_governance_ledger(
        (*replay.events, event),
        plan=plan_context.plan,
        authorization=authorization,
    )
    plan_context.spent = True
    authorization_context.spent = True
    pending = pending_type(pending_token)
    payload = _event_bytes(event)
    pending_contexts[pending] = _PendingAuthorizationContext(
        plan=plan_context.plan,
        authorization=authorization,
        predecessor_identity=replay.head_event_identity,
        event=event,
        event_bytes=payload,
    )
    return pending, payload


def prepare_authorization_consumption(
    plan_permit: ProductionRuntimePlanPermit,
    authorization_permit: ProductionAuthorizationPermit,
    replay: GovernanceReplay,
    *,
    timestamp_utc: str,
) -> tuple[PendingAuthorizationConsumption, bytes]:
    """Spend production permits and return write-ahead consumption bytes."""
    pending, payload = _prepare_authorization_consumption_core(
        plan_permit,
        authorization_permit,
        replay,
        timestamp_utc=timestamp_utc,
        plan_type=ProductionRuntimePlanPermit,
        authorization_type=ProductionAuthorizationPermit,
        pending_type=PendingAuthorizationConsumption,
        pending_token=_PRODUCTION_PENDING_AUTH_TOKEN,
        plan_contexts=_PRODUCTION_PLAN_CONTEXTS,
        authorization_contexts=_PRODUCTION_AUTH_CONTEXTS,
        pending_contexts=_PRODUCTION_PENDING_AUTH_CONTEXTS,
    )
    assert type(pending) is PendingAuthorizationConsumption
    return pending, payload


def _require_strict_decoded_pending_head(
    replay: GovernanceReplay,
    replay_context: _ReplayContext,
    *,
    expected_event: GovernanceEvent,
    expected_bytes: bytes,
) -> None:
    decoded_head = replay_context.strict_decoded_head_event
    if (
        decoded_head is None
        or not replay.events
        or replay.events[-1] is not decoded_head
        or replay_context.strict_decoded_head_bytes != expected_bytes
        or decoded_head.to_dict() != expected_event.to_dict()
        or encode_governance_ledger((decoded_head,)) != expected_bytes
    ):
        raise GovernanceContractError(
            "pending confirmation requires the exact strict-decoded event head"
        )


def _confirm_authorization_consumption_core(
    pending: object,
    replay: GovernanceReplay,
    *,
    pending_type: type,
    consumed_type: type,
    consumed_token: object,
    pending_contexts: weakref.WeakKeyDictionary,
    consumed_contexts: weakref.WeakKeyDictionary,
) -> object:
    if type(pending) is not pending_type:
        raise GovernanceContractError(
            "authorization confirmation requires exact pending type"
        )
    context = pending_contexts.get(pending)
    if context is None:
        raise GovernanceContractError(
            "authorization confirmation pending capability is absent or consumed"
        )
    replay_context = _validate_replay_capability(replay)
    _require_strict_decoded_pending_head(
        replay,
        replay_context,
        expected_event=context.event,
        expected_bytes=context.event_bytes,
    )
    if (
        replay.state != "authorization_consumed"
        or replay.head_event_identity != context.event["event_identity"]
        or replay.events[-1].to_dict() != context.event.to_dict()
        or replay_context.plan != context.plan
        or replay_context.authorization != context.authorization
        or replay_context.consumption_event_identity != context.event["event_identity"]
    ):
        raise GovernanceContractError(
            "authorization confirmation replay does not contain the exact event"
        )
    del pending_contexts[pending]
    consumed = consumed_type(consumed_token)
    consumed_contexts[consumed] = _ConsumedAuthorizationContext(
        plan=context.plan,
        authorization=context.authorization,
        consumption_event_identity=context.event["event_identity"],
    )
    return consumed


def confirm_authorization_consumption(
    pending: PendingAuthorizationConsumption,
    replay: GovernanceReplay,
) -> ConsumedAuthorizationPermit:
    """Confirm only a persisted, decoded, and fully replayed consumption event."""
    consumed = _confirm_authorization_consumption_core(
        pending,
        replay,
        pending_type=PendingAuthorizationConsumption,
        consumed_type=ConsumedAuthorizationPermit,
        consumed_token=_PRODUCTION_CONSUMED_AUTH_TOKEN,
        pending_contexts=_PRODUCTION_PENDING_AUTH_CONTEXTS,
        consumed_contexts=_PRODUCTION_CONSUMED_CONTEXTS,
    )
    assert type(consumed) is ConsumedAuthorizationPermit
    return consumed


def _reservation_transition(
    operation: Mapping[str, Any],
) -> tuple[str, str, str]:
    operation_id = operation["operation_id"]
    if operation_id == "state-generation":
        return (
            "state_generation_reserved",
            "authorization_consumed",
            "state_generation_running",
        )
    if operation_id == "offline-d9-teacher-labeling":
        return "teacher_reserved", "state_frozen", "teacher_running"
    if operation_id == "supervised-smoke":
        return "smoke_reserved", "corpus_frozen", "smoke_running"
    if operation["purpose"] == "seed":
        seed = operation["seed"]
        return (
            "seed_reserved",
            f"seed_{seed}_initialized",
            f"seed_{seed}_running",
        )
    raise GovernanceContractError(
        "governance-only freeze operations do not issue executor permits"
    )


def _prepare_operation_reservation_core(
    consumed: object,
    replay: GovernanceReplay,
    *,
    operation_id: str,
    timestamp_utc: str,
    consumed_type: type,
    pending_type: type,
    pending_token: object,
    consumed_contexts: Mapping[object, _ConsumedAuthorizationContext],
    pending_contexts: weakref.WeakKeyDictionary,
) -> tuple[object, bytes]:
    if type(consumed) is not consumed_type:
        raise GovernanceContractError(
            "operation reservation requires exact consumed authorization type"
        )
    consumed_context = consumed_contexts.get(consumed)
    if consumed_context is None:
        raise GovernanceContractError(
            "operation reservation lacks consumed authorization registry context"
        )
    replay_context = _validate_replay_capability(replay)
    if (
        consumed_context.pending_reservation_identity is not None
        or consumed_context.unconsumed_operation_permit_identity is not None
        or replay.terminal
        or replay.resource_ledger.open_operation_id is not None
        or replay.resource_ledger.restart_requires_failure_closure
        or replay_context.plan != consumed_context.plan
        or replay_context.authorization != consumed_context.authorization
        or replay.authorization_consumption_identity
        != _authorization_consumption_identity(
            consumed_context.authorization,
            consumed_context.plan,
        )
        or consumed_context.consumption_event_identity
        not in replay_context.event_identities
        or (
            consumed_context.last_reservation_identity is not None
            and consumed_context.last_reservation_identity
            not in replay_context.event_identities
        )
    ):
        raise GovernanceContractError(
            "operation reservation consumed/replay binding differs"
        )
    operation = _operation_by_id(consumed_context.plan, operation_id)
    event_type, from_state, to_state = _reservation_transition(operation)
    if replay.state != from_state or replay.head_event_identity is None:
        raise GovernanceContractError(
            "operation reservation is not the next frozen transition"
        )
    event = _make_event(
        sequence=len(replay.events),
        timestamp_utc=timestamp_utc,
        event_type=event_type,
        from_state=from_state,
        to_state=to_state,
        plan=consumed_context.plan,
        readiness_identity=consumed_context.authorization["readiness_identity"],
        authorization_identity=consumed_context.authorization["authorization_identity"],
        authorization_consumption_identity=replay.authorization_consumption_identity,
        operation=operation,
        prerequisite_event_identity=replay.head_event_identity,
        evidence=None,
        resource_reservation=operation["resource_reservation"],
        resource_observation=None,
        reason_code=None,
        previous_event_identity=replay.head_event_identity,
    )
    replay_governance_ledger(
        (*replay.events, event),
        plan=consumed_context.plan,
        authorization=consumed_context.authorization,
    )
    consumed_context.pending_reservation_identity = event["event_identity"]
    pending = pending_type(pending_token)
    payload = _event_bytes(event)
    pending_contexts[pending] = _PendingOperationContext(
        consumed_permit=consumed,
        plan=consumed_context.plan,
        authorization=consumed_context.authorization,
        operation_id=operation_id,
        attempt_identity=event["attempt_identity"],
        predecessor_identity=replay.head_event_identity,
        event=event,
        event_bytes=payload,
    )
    return pending, payload


def prepare_operation_reservation(
    consumed: ConsumedAuthorizationPermit,
    replay: GovernanceReplay,
    *,
    operation_id: str,
    timestamp_utc: str,
) -> tuple[PendingOperationReservation, bytes]:
    """Reserve the full operation cap before any production executor starts."""
    pending, payload = _prepare_operation_reservation_core(
        consumed,
        replay,
        operation_id=operation_id,
        timestamp_utc=timestamp_utc,
        consumed_type=ConsumedAuthorizationPermit,
        pending_type=PendingOperationReservation,
        pending_token=_PRODUCTION_PENDING_OPERATION_TOKEN,
        consumed_contexts=_PRODUCTION_CONSUMED_CONTEXTS,
        pending_contexts=_PRODUCTION_PENDING_OPERATION_CONTEXTS,
    )
    assert type(pending) is PendingOperationReservation
    return pending, payload


def _confirm_operation_reservation_core(
    pending: object,
    replay: GovernanceReplay,
    *,
    pending_type: type,
    operation_type: type,
    operation_token: object,
    pending_contexts: weakref.WeakKeyDictionary,
    consumed_contexts: Mapping[object, _ConsumedAuthorizationContext],
    operation_contexts: weakref.WeakKeyDictionary,
) -> object:
    if type(pending) is not pending_type:
        raise GovernanceContractError(
            "operation confirmation requires exact pending type"
        )
    context = pending_contexts.get(pending)
    if context is None:
        raise GovernanceContractError(
            "operation confirmation pending capability is absent or consumed"
        )
    consumed_context = consumed_contexts.get(context.consumed_permit)
    replay_context = _validate_replay_capability(replay)
    _require_strict_decoded_pending_head(
        replay,
        replay_context,
        expected_event=context.event,
        expected_bytes=context.event_bytes,
    )
    if (
        consumed_context is None
        or consumed_context.pending_reservation_identity
        != context.event["event_identity"]
        or consumed_context.unconsumed_operation_permit_identity is not None
        or replay.head_event_identity != context.event["event_identity"]
        or replay.events[-1].to_dict() != context.event.to_dict()
        or replay.resource_ledger.open_operation_id != context.operation_id
        or replay.resource_ledger.restart_requires_failure_closure is not True
        or replay_context.plan != context.plan
        or replay_context.authorization != context.authorization
        or replay_context.open_attempt_identity != context.attempt_identity
    ):
        raise GovernanceContractError(
            "operation confirmation replay does not contain the exact reservation"
        )
    del pending_contexts[pending]
    consumed_context.pending_reservation_identity = None
    consumed_context.last_reservation_identity = context.event["event_identity"]
    consumed_context.unconsumed_operation_permit_identity = context.event[
        "event_identity"
    ]
    operation = _operation_by_id(context.plan, context.operation_id)
    permit = operation_type(operation_token)
    operation_contexts[permit] = _OperationPermitContext(
        consumed_permit=context.consumed_permit,
        plan_identity=context.plan["plan_identity"],
        authorization_identity=context.authorization["authorization_identity"],
        consumption_identity=_authorization_consumption_identity(
            context.authorization,
            context.plan,
        ),
        operation_id=context.operation_id,
        attempt_identity=context.attempt_identity,
        purpose=operation["purpose"],
        seed=operation["seed"],
        reservation_event_identity=context.event["event_identity"],
        replay_head_identity=replay.head_event_identity,
    )
    return permit


def confirm_operation_reservation(
    pending: PendingOperationReservation,
    replay: GovernanceReplay,
) -> ProductionOperationPermit:
    """Confirm only an exact replayed reservation and return a one-use permit."""
    permit = _confirm_operation_reservation_core(
        pending,
        replay,
        pending_type=PendingOperationReservation,
        operation_type=ProductionOperationPermit,
        operation_token=_PRODUCTION_OPERATION_TOKEN,
        pending_contexts=_PRODUCTION_PENDING_OPERATION_CONTEXTS,
        consumed_contexts=_PRODUCTION_CONSUMED_CONTEXTS,
        operation_contexts=_PRODUCTION_OPERATION_CONTEXTS,
    )
    assert type(permit) is ProductionOperationPermit
    return permit


def _require_operation_permit_core(
    permit: object,
    replay: GovernanceReplay,
    *,
    operation_id: str,
    purpose: str,
    seed: int | None,
    permit_type: type,
    consumed_contexts: Mapping[object, _ConsumedAuthorizationContext],
    operation_contexts: Mapping[object, _OperationPermitContext],
) -> str:
    if type(permit) is not permit_type:
        raise GovernanceContractError("operation executor requires exact permit type")
    context = operation_contexts.get(permit)
    replay_context = _validate_replay_capability(replay)
    if context is None or context.spent:
        raise GovernanceContractError("operation permit is absent or already consumed")
    consumed_context = consumed_contexts.get(context.consumed_permit)
    if (
        consumed_context is None
        or consumed_context.unconsumed_operation_permit_identity
        != context.reservation_event_identity
        or replay.head_event_identity != context.replay_head_identity
        or replay.head_event_identity != context.reservation_event_identity
        or replay.resource_ledger.open_operation_id != context.operation_id
        or replay_context.open_attempt_identity != context.attempt_identity
        or replay.plan_identity != context.plan_identity
        or replay.authorization_identity != context.authorization_identity
        or replay.authorization_consumption_identity != context.consumption_identity
        or operation_id != context.operation_id
        or purpose != context.purpose
        or seed != context.seed
    ):
        raise GovernanceContractError("operation permit executor binding differs")
    context.spent = True
    consumed_context.unconsumed_operation_permit_identity = None
    return context.attempt_identity


def require_production_operation_permit(
    permit: ProductionOperationPermit,
    replay: GovernanceReplay,
    *,
    operation_id: str,
    purpose: str,
    seed: int | None,
) -> str:
    """Atomically consume one exact production operation permit."""
    return _require_operation_permit_core(
        permit,
        replay,
        operation_id=operation_id,
        purpose=purpose,
        seed=seed,
        permit_type=ProductionOperationPermit,
        consumed_contexts=_PRODUCTION_CONSUMED_CONTEXTS,
        operation_contexts=_PRODUCTION_OPERATION_CONTEXTS,
    )


def _test_checkpoint_ref(
    plan: RuntimePlanRecord,
    *,
    operation_id: str,
    role: str,
    seed: int | None,
    cursor: int,
) -> dict[str, Any]:
    descriptor = {
        "schema_version": "nmm.classical-a-pos-test-checkpoint-evidence.v1",
        "plan_identity": plan["plan_identity"],
        "operation_id": operation_id,
        "attempt_identity": _operation_attempt_identity(plan, operation_id),
        "purpose": _operation_by_id(plan, operation_id)["purpose"],
        "seed": seed,
        "cursor": cursor,
        "rng_identity": canonical_sha256(
            {
                "schema_version": "nmm.classical-a-pos-test-rng-state.v1",
                "operation_id": operation_id,
                "cursor": cursor,
            }
        ),
    }
    identity = canonical_sha256(descriptor)
    return {
        "role": role,
        "identity": identity,
        "file_sha256": canonical_sha256(
            {
                "schema_version": "nmm.classical-a-pos-test-file.v1",
                "checkpoint_identity": identity,
            }
        ),
        "size_bytes": 1 + cursor,
    }


def _build_test_complete_events(
    plan: RuntimePlanRecord,
    authorization: SingleUseAuthorizationRecord,
) -> tuple[GovernanceEvent, ...]:
    events: list[GovernanceEvent] = []
    base_time = datetime(2026, 9, 1, 0, 0, 0)
    readiness_identity = authorization["readiness_identity"]
    authorization_identity = authorization["authorization_identity"]
    consumption_identity = _authorization_consumption_identity(authorization, plan)

    def append(
        event_type: str,
        from_state: str,
        to_state: str,
        *,
        operation_id: str | None = None,
        prerequisite: str | None = None,
        checkpoint: Mapping[str, Any] | None = None,
        reservation: Mapping[str, int] | None = None,
        observation: Mapping[str, int] | None = None,
        include_authorization: bool = True,
        include_consumption: bool = True,
    ) -> GovernanceEvent:
        operation = (
            None if operation_id is None else _operation_by_id(plan, operation_id)
        )
        event = _make_event(
            sequence=len(events),
            timestamp_utc=(base_time + timedelta(seconds=len(events))).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            event_type=event_type,
            from_state=from_state,
            to_state=to_state,
            plan=plan,
            readiness_identity=readiness_identity,
            authorization_identity=(
                authorization_identity if include_authorization else None
            ),
            authorization_consumption_identity=(
                consumption_identity if include_consumption else None
            ),
            operation=operation,
            prerequisite_event_identity=prerequisite,
            evidence={"inputs": [], "outputs": [], "checkpoint": checkpoint},
            resource_reservation=reservation,
            resource_observation=observation,
            reason_code=None,
            previous_event_identity=(
                None if not events else events[-1]["event_identity"]
            ),
        )
        events.append(event)
        return event

    plan_event = append(
        "plan_frozen",
        "no_events",
        "plan_frozen",
        include_authorization=False,
        include_consumption=False,
    )
    readiness_event = append(
        "readiness_frozen",
        "plan_frozen",
        "ready_unauthorized",
        prerequisite=plan_event["event_identity"],
        include_authorization=False,
        include_consumption=False,
    )
    authorization_event = append(
        "authorization_registered",
        "ready_unauthorized",
        "authorized_unconsumed",
        prerequisite=readiness_event["event_identity"],
        include_consumption=False,
    )
    consumption_event = append(
        "authorization_consumed",
        "authorized_unconsumed",
        "authorization_consumed",
        prerequisite=authorization_event["event_identity"],
    )

    state_operation = _operation_by_id(plan, "state-generation")
    state_reserved = append(
        "state_generation_reserved",
        "authorization_consumed",
        "state_generation_running",
        operation_id="state-generation",
        prerequisite=consumption_event["event_identity"],
        reservation=state_operation["resource_reservation"],
    )
    state_completed = append(
        "state_generation_completed",
        "state_generation_running",
        "state_generated",
        operation_id="state-generation",
        prerequisite=state_reserved["event_identity"],
        observation=_resource_vector(state_generation_games=1_024),
    )
    state_frozen = append(
        "state_frozen",
        "state_generated",
        "state_frozen",
        operation_id="state-freeze",
        prerequisite=state_completed["event_identity"],
    )

    teacher_operation = _operation_by_id(plan, "offline-d9-teacher-labeling")
    teacher_reserved = append(
        "teacher_reserved",
        "state_frozen",
        "teacher_running",
        operation_id="offline-d9-teacher-labeling",
        prerequisite=state_frozen["event_identity"],
        reservation=teacher_operation["resource_reservation"],
    )
    teacher_completed = append(
        "teacher_completed",
        "teacher_running",
        "teacher_labeled",
        operation_id="offline-d9-teacher-labeling",
        prerequisite=teacher_reserved["event_identity"],
        observation=_resource_vector(
            teacher_nodes=4_000_000_000,
            teacher_positive_search_labels=1_280,
        ),
    )
    corpus_frozen = append(
        "corpus_frozen",
        "teacher_labeled",
        "corpus_frozen",
        operation_id="corpus-freeze",
        prerequisite=teacher_completed["event_identity"],
    )

    smoke_operation = _operation_by_id(plan, "supervised-smoke")
    smoke_reserved = append(
        "smoke_reserved",
        "corpus_frozen",
        "smoke_running",
        operation_id="supervised-smoke",
        prerequisite=corpus_frozen["event_identity"],
        reservation=smoke_operation["resource_reservation"],
    )
    smoke_passed = append(
        "smoke_passed",
        "smoke_running",
        "smoke_passed",
        operation_id="supervised-smoke",
        prerequisite=smoke_reserved["event_identity"],
        checkpoint=_test_checkpoint_ref(
            plan,
            operation_id="supervised-smoke",
            role=plan["checkpoint_policy"]["smoke_role"],
            seed=None,
            cursor=1,
        ),
        observation=_resource_vector(supervised_updates=1),
    )

    prior_completion = smoke_passed
    for seed_index, seed in enumerate(_SEEDS):
        operation_id = f"seed-{seed}"
        operation = _operation_by_id(plan, operation_id)
        initialized = append(
            "seed_fresh_init_committed",
            (
                "smoke_passed"
                if seed_index == 0
                else f"seed_{_SEEDS[seed_index - 1]}_complete"
            ),
            f"seed_{seed}_initialized",
            operation_id=operation_id,
            prerequisite=prior_completion["event_identity"],
            checkpoint=_test_checkpoint_ref(
                plan,
                operation_id=operation_id,
                role=plan["checkpoint_policy"]["state0_role"],
                seed=seed,
                cursor=0,
            ),
        )
        reserved = append(
            "seed_reserved",
            f"seed_{seed}_initialized",
            f"seed_{seed}_running",
            operation_id=operation_id,
            prerequisite=initialized["event_identity"],
            reservation=operation["resource_reservation"],
        )
        prior = reserved
        latest_checkpoint: Mapping[str, Any] | None = None
        for checkpoint_index in range(1, 20):
            cursor = checkpoint_index * 112
            latest_checkpoint = _test_checkpoint_ref(
                plan,
                operation_id=operation_id,
                role=plan["checkpoint_policy"]["latest_role"],
                seed=seed,
                cursor=cursor,
            )
            prior = append(
                "seed_checkpointed",
                f"seed_{seed}_running",
                f"seed_{seed}_running",
                operation_id=operation_id,
                prerequisite=prior["event_identity"],
                checkpoint=latest_checkpoint,
            )
            if seed_index == 0 and checkpoint_index == 1:
                paused = append(
                    "seed_paused",
                    f"seed_{seed}_running",
                    f"seed_{seed}_paused",
                    operation_id=operation_id,
                    prerequisite=prior["event_identity"],
                    checkpoint=latest_checkpoint,
                )
                prior = append(
                    "seed_resumed",
                    f"seed_{seed}_paused",
                    f"seed_{seed}_running",
                    operation_id=operation_id,
                    prerequisite=paused["event_identity"],
                    checkpoint=latest_checkpoint,
                )
        assert latest_checkpoint is not None
        prior_completion = append(
            "seed_completed",
            f"seed_{seed}_running",
            f"seed_{seed}_complete",
            operation_id=operation_id,
            prerequisite=prior["event_identity"],
            checkpoint=_test_checkpoint_ref(
                plan,
                operation_id=operation_id,
                role=plan["checkpoint_policy"]["complete_role"],
                seed=seed,
                cursor=2_240,
            ),
            observation=_resource_vector(supervised_updates=2_240),
        )

    append(
        "sequence_completed",
        f"seed_{_SEEDS[-1]}_complete",
        "sequence_complete",
        prerequisite=prior_completion["event_identity"],
    )
    replay = replay_governance_ledger(events, plan=plan, authorization=authorization)
    if replay.state != "sequence_complete":
        raise RuntimeError("internal complete governance test ledger did not complete")
    return tuple(events)


@dataclass(frozen=True, slots=True)
class _TestGovernanceFixture:
    plan: RuntimePlanRecord
    authorization: SingleUseAuthorizationRecord
    plan_permit: _TestRuntimePlanPermit
    authorization_permit: _TestAuthorizationPermit
    events: tuple[GovernanceEvent, ...]


def _issue_test_governance_fixture() -> _TestGovernanceFixture:
    """Issue deterministic internal-test records and non-production handles."""
    plan = _build_test_complete_runtime_plan_record()
    authorization = _build_test_authorization_record(plan)
    events = _build_test_complete_events(plan, authorization)
    plan_permit = _TestRuntimePlanPermit(_TEST_PLAN_TOKEN)
    authorization_permit = _TestAuthorizationPermit(_TEST_AUTHORIZATION_TOKEN)
    _TEST_PLAN_CONTEXTS[plan_permit] = _PlanPermitContext(plan=plan)
    _TEST_AUTH_CONTEXTS[authorization_permit] = _AuthorizationPermitContext(
        plan_identity=plan["plan_identity"],
        authorization=authorization,
    )
    return _TestGovernanceFixture(
        plan=plan,
        authorization=authorization,
        plan_permit=plan_permit,
        authorization_permit=authorization_permit,
        events=events,
    )


def _test_prepare_authorization_consumption(
    plan_permit: _TestRuntimePlanPermit,
    authorization_permit: _TestAuthorizationPermit,
    replay: GovernanceReplay,
    *,
    timestamp_utc: str,
) -> tuple[_TestPendingAuthorizationConsumption, bytes]:
    pending, payload = _prepare_authorization_consumption_core(
        plan_permit,
        authorization_permit,
        replay,
        timestamp_utc=timestamp_utc,
        plan_type=_TestRuntimePlanPermit,
        authorization_type=_TestAuthorizationPermit,
        pending_type=_TestPendingAuthorizationConsumption,
        pending_token=_TEST_PENDING_AUTH_TOKEN,
        plan_contexts=_TEST_PLAN_CONTEXTS,
        authorization_contexts=_TEST_AUTH_CONTEXTS,
        pending_contexts=_TEST_PENDING_AUTH_CONTEXTS,
    )
    assert type(pending) is _TestPendingAuthorizationConsumption
    return pending, payload


def _test_confirm_authorization_consumption(
    pending: _TestPendingAuthorizationConsumption,
    replay: GovernanceReplay,
) -> _TestConsumedAuthorizationPermit:
    consumed = _confirm_authorization_consumption_core(
        pending,
        replay,
        pending_type=_TestPendingAuthorizationConsumption,
        consumed_type=_TestConsumedAuthorizationPermit,
        consumed_token=_TEST_CONSUMED_AUTH_TOKEN,
        pending_contexts=_TEST_PENDING_AUTH_CONTEXTS,
        consumed_contexts=_TEST_CONSUMED_CONTEXTS,
    )
    assert type(consumed) is _TestConsumedAuthorizationPermit
    return consumed


def _test_prepare_operation_reservation(
    consumed: _TestConsumedAuthorizationPermit,
    replay: GovernanceReplay,
    *,
    operation_id: str,
    timestamp_utc: str,
) -> tuple[_TestPendingOperationReservation, bytes]:
    pending, payload = _prepare_operation_reservation_core(
        consumed,
        replay,
        operation_id=operation_id,
        timestamp_utc=timestamp_utc,
        consumed_type=_TestConsumedAuthorizationPermit,
        pending_type=_TestPendingOperationReservation,
        pending_token=_TEST_PENDING_OPERATION_TOKEN,
        consumed_contexts=_TEST_CONSUMED_CONTEXTS,
        pending_contexts=_TEST_PENDING_OPERATION_CONTEXTS,
    )
    assert type(pending) is _TestPendingOperationReservation
    return pending, payload


def _test_confirm_operation_reservation(
    pending: _TestPendingOperationReservation,
    replay: GovernanceReplay,
) -> _TestOperationPermit:
    permit = _confirm_operation_reservation_core(
        pending,
        replay,
        pending_type=_TestPendingOperationReservation,
        operation_type=_TestOperationPermit,
        operation_token=_TEST_OPERATION_TOKEN,
        pending_contexts=_TEST_PENDING_OPERATION_CONTEXTS,
        consumed_contexts=_TEST_CONSUMED_CONTEXTS,
        operation_contexts=_TEST_OPERATION_CONTEXTS,
    )
    assert type(permit) is _TestOperationPermit
    return permit


def _test_require_operation_permit(
    permit: _TestOperationPermit,
    replay: GovernanceReplay,
    *,
    operation_id: str,
    purpose: str,
    seed: int | None,
) -> str:
    return _require_operation_permit_core(
        permit,
        replay,
        operation_id=operation_id,
        purpose=purpose,
        seed=seed,
        permit_type=_TestOperationPermit,
        consumed_contexts=_TEST_CONSUMED_CONTEXTS,
        operation_contexts=_TEST_OPERATION_CONTEXTS,
    )
