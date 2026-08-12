"""Parent authorization for the mature target-refresh sequence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from learned_ai.training.run_contract import canonical_sha256
from learned_ai.validation.target_refresh_mature_fork_diagnostic import (
    READINESS_SCHEMA,
    contract_seeds,
)


AUTHORIZATION_SCHEMA = "nmm.target-refresh-mature-fork-sequence-authorization.v1"
STEP_SCHEMA = "nmm.target-refresh-mature-fork-sequence-step.v1"
DELEGATED_OPERATOR = "product-owner-delegated-agent"
PERMITTED_OPERATIONS = (
    "authorize-six-managed-arms-just-in-time",
    "run-six-managed-arms-once-in-frozen-order",
    "read-transition-checkpoints-on-cpu",
    "run-288-no-update-direct-crossplay-games-once",
    "publish-development-result-once",
)
PROHIBITED_OPERATIONS = (
    "automatic-retry",
    "automatic-extension",
    "automatic-resume-or-recovery",
    "held-out-evaluation",
    "model-promotion",
    "model-publication",
    "long-training-launch",
)


class MatureTargetRefreshSequenceError(RuntimeError):
    """Raised when the one-shot mature sequence cannot be proven safe."""


@dataclass(frozen=True)
class SequenceStep:
    kind: str
    launch_order: int | None = None
    seed: int | None = None
    condition: str | None = None
    arm_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"schema_version": STEP_SCHEMA, "kind": self.kind}
        for key in ("launch_order", "seed", "condition", "arm_id"):
            item = getattr(self, key)
            if item is not None:
                value[key] = item
        return value


def _identity(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MatureTargetRefreshSequenceError(f"{field} must be a SHA-256")
    return value


def validate_readiness_identity(readiness: Mapping[str, Any]) -> str:
    if readiness.get("schema_version") != READINESS_SCHEMA:
        raise MatureTargetRefreshSequenceError("readiness schema differs")
    identity = _identity(
        readiness.get("readiness_identity"), field="readiness identity"
    )
    body = dict(readiness)
    body.pop("readiness_identity", None)
    if identity != canonical_sha256(body):
        raise MatureTargetRefreshSequenceError("readiness identity differs")
    if (
        readiness.get("state")
        != "six_arm_plans_ready_for_one_parent_product_authorization"
        or readiness.get("launch_authorized") is not False
    ):
        raise MatureTargetRefreshSequenceError("readiness state differs")
    return identity


def build_sequence_steps(contract: Mapping[str, Any]) -> tuple[SequenceStep, ...]:
    arms = contract.get("arms")
    if not isinstance(arms, list) or len(arms) != 6:
        raise MatureTargetRefreshSequenceError("sequence arm cells differ")
    steps = tuple(
        SequenceStep(
            kind="run-arm",
            launch_order=int(arm["launch_order"]),
            seed=int(arm["seed"]),
            condition=str(arm["condition"]),
            arm_id=str(arm["arm_id"]),
        )
        for arm in arms
    )
    expected = [
        (seed, condition)
        for seed in contract_seeds(contract)
        for condition in ("refresh-mature", "stale-control")
    ]
    if [(step.seed, step.condition) for step in steps] != expected or [
        step.launch_order for step in steps
    ] != list(range(1, 7)):
        raise MatureTargetRefreshSequenceError("sequence launch order differs")
    return (*steps, SequenceStep(kind="publish-development-result"))


def _resource_envelope(contract: Mapping[str, Any]) -> dict[str, Any]:
    resources = contract.get("resources", {})
    expected = {
        "maximum_training_games_total": 3600,
        "maximum_training_games_per_arm": 600,
        "maximum_optimizer_consumed_transitions_total": 49152,
        "maximum_active_wall_hours_total": 4,
        "maximum_active_wall_hours_per_arm": 0.6,
        "maximum_no_update_games_total": 288,
        "maximum_requested_sanmill_node_ceilings": 172800000,
    }
    if {key: resources.get(key) for key in expected} != expected:
        raise MatureTargetRefreshSequenceError("resource envelope differs")
    return expected


def build_sequence_authorization(
    *,
    contract: Mapping[str, Any],
    readiness: Mapping[str, Any],
    expected_readiness_identity: str,
    decision_note: str,
    authorized_at_utc: str,
) -> dict[str, Any]:
    readiness_identity = validate_readiness_identity(readiness)
    if readiness_identity != _identity(
        expected_readiness_identity, field="expected readiness identity"
    ):
        raise MatureTargetRefreshSequenceError("expected readiness identity differs")
    if not decision_note.strip():
        raise MatureTargetRefreshSequenceError("decision note is required")
    if not authorized_at_utc.endswith("Z"):
        raise MatureTargetRefreshSequenceError("authorization time must be UTC")
    if readiness.get("contract", {}).get("plan_identity") != contract.get(
        "plan_identity"
    ):
        raise MatureTargetRefreshSequenceError("authorization contract differs")
    body = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "authorized_at_utc": authorized_at_utc,
        "authorized_by": "product-owner",
        "operator": DELEGATED_OPERATOR,
        "decision_note": decision_note.strip(),
        "objective": contract["objective"],
        "plan_identity": contract["plan_identity"],
        "readiness_identity": readiness_identity,
        "source_commit": readiness["source"]["head"],
        "resource_envelope": _resource_envelope(contract),
        "launch_order": [step.to_dict() for step in build_sequence_steps(contract)],
        "permitted_operations": list(PERMITTED_OPERATIONS),
        "prohibited_operations": list(PROHIBITED_OPERATIONS),
        "claim_boundary": contract["claim_boundary"],
        "one_parent_launch_attempt": True,
        "expiry": "consumed when the one parent launch starts or owner revokes",
    }
    return {**body, "authorization_identity": canonical_sha256(body)}


def validate_sequence_authorization(
    authorization: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    readiness: Mapping[str, Any],
    expected_readiness_identity: str,
) -> str:
    expected = build_sequence_authorization(
        contract=contract,
        readiness=readiness,
        expected_readiness_identity=expected_readiness_identity,
        decision_note=str(authorization.get("decision_note", "")),
        authorized_at_utc=str(authorization.get("authorized_at_utc", "")),
    )
    if dict(authorization) != expected:
        raise MatureTargetRefreshSequenceError("sequence authorization differs")
    return _identity(
        authorization.get("authorization_identity"),
        field="authorization identity",
    )


def execute_sequence_steps(
    steps: Sequence[SequenceStep],
    *,
    run_arm: Any,
    publish_result: Any,
) -> list[Any]:
    outputs = []
    for step in steps:
        if step.kind == "run-arm":
            outputs.append(run_arm(step))
        elif step.kind == "publish-development-result":
            outputs.append(publish_result(step))
        else:
            raise MatureTargetRefreshSequenceError(
                f"unsupported sequence step: {step.kind}"
            )
    return outputs


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "DELEGATED_OPERATOR",
    "MatureTargetRefreshSequenceError",
    "PROHIBITED_OPERATIONS",
    "PERMITTED_OPERATIONS",
    "STEP_SCHEMA",
    "SequenceStep",
    "build_sequence_authorization",
    "build_sequence_steps",
    "execute_sequence_steps",
    "validate_readiness_identity",
    "validate_sequence_authorization",
]
