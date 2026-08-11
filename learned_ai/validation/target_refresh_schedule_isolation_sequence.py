"""Parent authorization and ordering for the schedule-isolation sequence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from learned_ai.training.run_contract import canonical_sha256


AUTHORIZATION_SCHEMA = (
    "nmm.target-refresh-schedule-isolation-sequence-authorization.v1"
)
READINESS_SCHEMA = "nmm.target-refresh-equal-transition-readiness.v1"
SEQUENCE_STEP_SCHEMA = "nmm.target-refresh-schedule-isolation-step.v1"
AUTHORIZED_BY = "product-owner"
DELEGATED_OPERATOR = "product-owner-delegated-agent"
PERMITTED_OPERATIONS = (
    "authorize-prefix-just-in-time",
    "run-prefix-once",
    "prepare-same-seed-arms-once",
    "authorize-arm-just-in-time",
    "run-arm-once",
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
EXPIRY = "consumed when the one parent launch attempt starts or owner revokes"


class ScheduleIsolationSequenceError(RuntimeError):
    """Raised when the parent sequence cannot be proven safe."""


@dataclass(frozen=True)
class SequenceStep:
    """One deterministic operation in the parent launch sequence."""

    kind: str
    launch_order: int | None = None
    seed: int | None = None
    condition: str | None = None
    arm_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": SEQUENCE_STEP_SCHEMA,
            "kind": self.kind,
        }
        for key in ("launch_order", "seed", "condition", "arm_id"):
            item = getattr(self, key)
            if item is not None:
                value[key] = item
        return value


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScheduleIsolationSequenceError(f"{field} must be an object")
    return value


def _require_identity(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ScheduleIsolationSequenceError(f"{field} must be a SHA-256")
    return value


def _require_git_commit(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ScheduleIsolationSequenceError(f"{field} must be a full Git SHA")
    return value


def validate_readiness_identity(readiness: Mapping[str, Any]) -> str:
    """Validate the canonical identity and non-authorizing readiness state."""
    if readiness.get("schema_version") != READINESS_SCHEMA:
        raise ScheduleIsolationSequenceError("readiness schema differs")
    identity = _require_identity(
        readiness.get("readiness_identity"), field="readiness identity"
    )
    body = dict(readiness)
    body.pop("readiness_identity", None)
    if identity != canonical_sha256(body):
        raise ScheduleIsolationSequenceError("readiness identity differs")
    if readiness.get("state") != "prefix_plans_ready_for_product_authorization":
        raise ScheduleIsolationSequenceError("readiness state differs")
    if readiness.get("launch_authorized") is not False:
        raise ScheduleIsolationSequenceError("readiness unexpectedly authorizes launch")
    return identity


def build_sequence_steps(contract: Mapping[str, Any]) -> tuple[SequenceStep, ...]:
    """Return the only permitted prefix, preparation, arm, result order."""
    prefixes = contract.get("prefixes")
    arms = contract.get("arms")
    if not isinstance(prefixes, list) or not isinstance(arms, list):
        raise ScheduleIsolationSequenceError("contract sequence cells differ")
    by_seed: dict[int, list[Mapping[str, Any]]] = {}
    for arm in arms:
        if not isinstance(arm, Mapping):
            raise ScheduleIsolationSequenceError("arm record differs")
        by_seed.setdefault(int(arm["seed"]), []).append(arm)
    steps: list[SequenceStep] = []
    prior_order = 0
    for prefix in sorted(prefixes, key=lambda item: int(item["launch_order"])):
        seed = int(prefix["seed"])
        prefix_order = int(prefix["launch_order"])
        if prefix_order <= prior_order:
            raise ScheduleIsolationSequenceError("prefix launch order differs")
        steps.append(
            SequenceStep(kind="run-prefix", launch_order=prefix_order, seed=seed)
        )
        steps.append(SequenceStep(kind="prepare-seed-arms", seed=seed))
        seed_arms = sorted(
            by_seed.get(seed, []), key=lambda item: int(item["launch_order"])
        )
        if [str(item["condition"]) for item in seed_arms] != [
            "refresh-once",
            "no-refresh",
        ]:
            raise ScheduleIsolationSequenceError(
                f"seed {seed} arm condition order differs"
            )
        for arm in seed_arms:
            launch_order = int(arm["launch_order"])
            if launch_order <= prefix_order or launch_order <= prior_order:
                raise ScheduleIsolationSequenceError("arm launch order differs")
            steps.append(
                SequenceStep(
                    kind="run-arm",
                    launch_order=launch_order,
                    seed=seed,
                    condition=str(arm["condition"]),
                    arm_id=str(arm["arm_id"]),
                )
            )
            prior_order = launch_order
    if len(steps) != 12:
        raise ScheduleIsolationSequenceError("sequence step count differs")
    steps.append(SequenceStep(kind="publish-development-result"))
    return tuple(steps)


def _authorization_limits(contract: Mapping[str, Any]) -> dict[str, Any]:
    resources = _require_mapping(contract.get("resources"), field="resources")
    expected = {
        "maximum_active_wall_hours_total": 6.0,
        "maximum_actual_training_games_total": 3450,
        "maximum_contract_training_games_total": 3600,
        "maximum_development_measurement_games_total": 288,
        "scientific_post_fork_transitions_total": 49152,
    }
    observed = {key: resources.get(key) for key in expected}
    if observed != expected:
        raise ScheduleIsolationSequenceError("aggregate resource envelope differs")
    return expected


def build_sequence_authorization(
    *,
    contract: Mapping[str, Any],
    readiness: Mapping[str, Any],
    decision_note: str,
    authorized_at_utc: str,
) -> dict[str, Any]:
    """Build a structured parent grant after an explicit product decision."""
    readiness_identity = validate_readiness_identity(readiness)
    if not isinstance(decision_note, str) or not decision_note.strip():
        raise ScheduleIsolationSequenceError("decision note is required")
    if not isinstance(authorized_at_utc, str) or not authorized_at_utc.endswith("Z"):
        raise ScheduleIsolationSequenceError("authorization time must be UTC")
    contract_record = _require_mapping(
        readiness.get("contract"), field="readiness contract"
    )
    plan_identity = _require_identity(
        contract.get("plan_identity"), field="plan identity"
    )
    if contract_record.get("plan_identity") != plan_identity:
        raise ScheduleIsolationSequenceError("authorization contract differs")
    source = _require_mapping(readiness.get("source"), field="readiness source")
    source_commit = _require_git_commit(source.get("head"), field="source commit")
    body = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "authorized_at_utc": authorized_at_utc,
        "authorized_by": AUTHORIZED_BY,
        "decision_note": decision_note.strip(),
        "experiment_family_id": contract["experiment_family_id"],
        "objective": contract["objective"],
        "plan_identity": plan_identity,
        "readiness_identity": readiness_identity,
        "source_commit": source_commit,
        "resource_envelope": _authorization_limits(contract),
        "launch_order": [step.to_dict() for step in build_sequence_steps(contract)],
        "permitted_operations": list(PERMITTED_OPERATIONS),
        "prohibited_operations": list(PROHIBITED_OPERATIONS),
        "claim_boundary": contract["claim_boundary"],
        "one_parent_launch_attempt": True,
        "expiry": EXPIRY,
    }
    return {**body, "authorization_identity": canonical_sha256(body)}


def validate_sequence_authorization(
    authorization: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> str:
    """Validate a grant against current immutable plan and readiness inputs."""
    expected = build_sequence_authorization(
        contract=contract,
        readiness=readiness,
        decision_note=str(authorization.get("decision_note", "")),
        authorized_at_utc=str(authorization.get("authorized_at_utc", "")),
    )
    if dict(authorization) != expected:
        raise ScheduleIsolationSequenceError("sequence authorization differs")
    return _require_identity(
        authorization.get("authorization_identity"),
        field="authorization identity",
    )


def execute_sequence_steps(
    steps: Sequence[SequenceStep],
    *,
    run_prefix: Callable[[SequenceStep], Any],
    prepare_seed_arms: Callable[[SequenceStep], Any],
    run_arm: Callable[[SequenceStep], Any],
    publish_result: Callable[[SequenceStep], Any],
) -> list[Any]:
    """Execute exactly once in order; propagate the first failure unchanged."""
    outputs: list[Any] = []
    dispatch: dict[str, Callable[[SequenceStep], Any]] = {
        "run-prefix": run_prefix,
        "prepare-seed-arms": prepare_seed_arms,
        "run-arm": run_arm,
        "publish-development-result": publish_result,
    }
    for step in steps:
        handler = dispatch.get(step.kind)
        if handler is None:
            raise ScheduleIsolationSequenceError(
                f"unsupported sequence step: {step.kind}"
            )
        outputs.append(handler(step))
    return outputs


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "AUTHORIZED_BY",
    "DELEGATED_OPERATOR",
    "EXPIRY",
    "PERMITTED_OPERATIONS",
    "PROHIBITED_OPERATIONS",
    "ScheduleIsolationSequenceError",
    "SequenceStep",
    "build_sequence_authorization",
    "build_sequence_steps",
    "execute_sequence_steps",
    "validate_readiness_identity",
    "validate_sequence_authorization",
]
