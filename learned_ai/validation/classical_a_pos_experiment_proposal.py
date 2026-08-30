"""Immutable, authorization-free proposal for classical ``A_pos`` distillation.

This module only builds, verifies, and reads the preregistered proposal.  It has
no writer, launcher, runtime plan, path binding, controller, or authorization
issuer.  The proposal remains non-executable while every C1 unresolved binding
is carried forward verbatim.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.validation import (
    classical_a_pos_distillation_readiness as _readiness,
)

__all__ = (
    "build_classical_a_pos_experiment_proposal",
    "verify_classical_a_pos_experiment_proposal",
    "load_classical_a_pos_experiment_proposal",
)

_SCHEMA_VERSION = "nmm.classical-a-pos-offline-distillation-experiment-proposal.v1"
_EXPERIMENT_ID = "classical-a-pos-offline-distillation-v1"
_REVIEWED_BASE_COMMIT = "e210112540f8869c15f5d4737631a29a9a66341b"
_PROFILE_IDENTITY = "bfa8d2f8e19b1c24641e24e4765844f678cb9288838e5eda3f8782d7ace9cbe0"
_HEX = frozenset("0123456789abcdef")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment_id",
    "proposal_status",
    "launch_status",
    "authorization_status",
    "executable",
    "reviewed_base_commit",
    "profile_identity",
    "training_profile",
    "requested_resource_package",
    "authorization_envelope",
    "objective",
    "operation_order",
    "claim_boundaries",
    "stop_conditions",
    "prohibited_actions",
    "unresolved_bindings",
    "proposal_identity",
}


class _ExperimentProposalError(ValueError):
    """The frozen proposal or its canonical file representation is invalid."""


def _freeze(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _ExperimentProposalError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _ExperimentProposalError(f"{field} contains a non-string key")
            frozen[key] = _freeze(item, field=f"{field}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(
            _freeze(item, field=f"{field}[{index}]") for index, item in enumerate(value)
        )
    raise _ExperimentProposalError(
        f"{field} contains unsupported {type(value).__name__} data"
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _require_exact_keys(value: Mapping[str, Any]) -> None:
    if any(not isinstance(key, str) for key in value):
        raise _ExperimentProposalError("proposal contains a non-string key")
    actual = set(value)
    unknown = sorted(actual - _TOP_LEVEL_KEYS)
    missing = sorted(_TOP_LEVEL_KEYS - actual)
    if unknown:
        raise _ExperimentProposalError(
            f"proposal has unknown keys: {', '.join(unknown)}"
        )
    if missing:
        raise _ExperimentProposalError(
            f"proposal has missing keys: {', '.join(missing)}"
        )


def _require_exact_contract(value: Any, expected: Any, *, field: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(value, Mapping):
            raise _ExperimentProposalError(f"{field} must be an object")
        if any(not isinstance(key, str) for key in value):
            raise _ExperimentProposalError(f"{field} contains a non-string key")
        actual_keys = set(value)
        expected_keys = set(expected)
        unknown = sorted(actual_keys - expected_keys)
        missing = sorted(expected_keys - actual_keys)
        if unknown:
            raise _ExperimentProposalError(
                f"{field} has unknown keys: {', '.join(unknown)}"
            )
        if missing:
            raise _ExperimentProposalError(
                f"{field} has missing keys: {', '.join(missing)}"
            )
        for key in expected:
            _require_exact_contract(
                value[key],
                expected[key],
                field=f"{field}.{key}",
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
            raise _ExperimentProposalError(f"{field} must be an array")
        if len(value) != len(expected):
            raise _ExperimentProposalError(f"{field} length differs")
        for index, (actual_item, expected_item) in enumerate(
            zip(value, expected, strict=True)
        ):
            _require_exact_contract(
                actual_item,
                expected_item,
                field=f"{field}[{index}]",
            )
        return
    if type(value) is not type(expected) or value != expected:
        raise _ExperimentProposalError(f"{field} differs from the frozen proposal")


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise _ExperimentProposalError(
            f"{field} must be a 64-character lowercase SHA-256"
        )
    return value


def _validate_c1_contracts() -> tuple[
    Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]
]:
    try:
        profile = _readiness.validate_training_profile(
            _readiness.FROZEN_TRAINING_PROFILE
        )
        requested = _readiness.validate_requested_resource_package(
            _readiness.REQUESTED_RESOURCE_PACKAGE
        )
        authorization = _readiness.validate_authorization_envelope(
            _readiness.ZERO_AUTHORIZATION_ENVELOPE
        )
    except _readiness.DistillationReadinessError as exc:
        raise _ExperimentProposalError("the C1 readiness contract is invalid") from exc
    if canonical_sha256(profile) != _PROFILE_IDENTITY:
        raise _ExperimentProposalError("the C1 training profile identity drifted")
    return profile, requested, authorization


def _expected_body() -> dict[str, Any]:
    profile, requested, authorization = _validate_c1_contracts()
    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment_id": _EXPERIMENT_ID,
        "proposal_status": "proposed",
        "launch_status": "unlaunched",
        "authorization_status": "unauthorized",
        "executable": False,
        "reviewed_base_commit": _REVIEWED_BASE_COMMIT,
        "profile_identity": _PROFILE_IDENTITY,
        "training_profile": _thaw(profile),
        "requested_resource_package": _thaw(requested),
        "authorization_envelope": _thaw(authorization),
        "objective": {
            "objective_id": "offline-classical-a-pos-policy-distillation-v1",
            "teacher_boundary": (
                "current-head-exact-D9-post-ProductPositionalSafetyGate"
            ),
            "student_boundary": "C1-FROZEN_TRAINING_PROFILE",
            "result_boundary": "three-fresh-supervised-seed-checkpoints",
        },
        "operation_order": [
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
        ],
        "claim_boundaries": {
            "permitted": [
                "proposal-contract-is-frozen-and-authorization-free",
                (
                    "future-completed-run-may-report-frozen-corpus-imitation-"
                    "metrics-only"
                ),
            ],
            "excluded": [
                "playing-strength-improvement",
                "human-specialization",
                "heldout-generalization",
                "product-safety-or-product-readiness",
                "checkpoint-promotion",
                "publication-or-release",
            ],
        },
        "stop_conditions": [
            "proposal-or-c1-contract-drift",
            "any-unresolved-binding",
            "teacher-before-state-freeze",
            "missing-or-drifting-data-checkpoint-path-git-device-or-smoke-provenance",
            "quota-or-resource-cap-exceeded",
            "authorization-missing-expired-revoked-consumed-or-plan-mismatched",
            "nonfinite-or-failed-operation",
            "output-not-isolated",
            "prohibited-action-requested",
        ],
        "prohibited_actions": [
            "launch-smoke-or-training-from-this-proposal",
            "online-teacher-queries-during-student-training",
            "rl-a2c-ppo-value-or-entropy-training",
            "human-db-specialist-db-or-advisor-inputs",
            "heldout-evaluation-games",
            "automatic-retry",
            "checkpoint-or-corpus-relabeling",
            "post-hoc-corpus-shrink-or-quota-change",
            "product-route-switch",
            "checkpoint-promotion-publication-or-release",
        ],
        "unresolved_bindings": list(_readiness.UNRESOLVED_BINDINGS),
    }


def build_classical_a_pos_experiment_proposal() -> Mapping[str, Any]:
    """Return the immutable, unlaunched, and unauthorized frozen proposal."""
    body = _expected_body()
    proposal = {**body, "proposal_identity": canonical_sha256(body)}
    return verify_classical_a_pos_experiment_proposal(proposal)


def verify_classical_a_pos_experiment_proposal(
    proposal: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Verify the exact proposal contract and return an immutable copy."""
    if not isinstance(proposal, Mapping):
        raise _ExperimentProposalError("proposal must be an object")
    snapshot = _thaw(_freeze(proposal, field="proposal"))
    _require_exact_keys(snapshot)

    try:
        _readiness.validate_training_profile(snapshot["training_profile"])
        _readiness.validate_requested_resource_package(
            snapshot["requested_resource_package"]
        )
        _readiness.validate_authorization_envelope(snapshot["authorization_envelope"])
    except _readiness.DistillationReadinessError as exc:
        raise _ExperimentProposalError(
            "proposal embeds an invalid C1 contract"
        ) from exc

    expected_body = _expected_body()
    observed_body = {
        key: value for key, value in snapshot.items() if key != "proposal_identity"
    }
    _require_exact_contract(observed_body, expected_body, field="proposal")

    identity = _require_sha256(
        snapshot["proposal_identity"],
        field="proposal_identity",
    )
    if identity != canonical_sha256(observed_body):
        raise _ExperimentProposalError("proposal_identity does not match the body")
    return _freeze(snapshot, field="proposal")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _ExperimentProposalError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(token: str) -> None:
    raise _ExperimentProposalError(f"non-finite JSON constant is forbidden: {token}")


def load_classical_a_pos_experiment_proposal(
    path: str | Path,
) -> Mapping[str, Any]:
    """Read and verify one exact canonical proposal file without writing."""
    try:
        proposal_path = Path(path)
        raw = proposal_path.read_bytes()
    except (OSError, TypeError, ValueError) as exc:
        raise _ExperimentProposalError("proposal file cannot be read") from exc
    if raw.startswith(b"\xef\xbb\xbf"):
        raise _ExperimentProposalError("a UTF-8 BOM is forbidden")
    try:
        decoded = raw.decode("utf-8")
        parsed = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except _ExperimentProposalError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _ExperimentProposalError(
            "proposal file is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise _ExperimentProposalError("proposal file must contain one JSON object")

    verified = verify_classical_a_pos_experiment_proposal(parsed)
    if raw != canonical_json_bytes(verified) + b"\n":
        raise _ExperimentProposalError(
            "proposal file bytes are not canonical JSON followed by one LF"
        )
    return verified
