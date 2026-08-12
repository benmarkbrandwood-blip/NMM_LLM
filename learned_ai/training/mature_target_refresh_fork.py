"""Publish a controlled mature fork for one later target-refresh decision."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from learned_ai.training.checkpoint_envelope import (
    CheckpointDescriptor,
    load_checkpoint,
    save_checkpoint,
)
from learned_ai.training.generalist_run_manifest import utc_now_text
from learned_ai.training.run_contract import canonical_sha256


MATURE_FORK_KIND = "mature-target-refresh-fork-v1"


class MatureTargetRefreshForkError(RuntimeError):
    """Raised when a mature diagnostic fork cannot be published safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise MatureTargetRefreshForkError(f"{field} must be a SHA-256")
    normalized = value.removeprefix("sha256:").lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise MatureTargetRefreshForkError(f"{field} must be a SHA-256")
    return normalized


def _tree_equal(first: Any, second: Any) -> bool:
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        return (
            isinstance(first, torch.Tensor)
            and isinstance(second, torch.Tensor)
            and torch.equal(first, second)
        )
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return (
            isinstance(first, np.ndarray)
            and isinstance(second, np.ndarray)
            and np.array_equal(first, second)
        )
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        return (
            isinstance(first, Mapping)
            and isinstance(second, Mapping)
            and set(first) == set(second)
            and all(_tree_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(_tree_equal(left, right) for left, right in zip(first, second))
        )
    return bool(first == second)


def _model_target_differs(trainer_state: Mapping[str, Any], model_state: Any) -> bool:
    target = trainer_state.get("target_network")
    target_state = target.get("model_state") if isinstance(target, Mapping) else None
    if not isinstance(model_state, Mapping) or not isinstance(target_state, Mapping):
        raise MatureTargetRefreshForkError("source target network is incomplete")
    if set(model_state) != set(target_state):
        raise MatureTargetRefreshForkError("source model and target keys differ")
    return any(not torch.equal(model_state[key], target_state[key]) for key in model_state)


def publish_mature_target_refresh_fork(
    source: str | Path,
    destination: str | Path,
    *,
    expected_source_file_sha256: str,
    expected_source_payload_sha256: str,
    expected_source_config_sha256: str,
    expected_source_experiment_id: str,
    expected_source_game_count: int,
    expected_source_update_count: int,
    expected_source_post_fork_transitions: int,
    expected_specialist_db_sha256: str,
    target_resume_config_sha256: str,
    target_experiment_id: str,
    target_experiment_digest: str,
    target_run_id: str,
    temperature_origin: float,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create one shared fork while changing only declared neutral state."""
    expected_source_file = _sha256(
        expected_source_file_sha256,
        field="expected_source_file_sha256",
    )
    expected_source_payload = _sha256(
        expected_source_payload_sha256,
        field="expected_source_payload_sha256",
    )
    expected_source_config = _sha256(
        expected_source_config_sha256,
        field="expected_source_config_sha256",
    )
    expected_specialist = _sha256(
        expected_specialist_db_sha256,
        field="expected_specialist_db_sha256",
    )
    target_config = _sha256(
        target_resume_config_sha256,
        field="target_resume_config_sha256",
    )
    target_digest = _sha256(
        target_experiment_digest,
        field="target_experiment_digest",
    )
    if not isinstance(temperature_origin, (int, float)) or isinstance(
        temperature_origin, bool
    ):
        raise MatureTargetRefreshForkError("temperature_origin must be numeric")
    temperature_origin = float(temperature_origin)
    if not 0.2 <= temperature_origin <= 0.9:
        raise MatureTargetRefreshForkError("temperature_origin is out of range")
    for value, field in (
        (expected_source_game_count, "expected_source_game_count"),
        (expected_source_update_count, "expected_source_update_count"),
        (
            expected_source_post_fork_transitions,
            "expected_source_post_fork_transitions",
        ),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MatureTargetRefreshForkError(f"{field} must be positive")
    for value, field in (
        (expected_source_experiment_id, "expected_source_experiment_id"),
        (target_experiment_id, "target_experiment_id"),
        (target_run_id, "target_run_id"),
    ):
        if not isinstance(value, str) or not value:
            raise MatureTargetRefreshForkError(f"{field} is required")

    source_path = Path(source).resolve(strict=True)
    target_path = Path(destination).resolve(strict=False)
    if source_path == target_path:
        raise MatureTargetRefreshForkError("destination must differ from source")
    if target_path.exists():
        raise MatureTargetRefreshForkError("mature fork already exists")
    if _sha256_file(source_path) != expected_source_file:
        raise MatureTargetRefreshForkError("source file identity differs")

    source_envelope = load_checkpoint(source_path, map_location="cpu")
    source_descriptor = source_envelope.descriptor
    source_payload = source_envelope.payload
    if source_envelope.payload_sha256 != expected_source_payload:
        raise MatureTargetRefreshForkError("source payload identity differs")
    if source_descriptor.role != "transition_diagnostic_candidate":
        raise MatureTargetRefreshForkError("source role is not a transition candidate")
    if source_descriptor.config_sha256 != expected_source_config:
        raise MatureTargetRefreshForkError("source configuration differs")
    if source_descriptor.experiment_id != expected_source_experiment_id:
        raise MatureTargetRefreshForkError("source experiment differs")
    trainer_state = source_payload.trainer_state
    if trainer_state.get("game_count") != expected_source_game_count:
        raise MatureTargetRefreshForkError("source game count differs")
    if trainer_state.get("update_count") != expected_source_update_count:
        raise MatureTargetRefreshForkError("source update count differs")
    recovery = trainer_state.get("recovery_state")
    if not isinstance(recovery, Mapping):
        raise MatureTargetRefreshForkError("source recovery state is absent")
    prior_fork = recovery.get("target_refresh_fork_state")
    if not isinstance(prior_fork, Mapping):
        raise MatureTargetRefreshForkError("source prior fork state is absent")
    origin = prior_fork.get("post_fork_transition_origin")
    consumed = recovery.get("optimizer_consumed_transition_count")
    if (
        prior_fork.get("captured") is not True
        or prior_fork.get("treatment") != "no-refresh"
        or not isinstance(origin, int)
        or not isinstance(consumed, int)
        or consumed - origin != expected_source_post_fork_transitions
    ):
        raise MatureTargetRefreshForkError("source prior treatment boundary differs")
    pending = recovery.get("pending_steps")
    if not isinstance(pending, list):
        raise MatureTargetRefreshForkError("source pending queue is invalid")
    mutable = source_payload.data_state.get("mutable_assets")
    specialist = mutable.get("specialist_db") if isinstance(mutable, Mapping) else None
    if (
        not isinstance(specialist, Mapping)
        or specialist.get("sha256") != expected_specialist
        or source_descriptor.asset_identities.get("specialist_db")
        != expected_specialist
    ):
        raise MatureTargetRefreshForkError("source SpecialistDB identity differs")
    if not _model_target_differs(trainer_state, source_payload.model_state):
        raise MatureTargetRefreshForkError("source target already equals mature policy")

    normalized_recovery = copy.deepcopy(dict(recovery))
    normalized_recovery["pending_steps"] = []
    normalized_recovery["target_refresh_fork_state"] = {
        "schema_version": "nmm.target-refresh-fork-state.v1",
        "fork_game": expected_source_game_count,
        "captured": True,
        "treatment": None,
        "post_fork_transition_origin": None,
    }
    normalized_trainer = copy.deepcopy(dict(trainer_state))
    normalized_trainer["temperature"] = temperature_origin
    normalized_trainer["recovery_state"] = normalized_recovery
    target_payload = replace(source_payload, trainer_state=normalized_trainer)

    implementation = dict(source_descriptor.implementation)
    implementation.update(
        {
            "experiment_digest": f"sha256:{target_digest}",
            "mature_target_refresh_fork_kind": MATURE_FORK_KIND,
            "mature_target_refresh_source_checkpoint_id": (
                source_descriptor.checkpoint_id
            ),
            "mature_target_refresh_source_payload_sha256": (
                source_envelope.payload_sha256
            ),
            "mature_target_refresh_dropped_pending_count": str(len(pending)),
            "mature_target_refresh_temperature_origin": repr(temperature_origin),
        }
    )
    asset_identities = dict(source_descriptor.asset_identities)
    asset_identities["source_checkpoint"] = canonical_sha256(
        {
            "checkpoint_id": source_descriptor.checkpoint_id,
            "payload_sha256": source_envelope.payload_sha256,
        }
    )
    descriptor = CheckpointDescriptor(
        checkpoint_id=f"{source_descriptor.checkpoint_id}:mature-fork",
        run_id=target_run_id,
        experiment_id=target_experiment_id,
        parent_checkpoint_id=source_descriptor.checkpoint_id,
        role="target_refresh_fork",
        save_reason="normalize_mature_target_refresh_common_fork",
        created_at_utc=created_at_utc or utc_now_text(),
        config_sha256=target_config,
        feature_schema_version=source_descriptor.feature_schema_version,
        label_schema_version=source_descriptor.label_schema_version,
        database_schema_versions=source_descriptor.database_schema_versions,
        asset_identities=asset_identities,
        implementation=implementation,
    )
    save_checkpoint(target_path, descriptor, target_payload, previous_copies=0)
    published = load_checkpoint(target_path, map_location="cpu")

    source_trainer_static = dict(trainer_state)
    target_trainer_static = dict(published.payload.trainer_state)
    source_recovery = dict(source_trainer_static.pop("recovery_state"))
    target_recovery = dict(target_trainer_static.pop("recovery_state"))
    source_trainer_static.pop("temperature")
    target_trainer_static.pop("temperature")
    source_recovery.pop("pending_steps")
    target_recovery.pop("pending_steps")
    source_recovery.pop("target_refresh_fork_state")
    target_recovery.pop("target_refresh_fork_state")
    if not all(
        (
            _tree_equal(source_payload.model_state, published.payload.model_state),
            _tree_equal(
                source_payload.optimizer_state,
                published.payload.optimizer_state,
            ),
            _tree_equal(source_payload.rng_state, published.payload.rng_state),
            _tree_equal(source_payload.data_state, published.payload.data_state),
            _tree_equal(source_trainer_static, target_trainer_static),
            _tree_equal(source_recovery, target_recovery),
        )
    ):
        raise MatureTargetRefreshForkError("mature fork changed protected state")
    return {
        "schema_version": "nmm.mature-target-refresh-fork-publication.v1",
        "source_path": str(source_path),
        "source_file_sha256": expected_source_file,
        "source_checkpoint_id": source_descriptor.checkpoint_id,
        "source_payload_sha256": source_envelope.payload_sha256,
        "destination_path": str(target_path.resolve(strict=True)),
        "destination_file_sha256": _sha256_file(target_path),
        "destination_checkpoint_id": descriptor.checkpoint_id,
        "destination_payload_sha256": published.payload_sha256,
        "target_resume_config_sha256": target_config,
        "target_experiment_digest": f"sha256:{target_digest}",
        "game_count": expected_source_game_count,
        "update_count": expected_source_update_count,
        "prior_post_fork_transitions": expected_source_post_fork_transitions,
        "dropped_pending_transition_count": len(pending),
        "temperature_origin": temperature_origin,
        "normalization_allowlist": [
            "recovery_state.pending_steps -> []",
            "recovery_state.target_refresh_fork_state -> untreated mature fork",
            "trainer_state.temperature -> explicit mature schedule origin",
        ],
    }


__all__ = [
    "MATURE_FORK_KIND",
    "MatureTargetRefreshForkError",
    "publish_mature_target_refresh_fork",
]
