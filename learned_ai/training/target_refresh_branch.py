"""Publish an auditable descriptor-only branch from a target-refresh fork."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

from learned_ai.training.checkpoint_envelope import (
    CheckpointDescriptor,
    load_checkpoint,
    rebind_checkpoint_descriptor,
)
from learned_ai.training.generalist_run_manifest import utc_now_text


BRANCH_REBIND_KIND = "target-refresh-fork-v1"
BRANCH_TREATMENTS = frozenset({"refresh-once", "no-refresh"})


class TargetRefreshBranchError(RuntimeError):
    """Raised when a target-refresh fork cannot be safely rebound."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TargetRefreshBranchError(f"{field} must be a SHA-256")
    normalized = value.removeprefix("sha256:").lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise TargetRefreshBranchError(f"{field} must be a SHA-256")
    return normalized


def _validate_fork_payload(
    payload: Any,
    *,
    expected_game_count: int,
    expected_specialist_db_sha256: str,
) -> None:
    trainer_state = payload.trainer_state
    if trainer_state.get("game_count") != expected_game_count:
        raise TargetRefreshBranchError("fork game count differs")
    recovery = trainer_state.get("recovery_state")
    if not isinstance(recovery, dict):
        raise TargetRefreshBranchError("fork recovery state is absent")
    fork = recovery.get("target_refresh_fork_state")
    expected_fork = {
        "schema_version": "nmm.target-refresh-fork-state.v1",
        "fork_game": expected_game_count,
        "captured": True,
        "treatment": None,
        "post_fork_transition_origin": None,
    }
    if fork != expected_fork:
        raise TargetRefreshBranchError("fork treatment state differs")
    mutable = payload.data_state.get("mutable_assets")
    specialist = mutable.get("specialist_db") if isinstance(mutable, dict) else None
    if (
        not isinstance(specialist, dict)
        or specialist.get("sha256") != expected_specialist_db_sha256
    ):
        raise TargetRefreshBranchError("fork SpecialistDB identity differs")


def publish_target_refresh_branch_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    treatment: str,
    expected_source_config_sha256: str,
    target_resume_config_sha256: str,
    expected_experiment_id: str,
    expected_game_count: int,
    expected_specialist_db_sha256: str,
    target_experiment_digest: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Rebind only descriptor identities; preserve every training-state byte."""
    if treatment not in BRANCH_TREATMENTS:
        raise TargetRefreshBranchError("branch treatment is unsupported")
    source_config = _require_sha256(
        expected_source_config_sha256,
        field="expected_source_config_sha256",
    )
    target_config = _require_sha256(
        target_resume_config_sha256,
        field="target_resume_config_sha256",
    )
    specialist_sha256 = _require_sha256(
        expected_specialist_db_sha256,
        field="expected_specialist_db_sha256",
    )
    experiment_digest = _require_sha256(
        target_experiment_digest,
        field="target_experiment_digest",
    )
    if (
        isinstance(expected_game_count, bool)
        or not isinstance(expected_game_count, int)
        or expected_game_count <= 0
    ):
        raise TargetRefreshBranchError("expected_game_count must be positive")
    if not isinstance(expected_experiment_id, str) or not expected_experiment_id:
        raise TargetRefreshBranchError("expected_experiment_id is required")

    source_path = Path(source).resolve(strict=True)
    destination_path = Path(destination).resolve(strict=False)
    if source_path == destination_path:
        raise TargetRefreshBranchError("branch destination must differ from source")
    if destination_path.exists():
        raise TargetRefreshBranchError("branch checkpoint already exists")
    envelope = load_checkpoint(source_path, map_location="cpu")
    descriptor = envelope.descriptor
    if descriptor.role != "target_refresh_fork":
        raise TargetRefreshBranchError("source checkpoint role is not a fork")
    if descriptor.config_sha256 != source_config:
        raise TargetRefreshBranchError("source checkpoint configuration differs")
    if descriptor.experiment_id != expected_experiment_id:
        raise TargetRefreshBranchError("source checkpoint experiment differs")
    if descriptor.asset_identities.get("specialist_db") != specialist_sha256:
        raise TargetRefreshBranchError("source descriptor SpecialistDB differs")
    if "target_refresh_branch_kind" in descriptor.implementation:
        raise TargetRefreshBranchError("source checkpoint is already rebound")
    _validate_fork_payload(
        envelope.payload,
        expected_game_count=expected_game_count,
        expected_specialist_db_sha256=specialist_sha256,
    )

    implementation = dict(descriptor.implementation)
    implementation.update(
        {
            "experiment_digest": f"sha256:{experiment_digest}",
            "target_refresh_branch_kind": BRANCH_REBIND_KIND,
            "target_refresh_branch_source_checkpoint_id": (
                descriptor.checkpoint_id
            ),
            "target_refresh_branch_source_payload_sha256": (
                envelope.payload_sha256
            ),
            "target_refresh_branch_treatment": treatment,
        }
    )
    branch_descriptor: CheckpointDescriptor = replace(
        descriptor,
        checkpoint_id=f"{descriptor.checkpoint_id}:branch:{treatment}",
        parent_checkpoint_id=descriptor.checkpoint_id,
        save_reason=f"target-refresh-branch-rebind-{treatment}",
        created_at_utc=created_at_utc or utc_now_text(),
        config_sha256=target_config,
        implementation=implementation,
    )
    rebind_checkpoint_descriptor(
        source_path,
        destination_path,
        branch_descriptor,
        previous_copies=0,
    )
    branch = load_checkpoint(destination_path, map_location="cpu")
    _validate_fork_payload(
        branch.payload,
        expected_game_count=expected_game_count,
        expected_specialist_db_sha256=specialist_sha256,
    )
    if (
        branch.descriptor != branch_descriptor
        or branch.payload_sha256 != envelope.payload_sha256
    ):
        raise TargetRefreshBranchError(
            "branch publication changed checkpoint state"
        )
    return {
        "source_path": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "source_checkpoint_id": descriptor.checkpoint_id,
        "source_payload_sha256": envelope.payload_sha256,
        "destination_path": str(destination_path.resolve(strict=True)),
        "destination_sha256": _sha256_file(destination_path),
        "branch_checkpoint_id": branch_descriptor.checkpoint_id,
        "branch_payload_sha256": branch.payload_sha256,
        "target_resume_config_sha256": target_config,
        "target_experiment_digest": f"sha256:{experiment_digest}",
        "treatment": treatment,
    }


__all__ = [
    "BRANCH_REBIND_KIND",
    "BRANCH_TREATMENTS",
    "TargetRefreshBranchError",
    "publish_target_refresh_branch_checkpoint",
]
