"""Tests for descriptor-only target-refresh fork publication."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from learned_ai.training.checkpoint_envelope import (
    CheckpointDescriptor,
    CheckpointPayload,
    capture_rng_state,
    load_checkpoint,
    save_checkpoint,
)
from learned_ai.training.target_refresh_branch import (
    BRANCH_REBIND_KIND,
    TargetRefreshBranchError,
    publish_target_refresh_branch_checkpoint,
)


SPECIALIST_SHA256 = "d" * 64
SOURCE_CONFIG_SHA256 = "a" * 64
TARGET_CONFIG_SHA256 = "b" * 64
TARGET_EXPERIMENT_DIGEST = "c" * 64
EXPERIMENT_ID = "target-refresh-equal-transition-seed64"


def _payload(
    *,
    game_count: int = 50,
    specialist_sha256: str = SPECIALIST_SHA256,
    treatment: str | None = None,
) -> CheckpointPayload:
    return CheckpointPayload(
        model_state={"weight": torch.arange(6, dtype=torch.float32)},
        optimizer_state={"state": {}, "param_groups": [{"lr": 0.00005}]},
        scheduler_state=None,
        scaler_state=None,
        rng_state=capture_rng_state({"game": (3, (1, 2, 3), None)}),
        trainer_state={
            "game_count": game_count,
            "batch_count": 50,
            "update_count": 17,
            "difficulty": 1,
            "temperature": 0.85,
            "rolling_metrics": {"wins": [0.0, 0.5, 1.0]},
            "curriculum": {"games_at_level": 50},
            "target_network": {"games_since_update": 50},
            "recovery_state": {
                "target_refresh_fork_state": {
                    "schema_version": "nmm.target-refresh-fork-state.v1",
                    "fork_game": 50,
                    "captured": True,
                    "treatment": treatment,
                    "post_fork_transition_origin": None,
                },
                "optimizer_consumed_transition_count": 4096,
            },
            "model_config": {
                "policy_hidden": [512, 256, 128],
                "move_feat_dim": 122,
                "value_input_dim": 147,
            },
        },
        data_state={
            "cursor": {"completed_games": game_count},
            "consumed_snapshots": [],
            "cache": {},
            "buckets": {},
            "mutable_assets": {
                "specialist_db": {"sha256": specialist_sha256}
            },
        },
    )


def _descriptor(**changes: object) -> CheckpointDescriptor:
    values: dict[str, object] = {
        "checkpoint_id": "prefix-segment-0001:target-refresh-fork:1",
        "run_id": "prefix-segment-0001",
        "experiment_id": EXPERIMENT_ID,
        "parent_checkpoint_id": None,
        "role": "target_refresh_fork",
        "save_reason": "target-refresh-fork",
        "created_at_utc": "2026-08-11T00:00:00Z",
        "config_sha256": SOURCE_CONFIG_SHA256,
        "feature_schema_version": "s-gen-v2-lookahead-122",
        "label_schema_version": "sector-corrected-v1",
        "database_schema_versions": {
            "specialist_db": "sector-corrected-v1"
        },
        "asset_identities": {
            "malom_tablebase": "malom-identity",
            "specialist_db": SPECIALIST_SHA256,
        },
        "implementation": {
            "trainer": "s_gen_v2",
            "experiment_digest": "sha256:" + "e" * 64,
            "mif_suite_tag": "mif-suite-1.0",
        },
    }
    values.update(changes)
    return CheckpointDescriptor(**values)  # type: ignore[arg-type]


def _source_checkpoint(
    tmp_path: Path,
    *,
    descriptor: CheckpointDescriptor | None = None,
    payload: CheckpointPayload | None = None,
) -> Path:
    source = tmp_path / "target-refresh-fork.pt"
    save_checkpoint(
        source,
        descriptor or _descriptor(),
        payload or _payload(),
        previous_copies=0,
    )
    return source


def _publish(source: Path, destination: Path, **changes: object):
    values: dict[str, object] = {
        "treatment": "refresh-once",
        "expected_source_config_sha256": SOURCE_CONFIG_SHA256,
        "target_resume_config_sha256": TARGET_CONFIG_SHA256,
        "expected_experiment_id": EXPERIMENT_ID,
        "expected_game_count": 50,
        "expected_specialist_db_sha256": SPECIALIST_SHA256,
        "target_experiment_digest": TARGET_EXPERIMENT_DIGEST,
        "created_at_utc": "2026-08-11T00:01:00Z",
    }
    values.update(changes)
    return publish_target_refresh_branch_checkpoint(
        source,
        destination,
        **values,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("treatment", ["refresh-once", "no-refresh"])
def test_branch_rebind_changes_only_descriptor_identity(
    tmp_path: Path,
    treatment: str,
) -> None:
    source = _source_checkpoint(tmp_path)
    source_bytes = source.read_bytes()
    source_envelope = load_checkpoint(source)
    destination = tmp_path / treatment / "initial-target-refresh-fork.pt"

    record = _publish(source, destination, treatment=treatment)

    branch = load_checkpoint(destination)
    assert source.read_bytes() == source_bytes
    assert branch.payload_sha256 == source_envelope.payload_sha256
    assert torch.equal(
        branch.payload.model_state["weight"],
        source_envelope.payload.model_state["weight"],
    )
    descriptor = branch.descriptor
    assert descriptor.role == "target_refresh_fork"
    assert descriptor.run_id == source_envelope.descriptor.run_id
    assert descriptor.experiment_id == EXPERIMENT_ID
    assert descriptor.config_sha256 == TARGET_CONFIG_SHA256
    assert descriptor.parent_checkpoint_id == source_envelope.descriptor.checkpoint_id
    assert descriptor.checkpoint_id.endswith(f":branch:{treatment}")
    assert descriptor.save_reason == f"target-refresh-branch-rebind-{treatment}"
    implementation = dict(descriptor.implementation)
    assert implementation["target_refresh_branch_kind"] == BRANCH_REBIND_KIND
    assert implementation["target_refresh_branch_treatment"] == treatment
    assert implementation["target_refresh_branch_source_checkpoint_id"] == (
        source_envelope.descriptor.checkpoint_id
    )
    assert implementation["target_refresh_branch_source_payload_sha256"] == (
        source_envelope.payload_sha256
    )
    assert implementation["experiment_digest"] == (
        "sha256:" + TARGET_EXPERIMENT_DIGEST
    )
    expected_unchanged = source_envelope.descriptor.to_dict()
    observed = descriptor.to_dict()
    for field in (
        "checkpoint_id",
        "parent_checkpoint_id",
        "save_reason",
        "created_at_utc",
        "config_sha256",
        "implementation",
    ):
        expected_unchanged.pop(field)
        observed.pop(field)
    assert observed == expected_unchanged
    assert record["source_payload_sha256"] == branch.payload_sha256
    assert record["branch_payload_sha256"] == branch.payload_sha256
    assert record["destination_sha256"] == hashlib.sha256(
        destination.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"treatment": "capture"}, "treatment is unsupported"),
        (
            {"expected_source_config_sha256": "f" * 64},
            "configuration differs",
        ),
        ({"expected_experiment_id": "another"}, "experiment differs"),
        ({"expected_game_count": 49}, "game count differs"),
        (
            {"expected_specialist_db_sha256": "f" * 64},
            "SpecialistDB differs",
        ),
    ],
)
def test_branch_rebind_rejects_mismatched_expectations(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    source = _source_checkpoint(tmp_path)

    with pytest.raises(TargetRefreshBranchError, match=message):
        _publish(source, tmp_path / "branch.pt", **changes)


def test_branch_rebind_rejects_nonfork_treated_or_rebound_source(
    tmp_path: Path,
) -> None:
    nonfork = _source_checkpoint(tmp_path / "nonfork", descriptor=_descriptor(role="latest"))
    with pytest.raises(TargetRefreshBranchError, match="role is not a fork"):
        _publish(nonfork, tmp_path / "nonfork-branch.pt")

    treated = _source_checkpoint(
        tmp_path / "treated",
        payload=_payload(treatment="refresh-once"),
    )
    with pytest.raises(TargetRefreshBranchError, match="treatment state differs"):
        _publish(treated, tmp_path / "treated-branch.pt")

    implementation = dict(_descriptor().implementation)
    implementation["target_refresh_branch_kind"] = BRANCH_REBIND_KIND
    rebound = _source_checkpoint(
        tmp_path / "rebound",
        descriptor=_descriptor(implementation=implementation),
    )
    with pytest.raises(TargetRefreshBranchError, match="already rebound"):
        _publish(rebound, tmp_path / "rebound-branch.pt")


def test_branch_rebind_rejects_existing_destination(tmp_path: Path) -> None:
    source = _source_checkpoint(tmp_path)
    destination = tmp_path / "branch.pt"
    destination.write_bytes(b"do not overwrite")

    with pytest.raises(TargetRefreshBranchError, match="already exists"):
        _publish(source, destination)
    assert destination.read_bytes() == b"do not overwrite"
