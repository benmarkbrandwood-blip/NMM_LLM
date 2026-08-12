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
from learned_ai.training.mature_target_refresh_fork import (
    MATURE_FORK_KIND,
    MatureTargetRefreshForkError,
    publish_mature_target_refresh_fork,
)


SHA = "a" * 64
SOURCE_CONFIG = "b" * 64
TARGET_CONFIG = "c" * 64
SPECIALIST = "d" * 64


def _source(tmp_path: Path, *, equal_target: bool = False) -> Path:
    model = {"weight": torch.tensor([1.0, 2.0])}
    target = {"weight": model["weight"].clone() if equal_target else torch.zeros(2)}
    payload = CheckpointPayload(
        model_state=model,
        optimizer_state={"state": {}, "param_groups": [{"lr": 1e-4}]},
        scheduler_state=None,
        scaler_state=None,
        rng_state=capture_rng_state(),
        trainer_state={
            "game_count": 439,
            "batch_count": 439,
            "update_count": 146,
            "difficulty": 1,
            "temperature": 0.8379,
            "rolling_metrics": {},
            "curriculum": {},
            "target_network": {
                "games_since_update": 439,
                "model_state": target,
            },
            "recovery_state": {
                "pending_steps": ["one", "two"],
                "optimizer_consumed_transition_count": 9_344,
                "target_refresh_fork_state": {
                    "schema_version": "nmm.target-refresh-fork-state.v1",
                    "fork_game": 50,
                    "captured": True,
                    "treatment": "no-refresh",
                    "post_fork_transition_origin": 1_152,
                },
            },
            "model_config": {
                "policy_hidden": [512, 256, 128],
                "move_feat_dim": 134,
                "value_input_dim": 80,
            },
        },
        data_state={
            "cursor": {},
            "consumed_snapshots": [],
            "cache": {},
            "buckets": {},
            "mutable_assets": {"specialist_db": {"sha256": SPECIALIST}},
        },
    )
    descriptor = CheckpointDescriptor(
        checkpoint_id="source:transition:8192",
        run_id="source-run",
        experiment_id="source-experiment",
        parent_checkpoint_id="parent",
        role="transition_diagnostic_candidate",
        save_reason="exact_post_fork_transition_8192",
        created_at_utc="2026-08-12T00:00:00Z",
        config_sha256=SOURCE_CONFIG,
        feature_schema_version="s-gen-v2-move-134-value-80",
        label_schema_version="sector-corrected-v1",
        database_schema_versions={"specialist_db": "sector-corrected-v1"},
        asset_identities={
            "specialist_db": SPECIALIST,
            "human_db": SHA,
        },
        implementation={"trainer": "s_gen_v2"},
    )
    path = tmp_path / "source.pt"
    save_checkpoint(path, descriptor, payload, previous_copies=0)
    return path


def _publish(source: Path, destination: Path, **changes: object):
    envelope = load_checkpoint(source)
    values: dict[str, object] = {
        "expected_source_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "expected_source_payload_sha256": envelope.payload_sha256,
        "expected_source_config_sha256": SOURCE_CONFIG,
        "expected_source_experiment_id": "source-experiment",
        "expected_source_game_count": 439,
        "expected_source_update_count": 146,
        "expected_source_post_fork_transitions": 8_192,
        "expected_specialist_db_sha256": SPECIALIST,
        "target_resume_config_sha256": TARGET_CONFIG,
        "target_experiment_id": "mature-experiment",
        "target_experiment_digest": SHA,
        "target_run_id": "mature-common-fork",
        "temperature_origin": 0.838,
        "created_at_utc": "2026-08-12T00:01:00Z",
    }
    values.update(changes)
    return publish_mature_target_refresh_fork(
        source,
        destination,
        **values,  # type: ignore[arg-type]
    )


def test_mature_fork_changes_only_declared_neutral_state(tmp_path: Path) -> None:
    source = _source(tmp_path)
    source_envelope = load_checkpoint(source)
    destination = tmp_path / "mature-fork.pt"

    record = _publish(source, destination)

    fork = load_checkpoint(destination)
    recovery = fork.payload.trainer_state["recovery_state"]
    assert recovery["pending_steps"] == []
    assert recovery["optimizer_consumed_transition_count"] == 9_344
    assert recovery["target_refresh_fork_state"] == {
        "schema_version": "nmm.target-refresh-fork-state.v1",
        "fork_game": 439,
        "captured": True,
        "treatment": None,
        "post_fork_transition_origin": None,
    }
    assert fork.payload.trainer_state["temperature"] == pytest.approx(0.838)
    assert torch.equal(
        fork.payload.model_state["weight"],
        source_envelope.payload.model_state["weight"],
    )
    assert torch.equal(
        fork.payload.trainer_state["target_network"]["model_state"]["weight"],
        source_envelope.payload.trainer_state["target_network"]["model_state"][
            "weight"
        ],
    )
    assert fork.descriptor.role == "target_refresh_fork"
    assert fork.descriptor.config_sha256 == TARGET_CONFIG
    assert fork.descriptor.experiment_id == "mature-experiment"
    assert fork.descriptor.implementation["mature_target_refresh_fork_kind"] == (
        MATURE_FORK_KIND
    )
    assert record["dropped_pending_transition_count"] == 2
    assert record["prior_post_fork_transitions"] == 8_192


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"expected_source_game_count": 440}, "game count differs"),
        ({"expected_source_update_count": 145}, "update count differs"),
        (
            {"expected_source_post_fork_transitions": 4_096},
            "prior treatment boundary differs",
        ),
        ({"expected_specialist_db_sha256": "e" * 64}, "SpecialistDB"),
        ({"temperature_origin": 0.1}, "out of range"),
    ],
)
def test_mature_fork_rejects_identity_or_state_drift(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    source = _source(tmp_path)
    with pytest.raises(MatureTargetRefreshForkError, match=message):
        _publish(source, tmp_path / "fork.pt", **changes)


def test_mature_fork_requires_a_stale_target(tmp_path: Path) -> None:
    source = _source(tmp_path, equal_target=True)
    with pytest.raises(MatureTargetRefreshForkError, match="already equals"):
        _publish(source, tmp_path / "fork.pt")


def test_mature_fork_never_overwrites_destination(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "fork.pt"
    destination.write_bytes(b"preserve")

    with pytest.raises(MatureTargetRefreshForkError, match="already exists"):
        _publish(source, destination)
    assert destination.read_bytes() == b"preserve"
