"""Tests for integrity-checked CheckpointEnvelope v2 persistence."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from learned_ai.training.checkpoint_envelope import (
    CheckpointCompatibilityError,
    CheckpointDescriptor,
    CheckpointFormatError,
    CheckpointIntegrityError,
    CheckpointPayload,
    capture_rng_state,
    inspect_checkpoint,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    verify_checkpoint_compatibility,
)


def _descriptor(**changes) -> CheckpointDescriptor:
    values = {
        "checkpoint_id": "checkpoint-001",
        "run_id": "run-001",
        "experiment_id": "dev-v4-corrected",
        "parent_checkpoint_id": None,
        "role": "latest",
        "save_reason": "periodic",
        "created_at_utc": "2026-07-20T10:00:00Z",
        "config_sha256": "a" * 64,
        "feature_schema_version": "s-gen-v2-lookahead-122",
        "label_schema_version": "sector-corrected-v1",
        "database_schema_versions": {
            "specialist_db": "sector-corrected-v1"
        },
        "asset_identities": {"malom": "malom-identity"},
        "implementation": {"framework": "pytorch", "format": "python"},
    }
    values.update(changes)
    return CheckpointDescriptor(**values)


def _payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    return CheckpointPayload(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=None,
        scaler_state=None,
        rng_state=capture_rng_state(),
        trainer_state={
            "game_count": 7,
            "batch_count": 7,
            "update_count": 2,
            "difficulty": 1,
            "temperature": 0.8,
            "rolling_metrics": {"wins": [1.0, 0.5]},
            "curriculum": {"games_at_level": 7},
            "target_network": {"age": 7},
            "recovery_state": {"grace": 0},
            "model_config": {"policy_hidden": [16, 8]},
        },
        data_state={
            "cursor": {"next_game": 8},
            "consumed_snapshots": [],
            "cache": {},
            "buckets": {},
            "mutable_assets": {
                "specialist_db": {"sha256": "specialist-identity"}
            },
        },
    )


def test_checkpoint_round_trip_preserves_complete_state(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    payload = _payload(model, optimizer)
    path = tmp_path / "latest.pt"

    save_checkpoint(path, _descriptor(), payload)
    loaded = load_checkpoint(path)

    assert loaded.descriptor == _descriptor()
    assert loaded.payload.trainer_state["game_count"] == 7
    assert loaded.payload.optimizer_state is not None
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.payload.model_state[name], tensor)
    inspected, payload_hash, payload_size = inspect_checkpoint(path)
    assert inspected == loaded.descriptor
    assert payload_hash == loaded.payload_sha256
    assert payload_size == loaded.payload_size


def test_checkpoint_detects_payload_tampering(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "latest.pt"
    save_checkpoint(path, _descriptor(), _payload(model, optimizer))
    content = bytearray(path.read_bytes())
    content[-1] ^= 0x01
    path.write_bytes(content)

    with pytest.raises(CheckpointIntegrityError, match="SHA-256"):
        load_checkpoint(path)


def test_checkpoint_detects_descriptor_tampering(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "latest.pt"
    save_checkpoint(path, _descriptor(), _payload(model, optimizer))
    content = path.read_bytes()
    changed = content.replace(b"run-001", b"run-002", 1)
    assert changed != content
    path.write_bytes(changed)

    with pytest.raises(CheckpointIntegrityError, match="descriptor SHA-256"):
        load_checkpoint(path)


def test_checkpoint_rejects_legacy_or_truncated_files(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.pt"
    torch.save({"model": {}}, legacy)
    with pytest.raises(CheckpointFormatError, match="magic"):
        load_checkpoint(legacy)

    truncated = tmp_path / "truncated.pt"
    truncated.write_bytes(b"NMMCKP2\n")
    with pytest.raises(CheckpointFormatError, match="prefix is truncated"):
        load_checkpoint(truncated)


def test_checkpoint_payload_requires_complete_trainer_and_data_state() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    payload = _payload(model, optimizer)
    incomplete_trainer_state = dict(payload.trainer_state)
    incomplete_trainer_state.pop("recovery_state")

    with pytest.raises(CheckpointFormatError, match="trainer_state keys differ"):
        replace(payload, trainer_state=incomplete_trainer_state)

    incomplete_data_state = dict(payload.data_state)
    incomplete_data_state.pop("cursor")
    with pytest.raises(CheckpointFormatError, match="data_state keys differ"):
        replace(payload, data_state=incomplete_data_state)


def test_checkpoint_rotates_only_verified_previous_target(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "latest.pt"
    first = _descriptor(checkpoint_id="checkpoint-001")
    second = replace(
        first,
        checkpoint_id="checkpoint-002",
        parent_checkpoint_id="checkpoint-001",
    )

    save_checkpoint(path, first, _payload(model, optimizer), previous_copies=2)
    save_checkpoint(path, second, _payload(model, optimizer), previous_copies=2)

    assert load_checkpoint(path).descriptor.checkpoint_id == "checkpoint-002"
    assert load_checkpoint(tmp_path / "latest.prev0.pt").descriptor == first


def test_checkpoint_refuses_to_replace_corrupt_current_target(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "latest.pt"
    save_checkpoint(path, _descriptor(), _payload(model, optimizer))
    path.write_bytes(b"corrupt")

    with pytest.raises(CheckpointFormatError):
        save_checkpoint(
            path,
            replace(_descriptor(), checkpoint_id="checkpoint-002"),
            _payload(model, optimizer),
        )
    assert path.read_bytes() == b"corrupt"
    assert not (tmp_path / "latest.prev0.pt").exists()


def test_accepted_checkpoint_is_immutable(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "accepted.pt"
    descriptor = _descriptor(role="accepted")
    save_checkpoint(path, descriptor, _payload(model, optimizer))

    with pytest.raises(FileExistsError, match="accepted checkpoint already exists"):
        save_checkpoint(
            path,
            replace(descriptor, checkpoint_id="checkpoint-002"),
            _payload(model, optimizer),
        )


def test_exact_compatibility_reports_every_changed_semantic() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    envelope = type("Envelope", (), {"descriptor": _descriptor()})()

    with pytest.raises(CheckpointCompatibilityError) as raised:
        verify_checkpoint_compatibility(
            envelope,
            config_sha256="b" * 64,
            feature_schema_version="different-features",
            label_schema_version="different-labels",
            asset_identities={"malom": "different"},
            run_id="different-run",
        )

    message = str(raised.value)
    assert "config_sha256" in message
    assert "feature_schema_version" in message
    assert "label_schema_version" in message
    assert "asset_identities" in message
    assert "run_id" in message


def test_rng_capture_and_restore_replays_all_cpu_generators() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    component_rng = random.Random(23)
    state = capture_rng_state({"game": component_rng.getstate()})
    expected = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(())),
        component_rng.random(),
    )

    random.random()
    np.random.rand()
    torch.rand(())
    component_rng.random()
    restore_rng_state(state, component_rngs={"game": component_rng})

    actual = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(())),
        component_rng.random(),
    )
    assert actual == pytest.approx(expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rng_restore_normalizes_map_location_cuda_tensors() -> None:
    component_rng = random.Random(23)
    state = capture_rng_state({"game": component_rng.getstate()})
    expected_cpu = state["torch_cpu"].clone()
    relocated = dict(state)
    relocated["torch_cpu"] = state["torch_cpu"].cuda()
    relocated["torch_cuda"] = [item.cuda() for item in state["torch_cuda"]]
    torch.manual_seed(999)

    restore_rng_state(relocated, component_rngs={"game": component_rng})

    assert torch.equal(torch.get_rng_state(), expected_cpu)
