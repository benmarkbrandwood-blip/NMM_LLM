"""Explicit legacy-to-v2 checkpoint inspection and migration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.training.checkpoint_envelope import (
    CheckpointDescriptor,
    CheckpointFormatError,
    CheckpointPayload,
    capture_rng_state,
    is_checkpoint_envelope,
    save_checkpoint,
)
from learned_ai.validation.model_canary import capture_model_canary


def inspect_legacy_checkpoint(path: str | Path) -> dict[str, Any]:
    """Describe a legacy weight file without constructing a trainer."""
    source = Path(path)
    if is_checkpoint_envelope(source):
        raise CheckpointFormatError("source already uses checkpoint envelope v2")
    value = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise CheckpointFormatError("legacy checkpoint must be a mapping")
    state_key = "model" if "model" in value else "state_dict"
    if state_key not in value or not isinstance(value[state_key], dict):
        raise CheckpointFormatError("legacy checkpoint has no model state mapping")
    config = value.get("model_config")
    if not isinstance(config, dict):
        raise CheckpointFormatError("legacy checkpoint has no explicit model_config")
    tensors = value[state_key]
    return {
        "format": "legacy-pytorch-weights",
        "state_key": state_key,
        "model_config": config,
        "tensor_count": len(tensors),
        "stage": value.get("stage", "unknown"),
    }


def migrate_legacy_checkpoint(
    source: str | Path,
    destination: str | Path,
    descriptor: CheckpointDescriptor,
    *,
    write: bool = False,
) -> dict[str, Any]:
    """Dry-run or write a strict weights-only v2 migration with a model canary."""
    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("migration destination must differ from source")
    if destination_path.exists():
        raise FileExistsError(f"migration destination exists: {destination_path}")
    description = inspect_legacy_checkpoint(source_path)
    legacy = torch.load(source_path, map_location="cpu", weights_only=True)
    model = ScaffoldedPolicyNet.from_config(dict(description["model_config"]))
    model.load_state_dict(legacy[description["state_key"]], strict=True)
    canary = capture_model_canary(model)
    payload = CheckpointPayload(
        model_state=model.state_dict(),
        optimizer_state=None,
        scheduler_state=None,
        scaler_state=None,
        rng_state=capture_rng_state(),
        trainer_state={
            "game_count": 0,
            "batch_count": 0,
            "update_count": 0,
            "difficulty": 1,
            "temperature": 1.0,
            "rolling_metrics": {},
            "curriculum": {},
            "target_network": {},
            "recovery_state": {
                "source_checkpoint": str(source_path),
                "migration_mode": "weights-only",
            },
            "model_config": model.get_config(),
        },
        data_state={
            "cursor": {},
            "consumed_snapshots": [],
            "cache": {},
            "buckets": {},
            "mutable_assets": {},
        },
    )
    if write:
        save_checkpoint(destination_path, descriptor, payload)
    return {
        **description,
        "mode": "weights-only",
        "dry_run": not write,
        "destination": str(destination_path),
        "canary": canary.to_dict(),
        "canary_identity": canary.identity,
    }
