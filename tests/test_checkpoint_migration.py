from __future__ import annotations

from pathlib import Path

import pytest
import torch

from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.training.checkpoint_envelope import CheckpointDescriptor, load_checkpoint
from learned_ai.training.checkpoint_migration import migrate_legacy_checkpoint


def _descriptor() -> CheckpointDescriptor:
    return CheckpointDescriptor(
        checkpoint_id="migrated-001",
        run_id="migration-run",
        experiment_id="legacy-import",
        parent_checkpoint_id=None,
        role="candidate",
        save_reason="explicit_weights_only_migration",
        created_at_utc="2026-07-20T10:00:00Z",
        config_sha256="a" * 64,
        feature_schema_version="test-features",
        label_schema_version="test-labels",
        database_schema_versions={"specialist_db": "test-db"},
        asset_identities={"source": "test-source"},
        implementation={"trainer": "test", "format": "pytorch"},
    )


def _legacy(path: Path) -> None:
    model = ScaffoldedPolicyNet(
        move_feat_dim=8,
        value_input_dim=6,
        policy_hidden=(4,),
        value_hidden=(3,),
    )
    torch.save({"model": model.state_dict(), "model_config": model.get_config(), "stage": "old"}, path)


def test_migration_defaults_to_side_effect_free_dry_run(tmp_path: Path) -> None:
    source = tmp_path / "legacy.pt"
    destination = tmp_path / "migrated.pt"
    _legacy(source)

    report = migrate_legacy_checkpoint(source, destination, _descriptor())

    assert report["dry_run"] is True
    assert report["mode"] == "weights-only"
    assert len(report["canary_identity"]) == 64
    assert not destination.exists()


def test_migration_writes_reset_weights_only_envelope(tmp_path: Path) -> None:
    source = tmp_path / "legacy.pt"
    destination = tmp_path / "migrated.pt"
    _legacy(source)

    migrate_legacy_checkpoint(source, destination, _descriptor(), write=True)
    loaded = load_checkpoint(destination)

    assert loaded.payload.optimizer_state is None
    assert loaded.payload.trainer_state["game_count"] == 0
    assert loaded.payload.trainer_state["recovery_state"]["migration_mode"] == "weights-only"
    with pytest.raises(FileExistsError):
        migrate_legacy_checkpoint(source, destination, _descriptor(), write=True)
