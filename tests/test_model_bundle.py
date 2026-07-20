from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from learned_ai.delivery.model_bundle import ModelBundleError, export_model_bundle, verify_model_bundle
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.training.checkpoint_envelope import CheckpointDescriptor, CheckpointPayload, capture_rng_state, save_checkpoint


def _checkpoint(path: Path) -> None:
    model = ScaffoldedPolicyNet(move_feat_dim=134, value_input_dim=80, policy_hidden=(8,), value_hidden=(4,))
    descriptor = CheckpointDescriptor(
        checkpoint_id="c1", run_id="r1", experiment_id="e1", parent_checkpoint_id=None,
        role="candidate", save_reason="test", created_at_utc="2026-07-20T10:00:00Z",
        config_sha256="a" * 64, feature_schema_version="s-gen-v2-move-134-value-80",
        label_schema_version="sector-corrected-v1", database_schema_versions={"db": "v1"},
        asset_identities={"data": "identity"}, implementation={"trainer": "test"},
    )
    payload = CheckpointPayload(
        model_state=model.state_dict(), optimizer_state=None, scheduler_state=None,
        scaler_state=None, rng_state=capture_rng_state(),
        trainer_state={"game_count": 0, "batch_count": 0, "update_count": 0, "difficulty": 1,
            "temperature": 1.0, "rolling_metrics": {}, "curriculum": {}, "target_network": {},
            "recovery_state": {}, "model_config": model.get_config()},
        data_state={"cursor": {}, "consumed_snapshots": [], "cache": {}, "buckets": {}, "mutable_assets": {}},
    )
    save_checkpoint(path, descriptor, payload)


def test_bundle_export_and_verify_round_trip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    license_path = tmp_path / "LICENSE"
    license_path.write_text("AGPL test notice", encoding="utf-8")
    _checkpoint(checkpoint)

    manifest = export_model_bundle(checkpoint, tmp_path / "bundle", license_path=license_path)
    report = verify_model_bundle(tmp_path / "bundle")

    assert report["bundle_identity"] == manifest["bundle_identity"]
    assert len(manifest["inputs"]["move"]["names"]) == 134
    assert len(manifest["inputs"]["value"]["names"]) == 80


def test_bundle_rejects_weight_tampering(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    license_path = tmp_path / "LICENSE"
    license_path.write_text("AGPL test notice", encoding="utf-8")
    _checkpoint(checkpoint)
    export_model_bundle(checkpoint, tmp_path / "bundle", license_path=license_path)
    weights = tmp_path / "bundle" / "weights.pt"
    weights.write_bytes(weights.read_bytes() + b"tamper")

    with pytest.raises(ModelBundleError, match="weights integrity"):
        verify_model_bundle(tmp_path / "bundle")


def test_bundle_rejects_unknown_mandatory_field(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    license_path = tmp_path / "LICENSE"
    license_path.write_text("AGPL test notice", encoding="utf-8")
    _checkpoint(checkpoint)
    export_model_bundle(checkpoint, tmp_path / "bundle", license_path=license_path)
    manifest_path = tmp_path / "bundle" / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unknown"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelBundleError, match="unknown or incomplete"):
        verify_model_bundle(tmp_path / "bundle")
