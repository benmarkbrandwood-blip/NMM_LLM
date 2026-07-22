from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from learned_ai.delivery.training_route_bundle import (
    TrainingRouteBundleError,
    export_training_route_bundle,
    load_training_route_models,
    verify_training_route_bundle,
)
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.training.checkpoint_envelope import (
    CheckpointDescriptor,
    CheckpointPayload,
    capture_rng_state,
    save_checkpoint,
)
from learned_ai.training.run_contract import (
    AssetManifestRef,
    RunManifest,
    canonical_sha256,
    publish_run_manifest,
)


def _checkpoint(
    path: Path,
    *,
    include_target: bool = True,
    move_feat_dim: int = 134,
) -> None:
    torch.manual_seed(11)
    model = ScaffoldedPolicyNet(
        move_feat_dim=move_feat_dim,
        value_input_dim=80,
        policy_hidden=(8,),
        value_hidden=(4,),
    )
    torch.manual_seed(12)
    target = ScaffoldedPolicyNet.from_config(model.get_config())
    descriptor = CheckpointDescriptor(
        checkpoint_id="c1",
        run_id="r1",
        experiment_id="e1",
        parent_checkpoint_id=None,
        role="candidate",
        save_reason="test",
        created_at_utc="2026-07-23T10:00:00Z",
        config_sha256="a" * 64,
        feature_schema_version="s-gen-v2-move-134-value-80",
        label_schema_version="sector-corrected-v1",
        database_schema_versions={
            "human_db_malom_columns": "masked-unversioned",
            "specialist_db": "sector-corrected-v1",
        },
        asset_identities={
            "human_db": "1" * 64,
            "malom_tablebase": "2" * 64,
            "specialist_db": "3" * 64,
        },
        implementation={"trainer": "s_gen_v2", "framework": "pytorch"},
    )
    target_network = (
        {"games_since_update": 6, "model_state": target.state_dict()}
        if include_target
        else {}
    )
    payload = CheckpointPayload(
        model_state=model.state_dict(),
        optimizer_state=None,
        scheduler_state=None,
        scaler_state=None,
        rng_state=capture_rng_state(),
        trainer_state={
            "game_count": 5000,
            "batch_count": 5000,
            "update_count": 1,
            "difficulty": 1,
            "temperature": 0.2,
            "rolling_metrics": {},
            "curriculum": {},
            "target_network": target_network,
            "recovery_state": {},
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
    save_checkpoint(path, descriptor, payload)


def _run_manifest(path: Path, *, sim_ply_depth: int = 5) -> None:
    resolved = {
        "sim_ply_depth": sim_ply_depth,
        "no_sentinel": True,
        "no_value_net": True,
        "no_gap_net": True,
        "human_db": "machine-local-human-db",
        "specialist_db": "machine-local-specialist-db",
        "malom": "machine-local-malom",
    }
    manifest = RunManifest(
        run_id="r1",
        experiment_id="e1",
        parent_run_id=None,
        status="preflight_passed",
        created_at_utc="2026-07-23T09:00:00Z",
        git_commit="b" * 40,
        git_dirty=False,
        git_diff_sha256=None,
        command=("python", "scripts/train_s_gen_v2.py"),
        resolved_config=resolved,
        config_sha256=canonical_sha256(resolved),
        environment={"pytorch": str(torch.__version__)},
        seeds={"run": 42},
        assets=(
            AssetManifestRef(
                logical_name="human_db",
                role="empirical_human_database",
                identity="1" * 64,
                schema_version="unversioned-malom-columns",
                trust_level="empirical_frequencies_and_outcomes",
                intended_use="human_frequencies_and_empirical_outcomes_only",
            ),
            AssetManifestRef(
                logical_name="malom_tablebase",
                role="training_oracle",
                identity="2" * 64,
                schema_version="malom-ultra-strong-sec2",
                trust_level="sector-corrected-v1",
                intended_use="lookahead_termination",
            ),
        ),
        components={"sentinel": False, "value_net": False, "gap_net": False},
        outputs={"run_directory": "output"},
        checkpoint_policy={"start_mode": "fresh"},
        claim_boundaries=("test-only",),
    )
    publish_run_manifest(path, manifest)


def test_training_route_bundle_round_trip_preserves_target_and_route(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    run_manifest = tmp_path / "run-manifest.json"
    license_path = tmp_path / "LICENSE"
    _checkpoint(checkpoint)
    _run_manifest(run_manifest)
    license_path.write_text("AGPL test notice", encoding="utf-8")

    manifest = export_training_route_bundle(
        checkpoint,
        run_manifest,
        tmp_path / "route-bundle",
        license_path=license_path,
    )
    report = verify_training_route_bundle(tmp_path / "route-bundle")
    policy, target, loaded = load_training_route_models(
        tmp_path / "route-bundle"
    )

    assert report["bundle_identity"] == manifest["bundle_identity"]
    assert manifest["route"]["sim_ply_depth"] == 5
    assert manifest["route"]["ply_depth"] == 12
    assert manifest["target"]["games_since_update"] == 6
    assert manifest["resources"]["specialist_db"]["identity"] == "3" * 64
    assert policy.get_config() == target.get_config()
    assert loaded == manifest


def test_training_route_bundle_rejects_checkpoint_without_target_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    run_manifest = tmp_path / "run-manifest.json"
    license_path = tmp_path / "LICENSE"
    _checkpoint(checkpoint, include_target=False)
    _run_manifest(run_manifest)
    license_path.write_text("AGPL test notice", encoding="utf-8")

    with pytest.raises(TrainingRouteBundleError, match="target network"):
        export_training_route_bundle(
            checkpoint,
            run_manifest,
            tmp_path / "route-bundle",
            license_path=license_path,
        )


def test_training_route_bundle_rejects_non_training_depth(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    run_manifest = tmp_path / "run-manifest.json"
    license_path = tmp_path / "LICENSE"
    _checkpoint(checkpoint)
    _run_manifest(run_manifest, sim_ply_depth=0)
    license_path.write_text("AGPL test notice", encoding="utf-8")

    with pytest.raises(TrainingRouteBundleError, match="simulation depth"):
        export_training_route_bundle(
            checkpoint,
            run_manifest,
            tmp_path / "route-bundle",
            license_path=license_path,
        )


def test_training_route_bundle_rejects_incompatible_feature_width(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    run_manifest = tmp_path / "run-manifest.json"
    license_path = tmp_path / "LICENSE"
    _checkpoint(checkpoint, move_feat_dim=62)
    _run_manifest(run_manifest)
    license_path.write_text("AGPL test notice", encoding="utf-8")

    with pytest.raises(TrainingRouteBundleError, match="feature schema"):
        export_training_route_bundle(
            checkpoint,
            run_manifest,
            tmp_path / "route-bundle",
            license_path=license_path,
        )


def test_training_route_bundle_rejects_target_weight_tampering(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    run_manifest = tmp_path / "run-manifest.json"
    license_path = tmp_path / "LICENSE"
    _checkpoint(checkpoint)
    _run_manifest(run_manifest)
    license_path.write_text("AGPL test notice", encoding="utf-8")
    export_training_route_bundle(
        checkpoint,
        run_manifest,
        tmp_path / "route-bundle",
        license_path=license_path,
    )
    weights = tmp_path / "route-bundle" / "target-weights.pt"
    weights.write_bytes(weights.read_bytes() + b"tamper")

    with pytest.raises(TrainingRouteBundleError, match="target weight integrity"):
        verify_training_route_bundle(tmp_path / "route-bundle")


def test_training_route_bundle_rejects_unknown_manifest_field(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    run_manifest = tmp_path / "run-manifest.json"
    license_path = tmp_path / "LICENSE"
    _checkpoint(checkpoint)
    _run_manifest(run_manifest)
    license_path.write_text("AGPL test notice", encoding="utf-8")
    export_training_route_bundle(
        checkpoint,
        run_manifest,
        tmp_path / "route-bundle",
        license_path=license_path,
    )
    manifest_path = tmp_path / "route-bundle" / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unknown"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TrainingRouteBundleError, match="unknown or incomplete"):
        verify_training_route_bundle(tmp_path / "route-bundle")
