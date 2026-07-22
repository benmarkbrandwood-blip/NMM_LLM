"""Immutable policy bundles for the exact s_gen_v2 training input route."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import torch

from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.training.checkpoint_envelope import load_checkpoint
from learned_ai.training.run_contract import (
    canonical_json_bytes,
    canonical_sha256,
    load_run_manifest,
)
from learned_ai.validation.model_canary import (
    ModelCanary,
    capture_model_canary,
    verify_model_canary,
)


TRAINING_ROUTE_BUNDLE_SCHEMA = "nmm.training-route-bundle.v1"
TRAINING_ROUTE_NAME = "s-gen-v2-training-aligned-v1"

_MANIFEST_FIELDS = {
    "schema_version",
    "bundle_identity",
    "architecture",
    "policy",
    "target",
    "route",
    "resources",
    "producer",
    "runtime",
    "claims",
    "license",
}
_MODEL_FIELDS = {"model_identity", "weights", "canary"}
_WEIGHT_FIELDS = {"path", "format", "sha256"}
_CANARY_FIELDS = {"path", "identity", "atol", "rtol"}
_RESOURCE_FIELDS = {"identity", "mode", "purpose", "schema_version"}
_PRODUCER_FIELDS = {
    "run_id",
    "experiment_id",
    "checkpoint_id",
    "checkpoint_payload_sha256",
    "run_manifest_sha256",
}
_RUNTIME_FIELDS = {"framework", "backends", "precision"}
_ROUTE_FIELDS = {
    "name",
    "feature_width",
    "ply_depth",
    "sim_ply_depth",
    "lookahead_signals_per_ply",
    "learner_continuation",
    "opponent_continuation",
    "terminal_order",
    "components",
}
_ROUTE_COMPONENTS = {
    "sentinel": False,
    "value_net": False,
    "gap_net": False,
    "human_db": True,
    "specialist_db": True,
    "malom_tablebase": True,
}
_RESOURCE_PURPOSES = {
    "human_db": "lookahead-opponent-frequency",
    "specialist_db": "base-counterfactual-features",
    "malom_tablebase": "lookahead-terminal-early-exit",
}


class TrainingRouteBundleError(RuntimeError):
    """Raised when an aligned route bundle is incomplete or altered."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise TrainingRouteBundleError(f"{field} is not a SHA-256 identity")
    return value


def _artifact_path(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TrainingRouteBundleError(f"{field} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != value:
        raise TrainingRouteBundleError(f"{field} path must be one local file name")
    return root / relative


def _model_identity(config: Mapping[str, Any], weights_sha256: str) -> str:
    return canonical_sha256(
        {"architecture": dict(config), "weights_sha256": weights_sha256}
    )


def _route_contract(sim_ply_depth: int) -> dict[str, Any]:
    return {
        "name": TRAINING_ROUTE_NAME,
        "feature_width": 134,
        "ply_depth": 12,
        "sim_ply_depth": sim_ply_depth,
        "lookahead_signals_per_ply": 6,
        "learner_continuation": "frozen-target-argmax-zero-lookahead-v1",
        "opponent_continuation": "human-frequency-top1-else-heuristic-v1",
        "terminal_order": "rules-then-malom-v1",
        "components": dict(_ROUTE_COMPONENTS),
    }


def _validate_route(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _ROUTE_FIELDS:
        raise TrainingRouteBundleError("route fields are unknown or incomplete")
    if value["name"] != TRAINING_ROUTE_NAME:
        raise TrainingRouteBundleError("route name is unsupported")
    if value["feature_width"] != 134 or value["ply_depth"] != 12:
        raise TrainingRouteBundleError("route feature dimensions are unsupported")
    depth = value["sim_ply_depth"]
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 12:
        raise TrainingRouteBundleError("route simulation depth is invalid")
    expected = _route_contract(depth)
    if value != expected:
        raise TrainingRouteBundleError("route semantics differ from the supported contract")
    return value


def _resource_contract(
    descriptor: Any,
) -> dict[str, dict[str, str]]:
    identities = dict(descriptor.asset_identities)
    schemas = dict(descriptor.database_schema_versions)
    required = set(_RESOURCE_PURPOSES)
    if not required.issubset(identities):
        missing = sorted(required - set(identities))
        raise TrainingRouteBundleError(
            "checkpoint lacks route resource identities: " + ", ".join(missing)
        )
    for name in required:
        _require_sha256(identities[name], field=f"{name} identity")
    if schemas.get("human_db_malom_columns") != "masked-unversioned":
        raise TrainingRouteBundleError("HumanDB schema is not the trained contract")
    if schemas.get("specialist_db") != "sector-corrected-v1":
        raise TrainingRouteBundleError(
            "SpecialistDB schema is not sector-corrected-v1"
        )
    return {
        "human_db": {
            "identity": identities["human_db"],
            "mode": "read-only",
            "purpose": _RESOURCE_PURPOSES["human_db"],
            "schema_version": schemas.get(
                "human_db_malom_columns", "unknown"
            ),
        },
        "specialist_db": {
            "identity": identities["specialist_db"],
            "mode": "read-only",
            "purpose": _RESOURCE_PURPOSES["specialist_db"],
            "schema_version": schemas.get("specialist_db", "unknown"),
        },
        "malom_tablebase": {
            "identity": identities["malom_tablebase"],
            "mode": "read-only",
            "purpose": _RESOURCE_PURPOSES["malom_tablebase"],
            "schema_version": "malom-ultra-strong-sec2",
        },
    }


def _validate_export_contract(envelope: Any, run_manifest: Any) -> int:
    if envelope.descriptor.run_id != run_manifest.run_id:
        raise TrainingRouteBundleError("checkpoint and run manifest run IDs differ")
    if envelope.descriptor.experiment_id != run_manifest.experiment_id:
        raise TrainingRouteBundleError(
            "checkpoint and run manifest experiment IDs differ"
        )
    if run_manifest.git_dirty:
        raise TrainingRouteBundleError("training route requires a clean producer run")
    if envelope.descriptor.feature_schema_version != (
        "s-gen-v2-move-134-value-80"
    ):
        raise TrainingRouteBundleError("checkpoint feature schema is incompatible")
    if envelope.descriptor.label_schema_version != "sector-corrected-v1":
        raise TrainingRouteBundleError("checkpoint label schema is incompatible")
    if envelope.descriptor.implementation.get("trainer") != "s_gen_v2":
        raise TrainingRouteBundleError("checkpoint trainer is incompatible")

    config = dict(run_manifest.resolved_config)
    depth = config.get("sim_ply_depth")
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 12:
        raise TrainingRouteBundleError("training simulation depth is invalid")
    for flag in ("no_sentinel", "no_value_net", "no_gap_net"):
        if config.get(flag) is not True:
            raise TrainingRouteBundleError(
                f"training route requires explicit {flag}=true"
            )
    for component in ("sentinel", "value_net", "gap_net"):
        if run_manifest.components.get(component) is not False:
            raise TrainingRouteBundleError(
                f"run manifest does not disable {component}"
            )
    for resource in ("human_db", "specialist_db", "malom"):
        if not isinstance(config.get(resource), str) or not config[resource]:
            raise TrainingRouteBundleError(
                f"training route lacks configured {resource}"
            )

    assets = {asset.logical_name: asset.identity for asset in run_manifest.assets}
    checkpoint_assets = dict(envelope.descriptor.asset_identities)
    for name in ("human_db", "malom_tablebase"):
        if assets.get(name) != checkpoint_assets.get(name):
            raise TrainingRouteBundleError(
                f"checkpoint and run manifest {name} identities differ"
            )
    return depth


def _load_source_models(envelope: Any) -> tuple[ScaffoldedPolicyNet, ScaffoldedPolicyNet, int]:
    trainer_state = envelope.payload.trainer_state
    target_state = trainer_state.get("target_network")
    if not isinstance(target_state, Mapping) or set(target_state) != {
        "games_since_update",
        "model_state",
    }:
        raise TrainingRouteBundleError("checkpoint target network state is incomplete")
    target_age = target_state["games_since_update"]
    if isinstance(target_age, bool) or not isinstance(target_age, int) or target_age < 0:
        raise TrainingRouteBundleError("checkpoint target network age is invalid")
    model_config = trainer_state.get("model_config")
    if not isinstance(model_config, Mapping):
        raise TrainingRouteBundleError("checkpoint model configuration is missing")
    if (
        model_config.get("move_feat_dim") != 134
        or model_config.get("value_input_dim") != 80
    ):
        raise TrainingRouteBundleError("checkpoint feature schema is incompatible")
    try:
        policy = ScaffoldedPolicyNet.from_config(dict(model_config))
        target = ScaffoldedPolicyNet.from_config(dict(model_config))
        policy.load_state_dict(envelope.payload.model_state, strict=True)
        target.load_state_dict(target_state["model_state"], strict=True)
        policy.eval()
        target.eval()
    except (KeyError, TypeError, RuntimeError, ValueError) as exc:
        raise TrainingRouteBundleError(
            "checkpoint policy or target network is incompatible"
        ) from exc
    return policy, target, target_age


def _model_section(
    model: ScaffoldedPolicyNet,
    weights: Path,
    canary_path: Path,
) -> dict[str, Any]:
    weights_hash = _sha256_file(weights)
    canary = capture_model_canary(model)
    canary_path.write_bytes(canonical_json_bytes(canary.to_dict()))
    return {
        "model_identity": _model_identity(model.get_config(), weights_hash),
        "weights": {
            "path": weights.name,
            "format": "pytorch-state-dict",
            "sha256": weights_hash,
        },
        "canary": {
            "path": canary_path.name,
            "identity": canary.identity,
            "atol": 1e-6,
            "rtol": 1e-5,
        },
    }


def export_training_route_bundle(
    checkpoint: str | Path,
    run_manifest_path: str | Path,
    destination: str | Path,
    *,
    license_path: str | Path,
) -> dict[str, Any]:
    """Export final policy and frozen target with their exact route contract."""
    target_root = Path(destination)
    if target_root.exists():
        raise FileExistsError(f"route bundle destination exists: {target_root}")
    envelope = load_checkpoint(checkpoint, map_location="cpu")
    run_manifest = load_run_manifest(run_manifest_path)
    sim_ply_depth = _validate_export_contract(envelope, run_manifest)
    policy_model, target_model, target_age = _load_source_models(envelope)
    resources = _resource_contract(envelope.descriptor)

    temporary = target_root.with_name(
        f".{target_root.name}.{uuid4().hex}.tmp"
    )
    temporary.mkdir(parents=True)
    try:
        policy_weights = temporary / "policy-weights.pt"
        target_weights = temporary / "target-weights.pt"
        torch.save(policy_model.state_dict(), policy_weights)
        torch.save(target_model.state_dict(), target_weights)
        policy = _model_section(
            policy_model, policy_weights, temporary / "policy-canary.json"
        )
        target = _model_section(
            target_model, target_weights, temporary / "target-canary.json"
        )
        target["games_since_update"] = target_age

        license_target = temporary / "LICENSE.txt"
        shutil.copyfile(license_path, license_target)
        body = {
            "schema_version": TRAINING_ROUTE_BUNDLE_SCHEMA,
            "architecture": {
                "name": "ScaffoldedPolicyNet",
                "parameters": policy_model.get_config(),
            },
            "policy": policy,
            "target": target,
            "route": _route_contract(sim_ply_depth),
            "resources": resources,
            "producer": {
                "run_id": envelope.descriptor.run_id,
                "experiment_id": envelope.descriptor.experiment_id,
                "checkpoint_id": envelope.descriptor.checkpoint_id,
                "checkpoint_payload_sha256": envelope.payload_sha256,
                "run_manifest_sha256": run_manifest.manifest_sha256,
            },
            "runtime": {
                "framework": "pytorch",
                "backends": ["cpu", "cuda"],
                "precision": ["float32"],
            },
            "claims": [
                "training-input-route-reconstructable",
                "not-strength-evidence-without-frozen-protocol",
            ],
            "license": {
                "path": license_target.name,
                "spdx": "AGPL-3.0-or-later",
                "sha256": _sha256_file(license_target),
            },
        }
        manifest = {**body, "bundle_identity": canonical_sha256(body)}
        persisted_manifest = json.loads(canonical_json_bytes(manifest))
        (temporary / "bundle.json").write_bytes(
            canonical_json_bytes(persisted_manifest)
        )
        verify_training_route_bundle(temporary)
        target_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, target_root)
        return persisted_manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_manifest(root: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingRouteBundleError("cannot read training route manifest") from exc
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise TrainingRouteBundleError(
            "training route manifest fields are unknown or incomplete"
        )
    if manifest["schema_version"] != TRAINING_ROUTE_BUNDLE_SCHEMA:
        raise TrainingRouteBundleError("unsupported training route bundle schema")
    body = dict(manifest)
    identity = body.pop("bundle_identity")
    _require_sha256(identity, field="bundle identity")
    if canonical_sha256(body) != identity:
        raise TrainingRouteBundleError("training route bundle identity mismatch")
    return manifest


def _verify_model_section(
    root: Path,
    section: Any,
    config: Mapping[str, Any],
    *,
    name: str,
    device: str,
) -> tuple[ScaffoldedPolicyNet, dict[str, float]]:
    expected_fields = _MODEL_FIELDS | ({"games_since_update"} if name == "target" else set())
    if not isinstance(section, dict) or set(section) != expected_fields:
        raise TrainingRouteBundleError(f"{name} fields are unknown or incomplete")
    weights = section["weights"]
    canary_value = section["canary"]
    if not isinstance(weights, dict) or set(weights) != _WEIGHT_FIELDS:
        raise TrainingRouteBundleError(f"{name} weight fields are invalid")
    if not isinstance(canary_value, dict) or set(canary_value) != _CANARY_FIELDS:
        raise TrainingRouteBundleError(f"{name} canary fields are invalid")
    if weights["format"] != "pytorch-state-dict":
        raise TrainingRouteBundleError(f"{name} weight format is unsupported")
    weights_path = _artifact_path(root, weights["path"], field=f"{name} weight")
    expected_hash = _require_sha256(
        weights["sha256"], field=f"{name} weight identity"
    )
    try:
        observed_hash = _sha256_file(weights_path)
    except OSError as exc:
        raise TrainingRouteBundleError(f"cannot read {name} weights") from exc
    if observed_hash != expected_hash:
        raise TrainingRouteBundleError(f"{name} weight integrity mismatch")
    if section["model_identity"] != _model_identity(config, expected_hash):
        raise TrainingRouteBundleError(f"{name} model identity mismatch")

    canary_path = _artifact_path(
        root, canary_value["path"], field=f"{name} canary"
    )
    try:
        canary = ModelCanary.from_dict(
            json.loads(canary_path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as exc:
        raise TrainingRouteBundleError(f"cannot read {name} canary") from exc
    if canary.identity != canary_value["identity"]:
        raise TrainingRouteBundleError(f"{name} canary identity mismatch")
    try:
        model = ScaffoldedPolicyNet.from_config(dict(config))
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        differences = verify_model_canary(
            model,
            canary,
            device=device,
            atol=float(canary_value["atol"]),
            rtol=float(canary_value["rtol"]),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TrainingRouteBundleError(
            f"{name} model or canary verification failed"
        ) from exc
    return model, differences


def verify_training_route_bundle(
    path: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Verify route semantics, both models, all hashes, and both canaries."""
    root = Path(path)
    manifest = _load_manifest(root)
    _validate_route(manifest["route"])
    architecture = manifest["architecture"]
    if (
        not isinstance(architecture, dict)
        or set(architecture) != {"name", "parameters"}
        or architecture["name"] != "ScaffoldedPolicyNet"
        or not isinstance(architecture["parameters"], dict)
    ):
        raise TrainingRouteBundleError("bundle architecture is invalid")
    config = architecture["parameters"]
    if config.get("move_feat_dim") != 134 or config.get("value_input_dim") != 80:
        raise TrainingRouteBundleError("bundle feature schema is incompatible")

    resources = manifest["resources"]
    if not isinstance(resources, dict) or set(resources) != set(_RESOURCE_PURPOSES):
        raise TrainingRouteBundleError("route resources are unknown or incomplete")
    for name, resource in resources.items():
        if not isinstance(resource, dict) or set(resource) != _RESOURCE_FIELDS:
            raise TrainingRouteBundleError(f"{name} resource fields are invalid")
        _require_sha256(resource["identity"], field=f"{name} identity")
        if resource["mode"] != "read-only":
            raise TrainingRouteBundleError(f"{name} is not bound read-only")
        if resource["purpose"] != _RESOURCE_PURPOSES[name]:
            raise TrainingRouteBundleError(f"{name} purpose is incompatible")
        if not isinstance(resource["schema_version"], str) or not resource[
            "schema_version"
        ]:
            raise TrainingRouteBundleError(f"{name} schema version is invalid")

    producer = manifest["producer"]
    if not isinstance(producer, dict) or set(producer) != _PRODUCER_FIELDS:
        raise TrainingRouteBundleError("producer fields are unknown or incomplete")
    for field in ("run_id", "experiment_id", "checkpoint_id"):
        if not isinstance(producer[field], str) or not producer[field]:
            raise TrainingRouteBundleError(f"producer {field} is invalid")
    for field in ("checkpoint_payload_sha256", "run_manifest_sha256"):
        _require_sha256(producer[field], field=f"producer {field}")

    runtime = manifest["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != _RUNTIME_FIELDS:
        raise TrainingRouteBundleError("runtime fields are unknown or incomplete")
    if runtime != {
        "framework": "pytorch",
        "backends": ["cpu", "cuda"],
        "precision": ["float32"],
    }:
        raise TrainingRouteBundleError("runtime contract is unsupported")
    if manifest["claims"] != [
        "training-input-route-reconstructable",
        "not-strength-evidence-without-frozen-protocol",
    ]:
        raise TrainingRouteBundleError("bundle claims are unsupported")

    policy, policy_differences = _verify_model_section(
        root, manifest["policy"], config, name="policy", device=device
    )
    target, target_differences = _verify_model_section(
        root, manifest["target"], config, name="target", device=device
    )
    target_age = manifest["target"]["games_since_update"]
    if isinstance(target_age, bool) or not isinstance(target_age, int) or target_age < 0:
        raise TrainingRouteBundleError("target network age is invalid")

    license_value = manifest["license"]
    if not isinstance(license_value, dict) or set(license_value) != {
        "path",
        "spdx",
        "sha256",
    }:
        raise TrainingRouteBundleError("license fields are invalid")
    license_file = _artifact_path(
        root, license_value["path"], field="license"
    )
    try:
        license_hash = _sha256_file(license_file)
    except OSError as exc:
        raise TrainingRouteBundleError("cannot read route bundle license") from exc
    expected_license_hash = _require_sha256(
        license_value["sha256"], field="license identity"
    )
    if license_hash != expected_license_hash:
        raise TrainingRouteBundleError("license integrity mismatch")
    if license_value["spdx"] != "AGPL-3.0-or-later":
        raise TrainingRouteBundleError("route bundle license is unsupported")

    del policy, target
    return {
        "status": "verified",
        "bundle_identity": manifest["bundle_identity"],
        "policy_canary_differences": policy_differences,
        "target_canary_differences": target_differences,
    }


def load_training_route_models(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[ScaffoldedPolicyNet, ScaffoldedPolicyNet, dict[str, Any]]:
    """Return strictly verified policy and frozen-target models."""
    root = Path(path)
    verify_training_route_bundle(root, device=device)
    manifest = _load_manifest(root)
    config = manifest["architecture"]["parameters"]
    policy, _ = _verify_model_section(
        root, manifest["policy"], config, name="policy", device=device
    )
    target, _ = _verify_model_section(
        root, manifest["target"], config, name="target", device=device
    )
    return policy, target, manifest
