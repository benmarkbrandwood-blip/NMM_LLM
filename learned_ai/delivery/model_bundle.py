"""Self-describing, integrity-checked model bundles for local delivery."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.training.checkpoint_envelope import load_checkpoint
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.validation.model_canary import ModelCanary, capture_model_canary, verify_model_canary


BUNDLE_SCHEMA = "nmm.model-bundle.v1"
_MANIFEST_FIELDS = {
    "schema_version", "bundle_identity", "model_identity", "architecture",
    "weights", "inputs", "heads", "compatibility", "producer", "lineage",
    "runtime", "canary", "resources", "claims", "license",
}


class ModelBundleError(RuntimeError):
    """Raised when bundle semantics or integrity cannot be established."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_names(model: ScaffoldedPolicyNet) -> tuple[list[str], list[str]]:
    move = [f"sentinel_feature_{index:03d}" for index in range(58)]
    move.extend(["sentinel_score", "blended_absolute", "is_engine_top1", "blended_delta"])
    for ply in range(12):
        move.extend(
            f"lookahead_{ply:02d}_{name}"
            for name in ("heuristic", "learner_sentinel", "opponent_sentinel", "value", "gap", "simulated")
        )
    value = [f"board_context_{index:02d}" for index in range(23)]
    value.extend(
        f"history_{ply}_{name}"
        for ply in range(3)
        for name in ("from", "to", "capture")
    )
    value.extend(
        f"raw_board_{color}_{position:02d}"
        for color in ("white", "black")
        for position in range(24)
    )
    if len(move) != model.move_feat_dim or len(value) != model.value_input_dim:
        raise ModelBundleError("model dimensions do not match the supported v4 feature schema")
    return move, value


def export_model_bundle(
    checkpoint: str | Path,
    destination: str | Path,
    *,
    license_path: str | Path,
) -> dict[str, Any]:
    """Atomically export one immutable bundle from a verified v2 checkpoint."""
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"bundle destination exists: {target}")
    envelope = load_checkpoint(checkpoint, map_location="cpu")
    model = ScaffoldedPolicyNet.from_config(dict(envelope.payload.trainer_state["model_config"]))
    model.load_state_dict(envelope.payload.model_state, strict=True)
    canary = capture_model_canary(model)
    move_names, value_names = _feature_names(model)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        weights = temporary / "weights.pt"
        torch.save(model.state_dict(), weights)
        license_target = temporary / "LICENSE.txt"
        shutil.copyfile(license_path, license_target)
        weights_hash = _sha256_file(weights)
        body = {
            "schema_version": BUNDLE_SCHEMA,
            "model_identity": canonical_sha256(
                {"architecture": model.get_config(), "weights_sha256": weights_hash}
            ),
            "architecture": {"name": "ScaffoldedPolicyNet", "parameters": model.get_config()},
            "weights": {"path": "weights.pt", "format": "pytorch-state-dict", "sha256": weights_hash},
            "inputs": {
                "move": {"names": move_names, "dtype": "float32", "shape": [None, model.move_feat_dim], "perspective": "side-to-move"},
                "value": {"names": value_names, "dtype": "float32", "shape": [model.value_input_dim], "perspective": "side-to-move"},
                "normalization": "feature-specific-v4-encoder",
                "history": "three most recent moves plus board occupancy",
            },
            "heads": {
                "policy": {"shape": [None], "semantics": "one logit per supplied legal action", "transform": "softmax", "masking": "candidate-list defines legality"},
                "value": {"shape": [], "semantics": "side-to-move outcome estimate", "transform": "tanh"},
            },
            "compatibility": {"rules": "nmm-v4-corrected", "feature_schema": envelope.descriptor.feature_schema_version, "label_schema": envelope.descriptor.label_schema_version},
            "producer": {"run_id": envelope.descriptor.run_id, "checkpoint_id": envelope.descriptor.checkpoint_id, "checkpoint_payload_sha256": envelope.payload_sha256},
            "lineage": {"experiment_id": envelope.descriptor.experiment_id, "assets": dict(envelope.descriptor.asset_identities), "databases": dict(envelope.descriptor.database_schema_versions)},
            "runtime": {"framework": "pytorch", "backends": ["cpu", "cuda"], "precision": ["float32"]},
            "canary": {"path": "canary.json", "identity": canary.identity, "atol": 1e-6, "rtol": 1e-5},
            "resources": {"gpu_required": False, "single_gpu_evaluation": True},
            "claims": ["corrected-v4-infrastructure", "not-formal-playing-strength-evidence"],
            "license": {"path": "LICENSE.txt", "spdx": "AGPL-3.0-or-later", "sha256": _sha256_file(license_target)},
        }
        body["bundle_identity"] = canonical_sha256(body)
        (temporary / "canary.json").write_bytes(canonical_json_bytes(canary.to_dict()))
        (temporary / "bundle.json").write_bytes(canonical_json_bytes(body))
        verify_model_bundle(temporary)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, target)
        return body
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_model_bundle(path: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    """Verify strict manifest fields, file hashes, model loading, and canaries."""
    root = Path(path)
    manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise ModelBundleError("bundle manifest fields are unknown or incomplete")
    if manifest["schema_version"] != BUNDLE_SCHEMA:
        raise ModelBundleError("unsupported bundle schema")
    body = dict(manifest)
    identity = body.pop("bundle_identity")
    if canonical_sha256(body) != identity:
        raise ModelBundleError("bundle identity mismatch")
    for field in ("weights", "license"):
        artifact = root / manifest[field]["path"]
        if _sha256_file(artifact) != manifest[field]["sha256"]:
            raise ModelBundleError(f"{field} integrity mismatch")
    canary_value = json.loads((root / manifest["canary"]["path"]).read_text(encoding="utf-8"))
    canary = ModelCanary.from_dict(canary_value)
    if canary.identity != manifest["canary"]["identity"]:
        raise ModelBundleError("canary identity mismatch")
    model = ScaffoldedPolicyNet.from_config(manifest["architecture"]["parameters"])
    state = torch.load(root / manifest["weights"]["path"], map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    differences = verify_model_canary(
        model, canary, device=device,
        atol=float(manifest["canary"]["atol"]), rtol=float(manifest["canary"]["rtol"]),
    )
    return {"status": "verified", "bundle_identity": identity, "model_identity": manifest["model_identity"], "canary_differences": differences}


def compare_model_bundles(left: str | Path, right: str | Path) -> dict[str, Any]:
    """Compare declared identities and semantic schemas after verification."""
    left_report = verify_model_bundle(left)
    right_report = verify_model_bundle(right)
    left_manifest = json.loads((Path(left) / "bundle.json").read_text(encoding="utf-8"))
    right_manifest = json.loads((Path(right) / "bundle.json").read_text(encoding="utf-8"))
    fields = ("architecture", "inputs", "heads", "compatibility", "runtime")
    return {
        "left": left_report["bundle_identity"], "right": right_report["bundle_identity"],
        "same_model": left_report["model_identity"] == right_report["model_identity"],
        "semantic_differences": [field for field in fields if left_manifest[field] != right_manifest[field]],
    }
