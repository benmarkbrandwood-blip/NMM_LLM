"""Tests for dataset, typed-label, and semantic-cache contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from learned_ai.data.data_contract import (
    DatasetManifest,
    DatasetComponent,
    SemanticCacheKey,
    TypedLabel,
    load_dataset_manifest,
    publish_dataset_manifest,
    verify_dataset_snapshot,
)
from learned_ai.training.run_contract import ContractValidationError


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_id="malom-sector-corrected-v1",
        logical_name="malom_tablebase",
        role="training_oracle",
        source="local Malom import",
        schema_version="malom-ultra-strong-sec2",
        content_sha256="a" * 64,
        size_bytes=3,
        created_at_utc="2026-07-20T12:00:00Z",
        creation_process="validated inventory import",
        trust_level="sector-corrected-v1",
        allowed_consumers=("generalist_preflight", "malom_oracle"),
        validation=("component inventory", "sector decoder regression"),
        exclusions=("historical unversioned labels",),
        label_kinds=("theoretical_wdl",),
        components=(
            DatasetComponent(
                relative_path="std.secval", size_bytes=3, sha256="d" * 64
            ),
        ),
    )


def _cache_key() -> SemanticCacheKey:
    return SemanticCacheKey(
        canonical_state="state-v1",
        history_identity="history-v1",
        rules_version="nmm-rules-v1",
        perspective="W",
        pending_action="capture:pending",
        budget={"nodes": 1000},
        model_identity="model-v1",
        feature_schema_version="features-v1",
        asset_identities={"malom": "malom-v1"},
        config_sha256="b" * 64,
    )


def test_dataset_manifest_round_trip_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "malom.manifest.json"
    manifest = _manifest()

    publish_dataset_manifest(path, manifest)

    assert load_dataset_manifest(path) == manifest
    assert len(manifest.manifest_sha256) == 64
    with pytest.raises(FileExistsError):
        publish_dataset_manifest(path, manifest)


def test_dataset_snapshot_verification_checks_inventory_size_and_hash(
    tmp_path: Path,
) -> None:
    component = tmp_path / "std.secval"
    component.write_bytes(b"abc")
    import hashlib

    manifest = replace(
        _manifest(),
        components=(
            DatasetComponent(
                relative_path="std.secval",
                size_bytes=3,
                sha256=hashlib.sha256(b"abc").hexdigest(),
            ),
        ),
    )

    assert verify_dataset_snapshot(tmp_path, manifest, full_hash=True)[
        "component_count"
    ] == 1
    component.write_bytes(b"abcd")
    with pytest.raises(ContractValidationError, match="size changed"):
        verify_dataset_snapshot(tmp_path, manifest)


def test_dataset_manifest_rejects_unknown_schema_and_label_kind() -> None:
    raw = _manifest().to_dict()
    raw["schema_version"] = "unknown"
    with pytest.raises(ContractValidationError, match="unsupported"):
        DatasetManifest.from_dict(raw)
    with pytest.raises(ContractValidationError, match="label kinds"):
        replace(_manifest(), label_kinds=("generic_value",))


def test_typed_wdl_perspective_swap_is_an_involution() -> None:
    label = TypedLabel(
        kind="theoretical_wdl",
        value="W",
        perspective="W",
        rules_version="nmm-rules-v1",
        history_identity="fifty-move-state-v1",
        source_identity="malom-v1",
        validity_version="sector-corrected-v1",
    )

    swapped = label.swap_wdl_perspective()

    assert swapped.value == "L"
    assert swapped.perspective == "B"
    assert swapped.swap_wdl_perspective() == label
    assert TypedLabel.from_dict(label.to_dict()) == label


@pytest.mark.parametrize(
    ("changes",),
    [
        ({"canonical_state": "state-v2"},),
        ({"history_identity": "history-v2"},),
        ({"rules_version": "nmm-rules-v2"},),
        ({"perspective": "B"},),
        ({"pending_action": "capture:none"},),
        ({"budget": {"nodes": 1001}},),
        ({"model_identity": "model-v2"},),
        ({"feature_schema_version": "features-v2"},),
        ({"asset_identities": {"malom": "malom-v2"}},),
        ({"config_sha256": "c" * 64},),
    ],
)
def test_every_semantic_cache_dimension_changes_identity(changes: dict) -> None:
    original = _cache_key()

    changed = replace(original, **changes)

    assert changed.identity != original.identity
    assert SemanticCacheKey.from_dict(original.to_dict()) == original
