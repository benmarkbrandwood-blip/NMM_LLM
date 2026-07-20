"""Tests for versioned local run manifests and lifecycle event ledgers."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from learned_ai.training.run_contract import (
    AssetManifestRef,
    ContractValidationError,
    RunEvent,
    RunManifest,
    append_run_event,
    canonical_json_bytes,
    canonical_sha256,
    load_run_events,
    load_run_manifest,
    publish_run_manifest,
)


def _manifest(**changes) -> RunManifest:
    config = {
        "batch_games": 1,
        "max_games": 2,
        "temperature": 0.9,
        "features": ["board", "history"],
    }
    values = {
        "run_id": "run-001",
        "experiment_id": "dev-v4-corrected",
        "parent_run_id": None,
        "status": "preflight_passed",
        "created_at_utc": "2026-07-20T08:00:00Z",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "git_diff_sha256": None,
        "command": ("python", "scripts/train_s_gen_v2.py", "--max-games", "2"),
        "resolved_config": config,
        "config_sha256": canonical_sha256(config),
        "environment": {"python": "3.13.1", "device": "cpu"},
        "seeds": {"run": 42},
        "assets": (
            AssetManifestRef(
                logical_name="specialist_db",
                role="training_database",
                identity="empty-sector-corrected-v1",
                schema_version="sector-corrected-v1",
                trust_level="trusted",
                intended_use="corrected_labels_and_empirical_results",
            ),
        ),
        "components": {
            "sentinel": False,
            "value_net": False,
            "gap_net": False,
            "ppo": False,
        },
        "outputs": {"run_directory": "runs/run-001"},
        "checkpoint_policy": {
            "start_mode": "fresh",
            "roles": ["latest", "candidate"],
        },
        "claim_boundaries": (
            "integration evidence only",
            "not v5 acceptance evidence",
        ),
    }
    values.update(changes)
    return RunManifest(**values)


def _event(
    *,
    sequence: int = 0,
    status: str = "planned",
    previous_event_sha256: str | None = None,
) -> RunEvent:
    return RunEvent(
        run_id="run-001",
        sequence=sequence,
        timestamp_utc=f"2026-07-20T08:00:0{sequence}Z",
        status=status,
        event_type="status_changed",
        reason_code=None,
        details={"source": "test"},
        previous_event_sha256=previous_event_sha256,
    )


def test_canonical_json_has_stable_key_order_and_rejects_nonfinite_values() -> None:
    assert canonical_json_bytes({"z": 1, "a": [True, None]}) == (
        b'{"a":[true,null],"z":1}'
    )

    with pytest.raises(ContractValidationError, match="non-finite"):
        canonical_json_bytes({"loss": float("nan")})


def test_manifest_is_immutable_and_has_stable_identity() -> None:
    manifest = _manifest()
    original_hash = manifest.manifest_sha256

    with pytest.raises(TypeError):
        manifest.resolved_config["max_games"] = 4

    round_trip = RunManifest.from_dict(manifest.to_dict())
    assert round_trip == manifest
    assert round_trip.manifest_sha256 == original_hash


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"status": "unknown"}, "unsupported run status"),
        ({"created_at_utc": "2026-07-20"}, "RFC 3339 UTC"),
        ({"git_dirty": True}, "git_diff_sha256"),
        ({"config_sha256": "0" * 64}, "does not match"),
        ({"components": {"sentinel": "disabled"}}, "components must map"),
        ({"claim_boundaries": ()}, "claim_boundaries"),
    ],
)
def test_manifest_rejects_invalid_or_inconsistent_values(changes, message) -> None:
    with pytest.raises(ContractValidationError, match=message):
        _manifest(**changes)


def test_manifest_loader_rejects_unknown_keys_and_schema(tmp_path: Path) -> None:
    value = _manifest().to_dict()
    value["unexpected"] = True
    path = tmp_path / "run-manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ContractValidationError, match="unknown keys"):
        load_run_manifest(path)

    value.pop("unexpected")
    value["schema_version"] = "nmm.run-manifest.v999"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ContractValidationError, match="unsupported.*schema"):
        load_run_manifest(path)


@pytest.mark.parametrize("field", ["command", "assets", "claim_boundaries"])
def test_manifest_loader_rejects_strings_in_place_of_arrays(
    tmp_path: Path, field: str
) -> None:
    value = _manifest().to_dict()
    value[field] = "not-an-array"
    path = tmp_path / "run-manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ContractValidationError, match=f"{field} must be a JSON array"):
        load_run_manifest(path)


@pytest.mark.parametrize(
    "field", ["resolved_config", "environment", "seeds", "components", "outputs"]
)
def test_manifest_loader_rejects_arrays_in_place_of_objects(
    tmp_path: Path, field: str
) -> None:
    value = _manifest().to_dict()
    value[field] = []
    path = tmp_path / "run-manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ContractValidationError, match=f"{field} must be a JSON object"):
        load_run_manifest(path)


def test_manifest_publish_is_canonical_atomic_and_no_overwrite(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "nested" / "run-manifest.json"

    publish_run_manifest(path, manifest)

    assert path.read_bytes() == canonical_json_bytes(manifest.to_dict()) + b"\n"
    assert load_run_manifest(path) == manifest
    assert not list(path.parent.glob("*.tmp"))
    with pytest.raises(FileExistsError, match="already exists"):
        publish_run_manifest(path, replace(manifest, run_id="run-002"))
    assert load_run_manifest(path) == manifest


def test_event_ledger_round_trip_and_hash_chain(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = _event()
    second = _event(
        sequence=1,
        status="running",
        previous_event_sha256=first.event_sha256,
    )

    append_run_event(path, first)
    append_run_event(path, second)

    assert load_run_events(path) == (first, second)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["event_sha256"] == first.event_sha256
    assert records[1]["event"]["previous_event_sha256"] == first.event_sha256


def test_event_append_rejects_gap_wrong_run_and_wrong_hash(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = _event()
    append_run_event(path, first)

    with pytest.raises(ContractValidationError, match="sequence"):
        append_run_event(path, _event(sequence=2))

    wrong_run = replace(
        _event(sequence=1, previous_event_sha256=first.event_sha256),
        run_id="run-002",
    )
    with pytest.raises(ContractValidationError, match="run identities"):
        append_run_event(path, wrong_run)

    wrong_hash = _event(sequence=1, previous_event_sha256="0" * 64)
    with pytest.raises(ContractValidationError, match="hash chain"):
        append_run_event(path, wrong_hash)


def test_event_loader_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = _event()
    append_run_event(path, first)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["event"]["details"]["source"] = "tampered"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ContractValidationError, match="invalid event hash"):
        load_run_events(path)
