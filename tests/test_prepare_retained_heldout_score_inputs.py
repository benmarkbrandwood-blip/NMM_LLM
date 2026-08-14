"""Tests for successor-owned retained held-out score input snapshots."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path

import pytest

import tools.prepare_retained_heldout_score_inputs as prepare


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_sources(root: Path) -> dict[str, dict]:
    result = {}
    for short, candidate_id in (("v3", "candidate-v3"), ("v4", "candidate-v4")):
        bundle = root / f"{short}-source-bundle"
        bundle.mkdir(parents=True)
        (bundle / "bundle.json").write_text(short, encoding="utf-8")
        (bundle / "policy-weights.pt").write_bytes(short.encode("ascii") * 5)
        database = root / f"{short}-source.sqlite"
        database.write_bytes((short + "-db").encode("ascii"))
        result[candidate_id] = {
            "source_bundle": bundle,
            "target_bundle": f"{short}-route-bundle",
            "bundle_identity": f"{short}-identity",
            "source_specialist_db": database,
            "target_specialist_db": f"{short}-specialist.sqlite",
            "specialist_db_sha256": _sha(database),
        }
    return result


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD | stat.S_IWRITE)


def test_prepare_creates_byte_exact_read_only_successor_snapshots(
    tmp_path,
    monkeypatch,
) -> None:
    candidates = _make_sources(tmp_path / "sources")
    target = tmp_path / "successor" / "inputs"
    identities = {
        config["target_bundle"]: config["bundle_identity"]
        for config in candidates.values()
    }
    monkeypatch.setattr(prepare, "_ROOT", tmp_path)
    monkeypatch.setattr(prepare, "TARGET_ROOT", target)
    monkeypatch.setattr(prepare, "CANDIDATES", candidates)
    monkeypatch.setattr(
        prepare,
        "build_source_manifest",
        lambda: {"snapshot_identity": prepare.SOURCE_SNAPSHOT_IDENTITY},
    )
    monkeypatch.setattr(
        prepare,
        "verify_training_route_bundle",
        lambda path, device: {"bundle_identity": identities[path.name]},
    )
    try:
        manifest = prepare.prepare_inputs()
        assert manifest == prepare.build_manifest()
        assert manifest["source_snapshot_identity"] == (
            prepare.SOURCE_SNAPSHOT_IDENTITY
        )
        assert manifest["copy_semantics"] == {
            "successor_owned": True,
            "byte_exact": True,
            "source_paths_reused_at_runtime": False,
            "sqlite_sidecars_absent": True,
            "files_marked_read_only": True,
        }
        assert all(
            item["route_bundle"]["read_only_files"]
            and item["specialist_db"]["read_only_file"]
            for item in manifest["candidates"]
        )
        with pytest.raises(FileExistsError):
            prepare.prepare_inputs()
    finally:
        _make_writable(target)


def test_prepare_rejects_a_source_database_sidecar(tmp_path, monkeypatch) -> None:
    candidates = _make_sources(tmp_path / "sources")
    first = next(iter(candidates.values()))
    Path(str(first["source_specialist_db"]) + "-wal").write_bytes(b"sidecar")
    monkeypatch.setattr(prepare, "_ROOT", tmp_path)
    monkeypatch.setattr(prepare, "TARGET_ROOT", tmp_path / "target" / "inputs")
    monkeypatch.setattr(prepare, "CANDIDATES", candidates)
    monkeypatch.setattr(
        prepare,
        "build_source_manifest",
        lambda: {"snapshot_identity": prepare.SOURCE_SNAPSHOT_IDENTITY},
    )

    with pytest.raises(prepare.HeldoutScoreInputError, match="sidecars exist"):
        prepare.prepare_inputs()
