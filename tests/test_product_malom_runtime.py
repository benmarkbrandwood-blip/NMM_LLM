from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ai.malom_runtime import resolve_product_malom_runtime
from learned_ai.data.data_contract import DatasetComponent, DatasetManifest


def _manifest(path: Path, root: Path, *, trust: str = "sector-corrected-v1") -> None:
    component = root / "std.secval"
    payload = component.read_bytes()
    value = DatasetManifest(
        dataset_id="test-malom",
        logical_name="malom_tablebase",
        role="training_oracle",
        source="focused test",
        schema_version="malom-ultra-strong-sec2",
        content_sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        created_at_utc="2026-08-18T00:00:00Z",
        creation_process="focused test",
        trust_level=trust,
        allowed_consumers=("malom_oracle",),
        validation=("component inventory",),
        exclusions=(),
        label_kinds=("theoretical_wdl",),
        components=(
            DatasetComponent(
                relative_path="std.secval",
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            ),
        ),
    )
    path.write_text(json.dumps(value.to_dict()), encoding="utf-8")


class _Oracle:
    def query_value(self, _board):
        raise NotImplementedError

    def move_value(self, _parent, _child):
        raise NotImplementedError

    def terminal_move_value(self, _parent, _child):
        raise NotImplementedError


class _Adapter:
    def __init__(self, db_path: str, *, strict: bool) -> None:
        assert strict is True
        self.db_path = db_path
        self.oracle = _Oracle()
        self.closed = False

    def is_available(self) -> bool:
        return True

    def require_complete_oracle(self) -> _Oracle:
        return self.oracle

    def close(self) -> None:
        self.closed = True


def _make_db(path: Path, payload: bytes = b"sector") -> Path:
    path.mkdir()
    (path / "std.secval").write_bytes(payload)
    return path


def test_resolver_rejects_invalid_override_then_selects_local_registry(
    tmp_path: Path,
) -> None:
    valid = _make_db(tmp_path / "valid")
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, valid)
    local = tmp_path / "training_paths.local.json"
    local.write_text(json.dumps({"malom_db_path": str(valid)}), encoding="utf-8")

    result = resolve_product_malom_runtime(
        repo_root=tmp_path,
        settings={"malom_db_path": str(tmp_path / "stale-shared")},
        sentinel_path="",
        local_paths_path=local,
        manifest_path=manifest,
        environment={"NMM_MALOM_DB": str(tmp_path / "missing-override")},
        adapter_factory=_Adapter,
    )

    assert result.database is not None
    assert result.oracle is result.database.oracle
    assert result.status["selected_source"] == "local-registry:malom_db_path"
    assert result.status["validation"] == "passed"
    assert result.status["manifest_sha256"]
    attempts = result.status["candidates"]
    assert attempts[0]["source"] == "environment:NMM_MALOM_DB"
    assert attempts[0]["status"] == "rejected"
    assert "does not exist" in attempts[0]["reason"]
    assert attempts[1]["status"] == "selected"
    shared = next(
        row
        for row in attempts
        if row["source"] == "shared-settings:malom_db_path"
    )
    assert shared["status"] == "rejected"
    assert "does not exist" in shared["reason"]


def test_resolver_checks_inventory_before_trying_next_candidate(tmp_path: Path) -> None:
    invalid = _make_db(tmp_path / "invalid", b"wrong-size")
    valid = _make_db(tmp_path / "valid", b"right")
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, valid)
    local = tmp_path / "training_paths.local.json"
    local.write_text(json.dumps({"malom_db_path": str(invalid)}), encoding="utf-8")

    result = resolve_product_malom_runtime(
        repo_root=tmp_path,
        settings={"malom_db_path": str(valid)},
        sentinel_path="",
        local_paths_path=local,
        manifest_path=manifest,
        environment={},
        adapter_factory=_Adapter,
    )

    assert result.status["selected_source"] == "shared-settings:malom_db_path"
    local_attempt = next(
        row
        for row in result.status["candidates"]
        if row["source"] == "local-registry:malom_db_path"
    )
    assert local_attempt["status"] == "rejected"
    assert "size changed" in local_attempt["reason"]


def test_resolver_rejects_untrusted_manifest_without_opening_adapter(
    tmp_path: Path,
) -> None:
    valid = _make_db(tmp_path / "valid")
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, valid, trust="historical-unversioned")
    opened: list[str] = []

    def _must_not_open(db_path: str, *, strict: bool):
        opened.append(db_path)
        return _Adapter(db_path, strict=strict)

    result = resolve_product_malom_runtime(
        repo_root=tmp_path,
        settings={"malom_db_path": str(valid)},
        sentinel_path="",
        local_paths_path=tmp_path / "absent.json",
        manifest_path=manifest,
        environment={},
        adapter_factory=_must_not_open,
    )

    assert result.database is None
    assert result.oracle is None
    assert result.status["validation"] == "failed"
    assert opened == []
    rejected = [
        row for row in result.status["candidates"] if row["status"] == "rejected"
    ]
    assert rejected
    assert "sector-corrected-v1" in rejected[0]["reason"]
