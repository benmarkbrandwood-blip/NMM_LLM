"""Validated machine-local Malom resolution for the product runtime.

The resolver deliberately separates portable trust metadata from machine-local
paths.  It chooses the first configured candidate whose directory exactly
matches the tracked ``sector-corrected-v1`` manifest inventory.  A non-empty
path is never treated as evidence that the database is usable.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from learned_ai.data.data_contract import (
    DatasetManifest,
    load_dataset_manifest,
    verify_dataset_snapshot,
)
from learned_ai.data.malom_label_provenance import CURRENT_MALOM_LABEL_VERSION
from learned_ai.sentinel.db_teacher import ExternalSolvedDB


@dataclass(frozen=True)
class ProductMalomRuntime:
    """One resolved read-only adapter plus JSON-safe resolution evidence."""

    database: Any | None
    oracle: Any | None
    status: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_mapping(path: Path) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, Mapping):
        raise ValueError("root value is not a JSON object")
    return value


def _manifest_contract(path: Path) -> tuple[DatasetManifest, dict[str, Any]]:
    manifest = load_dataset_manifest(path)
    if manifest.logical_name != "malom_tablebase":
        raise RuntimeError("manifest logical_name is not malom_tablebase")
    if manifest.trust_level != CURRENT_MALOM_LABEL_VERSION:
        raise RuntimeError("manifest trust level is not sector-corrected-v1")
    if "theoretical_wdl" not in manifest.label_kinds:
        raise RuntimeError("manifest does not declare theoretical_wdl labels")
    if "malom_oracle" not in manifest.allowed_consumers:
        raise RuntimeError("manifest does not authorize the malom_oracle consumer")
    return manifest, {
        "manifest_path": str(path),
        "manifest_file_sha256": _sha256_file(path),
        "manifest_sha256": manifest.manifest_sha256,
        "content_sha256": manifest.content_sha256,
        "label_version": manifest.trust_level,
        "component_count": len(manifest.components),
        "size_bytes": manifest.size_bytes,
    }


def _candidate_path(value: Any, *, repo_root: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return str(path.resolve())


def resolve_product_malom_runtime(
    *,
    repo_root: str | Path,
    settings: Mapping[str, Any],
    sentinel_path: str,
    local_paths_path: str | Path,
    manifest_path: str | Path,
    environment: Mapping[str, str] | None = None,
    adapter_factory: Callable[..., Any] = ExternalSolvedDB,
) -> ProductMalomRuntime:
    """Resolve and validate the first usable product Malom candidate.

    Priority is explicit environment override, ignored machine-local path
    registry, legacy shared settings, then Sentinel configuration.  Every
    rejected candidate remains visible in ``status["candidates"]``.
    """

    root = Path(repo_root).resolve()
    local_path = Path(local_paths_path)
    if not local_path.is_absolute():
        local_path = root / local_path
    trusted_manifest_path = Path(manifest_path)
    if not trusted_manifest_path.is_absolute():
        trusted_manifest_path = root / trusted_manifest_path
    env = os.environ if environment is None else environment

    local_value: Any = ""
    local_error = ""
    if local_path.is_file():
        try:
            local_value = _strict_json_mapping(local_path).get("malom_db_path", "")
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            local_error = (
                "cannot read machine-local path registry: "
                f"{type(exc).__name__}: {exc}"
            )

    raw_candidates = [
        ("environment:NMM_MALOM_DB", env.get("NMM_MALOM_DB", ""), ""),
        ("local-registry:malom_db_path", local_value, local_error),
        ("shared-settings:malom_db_path", settings.get("malom_db_path", ""), ""),
        ("sentinel-config:external_db_path", sentinel_path, ""),
    ]
    candidates = [
        {
            "source": source,
            "path": _candidate_path(value, repo_root=root),
            "status": "pending",
            "reason": configuration_error,
        }
        for source, value, configuration_error in raw_candidates
    ]

    manifest: DatasetManifest | None = None
    manifest_status: dict[str, Any] = {
        "manifest_path": str(trusted_manifest_path.resolve()),
        "manifest_file_sha256": None,
        "manifest_sha256": None,
        "content_sha256": None,
        "label_version": None,
        "component_count": None,
        "size_bytes": None,
    }
    manifest_error = ""
    try:
        manifest, manifest_status = _manifest_contract(trusted_manifest_path)
    except Exception as exc:
        manifest_error = f"{type(exc).__name__}: {exc}"

    selected_database: Any | None = None
    selected_oracle: Any | None = None
    selected_source: str | None = None
    selected_path: str | None = None
    inventory_status: dict[str, Any] | None = None
    for candidate in candidates:
        if candidate["reason"]:
            candidate["status"] = "rejected"
            continue
        if not candidate["path"]:
            candidate["status"] = "not-configured"
            candidate["reason"] = "candidate path is empty"
            continue
        if manifest is None:
            candidate["status"] = "rejected"
            candidate["reason"] = f"trusted manifest validation failed: {manifest_error}"
            continue

        database: Any | None = None
        try:
            database_path = Path(candidate["path"])
            if not database_path.exists():
                raise RuntimeError("candidate path does not exist")
            if not database_path.is_dir():
                raise RuntimeError("candidate path is not a directory")
            candidate_inventory = verify_dataset_snapshot(
                database_path,
                manifest,
                full_hash=False,
            )
            database = adapter_factory(str(database_path), strict=True)
            if not database.is_available():
                raise RuntimeError("Malom adapter reports unavailable")
            oracle = database.require_complete_oracle()
            required = ("query_value", "move_value", "terminal_move_value")
            missing = [
                name for name in required if not callable(getattr(oracle, name, None))
            ]
            if missing:
                raise RuntimeError(
                    "complete oracle surface is missing " + ", ".join(missing)
                )
        except Exception as exc:
            close = getattr(database, "close", None)
            if callable(close):
                close()
            candidate["status"] = "rejected"
            candidate["reason"] = f"{type(exc).__name__}: {exc}"
            continue

        if selected_database is None:
            candidate["status"] = "selected"
            candidate["reason"] = "manifest, inventory, adapter, and oracle validated"
            selected_database = database
            selected_oracle = oracle
            selected_source = candidate["source"]
            selected_path = candidate["path"]
            inventory_status = candidate_inventory
        else:
            candidate["status"] = "validated-not-selected"
            candidate["reason"] = "a higher-priority validated candidate was selected"
            close = getattr(database, "close", None)
            if callable(close):
                close()

    status = {
        "schema_version": "nmm.product-malom-runtime-status.v1",
        "validation": "passed" if selected_database is not None else "failed",
        "selected_source": selected_source,
        "selected_path": selected_path,
        **manifest_status,
        "inventory_validation": inventory_status,
        "full_component_hashes_recomputed": False,
        "candidates": candidates,
        "disabled_reason": (
            "" if selected_database is not None else "no validated Malom candidate"
        ),
    }
    return ProductMalomRuntime(
        database=selected_database,
        oracle=selected_oracle,
        status=status,
    )
