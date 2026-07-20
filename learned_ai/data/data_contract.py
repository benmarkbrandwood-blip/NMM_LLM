"""Versioned contracts for datasets, labels, and semantic cache identities."""

from __future__ import annotations

import math
import os
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar
from uuid import uuid4

from learned_ai.training.run_contract import (
    ContractValidationError,
    canonical_json_bytes,
    canonical_sha256,
)


DATASET_MANIFEST_SCHEMA = "nmm.dataset-manifest.v1"
TYPED_LABEL_SCHEMA = "nmm.typed-label.v1"
SEMANTIC_CACHE_KEY_SCHEMA = "nmm.semantic-cache-key.v1"

LABEL_KINDS = frozenset(
    {
        "theoretical_wdl",
        "empirical_outcome",
        "human_observation",
        "teacher_score",
        "model_prediction",
    }
)
WDL_LABEL_KINDS = frozenset({"theoretical_wdl", "empirical_outcome"})
PERSPECTIVES = frozenset({"W", "B"})


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field} must be a non-empty string")
    return value


def _sha256(value: Any, *, field: str) -> str:
    text = _text(value, field=field).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ContractValidationError(f"{field} must be a 64-character SHA-256")
    return text


def _strings(value: Sequence[str], *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractValidationError(f"{field} must be a string sequence")
    result = tuple(value)
    for item in result:
        _text(item, field=field)
    if len(set(result)) != len(result):
        raise ContractValidationError(f"{field} must not contain duplicates")
    return result


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise ContractValidationError(
            f"{name} keys differ; unknown={sorted(actual - expected)}, "
            f"missing={sorted(expected - actual)}"
        )


@dataclass(frozen=True)
class DatasetComponent:
    """One immutable file inside a dataset snapshot."""

    relative_path: str
    size_bytes: int
    sha256: str

    _FIELDS: ClassVar[set[str]] = {"relative_path", "size_bytes", "sha256"}

    def __post_init__(self) -> None:
        path = _text(self.relative_path, field="relative_path")
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ContractValidationError(
                "dataset component path must be relative and contained"
            )
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ContractValidationError("component size_bytes must be an integer")
        if self.size_bytes < 0:
            raise ContractValidationError(
                "component size_bytes must be non-negative"
            )
        object.__setattr__(self, "sha256", _sha256(self.sha256, field="sha256"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DatasetComponent:
        _exact_keys(value, cls._FIELDS, name="dataset component")
        return cls(**{field: value[field] for field in cls._FIELDS})


@dataclass(frozen=True)
class DatasetManifest:
    """Immutable provenance and trust declaration for one data snapshot."""

    dataset_id: str
    logical_name: str
    role: str
    source: str
    schema_version: str
    content_sha256: str
    size_bytes: int
    created_at_utc: str
    creation_process: str
    trust_level: str
    allowed_consumers: tuple[str, ...]
    validation: tuple[str, ...]
    exclusions: tuple[str, ...]
    label_kinds: tuple[str, ...]
    components: tuple[DatasetComponent, ...]

    _FIELDS: ClassVar[set[str]] = {
        "schema_version",
        "dataset_id",
        "logical_name",
        "role",
        "source",
        "data_schema_version",
        "content_sha256",
        "size_bytes",
        "created_at_utc",
        "creation_process",
        "trust_level",
        "allowed_consumers",
        "validation",
        "exclusions",
        "label_kinds",
        "components",
    }

    def __post_init__(self) -> None:
        for field in (
            "dataset_id",
            "logical_name",
            "role",
            "source",
            "schema_version",
            "created_at_utc",
            "creation_process",
            "trust_level",
        ):
            _text(getattr(self, field), field=field)
        object.__setattr__(
            self,
            "content_sha256",
            _sha256(self.content_sha256, field="content_sha256"),
        )
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ContractValidationError("size_bytes must be an integer")
        if self.size_bytes < 0:
            raise ContractValidationError("size_bytes must be non-negative")
        for field in (
            "allowed_consumers",
            "validation",
            "exclusions",
            "label_kinds",
        ):
            object.__setattr__(
                self, field, _strings(getattr(self, field), field=field)
            )
        unknown_labels = set(self.label_kinds) - LABEL_KINDS
        if unknown_labels:
            raise ContractValidationError(
                f"unsupported label kinds: {sorted(unknown_labels)}"
            )
        components = tuple(self.components)
        if any(not isinstance(item, DatasetComponent) for item in components):
            raise ContractValidationError(
                "components must contain DatasetComponent values"
            )
        component_paths = [item.relative_path for item in components]
        if len(set(component_paths)) != len(component_paths):
            raise ContractValidationError("dataset component paths must be unique")
        if tuple(sorted(component_paths)) != tuple(component_paths):
            raise ContractValidationError("dataset components must be path-sorted")
        if sum(item.size_bytes for item in components) != self.size_bytes:
            raise ContractValidationError(
                "dataset size_bytes must equal the component size total"
            )
        object.__setattr__(self, "components", components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DATASET_MANIFEST_SCHEMA,
            "dataset_id": self.dataset_id,
            "logical_name": self.logical_name,
            "role": self.role,
            "source": self.source,
            "data_schema_version": self.schema_version,
            "content_sha256": self.content_sha256,
            "size_bytes": self.size_bytes,
            "created_at_utc": self.created_at_utc,
            "creation_process": self.creation_process,
            "trust_level": self.trust_level,
            "allowed_consumers": list(self.allowed_consumers),
            "validation": list(self.validation),
            "exclusions": list(self.exclusions),
            "label_kinds": list(self.label_kinds),
            "components": [item.to_dict() for item in self.components],
        }

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DatasetManifest:
        _exact_keys(value, cls._FIELDS, name="dataset manifest")
        if value["schema_version"] != DATASET_MANIFEST_SCHEMA:
            raise ContractValidationError("unsupported dataset manifest schema")
        return cls(
            dataset_id=value["dataset_id"],
            logical_name=value["logical_name"],
            role=value["role"],
            source=value["source"],
            schema_version=value["data_schema_version"],
            content_sha256=value["content_sha256"],
            size_bytes=value["size_bytes"],
            created_at_utc=value["created_at_utc"],
            creation_process=value["creation_process"],
            trust_level=value["trust_level"],
            allowed_consumers=tuple(value["allowed_consumers"]),
            validation=tuple(value["validation"]),
            exclusions=tuple(value["exclusions"]),
            label_kinds=tuple(value["label_kinds"]),
            components=tuple(
                DatasetComponent.from_dict(item) for item in value["components"]
            ),
        )


def verify_dataset_snapshot(
    root: str | Path,
    manifest: DatasetManifest,
    *,
    full_hash: bool = False,
) -> dict[str, Any]:
    """Verify an exact component inventory, optionally rehashing every file."""
    base = Path(root)
    observed_paths = tuple(
        sorted(
            path.relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.is_file()
        )
    )
    expected_paths = tuple(item.relative_path for item in manifest.components)
    if observed_paths != expected_paths:
        raise ContractValidationError("dataset component inventory has changed")
    for component in manifest.components:
        path = base / component.relative_path
        if path.stat().st_size != component.size_bytes:
            raise ContractValidationError(
                f"dataset component size changed: {component.relative_path}"
            )
        if full_hash:
            import hashlib

            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != component.sha256:
                raise ContractValidationError(
                    f"dataset component hash changed: {component.relative_path}"
                )
    return {
        "manifest_sha256": manifest.manifest_sha256,
        "component_count": len(manifest.components),
        "size_bytes": manifest.size_bytes,
        "full_hash": full_hash,
    }


def publish_dataset_manifest(path: str | Path, manifest: DatasetManifest) -> None:
    """Atomically publish an immutable dataset manifest without overwrite."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"dataset manifest already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_json_bytes(manifest.to_dict()))
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise FileExistsError(f"dataset manifest already exists: {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    """Load a dataset manifest with strict schema validation."""
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractValidationError(
                    f"duplicate dataset manifest key: {key!r}"
                )
            result[key] = value
        return result
    try:
        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except ContractValidationError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ContractValidationError("cannot read dataset manifest") from exc
    return DatasetManifest.from_dict(raw)


@dataclass(frozen=True)
class TypedLabel:
    """One label with explicit meaning, viewpoint, authority, and validity."""

    kind: str
    value: str | float
    perspective: str
    rules_version: str
    history_identity: str | None
    source_identity: str
    validity_version: str

    _FIELDS: ClassVar[set[str]] = {
        "schema_version",
        "kind",
        "value",
        "perspective",
        "rules_version",
        "history_identity",
        "source_identity",
        "validity_version",
    }

    def __post_init__(self) -> None:
        if self.kind not in LABEL_KINDS:
            raise ContractValidationError(f"unsupported label kind: {self.kind!r}")
        if self.perspective not in PERSPECTIVES:
            raise ContractValidationError("label perspective must be W or B")
        for field in ("rules_version", "source_identity", "validity_version"):
            _text(getattr(self, field), field=field)
        if self.history_identity is not None:
            _text(self.history_identity, field="history_identity")
        if self.kind in WDL_LABEL_KINDS:
            if self.value not in {"W", "D", "L"}:
                raise ContractValidationError(f"{self.kind} must be W, D, or L")
        elif self.kind in {"teacher_score", "model_prediction"}:
            if (
                isinstance(self.value, bool)
                or not isinstance(self.value, (int, float))
                or not math.isfinite(float(self.value))
            ):
                raise ContractValidationError(f"{self.kind} must be finite numeric")
        elif isinstance(self.value, str):
            _text(self.value, field="human_observation")
        elif (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(float(self.value))
        ):
            raise ContractValidationError(
                "human_observation must be non-empty text or finite numeric"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TYPED_LABEL_SCHEMA,
            "kind": self.kind,
            "value": self.value,
            "perspective": self.perspective,
            "rules_version": self.rules_version,
            "history_identity": self.history_identity,
            "source_identity": self.source_identity,
            "validity_version": self.validity_version,
        }

    def swap_wdl_perspective(self) -> TypedLabel:
        """Swap viewpoint and reverse W/L for a typed WDL label."""
        if self.kind not in WDL_LABEL_KINDS:
            raise ContractValidationError("only WDL labels support perspective swap")
        return TypedLabel(
            kind=self.kind,
            value={"W": "L", "D": "D", "L": "W"}[str(self.value)],
            perspective="B" if self.perspective == "W" else "W",
            rules_version=self.rules_version,
            history_identity=self.history_identity,
            source_identity=self.source_identity,
            validity_version=self.validity_version,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TypedLabel:
        _exact_keys(value, cls._FIELDS, name="typed label")
        if value["schema_version"] != TYPED_LABEL_SCHEMA:
            raise ContractValidationError("unsupported typed label schema")
        return cls(
            **{
                field: value[field]
                for field in cls._FIELDS
                if field != "schema_version"
            }
        )


@dataclass(frozen=True)
class SemanticCacheKey:
    """Identity for a cached result including every semantic input dimension."""

    canonical_state: str
    history_identity: str
    rules_version: str
    perspective: str
    pending_action: str
    budget: Mapping[str, Any]
    model_identity: str
    feature_schema_version: str
    asset_identities: Mapping[str, str]
    config_sha256: str

    _FIELDS: ClassVar[set[str]] = {
        "schema_version",
        "canonical_state",
        "history_identity",
        "rules_version",
        "perspective",
        "pending_action",
        "budget",
        "model_identity",
        "feature_schema_version",
        "asset_identities",
        "config_sha256",
    }

    def __post_init__(self) -> None:
        for field in (
            "canonical_state",
            "history_identity",
            "rules_version",
            "pending_action",
            "model_identity",
            "feature_schema_version",
        ):
            _text(getattr(self, field), field=field)
        if self.perspective not in PERSPECTIVES:
            raise ContractValidationError("cache perspective must be W or B")
        object.__setattr__(
            self, "config_sha256", _sha256(self.config_sha256, field="config_sha256")
        )
        if not isinstance(self.budget, Mapping) or not self.budget:
            raise ContractValidationError("cache budget must be a non-empty mapping")
        if not isinstance(self.asset_identities, Mapping) or not self.asset_identities:
            raise ContractValidationError(
                "cache asset_identities must be a non-empty mapping"
            )
        object.__setattr__(self, "budget", MappingProxyType(dict(self.budget)))
        object.__setattr__(
            self, "asset_identities", MappingProxyType(dict(self.asset_identities))
        )
        canonical_json_bytes(dict(self.budget))
        canonical_json_bytes(dict(self.asset_identities))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SEMANTIC_CACHE_KEY_SCHEMA,
            "canonical_state": self.canonical_state,
            "history_identity": self.history_identity,
            "rules_version": self.rules_version,
            "perspective": self.perspective,
            "pending_action": self.pending_action,
            "budget": dict(self.budget),
            "model_identity": self.model_identity,
            "feature_schema_version": self.feature_schema_version,
            "asset_identities": dict(self.asset_identities),
            "config_sha256": self.config_sha256,
        }

    @property
    def identity(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticCacheKey:
        _exact_keys(value, cls._FIELDS, name="semantic cache key")
        if value["schema_version"] != SEMANTIC_CACHE_KEY_SCHEMA:
            raise ContractValidationError("unsupported semantic cache key schema")
        return cls(
            **{
                field: value[field]
                for field in cls._FIELDS
                if field != "schema_version"
            }
        )
