"""Versioned, fail-closed contracts for local training runs.

The module deliberately has no PyTorch dependency.  Configuration resolution
and resource probing can therefore construct and verify a run contract before
CUDA, databases, or model output are initialized.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar
from uuid import uuid4


RUN_MANIFEST_SCHEMA = "nmm.run-manifest.v1"
RUN_EVENT_SCHEMA = "nmm.run-event.v1"

RUN_STATUSES = frozenset(
    {
        "planned",
        "preflight_passed",
        "running",
        "interrupted",
        "failed",
        "completed",
        "quarantined",
    }
)


class ContractValidationError(ValueError):
    """Raised when a persisted training contract is malformed or inconsistent."""


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    actual = set(value)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown or missing:
        parts: list[str] = []
        if unknown:
            parts.append(f"unknown keys: {', '.join(unknown)}")
        if missing:
            parts.append(f"missing keys: {', '.join(missing)}")
        raise ContractValidationError(f"{context} has {'; '.join(parts)}")


def _require_nonempty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field} must be a non-empty string")
    return value


def _require_optional_text(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_text(value, field=field)


def _require_utc_timestamp(value: Any, *, field: str) -> str:
    text = _require_nonempty_text(value, field=field)
    if not text.endswith("Z"):
        raise ContractValidationError(f"{field} must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(f"{text[:-1]}+00:00")
    except ValueError as exc:
        raise ContractValidationError(
            f"{field} must be an RFC 3339 UTC timestamp"
        ) from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ContractValidationError(f"{field} must use UTC")
    return text


def _require_sha256(value: Any, *, field: str) -> str:
    text = _require_nonempty_text(value, field=field).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ContractValidationError(f"{field} must be a 64-character SHA-256")
    return text


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be a JSON object")
    return value


def _require_json_array(value: Any, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ContractValidationError(f"{field} must be a JSON array")
    return value


def _freeze_json(value: Any, *, field: str = "value") -> Any:
    """Validate JSON data and return an immutable recursive representation."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractValidationError(f"{field} contains a non-string key")
            frozen[key] = _freeze_json(item, field=f"{field}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return tuple(
            _freeze_json(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    raise ContractValidationError(
        f"{field} contains a non-JSON value of type {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize validated JSON with stable ordering and no insignificant space."""
    frozen = _freeze_json(value)
    return json.dumps(
        _thaw_json(frozen),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 identity of canonical JSON data."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class AssetManifestRef:
    """Identity and trust declaration for one run input asset."""

    logical_name: str
    role: str
    identity: str
    schema_version: str
    trust_level: str
    intended_use: str

    _FIELDS: ClassVar[set[str]] = {
        "logical_name",
        "role",
        "identity",
        "schema_version",
        "trust_level",
        "intended_use",
    }

    def __post_init__(self) -> None:
        for field_name in self._FIELDS:
            _require_nonempty_text(getattr(self, field_name), field=field_name)

    def to_dict(self) -> dict[str, str]:
        return {
            "logical_name": self.logical_name,
            "role": self.role,
            "identity": self.identity,
            "schema_version": self.schema_version,
            "trust_level": self.trust_level,
            "intended_use": self.intended_use,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AssetManifestRef:
        if not isinstance(value, Mapping):
            raise ContractValidationError("asset must be a JSON object")
        _require_exact_keys(value, cls._FIELDS, context="asset")
        return cls(**{field_name: value[field_name] for field_name in cls._FIELDS})


@dataclass(frozen=True)
class RunManifest:
    """Immutable definition of a local training run and its claim boundary."""

    run_id: str
    experiment_id: str
    parent_run_id: str | None
    status: str
    created_at_utc: str
    git_commit: str
    git_dirty: bool
    git_diff_sha256: str | None
    command: tuple[str, ...]
    resolved_config: Mapping[str, Any]
    config_sha256: str
    environment: Mapping[str, Any]
    seeds: Mapping[str, int]
    assets: tuple[AssetManifestRef, ...]
    components: Mapping[str, bool]
    outputs: Mapping[str, str]
    checkpoint_policy: Mapping[str, Any]
    claim_boundaries: tuple[str, ...]

    schema_version: ClassVar[str] = RUN_MANIFEST_SCHEMA
    _FIELDS: ClassVar[set[str]] = {
        "schema_version",
        "run_id",
        "experiment_id",
        "parent_run_id",
        "status",
        "created_at_utc",
        "git_commit",
        "git_dirty",
        "git_diff_sha256",
        "command",
        "resolved_config",
        "config_sha256",
        "environment",
        "seeds",
        "assets",
        "components",
        "outputs",
        "checkpoint_policy",
        "claim_boundaries",
    }

    def __post_init__(self) -> None:
        _require_nonempty_text(self.run_id, field="run_id")
        _require_nonempty_text(self.experiment_id, field="experiment_id")
        _require_optional_text(self.parent_run_id, field="parent_run_id")
        if self.status not in RUN_STATUSES:
            raise ContractValidationError(f"unsupported run status: {self.status!r}")
        _require_utc_timestamp(self.created_at_utc, field="created_at_utc")
        _require_nonempty_text(self.git_commit, field="git_commit")
        if not isinstance(self.git_dirty, bool):
            raise ContractValidationError("git_dirty must be a boolean")
        if self.git_dirty:
            _require_sha256(self.git_diff_sha256, field="git_diff_sha256")
        elif self.git_diff_sha256 is not None:
            raise ContractValidationError(
                "git_diff_sha256 must be null when git_dirty is false"
            )

        command = tuple(_require_json_array(self.command, field="command"))
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ContractValidationError("command must contain non-empty strings")
        object.__setattr__(self, "command", command)

        resolved_config = _freeze_json(
            _require_mapping(self.resolved_config, field="resolved_config"),
            field="resolved_config",
        )
        object.__setattr__(self, "resolved_config", resolved_config)
        config_sha256 = _require_sha256(self.config_sha256, field="config_sha256")
        if config_sha256 != canonical_sha256(resolved_config):
            raise ContractValidationError(
                "config_sha256 does not match the canonical resolved_config"
            )
        object.__setattr__(self, "config_sha256", config_sha256)

        object.__setattr__(
            self,
            "environment",
            _freeze_json(
                _require_mapping(self.environment, field="environment"),
                field="environment",
            ),
        )
        object.__setattr__(
            self,
            "checkpoint_policy",
            _freeze_json(
                _require_mapping(
                    self.checkpoint_policy, field="checkpoint_policy"
                ),
                field="checkpoint_policy",
            ),
        )

        seeds = dict(_require_mapping(self.seeds, field="seeds"))
        if any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            for key, value in seeds.items()
        ):
            raise ContractValidationError("seeds must map non-empty names to integers")
        object.__setattr__(self, "seeds", MappingProxyType(seeds))

        assets = tuple(_require_json_array(self.assets, field="assets"))
        if any(not isinstance(asset, AssetManifestRef) for asset in assets):
            raise ContractValidationError("assets must contain AssetManifestRef values")
        asset_names = [asset.logical_name for asset in assets]
        if len(asset_names) != len(set(asset_names)):
            raise ContractValidationError("asset logical names must be unique")
        object.__setattr__(self, "assets", assets)

        components = dict(_require_mapping(self.components, field="components"))
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, bool)
            for key, value in components.items()
        ):
            raise ContractValidationError(
                "components must map non-empty names to booleans"
            )
        object.__setattr__(self, "components", MappingProxyType(components))

        outputs = dict(_require_mapping(self.outputs, field="outputs"))
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in outputs.items()
        ):
            raise ContractValidationError(
                "outputs must map non-empty names to non-empty strings"
            )
        object.__setattr__(self, "outputs", MappingProxyType(outputs))

        claim_boundaries = tuple(
            _require_json_array(self.claim_boundaries, field="claim_boundaries")
        )
        if not claim_boundaries or any(
            not isinstance(item, str) or not item.strip()
            for item in claim_boundaries
        ):
            raise ContractValidationError(
                "claim_boundaries must contain non-empty strings"
            )
        object.__setattr__(self, "claim_boundaries", claim_boundaries)

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "parent_run_id": self.parent_run_id,
            "status": self.status,
            "created_at_utc": self.created_at_utc,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "git_diff_sha256": self.git_diff_sha256,
            "command": list(self.command),
            "resolved_config": _thaw_json(self.resolved_config),
            "config_sha256": self.config_sha256,
            "environment": _thaw_json(self.environment),
            "seeds": dict(self.seeds),
            "assets": [asset.to_dict() for asset in self.assets],
            "components": dict(self.components),
            "outputs": dict(self.outputs),
            "checkpoint_policy": _thaw_json(self.checkpoint_policy),
            "claim_boundaries": list(self.claim_boundaries),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunManifest:
        if not isinstance(value, Mapping):
            raise ContractValidationError("run manifest must be a JSON object")
        _require_exact_keys(value, cls._FIELDS, context="run manifest")
        if value["schema_version"] != cls.schema_version:
            raise ContractValidationError(
                f"unsupported run manifest schema: {value['schema_version']!r}"
            )
        command_value = _require_json_array(value["command"], field="command")
        assets_value = _require_json_array(value["assets"], field="assets")
        claim_boundaries_value = _require_json_array(
            value["claim_boundaries"], field="claim_boundaries"
        )
        return cls(
            run_id=value["run_id"],
            experiment_id=value["experiment_id"],
            parent_run_id=value["parent_run_id"],
            status=value["status"],
            created_at_utc=value["created_at_utc"],
            git_commit=value["git_commit"],
            git_dirty=value["git_dirty"],
            git_diff_sha256=value["git_diff_sha256"],
            command=tuple(command_value),
            resolved_config=value["resolved_config"],
            config_sha256=value["config_sha256"],
            environment=value["environment"],
            seeds=value["seeds"],
            assets=tuple(AssetManifestRef.from_dict(item) for item in assets_value),
            components=value["components"],
            outputs=value["outputs"],
            checkpoint_policy=value["checkpoint_policy"],
            claim_boundaries=tuple(claim_boundaries_value),
        )


@dataclass(frozen=True)
class RunEvent:
    """One append-only lifecycle event linked to the preceding event hash."""

    run_id: str
    sequence: int
    timestamp_utc: str
    status: str
    event_type: str
    reason_code: str | None
    details: Mapping[str, Any]
    previous_event_sha256: str | None

    schema_version: ClassVar[str] = RUN_EVENT_SCHEMA
    _FIELDS: ClassVar[set[str]] = {
        "schema_version",
        "run_id",
        "sequence",
        "timestamp_utc",
        "status",
        "event_type",
        "reason_code",
        "details",
        "previous_event_sha256",
    }

    def __post_init__(self) -> None:
        _require_nonempty_text(self.run_id, field="run_id")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ContractValidationError("sequence must be an integer")
        if self.sequence < 0:
            raise ContractValidationError("sequence must be non-negative")
        _require_utc_timestamp(self.timestamp_utc, field="timestamp_utc")
        if self.status not in RUN_STATUSES:
            raise ContractValidationError(f"unsupported run status: {self.status!r}")
        _require_nonempty_text(self.event_type, field="event_type")
        _require_optional_text(self.reason_code, field="reason_code")
        object.__setattr__(
            self,
            "details",
            _freeze_json(
                _require_mapping(self.details, field="event details"),
                field="event details",
            ),
        )
        if self.previous_event_sha256 is not None:
            object.__setattr__(
                self,
                "previous_event_sha256",
                _require_sha256(
                    self.previous_event_sha256, field="previous_event_sha256"
                ),
            )

    @property
    def event_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "timestamp_utc": self.timestamp_utc,
            "status": self.status,
            "event_type": self.event_type,
            "reason_code": self.reason_code,
            "details": _thaw_json(self.details),
            "previous_event_sha256": self.previous_event_sha256,
        }

    def to_record(self) -> dict[str, Any]:
        return {"event": self.to_dict(), "event_sha256": self.event_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunEvent:
        if not isinstance(value, Mapping):
            raise ContractValidationError("run event must be a JSON object")
        _require_exact_keys(value, cls._FIELDS, context="run event")
        if value["schema_version"] != cls.schema_version:
            raise ContractValidationError(
                f"unsupported run event schema: {value['schema_version']!r}"
            )
        return cls(
            run_id=value["run_id"],
            sequence=value["sequence"],
            timestamp_utc=value["timestamp_utc"],
            status=value["status"],
            event_type=value["event_type"],
            reason_code=value["reason_code"],
            details=value["details"],
            previous_event_sha256=value["previous_event_sha256"],
        )


def publish_run_manifest(path: str | Path, manifest: RunManifest) -> None:
    """Publish a new manifest atomically without overwriting existing evidence."""
    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be a RunManifest")
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"run manifest already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    payload = canonical_json_bytes(manifest.to_dict()) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise FileExistsError(f"run manifest already exists: {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_run_manifest(path: str | Path) -> RunManifest:
    """Load and strictly validate a persisted run manifest."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"cannot read run manifest: {path}") from exc
    return RunManifest.from_dict(value)


def _decode_event_record(value: Any, *, line_number: int) -> RunEvent:
    if not isinstance(value, Mapping):
        raise ContractValidationError(
            f"event ledger line {line_number} must contain a JSON object"
        )
    _require_exact_keys(
        value, {"event", "event_sha256"}, context=f"event ledger line {line_number}"
    )
    event = RunEvent.from_dict(value["event"])
    recorded_hash = _require_sha256(
        value["event_sha256"], field=f"event ledger line {line_number} hash"
    )
    if recorded_hash != event.event_sha256:
        raise ContractValidationError(
            f"event ledger line {line_number} has an invalid event hash"
        )
    return event


def load_run_events(path: str | Path) -> tuple[RunEvent, ...]:
    """Load an event ledger and verify sequence, run identity, and hash chain."""
    ledger = Path(path)
    if not ledger.exists():
        return ()
    events: list[RunEvent] = []
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractValidationError(f"cannot read event ledger: {ledger}") from exc
    if any(not line.strip() for line in lines):
        raise ContractValidationError("event ledger contains a blank line")
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractValidationError(
                f"event ledger line {line_number} is not valid JSON"
            ) from exc
        event = _decode_event_record(value, line_number=line_number)
        expected_sequence = len(events)
        if event.sequence != expected_sequence:
            raise ContractValidationError(
                f"event ledger line {line_number} has sequence {event.sequence}; "
                f"expected {expected_sequence}"
            )
        if events:
            previous = events[-1]
            if event.run_id != previous.run_id:
                raise ContractValidationError("event ledger mixes run identities")
            if event.previous_event_sha256 != previous.event_sha256:
                raise ContractValidationError(
                    f"event ledger line {line_number} breaks the hash chain"
                )
        elif event.previous_event_sha256 is not None:
            raise ContractValidationError(
                "the first event must not declare a previous event hash"
            )
        events.append(event)
    return tuple(events)


def append_run_event(path: str | Path, event: RunEvent) -> None:
    """Append one validated event to a single-writer local event ledger."""
    if not isinstance(event, RunEvent):
        raise TypeError("event must be a RunEvent")
    target = Path(path)
    existing = load_run_events(target)
    if existing:
        previous = existing[-1]
        if event.run_id != previous.run_id:
            raise ContractValidationError("event ledger cannot mix run identities")
        if event.sequence != previous.sequence + 1:
            raise ContractValidationError("event sequence is not contiguous")
        if event.previous_event_sha256 != previous.event_sha256:
            raise ContractValidationError("event does not extend the ledger hash chain")
    else:
        if event.sequence != 0:
            raise ContractValidationError("the first event sequence must be zero")
        if event.previous_event_sha256 is not None:
            raise ContractValidationError(
                "the first event must not declare a previous event hash"
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(event.to_record()) + b"\n"
    with target.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
