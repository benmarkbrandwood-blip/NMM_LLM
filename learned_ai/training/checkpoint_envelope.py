"""Versioned, integrity-checked checkpoint envelopes for exact local resume."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import shutil
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar
from uuid import uuid4

import numpy as np
import torch

from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256


CHECKPOINT_SCHEMA = "nmm.checkpoint-envelope.v2"
CHECKPOINT_ROLES = frozenset({"latest", "best_train", "candidate", "accepted"})
CHECKPOINT_START_MODES = frozenset({"fresh", "weights-only", "exact-resume"})

_MAGIC = b"NMMCKP2\n"
_PREFIX = struct.Struct(">8sQ")
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_PAYLOAD_FIELDS = {
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "scaler_state",
    "rng_state",
    "trainer_state",
    "data_state",
}
_TRAINER_STATE_FIELDS = {
    "game_count",
    "batch_count",
    "update_count",
    "difficulty",
    "temperature",
    "rolling_metrics",
    "curriculum",
    "target_network",
    "recovery_state",
    "model_config",
}
_DATA_STATE_FIELDS = {
    "cursor",
    "consumed_snapshots",
    "cache",
    "buckets",
    "mutable_assets",
}
_RNG_STATE_FIELDS = {"python", "numpy", "torch_cpu", "torch_cuda", "components"}


class CheckpointError(RuntimeError):
    """Base class for checkpoint format, integrity, and compatibility failures."""


class CheckpointFormatError(CheckpointError):
    """Raised when a checkpoint does not conform to the v2 envelope format."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when checkpoint bytes do not match their declared identity."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a valid checkpoint is incompatible with the requested run."""


def _nonempty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CheckpointFormatError(f"{field} must be a non-empty string")
    return value


def _optional_text(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _nonempty_text(value, field=field)


def _sha256_text(value: Any, *, field: str) -> str:
    text = _nonempty_text(value, field=field).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise CheckpointFormatError(f"{field} must be a 64-character SHA-256")
    return text


def _utc_timestamp(value: Any, *, field: str) -> str:
    text = _nonempty_text(value, field=field)
    if not text.endswith("Z"):
        raise CheckpointFormatError(f"{field} must be an RFC 3339 UTC timestamp")
    try:
        datetime.fromisoformat(f"{text[:-1]}+00:00")
    except ValueError as exc:
        raise CheckpointFormatError(
            f"{field} must be an RFC 3339 UTC timestamp"
        ) from exc
    return text


def _string_mapping(value: Any, *, field: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise CheckpointFormatError(f"{field} must be a mapping")
    copied = dict(value)
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(item, str)
        or not item
        for key, item in copied.items()
    ):
        raise CheckpointFormatError(
            f"{field} must map non-empty strings to non-empty strings"
        )
    return MappingProxyType(copied)


@dataclass(frozen=True)
class CheckpointDescriptor:
    """JSON metadata that identifies one checkpoint and its compatibility."""

    checkpoint_id: str
    run_id: str
    experiment_id: str
    parent_checkpoint_id: str | None
    role: str
    save_reason: str
    created_at_utc: str
    config_sha256: str
    feature_schema_version: str
    label_schema_version: str
    database_schema_versions: Mapping[str, str]
    asset_identities: Mapping[str, str]
    implementation: Mapping[str, str]

    _FIELDS: ClassVar[set[str]] = {
        "checkpoint_id",
        "run_id",
        "experiment_id",
        "parent_checkpoint_id",
        "role",
        "save_reason",
        "created_at_utc",
        "config_sha256",
        "feature_schema_version",
        "label_schema_version",
        "database_schema_versions",
        "asset_identities",
        "implementation",
    }

    def __post_init__(self) -> None:
        _nonempty_text(self.checkpoint_id, field="checkpoint_id")
        _nonempty_text(self.run_id, field="run_id")
        _nonempty_text(self.experiment_id, field="experiment_id")
        _optional_text(self.parent_checkpoint_id, field="parent_checkpoint_id")
        if self.role not in CHECKPOINT_ROLES:
            raise CheckpointFormatError(f"unsupported checkpoint role: {self.role!r}")
        _nonempty_text(self.save_reason, field="save_reason")
        _utc_timestamp(self.created_at_utc, field="created_at_utc")
        object.__setattr__(
            self,
            "config_sha256",
            _sha256_text(self.config_sha256, field="config_sha256"),
        )
        _nonempty_text(self.feature_schema_version, field="feature_schema_version")
        _nonempty_text(self.label_schema_version, field="label_schema_version")
        object.__setattr__(
            self,
            "database_schema_versions",
            _string_mapping(
                self.database_schema_versions, field="database_schema_versions"
            ),
        )
        object.__setattr__(
            self,
            "asset_identities",
            _string_mapping(self.asset_identities, field="asset_identities"),
        )
        object.__setattr__(
            self,
            "implementation",
            _string_mapping(self.implementation, field="implementation"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "role": self.role,
            "save_reason": self.save_reason,
            "created_at_utc": self.created_at_utc,
            "config_sha256": self.config_sha256,
            "feature_schema_version": self.feature_schema_version,
            "label_schema_version": self.label_schema_version,
            "database_schema_versions": dict(self.database_schema_versions),
            "asset_identities": dict(self.asset_identities),
            "implementation": dict(self.implementation),
        }

    @classmethod
    def from_dict(cls, value: Any) -> CheckpointDescriptor:
        if not isinstance(value, Mapping):
            raise CheckpointFormatError("checkpoint descriptor must be a mapping")
        actual = set(value)
        if actual != cls._FIELDS:
            unknown = sorted(actual - cls._FIELDS)
            missing = sorted(cls._FIELDS - actual)
            raise CheckpointFormatError(
                f"checkpoint descriptor keys differ; unknown={unknown}, missing={missing}"
            )
        return cls(**{field: value[field] for field in cls._FIELDS})


@dataclass(frozen=True)
class CheckpointPayload:
    """Complete state required to continue a trainer without implicit resets."""

    model_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any] | None
    scheduler_state: Mapping[str, Any] | None
    scaler_state: Mapping[str, Any] | None
    rng_state: Mapping[str, Any]
    trainer_state: Mapping[str, Any]
    data_state: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field in ("model_state", "rng_state", "trainer_state", "data_state"):
            if not isinstance(getattr(self, field), Mapping):
                raise CheckpointFormatError(f"{field} must be a mapping")
        for field in ("optimizer_state", "scheduler_state", "scaler_state"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, Mapping):
                raise CheckpointFormatError(f"{field} must be a mapping or null")
        for field, expected in (
            ("rng_state", _RNG_STATE_FIELDS),
            ("trainer_state", _TRAINER_STATE_FIELDS),
            ("data_state", _DATA_STATE_FIELDS),
        ):
            actual = set(getattr(self, field))
            if actual != expected:
                raise CheckpointFormatError(
                    f"{field} keys differ; unknown={sorted(actual - expected)}, "
                    f"missing={sorted(expected - actual)}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "scaler_state": self.scaler_state,
            "rng_state": self.rng_state,
            "trainer_state": self.trainer_state,
            "data_state": self.data_state,
        }

    @classmethod
    def from_dict(cls, value: Any) -> CheckpointPayload:
        if not isinstance(value, Mapping):
            raise CheckpointFormatError("checkpoint payload must be a mapping")
        actual = set(value)
        if actual != _PAYLOAD_FIELDS:
            unknown = sorted(actual - _PAYLOAD_FIELDS)
            missing = sorted(_PAYLOAD_FIELDS - actual)
            raise CheckpointFormatError(
                f"checkpoint payload keys differ; unknown={unknown}, missing={missing}"
            )
        return cls(**{field: value[field] for field in _PAYLOAD_FIELDS})


@dataclass(frozen=True)
class CheckpointEnvelope:
    """A verified descriptor, payload identity, and deserialized state."""

    descriptor: CheckpointDescriptor
    payload_sha256: str
    payload_size: int
    payload: CheckpointPayload


def capture_rng_state(
    component_states: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture Python, NumPy, PyTorch CPU, and all available CUDA RNG states."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "components": dict(component_states or {}),
    }


def restore_rng_state(
    state: Mapping[str, Any],
    *,
    component_rngs: Mapping[str, Any] | None = None,
) -> None:
    """Restore every captured RNG state or fail rather than partially resume."""
    if not isinstance(state, Mapping) or set(state) != _RNG_STATE_FIELDS:
        raise CheckpointFormatError("RNG state is incomplete or has unknown fields")
    cuda_states = state["torch_cuda"]
    if cuda_states and not torch.cuda.is_available():
        raise CheckpointCompatibilityError(
            "checkpoint contains CUDA RNG state but CUDA is unavailable"
        )
    expected_components = state["components"]
    if not isinstance(expected_components, Mapping):
        raise CheckpointFormatError("checkpoint component RNG states must be a mapping")
    provided_components = dict(component_rngs or {})
    if set(expected_components) != set(provided_components):
        raise CheckpointCompatibilityError(
            "checkpoint component RNG identities do not match the trainer"
        )
    for name, component_state in expected_components.items():
        generator = provided_components[name]
        if not hasattr(generator, "setstate"):
            raise CheckpointCompatibilityError(
                f"component RNG {name!r} does not support setstate"
            )
    random.setstate(state["python"])
    np.random.set_state(tuple(state["numpy"]))
    torch_cpu_state = state["torch_cpu"]
    if not isinstance(torch_cpu_state, torch.Tensor):
        raise CheckpointFormatError("checkpoint CPU RNG state must be a tensor")
    torch.set_rng_state(torch_cpu_state.detach().cpu())
    if cuda_states:
        torch.cuda.set_rng_state_all(
            [item.detach().cpu() for item in cuda_states]
        )
    for name, component_state in expected_components.items():
        generator = provided_components[name]
        generator.setstate(component_state)


def _hash_stream(handle: Any, *, length: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = length
    while remaining is None or remaining > 0:
        read_size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
        chunk = handle.read(read_size)
        if not chunk:
            break
        digest.update(chunk)
        if remaining is not None:
            remaining -= len(chunk)
    if remaining not in (None, 0):
        raise CheckpointIntegrityError("checkpoint payload is truncated")
    return digest.hexdigest()


def _read_header(handle: Any) -> tuple[CheckpointDescriptor, str, int]:
    prefix = handle.read(_PREFIX.size)
    if len(prefix) != _PREFIX.size:
        raise CheckpointFormatError("checkpoint prefix is truncated")
    magic, header_size = _PREFIX.unpack(prefix)
    if magic != _MAGIC:
        raise CheckpointFormatError("checkpoint magic does not identify envelope v2")
    if header_size <= 0 or header_size > _MAX_HEADER_BYTES:
        raise CheckpointFormatError("checkpoint header size is invalid")
    header_bytes = handle.read(header_size)
    if len(header_bytes) != header_size:
        raise CheckpointFormatError("checkpoint header is truncated")
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointFormatError("checkpoint header is not valid JSON") from exc
    expected = {
        "schema_version",
        "descriptor",
        "descriptor_sha256",
        "payload_sha256",
        "payload_size",
    }
    if not isinstance(header, Mapping) or set(header) != expected:
        raise CheckpointFormatError("checkpoint header fields are invalid")
    if header["schema_version"] != CHECKPOINT_SCHEMA:
        raise CheckpointFormatError(
            f"unsupported checkpoint schema: {header['schema_version']!r}"
        )
    descriptor_hash = _sha256_text(
        header["descriptor_sha256"], field="descriptor_sha256"
    )
    if descriptor_hash != canonical_sha256(header["descriptor"]):
        raise CheckpointIntegrityError(
            "checkpoint descriptor SHA-256 does not match the header"
        )
    payload_hash = _sha256_text(header["payload_sha256"], field="payload_sha256")
    payload_size = header["payload_size"]
    if isinstance(payload_size, bool) or not isinstance(payload_size, int) or payload_size <= 0:
        raise CheckpointFormatError("payload_size must be a positive integer")
    return CheckpointDescriptor.from_dict(header["descriptor"]), payload_hash, payload_size


def inspect_checkpoint(
    path: str | Path, *, verify_payload: bool = True
) -> tuple[CheckpointDescriptor, str, int]:
    """Inspect metadata and optionally verify payload bytes without deserializing."""
    checkpoint = Path(path)
    try:
        with checkpoint.open("rb") as handle:
            descriptor, expected_hash, payload_size = _read_header(handle)
            payload_offset = handle.tell()
            actual_size = checkpoint.stat().st_size - payload_offset
            if actual_size != payload_size:
                raise CheckpointIntegrityError(
                    f"payload size is {actual_size}; expected {payload_size}"
                )
            if verify_payload:
                actual_hash = _hash_stream(handle, length=payload_size)
                if actual_hash != expected_hash:
                    raise CheckpointIntegrityError(
                        "checkpoint payload SHA-256 does not match the header"
                    )
            return descriptor, expected_hash, payload_size
    except OSError as exc:
        raise CheckpointError(f"cannot inspect checkpoint: {checkpoint}") from exc


def is_checkpoint_envelope(path: str | Path) -> bool:
    """Return whether a file declares the v2 magic without validating its content."""
    try:
        with Path(path).open("rb") as handle:
            return handle.read(len(_MAGIC)) == _MAGIC
    except OSError:
        return False


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> CheckpointEnvelope:
    """Verify and load exactly the requested checkpoint without fallback."""
    checkpoint = Path(path)
    try:
        with checkpoint.open("rb") as handle:
            descriptor, expected_hash, payload_size = _read_header(handle)
            payload_bytes = handle.read(payload_size)
            if len(payload_bytes) != payload_size or handle.read(1):
                raise CheckpointIntegrityError(
                    "checkpoint payload size does not match the header"
                )
            if hashlib.sha256(payload_bytes).hexdigest() != expected_hash:
                raise CheckpointIntegrityError(
                    "checkpoint payload SHA-256 does not match the header"
                )
        raw_payload = torch.load(
            io.BytesIO(payload_bytes), map_location=map_location, weights_only=False
        )
    except CheckpointError:
        raise
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        raise CheckpointFormatError(
            f"cannot deserialize checkpoint payload: {checkpoint}"
        ) from exc
    payload = CheckpointPayload.from_dict(raw_payload)
    return CheckpointEnvelope(descriptor, expected_hash, payload_size, payload)


def verify_checkpoint_compatibility(
    envelope: CheckpointEnvelope,
    *,
    config_sha256: str,
    feature_schema_version: str,
    label_schema_version: str,
    asset_identities: Mapping[str, str],
    run_id: str | None = None,
) -> None:
    """Require exact semantic compatibility for an exact-resume request."""
    descriptor = envelope.descriptor
    mismatches: list[str] = []
    if descriptor.config_sha256 != config_sha256:
        mismatches.append("config_sha256")
    if descriptor.feature_schema_version != feature_schema_version:
        mismatches.append("feature_schema_version")
    if descriptor.label_schema_version != label_schema_version:
        mismatches.append("label_schema_version")
    if dict(descriptor.asset_identities) != dict(asset_identities):
        mismatches.append("asset_identities")
    if run_id is not None and descriptor.run_id != run_id:
        mismatches.append("run_id")
    if mismatches:
        raise CheckpointCompatibilityError(
            "checkpoint is incompatible: " + ", ".join(mismatches)
        )


def _backup_path(path: Path, index: int) -> Path:
    return path.with_name(f"{path.stem}.prev{index}{path.suffix}")


def _copy_fsync(source: Path, destination: Path) -> None:
    with source.open("rb") as source_handle, destination.open("xb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
        target_handle.flush()
        os.fsync(target_handle.fileno())


def _rotate_verified_target(path: Path, previous_copies: int) -> None:
    if previous_copies <= 0 or not path.exists():
        return
    # Verify before preserving the current target as a known-good previous copy.
    inspect_checkpoint(path)
    temporary = path.with_name(f".{path.name}.previous.{uuid4().hex}.tmp")
    try:
        _copy_fsync(path, temporary)
        for index in range(previous_copies - 1, 0, -1):
            source = _backup_path(path, index - 1)
            if source.exists():
                os.replace(source, _backup_path(path, index))
        os.replace(temporary, _backup_path(path, 0))
    finally:
        if temporary.exists():
            temporary.unlink()


def save_checkpoint(
    path: str | Path,
    descriptor: CheckpointDescriptor,
    payload: CheckpointPayload,
    *,
    previous_copies: int = 4,
) -> None:
    """Verify, rotate known-good copies, and atomically publish one checkpoint."""
    if isinstance(previous_copies, bool) or not isinstance(previous_copies, int):
        raise ValueError("previous_copies must be a non-negative integer")
    if previous_copies < 0:
        raise ValueError("previous_copies must be a non-negative integer")
    target = Path(path)
    if descriptor.role == "accepted" and target.exists():
        raise FileExistsError(f"accepted checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    payload_temp = target.with_name(f".{target.name}.{token}.payload.tmp")
    envelope_temp = target.with_name(f".{target.name}.{token}.tmp")
    try:
        torch.save(payload.to_dict(), payload_temp)
        payload_size = payload_temp.stat().st_size
        with payload_temp.open("rb") as handle:
            payload_hash = _hash_stream(handle)
        header = {
            "schema_version": CHECKPOINT_SCHEMA,
            "descriptor": descriptor.to_dict(),
            "descriptor_sha256": canonical_sha256(descriptor.to_dict()),
            "payload_sha256": payload_hash,
            "payload_size": payload_size,
        }
        header_bytes = canonical_json_bytes(header)
        with envelope_temp.open("xb") as output, payload_temp.open("rb") as source:
            output.write(_PREFIX.pack(_MAGIC, len(header_bytes)))
            output.write(header_bytes)
            shutil.copyfileobj(source, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        loaded = load_checkpoint(envelope_temp)
        if loaded.descriptor != descriptor or loaded.payload_sha256 != payload_hash:
            raise CheckpointIntegrityError("checkpoint verification changed metadata")
        _rotate_verified_target(target, previous_copies)
        os.replace(envelope_temp, target)
    finally:
        for temporary in (payload_temp, envelope_temp):
            if temporary.exists():
                temporary.unlink()
