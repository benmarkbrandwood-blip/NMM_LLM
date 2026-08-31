"""Durable SQLite foundation for classical ``A_pos`` state generation.

The public production surface is intentionally non-launchable in C5a.  Runtime
preflight and runtime-bound capability types exist, but there is no production
runtime-plan issuer, inventory/referee issuer, runtime-preflight issuer,
runtime binder, runtime-completion issuer, or executor.  Only the separate
internal-test lane can exercise the bind core.  The implementation freezes the
single-file SQLite contract.  SQLite ``FULL`` synchronous durability is the
declared process boundary on the pinned Windows host; this module does not
claim directory fsync, resistance to an administrator, or protection from
hardware failure.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import stat
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from game.board import BoardState
from game.rules import get_all_legal_moves, get_game_phase, terminal_result
from learned_ai.evaluation.sanmill_uci import (
    EXPECTED_RULES_IDENTITY_SHA256,
    SanmillBridgeError,
    UciPositionState,
    project_stable_sanmill_fen,
)
from learned_ai.training import classical_a_pos_governance as _governance
from learned_ai.training import classical_a_pos_corpus as _corpus
from learned_ai.training.classical_a_pos_corpus import (
    FROZEN_CORPUS_LAYOUT,
    SINGLETON_LEDGER_SCHEMA,
    STATE_RECORD_SCHEMA,
    state_split_artifact_identity,
)
from learned_ai.training.classical_a_pos_governance import (
    ConsumedAuthorizationPermit,
    GovernanceReplay,
    PendingAuthorizationConsumption,
    PendingOperationReservation,
    ProductionAuthorizationPermit,
    ProductionOperationPermit,
    ProductionRuntimePlanPermit,
    RuntimePlanRecord,
    SingleUseAuthorizationRecord,
    build_state_frozen_event,
    build_state_generation_completed_event,
    confirm_authorization_consumption,
    confirm_operation_reservation,
    decode_governance_ledger,
    encode_governance_ledger,
    prepared_governance_event_bytes,
    replay_governance_ledger,
    require_production_operation_permit,
    verify_runtime_plan,
    verify_single_use_authorization,
)
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.training.sanmill_referee import (
    TRAINING_REFEREE_FORMAT,
    TRAINING_REFEREE_PROFILE,
    TRAINING_REFEREE_SEMANTIC_DIGEST,
    TRAINING_REPETITION_OBSERVATION,
    nmm_move_actions,
)

__all__ = (
    "GovernanceStoreContractError",
    "GovernanceStoreSpec",
    "DurableGovernanceStore",
    "ProductionAPosInventoryBinding",
    "ProductionStrictRefereeBinding",
    "ProductionStateGenerationRuntimePreflight",
    "ActiveStateGenerationAttempt",
    "RuntimeBoundStateGenerationAttempt",
    "PendingStateGenerationCommit",
    "ConfirmedStateGenerationCompletion",
    "ProductionStateGenerationRuntimeCompletion",
    "DurableStateFreezeBinding",
    "build_governance_store_spec",
    "initialize_governance_store",
    "open_governance_store",
    "commit_authorization_consumption",
    "commit_operation_reservation",
    "begin_state_generation_attempt",
    "prepare_state_generation_commit",
    "commit_state_generation",
    "commit_state_freeze",
    "restore_durable_state_freeze",
    "verify_durable_state_freeze",
)

_SPEC_SCHEMA = "nmm.classical-a-pos-governance-store-spec.v2"
_STORE_META_SCHEMA = "nmm.classical-a-pos-governance-store-meta.v2"
_SOURCE_GAME_SCHEMA = "nmm.classical-a-pos-source-game.v2"
_COMPLETION_SCHEMA = "nmm.classical-a-pos-state-generation-completion.v2"
_FREEZE_RECEIPT_SCHEMA = "nmm.classical-a-pos-durable-state-freeze.v2"
_CONTROLLER_STREAM_KIND = "controller"
_RUNTIME_STREAM_KIND = "state-generation-runtime"
_RUNTIME_STREAM_SCHEMA = "nmm.classical-a-pos-state-generation-runtime-stream.v1"
_RESOURCE_SNAPSHOT_SCHEMA = "nmm.classical-a-pos-state-generation-resource-snapshot.v1"
_RUNTIME_RECORD_TYPES = (
    "state-generation-resource-before",
    "state-generation-resource-after",
    "state-generation-resource-stability",
)
_RUNTIME_RECORD_SCHEMAS = {
    "state-generation-resource-before": (
        "nmm.classical-a-pos-state-generation-resource-before.v1"
    ),
    "state-generation-resource-after": (
        "nmm.classical-a-pos-state-generation-resource-after.v1"
    ),
    "state-generation-resource-stability": (
        "nmm.classical-a-pos-state-generation-resource-stability.v1"
    ),
}
_REPETITION_DRAW_OUTCOME = (
    "drawThreefoldRepetition",
    "draw_threefold_repetition",
)
_RULES_DRAW_OUTCOMES = frozenset(
    {
        ("drawFiftyMoveLegacy", "draw_fifty_move_legacy"),
        ("drawFullBoard", "draw_full_board"),
        ("drawStalemateCondition", "draw_stalemate_condition"),
        ("drawFiftyMove", "draw_fifty_move"),
        ("drawEndgameFiftyMove", "draw_endgame_fifty_move"),
    }
)
_RULES_WIN_OUTCOME_BY_LOCAL_REASON = {
    "fewer-than-three": ("loseFewerThanThree", "lose_fewer_than_three"),
    "no-legal-move": ("loseNoLegalMoves", "lose_no_legal_moves"),
}
_EXPERIMENT_ID = "classical-a-pos-offline-distillation-v1"
_PROPOSAL_IDENTITY = "edf3e1031ee4bd46b6c567891fcaab26fd288a49997410141e6e19745d070e5c"
_PROFILE_IDENTITY = "bfa8d2f8e19b1c24641e24e4765844f678cb9288838e5eda3f8782d7ace9cbe0"
_DATABASE_ROLE = "classical-a-pos-controller-governance"
_RELATIVE_DATABASE_PATH = "governance/classical-a-pos-controller.sqlite3"
_HEX = frozenset("0123456789abcdef")
_SIDECAR_SUFFIXES = ("-journal", "-wal", "-shm")

_SPEC_KEYS = {
    "schema_version",
    "experiment_id",
    "proposal_identity",
    "profile_identity",
    "plan_identity",
    "readiness_identity",
    "managed_git_state_identity",
    "launch_path_binding_identity",
    "state_freeze_teacher_order_event_chain_identity",
    "database_role",
    "output_root_identity",
    "normalized_relative_path",
    "schema_ddl_identity",
    "spec_identity",
}
_DOMAIN_ENVELOPE_KEYS = {
    "schema_version",
    "stream_identity",
    "stream_kind",
    "sequence",
    "record_type",
    "previous_record_identity",
}
_COMPLETION_GOVERNANCE_KEYS = {
    "experiment_id",
    "proposal_identity",
    "profile_identity",
    "store_spec_identity",
    "plan_identity",
    "readiness_identity",
    "managed_git_state_identity",
    "launch_path_binding_identity",
    "authorization_identity",
    "authorization_consumption_identity",
    "attempt_identity",
    "reservation_event_identity",
    "completion_event_identity",
}
_COMPLETION_RUNTIME_KEYS = {
    "host_preflight_identity",
    "state_generator_session_identity",
    "strict_referee_binding_identity",
    "a_pos_inventory_binding_identity",
    "runtime_evidence_stream_identity",
    "resource_snapshot_before_identity",
    "resource_snapshot_after_identity",
    "resource_stability_identity",
}
_COMPLETION_RESULT_KEYS = {
    "artifacts",
    "split_contract_identity",
    "resource_observation",
    "teacher_fields_present",
    "authoritative_storage",
}
_FREEZE_GOVERNANCE_KEYS = {
    "experiment_id",
    "proposal_identity",
    "profile_identity",
    "store_spec_identity",
    "plan_identity",
    "readiness_identity",
    "authorization_identity",
    "authorization_consumption_identity",
    "attempt_identity",
    "completion_event_identity",
    "completion_record_identity",
    "source_games_identity",
    "state_split_identity",
    "singleton_ledger_identity",
    "runtime_evidence_stream_identity",
    "resource_stability_identity",
}
_FREEZE_RESULT_KEYS = {
    "teacher_fields_present",
    "teacher_may_start_only_after_this_event",
    "authoritative_artifacts",
}
_COMPLETION_KEYS = (
    _DOMAIN_ENVELOPE_KEYS
    | _COMPLETION_GOVERNANCE_KEYS
    | _COMPLETION_RUNTIME_KEYS
    | _COMPLETION_RESULT_KEYS
)
_FREEZE_RECEIPT_KEYS = (
    _DOMAIN_ENVELOPE_KEYS | _FREEZE_GOVERNANCE_KEYS | _FREEZE_RESULT_KEYS
)

_RESOURCE_SNAPSHOT_KEYS = {
    "schema_version",
    "stage",
    "experiment_id",
    "proposal_identity",
    "profile_identity",
    "store_spec_identity",
    "plan_identity",
    "readiness_identity",
    "attempt_identity",
    "observed_at_utc",
    "host",
    "path_registry",
    "repository",
    "output_root",
    "malom",
    "sanmill",
    "generator",
    "snapshot_identity",
}
_HOST_SNAPSHOT_KEYS = {
    "schema_version",
    "platform",
    "machine_identity",
    "python_executable_sha256",
    "python_version",
    "torch_version",
    "device",
    "cuda_initialized",
    "available_memory_bytes",
}
_PATH_REGISTRY_SNAPSHOT_KEYS = {
    "schema_version",
    "registry_role",
    "file_sha256",
    "canonical_object_identity",
    "required_lookup_keys",
    "resolved_path_identities",
}
_REPOSITORY_SNAPSHOT_KEYS = {
    "schema_version",
    "root_identity",
    "head_commit",
    "head_tree",
    "status_identity",
    "implementation_files_identity",
    "allowed_untracked_roots",
}
_OUTPUT_ROOT_SNAPSHOT_KEYS = {
    "schema_version",
    "lookup_key",
    "canonical_path_identity",
    "root_identity",
    "volume_identity",
    "file_identity",
    "governance_database_identity",
    "artifact_namespace_identity",
    "free_bytes",
}
_MALOM_SNAPSHOT_KEYS = {
    "schema_version",
    "lookup_key",
    "path_identity",
    "label_version",
    "manifest_file_sha256",
    "manifest_sha256",
    "content_sha256",
    "component_count",
    "size_bytes",
    "component_metadata_identity",
    "full_hash_evidence_identity",
    "full_hash_verified",
    "oracle_implementation_identity",
}
_SANMILL_SNAPSHOT_KEYS = {
    "schema_version",
    "lookup_key",
    "checkout_path_identity",
    "installation_identity",
    "runtime_identity",
    "commit",
    "tree",
    "binary_sha256",
    "binary_size",
    "license_sha256",
    "strict_referee_semantic_digest",
    "checkout_clean",
    "referee_implementation_identity",
}
_GENERATOR_SNAPSHOT_KEYS = {
    "schema_version",
    "contract_identity",
    "model_implementation_identity",
    "encoder_implementation_identity",
    "initial_policy_state_sha256",
    "initial_rng_state_identity",
    "current_rng_state_identity",
    "model_config",
    "sampler_config",
}
_RUNTIME_RECORD_CONTEXT_KEYS = {
    "experiment_id",
    "proposal_identity",
    "profile_identity",
    "store_spec_identity",
    "plan_identity",
    "readiness_identity",
    "authorization_identity",
    "authorization_consumption_identity",
    "attempt_identity",
    "reservation_event_identity",
    "state_generator_session_identity",
    "host_preflight_identity",
}
_STABILITY_COMPARISON_FIELDS = (
    "host.platform",
    "host.machine_identity",
    "host.python_executable_sha256",
    "host.python_version",
    "host.torch_version",
    "host.device",
    "host.cuda_initialized",
    "path_registry.file_sha256",
    "path_registry.canonical_object_identity",
    "repository.head_commit",
    "repository.head_tree",
    "repository.status_identity",
    "repository.implementation_files_identity",
    "output_root.root_identity",
    "output_root.volume_identity",
    "output_root.file_identity",
    "malom.label_version",
    "malom.manifest_file_sha256",
    "malom.manifest_sha256",
    "malom.content_sha256",
    "malom.component_count",
    "malom.size_bytes",
    "malom.component_metadata_identity",
    "malom.full_hash_evidence_identity",
    "sanmill.installation_identity",
    "sanmill.runtime_identity",
    "sanmill.commit",
    "sanmill.tree",
    "sanmill.binary_sha256",
    "sanmill.binary_size",
    "sanmill.license_sha256",
    "sanmill.strict_referee_semantic_digest",
    "sanmill.checkout_clean",
    "generator.contract_identity",
    "generator.model_implementation_identity",
    "generator.encoder_implementation_identity",
    "generator.initial_policy_state_sha256",
    "generator.initial_rng_state_identity",
)

_HOST_SNAPSHOT_SCHEMA = "nmm.classical-a-pos-state-generation-host-snapshot.v1"
_PATH_REGISTRY_SNAPSHOT_SCHEMA = (
    "nmm.classical-a-pos-state-generation-path-registry-snapshot.v1"
)
_REPOSITORY_SNAPSHOT_SCHEMA = (
    "nmm.classical-a-pos-state-generation-repository-snapshot.v1"
)
_OUTPUT_ROOT_SNAPSHOT_SCHEMA = (
    "nmm.classical-a-pos-state-generation-output-root-snapshot.v1"
)
_MALOM_SNAPSHOT_SCHEMA = "nmm.classical-a-pos-state-generation-malom-snapshot.v1"
_SANMILL_SNAPSHOT_SCHEMA = "nmm.classical-a-pos-state-generation-sanmill-snapshot.v1"
_GENERATOR_SNAPSHOT_SCHEMA = (
    "nmm.classical-a-pos-state-generation-generator-snapshot.v1"
)
_MODEL_CONFIG_KEYS = (
    "policy",
    "policy_hidden",
    "value_hidden",
    "dropout",
    "model_init_seed",
)
_SAMPLER_CONFIG_KEYS = (
    "sampling",
    "temperature",
    "cpu_generator_seed",
    "initial_state",
    "candidate_colours",
    "max_games",
    "informative_only",
)

_DDL_STATEMENTS = (
    "CREATE TABLE meta (key TEXT PRIMARY KEY NOT NULL, value BLOB NOT NULL) WITHOUT ROWID",
    (
        "CREATE TABLE events (sequence INTEGER PRIMARY KEY NOT NULL, "
        "event_identity TEXT NOT NULL, previous_event_identity TEXT, "
        "event_bytes BLOB NOT NULL, size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL)"
    ),
    (
        "CREATE TABLE artifacts (role TEXT PRIMARY KEY NOT NULL, "
        "artifact_identity TEXT NOT NULL, artifact_bytes BLOB NOT NULL, "
        "size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL) WITHOUT ROWID"
    ),
    (
        "CREATE TABLE domain_records ("
        "stream_identity TEXT NOT NULL, stream_kind TEXT NOT NULL, "
        "sequence INTEGER NOT NULL, record_type TEXT NOT NULL, "
        "previous_record_identity TEXT, record_identity TEXT NOT NULL UNIQUE, "
        "record_bytes BLOB NOT NULL, size_bytes INTEGER NOT NULL, "
        "record_bytes_sha256 TEXT NOT NULL UNIQUE, "
        "PRIMARY KEY (stream_identity, sequence), "
        "FOREIGN KEY (previous_record_identity) REFERENCES domain_records(record_identity), "
        "CHECK (sequence >= 0), "
        "CHECK ((sequence = 0 AND previous_record_identity IS NULL) OR "
        "(sequence > 0 AND previous_record_identity IS NOT NULL)), "
        "CHECK (length(record_identity) = 64), "
        "CHECK (length(record_bytes_sha256) = 64), "
        "CHECK (record_identity = record_bytes_sha256), "
        "CHECK (size_bytes >= 0 AND length(record_bytes) = size_bytes)) WITHOUT ROWID"
    ),
    (
        "CREATE TRIGGER meta_reject_update BEFORE UPDATE ON meta "
        "BEGIN SELECT RAISE(ABORT, 'meta is immutable'); END"
    ),
    (
        "CREATE TRIGGER meta_reject_delete BEFORE DELETE ON meta "
        "BEGIN SELECT RAISE(ABORT, 'meta is immutable'); END"
    ),
    (
        "CREATE TRIGGER events_reject_update BEFORE UPDATE ON events "
        "BEGIN SELECT RAISE(ABORT, 'events are immutable'); END"
    ),
    (
        "CREATE TRIGGER events_reject_delete BEFORE DELETE ON events "
        "BEGIN SELECT RAISE(ABORT, 'events are immutable'); END"
    ),
    (
        "CREATE TRIGGER artifacts_reject_update BEFORE UPDATE ON artifacts "
        "BEGIN SELECT RAISE(ABORT, 'artifacts are immutable'); END"
    ),
    (
        "CREATE TRIGGER artifacts_reject_delete BEFORE DELETE ON artifacts "
        "BEGIN SELECT RAISE(ABORT, 'artifacts are immutable'); END"
    ),
    (
        "CREATE TRIGGER domain_records_reject_update BEFORE UPDATE ON domain_records "
        "BEGIN SELECT RAISE(ABORT, 'domain records are immutable'); END"
    ),
    (
        "CREATE TRIGGER domain_records_reject_delete BEFORE DELETE ON domain_records "
        "BEGIN SELECT RAISE(ABORT, 'domain records are immutable'); END"
    ),
)


class GovernanceStoreContractError(RuntimeError):
    """The durable governance store failed closed."""


class _DurableRuntimeBeforePostCommitError(GovernanceStoreContractError):
    """A durable runtime-before row lost its in-process capability root."""


_RUNTIME_BEFORE_POSTCOMMIT_FATAL_MESSAGE = (
    "fatal post-commit runtime-before capability loss; no resume or retry"
)


def _freeze(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GovernanceStoreContractError(f"{field} contains non-finite data")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise GovernanceStoreContractError(f"{field} has a non-string key")
            copied[key] = _freeze(item, field=f"{field}.{key}")
        return MappingProxyType(copied)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(
            _freeze(item, field=f"{field}[{index}]") for index, item in enumerate(value)
        )
    raise GovernanceStoreContractError(
        f"{field} contains unsupported {type(value).__name__} data"
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _require_exact_keys(
    value: Any,
    expected: set[str],
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GovernanceStoreContractError(f"{field} must be an object")
    actual = set(value)
    if actual != expected or any(not isinstance(key, str) for key in value):
        raise GovernanceStoreContractError(f"{field} keys differ")
    return value


def _require_exact(value: Any, expected: Any, *, field: str) -> None:
    if isinstance(expected, Mapping):
        checked = _require_exact_keys(value, set(expected), field=field)
        for key in expected:
            _require_exact(checked[key], expected[key], field=f"{field}.{key}")
        return
    if isinstance(expected, Sequence) and not isinstance(
        expected,
        (str, bytes, bytearray),
    ):
        if not isinstance(value, Sequence) or isinstance(
            value,
            (str, bytes, bytearray),
        ):
            raise GovernanceStoreContractError(f"{field} must be an array")
        if len(value) != len(expected):
            raise GovernanceStoreContractError(f"{field} length differs")
        for index, (observed, frozen) in enumerate(zip(value, expected, strict=True)):
            _require_exact(observed, frozen, field=f"{field}[{index}]")
        return
    if type(value) is not type(expected) or value != expected:
        raise GovernanceStoreContractError(f"{field} differs")


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise GovernanceStoreContractError(f"{field} must be a lowercase SHA-256")
    return value


def _require_bound_identity(value: Any, *, field: str) -> str:
    identity = _require_sha256(value, field=field)
    if identity == "0" * 64:
        raise GovernanceStoreContractError(f"{field} must not be a zero placeholder")
    return identity


def _require_git_oid(value: Any, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or value != value.lower()
        or any(character not in _HEX for character in value)
        or value == "0" * 40
    ):
        raise GovernanceStoreContractError(f"{field} must be a bound lowercase Git OID")
    return value


def _require_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GovernanceStoreContractError(f"{field} must be a non-negative integer")
    return value


def _require_nonempty_text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise GovernanceStoreContractError(f"{field} must be non-empty text")
    return value


def _require_text_choice(
    value: Any,
    choices: set[str],
    *,
    field: str,
) -> str:
    text = _require_nonempty_text(value, field=field)
    if text not in choices:
        raise GovernanceStoreContractError(f"{field} differs")
    return text


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sqlite_blob(value: Any, *, field: str) -> bytes:
    if type(value) is not bytes:
        raise GovernanceStoreContractError(f"SQLite {field} must be a BLOB")
    return value


def _require_plain_object(
    value: Any,
    expected_keys: set[str],
    *,
    field: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected_keys:
        raise GovernanceStoreContractError(f"{field} exact object keys/type differ")
    return value


def _require_finite_json(value: Any, *, field: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise GovernanceStoreContractError(f"{field} must be finite")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _require_finite_json(item, field=f"{field}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise GovernanceStoreContractError(f"{field} has a non-string key")
            _require_finite_json(item, field=f"{field}.{key}")
        return
    raise GovernanceStoreContractError(
        f"{field} contains unsupported {type(value).__name__} data"
    )


def _require_test_sha(label: str, *, spec_identity: str) -> str:
    return canonical_sha256(
        {
            "schema_version": "nmm.classical-a-pos-test-runtime-identity.v1",
            "store_spec_identity": spec_identity,
            "label": label,
        }
    )


def _test_resource_snapshot(
    spec: GovernanceStoreSpec,
    *,
    attempt_identity: str,
    stage: str,
    observed_at_utc: str,
) -> dict[str, Any]:
    def identity(label: str) -> str:
        return _require_test_sha(
            label,
            spec_identity=spec["spec_identity"],
        )

    initial_policy = identity("initial-policy-state")
    generator_contract = _corpus._generator_contract(initial_policy)
    model_config = {key: generator_contract[key] for key in _MODEL_CONFIG_KEYS}
    sampler_config = {key: generator_contract[key] for key in _SAMPLER_CONFIG_KEYS}
    host = {
        "schema_version": _HOST_SNAPSHOT_SCHEMA,
        "platform": "Windows",
        "machine_identity": identity("machine"),
        "python_executable_sha256": identity("python-executable"),
        "python_version": "3.12-test",
        "torch_version": "test-cpu",
        "device": "cpu",
        "cuda_initialized": False,
        "available_memory_bytes": 8_589_934_592,
    }
    path_registry = {
        "schema_version": _PATH_REGISTRY_SNAPSHOT_SCHEMA,
        "registry_role": "machine-local-training-path-registry",
        "file_sha256": identity("path-registry-file"),
        "canonical_object_identity": identity("path-registry-object"),
        "required_lookup_keys": [
            "malom_db_path",
            "sanmill_training_checkout",
            "classical_a_pos_output_root",
        ],
        "resolved_path_identities": {
            "malom_db_path": identity("malom-path"),
            "sanmill_training_checkout": identity("sanmill-path"),
            "classical_a_pos_output_root": identity("output-path"),
        },
    }
    repository = {
        "schema_version": _REPOSITORY_SNAPSHOT_SCHEMA,
        "root_identity": identity("repository-root"),
        "head_commit": "a" * 40,
        "head_tree": "b" * 40,
        "status_identity": identity("repository-status"),
        "implementation_files_identity": identity("implementation-files"),
        "allowed_untracked_roots": ["tmp"],
    }
    output_root = {
        "schema_version": _OUTPUT_ROOT_SNAPSHOT_SCHEMA,
        "lookup_key": "classical_a_pos_output_root",
        "canonical_path_identity": identity("output-canonical-path"),
        "root_identity": identity("output-root"),
        "volume_identity": identity("output-volume"),
        "file_identity": identity("output-file"),
        "governance_database_identity": identity("governance-database"),
        "artifact_namespace_identity": identity("artifact-namespace"),
        "free_bytes": 536_870_912_000,
    }
    malom = {
        "schema_version": _MALOM_SNAPSHOT_SCHEMA,
        "lookup_key": "malom_db_path",
        "path_identity": identity("malom-path"),
        "label_version": "sector-corrected-v1",
        "manifest_file_sha256": identity("malom-manifest-file"),
        "manifest_sha256": identity("malom-manifest"),
        "content_sha256": identity("malom-content"),
        "component_count": 512,
        "size_bytes": 83_582_223_577,
        "component_metadata_identity": identity("malom-components"),
        "full_hash_evidence_identity": identity("malom-full-hash-evidence"),
        "full_hash_verified": True,
        "oracle_implementation_identity": identity("malom-oracle"),
    }
    sanmill = {
        "schema_version": _SANMILL_SNAPSHOT_SCHEMA,
        "lookup_key": "sanmill_training_checkout",
        "checkout_path_identity": identity("sanmill-path"),
        "installation_identity": identity("sanmill-installation"),
        "runtime_identity": identity("sanmill-runtime"),
        "commit": "c" * 40,
        "tree": "d" * 40,
        "binary_sha256": identity("sanmill-binary"),
        "binary_size": 1_048_576,
        "license_sha256": identity("sanmill-license"),
        "strict_referee_semantic_digest": TRAINING_REFEREE_SEMANTIC_DIGEST,
        "checkout_clean": True,
        "referee_implementation_identity": identity("sanmill-referee"),
    }
    initial_rng = identity("initial-rng")
    generator = {
        "schema_version": _GENERATOR_SNAPSHOT_SCHEMA,
        "contract_identity": canonical_sha256(generator_contract),
        "model_implementation_identity": identity("model-implementation"),
        "encoder_implementation_identity": identity("encoder-implementation"),
        "initial_policy_state_sha256": initial_policy,
        "initial_rng_state_identity": initial_rng,
        "current_rng_state_identity": (
            initial_rng if stage == "before-execution" else identity("after-rng")
        ),
        "model_config": model_config,
        "sampler_config": sampler_config,
    }
    body = {
        "schema_version": _RESOURCE_SNAPSHOT_SCHEMA,
        "stage": stage,
        "experiment_id": _EXPERIMENT_ID,
        "proposal_identity": _PROPOSAL_IDENTITY,
        "profile_identity": _PROFILE_IDENTITY,
        "store_spec_identity": spec["spec_identity"],
        "plan_identity": spec["plan_identity"],
        "readiness_identity": spec["readiness_identity"],
        "attempt_identity": attempt_identity,
        "observed_at_utc": observed_at_utc,
        "host": host,
        "path_registry": path_registry,
        "repository": repository,
        "output_root": output_root,
        "malom": malom,
        "sanmill": sanmill,
        "generator": generator,
    }
    return {**body, "snapshot_identity": canonical_sha256(body)}


def _validate_resource_snapshot(
    value: Any,
    *,
    expected_stage: str,
    spec: GovernanceStoreSpec,
    attempt_identity: str,
) -> dict[str, Any]:
    snapshot = _require_plain_object(
        value,
        _RESOURCE_SNAPSHOT_KEYS,
        field="resource snapshot",
    )
    _require_finite_json(snapshot, field="resource snapshot")
    for key, expected in {
        "schema_version": _RESOURCE_SNAPSHOT_SCHEMA,
        "stage": expected_stage,
        "experiment_id": _EXPERIMENT_ID,
        "proposal_identity": _PROPOSAL_IDENTITY,
        "profile_identity": _PROFILE_IDENTITY,
        "store_spec_identity": spec["spec_identity"],
        "plan_identity": spec["plan_identity"],
        "readiness_identity": spec["readiness_identity"],
        "attempt_identity": attempt_identity,
    }.items():
        _require_exact(snapshot[key], expected, field=f"resource snapshot.{key}")
    _require_nonempty_text(
        snapshot["observed_at_utc"], field="resource snapshot.observed_at_utc"
    )
    host = _require_plain_object(snapshot["host"], _HOST_SNAPSHOT_KEYS, field="host")
    for key, expected in {
        "schema_version": _HOST_SNAPSHOT_SCHEMA,
        "platform": "Windows",
        "device": "cpu",
        "cuda_initialized": False,
    }.items():
        _require_exact(host[key], expected, field=f"host.{key}")
    for key in ("machine_identity", "python_executable_sha256"):
        _require_bound_identity(host[key], field=f"host.{key}")
    for key in ("python_version", "torch_version"):
        _require_nonempty_text(host[key], field=f"host.{key}")
    _require_nonnegative_int(host["available_memory_bytes"], field="host memory")

    paths = _require_plain_object(
        snapshot["path_registry"],
        _PATH_REGISTRY_SNAPSHOT_KEYS,
        field="path registry",
    )
    _require_exact(
        paths["schema_version"],
        _PATH_REGISTRY_SNAPSHOT_SCHEMA,
        field="path registry schema",
    )
    _require_exact(
        paths["registry_role"],
        "machine-local-training-path-registry",
        field="path registry role",
    )
    for key in ("file_sha256", "canonical_object_identity"):
        _require_bound_identity(paths[key], field=f"path registry.{key}")
    required_paths = [
        "malom_db_path",
        "sanmill_training_checkout",
        "classical_a_pos_output_root",
    ]
    _require_exact(
        paths["required_lookup_keys"], required_paths, field="required lookup keys"
    )
    resolved = _require_plain_object(
        paths["resolved_path_identities"],
        set(required_paths),
        field="resolved path identities",
    )
    for key in required_paths:
        _require_bound_identity(resolved[key], field=f"resolved path {key}")

    repository = _require_plain_object(
        snapshot["repository"],
        _REPOSITORY_SNAPSHOT_KEYS,
        field="repository",
    )
    _require_exact(
        repository["schema_version"],
        _REPOSITORY_SNAPSHOT_SCHEMA,
        field="repository schema",
    )
    for key in ("root_identity", "status_identity", "implementation_files_identity"):
        _require_bound_identity(repository[key], field=f"repository.{key}")
    for key in ("head_commit", "head_tree"):
        _require_git_oid(repository[key], field=f"repository.{key}")
    _require_exact(
        repository["allowed_untracked_roots"],
        ["tmp"],
        field="repository allowed untracked roots",
    )

    output = _require_plain_object(
        snapshot["output_root"],
        _OUTPUT_ROOT_SNAPSHOT_KEYS,
        field="output root",
    )
    _require_exact(
        output["schema_version"],
        _OUTPUT_ROOT_SNAPSHOT_SCHEMA,
        field="output schema",
    )
    _require_exact(
        output["lookup_key"],
        "classical_a_pos_output_root",
        field="output lookup",
    )
    for key in (
        "canonical_path_identity",
        "root_identity",
        "volume_identity",
        "file_identity",
        "governance_database_identity",
        "artifact_namespace_identity",
    ):
        _require_bound_identity(output[key], field=f"output root.{key}")
    _require_nonnegative_int(output["free_bytes"], field="output free bytes")

    malom = _require_plain_object(
        snapshot["malom"], _MALOM_SNAPSHOT_KEYS, field="malom"
    )
    for key, expected in {
        "schema_version": _MALOM_SNAPSHOT_SCHEMA,
        "lookup_key": "malom_db_path",
        "label_version": "sector-corrected-v1",
        "full_hash_verified": True,
    }.items():
        _require_exact(malom[key], expected, field=f"malom.{key}")
    for key in (
        "path_identity",
        "manifest_file_sha256",
        "manifest_sha256",
        "content_sha256",
        "component_metadata_identity",
        "full_hash_evidence_identity",
        "oracle_implementation_identity",
    ):
        _require_bound_identity(malom[key], field=f"malom.{key}")
    _require_nonnegative_int(malom["component_count"], field="malom components")
    _require_nonnegative_int(malom["size_bytes"], field="malom size")

    sanmill = _require_plain_object(
        snapshot["sanmill"], _SANMILL_SNAPSHOT_KEYS, field="sanmill"
    )
    for key, expected in {
        "schema_version": _SANMILL_SNAPSHOT_SCHEMA,
        "lookup_key": "sanmill_training_checkout",
        "checkout_clean": True,
    }.items():
        _require_exact(sanmill[key], expected, field=f"sanmill.{key}")
    for key in (
        "checkout_path_identity",
        "installation_identity",
        "runtime_identity",
        "binary_sha256",
        "license_sha256",
        "referee_implementation_identity",
    ):
        _require_bound_identity(sanmill[key], field=f"sanmill.{key}")
    _require_exact(
        sanmill["strict_referee_semantic_digest"],
        TRAINING_REFEREE_SEMANTIC_DIGEST,
        field="sanmill.strict_referee_semantic_digest",
    )
    for key in ("commit", "tree"):
        _require_git_oid(sanmill[key], field=f"sanmill.{key}")
    _require_nonnegative_int(sanmill["binary_size"], field="sanmill binary size")

    generator = _require_plain_object(
        snapshot["generator"],
        _GENERATOR_SNAPSHOT_KEYS,
        field="generator",
    )
    _require_exact(
        generator["schema_version"],
        _GENERATOR_SNAPSHOT_SCHEMA,
        field="generator schema",
    )
    for key in (
        "contract_identity",
        "model_implementation_identity",
        "encoder_implementation_identity",
        "initial_policy_state_sha256",
        "initial_rng_state_identity",
        "current_rng_state_identity",
    ):
        _require_bound_identity(generator[key], field=f"generator.{key}")
    frozen_generator = _corpus._generator_contract(
        generator["initial_policy_state_sha256"]
    )
    expected_model = {key: frozen_generator[key] for key in _MODEL_CONFIG_KEYS}
    expected_sampler = {key: frozen_generator[key] for key in _SAMPLER_CONFIG_KEYS}
    _require_exact(
        generator["model_config"], expected_model, field="generator.model_config"
    )
    _require_exact(
        generator["sampler_config"],
        expected_sampler,
        field="generator.sampler_config",
    )
    _require_exact(
        generator["contract_identity"],
        canonical_sha256(frozen_generator),
        field="generator.contract_identity",
    )
    if expected_stage == "before-execution":
        _require_exact(
            generator["current_rng_state_identity"],
            generator["initial_rng_state_identity"],
            field="before generator RNG",
        )
    body = {key: item for key, item in snapshot.items() if key != "snapshot_identity"}
    _require_exact(
        snapshot["snapshot_identity"],
        canonical_sha256(body),
        field="resource snapshot identity",
    )
    return snapshot


def _nested_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if type(current) is not dict or component not in current:
            raise GovernanceStoreContractError(f"stability path {path} is absent")
        current = current[component]
    return current


@dataclass(frozen=True, slots=True)
class GovernanceStoreSpec(Mapping[str, Any]):
    """Deeply immutable exact durable-store specification."""

    _data: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_data", _freeze(self._data, field="store spec"))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self._data)


def _expected_sqlite_master_rows() -> tuple[tuple[str, str, str, str], ...]:
    rows: list[tuple[str, str, str, str]] = []
    for statement in _DDL_STATEMENTS:
        words = statement.split()
        object_type = words[1].lower()
        name = words[2]
        if object_type == "table":
            table_name = name
        else:
            on_index = words.index("ON")
            table_name = words[on_index + 1]
        rows.append((object_type, name, table_name, statement))
    return tuple(sorted(rows))


_EXPECTED_SQLITE_MASTER_ROWS = _expected_sqlite_master_rows()
_SCHEMA_DDL_IDENTITY = canonical_sha256(
    {
        "schema_version": "nmm.classical-a-pos-governance-sqlite-ddl.v1",
        "sqlite_master": [list(row) for row in _EXPECTED_SQLITE_MASTER_ROWS],
    }
)


def _verified_spec(value: Mapping[str, Any]) -> GovernanceStoreSpec:
    raw = value.to_dict() if type(value) is GovernanceStoreSpec else _thaw(value)
    checked = _require_exact_keys(raw, _SPEC_KEYS, field="governance store spec")
    fixed = {
        "schema_version": _SPEC_SCHEMA,
        "experiment_id": _EXPERIMENT_ID,
        "proposal_identity": _PROPOSAL_IDENTITY,
        "profile_identity": _PROFILE_IDENTITY,
        "database_role": _DATABASE_ROLE,
        "normalized_relative_path": _RELATIVE_DATABASE_PATH,
        "schema_ddl_identity": _SCHEMA_DDL_IDENTITY,
    }
    for key, expected in fixed.items():
        _require_exact(checked[key], expected, field=f"store spec.{key}")
    for key in (
        "plan_identity",
        "readiness_identity",
        "managed_git_state_identity",
        "launch_path_binding_identity",
        "state_freeze_teacher_order_event_chain_identity",
        "output_root_identity",
    ):
        _require_sha256(checked[key], field=f"store spec.{key}")
    observed_identity = _require_sha256(
        checked["spec_identity"],
        field="store spec.spec_identity",
    )
    body = {key: item for key, item in checked.items() if key != "spec_identity"}
    if observed_identity != canonical_sha256(body):
        raise GovernanceStoreContractError("store spec identity differs")
    return GovernanceStoreSpec(checked)


def build_governance_store_spec(
    plan: Mapping[str, Any],
    *,
    readiness_identity: str,
    output_root_identity: str,
) -> GovernanceStoreSpec:
    """Build the nonce-free store spec from one complete frozen plan."""
    try:
        checked_plan = verify_runtime_plan(plan)
    except Exception as exc:
        raise GovernanceStoreContractError("store spec plan is invalid") from exc
    if (
        checked_plan["plan_status"] != "frozen"
        or checked_plan["issuable"] is not True
        or checked_plan["executable"] is not False
        or checked_plan["unresolved_bindings"] != ()
    ):
        raise GovernanceStoreContractError("store spec requires a complete frozen plan")
    bindings = checked_plan["technical_bindings"]
    body = {
        "schema_version": _SPEC_SCHEMA,
        "experiment_id": _EXPERIMENT_ID,
        "proposal_identity": _PROPOSAL_IDENTITY,
        "profile_identity": _PROFILE_IDENTITY,
        "plan_identity": checked_plan["plan_identity"],
        "readiness_identity": _require_sha256(
            readiness_identity,
            field="readiness_identity",
        ),
        "managed_git_state_identity": _require_sha256(
            bindings["managed_git_state"],
            field="managed_git_state binding",
        ),
        "launch_path_binding_identity": _require_sha256(
            bindings["launch_path_binding"],
            field="launch_path_binding binding",
        ),
        "state_freeze_teacher_order_event_chain_identity": _require_sha256(
            bindings["state_freeze_teacher_order_event_chain"],
            field="state_freeze_teacher_order_event_chain binding",
        ),
        "database_role": _DATABASE_ROLE,
        "output_root_identity": _require_sha256(
            output_root_identity,
            field="output_root_identity",
        ),
        "normalized_relative_path": _RELATIVE_DATABASE_PATH,
        "schema_ddl_identity": _SCHEMA_DDL_IDENTITY,
    }
    return _verified_spec({**body, "spec_identity": canonical_sha256(body)})


_PRODUCTION_STORE_TOKEN = object()
_PRODUCTION_INVENTORY_TOKEN = object()
_PRODUCTION_STRICT_REFEREE_TOKEN = object()
_PRODUCTION_RUNTIME_PREFLIGHT_TOKEN = object()
_PRODUCTION_ACTIVE_TOKEN = object()
_PRODUCTION_BOUND_ATTEMPT_TOKEN = object()
_PRODUCTION_PENDING_TOKEN = object()
_PRODUCTION_COMPLETION_TOKEN = object()
_PRODUCTION_RUNTIME_COMPLETION_TOKEN = object()
_PRODUCTION_FREEZE_TOKEN = object()
_TEST_STORE_TOKEN = object()
_TEST_INVENTORY_TOKEN = object()
_TEST_STRICT_REFEREE_TOKEN = object()
_TEST_RUNTIME_PREFLIGHT_TOKEN = object()
_TEST_ACTIVE_TOKEN = object()
_TEST_BOUND_ATTEMPT_TOKEN = object()
_TEST_PENDING_TOKEN = object()
_TEST_COMPLETION_TOKEN = object()
_TEST_RUNTIME_COMPLETION_TOKEN = object()
_TEST_FREEZE_TOKEN = object()


class _OpaqueCapability:
    __slots__ = ("__weakref__",)
    _token: object
    _description: str

    def __init__(self, token: object) -> None:
        if token is not self._token:
            raise GovernanceStoreContractError(
                f"{self._description} must come from its controlled issuer"
            )

    def __copy__(self) -> Any:
        raise TypeError(f"{self._description} cannot be copied")

    def __deepcopy__(self, memo: Any) -> Any:
        del memo
        raise TypeError(f"{self._description} cannot be copied")

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        raise TypeError(f"{self._description} cannot be serialized")


class DurableGovernanceStore(_OpaqueCapability):
    __slots__ = ()
    _token = _PRODUCTION_STORE_TOKEN
    _description = "durable governance store"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("durable governance store cannot be subclassed")


class ProductionAPosInventoryBinding(_OpaqueCapability):
    __slots__ = ()
    _token = _PRODUCTION_INVENTORY_TOKEN
    _description = "production A_pos inventory binding"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("production A_pos inventory binding cannot be subclassed")


class ProductionStrictRefereeBinding(_OpaqueCapability):
    __slots__ = ()
    _token = _PRODUCTION_STRICT_REFEREE_TOKEN
    _description = "production strict-referee binding"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("production strict-referee binding cannot be subclassed")


class ProductionStateGenerationRuntimePreflight(_OpaqueCapability):
    __slots__ = ()
    _token = _PRODUCTION_RUNTIME_PREFLIGHT_TOKEN
    _description = "production state-generation runtime preflight"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError(
            "production state-generation runtime preflight cannot be subclassed"
        )


class ActiveStateGenerationAttempt(_OpaqueCapability):
    __slots__ = ()
    _token = _PRODUCTION_ACTIVE_TOKEN
    _description = "active state-generation attempt"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("active state-generation attempt cannot be subclassed")


class RuntimeBoundStateGenerationAttempt(_OpaqueCapability):
    __slots__ = ()
    _token = _PRODUCTION_BOUND_ATTEMPT_TOKEN
    _description = "runtime-bound state-generation attempt"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("runtime-bound state-generation attempt cannot be subclassed")


class PendingStateGenerationCommit(_OpaqueCapability):
    __slots__ = ()
    _token = _PRODUCTION_PENDING_TOKEN
    _description = "pending state-generation commit"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("pending state-generation commit cannot be subclassed")


class ConfirmedStateGenerationCompletion(_OpaqueCapability):
    __slots__ = ()
    _token = _PRODUCTION_COMPLETION_TOKEN
    _description = "confirmed state-generation completion"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("confirmed state-generation completion cannot be subclassed")


class ProductionStateGenerationRuntimeCompletion(_OpaqueCapability):
    __slots__ = ()
    _token = _PRODUCTION_RUNTIME_COMPLETION_TOKEN
    _description = "production state-generation runtime completion"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError(
            "production state-generation runtime completion cannot be subclassed"
        )


class DurableStateFreezeBinding(_OpaqueCapability):
    __slots__ = ()
    _token = _PRODUCTION_FREEZE_TOKEN
    _description = "durable state-freeze binding"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("durable state-freeze binding cannot be subclassed")


class _TestDurableGovernanceStore(_OpaqueCapability):
    __slots__ = ()
    _token = _TEST_STORE_TOKEN
    _description = "test durable governance store"


class _TestAPosInventoryBinding(_OpaqueCapability):
    __slots__ = ()
    _token = _TEST_INVENTORY_TOKEN
    _description = "test A_pos inventory binding"


class _TestStrictRefereeBinding(_OpaqueCapability):
    __slots__ = ()
    _token = _TEST_STRICT_REFEREE_TOKEN
    _description = "test strict-referee binding"


class _TestStateGenerationRuntimePreflight(_OpaqueCapability):
    __slots__ = ()
    _token = _TEST_RUNTIME_PREFLIGHT_TOKEN
    _description = "test state-generation runtime preflight"


class _TestActiveStateGenerationAttempt(_OpaqueCapability):
    __slots__ = ()
    _token = _TEST_ACTIVE_TOKEN
    _description = "test active state-generation attempt"


class _TestRuntimeBoundStateGenerationAttempt(_OpaqueCapability):
    __slots__ = ()
    _token = _TEST_BOUND_ATTEMPT_TOKEN
    _description = "test runtime-bound state-generation attempt"


class _TestPendingStateGenerationCommit(_OpaqueCapability):
    __slots__ = ()
    _token = _TEST_PENDING_TOKEN
    _description = "test pending state-generation commit"


class _TestConfirmedStateGenerationCompletion(_OpaqueCapability):
    __slots__ = ()
    _token = _TEST_COMPLETION_TOKEN
    _description = "test confirmed state-generation completion"


class _TestStateGenerationRuntimeCompletion(_OpaqueCapability):
    __slots__ = ()
    _token = _TEST_RUNTIME_COMPLETION_TOKEN
    _description = "test state-generation runtime completion"


class _TestDurableStateFreezeBinding(_OpaqueCapability):
    __slots__ = ()
    _token = _TEST_FREEZE_TOKEN
    _description = "test durable state-freeze binding"


def initialize_governance_store(
    output_root: str | Path,
    spec: GovernanceStoreSpec,
    *,
    plan_permit: ProductionRuntimePlanPermit,
    authorization_permit: ProductionAuthorizationPermit,
) -> DurableGovernanceStore:
    """Initialize from one registry-backed production bootstrap claim."""
    store = _initialize_store_core(
        output_root,
        spec,
        plan_permit=plan_permit,
        authorization_permit=authorization_permit,
        domain="production",
        claim_bootstrap=_governance._claim_production_store_bootstrap,
        consume_claim=_governance._consume_production_store_bootstrap_claim,
        store_type=DurableGovernanceStore,
        store_token=_PRODUCTION_STORE_TOKEN,
        contexts=_PRODUCTION_STORE_CONTEXTS,
    )
    assert type(store) is DurableGovernanceStore
    return store


def open_governance_store(
    output_root: str | Path,
    spec: GovernanceStoreSpec,
    *,
    plan_permit: ProductionRuntimePlanPermit,
    authorization_permit: ProductionAuthorizationPermit,
) -> DurableGovernanceStore:
    """Open one exact production store through registry-backed permits."""
    checked_spec = _verified_spec(spec)
    try:
        opened = _governance._inspect_production_store_open_context(
            plan_permit,
            authorization_permit,
            store_spec_identity=checked_spec["spec_identity"],
        )
    except Exception as exc:
        raise GovernanceStoreContractError(
            "production store open permit binding differs"
        ) from exc
    store = _open_store_core(
        output_root,
        checked_spec,
        plan=opened.plan,
        authorization=opened.authorization,
        expected_domain="production",
        store_type=DurableGovernanceStore,
        store_token=_PRODUCTION_STORE_TOKEN,
        contexts=_PRODUCTION_STORE_CONTEXTS,
    )
    assert type(store) is DurableGovernanceStore
    return store


@dataclass(slots=True)
class _StoreContext:
    path: Path
    output_root: Path
    spec: GovernanceStoreSpec
    plan: RuntimePlanRecord
    authorization: SingleUseAuthorizationRecord
    domain: str
    replay: GovernanceReplay
    runtime: _StoreRuntimeContext | None


@dataclass(frozen=True, slots=True)
class _StoreRuntimeContext:
    host_preflight_identity: str
    state_generator_session_identity: str


@dataclass(frozen=True, slots=True)
class _InventoryContext:
    binding_identity: str
    verifier_identity: str
    inventory_verifier: Callable[
        [BoardState, Sequence[Mapping[str, Any]]], Sequence[bool]
    ]
    policy: _ArtifactPolicy


@dataclass(frozen=True, slots=True)
class _StrictRefereeContext:
    binding_identity: str
    runtime_identity: str
    attempt_identity: str
    complete_history_verifier: Callable[
        [Sequence[Mapping[str, Any]], Sequence[str]], UciPositionState
    ]
    prefix_history_verifier: Callable[[Sequence[Mapping[str, Any]]], str]
    domain: str


@dataclass(slots=True)
class _ActiveContext:
    store: object
    attempt_identity: str
    reservation_event_identity: str
    bind_consumed: bool = False
    spent: bool = False


@dataclass(slots=True)
class _RuntimePreflightContext:
    store: object
    active: object
    domain: str
    store_spec_identity: str
    plan_identity: str
    readiness_identity: str
    attempt_identity: str
    reservation_event_identity: str
    host_preflight_identity: str
    state_generator_session_identity: str
    resource_snapshot_before: Mapping[str, Any]
    resource_snapshot_before_identity: str
    preflight_identity: str
    spent: bool = False


@dataclass(slots=True)
class _BoundAttemptContext:
    store: object
    active: object
    domain: str
    preflight_identity: str
    host_preflight_identity: str
    state_generator_session_identity: str
    runtime_stream_identity: str
    resource_snapshot_before: Mapping[str, Any]
    resource_snapshot_before_identity: str
    before_record_bytes: bytes
    before_record_identity: str
    runtime_completion_issued: bool = False
    spent: bool = False


@dataclass(slots=True)
class _RuntimeCompletionContext:
    store: object
    bound_attempt: object
    runtime_stream_identity: str
    resource_snapshot_before: Mapping[str, Any]
    resource_snapshot_after: Mapping[str, Any]
    before_record_identity: str
    after_record_bytes: bytes
    after_record_identity: str
    stability_record_bytes: bytes
    stability_record_identity: str
    spent: bool = False


@dataclass(frozen=True, slots=True)
class _ArtifactPolicy:
    layout: Mapping[str, Mapping[str, Mapping[str, int]]]
    total_states: int
    maximum_source_games: int
    require_complete_game_block: bool
    domain: str


@dataclass(slots=True)
class _PendingStateContext:
    store: object
    active: object
    inventory_binding: object
    strict_referee_binding: object
    runtime_completion: object
    source_games_path: Path
    state_split_path: Path
    singleton_ledger_path: Path
    source_games_sha256: str
    state_split_sha256: str
    singleton_ledger_sha256: str
    source_games_file_key: tuple[int, int, int, int]
    state_split_file_key: tuple[int, int, int, int]
    singleton_ledger_file_key: tuple[int, int, int, int]
    timestamp_utc: str
    active_seconds: int
    policy: _ArtifactPolicy
    event_bytes: bytes
    spent: bool = False


@dataclass(slots=True)
class _CompletionContext:
    store: object
    completion_event_identity: str
    attempt_identity: str
    source_games_identity: str
    state_split_identity: str
    singleton_ledger_identity: str
    completion_record_identity: str
    runtime_evidence_stream_identity: str
    resource_stability_identity: str
    spent: bool = False


@dataclass(frozen=True, slots=True)
class _FreezeContext:
    store: object
    freeze_event_identity: str
    state_split_identity: str
    freeze_receipt_identity: str


_PRODUCTION_STORE_CONTEXTS: weakref.WeakKeyDictionary[
    DurableGovernanceStore,
    _StoreContext,
] = weakref.WeakKeyDictionary()
_TEST_STORE_CONTEXTS: weakref.WeakKeyDictionary[
    _TestDurableGovernanceStore,
    _StoreContext,
] = weakref.WeakKeyDictionary()
_PRODUCTION_INVENTORY_CONTEXTS: weakref.WeakKeyDictionary[
    ProductionAPosInventoryBinding,
    _InventoryContext,
] = weakref.WeakKeyDictionary()
_TEST_INVENTORY_CONTEXTS: weakref.WeakKeyDictionary[
    _TestAPosInventoryBinding,
    _InventoryContext,
] = weakref.WeakKeyDictionary()
_PRODUCTION_STRICT_REFEREE_CONTEXTS: weakref.WeakKeyDictionary[
    ProductionStrictRefereeBinding,
    _StrictRefereeContext,
] = weakref.WeakKeyDictionary()
_TEST_STRICT_REFEREE_CONTEXTS: weakref.WeakKeyDictionary[
    _TestStrictRefereeBinding,
    _StrictRefereeContext,
] = weakref.WeakKeyDictionary()
_PRODUCTION_ACTIVE_CONTEXTS: weakref.WeakKeyDictionary[
    ActiveStateGenerationAttempt,
    _ActiveContext,
] = weakref.WeakKeyDictionary()
_TEST_ACTIVE_CONTEXTS: weakref.WeakKeyDictionary[
    _TestActiveStateGenerationAttempt,
    _ActiveContext,
] = weakref.WeakKeyDictionary()
_PRODUCTION_RUNTIME_PREFLIGHT_CONTEXTS: weakref.WeakKeyDictionary[
    ProductionStateGenerationRuntimePreflight,
    _RuntimePreflightContext,
] = weakref.WeakKeyDictionary()
_TEST_RUNTIME_PREFLIGHT_CONTEXTS: weakref.WeakKeyDictionary[
    _TestStateGenerationRuntimePreflight,
    _RuntimePreflightContext,
] = weakref.WeakKeyDictionary()
_PRODUCTION_BOUND_ATTEMPT_CONTEXTS: weakref.WeakKeyDictionary[
    RuntimeBoundStateGenerationAttempt,
    _BoundAttemptContext,
] = weakref.WeakKeyDictionary()
_TEST_BOUND_ATTEMPT_CONTEXTS: weakref.WeakKeyDictionary[
    _TestRuntimeBoundStateGenerationAttempt,
    _BoundAttemptContext,
] = weakref.WeakKeyDictionary()
_PRODUCTION_PENDING_CONTEXTS: weakref.WeakKeyDictionary[
    PendingStateGenerationCommit,
    _PendingStateContext,
] = weakref.WeakKeyDictionary()
_TEST_PENDING_CONTEXTS: weakref.WeakKeyDictionary[
    _TestPendingStateGenerationCommit,
    _PendingStateContext,
] = weakref.WeakKeyDictionary()
_PRODUCTION_COMPLETION_CONTEXTS: weakref.WeakKeyDictionary[
    ConfirmedStateGenerationCompletion,
    _CompletionContext,
] = weakref.WeakKeyDictionary()
_TEST_COMPLETION_CONTEXTS: weakref.WeakKeyDictionary[
    _TestConfirmedStateGenerationCompletion,
    _CompletionContext,
] = weakref.WeakKeyDictionary()
_PRODUCTION_RUNTIME_COMPLETION_CONTEXTS: weakref.WeakKeyDictionary[
    ProductionStateGenerationRuntimeCompletion,
    _RuntimeCompletionContext,
] = weakref.WeakKeyDictionary()
_TEST_RUNTIME_COMPLETION_CONTEXTS: weakref.WeakKeyDictionary[
    _TestStateGenerationRuntimeCompletion,
    _RuntimeCompletionContext,
] = weakref.WeakKeyDictionary()
_PRODUCTION_FREEZE_CONTEXTS: weakref.WeakKeyDictionary[
    DurableStateFreezeBinding,
    _FreezeContext,
] = weakref.WeakKeyDictionary()
_TEST_FREEZE_CONTEXTS: weakref.WeakKeyDictionary[
    _TestDurableStateFreezeBinding,
    _FreezeContext,
] = weakref.WeakKeyDictionary()
_PRODUCTION_DURABLE_PENDING_ATTEMPTS: weakref.WeakSet[object] = weakref.WeakSet()
_TEST_DURABLE_PENDING_ATTEMPTS: weakref.WeakSet[object] = weakref.WeakSet()
_PRODUCTION_OPERATION_STORE_BINDINGS: weakref.WeakKeyDictionary[
    ProductionOperationPermit,
    DurableGovernanceStore,
] = weakref.WeakKeyDictionary()
_TEST_OPERATION_STORE_BINDINGS: weakref.WeakKeyDictionary[object, object] = (
    weakref.WeakKeyDictionary()
)
_PRODUCTION_PREFLIGHT_BY_ACTIVE: weakref.WeakKeyDictionary[object, object] = (
    weakref.WeakKeyDictionary()
)
_TEST_PREFLIGHT_BY_ACTIVE: weakref.WeakKeyDictionary[object, object] = (
    weakref.WeakKeyDictionary()
)

_PRODUCTION_POLICY = _ArtifactPolicy(
    layout=_freeze(FROZEN_CORPUS_LAYOUT.to_dict(), field="production layout"),
    total_states=16_384,
    maximum_source_games=1_024,
    require_complete_game_block=True,
    domain="production",
)


def _strict_json_object(payload: bytes, *, field: str) -> dict[str, Any]:
    if type(payload) is not bytes or payload.startswith(b"\xef\xbb\xbf"):
        raise GovernanceStoreContractError(f"{field} is not strict UTF-8 JSON")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise GovernanceStoreContractError(
                    f"{field} contains duplicate JSON key {key}"
                )
            result[key] = item
        return result

    def constant(token: str) -> None:
        raise GovernanceStoreContractError(
            f"{field} contains non-finite JSON constant {token}"
        )

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except GovernanceStoreContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GovernanceStoreContractError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise GovernanceStoreContractError(f"{field} must be a JSON object")
    if payload != canonical_json_bytes(parsed):
        raise GovernanceStoreContractError(f"{field} is not canonical JSON")
    return parsed


def _strict_jsonl_objects(payload: bytes, *, field: str) -> tuple[dict[str, Any], ...]:
    if type(payload) is not bytes or not payload or not payload.endswith(b"\n"):
        raise GovernanceStoreContractError(f"{field} requires canonical JSONL")
    if b"\r" in payload or payload.startswith(b"\xef\xbb\xbf"):
        raise GovernanceStoreContractError(f"{field} framing differs")
    lines = payload.split(b"\n")[:-1]
    if any(not line for line in lines):
        raise GovernanceStoreContractError(f"{field} contains a blank line")
    return tuple(
        _strict_json_object(line, field=f"{field}[{index}]")
        for index, line in enumerate(lines)
    )


def _check_no_sidecars(path: Path) -> None:
    present = [
        str(path) + suffix
        for suffix in _SIDECAR_SUFFIXES
        if Path(str(path) + suffix).exists()
    ]
    if present:
        raise GovernanceStoreContractError(
            "governance database sidecar is present: " + ", ".join(present)
        )


def _is_reparse(path: Path) -> bool:
    try:
        value = path.lstat()
    except OSError as exc:
        raise GovernanceStoreContractError("store path cannot be inspected") from exc
    attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    file_attributes = getattr(value, "st_file_attributes", 0)
    return path.is_symlink() or bool(file_attributes & attribute)


def _resolve_output_root(value: str | Path) -> Path:
    try:
        supplied = Path(value)
        absolute = supplied.absolute()
        resolved = supplied.resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise GovernanceStoreContractError("output root is missing or invalid") from exc
    if absolute != resolved or not resolved.is_dir() or _is_reparse(resolved):
        raise GovernanceStoreContractError(
            "output root must be one exact non-reparse directory"
        )
    return resolved


def _database_path(
    output_root: str | Path,
    spec: GovernanceStoreSpec,
    *,
    for_create: bool,
) -> tuple[Path, Path]:
    checked_spec = _verified_spec(spec)
    relative = PurePosixPath(checked_spec["normalized_relative_path"])
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != _RELATIVE_DATABASE_PATH
    ):
        raise GovernanceStoreContractError("store relative path differs")
    root = _resolve_output_root(output_root)
    namespace = root / relative.parts[0]
    path = root.joinpath(*relative.parts)
    if for_create:
        if namespace.exists() or path.exists():
            raise GovernanceStoreContractError(
                "governance store namespace already exists"
            )
        try:
            namespace.mkdir()
        except OSError as exc:
            raise GovernanceStoreContractError(
                "governance store namespace cannot be created"
            ) from exc
    else:
        if (
            not namespace.is_dir()
            or _is_reparse(namespace)
            or not path.is_file()
            or _is_reparse(path)
        ):
            raise GovernanceStoreContractError("governance database path differs")
        try:
            if path.stat().st_nlink != 1:
                raise GovernanceStoreContractError(
                    "governance database aliases are forbidden"
                )
        except OSError as exc:
            raise GovernanceStoreContractError(
                "governance database cannot be inspected"
            ) from exc
    try:
        parent_resolved = path.parent.resolve(strict=True)
    except OSError as exc:
        raise GovernanceStoreContractError("store parent cannot be resolved") from exc
    if parent_resolved != namespace or not path.is_relative_to(root):
        raise GovernanceStoreContractError("store path escaped its output root")
    _check_no_sidecars(path)
    return root, path


def _apply_connection_pragmas(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA busy_timeout=0")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA trusted_schema=OFF")
    connection.execute("PRAGMA synchronous=FULL")


def _open_write_connection(path: Path) -> sqlite3.Connection:
    _check_no_sidecars(path)
    try:
        connection = sqlite3.connect(path, timeout=0.0, isolation_level=None)
        _apply_connection_pragmas(connection)
        mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()
        if mode is None or str(mode[0]).lower() != "delete":
            raise GovernanceStoreContractError("SQLite journal mode is not DELETE")
        return connection
    except GovernanceStoreContractError:
        raise
    except sqlite3.Error as exc:
        raise GovernanceStoreContractError(
            "governance database write-open failed"
        ) from exc


def _open_read_connection(path: Path) -> sqlite3.Connection:
    _check_no_sidecars(path)
    try:
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode=ro",
            uri=True,
            timeout=0.0,
            isolation_level=None,
        )
        _apply_connection_pragmas(connection)
        return connection
    except sqlite3.Error as exc:
        raise GovernanceStoreContractError(
            "governance database read-open failed"
        ) from exc


def _verify_pragmas(connection: sqlite3.Connection) -> None:
    expected = {
        "journal_mode": "delete",
        "synchronous": 2,
        "foreign_keys": 1,
        "trusted_schema": 0,
        "busy_timeout": 0,
    }
    for pragma, frozen in expected.items():
        row = connection.execute(f"PRAGMA {pragma}").fetchone()
        observed = None if row is None else row[0]
        if isinstance(frozen, str):
            observed = str(observed).lower()
        if observed != frozen:
            raise GovernanceStoreContractError(f"SQLite {pragma} differs")


def _verify_schema(connection: sqlite3.Connection) -> None:
    _verify_pragmas(connection)
    rows = tuple(
        sorted(
            connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()
        )
    )
    if rows != _EXPECTED_SQLITE_MASTER_ROWS:
        raise GovernanceStoreContractError("SQLite schema objects differ")
    observed_identity = canonical_sha256(
        {
            "schema_version": "nmm.classical-a-pos-governance-sqlite-ddl.v1",
            "sqlite_master": [list(row) for row in rows],
        }
    )
    if observed_identity != _SCHEMA_DDL_IDENTITY:
        raise GovernanceStoreContractError("SQLite schema identity differs")


def _meta_payloads(
    spec: GovernanceStoreSpec,
    plan: RuntimePlanRecord,
    authorization: SingleUseAuthorizationRecord,
    *,
    domain: str,
) -> dict[str, bytes]:
    if domain not in {"production", "internal-test"}:
        raise GovernanceStoreContractError("store domain differs")
    meta = {
        "schema_version": _STORE_META_SCHEMA,
        "domain": domain,
        "spec_identity": spec["spec_identity"],
        "plan_identity": plan["plan_identity"],
        "authorization_identity": authorization["authorization_identity"],
        "schema_ddl_identity": _SCHEMA_DDL_IDENTITY,
        "durability_boundary": "sqlite-delete-full-new-readonly-reopen-v1",
        "directory_fsync_claimed": False,
        "administrator_or_hardware_resistance_claimed": False,
    }
    return {
        "store_meta": canonical_json_bytes(meta),
        "spec": canonical_json_bytes(spec),
        "plan": canonical_json_bytes(plan),
        "authorization": canonical_json_bytes(authorization),
    }


def _verify_meta(
    connection: sqlite3.Connection,
    *,
    spec: GovernanceStoreSpec,
    plan: RuntimePlanRecord,
    authorization: SingleUseAuthorizationRecord,
    domain: str,
) -> None:
    rows = connection.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
    observed: dict[str, bytes] = {}
    for key, value in rows:
        if type(key) is not str or key in observed:
            raise GovernanceStoreContractError("SQLite meta key differs")
        observed[key] = _require_sqlite_blob(value, field=f"meta.{key}")
    expected = _meta_payloads(spec, plan, authorization, domain=domain)
    if set(observed) != set(expected):
        raise GovernanceStoreContractError("SQLite meta keys differ")
    for key, frozen in expected.items():
        parsed = _strict_json_object(observed[key], field=f"meta.{key}")
        if canonical_json_bytes(parsed) != frozen or observed[key] != frozen:
            raise GovernanceStoreContractError(f"SQLite meta.{key} differs")


def _verify_blob_tables(
    connection: sqlite3.Connection,
) -> tuple[dict[str, tuple[str, bytes]], dict[str, tuple[str, dict[str, Any]]]]:
    allowed_artifacts = {
        "source-games",
        "state-split",
        "singleton-ledger",
        "state-freeze-receipt",
    }
    artifact_identities: set[str] = set()
    artifacts: dict[str, tuple[str, bytes]] = {}
    for role, identity, payload, size, digest in connection.execute(
        "SELECT role, artifact_identity, artifact_bytes, size_bytes, sha256 "
        "FROM artifacts ORDER BY role"
    ):
        if type(role) is not str or role not in allowed_artifacts:
            raise GovernanceStoreContractError("SQLite artifact role differs")
        checked_identity = _require_sha256(identity, field=f"artifact {role} identity")
        blob = _require_sqlite_blob(payload, field=f"artifact {role}")
        if (
            checked_identity in artifact_identities
            or _require_nonnegative_int(size, field=f"artifact {role} size")
            != len(blob)
            or _require_sha256(digest, field=f"artifact {role} SHA")
            != _sha256_bytes(blob)
        ):
            raise GovernanceStoreContractError("SQLite artifact integrity differs")
        artifact_identities.add(checked_identity)
        artifacts[str(role)] = (checked_identity, blob)
    allowed_records = {
        "state-generation-completion",
        "state-freeze",
        *_RUNTIME_RECORD_TYPES,
    }
    record_identities: set[str] = set()
    records: dict[str, tuple[str, dict[str, Any]]] = {}
    rows = connection.execute(
        "SELECT stream_identity, stream_kind, sequence, record_type, "
        "previous_record_identity, record_identity, record_bytes, size_bytes, "
        "record_bytes_sha256 FROM domain_records "
        "ORDER BY stream_identity, sequence"
    ).fetchall()
    prior_by_stream: dict[str, str] = {}
    expected_sequence_by_stream: dict[str, int] = {}
    kind_by_stream: dict[str, str] = {}
    for (
        stream_identity,
        stream_kind,
        sequence,
        record_type,
        previous_record_identity,
        identity,
        payload,
        size,
        digest,
    ) in rows:
        checked_stream_identity = _require_sha256(
            stream_identity,
            field="domain stream identity",
        )
        if type(record_type) is not str or record_type not in allowed_records:
            raise GovernanceStoreContractError("unknown C5a domain record type")
        expected_kind = (
            _RUNTIME_STREAM_KIND
            if record_type in _RUNTIME_RECORD_TYPES
            else _CONTROLLER_STREAM_KIND
        )
        if stream_kind != expected_kind:
            raise GovernanceStoreContractError("domain record stream kind differs")
        prior_kind = kind_by_stream.setdefault(checked_stream_identity, stream_kind)
        if prior_kind != stream_kind:
            raise GovernanceStoreContractError("domain stream kind drifted")
        checked_sequence = _require_nonnegative_int(
            sequence,
            field=f"domain record {record_type} sequence",
        )
        expected_sequence = expected_sequence_by_stream.get(checked_stream_identity, 0)
        expected_previous = prior_by_stream.get(checked_stream_identity)
        if (
            checked_sequence != expected_sequence
            or previous_record_identity != expected_previous
        ):
            raise GovernanceStoreContractError("domain record stream chain differs")
        expected_sequence_by_stream[checked_stream_identity] = expected_sequence + 1
        checked_identity = _require_sha256(
            identity,
            field=f"domain record {record_type} identity",
        )
        blob = _require_sqlite_blob(payload, field=f"domain record {record_type}")
        parsed = _strict_json_object(blob, field=f"domain record {record_type}")
        row_domain = {key: parsed.get(key) for key in _DOMAIN_ENVELOPE_KEYS}
        expected_schema = {
            "state-generation-completion": _COMPLETION_SCHEMA,
            "state-freeze": _FREEZE_RECEIPT_SCHEMA,
            **_RUNTIME_RECORD_SCHEMAS,
        }[record_type]
        expected_domain = {
            "schema_version": expected_schema,
            "stream_identity": checked_stream_identity,
            "stream_kind": expected_kind,
            "sequence": checked_sequence,
            "record_type": record_type,
            "previous_record_identity": previous_record_identity,
        }
        if (
            checked_identity in record_identities
            or record_type in records
            or canonical_json_bytes(parsed) != blob
            or canonical_sha256(parsed) != checked_identity
            or _sha256_bytes(blob) != checked_identity
            or _require_nonnegative_int(
                size,
                field=f"domain record {record_type} size",
            )
            != len(blob)
            or _require_sha256(
                digest,
                field=f"domain record {record_type} SHA",
            )
            != checked_identity
        ):
            raise GovernanceStoreContractError("SQLite domain record integrity differs")
        _require_exact(
            row_domain,
            expected_domain,
            field=f"domain record {record_type}.domain",
        )
        record_identities.add(checked_identity)
        prior_by_stream[checked_stream_identity] = checked_identity
        records[record_type] = (checked_identity, parsed)
    if (
        len(
            {
                stream
                for stream, kind in kind_by_stream.items()
                if kind == _CONTROLLER_STREAM_KIND
            }
        )
        > 1
        or len(
            {
                stream
                for stream, kind in kind_by_stream.items()
                if kind == _RUNTIME_STREAM_KIND
            }
        )
        > 1
    ):
        raise GovernanceStoreContractError(
            "C5a supports one stream per allowlisted kind"
        )
    return artifacts, records


def _read_and_replay(
    path: Path,
    *,
    spec: GovernanceStoreSpec,
    plan: RuntimePlanRecord,
    authorization: SingleUseAuthorizationRecord,
    domain: str,
    expected_head_identity: str | None = None,
) -> GovernanceReplay:
    _check_no_sidecars(path)
    connection = _open_read_connection(path)
    try:
        result = connection.execute("PRAGMA quick_check").fetchall()
        if result != [("ok",)]:
            raise GovernanceStoreContractError("SQLite quick_check failed")
        _verify_schema(connection)
        _verify_meta(
            connection,
            spec=spec,
            plan=plan,
            authorization=authorization,
            domain=domain,
        )
        artifacts, domain_records = _verify_blob_tables(connection)
        rows = connection.execute(
            "SELECT sequence, event_identity, previous_event_identity, event_bytes, "
            "size_bytes, sha256 FROM events ORDER BY sequence"
        ).fetchall()
    except GovernanceStoreContractError:
        raise
    except sqlite3.Error as exc:
        raise GovernanceStoreContractError("SQLite verification query failed") from exc
    finally:
        connection.close()
    payloads: list[bytes] = []
    previous_identity: str | None = None
    for expected_sequence, row in enumerate(rows):
        sequence, identity, previous, payload, size, digest = row
        blob = _require_sqlite_blob(payload, field=f"event {expected_sequence}")
        if (
            sequence != expected_sequence
            or previous != previous_identity
            or _require_nonnegative_int(size, field="event size") != len(blob)
            or _require_sha256(digest, field="event SHA") != _sha256_bytes(blob)
        ):
            raise GovernanceStoreContractError("durable event row integrity differs")
        decoded = decode_governance_ledger(blob)
        if len(decoded) != 1:
            raise GovernanceStoreContractError("durable event row is not one event")
        event = decoded[0]
        if (
            event["sequence"] != expected_sequence
            or event["event_identity"] != identity
            or event["previous_event_identity"] != previous
        ):
            raise GovernanceStoreContractError("durable event row binding differs")
        previous_identity = identity
        payloads.append(blob)
    decoded_events = decode_governance_ledger(b"".join(payloads))
    replay = replay_governance_ledger(
        decoded_events,
        plan=plan,
        authorization=authorization,
    )
    if replay.head_event_identity != previous_identity:
        raise GovernanceStoreContractError("durable replay head differs")
    _verify_blob_stage_and_identities(
        replay,
        artifacts=artifacts,
        domain_records=domain_records,
        domain=domain,
        spec=spec,
    )
    if (
        expected_head_identity is not None
        and previous_identity != expected_head_identity
    ):
        raise GovernanceStoreContractError("durable replay did not reach expected head")
    _check_no_sidecars(path)
    return replay


def _create_store_database(
    path: Path,
    *,
    spec: GovernanceStoreSpec,
    plan: RuntimePlanRecord,
    authorization: SingleUseAuthorizationRecord,
    domain: str,
    bootstrap_payload: bytes,
) -> GovernanceReplay:
    decoded = decode_governance_ledger(bootstrap_payload)
    replay = replay_governance_ledger(
        decoded,
        plan=plan,
        authorization=authorization,
    )
    if replay.state != "authorized_unconsumed" or len(replay.events) != 3:
        raise GovernanceStoreContractError(
            "store bootstrap must be exact plan/readiness/authorization genesis"
        )
    connection = _open_write_connection(path)
    committed = False
    try:
        connection.execute("BEGIN IMMEDIATE")
        for statement in _DDL_STATEMENTS:
            connection.execute(statement)
        for key, payload in _meta_payloads(
            spec,
            plan,
            authorization,
            domain=domain,
        ).items():
            connection.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                (key, payload),
            )
        for event in replay.events:
            payload = encode_governance_ledger((event,))
            connection.execute(
                "INSERT INTO events(sequence, event_identity, "
                "previous_event_identity, event_bytes, size_bytes, sha256) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event["sequence"],
                    event["event_identity"],
                    event["previous_event_identity"],
                    payload,
                    len(payload),
                    _sha256_bytes(payload),
                ),
            )
        connection.execute("COMMIT")
        committed = True
    except (sqlite3.Error, GovernanceStoreContractError) as exc:
        if not committed:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        raise GovernanceStoreContractError("governance store bootstrap failed") from exc
    finally:
        connection.close()
    _check_no_sidecars(path)
    return _read_and_replay(
        path,
        spec=spec,
        plan=plan,
        authorization=authorization,
        domain=domain,
        expected_head_identity=replay.head_event_identity,
    )


def _initialize_store_core(
    output_root: str | Path,
    spec: GovernanceStoreSpec,
    *,
    plan_permit: object,
    authorization_permit: object,
    domain: str,
    claim_bootstrap: Callable[..., object],
    consume_claim: Callable[[object], Any],
    store_type: type,
    store_token: object,
    contexts: weakref.WeakKeyDictionary,
) -> object:
    checked_spec = _verified_spec(spec)
    try:
        claim = claim_bootstrap(
            plan_permit,
            authorization_permit,
            store_spec_identity=checked_spec["spec_identity"],
        )
        # The claim becomes irreversibly spent before `_database_path` can create
        # a namespace or before SQLite performs its first write.
        bootstrap = consume_claim(claim)
    except Exception as exc:
        raise GovernanceStoreContractError(
            "governance-store bootstrap claim failed"
        ) from exc
    if (
        bootstrap.store_spec_identity != checked_spec["spec_identity"]
        or bootstrap.plan["plan_identity"] != checked_spec["plan_identity"]
        or bootstrap.authorization["readiness_identity"]
        != checked_spec["readiness_identity"]
    ):
        raise GovernanceStoreContractError("store bootstrap claim/spec differs")
    root, path = _database_path(output_root, checked_spec, for_create=True)
    replay = _create_store_database(
        path,
        spec=checked_spec,
        plan=bootstrap.plan,
        authorization=bootstrap.authorization,
        domain=domain,
        bootstrap_payload=bootstrap.bootstrap_ledger_bytes,
    )
    store = store_type(store_token)
    contexts[store] = _StoreContext(
        path=path,
        output_root=root,
        spec=checked_spec,
        plan=bootstrap.plan,
        authorization=bootstrap.authorization,
        domain=domain,
        replay=replay,
        runtime=None,
    )
    return store


def _open_store_core(
    output_root: str | Path,
    spec: GovernanceStoreSpec,
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    expected_domain: str,
    store_type: type,
    store_token: object,
    contexts: weakref.WeakKeyDictionary,
) -> object:
    checked_spec = _verified_spec(spec)
    try:
        checked_plan = verify_runtime_plan(plan)
        checked_authorization = verify_single_use_authorization(
            authorization,
            plan=checked_plan,
        )
    except Exception as exc:
        raise GovernanceStoreContractError(
            "store plan/authorization is invalid"
        ) from exc
    if (
        checked_spec["plan_identity"] != checked_plan["plan_identity"]
        or checked_spec["readiness_identity"]
        != checked_authorization["readiness_identity"]
    ):
        raise GovernanceStoreContractError("store spec plan/readiness binding differs")
    root, path = _database_path(output_root, checked_spec, for_create=False)
    replay = _read_and_replay(
        path,
        spec=checked_spec,
        plan=checked_plan,
        authorization=checked_authorization,
        domain=expected_domain,
    )
    if replay.state == "state_generation_running":
        connection = _open_read_connection(path)
        try:
            artifacts, records = _verify_blob_tables(connection)
        finally:
            connection.close()
        if not artifacts and set(records) == {"state-generation-resource-before"}:
            raise _DurableRuntimeBeforePostCommitError(
                _RUNTIME_BEFORE_POSTCOMMIT_FATAL_MESSAGE
            )
        raise GovernanceStoreContractError(
            "running state-generation store cannot be reopened"
        )
    store = store_type(store_token)
    contexts[store] = _StoreContext(
        path=path,
        output_root=root,
        spec=checked_spec,
        plan=checked_plan,
        authorization=checked_authorization,
        domain=expected_domain,
        replay=replay,
        runtime=None,
    )
    return store


def _require_store_context(
    store: object,
    *,
    store_type: type,
    contexts: Mapping[object, _StoreContext],
) -> _StoreContext:
    if type(store) is not store_type:
        raise GovernanceStoreContractError("durable store type/domain differs")
    context = contexts.get(store)
    if context is None:
        raise GovernanceStoreContractError("durable store is not registered")
    reopened = _read_and_replay(
        context.path,
        spec=context.spec,
        plan=context.plan,
        authorization=context.authorization,
        domain=context.domain,
        expected_head_identity=context.replay.head_event_identity,
    )
    context.replay = reopened
    return context


def _artifact_row(role: str, identity: str, payload: bytes) -> tuple[Any, ...]:
    return (
        role,
        _require_sha256(identity, field=f"artifact {role} identity"),
        payload,
        len(payload),
        _sha256_bytes(payload),
    )


def _controller_stream_identity(
    context: _StoreContext,
    *,
    attempt_identity: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": "nmm.classical-a-pos-domain-stream.v1",
            "stream_kind": _CONTROLLER_STREAM_KIND,
            "store_spec_identity": context.spec["spec_identity"],
            "plan_identity": context.plan["plan_identity"],
            "authorization_consumption_identity": (
                context.replay.authorization_consumption_identity
            ),
            "state_generation_attempt_identity": _require_sha256(
                attempt_identity,
                field="controller stream attempt identity",
            ),
        }
    )


def _domain_envelope(
    context: _StoreContext,
    *,
    schema_version: str,
    attempt_identity: str,
    sequence: int,
    record_type: str,
    previous_record_identity: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "stream_identity": _controller_stream_identity(
            context,
            attempt_identity=attempt_identity,
        ),
        "stream_kind": _CONTROLLER_STREAM_KIND,
        "sequence": sequence,
        "record_type": record_type,
        "previous_record_identity": previous_record_identity,
    }


def _runtime_stream_identity(
    context: _StoreContext,
    *,
    attempt_identity: str,
    reservation_event_identity: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": _RUNTIME_STREAM_SCHEMA,
            "stream_kind": _RUNTIME_STREAM_KIND,
            "experiment_id": _EXPERIMENT_ID,
            "proposal_identity": _PROPOSAL_IDENTITY,
            "profile_identity": _PROFILE_IDENTITY,
            "store_spec_identity": context.spec["spec_identity"],
            "plan_identity": context.plan["plan_identity"],
            "readiness_identity": context.spec["readiness_identity"],
            "authorization_identity": context.authorization["authorization_identity"],
            "authorization_consumption_identity": (
                context.replay.authorization_consumption_identity
            ),
            "attempt_identity": attempt_identity,
            "reservation_event_identity": reservation_event_identity,
        }
    )


def _runtime_record_context(
    context: _StoreContext,
    *,
    attempt_identity: str,
    reservation_event_identity: str,
) -> dict[str, Any]:
    runtime = context.runtime
    if runtime is None:
        raise GovernanceStoreContractError("state-generation runtime is unbound")
    return {
        "experiment_id": _EXPERIMENT_ID,
        "proposal_identity": _PROPOSAL_IDENTITY,
        "profile_identity": _PROFILE_IDENTITY,
        "store_spec_identity": context.spec["spec_identity"],
        "plan_identity": context.plan["plan_identity"],
        "readiness_identity": context.spec["readiness_identity"],
        "authorization_identity": context.authorization["authorization_identity"],
        "authorization_consumption_identity": (
            context.replay.authorization_consumption_identity
        ),
        "attempt_identity": attempt_identity,
        "reservation_event_identity": reservation_event_identity,
        "state_generator_session_identity": (runtime.state_generator_session_identity),
        "host_preflight_identity": runtime.host_preflight_identity,
    }


def _runtime_snapshot_record(
    context: _StoreContext,
    *,
    attempt_identity: str,
    reservation_event_identity: str,
    sequence: int,
    record_type: str,
    previous_record_identity: str | None,
    snapshot: Mapping[str, Any],
) -> tuple[bytes, str]:
    body = {
        "schema_version": _RUNTIME_RECORD_SCHEMAS[record_type],
        "stream_identity": _runtime_stream_identity(
            context,
            attempt_identity=attempt_identity,
            reservation_event_identity=reservation_event_identity,
        ),
        "stream_kind": _RUNTIME_STREAM_KIND,
        "sequence": sequence,
        "record_type": record_type,
        "previous_record_identity": previous_record_identity,
        **_runtime_record_context(
            context,
            attempt_identity=attempt_identity,
            reservation_event_identity=reservation_event_identity,
        ),
        "snapshot": _thaw(snapshot),
        "snapshot_identity": snapshot["snapshot_identity"],
    }
    payload = canonical_json_bytes(body)
    return payload, _sha256_bytes(payload)


def _runtime_stability_record(
    context: _StoreContext,
    *,
    attempt_identity: str,
    reservation_event_identity: str,
    before_record_identity: str,
    after_record_identity: str,
    before_snapshot: Mapping[str, Any],
    after_snapshot: Mapping[str, Any],
) -> tuple[bytes, str]:
    differences = [
        path
        for path in _STABILITY_COMPARISON_FIELDS
        if type(_nested_path(before_snapshot, path))
        is not type(_nested_path(after_snapshot, path))
        or _nested_path(before_snapshot, path) != _nested_path(after_snapshot, path)
    ]
    if differences:
        raise GovernanceStoreContractError("state-generation runtime resources drifted")
    body = {
        "schema_version": _RUNTIME_RECORD_SCHEMAS[
            "state-generation-resource-stability"
        ],
        "stream_identity": _runtime_stream_identity(
            context,
            attempt_identity=attempt_identity,
            reservation_event_identity=reservation_event_identity,
        ),
        "stream_kind": _RUNTIME_STREAM_KIND,
        "sequence": 2,
        "record_type": "state-generation-resource-stability",
        "previous_record_identity": after_record_identity,
        **_runtime_record_context(
            context,
            attempt_identity=attempt_identity,
            reservation_event_identity=reservation_event_identity,
        ),
        "before_record_identity": before_record_identity,
        "after_record_identity": after_record_identity,
        "before_snapshot_identity": before_snapshot["snapshot_identity"],
        "after_snapshot_identity": after_snapshot["snapshot_identity"],
        "comparison_fields": list(_STABILITY_COMPARISON_FIELDS),
        "differences": [],
        "stable": True,
    }
    payload = canonical_json_bytes(body)
    return payload, _sha256_bytes(payload)


def _artifact_contract_ref(
    role: str,
    identity: str,
    payload: bytes,
    *,
    record_count: int,
    a_pos_verifier_identity: str | None = None,
) -> dict[str, Any]:
    ref = {
        "role": role,
        "identity": _require_sha256(identity, field=f"artifact {role} identity"),
        "bytes_sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "record_count": _require_nonnegative_int(
            record_count,
            field=f"artifact {role} record count",
        ),
    }
    if role == "state-split":
        ref["a_pos_verifier_identity"] = _require_sha256(
            a_pos_verifier_identity,
            field="state-split verifier identity",
        )
    elif a_pos_verifier_identity is not None:
        raise GovernanceStoreContractError("non-state artifact has verifier identity")
    return ref


def _domain_record_row(payload: bytes) -> tuple[Any, ...]:
    parsed = _strict_json_object(payload, field="domain record")
    domain = _require_exact_keys(
        {key: parsed.get(key) for key in _DOMAIN_ENVELOPE_KEYS},
        _DOMAIN_ENVELOPE_KEYS,
        field="domain record envelope",
    )
    record_type = _require_nonempty_text(
        domain["record_type"], field="domain record type"
    )
    expected_schema = {
        "state-generation-completion": _COMPLETION_SCHEMA,
        "state-freeze": _FREEZE_RECEIPT_SCHEMA,
        **_RUNTIME_RECORD_SCHEMAS,
    }.get(record_type)
    if expected_schema is None or domain["schema_version"] != expected_schema:
        raise GovernanceStoreContractError("domain record schema/type differs")
    expected_kind = (
        _RUNTIME_STREAM_KIND
        if record_type in _RUNTIME_RECORD_TYPES
        else _CONTROLLER_STREAM_KIND
    )
    _require_exact(
        domain["stream_kind"], expected_kind, field="domain record stream kind"
    )
    checked_identity = canonical_sha256(parsed)
    if _sha256_bytes(payload) != checked_identity:
        raise GovernanceStoreContractError("domain record canonical bytes differ")
    return (
        _require_sha256(domain["stream_identity"], field="domain stream identity"),
        domain["stream_kind"],
        _require_nonnegative_int(domain["sequence"], field="domain sequence"),
        record_type,
        domain["previous_record_identity"],
        checked_identity,
        payload,
        len(payload),
        checked_identity,
    )


def _commit_event_and_rows(
    context: _StoreContext,
    event_payload: bytes,
    *,
    artifact_rows: Sequence[tuple[Any, ...]] = (),
    domain_rows: Sequence[tuple[Any, ...]] = (),
) -> GovernanceReplay:
    current = _read_and_replay(
        context.path,
        spec=context.spec,
        plan=context.plan,
        authorization=context.authorization,
        domain=context.domain,
        expected_head_identity=context.replay.head_event_identity,
    )
    decoded = decode_governance_ledger(event_payload)
    if len(decoded) != 1:
        raise GovernanceStoreContractError("commit requires exactly one event")
    event = decoded[0]
    if (
        event["sequence"] != len(current.events)
        or event["previous_event_identity"] != current.head_event_identity
    ):
        raise GovernanceStoreContractError("event does not extend durable replay head")
    replay_governance_ledger(
        (*current.events, event),
        plan=context.plan,
        authorization=context.authorization,
    )
    connection = _open_write_connection(context.path)
    committed = False
    try:
        connection.execute("BEGIN IMMEDIATE")
        _verify_schema(connection)
        _verify_meta(
            connection,
            spec=context.spec,
            plan=context.plan,
            authorization=context.authorization,
            domain=context.domain,
        )
        head_row = connection.execute(
            "SELECT sequence, event_identity FROM events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        expected_head = (
            None
            if not current.events
            else (len(current.events) - 1, current.head_event_identity)
        )
        if head_row != expected_head:
            raise GovernanceStoreContractError(
                "writer durable head changed before BEGIN IMMEDIATE"
            )
        for row in artifact_rows:
            connection.execute(
                "INSERT INTO artifacts(role, artifact_identity, artifact_bytes, "
                "size_bytes, sha256) VALUES (?, ?, ?, ?, ?)",
                row,
            )
        for row in domain_rows:
            connection.execute(
                "INSERT INTO domain_records(stream_identity, stream_kind, sequence, "
                "record_type, previous_record_identity, record_identity, "
                "record_bytes, size_bytes, record_bytes_sha256) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        connection.execute(
            "INSERT INTO events(sequence, event_identity, previous_event_identity, "
            "event_bytes, size_bytes, sha256) VALUES (?, ?, ?, ?, ?, ?)",
            (
                event["sequence"],
                event["event_identity"],
                event["previous_event_identity"],
                event_payload,
                len(event_payload),
                _sha256_bytes(event_payload),
            ),
        )
        connection.execute("COMMIT")
        committed = True
    except (sqlite3.Error, GovernanceStoreContractError) as exc:
        if not committed:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        raise GovernanceStoreContractError("durable governance commit failed") from exc
    finally:
        connection.close()
    _check_no_sidecars(context.path)
    replay = _read_and_replay(
        context.path,
        spec=context.spec,
        plan=context.plan,
        authorization=context.authorization,
        domain=context.domain,
        expected_head_identity=event["event_identity"],
    )
    context.replay = replay
    return replay


def _commit_domain_rows_only(
    context: _StoreContext,
    rows: Sequence[tuple[Any, ...]],
) -> GovernanceReplay:
    current = _read_and_replay(
        context.path,
        spec=context.spec,
        plan=context.plan,
        authorization=context.authorization,
        domain=context.domain,
        expected_head_identity=context.replay.head_event_identity,
    )
    connection = _open_write_connection(context.path)
    failure: Exception | None = None
    try:
        connection.execute("BEGIN IMMEDIATE")
        _verify_schema(connection)
        _verify_meta(
            connection,
            spec=context.spec,
            plan=context.plan,
            authorization=context.authorization,
            domain=context.domain,
        )
        head = connection.execute(
            "SELECT sequence, event_identity FROM events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        expected_head = (
            None
            if not current.events
            else (len(current.events) - 1, current.head_event_identity)
        )
        if head != expected_head:
            raise GovernanceStoreContractError("durable event head changed")
        for row in rows:
            connection.execute(
                "INSERT INTO domain_records(stream_identity, stream_kind, sequence, "
                "record_type, previous_record_identity, record_identity, "
                "record_bytes, size_bytes, record_bytes_sha256) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        artifacts, domain_records = _verify_blob_tables(connection)
        _verify_blob_stage_and_identities(
            current,
            artifacts=artifacts,
            domain_records=domain_records,
            domain=context.domain,
            spec=context.spec,
        )
        for expected_row in rows:
            observed_row = connection.execute(
                "SELECT stream_identity, stream_kind, sequence, record_type, "
                "previous_record_identity, record_identity, record_bytes, "
                "size_bytes, record_bytes_sha256 FROM domain_records "
                "WHERE stream_identity=? AND sequence=?",
                (expected_row[0], expected_row[2]),
            ).fetchone()
            if observed_row != tuple(expected_row):
                raise GovernanceStoreContractError(
                    "durable runtime evidence row differs before COMMIT"
                )
        connection.execute("COMMIT")
    except Exception as exc:
        failure = exc
    if failure is not None:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        connection.close()
        raise GovernanceStoreContractError(
            "durable runtime evidence commit failed before COMMIT"
        ) from failure
    try:
        connection.close()
    except Exception as exc:
        raise _DurableRuntimeBeforePostCommitError(
            _RUNTIME_BEFORE_POSTCOMMIT_FATAL_MESSAGE
        ) from exc
    try:
        replay = _read_and_replay(
            context.path,
            spec=context.spec,
            plan=context.plan,
            authorization=context.authorization,
            domain=context.domain,
            expected_head_identity=current.head_event_identity,
        )
    except Exception as exc:
        raise _DurableRuntimeBeforePostCommitError(
            _RUNTIME_BEFORE_POSTCOMMIT_FATAL_MESSAGE
        ) from exc
    return replay


def commit_authorization_consumption(
    store: DurableGovernanceStore,
    pending: PendingAuthorizationConsumption,
) -> ConsumedAuthorizationPermit:
    """Persist and strict-replay one production authorization consumption."""
    if (
        type(store) is not DurableGovernanceStore
        or type(pending) is not PendingAuthorizationConsumption
    ):
        raise GovernanceStoreContractError(
            "authorization durable commit requires exact production capabilities"
        )
    if pending in _PRODUCTION_DURABLE_PENDING_ATTEMPTS:
        raise GovernanceStoreContractError(
            "authorization durable commit was already attempted"
        )
    _PRODUCTION_DURABLE_PENDING_ATTEMPTS.add(pending)
    context = _require_store_context(
        store,
        store_type=DurableGovernanceStore,
        contexts=_PRODUCTION_STORE_CONTEXTS,
    )
    try:
        payload = prepared_governance_event_bytes(pending)
        replay = _commit_event_and_rows(context, payload)
        return confirm_authorization_consumption(pending, replay)
    except Exception as exc:
        if isinstance(exc, GovernanceStoreContractError):
            raise
        raise GovernanceStoreContractError(
            "authorization consumption durable confirmation failed"
        ) from exc


def commit_operation_reservation(
    store: DurableGovernanceStore,
    pending: PendingOperationReservation,
) -> ProductionOperationPermit:
    """Persist and strict-replay one production operation reservation."""
    if (
        type(store) is not DurableGovernanceStore
        or type(pending) is not PendingOperationReservation
    ):
        raise GovernanceStoreContractError(
            "operation durable commit requires exact production capabilities"
        )
    if pending in _PRODUCTION_DURABLE_PENDING_ATTEMPTS:
        raise GovernanceStoreContractError(
            "operation durable commit was already attempted"
        )
    _PRODUCTION_DURABLE_PENDING_ATTEMPTS.add(pending)
    context = _require_store_context(
        store,
        store_type=DurableGovernanceStore,
        contexts=_PRODUCTION_STORE_CONTEXTS,
    )
    try:
        payload = prepared_governance_event_bytes(pending)
        replay = _commit_event_and_rows(context, payload)
        permit = confirm_operation_reservation(pending, replay)
        _PRODUCTION_OPERATION_STORE_BINDINGS[permit] = store
        return permit
    except Exception as exc:
        if isinstance(exc, GovernanceStoreContractError):
            raise
        raise GovernanceStoreContractError(
            "operation reservation durable confirmation failed"
        ) from exc


def _test_initialize_governance_store(
    output_root: str | Path,
    spec: GovernanceStoreSpec,
    *,
    plan_permit: object,
    authorization_permit: object,
) -> _TestDurableGovernanceStore:
    store = _initialize_store_core(
        output_root,
        spec,
        plan_permit=plan_permit,
        authorization_permit=authorization_permit,
        domain="internal-test",
        claim_bootstrap=_governance._test_claim_store_bootstrap,
        consume_claim=_governance._test_consume_store_bootstrap_claim,
        store_type=_TestDurableGovernanceStore,
        store_token=_TEST_STORE_TOKEN,
        contexts=_TEST_STORE_CONTEXTS,
    )
    assert type(store) is _TestDurableGovernanceStore
    return store


def _test_open_governance_store(
    output_root: str | Path,
    spec: GovernanceStoreSpec,
    *,
    plan_permit: object,
    authorization_permit: object,
) -> _TestDurableGovernanceStore:
    checked_spec = _verified_spec(spec)
    try:
        opened = _governance._test_inspect_store_open_context(
            plan_permit,
            authorization_permit,
            store_spec_identity=checked_spec["spec_identity"],
        )
    except Exception as exc:
        raise GovernanceStoreContractError("test store open permits differ") from exc
    store = _open_store_core(
        output_root,
        checked_spec,
        plan=opened.plan,
        authorization=opened.authorization,
        expected_domain="internal-test",
        store_type=_TestDurableGovernanceStore,
        store_token=_TEST_STORE_TOKEN,
        contexts=_TEST_STORE_CONTEXTS,
    )
    assert type(store) is _TestDurableGovernanceStore
    return store


def _test_commit_authorization_consumption(
    store: _TestDurableGovernanceStore,
    pending: object,
) -> object:
    if pending in _TEST_DURABLE_PENDING_ATTEMPTS:
        raise GovernanceStoreContractError(
            "test authorization durable commit was already attempted"
        )
    _TEST_DURABLE_PENDING_ATTEMPTS.add(pending)
    context = _require_store_context(
        store,
        store_type=_TestDurableGovernanceStore,
        contexts=_TEST_STORE_CONTEXTS,
    )
    try:
        payload = _governance._test_prepared_governance_event_bytes(pending)
        replay = _commit_event_and_rows(context, payload)
        return _governance._test_confirm_authorization_consumption(pending, replay)
    except Exception as exc:
        if isinstance(exc, GovernanceStoreContractError):
            raise
        raise GovernanceStoreContractError(
            "test authorization durable confirmation failed"
        ) from exc


def _test_commit_operation_reservation(
    store: _TestDurableGovernanceStore,
    pending: object,
) -> object:
    if pending in _TEST_DURABLE_PENDING_ATTEMPTS:
        raise GovernanceStoreContractError(
            "test operation durable commit was already attempted"
        )
    _TEST_DURABLE_PENDING_ATTEMPTS.add(pending)
    context = _require_store_context(
        store,
        store_type=_TestDurableGovernanceStore,
        contexts=_TEST_STORE_CONTEXTS,
    )
    try:
        payload = _governance._test_prepared_governance_event_bytes(pending)
        replay = _commit_event_and_rows(context, payload)
        permit = _governance._test_confirm_operation_reservation(pending, replay)
        _TEST_OPERATION_STORE_BINDINGS[permit] = store
        return permit
    except Exception as exc:
        if isinstance(exc, GovernanceStoreContractError):
            raise
        raise GovernanceStoreContractError(
            "test operation durable confirmation failed"
        ) from exc


def _begin_state_generation_core(
    store: object,
    permit: object,
    *,
    store_type: type,
    permit_type: type,
    active_type: type,
    active_token: object,
    store_contexts: Mapping[object, _StoreContext],
    active_contexts: weakref.WeakKeyDictionary,
    permit_consumer: Callable[..., str],
    operation_store_bindings: Mapping[object, object],
) -> object:
    context = _require_store_context(
        store,
        store_type=store_type,
        contexts=store_contexts,
    )
    if type(permit) is not permit_type:
        raise GovernanceStoreContractError(
            "state-generation begin requires an exact operation permit"
        )
    if operation_store_bindings.get(permit) is not store:
        raise GovernanceStoreContractError(
            "state-generation operation permit is bound to another store"
        )
    if (
        context.replay.state != "state_generation_running"
        or context.replay.resource_ledger.open_operation_id != "state-generation"
        or not context.replay.events
        or context.replay.events[-1]["event_type"] != "state_generation_reserved"
    ):
        raise GovernanceStoreContractError(
            "state-generation reservation is not the durable replay head"
        )
    try:
        attempt_identity = permit_consumer(
            permit,
            context.replay,
            operation_id="state-generation",
            purpose="state",
            seed=None,
        )
    except Exception as exc:
        raise GovernanceStoreContractError(
            "state-generation operation permit could not be consumed"
        ) from exc
    checked_attempt_identity = _require_sha256(
        attempt_identity,
        field="state-generation attempt identity",
    )
    reservation_identity = context.replay.head_event_identity
    assert reservation_identity is not None
    active = active_type(active_token)
    active_contexts[active] = _ActiveContext(
        store=store,
        attempt_identity=checked_attempt_identity,
        reservation_event_identity=reservation_identity,
    )
    return active


def begin_state_generation_attempt(
    store: DurableGovernanceStore,
    permit: ProductionOperationPermit,
) -> ActiveStateGenerationAttempt:
    """Consume one durable production state-generation operation permit."""
    active = _begin_state_generation_core(
        store,
        permit,
        store_type=DurableGovernanceStore,
        permit_type=ProductionOperationPermit,
        active_type=ActiveStateGenerationAttempt,
        active_token=_PRODUCTION_ACTIVE_TOKEN,
        store_contexts=_PRODUCTION_STORE_CONTEXTS,
        active_contexts=_PRODUCTION_ACTIVE_CONTEXTS,
        permit_consumer=require_production_operation_permit,
        operation_store_bindings=_PRODUCTION_OPERATION_STORE_BINDINGS,
    )
    assert type(active) is ActiveStateGenerationAttempt
    return active


def _test_begin_state_generation_attempt(
    store: _TestDurableGovernanceStore,
    permit: object,
) -> _TestActiveStateGenerationAttempt:
    active = _begin_state_generation_core(
        store,
        permit,
        store_type=_TestDurableGovernanceStore,
        permit_type=_governance._TestOperationPermit,
        active_type=_TestActiveStateGenerationAttempt,
        active_token=_TEST_ACTIVE_TOKEN,
        store_contexts=_TEST_STORE_CONTEXTS,
        active_contexts=_TEST_ACTIVE_CONTEXTS,
        permit_consumer=_governance._test_require_operation_permit,
        operation_store_bindings=_TEST_OPERATION_STORE_BINDINGS,
    )
    assert type(active) is _TestActiveStateGenerationAttempt
    return active


def _issue_test_state_generation_runtime_preflight(
    store: _TestDurableGovernanceStore,
    active: _TestActiveStateGenerationAttempt,
    *,
    resource_snapshot_before: Mapping[str, Any] | None = None,
) -> _TestStateGenerationRuntimePreflight:
    if (
        type(store) is not _TestDurableGovernanceStore
        or type(active) is not _TestActiveStateGenerationAttempt
    ):
        raise GovernanceStoreContractError(
            "test runtime preflight requires exact store/active capabilities"
        )
    active_context = _TEST_ACTIVE_CONTEXTS.get(active)
    if (
        active_context is None
        or active_context.store is not store
        or active_context.bind_consumed
        or active_context.spent
        or _TEST_PREFLIGHT_BY_ACTIVE.get(active) is not None
    ):
        raise GovernanceStoreContractError(
            "test runtime preflight active attempt is unavailable"
        )
    store_context = _require_store_context(
        store,
        store_type=_TestDurableGovernanceStore,
        contexts=_TEST_STORE_CONTEXTS,
    )
    if store_context.runtime is not None:
        raise GovernanceStoreContractError("test store runtime is already bound")
    before = _validate_resource_snapshot(
        (
            _test_resource_snapshot(
                store_context.spec,
                attempt_identity=active_context.attempt_identity,
                stage="before-execution",
                observed_at_utc=store_context.replay.events[-1]["timestamp_utc"],
            )
            if resource_snapshot_before is None
            else resource_snapshot_before
        ),
        expected_stage="before-execution",
        spec=store_context.spec,
        attempt_identity=active_context.attempt_identity,
    )
    host_identity = canonical_sha256(before["host"])
    session_identity = _require_test_sha(
        "state-generator-session",
        spec_identity=store_context.spec["spec_identity"],
    )
    preflight_body = {
        "schema_version": "nmm.classical-a-pos-state-generation-runtime-preflight.v1",
        "domain": "internal-test",
        "store_spec_identity": store_context.spec["spec_identity"],
        "plan_identity": store_context.plan["plan_identity"],
        "readiness_identity": store_context.spec["readiness_identity"],
        "attempt_identity": active_context.attempt_identity,
        "reservation_event_identity": active_context.reservation_event_identity,
        "host_preflight_identity": host_identity,
        "state_generator_session_identity": session_identity,
        "resource_snapshot_before_identity": before["snapshot_identity"],
    }
    preflight = _TestStateGenerationRuntimePreflight(_TEST_RUNTIME_PREFLIGHT_TOKEN)
    _TEST_RUNTIME_PREFLIGHT_CONTEXTS[preflight] = _RuntimePreflightContext(
        store=store,
        active=active,
        domain="internal-test",
        store_spec_identity=store_context.spec["spec_identity"],
        plan_identity=store_context.plan["plan_identity"],
        readiness_identity=store_context.spec["readiness_identity"],
        attempt_identity=active_context.attempt_identity,
        reservation_event_identity=active_context.reservation_event_identity,
        host_preflight_identity=host_identity,
        state_generator_session_identity=session_identity,
        resource_snapshot_before=_freeze(before, field="runtime preflight snapshot"),
        resource_snapshot_before_identity=before["snapshot_identity"],
        preflight_identity=canonical_sha256(preflight_body),
    )
    _TEST_PREFLIGHT_BY_ACTIVE[active] = preflight
    return preflight


def _bind_state_generation_attempt_core(
    preflight: object,
    store: object,
    active: object,
    *,
    preflight_type: type,
    store_type: type,
    active_type: type,
    bound_type: type,
    bound_token: object,
    preflight_contexts: weakref.WeakKeyDictionary,
    store_contexts: weakref.WeakKeyDictionary,
    active_contexts: weakref.WeakKeyDictionary,
    bound_contexts: weakref.WeakKeyDictionary,
) -> object:
    if (
        type(preflight) is not preflight_type
        or type(store) is not store_type
        or type(active) is not active_type
    ):
        raise GovernanceStoreContractError(
            "runtime bind requires exact preflight/store/active capabilities"
        )
    preflight_context = preflight_contexts.get(preflight)
    active_context = active_contexts.get(active)
    if (
        preflight_context is None
        or active_context is None
        or preflight_context.spent
        or active_context.bind_consumed
        or active_context.spent
        or preflight_context.store is not store
        or preflight_context.active is not active
        or active_context.store is not store
        or preflight_context.attempt_identity != active_context.attempt_identity
        or preflight_context.reservation_event_identity
        != active_context.reservation_event_identity
    ):
        raise GovernanceStoreContractError("runtime bind capability binding differs")
    # Both roots are one-shot before any durable read or write attempt.
    preflight_context.spent = True
    active_context.bind_consumed = True
    store_context = _require_store_context(
        store,
        store_type=store_type,
        contexts=store_contexts,
    )
    if (
        store_context.runtime is not None
        or store_context.domain != preflight_context.domain
        or store_context.spec["spec_identity"] != preflight_context.store_spec_identity
        or store_context.plan["plan_identity"] != preflight_context.plan_identity
        or store_context.spec["readiness_identity"]
        != preflight_context.readiness_identity
        or store_context.replay.state != "state_generation_running"
    ):
        raise GovernanceStoreContractError("runtime bind store context differs")
    before = _validate_resource_snapshot(
        _thaw(preflight_context.resource_snapshot_before),
        expected_stage="before-execution",
        spec=store_context.spec,
        attempt_identity=active_context.attempt_identity,
    )
    expected_host_identity = canonical_sha256(before["host"])
    checked_host_identity = _require_bound_identity(
        preflight_context.host_preflight_identity,
        field="runtime preflight host identity",
    )
    checked_session_identity = _require_bound_identity(
        preflight_context.state_generator_session_identity,
        field="runtime preflight generator session identity",
    )
    expected_preflight_identity = canonical_sha256(
        {
            "schema_version": (
                "nmm.classical-a-pos-state-generation-runtime-preflight.v1"
            ),
            "domain": preflight_context.domain,
            "store_spec_identity": store_context.spec["spec_identity"],
            "plan_identity": store_context.plan["plan_identity"],
            "readiness_identity": store_context.spec["readiness_identity"],
            "attempt_identity": active_context.attempt_identity,
            "reservation_event_identity": active_context.reservation_event_identity,
            "host_preflight_identity": expected_host_identity,
            "state_generator_session_identity": checked_session_identity,
            "resource_snapshot_before_identity": before["snapshot_identity"],
        }
    )
    if (
        checked_host_identity != expected_host_identity
        or preflight_context.resource_snapshot_before_identity
        != before["snapshot_identity"]
        or preflight_context.preflight_identity != expected_preflight_identity
    ):
        raise GovernanceStoreContractError("runtime preflight identity binding differs")
    runtime = _StoreRuntimeContext(
        host_preflight_identity=checked_host_identity,
        state_generator_session_identity=checked_session_identity,
    )
    prospective_context = _StoreContext(
        path=store_context.path,
        output_root=store_context.output_root,
        spec=store_context.spec,
        plan=store_context.plan,
        authorization=store_context.authorization,
        domain=store_context.domain,
        replay=store_context.replay,
        runtime=runtime,
    )
    before_bytes, before_identity = _runtime_snapshot_record(
        prospective_context,
        attempt_identity=active_context.attempt_identity,
        reservation_event_identity=active_context.reservation_event_identity,
        sequence=0,
        record_type="state-generation-resource-before",
        previous_record_identity=None,
        snapshot=before,
    )
    bound = bound_type(bound_token)
    bound_context = _BoundAttemptContext(
        store=store,
        active=active,
        domain=preflight_context.domain,
        preflight_identity=preflight_context.preflight_identity,
        host_preflight_identity=preflight_context.host_preflight_identity,
        state_generator_session_identity=(
            preflight_context.state_generator_session_identity
        ),
        runtime_stream_identity=_runtime_stream_identity(
            prospective_context,
            attempt_identity=active_context.attempt_identity,
            reservation_event_identity=active_context.reservation_event_identity,
        ),
        resource_snapshot_before=_freeze(before, field="bound runtime before snapshot"),
        resource_snapshot_before_identity=(
            preflight_context.resource_snapshot_before_identity
        ),
        before_record_bytes=before_bytes,
        before_record_identity=before_identity,
    )
    try:
        replay = _commit_domain_rows_only(
            prospective_context,
            (_domain_record_row(before_bytes),),
        )
    except _DurableRuntimeBeforePostCommitError:
        store_context.runtime = None
        active_context.spent = True
        raise
    try:
        store_context.replay = replay
        store_context.runtime = runtime
        bound_contexts[bound] = bound_context
    except Exception as exc:
        try:
            bound_contexts.pop(bound, None)
        except Exception:
            pass
        store_context.runtime = None
        active_context.spent = True
        raise _DurableRuntimeBeforePostCommitError(
            _RUNTIME_BEFORE_POSTCOMMIT_FATAL_MESSAGE
        ) from exc
    return bound


def _test_bind_state_generation_attempt(
    preflight: _TestStateGenerationRuntimePreflight,
    store: _TestDurableGovernanceStore,
    active: _TestActiveStateGenerationAttempt,
) -> _TestRuntimeBoundStateGenerationAttempt:
    bound = _bind_state_generation_attempt_core(
        preflight,
        store,
        active,
        preflight_type=_TestStateGenerationRuntimePreflight,
        store_type=_TestDurableGovernanceStore,
        active_type=_TestActiveStateGenerationAttempt,
        bound_type=_TestRuntimeBoundStateGenerationAttempt,
        bound_token=_TEST_BOUND_ATTEMPT_TOKEN,
        preflight_contexts=_TEST_RUNTIME_PREFLIGHT_CONTEXTS,
        store_contexts=_TEST_STORE_CONTEXTS,
        active_contexts=_TEST_ACTIVE_CONTEXTS,
        bound_contexts=_TEST_BOUND_ATTEMPT_CONTEXTS,
    )
    assert type(bound) is _TestRuntimeBoundStateGenerationAttempt
    return bound


def _issue_test_state_generation_runtime_completion(
    bound_attempt: _TestRuntimeBoundStateGenerationAttempt,
    *,
    resource_snapshot_after: Mapping[str, Any] | None = None,
) -> _TestStateGenerationRuntimeCompletion:
    if type(bound_attempt) is not _TestRuntimeBoundStateGenerationAttempt:
        raise GovernanceStoreContractError(
            "test runtime completion requires an exact bound attempt"
        )
    bound_context = _TEST_BOUND_ATTEMPT_CONTEXTS.get(bound_attempt)
    if (
        bound_context is None
        or bound_context.runtime_completion_issued
        or bound_context.spent
    ):
        raise GovernanceStoreContractError(
            "test runtime completion bound attempt is unavailable"
        )
    active_context = _TEST_ACTIVE_CONTEXTS.get(bound_context.active)
    if active_context is None:
        raise GovernanceStoreContractError(
            "test runtime completion active root differs"
        )
    store_context = _require_store_context(
        bound_context.store,
        store_type=_TestDurableGovernanceStore,
        contexts=_TEST_STORE_CONTEXTS,
    )
    runtime = store_context.runtime
    if (
        runtime is None
        or runtime.host_preflight_identity != bound_context.host_preflight_identity
        or runtime.state_generator_session_identity
        != bound_context.state_generator_session_identity
    ):
        raise GovernanceStoreContractError("test runtime completion binding differs")
    before = _validate_resource_snapshot(
        _thaw(bound_context.resource_snapshot_before),
        expected_stage="before-execution",
        spec=store_context.spec,
        attempt_identity=active_context.attempt_identity,
    )
    if resource_snapshot_after is None:
        after_candidate = _thaw(before)
        after_candidate["stage"] = "after-execution"
        after_candidate["observed_at_utc"] = "2026-09-01T00:00:05Z"
        after_candidate["generator"]["current_rng_state_identity"] = _require_test_sha(
            "after-rng",
            spec_identity=store_context.spec["spec_identity"],
        )
        after_body = {
            key: item
            for key, item in after_candidate.items()
            if key != "snapshot_identity"
        }
        after_candidate["snapshot_identity"] = canonical_sha256(after_body)
    else:
        after_candidate = resource_snapshot_after
    after = _validate_resource_snapshot(
        after_candidate,
        expected_stage="after-execution",
        spec=store_context.spec,
        attempt_identity=active_context.attempt_identity,
    )
    after_bytes, after_identity = _runtime_snapshot_record(
        store_context,
        attempt_identity=active_context.attempt_identity,
        reservation_event_identity=active_context.reservation_event_identity,
        sequence=1,
        record_type="state-generation-resource-after",
        previous_record_identity=bound_context.before_record_identity,
        snapshot=after,
    )
    stability_bytes, stability_identity = _runtime_stability_record(
        store_context,
        attempt_identity=active_context.attempt_identity,
        reservation_event_identity=active_context.reservation_event_identity,
        before_record_identity=bound_context.before_record_identity,
        after_record_identity=after_identity,
        before_snapshot=before,
        after_snapshot=after,
    )
    completion = _TestStateGenerationRuntimeCompletion(_TEST_RUNTIME_COMPLETION_TOKEN)
    _TEST_RUNTIME_COMPLETION_CONTEXTS[completion] = _RuntimeCompletionContext(
        store=bound_context.store,
        bound_attempt=bound_attempt,
        runtime_stream_identity=bound_context.runtime_stream_identity,
        resource_snapshot_before=_freeze(before, field="runtime completion before"),
        resource_snapshot_after=_freeze(after, field="runtime completion after"),
        before_record_identity=bound_context.before_record_identity,
        after_record_bytes=after_bytes,
        after_record_identity=after_identity,
        stability_record_bytes=stability_bytes,
        stability_record_identity=stability_identity,
    )
    bound_context.runtime_completion_issued = True
    return completion


def _test_artifact_policy(
    layout: Mapping[str, Mapping[str, Mapping[str, int]]],
    *,
    maximum_source_games: int,
) -> _ArtifactPolicy:
    splits = {"train", "dev"}
    strata = {"placement", "movement", "flying"}
    colours = {"W", "B"}
    if not isinstance(layout, Mapping) or set(layout) != splits:
        raise GovernanceStoreContractError("test artifact layout split keys differ")
    copied: dict[str, dict[str, dict[str, int]]] = {}
    total = 0
    for split in ("train", "dev"):
        if not isinstance(layout[split], Mapping) or set(layout[split]) != strata:
            raise GovernanceStoreContractError("test artifact layout strata differ")
        copied[split] = {}
        for stratum in ("placement", "movement", "flying"):
            if (
                not isinstance(layout[split][stratum], Mapping)
                or set(layout[split][stratum]) != colours
            ):
                raise GovernanceStoreContractError(
                    "test artifact layout colours differ"
                )
            copied[split][stratum] = {}
            for colour in ("W", "B"):
                count = _require_nonnegative_int(
                    layout[split][stratum][colour],
                    field="test artifact cell count",
                )
                copied[split][stratum][colour] = count
                total += count
    maximum = _require_nonnegative_int(
        maximum_source_games,
        field="test maximum source games",
    )
    if maximum == 0:
        raise GovernanceStoreContractError("test maximum source games must be positive")
    return _ArtifactPolicy(
        layout=_freeze(copied, field="test artifact layout"),
        total_states=total,
        maximum_source_games=maximum,
        require_complete_game_block=False,
        domain="internal-test",
    )


def _issue_test_a_pos_inventory_binding(
    *,
    verifier_identity: str,
    inventory_verifier: Callable[
        [BoardState, Sequence[Mapping[str, Any]]], Sequence[bool]
    ],
    layout: Mapping[str, Mapping[str, Mapping[str, int]]],
    maximum_source_games: int,
) -> _TestAPosInventoryBinding:
    if not callable(inventory_verifier):
        raise GovernanceStoreContractError("test inventory verifier is not callable")
    policy = _test_artifact_policy(
        layout,
        maximum_source_games=maximum_source_games,
    )
    checked_verifier_identity = _require_bound_identity(
        verifier_identity,
        field="test inventory verifier identity",
    )
    binding = _TestAPosInventoryBinding(_TEST_INVENTORY_TOKEN)
    _TEST_INVENTORY_CONTEXTS[binding] = _InventoryContext(
        binding_identity=_inventory_binding_identity(
            checked_verifier_identity,
            policy,
        ),
        verifier_identity=checked_verifier_identity,
        inventory_verifier=inventory_verifier,
        policy=policy,
    )
    return binding


def _inventory_binding_identity(
    verifier_identity: str,
    policy: _ArtifactPolicy,
) -> str:
    return canonical_sha256(
        {
            "schema_version": "nmm.classical-a-pos-inventory-binding.v1",
            "domain": policy.domain,
            "verifier_identity": _require_sha256(
                verifier_identity,
                field="inventory verifier identity",
            ),
            "policy": _policy_payload(policy),
        }
    )


def _issue_test_strict_referee_binding(
    *,
    binding_identity: str,
    runtime_identity: str,
    attempt_identity: str,
    complete_history_verifier: Callable[
        [Sequence[Mapping[str, Any]], Sequence[str]], UciPositionState
    ],
    prefix_history_verifier: Callable[[Sequence[Mapping[str, Any]]], str],
) -> _TestStrictRefereeBinding:
    if not callable(complete_history_verifier) or not callable(prefix_history_verifier):
        raise GovernanceStoreContractError("test strict referee verifier differs")
    binding = _TestStrictRefereeBinding(_TEST_STRICT_REFEREE_TOKEN)
    _TEST_STRICT_REFEREE_CONTEXTS[binding] = _StrictRefereeContext(
        binding_identity=_require_bound_identity(
            binding_identity,
            field="test strict referee binding identity",
        ),
        runtime_identity=_require_bound_identity(
            runtime_identity,
            field="test strict referee runtime identity",
        ),
        attempt_identity=_require_bound_identity(
            attempt_identity,
            field="test strict referee attempt identity",
        ),
        complete_history_verifier=complete_history_verifier,
        prefix_history_verifier=prefix_history_verifier,
        domain="internal-test",
    )
    return binding


def _atomic_action(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"from", "to", "capture"}:
        raise GovernanceStoreContractError(
            f"{field} must be one exact atomic from/to/capture action"
        )
    source = value["from"]
    target = value["to"]
    capture = value["capture"]
    if source is not None and not isinstance(source, str):
        raise GovernanceStoreContractError(f"{field}.from differs")
    if not isinstance(target, str) or not target:
        raise GovernanceStoreContractError(f"{field}.to differs")
    if capture is not None and not isinstance(capture, str):
        raise GovernanceStoreContractError(f"{field}.capture differs")
    return {"from": source, "to": target, "capture": capture}


def _action_key(value: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return value["from"], value["to"], value["capture"]


def _replay_history(
    value: Any,
    *,
    field: str,
) -> tuple[tuple[dict[str, Any], ...], BoardState]:
    if not isinstance(value, list):
        raise GovernanceStoreContractError(f"{field} must be an array")
    history = tuple(
        _atomic_action(move, field=f"{field}[{index}]")
        for index, move in enumerate(value)
    )
    board = BoardState.new_game()
    for index, move in enumerate(history):
        already_terminal, _winner, _reason = terminal_result(board)
        if already_terminal:
            raise GovernanceStoreContractError(
                f"{field}[{index}] continues after local terminal state"
            )
        legal = tuple(dict(item) for item in get_all_legal_moves(board))
        if move not in legal:
            raise GovernanceStoreContractError(
                f"{field}[{index}] is not legal in its exact prefix"
            )
        board = board.apply_move(move)
    return history, board


def _file_key(path: Path) -> tuple[int, int, int, int]:
    try:
        status = path.stat()
    except OSError as exc:
        raise GovernanceStoreContractError("artifact path cannot be inspected") from exc
    if status.st_nlink != 1:
        raise GovernanceStoreContractError("artifact aliases are forbidden")
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        int(status.st_mtime_ns),
    )


def _read_plan_owned_artifact(
    value: str | Path,
    *,
    store_context: _StoreContext,
    field: str,
) -> tuple[Path, bytes, tuple[int, int, int, int]]:
    try:
        supplied = Path(value)
        absolute = supplied.absolute()
        path = supplied.resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise GovernanceStoreContractError(f"{field} path is missing") from exc
    governance_namespace = store_context.path.parent
    if (
        absolute != path
        or not path.is_file()
        or _is_reparse(path)
        or not path.is_relative_to(store_context.output_root)
        or path.is_relative_to(governance_namespace)
    ):
        raise GovernanceStoreContractError(
            f"{field} must be one plan-owned non-governance file"
        )
    before = _file_key(path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise GovernanceStoreContractError(f"{field} cannot be read") from exc
    after = _file_key(path)
    if before != after or len(payload) != before[2]:
        raise GovernanceStoreContractError(f"{field} changed during read")
    return path, payload, after


_SOURCE_GAME_FIELDS = {
    "schema_version",
    "state_generation_attempt_identity",
    "game_id",
    "game_index",
    "split",
    "candidate_color",
    "complete_history",
    "sanmill_runtime_identity",
    "sanmill_final_state",
    "sanmill_final_state_identity",
    "strict_terminal",
    "referee_binding_identity",
    "record_identity",
}
_COMPLETE_HISTORY_FIELDS = {
    "logical_moves",
    "logical_ply_count",
    "logical_moves_sha256",
    "sanmill_actions",
    "action_token_count",
    "sanmill_actions_sha256",
}
_SANMILL_FINAL_STATE_FIELDS = {
    "status",
    "ruleset_id",
    "rules_identity_sha256",
    "history_origin",
    "fen",
    "side_to_move",
    "phase",
    "action",
    "terminal",
    "removal_pending",
    "pending_removal_count",
    "pending_removals",
    "legal_actions",
    "action_token_count",
    "logical_ply_count",
    "logical_plies_by_side",
    "no_capture_count",
    "repetition_current_count",
    "repetition_history_length",
    "snapshot_history_length",
    "history_sha256",
    "outcome",
    "strict_referee_identity",
}
_STRICT_TERMINAL_FIELDS = {
    "terminal",
    "termination_class",
    "winner",
    "outcome_reason",
    "outcome_reason_code",
    "local_board",
}
_LOCAL_TERMINAL_FIELDS = {"terminal", "winner", "reason"}
_STATE_FIELDS = {
    "schema_version",
    "state_record_identity",
    "example_id",
    "game_id",
    "game_index",
    "split",
    "stratum",
    "candidate_color",
    "logical_ply",
    "history_moves",
    "history_sha256",
    "sanmill_history_sha256",
    "board_fen",
    "legal_actions",
    "a_pos_mask",
    "a_pos_verification",
}
_SINGLETON_ENTRY_FIELDS = {
    "game_id",
    "game_index",
    "split",
    "stratum",
    "candidate_color",
    "logical_ply",
    "history_moves",
    "history_sha256",
    "board_fen",
    "legal_actions",
    "a_pos_mask",
    "a_pos_count",
    "entry_identity",
}


@dataclass(frozen=True, slots=True)
class _ValidatedArtifacts:
    source_games_bytes: bytes
    state_split_bytes: bytes
    singleton_ledger_bytes: bytes
    source_games_identity: str
    state_split_identity: str
    singleton_ledger_identity: str
    source_game_count: int
    state_count: int
    singleton_count: int
    verifier_identity: str


def _split_contract_for_policy(policy: _ArtifactPolicy) -> dict[str, Any]:
    return {
        "strategy": "whole-game",
        "algorithm": {
            "game_index_origin": 0,
            "block_size_games": 16,
            "candidate_colour_by_parity": {"even": "W", "odd": "B"},
            "dev_remainders": [0, 1],
            "train_remainders": list(range(2, 16)),
            "stop_only_after_complete_block": policy.require_complete_game_block,
            "maximum_games": policy.maximum_source_games,
        },
        "counts": _thaw(policy.layout),
    }


def _policy_payload(policy: _ArtifactPolicy) -> dict[str, Any]:
    return {
        "domain": policy.domain,
        "layout": _thaw(policy.layout),
        "total_states": policy.total_states,
        "maximum_source_games": policy.maximum_source_games,
        "require_complete_game_block": policy.require_complete_game_block,
    }


def _policy_from_payload(value: Any, *, domain: str) -> _ArtifactPolicy:
    checked = _require_exact_keys(
        value,
        {
            "domain",
            "layout",
            "total_states",
            "maximum_source_games",
            "require_complete_game_block",
        },
        field="stored artifact policy",
    )
    _require_exact(checked["domain"], domain, field="stored artifact policy.domain")
    if type(checked["require_complete_game_block"]) is not bool:
        raise GovernanceStoreContractError("stored artifact block policy differs")
    if domain == "production":
        _require_exact(
            checked,
            _policy_payload(_PRODUCTION_POLICY),
            field="stored production artifact policy",
        )
        return _PRODUCTION_POLICY
    policy = _test_artifact_policy(
        checked["layout"],
        maximum_source_games=_require_nonnegative_int(
            checked["maximum_source_games"],
            field="stored test maximum source games",
        ),
    )
    _require_exact(
        checked["total_states"],
        policy.total_states,
        field="stored test total states",
    )
    _require_exact(
        checked["require_complete_game_block"],
        False,
        field="stored test block policy",
    )
    return policy


def _recompute_state_split_blob_identity(
    payload: bytes,
    *,
    policy: _ArtifactPolicy,
) -> tuple[str, str, int]:
    records = _strict_jsonl_objects(payload, field="stored state-split")
    if len(records) != policy.total_states:
        raise GovernanceStoreContractError("stored state-split count differs")
    identities: list[str] = []
    verifier_identity: str | None = None
    previous_verification = "0" * 64
    for record in records:
        _require_exact_keys(record, _STATE_FIELDS, field="stored state record")
        body = {
            key: item for key, item in record.items() if key != "state_record_identity"
        }
        identity = _require_sha256(
            record["state_record_identity"],
            field="stored state identity",
        )
        if identity != canonical_sha256(body):
            raise GovernanceStoreContractError("stored state record identity differs")
        verification = _require_exact_keys(
            record["a_pos_verification"],
            {
                "verifier_identity",
                "inventory_sha256",
                "previous_verification_sha256",
                "verification_sha256",
            },
            field="stored state verification",
        )
        current_verifier = _require_sha256(
            verification["verifier_identity"],
            field="stored verifier identity",
        )
        if verifier_identity is None:
            verifier_identity = current_verifier
        if (
            current_verifier != verifier_identity
            or verification["previous_verification_sha256"] != previous_verification
        ):
            raise GovernanceStoreContractError("stored verification chain differs")
        verification_body = {
            key: item
            for key, item in verification.items()
            if key != "verification_sha256"
        }
        previous_verification = _require_sha256(
            verification["verification_sha256"],
            field="stored verification SHA",
        )
        if previous_verification != canonical_sha256(verification_body):
            raise GovernanceStoreContractError("stored verification hash differs")
        inventory = {
            "board_fen": record["board_fen"],
            "legal_actions": record["legal_actions"],
            "a_pos_mask": record["a_pos_mask"],
        }
        if verification["inventory_sha256"] != canonical_sha256(inventory):
            raise GovernanceStoreContractError("stored inventory hash differs")
        identities.append(identity)
    if verifier_identity is None:
        raise GovernanceStoreContractError("stored state verifier is absent")
    split_identity = canonical_sha256(_split_contract_for_policy(policy))
    return (
        state_split_artifact_identity(
            state_record_identities=identities,
            split_identity=split_identity,
            verifier_identity=verifier_identity,
        ),
        verifier_identity,
        len(records),
    )


def _recompute_singleton_blob_identity(payload: bytes) -> tuple[str, int]:
    ledger = _strict_json_object(payload, field="stored singleton ledger")
    _require_exact_keys(
        ledger,
        {"schema_version", "count", "entries", "identity"},
        field="stored singleton ledger",
    )
    body = {key: item for key, item in ledger.items() if key != "identity"}
    identity = _require_sha256(
        ledger["identity"],
        field="stored singleton identity",
    )
    if identity != canonical_sha256(body):
        raise GovernanceStoreContractError("stored singleton identity differs")
    entries = ledger["entries"]
    count = _require_nonnegative_int(ledger["count"], field="stored singleton count")
    if not isinstance(entries, list) or count != len(entries):
        raise GovernanceStoreContractError("stored singleton count differs")
    return identity, count


def _infer_internal_test_policy(
    source_payload: bytes,
    state_payload: bytes,
) -> _ArtifactPolicy:
    source_count = len(_strict_jsonl_objects(source_payload, field="source-games"))
    records = _strict_jsonl_objects(state_payload, field="state-split")
    layout = {
        split: {
            stratum: {colour: 0 for colour in ("W", "B")}
            for stratum in ("placement", "movement", "flying")
        }
        for split in ("train", "dev")
    }
    for record in records:
        if not isinstance(record, Mapping):
            raise GovernanceStoreContractError("test state record differs")
        split = _require_text_choice(
            record.get("split"), {"train", "dev"}, field="test state split"
        )
        stratum = _require_text_choice(
            record.get("stratum"),
            {"placement", "movement", "flying"},
            field="test state stratum",
        )
        colour = _require_text_choice(
            record.get("candidate_color"),
            {"W", "B"},
            field="test state colour",
        )
        layout[split][stratum][colour] += 1
    return _test_artifact_policy(
        layout,
        maximum_source_games=source_count,
    )


def _controller_stream_identity_from_records(
    *,
    spec: GovernanceStoreSpec,
    replay: GovernanceReplay,
    attempt_identity: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": "nmm.classical-a-pos-domain-stream.v1",
            "stream_kind": _CONTROLLER_STREAM_KIND,
            "store_spec_identity": spec["spec_identity"],
            "plan_identity": replay.plan_identity,
            "authorization_consumption_identity": (
                replay.authorization_consumption_identity
            ),
            "state_generation_attempt_identity": attempt_identity,
        }
    )


def _validate_runtime_records(
    replay: GovernanceReplay,
    *,
    spec: GovernanceStoreSpec,
    domain_records: Mapping[str, tuple[str, Mapping[str, Any]]],
) -> dict[str, Any] | None:
    runtime_records = {
        key: domain_records[key]
        for key in _RUNTIME_RECORD_TYPES
        if key in domain_records
    }
    if not runtime_records:
        return None
    if "state-generation-resource-before" not in runtime_records:
        raise GovernanceStoreContractError("runtime stream lacks before evidence")
    has_after = "state-generation-resource-after" in runtime_records
    has_stability = "state-generation-resource-stability" in runtime_records
    if has_after is not has_stability:
        raise GovernanceStoreContractError("runtime completion evidence is partial")
    reservations = tuple(
        event
        for event in replay.events
        if event["event_type"] == "state_generation_reserved"
    )
    if len(reservations) != 1:
        raise GovernanceStoreContractError(
            "runtime reservation event cardinality differs"
        )
    reservation = reservations[0]
    attempt_identity = reservation["attempt_identity"]
    stream_body = {
        "schema_version": _RUNTIME_STREAM_SCHEMA,
        "stream_kind": _RUNTIME_STREAM_KIND,
        "experiment_id": replay.experiment_id,
        "proposal_identity": replay.proposal_identity,
        "profile_identity": _PROFILE_IDENTITY,
        "store_spec_identity": spec["spec_identity"],
        "plan_identity": replay.plan_identity,
        "readiness_identity": replay.readiness_identity,
        "authorization_identity": replay.authorization_identity,
        "authorization_consumption_identity": replay.authorization_consumption_identity,
        "attempt_identity": attempt_identity,
        "reservation_event_identity": reservation["event_identity"],
    }
    expected_stream_identity = canonical_sha256(stream_body)
    before_identity, before = runtime_records["state-generation-resource-before"]
    expected_snapshot_record_keys = (
        _DOMAIN_ENVELOPE_KEYS
        | _RUNTIME_RECORD_CONTEXT_KEYS
        | {"snapshot", "snapshot_identity"}
    )
    _require_exact_keys(before, expected_snapshot_record_keys, field="runtime before")
    before_snapshot = _validate_resource_snapshot(
        before["snapshot"],
        expected_stage="before-execution",
        spec=spec,
        attempt_identity=attempt_identity,
    )
    host_preflight_identity = canonical_sha256(before_snapshot["host"])
    common = {
        "experiment_id": replay.experiment_id,
        "proposal_identity": replay.proposal_identity,
        "profile_identity": _PROFILE_IDENTITY,
        "store_spec_identity": spec["spec_identity"],
        "plan_identity": replay.plan_identity,
        "readiness_identity": replay.readiness_identity,
        "authorization_identity": replay.authorization_identity,
        "authorization_consumption_identity": replay.authorization_consumption_identity,
        "attempt_identity": attempt_identity,
        "reservation_event_identity": reservation["event_identity"],
        "state_generator_session_identity": before["state_generator_session_identity"],
        "host_preflight_identity": host_preflight_identity,
    }
    _require_bound_identity(
        common["state_generator_session_identity"],
        field="runtime state generator session identity",
    )
    _require_bound_identity(
        common["host_preflight_identity"],
        field="runtime host preflight identity",
    )
    expected_before = {
        "schema_version": _RUNTIME_RECORD_SCHEMAS["state-generation-resource-before"],
        "stream_identity": expected_stream_identity,
        "stream_kind": _RUNTIME_STREAM_KIND,
        "sequence": 0,
        "record_type": "state-generation-resource-before",
        "previous_record_identity": None,
        **common,
        "snapshot": before_snapshot,
        "snapshot_identity": before_snapshot["snapshot_identity"],
    }
    _require_exact(before, expected_before, field="runtime before record")
    result: dict[str, Any] = {
        "stream_identity": expected_stream_identity,
        "before_record_identity": before_identity,
        "before_snapshot_identity": before_snapshot["snapshot_identity"],
        "host_preflight_identity": host_preflight_identity,
        "state_generator_session_identity": common["state_generator_session_identity"],
        "complete": False,
    }
    if not has_after:
        return result
    after_identity, after = runtime_records["state-generation-resource-after"]
    stability_identity, stability = runtime_records[
        "state-generation-resource-stability"
    ]
    _require_exact_keys(after, expected_snapshot_record_keys, field="runtime after")
    after_snapshot = _validate_resource_snapshot(
        after["snapshot"],
        expected_stage="after-execution",
        spec=spec,
        attempt_identity=attempt_identity,
    )
    expected_after = {
        "schema_version": _RUNTIME_RECORD_SCHEMAS["state-generation-resource-after"],
        "stream_identity": expected_stream_identity,
        "stream_kind": _RUNTIME_STREAM_KIND,
        "sequence": 1,
        "record_type": "state-generation-resource-after",
        "previous_record_identity": before_identity,
        **common,
        "snapshot": after_snapshot,
        "snapshot_identity": after_snapshot["snapshot_identity"],
    }
    _require_exact(after, expected_after, field="runtime after record")
    expected_stability_keys = (
        _DOMAIN_ENVELOPE_KEYS
        | _RUNTIME_RECORD_CONTEXT_KEYS
        | {
            "before_record_identity",
            "after_record_identity",
            "before_snapshot_identity",
            "after_snapshot_identity",
            "comparison_fields",
            "differences",
            "stable",
        }
    )
    _require_exact_keys(stability, expected_stability_keys, field="runtime stability")
    for path in _STABILITY_COMPARISON_FIELDS:
        _require_exact(
            _nested_path(after_snapshot, path),
            _nested_path(before_snapshot, path),
            field=f"runtime stability.{path}",
        )
    expected_stability = {
        "schema_version": _RUNTIME_RECORD_SCHEMAS[
            "state-generation-resource-stability"
        ],
        "stream_identity": expected_stream_identity,
        "stream_kind": _RUNTIME_STREAM_KIND,
        "sequence": 2,
        "record_type": "state-generation-resource-stability",
        "previous_record_identity": after_identity,
        **common,
        "before_record_identity": before_identity,
        "after_record_identity": after_identity,
        "before_snapshot_identity": before_snapshot["snapshot_identity"],
        "after_snapshot_identity": after_snapshot["snapshot_identity"],
        "comparison_fields": list(_STABILITY_COMPARISON_FIELDS),
        "differences": [],
        "stable": True,
    }
    _require_exact(stability, expected_stability, field="runtime stability record")
    result.update(
        {
            "complete": True,
            "after_record_identity": after_identity,
            "after_snapshot_identity": after_snapshot["snapshot_identity"],
            "stability_identity": stability_identity,
        }
    )
    return result


def _verify_blob_stage_and_identities(
    replay: GovernanceReplay,
    *,
    artifacts: Mapping[str, tuple[str, bytes]],
    domain_records: Mapping[str, tuple[str, Mapping[str, Any]]],
    domain: str,
    spec: GovernanceStoreSpec,
) -> None:
    generated = "state-generation" in replay.completed_operations
    frozen = "state-freeze" in replay.completed_operations
    runtime = _validate_runtime_records(
        replay,
        spec=spec,
        domain_records=domain_records,
    )
    runtime_record_names = {
        name for name in _RUNTIME_RECORD_TYPES if name in domain_records
    }
    expected_artifacts = (
        {"source-games", "state-split", "singleton-ledger"} if generated else set()
    )
    expected_records = (
        {"state-generation-completion", *_RUNTIME_RECORD_TYPES}
        if generated
        else set(runtime_record_names)
    )
    if generated:
        if runtime is None or runtime["complete"] is not True:
            raise GovernanceStoreContractError(
                "generated state lacks complete runtime evidence"
            )
    elif runtime is not None:
        if (
            replay.state != "state_generation_running"
            or runtime["complete"] is not False
            or runtime_record_names != {"state-generation-resource-before"}
        ):
            raise GovernanceStoreContractError(
                "runtime evidence does not match the running stage"
            )
    if frozen:
        expected_artifacts.add("state-freeze-receipt")
        expected_records.add("state-freeze")
    if set(artifacts) != expected_artifacts or set(domain_records) != expected_records:
        raise GovernanceStoreContractError(
            "durable artifact/domain rows do not match replay stage"
        )
    if not generated:
        return
    assert runtime is not None
    completion_identity, completion = domain_records["state-generation-completion"]
    _require_exact_keys(
        completion,
        _COMPLETION_KEYS,
        field="stored state-generation completion",
    )
    _require_exact(
        completion["schema_version"],
        _COMPLETION_SCHEMA,
        field="stored completion schema",
    )
    source_row_identity, source_payload = artifacts["source-games"]
    state_row_identity, state_payload = artifacts["state-split"]
    singleton_row_identity, singleton_payload = artifacts["singleton-ledger"]
    policy = (
        _PRODUCTION_POLICY
        if domain == "production"
        else _infer_internal_test_policy(source_payload, state_payload)
    )
    source_identity, games = _validate_source_games(source_payload, policy=policy)
    state_identity, verifier_identity, state_count = (
        _recompute_state_split_blob_identity(state_payload, policy=policy)
    )
    singleton_identity, singleton_count = _recompute_singleton_blob_identity(
        singleton_payload
    )
    completion_events = tuple(
        event
        for event in replay.events
        if event["event_type"] == "state_generation_completed"
    )
    if len(completion_events) != 1:
        raise GovernanceStoreContractError("completion event cardinality differs")
    completion_event = completion_events[0]
    attempt_identity = completion_event["attempt_identity"]
    stream_identity = _controller_stream_identity_from_records(
        spec=spec,
        replay=replay,
        attempt_identity=attempt_identity,
    )
    referee_identities = {item["referee_binding_identity"] for item in games.values()}
    runtime_identities = {item["sanmill_runtime_identity"] for item in games.values()}
    if len(referee_identities) != 1 or len(runtime_identities) != 1:
        raise GovernanceStoreContractError("source runtime/referee identity differs")
    expected_completion = {
        "schema_version": _COMPLETION_SCHEMA,
        "stream_identity": stream_identity,
        "stream_kind": _CONTROLLER_STREAM_KIND,
        "sequence": 0,
        "record_type": "state-generation-completion",
        "previous_record_identity": None,
        "experiment_id": replay.experiment_id,
        "proposal_identity": replay.proposal_identity,
        "profile_identity": _PROFILE_IDENTITY,
        "store_spec_identity": spec["spec_identity"],
        "plan_identity": replay.plan_identity,
        "readiness_identity": replay.readiness_identity,
        "managed_git_state_identity": spec["managed_git_state_identity"],
        "launch_path_binding_identity": spec["launch_path_binding_identity"],
        "authorization_identity": replay.authorization_identity,
        "authorization_consumption_identity": (
            replay.authorization_consumption_identity
        ),
        "attempt_identity": attempt_identity,
        "reservation_event_identity": completion_event["prerequisite_event_identity"],
        "completion_event_identity": completion_event["event_identity"],
        "host_preflight_identity": runtime["host_preflight_identity"],
        "state_generator_session_identity": runtime["state_generator_session_identity"],
        "strict_referee_binding_identity": next(iter(referee_identities)),
        "a_pos_inventory_binding_identity": _inventory_binding_identity(
            verifier_identity,
            policy,
        ),
        "runtime_evidence_stream_identity": runtime["stream_identity"],
        "resource_snapshot_before_identity": runtime["before_snapshot_identity"],
        "resource_snapshot_after_identity": runtime["after_snapshot_identity"],
        "resource_stability_identity": runtime["stability_identity"],
        "artifacts": [
            _artifact_contract_ref(
                "source-games",
                source_identity,
                source_payload,
                record_count=len(games),
            ),
            _artifact_contract_ref(
                "state-split",
                state_identity,
                state_payload,
                record_count=state_count,
                a_pos_verifier_identity=verifier_identity,
            ),
            _artifact_contract_ref(
                "singleton-ledger",
                singleton_identity,
                singleton_payload,
                record_count=singleton_count,
            ),
        ],
        "split_contract_identity": canonical_sha256(_split_contract_for_policy(policy)),
        "resource_observation": _thaw(completion_event["resource_observation"]),
        "teacher_fields_present": False,
        "authoritative_storage": "sqlite-immutable-blob-second-read",
    }
    expected_completion_evidence = {
        "inputs": [],
        "outputs": [
            _evidence_ref("source-games", source_identity, source_payload),
            _evidence_ref("state-split", state_identity, state_payload),
            _evidence_ref(
                "singleton-ledger",
                singleton_identity,
                singleton_payload,
            ),
        ],
        "checkpoint": None,
    }
    if (
        source_row_identity != source_identity
        or state_row_identity != state_identity
        or singleton_row_identity != singleton_identity
        or completion_identity != canonical_sha256(completion)
        or completion_identity != _sha256_bytes(canonical_json_bytes(completion))
    ):
        raise GovernanceStoreContractError("durable artifact semantic identity differs")
    _require_exact(
        completion,
        expected_completion,
        field="stored state-generation completion",
    )
    _require_exact(
        completion_event["evidence"],
        expected_completion_evidence,
        field="stored state-generation completion evidence",
    )
    for field, expected in {
        "operation_id": "state-generation",
        "purpose": "state",
        "seed": None,
        "from_state": "state_generation_running",
        "to_state": "state_generated",
    }.items():
        _require_exact(
            completion_event[field],
            expected,
            field=f"stored completion event.{field}",
        )
    _require_exact(
        completion_event["resource_observation"]["state_generation_games"],
        len(games),
        field="stored completion observed games",
    )
    if not frozen:
        return
    receipt_row_identity, receipt_payload = artifacts["state-freeze-receipt"]
    freeze_identity, freeze_record = domain_records["state-freeze"]
    receipt = _strict_json_object(receipt_payload, field="stored freeze receipt")
    _require_exact_keys(
        receipt,
        _FREEZE_RECEIPT_KEYS,
        field="stored state-freeze receipt",
    )
    freeze_events = tuple(
        event for event in replay.events if event["event_type"] == "state_frozen"
    )
    if len(freeze_events) != 1:
        raise GovernanceStoreContractError("state-freeze event cardinality differs")
    freeze_event = freeze_events[0]
    expected_receipt = {
        "schema_version": _FREEZE_RECEIPT_SCHEMA,
        "stream_identity": stream_identity,
        "stream_kind": _CONTROLLER_STREAM_KIND,
        "sequence": 1,
        "record_type": "state-freeze",
        "previous_record_identity": completion_identity,
        "experiment_id": replay.experiment_id,
        "proposal_identity": replay.proposal_identity,
        "profile_identity": _PROFILE_IDENTITY,
        "store_spec_identity": spec["spec_identity"],
        "plan_identity": replay.plan_identity,
        "readiness_identity": replay.readiness_identity,
        "authorization_identity": replay.authorization_identity,
        "authorization_consumption_identity": (
            replay.authorization_consumption_identity
        ),
        "attempt_identity": attempt_identity,
        "completion_event_identity": completion_event["event_identity"],
        "completion_record_identity": completion_identity,
        "source_games_identity": source_identity,
        "state_split_identity": state_identity,
        "singleton_ledger_identity": singleton_identity,
        "runtime_evidence_stream_identity": runtime["stream_identity"],
        "resource_stability_identity": runtime["stability_identity"],
        "teacher_fields_present": False,
        "teacher_may_start_only_after_this_event": True,
        "authoritative_artifacts": "sqlite-immutable-blobs",
    }
    expected_freeze_evidence = {
        "inputs": [
            _evidence_ref("source-games", source_identity, source_payload),
            _evidence_ref("state-split", state_identity, state_payload),
            _evidence_ref(
                "singleton-ledger",
                singleton_identity,
                singleton_payload,
            ),
        ],
        "outputs": [
            _evidence_ref(
                "state-freeze-receipt",
                receipt_row_identity,
                receipt_payload,
            )
        ],
        "checkpoint": None,
    }
    if (
        receipt_row_identity != canonical_sha256(receipt)
        or freeze_identity != receipt_row_identity
        or freeze_record != receipt
        or freeze_identity != _sha256_bytes(receipt_payload)
    ):
        raise GovernanceStoreContractError("durable freeze semantic identity differs")
    _require_exact(
        receipt,
        expected_receipt,
        field="stored state-freeze receipt",
    )
    _require_exact(
        freeze_event["evidence"],
        expected_freeze_evidence,
        field="stored state-freeze evidence",
    )
    for field, expected in {
        "operation_id": "state-freeze",
        "purpose": "governance",
        "seed": None,
        "from_state": "state_generated",
        "to_state": "state_frozen",
        "prerequisite_event_identity": completion_event["event_identity"],
    }.items():
        _require_exact(
            freeze_event[field],
            expected,
            field=f"stored state-freeze event.{field}",
        )


def _validate_sanmill_final_state(
    value: Any,
    *,
    history: Sequence[Mapping[str, Any]],
    actions: Sequence[str],
    board: BoardState,
    strict_referee: _StrictRefereeContext | None,
    field: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    final_state = dict(
        _require_exact_keys(value, _SANMILL_FINAL_STATE_FIELDS, field=field)
    )
    expected_referee = {
        "format": TRAINING_REFEREE_FORMAT,
        "profile": TRAINING_REFEREE_PROFILE,
        "repetitionObservation": TRAINING_REPETITION_OBSERVATION,
        "originCounted": True,
        "semanticDigest": TRAINING_REFEREE_SEMANTIC_DIGEST,
    }
    for name, expected in {
        "status": "terminal",
        "ruleset_id": "nmm",
        "rules_identity_sha256": EXPECTED_RULES_IDENTITY_SHA256,
        "history_origin": "game_start",
        "phase": "game_over",
        "action": "game_over",
        "terminal": True,
        "removal_pending": False,
        "pending_removal_count": 0,
        "pending_removals": [0, 0],
        "legal_actions": [],
        "action_token_count": len(actions),
        "logical_ply_count": len(history),
        "strict_referee_identity": expected_referee,
    }.items():
        _require_exact(final_state[name], expected, field=f"{field}.{name}")
    side_to_move = final_state["side_to_move"]
    if side_to_move is not None and (
        type(side_to_move) is not str or side_to_move not in {"white", "black"}
    ):
        raise GovernanceStoreContractError(f"{field}.side_to_move differs")
    for name in (
        "logical_plies_by_side",
        "no_capture_count",
        "repetition_current_count",
        "repetition_history_length",
        "snapshot_history_length",
    ):
        if name == "logical_plies_by_side":
            counts = final_state[name]
            if (
                not isinstance(counts, list)
                or len(counts) != 2
                or any(type(item) is not int or item < 0 for item in counts)
                or sum(counts) != len(history)
            ):
                raise GovernanceStoreContractError(f"{field}.{name} differs")
        else:
            _require_nonnegative_int(final_state[name], field=f"{field}.{name}")
    _require_sha256(final_state["history_sha256"], field=f"{field}.history_sha256")
    outcome = _require_exact_keys(
        final_state["outcome"],
        {"terminal", "winner", "winner_code", "reason", "reason_code"},
        field=f"{field}.outcome",
    )
    _require_exact(outcome["terminal"], True, field=f"{field}.outcome.terminal")
    winner = outcome["winner"]
    winner_code = outcome["winner_code"]
    if not (
        (winner is None and winner_code is None)
        or (
            type(winner) is str
            and winner == "white"
            and type(winner_code) is int
            and winner_code == 0
        )
        or (
            type(winner) is str
            and winner == "black"
            and type(winner_code) is int
            and winner_code == 1
        )
    ):
        raise GovernanceStoreContractError(f"{field}.outcome winner differs")
    _require_nonempty_text(outcome["reason"], field=f"{field}.outcome.reason")
    _require_nonempty_text(outcome["reason_code"], field=f"{field}.outcome.reason_code")
    if outcome["reason"] == "ongoing" or outcome["reason_code"] == "ongoing":
        raise GovernanceStoreContractError(f"{field}.outcome is not terminal")
    try:
        projected = project_stable_sanmill_fen(final_state["fen"], terminal=True)
    except (SanmillBridgeError, ValueError, TypeError) as exc:
        raise GovernanceStoreContractError(f"{field}.fen cannot be projected") from exc
    if (
        projected.positions != board.positions
        or projected.pieces_placed != board.pieces_placed
        or projected.pieces_on_board != board.pieces_on_board
        or projected.turn != board.turn
    ):
        raise GovernanceStoreContractError(f"{field}.fen projection differs")
    if (
        side_to_move is not None
        and {
            "white": "W",
            "black": "B",
        }[side_to_move]
        != board.turn
    ):
        raise GovernanceStoreContractError(f"{field}.side_to_move differs from replay")
    if strict_referee is not None:
        try:
            observed = strict_referee.complete_history_verifier(history, actions)
        except Exception as exc:
            raise GovernanceStoreContractError(
                f"{field} live strict-referee verification failed"
            ) from exc
        if type(observed) is not UciPositionState:
            raise GovernanceStoreContractError(
                f"{field} live strict-referee state type differs"
            )
        _require_exact(
            observed.portable_record(),
            final_state,
            field=f"{field} live strict-referee state",
        )
    return final_state, dict(outcome)


def _validate_strict_terminal(
    value: Any,
    *,
    board: BoardState,
    outcome: Mapping[str, Any],
    field: str,
) -> None:
    terminal = _require_exact_keys(value, _STRICT_TERMINAL_FIELDS, field=field)
    _require_exact(terminal["terminal"], True, field=f"{field}.terminal")
    classification = _require_text_choice(
        terminal["termination_class"],
        {"rules-win", "repetition-draw", "rules-draw"},
        field=f"{field}.termination_class",
    )
    _require_exact(
        terminal["outcome_reason"],
        outcome["reason"],
        field=f"{field}.outcome_reason",
    )
    _require_exact(
        terminal["outcome_reason_code"],
        outcome["reason_code"],
        field=f"{field}.outcome_reason_code",
    )
    local_terminal, local_winner, local_reason = terminal_result(board)
    local = {
        "terminal": local_terminal,
        "winner": local_winner,
        "reason": local_reason,
    }
    _require_exact_keys(
        terminal["local_board"], _LOCAL_TERMINAL_FIELDS, field=f"{field}.local_board"
    )
    _require_exact(terminal["local_board"], local, field=f"{field}.local_board")
    expected_winner = {"W": "white", "B": "black", None: None}[local_winner]
    outcome_key = (outcome["reason"], outcome["reason_code"])
    if local_terminal:
        expected_outcome = _RULES_WIN_OUTCOME_BY_LOCAL_REASON.get(local_reason)
        if (
            classification != "rules-win"
            or outcome["winner"] != expected_winner
            or expected_outcome is None
            or outcome_key != expected_outcome
        ):
            raise GovernanceStoreContractError(
                f"{field} conflicts with local rules terminal"
            )
    elif classification == "rules-win":
        raise GovernanceStoreContractError(f"{field} invents a local rules win")
    elif outcome["winner"] is not None:
        raise GovernanceStoreContractError(f"{field} draw has a winner")
    elif classification == "repetition-draw":
        if outcome_key != _REPETITION_DRAW_OUTCOME:
            raise GovernanceStoreContractError(
                f"{field} repetition reason differs from pinned Sanmill"
            )
    elif outcome_key not in _RULES_DRAW_OUTCOMES:
        raise GovernanceStoreContractError(
            f"{field} rules-draw reason differs from pinned Sanmill"
        )
    _require_exact(terminal["winner"], local_winner, field=f"{field}.winner")


def _validate_source_games(
    payload: bytes,
    *,
    policy: _ArtifactPolicy,
    attempt_identity: str | None = None,
    strict_referee: _StrictRefereeContext | None = None,
) -> tuple[str, dict[str, dict[str, Any]]]:
    records = _strict_jsonl_objects(payload, field="source-games")
    count = len(records)
    if (
        count == 0
        or count > policy.maximum_source_games
        or (policy.require_complete_game_block and count % 16 != 0)
    ):
        raise GovernanceStoreContractError("source-game count/block contract differs")
    games: dict[str, dict[str, Any]] = {}
    indices: set[int] = set()
    record_identities: list[str] = []
    seen_record_identities: set[str] = set()
    history_identities: set[str] = set()
    sanmill_history_identities: set[str] = set()
    for expected_index, record in enumerate(records):
        _require_exact_keys(record, _SOURCE_GAME_FIELDS, field="source game")
        _require_exact(
            record["schema_version"],
            _SOURCE_GAME_SCHEMA,
            field="source game.schema_version",
        )
        record_attempt = _require_bound_identity(
            record["state_generation_attempt_identity"],
            field="source game.state_generation_attempt_identity",
        )
        if attempt_identity is not None and record_attempt != attempt_identity:
            raise GovernanceStoreContractError("source game attempt identity differs")
        if (
            strict_referee is not None
            and record_attempt != strict_referee.attempt_identity
        ):
            raise GovernanceStoreContractError(
                "source game strict-referee attempt identity differs"
            )
        game_id = _require_sha256(
            record["game_id"],
            field="source game.game_id",
        )
        game_index = _require_nonnegative_int(
            record["game_index"],
            field="source game.game_index",
        )
        if game_index != expected_index:
            raise GovernanceStoreContractError(
                "source games must use contiguous canonical game indices"
            )
        split = _require_text_choice(
            record["split"], {"train", "dev"}, field="source game.split"
        )
        expected_split = "dev" if game_index % 16 in {0, 1} else "train"
        if split != expected_split:
            raise GovernanceStoreContractError("source game split differs")
        colour = _require_text_choice(
            record["candidate_color"],
            {"W", "B"},
            field="source game.candidate_color",
        )
        expected_colour = "W" if game_index % 2 == 0 else "B"
        if colour != expected_colour:
            raise GovernanceStoreContractError("source game candidate colour differs")
        complete_history = _require_exact_keys(
            record["complete_history"],
            _COMPLETE_HISTORY_FIELDS,
            field="source game.complete_history",
        )
        history, board = _replay_history(
            complete_history["logical_moves"],
            field="source game.complete_history.logical_moves",
        )
        if not history:
            raise GovernanceStoreContractError("source game history must be non-empty")
        history_identity = canonical_sha256(list(history))
        if (
            _require_sha256(
                complete_history["logical_moves_sha256"],
                field="source game.complete_history.logical_moves_sha256",
            )
            != history_identity
        ):
            raise GovernanceStoreContractError("source game history identity differs")
        _require_exact(
            complete_history["logical_ply_count"],
            len(history),
            field="source game.complete_history.logical_ply_count",
        )
        expected_actions = tuple(
            action for move in history for action in nmm_move_actions(move)
        )
        raw_actions = complete_history["sanmill_actions"]
        if not isinstance(raw_actions, list) or any(
            type(action) is not str or not action for action in raw_actions
        ):
            raise GovernanceStoreContractError("source game Sanmill actions differ")
        _require_exact(
            raw_actions,
            list(expected_actions),
            field="source game.complete_history.sanmill_actions",
        )
        _require_exact(
            complete_history["action_token_count"],
            len(expected_actions),
            field="source game.complete_history.action_token_count",
        )
        actions_identity = canonical_sha256(list(expected_actions))
        if (
            _require_sha256(
                complete_history["sanmill_actions_sha256"],
                field="source game.complete_history.sanmill_actions_sha256",
            )
            != actions_identity
        ):
            raise GovernanceStoreContractError(
                "source game Sanmill action hash differs"
            )
        runtime_identity = _require_bound_identity(
            record["sanmill_runtime_identity"],
            field="source game.sanmill_runtime_identity",
        )
        referee_identity = _require_bound_identity(
            record["referee_binding_identity"],
            field="source game.referee_binding_identity",
        )
        if strict_referee is not None and (
            runtime_identity != strict_referee.runtime_identity
            or referee_identity != strict_referee.binding_identity
        ):
            raise GovernanceStoreContractError(
                "source game strict-referee identity differs"
            )
        final_state, outcome = _validate_sanmill_final_state(
            record["sanmill_final_state"],
            history=history,
            actions=expected_actions,
            board=board,
            strict_referee=strict_referee,
            field="source game.sanmill_final_state",
        )
        final_state_identity = _require_sha256(
            record["sanmill_final_state_identity"],
            field="source game.sanmill_final_state_identity",
        )
        if final_state_identity != canonical_sha256(final_state):
            raise GovernanceStoreContractError(
                "source game final state identity differs"
            )
        _validate_strict_terminal(
            record["strict_terminal"],
            board=board,
            outcome=outcome,
            field="source game.strict_terminal",
        )
        expected_game_id = canonical_sha256(
            {
                "schema_version": "nmm.classical-a-pos-source-game-id.v1",
                "state_generation_attempt_identity": record_attempt,
                "game_index": game_index,
                "logical_moves_sha256": history_identity,
                "sanmill_final_state_identity": final_state_identity,
                "referee_binding_identity": referee_identity,
            }
        )
        if game_id != expected_game_id:
            raise GovernanceStoreContractError("source game ID binding differs")
        body = {key: item for key, item in record.items() if key != "record_identity"}
        identity = _require_sha256(
            record["record_identity"],
            field="source game.record_identity",
        )
        if identity != canonical_sha256(body):
            raise GovernanceStoreContractError("source game record identity differs")
        sanmill_history_identity = _require_sha256(
            final_state["history_sha256"],
            field="source game.sanmill_final_state.history_sha256",
        )
        if (
            game_id in games
            or game_index in indices
            or identity in seen_record_identities
            or history_identity in history_identities
            or sanmill_history_identity in sanmill_history_identities
        ):
            raise GovernanceStoreContractError(
                "source game identity/index/history repeats"
            )
        games[game_id] = {
            "game_index": game_index,
            "split": split,
            "candidate_color": colour,
            "history_moves": history,
            "sanmill_actions": expected_actions,
            "sanmill_final_state_identity": final_state_identity,
            "sanmill_runtime_identity": runtime_identity,
            "referee_binding_identity": referee_identity,
            "record_identity": identity,
        }
        indices.add(game_index)
        history_identities.add(history_identity)
        sanmill_history_identities.add(sanmill_history_identity)
        seen_record_identities.add(identity)
        record_identities.append(identity)
    identity = canonical_sha256(
        {
            "schema_version": "nmm.classical-a-pos-source-games-artifact.v2",
            "game_count": count,
            "record_identities": record_identities,
        }
    )
    return identity, games


def _live_inventory_mask(
    board: BoardState,
    legal: tuple[dict[str, Any], ...],
    *,
    inventory: _InventoryContext,
    field: str,
) -> tuple[bool, ...]:
    try:
        observed = tuple(inventory.inventory_verifier(board, legal))
    except Exception as exc:
        raise GovernanceStoreContractError(f"{field} live A_pos query failed") from exc
    if len(observed) != len(legal) or any(type(item) is not bool for item in observed):
        raise GovernanceStoreContractError(f"{field} live A_pos mask shape differs")
    return observed


def _validate_state_split(
    payload: bytes,
    *,
    games: Mapping[str, Mapping[str, Any]],
    inventory: _InventoryContext,
    strict_referee: _StrictRefereeContext,
) -> tuple[str, set[str], tuple[str, ...]]:
    policy = inventory.policy
    records = _strict_jsonl_objects(payload, field="state-split")
    if len(records) != policy.total_states:
        raise GovernanceStoreContractError("state-split count differs")
    expected_counts = _thaw(policy.layout)
    observed_counts = {
        split: {
            stratum: {colour: 0 for colour in ("W", "B")}
            for stratum in ("placement", "movement", "flying")
        }
        for split in ("train", "dev")
    }
    prior_selection_keys: dict[tuple[str, str, str], str] = {}
    prior_verification = "0" * 64
    state_identities: list[str] = []
    seen_state_identities: set[str] = set()
    history_identities: set[str] = set()
    example_identities: set[str] = set()
    prefixes: set[tuple[str, int]] = set()
    game_splits: dict[str, str] = {}
    for record in records:
        _require_exact_keys(record, _STATE_FIELDS, field="state record")
        _require_exact(
            record["schema_version"],
            STATE_RECORD_SCHEMA,
            field="state record.schema_version",
        )
        if any("teacher" in key.lower() for key in record):
            raise GovernanceStoreContractError("state record contains teacher data")
        game_id = _require_nonempty_text(
            record["game_id"],
            field="state record.game_id",
        )
        if game_id not in games:
            raise GovernanceStoreContractError("state record game is not source-bound")
        source = games[game_id]
        game_index = _require_nonnegative_int(
            record["game_index"],
            field="state record.game_index",
        )
        if game_index != source["game_index"]:
            raise GovernanceStoreContractError("state/source game index differs")
        split = _require_text_choice(
            record["split"],
            {"train", "dev"},
            field="state record.split",
        )
        stratum = _require_text_choice(
            record["stratum"],
            {"placement", "movement", "flying"},
            field="state record.stratum",
        )
        colour = _require_text_choice(
            record["candidate_color"],
            {"W", "B"},
            field="state record.candidate_color",
        )
        expected_split = "dev" if game_index % 16 in {0, 1} else "train"
        if (
            split != expected_split
            or split != source["split"]
            or colour != source["candidate_color"]
        ):
            raise GovernanceStoreContractError("whole-game split/colour differs")
        if game_splits.setdefault(game_id, split) != split:
            raise GovernanceStoreContractError("one source game crosses split")
        history, board = _replay_history(
            record["history_moves"],
            field="state record.history_moves",
        )
        logical_ply = _require_nonnegative_int(
            record["logical_ply"],
            field="state record.logical_ply",
        )
        if logical_ply != len(history):
            raise GovernanceStoreContractError("state logical ply differs")
        source_history = source["history_moves"]
        if (
            len(history) > len(source_history)
            or tuple(source_history[: len(history)]) != history
        ):
            raise GovernanceStoreContractError(
                "state is not an exact source-game prefix"
            )
        if board.turn != colour or record["board_fen"] != board.to_fen_string():
            raise GovernanceStoreContractError("state board/candidate binding differs")
        expected_stratum = {
            "place": "placement",
            "move": "movement",
            "fly": "flying",
        }[get_game_phase(board, colour)]
        if stratum != expected_stratum:
            raise GovernanceStoreContractError("state stratum differs")
        history_identity = canonical_sha256(list(history))
        if (
            _require_sha256(
                record["history_sha256"],
                field="state record.history_sha256",
            )
            != history_identity
        ):
            raise GovernanceStoreContractError("state history identity differs")
        sanmill_history_identity = _require_sha256(
            record["sanmill_history_sha256"],
            field="state record.sanmill_history_sha256",
        )
        try:
            observed_sanmill_history = strict_referee.prefix_history_verifier(history)
        except Exception as exc:
            raise GovernanceStoreContractError(
                "state prefix live strict-referee verification failed"
            ) from exc
        if (
            _require_sha256(
                observed_sanmill_history,
                field="state prefix live Sanmill history identity",
            )
            != sanmill_history_identity
        ):
            raise GovernanceStoreContractError("state prefix Sanmill history differs")
        example_identity = _require_sha256(
            record["example_id"],
            field="state record.example_id",
        )
        expected_example_identity = canonical_sha256(
            {"history_sha256": history_identity, "board_fen": board.to_fen_string()}
        )
        if example_identity != expected_example_identity:
            raise GovernanceStoreContractError("state example identity differs")
        raw_legal = record["legal_actions"]
        if not isinstance(raw_legal, list):
            raise GovernanceStoreContractError("state legal_actions must be an array")
        legal = tuple(
            _atomic_action(move, field=f"state legal_actions[{index}]")
            for index, move in enumerate(raw_legal)
        )
        expected_legal = tuple(dict(move) for move in get_all_legal_moves(board))
        if legal != expected_legal or len({_action_key(move) for move in legal}) != len(
            legal
        ):
            raise GovernanceStoreContractError("state legal inventory/order differs")
        raw_mask = record["a_pos_mask"]
        if (
            not isinstance(raw_mask, list)
            or len(raw_mask) != len(legal)
            or any(type(item) is not bool for item in raw_mask)
        ):
            raise GovernanceStoreContractError("state A_pos mask shape differs")
        mask = tuple(raw_mask)
        if sum(mask) <= 1:
            raise GovernanceStoreContractError(
                "state artifact contains non-informative A_pos"
            )
        if (
            _live_inventory_mask(
                board,
                legal,
                inventory=inventory,
                field="state record",
            )
            != mask
        ):
            raise GovernanceStoreContractError("state live A_pos inventory differs")
        verification = _require_exact_keys(
            record["a_pos_verification"],
            {
                "verifier_identity",
                "inventory_sha256",
                "previous_verification_sha256",
                "verification_sha256",
            },
            field="state A_pos verification",
        )
        if (
            verification["verifier_identity"] != inventory.verifier_identity
            or verification["previous_verification_sha256"] != prior_verification
            or verification["inventory_sha256"]
            != canonical_sha256(
                {
                    "board_fen": board.to_fen_string(),
                    "legal_actions": list(legal),
                    "a_pos_mask": list(mask),
                }
            )
        ):
            raise GovernanceStoreContractError(
                "state A_pos verification binding differs"
            )
        verification_body = {
            key: item
            for key, item in verification.items()
            if key != "verification_sha256"
        }
        prior_verification = _require_sha256(
            verification["verification_sha256"],
            field="state verification SHA",
        )
        if prior_verification != canonical_sha256(verification_body):
            raise GovernanceStoreContractError("state verification chain hash differs")
        state_body = {
            key: item for key, item in record.items() if key != "state_record_identity"
        }
        state_identity = _require_sha256(
            record["state_record_identity"],
            field="state_record_identity",
        )
        if state_identity != canonical_sha256(state_body):
            raise GovernanceStoreContractError("state record identity differs")
        prefix = (game_id, logical_ply)
        if (
            state_identity in seen_state_identities
            or history_identity in history_identities
            or example_identity in example_identities
            or prefix in prefixes
        ):
            raise GovernanceStoreContractError("state identity/history/prefix repeats")
        state_identities.append(state_identity)
        seen_state_identities.add(state_identity)
        history_identities.add(history_identity)
        example_identities.add(example_identity)
        prefixes.add(prefix)
        cell = (split, stratum, colour)
        selection_key = canonical_sha256(
            {"selection_seed": 2026083091, "history_sha256": history_identity}
        )
        prior_selection_key = prior_selection_keys.get(cell)
        if prior_selection_key is not None and selection_key <= prior_selection_key:
            raise GovernanceStoreContractError(
                "state canonical selection order differs"
            )
        prior_selection_keys[cell] = selection_key
        observed_counts[split][stratum][colour] += 1
    if observed_counts != expected_counts:
        raise GovernanceStoreContractError("state 12-cell quota differs")
    split_identity = canonical_sha256(_split_contract_for_policy(policy))
    identity = state_split_artifact_identity(
        state_record_identities=state_identities,
        split_identity=split_identity,
        verifier_identity=inventory.verifier_identity,
    )
    return identity, history_identities, tuple(state_identities)


def _validate_singleton_ledger(
    payload: bytes,
    *,
    games: Mapping[str, Mapping[str, Any]],
    state_history_identities: set[str],
    inventory: _InventoryContext,
) -> tuple[str, int]:
    ledger = _strict_json_object(payload, field="singleton ledger")
    _require_exact_keys(
        ledger,
        {"schema_version", "count", "entries", "identity"},
        field="singleton ledger",
    )
    _require_exact(
        ledger["schema_version"],
        SINGLETON_LEDGER_SCHEMA,
        field="singleton ledger.schema_version",
    )
    entries = ledger["entries"]
    if not isinstance(entries, list):
        raise GovernanceStoreContractError("singleton entries must be an array")
    count = _require_nonnegative_int(ledger["count"], field="singleton count")
    if count != len(entries):
        raise GovernanceStoreContractError("singleton count differs")
    body = {key: item for key, item in ledger.items() if key != "identity"}
    identity = _require_sha256(ledger["identity"], field="singleton identity")
    if identity != canonical_sha256(body):
        raise GovernanceStoreContractError("singleton ledger identity differs")
    seen_history: set[str] = set()
    seen_prefixes: set[tuple[str, int]] = set()
    for entry in entries:
        _require_exact_keys(entry, _SINGLETON_ENTRY_FIELDS, field="singleton entry")
        game_id = _require_nonempty_text(
            entry["game_id"],
            field="singleton game_id",
        )
        if game_id not in games:
            raise GovernanceStoreContractError("singleton game is not source-bound")
        source = games[game_id]
        game_index = _require_nonnegative_int(
            entry["game_index"],
            field="singleton game_index",
        )
        if game_index != source["game_index"]:
            raise GovernanceStoreContractError("singleton/source game index differs")
        split = _require_text_choice(
            entry["split"],
            {"train", "dev"},
            field="singleton split",
        )
        colour = _require_text_choice(
            entry["candidate_color"],
            {"W", "B"},
            field="singleton candidate_color",
        )
        expected_split = "dev" if game_index % 16 in {0, 1} else "train"
        if split != expected_split or colour != source["candidate_color"]:
            raise GovernanceStoreContractError("singleton split/colour differs")
        history, board = _replay_history(
            entry["history_moves"],
            field="singleton history_moves",
        )
        logical_ply = _require_nonnegative_int(
            entry["logical_ply"],
            field="singleton logical_ply",
        )
        if logical_ply != len(history):
            raise GovernanceStoreContractError("singleton logical ply differs")
        source_history = source["history_moves"]
        if (
            len(history) > len(source_history)
            or tuple(source_history[: len(history)]) != history
        ):
            raise GovernanceStoreContractError("singleton is not a source-game prefix")
        if board.turn != colour or entry["board_fen"] != board.to_fen_string():
            raise GovernanceStoreContractError("singleton board binding differs")
        stratum = {
            "place": "placement",
            "move": "movement",
            "fly": "flying",
        }[get_game_phase(board, colour)]
        if (
            _require_text_choice(
                entry["stratum"],
                {"placement", "movement", "flying"},
                field="singleton stratum",
            )
            != stratum
        ):
            raise GovernanceStoreContractError("singleton stratum differs")
        history_identity = canonical_sha256(list(history))
        if (
            _require_sha256(
                entry["history_sha256"],
                field="singleton history_sha256",
            )
            != history_identity
        ):
            raise GovernanceStoreContractError("singleton history identity differs")
        raw_legal = entry["legal_actions"]
        if not isinstance(raw_legal, list):
            raise GovernanceStoreContractError(
                "singleton legal_actions must be an array"
            )
        legal = tuple(
            _atomic_action(move, field=f"singleton legal_actions[{index}]")
            for index, move in enumerate(raw_legal)
        )
        if legal != tuple(dict(move) for move in get_all_legal_moves(board)):
            raise GovernanceStoreContractError("singleton legal order differs")
        raw_mask = entry["a_pos_mask"]
        if (
            not isinstance(raw_mask, list)
            or len(raw_mask) != len(legal)
            or any(type(item) is not bool for item in raw_mask)
        ):
            raise GovernanceStoreContractError("singleton A_pos mask differs")
        mask = tuple(raw_mask)
        if (
            _require_nonnegative_int(
                entry["a_pos_count"],
                field="singleton a_pos_count",
            )
            != 1
            or sum(mask) != 1
        ):
            raise GovernanceStoreContractError("singleton entry is not |A_pos|=1")
        if (
            _live_inventory_mask(
                board,
                legal,
                inventory=inventory,
                field="singleton entry",
            )
            != mask
        ):
            raise GovernanceStoreContractError("singleton live A_pos differs")
        entry_body = {
            key: item for key, item in entry.items() if key != "entry_identity"
        }
        if _require_sha256(
            entry["entry_identity"],
            field="singleton entry identity",
        ) != canonical_sha256(entry_body):
            raise GovernanceStoreContractError("singleton entry identity differs")
        prefix = (game_id, logical_ply)
        if (
            history_identity in state_history_identities
            or history_identity in seen_history
            or prefix in seen_prefixes
        ):
            raise GovernanceStoreContractError("singleton identity/prefix repeats")
        seen_history.add(history_identity)
        seen_prefixes.add(prefix)
    return identity, count


def _validate_artifacts(
    source_games_bytes: bytes,
    state_split_bytes: bytes,
    singleton_ledger_bytes: bytes,
    *,
    inventory: _InventoryContext,
    strict_referee: _StrictRefereeContext,
    attempt_identity: str,
) -> _ValidatedArtifacts:
    source_identity, games = _validate_source_games(
        source_games_bytes,
        policy=inventory.policy,
        attempt_identity=attempt_identity,
        strict_referee=strict_referee,
    )
    state_identity, state_histories, state_ids = _validate_state_split(
        state_split_bytes,
        games=games,
        inventory=inventory,
        strict_referee=strict_referee,
    )
    singleton_identity, singleton_count = _validate_singleton_ledger(
        singleton_ledger_bytes,
        games=games,
        state_history_identities=state_histories,
        inventory=inventory,
    )
    return _ValidatedArtifacts(
        source_games_bytes=source_games_bytes,
        state_split_bytes=state_split_bytes,
        singleton_ledger_bytes=singleton_ledger_bytes,
        source_games_identity=source_identity,
        state_split_identity=state_identity,
        singleton_ledger_identity=singleton_identity,
        source_game_count=len(games),
        state_count=len(state_ids),
        singleton_count=singleton_count,
        verifier_identity=inventory.verifier_identity,
    )


def _evidence_ref(role: str, identity: str, payload: bytes) -> dict[str, Any]:
    return {
        "role": role,
        "identity": _require_sha256(identity, field=f"evidence {role} identity"),
        "file_sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _prepare_state_generation_commit_core(
    bound_attempt: object,
    inventory_binding: object,
    strict_referee_binding: object,
    runtime_completion: object,
    *,
    source_games_path: str | Path,
    state_split_path: str | Path,
    singleton_ledger_path: str | Path,
    timestamp_utc: str,
    active_seconds: int,
    bound_type: type,
    inventory_type: type,
    strict_referee_type: type,
    runtime_completion_type: type,
    pending_type: type,
    pending_token: object,
    store_type: type,
    store_contexts: Mapping[object, _StoreContext],
    active_contexts: Mapping[object, _ActiveContext],
    bound_contexts: Mapping[object, _BoundAttemptContext],
    inventory_contexts: Mapping[object, _InventoryContext],
    strict_referee_contexts: Mapping[object, _StrictRefereeContext],
    runtime_completion_contexts: Mapping[object, _RuntimeCompletionContext],
    pending_contexts: weakref.WeakKeyDictionary,
) -> object:
    if (
        type(bound_attempt) is not bound_type
        or type(inventory_binding) is not inventory_type
        or type(strict_referee_binding) is not strict_referee_type
        or type(runtime_completion) is not runtime_completion_type
    ):
        raise GovernanceStoreContractError(
            "state-generation prepare requires exact active/inventory capabilities"
        )
    bound_context = bound_contexts.get(bound_attempt)
    active_context = (
        None if bound_context is None else active_contexts.get(bound_context.active)
    )
    inventory_context = inventory_contexts.get(inventory_binding)
    strict_referee_context = strict_referee_contexts.get(strict_referee_binding)
    runtime_context = runtime_completion_contexts.get(runtime_completion)
    if (
        active_context is None
        or bound_context is None
        or bound_context.spent
        or inventory_context is None
        or strict_referee_context is None
        or runtime_context is None
        or runtime_context.spent
        or runtime_context.bound_attempt is not bound_attempt
        or runtime_context.store is not bound_context.store
        or runtime_context.runtime_stream_identity
        != bound_context.runtime_stream_identity
        or active_context.spent
    ):
        raise GovernanceStoreContractError(
            "state-generation active/inventory capability is absent or consumed"
        )
    store_context = _require_store_context(
        bound_context.store,
        store_type=store_type,
        contexts=store_contexts,
    )
    if (
        store_context.replay.state != "state_generation_running"
        or store_context.replay.head_event_identity
        != active_context.reservation_event_identity
        or store_context.replay.events[-1]["attempt_identity"]
        != active_context.attempt_identity
        or inventory_context.policy.domain != store_context.domain
        or strict_referee_context.domain != store_context.domain
        or strict_referee_context.attempt_identity != active_context.attempt_identity
        or store_context.runtime is None
        or store_context.runtime.host_preflight_identity
        != bound_context.host_preflight_identity
        or store_context.runtime.state_generator_session_identity
        != bound_context.state_generator_session_identity
    ):
        raise GovernanceStoreContractError(
            "state-generation prepare capability/replay domain differs"
        )
    seconds = _require_nonnegative_int(
        active_seconds,
        field="state-generation active_seconds",
    )
    # The attempt is consumed before touching artifacts.  Any read, validation,
    # or later persistence failure therefore cannot reuse this C5a attempt.
    active_context.spent = True
    bound_context.spent = True
    runtime_context.spent = True
    source_path, source_bytes, source_key = _read_plan_owned_artifact(
        source_games_path,
        store_context=store_context,
        field="source-games",
    )
    state_path, state_bytes, state_key = _read_plan_owned_artifact(
        state_split_path,
        store_context=store_context,
        field="state-split",
    )
    singleton_path, singleton_bytes, singleton_key = _read_plan_owned_artifact(
        singleton_ledger_path,
        store_context=store_context,
        field="singleton-ledger",
    )
    validated = _validate_artifacts(
        source_bytes,
        state_bytes,
        singleton_bytes,
        inventory=inventory_context,
        strict_referee=strict_referee_context,
        attempt_identity=active_context.attempt_identity,
    )
    evidence = {
        "inputs": [],
        "outputs": [
            _evidence_ref(
                "source-games",
                validated.source_games_identity,
                source_bytes,
            ),
            _evidence_ref(
                "state-split",
                validated.state_split_identity,
                state_bytes,
            ),
            _evidence_ref(
                "singleton-ledger",
                validated.singleton_ledger_identity,
                singleton_bytes,
            ),
        ],
        "checkpoint": None,
    }
    try:
        _event, event_bytes = build_state_generation_completed_event(
            store_context.replay,
            timestamp_utc=timestamp_utc,
            evidence=evidence,
            state_generation_games=validated.source_game_count,
            active_seconds=seconds,
        )
    except Exception as exc:
        raise GovernanceStoreContractError(
            "state-generation completion event preparation failed"
        ) from exc
    pending = pending_type(pending_token)
    pending_contexts[pending] = _PendingStateContext(
        store=bound_context.store,
        active=bound_attempt,
        inventory_binding=inventory_binding,
        strict_referee_binding=strict_referee_binding,
        runtime_completion=runtime_completion,
        source_games_path=source_path,
        state_split_path=state_path,
        singleton_ledger_path=singleton_path,
        source_games_sha256=_sha256_bytes(source_bytes),
        state_split_sha256=_sha256_bytes(state_bytes),
        singleton_ledger_sha256=_sha256_bytes(singleton_bytes),
        source_games_file_key=source_key,
        state_split_file_key=state_key,
        singleton_ledger_file_key=singleton_key,
        timestamp_utc=timestamp_utc,
        active_seconds=seconds,
        policy=inventory_context.policy,
        event_bytes=event_bytes,
    )
    return pending


def prepare_state_generation_commit(
    bound_attempt: RuntimeBoundStateGenerationAttempt,
    inventory_binding: ProductionAPosInventoryBinding,
    strict_referee_binding: ProductionStrictRefereeBinding,
    runtime_completion: ProductionStateGenerationRuntimeCompletion,
    *,
    source_games_path: str | Path,
    state_split_path: str | Path,
    singleton_ledger_path: str | Path,
    timestamp_utc: str,
    active_seconds: int,
) -> PendingStateGenerationCommit:
    """First-read and validate production artifacts without granting completion."""
    pending = _prepare_state_generation_commit_core(
        bound_attempt,
        inventory_binding,
        strict_referee_binding,
        runtime_completion,
        source_games_path=source_games_path,
        state_split_path=state_split_path,
        singleton_ledger_path=singleton_ledger_path,
        timestamp_utc=timestamp_utc,
        active_seconds=active_seconds,
        bound_type=RuntimeBoundStateGenerationAttempt,
        inventory_type=ProductionAPosInventoryBinding,
        strict_referee_type=ProductionStrictRefereeBinding,
        runtime_completion_type=ProductionStateGenerationRuntimeCompletion,
        pending_type=PendingStateGenerationCommit,
        pending_token=_PRODUCTION_PENDING_TOKEN,
        store_type=DurableGovernanceStore,
        store_contexts=_PRODUCTION_STORE_CONTEXTS,
        active_contexts=_PRODUCTION_ACTIVE_CONTEXTS,
        bound_contexts=_PRODUCTION_BOUND_ATTEMPT_CONTEXTS,
        inventory_contexts=_PRODUCTION_INVENTORY_CONTEXTS,
        strict_referee_contexts=_PRODUCTION_STRICT_REFEREE_CONTEXTS,
        runtime_completion_contexts=_PRODUCTION_RUNTIME_COMPLETION_CONTEXTS,
        pending_contexts=_PRODUCTION_PENDING_CONTEXTS,
    )
    assert type(pending) is PendingStateGenerationCommit
    return pending


def _test_prepare_state_generation_commit(
    bound_attempt: _TestRuntimeBoundStateGenerationAttempt,
    inventory_binding: _TestAPosInventoryBinding,
    strict_referee_binding: _TestStrictRefereeBinding,
    runtime_completion: _TestStateGenerationRuntimeCompletion,
    *,
    source_games_path: str | Path,
    state_split_path: str | Path,
    singleton_ledger_path: str | Path,
    timestamp_utc: str,
    active_seconds: int,
) -> _TestPendingStateGenerationCommit:
    pending = _prepare_state_generation_commit_core(
        bound_attempt,
        inventory_binding,
        strict_referee_binding,
        runtime_completion,
        source_games_path=source_games_path,
        state_split_path=state_split_path,
        singleton_ledger_path=singleton_ledger_path,
        timestamp_utc=timestamp_utc,
        active_seconds=active_seconds,
        bound_type=_TestRuntimeBoundStateGenerationAttempt,
        inventory_type=_TestAPosInventoryBinding,
        strict_referee_type=_TestStrictRefereeBinding,
        runtime_completion_type=_TestStateGenerationRuntimeCompletion,
        pending_type=_TestPendingStateGenerationCommit,
        pending_token=_TEST_PENDING_TOKEN,
        store_type=_TestDurableGovernanceStore,
        store_contexts=_TEST_STORE_CONTEXTS,
        active_contexts=_TEST_ACTIVE_CONTEXTS,
        bound_contexts=_TEST_BOUND_ATTEMPT_CONTEXTS,
        inventory_contexts=_TEST_INVENTORY_CONTEXTS,
        strict_referee_contexts=_TEST_STRICT_REFEREE_CONTEXTS,
        runtime_completion_contexts=_TEST_RUNTIME_COMPLETION_CONTEXTS,
        pending_contexts=_TEST_PENDING_CONTEXTS,
    )
    assert type(pending) is _TestPendingStateGenerationCommit
    return pending


def _second_read(
    path: Path,
    expected_key: tuple[int, int, int, int],
    expected_sha256: str,
    *,
    store_context: _StoreContext,
    field: str,
) -> bytes:
    observed_path, payload, observed_key = _read_plan_owned_artifact(
        path,
        store_context=store_context,
        field=field,
    )
    if (
        observed_path != path
        or observed_key != expected_key
        or _sha256_bytes(payload) != expected_sha256
    ):
        raise GovernanceStoreContractError(f"{field} changed after preparation")
    return payload


def _commit_state_generation_core(
    store: object,
    pending: object,
    *,
    store_type: type,
    pending_type: type,
    completion_type: type,
    completion_token: object,
    store_contexts: Mapping[object, _StoreContext],
    pending_contexts: Mapping[object, _PendingStateContext],
    inventory_contexts: Mapping[object, _InventoryContext],
    strict_referee_contexts: Mapping[object, _StrictRefereeContext],
    runtime_completion_contexts: Mapping[object, _RuntimeCompletionContext],
    completion_contexts: weakref.WeakKeyDictionary,
) -> object:
    if type(pending) is not pending_type:
        raise GovernanceStoreContractError(
            "state-generation commit requires an exact pending capability"
        )
    pending_context = pending_contexts.get(pending)
    if (
        pending_context is None
        or pending_context.spent
        or pending_context.store is not store
    ):
        raise GovernanceStoreContractError(
            "state-generation pending capability is absent, consumed, or cross-store"
        )
    store_context = _require_store_context(
        store,
        store_type=store_type,
        contexts=store_contexts,
    )
    inventory_context = inventory_contexts.get(pending_context.inventory_binding)
    strict_referee_context = strict_referee_contexts.get(
        pending_context.strict_referee_binding
    )
    runtime_context = runtime_completion_contexts.get(
        pending_context.runtime_completion
    )
    if (
        inventory_context is None
        or strict_referee_context is None
        or runtime_context is None
        or runtime_context.store is not store
        or runtime_context.bound_attempt is not pending_context.active
        or inventory_context.policy != pending_context.policy
        or strict_referee_context.attempt_identity
        != store_context.replay.events[-1]["attempt_identity"]
        or store_context.replay.state != "state_generation_running"
    ):
        raise GovernanceStoreContractError(
            "state-generation commit inventory/replay context differs"
        )
    # Persistence is one-shot.  A failed second read or COMMIT is not retryable.
    pending_context.spent = True
    source_bytes = _second_read(
        pending_context.source_games_path,
        pending_context.source_games_file_key,
        pending_context.source_games_sha256,
        store_context=store_context,
        field="source-games",
    )
    state_bytes = _second_read(
        pending_context.state_split_path,
        pending_context.state_split_file_key,
        pending_context.state_split_sha256,
        store_context=store_context,
        field="state-split",
    )
    singleton_bytes = _second_read(
        pending_context.singleton_ledger_path,
        pending_context.singleton_ledger_file_key,
        pending_context.singleton_ledger_sha256,
        store_context=store_context,
        field="singleton-ledger",
    )
    validated = _validate_artifacts(
        source_bytes,
        state_bytes,
        singleton_bytes,
        inventory=inventory_context,
        strict_referee=strict_referee_context,
        attempt_identity=strict_referee_context.attempt_identity,
    )
    decoded_event = decode_governance_ledger(pending_context.event_bytes)
    if len(decoded_event) != 1:
        raise GovernanceStoreContractError("pending completion event bytes differ")
    event = decoded_event[0]
    outputs = event["evidence"]["outputs"]
    expected_outputs = [
        _evidence_ref("source-games", validated.source_games_identity, source_bytes),
        _evidence_ref("state-split", validated.state_split_identity, state_bytes),
        _evidence_ref(
            "singleton-ledger",
            validated.singleton_ledger_identity,
            singleton_bytes,
        ),
    ]
    if outputs != tuple(_freeze(expected_outputs, field="expected outputs")):
        raise GovernanceStoreContractError("pending event artifact evidence differs")
    completion_body = {
        **_domain_envelope(
            store_context,
            schema_version=_COMPLETION_SCHEMA,
            attempt_identity=event["attempt_identity"],
            sequence=0,
            record_type="state-generation-completion",
            previous_record_identity=None,
        ),
        "experiment_id": _EXPERIMENT_ID,
        "proposal_identity": _PROPOSAL_IDENTITY,
        "profile_identity": _PROFILE_IDENTITY,
        "store_spec_identity": store_context.spec["spec_identity"],
        "plan_identity": store_context.plan["plan_identity"],
        "readiness_identity": store_context.spec["readiness_identity"],
        "managed_git_state_identity": store_context.spec["managed_git_state_identity"],
        "launch_path_binding_identity": store_context.spec[
            "launch_path_binding_identity"
        ],
        "authorization_identity": store_context.authorization["authorization_identity"],
        "authorization_consumption_identity": (
            store_context.replay.authorization_consumption_identity
        ),
        "attempt_identity": event["attempt_identity"],
        "reservation_event_identity": event["prerequisite_event_identity"],
        "completion_event_identity": event["event_identity"],
        "host_preflight_identity": store_context.runtime.host_preflight_identity,
        "state_generator_session_identity": (
            store_context.runtime.state_generator_session_identity
        ),
        "strict_referee_binding_identity": strict_referee_context.binding_identity,
        "a_pos_inventory_binding_identity": inventory_context.binding_identity,
        "runtime_evidence_stream_identity": runtime_context.runtime_stream_identity,
        "resource_snapshot_before_identity": runtime_context.resource_snapshot_before[
            "snapshot_identity"
        ],
        "resource_snapshot_after_identity": runtime_context.resource_snapshot_after[
            "snapshot_identity"
        ],
        "resource_stability_identity": runtime_context.stability_record_identity,
        "artifacts": [
            _artifact_contract_ref(
                "source-games",
                validated.source_games_identity,
                source_bytes,
                record_count=validated.source_game_count,
            ),
            _artifact_contract_ref(
                "state-split",
                validated.state_split_identity,
                state_bytes,
                record_count=validated.state_count,
                a_pos_verifier_identity=validated.verifier_identity,
            ),
            _artifact_contract_ref(
                "singleton-ledger",
                validated.singleton_ledger_identity,
                singleton_bytes,
                record_count=validated.singleton_count,
            ),
        ],
        "split_contract_identity": canonical_sha256(
            _split_contract_for_policy(pending_context.policy)
        ),
        "resource_observation": _thaw(event["resource_observation"]),
        "teacher_fields_present": False,
        "authoritative_storage": "sqlite-immutable-blob-second-read",
    }
    completion_identity = canonical_sha256(completion_body)
    completion_bytes = canonical_json_bytes(completion_body)
    replay = _commit_event_and_rows(
        store_context,
        pending_context.event_bytes,
        artifact_rows=(
            _artifact_row(
                "source-games",
                validated.source_games_identity,
                source_bytes,
            ),
            _artifact_row(
                "state-split",
                validated.state_split_identity,
                state_bytes,
            ),
            _artifact_row(
                "singleton-ledger",
                validated.singleton_ledger_identity,
                singleton_bytes,
            ),
        ),
        domain_rows=(
            _domain_record_row(runtime_context.after_record_bytes),
            _domain_record_row(runtime_context.stability_record_bytes),
            _domain_record_row(completion_bytes),
        ),
    )
    if replay.state != "state_generated":
        raise GovernanceStoreContractError("durable state-generation did not complete")
    completion = completion_type(completion_token)
    completion_contexts[completion] = _CompletionContext(
        store=store,
        completion_event_identity=event["event_identity"],
        attempt_identity=event["attempt_identity"],
        source_games_identity=validated.source_games_identity,
        state_split_identity=validated.state_split_identity,
        singleton_ledger_identity=validated.singleton_ledger_identity,
        completion_record_identity=completion_identity,
        runtime_evidence_stream_identity=runtime_context.runtime_stream_identity,
        resource_stability_identity=runtime_context.stability_record_identity,
    )
    return completion


def commit_state_generation(
    store: DurableGovernanceStore,
    pending: PendingStateGenerationCommit,
) -> ConfirmedStateGenerationCompletion:
    """Second-read, atomically persist, reopen, and confirm state generation."""
    completion = _commit_state_generation_core(
        store,
        pending,
        store_type=DurableGovernanceStore,
        pending_type=PendingStateGenerationCommit,
        completion_type=ConfirmedStateGenerationCompletion,
        completion_token=_PRODUCTION_COMPLETION_TOKEN,
        store_contexts=_PRODUCTION_STORE_CONTEXTS,
        pending_contexts=_PRODUCTION_PENDING_CONTEXTS,
        inventory_contexts=_PRODUCTION_INVENTORY_CONTEXTS,
        strict_referee_contexts=_PRODUCTION_STRICT_REFEREE_CONTEXTS,
        runtime_completion_contexts=_PRODUCTION_RUNTIME_COMPLETION_CONTEXTS,
        completion_contexts=_PRODUCTION_COMPLETION_CONTEXTS,
    )
    assert type(completion) is ConfirmedStateGenerationCompletion
    return completion


def _test_commit_state_generation(
    store: _TestDurableGovernanceStore,
    pending: _TestPendingStateGenerationCommit,
) -> _TestConfirmedStateGenerationCompletion:
    completion = _commit_state_generation_core(
        store,
        pending,
        store_type=_TestDurableGovernanceStore,
        pending_type=_TestPendingStateGenerationCommit,
        completion_type=_TestConfirmedStateGenerationCompletion,
        completion_token=_TEST_COMPLETION_TOKEN,
        store_contexts=_TEST_STORE_CONTEXTS,
        pending_contexts=_TEST_PENDING_CONTEXTS,
        inventory_contexts=_TEST_INVENTORY_CONTEXTS,
        strict_referee_contexts=_TEST_STRICT_REFEREE_CONTEXTS,
        runtime_completion_contexts=_TEST_RUNTIME_COMPLETION_CONTEXTS,
        completion_contexts=_TEST_COMPLETION_CONTEXTS,
    )
    assert type(completion) is _TestConfirmedStateGenerationCompletion
    return completion


def _artifact_refs_from_database(
    context: _StoreContext,
    *,
    roles: Sequence[str],
) -> list[dict[str, Any]]:
    connection = _open_read_connection(context.path)
    try:
        rows = connection.execute(
            "SELECT role, artifact_identity, artifact_bytes FROM artifacts "
            "ORDER BY role"
        ).fetchall()
    except sqlite3.Error as exc:
        raise GovernanceStoreContractError("artifact evidence read failed") from exc
    finally:
        connection.close()
    by_role: dict[str, dict[str, Any]] = {}
    for role, identity, payload in rows:
        if type(role) is not str or role in by_role:
            raise GovernanceStoreContractError("artifact evidence role differs")
        by_role[role] = _evidence_ref(
            role,
            identity,
            _require_sqlite_blob(payload, field=f"artifact evidence {role}"),
        )
    if any(role not in by_role for role in roles):
        raise GovernanceStoreContractError("durable artifact evidence is incomplete")
    return [by_role[role] for role in roles]


def _commit_state_freeze_core(
    store: object,
    completion: object,
    *,
    timestamp_utc: str,
    store_type: type,
    completion_type: type,
    freeze_type: type,
    freeze_token: object,
    store_contexts: Mapping[object, _StoreContext],
    completion_contexts: Mapping[object, _CompletionContext],
    freeze_contexts: weakref.WeakKeyDictionary,
) -> object:
    if type(completion) is not completion_type:
        raise GovernanceStoreContractError(
            "state freeze requires an exact confirmed completion"
        )
    completion_context = completion_contexts.get(completion)
    if (
        completion_context is None
        or completion_context.spent
        or completion_context.store is not store
    ):
        raise GovernanceStoreContractError(
            "state completion is absent, consumed, or cross-store"
        )
    store_context = _require_store_context(
        store,
        store_type=store_type,
        contexts=store_contexts,
    )
    if (
        store_context.replay.state != "state_generated"
        or store_context.replay.head_event_identity
        != completion_context.completion_event_identity
    ):
        raise GovernanceStoreContractError("state freeze durable predecessor differs")
    # Freeze is one-shot even when the following SQLite transaction fails.
    completion_context.spent = True
    artifact_refs = _artifact_refs_from_database(
        store_context,
        roles=("source-games", "state-split", "singleton-ledger"),
    )
    by_role = {item["role"]: item for item in artifact_refs}
    if (
        by_role["source-games"]["identity"] != completion_context.source_games_identity
        or by_role["state-split"]["identity"] != completion_context.state_split_identity
        or by_role["singleton-ledger"]["identity"]
        != completion_context.singleton_ledger_identity
    ):
        raise GovernanceStoreContractError("state freeze artifact identity differs")
    receipt_body = {
        **_domain_envelope(
            store_context,
            schema_version=_FREEZE_RECEIPT_SCHEMA,
            attempt_identity=completion_context.attempt_identity,
            sequence=1,
            record_type="state-freeze",
            previous_record_identity=completion_context.completion_record_identity,
        ),
        "experiment_id": _EXPERIMENT_ID,
        "proposal_identity": _PROPOSAL_IDENTITY,
        "profile_identity": _PROFILE_IDENTITY,
        "store_spec_identity": store_context.spec["spec_identity"],
        "plan_identity": store_context.plan["plan_identity"],
        "readiness_identity": store_context.spec["readiness_identity"],
        "authorization_identity": store_context.authorization["authorization_identity"],
        "authorization_consumption_identity": (
            store_context.replay.authorization_consumption_identity
        ),
        "attempt_identity": completion_context.attempt_identity,
        "completion_event_identity": completion_context.completion_event_identity,
        "completion_record_identity": completion_context.completion_record_identity,
        "source_games_identity": completion_context.source_games_identity,
        "state_split_identity": completion_context.state_split_identity,
        "singleton_ledger_identity": completion_context.singleton_ledger_identity,
        "runtime_evidence_stream_identity": (
            completion_context.runtime_evidence_stream_identity
        ),
        "resource_stability_identity": completion_context.resource_stability_identity,
        "teacher_fields_present": False,
        "teacher_may_start_only_after_this_event": True,
        "authoritative_artifacts": "sqlite-immutable-blobs",
    }
    receipt_identity = canonical_sha256(receipt_body)
    receipt_bytes = canonical_json_bytes(receipt_body)
    try:
        event, event_bytes = build_state_frozen_event(
            store_context.replay,
            timestamp_utc=timestamp_utc,
            evidence={
                "inputs": artifact_refs,
                "outputs": [
                    _evidence_ref(
                        "state-freeze-receipt",
                        receipt_identity,
                        receipt_bytes,
                    )
                ],
                "checkpoint": None,
            },
        )
    except Exception as exc:
        raise GovernanceStoreContractError(
            "state-freeze event preparation failed"
        ) from exc
    replay = _commit_event_and_rows(
        store_context,
        event_bytes,
        artifact_rows=(
            _artifact_row(
                "state-freeze-receipt",
                receipt_identity,
                receipt_bytes,
            ),
        ),
        domain_rows=(_domain_record_row(receipt_bytes),),
    )
    if replay.state != "state_frozen":
        raise GovernanceStoreContractError("durable state freeze did not complete")
    freeze = freeze_type(freeze_token)
    freeze_contexts[freeze] = _FreezeContext(
        store=store,
        freeze_event_identity=event["event_identity"],
        state_split_identity=completion_context.state_split_identity,
        freeze_receipt_identity=receipt_identity,
    )
    return freeze


def commit_state_freeze(
    store: DurableGovernanceStore,
    completion: ConfirmedStateGenerationCompletion,
    *,
    timestamp_utc: str,
) -> DurableStateFreezeBinding:
    """Persist a separate state-freeze receipt/event and reopen before binding."""
    freeze = _commit_state_freeze_core(
        store,
        completion,
        timestamp_utc=timestamp_utc,
        store_type=DurableGovernanceStore,
        completion_type=ConfirmedStateGenerationCompletion,
        freeze_type=DurableStateFreezeBinding,
        freeze_token=_PRODUCTION_FREEZE_TOKEN,
        store_contexts=_PRODUCTION_STORE_CONTEXTS,
        completion_contexts=_PRODUCTION_COMPLETION_CONTEXTS,
        freeze_contexts=_PRODUCTION_FREEZE_CONTEXTS,
    )
    assert type(freeze) is DurableStateFreezeBinding
    return freeze


def _test_commit_state_freeze(
    store: _TestDurableGovernanceStore,
    completion: _TestConfirmedStateGenerationCompletion,
    *,
    timestamp_utc: str,
) -> _TestDurableStateFreezeBinding:
    freeze = _commit_state_freeze_core(
        store,
        completion,
        timestamp_utc=timestamp_utc,
        store_type=_TestDurableGovernanceStore,
        completion_type=_TestConfirmedStateGenerationCompletion,
        freeze_type=_TestDurableStateFreezeBinding,
        freeze_token=_TEST_FREEZE_TOKEN,
        store_contexts=_TEST_STORE_CONTEXTS,
        completion_contexts=_TEST_COMPLETION_CONTEXTS,
        freeze_contexts=_TEST_FREEZE_CONTEXTS,
    )
    assert type(freeze) is _TestDurableStateFreezeBinding
    return freeze


def _restore_durable_state_freeze_core(
    store: object,
    *,
    store_type: type,
    freeze_type: type,
    freeze_token: object,
    store_contexts: Mapping[object, _StoreContext],
    freeze_contexts: weakref.WeakKeyDictionary,
) -> object:
    context = _require_store_context(
        store,
        store_type=store_type,
        contexts=store_contexts,
    )
    if context.replay.state != "state_frozen":
        raise GovernanceStoreContractError(
            "durable state freeze can only be restored from a frozen store"
        )
    connection = _open_read_connection(context.path)
    try:
        row = connection.execute(
            "SELECT artifact_identity, artifact_bytes FROM artifacts "
            "WHERE role='state-freeze-receipt'"
        ).fetchone()
    except sqlite3.Error as exc:
        raise GovernanceStoreContractError("state-freeze restore read failed") from exc
    finally:
        connection.close()
    if row is None:
        raise GovernanceStoreContractError("state-freeze restore receipt is missing")
    receipt_identity = _require_sha256(
        row[0], field="restored state-freeze receipt identity"
    )
    payload = _require_sqlite_blob(row[1], field="restored state-freeze receipt")
    receipt = _strict_json_object(payload, field="restored state-freeze receipt")
    _require_exact_keys(
        receipt,
        _FREEZE_RECEIPT_KEYS,
        field="restored state-freeze receipt",
    )
    if (
        canonical_sha256(receipt) != receipt_identity
        or _sha256_bytes(payload) != receipt_identity
    ):
        raise GovernanceStoreContractError("restored state-freeze receipt differs")
    freeze_events = tuple(
        event
        for event in context.replay.events
        if event["event_type"] == "state_frozen"
    )
    if len(freeze_events) != 1:
        raise GovernanceStoreContractError("restored state-freeze event differs")
    freeze = freeze_type(freeze_token)
    freeze_contexts[freeze] = _FreezeContext(
        store=store,
        freeze_event_identity=freeze_events[0]["event_identity"],
        state_split_identity=_require_sha256(
            receipt["state_split_identity"],
            field="restored state-split identity",
        ),
        freeze_receipt_identity=receipt_identity,
    )
    return freeze


def restore_durable_state_freeze(
    store: DurableGovernanceStore,
) -> DurableStateFreezeBinding:
    """Reissue a production freeze binding from a fully verified frozen store."""
    freeze = _restore_durable_state_freeze_core(
        store,
        store_type=DurableGovernanceStore,
        freeze_type=DurableStateFreezeBinding,
        freeze_token=_PRODUCTION_FREEZE_TOKEN,
        store_contexts=_PRODUCTION_STORE_CONTEXTS,
        freeze_contexts=_PRODUCTION_FREEZE_CONTEXTS,
    )
    assert type(freeze) is DurableStateFreezeBinding
    return freeze


def _test_restore_durable_state_freeze(
    store: _TestDurableGovernanceStore,
) -> _TestDurableStateFreezeBinding:
    freeze = _restore_durable_state_freeze_core(
        store,
        store_type=_TestDurableGovernanceStore,
        freeze_type=_TestDurableStateFreezeBinding,
        freeze_token=_TEST_FREEZE_TOKEN,
        store_contexts=_TEST_STORE_CONTEXTS,
        freeze_contexts=_TEST_FREEZE_CONTEXTS,
    )
    assert type(freeze) is _TestDurableStateFreezeBinding
    return freeze


def _verify_freeze_core(
    store: object,
    binding: object,
    *,
    store_type: type,
    freeze_type: type,
    store_contexts: Mapping[object, _StoreContext],
    freeze_contexts: Mapping[object, _FreezeContext],
) -> str:
    if type(binding) is not freeze_type:
        raise GovernanceStoreContractError("state-freeze binding type/domain differs")
    freeze_context = freeze_contexts.get(binding)
    if freeze_context is None or freeze_context.store is not store:
        raise GovernanceStoreContractError(
            "state-freeze binding is absent or cross-store"
        )
    store_context = _require_store_context(
        store,
        store_type=store_type,
        contexts=store_contexts,
    )
    if (
        store_context.replay.state != "state_frozen"
        or store_context.replay.head_event_identity
        != freeze_context.freeze_event_identity
    ):
        raise GovernanceStoreContractError("durable state-freeze replay differs")
    connection = _open_read_connection(store_context.path)
    try:
        artifact = connection.execute(
            "SELECT artifact_identity, artifact_bytes FROM artifacts "
            "WHERE role='state-freeze-receipt'"
        ).fetchone()
        record = connection.execute(
            "SELECT record_identity, record_bytes FROM domain_records "
            "WHERE record_type='state-freeze'"
        ).fetchone()
    except sqlite3.Error as exc:
        raise GovernanceStoreContractError("state-freeze receipt read failed") from exc
    finally:
        connection.close()
    if artifact is None or record is None:
        raise GovernanceStoreContractError("state-freeze receipt is missing")
    artifact_identity = _require_sha256(
        artifact[0],
        field="state-freeze artifact identity",
    )
    artifact_payload = _require_sqlite_blob(
        artifact[1],
        field="state-freeze artifact",
    )
    record_identity = _require_sha256(
        record[0],
        field="state-freeze record identity",
    )
    record_payload = _require_sqlite_blob(
        record[1],
        field="state-freeze record",
    )
    if (
        artifact_identity != freeze_context.freeze_receipt_identity
        or record_identity != freeze_context.freeze_receipt_identity
        or artifact_payload != record_payload
    ):
        raise GovernanceStoreContractError("state-freeze receipt identity differs")
    receipt = _strict_json_object(artifact_payload, field="state-freeze receipt")
    if (
        canonical_sha256(receipt) != freeze_context.freeze_receipt_identity
        or receipt.get("state_split_identity") != freeze_context.state_split_identity
        or receipt.get("teacher_fields_present") is not False
        or receipt.get("teacher_may_start_only_after_this_event") is not True
    ):
        raise GovernanceStoreContractError("state-freeze receipt contract differs")
    return freeze_context.state_split_identity


def verify_durable_state_freeze(
    store: DurableGovernanceStore,
    binding: DurableStateFreezeBinding,
) -> str:
    """Reopen and verify the exact production durable state-freeze binding."""
    return _verify_freeze_core(
        store,
        binding,
        store_type=DurableGovernanceStore,
        freeze_type=DurableStateFreezeBinding,
        store_contexts=_PRODUCTION_STORE_CONTEXTS,
        freeze_contexts=_PRODUCTION_FREEZE_CONTEXTS,
    )


def _test_verify_durable_state_freeze(
    store: _TestDurableGovernanceStore,
    binding: _TestDurableStateFreezeBinding,
) -> str:
    return _verify_freeze_core(
        store,
        binding,
        store_type=_TestDurableGovernanceStore,
        freeze_type=_TestDurableStateFreezeBinding,
        store_contexts=_TEST_STORE_CONTEXTS,
        freeze_contexts=_TEST_FREEZE_CONTEXTS,
    )
