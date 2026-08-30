"""Qualified, read-only loading for the classical ``A_pos`` student.

This slice deliberately contains no move selection, product routing, fallback,
database, advisor, encoder, or teacher path.  A complete supervised checkpoint
can become an opaque inference handle only after a separately issued,
single-use qualification binding matches the exact file and every frozen
identity checked here.
"""

from __future__ import annotations

import hashlib
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch

from learned_ai.models.scaffolded_encoder import VALUE_INPUT_DIM
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.training.checkpoint_envelope import CheckpointError, load_checkpoint
from learned_ai.training.classical_a_pos_corpus import CORPUS_SCHEMA
from learned_ai.training.classical_a_pos_supervised import (
    FEATURE_SCHEMA,
    LABEL_SCHEMA,
    PRODUCTION_EPOCHS,
    PRODUCTION_POLICY_HIDDEN,
    PRODUCTION_SEEDS,
    PRODUCTION_TRAIN_EXAMPLES,
    PRODUCTION_UPDATES_PER_SEED,
    SEED_COMPLETE_ROLE,
    TRAINER_ID,
    SupervisedState,
    SupervisedTrainingError,
    _validate_checkpoint_permutation_state,
    _validate_serialized_optimizer_state,
    canonical_training_state_identity,
    production_loop_config,
)
from learned_ai.training.run_contract import canonical_sha256


_QUALIFICATION_TOKEN = object()
_HANDLE_TOKEN = object()
_ASSET_KEYS = {
    "corpus",
    "split",
    "plan",
    "training_records",
    "encoded_payload",
}
_IMPLEMENTATION_KEYS = {
    "trainer",
    "optimizer",
    "seed",
    "training_kind",
    "lineage_kind",
    "start_mode",
    "expected_initial_model_state_identity",
    "purpose",
}
_MODEL_CONFIG = {
    "move_feat_dim": 62,
    "value_input_dim": VALUE_INPUT_DIM,
    "policy_hidden": PRODUCTION_POLICY_HIDDEN,
    "value_hidden": (),
    "dropout": 0.0,
}
_COMPLETE_CURSOR = {
    "epoch": PRODUCTION_EPOCHS,
    "batch_in_epoch": 0,
    "update_count": PRODUCTION_UPDATES_PER_SEED,
    "sample_cursor": 0,
    "permutation": None,
    "completed": True,
}


class StudentQualificationError(RuntimeError):
    """A checkpoint or opaque qualification failed closed."""


class _OpaqueCapability:
    __slots__ = ("__weakref__",)

    _creation_token: object
    _kind: str

    def __init__(self, token: object) -> None:
        if token is not self._creation_token:
            raise StudentQualificationError(
                f"{self._kind} must be issued by its controlled issuer"
            )

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        raise TypeError(f"{self._kind} cannot be serialized")

    def __copy__(self) -> Any:
        raise TypeError(f"{self._kind} cannot be copied")

    def __deepcopy__(self, memo: Any) -> Any:
        del memo
        raise TypeError(f"{self._kind} cannot be copied")


class QualifiedCompleteSeedBinding(_OpaqueCapability):
    """Single-use opaque authority for one exact qualified checkpoint file."""

    __slots__ = ()

    _creation_token = _QUALIFICATION_TOKEN
    _kind = "qualified complete seed binding"


class QualifiedStudentHandle(_OpaqueCapability):
    """Opaque frozen inference capability produced by the strict loader."""

    __slots__ = ()

    _creation_token = _HANDLE_TOKEN
    _kind = "qualified student handle"


@dataclass(frozen=True)
class _QualificationContext:
    resolved_path: str
    file_size: int
    file_sha256: str
    checkpoint_file_identity: str
    payload_size: int
    payload_sha256: str
    checkpoint_id: str
    run_id: str
    experiment_id: str
    plan_identity: str
    corpus_identity: str
    split_identity: str
    training_records_identity: str
    encoded_payload_identity: str
    model_identity: str
    seed: int
    qualification_identity: str


@dataclass(frozen=True)
class _QualifiedStudentContext:
    model: ScaffoldedPolicyNet
    resolved_path: str
    file_size: int
    file_sha256: str
    checkpoint_file_identity: str
    payload_size: int
    payload_sha256: str
    checkpoint_id: str
    run_id: str
    experiment_id: str
    plan_identity: str
    corpus_identity: str
    split_identity: str
    training_records_identity: str
    encoded_payload_identity: str
    model_identity: str
    seed: int
    qualification_identity: str


_QUALIFICATION_CONTEXTS: weakref.WeakKeyDictionary[
    QualifiedCompleteSeedBinding,
    _QualificationContext,
] = weakref.WeakKeyDictionary()
_QUALIFIED_STUDENT_CONTEXTS: weakref.WeakKeyDictionary[
    QualifiedStudentHandle,
    _QualifiedStudentContext,
] = weakref.WeakKeyDictionary()


def _file_identity(path: str | Path) -> tuple[str, int, str, str]:
    try:
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_file():
            raise StudentQualificationError("qualified checkpoint path is not a file")
        digest = hashlib.sha256()
        size = 0
        with resolved.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    except StudentQualificationError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise StudentQualificationError(
            "qualified checkpoint path cannot be read"
        ) from exc
    sha256 = digest.hexdigest()
    identity = canonical_sha256(
        {
            "schema": "nmm.qualified-student-file.v1",
            "resolved_path": str(resolved),
            "size": size,
            "sha256": sha256,
        }
    )
    return str(resolved), size, sha256, identity


def _qualification_record(context: _QualificationContext) -> dict[str, Any]:
    return {
        field.name: getattr(context, field.name)
        for field in fields(_QualificationContext)
        if field.name != "qualification_identity"
    }


def _qualification_identity(context: _QualificationContext) -> str:
    return canonical_sha256(
        {
            "schema": "nmm.qualified-complete-seed-binding.v1",
            "binding": _qualification_record(context),
        }
    )


def _mapping_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


def _issue_test_qualified_complete_seed_binding(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> QualifiedCompleteSeedBinding:
    """Issue a test-only binding; there is intentionally no production issuer.

    The test issuer records observed identities without certifying their
    semantics.  ``load_qualified_complete_student`` remains the independent
    fail-closed semantic boundary.
    """

    try:
        resolved, file_size, file_sha256, file_identity = _file_identity(path)
        envelope = load_checkpoint(resolved, map_location="cpu")
        descriptor = envelope.descriptor
        payload = envelope.payload
        assets = descriptor.asset_identities
        cache = payload.data_state.get("cache")
        state_identities = _mapping_value(cache, "canonical_state_identities")
        implementation = descriptor.implementation
        seed_raw = _mapping_value(implementation, "seed")
        try:
            seed = int(seed_raw)
        except (TypeError, ValueError):
            seed = -1
        context = _QualificationContext(
            resolved_path=resolved,
            file_size=file_size,
            file_sha256=file_sha256,
            checkpoint_file_identity=file_identity,
            payload_size=envelope.payload_size,
            payload_sha256=envelope.payload_sha256,
            checkpoint_id=descriptor.checkpoint_id,
            run_id=descriptor.run_id,
            experiment_id=descriptor.experiment_id,
            plan_identity=_mapping_value(assets, "plan"),
            corpus_identity=_mapping_value(assets, "corpus"),
            split_identity=_mapping_value(assets, "split"),
            training_records_identity=_mapping_value(assets, "training_records"),
            encoded_payload_identity=_mapping_value(assets, "encoded_payload"),
            model_identity=_mapping_value(state_identities, "model"),
            seed=seed,
            qualification_identity="",
        )
        requested = dict(overrides or {})
        allowed = {
            field.name
            for field in fields(_QualificationContext)
            if field.name != "qualification_identity"
        }
        if set(requested) - allowed:
            raise StudentQualificationError(
                "test qualification override fields are unsupported"
            )
        if requested:
            context = replace(context, **requested)
        context = replace(
            context,
            qualification_identity=_qualification_identity(context),
        )
    except StudentQualificationError:
        raise
    except (CheckpointError, KeyError, TypeError, ValueError) as exc:
        raise StudentQualificationError(
            "test qualification could not inspect the checkpoint"
        ) from exc
    binding = QualifiedCompleteSeedBinding(_QUALIFICATION_TOKEN)
    _QUALIFICATION_CONTEXTS[binding] = context
    return binding


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StudentQualificationError(f"{field} identity is not a SHA-256")
    return value


def _require_exact_mapping(
    value: Any,
    expected: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StudentQualificationError(f"qualified checkpoint {field} differs")
    observed = dict(value)
    reference = dict(expected)
    try:
        identities_match = canonical_training_state_identity(
            observed
        ) == canonical_training_state_identity(reference)
    except SupervisedTrainingError:
        identities_match = False
    if observed != reference or not identities_match:
        raise StudentQualificationError(f"qualified checkpoint {field} differs")
    return observed


def _require_finite_model_state(value: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise StudentQualificationError("qualified model state is missing")
    for name, tensor in value.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise StudentQualificationError("qualified model state tensors differ")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise StudentQualificationError(
                "qualified model state must contain only finite tensors"
            )
    return value


def _new_exact_model(*, initialization_seed: int | None = None) -> ScaffoldedPolicyNet:
    caller_rng = torch.get_rng_state()
    try:
        if initialization_seed is not None:
            torch.manual_seed(initialization_seed)
        return ScaffoldedPolicyNet(
            move_feat_dim=62,
            value_input_dim=VALUE_INPUT_DIM,
            policy_hidden=PRODUCTION_POLICY_HIDDEN,
            value_hidden=(),
            dropout=0.0,
        )
    finally:
        torch.set_rng_state(caller_rng)


def _validate_initial_lineage_identity(seed: int, observed: Any) -> str:
    identity = _require_sha256(
        observed,
        field="expected initial model state",
    )
    expected = canonical_training_state_identity(
        _new_exact_model(initialization_seed=seed).state_dict()
    )
    if identity != expected:
        raise StudentQualificationError(
            "fresh supervised seed initial model identity differs"
        )
    return identity


def _validate_complete_envelope(
    envelope: Any,
    qualification: _QualificationContext,
) -> tuple[ScaffoldedPolicyNet, str, int]:
    descriptor = envelope.descriptor
    payload = envelope.payload
    if descriptor.role != SEED_COMPLETE_ROLE:
        raise StudentQualificationError(
            "qualified checkpoint must have the supervised complete role"
        )
    if descriptor.save_reason != "supervised-complete":
        raise StudentQualificationError("qualified complete save reason differs")
    if descriptor.parent_checkpoint_id is not None:
        raise StudentQualificationError(
            "qualified complete parent checkpoint identity differs"
        )
    implementation = descriptor.implementation
    if not isinstance(implementation, Mapping) or set(implementation) != (
        _IMPLEMENTATION_KEYS
    ):
        raise StudentQualificationError("qualified implementation fields differ")
    if implementation["purpose"] != "seed":
        raise StudentQualificationError("qualified checkpoint purpose differs")
    if implementation["lineage_kind"] != "fresh_supervised_seed":
        raise StudentQualificationError("qualified checkpoint lineage differs")
    if implementation["start_mode"] != "fresh":
        raise StudentQualificationError("qualified checkpoint lineage start differs")
    if implementation["trainer"] != TRAINER_ID:
        raise StudentQualificationError("qualified checkpoint trainer differs")
    if implementation["optimizer"] != "Adam-policy_mlp-only":
        raise StudentQualificationError("qualified checkpoint Adam contract differs")
    if implementation["training_kind"] != "offline-supervised-not-rl":
        raise StudentQualificationError("qualified training kind differs")
    seed_text = implementation["seed"]
    try:
        seed = int(seed_text)
    except (TypeError, ValueError) as exc:
        raise StudentQualificationError("qualified seed is invalid") from exc
    if seed not in PRODUCTION_SEEDS or seed_text != str(seed):
        raise StudentQualificationError("qualified seed is not frozen")
    _validate_initial_lineage_identity(
        seed,
        implementation["expected_initial_model_state_identity"],
    )

    config = production_loop_config(seed)
    config_identity = canonical_sha256(config.to_dict())
    if descriptor.config_sha256 != config_identity:
        raise StudentQualificationError("qualified trainer config identity differs")
    if descriptor.feature_schema_version != FEATURE_SCHEMA:
        raise StudentQualificationError("qualified feature schema differs")
    if descriptor.label_schema_version != LABEL_SCHEMA:
        raise StudentQualificationError("qualified label schema differs")
    if dict(descriptor.database_schema_versions) != {"corpus": CORPUS_SCHEMA}:
        raise StudentQualificationError("qualified corpus schema differs")

    assets = descriptor.asset_identities
    if not isinstance(assets, Mapping) or set(assets) != _ASSET_KEYS:
        raise StudentQualificationError("qualified asset identity fields differ")
    for name, identity in assets.items():
        _require_sha256(identity, field=name)

    try:
        cursor = SupervisedState.from_dict(payload.data_state["cursor"])
    except (KeyError, TypeError, SupervisedTrainingError) as exc:
        raise StudentQualificationError("qualified complete cursor differs") from exc
    if cursor.to_dict() != _COMPLETE_CURSOR:
        raise StudentQualificationError("qualified complete cursor differs")

    model = _new_exact_model()
    model_config = payload.trainer_state.get("model_config")
    _require_exact_mapping(model_config, _MODEL_CONFIG, field="model config")
    model_state = _require_finite_model_state(payload.model_state)
    try:
        model.load_state_dict(model_state, strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise StudentQualificationError(
            "qualified model state is incompatible with the exact model config"
        ) from exc
    if any(
        not bool(torch.isfinite(tensor).all())
        for tensor in model.state_dict().values()
        if tensor.is_floating_point() or tensor.is_complex()
    ):
        raise StudentQualificationError("qualified model state is non-finite")
    model_identity = canonical_training_state_identity(model.state_dict())

    expected_curriculum = {
        "training_semantics": config.to_dict()["training_semantics"],
        "purpose": "seed",
        "epoch": PRODUCTION_EPOCHS,
        "batch_in_epoch": 0,
    }
    expected_recovery = {
        "exact_resume": True,
        "cursor": _COMPLETE_CURSOR,
    }
    trainer_state = payload.trainer_state
    if type(trainer_state.get("game_count")) is not int or (
        trainer_state["game_count"] != 0
    ):
        raise StudentQualificationError("qualified complete trainer game count differs")
    for name in ("batch_count", "update_count"):
        if type(trainer_state.get(name)) is not int or trainer_state[name] != (
            PRODUCTION_UPDATES_PER_SEED
        ):
            raise StudentQualificationError(
                "qualified complete trainer update counts differ"
            )
    if type(trainer_state.get("difficulty")) is not int or (
        trainer_state["difficulty"] != 9
    ):
        raise StudentQualificationError("qualified complete difficulty differs")
    if type(trainer_state.get("temperature")) is not float or (
        trainer_state["temperature"] != 1.0
    ):
        raise StudentQualificationError("qualified complete temperature differs")
    _require_exact_mapping(
        trainer_state.get("rolling_metrics"),
        {},
        field="complete metrics",
    )
    _require_exact_mapping(
        trainer_state.get("curriculum"),
        expected_curriculum,
        field="complete curriculum",
    )
    _require_exact_mapping(
        trainer_state.get("target_network"),
        {"enabled": False},
        field="complete target-network state",
    )
    _require_exact_mapping(
        trainer_state.get("recovery_state"),
        expected_recovery,
        field="complete recovery state",
    )

    expected_lineage = {
        "lineage_kind": "fresh_supervised_seed",
        "seed": seed,
        "expected_initial_model_state_identity": implementation[
            "expected_initial_model_state_identity"
        ],
        "start_mode": "fresh",
        "purpose": "seed",
    }
    buckets = payload.data_state.get("buckets")
    expected_buckets = {
        "corpus_identity": assets["corpus"],
        "split_identity": assets["split"],
        "plan_identity": assets["plan"],
        "config_identity": config_identity,
        "seed": seed,
        "sample_count": PRODUCTION_TRAIN_EXAMPLES,
        "training_records_identity": assets["training_records"],
        "encoded_payload_identity": assets["encoded_payload"],
        "fresh_lineage": expected_lineage,
        "purpose": "seed",
    }
    _require_exact_mapping(buckets, expected_buckets, field="identity buckets")
    if payload.data_state.get("consumed_snapshots") != [assets["corpus"]]:
        raise StudentQualificationError("qualified corpus snapshot identity differs")
    _require_exact_mapping(
        payload.data_state.get("mutable_assets"),
        {},
        field="mutable assets",
    )

    if payload.optimizer_state is None:
        raise StudentQualificationError("qualified complete Adam state is missing")
    if payload.scheduler_state is not None or payload.scaler_state is not None:
        raise StudentQualificationError(
            "qualified complete scheduler/scaler must be absent"
        )
    try:
        _validate_serialized_optimizer_state(
            model,
            payload.optimizer_state,
            expected_updates=PRODUCTION_UPDATES_PER_SEED,
        )
    except SupervisedTrainingError as exc:
        raise StudentQualificationError(
            "qualified complete Adam state must have step 2,240 for every policy "
            f"parameter: {exc}"
        ) from exc

    cache = payload.data_state.get("cache")
    if not isinstance(cache, Mapping) or set(cache) != {"canonical_state_identities"}:
        raise StudentQualificationError("qualified canonical identities are missing")
    identities = cache["canonical_state_identities"]
    if not isinstance(identities, Mapping) or set(identities) != {
        "model",
        "optimizer",
        "rng",
        "cursor",
    }:
        raise StudentQualificationError("qualified canonical identity fields differ")
    observed_identities = {
        "model": model_identity,
        "optimizer": canonical_training_state_identity(payload.optimizer_state),
        "rng": canonical_training_state_identity(payload.rng_state),
        "cursor": canonical_training_state_identity(cursor.to_dict()),
    }
    if dict(identities) != observed_identities:
        raise StudentQualificationError(
            "qualified descriptor/payload canonical identity differs"
        )
    try:
        _validate_checkpoint_permutation_state(
            payload.rng_state,
            cursor,
            sample_count=PRODUCTION_TRAIN_EXAMPLES,
            config=config,
        )
    except SupervisedTrainingError as exc:
        raise StudentQualificationError(
            "qualified complete permutation RNG differs"
        ) from exc

    if descriptor.checkpoint_id != (
        f"{descriptor.run_id}:update:{PRODUCTION_UPDATES_PER_SEED}"
    ):
        raise StudentQualificationError("qualified checkpoint identity differs")
    if seed != qualification.seed:
        raise StudentQualificationError("qualified seed differs from qualification")
    cross_checks = {
        "checkpoint": (descriptor.checkpoint_id, qualification.checkpoint_id),
        "run": (descriptor.run_id, qualification.run_id),
        "experiment": (descriptor.experiment_id, qualification.experiment_id),
        "plan": (assets["plan"], qualification.plan_identity),
        "corpus": (assets["corpus"], qualification.corpus_identity),
        "split": (assets["split"], qualification.split_identity),
        "training records": (
            assets["training_records"],
            qualification.training_records_identity,
        ),
        "encoded payload": (
            assets["encoded_payload"],
            qualification.encoded_payload_identity,
        ),
        "model": (model_identity, qualification.model_identity),
    }
    for name, (observed, qualified) in cross_checks.items():
        if observed != qualified:
            raise StudentQualificationError(
                f"qualified {name} identity differs from qualification"
            )

    model.cpu()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, model_identity, seed


def load_qualified_complete_student(
    path: str | Path,
    qualification: QualifiedCompleteSeedBinding,
) -> QualifiedStudentHandle:
    """Consume one qualification and return an opaque frozen CPU handle."""

    if type(qualification) is not QualifiedCompleteSeedBinding:
        raise StudentQualificationError(
            "qualification must be an unused issued complete-seed binding"
        )
    context = _QUALIFICATION_CONTEXTS.pop(qualification, None)
    if context is None:
        raise StudentQualificationError(
            "qualification is not an unused issued complete-seed binding"
        )
    if _qualification_identity(context) != context.qualification_identity:
        raise StudentQualificationError("qualification identity is invalid")
    resolved, file_size, file_sha256, file_identity = _file_identity(path)
    if resolved != context.resolved_path:
        raise StudentQualificationError("qualification checkpoint path differs")
    if (
        file_size != context.file_size
        or file_sha256 != context.file_sha256
        or file_identity != context.checkpoint_file_identity
    ):
        raise StudentQualificationError("qualification checkpoint file differs")
    try:
        envelope = load_checkpoint(resolved, map_location="cpu")
    except CheckpointError as exc:
        raise StudentQualificationError(
            "qualified checkpoint envelope cannot be loaded"
        ) from exc
    after_resolved, after_size, after_sha256, after_identity = _file_identity(resolved)
    if (
        after_resolved != resolved
        or after_size != file_size
        or after_sha256 != file_sha256
        or after_identity != file_identity
        or envelope.payload_size != context.payload_size
        or envelope.payload_sha256 != context.payload_sha256
    ):
        raise StudentQualificationError("qualification checkpoint file changed")
    try:
        model, model_identity, seed = _validate_complete_envelope(
            envelope,
            context,
        )
    except StudentQualificationError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise StudentQualificationError(
            "qualified complete checkpoint structure differs"
        ) from exc
    handle = QualifiedStudentHandle(_HANDLE_TOKEN)
    _QUALIFIED_STUDENT_CONTEXTS[handle] = _QualifiedStudentContext(
        model=model,
        resolved_path=resolved,
        file_size=file_size,
        file_sha256=file_sha256,
        checkpoint_file_identity=file_identity,
        payload_size=envelope.payload_size,
        payload_sha256=envelope.payload_sha256,
        checkpoint_id=envelope.descriptor.checkpoint_id,
        run_id=envelope.descriptor.run_id,
        experiment_id=envelope.descriptor.experiment_id,
        plan_identity=context.plan_identity,
        corpus_identity=context.corpus_identity,
        split_identity=context.split_identity,
        training_records_identity=context.training_records_identity,
        encoded_payload_identity=context.encoded_payload_identity,
        model_identity=model_identity,
        seed=seed,
        qualification_identity=context.qualification_identity,
    )
    return handle


__all__ = [
    "QualifiedCompleteSeedBinding",
    "QualifiedStudentHandle",
    "StudentQualificationError",
    "load_qualified_complete_student",
]
