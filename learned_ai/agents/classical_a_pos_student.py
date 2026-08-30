"""Qualified loading and fail-closed research routing for an ``A_pos`` student.

The production-shaped route accepts only an opaque, qualified complete-seed
handle.  It is not connected to any product, Web, CLI, database, advisor, or
training controller.  Its sole fallback capability represents one fresh exact
D9 post-gate attempt and deliberately has no production issuer in this slice.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import secrets
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.agents.positional_safety import (
    ProductPositionalSafetyGate,
    ProductSafetyOutcome,
    legal_inventory_identity,
)
from learned_ai.models.scaffolded_encoder import VALUE_INPUT_DIM, encode_position
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


_ROUTE_TOKEN = object()
_TEST_ROUTE_TOKEN = object()
_IMPLEMENTATION_TOKEN = object()
_FACTORY_TOKEN = object()
_SEARCH_TOKEN = object()
_ATTEMPT_TOKEN = object()
_STUDENT_SOURCE = "generalist-classical-d9-distilled-v1"
_EXACT_D9_SOURCE = "classical-d9-exact-post-gate"
_INTERNAL_TEST_SCOPE = "internal-test"
_PRODUCTION_SCOPE = "production"
_WDL_TIERS = frozenset({"W", "D", "L"})
_PRODUCTION_ENCODER = encode_position
_PRODUCTION_POLICY_LOGITS = ScaffoldedPolicyNet.policy_logits
_PRODUCTION_GET_CONFIG = ScaffoldedPolicyNet.get_config
_PRODUCTION_STATE_DICT = torch.nn.Module.state_dict
_PRODUCTION_PARAMETERS = torch.nn.Module.parameters
_PRODUCTION_GATE_CONSTRAIN = ProductPositionalSafetyGate.constrain
_FRESH_SEARCH_STATE_FIELDS = frozenset(
    {
        "transposition_table_entries",
        "history_entries",
        "cache_entries",
        "search_invocations",
    }
)
_EXACT_D9_ROUTE_CONTRACT_IDENTITY = canonical_sha256(
    {
        "schema": "nmm.classical-a-pos-exact-d9-route-contract.v1",
        "difficulty": 9,
        "fresh_search_per_fallback": True,
        "raw_search_result": "atomic-move-only",
        "post_gate": "captured-product-a-pos-gate",
        "failure_mode": "route-unavailable",
    }
)


class ResearchRouteUnavailable(RuntimeError):
    """The research route could not return a fully proven safe move."""


class _StudentPathError(RuntimeError):
    """A student inference or gate-evidence invariant failed."""


class _OpaqueResearchCapability:
    __slots__ = ("__weakref__",)

    _creation_token: object
    _kind: str

    def __init__(self, token: object) -> None:
        if token is not self._creation_token:
            raise ResearchRouteUnavailable(
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


@dataclass(frozen=True)
class ExactD9PostGateEvidence:
    """Evidence emitted by one consumed exact-D9 post-gate attempt."""

    outcome: Any
    attempt_identity: str
    factory_identity: str
    search_instance_identity: str
    search_initial_state_identity: str
    search_invocation_identity: str
    route_contract_identity: str
    route_implementation_identity: str
    route_effective_config_identity: str
    gate_runtime_identity: str
    gate_implementation_identity: str
    difficulty: int
    legal_inventory_identity: str


class ExactD9RouteImplementationBinding(_OpaqueResearchCapability):
    """Opaque commitment to one exact-D9 search implementation route."""

    __slots__ = ()

    _creation_token = _IMPLEMENTATION_TOKEN
    _kind = "exact D9 route implementation binding"


class ExactD9PostGateFactory(_OpaqueResearchCapability):
    """Opaque source of freshly constructed exact-D9 search capabilities."""

    __slots__ = ()

    _creation_token = _FACTORY_TOKEN
    _kind = "exact D9 post-gate factory"

    def create(self) -> IssuedExactD9Search:
        context = _EXACT_D9_FACTORY_CONTEXTS.get(self)
        if context is None:
            raise ResearchRouteUnavailable(
                "exact D9 factory lacks an issued private context"
            )
        binding_context = _EXACT_D9_IMPLEMENTATION_CONTEXTS.get(context.implementation)
        if binding_context is None:
            raise ResearchRouteUnavailable(
                "exact D9 route implementation binding expired"
            )
        _validate_implementation_runtime(binding_context)
        context.create_count += 1
        if context.reuse_search_handle and context.last_search_handle is not None:
            return context.last_search_handle
        try:
            search = binding_context.search_constructor.invoke()
        except Exception as exc:
            raise ResearchRouteUnavailable(
                "exact D9 factory could not create a fresh search instance"
            ) from exc
        if search is None:
            raise ResearchRouteUnavailable(
                "exact D9 constructor returned no search instance"
            )
        try:
            freshness_root = binding_context.freshness_root_accessor.invoke(search)
        except Exception as exc:
            raise ResearchRouteUnavailable(
                "exact D9 freshness-root accessor failed"
            ) from exc
        if freshness_root is None:
            raise ResearchRouteUnavailable(
                "exact D9 freshness-root accessor returned no root"
            )
        if any(search is prior for prior in context.issued_searches):
            raise ResearchRouteUnavailable(
                "exact D9 constructor reused a prior search instance"
            )
        if any(freshness_root is prior for prior in context.issued_freshness_roots):
            raise ResearchRouteUnavailable(
                "exact D9 constructor reused a prior freshness root"
            )
        _LIVE_EXACT_D9_SEARCHES.claim(search, field="search instance")
        _LIVE_EXACT_D9_FRESHNESS_ROOTS.claim(
            freshness_root,
            field="freshness root",
        )
        try:
            initial_snapshot = binding_context.fresh_state_attestor.invoke(
                search,
                freshness_root,
            )
            initial_state_identity = _fresh_search_state_identity(initial_snapshot)
        except Exception as exc:
            raise ResearchRouteUnavailable(
                "exact D9 search did not attest an empty initial state"
            ) from exc
        context.issued_searches.append(search)
        context.issued_freshness_roots.append(freshness_root)
        search_instance_identity = canonical_sha256(
            {
                "schema": "nmm.issued-exact-d9-search.v1",
                "factory_identity": context.factory_identity,
                "route_implementation_identity": (
                    binding_context.route_implementation_identity
                ),
                "ordinal": context.create_count,
                "nonce": secrets.token_hex(32),
            }
        )
        issued = IssuedExactD9Search(_SEARCH_TOKEN)
        _ISSUED_EXACT_D9_SEARCH_CONTEXTS[issued] = _IssuedExactD9SearchContext(
            implementation=context.implementation,
            factory=self,
            search=search,
            freshness_root=freshness_root,
            search_instance_identity=search_instance_identity,
            search_initial_state_identity=initial_state_identity,
        )
        if context.reuse_search_handle:
            context.last_search_handle = issued
        return issued


class IssuedExactD9Search(_OpaqueResearchCapability):
    """One issued exact-D9 search and its independently tracked state root."""

    __slots__ = ()

    _creation_token = _SEARCH_TOKEN
    _kind = "issued exact D9 search"

    def issue_attempt(self) -> FreshExactD9PostGateAttempt:
        context = _ISSUED_EXACT_D9_SEARCH_CONTEXTS.get(self)
        if context is None:
            raise ResearchRouteUnavailable(
                "issued exact D9 search lacks a private context"
            )
        if context.consumed:
            raise ResearchRouteUnavailable(
                "issued exact D9 search was already consumed"
            )
        context.consumed = True
        binding_context = _EXACT_D9_IMPLEMENTATION_CONTEXTS.get(context.implementation)
        factory_context = _EXACT_D9_FACTORY_CONTEXTS.get(context.factory)
        if binding_context is None or factory_context is None:
            raise ResearchRouteUnavailable("issued exact D9 search binding expired")
        _validate_implementation_runtime(binding_context)
        try:
            observed_root = binding_context.freshness_root_accessor.invoke(
                context.search
            )
            observed_initial = _fresh_search_state_identity(
                binding_context.fresh_state_attestor.invoke(
                    context.search,
                    context.freshness_root,
                )
            )
        except Exception as exc:
            raise ResearchRouteUnavailable(
                "issued exact D9 search initial state cannot be reverified"
            ) from exc
        if (
            observed_root is not context.freshness_root
            or observed_initial != context.search_initial_state_identity
        ):
            raise ResearchRouteUnavailable(
                "issued exact D9 search initial state or freshness root drifted"
            )
        attempt_identity = canonical_sha256(
            {
                "schema": "nmm.fresh-exact-d9-post-gate-attempt.v2",
                "factory_identity": factory_context.factory_identity,
                "search_instance_identity": context.search_instance_identity,
                "nonce": secrets.token_hex(32),
            }
        )
        attempt = FreshExactD9PostGateAttempt(_ATTEMPT_TOKEN)
        _FRESH_EXACT_D9_ATTEMPT_CONTEXTS[attempt] = _ExactD9AttemptContext(
            implementation=context.implementation,
            factory=context.factory,
            issued_search=self,
            attempt_identity=attempt_identity,
        )
        return attempt


class FreshExactD9PostGateAttempt(_OpaqueResearchCapability):
    """One fresh exact-D9 instance that can be queried exactly once."""

    __slots__ = ()

    _creation_token = _ATTEMPT_TOKEN
    _kind = "fresh exact D9 post-gate attempt"

    def choose_once(
        self,
        board: BoardState,
        *,
        expected_legal_inventory_identity: str,
    ) -> ExactD9PostGateEvidence:
        context = _FRESH_EXACT_D9_ATTEMPT_CONTEXTS.get(self)
        if context is None:
            raise ResearchRouteUnavailable(
                "fresh exact D9 attempt lacks an issued private context"
            )
        if context.consumed:
            raise ResearchRouteUnavailable(
                "fresh exact D9 attempt was already consumed"
            )
        context.consumed = True
        binding_context = _EXACT_D9_IMPLEMENTATION_CONTEXTS.get(context.implementation)
        factory_context = _EXACT_D9_FACTORY_CONTEXTS.get(context.factory)
        search_context = _ISSUED_EXACT_D9_SEARCH_CONTEXTS.get(context.issued_search)
        if binding_context is None or factory_context is None or search_context is None:
            raise ResearchRouteUnavailable("fresh exact D9 attempt binding expired")
        _validate_implementation_runtime(binding_context)
        _require_route_sha256(
            expected_legal_inventory_identity,
            field="expected legal inventory",
        )
        try:
            independently_enumerated = [
                dict(move) for move in get_all_legal_moves(board)
            ]
            observed_identity = legal_inventory_identity(independently_enumerated)
        except Exception as exc:
            raise ResearchRouteUnavailable(
                "fresh exact D9 attempt cannot verify the legal inventory"
            ) from exc
        if observed_identity != expected_legal_inventory_identity:
            raise ResearchRouteUnavailable(
                "fresh exact D9 attempt legal inventory identity differs"
            )
        try:
            observed_root = binding_context.freshness_root_accessor.invoke(
                search_context.search
            )
            observed_initial = _fresh_search_state_identity(
                binding_context.fresh_state_attestor.invoke(
                    search_context.search,
                    search_context.freshness_root,
                )
            )
        except Exception as exc:
            raise ResearchRouteUnavailable(
                "fresh exact D9 attempt cannot reverify the initial search state"
            ) from exc
        if (
            observed_root is not search_context.freshness_root
            or observed_initial != search_context.search_initial_state_identity
        ):
            raise ResearchRouteUnavailable(
                "fresh exact D9 attempt search state is not initial and empty"
            )
        try:
            raw_move = binding_context.choose_raw.invoke(
                search_context.search,
                board,
            )
        except Exception as exc:
            raise ResearchRouteUnavailable(
                "fresh exact D9 raw search invocation failed"
            ) from exc
        if isinstance(raw_move, ProductSafetyOutcome):
            raise ResearchRouteUnavailable(
                "fresh exact D9 search returned a self-authored gate outcome"
            )
        try:
            raw = dict(raw_move)
            raw_key = _atomic_move_key(raw)
            legal_keys = {_atomic_move_key(move) for move in independently_enumerated}
        except Exception as exc:
            raise ResearchRouteUnavailable(
                "fresh exact D9 search did not return one atomic move"
            ) from exc
        if raw_key not in legal_keys:
            raise ResearchRouteUnavailable(
                "fresh exact D9 raw move is outside the legal inventory"
            )
        context.raw_move = raw
        restricted_selector = None
        if binding_context.restricted_selector is not None:

            def restricted_selector(safe_moves: list[dict[str, Any]]) -> Any:
                return binding_context.restricted_selector.invoke(
                    search_context.search,
                    board,
                    safe_moves,
                )

        _validate_bound_gate_runtime(
            binding_context.gate,
            binding_context.gate_constrain,
            production=binding_context.issuer_scope == _PRODUCTION_SCOPE,
        )
        try:
            outcome = binding_context.gate_constrain(
                binding_context.gate,
                board,
                raw,
                source=_EXACT_D9_SOURCE,
                difficulty=9,
                candidate_moves=None,
                candidate_scores=None,
                safe_selector=restricted_selector,
                query_failure_move=raw,
            )
        except Exception as exc:
            raise ResearchRouteUnavailable(
                "fresh exact D9 post-gate invocation failed"
            ) from exc
        search_invocation_identity = canonical_sha256(
            {
                "schema": "nmm.exact-d9-search-invocation.v1",
                "attempt_identity": context.attempt_identity,
                "search_instance_identity": (search_context.search_instance_identity),
                "legal_inventory_identity": expected_legal_inventory_identity,
                "raw_move": {
                    "from": raw["from"],
                    "to": raw["to"],
                    "capture": raw["capture"],
                },
            }
        )
        context.search_invocation_identity = search_invocation_identity
        evidence = ExactD9PostGateEvidence(
            outcome=outcome,
            attempt_identity=context.attempt_identity,
            factory_identity=factory_context.factory_identity,
            search_instance_identity=search_context.search_instance_identity,
            search_initial_state_identity=(
                search_context.search_initial_state_identity
            ),
            search_invocation_identity=search_invocation_identity,
            route_contract_identity=binding_context.route_contract_identity,
            route_implementation_identity=(
                binding_context.route_implementation_identity
            ),
            route_effective_config_identity=(
                binding_context.route_effective_config_identity
            ),
            gate_runtime_identity=binding_context.gate_runtime_identity,
            gate_implementation_identity=(binding_context.gate_implementation_identity),
            difficulty=binding_context.difficulty,
            legal_inventory_identity=expected_legal_inventory_identity,
        )
        if factory_context.evidence_overrides:
            try:
                evidence = replace(
                    evidence,
                    **factory_context.evidence_overrides,
                )
            except TypeError as exc:
                raise ResearchRouteUnavailable(
                    "fresh exact D9 evidence override fields differ"
                ) from exc
        return evidence


_FACTORY_CREATE_METHOD = ExactD9PostGateFactory.create
_SEARCH_ISSUE_ATTEMPT_METHOD = IssuedExactD9Search.issue_attempt
_ATTEMPT_CHOOSE_ONCE_METHOD = FreshExactD9PostGateAttempt.choose_once


def _validate_capability_method_bindings() -> None:
    if (
        ExactD9PostGateFactory.create is not _FACTORY_CREATE_METHOD
        or IssuedExactD9Search.issue_attempt is not _SEARCH_ISSUE_ATTEMPT_METHOD
        or FreshExactD9PostGateAttempt.choose_once is not _ATTEMPT_CHOOSE_ONCE_METHOD
    ):
        raise ResearchRouteUnavailable(
            "exact D9 capability class method binding drifted"
        )


@dataclass
class _ExactD9ImplementationContext:
    issuer_scope: str
    gate: Any
    gate_constrain: Any
    gate_runtime_identity: str
    gate_implementation_identity: str
    search_constructor: _CommittedCallable
    freshness_root_accessor: _CommittedCallable
    fresh_state_attestor: _CommittedCallable
    choose_raw: _CommittedCallable
    restricted_selector: _CommittedCallable | None
    difficulty: int
    route_contract_identity: str
    route_implementation_identity: str
    route_effective_config_identity: str
    factory_issued: bool = False


@dataclass
class _ExactD9FactoryContext:
    implementation: ExactD9RouteImplementationBinding
    gate: Any
    issuer_scope: str
    difficulty: int
    evidence_overrides: dict[str, Any]
    reuse_search_handle: bool
    factory_identity: str
    create_count: int = 0
    last_search_handle: IssuedExactD9Search | None = None
    issued_searches: list[Any] | None = None
    issued_freshness_roots: list[Any] | None = None
    owner_route: weakref.ReferenceType[Any] | None = None
    bound_once: bool = False

    def __post_init__(self) -> None:
        if self.issued_searches is None:
            self.issued_searches = []
        if self.issued_freshness_roots is None:
            self.issued_freshness_roots = []


@dataclass
class _IssuedExactD9SearchContext:
    implementation: ExactD9RouteImplementationBinding
    factory: ExactD9PostGateFactory
    search: Any
    freshness_root: Any
    search_instance_identity: str
    search_initial_state_identity: str
    consumed: bool = False


@dataclass
class _ExactD9AttemptContext:
    implementation: ExactD9RouteImplementationBinding
    factory: ExactD9PostGateFactory
    issued_search: IssuedExactD9Search
    attempt_identity: str
    raw_move: dict[str, Any] | None = None
    search_invocation_identity: str | None = None
    consumed: bool = False


class _LiveObjectIdentityRegistry:
    """Track reused live objects by identity without invoking caller equality."""

    def __init__(self) -> None:
        self._weak: list[weakref.ReferenceType[Any]] = []
        self._strong: list[Any] = []

    def claim(self, value: Any, *, field: str) -> None:
        live_refs: list[weakref.ReferenceType[Any]] = []
        reused = False
        for reference in self._weak:
            prior = reference()
            if prior is not None:
                live_refs.append(reference)
                reused = reused or prior is value
        self._weak = live_refs
        reused = reused or any(prior is value for prior in self._strong)
        if reused:
            raise ResearchRouteUnavailable(
                f"exact D9 reused a live {field} across factories"
            )
        try:
            self._weak.append(weakref.ref(value))
        except TypeError:
            self._strong.append(value)


_EXACT_D9_IMPLEMENTATION_CONTEXTS: weakref.WeakKeyDictionary[
    ExactD9RouteImplementationBinding,
    _ExactD9ImplementationContext,
] = weakref.WeakKeyDictionary()
_LIVE_EXACT_D9_SEARCHES = _LiveObjectIdentityRegistry()
_LIVE_EXACT_D9_FRESHNESS_ROOTS = _LiveObjectIdentityRegistry()
_EXACT_D9_FACTORY_CONTEXTS: weakref.WeakKeyDictionary[
    ExactD9PostGateFactory,
    _ExactD9FactoryContext,
] = weakref.WeakKeyDictionary()
_ISSUED_EXACT_D9_SEARCH_CONTEXTS: weakref.WeakKeyDictionary[
    IssuedExactD9Search,
    _IssuedExactD9SearchContext,
] = weakref.WeakKeyDictionary()
_FRESH_EXACT_D9_ATTEMPT_CONTEXTS: weakref.WeakKeyDictionary[
    FreshExactD9PostGateAttempt,
    _ExactD9AttemptContext,
] = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class _CommittedCallable:
    callback: Any
    owner: Any | None
    implementation: Any
    code: Any | None
    invocation_kind: str
    identity: str

    @classmethod
    def capture(cls, callback: Any, *, field: str) -> _CommittedCallable:
        if not callable(callback):
            raise ResearchRouteUnavailable(f"{field} is not callable")
        if inspect.ismethod(callback) and callback.__self__ is not None:
            owner = callback.__self__
            implementation = callback.__func__
            kind = "bound-method"
        elif inspect.isfunction(callback):
            owner = None
            implementation = callback
            kind = "function"
        else:
            owner = callback
            implementation = type(callback).__call__
            kind = "callable-object"
        code = getattr(implementation, "__code__", None)
        code_bytes = b"" if code is None else code.co_code
        identity = canonical_sha256(
            {
                "schema": "nmm.committed-callable.v1",
                "field": field,
                "kind": kind,
                "module": getattr(implementation, "__module__", ""),
                "qualname": getattr(implementation, "__qualname__", ""),
                "code_sha256": hashlib.sha256(code_bytes).hexdigest(),
                "object_identity": id(callback),
                "owner_identity": None if owner is None else id(owner),
            }
        )
        return cls(
            callback=callback,
            owner=owner,
            implementation=implementation,
            code=code,
            invocation_kind=kind,
            identity=identity,
        )

    def validate(self) -> None:
        if getattr(self.implementation, "__code__", None) is not self.code:
            raise ResearchRouteUnavailable(
                "committed exact D9 implementation method drifted"
            )
        if self.invocation_kind == "bound-method":
            if (
                self.owner is None
                or getattr(type(self.owner), self.implementation.__name__, None)
                is not self.implementation
            ):
                raise ResearchRouteUnavailable(
                    "committed exact D9 bound method drifted"
                )
        elif self.invocation_kind == "callable-object":
            owner_dict = getattr(self.owner, "__dict__", {})
            if (
                self.owner is None
                or type(self.owner).__call__ is not self.implementation
                or "__call__" in owner_dict
            ):
                raise ResearchRouteUnavailable(
                    "committed exact D9 callable implementation drifted"
                )

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        self.validate()
        if self.invocation_kind in {"bound-method", "callable-object"}:
            return self.implementation(self.owner, *args, **kwargs)
        return self.implementation(*args, **kwargs)


def _gate_runtime_identity(gate: Any) -> str:
    return canonical_sha256(
        {
            "schema": "nmm.in-process-a-pos-gate-runtime.v2",
            "type": f"{type(gate).__module__}.{type(gate).__qualname__}",
            "object_identity": id(gate),
        }
    )


def _gate_implementation_identity(unbound_constrain: Any) -> str:
    code = getattr(unbound_constrain, "__code__", None)
    return canonical_sha256(
        {
            "schema": "nmm.in-process-a-pos-gate-implementation.v1",
            "module": getattr(unbound_constrain, "__module__", ""),
            "qualname": getattr(unbound_constrain, "__qualname__", ""),
            "code_sha256": hashlib.sha256(
                b"" if code is None else code.co_code
            ).hexdigest(),
            "object_identity": id(unbound_constrain),
        }
    )


def _capture_gate_constrain(gate: Any, *, production: bool) -> Any:
    if production and type(gate) is not ProductPositionalSafetyGate:
        raise ResearchRouteUnavailable(
            "production route requires the exact product A_pos gate"
        )
    class_method = getattr(type(gate), "constrain", None)
    instance_dict = getattr(gate, "__dict__", {})
    if not callable(class_method) or "constrain" in instance_dict:
        raise ResearchRouteUnavailable(
            "A_pos gate constrain method is missing or instance-shadowed"
        )
    if production and (
        ProductPositionalSafetyGate.constrain is not _PRODUCTION_GATE_CONSTRAIN
        or class_method is not _PRODUCTION_GATE_CONSTRAIN
    ):
        raise ResearchRouteUnavailable(
            "product A_pos gate constrain class method drifted"
        )
    return class_method


def _validate_bound_gate_runtime(
    gate: Any,
    captured_constrain: Any,
    *,
    production: bool,
) -> None:
    observed = _capture_gate_constrain(gate, production=production)
    if observed is not captured_constrain:
        raise ResearchRouteUnavailable(
            "A_pos gate constrain implementation changed after binding"
        )


def _fresh_search_state_identity(snapshot: Any) -> str:
    if not isinstance(snapshot, Mapping) or set(snapshot) != _FRESH_SEARCH_STATE_FIELDS:
        raise ResearchRouteUnavailable("exact D9 fresh-state attestation fields differ")
    normalized: dict[str, int] = {}
    for field in sorted(_FRESH_SEARCH_STATE_FIELDS):
        value = snapshot[field]
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise ResearchRouteUnavailable(
                "exact D9 fresh-state attestation is not empty"
            )
        normalized[field] = value
    return canonical_sha256(
        {
            "schema": "nmm.exact-d9-empty-search-state.v1",
            "state": normalized,
        }
    )


def _validate_implementation_runtime(
    context: _ExactD9ImplementationContext,
) -> None:
    _validate_bound_gate_runtime(
        context.gate,
        context.gate_constrain,
        production=context.issuer_scope == _PRODUCTION_SCOPE,
    )
    if _gate_runtime_identity(context.gate) != context.gate_runtime_identity:
        raise ResearchRouteUnavailable("exact D9 gate runtime identity drifted")
    if (
        _gate_implementation_identity(context.gate_constrain)
        != context.gate_implementation_identity
    ):
        raise ResearchRouteUnavailable("exact D9 gate implementation identity drifted")
    context.search_constructor.validate()
    context.freshness_root_accessor.validate()
    context.fresh_state_attestor.validate()
    context.choose_raw.validate()
    if context.restricted_selector is not None:
        context.restricted_selector.validate()


def _issue_test_exact_d9_route_implementation_binding(
    *,
    gate: Any,
    search_constructor: Any,
    freshness_root_accessor: Any,
    fresh_state_attestor: Any,
    choose_raw: Any,
    restricted_selector: Any | None = None,
    difficulty: int = 9,
    effective_config: Mapping[str, Any] | None = None,
) -> ExactD9RouteImplementationBinding:
    """Issue an internal-test implementation commitment; never production."""

    if gate is None:
        raise ResearchRouteUnavailable("test exact D9 implementation requires a gate")
    if type(difficulty) is not int or difficulty != 9:
        raise ResearchRouteUnavailable("test exact D9 difficulty is invalid")
    gate_constrain = _capture_gate_constrain(gate, production=False)
    constructor = _CommittedCallable.capture(
        search_constructor,
        field="search_constructor",
    )
    root_accessor = _CommittedCallable.capture(
        freshness_root_accessor,
        field="freshness_root_accessor",
    )
    state_attestor = _CommittedCallable.capture(
        fresh_state_attestor,
        field="fresh_state_attestor",
    )
    raw_chooser = _CommittedCallable.capture(choose_raw, field="choose_raw")
    selector = (
        None
        if restricted_selector is None
        else _CommittedCallable.capture(
            restricted_selector,
            field="restricted_selector",
        )
    )
    config = dict(
        effective_config
        or {
            "engine": "internal-test-exact-d9",
            "difficulty": 9,
            "threads": 1,
            "fresh_instance_per_fallback": True,
        }
    )
    if config.get("difficulty") != 9:
        raise ResearchRouteUnavailable(
            "test exact D9 effective config difficulty differs"
        )
    effective_config_identity = canonical_sha256(
        {
            "schema": "nmm.exact-d9-route-effective-config.v1",
            "config": config,
        }
    )
    implementation_identity = canonical_sha256(
        {
            "schema": "nmm.internal-test-exact-d9-route-implementation.v1",
            "constructor": constructor.identity,
            "freshness_root_accessor": root_accessor.identity,
            "fresh_state_attestor": state_attestor.identity,
            "choose_raw": raw_chooser.identity,
            "restricted_selector": None if selector is None else selector.identity,
            "effective_config": effective_config_identity,
        }
    )
    binding = ExactD9RouteImplementationBinding(_IMPLEMENTATION_TOKEN)
    _EXACT_D9_IMPLEMENTATION_CONTEXTS[binding] = _ExactD9ImplementationContext(
        issuer_scope=_INTERNAL_TEST_SCOPE,
        gate=gate,
        gate_constrain=gate_constrain,
        gate_runtime_identity=_gate_runtime_identity(gate),
        gate_implementation_identity=_gate_implementation_identity(gate_constrain),
        search_constructor=constructor,
        freshness_root_accessor=root_accessor,
        fresh_state_attestor=state_attestor,
        choose_raw=raw_chooser,
        restricted_selector=selector,
        difficulty=difficulty,
        route_contract_identity=_EXACT_D9_ROUTE_CONTRACT_IDENTITY,
        route_implementation_identity=implementation_identity,
        route_effective_config_identity=effective_config_identity,
    )
    return binding


def _issue_test_exact_d9_post_gate_factory(
    *,
    implementation: ExactD9RouteImplementationBinding,
    evidence_overrides: Mapping[str, Any] | None = None,
    reuse_search_handle: bool = False,
) -> ExactD9PostGateFactory:
    """Issue an internal-test factory; no production issuer exists here."""

    if type(implementation) is not ExactD9RouteImplementationBinding:
        raise ResearchRouteUnavailable(
            "test factory requires an issued route implementation binding"
        )
    implementation_context = _EXACT_D9_IMPLEMENTATION_CONTEXTS.get(implementation)
    if implementation_context is None:
        raise ResearchRouteUnavailable(
            "test route implementation binding lacks private context"
        )
    if implementation_context.issuer_scope != _INTERNAL_TEST_SCOPE:
        raise ResearchRouteUnavailable(
            "test factory cannot upgrade a production implementation scope"
        )
    if implementation_context.factory_issued:
        raise ResearchRouteUnavailable(
            "test route implementation binding was already consumed"
        )
    _validate_implementation_runtime(implementation_context)
    overrides = dict(evidence_overrides or {})
    allowed_overrides = {field.name for field in fields(ExactD9PostGateEvidence)}
    if set(overrides) - allowed_overrides:
        raise ResearchRouteUnavailable("test exact D9 evidence override fields differ")
    factory = ExactD9PostGateFactory(_FACTORY_TOKEN)
    _EXACT_D9_FACTORY_CONTEXTS[factory] = _ExactD9FactoryContext(
        implementation=implementation,
        gate=implementation_context.gate,
        issuer_scope=_INTERNAL_TEST_SCOPE,
        difficulty=implementation_context.difficulty,
        evidence_overrides=overrides,
        reuse_search_handle=bool(reuse_search_handle),
        factory_identity=canonical_sha256(
            {
                "schema": "nmm.internal-test-exact-d9-post-gate-factory.v2",
                "route_implementation_identity": (
                    implementation_context.route_implementation_identity
                ),
                "nonce": secrets.token_hex(32),
            }
        ),
    )
    implementation_context.factory_issued = True
    return factory


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


@dataclass(frozen=True)
class ClassicalAPosRouteOutcome:
    """Deeply immutable evidence for one research-route selection."""

    move: Mapping[str, Any]
    route: str
    safety_decision: Mapping[str, Any]
    checkpoint_identity: str
    fallback_reason: str | None

    def __post_init__(self) -> None:
        if self.route not in {"student", "exact-d9-post-gate-fallback"}:
            raise ResearchRouteUnavailable("research outcome route is invalid")
        if not isinstance(self.checkpoint_identity, str) or not (
            self.checkpoint_identity
        ):
            raise ResearchRouteUnavailable(
                "research outcome checkpoint identity is missing"
            )
        if self.route == "student" and self.fallback_reason is not None:
            raise ResearchRouteUnavailable(
                "student outcome cannot claim a fallback reason"
            )
        if self.route == "exact-d9-post-gate-fallback" and (
            not isinstance(self.fallback_reason, str) or not self.fallback_reason
        ):
            raise ResearchRouteUnavailable(
                "exact D9 fallback outcome requires its triggering reason"
            )
        try:
            _atomic_move_key(self.move)
        except _StudentPathError as exc:
            raise ResearchRouteUnavailable(
                "research outcome atomic move is invalid"
            ) from exc
        object.__setattr__(self, "move", _freeze_value(dict(self.move)))
        object.__setattr__(
            self,
            "safety_decision",
            _freeze_value(dict(self.safety_decision)),
        )


class _OpaqueRoute(_OpaqueResearchCapability):
    __slots__ = ()

    def choose_move(self, board: BoardState) -> ClassicalAPosRouteOutcome:
        return _choose_move(self, board)


class ClassicalAPosStudentRoute(_OpaqueRoute):
    """Production-shaped route that accepts only a qualified B1 handle."""

    __slots__ = ()

    _creation_token = _ROUTE_TOKEN
    _kind = "qualified classical A_pos student route"

    @classmethod
    def from_qualified(
        cls,
        *,
        student: QualifiedStudentHandle,
        gate: ProductPositionalSafetyGate,
        exact_d9_factory: ExactD9PostGateFactory,
    ) -> ClassicalAPosStudentRoute:
        if cls is not ClassicalAPosStudentRoute:
            raise ResearchRouteUnavailable(
                "qualified student route subclasses are not supported"
            )
        if type(student) is not QualifiedStudentHandle:
            raise ResearchRouteUnavailable(
                "production route requires an issued qualified student handle"
            )
        student_context = _QUALIFIED_STUDENT_CONTEXTS.get(student)
        if student_context is None:
            raise ResearchRouteUnavailable(
                "production route requires a live qualified student context"
            )
        if (
            student in _ROUTED_STUDENT_HANDLES
            or student in _ROUTE_ATTEMPTED_STUDENT_HANDLES
        ):
            raise ResearchRouteUnavailable(
                "qualified student handle is already bound or consumed"
            )
        _ROUTE_ATTEMPTED_STUDENT_HANDLES.add(student)
        gate_constrain = _capture_gate_constrain(gate, production=True)
        _validate_qualified_model_runtime(student_context)
        context = _RouteContext(
            model=student_context.model,
            encoder=_PRODUCTION_ENCODER,
            gate=gate,
            gate_constrain=gate_constrain,
            gate_runtime_identity=_gate_runtime_identity(gate),
            gate_implementation_identity=_gate_implementation_identity(gate_constrain),
            exact_d9_factory=exact_d9_factory,
            checkpoint_identity=canonical_sha256(
                {
                    "schema": "nmm.qualified-student-route-provenance.v1",
                    "checkpoint_id": student_context.checkpoint_id,
                    "checkpoint_file_identity": (
                        student_context.checkpoint_file_identity
                    ),
                    "model_identity": student_context.model_identity,
                    "qualification_identity": student_context.qualification_identity,
                }
            ),
            model_identity=student_context.model_identity,
            qualification_identity=student_context.qualification_identity,
            production=True,
            student_handle=student,
        )
        route = cls(_ROUTE_TOKEN)
        _bind_route(route, context)
        _ROUTED_STUDENT_HANDLES.add(student)
        return route


class _TestClassicalAPosStudentRoute(_OpaqueRoute):
    """Distinct raw-injection route that cannot enter production construction."""

    __slots__ = ()

    _creation_token = _TEST_ROUTE_TOKEN
    _kind = "test-only classical A_pos student route"


@dataclass
class _RouteContext:
    model: Any
    encoder: Any
    gate: Any
    gate_constrain: Any
    gate_runtime_identity: str
    gate_implementation_identity: str
    exact_d9_factory: ExactD9PostGateFactory
    checkpoint_identity: str
    model_identity: str
    qualification_identity: str
    production: bool
    student_handle: QualifiedStudentHandle | None
    seen_search_identities: set[str] | None = None
    seen_search_handles: weakref.WeakSet[IssuedExactD9Search] | None = None
    seen_attempt_identities: set[str] | None = None
    seen_attempts: weakref.WeakSet[FreshExactD9PostGateAttempt] | None = None
    seen_fallback_outcomes: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.seen_search_identities is None:
            self.seen_search_identities = set()
        if self.seen_search_handles is None:
            self.seen_search_handles = weakref.WeakSet()
        if self.seen_attempt_identities is None:
            self.seen_attempt_identities = set()
        if self.seen_attempts is None:
            self.seen_attempts = weakref.WeakSet()
        if self.seen_fallback_outcomes is None:
            self.seen_fallback_outcomes = []


_ROUTE_CONTEXTS: weakref.WeakKeyDictionary[
    _OpaqueRoute,
    _RouteContext,
] = weakref.WeakKeyDictionary()
_ROUTED_STUDENT_HANDLES: weakref.WeakSet[QualifiedStudentHandle] = weakref.WeakSet()
_ROUTE_ATTEMPTED_STUDENT_HANDLES: weakref.WeakSet[QualifiedStudentHandle] = (
    weakref.WeakSet()
)


def _bind_route(route: _OpaqueRoute, context: _RouteContext) -> None:
    _validate_capability_method_bindings()
    if type(context.exact_d9_factory) is not ExactD9PostGateFactory:
        raise ResearchRouteUnavailable(
            "route requires an issued exact D9 post-gate factory"
        )
    factory_context = _EXACT_D9_FACTORY_CONTEXTS.get(context.exact_d9_factory)
    if factory_context is None:
        raise ResearchRouteUnavailable(
            "route exact D9 factory lacks an issued private context"
        )
    binding_context = _EXACT_D9_IMPLEMENTATION_CONTEXTS.get(
        factory_context.implementation
    )
    if binding_context is None:
        raise ResearchRouteUnavailable(
            "route exact D9 implementation binding lacks private context"
        )
    expected_scope = (
        _PRODUCTION_SCOPE
        if type(route) is ClassicalAPosStudentRoute
        else _INTERNAL_TEST_SCOPE
    )
    if (
        factory_context.issuer_scope != expected_scope
        or binding_context.issuer_scope != expected_scope
    ):
        raise ResearchRouteUnavailable(
            "exact D9 factory issuer scope cannot enter this route"
        )
    if factory_context.gate is not context.gate:
        raise ResearchRouteUnavailable(
            "exact D9 factory is bound to a different A_pos gate"
        )
    if type(factory_context.difficulty) is not int or factory_context.difficulty != 9:
        raise ResearchRouteUnavailable(
            "exact D9 factory difficulty must be exactly nine"
        )
    _validate_implementation_runtime(binding_context)
    _validate_bound_gate_runtime(
        context.gate,
        context.gate_constrain,
        production=context.production,
    )
    if (
        binding_context.gate_constrain is not context.gate_constrain
        or binding_context.gate_runtime_identity != context.gate_runtime_identity
        or binding_context.gate_implementation_identity
        != context.gate_implementation_identity
    ):
        raise ResearchRouteUnavailable(
            "route and exact D9 factory gate commitments differ"
        )
    if factory_context.bound_once:
        raise ResearchRouteUnavailable(
            "exact D9 factory is already bound to another route"
        )
    if factory_context.create_count != 0:
        raise ResearchRouteUnavailable("exact D9 factory was used before route binding")
    _ROUTE_CONTEXTS[route] = context
    factory_context.owner_route = weakref.ref(route)
    factory_context.bound_once = True


def _make_test_classical_a_pos_route(
    *,
    model: Any,
    encoder: Any,
    gate: Any,
    exact_d9_factory: ExactD9PostGateFactory,
) -> _TestClassicalAPosStudentRoute:
    """Build a distinct test-only route with explicit raw dependencies."""

    if not callable(getattr(model, "policy_logits", None)):
        raise ResearchRouteUnavailable("test student model lacks policy_logits")
    if not callable(encoder):
        raise ResearchRouteUnavailable("test student encoder is not callable")
    gate_constrain = _capture_gate_constrain(gate, production=False)
    route = _TestClassicalAPosStudentRoute(_TEST_ROUTE_TOKEN)
    _bind_route(
        route,
        _RouteContext(
            model=model,
            encoder=encoder,
            gate=gate,
            gate_constrain=gate_constrain,
            gate_runtime_identity=_gate_runtime_identity(gate),
            gate_implementation_identity=_gate_implementation_identity(gate_constrain),
            exact_d9_factory=exact_d9_factory,
            checkpoint_identity="internal-test-unqualified",
            model_identity="internal-test-unqualified",
            qualification_identity="internal-test-unqualified",
            production=False,
            student_handle=None,
        ),
    )
    return route


def _validate_qualified_model_runtime(context: _QualifiedStudentContext) -> None:
    model = context.model
    if type(model) is not ScaffoldedPolicyNet or model.training is not False:
        raise ResearchRouteUnavailable(
            "qualified student model implementation or eval mode differs"
        )
    if (
        ScaffoldedPolicyNet.policy_logits is not _PRODUCTION_POLICY_LOGITS
        or ScaffoldedPolicyNet.get_config is not _PRODUCTION_GET_CONFIG
        or torch.nn.Module.state_dict is not _PRODUCTION_STATE_DICT
        or torch.nn.Module.parameters is not _PRODUCTION_PARAMETERS
        or any(
            name in model.__dict__
            for name in ("policy_logits", "get_config", "state_dict", "parameters")
        )
    ):
        raise ResearchRouteUnavailable("qualified student model method binding drifted")
    try:
        observed_config = dict(_PRODUCTION_GET_CONFIG(model))
        config_matches = canonical_training_state_identity(
            observed_config
        ) == canonical_training_state_identity(_MODEL_CONFIG)
    except (RuntimeError, TypeError, ValueError, SupervisedTrainingError) as exc:
        raise ResearchRouteUnavailable(
            "qualified student model config cannot be verified"
        ) from exc
    if not config_matches:
        raise ResearchRouteUnavailable("qualified student model config drifted")
    parameters = tuple(_PRODUCTION_PARAMETERS(model))
    if any(
        parameter.device.type != "cpu" or parameter.requires_grad
        for parameter in parameters
    ):
        raise ResearchRouteUnavailable(
            "qualified student model must remain frozen on CPU"
        )
    model_state = _PRODUCTION_STATE_DICT(model)
    if any(
        not bool(torch.isfinite(tensor).all())
        for tensor in model_state.values()
        if tensor.is_floating_point() or tensor.is_complex()
    ):
        raise ResearchRouteUnavailable("qualified student model became non-finite")
    if canonical_training_state_identity(model_state) != context.model_identity:
        raise ResearchRouteUnavailable("qualified student model identity drifted")


def _require_route_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ResearchRouteUnavailable(f"{field} identity is invalid")
    return value


def _atomic_move_key(move: Any) -> tuple[Any, Any, Any]:
    if not isinstance(move, Mapping) or any(
        field not in move for field in ("from", "to", "capture")
    ):
        raise _StudentPathError("atomic legal action must contain from, to, capture")
    return move["from"], move["to"], move["capture"]


def _complete_legal_inventory(
    board: BoardState,
) -> tuple[list[dict[str, Any]], tuple[tuple[Any, Any, Any], ...], str]:
    try:
        legal = [dict(move) for move in get_all_legal_moves(board)]
        if not legal:
            raise _StudentPathError("complete legal inventory is empty")
        keys = tuple(_atomic_move_key(move) for move in legal)
        if len(set(keys)) != len(keys):
            raise _StudentPathError("complete legal inventory contains duplicates")
        identity = legal_inventory_identity(legal)
    except _StudentPathError:
        raise
    except Exception as exc:
        raise _StudentPathError(
            "complete legal inventory cannot be enumerated"
        ) from exc
    return legal, keys, identity


def _student_probabilities(
    context: _RouteContext,
    board: BoardState,
    legal: list[dict[str, Any]],
    legal_keys: tuple[tuple[Any, Any, Any], ...],
) -> tuple[list[float], int]:
    try:
        encoded = context.encoder(
            board,
            board.turn,
            sentinel_advisor=None,
            db=None,
            value_net=None,
            specialist_db=None,
            wdl_db=None,
            strict=True,
        )
    except Exception as exc:
        raise _StudentPathError("student strict encoder raised") from exc
    if encoded is None:
        raise _StudentPathError("student strict encoder returned no position")
    try:
        encoded_moves = list(encoded.legal_moves)
        encoded_keys = tuple(_atomic_move_key(move) for move in encoded_moves)
    except Exception as exc:
        raise _StudentPathError(
            "student encoder legal action inventory is invalid"
        ) from exc
    if encoded_keys != legal_keys:
        raise _StudentPathError(
            "student encoder legal action order differs from complete legal order"
        )
    features = getattr(encoded, "feat_matrix", None)
    if (
        type(features) is not np.ndarray
        or features.dtype != np.dtype(np.float32)
        or features.ndim != 2
        or features.shape != (len(legal), 62)
        or not bool(np.isfinite(features).all())
    ):
        raise _StudentPathError(
            "student features must be finite CPU numpy float32 with shape (N,62)"
        )
    tensor = torch.tensor(features, dtype=torch.float32, device="cpu")
    try:
        with torch.inference_mode():
            logits = (
                _PRODUCTION_POLICY_LOGITS(context.model, tensor)
                if context.production
                else context.model.policy_logits(tensor)
            )
    except Exception as exc:
        raise _StudentPathError("student policy_logits raised") from exc
    if (
        not isinstance(logits, torch.Tensor)
        or logits.device.type != "cpu"
        or not logits.is_floating_point()
        or logits.ndim != 1
        or tuple(logits.shape) != (len(legal),)
        or not bool(torch.isfinite(logits).all())
    ):
        raise _StudentPathError(
            "student logits must be finite one-dimensional CPU floats for every legal move"
        )
    values = [
        float(value)
        for value in logits.detach().to(device="cpu", dtype=torch.float64).tolist()
    ]
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise _StudentPathError("student stable softmax denominator is invalid")
    probabilities = [weight / total for weight in weights]
    probability_total = math.fsum(probabilities)
    if (
        any(not math.isfinite(value) or value < 0.0 for value in probabilities)
        or not math.isfinite(probability_total)
        or abs(probability_total - 1.0) > 1e-12
    ):
        raise _StudentPathError("student stable softmax probabilities are invalid")
    original_index = max(range(len(values)), key=values.__getitem__)
    return probabilities, original_index


def _marker_keys(decision: Mapping[str, Any]) -> list[str]:
    markers = ("fail", "degrad", "alternat", "fallback")
    found: list[str] = []
    for key in decision:
        if not isinstance(key, str):
            found.append(str(key))
            continue
        normalized = key.lower().replace("-", "_")
        if any(marker in normalized for marker in markers):
            found.append(key)
    return found


def _validate_safety_outcome(
    outcome: Any,
    *,
    legal: list[dict[str, Any]],
    legal_keys: tuple[tuple[Any, Any, Any], ...],
    inventory_identity: str,
    route_kind: str,
    expected_original: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(outcome) is not ProductSafetyOutcome:
        raise _StudentPathError("post-gate outcome type differs")
    if not isinstance(outcome.decision, Mapping):
        raise _StudentPathError("post-gate decision is not a mapping")
    try:
        decision = dict(outcome.decision)
    except Exception as exc:
        raise _StudentPathError("post-gate decision cannot be copied") from exc
    if _marker_keys(decision):
        raise _StudentPathError(
            "post-gate decision contains a failure/degradation/alternative/fallback marker"
        )
    if decision.get("status") != "applied":
        raise _StudentPathError("post-gate status is not clean applied")
    if decision.get("selection_error") is not None:
        raise _StudentPathError("post-gate selection error is not empty")
    allowed_rules = (
        {"original-already-in-A_pos", "model-argmax-inside-A_pos"}
        if route_kind == "student"
        else {"original-already-in-A_pos", "restricted-root-research"}
    )
    rule = decision.get("selection_rule")
    if rule not in allowed_rules:
        raise _StudentPathError("post-gate selection rule is not clean")
    if (
        decision.get("mode") != "A_pos"
        or decision.get("positional_only") is not True
        or decision.get("history_aware") is not False
    ):
        raise _StudentPathError("post-gate A_pos mode evidence differs")
    verified = decision.get("candidate_order_verified")
    if route_kind == "student":
        if verified is not True:
            raise _StudentPathError("student gate candidate order was not verified")
    elif type(verified) is not bool:
        raise _StudentPathError("fallback gate candidate-order evidence is invalid")
    if decision.get("legal_inventory_identity") != inventory_identity:
        raise _StudentPathError("post-gate legal inventory identity differs")
    parent_tier = decision.get("parent_tier")
    if parent_tier not in _WDL_TIERS or decision.get("selected_tier") != parent_tier:
        raise _StudentPathError("post-gate selected tier is outside parent A_pos")
    if type(decision.get("difficulty")) is not int or decision["difficulty"] != 9:
        raise _StudentPathError("post-gate difficulty differs from exact D9")
    source = decision.get("source")
    if route_kind == "student":
        if source != _STUDENT_SOURCE:
            raise _StudentPathError("student gate source identity differs")
    elif source != _EXACT_D9_SOURCE:
        raise _StudentPathError("fallback gate source identity differs")
    if type(decision.get("legal_move_count")) is not int or decision[
        "legal_move_count"
    ] != len(legal):
        raise _StudentPathError("post-gate legal move count differs")
    safe_count = decision.get("safe_move_count")
    if type(safe_count) is not int or safe_count <= 0 or safe_count > len(legal):
        raise _StudentPathError("post-gate safe move count differs")
    try:
        outcome_move = dict(outcome.move)
        selected_move = dict(decision["selected_move"])
        original_move = dict(decision["original_move"])
        outcome_key = _atomic_move_key(outcome_move)
        selected_key = _atomic_move_key(selected_move)
        original_key = _atomic_move_key(original_move)
    except Exception as exc:
        raise _StudentPathError("post-gate atomic move evidence is invalid") from exc
    if outcome_key != selected_key:
        raise _StudentPathError("post-gate outcome and selected move differ")
    if outcome_key not in set(legal_keys) or original_key not in set(legal_keys):
        raise _StudentPathError(
            "post-gate move is outside the original legal inventory"
        )
    if expected_original is not None and original_key != _atomic_move_key(
        expected_original
    ):
        raise _StudentPathError("post-gate original move identity differs")
    if rule == "original-already-in-A_pos" and outcome_key != original_key:
        raise _StudentPathError("original-already rule changed the original move")
    return outcome_move, decision


def _student_choice(
    context: _RouteContext,
    board: BoardState,
    legal: list[dict[str, Any]],
    legal_keys: tuple[tuple[Any, Any, Any], ...],
    inventory_identity: str,
) -> ClassicalAPosRouteOutcome:
    if context.production:
        student_context = (
            None
            if context.student_handle is None
            else _QUALIFIED_STUDENT_CONTEXTS.get(context.student_handle)
        )
        if student_context is None:
            raise _StudentPathError("qualified student context expired")
        _validate_qualified_model_runtime(student_context)
    probabilities, original_index = _student_probabilities(
        context,
        board,
        legal,
        legal_keys,
    )
    original = legal[original_index]
    _validate_bound_gate_runtime(
        context.gate,
        context.gate_constrain,
        production=context.production,
    )
    try:
        outcome = context.gate_constrain(
            context.gate,
            board,
            original,
            source=_STUDENT_SOURCE,
            difficulty=9,
            candidate_moves=legal,
            candidate_scores=probabilities,
            safe_selector=None,
            query_failure_move=original,
        )
    except Exception as exc:
        raise _StudentPathError("student gate raised") from exc
    move, decision = _validate_safety_outcome(
        outcome,
        legal=legal,
        legal_keys=legal_keys,
        inventory_identity=inventory_identity,
        route_kind="student",
        expected_original=original,
    )
    return ClassicalAPosRouteOutcome(
        move=move,
        route="student",
        safety_decision=decision,
        checkpoint_identity=context.checkpoint_identity,
        fallback_reason=None,
    )


def _fallback_choice(
    route: _OpaqueRoute,
    context: _RouteContext,
    board: BoardState,
    legal: list[dict[str, Any]],
    legal_keys: tuple[tuple[Any, Any, Any], ...],
    inventory_identity: str,
    *,
    reason: str,
) -> ClassicalAPosRouteOutcome:
    _validate_capability_method_bindings()
    factory_context = _EXACT_D9_FACTORY_CONTEXTS.get(context.exact_d9_factory)
    owner = None if factory_context is None else factory_context.owner_route
    binding_context = (
        None
        if factory_context is None
        else _EXACT_D9_IMPLEMENTATION_CONTEXTS.get(factory_context.implementation)
    )
    if (
        factory_context is None
        or binding_context is None
        or owner is None
        or owner() is not route
        or factory_context.gate is not context.gate
        or type(factory_context.difficulty) is not int
        or factory_context.difficulty != 9
    ):
        raise ResearchRouteUnavailable(
            "exact D9 fallback factory binding became unavailable"
        )
    _validate_bound_gate_runtime(
        context.gate,
        context.gate_constrain,
        production=context.production,
    )
    _validate_implementation_runtime(binding_context)
    try:
        issued_search = _FACTORY_CREATE_METHOD(context.exact_d9_factory)
    except Exception as exc:
        raise ResearchRouteUnavailable(
            f"exact D9 post-gate fallback could not create a fresh instance: {exc}"
        ) from exc
    if type(issued_search) is not IssuedExactD9Search:
        raise ResearchRouteUnavailable(
            "exact D9 fallback factory returned a non-issued search type"
        )
    search_context = _ISSUED_EXACT_D9_SEARCH_CONTEXTS.get(issued_search)
    if search_context is None:
        raise ResearchRouteUnavailable("exact D9 fallback search lacks issued evidence")
    assert context.seen_search_handles is not None
    assert context.seen_search_identities is not None
    if (
        issued_search in context.seen_search_handles
        or search_context.search_instance_identity in context.seen_search_identities
    ):
        raise ResearchRouteUnavailable(
            "exact D9 fallback attempted to reuse a search instance"
        )
    context.seen_search_handles.add(issued_search)
    context.seen_search_identities.add(search_context.search_instance_identity)
    try:
        attempt = _SEARCH_ISSUE_ATTEMPT_METHOD(issued_search)
    except Exception as exc:
        raise ResearchRouteUnavailable(
            f"exact D9 issued search could not create a one-shot attempt: {exc}"
        ) from exc
    if type(attempt) is not FreshExactD9PostGateAttempt:
        raise ResearchRouteUnavailable(
            "exact D9 fallback factory returned a non-fresh attempt type"
        )
    attempt_context = _FRESH_EXACT_D9_ATTEMPT_CONTEXTS.get(attempt)
    if attempt_context is None:
        raise ResearchRouteUnavailable(
            "exact D9 fallback attempt lacks issued evidence"
        )
    assert context.seen_attempts is not None
    assert context.seen_attempt_identities is not None
    if (
        attempt in context.seen_attempts
        or attempt_context.attempt_identity in context.seen_attempt_identities
    ):
        raise ResearchRouteUnavailable(
            "exact D9 fallback attempted to reuse a non-fresh instance"
        )
    context.seen_attempts.add(attempt)
    context.seen_attempt_identities.add(attempt_context.attempt_identity)
    if (
        attempt_context.implementation is not factory_context.implementation
        or attempt_context.factory is not context.exact_d9_factory
        or attempt_context.issued_search is not issued_search
        or search_context.implementation is not factory_context.implementation
        or search_context.factory is not context.exact_d9_factory
    ):
        raise ResearchRouteUnavailable("exact D9 fallback attempt binding differs")
    try:
        evidence = _ATTEMPT_CHOOSE_ONCE_METHOD(
            attempt,
            board,
            expected_legal_inventory_identity=inventory_identity,
        )
    except Exception as exc:
        raise ResearchRouteUnavailable(
            f"exact D9 post-gate fallback attempt failed closed: {exc}"
        ) from exc
    if type(evidence) is not ExactD9PostGateEvidence:
        raise ResearchRouteUnavailable("exact D9 fallback evidence type differs")
    if (
        attempt_context.raw_move is None
        or evidence.attempt_identity != attempt_context.attempt_identity
        or evidence.factory_identity != factory_context.factory_identity
        or evidence.search_instance_identity != search_context.search_instance_identity
        or evidence.search_initial_state_identity
        != search_context.search_initial_state_identity
        or evidence.search_invocation_identity
        != attempt_context.search_invocation_identity
        or evidence.route_contract_identity != binding_context.route_contract_identity
        or evidence.route_implementation_identity
        != binding_context.route_implementation_identity
        or evidence.route_effective_config_identity
        != binding_context.route_effective_config_identity
        or evidence.gate_runtime_identity != binding_context.gate_runtime_identity
        or evidence.gate_implementation_identity
        != binding_context.gate_implementation_identity
        or type(evidence.difficulty) is not int
        or evidence.difficulty != 9
        or evidence.legal_inventory_identity != inventory_identity
    ):
        raise ResearchRouteUnavailable(
            "exact D9 fallback freshness or identity evidence differs"
        )
    assert context.seen_fallback_outcomes is not None
    if any(
        evidence.outcome is prior_outcome
        for prior_outcome in context.seen_fallback_outcomes
    ):
        raise ResearchRouteUnavailable(
            "exact D9 fallback attempted to reuse a prior result object"
        )
    context.seen_fallback_outcomes.append(evidence.outcome)
    try:
        move, decision = _validate_safety_outcome(
            evidence.outcome,
            legal=legal,
            legal_keys=legal_keys,
            inventory_identity=inventory_identity,
            route_kind="fallback",
            expected_original=attempt_context.raw_move,
        )
    except _StudentPathError as exc:
        raise ResearchRouteUnavailable(
            "exact D9 fallback gate evidence failed closed"
        ) from exc
    return ClassicalAPosRouteOutcome(
        move=move,
        route="exact-d9-post-gate-fallback",
        safety_decision=decision,
        checkpoint_identity=context.checkpoint_identity,
        fallback_reason=reason,
    )


def _student_failure_reason(exc: Exception) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _choose_move(
    route: _OpaqueRoute,
    board: BoardState,
) -> ClassicalAPosRouteOutcome:
    if type(route) not in {
        ClassicalAPosStudentRoute,
        _TestClassicalAPosStudentRoute,
    }:
        raise ResearchRouteUnavailable("research route type is not issued")
    context = _ROUTE_CONTEXTS.get(route)
    if context is None:
        raise ResearchRouteUnavailable("research route lacks a private context")
    if not isinstance(board, BoardState):
        raise ResearchRouteUnavailable("research route requires a BoardState")
    _validate_bound_gate_runtime(
        context.gate,
        context.gate_constrain,
        production=context.production,
    )
    try:
        legal, legal_keys, inventory_identity = _complete_legal_inventory(board)
    except Exception as exc:
        raise ResearchRouteUnavailable(
            "research route cannot establish a complete legal inventory"
        ) from exc
    try:
        return _student_choice(
            context,
            board,
            legal,
            legal_keys,
            inventory_identity,
        )
    except Exception as exc:
        reason = _student_failure_reason(exc)
    return _fallback_choice(
        route,
        context,
        board,
        legal,
        legal_keys,
        inventory_identity,
        reason=reason,
    )


__all__ = [
    "ClassicalAPosRouteOutcome",
    "ClassicalAPosStudentRoute",
    "ExactD9PostGateEvidence",
    "ExactD9PostGateFactory",
    "ExactD9RouteImplementationBinding",
    "FreshExactD9PostGateAttempt",
    "IssuedExactD9Search",
    "QualifiedCompleteSeedBinding",
    "QualifiedStudentHandle",
    "ResearchRouteUnavailable",
    "StudentQualificationError",
    "load_qualified_complete_student",
]
