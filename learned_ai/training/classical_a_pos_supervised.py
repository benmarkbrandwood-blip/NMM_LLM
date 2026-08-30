"""Pure offline masked-CE core for the classical ``A_pos`` student.

This module contains no rollout, reward, value, A2C, PPO, or online-teacher
path. Variable-length legal inventories are trained one state at a time inside
each optimizer batch so only logits selected by the hard ``A_pos`` mask enter
the loss graph.
"""

from __future__ import annotations

import hashlib
import math
import random
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from learned_ai.models.scaffolded_encoder import VALUE_INPUT_DIM, encode_position
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.training.checkpoint_envelope import (
    CheckpointCompatibilityError,
    CheckpointDescriptor,
    CheckpointPayload,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from learned_ai.training.classical_a_pos_corpus import (
    CORPUS_SCHEMA,
    FrozenCorpus,
    FrozenCorpusExample,
    action_key,
)
from learned_ai.training.run_contract import canonical_sha256


FEATURE_SCHEMA = "nmm.scaffolded-base62.v1"
LABEL_SCHEMA = "nmm.exact-d9-post-gate-a-pos-one-hot.v1"
TRAINER_ID = "nmm.classical-a-pos-offline-supervised.v1"
PRODUCTION_TRAIN_EXAMPLES = 14_336
PRODUCTION_BATCH_SIZE = 128
PRODUCTION_EPOCHS = 20
PRODUCTION_BATCHES_PER_EPOCH = 112
PRODUCTION_UPDATES_PER_SEED = 2_240
PRODUCTION_SEEDS = (2026083001, 2026083002, 2026083003)
PRODUCTION_POLICY_HIDDEN = (128, 64)
SEED_LATEST_ROLE = "supervised_seed_latest"
SEED_COMPLETE_ROLE = "supervised_seed_complete"
SMOKE_DISPOSABLE_ROLE = "supervised_smoke_disposable"


def _adam_group_contract() -> dict[str, Any]:
    return {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": False,
    }


class SupervisedTrainingError(RuntimeError):
    """The pure supervised contract could not be executed exactly."""


class SupervisedPurpose(str, Enum):
    """Controller-frozen purpose for an opaque supervised capability."""

    SEED = "seed"
    SMOKE = "smoke"


@dataclass(frozen=True)
class FreshSupervisedLineage:
    lineage_kind: str
    seed: int
    expected_initial_model_state_identity: str
    start_mode: str
    purpose: SupervisedPurpose

    def to_dict(self) -> dict[str, Any]:
        return {
            "lineage_kind": self.lineage_kind,
            "seed": self.seed,
            "expected_initial_model_state_identity": (
                self.expected_initial_model_state_identity
            ),
            "start_mode": self.start_mode,
            "purpose": self.purpose.value,
        }


_FRESH_MODEL_LINEAGES: weakref.WeakKeyDictionary[
    ScaffoldedPolicyNet,
    FreshSupervisedLineage,
] = weakref.WeakKeyDictionary()
_MODEL_ROUTE: weakref.WeakKeyDictionary[ScaffoldedPolicyNet, str] = (
    weakref.WeakKeyDictionary()
)
_FACTORY_OPTIMIZERS: weakref.WeakKeyDictionary[
    ScaffoldedPolicyNet,
    weakref.ReferenceType[torch.optim.Adam],
] = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class SupervisedExample:
    features: torch.Tensor
    a_pos_mask: torch.Tensor
    teacher_index: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.features, torch.Tensor)
            or self.features.ndim != 2
            or self.features.shape[1] != 62
            or self.features.shape[0] <= 1
            or not self.features.is_floating_point()
            or not bool(torch.isfinite(self.features).all())
        ):
            raise SupervisedTrainingError(
                "supervised features must be finite [legal,62] floats"
            )
        if (
            not isinstance(self.a_pos_mask, torch.Tensor)
            or self.a_pos_mask.dtype is not torch.bool
            or self.a_pos_mask.ndim != 1
            or self.a_pos_mask.shape[0] != self.features.shape[0]
            or int(self.a_pos_mask.sum().item()) <= 1
        ):
            raise SupervisedTrainingError(
                "supervised A_pos mask must align with legal features and be informative"
            )
        if (
            isinstance(self.teacher_index, bool)
            or not isinstance(self.teacher_index, int)
            or not 0 <= self.teacher_index < self.features.shape[0]
            or not bool(self.a_pos_mask[self.teacher_index])
        ):
            raise SupervisedTrainingError("teacher index must be inside A_pos")


@dataclass(frozen=True)
class SupervisedLoopConfig:
    batch_size: int
    epochs: int
    grad_clip_norm: float
    permutation_seed: int

    def __post_init__(self) -> None:
        for name in ("batch_size", "epochs", "permutation_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SupervisedTrainingError(f"{name} must be a positive integer")
        if (
            isinstance(self.grad_clip_norm, bool)
            or not isinstance(self.grad_clip_norm, (int, float))
            or not math.isfinite(float(self.grad_clip_norm))
            or float(self.grad_clip_norm) <= 0.0
        ):
            raise SupervisedTrainingError("grad_clip_norm must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_semantics": (
                "offline-supervised-hard-a-pos-masked-one-hot-cross-entropy"
            ),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "grad_clip_norm": float(self.grad_clip_norm),
            "permutation_seed": self.permutation_seed,
            "optimizer": {
                "kind": "Adam",
                "parameter_scope": "policy_mlp-only",
                "lr": 1e-3,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "schedule": "constant",
            },
            "rl": {
                "enabled": False,
                "a2c": False,
                "ppo": False,
                "value_loss": False,
                "entropy": False,
            },
        }


def production_loop_config(seed: int) -> SupervisedLoopConfig:
    if seed not in PRODUCTION_SEEDS:
        raise SupervisedTrainingError("production student seed is not frozen")
    return SupervisedLoopConfig(
        batch_size=PRODUCTION_BATCH_SIZE,
        epochs=PRODUCTION_EPOCHS,
        grad_clip_norm=1.0,
        permutation_seed=seed,
    )


def _validate_production_config(config: SupervisedLoopConfig) -> None:
    expected = production_loop_config(config.permutation_seed)
    if config != expected:
        raise SupervisedTrainingError(
            "production loop must be batch128/20epochs/clip1 with a frozen seed"
        )


@dataclass(frozen=True)
class SupervisedState:
    """Next-batch cursor plus the active epoch permutation."""

    epoch: int
    batch_in_epoch: int
    update_count: int
    sample_cursor: int
    permutation: tuple[int, ...] | None
    completed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "update_count": self.update_count,
            "sample_cursor": self.sample_cursor,
            "permutation": (
                None if self.permutation is None else list(self.permutation)
            ),
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, value: Any) -> SupervisedState:
        if not isinstance(value, Mapping) or set(value) != {
            "epoch",
            "batch_in_epoch",
            "update_count",
            "sample_cursor",
            "permutation",
            "completed",
        }:
            raise SupervisedTrainingError("supervised cursor keys differ")
        integers: dict[str, int] = {}
        for field in ("epoch", "batch_in_epoch", "update_count", "sample_cursor"):
            item = value[field]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise SupervisedTrainingError(f"supervised cursor {field} is invalid")
            integers[field] = item
        raw_permutation = value["permutation"]
        if raw_permutation is None:
            permutation = None
        elif isinstance(raw_permutation, list) and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in raw_permutation
        ):
            permutation = tuple(raw_permutation)
        else:
            raise SupervisedTrainingError("supervised cursor permutation is invalid")
        if not isinstance(value["completed"], bool):
            raise SupervisedTrainingError("supervised cursor completion is invalid")
        return cls(
            epoch=integers["epoch"],
            batch_in_epoch=integers["batch_in_epoch"],
            update_count=integers["update_count"],
            sample_cursor=integers["sample_cursor"],
            permutation=permutation,
            completed=value["completed"],
        )


@dataclass(frozen=True)
class SupervisedUpdateMetrics:
    update_count: int
    batch_indices: tuple[int, ...]
    loss: float
    gradient_norm: float


@dataclass(frozen=True)
class SupervisedSmokeResult:
    training_semantics: str
    optimizer_updates: int
    finite_loss: bool
    finite_gradient_norm: bool
    parameters_changed: bool
    reinforcement_learning: bool
    loss: float
    gradient_norm: float


@dataclass(frozen=True)
class SupervisedSmokeCursor:
    """Terminal, disposable evidence cursor; it is never a seed cursor."""

    schema: str
    terminal: bool
    resumable: bool
    update_count: int
    optimizer_updates: int
    batch_start: int
    batch_stop: int
    permutation_calls: int
    rng_before_identity: str
    rng_after_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "terminal": self.terminal,
            "resumable": self.resumable,
            "update_count": self.update_count,
            "optimizer_updates": self.optimizer_updates,
            "batch_start": self.batch_start,
            "batch_stop": self.batch_stop,
            "permutation_calls": self.permutation_calls,
            "rng_before_identity": self.rng_before_identity,
            "rng_after_identity": self.rng_after_identity,
        }


@dataclass(frozen=True)
class _SmokeCompletionContext:
    cursor: SupervisedSmokeCursor
    result: SupervisedSmokeResult
    batch_identity: str
    initial_model_state_identity: str
    completed_model_state_identity: str
    completed_optimizer_state_identity: str
    value_state_identity: str
    permutation_generator_identity: str


_SESSION_CREATION_TOKEN = object()
_PLAN_BINDING_CREATION_TOKEN = object()


class SupervisedPlanBinding:
    """Opaque plan identity input; no production issuer exists in this slice."""

    __slots__ = ("__weakref__",)

    def __init__(self, token: object) -> None:
        if token is not _PLAN_BINDING_CREATION_TOKEN:
            raise SupervisedTrainingError(
                "supervised plan bindings are controller-issued only"
            )

    def __reduce__(self) -> Any:
        raise TypeError("supervised plan bindings cannot be serialized")


@dataclass(frozen=True)
class _PlanBindingContext:
    corpus_identity: str
    split_identity: str
    plan_identity: str
    run_id: str
    experiment_id: str
    seed: int
    purpose: SupervisedPurpose


_PLAN_BINDING_CONTEXTS: weakref.WeakKeyDictionary[
    SupervisedPlanBinding,
    _PlanBindingContext,
] = weakref.WeakKeyDictionary()


class _OpaqueSupervisedSession:
    __slots__ = ("__weakref__",)

    def __init__(self, token: object) -> None:
        if token is not _SESSION_CREATION_TOKEN:
            raise SupervisedTrainingError(
                "production supervised sessions are factory-issued only"
            )

    def __reduce__(self) -> Any:
        raise TypeError("production supervised sessions cannot be serialized")


class FreshSupervisedSession(_OpaqueSupervisedSession):
    """Opaque capability for one fresh production attempt."""

    __slots__ = ()


class ResumedSupervisedSession(_OpaqueSupervisedSession):
    """Opaque capability issued only after an exact checkpoint restore."""

    __slots__ = ()


class SupervisedSmokeSession(_OpaqueSupervisedSession):
    """Opaque capability for one disposable supervised smoke update."""

    __slots__ = ()


@dataclass(frozen=True)
class _ProductionSessionContext:
    model: ScaffoldedPolicyNet
    optimizer: torch.optim.Adam
    permutation_generator: torch.Generator
    corpus: FrozenCorpus
    examples: tuple[SupervisedExample, ...]
    corpus_identity: str
    split_identity: str
    training_records_identity: str
    encoded_payload_identity: str
    plan_identity: str
    run_id: str
    experiment_id: str
    seed: int
    config: SupervisedLoopConfig
    lineage: FreshSupervisedLineage
    purpose: SupervisedPurpose


_SESSION_CONTEXTS: weakref.WeakKeyDictionary[
    _OpaqueSupervisedSession,
    _ProductionSessionContext,
] = weakref.WeakKeyDictionary()
_MODEL_SESSION_OWNERS: weakref.WeakKeyDictionary[
    ScaffoldedPolicyNet,
    weakref.ReferenceType[_OpaqueSupervisedSession],
] = weakref.WeakKeyDictionary()
_OPTIMIZER_SESSION_OWNERS: weakref.WeakKeyDictionary[
    torch.optim.Adam,
    weakref.ReferenceType[_OpaqueSupervisedSession],
] = weakref.WeakKeyDictionary()
_GENERATOR_SESSION_OWNERS: dict[
    int,
    weakref.ReferenceType[_OpaqueSupervisedSession],
] = {}
_SMOKE_COMPLETIONS: weakref.WeakKeyDictionary[
    SupervisedSmokeSession,
    _SmokeCompletionContext,
] = weakref.WeakKeyDictionary()
_SAVED_SMOKE_SESSIONS: weakref.WeakSet[SupervisedSmokeSession] = weakref.WeakSet()


def hard_a_pos_masked_cross_entropy(
    logits: torch.Tensor,
    a_pos_mask: torch.Tensor,
    *,
    teacher_index: int,
) -> torch.Tensor:
    """One-hot CE whose graph contains only ``A_pos`` logits."""
    if logits.ndim != 1:
        raise SupervisedTrainingError("policy logits must be one-dimensional")
    if not bool(torch.isfinite(logits).all()):
        raise SupervisedTrainingError(
            "full legal policy logits contain non-finite values"
        )
    if (
        a_pos_mask.dtype is not torch.bool
        or a_pos_mask.ndim != 1
        or a_pos_mask.shape != logits.shape
        or int(a_pos_mask.sum().item()) <= 1
    ):
        raise SupervisedTrainingError("hard A_pos mask/logit shape differs")
    if (
        isinstance(teacher_index, bool)
        or not isinstance(teacher_index, int)
        or not 0 <= teacher_index < logits.shape[0]
        or not bool(a_pos_mask[teacher_index])
    ):
        raise SupervisedTrainingError("teacher target is outside A_pos")
    safe_logits = logits[a_pos_mask]
    safe_indices = torch.nonzero(a_pos_mask, as_tuple=False).squeeze(1)
    target_position = torch.nonzero(
        safe_indices == teacher_index, as_tuple=False
    ).squeeze()
    return F.cross_entropy(safe_logits.unsqueeze(0), target_position.reshape(1))


def _encode_frozen_training_examples(
    corpus: Any,
    *,
    encoder: Callable[..., Any] = encode_position,
) -> tuple[SupervisedExample, ...]:
    """Encode train records with every forbidden advisor explicitly absent."""
    output: list[SupervisedExample] = []
    for item in corpus.examples:
        if item.split != "train":
            continue
        encoded = encoder(
            item.board,
            item.candidate_color,
            sentinel_advisor=None,
            db=None,
            value_net=None,
            specialist_db=None,
            wdl_db=None,
            strict=True,
        )
        if encoded is None:
            raise SupervisedTrainingError("corpus state encoded as terminal")
        observed = tuple(action_key(move) for move in encoded.legal_moves)
        expected = tuple(action_key(move) for move in item.legal_actions)
        if observed != expected:
            raise SupervisedTrainingError(
                "encoder legal action order differs from frozen corpus"
            )
        features = torch.as_tensor(
            np.asarray(encoded.feat_matrix, dtype=np.float32),
            dtype=torch.float32,
            device="cpu",
        ).clone()
        output.append(
            SupervisedExample(
                features=features,
                a_pos_mask=torch.tensor(item.a_pos_mask, dtype=torch.bool),
                teacher_index=item.teacher_index,
            )
        )
    if not output:
        raise SupervisedTrainingError("frozen corpus has no train examples")
    return tuple(output)


def encode_frozen_training_examples(
    corpus: FrozenCorpus,
) -> tuple[SupervisedExample, ...]:
    """Strict production encoding from a verified frozen corpus only."""
    if not isinstance(corpus, FrozenCorpus):
        raise SupervisedTrainingError(
            "production training input must be a verified FrozenCorpus"
        )
    output = _encode_frozen_training_examples(corpus, encoder=encode_position)
    if len(output) != PRODUCTION_TRAIN_EXAMPLES:
        raise SupervisedTrainingError(
            "production corpus must contain exactly 14,336 train examples"
        )
    return output


def _ordered_training_records_identity(corpus: FrozenCorpus) -> str:
    if any(not isinstance(item, FrozenCorpusExample) for item in corpus.examples):
        raise SupervisedTrainingError("production corpus contains a non-frozen record")
    records = tuple(item for item in corpus.examples if item.split == "train")
    if len(records) != PRODUCTION_TRAIN_EXAMPLES:
        raise SupervisedTrainingError(
            "production corpus ordered training records differ"
        )
    return canonical_sha256(
        {
            "schema": "nmm.classical-a-pos-ordered-training-records.v1",
            "records": [
                {
                    "index": index,
                    "record_sha256": _required_sha256(
                        item.record_sha256,
                        field="training record identity",
                    ),
                    "state_record_identity": _required_sha256(
                        item.state_record_identity,
                        field="training state record identity",
                    ),
                    "example_id": item.example_id,
                }
                for index, item in enumerate(records)
            ],
        }
    )


def _encoded_training_payload_identity(
    examples: Sequence[SupervisedExample],
) -> str:
    if len(examples) != PRODUCTION_TRAIN_EXAMPLES:
        raise SupervisedTrainingError("production encoded payload count differs")
    return canonical_sha256(
        {
            "schema": "nmm.classical-a-pos-encoded-training-payload.v1",
            "examples": [
                {
                    "index": index,
                    "features": canonical_training_state_identity(item.features),
                    "a_pos_mask": item.a_pos_mask.tolist(),
                    "teacher_index": item.teacher_index,
                }
                for index, item in enumerate(examples)
            ],
        }
    )


def _smoke_batch_identity(
    examples: Sequence[SupervisedExample],
) -> str:
    """Bind the ordered first 128 examples under a smoke-only domain."""
    if len(examples) != PRODUCTION_BATCH_SIZE:
        raise SupervisedTrainingError(
            "supervised smoke batch must contain exactly 128 encoded examples"
        )
    if any(type(item) is not SupervisedExample for item in examples):
        raise SupervisedTrainingError(
            "supervised smoke batch contains a non-supervised example"
        )
    return canonical_sha256(
        {
            "schema": "nmm.classical-a-pos-supervised-smoke-batch.v1",
            "source_range": {
                "start": 0,
                "stop": PRODUCTION_BATCH_SIZE,
            },
            "examples": [
                {
                    "source_index": index,
                    "features": canonical_training_state_identity(item.features),
                    "a_pos_mask": item.a_pos_mask.tolist(),
                    "teacher_index": item.teacher_index,
                }
                for index, item in enumerate(examples)
            ],
        }
    )


def make_policy_only_adam(
    model: torch.nn.Module, *, lr: float = 1e-3
) -> torch.optim.Adam:
    if float(lr) != 1e-3:
        raise SupervisedTrainingError("frozen supervised Adam lr must be 1e-3")
    _model_config(model)
    policy = getattr(model, "policy_mlp", None)
    value = getattr(model, "value_mlp", None)
    if not isinstance(policy, torch.nn.Module) or not isinstance(
        value, torch.nn.Module
    ):
        raise SupervisedTrainingError("student model lacks separate policy/value MLPs")
    for parameter in value.parameters():
        parameter.requires_grad_(False)
    parameters = list(policy.parameters())
    if not parameters:
        raise SupervisedTrainingError("student policy MLP has no parameters")
    return torch.optim.Adam(
        parameters,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )


def _model_state_identity(model: ScaffoldedPolicyNet) -> str:
    return canonical_training_state_identity(model.state_dict())


def _require_factory_lineage(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    seed: int,
    purpose: SupervisedPurpose,
    allowed_routes: set[str],
    require_initial_state: bool,
) -> FreshSupervisedLineage:
    _model_config(model)
    lineage = _FRESH_MODEL_LINEAGES.get(model)
    route = _MODEL_ROUTE.get(model)
    optimizer_reference = _FACTORY_OPTIMIZERS.get(model)
    if (
        lineage is None
        or lineage.lineage_kind != f"fresh_supervised_{purpose.value}"
        or lineage.start_mode != "fresh"
        or lineage.seed != seed
        or lineage.purpose is not purpose
        or route not in allowed_routes
        or optimizer_reference is None
        or optimizer_reference() is not optimizer
    ):
        raise SupervisedTrainingError(
            "production model lacks matching factory-signed fresh lineage"
        )
    _required_sha256(
        lineage.expected_initial_model_state_identity,
        field="expected initial model state identity",
    )
    if require_initial_state and _model_state_identity(model) != (
        lineage.expected_initial_model_state_identity
    ):
        raise SupervisedTrainingError(
            "production model differs from its frozen seed initialization"
        )
    return lineage


def _create_fresh_production_student(
    seed: int,
    *,
    purpose: SupervisedPurpose,
) -> tuple[ScaffoldedPolicyNet, torch.optim.Adam]:
    """Create a CPU-seeded student while restoring every caller CPU RNG."""
    if seed not in PRODUCTION_SEEDS:
        raise SupervisedTrainingError("production student seed is not frozen")
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = ScaffoldedPolicyNet(
            move_feat_dim=62,
            policy_hidden=PRODUCTION_POLICY_HIDDEN,
            value_hidden=(),
            dropout=0.0,
        )
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(torch_rng_state)
    optimizer = make_policy_only_adam(model)
    lineage = FreshSupervisedLineage(
        lineage_kind=f"fresh_supervised_{purpose.value}",
        seed=seed,
        expected_initial_model_state_identity=_model_state_identity(model),
        start_mode="fresh",
        purpose=purpose,
    )
    _FRESH_MODEL_LINEAGES[model] = lineage
    _MODEL_ROUTE[model] = "fresh"
    _FACTORY_OPTIMIZERS[model] = weakref.ref(optimizer)
    return model, optimizer


def _activate_session(
    session_type: (
        type[FreshSupervisedSession]
        | type[ResumedSupervisedSession]
        | type[SupervisedSmokeSession]
    ),
    context: _ProductionSessionContext,
    *,
    replaced_session: _OpaqueSupervisedSession | None = None,
) -> FreshSupervisedSession | ResumedSupervisedSession | SupervisedSmokeSession:
    session = session_type(_SESSION_CREATION_TOKEN)
    if replaced_session is not None:
        _SESSION_CONTEXTS.pop(replaced_session, None)
    _SESSION_CONTEXTS[session] = context
    _MODEL_SESSION_OWNERS[context.model] = weakref.ref(session)
    _OPTIMIZER_SESSION_OWNERS[context.optimizer] = weakref.ref(session)
    _GENERATOR_SESSION_OWNERS[id(context.permutation_generator)] = weakref.ref(session)
    return session


def _production_session_context(
    session: (
        FreshSupervisedSession | ResumedSupervisedSession | SupervisedSmokeSession
    ),
    *,
    allowed_types: tuple[
        type[FreshSupervisedSession]
        | type[ResumedSupervisedSession]
        | type[SupervisedSmokeSession],
        ...,
    ],
    expected_purpose: SupervisedPurpose,
) -> _ProductionSessionContext:
    if type(session) not in allowed_types:
        raise SupervisedTrainingError("supervised session kind is not allowed here")
    context = _SESSION_CONTEXTS.get(session)
    model_owner = None if context is None else _MODEL_SESSION_OWNERS.get(context.model)
    optimizer_owner = (
        None if context is None else _OPTIMIZER_SESSION_OWNERS.get(context.optimizer)
    )
    generator_owner = (
        None
        if context is None
        else _GENERATOR_SESSION_OWNERS.get(id(context.permutation_generator))
    )
    if (
        context is None
        or model_owner is None
        or model_owner() is not session
        or optimizer_owner is None
        or optimizer_owner() is not session
        or generator_owner is None
        or generator_owner() is not session
        or _FACTORY_OPTIMIZERS.get(context.model) is None
        or _FACTORY_OPTIMIZERS[context.model]() is not context.optimizer
        or _FRESH_MODEL_LINEAGES.get(context.model) != context.lineage
        or context.purpose is not expected_purpose
        or context.lineage.purpose is not expected_purpose
    ):
        raise SupervisedTrainingError(
            "production session capability/model/optimizer binding differs"
        )
    _validate_production_config(context.config)
    if (
        not isinstance(context.corpus, FrozenCorpus)
        or context.corpus.corpus_identity != context.corpus_identity
        or context.corpus.split_identity != context.split_identity
        or len(context.examples) != PRODUCTION_TRAIN_EXAMPLES
        or any(not isinstance(item, SupervisedExample) for item in context.examples)
        or context.seed != context.config.permutation_seed
        or context.lineage.seed != context.seed
        or type(context.purpose) is not SupervisedPurpose
        or not isinstance(context.run_id, str)
        or not context.run_id
        or not isinstance(context.experiment_id, str)
        or not context.experiment_id
    ):
        raise SupervisedTrainingError("production session frozen binding differs")
    _required_sha256(context.corpus_identity, field="corpus_identity")
    _required_sha256(context.split_identity, field="split_identity")
    _required_sha256(
        context.training_records_identity,
        field="training_records_identity",
    )
    _required_sha256(
        context.encoded_payload_identity,
        field="encoded_payload_identity",
    )
    _required_sha256(context.plan_identity, field="plan_identity")
    if (
        _ordered_training_records_identity(context.corpus)
        != context.training_records_identity
        or _encoded_training_payload_identity(context.examples)
        != context.encoded_payload_identity
    ):
        raise SupervisedTrainingError(
            "production session ordered corpus/encoded payload binding differs"
        )
    return context


def _issue_test_plan_binding(
    corpus: FrozenCorpus,
    *,
    plan_identity: str,
    run_id: str,
    experiment_id: str,
    seed: int,
    purpose: SupervisedPurpose | str,
) -> SupervisedPlanBinding:
    """Issue a non-launch test binding; the future controller must not call this."""
    if not isinstance(corpus, FrozenCorpus):
        raise SupervisedTrainingError("test plan binding requires FrozenCorpus")
    try:
        frozen_purpose = SupervisedPurpose(purpose)
    except (TypeError, ValueError) as exc:
        raise SupervisedTrainingError("test plan purpose is invalid") from exc
    context = _PlanBindingContext(
        corpus_identity=_required_sha256(
            corpus.corpus_identity,
            field="corpus_identity",
        ),
        split_identity=_required_sha256(
            corpus.split_identity,
            field="split_identity",
        ),
        plan_identity=_required_sha256(plan_identity, field="plan_identity"),
        run_id=run_id,
        experiment_id=experiment_id,
        seed=seed,
        purpose=frozen_purpose,
    )
    if not context.run_id or not context.experiment_id:
        raise SupervisedTrainingError("test plan run/experiment identity is empty")
    production_loop_config(seed)
    binding = SupervisedPlanBinding(_PLAN_BINDING_CREATION_TOKEN)
    _PLAN_BINDING_CONTEXTS[binding] = context
    return binding


def _consume_plan_binding(
    corpus: FrozenCorpus,
    *,
    plan_binding: SupervisedPlanBinding,
    expected_purpose: SupervisedPurpose,
) -> tuple[_ProductionSessionContext, SupervisedState]:
    if not isinstance(corpus, FrozenCorpus):
        raise SupervisedTrainingError(
            "production session requires a verified FrozenCorpus"
        )
    if type(plan_binding) is not SupervisedPlanBinding:
        raise SupervisedTrainingError(
            "production session requires a controller plan binding"
        )
    binding = _PLAN_BINDING_CONTEXTS.pop(plan_binding, None)
    if binding is None:
        raise SupervisedTrainingError(
            "production session lacks an unused controller plan binding"
        )
    if binding.purpose is not expected_purpose:
        raise SupervisedTrainingError(
            "controller plan purpose differs from requested supervised session"
        )
    corpus_identity = _required_sha256(corpus.corpus_identity, field="corpus_identity")
    split_identity = _required_sha256(corpus.split_identity, field="split_identity")
    if (
        binding.corpus_identity != corpus_identity
        or binding.split_identity != split_identity
    ):
        raise SupervisedTrainingError("plan binding and FrozenCorpus identity differ")
    config = production_loop_config(binding.seed)
    examples = encode_frozen_training_examples(corpus)
    training_records_identity = _ordered_training_records_identity(corpus)
    encoded_payload_identity = _encoded_training_payload_identity(examples)
    model, optimizer = _create_fresh_production_student(
        binding.seed,
        purpose=expected_purpose,
    )
    permutation_generator = torch.Generator(device="cpu")
    state = create_fresh_supervised_state(
        permutation_generator=permutation_generator,
        config=config,
    )
    _validate_permutation_generator(
        permutation_generator,
        state,
        sample_count=PRODUCTION_TRAIN_EXAMPLES,
        config=config,
    )
    lineage = _FRESH_MODEL_LINEAGES[model]
    context = _ProductionSessionContext(
        model=model,
        optimizer=optimizer,
        permutation_generator=permutation_generator,
        corpus=corpus,
        examples=examples,
        corpus_identity=corpus_identity,
        split_identity=split_identity,
        training_records_identity=training_records_identity,
        encoded_payload_identity=encoded_payload_identity,
        plan_identity=binding.plan_identity,
        run_id=binding.run_id,
        experiment_id=binding.experiment_id,
        seed=binding.seed,
        config=config,
        lineage=lineage,
        purpose=expected_purpose,
    )
    return context, state


def issue_fresh_supervised_session(
    corpus: FrozenCorpus,
    *,
    plan_binding: SupervisedPlanBinding,
) -> tuple[FreshSupervisedSession, SupervisedState]:
    """Consume one seed-purpose binding; this slice has no launch issuer."""
    context, state = _consume_plan_binding(
        corpus,
        plan_binding=plan_binding,
        expected_purpose=SupervisedPurpose.SEED,
    )
    session = _activate_session(FreshSupervisedSession, context)
    assert isinstance(session, FreshSupervisedSession)
    return session, state


def issue_supervised_smoke_session(
    corpus: FrozenCorpus,
    *,
    plan_binding: SupervisedPlanBinding,
) -> SupervisedSmokeSession:
    """Consume one smoke-purpose binding into a disposable capability."""
    context, _unused_seed_cursor = _consume_plan_binding(
        corpus,
        plan_binding=plan_binding,
        expected_purpose=SupervisedPurpose.SMOKE,
    )
    session = _activate_session(SupervisedSmokeSession, context)
    assert isinstance(session, SupervisedSmokeSession)
    return session


def create_fresh_supervised_state(
    *,
    permutation_generator: torch.Generator,
    config: SupervisedLoopConfig,
) -> SupervisedState:
    if permutation_generator.device.type != "cpu":
        raise SupervisedTrainingError("permutation generator must be CPU-only")
    permutation_generator.manual_seed(config.permutation_seed)
    state = SupervisedState(
        epoch=0,
        batch_in_epoch=0,
        update_count=0,
        sample_cursor=0,
        permutation=None,
        completed=False,
    )
    return state


def _batches_per_epoch(
    *,
    sample_count: int,
    config: SupervisedLoopConfig,
) -> int:
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or sample_count % config.batch_size != 0
    ):
        raise SupervisedTrainingError(
            "supervised sample count must form full optimizer batches"
        )
    return sample_count // config.batch_size


def _validate_supervised_state(
    state: SupervisedState,
    *,
    sample_count: int,
    config: SupervisedLoopConfig,
) -> int:
    if not isinstance(state, SupervisedState):
        raise SupervisedTrainingError("supervised cursor type differs")
    for field in ("epoch", "batch_in_epoch", "update_count", "sample_cursor"):
        value = getattr(state, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SupervisedTrainingError(f"supervised cursor {field} is invalid")
    if not isinstance(state.completed, bool):
        raise SupervisedTrainingError("supervised completion flag is invalid")
    batches_per_epoch = _batches_per_epoch(
        sample_count=sample_count,
        config=config,
    )
    if not 0 <= state.epoch <= config.epochs:
        raise SupervisedTrainingError("supervised epoch cursor is out of bounds")
    if not 0 <= state.batch_in_epoch < batches_per_epoch and not (
        state.completed and state.epoch == config.epochs and state.batch_in_epoch == 0
    ):
        raise SupervisedTrainingError("supervised batch cursor is out of bounds")
    expected_completed = state.epoch == config.epochs
    if state.completed != expected_completed:
        raise SupervisedTrainingError("supervised completion flag is inconsistent")
    expected_updates = (
        config.epochs * batches_per_epoch
        if state.completed
        else state.epoch * batches_per_epoch + state.batch_in_epoch
    )
    if state.update_count != expected_updates:
        raise SupervisedTrainingError("supervised update cursor algebra differs")
    expected_sample_cursor = (
        0 if state.completed else state.batch_in_epoch * config.batch_size
    )
    if state.sample_cursor != expected_sample_cursor:
        raise SupervisedTrainingError("supervised sample cursor algebra differs")
    if state.completed or state.batch_in_epoch == 0:
        if state.permutation is not None:
            raise SupervisedTrainingError(
                "inactive epoch cursor must not retain a permutation"
            )
    elif (
        state.permutation is None
        or not isinstance(state.permutation, tuple)
        or len(state.permutation) != sample_count
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in state.permutation
        )
        or set(state.permutation) != set(range(sample_count))
    ):
        raise SupervisedTrainingError("active epoch permutation differs")
    return batches_per_epoch


def _validate_permutation_generator(
    permutation_generator: torch.Generator,
    state: SupervisedState,
    *,
    sample_count: int,
    config: SupervisedLoopConfig,
) -> None:
    if type(permutation_generator) is not torch.Generator:
        raise SupervisedTrainingError(
            "permutation generator must be the frozen torch.Generator"
        )
    if permutation_generator.device.type != "cpu":
        raise SupervisedTrainingError("permutation generator must be CPU-only")
    _validate_supervised_state(
        state,
        sample_count=sample_count,
        config=config,
    )
    replay = torch.Generator(device="cpu")
    replay.manual_seed(config.permutation_seed)
    expected_active_permutation: tuple[int, ...] | None = None
    generated_epochs = state.epoch + int(state.permutation is not None)
    for _epoch in range(generated_epochs):
        expected_active_permutation = tuple(
            int(value)
            for value in torch.randperm(sample_count, generator=replay).tolist()
        )
    if state.permutation is not None and state.permutation != (
        expected_active_permutation
    ):
        raise SupervisedTrainingError(
            "active permutation differs from deterministic generator replay"
        )
    if not torch.equal(permutation_generator.get_state(), replay.get_state()):
        raise SupervisedTrainingError(
            "permutation generator state differs from deterministic cursor replay"
        )


def _validate_checkpoint_permutation_state(
    rng_state: Any,
    state: SupervisedState,
    *,
    sample_count: int,
    config: SupervisedLoopConfig,
) -> None:
    if not isinstance(rng_state, Mapping):
        raise SupervisedTrainingError("checkpoint RNG state is not a mapping")
    components = rng_state.get("components")
    if not isinstance(components, Mapping) or set(components) != {
        "permutation_generator"
    }:
        raise SupervisedTrainingError(
            "checkpoint permutation generator component differs"
        )
    component_state = components["permutation_generator"]
    if not isinstance(component_state, torch.Tensor):
        raise SupervisedTrainingError(
            "checkpoint permutation generator state is not a tensor"
        )
    replay_target = torch.Generator(device="cpu")
    try:
        replay_target.set_state(component_state.detach().cpu())
    except RuntimeError as exc:
        raise SupervisedTrainingError(
            "checkpoint permutation generator state is invalid"
        ) from exc
    _validate_permutation_generator(
        replay_target,
        state,
        sample_count=sample_count,
        config=config,
    )


def _validate_optimizer_scope(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    _model_config(model)
    if type(optimizer) is not torch.optim.Adam:
        raise SupervisedTrainingError("optimizer must be the frozen torch Adam")
    if len(optimizer.param_groups) != 1:
        raise SupervisedTrainingError("frozen Adam must contain exactly one group")
    group = optimizer.param_groups[0]
    expected_group = _adam_group_contract()
    if set(group) != {"params", *expected_group} or any(
        group.get(key) != value for key, value in expected_group.items()
    ):
        raise SupervisedTrainingError("frozen Adam group semantics differ")
    expected_parameters = list(model.policy_mlp.parameters())
    actual_parameters = list(group["params"])
    if len(actual_parameters) != len(expected_parameters) or any(
        actual is not expected
        for actual, expected in zip(
            actual_parameters,
            expected_parameters,
            strict=True,
        )
    ):
        raise SupervisedTrainingError(
            "optimizer policy parameter order or scope differs"
        )
    optimized = {id(parameter) for parameter in actual_parameters}
    expected = {id(parameter) for parameter in expected_parameters}
    forbidden = {id(parameter) for parameter in model.value_mlp.parameters()}
    if optimized != expected or optimized & forbidden:
        raise SupervisedTrainingError("optimizer scope is not policy_mlp-only")
    if any(not parameter.requires_grad for parameter in model.policy_mlp.parameters()):
        raise SupervisedTrainingError("policy parameters must require gradients")
    if any(parameter.requires_grad for parameter in model.value_mlp.parameters()):
        raise SupervisedTrainingError("value parameters must be frozen")
    state_scope = {id(parameter) for parameter in optimizer.state}
    if not state_scope <= expected:
        raise SupervisedTrainingError("optimizer state contains non-policy parameters")


def _optimizer_update_count(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    _validate_optimizer_scope(model, optimizer)
    counts: list[int] = []
    for parameter in model.policy_mlp.parameters():
        state = optimizer.state.get(parameter)
        if not state:
            counts.append(0)
            continue
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise SupervisedTrainingError("frozen Adam parameter state keys differ")
        step_raw = state["step"]
        if not isinstance(step_raw, (torch.Tensor, int, float)):
            raise SupervisedTrainingError("frozen Adam step counter is invalid")
        if isinstance(step_raw, torch.Tensor) and (
            step_raw.numel() != 1 or not bool(torch.isfinite(step_raw).all())
        ):
            raise SupervisedTrainingError("frozen Adam step counter is invalid")
        step_value = step_raw.item() if isinstance(step_raw, torch.Tensor) else step_raw
        if (
            isinstance(step_value, bool)
            or not isinstance(step_value, (int, float))
            or not math.isfinite(float(step_value))
            or int(step_value) != float(step_value)
            or int(step_value) < 0
        ):
            raise SupervisedTrainingError("frozen Adam step counter is invalid")
        for name in ("exp_avg", "exp_avg_sq"):
            moment = state[name]
            if (
                not isinstance(moment, torch.Tensor)
                or moment.shape != parameter.shape
                or moment.dtype != parameter.dtype
                or moment.device != parameter.device
                or not bool(torch.isfinite(moment).all())
            ):
                raise SupervisedTrainingError("frozen Adam moment state is invalid")
        counts.append(int(step_value))
    if len(set(counts)) != 1:
        raise SupervisedTrainingError("frozen Adam parameter steps diverge")
    return counts[0]


def _validate_serialized_optimizer_state(
    model: torch.nn.Module,
    value: Any,
    *,
    expected_updates: int,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"state", "param_groups"}:
        raise SupervisedTrainingError("serialized Adam state keys differ")
    groups = value["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1:
        raise SupervisedTrainingError("serialized Adam group count differs")
    group = groups[0]
    expected_group = _adam_group_contract()
    if (
        not isinstance(group, Mapping)
        or set(group) != {"params", *expected_group}
        or any(group.get(key) != item for key, item in expected_group.items())
    ):
        raise SupervisedTrainingError("serialized Adam group semantics differ")
    parameters = list(model.policy_mlp.parameters())
    expected_indices = list(range(len(parameters)))
    if group.get("params") != expected_indices:
        raise SupervisedTrainingError("serialized Adam parameter order differs")
    states = value["state"]
    if not isinstance(states, Mapping):
        raise SupervisedTrainingError("serialized Adam parameter state differs")
    expected_state_indices = set() if expected_updates == 0 else set(expected_indices)
    if set(states) != expected_state_indices:
        raise SupervisedTrainingError("serialized Adam state scope differs")
    for index, parameter in enumerate(parameters):
        if expected_updates == 0:
            continue
        state = states[index]
        if not isinstance(state, Mapping) or set(state) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise SupervisedTrainingError("serialized Adam parameter keys differ")
        step = state["step"]
        if isinstance(step, torch.Tensor) and (
            step.numel() != 1 or not bool(torch.isfinite(step).all())
        ):
            raise SupervisedTrainingError("serialized Adam step counter differs")
        step_value = step.item() if isinstance(step, torch.Tensor) else step
        if (
            isinstance(step_value, bool)
            or not isinstance(step_value, (int, float))
            or not math.isfinite(float(step_value))
            or int(step_value) != expected_updates
            or float(step_value) != expected_updates
        ):
            raise SupervisedTrainingError("serialized Adam step counter differs")
        for name in ("exp_avg", "exp_avg_sq"):
            moment = state[name]
            if (
                not isinstance(moment, torch.Tensor)
                or moment.shape != parameter.shape
                or moment.dtype != parameter.dtype
                or not bool(torch.isfinite(moment).all())
            ):
                raise SupervisedTrainingError("serialized Adam moment state differs")


def _apply_optimizer_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[SupervisedExample],
    indices: Sequence[int],
    *,
    config: SupervisedLoopConfig,
    prior_update_count: int,
) -> SupervisedUpdateMetrics:
    if not indices:
        raise SupervisedTrainingError("optimizer batch is empty")
    optimizer.zero_grad(set_to_none=True)
    losses: list[torch.Tensor] = []
    for index in indices:
        example = examples[index]
        logits = model.policy_logits(example.features)
        losses.append(
            hard_a_pos_masked_cross_entropy(
                logits,
                example.a_pos_mask,
                teacher_index=example.teacher_index,
            )
        )
    loss = torch.stack(losses).mean()
    if not bool(torch.isfinite(loss)):
        raise SupervisedTrainingError("masked CE loss is non-finite")
    loss.backward()
    policy_parameters = tuple(model.policy_mlp.parameters())
    if any(
        parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
        for parameter in policy_parameters
    ):
        raise SupervisedTrainingError("policy gradients are missing or non-finite")
    if any(parameter.grad is not None for parameter in model.value_mlp.parameters()):
        raise SupervisedTrainingError("value parameters received a forbidden gradient")
    gradient_norm_raw = torch.nn.utils.clip_grad_norm_(
        policy_parameters,
        max_norm=float(config.grad_clip_norm),
    )
    gradient_norm = float(gradient_norm_raw)
    if not math.isfinite(gradient_norm):
        raise SupervisedTrainingError("policy gradient norm is non-finite")
    if any(
        not bool(torch.isfinite(parameter.grad).all())
        for parameter in policy_parameters
    ):
        raise SupervisedTrainingError("clipped policy gradients are non-finite")
    optimizer.step()
    if _optimizer_update_count(model, optimizer) != prior_update_count + 1:
        raise SupervisedTrainingError("Adam step counter did not advance exactly once")
    return SupervisedUpdateMetrics(
        update_count=prior_update_count + 1,
        batch_indices=tuple(int(index) for index in indices),
        loss=float(loss.detach().cpu()),
        gradient_norm=gradient_norm,
    )


def _run_supervised_updates(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[SupervisedExample],
    state: SupervisedState,
    *,
    permutation_generator: torch.Generator,
    config: SupervisedLoopConfig,
    max_updates: int | None = None,
    update_observer: Callable[[SupervisedUpdateMetrics], None] | None = None,
) -> SupervisedState:
    """Advance the deterministic loop; ``max_updates`` is a pause boundary."""
    if not examples:
        raise SupervisedTrainingError("supervised dataset is empty")
    if max_updates is not None and (
        isinstance(max_updates, bool)
        or not isinstance(max_updates, int)
        or max_updates <= 0
    ):
        raise SupervisedTrainingError("max_updates must be positive when supplied")
    if permutation_generator.device.type != "cpu":
        raise SupervisedTrainingError("permutation generator must be CPU-only")
    _validate_optimizer_scope(model, optimizer)
    sample_count = len(examples)
    batches_per_epoch = _validate_supervised_state(
        state,
        sample_count=sample_count,
        config=config,
    )
    _validate_permutation_generator(
        permutation_generator,
        state,
        sample_count=sample_count,
        config=config,
    )
    if _optimizer_update_count(model, optimizer) != state.update_count:
        raise SupervisedTrainingError(
            "cursor and Adam step disagree; repeated or missing update"
        )
    if state.completed:
        return state

    current = state
    performed = 0
    while current.epoch < config.epochs:
        if current.permutation is None:
            permutation = tuple(
                int(value)
                for value in torch.randperm(
                    sample_count, generator=permutation_generator
                ).tolist()
            )
            current = replace(current, permutation=permutation)
        permutation = current.permutation
        assert permutation is not None
        if len(permutation) != sample_count or set(permutation) != set(
            range(sample_count)
        ):
            raise SupervisedTrainingError("active epoch permutation differs")
        if current.batch_in_epoch >= batches_per_epoch:
            raise SupervisedTrainingError("batch cursor exceeds active permutation")
        start = current.sample_cursor
        stop = start + config.batch_size
        indices = permutation[start:stop]
        metrics = _apply_optimizer_batch(
            model,
            optimizer,
            examples,
            indices,
            config=config,
            prior_update_count=current.update_count,
        )

        next_batch = current.batch_in_epoch + 1
        next_epoch = current.epoch
        next_sample_cursor = stop
        next_permutation: tuple[int, ...] | None = permutation
        if next_batch == batches_per_epoch:
            next_epoch += 1
            next_batch = 0
            next_sample_cursor = 0
            next_permutation = None
        completed = next_epoch == config.epochs
        current = SupervisedState(
            epoch=next_epoch,
            batch_in_epoch=next_batch,
            update_count=current.update_count + 1,
            sample_cursor=next_sample_cursor,
            permutation=next_permutation,
            completed=completed,
        )
        _validate_supervised_state(
            current,
            sample_count=sample_count,
            config=config,
        )
        if update_observer is not None:
            update_observer(metrics)
        performed += 1
        if completed or (max_updates is not None and performed >= max_updates):
            _validate_permutation_generator(
                permutation_generator,
                current,
                sample_count=sample_count,
                config=config,
            )
            return current
    raise SupervisedTrainingError("supervised loop exited without a terminal cursor")


def run_supervised_updates(
    session: FreshSupervisedSession | ResumedSupervisedSession,
    state: SupervisedState,
    *,
    max_updates: int | None = None,
    update_observer: Callable[[SupervisedUpdateMetrics], None] | None = None,
) -> SupervisedState:
    """Advance only the corpus/plan/run bound into the opaque session."""
    context = _production_session_context(
        session,
        allowed_types=(FreshSupervisedSession, ResumedSupervisedSession),
        expected_purpose=SupervisedPurpose.SEED,
    )
    if state.completed:
        raise SupervisedTrainingError("completed supervised seeds cannot be advanced")
    _require_factory_lineage(
        context.model,
        context.optimizer,
        seed=context.seed,
        purpose=SupervisedPurpose.SEED,
        allowed_routes=(
            {"fresh", "resumed_seed"}
            if state.update_count == 0
            else {"seed", "resumed_seed"}
        ),
        require_initial_state=state.update_count == 0,
    )
    _validate_permutation_generator(
        context.permutation_generator,
        state,
        sample_count=PRODUCTION_TRAIN_EXAMPLES,
        config=context.config,
    )
    if (
        _batches_per_epoch(
            sample_count=len(context.examples),
            config=context.config,
        )
        != PRODUCTION_BATCHES_PER_EPOCH
    ):
        raise SupervisedTrainingError("production batches per epoch differ")
    result = _run_supervised_updates(
        context.model,
        context.optimizer,
        context.examples,
        state,
        permutation_generator=context.permutation_generator,
        config=context.config,
        max_updates=max_updates,
        update_observer=update_observer,
    )
    if result.completed and result.update_count != PRODUCTION_UPDATES_PER_SEED:
        raise SupervisedTrainingError("production update total differs from 2,240")
    _MODEL_ROUTE[context.model] = "seed"
    return result


def _run_supervised_smoke_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[SupervisedExample],
    *,
    seed: int,
) -> SupervisedSmokeResult:
    """Perform one disposable real backward/Adam update, never an RL smoke."""
    config = production_loop_config(seed)
    _validate_optimizer_scope(model, optimizer)
    if len(examples) != PRODUCTION_BATCH_SIZE:
        raise SupervisedTrainingError(
            "supervised smoke requires one complete 128-example batch"
        )
    if _optimizer_update_count(model, optimizer) != 0:
        raise SupervisedTrainingError(
            "supervised smoke must start from fresh Adam state"
        )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.policy_mlp.named_parameters()
    }
    metrics = _apply_optimizer_batch(
        model,
        optimizer,
        examples,
        tuple(range(PRODUCTION_BATCH_SIZE)),
        config=config,
        prior_update_count=0,
    )
    changed = any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.policy_mlp.named_parameters()
    )
    if not changed:
        raise SupervisedTrainingError("supervised smoke changed no policy parameter")
    return SupervisedSmokeResult(
        training_semantics=(
            "offline-supervised-hard-a-pos-masked-one-hot-cross-entropy"
        ),
        optimizer_updates=1,
        finite_loss=math.isfinite(metrics.loss),
        finite_gradient_norm=math.isfinite(metrics.gradient_norm),
        parameters_changed=True,
        reinforcement_learning=False,
        loss=metrics.loss,
        gradient_norm=metrics.gradient_norm,
    )


def run_supervised_smoke_update(
    session: SupervisedSmokeSession,
) -> SupervisedSmokeResult:
    """Run one disposable update from a fresh, fully bound session."""
    context = _production_session_context(
        session,
        allowed_types=(SupervisedSmokeSession,),
        expected_purpose=SupervisedPurpose.SMOKE,
    )
    if session in _SMOKE_COMPLETIONS:
        raise SupervisedTrainingError("supervised smoke was already run")
    _require_factory_lineage(
        context.model,
        context.optimizer,
        seed=context.seed,
        purpose=SupervisedPurpose.SMOKE,
        allowed_routes={"fresh"},
        require_initial_state=True,
    )
    fresh_state = SupervisedState(
        epoch=0,
        batch_in_epoch=0,
        update_count=0,
        sample_cursor=0,
        permutation=None,
        completed=False,
    )
    _validate_permutation_generator(
        context.permutation_generator,
        fresh_state,
        sample_count=PRODUCTION_TRAIN_EXAMPLES,
        config=context.config,
    )
    generator_before = canonical_training_state_identity(
        context.permutation_generator.get_state()
    )
    rng_before = canonical_training_state_identity(
        capture_rng_state(
            {"permutation_generator": context.permutation_generator.get_state()}
        )
    )
    initial_model_identity = _model_state_identity(context.model)
    value_identity = canonical_training_state_identity(
        context.model.value_mlp.state_dict()
    )
    smoke_examples = context.examples[:PRODUCTION_BATCH_SIZE]
    batch_identity = _smoke_batch_identity(smoke_examples)
    try:
        result = _run_supervised_smoke_update(
            context.model,
            context.optimizer,
            smoke_examples,
            seed=context.seed,
        )
    except Exception:
        _MODEL_ROUTE[context.model] = "smoke_failed"
        raise
    generator_after = canonical_training_state_identity(
        context.permutation_generator.get_state()
    )
    rng_after = canonical_training_state_identity(
        capture_rng_state(
            {"permutation_generator": context.permutation_generator.get_state()}
        )
    )
    _validate_permutation_generator(
        context.permutation_generator,
        fresh_state,
        sample_count=PRODUCTION_TRAIN_EXAMPLES,
        config=context.config,
    )
    if (
        generator_after != generator_before
        or rng_after != rng_before
        or _optimizer_update_count(context.model, context.optimizer) != 1
        or not result.finite_loss
        or not result.finite_gradient_norm
        or not result.parameters_changed
        or result.reinforcement_learning
        or _model_state_identity(context.model) == initial_model_identity
        or canonical_training_state_identity(context.model.value_mlp.state_dict())
        != value_identity
    ):
        _MODEL_ROUTE[context.model] = "smoke_failed"
        raise SupervisedTrainingError("supervised smoke evidence differs")
    cursor = SupervisedSmokeCursor(
        schema="nmm.supervised-smoke-cursor.v1",
        terminal=True,
        resumable=False,
        update_count=1,
        optimizer_updates=1,
        batch_start=0,
        batch_stop=PRODUCTION_BATCH_SIZE,
        permutation_calls=0,
        rng_before_identity=rng_before,
        rng_after_identity=rng_after,
    )
    _MODEL_ROUTE[context.model] = "smoke"
    _SMOKE_COMPLETIONS[session] = _SmokeCompletionContext(
        cursor=cursor,
        result=result,
        batch_identity=batch_identity,
        initial_model_state_identity=initial_model_identity,
        completed_model_state_identity=_model_state_identity(context.model),
        completed_optimizer_state_identity=canonical_training_state_identity(
            context.optimizer.state_dict()
        ),
        value_state_identity=value_identity,
        permutation_generator_identity=generator_after,
    )
    return result


class _TorchGeneratorAdapter:
    def __init__(self, generator: torch.Generator) -> None:
        self.generator = generator

    def setstate(self, state: Any) -> None:
        if not isinstance(state, torch.Tensor):
            raise CheckpointCompatibilityError(
                "permutation generator state is not a tensor"
            )
        self.generator.set_state(state.detach().cpu())


def _canonical_state_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "kind": "torch.Tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "numpy.ndarray",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, Mapping):
        entries = [
            {
                "key_type": type(key).__name__,
                "key": str(key),
                "value": _canonical_state_value(item),
            }
            for key, item in value.items()
        ]
        entries.sort(key=lambda item: (item["key_type"], item["key"]))
        return {"kind": "mapping", "entries": entries}
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_canonical_state_value(item) for item in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_canonical_state_value(item) for item in value],
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return {"kind": "float", "hex": float(value).hex()}
    raise SupervisedTrainingError(
        f"unsupported canonical training state value: {type(value).__name__}"
    )


def canonical_training_state_identity(value: Any) -> str:
    """Hash semantic tensor/state content independent of pickle bytes."""
    return canonical_sha256(_canonical_state_value(value))


def _model_config(model: torch.nn.Module) -> dict[str, Any]:
    if type(model) is not ScaffoldedPolicyNet:
        raise SupervisedTrainingError(
            "student must be the frozen ScaffoldedPolicyNet implementation"
        )
    get_config = getattr(model, "get_config", None)
    if not callable(get_config):
        raise SupervisedTrainingError("student model cannot report its config")
    config = dict(get_config())
    if (
        set(config)
        != {
            "move_feat_dim",
            "value_input_dim",
            "policy_hidden",
            "value_hidden",
            "dropout",
        }
        or config.get("move_feat_dim") != 62
        or config.get("value_input_dim") != VALUE_INPUT_DIM
        or tuple(config.get("policy_hidden", ())) != PRODUCTION_POLICY_HIDDEN
        or tuple(config.get("value_hidden", ())) != ()
        or config.get("dropout") != 0.0
    ):
        raise SupervisedTrainingError(
            "student model must be base62/policy(128,64)/value-empty/dropout0"
        )
    return config


def _required_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SupervisedTrainingError(f"{field} must be a lowercase SHA-256")
    return value


def _internal_test_lineage(
    model: ScaffoldedPolicyNet,
    *,
    seed: int,
) -> FreshSupervisedLineage:
    return FreshSupervisedLineage(
        lineage_kind="internal_test_profile",
        seed=seed,
        expected_initial_model_state_identity=_model_state_identity(model),
        start_mode="internal-test",
        purpose=SupervisedPurpose.SEED,
    )


def _checkpoint_lineage(
    implementation: Mapping[str, str],
) -> FreshSupervisedLineage:
    expected_fields = {
        "trainer",
        "optimizer",
        "seed",
        "training_kind",
        "lineage_kind",
        "start_mode",
        "expected_initial_model_state_identity",
        "purpose",
    }
    if set(implementation) != expected_fields:
        raise SupervisedTrainingError("checkpoint lineage fields differ")
    seed_text = implementation["seed"]
    try:
        seed = int(seed_text)
    except (TypeError, ValueError) as exc:
        raise SupervisedTrainingError("checkpoint lineage seed is invalid") from exc
    if seed <= 0 or str(seed) != seed_text:
        raise SupervisedTrainingError("checkpoint lineage seed is invalid")
    try:
        purpose = SupervisedPurpose(implementation["purpose"])
    except (TypeError, ValueError) as exc:
        raise SupervisedTrainingError("checkpoint lineage purpose differs") from exc
    lineage = FreshSupervisedLineage(
        lineage_kind=implementation["lineage_kind"],
        seed=seed,
        expected_initial_model_state_identity=_required_sha256(
            implementation["expected_initial_model_state_identity"],
            field="expected initial model state identity",
        ),
        start_mode=implementation["start_mode"],
        purpose=purpose,
    )
    if lineage.purpose is not SupervisedPurpose.SEED or (
        lineage.lineage_kind,
        lineage.start_mode,
    ) not in {
        ("fresh_supervised_seed", "fresh"),
        ("internal_test_profile", "internal-test"),
    }:
        raise SupervisedTrainingError("checkpoint lineage kind/start mode differs")
    return lineage


def _internal_payload_identities(
    *,
    corpus_identity: str,
    split_identity: str,
    sample_count: int,
    config: SupervisedLoopConfig,
) -> tuple[str, str]:
    return (
        canonical_sha256(
            {
                "scope": "internal-supervised-record-order",
                "corpus": corpus_identity,
                "split": split_identity,
                "sample_count": sample_count,
            }
        ),
        canonical_sha256(
            {
                "scope": "internal-supervised-encoded-payload",
                "corpus": corpus_identity,
                "split": split_identity,
                "sample_count": sample_count,
                "config": config.to_dict(),
            }
        ),
    )


def _save_supervised_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: SupervisedState,
    permutation_generator: torch.Generator,
    config: SupervisedLoopConfig,
    sample_count: int,
    corpus_identity: str,
    split_identity: str,
    plan_identity: str,
    seed: int,
    run_id: str,
    experiment_id: str,
    role: str,
    created_at_utc: str,
    lineage: FreshSupervisedLineage | None = None,
    training_records_identity: str | None = None,
    encoded_payload_identity: str | None = None,
) -> Path:
    _validate_optimizer_scope(model, optimizer)
    if permutation_generator.device.type != "cpu":
        raise SupervisedTrainingError("permutation generator must be CPU-only")
    _validate_supervised_state(
        state,
        sample_count=sample_count,
        config=config,
    )
    _validate_permutation_generator(
        permutation_generator,
        state,
        sample_count=sample_count,
        config=config,
    )
    if _optimizer_update_count(model, optimizer) != state.update_count:
        raise SupervisedTrainingError("checkpoint cursor and Adam step differ")
    if seed != config.permutation_seed:
        raise SupervisedTrainingError("checkpoint seed and permutation seed differ")
    resolved_lineage = lineage or _internal_test_lineage(model, seed=seed)
    if resolved_lineage.seed != seed:
        raise SupervisedTrainingError("checkpoint fresh lineage seed differs")
    initial_model_identity = _required_sha256(
        resolved_lineage.expected_initial_model_state_identity,
        field="expected initial model state identity",
    )
    if resolved_lineage.purpose is not SupervisedPurpose.SEED:
        raise SupervisedTrainingError("seed checkpoint lineage purpose differs")
    if resolved_lineage.lineage_kind == "fresh_supervised_seed" and (
        resolved_lineage.start_mode != "fresh"
        or (
            state.update_count == 0
            and _model_state_identity(model) != initial_model_identity
        )
    ):
        raise SupervisedTrainingError("checkpoint fresh model lineage differs")
    corpus_hash = _required_sha256(corpus_identity, field="corpus_identity")
    split_hash = _required_sha256(split_identity, field="split_identity")
    plan_hash = _required_sha256(plan_identity, field="plan_identity")
    default_records, default_encoded = _internal_payload_identities(
        corpus_identity=corpus_hash,
        split_identity=split_hash,
        sample_count=sample_count,
        config=config,
    )
    training_records_hash = _required_sha256(
        training_records_identity or default_records,
        field="training_records_identity",
    )
    encoded_payload_hash = _required_sha256(
        encoded_payload_identity or default_encoded,
        field="encoded_payload_identity",
    )
    config_record = config.to_dict()
    config_identity = canonical_sha256(config_record)
    model_config = _model_config(model)
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    _validate_serialized_optimizer_state(
        model,
        optimizer_state,
        expected_updates=state.update_count,
    )
    rng_state = capture_rng_state(
        {"permutation_generator": permutation_generator.get_state()}
    )
    state_identities = {
        "model": canonical_training_state_identity(model_state),
        "optimizer": canonical_training_state_identity(optimizer_state),
        "rng": canonical_training_state_identity(rng_state),
        "cursor": canonical_training_state_identity(state.to_dict()),
    }
    payload = CheckpointPayload(
        model_state=model_state,
        optimizer_state=optimizer_state,
        scheduler_state=None,
        scaler_state=None,
        rng_state=rng_state,
        trainer_state={
            "game_count": 0,
            "batch_count": state.update_count,
            "update_count": state.update_count,
            "difficulty": 9,
            "temperature": 1.0,
            "rolling_metrics": {},
            "curriculum": {
                "training_semantics": config_record["training_semantics"],
                "purpose": SupervisedPurpose.SEED.value,
                "epoch": state.epoch,
                "batch_in_epoch": state.batch_in_epoch,
            },
            "target_network": {"enabled": False},
            "recovery_state": {
                "exact_resume": True,
                "cursor": state.to_dict(),
            },
            "model_config": model_config,
        },
        data_state={
            "cursor": state.to_dict(),
            "consumed_snapshots": [corpus_hash],
            "cache": {"canonical_state_identities": state_identities},
            "buckets": {
                "corpus_identity": corpus_hash,
                "split_identity": split_hash,
                "plan_identity": plan_hash,
                "config_identity": config_identity,
                "seed": seed,
                "sample_count": sample_count,
                "training_records_identity": training_records_hash,
                "encoded_payload_identity": encoded_payload_hash,
                "fresh_lineage": resolved_lineage.to_dict(),
                "purpose": SupervisedPurpose.SEED.value,
            },
            "mutable_assets": {},
        },
    )
    descriptor = CheckpointDescriptor(
        checkpoint_id=f"{run_id}:update:{state.update_count}",
        run_id=run_id,
        experiment_id=experiment_id,
        parent_checkpoint_id=None,
        role=role,
        save_reason="supervised-complete" if state.completed else "supervised-pause",
        created_at_utc=created_at_utc,
        config_sha256=config_identity,
        feature_schema_version=FEATURE_SCHEMA,
        label_schema_version=LABEL_SCHEMA,
        database_schema_versions={"corpus": CORPUS_SCHEMA},
        asset_identities={
            "corpus": corpus_hash,
            "split": split_hash,
            "plan": plan_hash,
            "training_records": training_records_hash,
            "encoded_payload": encoded_payload_hash,
        },
        implementation={
            "trainer": TRAINER_ID,
            "optimizer": "Adam-policy_mlp-only",
            "seed": str(seed),
            "training_kind": "offline-supervised-not-rl",
            "lineage_kind": resolved_lineage.lineage_kind,
            "start_mode": resolved_lineage.start_mode,
            "expected_initial_model_state_identity": initial_model_identity,
            "purpose": SupervisedPurpose.SEED.value,
        },
    )
    target = Path(path)
    save_checkpoint(target, descriptor, payload, previous_copies=0)
    return target


def _load_supervised_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    permutation_generator: torch.Generator,
    config: SupervisedLoopConfig,
    sample_count: int,
    corpus_identity: str,
    split_identity: str,
    plan_identity: str,
    expected_seed: int,
    expected_run_id: str,
    expected_experiment_id: str,
    expected_role: str | None,
    expected_lineage: FreshSupervisedLineage | None = None,
    training_records_identity: str | None = None,
    encoded_payload_identity: str | None = None,
) -> SupervisedState:
    _validate_optimizer_scope(model, optimizer)
    if permutation_generator.device.type != "cpu":
        raise SupervisedTrainingError("permutation generator must be CPU-only")
    if expected_seed != config.permutation_seed:
        raise SupervisedTrainingError("resume seed and permutation seed differ")
    corpus_hash = _required_sha256(corpus_identity, field="corpus_identity")
    split_hash = _required_sha256(split_identity, field="split_identity")
    plan_hash = _required_sha256(plan_identity, field="plan_identity")
    default_records, default_encoded = _internal_payload_identities(
        corpus_identity=corpus_hash,
        split_identity=split_hash,
        sample_count=sample_count,
        config=config,
    )
    training_records_hash = _required_sha256(
        training_records_identity or default_records,
        field="training_records_identity",
    )
    encoded_payload_hash = _required_sha256(
        encoded_payload_identity or default_encoded,
        field="encoded_payload_identity",
    )
    config_identity = canonical_sha256(config.to_dict())
    envelope = load_checkpoint(path, map_location="cpu")
    descriptor = envelope.descriptor
    if expected_role is not None and descriptor.role != expected_role:
        raise SupervisedTrainingError("supervised checkpoint role differs")
    implementation = dict(descriptor.implementation)
    observed_lineage = _checkpoint_lineage(implementation)
    if (
        descriptor.config_sha256 != config_identity
        or descriptor.feature_schema_version != FEATURE_SCHEMA
        or descriptor.label_schema_version != LABEL_SCHEMA
        or dict(descriptor.database_schema_versions) != {"corpus": CORPUS_SCHEMA}
        or dict(descriptor.asset_identities)
        != {
            "corpus": corpus_hash,
            "split": split_hash,
            "plan": plan_hash,
            "training_records": training_records_hash,
            "encoded_payload": encoded_payload_hash,
        }
        or implementation["trainer"] != TRAINER_ID
        or implementation["optimizer"] != "Adam-policy_mlp-only"
        or implementation["seed"] != str(expected_seed)
        or implementation["training_kind"] != "offline-supervised-not-rl"
        or descriptor.run_id != expected_run_id
        or descriptor.experiment_id != expected_experiment_id
    ):
        raise SupervisedTrainingError("supervised checkpoint identity differs")
    if expected_lineage is None:
        if observed_lineage.lineage_kind != "internal_test_profile":
            raise SupervisedTrainingError(
                "private checkpoint load lacks its internal test lineage"
            )
    elif observed_lineage != expected_lineage:
        raise SupervisedTrainingError("supervised checkpoint fresh lineage differs")
    trainer_state = envelope.payload.trainer_state
    data_state = envelope.payload.data_state
    try:
        state = SupervisedState.from_dict(data_state["cursor"])
    except (KeyError, TypeError) as exc:
        raise SupervisedTrainingError(
            "supervised checkpoint cursor is missing"
        ) from exc
    _validate_supervised_state(
        state,
        sample_count=sample_count,
        config=config,
    )
    derived_role = SEED_COMPLETE_ROLE if state.completed else SEED_LATEST_ROLE
    accepted_role = expected_role if expected_role is not None else derived_role
    if descriptor.role != accepted_role:
        raise SupervisedTrainingError("supervised checkpoint role differs")
    if expected_lineage is not None and accepted_role != derived_role:
        raise SupervisedTrainingError("production checkpoint role was not derived")
    expected_buckets = {
        "corpus_identity": corpus_hash,
        "split_identity": split_hash,
        "plan_identity": plan_hash,
        "config_identity": config_identity,
        "seed": expected_seed,
        "sample_count": sample_count,
        "training_records_identity": training_records_hash,
        "encoded_payload_identity": encoded_payload_hash,
        "fresh_lineage": observed_lineage.to_dict(),
        "purpose": SupervisedPurpose.SEED.value,
    }
    expected_curriculum = {
        "training_semantics": config.to_dict()["training_semantics"],
        "purpose": SupervisedPurpose.SEED.value,
        "epoch": state.epoch,
        "batch_in_epoch": state.batch_in_epoch,
    }
    if (
        trainer_state["game_count"] != 0
        or trainer_state["batch_count"] != state.update_count
        or trainer_state["update_count"] != state.update_count
        or trainer_state["difficulty"] != 9
        or trainer_state["temperature"] != 1.0
        or trainer_state["rolling_metrics"] != {}
        or trainer_state["curriculum"] != expected_curriculum
        or trainer_state["target_network"] != {"enabled": False}
        or trainer_state["recovery_state"]
        != {"exact_resume": True, "cursor": state.to_dict()}
        or data_state["consumed_snapshots"] != [corpus_hash]
        or data_state["buckets"] != expected_buckets
        or data_state["mutable_assets"] != {}
        or trainer_state["model_config"] != _model_config(model)
        or descriptor.checkpoint_id != f"{expected_run_id}:update:{state.update_count}"
        or descriptor.save_reason
        != ("supervised-complete" if state.completed else "supervised-pause")
    ):
        raise SupervisedTrainingError("supervised checkpoint cursor differs")
    if (
        envelope.payload.optimizer_state is None
        or envelope.payload.scheduler_state is not None
        or envelope.payload.scaler_state is not None
    ):
        raise SupervisedTrainingError("supervised checkpoint lacks optimizer state")
    _validate_serialized_optimizer_state(
        model,
        envelope.payload.optimizer_state,
        expected_updates=state.update_count,
    )
    cache = data_state.get("cache")
    if not isinstance(cache, Mapping) or set(cache) != {"canonical_state_identities"}:
        raise SupervisedTrainingError(
            "checkpoint canonical state identities are missing"
        )
    expected_state_identities = cache["canonical_state_identities"]
    if not isinstance(expected_state_identities, Mapping) or set(
        expected_state_identities
    ) != {"model", "optimizer", "rng", "cursor"}:
        raise SupervisedTrainingError("checkpoint canonical state identity keys differ")
    observed_state_identities = {
        "model": canonical_training_state_identity(envelope.payload.model_state),
        "optimizer": canonical_training_state_identity(
            envelope.payload.optimizer_state
        ),
        "rng": canonical_training_state_identity(envelope.payload.rng_state),
        "cursor": canonical_training_state_identity(state.to_dict()),
    }
    if dict(expected_state_identities) != observed_state_identities:
        raise SupervisedTrainingError("checkpoint canonical state identity differs")
    _validate_checkpoint_permutation_state(
        envelope.payload.rng_state,
        state,
        sample_count=sample_count,
        config=config,
    )
    model.load_state_dict(envelope.payload.model_state, strict=True)
    optimizer.load_state_dict(envelope.payload.optimizer_state)
    _validate_optimizer_scope(model, optimizer)
    if _optimizer_update_count(model, optimizer) != state.update_count:
        raise SupervisedTrainingError("resumed Adam step and cursor disagree")
    restore_rng_state(
        envelope.payload.rng_state,
        component_rngs={
            "permutation_generator": _TorchGeneratorAdapter(permutation_generator)
        },
    )
    _validate_permutation_generator(
        permutation_generator,
        state,
        sample_count=sample_count,
        config=config,
    )
    return state


def _smoke_result_dict(result: SupervisedSmokeResult) -> dict[str, Any]:
    return {
        "training_semantics": result.training_semantics,
        "optimizer_updates": result.optimizer_updates,
        "finite_loss": result.finite_loss,
        "finite_gradient_norm": result.finite_gradient_norm,
        "parameters_changed": result.parameters_changed,
        "reinforcement_learning": result.reinforcement_learning,
        "loss": result.loss,
        "gradient_norm": result.gradient_norm,
    }


def save_supervised_smoke_checkpoint(
    path: str | Path,
    *,
    session: SupervisedSmokeSession,
    created_at_utc: str,
) -> Path:
    """Persist one terminal non-resumable supervised smoke proof."""
    context = _production_session_context(
        session,
        allowed_types=(SupervisedSmokeSession,),
        expected_purpose=SupervisedPurpose.SMOKE,
    )
    if session in _SAVED_SMOKE_SESSIONS:
        raise SupervisedTrainingError("supervised smoke evidence was already saved")
    completion = _SMOKE_COMPLETIONS.get(session)
    if completion is None:
        raise SupervisedTrainingError(
            "supervised smoke was not run for exactly one update"
        )
    lineage = _require_factory_lineage(
        context.model,
        context.optimizer,
        seed=context.seed,
        purpose=SupervisedPurpose.SMOKE,
        allowed_routes={"smoke"},
        require_initial_state=False,
    )
    cursor = completion.cursor
    current_generator_identity = canonical_training_state_identity(
        context.permutation_generator.get_state()
    )
    if (
        lineage.lineage_kind != "fresh_supervised_smoke"
        or lineage.purpose is not SupervisedPurpose.SMOKE
        or cursor.schema != "nmm.supervised-smoke-cursor.v1"
        or not cursor.terminal
        or cursor.resumable
        or cursor.update_count != 1
        or cursor.optimizer_updates != 1
        or (cursor.batch_start, cursor.batch_stop)
        != (
            0,
            PRODUCTION_BATCH_SIZE,
        )
        or cursor.permutation_calls != 0
        or cursor.rng_before_identity != cursor.rng_after_identity
        or completion.permutation_generator_identity != current_generator_identity
        or _optimizer_update_count(context.model, context.optimizer) != 1
        or _model_state_identity(context.model)
        != completion.completed_model_state_identity
        or completion.initial_model_state_identity
        != lineage.expected_initial_model_state_identity
        or completion.completed_model_state_identity
        == completion.initial_model_state_identity
        or canonical_training_state_identity(context.optimizer.state_dict())
        != completion.completed_optimizer_state_identity
        or canonical_training_state_identity(context.model.value_mlp.state_dict())
        != completion.value_state_identity
        or _smoke_batch_identity(context.examples[:PRODUCTION_BATCH_SIZE])
        != completion.batch_identity
        or not completion.result.finite_loss
        or not completion.result.finite_gradient_norm
        or not completion.result.parameters_changed
        or completion.result.reinforcement_learning
        or completion.result.optimizer_updates != 1
    ):
        raise SupervisedTrainingError("supervised smoke completion evidence differs")
    model_state = context.model.state_dict()
    optimizer_state = context.optimizer.state_dict()
    _validate_serialized_optimizer_state(
        context.model,
        optimizer_state,
        expected_updates=1,
    )
    rng_state = capture_rng_state(
        {"permutation_generator": context.permutation_generator.get_state()}
    )
    cursor_record = cursor.to_dict()
    config_record = context.config.to_dict()
    config_identity = canonical_sha256(config_record)
    smoke_result = _smoke_result_dict(completion.result)
    canonical_identities = {
        "model": canonical_training_state_identity(model_state),
        "optimizer": canonical_training_state_identity(optimizer_state),
        "rng": canonical_training_state_identity(rng_state),
        "cursor": canonical_training_state_identity(cursor_record),
        "batch": completion.batch_identity,
    }
    payload = CheckpointPayload(
        model_state=model_state,
        optimizer_state=optimizer_state,
        scheduler_state=None,
        scaler_state=None,
        rng_state=rng_state,
        trainer_state={
            "game_count": 0,
            "batch_count": 1,
            "update_count": 1,
            "difficulty": 9,
            "temperature": 1.0,
            "rolling_metrics": smoke_result,
            "curriculum": {
                "training_semantics": completion.result.training_semantics,
                "purpose": SupervisedPurpose.SMOKE.value,
                "disposable": True,
            },
            "target_network": {"enabled": False},
            "recovery_state": {
                "exact_resume": False,
                "resumable": False,
                "terminal": True,
            },
            "model_config": _model_config(context.model),
        },
        data_state={
            "cursor": cursor_record,
            "consumed_snapshots": [context.corpus_identity],
            "cache": {
                "canonical_state_identities": canonical_identities,
                "smoke_result": smoke_result,
            },
            "buckets": {
                "corpus_identity": context.corpus_identity,
                "split_identity": context.split_identity,
                "plan_identity": context.plan_identity,
                "config_identity": config_identity,
                "seed": context.seed,
                "sample_count": PRODUCTION_TRAIN_EXAMPLES,
                "batch_identity": completion.batch_identity,
                "training_records_identity": context.training_records_identity,
                "encoded_payload_identity": context.encoded_payload_identity,
                "fresh_lineage": lineage.to_dict(),
                "purpose": SupervisedPurpose.SMOKE.value,
            },
            "mutable_assets": {},
        },
    )
    descriptor = CheckpointDescriptor(
        checkpoint_id=f"{context.run_id}:smoke:update:1",
        run_id=context.run_id,
        experiment_id=context.experiment_id,
        parent_checkpoint_id=None,
        role=SMOKE_DISPOSABLE_ROLE,
        save_reason="supervised-smoke-disposable",
        created_at_utc=created_at_utc,
        config_sha256=config_identity,
        feature_schema_version=FEATURE_SCHEMA,
        label_schema_version=LABEL_SCHEMA,
        database_schema_versions={"corpus": CORPUS_SCHEMA},
        asset_identities={
            "corpus": context.corpus_identity,
            "split": context.split_identity,
            "plan": context.plan_identity,
            "training_records": context.training_records_identity,
            "encoded_payload": context.encoded_payload_identity,
            "smoke_batch": completion.batch_identity,
        },
        implementation={
            "trainer": TRAINER_ID,
            "optimizer": "Adam-policy_mlp-only",
            "seed": str(context.seed),
            "training_kind": "offline-supervised-smoke-not-rl",
            "lineage_kind": lineage.lineage_kind,
            "start_mode": lineage.start_mode,
            "expected_initial_model_state_identity": (
                lineage.expected_initial_model_state_identity
            ),
            "purpose": SupervisedPurpose.SMOKE.value,
        },
    )
    target = Path(path)
    save_checkpoint(target, descriptor, payload, previous_copies=0)
    _SAVED_SMOKE_SESSIONS.add(session)
    return target


def save_supervised_checkpoint(
    path: str | Path,
    *,
    session: FreshSupervisedSession | ResumedSupervisedSession,
    state: SupervisedState,
    created_at_utc: str,
) -> Path:
    """Save without accepting any caller-supplied identity relabeling."""
    context = _production_session_context(
        session,
        allowed_types=(FreshSupervisedSession, ResumedSupervisedSession),
        expected_purpose=SupervisedPurpose.SEED,
    )
    lineage = _require_factory_lineage(
        context.model,
        context.optimizer,
        seed=context.seed,
        purpose=SupervisedPurpose.SEED,
        allowed_routes=(
            {"fresh", "resumed_seed"}
            if state.update_count == 0
            else {"seed", "resumed_seed"}
        ),
        require_initial_state=state.update_count == 0,
    )
    role = SEED_COMPLETE_ROLE if state.completed else SEED_LATEST_ROLE
    return _save_supervised_checkpoint(
        path,
        model=context.model,
        optimizer=context.optimizer,
        state=state,
        permutation_generator=context.permutation_generator,
        config=context.config,
        sample_count=PRODUCTION_TRAIN_EXAMPLES,
        corpus_identity=context.corpus_identity,
        split_identity=context.split_identity,
        plan_identity=context.plan_identity,
        seed=context.seed,
        run_id=context.run_id,
        experiment_id=context.experiment_id,
        role=role,
        created_at_utc=created_at_utc,
        lineage=lineage,
        training_records_identity=context.training_records_identity,
        encoded_payload_identity=context.encoded_payload_identity,
    )


def load_supervised_checkpoint(
    path: str | Path,
    *,
    session: FreshSupervisedSession,
) -> tuple[ResumedSupervisedSession, SupervisedState]:
    """Consume one fresh binder and return an exact resumed capability."""
    context = _production_session_context(
        session,
        allowed_types=(FreshSupervisedSession,),
        expected_purpose=SupervisedPurpose.SEED,
    )
    lineage = _require_factory_lineage(
        context.model,
        context.optimizer,
        seed=context.seed,
        purpose=SupervisedPurpose.SEED,
        allowed_routes={"fresh"},
        require_initial_state=True,
    )
    _validate_optimizer_scope(context.model, context.optimizer)
    if _optimizer_update_count(context.model, context.optimizer) != 0:
        raise SupervisedTrainingError(
            "production resume target optimizer is not factory-fresh"
        )
    fresh_state = SupervisedState(
        epoch=0,
        batch_in_epoch=0,
        update_count=0,
        sample_cursor=0,
        permutation=None,
        completed=False,
    )
    _validate_permutation_generator(
        context.permutation_generator,
        fresh_state,
        sample_count=PRODUCTION_TRAIN_EXAMPLES,
        config=context.config,
    )
    state = _load_supervised_checkpoint(
        path,
        model=context.model,
        optimizer=context.optimizer,
        permutation_generator=context.permutation_generator,
        config=context.config,
        sample_count=PRODUCTION_TRAIN_EXAMPLES,
        corpus_identity=context.corpus_identity,
        split_identity=context.split_identity,
        plan_identity=context.plan_identity,
        expected_seed=context.seed,
        expected_run_id=context.run_id,
        expected_experiment_id=context.experiment_id,
        expected_role=SEED_LATEST_ROLE,
        expected_lineage=lineage,
        training_records_identity=context.training_records_identity,
        encoded_payload_identity=context.encoded_payload_identity,
    )
    if state.completed or state.update_count >= PRODUCTION_UPDATES_PER_SEED:
        raise SupervisedTrainingError(
            "production resume accepts only incomplete latest seed checkpoints"
        )
    _MODEL_ROUTE[context.model] = "resumed_seed"
    resumed = _activate_session(
        ResumedSupervisedSession,
        context,
        replaced_session=session,
    )
    assert isinstance(resumed, ResumedSupervisedSession)
    return resumed, state
