from __future__ import annotations

import copy
import hashlib
import inspect
import pickle
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import learned_ai.agents.classical_a_pos_student as student_module
import learned_ai.training.classical_a_pos_supervised as supervised_module
from learned_ai.training.checkpoint_envelope import (
    CheckpointFormatError,
    load_checkpoint,
    save_checkpoint,
)
from learned_ai.agents.classical_a_pos_student import (
    QualifiedCompleteSeedBinding,
    QualifiedStudentHandle,
    StudentQualificationError,
    _issue_test_qualified_complete_seed_binding,
    load_qualified_complete_student,
)
from learned_ai.training.classical_a_pos_supervised import (
    SEED_COMPLETE_ROLE,
    SupervisedPurpose,
    SupervisedState,
    _create_fresh_production_student,
    _save_supervised_checkpoint,
    canonical_training_state_identity,
    create_fresh_supervised_state,
    production_loop_config,
)


CORPUS_IDENTITY = "a" * 64
SPLIT_IDENTITY = "b" * 64
PLAN_IDENTITY = "c" * 64
RECORDS_IDENTITY = "d" * 64
ENCODED_IDENTITY = "e" * 64
SEED = 2026083001
RUN_ID = "qualified-complete-seed-2026083001"
EXPERIMENT_ID = "classical-a-pos-supervised"


def _write_complete_checkpoint(path: Path) -> Path:
    model, optimizer = _create_fresh_production_student(
        SEED,
        purpose=SupervisedPurpose.SEED,
    )
    for parameter in model.policy_mlp.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(2240.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    config = production_loop_config(SEED)
    generator = torch.Generator(device="cpu")
    create_fresh_supervised_state(
        permutation_generator=generator,
        config=config,
    )
    for _epoch in range(20):
        torch.randperm(14_336, generator=generator)
    state = SupervisedState(
        epoch=20,
        batch_in_epoch=0,
        update_count=2_240,
        sample_cursor=0,
        permutation=None,
        completed=True,
    )
    return _save_supervised_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        state=state,
        permutation_generator=generator,
        config=config,
        sample_count=14_336,
        corpus_identity=CORPUS_IDENTITY,
        split_identity=SPLIT_IDENTITY,
        plan_identity=PLAN_IDENTITY,
        seed=SEED,
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        role=SEED_COMPLETE_ROLE,
        created_at_utc="2026-08-31T00:00:00Z",
        lineage=supervised_module._FRESH_MODEL_LINEAGES[model],
        training_records_identity=RECORDS_IDENTITY,
        encoded_payload_identity=ENCODED_IDENTITY,
    )


@pytest.fixture
def complete_checkpoint(tmp_path: Path) -> Path:
    return _write_complete_checkpoint(tmp_path / "complete.pt")


def _resign(
    source: Path,
    target: Path,
    *,
    descriptor=None,
    payload=None,
) -> Path:
    envelope = load_checkpoint(source)
    save_checkpoint(
        target,
        descriptor or envelope.descriptor,
        payload or envelope.payload,
        previous_copies=0,
    )
    return target


def _qualified_load(path: Path, *, overrides: dict | None = None):
    binding = _issue_test_qualified_complete_seed_binding(
        path,
        overrides=overrides,
    )
    return binding, load_qualified_complete_student(path, qualification=binding)


def test_valid_complete_checkpoint_loads_as_opaque_frozen_cpu_handle(
    complete_checkpoint: Path,
) -> None:
    binding, handle = _qualified_load(complete_checkpoint)
    context = student_module._QUALIFIED_STUDENT_CONTEXTS[handle]

    assert type(binding) is QualifiedCompleteSeedBinding
    assert type(handle) is QualifiedStudentHandle
    assert not hasattr(handle, "model")
    assert not hasattr(handle, "optimizer")
    assert context.model.training is False
    assert all(
        parameter.device.type == "cpu" for parameter in context.model.parameters()
    )
    assert all(not parameter.requires_grad for parameter in context.model.parameters())
    assert context.seed == SEED
    assert context.plan_identity == PLAN_IDENTITY
    assert context.corpus_identity == CORPUS_IDENTITY
    assert context.split_identity == SPLIT_IDENTITY
    assert context.training_records_identity == RECORDS_IDENTITY
    assert context.encoded_payload_identity == ENCODED_IDENTITY
    assert len(context.qualification_identity) == 64


def test_opaque_binding_and_handle_cannot_be_forged_or_serialized(
    complete_checkpoint: Path,
) -> None:
    with pytest.raises(StudentQualificationError, match="issued"):
        QualifiedCompleteSeedBinding(object())
    with pytest.raises(StudentQualificationError, match="issued"):
        QualifiedStudentHandle(object())

    binding = _issue_test_qualified_complete_seed_binding(complete_checkpoint)
    assert not hasattr(binding, "__dict__")
    for name in ("model", "optimizer", "context", "injected"):
        with pytest.raises(AttributeError):
            setattr(binding, name, object())
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(binding)
    handle = load_qualified_complete_student(
        complete_checkpoint,
        qualification=binding,
    )
    assert not hasattr(handle, "__dict__")
    for name in ("model", "optimizer", "context", "injected"):
        with pytest.raises(AttributeError):
            setattr(handle, name, object())
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(handle)
    assert set(inspect.signature(load_qualified_complete_student).parameters) == {
        "path",
        "qualification",
    }


def test_qualification_is_path_file_and_single_use_bound(
    complete_checkpoint: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "copy.pt"
    copied.write_bytes(complete_checkpoint.read_bytes())
    binding = _issue_test_qualified_complete_seed_binding(complete_checkpoint)
    with pytest.raises(StudentQualificationError, match="path"):
        load_qualified_complete_student(copied, qualification=binding)
    with pytest.raises(StudentQualificationError, match="unused"):
        load_qualified_complete_student(
            complete_checkpoint,
            qualification=binding,
        )

    binding = _issue_test_qualified_complete_seed_binding(complete_checkpoint)
    original = load_checkpoint(complete_checkpoint)
    _resign(
        complete_checkpoint,
        complete_checkpoint,
        descriptor=replace(original.descriptor, checkpoint_id="changed-after-binding"),
    )
    with pytest.raises(StudentQualificationError, match="file"):
        load_qualified_complete_student(
            complete_checkpoint,
            qualification=binding,
        )


@pytest.mark.parametrize(
    "role",
    ["supervised_seed_latest", "supervised_smoke_disposable"],
)
def test_noncomplete_checkpoint_roles_are_rejected(
    complete_checkpoint: Path,
    tmp_path: Path,
    role: str,
) -> None:
    envelope = load_checkpoint(complete_checkpoint)
    changed = _resign(
        complete_checkpoint,
        tmp_path / f"{role}.pt",
        descriptor=replace(envelope.descriptor, role=role),
    )
    with pytest.raises(StudentQualificationError, match="complete role"):
        _qualified_load(changed)


@pytest.mark.parametrize("mutation", ["purpose", "lineage", "trainer"])
def test_seed_purpose_lineage_and_trainer_are_exact(
    complete_checkpoint: Path,
    tmp_path: Path,
    mutation: str,
) -> None:
    envelope = load_checkpoint(complete_checkpoint)
    implementation = dict(envelope.descriptor.implementation)
    if mutation == "purpose":
        implementation["purpose"] = "smoke"
    elif mutation == "lineage":
        implementation["lineage_kind"] = "fresh_supervised_smoke"
    else:
        implementation["trainer"] = "other-trainer"
    changed = _resign(
        complete_checkpoint,
        tmp_path / f"wrong-{mutation}.pt",
        descriptor=replace(envelope.descriptor, implementation=implementation),
    )
    with pytest.raises(StudentQualificationError, match=mutation):
        _qualified_load(changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epoch", 19),
        ("batch_in_epoch", 1),
        ("update_count", 2239),
        ("sample_cursor", 128),
        ("permutation", list(range(14_336))),
        ("completed", False),
    ],
)
def test_complete_cursor_is_exact(
    complete_checkpoint: Path,
    tmp_path: Path,
    field: str,
    value,
) -> None:
    envelope = load_checkpoint(complete_checkpoint)
    data_state = copy.deepcopy(dict(envelope.payload.data_state))
    cursor = copy.deepcopy(dict(data_state["cursor"]))
    cursor[field] = value
    data_state["cursor"] = cursor
    changed = _resign(
        complete_checkpoint,
        tmp_path / f"cursor-{field}.pt",
        payload=replace(envelope.payload, data_state=data_state),
    )
    with pytest.raises(StudentQualificationError, match="cursor"):
        _qualified_load(changed)


def test_model_config_and_finite_model_state_are_exact(
    complete_checkpoint: Path,
    tmp_path: Path,
) -> None:
    envelope = load_checkpoint(complete_checkpoint)
    trainer_state = copy.deepcopy(dict(envelope.payload.trainer_state))
    trainer_state["model_config"] = {
        **dict(trainer_state["model_config"]),
        "policy_hidden": (64, 64),
    }
    wrong_config = _resign(
        complete_checkpoint,
        tmp_path / "wrong-model-config.pt",
        payload=replace(envelope.payload, trainer_state=trainer_state),
    )
    with pytest.raises(StudentQualificationError, match="model config"):
        _qualified_load(wrong_config)

    model_state = copy.deepcopy(dict(envelope.payload.model_state))
    first_name = next(iter(model_state))
    model_state[first_name] = model_state[first_name].clone()
    model_state[first_name].view(-1)[0] = float("nan")
    data_state = copy.deepcopy(dict(envelope.payload.data_state))
    cache = copy.deepcopy(dict(data_state["cache"]))
    identities = copy.deepcopy(dict(cache["canonical_state_identities"]))
    identities["model"] = canonical_training_state_identity(model_state)
    cache["canonical_state_identities"] = identities
    data_state["cache"] = cache
    # Envelope v2 rejects a non-finite payload before it can be re-signed; the
    # qualified loader independently repeats this invariant after deserialize.
    with pytest.raises(CheckpointFormatError, match="non-finite"):
        replace(
            envelope.payload,
            model_state=model_state,
            data_state=data_state,
        )


def test_adam_step_and_complete_recovery_are_exact(
    complete_checkpoint: Path,
    tmp_path: Path,
) -> None:
    envelope = load_checkpoint(complete_checkpoint)
    optimizer_state = copy.deepcopy(dict(envelope.payload.optimizer_state))
    first_index = next(iter(optimizer_state["state"]))
    optimizer_state["state"][first_index]["step"] = torch.tensor(2239.0)
    data_state = copy.deepcopy(dict(envelope.payload.data_state))
    cache = copy.deepcopy(dict(data_state["cache"]))
    identities = copy.deepcopy(dict(cache["canonical_state_identities"]))
    identities["optimizer"] = canonical_training_state_identity(optimizer_state)
    cache["canonical_state_identities"] = identities
    data_state["cache"] = cache
    wrong_step = _resign(
        complete_checkpoint,
        tmp_path / "wrong-adam-step.pt",
        payload=replace(
            envelope.payload,
            optimizer_state=optimizer_state,
            data_state=data_state,
        ),
    )
    with pytest.raises(StudentQualificationError, match="Adam|2,240"):
        _qualified_load(wrong_step)

    trainer_state = copy.deepcopy(dict(envelope.payload.trainer_state))
    trainer_state["recovery_state"] = {
        "exact_resume": False,
        "cursor": dict(envelope.payload.data_state["cursor"]),
    }
    wrong_recovery = _resign(
        complete_checkpoint,
        tmp_path / "wrong-recovery.pt",
        payload=replace(envelope.payload, trainer_state=trainer_state),
    )
    with pytest.raises(StudentQualificationError, match="recovery"):
        _qualified_load(wrong_recovery)


@pytest.mark.parametrize(
    "asset",
    ["plan", "corpus", "split", "training_records", "encoded_payload"],
)
def test_descriptor_and_payload_asset_identities_are_cross_bound(
    complete_checkpoint: Path,
    tmp_path: Path,
    asset: str,
) -> None:
    envelope = load_checkpoint(complete_checkpoint)
    assets = dict(envelope.descriptor.asset_identities)
    assets[asset] = "f" * 64
    crossed = _resign(
        complete_checkpoint,
        tmp_path / f"crossed-{asset}.pt",
        descriptor=replace(envelope.descriptor, asset_identities=assets),
    )
    with pytest.raises(StudentQualificationError, match="identity"):
        _qualified_load(crossed)


def test_qualification_corpus_and_model_identities_are_independent(
    complete_checkpoint: Path,
) -> None:
    with pytest.raises(StudentQualificationError, match="corpus|qualification"):
        _qualified_load(
            complete_checkpoint,
            overrides={"corpus_identity": "f" * 64},
        )
    with pytest.raises(StudentQualificationError, match="model|qualification"):
        _qualified_load(
            complete_checkpoint,
            overrides={"model_identity": "f" * 64},
        )


def test_checkpoint_file_identity_is_full_sha256(
    complete_checkpoint: Path,
) -> None:
    binding = _issue_test_qualified_complete_seed_binding(complete_checkpoint)
    context = student_module._QUALIFICATION_CONTEXTS[binding]
    assert (
        context.file_sha256
        == hashlib.sha256(complete_checkpoint.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("config_sha256", "f" * 64, "config"),
        ("feature_schema_version", "wrong-feature", "feature schema"),
        ("label_schema_version", "wrong-label", "label schema"),
        ("database_schema_versions", {"corpus": "wrong-corpus"}, "corpus schema"),
    ],
)
def test_checkpoint_schemas_and_config_identity_are_exact(
    complete_checkpoint: Path,
    tmp_path: Path,
    field: str,
    value,
    message: str,
) -> None:
    envelope = load_checkpoint(complete_checkpoint)
    changed = _resign(
        complete_checkpoint,
        tmp_path / f"wrong-{field}.pt",
        descriptor=replace(envelope.descriptor, **{field: value}),
    )
    with pytest.raises(StudentQualificationError, match=message):
        _qualified_load(changed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("optimizer", "SGD", "Adam"),
        ("seed", "2026083999", "seed"),
        ("training_kind", "reinforcement-learning", "training kind"),
        ("start_mode", "exact-resume", "lineage start"),
        ("expected_initial_model_state_identity", "f" * 64, "initial model"),
    ],
)
def test_complete_implementation_contract_is_exact(
    complete_checkpoint: Path,
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    envelope = load_checkpoint(complete_checkpoint)
    implementation = dict(envelope.descriptor.implementation)
    implementation[field] = value
    changed = _resign(
        complete_checkpoint,
        tmp_path / f"wrong-implementation-{field}.pt",
        descriptor=replace(envelope.descriptor, implementation=implementation),
    )
    with pytest.raises(StudentQualificationError, match=message):
        _qualified_load(changed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("parent_checkpoint_id", "unexpected-parent", "parent checkpoint"),
        ("save_reason", "supervised-pause", "save reason"),
        ("checkpoint_id", "unrelated-complete", "checkpoint identity"),
    ],
)
def test_complete_descriptor_metadata_is_exact(
    complete_checkpoint: Path,
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    envelope = load_checkpoint(complete_checkpoint)
    changed = _resign(
        complete_checkpoint,
        tmp_path / f"wrong-descriptor-{field}.pt",
        descriptor=replace(envelope.descriptor, **{field: value}),
    )
    with pytest.raises(StudentQualificationError, match=message):
        _qualified_load(changed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("game_count", 1, "game count"),
        ("batch_count", 2_239, "update counts"),
        ("update_count", 2_239, "update counts"),
        ("difficulty", 10, "difficulty"),
        ("temperature", 1, "temperature"),
        ("rolling_metrics", {"loss": 0.0}, "metrics"),
        ("target_network", {"enabled": True}, "target-network"),
    ],
)
def test_complete_trainer_accounting_is_exact(
    complete_checkpoint: Path,
    tmp_path: Path,
    field: str,
    value,
    message: str,
) -> None:
    envelope = load_checkpoint(complete_checkpoint)
    trainer_state = copy.deepcopy(dict(envelope.payload.trainer_state))
    trainer_state[field] = value
    changed = _resign(
        complete_checkpoint,
        tmp_path / f"wrong-trainer-{field}.pt",
        payload=replace(envelope.payload, trainer_state=trainer_state),
    )
    with pytest.raises(StudentQualificationError, match=message):
        _qualified_load(changed)


@pytest.mark.parametrize("field", ["scheduler_state", "scaler_state"])
def test_scheduler_and_scaler_must_be_absent(
    complete_checkpoint: Path,
    tmp_path: Path,
    field: str,
) -> None:
    envelope = load_checkpoint(complete_checkpoint)
    changed = _resign(
        complete_checkpoint,
        tmp_path / f"wrong-{field}.pt",
        payload=replace(envelope.payload, **{field: {"enabled": True}}),
    )
    with pytest.raises(StudentQualificationError, match="scheduler/scaler"):
        _qualified_load(changed)


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_strict_model_keys_are_required(
    complete_checkpoint: Path,
    tmp_path: Path,
    mutation: str,
) -> None:
    envelope = load_checkpoint(complete_checkpoint)
    model_state = copy.deepcopy(dict(envelope.payload.model_state))
    if mutation == "extra":
        model_state["unexpected.weight"] = torch.zeros(1)
    else:
        del model_state[next(iter(model_state))]
    data_state = copy.deepcopy(dict(envelope.payload.data_state))
    cache = copy.deepcopy(dict(data_state["cache"]))
    identities = copy.deepcopy(dict(cache["canonical_state_identities"]))
    identities["model"] = canonical_training_state_identity(model_state)
    cache["canonical_state_identities"] = identities
    data_state["cache"] = cache
    changed = _resign(
        complete_checkpoint,
        tmp_path / f"{mutation}-model-key.pt",
        payload=replace(
            envelope.payload,
            model_state=model_state,
            data_state=data_state,
        ),
    )
    with pytest.raises(StudentQualificationError, match="exact model config"):
        _qualified_load(changed)


def test_canonical_model_cache_is_required(
    complete_checkpoint: Path,
    tmp_path: Path,
) -> None:
    envelope = load_checkpoint(complete_checkpoint)
    data_state = copy.deepcopy(dict(envelope.payload.data_state))
    cache = copy.deepcopy(dict(data_state["cache"]))
    identities = copy.deepcopy(dict(cache["canonical_state_identities"]))
    identities["model"] = "f" * 64
    cache["canonical_state_identities"] = identities
    data_state["cache"] = cache
    changed = _resign(
        complete_checkpoint,
        tmp_path / "wrong-model-cache.pt",
        payload=replace(envelope.payload, data_state=data_state),
    )
    with pytest.raises(StudentQualificationError, match="canonical identity"):
        _qualified_load(changed)


def test_complete_permutation_rng_is_replayed_not_merely_self_reported(
    complete_checkpoint: Path,
    tmp_path: Path,
) -> None:
    envelope = load_checkpoint(complete_checkpoint)
    rng_state = copy.deepcopy(dict(envelope.payload.rng_state))
    rng_state["components"] = {
        "permutation_generator": torch.Generator(device="cpu")
        .manual_seed(123)
        .get_state()
    }
    data_state = copy.deepcopy(dict(envelope.payload.data_state))
    cache = copy.deepcopy(dict(data_state["cache"]))
    identities = copy.deepcopy(dict(cache["canonical_state_identities"]))
    identities["rng"] = canonical_training_state_identity(rng_state)
    cache["canonical_state_identities"] = identities
    data_state["cache"] = cache
    changed = _resign(
        complete_checkpoint,
        tmp_path / "wrong-permutation-rng.pt",
        payload=replace(
            envelope.payload,
            rng_state=rng_state,
            data_state=data_state,
        ),
    )
    with pytest.raises(StudentQualificationError, match="permutation RNG"):
        _qualified_load(changed)


@pytest.mark.parametrize(
    "override",
    [
        {"checkpoint_id": "wrong-checkpoint"},
        {"run_id": "wrong-run"},
        {"experiment_id": "wrong-experiment"},
        {"plan_identity": "f" * 64},
        {"corpus_identity": "f" * 64},
        {"split_identity": "f" * 64},
        {"training_records_identity": "f" * 64},
        {"encoded_payload_identity": "f" * 64},
        {"model_identity": "f" * 64},
        {"seed": 2026083002},
    ],
)
def test_every_semantic_identity_is_qualification_bound(
    complete_checkpoint: Path,
    override: dict,
) -> None:
    with pytest.raises(StudentQualificationError, match="qualification"):
        _qualified_load(complete_checkpoint, overrides=override)


@pytest.mark.parametrize(
    "override",
    [
        {"file_size": 1},
        {"file_sha256": "f" * 64},
        {"checkpoint_file_identity": "f" * 64},
        {"payload_size": 1},
        {"payload_sha256": "f" * 64},
    ],
)
def test_file_and_payload_bytes_are_qualification_bound(
    complete_checkpoint: Path,
    override: dict,
) -> None:
    binding = _issue_test_qualified_complete_seed_binding(
        complete_checkpoint,
        overrides=override,
    )
    with pytest.raises(StudentQualificationError, match="file"):
        load_qualified_complete_student(
            complete_checkpoint,
            qualification=binding,
        )
