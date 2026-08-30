from __future__ import annotations

import inspect
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import learned_ai.training.classical_a_pos_supervised as supervised_module
from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.training.checkpoint_envelope import load_checkpoint, save_checkpoint
from learned_ai.training.classical_a_pos_corpus import FrozenCorpus
from learned_ai.training.classical_a_pos_supervised import (
    FreshSupervisedSession,
    SupervisedExample,
    SupervisedLoopConfig,
    SupervisedPlanBinding,
    SupervisedState,
    SupervisedTrainingError,
    _create_fresh_production_student,
    _encode_frozen_training_examples,
    _issue_test_plan_binding,
    _load_supervised_checkpoint,
    _run_supervised_updates,
    _run_supervised_smoke_update,
    _save_supervised_checkpoint,
    canonical_training_state_identity,
    create_fresh_supervised_state,
    encode_frozen_training_examples,
    hard_a_pos_masked_cross_entropy,
    issue_fresh_supervised_session,
    load_supervised_checkpoint,
    make_policy_only_adam,
    production_loop_config,
    run_supervised_updates,
    run_supervised_smoke_update,
    save_supervised_checkpoint,
)


def _examples(count: int = 4) -> tuple[SupervisedExample, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(991)
    examples = []
    for index in range(count):
        features = torch.randn((3, 62), generator=generator)
        examples.append(
            SupervisedExample(
                features=features,
                a_pos_mask=torch.tensor([True, False, True]),
                teacher_index=0 if index % 2 == 0 else 2,
            )
        )
    return tuple(examples)


@pytest.fixture
def production_corpus(monkeypatch: pytest.MonkeyPatch) -> FrozenCorpus:
    synthetic_examples = (_examples(1)[0],) * 14_336
    monkeypatch.setattr(
        supervised_module,
        "encode_frozen_training_examples",
        lambda _corpus: synthetic_examples,
    )
    monkeypatch.setattr(
        supervised_module,
        "_ordered_training_records_identity",
        lambda _corpus: "d" * 64,
    )
    monkeypatch.setattr(
        supervised_module,
        "_encoded_training_payload_identity",
        lambda _examples: "e" * 64,
    )
    return FrozenCorpus(
        manifest={},
        examples=(),
        corpus_identity="a" * 64,
        split_identity="b" * 64,
    )


def _issue_test_session(
    corpus: FrozenCorpus,
    *,
    seed: int = 2026083001,
    plan_identity: str = "c" * 64,
    run_id: str | None = None,
):
    binding = _issue_test_plan_binding(
        corpus,
        plan_identity=plan_identity,
        run_id=run_id or f"seed-{seed}",
        experiment_id="classical-a-pos-supervised",
        seed=seed,
        purpose="seed",
    )
    return issue_fresh_supervised_session(corpus, plan_binding=binding)


def _model_and_optimizer(seed: int):
    torch.manual_seed(seed)
    model = ScaffoldedPolicyNet(
        move_feat_dim=62,
        policy_hidden=(128, 64),
        value_hidden=(),
        dropout=0.0,
    )
    optimizer = make_policy_only_adam(model, lr=1e-3)
    return model, optimizer


def test_masked_ce_gives_unsafe_logits_exactly_zero_gradient() -> None:
    logits = torch.tensor([1.0, 1000.0, -1.0], requires_grad=True)

    loss = hard_a_pos_masked_cross_entropy(
        logits,
        torch.tensor([True, False, True]),
        teacher_index=0,
    )
    loss.backward()

    assert torch.equal(logits.grad[1], torch.tensor(0.0))
    assert torch.allclose(
        loss.detach(),
        torch.nn.functional.cross_entropy(
            torch.tensor([[1.0, -1.0]]), torch.tensor([0])
        ),
    )


def test_masked_ce_rejects_nonfinite_unsafe_logit() -> None:
    logits = torch.tensor([1.0, float("nan"), -1.0], requires_grad=True)

    with pytest.raises(SupervisedTrainingError, match="full legal policy logits"):
        hard_a_pos_masked_cross_entropy(
            logits,
            torch.tensor([True, False, True]),
            teacher_index=0,
        )


def test_optimizer_contains_policy_mlp_only() -> None:
    model, optimizer = _model_and_optimizer(4)
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert optimized == {id(parameter) for parameter in model.policy_mlp.parameters()}
    assert optimized.isdisjoint(
        {id(parameter) for parameter in model.value_mlp.parameters()}
    )


def test_optimizer_type_and_all_group_semantics_fail_closed() -> None:
    examples = _examples()
    config = SupervisedLoopConfig(
        batch_size=2,
        epochs=2,
        grad_clip_norm=1.0,
        permutation_seed=2026083001,
    )

    wrong_group_model, wrong_group_optimizer = _model_and_optimizer(3)
    wrong_group_optimizer.param_groups[0]["lr"] = 2e-3
    generator = torch.Generator(device="cpu")
    state = create_fresh_supervised_state(
        permutation_generator=generator,
        config=config,
    )
    with pytest.raises(SupervisedTrainingError, match="Adam group semantics"):
        _run_supervised_updates(
            wrong_group_model,
            wrong_group_optimizer,
            examples,
            state,
            permutation_generator=generator,
            config=config,
            max_updates=1,
        )

    sgd_model, _unused_adam = _model_and_optimizer(4)
    sgd = torch.optim.SGD(sgd_model.policy_mlp.parameters(), lr=1e-3)
    generator = torch.Generator(device="cpu")
    state = create_fresh_supervised_state(
        permutation_generator=generator,
        config=config,
    )
    with pytest.raises(SupervisedTrainingError, match="frozen torch Adam"):
        _run_supervised_updates(
            sgd_model,
            sgd,
            examples,
            state,
            permutation_generator=generator,
            config=config,
            max_updates=1,
        )

    reordered_model, reordered_optimizer = _model_and_optimizer(5)
    reordered_optimizer.param_groups[0]["params"] = list(
        reversed(reordered_optimizer.param_groups[0]["params"])
    )
    generator = torch.Generator(device="cpu")
    state = create_fresh_supervised_state(
        permutation_generator=generator,
        config=config,
    )
    with pytest.raises(SupervisedTrainingError, match="parameter order"):
        _run_supervised_updates(
            reordered_model,
            reordered_optimizer,
            examples,
            state,
            permutation_generator=generator,
            config=config,
            max_updates=1,
        )


def test_production_profile_rejects_old_model_and_mini_loop() -> None:
    torch.manual_seed(4)
    old_model = ScaffoldedPolicyNet(
        move_feat_dim=62,
        policy_hidden=(8,),
        value_hidden=(),
        dropout=0.0,
    )
    with pytest.raises(SupervisedTrainingError, match=r"policy\(128,64\)"):
        make_policy_only_adam(old_model)

    config = production_loop_config(2026083001)
    assert config.batch_size == 128
    assert config.epochs == 20
    with pytest.raises(SupervisedTrainingError, match="verified FrozenCorpus"):
        issue_fresh_supervised_session(
            SimpleNamespace(examples=()),
            plan_binding=object(),
        )


def test_corpus_encoding_never_reads_forbidden_advisors_or_databases() -> None:
    board = BoardState.new_game()
    legal = get_all_legal_moves(board)
    example = SimpleNamespace(
        split="train",
        board=board,
        candidate_color="W",
        legal_actions=tuple(legal),
        a_pos_mask=tuple(index < 2 for index in range(len(legal))),
        teacher_index=0,
    )
    corpus = SimpleNamespace(examples=(example,))
    calls: list[dict] = []

    def encoder(_board, _player, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            legal_moves=legal,
            feat_matrix=np.zeros((len(legal), 62), dtype=np.float32),
        )

    encoded = _encode_frozen_training_examples(corpus, encoder=encoder)

    assert len(encoded) == 1
    assert calls == [
        {
            "sentinel_advisor": None,
            "db": None,
            "value_net": None,
            "specialist_db": None,
            "wdl_db": None,
            "strict": True,
        }
    ]


def test_public_encoder_rejects_unverified_corpus_object() -> None:
    with pytest.raises(SupervisedTrainingError, match="verified FrozenCorpus"):
        encode_frozen_training_examples(SimpleNamespace(examples=()))
    with pytest.raises(SupervisedTrainingError, match="verified FrozenCorpus"):
        issue_fresh_supervised_session(
            SimpleNamespace(examples=()),
            plan_binding=object(),
        )


def test_frozen_student_seeds_create_fresh_exact_models() -> None:
    random.seed(771)
    np.random.seed(772)
    torch.manual_seed(773)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    first, first_optimizer = _create_fresh_production_student(
        2026083001,
        purpose=supervised_module.SupervisedPurpose.SEED,
    )
    repeated, repeated_optimizer = _create_fresh_production_student(
        2026083001,
        purpose=supervised_module.SupervisedPurpose.SEED,
    )
    other, _other_optimizer = _create_fresh_production_student(
        2026083002,
        purpose=supervised_module.SupervisedPurpose.SEED,
    )

    assert canonical_training_state_identity(first.state_dict()) == (
        canonical_training_state_identity(repeated.state_dict())
    )
    assert canonical_training_state_identity(first.state_dict()) != (
        canonical_training_state_identity(other.state_dict())
    )
    assert first.get_config()["policy_hidden"] == (128, 64)
    assert first.get_config()["value_hidden"] == ()
    assert first_optimizer.state == {}
    assert repeated_optimizer.state == {}
    assert random.getstate() == python_state
    observed_numpy_state = np.random.get_state()
    assert observed_numpy_state[0] == numpy_state[0]
    assert np.array_equal(observed_numpy_state[1], numpy_state[1])
    assert observed_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_cursor_algebra_and_adam_step_prevent_duplicate_update() -> None:
    examples = _examples()
    config = SupervisedLoopConfig(
        batch_size=2,
        epochs=2,
        grad_clip_norm=1.0,
        permutation_seed=2026083001,
    )
    model, optimizer = _model_and_optimizer(11)
    generator = torch.Generator(device="cpu")
    fresh = create_fresh_supervised_state(
        permutation_generator=generator,
        config=config,
    )
    advanced = _run_supervised_updates(
        model,
        optimizer,
        examples,
        fresh,
        permutation_generator=generator,
        config=config,
        max_updates=1,
    )

    assert advanced.batch_in_epoch == 1
    assert advanced.sample_cursor == 2
    with pytest.raises(
        SupervisedTrainingError,
        match="generator state|repeated or missing update",
    ):
        _run_supervised_updates(
            model,
            optimizer,
            examples,
            fresh,
            permutation_generator=generator,
            config=config,
            max_updates=1,
        )
    forged = replace(advanced, update_count=0)
    with pytest.raises(SupervisedTrainingError, match="cursor algebra"):
        _run_supervised_updates(
            model,
            optimizer,
            examples,
            forged,
            permutation_generator=generator,
            config=config,
            max_updates=1,
        )


def test_supervised_smoke_is_one_real_finite_non_rl_update() -> None:
    model, optimizer = _create_fresh_production_student(
        2026083001,
        purpose=supervised_module.SupervisedPurpose.SEED,
    )

    result = _run_supervised_smoke_update(
        model,
        optimizer,
        _examples(128),
        seed=2026083001,
    )

    assert result.optimizer_updates == 1
    assert result.finite_loss is True
    assert result.finite_gradient_norm is True
    assert result.parameters_changed is True
    assert result.reinforcement_learning is False
    assert result.training_semantics.startswith("offline-supervised")
    assert all(parameter.grad is None for parameter in model.value_mlp.parameters())


def test_smoke_batch_identity_is_domain_separated_ordered_exact_128() -> None:
    examples = _examples(129)

    with pytest.raises(SupervisedTrainingError, match="128"):
        supervised_module._smoke_batch_identity(examples[:127])
    with pytest.raises(SupervisedTrainingError, match="128"):
        supervised_module._smoke_batch_identity(examples)

    expected = supervised_module._smoke_batch_identity(examples[:128])
    reordered = list(examples[:128])
    reordered[0], reordered[1] = reordered[1], reordered[0]
    assert supervised_module._smoke_batch_identity(tuple(reordered)) != expected
    drifted = list(examples[:128])
    drifted[0] = SupervisedExample(
        features=drifted[0].features.clone().add_(0.25),
        a_pos_mask=drifted[0].a_pos_mask.clone(),
        teacher_index=drifted[0].teacher_index,
    )
    assert supervised_module._smoke_batch_identity(tuple(drifted)) != expected


def test_public_smoke_uses_real_domain_separated_batch_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = _examples(128) * 112
    monkeypatch.setattr(
        supervised_module,
        "encode_frozen_training_examples",
        lambda _corpus: encoded,
    )
    monkeypatch.setattr(
        supervised_module,
        "_ordered_training_records_identity",
        lambda _corpus: "d" * 64,
    )
    corpus = FrozenCorpus(
        manifest={},
        examples=(),
        corpus_identity="a" * 64,
        split_identity="b" * 64,
    )
    binding = _issue_test_plan_binding(
        corpus,
        plan_identity="c" * 64,
        run_id="real-smoke-batch-identity",
        experiment_id="classical-a-pos-supervised",
        seed=2026083001,
        purpose="smoke",
    )
    session = supervised_module.issue_supervised_smoke_session(
        corpus,
        plan_binding=binding,
    )

    result = run_supervised_smoke_update(session)
    assert result.optimizer_updates == 1
    target = supervised_module.save_supervised_smoke_checkpoint(
        tmp_path / "real-smoke-batch-identity.pt",
        session=session,
        created_at_utc="2026-08-30T00:00:00Z",
    )
    envelope = load_checkpoint(target)
    assert envelope.descriptor.asset_identities["smoke_batch"] == (
        supervised_module._smoke_batch_identity(encoded[:128])
    )


def test_nonfinite_policy_gradient_fails_before_adam_step() -> None:
    examples = _examples()
    config = SupervisedLoopConfig(
        batch_size=2,
        epochs=2,
        grad_clip_norm=1.0,
        permutation_seed=2026083001,
    )
    model, optimizer = _model_and_optimizer(12)
    first_parameter = next(model.policy_mlp.parameters())
    first_parameter.register_hook(
        lambda gradient: torch.full_like(gradient, float("nan"))
    )
    generator = torch.Generator(device="cpu")
    state = create_fresh_supervised_state(
        permutation_generator=generator,
        config=config,
    )

    with pytest.raises(SupervisedTrainingError, match="non-finite"):
        _run_supervised_updates(
            model,
            optimizer,
            examples,
            state,
            permutation_generator=generator,
            config=config,
            max_updates=1,
        )
    assert optimizer.state == {}


def _session_context(session):
    return supervised_module._SESSION_CONTEXTS[session]


def _save_fresh_production_checkpoint(
    tmp_path: Path,
    corpus: FrozenCorpus,
    *,
    seed: int = 2026083001,
    plan_identity: str = "c" * 64,
) -> tuple[Path, FreshSupervisedSession, SupervisedState]:
    session, state = _issue_test_session(
        corpus,
        seed=seed,
        plan_identity=plan_identity,
    )
    checkpoint = tmp_path / f"seed-{seed}-fresh.pt"
    save_supervised_checkpoint(
        checkpoint,
        session=session,
        state=state,
        created_at_utc="2026-08-30T00:00:00Z",
    )
    return checkpoint, session, state


def _load_bound_checkpoint(
    checkpoint: Path,
    corpus: FrozenCorpus,
    *,
    seed: int = 2026083001,
    plan_identity: str = "c" * 64,
    run_id: str | None = None,
):
    fresh_session, _fresh_state = _issue_test_session(
        corpus,
        seed=seed,
        plan_identity=plan_identity,
        run_id=run_id,
    )
    resumed_session, state = load_supervised_checkpoint(
        checkpoint,
        session=fresh_session,
    )
    return fresh_session, resumed_session, state


def test_public_checkpoint_accepts_only_matching_bound_session(
    tmp_path: Path,
    production_corpus: FrozenCorpus,
) -> None:
    checkpoint, _session, _state = _save_fresh_production_checkpoint(
        tmp_path,
        production_corpus,
    )
    envelope = load_checkpoint(checkpoint)

    assert envelope.descriptor.role == "supervised_seed_latest"
    assert envelope.descriptor.asset_identities["training_records"] == "d" * 64
    assert envelope.descriptor.asset_identities["encoded_payload"] == "e" * 64
    initial_identity = envelope.descriptor.implementation[
        "expected_initial_model_state_identity"
    ]
    assert initial_identity == canonical_training_state_identity(
        _create_fresh_production_student(
            2026083001,
            purpose=supervised_module.SupervisedPurpose.SEED,
        )[0].state_dict()
    )
    _fresh, resumed, state = _load_bound_checkpoint(
        checkpoint,
        production_corpus,
    )
    context = _session_context(resumed)
    assert state.update_count == 0
    assert canonical_training_state_identity(context.model.state_dict()) == (
        initial_identity
    )
    assert context.optimizer.state == {}


def test_public_surface_cannot_relabel_or_switch_bound_corpus(
    tmp_path: Path,
    production_corpus: FrozenCorpus,
) -> None:
    checkpoint, session, state = _save_fresh_production_checkpoint(
        tmp_path,
        production_corpus,
    )
    assert set(inspect.signature(save_supervised_checkpoint).parameters) == {
        "path",
        "session",
        "state",
        "created_at_utc",
    }
    assert set(inspect.signature(run_supervised_updates).parameters) == {
        "session",
        "state",
        "max_updates",
        "update_observer",
    }
    assert set(inspect.signature(load_supervised_checkpoint).parameters) == {
        "path",
        "session",
    }
    assert set(inspect.signature(issue_fresh_supervised_session).parameters) == {
        "corpus",
        "plan_binding",
    }
    second = tmp_path / "same-bound-identity.pt"
    save_supervised_checkpoint(
        second,
        session=session,
        state=state,
        created_at_utc="2026-08-30T00:00:00Z",
    )
    assert load_checkpoint(second).descriptor.asset_identities == (
        load_checkpoint(checkpoint).descriptor.asset_identities
    )

    different_corpus = replace(production_corpus, corpus_identity="d" * 64)
    with pytest.raises(SupervisedTrainingError, match="checkpoint identity"):
        _load_bound_checkpoint(checkpoint, different_corpus)
    with pytest.raises(SupervisedTrainingError, match="checkpoint identity"):
        _load_bound_checkpoint(checkpoint, production_corpus, plan_identity="d" * 64)
    with pytest.raises(SupervisedTrainingError, match="checkpoint identity"):
        _load_bound_checkpoint(checkpoint, production_corpus, seed=2026083002)

    with pytest.raises(SupervisedTrainingError, match="controller-issued"):
        SupervisedPlanBinding(object())
    binding = _issue_test_plan_binding(
        production_corpus,
        plan_identity="c" * 64,
        run_id="one-shot-binding",
        experiment_id="classical-a-pos-supervised",
        seed=2026083001,
        purpose="seed",
    )
    issue_fresh_supervised_session(production_corpus, plan_binding=binding)
    with pytest.raises(SupervisedTrainingError, match="unused controller plan binding"):
        issue_fresh_supervised_session(production_corpus, plan_binding=binding)


def test_resume_binds_ordered_records_and_encoded_payload(
    tmp_path: Path,
    production_corpus: FrozenCorpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, _session, _state = _save_fresh_production_checkpoint(
        tmp_path,
        production_corpus,
    )
    base = _examples(1)[0]
    alternate = SupervisedExample(
        features=base.features.clone(),
        a_pos_mask=base.a_pos_mask.clone(),
        teacher_index=2,
    )
    monkeypatch.setattr(
        supervised_module,
        "_ordered_training_records_identity",
        lambda corpus: corpus.manifest.get("training_records_identity", "d" * 64),
    )
    monkeypatch.setattr(
        supervised_module,
        "encode_frozen_training_examples",
        lambda corpus: (
            (alternate if corpus.manifest.get("alternate_encoding") else base),
        )
        * 14_336,
    )
    monkeypatch.setattr(
        supervised_module,
        "_encoded_training_payload_identity",
        lambda examples: "f" * 64 if examples[0].teacher_index == 2 else "e" * 64,
    )

    same_top_level_different_records = replace(
        production_corpus,
        manifest={"training_records_identity": "f" * 64},
    )
    with pytest.raises(SupervisedTrainingError, match="checkpoint identity"):
        _load_bound_checkpoint(checkpoint, same_top_level_different_records)
    same_top_level_different_encoding = replace(
        production_corpus,
        manifest={"alternate_encoding": True},
    )
    with pytest.raises(SupervisedTrainingError, match="checkpoint identity"):
        _load_bound_checkpoint(checkpoint, same_top_level_different_encoding)


def test_resigned_initial_identity_and_smoke_role_cannot_resume_seed(
    tmp_path: Path,
    production_corpus: FrozenCorpus,
) -> None:
    checkpoint, _session, _state = _save_fresh_production_checkpoint(
        tmp_path,
        production_corpus,
    )
    envelope = load_checkpoint(checkpoint)

    replacement_identity = "f" * 64
    implementation = dict(envelope.descriptor.implementation)
    implementation["expected_initial_model_state_identity"] = replacement_identity
    data_state = dict(envelope.payload.data_state)
    buckets = dict(data_state["buckets"])
    lineage = dict(buckets["fresh_lineage"])
    lineage["expected_initial_model_state_identity"] = replacement_identity
    buckets["fresh_lineage"] = lineage
    data_state["buckets"] = buckets
    resigned_identity = tmp_path / "resigned-initial-identity.pt"
    save_checkpoint(
        resigned_identity,
        replace(envelope.descriptor, implementation=implementation),
        replace(envelope.payload, data_state=data_state),
        previous_copies=0,
    )
    with pytest.raises(SupervisedTrainingError, match="fresh lineage differs"):
        _load_bound_checkpoint(resigned_identity, production_corpus)

    resigned_smoke = tmp_path / "resigned-smoke-role.pt"
    save_checkpoint(
        resigned_smoke,
        replace(envelope.descriptor, role="supervised_smoke_disposable"),
        envelope.payload,
        previous_copies=0,
    )
    with pytest.raises(SupervisedTrainingError, match="checkpoint role differs"):
        _load_bound_checkpoint(resigned_smoke, production_corpus)


def test_public_state_zero_resume_advances_and_resumes_one_real_update(
    tmp_path: Path,
    production_corpus: FrozenCorpus,
) -> None:
    checkpoint, _source_session, _source_state = _save_fresh_production_checkpoint(
        tmp_path, production_corpus
    )
    consumed_fresh, resumed, state = _load_bound_checkpoint(
        checkpoint,
        production_corpus,
    )
    with pytest.raises(SupervisedTrainingError, match="session capability"):
        save_supervised_checkpoint(
            tmp_path / "consumed-fresh.pt",
            session=consumed_fresh,
            state=state,
            created_at_utc="2026-08-30T00:00:00Z",
        )

    resaved_zero = tmp_path / "resaved-update-zero.pt"
    save_supervised_checkpoint(
        resaved_zero,
        session=resumed,
        state=state,
        created_at_utc="2026-08-30T00:00:00Z",
    )
    assert load_checkpoint(resaved_zero).descriptor.role == "supervised_seed_latest"

    state = run_supervised_updates(resumed, state, max_updates=1)
    context = _session_context(resumed)
    assert state.update_count == 1
    assert state.batch_in_epoch == 1
    assert all(
        int(item["step"].item()) == 1 for item in context.optimizer.state.values()
    )

    update_one = tmp_path / "update-one.pt"
    save_supervised_checkpoint(
        update_one,
        session=resumed,
        state=state,
        created_at_utc="2026-08-30T00:00:00Z",
    )
    _fresh_two, resumed_two, resumed_state = _load_bound_checkpoint(
        update_one,
        production_corpus,
    )
    resumed_context = _session_context(resumed_two)
    assert resumed_state == state
    assert all(
        int(item["step"].item()) == 1
        for item in resumed_context.optimizer.state.values()
    )
    assert canonical_training_state_identity(resumed_context.model.state_dict()) == (
        canonical_training_state_identity(context.model.state_dict())
    )

    other_session, other_state = _issue_test_session(production_corpus)
    with pytest.raises(
        SupervisedTrainingError,
        match="factory-signed fresh lineage|generator|Adam step",
    ):
        run_supervised_updates(other_session, state, max_updates=1)
    assert other_state.update_count == 0


def test_session_generator_swap_or_drift_fails_closed(
    tmp_path: Path,
    production_corpus: FrozenCorpus,
) -> None:
    session, state = _issue_test_session(production_corpus)
    context = _session_context(session)
    other_session, _other_state = _issue_test_session(production_corpus)
    other_context = _session_context(other_session)
    canonical_clone = torch.Generator(device="cpu")
    canonical_clone.set_state(context.permutation_generator.get_state())
    crossed_contexts = (
        replace(context, model=other_context.model),
        replace(context, optimizer=other_context.optimizer),
        replace(context, permutation_generator=other_context.permutation_generator),
        replace(context, permutation_generator=canonical_clone),
    )
    for index, crossed in enumerate(crossed_contexts):
        supervised_module._SESSION_CONTEXTS[session] = crossed
        try:
            with pytest.raises(SupervisedTrainingError, match="session capability"):
                save_supervised_checkpoint(
                    tmp_path / f"crossed-capability-{index}.pt",
                    session=session,
                    state=state,
                    created_at_utc="2026-08-30T00:00:00Z",
                )
        finally:
            supervised_module._SESSION_CONTEXTS[session] = context

    torch.rand(1, generator=context.permutation_generator)
    with pytest.raises(SupervisedTrainingError, match="generator state"):
        save_supervised_checkpoint(
            tmp_path / "advanced-generator.pt",
            session=session,
            state=state,
            created_at_utc="2026-08-30T00:00:00Z",
        )


def test_public_smoke_route_cannot_be_saved_or_advanced_as_seed(
    tmp_path: Path,
    production_corpus: FrozenCorpus,
) -> None:
    binding = _issue_test_plan_binding(
        production_corpus,
        plan_identity="c" * 64,
        run_id="smoke-route",
        experiment_id="classical-a-pos-supervised",
        seed=2026083001,
        purpose="smoke",
    )
    session = supervised_module.issue_supervised_smoke_session(
        production_corpus,
        plan_binding=binding,
    )
    state = SupervisedState(0, 0, 0, 0, None, False)
    result = run_supervised_smoke_update(session)
    assert result.optimizer_updates == 1

    with pytest.raises(SupervisedTrainingError, match="purpose|session kind"):
        save_supervised_checkpoint(
            tmp_path / "smoke-cannot-be-seed.pt",
            session=session,
            state=state,
            created_at_utc="2026-08-30T00:00:00Z",
        )
    with pytest.raises(SupervisedTrainingError, match="purpose|session kind"):
        run_supervised_updates(session, state, max_updates=1)


def test_complete_checkpoint_cannot_issue_resumed_session(
    tmp_path: Path,
    production_corpus: FrozenCorpus,
) -> None:
    session, _state = _issue_test_session(production_corpus)
    context = _session_context(session)
    for parameter in context.model.policy_mlp.parameters():
        context.optimizer.state[parameter] = {
            "step": torch.tensor(2240.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    for _epoch in range(20):
        torch.randperm(14_336, generator=context.permutation_generator)
    complete = SupervisedState(
        epoch=20,
        batch_in_epoch=0,
        update_count=2_240,
        sample_cursor=0,
        permutation=None,
        completed=True,
    )
    complete_path = tmp_path / "complete-must-not-resume.pt"
    _save_supervised_checkpoint(
        complete_path,
        model=context.model,
        optimizer=context.optimizer,
        state=complete,
        permutation_generator=context.permutation_generator,
        config=context.config,
        sample_count=14_336,
        corpus_identity=context.corpus_identity,
        split_identity=context.split_identity,
        plan_identity=context.plan_identity,
        seed=context.seed,
        run_id=context.run_id,
        experiment_id=context.experiment_id,
        role="supervised_seed_complete",
        created_at_utc="2026-08-30T00:00:00Z",
        lineage=context.lineage,
        training_records_identity=context.training_records_identity,
        encoded_payload_identity=context.encoded_payload_identity,
    )

    target, target_state = _issue_test_session(
        production_corpus,
        run_id=context.run_id,
    )
    target_context = _session_context(target)
    initial_identity = canonical_training_state_identity(
        target_context.model.state_dict()
    )
    with pytest.raises(SupervisedTrainingError, match="latest|role"):
        load_supervised_checkpoint(complete_path, session=target)
    assert canonical_training_state_identity(target_context.model.state_dict()) == (
        initial_identity
    )
    save_supervised_checkpoint(
        tmp_path / "fresh-still-usable.pt",
        session=target,
        state=target_state,
        created_at_utc="2026-08-30T00:00:00Z",
    )


def test_seed_and_smoke_plan_purposes_are_bidirectionally_isolated(
    tmp_path: Path,
    production_corpus: FrozenCorpus,
) -> None:
    seed_binding = _issue_test_plan_binding(
        production_corpus,
        plan_identity="c" * 64,
        run_id="seed-purpose",
        experiment_id="classical-a-pos-supervised",
        seed=2026083001,
        purpose="seed",
    )
    with pytest.raises(SupervisedTrainingError, match="purpose"):
        supervised_module.issue_supervised_smoke_session(
            production_corpus,
            plan_binding=seed_binding,
        )

    smoke_binding = _issue_test_plan_binding(
        production_corpus,
        plan_identity="c" * 64,
        run_id="smoke-purpose",
        experiment_id="classical-a-pos-supervised",
        seed=2026083001,
        purpose="smoke",
    )
    with pytest.raises(SupervisedTrainingError, match="purpose"):
        issue_fresh_supervised_session(
            production_corpus,
            plan_binding=smoke_binding,
        )

    smoke_binding = _issue_test_plan_binding(
        production_corpus,
        plan_identity="c" * 64,
        run_id="smoke-session",
        experiment_id="classical-a-pos-supervised",
        seed=2026083001,
        purpose="smoke",
    )
    smoke_session = supervised_module.issue_supervised_smoke_session(
        production_corpus,
        plan_binding=smoke_binding,
    )
    fresh_state = SupervisedState(0, 0, 0, 0, None, False)
    with pytest.raises(SupervisedTrainingError, match="purpose|session kind"):
        run_supervised_updates(smoke_session, fresh_state, max_updates=1)
    with pytest.raises(SupervisedTrainingError, match="purpose|session kind"):
        save_supervised_checkpoint(
            tmp_path / "smoke-as-seed.pt",
            session=smoke_session,
            state=fresh_state,
            created_at_utc="2026-08-30T00:00:00Z",
        )

    seed_session, _seed_state = _issue_test_session(production_corpus)
    with pytest.raises(SupervisedTrainingError, match="purpose|session kind"):
        run_supervised_smoke_update(seed_session)


def test_smoke_disposable_checkpoint_requires_exactly_one_update_and_never_loads(
    tmp_path: Path,
    production_corpus: FrozenCorpus,
) -> None:
    binding = _issue_test_plan_binding(
        production_corpus,
        plan_identity="c" * 64,
        run_id="smoke-evidence",
        experiment_id="classical-a-pos-supervised",
        seed=2026083001,
        purpose="smoke",
    )
    session = supervised_module.issue_supervised_smoke_session(
        production_corpus,
        plan_binding=binding,
    )
    smoke_path = tmp_path / "smoke-disposable.pt"
    with pytest.raises(SupervisedTrainingError, match="not run|one update"):
        supervised_module.save_supervised_smoke_checkpoint(
            smoke_path,
            session=session,
            created_at_utc="2026-08-30T00:00:00Z",
        )

    result = run_supervised_smoke_update(session)
    assert result.optimizer_updates == 1
    with pytest.raises(SupervisedTrainingError, match="already run"):
        run_supervised_smoke_update(session)
    supervised_module.save_supervised_smoke_checkpoint(
        smoke_path,
        session=session,
        created_at_utc="2026-08-30T00:00:00Z",
    )
    envelope = load_checkpoint(smoke_path)
    assert envelope.descriptor.role == "supervised_smoke_disposable"
    cursor = envelope.payload.data_state["cursor"]
    assert set(cursor) == {
        "schema",
        "terminal",
        "resumable",
        "update_count",
        "optimizer_updates",
        "batch_start",
        "batch_stop",
        "permutation_calls",
        "rng_before_identity",
        "rng_after_identity",
    }
    assert cursor == {
        **cursor,
        "schema": "nmm.supervised-smoke-cursor.v1",
        "terminal": True,
        "resumable": False,
        "update_count": 1,
        "optimizer_updates": 1,
        "batch_start": 0,
        "batch_stop": 128,
        "permutation_calls": 0,
    }
    assert cursor["rng_before_identity"] == cursor["rng_after_identity"]
    with pytest.raises(SupervisedTrainingError, match="already saved"):
        supervised_module.save_supervised_smoke_checkpoint(
            tmp_path / "second-smoke-save.pt",
            session=session,
            created_at_utc="2026-08-30T00:00:00Z",
        )
    with pytest.raises(SupervisedTrainingError, match="latest|role|purpose"):
        _load_bound_checkpoint(smoke_path, production_corpus)

    resigned_as_latest = tmp_path / "smoke-resigned-as-seed-latest.pt"
    save_checkpoint(
        resigned_as_latest,
        replace(envelope.descriptor, role="supervised_seed_latest"),
        envelope.payload,
        previous_copies=0,
    )
    with pytest.raises(SupervisedTrainingError, match="lineage|purpose"):
        _load_bound_checkpoint(
            resigned_as_latest,
            production_corpus,
            run_id="smoke-evidence",
        )


def _write_midpoint_checkpoint(
    tmp_path: Path,
) -> tuple[Path, tuple[SupervisedExample, ...], SupervisedLoopConfig]:
    examples = _examples()
    config = SupervisedLoopConfig(
        batch_size=2,
        epochs=2,
        grad_clip_norm=1.0,
        permutation_seed=2026083001,
    )
    model, optimizer = _model_and_optimizer(19)
    generator = torch.Generator(device="cpu")
    state = create_fresh_supervised_state(
        permutation_generator=generator,
        config=config,
    )
    state = _run_supervised_updates(
        model,
        optimizer,
        examples,
        state,
        permutation_generator=generator,
        config=config,
        max_updates=1,
    )
    checkpoint = tmp_path / "midpoint.pt"
    _save_supervised_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        state=state,
        permutation_generator=generator,
        config=config,
        sample_count=len(examples),
        corpus_identity="a" * 64,
        split_identity="b" * 64,
        plan_identity="c" * 64,
        seed=2026083001,
        run_id="identity-run",
        experiment_id="identity-experiment",
        role="latest",
        created_at_utc="2026-08-30T00:00:00Z",
    )
    return checkpoint, examples, config


def _load_midpoint(
    checkpoint: Path,
    examples: tuple[SupervisedExample, ...],
    config: SupervisedLoopConfig,
    **overrides,
):
    model, optimizer = _model_and_optimizer(999)
    generator = torch.Generator(device="cpu")
    kwargs = {
        "corpus_identity": "a" * 64,
        "split_identity": "b" * 64,
        "plan_identity": "c" * 64,
        "expected_seed": 2026083001,
        "expected_run_id": "identity-run",
        "expected_experiment_id": "identity-experiment",
        "expected_role": "latest",
    }
    kwargs.update(overrides)
    return _load_supervised_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        permutation_generator=generator,
        config=config,
        sample_count=len(examples),
        **kwargs,
    )


def test_checkpoint_binds_run_experiment_plan_role_seed_and_assets(
    tmp_path: Path,
) -> None:
    checkpoint, examples, config = _write_midpoint_checkpoint(tmp_path)
    drifts = (
        {"corpus_identity": "d" * 64},
        {"split_identity": "d" * 64},
        {"plan_identity": "d" * 64},
        {"expected_run_id": "other-run"},
        {"expected_experiment_id": "other-experiment"},
        {"expected_role": "candidate"},
    )
    for drift in drifts:
        with pytest.raises(
            SupervisedTrainingError,
            match="checkpoint (identity|role)",
        ):
            _load_midpoint(checkpoint, examples, config, **drift)
    with pytest.raises(SupervisedTrainingError, match="resume seed"):
        _load_midpoint(
            checkpoint,
            examples,
            config,
            expected_seed=2026083002,
        )


def test_weights_only_and_resigned_duplicate_cursor_fail_closed(
    tmp_path: Path,
) -> None:
    checkpoint, examples, config = _write_midpoint_checkpoint(tmp_path)
    envelope = load_checkpoint(checkpoint)

    weights_only = tmp_path / "weights-only.pt"
    save_checkpoint(
        weights_only,
        envelope.descriptor,
        replace(envelope.payload, optimizer_state=None),
        previous_copies=0,
    )
    with pytest.raises(SupervisedTrainingError, match="lacks optimizer state"):
        _load_midpoint(weights_only, examples, config)

    wrong_optimizer = {
        "state": envelope.payload.optimizer_state["state"],
        "param_groups": [dict(envelope.payload.optimizer_state["param_groups"][0])],
    }
    wrong_optimizer["param_groups"][0]["lr"] = 2e-3
    wrong_optimizer_data = dict(envelope.payload.data_state)
    wrong_optimizer_cache = dict(wrong_optimizer_data["cache"])
    wrong_optimizer_identities = dict(
        wrong_optimizer_cache["canonical_state_identities"]
    )
    wrong_optimizer_identities["optimizer"] = canonical_training_state_identity(
        wrong_optimizer
    )
    wrong_optimizer_cache["canonical_state_identities"] = wrong_optimizer_identities
    wrong_optimizer_data["cache"] = wrong_optimizer_cache
    wrong_optimizer_path = tmp_path / "resigned-wrong-adam.pt"
    save_checkpoint(
        wrong_optimizer_path,
        envelope.descriptor,
        replace(
            envelope.payload,
            optimizer_state=wrong_optimizer,
            data_state=wrong_optimizer_data,
        ),
        previous_copies=0,
    )
    with pytest.raises(SupervisedTrainingError, match="Adam group semantics"):
        _load_midpoint(wrong_optimizer_path, examples, config)

    cursor = dict(envelope.payload.data_state["cursor"])
    cursor["update_count"] = 0
    trainer_state = dict(envelope.payload.trainer_state)
    trainer_state["batch_count"] = 0
    trainer_state["update_count"] = 0
    trainer_state["recovery_state"] = {
        "exact_resume": True,
        "cursor": cursor,
    }
    data_state = dict(envelope.payload.data_state)
    data_state["cursor"] = cursor
    cache = dict(data_state["cache"])
    identities = dict(cache["canonical_state_identities"])
    identities["cursor"] = canonical_training_state_identity(cursor)
    cache["canonical_state_identities"] = identities
    data_state["cache"] = cache
    resigned = tmp_path / "resigned-duplicate.pt"
    save_checkpoint(
        resigned,
        envelope.descriptor,
        replace(
            envelope.payload,
            trainer_state=trainer_state,
            data_state=data_state,
        ),
        previous_copies=0,
    )
    with pytest.raises(SupervisedTrainingError, match="cursor algebra"):
        _load_midpoint(resigned, examples, config)


def test_resigned_generator_and_permutation_must_match_canonical_replay(
    tmp_path: Path,
) -> None:
    checkpoint, examples, config = _write_midpoint_checkpoint(tmp_path)
    envelope = load_checkpoint(checkpoint)
    rogue_generator = torch.Generator(device="cpu")
    rogue_generator.manual_seed(config.permutation_seed + 999)
    rogue_permutation = torch.randperm(
        len(examples),
        generator=rogue_generator,
    ).tolist()

    cursor = dict(envelope.payload.data_state["cursor"])
    cursor["permutation"] = rogue_permutation
    trainer_state = dict(envelope.payload.trainer_state)
    trainer_state["recovery_state"] = {
        "exact_resume": True,
        "cursor": cursor,
    }
    data_state = dict(envelope.payload.data_state)
    data_state["cursor"] = cursor
    rng_state = dict(envelope.payload.rng_state)
    components = dict(rng_state["components"])
    components["permutation_generator"] = rogue_generator.get_state()
    rng_state["components"] = components
    cache = dict(data_state["cache"])
    identities = dict(cache["canonical_state_identities"])
    identities["cursor"] = canonical_training_state_identity(cursor)
    identities["rng"] = canonical_training_state_identity(rng_state)
    cache["canonical_state_identities"] = identities
    data_state["cache"] = cache
    resigned = tmp_path / "resigned-rogue-permutation-generator.pt"
    save_checkpoint(
        resigned,
        envelope.descriptor,
        replace(
            envelope.payload,
            rng_state=rng_state,
            trainer_state=trainer_state,
            data_state=data_state,
        ),
        previous_copies=0,
    )

    with pytest.raises(SupervisedTrainingError, match="deterministic generator replay"):
        _load_midpoint(resigned, examples, config)


def test_continuous_and_split_exact_resume_have_identical_semantic_state(
    tmp_path: Path,
) -> None:
    examples = _examples()
    config = SupervisedLoopConfig(
        batch_size=2,
        epochs=2,
        grad_clip_norm=1.0,
        permutation_seed=2026083001,
    )

    continuous_model, continuous_optimizer = _model_and_optimizer(73)
    continuous_generator = torch.Generator(device="cpu")
    continuous = create_fresh_supervised_state(
        permutation_generator=continuous_generator,
        config=config,
    )
    continuous_trace = []
    continuous = _run_supervised_updates(
        continuous_model,
        continuous_optimizer,
        examples,
        continuous,
        permutation_generator=continuous_generator,
        config=config,
        update_observer=continuous_trace.append,
    )
    continuous_path = tmp_path / "continuous.pt"
    _save_supervised_checkpoint(
        continuous_path,
        model=continuous_model,
        optimizer=continuous_optimizer,
        state=continuous,
        permutation_generator=continuous_generator,
        config=config,
        sample_count=len(examples),
        corpus_identity="a" * 64,
        split_identity="b" * 64,
        plan_identity="c" * 64,
        seed=2026083001,
        run_id="parity",
        experiment_id="synthetic-parity",
        role="latest",
        created_at_utc="2026-08-30T00:00:00Z",
    )

    split_model, split_optimizer = _model_and_optimizer(73)
    split_generator = torch.Generator(device="cpu")
    split = create_fresh_supervised_state(
        permutation_generator=split_generator,
        config=config,
    )
    split_trace = []
    split = _run_supervised_updates(
        split_model,
        split_optimizer,
        examples,
        split,
        permutation_generator=split_generator,
        config=config,
        max_updates=1,
        update_observer=split_trace.append,
    )
    midpoint = tmp_path / "midpoint.pt"
    _save_supervised_checkpoint(
        midpoint,
        model=split_model,
        optimizer=split_optimizer,
        state=split,
        permutation_generator=split_generator,
        config=config,
        sample_count=len(examples),
        corpus_identity="a" * 64,
        split_identity="b" * 64,
        plan_identity="c" * 64,
        seed=2026083001,
        run_id="parity-midpoint",
        experiment_id="synthetic-parity",
        role="latest",
        created_at_utc="2026-08-30T00:00:00Z",
    )

    resumed_model, resumed_optimizer = _model_and_optimizer(999)
    resumed_generator = torch.Generator(device="cpu")
    resumed = _load_supervised_checkpoint(
        midpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        permutation_generator=resumed_generator,
        config=config,
        sample_count=len(examples),
        corpus_identity="a" * 64,
        split_identity="b" * 64,
        plan_identity="c" * 64,
        expected_seed=2026083001,
        expected_run_id="parity-midpoint",
        expected_experiment_id="synthetic-parity",
        expected_role="latest",
    )
    assert resumed == split
    assert torch.equal(split_generator.get_state(), resumed_generator.get_state())
    assert canonical_training_state_identity(split_model.state_dict()) == (
        canonical_training_state_identity(resumed_model.state_dict())
    )
    assert canonical_training_state_identity(split_optimizer.state_dict()) == (
        canonical_training_state_identity(resumed_optimizer.state_dict())
    )
    for example in examples:
        assert torch.equal(
            split_model.policy_logits(example.features),
            resumed_model.policy_logits(example.features),
        )
    assert split.permutation is not None
    next_batch = split.permutation[
        split.sample_cursor : split.sample_cursor + config.batch_size
    ]
    assert next_batch == continuous_trace[1].batch_indices
    resumed_trace = []
    resumed = _run_supervised_updates(
        resumed_model,
        resumed_optimizer,
        examples,
        resumed,
        permutation_generator=resumed_generator,
        config=config,
        update_observer=resumed_trace.append,
    )
    split_path = tmp_path / "split.pt"
    _save_supervised_checkpoint(
        split_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        state=resumed,
        permutation_generator=resumed_generator,
        config=config,
        sample_count=len(examples),
        corpus_identity="a" * 64,
        split_identity="b" * 64,
        plan_identity="c" * 64,
        seed=2026083001,
        run_id="parity",
        experiment_id="synthetic-parity",
        role="latest",
        created_at_utc="2026-08-30T00:00:00Z",
    )

    left = load_checkpoint(continuous_path)
    right = load_checkpoint(split_path)
    assert left.descriptor == right.descriptor
    assert left.payload_sha256 != ""
    assert right.payload_sha256 != ""
    assert canonical_training_state_identity(left.payload.to_dict()) == (
        canonical_training_state_identity(right.payload.to_dict())
    )
    assert continuous_trace == split_trace + resumed_trace
    assert continuous == resumed
    assert continuous.update_count == 4
    assert continuous.permutation is None
    for continuous_example, resumed_example in zip(
        examples,
        examples,
        strict=True,
    ):
        assert torch.equal(
            continuous_model.policy_logits(continuous_example.features),
            resumed_model.policy_logits(resumed_example.features),
        )
