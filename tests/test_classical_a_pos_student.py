from __future__ import annotations

import gc
import inspect
import pickle
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from game.board import BoardState
from game.rules import get_all_legal_moves
import learned_ai.agents.classical_a_pos_student as student_module
from learned_ai.agents.classical_a_pos_student import (
    ClassicalAPosRouteOutcome,
    ClassicalAPosStudentRoute,
    ExactD9PostGateFactory,
    ExactD9RouteImplementationBinding,
    FreshExactD9PostGateAttempt,
    IssuedExactD9Search,
    ResearchRouteUnavailable,
    _TestClassicalAPosStudentRoute,
    _issue_test_exact_d9_post_gate_factory,
    _issue_test_exact_d9_route_implementation_binding,
    _issue_test_qualified_complete_seed_binding,
    _make_test_classical_a_pos_route,
    load_qualified_complete_student,
)
from learned_ai.agents.positional_safety import (
    ProductPositionalSafetyGate,
    ProductSafetyOutcome,
    legal_inventory_identity,
)
import learned_ai.training.classical_a_pos_supervised as supervised_module
from learned_ai.training.classical_a_pos_supervised import (
    SEED_COMPLETE_ROLE,
    SupervisedPurpose,
    SupervisedState,
    _create_fresh_production_student,
    _save_supervised_checkpoint,
    create_fresh_supervised_state,
    production_loop_config,
)


SOURCE = "generalist-classical-d9-distilled-v1"
SEED = 2026083001
EVIDENCE_IDENTITY_FIELDS = (
    "attempt_identity",
    "factory_identity",
    "search_instance_identity",
    "search_initial_state_identity",
    "search_invocation_identity",
    "route_contract_identity",
    "route_implementation_identity",
    "route_effective_config_identity",
    "gate_runtime_identity",
    "gate_implementation_identity",
    "legal_inventory_identity",
)


class _Model:
    def __init__(self, result) -> None:
        self.result = result
        self.inference_modes: list[bool] = []

    def policy_logits(self, _features: torch.Tensor):
        self.inference_modes.append(torch.is_inference_mode_enabled())
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _encoder(board, player, **kwargs):
    assert player == board.turn
    assert kwargs == {
        "sentinel_advisor": None,
        "db": None,
        "value_net": None,
        "specialist_db": None,
        "wdl_db": None,
        "strict": True,
    }
    legal = get_all_legal_moves(board)
    return SimpleNamespace(
        feat_matrix=np.zeros((len(legal), 62), dtype=np.float32),
        legal_moves=legal,
    )


def _clean_decision(
    board: BoardState,
    selected_move: dict,
    *,
    original_move: dict | None = None,
    rule: str = "original-already-in-A_pos",
    candidate_order_verified: bool,
    source: str = SOURCE,
) -> dict:
    legal = get_all_legal_moves(board)
    original = selected_move if original_move is None else original_move
    return {
        "status": "applied",
        "source": source,
        "difficulty": 9,
        "original_move": dict(original),
        "selected_move": dict(selected_move),
        "selection_rule": rule,
        "selection_error": None,
        "mode": "A_pos",
        "positional_only": True,
        "history_aware": False,
        "candidate_order_verified": candidate_order_verified,
        "legal_inventory_identity": legal_inventory_identity(legal),
        "parent_tier": "W",
        "selected_tier": "W",
        "legal_move_count": len(legal),
        "safe_move_count": len(legal),
    }


class _Gate:
    def __init__(
        self,
        *,
        decision_updates: dict | None = None,
        selected_index: int | None = None,
        outcome_move: dict | None = None,
        error: Exception | None = None,
        fallback_decision_updates: dict | None = None,
        fallback_outcome: ProductSafetyOutcome | None = None,
    ) -> None:
        self.decision_updates = dict(decision_updates or {})
        self.selected_index = selected_index
        self.outcome_move = outcome_move
        self.error = error
        self.fallback_decision_updates = dict(fallback_decision_updates or {})
        self.fallback_outcome = fallback_outcome
        self.calls: list[dict] = []

    def constrain(
        self,
        board,
        original_move,
        *,
        source,
        difficulty,
        candidate_moves,
        candidate_scores,
        safe_selector,
        query_failure_move,
    ) -> ProductSafetyOutcome:
        is_student = candidate_moves is not None
        if self.error is not None and is_student:
            raise self.error
        supplied = (
            [dict(move) for move in candidate_moves]
            if candidate_moves is not None
            else [dict(move) for move in get_all_legal_moves(board)]
        )
        self.calls.append(
            {
                "original_move": dict(original_move),
                "source": source,
                "difficulty": difficulty,
                "candidate_moves": supplied,
                "candidate_scores": (
                    None if candidate_scores is None else list(candidate_scores)
                ),
                "safe_selector": safe_selector,
                "query_failure_move": dict(query_failure_move),
            }
        )
        if not is_student and self.fallback_outcome is not None:
            return self.fallback_outcome
        if not is_student and safe_selector is not None:
            selected = dict(safe_selector(supplied))
            selection_rule = "restricted-root-research"
        else:
            selected = (
                dict(original_move)
                if self.selected_index is None or not is_student
                else supplied[self.selected_index]
            )
            selection_rule = (
                "original-already-in-A_pos"
                if self.selected_index is None or not is_student
                else "model-argmax-inside-A_pos"
            )
        decision = _clean_decision(
            board,
            selected,
            original_move=dict(original_move),
            rule=selection_rule,
            candidate_order_verified=True,
            source=source,
        )
        decision.update(
            self.decision_updates if is_student else self.fallback_decision_updates
        )
        return ProductSafetyOutcome(
            dict((self.outcome_move if is_student else None) or selected),
            decision,
        )


def _fallback_outcome(
    board: BoardState,
    *,
    move_index: int = 0,
    decision_updates: dict | None = None,
    outcome_move: dict | None = None,
) -> ProductSafetyOutcome:
    legal = get_all_legal_moves(board)
    selected = legal[move_index]
    decision = _clean_decision(
        board,
        selected,
        rule="original-already-in-A_pos",
        candidate_order_verified=False,
        source="classical-d9-exact-post-gate",
    )
    decision.update(decision_updates or {})
    return ProductSafetyOutcome(dict(outcome_move or selected), decision)


class _TestSearch:
    def __init__(
        self,
        choose_raw,
        *,
        freshness_root=None,
        initial_state: dict | None = None,
    ) -> None:
        self.choose_raw = choose_raw
        self.freshness_root = self if freshness_root is None else freshness_root
        self.initial_state = dict(
            initial_state
            or {
                "transposition_table_entries": 0,
                "history_entries": 0,
                "cache_entries": 0,
                "search_invocations": 0,
            }
        )


def _default_raw_move(board: BoardState) -> dict:
    return dict(get_all_legal_moves(board)[0])


def _factory(
    gate,
    *,
    search_constructor=None,
    raw_choose=None,
    freshness_root_accessor=None,
    fresh_state_attestor=None,
    restricted_selector=None,
    difficulty: int = 9,
    evidence_overrides: dict | None = None,
    reuse_search_handle: bool = False,
) -> ExactD9PostGateFactory:
    resolved_raw_choose = raw_choose or _default_raw_move
    if search_constructor is None:

        def search_constructor():
            return _TestSearch(resolved_raw_choose)

    resolved_root_accessor = freshness_root_accessor or (
        lambda search: search.freshness_root
    )
    resolved_attestor = fresh_state_attestor or (
        lambda search, _root: dict(search.initial_state)
    )
    implementation = _issue_test_exact_d9_route_implementation_binding(
        gate=gate,
        search_constructor=search_constructor,
        freshness_root_accessor=resolved_root_accessor,
        fresh_state_attestor=resolved_attestor,
        choose_raw=lambda search, board: search.choose_raw(board),
        restricted_selector=restricted_selector,
        difficulty=difficulty,
    )
    return _issue_test_exact_d9_post_gate_factory(
        implementation=implementation,
        evidence_overrides=evidence_overrides,
        reuse_search_handle=reuse_search_handle,
    )


def _route(
    *,
    model_result=None,
    model: _Model | None = None,
    encoder=_encoder,
    gate: _Gate | None = None,
    factory: ExactD9PostGateFactory | None = None,
) -> tuple[_TestClassicalAPosStudentRoute, _Gate, ExactD9PostGateFactory]:
    board = BoardState.new_game()
    legal = get_all_legal_moves(board)
    resolved_model = model or _Model(
        torch.zeros(len(legal), dtype=torch.float32)
        if model_result is None
        else model_result
    )
    resolved_gate = gate or _Gate()
    resolved_factory = factory or _factory(resolved_gate)
    route = _make_test_classical_a_pos_route(
        model=resolved_model,
        encoder=encoder,
        gate=resolved_gate,
        exact_d9_factory=resolved_factory,
    )
    return route, resolved_gate, resolved_factory


def _write_complete_checkpoint(path):
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
    return _save_supervised_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        state=SupervisedState(20, 0, 2240, 0, None, True),
        permutation_generator=generator,
        config=config,
        sample_count=14_336,
        corpus_identity="a" * 64,
        split_identity="b" * 64,
        plan_identity="c" * 64,
        seed=SEED,
        run_id="qualified-route-seed-2026083001",
        experiment_id="classical-a-pos-supervised",
        role=SEED_COMPLETE_ROLE,
        created_at_utc="2026-08-31T00:00:00Z",
        lineage=supervised_module._FRESH_MODEL_LINEAGES[model],
        training_records_identity="d" * 64,
        encoded_payload_identity="e" * 64,
    )


def test_internal_test_factory_cannot_upgrade_into_production_route(
    tmp_path,
) -> None:
    path = _write_complete_checkpoint(tmp_path / "complete.pt")
    binding = _issue_test_qualified_complete_seed_binding(path)
    handle = load_qualified_complete_student(path, qualification=binding)
    gate = ProductPositionalSafetyGate()
    factory = _factory(gate)

    with pytest.raises(ResearchRouteUnavailable, match="issuer scope"):
        ClassicalAPosStudentRoute.from_qualified(
            student=handle,
            gate=gate,
            exact_d9_factory=factory,
        )
    with pytest.raises(ResearchRouteUnavailable, match="consumed|already bound"):
        ClassicalAPosStudentRoute.from_qualified(
            student=handle,
            gate=gate,
            exact_d9_factory=_factory(gate),
        )


def test_production_rejects_instance_shadowed_real_gate_before_binding(
    tmp_path,
) -> None:
    path = _write_complete_checkpoint(tmp_path / "complete-shadowed-gate.pt")
    binding = _issue_test_qualified_complete_seed_binding(path)
    handle = load_qualified_complete_student(path, qualification=binding)
    gate = ProductPositionalSafetyGate()
    factory = _factory(gate)
    gate.constrain = lambda *_args, **_kwargs: None

    with pytest.raises(ResearchRouteUnavailable, match="gate.*method|constrain"):
        ClassicalAPosStudentRoute.from_qualified(
            student=handle,
            gate=gate,
            exact_d9_factory=factory,
        )


def test_production_rechecks_real_gate_method_and_uses_captured_unbound_method() -> (
    None
):
    gate = ProductPositionalSafetyGate()
    factory = _factory(gate)
    route = _make_test_classical_a_pos_route(
        model=_Model(torch.zeros(24)),
        encoder=_encoder,
        gate=gate,
        exact_d9_factory=factory,
    )
    forged_calls: list[bool] = []

    def forged_constrain(board, original_move, **_kwargs):
        forged_calls.append(True)
        return ProductSafetyOutcome(
            dict(original_move),
            _clean_decision(
                board,
                dict(original_move),
                candidate_order_verified=True,
            ),
        )

    gate.constrain = forged_constrain
    with pytest.raises(ResearchRouteUnavailable, match="gate.*shadow|constrain"):
        route.choose_move(BoardState.new_game())
    assert forged_calls == []


def test_raw_models_and_test_routes_cannot_enter_production_constructor() -> None:
    gate = ProductPositionalSafetyGate()
    factory = _factory(gate)
    with pytest.raises((TypeError, ResearchRouteUnavailable)):
        ClassicalAPosStudentRoute(
            model=_Model(torch.zeros(24)),
            encoder=_encoder,
            gate=gate,
            exact_d9_factory=factory,
        )
    with pytest.raises(ResearchRouteUnavailable, match="qualified"):
        ClassicalAPosStudentRoute.from_qualified(
            student=object(),
            gate=gate,
            exact_d9_factory=factory,
        )
    test_route, _, _ = _route()
    assert type(test_route) is _TestClassicalAPosStudentRoute
    with pytest.raises(ResearchRouteUnavailable, match="qualified"):
        ClassicalAPosStudentRoute.from_qualified(
            student=test_route,
            gate=gate,
            exact_d9_factory=factory,
        )
    assert set(
        inspect.signature(ClassicalAPosStudentRoute.from_qualified).parameters
    ) == {
        "student",
        "gate",
        "exact_d9_factory",
    }


@pytest.mark.parametrize("mutation", ["method", "config-type"])
def test_qualified_model_runtime_cannot_be_mutated_before_route_binding(
    tmp_path,
    mutation: str,
) -> None:
    path = _write_complete_checkpoint(tmp_path / f"complete-{mutation}.pt")
    binding = _issue_test_qualified_complete_seed_binding(path)
    handle = load_qualified_complete_student(path, qualification=binding)
    model = student_module._QUALIFIED_STUDENT_CONTEXTS[handle].model
    if mutation == "method":
        model.policy_logits = lambda _features: torch.zeros(24)
    else:
        model.move_feat_dim = 62.0
    gate = ProductPositionalSafetyGate()
    with pytest.raises(ResearchRouteUnavailable, match="model"):
        ClassicalAPosStudentRoute.from_qualified(
            student=handle,
            gate=gate,
            exact_d9_factory=_factory(gate),
        )


@pytest.mark.parametrize("failure", ["raw-factory", "wrong-gate"])
def test_failed_production_binding_consumes_the_qualified_handle_once(
    tmp_path,
    failure: str,
) -> None:
    path = _write_complete_checkpoint(tmp_path / f"complete-consumed-{failure}.pt")
    binding = _issue_test_qualified_complete_seed_binding(path)
    handle = load_qualified_complete_student(path, qualification=binding)
    gate = ProductPositionalSafetyGate()
    wrong_gate = ProductPositionalSafetyGate()

    invalid_factory = object() if failure == "raw-factory" else _factory(wrong_gate)
    with pytest.raises(
        ResearchRouteUnavailable,
        match="issued exact D9|different.*gate|issuer scope",
    ):
        ClassicalAPosStudentRoute.from_qualified(
            student=handle,
            gate=gate,
            exact_d9_factory=invalid_factory,
        )
    with pytest.raises(ResearchRouteUnavailable, match="consumed|already bound"):
        ClassicalAPosStudentRoute.from_qualified(
            student=handle,
            gate=gate,
            exact_d9_factory=_factory(gate),
        )


def test_student_passes_complete_order_and_stable_softmax_to_gate() -> None:
    board = BoardState.new_game()
    legal = get_all_legal_moves(board)
    model = _Model(torch.arange(len(legal), dtype=torch.float64))
    route, gate, factory = _route(model=model)

    outcome = route.choose_move(board)

    assert type(outcome) is ClassicalAPosRouteOutcome
    assert outcome.route == "student"
    assert outcome.move == legal[-1]
    assert outcome.fallback_reason is None
    with pytest.raises(TypeError):
        outcome.move["to"] = "tampered"
    with pytest.raises(TypeError):
        outcome.safety_decision["status"] = "tampered"
    assert model.inference_modes == [True]
    assert student_module._EXACT_D9_FACTORY_CONTEXTS[factory].create_count == 0
    call = gate.calls[0]
    assert call["candidate_moves"] == legal
    assert call["original_move"] == legal[-1]
    assert call["query_failure_move"] == legal[-1]
    assert call["source"] == SOURCE
    assert call["difficulty"] == 9
    assert call["safe_selector"] is None
    assert all(np.isfinite(call["candidate_scores"]))
    assert all(value >= 0.0 for value in call["candidate_scores"])
    assert abs(sum(call["candidate_scores"]) - 1.0) <= 1e-12


def test_extreme_logits_are_stable_and_first_argmax_wins_ties() -> None:
    board = BoardState.new_game()
    legal = get_all_legal_moves(board)
    logits = torch.full((len(legal),), -1e300, dtype=torch.float64)
    logits[0] = 1e300
    logits[1] = 1e300
    route, gate, _ = _route(model_result=logits)

    outcome = route.choose_move(board)

    assert outcome.route == "student"
    assert outcome.move == legal[0]
    assert gate.calls[0]["candidate_scores"][:2] == [0.5, 0.5]


@pytest.mark.parametrize("mutation", ["duplicate", "missing-key"])
def test_invalid_primary_legal_inventory_fails_closed_before_any_model_or_factory(
    monkeypatch,
    mutation: str,
) -> None:
    legal = get_all_legal_moves(BoardState.new_game())
    invalid = [dict(move) for move in legal]
    if mutation == "duplicate":
        invalid[1] = dict(invalid[0])
    else:
        del invalid[0]["capture"]
    monkeypatch.setattr(
        student_module,
        "get_all_legal_moves",
        lambda _board: invalid,
    )
    model = _Model(torch.zeros(len(legal)))
    route, _, factory = _route(model=model)

    with pytest.raises(ResearchRouteUnavailable, match="legal inventory"):
        route.choose_move(BoardState.new_game())
    assert model.inference_modes == []
    assert student_module._EXACT_D9_FACTORY_CONTEXTS[factory].create_count == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "dtype",
        "61-features",
        "63-features",
        "short-rows",
        "nan",
        "inf",
        "reversed-actions",
        "short-actions",
        "duplicate-actions",
    ],
)
def test_every_encoder_contract_failure_uses_exact_fallback(mutation: str) -> None:
    board = BoardState.new_game()

    def broken_encoder(board, player, **kwargs):
        encoded = _encoder(board, player, **kwargs)
        if mutation == "none":
            return None
        if mutation == "dtype":
            encoded.feat_matrix = encoded.feat_matrix.astype(np.float64)
        elif mutation == "61-features":
            encoded.feat_matrix = encoded.feat_matrix[:, :61]
        elif mutation == "63-features":
            encoded.feat_matrix = np.pad(encoded.feat_matrix, ((0, 0), (0, 1)))
        elif mutation == "short-rows":
            encoded.feat_matrix = encoded.feat_matrix[:-1]
        elif mutation == "nan":
            encoded.feat_matrix[0, 0] = np.nan
        elif mutation == "inf":
            encoded.feat_matrix[0, 0] = np.inf
        elif mutation == "reversed-actions":
            encoded.legal_moves = list(reversed(encoded.legal_moves))
        elif mutation == "short-actions":
            encoded.legal_moves = encoded.legal_moves[:-1]
        elif mutation == "duplicate-actions":
            encoded.legal_moves[1] = dict(encoded.legal_moves[0])
        return encoded

    route, gate, factory = _route(encoder=broken_encoder)
    outcome = route.choose_move(board)

    assert outcome.route == "exact-d9-post-gate-fallback"
    assert outcome.fallback_reason
    assert len(gate.calls) == 1
    assert gate.calls[0]["source"] == "classical-d9-exact-post-gate"
    assert gate.calls[0]["candidate_scores"] is None
    assert student_module._EXACT_D9_FACTORY_CONTEXTS[factory].create_count == 1


@pytest.mark.parametrize(
    "result",
    [
        0.0,
        [0.0] * 24,
        torch.tensor(0.0),
        torch.zeros((24, 1)),
        torch.zeros(23),
        torch.zeros(25),
        torch.zeros(24, dtype=torch.int64),
        torch.full((24,), float("nan")),
        torch.full((24,), float("inf")),
        RuntimeError("model exploded"),
    ],
)
def test_every_logit_contract_failure_uses_exact_fallback(result) -> None:
    route, gate, factory = _route(model_result=result)

    outcome = route.choose_move(BoardState.new_game())

    assert outcome.route == "exact-d9-post-gate-fallback"
    assert outcome.fallback_reason
    assert len(gate.calls) == 1
    assert gate.calls[0]["source"] == "classical-d9-exact-post-gate"
    assert gate.calls[0]["candidate_scores"] is None
    assert student_module._EXACT_D9_FACTORY_CONTEXTS[factory].create_count == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"status": "unfiltered-query-failure"},
        {"status": "bypassed-low-difficulty"},
        {"selection_error": "rerank failed"},
        {"selection_rule": "canonical-safe-fallback"},
        {"selection_rule": "restricted-root-research"},
        {"selection_rule": "unknown-rule"},
        {"mode": "full-history"},
        {"positional_only": False},
        {"history_aware": True},
        {"candidate_order_verified": False},
        {"legal_inventory_identity": "f" * 64},
        {"parent_tier": "X"},
        {"selected_tier": "D"},
        {"failure": "oracle failed"},
        {"degradation": True},
        {"degraded": False},
        {"alternative": "canonical"},
        {"alternate": False},
        {"fallback": False},
        {"used_fallback": False},
    ],
)
def test_any_student_gate_degradation_or_forgery_uses_fallback(updates: dict) -> None:
    gate = _Gate(decision_updates=updates)
    route, _, factory = _route(gate=gate)

    outcome = route.choose_move(BoardState.new_game())

    assert outcome.route == "exact-d9-post-gate-fallback"
    assert student_module._EXACT_D9_FACTORY_CONTEXTS[factory].create_count == 1


def test_gate_exception_illegal_move_and_decision_outcome_mismatch_use_fallback() -> (
    None
):
    board = BoardState.new_game()
    legal = get_all_legal_moves(board)
    gates = [
        _Gate(error=RuntimeError("gate exploded")),
        _Gate(outcome_move={"from": None, "to": "not-a-square", "capture": None}),
        _Gate(
            outcome_move=legal[1],
            decision_updates={"selected_move": legal[2]},
        ),
    ]
    for gate in gates:
        route, _, factory = _route(gate=gate)
        outcome = route.choose_move(board)
        assert outcome.route == "exact-d9-post-gate-fallback"
        assert student_module._EXACT_D9_FACTORY_CONTEXTS[factory].create_count == 1


def test_clean_restricted_root_exact_fallback_is_accepted() -> None:
    board = BoardState.new_game()
    legal = get_all_legal_moves(board)
    gate = _Gate(error=RuntimeError("force exact fallback"))
    factory = _factory(
        gate,
        restricted_selector=lambda _search, _board, safe: dict(safe[3]),
    )
    route, _, _ = _route(gate=gate, factory=factory)

    outcome = route.choose_move(board)

    assert outcome.route == "exact-d9-post-gate-fallback"
    assert outcome.move == legal[3]
    assert "gate" in outcome.fallback_reason
    assert student_module._EXACT_D9_FACTORY_CONTEXTS[factory].create_count == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"status": "unfiltered-query-failure"},
        {"selection_error": "selection failed"},
        {"selection_rule": "canonical-safe-fallback"},
        {"selection_rule": "model-argmax-inside-A_pos"},
        {"mode": "wrong"},
        {"positional_only": False},
        {"history_aware": True},
        {"legal_inventory_identity": "f" * 64},
        {"parent_tier": "D", "selected_tier": "W"},
        {"source": "forged-classical-route"},
        {"failure": "query failed"},
        {"fallback": True},
    ],
)
def test_invalid_fallback_gate_evidence_makes_route_unavailable(updates: dict) -> None:
    gate = _Gate(
        error=RuntimeError("force fallback"),
        fallback_decision_updates=updates,
    )
    route, _, _ = _route(gate=gate, factory=_factory(gate))
    with pytest.raises(ResearchRouteUnavailable, match="exact D9|fallback"):
        route.choose_move(BoardState.new_game())


def test_fallback_gate_must_bind_original_to_raw_exact_d9_move() -> None:
    board = BoardState.new_game()
    legal = get_all_legal_moves(board)
    gate = _Gate(
        error=RuntimeError("force fallback"),
        fallback_decision_updates={"original_move": legal[1]},
    )
    route, _, _ = _route(
        gate=gate,
        factory=_factory(gate, raw_choose=lambda _board: dict(legal[0])),
    )

    with pytest.raises(ResearchRouteUnavailable, match="gate evidence"):
        route.choose_move(board)


@pytest.mark.parametrize(
    "overrides",
    [
        {"difficulty": 8},
        *[{field: "f" * 64} for field in EVIDENCE_IDENTITY_FIELDS],
        *[{field: None} for field in EVIDENCE_IDENTITY_FIELDS],
    ],
)
def test_invalid_exact_attempt_evidence_makes_route_unavailable(
    overrides: dict,
) -> None:
    gate = _Gate(error=RuntimeError("force fallback"))
    factory = _factory(gate, evidence_overrides=overrides)
    route, _, _ = _route(gate=gate, factory=factory)
    with pytest.raises(ResearchRouteUnavailable, match="exact D9|fallback"):
        route.choose_move(BoardState.new_game())


def test_factory_and_attempt_are_opaque_single_use_capabilities() -> None:
    gate = _Gate()
    factory = _factory(gate)
    assert not hasattr(factory, "__dict__")
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(factory)
    implementation = student_module._EXACT_D9_FACTORY_CONTEXTS[factory].implementation
    assert type(implementation) is ExactD9RouteImplementationBinding
    assert not hasattr(implementation, "__dict__")
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(implementation)
    issued_search = factory.create()
    assert type(issued_search) is IssuedExactD9Search
    assert not hasattr(issued_search, "__dict__")
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(issued_search)
    attempt = issued_search.issue_attempt()
    assert type(attempt) is FreshExactD9PostGateAttempt
    assert not hasattr(attempt, "__dict__")
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(attempt)
    board = BoardState.new_game()
    inventory = legal_inventory_identity(get_all_legal_moves(board))
    evidence = attempt.choose_once(
        board,
        expected_legal_inventory_identity=inventory,
    )
    assert {
        "attempt_identity",
        "factory_identity",
        "search_instance_identity",
        "search_initial_state_identity",
        "search_invocation_identity",
        "route_contract_identity",
        "route_implementation_identity",
        "route_effective_config_identity",
        "gate_runtime_identity",
        "gate_implementation_identity",
    } <= {field.name for field in fields(type(evidence))}
    with pytest.raises(ResearchRouteUnavailable, match="consumed"):
        issued_search.issue_attempt()
    with pytest.raises(ResearchRouteUnavailable, match="consumed"):
        attempt.choose_once(
            board,
            expected_legal_inventory_identity=inventory,
        )


def test_failed_search_handle_issue_is_consumed_and_cannot_retry() -> None:
    factory = _factory(_Gate())
    issued_search = factory.create()
    search_context = student_module._ISSUED_EXACT_D9_SEARCH_CONTEXTS[issued_search]
    search_context.search.initial_state["cache_entries"] = 1

    with pytest.raises(ResearchRouteUnavailable, match="initial state|empty"):
        issued_search.issue_attempt()
    search_context.search.initial_state["cache_entries"] = 0
    with pytest.raises(ResearchRouteUnavailable, match="consumed"):
        issued_search.issue_attempt()


def test_factory_reusing_attempt_is_detected_across_fallbacks() -> None:
    gate = _Gate(error=RuntimeError("force fallback"))
    factory = _factory(gate, reuse_search_handle=True)
    route, _, _ = _route(gate=gate, factory=factory)

    first = route.choose_move(BoardState.new_game())
    assert first.route == "exact-d9-post-gate-fallback"
    with pytest.raises(ResearchRouteUnavailable, match="fresh|consumed|reuse"):
        route.choose_move(BoardState.new_game())


def test_factory_reusing_same_outcome_across_fresh_attempts_is_detected() -> None:
    board = BoardState.new_game()
    shared_outcome = _fallback_outcome(board)
    gate = _Gate(
        error=RuntimeError("force fallback"),
        fallback_outcome=shared_outcome,
    )
    route, _, _ = _route(gate=gate, factory=_factory(gate))
    assert route.choose_move(board).route == "exact-d9-post-gate-fallback"
    with pytest.raises(ResearchRouteUnavailable, match="reuse|result|evidence"):
        route.choose_move(board)


def test_new_wrappers_cannot_hide_one_reused_stateful_search_instance() -> None:
    class StatefulChooser:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, board):
            self.calls += 1
            return dict(get_all_legal_moves(board)[0])

    shared_search = StatefulChooser()
    gate = _Gate(error=RuntimeError("force fallback"))

    def search_constructor():
        return _TestSearch(
            lambda board: shared_search(board),
            freshness_root=shared_search,
        )

    route, _, _ = _route(
        gate=gate,
        factory=_factory(gate, search_constructor=search_constructor),
    )
    assert route.choose_move(BoardState.new_game()).route == (
        "exact-d9-post-gate-fallback"
    )
    with pytest.raises(ResearchRouteUnavailable, match="fresh|reuse|search"):
        route.choose_move(BoardState.new_game())
    assert shared_search.calls == 1


def test_exact_search_object_cannot_be_returned_twice_with_distinct_attempts() -> None:
    shared_search = _TestSearch(_default_raw_move)
    gate = _Gate(error=RuntimeError("force fallback"))
    route, _, _ = _route(
        gate=gate,
        factory=_factory(
            gate,
            search_constructor=lambda: shared_search,
        ),
    )

    assert route.choose_move(BoardState.new_game()).route == (
        "exact-d9-post-gate-fallback"
    )
    with pytest.raises(ResearchRouteUnavailable, match="reused.*search"):
        route.choose_move(BoardState.new_game())


def test_search_and_root_reuse_is_rejected_across_distinct_factories() -> None:
    shared_root = object()
    shared_search = _TestSearch(
        _default_raw_move,
        freshness_root=shared_root,
    )
    gate = _Gate()
    first = _factory(
        gate,
        search_constructor=lambda: shared_search,
    )
    second = _factory(
        gate,
        search_constructor=lambda: shared_search,
    )

    first_search = first.create()
    assert type(first_search) is IssuedExactD9Search
    with pytest.raises(ResearchRouteUnavailable, match="across factories"):
        second.create()


def test_nonempty_search_state_fails_before_raw_search_or_gate() -> None:
    raw_calls: list[bool] = []

    def raw_choose(board):
        raw_calls.append(True)
        return _default_raw_move(board)

    initial_state = {
        "transposition_table_entries": 1,
        "history_entries": 0,
        "cache_entries": 0,
        "search_invocations": 0,
    }
    gate = _Gate(error=RuntimeError("force fallback"))
    route, _, _ = _route(
        gate=gate,
        factory=_factory(
            gate,
            search_constructor=lambda: _TestSearch(
                raw_choose,
                initial_state=initial_state,
            ),
        ),
    )

    with pytest.raises(ResearchRouteUnavailable, match="initial state|empty"):
        route.choose_move(BoardState.new_game())
    assert raw_calls == []
    assert gate.calls == []


def test_committed_route_method_drift_fails_before_search_construction() -> None:
    constructor_calls: list[bool] = []

    class Constructor:
        def __call__(self):
            constructor_calls.append(True)
            return _TestSearch(_default_raw_move)

    constructor = Constructor()
    gate = _Gate(error=RuntimeError("force fallback"))
    implementation = _issue_test_exact_d9_route_implementation_binding(
        gate=gate,
        search_constructor=constructor,
        freshness_root_accessor=lambda search: search.freshness_root,
        fresh_state_attestor=lambda search, _root: dict(search.initial_state),
        choose_raw=lambda search, board: search.choose_raw(board),
    )
    factory = _issue_test_exact_d9_post_gate_factory(
        implementation=implementation,
    )
    route, _, _ = _route(gate=gate, factory=factory)

    def drifted_constructor(self):
        del self
        raise AssertionError("drifted constructor must not run")

    Constructor.__call__ = drifted_constructor
    with pytest.raises(
        ResearchRouteUnavailable,
        match="implementation drifted|method drifted",
    ):
        route.choose_move(BoardState.new_game())
    assert constructor_calls == []


@pytest.mark.parametrize(
    ("owner", "method"),
    [
        (ExactD9PostGateFactory, "create"),
        (IssuedExactD9Search, "issue_attempt"),
        (FreshExactD9PostGateAttempt, "choose_once"),
    ],
)
def test_capability_class_method_drift_is_never_invoked(
    monkeypatch,
    owner,
    method: str,
) -> None:
    forged_calls: list[bool] = []
    gate = _Gate(error=RuntimeError("force fallback"))
    route, _, _ = _route(gate=gate)

    def forged(*_args, **_kwargs):
        forged_calls.append(True)
        raise AssertionError("drifted capability method must not run")

    monkeypatch.setattr(owner, method, forged)
    with pytest.raises(ResearchRouteUnavailable, match="method binding drifted"):
        route.choose_move(BoardState.new_game())
    assert forged_calls == []


def test_raw_search_cannot_self_author_post_gate_outcome() -> None:
    board = BoardState.new_game()
    forged = _fallback_outcome(board)
    gate = _Gate(error=RuntimeError("force fallback"))
    route, _, _ = _route(
        gate=gate,
        factory=_factory(gate, raw_choose=lambda _board: forged),
    )

    with pytest.raises(
        ResearchRouteUnavailable,
        match="self-authored|raw search",
    ):
        route.choose_move(board)
    assert gate.calls == []


def test_factory_create_chooser_and_double_failure_never_return_raw_move() -> None:
    board = BoardState.new_game()

    def raising_factory():
        raise RuntimeError("cannot create exact engine")

    gate = _Gate(error=RuntimeError("student gate failed"))
    route, _, _ = _route(
        gate=gate,
        factory=_factory(gate, search_constructor=raising_factory),
    )
    with pytest.raises(ResearchRouteUnavailable):
        route.choose_move(board)

    def raising_raw_choose(_board):
        raise RuntimeError("exact D9 failed")

    gate = _Gate(error=RuntimeError("student gate failed"))
    route, _, _ = _route(
        gate=gate,
        factory=_factory(gate, raw_choose=raising_raw_choose),
    )
    with pytest.raises(ResearchRouteUnavailable):
        route.choose_move(board)


def test_factory_must_be_bound_to_same_gate_and_difficulty_nine() -> None:
    gate = _Gate()
    other_gate = _Gate()
    with pytest.raises(ResearchRouteUnavailable, match="gate"):
        _make_test_classical_a_pos_route(
            model=_Model(torch.zeros(24)),
            encoder=_encoder,
            gate=gate,
            exact_d9_factory=_factory(other_gate),
        )
    with pytest.raises(ResearchRouteUnavailable, match="difficulty"):
        _make_test_classical_a_pos_route(
            model=_Model(torch.zeros(24)),
            encoder=_encoder,
            gate=gate,
            exact_d9_factory=_factory(gate, difficulty=8),
        )


def test_factory_binding_cannot_be_reissued_after_original_route_is_gone() -> None:
    gate = _Gate()
    factory = _factory(gate)
    route = _make_test_classical_a_pos_route(
        model=_Model(torch.zeros(24)),
        encoder=_encoder,
        gate=gate,
        exact_d9_factory=factory,
    )
    del route
    gc.collect()

    with pytest.raises(ResearchRouteUnavailable, match="already bound"):
        _make_test_classical_a_pos_route(
            model=_Model(torch.zeros(24)),
            encoder=_encoder,
            gate=gate,
            exact_d9_factory=factory,
        )
