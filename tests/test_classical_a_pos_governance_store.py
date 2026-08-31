from __future__ import annotations

import inspect
import copy
import hashlib
import json
import os
import pickle
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import learned_ai.training.classical_a_pos_governance as governance
import learned_ai.training.classical_a_pos_governance_store as store_module
from game.board import POSITIONS, BoardState
from game.rules import get_all_legal_moves, terminal_result
from learned_ai.evaluation.sanmill_uci import (
    EXPECTED_RULES_IDENTITY_SHA256,
    UciPositionState,
    UciStrictRefereeIdentity,
)
from learned_ai.training.classical_a_pos_governance_store import (
    ActiveStateGenerationAttempt,
    ConfirmedStateGenerationCompletion,
    DurableGovernanceStore,
    DurableStateFreezeBinding,
    GovernanceStoreContractError,
    GovernanceStoreSpec,
    PendingStateGenerationCommit,
    ProductionAPosInventoryBinding,
    build_governance_store_spec,
)
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.training.sanmill_referee import (
    TRAINING_REFEREE_FORMAT,
    TRAINING_REFEREE_PROFILE,
    TRAINING_REFEREE_SEMANTIC_DIGEST,
    TRAINING_REPETITION_OBSERVATION,
    nmm_move_actions,
)


EXPECTED_PUBLIC_API = (
    "GovernanceStoreContractError",
    "GovernanceStoreSpec",
    "DurableGovernanceStore",
    "ProductionAPosInventoryBinding",
    "ProductionStrictRefereeBinding",
    "ActiveStateGenerationAttempt",
    "PendingStateGenerationCommit",
    "ConfirmedStateGenerationCompletion",
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
    "verify_durable_state_freeze",
)


def _mutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable(item) for item in value]
    return value


def _sha(label: str) -> str:
    return canonical_sha256(
        {
            "schema_version": "nmm.classical-a-pos-store-test-identity.v1",
            "label": label,
        }
    )


def test_store_public_surface_is_exact_and_has_no_raw_append() -> None:
    assert store_module.__all__ == EXPECTED_PUBLIC_API
    assert not hasattr(store_module, "append_event")
    assert not hasattr(store_module, "append_governance_event")
    for name in EXPECTED_PUBLIC_API[8:]:
        assert callable(getattr(store_module, name))


def test_c4a_source_games_and_domain_stream_schema_are_v2() -> None:
    assert store_module._SOURCE_GAME_SCHEMA == "nmm.classical-a-pos-source-game.v2"
    domain_ddl = next(
        statement
        for statement in store_module._DDL_STATEMENTS
        if statement.startswith("CREATE TABLE domain_records")
    )
    for required in (
        "stream_identity",
        "stream_kind",
        "sequence",
        "previous_record_identity",
        "record_bytes_sha256",
        "PRIMARY KEY (stream_identity, sequence)",
        "FOREIGN KEY (previous_record_identity)",
        "CHECK (record_identity = record_bytes_sha256)",
    ):
        assert required in domain_ddl


def test_c4a_declares_unissued_production_strict_referee_binding() -> None:
    binding_type = store_module.ProductionStrictRefereeBinding
    assert "ProductionStrictRefereeBinding" in store_module.__all__
    with pytest.raises((TypeError, GovernanceStoreContractError)):
        binding_type(object())
    assert not hasattr(store_module, "issue_production_strict_referee_binding")


def test_store_spec_is_exact_plan_bound_deeply_immutable_and_nonce_free() -> None:
    fixture = governance._issue_test_governance_fixture()
    spec = build_governance_store_spec(
        fixture.plan,
        readiness_identity=fixture.authorization["readiness_identity"],
        output_root_identity=_sha("output-root"),
    )
    assert type(spec) is GovernanceStoreSpec
    assert set(spec) == {
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
    assert "nonce" not in spec
    assert spec["normalized_relative_path"] == (
        "governance/classical-a-pos-controller.sqlite3"
    )
    assert (
        spec["managed_git_state_identity"]
        == fixture.plan["technical_bindings"]["managed_git_state"]
    )
    assert (
        spec["launch_path_binding_identity"]
        == fixture.plan["technical_bindings"]["launch_path_binding"]
    )
    assert (
        spec["state_freeze_teacher_order_event_chain_identity"]
        == fixture.plan["technical_bindings"]["state_freeze_teacher_order_event_chain"]
    )
    body = {
        key: _mutable(value) for key, value in spec.items() if key != "spec_identity"
    }
    assert spec["spec_identity"] == canonical_sha256(body)
    with pytest.raises(TypeError):
        spec["database_role"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    "capability_type",
    [
        DurableGovernanceStore,
        ProductionAPosInventoryBinding,
        store_module.ProductionStrictRefereeBinding,
        ActiveStateGenerationAttempt,
        PendingStateGenerationCommit,
        ConfirmedStateGenerationCompletion,
        DurableStateFreezeBinding,
    ],
)
def test_production_store_capabilities_reject_direct_construction(
    capability_type: type,
) -> None:
    assert "__dict__" not in capability_type.__dict__
    with pytest.raises((TypeError, GovernanceStoreContractError)):
        capability_type(object())


def test_store_module_has_no_cli_subprocess_training_or_smoke_surface() -> None:
    source = inspect.getsource(store_module)
    for forbidden in (
        "subprocess",
        "argparse",
        "click",
        "train(",
        "smoke(",
        "_make_event",
        "_ReplayContext",
        "_issue_test_governance_fixture",
    ):
        assert forbidden not in source
    annotations = inspect.signature(build_governance_store_spec).parameters
    assert "store_nonce" not in annotations


def test_production_store_cannot_bootstrap_from_an_empty_namespace(
    tmp_path: Path,
) -> None:
    fixture = governance._issue_test_governance_fixture()
    spec = build_governance_store_spec(
        fixture.plan,
        readiness_identity=fixture.authorization["readiness_identity"],
        output_root_identity=_sha("empty-output-root"),
    )
    with pytest.raises(GovernanceStoreContractError):
        store_module.initialize_governance_store(
            tmp_path,
            spec,
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
        )
    with pytest.raises(GovernanceStoreContractError):
        store_module.open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )
    assert not (tmp_path / "governance").exists()


def test_c4a_public_open_cannot_upgrade_a_disk_relabelled_test_database(
    tmp_path: Path,
) -> None:
    fixture, spec, _store = _initialize_test_store(tmp_path)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        raw_meta = bytes(
            connection.execute(
                "SELECT value FROM meta WHERE key='store_meta'"
            ).fetchone()[0]
        )
        meta = json.loads(raw_meta)
        meta["domain"] = "production"
        connection.execute("DROP TRIGGER meta_reject_update")
        connection.execute(
            "UPDATE meta SET value=? WHERE key='store_meta'",
            (canonical_json_bytes(meta),),
        )
        trigger = next(
            statement
            for statement in store_module._DDL_STATEMENTS
            if statement.startswith("CREATE TRIGGER meta_reject_update")
        )
        connection.execute(trigger)
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module.open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def _test_layout() -> dict[str, dict[str, dict[str, int]]]:
    return {
        "train": {
            "placement": {"W": 0, "B": 0},
            "movement": {"W": 0, "B": 0},
            "flying": {"W": 0, "B": 0},
        },
        "dev": {
            "placement": {"W": 1, "B": 0},
            "movement": {"W": 0, "B": 0},
            "flying": {"W": 0, "B": 0},
        },
    }


_TERMINAL_ACTIONS = (
    "d6",
    "f4",
    "d2",
    "c4",
    "b4",
    "f2",
    "f6",
    "d7",
    "b6",
    "xc4",
    "d1",
    "b2",
    "xd1",
    "g4",
    "e4",
    "c5",
    "a4",
    "e5",
    "c4",
    "xc5",
    "d3",
    "c4-c3",
    "d3-e3",
    "d6-d5",
    "e3-d3",
    "c3-c4",
    "xd7",
    "g4-g1",
    "b6-d6",
    "f4-g4",
    "d2-d1",
    "d3-d2",
    "d6-d7",
    "g4-g7",
    "f6-f4",
    "g1-g4",
    "b4-b6",
    "d2-d3",
    "e4-e3",
    "g4-g1",
    "b2-d2",
    "g7-g4",
    "e3-e4",
    "d3-c3",
    "d7-g7",
    "c3-d3",
    "b6-b4",
    "xd3",
)
_NODE_FOR_NMM_POSITION = (
    23,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    15,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    7,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
)
_TEST_SANMILL_RUNTIME_IDENTITY = _sha("strict-sanmill-runtime")
_TEST_REFEREE_BINDING_IDENTITY = _sha("strict-referee-binding")


def _terminal_history() -> tuple[list[dict[str, Any]], BoardState]:
    board = BoardState.new_game()
    logical_moves: list[dict[str, Any]] = []
    action_index = 0
    while action_index < len(_TERMINAL_ACTIONS):
        turn_actions = [_TERMINAL_ACTIONS[action_index]]
        action_index += 1
        if action_index < len(_TERMINAL_ACTIONS) and _TERMINAL_ACTIONS[
            action_index
        ].startswith("x"):
            turn_actions.append(_TERMINAL_ACTIONS[action_index])
            action_index += 1
        matches = [
            dict(move)
            for move in get_all_legal_moves(board)
            if nmm_move_actions(move) == tuple(turn_actions)
        ]
        assert len(matches) == 1
        logical_moves.append(matches[0])
        board = board.apply_move(matches[0])
    assert terminal_result(board) == (True, "W", "no-legal-move")
    return logical_moves, board


def _terminal_tgf_fen(board: BoardState) -> str:
    nodes = ["*"] * 24
    for position_index, position in enumerate(POSITIONS):
        piece = board.positions[position]
        nodes[_NODE_FOR_NMM_POSITION[position_index]] = {"": "*", "W": "O", "B": "@"}[
            piece
        ]
    board_field = "/".join("".join(nodes[start : start + 8]) for start in (0, 8, 16))
    return (
        f"{board_field} b o ? {board.pieces_on_board['W']} "
        f"{9 - board.pieces_placed['W']} {board.pieces_on_board['B']} "
        f"{9 - board.pieces_placed['B']} 0 0 -1 -1 -1 -1 0 0 1 ids:nodes"
    )


def _typed_terminal_state(
    logical_moves: list[dict[str, Any]],
    board: BoardState,
) -> UciPositionState:
    actions = tuple(
        action for move in logical_moves for action in nmm_move_actions(move)
    )
    return UciPositionState(
        status="terminal",
        ruleset_id="nmm",
        rules_identity_sha256=EXPECTED_RULES_IDENTITY_SHA256,
        rules_options={},
        history_origin="game_start",
        fen=_terminal_tgf_fen(board),
        side_to_move=None,
        phase="game_over",
        action="game_over",
        pending_removal_count=0,
        pending_removals=(0, 0),
        legal_actions=(),
        action_token_count=len(actions),
        logical_ply_count=len(logical_moves),
        logical_plies_by_side=((len(logical_moves) + 1) // 2, len(logical_moves) // 2),
        no_capture_count=0,
        repetition_current_count=1,
        repetition_history_length=len(logical_moves) + 1,
        snapshot_history_length=len(logical_moves) + 1,
        history_sha256=canonical_sha256(list(actions)),
        terminal=True,
        winner="white",
        winner_code=0,
        outcome_reason="loseNoLegalMoves",
        outcome_reason_code="lose_no_legal_moves",
        raw_line="test-only-typed-terminal",
        strict_referee_identity=UciStrictRefereeIdentity(
            format=TRAINING_REFEREE_FORMAT,
            profile=TRAINING_REFEREE_PROFILE,
            repetition_observation=TRAINING_REPETITION_OBSERVATION,
            origin_counted=True,
            semantic_digest=TRAINING_REFEREE_SEMANTIC_DIGEST,
        ),
    )


def _typed_draw_state(
    logical_moves: list[dict[str, Any]],
    board: BoardState,
    *,
    reason: str,
    reason_code: str,
) -> UciPositionState:
    actions = tuple(
        action for move in logical_moves for action in nmm_move_actions(move)
    )
    return UciPositionState(
        status="terminal",
        ruleset_id="nmm",
        rules_identity_sha256=EXPECTED_RULES_IDENTITY_SHA256,
        rules_options={},
        history_origin="game_start",
        fen=_terminal_tgf_fen(board),
        side_to_move=None,
        phase="game_over",
        action="game_over",
        pending_removal_count=0,
        pending_removals=(0, 0),
        legal_actions=(),
        action_token_count=len(actions),
        logical_ply_count=len(logical_moves),
        logical_plies_by_side=((len(logical_moves) + 1) // 2, len(logical_moves) // 2),
        no_capture_count=0,
        repetition_current_count=3 if reason == "drawThreefoldRepetition" else 1,
        repetition_history_length=len(logical_moves) + 1,
        snapshot_history_length=len(logical_moves) + 1,
        history_sha256=canonical_sha256(list(actions)),
        terminal=True,
        winner=None,
        winner_code=None,
        outcome_reason=reason,
        outcome_reason_code=reason_code,
        raw_line="test-only-typed-draw",
        strict_referee_identity=UciStrictRefereeIdentity(
            format=TRAINING_REFEREE_FORMAT,
            profile=TRAINING_REFEREE_PROFILE,
            repetition_observation=TRAINING_REPETITION_OBSERVATION,
            origin_counted=True,
            semantic_digest=TRAINING_REFEREE_SEMANTIC_DIGEST,
        ),
    )


def _source_bytes_for_history(
    logical_moves: list[dict[str, Any]],
    final_state: UciPositionState,
    *,
    attempt_identity: str,
    termination_class: str,
) -> bytes:
    board = BoardState.new_game()
    for move in logical_moves:
        board = board.apply_move(move)
    local_terminal, local_winner, local_reason = terminal_result(board)
    actions = [action for move in logical_moves for action in nmm_move_actions(move)]
    logical_identity = canonical_sha256(logical_moves)
    final_record = final_state.portable_record()
    final_identity = canonical_sha256(final_record)
    game_id = canonical_sha256(
        {
            "schema_version": "nmm.classical-a-pos-source-game-id.v1",
            "state_generation_attempt_identity": attempt_identity,
            "game_index": 0,
            "logical_moves_sha256": logical_identity,
            "sanmill_final_state_identity": final_identity,
            "referee_binding_identity": _TEST_REFEREE_BINDING_IDENTITY,
        }
    )
    body = {
        "schema_version": "nmm.classical-a-pos-source-game.v2",
        "state_generation_attempt_identity": attempt_identity,
        "game_id": game_id,
        "game_index": 0,
        "split": "dev",
        "candidate_color": "W",
        "complete_history": {
            "logical_moves": logical_moves,
            "logical_ply_count": len(logical_moves),
            "logical_moves_sha256": logical_identity,
            "sanmill_actions": actions,
            "action_token_count": len(actions),
            "sanmill_actions_sha256": canonical_sha256(actions),
        },
        "sanmill_runtime_identity": _TEST_SANMILL_RUNTIME_IDENTITY,
        "sanmill_final_state": final_record,
        "sanmill_final_state_identity": final_identity,
        "strict_terminal": {
            "terminal": True,
            "termination_class": termination_class,
            "winner": local_winner,
            "outcome_reason": final_state.outcome_reason,
            "outcome_reason_code": final_state.outcome_reason_code,
            "local_board": {
                "terminal": local_terminal,
                "winner": local_winner,
                "reason": local_reason,
            },
        },
        "referee_binding_identity": _TEST_REFEREE_BINDING_IDENTITY,
    }
    return (
        canonical_json_bytes({**body, "record_identity": canonical_sha256(body)})
        + b"\n"
    )


def _source_validation_context(
    attempt_identity: str,
    state: UciPositionState,
):
    binding = store_module._issue_test_strict_referee_binding(
        binding_identity=_TEST_REFEREE_BINDING_IDENTITY,
        runtime_identity=_TEST_SANMILL_RUNTIME_IDENTITY,
        attempt_identity=attempt_identity,
        complete_history_verifier=lambda _moves, _actions: state,
        prefix_history_verifier=lambda moves: canonical_sha256(
            [action for move in moves for action in nmm_move_actions(move)]
        ),
    )
    return store_module._TEST_STRICT_REFEREE_CONTEXTS[binding]


def _artifact_bytes_for_attempt(
    verifier_identity: str,
    attempt_identity: str,
) -> tuple[bytes, bytes, bytes]:
    complete_moves, final_board = _terminal_history()
    final_state = _typed_terminal_state(complete_moves, final_board).portable_record()
    final_state_identity = canonical_sha256(final_state)
    complete_actions = [
        action for move in complete_moves for action in nmm_move_actions(move)
    ]
    complete_history_identity = canonical_sha256(complete_moves)
    game_id = canonical_sha256(
        {
            "schema_version": "nmm.classical-a-pos-source-game-id.v1",
            "state_generation_attempt_identity": attempt_identity,
            "game_index": 0,
            "logical_moves_sha256": complete_history_identity,
            "sanmill_final_state_identity": final_state_identity,
            "referee_binding_identity": _TEST_REFEREE_BINDING_IDENTITY,
        }
    )
    source_body = {
        "schema_version": "nmm.classical-a-pos-source-game.v2",
        "state_generation_attempt_identity": attempt_identity,
        "game_id": game_id,
        "game_index": 0,
        "split": "dev",
        "candidate_color": "W",
        "complete_history": {
            "logical_moves": complete_moves,
            "logical_ply_count": len(complete_moves),
            "logical_moves_sha256": complete_history_identity,
            "sanmill_actions": complete_actions,
            "action_token_count": len(complete_actions),
            "sanmill_actions_sha256": canonical_sha256(complete_actions),
        },
        "sanmill_runtime_identity": _TEST_SANMILL_RUNTIME_IDENTITY,
        "sanmill_final_state": final_state,
        "sanmill_final_state_identity": final_state_identity,
        "strict_terminal": {
            "terminal": True,
            "termination_class": "rules-win",
            "winner": "W",
            "outcome_reason": "loseNoLegalMoves",
            "outcome_reason_code": "lose_no_legal_moves",
            "local_board": {
                "terminal": True,
                "winner": "W",
                "reason": "no-legal-move",
            },
        },
        "referee_binding_identity": _TEST_REFEREE_BINDING_IDENTITY,
    }
    source_record = {**source_body, "record_identity": canonical_sha256(source_body)}
    source_bytes = canonical_json_bytes(source_record) + b"\n"

    history: list[dict[str, Any]] = []
    board = BoardState.new_game()
    legal = [dict(move) for move in get_all_legal_moves(board)]
    mask = [index < 2 for index in range(len(legal))]
    history_identity = canonical_sha256(history)
    inventory = {
        "board_fen": board.to_fen_string(),
        "legal_actions": legal,
        "a_pos_mask": mask,
    }
    verification_body = {
        "verifier_identity": verifier_identity,
        "inventory_sha256": canonical_sha256(inventory),
        "previous_verification_sha256": "0" * 64,
    }
    verification = {
        **verification_body,
        "verification_sha256": canonical_sha256(verification_body),
    }
    state_body = {
        "schema_version": "nmm.classical-a-pos-state-record.v1",
        "example_id": canonical_sha256(
            {
                "history_sha256": history_identity,
                "board_fen": board.to_fen_string(),
            }
        ),
        "game_id": game_id,
        "game_index": 0,
        "split": "dev",
        "stratum": "placement",
        "candidate_color": "W",
        "logical_ply": 0,
        "history_moves": history,
        "history_sha256": history_identity,
        "sanmill_history_sha256": canonical_sha256([]),
        "board_fen": board.to_fen_string(),
        "legal_actions": legal,
        "a_pos_mask": mask,
        "a_pos_verification": verification,
    }
    state_record = {
        **state_body,
        "state_record_identity": canonical_sha256(state_body),
    }
    state_bytes = canonical_json_bytes(state_record) + b"\n"
    singleton_body = {
        "schema_version": "nmm.classical-a-pos-singleton-ledger.v1",
        "count": 0,
        "entries": [],
    }
    singleton_bytes = canonical_json_bytes(
        {**singleton_body, "identity": canonical_sha256(singleton_body)}
    )
    return source_bytes, state_bytes, singleton_bytes


def _initialize_test_store(tmp_path: Path):
    fixture = governance._issue_test_governance_fixture()
    spec = build_governance_store_spec(
        fixture.plan,
        readiness_identity=fixture.authorization["readiness_identity"],
        output_root_identity=_sha("test-output-root"),
    )
    bootstrap = governance.encode_governance_ledger(fixture.events[:3])
    store = store_module._test_initialize_governance_store(
        tmp_path,
        spec,
        plan=fixture.plan,
        authorization=fixture.authorization,
        bootstrap_ledger_bytes=bootstrap,
    )
    return fixture, spec, store


def _advance_test_store_to_active(tmp_path: Path):
    fixture, spec, store = _initialize_test_store(tmp_path)
    context = store_module._TEST_STORE_CONTEXTS[store]
    pending_auth, _ = governance._test_prepare_authorization_consumption(
        fixture.plan_permit,
        fixture.authorization_permit,
        context.replay,
        timestamp_utc="2026-09-01T00:00:03Z",
    )
    consumed = store_module._test_commit_authorization_consumption(
        store,
        pending_auth,
    )
    pending_operation, _ = governance._test_prepare_operation_reservation(
        consumed,
        context.replay,
        operation_id="state-generation",
        timestamp_utc="2026-09-01T00:00:04Z",
    )
    operation_permit = store_module._test_commit_operation_reservation(
        store,
        pending_operation,
    )
    active = store_module._test_begin_state_generation_attempt(
        store,
        operation_permit,
    )
    return fixture, spec, store, active


def _write_artifacts(
    root: Path,
    verifier_identity: str,
    attempt_identity: str,
    *,
    source_bytes: bytes | None = None,
    state_bytes: bytes | None = None,
    singleton_bytes: bytes | None = None,
) -> tuple[Path, Path, Path]:
    valid_source, valid_state, valid_singleton = _artifact_bytes_for_attempt(
        verifier_identity,
        attempt_identity,
    )
    source_path = root / "source-games.jsonl"
    state_path = root / "state-split.jsonl"
    singleton_path = root / "singletons.json"
    source_path.write_bytes(valid_source if source_bytes is None else source_bytes)
    state_path.write_bytes(valid_state if state_bytes is None else state_bytes)
    singleton_path.write_bytes(
        valid_singleton if singleton_bytes is None else singleton_bytes
    )
    return source_path, state_path, singleton_path


def _test_inventory_binding(verifier_identity: str):
    return store_module._issue_test_a_pos_inventory_binding(
        verifier_identity=verifier_identity,
        inventory_verifier=lambda _board, legal: tuple(
            index < 2 for index in range(len(legal))
        ),
        layout=_test_layout(),
        maximum_source_games=1,
    )


def _test_strict_referee_binding(attempt_identity: str):
    expected_moves, expected_board = _terminal_history()
    expected_state = _typed_terminal_state(expected_moves, expected_board)

    def verify_complete(
        logical_moves: Any,
        sanmill_actions: Any,
    ) -> UciPositionState:
        assert list(logical_moves) == expected_moves
        assert tuple(sanmill_actions) == tuple(_TERMINAL_ACTIONS)
        return expected_state

    def verify_prefix(logical_moves: Any) -> str:
        actions = [
            action for move in logical_moves for action in nmm_move_actions(move)
        ]
        return canonical_sha256(actions)

    return store_module._issue_test_strict_referee_binding(
        binding_identity=_TEST_REFEREE_BINDING_IDENTITY,
        runtime_identity=_TEST_SANMILL_RUNTIME_IDENTITY,
        attempt_identity=attempt_identity,
        complete_history_verifier=verify_complete,
        prefix_history_verifier=verify_prefix,
    )


def _prepare_valid_state_commit(tmp_path: Path):
    fixture, spec, store, active = _advance_test_store_to_active(tmp_path)
    verifier_identity = _sha("live-a-pos-verifier")
    inventory = _test_inventory_binding(verifier_identity)
    attempt_identity = store_module._TEST_ACTIVE_CONTEXTS[active].attempt_identity
    strict_referee = _test_strict_referee_binding(attempt_identity)
    source_path, state_path, singleton_path = _write_artifacts(
        tmp_path,
        verifier_identity,
        attempt_identity,
    )
    pending = store_module._test_prepare_state_generation_commit(
        active,
        inventory,
        strict_referee,
        source_games_path=source_path,
        state_split_path=state_path,
        singleton_ledger_path=singleton_path,
        timestamp_utc="2026-09-01T00:00:05Z",
        active_seconds=11,
    )
    return (
        fixture,
        spec,
        store,
        active,
        inventory,
        pending,
        source_path,
        state_path,
        singleton_path,
    )


def test_source_game_v2_rejects_legacy_empty_nonterminal_records() -> None:
    legacy_records = []
    for game_index in range(16):
        body = {
            "schema_version": "nmm.classical-a-pos-source-game.v1",
            "game_id": f"legacy-{game_index}",
            "game_index": game_index,
            "candidate_color": "W" if game_index % 2 == 0 else "B",
            "history_moves": [],
            "history_sha256": canonical_sha256([]),
        }
        legacy_records.append({**body, "record_identity": canonical_sha256(body)})
    payload = b"".join(canonical_json_bytes(item) + b"\n" for item in legacy_records)
    with pytest.raises(GovernanceStoreContractError):
        store_module._validate_source_games(
            payload,
            policy=store_module._PRODUCTION_POLICY,
        )


@pytest.mark.parametrize(
    ("termination_class", "reason", "reason_code"),
    [
        (
            "repetition-draw",
            "drawThreefoldRepetition",
            "draw_threefold_repetition",
        ),
        ("rules-draw", "drawFiftyMove", "draw_fifty_move"),
    ],
)
def test_source_game_v2_accepts_strict_sanmill_draw_after_local_nonterminal(
    termination_class: str,
    reason: str,
    reason_code: str,
) -> None:
    board = BoardState.new_game()
    move = dict(get_all_legal_moves(board)[0])
    final_board = board.apply_move(move)
    assert terminal_result(final_board) == (False, None, None)
    attempt_identity = _sha(f"draw-attempt-{termination_class}")
    state = _typed_draw_state(
        [move], final_board, reason=reason, reason_code=reason_code
    )
    payload = _source_bytes_for_history(
        [move],
        state,
        attempt_identity=attempt_identity,
        termination_class=termination_class,
    )
    identity, games = store_module._validate_source_games(
        payload,
        policy=store_module._test_artifact_policy(
            _test_layout(),
            maximum_source_games=1,
        ),
        attempt_identity=attempt_identity,
        strict_referee=_source_validation_context(attempt_identity, state),
    )
    assert len(identity) == 64
    assert len(games) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "empty-history",
        "terminal-legal-actions",
        "ongoing-status",
        "logical-count",
        "action-count",
        "logical-hash",
        "action-hash",
        "fen",
        "fen-turn",
        "side-to-move",
        "final-identity",
        "runtime-identity",
        "referee-identity",
        "winner-conflict",
        "continue-after-terminal",
    ],
)
def test_source_game_v2_resigned_contract_attacks_fail_closed(mutation: str) -> None:
    attempt_identity = _sha(f"source-attack-{mutation}")
    moves, board = _terminal_history()
    state = _typed_terminal_state(moves, board)
    record = json.loads(
        _source_bytes_for_history(
            moves,
            state,
            attempt_identity=attempt_identity,
            termination_class="rules-win",
        )
    )
    history = record["complete_history"]
    if mutation == "empty-history":
        history["logical_moves"] = []
        history["logical_ply_count"] = 0
        history["logical_moves_sha256"] = canonical_sha256([])
        history["sanmill_actions"] = []
        history["action_token_count"] = 0
        history["sanmill_actions_sha256"] = canonical_sha256([])
    elif mutation == "terminal-legal-actions":
        record["sanmill_final_state"]["legal_actions"] = ["a1"]
    elif mutation == "ongoing-status":
        record["sanmill_final_state"]["status"] = "ok"
    elif mutation == "logical-count":
        history["logical_ply_count"] += 1
    elif mutation == "action-count":
        history["action_token_count"] += 1
    elif mutation == "logical-hash":
        history["logical_moves_sha256"] = "f" * 64
    elif mutation == "action-hash":
        history["sanmill_actions_sha256"] = "f" * 64
    elif mutation == "fen":
        record["sanmill_final_state"]["fen"] = (
            "**O**O**/**@**@**/******** w o ? 2 0 2 0 0 0 -1 -1 -1 -1 0 0 1 ids:nodes"
        )
    elif mutation == "fen-turn":
        fen_fields = record["sanmill_final_state"]["fen"].split()
        assert fen_fields[1] == "b"
        fen_fields[1] = "w"
        record["sanmill_final_state"]["fen"] = " ".join(fen_fields)
    elif mutation == "side-to-move":
        record["sanmill_final_state"]["side_to_move"] = "white"
    elif mutation == "final-identity":
        record["sanmill_final_state_identity"] = "f" * 64
    elif mutation == "runtime-identity":
        record["sanmill_runtime_identity"] = "f" * 64
    elif mutation == "referee-identity":
        record["referee_binding_identity"] = "f" * 64
    elif mutation == "winner-conflict":
        record["strict_terminal"]["winner"] = "B"
    elif mutation == "continue-after-terminal":
        extra = dict(moves[0])
        history["logical_moves"].append(extra)
        history["logical_ply_count"] += 1
        history["logical_moves_sha256"] = canonical_sha256(history["logical_moves"])
        history["sanmill_actions"].extend(nmm_move_actions(extra))
        history["action_token_count"] = len(history["sanmill_actions"])
        history["sanmill_actions_sha256"] = canonical_sha256(history["sanmill_actions"])
    else:
        raise AssertionError(mutation)
    final_state = record["sanmill_final_state"]
    if mutation != "final-identity":
        record["sanmill_final_state_identity"] = canonical_sha256(final_state)
    record["game_id"] = canonical_sha256(
        {
            "schema_version": "nmm.classical-a-pos-source-game-id.v1",
            "state_generation_attempt_identity": attempt_identity,
            "game_index": 0,
            "logical_moves_sha256": history["logical_moves_sha256"],
            "sanmill_final_state_identity": record["sanmill_final_state_identity"],
            "referee_binding_identity": record["referee_binding_identity"],
        }
    )
    body = {key: value for key, value in record.items() if key != "record_identity"}
    record["record_identity"] = canonical_sha256(body)
    with pytest.raises(GovernanceStoreContractError):
        store_module._validate_source_games(
            canonical_json_bytes(record) + b"\n",
            policy=store_module._test_artifact_policy(
                _test_layout(),
                maximum_source_games=1,
            ),
            attempt_identity=attempt_identity,
            strict_referee=_source_validation_context(attempt_identity, state),
        )


@pytest.mark.parametrize(
    ("termination_class", "reason", "reason_code"),
    [
        ("repetition-draw", "drawFiftyMove", "draw_fifty_move"),
        (
            "rules-draw",
            "drawThreefoldRepetition",
            "draw_threefold_repetition",
        ),
        ("rules-draw", "drawUnknownPinnedReason", "draw_unknown_pinned_reason"),
    ],
)
def test_strict_terminal_rejects_resigned_draw_class_or_reason_mismatch(
    termination_class: str,
    reason: str,
    reason_code: str,
) -> None:
    board = BoardState.new_game()
    board = board.apply_move(dict(get_all_legal_moves(board)[0]))
    assert terminal_result(board) == (False, None, None)
    with pytest.raises(GovernanceStoreContractError):
        store_module._validate_strict_terminal(
            {
                "terminal": True,
                "termination_class": termination_class,
                "winner": None,
                "outcome_reason": reason,
                "outcome_reason_code": reason_code,
                "local_board": {
                    "terminal": False,
                    "winner": None,
                    "reason": None,
                },
            },
            board=board,
            outcome={
                "terminal": True,
                "winner": None,
                "winner_code": None,
                "reason": reason,
                "reason_code": reason_code,
            },
            field="strict terminal",
        )


def test_strict_terminal_rejects_resigned_rules_win_reason_code_drift() -> None:
    _moves, board = _terminal_history()
    reason = "loseNoLegalMoves"
    wrong_reason_code = "lose_no_legal_move"
    with pytest.raises(GovernanceStoreContractError):
        store_module._validate_strict_terminal(
            {
                "terminal": True,
                "termination_class": "rules-win",
                "winner": "W",
                "outcome_reason": reason,
                "outcome_reason_code": wrong_reason_code,
                "local_board": {
                    "terminal": True,
                    "winner": "W",
                    "reason": "no-legal-move",
                },
            },
            board=board,
            outcome={
                "terminal": True,
                "winner": "white",
                "winner_code": 0,
                "reason": reason,
                "reason_code": wrong_reason_code,
            },
            field="strict terminal",
        )


@pytest.mark.parametrize("mutation", ["index", "split", "colour", "duplicate-history"])
def test_source_game_index_split_colour_and_history_uniqueness_fail_closed(
    mutation: str,
) -> None:
    attempt_identity = _sha(f"source-index-attack-{mutation}")
    moves, board = _terminal_history()
    state = _typed_terminal_state(moves, board)
    first = json.loads(
        _source_bytes_for_history(
            moves,
            state,
            attempt_identity=attempt_identity,
            termination_class="rules-win",
        )
    )
    records = [first]
    target = first
    if mutation == "index":
        target["game_index"] = 1
    elif mutation == "split":
        target["split"] = "train"
    elif mutation == "colour":
        target["candidate_color"] = "B"
    else:
        target = json.loads(canonical_json_bytes(first))
        target["game_index"] = 1
        target["candidate_color"] = "B"
        records.append(target)
    if mutation in {"index", "duplicate-history"}:
        target["game_id"] = canonical_sha256(
            {
                "schema_version": "nmm.classical-a-pos-source-game-id.v1",
                "state_generation_attempt_identity": attempt_identity,
                "game_index": target["game_index"],
                "logical_moves_sha256": target["complete_history"][
                    "logical_moves_sha256"
                ],
                "sanmill_final_state_identity": target["sanmill_final_state_identity"],
                "referee_binding_identity": target["referee_binding_identity"],
            }
        )
    for record in records:
        body = {key: value for key, value in record.items() if key != "record_identity"}
        record["record_identity"] = canonical_sha256(body)
    payload = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    with pytest.raises(GovernanceStoreContractError):
        store_module._validate_source_games(
            payload,
            policy=store_module._test_artifact_policy(
                _test_layout(),
                maximum_source_games=len(records),
            ),
            attempt_identity=attempt_identity,
            strict_referee=_source_validation_context(attempt_identity, state),
        )


def _resign_state_record(record: dict[str, Any]) -> bytes:
    if "a_pos_verification" in record:
        verification = record["a_pos_verification"]
        verification["inventory_sha256"] = canonical_sha256(
            {
                "board_fen": record["board_fen"],
                "legal_actions": record["legal_actions"],
                "a_pos_mask": record["a_pos_mask"],
            }
        )
        verification_body = {
            key: value
            for key, value in verification.items()
            if key != "verification_sha256"
        }
        verification["verification_sha256"] = canonical_sha256(verification_body)
    body = {
        key: value for key, value in record.items() if key != "state_record_identity"
    }
    record["state_record_identity"] = canonical_sha256(body)
    return canonical_json_bytes(record) + b"\n"


def _mutated_artifact_bytes(
    verifier_identity: str,
    attempt_identity: str,
    mutation: str,
) -> tuple[bytes, bytes, bytes]:
    source, state, singleton = _artifact_bytes_for_attempt(
        verifier_identity,
        attempt_identity,
    )
    record = json.loads(state)
    if mutation == "mask":
        record["a_pos_mask"] = [
            index in {2, 3} for index in range(len(record["legal_actions"]))
        ]
        state = _resign_state_record(record)
    elif mutation == "legal-order":
        record["legal_actions"][0], record["legal_actions"][1] = (
            record["legal_actions"][1],
            record["legal_actions"][0],
        )
        state = _resign_state_record(record)
    elif mutation == "teacher-field":
        record["teacher_action"] = record["legal_actions"][0]
        state = _resign_state_record(record)
    elif mutation == "split":
        record["split"] = "train"
        state = _resign_state_record(record)
    elif mutation == "prefix":
        record["logical_ply"] = 1
        state = _resign_state_record(record)
    elif mutation == "identity":
        record["state_record_identity"] = "f" * 64
        state = canonical_json_bytes(record) + b"\n"
    elif mutation == "quota-plus-one":
        state += state
    elif mutation == "quota-minus-one":
        state = b""
    elif mutation == "singleton-count":
        ledger = json.loads(singleton)
        ledger["count"] = 1
        ledger_body = {key: value for key, value in ledger.items() if key != "identity"}
        ledger["identity"] = canonical_sha256(ledger_body)
        singleton = canonical_json_bytes(ledger)
    elif mutation == "source-index":
        game = json.loads(source)
        game["game_index"] = 1
        game_body = {
            key: value for key, value in game.items() if key != "record_identity"
        }
        game["record_identity"] = canonical_sha256(game_body)
        source = canonical_json_bytes(game) + b"\n"
    elif mutation == "game-id-type":
        record["game_id"] = ["game-0000"]
        state = _resign_state_record(record)
    elif mutation == "split-type":
        record["split"] = ["dev"]
        state = _resign_state_record(record)
    else:
        raise AssertionError(mutation)
    return source, state, singleton


def test_internal_test_store_durably_completes_and_freezes_state_generation(
    tmp_path: Path,
) -> None:
    fixture, spec, store, active = _advance_test_store_to_active(tmp_path)
    context = store_module._TEST_STORE_CONTEXTS[store]

    verifier_identity = _sha("live-a-pos-verifier")
    inventory_binding = store_module._issue_test_a_pos_inventory_binding(
        verifier_identity=verifier_identity,
        inventory_verifier=lambda _board, legal: tuple(
            index < 2 for index in range(len(legal))
        ),
        layout=_test_layout(),
        maximum_source_games=1,
    )
    attempt_identity = store_module._TEST_ACTIVE_CONTEXTS[active].attempt_identity
    strict_referee = _test_strict_referee_binding(attempt_identity)
    source_path, state_path, singleton_path = _write_artifacts(
        tmp_path,
        verifier_identity,
        attempt_identity,
    )

    pending_state = store_module._test_prepare_state_generation_commit(
        active,
        inventory_binding,
        strict_referee,
        source_games_path=source_path,
        state_split_path=state_path,
        singleton_ledger_path=singleton_path,
        timestamp_utc="2026-09-01T00:00:05Z",
        active_seconds=11,
    )
    completion = store_module._test_commit_state_generation(store, pending_state)
    assert context.replay.state == "state_generated"

    # Durable SQLite BLOBs, not mutable generator outputs, are authoritative.
    source_path.write_bytes(b"external file changed after durable completion\n")
    freeze = store_module._test_commit_state_freeze(
        store,
        completion,
        timestamp_utc="2026-09-01T00:00:06Z",
    )
    state_identity = store_module._test_verify_durable_state_freeze(store, freeze)
    assert (
        state_identity
        == store_module._TEST_FREEZE_CONTEXTS[freeze].state_split_identity
    )
    assert context.replay.state == "state_frozen"

    reopened = store_module._test_open_governance_store(
        tmp_path,
        spec,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    assert store_module._TEST_STORE_CONTEXTS[reopened].replay.state == "state_frozen"
    with pytest.raises(GovernanceStoreContractError):
        store_module.open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )
    with pytest.raises(GovernanceStoreContractError):
        store_module.verify_durable_state_freeze(store, freeze)


def test_completion_and_freeze_records_have_exact_flat_contracts(
    tmp_path: Path,
) -> None:
    _fixture, _spec, store, _active, _inventory, pending, *_paths = (
        _prepare_valid_state_commit(tmp_path)
    )
    completion = store_module._test_commit_state_generation(store, pending)
    store_module._test_commit_state_freeze(
        store,
        completion,
        timestamp_utc="2026-09-01T00:00:06Z",
    )
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT stream_identity, stream_kind, sequence, record_type, "
            "previous_record_identity, record_identity, record_bytes, "
            "record_bytes_sha256 FROM domain_records ORDER BY sequence"
        ).fetchall()
    finally:
        connection.close()
    assert len(rows) == 2
    completion_record = json.loads(bytes(rows[0][6]))
    freeze_record = json.loads(bytes(rows[1][6]))
    assert set(completion_record) == store_module._COMPLETION_KEYS
    assert store_module._DOMAIN_ENVELOPE_KEYS < set(completion_record)
    assert store_module._COMPLETION_GOVERNANCE_KEYS < set(completion_record)
    assert store_module._COMPLETION_RUNTIME_KEYS < set(completion_record)
    assert store_module._COMPLETION_RESULT_KEYS < set(completion_record)
    assert not {"domain", "governance", "runtime", "result"} & set(completion_record)
    assert "attempt_identity" in completion_record
    assert "state_generation_attempt_identity" not in completion_record
    assert set(freeze_record) == store_module._FREEZE_RECEIPT_KEYS
    assert store_module._DOMAIN_ENVELOPE_KEYS < set(freeze_record)
    assert store_module._FREEZE_GOVERNANCE_KEYS < set(freeze_record)
    assert store_module._FREEZE_RESULT_KEYS < set(freeze_record)
    assert not {"domain", "governance", "runtime", "result"} & set(freeze_record)
    assert "attempt_identity" in freeze_record
    assert rows[0][0] == rows[1][0]
    assert rows[0][1] == rows[1][1] == "controller"
    assert rows[0][0] == canonical_sha256(
        {
            "schema_version": "nmm.classical-a-pos-domain-stream.v1",
            "stream_kind": "controller",
            "store_spec_identity": completion_record["store_spec_identity"],
            "plan_identity": completion_record["plan_identity"],
            "authorization_consumption_identity": completion_record[
                "authorization_consumption_identity"
            ],
            "state_generation_attempt_identity": completion_record["attempt_identity"],
        }
    )
    assert rows[0][2:4] == (0, "state-generation-completion")
    assert rows[0][4] is None
    assert rows[1][2:4] == (1, "state-freeze")
    assert rows[1][4] == rows[0][5]
    for row in rows:
        assert row[5] == row[7] == hashlib.sha256(bytes(row[6])).hexdigest()
    for field in store_module._COMPLETION_RUNTIME_KEYS:
        identity = completion_record[field]
        assert (
            isinstance(identity, str) and len(identity) == 64 and identity != "0" * 64
        )
    assert completion_record["teacher_fields_present"] is False
    assert {
        field: freeze_record[field] for field in store_module._FREEZE_RESULT_KEYS
    } == {
        "teacher_fields_present": False,
        "teacher_may_start_only_after_this_event": True,
        "authoritative_artifacts": "sqlite-immutable-blobs",
    }
    assert [set(item) for item in completion_record["artifacts"]] == [
        {"role", "identity", "bytes_sha256", "size_bytes", "record_count"},
        {
            "role",
            "identity",
            "bytes_sha256",
            "size_bytes",
            "record_count",
            "a_pos_verifier_identity",
        },
        {"role", "identity", "bytes_sha256", "size_bytes", "record_count"},
    ]


def _frozen_test_database(tmp_path: Path):
    fixture, spec, store, _active, _inventory, pending, *_paths = (
        _prepare_valid_state_commit(tmp_path)
    )
    completion = store_module._test_commit_state_generation(store, pending)
    store_module._test_commit_state_freeze(
        store,
        completion,
        timestamp_utc="2026-09-01T00:00:06Z",
    )
    return (
        fixture,
        spec,
        tmp_path / "governance" / "classical-a-pos-controller.sqlite3",
    )


@pytest.mark.parametrize(
    "attack",
    [
        "gap",
        "wrong-previous",
        "cross-stream",
        "unknown-teacher-stream",
        "unknown-teacher-record",
    ],
)
def test_domain_record_stream_gap_cross_stream_and_unknown_types_fail_closed(
    tmp_path: Path,
    attack: str,
) -> None:
    fixture, spec, database = _frozen_test_database(tmp_path)
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        if attack == "gap":
            connection.execute("DROP TRIGGER domain_records_reject_delete")
            connection.execute(
                "DELETE FROM domain_records WHERE record_type="
                "'state-generation-completion'"
            )
            trigger_name = "domain_records_reject_delete"
        elif attack in {"unknown-teacher-stream", "unknown-teacher-record"}:
            prior = connection.execute(
                "SELECT stream_identity, record_identity FROM domain_records "
                "ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            if attack == "unknown-teacher-stream":
                stream_identity = _sha("future-teacher-stream")
                stream_kind = "teacher-label-stream"
                sequence = 0
                previous = None
            else:
                stream_identity = prior[0]
                stream_kind = "controller"
                sequence = 2
                previous = prior[1]
            record = {
                "schema_version": "nmm.classical-a-pos-future-teacher-record.v1",
                "stream_identity": stream_identity,
                "stream_kind": stream_kind,
                "sequence": sequence,
                "record_type": "teacher-label-reserved",
                "previous_record_identity": previous,
            }
            payload = canonical_json_bytes(record)
            identity = hashlib.sha256(payload).hexdigest()
            connection.execute(
                "INSERT INTO domain_records(stream_identity, stream_kind, sequence, "
                "record_type, previous_record_identity, record_identity, "
                "record_bytes, size_bytes, record_bytes_sha256) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    stream_identity,
                    stream_kind,
                    sequence,
                    "teacher-label-reserved",
                    previous,
                    identity,
                    payload,
                    len(payload),
                    identity,
                ),
            )
            trigger_name = None
        else:
            row = connection.execute(
                "SELECT record_bytes FROM domain_records WHERE record_type='state-freeze'"
            ).fetchone()
            record = json.loads(bytes(row[0]))
            if attack == "wrong-previous":
                record["previous_record_identity"] = "f" * 64
            else:
                record["stream_identity"] = _sha("cross-stream")
            payload = canonical_json_bytes(record)
            identity = hashlib.sha256(payload).hexdigest()
            connection.execute("DROP TRIGGER domain_records_reject_update")
            connection.execute(
                "UPDATE domain_records SET stream_identity=?, "
                "previous_record_identity=?, record_identity=?, record_bytes=?, "
                "size_bytes=?, record_bytes_sha256=? WHERE record_type='state-freeze'",
                (
                    record["stream_identity"],
                    record["previous_record_identity"],
                    identity,
                    payload,
                    len(payload),
                    identity,
                ),
            )
            trigger_name = "domain_records_reject_update"
        if trigger_name is not None:
            trigger = next(
                statement
                for statement in store_module._DDL_STATEMENTS
                if statement.startswith(f"CREATE TRIGGER {trigger_name}")
            )
            connection.execute(trigger)
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def _generated_test_database(tmp_path: Path):
    fixture, spec, store, _active, _inventory, pending, *_paths = (
        _prepare_valid_state_commit(tmp_path)
    )
    store_module._test_commit_state_generation(store, pending)
    return (
        fixture,
        spec,
        tmp_path / "governance" / "classical-a-pos-controller.sqlite3",
    )


def test_fully_resigned_wrong_controller_stream_identity_fails_closed(
    tmp_path: Path,
) -> None:
    fixture, spec, database = _generated_test_database(tmp_path)
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        record = json.loads(
            bytes(
                connection.execute(
                    "SELECT record_bytes FROM domain_records WHERE sequence=0"
                ).fetchone()[0]
            )
        )
        record["stream_identity"] = _sha("wrong-controller-stream")
        payload = canonical_json_bytes(record)
        identity = hashlib.sha256(payload).hexdigest()
        connection.execute("DROP TRIGGER domain_records_reject_update")
        connection.execute(
            "UPDATE domain_records SET stream_identity=?, record_identity=?, "
            "record_bytes=?, size_bytes=?, record_bytes_sha256=? WHERE sequence=0",
            (
                record["stream_identity"],
                identity,
                payload,
                len(payload),
                identity,
            ),
        )
        trigger = next(
            statement
            for statement in store_module._DDL_STATEMENTS
            if statement.startswith("CREATE TRIGGER domain_records_reject_update")
        )
        connection.execute(trigger)
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


@pytest.mark.parametrize(
    "raw_attack",
    ["bom", "duplicate-key", "nan", "noncanonical", "stale-row-hash"],
)
def test_domain_record_raw_bytes_and_row_hash_attacks_fail_closed(
    tmp_path: Path,
    raw_attack: str,
) -> None:
    fixture, spec, database = _generated_test_database(tmp_path)
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        original = bytes(
            connection.execute(
                "SELECT record_bytes FROM domain_records WHERE sequence=0"
            ).fetchone()[0]
        )
        if raw_attack == "bom":
            payload = b"\xef\xbb\xbf" + original
        elif raw_attack == "duplicate-key":
            payload = b'{"schema_version":"duplicate",' + original[1:]
        elif raw_attack == "nan":
            payload = b'{"domain":NaN}'
        elif raw_attack == "noncanonical":
            payload = json.dumps(json.loads(original), indent=1).encode("utf-8")
        else:
            payload = original + b" "
        identity = hashlib.sha256(payload).hexdigest()
        connection.execute("DROP TRIGGER domain_records_reject_update")
        if raw_attack == "stale-row-hash":
            connection.execute(
                "UPDATE domain_records SET record_bytes=?, size_bytes=? "
                "WHERE sequence=0",
                (payload, len(payload)),
            )
        else:
            connection.execute(
                "UPDATE domain_records SET record_identity=?, record_bytes=?, "
                "size_bytes=?, record_bytes_sha256=? WHERE sequence=0",
                (identity, payload, len(payload), identity),
            )
        trigger = next(
            statement
            for statement in store_module._DDL_STATEMENTS
            if statement.startswith("CREATE TRIGGER domain_records_reject_update")
        )
        connection.execute(trigger)
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_domain_record_duplicate_sequence_and_identity_are_sqlite_rejected(
    tmp_path: Path,
) -> None:
    _fixture, _spec, database = _generated_test_database(tmp_path)
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        row = connection.execute(
            "SELECT stream_identity, stream_kind, sequence, record_type, "
            "previous_record_identity, record_identity, record_bytes, size_bytes, "
            "record_bytes_sha256 FROM domain_records WHERE sequence=0"
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO domain_records(stream_identity, stream_kind, sequence, "
                "record_type, previous_record_identity, record_identity, "
                "record_bytes, size_bytes, record_bytes_sha256) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        duplicate_identity_row = (
            _sha("duplicate-identity-stream"),
            row[1],
            0,
            row[3],
            None,
            row[5],
            row[6],
            row[7],
            row[8],
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO domain_records(stream_identity, stream_kind, sequence, "
                "record_type, previous_record_identity, record_identity, "
                "record_bytes, size_bytes, record_bytes_sha256) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                duplicate_identity_row,
            )
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("host_preflight_identity", None),
        ("state_generator_session_identity", "0" * 64),
        ("strict_referee_binding_identity", "test-placeholder"),
        ("a_pos_inventory_binding_identity", "f" * 64),
        ("resource_snapshot_before_identity", "0" * 64),
        ("attempt_identity", "f" * 64),
    ],
)
def test_resigned_completion_missing_null_zero_or_drifting_identity_fails(
    tmp_path: Path,
    field: str,
    replacement: Any,
) -> None:
    fixture, spec, store, _active, _inventory, pending, *_paths = (
        _prepare_valid_state_commit(tmp_path)
    )
    store_module._test_commit_state_generation(store, pending)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        record = json.loads(
            bytes(
                connection.execute(
                    "SELECT record_bytes FROM domain_records WHERE "
                    "record_type='state-generation-completion'"
                ).fetchone()[0]
            )
        )
        if replacement is None:
            del record[field]
        else:
            record[field] = replacement
        payload = canonical_json_bytes(record)
        identity = hashlib.sha256(payload).hexdigest()
        connection.execute("DROP TRIGGER domain_records_reject_update")
        connection.execute(
            "UPDATE domain_records SET record_identity=?, record_bytes=?, "
            "size_bytes=?, record_bytes_sha256=? WHERE "
            "record_type='state-generation-completion'",
            (identity, payload, len(payload), identity),
        )
        trigger = next(
            statement
            for statement in store_module._DDL_STATEMENTS
            if statement.startswith("CREATE TRIGGER domain_records_reject_update")
        )
        connection.execute(trigger)
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("store_spec_identity", None),
        ("completion_record_identity", "0" * 64),
        ("attempt_identity", "test-placeholder"),
    ],
)
def test_resigned_freeze_missing_zero_or_placeholder_governance_fails_closed(
    tmp_path: Path,
    field: str,
    replacement: Any,
) -> None:
    fixture, spec, database = _frozen_test_database(tmp_path)
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        receipt = json.loads(
            bytes(
                connection.execute(
                    "SELECT record_bytes FROM domain_records WHERE "
                    "record_type='state-freeze'"
                ).fetchone()[0]
            )
        )
        if replacement is None:
            del receipt[field]
        else:
            receipt[field] = replacement
        payload = canonical_json_bytes(receipt)
        identity = hashlib.sha256(payload).hexdigest()
        connection.execute("DROP TRIGGER artifacts_reject_update")
        connection.execute("DROP TRIGGER domain_records_reject_update")
        connection.execute(
            "UPDATE artifacts SET artifact_identity=?, artifact_bytes=?, "
            "size_bytes=?, sha256=? WHERE role='state-freeze-receipt'",
            (identity, payload, len(payload), identity),
        )
        connection.execute(
            "UPDATE domain_records SET record_identity=?, record_bytes=?, "
            "size_bytes=?, record_bytes_sha256=? WHERE record_type='state-freeze'",
            (identity, payload, len(payload), identity),
        )
        for trigger_name in (
            "artifacts_reject_update",
            "domain_records_reject_update",
        ):
            trigger = next(
                statement
                for statement in store_module._DDL_STATEMENTS
                if statement.startswith(f"CREATE TRIGGER {trigger_name}")
            )
            connection.execute(trigger)
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_durable_capabilities_are_opaque_noncopyable_and_one_use(
    tmp_path: Path,
) -> None:
    (
        _fixture,
        _spec,
        store,
        active,
        inventory,
        pending,
        source_path,
        state_path,
        singleton_path,
    ) = _prepare_valid_state_commit(tmp_path)
    strict_referee = store_module._TEST_PENDING_CONTEXTS[pending].strict_referee_binding
    for capability in (store, active, inventory, strict_referee, pending):
        assert not hasattr(capability, "__dict__")
        with pytest.raises(AttributeError):
            capability.context = object()  # type: ignore[attr-defined]
        with pytest.raises(TypeError):
            copy.copy(capability)
        with pytest.raises(TypeError):
            copy.deepcopy(capability)
        with pytest.raises(TypeError):
            pickle.dumps(capability)
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_prepare_state_generation_commit(
            active,
            inventory,
            strict_referee,
            source_games_path=source_path,
            state_split_path=state_path,
            singleton_ledger_path=singleton_path,
            timestamp_utc="2026-09-01T00:00:05Z",
            active_seconds=11,
        )

    completion = store_module._test_commit_state_generation(store, pending)
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_commit_state_generation(store, pending)
    freeze = store_module._test_commit_state_freeze(
        store,
        completion,
        timestamp_utc="2026-09-01T00:00:06Z",
    )
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_commit_state_freeze(
            store,
            completion,
            timestamp_utc="2026-09-01T00:00:07Z",
        )
    assert store_module._test_verify_durable_state_freeze(store, freeze)


@pytest.mark.parametrize("suffix", ["-journal", "-wal", "-shm"])
def test_any_sqlite_sidecar_fails_closed_without_deleting_it(
    tmp_path: Path,
    suffix: str,
) -> None:
    fixture, spec, _store = _initialize_test_store(tmp_path)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    sidecar = Path(str(database) + suffix)
    sidecar.write_bytes(b"unexpected-sidecar")
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )
    assert sidecar.read_bytes() == b"unexpected-sidecar"


def test_sqlite_pragmas_schema_and_immutability_triggers_are_exact(
    tmp_path: Path,
) -> None:
    fixture, spec, _store = _initialize_test_store(tmp_path)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        assert connection.execute("PRAGMA synchronous").fetchone() == (2,)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE meta SET value=value WHERE key='store_meta'")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM events WHERE sequence=0")
    finally:
        connection.close()
    reopened = store_module._test_open_governance_store(
        tmp_path,
        spec,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    assert store_module._TEST_STORE_CONTEXTS[reopened].replay.state == (
        "authorized_unconsumed"
    )


def test_schema_or_meta_drift_fails_after_attacker_restores_trigger_text(
    tmp_path: Path,
) -> None:
    fixture, spec, _store = _initialize_test_store(tmp_path)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        connection.execute("DROP TRIGGER meta_reject_update")
        connection.execute(
            "UPDATE meta SET value=? WHERE key='store_meta'",
            (canonical_json_bytes({"forged": True}),),
        )
        trigger = next(
            statement
            for statement in store_module._DDL_STATEMENTS
            if statement.startswith("CREATE TRIGGER meta_reject_update")
        )
        connection.execute(trigger)
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_missing_immutability_trigger_fails_exact_schema_check(tmp_path: Path) -> None:
    fixture, spec, _store = _initialize_test_store(tmp_path)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        connection.execute("DROP TRIGGER artifacts_reject_delete")
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_extra_schema_object_fails_closed(tmp_path: Path) -> None:
    fixture, spec, _store = _initialize_test_store(tmp_path)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        connection.execute("CREATE TABLE rogue(value INTEGER)")
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_event_gap_fails_after_attacker_restores_delete_trigger(tmp_path: Path) -> None:
    fixture, spec, _store = _initialize_test_store(tmp_path)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        connection.execute("DROP TRIGGER events_reject_delete")
        connection.execute("DELETE FROM events WHERE sequence=1")
        trigger = next(
            statement
            for statement in store_module._DDL_STATEMENTS
            if statement.startswith("CREATE TRIGGER events_reject_delete")
        )
        connection.execute(trigger)
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_locked_commit_fails_immediately_and_pending_is_permanently_spent(
    tmp_path: Path,
) -> None:
    fixture, spec, store = _initialize_test_store(tmp_path)
    context = store_module._TEST_STORE_CONTEXTS[store]
    pending, _ = governance._test_prepare_authorization_consumption(
        fixture.plan_permit,
        fixture.authorization_permit,
        context.replay,
        timestamp_utc="2026-09-01T00:00:03Z",
    )
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    lock = sqlite3.connect(database, timeout=0.0, isolation_level=None)
    try:
        lock.execute("PRAGMA busy_timeout=0")
        lock.execute("BEGIN IMMEDIATE")
        with pytest.raises(GovernanceStoreContractError):
            store_module._test_commit_authorization_consumption(store, pending)
    finally:
        lock.execute("ROLLBACK")
        lock.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_commit_authorization_consumption(store, pending)
    reopened = store_module._test_open_governance_store(
        tmp_path,
        spec,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    assert store_module._TEST_STORE_CONTEXTS[reopened].replay.state == (
        "authorized_unconsumed"
    )


def test_injected_event_insert_fault_rolls_back_artifacts_and_spends_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        spec,
        store,
        _active,
        _inventory,
        pending,
        *_paths,
    ) = _prepare_valid_state_commit(tmp_path)
    real_open = store_module._open_write_connection

    class FailingConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def execute(self, sql: str, parameters: tuple[Any, ...] = ()):
            if sql.startswith("INSERT INTO events"):
                raise sqlite3.OperationalError("injected event insert fault")
            return self.connection.execute(sql, parameters)

        def close(self) -> None:
            self.connection.close()

    monkeypatch.setattr(
        store_module,
        "_open_write_connection",
        lambda path: FailingConnection(real_open(path)),
    )
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_commit_state_generation(store, pending)
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_commit_state_generation(store, pending)
    monkeypatch.setattr(store_module, "_open_write_connection", real_open)

    reopened = store_module._test_open_governance_store(
        tmp_path,
        spec,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    assert store_module._TEST_STORE_CONTEXTS[reopened].replay.state == (
        "state_generation_running"
    )
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM domain_records").fetchone() == (
            0,
        )
    finally:
        connection.close()


def test_state_freeze_lock_failure_spends_completion_without_partial_freeze(
    tmp_path: Path,
) -> None:
    fixture, spec, store, _active, _inventory, pending, *_paths = (
        _prepare_valid_state_commit(tmp_path)
    )
    completion = store_module._test_commit_state_generation(store, pending)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    lock = sqlite3.connect(database, timeout=0.0, isolation_level=None)
    try:
        lock.execute("PRAGMA busy_timeout=0")
        lock.execute("BEGIN IMMEDIATE")
        with pytest.raises(GovernanceStoreContractError):
            store_module._test_commit_state_freeze(
                store,
                completion,
                timestamp_utc="2026-09-01T00:00:06Z",
            )
    finally:
        lock.execute("ROLLBACK")
        lock.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_commit_state_freeze(
            store,
            completion,
            timestamp_utc="2026-09-01T00:00:07Z",
        )
    reopened = store_module._test_open_governance_store(
        tmp_path,
        spec,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    assert store_module._TEST_STORE_CONTEXTS[reopened].replay.state == (
        "state_generated"
    )


def test_artifact_toctou_fails_before_sqlite_insert_and_cannot_retry(
    tmp_path: Path,
) -> None:
    (
        fixture,
        spec,
        store,
        _active,
        _inventory,
        pending,
        _source_path,
        state_path,
        _singleton_path,
    ) = _prepare_valid_state_commit(tmp_path)
    state_path.write_bytes(state_path.read_bytes() + b"\n")
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_commit_state_generation(store, pending)
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_commit_state_generation(store, pending)
    reopened = store_module._test_open_governance_store(
        tmp_path,
        spec,
        plan=fixture.plan,
        authorization=fixture.authorization,
    )
    assert store_module._TEST_STORE_CONTEXTS[reopened].replay.state == (
        "state_generation_running"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "mask",
        "legal-order",
        "teacher-field",
        "split",
        "prefix",
        "identity",
        "quota-plus-one",
        "quota-minus-one",
        "singleton-count",
        "source-index",
        "game-id-type",
        "split-type",
    ],
)
def test_artifact_contract_mutations_fail_closed_and_consume_attempt(
    tmp_path: Path,
    mutation: str,
) -> None:
    _fixture, _spec, _store, active = _advance_test_store_to_active(tmp_path)
    verifier_identity = _sha("live-a-pos-verifier")
    inventory = _test_inventory_binding(verifier_identity)
    attempt_identity = store_module._TEST_ACTIVE_CONTEXTS[active].attempt_identity
    strict_referee = _test_strict_referee_binding(attempt_identity)
    source_bytes, state_bytes, singleton_bytes = _mutated_artifact_bytes(
        verifier_identity,
        attempt_identity,
        mutation,
    )
    source_path, state_path, singleton_path = _write_artifacts(
        tmp_path,
        verifier_identity,
        attempt_identity,
        source_bytes=source_bytes,
        state_bytes=state_bytes,
        singleton_bytes=singleton_bytes,
    )
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_prepare_state_generation_commit(
            active,
            inventory,
            strict_referee,
            source_games_path=source_path,
            state_split_path=state_path,
            singleton_ledger_path=singleton_path,
            timestamp_utc="2026-09-01T00:00:05Z",
            active_seconds=11,
        )

    valid_source, valid_state, valid_singleton = _artifact_bytes_for_attempt(
        verifier_identity,
        attempt_identity,
    )
    source_path.write_bytes(valid_source)
    state_path.write_bytes(valid_state)
    singleton_path.write_bytes(valid_singleton)
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_prepare_state_generation_commit(
            active,
            inventory,
            strict_referee,
            source_games_path=source_path,
            state_split_path=state_path,
            singleton_ledger_path=singleton_path,
            timestamp_utc="2026-09-01T00:00:05Z",
            active_seconds=11,
        )


def test_fully_resigned_state_blob_and_completion_still_fail_event_binding(
    tmp_path: Path,
) -> None:
    fixture, spec, store, _active, _inventory, pending, *_paths = (
        _prepare_valid_state_commit(tmp_path)
    )
    store_module._test_commit_state_generation(store, pending)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        state_payload = bytes(
            connection.execute(
                "SELECT artifact_bytes FROM artifacts WHERE role='state-split'"
            ).fetchone()[0]
        )
        state_record = json.loads(state_payload)
        state_record["sanmill_history_sha256"] = _sha("attacker-history")
        resigned_state = _resign_state_record(state_record)
        policy = store_module._test_artifact_policy(
            _test_layout(),
            maximum_source_games=1,
        )
        resigned_state_identity = store_module._recompute_state_split_blob_identity(
            resigned_state,
            policy=policy,
        )[0]

        completion_payload = bytes(
            connection.execute(
                "SELECT record_bytes FROM domain_records "
                "WHERE record_type='state-generation-completion'"
            ).fetchone()[0]
        )
        completion = json.loads(completion_payload)
        state_ref = next(
            item for item in completion["artifacts"] if item["role"] == "state-split"
        )
        state_ref["identity"] = resigned_state_identity
        state_ref["bytes_sha256"] = hashlib.sha256(resigned_state).hexdigest()
        state_ref["size_bytes"] = len(resigned_state)
        resigned_completion = canonical_json_bytes(completion)
        resigned_completion_identity = canonical_sha256(completion)

        connection.execute("DROP TRIGGER artifacts_reject_update")
        connection.execute("DROP TRIGGER domain_records_reject_update")
        connection.execute(
            "UPDATE artifacts SET artifact_identity=?, artifact_bytes=?, "
            "size_bytes=?, sha256=? WHERE role='state-split'",
            (
                resigned_state_identity,
                resigned_state,
                len(resigned_state),
                hashlib.sha256(resigned_state).hexdigest(),
            ),
        )
        connection.execute(
            "UPDATE domain_records SET record_identity=?, record_bytes=?, "
            "size_bytes=?, record_bytes_sha256=? "
            "WHERE record_type='state-generation-completion'",
            (
                resigned_completion_identity,
                resigned_completion,
                len(resigned_completion),
                hashlib.sha256(resigned_completion).hexdigest(),
            ),
        )
        for trigger_name in (
            "artifacts_reject_update",
            "domain_records_reject_update",
        ):
            trigger = next(
                statement
                for statement in store_module._DDL_STATEMENTS
                if statement.startswith(f"CREATE TRIGGER {trigger_name}")
            )
            connection.execute(trigger)
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_fully_resigned_completion_literal_drift_fails_closed(tmp_path: Path) -> None:
    fixture, spec, store, _active, _inventory, pending, *_paths = (
        _prepare_valid_state_commit(tmp_path)
    )
    store_module._test_commit_state_generation(store, pending)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        row = connection.execute(
            "SELECT record_bytes FROM domain_records "
            "WHERE record_type='state-generation-completion'"
        ).fetchone()
        completion = json.loads(bytes(row[0]))
        completion["authoritative_storage"] = "attacker-relabelled-storage"
        payload = canonical_json_bytes(completion)
        identity = canonical_sha256(completion)
        connection.execute("DROP TRIGGER domain_records_reject_update")
        connection.execute(
            "UPDATE domain_records SET record_identity=?, record_bytes=?, "
            "size_bytes=?, record_bytes_sha256=? "
            "WHERE record_type='state-generation-completion'",
            (identity, payload, len(payload), hashlib.sha256(payload).hexdigest()),
        )
        trigger = next(
            statement
            for statement in store_module._DDL_STATEMENTS
            if statement.startswith("CREATE TRIGGER domain_records_reject_update")
        )
        connection.execute(trigger)
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_fully_resigned_freeze_receipt_still_fails_frozen_event_binding(
    tmp_path: Path,
) -> None:
    fixture, spec, store, _active, _inventory, pending, *_paths = (
        _prepare_valid_state_commit(tmp_path)
    )
    completion = store_module._test_commit_state_generation(store, pending)
    store_module._test_commit_state_freeze(
        store,
        completion,
        timestamp_utc="2026-09-01T00:00:06Z",
    )
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        row = connection.execute(
            "SELECT artifact_bytes FROM artifacts WHERE role='state-freeze-receipt'"
        ).fetchone()
        receipt = json.loads(bytes(row[0]))
        receipt["authoritative_artifacts"] = "attacker-relabelled-artifacts"
        payload = canonical_json_bytes(receipt)
        identity = canonical_sha256(receipt)
        connection.execute("DROP TRIGGER artifacts_reject_update")
        connection.execute("DROP TRIGGER domain_records_reject_update")
        connection.execute(
            "UPDATE artifacts SET artifact_identity=?, artifact_bytes=?, "
            "size_bytes=?, sha256=? WHERE role='state-freeze-receipt'",
            (identity, payload, len(payload), hashlib.sha256(payload).hexdigest()),
        )
        connection.execute(
            "UPDATE domain_records SET record_identity=?, record_bytes=?, "
            "size_bytes=?, record_bytes_sha256=? WHERE record_type='state-freeze'",
            (identity, payload, len(payload), hashlib.sha256(payload).hexdigest()),
        )
        for trigger_name in (
            "artifacts_reject_update",
            "domain_records_reject_update",
        ):
            trigger = next(
                statement
                for statement in store_module._DDL_STATEMENTS
                if statement.startswith(f"CREATE TRIGGER {trigger_name}")
            )
            connection.execute(trigger)
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_incomplete_artifact_rows_cannot_appear_before_completion(
    tmp_path: Path,
) -> None:
    fixture, spec, _store, _active = _advance_test_store_to_active(tmp_path)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    payload = canonical_json_bytes({"unexpected": "early artifact"})
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        connection.execute(
            "INSERT INTO artifacts(role, artifact_identity, artifact_bytes, "
            "size_bytes, sha256) VALUES (?, ?, ?, ?, ?)",
            (
                "source-games",
                _sha("early-artifact"),
                payload,
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            ),
        )
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_corrupt_database_header_fails_closed(tmp_path: Path) -> None:
    fixture, spec, _store = _initialize_test_store(tmp_path)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    database.write_bytes(b"not-a-sqlite-database")
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_sqlite_dynamic_type_spoof_is_wrapped_as_store_contract_failure(
    tmp_path: Path,
) -> None:
    fixture, spec, store, _active, _inventory, pending, *_paths = (
        _prepare_valid_state_commit(tmp_path)
    )
    store_module._test_commit_state_generation(store, pending)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        connection.execute("DROP TRIGGER artifacts_reject_update")
        connection.execute(
            "UPDATE artifacts SET artifact_bytes=?, size_bytes=?, sha256=? "
            "WHERE role='state-split'",
            ("text-not-blob", 13, hashlib.sha256(b"text-not-blob").hexdigest()),
        )
        trigger = next(
            statement
            for statement in store_module._DDL_STATEMENTS
            if statement.startswith("CREATE TRIGGER artifacts_reject_update")
        )
        connection.execute(trigger)
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


def test_existing_governance_namespace_is_not_reused_or_modified(
    tmp_path: Path,
) -> None:
    fixture = governance._issue_test_governance_fixture()
    spec = build_governance_store_spec(
        fixture.plan,
        readiness_identity=fixture.authorization["readiness_identity"],
        output_root_identity=_sha("existing-namespace-output"),
    )
    namespace = tmp_path / "governance"
    namespace.mkdir()
    sentinel = namespace / "owner-data.txt"
    sentinel.write_bytes(b"preserve-me")
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_initialize_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
            bootstrap_ledger_bytes=governance.encode_governance_ledger(
                fixture.events[:3]
            ),
        )
    assert sentinel.read_bytes() == b"preserve-me"
    assert not (namespace / "classical-a-pos-controller.sqlite3").exists()


def test_governance_database_hardlink_alias_fails_closed(tmp_path: Path) -> None:
    fixture, spec, _store = _initialize_test_store(tmp_path)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    os.link(database, tmp_path / "database-hardlink.sqlite3")
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan=fixture.plan,
            authorization=fixture.authorization,
        )


@pytest.mark.parametrize(
    "path_attack",
    ["governance", "outside", "hardlink", "lexical-alias"],
)
def test_state_artifact_paths_must_be_unaliased_plan_owned_files(
    tmp_path: Path,
    path_attack: str,
) -> None:
    _fixture, _spec, _store, active = _advance_test_store_to_active(tmp_path)
    verifier_identity = _sha("live-a-pos-verifier")
    inventory = _test_inventory_binding(verifier_identity)
    attempt_identity = store_module._TEST_ACTIVE_CONTEXTS[active].attempt_identity
    strict_referee = _test_strict_referee_binding(attempt_identity)
    source_bytes, state_bytes, singleton_bytes = _artifact_bytes_for_attempt(
        verifier_identity,
        attempt_identity,
    )
    source_path, state_path, singleton_path = _write_artifacts(
        tmp_path,
        verifier_identity,
        attempt_identity,
    )
    if path_attack == "governance":
        source_path = tmp_path / "governance" / "source-games.jsonl"
        source_path.write_bytes(source_bytes)
    elif path_attack == "outside":
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.mkdir()
        source_path = outside / "source-games.jsonl"
        source_path.write_bytes(source_bytes)
    elif path_attack == "hardlink":
        hardlink = tmp_path / "source-games-hardlink.jsonl"
        os.link(source_path, hardlink)
        source_path = hardlink
    else:
        (tmp_path / "alias-segment").mkdir()
        source_path = tmp_path / "alias-segment" / ".." / source_path.name
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_prepare_state_generation_commit(
            active,
            inventory,
            strict_referee,
            source_games_path=source_path,
            state_split_path=state_path,
            singleton_ledger_path=singleton_path,
            timestamp_utc="2026-09-01T00:00:05Z",
            active_seconds=11,
        )


def test_production_artifact_policy_is_not_downgradable_to_test_layout() -> None:
    policy = store_module._PRODUCTION_POLICY
    layout = _mutable(policy.layout)
    assert policy.domain == "production"
    assert policy.total_states == 16_384
    assert policy.maximum_source_games == 1_024
    assert policy.require_complete_game_block is True
    assert layout == store_module.FROZEN_CORPUS_LAYOUT.to_dict()
    cells = [
        layout[split][stratum][colour]
        for split in ("train", "dev")
        for stratum in ("placement", "movement", "flying")
        for colour in ("W", "B")
    ]
    assert len(cells) == 12
    assert sum(cells) == 16_384


def test_public_store_apis_reject_test_domain_mappings_and_rogue_capabilities(
    tmp_path: Path,
) -> None:
    _fixture, _spec, test_store = _initialize_test_store(tmp_path)
    with pytest.raises(GovernanceStoreContractError):
        store_module.commit_authorization_consumption(test_store, {})
    with pytest.raises(GovernanceStoreContractError):
        store_module.commit_operation_reservation(test_store, {})

    rogue_store = object.__new__(DurableGovernanceStore)
    rogue_pending = object.__new__(PendingStateGenerationCommit)
    with pytest.raises(GovernanceStoreContractError):
        store_module.commit_state_generation(rogue_store, rogue_pending)
    assert not hasattr(store_module, "issue_production_a_pos_inventory_binding")
    assert not hasattr(store_module, "issue_production_strict_referee_binding")

    strict_root = tmp_path / "strict-domain"
    strict_root.mkdir()
    _fixture, _spec, _test_store, active = _advance_test_store_to_active(strict_root)
    verifier_identity = _sha("strict-domain-verifier")
    inventory = _test_inventory_binding(verifier_identity)
    attempt_identity = store_module._TEST_ACTIVE_CONTEXTS[active].attempt_identity
    strict_referee = _test_strict_referee_binding(attempt_identity)
    source_path, state_path, singleton_path = _write_artifacts(
        strict_root,
        verifier_identity,
        attempt_identity,
    )
    with pytest.raises(GovernanceStoreContractError):
        store_module.prepare_state_generation_commit(
            active,
            inventory,
            strict_referee,
            source_games_path=source_path,
            state_split_path=state_path,
            singleton_ledger_path=singleton_path,
            timestamp_utc="2026-09-01T00:00:05Z",
            active_seconds=11,
        )
