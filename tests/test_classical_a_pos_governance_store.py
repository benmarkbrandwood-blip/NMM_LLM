from __future__ import annotations

import inspect
import copy
import hashlib
import json
import os
import pickle
import sqlite3
import weakref
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import learned_ai.training.classical_a_pos_governance as governance
import learned_ai.training.classical_a_pos_corpus as corpus_module
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


def test_c5a_store_completion_freeze_and_meta_contracts_are_v2() -> None:
    assert store_module._SPEC_SCHEMA == "nmm.classical-a-pos-governance-store-spec.v2"
    assert (
        store_module._STORE_META_SCHEMA
        == "nmm.classical-a-pos-governance-store-meta.v2"
    )
    assert (
        store_module._COMPLETION_SCHEMA
        == "nmm.classical-a-pos-state-generation-completion.v2"
    )
    assert (
        store_module._FREEZE_RECEIPT_SCHEMA
        == "nmm.classical-a-pos-durable-state-freeze.v2"
    )
    assert store_module._COMPLETION_RUNTIME_KEYS == {
        "host_preflight_identity",
        "state_generator_session_identity",
        "strict_referee_binding_identity",
        "a_pos_inventory_binding_identity",
        "runtime_evidence_stream_identity",
        "resource_snapshot_before_identity",
        "resource_snapshot_after_identity",
        "resource_stability_identity",
    }
    assert {
        "runtime_evidence_stream_identity",
        "resource_stability_identity",
    } < store_module._FREEZE_GOVERNANCE_KEYS


def test_c5a_public_store_entrypoints_accept_only_exact_production_permits() -> None:
    initialize = inspect.signature(store_module.initialize_governance_store).parameters
    open_store = inspect.signature(store_module.open_governance_store).parameters
    assert tuple(initialize) == (
        "output_root",
        "spec",
        "plan_permit",
        "authorization_permit",
    )
    assert tuple(open_store) == (
        "output_root",
        "spec",
        "plan_permit",
        "authorization_permit",
    )
    assert "plan" not in open_store
    assert "authorization" not in open_store


def test_c5a_runtime_completion_is_opaque_unissued_and_restore_is_public() -> None:
    completion_type = store_module.ProductionStateGenerationRuntimeCompletion
    assert "ProductionStateGenerationRuntimeCompletion" in store_module.__all__
    assert "restore_durable_state_freeze" in store_module.__all__
    with pytest.raises((TypeError, GovernanceStoreContractError)):
        completion_type(object())
    rogue = object.__new__(completion_type)
    assert not hasattr(rogue, "__dict__")
    with pytest.raises(AttributeError):
        rogue.context = object()  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        copy.copy(rogue)
    with pytest.raises(TypeError):
        pickle.dumps(rogue)
    assert not hasattr(
        store_module,
        "issue_production_state_generation_runtime_completion",
    )


def test_c5a_runtime_preflight_and_bound_attempt_are_opaque_and_unissued() -> None:
    for capability_type in (
        store_module.ProductionStateGenerationRuntimePreflight,
        store_module.RuntimeBoundStateGenerationAttempt,
    ):
        assert "__dict__" not in capability_type.__dict__
        with pytest.raises((TypeError, GovernanceStoreContractError)):
            capability_type(object())
        rogue = object.__new__(capability_type)
        with pytest.raises(AttributeError):
            rogue.context = object()  # type: ignore[attr-defined]
        with pytest.raises(TypeError):
            copy.copy(rogue)
        with pytest.raises(TypeError):
            pickle.dumps(rogue)
    assert not hasattr(
        store_module, "issue_production_state_generation_runtime_preflight"
    )
    assert not hasattr(store_module, "bind_production_state_generation_attempt")


def test_c5a_postcommit_runtime_before_error_is_private_and_fatal() -> None:
    error_type = store_module._DurableRuntimeBeforePostCommitError
    assert issubclass(error_type, GovernanceStoreContractError)
    assert "_DurableRuntimeBeforePostCommitError" not in store_module.__all__


def test_c5a_runtime_stream_allowlist_is_before_after_stability_in_order() -> None:
    assert store_module._RUNTIME_STREAM_KIND == "state-generation-runtime"
    assert store_module._RUNTIME_RECORD_TYPES == (
        "state-generation-resource-before",
        "state-generation-resource-after",
        "state-generation-resource-stability",
    )
    assert (
        store_module._RESOURCE_SNAPSHOT_SCHEMA
        == "nmm.classical-a-pos-state-generation-resource-snapshot.v1"
    )
    assert store_module._RUNTIME_RECORD_SCHEMAS == {
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
    assert (
        store_module._RUNTIME_STREAM_SCHEMA
        == "nmm.classical-a-pos-state-generation-runtime-stream.v1"
    )
    assert store_module._RESOURCE_SNAPSHOT_KEYS == {
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
    assert store_module._HOST_SNAPSHOT_KEYS == {
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
    assert store_module._PATH_REGISTRY_SNAPSHOT_KEYS == {
        "schema_version",
        "registry_role",
        "file_sha256",
        "canonical_object_identity",
        "required_lookup_keys",
        "resolved_path_identities",
    }
    assert store_module._REPOSITORY_SNAPSHOT_KEYS == {
        "schema_version",
        "root_identity",
        "head_commit",
        "head_tree",
        "status_identity",
        "implementation_files_identity",
        "allowed_untracked_roots",
    }
    assert store_module._OUTPUT_ROOT_SNAPSHOT_KEYS == {
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
    assert store_module._MALOM_SNAPSHOT_KEYS == {
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
    assert store_module._SANMILL_SNAPSHOT_KEYS == {
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
    assert store_module._GENERATOR_SNAPSHOT_KEYS == {
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
    assert store_module._RUNTIME_RECORD_CONTEXT_KEYS == {
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
    assert store_module._STABILITY_COMPARISON_FIELDS == (
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


def test_c5a_internal_store_helpers_remove_raw_records_and_bootstrap_bytes() -> None:
    initialize = inspect.signature(
        store_module._test_initialize_governance_store
    ).parameters
    open_store = inspect.signature(store_module._test_open_governance_store).parameters
    assert tuple(initialize) == (
        "output_root",
        "spec",
        "plan_permit",
        "authorization_permit",
    )
    assert tuple(open_store) == (
        "output_root",
        "spec",
        "plan_permit",
        "authorization_permit",
    )
    for forbidden in ("plan", "authorization", "bootstrap_ledger_bytes"):
        assert forbidden not in initialize
        assert forbidden not in open_store


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
        store_module.ProductionStateGenerationRuntimePreflight,
        ActiveStateGenerationAttempt,
        store_module.RuntimeBoundStateGenerationAttempt,
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
        )


def test_c5a_bootstrap_failure_spends_claim_and_cannot_mutate_second_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = governance._issue_test_governance_fixture()
    spec = build_governance_store_spec(
        fixture.plan,
        readiness_identity=fixture.authorization["readiness_identity"],
        output_root_identity=_sha("failed-bootstrap-output-root"),
    )
    original_create = store_module._create_store_database

    def reject_first_mutation(*_args: Any, **_kwargs: Any) -> None:
        raise GovernanceStoreContractError("injected first SQLite mutation failure")

    monkeypatch.setattr(
        store_module,
        "_create_store_database",
        reject_first_mutation,
    )
    first_root = tmp_path / "first"
    first_root.mkdir()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_initialize_governance_store(
            first_root,
            spec,
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
        )

    monkeypatch.setattr(store_module, "_create_store_database", original_create)
    second_root = tmp_path / "second"
    second_root.mkdir()
    sentinel = second_root / "owner-sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_initialize_governance_store(
            second_root,
            spec,
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (second_root / "governance").exists()


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
    store = store_module._test_initialize_governance_store(
        tmp_path,
        spec,
        plan_permit=fixture.plan_permit,
        authorization_permit=fixture.authorization_permit,
    )
    return fixture, spec, store


def _advance_test_store_to_reserved(tmp_path: Path):
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
    return fixture, spec, store, operation_permit


def _advance_test_store_to_active(tmp_path: Path):
    fixture, spec, store, operation_permit = _advance_test_store_to_reserved(tmp_path)
    active = store_module._test_begin_state_generation_attempt(
        store,
        operation_permit,
    )
    return fixture, spec, store, active


def _advance_test_store_to_bound(tmp_path: Path):
    fixture, spec, store, active = _advance_test_store_to_active(tmp_path)
    preflight = store_module._issue_test_state_generation_runtime_preflight(
        store,
        active,
    )
    bound = store_module._test_bind_state_generation_attempt(
        preflight,
        store,
        active,
    )
    return fixture, spec, store, active, bound


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
    fixture, spec, store, active, bound = _advance_test_store_to_bound(tmp_path)
    verifier_identity = _sha("live-a-pos-verifier")
    inventory = _test_inventory_binding(verifier_identity)
    attempt_identity = store_module._TEST_ACTIVE_CONTEXTS[active].attempt_identity
    strict_referee = _test_strict_referee_binding(attempt_identity)
    source_path, state_path, singleton_path = _write_artifacts(
        tmp_path,
        verifier_identity,
        attempt_identity,
    )
    runtime_completion = store_module._issue_test_state_generation_runtime_completion(
        bound
    )
    pending = store_module._test_prepare_state_generation_commit(
        bound,
        inventory,
        strict_referee,
        runtime_completion,
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


def _assert_exact_lone_runtime_before(database: Path) -> str:
    connection = sqlite3.connect(database)
    try:
        artifacts, records = store_module._verify_blob_tables(connection)
        rows = connection.execute(
            "SELECT sequence, record_type, previous_record_identity, "
            "record_identity FROM domain_records "
            "WHERE stream_kind='state-generation-runtime'"
        ).fetchall()
    finally:
        connection.close()
    assert artifacts == {}
    assert set(records) == {"state-generation-resource-before"}
    assert len(rows) == 1
    sequence, record_type, previous, record_identity = rows[0]
    assert sequence == 0
    assert record_type == "state-generation-resource-before"
    assert previous is None
    assert records[record_type][0] == record_identity
    return record_identity


def test_c5a_store_runtime_is_none_until_explicit_bind_and_after_nonrunning_open(
    tmp_path: Path,
) -> None:
    fixture, spec, store = _initialize_test_store(tmp_path)
    assert store_module._TEST_STORE_CONTEXTS[store].runtime is None
    reopened = store_module._test_open_governance_store(
        tmp_path,
        spec,
        plan_permit=fixture.plan_permit,
        authorization_permit=fixture.authorization_permit,
    )
    assert store_module._TEST_STORE_CONTEXTS[reopened].runtime is None

    generated_root = tmp_path / "generated"
    generated_root.mkdir()
    prepared = _prepare_valid_state_commit(generated_root)
    generated_fixture, generated_spec, generated_store, pending = (
        prepared[0],
        prepared[1],
        prepared[2],
        prepared[5],
    )
    store_module._test_commit_state_generation(generated_store, pending)
    generated_reopened = store_module._test_open_governance_store(
        generated_root,
        generated_spec,
        plan_permit=generated_fixture.plan_permit,
        authorization_permit=generated_fixture.authorization_permit,
    )
    generated_context = store_module._TEST_STORE_CONTEXTS[generated_reopened]
    assert generated_context.replay.state == "state_generated"
    assert generated_context.runtime is None
    database = generated_root / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database)
    try:
        assert connection.execute(
            "SELECT sequence, record_type FROM domain_records "
            "WHERE stream_kind='state-generation-runtime' ORDER BY sequence"
        ).fetchall() == [
            (0, "state-generation-resource-before"),
            (1, "state-generation-resource-after"),
            (2, "state-generation-resource-stability"),
        ]
    finally:
        connection.close()


def test_c5a_postcommit_replay_failure_preserves_seq0_and_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, spec, store, active = _advance_test_store_to_active(tmp_path)
    preflight = store_module._issue_test_state_generation_runtime_preflight(
        store,
        active,
    )
    real_replay = store_module._read_and_replay
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"

    def fail_after_runtime_before_commit(*args: Any, **kwargs: Any):
        replay = real_replay(*args, **kwargs)
        connection = sqlite3.connect(database)
        try:
            count = connection.execute(
                "SELECT COUNT(*) FROM domain_records "
                "WHERE stream_kind='state-generation-runtime'"
            ).fetchone()[0]
        finally:
            connection.close()
        if count == 1:
            raise GovernanceStoreContractError("injected post-COMMIT replay failure")
        return replay

    monkeypatch.setattr(
        store_module, "_read_and_replay", fail_after_runtime_before_commit
    )
    with pytest.raises(
        store_module._DurableRuntimeBeforePostCommitError,
        match="fatal post-commit runtime-before capability loss; no resume or retry",
    ):
        store_module._test_bind_state_generation_attempt(preflight, store, active)
    monkeypatch.setattr(store_module, "_read_and_replay", real_replay)

    before_identity = _assert_exact_lone_runtime_before(database)
    store_context = store_module._TEST_STORE_CONTEXTS[store]
    active_context = store_module._TEST_ACTIVE_CONTEXTS[active]
    preflight_context = store_module._TEST_RUNTIME_PREFLIGHT_CONTEXTS[preflight]
    assert store_context.runtime is None
    assert active_context.bind_consumed is True
    assert active_context.spent is True
    assert preflight_context.spent is True
    assert all(
        context.active is not active
        for context in store_module._TEST_BOUND_ATTEMPT_CONTEXTS.values()
    )
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_bind_state_generation_attempt(preflight, store, active)
    with pytest.raises(
        store_module._DurableRuntimeBeforePostCommitError,
        match="fatal post-commit runtime-before capability loss; no resume or retry",
    ):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
        )

    with pytest.raises(GovernanceStoreContractError):
        store_module._issue_test_state_generation_runtime_completion(
            {"before_record_identity": before_identity}
        )
    rogue_bound = object.__new__(store_module._TestRuntimeBoundStateGenerationAttempt)
    rogue_completion = object.__new__(
        store_module._TestStateGenerationRuntimeCompletion
    )
    with pytest.raises(GovernanceStoreContractError):
        store_module._issue_test_state_generation_runtime_completion(rogue_bound)
    verifier_identity = _sha("postcommit-loss-verifier")
    inventory = _test_inventory_binding(verifier_identity)
    strict_referee = _test_strict_referee_binding(active_context.attempt_identity)
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_prepare_state_generation_commit(
            rogue_bound,
            inventory,
            strict_referee,
            rogue_completion,
            source_games_path=tmp_path / "must-not-read-source.jsonl",
            state_split_path=tmp_path / "must-not-read-state.jsonl",
            singleton_ledger_path=tmp_path / "must-not-read-singletons.json",
            timestamp_utc="2026-09-01T00:00:05Z",
            active_seconds=0,
        )
    for forbidden in (
        "execute_state_generation",
        "resume_state_generation_from_runtime_before",
        "restore_runtime_bound_state_generation_attempt",
    ):
        assert not hasattr(store_module, forbidden)


def test_c5a_bound_registration_failure_is_postcommit_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, spec, store, active = _advance_test_store_to_active(tmp_path)
    preflight = store_module._issue_test_state_generation_runtime_preflight(
        store,
        active,
    )
    original_registry = store_module._TEST_BOUND_ATTEMPT_CONTEXTS

    class _FailingBoundRegistry(weakref.WeakKeyDictionary):
        def __setitem__(self, key: object, value: object) -> None:
            super().__setitem__(key, value)
            raise RuntimeError("injected WeakKey registration failure")

    failing_registry = _FailingBoundRegistry()
    monkeypatch.setattr(
        store_module,
        "_TEST_BOUND_ATTEMPT_CONTEXTS",
        failing_registry,
    )
    with pytest.raises(
        store_module._DurableRuntimeBeforePostCommitError,
        match="fatal post-commit runtime-before capability loss; no resume or retry",
    ):
        store_module._test_bind_state_generation_attempt(preflight, store, active)
    monkeypatch.setattr(
        store_module,
        "_TEST_BOUND_ATTEMPT_CONTEXTS",
        original_registry,
    )

    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    _assert_exact_lone_runtime_before(database)
    assert not failing_registry
    assert store_module._TEST_STORE_CONTEXTS[store].runtime is None
    assert store_module._TEST_ACTIVE_CONTEXTS[active].spent is True
    assert store_module._TEST_RUNTIME_PREFLIGHT_CONTEXTS[preflight].spent is True
    with pytest.raises(
        store_module._DurableRuntimeBeforePostCommitError,
        match="fatal post-commit runtime-before capability loss; no resume or retry",
    ):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
    fixture, spec, store, active, bound = _advance_test_store_to_bound(tmp_path)
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

    runtime_completion = store_module._issue_test_state_generation_runtime_completion(
        bound
    )
    pending_state = store_module._test_prepare_state_generation_commit(
        bound,
        inventory_binding,
        strict_referee,
        runtime_completion,
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
        plan_permit=fixture.plan_permit,
        authorization_permit=fixture.authorization_permit,
    )
    assert store_module._TEST_STORE_CONTEXTS[reopened].replay.state == "state_frozen"
    with pytest.raises(GovernanceStoreContractError):
        store_module.open_governance_store(
            tmp_path,
            spec,
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
            "record_bytes_sha256 FROM domain_records "
            "WHERE stream_kind='controller' ORDER BY sequence"
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
                "WHERE stream_kind='controller' "
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
                    "SELECT record_bytes FROM domain_records "
                    "WHERE stream_kind='controller' AND sequence=0"
                ).fetchone()[0]
            )
        )
        record["stream_identity"] = _sha("wrong-controller-stream")
        payload = canonical_json_bytes(record)
        identity = hashlib.sha256(payload).hexdigest()
        connection.execute("DROP TRIGGER domain_records_reject_update")
        connection.execute(
            "UPDATE domain_records SET stream_identity=?, record_identity=?, "
            "record_bytes=?, size_bytes=?, record_bytes_sha256=? "
            "WHERE stream_kind='controller' AND sequence=0",
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
                "SELECT record_bytes FROM domain_records "
                "WHERE stream_kind='controller' AND sequence=0"
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
                "WHERE stream_kind='controller' AND sequence=0",
                (payload, len(payload)),
            )
        else:
            connection.execute(
                "UPDATE domain_records SET record_identity=?, record_bytes=?, "
                "size_bytes=?, record_bytes_sha256=? "
                "WHERE stream_kind='controller' AND sequence=0",
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
            "record_bytes_sha256 FROM domain_records "
            "WHERE stream_kind='controller' AND sequence=0"
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
    pending_context = store_module._TEST_PENDING_CONTEXTS[pending]
    bound = pending_context.active
    strict_referee = pending_context.strict_referee_binding
    runtime_completion = pending_context.runtime_completion
    for capability in (
        store,
        active,
        bound,
        inventory,
        strict_referee,
        runtime_completion,
        pending,
    ):
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
            bound,
            inventory,
            strict_referee,
            runtime_completion,
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
        plan_permit=fixture.plan_permit,
        authorization_permit=fixture.authorization_permit,
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
        plan_permit=fixture.plan_permit,
        authorization_permit=fixture.authorization_permit,
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

    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
        )
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM domain_records "
            "WHERE stream_kind='state-generation-runtime'"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM domain_records WHERE stream_kind='controller'"
        ).fetchone() == (0,)
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
        plan_permit=fixture.plan_permit,
        authorization_permit=fixture.authorization_permit,
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
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
        )
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM domain_records "
            "WHERE stream_kind='state-generation-runtime'"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM domain_records WHERE stream_kind='controller'"
        ).fetchone() == (0,)
    finally:
        connection.close()


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
    _fixture, _spec, _store, active, bound = _advance_test_store_to_bound(tmp_path)
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
    runtime_completion = store_module._issue_test_state_generation_runtime_completion(
        bound
    )
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_prepare_state_generation_commit(
            bound,
            inventory,
            strict_referee,
            runtime_completion,
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
            bound,
            inventory,
            strict_referee,
            runtime_completion,
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
        )


def test_c5a_begin_is_unbound_and_persists_no_runtime_evidence(
    tmp_path: Path,
) -> None:
    _fixture, _spec, store, _active = _advance_test_store_to_active(tmp_path)
    assert store_module._TEST_STORE_CONTEXTS[store].runtime is None
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT stream_kind, sequence, record_type FROM domain_records "
            "WHERE stream_kind='state-generation-runtime' ORDER BY sequence"
        ).fetchall()
    finally:
        connection.close()
    assert rows == []


def test_c5a_explicit_test_preflight_bind_persists_runtime_before(
    tmp_path: Path,
) -> None:
    _fixture, _spec, _store, _active, _bound = _advance_test_store_to_bound(tmp_path)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT stream_kind, sequence, record_type FROM domain_records "
            "WHERE stream_kind='state-generation-runtime' ORDER BY sequence"
        ).fetchall()
    finally:
        connection.close()
    assert rows == [("state-generation-runtime", 0, "state-generation-resource-before")]


def test_c5a_completion_appends_runtime_after_stability_and_controller_atomically(
    tmp_path: Path,
) -> None:
    prepared = _prepare_valid_state_commit(tmp_path)
    store = prepared[2]
    pending = prepared[5]
    store_module._test_commit_state_generation(store, pending)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database)
    try:
        runtime_rows = connection.execute(
            "SELECT sequence, record_type FROM domain_records "
            "WHERE stream_kind='state-generation-runtime' ORDER BY sequence"
        ).fetchall()
        controller_rows = connection.execute(
            "SELECT sequence, record_type FROM domain_records "
            "WHERE stream_kind='controller' ORDER BY sequence"
        ).fetchall()
    finally:
        connection.close()
    assert runtime_rows == [
        (0, "state-generation-resource-before"),
        (1, "state-generation-resource-after"),
        (2, "state-generation-resource-stability"),
    ]
    assert controller_rows == [(0, "state-generation-completion")]


def test_c5a_runtime_records_and_completion_have_exact_identity_chain(
    tmp_path: Path,
) -> None:
    prepared = _prepare_valid_state_commit(tmp_path)
    store = prepared[2]
    pending = prepared[5]
    store_module._test_commit_state_generation(store, pending)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT record_identity, record_bytes FROM domain_records "
            "WHERE stream_kind='state-generation-runtime' ORDER BY sequence"
        ).fetchall()
        completion_bytes = bytes(
            connection.execute(
                "SELECT record_bytes FROM domain_records "
                "WHERE record_type='state-generation-completion'"
            ).fetchone()[0]
        )
    finally:
        connection.close()

    assert len(rows) == 3
    before, after, stability = [json.loads(bytes(row[1])) for row in rows]
    before_keys = (
        store_module._DOMAIN_ENVELOPE_KEYS
        | store_module._RUNTIME_RECORD_CONTEXT_KEYS
        | {"snapshot", "snapshot_identity"}
    )
    stability_keys = (
        store_module._DOMAIN_ENVELOPE_KEYS
        | store_module._RUNTIME_RECORD_CONTEXT_KEYS
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
    assert set(before) == before_keys
    assert set(after) == before_keys
    assert set(stability) == stability_keys
    assert before["sequence"] == 0
    assert before["previous_record_identity"] is None
    assert after["sequence"] == 1
    assert after["previous_record_identity"] == rows[0][0]
    assert stability["sequence"] == 2
    assert stability["previous_record_identity"] == rows[1][0]
    assert stability["before_record_identity"] == rows[0][0]
    assert stability["after_record_identity"] == rows[1][0]
    assert stability["comparison_fields"] == list(
        store_module._STABILITY_COMPARISON_FIELDS
    )
    assert stability["differences"] == []
    assert stability["stable"] is True
    before_snapshot_body = {
        key: value
        for key, value in before["snapshot"].items()
        if key != "snapshot_identity"
    }
    after_snapshot_body = {
        key: value
        for key, value in after["snapshot"].items()
        if key != "snapshot_identity"
    }
    assert before["snapshot_identity"] == before["snapshot"]["snapshot_identity"]
    assert after["snapshot_identity"] == after["snapshot"]["snapshot_identity"]
    assert before["snapshot_identity"] == canonical_sha256(before_snapshot_body)
    assert after["snapshot_identity"] == canonical_sha256(after_snapshot_body)
    assert before["snapshot"]["stage"] == "before-execution"
    assert after["snapshot"]["stage"] == "after-execution"
    before_generator = before["snapshot"]["generator"]
    frozen_generator = corpus_module._generator_contract(
        before_generator["initial_policy_state_sha256"]
    )
    assert before_generator["model_config"] == {
        key: frozen_generator[key]
        for key in (
            "policy",
            "policy_hidden",
            "value_hidden",
            "dropout",
            "model_init_seed",
        )
    }
    assert before_generator["sampler_config"] == {
        key: frozen_generator[key]
        for key in (
            "sampling",
            "temperature",
            "cpu_generator_seed",
            "initial_state",
            "candidate_colours",
            "max_games",
            "informative_only",
        )
    }
    assert before_generator["contract_identity"] == canonical_sha256(frozen_generator)
    assert (
        before_generator["current_rng_state_identity"]
        == before_generator["initial_rng_state_identity"]
    )
    stream_body = {
        "schema_version": store_module._RUNTIME_STREAM_SCHEMA,
        "stream_kind": store_module._RUNTIME_STREAM_KIND,
        "experiment_id": before["experiment_id"],
        "proposal_identity": before["proposal_identity"],
        "profile_identity": before["profile_identity"],
        "store_spec_identity": before["store_spec_identity"],
        "plan_identity": before["plan_identity"],
        "readiness_identity": before["readiness_identity"],
        "authorization_identity": before["authorization_identity"],
        "authorization_consumption_identity": before[
            "authorization_consumption_identity"
        ],
        "attempt_identity": before["attempt_identity"],
        "reservation_event_identity": before["reservation_event_identity"],
    }
    assert before["stream_identity"] == canonical_sha256(stream_body)
    assert after["stream_identity"] == before["stream_identity"]
    assert stability["stream_identity"] == before["stream_identity"]
    assert stability["before_snapshot_identity"] == before["snapshot_identity"]
    assert stability["after_snapshot_identity"] == after["snapshot_identity"]

    completion = json.loads(completion_bytes)
    assert completion["runtime_evidence_stream_identity"] == before["stream_identity"]
    assert (
        completion["resource_snapshot_before_identity"] == before["snapshot_identity"]
    )
    assert completion["resource_snapshot_after_identity"] == after["snapshot_identity"]
    assert completion["resource_stability_identity"] == rows[2][0]


def _resign_runtime_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in snapshot.items() if key != "snapshot_identity"}
    snapshot["snapshot_identity"] = canonical_sha256(body)
    return snapshot


def _after_snapshot_for_bound(bound: object) -> dict[str, Any]:
    context = store_module._TEST_BOUND_ATTEMPT_CONTEXTS[bound]
    snapshot = _mutable(context.resource_snapshot_before)
    snapshot["stage"] = "after-execution"
    snapshot["observed_at_utc"] = "2026-09-01T00:00:05Z"
    snapshot["generator"]["current_rng_state_identity"] = _sha("after-rng")
    return _resign_runtime_snapshot(snapshot)


def _before_snapshot_for_active(active: object) -> dict[str, Any]:
    active_context = store_module._TEST_ACTIVE_CONTEXTS[active]
    store_context = store_module._TEST_STORE_CONTEXTS[active_context.store]
    return store_module._test_resource_snapshot(
        store_context.spec,
        attempt_identity=active_context.attempt_identity,
        stage="before-execution",
        observed_at_utc=store_context.replay.events[-1]["timestamp_utc"],
    )


_BOUND_SNAPSHOT_IDENTITY_PATHS = (
    ("host", "machine_identity"),
    ("host", "python_executable_sha256"),
    ("path_registry", "file_sha256"),
    ("path_registry", "canonical_object_identity"),
    ("path_registry", "resolved_path_identities", "malom_db_path"),
    ("path_registry", "resolved_path_identities", "sanmill_training_checkout"),
    ("path_registry", "resolved_path_identities", "classical_a_pos_output_root"),
    ("repository", "root_identity"),
    ("repository", "status_identity"),
    ("repository", "implementation_files_identity"),
    ("output_root", "canonical_path_identity"),
    ("output_root", "root_identity"),
    ("output_root", "volume_identity"),
    ("output_root", "file_identity"),
    ("output_root", "governance_database_identity"),
    ("output_root", "artifact_namespace_identity"),
    ("malom", "path_identity"),
    ("malom", "manifest_file_sha256"),
    ("malom", "manifest_sha256"),
    ("malom", "content_sha256"),
    ("malom", "component_metadata_identity"),
    ("malom", "full_hash_evidence_identity"),
    ("malom", "oracle_implementation_identity"),
    ("sanmill", "checkout_path_identity"),
    ("sanmill", "installation_identity"),
    ("sanmill", "runtime_identity"),
    ("sanmill", "binary_sha256"),
    ("sanmill", "license_sha256"),
    ("sanmill", "referee_implementation_identity"),
    ("generator", "contract_identity"),
    ("generator", "model_implementation_identity"),
    ("generator", "encoder_implementation_identity"),
    ("generator", "initial_policy_state_sha256"),
    ("generator", "initial_rng_state_identity"),
    ("generator", "current_rng_state_identity"),
)


@pytest.mark.parametrize("path", _BOUND_SNAPSHOT_IDENTITY_PATHS)
def test_c5a_runtime_snapshot_rejects_every_zero_bound_identity(
    tmp_path: Path,
    path: tuple[str, ...],
) -> None:
    _fixture, _spec, store, active = _advance_test_store_to_active(tmp_path)
    snapshot = _before_snapshot_for_active(active)
    target: dict[str, Any] = snapshot
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = "0" * 64
    _resign_runtime_snapshot(snapshot)

    with pytest.raises(GovernanceStoreContractError):
        store_module._issue_test_state_generation_runtime_preflight(
            store,
            active,
            resource_snapshot_before=snapshot,
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ((owner, field), replacement)
        for owner, fields in (
            ("repository", ("head_commit", "head_tree")),
            ("sanmill", ("commit", "tree")),
        )
        for field in fields
        for replacement in ("A" * 40, "a" * 39, "g" * 40, "0" * 40)
    ],
)
def test_c5a_runtime_snapshot_rejects_malformed_repository_and_sanmill_oids(
    tmp_path: Path,
    path: tuple[str, str],
    replacement: str,
) -> None:
    _fixture, _spec, store, active = _advance_test_store_to_active(tmp_path)
    snapshot = _before_snapshot_for_active(active)
    snapshot[path[0]][path[1]] = replacement
    _resign_runtime_snapshot(snapshot)

    with pytest.raises(GovernanceStoreContractError):
        store_module._issue_test_state_generation_runtime_preflight(
            store,
            active,
            resource_snapshot_before=snapshot,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("host_preflight_identity", _sha("wrong-host-preflight")),
        ("host_preflight_identity", "0" * 64),
        ("state_generator_session_identity", "0" * 64),
    ],
)
def test_c5a_runtime_bind_rechecks_preflight_identities_before_sqlite_write(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    _fixture, _spec, store, active = _advance_test_store_to_active(tmp_path)
    preflight = store_module._issue_test_state_generation_runtime_preflight(
        store,
        active,
    )
    context = store_module._TEST_RUNTIME_PREFLIGHT_CONTEXTS[preflight]
    setattr(context, field, replacement)

    with pytest.raises(GovernanceStoreContractError):
        store_module._test_bind_state_generation_attempt(preflight, store, active)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM domain_records "
            "WHERE stream_kind='state-generation-runtime'"
        ).fetchone() == (0,)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "attack",
    [
        "top-missing",
        "top-unknown",
        "host-missing",
        "host-unknown",
        "host-custom-mapping",
        "path-required-tuple",
        "repository-allowed-set",
        "host-nonfinite",
        "component-count-bool",
        "component-count-float",
        "checkout-clean-int",
        "cuda-initialized-int",
        "wrong-fixed-lookup",
        "wrong-model-config",
        "wrong-sampler-config",
    ],
)
def test_c5a_resigned_runtime_snapshot_contract_attacks_fail_closed(
    tmp_path: Path,
    attack: str,
) -> None:
    _fixture, _spec, _store, _active, bound = _advance_test_store_to_bound(tmp_path)
    after = _after_snapshot_for_bound(bound)

    if attack == "top-missing":
        del after["repository"]
    elif attack == "top-unknown":
        after["attacker"] = "field"
    elif attack == "host-missing":
        del after["host"]["platform"]
    elif attack == "host-unknown":
        after["host"]["attacker"] = "field"
    elif attack == "host-custom-mapping":

        class _CustomMapping(dict[str, Any]):
            pass

        after["host"] = _CustomMapping(after["host"])
    elif attack == "path-required-tuple":
        after["path_registry"]["required_lookup_keys"] = tuple(
            after["path_registry"]["required_lookup_keys"]
        )
    elif attack == "repository-allowed-set":
        after["repository"]["allowed_untracked_roots"] = {"tmp"}
    elif attack == "host-nonfinite":
        after["host"]["available_memory_bytes"] = float("nan")
    elif attack == "component-count-bool":
        after["malom"]["component_count"] = True
    elif attack == "component-count-float":
        after["malom"]["component_count"] = float(after["malom"]["component_count"])
    elif attack == "checkout-clean-int":
        after["sanmill"]["checkout_clean"] = 1
    elif attack == "cuda-initialized-int":
        after["host"]["cuda_initialized"] = 0
    elif attack == "wrong-fixed-lookup":
        after["malom"]["lookup_key"] = "attacker_malom_path"
    elif attack == "wrong-model-config":
        after["generator"]["model_config"]["dropout"] = 0
    elif attack == "wrong-sampler-config":
        after["generator"]["sampler_config"]["temperature"] = 1
    else:  # pragma: no cover - the parameter list is exhaustive
        raise AssertionError(attack)
    if attack not in {"repository-allowed-set", "host-nonfinite"}:
        _resign_runtime_snapshot(after)
    with pytest.raises(GovernanceStoreContractError):
        store_module._issue_test_state_generation_runtime_completion(
            bound,
            resource_snapshot_after=after,
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("host", "machine_identity"), _sha("different-machine")),
        (("repository", "head_commit"), "f" * 40),
        (("output_root", "file_identity"), _sha("different-output-file")),
        (("malom", "content_sha256"), _sha("different-malom-content")),
        (("sanmill", "binary_sha256"), _sha("different-sanmill-binary")),
        (("generator", "initial_rng_state_identity"), _sha("different-initial-rng")),
    ],
)
def test_c5a_type_sensitive_stability_drift_cannot_issue_runtime_completion(
    tmp_path: Path,
    path: tuple[str, str],
    replacement: Any,
) -> None:
    _fixture, _spec, _store, _active, bound = _advance_test_store_to_bound(tmp_path)
    after = _after_snapshot_for_bound(bound)
    after[path[0]][path[1]] = replacement
    _resign_runtime_snapshot(after)
    with pytest.raises(GovernanceStoreContractError):
        store_module._issue_test_state_generation_runtime_completion(
            bound,
            resource_snapshot_after=after,
        )


def _rewrite_runtime_record(
    database: Path,
    *,
    record_type: str,
    payload: bytes,
    row_stream_identity: str | None = None,
    row_stream_kind: str | None = None,
    row_sequence: int | None = None,
    row_record_type: str | None = None,
    row_previous_identity: str | None | object = Ellipsis,
) -> None:
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        row = connection.execute(
            "SELECT stream_identity, stream_kind, sequence, record_type, "
            "previous_record_identity FROM domain_records WHERE record_type=?",
            (record_type,),
        ).fetchone()
        assert row is not None
        identity = hashlib.sha256(payload).hexdigest()
        previous = (
            row[4] if row_previous_identity is Ellipsis else row_previous_identity
        )
        connection.execute("DROP TRIGGER domain_records_reject_update")
        connection.execute(
            "UPDATE domain_records SET stream_identity=?, stream_kind=?, sequence=?, "
            "record_type=?, previous_record_identity=?, record_identity=?, "
            "record_bytes=?, size_bytes=?, record_bytes_sha256=? "
            "WHERE record_type=?",
            (
                row[0] if row_stream_identity is None else row_stream_identity,
                row[1] if row_stream_kind is None else row_stream_kind,
                row[2] if row_sequence is None else row_sequence,
                row[3] if row_record_type is None else row_record_type,
                previous,
                identity,
                payload,
                len(payload),
                identity,
                record_type,
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


@pytest.mark.parametrize(
    "attack",
    [
        "bom",
        "duplicate-key",
        "nan",
        "noncanonical",
        "resigned-unstable",
        "old-v1",
        "unknown-type",
        "wrong-stream-kind",
        "gap",
        "wrong-prev",
        "cross-stream",
    ],
)
def test_c5a_runtime_stream_corruption_and_resigning_fail_closed(
    tmp_path: Path,
    attack: str,
) -> None:
    prepared = _prepare_valid_state_commit(tmp_path)
    fixture, spec, store, pending = prepared[0], prepared[1], prepared[2], prepared[5]
    store_module._test_commit_state_generation(store, pending)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database)
    try:
        original = bytes(
            connection.execute(
                "SELECT record_bytes FROM domain_records "
                "WHERE record_type='state-generation-resource-stability'"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    stability = json.loads(original)
    rewrite: dict[str, Any] = {}
    if attack == "bom":
        payload = b"\xef\xbb\xbf" + original
    elif attack == "duplicate-key":
        payload = original[:-1] + b',"stable":true}'
    elif attack == "nan":
        payload = original[:-1] + b',"attacker":NaN}'
    elif attack == "noncanonical":
        payload = json.dumps(stability, indent=2, sort_keys=True).encode("utf-8")
    elif attack == "resigned-unstable":
        stability["stable"] = False
        stability["differences"] = ["repository.head_commit"]
        payload = canonical_json_bytes(stability)
    elif attack == "old-v1":
        stability["schema_version"] = (
            "nmm.classical-a-pos-state-generation-resource-stability.v0"
        )
        payload = canonical_json_bytes(stability)
    elif attack == "unknown-type":
        stability["record_type"] = "teacher-label-reserved"
        stability["schema_version"] = "nmm.classical-a-pos-teacher-label.v1"
        payload = canonical_json_bytes(stability)
        rewrite["row_record_type"] = "teacher-label-reserved"
    elif attack == "wrong-stream-kind":
        stability["stream_kind"] = "attacker-runtime"
        payload = canonical_json_bytes(stability)
        rewrite["row_stream_kind"] = "attacker-runtime"
    elif attack == "gap":
        stability["sequence"] = 4
        payload = canonical_json_bytes(stability)
        rewrite["row_sequence"] = 4
    elif attack == "wrong-prev":
        stability["previous_record_identity"] = _sha("wrong-runtime-prev")
        payload = canonical_json_bytes(stability)
        rewrite["row_previous_identity"] = stability["previous_record_identity"]
    elif attack == "cross-stream":
        stability["stream_identity"] = _sha("cross-runtime-stream")
        payload = canonical_json_bytes(stability)
        rewrite["row_stream_identity"] = stability["stream_identity"]
    else:  # pragma: no cover - the parameter list is exhaustive
        raise AssertionError(attack)
    _rewrite_runtime_record(
        database,
        record_type="state-generation-resource-stability",
        payload=payload,
        **rewrite,
    )
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
        )


def test_c5a_runtime_stream_duplicate_identity_is_rejected_by_sqlite(
    tmp_path: Path,
) -> None:
    prepared = _prepare_valid_state_commit(tmp_path)
    store = prepared[2]
    pending = prepared[5]
    store_module._test_commit_state_generation(store, pending)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        row = connection.execute(
            "SELECT stream_kind, record_type, record_identity, record_bytes, "
            "size_bytes, record_bytes_sha256 FROM domain_records "
            "WHERE record_type='state-generation-resource-before'"
        ).fetchone()
        assert row is not None
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO domain_records(stream_identity, stream_kind, sequence, "
                "record_type, previous_record_identity, record_identity, "
                "record_bytes, size_bytes, record_bytes_sha256) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _sha("duplicate-runtime-stream"),
                    row[0],
                    0,
                    row[1],
                    None,
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                ),
            )
    finally:
        connection.close()


@pytest.mark.parametrize("artifact", ["spec", "meta", "completion", "freeze"])
def test_c5a_resigned_v1_store_artifacts_fail_closed(
    tmp_path: Path,
    artifact: str,
) -> None:
    if artifact in {"completion", "freeze"}:
        prepared = _prepare_valid_state_commit(tmp_path)
        fixture, spec, store, pending = (
            prepared[0],
            prepared[1],
            prepared[2],
            prepared[5],
        )
        completion = store_module._test_commit_state_generation(store, pending)
        if artifact == "freeze":
            store_module._test_commit_state_freeze(
                store,
                completion,
                timestamp_utc="2026-09-01T00:00:06Z",
            )
    else:
        fixture, spec, _store = _initialize_test_store(tmp_path)
    if artifact == "spec":
        raw_spec = _mutable(spec)
        raw_spec["schema_version"] = "nmm.classical-a-pos-governance-store-spec.v1"
        body = {key: value for key, value in raw_spec.items() if key != "spec_identity"}
        raw_spec["spec_identity"] = canonical_sha256(body)
        old_spec = GovernanceStoreSpec(raw_spec)
        with pytest.raises(GovernanceStoreContractError):
            store_module._test_open_governance_store(
                tmp_path,
                old_spec,
                plan_permit=fixture.plan_permit,
                authorization_permit=fixture.authorization_permit,
            )
        return

    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    if artifact == "meta":
        connection = sqlite3.connect(database, isolation_level=None)
        try:
            payload = bytes(
                connection.execute(
                    "SELECT value FROM meta WHERE key='store_meta'"
                ).fetchone()[0]
            )
            meta = json.loads(payload)
            meta["schema_version"] = "nmm.classical-a-pos-governance-store-meta.v1"
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
    else:
        record_type = (
            "state-generation-completion"
            if artifact == "completion"
            else "state-freeze"
        )
        connection = sqlite3.connect(database)
        try:
            payload = bytes(
                connection.execute(
                    "SELECT record_bytes FROM domain_records WHERE record_type=?",
                    (record_type,),
                ).fetchone()[0]
            )
        finally:
            connection.close()
        record = json.loads(payload)
        record["schema_version"] = (
            "nmm.classical-a-pos-state-generation-completion.v1"
            if artifact == "completion"
            else "nmm.classical-a-pos-durable-state-freeze.v1"
        )
        resigned_v1 = canonical_json_bytes(record)
        _rewrite_runtime_record(
            database,
            record_type=record_type,
            payload=resigned_v1,
        )
        connection = store_module._open_read_connection(database)
        try:
            with pytest.raises(GovernanceStoreContractError):
                store_module._verify_blob_tables(connection)
        finally:
            connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
        )


def test_c5a_running_store_reopen_fails_closed(tmp_path: Path) -> None:
    fixture, spec, _store, _active = _advance_test_store_to_active(tmp_path)
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
        )


def test_c5a_runtime_bind_failure_spends_preflight_and_active_without_rows(
    tmp_path: Path,
) -> None:
    _fixture, _spec, store, active = _advance_test_store_to_active(tmp_path)
    preflight = store_module._issue_test_state_generation_runtime_preflight(
        store,
        active,
    )
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    lock = sqlite3.connect(database, timeout=0, isolation_level=None)
    try:
        lock.execute("BEGIN IMMEDIATE")
        with pytest.raises(GovernanceStoreContractError):
            store_module._test_bind_state_generation_attempt(
                preflight,
                store,
                active,
            )
    finally:
        lock.execute("ROLLBACK")
        lock.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_bind_state_generation_attempt(
            preflight,
            store,
            active,
        )
    with pytest.raises(GovernanceStoreContractError):
        store_module._issue_test_state_generation_runtime_preflight(
            store,
            active,
        )
    connection = sqlite3.connect(database)
    try:
        count = connection.execute(
            "SELECT COUNT(*) FROM domain_records "
            "WHERE stream_kind='state-generation-runtime'"
        ).fetchone()[0]
    finally:
        connection.close()
    assert count == 0


def test_c5a_runtime_before_rejects_reused_and_cross_store_operation_permits(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    _first_fixture, _first_spec, first_store, first_permit = (
        _advance_test_store_to_reserved(first_root)
    )
    _second_fixture, _second_spec, second_store, _second_permit = (
        _advance_test_store_to_reserved(second_root)
    )

    with pytest.raises(GovernanceStoreContractError):
        store_module._test_begin_state_generation_attempt(
            second_store,
            first_permit,
        )
    store_module._test_begin_state_generation_attempt(first_store, first_permit)
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_begin_state_generation_attempt(first_store, first_permit)


def test_c5a_runtime_bind_rejects_cross_store_wrong_active_reuse_and_forgery(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first-bind"
    second_root = tmp_path / "second-bind"
    first_root.mkdir()
    second_root.mkdir()
    _first_fixture, _first_spec, first_store, first_active = (
        _advance_test_store_to_active(first_root)
    )
    _second_fixture, _second_spec, second_store, second_active = (
        _advance_test_store_to_active(second_root)
    )
    first_preflight = store_module._issue_test_state_generation_runtime_preflight(
        first_store,
        first_active,
    )
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_bind_state_generation_attempt(
            first_preflight,
            second_store,
            second_active,
        )
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_bind_state_generation_attempt({}, first_store, first_active)
    rogue = object.__new__(store_module._TestStateGenerationRuntimePreflight)
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_bind_state_generation_attempt(
            rogue, first_store, first_active
        )

    second_preflight = store_module._issue_test_state_generation_runtime_preflight(
        second_store,
        second_active,
    )
    bound = store_module._test_bind_state_generation_attempt(
        second_preflight,
        second_store,
        second_active,
    )
    assert type(bound) is store_module._TestRuntimeBoundStateGenerationAttempt
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_bind_state_generation_attempt(
            second_preflight,
            second_store,
            second_active,
        )
    with pytest.raises(GovernanceStoreContractError):
        store_module._issue_test_state_generation_runtime_preflight(
            second_store,
            second_active,
        )


def test_c5a_unbound_active_cannot_issue_completion_or_prepare_artifacts(
    tmp_path: Path,
) -> None:
    _fixture, _spec, _store, active = _advance_test_store_to_active(tmp_path)
    with pytest.raises(GovernanceStoreContractError):
        store_module._issue_test_state_generation_runtime_completion(active)

    other_root = tmp_path / "bound-runtime"
    other_root.mkdir()
    _other_fixture, _other_spec, _other_store, _other_active, other_bound = (
        _advance_test_store_to_bound(other_root)
    )
    runtime_completion = store_module._issue_test_state_generation_runtime_completion(
        other_bound
    )
    verifier_identity = _sha("unbound-active-verifier")
    inventory = _test_inventory_binding(verifier_identity)
    attempt_identity = store_module._TEST_ACTIVE_CONTEXTS[active].attempt_identity
    strict_referee = _test_strict_referee_binding(attempt_identity)
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_prepare_state_generation_commit(
            active,
            inventory,
            strict_referee,
            runtime_completion,
            source_games_path=tmp_path / "must-not-read-source.jsonl",
            state_split_path=tmp_path / "must-not-read-state.jsonl",
            singleton_ledger_path=tmp_path / "must-not-read-singletons.json",
            timestamp_utc="2026-09-01T00:00:05Z",
            active_seconds=0,
        )


def test_c5a_runtime_bind_rejects_preexisting_runtime_row_and_is_one_shot(
    tmp_path: Path,
) -> None:
    _fixture, _spec, store, active = _advance_test_store_to_active(tmp_path)
    preflight = store_module._issue_test_state_generation_runtime_preflight(
        store,
        active,
    )
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    context = store_module._TEST_ACTIVE_CONTEXTS[active]
    payload = canonical_json_bytes(
        {
            "schema_version": (
                "nmm.classical-a-pos-state-generation-resource-before.v1"
            ),
            "stream_identity": _sha("preexisting-runtime-stream"),
            "stream_kind": "state-generation-runtime",
            "sequence": 0,
            "record_type": "state-generation-resource-before",
            "previous_record_identity": None,
            "attempt_identity": context.attempt_identity,
        }
    )
    identity = hashlib.sha256(payload).hexdigest()
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        connection.execute(
            "INSERT INTO domain_records(stream_identity, stream_kind, sequence, "
            "record_type, previous_record_identity, record_identity, record_bytes, "
            "size_bytes, record_bytes_sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _sha("preexisting-runtime-stream"),
                "state-generation-runtime",
                0,
                "state-generation-resource-before",
                None,
                identity,
                payload,
                len(payload),
                identity,
            ),
        )
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_bind_state_generation_attempt(preflight, store, active)
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_bind_state_generation_attempt(preflight, store, active)


def test_c5a_completion_lock_failure_has_no_partial_after_or_stability_and_no_retry(
    tmp_path: Path,
) -> None:
    prepared = _prepare_valid_state_commit(tmp_path)
    store = prepared[2]
    pending = prepared[5]
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    lock = sqlite3.connect(database, timeout=0, isolation_level=None)
    try:
        lock.execute("BEGIN IMMEDIATE")
        with pytest.raises(GovernanceStoreContractError):
            store_module._test_commit_state_generation(store, pending)
    finally:
        lock.execute("ROLLBACK")
        lock.close()
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_commit_state_generation(store, pending)
    connection = sqlite3.connect(database)
    try:
        runtime_rows = connection.execute(
            "SELECT sequence, record_type FROM domain_records "
            "WHERE stream_kind='state-generation-runtime' ORDER BY sequence"
        ).fetchall()
        completion_count = connection.execute(
            "SELECT COUNT(*) FROM domain_records "
            "WHERE record_type='state-generation-completion'"
        ).fetchone()[0]
        controller_count = connection.execute(
            "SELECT COUNT(*) FROM domain_records WHERE stream_kind='controller'"
        ).fetchone()[0]
        artifact_count = connection.execute(
            "SELECT COUNT(*) FROM artifacts"
        ).fetchone()[0]
    finally:
        connection.close()
    assert runtime_rows == [
        (0, "state-generation-resource-before"),
    ]
    assert completion_count == 0
    assert controller_count == 0
    assert artifact_count == 0


def test_c5a_restore_durable_state_freeze_reissues_only_from_frozen_store(
    tmp_path: Path,
) -> None:
    prepared = _prepare_valid_state_commit(tmp_path)
    fixture, spec, store, pending = prepared[0], prepared[1], prepared[2], prepared[5]
    completion = store_module._test_commit_state_generation(store, pending)
    original = store_module._test_commit_state_freeze(
        store,
        completion,
        timestamp_utc="2026-09-01T00:00:06Z",
    )
    expected_state_identity = store_module._test_verify_durable_state_freeze(
        store,
        original,
    )
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    connection = sqlite3.connect(database)
    try:
        completion_record = json.loads(
            bytes(
                connection.execute(
                    "SELECT record_bytes FROM domain_records "
                    "WHERE record_type='state-generation-completion'"
                ).fetchone()[0]
            )
        )
        freeze_record = json.loads(
            bytes(
                connection.execute(
                    "SELECT record_bytes FROM domain_records "
                    "WHERE record_type='state-freeze'"
                ).fetchone()[0]
            )
        )
    finally:
        connection.close()
    assert (
        freeze_record["runtime_evidence_stream_identity"]
        == completion_record["runtime_evidence_stream_identity"]
    )
    assert (
        freeze_record["resource_stability_identity"]
        == completion_record["resource_stability_identity"]
    )
    reopened = store_module._test_open_governance_store(
        tmp_path,
        spec,
        plan_permit=fixture.plan_permit,
        authorization_permit=fixture.authorization_permit,
    )
    restored = store_module._test_restore_durable_state_freeze(reopened)
    with pytest.raises(GovernanceStoreContractError):
        store_module.restore_durable_state_freeze(reopened)
    assert (
        store_module._test_verify_durable_state_freeze(reopened, restored)
        == expected_state_identity
    )


def test_c5a_restore_rejects_generated_but_not_frozen_store(tmp_path: Path) -> None:
    prepared = _prepare_valid_state_commit(tmp_path)
    store = prepared[2]
    pending = prepared[5]
    store_module._test_commit_state_generation(store, pending)
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_restore_durable_state_freeze(store)


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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
        )


def test_corrupt_database_header_fails_closed(tmp_path: Path) -> None:
    fixture, spec, _store = _initialize_test_store(tmp_path)
    database = tmp_path / "governance" / "classical-a-pos-controller.sqlite3"
    database.write_bytes(b"not-a-sqlite-database")
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_open_governance_store(
            tmp_path,
            spec,
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
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
            plan_permit=fixture.plan_permit,
            authorization_permit=fixture.authorization_permit,
        )


@pytest.mark.parametrize(
    "path_attack",
    ["governance", "outside", "hardlink", "lexical-alias"],
)
def test_state_artifact_paths_must_be_unaliased_plan_owned_files(
    tmp_path: Path,
    path_attack: str,
) -> None:
    _fixture, _spec, _store, active, bound = _advance_test_store_to_bound(tmp_path)
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
    runtime_completion = store_module._issue_test_state_generation_runtime_completion(
        bound
    )
    with pytest.raises(GovernanceStoreContractError):
        store_module._test_prepare_state_generation_commit(
            bound,
            inventory,
            strict_referee,
            runtime_completion,
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
            object(),
            source_games_path=source_path,
            state_split_path=state_path,
            singleton_ledger_path=singleton_path,
            timestamp_utc="2026-09-01T00:00:05Z",
            active_seconds=11,
        )
