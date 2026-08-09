from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import learned_ai.evaluation.heldout_evaluation as heldout
from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.training.sanmill_referee import (
    SanmillAppliedTurn,
    SanmillTrainingGame,
    inspect_sanmill_training_installation,
)


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATHS = ROOT / "data" / "training_paths.local.json"


def _training_installation():
    if not LOCAL_PATHS.is_file():
        pytest.skip("requires the ignored local training path registry")
    config = json.loads(LOCAL_PATHS.read_text(encoding="utf-8"))
    checkout_value = config.get("sanmill_training_checkout")
    if not isinstance(checkout_value, str) or not checkout_value:
        pytest.skip("requires sanmill_training_checkout")
    checkout = Path(checkout_value)
    if not checkout.is_absolute():
        checkout = ROOT / checkout
    return inspect_sanmill_training_installation(checkout)


@pytest.fixture(scope="module")
def contract() -> heldout.FrozenHeldoutContract:
    return heldout.load_frozen_heldout_contract()


@pytest.fixture(scope="module")
def runtime_spec(contract) -> dict:
    readiness = {
        "gates": [
            {
                "gate": "repository",
                "result": "pass",
                "observed": {
                    "branch": "dev",
                    "head": "a" * 40,
                    "tree": "b" * 40,
                    "upstream_commit": "a" * 40,
                },
            }
        ],
        "readiness_identity": "c" * 64,
    }
    return heldout.build_runtime_spec(contract, readiness)


def _history(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _search(turn: dict) -> dict:
    return {
        "status": "ok",
        "full_turn_actions": list(turn["actions"]),
        "logical_move_id": "logical:test",
        "model_action": dict(turn["move"]),
        "logical_ply_delta": 1,
        "resulting_fen": turn["sanmill_fen_after"],
        "resulting_side_to_move": "B",
        "terminal": turn["terminal"],
        "winner": None,
        "winner_code": None,
        "outcome_reason": turn["outcome_reason"],
        "effective_depth": 12,
        "completed_depth": 12,
        "score_kind": "cp",
        "score": 0,
        "score_perspective": "side-to-move",
        "node_budget": 500_000,
        "primary_nodes": 100,
        "removal_nodes": 0,
        "total_nodes": 100,
        "search_calls": 1,
    }


def _turn(
    *,
    ordinal: int,
    index: int,
    actor: str,
    before: str,
    terminal: bool,
    outcome_reason: str,
) -> dict:
    mover = "W" if index % 2 == 1 else "B"
    move = {"from": None, "to": "a7" if mover == "W" else "d7", "capture": None}
    after = _history(f"game-{ordinal}-turn-{index}")
    turn = {
        "post_prefix_logical_ply": index,
        "mover_color": mover,
        "actor": actor,
        "move": move,
        "actions": [move["to"]],
        "before_history_sha256": before,
        "after_history_sha256": after,
        "logical_ply_count": 12 + index,
        "local_fen_after": f"local-fen-{ordinal}-{index}",
        "sanmill_fen_after": f"sanmill-fen-{ordinal}-{index}",
        "terminal": terminal,
        "outcome_reason": outcome_reason,
        "search": None,
    }
    if actor == "sanmill":
        turn["search"] = _search(turn)
    return turn


def _game_record(
    spec: dict,
    ordinal: int,
    previous_hash: str | None,
) -> dict:
    scheduled = spec["schedule"][ordinal]
    candidate = scheduled["candidate_color"]
    reason = "loseFewerThanThree"
    turn_count = 1 if candidate == "W" else 2
    prefix_history = scheduled["expected_prefix_history_sha256"]
    turns = []
    before = prefix_history
    for index in range(1, turn_count + 1):
        mover = "W" if index % 2 == 1 else "B"
        actor = "candidate" if mover == candidate else "sanmill"
        turn = _turn(
            ordinal=ordinal,
            index=index,
            actor=actor,
            before=before,
            terminal=index == turn_count,
            outcome_reason=reason if index == turn_count else "ongoing",
        )
        turns.append(turn)
        before = turn["after_history_sha256"]
    prefix_action_tokens = 12
    final_action_tokens = prefix_action_tokens + sum(
        len(turn["actions"]) for turn in turns
    )
    winner_name = "white" if candidate == "W" else "black"
    winner_code = 0 if candidate == "W" else 1
    final = {
        "status": "terminal",
        "ruleset_id": "nmm-test",
        "rules_identity_sha256": heldout.EXPECTED_RULES_IDENTITY_SHA256,
        "history_origin": "startpos",
        "fen": turns[-1]["sanmill_fen_after"],
        "side_to_move": None,
        "phase": "moving",
        "action": "select",
        "terminal": True,
        "removal_pending": False,
        "pending_removal_count": 0,
        "pending_removals": [0, 0],
        "legal_actions": [],
        "action_token_count": final_action_tokens,
        "logical_ply_count": 12 + turn_count,
        "logical_plies_by_side": [
            6 + (turn_count + 1) // 2,
            6 + turn_count // 2,
        ],
        "no_capture_count": 0,
        "repetition_current_count": 1,
        "repetition_history_length": 1,
        "snapshot_history_length": 13 + turn_count,
        "history_sha256": turns[-1]["after_history_sha256"],
        "outcome": {
            "terminal": True,
            "winner": winner_name,
            "winner_code": winner_code,
            "reason": reason,
            "reason_code": "lose_fewer_than_three",
        },
        "strict_referee_identity": {
            "format": "SANMILL-STRICT-REFEREE-RULES/1",
            "profile": "mif-stable-moving-v1",
            "repetitionObservation": "stable-moving-v1",
            "originCounted": True,
            "semanticDigest": heldout.TRAINING_REFEREE_SEMANTIC_DIGEST,
        },
    }
    return {
        "schema_version": heldout.HELDOUT_GAME_SCHEMA,
        "spec_identity": spec["spec_identity"],
        "ordinal": ordinal,
        "pair_index": scheduled["pair_index"],
        "game_in_pair": scheduled["game_in_pair"],
        "game_id": scheduled["game_id"],
        "source_core_id": scheduled["source_core_id"],
        "stratum": scheduled["stratum"],
        "strict_independence_sensitivity": scheduled["strict_independence_sensitivity"],
        "candidate_color": candidate,
        "candidate_score": 1.0,
        "winner": candidate,
        "outcome_reason": reason,
        "prefix": {
            "prefix_identity": scheduled["prefix_identity"],
            "expected_history_sha256": prefix_history,
            "observed_history_sha256": prefix_history,
            "action_token_count": prefix_action_tokens,
            "logical_ply_count": 12,
            "logical_plies_by_side": [6, 6],
            "final_nmm_fen": "prefix-local-fen",
            "final_sanmill_fen": "prefix-sanmill-fen",
        },
        "post_prefix_logical_plies": turn_count,
        "final_state": final,
        "turns": turns,
        "search_summary": heldout._search_summary(turns),
        "game_elapsed_seconds": 0.5,
        "cumulative_active_seconds": float(ordinal + 1),
        "complete": True,
        "previous_record_sha256": previous_hash,
    }


def test_frozen_contract_builds_one_ordered_color_swapped_schedule(
    contract,
    runtime_spec,
) -> None:
    assert len(contract.records) == 64
    assert len(runtime_spec["schedule"]) == 128
    for pair_index in range(64):
        white, black = runtime_spec["schedule"][pair_index * 2 : pair_index * 2 + 2]
        assert white["pair_index"] == black["pair_index"] == pair_index
        assert white["source_core_id"] == black["source_core_id"]
        assert white["candidate_color"] == "W"
        assert black["candidate_color"] == "B"
        assert white["prefix_identity"] == black["prefix_identity"]


def test_game_ledger_rejects_noncanonical_or_crlf_evidence(
    tmp_path,
    runtime_spec,
) -> None:
    record = _game_record(runtime_spec, 0, None)
    canonical = tmp_path / "canonical.jsonl"
    heldout.append_game_record(canonical, record, must_create=True)
    loaded, tail = heldout.load_game_ledger(runtime_spec, canonical)
    assert loaded == [record]
    assert tail == canonical_sha256(record)

    wrapper = {"record": record, "record_sha256": canonical_sha256(record)}
    noncanonical = tmp_path / "noncanonical.jsonl"
    noncanonical.write_bytes(json.dumps(wrapper).encode("utf-8") + b"\n")
    with pytest.raises(heldout.HeldoutEvaluationError, match="canonical JSON"):
        heldout.load_game_ledger(runtime_spec, noncanonical)

    crlf = tmp_path / "crlf.jsonl"
    crlf.write_bytes(canonical_json_bytes(wrapper) + b"\r\n")
    with pytest.raises(heldout.HeldoutEvaluationError, match="LF-framed"):
        heldout.load_game_ledger(runtime_spec, crlf)


def test_game_ledger_rejects_role_and_search_evidence_drift(
    tmp_path,
    runtime_spec,
) -> None:
    record = _game_record(runtime_spec, 0, None)
    record["turns"][0]["actor"] = "sanmill"
    record["turns"][0]["search"] = _search(record["turns"][0])
    record["search_summary"] = heldout._search_summary(record["turns"])
    path = tmp_path / "wrong-role.jsonl"
    heldout.append_game_record(path, record, must_create=True)
    with pytest.raises(heldout.HeldoutEvaluationError, match="actor and color"):
        heldout.load_game_ledger(runtime_spec, path)


def test_complete_ledger_recomputes_frozen_primary_and_subgroup_results(
    tmp_path,
    runtime_spec,
) -> None:
    spec_path = tmp_path / "spec.json"
    ledger_path = tmp_path / "games.jsonl"
    heldout.write_new_canonical(spec_path, runtime_spec)
    previous = None
    for ordinal in range(128):
        record = _game_record(runtime_spec, ordinal, previous)
        previous = heldout.append_game_record(
            ledger_path,
            record,
            must_create=ordinal == 0,
        )

    result = heldout.recompute_heldout_evaluation(spec_path, ledger_path)

    assert result["primary"]["decision"] == "candidate_ahead"
    assert result["primary"]["support_pairs"] == 64
    assert result["primary"]["interval"] == [1.0, 1.0]
    assert result["by_candidate_color"]["W"]["games"] == 64
    assert result["by_candidate_color"]["B"]["games"] == 64
    assert result["strict_independence_sensitivity"]["support_pairs"] == 34
    assert result["by_source_stratum"]["book"]["support_pairs"] == 22
    assert result["by_source_stratum"]["human_db"]["support_pairs"] == 21
    assert result["by_source_stratum"]["perfect_db"]["support_pairs"] == 21
    assert result["termination_reasons"] == {"loseFewerThanThree": 128}


def test_replace_canonical_retries_a_transient_windows_reader(
    tmp_path,
    monkeypatch,
) -> None:
    target = tmp_path / "progress.json"
    heldout.write_new_canonical(target, {"generation": 1})
    real_replace = heldout.os.replace
    calls = 0
    sleeps: list[float] = []

    def transient_replace(source, destination) -> None:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise PermissionError(5, "target is temporarily open by a reader")
        real_replace(source, destination)

    monkeypatch.setattr(heldout.os, "replace", transient_replace)
    monkeypatch.setattr(heldout.time, "sleep", sleeps.append)

    heldout.replace_canonical(target, {"generation": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"generation": 2}
    assert calls == 3
    assert sleeps == [
        heldout.ATOMIC_REPLACE_RETRY_SECONDS,
        heldout.ATOMIC_REPLACE_RETRY_SECONDS,
    ]
    assert list(tmp_path.glob(".progress.json.*.tmp")) == []


def test_replace_canonical_fails_closed_after_bounded_permission_retries(
    tmp_path,
    monkeypatch,
) -> None:
    target = tmp_path / "progress.json"
    heldout.write_new_canonical(target, {"generation": 1})
    calls = 0

    def blocked_replace(_source, _destination) -> None:
        nonlocal calls
        calls += 1
        raise PermissionError(5, "target remains inaccessible")

    monkeypatch.setattr(heldout.os, "replace", blocked_replace)
    monkeypatch.setattr(heldout.time, "sleep", lambda _seconds: None)

    with pytest.raises(PermissionError, match="remains inaccessible"):
        heldout.replace_canonical(target, {"generation": 2})

    assert calls == heldout.ATOMIC_REPLACE_PERMISSION_RETRIES
    assert json.loads(target.read_text(encoding="utf-8")) == {"generation": 1}
    assert list(tmp_path.glob(".progress.json.*.tmp")) == []


def test_safety_ceiling_invalidates_instead_of_manufacturing_a_draw(
    monkeypatch,
    runtime_spec,
) -> None:
    spec = copy.deepcopy(runtime_spec)
    spec["protocol"]["max_post_prefix_logical_plies"] = 1
    scheduled = spec["schedule"][0]

    class FakePolicy:
        @staticmethod
        def choose_move(board):
            return get_all_legal_moves(board)[0]

    class FakeGame:
        def __init__(self, *_args, **_kwargs):
            self.state = SimpleNamespace(
                history_sha256=scheduled["expected_prefix_history_sha256"],
                logical_ply_count=12,
                fen="prefix-sanmill-fen",
                terminal=False,
                outcome_reason="ongoing",
            )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def apply_nmm_move(self, board, move):
            self.state = SimpleNamespace(
                history_sha256=_history("safety-cap"),
                logical_ply_count=13,
                fen="ongoing-sanmill-fen",
                terminal=False,
                outcome_reason="ongoing",
            )
            return SanmillAppliedTurn(
                move=move,
                actions=tuple(heldout.nmm_move_actions(move)),
                state=self.state,
                search=None,
            )

    monkeypatch.setattr(
        heldout,
        "replay_frozen_prefix",
        lambda _game, _record, progress: (
            BoardState.new_game(),
            {
                "prefix_identity": scheduled["prefix_identity"],
                "observed_history_sha256": scheduled["expected_prefix_history_sha256"],
            },
        ),
    )
    clock = heldout.ActiveClock(base_seconds=0.0, max_seconds=60.0)

    with pytest.raises(heldout.HeldoutEvaluationInvalid, match="safety cap"):
        heldout.play_heldout_game(
            spec=spec,
            schedule_item=scheduled,
            corpus_record={},
            policy=FakePolicy(),
            installation=None,
            previous_record_sha256=None,
            clock=clock,
            progress_callback=lambda _stage, _ply: None,
            game_factory=FakeGame,
        )


def test_resume_preflight_binds_same_spec_progress_and_host(
    tmp_path,
    contract,
) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    readiness_body = {
        "schema_version": heldout.HELDOUT_READINESS_SCHEMA,
        "gates": [
            {
                "gate": "repository",
                "result": "pass",
                "observed": {
                    "branch": "dev",
                    "head": head,
                    "tree": "e" * 40,
                    "upstream_commit": head,
                },
            }
        ],
    }
    readiness = {
        **readiness_body,
        "readiness_identity": canonical_sha256(readiness_body),
    }
    spec = heldout.build_runtime_spec(contract, readiness)
    paths = heldout.HeldoutPaths(
        paths_config=tmp_path / "paths.json",
        candidate_bundle=tmp_path / "bundle",
        checkpoint=tmp_path / "checkpoint.pt",
        corpus=tmp_path / "corpus.json",
        exposure_audit=tmp_path / "audit.json",
        human_db=tmp_path / "human.sqlite",
        specialist_db=tmp_path / "specialist.sqlite",
        malom_db=tmp_path / "malom",
        malom_manifest=tmp_path / "malom.json",
        ruleset_manifest=tmp_path / "ruleset.json",
        sanmill_checkout=tmp_path / "sanmill",
        output_root=tmp_path,
        output_plan=tmp_path / "plan.json",
        output_authorization=tmp_path / "authorization.json",
        output_spec=tmp_path / "spec.json",
        output_ledger=tmp_path / "games.jsonl",
        output_report=tmp_path / "report.json",
    )
    paths.output_plan.write_bytes(contract.plan_path.read_bytes())
    paths.output_authorization.write_bytes(contract.authorization_path.read_bytes())
    heldout.write_new_canonical(tmp_path / "readiness.json", readiness)
    heldout.write_new_canonical(paths.output_spec, spec)
    heldout._write_progress(
        tmp_path / "progress.json",
        heldout._progress_body(
            spec["spec_identity"],
            completed_games=0,
            current_game_ordinal=None,
            current_stage=None,
            current_stage_ply=0,
            active_seconds=0.0,
            ledger_tail_sha256=None,
        ),
    )

    observed = heldout._resume_continuity_record(contract, paths)

    assert observed["completed_games"] == 0
    assert observed["missing_suffix_games"] == 128
    assert observed["authorization_consumed"] is False

    heldout._write_progress(
        tmp_path / "progress.json",
        heldout._progress_body(
            spec["spec_identity"],
            completed_games=0,
            current_game_ordinal=0,
            current_stage="game",
            current_stage_ply=1,
            active_seconds=1.0,
            ledger_tail_sha256="f" * 64,
        ),
    )
    with pytest.raises(heldout.HeldoutEvaluationError, match="ledger tail"):
        heldout._resume_continuity_record(contract, paths)


def test_cli_never_launches_without_explicit_flag() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_heldout_evaluation.py", "run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 2
    assert "explicit --launch flag" in result.stderr


def test_pinned_sanmill_runtime_matches_closed_evidence_shapes() -> None:
    installation = _training_installation()
    board = BoardState.new_game()
    with SanmillTrainingGame(installation, seed=42) as game:
        before_history = game.state.history_sha256
        applied = game.search_and_apply(board, node_budget=500_000, depth=None)
        board_after = board.apply_move(applied.move)
        turn = heldout._turn_record(
            post_prefix_ply=1,
            mover_color="W",
            actor="sanmill",
            board_after=board_after,
            before_history=before_history,
            applied=applied,
        )
        assert set(applied.search.semantic_record()) == heldout._SEARCH_FIELDS
        heldout._validate_search(applied.search.semantic_record(), turn=turn)

    action_history = (
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
    board = BoardState.new_game()
    action_index = 0
    with SanmillTrainingGame(installation, seed=42) as game:
        while action_index < len(action_history):
            actions = [action_history[action_index]]
            action_index += 1
            if action_index < len(action_history) and action_history[
                action_index
            ].startswith("x"):
                actions.append(action_history[action_index])
                action_index += 1
            move = heldout._matching_move(board, actions)
            game.apply_nmm_move(board, move)
            board = board.apply_move(move)
        final_state = game.state.portable_record()

    assert final_state["terminal"] is True
    assert set(final_state) == heldout._FINAL_STATE_FIELDS
    assert set(final_state["outcome"]) == heldout._OUTCOME_FIELDS
    assert set(final_state["strict_referee_identity"]) == (
        heldout._STRICT_REFEREE_FIELDS
    )


def test_first_frozen_prefix_replays_through_pinned_sanmill(contract) -> None:
    installation = _training_installation()

    with SanmillTrainingGame(installation, seed=42) as game:
        _board, prefix = heldout.replay_frozen_prefix(game, contract.records[0])

    assert prefix["logical_ply_count"] == 12
    assert prefix["logical_plies_by_side"] == [6, 6]
    assert (
        prefix["observed_history_sha256"]
        == contract.records[0]["execution_record"]["final"]["history_sha256"]
    )
