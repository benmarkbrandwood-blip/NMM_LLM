from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    EstimatorAccess,
    EstimatorReadinessError,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    ARMS,
    EXPECTED_GAMES,
    EXPECTED_STARTS,
    ResourceLedger,
    SafeGuidanceGameplayIncomplete,
    SafeGuidanceGameplayError,
    _assert_canary_selection,
    _checked_position_state,
    _checked_search_result,
    _pooled_action_key,
    analyze_games,
    append_game_record,
    append_resource_checkpoint,
    build_schedule,
    load_game_records,
    load_plan,
    load_pool,
    load_resource_checkpoints,
    select_schedule_excluding_starts,
    validate_game_record,
    verify_resource_game_alignment,
)
from learned_ai.evaluation.sanmill_uci import (
    UciLogicalTurnResult,
    UciPositionState,
)


def _states() -> list[dict]:
    phases = ("placement", "movement", "flying")
    rows = []
    for index in range(EXPECTED_STARTS):
        rows.append(
            {
                "state_id": f"state-{index:03d}",
                "phase": phases[index // 85],
            }
        )
    return rows


def _plan() -> dict:
    return {
        "primary_decision": {"maximum_half_width": 0.015},
    }


def _games(full_score: float = 1.0, random_score: float = 0.5) -> list[dict]:
    rows = []
    for item in build_schedule(_states()):
        score = {
            "random-safe": random_score,
            "full-guided": full_score,
            "geometry-guided": 0.5,
        }[item["arm"]]
        winner = None
        if score == 1.0:
            winner = item["candidate_color"]
        elif score == 0.0:
            winner = "B" if item["candidate_color"] == "W" else "W"
        winner_name = {None: None, "W": "white", "B": "black"}[winner]
        reason = "drawThreefoldRepetition" if score == 0.5 else "win"
        rows.append(
            {
                **item,
                "schema_version": (
                    "nmm.sanmill-safe-guidance-gameplay-game.v1"
                ),
                "unit_index": item["unit_index"],
                "oof_fold": 0,
                "strict_start": {"history_sha256": "history"},
                "post_start_logical_plies": 1,
                "termination_class": "rules_terminal",
                "outcome_reason": reason,
                "winner": winner,
                "candidate_score": score,
                "final_state": {
                    "outcome": {
                        "terminal": True,
                        "winner": winner_name,
                        "winner_code": None,
                        "reason": reason,
                        "reason_code": reason,
                    }
                },
                "final_positional": {
                    "side_to_move": "W",
                    "side_to_move_wdl": "D",
                    "history_aware": False,
                },
                "turns": [
                    {
                        "post_start_ply": 1,
                        "mover_color": item["candidate_color"],
                        "actions": ["a7"],
                        "move": {"from": None, "to": "a7", "capture": None},
                        "candidate_choice": None,
                        "engine_response": None,
                    }
                ],
                "induced_events": [],
                "game_elapsed_seconds": 0.1,
            }
        )
    return rows


def test_tracked_protocol_is_canonical_and_bounded() -> None:
    root = Path(__file__).resolve().parent.parent
    plan, _file_sha = load_plan(
        root / "docs/experiments/sanmill-safe-guidance-gameplay-v1.json"
    )
    assert plan["experiment"]["starts"] == 255
    assert plan["experiment"]["games"] == 1530
    assert plan["precision_preregistration"]["maximum_95_half_width"] == 0.015
    assert plan["resource_envelope"]["maximum_complete_games"] == 1536
    assert plan["resource_envelope"]["maximum_engine_single_step_searches"] == 80000
    assert plan["claim_boundary"]["human_trap_claim"] is False


def test_schedule_is_adjacent_by_start_color_and_arm() -> None:
    schedule = build_schedule(_states())
    assert len(schedule) == EXPECTED_GAMES
    assert [row["arm"] for row in schedule[:3]] == list(ARMS)
    assert [row["candidate_color"] for row in schedule[:6]] == [
        "W",
        "W",
        "W",
        "B",
        "B",
        "B",
    ]
    assert len({row["game_id"] for row in schedule}) == EXPECTED_GAMES


def test_attempt_exclusion_preserves_surviving_game_identities() -> None:
    original = build_schedule(_states())
    selected = select_schedule_excluding_starts(
        original,
        excluded_start_ids=["state-000"],
    )
    assert len(selected) == EXPECTED_GAMES - 6
    assert min(row["ordinal"] for row in selected) == 6
    assert [row["game_id"] for row in selected] == [
        row["game_id"] for row in original[6:]
    ]


def test_attempt_reduced_analysis_requires_exact_surviving_starts() -> None:
    records = [row for row in _games() if row["start_id"] != "state-000"]
    expected = [row["state_id"] for row in _states()[1:]]
    report = analyze_games(records, _plan(), expected_start_ids=expected)
    assert report["completed_starts"] == EXPECTED_STARTS - 1
    assert report["completed_games"] == EXPECTED_GAMES - 6
    with pytest.raises(
        SafeGuidanceGameplayError,
        match="analysis start membership differs",
    ):
        analyze_games(
            records,
            _plan(),
            expected_start_ids=[*expected[:-1], "wrong-start"],
        )


def test_tracked_start_pool_is_complete_history_and_candidate_blind() -> None:
    root = Path(__file__).resolve().parent.parent
    pool, _file_sha = load_pool(
        root
        / "docs/experiments/sanmill-safe-guidance-gameplay-start-pool-v1.json"
    )
    assert pool["pool_identity"] == (
        "385a376dd82953c23c232f34e3dd5a84e5887b978c60627657eccfa6821eb6e9"
    )
    assert pool["state_membership_identity"] == (
        "cb84ed8180b103d7c25d56a5051fb2476047788505ed0cb9f437c39c9048fb15"
    )
    assert pool["prior_coordinate_exclusion"]["coordinates"] == 396
    assert pool["prior_coordinate_exclusion"]["selected_overlap"] == 0
    assert pool["selection_blindness"]["human_estimator_prediction_reads"] == 0
    assert pool["selection_blindness"]["sanmill_observations"] == 0
    assert all(
        len(row["logical_turns"]) == row["logical_ply"] for row in pool["states"]
    )
    assert pool["access_audit"]["official_final_test_content_reads"] == 0
    first_action = pool["states"][0]["a_pos"][0]
    assert _pooled_action_key(first_action) == (
        str(first_action["move"].get("from") or ""),
        str(first_action["move"].get("to") or ""),
        str(first_action["move"].get("capture") or ""),
    )


def test_canary_accepts_matching_nested_move_envelope() -> None:
    actions = [
        {"move": {"from": None, "to": "a7", "capture": None}},
        {"move": {"from": None, "to": "b6", "capture": None}},
    ]
    _assert_canary_selection(
        specification="full",
        actions=actions,
        selected_index=1,
        selected_risk=0.25,
        expected={
            "selected_action": {"from": None, "to": "b6", "capture": None},
            "maximum_risk": 0.25,
        },
    )


def test_canary_rejects_a_genuinely_mismatched_move() -> None:
    """This must fail if the corrected canary loses move discrimination."""
    actions = [
        {"move": {"from": None, "to": "a7", "capture": None}},
        {"move": {"from": None, "to": "b6", "capture": None}},
    ]
    with pytest.raises(
        SafeGuidanceGameplayError,
        match="full guide canary selected move differs",
    ):
        _assert_canary_selection(
            specification="full",
            actions=actions,
            selected_index=0,
            selected_risk=0.25,
            expected={
                "selected_action": {"from": None, "to": "b6", "capture": None},
                "maximum_risk": 0.25,
            },
        )


def test_primary_analysis_clusters_both_colors_at_start() -> None:
    report = analyze_games(_games(), _plan())
    primary = report["primary_full_minus_random_start_clustered_score"]
    assert primary["support"] == EXPECTED_STARTS
    assert primary["mean"] == 0.5
    assert primary["half_width"] == 0.0
    assert report["decision"] == "full_guidance_higher_fixed_runtime_score"


def test_safety_cap_cannot_enter_primary_as_draw() -> None:
    rows = _games()
    rows[0] = {
        **rows[0],
        "termination_class": "safety_cap_incomplete",
        "outcome_reason": "safety_cap_incomplete",
        "winner": None,
        "candidate_score": None,
        "final_state": {
            "outcome": {
                "terminal": False,
                "winner": None,
                "winner_code": None,
                "reason": "ongoing",
                "reason_code": "ongoing",
            }
        },
    }
    with pytest.raises(SafeGuidanceGameplayIncomplete, match="incomplete game"):
        analyze_games(rows, _plan())


def test_resource_ledger_fails_closed_at_any_ceiling() -> None:
    ledger = ResourceLedger(
        engine_searches=0,
        malom_queries=0,
        active_seconds_before_run=0.0,
        maximum_engine_searches=1,
        maximum_malom_queries=1,
        maximum_active_seconds=60.0,
    )
    ledger.add_engine()
    ledger.add_malom(1)
    with pytest.raises(SafeGuidanceGameplayIncomplete, match="engine-search"):
        ledger.add_engine()


def _terminal_state() -> UciPositionState:
    return UciPositionState(
        status="ok",
        ruleset_id="rules",
        rules_identity_sha256="rules-sha",
        rules_options={},
        history_origin="startpos",
        fen="fen",
        side_to_move=None,
        phase="gameover",
        action="none",
        pending_removal_count=0,
        pending_removals=(0, 0),
        legal_actions=(),
        action_token_count=3,
        logical_ply_count=3,
        logical_plies_by_side=(2, 1),
        no_capture_count=0,
        repetition_current_count=3,
        repetition_history_length=3,
        snapshot_history_length=3,
        history_sha256="history",
        terminal=True,
        winner=None,
        winner_code=2,
        outcome_reason="drawThreefoldRepetition",
        outcome_reason_code="5",
        raw_line="state_json {}",
    )


def test_terminal_contract_requires_nested_portable_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _terminal_state()
    assert _checked_position_state(state)["outcome"]["winner"] is None

    def old_broken_portable(_self: UciPositionState) -> dict:
        return {
            "terminal": True,
            "winner": None,
            "outcome_reason": "drawThreefoldRepetition",
            "history_sha256": "history",
            "logical_ply_count": 3,
            "no_capture_count": 0,
            "repetition_current_count": 3,
        }

    monkeypatch.setattr(UciPositionState, "portable_record", old_broken_portable)
    with pytest.raises(
        SafeGuidanceGameplayError,
        match="portable Sanmill outcome is absent",
    ):
        _checked_position_state(state)


def test_game_record_rejects_top_level_nested_winner_mismatch() -> None:
    record = _games()[0]
    record["final_state"]["outcome"]["winner"] = "black"
    with pytest.raises(
        SafeGuidanceGameplayError,
        match="terminal game outcome contract differs",
    ):
        validate_game_record(record)


def test_game_record_requires_inducement_decomposition_before_analysis() -> None:
    record = _games()[0]
    record["induced_events"] = [
        {
            "event_index": 0,
            "transition": "D->L",
            "history_actions_before": [],
            "primary_semantic_search": {},
            "budget_flags": {"100000": True},
            "budget_type": None,
        }
    ]
    validate_game_record(record)
    with pytest.raises(
        SafeGuidanceGameplayError,
        match="induced-event decomposition contract differs",
    ):
        validate_game_record(record, require_decomposition=True)


def test_logical_search_contract_rejects_wrong_budget() -> None:
    result = UciLogicalTurnResult(
        status="ok",
        full_turn_actions=("a7",),
        logical_move_id="a7",
        model_action={"from": None, "to": "a7", "capture": None},
        logical_ply_delta=1,
        resulting_fen="fen",
        resulting_side_to_move="black",
        terminal=False,
        winner=None,
        winner_code=None,
        outcome_reason="ongoing",
        effective_depth=1,
        completed_depth=1,
        score_kind="cp",
        score=0,
        score_perspective="white",
        node_budget=1_000,
        primary_nodes=1_000,
        removal_nodes=0,
        total_nodes=1_000,
        search_calls=1,
        elapsed_seconds=0.1,
        raw_line="bestmove_json {}",
    )
    with pytest.raises(
        SafeGuidanceGameplayError,
        match="logical result contract differs",
    ):
        _checked_search_result(result, expected_node_budget=100_000)


def test_resource_journal_survives_abnormal_subprocess_exit(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "resource-journal.jsonl"
    root = Path(__file__).resolve().parent.parent
    code = f"""
import os
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import append_resource_checkpoint
p = {str(journal)!r}
r0 = {{'engine_single_step_searches': 0, 'malom_read_only_queries': 0, 'active_seconds': 0.0}}
r1 = {{'engine_single_step_searches': 2, 'malom_read_only_queries': 11, 'active_seconds': 1.0}}
r2 = {{'engine_single_step_searches': 5, 'malom_read_only_queries': 29, 'active_seconds': 2.5}}
h1 = append_resource_checkpoint(p, completion_index=0, complete_games_before=4, game_record={{'ordinal': 7, 'game_id': 'game-1'}}, resources_before=r0, resources_after=r1, previous_checkpoint_sha256=None)
append_resource_checkpoint(p, completion_index=1, complete_games_before=4, game_record={{'ordinal': 8, 'game_id': 'game-2'}}, resources_before=r1, resources_after=r2, previous_checkpoint_sha256=h1)
os._exit(73)
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=root, check=False)
    assert result.returncode == 73
    recovered = load_resource_checkpoints(
        journal,
        expected_baseline={
            "engine_single_step_searches": 0,
            "malom_read_only_queries": 0,
            "active_seconds": 0.0,
        },
        complete_games_before=4,
    )
    assert recovered["checkpoint_count"] == 2
    assert recovered["complete_games_after"] == 6
    assert recovered["last_resources"] == {
        "engine_single_step_searches": 5,
        "malom_read_only_queries": 29,
        "active_seconds": 2.5,
    }


def test_resource_and_game_journals_recover_the_same_completed_game(
    tmp_path: Path,
) -> None:
    record = _games()[0]
    baseline = {
        "engine_single_step_searches": 0,
        "malom_read_only_queries": 0,
        "active_seconds": 0.0,
    }
    after = {
        "engine_single_step_searches": 1,
        "malom_read_only_queries": 9,
        "active_seconds": 0.5,
    }
    resource_path = tmp_path / "resources.jsonl"
    games_path = tmp_path / "games.jsonl"
    append_resource_checkpoint(
        resource_path,
        completion_index=0,
        complete_games_before=4,
        game_record=record,
        resources_before=baseline,
        resources_after=after,
        previous_checkpoint_sha256=None,
    )
    append_game_record(games_path, record, previous_record_sha256=None)
    resources = load_resource_checkpoints(
        resource_path,
        expected_baseline=baseline,
        complete_games_before=4,
    )
    games = load_game_records(games_path)
    verify_resource_game_alignment(resources, games)
    assert games["record_count"] == 1

    games["records"][0]["game_id"] = "wrong-game"
    with pytest.raises(
        SafeGuidanceGameplayError,
        match="resource/game recovery alignment differs",
    ):
        verify_resource_game_alignment(resources, games)


def test_protected_guard_raises_before_any_content_producer(tmp_path: Path) -> None:
    access = EstimatorAccess(
        official_partition_by_session={
            "selection": "selection",
            "confirmation": "confirmation",
            "final": "final-test",
            "internal": "train",
        },
        research_partition_by_session={
            "internal": "research-confirmation",
        },
        allowed_sessions=frozenset(),
    )
    called = False

    def producer() -> str:
        nonlocal called
        called = True
        return str(tmp_path)

    for session in ("selection", "confirmation", "final", "internal"):
        with pytest.raises(EstimatorReadinessError, match="denied"):
            access.derive(session, access_kind="gameplay", producer=producer)
    assert called is False


def test_protocol_file_contains_no_result_surface() -> None:
    root = Path(__file__).resolve().parent.parent
    value = json.loads(
        (root / "docs/experiments/sanmill-safe-guidance-gameplay-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert "result" not in value
    assert value["status"] == "frozen_before_start_pool_or_gameplay"
