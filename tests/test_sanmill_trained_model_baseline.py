from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import learned_ai.evaluation.sanmill_trained_model_baseline as baseline
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    EstimatorAccess,
    EstimatorReadinessError,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    append_resource_checkpoint,
    load_resource_checkpoints,
)


def _terminal_record() -> dict:
    choice = {
        "candidate_id": "retained-v4",
        "candidate_runtime_identity": "a" * 64,
        "route_phase": "training-aligned-route",
        "safety_mode": "free",
        "safe_set": None,
        "positional_only": True,
        "parent_tier": "D",
        "selected_after_tier": "L",
        "a_pos_cardinality": 2,
        "self_downgrade_transition": "D->L",
        "selected_move": {"from": "a1", "to": "a4", "capture": None},
        "legal_move_count": 3,
        "allowed_move_count": 3,
        "selected_legal_index": 0,
        "selected_allowed_index": 0,
        "selected_score": 1.0,
        "exact_score_tie_count_within_allowed": 1,
        "score_vector_identity": "b" * 64,
    }
    turn = {
        "post_start_ply": 1,
        "absolute_logical_ply": 20,
        "mover_color": "W",
        "actor": "candidate",
        "phase": "movement",
        "move": {"from": "a1", "to": "a4", "capture": None},
        "actions": ["a1-a4"],
        "history_sha256_before": "c" * 64,
        "history_sha256_after": "d" * 64,
        "no_capture_count": 1,
        "repetition_current_count": 1,
        "repetition_history_length": 20,
        "terminal": True,
        "outcome_reason": "loseNoLegalMoves",
        "candidate_choice": choice,
        "engine_search": None,
    }
    return {
        "schema_version": baseline.GAME_SCHEMA,
        "ordinal": 0,
        "game_id": "game-0",
        "unit_index": 0,
        "start_id": "start-0",
        "phase": "movement",
        "arm": "retained-v4-free",
        "candidate_color": "W",
        "strict_start": {"logical_ply_count": 19},
        "post_start_logical_plies": 1,
        "termination_class": "rules_terminal",
        "outcome_reason": "loseNoLegalMoves",
        "winner": "W",
        "candidate_score": 1.0,
        "final_state": {
            "outcome": {
                "terminal": True,
                "winner": "white",
                "reason": "loseNoLegalMoves",
            }
        },
        "final_positional": {
            "side_to_move": "B",
            "side_to_move_wdl": "L",
            "history_aware": False,
        },
        "turns": [turn],
        "self_downgrade_events": [
            {
                "post_start_ply": 1,
                "phase": "movement",
                "transition": "D->L",
                "move": turn["move"],
            }
        ],
        "game_elapsed_seconds": 0.1,
        "rehearsal_only": False,
    }


def test_schedule_reuses_exact_attempt_002_formal_membership() -> None:
    root = Path(__file__).resolve().parent.parent
    pool = json.loads(
        (
            root
            / "docs/experiments/"
            "sanmill-safe-guidance-gameplay-start-pool-v1.json"
        ).read_text(encoding="utf-8")
    )
    states = baseline.formal_states(
        pool,
        excluded_start_ids=[
            "00092c974cabf05874f066b8948e791f9fdc82d84a65759da1ba78f212a643b0"
        ],
    )
    schedule = baseline.build_schedule(states, namespace="test-schedule")
    assert len(states) == 254
    assert len(schedule) == 2032
    assert {row["arm"] for row in schedule} == set(baseline.ARMS)
    assert all(
        sum(row["start_id"] == state["state_id"] for row in schedule) == 8
        for state in states
    )


def test_allowed_argmax_rejects_a_truly_missing_move() -> None:
    legal = [
        {"from": None, "to": "a1", "capture": None},
        {"from": None, "to": "a4", "capture": None},
    ]
    scores = np.asarray([0.1, 0.9], dtype=np.float64)
    move, audit = baseline._select_scored_move(
        legal_moves=legal,
        scores=scores,
        allowed_keys={("", "a1", "")},
    )
    assert move == legal[0]
    assert audit["selected_score"] == 0.1

    with pytest.raises(baseline.TrainedModelBaselineError, match="subset"):
        baseline._select_scored_move(
            legal_moves=legal,
            scores=scores,
            allowed_keys={("", "g7", "")},
        )


def test_counting_malom_proxy_records_each_completed_query() -> None:
    class Delegate:
        def query(self, board: object) -> str:
            assert board == "board"
            return "value"

    class Ledger:
        def __init__(self) -> None:
            self.queries = 0

        def add_malom(self, count: int) -> None:
            self.queries += count

    ledger = Ledger()
    proxy = baseline._CountingMalomProxy(Delegate(), ledger)
    assert proxy.query("board") == "value"
    assert ledger.queries == 1


def test_game_record_rejects_truly_mismatched_terminal_winner() -> None:
    record = _terminal_record()
    baseline.validate_game_record(record)
    record["final_state"]["outcome"]["winner"] = "black"
    with pytest.raises(baseline.TrainedModelBaselineError, match="terminal"):
        baseline.validate_game_record(record)


def test_resource_checkpoint_survives_before_game_record_crash(tmp_path: Path) -> None:
    record = _terminal_record()
    resource_path = tmp_path / "resource.jsonl"
    before = {
        "engine_single_step_searches": 0,
        "malom_read_only_queries": 0,
        "active_seconds": 0.0,
    }
    after = {
        "engine_single_step_searches": 1,
        "malom_read_only_queries": 7,
        "active_seconds": 0.25,
    }
    append_resource_checkpoint(
        resource_path,
        completion_index=0,
        complete_games_before=10,
        game_record=record,
        resources_before=before,
        resources_after=after,
        previous_checkpoint_sha256=None,
    )
    recovered = load_resource_checkpoints(
        resource_path,
        expected_baseline=before,
        complete_games_before=10,
    )
    assert recovered["checkpoint_count"] == 1
    assert recovered["last_resources"] == after
    assert not (tmp_path / "games.jsonl").exists()


def test_specialist_gameai_audit_proves_no_score_path_read() -> None:
    result = baseline.audit_specialist_gameai_dependency()
    assert result["read_methods"] == []
    assert result["score_path_reads_gameai"] is False
    assert result["presearch_effect_on_successful_argmax"] is False


def test_protected_guard_fails_before_any_content_producer() -> None:
    access = EstimatorAccess(
        official_partition_by_session={
            "selection": "selection",
            "confirmation": "confirmation",
            "final": "final-test",
            "research": "train",
        },
        research_partition_by_session={"research": "research-confirmation"},
        allowed_sessions=frozenset(),
    )
    called = False

    def producer() -> str:
        nonlocal called
        called = True
        return "forbidden"

    for session in ("selection", "confirmation", "final", "research"):
        with pytest.raises(EstimatorReadinessError, match="denied"):
            access.derive(session, access_kind="gameplay", producer=producer)
    assert called is False


def test_frozen_plan_contains_no_candidate_result_surface() -> None:
    root = Path(__file__).resolve().parent.parent
    plan, _digest = baseline.load_plan(
        root / "docs/experiments/sanmill-trained-model-baseline-v1.json"
    )
    assert plan["status"] == "frozen_before_rehearsal_or_candidate_outcomes"
    assert "result" not in plan
    assert plan["experiment"]["formal_games"] == 2032
    assert plan["resource_envelope"]["planned_total_complete_games"] == 2042
