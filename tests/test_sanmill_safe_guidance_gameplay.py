from __future__ import annotations

import json
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
    _pooled_action_key,
    analyze_games,
    build_schedule,
    load_plan,
    load_pool,
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
        rows.append(
            {
                **item,
                "termination_class": "rules_terminal",
                "outcome_reason": "drawThreefoldRepetition" if score == 0.5 else "win",
                "candidate_score": score,
                "induced_events": [],
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
        "candidate_score": None,
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
