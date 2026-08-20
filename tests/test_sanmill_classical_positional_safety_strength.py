from __future__ import annotations

import copy

import pytest

from learned_ai.evaluation.sanmill_classical_positional_safety_strength import (
    ClassicalPositionalSafetyStrengthError,
    analyze_filtered_contrasts,
    compare_classical_ledgers,
    restricted_root_select,
)


def _classical_record(*, move_to: str = "a1", clock: int = 4) -> dict:
    return {
        "game_id": "game-1",
        "start_id": "start-1",
        "candidate_color": "W",
        "candidate_score": 0.5,
        "winner": None,
        "outcome_reason": "drawThreefoldRepetition",
        "post_start_logical_plies": 1,
        "final_state": {
            "history_sha256": "history-1",
            "no_capture_count": clock,
            "repetition_current_count": 3,
        },
        "turns": [
            {
                "actor": "classical-search",
                "phase": "placement",
                "move": {"from": None, "to": move_to, "capture": None},
                "actions": [f"({move_to})"],
                "history_sha256_after": "history-1",
                "no_capture_count": clock,
                "repetition_current_count": 3,
                "repetition_history_length": 4,
                "terminal": True,
                "outcome_reason": "drawThreefoldRepetition",
                "candidate_choice": {
                    "parent_tier": "D",
                    "selected_after_tier": "D",
                    "self_downgrade_transition": None,
                    "search": {
                        "nodes": 10,
                        "completed_depth": 2,
                        "bypassed_search": False,
                    },
                },
            }
        ],
    }


def test_classical_comparison_rejects_a_real_turn_difference() -> None:
    reference = _classical_record()
    observed = copy.deepcopy(reference)
    observed["turns"][0]["move"]["to"] = "d1"
    comparison = compare_classical_ledgers([observed], [reference])
    assert comparison["exact_match"] is False
    assert comparison["differing_game_ids"] == ["game-1"]
    assert comparison["difference_categories"] == {"move_sequence": 1}


def test_restricted_root_select_is_score_then_canonical() -> None:
    moves = [
        {"from": "a1", "to": "a4", "capture": None},
        {"from": "d1", "to": "d2", "capture": None},
        {"from": "g1", "to": "g4", "capture": None},
    ]
    selected = restricted_root_select(
        [
            (moves[2], 0.5),
            (moves[1], 0.75),
            (moves[0], 0.75),
        ]
    )
    assert selected == moves[0]


def test_restricted_root_select_fails_closed_on_empty_scores() -> None:
    with pytest.raises(ClassicalPositionalSafetyStrengthError):
        restricted_root_select([])


def _paired_game(
    *, arm: str, difficulty: int, start: str, color: str, score: float
) -> dict:
    filtered = arm.endswith("a-pos")
    return {
        "arm": arm,
        "difficulty": difficulty,
        "start_id": start,
        "phase": "placement",
        "candidate_color": color,
        "candidate_score": score,
        "termination_class": "rules_terminal",
        "outcome_reason": "drawFiftyMove",
        "turns": [
            {
                "actor": "classical-search",
                "phase": "placement",
                "candidate_choice": {
                    "intervened": False,
                    "self_downgrade_transition": None,
                    "gate_decision": (
                        {
                            "selection_rule": "original-already-in-A_pos",
                            "selection_error": None,
                        }
                        if filtered
                        else None
                    ),
                    "search": {
                        "bypassed_search": False,
                        "nodes": 10,
                        "completed_depth": 2,
                        "elapsed_seconds": 0.01,
                    },
                    "restricted_root_research": {
                        "called": False,
                        "nodes": 0,
                        "elapsed_seconds": 0.0,
                    },
                },
            }
        ],
        "self_downgrade_events": [],
    }


def test_primary_analysis_clusters_two_colors_by_start() -> None:
    records = []
    for start in ("s1", "s2", "s3", "s4"):
        for color in ("W", "B"):
            for difficulty in (9, 10):
                records.append(
                    _paired_game(
                        arm=f"classical-d{difficulty}-unfiltered",
                        difficulty=difficulty,
                        start=start,
                        color=color,
                        score=0.5,
                    )
                )
                records.append(
                    _paired_game(
                        arm=f"classical-d{difficulty}-a-pos",
                        difficulty=difficulty,
                        start=start,
                        color=color,
                        score=(
                            1.0
                            if difficulty == 9 and start == "s1"
                            else 0.5
                        ),
                    )
                )
    analysis = analyze_filtered_contrasts(
        records,
        start_ids=("s1", "s2", "s3", "s4"),
        maximum_half_width=1.0,
    )
    primary = analysis["primary"]["difficulty_9_filtered_minus_unfiltered"]
    assert primary["support"] == 4
    assert primary["mean"] == pytest.approx(0.125)
    assert primary["precision_adequate"] is True
