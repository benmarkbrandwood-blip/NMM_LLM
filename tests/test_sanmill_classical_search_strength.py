from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from learned_ai.evaluation.human_f0h0_feasibility import canonical_sha256
from learned_ai.evaluation.sanmill_classical_search_strength import (
    ClassicalSearchStrengthError,
    calibration_membership,
    calibration_summary,
    exact_subset_gate,
    paired_interval,
    phase_balanced_membership,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _state(index: int, phase: str, board: str = "WWBBB...................") -> dict:
    return {
        "state_id": f"state-{phase}-{index:03d}",
        "phase": phase,
        "fen": f"{board}|W|5|4",
    }


def test_phase_balanced_membership_is_deterministic_and_excludes() -> None:
    states = [
        _state(index, phase)
        for phase in ("placement", "movement", "flying")
        for index in range(12)
    ]
    excluded = ["state-placement-000", "state-movement-000"]
    first = phase_balanced_membership(
        states,
        count=10,
        namespace="test-membership",
        excluded_start_ids=excluded,
    )
    second = phase_balanced_membership(
        list(reversed(states)),
        count=10,
        namespace="test-membership",
        excluded_start_ids=excluded,
    )
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert not set(first) & set(excluded)
    assert canonical_sha256(first) == canonical_sha256(second)


def test_calibration_membership_rejects_early_placement_states() -> None:
    states = [
        _state(index, "placement", "WWBB....................")
        for index in range(4)
    ]
    states += [_state(index + 10, "placement") for index in range(4)]
    states += [
        _state(index, phase)
        for phase in ("movement", "flying")
        for index in range(4)
    ]
    selected = calibration_membership(
        states,
        per_phase=4,
        namespace="test-calibration",
        excluded_start_ids=[],
    )
    assert len(selected) == 12
    assert all("placement-00" not in state_id for state_id in selected[:4])


def test_calibration_summary_uses_positive_node_median_and_rounds_down() -> None:
    rows = []
    for difficulty, nodes in ((9, [0, 11_999, 20_500, 30_999]), (10, [21_000, 31_999, 41_001])):
        for index, node_count in enumerate(nodes):
            rows.append(
                {
                    "difficulty": difficulty,
                    "nodes": node_count,
                    "elapsed_seconds": float(index + 1),
                    "completed_depth": 4,
                }
            )
    result = calibration_summary(rows)
    assert result["9"]["mapped_node_budget"] == 20_000
    assert result["9"]["bypassed_states"] == 1
    assert result["10"]["mapped_node_budget"] == 31_000


def _known_answer_record() -> dict:
    return {
        "game_id": "g",
        "start_id": "s",
        "candidate_color": "W",
        "candidate_score": 0.5,
        "winner": None,
        "outcome_reason": "drawFiftyMove",
        "post_start_logical_plies": 1,
        "final_state": {
            "history_sha256": "history",
            "no_capture_count": 100,
            "repetition_current_count": 1,
        },
        "turns": [
            {
                "actions": ["(0,0)", "(1,1)"],
                "history_sha256_after": "history",
                "no_capture_count": 100,
                "repetition_current_count": 1,
                "outcome_reason": "drawFiftyMove",
            }
        ],
    }


def test_known_answer_gate_rejects_a_real_clock_mismatch() -> None:
    expected = _known_answer_record()
    observed = copy.deepcopy(expected)
    observed["turns"][0]["no_capture_count"] = 99
    gate = exact_subset_gate([observed], [expected])
    assert gate["passed"] is False
    assert gate["differing_game_ids"] == ["g"]


def test_known_answer_gate_accepts_an_exact_record() -> None:
    record = _known_answer_record()
    gate = exact_subset_gate([record], [copy.deepcopy(record)])
    assert gate["passed"] is True
    assert gate["observed_identity"] == gate["reference_identity"]


def test_paired_interval_requires_multiple_independent_starts() -> None:
    with pytest.raises(ClassicalSearchStrengthError):
        paired_interval([0.0])
    interval = paired_interval([0.0, 0.5, -0.5, 0.0])
    assert interval["support"] == 4
    assert interval["mean"] == 0.0
    assert interval["half_width"] > 0.0


def test_frozen_calibration_plan_binds_implementation_and_zero_games() -> None:
    path = ROOT / "docs/experiments/sanmill-classical-search-calibration-v1.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    body = dict(plan)
    identity = body.pop("plan_identity")
    assert canonical_sha256(body) == identity
    assert plan["resource_envelope"]["complete_games"] == 0
    assert plan["calibration"]["no_complete_games"] is True
    assert {
        name: sha256_file(ROOT / name)
        for name in plan["implementation_files"]
    } == plan["implementation_files"]
