"""Focused tests for deterministic auxiliary-calibration result analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from learned_ai.evaluation.malom_policy_auxiliary_calibration_result import (
    MalomPolicyAuxiliaryCalibrationResultError,
    decide_calibration_result,
    publish_result,
    summarize_game_rows,
    summarize_update_rows,
)
from learned_ai.training.run_contract import canonical_json_bytes


def _game_row(
    game: int,
    *,
    source: str,
    colour: str,
    coefficient: float,
) -> dict:
    active = coefficient > 0.0
    return {
        "game_id": f"game:{game:04d}",
        "game": game,
        "difficulty": 1,
        "learner_color": colour,
        "temperature": 0.9 - game / 10000,
        "outcome": -1.0,
        "ply": 30,
        "steps": 15,
        "update_policy_loss": None,
        "update_value_loss": None,
        "update_entropy": None,
        "reward_total_mean": -0.5,
        "reward_mill_bonus_mean": 0.0,
        "mill_bonus_awarded_total": 0.0,
        "chosen_prob_mean": 0.2,
        "entropy_mean": 2.0,
        "policy_top1_rate": 0.2,
        "heuristic_top1_rate": 0.1,
        "malom_preserving_move_rate": 0.8,
        "malom_downgrade_move_rate": 0.2,
        "game_type": source,
        "phase_bucket": "main",
        "is_branch": 0,
        "termination_reason": "lose_no_legal_moves",
        "opponent_node_budget": 1000 if source == "vs_sanmill" else None,
        "formed_mill_count": 1,
        "formed_mill_move_count": 1,
        "formed_mill_malom_unknown_count": 0,
        "formed_mill_malom_downgrade_count": 0,
        "formed_mill_malom_downgrade_rate": 0.0,
        "formed_mill_malom_known_place": 1,
        "formed_mill_malom_known_move": 0,
        "formed_mill_malom_known_fly": 0,
        "formed_mill_malom_downgrade_place": 0,
        "formed_mill_malom_downgrade_move": 0,
        "formed_mill_malom_downgrade_fly": 0,
        "malom_action_labelled_move_rate": 1.0 if active else 0.0,
        "malom_preserving_action_count_mean": 3.0 if active else 0.0,
        "malom_downgrading_action_count_mean": 1.0 if active else 0.0,
        "malom_informative_action_set_rate": 0.5 if active else 0.0,
        "malom_preserving_probability_mean": 0.7 if active else 0.0,
        "malom_known_move_rate": 1.0,
        "malom_known_place": 15,
        "malom_known_move": 0,
        "malom_known_fly": 0,
        "malom_downgrade_place": 3,
        "malom_downgrade_move": 0,
        "malom_downgrade_fly": 0,
        "malom_downgrade_count": 3,
    }


def _hundred_rows(coefficient: float) -> tuple[list[dict], dict[str, int]]:
    classes = (
        ("vs_frozen", "B"),
        ("vs_frozen", "W"),
        ("vs_sanmill", "B"),
        ("vs_sanmill", "W"),
    )
    rows = [
        _game_row(
            game,
            source=classes[(game - 1) % 4][0],
            colour=classes[(game - 1) % 4][1],
            coefficient=coefficient,
        )
        for game in range(1, 101)
    ]
    schedule = {
        "frozen_black": 25,
        "frozen_white": 25,
        "sanmill_black": 25,
        "sanmill_white": 25,
    }
    return rows, schedule


def test_game_summary_requires_active_labels_and_keeps_complete_windows() -> None:
    rows, schedule = _hundred_rows(0.1)

    summary = summarize_game_rows(
        rows,
        coefficient=0.1,
        expected_games=100,
        expected_schedule_counts=schedule,
    )

    assert summary["games"] == 100
    assert summary["selected_action_quality"]["whole_run"] == {
        "known_actions": 1500,
        "downgrading_actions": 300,
        "downgrade_rate": 0.2,
        "by_phase": {
            "place": {
                "known_actions": 1500,
                "downgrading_actions": 300,
                "downgrade_rate": 0.2,
            },
            "move": {
                "known_actions": 0,
                "downgrading_actions": 0,
                "downgrade_rate": None,
            },
            "fly": {
                "known_actions": 0,
                "downgrading_actions": 0,
                "downgrade_rate": None,
            },
        },
    }
    assert len(summary["curves"]["raw"]) == 100
    assert len(summary["curves"]["rolling_50_complete_windows_only"]) == 51
    assert summary["curves"]["validation"]["available"] is False


def test_game_summary_rejects_missing_active_label_support() -> None:
    rows, schedule = _hundred_rows(0.1)
    rows[4]["malom_action_labelled_move_rate"] = 0.5

    with pytest.raises(
        MalomPolicyAuxiliaryCalibrationResultError,
        match="incomplete game label support",
    ):
        summarize_game_rows(
            rows,
            coefficient=0.1,
            expected_games=100,
            expected_schedule_counts=schedule,
        )


def test_game_summary_accepts_the_zero_coefficient_control_route() -> None:
    rows, schedule = _hundred_rows(0.0)

    summary = summarize_game_rows(
        rows,
        coefficient=0.0,
        expected_games=100,
        expected_schedule_counts=schedule,
    )

    assert summary["games"] == 100
    assert all(
        point["malom_action_labelled_move_rate"] == 0.0
        for point in summary["curves"]["raw"]
    )


def _update_row(*, coefficient: float, labelled_steps: int | None = None) -> dict:
    active = coefficient > 0.0
    return {
        "game": 10,
        "policy_loss": -0.2,
        "value_loss": 0.4,
        "entropy": 2.0,
        "lr": 0.0001,
        "batch_steps": 64,
        "reason": "periodic",
        "malom_policy_aux_loss": 1.5 if active else 0.0,
        "malom_policy_aux_informative_steps": 20 if active else 0,
        "malom_policy_aux_labelled_steps": (
            labelled_steps if labelled_steps is not None else (64 if active else 0)
        ),
        "malom_policy_aux_mean_preserving_mass": 0.7 if active else 0.0,
    }


def test_update_summary_quantifies_scaled_auxiliary_pressure() -> None:
    summary = summarize_update_rows(
        [_update_row(coefficient=0.1), _update_row(coefficient=0.1)],
        coefficient=0.1,
        expected_games=100,
    )

    metrics = summary["summary"]
    assert metrics["label_coverage"] == 1.0
    assert metrics["informative_rate"] == pytest.approx(20 / 64)
    assert metrics["median_absolute_policy_loss"] == 0.2
    assert metrics["median_scaled_auxiliary_loss"] == pytest.approx(0.15)
    assert metrics[
        "scaled_auxiliary_to_absolute_policy_loss_ratio"
    ] == pytest.approx(0.75)


def test_update_summary_rejects_incomplete_exact_labels() -> None:
    with pytest.raises(
        MalomPolicyAuxiliaryCalibrationResultError,
        match="incomplete exact labels",
    ):
        summarize_update_rows(
            [_update_row(coefficient=0.1, labelled_steps=63)],
            coefficient=0.1,
            expected_games=100,
        )


def _decision_arm(
    coefficient: float,
    *,
    mass: float,
    entropy: float = 2.5,
    scratch_mass: float = 0.36,
    scratch_entropy: float = 2.6,
    repetition: float = 0.05,
    scale_ratio: float | None = 0.5,
) -> dict:
    scratch = {
        "all": {
            "critical_value_preserving_probability_mass_scheduled": (
                scratch_mass
            ),
            "mean_entropy_scheduled": scratch_entropy,
        }
    }
    return {
        "arm_id": f"c{coefficient}",
        "malom_policy_aux_coef": coefficient,
        "fixed_state_metrics": {
            "scratch": scratch,
            "candidate": {
                "all": {
                    "critical_value_preserving_probability_mass_scheduled": mass,
                    "mean_entropy_scheduled": entropy,
                }
            },
        },
        "metrics": {
            "termination": {"repetition_draw_rate": repetition}
        },
        "optimizer_updates": {
            "summary": {
                "scaled_auxiliary_to_absolute_policy_loss_ratio": scale_ratio,
                "label_coverage": 0.0 if coefficient == 0.0 else 1.0,
                "total_informative_steps": 0 if coefficient == 0.0 else 10,
            }
        },
        "policy_health": {"passed": True},
    }


def _decision_rule() -> dict:
    return {
        "eligible": "text",
        "selection": "text",
        "unsafe": "text",
        "minimum_fixed_state_preserving_mass_gain_over_control": 0.001,
        "maximum_scaled_auxiliary_to_absolute_policy_loss_ratio": 1.0,
        "maximum_repetition_draw_rate_increase_over_control": 0.1,
        "maximum_fixed_state_entropy_drop_over_control": 0.15,
    }


def test_decision_selects_the_lowest_eligible_nonzero_coefficient() -> None:
    arms = [
        _decision_arm(0.0, mass=0.4),
        _decision_arm(0.03, mass=0.4005),
        _decision_arm(0.1, mass=0.402, scale_ratio=0.8),
        _decision_arm(0.3, mass=0.405, scale_ratio=0.4),
    ]

    decision = decide_calibration_result(arms, decision_rule=_decision_rule())

    assert decision["verdict"] == (
        "coefficient_selected_for_multiseed_effectiveness_preparation"
    )
    assert decision["selected_coefficient"] == 0.1
    assert not decision["comparisons"][0]["eligible"]
    assert decision["comparisons"][1]["eligible"]


def test_decision_is_inconclusive_when_safety_or_scale_gates_fail() -> None:
    arms = [
        _decision_arm(0.0, mass=0.4),
        _decision_arm(0.03, mass=0.402, scale_ratio=1.1),
        _decision_arm(0.1, mass=0.402, entropy=2.3),
        _decision_arm(0.3, mass=0.402, repetition=0.2),
    ]

    decision = decide_calibration_result(arms, decision_rule=_decision_rule())

    assert decision["verdict"] == "inconclusive_recalibration_required"
    assert decision["selected_coefficient"] is None
    assert not any(item["eligible"] for item in decision["comparisons"])


def test_decision_normalizes_each_candidate_to_its_own_scratch_route() -> None:
    arms = [
        _decision_arm(0.0, mass=0.4, scratch_mass=0.36),
        _decision_arm(0.03, mass=0.42, scratch_mass=0.38),
        _decision_arm(0.1, mass=0.4215, scratch_mass=0.38),
        _decision_arm(0.3, mass=0.4, scratch_mass=0.36),
    ]

    decision = decide_calibration_result(arms, decision_rule=_decision_rule())

    assert not decision["comparisons"][0]["eligible"]
    assert decision["comparisons"][0][
        "fixed_state_preserving_mass_gain_over_control_change"
    ] == pytest.approx(0.0)
    assert decision["selected_coefficient"] == 0.1


def test_result_publication_is_canonical_and_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    report = {"schema_version": "test", "result_identity": "a" * 64}

    publish_result(output, report)

    assert output.read_bytes() == canonical_json_bytes(report)
    with pytest.raises(
        MalomPolicyAuxiliaryCalibrationResultError,
        match="already exists",
    ):
        publish_result(output, report)
