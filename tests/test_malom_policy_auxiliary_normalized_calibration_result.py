"""Focused tests for normalized auxiliary calibration result analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from learned_ai.evaluation.malom_policy_auxiliary_normalized_calibration_result import (
    MalomPolicyAuxiliaryNormalizedCalibrationResultError,
    decide_normalized_calibration_result,
    publish_result,
    summarize_normalized_update_rows,
)


def _control_update() -> dict:
    return {
        "game": 10,
        "policy_loss": -0.2,
        "value_loss": 0.4,
        "entropy": 2.0,
        "lr": 0.0001,
        "batch_steps": 64,
        "reason": "periodic",
    }


def _normalized_update(
    *,
    status: str = "normalized",
    ordinary_norm: float = 2.0,
    raw_norm: float = 4.0,
    coefficient: float = 0.125,
    capped: bool = False,
    informative: int = 20,
) -> dict:
    applied_norm = coefficient * raw_norm
    ratio = applied_norm / ordinary_norm if ordinary_norm > 1e-12 else 0.0
    cosine = 0.5 if ordinary_norm > 0.0 and raw_norm > 0.0 else None
    return {
        **_control_update(),
        "malom_policy_aux_loss": 1.2 if informative else 0.0,
        "malom_policy_aux_informative_steps": informative,
        "malom_policy_aux_labelled_steps": 64,
        "malom_policy_aux_mean_preserving_mass": 0.7,
        "malom_policy_aux_labelled_by_phase": {
            "placement": 40,
            "movement": 20,
            "flying": 4,
        },
        "malom_policy_aux_informative_by_phase": {
            "placement": informative,
        },
        "malom_policy_aux_mode": "policy-head-normalized",
        "malom_policy_aux_scale_status": status,
        "malom_policy_aux_target_policy_head_ratio": 0.25,
        "malom_policy_aux_coef_cap": 0.25,
        "malom_policy_aux_denominator_floor": 1e-12,
        "malom_policy_aux_effective_coef": coefficient,
        "malom_policy_aux_coefficient_capped": capped,
        "malom_policy_aux_ordinary_policy_head_gradient_l2": ordinary_norm,
        "malom_policy_aux_raw_auxiliary_gradient_l2": raw_norm,
        "malom_policy_aux_applied_auxiliary_gradient_l2": applied_norm,
        "malom_policy_aux_applied_to_ordinary_policy_head_ratio": ratio,
        "malom_policy_auxiliary_to_ordinary_policy_head_cosine": cosine,
    }


def _summarize(rows: list[dict], *, normalized: bool = True) -> dict:
    return summarize_normalized_update_rows(
        rows,
        normalized=normalized,
        expected_games=100,
        target_ratio=0.25,
        coefficient_cap=0.25,
        denominator_floor=1e-12,
    )


def test_normalized_update_summary_preserves_scale_and_phase_evidence() -> None:
    result = _summarize([_normalized_update()])

    summary = result["summary"]
    assert summary["label_coverage"] == 1.0
    assert summary["total_informative_steps"] == 20
    assert summary["effective_coefficient"]["median"] == 0.125
    assert summary["applied_to_ordinary_policy_head_ratio"]["max"] == 0.25
    assert summary["auxiliary_to_ordinary_policy_head_cosine"]["median"] == 0.5
    assert summary["labelled_steps_by_phase"] == {
        "placement": 40,
        "movement": 20,
        "flying": 4,
    }
    assert summary["status_counts"] == {"normalized": 1}


def test_update_curves_omit_incomplete_leading_windows() -> None:
    result = _summarize([_normalized_update() for _ in range(5)])

    assert len(result["curves"]["raw"]) == 5
    rolling = result["curves"]["rolling_5_complete_windows_only"]
    assert len(rolling) == 1
    assert rolling[0]["window_updates"] == 5
    assert rolling[0]["policy_loss"] == -0.2
    assert rolling[0]["malom_policy_aux_effective_coef"] == 0.125


def test_capped_update_keeps_ratio_below_target() -> None:
    result = _summarize(
        [
            _normalized_update(
                status="capped",
                ordinary_norm=10.0,
                raw_norm=1.0,
                coefficient=0.25,
                capped=True,
            )
        ]
    )

    summary = result["summary"]
    assert summary["capped_updates"] == 1
    assert summary["applied_to_ordinary_policy_head_ratio"]["max"] == 0.025


def test_no_informative_and_zero_ordinary_gradient_statuses_are_valid() -> None:
    no_information = _normalized_update(
        status="no_informative_steps",
        ordinary_norm=2.0,
        raw_norm=0.0,
        coefficient=0.0,
        informative=0,
    )
    zero_ordinary = _normalized_update(
        status="ordinary_policy_gradient_below_floor",
        ordinary_norm=0.0,
        raw_norm=4.0,
        coefficient=0.0,
    )

    summary = _summarize([no_information, zero_ordinary])["summary"]

    assert summary["status_counts"] == {
        "no_informative_steps": 1,
        "ordinary_policy_gradient_below_floor": 1,
    }
    assert summary["applied_to_ordinary_policy_head_ratio"]["max"] == 0.0


def test_normalized_update_rejects_a_non_reconciling_ratio() -> None:
    row = _normalized_update()
    row["malom_policy_aux_applied_to_ordinary_policy_head_ratio"] = 0.2

    with pytest.raises(
        MalomPolicyAuxiliaryNormalizedCalibrationResultError,
        match="applied ratio does not reconcile",
    ):
        _summarize([row])


def test_normalized_update_rejects_incomplete_phase_support() -> None:
    row = _normalized_update()
    row["malom_policy_aux_labelled_by_phase"]["placement"] = 39

    with pytest.raises(
        MalomPolicyAuxiliaryNormalizedCalibrationResultError,
        match="phase support does not reconcile",
    ):
        _summarize([row])


def test_control_accepts_only_a_fully_disabled_auxiliary_route() -> None:
    result = _summarize([_control_update()], normalized=False)

    assert result["summary"]["label_coverage"] == 0.0
    assert result["summary"]["status_counts"] == {"disabled": 1}
    changed = _control_update()
    changed["malom_policy_aux_loss"] = 0.0
    with pytest.raises(
        MalomPolicyAuxiliaryNormalizedCalibrationResultError,
        match="control update contains auxiliary diagnostics",
    ):
        _summarize([changed], normalized=False)


def _fixed_state(change: float, entropy_change: float) -> dict:
    return {
        "candidate": {
            "all": {
                "critical_value_preserving_probability_mass_scheduled": (
                    0.5 + change
                ),
                "mean_entropy_scheduled": 2.0 + entropy_change,
            }
        },
        "scratch": {
            "all": {
                "critical_value_preserving_probability_mass_scheduled": 0.5,
                "mean_entropy_scheduled": 2.0,
            }
        },
    }


def _arm(
    seed: int,
    condition: str,
    *,
    mass_change: float,
    entropy_change: float = -0.02,
    repetition_rate: float = 0.02,
    maximum_ratio: float = 0.25,
    health: bool = True,
) -> dict:
    return {
        "arm_id": f"seed{seed}-{condition}",
        "seed": seed,
        "condition": condition,
        "fixed_state_metrics": _fixed_state(mass_change, entropy_change),
        "metrics": {
            "termination": {"repetition_draw_rate": repetition_rate}
        },
        "optimizer_updates": {
            "summary": {
                "label_coverage": 1.0 if condition != "control" else 0.0,
                "total_informative_steps": (
                    20 if condition != "control" else 0
                ),
                "applied_to_ordinary_policy_head_ratio": {
                    "max": maximum_ratio if condition != "control" else None
                },
            }
        },
        "policy_health": {"passed": health},
    }


def _decision_arms() -> list[dict]:
    gains = {55: 0.002, 56: 0.0015, 57: -0.0005}
    arms: list[dict] = []
    for seed in (55, 56, 57):
        arms.append(_arm(seed, "control", mass_change=0.0))
        arms.append(
            _arm(seed, "normalized-0.25", mass_change=gains[seed])
        )
    return arms


def _decision_rule() -> dict:
    return {
        "maximum_fixed_state_entropy_drop_over_control": 0.15,
        "maximum_repetition_draw_rate_increase_over_control": 0.1,
        "minimum_fixed_state_preserving_mass_median_gain": 0.001,
        "minimum_positive_seed_pairs": 2,
        "normalized_applied_ratio_upper_bound": 0.250001,
        "training_wdl_is_not_a_selection_metric": True,
    }


def test_paired_decision_requires_two_positive_seeds_and_median_gate() -> None:
    decision = decide_normalized_calibration_result(
        _decision_arms(), decision_rule=_decision_rule()
    )

    assert decision["eligible"] is True
    assert decision["positive_seed_pairs"] == 2
    assert decision["median_paired_preserving_mass_gain"] == pytest.approx(
        0.0015
    )
    assert decision["training_wdl_used_for_selection"] is False
    assert decision["verdict"] == (
        "normalized_mechanism_eligible_for_effectiveness_experiment"
    )


def test_any_pair_safety_failure_stops_the_mechanism() -> None:
    arms = _decision_arms()
    treatment = next(
        arm
        for arm in arms
        if arm["seed"] == 56 and arm["condition"] == "normalized-0.25"
    )
    treatment["metrics"]["termination"]["repetition_draw_rate"] = 0.2

    decision = decide_normalized_calibration_result(
        arms, decision_rule=_decision_rule()
    )

    assert decision["eligible"] is False
    assert decision["safety_gates_passed"] is False
    assert decision["verdict"] == "inconclusive_stop_and_redesign"


def test_result_publisher_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    report = {"result_identity": "test"}

    publish_result(path, report)
    with pytest.raises(
        MalomPolicyAuxiliaryNormalizedCalibrationResultError,
        match="already exists",
    ):
        publish_result(path, report)
