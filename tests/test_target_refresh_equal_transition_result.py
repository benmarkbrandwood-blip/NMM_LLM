"""Tests for the frozen equal-transition policy-distribution decision rule."""

from __future__ import annotations

import copy

import pytest

from learned_ai.evaluation.target_refresh_equal_transition_result import (
    EXPECTED_BOUNDARIES,
    EXPECTED_SEEDS,
    TargetRefreshEqualTransitionResultError,
    classify_transition_policy_divergence,
)


def _summary(*, js: float, tv: float, malom: float) -> dict:
    phase_record = {
        "mean_jensen_shannon_nats": js,
        "mean_total_variation": tv,
        "mean_abs_malom_preserving_probability_mass_delta": abs(malom),
        "mean_no_refresh_minus_refresh_malom_preserving_mass": malom,
    }
    return {
        phase: {
            "top1_agreement_rate": 0.75,
            "distributions": {"temperature_0.2": copy.deepcopy(phase_record)},
        }
        for phase in ("all", "placement", "movement", "flying")
    }


def _matrix(*, js: float = 1e-6, tv: float = 1e-4, malom: float = 1e-4):
    return {
        str(seed): {
            str(boundary): _summary(js=js, tv=tv, malom=malom)
            for boundary in EXPECTED_BOUNDARIES
        }
        for seed in EXPECTED_SEEDS
    }


def test_all_three_final_boundaries_near_identical_stop() -> None:
    decision = classify_transition_policy_divergence(_matrix())

    assert decision["classification"] == "near_identical"
    assert decision["next_design"] == "stop_without_automatic_extension"
    assert decision["persistent_material_seeds"] == []


def test_two_seeds_with_persistent_js_divergence_are_material() -> None:
    matrix = _matrix(js=0.001, tv=0.03, malom=0.03)
    for seed in (64, 65):
        matrix[str(seed)]["4096"] = _summary(js=0.006, tv=0.03, malom=0.03)
        matrix[str(seed)]["8192"] = _summary(js=0.007, tv=0.04, malom=0.04)

    decision = classify_transition_policy_divergence(matrix)

    assert decision["classification"] == "materially_diverged"
    assert decision["persistent_material_seeds"] == ["64", "65"]
    assert decision["by_seed"]["64"][
        "persistent_material_triggers_4096_to_8192"
    ] == ["all_mean_jensen_shannon"]


def test_malom_trigger_requires_same_direction_at_both_boundaries() -> None:
    matrix = _matrix(js=0.001, tv=0.03, malom=0.03)
    for seed in (64, 65):
        matrix[str(seed)]["4096"] = _summary(js=0.001, tv=0.03, malom=0.06)
        matrix[str(seed)]["8192"] = _summary(js=0.001, tv=0.03, malom=-0.07)

    decision = classify_transition_policy_divergence(matrix)

    assert decision["classification"] == "inconclusive_late_onset"
    assert decision["persistent_material_seeds"] == []
    assert decision["final_only_material_seeds"] == ["64", "65"]


def test_material_effect_first_seen_only_at_final_is_late_onset() -> None:
    matrix = _matrix(js=0.001, tv=0.03, malom=0.03)
    for seed in (64, 66):
        matrix[str(seed)]["8192"] = _summary(js=0.008, tv=0.06, malom=0.03)

    decision = classify_transition_policy_divergence(matrix)

    assert decision["classification"] == "inconclusive_late_onset"
    assert decision["final_only_material_seeds"] == ["64", "66"]


def test_one_persistent_seed_is_inconclusive() -> None:
    matrix = _matrix(js=0.001, tv=0.03, malom=0.03)
    matrix["64"]["4096"] = _summary(js=0.008, tv=0.06, malom=0.03)
    matrix["64"]["8192"] = _summary(js=0.008, tv=0.06, malom=0.03)

    decision = classify_transition_policy_divergence(matrix)

    assert decision["classification"] == "inconclusive"
    assert decision["persistent_material_seeds"] == ["64"]


def test_missing_seed_or_boundary_fails_closed() -> None:
    matrix = _matrix()
    del matrix["66"]["2048"]

    with pytest.raises(
        TargetRefreshEqualTransitionResultError,
        match="transition boundaries differ",
    ):
        classify_transition_policy_divergence(matrix)


def test_fresh_three_seed_set_uses_same_frozen_rule() -> None:
    seeds = (67, 68, 69)
    matrix = {
        str(seed): {
            str(boundary): _summary(js=1e-6, tv=1e-4, malom=1e-4)
            for boundary in EXPECTED_BOUNDARIES
        }
        for seed in seeds
    }

    decision = classify_transition_policy_divergence(matrix, seeds=seeds)

    assert decision["classification"] == "near_identical"
    assert decision["seeds"] == [67, 68, 69]
