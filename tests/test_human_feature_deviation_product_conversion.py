from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from learned_ai.evaluation.human_f0h0_feasibility import canonical_sha256
from learned_ai.evaluation.human_feature_deviation_product_conversion import (
    ConversionError,
    apply_logistic_calibrator,
    fit_logistic_calibrator,
    jensen_shannon_divergence,
    load_conversion_plan,
    load_effective_conversion_plan,
    normalized_safe_weights,
    product_scenario_thresholds,
    tier_loss_outcomes,
    uplift_against_reference,
)


ROOT = Path(__file__).resolve().parent.parent
PLAN = (
    ROOT
    / "docs/experiments/human-feature-deviation-product-conversion-"
    "derivation-v1.json"
)
PLAN_V2 = (
    ROOT
    / "docs/experiments/human-feature-deviation-product-conversion-"
    "derivation-v2.json"
)


def test_conversion_plan_is_sealed_and_not_a_new_research_question() -> None:
    plan, _file_sha = load_conversion_plan(PLAN)
    body = dict(plan)
    identity = body.pop("derivation_identity")

    assert canonical_sha256(body) == identity
    assert identity == (
        "5f45c0296d86553b7fa67e1aea45269690da15ccf00d840f0c3709479c575e1b"
    )
    assert plan["not_a_new_research_preregistration"] is True
    assert plan["execution_boundary"]["research_confirmation_reads"] == 0
    assert plan["execution_boundary"]["official_final_test_reads"] == 0


def test_tier_loss_mapping_keeps_d_to_l_separate() -> None:
    assert tier_loss_outcomes("W") == frozenset({"D", "L"})
    assert tier_loss_outcomes("D") == frozenset({"L"})
    assert tier_loss_outcomes("L") == frozenset()
    with pytest.raises(ConversionError):
        tier_loss_outcomes("unknown")


def test_v2_corrects_only_machine_exact_reproduction_values() -> None:
    effective, identities = load_effective_conversion_plan(
        PLAN_V2, inherited_v1_path=PLAN
    )
    required = effective["frozen_estimator_reuse"]["required_reproduction"]

    assert identities["v2_derivation_identity"] == (
        "d4ead92b6ff1b8be07d5a16ab1041547e973ee50d4ae794e3ab5fc1a06d20444"
    )
    assert required["geometry_average_unique_player_log_loss"] == (
        2.146902903833993
    )
    assert required["full_average_unique_player_log_loss"] == 1.986710540646765
    assert required["paired_log_loss_improvement"] == 0.1601923631872277
    assert required["tolerance"] == 1e-12


def test_safe_reference_weights_normalize_and_fail_closed() -> None:
    weights = normalized_safe_weights(
        np.asarray([0.1, 0.2, 0.7]), np.asarray([True, False, True])
    )
    assert np.allclose(weights, [0.125, 0.875])

    with pytest.raises(ConversionError):
        normalized_safe_weights(
            np.asarray([0.0, 1.0]), np.asarray([False, False])
        )


def test_singleton_uplift_is_exactly_zero() -> None:
    result = uplift_against_reference(
        np.asarray([0.37]), np.asarray([1.0])
    )
    assert result == {
        "maximum": 0.37,
        "reference": 0.37,
        "uplift": 0.0,
        "argmax_index": 0,
    }


def test_calibrator_preserves_exact_zero_without_a_prior() -> None:
    probabilities = np.asarray([0.05, 0.1, 0.2, 0.4, 0.7] * 20)
    events = np.asarray([0, 0, 0, 1, 1] * 20, dtype=np.float64)
    players = np.asarray([f"p{index // 5}" for index in range(100)])
    fit = fit_logistic_calibrator(probabilities, events, players)

    calibrated = apply_logistic_calibrator(
        np.asarray([0.0, 0.1, 0.9]), fit["intercept"], fit["slope"]
    )
    assert calibrated[0] == 0.0
    assert np.all(np.isfinite(calibrated))
    assert fit["slope"] > 0.0


def test_calibrator_rejects_an_event_at_exact_zero_risk() -> None:
    with pytest.raises(ConversionError):
        fit_logistic_calibrator(
            np.asarray([0.0, 0.2, 0.8]),
            np.asarray([1.0, 0.0, 1.0]),
            np.asarray(["a", "b", "c"]),
        )


def test_jensen_shannon_is_zero_only_for_identical_mass() -> None:
    assert jensen_shannon_divergence([0.5, 0.5], [0.5, 0.5]) == 0.0
    value = jensen_shannon_divergence([1.0, 0.0], [0.0, 1.0])
    assert math.isclose(value, math.log(2.0))


def test_product_thresholds_are_necessary_upper_bound_requirements() -> None:
    rows = product_scenario_thresholds(
        score_points_per_100=[0.5, 1.0, 2.0],
        mean_parent_d_opportunities_per_side_game=10.0,
    )
    assert [row["necessary_single_step_uplift"] for row in rows] == [
        0.001,
        0.002,
        0.004,
    ]
    assert all(row["perfect_redemption_upper_bound"] for row in rows)
    assert all(row["log_loss_equivalent"] is None for row in rows)


def test_plan_json_has_no_pending_identity() -> None:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    assert value["derivation_identity"] != "PENDING"
