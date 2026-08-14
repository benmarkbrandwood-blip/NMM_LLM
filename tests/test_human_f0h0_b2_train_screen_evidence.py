from pathlib import Path

import pytest

from learned_ai.evaluation.human_f0h0_b2_train_screen import (
    load_screen_plan,
    load_screen_result,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_V1 = ROOT / "docs/experiments/f0-h0-b2-train-rejection-screen-v1.json"
PLAN_V2 = ROOT / "docs/experiments/f0-h0-b2-train-rejection-screen-v2.json"
RESULT_V1 = ROOT / (
    "docs/evidence/f0-h0-b2-train-rejection-screen-manifest-2026-08-15.json"
)
RESULT_V2 = ROOT / (
    "docs/evidence/f0-h0-b2-train-rejection-screen-v2-manifest-2026-08-15.json"
)


def test_corrected_result_is_sealed_and_keeps_the_v1_contract() -> None:
    plan_v1, _plan_v1_sha = load_screen_plan(PLAN_V1)
    plan_v2, plan_v2_sha = load_screen_plan(PLAN_V2)
    result_v2, result_v2_sha = load_screen_result(RESULT_V2)

    assert plan_v2["thresholds"] == plan_v1["thresholds"]
    assert (
        plan_v2["sample"]["train_session_ids_identity"]
        == plan_v1["sample"]["train_session_ids_identity"]
    )
    assert result_v2["lineage"]["plan_identity"] == plan_v2["plan_identity"]
    assert result_v2["lineage"]["plan_file_sha256"] == plan_v2_sha
    assert result_v2["result_identity"] == (
        "8bd2da62785e9c8cda0a055e98213959cbdf8f88aa860384171f00f4f39c6bdc"
    )
    assert result_v2_sha == (
        "9f2d1d8e85a358ffd5db4c13cef46e7fb9d41e3c73d89e3d2a4c3bc227cbe809"
    )


def test_historical_result_is_preserved_but_not_the_decision_source() -> None:
    result_v1, result_v1_sha = load_screen_result(RESULT_V1)
    plan_v2, _plan_v2_sha = load_screen_plan(PLAN_V2)

    assert result_v1["result_identity"] == (
        "84434066d5c4fc58fd82585a32c5bff9ef69ecfe4badb2a0c3289b4b0fb7068b"
    )
    assert result_v1_sha == (
        "e03e6e63a788e53f76479c48995d3c02657231c74a78d6396a35cb57ce403297"
    )
    assert (
        plan_v2["technical_correction"]["supersedes_result_identity"]
        == (result_v1["result_identity"])
    )


def test_corrected_screen_stops_on_support_and_concentration() -> None:
    result, _result_sha = load_screen_result(RESULT_V2)
    support = result["dimensions"]["independent_support"]
    concentration = result["dimensions"]["concentration"]
    gates = result["gate_results"]

    assert result["decision"] == "stop_condition_triggered"
    assert support["ring16_positional_state_class"][
        "supported_decision_fraction"
    ] == pytest.approx(0.2016283179247677)
    assert (
        gates["independent_support"]["minimum_ring16_supported_decision_fraction"]
        is False
    )
    assert concentration["players"]["gini"] == pytest.approx(0.8035982455508652)
    assert concentration["players"]["kish_effective_units"] == pytest.approx(
        177.74444635313031
    )
    assert gates["concentration"]["maximum_player_gini"] is False
    assert gates["concentration"]["minimum_player_kish_effective_units"] is False
    failed = {
        f"{dimension}.{name}"
        for dimension, rows in gates.items()
        for name, passed in rows.items()
        if not passed
    }
    assert failed == {
        "independent_support.minimum_ring16_supported_decision_fraction",
        "concentration.maximum_player_gini",
        "concentration.maximum_player_top_5_percent_share",
        "concentration.maximum_player_top_10_percent_share",
        "concentration.minimum_player_kish_effective_units",
        "concentration.maximum_supported_modifiable_player_top_5_share",
        "concentration.maximum_ring16_state_top_1_percent_share",
    }


def test_modifiable_reach_and_estimability_are_not_the_stop_reasons() -> None:
    result, _result_sha = load_screen_result(RESULT_V2)
    reach = result["dimensions"]["modifiable_state_reachability"]
    product = result["dimensions"]["product_effect_upper_bound"]

    assert reach["a_pos_cardinality_greater_than_one"]["point"] == pytest.approx(
        0.8868512279901193
    )
    assert all(result["gate_results"]["modifiable_reachability"].values())
    assert product["four_a_estimability"]["decision"] == (
        "state_level_empirically_estimable"
    )
    assert (
        product["four_a_estimability"]["ring16_positional_state_class"][
            "classes_with_two_observed_safe_actions_each_at_least_m"
        ]
        == 394
    )
    assert all(result["gate_results"]["estimability"].values())


def test_zero_event_transitions_do_not_receive_prior_generated_lift() -> None:
    result, _result_sha = load_screen_result(RESULT_V2)
    effects = result["dimensions"]["product_effect_upper_bound"][
        "four_b_state_conditioned_effect"
    ]["uncorrected_and_corrected_by_transition"]

    for event in ("W->D", "W->L"):
        row = effects[event]
        assert row["status"] == "no_observed_transition_events"
        assert row["observed_transition_events"] == 0
        assert row["corrected_point"] == 0.0
        assert row["conservative_upper_95"] < 0.01
    assert effects["D->L"]["status"] == "estimated"
    assert effects["D->L"]["corrected_point"] == pytest.approx(0.05633437381748742)


def test_access_and_oracle_boundaries_remained_closed() -> None:
    result, _result_sha = load_screen_result(RESULT_V2)

    assert result["sample"]["membership_composition"] == {
        "train": 9_113,
        "selection": 887,
        "confirmation": 0,
        "final-test": 0,
    }
    assert result["sample"]["analysis_games"] == 9_113
    assert result["sample"]["analysis_decisions"] == 429_523
    assert result["bases"]["oracle_positional_sample"]["coverage_fraction"] == 1.0
    assert result["dimensions"]["product_effect_upper_bound"]["product_scope"][
        "recorded_and_independently_replayed_outcomes"
    ] == {"W": 3_214, "B": 3_786, "D": 8_135}
    assert (
        result["access_audit"]["selection_raw_games_or_decisions_or_features_read"] == 0
    )
    assert (
        result["access_audit"]["confirmation_raw_games_or_decisions_or_features_read"]
        == 0
    )
    assert (
        result["access_audit"]["final_test_raw_games_or_decisions_or_features_read"]
        == 0
    )
    assert all(
        value == 0 for value in result["prohibited_operations_observed"].values()
    )
