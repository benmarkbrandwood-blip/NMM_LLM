from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from learned_ai.evaluation.human_f0h0_feasibility import canonical_sha256


ROOT = Path(__file__).resolve().parent.parent
RESULT_PATH = (
    ROOT
    / "docs/evidence/human-feature-deviation-product-conversion-"
    "manifest-2026-08-15.json"
)
PLAN_V2_PATH = (
    ROOT
    / "docs/experiments/human-feature-deviation-product-conversion-"
    "derivation-v2.json"
)


def _sealed(path: Path, identity_field: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = value.pop(identity_field)
    assert canonical_sha256(value) == identity
    value[identity_field] = identity
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_conversion_result_is_sealed_and_binds_corrected_plan() -> None:
    result = _sealed(RESULT_PATH, "result_identity")
    plan = _sealed(PLAN_V2_PATH, "derivation_identity")

    assert result["result_identity"] == (
        "3da605d1d92d1a53b00dc9dabda1ac95c2e4624ec53354bddc0f8a7f53301d5f"
    )
    assert _sha256(RESULT_PATH) == (
        "885000ea45507ae8a4e64a0aa114c2425b02b5f5e5c4d35570c58e526a2d3d0a"
    )
    identities = result["conversion_plan_identities"]
    assert identities["v2_derivation_identity"] == plan["derivation_identity"]
    assert identities["v2_plan_file_sha256"] == _sha256(PLAN_V2_PATH)


def test_frozen_oof_estimator_is_reproduced_without_refitting() -> None:
    result = _sealed(RESULT_PATH, "result_identity")
    reproduction = result["analysis"]["sealed_OOF_reproduction"]

    assert reproduction["passed"] is True
    assert reproduction["refit_performed"] is False
    assert reproduction["parent_D_decisions"] == 199234
    assert reproduction["D_to_L_events"] == 10416
    assert math.isclose(
        reproduction["paired_log_loss_improvement"],
        0.1601923631872277,
        abs_tol=1e-15,
    )


def test_conversion_preserves_singletons_and_corrects_raw_uplift_downward() -> None:
    result = _sealed(RESULT_PATH, "result_identity")
    analysis = result["analysis"]
    assert analysis["sample"]["A_pos_cardinality_one_decisions"] == 33369
    assert math.isclose(
        analysis["sample"]["A_pos_cardinality_one_fraction"],
        0.11420230533347936,
    )

    for reference in (
        "uniform_A_pos",
        "geometry_A_pos",
        "human_frequency_A_pos",
    ):
        endpoint = analysis["uplift_distributions"][reference][
            "primary_D_to_L"
        ]
        assert (
            endpoint["parent_D_corrected"]["average_unique_player_mean"]
            < endpoint["parent_D_raw"]["average_unique_player_mean"]
        )
        singleton = endpoint["by_A_pos_cardinality"]["1"]
        assert singleton["raw"]["average_unique_player_mean"] == 0.0
        assert singleton["corrected"]["average_unique_player_mean"] == 0.0
        assert singleton["corrected"]["zero_fraction"] == 1.0


def test_calibration_and_policy_support_diagnostics_are_not_causal_claims() -> None:
    result = _sealed(RESULT_PATH, "result_identity")
    analysis = result["analysis"]
    observed = analysis["calibration"]["observed_OOF_parent_D"]

    assert observed["zero_risk_events"] == 0
    assert observed["zero_events_not_smoothed"] is True
    assert 0.0 < observed["calibration_slope"] < 1.0
    assert observed["expected_calibration_error"] > 0.1
    assert all(
        fold["converged"] and fold["slope"] > 0.0
        for fold in analysis["calibration"]["fold_external_calibrators"].values()
    )
    assert all(
        support["selected_low_support_mass"] == 0.0
        for support in analysis["policy_shift_support"].values()
    )


def test_log_loss_does_not_supply_an_argmax_or_product_equivalent() -> None:
    result = _sealed(RESULT_PATH, "result_identity")
    analysis = result["analysis"]
    relationship = analysis["log_loss_argmax_relationship"]
    product = analysis["product_upper_bound"]

    assert relationship["empirical_proportionality_accepted"] is False
    assert abs(
        relationship["player_level_Pearson_log_loss_vs_argmax_regret"]
    ) < 0.03
    assert abs(
        relationship["player_level_Spearman_log_loss_vs_argmax_regret"]
    ) < 0.02
    assert product["D_discrimination_equivalent_identified"] is False
    assert product["log_loss_equivalent_identified"] is False
    assert all(
        row["log_loss_equivalent"] is None
        and row["D_discrimination_equivalent"] is None
        and row["necessary_not_sufficient"]
        for row in product["scenario_thresholds"]
    )


def test_result_fails_closed_at_conversion_and_records_zero_protected_access() -> None:
    result = _sealed(RESULT_PATH, "result_identity")
    audit = result["access_audit"]
    conclusion = result["conclusion"]

    assert conclusion["decision"] == "C_conversion_not_established"
    assert conclusion["product_tiers_decidable_at_487_players"] == []
    assert conclusion["new_research_question_created"] is False
    assert conclusion["frozen_B_not_ready_decision_changed"] is False
    assert result["claim_boundary"]["positional_only"] is True
    for key in (
        "research_confirmation_content_reads",
        "official_selection_content_reads",
        "official_confirmation_content_reads",
        "official_final_test_content_reads",
        "source_pool_2eb04f54_reads_or_consumption",
        "human_db_reads",
        "database_writes",
        "games_searches_strategy_models_or_training",
    ):
        assert audit[key] == 0


def test_query_and_runtime_limits_were_respected() -> None:
    result = _sealed(RESULT_PATH, "result_identity")
    accounting = result["analysis"]["query_accounting"]

    assert accounting["total_Malom_queries"] == 44347095
    assert accounting["total_Malom_queries"] < 80000000
    assert accounting["elapsed_seconds"] < 4 * 60 * 60
