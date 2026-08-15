from __future__ import annotations

import json
import math
from pathlib import Path

from learned_ai.evaluation.human_f0h0_feasibility import canonical_sha256


ROOT = Path(__file__).resolve().parent.parent


def _sealed(relative: str, identity_field: str) -> dict:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    identity = value.pop(identity_field)
    assert canonical_sha256(value) == identity
    value[identity_field] = identity
    return value


def test_estimator_readiness_result_is_sealed_and_binds_frozen_inputs() -> None:
    result = _sealed(
        "docs/evidence/human-feature-deviation-estimator-readiness-"
        "manifest-2026-08-15.json",
        "result_identity",
    )
    structure = _sealed(
        "docs/experiments/human-feature-deviation-estimator-crossfit-v1.json",
        "structure_identity",
    )

    assert result["result_identity"] == (
        "0df4a8bcfab8636048c8b005945a1d4bd719b23f377c06d25a6d6e5b745d0ec2"
    )
    assert result["crossfit_structure_identity"] == structure["structure_identity"]
    assert result["input_identities"]["selected_split_identity"] == (
        "8187ffa06cc73f4e052b7481f06dc3629a23feace63e086c7075c74c17940028"
    )
    assert result["expanded_exploration"]["games"] == 6400
    assert result["expanded_exploration"]["covered_decisions"] == 292192


def test_readiness_fails_only_the_frozen_log_loss_power_gate() -> None:
    result = _sealed(
        "docs/evidence/human-feature-deviation-estimator-readiness-"
        "manifest-2026-08-15.json",
        "result_identity",
    )
    analysis = result["analysis"]
    paired = analysis["paired_log_loss"]
    power = paired["power"]

    assert analysis["readiness"] == {
        "claim_or_later_gate_authority": False,
        "current_corpus_intrinsic_failures": ["paired_log_loss_player_SD"],
        "decision": "B_not_ready_fail_closed",
        "implementation_or_design_failures": [],
        "research_confirmation_opened": False,
    }
    sd_upper = paired["geometry_minus_full"]["sd_interval"][1]
    assert sd_upper > power["optimistic"]["maximum_SD_for_0_01_effect"]
    assert power["optimistic"]["passes"] is False
    assert power["kish_sensitivity"]["passes"] is False
    assert power["required_players_at_SD_upper"] == 1759
    assert math.isclose(
        power["optimistic"]["minimum_detectable_effect_at_SD_upper"],
        power["optimistic"]["coefficient"] * sd_upper,
    )


def test_d_to_l_projection_uses_the_binding_lower_bound_contract() -> None:
    result = _sealed(
        "docs/evidence/human-feature-deviation-estimator-readiness-"
        "manifest-2026-08-15.json",
        "result_identity",
    )
    endpoint = result["analysis"]["D_to_L"]

    assert endpoint["handoff_half_width_description_used"] is False
    assert endpoint["contrast"]["events"] == 10416
    assert endpoint["contrast"]["event_players"] == 769
    assert endpoint["contrast_projection"]["optimistic"]["passes"] is True
    assert endpoint["contrast_projection"]["kish_sensitivity"]["passes"] is True
    assert endpoint["Brier_projection"]["optimistic"]["passes"] is True
    assert result["D_to_L_contract_reconciliation"]["materially_different"] is True


def test_result_records_zero_protected_access_and_no_smoothing() -> None:
    result = _sealed(
        "docs/evidence/human-feature-deviation-estimator-readiness-"
        "manifest-2026-08-15.json",
        "result_identity",
    )
    audit = result["access_audit"]

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
    assert audit["denied"] == {}
    assert result["analysis"]["D_to_L"]["contrast"][
        "zero_events_not_smoothed"
    ]
