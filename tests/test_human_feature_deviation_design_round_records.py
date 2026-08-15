from __future__ import annotations

import json
from pathlib import Path

from learned_ai.evaluation.human_f0h0_feasibility import canonical_sha256
from learned_ai.evaluation.human_feature_deviation_design_round import (
    V2_FEATURE_NAMES,
    load_design_plan,
    load_split_v2,
)


ROOT = Path(__file__).resolve().parent.parent


def _sealed(relative: str, identity_field: str) -> dict:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    identity = value[identity_field]
    body = dict(value)
    body.pop(identity_field)
    assert canonical_sha256(body) == identity
    return value


def test_rebalance_v2_selects_the_frozen_player_isolated_split() -> None:
    plan, _ = load_design_plan(
        ROOT / "docs/experiments/human-feature-deviation-design-round-v2.json"
    )
    split, _ = load_split_v2(
        ROOT / "docs/experiments/human-feature-deviation-train-split-v3.json"
    )
    result = _sealed(
        "docs/evidence/human-feature-deviation-precision-rebalance-v2-"
        "manifest-2026-08-15.json",
        "result_identity",
    )

    assert split["design_round_plan_identity"] == plan["plan_identity"]
    assert split["split_identity"] == result["selected_split"]["split_identity"]
    assert result["selected_candidate"] == "community-cut-confirmation-50"
    assert split["player_membership"]["pairwise_player_overlap"] == 0
    assert split["partitions"]["research-confirmation"]["games"] == 2543
    assert result["access_audit"]["human_outcome_or_action_variables_read"] == 0
    assert result["access_audit"]["research_confirmation_content_reads"] == 0


def test_extension_is_exploration_only_and_preserves_zero_events() -> None:
    result = _sealed(
        "docs/evidence/human-feature-deviation-exploration-extension-"
        "manifest-2026-08-15.json",
        "result_identity",
    )

    assert result["sample"]["games"] == 1024
    assert result["sample"]["covered_decisions"] == 48855
    assert result["sample"]["abstained_decisions"] == 0
    assert result["oracle"]["queries"] == 632094
    assert result["structural_diagnostics"]["simultaneous_double_mill_choice_sets"] == 0
    affine = result["structural_diagnostics"]["closes_mill_material_balance_after"]
    assert affine["choice_sets"] == affine["exact_affine_choice_sets"] == 48855
    assert result["positional_labels"]["chosen_tier_loss_counts"] == {
        "D->L": 1642,
        "W->D": 919,
        "W->L": 83,
    }
    for name in (
        "research_confirmation_content_reads",
        "selection_content_reads",
        "confirmation_content_reads",
        "final_test_content_reads",
        "source_pool_2eb04f54_records_read_or_consumed",
    ):
        assert result["access_audit"][name] == 0


def test_v2_preregistration_is_frozen_but_confirmation_blocked() -> None:
    plan = _sealed(
        "docs/experiments/human-feature-deviation-screen-v2.json",
        "plan_identity",
    )
    recovery = _sealed(
        "docs/experiments/human-feature-deviation-exploration-extension-"
        "recovery-v1.json",
        "recovery_identity",
    )

    assert tuple(plan["feature_dictionary"]["ordered_features"]) == V2_FEATURE_NAMES
    assert plan["precision_execution_gate"]["status"].startswith("not_met")
    assert plan["confirmation_execution_authorized"] is False
    assert recovery["failed_attempt"]["malom_queries"] == 0
    assert recovery["retry_contract"]["partial_prefix_reuse"] is False
