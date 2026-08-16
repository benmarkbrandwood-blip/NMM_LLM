#!/usr/bin/env python3
"""Freeze the sole primary Sanmill-human transfer estimator."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.human_f0h0_feasibility import write_sealed_json
from learned_ai.evaluation.sanmill_human_transfer import (
    AUDIT_SCHEMA,
    PLAN_SCHEMA,
    load_sealed,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        default=(
            "docs/evidence/sanmill-human-transfer-coverage-audit-"
            "2026-08-16.json"
        ),
    )
    parser.add_argument(
        "--output",
        default="docs/experiments/sanmill-human-transfer-v1.json",
    )
    args = parser.parse_args()
    audit, audit_sha = load_sealed(
        _ROOT / args.audit,
        identity_field="audit_identity",
        schema=AUDIT_SCHEMA,
    )
    coverage = audit["analysis"]["coverage"]
    if coverage["available_states"] != 360 or coverage["fraction"] != 1.0:
        parser.error("full structural OOF coverage was not established")
    payload = {
        "schema_version": PLAN_SCHEMA,
        "status": "frozen_before_transfer_outcome_calculation",
        "experiment_id": "sanmill-human-transfer-v1",
        "research_question": (
            "Does the frozen OOF human tier-loss risk ranking select the same "
            "positional-safe actions that induce a one-step downgrade from the "
            "exact pinned Sanmill runtime at 100,000 nodes?"
        ),
        "post_hoc_control": {
            "data_availability_audited_before_freeze": True,
            "audit_identity": audit["audit_identity"],
            "audit_file_sha256": audit_sha,
            "outcomes_or_predictions_calculated_during_audit": False,
            "one_primary_estimator": True,
            "thresholds_frozen_before_any_transfer_calculation": True,
        },
        "input_identities": audit["input_identities"],
        "coverage_gate": {
            "reference_states": 360,
            "minimum_states": 330,
            "minimum_fraction": 0.90,
            "observed_before_freeze": 360,
            "full_coverage_established": True,
            "partial_subset_reweighting_or_imputation": False,
        },
        "primary_estimator": {
            "specification": "full_10_feature_conditional_choice_model",
            "prediction": (
                "raw OOF probability that the opponent selects any response "
                "with lower positional WDL tier than the response-state tier"
            ),
            "tier_scope": {
                "W": ["W->D", "W->L"],
                "D": ["D->L"],
                "L": [],
                "rationale": (
                    "match the main engine endpoint across all frozen W and D "
                    "response states; the earlier D-only endpoint remains a "
                    "different frozen research question"
                ),
            },
            "fold": (
                "use the source state's acting-player held-out fold; both source "
                "players are in that same fold by the frozen sample contract"
            ),
            "estimator_refit": False,
            "model_parameter_source": (
                "sealed readiness fold means, scales, and coefficients"
            ),
            "risk_reconstruction": (
                "reuse the exact product-conversion feature, numerical, and "
                "tier-loss functions on each persisted successor FEN"
            ),
            "action_order": "lexicographic (from, to, capture), empty before text",
            "argmax_tie_rule": (
                "choose the first action in frozen canonical action order among "
                "exactly equal maximum risks"
            ),
            "e_a": (
                "one iff the sealed main result records W->D, W->L, or D->L "
                "for the same action at exactly 100,000 nodes"
            ),
            "A": "state-uniform mean of e(argmax_a p_hat(a))",
            "b": "state-uniform mean of within-state uniform e(a)",
            "o": "state-uniform mean of one iff any A_pos action has e(a)=1",
            "transfer": "(A - b) / (o - b)",
            "matched_support": "all 360 structurally covered frozen states",
        },
        "geometry_control": {
            "specification": "nested_3_feature_geometry_model",
            "repeat_primary_quantities": True,
            "secondary_only": True,
            "can_flip_primary_decision": False,
            "interpretation": (
                "if full does not exceed geometry, observed alignment is not "
                "evidence for the human-specific tactical feature increment"
            ),
        },
        "interval": {
            "method": "state-level nonparametric percentile bootstrap",
            "repetitions": 20_000,
            "seed": "sanmill-human-transfer-bootstrap-v1-20260816",
            "resampled_unit": "one source-game-unique frozen state",
            "recompute_A_b_o_and_ratio_within_each_resample": True,
            "zero_event_rule": "no prior and no pseudo-event",
        },
        "decision_rule": {
            "existence": "lower 95% bound of A_minus_b is strictly above zero",
            "substantive_transfer_threshold": 0.25,
            "substantive_rule": (
                "existence passes and the lower 95% bound of transfer is at "
                "least 0.25"
            ),
            "threshold_rationale": (
                "recovering less than one quarter of the measured oracle action-"
                "selection headroom is too small to support shared-structure "
                "language; using the lower bound prevents a noisy point estimate "
                "from qualifying"
            ),
            "A": "A_substantive_transfer_exists",
            "B": "B_transfer_exists_but_not_substantive",
            "C": "C_no_transfer",
            "D": "D_insufficient_coverage_or_required_input",
            "geometry_cannot_flip": True,
        },
        "secondary_analyses": {
            "phase": ["placement", "movement", "flying"],
            "budget_type": ["budget_invariant", "budget_sensitive"],
            "A_pos_cardinality": ["1", "2", "3-4", "5-8", "9-plus"],
            "discrimination": (
                "within-state AUC among states containing both inducing and "
                "non-inducing actions, followed by state bootstrap"
            ),
            "full_minus_geometry": "paired selected-downgrade difference",
            "cannot_flip_primary": True,
        },
        "resource_envelope": {
            "maximum_malom_queries": 500_000,
            "maximum_active_seconds": 3_600,
            "maximum_sanmill_queries": 0,
            "maximum_complete_games": 0,
            "maximum_policy_model_loads": 0,
            "maximum_estimator_refits": 0,
            "maximum_training_updates": 0,
            "maximum_database_writes": 0,
            "stop_at_any_limit": True,
            "automatic_retry_resume_or_extension": False,
        },
        "claim_boundary": {
            "positional_only": True,
            "safe_set": "A_pos",
            "A_allow_claim": False,
            "source_domain": "observed PlayOK-like source only",
            "engine_runtime": "exact pinned Sanmill runtime at 100,000 nodes",
            "single_step_only": True,
            "human_trap_ability": False,
            "causal_inducement": False,
            "multi_step_redemption_or_product_value": False,
            "existing_F0_H0_stop_remains_effective": True,
            "existing_estimator_B_not_ready_remains_effective": True,
            "existing_conversion_C_not_established_remains_effective": True,
            "existing_mechanism_gate_passed_remains_effective": True,
        },
        "protected_access": {
            "official_selection": "unopened",
            "official_confirmation": "unopened",
            "official_final_test": "unopened",
            "research_confirmation": "unopened",
            "source_pool_2eb04f54_remaining_108": "unread_and_unconsumed",
        },
    }
    sealed = write_sealed_json(
        _ROOT / args.output, payload, identity_field="plan_identity"
    )
    print(sealed["plan_identity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
