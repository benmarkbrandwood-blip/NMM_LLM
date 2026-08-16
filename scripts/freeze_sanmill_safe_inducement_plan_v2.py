#!/usr/bin/env python3
"""Freeze the v2 Sanmill safe-inducement main protocol before queries."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.human_f0h0_b2_freeze import load_membership
from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    load_f0d0_boundary,
    write_sealed_json,
)
from learned_ai.evaluation.human_feature_deviation_design_round import load_split_v2
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    load_crossfit_structure,
)
from learned_ai.evaluation.sanmill_safe_inducement import (
    MAIN_PLAN_SCHEMA,
    MAIN_POOL_SELECTION_SEED,
    PHASES,
    count_source_phase_frequencies,
    load_plan,
    load_state_pool,
)


def _sealed_result(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    identity = value.pop("result_identity", None)
    if value.get("schema_version") != "nmm.sanmill-safe-inducement-preprobe-result.v1":
        raise RuntimeError("preprobe result schema differs")
    if identity != canonical_sha256(value):
        raise RuntimeError("preprobe result identity differs")
    value["result_identity"] = identity
    return value, hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v1-plan",
        default="docs/experiments/sanmill-safe-inducement-mechanism-v1.json",
    )
    parser.add_argument(
        "--v1-pool",
        default=(
            "docs/experiments/sanmill-safe-inducement-preprobe-"
            "state-pool-v1.json"
        ),
    )
    parser.add_argument(
        "--preprobe-result",
        default=(
            "docs/evidence/sanmill-safe-inducement-preprobe-"
            "manifest-2026-08-16.json"
        ),
    )
    parser.add_argument(
        "--f0d0-manifest",
        default=(
            "docs/evidence/f0-d0-human-raw-reconstructability-"
            "manifest-2026-08-14.json"
        ),
    )
    parser.add_argument(
        "--official-membership",
        default="docs/experiments/f0-h0-design-b2-frozen-membership-v1.json",
    )
    parser.add_argument(
        "--research-split",
        default="docs/experiments/human-feature-deviation-train-split-v3.json",
    )
    parser.add_argument(
        "--crossfit-structure",
        default=(
            "docs/experiments/human-feature-deviation-estimator-crossfit-v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        default="docs/experiments/sanmill-safe-inducement-mechanism-v2.json",
    )
    args = parser.parse_args()

    v1, v1_file_sha = load_plan(_ROOT / args.v1_plan)
    v1_pool, v1_pool_file_sha = load_state_pool(_ROOT / args.v1_pool)
    result, result_file_sha = _sealed_result(_ROOT / args.preprobe_result)
    if (
        result["plan_identity"] != v1["plan_identity"]
        or result["state_pool_identity"] != v1_pool["pool_identity"]
        or result["analysis"]["decision"]["recommended_main_node_budget"]
        != 100_000
    ):
        parser.error("preprobe plan, pool, and result do not bind one lineage")

    boundary = load_f0d0_boundary(_ROOT / args.f0d0_manifest)
    membership, membership_file_sha = load_membership(
        _ROOT / args.official_membership
    )
    research_split, research_split_file_sha = load_split_v2(
        _ROOT / args.research_split
    )
    crossfit, crossfit_file_sha = load_crossfit_structure(
        _ROOT / args.crossfit_structure
    )
    frequencies = count_source_phase_frequencies(
        repository_root=_ROOT,
        boundary=boundary,
        official_membership=membership,
        research_split=research_split,
        crossfit_structure=crossfit,
    )
    excluded = sorted(
        [str(row["session_id"]), int(row["logical_ply"])]
        for row in v1_pool["states"]
    )

    v1_main = v1["main_experiment_unlaunched"]
    payload = {
        "schema_version": MAIN_PLAN_SCHEMA,
        "status": "frozen_protocol_v2_execution_unlaunched",
        "experiment_id": "sanmill-safe-inducement-mechanism-v2",
        "version_disposition": {
            "v1_preserved_unchanged": True,
            "v2_changes_only": [
                "multi_budget_invariant_vs_sensitive_decomposition",
                "source_phase_frequency_weighted_secondary_metric",
            ],
            "all_v1_main_thresholds_unchanged": True,
        },
        "research_question": v1["research_question"],
        "claim_boundary": v1["claim_boundary"],
        "input_identities": {
            "v1_plan_identity": v1["plan_identity"],
            "v1_plan_file_sha256": v1_file_sha,
            "preprobe_state_pool_identity": v1_pool["pool_identity"],
            "preprobe_state_pool_membership_identity": v1_pool[
                "state_membership_identity"
            ],
            "preprobe_state_pool_file_sha256": v1_pool_file_sha,
            "preprobe_result_identity": result["result_identity"],
            "preprobe_result_file_sha256": result_file_sha,
            "source_crossfit_structure_identity": crossfit["structure_identity"],
            "source_crossfit_structure_file_sha256": crossfit_file_sha,
            "f0d0_corpus_identity": boundary.manifest["identities"][
                "corpus_identity"
            ],
            "official_membership_identity": membership["membership_identity"],
            "official_membership_file_sha256": membership_file_sha,
            "research_split_identity": research_split["split_identity"],
            "research_split_file_sha256": research_split_file_sha,
            "malom_content_sha256": v1["input_identities"][
                "malom_content_sha256"
            ],
            "malom_trust_level": "sector-corrected-v1",
        },
        "sanmill_contract": v1["sanmill_contract"],
        "state_pool_contract": {
            "states": 360,
            "states_per_phase": 120,
            "phase_order": list(PHASES),
            "source_game_reuse": False,
            "source_population": (
                "frozen_6400_game_research_exploration_crossfit_sample"
            ),
            "selection_seed": MAIN_POOL_SELECTION_SEED,
            "rank": "SHA-256(seed NUL session_id NUL logical_ply NUL phase)",
            "excluded_preprobe_coordinates": len(excluded),
            "excluded_preprobe_coordinates_identity": canonical_sha256(excluded),
            "result_variables_or_estimator_predictions_allowed": False,
            "sanmill_observations_allowed": False,
            "replacement_after_malom_or_engine_observation": False,
            "official_or_research_confirmation_sources_allowed": False,
            "source_pool_2eb04f54_allowed": False,
        },
        "determinism_gate": {
            "must_pass_before_measurement": True,
            "fixture_selection": (
                "two lowest logical-ply states per phase having a nonterminal "
                "canonical first A_pos action"
            ),
            "fixtures_per_phase": 2,
            "budgets": [1_000, 100_000, 500_000],
            "same_process_repetitions": 2,
            "fresh_process_orders": ["forward", "reverse"],
            "comparison": "exact equality of UciLogicalTurnResult.semantic_record",
            "timing_and_raw_protocol_text_excluded": True,
            "failure_action": "execution_incomplete; no statistical measurement",
        },
        "estimands": {
            key: value
            for key, value in v1["estimands"].items()
            if key
            in {
                "state_weighting",
                "within_state_weighting",
                "b_i",
                "o_i",
                "gain_i",
                "b",
                "o",
                "o_minus_b",
                "strict_terminal_successor",
                "engine_search_or_oracle_failure",
                "downgrade_transitions",
                "within_tier_regret_is_primary",
                "independent_unit",
                "additional_clustering",
            }
        },
        "main_experiment": {
            "execution_count": 1,
            "node_budgets": [1_000, 100_000, 500_000],
            "primary_node_budget": 100_000,
            "primary_estimand": "state-uniform o_minus_b",
            "measurement_order_seed": (
                "sanmill-safe-inducement-main-cell-order-v2-20260816"
            ),
            "measurement": "exhaust complete A_pos(S) once at every budget",
            "strata": ["phase", "engine_reply_state_WDL_tier"],
            "downgrade_transitions": ["W->D", "W->L", "D->L"],
            "interval": v1_main["interval"],
            "mechanism_success_gate": v1_main["mechanism_success_gate"],
            "abstention": v1_main["abstention"],
            "budget_decomposition": {
                "unit": "one frozen state and one complete A_pos action",
                "budget_invariant": (
                    "the action induces a positional tier downgrade at all "
                    "three tested budgets"
                ),
                "budget_sensitive": (
                    "the action induces a positional tier downgrade at a "
                    "nonempty strict subset of tested budgets"
                ),
                "o_inv": (
                    "share of states having at least one budget-invariant "
                    "inducing action"
                ),
                "o_sens": (
                    "share of states having no invariant action but at least "
                    "one budget-sensitive inducing action"
                ),
                "identity": "o_union = o_inv + o_sens",
                "fixed_blind_spot_interpretation_threshold": 0.80,
                "interpretation_rule": (
                    "if at least 80% of union-induced states are invariant, "
                    "describe the effect as a fixed engine evaluation blind "
                    "spot, not complexity-manufacturing trap ability"
                ),
                "affects_primary_gate": False,
            },
            "frequency_weighted_secondary": {
                "phase_counts": frequencies["counts"],
                "total_decisions": frequencies["total"],
                "weights": frequencies["weights"],
                "count_identity": frequencies["count_identity"],
                "source": (
                    "independent strict replay of the frozen 6400-game "
                    "research-exploration crossfit population"
                ),
                "node_budget": 100_000,
                "estimand": (
                    "sum of source phase-frequency weight times phase-specific "
                    "state-uniform o_minus_b"
                ),
                "threshold": None,
                "can_flip_primary_decision": False,
                "population_boundary": "observed PlayOK-like source domain only",
            },
            "resource_envelope": v1_main["resource_envelope"],
            "recovery": {
                "automatic_retry": False,
                "automatic_resume": False,
                "host_interruption_missing_suffix_requires_separate_authorization": True,
                "batching_to_evade_limits": False,
            },
            "forbidden_claims": v1_main["forbidden_claims"],
        },
        "authorization_state": {
            "direct_statement_received": True,
            "operator": "product-owner-direct",
            "canonical_authorization_record_pending_child_pool_identity": True,
            "measurement_started": False,
        },
        "phase_frequency_reproduction": frequencies,
        "protected_access": v1["protected_access"],
    }
    sealed = write_sealed_json(
        _ROOT / args.output,
        payload,
        identity_field="plan_identity",
    )
    print(sealed["plan_identity"])
    print(frequencies["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
