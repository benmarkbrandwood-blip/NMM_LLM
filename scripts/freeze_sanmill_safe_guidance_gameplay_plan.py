#!/usr/bin/env python3
"""Freeze the pre-game protocol for the safe-guidance gameplay experiment."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    write_sealed_json,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    ARMS,
    EXPECTED_GAMES,
    EXPECTED_STARTS,
    MAX_POST_START_LOGICAL_PLIES,
    PHASES,
    PLAN_SCHEMA,
    PRIMARY_NODE_BUDGET,
    STARTS_PER_PHASE,
    load_sealed,
    sha256_file,
)
from learned_ai.evaluation.sanmill_safe_inducement import (
    MAIN_POOL_SCHEMA,
    POOL_SCHEMA as PREPROBE_POOL_SCHEMA,
)


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-v2-plan",
        default="docs/experiments/sanmill-safe-inducement-mechanism-v2.json",
    )
    parser.add_argument(
        "--main-pool",
        default="docs/experiments/sanmill-safe-inducement-main-state-pool-v2.json",
    )
    parser.add_argument(
        "--preprobe-pool",
        default=(
            "docs/experiments/sanmill-safe-inducement-preprobe-state-pool-v1.json"
        ),
    )
    parser.add_argument(
        "--transfer-plan",
        default="docs/experiments/sanmill-human-transfer-v1.json",
    )
    parser.add_argument(
        "--transfer-result",
        default="docs/evidence/sanmill-human-transfer-manifest-2026-08-16.json",
    )
    parser.add_argument(
        "--readiness-result",
        default=(
            "docs/evidence/human-feature-deviation-estimator-readiness-"
            "manifest-2026-08-15.json"
        ),
    )
    parser.add_argument(
        "--heldout-report",
        default=(
            "learned_ai/checkpoints/evaluation/"
            "sanmill-retained-v3-v4-heldout-score-v1/report.json"
        ),
    )
    parser.add_argument(
        "--output",
        default="docs/experiments/sanmill-safe-guidance-gameplay-v1.json",
    )
    args = parser.parse_args()

    output = _ROOT / args.output
    if output.exists():
        parser.error("gameplay protocol already exists")
    main_plan = json.loads((_ROOT / args.main_v2_plan).read_text(encoding="utf-8"))
    if main_plan.get("plan_identity") != (
        "22f79951080afe01f11e0d9f2bbd16e8421fd92f2e96841da3627405271b89bf"
    ):
        parser.error("main v2 protocol identity differs")
    main_pool, main_pool_sha = load_sealed(
        _ROOT / args.main_pool,
        schema=MAIN_POOL_SCHEMA,
        identity_field="pool_identity",
    )
    preprobe_pool, preprobe_pool_sha = load_sealed(
        _ROOT / args.preprobe_pool,
        schema=PREPROBE_POOL_SCHEMA,
        identity_field="pool_identity",
    )
    excluded = {
        (str(row["session_id"]), int(row["logical_ply"]))
        for pool in (main_pool, preprobe_pool)
        for row in pool["states"]
    }
    if len(excluded) != 396:
        parser.error("prior-state exclusion union differs")
    heldout = json.loads((_ROOT / args.heldout_report).read_text(encoding="utf-8"))
    observed_sd = float(
        heldout["paired"]["primary_start_clustered_score_v4_minus_v3"][
            "sample_standard_deviation"
        ]
    )
    if not math.isclose(observed_sd, 0.08605338688463011, abs_tol=1e-15):
        parser.error("independent score-SD evidence differs")
    planning_sd = 0.12
    expected_half = 1.96 * planning_sd / math.sqrt(EXPECTED_STARTS)
    mde_80 = (1.96 + 0.8416212335729143) * planning_sd / math.sqrt(
        EXPECTED_STARTS
    )
    mde_90 = (1.96 + 1.2815515655446004) * planning_sd / math.sqrt(
        EXPECTED_STARTS
    )
    transfer = json.loads((_ROOT / args.transfer_result).read_text(encoding="utf-8"))
    if transfer.get("result_identity") != (
        "c6dce5690a138361238ddd4661cce78251e67fb0f9d8003a0473a0b12c1a2700"
    ):
        parser.error("transfer result identity differs")

    payload = {
        "schema_version": PLAN_SCHEMA,
        "experiment_id": "sanmill-safe-guidance-gameplay-v1-20260816",
        "status": "frozen_before_start_pool_or_gameplay",
        "frozen_at_utc": "2026-08-16T00:00:00Z",
        "repository_base_commit": _git("rev-parse", "HEAD"),
        "question": (
            "Under one exact fixed Sanmill runtime, does the frozen full-feature "
            "A_pos guidance rule improve strict complete-game score over a "
            "matched uniform-random A_pos rule?"
        ),
        "experiment": {
            "arms": list(ARMS),
            "primary_comparison": "full-guided minus random-safe",
            "geometry_role": (
                "secondary nested control; retained because it directly tests "
                "whether the prior full-over-geometry one-step increment survives "
                "complete-game conversion; it cannot change the primary decision"
            ),
            "phases": list(PHASES),
            "starts_per_phase": STARTS_PER_PHASE,
            "starts": EXPECTED_STARTS,
            "colors_per_start": 2,
            "games_per_start": 6,
            "games": EXPECTED_GAMES,
            "primary_node_budget": PRIMARY_NODE_BUDGET,
            "budget_decomposition": [1_000, 100_000, 500_000],
            "max_post_start_logical_plies": MAX_POST_START_LOGICAL_PLIES,
            "safety_cap_disposition": "incomplete; never a draw",
            "random_safe_seed": (
                "sanmill-safe-guidance-random-a-pos-v1-20260816"
            ),
            "schedule": (
                "state identity order, then candidate W/B, then random-safe, "
                "full-guided, geometry-guided"
            ),
        },
        "policy_contract": {
            "safe_set": "complete corrected Malom A_pos at every candidate turn",
            "random_safe": (
                "uniform over canonical (from,to,capture)-ordered A_pos using the "
                "frozen per-game random stream"
            ),
            "full_guided": (
                "canonical argmax of the frozen ten-feature successor response "
                "downgrade risk from the source start's persisted OOF fold"
            ),
            "geometry_guided": (
                "same as full-guided using the frozen nested three-feature geometry "
                "coefficients"
            ),
            "ties": "exact risk ties select canonical first action",
            "estimator_refit_or_tuning": False,
            "fold_binding": (
                "the source session's existing same-fold cross-fit assignment is "
                "held fixed for the complete generated trajectory"
            ),
        },
        "primary_estimand": {
            "independent_unit": "frozen start",
            "within_start": (
                "for each arm, average strict W/D/L score over candidate W and B"
            ),
            "contrast": "full-guided minus random-safe",
            "interval": "mean plus/minus 1.96 start-level sample-SE",
            "population_inference": False,
        },
        "precision_preregistration": {
            "independent_prior_evidence": {
                "source": args.heldout_report,
                "file_sha256": sha256_file(_ROOT / args.heldout_report),
                "completed_starts": 253,
                "observed_start_level_score_difference_sd": observed_sd,
            },
            "conservative_planning_sd": planning_sd,
            "expected_95_half_width": expected_half,
            "maximum_95_half_width": 0.015,
            "approximate_two_sided_mde_80_power": mde_80,
            "approximate_two_sided_mde_90_power": mde_90,
            "sample_size_frozen_before_new_outcomes": True,
        },
        "primary_decision": {
            "all_1530_games_must_be_strict_rules_terminal": True,
            "maximum_half_width": 0.015,
            "full_higher": "interval lower bound > 0",
            "random_higher": "interval upper bound < 0",
            "otherwise": "inconclusive",
            "equivalence_claim": False,
            "no_result_based_early_stop_or_extension": True,
        },
        "secondary_metrics": {
            "cannot_flip_primary": True,
            "items": [
                "full-guided minus geometry-guided start-clustered score",
                "W/D/L and strict terminal reasons by arm and source phase",
                "W-to-D, W-to-L, and D-to-L positional downgrade events",
                "win rate among games with at least one induced downgrade",
                "rule draws with and without any induced downgrade",
                "budget-invariant versus budget-sensitive induced events",
                "terminal side-to-move positional WDL without referee override",
            ],
            "budget_classification": (
                "only an observed 100k induced event is decomposed; fresh exact "
                "roots are searched at 1k and 500k after its game, invariant means "
                "downgrade at all three budgets, otherwise sensitive"
            ),
        },
        "start_pool_contract": {
            "states_per_phase": STARTS_PER_PHASE,
            "states": EXPECTED_STARTS,
            "source": "frozen 6400-game research-exploration same-fold sample",
            "source_game_reuse": False,
            "selection_seed": (
                "sanmill-safe-guidance-gameplay-start-v1-20260816"
            ),
            "rank": "SHA-256(seed NUL session_id NUL logical_ply NUL phase)",
            "blind_to": [
                "Sanmill observations",
                "human estimator predictions",
                "human chosen action and outcome",
            ],
            "exclude_prior_coordinates": 396,
            "exclude_prior_coordinates_identity": canonical_sha256(
                sorted([session_id, ply] for session_id, ply in excluded)
            ),
            "main_v2_pool_identity": main_pool["pool_identity"],
            "main_v2_pool_file_sha256": main_pool_sha,
            "preprobe_pool_identity": preprobe_pool["pool_identity"],
            "preprobe_pool_file_sha256": preprobe_pool_sha,
            "migration_states_same_as_main_v2": True,
            "replacement_after_malom_or_engine_observation": False,
            "complete_logical_history_required": True,
        },
        "sanmill_contract": {
            **main_plan["sanmill_contract"],
            "fresh_process_per_measurement_cell": False,
            "game_process": "one fresh strict referee process per complete game",
            "budget_decomposition_process": (
                "one fresh process per event-budget cell after the game"
            ),
        },
        "resource_envelope": {
            "maximum_independent_starts": 256,
            "maximum_complete_games": 1536,
            "maximum_engine_single_step_searches": 80_000,
            "maximum_malom_queries": 20_000_000,
            "maximum_active_seconds": 21_600,
            "maximum_concurrent_evaluators": 1,
            "maximum_concurrent_sanmill_processes": 1,
            "maximum_training_updates": 0,
            "maximum_database_writes": 0,
            "stop_at_any_limit": True,
            "automatic_retry_resume_batching_or_extension": False,
            "host_interruption_recovery_authorized": False,
        },
        "preflight_contract": {
            "determinism_budgets": [1_000, 100_000, 500_000],
            "fixtures_per_phase": 2,
            "opposite_order_and_same_process_repeat": True,
            "guide_canary_states": 6,
            "guide_canary_source": (
                "frozen transfer action rows, exact full and geometry selections"
            ),
            "zero_complete_games": True,
        },
        "input_identities": {
            "main_v2_plan_identity": main_plan["plan_identity"],
            "main_v2_plan_file_sha256": sha256_file(_ROOT / args.main_v2_plan),
            "transfer_plan_file_sha256": sha256_file(_ROOT / args.transfer_plan),
            "transfer_result_identity": transfer["result_identity"],
            "transfer_result_file_sha256": sha256_file(_ROOT / args.transfer_result),
            "readiness_result_identity": (
                "0df4a8bcfab8636048c8b005945a1d4bd719b23f377c06d25a6d6e5b745d0ec2"
            ),
            "readiness_result_file_sha256": sha256_file(
                _ROOT / args.readiness_result
            ),
            "malom_content_sha256": main_plan["input_identities"][
                "malom_content_sha256"
            ],
            "malom_trust_level": "sector-corrected-v1",
        },
        "authorization_basis": {
            "type": "product-owner-direct",
            "date": "2026-08-16",
            "one_time": True,
            "exact_envelope_reproduced": True,
        },
        "claim_boundary": {
            "safe_set": "A_pos",
            "positional_only": True,
            "A_allow_claim": False,
            "strict_referee_is_only_terminal_authority": True,
            "fixed_runtime_and_start_pool_only": True,
            "human_trap_claim": False,
            "playing_strength_claim": False,
            "other_engine_or_node_transport": False,
            "product_user_transport": False,
            "promotion_deployment_publication_training_or_release": False,
            "existing_F0_H0_stop_remains_effective": True,
            "existing_estimator_B_not_ready_remains_effective": True,
            "existing_conversion_C_not_established_remains_effective": True,
            "existing_mechanism_and_transfer_decisions_remain_effective": True,
        },
        "protected_access": {
            "official_selection_confirmation_final_test": "unopened",
            "research_confirmation": "unopened",
            "source_pool_2eb04f54_remaining_108": "unread_and_unconsumed",
        },
    }
    sealed = write_sealed_json(output, payload, identity_field="plan_identity")
    print(sealed["plan_identity"])
    print(sealed["precision_preregistration"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
