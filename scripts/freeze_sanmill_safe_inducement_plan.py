#!/usr/bin/env python3
"""Freeze the preprobe contract and unlaunched main experiment protocol."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.human_f0h0_feasibility import write_sealed_json
from learned_ai.evaluation.sanmill_safe_inducement import (
    PHASES,
    PLAN_SCHEMA,
    load_state_pool,
)
from learned_ai.training.sanmill_referee import (
    inspect_sanmill_training_installation,
    training_installation_record,
)


def _local_path(config_path: Path, key: str) -> Path:
    value = json.loads(config_path.read_text(encoding="utf-8"))
    raw = value.get(key) if isinstance(value, dict) else None
    if not isinstance(raw, str) or not raw:
        raise RuntimeError(f"local path is absent: {key}")
    path = Path(raw)
    return path if path.is_absolute() else (_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-pool",
        default=(
            "docs/experiments/sanmill-safe-inducement-preprobe-"
            "state-pool-v1.json"
        ),
    )
    parser.add_argument("--paths-config", default="data/training_paths.local.json")
    parser.add_argument(
        "--output",
        default="docs/experiments/sanmill-safe-inducement-mechanism-v1.json",
    )
    args = parser.parse_args()

    pool, pool_file_sha = load_state_pool(_ROOT / args.state_pool)
    checkout = _local_path(
        _ROOT / args.paths_config,
        "sanmill_training_checkout",
    )
    installation = inspect_sanmill_training_installation(checkout)
    runtime = training_installation_record(installation, seed=42)
    fixtures = []
    for phase in PHASES:
        rows = sorted(
            (row for row in pool["states"] if row["phase"] == phase),
            key=lambda row: (row["logical_ply"], row["state_id"]),
        )
        fixtures.extend(
            {
                "state_id": row["state_id"],
                "a_pos_index": 0,
                "selection": "two lowest logical ply states per phase; first canonical A_pos action",
            }
            for row in rows[:2]
        )

    payload = {
        "schema_version": PLAN_SCHEMA,
        "status": "frozen_preprobe_authorized_main_unlaunched",
        "experiment_id": "sanmill-safe-inducement-mechanism-v1",
        "research_question": (
            "Within A_pos(S), can directed action choice increase the probability "
            "that one fixed-node deterministic Sanmill response loses its own "
            "best positional WDL tier?"
        ),
        "claim_boundary": {
            "opponent": "exact pinned deterministic Sanmill runtime only",
            "safe_set": "A_pos",
            "positional_only": True,
            "A_allow_claim": False,
            "human_trap_claim": False,
            "playing_strength_claim": False,
            "other_engines_nodes_users_or_products": False,
            "promotion_deployment_publication_or_release": False,
            "existing_F0_H0_stop_remains_effective": True,
            "existing_estimator_B_not_ready_remains_effective": True,
            "existing_conversion_C_not_established_remains_effective": True,
        },
        "input_identities": {
            "state_pool": {
                "pool_identity": pool["pool_identity"],
                "membership_identity": pool["state_membership_identity"],
                "file_sha256": pool_file_sha,
                "states": pool["state_count"],
                "A_pos_actions": sum(
                    int(row["a_pos_cardinality"]) for row in pool["states"]
                ),
            },
            "source_crossfit_structure_identity": pool["source"][
                "crossfit_structure_identity"
            ],
            "malom_content_sha256": pool["input_identities"]["malom"][
                "content_sha256"
            ],
            "malom_trust_level": "sector-corrected-v1",
        },
        "sanmill_contract": {
            "runtime_identity": runtime["identity"],
            "commit": runtime["checkout_head"],
            "tree": runtime["tree"],
            "binary_sha256": runtime["binary_sha256"],
            "binary_size": runtime["binary_size"],
            "license": "GNU AGPL version 3",
            "license_sha256": runtime["license"]["sha256"],
            "rules_identity_sha256": (
                "3e62cb93a1e0afe4534ce4824d233344816050b547bb8761dd7fe985d8ad399f"
            ),
            "strict_referee_profile": "mif-stable-moving-v1",
            "strict_referee_semantic_digest": (
                "sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94"
            ),
            "seed": 42,
            "protocol_timeout_seconds": 10.0,
            "search_timeout_seconds": 120.0,
            "command": "go logical nodes N; no explicit depth",
            "fixed_options": runtime["strict_options"],
            "time_control_priority": "MoveTimeMs=0; fixed nodes are binding except retained phase-depth early stop",
            "fresh_process_per_measurement_cell": True,
            "concurrent_processes": 1,
        },
        "determinism_gate": {
            "must_pass_before_measurement": True,
            "fixtures": fixtures,
            "budgets": [1_000, 10_000, 100_000, 500_000],
            "same_process_repetitions": 2,
            "fresh_process_orders": ["forward", "reverse"],
            "comparison": "exact equality of UciLogicalTurnResult.semantic_record",
            "timing_and_raw_protocol_text_excluded": True,
            "failure_action": "conclusion C; no statistical measurement",
        },
        "estimands": {
            "state_weighting": "each frozen state has equal weight",
            "within_state_weighting": "uniform over complete A_pos(S)",
            "b_i": "downgrading responses divided by |A_pos(S)|",
            "o_i": "one iff at least one A_pos(S) action induces a downgrade",
            "gain_i": "o_i - b_i",
            "b": "mean_i(b_i)",
            "o": "mean_i(o_i)",
            "o_minus_b": "mean_i(gain_i)",
            "strict_terminal_successor": (
                "known zero response-downgrade indicator, retained in A_pos denominator"
            ),
            "engine_search_or_oracle_failure": "abstain entire state-budget cell",
            "downgrade_transitions": ["W->D", "W->L", "D->L"],
            "within_tier_regret_is_primary": False,
            "independent_unit": "source-game-unique frozen state",
            "additional_clustering": "none because each source game appears once",
        },
        "preprobe": {
            "authorization": "product-owner-direct-current-task",
            "node_budgets": [1_000, 10_000, 100_000, 500_000],
            "budget_rationale": (
                "10^3, 10^4, and 10^5 cover three search scales; 500,000 is "
                "the formal-evaluation upper anchor"
            ),
            "measurement_order_seed": (
                "sanmill-safe-inducement-preprobe-cell-order-v1-20260816"
            ),
            "measurement": "exhaust every frozen A_pos action at every budget",
            "strata": ["phase", "engine_reply_state_WDL_tier"],
            "timing": "per search plus per process/replay cell wall time",
            "uncertainty": {
                "method": "state-level nonparametric percentile bootstrap",
                "bootstrap_repetitions": 10_000,
                "bootstrap_seed": (
                    "sanmill-safe-inducement-preprobe-bootstrap-v1-20260816"
                ),
                "zero_event_rule": "return empirical zero with no prior or pseudo-event",
                "interpretation": "fixed-pool engineering interval, not population inference",
            },
            "signal_gate": {
                "minimum_evaluable_states": 30,
                "minimum_downgrade_actions": 2,
                "minimum_baseline_rate": 0.01,
                "maximum_baseline_rate": 0.30,
                "minimum_oracle_gain": 0.05,
                "budget_selection": "highest budget passing every gate",
                "rationale": (
                    "below 1% is operationally near-zero in this 36-state probe; "
                    "above 30% makes random safe choice too error-rich; a 5pp "
                    "oracle increment is the minimum mechanism value worth a "
                    "separate learnability experiment"
                ),
            },
            "resource_envelope": {
                "maximum_engine_single_step_queries": 100_000,
                "maximum_active_seconds": 7_200,
                "maximum_concurrent_evaluators": 1,
                "maximum_concurrent_sanmill_processes": 1,
                "maximum_complete_games": 0,
                "maximum_model_loads": 0,
                "maximum_training_updates": 0,
                "stop_at_any_limit": True,
                "automatic_retry_or_extension": False,
            },
        },
        "preprobe_decision": {
            "A": "at least one budget passes every frozen signal gate",
            "B": "no budget passes every frozen signal gate",
            "C": "determinism, identity, A_pos, or estimator definition fails",
            "A_action": "report recommended budget and request separate main authorization",
            "B_action": "close this engine-inducement direction",
            "C_action": "stop and redesign; do not average repeated stochastic searches",
        },
        "main_experiment_unlaunched": {
            "execution_authorized": False,
            "activation": "only after conclusion A and separate product-owner authorization",
            "state_pool_rule": (
                "360 source-game-unique states, 120 per phase, selected from the "
                "same frozen 6,400-game research-exploration population by a new "
                "blind SHA-256 namespace, excluding preprobe states"
            ),
            "node_budget": "frozen highest preprobe budget passing all signal gates",
            "measurement": "exhaust complete A_pos(S) once per state",
            "primary_estimand": "state-uniform o_minus_b",
            "secondary_estimands": ["b", "o"],
            "strata": ["phase", "engine_reply_state_WDL_tier"],
            "downgrade_transitions": ["W->D", "W->L", "D->L"],
            "interval": {
                "method": "state-level nonparametric percentile bootstrap",
                "repetitions": 20_000,
                "seed": "sanmill-safe-inducement-main-bootstrap-v1",
                "zero_event_rule": "no prior and no pseudo-event",
            },
            "mechanism_success_gate": {
                "minimum_point_o_minus_b": 0.05,
                "minimum_lower_95_o_minus_b": 0.05,
                "minimum_evaluable_states": 330,
                "determinism_gate": True,
                "all_conditions_conjunctive": True,
                "rationale": (
                    "require at least one extra induced downgrade per 20 source "
                    "states even after uncertainty before paying for prediction"
                ),
            },
            "abstention": (
                "any missing Malom value, illegal bridge result, or failed engine "
                "search abstains the whole state and is never converted to zero"
            ),
            "resource_envelope": {
                "maximum_states": 360,
                "maximum_engine_single_step_queries": 40_000,
                "maximum_malom_queries": 250_000,
                "maximum_active_seconds": 14_400,
                "maximum_concurrent_evaluators": 1,
                "maximum_concurrent_sanmill_processes": 1,
                "maximum_complete_games": 0,
                "maximum_model_loads": 0,
                "maximum_training_updates": 0,
                "stop_at_any_limit": True,
                "automatic_retry_or_extension": False,
            },
            "forbidden_claims": [
                "human trap ability",
                "playing strength",
                "refresh causality",
                "promotion",
                "deployment",
                "publication",
                "release",
            ],
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
        _ROOT / args.output,
        payload,
        identity_field="plan_identity",
    )
    print(sealed["plan_identity"])
    print(runtime["identity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
