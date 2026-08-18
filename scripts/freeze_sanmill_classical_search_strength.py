#!/usr/bin/env python3
"""Freeze calibration and formal plans for classical-search measurement."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    write_sealed_json,
)
from learned_ai.evaluation.sanmill_classical_search_strength import (
    AUTHORIZATION_SCHEMA,
    CALIBRATION_PLAN_SCHEMA,
    CALIBRATION_RESULT_SCHEMA,
    PLAN_SCHEMA,
    ClassicalSearchStrengthError,
    calibration_membership,
    phase_balanced_membership,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    POOL_SCHEMA,
    load_sealed,
    sha256_file,
)


CALIBRATION_PLAN_PATH = Path(
    "docs/experiments/sanmill-classical-search-calibration-v1.json"
)
CALIBRATION_RESULT_PATH = Path(
    "docs/evidence/sanmill-classical-search-calibration-v1-2026-08-18.json"
)
FINAL_PLAN_PATH = Path(
    "docs/experiments/sanmill-classical-search-strength-v2.json"
)
AUTHORIZATION_PATH = Path(
    "docs/experiments/sanmill-classical-search-strength-v2/authorization.json"
)
SUPERSEDED_PLAN_PATH = Path(
    "docs/experiments/sanmill-classical-search-strength-v1.json"
)
SUPERSEDED_AUTHORIZATION_PATH = Path(
    "docs/experiments/sanmill-classical-search-strength-v1/authorization.json"
)

PRODUCT_ROOT = Path("tmp/classical-search-main-snapshot-4e4a724/tree")
NATIVE_SITE = Path("tmp/classical-search-main-snapshot-4e4a724/site")
START_POOL_PATH = Path(
    "docs/experiments/sanmill-safe-guidance-gameplay-start-pool-v1.json"
)
LIGHTWEIGHT_PLAN_PATH = Path(
    "docs/experiments/sanmill-trained-model-lightweight-v1.json"
)
LIGHTWEIGHT_RESULT_PATH = Path(
    "docs/evidence/sanmill-trained-model-lightweight-v1-manifest-2026-08-17.json"
)


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _alias_package(name: str, package_dir: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ClassicalSearchStrengthError("product package spec is absent")
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return package


def _resource_hashes() -> dict[str, str]:
    paths = {
        "evolved_weights": ROOT / "data/weights/best.json",
        "fullgame_db": ROOT / "data/endgame/fullgame.bin",
        "value_place": ROOT / "data/value_net_phase_place.npz",
        "value_move": ROOT / "data/value_net_phase_move.npz",
        "value_fly": ROOT / "data/value_net_phase_fly.npz",
        "gap_net": ROOT / "data/gap_net.npz",
    }
    result = {key: sha256_file(path) for key, path in paths.items()}
    for path in sorted((ROOT / "data/endgame").glob("*.wdl")):
        result[f"endgame/{path.name}"] = sha256_file(path)
    return result


def _resolved_weights(product_root: Path) -> dict[str, Any]:
    _alias_package("freeze_product_main_ai", product_root / "ai")
    heuristics = importlib.import_module("freeze_product_main_ai.heuristics")
    evolved = json.loads(
        (ROOT / "data/weights/best.json").read_text(encoding="utf-8")
    )
    def weight(key: str, default: int) -> int:
        return int(evolved.get(key, default))

    value = heuristics.HeuristicWeights(
        close_mill=weight("close_mill", 500),
        cycling_mill=weight("cycling_mill", 300),
        block_opponent_mill=weight("block_opponent_mill", 400),
        stop_opponent_mills=weight("stop_opponent_mills", 450),
        feeder_diamond=weight("feeder_diamond", 200),
        mill_wrapping=weight("mill_wrapping", 150),
        cardinal_block=weight("cardinal_block", 400),
        scatter_placement=weight("scatter_placement", 100),
        setup_mill=weight("setup_mill", 150),
        mill_opening=weight("mill_opening", 200),
        long_term_position=weight("long_term_position", 100),
        mill_count_scale=weight("mill_count_scale", 100),
        mobility_scale=weight("mobility_scale", 100),
        blocked_scale=weight("blocked_scale", 100),
        make_mistakes=weight("make_mistakes", 0),
        opening_adherence=weight("opening_adherence", 50),
        value_net_blend=weight("value_net_blend", 80),
        humanlike_blend=weight("humanlike_blend", 0),
        cross_mill_cycling=weight("cross_mill_cycling", 300),
        move_variance_pct=weight("move_variance_pct", 0),
    )
    return dict(value.__dict__)


def _implementation_hashes() -> dict[str, str]:
    paths = (
        "learned_ai/evaluation/sanmill_classical_search_strength.py",
        "scripts/run_sanmill_classical_search_strength.py",
        "scripts/freeze_sanmill_classical_search_strength.py",
    )
    return {path: sha256_file(ROOT / path) for path in paths}


def _product_contract() -> dict[str, Any]:
    product_root = (ROOT / PRODUCT_ROOT).resolve()
    native_site = (ROOT / NATIVE_SITE).resolve()
    pyd = next((native_site / "nmm_core").glob("*.pyd"))
    main_commit = _git("rev-parse", "origin/main")
    main_tree = _git("rev-parse", "origin/main^{tree}")
    expected_commit = "4e4a7241e9d5427100b46dfe34f5ae384ff9f613"
    if main_commit != expected_commit:
        raise ClassicalSearchStrengthError(
            "origin/main changed after the audited product route"
        )
    implementation = {
        "game_ai": sha256_file(product_root / "ai/game_ai.py"),
        "heuristics": sha256_file(product_root / "ai/heuristics.py"),
        "native_extension": sha256_file(pyd),
    }
    return {
        "source_commit": main_commit,
        "source_tree": main_tree,
        "source_archive_rule": "git archive origin/main at the frozen commit",
        "implementation_sha256": implementation,
        "native_wheel_sha256": sha256_file(
            next((ROOT / PRODUCT_ROOT.parent / "wheels").glob("*.whl"))
        ),
        "resource_sha256": _resource_hashes(),
        "resolved_weights": _resolved_weights(product_root),
        "route": "origin/main server-canonical balanced AI-vs-AI GameAI",
        "max_depth": 14,
        "timed_search_threads": 2,
        "deterministic_search_threads": 1,
        "extended_qsearch": True,
        "malom_in_product_route": "absent-stale-tracked-path",
        "fullgame_db": "enabled-read-only",
        "endgame_solved_db": "enabled-read-only",
        "phase_value_net": "enabled-read-only-at-resolved-weight",
        "gap_net": "enabled-read-only",
        "human_pref_net": "absent",
        "session_opening_or_trajectory_context": "not-used-AI-vs-AI-canonical-route",
        "sanmill_seed": 42,
    }


def freeze_calibration() -> int:
    if (ROOT / CALIBRATION_PLAN_PATH).exists():
        raise ClassicalSearchStrengthError("calibration plan already exists")
    pool, pool_sha = load_sealed(
        ROOT / START_POOL_PATH,
        schema=POOL_SCHEMA,
        identity_field="pool_identity",
    )
    excluded = [
        "00092c974cabf05874f066b8948e791f9fdc82d84a65759da1ba78f212a643b0"
    ]
    state_ids = calibration_membership(
        pool["states"],
        per_phase=4,
        namespace="sanmill-classical-search-calibration-v1-20260818",
        excluded_start_ids=excluded,
    )
    payload = {
        "schema_version": CALIBRATION_PLAN_SCHEMA,
        "status": "frozen_before_any_classical_search_observation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Map product difficulty 9/10 wall clocks to deterministic work budgets without game outcomes.",
        "source_commit_before_plan": _git("rev-parse", "HEAD"),
        "product_contract": _product_contract(),
        "start_pool": {
            "path": str(START_POOL_PATH).replace("\\", "/"),
            "pool_identity": pool["pool_identity"],
            "file_sha256": pool_sha,
            "excluded_start_ids": excluded,
        },
        "calibration": {
            "selection_namespace": "sanmill-classical-search-calibration-v1-20260818",
            "selection_is_result_blind": True,
            "states_per_phase": 4,
            "placement_requires_more_than_four_pieces_on_board": True,
            "state_ids": state_ids,
            "membership_identity": canonical_sha256(state_ids),
            "wall_clock_settings": {"9": 30, "10": 60},
            "cold_fresh_instance_once_per_state": True,
            "node_mapping": "floor median positive-node count to 1000, minimum 1000",
            "fixed_node_determinism": "two fresh single-thread instances on one blind state per phase",
            "formal_measurement_excludes_all_calibration_states": True,
            "no_complete_games": True,
        },
        "precision_design": {
            "target_maximum_half_width": 0.075,
            "reason": "7.5pp is narrow enough to distinguish the observed 30/45/56-percent scale, not a 1pp equivalence problem",
            "sample_size_candidates_descending": [128, 96, 64, 48, 32, 24, 16, 12, 8],
            "runtime_rule": "largest candidate whose conservative projected total is at most 75 percent of the remaining active-time envelope and whose games fit the cap",
            "prior_max_paired_sd": 0.28776628775772345,
        },
        "resource_envelope": {
            "maximum_calibration_seconds": 3600,
            "complete_games": 0,
            "database_writes": 0,
            "training_updates": 0,
        },
        "protected_access": {
            "official_selection_content_reads": 0,
            "official_confirmation_content_reads": 0,
            "official_final_test_content_reads": 0,
            "research_confirmation_content_reads": 0,
            "source_pool_2eb04f54_reads_or_consumption": 0,
        },
        "outputs": {
            "calibration_result": str(CALIBRATION_RESULT_PATH).replace("\\", "/")
        },
        "implementation_files": _implementation_hashes(),
        "claim_boundary": {
            "timing_and_determinism_only": True,
            "no_strength_result": True,
            "no_product_or_human_claim": True,
            "no_promotion_or_deployment": True,
        },
    }
    sealed = write_sealed_json(
        ROOT / CALIBRATION_PLAN_PATH, payload, identity_field="plan_identity"
    )
    print(sealed["plan_identity"])
    return 0


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ClassicalSearchStrengthError("empty runtime sample")
    index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _planning_inputs(calibration: dict[str, Any]) -> dict[str, Any]:
    per_arm_seconds: dict[str, float] = {}
    for difficulty in (9, 10):
        rows = calibration["fixed_node_checks"][str(difficulty)][
            "fresh_instance_checks"
        ]
        seconds = [
            float(repeat["elapsed_seconds"])
            for row in rows
            for repeat in (row["first"], row["second"])
        ]
        per_arm_seconds[str(difficulty)] = _percentile(seconds, 0.75)

    reference = json.loads(
        (ROOT / "docs/evidence/sanmill-safe-guidance-gameplay-attempt-002-manifest-2026-08-16.json").read_text(
            encoding="utf-8"
        )
    )
    random_games = [row for row in reference["games"] if row["arm"] == "random-safe"]
    candidate_turns = [
        sum(turn["actor"] == "candidate" for turn in row["turns"])
        for row in random_games
    ]
    reproduction_seconds = [float(row["game_elapsed_seconds"]) for row in random_games]
    return {
        "fixed_node_seconds_p75": per_arm_seconds,
        "candidate_turns_per_game_p75": _percentile(
            [float(value) for value in candidate_turns], 0.75
        ),
        "known_answer_seconds_per_game_p75": _percentile(
            reproduction_seconds, 0.75
        ),
    }


def _select_sample_size(
    *,
    calibration_seconds: float,
    planning: dict[str, Any],
    budget_points: int,
) -> tuple[int, list[dict[str, Any]]]:
    rows = []
    available = 64_800.0 - calibration_seconds
    time_ceiling = 0.85 * available
    per_start_classical = 2.0 * planning["candidate_turns_per_game_p75"] * sum(
        planning["fixed_node_seconds_p75"].values()
    )
    per_start_reproduction = 2.0 * planning["known_answer_seconds_per_game_p75"]
    for count in (128, 96, 64, 48, 32, 24, 16, 12, 8):
        predicted = count * (per_start_classical + per_start_reproduction)
        games = count * 2 * (1 + budget_points)
        feasible = predicted <= time_ceiling and games <= 1_600
        rows.append(
            {
                "starts": count,
                "predicted_seconds": predicted,
                "planned_games": games,
                "within_time_design_ceiling": predicted <= time_ceiling,
                "within_game_ceiling": games <= 1_600,
                "selected": False,
            }
        )
        if feasible:
            rows[-1]["selected"] = True
            return count, rows
    raise ClassicalSearchStrengthError(
        "no preregistered sample size fits the conservative resource design"
    )


def freeze_final() -> int:
    if (ROOT / FINAL_PLAN_PATH).exists() or (ROOT / AUTHORIZATION_PATH).exists():
        raise ClassicalSearchStrengthError("formal plan or authorization exists")
    calibration_plan, calibration_plan_sha = load_sealed(
        ROOT / CALIBRATION_PLAN_PATH,
        schema=CALIBRATION_PLAN_SCHEMA,
        identity_field="plan_identity",
    )
    calibration, calibration_sha = load_sealed(
        ROOT / CALIBRATION_RESULT_PATH,
        schema=CALIBRATION_RESULT_SCHEMA,
        identity_field="result_identity",
    )
    if (
        calibration["plan_identity"] != calibration_plan["plan_identity"]
        or calibration["plan_file_sha256"] != calibration_plan_sha
    ):
        raise ClassicalSearchStrengthError("calibration binding differs")
    pool, pool_sha = load_sealed(
        ROOT / START_POOL_PATH,
        schema=POOL_SCHEMA,
        identity_field="pool_identity",
    )
    budgets = {
        difficulty: int(
            calibration["node_budget_mapping"][str(difficulty)][
                "mapped_node_budget"
            ]
        )
        for difficulty in (9, 10)
    }
    planning = _planning_inputs(calibration)
    count, sample_rows = _select_sample_size(
        calibration_seconds=float(calibration["resources"]["active_seconds"]),
        planning=planning,
        budget_points=2,
    )
    calibration_ids = list(calibration_plan["calibration"]["state_ids"])
    start_ids = phase_balanced_membership(
        pool["states"],
        count=count,
        namespace="sanmill-classical-search-strength-v1-formal-20260818",
        excluded_start_ids=[
            *calibration_plan["start_pool"]["excluded_start_ids"],
            *calibration_ids,
        ],
    )
    phase_by_id = {
        str(row["state_id"]): str(row["phase"])
        for row in pool["states"]
        if str(row["state_id"]) in set(start_ids)
    }
    lightweight_plan = json.loads(
        (ROOT / LIGHTWEIGHT_PLAN_PATH).read_text(encoding="utf-8")
    )
    result_file_sha = sha256_file(ROOT / LIGHTWEIGHT_RESULT_PATH)
    expected_half_width = (
        1.96
        * float(calibration_plan["precision_design"]["prior_max_paired_sd"])
        / math.sqrt(count)
    )
    superseded_plan, superseded_plan_sha = load_sealed(
        ROOT / SUPERSEDED_PLAN_PATH,
        schema=PLAN_SCHEMA,
        identity_field="plan_identity",
    )
    superseded_authorization, superseded_authorization_sha = load_sealed(
        ROOT / SUPERSEDED_AUTHORIZATION_PATH,
        schema=AUTHORIZATION_SCHEMA,
        identity_field="authorization_identity",
    )
    arms = [
        {
            "arm": f"classical-difficulty-{difficulty}-nodes-{budgets[difficulty]}",
            "difficulty": difficulty,
            "node_budget": budgets[difficulty],
            "wall_clock_mapping_seconds": 30 if difficulty == 9 else 60,
        }
        for difficulty in (9, 10)
    ]
    planned_games = count * 2 * (1 + len(arms))
    implementation = _implementation_hashes()
    payload = {
        "schema_version": PLAN_SCHEMA,
        "status": "frozen_v2_after_timing_calibration_before_known_answer_or_classical_games",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "How strong are origin/main difficulty 9/10 classical coordinators against the pinned 100k-node Sanmill runtime?",
        "source_commit_before_plan": _git("rev-parse", "HEAD"),
        "supersedes_unexecuted_v1": {
            "plan_path": str(SUPERSEDED_PLAN_PATH).replace("\\", "/"),
            "plan_identity": superseded_plan["plan_identity"],
            "plan_file_sha256": superseded_plan_sha,
            "authorization_path": str(SUPERSEDED_AUTHORIZATION_PATH).replace(
                "\\", "/"
            ),
            "authorization_identity": superseded_authorization[
                "authorization_identity"
            ],
            "authorization_file_sha256": superseded_authorization_sha,
            "measurement_marker_created": False,
            "known_answer_or_candidate_games_observed": 0,
            "reason": "v1 selected 32 starts with projected 9.97pp half-width while retaining a 7.5pp precision gate; v2 corrects this timing-only design inconsistency before any outcome read",
        },
        "calibration": {
            "plan_identity": calibration_plan["plan_identity"],
            "plan_file_sha256": calibration_plan_sha,
            "result_identity": calibration["result_identity"],
            "result_file_sha256": calibration_sha,
            "active_seconds": calibration["resources"]["active_seconds"],
            "node_budget_mapping": calibration["node_budget_mapping"],
            "fixed_node_checks": calibration["fixed_node_checks"],
        },
        "product_contract": calibration_plan["product_contract"],
        "start_pool": {
            "path": str(START_POOL_PATH).replace("\\", "/"),
            "pool_identity": pool["pool_identity"],
            "file_sha256": pool_sha,
            "excluded_start_ids": calibration_plan["start_pool"][
                "excluded_start_ids"
            ],
        },
        "start_subset": {
            "selection_namespace": "sanmill-classical-search-strength-v2-formal-20260818",
            "selection_is_result_blind": True,
            "calibration_states_excluded": calibration_ids,
            "state_ids": start_ids,
            "phase_by_state_id": phase_by_id,
            "membership_identity": canonical_sha256(start_ids),
            "starts": count,
        },
        "sample_size_design": {
            "planning_inputs": planning,
            "candidate_table_until_selection": sample_rows,
            "selected_starts": count,
            "resource_fraction_ceiling": 0.85,
            "resource_rule_change_from_calibration_plan": "raised from 0.75 to 0.85 only to resolve the timing-revealed precision inconsistency; no game outcome or known-answer content was read",
            "observed_prior_max_paired_sd": calibration_plan[
                "precision_design"
            ]["prior_max_paired_sd"],
            "projected_half_width_at_selected_n": expected_half_width,
            "target_maximum_half_width": 0.085,
            "interpretation": "8.5pp is below the smallest 11pp separation in the 30/45/56-percent scale; small differences near v4 may remain inconclusive",
        },
        "experiment": {
            "classical_arms": arms,
            "colors_per_start": 2,
            "schedule_namespace": "sanmill-classical-search-strength-v2-games-20260818",
            "known_answer_games": count * 2,
            "classical_games": count * 2 * len(arms),
            "planned_complete_games": planned_games,
            "maximum_post_start_logical_plies": 1536,
            "safety_cap_disposition": "incomplete-never-a-draw",
        },
        "primary_estimand": {
            "score": "strict W/D/L encoded 1/0.5/0",
            "unit": "one start after averaging candidate W and B",
            "contrasts": "each classical budget point minus each old arm on the identical subset",
            "interval": "normal 95 percent interval over start-level paired differences",
        },
        "primary_decision": {
            "maximum_half_width": 0.085,
            "classical_higher": "interval lower bound above zero",
            "classical_lower": "interval upper bound below zero",
            "direction_inconclusive": "interval includes zero",
            "precision_inadequate": "half width above 8.5 percentage points",
            "no_result_based_early_stop_or_extension": True,
        },
        "secondary_metrics": {
            "items": [
                "positional self-downgrade events W-to-D, W-to-L and D-to-L",
                "terminal reasons and strict WDL by frozen source phase",
                "candidate search nodes, completed depth and elapsed work",
            ],
            "cannot_flip_primary": True,
        },
        "known_answer": {
            "required_before_product_runtime_load": True,
            "arm": "random-safe",
            "reference_path": lightweight_plan["known_answer_reproduction"][
                "reference_result"
            ]["path"],
            "reference": {
                "identity": lightweight_plan["known_answer_reproduction"][
                    "reference_result"
                ]["identity"],
                "file_sha256": lightweight_plan["known_answer_reproduction"][
                    "reference_result"
                ]["file_sha256"],
            },
            "per_game_exact_fields": "moves, terminal reason, strict history, no-progress and repetition clocks",
            "failure_disposition": "hard stop before classical games",
        },
        "guidance_input": lightweight_plan["guidance_runtime_input"],
        "sanmill_contract": lightweight_plan["sanmill_contract"],
        "malom_contract": lightweight_plan["malom_contract"],
        "prior_results": {
            "path": str(LIGHTWEIGHT_RESULT_PATH).replace("\\", "/"),
            "file_sha256": result_file_sha,
            "recomputed_only_on_formal_subset": True,
        },
        "resource_envelope": {
            "maximum_active_seconds": 64_800,
            "maximum_complete_games": 1_600,
            "planned_complete_games": planned_games,
            "maximum_parallel_measurement_processes": 1,
            "maximum_parallel_sanmill_processes": 1,
            "engine_search_anomaly_ceiling": 2_000_000,
            "malom_query_anomaly_ceiling": 50_000_000,
            "training_updates": 0,
            "checkpoint_modifications": 0,
            "database_writes": 0,
        },
        "protected_access": calibration_plan["protected_access"],
        "implementation_files": implementation,
        "outputs": {
            "namespace": "out/evaluation/sanmill-classical-search-strength-v2-20260818-001",
            "result": "docs/evidence/sanmill-classical-search-strength-v2-manifest-2026-08-18.json",
            "evidence_document": "docs/evidence/sanmill-classical-search-strength-v2-2026-08-18.md",
        },
        "interpretation_rules": {
            "all_trained_models_below": "if both classical point estimates and intervals establish superiority over all trained arms, state it without softening",
            "small_v4_difference": "do not claim equality or superiority when the interval includes zero",
            "product_clock_mapping": "node budgets are calibrated proxies, not exact wall-clock product executions",
            "fixed_node_threads": "one thread is required for determinism; product uses two timed threads",
        },
        "claim_boundary": {
            "internal_directional_measurement_only": True,
            "exact_fixed_runtime_and_start_subset_only": True,
            "human_or_product_population_claim": False,
            "equivalence_claim": False,
            "promotion_or_deployment": False,
            "training_authorization": False,
            "position_safety": "A_pos diagnostic only; classical moves are unconstrained",
        },
    }
    sealed_plan = write_sealed_json(
        ROOT / FINAL_PLAN_PATH, payload, identity_field="plan_identity"
    )
    plan_sha = sha256_file(ROOT / FINAL_PLAN_PATH)
    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "operator": "product-owner-direct",
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_basis": "Product owner request dated 2026-08-18 to execute one bounded difficulty 9/10 classical-search measurement; timing-only v2 precision correction froze before any outcome read.",
        "plan_identity": sealed_plan["plan_identity"],
        "plan_file_sha256": plan_sha,
        "source_commit": _git("rev-parse", "HEAD"),
        "output_namespace": payload["outputs"]["namespace"],
        "resource_envelope": payload["resource_envelope"],
        "one_execution_only": True,
        "automatic_retry_or_resume": False,
        "prohibited": [
            "training",
            "weight updates",
            "checkpoint changes",
            "database writes",
            "protected-segment access",
            "promotion or deployment",
        ],
    }
    sealed_auth = write_sealed_json(
        ROOT / AUTHORIZATION_PATH,
        authorization,
        identity_field="authorization_identity",
    )
    print(sealed_plan["plan_identity"])
    print(sealed_auth["authorization_identity"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("calibration", "final"))
    args = parser.parse_args()
    if args.stage == "calibration":
        return freeze_calibration()
    return freeze_final()


if __name__ == "__main__":
    raise SystemExit(main())
