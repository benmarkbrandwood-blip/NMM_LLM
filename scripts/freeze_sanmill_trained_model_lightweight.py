"""Freeze the lightweight trained-model plan or its one-time authorization."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    write_sealed_json,
)
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    RESULT_SCHEMA as READINESS_SCHEMA,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    PLAN_SCHEMA as GUIDANCE_PLAN_SCHEMA,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    POOL_SCHEMA,
    load_sealed as load_guidance_sealed,
)
from learned_ai.evaluation.sanmill_trained_model_baseline import (
    load_plan as load_candidate_source_plan,
)
from learned_ai.evaluation.sanmill_trained_model_lightweight import (
    AUTHORIZATION_SCHEMA,
    CANDIDATE_ARMS,
    EXPECTED_CANDIDATE_GAMES,
    EXPECTED_REPRODUCTION_GAMES,
    EXPECTED_STARTS,
    EXPECTED_TOTAL_GAMES,
    MAXIMUM_HALF_WIDTH,
    PLAN_SCHEMA,
    sha256_file,
)


PLAN_PATH = (
    ROOT / "docs/experiments/sanmill-trained-model-lightweight-v1.json"
)
AUTHORIZATION_PATH = (
    ROOT
    / "docs/experiments/sanmill-trained-model-lightweight-v1/authorization.json"
)
GUIDANCE_PLAN_PATH = (
    ROOT / "docs/experiments/sanmill-safe-guidance-gameplay-v1.json"
)
POOL_PATH = (
    ROOT / "docs/experiments/sanmill-safe-guidance-gameplay-start-pool-v1.json"
)
REFERENCE_PATH = (
    ROOT
    / "docs/evidence/"
    "sanmill-safe-guidance-gameplay-attempt-002-manifest-2026-08-16.json"
)
READINESS_PATH = (
    ROOT
    / "docs/evidence/"
    "human-feature-deviation-estimator-readiness-manifest-2026-08-15.json"
)
CANDIDATE_SOURCE_PLAN_PATH = (
    ROOT / "docs/experiments/sanmill-trained-model-baseline-v1.json"
)
MALOM_MANIFEST_PATH = ROOT / "data/manifests/malom-sector-corrected-v1.json"
OUTPUT_NAMESPACE = (
    "out/evaluation/sanmill-trained-model-lightweight-v1-20260817-001"
)
RESULT_PATH = (
    "docs/evidence/"
    "sanmill-trained-model-lightweight-v1-manifest-2026-08-17.json"
)
EVIDENCE_PATH = (
    "docs/evidence/sanmill-trained-model-lightweight-v1-2026-08-17.md"
)
IMPLEMENTATION_FILES = (
    "learned_ai/evaluation/sanmill_safe_guidance_gameplay.py",
    "learned_ai/evaluation/sanmill_trained_model_baseline.py",
    "learned_ai/evaluation/sanmill_trained_model_lightweight.py",
    "learned_ai/evaluation/training_aligned_policy.py",
    "scripts/run_sanmill_trained_model_lightweight.py",
)


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_reference() -> tuple[dict[str, Any], str]:
    raw = REFERENCE_PATH.read_bytes()
    value = json.loads(raw)
    body = dict(value)
    identity = body.pop("result_identity", None)
    if (
        value.get("schema_version")
        != "nmm.sanmill-safe-guidance-gameplay-result.v1"
        or canonical_sha256(body) != identity
        or identity
        != "f7abf95ca688860fa3b871757f42397b351a3c00701ba27363028f800a9222f0"
    ):
        raise RuntimeError("attempt-002 reference identity differs")
    return value, sha256_file(REFERENCE_PATH)


def _freeze_plan() -> None:
    if not (ROOT / "scripts/run_sanmill_trained_model_lightweight.py").is_file():
        raise RuntimeError("lightweight runner is absent")
    guidance, guidance_sha = load_guidance_sealed(
        GUIDANCE_PLAN_PATH,
        schema=GUIDANCE_PLAN_SCHEMA,
        identity_field="plan_identity",
    )
    pool, pool_sha = load_guidance_sealed(
        POOL_PATH,
        schema=POOL_SCHEMA,
        identity_field="pool_identity",
    )
    readiness, readiness_sha = load_guidance_sealed(
        READINESS_PATH,
        schema=READINESS_SCHEMA,
        identity_field="result_identity",
    )
    reference, reference_sha = _load_reference()
    candidate_source, candidate_source_sha = load_candidate_source_plan(
        CANDIDATE_SOURCE_PLAN_PATH
    )
    random_summary = reference["analysis"]["by_arm"]["random-safe"]
    if (
        random_summary["strict_wdl"]
        != {
            "wins": 21,
            "draws": 414,
            "losses": 73,
            "score_rate": 0.44881889763779526,
        }
        or random_summary["termination_reasons"]
        != {
            "drawFiftyMove": 305,
            "drawThreefoldRepetition": 109,
            "loseFewerThanThree": 47,
            "loseNoLegalMoves": 47,
        }
    ):
        raise RuntimeError("attempt-002 known answer differs")
    implementation = {
        path: sha256_file(ROOT / path) for path in IMPLEMENTATION_FILES
    }
    payload = {
        "schema_version": PLAN_SCHEMA,
        "experiment_id": "sanmill-trained-model-lightweight-v1",
        "status": "frozen_before_any_candidate_outcome",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": (
            "How do the retained-v4 and active three-specialist routes, free or "
            "A_pos-constrained, score against the exactly reproduced attempt-002 "
            "zero-training Malom-safe random baseline?"
        ),
        "repository_design_base_commit": _git("rev-parse", "HEAD"),
        "experiment": {
            "starts": EXPECTED_STARTS,
            "colors_per_start": 2,
            "reproduction_arm": "random-safe",
            "reproduction_games": EXPECTED_REPRODUCTION_GAMES,
            "candidate_arms": list(CANDIDATE_ARMS),
            "games_per_candidate_arm": EXPECTED_REPRODUCTION_GAMES,
            "candidate_games": EXPECTED_CANDIDATE_GAMES,
            "planned_total_games": EXPECTED_TOTAL_GAMES,
            "candidate_schedule_namespace": (
                "sanmill-trained-model-lightweight-v1-candidate-game"
            ),
            "primary_node_budget": 100000,
            "maximum_post_start_logical_plies": 1536,
            "safety_cap_disposition": "incomplete-never-a-draw",
        },
        "known_answer_reproduction": {
            "arm": "random-safe",
            "required_exact_match": True,
            "failure_disposition": "stop-before-any-candidate-model-load-or-game",
            "reference_result": {
                "path": str(REFERENCE_PATH.relative_to(ROOT)).replace("\\", "/"),
                "identity": reference["result_identity"],
                "file_sha256": reference_sha,
            },
            "expected": {
                "games": 508,
                "wins": 21,
                "draws": 414,
                "losses": 73,
                "score_rate": 0.44881889763779526,
                "termination_reasons": random_summary["termination_reasons"],
            },
            "per_game_match": (
                "game/start/color, WDL, terminal reason, terminal counters and "
                "history, logical plies, and full action-sequence identity"
            ),
            "random_safe_seed": guidance["experiment"]["random_safe_seed"],
            "schedule_rule": (
                "rebuild the original three-arm schedule, remove only the frozen "
                "failed start, retain original random-safe game IDs and ordinals"
            ),
        },
        "baseline": {
            "kind": "new exact reproduction of attempt-002 random-safe",
            "starts": 254,
            "games": 508,
            "random_safe_score_rate": 0.44881889763779526,
            "strict_wdl": {"wins": 21, "draws": 414, "losses": 73},
            "reference_result_identity": reference["result_identity"],
            "reference_result_file_sha256": reference_sha,
        },
        "primary_estimand": {
            "unit": "one start after averaging candidate W and B scores",
            "contrasts": [f"{arm} minus reproduced-random-safe" for arm in CANDIDATE_ARMS],
            "score": "strict W/D/L encoded as 1/0.5/0",
            "interval": (
                "start-clustered normal 95 percent interval using 1.96 times "
                "sample standard error"
            ),
        },
        "primary_decision": {
            "maximum_95_half_width": MAXIMUM_HALF_WIDTH,
            "trained_higher": "interval lower bound > 0",
            "safe_random_higher": "interval upper bound < 0",
            "no_directional_decision": (
                "interval contains zero and half width <= 0.015"
            ),
            "inconclusive_precision": "half width > 0.015",
            "equivalence_claim": False,
            "no_result_based_early_stop_or_extension": True,
        },
        "secondary_metrics": {
            "cannot_flip_primary": True,
            "items": [
                "self positional downgrade events per candidate turn, split W-to-D, W-to-L, D-to-L",
                "strict W/D/L and terminal reasons by arm and frozen source phase",
                "A_pos-constrained minus free within candidate",
                "A_pos-constrained arms minus attempt-002 full-guided 44.2913 percent",
            ],
        },
        "interpretation_rules": {
            "free_arm_asymmetry": (
                "random-safe has an A_pos guarantee while free arms do not"
            ),
            "constrained_comparability": (
                "A_pos-constrained arms remove action-safety asymmetry and are "
                "directly comparable with attempt-002 full-guided"
            ),
            "retained_v4_malom_note": (
                "retained-v4 also uses Malom in lookahead terminal early exits"
            ),
            "active_specialist_product_deviation": candidate_source[
                "candidate_runtime"
            ]["active_specialists"]["product_presearch_deviation"],
            "all_below_rule": (
                "if every trained arm is below random-safe, state that result "
                "without softening"
            ),
        },
        "start_pool": {
            "path": str(POOL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "pool_identity": pool["pool_identity"],
            "pool_file_sha256": pool_sha,
            "original_membership_identity": pool["state_membership_identity"],
            "formal_membership_identity": reference[
                "formal_start_membership_identity"
            ],
            "excluded_start_ids": reference["excluded_failed_start_ids"],
            "reuse_boundary": (
                "same candidate-blind frozen starts for paired internal comparison; "
                "not a new held-out or population sample"
            ),
        },
        "guidance_runtime_input": {
            "plan_path": str(GUIDANCE_PLAN_PATH.relative_to(ROOT)).replace("\\", "/"),
            "plan_identity": guidance["plan_identity"],
            "plan_file_sha256": guidance_sha,
            "readiness_path": str(READINESS_PATH.relative_to(ROOT)).replace("\\", "/"),
            "readiness_identity": readiness["result_identity"],
            "readiness_file_sha256": readiness_sha,
        },
        "candidate_runtime": candidate_source["candidate_runtime"],
        "candidate_runtime_source": {
            "path": str(CANDIDATE_SOURCE_PLAN_PATH.relative_to(ROOT)).replace("\\", "/"),
            "identity": candidate_source["plan_identity"],
            "file_sha256": candidate_source_sha,
            "role": (
                "path/hash contract source only; its authorizations, registry, "
                "coverage, rehearsal and pass state are neither loaded nor reused"
            ),
        },
        "sanmill_contract": candidate_source["sanmill_contract"],
        "malom_contract": {
            **candidate_source["malom_contract"],
            "manifest_path": str(MALOM_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
            "manifest_file_sha256": sha256_file(MALOM_MANIFEST_PATH),
        },
        "candidate_load_checks": {
            "retained_v4": [
                "bundle identity and file manifest",
                "checkpoint sha256 and payload sha256",
                "SpecialistDB identity and sector-corrected-v1 labels",
                "feature width 134 on sampled states",
                "same scorer path as frozen v3/v4 evaluation harness",
            ],
            "active_specialists": [
                "all three checkpoint SHA-256 values",
                "feature width 134 on sampled states",
                "two consecutive scores and selected moves identical",
                "product alpha-beta presearch deviation recorded",
            ],
            "timing": "after known-answer reproduction passes, before candidate games",
        },
        "resource_envelope": {
            "authorized_literal_maximum_complete_games": 3048,
            "planned_complete_games": EXPECTED_TOTAL_GAMES,
            "planned_is_narrower_than_literal_authorization": True,
            "maximum_active_seconds": 21600,
            "maximum_training_updates": 0,
            "maximum_database_writes": 0,
            "maximum_candidate_model_fits": 0,
            "internal_anomaly_engine_search_ceiling": 1000000,
            "internal_anomaly_malom_query_ceiling": 50000000,
        },
        "protected_access": {
            "official_selection_reads": 0,
            "official_confirmation_reads": 0,
            "official_final_test_reads": 0,
            "research_confirmation_reads": 0,
            "source_pool_2eb04f54_reads_or_consumption": 0,
        },
        "claim_boundary": {
            "internal_directional_measurement_only": True,
            "promotion_or_deployment": False,
            "public_claim": False,
            "human_trap_claim": False,
            "other_engine_or_runtime_generalization": False,
            "equivalence_claim": False,
            "position_safety": "A_pos positional-only where constrained",
            "A_allow_claim": False,
        },
        "outputs": {
            "namespace": OUTPUT_NAMESPACE,
            "result": RESULT_PATH,
            "evidence_document": EVIDENCE_PATH,
            "raw_reproduction_ledger": f"{OUTPUT_NAMESPACE}/reproduction-games.jsonl",
            "raw_candidate_ledger": f"{OUTPUT_NAMESPACE}/candidate-games.jsonl",
        },
        "implementation_files": implementation,
    }
    sealed = write_sealed_json(
        PLAN_PATH,
        payload,
        identity_field="plan_identity",
    )
    print(sealed["plan_identity"])


def _freeze_authorization() -> None:
    from learned_ai.evaluation.sanmill_trained_model_lightweight import load_plan

    plan, plan_sha = load_plan(PLAN_PATH)
    implementation = {
        path: sha256_file(ROOT / path) for path in plan["implementation_files"]
    }
    if implementation != plan["implementation_files"]:
        raise RuntimeError("implementation changed after plan freeze")
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": "authorized_once_unconsumed",
        "operator": "product-owner-direct",
        "grant_count": 1,
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_basis": (
            "Product owner direct instruction dated 2026-08-17 for one new "
            "lightweight internal measurement against the exactly reproduced "
            "attempt-002 Malom-safe random baseline"
        ),
        "plan": {
            "path": str(PLAN_PATH.relative_to(ROOT)).replace("\\", "/"),
            "identity": plan["plan_identity"],
            "file_sha256": plan_sha,
        },
        "source": {
            "commit": _git("rev-parse", "HEAD"),
            "tree": _git("rev-parse", "HEAD^{tree}"),
        },
        "implementation_files": implementation,
        "output_namespace": plan["outputs"]["namespace"],
        "resource_envelope": plan["resource_envelope"],
        "forbidden_actions": [
            "training, fitting, tuning, weight or checkpoint changes",
            "database writes",
            "protected partition reads",
            "source pool 2eb04f54 access",
            "old frozen record mutation or reinterpretation",
            "required-input substitution or neutral fallback",
        ],
        "execution_count": 1,
        "automatic_resume_or_extension": False,
    }
    sealed = write_sealed_json(
        AUTHORIZATION_PATH,
        payload,
        identity_field="authorization_identity",
    )
    print(sealed["authorization_identity"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "authorization"))
    args = parser.parse_args()
    if args.mode == "plan":
        _freeze_plan()
    else:
        _freeze_authorization()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
