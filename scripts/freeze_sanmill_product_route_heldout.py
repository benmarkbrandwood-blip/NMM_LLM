#!/usr/bin/env python3
"""Freeze the authorized 108-start product-route held-out protocol."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.human_f0h0_feasibility import (  # noqa: E402
    canonical_sha256,
    write_sealed_json,
)
from learned_ai.evaluation.sanmill_classical_positional_safety_strength import (  # noqa: E402
    _canonical_weights,
)
from learned_ai.evaluation.sanmill_product_route_heldout import (  # noqa: E402
    ARMS,
    AUTHORIZATION_SCHEMA,
    CONSUMED_PREFIX_IDENTITY,
    EXPECTED_GAMES,
    EXPECTED_STARTS,
    MATERIAL_LOWER_BOUND,
    MAXIMUM_HALF_WIDTH,
    PLAN_SCHEMA,
    POOL_IDENTITY,
    POOL_RECORDS_IDENTITY,
    build_schedule,
    membership_only_suffix,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (  # noqa: E402
    sha256_file,
)


PLAN_PATH = ROOT / "docs/experiments/sanmill-product-route-heldout-v1.json"
FREEZE_AUDIT_PATH = ROOT / (
    "docs/experiments/sanmill-product-route-heldout-v1/freeze-audit.json"
)
AUTHORIZATION_PATH = ROOT / (
    "docs/experiments/sanmill-product-route-heldout-v1/authorization.json"
)
POOL_PATH = ROOT / (
    "docs/experiments/sanmill-retained-v3-v4-late-import-heldout-pool-v1.json"
)
PRIOR_CURRENT_DEV_PLAN = ROOT / (
    "docs/experiments/sanmill-classical-positional-safety-strength-v1.json"
)
FREEZE_AUDIT_SCHEMA = "nmm.sanmill-product-route-heldout-freeze-audit.v1"


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _file_record(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _product_contract(prior: dict[str, Any], head: str, tree: str) -> dict[str, Any]:
    import nmm_core

    native_module = getattr(nmm_core, "nmm_core", None)
    native_path = Path(str(getattr(native_module, "__file__", "")))
    if not native_path.is_file():
        raise RuntimeError("active dev native extension is unavailable")
    evolved = json.loads((ROOT / "data/weights/best.json").read_text(encoding="utf-8"))
    weights = _canonical_weights(evolved)
    product = copy.deepcopy(prior["product_contract"])
    product.update(
        {
            "route": "current dev GameAI classical coordinator used by both product routes",
            "source_commit": head,
            "source_tree": tree,
            "implementation_sha256": {
                "game_ai": sha256_file(ROOT / "ai/game_ai.py"),
                "heuristics": sha256_file(ROOT / "ai/heuristics.py"),
                "native_extension": sha256_file(native_path),
            },
            "filter_implementation_sha256": {
                "web_app": sha256_file(ROOT / "web/app.py"),
                "positional_safety": sha256_file(
                    ROOT / "learned_ai/agents/positional_safety.py"
                ),
                "malom_runtime": sha256_file(ROOT / "ai/malom_runtime.py"),
            },
            "resolved_weights": dict(weights.__dict__),
            "deterministic_search_threads": 1,
            "max_depth": 14,
            "fresh_ai_per_game": True,
            "rust_tt_within_game": True,
            "sanmill_seed": 42,
            "product_malom_adapter_runtime_only": True,
            "actual_delivery": {
                "specialist_first": (
                    "classical coordinator then SpecialistRouter override then final "
                    "ProductPositionalSafetyGate"
                ),
                "classical_first": (
                    "classical coordinator then final ProductPositionalSafetyGate"
                ),
                "final_product_choke": "web.app._finalize_product_ai_move",
                "final_gate": (
                    "learned_ai.agents.positional_safety."
                    "ProductPositionalSafetyGate"
                ),
                "specialist_override": (
                    "learned_ai.agents.specialist_router.SpecialistRouter.score_moves"
                ),
                "position_only": True,
                "history_aware": False,
            },
        }
    )
    return product


def _specialist_contract() -> dict[str, Any]:
    resource_files = {
        "human_db": _file_record("data/human_db.sqlite"),
        "sentinel_checkpoint": _file_record(
            "learned_ai/sentinel/checkpoints/best.pt"
        ),
        "open_checkpoint": _file_record(
            "learned_ai/checkpoints/scaffolded/s_open_v2/best.pt"
        ),
        "mid_checkpoint": _file_record(
            "learned_ai/checkpoints/scaffolded/s_mid_v2/best.pt"
        ),
        "end_checkpoint": _file_record(
            "learned_ai/checkpoints/scaffolded/s_end_v2/best.pt"
        ),
    }
    specialist_path = ROOT / "data/specialist_db.sqlite"
    if specialist_path.exists():
        specialist_db = {
            **_file_record("data/specialist_db.sqlite"),
            "expected": "present-read-only",
        }
    else:
        specialist_db = {
            "path": "data/specialist_db.sqlite",
            "expected": "absent",
        }
    identity_body = {
        "resource_files": resource_files,
        "specialist_db": specialist_db,
        "checkpoint_root": "learned_ai/checkpoints/scaffolded",
        "ply_depth": 12,
        "feature_width": 134,
        "route_source_sha256": {
            "specialist_router": sha256_file(
                ROOT / "learned_ai/agents/specialist_router.py"
            ),
            "scaffolded_encoder": sha256_file(
                ROOT / "learned_ai/models/scaffolded_encoder.py"
            ),
            "web_app": sha256_file(ROOT / "web/app.py"),
        },
        "human_db_mode": "read-only immutable measurement equivalent",
        "runtime_quarantine": "disabled-no-inference-effect-no-writes",
    }
    return {
        **identity_body,
        "identity_body": identity_body,
        "runtime_identity": canonical_sha256(identity_body),
    }


def _self_consistency(plan: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "four_distinct_arms": len(plan["arms"]) == 4
        and tuple(row["arm"] for row in plan["arms"]) == ARMS,
        "planned_games_equal_authorized_maximum": (
            plan["resource_envelope"]["planned_complete_games"]
            == EXPECTED_GAMES
            == plan["resource_envelope"]["maximum_complete_games"]
        ),
        "membership_only_before_freeze": plan["source_pool"][
            "pre_freeze_access"
        ]
        == "ID-and-record-identity membership only",
        "content_only_after_freeze": plan["source_pool"]["content_open_rule"]
        == "after plan, schedule, thresholds, and authorization are frozen",
        "no_retry_resume_or_extension": not any(
            plan["execution_policy"][key]
            for key in ("retry", "resume", "recovery", "extension")
        ),
        "results_hidden_until_all_arms_complete": plan["execution_policy"][
            "intermediate_route_result_analysis"
        ]
        is False,
        "primary_thresholds_exact": (
            plan["primary"]["maximum_95_half_width"] == MAXIMUM_HALF_WIDTH
            and plan["primary"]["material_lower_bound"] == MATERIAL_LOWER_BOUND
        ),
        "position_only_not_history_aware": plan["malom_contract"]["safe_set"]
        == "A_pos"
        and plan["malom_contract"]["history_aware"] is False
        and plan["malom_contract"]["A_allow_claim"] is False,
        "no_automatic_product_change": plan["claim_boundary"][
            "automatic_default_route_change"
        ]
        is False,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "conflicts": [] if all(checks.values()) else [
            key for key, passed in checks.items() if not passed
        ],
    }


def build_plan() -> dict[str, Any]:
    if _git("branch", "--show-current") != "dev":
        raise RuntimeError("plan freeze requires dev")
    if _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("plan freeze requires a clean tracked tree")
    head = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    if _git("rev-parse", "origin/dev") != head:
        raise RuntimeError("implementation commit must be published before freeze")

    corpus = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    membership = membership_only_suffix(corpus)
    prior = json.loads(PRIOR_CURRENT_DEV_PLAN.read_text(encoding="utf-8"))
    old_48 = list(prior["start_subset"]["state_ids"])
    if set(old_48) & {row["start_id"] for row in membership}:
        raise RuntimeError("old 48 development starts overlap held-out suffix")
    schedule_namespace = "sanmill-product-route-heldout-v1-20260821-001"
    schedule = build_schedule(membership, namespace=schedule_namespace)
    product = _product_contract(prior, head, tree)
    specialist = _specialist_contract()
    malom_manifest = ROOT / "data/manifests/malom-sector-corrected-v1.json"
    malom = json.loads(malom_manifest.read_text(encoding="utf-8"))

    implementation_paths = (
        "learned_ai/evaluation/sanmill_product_route_heldout.py",
        "scripts/freeze_sanmill_product_route_heldout.py",
        "scripts/run_sanmill_product_route_heldout.py",
        "scripts/recompute_sanmill_product_route_heldout.py",
    )
    body: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": "sanmill-product-route-heldout-v1-20260821-001",
        "status": "frozen_authorized_once",
        "frozen_at_utc": "2026-08-21T00:00:00Z",
        "objective": (
            "On the final 108 never-consumed starts, compare the current dev "
            "specialist-first and classical-first human-vs-AI product routes, "
            "holding the final position-only A_pos gate fixed."
        ),
        "implementation_commit": head,
        "implementation_tree": tree,
        "implementation_files": {
            path: sha256_file(ROOT / path) for path in implementation_paths
        },
        "source_pool": {
            "path": str(POOL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "file_sha256": sha256_file(POOL_PATH),
            "pool_identity": POOL_IDENTITY,
            "records_identity": POOL_RECORDS_IDENTITY,
            "records_total": 361,
            "consumed_prefix_records": 253,
            "consumed_prefix_identity": CONSUMED_PREFIX_IDENTITY,
            "suffix_records": EXPECTED_STARTS,
            "suffix_start_ids": [row["start_id"] for row in membership],
            "suffix_record_identities": [
                row["record_identity"] for row in membership
            ],
            "suffix_start_ids_identity": canonical_sha256(
                [row["start_id"] for row in membership]
            ),
            "suffix_record_identities_identity": canonical_sha256(
                [row["record_identity"] for row in membership]
            ),
            "suffix_membership_identity": canonical_sha256(membership),
            "order": "existing frozen master order, records 254 through 361",
            "replacement_or_reordering": False,
            "old_48_development_start_ids_identity": canonical_sha256(old_48),
            "old_48_overlap": 0,
            "pre_freeze_access": "ID-and-record-identity membership only",
            "content_open_rule": (
                "after plan, schedule, thresholds, and authorization are frozen"
            ),
        },
        "arms": [
            {
                "arm": arm,
                "difficulty": int(arm[1 : arm.index("-")]),
                "route": (
                    "specialist-first" if "specialist-first" in arm else "classical-first"
                ),
                "final_gate": "same delivered position-only A_pos",
                "starts": EXPECTED_STARTS,
                "colors_per_start": 2,
                "games": EXPECTED_STARTS * 2,
            }
            for arm in ARMS
        ],
        "schedule": {
            "namespace": schedule_namespace,
            "order": (
                "master start order; difficulty 9 then 10; candidate W then B; "
                "specialist-first then classical-first"
            ),
            "records": len(schedule),
            "identity": canonical_sha256(schedule),
            "result_contingent_changes": False,
        },
        "product_contract": product,
        "specialist_contract": specialist,
        "sanmill_contract": prior["sanmill_contract"],
        "malom_contract": {
            "manifest_path": str(malom_manifest.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "manifest_file_sha256": sha256_file(malom_manifest),
            "manifest_sha256": sha256_file(malom_manifest),
            "content_sha256": malom["content_sha256"],
            "dataset_id": malom["dataset_id"],
            "trust_level": "sector-corrected-v1",
            "label_version": "sector-corrected-v1",
            "safe_set": "A_pos",
            "history_aware": False,
            "A_allow_claim": False,
            "read_only": True,
        },
        "preflight_canary": {
            "pool_path": prior["start_pool"]["path"],
            "pool_file_sha256": prior["start_pool"]["file_sha256"],
            "pool_identity": prior["start_pool"]["pool_identity"],
            "state_id": old_48[0],
            "main_sample": False,
            "complete_games": 0,
            "required_routes": ["specialist-first", "classical-first"],
        },
        "primary": {
            "estimands": {
                "difficulty_9": "classical-first minus specialist-first strict WDL score",
                "difficulty_10": "classical-first minus specialist-first strict WDL score",
            },
            "independent_unit": "one start, average both candidate colors",
            "interval": "normal 95 percent interval over 108 start-level differences",
            "maximum_95_half_width": MAXIMUM_HALF_WIDTH,
            "material_lower_bound": MATERIAL_LOWER_BOUND,
            "decisions": {
                "candidate": (
                    "half-width <= 0.04 and lower bound >= 0.05 => "
                    "classical_first_material_route_candidate"
                ),
                "no_change": (
                    "precision adequate and lower bound < 0.05 => "
                    "no_classical_first_route_change_supported"
                ),
                "specialist_higher_note": (
                    "upper bound < 0 => specialist_first_higher"
                ),
                "precision_stop": (
                    "half-width > 0.04 => precision_inadequate_stop"
                ),
            },
            "difficulty_decisions_separate": True,
        },
        "precision_basis": {
            "old_48_start_paired_half_width": 0.04785,
            "sqrt_scaled_to_108": 0.0319,
            "engineering_maximum": MAXIMUM_HALF_WIDTH,
            "old_48_results_in_main_sample": False,
        },
        "secondary_metrics": [
            "absolute strict WDL",
            "start and action phase",
            "terminal reasons",
            "specialist coverage and fallback",
            "final A_pos intervention and failure",
            "internal Malom or product database bypass",
            "per-step work",
            "real route latency",
        ],
        "resource_envelope": {
            "planned_complete_games": EXPECTED_GAMES,
            "maximum_complete_games": EXPECTED_GAMES,
            "maximum_active_seconds": 18_000,
            "maximum_parallel_measurement_processes": 1,
            "maximum_parallel_sanmill_processes": 1,
            "engine_search_anomaly_ceiling": 5_000_000,
            "malom_query_anomaly_ceiling": 50_000_000,
            "training_updates": 0,
            "model_fits": 0,
            "database_writes": 0,
        },
        "execution_policy": {
            "one_shot": True,
            "retry": False,
            "resume": False,
            "recovery": False,
            "extension": False,
            "early_stop_on_results": False,
            "intermediate_route_result_analysis": False,
            "all_arms_before_analysis": True,
            "fresh_product_ai_per_game": True,
            "within_game_rust_tt_retained": True,
        },
        "protected_access": {
            "official_selection": "unopened",
            "official_confirmation": "unopened",
            "official_final_test": "unopened",
            "research_confirmation": "unopened",
            "consumed_253_used_as_new_evidence": False,
            "source_pool_outside_suffix": "not a main sample",
        },
        "outputs": {
            "namespace": "out/evaluation/sanmill-product-route-heldout-v1-20260821-001",
            "games": (
                "out/evaluation/sanmill-product-route-heldout-v1-20260821-001/"
                "games.jsonl"
            ),
            "preflight": (
                "out/evaluation/sanmill-product-route-heldout-v1-20260821-001/"
                "preflight.json"
            ),
            "completion": (
                "out/evaluation/sanmill-product-route-heldout-v1-20260821-001/"
                "completion.json"
            ),
            "result": (
                "docs/evidence/sanmill-product-route-heldout-v1-manifest-"
                "2026-08-21.json"
            ),
            "independent_recompute": (
                "docs/evidence/sanmill-product-route-heldout-v1-independent-"
                "recompute-2026-08-21.json"
            ),
            "freeze_audit": str(FREEZE_AUDIT_PATH.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "authorization": str(AUTHORIZATION_PATH.relative_to(ROOT)).replace(
                "\\", "/"
            ),
        },
        "claim_boundary": {
            "exact_current_dev_product_routes_only": True,
            "exact_108_start_heldout_suffix_only": True,
            "position_only_A_pos": True,
            "history_aware_safety_claim": False,
            "overall_strength_claim": False,
            "human_opponent_claim": False,
            "causal_mechanism_claim": False,
            "equivalence_claim": False,
            "automatic_default_route_change": False,
            "automatic_promotion_deployment_release": False,
        },
    }
    precheck = _self_consistency(body)
    if not precheck["passed"]:
        raise RuntimeError(f"pre-freeze contract conflict: {precheck['conflicts']}")
    body["contract_self_consistency"] = {
        "before_freeze": precheck,
        "after_freeze_audit_required": True,
    }
    return body


def run(_args: argparse.Namespace) -> int:
    for path in (PLAN_PATH, FREEZE_AUDIT_PATH, AUTHORIZATION_PATH):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite frozen output: {path}")
    plan = write_sealed_json(PLAN_PATH, build_plan(), identity_field="plan_identity")
    loaded = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    body = dict(loaded)
    identity = body.pop("plan_identity")
    if identity != plan["plan_identity"] or canonical_sha256(body) != identity:
        raise RuntimeError("post-freeze plan identity differs")
    postcheck = _self_consistency(loaded)
    if not postcheck["passed"]:
        raise RuntimeError(f"post-freeze contract conflict: {postcheck['conflicts']}")
    audit_body = {
        "schema_version": FREEZE_AUDIT_SCHEMA,
        "status": "post_freeze_contract_self_consistency_passed",
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": sha256_file(PLAN_PATH),
        "check": postcheck,
        "candidate_moves_read": 0,
        "candidate_results_read": 0,
        "suffix_content_fields_read": [],
        "membership_fields_read": ["start_id", "record_identity"],
    }
    audit = write_sealed_json(
        FREEZE_AUDIT_PATH, audit_body, identity_field="freeze_audit_identity"
    )
    authority_contract = {
        "source_thread_id": "01a02210-4b1a-7483-bb97-09cf02e7e7ce",
        "authority_type": "product-owner-direct",
        "one_shot": True,
        "objective": "final 108-start specialist-first versus classical-first product route comparison",
        "maximum_complete_games": EXPECTED_GAMES,
        "maximum_active_seconds": 18_000,
        "no_retry_resume_recovery_or_extension": True,
        "no_automatic_product_change_or_release": True,
    }
    authorization_body = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": "authorized_once_unconsumed",
        "operator": "product-owner-direct",
        "authority_reference": "codex_delegation in source thread",
        "authority_contract": authority_contract,
        "authority_contract_identity": canonical_sha256(authority_contract),
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": sha256_file(PLAN_PATH),
        "freeze_audit_identity": audit["freeze_audit_identity"],
        "freeze_audit_file_sha256": sha256_file(FREEZE_AUDIT_PATH),
        "implementation_commit": plan["implementation_commit"],
        "implementation_tree": plan["implementation_tree"],
        "output_namespace": plan["outputs"]["namespace"],
        "resource_envelope": plan["resource_envelope"],
        "retry_resume_recovery_or_extension": False,
    }
    authorization = write_sealed_json(
        AUTHORIZATION_PATH,
        authorization_body,
        identity_field="authorization_identity",
    )
    print(
        json.dumps(
            {
                "plan_identity": plan["plan_identity"],
                "freeze_audit_identity": audit["freeze_audit_identity"],
                "authorization_identity": authorization["authorization_identity"],
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser().parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
