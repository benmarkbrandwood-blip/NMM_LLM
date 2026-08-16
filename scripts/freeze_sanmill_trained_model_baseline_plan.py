#!/usr/bin/env python3
"""Freeze the trained-model versus safe-random gameplay protocol."""

# ruff: noqa: E402

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    write_sealed_json,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import load_pool, sha256_file
from learned_ai.evaluation.sanmill_trained_model_baseline import (
    ARMS,
    PLAN_SCHEMA,
    build_schedule,
    formal_states,
)


PLAN_PATH = _ROOT / "docs/experiments/sanmill-trained-model-baseline-v1.json"
POOL_PATH = (
    _ROOT / "docs/experiments/sanmill-safe-guidance-gameplay-start-pool-v1.json"
)
BASELINE_PATH = (
    _ROOT
    / "docs/evidence/sanmill-safe-guidance-gameplay-attempt-002-"
    "manifest-2026-08-16.json"
)
EXCLUDED_START = "00092c974cabf05874f066b8948e791f9fdc82d84a65759da1ba78f212a643b0"
SANMILL_RUNTIME_IDENTITY = (
    "705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea"
)
MALOM_CONTENT_SHA256 = (
    "c414fe88778f8d1d95cd3015532b43cad59f09e8398d8e46c42188b6829f3544"
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


def _file(path: str) -> dict[str, Any]:
    target = _ROOT / path
    if not target.is_file():
        raise RuntimeError(f"required candidate file is absent: {path}")
    return {
        "path": path,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def _baseline() -> tuple[dict[str, Any], str]:
    value = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    identity = value.get("result_identity")
    body = dict(value)
    body.pop("result_identity", None)
    if canonical_sha256(body) != identity:
        raise RuntimeError("attempt-002 result identity differs")
    arm = value["analysis"]["by_arm"]["random-safe"]
    strict = arm["strict_wdl"]
    if (
        value["analysis"]["completed_starts"] != 254
        or arm["games"] != 508
        or strict
        != {
            "wins": 21,
            "draws": 414,
            "losses": 73,
            "score_rate": 0.44881889763779526,
        }
    ):
        raise RuntimeError("attempt-002 safe-random baseline differs")
    return value, sha256_file(BASELINE_PATH)


def main() -> int:
    if PLAN_PATH.exists():
        raise SystemExit("plan already exists")
    if _git("branch", "--show-current") != "dev":
        raise SystemExit("plan freeze requires dev")

    pool, pool_sha = load_pool(POOL_PATH)
    states = formal_states(pool, excluded_start_ids=[EXCLUDED_START])
    state_ids = sorted(str(row["state_id"]) for row in states)
    membership_identity = canonical_sha256(state_ids)
    if membership_identity != (
        "610f62e74b4a70500adfcaa3e0c19769dd178b480ef9765115ae6ab9a5af13d2"
    ):
        raise RuntimeError("attempt-002 formal membership differs")
    schedule = build_schedule(
        states,
        namespace="sanmill-trained-model-baseline-v1-formal-game",
    )
    baseline, baseline_sha = _baseline()

    route_bundle = (
        "learned_ai/checkpoints/evaluation/"
        "sanmill-retained-v3-v4-phase-process-generalization-v1/inputs/"
        "v4-route-bundle"
    )
    v4_bundle = json.loads(
        (_ROOT / route_bundle / "bundle.json").read_text(encoding="utf-8")
    )
    if v4_bundle["bundle_identity"] != (
        "817d2e36fbd0b614c5c48737ee987f684b99eb6ff697591618123ec7307a2d0f"
    ):
        raise RuntimeError("retained-v4 route bundle differs")
    if sha256_file(_ROOT / route_bundle / "bundle.json") != (
        "e118590aebc60dd858ae4a2600937f8153af4e0b2431781f5d2295bb85b65000"
    ):
        raise RuntimeError("retained-v4 route manifest differs")

    v4_checkpoint = _file(
        "learned_ai/checkpoints/scaffolded/"
        "s_gen_v2_sanmill_refereed/"
        "managed-sanmill-no-refresh-retained-v4-seed70-attempt-003/"
        "segments/segment-0020/latest.pt"
    )
    if v4_checkpoint["sha256"] != (
        "295b268e697255908f9c7517f4697ca251a10ec0f13d922cbcbab2260fb5105d"
    ):
        raise RuntimeError("retained-v4 checkpoint differs")
    v4_specialist_db = _file(
        "learned_ai/checkpoints/evaluation/"
        "sanmill-retained-v3-v4-phase-process-generalization-v1/inputs/"
        "v4-specialist-db-snapshot.sqlite"
    )
    if v4_specialist_db["sha256"] != (
        "3d69d1acb007dbd26a48ae1c6acec4bb29f905ffedd21c816ad1771a6cf942ed"
    ):
        raise RuntimeError("retained-v4 SpecialistDB differs")

    specialist_files = {
        "checkpoint_open": _file(
            "learned_ai/checkpoints/scaffolded/s_open_v2/best.pt"
        ),
        "checkpoint_mid": _file(
            "learned_ai/checkpoints/scaffolded/s_mid_v2/best.pt"
        ),
        "checkpoint_end": _file(
            "learned_ai/checkpoints/scaffolded/s_end_v2/best.pt"
        ),
        "sentinel_checkpoint": _file(
            "learned_ai/sentinel/checkpoints/best.pt"
        ),
        "value_place": _file("data/value_net_phase_place.npz"),
        "value_move": _file("data/value_net_phase_move.npz"),
        "value_fly": _file("data/value_net_phase_fly.npz"),
        "gap_net": _file("data/gap_net.npz"),
        "human_db": _file("data/human_db.sqlite"),
    }
    expected_specialist_hashes = {
        "checkpoint_open": (
            "d020e1442676e16cdced6c91dac958817c3a22a283cc293d6e19930a87703701"
        ),
        "checkpoint_mid": (
            "a587ab995224a1d43c99fd2f42e4bff9c060ac6da55edcddb43a39fc07ef26d2"
        ),
        "checkpoint_end": (
            "5de51a1afd5794374d4394cce2950957a23f02504b5c5952a062d91414b94be8"
        ),
    }
    if any(
        specialist_files[name]["sha256"] != digest
        for name, digest in expected_specialist_hashes.items()
    ):
        raise RuntimeError("active specialist checkpoint lineage differs")
    if (_ROOT / "data/specialist_db.sqlite").exists():
        raise RuntimeError("product SpecialistDB is no longer absent")
    specialist_runtime_identity = canonical_sha256(
        {
            "resource_files": specialist_files,
            "product_specialist_db": {
                "path": "data/specialist_db.sqlite",
                "expected": "absent",
            },
            "ply_depth": 12,
            "presearch": "none-successful-specialist-score-path-does-not-read-gameai",
        }
    )

    maximum_half_width = 0.015
    conservative_sd = 0.12
    payload = {
        "schema_version": PLAN_SCHEMA,
        "status": "frozen_before_rehearsal_or_candidate_outcomes",
        "experiment_id": "sanmill-trained-model-baseline-v1-20260816",
        "frozen_at_utc": "2026-08-16T00:00:00Z",
        "repository_base_commit": _git("rev-parse", "HEAD"),
        "question": (
            "How do the exact retained-v4 training route and active specialist "
            "route score against the already measured zero-training uniform "
            "A_pos policy on the same 254 starts and pinned Sanmill runtime?"
        ),
        "experiment": {
            "arms": list(ARMS),
            "starts": 254,
            "colors_per_start": 2,
            "games_per_arm": 508,
            "formal_games": len(schedule),
            "primary_node_budget": 100_000,
            "maximum_post_start_logical_plies": 1536,
            "safety_cap_disposition": "incomplete-never-a-draw",
            "schedule": (
                "frozen pool order, W then B, then the frozen four-arm order"
            ),
            "retained_v3_excluded": (
                "five formal arms would consume all 2540 authorized games and "
                "leave no capacity for the mandatory per-arm rehearsal"
            ),
        },
        "baseline": {
            "kind": "zero-training uniform-random A_pos",
            "result_path": str(BASELINE_PATH.relative_to(_ROOT)).replace("\\", "/"),
            "result_identity": baseline["result_identity"],
            "result_file_sha256": baseline_sha,
            "starts": 254,
            "games": 508,
            "strict_wdl": {"wins": 21, "draws": 414, "losses": 73},
            "random_safe_score_rate": 0.44881889763779526,
            "known_before_candidate_measurement": True,
        },
        "start_pool": {
            "path": str(POOL_PATH.relative_to(_ROOT)).replace("\\", "/"),
            "pool_identity": pool["pool_identity"],
            "pool_file_sha256": pool_sha,
            "original_membership_identity": pool["state_membership_identity"],
            "excluded_start_ids": [EXCLUDED_START],
            "formal_membership_identity": membership_identity,
            "formal_start_ids_identity": canonical_sha256(state_ids),
            "selection_was_candidate_model_independent": True,
            "selection_used_candidate_predictions": False,
            "baseline_outcomes_are_already_known": True,
            "newly_blind_pool": False,
            "reuse_boundary": (
                "model-blind frozen reuse, not a new held-out or population sample"
            ),
            "exposure_audit_required_before_measurement": True,
            "source_game_exposure_may_be_unidentifiable": True,
        },
        "candidate_runtime": {
            "retained_v4": {
                "candidate_id": "retained-v4",
                "bundle": {
                    "path": route_bundle,
                    "identity": v4_bundle["bundle_identity"],
                    "manifest_sha256": sha256_file(
                        _ROOT / route_bundle / "bundle.json"
                    ),
                },
                "checkpoint": {
                    **v4_checkpoint,
                    "payload_sha256": (
                        "ed7932bc7c11b1aa41274ea0de7bd08902812b1188ca4739b6d0d8dc15e46727"
                    ),
                },
                "human_db": {
                    "path": "data/human_db.sqlite",
                    "identity": v4_bundle["resources"]["human_db"]["identity"],
                    "file_sha256": specialist_files["human_db"]["sha256"],
                    "read_only": True,
                },
                "specialist_db": {
                    **v4_specialist_db,
                    "identity": v4_bundle["resources"]["specialist_db"]["identity"],
                    "label_version": "sector-corrected-v1",
                    "read_only": True,
                },
                "route_name": "s-gen-v2-training-aligned-v1",
                "natural_free_argmax": True,
                "a_pos_arm": (
                    "same full-route logits, final argmax restricted to runtime A_pos"
                ),
                "malom_use": "lookahead terminal early exit and A_pos diagnostic",
                "product_generalist_agent_forbidden": True,
            },
            "active_specialists": {
                "candidate_id": "active-specialists",
                "checkpoint_root": "learned_ai/checkpoints/scaffolded",
                "resource_files": specialist_files,
                "runtime_identity": specialist_runtime_identity,
                "phase_value_net_base": "data/value_net_phase",
                "product_specialist_db": {
                    "path": "data/specialist_db.sqlite",
                    "expected": "absent",
                },
                "ply_depth": 12,
                "presearch": "none-successful-specialist-score-path-does-not-read-gameai",
                "product_presearch_deviation": {
                    "product": "30/60-second alpha-beta then specialist override",
                    "evaluation": "no alpha-beta presearch; strict specialist argmax",
                    "reason": (
                        "SpecialistRouter.set_gameai only stores _gameai and the "
                        "score path never reads it"
                    ),
                    "bias_if_specialist_succeeds": "none identified",
                    "failure_difference": (
                        "evaluation fails closed; product silently falls back to search"
                    ),
                },
                "specialist_db_route": "absent exactly as current product startup",
                "malom_use": "none in inference route; A_pos diagnostic/control only",
                "lineage": "untraceable beyond the three frozen checkpoint payloads",
                "a_pos_arm": (
                    "same full-route probabilities, final argmax restricted to A_pos"
                ),
            },
        },
        "primary_estimand": {
            "for_each_arm": "trained arm minus attempt-002 random-safe",
            "independent_unit": "frozen start",
            "within_start": "average candidate White and Black strict W/D/L scores",
            "pairing": "same start and colors against persisted random-safe results",
            "interval": "mean plus/minus 1.96 start-level sample-SE",
            "population_inference": False,
        },
        "precision_preregistration": {
            "support": 254,
            "conservative_start_difference_sd": conservative_sd,
            "expected_95_half_width": (
                1.96 * conservative_sd / math.sqrt(254)
            ),
            "maximum_95_half_width": maximum_half_width,
            "prior_attempt_002_full_minus_random_half_width": (
                baseline["analysis"][
                    "primary_full_minus_random_start_clustered_score"
                ]["half_width"]
            ),
            "analytic_interval_no_bootstrap_repetitions": True,
            "sample_size_frozen_before_candidate_outcomes": True,
        },
        "primary_decision": {
            "maximum_95_half_width": maximum_half_width,
            "trained_higher": "paired interval lower bound > 0",
            "safe_random_higher": "paired interval upper bound < 0",
            "no_directional_decision": (
                "interval contains zero and half width <= 0.015"
            ),
            "inconclusive_precision": "half width > 0.015",
            "all_games_must_be_strict_rules_terminal": True,
            "equivalence_claim": False,
            "no_result_based_early_stop_or_extension": True,
        },
        "secondary_metrics": {
            "cannot_flip_primary": True,
            "items": [
                "candidate self-downgrades per candidate turn, split W-to-D, W-to-L, D-to-L",
                "strict W/D/L and terminal reasons by arm and frozen source phase",
                "rule-draw share",
                "A_pos-constrained minus free within each candidate",
                "each A_pos-constrained arm minus prior full-guided A_pos arm",
                "exact-start exposure strata where independently recoverable",
            ],
        },
        "information_asymmetry": {
            "random_safe_has_A_pos_guarantee": True,
            "free_arms_have_A_pos_guarantee": False,
            "constrained_arms_remove_action_safety_asymmetry": True,
            "retained_v4_lookahead_uses_malom_terminal_early_exit": True,
            "interpretation": "trained model versus Malom-safe random",
            "training_value_claim_from_free_arm_alone": False,
        },
        "sanmill_contract": {
            **baseline["sanmill_runtime"],
            "runtime_identity": SANMILL_RUNTIME_IDENTITY,
            "seed": 42,
            "primary_node_budget": 100_000,
            "protocol_timeout_seconds": 10.0,
            "search_timeout_seconds": 120.0,
            "strict_referee_profile": "mif-stable-moving-v1",
            "strict_referee_semantic_digest": (
                "sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94"
            ),
        },
        "malom_contract": {
            "trust_level": "sector-corrected-v1",
            "content_sha256": MALOM_CONTENT_SHA256,
            "safe_set": "A_pos",
            "history_aware": False,
            "A_allow_claim": False,
        },
        "rehearsal": {
            "formal_result_eligibility": False,
            "live_games_per_arm": 2,
            "live_games": 8,
            "scripted_terminal_contract_games": 2,
            "total_complete_games": 10,
            "live_start": {
                "source": (
                    "docs/experiments/"
                    "sanmill-retained-v3-v4-phase-process-corpus-v1.json"
                ),
                "source_file_sha256": (
                    "8353ff3e52465bf99f7cf468a9cbcb4681a673ac2cebcdae00c253df8a22670b"
                ),
                "source_corpus_identity": (
                    "3be3d76c34511e0f78d0f5bfe4a338c415c393306a955538bb85823e9d62c080"
                ),
                "source_start_id": "phase-process-001",
                "source_record_identity": (
                    "a1c9285edd203f9ff7e911365aa583e449550e314509592d0b1ed33ffc8f6063"
                ),
                "strict_history_sha256": (
                    "5150ab6b9b21fe6537df06a628a191754d06b5b562936d2f7207d63ddcabfe28"
                ),
                "outside_formal_pool_required": True,
                "candidate_colors": ["W", "B"],
            },
            "scripted_decisive_case": {
                "source_ledger": (
                    "learned_ai/checkpoints/evaluation/"
                    "sanmill-retained-v3-v4-phase-process-generalization-v1/"
                    "games.jsonl"
                ),
                "source_ledger_sha256": (
                    "45506e5cedf5ab9bdcba9dd687349869b639fb8bd46fd8990cbaf4bb79ef3211"
                ),
                "source_game_id": (
                    "phase-process-game:"
                    "c443a38ef4122889ace1df3b3a2dc84520a5624f00f77775881a891ab4970f82"
                ),
                "source_record_sha256": (
                    "140c38d90969b203e21119b7f38fcefe99b48c80ec4bf1583139a6077cc959e4"
                ),
            },
            "scripted_threefold_case": {
                "history_actions_identity": (
                    "20f167e886c512f202fc6b166ad07d7f714601d3b0e45e2e1f5bc3afa836e542"
                ),
                "fen": "W.BW.B.BBWWB.W.W.BWWBBWB|B|9|9",
                "terminal_action": "g7-d7",
            },
            "required_coverage": {
                "each_arm_live_complete_games": 2,
                "strict_draw_path": True,
                "strict_decisive_path": True,
                "result_packaging": True,
                "resource_checkpoint_before_game_record": True,
                "completion_and_analysis": True,
            },
            "output_namespace": (
                "out/evaluation/sanmill-trained-model-baseline-v1-"
                "rehearsal-20260816-001"
            ),
            "tracked_result": (
                "docs/evidence/sanmill-trained-model-baseline-v1-"
                "rehearsal-2026-08-16.json"
            ),
        },
        "preflight": {
            "zero_formal_games": True,
            "sanmill_determinism_fixtures_per_phase": 1,
            "sanmill_determinism_budgets": [100_000],
            "candidate_determinism_fixtures_per_phase": 1,
            "same_process_repeat_and_opposite_candidate_order": True,
            "all_254_starts_strictly_replayed": True,
            "protected_guard_true_failure_test": True,
            "candidate_identity_and_route_canaries": True,
        },
        "outputs": {
            "authorization": (
                "docs/experiments/sanmill-trained-model-baseline-v1/"
                "authorization.json"
            ),
            "rehearsal_result": (
                "docs/evidence/sanmill-trained-model-baseline-v1-"
                "rehearsal-2026-08-16.json"
            ),
            "preflight_result": (
                "docs/evidence/sanmill-trained-model-baseline-v1-"
                "preflight-2026-08-16.json"
            ),
            "formal_output_namespace": (
                "out/evaluation/sanmill-trained-model-baseline-v1-"
                "20260816-001"
            ),
            "formal_result": (
                "docs/evidence/sanmill-trained-model-baseline-v1-"
                "manifest-2026-08-16.json"
            ),
            "evidence_document": (
                "docs/evidence/sanmill-trained-model-baseline-v1-2026-08-16.md"
            ),
        },
        "resource_envelope": {
            "maximum_reused_formal_starts": 254,
            "maximum_complete_games": 2_540,
            "maximum_engine_single_step_searches": 120_000,
            "maximum_malom_queries": 25_000_000,
            "maximum_active_seconds": 21_600,
            "maximum_concurrent_evaluators": 1,
            "maximum_concurrent_sanmill_processes": 1,
            "maximum_training_updates": 0,
            "maximum_database_writes": 0,
            "planned_rehearsal_games": 10,
            "planned_formal_games": len(schedule),
            "planned_total_complete_games": 10 + len(schedule),
            "stop_at_any_limit": True,
            "automatic_retry_resume_batching_or_extension": False,
            "host_interruption_recovery_authorized": False,
        },
        "protected_access": {
            "official_selection_confirmation_final_test": "unopened",
            "research_confirmation": "unopened",
            "source_pool_2eb04f54_remaining_108": "unread_and_unconsumed",
        },
        "claim_boundary": {
            "fixed_runtime_start_pool_and_candidate_routes_only": True,
            "positional_only": True,
            "playing_strength_population_claim": False,
            "human_trap_or_product_user_claim": False,
            "training_value_claim": False,
            "promotion_deployment_publication_release_or_training": False,
            "other_engine_or_runtime_transport": False,
            "equivalence_claim": False,
        },
    }
    sealed = write_sealed_json(PLAN_PATH, payload, identity_field="plan_identity")
    print(sealed["plan_identity"])
    print(membership_identity)
    print(specialist_runtime_identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
