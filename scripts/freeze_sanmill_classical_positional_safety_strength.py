#!/usr/bin/env python3
"""Freeze the paired current-dev classical ``A_pos`` strength protocol."""

# ruff: noqa: E402

from __future__ import annotations

import json
import math
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
from learned_ai.evaluation.sanmill_classical_positional_safety_strength import (
    AUTHORIZATION_SCHEMA,
    PLAN_SCHEMA,
    ClassicalPositionalSafetyStrengthError,
    _canonical_weights,
)
from learned_ai.evaluation.sanmill_classical_search_strength import (
    prior_scores_by_start,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    POOL_SCHEMA,
    load_sealed,
    sha256_file,
)


PLAN_PATH = Path(
    "docs/experiments/sanmill-classical-positional-safety-strength-v1.json"
)
AUTHORIZATION_PATH = Path(
    "docs/experiments/sanmill-classical-positional-safety-strength-v1/"
    "authorization.json"
)
V2_PLAN_PATH = Path("docs/experiments/sanmill-classical-search-strength-v2.json")
V2_RESULT_PATH = Path(
    "docs/evidence/sanmill-classical-search-strength-v2-manifest-identity-"
    "corrected-2026-08-18.json"
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


def _native_path() -> Path:
    import nmm_core

    native = getattr(nmm_core, "nmm_core", None)
    path = Path(str(getattr(native, "__file__", "")))
    if not path.is_file():
        raise ClassicalPositionalSafetyStrengthError(
            "active dev native extension is absent"
        )
    return path


def _resource_hashes() -> dict[str, str]:
    paths = {
        "evolved_weights": ROOT / "data/weights/best.json",
        "fullgame_db": ROOT / "data/endgame/fullgame.bin",
        "value_place": ROOT / "data/value_net_phase_place.npz",
        "value_move": ROOT / "data/value_net_phase_move.npz",
        "value_fly": ROOT / "data/value_net_phase_fly.npz",
        "gap_net": ROOT / "data/gap_net.npz",
    }
    values = {name: sha256_file(path) for name, path in paths.items()}
    for path in sorted((ROOT / "data/endgame").glob("*.wdl")):
        values[f"endgame/{path.name}"] = sha256_file(path)
    return values


def _implementation_hashes() -> dict[str, str]:
    paths = (
        "learned_ai/evaluation/"
        "sanmill_classical_positional_safety_strength.py",
        "scripts/freeze_sanmill_classical_positional_safety_strength.py",
        "scripts/run_sanmill_classical_positional_safety_strength.py",
        "tests/test_sanmill_classical_positional_safety_strength.py",
    )
    return {path: sha256_file(ROOT / path) for path in paths}


def _product_contract(v2_plan: dict[str, Any]) -> dict[str, Any]:
    evolved = json.loads(
        (ROOT / "data/weights/best.json").read_text(encoding="utf-8")
    )
    source_commit = _git("rev-parse", "HEAD")
    if source_commit != "776aa48095225149143abff8ea6a6965486f5229":
        raise ClassicalPositionalSafetyStrengthError(
            "product source is not the authorized dev commit 776aa48"
        )
    return {
        "source_commit": source_commit,
        "source_tree": _git("rev-parse", "HEAD^{tree}"),
        "route": "current dev GameAI classical coordinator",
        "actual_delivery": {
            "final_gate": "learned_ai.agents.positional_safety.ProductPositionalSafetyGate",
            "final_product_choke": "web.app._finalize_product_ai_move",
            "unsafe_move_replacement": "fixed-depth-2 restricted-root research over A_pos, canonical from/to/capture tie break",
            "root_move_restriction_inside_primary_search": False,
            "filter_scope": "difficulty 9/10 and explicit learned sources",
            "interactive_route_caveat": "human-vs-AI difficulty 9/10 still uses specialist override when specialist resources load; this experiment is the delivered classical fallback and AI-vs-AI classical route",
        },
        "implementation_sha256": {
            "game_ai": sha256_file(ROOT / "ai/game_ai.py"),
            "heuristics": sha256_file(ROOT / "ai/heuristics.py"),
            "native_extension": sha256_file(_native_path()),
        },
        "filter_implementation_sha256": {
            "web_app": sha256_file(ROOT / "web/app.py"),
            "positional_safety": sha256_file(
                ROOT / "learned_ai/agents/positional_safety.py"
            ),
            "malom_runtime": sha256_file(ROOT / "ai/malom_runtime.py"),
        },
        "resource_sha256": _resource_hashes(),
        "resolved_weights": dict(_canonical_weights(evolved).__dict__),
        "max_depth": int(v2_plan["product_contract"]["max_depth"]),
        "deterministic_search_threads": int(
            v2_plan["product_contract"]["deterministic_search_threads"]
        ),
        "extended_qsearch": True,
        "fullgame_db": "enabled-read-only",
        "endgame_solved_db": "enabled-read-only",
        "phase_value_net": "enabled-read-only-at-resolved-weight",
        "gap_net": "enabled-read-only",
        "malom_inside_classical_search": "enabled read-only through the validated product resolver; existing full-value fast path is held identical in both dev arms",
        "increment_isolated_by_primary_contrast": "only the final ProductPositionalSafetyGate is disabled versus enabled; internal GameAI Malom access is identical",
        "sanmill_seed": 42,
        "fresh_ai_per_game": True,
        "rust_tt_within_game": True,
    }


def _precision_design(
    *,
    v2_result: dict[str, Any],
    lightweight_result: dict[str, Any],
    start_ids: list[str],
) -> dict[str, Any]:
    prior = prior_scores_by_start(lightweight_result, start_ids=start_ids)
    specialist_differences = [
        prior["active-specialists-a-pos"][start_id]
        - prior["active-specialists-free"][start_id]
        for start_id in sorted(start_ids)
    ]
    specialist_sd = __import__("statistics").stdev(specialist_differences)
    specialist_rate = 732 / 9360
    rows: dict[str, Any] = {}
    for difficulty in (9, 10):
        arm = (
            f"classical-difficulty-{difficulty}-nodes-"
            f"{13887000 if difficulty == 9 else 18367000}"
        )
        observed = v2_result["analysis"]["by_arm"][arm]["self_downgrade"]
        rate = float(observed["event_rate"])
        projected_sd = specialist_sd * math.sqrt(rate / specialist_rate)
        projected_half_width = 1.96 * projected_sd / math.sqrt(len(start_ids))
        rows[str(difficulty)] = {
            "v2_original_self_downgrade_events": int(observed["events"]),
            "v2_candidate_turns": int(observed["candidate_turns"]),
            "v2_original_self_downgrade_rate": rate,
            "rate_scaled_projected_start_sd": projected_sd,
            "rate_scaled_projected_half_width": projected_half_width,
        }
    return {
        "basis": "same-48-start active-specialists A_pos-minus-free paired variance, scaled by sqrt of v2 classical versus specialist observed intervention opportunity rate before any current-dev outcome",
        "specialist_same_start_difference_sd": specialist_sd,
        "specialist_observed_self_downgrade_rate": specialist_rate,
        "by_difficulty": rows,
        "maximum_half_width": 0.045,
        "interpretation": "4.5pp leaves margin above the 3.78-3.92pp rate-scaled projections and is materially narrower than v2 cross-arm half-widths; it is a planning precision gate, not a guaranteed result",
    }


def main() -> int:
    if (ROOT / PLAN_PATH).exists() or (ROOT / AUTHORIZATION_PATH).exists():
        raise ClassicalPositionalSafetyStrengthError(
            "paired plan or authorization already exists"
        )
    v2_plan, v2_plan_sha = load_sealed(
        ROOT / V2_PLAN_PATH,
        schema="nmm.sanmill-classical-search-strength-plan.v1",
        identity_field="plan_identity",
    )
    v2_result, v2_result_sha = load_sealed(
        ROOT / V2_RESULT_PATH,
        schema="nmm.sanmill-classical-search-strength-result.v1",
        identity_field="result_identity",
    )
    if (
        v2_result["plan_identity"] != v2_plan["plan_identity"]
        or v2_result["formal_start_membership_identity"]
        != v2_plan["start_subset"]["membership_identity"]
    ):
        raise ClassicalPositionalSafetyStrengthError("v2 binding differs")
    lightweight = json.loads(
        (ROOT / LIGHTWEIGHT_RESULT_PATH).read_text(encoding="utf-8")
    )
    lightweight_sha = sha256_file(ROOT / LIGHTWEIGHT_RESULT_PATH)
    pool, pool_sha = load_sealed(
        ROOT / v2_plan["start_pool"]["path"],
        schema=POOL_SCHEMA,
        identity_field="pool_identity",
    )
    if (
        pool["pool_identity"] != v2_plan["start_pool"]["pool_identity"]
        or pool_sha != v2_plan["start_pool"]["file_sha256"]
    ):
        raise ClassicalPositionalSafetyStrengthError("start pool differs")
    start_ids = list(v2_plan["start_subset"]["state_ids"])
    if canonical_sha256(start_ids) != v2_plan["start_subset"]["membership_identity"]:
        raise ClassicalPositionalSafetyStrengthError("start membership differs")

    raw_classical = ROOT / v2_result["machine_records"]["raw_classical_ledger"]
    raw_reproduction = ROOT / v2_result["machine_records"][
        "raw_reproduction_ledger"
    ]
    if not raw_classical.is_file() or not raw_reproduction.is_file():
        raise ClassicalPositionalSafetyStrengthError(
            "v2 local raw known-answer ledger is absent"
        )
    precision = _precision_design(
        v2_result=v2_result,
        lightweight_result=lightweight,
        start_ids=start_ids,
    )
    arms = []
    for difficulty, budget in ((9, 13_887_000), (10, 18_367_000)):
        reference_arm = (
            f"classical-difficulty-{difficulty}-nodes-{budget}"
        )
        arms.extend(
            [
                {
                    "arm": f"classical-d{difficulty}-unfiltered",
                    "reference_arm": reference_arm,
                    "difficulty": difficulty,
                    "node_budget": budget,
                    "filtered": False,
                },
                {
                    "arm": f"classical-d{difficulty}-a-pos",
                    "reference_arm": reference_arm,
                    "difficulty": difficulty,
                    "node_budget": budget,
                    "filtered": True,
                },
            ]
        )
    planned_games = len(start_ids) * 2 * (1 + len(arms))
    if planned_games != 480:
        raise ClassicalPositionalSafetyStrengthError("planned games differ")
    payload = {
        "schema_version": PLAN_SCHEMA,
        "status": "frozen_before_current-dev_known-answer-or-candidate-outcome",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "What is the strict WDL effect of the delivered dev ProductPositionalSafetyGate on the same dev classical coordinator at difficulty 9 and 10?",
        "source_commit_before_plan": _git("rev-parse", "HEAD"),
        "v2_reference": {
            "plan_path": str(V2_PLAN_PATH).replace("\\", "/"),
            "plan_identity": v2_plan["plan_identity"],
            "plan_file_sha256": v2_plan_sha,
            "result_path": str(V2_RESULT_PATH).replace("\\", "/"),
            "result_identity": v2_result["result_identity"],
            "result_file_sha256": v2_result_sha,
            "raw_classical_ledger": str(raw_classical.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "raw_classical_ledger_sha256": sha256_file(raw_classical),
            "raw_reproduction_ledger": str(
                raw_reproduction.relative_to(ROOT)
            ).replace("\\", "/"),
            "raw_reproduction_ledger_sha256": sha256_file(raw_reproduction),
            "origin_main_source_commit": v2_plan["product_contract"][
                "source_commit"
            ],
            "exact_192_game_semantic_comparison": True,
            "difference_is_a_finding_not_a_hard_stop": True,
        },
        "start_pool": v2_plan["start_pool"],
        "start_subset": v2_plan["start_subset"],
        "product_contract": _product_contract(v2_plan),
        "experiment": {
            "arms": arms,
            "known_answer_arm": "random-safe",
            "known_answer_games": 96,
            "unfiltered_games": 192,
            "filtered_games": 192,
            "planned_complete_games": planned_games,
            "colors_per_start": 2,
            "maximum_post_start_logical_plies": 1536,
            "safety_cap_disposition": "incomplete-never-a-draw",
            "unfiltered_game_id_namespace": v2_plan["experiment"][
                "schedule_namespace"
            ],
            "filtered_game_id_namespace": "sanmill-classical-positional-safety-strength-v1-filtered-20260820",
            "no_result_based_early_stop_or_extension": True,
        },
        "primary_estimand": {
            "score": "strict W/D/L encoded 1/0.5/0",
            "unit": "one start after averaging candidate W and B",
            "contrasts": "current-dev A_pos-filtered minus current-dev unfiltered, separately at difficulty 9 and 10",
            "interval": "normal 95 percent interval over 48 start-level paired differences",
        },
        "precision_design": precision,
        "primary_decision": {
            "maximum_half_width": precision["maximum_half_width"],
            "filtered_higher": "interval lower bound above zero",
            "filtered_lower": "interval upper bound below zero",
            "direction_inconclusive": "interval includes zero",
            "precision_inadequate": "half width above 4.5 percentage points",
        },
        "secondary_metrics": {
            "items": [
                "actual intervention count and rate",
                "original self-downgrade transitions W-to-D, W-to-L, D-to-L",
                "strict terminal reasons",
                "source-phase WDL and intervention counts",
                "primary search and restricted-root work",
                "gate selection failures and latency",
            ],
            "cannot_flip_primary": True,
        },
        "known_answer": {
            "required_before_unfiltered": True,
            "reference_path": v2_plan["known_answer"]["reference_path"],
            "reference": v2_plan["known_answer"]["reference"],
            "expected_subset_identity": "45b4ed7c905fa53cc0813dc0a48f751fbb4e0a990c5515a6c07b18c231c2de68",
            "per_game_exact_fields": "moves, terminal reason, strict history, no-progress and repetition clocks",
            "failure_disposition": "hard stop before all current-dev classical games",
        },
        "guidance_input": v2_plan["guidance_input"],
        "sanmill_contract": v2_plan["sanmill_contract"],
        "malom_contract": v2_plan["malom_contract"],
        "prior_results": {
            "path": str(LIGHTWEIGHT_RESULT_PATH).replace("\\", "/"),
            "file_sha256": lightweight_sha,
            "same_subset_scores": v2_result["prior_scores_same_subset"],
        },
        "unfiltered_gate_rule": {
            "exact_match": "report branch divergence absent and continue",
            "difference": "categorize source/runtime/resource/filter-disabled side effects; continue only if same-dev unfiltered and filtered remain a valid internally paired comparison",
            "disabled_filter_side_effect_canary_required": True,
        },
        "resource_envelope": {
            "maximum_active_seconds": 18_000,
            "maximum_complete_games": 600,
            "planned_complete_games": planned_games,
            "maximum_parallel_measurement_processes": 1,
            "maximum_parallel_sanmill_processes": 1,
            "engine_search_anomaly_ceiling": 2_000_000,
            "malom_query_anomaly_ceiling": 50_000_000,
            "training_updates": 0,
            "checkpoint_modifications": 0,
            "database_writes": 0,
        },
        "protected_access": v2_plan["protected_access"],
        "implementation_files": _implementation_hashes(),
        "outputs": {
            "namespace": "out/evaluation/sanmill-classical-positional-safety-strength-v1-20260820-001",
            "result": "docs/evidence/sanmill-classical-positional-safety-strength-v1-manifest-2026-08-20.json",
            "evidence_document": "docs/evidence/sanmill-classical-positional-safety-strength-v1-2026-08-20.md",
        },
        "interpretation_rules": {
            "negative_or_inconclusive": "report without softening",
            "A_pos": "position-only WDL preservation, never A_allow or full-rule safety",
            "old_arms": "descriptive same-subset context only; the primary is current-dev filtered minus current-dev unfiltered",
            "product_default": "do not call this the sole human-facing default while specialist override remains active when loaded",
            "branch_difference": "do not attribute main-versus-dev behavior solely to one file without causal isolation",
        },
        "claim_boundary": {
            "exact_fixed_dev_runtime_and_start_subset_only": True,
            "internal_directional_measurement_only": True,
            "human_or_product_population_claim": False,
            "equivalence_claim": False,
            "promotion_or_deployment": False,
            "training_authorization": False,
            "position_safety": "A_pos positional-only; strict referee remains the sole terminal authority",
        },
    }
    sealed_plan = write_sealed_json(
        ROOT / PLAN_PATH, payload, identity_field="plan_identity"
    )
    plan_sha = sha256_file(ROOT / PLAN_PATH)
    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "operator": "product-owner-direct",
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_basis": "Product owner request dated 2026-08-20 to run once the bounded current-dev unfiltered-versus-A_pos classical strength measurement.",
        "plan_identity": sealed_plan["plan_identity"],
        "plan_file_sha256": plan_sha,
        "source_commit": payload["product_contract"]["source_commit"],
        "output_namespace": payload["outputs"]["namespace"],
        "resource_envelope": payload["resource_envelope"],
        "one_execution_only": True,
        "automatic_retry_or_resume": False,
        "prohibited": [
            "training or weight updates",
            "checkpoint or alias changes",
            "database writes",
            "protected segment or source-pool access",
            "rewriting frozen records",
            "changing product filter code to pass the measurement",
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


if __name__ == "__main__":
    raise SystemExit(main())
