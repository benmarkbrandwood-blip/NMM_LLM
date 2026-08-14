#!/usr/bin/env python3
"""Freeze the high-precision retained-v3/v4 held-out score machine plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.retained_heldout_score import (  # noqa: E402
    EXPECTED_GAMES,
    EXPECTED_PHASE_COUNTS,
    EXPECTED_POOL_IDENTITY,
    EXPECTED_POOL_RECORDS_IDENTITY,
    EXPECTED_PREFIX_RECORDS_IDENTITY,
    EXPECTED_STARTS,
    MAX_POST_START_LOGICAL_PLIES,
    MAX_PRIMARY_HALF_WIDTH,
    PLAN_SCHEMA,
    SANMILL_NODE_CEILING,
    load_corpus_records,
)
from learned_ai.training.run_contract import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from scripts.run_retained_heldout_score import (  # noqa: E402
    MAX_ACTIVE_HOURS,
    load_plan,
)
from tools.prepare_retained_heldout_score_inputs import (  # noqa: E402
    SOURCE_SNAPSHOT_IDENTITY,
    build_manifest,
)


SOURCE_PLAN = _ROOT / (
    "docs/experiments/sanmill-retained-v3-v4-phase-process-generalization-v1.json"
)
SOURCE_PLAN_IDENTITY = (
    "4c85ff3362927db9b63014e0c91022a5d169d19efa4aa85b3a643febd0ce3256"
)
CORPUS = _ROOT / (
    "docs/experiments/sanmill-retained-v3-v4-late-import-heldout-pool-v1.json"
)
OUTPUT = _ROOT / ("docs/experiments/sanmill-retained-v3-v4-heldout-score-v1.json")
IMPLEMENTATION_COMMIT = "5eb142383f710c17377deedc8b1cfcc5287daa02"
OUTPUT_ROOT = (
    "learned_ai/checkpoints/evaluation/sanmill-retained-v3-v4-heldout-score-v1"
)


class FreezeHeldoutScorePlanError(RuntimeError):
    """Raised when a frozen input or derived plan differs."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FreezeHeldoutScorePlanError(f"{path.name} is not an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_identity(payload: Mapping[str, Any], field: str) -> str:
    identity = payload.get(field)
    body = {key: value for key, value in payload.items() if key != field}
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise FreezeHeldoutScorePlanError(f"{field} differs")
    return identity


def build_plan() -> dict[str, Any]:
    """Build the exact plan from the frozen pool, routes, and product choice."""
    if len(IMPLEMENTATION_COMMIT) != 40:
        raise FreezeHeldoutScorePlanError("implementation commit is not frozen")
    source = _read_json(SOURCE_PLAN)
    if _canonical_identity(source, "plan_identity") != SOURCE_PLAN_IDENTITY:
        raise FreezeHeldoutScorePlanError("completed source plan differs")
    corpus = _read_json(CORPUS)
    prefix = load_corpus_records(corpus)
    if (
        corpus.get("pool_identity") != EXPECTED_POOL_IDENTITY
        or corpus.get("records_identity") != EXPECTED_POOL_RECORDS_IDENTITY
        or canonical_sha256([str(record["record_identity"]) for record in prefix])
        != EXPECTED_PREFIX_RECORDS_IDENTITY
    ):
        raise FreezeHeldoutScorePlanError("held-out pool or prefix differs")
    manifest = build_manifest()
    if manifest.get("source_snapshot_identity") != SOURCE_SNAPSHOT_IDENTITY:
        raise FreezeHeldoutScorePlanError("held-out input source differs")
    manifest_path = _ROOT / manifest["target_root"] / "manifest.json"
    manifest_by_candidate = {
        item["candidate_id"]: item for item in manifest["candidates"]
    }

    candidates = []
    for inherited in source["candidates"]:
        candidate_id = str(inherited["candidate_id"])
        snapshot = manifest_by_candidate[candidate_id]
        candidates.append(
            {
                **inherited,
                "bundle": {
                    **inherited["bundle"],
                    "path": snapshot["route_bundle"]["path"],
                    "files_identity": snapshot["route_bundle"]["files_identity"],
                    "read_only": True,
                },
                "specialist_db": {
                    **inherited["specialist_db"],
                    "path": snapshot["specialist_db"]["path"],
                    "read_only": True,
                    "sidecars_absent": True,
                },
            }
        )

    max_search_turns = EXPECTED_GAMES * (MAX_POST_START_LOGICAL_PLIES // 2)
    body = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": "sanmill-retained-v3-v4-heldout-score-v1",
        "diagnostic_id": "dev-v4-sanmill-retained-v3-v4-heldout-score-v1",
        "status": "frozen_awaiting_product_authorization",
        "decision_date": "2026-08-14",
        "objective": (
            "Estimate the named retained-v4 minus retained-v3 candidate-score "
            "difference against pinned strict Sanmill on the preregistered "
            "253-start late-import held-out prefix with a target 95% "
            "engineering half-width of at most 1.5 percentage points."
        ),
        "implementation": {"branch": "dev", "commit": IMPLEMENTATION_COMMIT},
        "rules": source["rules"],
        "baseline": source["baseline"],
        "data": source["data"],
        "inputs": {
            "root": manifest["target_root"],
            "manifest_path": f"{manifest['target_root']}/manifest.json",
            "manifest_file_sha256": _sha256_file(manifest_path),
            "snapshot_identity": manifest["snapshot_identity"],
            "source_snapshot_identity": SOURCE_SNAPSHOT_IDENTITY,
            "successor_owned": True,
            "read_only": True,
            "sqlite_sidecars_absent": True,
        },
        "candidates": candidates,
        "corpus": {
            "path": CORPUS.relative_to(_ROOT).as_posix(),
            "file_sha256": _sha256_file(CORPUS),
            "pool_identity": EXPECTED_POOL_IDENTITY,
            "pool_records_identity": EXPECTED_POOL_RECORDS_IDENTITY,
            "prefix_records_identity": EXPECTED_PREFIX_RECORDS_IDENTITY,
            "prefix_rule": "first 253 records in the frozen master order",
            "records": EXPECTED_STARTS,
            "phase_counts": EXPECTED_PHASE_COUNTS,
            "status": "candidate_blind_training_disjoint_heldout_prefix_frozen",
            "held_out": True,
            "candidate_policy_loaded_during_selection": False,
            "candidate_outcomes_read_during_selection": 0,
            "source_game_unique": True,
            "ring16_unique": True,
        },
        "protocol": {
            "candidate_move_selection": (
                "deterministic CPU float32 policy argmax over each exact "
                "s-gen-v2-training-aligned-v1 route"
            ),
            "candidate_colors_per_start": 2,
            "color_swap": True,
            "complete_variable_history_replay": True,
            "start_verification": (
                "local FEN plus strict Sanmill FEN, history SHA-256, clocks, "
                "logical plies and nonterminal state"
            ),
            "horizon_post_start_logical_plies": 108,
            "horizon_disposition": (
                "secondary ongoing indicator only, never a draw or strength proxy"
            ),
            "max_post_start_logical_plies": MAX_POST_START_LOGICAL_PLIES,
            "safety_cap_disposition": (
                "incomplete-invalid-for-primary-score-and-WDL-not-draw"
            ),
            "strict_referee": (
                "Sanmill complete-history state after every logical turn"
            ),
            "sanmill_node_ceiling_per_turn": SANMILL_NODE_CEILING,
            "exact_same_spec_resume_only": True,
            "host_interruption_missing_suffix_resume_only": True,
            "automatic_retry": False,
            "semantic_failure_recovery": False,
            "result_based_early_stop": False,
            "malom_snapshot_diagnostic": (
                "history-free theoretical WDL at the relative horizon"
            ),
            "malom_move_diagnostic": (
                "exact WDL downgrade conditional on settled-board query coverage"
            ),
            "mechanism_reanalysis": "none",
        },
        "analysis": {
            "unit": (
                "one frozen start after averaging its candidate-White and "
                "candidate-Black v4-minus-v3 score differences"
            ),
            "candidate_score": {"win": 1.0, "draw": 0.5, "loss": 0.0},
            "primary_estimand": (
                "mean across 253 starts of the within-colour v4-minus-v3 "
                "candidate-score difference after averaging both colours"
            ),
            "engineering_interval": {
                "method": "normal-interval-on-start-clustered-score-difference",
                "z": 1.96,
                "maximum_primary_half_width": MAX_PRIMARY_HALF_WIDTH,
                "interpretation": (
                    "named-route fixed held-out corpus engineering interval; "
                    "not a population variance guarantee or equivalence margin"
                ),
            },
            "primary_decision_rule": {
                "v4_higher_fixed_heldout_score": (
                    "lower_bound > 0 and half_width <= 0.015 and all games "
                    "reach strict rules terminals"
                ),
                "v3_higher_fixed_heldout_score": (
                    "upper_bound < 0 and half_width <= 0.015 and all games "
                    "reach strict rules terminals"
                ),
                "inconclusive": (
                    "interval includes zero and half_width <= 0.015 and all "
                    "games reach strict rules terminals"
                ),
                "inconclusive_precision": "half_width > 0.015",
                "inconclusive_incomplete_safety_cap": (
                    "one or more games lack a strict rules terminal"
                ),
            },
            "secondary_metrics": [
                "candidate score and strict W/D/L by phase and candidate colour",
                "108-post-start-ply survival with start-clustered difference",
                "start, horizon and final no-capture and repetition state",
                "post-start and total logical-ply distributions",
                "strict termination reasons and invalid safety caps",
                "Malom query coverage and exact coarse-WDL move deltas",
            ],
            "secondary_interpretation": (
                "descriptive process evidence only; no secondary endpoint may "
                "replace the paired-score primary after outcomes are observed"
            ),
        },
        "workload": {
            "unique_starts": EXPECTED_STARTS,
            "candidate_colors_per_start": 2,
            "candidates_per_color_unit": 2,
            "matched_color_units": EXPECTED_STARTS * 2,
            "games": EXPECTED_GAMES,
            "max_active_hours": MAX_ACTIVE_HOURS,
            "max_sanmill_search_turns": max_search_turns,
            "max_summed_node_ceiling": max_search_turns * SANMILL_NODE_CEILING,
            "safe_exact_resume_same_spec": True,
            "host_interruption_exact_resume_only": True,
            "automatic_retry_or_recovery": False,
            "semantic_failure_recovery": False,
            "expansion": False,
        },
        "claim_boundary": {
            "candidate_blind_source_selection": True,
            "held_out": True,
            "named_route_fixed_corpus_score_relation": True,
            "general_playing_strength_or_elo_claim": False,
            "refresh_causal_claim": False,
            "equivalence_claim": False,
            "training_or_update": False,
            "automatic_promotion": False,
            "publication": False,
            "release": False,
        },
        "gates": [
            "clean published dev with implementation commit as ancestor",
            "exact plan and stable source-readiness-bound product authorization",
            "successor-owned read-only route and sidecar-free database snapshots",
            "exact final checkpoints and CPU-verified training-aligned routes",
            "common HumanDB and corrected Malom identities verify read-only",
            "exact frozen 253-start prefix and adjacent 1012-game schedule",
            "pinned strict Sanmill runtime and 500000-node deterministic canary",
            "all 253 complete histories replay without loading a candidate",
            "focused evaluator, runner, ledger and web tests pass",
            "mandatory Malom and label-provenance tests pass",
            "no competing trainer or evaluator",
            "fresh ignored targets or exact same-spec partial missing suffix",
        ],
        "outputs": {
            "root": OUTPUT_ROOT,
            "authorization": f"{OUTPUT_ROOT}/authorization.json",
            "specification": f"{OUTPUT_ROOT}/spec.json",
            "ledger": f"{OUTPUT_ROOT}/games.jsonl",
            "progress": f"{OUTPUT_ROOT}/progress.json",
            "report": f"{OUTPUT_ROOT}/report.json",
            "completion": f"{OUTPUT_ROOT}/completion.json",
            "failure": f"{OUTPUT_ROOT}/failure.json",
        },
    }
    return {**body, "plan_identity": canonical_sha256(body)}


def write_plan(plan: Mapping[str, Any], output: Path = OUTPUT) -> None:
    if output.exists():
        raise FileExistsError(f"held-out score plan already exists: {output}")
    output.write_bytes(canonical_json_bytes(plan))
    if load_plan(output) != plan:
        raise FreezeHeldoutScorePlanError("persisted held-out score plan differs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    plan = build_plan()
    write_plan(plan, args.output)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
