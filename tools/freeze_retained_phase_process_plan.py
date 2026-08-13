#!/usr/bin/env python3
"""Freeze the immutable retained phase-process machine plan."""

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

from learned_ai.evaluation.retained_phase_process_generalization import (  # noqa: E402
    PLAN_SCHEMA,
)
from learned_ai.evaluation.retained_phase_process_corpus import (  # noqa: E402
    validate_retained_phase_process_corpus,
)
from learned_ai.training.run_contract import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from scripts.run_retained_phase_process_generalization import (  # noqa: E402
    load_plan,
)
from tools.prepare_retained_phase_process_inputs import (  # noqa: E402
    build_manifest,
)


SOURCE_PLAN = _ROOT / (
    "docs/experiments/sanmill-retained-v3-v4-passivity-diagnostic-v1.json"
)
SOURCE_PLAN_IDENTITY = (
    "035c68f80b94dddb8d139d56c38c86c4fde29fa13de5e19db1f4e1fe484c318e"
)
CORPUS = _ROOT / (
    "docs/experiments/sanmill-retained-v3-v4-phase-process-corpus-v1.json"
)
OUTPUT = _ROOT / (
    "docs/experiments/"
    "sanmill-retained-v3-v4-phase-process-generalization-v1.json"
)
IMPLEMENTATION_COMMIT = "32e8843b791ea0ebbf149b5ad4ccfb96ad13318f"
OUTPUT_ROOT = (
    "learned_ai/checkpoints/evaluation/"
    "sanmill-retained-v3-v4-phase-process-generalization-v1"
)


class FreezePhaseProcessPlanError(RuntimeError):
    """Raised when a frozen source or derived plan differs."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FreezePhaseProcessPlanError(f"{path.name} is not an object")
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
        raise FreezePhaseProcessPlanError(f"{field} differs")
    return identity


def build_plan() -> dict[str, Any]:
    """Build the exact plan from frozen predecessor and successor inputs."""
    source = _read_json(SOURCE_PLAN)
    if _canonical_identity(source, "plan_identity") != SOURCE_PLAN_IDENTITY:
        raise FreezePhaseProcessPlanError("completed source plan differs")
    corpus = _read_json(CORPUS)
    validate_retained_phase_process_corpus(corpus)
    manifest = build_manifest()
    manifest_path = Path(manifest["target_root"]) / "manifest.json"
    if not manifest_path.is_absolute():
        manifest_path = _ROOT / manifest_path
    manifest_by_candidate = {
        item["candidate_id"]: item for item in manifest["candidates"]
    }

    candidates = []
    for inherited in source["candidates"]:
        candidate_id = str(inherited["candidate_id"])
        snapshot = manifest_by_candidate[candidate_id]
        candidate = {
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
        candidates.append(candidate)

    body = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": "sanmill-retained-v3-v4-phase-process-generalization-v1",
        "diagnostic_id": (
            "dev-v4-sanmill-retained-v3-v4-phase-process-generalization-v1"
        ),
        "status": "frozen_awaiting_product_authorization",
        "decision_date": "2026-08-13",
        "objective": (
            "Describe whether the named retained-v4 route has higher strict-"
            "referee continuation survival than retained-v3 through 108 "
            "additional logical plies after fixed phase-history starts."
        ),
        "implementation": {"branch": "dev", "commit": IMPLEMENTATION_COMMIT},
        "rules": source["rules"],
        "baseline": source["baseline"],
        "data": source["data"],
        "inputs": {
            "root": manifest["target_root"],
            "manifest_path": (
                f"{manifest['target_root']}/manifest.json"
            ),
            "manifest_file_sha256": _sha256_file(manifest_path),
            "snapshot_identity": manifest["snapshot_identity"],
            "source_completed_plan_identity": SOURCE_PLAN_IDENTITY,
            "successor_owned": True,
            "read_only": True,
            "sqlite_sidecars_absent": True,
        },
        "candidates": candidates,
        "corpus": {
            "path": CORPUS.relative_to(_ROOT).as_posix(),
            "file_sha256": _sha256_file(CORPUS),
            "identity": corpus["corpus_identity"],
            "records_identity": corpus["records_identity"],
            "records": 39,
            "phase_counts": {"placement": 18, "movement": 14, "flying": 7},
            "source_history_logical_ply_range": [7, 178],
            "status": "project_visible_candidate_blind_fixed_process_corpus",
            "held_out": False,
            "prior_opening_exact_overlap": 0,
            "prior_opening_ring16_overlap": 0,
            "trainer_visible_d4_start_hits": 0,
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
                "ongoing indicator only, never a draw or final adjudication"
            ),
            "max_post_start_logical_plies": 1536,
            "safety_cap_disposition": (
                "incomplete-invalid-for-eventual-WDL-not-draw"
            ),
            "strict_referee": (
                "Sanmill complete-history state after every logical turn"
            ),
            "sanmill_node_ceiling_per_turn": 500000,
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
            "mechanism_reanalysis": (
                "zero-new-game complete-ledger safe-capture, suffix-revisit "
                "and full-order audit inside the same active-time budget"
            ),
        },
        "analysis": {
            "unit": (
                "one frozen start after averaging its candidate-White and "
                "candidate-Black v4-minus-v3 differences"
            ),
            "primary_estimand": (
                "mean across 39 starts of v4 minus v3 ongoing after 108 "
                "post-start logical plies"
            ),
            "engineering_interval": {
                "method": "normal-interval-on-start-clustered-difference",
                "z": 1.96,
                "maximum_primary_half_width": 0.10,
                "interpretation": (
                    "fixed project-visible corpus engineering interval, not "
                    "population inference"
                ),
            },
            "primary_decision_rule": {
                "v4_higher_108_post_start_ply_survival": (
                    "lower_bound > 0 and half_width <= 0.10"
                ),
                "v3_higher_108_post_start_ply_survival": (
                    "upper_bound < 0 and half_width <= 0.10"
                ),
                "inconclusive": (
                    "interval includes zero and half_width <= 0.10"
                ),
                "inconclusive_precision": "half_width > 0.10",
            },
            "secondary_metrics": [
                "phase and candidate-colour survival with denominators",
                "start, horizon and final no-capture and repetition state",
                "post-start and total logical-ply distributions",
                "strict outcome reasons and eventual descriptive WDL",
                "Malom query coverage and exact coarse-WDL move deltas",
                "safe-capture opportunity, selection and suffix-revisit rates",
                "complete Malom-order opportunity and normalized ordinal regret",
            ],
            "mechanism_metrics": (
                "exploratory with start-clustered intervals and no directional "
                "acceptance gate"
            ),
        },
        "workload": {
            "unique_starts": 39,
            "candidate_colors_per_start": 2,
            "candidates_per_color_unit": 2,
            "matched_color_units": 78,
            "games": 156,
            "max_active_hours": 2.0,
            "max_sanmill_search_turns": 119808,
            "max_summed_node_ceiling": 59904000000,
            "mechanism_reanalysis_new_games": 0,
            "safe_exact_resume_same_spec": True,
            "host_interruption_exact_resume_only": True,
            "automatic_retry_or_recovery": False,
            "semantic_failure_recovery": False,
            "expansion": False,
        },
        "claim_boundary": {
            "corpus_previously_project_visible": True,
            "held_out": False,
            "held_out_strength_claim": False,
            "playing_strength_claim": False,
            "refresh_causal_claim": False,
            "equivalence_claim": False,
            "training_or_update": False,
            "promotion": False,
            "publication": False,
            "release": False,
        },
        "gates": [
            "clean published dev with implementation commit as ancestor",
            "exact plan and stable source-readiness-bound product authorization",
            "successor-owned read-only route and sidecar-free database snapshots",
            "exact final checkpoints and CPU-verified training-aligned routes",
            "common HumanDB and corrected Malom identities verify read-only",
            "exact 39-start corpus and adjacent 156-game schedule",
            "pinned strict Sanmill runtime and 500000-node deterministic canary",
            "all 39 variable histories replay without loading a candidate",
            "focused evaluator, runner, ledger, mechanism and web tests pass",
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
            "mechanism_report": f"{OUTPUT_ROOT}/mechanism-report.json",
            "completion": f"{OUTPUT_ROOT}/completion.json",
            "failure": f"{OUTPUT_ROOT}/failure.json",
        },
    }
    return {**body, "plan_identity": canonical_sha256(body)}


def write_plan(plan: Mapping[str, Any], output: Path = OUTPUT) -> None:
    if output.exists():
        raise FileExistsError(f"phase-process plan already exists: {output}")
    output.write_bytes(canonical_json_bytes(plan))
    loaded = load_plan(output)
    if loaded != plan:
        raise FreezePhaseProcessPlanError("persisted phase-process plan differs")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan = build_plan()
    write_plan(plan, args.output)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
