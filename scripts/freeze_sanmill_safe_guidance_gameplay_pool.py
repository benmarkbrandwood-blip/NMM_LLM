#!/usr/bin/env python3
"""Freeze the candidate-blind complete-history gameplay start pool."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.malom_db import MalomDB
from learned_ai.evaluation.human_f0h0_b2_freeze import load_membership
from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    load_f0d0_boundary,
    sha256_file,
    verify_malom_snapshot,
    write_sealed_json,
)
from learned_ai.evaluation.human_feature_deviation_design_round import load_split_v2
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    EstimatorAccess,
    load_crossfit_structure,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    POOL_SCHEMA,
    load_plan,
)
from learned_ai.evaluation.sanmill_safe_inducement import (
    MAIN_POOL_SCHEMA,
    POOL_SCHEMA as PREPROBE_POOL_SCHEMA,
    build_state_pool,
    load_state_pool,
)
from learned_ai.training.sanmill_referee import nmm_move_actions


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
        "--plan", default="docs/experiments/sanmill-safe-guidance-gameplay-v1.json"
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
        default="docs/experiments/human-feature-deviation-estimator-crossfit-v1.json",
    )
    parser.add_argument("--paths-config", default="data/training_paths.local.json")
    parser.add_argument(
        "--malom-manifest",
        default="data/manifests/malom-sector-corrected-v1.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "docs/experiments/sanmill-safe-guidance-gameplay-start-pool-v1.json"
        ),
    )
    args = parser.parse_args()

    output = _ROOT / args.output
    if output.exists():
        parser.error("gameplay start pool already exists")
    plan, plan_file_sha = load_plan(_ROOT / args.plan)
    main_pool, main_pool_sha = load_state_pool(
        _ROOT / args.main_pool, schema=MAIN_POOL_SCHEMA
    )
    preprobe_pool, preprobe_pool_sha = load_state_pool(
        _ROOT / args.preprobe_pool, schema=PREPROBE_POOL_SCHEMA
    )
    excluded = frozenset(
        (str(row["session_id"]), int(row["logical_ply"]))
        for pool in (main_pool, preprobe_pool)
        for row in pool["states"]
    )
    contract = plan["start_pool_contract"]
    if (
        len(excluded) != int(contract["exclude_prior_coordinates"])
        or canonical_sha256(
            sorted([session_id, ply] for session_id, ply in excluded)
        )
        != contract["exclude_prior_coordinates_identity"]
    ):
        parser.error("frozen prior-coordinate exclusion differs")

    boundary = load_f0d0_boundary(_ROOT / args.f0d0_manifest)
    membership, membership_sha = load_membership(_ROOT / args.official_membership)
    research_split, research_split_sha = load_split_v2(_ROOT / args.research_split)
    structure, structure_sha = load_crossfit_structure(
        _ROOT / args.crossfit_structure
    )
    malom_path = _local_path(_ROOT / args.paths_config, "malom_db_path")
    malom_snapshot = verify_malom_snapshot(
        malom_path=malom_path,
        manifest_path=_ROOT / args.malom_manifest,
        full_hash=False,
    )
    if (
        malom_snapshot["trust_level"] != "sector-corrected-v1"
        or malom_snapshot["content_sha256"]
        != plan["input_identities"]["malom_content_sha256"]
    ):
        parser.error("Malom snapshot differs from frozen protocol")
    database = MalomDB(malom_path)
    try:
        payload = build_state_pool(
            repository_root=_ROOT,
            boundary=boundary,
            official_membership=membership,
            research_split=research_split,
            crossfit_structure=structure,
            database=database,
            states_per_phase=int(contract["states_per_phase"]),
            schema_version=POOL_SCHEMA,
            selection_seed=str(contract["selection_seed"]),
            excluded_coordinates=excluded,
        )
    finally:
        database.close()
    if payload["state_count"] != int(contract["states"]):
        parser.error("gameplay pool state count differs")

    sample_rows = structure["structure"]["sample_games"]
    sample_by_session = {str(row["session_id"]): row for row in sample_rows}
    selected_ids = [str(row["session_id"]) for row in payload["states"]]
    access = EstimatorAccess.from_memberships(
        membership, research_split, allowed_sessions=selected_ids
    )
    records = {record.session_id: record for record in boundary.records}
    for state in payload["states"]:
        session_id = str(state["session_id"])
        source = sample_by_session.get(session_id)
        if source is None:
            parser.error("selected session lost cross-fit membership")
        fold = int(source["fold"])
        if not 0 <= fold < 5:
            parser.error("selected session fold differs")
        decisions = access.load_decisions(
            _ROOT, records[session_id], boundary
        )
        prior = [
            decision
            for decision in decisions
            if decision.logical_ply < int(state["logical_ply"])
        ]
        logical_turns = [list(nmm_move_actions(row.move)) for row in prior]
        flattened = [action for turn in logical_turns for action in turn]
        if (
            len(logical_turns) != int(state["logical_ply"])
            or flattened != state["history_actions"]
        ):
            parser.error("selected complete logical history differs")
        state["logical_turns"] = logical_turns
        state["oof_fold"] = fold
        state["oof_fold_source"] = (
            "persisted source-session fold; no estimator value was read"
        )

    selected_coordinates = {
        (str(row["session_id"]), int(row["logical_ply"]))
        for row in payload["states"]
    }
    if selected_coordinates & excluded:
        parser.error("gameplay pool overlaps a prior measured coordinate")
    payload["status"] = "frozen_before_any_gameplay_or_sanmill_observation"
    payload["plan_binding"] = {
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": plan_file_sha,
    }
    payload["prior_coordinate_exclusion"] = {
        "coordinates": len(excluded),
        "coordinates_identity": canonical_sha256(
            sorted([session_id, ply] for session_id, ply in excluded)
        ),
        "selected_overlap": 0,
        "main_pool_identity": main_pool["pool_identity"],
        "main_pool_file_sha256": main_pool_sha,
        "preprobe_pool_identity": preprobe_pool["pool_identity"],
        "preprobe_pool_file_sha256": preprobe_pool_sha,
        "migration_pool_is_main_pool": True,
    }
    payload["selection_blindness"] = {
        "selection_completed_before_malom_construction": True,
        "human_estimator_prediction_reads": 0,
        "sanmill_observations": 0,
        "human_chosen_action_or_result_used_for_rank": False,
        "replacement_after_oracle_or_engine_observation": False,
    }
    payload["complete_history"] = {
        "logical_turns_persisted": True,
        "action_tokens_persisted": True,
        "strict_rule_clocks_computed_only_by_runtime_replay": True,
        "starts": len(payload["states"]),
    }
    payload["input_identities"] = {
        "f0d0_corpus_identity": boundary.manifest["identities"]["corpus_identity"],
        "f0d0_manifest_identity": boundary.manifest["manifest_identity"],
        "f0d0_manifest_file_sha256": sha256_file(_ROOT / args.f0d0_manifest),
        "official_membership_identity": membership["membership_identity"],
        "official_membership_file_sha256": membership_sha,
        "research_split_identity": research_split["split_identity"],
        "research_split_file_sha256": research_split_sha,
        "crossfit_structure_identity": structure["structure_identity"],
        "crossfit_structure_file_sha256": structure_sha,
        "malom_snapshot": malom_snapshot,
    }
    payload["access_audit"]["research_exploration_raw_replays"] += sum(
        access.successful.values()
    )
    payload["access_audit"]["official_selection_content_reads"] = 0
    payload["access_audit"]["official_confirmation_content_reads"] = 0
    payload["access_audit"]["official_final_test_content_reads"] = 0
    payload["access_audit"]["research_confirmation_content_reads"] = 0
    payload["access_audit"]["source_pool_2eb04f54_reads_or_consumption"] = 0
    sealed = write_sealed_json(output, payload, identity_field="pool_identity")
    print(sealed["pool_identity"])
    print(sealed["state_membership_identity"])
    print(sealed["resource_use"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
