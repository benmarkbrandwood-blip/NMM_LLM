#!/usr/bin/env python3
"""Freeze estimator folds and expanded exploration membership structurally."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.human_f0h0_b2_freeze import load_membership
from learned_ai.evaluation.human_f0h0_feasibility import (
    load_f0d0_boundary,
    write_sealed_json,
)
from learned_ai.evaluation.human_feature_deviation_design_round import load_split_v2
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    EstimatorReadinessError,
    STRUCTURE_SCHEMA,
    build_community_crossfit_structure,
    load_effective_readiness_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default=(
            "docs/experiments/human-feature-deviation-estimator-readiness-v2.json"
        ),
    )
    parser.add_argument(
        "--inherited-v1-plan",
        default=(
            "docs/experiments/human-feature-deviation-estimator-readiness-v1.json"
        ),
    )
    parser.add_argument(
        "--split",
        default="docs/experiments/human-feature-deviation-train-split-v3.json",
    )
    parser.add_argument(
        "--f0d0-manifest",
        default=(
            "docs/evidence/f0-d0-human-raw-reconstructability-manifest-2026-08-14.json"
        ),
    )
    parser.add_argument(
        "--b2-membership",
        default="docs/experiments/f0-h0-design-b2-frozen-membership-v1.json",
    )
    parser.add_argument(
        "--output",
        default=("docs/experiments/human-feature-deviation-estimator-crossfit-v1.json"),
    )
    args = parser.parse_args()

    effective, plan_identities = load_effective_readiness_plan(
        _ROOT / args.plan,
        inherited_v1_path=_ROOT / args.inherited_v1_plan,
    )
    split, split_file_sha = load_split_v2(_ROOT / args.split)
    selected = effective["lineage"]
    if (
        split["split_identity"] != selected["selected_split_identity"]
        or split_file_sha != selected["selected_split_file_sha256"]
    ):
        parser.error("selected activity-balanced split identity differs")
    boundary = load_f0d0_boundary(_ROOT / args.f0d0_manifest)
    membership, membership_file_sha = load_membership(_ROOT / args.b2_membership)
    if (
        membership["membership_identity"] != selected["b2_membership_identity"]
        or membership_file_sha != selected["b2_membership_file_sha256"]
    ):
        parser.error("official B2 membership identity differs")

    assigned = split["player_membership"]["research-exploration"]["player_keys"]
    session_ids = split["partitions"]["research-exploration"]["session_ids"]
    record_by_id = {record.session_id: record for record in boundary.records}
    try:
        records = [record_by_id[session_id] for session_id in session_ids]
    except KeyError as exc:
        parser.error(f"research-exploration session missing from F0-D0: {exc}")
    games = [
        (
            record.session_id,
            record.player_keys[0],
            record.player_keys[1],
            record.move_count,
        )
        for record in records
    ]
    contract = effective["cross_fit_contract"]
    expansion = effective["exploration_expansion"]
    structure = build_community_crossfit_structure(
        assigned_players=assigned,
        games=games,
        folds=int(contract["folds"]),
        community_resolution=float(contract["community_resolution"]),
        community_seed=int(contract["community_seed"]),
        sample_seed=str(expansion["selection_seed"]),
        maximum_games_per_fold=int(expansion["maximum_games_per_fold"]),
    )
    gates = contract["pre_outcome_structural_gates"]
    failures: list[str] = []
    for row in structure["fold_metrics"]:
        if row["assigned_players"] < int(gates["minimum_assigned_players_per_fold"]):
            failures.append(f"fold {row['fold']} assigned players below minimum")
        if row["assigned_players"] > int(gates["maximum_assigned_players_per_fold"]):
            failures.append(f"fold {row['fold']} assigned players above maximum")
        if row["participating_players"] < int(
            gates["minimum_participating_players_per_fold"]
        ):
            failures.append(f"fold {row['fold']} participating players below minimum")
        if row["games"] < int(gates["minimum_internal_games_per_fold"]):
            failures.append(f"fold {row['fold']} games below minimum")
    if structure["sample_players"] < int(gates["minimum_total_sample_players"]):
        failures.append("total sample players below minimum")
    hard = expansion["hard_budget"]
    if structure["sample_game_count"] > int(expansion["maximum_total_games"]):
        failures.append("sample games exceed maximum")
    if structure["sample_decisions"] > int(hard["maximum_decisions"]):
        failures.append("sample decisions exceed maximum")
    if failures:
        raise EstimatorReadinessError("; ".join(failures))

    f0d0_path = _ROOT / args.f0d0_manifest
    payload = {
        "schema_version": STRUCTURE_SCHEMA,
        "status": "frozen_structure_before_new_outcome_or_malom_read",
        "plan_identities": plan_identities,
        "input_identities": {
            "screen_v2_plan_identity": selected["screen_v2_plan_identity"],
            "selected_split_identity": split["split_identity"],
            "selected_split_file_sha256": split_file_sha,
            "f0d0_corpus_identity": selected["f0d0_corpus_identity"],
            "f0d0_manifest_identity": selected["f0d0_manifest_identity"],
            "f0d0_manifest_file_sha256": hashlib.sha256(
                f0d0_path.read_bytes()
            ).hexdigest(),
            "b2_membership_identity": membership["membership_identity"],
            "b2_membership_file_sha256": membership_file_sha,
        },
        "cross_fit_contract": dict(contract),
        "structural_gates": {
            "thresholds": dict(gates),
            "failures": failures,
            "passed": not failures,
        },
        "structure": structure,
        "access_audit": {
            "raw_game_files_opened": 0,
            "human_actions_features_malom_labels_or_outcomes_read": 0,
            "research_confirmation_content_reads": 0,
            "official_selection_content_reads": 0,
            "official_confirmation_content_reads": 0,
            "official_final_test_content_reads": 0,
            "source_pool_2eb04f54_reads_or_consumption": 0,
            "human_db_reads": 0,
            "database_writes": 0,
            "games_searches_models_or_training": 0,
        },
    }
    sealed = write_sealed_json(
        _ROOT / args.output,
        payload,
        identity_field="structure_identity",
    )
    print(sealed["structure_identity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
