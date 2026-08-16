#!/usr/bin/env python3
"""Freeze the blind 360-state pool for the v2 main mechanism experiment."""

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
    load_crossfit_structure,
)
from learned_ai.evaluation.sanmill_safe_inducement import (
    MAIN_POOL_SCHEMA,
    build_state_pool,
    load_main_plan,
    load_state_pool,
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
        "--plan",
        default="docs/experiments/sanmill-safe-inducement-mechanism-v2.json",
    )
    parser.add_argument(
        "--preprobe-pool",
        default=(
            "docs/experiments/sanmill-safe-inducement-preprobe-"
            "state-pool-v1.json"
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
        default=(
            "docs/experiments/human-feature-deviation-estimator-crossfit-v1.json"
        ),
    )
    parser.add_argument("--paths-config", default="data/training_paths.local.json")
    parser.add_argument(
        "--malom-manifest",
        default="data/manifests/malom-sector-corrected-v1.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "docs/experiments/sanmill-safe-inducement-main-"
            "state-pool-v2.json"
        ),
    )
    args = parser.parse_args()

    plan, plan_file_sha = load_main_plan(_ROOT / args.plan)
    preprobe, preprobe_file_sha = load_state_pool(_ROOT / args.preprobe_pool)
    contract = plan["state_pool_contract"]
    excluded = frozenset(
        (str(row["session_id"]), int(row["logical_ply"]))
        for row in preprobe["states"]
    )
    excluded_identity = canonical_sha256(
        sorted([session_id, ply] for session_id, ply in excluded)
    )
    if (
        len(excluded) != int(contract["excluded_preprobe_coordinates"])
        or excluded_identity != contract["excluded_preprobe_coordinates_identity"]
    ):
        parser.error("preprobe exclusion set differs from frozen v2 protocol")

    boundary = load_f0d0_boundary(_ROOT / args.f0d0_manifest)
    membership, membership_file_sha = load_membership(
        _ROOT / args.official_membership
    )
    research_split, research_split_file_sha = load_split_v2(
        _ROOT / args.research_split
    )
    structure, structure_file_sha = load_crossfit_structure(
        _ROOT / args.crossfit_structure
    )
    malom_path = _local_path(_ROOT / args.paths_config, "malom_db_path")
    malom = verify_malom_snapshot(
        malom_path=malom_path,
        manifest_path=_ROOT / args.malom_manifest,
        full_hash=False,
    )
    if malom.get("trust_level") != "sector-corrected-v1":
        parser.error("Malom snapshot is not sector-corrected-v1")

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
            schema_version=MAIN_POOL_SCHEMA,
            selection_seed=str(contract["selection_seed"]),
            excluded_coordinates=excluded,
        )
    finally:
        database.close()
    if int(payload["state_count"]) != int(contract["states"]):
        parser.error("main state count differs from frozen protocol")
    selected_coordinates = {
        (str(row["session_id"]), int(row["logical_ply"]))
        for row in payload["states"]
    }
    if selected_coordinates & excluded:
        parser.error("main pool overlaps the preprobe pool")
    payload["plan_binding"] = {
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": plan_file_sha,
    }
    payload["preprobe_exclusion"] = {
        "pool_identity": preprobe["pool_identity"],
        "pool_file_sha256": preprobe_file_sha,
        "coordinates": len(excluded),
        "coordinates_identity": excluded_identity,
        "selected_overlap": 0,
    }
    payload["input_identities"] = {
        "f0d0_corpus_identity": boundary.manifest["identities"]["corpus_identity"],
        "f0d0_manifest_identity": boundary.manifest["manifest_identity"],
        "f0d0_manifest_file_sha256": sha256_file(_ROOT / args.f0d0_manifest),
        "official_membership_identity": membership["membership_identity"],
        "official_membership_file_sha256": membership_file_sha,
        "research_split_identity": research_split["split_identity"],
        "research_split_file_sha256": research_split_file_sha,
        "crossfit_structure_identity": structure["structure_identity"],
        "crossfit_structure_file_sha256": structure_file_sha,
        "malom": malom,
    }
    sealed = write_sealed_json(
        _ROOT / args.output,
        payload,
        identity_field="pool_identity",
    )
    print(sealed["pool_identity"])
    print(sealed["state_membership_identity"])
    print(sealed["oracle_construction"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
