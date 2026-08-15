#!/usr/bin/env python3
"""Run frozen exploration-only estimator readiness calibration."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.malom_db import MalomDB
from learned_ai.evaluation.human_f0h0_b2_freeze import load_membership
from learned_ai.evaluation.human_f0h0_feasibility import (
    load_f0d0_boundary,
    sha256_file,
    verify_malom_snapshot,
    write_sealed_json,
)
from learned_ai.evaluation.human_feature_deviation_design_round import load_split_v2
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    EstimatorReadinessError,
    RESULT_SCHEMA,
    extract_exploration_observations,
    load_crossfit_structure,
    load_effective_readiness_plan,
    run_crossfit_readiness,
)


def _malom_path(config_path: Path) -> Path:
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EstimatorReadinessError("local path configuration unavailable") from exc
    raw = value.get("malom_db_path") if isinstance(value, dict) else None
    if not isinstance(raw, str) or not raw:
        raise EstimatorReadinessError("malom_db_path is absent")
    path = Path(raw)
    return path if path.is_absolute() else (_ROOT / path).resolve()


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
        "--structure",
        default=("docs/experiments/human-feature-deviation-estimator-crossfit-v1.json"),
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
    parser.add_argument("--paths-config", default="data/training_paths.local.json")
    parser.add_argument(
        "--malom-manifest",
        default="data/manifests/malom-sector-corrected-v1.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "docs/evidence/human-feature-deviation-estimator-readiness-"
            "manifest-2026-08-15.json"
        ),
    )
    args = parser.parse_args()

    database: MalomDB | None = None
    effective, plan_identities = load_effective_readiness_plan(
        _ROOT / args.plan,
        inherited_v1_path=_ROOT / args.inherited_v1_plan,
    )
    structure, structure_file_sha = load_crossfit_structure(_ROOT / args.structure)
    if (
        structure["plan_identities"]["v2_plan_identity"]
        != plan_identities["v2_plan_identity"]
    ):
        parser.error("crossfit structure does not bind the effective plan")
    split, split_file_sha = load_split_v2(_ROOT / args.split)
    if split["split_identity"] != effective["lineage"]["selected_split_identity"]:
        parser.error("selected research split differs")
    boundary = load_f0d0_boundary(_ROOT / args.f0d0_manifest)
    membership, membership_file_sha = load_membership(_ROOT / args.b2_membership)
    if (
        membership["membership_identity"]
        != effective["lineage"]["b2_membership_identity"]
    ):
        parser.error("official B2 membership differs")
    malom_path = _malom_path(_ROOT / args.paths_config)
    malom_snapshot = verify_malom_snapshot(
        malom_path=malom_path,
        manifest_path=_ROOT / args.malom_manifest,
        full_hash=False,
    )
    if malom_snapshot.get("trust_level") != "sector-corrected-v1":
        parser.error("Malom label version is not sector-corrected-v1")
    database = MalomDB(malom_path)
    try:
        observations, extraction = extract_exploration_observations(
            repository_root=_ROOT,
            boundary=boundary,
            official_membership=membership,
            research_split=split,
            structure=structure,
            plan=effective,
            database=database,
        )
        fit_started = time.perf_counter()
        analysis = run_crossfit_readiness(
            observations=observations,
            extraction=extraction,
            plan=effective,
        )
        fit_seconds = time.perf_counter() - fit_started
    finally:
        database.close()

    implementation_paths = [
        "learned_ai/evaluation/human_feature_deviation_estimator_readiness.py",
        "scripts/freeze_human_feature_deviation_estimator_crossfit.py",
        "scripts/run_human_feature_deviation_estimator_readiness.py",
        "tests/test_human_feature_deviation_estimator_readiness.py",
    ]
    payload = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed_exploration_only_estimator_readiness",
        "plan_identities": plan_identities,
        "crossfit_structure_identity": structure["structure_identity"],
        "input_identities": {
            "screen_v2_plan_identity": effective["lineage"]["screen_v2_plan_identity"],
            "selected_split_identity": split["split_identity"],
            "selected_split_file_sha256": split_file_sha,
            "crossfit_structure_file_sha256": structure_file_sha,
            "f0d0_corpus_identity": effective["lineage"]["f0d0_corpus_identity"],
            "f0d0_manifest_identity": effective["lineage"]["f0d0_manifest_identity"],
            "f0d0_manifest_file_sha256": sha256_file(_ROOT / args.f0d0_manifest),
            "b2_membership_identity": membership["membership_identity"],
            "b2_membership_file_sha256": membership_file_sha,
            "malom": malom_snapshot,
            "implementation_files": {
                path: sha256_file(_ROOT / path) for path in implementation_paths
            },
        },
        "frozen_contract": {
            "feature_fields": effective["feature_contract"]["ordered_features"],
            "numerical_contract": effective["numerical_contract"],
            "cross_fit_contract": effective["cross_fit_contract"],
            "uncertainty_contract": effective["uncertainty_contract"],
            "power_contract": effective["power_contract"],
            "D_to_L_contract": effective["d_to_l_contract"],
        },
        "expanded_exploration": extraction,
        "fit_elapsed_seconds": fit_seconds,
        "analysis": analysis,
        "D_to_L_contract_reconciliation": {
            "binding": (
                "top-minus-bottom risk-quintile D-to-L difference is at least "
                "0.02 and its lower 95 percent bound is at least 0.02"
            ),
            "nonbinding_handoff_summary": "95 percent half-width at most 0.02",
            "materially_different": True,
            "binding_source": (
                "docs/experiments/human-feature-deviation-screen-v2.json"
            ),
        },
        "claim_boundary": {
            "exploratory_only": True,
            "safe_set": "A_pos",
            "positional_only": True,
            "A_allow_claim": False,
            "F0_H0_stop_remains_effective": True,
            "research_confirmation_opened": False,
            "human_trap_product_strength_or_causal_claim": False,
            "later_gate_or_training_authority": False,
        },
        "known_biases": effective["known_biases"],
        "access_audit": extraction["access_audit"],
    }
    sealed = write_sealed_json(
        _ROOT / args.output,
        payload,
        identity_field="result_identity",
    )
    print(sealed["result_identity"])
    print(analysis["readiness"]["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
