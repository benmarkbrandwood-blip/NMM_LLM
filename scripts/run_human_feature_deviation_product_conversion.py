#!/usr/bin/env python3
"""Run the frozen exploration-only product-conversion derivation."""

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
    load_f0d0_boundary,
    sha256_file,
    verify_malom_snapshot,
    write_sealed_json,
)
from learned_ai.evaluation.human_feature_deviation_design_round import load_split_v2
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    load_crossfit_structure,
    load_effective_readiness_plan,
)
from learned_ai.evaluation.human_feature_deviation_product_conversion import (
    ConversionError,
    RESULT_SCHEMA,
    derive_product_conversion,
    load_effective_conversion_plan,
    load_readiness_result,
)


def _malom_path(config_path: Path) -> Path:
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionError("local path configuration unavailable") from exc
    raw = value.get("malom_db_path") if isinstance(value, dict) else None
    if not isinstance(raw, str) or not raw:
        raise ConversionError("malom_db_path is absent")
    path = Path(raw)
    return path if path.is_absolute() else (_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conversion-plan",
        default=(
            "docs/experiments/human-feature-deviation-product-conversion-"
            "derivation-v2.json"
        ),
    )
    parser.add_argument(
        "--inherited-conversion-plan",
        default=(
            "docs/experiments/human-feature-deviation-product-conversion-"
            "derivation-v1.json"
        ),
    )
    parser.add_argument(
        "--readiness-plan",
        default=(
            "docs/experiments/human-feature-deviation-estimator-readiness-v2.json"
        ),
    )
    parser.add_argument(
        "--inherited-readiness-plan",
        default=(
            "docs/experiments/human-feature-deviation-estimator-readiness-v1.json"
        ),
    )
    parser.add_argument(
        "--structure",
        default=("docs/experiments/human-feature-deviation-estimator-crossfit-v1.json"),
    )
    parser.add_argument(
        "--readiness-result",
        default=(
            "docs/evidence/human-feature-deviation-estimator-readiness-"
            "manifest-2026-08-15.json"
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
    parser.add_argument("--paths-config", default="data/training_paths.local.json")
    parser.add_argument(
        "--malom-manifest",
        default="data/manifests/malom-sector-corrected-v1.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "docs/evidence/human-feature-deviation-product-conversion-"
            "manifest-2026-08-15.json"
        ),
    )
    args = parser.parse_args()

    conversion, conversion_identities = load_effective_conversion_plan(
        _ROOT / args.conversion_plan,
        inherited_v1_path=_ROOT / args.inherited_conversion_plan,
    )
    readiness_plan, readiness_plan_identities = load_effective_readiness_plan(
        _ROOT / args.readiness_plan,
        inherited_v1_path=_ROOT / args.inherited_readiness_plan,
    )
    structure, structure_file_sha = load_crossfit_structure(_ROOT / args.structure)
    readiness_result, readiness_result_sha = load_readiness_result(
        _ROOT / args.readiness_result
    )
    split, split_file_sha = load_split_v2(_ROOT / args.split)
    boundary = load_f0d0_boundary(_ROOT / args.f0d0_manifest)
    membership, membership_file_sha = load_membership(_ROOT / args.b2_membership)

    lineage = conversion["input_lineage"]
    if readiness_result["result_identity"] != lineage["readiness_result_identity"]:
        parser.error("readiness result identity differs")
    if readiness_result_sha != lineage["readiness_result_file_sha256"]:
        parser.error("readiness result file SHA-256 differs")
    if (
        readiness_plan_identities["v2_plan_identity"]
        != lineage["readiness_plan_v2_identity"]
    ):
        parser.error("readiness plan identity differs")
    if structure["structure_identity"] != lineage["crossfit_structure_identity"]:
        parser.error("crossfit structure identity differs")
    if structure["structure"]["sample_session_identity"] != lineage[
        "crossfit_sample_identity"
    ]:
        parser.error("crossfit sample identity differs")
    if split["split_identity"] != lineage["selected_research_split_identity"]:
        parser.error("research split identity differs")
    if membership["membership_identity"] != lineage["official_membership_identity"]:
        parser.error("official membership identity differs")
    if boundary.manifest["manifest_identity"] != lineage["f0_d0_manifest_identity"]:
        parser.error("F0-D0 manifest identity differs")
    if boundary.manifest["identities"]["corpus_identity"] != lineage[
        "f0_d0_corpus_identity"
    ]:
        parser.error("F0-D0 corpus identity differs")

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
        analysis = derive_product_conversion(
            repository_root=_ROOT,
            boundary=boundary,
            official_membership=membership,
            research_split=split,
            structure=structure,
            readiness_plan=readiness_plan,
            conversion_plan=conversion,
            readiness_result=readiness_result,
            database=database,
        )
    finally:
        database.close()

    chain = [
        {
            "link": "identification",
            "estimability": "estimable_on_existing_exploration_as_prediction_only",
            "evidence": "sealed out-of-fold log loss, D-to-L calibration, and discrimination",
            "causal": False,
        },
        {
            "link": "safe_steerability",
            "estimability": "predictive_plugin_only_not_causal",
            "evidence": "A_pos successor risk spread and three reference-policy contrasts",
            "blocking_gap": "counterfactual opponent response under a deliberately selected unseen successor",
        },
        {
            "link": "single_step_inducement",
            "estimability": "not_identified_from_human_human_observational_paths",
            "blocking_gap": "action-specific causal effect and residual argmax ranking error",
        },
        {
            "link": "multi_step_accumulation",
            "estimability": "not_identified",
            "blocking_gap": "policy-induced visitation, dependence, opponent adaptation, and repeated-opportunity stopping",
        },
        {
            "link": "redemption",
            "estimability": "not_identified",
            "blocking_gap": "learner-versus-human conversion after positional D-to-L",
            "perfect_redemption_upper_bound_only": True,
        },
        {
            "link": "product_effect",
            "estimability": "not_identified",
            "blocking_gap": "both causal inducement and redemption are missing",
        },
    ]
    conclusion = {
        "decision": "C_conversion_not_established",
        "unique_reason": (
            "predictive successor-risk differences do not identify action-specific "
            "causal inducement, multi-step policy visitation, or redemption"
        ),
        "data_method_or_concept": {
            "safe_steerability": "method_and_observational-data limitation",
            "multi_step": "missing gameplay evidence",
            "redemption": "missing learner-versus-human gameplay evidence",
            "log_loss_equivalence": "conceptual non-equivalence confirmed empirically",
        },
        "product_tiers_decidable_at_487_players": [],
        "frozen_B_not_ready_decision_changed": False,
        "new_research_question_created": False,
    }
    implementation_paths = [
        "learned_ai/evaluation/human_feature_deviation_product_conversion.py",
        "scripts/run_human_feature_deviation_product_conversion.py",
        "tests/test_human_feature_deviation_product_conversion.py",
    ]
    payload = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed_exploratory_derivation_only",
        "conversion_plan_identities": conversion_identities,
        "input_identities": {
            "readiness_plan_identities": readiness_plan_identities,
            "readiness_result_identity": readiness_result["result_identity"],
            "readiness_result_file_sha256": readiness_result_sha,
            "crossfit_structure_identity": structure["structure_identity"],
            "crossfit_structure_file_sha256": structure_file_sha,
            "research_split_identity": split["split_identity"],
            "research_split_file_sha256": split_file_sha,
            "f0d0_manifest_identity": boundary.manifest["manifest_identity"],
            "f0d0_corpus_identity": boundary.manifest["identities"][
                "corpus_identity"
            ],
            "f0d0_manifest_file_sha256": sha256_file(_ROOT / args.f0d0_manifest),
            "official_membership_identity": membership["membership_identity"],
            "official_membership_file_sha256": membership_file_sha,
            "malom": malom_snapshot,
            "implementation_files": {
                path: sha256_file(_ROOT / path) for path in implementation_paths
            },
        },
        "conversion_chain": chain,
        "analysis": analysis,
        "conclusion": conclusion,
        "claim_boundary": conversion["claim_boundary"],
        "known_biases": conversion["known_biases"],
        "access_audit": analysis["access_audit"],
    }
    sealed = write_sealed_json(
        _ROOT / args.output,
        payload,
        identity_field="result_identity",
    )
    print(sealed["result_identity"])
    print(conclusion["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
