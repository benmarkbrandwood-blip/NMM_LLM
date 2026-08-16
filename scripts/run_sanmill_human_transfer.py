#!/usr/bin/env python3
"""Run the frozen zero-engine Sanmill-human transfer analysis once."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.malom_db import MalomDB
from learned_ai.evaluation.human_f0h0_feasibility import (
    verify_malom_snapshot,
    write_sealed_json,
)
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    load_crossfit_structure,
)
from learned_ai.evaluation.human_feature_deviation_product_conversion import (
    load_readiness_result,
)
from learned_ai.evaluation.sanmill_human_transfer import (
    PLAN_SCHEMA,
    RESULT_SCHEMA,
    TransferError,
    analyze_transfer,
    load_sealed,
)
from learned_ai.evaluation.sanmill_safe_inducement import (
    MAIN_POOL_SCHEMA,
    load_state_pool,
)


def _malom_path(config_path: Path) -> Path:
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransferError("local path configuration is unavailable") from exc
    raw = value.get("malom_db_path") if isinstance(value, dict) else None
    if not isinstance(raw, str) or not raw:
        raise TransferError("malom_db_path is absent")
    path = Path(raw)
    return path if path.is_absolute() else (_ROOT / path).resolve()


def _head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", default="docs/experiments/sanmill-human-transfer-v1.json"
    )
    parser.add_argument(
        "--pool",
        default="docs/experiments/sanmill-safe-inducement-main-state-pool-v2.json",
    )
    parser.add_argument(
        "--main-result",
        default=(
            "docs/evidence/sanmill-safe-inducement-main-v2-"
            "manifest-2026-08-16.json"
        ),
    )
    parser.add_argument(
        "--correction",
        default=(
            "docs/evidence/sanmill-safe-inducement-main-v2-analysis-"
            "correction-2026-08-16.json"
        ),
    )
    parser.add_argument(
        "--readiness",
        default=(
            "docs/evidence/human-feature-deviation-estimator-readiness-"
            "manifest-2026-08-15.json"
        ),
    )
    parser.add_argument(
        "--conversion",
        default=(
            "docs/evidence/human-feature-deviation-product-conversion-"
            "manifest-2026-08-15.json"
        ),
    )
    parser.add_argument(
        "--crossfit",
        default="docs/experiments/human-feature-deviation-estimator-crossfit-v1.json",
    )
    parser.add_argument("--paths-config", default="data/training_paths.local.json")
    parser.add_argument(
        "--malom-manifest",
        default="data/manifests/malom-sector-corrected-v1.json",
    )
    parser.add_argument(
        "--output",
        default="docs/evidence/sanmill-human-transfer-manifest-2026-08-16.json",
    )
    args = parser.parse_args()

    plan, plan_sha = load_sealed(
        _ROOT / args.plan, identity_field="plan_identity", schema=PLAN_SCHEMA
    )
    pool, pool_sha = load_state_pool(_ROOT / args.pool, schema=MAIN_POOL_SCHEMA)
    main_result, main_result_sha = load_sealed(
        _ROOT / args.main_result,
        identity_field="result_identity",
        schema="nmm.sanmill-safe-inducement-main-result.v2",
    )
    correction, correction_sha = load_sealed(
        _ROOT / args.correction,
        identity_field="correction_identity",
        schema="nmm.sanmill-safe-inducement-main-analysis-correction.v1",
    )
    readiness, readiness_sha = load_readiness_result(_ROOT / args.readiness)
    conversion, conversion_sha = load_sealed(
        _ROOT / args.conversion,
        identity_field="result_identity",
        schema="nmm.human-feature-deviation-product-conversion-result.v1",
    )
    crossfit, crossfit_sha = load_crossfit_structure(_ROOT / args.crossfit)

    expected = plan["input_identities"]
    actual = {
        "main_pool_identity": pool["pool_identity"],
        "main_pool_file_sha256": pool_sha,
        "crossfit_structure_identity": crossfit["structure_identity"],
        "crossfit_structure_file_sha256": crossfit_sha,
        "crossfit_sample_session_identity": crossfit["structure"][
            "sample_session_identity"
        ],
        "readiness_result_identity": readiness["result_identity"],
        "readiness_result_file_sha256": readiness_sha,
        "conversion_result_identity": conversion["result_identity"],
        "conversion_result_file_sha256": conversion_sha,
        "main_result_identity": main_result["result_identity"],
        "main_result_file_sha256": main_result_sha,
    }
    if actual != expected:
        parser.error("frozen input lineage differs")
    if correction["source_result_identity"] != main_result["result_identity"]:
        parser.error("main analysis correction binds another result")
    reproduction = conversion["analysis"]["sealed_OOF_reproduction"]
    if reproduction.get("passed") is not True or reproduction.get("refit_performed"):
        parser.error("sealed OOF reproduction evidence is absent")

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
        analysis = analyze_transfer(
            plan=plan,
            pool=pool,
            main_result=main_result,
            correction=correction,
            readiness=readiness,
            crossfit=crossfit,
            database=database,
        )
    finally:
        database.close()

    payload = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed_once_zero_engine_transfer_analysis",
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": plan_sha,
        "source_commit": _head(),
        "input_identities": {
            **actual,
            "main_analysis_correction_identity": correction["correction_identity"],
            "main_analysis_correction_file_sha256": correction_sha,
            "malom": malom_snapshot,
        },
        "frozen_estimator_reproduction_evidence": {
            "source": args.conversion,
            "result_identity": conversion["result_identity"],
            "sealed_OOF_reproduction": reproduction,
            "same_fold_parameters_and_numerical_contract_reused": True,
            "estimator_refit_performed": False,
        },
        "analysis": analysis,
        "access_audit": {
            "official_selection_content_reads": 0,
            "official_confirmation_content_reads": 0,
            "official_final_test_content_reads": 0,
            "research_confirmation_content_reads": 0,
            "source_pool_2eb04f54_reads_or_consumption": 0,
            "sanmill_queries": 0,
            "complete_games": 0,
            "policy_model_loads": 0,
            "training_or_weight_updates": 0,
            "database_writes": 0,
        },
        "claim_boundary": plan["claim_boundary"],
        "known_biases": {
            "F0_D0_history_recovery_attrition": (
                "excluded 1,751 games contain only 35 draws; retained 92,789 "
                "games contain 26,157 draws"
            ),
            "missing_verifiable_terminal_basis_games": 54_923,
            "source_domain": "observed PlayOK-like source only",
            "unreconstructable_conditions": [
                "UI orientation",
                "time control",
                "exact rules variant",
            ],
        },
        "implementation_files": [
            "learned_ai/evaluation/sanmill_human_transfer.py",
            "scripts/audit_sanmill_human_transfer_coverage.py",
            "scripts/freeze_sanmill_human_transfer_plan.py",
            "scripts/run_sanmill_human_transfer.py",
            "tests/test_sanmill_human_transfer.py",
        ],
    }
    sealed = write_sealed_json(
        _ROOT / args.output, payload, identity_field="result_identity"
    )
    print(sealed["result_identity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
