#!/usr/bin/env python3
"""Freeze the outcome-blind coverage audit for Sanmill-human transfer."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.human_f0h0_feasibility import write_sealed_json
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    load_crossfit_structure,
)
from learned_ai.evaluation.human_feature_deviation_product_conversion import (
    load_readiness_result,
)
from learned_ai.evaluation.sanmill_human_transfer import (
    AUDIT_SCHEMA,
    audit_coverage,
    load_sealed,
)
from learned_ai.evaluation.sanmill_safe_inducement import (
    MAIN_POOL_SCHEMA,
    load_state_pool,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool",
        default="docs/experiments/sanmill-safe-inducement-main-state-pool-v2.json",
    )
    parser.add_argument(
        "--crossfit",
        default="docs/experiments/human-feature-deviation-estimator-crossfit-v1.json",
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
        "--main-result",
        default=(
            "docs/evidence/sanmill-safe-inducement-main-v2-"
            "manifest-2026-08-16.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "docs/evidence/sanmill-human-transfer-coverage-audit-"
            "2026-08-16.json"
        ),
    )
    args = parser.parse_args()

    pool, pool_sha = load_state_pool(_ROOT / args.pool, schema=MAIN_POOL_SCHEMA)
    crossfit, crossfit_sha = load_crossfit_structure(_ROOT / args.crossfit)
    readiness, readiness_sha = load_readiness_result(_ROOT / args.readiness)
    conversion, conversion_sha = load_sealed(
        _ROOT / args.conversion,
        identity_field="result_identity",
        schema="nmm.human-feature-deviation-product-conversion-result.v1",
    )
    main_result, main_result_sha = load_sealed(
        _ROOT / args.main_result,
        identity_field="result_identity",
        schema="nmm.sanmill-safe-inducement-main-result.v2",
    )
    analysis = audit_coverage(
        pool=pool,
        crossfit=crossfit,
        readiness=readiness,
        conversion=conversion,
        main_result=main_result,
    )
    payload = {
        "schema_version": AUDIT_SCHEMA,
        "status": "completed_before_transfer_estimator_freeze",
        "method_boundary": {
            "engine_outcomes_read": False,
            "human_risk_predictions_calculated": False,
            "malom_queries": 0,
            "sanmill_queries": 0,
            "raw_game_content_reads": 0,
            "structural_fields_only": [
                "session membership",
                "side-to-move player",
                "player fold",
                "fold parameter availability",
                "A_pos successor material availability",
                "main-result state presence",
            ],
        },
        "input_identities": {
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
        },
        "analysis": analysis,
        "access_audit": {
            "official_selection_content_reads": 0,
            "official_confirmation_content_reads": 0,
            "official_final_test_content_reads": 0,
            "research_confirmation_content_reads": 0,
            "source_pool_2eb04f54_reads_or_consumption": 0,
        },
    }
    sealed = write_sealed_json(
        _ROOT / args.output, payload, identity_field="audit_identity"
    )
    print(sealed["audit_identity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
