#!/usr/bin/env python3
"""Run outcome-blind precision and player-split rebalancing analysis."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.human_f0h0_b2_freeze import (
    B2FreezeError,
    load_membership,
)
from learned_ai.evaluation.human_f0h0_feasibility import (
    F0H0Error,
    canonical_json_bytes,
    canonical_sha256,
    load_f0d0_boundary,
    write_sealed_json,
)
from learned_ai.evaluation.human_feature_deviation import (
    EXPECTED_B2_MEMBERSHIP_FILE_SHA256,
    FeatureDeviationError,
    load_research_split,
)
from learned_ai.evaluation.human_feature_deviation_design_round import (
    analyze_and_rebalance,
    load_design_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default=("docs/experiments/human-feature-deviation-design-round-v1.json"),
    )
    parser.add_argument(
        "--v1-split",
        default=("docs/experiments/human-feature-deviation-train-split-v1.json"),
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
        "--split-output",
        default=("docs/experiments/human-feature-deviation-train-split-v2.json"),
    )
    parser.add_argument(
        "--manifest-output",
        default=(
            "docs/evidence/human-feature-deviation-precision-rebalance-"
            "manifest-2026-08-15.json"
        ),
    )
    args = parser.parse_args()

    try:
        plan, plan_file_sha = load_design_plan(_ROOT / args.plan)
        v1_split, v1_split_file_sha = load_research_split(_ROOT / args.v1_split)
        boundary = load_f0d0_boundary(_ROOT / args.f0d0_manifest)
        membership, membership_file_sha = load_membership(_ROOT / args.b2_membership)
        if membership_file_sha != EXPECTED_B2_MEMBERSHIP_FILE_SHA256:
            raise FeatureDeviationError("official B2 membership file differs")
        manifest, split = analyze_and_rebalance(
            boundary=boundary,
            official_membership=membership,
            v1_split=v1_split,
            plan=plan,
        )
        split_identity = canonical_sha256(split)
        sealed_split = {**split, "split_identity": split_identity}
        split_bytes = canonical_json_bytes(sealed_split)
        manifest["input_file_sha256"] = {
            "design_round_plan": plan_file_sha,
            "v1_split": v1_split_file_sha,
            "f0d0_manifest": boundary.file_sha256,
            "b2_membership": membership_file_sha,
        }
        manifest["selected_split"] = {
            "split_identity": split_identity,
            "prospective_file_sha256": hashlib.sha256(split_bytes).hexdigest(),
        }
        written_split = write_sealed_json(
            _ROOT / args.split_output,
            split,
            identity_field="split_identity",
        )
        if written_split["split_identity"] != split_identity:
            raise FeatureDeviationError("written v2 split identity differs")
        written_manifest = write_sealed_json(
            _ROOT / args.manifest_output,
            manifest,
            identity_field="result_identity",
        )
    except (
        B2FreezeError,
        F0H0Error,
        FeatureDeviationError,
        OSError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(written_manifest["result_identity"])
    print(written_split["split_identity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
