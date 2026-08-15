#!/usr/bin/env python3
"""Freeze the player-isolated train split for feature-deviation research."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
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
    load_f0d0_boundary,
    write_sealed_json,
)
from learned_ai.evaluation.human_feature_deviation import (
    EXPECTED_B2_MEMBERSHIP_FILE_SHA256,
    FeatureDeviationError,
    build_train_internal_split,
    load_plan,
    verify_implementation_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="docs/experiments/human-feature-deviation-screen-v1.json",
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
        default=("docs/experiments/human-feature-deviation-train-split-v1.json"),
    )
    args = parser.parse_args()

    try:
        plan, _plan_file_sha = load_plan(_ROOT / args.plan)
        verify_implementation_artifacts(_ROOT, plan)
        boundary = load_f0d0_boundary(_ROOT / args.f0d0_manifest)
        membership, membership_file_sha = load_membership(_ROOT / args.b2_membership)
        if membership_file_sha != EXPECTED_B2_MEMBERSHIP_FILE_SHA256:
            raise FeatureDeviationError("official B2 membership file differs")
        payload = build_train_internal_split(boundary, membership, plan)
        sealed = write_sealed_json(
            _ROOT / args.output,
            payload,
            identity_field="split_identity",
        )
    except (
        B2FreezeError,
        F0H0Error,
        FeatureDeviationError,
        OSError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(sealed["split_identity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
