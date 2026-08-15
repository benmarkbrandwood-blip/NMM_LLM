#!/usr/bin/env python3
"""Run the frozen 128-game feature-deviation exploration pilot only."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.malom_db import MalomDB
from learned_ai.evaluation.human_f0h0_b2_freeze import (
    B2FreezeError,
    load_membership,
)
from learned_ai.evaluation.human_f0h0_feasibility import (
    F0H0Error,
    load_f0d0_boundary,
    verify_malom_snapshot,
    write_sealed_json,
)
from learned_ai.evaluation.human_feature_deviation import (
    EXPECTED_B2_MEMBERSHIP_FILE_SHA256,
    FeatureDeviationError,
    load_plan,
    load_research_split,
    run_exploration,
    verify_implementation_artifacts,
)


def _malom_path(config_path: Path) -> Path:
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeatureDeviationError("local path configuration is unavailable") from exc
    raw = value.get("malom_db_path") if isinstance(value, dict) else None
    if not isinstance(raw, str) or not raw:
        raise FeatureDeviationError("malom_db_path is absent")
    path = Path(raw)
    return path if path.is_absolute() else (_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="docs/experiments/human-feature-deviation-screen-v1.json",
    )
    parser.add_argument(
        "--split",
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
        "--paths-config",
        default="data/training_paths.local.json",
    )
    parser.add_argument(
        "--malom-manifest",
        default="data/manifests/malom-sector-corrected-v1.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "docs/evidence/human-feature-deviation-exploration-manifest-2026-08-15.json"
        ),
    )
    args = parser.parse_args()

    database: MalomDB | None = None
    try:
        plan, _plan_file_sha = load_plan(_ROOT / args.plan)
        verify_implementation_artifacts(_ROOT, plan)
        split, _split_file_sha = load_research_split(_ROOT / args.split)
        boundary = load_f0d0_boundary(_ROOT / args.f0d0_manifest)
        membership, membership_file_sha = load_membership(_ROOT / args.b2_membership)
        if membership_file_sha != EXPECTED_B2_MEMBERSHIP_FILE_SHA256:
            raise FeatureDeviationError("official B2 membership file differs")
        malom_path = _malom_path(_ROOT / args.paths_config)
        malom_snapshot = verify_malom_snapshot(
            malom_path=malom_path,
            manifest_path=_ROOT / args.malom_manifest,
            full_hash=False,
        )
        database = MalomDB(malom_path)
        payload = run_exploration(
            repository_root=_ROOT,
            boundary=boundary,
            official_membership=membership,
            research_split=split,
            plan=plan,
            database=database,
            malom_snapshot=malom_snapshot,
        )
        sealed = write_sealed_json(
            _ROOT / args.output,
            payload,
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
    finally:
        if database is not None:
            database.close()
    print(sealed["result_identity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
