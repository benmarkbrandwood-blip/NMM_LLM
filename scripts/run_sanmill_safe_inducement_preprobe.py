#!/usr/bin/env python3
"""Run the frozen bounded safe-inducement Sanmill preprobe."""

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
    sha256_file,
    verify_malom_snapshot,
    write_sealed_json,
)
from learned_ai.evaluation.sanmill_safe_inducement import (
    RESULT_SCHEMA,
    load_plan,
    load_state_pool,
    run_preprobe,
)
from learned_ai.training.sanmill_referee import (
    inspect_sanmill_training_installation,
    training_installation_record,
)


def _paths(config_path: Path) -> dict[str, object]:
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("local path registry is not an object")
    return value


def _local_path(value: object, *, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"local path is absent: {key}")
    path = Path(value)
    return path if path.is_absolute() else (_ROOT / path).resolve()


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="docs/experiments/sanmill-safe-inducement-mechanism-v1.json",
    )
    parser.add_argument(
        "--state-pool",
        default=(
            "docs/experiments/sanmill-safe-inducement-preprobe-"
            "state-pool-v1.json"
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
            "docs/evidence/sanmill-safe-inducement-preprobe-"
            "manifest-2026-08-16.json"
        ),
    )
    args = parser.parse_args()

    plan, plan_file_sha = load_plan(_ROOT / args.plan)
    pool, pool_file_sha = load_state_pool(_ROOT / args.state_pool)
    expected_pool = plan["input_identities"]["state_pool"]
    if (
        expected_pool.get("pool_identity") != pool["pool_identity"]
        or expected_pool.get("file_sha256") != pool_file_sha
    ):
        parser.error("plan and frozen state pool differ")

    paths = _paths(_ROOT / args.paths_config)
    checkout = _local_path(paths.get("sanmill_training_checkout"), key="sanmill")
    malom_path = _local_path(paths.get("malom_db_path"), key="malom")
    installation = inspect_sanmill_training_installation(checkout)
    installation_record = training_installation_record(
        installation,
        seed=int(plan["sanmill_contract"]["seed"]),
    )
    if installation_record["identity"] != plan["sanmill_contract"][
        "runtime_identity"
    ]:
        parser.error("Sanmill runtime identity differs from frozen plan")
    malom = verify_malom_snapshot(
        malom_path=malom_path,
        manifest_path=_ROOT / args.malom_manifest,
        full_hash=False,
    )
    if malom.get("trust_level") != "sector-corrected-v1":
        parser.error("Malom snapshot is not sector-corrected-v1")

    database = MalomDB(malom_path)
    try:
        analysis = run_preprobe(
            installation=installation,
            database=database,
            pool=pool,
            plan=plan,
        )
    finally:
        database.close()
    payload = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed_bounded_single_step_preprobe",
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": plan_file_sha,
        "state_pool_identity": pool["pool_identity"],
        "state_pool_file_sha256": pool_file_sha,
        "source_commit": _git("rev-parse", "HEAD"),
        "source_tree": _git("rev-parse", "HEAD^{tree}"),
        "sanmill_runtime": installation_record,
        "malom_snapshot": malom,
        "analysis": analysis,
        "claim_boundary": plan["claim_boundary"],
        "access_audit": {
            "official_selection_content_reads": 0,
            "official_confirmation_content_reads": 0,
            "official_final_test_content_reads": 0,
            "research_confirmation_content_reads": 0,
            "source_pool_2eb04f54_reads_or_consumption": 0,
            "model_loads": 0,
            "complete_games": 0,
            "training_or_weight_updates": 0,
            "database_writes": 0,
        },
        "implementation_files": {
            path: sha256_file(_ROOT / path)
            for path in (
                "learned_ai/evaluation/sanmill_safe_inducement.py",
                "scripts/freeze_sanmill_safe_inducement_state_pool.py",
                "scripts/run_sanmill_safe_inducement_preprobe.py",
                "tests/test_sanmill_safe_inducement.py",
            )
        },
    }
    sealed = write_sealed_json(
        _ROOT / args.output,
        payload,
        identity_field="result_identity",
    )
    print(sealed["result_identity"])
    print(analysis["decision"]["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
