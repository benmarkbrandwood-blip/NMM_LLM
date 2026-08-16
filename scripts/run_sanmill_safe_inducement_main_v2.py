#!/usr/bin/env python3
"""Execute the once-only v2 Sanmill safe-inducement main experiment."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
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
    MAIN_POOL_SCHEMA,
    MAIN_RESULT_SCHEMA,
    SafeInducementError,
    load_main_authorization,
    load_main_plan,
    load_main_preflight,
    load_state_pool,
    run_main_experiment,
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


def _running_tgf_processes() -> int:
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq tgf.exe", "/FO", "CSV", "/NH"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        raise SafeInducementError("cannot inspect existing Sanmill processes")
    return sum(
        1
        for line in result.stdout.splitlines()
        if line.strip().lower().startswith('"tgf.exe"')
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="docs/experiments/sanmill-safe-inducement-mechanism-v2.json",
    )
    parser.add_argument(
        "--state-pool",
        default=(
            "docs/experiments/sanmill-safe-inducement-main-"
            "state-pool-v2.json"
        ),
    )
    parser.add_argument(
        "--authorization",
        default=(
            "docs/experiments/sanmill-safe-inducement-main-v2/"
            "authorization.json"
        ),
    )
    parser.add_argument(
        "--preflight",
        default=(
            "docs/evidence/sanmill-safe-inducement-main-v2-"
            "preflight-2026-08-16.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "docs/evidence/sanmill-safe-inducement-main-v2-"
            "manifest-2026-08-16.json"
        ),
    )
    parser.add_argument("--paths-config", default="data/training_paths.local.json")
    parser.add_argument(
        "--malom-manifest",
        default="data/manifests/malom-sector-corrected-v1.json",
    )
    args = parser.parse_args()

    output_path = _ROOT / args.output
    if output_path.exists():
        parser.error("main result already exists; retry or second execution forbidden")
    plan, plan_file_sha = load_main_plan(_ROOT / args.plan)
    pool, pool_file_sha = load_state_pool(
        _ROOT / args.state_pool,
        schema=MAIN_POOL_SCHEMA,
    )
    authorization, authorization_file_sha = load_main_authorization(
        _ROOT / args.authorization
    )
    preflight, preflight_file_sha = load_main_preflight(_ROOT / args.preflight)
    bindings = (
        authorization["plan"]["identity"],
        authorization["state_pool"]["identity"],
        preflight["plan_identity"],
        preflight["state_pool_identity"],
    )
    if bindings != (
        plan["plan_identity"],
        pool["pool_identity"],
        plan["plan_identity"],
        pool["pool_identity"],
    ):
        parser.error("plan, pool, authorization, and preflight differ")
    if preflight["authorization_identity"] != authorization[
        "authorization_identity"
    ]:
        parser.error("preflight authorization binding differs")

    run_output = _ROOT / str(preflight["run_output_namespace"])
    binding_path = run_output / "authorization-binding.json"
    marker_path = run_output / "measurement-started.json"
    if not run_output.is_dir() or not binding_path.is_file() or marker_path.exists():
        parser.error("fresh once-only run namespace is unavailable")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if (
        binding.get("authorization_identity")
        != authorization["authorization_identity"]
        or binding.get("preflight_identity") != preflight["preflight_identity"]
    ):
        parser.error("run namespace binding differs")
    if _running_tgf_processes() != 0:
        parser.error("a Sanmill process is already running")

    implementation_files = {
        path: sha256_file(_ROOT / path)
        for path in (
            "learned_ai/evaluation/sanmill_safe_inducement.py",
            "scripts/freeze_sanmill_safe_inducement_plan_v2.py",
            "scripts/freeze_sanmill_safe_inducement_main_pool_v2.py",
            "scripts/preflight_sanmill_safe_inducement_main_v2.py",
            "scripts/run_sanmill_safe_inducement_main_v2.py",
        )
    }
    for path, expected in preflight["implementation_files"].items():
        if implementation_files.get(path) != expected:
            parser.error(f"implementation changed after preflight: {path}")

    paths = _paths(_ROOT / args.paths_config)
    checkout = _local_path(paths.get("sanmill_training_checkout"), key="sanmill")
    malom_path = _local_path(paths.get("malom_db_path"), key="malom")
    installation = inspect_sanmill_training_installation(checkout)
    runtime = training_installation_record(
        installation,
        seed=int(plan["sanmill_contract"]["seed"]),
    )
    if runtime["identity"] != preflight["sanmill_runtime"]["identity"]:
        parser.error("Sanmill runtime changed after preflight")
    malom = verify_malom_snapshot(
        malom_path=malom_path,
        manifest_path=_ROOT / args.malom_manifest,
        full_hash=False,
    )
    if (
        malom.get("trust_level") != "sector-corrected-v1"
        or malom.get("content_sha256")
        != preflight["malom_snapshot"]["content_sha256"]
    ):
        parser.error("Malom snapshot changed after preflight")

    lock_path = _ROOT / "out/evaluation/sanmill-safe-inducement-main-v2.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SafeInducementError("another main evaluator lock exists") from exc
    try:
        os.write(descriptor, authorization["authorization_identity"].encode("ascii"))
        os.close(descriptor)
        marker_path.write_text(
            json.dumps(
                {
                    "authorization_identity": authorization[
                        "authorization_identity"
                    ],
                    "preflight_identity": preflight["preflight_identity"],
                    "source_commit": _git("rev-parse", "HEAD"),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        database = MalomDB(malom_path)
        try:
            analysis = run_main_experiment(
                installation=installation,
                database=database,
                pool=pool,
                plan=plan,
                preflight=preflight,
            )
        finally:
            database.close()
        payload = {
            "schema_version": MAIN_RESULT_SCHEMA,
            "status": "completed_once_bounded_single_step_main_experiment",
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": plan_file_sha,
            "state_pool_identity": pool["pool_identity"],
            "state_pool_membership_identity": pool["state_membership_identity"],
            "state_pool_file_sha256": pool_file_sha,
            "authorization_identity": authorization["authorization_identity"],
            "authorization_file_sha256": authorization_file_sha,
            "preflight_identity": preflight["preflight_identity"],
            "preflight_file_sha256": preflight_file_sha,
            "source_commit": _git("rev-parse", "HEAD"),
            "source_tree": _git("rev-parse", "HEAD^{tree}"),
            "sanmill_runtime": runtime,
            "malom_snapshot": malom,
            "analysis": analysis,
            "claim_boundary": plan["claim_boundary"],
            "access_audit": {
                "official_selection_content_reads": 0,
                "official_confirmation_content_reads": 0,
                "official_final_test_content_reads": 0,
                "research_confirmation_content_reads": 0,
                "source_pool_2eb04f54_reads_or_consumption": 0,
                "human_estimator_prediction_reads": 0,
                "model_loads": 0,
                "complete_games": 0,
                "training_or_weight_updates": 0,
                "database_writes": 0,
            },
            "implementation_files": implementation_files,
            "execution_policy": {
                "execution_count": 1,
                "automatic_retry_resume_or_extension": False,
                "host_interruption_recovery_authorized": False,
                "result_based_early_stop": False,
            },
        }
        sealed = write_sealed_json(
            output_path,
            payload,
            identity_field="result_identity",
        )
        (run_output / "result-binding.json").write_text(
            json.dumps(
                {
                    "result_identity": sealed["result_identity"],
                    "decision": analysis["decision"]["decision"],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    finally:
        if lock_path.exists():
            lock_path.unlink()
    print(sealed["result_identity"])
    print(analysis["decision"]["decision"])
    print(analysis["resource_use"]["aggregate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
