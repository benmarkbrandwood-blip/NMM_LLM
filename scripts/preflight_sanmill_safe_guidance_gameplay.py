#!/usr/bin/env python3
"""Authorize and preflight the once-only safe-guidance gameplay run."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.malom_db import MalomDB
from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    verify_malom_snapshot,
    write_sealed_json,
)
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    RESULT_SCHEMA as READINESS_SCHEMA,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    AUTHORIZATION_SCHEMA,
    PHASES,
    PREFLIGHT_SCHEMA,
    ResourceLedger,
    SafeGuidanceGameplayError,
    load_plan,
    load_pool,
    load_sealed,
    replay_start,
    run_guide_canary,
    sha256_file,
)
from learned_ai.evaluation.sanmill_safe_inducement import (
    MAIN_POOL_SCHEMA,
    run_determinism_gate,
)
from learned_ai.training.sanmill_referee import (
    SanmillTrainingGame,
    inspect_sanmill_training_installation,
    training_installation_record,
)


TRANSFER_RESULT_SCHEMA = "nmm.sanmill-human-transfer-result.v1"


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


def _paths(config_path: Path) -> dict[str, object]:
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SafeGuidanceGameplayError("local path registry is not an object")
    return value


def _local_path(value: object, *, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SafeGuidanceGameplayError(f"local path is absent: {key}")
    path = Path(value)
    return path if path.is_absolute() else (_ROOT / path).resolve()


def _run_gate(command: list[str], *, expect_success: bool = True) -> dict[str, object]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    record = {
        "command": command,
        "returncode": result.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_tail": result.stdout[-8_000:],
        "stderr_tail": result.stderr[-8_000:],
    }
    if expect_success and result.returncode:
        raise SafeGuidanceGameplayError(f"preflight gate failed: {command}")
    return record


def _running_tgf_processes() -> int:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "@(Get-Process -Name tgf -ErrorAction SilentlyContinue).Count",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        raise SafeGuidanceGameplayError("cannot inspect existing Sanmill processes")
    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise SafeGuidanceGameplayError("Sanmill process count is malformed") from exc


def _fixtures(pool: dict[str, object]) -> list[dict[str, object]]:
    fixtures = []
    states = pool["states"]
    assert isinstance(states, list)
    for phase in PHASES:
        rows = sorted(
            (row for row in states if row["phase"] == phase),
            key=lambda row: (row["logical_ply"], row["state_id"]),
        )
        eligible = [row for row in rows if not row["a_pos"][0]["board_terminal"]]
        if len(eligible) < 2:
            raise SafeGuidanceGameplayError("insufficient determinism fixtures")
        fixtures.extend(
            {
                "state_id": row["state_id"],
                "a_pos_index": 0,
                "selection": (
                    "two lowest-ply nonterminal canonical-first A_pos starts per phase"
                ),
            }
            for row in eligible[:2]
        )
    return fixtures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", default="docs/experiments/sanmill-safe-guidance-gameplay-v1.json"
    )
    parser.add_argument(
        "--pool",
        default="docs/experiments/sanmill-safe-guidance-gameplay-start-pool-v1.json",
    )
    parser.add_argument(
        "--authorization",
        default=(
            "docs/experiments/sanmill-safe-guidance-gameplay-v1/authorization.json"
        ),
    )
    parser.add_argument(
        "--preflight-output",
        default=(
            "docs/evidence/sanmill-safe-guidance-gameplay-preflight-2026-08-16.json"
        ),
    )
    parser.add_argument(
        "--run-output-dir",
        default="out/evaluation/sanmill-safe-guidance-gameplay-v1-20260816-001",
    )
    parser.add_argument(
        "--readiness-result",
        default=(
            "docs/evidence/human-feature-deviation-estimator-readiness-"
            "manifest-2026-08-15.json"
        ),
    )
    parser.add_argument(
        "--transfer-result",
        default="docs/evidence/sanmill-human-transfer-manifest-2026-08-16.json",
    )
    parser.add_argument(
        "--main-pool",
        default="docs/experiments/sanmill-safe-inducement-main-state-pool-v2.json",
    )
    parser.add_argument("--paths-config", default="data/training_paths.local.json")
    parser.add_argument(
        "--malom-manifest",
        default="data/manifests/malom-sector-corrected-v1.json",
    )
    args = parser.parse_args()

    plan_path = _ROOT / args.plan
    pool_path = _ROOT / args.pool
    authorization_path = _ROOT / args.authorization
    preflight_path = _ROOT / args.preflight_output
    run_output = _ROOT / args.run_output_dir
    if authorization_path.exists() or preflight_path.exists() or run_output.exists():
        parser.error("authorization, preflight, and run namespace must be new")
    plan, plan_sha = load_plan(plan_path)
    pool, pool_sha = load_pool(pool_path)
    if pool["plan_binding"]["plan_identity"] != plan["plan_identity"]:
        parser.error("plan and start pool differ")
    if _git("branch", "--show-current") != "dev":
        parser.error("preflight requires dev")
    if _git("status", "--short", "--untracked-files=no"):
        parser.error("tracked worktree must be clean before authorization")
    if _running_tgf_processes() != 0:
        parser.error("a Sanmill process is already running")

    source_commit = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    envelope = plan["resource_envelope"]
    authorization_payload = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": "authorized_once_gameplay_unconsumed",
        "operator": "product-owner-direct",
        "authorized_on": "2026-08-16",
        "authorization_basis": (
            "Direct product-owner authorization in the task titled Freeze and "
            "execute one safe-guidance versus random-safe gameplay experiment."
        ),
        "grant_count": 1,
        "plan": {
            "identity": plan["plan_identity"],
            "file_sha256": plan_sha,
            "tracked_file": args.plan,
        },
        "start_pool": {
            "identity": pool["pool_identity"],
            "membership_identity": pool["state_membership_identity"],
            "file_sha256": pool_sha,
            "tracked_file": args.pool,
            "starts": pool["state_count"],
        },
        "execution_scope": envelope,
        "permitted": [
            "one zero-game preflight with bounded determinism searches",
            "one exact execution of 1530 strict complete games",
            "read-only Sanmill searches and corrected Malom queries",
        ],
        "prohibited": [
            "automatic retry, resume, recovery, batching, or extension",
            "second execution",
            "training, fitting, tuning, model loading, or weight updates",
            "database writes, promotion, deployment, publication, or release",
        ],
        "consumption_rule": (
            "consumed when measurement-started.json is created before game zero"
        ),
        "host_interruption": (
            "no recovery authorized; exact missing-suffix continuation requires "
            "separate product-owner authorization"
        ),
        "source_commit_at_freeze": source_commit,
        "source_tree_at_freeze": source_tree,
    }
    authorization_path.parent.mkdir(parents=True, exist_ok=False)
    authorization = write_sealed_json(
        authorization_path,
        authorization_payload,
        identity_field="authorization_identity",
    )
    run_output.mkdir(parents=True, exist_ok=False)

    verification_started = time.perf_counter()
    python = str(_ROOT / ".venv/Scripts/python.exe")
    focused = _run_gate(
        [
            python,
            "-m",
            "pytest",
            "tests/test_sanmill_safe_guidance_gameplay.py",
            "-q",
            "--basetemp",
            str(run_output / "pytest-focused"),
        ]
    )
    mandatory = _run_gate(
        [
            python,
            "-m",
            "pytest",
            "tests/test_malom_db.py",
            "tests/test_sentinel_db_teacher.py",
            "tests/test_malom_label_provenance.py",
            "-q",
            "--basetemp",
            str(run_output / "pytest-malom"),
        ]
    )
    ruff_targets = [
        "learned_ai/evaluation/sanmill_safe_guidance_gameplay.py",
        "scripts/freeze_sanmill_safe_guidance_gameplay_plan.py",
        "scripts/freeze_sanmill_safe_guidance_gameplay_pool.py",
        "scripts/preflight_sanmill_safe_guidance_gameplay.py",
        "scripts/run_sanmill_safe_guidance_gameplay.py",
        "tests/test_sanmill_safe_guidance_gameplay.py",
    ]
    ruff = _run_gate(["ruff", "check", *ruff_targets])

    paths = _paths(_ROOT / args.paths_config)
    checkout = _local_path(paths.get("sanmill_training_checkout"), key="sanmill")
    malom_path = _local_path(paths.get("malom_db_path"), key="malom")
    installation = inspect_sanmill_training_installation(checkout)
    runtime = training_installation_record(
        installation, seed=int(plan["sanmill_contract"]["seed"])
    )
    if runtime["identity"] != plan["sanmill_contract"]["runtime_identity"]:
        raise SafeGuidanceGameplayError("Sanmill runtime identity differs")
    malom = verify_malom_snapshot(
        malom_path=malom_path,
        manifest_path=_ROOT / args.malom_manifest,
        full_hash=False,
    )
    if (
        malom["trust_level"] != "sector-corrected-v1"
        or malom["content_sha256"]
        != plan["input_identities"]["malom_content_sha256"]
    ):
        raise SafeGuidanceGameplayError("Malom snapshot differs")
    readiness, readiness_sha = load_sealed(
        _ROOT / args.readiness_result,
        schema=READINESS_SCHEMA,
        identity_field="result_identity",
    )
    transfer, transfer_sha = load_sealed(
        _ROOT / args.transfer_result,
        schema=TRANSFER_RESULT_SCHEMA,
        identity_field="result_identity",
    )
    main_pool, main_pool_sha = load_sealed(
        _ROOT / args.main_pool,
        schema=MAIN_POOL_SCHEMA,
        identity_field="pool_identity",
    )
    database = MalomDB(malom_path)
    try:
        guide_canary = run_guide_canary(
            main_pool=main_pool,
            transfer_result=transfer,
            readiness=readiness,
            database=database,
            states_per_phase=2,
        )
        if not guide_canary["passed"]:
            raise SafeGuidanceGameplayError("frozen guide canary failed")
    finally:
        database.close()

    runtime_plan = dict(plan)
    runtime_plan["determinism_gate"] = {
        "fixtures": _fixtures(pool),
        "budgets": plan["preflight_contract"]["determinism_budgets"],
    }
    engine_queries = 0

    def count_engine() -> None:
        nonlocal engine_queries
        engine_queries += 1
        if engine_queries > int(envelope["maximum_engine_single_step_searches"]):
            raise SafeGuidanceGameplayError("preflight engine ceiling exceeded")

    determinism = run_determinism_gate(
        installation=installation,
        pool=pool,
        plan=runtime_plan,
        query_counter=count_engine,
    )
    if not determinism["passed"]:
        raise SafeGuidanceGameplayError("Sanmill determinism gate failed")

    validation_ledger = ResourceLedger(
        engine_searches=engine_queries,
        malom_queries=int(pool["resource_use"]["malom_queries"])
        + int(guide_canary["malom_queries"]),
        active_seconds_before_run=float(pool["resource_use"]["construction_active_seconds"]),
        maximum_engine_searches=int(envelope["maximum_engine_single_step_searches"]),
        maximum_malom_queries=int(envelope["maximum_malom_queries"]),
        maximum_active_seconds=float(envelope["maximum_active_seconds"]),
    )
    start_clock_counts = []
    for state in pool["states"]:
        with SanmillTrainingGame(
            installation, seed=int(plan["sanmill_contract"]["seed"])
        ) as game:
            _board, strict = replay_start(game, state, validation_ledger)
        start_clock_counts.append(
            {
                "state_id": state["state_id"],
                "logical_ply_count": strict["logical_ply_count"],
                "no_capture_count": strict["no_capture_count"],
                "repetition_current_count": strict["repetition_current_count"],
                "history_sha256": strict["history_sha256"],
            }
        )
    if len(start_clock_counts) != 255:
        raise SafeGuidanceGameplayError("strict start validation count differs")

    preflight_seconds = time.perf_counter() - verification_started
    pool_resource = {
        "engine_single_step_searches": 0,
        "malom_queries": int(pool["resource_use"]["malom_queries"]),
        "active_seconds": float(pool["resource_use"]["construction_active_seconds"]),
    }
    preflight_resource = {
        "engine_single_step_searches": engine_queries,
        "malom_queries": int(guide_canary["malom_queries"]),
        "active_seconds": preflight_seconds,
    }
    aggregate = {
        "engine_single_step_searches": engine_queries,
        "malom_queries": pool_resource["malom_queries"]
        + preflight_resource["malom_queries"],
        "active_seconds": pool_resource["active_seconds"]
        + preflight_resource["active_seconds"],
        "complete_games": 0,
    }
    if (
        aggregate["engine_single_step_searches"]
        >= int(envelope["maximum_engine_single_step_searches"])
        or aggregate["malom_queries"] >= int(envelope["maximum_malom_queries"])
        or aggregate["active_seconds"] >= float(envelope["maximum_active_seconds"])
    ):
        raise SafeGuidanceGameplayError("preflight reached a resource ceiling")

    implementation_files = {
        path: sha256_file(_ROOT / path)
        for path in ruff_targets
        if not path.startswith("tests/")
    }
    preflight_payload = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "ready_for_one_authorized_execution",
        "complete_games": 0,
        "measurement_searches": 0,
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": plan_sha,
        "start_pool_identity": pool["pool_identity"],
        "start_pool_membership_identity": pool["state_membership_identity"],
        "start_pool_file_sha256": pool_sha,
        "authorization_identity": authorization["authorization_identity"],
        "source_commit": source_commit,
        "source_tree": source_tree,
        "run_output_namespace": args.run_output_dir,
        "run_output_was_absent_before_preflight": True,
        "verification": {
            "focused_pytest": focused,
            "mandatory_malom_db_teacher_provenance": mandatory,
            "task_scope_ruff": ruff,
        },
        "sanmill_runtime": runtime,
        "malom_snapshot": malom,
        "determinism": determinism,
        "guide_canary": {
            **guide_canary,
            "readiness_result_identity": readiness["result_identity"],
            "readiness_result_file_sha256": readiness_sha,
            "transfer_result_identity": transfer["result_identity"],
            "transfer_result_file_sha256": transfer_sha,
            "main_pool_identity": main_pool["pool_identity"],
            "main_pool_file_sha256": main_pool_sha,
        },
        "strict_start_validation": {
            "starts": len(start_clock_counts),
            "all_nonterminal": True,
            "all_histories_replayed": True,
            "clock_records_identity": canonical_sha256(start_clock_counts),
        },
        "resource_components": {
            "pool_construction": pool_resource,
            "preflight": preflight_resource,
        },
        "aggregate_resource_use_before_measurement": aggregate,
        "protected_access": {
            "guard_test_executed_and_failed_closed": True,
            "official_selection_content_reads": 0,
            "official_confirmation_content_reads": 0,
            "official_final_test_content_reads": 0,
            "research_confirmation_content_reads": 0,
            "source_pool_2eb04f54_reads_or_consumption": 0,
        },
        "forbidden_operations": {
            "complete_games": 0,
            "model_loads": 0,
            "estimator_fits_or_tuning": 0,
            "training_or_weight_updates": 0,
            "database_writes": 0,
        },
        "historical_sanmill_checkout_route": {
            "used_for_this_experiment": False,
            "training_checkout_used_instead": True,
            "known_fail_closed_drift_not_hidden_or_repaired": True,
            "prior_record": (
                "sanmill-human-transfer-2026-08-16: 63 passes and seven "
                "fail-closed moving-checkout integration failures"
            ),
        },
        "implementation_files": implementation_files,
    }
    preflight = write_sealed_json(
        preflight_path, preflight_payload, identity_field="preflight_identity"
    )
    (run_output / "authorization-binding.json").write_text(
        json.dumps(
            {
                "authorization_identity": authorization["authorization_identity"],
                "preflight_identity": preflight["preflight_identity"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(authorization["authorization_identity"])
    print(preflight["preflight_identity"])
    print(aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
