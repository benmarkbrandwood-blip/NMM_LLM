#!/usr/bin/env python3
"""Run the once-only frozen safe-guidance complete-game experiment."""

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
    verify_malom_snapshot,
    write_sealed_json,
)
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    RESULT_SCHEMA as READINESS_SCHEMA,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    EXPECTED_GAMES,
    RESULT_SCHEMA,
    ResourceLedger,
    SafeGuidanceGameplayError,
    analyze_games,
    append_game_record,
    build_schedule,
    classify_induced_events,
    compact_game,
    load_authorization,
    load_plan,
    load_pool,
    load_preflight,
    load_sealed,
    play_game,
    sha256_file,
)
from learned_ai.training.sanmill_referee import (
    inspect_sanmill_training_installation,
    training_installation_record,
)


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
        "--preflight",
        default=(
            "docs/evidence/sanmill-safe-guidance-gameplay-preflight-2026-08-16.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "docs/evidence/sanmill-safe-guidance-gameplay-manifest-2026-08-16.json"
        ),
    )
    parser.add_argument(
        "--readiness-result",
        default=(
            "docs/evidence/human-feature-deviation-estimator-readiness-"
            "manifest-2026-08-15.json"
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
        parser.error("gameplay result already exists; second execution forbidden")
    plan, plan_sha = load_plan(_ROOT / args.plan)
    pool, pool_sha = load_pool(_ROOT / args.pool)
    authorization, authorization_sha = load_authorization(_ROOT / args.authorization)
    preflight, preflight_sha = load_preflight(_ROOT / args.preflight)
    if (
        authorization["plan"]["identity"] != plan["plan_identity"]
        or authorization["start_pool"]["identity"] != pool["pool_identity"]
        or preflight["plan_identity"] != plan["plan_identity"]
        or preflight["start_pool_identity"] != pool["pool_identity"]
        or preflight["authorization_identity"]
        != authorization["authorization_identity"]
    ):
        parser.error("plan, pool, authorization, and preflight bindings differ")
    run_output = _ROOT / str(preflight["run_output_namespace"])
    binding_path = run_output / "authorization-binding.json"
    marker_path = run_output / "measurement-started.json"
    completed_marker = run_output / "measurement-completed.json"
    raw_ledger = run_output / "games.jsonl"
    progress_path = run_output / "progress.json"
    if (
        not run_output.is_dir()
        or not binding_path.is_file()
        or marker_path.exists()
        or completed_marker.exists()
        or raw_ledger.exists()
        or progress_path.exists()
    ):
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
        for path in preflight["implementation_files"]
    }
    if implementation_files != preflight["implementation_files"]:
        parser.error("implementation changed after preflight")
    paths = _paths(_ROOT / args.paths_config)
    checkout = _local_path(paths.get("sanmill_training_checkout"), key="sanmill")
    malom_path = _local_path(paths.get("malom_db_path"), key="malom")
    installation = inspect_sanmill_training_installation(checkout)
    runtime = training_installation_record(
        installation, seed=int(plan["sanmill_contract"]["seed"])
    )
    if runtime["identity"] != preflight["sanmill_runtime"]["identity"]:
        parser.error("Sanmill runtime changed after preflight")
    malom = verify_malom_snapshot(
        malom_path=malom_path,
        manifest_path=_ROOT / args.malom_manifest,
        full_hash=False,
    )
    if (
        malom["trust_level"] != "sector-corrected-v1"
        or malom["content_sha256"]
        != preflight["malom_snapshot"]["content_sha256"]
    ):
        parser.error("Malom snapshot changed after preflight")
    readiness, readiness_sha = load_sealed(
        _ROOT / args.readiness_result,
        schema=READINESS_SCHEMA,
        identity_field="result_identity",
    )
    if readiness["result_identity"] != plan["input_identities"][
        "readiness_result_identity"
    ]:
        parser.error("readiness result changed after plan freeze")

    lock_path = _ROOT / "out/evaluation/sanmill-safe-guidance-gameplay-v1.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SafeGuidanceGameplayError("another gameplay evaluator lock exists") from exc
    try:
        os.write(descriptor, authorization["authorization_identity"].encode("ascii"))
        os.close(descriptor)
        marker_path.write_text(
            json.dumps(
                {
                    "authorization_identity": authorization["authorization_identity"],
                    "preflight_identity": preflight["preflight_identity"],
                    "source_commit": _git("rev-parse", "HEAD"),
                    "recovery_authorized": False,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        envelope = plan["resource_envelope"]
        prior = preflight["aggregate_resource_use_before_measurement"]
        ledger = ResourceLedger(
            engine_searches=int(prior["engine_single_step_searches"]),
            malom_queries=int(prior["malom_queries"]),
            active_seconds_before_run=float(prior["active_seconds"]),
            maximum_engine_searches=int(
                envelope["maximum_engine_single_step_searches"]
            ),
            maximum_malom_queries=int(envelope["maximum_malom_queries"]),
            maximum_active_seconds=float(envelope["maximum_active_seconds"]),
        )
        schedule = build_schedule(pool["states"])
        states = {str(row["state_id"]): row for row in pool["states"]}
        records: list[dict] = []
        previous_hash = None
        database = MalomDB(malom_path)
        try:
            for item in schedule:
                record = play_game(
                    schedule_item=item,
                    start_state=states[item["start_id"]],
                    plan=plan,
                    readiness=readiness,
                    database=database,
                    installation=installation,
                    ledger=ledger,
                )
                classify_induced_events(
                    game_record=record,
                    plan=plan,
                    database=database,
                    installation=installation,
                    ledger=ledger,
                )
                previous_hash = append_game_record(
                    raw_ledger,
                    record,
                    previous_record_sha256=previous_hash,
                )
                records.append(record)
                progress = {
                    "completed_games": len(records),
                    "expected_games": EXPECTED_GAMES,
                    "completed_starts": len(records) // 6,
                    "ledger_tail_record_sha256": previous_hash,
                    "resources": ledger.record(),
                    "automatic_resume": False,
                }
                progress_path.write_text(
                    json.dumps(progress, sort_keys=True), encoding="utf-8"
                )
                if len(records) % 10 == 0 or len(records) == EXPECTED_GAMES:
                    print(json.dumps(progress, sort_keys=True), flush=True)
        finally:
            database.close()

        analysis = analyze_games(records, plan)
        resources = ledger.record()
        if len(records) != EXPECTED_GAMES:
            raise SafeGuidanceGameplayError("gameplay execution count differs")
        compact = [compact_game(record) for record in records]
        payload = {
            "schema_version": RESULT_SCHEMA,
            "status": "completed_once_bounded_complete_game_experiment",
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": plan_sha,
            "start_pool_identity": pool["pool_identity"],
            "start_pool_membership_identity": pool["state_membership_identity"],
            "start_pool_file_sha256": pool_sha,
            "authorization_identity": authorization["authorization_identity"],
            "authorization_file_sha256": authorization_sha,
            "preflight_identity": preflight["preflight_identity"],
            "preflight_file_sha256": preflight_sha,
            "source_commit": _git("rev-parse", "HEAD"),
            "source_tree": _git("rev-parse", "HEAD^{tree}"),
            "sanmill_runtime": runtime,
            "malom_snapshot": malom,
            "readiness_result_identity": readiness["result_identity"],
            "readiness_result_file_sha256": readiness_sha,
            "raw_ledger": {
                "path": str(raw_ledger.relative_to(_ROOT)).replace("\\", "/"),
                "file_sha256": sha256_file(raw_ledger),
                "tail_record_sha256": previous_hash,
                "records": len(records),
                "tracked": False,
            },
            "analysis": analysis,
            "games": compact,
            "resource_use": {
                **resources,
                "complete_games": len(records),
                "independent_starts": len(pool["states"]),
                "resource_envelope": envelope,
                "within_all_limits": True,
            },
            "access_audit": {
                "official_selection_content_reads": 0,
                "official_confirmation_content_reads": 0,
                "official_final_test_content_reads": 0,
                "research_confirmation_content_reads": 0,
                "source_pool_2eb04f54_reads_or_consumption": 0,
                "model_loads": 0,
                "estimator_refits_or_tuning": 0,
                "training_or_weight_updates": 0,
                "database_writes": 0,
            },
            "execution_policy": {
                "execution_count": 1,
                "automatic_retry_resume_batching_or_extension": False,
                "host_interruption_recovery_authorized": False,
                "result_based_early_stop": False,
            },
            "claim_boundary": plan["claim_boundary"],
            "implementation_files": implementation_files,
        }
        sealed = write_sealed_json(
            output_path, payload, identity_field="result_identity"
        )
        completed_marker.write_text(
            json.dumps(
                {
                    "result_identity": sealed["result_identity"],
                    "decision": analysis["decision"],
                    "resources": resources,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(sealed["result_identity"], flush=True)
        print(analysis["decision"], flush=True)
        print(json.dumps(resources, sort_keys=True), flush=True)
    finally:
        if lock_path.exists():
            lock_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
