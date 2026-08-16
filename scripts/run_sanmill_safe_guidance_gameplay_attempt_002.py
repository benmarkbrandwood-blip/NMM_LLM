#!/usr/bin/env python3
"""Run the one-shot formal attempt-002 safe-guidance experiment."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

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
    REHEARSAL_RESULT_SCHEMA,
    RESULT_SCHEMA,
    ResourceLedger,
    SafeGuidanceGameplayError,
    analyze_games,
    append_game_record,
    append_resource_checkpoint,
    build_schedule,
    classify_induced_events,
    compact_game,
    load_attempt_spec,
    load_authorization,
    load_game_records,
    load_plan,
    load_pool,
    load_preflight,
    load_resource_checkpoints,
    load_sealed,
    play_game,
    select_schedule_excluding_starts,
    sha256_file,
    verify_resource_game_alignment,
    write_json_atomic,
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


def _tree_identity(path: Path) -> dict[str, Any]:
    rows = []
    for source in sorted(item for item in path.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": source.relative_to(path).as_posix(),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    return {
        "files": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "file_manifest_identity": canonical_sha256(rows),
    }


def _require_attempt_001_unchanged(spec: Mapping[str, Any]) -> None:
    for key in ("first_output_tree", "consumed_output_tree"):
        expected = spec["attempt_001_preservation"][key]
        observed = _tree_identity(_ROOT / str(expected["path"]))
        required = {
            "files": expected["files"],
            "bytes": expected["bytes"],
            "file_manifest_identity": expected["file_manifest_identity"],
        }
        if observed != required:
            raise SafeGuidanceGameplayError("attempt-001 output tree changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attempt-spec",
        default="docs/experiments/sanmill-safe-guidance-gameplay-attempt-002-v1.json",
    )
    parser.add_argument(
        "--plan", default="docs/experiments/sanmill-safe-guidance-gameplay-v1.json"
    )
    parser.add_argument(
        "--pool",
        default="docs/experiments/sanmill-safe-guidance-gameplay-start-pool-v1.json",
    )
    parser.add_argument(
        "--rehearsal-result",
        default=(
            "docs/evidence/sanmill-safe-guidance-gameplay-attempt-002-"
            "rehearsal-2026-08-16.json"
        ),
    )
    parser.add_argument(
        "--authorization",
        default=(
            "docs/experiments/sanmill-safe-guidance-gameplay-v1/"
            "attempt-002/authorization.json"
        ),
    )
    parser.add_argument(
        "--preflight",
        default=(
            "docs/evidence/sanmill-safe-guidance-gameplay-attempt-002-"
            "preflight-2026-08-16.json"
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

    if _git("branch", "--show-current") != "dev":
        parser.error("attempt-002 execution requires dev")
    if _git("status", "--short", "--untracked-files=no"):
        parser.error("tracked worktree must be clean before formal execution")
    if _running_tgf_processes() != 0:
        parser.error("a Sanmill process is already running")

    spec, spec_sha = load_attempt_spec(_ROOT / args.attempt_spec)
    plan, plan_sha = load_plan(_ROOT / args.plan)
    pool, pool_sha = load_pool(_ROOT / args.pool)
    rehearsal, rehearsal_sha = load_sealed(
        _ROOT / args.rehearsal_result,
        schema=REHEARSAL_RESULT_SCHEMA,
        identity_field="rehearsal_identity",
    )
    authorization, authorization_sha = load_authorization(_ROOT / args.authorization)
    preflight, preflight_sha = load_preflight(_ROOT / args.preflight)
    if (
        spec["plan_identity"] != plan["plan_identity"]
        or spec["start_pool_identity"] != pool["pool_identity"]
        or authorization["attempt"]["identity"] != spec["attempt_identity"]
        or authorization["attempt"]["file_sha256"] != spec_sha
        or authorization["plan"]["identity"] != plan["plan_identity"]
        or authorization["plan"]["file_sha256"] != plan_sha
        or authorization["start_pool"]["identity"] != pool["pool_identity"]
        or authorization["start_pool"]["file_sha256"] != pool_sha
        or authorization["rehearsal"]["identity"] != rehearsal["rehearsal_identity"]
        or authorization["rehearsal"]["file_sha256"] != rehearsal_sha
        or preflight["attempt_identity"] != spec["attempt_identity"]
        or preflight["authorization_identity"]
        != authorization["authorization_identity"]
        or preflight["authorization_file_sha256"] != authorization_sha
        or preflight["rehearsal_identity"] != rehearsal["rehearsal_identity"]
        or preflight["rehearsal_file_sha256"] != rehearsal_sha
    ):
        parser.error("attempt-002 execution bindings differ")
    _require_attempt_001_unchanged(spec)
    rehearsal_output = _ROOT / str(spec["rehearsal"]["run_output_namespace"])
    if _tree_identity(rehearsal_output) != preflight["rehearsal_output_tree"]:
        parser.error("attempt-002 rehearsal output changed after preflight")

    implementation_files = {
        path: sha256_file(_ROOT / path) for path in preflight["implementation_files"]
    }
    if implementation_files != preflight["implementation_files"]:
        parser.error("attempt-002 implementation changed after preflight")

    full_schedule = build_schedule(pool["states"])
    schedule = select_schedule_excluding_starts(
        full_schedule,
        excluded_start_ids=spec["formal_execution"]["excluded_start_ids"],
    )
    expected_start_ids = sorted({str(row["start_id"]) for row in schedule})
    if (
        len(schedule) != preflight["formal_games"]
        or len(expected_start_ids) != preflight["formal_starts"]
        or canonical_sha256(expected_start_ids)
        != preflight["formal_start_membership_identity"]
    ):
        parser.error("attempt-002 formal schedule changed after preflight")

    run_output = _ROOT / str(spec["formal_execution"]["run_output_namespace"])
    binding_path = run_output / "authorization-binding.json"
    marker_path = run_output / "measurement-started.json"
    completed_marker = run_output / "measurement-completed.json"
    raw_ledger = run_output / "games.jsonl"
    resource_journal = run_output / "resource-checkpoints.jsonl"
    resource_baseline_path = run_output / "resource-baseline.json"
    progress_path = run_output / "progress.json"
    result_path = _ROOT / str(spec["formal_execution"]["tracked_result"])
    forbidden_existing = (
        marker_path,
        completed_marker,
        raw_ledger,
        resource_journal,
        resource_baseline_path,
        progress_path,
        result_path,
    )
    if not run_output.is_dir() or not binding_path.is_file() or any(
        path.exists() for path in forbidden_existing
    ):
        parser.error("fresh attempt-002 measurement namespace is unavailable")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if (
        binding.get("attempt_identity") != spec["attempt_identity"]
        or binding.get("authorization_identity")
        != authorization["authorization_identity"]
        or binding.get("preflight_identity") != preflight["preflight_identity"]
        or binding.get("formal_start_membership_identity")
        != preflight["formal_start_membership_identity"]
    ):
        parser.error("attempt-002 run namespace binding differs")

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
    if (
        readiness["result_identity"]
        != preflight["guide_canary"]["readiness_result_identity"]
        or readiness_sha
        != preflight["guide_canary"]["readiness_result_file_sha256"]
    ):
        parser.error("frozen guide contract changed after preflight")

    aggregate_before = preflight["aggregate_resource_use_before_measurement"]
    envelope = spec["resource_envelope"]
    if (
        int(aggregate_before["complete_games"]) + len(schedule)
        > int(envelope["maximum_complete_games"])
        or int(aggregate_before["independent_starts"]) + len(expected_start_ids)
        > int(envelope["maximum_independent_starts"])
    ):
        parser.error("attempt-002 planned formal execution exceeds its envelope")

    lock_path = _ROOT / "out/evaluation/sanmill-safe-guidance-gameplay-attempt-002.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SafeGuidanceGameplayError("another gameplay evaluator lock exists") from exc
    try:
        os.write(descriptor, authorization["authorization_identity"].encode("ascii"))
        os.close(descriptor)
        write_json_atomic(
            marker_path,
            {
                "attempt_identity": spec["attempt_identity"],
                "authorization_identity": authorization["authorization_identity"],
                "preflight_identity": preflight["preflight_identity"],
                "source_commit": _git("rev-parse", "HEAD"),
                "recovery_authorized": False,
                "execution_count": 1,
            },
        )
        ledger = ResourceLedger(
            engine_searches=int(aggregate_before["engine_single_step_searches"]),
            malom_queries=int(aggregate_before["malom_read_only_queries"]),
            active_seconds_before_run=float(aggregate_before["active_seconds"]),
            maximum_engine_searches=int(
                envelope["maximum_engine_single_step_searches"]
            ),
            maximum_malom_queries=int(envelope["maximum_malom_queries"]),
            maximum_active_seconds=float(envelope["maximum_active_seconds"]),
        )
        formal_baseline = ledger.record()
        write_json_atomic(resource_baseline_path, formal_baseline)
        resources_before = formal_baseline
        states = {str(row["state_id"]): row for row in pool["states"]}
        records: list[dict[str, Any]] = []
        previous_game_sha: str | None = None
        previous_checkpoint_sha: str | None = None
        database = MalomDB(malom_path)
        try:
            for completion_index, item in enumerate(schedule):
                record = play_game(
                    schedule_item=item,
                    start_state=states[str(item["start_id"])],
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
                resources_after = ledger.record()
                previous_checkpoint_sha = append_resource_checkpoint(
                    resource_journal,
                    completion_index=completion_index,
                    complete_games_before=int(aggregate_before["complete_games"]),
                    game_record=record,
                    resources_before=resources_before,
                    resources_after=resources_after,
                    previous_checkpoint_sha256=previous_checkpoint_sha,
                )
                previous_game_sha = append_game_record(
                    raw_ledger,
                    record,
                    previous_record_sha256=previous_game_sha,
                )
                resources_before = resources_after
                records.append(record)
                progress = {
                    "formal_completed_games": len(records),
                    "formal_expected_games": len(schedule),
                    "attempt_completed_games": (
                        int(aggregate_before["complete_games"]) + len(records)
                    ),
                    "formal_completed_starts": len(records) // 6,
                    "resource_checkpoint_tail": previous_checkpoint_sha,
                    "game_record_tail": previous_game_sha,
                    "resources": resources_after,
                    "automatic_resume": False,
                }
                write_json_atomic(progress_path, progress)
                if len(records) % 10 == 0 or len(records) == len(schedule):
                    print(json.dumps(progress, sort_keys=True), flush=True)
        finally:
            database.close()

        resource_recovery = load_resource_checkpoints(
            resource_journal,
            expected_baseline=formal_baseline,
            complete_games_before=int(aggregate_before["complete_games"]),
        )
        game_recovery = load_game_records(raw_ledger)
        verify_resource_game_alignment(resource_recovery, game_recovery)
        if (
            resource_recovery["checkpoint_count"] != len(schedule)
            or game_recovery["record_count"] != len(schedule)
            or resource_recovery["last_resources"] != resources_before
        ):
            raise SafeGuidanceGameplayError("formal durable recovery differs")

        analysis = analyze_games(
            records,
            plan,
            expected_start_ids=expected_start_ids,
        )
        compact = [compact_game(record) for record in records]
        aggregate_resources = resource_recovery["last_resources"]
        payload = {
            "schema_version": RESULT_SCHEMA,
            "status": "completed_once_bounded_attempt_002_complete_game_experiment",
            "attempt_identity": spec["attempt_identity"],
            "attempt_file_sha256": spec_sha,
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": plan_sha,
            "start_pool_identity": pool["pool_identity"],
            "start_pool_membership_identity": pool["state_membership_identity"],
            "start_pool_file_sha256": pool_sha,
            "formal_start_membership_identity": preflight[
                "formal_start_membership_identity"
            ],
            "excluded_failed_start_ids": spec["formal_execution"][
                "excluded_start_ids"
            ],
            "authorization_identity": authorization["authorization_identity"],
            "authorization_file_sha256": authorization_sha,
            "preflight_identity": preflight["preflight_identity"],
            "preflight_file_sha256": preflight_sha,
            "rehearsal_identity": rehearsal["rehearsal_identity"],
            "rehearsal_file_sha256": rehearsal_sha,
            "source_commit": _git("rev-parse", "HEAD"),
            "source_tree": _git("rev-parse", "HEAD^{tree}"),
            "sanmill_runtime": runtime,
            "malom_snapshot": malom,
            "readiness_result_identity": readiness["result_identity"],
            "readiness_result_file_sha256": readiness_sha,
            "raw_ledger": {
                "path": str(raw_ledger.relative_to(_ROOT)).replace("\\", "/"),
                "file_sha256": game_recovery["file_sha256"],
                "tail_record_sha256": game_recovery["tail_record_sha256"],
                "records": game_recovery["record_count"],
                "tracked": False,
            },
            "resource_checkpoint_journal": {
                "path": str(resource_journal.relative_to(_ROOT)).replace("\\", "/"),
                "file_sha256": resource_recovery["file_sha256"],
                "tail_checkpoint_sha256": resource_recovery[
                    "tail_checkpoint_sha256"
                ],
                "checkpoints": resource_recovery["checkpoint_count"],
                "complete_games_after": resource_recovery[
                    "complete_games_after"
                ],
                "tracked": False,
            },
            "analysis": analysis,
            "games": compact,
            "resource_use": {
                **aggregate_resources,
                "attempt_complete_games": (
                    int(aggregate_before["complete_games"]) + len(records)
                ),
                "formal_complete_games": len(records),
                "attempt_independent_starts": (
                    int(aggregate_before["independent_starts"])
                    + len(expected_start_ids)
                ),
                "formal_independent_starts": len(expected_start_ids),
                "resource_envelope": envelope,
                "within_all_limits": True,
                "attempt_001_sunk_cost_included": False,
            },
            "attempt_001_sunk_cost_outside_new_envelope": spec[
                "sunk_cost_outside_attempt_002_envelope"
            ],
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
                "attempt_003_authorized": False,
            },
            "claim_boundary": spec["claim_boundary"],
            "implementation_files": implementation_files,
        }
        sealed = write_sealed_json(
            result_path,
            payload,
            identity_field="result_identity",
        )
        write_json_atomic(
            completed_marker,
            {
                "result_identity": sealed["result_identity"],
                "decision": analysis["decision"],
                "resources": aggregate_resources,
            },
        )
        print(sealed["result_identity"], flush=True)
        print(analysis["decision"], flush=True)
        print(json.dumps(aggregate_resources, sort_keys=True), flush=True)
    finally:
        if lock_path.exists():
            lock_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
