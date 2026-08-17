#!/usr/bin/env python3
"""Run the one-shot trained-model versus safe-random formal experiment."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.malom_db import MalomDB
from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    verify_malom_snapshot,
    write_sealed_json,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    ResourceLedger,
    append_resource_checkpoint,
    load_pool,
    load_resource_checkpoints,
    sha256_file,
    write_json_atomic,
)
from learned_ai.evaluation.sanmill_trained_model_baseline import (
    ARMS,
    RESULT_SCHEMA,
    TrainedModelBaselineError,
    analyze_games,
    append_game_record,
    build_schedule,
    compact_game,
    formal_states,
    load_attempt_authorization,
    load_attempt_spec,
    load_game_records,
    load_model_policies,
    load_plan,
    load_preflight,
    load_rehearsal,
    play_game,
    verify_resource_game_alignment,
)
from learned_ai.evaluation.sanmill_trained_model_boundary_registry import (
    coverage_contract,
    load_boundary_registry,
    verify_rehearsal_coverage,
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
        raise TrainedModelBaselineError("local path registry is not an object")
    return value


def _local_path(value: object, *, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TrainedModelBaselineError(f"local path is absent: {key}")
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
        raise TrainedModelBaselineError("cannot inspect Sanmill process count")
    return int(result.stdout.strip())


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


def _load_baseline(plan: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = _ROOT / str(plan["baseline"]["result_path"])
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    identity = body.pop("result_identity", None)
    digest = sha256_file(path)
    if (
        canonical_sha256(body) != identity
        or identity != plan["baseline"]["result_identity"]
        or digest != plan["baseline"]["result_file_sha256"]
    ):
        raise TrainedModelBaselineError("safe-random baseline identity differs")
    return value, digest


def _exposure_strata(
    *,
    records: Sequence[Mapping[str, Any]],
    baseline_manifest: Mapping[str, Any],
    exposure: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_games = [
        row for row in baseline_manifest["games"] if row["arm"] == "random-safe"
    ]
    flags = (
        "v4_training_human_db_exact_state_present",
        "v4_training_specialist_db_exact_state_present",
    )
    exposure_by_start = {
        str(row["start_id"]): row for row in exposure["rows"]
    }
    result = {}
    for flag in flags:
        groups = {}
        for value in (False, True):
            start_ids = {
                start_id
                for start_id, row in exposure_by_start.items()
                if bool(row[flag]) is value
            }
            if not start_ids:
                groups[str(value).lower()] = {
                    "starts": 0,
                    "baseline_games": 0,
                    "by_arm": {arm: {"games": 0, "score_rate": None} for arm in ARMS},
                }
                continue
            baseline_scores = [
                float(row["candidate_score"])
                for row in baseline_games
                if row["start_id"] in start_ids
            ]
            by_arm = {}
            for arm in ARMS:
                rows = [
                    row
                    for row in records
                    if row["arm"] == arm and row["start_id"] in start_ids
                ]
                scores = [float(row["candidate_score"]) for row in rows]
                by_arm[arm] = {
                    "games": len(rows),
                    "strict_wdl": {
                        "wins": sum(score == 1.0 for score in scores),
                        "draws": sum(score == 0.5 for score in scores),
                        "losses": sum(score == 0.0 for score in scores),
                    },
                    "score_rate": sum(scores) / len(scores),
                    "minus_safe_random": (
                        sum(scores) / len(scores)
                        - sum(baseline_scores) / len(baseline_scores)
                    ),
                }
            groups[str(value).lower()] = {
                "starts": len(start_ids),
                "baseline_games": len(baseline_scores),
                "baseline_score_rate": sum(baseline_scores) / len(baseline_scores),
                "by_arm": by_arm,
            }
        result[flag] = groups
    return {
        "descriptive_only": True,
        "preflight_rows_identity": exposure["rows_identity"],
        "strata": result,
        "source_game_exposure": exposure["source_game_exposure"],
        "active_specialist_training_exposure": exposure[
            "active_specialist_training_exposure"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", default="docs/experiments/sanmill-trained-model-baseline-v1.json"
    )
    parser.add_argument(
        "--attempt",
        default=(
            "docs/experiments/sanmill-trained-model-baseline-"
            "attempt-002.json"
        ),
    )
    parser.add_argument(
        "--authorization",
        default=(
            "docs/experiments/sanmill-trained-model-baseline-attempt-002/"
            "authorization.json"
        ),
    )
    parser.add_argument(
        "--rehearsal",
        default=(
            "docs/evidence/sanmill-trained-model-baseline-v1-"
            "attempt-002-rehearsal-2026-08-16.json"
        ),
    )
    parser.add_argument(
        "--preflight",
        default=(
            "docs/evidence/sanmill-trained-model-baseline-v1-"
            "attempt-002-preflight-2026-08-16.json"
        ),
    )
    parser.add_argument(
        "--pool",
        default="docs/experiments/sanmill-safe-guidance-gameplay-start-pool-v1.json",
    )
    parser.add_argument("--paths-config", default="data/training_paths.local.json")
    parser.add_argument(
        "--malom-manifest", default="data/manifests/malom-sector-corrected-v1.json"
    )
    parser.add_argument(
        "--boundary-registry",
        default=(
            "docs/experiments/"
            "sanmill-trained-model-baseline-boundary-registry-v1.json"
        ),
    )
    args = parser.parse_args()

    if _git("branch", "--show-current") != "dev":
        parser.error("formal execution requires dev")
    if _git("status", "--short", "--untracked-files=no"):
        parser.error("tracked worktree must be clean before formal execution")
    if _running_tgf_processes() != 0:
        parser.error("a Sanmill process is already running")
    if sys.getprofile() is not None:
        parser.error("formal execution forbids the rehearsal profiling hook")
    plan, plan_sha = load_plan(_ROOT / args.plan)
    attempt, attempt_sha = load_attempt_spec(_ROOT / args.attempt)
    authorization, authorization_sha = load_attempt_authorization(
        _ROOT / args.authorization
    )
    rehearsal, rehearsal_sha = load_rehearsal(_ROOT / args.rehearsal)
    preflight, preflight_sha = load_preflight(_ROOT / args.preflight)
    pool, pool_sha = load_pool(_ROOT / args.pool)
    registry, registry_sha = load_boundary_registry(_ROOT / args.boundary_registry)
    contract = coverage_contract(registry)
    if (
        attempt["plan"]["identity"] != plan["plan_identity"]
        or attempt["plan"]["file_sha256"] != plan_sha
        or authorization["attempt"]["identity"] != attempt["attempt_identity"]
        or authorization["attempt"]["file_sha256"] != attempt_sha
        or authorization["status"] != "authorized_once_measurement_unconsumed"
        or rehearsal["attempt_identity"] != attempt["attempt_identity"]
        or rehearsal["authorization_identity"]
        != authorization["authorization_identity"]
        or preflight["authorization_identity"]
        != authorization["authorization_identity"]
        or preflight["rehearsal_identity"] != rehearsal["rehearsal_identity"]
        or preflight["status"] != "ready_for_one_authorized_execution"
        or pool["pool_identity"] != plan["start_pool"]["pool_identity"]
        or pool_sha != plan["start_pool"]["pool_file_sha256"]
        or preflight["attempt_identity"] != attempt["attempt_identity"]
        or attempt["resource_envelope"] != plan["resource_envelope"]
        or attempt["boundary_registry"]["identity"]
        != registry["registry_identity"]
        or attempt["boundary_registry"]["file_sha256"] != registry_sha
        or attempt["coverage_contract"] != contract
        or authorization["boundary_registry"]["identity"]
        != registry["registry_identity"]
        or authorization["coverage_contract"]["identity"]
        != contract["coverage_contract_identity"]
        or rehearsal["boundary_registry"]["identity"]
        != registry["registry_identity"]
        or preflight["verification"]["boundary_registry"]["identity"]
        != registry["registry_identity"]
    ):
        parser.error("formal execution bindings differ")
    coverage_record = rehearsal["boundary_coverage_event_ledger"]
    dynamic_coverage = verify_rehearsal_coverage(
        _ROOT / str(coverage_record["path"]),
        registry,
    )
    if (
        dynamic_coverage["coverage_ledger_identity"]
        != coverage_record["coverage_ledger_identity"]
        or dynamic_coverage
        != preflight["verification"]["boundary_registry"][
            "rehearsal_dynamic_coverage"
        ]
    ):
        parser.error("formal rehearsal boundary coverage differs")
    if _tree_identity(
        _ROOT / str(attempt["outputs"]["rehearsal_namespace"])
    ) != preflight["rehearsal_output_tree"]:
        parser.error("rehearsal output changed after preflight")
    implementation = {
        path: sha256_file(_ROOT / path) for path in preflight["implementation_files"]
    }
    if (
        implementation != preflight["implementation_files"]
        or implementation != authorization["implementation_files"]
        or implementation != attempt["implementation_files"]
    ):
        parser.error("formal implementation changed after preflight")

    states = formal_states(
        pool,
        excluded_start_ids=plan["start_pool"]["excluded_start_ids"],
    )
    state_by_id = {str(row["state_id"]): row for row in states}
    start_ids = sorted(state_by_id)
    if canonical_sha256(start_ids) != preflight["formal_start_membership_identity"]:
        parser.error("formal start membership changed after preflight")
    schedule = build_schedule(
        states,
        namespace="sanmill-trained-model-baseline-v1-formal-game",
    )
    if len(schedule) != preflight["formal_games"]:
        parser.error("formal schedule changed after preflight")

    run_output = _ROOT / str(attempt["outputs"]["formal_output_namespace"])
    result_path = _ROOT / str(attempt["outputs"]["formal_result"])
    binding_path = run_output / "authorization-binding.json"
    marker_path = run_output / "measurement-started.json"
    completed_marker = run_output / "measurement-completed.json"
    failure_path = run_output / "execution-failure.json"
    raw_ledger = run_output / "games.jsonl"
    resource_journal = run_output / "resource-checkpoints.jsonl"
    resource_baseline_path = run_output / "resource-baseline.json"
    progress_path = run_output / "progress.json"
    if (
        not run_output.is_dir()
        or not binding_path.is_file()
        or result_path.exists()
        or any(
            path.exists()
            for path in (
                marker_path,
                completed_marker,
                failure_path,
                raw_ledger,
                resource_journal,
                resource_baseline_path,
                progress_path,
            )
        )
    ):
        parser.error("fresh formal measurement namespace is unavailable")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if (
        binding.get("plan_identity") != plan["plan_identity"]
        or binding.get("attempt_identity") != attempt["attempt_identity"]
        or binding.get("authorization_identity")
        != authorization["authorization_identity"]
        or binding.get("preflight_identity") != preflight["preflight_identity"]
        or binding.get("formal_start_membership_identity")
        != preflight["formal_start_membership_identity"]
        or binding.get("boundary_registry_identity")
        != registry["registry_identity"]
        or binding.get("rehearsal_coverage_ledger_identity")
        != dynamic_coverage["coverage_ledger_identity"]
    ):
        parser.error("formal output namespace binding differs")

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
    baseline_manifest, baseline_sha = _load_baseline(plan)

    aggregate_before = preflight["aggregate_resource_use_before_measurement"]
    envelope = plan["resource_envelope"]
    if (
        int(aggregate_before["complete_games"]) + len(schedule)
        > int(envelope["maximum_complete_games"])
        or len(start_ids) > int(envelope["maximum_reused_formal_starts"])
    ):
        parser.error("planned formal execution exceeds the authorized envelope")

    lock_path = _ROOT / "out/evaluation/sanmill-trained-model-baseline-v1.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise TrainedModelBaselineError("another gameplay evaluator lock exists") from exc
    records: list[dict[str, Any]] = []
    resources_after: dict[str, Any] | None = None
    try:
        os.write(descriptor, authorization["authorization_identity"].encode("ascii"))
        os.close(descriptor)
        write_json_atomic(
            marker_path,
            {
                "plan_identity": plan["plan_identity"],
                "attempt_identity": attempt["attempt_identity"],
                "authorization_identity": authorization["authorization_identity"],
                "preflight_identity": preflight["preflight_identity"],
                "boundary_registry_identity": registry["registry_identity"],
                "rehearsal_coverage_ledger_identity": dynamic_coverage[
                    "coverage_ledger_identity"
                ],
                "source_commit": _git("rev-parse", "HEAD"),
                "execution_count": 1,
                "recovery_authorized": False,
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
        previous_game = None
        previous_checkpoint = None
        database = MalomDB(malom_path, query_observer=ledger.add_malom)
        try:
            with load_model_policies(
                plan=plan,
                root=_ROOT,
                malom_path=malom_path,
                malom_manifest_path=_ROOT / args.malom_manifest,
                ledger=ledger,
            ) as policies:
                for completion_index, item in enumerate(schedule):
                    record = play_game(
                        schedule_item=item,
                        start_state=state_by_id[str(item["start_id"])],
                        plan=plan,
                        policies=policies,
                        database=database,
                        installation=installation,
                        ledger=ledger,
                    )
                    resources_after = ledger.record()
                    previous_checkpoint = append_resource_checkpoint(
                        resource_journal,
                        completion_index=completion_index,
                        complete_games_before=int(aggregate_before["complete_games"]),
                        game_record=record,
                        resources_before=resources_before,
                        resources_after=resources_after,
                        previous_checkpoint_sha256=previous_checkpoint,
                    )
                    previous_game = append_game_record(
                        raw_ledger,
                        record,
                        previous_record_sha256=previous_game,
                    )
                    resources_before = resources_after
                    records.append(record)
                    progress = {
                        "formal_completed_games": len(records),
                        "formal_expected_games": len(schedule),
                        "aggregate_completed_games": (
                            int(aggregate_before["complete_games"]) + len(records)
                        ),
                        "formal_completed_starts": len(records) // 8,
                        "resource_checkpoint_tail": previous_checkpoint,
                        "game_record_tail": previous_game,
                        "resources": resources_after,
                        "automatic_resume": False,
                    }
                    write_json_atomic(progress_path, progress)
                    if len(records) % 8 == 0 or len(records) == len(schedule):
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
            raise TrainedModelBaselineError("formal durable recovery differs")
        analysis = analyze_games(
            records,
            plan=plan,
            baseline_manifest=baseline_manifest,
            expected_start_ids=start_ids,
        )
        analysis["exposure_stratification"] = _exposure_strata(
            records=records,
            baseline_manifest=baseline_manifest,
            exposure=preflight["start_exposure"],
        )
        compact = [compact_game(record) for record in records]
        aggregate_resources = resource_recovery["last_resources"]
        candidate_phase_counts = {
            arm: dict(
                sorted(
                    Counter(
                        turn["candidate_choice"]["route_phase"]
                        for row in records
                        if row["arm"] == arm
                        for turn in row["turns"]
                        if turn["actor"] == "candidate"
                    ).items()
                )
            )
            for arm in ARMS
        }
        payload = {
            "schema_version": RESULT_SCHEMA,
            "status": "completed_once_bounded_trained_model_baseline_experiment",
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": plan_sha,
            "attempt_identity": attempt["attempt_identity"],
            "attempt_file_sha256": attempt_sha,
            "authorization_identity": authorization["authorization_identity"],
            "authorization_file_sha256": authorization_sha,
            "rehearsal_identity": rehearsal["rehearsal_identity"],
            "rehearsal_file_sha256": rehearsal_sha,
            "preflight_identity": preflight["preflight_identity"],
            "preflight_file_sha256": preflight_sha,
            "boundary_registry": {
                "identity": registry["registry_identity"],
                "file_sha256": registry_sha,
                "coverage_contract": contract,
                "rehearsal_coverage": dynamic_coverage,
                "profiling_enabled_during_formal": False,
            },
            "start_pool_identity": pool["pool_identity"],
            "start_pool_file_sha256": pool_sha,
            "formal_start_membership_identity": preflight[
                "formal_start_membership_identity"
            ],
            "baseline_result_identity": baseline_manifest["result_identity"],
            "baseline_result_file_sha256": baseline_sha,
            "source_commit": _git("rev-parse", "HEAD"),
            "source_tree": _git("rev-parse", "HEAD^{tree}"),
            "sanmill_runtime": runtime,
            "malom_snapshot": malom,
            "candidate_runtime": plan["candidate_runtime"],
            "candidate_route_phase_counts": candidate_phase_counts,
            "analysis": analysis,
            "games": compact,
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
            "resource_use": {
                **aggregate_resources,
                "formal_complete_games": len(records),
                "formal_reused_starts": len(start_ids),
                "aggregate_complete_games": (
                    int(aggregate_before["complete_games"]) + len(records)
                ),
                "within_all_limits": True,
                "resource_envelope": envelope,
            },
            "access_audit": {
                "official_selection_content_reads": 0,
                "official_confirmation_content_reads": 0,
                "official_final_test_content_reads": 0,
                "research_confirmation_content_reads": 0,
                "source_pool_2eb04f54_reads_or_consumption": 0,
                "candidate_policy_routes_loaded_read_only": 2,
                "model_fits_or_tuning": 0,
                "training_or_weight_updates": 0,
                "checkpoint_edits_copies_renames_or_alias_changes": 0,
                "database_writes": 0,
            },
            "claim_boundary": plan["claim_boundary"],
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
                "formal_complete_games": len(records),
                "formal_reused_starts": len(start_ids),
                "resources": aggregate_resources,
                "automatic_resume": False,
            },
        )
        print(sealed["result_identity"])
        print(json.dumps(analysis["primary"], sort_keys=True))
        print(json.dumps(aggregate_resources, sort_keys=True))
        return 0
    except BaseException as exc:
        failure = {
            "status": "failed_closed_after_measurement_marker",
            "plan_identity": plan["plan_identity"],
            "attempt_identity": attempt["attempt_identity"],
            "authorization_identity": authorization["authorization_identity"],
            "preflight_identity": preflight["preflight_identity"],
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "completed_game_records_in_memory": len(records),
            "last_resources_in_memory": resources_after,
            "automatic_retry_or_resume": False,
            "new_authorization_required": True,
        }
        try:
            write_json_atomic(failure_path, failure)
        except Exception:
            pass
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
