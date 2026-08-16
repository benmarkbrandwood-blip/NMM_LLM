#!/usr/bin/env python3
"""Run the one-shot attempt-002 zero-measurement preflight."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    RESULT_SCHEMA as READINESS_SCHEMA,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    AUTHORIZATION_SCHEMA,
    PHASES,
    PREFLIGHT_SCHEMA,
    REHEARSAL_RESULT_SCHEMA,
    ResourceLedger,
    SafeGuidanceGameplayError,
    build_schedule,
    load_attempt_spec,
    load_game_records,
    load_plan,
    load_pool,
    load_resource_checkpoints,
    load_sealed,
    replay_start,
    run_guide_canary,
    select_schedule_excluding_starts,
    sha256_file,
    verify_resource_game_alignment,
    write_json_atomic,
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


def _run_gate(command: list[str]) -> dict[str, object]:
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
    if result.returncode:
        raise SafeGuidanceGameplayError(f"attempt-002 preflight gate failed: {command}")
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


def _fixtures(states: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    fixtures: list[dict[str, object]] = []
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
                    "two lowest-ply nonterminal canonical-first A_pos starts "
                    "per phase after the frozen attempt-002 exclusion"
                ),
            }
            for row in eligible[:2]
        )
    return fixtures


def _authorization_payload(
    *,
    spec: Mapping[str, Any],
    spec_sha: str,
    plan: Mapping[str, Any],
    plan_sha: str,
    pool: Mapping[str, Any],
    pool_sha: str,
    rehearsal: Mapping[str, Any],
    rehearsal_sha: str,
    source_commit: str,
    source_tree: str,
) -> dict[str, Any]:
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": "authorized_once_gameplay_unconsumed",
        "operator": "product-owner-direct",
        "grant_count": 1,
        "authorized_on": "2026-08-16",
        "authorization_basis": (
            "Direct product-owner authorization for one new attempt-002, "
            "conditional on the frozen rehearsal and all four preconditions."
        ),
        "attempt": {
            "identity": spec["attempt_identity"],
            "file_sha256": spec_sha,
            "tracked_file": (
                "docs/experiments/"
                "sanmill-safe-guidance-gameplay-attempt-002-v1.json"
            ),
        },
        "plan": {
            "identity": plan["plan_identity"],
            "file_sha256": plan_sha,
            "tracked_file": "docs/experiments/sanmill-safe-guidance-gameplay-v1.json",
        },
        "start_pool": {
            "identity": pool["pool_identity"],
            "membership_identity": pool["state_membership_identity"],
            "file_sha256": pool_sha,
            "tracked_file": (
                "docs/experiments/"
                "sanmill-safe-guidance-gameplay-start-pool-v1.json"
            ),
            "frozen_starts": len(pool["states"]),
            "excluded_start_ids": spec["formal_execution"]["excluded_start_ids"],
            "formal_starts": spec["formal_execution"]["starts"],
        },
        "rehearsal": {
            "identity": rehearsal["rehearsal_identity"],
            "file_sha256": rehearsal_sha,
            "status": rehearsal["status"],
            "complete_games": rehearsal["resource_use"]["complete_games"],
            "independent_starts": rehearsal["resource_use"]["independent_starts"],
            "formal_result_eligibility": False,
        },
        "execution_scope": spec["resource_envelope"],
        "permitted": [
            "the already completed four-game non-evidence technical rehearsal",
            "one zero-measurement attempt-002 preflight",
            "one exact 1524-game formal execution after all gates pass",
            "read-only Sanmill searches and sector-corrected-v1 Malom queries",
        ],
        "prohibited": [
            "automatic retry, resume, recovery, batching, or extension",
            "attempt-003 or a second formal attempt-002 execution",
            "training, fitting, tuning, model loading, or weight updates",
            "database writes, promotion, deployment, publication, or release",
        ],
        "consumption_rule": (
            "consumed when the fresh attempt-002 measurement-started marker is "
            "created before the first formal game"
        ),
        "host_interruption": (
            "no recovery authorized; a semantics-identical missing-suffix "
            "continuation requires a new direct authorization"
        ),
        "attempt_001_sunk_cost": spec["sunk_cost_outside_attempt_002_envelope"],
        "source_commit_at_freeze": source_commit,
        "source_tree_at_freeze": source_tree,
    }


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
        "--authorization-output",
        default=(
            "docs/experiments/sanmill-safe-guidance-gameplay-v1/"
            "attempt-002/authorization.json"
        ),
    )
    parser.add_argument(
        "--preflight-output",
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

    spec_path = _ROOT / args.attempt_spec
    plan_path = _ROOT / args.plan
    pool_path = _ROOT / args.pool
    rehearsal_path = _ROOT / args.rehearsal_result
    authorization_path = _ROOT / args.authorization_output
    preflight_path = _ROOT / args.preflight_output
    if authorization_path.exists() or preflight_path.exists():
        parser.error("attempt-002 authorization or preflight already exists")
    if _git("branch", "--show-current") != "dev":
        parser.error("attempt-002 preflight requires dev")
    if _git("status", "--short", "--untracked-files=no"):
        parser.error("tracked worktree must be clean before attempt-002 preflight")
    if _running_tgf_processes() != 0:
        parser.error("a Sanmill process is already running")

    spec, spec_sha = load_attempt_spec(spec_path)
    plan, plan_sha = load_plan(plan_path)
    pool, pool_sha = load_pool(pool_path)
    rehearsal, rehearsal_sha = load_sealed(
        rehearsal_path,
        schema=REHEARSAL_RESULT_SCHEMA,
        identity_field="rehearsal_identity",
    )
    if (
        spec["plan_identity"] != plan["plan_identity"]
        or spec["plan_file_sha256"] != plan_sha
        or spec["start_pool_identity"] != pool["pool_identity"]
        or spec["start_pool_membership_identity"]
        != pool["state_membership_identity"]
        or spec["start_pool_file_sha256"] != pool_sha
        or rehearsal["attempt_identity"] != spec["attempt_identity"]
        or rehearsal["attempt_file_sha256"] != spec_sha
        or rehearsal["status"] != "passed_non_evidence_technical_rehearsal"
        or rehearsal["formal_result_eligibility"] is not False
    ):
        parser.error("attempt-002 frozen bindings or rehearsal differ")
    _require_attempt_001_unchanged(spec)

    rehearsal_output = _ROOT / str(spec["rehearsal"]["run_output_namespace"])
    rehearsal_tree = _tree_identity(rehearsal_output)
    rehearsal_baseline = json.loads(
        (rehearsal_output / "resource-baseline.json").read_text(encoding="utf-8")
    )
    rehearsal_resources = load_resource_checkpoints(
        rehearsal_output / "resource-checkpoints.jsonl",
        expected_baseline=rehearsal_baseline,
        complete_games_before=0,
    )
    rehearsal_games = load_game_records(rehearsal_output / "rehearsal-games.jsonl")
    verify_resource_game_alignment(rehearsal_resources, rehearsal_games)
    if (
        rehearsal_resources["checkpoint_count"] != 4
        or rehearsal_games["record_count"] != 4
        or rehearsal_resources["last_resources"]
        != {
            "engine_single_step_searches": rehearsal["resource_use"][
                "engine_single_step_searches"
            ],
            "malom_read_only_queries": rehearsal["resource_use"][
                "malom_read_only_queries"
            ],
            "active_seconds": rehearsal["resource_use"]["active_seconds"],
        }
    ):
        parser.error("rehearsal durable artifacts differ")

    full_schedule = build_schedule(pool["states"])
    formal_schedule = select_schedule_excluding_starts(
        full_schedule,
        excluded_start_ids=spec["formal_execution"]["excluded_start_ids"],
    )
    formal_start_ids = sorted({str(row["start_id"]) for row in formal_schedule})
    formal_states = [
        row for row in pool["states"] if str(row["state_id"]) in formal_start_ids
    ]
    if (
        len(formal_states) != spec["formal_execution"]["starts"]
        or len(formal_schedule) != spec["formal_execution"]["games"]
        or spec["resource_envelope"]["planned_total_complete_games"]
        != 4 + len(formal_schedule)
        or spec["resource_envelope"]["planned_total_independent_starts"]
        != 2 + len(formal_states)
    ):
        parser.error("attempt-002 formal schedule or envelope differs")

    run_output = _ROOT / str(spec["formal_execution"]["run_output_namespace"])
    if run_output.exists():
        parser.error("attempt-002 formal output namespace is not fresh")
    run_output.mkdir(parents=True, exist_ok=False)
    verification_started = time.perf_counter()
    python = str(_ROOT / ".venv/Scripts/python.exe")
    regression_nodes = [
        "test_terminal_contract_requires_nested_portable_outcome",
        "test_logical_search_contract_rejects_wrong_budget",
        "test_game_record_rejects_top_level_nested_winner_mismatch",
        "test_game_record_requires_inducement_decomposition_before_analysis",
        "test_resource_journal_survives_abnormal_subprocess_exit",
        "test_resource_and_game_journals_recover_the_same_completed_game",
        "test_protected_guard_raises_before_any_content_producer",
    ]
    regression = _run_gate(
        [
            python,
            "-m",
            "pytest",
            *[
                "tests/test_sanmill_safe_guidance_gameplay.py::" + node
                for node in regression_nodes
            ],
            "-q",
            "--basetemp",
            str(run_output / "pytest-contract-regressions"),
        ]
    )
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
        "scripts/preflight_sanmill_safe_guidance_gameplay_attempt_002.py",
        "scripts/run_sanmill_safe_guidance_gameplay_attempt_002.py",
        "scripts/rehearse_sanmill_safe_guidance_gameplay_attempt_002.py",
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
    if (
        readiness["result_identity"]
        != plan["input_identities"]["readiness_result_identity"]
        or readiness_sha
        != plan["input_identities"]["readiness_result_file_sha256"]
        or transfer["result_identity"]
        != plan["input_identities"]["transfer_result_identity"]
        or transfer_sha != plan["input_identities"]["transfer_result_file_sha256"]
    ):
        raise SafeGuidanceGameplayError("frozen guide inputs differ")

    database = MalomDB(malom_path)
    try:
        guide_canary = run_guide_canary(
            main_pool=main_pool,
            transfer_result=transfer,
            readiness=readiness,
            database=database,
            states_per_phase=2,
        )
    finally:
        database.close()
    if (
        guide_canary["passed"] is not True
        or int(guide_canary["malom_queries"]) != 1_000
    ):
        raise SafeGuidanceGameplayError("frozen guide canary failed")

    runtime_plan = dict(plan)
    runtime_plan["determinism_gate"] = {
        "fixtures": _fixtures(formal_states),
        "budgets": plan["preflight_contract"]["determinism_budgets"],
    }
    engine_queries = 0

    def count_engine() -> None:
        nonlocal engine_queries
        engine_queries += 1
        if engine_queries + int(
            rehearsal["resource_use"]["engine_single_step_searches"]
        ) > int(spec["resource_envelope"]["maximum_engine_single_step_searches"]):
            raise SafeGuidanceGameplayError("preflight engine ceiling exceeded")

    determinism = run_determinism_gate(
        installation=installation,
        pool={**pool, "states": formal_states},
        plan=runtime_plan,
        query_counter=count_engine,
    )
    if determinism["passed"] is not True:
        raise SafeGuidanceGameplayError("Sanmill determinism gate failed")

    elapsed_before_replay = time.perf_counter() - verification_started
    validation_ledger = ResourceLedger(
        engine_searches=(
            int(rehearsal["resource_use"]["engine_single_step_searches"])
            + engine_queries
        ),
        malom_queries=(
            int(rehearsal["resource_use"]["malom_read_only_queries"])
            + int(guide_canary["malom_queries"])
        ),
        active_seconds_before_run=(
            float(rehearsal["resource_use"]["active_seconds"])
            + elapsed_before_replay
        ),
        maximum_engine_searches=int(
            spec["resource_envelope"]["maximum_engine_single_step_searches"]
        ),
        maximum_malom_queries=int(spec["resource_envelope"]["maximum_malom_queries"]),
        maximum_active_seconds=float(
            spec["resource_envelope"]["maximum_active_seconds"]
        ),
    )
    strict_starts = []
    for state in formal_states:
        with SanmillTrainingGame(
            installation, seed=int(plan["sanmill_contract"]["seed"])
        ) as game:
            _board, strict = replay_start(game, state, validation_ledger)
        strict_starts.append(
            {
                "state_id": state["state_id"],
                "logical_ply_count": strict["logical_ply_count"],
                "no_capture_count": strict["no_capture_count"],
                "repetition_current_count": strict["repetition_current_count"],
                "history_sha256": strict["history_sha256"],
            }
        )
    if len(strict_starts) != 254:
        raise SafeGuidanceGameplayError("formal strict-start validation count differs")

    preflight_seconds = time.perf_counter() - verification_started
    aggregate = {
        "engine_single_step_searches": (
            int(rehearsal["resource_use"]["engine_single_step_searches"])
            + engine_queries
        ),
        "malom_read_only_queries": (
            int(rehearsal["resource_use"]["malom_read_only_queries"])
            + int(guide_canary["malom_queries"])
        ),
        "active_seconds": (
            float(rehearsal["resource_use"]["active_seconds"])
            + preflight_seconds
        ),
        "complete_games": 4,
        "independent_starts": 2,
    }
    envelope = spec["resource_envelope"]
    if (
        aggregate["engine_single_step_searches"]
        >= int(envelope["maximum_engine_single_step_searches"])
        or aggregate["malom_read_only_queries"]
        >= int(envelope["maximum_malom_queries"])
        or aggregate["active_seconds"] >= float(envelope["maximum_active_seconds"])
        or aggregate["complete_games"] + len(formal_schedule)
        > int(envelope["maximum_complete_games"])
        or aggregate["independent_starts"] + len(formal_states)
        > int(envelope["maximum_independent_starts"])
    ):
        raise SafeGuidanceGameplayError("attempt-002 preflight reached an envelope")

    source_commit = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    implementation_files = {
        path: sha256_file(_ROOT / path)
        for path in ruff_targets
        if not path.startswith("tests/")
    }
    authorization = write_sealed_json(
        authorization_path,
        _authorization_payload(
            spec=spec,
            spec_sha=spec_sha,
            plan=plan,
            plan_sha=plan_sha,
            pool=pool,
            pool_sha=pool_sha,
            rehearsal=rehearsal,
            rehearsal_sha=rehearsal_sha,
            source_commit=source_commit,
            source_tree=source_tree,
        ),
        identity_field="authorization_identity",
    )
    authorization_sha = sha256_file(authorization_path)
    preflight_payload = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "ready_for_one_authorized_execution",
        "attempt_number": 2,
        "complete_games": 0,
        "measurement_searches": 0,
        "attempt_identity": spec["attempt_identity"],
        "attempt_file_sha256": spec_sha,
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": plan_sha,
        "start_pool_identity": pool["pool_identity"],
        "start_pool_membership_identity": pool["state_membership_identity"],
        "start_pool_file_sha256": pool_sha,
        "formal_excluded_start_ids": spec["formal_execution"][
            "excluded_start_ids"
        ],
        "formal_start_membership_identity": canonical_sha256(formal_start_ids),
        "formal_starts": len(formal_states),
        "formal_games": len(formal_schedule),
        "authorization_identity": authorization["authorization_identity"],
        "authorization_file_sha256": authorization_sha,
        "rehearsal_identity": rehearsal["rehearsal_identity"],
        "rehearsal_file_sha256": rehearsal_sha,
        "rehearsal_output_tree": rehearsal_tree,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "run_output_namespace": spec["formal_execution"]["run_output_namespace"],
        "run_output_was_absent_before_preflight": True,
        "verification": {
            "contract_and_crash_regressions": regression,
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
            "starts": len(strict_starts),
            "excluded_failed_start_replayed": False,
            "all_nonterminal": True,
            "all_histories_replayed": True,
            "clock_records_identity": canonical_sha256(strict_starts),
        },
        "resource_components": {
            "non_evidence_rehearsal": rehearsal["resource_use"],
            "zero_measurement_preflight": {
                "engine_single_step_searches": engine_queries,
                "malom_read_only_queries": int(guide_canary["malom_queries"]),
                "active_seconds": preflight_seconds,
                "complete_games": 0,
            },
        },
        "aggregate_resource_use_before_measurement": aggregate,
        "attempt_001_sunk_cost_outside_new_envelope": spec[
            "sunk_cost_outside_attempt_002_envelope"
        ],
        "protected_access": {
            "guard_test_executed_and_failed_closed": True,
            "official_selection_content_reads": 0,
            "official_confirmation_content_reads": 0,
            "official_final_test_content_reads": 0,
            "research_confirmation_content_reads": 0,
            "source_pool_2eb04f54_reads_or_consumption": 0,
        },
        "forbidden_operations": {
            "formal_complete_games": 0,
            "measurement_marker_created": False,
            "model_loads": 0,
            "estimator_fits_or_tuning": 0,
            "training_or_weight_updates": 0,
            "database_writes": 0,
        },
        "historical_sanmill_checkout_route": {
            "used_for_this_experiment": False,
            "training_checkout_used_instead": True,
            "known_fail_closed_drift_not_hidden_or_repaired": True,
        },
        "execution_policy": spec["execution_policy"],
        "implementation_files": implementation_files,
    }
    preflight = write_sealed_json(
        preflight_path,
        preflight_payload,
        identity_field="preflight_identity",
    )
    write_json_atomic(
        run_output / "authorization-binding.json",
        {
            "attempt_identity": spec["attempt_identity"],
            "authorization_identity": authorization["authorization_identity"],
            "preflight_identity": preflight["preflight_identity"],
            "formal_start_membership_identity": canonical_sha256(formal_start_ids),
        },
    )
    print(authorization["authorization_identity"])
    print(preflight["preflight_identity"])
    print(json.dumps(aggregate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
