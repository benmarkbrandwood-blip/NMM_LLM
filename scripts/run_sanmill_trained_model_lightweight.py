#!/usr/bin/env python3
"""Run one lightweight trained-model measurement after exact reproduction."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.malom_db import MalomDB
from game.board import BoardState
from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    verify_malom_snapshot,
    write_sealed_json,
)
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    RESULT_SCHEMA as READINESS_SCHEMA,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    PLAN_SCHEMA as GUIDANCE_PLAN_SCHEMA,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    POOL_SCHEMA,
    ResourceLedger,
    _matching_move,
    append_game_record as append_reproduction_record,
    load_sealed as load_guidance_sealed,
    play_game as play_reproduction_game,
    write_json_atomic,
)
from learned_ai.evaluation.sanmill_trained_model_baseline import (
    analyze_games as analyze_candidate_games,
)
from learned_ai.evaluation.sanmill_trained_model_baseline import (
    append_game_record as append_candidate_record,
)
from learned_ai.evaluation.sanmill_trained_model_baseline import (
    load_model_policies,
    play_game as play_candidate_game,
    specialist_runtime_record,
)
from learned_ai.evaluation.sanmill_trained_model_lightweight import (
    CANDIDATE_ARMS,
    EXPECTED_CANDIDATE_GAMES,
    EXPECTED_REPRODUCTION_GAMES,
    EXPECTED_TOTAL_GAMES,
    RESULT_SCHEMA,
    LightweightMeasurementError,
    candidate_schedule,
    compact_machine_record,
    exact_reproduction_gate,
    load_authorization,
    load_plan,
    reproduction_schedule,
    sha256_file,
)
from learned_ai.training.sanmill_referee import (
    inspect_sanmill_training_installation,
    training_installation_record,
)


def _git(*arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _paths(config_path: Path) -> dict[str, object]:
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LightweightMeasurementError("local path registry is not an object")
    return value


def _local_path(value: object, *, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise LightweightMeasurementError(f"local path is absent: {key}")
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _running_tgf_processes() -> int:
    completed = subprocess.run(
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
    if completed.returncode:
        raise LightweightMeasurementError("cannot inspect Sanmill processes")
    try:
        return int(completed.stdout.strip())
    except ValueError as exc:
        raise LightweightMeasurementError(
            "Sanmill process count is malformed"
        ) from exc


def _load_plain_sealed_reference(
    path: Path,
    *,
    expected_identity: str,
    expected_file_sha256: str,
) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    body = dict(value)
    identity = body.pop("result_identity", None)
    if (
        not isinstance(value, dict)
        or canonical_sha256(body) != identity
        or identity != expected_identity
        or sha256_file(path) != expected_file_sha256
    ):
        raise LightweightMeasurementError("known-answer reference differs")
    return value


def _require_runtime_equal(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    fields = (
        "identity",
        "commit",
        "tree",
        "binary_relative_path",
        "binary_sha256",
        "binary_size",
        "strict_options",
        "strict_referee",
    )
    mismatches = [field for field in fields if observed.get(field) != expected.get(field)]
    if mismatches:
        raise LightweightMeasurementError(
            f"Sanmill runtime comparability differs: {mismatches}"
        )


def _board_from_state(state: Mapping[str, Any]) -> BoardState:
    board = BoardState.new_game()
    for actions in state["logical_turns"]:
        move = _matching_move(board, actions)
        board = board.apply_move(move)
    if board.to_fen_string() != state["fen"]:
        raise LightweightMeasurementError("candidate sample replay differs")
    return board


def _candidate_load_checks(
    *,
    policies: Any,
    states: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    samples = []
    for phase in ("placement", "movement", "flying"):
        sample = next((row for row in states if row["phase"] == phase), None)
        if sample is None:
            raise LightweightMeasurementError(f"sample phase is absent: {phase}")
        samples.append(sample)
    observations = []
    for state in samples:
        board = _board_from_state(state)
        for arm in ("retained-v4-free", "active-specialists-free"):
            scorer = policies.scorer(arm)
            moves_a, scores_a, route_a = scorer.score(board)
            moves_b, scores_b, route_b = scorer.score(board)
            normalized_a = [
                (move.get("from"), move.get("to"), move.get("capture"))
                for move in moves_a
            ]
            normalized_b = [
                (move.get("from"), move.get("to"), move.get("capture"))
                for move in moves_b
            ]
            deterministic = (
                normalized_a == normalized_b
                and route_a == route_b
                and np.array_equal(scores_a, scores_b)
                and len(moves_a) > 0
            )
            if not deterministic:
                raise LightweightMeasurementError(
                    f"candidate determinism differs: {arm}/{state['phase']}"
                )
            selected = normalized_a[int(np.argmax(scores_a))]
            observations.append(
                {
                    "candidate_arm": arm,
                    "start_id": state["state_id"],
                    "source_phase": state["phase"],
                    "runtime_identity": scorer.identity,
                    "route_phase": route_a,
                    "moves": len(moves_a),
                    "score_vector_identity": canonical_sha256(scores_a.tolist()),
                    "selected_move": list(selected),
                    "second_call_exact": True,
                }
            )
    v4_scorer = policies.scorer("retained-v4-free")
    v4_manifest = v4_scorer._policy.manifest
    feature_width = int(v4_manifest["route"]["feature_width"])
    if feature_width != 134:
        raise LightweightMeasurementError("retained-v4 feature width differs")
    expected_v4 = plan["candidate_runtime"]["retained_v4"]
    expected_specialist = plan["candidate_runtime"]["active_specialists"]
    if (
        v4_scorer.identity != expected_v4["bundle"]["identity"]
        or policies.scorer("active-specialists-free").identity
        != expected_specialist["runtime_identity"]
    ):
        raise LightweightMeasurementError("candidate runtime identity differs")
    return {
        "passed": True,
        "sample_states": len(samples),
        "sample_scores": len(observations),
        "feature_width": feature_width,
        "retained_v4_bundle_identity": v4_scorer.identity,
        "active_specialist_runtime_identity": policies.scorer(
            "active-specialists-free"
        ).identity,
        "frozen_harness_module_sha256": plan["implementation_files"][
            "learned_ai/evaluation/sanmill_trained_model_baseline.py"
        ],
        "execution_uses_same_harness_scorers": True,
        "observations": observations,
        "specialist_checkpoint_sha256": {
            name: value["sha256"]
            for name, value in expected_specialist["resource_files"].items()
            if name.startswith("checkpoint_")
        },
        "product_presearch_deviation": expected_specialist[
            "product_presearch_deviation"
        ],
    }


def _database_snapshot(plan: Mapping[str, Any]) -> dict[str, Any]:
    v4 = plan["candidate_runtime"]["retained_v4"]
    active = plan["candidate_runtime"]["active_specialists"]
    paths = {
        "human_db": ROOT / str(v4["human_db"]["path"]),
        "retained_v4_specialist_db": ROOT / str(v4["specialist_db"]["path"]),
        "active_human_db": ROOT
        / str(active["resource_files"]["human_db"]["path"]),
    }
    snapshot = {}
    for name, path in paths.items():
        stat = path.stat()
        snapshot[name] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
            "journal_exists": Path(f"{path}-journal").exists(),
            "wal_exists": Path(f"{path}-wal").exists(),
            "shm_exists": Path(f"{path}-shm").exists(),
        }
    return snapshot


def _append_progress(
    path: Path,
    *,
    stage: str,
    completed: int,
    expected: int,
    total_completed: int,
    ledger: ResourceLedger,
) -> None:
    value = {
        "stage": stage,
        "stage_completed_games": completed,
        "stage_expected_games": expected,
        "total_completed_games": total_completed,
        "planned_total_games": EXPECTED_TOTAL_GAMES,
        "resources": ledger.record(),
    }
    write_json_atomic(path, value)
    if completed % 10 == 0 or completed == expected:
        print(json.dumps(value, sort_keys=True), flush=True)


def _ledger_record(path: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "records": len(records),
        "file_sha256": sha256_file(path),
        "tail_record_sha256": (
            json.loads(path.read_bytes().splitlines()[-1])["record_sha256"]
            if records
            else None
        ),
        "tracked": False,
    }


def _write_failure(path: Path, *, stage: str, error: BaseException) -> None:
    write_json_atomic(
        path,
        {
            "status": "failed_closed",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="docs/experiments/sanmill-trained-model-lightweight-v1.json",
    )
    parser.add_argument(
        "--authorization",
        default=(
            "docs/experiments/sanmill-trained-model-lightweight-v1/"
            "authorization.json"
        ),
    )
    parser.add_argument("--paths-config", default="data/training_paths.local.json")
    args = parser.parse_args()

    stage = "static-preflight"
    output: Path | None = None
    failure_path: Path | None = None
    lock_path = ROOT / "out/evaluation/sanmill-trained-model-lightweight-v1.lock"
    descriptor: int | None = None
    try:
        if _git("branch", "--show-current") != "dev":
            raise LightweightMeasurementError("formal measurement requires dev")
        if _git("status", "--short", "--untracked-files=no"):
            raise LightweightMeasurementError("tracked worktree must be clean")
        if _running_tgf_processes() != 0:
            raise LightweightMeasurementError("a Sanmill process is already running")
        plan, plan_sha = load_plan(ROOT / args.plan)
        authorization, authorization_sha = load_authorization(
            ROOT / args.authorization
        )
        if (
            authorization["plan"]["identity"] != plan["plan_identity"]
            or authorization["plan"]["file_sha256"] != plan_sha
            or authorization["resource_envelope"] != plan["resource_envelope"]
            or authorization["output_namespace"] != plan["outputs"]["namespace"]
            or not _is_ancestor(authorization["source"]["commit"], "HEAD")
        ):
            raise LightweightMeasurementError("authorization binding differs")
        implementation = {
            path: sha256_file(ROOT / path) for path in plan["implementation_files"]
        }
        if (
            implementation != plan["implementation_files"]
            or implementation != authorization["implementation_files"]
        ):
            raise LightweightMeasurementError("implementation binding differs")

        output = ROOT / str(plan["outputs"]["namespace"])
        failure_path = output / "failure.json"
        result_path = ROOT / str(plan["outputs"]["result"])
        if output.exists() or result_path.exists():
            raise LightweightMeasurementError("fresh output namespace is unavailable")
        output.mkdir(parents=True, exist_ok=False)
        progress_path = output / "progress.json"
        reproduction_ledger_path = output / "reproduction-games.jsonl"
        candidate_ledger_path = output / "candidate-games.jsonl"
        reproduction_gate_path = output / "reproduction-gate.json"
        preflight_path = output / "preflight.json"
        marker_path = output / "measurement-started.json"
        completed_path = output / "measurement-completed.json"

        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise LightweightMeasurementError(
                "another lightweight evaluator lock exists"
            ) from exc
        os.write(descriptor, authorization["authorization_identity"].encode("ascii"))
        os.close(descriptor)
        descriptor = None

        ledger = ResourceLedger(
            engine_searches=0,
            malom_queries=0,
            active_seconds_before_run=0.0,
            maximum_engine_searches=int(
                plan["resource_envelope"][
                    "internal_anomaly_engine_search_ceiling"
                ]
            ),
            maximum_malom_queries=int(
                plan["resource_envelope"]["internal_anomaly_malom_query_ceiling"]
            ),
            maximum_active_seconds=float(
                plan["resource_envelope"]["maximum_active_seconds"]
            ),
        )
        write_json_atomic(
            marker_path,
            {
                "plan_identity": plan["plan_identity"],
                "authorization_identity": authorization["authorization_identity"],
                "source_commit": _git("rev-parse", "HEAD"),
                "started_at_unix": time.time(),
                "execution_count": 1,
            },
        )

        guidance_input = plan["guidance_runtime_input"]
        guidance, guidance_sha = load_guidance_sealed(
            ROOT / guidance_input["plan_path"],
            schema=GUIDANCE_PLAN_SCHEMA,
            identity_field="plan_identity",
        )
        pool, pool_sha = load_guidance_sealed(
            ROOT / plan["start_pool"]["path"],
            schema=POOL_SCHEMA,
            identity_field="pool_identity",
        )
        readiness, readiness_sha = load_guidance_sealed(
            ROOT / guidance_input["readiness_path"],
            schema=READINESS_SCHEMA,
            identity_field="result_identity",
        )
        if (
            guidance["plan_identity"] != guidance_input["plan_identity"]
            or guidance_sha != guidance_input["plan_file_sha256"]
            or pool["pool_identity"] != plan["start_pool"]["pool_identity"]
            or pool_sha != plan["start_pool"]["pool_file_sha256"]
            or readiness["result_identity"]
            != guidance_input["readiness_identity"]
            or readiness_sha != guidance_input["readiness_file_sha256"]
        ):
            raise LightweightMeasurementError("frozen gameplay input differs")
        reference_record = plan["known_answer_reproduction"]["reference_result"]
        reference = _load_plain_sealed_reference(
            ROOT / reference_record["path"],
            expected_identity=reference_record["identity"],
            expected_file_sha256=reference_record["file_sha256"],
        )

        local_paths = _paths(ROOT / args.paths_config)
        checkout = _local_path(
            local_paths.get("sanmill_training_checkout"), key="sanmill"
        )
        malom_path = _local_path(local_paths.get("malom_db_path"), key="malom")
        installation = inspect_sanmill_training_installation(checkout)
        runtime = training_installation_record(
            installation, seed=int(plan["sanmill_contract"]["seed"])
        )
        _require_runtime_equal(runtime, plan["sanmill_contract"])
        malom = verify_malom_snapshot(
            malom_path=malom_path,
            manifest_path=ROOT / plan["malom_contract"]["manifest_path"],
            full_hash=False,
        )
        if (
            malom["trust_level"] != "sector-corrected-v1"
            or malom["content_sha256"]
            != plan["malom_contract"]["content_sha256"]
        ):
            raise LightweightMeasurementError("Malom snapshot differs")
        reproduction_rows = reproduction_schedule(
            pool,
            excluded_start_ids=plan["start_pool"]["excluded_start_ids"],
        )
        candidate_rows = candidate_schedule(
            pool,
            excluded_start_ids=plan["start_pool"]["excluded_start_ids"],
            namespace=plan["experiment"]["candidate_schedule_namespace"],
        )
        if len(reproduction_rows) + len(candidate_rows) != EXPECTED_TOTAL_GAMES:
            raise LightweightMeasurementError("planned game total differs")
        states = {str(row["state_id"]): row for row in pool["states"]}
        formal_states = [
            states[str(row["start_id"])]
            for row in candidate_rows[:: len(CANDIDATE_ARMS) * 2]
        ]
        formal_start_ids = sorted({str(row["start_id"]) for row in candidate_rows})
        if (
            canonical_sha256(formal_start_ids)
            != plan["start_pool"]["formal_membership_identity"]
        ):
            raise LightweightMeasurementError("formal start identity differs")
        before_databases = _database_snapshot(plan)
        write_json_atomic(
            preflight_path,
            {
                "status": "ready_for_known_answer_reproduction",
                "plan_identity": plan["plan_identity"],
                "authorization_identity": authorization["authorization_identity"],
                "source_commit": _git("rev-parse", "HEAD"),
                "sanmill_runtime": runtime,
                "malom_snapshot": {
                    "content_sha256": malom["content_sha256"],
                    "trust_level": malom["trust_level"],
                    "manifest_file_sha256": malom["manifest_file_sha256"],
                },
                "start_pool_identity": pool["pool_identity"],
                "formal_start_membership_identity": canonical_sha256(
                    formal_start_ids
                ),
                "reproduction_games": len(reproduction_rows),
                "candidate_games_after_gate": len(candidate_rows),
                "candidate_models_loaded": 0,
                "protected_content_reads": 0,
                "source_pool_2eb04f54_reads_or_consumption": 0,
                "database_snapshot_before": before_databases,
            },
        )

        stage = "known-answer-reproduction"
        reproduction_records: list[dict[str, Any]] = []
        previous_reproduction: str | None = None
        reproduction_database = MalomDB(malom_path)
        try:
            for index, item in enumerate(reproduction_rows, start=1):
                if index + len(candidate_rows) > int(
                    plan["resource_envelope"][
                        "authorized_literal_maximum_complete_games"
                    ]
                ):
                    raise LightweightMeasurementError("game envelope exceeded")
                record = play_reproduction_game(
                    schedule_item=item,
                    start_state=states[str(item["start_id"])],
                    plan=guidance,
                    readiness=readiness,
                    database=reproduction_database,
                    installation=installation,
                    ledger=ledger,
                )
                previous_reproduction = append_reproduction_record(
                    reproduction_ledger_path,
                    record,
                    previous_record_sha256=previous_reproduction,
                )
                reproduction_records.append(record)
                _append_progress(
                    progress_path,
                    stage=stage,
                    completed=index,
                    expected=EXPECTED_REPRODUCTION_GAMES,
                    total_completed=index,
                    ledger=ledger,
                )
        finally:
            reproduction_database.close()
        gate = exact_reproduction_gate(reproduction_records, reference)
        write_json_atomic(reproduction_gate_path, gate)
        if not gate["passed"]:
            raise LightweightMeasurementError(
                "known-answer reproduction did not match exactly"
            )

        stage = "candidate-load-checks"
        candidate_records: list[dict[str, Any]] = []
        previous_candidate: str | None = None
        candidate_database = MalomDB(malom_path, query_observer=ledger.add_malom)
        try:
            with load_model_policies(
                plan=plan,
                root=ROOT,
                malom_path=malom_path,
                malom_manifest_path=ROOT
                / plan["malom_contract"]["manifest_path"],
                ledger=ledger,
            ) as policies:
                load_checks = _candidate_load_checks(
                    policies=policies,
                    states=formal_states,
                    plan=plan,
                )
                stage = "candidate-measurement"
                for index, item in enumerate(candidate_rows, start=1):
                    total_completed = EXPECTED_REPRODUCTION_GAMES + index
                    if total_completed > EXPECTED_TOTAL_GAMES:
                        raise LightweightMeasurementError("game envelope exceeded")
                    record = play_candidate_game(
                        schedule_item=item,
                        start_state=states[str(item["start_id"])],
                        plan=plan,
                        policies=policies,
                        database=candidate_database,
                        installation=installation,
                        ledger=ledger,
                    )
                    previous_candidate = append_candidate_record(
                        candidate_ledger_path,
                        record,
                        previous_record_sha256=previous_candidate,
                    )
                    candidate_records.append(record)
                    _append_progress(
                        progress_path,
                        stage=stage,
                        completed=index,
                        expected=EXPECTED_CANDIDATE_GAMES,
                        total_completed=total_completed,
                        ledger=ledger,
                    )
        finally:
            candidate_database.close()

        stage = "analysis"
        synthetic_baseline = {
            "games": [
                *reproduction_records,
                *[
                    row
                    for row in reference["games"]
                    if row["arm"] == "full-guided"
                ],
            ]
        }
        analysis = analyze_candidate_games(
            candidate_records,
            plan=plan,
            baseline_manifest=synthetic_baseline,
            expected_start_ids=formal_start_ids,
        )
        after_databases = _database_snapshot(plan)
        if after_databases != before_databases:
            raise LightweightMeasurementError("read-only database snapshot changed")
        resources = ledger.record()
        if (
            len(reproduction_records) != EXPECTED_REPRODUCTION_GAMES
            or len(candidate_records) != EXPECTED_CANDIDATE_GAMES
            or resources["active_seconds"]
            > float(plan["resource_envelope"]["maximum_active_seconds"])
        ):
            raise LightweightMeasurementError("completed resource record differs")
        reproduction_ledger = _ledger_record(
            reproduction_ledger_path, reproduction_records
        )
        candidate_ledger = _ledger_record(candidate_ledger_path, candidate_records)
        result_payload = {
            "schema_version": RESULT_SCHEMA,
            "status": "completed_once_lightweight_internal_measurement",
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": plan_sha,
            "authorization_identity": authorization["authorization_identity"],
            "authorization_file_sha256": authorization_sha,
            "source_commit": _git("rev-parse", "HEAD"),
            "source_tree": _git("rev-parse", "HEAD^{tree}"),
            "sanmill_runtime": runtime,
            "malom_snapshot": {
                "content_sha256": malom["content_sha256"],
                "trust_level": malom["trust_level"],
                "manifest_file_sha256": malom["manifest_file_sha256"],
            },
            "start_pool_identity": pool["pool_identity"],
            "formal_start_membership_identity": canonical_sha256(
                formal_start_ids
            ),
            "known_answer_reproduction": gate,
            "candidate_load_checks": load_checks,
            "analysis": analysis,
            "machine_records": {
                "reproduction_ledger": reproduction_ledger,
                "candidate_ledger": candidate_ledger,
                "compact_records": [
                    *[
                        compact_machine_record(record)
                        for record in reproduction_records
                    ],
                    *[
                        compact_machine_record(record)
                        for record in candidate_records
                    ],
                ],
            },
            "resources": {
                **resources,
                "complete_games": EXPECTED_TOTAL_GAMES,
                "reproduction_games": EXPECTED_REPRODUCTION_GAMES,
                "candidate_games": EXPECTED_CANDIDATE_GAMES,
                "within_all_limits": True,
                "envelope": plan["resource_envelope"],
            },
            "database_read_only_audit": {
                "before": before_databases,
                "after": after_databases,
                "byte_for_byte_and_metadata_unchanged": True,
                "database_writes": 0,
            },
            "specialist_runtime": specialist_runtime_record(plan),
            "access_audit": {
                "official_selection_content_reads": 0,
                "official_confirmation_content_reads": 0,
                "official_final_test_content_reads": 0,
                "research_confirmation_content_reads": 0,
                "source_pool_2eb04f54_reads_or_consumption": 0,
                "training_or_weight_updates": 0,
                "model_fits_or_tuning": 0,
                "checkpoint_or_alias_changes": 0,
                "database_writes": 0,
            },
            "implementation_files": implementation,
            "claim_boundary": plan["claim_boundary"],
        }
        sealed = write_sealed_json(
            result_path,
            result_payload,
            identity_field="result_identity",
        )
        write_json_atomic(
            completed_path,
            {
                "result_identity": sealed["result_identity"],
                "overall_decision": analysis["overall_decision"],
                "resources": resources,
            },
        )
        print(sealed["result_identity"], flush=True)
        print(analysis["overall_decision"], flush=True)
        return 0
    except Exception as exc:
        if failure_path is not None and output is not None and output.is_dir():
            try:
                _write_failure(failure_path, stage=stage, error=exc)
            except Exception:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if lock_path.exists():
            lock_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
