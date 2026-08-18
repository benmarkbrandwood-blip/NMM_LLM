#!/usr/bin/env python3
"""Calibrate and run the frozen ``main`` classical-search measurement."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.malom_db import MalomDB
from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    verify_malom_snapshot,
    write_sealed_json,
)
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    RESULT_SCHEMA as READINESS_SCHEMA,
)
from learned_ai.evaluation.sanmill_classical_search_strength import (
    AUTHORIZATION_SCHEMA,
    CALIBRATION_PLAN_SCHEMA,
    CALIBRATION_RESULT_SCHEMA,
    PLAN_SCHEMA,
    RESULT_SCHEMA,
    ClassicalSearchStrengthError,
    ProductMainRuntime,
    analyze_games,
    board_from_state,
    calibration_summary,
    compact_game,
    exact_subset_gate,
    play_classical_game,
    prior_scores_by_start,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    PLAN_SCHEMA as GUIDANCE_PLAN_SCHEMA,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    POOL_SCHEMA,
    ResourceLedger,
    append_game_record as append_reproduction_record,
    build_schedule as build_guidance_schedule,
    load_sealed as load_guidance_sealed,
    play_game as play_reproduction_game,
    select_schedule_excluding_starts,
    sha256_file,
    write_json_atomic,
)
from learned_ai.training.sanmill_referee import (
    inspect_sanmill_training_installation,
    training_installation_record,
)
from learned_ai.training.run_contract import canonical_json_bytes


def _git(*arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _load_sealed(
    path: Path, *, schema: str, identity_field: str
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise ClassicalSearchStrengthError(f"sealed schema differs: {path}")
    body = dict(value)
    identity = body.pop(identity_field, None)
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise ClassicalSearchStrengthError(f"sealed identity differs: {path}")
    return value, sha256_file(path)


def _local_paths(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ClassicalSearchStrengthError("local path registry is not an object")
    return value


def _local_path(value: Any, *, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ClassicalSearchStrengthError(f"local path is absent: {key}")
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


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
        raise ClassicalSearchStrengthError("cannot inspect Sanmill processes")
    return int(result.stdout.strip())


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
        raise ClassicalSearchStrengthError(
            f"Sanmill runtime comparability differs: {mismatches}"
        )


def _append_jsonl_record(
    path: Path,
    record: Mapping[str, Any],
    *,
    previous_record_sha256: str | None,
) -> str:
    body = dict(record)
    body["previous_record_sha256"] = previous_record_sha256
    digest = canonical_sha256(body)
    wrapper = {**body, "record_sha256": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(wrapper) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return digest


def _load_plain_reference(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    body = dict(value)
    identity = body.pop("result_identity", None)
    if (
        not isinstance(value, dict)
        or canonical_sha256(body) != identity
        or identity != contract["identity"]
        or sha256_file(path) != contract["file_sha256"]
    ):
        raise ClassicalSearchStrengthError("known-answer reference differs")
    return value


def _runtime_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    product_root = (ROOT / args.product_root).resolve()
    native_site = (ROOT / args.native_site).resolve()
    if not product_root.is_dir() or not native_site.is_dir():
        raise ClassicalSearchStrengthError("frozen product runtime path is absent")
    return product_root, native_site


def _require_implementation_binding(plan: Mapping[str, Any]) -> dict[str, str]:
    expected = plan.get("implementation_files")
    if not isinstance(expected, Mapping):
        raise ClassicalSearchStrengthError("implementation binding is absent")
    observed = {str(path): sha256_file(ROOT / str(path)) for path in expected}
    if observed != dict(expected):
        raise ClassicalSearchStrengthError("implementation binding differs")
    return observed


def run_calibration(args: argparse.Namespace) -> int:
    plan, plan_file_sha = _load_sealed(
        ROOT / args.plan,
        schema=CALIBRATION_PLAN_SCHEMA,
        identity_field="plan_identity",
    )
    _require_implementation_binding(plan)
    result_path = ROOT / plan["outputs"]["calibration_result"]
    if result_path.exists():
        raise ClassicalSearchStrengthError("calibration result already exists")
    product_root, native_site = _runtime_paths(args)
    pool, pool_sha = load_guidance_sealed(
        ROOT / plan["start_pool"]["path"],
        schema=POOL_SCHEMA,
        identity_field="pool_identity",
    )
    if (
        pool["pool_identity"] != plan["start_pool"]["pool_identity"]
        or pool_sha != plan["start_pool"]["file_sha256"]
    ):
        raise ClassicalSearchStrengthError("calibration start pool differs")
    state_by_id = {str(row["state_id"]): row for row in pool["states"]}
    selected_ids = list(plan["calibration"]["state_ids"])
    if canonical_sha256(selected_ids) != plan["calibration"]["membership_identity"]:
        raise ClassicalSearchStrengthError("calibration membership differs")
    runtime = ProductMainRuntime(
        product_root=product_root,
        native_site=native_site,
        resource_root=ROOT,
        expected=plan["product_contract"],
    )
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    try:
        for difficulty in (9, 10):
            for state_id in selected_ids:
                state = state_by_id[state_id]
                board = board_from_state(state)
                ai = runtime.new_ai(
                    color=board.turn,
                    difficulty=difficulty,
                    node_budget=None,
                    search_threads=int(plan["product_contract"]["timed_search_threads"]),
                    max_depth=int(plan["product_contract"]["max_depth"]),
                )
                observation = runtime.choose(ai, board)
                records.append(
                    {
                        "state_id": state_id,
                        "phase": state["phase"],
                        "difficulty": difficulty,
                        "nominal_seconds": 30 if difficulty == 9 else 60,
                        **observation.record(),
                    }
                )
                del ai
        mapping = calibration_summary(records)

        canary_ids = [
            next(
                state_id
                for state_id in selected_ids
                if state_by_id[state_id]["phase"] == phase
            )
            for phase in ("placement", "movement", "flying")
        ]
        fixed_node_checks: dict[str, Any] = {}
        for difficulty in (9, 10):
            budget = int(mapping[str(difficulty)]["mapped_node_budget"])
            rows = []
            for state_id in canary_ids:
                state = state_by_id[state_id]
                board = board_from_state(state)
                repeats = []
                for _ in range(2):
                    ai = runtime.new_ai(
                        color=board.turn,
                        difficulty=difficulty,
                        node_budget=budget,
                        search_threads=int(
                            plan["product_contract"]["deterministic_search_threads"]
                        ),
                        max_depth=int(plan["product_contract"]["max_depth"]),
                    )
                    repeats.append(runtime.choose(ai, board))
                    del ai
                if repeats[0].move != repeats[1].move:
                    raise ClassicalSearchStrengthError(
                        "fixed-node fresh-instance move is not deterministic"
                    )
                rows.append(
                    {
                        "state_id": state_id,
                        "phase": state["phase"],
                        "first": repeats[0].record(),
                        "second": repeats[1].record(),
                        "fresh_instance_move_equal": True,
                    }
                )

            tt_state = state_by_id[canary_ids[0]]
            tt_board = board_from_state(tt_state)
            warm_ai = runtime.new_ai(
                color=tt_board.turn,
                difficulty=difficulty,
                node_budget=budget,
                search_threads=int(
                    plan["product_contract"]["deterministic_search_threads"]
                ),
                max_depth=int(plan["product_contract"]["max_depth"]),
            )
            cold = runtime.choose(warm_ai, tt_board)
            warm = runtime.choose(warm_ai, tt_board)
            fixed_node_checks[str(difficulty)] = {
                "node_budget": budget,
                "fresh_instance_checks": rows,
                "same_instance_tt_check": {
                    "state_id": tt_state["state_id"],
                    "cold": cold.record(),
                    "warm": warm.record(),
                    "move_equal": cold.move == warm.move,
                    "rust_tt_persists_across_choose_move_calls": True,
                    "python_tt_is_cleared_by_choose_move": True,
                },
                "formal_determinism_gate_passed": True,
            }
            del warm_ai
    finally:
        runtime.close()
    elapsed = time.perf_counter() - started
    if elapsed > float(plan["resource_envelope"]["maximum_calibration_seconds"]):
        raise ClassicalSearchStrengthError("calibration active-time envelope exceeded")
    payload = {
        "schema_version": CALIBRATION_RESULT_SCHEMA,
        "status": "completed_timing_only_before_formal_subset_freeze",
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": plan_file_sha,
        "source_commit": _git("rev-parse", "HEAD"),
        "observations": records,
        "node_budget_mapping": mapping,
        "fixed_node_checks": fixed_node_checks,
        "resources": {
            "active_seconds": elapsed,
            "complete_games": 0,
            "sanmill_processes_started": 0,
            "database_writes": 0,
        },
        "access_audit": plan["protected_access"],
        "claim_boundary": plan["claim_boundary"],
    }
    sealed = write_sealed_json(
        result_path, payload, identity_field="result_identity"
    )
    print(sealed["result_identity"], flush=True)
    return 0


def _schedule(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for arm in plan["experiment"]["classical_arms"]:
        for unit_index, start_id in enumerate(plan["start_subset"]["state_ids"]):
            phase = plan["start_subset"]["phase_by_state_id"][start_id]
            for color in ("W", "B"):
                rows.append(
                    {
                        "ordinal": ordinal,
                        "unit_index": unit_index,
                        "game_id": canonical_sha256(
                            {
                                "namespace": plan["experiment"]["schedule_namespace"],
                                "arm": arm["arm"],
                                "start_id": start_id,
                                "candidate_color": color,
                            }
                        ),
                        "start_id": start_id,
                        "phase": phase,
                        "arm": arm["arm"],
                        "difficulty": arm["difficulty"],
                        "node_budget": arm["node_budget"],
                        "candidate_color": color,
                    }
                )
                ordinal += 1
    return rows


def _reproduction_schedule(
    pool: Mapping[str, Any], *, excluded_start_ids: Sequence[str]
) -> list[dict[str, Any]]:
    full = build_guidance_schedule(pool["states"])
    selected = select_schedule_excluding_starts(
        full, excluded_start_ids=excluded_start_ids
    )
    rows = [dict(row) for row in selected if row["arm"] == "random-safe"]
    expected = (len(pool["states"]) - len(excluded_start_ids)) * 2
    if len(rows) != expected:
        raise ClassicalSearchStrengthError("reproduction schedule differs")
    return rows


def _database_snapshot(paths: Sequence[Path]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for path in paths:
        stat = path.stat()
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        snapshot[relative] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
            "journal_exists": Path(f"{path}-journal").exists(),
            "wal_exists": Path(f"{path}-wal").exists(),
            "shm_exists": Path(f"{path}-shm").exists(),
        }
    return snapshot


def _compact_reproduction(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "game_id": record["game_id"],
        "start_id": record["start_id"],
        "candidate_color": record["candidate_color"],
        "candidate_score": record["candidate_score"],
        "winner": record["winner"],
        "outcome_reason": record["outcome_reason"],
        "post_start_logical_plies": record["post_start_logical_plies"],
        "final_history_sha256": record["final_state"]["history_sha256"],
        "turn_actions_identity": canonical_sha256(
            [turn["actions"] for turn in record["turns"]]
        ),
    }


def run_measurement(args: argparse.Namespace) -> int:
    stage = "static-preflight"
    output: Path | None = None
    failure_path: Path | None = None
    lock_path = ROOT / "out/evaluation/sanmill-classical-search-strength-v1.lock"
    descriptor: int | None = None
    try:
        if _git("branch", "--show-current") != "dev":
            raise ClassicalSearchStrengthError("formal measurement requires dev")
        if _git("status", "--short", "--untracked-files=no"):
            raise ClassicalSearchStrengthError("tracked worktree must be clean")
        if _running_tgf_processes() != 0:
            raise ClassicalSearchStrengthError("a Sanmill process is already running")
        plan, plan_sha = _load_sealed(
            ROOT / args.plan, schema=PLAN_SCHEMA, identity_field="plan_identity"
        )
        authorization, authorization_sha = _load_sealed(
            ROOT / args.authorization,
            schema=AUTHORIZATION_SCHEMA,
            identity_field="authorization_identity",
        )
        if (
            authorization["plan_identity"] != plan["plan_identity"]
            or authorization["plan_file_sha256"] != plan_sha
            or authorization["resource_envelope"] != plan["resource_envelope"]
            or authorization["output_namespace"] != plan["outputs"]["namespace"]
            or not _is_ancestor(authorization["source_commit"], "HEAD")
        ):
            raise ClassicalSearchStrengthError("authorization binding differs")
        implementation = _require_implementation_binding(plan)

        output = ROOT / plan["outputs"]["namespace"]
        result_path = ROOT / plan["outputs"]["result"]
        failure_path = output / "failure.json"
        if output.exists() or result_path.exists():
            raise ClassicalSearchStrengthError("fresh output namespace is unavailable")
        output.mkdir(parents=True, exist_ok=False)
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ClassicalSearchStrengthError("another evaluator lock exists") from exc
        os.write(descriptor, authorization["authorization_identity"].encode("ascii"))
        os.close(descriptor)
        descriptor = None

        paths = _local_paths(ROOT / args.paths_config)
        checkout = _local_path(paths.get("sanmill_training_checkout"), key="sanmill")
        malom_path = _local_path(paths.get("malom_db_path"), key="malom")
        installation = inspect_sanmill_training_installation(checkout)
        sanmill_runtime = training_installation_record(
            installation, seed=int(plan["sanmill_contract"]["seed"])
        )
        _require_runtime_equal(sanmill_runtime, plan["sanmill_contract"])
        malom_snapshot = verify_malom_snapshot(
            malom_path=malom_path,
            manifest_path=ROOT / plan["malom_contract"]["manifest_path"],
            full_hash=False,
        )
        if (
            malom_snapshot["trust_level"] != "sector-corrected-v1"
            or malom_snapshot["content_sha256"]
            != plan["malom_contract"]["content_sha256"]
        ):
            raise ClassicalSearchStrengthError("Malom snapshot differs")

        guidance, guidance_sha = load_guidance_sealed(
            ROOT / plan["guidance_input"]["plan_path"],
            schema=GUIDANCE_PLAN_SCHEMA,
            identity_field="plan_identity",
        )
        pool, pool_sha = load_guidance_sealed(
            ROOT / plan["start_pool"]["path"],
            schema=POOL_SCHEMA,
            identity_field="pool_identity",
        )
        readiness, readiness_sha = load_guidance_sealed(
            ROOT / plan["guidance_input"]["readiness_path"],
            schema=READINESS_SCHEMA,
            identity_field="result_identity",
        )
        if (
            guidance["plan_identity"] != plan["guidance_input"]["plan_identity"]
            or guidance_sha != plan["guidance_input"]["plan_file_sha256"]
            or readiness["result_identity"]
            != plan["guidance_input"]["readiness_identity"]
            or readiness_sha != plan["guidance_input"]["readiness_file_sha256"]
            or pool["pool_identity"] != plan["start_pool"]["pool_identity"]
            or pool_sha != plan["start_pool"]["file_sha256"]
        ):
            raise ClassicalSearchStrengthError("frozen gameplay input differs")
        start_ids = list(plan["start_subset"]["state_ids"])
        if canonical_sha256(start_ids) != plan["start_subset"]["membership_identity"]:
            raise ClassicalSearchStrengthError("formal subset identity differs")
        states = {str(row["state_id"]): row for row in pool["states"]}

        reference = _load_plain_reference(
            ROOT / plan["known_answer"]["reference_path"],
            plan["known_answer"]["reference"],
        )
        reference_games = [
            row
            for row in reference["games"]
            if row["arm"] == "random-safe" and row["start_id"] in set(start_ids)
        ]
        reproduction_rows = [
            row
            for row in _reproduction_schedule(
                pool, excluded_start_ids=plan["start_pool"]["excluded_start_ids"]
            )
            if row["start_id"] in set(start_ids)
        ]
        classical_rows = _schedule(plan)
        planned_games = len(reproduction_rows) + len(classical_rows)
        if planned_games != int(plan["resource_envelope"]["planned_complete_games"]):
            raise ClassicalSearchStrengthError("planned game count differs")
        if planned_games > int(plan["resource_envelope"]["maximum_complete_games"]):
            raise ClassicalSearchStrengthError("authorized game envelope differs")

        tracked_databases = [
            ROOT / "data/endgame/fullgame.bin",
            ROOT / "data/value_net_phase_place.npz",
            ROOT / "data/value_net_phase_move.npz",
            ROOT / "data/value_net_phase_fly.npz",
            ROOT / "data/gap_net.npz",
            *sorted((ROOT / "data/endgame").glob("*.wdl")),
        ]
        before_databases = _database_snapshot(tracked_databases)
        write_json_atomic(
            output / "preflight.json",
            {
                "status": "ready_for_known_answer_gate",
                "plan_identity": plan["plan_identity"],
                "authorization_identity": authorization["authorization_identity"],
                "sanmill_runtime": sanmill_runtime,
                "malom_snapshot": {
                    "trust_level": malom_snapshot["trust_level"],
                    "content_sha256": malom_snapshot["content_sha256"],
                },
                "formal_start_membership_identity": canonical_sha256(start_ids),
                "known_answer_games": len(reproduction_rows),
                "classical_games_after_gate": len(classical_rows),
                "product_runtime_loaded": False,
                "protected_content_reads": 0,
            },
        )

        ledger = ResourceLedger(
            engine_searches=0,
            malom_queries=0,
            active_seconds_before_run=float(plan["calibration"]["active_seconds"]),
            maximum_engine_searches=int(plan["resource_envelope"]["engine_search_anomaly_ceiling"]),
            maximum_malom_queries=int(plan["resource_envelope"]["malom_query_anomaly_ceiling"]),
            maximum_active_seconds=float(plan["resource_envelope"]["maximum_active_seconds"]),
        )
        write_json_atomic(
            output / "measurement-started.json",
            {
                "plan_identity": plan["plan_identity"],
                "authorization_identity": authorization["authorization_identity"],
                "started_at_unix": time.time(),
                "execution_count": 1,
            },
        )

        stage = "known-answer-reproduction"
        reproduction_path = output / "reproduction-games.jsonl"
        reproduction_records: list[dict[str, Any]] = []
        previous: str | None = None
        reproduction_database = MalomDB(malom_path)
        try:
            for index, item in enumerate(reproduction_rows, start=1):
                record = play_reproduction_game(
                    schedule_item=item,
                    start_state=states[str(item["start_id"])],
                    plan=guidance,
                    readiness=readiness,
                    database=reproduction_database,
                    installation=installation,
                    ledger=ledger,
                )
                previous = append_reproduction_record(
                    reproduction_path,
                    record,
                    previous_record_sha256=previous,
                )
                reproduction_records.append(record)
                write_json_atomic(
                    output / "progress.json",
                    {
                        "stage": stage,
                        "completed_games": index,
                        "planned_games": planned_games,
                        "resources": ledger.record(),
                    },
                )
        finally:
            reproduction_database.close()
        gate = exact_subset_gate(reproduction_records, reference_games)
        write_json_atomic(output / "known-answer-gate.json", gate)
        if not gate["passed"]:
            raise ClassicalSearchStrengthError("known-answer gate did not match exactly")

        stage = "classical-measurement"
        product_root, native_site = _runtime_paths(args)
        product_runtime = ProductMainRuntime(
            product_root=product_root,
            native_site=native_site,
            resource_root=ROOT,
            expected=plan["product_contract"],
        )
        candidate_path = output / "classical-games.jsonl"
        candidate_records: list[dict[str, Any]] = []
        previous = None
        candidate_database = MalomDB(malom_path)
        try:
            for index, item in enumerate(classical_rows, start=1):
                record = play_classical_game(
                    schedule_item=item,
                    start_state=states[str(item["start_id"])],
                    product_runtime=product_runtime,
                    product_contract=plan["product_contract"],
                    database=candidate_database,
                    installation=installation,
                    ledger=ledger,
                )
                previous = _append_jsonl_record(
                    candidate_path,
                    record,
                    previous_record_sha256=previous,
                )
                candidate_records.append(record)
                completed = len(reproduction_records) + index
                write_json_atomic(
                    output / "progress.json",
                    {
                        "stage": stage,
                        "completed_games": completed,
                        "planned_games": planned_games,
                        "resources": ledger.record(),
                    },
                )
                if index % 4 == 0 or index == len(classical_rows):
                    print(
                        json.dumps(
                            {
                                "completed": index,
                                "expected": len(classical_rows),
                                "resources": ledger.record(),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        finally:
            candidate_database.close()
            product_runtime.close()

        stage = "analysis"
        prior_manifest = json.loads(
            (ROOT / plan["prior_results"]["path"]).read_text(encoding="utf-8")
        )
        if sha256_file(ROOT / plan["prior_results"]["path"]) != plan["prior_results"]["file_sha256"]:
            raise ClassicalSearchStrengthError("prior result manifest differs")
        prior_scores = prior_scores_by_start(prior_manifest, start_ids=start_ids)
        analysis = analyze_games(
            candidate_records,
            prior_scores=prior_scores,
            start_ids=start_ids,
            maximum_half_width=float(plan["primary_decision"]["maximum_half_width"]),
        )
        after_databases = _database_snapshot(tracked_databases)
        if after_databases != before_databases:
            raise ClassicalSearchStrengthError("read-only resource snapshot changed")
        resources = ledger.record()
        if resources["active_seconds"] > float(plan["resource_envelope"]["maximum_active_seconds"]):
            raise ClassicalSearchStrengthError("active-time envelope exceeded")
        payload = {
            "schema_version": RESULT_SCHEMA,
            "status": "completed_once_internal_classical_search_measurement",
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": plan_sha,
            "authorization_identity": authorization["authorization_identity"],
            "authorization_file_sha256": authorization_sha,
            "source_commit": _git("rev-parse", "HEAD"),
            "sanmill_runtime": sanmill_runtime,
            "malom_snapshot": {
                "trust_level": malom_snapshot["trust_level"],
                "content_sha256": malom_snapshot["content_sha256"],
            },
            "start_pool_identity": pool["pool_identity"],
            "formal_start_membership_identity": canonical_sha256(start_ids),
            "known_answer_gate": gate,
            "analysis": analysis,
            "prior_scores_same_subset": {
                arm: {
                    "starts": len(values),
                    "score_rate": sum(values.values()) / len(values),
                }
                for arm, values in prior_scores.items()
            },
            "machine_records": {
                "reproduction": [_compact_reproduction(row) for row in reproduction_records],
                "classical": [compact_game(row) for row in candidate_records],
                "raw_reproduction_ledger": str(reproduction_path.relative_to(ROOT)).replace("\\", "/"),
                "raw_classical_ledger": str(candidate_path.relative_to(ROOT)).replace("\\", "/"),
            },
            "resources": {
                **resources,
                "complete_games": planned_games,
                "known_answer_games": len(reproduction_records),
                "classical_games": len(candidate_records),
                "within_all_limits": True,
                "envelope": plan["resource_envelope"],
            },
            "database_read_only_audit": {
                "before": before_databases,
                "after": after_databases,
                "unchanged": True,
                "database_writes": 0,
            },
            "access_audit": plan["protected_access"],
            "implementation_files": implementation,
            "claim_boundary": plan["claim_boundary"],
        }
        sealed = write_sealed_json(
            result_path, payload, identity_field="result_identity"
        )
        write_json_atomic(
            output / "measurement-completed.json",
            {
                "result_identity": sealed["result_identity"],
                "resources": resources,
            },
        )
        print(sealed["result_identity"], flush=True)
        return 0
    except Exception as exc:
        if failure_path is not None and output is not None and output.is_dir():
            try:
                write_json_atomic(
                    failure_path,
                    {
                        "status": "failed_closed",
                        "stage": stage,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            except Exception:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if lock_path.exists():
            lock_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "measure"):
        child = subparsers.add_parser(name)
        child.add_argument("--plan", required=True)
        child.add_argument(
            "--product-root",
            default="tmp/classical-search-main-snapshot-4e4a724/tree",
        )
        child.add_argument(
            "--native-site",
            default="tmp/classical-search-main-snapshot-4e4a724/site",
        )
        child.add_argument(
            "--paths-config", default="data/training_paths.local.json"
        )
    measure = subparsers.choices["measure"]
    measure.add_argument("--authorization", required=True)
    args = parser.parse_args()
    if args.command == "calibrate":
        return run_calibration(args)
    return run_measurement(args)


if __name__ == "__main__":
    raise SystemExit(main())
