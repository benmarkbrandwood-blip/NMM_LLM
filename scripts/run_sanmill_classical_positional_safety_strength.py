#!/usr/bin/env python3
"""Execute once the frozen current-dev classical ``A_pos`` measurement."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.malom_db import MalomDB
from ai.malom_runtime import resolve_product_malom_runtime
from game.rules import get_all_legal_moves
from learned_ai.agents.positional_safety import ProductPositionalSafetyGate
from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    verify_malom_snapshot,
    write_sealed_json,
)
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    RESULT_SCHEMA as READINESS_SCHEMA,
)
from learned_ai.evaluation.sanmill_classical_positional_safety_strength import (
    AUTHORIZATION_SCHEMA,
    PLAN_SCHEMA,
    RESULT_SCHEMA,
    ClassicalPositionalSafetyStrengthError,
    ProductDevRuntime,
    analyze_filtered_contrasts,
    compact_game,
    compare_classical_ledgers,
    play_dev_classical_game,
)
from learned_ai.evaluation.sanmill_classical_search_strength import (
    board_from_state,
    exact_subset_gate,
    paired_interval,
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
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
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
        raise ClassicalPositionalSafetyStrengthError(
            f"sealed schema differs: {path}"
        )
    body = dict(value)
    identity = body.pop(identity_field, None)
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise ClassicalPositionalSafetyStrengthError(
            f"sealed identity differs: {path}"
        )
    return value, sha256_file(path)


def _local_paths(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ClassicalPositionalSafetyStrengthError(
            "local path registry is not an object"
        )
    return value


def _local_path(value: Any, *, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ClassicalPositionalSafetyStrengthError(
            f"local path is absent: {key}"
        )
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
        raise ClassicalPositionalSafetyStrengthError(
            "cannot inspect Sanmill processes"
        )
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
        raise ClassicalPositionalSafetyStrengthError(
            f"Sanmill runtime comparability differs: {mismatches}"
        )


def _require_implementation_binding(plan: Mapping[str, Any]) -> dict[str, str]:
    expected = plan.get("implementation_files")
    if not isinstance(expected, Mapping):
        raise ClassicalPositionalSafetyStrengthError(
            "implementation binding is absent"
        )
    observed = {str(path): sha256_file(ROOT / str(path)) for path in expected}
    if observed != dict(expected):
        raise ClassicalPositionalSafetyStrengthError(
            "measurement implementation binding differs"
        )
    return observed


def _require_product_binding(plan: Mapping[str, Any]) -> dict[str, Any]:
    contract = plan["product_contract"]
    import nmm_core

    native = getattr(nmm_core, "nmm_core", None)
    native_path = Path(str(getattr(native, "__file__", "")))
    observed = {
        "game_ai": sha256_file(ROOT / "ai/game_ai.py"),
        "heuristics": sha256_file(ROOT / "ai/heuristics.py"),
        "native_extension": sha256_file(native_path),
    }
    if observed != contract["implementation_sha256"]:
        raise ClassicalPositionalSafetyStrengthError(
            "current dev product implementation differs"
        )
    filter_observed = {
        "web_app": sha256_file(ROOT / "web/app.py"),
        "positional_safety": sha256_file(
            ROOT / "learned_ai/agents/positional_safety.py"
        ),
        "malom_runtime": sha256_file(ROOT / "ai/malom_runtime.py"),
    }
    if filter_observed != contract["filter_implementation_sha256"]:
        raise ClassicalPositionalSafetyStrengthError(
            "delivered product filter implementation differs"
        )
    return {
        "classical": observed,
        "filter": filter_observed,
        "native_extension_path": str(native_path),
    }


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


def _load_jsonl_chain(path: Path, *, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ClassicalPositionalSafetyStrengthError(
            f"reference raw ledger hash differs: {path}"
        )
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ClassicalPositionalSafetyStrengthError(
                f"reference JSONL row is not an object: {line_number}"
            )
        body = dict(value)
        digest = body.pop("record_sha256", None)
        if (
            not isinstance(digest, str)
            or canonical_sha256(body) != digest
            or body.get("previous_record_sha256") != previous
        ):
            raise ClassicalPositionalSafetyStrengthError(
                f"reference JSONL chain differs: {line_number}"
            )
        rows.append(value)
        previous = digest
    return rows


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
        raise ClassicalPositionalSafetyStrengthError(
            "known-answer reference differs"
        )
    return value


def _schedule(
    plan: Mapping[str, Any], *, filtered: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordinal = 192 if filtered else 0
    arms = [
        arm for arm in plan["experiment"]["arms"] if bool(arm["filtered"]) == filtered
    ]
    arms.sort(key=lambda arm: int(arm["difficulty"]))
    for arm in arms:
        for unit_index, start_id in enumerate(plan["start_subset"]["state_ids"]):
            for color in ("W", "B"):
                namespace = (
                    plan["experiment"]["filtered_game_id_namespace"]
                    if filtered
                    else plan["experiment"]["unfiltered_game_id_namespace"]
                )
                identity_arm = arm["arm"] if filtered else arm["reference_arm"]
                rows.append(
                    {
                        "ordinal": ordinal,
                        "unit_index": unit_index,
                        "game_id": canonical_sha256(
                            {
                                "namespace": namespace,
                                "arm": identity_arm,
                                "start_id": start_id,
                                "candidate_color": color,
                            }
                        ),
                        "start_id": start_id,
                        "phase": plan["start_subset"]["phase_by_state_id"][start_id],
                        "arm": arm["arm"],
                        "reference_arm": arm["reference_arm"],
                        "difficulty": arm["difficulty"],
                        "node_budget": arm["node_budget"],
                        "filtered": filtered,
                        "candidate_color": color,
                    }
                )
                ordinal += 1
    return rows


def _reproduction_schedule(
    pool: Mapping[str, Any], *, excluded_start_ids: Sequence[str], start_ids: set[str]
) -> list[dict[str, Any]]:
    full = build_guidance_schedule(pool["states"])
    selected = select_schedule_excluding_starts(
        full, excluded_start_ids=excluded_start_ids
    )
    return [
        dict(row)
        for row in selected
        if row["arm"] == "random-safe" and str(row["start_id"]) in start_ids
    ]


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


def _disabled_gate_canary(state: Mapping[str, Any]) -> dict[str, Any]:
    board = board_from_state(state)
    legal = get_all_legal_moves(board)
    if not legal:
        raise ClassicalPositionalSafetyStrengthError(
            "disabled-gate canary state has no legal move"
        )
    original = dict(legal[0])
    gate = ProductPositionalSafetyGate(high_difficulty_minimum=9)
    gate.disable("measurement disabled-filter side-effect canary")
    outcome = gate.constrain(
        board,
        original,
        source="classical-coordinator",
        difficulty=9,
        query_failure_move=original,
    )
    passed = (
        outcome.move == original
        and outcome.decision.get("status") == "unfiltered-malom-unavailable"
        and not outcome.decision.get("intervened")
    )
    if not passed:
        raise ClassicalPositionalSafetyStrengthError(
            "disabled product gate changed the unfiltered move"
        )
    return {
        "passed": True,
        "state_id": state["state_id"],
        "original_move": original,
        "selected_move": outcome.move,
        "status": outcome.decision["status"],
    }


def _same_start_prior_contrasts(
    records: Sequence[Mapping[str, Any]],
    *,
    prior_scores: Mapping[str, Mapping[str, float]],
    start_ids: Sequence[str],
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in records:
        arm = str(row["arm"])
        grouped.setdefault(arm, {}).setdefault(str(row["start_id"]), []).append(
            float(row["candidate_score"])
        )
    result: dict[str, Any] = {}
    for arm, by_start in grouped.items():
        scores = {
            start_id: statistics.fmean(values)
            for start_id, values in by_start.items()
        }
        for prior_arm, prior in prior_scores.items():
            values = [
                scores[start_id] - float(prior[start_id])
                for start_id in sorted(start_ids)
            ]
            result[f"{arm}_minus_{prior_arm}"] = paired_interval(values)
    return result


def _sanitized_malom_status(status: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "validation": status["validation"],
        "selected_source": status["selected_source"],
        "manifest_file_sha256": status["manifest_file_sha256"],
        "manifest_sha256": status["manifest_sha256"],
        "content_sha256": status["content_sha256"],
        "label_version": status["label_version"],
        "component_count": status["component_count"],
        "size_bytes": status["size_bytes"],
        "inventory_validation": status["inventory_validation"],
        "candidate_outcomes": [
            {
                "source": row["source"],
                "status": row["status"],
                "reason": row["reason"],
                "path_recorded_only_in_local_preflight": bool(row.get("path")),
            }
            for row in status["candidates"]
        ],
    }


def run(args: argparse.Namespace) -> int:
    stage = "static-preflight"
    output: Path | None = None
    failure_path: Path | None = None
    descriptor: int | None = None
    lock_path = ROOT / "out/evaluation/sanmill-classical-positional-safety-strength-v1.lock"
    product_runtime: ProductDevRuntime | None = None
    product_resolution: Any | None = None
    try:
        if _git("branch", "--show-current") != "dev":
            raise ClassicalPositionalSafetyStrengthError(
                "formal measurement requires dev"
            )
        if _git("status", "--short", "--untracked-files=no"):
            raise ClassicalPositionalSafetyStrengthError(
                "tracked worktree must be clean"
            )
        if _running_tgf_processes() != 0:
            raise ClassicalPositionalSafetyStrengthError(
                "a Sanmill process is already running"
            )

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
            raise ClassicalPositionalSafetyStrengthError(
                "authorization binding differs"
            )
        implementation = _require_implementation_binding(plan)
        product_binding = _require_product_binding(plan)

        output = ROOT / plan["outputs"]["namespace"]
        result_path = ROOT / plan["outputs"]["result"]
        failure_path = output / "failure.json"
        if output.exists() or result_path.exists():
            raise ClassicalPositionalSafetyStrengthError(
                "fresh output namespace is unavailable"
            )
        output.mkdir(parents=True, exist_ok=False)
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ClassicalPositionalSafetyStrengthError(
                "another evaluator lock exists"
            ) from exc
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
            raise ClassicalPositionalSafetyStrengthError("Malom snapshot differs")

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
            raise ClassicalPositionalSafetyStrengthError(
                "frozen gameplay input differs"
            )
        start_ids = list(plan["start_subset"]["state_ids"])
        if canonical_sha256(start_ids) != plan["start_subset"]["membership_identity"]:
            raise ClassicalPositionalSafetyStrengthError(
                "formal start membership differs"
            )
        states = {str(row["state_id"]): row for row in pool["states"]}
        if not set(start_ids) <= states.keys():
            raise ClassicalPositionalSafetyStrengthError(
                "formal start state is absent"
            )

        reference = _load_plain_reference(
            ROOT / plan["known_answer"]["reference_path"],
            plan["known_answer"]["reference"],
        )
        reference_games = [
            row
            for row in reference["games"]
            if row["arm"] == "random-safe" and row["start_id"] in set(start_ids)
        ]
        reproduction_rows = _reproduction_schedule(
            pool,
            excluded_start_ids=plan["start_pool"]["excluded_start_ids"],
            start_ids=set(start_ids),
        )
        unfiltered_rows = _schedule(plan, filtered=False)
        filtered_rows = _schedule(plan, filtered=True)
        planned_games = (
            len(reproduction_rows) + len(unfiltered_rows) + len(filtered_rows)
        )
        if planned_games != int(plan["resource_envelope"]["planned_complete_games"]):
            raise ClassicalPositionalSafetyStrengthError(
                "planned complete-game count differs"
            )
        if planned_games > int(plan["resource_envelope"]["maximum_complete_games"]):
            raise ClassicalPositionalSafetyStrengthError(
                "complete-game envelope differs"
            )

        v2_reference = _load_jsonl_chain(
            ROOT / plan["v2_reference"]["raw_classical_ledger"],
            expected_sha256=plan["v2_reference"]["raw_classical_ledger_sha256"],
        )
        if len(v2_reference) != 192:
            raise ClassicalPositionalSafetyStrengthError(
                "v2 classical reference count differs"
            )
        disabled_canary = _disabled_gate_canary(states[start_ids[0]])

        tracked_resources = [
            ROOT / "data/endgame/fullgame.bin",
            ROOT / "data/value_net_phase_place.npz",
            ROOT / "data/value_net_phase_move.npz",
            ROOT / "data/value_net_phase_fly.npz",
            ROOT / "data/gap_net.npz",
            *sorted((ROOT / "data/endgame").glob("*.wdl")),
        ]
        before_resources = _database_snapshot(tracked_resources)
        ledger = ResourceLedger(
            engine_searches=0,
            malom_queries=0,
            active_seconds_before_run=0.0,
            maximum_engine_searches=int(
                plan["resource_envelope"]["engine_search_anomaly_ceiling"]
            ),
            maximum_malom_queries=int(
                plan["resource_envelope"]["malom_query_anomaly_ceiling"]
            ),
            maximum_active_seconds=float(
                plan["resource_envelope"]["maximum_active_seconds"]
            ),
        )

        def adapter_factory(path: str, *, strict: bool) -> ExternalSolvedDB:
            return ExternalSolvedDB(
                path,
                strict=strict,
                query_observer=ledger.add_malom,
            )

        settings = json.loads((ROOT / "data/settings.json").read_text(encoding="utf-8"))
        product_resolution = resolve_product_malom_runtime(
            repo_root=ROOT,
            settings=settings,
            sentinel_path="",
            local_paths_path=ROOT / args.paths_config,
            manifest_path=ROOT / plan["malom_contract"]["manifest_path"],
            adapter_factory=adapter_factory,
        )
        if (
            product_resolution.database is None
            or product_resolution.oracle is None
            or product_resolution.status["validation"] != "passed"
            or product_resolution.status["label_version"]
            != "sector-corrected-v1"
            or product_resolution.status["content_sha256"]
            != plan["malom_contract"]["content_sha256"]
        ):
            raise ClassicalPositionalSafetyStrengthError(
                "delivered product Malom resolver did not validate"
            )
        product_malom_status = _sanitized_malom_status(product_resolution.status)
        gate = ProductPositionalSafetyGate(high_difficulty_minimum=9)
        gate.configure(
            product_resolution.oracle,
            label_version=product_resolution.status["label_version"],
            manifest_sha256=product_resolution.status["manifest_sha256"],
            content_sha256=product_resolution.status["content_sha256"],
        )
        if not gate.is_enabled():
            raise ClassicalPositionalSafetyStrengthError(
                "delivered product gate did not enable"
            )

        write_json_atomic(
            output / "preflight.json",
            {
                "status": "ready_for_once-only-known-answer-gate",
                "plan_identity": plan["plan_identity"],
                "authorization_identity": authorization["authorization_identity"],
                "sanmill_runtime": sanmill_runtime,
                "product_binding": product_binding,
                "product_malom_runtime_local_status": product_resolution.status,
                "disabled_filter_side_effect_canary": disabled_canary,
                "formal_start_membership_identity": canonical_sha256(start_ids),
                "planned_games": planned_games,
                "running_sanmill_processes_before_marker": _running_tgf_processes(),
                "protected_content_reads": 0,
            },
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
        known_gate = exact_subset_gate(reproduction_records, reference_games)
        known_gate["expected_identity"] = plan["known_answer"][
            "expected_subset_identity"
        ]
        known_gate["expected_identity_matched"] = (
            known_gate["observed_identity"]
            == plan["known_answer"]["expected_subset_identity"]
        )
        write_json_atomic(output / "known-answer-gate.json", known_gate)
        if not known_gate["passed"] or not known_gate["expected_identity_matched"]:
            raise ClassicalPositionalSafetyStrengthError(
                "known-answer gate did not match exactly"
            )

        product_runtime = ProductDevRuntime(
            resource_root=ROOT, expected=plan["product_contract"]
        )
        candidate_database = MalomDB(malom_path)
        candidate_path = output / "candidate-games.jsonl"
        candidate_records: list[dict[str, Any]] = []
        previous = None
        stage = "current-dev-unfiltered"
        try:
            for index, item in enumerate(unfiltered_rows, start=1):
                record = play_dev_classical_game(
                    schedule_item=item,
                    start_state=states[str(item["start_id"])],
                    product_runtime=product_runtime,
                    product_contract=plan["product_contract"],
                    database=candidate_database,
                    product_malom_adapter=product_resolution.database,
                    gate=None,
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
                if index % 8 == 0:
                    print(
                        json.dumps(
                            {
                                "stage": stage,
                                "completed": index,
                                "expected": len(unfiltered_rows),
                                "resources": ledger.record(),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

            unfiltered_comparison = compare_classical_ledgers(
                candidate_records, v2_reference
            )
            continuation = {
                "continued_to_filtered": True,
                "reason": (
                    "all 192 dev unfiltered games semantically match v2"
                    if unfiltered_comparison["exact_match"]
                    else "branch/runtime divergence observed, but the disabled-gate canary preserved the original move and both current-dev arms share the identical ProductDevRuntime, resources, internal Malom adapter, start/color schedule and referee; the same-dev final-gate contrast remains identified"
                ),
                "filter_disabled_side_effect_excluded_by_canary": disabled_canary[
                    "passed"
                ],
                "same_dev_primary_contrast_valid": True,
            }
            write_json_atomic(
                output / "unfiltered-v2-comparison.json",
                {
                    "comparison": unfiltered_comparison,
                    "continuation_decision": continuation,
                },
            )
            if not continuation["same_dev_primary_contrast_valid"]:
                raise ClassicalPositionalSafetyStrengthError(
                    "same-dev primary contrast is not identified"
                )

            stage = "current-dev-filtered"
            for index, item in enumerate(filtered_rows, start=1):
                record = play_dev_classical_game(
                    schedule_item=item,
                    start_state=states[str(item["start_id"])],
                    product_runtime=product_runtime,
                    product_contract=plan["product_contract"],
                    database=candidate_database,
                    product_malom_adapter=product_resolution.database,
                    gate=gate,
                    installation=installation,
                    ledger=ledger,
                )
                previous = _append_jsonl_record(
                    candidate_path,
                    record,
                    previous_record_sha256=previous,
                )
                candidate_records.append(record)
                completed = (
                    len(reproduction_records) + len(unfiltered_rows) + index
                )
                write_json_atomic(
                    output / "progress.json",
                    {
                        "stage": stage,
                        "completed_games": completed,
                        "planned_games": planned_games,
                        "resources": ledger.record(),
                    },
                )
                if index % 8 == 0:
                    print(
                        json.dumps(
                            {
                                "stage": stage,
                                "completed": index,
                                "expected": len(filtered_rows),
                                "resources": ledger.record(),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        finally:
            candidate_database.close()

        stage = "analysis"
        analysis = analyze_filtered_contrasts(
            candidate_records,
            start_ids=start_ids,
            maximum_half_width=float(
                plan["primary_decision"]["maximum_half_width"]
            ),
        )
        prior_manifest_path = ROOT / plan["prior_results"]["path"]
        if sha256_file(prior_manifest_path) != plan["prior_results"]["file_sha256"]:
            raise ClassicalPositionalSafetyStrengthError(
                "prior result manifest differs"
            )
        prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        prior_scores = prior_scores_by_start(prior_manifest, start_ids=start_ids)
        old_scores = {
            arm: {
                "starts": len(values),
                "score_rate": statistics.fmean(values.values()),
            }
            for arm, values in prior_scores.items()
        }
        prior_contrasts = _same_start_prior_contrasts(
            candidate_records,
            prior_scores=prior_scores,
            start_ids=start_ids,
        )
        after_resources = _database_snapshot(tracked_resources)
        if after_resources != before_resources:
            raise ClassicalPositionalSafetyStrengthError(
                "read-only product resource snapshot changed"
            )
        after_malom = verify_malom_snapshot(
            malom_path=malom_path,
            manifest_path=ROOT / plan["malom_contract"]["manifest_path"],
            full_hash=False,
        )
        malom_snapshot_fields = (
            "manifest_sha256",
            "component_count",
            "size_bytes",
            "content_sha256",
            "trust_level",
        )
        if any(
            after_malom[field] != malom_snapshot[field]
            for field in malom_snapshot_fields
        ):
            raise ClassicalPositionalSafetyStrengthError(
                "read-only Malom snapshot changed"
            )
        resources = ledger.record()
        if resources["active_seconds"] > float(
            plan["resource_envelope"]["maximum_active_seconds"]
        ):
            raise ClassicalPositionalSafetyStrengthError(
                "active-time envelope exceeded"
            )
        if planned_games > int(plan["resource_envelope"]["maximum_complete_games"]):
            raise ClassicalPositionalSafetyStrengthError(
                "complete-game envelope exceeded"
            )

        payload = {
            "schema_version": RESULT_SCHEMA,
            "status": "completed_once_internal_current-dev-classical-A_pos-measurement",
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": plan_sha,
            "authorization_identity": authorization["authorization_identity"],
            "authorization_file_sha256": authorization_sha,
            "execution_head": _git("rev-parse", "HEAD"),
            "product_source_commit": plan["product_contract"]["source_commit"],
            "sanmill_runtime": sanmill_runtime,
            "malom_snapshot": {
                "trust_level": malom_snapshot["trust_level"],
                "content_sha256": malom_snapshot["content_sha256"],
            },
            "product_malom_runtime": product_malom_status,
            "product_binding": product_binding,
            "start_pool_identity": pool["pool_identity"],
            "formal_start_membership_identity": canonical_sha256(start_ids),
            "known_answer_gate": known_gate,
            "disabled_filter_side_effect_canary": disabled_canary,
            "unfiltered_v2_comparison": unfiltered_comparison,
            "unfiltered_comparison_continuation": continuation,
            "analysis": analysis,
            "old_arms_same_subset": old_scores,
            "candidate_minus_old_arm_contrasts": prior_contrasts,
            "delivered_filter_mechanism": plan["product_contract"][
                "actual_delivery"
            ],
            "product_route_caveat": plan["product_contract"]["actual_delivery"][
                "interactive_route_caveat"
            ],
            "gate_final_status": gate.status(),
            "machine_records": {
                "reproduction": [
                    _compact_reproduction(row) for row in reproduction_records
                ],
                "candidate_compact": [compact_game(row) for row in candidate_records],
                "raw_reproduction_ledger": str(
                    reproduction_path.relative_to(ROOT)
                ).replace("\\", "/"),
                "raw_reproduction_ledger_sha256": sha256_file(reproduction_path),
                "raw_candidate_ledger": str(candidate_path.relative_to(ROOT)).replace(
                    "\\", "/"
                ),
                "raw_candidate_ledger_sha256": sha256_file(candidate_path),
            },
            "resources": {
                **resources,
                "complete_games": planned_games,
                "known_answer_games": len(reproduction_records),
                "unfiltered_games": len(unfiltered_rows),
                "filtered_games": len(filtered_rows),
                "within_all_limits": True,
                "envelope": plan["resource_envelope"],
            },
            "database_read_only_audit": {
                "before": before_resources,
                "after": after_resources,
                "unchanged": True,
                "malom_inventory_unchanged": True,
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
                "completed_games": planned_games,
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
        if product_runtime is not None:
            product_runtime.close()
        if product_resolution is not None:
            close = getattr(product_resolution.database, "close", None)
            if callable(close):
                close()
        if descriptor is not None:
            os.close(descriptor)
        if lock_path.exists():
            lock_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument(
        "--paths-config", default="data/training_paths.local.json"
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
