#!/usr/bin/env python3
"""Run an authorization-bound all-surface non-evidence rehearsal."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.malom_db import MalomDB
from game.board import BoardState
from game.rules import get_all_legal_moves, terminal_wdl
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
    PHASES,
    REHEARSAL_SCHEMA,
    TrainedModelBaselineError,
    append_game_record,
    audit_instrumentation_surface,
    compact_game,
    formal_states,
    load_attempt_authorization,
    load_attempt_spec,
    load_game_records,
    load_model_policies,
    load_plan,
    play_game,
    replay_scripted_terminal_game,
    verify_resource_game_alignment,
)
from learned_ai.evaluation.sanmill_trained_model_boundary_registry import (
    BoundaryCoverageRecorder,
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


def _load_phase_corpus(attempt: Mapping[str, Any]) -> dict[str, Any]:
    spec = attempt["rehearsal"]["phase_corpus"]
    path = _ROOT / str(spec["path"])
    if sha256_file(path) != spec["file_sha256"]:
        raise TrainedModelBaselineError("rehearsal phase corpus file differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("corpus_identity") != spec["identity"]:
        raise TrainedModelBaselineError("rehearsal phase corpus identity differs")
    return value


def _state_from_record(
    record: Mapping[str, Any], *, attempt_id: str
) -> dict[str, Any]:
    return {
        "state_id": f"{attempt_id}:{record['start_id']}",
        "source_start_id": str(record["start_id"]),
        "phase": str(record["phase"]),
        "fen": str(record["fen"]),
        "logical_turns": [list(turn) for turn in record["logical_turns"]],
        "history_actions": list(record["action_history"]),
        "logical_ply": int(record["logical_ply_count"]),
        "strict_history_sha256": str(record["strict_start"]["history_sha256"]),
    }


def _rehearsal_states(
    *,
    attempt: Mapping[str, Any],
    plan: Mapping[str, Any],
    pool: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    corpus = _load_phase_corpus(attempt)
    by_id = {str(row["start_id"]): row for row in corpus["records"]}
    requested = {
        str(row["source_start_id"]): row
        for row in (
            list(attempt["rehearsal"]["live_starts"])
            + [case["start"] for case in attempt["rehearsal"]["scripted_cases"]]
        )
    }
    states = {}
    for source_id, spec in requested.items():
        record = by_id.get(source_id)
        if (
            record is None
            or record.get("record_identity") != spec["record_identity"]
            or record.get("phase") != spec["phase"]
            or record.get("strict_start", {}).get("history_sha256")
            != spec["strict_history_sha256"]
        ):
            raise TrainedModelBaselineError("rehearsal phase start differs")
        states[source_id] = _state_from_record(
            record,
            attempt_id=str(attempt["attempt_id"]),
        )

    formal = formal_states(
        pool,
        excluded_start_ids=plan["start_pool"]["excluded_start_ids"],
    )
    formal_fens = {str(row["fen"]) for row in formal}
    formal_histories = {tuple(row["history_actions"]) for row in formal}
    if any(
        state["fen"] in formal_fens
        or tuple(state["history_actions"]) in formal_histories
        for state in states.values()
    ):
        raise TrainedModelBaselineError("rehearsal overlaps formal pool")
    return states


def _source_games(attempt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    spec = attempt["rehearsal"]["scripted_source_ledger"]
    path = _ROOT / str(spec["path"])
    if sha256_file(path) != spec["file_sha256"]:
        raise TrainedModelBaselineError("scripted source ledger differs")
    wanted = {
        str(case["source_game_id"]): case
        for case in attempt["rehearsal"]["scripted_cases"]
    }
    found = {}
    for encoded in path.read_bytes().splitlines():
        wrapper = json.loads(encoded)
        record = wrapper.get("record", {})
        game_id = str(record.get("game_id"))
        if game_id not in wanted:
            continue
        case = wanted[game_id]
        if (
            wrapper.get("record_sha256") != case["source_record_sha256"]
            or record.get("start_id") != case["start"]["source_start_id"]
            or record.get("outcome_reason") != case["outcome_reason"]
            or record.get("termination_class") != "rules_terminal"
        ):
            raise TrainedModelBaselineError("scripted terminal source differs")
        found[game_id] = dict(record)
    if set(found) != set(wanted):
        raise TrainedModelBaselineError("scripted terminal source is incomplete")
    return found


def _item(
    *,
    ordinal: int,
    attempt_id: str,
    state: Mapping[str, Any],
    arm: str,
    candidate_color: str,
) -> dict[str, Any]:
    body = {
        "namespace": f"{attempt_id}-rehearsal-game",
        "ordinal": ordinal,
        "start_id": state["state_id"],
        "arm": arm,
        "candidate_color": candidate_color,
    }
    return {
        "ordinal": ordinal,
        "unit_index": ordinal,
        "start_id": state["state_id"],
        "phase": state["phase"],
        "arm": arm,
        "candidate_color": candidate_color,
        "game_id": canonical_sha256(body),
    }


def _malom_surface_canary(
    *,
    policies: Any,
    database: MalomDB,
    live_states: Sequence[Mapping[str, Any]],
    ledger: ResourceLedger,
) -> dict[str, Any]:
    external = policies._scorers["retained-v4"]._policy.malom
    rows = []
    for state in live_states:
        board = BoardState.from_fen_string(str(state["fen"]))
        legal = get_all_legal_moves(board)
        nonterminal = [
            move for move in legal if terminal_wdl(board.apply_move(move)) is None
        ]
        if not nonterminal:
            raise TrainedModelBaselineError("Malom canary has no nonterminal move")
        move = dict(nonterminal[0])
        after = board.apply_move(move)
        method_rows = []

        def observe(name: str, expected_delta: int, call: Any) -> Any:
            before = ledger.malom_queries
            value = call()
            delta = ledger.malom_queries - before
            if delta != expected_delta:
                raise TrainedModelBaselineError(
                    f"Malom canary accounting differs for {name}"
                )
            method_rows.append(
                {
                    "method": name,
                    "query_delta": delta,
                    "return_type": type(value).__name__,
                }
            )
            return value

        before_available = ledger.malom_queries
        if external.is_available() is not True:
            raise TrainedModelBaselineError("retained-v4 Malom is unavailable")
        if ledger.malom_queries != before_available:
            raise TrainedModelBaselineError("Malom availability probe was counted")
        state_value = observe("query_state", 1, lambda: external.query_state(board))
        alias_value = observe("query", 1, lambda: external.query(board))
        quality = observe(
            "query_move_quality",
            2,
            lambda: external.query_move_quality(board, move),
        )
        expected_all = 1 + sum(
            terminal_wdl(board.apply_move(candidate)) is None for candidate in legal
        )
        all_moves = observe(
            "query_all_moves",
            expected_all,
            lambda: external.query_all_moves(board, board.turn),
        )
        trajectory = observe(
            "query_trajectory",
            2,
            lambda: external.query_trajectory([board, after]),
        )
        direct = observe("MalomDB.query_value", 1, lambda: database.query_value(board))
        terminal_value = observe(
            "MalomDB.terminal_move_value",
            0,
            lambda: database.terminal_move_value(direct, "L"),
        )
        if (
            state_value is None
            or alias_value is None
            or quality is None
            or len(all_moves) != len(legal)
            or len(trajectory) != 2
            or any(value is None for value in trajectory)
            or direct is None
            or terminal_value is None
        ):
            raise TrainedModelBaselineError("Malom canary return shape differs")
        rows.append(
            {
                "source_start_id": state["source_start_id"],
                "phase": state["phase"],
                "legal_moves": len(legal),
                "methods": method_rows,
            }
        )
    return {
        "passed": True,
        "phases": sorted({row["phase"] for row in rows}),
        "rows": rows,
        "methods_exercised": sorted(
            {method["method"] for row in rows for method in row["methods"]}
            | {"is_available"}
        ),
    }


def _terminal_malom_canary(
    *,
    external: Any,
    decisive_state: Mapping[str, Any],
    decisive_record: Mapping[str, Any],
    ledger: ResourceLedger,
) -> dict[str, Any]:
    board = BoardState.from_fen_string(str(decisive_state["fen"]))
    for turn in decisive_record["turns"]:
        move_key = turn["move"]
        matches = [
            move
            for move in get_all_legal_moves(board)
            if all(move.get(key) == move_key.get(key) for key in ("from", "to", "capture"))
        ]
        if len(matches) != 1:
            raise TrainedModelBaselineError("terminal canary move differs")
        board = board.apply_move(matches[0])
    if terminal_wdl(board) is None:
        raise TrainedModelBaselineError("terminal Malom canary is not board-terminal")
    before = ledger.malom_queries
    value = external.query_state(board)
    if value is None or ledger.malom_queries != before:
        raise TrainedModelBaselineError("terminal Malom path queried the tablebase")
    return {
        "passed": True,
        "method": "ExternalSolvedDB.query_state",
        "board_terminal_short_circuit": True,
        "malom_query_delta": 0,
        "wdl": value,
    }


def _verify_prior_attempts_preserved(
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify every frozen prior record and output byte for byte."""
    spec = attempt["preserved_history"]
    observed_files = []
    for frozen in spec["tracked_files"]:
        path = _ROOT / str(frozen["path"])
        if sha256_file(path) != frozen["file_sha256"]:
            raise TrainedModelBaselineError("prior tracked evidence differs")
        identity_field = frozen.get("identity_field")
        if identity_field is not None:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get(identity_field) != frozen.get("identity"):
                raise TrainedModelBaselineError("prior tracked identity differs")
        observed_files.append(str(frozen["path"]))

    observed_namespaces = []
    for frozen in spec["output_namespaces"]:
        output_dir = _ROOT / str(frozen["path"])
        if not output_dir.is_dir():
            raise TrainedModelBaselineError("prior output namespace is absent")
        actual = sorted(
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*")
            if path.is_file()
        )
        if actual != sorted(frozen["artifacts"]):
            raise TrainedModelBaselineError("prior output tree differs")
        for relative, expected in frozen["artifacts"].items():
            path = output_dir / relative
            if (
                path.stat().st_size != int(expected["bytes"])
                or sha256_file(path) != expected["sha256"]
            ):
                raise TrainedModelBaselineError("prior output artifact differs")
        observed_namespaces.append(str(frozen["path"]))
    summary = {
        "passed": True,
        "tracked_files": sorted(observed_files),
        "output_namespaces": sorted(observed_namespaces),
        "preserved_byte_for_byte": True,
        "authorization_reused": False,
    }
    return {**summary, "preservation_identity": canonical_sha256(summary)}


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
        parser.error("trained-model rehearsal requires dev")
    if _git("status", "--short", "--untracked-files=no"):
        parser.error("tracked worktree must be clean before rehearsal")
    if _running_tgf_processes() != 0:
        parser.error("a Sanmill process is already running")
    plan, plan_sha = load_plan(_ROOT / args.plan)
    attempt, attempt_sha = load_attempt_spec(_ROOT / args.attempt)
    authorization, authorization_sha = load_attempt_authorization(
        _ROOT / args.authorization
    )
    pool, pool_sha = load_pool(_ROOT / args.pool)
    registry, registry_sha = load_boundary_registry(_ROOT / args.boundary_registry)
    contract = coverage_contract(registry)
    if (
        attempt["plan"]["identity"] != plan["plan_identity"]
        or attempt["plan"]["file_sha256"] != plan_sha
        or authorization["attempt"]["identity"] != attempt["attempt_identity"]
        or authorization["attempt"]["file_sha256"] != attempt_sha
        or authorization["status"] != "authorized_once_measurement_unconsumed"
        or pool["pool_identity"] != plan["start_pool"]["pool_identity"]
        or pool_sha != plan["start_pool"]["pool_file_sha256"]
        or attempt["boundary_registry"]["identity"]
        != registry["registry_identity"]
        or attempt["boundary_registry"]["file_sha256"] != registry_sha
        or attempt["coverage_contract"] != contract
        or authorization["boundary_registry"]["identity"]
        != registry["registry_identity"]
        or authorization["coverage_contract"]["identity"]
        != contract["coverage_contract_identity"]
    ):
        parser.error("trained-model rehearsal bindings differ")
    implementation_files = authorization["implementation_files"]
    observed_implementation = {
        path: sha256_file(_ROOT / path) for path in implementation_files
    }
    if (
        implementation_files != attempt["implementation_files"]
        or observed_implementation != implementation_files
        or attempt["resource_envelope"] != plan["resource_envelope"]
    ):
        parser.error("trained-model implementation or resource envelope differs")

    output_dir = _ROOT / str(attempt["outputs"]["rehearsal_namespace"])
    result_path = _ROOT / str(attempt["outputs"]["rehearsal_result"])
    if output_dir.exists() or result_path.exists():
        parser.error("trained-model rehearsal namespace or result already exists")
    instrumentation = audit_instrumentation_surface(_ROOT)
    if instrumentation["passed"] is not True:
        parser.error("instrumentation surface audit failed")
    preserved_history = _verify_prior_attempts_preserved(attempt)

    states = _rehearsal_states(attempt=attempt, plan=plan, pool=pool)
    sources = _source_games(attempt)
    live_states = [
        states[str(row["source_start_id"])]
        for row in attempt["rehearsal"]["live_starts"]
    ]
    live_cases = []
    ordinal = 0
    for state in live_states:
        for arm in ARMS:
            for color in ("W", "B"):
                live_cases.append(
                    (
                        _item(
                            ordinal=ordinal,
                            attempt_id=str(attempt["attempt_id"]),
                            state=state,
                            arm=arm,
                            candidate_color=color,
                        ),
                        state,
                    )
                )
                ordinal += 1
    scripted_cases = []
    for case in attempt["rehearsal"]["scripted_cases"]:
        state = states[str(case["start"]["source_start_id"])]
        source = sources[str(case["source_game_id"])]
        scripted_cases.append(
            (
                _item(
                    ordinal=ordinal,
                    attempt_id=str(attempt["attempt_id"]),
                    state=state,
                    arm=str(case["arm"]),
                    candidate_color=str(source["candidate_color"]),
                ),
                state,
                [list(turn["actions"]) for turn in source["turns"]],
                case,
                source,
            )
        )
        ordinal += 1
    expected_games = int(attempt["rehearsal"]["complete_games"])
    if len(live_cases) != 24 or len(scripted_cases) != 3 or ordinal != expected_games:
        parser.error("trained-model rehearsal schedule differs")

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        output_dir / "NON-EVIDENCE.json",
        {
            "plan_identity": plan["plan_identity"],
            "attempt_identity": attempt["attempt_identity"],
            "authorization_identity": authorization["authorization_identity"],
            "formal_result_eligibility": False,
            "purpose": "boundary-registry all-surface technical rehearsal only",
        },
    )

    coverage_ledger = output_dir / "boundary-coverage-events.jsonl"
    coverage_recorder = BoundaryCoverageRecorder(
        registry,
        coverage_ledger,
        formal_result_eligibility=False,
    )
    coverage_recorder.__enter__()
    envelope = plan["resource_envelope"]
    sunk = attempt["cumulative_sunk_resources_before_attempt"]
    ledger = ResourceLedger(
        engine_searches=int(sunk["engine_single_step_searches"]),
        malom_queries=int(sunk["malom_read_only_queries"]),
        active_seconds_before_run=float(sunk["active_seconds"]),
        maximum_engine_searches=int(envelope["maximum_engine_single_step_searches"]),
        maximum_malom_queries=int(envelope["maximum_malom_queries"]),
        maximum_active_seconds=float(envelope["maximum_active_seconds"]),
    )
    paths = _paths(_ROOT / args.paths_config)
    checkout = _local_path(paths.get("sanmill_training_checkout"), key="sanmill")
    malom_path = _local_path(paths.get("malom_db_path"), key="malom")
    installation = inspect_sanmill_training_installation(checkout)
    runtime = training_installation_record(
        installation, seed=int(plan["sanmill_contract"]["seed"])
    )
    if runtime["identity"] != plan["sanmill_contract"]["runtime_identity"]:
        raise TrainedModelBaselineError("trained-model Sanmill runtime differs")
    malom = verify_malom_snapshot(
        malom_path=malom_path,
        manifest_path=_ROOT / args.malom_manifest,
        full_hash=False,
    )
    if (
        malom["trust_level"] != "sector-corrected-v1"
        or malom["content_sha256"] != plan["malom_contract"]["content_sha256"]
    ):
        raise TrainedModelBaselineError("trained-model Malom snapshot differs")

    raw_games = output_dir / "rehearsal-games.jsonl"
    resource_journal = output_dir / "resource-checkpoints.jsonl"
    baseline = ledger.record()
    write_json_atomic(output_dir / "resource-baseline.json", baseline)
    resources_before = baseline
    previous_game = None
    previous_checkpoint = None
    records: list[dict[str, Any]] = []
    database = MalomDB(malom_path, query_observer=ledger.add_malom)
    malom_canary = None
    terminal_canary = None
    try:
        with load_model_policies(
            plan=plan,
            root=_ROOT,
            malom_path=malom_path,
            malom_manifest_path=_ROOT / args.malom_manifest,
            ledger=ledger,
        ) as policies:
            malom_canary = _malom_surface_canary(
                policies=policies,
                database=database,
                live_states=live_states,
                ledger=ledger,
            )
            for item, state in live_cases:
                record = play_game(
                    schedule_item=item,
                    start_state=state,
                    plan=plan,
                    policies=policies,
                    database=database,
                    installation=installation,
                    ledger=ledger,
                    rehearsal_only=True,
                )
                resources_after = ledger.record()
                previous_checkpoint = append_resource_checkpoint(
                    resource_journal,
                    completion_index=len(records),
                    complete_games_before=int(sunk["complete_games"]),
                    game_record=record,
                    resources_before=resources_before,
                    resources_after=resources_after,
                    previous_checkpoint_sha256=previous_checkpoint,
                )
                previous_game = append_game_record(
                    raw_games,
                    record,
                    previous_record_sha256=previous_game,
                )
                resources_before = resources_after
                records.append(record)
                write_json_atomic(
                    output_dir / "progress.json",
                    {
                        "completed_games": len(records),
                        "expected_games": expected_games,
                        "resource_checkpoint_tail": previous_checkpoint,
                        "game_record_tail": previous_game,
                        "resources": resources_after,
                        "formal_result_eligibility": False,
                    },
                )

            external = policies._scorers["retained-v4"]._policy.malom
            for item, state, turns, _case, _source in scripted_cases:
                record = replay_scripted_terminal_game(
                    schedule_item=item,
                    start_state=state,
                    continuation_turns=turns,
                    plan=plan,
                    database=database,
                    installation=installation,
                    ledger=ledger,
                )
                resources_after = ledger.record()
                previous_checkpoint = append_resource_checkpoint(
                    resource_journal,
                    completion_index=len(records),
                    complete_games_before=int(sunk["complete_games"]),
                    game_record=record,
                    resources_before=resources_before,
                    resources_after=resources_after,
                    previous_checkpoint_sha256=previous_checkpoint,
                )
                previous_game = append_game_record(
                    raw_games,
                    record,
                    previous_record_sha256=previous_game,
                )
                resources_before = resources_after
                records.append(record)
                write_json_atomic(
                    output_dir / "progress.json",
                    {
                        "completed_games": len(records),
                        "expected_games": expected_games,
                        "resource_checkpoint_tail": previous_checkpoint,
                        "game_record_tail": previous_game,
                        "resources": resources_after,
                        "formal_result_eligibility": False,
                    },
                )
            decisive_case = next(
                row for row in scripted_cases if row[3]["outcome_class"] == "decisive"
            )
            terminal_canary = _terminal_malom_canary(
                external=external,
                decisive_state=decisive_case[1],
                decisive_record=decisive_case[4],
                ledger=ledger,
            )
    finally:
        database.close()

    resource_recovery = load_resource_checkpoints(
        resource_journal,
        expected_baseline=baseline,
        complete_games_before=int(sunk["complete_games"]),
    )
    game_recovery = load_game_records(raw_games)
    verify_resource_game_alignment(resource_recovery, game_recovery)
    if (
        resource_recovery["checkpoint_count"] != expected_games
        or game_recovery["record_count"] != expected_games
        or resource_recovery["last_resources"] != resources_before
    ):
        raise TrainedModelBaselineError("rehearsal durable recovery differs")

    live = [row for row in records if row["arm"] in ARMS]
    scripted = [row for row in records if row["arm"] not in ARMS]
    matrix = Counter(
        (str(row["arm"]), str(row["candidate_color"]), str(row["phase"]))
        for row in live
    )
    expected_matrix = Counter(
        (arm, color, phase)
        for phase in PHASES
        for arm in ARMS
        for color in ("W", "B")
    )
    candidate_choices = [
        turn["candidate_choice"]
        for row in live
        for turn in row["turns"]
        if turn["actor"] == "candidate"
    ]
    safety_modes = Counter(choice["safety_mode"] for choice in candidate_choices)
    turn_phases = Counter(
        str(turn["phase"]) for row in live for turn in row["turns"]
    )
    reasons = Counter(str(row["outcome_reason"]) for row in records)
    required_reasons = {
        "drawThreefoldRepetition",
        "drawFiftyMove",
    }
    if (
        matrix != expected_matrix
        or any(row["termination_class"] != "rules_terminal" for row in records)
        or not required_reasons <= set(reasons)
        or not any(row["winner"] is not None for row in records)
        or any(turn_phases[phase] == 0 for phase in PHASES)
        or safety_modes["free"] == 0
        or safety_modes["A_pos-constrained"] == 0
        or malom_canary is None
        or terminal_canary is None
    ):
        raise TrainedModelBaselineError("trained-model required coverage failed")

    compact_games = [compact_game(row) for row in records]
    sealed_writer_canary = write_sealed_json(
        output_dir / "sealed-writer-canary.json",
        {
            "schema_version": "nmm.boundary-sealed-writer-canary.v1",
            "formal_result_eligibility": False,
            "attempt_identity": attempt["attempt_identity"],
        },
        identity_field="canary_identity",
    )
    coverage_recorder.__exit__(None, None, None)
    dynamic_coverage = verify_rehearsal_coverage(coverage_ledger, registry)
    if (
        dynamic_coverage["passed"] is not True
        or dynamic_coverage["expected_boundary_ids"]
        != contract["expected_boundary_ids"]
    ):
        raise TrainedModelBaselineError("registry dynamic coverage differs")

    payload = {
        "schema_version": REHEARSAL_SCHEMA,
        "status": "passed_non_evidence_technical_rehearsal",
        "formal_result_eligibility": False,
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": plan_sha,
        "attempt_identity": attempt["attempt_identity"],
        "attempt_file_sha256": attempt_sha,
        "authorization_identity": authorization["authorization_identity"],
        "authorization_file_sha256": authorization_sha,
        "start_pool_identity": pool["pool_identity"],
        "start_pool_file_sha256": pool_sha,
        "source_commit": _git("rev-parse", "HEAD"),
        "source_tree": _git("rev-parse", "HEAD^{tree}"),
        "sanmill_runtime": runtime,
        "malom_snapshot": malom,
        "instrumentation_audit": instrumentation,
        "boundary_registry": {
            "identity": registry["registry_identity"],
            "file_sha256": registry_sha,
            "coverage_contract": contract,
            "dynamic_coverage": dynamic_coverage,
            "sealed_writer_canary": sealed_writer_canary,
        },
        "prior_failed_attempts_preservation": preserved_history,
        "malom_surface_canary": malom_canary,
        "terminal_malom_canary": terminal_canary,
        "coverage": {
            "live_model_games": len(live),
            "scripted_terminal_games": len(scripted),
            "arm_color_start_phase_cells": len(matrix),
            "arm_color_start_phase_matrix_complete": matrix == expected_matrix,
            "live_games_by_arm": dict(sorted(Counter(row["arm"] for row in live).items())),
            "live_games_by_candidate_color": dict(
                sorted(Counter(row["candidate_color"] for row in live).items())
            ),
            "live_start_phases": dict(
                sorted(Counter(row["phase"] for row in live).items())
            ),
            "actual_turn_phases": dict(sorted(turn_phases.items())),
            "candidate_safety_modes": dict(sorted(safety_modes.items())),
            "rules_terminal_games": len(records),
            "draw_games": sum(row["winner"] is None for row in records),
            "decisive_games": sum(row["winner"] is not None for row in records),
            "termination_reasons": dict(sorted(reasons.items())),
            "threefold_draw_path": reasons["drawThreefoldRepetition"] > 0,
            "fifty_move_draw_path": reasons["drawFiftyMove"] > 0,
            "decisive_path": any(row["winner"] is not None for row in records),
            "result_packaging": True,
            "resource_checkpoint_before_game_record": True,
            "durable_recovery_and_alignment": True,
            "completion_and_analysis": True,
        },
        "games": compact_games,
        "boundary_coverage_event_ledger": {
            "path": str(coverage_ledger.relative_to(_ROOT)).replace("\\", "/"),
            "records": dynamic_coverage["event_count"],
            "file_sha256": dynamic_coverage["file_sha256"],
            "tail_event_sha256": dynamic_coverage["tail_event_sha256"],
            "coverage_ledger_identity": dynamic_coverage[
                "coverage_ledger_identity"
            ],
            "tracked": False,
            "formal_result_eligibility": False,
        },
        "raw_game_ledger": {
            "path": str(raw_games.relative_to(_ROOT)).replace("\\", "/"),
            "records": game_recovery["record_count"],
            "file_sha256": game_recovery["file_sha256"],
            "tail_record_sha256": game_recovery["tail_record_sha256"],
            "tracked": False,
        },
        "resource_checkpoint_journal": {
            "path": str(resource_journal.relative_to(_ROOT)).replace("\\", "/"),
            "records": resource_recovery["checkpoint_count"],
            "file_sha256": resource_recovery["file_sha256"],
            "tail_checkpoint_sha256": resource_recovery[
                "tail_checkpoint_sha256"
            ],
            "tracked": False,
        },
        "resource_use": {
            **resources_before,
            "complete_games": len(records),
            "cumulative_complete_games": (
                int(sunk["complete_games"]) + len(records)
            ),
            "formal_reused_starts": 0,
            "includes_failed_attempt_sunk_resources": True,
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
        identity_field="rehearsal_identity",
    )
    write_json_atomic(
        output_dir / "rehearsal-completed.json",
        {
            "rehearsal_identity": sealed["rehearsal_identity"],
            "formal_result_eligibility": False,
            "resources": resources_before,
        },
    )
    print(sealed["rehearsal_identity"])
    print(json.dumps(payload["coverage"], sort_keys=True))
    print(json.dumps(resources_before, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
