#!/usr/bin/env python3
"""Run the frozen non-evidence trained-model technical rehearsal."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.malom_db import MalomDB
from game.board import BoardState
from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    verify_malom_snapshot,
    write_sealed_json,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    ResourceLedger,
    append_resource_checkpoint,
    compact_game as compact_guidance_game,
    load_pool,
    load_resource_checkpoints,
    replay_scripted_rehearsal_game,
    sha256_file,
    validate_game_record as validate_guidance_game,
    write_json_atomic,
)
from learned_ai.evaluation.sanmill_trained_model_baseline import (
    ARMS,
    GAME_SCHEMA,
    REHEARSAL_SCHEMA,
    TrainedModelBaselineError,
    compact_game,
    formal_states,
    load_authorization,
    load_model_policies,
    load_plan,
    play_game,
    validate_game_record,
)
from learned_ai.training.run_contract import canonical_json_bytes
from learned_ai.training.sanmill_referee import (
    inspect_sanmill_training_installation,
    training_installation_record,
)


THREEFOLD_PREFIX = tuple(
    "d6 f4 d2 b4 e4 d5 c4 d3 g4 d7 a4 d1 e5 e3 c3 c5 f6 b6 "
    "a4-a7 b4-a4 c4-b4 c5-c4 g4-g1 d7-g7 g1-g4 g7-d7 "
    "g4-g1 d7-g7 g1-g4".split()
)
PHASE_CORPUS_SCHEMA = "nmm.retained-phase-process-corpus.v1"


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


def _phase_start(
    plan: Mapping[str, Any],
    pool: Mapping[str, Any],
) -> dict[str, Any]:
    spec = plan["rehearsal"]["live_start"]
    source_path = _ROOT / str(spec["source"])
    if sha256_file(source_path) != spec["source_file_sha256"]:
        raise TrainedModelBaselineError("rehearsal phase corpus differs")
    value = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != PHASE_CORPUS_SCHEMA
        or value.get("corpus_identity") != spec["source_corpus_identity"]
    ):
        raise TrainedModelBaselineError("rehearsal phase corpus identity differs")
    matches = [
        row
        for row in value["records"]
        if row["start_id"] == spec["source_start_id"]
    ]
    if (
        len(matches) != 1
        or matches[0]["record_identity"] != spec["source_record_identity"]
        or matches[0]["strict_start"]["history_sha256"]
        != spec["strict_history_sha256"]
    ):
        raise TrainedModelBaselineError("rehearsal phase record differs")
    row = matches[0]
    state = {
        "state_id": "trained-model-rehearsal-phase-process-001",
        "phase": str(row["phase"]),
        "fen": str(row["fen"]),
        "logical_turns": [list(turn) for turn in row["logical_turns"]],
        "history_actions": list(row["action_history"]),
        "logical_ply": int(row["logical_ply_count"]),
        "strict_history_sha256": str(spec["strict_history_sha256"]),
    }
    formal = formal_states(
        pool,
        excluded_start_ids=plan["start_pool"]["excluded_start_ids"],
    )
    if (
        state["fen"] in {str(item["fen"]) for item in formal}
        or tuple(state["history_actions"])
        in {tuple(item["history_actions"]) for item in formal}
    ):
        raise TrainedModelBaselineError("live rehearsal start overlaps formal pool")
    return state


def _threefold_start(
    plan: Mapping[str, Any],
    pool: Mapping[str, Any],
) -> dict[str, Any]:
    from learned_ai.evaluation.sanmill_safe_guidance_gameplay import _matching_move

    board = BoardState.new_game()
    for action in THREEFOLD_PREFIX:
        board = board.apply_move(_matching_move(board, [action]))
    spec = plan["rehearsal"]["scripted_threefold_case"]
    state = {
        "state_id": "trained-model-rehearsal-threefold",
        "phase": "movement",
        "fen": board.to_fen_string(),
        "logical_turns": [[action] for action in THREEFOLD_PREFIX],
        "history_actions": list(THREEFOLD_PREFIX),
        "logical_ply": len(THREEFOLD_PREFIX),
    }
    formal = formal_states(
        pool,
        excluded_start_ids=plan["start_pool"]["excluded_start_ids"],
    )
    if (
        state["fen"] != spec["fen"]
        or canonical_sha256(state["history_actions"])
        != spec["history_actions_identity"]
        or state["fen"] in {str(item["fen"]) for item in formal}
        or tuple(state["history_actions"])
        in {tuple(item["history_actions"]) for item in formal}
    ):
        raise TrainedModelBaselineError("threefold rehearsal start differs")
    return state


def _decisive_turns(plan: Mapping[str, Any]) -> list[list[str]]:
    spec = plan["rehearsal"]["scripted_decisive_case"]
    path = _ROOT / str(spec["source_ledger"])
    if sha256_file(path) != spec["source_ledger_sha256"]:
        raise TrainedModelBaselineError("decisive rehearsal ledger differs")
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        wrapper = json.loads(line)
        if wrapper.get("record", {}).get("game_id") == spec["source_game_id"]:
            matches.append(wrapper)
    if (
        len(matches) != 1
        or matches[0].get("record_sha256") != spec["source_record_sha256"]
        or matches[0]["record"].get("winner") is None
        or matches[0]["record"].get("termination_class") != "rules_terminal"
    ):
        raise TrainedModelBaselineError("decisive rehearsal record differs")
    return [list(turn["actions"]) for turn in matches[0]["record"]["turns"]]


def _item(
    *,
    ordinal: int,
    state: Mapping[str, Any],
    arm: str,
    candidate_color: str,
) -> dict[str, Any]:
    body = {
        "namespace": "sanmill-trained-model-baseline-v1-rehearsal-game",
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


def _append_any_record(
    path: Path,
    record: Mapping[str, Any],
    *,
    previous_sha256: str | None,
) -> str:
    if record.get("schema_version") == GAME_SCHEMA:
        validate_game_record(record)
    else:
        validate_guidance_game(record)
    body = {**record, "previous_record_sha256": previous_sha256}
    identity = canonical_sha256(body)
    wrapper = {"record": body, "record_sha256": identity}
    with path.open("xb" if previous_sha256 is None else "ab") as handle:
        handle.write(canonical_json_bytes(wrapper) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return identity


def _load_any_records(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise TrainedModelBaselineError("rehearsal game ledger is partial")
    previous = None
    records = []
    for encoded in raw.splitlines():
        wrapper = json.loads(encoded)
        body = wrapper["record"]
        identity = wrapper["record_sha256"]
        if (
            body.get("previous_record_sha256") != previous
            or canonical_sha256(body) != identity
        ):
            raise TrainedModelBaselineError("rehearsal game chain differs")
        record = dict(body)
        record.pop("previous_record_sha256")
        if record.get("schema_version") == GAME_SCHEMA:
            validate_game_record(record)
        else:
            validate_guidance_game(record)
        records.append(record)
        previous = identity
    return {
        "records": records,
        "record_count": len(records),
        "tail_record_sha256": previous,
        "file_sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", default="docs/experiments/sanmill-trained-model-baseline-v1.json"
    )
    parser.add_argument(
        "--authorization",
        default=(
            "docs/experiments/sanmill-trained-model-baseline-v1/authorization.json"
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
    args = parser.parse_args()

    if _git("branch", "--show-current") != "dev":
        parser.error("rehearsal requires dev")
    if _git("status", "--short", "--untracked-files=no"):
        parser.error("tracked worktree must be clean before rehearsal")
    if _running_tgf_processes() != 0:
        parser.error("a Sanmill process is already running")
    plan, plan_sha = load_plan(_ROOT / args.plan)
    authorization, authorization_sha = load_authorization(_ROOT / args.authorization)
    pool, pool_sha = load_pool(_ROOT / args.pool)
    if (
        authorization["plan"]["identity"] != plan["plan_identity"]
        or authorization["plan"]["file_sha256"] != plan_sha
        or authorization["start_pool"]["identity"] != pool["pool_identity"]
        or authorization["start_pool"]["file_sha256"] != pool_sha
        or authorization["status"] != "authorized_once_measurement_unconsumed"
    ):
        parser.error("rehearsal bindings differ")

    output_dir = _ROOT / str(plan["rehearsal"]["output_namespace"])
    result_path = _ROOT / str(plan["rehearsal"]["tracked_result"])
    if output_dir.exists() or result_path.exists():
        parser.error("rehearsal namespace or result already exists")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        output_dir / "NON-EVIDENCE.json",
        {
            "plan_identity": plan["plan_identity"],
            "authorization_identity": authorization["authorization_identity"],
            "formal_result_eligibility": False,
            "purpose": "technical end-to-end rehearsal only",
        },
    )

    envelope = plan["resource_envelope"]
    ledger = ResourceLedger(
        engine_searches=0,
        malom_queries=0,
        active_seconds_before_run=0.0,
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
        raise TrainedModelBaselineError("rehearsal Sanmill runtime differs")
    malom = verify_malom_snapshot(
        malom_path=malom_path,
        manifest_path=_ROOT / args.malom_manifest,
        full_hash=False,
    )
    if (
        malom["trust_level"] != "sector-corrected-v1"
        or malom["content_sha256"] != plan["malom_contract"]["content_sha256"]
    ):
        raise TrainedModelBaselineError("rehearsal Malom snapshot differs")

    live_start = _phase_start(plan, pool)
    threefold_start = _threefold_start(plan, pool)
    decisive_turns = _decisive_turns(plan)
    live_cases = [
        (
            _item(
                ordinal=ordinal,
                state=live_start,
                arm=arm,
                candidate_color=color,
            ),
            live_start,
        )
        for ordinal, (arm, color) in enumerate(
            (arm, color) for arm in ARMS for color in ("W", "B")
        )
    ]
    scripted_cases = [
        (
            _item(
                ordinal=8,
                state=live_start,
                arm="scripted-known-decisive",
                candidate_color="B",
            ),
            live_start,
            decisive_turns,
        ),
        (
            _item(
                ordinal=9,
                state=threefold_start,
                arm="scripted-threefold-draw",
                candidate_color="B",
            ),
            threefold_start,
            [[plan["rehearsal"]["scripted_threefold_case"]["terminal_action"]]],
        ),
    ]
    if len(live_cases) != 8 or len(scripted_cases) != 2:
        raise TrainedModelBaselineError("rehearsal case count differs")

    raw_games = output_dir / "rehearsal-games.jsonl"
    resource_journal = output_dir / "resource-checkpoints.jsonl"
    baseline = ledger.record()
    write_json_atomic(output_dir / "resource-baseline.json", baseline)
    resources_before = baseline
    previous_game = None
    previous_checkpoint = None
    records: list[dict[str, Any]] = []
    database = MalomDB(malom_path)
    try:
        with load_model_policies(
            plan=plan,
            root=_ROOT,
            malom_path=malom_path,
            malom_manifest_path=_ROOT / args.malom_manifest,
            ledger=ledger,
        ) as policies:
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
                    complete_games_before=0,
                    game_record=record,
                    resources_before=resources_before,
                    resources_after=resources_after,
                    previous_checkpoint_sha256=previous_checkpoint,
                )
                previous_game = _append_any_record(
                    raw_games,
                    record,
                    previous_sha256=previous_game,
                )
                resources_before = resources_after
                records.append(record)
                write_json_atomic(
                    output_dir / "progress.json",
                    {
                        "completed_games": len(records),
                        "expected_games": 10,
                        "resource_checkpoint_tail": previous_checkpoint,
                        "game_record_tail": previous_game,
                        "resources": resources_after,
                        "formal_result_eligibility": False,
                    },
                )
        for item, state, turns in scripted_cases:
            record = replay_scripted_rehearsal_game(
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
                complete_games_before=0,
                game_record=record,
                resources_before=resources_before,
                resources_after=resources_after,
                previous_checkpoint_sha256=previous_checkpoint,
            )
            previous_game = _append_any_record(
                raw_games,
                record,
                previous_sha256=previous_game,
            )
            resources_before = resources_after
            records.append(record)
            write_json_atomic(
                output_dir / "progress.json",
                {
                    "completed_games": len(records),
                    "expected_games": 10,
                    "resource_checkpoint_tail": previous_checkpoint,
                    "game_record_tail": previous_game,
                    "resources": resources_after,
                    "formal_result_eligibility": False,
                },
            )
    finally:
        database.close()

    resource_recovery = load_resource_checkpoints(
        resource_journal,
        expected_baseline=baseline,
        complete_games_before=0,
    )
    game_recovery = _load_any_records(raw_games)
    if (
        resource_recovery["checkpoint_count"] != 10
        or game_recovery["record_count"] != 10
        or resource_recovery["last_resources"] != resources_before
    ):
        raise TrainedModelBaselineError("rehearsal durable recovery differs")
    for checkpoint, record in zip(
        resource_recovery["checkpoints"], records, strict=True
    ):
        if checkpoint["game_record_identity"] != canonical_sha256(record):
            raise TrainedModelBaselineError("rehearsal resource/game alignment differs")

    live = [row for row in records if row["schema_version"] == GAME_SCHEMA]
    scripted = [row for row in records if row["schema_version"] != GAME_SCHEMA]
    live_counts = Counter(str(row["arm"]) for row in live)
    all_reasons = Counter(str(row["outcome_reason"]) for row in records)
    draw_games = sum(row["winner"] is None for row in records)
    decisive_games = sum(row["winner"] is not None for row in records)
    if (
        live_counts != Counter({arm: 2 for arm in ARMS})
        or any(row["termination_class"] != "rules_terminal" for row in records)
        or draw_games < 1
        or decisive_games < 1
    ):
        raise TrainedModelBaselineError("rehearsal required coverage failed")
    compact = [compact_game(row) for row in live] + [
        compact_guidance_game(row) for row in scripted
    ]
    payload = {
        "schema_version": REHEARSAL_SCHEMA,
        "status": "passed_non_evidence_technical_rehearsal",
        "formal_result_eligibility": False,
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": plan_sha,
        "authorization_identity": authorization["authorization_identity"],
        "authorization_file_sha256": authorization_sha,
        "start_pool_identity": pool["pool_identity"],
        "start_pool_file_sha256": pool_sha,
        "source_commit": _git("rev-parse", "HEAD"),
        "source_tree": _git("rev-parse", "HEAD^{tree}"),
        "sanmill_runtime": runtime,
        "malom_snapshot": malom,
        "coverage": {
            "live_model_games": len(live),
            "live_games_by_arm": dict(sorted(live_counts.items())),
            "scripted_terminal_contract_games": len(scripted),
            "rules_terminal_games": len(records),
            "draw_games": draw_games,
            "decisive_games": decisive_games,
            "termination_reasons": dict(sorted(all_reasons.items())),
            "result_packaging": True,
            "resource_checkpoint_before_game_record": True,
            "completion_and_analysis": True,
        },
        "games": compact,
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
            "formal_reused_starts": 0,
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
