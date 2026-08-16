#!/usr/bin/env python3
"""Run the frozen four-game non-evidence attempt-002 technical rehearsal."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    RESULT_SCHEMA as READINESS_SCHEMA,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    REHEARSAL_RESULT_SCHEMA,
    ResourceLedger,
    SafeGuidanceGameplayError,
    analyze_rehearsal_records,
    append_game_record,
    append_resource_checkpoint,
    classify_induced_events,
    compact_game,
    load_attempt_spec,
    load_plan,
    load_pool,
    load_resource_checkpoints,
    load_sealed,
    play_game,
    replay_scripted_rehearsal_game,
    sha256_file,
    write_json_atomic,
)
from learned_ai.training.sanmill_referee import (
    inspect_sanmill_training_installation,
    training_installation_record,
)


THREEFOLD_PREFIX = tuple(
    "d6 f4 d2 b4 e4 d5 c4 d3 g4 d7 a4 d1 e5 e3 c3 c5 f6 b6 "
    "a4-a7 b4-a4 c4-b4 c5-c4 g4-g1 d7-g7 g1-g4 g7-d7 "
    "g4-g1 d7-g7 g1-g4".split()
)
THREEFOLD_FINAL = "g7-d7"
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
    preservation = spec["attempt_001_preservation"]
    for key in ("first_output_tree", "consumed_output_tree"):
        expected = preservation[key]
        observed = _tree_identity(_ROOT / str(expected["path"]))
        if observed != {
            "files": expected["files"],
            "bytes": expected["bytes"],
            "file_manifest_identity": expected["file_manifest_identity"],
        }:
            raise SafeGuidanceGameplayError("attempt-001 output tree changed")


def _phase_start(spec: Mapping[str, Any], pool: Mapping[str, Any]) -> dict[str, Any]:
    entry = spec["rehearsal"]["starts"][0]
    source, source_sha = load_sealed(
        _ROOT / str(entry["source_corpus"]),
        schema=PHASE_CORPUS_SCHEMA,
        identity_field="corpus_identity",
    )
    if source_sha != entry["source_corpus_sha256"]:
        raise SafeGuidanceGameplayError("rehearsal phase corpus file differs")
    rows = [
        row for row in source["records"] if row["start_id"] == entry["source_start_id"]
    ]
    if len(rows) != 1 or rows[0]["record_identity"] != entry["source_record_identity"]:
        raise SafeGuidanceGameplayError("rehearsal phase source record differs")
    row = rows[0]
    state = {
        "state_id": entry["rehearsal_start_id"],
        "phase": str(row["phase"]),
        "fen": str(row["fen"]),
        "logical_turns": [list(turn) for turn in row["logical_turns"]],
        "history_actions": list(row["action_history"]),
        "logical_ply": int(row["logical_ply_count"]),
        "oof_fold": 0,
        "strict_history_sha256": str(entry["strict_history_sha256"]),
    }
    if (
        canonical_sha256(state["history_actions"])
        != entry["history_actions_identity"]
        or state["fen"] in {str(item["fen"]) for item in pool["states"]}
        or tuple(state["history_actions"])
        in {tuple(item["history_actions"]) for item in pool["states"]}
    ):
        raise SafeGuidanceGameplayError("phase rehearsal start overlaps frozen pool")
    return state


def _threefold_start(
    spec: Mapping[str, Any], pool: Mapping[str, Any]
) -> dict[str, Any]:
    entry = spec["rehearsal"]["starts"][1]
    board = BoardState.new_game()
    from learned_ai.evaluation.sanmill_safe_guidance_gameplay import _matching_move

    for action in THREEFOLD_PREFIX:
        board = board.apply_move(_matching_move(board, [action]))
    state = {
        "state_id": entry["rehearsal_start_id"],
        "phase": "movement",
        "fen": board.to_fen_string(),
        "logical_turns": [[action] for action in THREEFOLD_PREFIX],
        "history_actions": list(THREEFOLD_PREFIX),
        "logical_ply": len(THREEFOLD_PREFIX),
        "oof_fold": 0,
    }
    if (
        state["fen"] != entry["fen"]
        or canonical_sha256(state["history_actions"])
        != entry["history_actions_identity"]
        or state["fen"] in {str(item["fen"]) for item in pool["states"]}
        or tuple(state["history_actions"])
        in {tuple(item["history_actions"]) for item in pool["states"]}
    ):
        raise SafeGuidanceGameplayError("threefold rehearsal start differs or overlaps")
    return state


def _decisive_turns(spec: Mapping[str, Any]) -> list[list[str]]:
    case = spec["rehearsal"]["cases"][2]
    ledger_path = _ROOT / str(case["source_game_ledger"])
    if sha256_file(ledger_path) != case["source_game_ledger_sha256"]:
        raise SafeGuidanceGameplayError("decisive rehearsal source ledger differs")
    matches = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        wrapper = json.loads(line)
        record = wrapper.get("record", {})
        if record.get("game_id") == case["source_game_id"]:
            matches.append(wrapper)
    if (
        len(matches) != 1
        or matches[0].get("record_sha256") != case["source_record_sha256"]
        or matches[0]["record"].get("termination_class") != "rules_terminal"
        or matches[0]["record"].get("winner") is None
    ):
        raise SafeGuidanceGameplayError("decisive rehearsal source record differs")
    return [list(turn["actions"]) for turn in matches[0]["record"]["turns"]]


def _schedule_item(
    *, ordinal: int, start: Mapping[str, Any], arm: str, candidate_color: str
) -> dict[str, Any]:
    body = {
        "namespace": "sanmill-safe-guidance-gameplay-attempt-002-rehearsal-game-v1",
        "ordinal": ordinal,
        "start_id": start["state_id"],
        "candidate_color": candidate_color,
        "arm": arm,
    }
    return {
        "ordinal": ordinal,
        "unit_index": ordinal,
        "start_index": 0 if ordinal < 3 else 1,
        "start_id": start["state_id"],
        "phase": start["phase"],
        "candidate_color": candidate_color,
        "arm": arm,
        "game_id": canonical_sha256(body),
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
        parser.error("rehearsal requires dev")
    if _git("status", "--short", "--untracked-files=no"):
        parser.error("tracked worktree must be clean before rehearsal")
    if _running_tgf_processes() != 0:
        parser.error("a Sanmill process is already running")
    spec, spec_sha = load_attempt_spec(_ROOT / args.attempt_spec)
    plan, plan_sha = load_plan(_ROOT / args.plan)
    pool, pool_sha = load_pool(_ROOT / args.pool)
    if (
        spec["plan_identity"] != plan["plan_identity"]
        or spec["start_pool_identity"] != pool["pool_identity"]
    ):
        parser.error("rehearsal bindings differ")
    _require_attempt_001_unchanged(spec)
    output_dir = _ROOT / str(spec["rehearsal"]["run_output_namespace"])
    result_path = _ROOT / str(spec["rehearsal"]["tracked_result"])
    if output_dir.exists() or result_path.exists():
        parser.error("rehearsal namespace or result already exists")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        output_dir / "NON-EVIDENCE.json",
        {
            "attempt_identity": spec["attempt_identity"],
            "formal_result_eligibility": False,
            "purpose": "technical end-to-end rehearsal only",
        },
    )

    envelope = spec["resource_envelope"]
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
        raise SafeGuidanceGameplayError("rehearsal Sanmill runtime differs")
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
        raise SafeGuidanceGameplayError("rehearsal Malom snapshot differs")
    readiness, readiness_sha = load_sealed(
        _ROOT / args.readiness_result,
        schema=READINESS_SCHEMA,
        identity_field="result_identity",
    )
    if (
        readiness["result_identity"]
        != plan["input_identities"]["readiness_result_identity"]
    ):
        raise SafeGuidanceGameplayError("rehearsal guide contract differs")

    phase_start = _phase_start(spec, pool)
    threefold_start = _threefold_start(spec, pool)
    decisive_turns = _decisive_turns(spec)
    cases = [
        (
            _schedule_item(
                ordinal=0,
                start=phase_start,
                arm="random-safe",
                candidate_color="W",
            ),
            phase_start,
            None,
        ),
        (
            _schedule_item(
                ordinal=1,
                start=phase_start,
                arm="full-guided",
                candidate_color="W",
            ),
            phase_start,
            None,
        ),
        (
            _schedule_item(
                ordinal=2,
                start=phase_start,
                arm="scripted-known-decisive",
                candidate_color="B",
            ),
            phase_start,
            decisive_turns,
        ),
        (
            _schedule_item(
                ordinal=3,
                start=threefold_start,
                arm="scripted-threefold-draw",
                candidate_color="B",
            ),
            threefold_start,
            [[THREEFOLD_FINAL]],
        ),
    ]
    if len(cases) != int(spec["rehearsal"]["games"]):
        raise SafeGuidanceGameplayError("rehearsal case count differs")

    raw_games = output_dir / "rehearsal-games.jsonl"
    resource_journal = output_dir / "resource-checkpoints.jsonl"
    progress_path = output_dir / "progress.json"
    baseline = ledger.record()
    write_json_atomic(output_dir / "resource-baseline.json", baseline)
    resources_before = baseline
    prior_game_sha = None
    prior_checkpoint_sha = None
    records: list[dict[str, Any]] = []
    database = MalomDB(malom_path)
    try:
        for completion_index, (item, state, scripted) in enumerate(cases):
            if scripted is None:
                record = play_game(
                    schedule_item=item,
                    start_state=state,
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
            else:
                record = replay_scripted_rehearsal_game(
                    schedule_item=item,
                    start_state=state,
                    continuation_turns=scripted,
                    plan=plan,
                    database=database,
                    installation=installation,
                    ledger=ledger,
                )
            resources_after = ledger.record()
            prior_checkpoint_sha = append_resource_checkpoint(
                resource_journal,
                completion_index=completion_index,
                complete_games_before=0,
                game_record=record,
                resources_before=resources_before,
                resources_after=resources_after,
                previous_checkpoint_sha256=prior_checkpoint_sha,
            )
            prior_game_sha = append_game_record(
                raw_games,
                record,
                previous_record_sha256=prior_game_sha,
            )
            resources_before = resources_after
            records.append(record)
            write_json_atomic(
                progress_path,
                {
                    "completed_games": len(records),
                    "expected_games": len(cases),
                    "resource_checkpoint_tail": prior_checkpoint_sha,
                    "game_record_tail": prior_game_sha,
                    "resources": resources_after,
                    "formal_result_eligibility": False,
                },
            )
    finally:
        database.close()

    analysis = analyze_rehearsal_records(records)
    recovered = load_resource_checkpoints(
        resource_journal,
        expected_baseline=baseline,
        complete_games_before=0,
    )
    if (
        recovered["checkpoint_count"] != len(records)
        or recovered["last_resources"] != resources_before
        or len({record["start_id"] for record in records}) != 2
    ):
        raise SafeGuidanceGameplayError("rehearsal durable recovery differs")
    compact = [compact_game(record) for record in records]
    payload = {
        "schema_version": REHEARSAL_RESULT_SCHEMA,
        "status": "passed_non_evidence_technical_rehearsal",
        "formal_result_eligibility": False,
        "attempt_identity": spec["attempt_identity"],
        "attempt_file_sha256": spec_sha,
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": plan_sha,
        "start_pool_identity": pool["pool_identity"],
        "start_pool_file_sha256": pool_sha,
        "source_commit": _git("rev-parse", "HEAD"),
        "source_tree": _git("rev-parse", "HEAD^{tree}"),
        "sanmill_runtime": runtime,
        "malom_snapshot": malom,
        "readiness_result_identity": readiness["result_identity"],
        "readiness_result_file_sha256": readiness_sha,
        "analysis": analysis,
        "games": compact,
        "raw_game_ledger": {
            "path": str(raw_games.relative_to(_ROOT)).replace("\\", "/"),
            "records": len(records),
            "file_sha256": sha256_file(raw_games),
            "tail_record_sha256": prior_game_sha,
            "tracked": False,
        },
        "resource_checkpoint_journal": {
            "path": str(resource_journal.relative_to(_ROOT)).replace("\\", "/"),
            **recovered,
            "tracked": False,
        },
        "resource_use": {
            **resources_before,
            "complete_games": len(records),
            "independent_starts": 2,
            "resource_envelope": envelope,
            "within_all_limits": True,
            "attempt_001_sunk_cost_included": False,
        },
        "attempt_001_preservation_reverified": True,
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
        "claim_boundary": spec["claim_boundary"],
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
    print(json.dumps(analysis, sort_keys=True))
    print(json.dumps(resources_before, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
