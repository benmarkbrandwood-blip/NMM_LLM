#!/usr/bin/env python3
"""Run the zero-formal-game trained-model baseline preflight."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.human_db import HumanDB
from ai.malom_db import MalomDB
from game.board import BoardState
from learned_ai.data.specialist_db import SpecialistDB
from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    verify_malom_snapshot,
    write_sealed_json,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    ResourceLedger,
    _checked_oracle_inventory,
    load_pool,
    load_resource_checkpoints,
    replay_start,
    sha256_file,
    write_json_atomic,
)
from learned_ai.evaluation.sanmill_safe_inducement import (
    WDL_RANK,
    run_determinism_gate,
)
from learned_ai.evaluation import sanmill_trained_model_baseline as baseline
from learned_ai.evaluation.sanmill_trained_model_baseline import (
    PREFLIGHT_SCHEMA,
    TrainedModelBaselineError,
    audit_instrumentation_surface,
    audit_specialist_gameai_dependency,
    formal_states,
    load_attempt_authorization,
    load_attempt_spec,
    load_model_policies,
    load_plan,
    load_rehearsal,
    specialist_runtime_record,
)
from learned_ai.training.checkpoint_envelope import load_checkpoint
from learned_ai.training.sanmill_referee import (
    SanmillTrainingGame,
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


def _run_gate(command: list[str]) -> dict[str, Any]:
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
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if result.returncode:
        raise TrainedModelBaselineError(
            "verification gate failed: " + " ".join(command)
        )
    return record


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


def _fixtures(states: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values = []
    for phase in baseline.PHASES:
        matches = [row for row in states if row["phase"] == phase]
        if not matches or not matches[0]["a_pos"]:
            raise TrainedModelBaselineError("preflight phase fixture is absent")
        values.append({"state_id": matches[0]["state_id"], "a_pos_index": 0})
    return values


def _score_snapshot(scorer: Any, board: BoardState) -> dict[str, Any]:
    moves, scores, route_phase = scorer.score(board)
    array = np.asarray(scores, dtype=np.float64)
    if not moves or array.shape != (len(moves),) or not np.isfinite(array).all():
        raise TrainedModelBaselineError("candidate canary score contract differs")
    index = int(np.argmax(array))
    return {
        "route_phase": route_phase,
        "moves_identity": canonical_sha256(
            [baseline._normal_move(move) for move in moves]
        ),
        "scores_identity": canonical_sha256(array.tolist()),
        "argmax_move": baseline._normal_move(moves[index]),
        "argmax_score": float(array[index]),
        "legal_moves": len(moves),
    }


def _candidate_determinism(
    *,
    policies: Any,
    states: Sequence[Mapping[str, Any]],
    database: MalomDB,
    ledger: ResourceLedger,
) -> dict[str, Any]:
    observations = []
    fixture_states = [
        next(row for row in states if row["phase"] == phase)
        for phase in baseline.PHASES
    ]
    for state in fixture_states:
        board = BoardState.from_fen_string(str(state["fen"]))
        queries_before = ledger.malom_queries
        parent, inventory, queries = _checked_oracle_inventory(board, database)
        if ledger.malom_queries - queries_before != queries:
            raise TrainedModelBaselineError(
                "candidate determinism Malom accounting differs"
            )
        best_rank = max(WDL_RANK[value.outcome] for _move, value in inventory)
        safe_keys = {
            baseline._move_key(move)
            for move, value in inventory
            if WDL_RANK[value.outcome] == best_rank
        }
        per_candidate: dict[str, list[dict[str, Any]]] = {
            candidate: [] for candidate in baseline.CANDIDATES
        }
        for order_name, candidates in (
            ("forward", baseline.CANDIDATES),
            ("reverse", tuple(reversed(baseline.CANDIDATES))),
        ):
            for candidate in candidates:
                scorer = policies._scorers[candidate]
                for repeat in range(2):
                    snapshot = _score_snapshot(scorer, board)
                    snapshot.update({"order": order_name, "repeat": repeat})
                    per_candidate[candidate].append(snapshot)
        rows = {}
        for candidate, values in per_candidate.items():
            semantic = [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"order", "repeat"}
                }
                for row in values
            ]
            if any(value != semantic[0] for value in semantic[1:]):
                raise TrainedModelBaselineError("candidate determinism canary failed")
            scorer = policies._scorers[candidate]
            moves, scores, _phase = scorer.score(board)
            free_move, free = baseline._select_scored_move(
                legal_moves=moves,
                scores=np.asarray(scores, dtype=np.float64),
                allowed_keys={baseline._move_key(move) for move in moves},
            )
            safe_move, safe = baseline._select_scored_move(
                legal_moves=moves,
                scores=np.asarray(scores, dtype=np.float64),
                allowed_keys=safe_keys,
            )
            rows[candidate] = {
                "repeats": values,
                "identical": True,
                "free_argmax_move": baseline._normal_move(free_move),
                "free_argmax_score": free["selected_score"],
                "a_pos_argmax_move": baseline._normal_move(safe_move),
                "a_pos_argmax_score": safe["selected_score"],
            }
        observations.append(
            {
                "state_id": state["state_id"],
                "phase": state["phase"],
                "parent_tier": parent,
                "a_pos_cardinality": len(safe_keys),
                "candidates": rows,
            }
        )

    class _PoisonGameAI:
        def __getattribute__(self, name: str) -> Any:
            if name.startswith("__"):
                return object.__getattribute__(self, name)
            raise AssertionError(f"warmed GameAI was read: {name}")

    specialist = policies._scorers["active-specialists"]
    board = BoardState.from_fen_string(str(fixture_states[0]["fen"]))
    before = _score_snapshot(specialist, board)
    specialist._router.set_gameai(_PoisonGameAI())
    after = _score_snapshot(specialist, board)
    specialist._router.set_gameai(None)
    if before != after:
        raise TrainedModelBaselineError("specialist GameAI poison canary differs")
    return {
        "passed": True,
        "fixtures": len(fixture_states),
        "same_process_repeats_per_order": 2,
        "opposite_candidate_order": True,
        "observations": observations,
        "specialist_gameai_poison_canary": {
            "passed": True,
            "snapshot": before,
        },
    }


def _checkpoint_metadata(plan: Mapping[str, Any]) -> dict[str, Any]:
    v4 = plan["candidate_runtime"]["retained_v4"]["checkpoint"]
    envelope = load_checkpoint(_ROOT / str(v4["path"]), map_location="cpu")
    if envelope.payload_sha256 != v4["payload_sha256"]:
        raise TrainedModelBaselineError("retained-v4 checkpoint payload differs")
    specialist = plan["candidate_runtime"]["active_specialists"]
    rows = {}
    for phase, key in (
        ("open", "checkpoint_open"),
        ("mid", "checkpoint_mid"),
        ("end", "checkpoint_end"),
    ):
        resource = specialist["resource_files"][key]
        payload = torch.load(
            _ROOT / str(resource["path"]),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(payload, Mapping):
            raise TrainedModelBaselineError("specialist checkpoint payload differs")
        config = dict(payload.get("model_config", {}))
        if int(config.get("move_feat_dim", -1)) != 134:
            raise TrainedModelBaselineError("specialist feature width differs")
        rows[phase] = {
            "file_sha256": resource["sha256"],
            "bytes": resource["bytes"],
            "model_config": config,
            "game_count": payload.get("game_count"),
            "difficulty": payload.get("difficulty"),
            "best_rate": payload.get("best_rate"),
            "learning_rate": payload.get("learning_rate"),
            "temperature": payload.get("temperature"),
        }
    return {
        "retained_v4": {
            "checkpoint_id": envelope.descriptor.checkpoint_id,
            "payload_sha256": envelope.payload_sha256,
            "file_sha256": v4["sha256"],
        },
        "active_specialists": {
            "lineage": "untraceable",
            "checkpoints": rows,
        },
    }


def _exposure_audit(
    *,
    plan: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    v4 = plan["candidate_runtime"]["retained_v4"]
    human = HumanDB(
        _ROOT / str(v4["human_db"]["path"]),
        read_only=True,
        immutable=True,
    )
    specialist = SpecialistDB(
        _ROOT / str(v4["specialist_db"]["path"]),
        read_only=True,
    )
    specialist.require_trusted_malom_labels()
    rows = []
    try:
        for state in states:
            board = BoardState.from_fen_string(str(state["fen"]))
            human_stats = human.query_position(board)
            specialist_stats = specialist.query_wdl_evidence(board, min_samples=1)
            rows.append(
                {
                    "start_id": state["state_id"],
                    "source_phase": state["phase"],
                    "v4_training_human_db_exact_state_present": (
                        human_stats is not None
                    ),
                    "v4_training_specialist_db_exact_state_present": (
                        specialist_stats is not None
                    ),
                    "human_db_total_games": (
                        None if human_stats is None else human_stats.total_games
                    ),
                    "specialist_db_empirical_total": (
                        None
                        if specialist_stats is None
                        else sum(specialist_stats.empirical_counts)
                    ),
                }
            )
    finally:
        specialist.close()
        human.close()
    return {
        "starts": len(rows),
        "strata": {
            "v4_human_db_exact_state_present": sum(
                row["v4_training_human_db_exact_state_present"] for row in rows
            ),
            "v4_specialist_db_exact_state_present": sum(
                row["v4_training_specialist_db_exact_state_present"] for row in rows
            ),
        },
        "rows_identity": canonical_sha256(rows),
        "rows": rows,
        "source_game_exposure": (
            "indeterminate because aggregate HumanDB/SpecialistDB do not preserve "
            "per-game membership"
        ),
        "active_specialist_training_exposure": (
            "indeterminate because the three serving checkpoints have untraceable lineage"
        ),
        "selection_was_candidate_independent": True,
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
        "--pool",
        default="docs/experiments/sanmill-safe-guidance-gameplay-start-pool-v1.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "docs/evidence/sanmill-trained-model-baseline-v1-"
            "attempt-002-preflight-2026-08-16.json"
        ),
    )
    parser.add_argument("--paths-config", default="data/training_paths.local.json")
    parser.add_argument(
        "--malom-manifest", default="data/manifests/malom-sector-corrected-v1.json"
    )
    args = parser.parse_args()

    if _git("branch", "--show-current") != "dev":
        parser.error("preflight requires dev")
    if _git("status", "--short", "--untracked-files=no"):
        parser.error("tracked worktree must be clean before preflight")
    if _running_tgf_processes() != 0:
        parser.error("a Sanmill process is already running")
    plan, plan_sha = load_plan(_ROOT / args.plan)
    attempt, attempt_sha = load_attempt_spec(_ROOT / args.attempt)
    authorization, authorization_sha = load_attempt_authorization(
        _ROOT / args.authorization
    )
    rehearsal, rehearsal_sha = load_rehearsal(_ROOT / args.rehearsal)
    pool, pool_sha = load_pool(_ROOT / args.pool)
    if (
        attempt["plan"]["identity"] != plan["plan_identity"]
        or attempt["plan"]["file_sha256"] != plan_sha
        or authorization["attempt"]["identity"] != attempt["attempt_identity"]
        or authorization["attempt"]["file_sha256"] != attempt_sha
        or authorization["status"] != "authorized_once_measurement_unconsumed"
        or attempt["resource_envelope"] != plan["resource_envelope"]
        or rehearsal["plan_identity"] != plan["plan_identity"]
        or rehearsal["attempt_identity"] != attempt["attempt_identity"]
        or rehearsal["authorization_identity"]
        != authorization["authorization_identity"]
        or rehearsal["status"] != "passed_non_evidence_technical_rehearsal"
        or rehearsal["formal_result_eligibility"] is not False
        or pool["pool_identity"] != plan["start_pool"]["pool_identity"]
        or pool_sha != plan["start_pool"]["pool_file_sha256"]
    ):
        parser.error("preflight frozen bindings differ")

    formal = formal_states(
        pool,
        excluded_start_ids=plan["start_pool"]["excluded_start_ids"],
    )
    formal_ids = sorted(str(row["state_id"]) for row in formal)
    if canonical_sha256(formal_ids) != plan["start_pool"][
        "formal_membership_identity"
    ]:
        parser.error("preflight formal membership differs")
    rehearsal_output = _ROOT / str(attempt["outputs"]["rehearsal_namespace"])
    rehearsal_tree = _tree_identity(rehearsal_output)
    rehearsal_baseline = json.loads(
        (rehearsal_output / "resource-baseline.json").read_text(encoding="utf-8")
    )
    rehearsal_recovery = load_resource_checkpoints(
        rehearsal_output / "resource-checkpoints.jsonl",
        expected_baseline=rehearsal_baseline,
        complete_games_before=int(
            attempt["cumulative_sunk_resources_before_attempt_002"]["complete_games"]
        ),
    )
    if (
        rehearsal_recovery["checkpoint_count"]
        != int(attempt["rehearsal"]["complete_games"])
        or rehearsal_recovery["last_resources"]["engine_single_step_searches"]
        != rehearsal["resource_use"]["engine_single_step_searches"]
        or rehearsal_recovery["last_resources"]["malom_read_only_queries"]
        != rehearsal["resource_use"]["malom_read_only_queries"]
    ):
        parser.error("rehearsal durable resource record differs")

    output_path = _ROOT / args.output
    run_output = _ROOT / str(attempt["outputs"]["formal_output_namespace"])
    if output_path != _ROOT / str(attempt["outputs"]["preflight_result"]):
        parser.error("preflight output differs from frozen attempt")
    if output_path.exists() or run_output.exists():
        parser.error("preflight result or formal output namespace already exists")
    run_output.mkdir(parents=True, exist_ok=False)
    verification_started = time.perf_counter()
    python = str(_ROOT / ".venv/Scripts/python.exe")
    regression_nodes = [
        "test_allowed_argmax_rejects_a_truly_missing_move",
        "test_game_record_rejects_truly_mismatched_terminal_winner",
        "test_resource_checkpoint_survives_before_game_record_crash",
        "test_counting_malom_proxy_records_each_completed_query",
        "test_instrumentation_surface_is_complete_and_signature_compatible",
        "test_instrumentation_surface_rejects_generic_signature_drift",
        "test_instrumentation_surface_rejects_unregistered_intercept_method",
        "test_specialist_gameai_audit_proves_no_score_path_read",
        "test_protected_guard_fails_before_any_content_producer",
    ]
    regression = _run_gate(
        [
            python,
            "-m",
            "pytest",
            *[
                "tests/test_sanmill_trained_model_baseline.py::" + node
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
            "tests/test_sanmill_trained_model_baseline.py",
            "tests/test_training_aligned_policy.py",
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
        "ai/malom_db.py",
        "learned_ai/sentinel/db_teacher.py",
        "learned_ai/evaluation/sanmill_safe_guidance_gameplay.py",
        "learned_ai/evaluation/sanmill_safe_inducement.py",
        "learned_ai/evaluation/training_aligned_policy.py",
        "learned_ai/evaluation/sanmill_trained_model_baseline.py",
        "scripts/rehearse_sanmill_trained_model_baseline_attempt_002.py",
        "scripts/preflight_sanmill_trained_model_baseline.py",
        "scripts/run_sanmill_trained_model_baseline.py",
        "tests/test_training_aligned_policy.py",
        "tests/test_sanmill_trained_model_baseline.py",
    ]
    ruff = _run_gate(["ruff", "check", *ruff_targets])
    instrumentation = audit_instrumentation_surface(_ROOT)
    if instrumentation["passed"] is not True:
        raise TrainedModelBaselineError("instrumentation surface audit failed")

    paths = _paths(_ROOT / args.paths_config)
    checkout = _local_path(paths.get("sanmill_training_checkout"), key="sanmill")
    malom_path = _local_path(paths.get("malom_db_path"), key="malom")
    installation = inspect_sanmill_training_installation(checkout)
    runtime = training_installation_record(
        installation, seed=int(plan["sanmill_contract"]["seed"])
    )
    if runtime["identity"] != plan["sanmill_contract"]["runtime_identity"]:
        raise TrainedModelBaselineError("Sanmill runtime differs from attempt-002")
    malom = verify_malom_snapshot(
        malom_path=malom_path,
        manifest_path=_ROOT / args.malom_manifest,
        full_hash=False,
    )
    if (
        malom["trust_level"] != "sector-corrected-v1"
        or malom["content_sha256"] != plan["malom_contract"]["content_sha256"]
    ):
        raise TrainedModelBaselineError("Malom snapshot differs")

    elapsed_before_runtime = time.perf_counter() - verification_started
    rehearsal_resources = rehearsal["resource_use"]
    envelope = plan["resource_envelope"]
    ledger = ResourceLedger(
        engine_searches=int(rehearsal_resources["engine_single_step_searches"]),
        malom_queries=int(rehearsal_resources["malom_read_only_queries"]),
        active_seconds_before_run=(
            float(rehearsal_resources["active_seconds"])
            + elapsed_before_runtime
        ),
        maximum_engine_searches=int(envelope["maximum_engine_single_step_searches"]),
        maximum_malom_queries=int(envelope["maximum_malom_queries"]),
        maximum_active_seconds=float(envelope["maximum_active_seconds"]),
    )
    runtime_plan = dict(plan)
    runtime_plan["determinism_gate"] = {
        "fixtures": _fixtures(formal),
        "budgets": plan["preflight"]["sanmill_determinism_budgets"],
    }
    engine_before = ledger.engine_searches
    sanmill_determinism = run_determinism_gate(
        installation=installation,
        pool={**pool, "states": formal},
        plan=runtime_plan,
        query_counter=ledger.add_engine,
    )
    if sanmill_determinism["passed"] is not True:
        raise TrainedModelBaselineError("Sanmill determinism gate failed")

    database = MalomDB(malom_path, query_observer=ledger.add_malom)
    try:
        with load_model_policies(
            plan=plan,
            root=_ROOT,
            malom_path=malom_path,
            malom_manifest_path=_ROOT / args.malom_manifest,
            ledger=ledger,
        ) as policies:
            candidate_determinism = _candidate_determinism(
                policies=policies,
                states=formal,
                database=database,
                ledger=ledger,
            )
    finally:
        database.close()

    checkpoint_metadata = _checkpoint_metadata(plan)
    gameai_audit = audit_specialist_gameai_dependency()
    if gameai_audit["score_path_reads_gameai"] is not False:
        raise TrainedModelBaselineError("specialist score path reads warmed GameAI")
    strict_starts = []
    for state in formal:
        with SanmillTrainingGame(
            installation, seed=int(plan["sanmill_contract"]["seed"])
        ) as game:
            _board, strict = replay_start(game, state, ledger)
        strict_starts.append(
            {
                "state_id": state["state_id"],
                "logical_ply_count": strict["logical_ply_count"],
                "no_capture_count": strict["no_capture_count"],
                "repetition_current_count": strict["repetition_current_count"],
                "history_sha256": strict["history_sha256"],
            }
        )
    exposure = _exposure_audit(plan=plan, states=formal)
    aggregate = ledger.record()
    aggregate.update(
        {
            "complete_games": int(rehearsal_resources["complete_games"]),
            "formal_reused_starts": 0,
        }
    )
    if (
        aggregate["complete_games"] + int(plan["experiment"]["formal_games"])
        > int(envelope["maximum_complete_games"])
        or aggregate["engine_single_step_searches"]
        >= int(envelope["maximum_engine_single_step_searches"])
        or aggregate["malom_read_only_queries"]
        >= int(envelope["maximum_malom_queries"])
        or aggregate["active_seconds"] >= float(envelope["maximum_active_seconds"])
    ):
        raise TrainedModelBaselineError("preflight reached a resource envelope")

    implementation_files = authorization["implementation_files"]
    observed_implementation = {
        path: sha256_file(_ROOT / path) for path in implementation_files
    }
    if (
        observed_implementation != implementation_files
        or implementation_files != attempt["implementation_files"]
    ):
        raise TrainedModelBaselineError("authorized implementation files changed")
    source_commit = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    preflight_seconds = time.perf_counter() - verification_started
    payload = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "ready_for_one_authorized_execution",
        "formal_complete_games": 0,
        "measurement_marker_created": False,
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": plan_sha,
        "attempt_identity": attempt["attempt_identity"],
        "attempt_file_sha256": attempt_sha,
        "authorization_identity": authorization["authorization_identity"],
        "authorization_file_sha256": authorization_sha,
        "rehearsal_identity": rehearsal["rehearsal_identity"],
        "rehearsal_file_sha256": rehearsal_sha,
        "rehearsal_output_tree": rehearsal_tree,
        "start_pool_identity": pool["pool_identity"],
        "start_pool_file_sha256": pool_sha,
        "formal_start_membership_identity": canonical_sha256(formal_ids),
        "formal_starts": len(formal),
        "formal_games": int(plan["experiment"]["formal_games"]),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "run_output_namespace": attempt["outputs"]["formal_output_namespace"],
        "run_output_was_absent_before_preflight": True,
        "verification": {
            "contract_and_crash_regressions": regression,
            "focused_pytest": focused,
            "mandatory_malom_db_teacher_provenance": mandatory,
            "task_scope_ruff": ruff,
            "instrumentation_surface_audit": instrumentation,
        },
        "sanmill_runtime": runtime,
        "sanmill_runtime_matches_attempt_002": True,
        "malom_snapshot": malom,
        "sanmill_determinism": sanmill_determinism,
        "candidate_determinism": candidate_determinism,
        "candidate_checkpoint_metadata": checkpoint_metadata,
        "specialist_runtime": specialist_runtime_record(plan),
        "strict_start_validation": {
            "starts": len(strict_starts),
            "all_nonterminal": True,
            "all_histories_replayed": True,
            "clock_records_identity": canonical_sha256(strict_starts),
        },
        "start_exposure": exposure,
        "resource_components": {
            "non_evidence_rehearsal": rehearsal_resources,
            "zero_formal_game_preflight": {
                "engine_single_step_searches": (
                    ledger.engine_searches - engine_before
                ),
                "malom_read_only_queries": (
                    ledger.malom_queries
                    - int(rehearsal_resources["malom_read_only_queries"])
                ),
                "active_seconds": preflight_seconds,
                "complete_games": 0,
            },
        },
        "aggregate_resource_use_before_measurement": aggregate,
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
            "model_fits_or_tuning": 0,
            "training_or_weight_updates": 0,
            "checkpoint_edits_copies_renames_or_alias_changes": 0,
            "database_writes": 0,
        },
        "historical_sanmill_checkout_route": {
            "path": str(paths.get("sanmill_checkout", "")),
            "used_for_this_experiment": False,
            "training_checkout_used_instead": True,
            "known_fail_closed_drift_not_hidden_or_repaired": True,
        },
        "implementation_files": implementation_files,
        "claim_boundary": plan["claim_boundary"],
    }
    sealed = write_sealed_json(
        output_path,
        payload,
        identity_field="preflight_identity",
    )
    write_json_atomic(
        run_output / "authorization-binding.json",
        {
            "plan_identity": plan["plan_identity"],
            "attempt_identity": attempt["attempt_identity"],
            "authorization_identity": authorization["authorization_identity"],
            "preflight_identity": sealed["preflight_identity"],
            "formal_start_membership_identity": canonical_sha256(formal_ids),
        },
    )
    print(sealed["preflight_identity"])
    print(json.dumps(aggregate, sort_keys=True))
    print(json.dumps(exposure["strata"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
