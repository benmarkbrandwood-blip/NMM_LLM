#!/usr/bin/env python3
"""Publish the frozen mature target-refresh development result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.data.data_contract import load_dataset_manifest  # noqa: E402
from learned_ai.evaluation.common_anchor_policy_distribution import (  # noqa: E402
    DEFAULT_DIVERGENCE_THRESHOLDS,
)
from learned_ai.evaluation.phase_corpus import validate_phase_corpus  # noqa: E402
from learned_ai.evaluation.phase_replay_development_corpus import (  # noqa: E402
    replay_record_into_sanmill_game,
    validate_phase_replay_development_corpus,
    validate_phase_replay_sanmill_audit,
)
from learned_ai.evaluation.target_refresh_mature_fork_result import (  # noqa: E402
    LEDGER_SCHEMA,
    POLICY_BOUNDARIES,
    RESULT_SCHEMA,
    build_direct_crossplay_schedule,
    classify_mature_policy_divergence,
    decide_mature_result,
    summarize_direct_crossplay,
)
from learned_ai.models.lookahead_advisor import LookaheadAdvisor  # noqa: E402
from learned_ai.sentinel.db_teacher import ExternalSolvedDB  # noqa: E402
from learned_ai.training.checkpoint_envelope import load_checkpoint  # noqa: E402
from learned_ai.training.generalist_preflight import _probe_human_db  # noqa: E402
from learned_ai.training.managed_generalist import (  # noqa: E402
    _validate_policy_health_report,
    load_managed_plan,
)
from learned_ai.training.run_contract import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from learned_ai.training.sanmill_referee import SanmillTrainingGame  # noqa: E402
from learned_ai.validation.sanmill_node_calibration import (  # noqa: E402
    load_local_installation,
)
from learned_ai.validation.target_refresh_mature_fork_diagnostic import (  # noqa: E402
    READINESS_SCHEMA,
    TRAINER_TREATMENT,
    load_contract,
)
from scripts import train_s_gen_v2 as trainer  # noqa: E402
from scripts.analyze_common_anchor_policy_distribution import (  # noqa: E402
    CommonAnchorAnalysisError,
    _build_feature_corpus,
    _compare_checkpoint_pair,
    _load_policy,
    _open_immutable_human_db,
    _read_only_observations,
    _state_dict_sha256,
)
from scripts.run_target_refresh_direct_crossplay import (  # noqa: E402
    _build_policy_generators,
    _normalised_move,
    _sample_policy_move,
)


DEFAULT_CONTRACT = ROOT / (
    "docs/experiments/sanmill-target-refresh-mature-fork-diagnostic-v1-attempt-002.json"
)
DEFAULT_READINESS = ROOT / (
    "out/target-refresh-mature-fork-diagnostic-v1-attempt-002/readiness.json"
)
DEFAULT_POLICY_CORPUS = ROOT / "docs/experiments/dev-v4-phase-covered-corpus-v1.json"
DEFAULT_REPLAY_CORPUS = (
    ROOT / "docs/experiments/dev-v4-phase-replay-development-corpus-v1.json"
)
DEFAULT_REPLAY_AUDIT = ROOT / (
    "docs/evidence/phase-replay-development-corpus-sanmill-audit-2026-08-11.json"
)
DEFAULT_PATHS_CONFIG = ROOT / "data/training_paths.local.json"
DEFAULT_MALOM_MANIFEST = ROOT / "data/manifests/malom-sector-corrected-v1.json"
DEFAULT_LEDGER = ROOT / (
    "out/target-refresh-mature-fork-diagnostic-v1-attempt-002/"
    "development-direct-crossplay-ledger.jsonl"
)
DEFAULT_OUTPUT = ROOT / (
    "out/target-refresh-mature-fork-diagnostic-v1-attempt-002/result.json"
)
_POST_TRAINING_ANALYSIS_PATHS = frozenset(
    {
        "docs/evidence/target-refresh-mature-fork-diagnostic-"
        "attempt-002-failure-2026-08-12.md",
        "docs/experiments/sanmill-target-refresh-mature-fork-"
        "analysis-recovery-v1.json",
        "docs/experiments/sanmill-target-refresh-mature-fork-"
        "analysis-recovery-v1.md",
        "docs/handoff/windows-training-2026-07-20.md",
        "scripts/report_target_refresh_mature_fork_diagnostic.py",
        "scripts/run_target_refresh_mature_fork_analysis_recovery.py",
        "tests/test_target_refresh_mature_fork_analysis_recovery.py",
        "tests/test_target_refresh_mature_fork_diagnostic.py",
        "tests/test_target_refresh_mature_fork_report.py",
    }
)


class MatureTargetRefreshReportError(RuntimeError):
    """Raised when mature target-refresh evidence is incomplete."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise MatureTargetRefreshReportError(
            "evidence path is outside the repository"
        ) from exc


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MatureTargetRefreshReportError(f"cannot read JSON: {path}") from exc
    if (
        not isinstance(value, dict)
        or b"\r" in raw
        or not raw.endswith(b"\n")
        or raw != canonical_json_bytes(value) + b"\n"
    ):
        raise MatureTargetRefreshReportError(f"JSON is not canonical: {path}")
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MatureTargetRefreshReportError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MatureTargetRefreshReportError(f"JSON root is not an object: {path}")
    return value


def _read_hash_bound_json_object(
    path: Path, *, expected_sha256: str, label: str
) -> dict[str, Any]:
    """Load a frozen reference object after proving its exact byte identity."""
    if _sha256_file(path) != expected_sha256:
        raise MatureTargetRefreshReportError(f"{label} identity differs")
    return _read_json_object(path)


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw_lines = path.read_bytes().splitlines(keepends=True)
    except OSError as exc:
        raise MatureTargetRefreshReportError(f"cannot read JSONL: {path}") from exc
    if not raw_lines:
        raise MatureTargetRefreshReportError(f"JSONL is empty: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(raw_lines, 1):
        if not raw.endswith(b"\n") or b"\r" in raw:
            raise MatureTargetRefreshReportError(
                f"JSONL framing differs: {path}:{line_number}"
            )
        try:
            value = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise MatureTargetRefreshReportError(
                f"invalid JSONL row: {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise MatureTargetRefreshReportError(
                f"JSONL row is not an object: {path}:{line_number}"
            )
        rows.append(value)
    return rows


def _git_output(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MatureTargetRefreshReportError("Git audit failed") from exc


def _git_identity(expected_training_commit: str) -> dict[str, Any]:
    head = _git_output("rev-parse", "HEAD")
    origin = _git_output("rev-parse", "origin/dev")
    branch = _git_output("branch", "--show-current")
    dirty = _git_output("status", "--porcelain=v1", "--untracked-files=no")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected_training_commit, head],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if head != origin or branch != "dev" or dirty or ancestor.returncode != 0:
        raise MatureTargetRefreshReportError(
            "result requires a clean published descendant of the training source"
        )
    return {
        "branch": branch,
        "training_head": expected_training_commit,
        "analysis_head": head,
        "origin_dev": origin,
        "tracked_clean": True,
        "published": True,
        "training_source_is_ancestor": True,
    }


def _git_source(expected: str) -> dict[str, Any]:
    source = _git_identity(expected)
    if source["analysis_head"] != expected:
        raise MatureTargetRefreshReportError(
            "result requires the exact clean published training source"
        )
    return {**source, "post_training_analysis_paths": []}


def _inspect_analysis_source(expected_training_commit: str) -> dict[str, Any]:
    """Permit only published, explicitly analysis-only descendant changes."""
    source = _git_identity(expected_training_commit)
    analysis_head = str(source["analysis_head"])
    changed_paths: list[str] = []
    if analysis_head != expected_training_commit:
        changed_paths = sorted(
            path
            for path in _git_output(
                "diff",
                "--name-only",
                f"{expected_training_commit}..{analysis_head}",
                "--",
            ).splitlines()
            if path
        )
        if not changed_paths or not set(changed_paths).issubset(
            _POST_TRAINING_ANALYSIS_PATHS
        ):
            raise MatureTargetRefreshReportError(
                "post-training source changes are not analysis-only"
            )
    return {**source, "post_training_analysis_paths": changed_paths}


def _validate_readiness(
    readiness: Mapping[str, Any], *, contract: Mapping[str, Any]
) -> str:
    if readiness.get("schema_version") != READINESS_SCHEMA:
        raise MatureTargetRefreshReportError("readiness schema differs")
    body = dict(readiness)
    identity = body.pop("readiness_identity", None)
    if identity != canonical_sha256(body):
        raise MatureTargetRefreshReportError("readiness identity differs")
    if (
        readiness.get("state")
        != "six_arm_plans_ready_for_one_parent_product_authorization"
        or readiness.get("launch_authorized") is not False
        or readiness.get("contract", {}).get("plan_identity")
        != contract["plan_identity"]
    ):
        raise MatureTargetRefreshReportError("readiness state differs")
    return str(identity)


def _arm_map(contract: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    return {(int(arm["seed"]), str(arm["condition"])): arm for arm in contract["arms"]}


def _source_map(contract: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(source["seed"]): source for source in contract["sources"]}


def _readiness_seed_map(
    readiness: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    return {int(record["seed"]): record for record in readiness["seeds"]}


def _segment(arm: Mapping[str, Any]) -> Path:
    segment = ROOT / str(arm["control_dir"]) / "segments" / "segment-0001"
    if not segment.is_dir():
        raise MatureTargetRefreshReportError(
            f"completed segment is absent: {arm['arm_id']}"
        )
    return segment


def _load_common_fork(
    *,
    source: Mapping[str, Any],
    arm: Mapping[str, Any],
    readiness_seed: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    path = (ROOT / str(source["common_fork_path"])).resolve(strict=True)
    envelope = load_checkpoint(path, map_location="cpu")
    state = envelope.payload.trainer_state
    recovery = state.get("recovery_state", {})
    fork = recovery.get("target_refresh_fork_state", {})
    publication = readiness_seed.get("common_fork", {})
    if (
        _sha256_file(path) != publication.get("destination_file_sha256")
        or envelope.descriptor.role != "target_refresh_fork"
        or envelope.descriptor.experiment_id != arm["experiment_id"]
        or state.get("game_count") != source["game_count"]
        or state.get("update_count") != source["update_count"]
        or fork.get("captured") is not True
        or fork.get("fork_game") != source["game_count"]
        or fork.get("treatment") is not None
        or fork.get("post_fork_transition_origin") is not None
        or envelope.descriptor.implementation.get("mature_target_refresh_fork_kind")
        != "mature-target-refresh-fork-v1"
    ):
        raise MatureTargetRefreshReportError(
            f"mature common-fork semantics differ: seed {source['seed']}"
        )
    return envelope, {
        "path": _relative(path),
        "file_sha256": _sha256_file(path),
        "checkpoint_id": envelope.descriptor.checkpoint_id,
        "payload_sha256": envelope.payload_sha256,
        "model_state_sha256": _state_dict_sha256(envelope.payload.model_state),
        "game_count": int(state["game_count"]),
        "update_count": int(state["update_count"]),
        "optimizer_consumed_transition_count": int(
            recovery["optimizer_consumed_transition_count"]
        ),
        "pending_transition_count": len(recovery["pending_steps"]),
    }


def _load_candidate_pair(
    *,
    arms: Mapping[tuple[int, str], Mapping[str, Any]],
    readiness_seed: Mapping[str, Any],
    seed: int,
    boundary: int,
    common_record: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    readiness_arms = {str(item["condition"]): item for item in readiness_seed["arms"]}
    models: dict[str, Any] = {}
    records: dict[str, Any] = {}
    model_config: dict[str, Any] | None = None
    immutable_assets: dict[str, Any] | None = None
    for condition in ("refresh-mature", "stale-control"):
        arm = arms[(seed, condition)]
        branch = (
            ROOT / str(arm["control_dir"]) / "initial-mature-target-refresh-fork.pt"
        ).resolve(strict=True)
        branch_envelope = load_checkpoint(branch, map_location=device)
        branch_record = readiness_arms[condition]["branch_checkpoint"]
        treatment = TRAINER_TREATMENT[condition]
        expected_branch_id = f"{common_record['checkpoint_id']}:branch:{treatment}"
        if (
            _sha256_file(branch) != branch_record["destination_sha256"]
            or branch_envelope.descriptor.role != "target_refresh_fork"
            or branch_envelope.descriptor.experiment_id != arm["experiment_id"]
            or branch_envelope.descriptor.checkpoint_id != expected_branch_id
            or branch_envelope.descriptor.parent_checkpoint_id
            != common_record["checkpoint_id"]
            or branch_envelope.payload_sha256 != common_record["payload_sha256"]
            or branch_envelope.descriptor.implementation.get(
                "target_refresh_branch_treatment"
            )
            != treatment
        ):
            raise MatureTargetRefreshReportError(
                f"branch semantics differ: seed {seed} {condition}"
            )
        path = _segment(arm) / f"transition-{boundary:08d}.pt"
        envelope = load_checkpoint(path, map_location=device)
        state = envelope.payload.trainer_state
        recovery = state.get("recovery_state", {})
        fork = recovery.get("target_refresh_fork_state", {})
        expected_implementation = {
            key: value
            for key, value in branch_envelope.descriptor.implementation.items()
            if not key.startswith("target_refresh_branch_")
        }
        origin = fork.get("post_fork_transition_origin")
        consumed = recovery.get("optimizer_consumed_transition_count")
        if (
            envelope.descriptor.role != "transition_diagnostic_candidate"
            or envelope.descriptor.experiment_id != arm["experiment_id"]
            or envelope.descriptor.config_sha256
            != branch_envelope.descriptor.config_sha256
            or dict(envelope.descriptor.implementation) != expected_implementation
            or fork.get("captured") is not True
            or fork.get("fork_game") != common_record["game_count"]
            or fork.get("treatment") != treatment
            or not isinstance(origin, int)
            or not isinstance(consumed, int)
            or consumed - origin != boundary
            or len(recovery.get("pending_steps", [])) >= 64
            or str(recovery.get("source_checkpoint")) != str(branch)
        ):
            raise MatureTargetRefreshReportError(
                f"candidate semantics differ: seed {seed} {condition} {boundary}"
            )
        current_config = dict(state["model_config"])
        if model_config is None:
            model_config = current_config
        elif current_config != model_config:
            raise MatureTargetRefreshReportError(
                f"paired model configurations differ: seed {seed}"
            )
        assets = dict(envelope.descriptor.asset_identities)
        current_assets = {
            key: assets[key]
            for key in (
                "human_db",
                "malom_tablebase",
                "mif_suite_1_0",
                "sanmill_training_runtime",
                "training_ruleset",
            )
        }
        if immutable_assets is None:
            immutable_assets = current_assets
        elif current_assets != immutable_assets:
            raise MatureTargetRefreshReportError(
                f"paired immutable assets differ: seed {seed}"
            )
        model_key = "refresh" if condition == "refresh-mature" else "no-refresh"
        models[model_key] = _load_policy(envelope, device=device)
        records[condition] = {
            "path": _relative(path),
            "file_sha256": _sha256_file(path),
            "checkpoint_id": envelope.descriptor.checkpoint_id,
            "model_state_sha256": _state_dict_sha256(envelope.payload.model_state),
            "game_count": int(state["game_count"]),
            "update_count": int(state["update_count"]),
            "optimizer_consumed_transition_count": consumed,
            "post_mature_fork_consumed_transition_count": boundary,
            "pending_transition_count": len(recovery["pending_steps"]),
            "common_fork_checkpoint_id": common_record["checkpoint_id"],
            "immutable_asset_identities": current_assets,
        }
    return models, records


def _training_outcome(row: Mapping[str, Any]) -> str:
    outcome = float(row["outcome"])
    if outcome == trainer.WIN_REWARD:
        return "win"
    if outcome == trainer.LOSS_REWARD:
        return "loss"
    if outcome in {trainer.DRAW_SHORT, trainer.DRAW_LONG}:
        return "draw"
    raise MatureTargetRefreshReportError("training outcome is unknown")


def _training_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(_training_outcome(row) for row in rows)
    games = len(rows)
    return {
        "games": games,
        "wins": outcomes["win"],
        "draws": outcomes["draw"],
        "losses": outcomes["loss"],
        "score_rate": (
            (outcomes["win"] + 0.5 * outcomes["draw"]) / games if games else None
        ),
        "mean_logical_plies": (
            sum(float(row["ply"]) for row in rows) / games if games else None
        ),
    }


def _training_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    windows = []
    for start in range(0, len(rows), 50):
        window = rows[start : start + 50]
        windows.append(
            {
                "first_game": int(window[0]["game"]),
                "last_game": int(window[-1]["game"]),
                "complete_50_game_window": len(window) == 50,
                **_training_group(window),
                "mean_entropy": sum(float(row["entropy_mean"]) for row in window)
                / len(window),
                "mean_chosen_probability": sum(
                    float(row["chosen_prob_mean"]) for row in window
                )
                / len(window),
                "mean_malom_preserving_rate": sum(
                    float(row["malom_preserving_move_rate"]) for row in window
                )
                / len(window),
            }
        )
    return {
        "overall": _training_group(rows),
        "by_opponent": {
            kind: _training_group([row for row in rows if row.get("game_type") == kind])
            for kind in ("vs_frozen", "vs_sanmill")
        },
        "by_learner_colour": {
            colour: _training_group(
                [row for row in rows if row.get("learner_color") == colour]
            )
            for colour in ("W", "B")
        },
        "by_termination_reason": {
            reason: _training_group(
                [row for row in rows if row.get("termination_reason") == reason]
            )
            for reason in sorted({str(row["termination_reason"]) for row in rows})
        },
        "fixed_blocks_up_to_50_games": windows,
    }


def _audit_training(
    *,
    arms: Mapping[tuple[int, str], Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    expected_counts = list(range(64, 8192 + 1, 64))
    report: dict[str, Any] = {}
    requested_sanmill_nodes = 0
    for seed in (67, 68, 69):
        projections: dict[str, list[dict[str, Any]]] = {}
        seed_report: dict[str, Any] = {}
        for condition in ("refresh-mature", "stale-control"):
            segment = _segment(arms[(seed, condition)])
            updates = _strict_jsonl(segment / "update_log.jsonl")
            selected = [
                row
                for row in updates
                if row.get("post_fork_consumed_transition_count") is not None
                and int(row["post_fork_consumed_transition_count"]) <= 8192
            ]
            counts = [
                int(row["post_fork_consumed_transition_count"]) for row in selected
            ]
            if counts != expected_counts or any(
                int(row.get("batch_steps", 0)) != 64
                or float(row.get("lr", 0.0)) != 0.0001
                for row in selected
            ):
                raise MatureTargetRefreshReportError(
                    f"exact update schedule differs: seed {seed} {condition}"
                )
            fields = (
                "post_fork_consumed_transition_count",
                "batch_steps",
                "lr",
                "behaviour_temperature_min",
                "behaviour_temperature_mean",
                "behaviour_temperature_max",
            )
            for row in selected:
                if any(
                    isinstance(row.get(field), bool)
                    or not isinstance(row.get(field), (int, float))
                    or not math.isfinite(float(row[field]))
                    for field in fields[2:]
                ):
                    raise MatureTargetRefreshReportError(
                        "training schedule contains a non-finite value"
                    )
            projections[condition] = [
                {field: row[field] for field in fields} for row in selected
            ]
            train_path = segment / "train_log.jsonl"
            train_rows = _strict_jsonl(train_path)
            sanmill_rows = [
                row for row in train_rows if row.get("game_type") == "vs_sanmill"
            ]
            if not sanmill_rows or any(
                row.get("opponent_node_budget") != 1000 for row in sanmill_rows
            ):
                raise MatureTargetRefreshReportError(
                    f"fixed Sanmill work differs: seed {seed} {condition}"
                )
            requested_sanmill_nodes += sum(
                int(row.get("opponent_search_calls", 0))
                * int(row["opponent_node_budget"])
                for row in sanmill_rows
            )
            plan = load_managed_plan(
                ROOT / str(arms[(seed, condition)]["control_dir"]) / "plan.json"
            )
            checkpoint_path = segment / "latest.pt"
            checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
            policy_health = _validate_policy_health_report(
                plan,
                segment_index=1,
                report_path=segment / "policy-health.json",
                checkpoint=checkpoint_path,
                specialist_db=(
                    ROOT / str(arms[(seed, condition)]["specialist_db"])
                ).resolve(strict=True),
                completed_games=int(checkpoint.payload.trainer_state["game_count"]),
                runtime_commit=plan.git_commit,
            )
            if policy_health["passed"] is not True:
                raise MatureTargetRefreshReportError(
                    f"policy-health gate failed: seed {seed} {condition}"
                )
            seed_report[condition] = {
                "train_log": {
                    "path": _relative(train_path),
                    "sha256": _sha256_file(train_path),
                    "rows": len(train_rows),
                },
                "update_log": {
                    "path": _relative(segment / "update_log.jsonl"),
                    "sha256": _sha256_file(segment / "update_log.jsonl"),
                    "rows": len(selected),
                },
                "policy_health": {
                    "path": _relative(segment / "policy-health.json"),
                    "sha256": _sha256_file(segment / "policy-health.json"),
                    "evidence_id": policy_health["evidence_id"],
                    "passed": True,
                    "metrics": policy_health["metrics"],
                    "thresholds": policy_health["thresholds"],
                },
                "summary": _training_summary(train_rows),
            }
        if projections["refresh-mature"] != projections["stale-control"]:
            raise MatureTargetRefreshReportError(
                f"paired temperature or optimizer exposure differs: seed {seed}"
            )
        seed_report["paired_schedule"] = {
            "exact_update_batches": len(expected_counts),
            "post_mature_fork_consumed_transitions": 8192,
            "temperature_lr_exposure_byte_equal": True,
            "temperature_lr_exposure_identity": canonical_sha256(
                projections["refresh-mature"]
            ),
            "sanmill_node_budget": 1000,
        }
        report[str(seed)] = seed_report
    maximum = int(contract["resources"]["maximum_requested_sanmill_node_ceilings"])
    if requested_sanmill_nodes > maximum:
        raise MatureTargetRefreshReportError(
            "aggregate requested Sanmill node ceiling exceeded"
        )
    return {
        "by_seed": report,
        "requested_sanmill_nodes": requested_sanmill_nodes,
        "maximum_requested_sanmill_node_ceilings": maximum,
        "requested_sanmill_node_limit_passed": True,
    }


def _run_direct_game(
    *,
    scheduled: Mapping[str, Any],
    record: Mapping[str, Any],
    models: Mapping[str, Any],
    advisor: LookaheadAdvisor,
    installation: Any,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    generators = _build_policy_generators(scheduled)
    colour_condition = {
        str(scheduled["refresh_mature_colour"]): "refresh-mature",
        str(scheduled["stale_control_colour"]): "stale-control",
    }
    measurement = contract["measurement_contract"]["direct_crossplay"]
    max_plies = int(measurement["max_post_start_logical_plies"])
    temperature = float(measurement["policy_sampling_temperature"])
    moves: list[dict[str, str | None]] = []
    with SanmillTrainingGame(installation, seed=int(scheduled["referee_seed"])) as game:
        board = replay_record_into_sanmill_game(record, game)
        start_state = game.state
        for _ in range(max_plies):
            if game.state.terminal:
                break
            player = board.turn
            condition = colour_condition[player]
            move = _sample_policy_move(
                board=board,
                model=models[condition],
                advisor=advisor,
                generator=generators[player],
                temperature=temperature,
            )
            game.apply_nmm_move(board, move)
            board = board.apply_move(move)
            moves.append(_normalised_move(move))
        end_state = game.state
    post_start = int(end_state.logical_ply_count - start_state.logical_ply_count)
    if post_start != len(moves):
        raise MatureTargetRefreshReportError(
            "strict referee and ledger ply counts differ"
        )
    if end_state.terminal:
        winner = end_state.winner
        reason = end_state.outcome_reason_code
    else:
        if post_start != max_plies:
            raise MatureTargetRefreshReportError(
                "direct game stopped without a terminal reason"
            )
        winner = None
        reason = "max-ply-truncation"
    refresh_name = "white" if scheduled["refresh_mature_colour"] == "W" else "black"
    score = 0.5 if winner is None else 1.0 if winner == refresh_name else 0.0
    return {
        "schema_version": LEDGER_SCHEMA,
        "plan_identity": contract["plan_identity"],
        **dict(scheduled),
        "phase": str(record["phase"]),
        "refresh_mature_score": score,
        "outcome_class": {0.0: "loss", 0.5: "draw", 1.0: "win"}[score],
        "winner": winner,
        "termination_reason": reason,
        "post_start_logical_plies": post_start,
        "start_history_sha256": start_state.history_sha256,
        "end_history_sha256": end_state.history_sha256,
        "moves": moves,
    }


def _write_outputs(
    *,
    ledger: Path,
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    report: Mapping[str, Any],
) -> None:
    if ledger.exists() or output.exists():
        raise MatureTargetRefreshReportError("result target already exists")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("xb") as handle:
        handle.write(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
        handle.flush()
        os.fsync(handle.fileno())
    with output.open("xb") as handle:
        handle.write(canonical_json_bytes(report) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--policy-corpus", type=Path, default=DEFAULT_POLICY_CORPUS)
    parser.add_argument("--replay-corpus", type=Path, default=DEFAULT_REPLAY_CORPUS)
    parser.add_argument("--replay-audit", type=Path, default=DEFAULT_REPLAY_AUDIT)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--malom-manifest", type=Path, default=DEFAULT_MALOM_MANIFEST)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--allow-published-analysis-descendant", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = {
        name: value.resolve()
        for name, value in {
            "contract": args.contract,
            "readiness": args.readiness,
            "policy_corpus": args.policy_corpus,
            "replay_corpus": args.replay_corpus,
            "replay_audit": args.replay_audit,
            "paths_config": args.paths_config,
            "malom_manifest": args.malom_manifest,
            "ledger": args.ledger,
            "output": args.output,
        }.items()
    }
    for label in (
        "contract",
        "readiness",
        "policy_corpus",
        "replay_corpus",
        "replay_audit",
        "paths_config",
        "malom_manifest",
    ):
        if not paths[label].is_file():
            raise MatureTargetRefreshReportError(f"input is absent: {label}")
    if paths["ledger"].exists() or paths["output"].exists():
        raise MatureTargetRefreshReportError("result targets already exist")
    contract = load_contract(paths["contract"])
    readiness = _strict_json(paths["readiness"])
    readiness_identity = _validate_readiness(readiness, contract=contract)
    source_git = (
        _inspect_analysis_source(str(readiness["source"]["head"]))
        if args.allow_published_analysis_descendant
        else _git_source(str(readiness["source"]["head"]))
    )

    policy_contract = contract["measurement_contract"]["policy_distribution"]
    direct_contract = contract["measurement_contract"]["direct_crossplay"]
    policy_corpus = _read_hash_bound_json_object(
        paths["policy_corpus"],
        expected_sha256=str(policy_contract["fixed_corpus_sha256"]),
        label="policy corpus",
    )
    validate_phase_corpus(policy_corpus)
    replay_corpus = _read_hash_bound_json_object(
        paths["replay_corpus"],
        expected_sha256=str(direct_contract["replay_corpus_sha256"]),
        label="replay corpus",
    )
    validate_phase_replay_development_corpus(replay_corpus)
    if replay_corpus["corpus_identity"] != direct_contract["replay_corpus_identity"]:
        raise MatureTargetRefreshReportError("replay corpus identity differs")
    replay_audit = _read_hash_bound_json_object(
        paths["replay_audit"],
        expected_sha256=str(direct_contract["replay_audit_sha256"]),
        label="replay audit",
    )
    validate_phase_replay_sanmill_audit(replay_audit, corpus=replay_corpus)
    if replay_audit["audit_identity"] != direct_contract["replay_audit_identity"]:
        raise MatureTargetRefreshReportError("replay audit identity differs")

    settings = _read_json_object(paths["paths_config"])
    human_value = Path(str(settings["human_db_path"]))
    human_path = human_value if human_value.is_absolute() else ROOT / human_value
    malom_value = Path(str(settings["malom_db_path"]))
    malom_path = malom_value if malom_value.is_absolute() else ROOT / malom_value
    human_path = human_path.resolve(strict=True)
    malom_path = malom_path.resolve(strict=True)
    human_report = _probe_human_db(human_path)
    data = contract["rules_and_data"]
    if (
        human_report.get("error")
        or human_report.get("identity") != data["human_db_identity"]
        or human_report.get("malom_columns_policy") != "masked_historical_labels"
    ):
        raise MatureTargetRefreshReportError("HumanDB identity or policy differs")
    malom_manifest = load_dataset_manifest(paths["malom_manifest"])
    if malom_manifest.manifest_sha256 != data["malom_manifest_identity"]:
        raise MatureTargetRefreshReportError("Malom manifest identity differs")
    std_anchor = next(
        (
            component
            for component in malom_manifest.components
            if component.relative_path == "std.secval"
        ),
        None,
    )
    if (
        std_anchor is None
        or _sha256_file(malom_path / "std.secval") != std_anchor.sha256
    ):
        raise MatureTargetRefreshReportError("Malom std.secval identity differs")

    installation = load_local_installation(paths["paths_config"])
    training_audit = _audit_training(arms=_arm_map(contract), contract=contract)
    before = _read_only_observations(human_db_path=human_path, malom_path=malom_path)
    human_db = _open_immutable_human_db(human_path)
    malom = ExternalSolvedDB(str(malom_path), strict=True)
    if not malom.is_available():
        human_db.close()
        raise MatureTargetRefreshReportError("Malom dependency is unavailable")
    arms = _arm_map(contract)
    sources = _source_map(contract)
    readiness_seeds = _readiness_seed_map(readiness)
    replay_records = {
        int(record["record_index"]): record for record in replay_corpus["records"]
    }
    schedule = build_direct_crossplay_schedule(contract)
    policy_summaries: dict[str, dict[str, Mapping[str, Any]]] = {}
    seed_reports: dict[str, Any] = {}
    direct_rows: list[dict[str, Any]] = []
    try:
        for seed in (67, 68, 69):
            common, common_record = _load_common_fork(
                source=sources[seed],
                arm=arms[(seed, "refresh-mature")],
                readiness_seed=readiness_seeds[seed],
            )
            anchor_model = _load_policy(common, device=torch.device("cpu"))
            try:
                states, feature_record = _build_feature_corpus(
                    corpus=policy_corpus,
                    anchor_model=anchor_model,
                    human_db=human_db,
                    malom=malom,
                    device=torch.device("cpu"),
                )
            except CommonAnchorAnalysisError as exc:
                raise MatureTargetRefreshReportError(str(exc)) from exc
            route = feature_record["feature_route"]
            route.pop("common_game_50_anchor", None)
            route["same_seed_mature_common_fork"] = True
            advisor = LookaheadAdvisor(
                sentinel=None,
                evaluate_fn=trainer._simple_evaluate,
                value_net=None,
                gap_net=None,
                human_db=human_db,
                use_sentinel=True,
                endgame_db=malom,
                ply_depth=12,
                frozen_model=anchor_model,
                frozen_device=torch.device("cpu"),
                sim_ply_depth=5,
                strict=True,
            )
            policy_summaries[str(seed)] = {}
            boundaries = []
            final_models: dict[str, Any] | None = None
            for boundary in POLICY_BOUNDARIES:
                models, checkpoints = _load_candidate_pair(
                    arms=arms,
                    readiness_seed=readiness_seeds[seed],
                    seed=seed,
                    boundary=boundary,
                    common_record=common_record,
                    device=torch.device("cpu"),
                )
                try:
                    state_rows, summary = _compare_checkpoint_pair(
                        states=states, models=models, device=torch.device("cpu")
                    )
                except CommonAnchorAnalysisError as exc:
                    raise MatureTargetRefreshReportError(str(exc)) from exc
                policy_summaries[str(seed)][str(boundary)] = summary
                boundaries.append(
                    {
                        "post_mature_fork_consumed_transitions": boundary,
                        "checkpoints": checkpoints,
                        "summary": summary,
                        "states": state_rows,
                    }
                )
                if boundary == 8192:
                    final_models = {
                        "refresh-mature": models["refresh"],
                        "stale-control": models["no-refresh"],
                    }
            if final_models is None:
                raise MatureTargetRefreshReportError("final models are absent")
            scheduled_seed = [row for row in schedule if row["seed"] == seed]
            for scheduled in scheduled_seed:
                direct_rows.append(
                    _run_direct_game(
                        scheduled=scheduled,
                        record=replay_records[int(scheduled["record_index"])],
                        models=final_models,
                        advisor=advisor,
                        installation=installation,
                        contract=contract,
                    )
                )
                if len(direct_rows) % 24 == 0:
                    print(
                        "[mature-refresh] completed direct games "
                        f"{len(direct_rows)}/{len(schedule)}",
                        flush=True,
                    )
            seed_reports[str(seed)] = {
                "common_fork": common_record,
                "feature_corpus": feature_record,
                "boundaries": boundaries,
            }
    finally:
        human_db.close()
        malom.close()

    after = _read_only_observations(human_db_path=human_path, malom_path=malom_path)
    if before != after:
        raise MatureTargetRefreshReportError(
            "read-only data observations changed during analysis"
        )
    policy_decision = classify_mature_policy_divergence(
        policy_summaries, thresholds=DEFAULT_DIVERGENCE_THRESHOLDS
    )
    direct_summary = summarize_direct_crossplay(contract, direct_rows)
    combined = decide_mature_result(
        policy_decision=policy_decision,
        direct_crossplay=direct_summary,
    )
    ledger_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in direct_rows)
    ledger_sha256 = hashlib.sha256(ledger_bytes).hexdigest()
    report_core = {
        "schema_version": RESULT_SCHEMA,
        "scope": {
            "candidate_models_loaded": True,
            "training_games": 0,
            "optimizer_updates": 0,
            "database_writes": 0,
            "checkpoint_writes": 0,
            "no_update_development_games": len(direct_rows),
            "held_out_strength_claim": False,
            "promotion_publication_or_long_run_authority": False,
        },
        "identities": {
            "source": source_git,
            "contract": {
                "path": _relative(paths["contract"]),
                "sha256": _sha256_file(paths["contract"]),
                "plan_identity": contract["plan_identity"],
            },
            "readiness": {
                "path": _relative(paths["readiness"]),
                "sha256": _sha256_file(paths["readiness"]),
                "readiness_identity": readiness_identity,
            },
            "policy_corpus": {
                "path": _relative(paths["policy_corpus"]),
                "sha256": _sha256_file(paths["policy_corpus"]),
                "corpus_identity": policy_corpus["corpus_identity"],
            },
            "replay_corpus": {
                "path": _relative(paths["replay_corpus"]),
                "sha256": _sha256_file(paths["replay_corpus"]),
                "corpus_identity": replay_corpus["corpus_identity"],
            },
            "replay_audit": {
                "path": _relative(paths["replay_audit"]),
                "sha256": _sha256_file(paths["replay_audit"]),
                "audit_identity": replay_audit["audit_identity"],
            },
            "direct_crossplay_ledger": {
                "path": _relative(paths["ledger"]),
                "sha256": ledger_sha256,
                "rows": len(direct_rows),
            },
            "human_db": {
                "identity": human_report["identity"],
                "historical_malom_labels": "masked",
            },
            "malom_manifest_identity": malom_manifest.manifest_sha256,
        },
        "hyperparameters": contract["common_training_contract"],
        "measurement_contract": contract["measurement_contract"],
        "training_evidence": training_audit,
        "read_only_observations": {"before": before, "after": after},
        "policy_distribution": {
            "by_seed": seed_reports,
            "decision": policy_decision,
        },
        "direct_crossplay": direct_summary,
        "decision": combined,
        "interpretation": {
            "observed_facts": (
                "exact update exposure, fixed learning rate and node work, "
                "complete training logs, full-action policy distributions, and "
                "paired colour-swapped phase outcomes"
            ),
            "hypothesis": contract["hypothesis"],
            "supporting_evidence": (
                "same-seed branches share one mature fork and common random "
                "streams while differing only in the one target refresh"
            ),
            "counterevidence": (
                "seed disagreement, phase or colour harm, truncation, policy "
                "non-persistence, and training/evaluation disagreement"
            ),
            "next_validation_experiment": (
                "a supported setting still requires a separately frozen "
                "retained-run plan and held-out evaluation"
            ),
        },
        "claim_boundary": contract["claim_boundary"],
    }
    report = {**report_core, "result_identity": canonical_sha256(report_core)}
    _write_outputs(
        ledger=paths["ledger"], rows=direct_rows, output=paths["output"], report=report
    )
    if _sha256_file(paths["ledger"]) != ledger_sha256:
        raise MatureTargetRefreshReportError("ledger hash differs after publication")
    print(f"ledger={_relative(paths['ledger'])}")
    print(f"ledger_sha256={ledger_sha256}")
    print(f"report={_relative(paths['output'])}")
    print(f"report_sha256={_sha256_file(paths['output'])}")
    print(f"result_identity={report['result_identity']}")
    print(f"classification={combined['classification']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MatureTargetRefreshReportError as exc:
        print(f"fatal_stop: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
