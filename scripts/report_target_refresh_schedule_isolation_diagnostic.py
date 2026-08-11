"""Run and publish the schedule-isolated target-refresh development result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
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
from learned_ai.evaluation.target_refresh_equal_transition_result import (  # noqa: E402
    EXPECTED_BOUNDARIES,
    classify_transition_policy_divergence,
)
from learned_ai.evaluation.target_refresh_schedule_isolation_result import (  # noqa: E402
    MAX_POST_START_LOGICAL_PLIES,
    OUTCOME_BOUNDARIES,
    PRIMARY_TEMPERATURE,
    RESULT_SCHEMA,
    ScheduleIsolationResultError,
    build_outcome_measurement_schedule,
    decide_schedule_isolation_result,
    validate_and_summarize_outcome_rows,
)
from learned_ai.models.lookahead_advisor import LookaheadAdvisor  # noqa: E402
from learned_ai.sentinel.db_teacher import ExternalSolvedDB  # noqa: E402
from learned_ai.training.generalist_preflight import _probe_human_db  # noqa: E402
from learned_ai.training.run_contract import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from learned_ai.training.sanmill_referee import SanmillTrainingGame  # noqa: E402
from learned_ai.validation.sanmill_node_calibration import (  # noqa: E402
    load_local_installation,
)
from learned_ai.validation.target_refresh_equal_transition_diagnostic import (  # noqa: E402
    SCHEDULE_ISOLATION_CONTRACT_SCHEMA,
    load_equal_transition_contract,
)
from scripts import train_s_gen_v2 as trainer  # noqa: E402
from scripts.analyze_common_anchor_policy_distribution import (  # noqa: E402
    CommonAnchorAnalysisError,
    _build_feature_corpus,
    _compare_checkpoint_pair,
    _load_policy,
    _open_immutable_human_db,
    _read_only_observations,
)  # noqa: E402
from scripts.report_target_refresh_equal_transition_diagnostic import (  # noqa: E402
    EqualTransitionReportError,
    _arm_by_cell,
    _git_identity,
    _load_candidate_pair,
    _load_fork,
    _prefix_by_seed,
    _relative,
    _sha256_file,
    _strict_json,
    _validate_readiness,
)  # noqa: E402


DEFAULT_CONTRACT = ROOT / (
    "docs/experiments/"
    "sanmill-target-refresh-schedule-isolation-diagnostic-v2.json"
)
DEFAULT_READINESS = ROOT / (
    "out/target-refresh-schedule-isolation-diagnostic-v2/readiness.json"
)
DEFAULT_POLICY_CORPUS = (
    ROOT / "docs/experiments/dev-v4-phase-covered-corpus-v1.json"
)
DEFAULT_REPLAY_CORPUS = (
    ROOT / "docs/experiments/dev-v4-phase-replay-development-corpus-v1.json"
)
DEFAULT_REPLAY_AUDIT = ROOT / (
    "docs/evidence/"
    "phase-replay-development-corpus-sanmill-audit-2026-08-11.json"
)
DEFAULT_PATHS_CONFIG = ROOT / "data/training_paths.local.json"
DEFAULT_MALOM_MANIFEST = ROOT / "data/manifests/malom-sector-corrected-v1.json"
DEFAULT_LEDGER = ROOT / (
    "out/target-refresh-schedule-isolation-diagnostic-v2/"
    "development-outcome-ledger.jsonl"
)
DEFAULT_OUTPUT = ROOT / (
    "out/target-refresh-schedule-isolation-diagnostic-v2/result.json"
)
EXPECTED_POLICY_CORPUS_SHA256 = (
    "cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e"
)
_POST_TRAINING_ANALYSIS_PATHS = frozenset(
    {
        "docs/evidence/target-refresh-schedule-isolation-"
        "diagnostic-v2-attempt-001-failure-2026-08-11.md",
        "docs/experiments/sanmill-target-refresh-schedule-isolation-"
        "analysis-recovery-v1.json",
        "docs/experiments/sanmill-target-refresh-schedule-isolation-"
        "analysis-recovery-v1.md",
        "scripts/report_target_refresh_schedule_isolation_diagnostic.py",
        "scripts/run_target_refresh_schedule_isolation_analysis_recovery.py",
        "tests/test_target_refresh_schedule_isolation_analysis_recovery.py",
        "tests/test_target_refresh_schedule_isolation_contract.py",
        "tests/test_target_refresh_schedule_isolation_report.py",
    }
)


class ScheduleIsolationReportError(RuntimeError):
    """Raised when immutable successor evidence cannot be established."""


def _git_output(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ScheduleIsolationReportError("Git evidence audit failed") from exc


def _inspect_analysis_source(expected_training_commit: str) -> dict[str, Any]:
    """Allow a published descendant only when every change is analysis-only."""
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
            raise ScheduleIsolationReportError(
                "post-training source changes are not analysis-only"
            )
    return {
        **source,
        "post_training_analysis_paths": changed_paths,
    }


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    framing: str | None = None
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.endswith("\r\n"):
                    observed_framing = "\r\n"
                    payload = line[:-2]
                elif line.endswith("\n"):
                    observed_framing = "\n"
                    payload = line[:-1]
                else:
                    raise ScheduleIsolationReportError(
                        f"JSONL framing differs: {path}:{line_number}"
                    )
                if framing is None:
                    framing = observed_framing
                if (
                    observed_framing != framing
                    or "\r" in payload
                    or "\n" in payload
                ):
                    raise ScheduleIsolationReportError(
                        f"JSONL framing differs: {path}:{line_number}"
                    )

                def reject_duplicate_keys(
                    pairs: list[tuple[str, Any]],
                ) -> dict[str, Any]:
                    value: dict[str, Any] = {}
                    for key, item in pairs:
                        if key in value:
                            raise ScheduleIsolationReportError(
                                f"duplicate JSON key: {path}:{line_number}:{key}"
                            )
                        value[key] = item
                    return value

                value = json.loads(
                    payload,
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ScheduleIsolationReportError(
                            f"non-finite JSON value: {path}:{line_number}:{token}"
                        )
                    ),
                )
                if not isinstance(value, dict):
                    raise ScheduleIsolationReportError(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScheduleIsolationReportError(f"cannot read JSONL: {path}") from exc
    return rows


def _segment(control_dir: str) -> Path:
    path = ROOT / control_dir / "segments" / "segment-0001"
    if not path.is_dir():
        raise ScheduleIsolationReportError(f"completed segment is absent: {path}")
    return path


def _audit_paired_training_schedules(
    *,
    arms: Mapping[tuple[int, str], Mapping[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Prove batch temperature exposure and Sanmill work were paired."""
    reports: dict[str, Any] = {}
    expected_counts = list(range(64, EXPECTED_BOUNDARIES[-1] + 1, 64))
    for seed in seeds:
        condition_rows: dict[str, list[dict[str, Any]]] = {}
        condition_train_rows: dict[str, list[dict[str, Any]]] = {}
        for condition in ("refresh-once", "no-refresh"):
            segment = _segment(str(arms[(seed, condition)]["control_dir"]))
            update_rows = _strict_jsonl(segment / "update_log.jsonl")
            selected = [
                row
                for row in update_rows
                if row.get("post_fork_consumed_transition_count") is not None
                and int(row["post_fork_consumed_transition_count"])
                <= EXPECTED_BOUNDARIES[-1]
            ]
            counts = [
                int(row["post_fork_consumed_transition_count"])
                for row in selected
            ]
            if counts != expected_counts or any(
                int(row.get("batch_steps", 0)) != 64 for row in selected
            ):
                raise ScheduleIsolationReportError(
                    f"exact update schedule differs: seed {seed} {condition}"
                )
            for row in selected:
                for field in (
                    "behaviour_temperature_min",
                    "behaviour_temperature_mean",
                    "behaviour_temperature_max",
                ):
                    value = row.get(field)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        raise ScheduleIsolationReportError(
                            f"temperature evidence is invalid: {field}"
                        )
            condition_rows[condition] = selected
            train_rows = _strict_jsonl(segment / "train_log.jsonl")
            sanmill_rows = [
                row for row in train_rows if row.get("game_type") == "vs_sanmill"
            ]
            if not sanmill_rows or any(
                row.get("opponent_node_budget") != 1_000 for row in sanmill_rows
            ):
                raise ScheduleIsolationReportError(
                    f"fixed Sanmill work differs: seed {seed} {condition}"
                )
            condition_train_rows[condition] = sanmill_rows

        fields = (
            "post_fork_consumed_transition_count",
            "batch_steps",
            "behaviour_temperature_min",
            "behaviour_temperature_mean",
            "behaviour_temperature_max",
        )
        projections = {
            condition: [
                {field: row[field] for field in fields}
                for row in condition_rows[condition]
            ]
            for condition in ("refresh-once", "no-refresh")
        }
        if projections["refresh-once"] != projections["no-refresh"]:
            raise ScheduleIsolationReportError(
                f"paired behavior-temperature exposure differs: seed {seed}"
            )
        reports[str(seed)] = {
            "exact_update_batches": len(expected_counts),
            "post_fork_consumed_transitions": EXPECTED_BOUNDARIES[-1],
            "temperature_exposure_byte_equal": True,
            "temperature_exposure_identity": canonical_sha256(
                projections["refresh-once"]
            ),
            "sanmill_node_budget": 1_000,
            "sanmill_training_games": {
                condition: len(condition_train_rows[condition])
                for condition in ("refresh-once", "no-refresh")
            },
        }
    return reports


def _outcome_class(training_outcome: float) -> tuple[str, float]:
    if training_outcome == trainer.WIN_REWARD:
        return "win", 1.0
    if training_outcome == trainer.LOSS_REWARD:
        return "loss", 0.0
    if training_outcome in (trainer.DRAW_SHORT, trainer.DRAW_LONG):
        return "draw", 0.5
    raise ScheduleIsolationReportError("rollout outcome is not a known class")


def _complete_outcome_row(
    scheduled: Mapping[str, Any],
    *,
    result: Any,
    start_state: Any,
    end_state: Any,
    candidate_checkpoint_id: str,
    anchor_checkpoint_id: str,
) -> dict[str, Any]:
    outcome_class, score = _outcome_class(float(result.outcome))
    post_start = int(end_state.logical_ply_count) - int(
        start_state.logical_ply_count
    )
    if post_start != int(result.ply):
        raise ScheduleIsolationReportError(
            "Sanmill and rollout logical-ply counts differ"
        )
    return {
        **scheduled,
        "candidate_checkpoint_id": candidate_checkpoint_id,
        "anchor_checkpoint_id": anchor_checkpoint_id,
        "start_history_sha256": start_state.history_sha256,
        "end_history_sha256": end_state.history_sha256,
        "start_logical_ply_count": int(start_state.logical_ply_count),
        "end_logical_ply_count": int(end_state.logical_ply_count),
        "training_reward_outcome": float(result.outcome),
        "outcome_class": outcome_class,
        "score": score,
        "post_start_logical_plies": post_start,
        "termination_reason": str(result.termination_reason),
        "policy_observations": trainer._development_measurement_metrics(result),
    }


def _run_outcome_game(
    *,
    scheduled: Mapping[str, Any],
    record: Mapping[str, Any],
    installation: Any,
    candidate_model: Any,
    anchor_opponent: Any,
    lookahead_advisor: Any,
    human_db: Any,
    malom: Any,
    device: torch.device,
    candidate_checkpoint_id: str,
    anchor_checkpoint_id: str,
) -> dict[str, Any]:
    candidate_color = str(scheduled["candidate_color"])
    opponent_color = "B" if candidate_color == "W" else "W"
    torch_seed = int(scheduled["torch_seed"])
    with SanmillTrainingGame(installation, seed=torch_seed) as game:
        board = replay_record_into_sanmill_game(record, game)
        start_state = game.state
        result = trainer._rollout(
            model=candidate_model,
            device=device,
            start_board=board,
            learner_color=candidate_color,
            opponent=anchor_opponent,
            opp_color=opponent_color,
            sentinel=None,
            value_net=None,
            temperature=PRIMARY_TEMPERATURE,
            max_ply=MAX_POST_START_LOGICAL_PLIES,
            record_branches=False,
            branch_every=0,
            retry_ply=0,
            lookahead_advisor=lookahead_advisor,
            game_difficulty=1,
            human_db=human_db,
            specialist_db=None,
            malom_db=malom,
            deep_game=False,
            torch_generator=trainer._game_torch_generator(torch_seed),
            sanmill_game=game,
            persist_rollout_evidence=False,
            mill_bonus_mode="malom-preserving-only",
            malom_policy_aux_coef=0.0,
            malom_policy_aux_mode="fixed",
        )
        end_state = game.state
    if result.specialist_read_stats:
        raise ScheduleIsolationReportError(
            "development measurement unexpectedly read SpecialistDB"
        )
    return _complete_outcome_row(
        scheduled,
        result=result,
        start_state=start_state,
        end_state=end_state,
        candidate_checkpoint_id=candidate_checkpoint_id,
        anchor_checkpoint_id=anchor_checkpoint_id,
    )


def _write_outputs(
    *,
    ledger_path: Path,
    ledger_rows: Sequence[Mapping[str, Any]],
    result_path: Path,
    report: Mapping[str, Any],
) -> None:
    if ledger_path.exists() or result_path.exists():
        raise ScheduleIsolationReportError("result or outcome ledger already exists")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_tmp = ledger_path.with_name(ledger_path.name + ".tmp")
    result_tmp = result_path.with_name(result_path.name + ".tmp")
    if ledger_tmp.exists() or result_tmp.exists():
        raise ScheduleIsolationReportError("temporary result output already exists")
    ledger_tmp.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in ledger_rows)
    )
    result_tmp.write_bytes(canonical_json_bytes(report))
    ledger_tmp.replace(ledger_path)
    result_tmp.replace(result_path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--policy-corpus", type=Path, default=DEFAULT_POLICY_CORPUS)
    parser.add_argument("--replay-corpus", type=Path, default=DEFAULT_REPLAY_CORPUS)
    parser.add_argument("--replay-audit", type=Path, default=DEFAULT_REPLAY_AUDIT)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument(
        "--malom-manifest", type=Path, default=DEFAULT_MALOM_MANIFEST
    )
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
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
            raise ScheduleIsolationReportError(f"input is absent: {label}")
    if paths["ledger"].exists() or paths["output"].exists():
        raise ScheduleIsolationReportError("result targets already exist")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ScheduleIsolationReportError("requested CUDA device is unavailable")
    device = torch.device(args.device)

    try:
        contract = load_equal_transition_contract(paths["contract"])
    except Exception as exc:
        raise ScheduleIsolationReportError(str(exc)) from exc
    if contract["schema_version"] != SCHEDULE_ISOLATION_CONTRACT_SCHEMA:
        raise ScheduleIsolationReportError("result requires the v2 contract")
    seeds = tuple(int(seed) for seed in contract["pairing"]["seeds"])
    readiness = _validate_readiness(paths["readiness"], contract=contract)
    source = _inspect_analysis_source(str(readiness["source"]["head"]))

    if _sha256_file(paths["policy_corpus"]) != EXPECTED_POLICY_CORPUS_SHA256:
        raise ScheduleIsolationReportError("fixed policy corpus identity differs")
    policy_corpus = _strict_json(paths["policy_corpus"])
    validate_phase_corpus(policy_corpus)
    replay_corpus = _strict_json(paths["replay_corpus"])
    validate_phase_replay_development_corpus(replay_corpus)
    replay_audit = _strict_json(paths["replay_audit"])
    validate_phase_replay_sanmill_audit(replay_audit, corpus=replay_corpus)
    outcome_contract = contract["measurement_contract"]["outcome_measurement"]
    if (
        _sha256_file(paths["replay_corpus"])
        != outcome_contract["fixed_replay_corpus_sha256"]
        or replay_corpus["corpus_identity"]
        != outcome_contract["fixed_replay_corpus_identity"]
        or _sha256_file(paths["replay_audit"])
        != outcome_contract["strict_replay_audit_sha256"]
        or replay_audit["audit_identity"]
        != outcome_contract["strict_replay_audit_identity"]
    ):
        raise ScheduleIsolationReportError("replay evidence binding differs")

    settings = _strict_json(paths["paths_config"])
    human_value = Path(str(settings["human_db_path"]))
    human_path = (
        human_value.resolve()
        if human_value.is_absolute()
        else (ROOT / human_value).resolve()
    )
    malom_value = Path(str(settings["malom_db_path"]))
    malom_path = (
        malom_value.resolve()
        if malom_value.is_absolute()
        else (ROOT / malom_value).resolve()
    )
    human_report = _probe_human_db(human_path)
    if (
        human_report.get("error")
        or human_report.get("identity")
        != contract["data_contract"]["human_db_identity"]
        or human_report.get("malom_columns_policy")
        != "masked_historical_labels"
    ):
        raise ScheduleIsolationReportError("HumanDB identity or policy differs")
    malom_manifest = load_dataset_manifest(paths["malom_manifest"])
    if malom_manifest.manifest_sha256 != contract["data_contract"][
        "malom_manifest_identity"
    ]:
        raise ScheduleIsolationReportError("Malom manifest identity differs")
    std_anchor = next(
        (
            component
            for component in malom_manifest.components
            if component.relative_path == "std.secval"
        ),
        None,
    )
    if std_anchor is None or _sha256_file(malom_path / "std.secval") != (
        std_anchor.sha256
    ):
        raise ScheduleIsolationReportError("Malom std.secval identity differs")
    installation = load_local_installation(paths["paths_config"])
    schedule_audit = _audit_paired_training_schedules(
        arms=_arm_by_cell(contract),
        seeds=seeds,
    )

    before = _read_only_observations(
        human_db_path=human_path,
        malom_path=malom_path,
    )
    human_db = _open_immutable_human_db(human_path)
    malom = ExternalSolvedDB(str(malom_path), strict=True)
    if not malom.is_available():
        human_db.close()
        raise ScheduleIsolationReportError("Malom dependency is unavailable")
    prefixes = _prefix_by_seed(contract)
    arms = _arm_by_cell(contract)
    schedule = build_outcome_measurement_schedule(
        seeds=seeds,
        corpus=replay_corpus,
    )
    replay_records = {
        int(record["record_index"]): record for record in replay_corpus["records"]
    }
    policy_summaries: dict[str, dict[str, Mapping[str, Any]]] = {}
    seed_reports: dict[str, Any] = {}
    outcome_rows: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            fork_envelope, fork_record = _load_fork(prefixes[seed])
            anchor_model = _load_policy(fork_envelope, device=device)
            try:
                states, feature_record = _build_feature_corpus(
                    corpus=policy_corpus,
                    anchor_model=anchor_model,
                    human_db=human_db,
                    malom=malom,
                    device=device,
                )
            except CommonAnchorAnalysisError as exc:
                raise ScheduleIsolationReportError(str(exc)) from exc
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
                frozen_device=device,
                sim_ply_depth=5,
                strict=True,
            )
            anchor_opponent = trainer.FrozenModelOpponent(
                anchor_model,
                device,
                sentinel=None,
                value_net=None,
                lookahead_advisor=advisor,
                specialist_db=None,
            )
            policy_summaries[str(seed)] = {}
            boundaries: list[dict[str, Any]] = []
            for boundary in EXPECTED_BOUNDARIES:
                models, checkpoints = _load_candidate_pair(
                    arms,
                    seed=seed,
                    boundary=boundary,
                    fork_record=fork_record,
                    device=device,
                )
                try:
                    state_records, summary = _compare_checkpoint_pair(
                        states=states,
                        models=models,
                        device=device,
                    )
                except CommonAnchorAnalysisError as exc:
                    raise ScheduleIsolationReportError(str(exc)) from exc
                policy_summaries[str(seed)][str(boundary)] = summary
                boundary_record: dict[str, Any] = {
                    "post_fork_consumed_transitions": boundary,
                    "checkpoints": checkpoints,
                    "summary": summary,
                    "states": state_records,
                }
                if boundary in OUTCOME_BOUNDARIES:
                    scheduled_rows = [
                        row
                        for row in schedule
                        if row["seed"] == seed
                        and row["post_fork_consumed_transitions"] == boundary
                    ]
                    for scheduled in scheduled_rows:
                        condition = str(scheduled["condition"])
                        model_key = (
                            "refresh"
                            if condition == "refresh-once"
                            else "no-refresh"
                        )
                        outcome_rows.append(
                            _run_outcome_game(
                                scheduled=scheduled,
                                record=replay_records[
                                    int(scheduled["record_index"])
                                ],
                                installation=installation,
                                candidate_model=models[model_key],
                                anchor_opponent=anchor_opponent,
                                lookahead_advisor=advisor,
                                human_db=human_db,
                                malom=malom,
                                device=device,
                                candidate_checkpoint_id=checkpoints[condition][
                                    "checkpoint_id"
                                ],
                                anchor_checkpoint_id=fork_record["checkpoint_id"],
                            )
                        )
                        if len(outcome_rows) % 24 == 0:
                            print(
                                "[schedule-isolation] completed outcome games "
                                f"{len(outcome_rows)}/{len(schedule)}"
                            )
                    boundary_record["development_outcome_games"] = len(
                        scheduled_rows
                    )
                boundaries.append(boundary_record)
            seed_reports[str(seed)] = {
                "fork": fork_record,
                "feature_corpus": feature_record,
                "boundaries": boundaries,
            }
    finally:
        human_db.close()
        malom.close()

    after = _read_only_observations(
        human_db_path=human_path,
        malom_path=malom_path,
    )
    if before != after:
        raise ScheduleIsolationReportError(
            "read-only source observations changed during analysis"
        )
    try:
        outcome_summary = validate_and_summarize_outcome_rows(
            outcome_rows,
            seeds=seeds,
            corpus=replay_corpus,
        )
    except ScheduleIsolationResultError as exc:
        raise ScheduleIsolationReportError(str(exc)) from exc
    policy_decision = classify_transition_policy_divergence(
        policy_summaries,
        thresholds=DEFAULT_DIVERGENCE_THRESHOLDS,
        seeds=seeds,
    )
    combined_decision = decide_schedule_isolation_result(
        policy_decision=policy_decision,
        outcome_summary=outcome_summary,
        seeds=seeds,
    )
    ledger_bytes = b"".join(
        canonical_json_bytes(row) + b"\n" for row in outcome_rows
    )
    ledger_sha256 = hashlib.sha256(ledger_bytes).hexdigest()
    report_core = {
        "schema_version": RESULT_SCHEMA,
        "scope": {
            "candidate_models_loaded": True,
            "checkpoint_writes": 0,
            "database_writes": 0,
            "held_out_strength_claim": False,
            "no_update_development_games": len(outcome_rows),
            "optimizer_updates": 0,
            "promotion_publication_or_long_run_authority": False,
            "training_games": 0,
        },
        "identities": {
            "source": source,
            "contract": {
                "path": _relative(paths["contract"]),
                "sha256": _sha256_file(paths["contract"]),
                "plan_identity": contract["plan_identity"],
            },
            "readiness": {
                "path": _relative(paths["readiness"]),
                "sha256": _sha256_file(paths["readiness"]),
                "readiness_identity": readiness["readiness_identity"],
            },
            "result_implementation": {
                "frozen_contract_record": contract["analysis"][
                    "result_implementation"
                ],
                "executed_publisher": {
                    "path": _relative(Path(__file__)),
                    "sha256": _sha256_file(Path(__file__)),
                },
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
            "outcome_ledger": {
                "path": _relative(paths["ledger"]),
                "sha256": ledger_sha256,
                "rows": len(outcome_rows),
            },
            "human_db": {
                "lookup_key": "human_db_path",
                "identity": human_report["identity"],
                "historical_malom_labels": "masked",
            },
            "malom_manifest_identity": malom_manifest.manifest_sha256,
        },
        "hyperparameters": contract["common_training_contract"],
        "measurement_contract": contract["measurement_contract"],
        "training_schedule_audit": schedule_audit,
        "read_only_observations": {"before": before, "after": after},
        "policy_distribution": {
            "by_seed": seed_reports,
            "decision": policy_decision,
        },
        "development_outcomes": outcome_summary,
        "decision": combined_decision,
        "interpretation": {
            "observed_facts": (
                "exact optimizer transitions, per-batch behavior temperatures, "
                "fixed-node opponent work, full-action policy distributions, "
                "and paired phase/color/termination outcomes"
            ),
            "hypothesis": contract["hypothesis"],
            "supporting_evidence": (
                "same-seed branches share one fork, exact update exposure, one "
                "temperature sequence, one node budget, starts, colors and "
                "common random numbers"
            ),
            "counterevidence": (
                "seed disagreement, phase harm, Malom regression, truncation "
                "increase and policy/outcome disagreement remain explicit gates"
            ),
            "next_validation_experiment": (
                "only a supported result may select a setting for a separately "
                "frozen retained-run plan; this report cannot launch it"
            ),
        },
        "claim_boundary": contract["claim_boundary"],
    }
    report = {**report_core, "result_identity": canonical_sha256(report_core)}
    _write_outputs(
        ledger_path=paths["ledger"],
        ledger_rows=outcome_rows,
        result_path=paths["output"],
        report=report,
    )
    if _sha256_file(paths["ledger"]) != ledger_sha256:
        raise ScheduleIsolationReportError("published outcome ledger hash differs")
    print(f"ledger={_relative(paths['ledger'])}")
    print(f"ledger_sha256={ledger_sha256}")
    print(f"report={_relative(paths['output'])}")
    print(f"report_sha256={_sha256_file(paths['output'])}")
    print(f"result_identity={report['result_identity']}")
    print(f"classification={combined_decision['classification']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ScheduleIsolationReportError, EqualTransitionReportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
