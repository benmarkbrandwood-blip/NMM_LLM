#!/usr/bin/env python3
"""Preflight, authorize, or run the frozen schedule-isolation sequence once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.training.managed_generalist import (  # noqa: E402
    authorize_plan,
    load_managed_plan,
    run_authorized_plan,
)
from learned_ai.training.generalist_run_manifest import utc_now_text  # noqa: E402
from learned_ai.training.run_contract import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from learned_ai.validation.target_refresh_equal_transition_arms import (  # noqa: E402
    prepare_seed_arms,
)
from learned_ai.validation.target_refresh_equal_transition_diagnostic import (  # noqa: E402
    load_equal_transition_contract,
)
from learned_ai.validation.target_refresh_schedule_isolation_sequence import (  # noqa: E402
    DELEGATED_OPERATOR,
    DEVELOPMENT_ANALYSIS_DEVICE,
    ScheduleIsolationSequenceError,
    SequenceStep,
    build_sequence_authorization,
    build_sequence_steps,
    execute_sequence_steps,
    validate_readiness_identity,
    validate_sequence_authorization,
)
from scripts import report_target_refresh_schedule_isolation_diagnostic as reporter  # noqa: E402


FAMILY = "target-refresh-schedule-isolation-diagnostic-v2"
DEFAULT_CONTRACT = ROOT / (
    "docs/experiments/"
    "sanmill-target-refresh-schedule-isolation-diagnostic-v2.json"
)
DEFAULT_READINESS = ROOT / f"out/{FAMILY}/readiness.json"
DEFAULT_AUTHORIZATION = ROOT / f"out/{FAMILY}/sequence-authorization.json"
DEFAULT_MARKER = ROOT / f"out/{FAMILY}/sequence-launch.json"
DEFAULT_FAILURE = ROOT / f"out/{FAMILY}/sequence-failure.json"
DEFAULT_RESULT = ROOT / f"out/{FAMILY}/sequence-result.json"
DEFAULT_PATHS_CONFIG = ROOT / "data/training_paths.local.json"
DEFAULT_DEVELOPMENT_RESULT = ROOT / f"out/{FAMILY}/result.json"
DEFAULT_DEVELOPMENT_LEDGER = ROOT / (
    f"out/{FAMILY}/development-outcome-ledger.jsonl"
)
SEQUENCE_READINESS_SCHEMA = (
    "nmm.target-refresh-schedule-isolation-sequence-readiness.v2"
)
SEQUENCE_LAUNCH_SCHEMA = "nmm.target-refresh-schedule-isolation-launch.v2"
SEQUENCE_RESULT_SCHEMA = "nmm.target-refresh-schedule-isolation-sequence-result.v2"
SEQUENCE_FAILURE_SCHEMA = (
    "nmm.target-refresh-schedule-isolation-sequence-failure.v2"
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ScheduleIsolationSequenceError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ScheduleIsolationSequenceError(
                    f"non-finite JSON value in {path.name}: {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScheduleIsolationSequenceError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ScheduleIsolationSequenceError(f"JSON root is not an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside_root(path: str | Path, *, field: str) -> Path:
    candidate = Path(path)
    resolved = (
        candidate.resolve(strict=False)
        if candidate.is_absolute()
        else (ROOT / candidate).resolve(strict=False)
    )
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ScheduleIsolationSequenceError(
            f"{field} is outside the repository"
        ) from exc
    return resolved


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ScheduleIsolationSequenceError(
            "Git audit failed: " + " ".join(arguments)
        )
    return result.stdout.strip()


def _source_identity(expected_commit: str) -> dict[str, Any]:
    branch = _git_output("branch", "--show-current")
    head = _git_output("rev-parse", "HEAD")
    origin_dev = _git_output("rev-parse", "origin/dev")
    status = _git_output("status", "--porcelain=v1", "--untracked-files=no")
    if branch != "dev" or head != origin_dev or head != expected_commit or status:
        raise ScheduleIsolationSequenceError(
            "sequence requires the exact clean published readiness source"
        )
    return {
        "branch": branch,
        "head": head,
        "origin_dev": origin_dev,
        "tracked_clean": True,
        "published": True,
    }


def _publish_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    target = _inside_root(path, field="output")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ScheduleIsolationSequenceError(
            f"output already exists: {target}"
        ) from exc


def _prefix_records_by_seed(readiness: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    raw = readiness.get("prefixes")
    if not isinstance(raw, list) or len(raw) != 3:
        raise ScheduleIsolationSequenceError("prefix readiness cells differ")
    records = {int(item["seed"]): item for item in raw if isinstance(item, Mapping)}
    if len(records) != len(raw):
        raise ScheduleIsolationSequenceError("prefix readiness seeds differ")
    return records


def _sequence_readiness_result(
    technical_body: Mapping[str, Any],
    *,
    authorization_present: bool,
) -> dict[str, Any]:
    return {
        **dict(technical_body),
        "authorization_present": authorization_present,
        "launch_authorized": authorization_present,
        "sequence_readiness_identity": canonical_sha256(technical_body),
    }


def inspect_sequence_readiness(
    *,
    contract_path: Path,
    readiness_path: Path,
    authorization_path: Path = DEFAULT_AUTHORIZATION,
    marker_path: Path = DEFAULT_MARKER,
    failure_path: Path = DEFAULT_FAILURE,
    sequence_result_path: Path = DEFAULT_RESULT,
    allow_authorization: bool = False,
) -> dict[str, Any]:
    """Revalidate all currently available inputs without starting training."""
    contract_path = contract_path.resolve(strict=True)
    readiness_path = readiness_path.resolve(strict=True)
    contract = load_equal_transition_contract(contract_path)
    readiness = _strict_json(readiness_path)
    readiness_identity = validate_readiness_identity(readiness)
    contract_record = readiness.get("contract", {})
    if (
        contract_record.get("plan_identity") != contract["plan_identity"]
        or contract_record.get("sha256") != _sha256_file(contract_path)
    ):
        raise ScheduleIsolationSequenceError("readiness contract binding differs")
    source_record = readiness.get("source", {})
    source = _source_identity(str(source_record.get("head", "")))
    if source_record.get("origin_dev") != source["origin_dev"]:
        raise ScheduleIsolationSequenceError("readiness origin/dev differs")

    prefix_records = _prefix_records_by_seed(readiness)
    prefix_audit: list[dict[str, Any]] = []
    for prefix in sorted(contract["prefixes"], key=lambda item: item["launch_order"]):
        seed = int(prefix["seed"])
        record = prefix_records.get(seed)
        if record is None:
            raise ScheduleIsolationSequenceError(f"prefix readiness is absent: {seed}")
        plan_path = _inside_root(record["plan_path"], field="prefix plan")
        preflight_path = _inside_root(
            record["preflight"]["path"], field="prefix preflight"
        )
        db_path = _inside_root(
            record["specialist_db"]["path"], field="prefix SpecialistDB"
        )
        if not plan_path.is_file() or not preflight_path.is_file() or not db_path.is_file():
            raise ScheduleIsolationSequenceError(f"prefix input is absent: {seed}")
        plan = load_managed_plan(plan_path)
        expected_control = _inside_root(prefix["control_dir"], field="control dir")
        expected_db = _inside_root(prefix["specialist_db"], field="SpecialistDB")
        if (
            plan.plan_sha256 != record["plan_sha256"]
            or _sha256_file(preflight_path) != record["preflight"]["sha256"]
            or _sha256_file(db_path) != record["specialist_db"]["sha256"]
            or Path(plan.control_dir).resolve() != expected_control
            or db_path != expected_db
            or plan.git_commit != source["head"]
            or record.get("authorization_present") is not False
            or record.get("segment_output_present") is not False
            or (expected_control / "authorization.json").exists()
            or (expected_control / "segments").exists()
        ):
            raise ScheduleIsolationSequenceError(
                f"prefix plan or fresh-output identity differs: {seed}"
            )
        prefix_audit.append(
            {
                "seed": seed,
                "launch_order": int(prefix["launch_order"]),
                "plan_path": str(plan_path),
                "plan_sha256": plan.plan_sha256,
                "plan_file_sha256": _sha256_file(plan_path),
                "preflight_sha256": _sha256_file(preflight_path),
                "specialist_db_sha256": _sha256_file(db_path),
                "authorization_present": False,
                "segment_output_present": False,
            }
        )

    for arm in contract["arms"]:
        control = _inside_root(arm["control_dir"], field="arm control")
        database = _inside_root(arm["specialist_db"], field="arm SpecialistDB")
        if control.exists() or database.exists():
            raise ScheduleIsolationSequenceError(
                f"deferred arm target already exists: {arm['arm_id']}"
            )
    required_absent = [
        marker_path,
        failure_path,
        sequence_result_path,
        DEFAULT_DEVELOPMENT_RESULT,
        DEFAULT_DEVELOPMENT_LEDGER,
    ]
    if not allow_authorization:
        required_absent.append(authorization_path)
    existing = [str(path) for path in required_absent if path.exists()]
    if existing:
        raise ScheduleIsolationSequenceError(
            "one-shot sequence output already exists: " + ", ".join(existing)
        )
    technical_body = {
        "schema_version": SEQUENCE_READINESS_SCHEMA,
        "state": "ready_for_parent_product_authorization",
        "identity_scope": (
            "technical readiness excluding the subsequently recorded parent "
            "authorization presence"
        ),
        "source": source,
        "contract": {
            "path": str(contract_path),
            "sha256": _sha256_file(contract_path),
            "plan_identity": contract["plan_identity"],
        },
        "readiness": {
            "path": str(readiness_path),
            "sha256": _sha256_file(readiness_path),
            "readiness_identity": readiness_identity,
        },
        "prefixes": prefix_audit,
        "deferred_arms": 6,
        "ordered_steps": [
            step.to_dict() for step in build_sequence_steps(contract)
        ],
        "resource_envelope": contract["resources"],
        "development_analysis_device": DEVELOPMENT_ANALYSIS_DEVICE,
        "claim_boundary": contract["claim_boundary"],
    }
    return _sequence_readiness_result(
        technical_body,
        authorization_present=authorization_path.exists(),
    )


def record_sequence_authorization(
    *,
    contract_path: Path,
    readiness_path: Path,
    authorization_path: Path,
    expected_sequence_readiness_identity: str,
    decision_note: str,
) -> dict[str, Any]:
    """Persist the parent grant only after the owner supplies its exact decision."""
    sequence_readiness = inspect_sequence_readiness(
        contract_path=contract_path,
        readiness_path=readiness_path,
        authorization_path=authorization_path,
    )
    contract = load_equal_transition_contract(contract_path)
    readiness = _strict_json(readiness_path)
    validate_readiness_identity(readiness)
    observed = sequence_readiness["sequence_readiness_identity"]
    if observed != expected_sequence_readiness_identity:
        raise ScheduleIsolationSequenceError(
            "expected sequence readiness identity differs"
        )
    authorization = build_sequence_authorization(
        contract=contract,
        readiness=readiness,
        sequence_readiness_identity=observed,
        decision_note=decision_note,
        authorized_at_utc=utc_now_text(),
    )
    _publish_exclusive(authorization_path, authorization)
    return authorization


def _child_paths(entry: Mapping[str, Any]) -> tuple[Path, Path]:
    control = _inside_root(entry["control_dir"], field="child control")
    return control / "plan.json", control / "authorization.json"


def _authorize_and_run_child(
    entry: Mapping[str, Any],
    *,
    parent_authorization_identity: str,
) -> dict[str, Any]:
    plan_path, authorization_path = _child_paths(entry)
    if authorization_path.exists():
        raise ScheduleIsolationSequenceError(
            f"child authorization already exists: {entry.get('arm_id', entry['seed'])}"
        )
    plan = load_managed_plan(plan_path)
    authorize_plan(
        plan_path,
        authorization_path,
        authorized_by=DELEGATED_OPERATOR,
        decision_note=(
            "Delegated by frozen parent sequence authorization "
            f"{parent_authorization_identity}; one attempt, no retry or recovery"
        ),
    )
    status = run_authorized_plan(plan_path, authorization_path)
    if status.get("state") != "completed":
        raise ScheduleIsolationSequenceError(
            f"child plan did not complete: {plan.plan_id}"
        )
    ledger = Path(plan.control_dir) / "controller-events.jsonl"
    return {
        "plan_id": plan.plan_id,
        "plan_sha256": plan.plan_sha256,
        "authorization_sha256": _sha256_file(authorization_path),
        "controller_ledger_sha256": _sha256_file(ledger),
        "completed_games": status["progress"]["completed_games"],
        "completed_segments": status["progress"]["completed_segments"],
        "completed_post_fork_transitions": status["progress"].get(
            "completed_post_fork_transitions"
        ),
        "elapsed_hours": status["progress"]["elapsed_hours"],
        "state": status["state"],
    }


def _training_resource_audit(
    completed: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    resources = contract["resources"]
    prefixes = [item for item in completed if item.get("kind") == "run-prefix"]
    arms = [item for item in completed if item.get("kind") == "run-arm"]
    if len(prefixes) != 3 or len(arms) != 6:
        raise ScheduleIsolationSequenceError(
            "completed training sequence cell count differs"
        )
    prefix_game_count = int(resources["prefix_game_count"])
    prefix_games = [int(item["result"]["completed_games"]) for item in prefixes]
    if any(value != prefix_game_count for value in prefix_games):
        raise ScheduleIsolationSequenceError("prefix game count differs")
    arm_absolute_games = [int(item["result"]["completed_games"]) for item in arms]
    if any(
        value < prefix_game_count
        or value > int(resources["arm_absolute_game_count_ceiling"])
        for value in arm_absolute_games
    ):
        raise ScheduleIsolationSequenceError("arm game count differs")
    transitions = [
        item["result"].get("completed_post_fork_transitions") for item in arms
    ]
    expected_per_arm = int(resources["scientific_post_fork_transitions_per_arm"])
    if any(value != expected_per_arm for value in transitions):
        raise ScheduleIsolationSequenceError("arm post-fork transition count differs")
    contract_games = sum(arm_absolute_games)
    actual_games = sum(prefix_games) + sum(
        value - prefix_game_count for value in arm_absolute_games
    )
    transition_total = sum(int(value) for value in transitions)
    managed_active_hours = sum(
        float(item["result"]["elapsed_hours"]) for item in (*prefixes, *arms)
    )
    if contract_games > int(resources["maximum_contract_training_games_total"]):
        raise ScheduleIsolationSequenceError("contract game ceiling exceeded")
    if actual_games > int(resources["maximum_actual_training_games_total"]):
        raise ScheduleIsolationSequenceError("actual training game ceiling exceeded")
    if transition_total != int(resources["scientific_post_fork_transitions_total"]):
        raise ScheduleIsolationSequenceError(
            "aggregate post-fork transition count differs"
        )
    maximum_active_hours = float(resources["maximum_active_wall_hours_total"])
    if managed_active_hours > maximum_active_hours + 0.0005:
        raise ScheduleIsolationSequenceError("managed active-hour ceiling exceeded")
    return {
        "prefix_training_games": sum(prefix_games),
        "arm_absolute_game_counts": arm_absolute_games,
        "arm_contract_games_total": contract_games,
        "actual_training_games_total": actual_games,
        "post_fork_transitions_per_arm": transitions,
        "post_fork_transitions_total": transition_total,
        "managed_active_hours_total": managed_active_hours,
        "maximum_managed_active_hours": maximum_active_hours,
        "training_game_limits_passed": True,
    }


def launch_sequence(
    *,
    contract_path: Path,
    readiness_path: Path,
    authorization_path: Path,
    expected_sequence_readiness_identity: str,
    paths_config: Path,
    run_id: str,
    marker_path: Path = DEFAULT_MARKER,
    failure_path: Path = DEFAULT_FAILURE,
    sequence_result_path: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    """Consume one parent attempt and run every child exactly once in order."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ScheduleIsolationSequenceError("run id is required")
    readiness_report = inspect_sequence_readiness(
        contract_path=contract_path,
        readiness_path=readiness_path,
        authorization_path=authorization_path,
        marker_path=marker_path,
        failure_path=failure_path,
        sequence_result_path=sequence_result_path,
        allow_authorization=True,
    )
    contract = load_equal_transition_contract(contract_path)
    readiness = _strict_json(readiness_path)
    managed_readiness_identity = validate_readiness_identity(readiness)
    sequence_readiness_identity = readiness_report["sequence_readiness_identity"]
    if sequence_readiness_identity != expected_sequence_readiness_identity:
        raise ScheduleIsolationSequenceError(
            "expected sequence readiness identity differs"
        )
    authorization = _strict_json(authorization_path)
    parent_identity = validate_sequence_authorization(
        authorization,
        contract=contract,
        readiness=readiness,
        sequence_readiness_identity=sequence_readiness_identity,
    )
    marker_body = {
        "schema_version": SEQUENCE_LAUNCH_SCHEMA,
        "status": "started_once",
        "run_id": run_id,
        "started_at_utc": utc_now_text(),
        "authorization_identity": parent_identity,
        "managed_readiness_identity": managed_readiness_identity,
        "sequence_readiness_identity": sequence_readiness_identity,
        "plan_identity": contract["plan_identity"],
        "source_commit": readiness_report["source"]["head"],
        "development_analysis_device": DEVELOPMENT_ANALYSIS_DEVICE,
        "retry_or_recovery_authorized": False,
    }
    marker = {**marker_body, "launch_identity": canonical_sha256(marker_body)}
    _publish_exclusive(marker_path, marker)

    prefixes = {int(item["seed"]): item for item in contract["prefixes"]}
    arms = {
        (int(item["seed"]), str(item["condition"])): item
        for item in contract["arms"]
    }
    completed: list[dict[str, Any]] = []

    def run_prefix(step: SequenceStep) -> dict[str, Any]:
        if step.seed is None:
            raise ScheduleIsolationSequenceError("prefix seed is absent")
        child = _authorize_and_run_child(
            prefixes[step.seed],
            parent_authorization_identity=parent_identity,
        )
        record = {**step.to_dict(), "result": child}
        completed.append(record)
        return record

    def prepare_arms(step: SequenceStep) -> dict[str, Any]:
        if step.seed is None:
            raise ScheduleIsolationSequenceError("arm-preparation seed is absent")
        report_path = ROOT / f"out/{FAMILY}/seed{step.seed}-arm-readiness.json"
        report = prepare_seed_arms(
            root=ROOT,
            contract_path=contract_path,
            paths_config=paths_config,
            seed=step.seed,
            report_path=report_path,
            python_executable=sys.executable,
        )
        if (
            report.get("state")
            != "seed_arm_plans_ready_for_product_authorization"
            or report.get("launch_authorized") is not False
        ):
            raise ScheduleIsolationSequenceError(
                f"seed {step.seed} arm readiness differs"
            )
        record = {
            **step.to_dict(),
            "result": {
                "readiness_path": str(report_path),
                "readiness_sha256": _sha256_file(report_path),
                "readiness_identity": report["readiness_identity"],
                "state": report["state"],
            },
        }
        completed.append(record)
        return record

    def run_arm(step: SequenceStep) -> dict[str, Any]:
        if step.seed is None or step.condition is None:
            raise ScheduleIsolationSequenceError("arm identity is incomplete")
        child = _authorize_and_run_child(
            arms[(step.seed, step.condition)],
            parent_authorization_identity=parent_identity,
        )
        record = {**step.to_dict(), "result": child}
        completed.append(record)
        return record

    def publish_result(step: SequenceStep) -> dict[str, Any]:
        training_resource_audit = _training_resource_audit(
            completed,
            contract=contract,
        )
        exit_code = reporter.main(
            [
                "--contract",
                str(contract_path),
                "--readiness",
                str(readiness_path),
                "--paths-config",
                str(paths_config),
                "--ledger",
                str(DEFAULT_DEVELOPMENT_LEDGER),
                "--output",
                str(DEFAULT_DEVELOPMENT_RESULT),
                "--device",
                DEVELOPMENT_ANALYSIS_DEVICE,
            ]
        )
        if exit_code != 0:
            raise ScheduleIsolationSequenceError("development publisher failed")
        result = _strict_json(DEFAULT_DEVELOPMENT_RESULT)
        scope = result.get("scope", {})
        expected_measurement_games = int(
            contract["resources"]["maximum_development_measurement_games_total"]
        )
        if (
            scope.get("no_update_development_games") != expected_measurement_games
            or scope.get("optimizer_updates") != 0
            or scope.get("training_games") != 0
            or scope.get("checkpoint_writes") != 0
            or scope.get("database_writes") != 0
        ):
            raise ScheduleIsolationSequenceError(
                "development measurement resource accounting differs"
            )
        resource_audit = {
            **training_resource_audit,
            "development_measurement_games_total": expected_measurement_games,
            "development_measurement_optimizer_updates": 0,
            "development_measurement_training_writes": False,
            "aggregate_resource_limits_passed": True,
        }
        record = {
            **step.to_dict(),
            "result": {
                "path": str(DEFAULT_DEVELOPMENT_RESULT),
                "sha256": _sha256_file(DEFAULT_DEVELOPMENT_RESULT),
                "result_identity": result["result_identity"],
                "classification": result["decision"]["classification"],
                "resource_audit": resource_audit,
            },
        }
        completed.append(record)
        return record

    try:
        execute_sequence_steps(
            build_sequence_steps(contract),
            run_prefix=run_prefix,
            prepare_seed_arms=prepare_arms,
            run_arm=run_arm,
            publish_result=publish_result,
        )
    except Exception as exc:
        failure_body = {
            "schema_version": SEQUENCE_FAILURE_SCHEMA,
            "status": "failed_closed",
            "run_id": run_id,
            "failed_at_utc": utc_now_text(),
            "authorization_identity": parent_identity,
            "managed_readiness_identity": managed_readiness_identity,
            "sequence_readiness_identity": sequence_readiness_identity,
            "plan_identity": contract["plan_identity"],
            "completed_steps": completed,
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "retry_or_recovery_authorized": False,
            "held_out_promotion_publication_or_long_run_authorized": False,
        }
        failure = {
            **failure_body,
            "failure_identity": canonical_sha256(failure_body),
        }
        _publish_exclusive(failure_path, failure)
        raise

    result_body = {
        "schema_version": SEQUENCE_RESULT_SCHEMA,
        "status": "completed_once",
        "run_id": run_id,
        "completed_at_utc": utc_now_text(),
        "launch_identity": marker["launch_identity"],
        "authorization_identity": parent_identity,
        "managed_readiness_identity": managed_readiness_identity,
        "sequence_readiness_identity": sequence_readiness_identity,
        "plan_identity": contract["plan_identity"],
        "completed_steps": completed,
        "development_result": completed[-1]["result"],
        "claim_boundary": contract["claim_boundary"],
        "held_out_promotion_publication_or_long_run_authorized": False,
    }
    result = {**result_body, "sequence_result_identity": canonical_sha256(result_body)}
    _publish_exclusive(sequence_result_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--record-authorization", action="store_true")
    action.add_argument("--launch", choices=("once",))
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--expected-sequence-readiness-identity")
    parser.add_argument("--decision-note")
    parser.add_argument("--run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.preflight:
            if any(
                value is not None
                for value in (
                    args.expected_sequence_readiness_identity,
                    args.decision_note,
                    args.run_id,
                )
            ):
                raise ScheduleIsolationSequenceError(
                    "preflight does not accept authorization or run fields"
                )
            result = inspect_sequence_readiness(
                contract_path=args.contract,
                readiness_path=args.readiness,
                authorization_path=args.authorization,
            )
        elif args.record_authorization:
            if not args.expected_sequence_readiness_identity or not args.decision_note:
                raise ScheduleIsolationSequenceError(
                    "record-authorization requires sequence readiness identity "
                    "and decision"
                )
            if args.run_id is not None:
                raise ScheduleIsolationSequenceError(
                    "record-authorization does not accept a run id"
                )
            result = record_sequence_authorization(
                contract_path=args.contract,
                readiness_path=args.readiness,
                authorization_path=args.authorization,
                expected_sequence_readiness_identity=(
                    args.expected_sequence_readiness_identity
                ),
                decision_note=args.decision_note,
            )
        else:
            if not args.expected_sequence_readiness_identity or not args.run_id:
                raise ScheduleIsolationSequenceError(
                    "launch requires sequence readiness identity and run id"
                )
            if args.decision_note is not None:
                raise ScheduleIsolationSequenceError(
                    "launch reads the recorded decision from authorization"
                )
            result = launch_sequence(
                contract_path=args.contract,
                readiness_path=args.readiness,
                authorization_path=args.authorization,
                expected_sequence_readiness_identity=(
                    args.expected_sequence_readiness_identity
                ),
                paths_config=args.paths_config,
                run_id=args.run_id,
            )
    except (ScheduleIsolationSequenceError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
