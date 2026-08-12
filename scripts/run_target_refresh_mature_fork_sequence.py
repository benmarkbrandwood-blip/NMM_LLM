#!/usr/bin/env python3
"""Preflight, authorize, and run the mature target-refresh sequence once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.training.generalist_run_manifest import utc_now_text  # noqa: E402
from learned_ai.training.managed_generalist import (  # noqa: E402
    authorize_plan,
    load_managed_plan,
    run_authorized_plan,
)
from learned_ai.training.run_contract import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from learned_ai.validation.target_refresh_equal_transition_arms import (  # noqa: E402
    _inspect_closed_specialist_database,
)
from learned_ai.validation.target_refresh_mature_fork_diagnostic import (  # noqa: E402
    load_contract,
)
from learned_ai.validation.target_refresh_mature_fork_sequence import (  # noqa: E402
    DELEGATED_OPERATOR,
    MatureTargetRefreshSequenceError,
    SequenceStep,
    build_sequence_authorization,
    build_sequence_steps,
    execute_sequence_steps,
    validate_readiness_identity,
    validate_sequence_authorization,
)
from scripts import report_target_refresh_mature_fork_diagnostic as reporter  # noqa: E402


FAMILY = "target-refresh-mature-fork-diagnostic-v1-attempt-002"
DEFAULT_CONTRACT = ROOT / (
    "docs/experiments/sanmill-target-refresh-mature-fork-diagnostic-v1-attempt-002.json"
)
DEFAULT_READINESS = ROOT / f"out/{FAMILY}/readiness.json"
DEFAULT_AUTHORIZATION = ROOT / f"out/{FAMILY}/sequence-authorization.json"
DEFAULT_LAUNCH = ROOT / f"out/{FAMILY}/sequence-launch.json"
DEFAULT_COMPLETION = ROOT / f"out/{FAMILY}/sequence-completion.json"
DEFAULT_FAILURE = ROOT / f"out/{FAMILY}/sequence-failure.json"
DEFAULT_PATHS_CONFIG = ROOT / "data/training_paths.local.json"
LAUNCH_SCHEMA = "nmm.target-refresh-mature-fork-sequence-launch.v1"
COMPLETION_SCHEMA = "nmm.target-refresh-mature-fork-sequence-completion.v1"
FAILURE_SCHEMA = "nmm.target-refresh-mature-fork-sequence-failure.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside_root(value: str | Path, *, field: str) -> Path:
    path = Path(value)
    resolved = (
        path.resolve(strict=False)
        if path.is_absolute()
        else (ROOT / path).resolve(strict=False)
    )
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise MatureTargetRefreshSequenceError(
            f"{field} is outside the repository"
        ) from exc
    return resolved


def _contract_output_paths(contract: Mapping[str, Any]) -> dict[str, Path]:
    outputs = contract.get("result_outputs")
    required = ("authorization", "launch", "completion", "failure", "ledger", "result")
    if not isinstance(outputs, Mapping) or any(
        not isinstance(outputs.get(name), str) or not outputs[name] for name in required
    ):
        raise MatureTargetRefreshSequenceError("contract result outputs differ")
    resolved = {
        name: _inside_root(outputs[name], field=f"contract {name} output")
        for name in required
    }
    if len(set(resolved.values())) != len(resolved):
        raise MatureTargetRefreshSequenceError("contract result outputs overlap")
    return resolved


def _bind_output_path(
    supplied: Path | None,
    *,
    expected: Path,
    field: str,
) -> Path:
    if supplied is None:
        return expected
    resolved = _inside_root(supplied, field=field)
    if resolved != expected:
        raise MatureTargetRefreshSequenceError(f"{field} differs from contract")
    return resolved


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MatureTargetRefreshSequenceError(f"cannot read JSON: {path}") from exc
    if (
        not isinstance(value, dict)
        or b"\r" in raw
        or not raw.endswith(b"\n")
        or raw != canonical_json_bytes(value) + b"\n"
    ):
        raise MatureTargetRefreshSequenceError(f"JSON is not canonical: {path}")
    return value


def _publish_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    target = _inside_root(path, field="sequence output")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise MatureTargetRefreshSequenceError(
            f"sequence output already exists: {target}"
        ) from exc


def _git_source(expected: str) -> dict[str, Any]:
    def output(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise MatureTargetRefreshSequenceError("Git audit failed")
        return result.stdout.strip()

    branch = output("branch", "--show-current")
    head = output("rev-parse", "HEAD")
    origin = output("rev-parse", "origin/dev")
    dirty = output("status", "--porcelain=v1", "--untracked-files=no")
    if branch != "dev" or head != expected or origin != expected or dirty:
        raise MatureTargetRefreshSequenceError(
            "sequence requires the exact clean published readiness source"
        )
    return {
        "branch": branch,
        "head": head,
        "origin_dev": origin,
        "tracked_clean": True,
        "published": True,
    }


def _readiness_arm_map(
    readiness: Mapping[str, Any],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    result = {}
    for seed_record in readiness.get("seeds", []):
        seed = int(seed_record["seed"])
        for arm in seed_record["arms"]:
            result[(seed, str(arm["condition"]))] = arm
    return result


def inspect_readiness(
    *,
    contract_path: Path,
    readiness_path: Path,
    authorization_path: Path | None = None,
    launch_path: Path | None = None,
    completion_path: Path | None = None,
    failure_path: Path | None = None,
    allow_authorization: bool = False,
) -> dict[str, Any]:
    contract = load_contract(contract_path.resolve(strict=True))
    outputs = _contract_output_paths(contract)
    authorization_path = _bind_output_path(
        authorization_path,
        expected=outputs["authorization"],
        field="sequence authorization output",
    )
    launch_path = _bind_output_path(
        launch_path,
        expected=outputs["launch"],
        field="sequence launch output",
    )
    completion_path = _bind_output_path(
        completion_path,
        expected=outputs["completion"],
        field="sequence completion output",
    )
    failure_path = _bind_output_path(
        failure_path,
        expected=outputs["failure"],
        field="sequence failure output",
    )
    readiness = _strict_json(readiness_path.resolve(strict=True))
    readiness_identity = validate_readiness_identity(readiness)
    if readiness["contract"]["plan_identity"] != contract["plan_identity"] or readiness[
        "contract"
    ]["sha256"] != _sha256_file(contract_path):
        raise MatureTargetRefreshSequenceError("readiness contract binding differs")
    source = _git_source(str(readiness["source"]["head"]))
    readiness_arms = _readiness_arm_map(readiness)
    if len(readiness_arms) != 6:
        raise MatureTargetRefreshSequenceError("readiness arm cells differ")
    audits = []
    for arm in contract["arms"]:
        key = (int(arm["seed"]), str(arm["condition"]))
        record = readiness_arms.get(key)
        if record is None:
            raise MatureTargetRefreshSequenceError(f"readiness arm is absent: {key}")
        control = _inside_root(arm["control_dir"], field="arm control")
        plan_path = control / "plan.json"
        preflight_path = _inside_root(
            record["preflight"]["path"], field="arm preflight"
        )
        db_path = _inside_root(arm["specialist_db"], field="arm database")
        branch_path = _inside_root(
            record["branch_checkpoint"]["destination_path"],
            field="arm branch",
        )
        plan = load_managed_plan(plan_path)
        database = _inspect_closed_specialist_database(db_path)
        if (
            plan.plan_sha256 != record["plan_sha256"]
            or plan.git_commit != source["head"]
            or plan.policy_health is None
            or _sha256_file(preflight_path) != record["preflight"]["sha256"]
            or record["preflight"]["verdict"] != "needs_decision"
            or database["sha256"] != record["specialist_db"]["sha256"]
            or _sha256_file(branch_path)
            != record["branch_checkpoint"]["destination_sha256"]
            or record.get("authorization_present") is not False
            or record.get("segment_output_present") is not False
            or (control / "authorization.json").exists()
            or (control / "segments").exists()
        ):
            raise MatureTargetRefreshSequenceError(
                f"fresh arm evidence differs: {arm['arm_id']}"
            )
        audits.append(
            {
                "arm_id": arm["arm_id"],
                "launch_order": arm["launch_order"],
                "plan_path": str(plan_path),
                "plan_sha256": plan.plan_sha256,
                "preflight_sha256": record["preflight"]["sha256"],
                "branch_sha256": record["branch_checkpoint"]["destination_sha256"],
                "specialist_db_sha256": database["sha256"],
                "policy_health_gate": plan.policy_health.to_dict(),
            }
        )
    required_absent = [launch_path, completion_path, failure_path]
    if not allow_authorization:
        required_absent.append(authorization_path)
    occupied = [str(path) for path in required_absent if path.exists()]
    if occupied:
        raise MatureTargetRefreshSequenceError(
            "one-shot sequence output already exists: " + ", ".join(occupied)
        )
    return {
        "schema_version": "nmm.target-refresh-mature-fork-final-readiness.v1",
        "state": "ready_for_one_parent_product_authorization",
        "verdict": "needs_decision" if not allow_authorization else "ready",
        "launch_authorized": allow_authorization and authorization_path.exists(),
        "readiness_identity": readiness_identity,
        "plan_identity": contract["plan_identity"],
        "source": source,
        "arms": audits,
        "resource_envelope": contract["resources"],
        "claim_boundary": contract["claim_boundary"],
    }


def record_authorization(
    *,
    contract_path: Path,
    readiness_path: Path,
    authorization_path: Path,
    expected_readiness_identity: str,
    decision_note: str,
) -> dict[str, Any]:
    inspect_readiness(
        contract_path=contract_path,
        readiness_path=readiness_path,
        authorization_path=authorization_path,
    )
    contract = load_contract(contract_path)
    readiness = _strict_json(readiness_path)
    authorization = build_sequence_authorization(
        contract=contract,
        readiness=readiness,
        expected_readiness_identity=expected_readiness_identity,
        decision_note=decision_note,
        authorized_at_utc=utc_now_text(),
    )
    _publish_exclusive(authorization_path, authorization)
    return authorization


def _authorize_and_run_arm(
    arm: Mapping[str, Any], *, parent_authorization_identity: str
) -> dict[str, Any]:
    control = _inside_root(arm["control_dir"], field="arm control")
    plan_path = control / "plan.json"
    authorization_path = control / "authorization.json"
    if authorization_path.exists():
        raise MatureTargetRefreshSequenceError(
            f"child authorization already exists: {arm['arm_id']}"
        )
    plan = load_managed_plan(plan_path)
    authorize_plan(
        plan_path,
        authorization_path,
        authorized_by=DELEGATED_OPERATOR,
        decision_note=(
            "Delegated by frozen mature target-refresh parent authorization "
            f"{parent_authorization_identity}; one attempt, no recovery"
        ),
    )
    status = run_authorized_plan(plan_path, authorization_path)
    if status.get("state") != "completed":
        raise MatureTargetRefreshSequenceError(
            f"child plan did not complete: {plan.plan_id}"
        )
    events = control / "controller-events.jsonl"
    return {
        "plan_id": plan.plan_id,
        "plan_sha256": plan.plan_sha256,
        "authorization_sha256": _sha256_file(authorization_path),
        "controller_events_sha256": _sha256_file(events),
        "completed_games": status["progress"]["completed_games"],
        "completed_segments": status["progress"]["completed_segments"],
        "completed_post_fork_transitions": status["progress"].get(
            "completed_post_fork_transitions"
        ),
        "elapsed_hours": status["progress"]["elapsed_hours"],
        "state": status["state"],
    }


def _resource_audit(
    completed: Sequence[Mapping[str, Any]], *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    arms = [item for item in completed if item.get("kind") == "run-arm"]
    if len(arms) != 6:
        raise MatureTargetRefreshSequenceError("completed arm count differs")
    sources = {int(item["seed"]): item for item in contract["sources"]}
    actual_games = []
    transitions = []
    active_hours = 0.0
    for item in arms:
        seed = int(item["seed"])
        result = item["result"]
        delta = int(result["completed_games"]) - int(sources[seed]["game_count"])
        if not 0 < delta <= contract["resources"]["maximum_training_games_per_arm"]:
            raise MatureTargetRefreshSequenceError("arm game count differs")
        if result["completed_post_fork_transitions"] != 8192:
            raise MatureTargetRefreshSequenceError("arm transition count differs")
        actual_games.append(delta)
        transitions.append(8192)
        active_hours += float(result["elapsed_hours"])
    if (
        sum(actual_games) > contract["resources"]["maximum_training_games_total"]
        or sum(transitions)
        != contract["resources"]["maximum_optimizer_consumed_transitions_total"]
        or active_hours > contract["resources"]["maximum_active_wall_hours_total"]
    ):
        raise MatureTargetRefreshSequenceError("aggregate resource limit exceeded")
    return {
        "training_games_per_arm": actual_games,
        "training_games_total": sum(actual_games),
        "post_mature_fork_transitions_per_arm": transitions,
        "post_mature_fork_transitions_total": sum(transitions),
        "managed_active_hours_total": active_hours,
        "training_resource_limits_passed": True,
    }


def launch_sequence(
    *,
    contract_path: Path,
    readiness_path: Path,
    authorization_path: Path,
    expected_readiness_identity: str,
    paths_config: Path,
    run_id: str,
    launch_path: Path | None = None,
    completion_path: Path | None = None,
    failure_path: Path | None = None,
    ledger_path: Path | None = None,
    result_path: Path | None = None,
) -> dict[str, Any]:
    if not run_id.strip():
        raise MatureTargetRefreshSequenceError("run id is required")
    contract = load_contract(contract_path)
    outputs = _contract_output_paths(contract)
    authorization_path = _bind_output_path(
        authorization_path,
        expected=outputs["authorization"],
        field="sequence authorization output",
    )
    launch_path = _bind_output_path(
        launch_path,
        expected=outputs["launch"],
        field="sequence launch output",
    )
    completion_path = _bind_output_path(
        completion_path,
        expected=outputs["completion"],
        field="sequence completion output",
    )
    failure_path = _bind_output_path(
        failure_path,
        expected=outputs["failure"],
        field="sequence failure output",
    )
    ledger_path = _bind_output_path(
        ledger_path,
        expected=outputs["ledger"],
        field="development ledger output",
    )
    result_path = _bind_output_path(
        result_path,
        expected=outputs["result"],
        field="development result output",
    )
    inspect_readiness(
        contract_path=contract_path,
        readiness_path=readiness_path,
        authorization_path=authorization_path,
        launch_path=launch_path,
        completion_path=completion_path,
        failure_path=failure_path,
        allow_authorization=True,
    )
    readiness = _strict_json(readiness_path)
    authorization = _strict_json(authorization_path)
    parent_identity = validate_sequence_authorization(
        authorization,
        contract=contract,
        readiness=readiness,
        expected_readiness_identity=expected_readiness_identity,
    )
    launch_body = {
        "schema_version": LAUNCH_SCHEMA,
        "status": "started_once",
        "run_id": run_id,
        "started_at_utc": utc_now_text(),
        "authorization_identity": parent_identity,
        "readiness_identity": expected_readiness_identity,
        "plan_identity": contract["plan_identity"],
        "source_commit": readiness["source"]["head"],
        "retry_or_recovery_authorized": False,
    }
    launch = {**launch_body, "launch_identity": canonical_sha256(launch_body)}
    _publish_exclusive(launch_path, launch)
    started = time.monotonic()
    arms = {(int(arm["seed"]), str(arm["condition"])): arm for arm in contract["arms"]}
    completed: list[dict[str, Any]] = []

    def run_arm(step: SequenceStep) -> dict[str, Any]:
        if step.seed is None or step.condition is None:
            raise MatureTargetRefreshSequenceError("arm identity is incomplete")
        result = _authorize_and_run_arm(
            arms[(step.seed, step.condition)],
            parent_authorization_identity=parent_identity,
        )
        record = {**step.to_dict(), "result": result}
        completed.append(record)
        return record

    def publish_result(step: SequenceStep) -> dict[str, Any]:
        training_resources = _resource_audit(completed, contract=contract)
        elapsed = (time.monotonic() - started) / 3600.0
        if elapsed >= contract["resources"]["maximum_active_wall_hours_total"]:
            raise MatureTargetRefreshSequenceError(
                "no active wall time remains for development analysis"
            )
        result_code = reporter.main(
            [
                "--contract",
                str(contract_path),
                "--readiness",
                str(readiness_path),
                "--paths-config",
                str(paths_config),
                "--ledger",
                str(ledger_path),
                "--output",
                str(result_path),
                "--device",
                "cpu",
            ]
        )
        if result_code != 0:
            raise MatureTargetRefreshSequenceError("development reporter failed")
        result = _strict_json(result_path)
        if (
            result["scope"]["no_update_development_games"] != 288
            or result["scope"]["training_games"] != 0
            or result["scope"]["optimizer_updates"] != 0
        ):
            raise MatureTargetRefreshSequenceError(
                "development result resource accounting differs"
            )
        record = {
            **step.to_dict(),
            "result": {
                "path": str(result_path),
                "sha256": _sha256_file(result_path),
                "result_identity": result["result_identity"],
                "classification": result.get(
                    "replication_decision", result["decision"]
                )["classification"],
                "cohort_classification": result["decision"]["classification"],
                "training_resource_audit": training_resources,
                "no_update_games": 288,
            },
        }
        completed.append(record)
        return record

    try:
        execute_sequence_steps(
            build_sequence_steps(contract),
            run_arm=run_arm,
            publish_result=publish_result,
        )
        elapsed_hours = (time.monotonic() - started) / 3600.0
        if elapsed_hours > contract["resources"]["maximum_active_wall_hours_total"]:
            raise MatureTargetRefreshSequenceError(
                "sequence active wall-time limit exceeded"
            )
    except Exception as exc:
        failure_body = {
            "schema_version": FAILURE_SCHEMA,
            "status": "failed_closed",
            "run_id": run_id,
            "failed_at_utc": utc_now_text(),
            "elapsed_hours": (time.monotonic() - started) / 3600.0,
            "launch_identity": launch["launch_identity"],
            "authorization_identity": parent_identity,
            "readiness_identity": expected_readiness_identity,
            "plan_identity": contract["plan_identity"],
            "completed_steps": completed,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "retry_or_recovery_authorized": False,
            "held_out_promotion_publication_or_long_run_authorized": False,
        }
        _publish_exclusive(
            failure_path,
            {**failure_body, "failure_identity": canonical_sha256(failure_body)},
        )
        raise
    completion_body = {
        "schema_version": COMPLETION_SCHEMA,
        "status": "completed_once",
        "run_id": run_id,
        "completed_at_utc": utc_now_text(),
        "elapsed_hours": (time.monotonic() - started) / 3600.0,
        "launch_identity": launch["launch_identity"],
        "authorization_identity": parent_identity,
        "readiness_identity": expected_readiness_identity,
        "plan_identity": contract["plan_identity"],
        "completed_steps": completed,
        "development_result": completed[-1]["result"],
        "claim_boundary": contract["claim_boundary"],
        "held_out_promotion_publication_or_long_run_authorized": False,
    }
    completion = {
        **completion_body,
        "completion_identity": canonical_sha256(completion_body),
    }
    _publish_exclusive(completion_path, completion)
    return completion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--record-authorization", action="store_true")
    action.add_argument("--launch", choices=("once",))
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--expected-readiness-identity")
    parser.add_argument("--decision-note")
    parser.add_argument("--run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = args.contract.resolve()
    readiness = args.readiness.resolve()
    authorization = args.authorization.resolve() if args.authorization else None
    if args.preflight:
        result = inspect_readiness(
            contract_path=contract,
            readiness_path=readiness,
            authorization_path=authorization,
        )
    elif args.record_authorization:
        if not args.expected_readiness_identity or not args.decision_note:
            raise MatureTargetRefreshSequenceError(
                "record-authorization requires identity and decision note"
            )
        result = record_authorization(
            contract_path=contract,
            readiness_path=readiness,
            authorization_path=(
                authorization
                if authorization is not None
                else _contract_output_paths(load_contract(contract))["authorization"]
            ),
            expected_readiness_identity=args.expected_readiness_identity,
            decision_note=args.decision_note,
        )
    else:
        if not args.expected_readiness_identity or not args.run_id:
            raise MatureTargetRefreshSequenceError(
                "launch requires identity and run id"
            )
        result = launch_sequence(
            contract_path=contract,
            readiness_path=readiness,
            authorization_path=(
                authorization
                if authorization is not None
                else _contract_output_paths(load_contract(contract))["authorization"]
            ),
            expected_readiness_identity=args.expected_readiness_identity,
            paths_config=args.paths_config.resolve(strict=True),
            run_id=args.run_id,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MatureTargetRefreshSequenceError as exc:
        print(f"fatal_stop: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
