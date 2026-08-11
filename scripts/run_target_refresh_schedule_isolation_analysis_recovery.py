#!/usr/bin/env python3
"""Preflight, authorize, and run one analysis-only recovery attempt."""

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
from learned_ai.training.run_contract import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from learned_ai.validation.target_refresh_equal_transition_diagnostic import (  # noqa: E402
    load_equal_transition_contract,
)
from scripts import report_target_refresh_schedule_isolation_diagnostic as reporter  # noqa: E402
from scripts.report_target_refresh_equal_transition_diagnostic import (  # noqa: E402
    _arm_by_cell,
)
from scripts.run_target_refresh_schedule_isolation_sequence import (  # noqa: E402
    _training_resource_audit,
)


PLAN_SCHEMA = "nmm.target-refresh-schedule-isolation-analysis-recovery-plan.v1"
READINESS_SCHEMA = (
    "nmm.target-refresh-schedule-isolation-analysis-recovery-readiness.v1"
)
AUTHORIZATION_SCHEMA = (
    "nmm.target-refresh-schedule-isolation-analysis-recovery-authorization.v1"
)
LAUNCH_SCHEMA = (
    "nmm.target-refresh-schedule-isolation-analysis-recovery-launch.v1"
)
COMPLETION_SCHEMA = (
    "nmm.target-refresh-schedule-isolation-analysis-recovery-completion.v1"
)
FAILURE_SCHEMA = (
    "nmm.target-refresh-schedule-isolation-analysis-recovery-failure.v1"
)
DEFAULT_PLAN = ROOT / (
    "docs/experiments/"
    "sanmill-target-refresh-schedule-isolation-analysis-recovery-v1.json"
)
OUTPUT_ROOT = ROOT / "out/target-refresh-schedule-isolation-diagnostic-v2"
DEFAULT_READINESS = OUTPUT_ROOT / "analysis-recovery-readiness.json"
DEFAULT_AUTHORIZATION = OUTPUT_ROOT / "analysis-recovery-authorization.json"
DEFAULT_LAUNCH = OUTPUT_ROOT / "analysis-recovery-launch.json"
DEFAULT_COMPLETION = OUTPUT_ROOT / "analysis-recovery-completion.json"
DEFAULT_FAILURE = OUTPUT_ROOT / "analysis-recovery-failure.json"
DEFAULT_STDOUT = OUTPUT_ROOT / "analysis-recovery.stdout.log"
DEFAULT_STDERR = OUTPUT_ROOT / "analysis-recovery.stderr.log"


class AnalysisRecoveryError(RuntimeError):
    """Raised when the analysis-only recovery cannot be proven safe."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AnalysisRecoveryError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AnalysisRecoveryError(
                    f"non-finite JSON value in {path.name}: {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisRecoveryError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AnalysisRecoveryError(f"JSON root is not an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside_root(value: str | Path, *, field: str) -> Path:
    candidate = Path(value)
    resolved = (
        candidate.resolve(strict=False)
        if candidate.is_absolute()
        else (ROOT / candidate).resolve(strict=False)
    )
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise AnalysisRecoveryError(f"{field} is outside the repository") from exc
    return resolved


def _relative(path: Path) -> str:
    return path.resolve(strict=False).relative_to(ROOT.resolve()).as_posix()


def _publish_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    target = _inside_root(path, field="output")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise AnalysisRecoveryError(f"output already exists: {target}") from exc


def _publish_bytes_exclusive(path: Path, value: bytes) -> None:
    target = _inside_root(path, field="log output")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise AnalysisRecoveryError(f"output already exists: {target}") from exc


def load_recovery_plan(path: Path) -> dict[str, Any]:
    plan = _strict_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise AnalysisRecoveryError("analysis recovery plan schema differs")
    identity = plan.get("plan_identity")
    body = dict(plan)
    body.pop("plan_identity", None)
    if identity != canonical_sha256(body):
        raise AnalysisRecoveryError("analysis recovery plan identity differs")
    scope = plan.get("resource_envelope")
    expected_scope = {
        "candidate_models_loaded": True,
        "checkpoint_writes": 0,
        "database_writes": 0,
        "maximum_active_wall_hours": 5.5,
        "no_update_development_games": 288,
        "optimizer_updates": 0,
        "training_games": 0,
    }
    if scope != expected_scope:
        raise AnalysisRecoveryError("analysis recovery resource envelope differs")
    if plan.get("status") != "designed_unlaunched_needs_authorization":
        raise AnalysisRecoveryError("analysis recovery plan status differs")
    return plan


def _artifact(record: Mapping[str, Any], *, field: str) -> tuple[Path, dict[str, Any]]:
    path = _inside_root(str(record.get("path", "")), field=field)
    if not path.is_file() or _sha256_file(path) != record.get("sha256"):
        raise AnalysisRecoveryError(f"{field} identity differs")
    return path, _strict_json(path)


def _canonical_identity(
    value: Mapping[str, Any], *, field: str, identity_field: str
) -> str:
    identity = value.get(identity_field)
    body = dict(value)
    body.pop(identity_field, None)
    if identity != canonical_sha256(body):
        raise AnalysisRecoveryError(f"{field} canonical identity differs")
    return str(identity)


def _audit_completed_training(
    *, plan: Mapping[str, Any], parent_contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, failure = _artifact(plan["parent_attempt"]["failure"], field="failure")
    failure_identity = _canonical_identity(
        failure,
        field="failure",
        identity_field="failure_identity",
    )
    expected = plan["parent_attempt"]
    if (
        failure_identity != expected["failure"]["identity"]
        or failure.get("status") != "failed_closed"
        or failure.get("plan_identity") != parent_contract["plan_identity"]
        or failure.get("retry_or_recovery_authorized") is not False
        or failure.get("failure", {}).get("type")
        != "ScheduleIsolationReportError"
        or "JSONL framing differs" not in str(
            failure.get("failure", {}).get("message", "")
        )
    ):
        raise AnalysisRecoveryError("parent failure semantics differ")
    completed = failure.get("completed_steps")
    if not isinstance(completed, list):
        raise AnalysisRecoveryError("parent completed steps are absent")
    resources = _training_resource_audit(completed, contract=parent_contract)
    if resources != plan["completed_training"]:
        raise AnalysisRecoveryError("completed training resource audit differs")
    if any(item.get("kind") == "publish-development-result" for item in completed):
        raise AnalysisRecoveryError("parent attempt already ran development analysis")
    return resources, failure


def _validate_parent_artifacts(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent = plan["parent_attempt"]
    contract_path, _ = _artifact(parent["contract"], field="parent contract")
    parent_contract = load_equal_transition_contract(contract_path)
    if parent_contract["plan_identity"] != parent["contract"]["identity"]:
        raise AnalysisRecoveryError("parent plan identity differs")
    readiness_path, readiness = _artifact(
        parent["readiness"], field="parent readiness"
    )
    validated_readiness = reporter._validate_readiness(
        readiness_path, contract=parent_contract
    )
    if (
        readiness != validated_readiness
        or readiness["readiness_identity"] != parent["readiness"]["identity"]
    ):
        raise AnalysisRecoveryError("parent readiness identity differs")
    for name, identity_field in (
        ("authorization", "authorization_identity"),
        ("launch", "launch_identity"),
    ):
        _, value = _artifact(parent[name], field=f"parent {name}")
        if _canonical_identity(
            value, field=f"parent {name}", identity_field=identity_field
        ) != parent[name]["identity"]:
            raise AnalysisRecoveryError(f"parent {name} identity differs")
    resources, failure = _audit_completed_training(
        plan=plan,
        parent_contract=parent_contract,
    )
    return parent_contract, resources, failure


def _validate_implementation(plan: Mapping[str, Any]) -> dict[str, Any]:
    implementation = plan["analysis_implementation"]
    source = reporter._inspect_analysis_source(
        str(plan["parent_attempt"]["training_source_commit"])
    )
    analysis_head = str(source["analysis_head"])
    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            str(implementation["minimum_commit"]),
            analysis_head,
        ],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise AnalysisRecoveryError("analysis implementation commit is absent")
    for name in ("publisher", "runner"):
        path = _inside_root(implementation[name]["path"], field=name)
        if _sha256_file(path) != implementation[name]["sha256"]:
            raise AnalysisRecoveryError(f"{name} implementation identity differs")
    return source


def _output_paths(plan: Mapping[str, Any]) -> dict[str, Path]:
    return {
        name: _inside_root(value, field=f"{name} output")
        for name, value in plan["outputs"].items()
    }


def _require_control_path(
    plan: Mapping[str, Any], *, name: str, observed: Path
) -> Path:
    expected = _inside_root(
        plan["control_files"][name], field=f"{name} control file"
    )
    if observed.resolve(strict=False) != expected:
        raise AnalysisRecoveryError(f"{name} control path differs")
    return expected


def build_readiness_body(
    *, plan_path: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    parent_contract, resources, failure = _validate_parent_artifacts(plan)
    source = _validate_implementation(plan)
    outputs = _output_paths(plan)
    occupied = [name for name, path in outputs.items() if path.exists()]
    if occupied:
        raise AnalysisRecoveryError(
            "analysis recovery outputs already exist: " + ", ".join(occupied)
        )
    schedule_audit = reporter._audit_paired_training_schedules(
        arms=_arm_by_cell(parent_contract),
        seeds=tuple(int(seed) for seed in parent_contract["pairing"]["seeds"]),
    )
    return {
        "schema_version": READINESS_SCHEMA,
        "state": "ready_for_product_authorization",
        "launch_authorized": False,
        "plan": {
            "path": _relative(plan_path),
            "sha256": _sha256_file(plan_path),
            "plan_identity": plan["plan_identity"],
        },
        "source": source,
        "parent_attempt": {
            "failure_identity": failure["failure_identity"],
            "run_id": failure["run_id"],
            "development_games_completed": 0,
            "result_published": False,
        },
        "completed_training": resources,
        "training_schedule_audit": schedule_audit,
        "resource_envelope": plan["resource_envelope"],
        "claim_boundary": plan["claim_boundary"],
    }


def prepare_readiness(*, plan_path: Path, readiness_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    plan = load_recovery_plan(plan_path)
    _require_control_path(plan, name="readiness", observed=readiness_path)
    body = build_readiness_body(plan_path=plan_path, plan=plan)
    readiness = {**body, "readiness_identity": canonical_sha256(body)}
    _publish_exclusive(readiness_path, readiness)
    return readiness


def _validate_readiness(
    *, plan_path: Path, plan: Mapping[str, Any], readiness_path: Path
) -> tuple[dict[str, Any], str]:
    readiness = _strict_json(readiness_path)
    if readiness.get("schema_version") != READINESS_SCHEMA:
        raise AnalysisRecoveryError("analysis recovery readiness schema differs")
    identity = _canonical_identity(
        readiness,
        field="analysis recovery readiness",
        identity_field="readiness_identity",
    )
    expected = build_readiness_body(plan_path=plan_path, plan=plan)
    if {key: value for key, value in readiness.items() if key != "readiness_identity"} != expected:
        raise AnalysisRecoveryError("analysis recovery readiness has drifted")
    return readiness, identity


def build_authorization(
    *,
    plan: Mapping[str, Any],
    readiness_identity: str,
    authorized_at_utc: str,
    decision_note: str,
) -> dict[str, Any]:
    if not decision_note.strip():
        raise AnalysisRecoveryError("authorization decision note is required")
    body = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "authorized_at_utc": authorized_at_utc,
        "authorized_by": "product-owner",
        "operator": "product-owner-delegated-agent",
        "decision_note": decision_note.strip(),
        "plan_identity": plan["plan_identity"],
        "readiness_identity": readiness_identity,
        "resource_envelope": plan["resource_envelope"],
        "permitted_operations": [
            "load-completed-candidate-checkpoints-read-only",
            "run-288-no-update-development-games-once",
            "publish-development-ledger-and-result-once",
        ],
        "prohibited_operations": [
            "training-retry-or-resume",
            "optimizer-update",
            "database-or-checkpoint-write",
            "automatic-retry-or-extension",
            "held-out-evaluation",
            "model-promotion-or-publication",
            "long-training-launch",
        ],
        "one_attempt": True,
        "expiry": "consumed when the one analysis recovery launch starts",
        "claim_boundary": plan["claim_boundary"],
    }
    return {**body, "authorization_identity": canonical_sha256(body)}


def record_authorization(
    *,
    plan_path: Path,
    readiness_path: Path,
    authorization_path: Path,
    expected_readiness_identity: str,
    decision_note: str,
) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    plan = load_recovery_plan(plan_path)
    _require_control_path(plan, name="readiness", observed=readiness_path)
    _require_control_path(
        plan, name="authorization", observed=authorization_path
    )
    _, readiness_identity = _validate_readiness(
        plan_path=plan_path,
        plan=plan,
        readiness_path=readiness_path.resolve(strict=True),
    )
    if readiness_identity != expected_readiness_identity:
        raise AnalysisRecoveryError("expected readiness identity differs")
    authorization = build_authorization(
        plan=plan,
        readiness_identity=readiness_identity,
        authorized_at_utc=utc_now_text(),
        decision_note=decision_note,
    )
    _publish_exclusive(authorization_path, authorization)
    return authorization


def _validate_authorization(
    *,
    path: Path,
    plan: Mapping[str, Any],
    readiness_identity: str,
) -> tuple[dict[str, Any], str]:
    authorization = _strict_json(path)
    identity = _canonical_identity(
        authorization,
        field="analysis recovery authorization",
        identity_field="authorization_identity",
    )
    expected = build_authorization(
        plan=plan,
        readiness_identity=readiness_identity,
        authorized_at_utc=str(authorization.get("authorized_at_utc", "")),
        decision_note=str(authorization.get("decision_note", "")),
    )
    if authorization != expected:
        raise AnalysisRecoveryError("analysis recovery authorization differs")
    return authorization, identity


def _reporter_command(plan: Mapping[str, Any]) -> list[str]:
    parent_contract = plan["parent_attempt"]["contract"]["path"]
    parent_readiness = plan["parent_attempt"]["readiness"]["path"]
    outputs = plan["outputs"]
    return [
        sys.executable,
        str(ROOT / "scripts/report_target_refresh_schedule_isolation_diagnostic.py"),
        "--contract",
        str(ROOT / parent_contract),
        "--readiness",
        str(ROOT / parent_readiness),
        "--paths-config",
        str(ROOT / plan["local_inputs"]["paths_config"]),
        "--ledger",
        str(ROOT / outputs["development_ledger"]),
        "--output",
        str(ROOT / outputs["development_result"]),
        "--device",
        "cpu",
    ]


def launch_once(
    *,
    plan_path: Path,
    readiness_path: Path,
    authorization_path: Path,
    expected_readiness_identity: str,
    run_id: str,
) -> dict[str, Any]:
    if not run_id.strip():
        raise AnalysisRecoveryError("analysis recovery run id is required")
    plan_path = plan_path.resolve(strict=True)
    plan = load_recovery_plan(plan_path)
    _require_control_path(plan, name="readiness", observed=readiness_path)
    _require_control_path(
        plan, name="authorization", observed=authorization_path
    )
    readiness, readiness_identity = _validate_readiness(
        plan_path=plan_path,
        plan=plan,
        readiness_path=readiness_path.resolve(strict=True),
    )
    if readiness_identity != expected_readiness_identity:
        raise AnalysisRecoveryError("expected readiness identity differs")
    _, authorization_identity = _validate_authorization(
        path=authorization_path.resolve(strict=True),
        plan=plan,
        readiness_identity=readiness_identity,
    )
    outputs = _output_paths(plan)
    marker_body = {
        "schema_version": LAUNCH_SCHEMA,
        "status": "started_once",
        "run_id": run_id,
        "started_at_utc": utc_now_text(),
        "plan_identity": plan["plan_identity"],
        "readiness_identity": readiness_identity,
        "authorization_identity": authorization_identity,
        "analysis_source_commit": readiness["source"]["analysis_head"],
        "training_source_commit": plan["parent_attempt"][
            "training_source_commit"
        ],
        "training_retry_or_resume": False,
    }
    marker = {**marker_body, "launch_identity": canonical_sha256(marker_body)}
    _publish_exclusive(outputs["launch"], marker)
    command = _reporter_command(plan)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            timeout=float(plan["resource_envelope"]["maximum_active_wall_hours"])
            * 3600,
        )
        elapsed_hours = (time.monotonic() - started) / 3600
        _publish_bytes_exclusive(outputs["stdout_log"], completed.stdout)
        _publish_bytes_exclusive(outputs["stderr_log"], completed.stderr)
        if completed.returncode != 0:
            raise AnalysisRecoveryError(
                f"analysis reporter exited with code {completed.returncode}"
            )
        result = _strict_json(outputs["development_result"])
        scope = result.get("scope", {})
        if (
            scope.get("no_update_development_games") != 288
            or scope.get("optimizer_updates") != 0
            or scope.get("training_games") != 0
            or scope.get("checkpoint_writes") != 0
            or scope.get("database_writes") != 0
        ):
            raise AnalysisRecoveryError("analysis result resource accounting differs")
        completion_body = {
            "schema_version": COMPLETION_SCHEMA,
            "status": "completed_once",
            "run_id": run_id,
            "completed_at_utc": utc_now_text(),
            "elapsed_hours": elapsed_hours,
            "launch_identity": marker["launch_identity"],
            "authorization_identity": authorization_identity,
            "plan_identity": plan["plan_identity"],
            "development_result": {
                "path": _relative(outputs["development_result"]),
                "sha256": _sha256_file(outputs["development_result"]),
                "result_identity": result["result_identity"],
                "classification": result["decision"]["classification"],
            },
            "development_ledger": {
                "path": _relative(outputs["development_ledger"]),
                "sha256": _sha256_file(outputs["development_ledger"]),
                "rows": 288,
            },
            "training_games": 0,
            "optimizer_updates": 0,
            "held_out_promotion_publication_or_long_run_authorized": False,
        }
        completion = {
            **completion_body,
            "completion_identity": canonical_sha256(completion_body),
        }
        _publish_exclusive(outputs["completion"], completion)
        return completion
    except Exception as exc:
        failure_body = {
            "schema_version": FAILURE_SCHEMA,
            "status": "failed_closed",
            "run_id": run_id,
            "failed_at_utc": utc_now_text(),
            "elapsed_hours": (time.monotonic() - started) / 3600,
            "launch_identity": marker["launch_identity"],
            "authorization_identity": authorization_identity,
            "plan_identity": plan["plan_identity"],
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "retry_or_recovery_authorized": False,
            "training_retry_or_resume": False,
            "held_out_promotion_publication_or_long_run_authorized": False,
        }
        failure = {
            **failure_body,
            "failure_identity": canonical_sha256(failure_body),
        }
        _publish_exclusive(outputs["failure"], failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--record-authorization", action="store_true")
    action.add_argument("--launch", choices=("once",))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--expected-readiness-identity")
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
                    args.expected_readiness_identity,
                    args.decision_note,
                    args.run_id,
                )
            ):
                raise AnalysisRecoveryError(
                    "preflight does not accept authorization or run fields"
                )
            result = prepare_readiness(
                plan_path=args.plan,
                readiness_path=args.readiness,
            )
        elif args.record_authorization:
            if not args.expected_readiness_identity or not args.decision_note:
                raise AnalysisRecoveryError(
                    "record-authorization requires readiness identity and decision"
                )
            if args.run_id is not None:
                raise AnalysisRecoveryError(
                    "record-authorization does not accept a run id"
                )
            result = record_authorization(
                plan_path=args.plan,
                readiness_path=args.readiness,
                authorization_path=args.authorization,
                expected_readiness_identity=args.expected_readiness_identity,
                decision_note=args.decision_note,
            )
        else:
            if not args.expected_readiness_identity or not args.run_id:
                raise AnalysisRecoveryError(
                    "launch requires readiness identity and run id"
                )
            if args.decision_note is not None:
                raise AnalysisRecoveryError(
                    "launch reads the recorded authorization decision"
                )
            result = launch_once(
                plan_path=args.plan,
                readiness_path=args.readiness,
                authorization_path=args.authorization,
                expected_readiness_identity=args.expected_readiness_identity,
                run_id=args.run_id,
            )
    except (
        AnalysisRecoveryError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
