#!/usr/bin/env python3
"""Prepare and run one zero-training mature-refresh analysis recovery."""

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
from learned_ai.validation.target_refresh_mature_fork_diagnostic import (  # noqa: E402
    load_contract,
)
from scripts import report_target_refresh_mature_fork_diagnostic as reporter  # noqa: E402
from scripts.run_target_refresh_mature_fork_sequence import (  # noqa: E402
    _resource_audit,
)


PLAN_SCHEMA = "nmm.target-refresh-mature-fork-analysis-recovery-plan.v1"
READINESS_SCHEMA = "nmm.target-refresh-mature-fork-analysis-recovery-readiness.v1"
AUTHORIZATION_SCHEMA = (
    "nmm.target-refresh-mature-fork-analysis-recovery-authorization.v1"
)
LAUNCH_SCHEMA = "nmm.target-refresh-mature-fork-analysis-recovery-launch.v1"
COMPLETION_SCHEMA = (
    "nmm.target-refresh-mature-fork-analysis-recovery-completion.v1"
)
FAILURE_SCHEMA = "nmm.target-refresh-mature-fork-analysis-recovery-failure.v1"
ARTIFACT_SCHEMA = "nmm.target-refresh-mature-fork-completed-artifacts.v1"
DEFAULT_PLAN = ROOT / (
    "docs/experiments/"
    "sanmill-target-refresh-mature-fork-analysis-recovery-v1.json"
)
OUTPUT_ROOT = ROOT / (
    "out/target-refresh-mature-fork-diagnostic-v1-"
    "attempt-002-analysis-recovery-v1"
)
DEFAULT_READINESS = OUTPUT_ROOT / "readiness.json"
DEFAULT_AUTHORIZATION = OUTPUT_ROOT / "authorization.json"


class MatureRefreshAnalysisRecoveryError(RuntimeError):
    """Raised when analysis-only recovery cannot be proven safe."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise MatureRefreshAnalysisRecoveryError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                MatureRefreshAnalysisRecoveryError(
                    f"non-finite JSON value in {path.name}: {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MatureRefreshAnalysisRecoveryError(
            f"cannot read JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise MatureRefreshAnalysisRecoveryError(
            f"JSON root is not an object: {path}"
        )
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
        raise MatureRefreshAnalysisRecoveryError(
            f"{field} is outside the repository"
        ) from exc
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
        raise MatureRefreshAnalysisRecoveryError(
            f"output already exists: {target}"
        ) from exc


def _publish_bytes_exclusive(path: Path, value: bytes) -> None:
    target = _inside_root(path, field="log output")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise MatureRefreshAnalysisRecoveryError(
            f"output already exists: {target}"
        ) from exc


def _canonical_identity(
    value: Mapping[str, Any], *, field: str, identity_field: str
) -> str:
    identity = value.get(identity_field)
    body = dict(value)
    body.pop(identity_field, None)
    if identity != canonical_sha256(body):
        raise MatureRefreshAnalysisRecoveryError(
            f"{field} canonical identity differs"
        )
    return str(identity)


def _artifact(
    record: Mapping[str, Any], *, field: str
) -> tuple[Path, dict[str, Any]]:
    path = _inside_root(str(record.get("path", "")), field=field)
    if not path.is_file() or _sha256_file(path) != record.get("sha256"):
        raise MatureRefreshAnalysisRecoveryError(f"{field} identity differs")
    return path, _read_json_object(path)


def load_recovery_plan(path: Path) -> dict[str, Any]:
    plan = _read_json_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise MatureRefreshAnalysisRecoveryError("analysis recovery schema differs")
    identity = plan.get("plan_identity")
    body = dict(plan)
    body.pop("plan_identity", None)
    if identity != canonical_sha256(body):
        raise MatureRefreshAnalysisRecoveryError("analysis recovery identity differs")
    expected_resources = {
        "candidate_models_loaded": True,
        "checkpoint_writes": 0,
        "database_writes": 0,
        "maximum_active_wall_hours": 3.5,
        "no_update_development_games": 288,
        "optimizer_updates": 0,
        "training_games": 0,
    }
    if plan.get("resource_envelope") != expected_resources:
        raise MatureRefreshAnalysisRecoveryError(
            "analysis recovery resource envelope differs"
        )
    if plan.get("status") != "designed_unlaunched_needs_authorization":
        raise MatureRefreshAnalysisRecoveryError("analysis recovery status differs")
    return plan


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MatureRefreshAnalysisRecoveryError(f"completed artifact is absent: {path}")
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def build_completed_artifact_manifest(
    *, contract: Mapping[str, Any], failure: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind every completed branch input used by the recovery reporter."""
    arms = {
        (int(arm["seed"]), str(arm["condition"])): arm
        for arm in contract["arms"]
    }
    completed = sorted(
        failure["completed_steps"], key=lambda item: int(item["launch_order"])
    )
    rows = []
    for step in completed:
        key = (int(step["seed"]), str(step["condition"]))
        arm = arms.get(key)
        if arm is None or step.get("kind") != "run-arm":
            raise MatureRefreshAnalysisRecoveryError(
                "completed arm identity differs"
            )
        result = step["result"]
        control = _inside_root(str(arm["control_dir"]), field="arm control")
        segment = control / "segments" / "segment-0001"
        files = {
            "plan": _file_record(control / "plan.json"),
            "authorization": _file_record(control / "authorization.json"),
            "controller_events": _file_record(control / "controller-events.jsonl"),
            "initial_branch": _file_record(
                control / "initial-mature-target-refresh-fork.pt"
            ),
            "train_log": _file_record(segment / "train_log.jsonl"),
            "update_log": _file_record(segment / "update_log.jsonl"),
            "policy_health": _file_record(segment / "policy-health.json"),
            "transition_4096": _file_record(segment / "transition-00004096.pt"),
            "transition_8192": _file_record(segment / "transition-00008192.pt"),
            "latest": _file_record(segment / "latest.pt"),
        }
        if (
            reporter.load_managed_plan(control / "plan.json").plan_sha256
            != result["plan_sha256"]
            or files["authorization"]["sha256"]
            != result["authorization_sha256"]
            or files["controller_events"]["sha256"]
            != result["controller_events_sha256"]
        ):
            raise MatureRefreshAnalysisRecoveryError(
                "completed controller artifact differs"
            )
        database = _inside_root(str(arm["specialist_db"]), field="arm database")
        sidecars = [
            path
            for path in (
                Path(str(database) + "-wal"),
                Path(str(database) + "-shm"),
                Path(str(database) + "-journal"),
            )
            if path.exists()
        ]
        if sidecars:
            raise MatureRefreshAnalysisRecoveryError(
                f"completed database has sidecars: {arm['arm_id']}"
            )
        rows.append(
            {
                "launch_order": int(step["launch_order"]),
                "seed": key[0],
                "condition": key[1],
                "arm_id": str(arm["arm_id"]),
                "files": files,
                "specialist_db": _file_record(database),
            }
        )
    body = {"schema_version": ARTIFACT_SCHEMA, "arms": rows}
    return {**body, "artifact_identity": canonical_sha256(body)}


def _validate_parent_artifacts(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent = plan["parent_attempt"]
    contract_path, _ = _artifact(parent["contract"], field="parent contract")
    contract = load_contract(contract_path)
    if contract["plan_identity"] != parent["contract"]["identity"]:
        raise MatureRefreshAnalysisRecoveryError("parent plan identity differs")
    _, readiness = _artifact(parent["readiness"], field="parent readiness")
    readiness_identity = reporter._validate_readiness(readiness, contract=contract)
    if readiness_identity != parent["readiness"]["identity"]:
        raise MatureRefreshAnalysisRecoveryError("parent readiness identity differs")
    values: dict[str, dict[str, Any]] = {}
    for name, identity_field in (
        ("authorization", "authorization_identity"),
        ("launch", "launch_identity"),
        ("failure", "failure_identity"),
    ):
        _, value = _artifact(parent[name], field=f"parent {name}")
        identity = _canonical_identity(
            value,
            field=f"parent {name}",
            identity_field=identity_field,
        )
        if identity != parent[name]["identity"]:
            raise MatureRefreshAnalysisRecoveryError(
                f"parent {name} identity differs"
            )
        values[name] = value
    authorization = values["authorization"]
    launch = values["launch"]
    failure = values["failure"]
    expected = parent
    if (
        authorization.get("plan_identity") != contract["plan_identity"]
        or authorization.get("readiness_identity") != readiness_identity
        or launch.get("authorization_identity")
        != authorization["authorization_identity"]
        or launch.get("source_commit") != expected["training_source_commit"]
        or failure.get("status") != "failed_closed"
        or failure.get("plan_identity") != contract["plan_identity"]
        or failure.get("readiness_identity") != readiness_identity
        or failure.get("authorization_identity")
        != authorization["authorization_identity"]
        or failure.get("launch_identity") != launch["launch_identity"]
        or failure.get("retry_or_recovery_authorized") is not False
        or failure.get("failure", {}).get("type")
        != "MatureTargetRefreshReportError"
        or "JSON is not canonical" not in str(
            failure.get("failure", {}).get("message", "")
        )
        or "dev-v4-phase-covered-corpus-v1.json" not in str(
            failure.get("failure", {}).get("message", "")
        )
    ):
        raise MatureRefreshAnalysisRecoveryError("parent failure semantics differ")
    completed = failure.get("completed_steps")
    if not isinstance(completed, list):
        raise MatureRefreshAnalysisRecoveryError("parent completed steps are absent")
    resources = _resource_audit(completed, contract=contract)
    if resources != plan["completed_training"]:
        raise MatureRefreshAnalysisRecoveryError(
            "completed training resource audit differs"
        )
    manifest = build_completed_artifact_manifest(contract=contract, failure=failure)
    if manifest["artifact_identity"] != plan["completed_artifact_identity"]:
        raise MatureRefreshAnalysisRecoveryError(
            "completed training artifact identity differs"
        )
    for name in ("completion", "ledger", "result"):
        path = _inside_root(contract["result_outputs"][name], field=f"parent {name}")
        if path.exists():
            raise MatureRefreshAnalysisRecoveryError(
                f"parent {name} unexpectedly exists"
            )
    training_audit = reporter._audit_training(
        arms=reporter._arm_map(contract), contract=contract
    )
    return contract, resources, manifest, training_audit


def _validate_implementation(plan: Mapping[str, Any]) -> dict[str, Any]:
    implementation = plan["analysis_implementation"]
    source = reporter._inspect_analysis_source(
        str(plan["parent_attempt"]["training_source_commit"])
    )
    analysis_head = str(source["analysis_head"])
    ancestor = subprocess.run(
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
    if ancestor.returncode != 0:
        raise MatureRefreshAnalysisRecoveryError(
            "analysis implementation commit is absent"
        )
    for name in ("publisher", "runner"):
        path = _inside_root(implementation[name]["path"], field=name)
        if _sha256_file(path) != implementation[name]["sha256"]:
            raise MatureRefreshAnalysisRecoveryError(
                f"{name} implementation identity differs"
            )
    return source


def _validate_local_dependencies(plan: Mapping[str, Any]) -> dict[str, Any]:
    config_path = _inside_root(
        plan["local_inputs"]["paths_config"], field="paths config"
    ).resolve(strict=True)
    settings = reporter._read_json_object(config_path)
    human_value = Path(str(settings["human_db_path"]))
    human_path = (
        human_value if human_value.is_absolute() else ROOT / human_value
    ).resolve(strict=True)
    malom_value = Path(str(settings["malom_db_path"]))
    malom_path = (
        malom_value if malom_value.is_absolute() else ROOT / malom_value
    ).resolve(strict=True)
    data = plan["rules_and_data"]
    human = reporter._probe_human_db(human_path)
    if (
        human.get("error")
        or human.get("identity") != data["human_db_identity"]
        or human.get("malom_columns_policy") != "masked_historical_labels"
    ):
        raise MatureRefreshAnalysisRecoveryError("HumanDB identity differs")
    manifest_path = _inside_root(
        plan["local_inputs"]["malom_manifest"], field="Malom manifest"
    ).resolve(strict=True)
    manifest = reporter.load_dataset_manifest(manifest_path)
    std = next(
        (
            component
            for component in manifest.components
            if component.relative_path == "std.secval"
        ),
        None,
    )
    if (
        manifest.manifest_sha256 != data["malom_manifest_identity"]
        or std is None
        or _sha256_file(malom_path / "std.secval") != std.sha256
    ):
        raise MatureRefreshAnalysisRecoveryError("Malom identity differs")
    malom = reporter.ExternalSolvedDB(str(malom_path), strict=True)
    try:
        if not malom.is_available():
            raise MatureRefreshAnalysisRecoveryError("Malom is unavailable")
    finally:
        malom.close()
    installation = reporter.load_local_installation(config_path)
    if (
        installation.commit != data["sanmill_commit"]
        or installation.tree != data["sanmill_tree"]
        or installation.binary_sha256 != data["sanmill_binary_sha256"]
    ):
        raise MatureRefreshAnalysisRecoveryError("Sanmill identity differs")
    return {
        "paths_config_sha256": _sha256_file(config_path),
        "human_db_identity": human["identity"],
        "historical_human_malom_labels": "masked",
        "malom_manifest_identity": manifest.manifest_sha256,
        "malom_std_secval_sha256": std.sha256,
        "sanmill_commit": installation.commit,
        "sanmill_tree": installation.tree,
        "sanmill_binary_sha256": installation.binary_sha256,
    }


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
        raise MatureRefreshAnalysisRecoveryError(f"{name} control path differs")
    return expected


def build_readiness_body(
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    allow_authorization: bool = False,
) -> dict[str, Any]:
    contract, resources, manifest, training_audit = _validate_parent_artifacts(plan)
    source = _validate_implementation(plan)
    dependencies = _validate_local_dependencies(plan)
    outputs = _output_paths(plan)
    occupied = [name for name, path in outputs.items() if path.exists()]
    authorization_path = _inside_root(
        plan["control_files"]["authorization"], field="authorization"
    )
    if authorization_path.exists() and not allow_authorization:
        occupied.append("authorization")
    if occupied:
        raise MatureRefreshAnalysisRecoveryError(
            "analysis recovery outputs already exist: " + ", ".join(occupied)
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
            "plan_identity": contract["plan_identity"],
            "failure_identity": plan["parent_attempt"]["failure"]["identity"],
            "run_id": plan["parent_attempt"]["run_id"],
            "development_games_completed": 0,
            "result_published": False,
        },
        "completed_training": resources,
        "completed_artifacts": {
            "identity": manifest["artifact_identity"],
            "arm_count": len(manifest["arms"]),
        },
        "training_audit_identity": canonical_sha256(training_audit),
        "local_dependencies": dependencies,
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
    readiness = _read_json_object(readiness_path)
    if readiness.get("schema_version") != READINESS_SCHEMA:
        raise MatureRefreshAnalysisRecoveryError("recovery readiness schema differs")
    identity = _canonical_identity(
        readiness,
        field="analysis recovery readiness",
        identity_field="readiness_identity",
    )
    expected = build_readiness_body(
        plan_path=plan_path,
        plan=plan,
        allow_authorization=True,
    )
    observed = {
        key: value for key, value in readiness.items() if key != "readiness_identity"
    }
    if observed != expected:
        raise MatureRefreshAnalysisRecoveryError("recovery readiness has drifted")
    return readiness, identity


def build_authorization(
    *,
    plan: Mapping[str, Any],
    readiness_identity: str,
    authorized_at_utc: str,
    decision_note: str,
) -> dict[str, Any]:
    if not decision_note.strip():
        raise MatureRefreshAnalysisRecoveryError("authorization note is required")
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
            "load-six-completed-candidate-pairs-read-only",
            "run-288-cpu-no-update-development-games-once",
            "write-isolated-analysis-ledger-result-and-control-evidence-once",
        ],
        "prohibited_operations": [
            "training-retry-resume-or-recovery",
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
    _require_control_path(plan, name="authorization", observed=authorization_path)
    _, identity = _validate_readiness(
        plan_path=plan_path,
        plan=plan,
        readiness_path=readiness_path.resolve(strict=True),
    )
    if identity != expected_readiness_identity:
        raise MatureRefreshAnalysisRecoveryError(
            "expected readiness identity differs"
        )
    authorization = build_authorization(
        plan=plan,
        readiness_identity=identity,
        authorized_at_utc=utc_now_text(),
        decision_note=decision_note,
    )
    _publish_exclusive(authorization_path, authorization)
    return authorization


def _validate_authorization(
    *, path: Path, plan: Mapping[str, Any], readiness_identity: str
) -> tuple[dict[str, Any], str]:
    authorization = _read_json_object(path)
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
        raise MatureRefreshAnalysisRecoveryError(
            "analysis recovery authorization differs"
        )
    return authorization, identity


def _reporter_command(plan: Mapping[str, Any]) -> list[str]:
    parent = plan["parent_attempt"]
    outputs = plan["outputs"]
    return [
        sys.executable,
        str(ROOT / "scripts/report_target_refresh_mature_fork_diagnostic.py"),
        "--contract",
        str(ROOT / parent["contract"]["path"]),
        "--readiness",
        str(ROOT / parent["readiness"]["path"]),
        "--paths-config",
        str(ROOT / plan["local_inputs"]["paths_config"]),
        "--malom-manifest",
        str(ROOT / plan["local_inputs"]["malom_manifest"]),
        "--ledger",
        str(ROOT / outputs["development_ledger"]),
        "--output",
        str(ROOT / outputs["development_result"]),
        "--device",
        "cpu",
        "--allow-published-analysis-descendant",
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
        raise MatureRefreshAnalysisRecoveryError("analysis run id is required")
    plan_path = plan_path.resolve(strict=True)
    plan = load_recovery_plan(plan_path)
    _require_control_path(plan, name="readiness", observed=readiness_path)
    _require_control_path(plan, name="authorization", observed=authorization_path)
    readiness, readiness_identity = _validate_readiness(
        plan_path=plan_path,
        plan=plan,
        readiness_path=readiness_path.resolve(strict=True),
    )
    if readiness_identity != expected_readiness_identity:
        raise MatureRefreshAnalysisRecoveryError(
            "expected readiness identity differs"
        )
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
        "training_source_commit": plan["parent_attempt"]["training_source_commit"],
        "training_retry_or_resume": False,
    }
    marker = {**marker_body, "launch_identity": canonical_sha256(marker_body)}
    _publish_exclusive(outputs["launch"], marker)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            _reporter_command(plan),
            cwd=ROOT,
            check=False,
            capture_output=True,
            timeout=float(plan["resource_envelope"]["maximum_active_wall_hours"])
            * 3600,
        )
        elapsed_hours = (time.monotonic() - started) / 3600.0
        _publish_bytes_exclusive(outputs["stdout_log"], completed.stdout)
        _publish_bytes_exclusive(outputs["stderr_log"], completed.stderr)
        if completed.returncode != 0:
            raise MatureRefreshAnalysisRecoveryError(
                f"analysis reporter exited with code {completed.returncode}"
            )
        result = _read_json_object(outputs["development_result"])
        result_identity = _canonical_identity(
            result,
            field="analysis result",
            identity_field="result_identity",
        )
        scope = result.get("scope", {})
        if (
            scope.get("no_update_development_games") != 288
            or scope.get("optimizer_updates") != 0
            or scope.get("training_games") != 0
            or scope.get("checkpoint_writes") != 0
            or scope.get("database_writes") != 0
        ):
            raise MatureRefreshAnalysisRecoveryError(
                "analysis result resource accounting differs"
            )
        ledger_rows = sum(
            1
            for line in outputs["development_ledger"].read_bytes().splitlines()
            if line
        )
        if ledger_rows != 288:
            raise MatureRefreshAnalysisRecoveryError(
                "analysis ledger row count differs"
            )
        ledger_sha256 = _sha256_file(outputs["development_ledger"])
        ledger_record = result.get("identities", {}).get(
            "direct_crossplay_ledger", {}
        )
        if (
            elapsed_hours > plan["resource_envelope"]["maximum_active_wall_hours"]
            or result.get("identities", {})
            .get("contract", {})
            .get("plan_identity")
            != plan["parent_attempt"]["contract"]["identity"]
            or result.get("identities", {}).get("source", {}).get("analysis_head")
            != readiness["source"]["analysis_head"]
            or ledger_record.get("sha256") != ledger_sha256
            or ledger_record.get("rows") != ledger_rows
        ):
            raise MatureRefreshAnalysisRecoveryError(
                "analysis result identity binding differs"
            )
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
                "result_identity": result_identity,
                "classification": result["decision"]["classification"],
            },
            "development_ledger": {
                "path": _relative(outputs["development_ledger"]),
                "sha256": ledger_sha256,
                "rows": ledger_rows,
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
            "elapsed_hours": (time.monotonic() - started) / 3600.0,
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
                raise MatureRefreshAnalysisRecoveryError(
                    "preflight does not accept authorization or run fields"
                )
            result = prepare_readiness(
                plan_path=args.plan,
                readiness_path=args.readiness,
            )
        elif args.record_authorization:
            if not args.expected_readiness_identity or not args.decision_note:
                raise MatureRefreshAnalysisRecoveryError(
                    "record-authorization requires readiness identity and decision"
                )
            if args.run_id is not None:
                raise MatureRefreshAnalysisRecoveryError(
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
                raise MatureRefreshAnalysisRecoveryError(
                    "launch requires readiness identity and run id"
                )
            if args.decision_note is not None:
                raise MatureRefreshAnalysisRecoveryError(
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
        MatureRefreshAnalysisRecoveryError,
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
