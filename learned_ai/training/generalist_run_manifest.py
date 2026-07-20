"""Manifest construction and lifecycle publication for Generalist v2 runs."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from learned_ai.training.run_contract import (
    AssetManifestRef,
    ContractValidationError,
    RunEvent,
    RunManifest,
    append_run_event,
    canonical_sha256,
    load_run_events,
    publish_run_manifest,
)


RUN_MANIFEST_NAME = "run-manifest.json"
RUN_EVENT_LEDGER_NAME = "run-events.jsonl"


def utc_now_text() -> str:
    """Return a canonical RFC 3339 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _portable_path(path: str, root: Path) -> str:
    candidate = Path(path).resolve(strict=False)
    try:
        return candidate.relative_to(root.resolve()).as_posix()
    except ValueError:
        return f"external:{candidate.name}"


def collect_runtime_environment() -> dict[str, Any]:
    """Collect runtime identity after all read-only preflight gates pass."""
    cuda_available = torch.cuda.is_available()
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
    }
    if cuda_available:
        environment["cuda_device_count"] = torch.cuda.device_count()
        environment["cuda_device_name"] = torch.cuda.get_device_name(0)
    return environment


def _asset_refs(
    report: Mapping[str, Any], *, start_mode: str
) -> tuple[AssetManifestRef, ...]:
    checks = report["checks"]
    malom = checks["malom"]
    specialist = checks["specialist_db"]
    human = checks["human_db"]
    for name, value in (
        ("malom", malom),
        ("specialist_db", specialist),
        ("human_db", human),
    ):
        if not value.get("identity"):
            raise ContractValidationError(
                f"preflight check {name} does not provide an asset identity"
            )
    assets = [
        AssetManifestRef(
            logical_name="malom_tablebase",
            role="training_oracle",
            identity=malom["identity"],
            schema_version="malom-ultra-strong-sec2",
            trust_level="sector-corrected-v1",
            intended_use="lookahead_termination_and_corrected_label_queries",
        ),
        AssetManifestRef(
            logical_name="specialist_db",
            role="training_database",
            identity=specialist["identity"],
            schema_version=specialist["label_version"],
            trust_level=specialist["trust"],
            intended_use="corrected_labels_and_empirical_results",
        ),
        AssetManifestRef(
            logical_name="human_db",
            role="empirical_human_database",
            identity=human["identity"],
            schema_version=human.get("label_version") or "unversioned-malom-columns",
            trust_level=human["trust"],
            intended_use="human_frequencies_and_empirical_outcomes_only",
        ),
    ]
    checkpoint = checks.get("checkpoint")
    if checkpoint is not None:
        exact_resume = start_mode == "exact-resume"
        assets.append(
            AssetManifestRef(
                logical_name="source_checkpoint",
                role="weights_import",
                identity=checkpoint["identity"],
                schema_version=checkpoint["format"],
                trust_level=(
                    "integrity_verified_exact_resume"
                    if exact_resume
                    else "lineage_labeled_weights_only"
                ),
                intended_use=(
                    "complete_training_state_continuation"
                    if exact_resume
                    else "model_weights_only"
                ),
            )
        )
    return tuple(assets)


def build_generalist_run_manifest(
    args: Any,
    *,
    report: Mapping[str, Any],
    root: Path,
    command: Sequence[str],
    run_id: str,
    experiment_id: str,
    parent_run_id: str | None = None,
    created_at_utc: str | None = None,
    environment: Mapping[str, Any] | None = None,
) -> RunManifest:
    """Build the immutable manifest for a preflight-approved fresh run."""
    expected_verdict = (
        "ready_for_smoke" if report["mode"] == "smoke" else "ready_for_long_run"
    )
    if report["verdict"] != expected_verdict:
        raise ContractValidationError(
            f"cannot build a run manifest from verdict {report['verdict']!r}"
        )
    config = report["resolved_config"]
    config_sha256 = canonical_sha256(config)
    if config_sha256 != report["config_sha256"]:
        raise ContractValidationError("preflight configuration hash is inconsistent")
    git = report["git"]
    return RunManifest(
        run_id=run_id,
        experiment_id=experiment_id,
        parent_run_id=parent_run_id,
        status="preflight_passed",
        created_at_utc=created_at_utc or utc_now_text(),
        git_commit=git["commit"],
        git_dirty=git["dirty"],
        git_diff_sha256=git["diff_sha256"],
        command=tuple(command),
        resolved_config=config,
        config_sha256=config_sha256,
        environment=dict(environment or collect_runtime_environment()),
        seeds={"run": args.seed},
        assets=_asset_refs(report, start_mode=args.start_mode),
        components={
            "sentinel": not args.no_sentinel,
            "value_net": not args.no_value_net,
            "gap_net": not args.no_gap_net,
            "ppo": bool(args.ppo),
        },
        outputs={
            "run_directory": _portable_path(args.out_dir, root),
            "specialist_db": _portable_path(args.specialist_db, root),
            "manifest": RUN_MANIFEST_NAME,
            "event_ledger": RUN_EVENT_LEDGER_NAME,
        },
        checkpoint_policy={
            "start_mode": args.start_mode,
            "automatic_resume": False,
            "historical_checkpoints": "weights_only",
            "roles": ["latest", "best_train", "candidate", "accepted"],
        },
        claim_boundaries=(
            "corrected v4 infrastructure evidence",
            "not playing-strength evidence",
            "not a completed v5 implementation",
        ),
    )


def publish_initial_run_contract(output_dir: str | Path, manifest: RunManifest) -> None:
    """Atomically create a new run directory with manifest and initial event."""
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"run output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.contract.{uuid4().hex}.tmp"
    staging.mkdir()
    try:
        publish_run_manifest(staging / RUN_MANIFEST_NAME, manifest)
        initial = RunEvent(
            run_id=manifest.run_id,
            sequence=0,
            timestamp_utc=manifest.created_at_utc,
            status="preflight_passed",
            event_type="preflight_passed",
            reason_code=None,
            details={
                "manifest_sha256": manifest.manifest_sha256,
                "config_sha256": manifest.config_sha256,
            },
            previous_event_sha256=None,
        )
        append_run_event(staging / RUN_EVENT_LEDGER_NAME, initial)
        if target.exists():
            raise FileExistsError(f"run output already exists: {target}")
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def append_run_lifecycle_event(
    output_dir: str | Path,
    *,
    run_id: str,
    status: str,
    event_type: str,
    reason_code: str | None = None,
    details: Mapping[str, Any] | None = None,
    timestamp_utc: str | None = None,
) -> RunEvent:
    """Append the next lifecycle event to an existing run contract."""
    ledger = Path(output_dir) / RUN_EVENT_LEDGER_NAME
    events = load_run_events(ledger)
    if not events:
        raise ContractValidationError("run event ledger has no initial event")
    previous = events[-1]
    if previous.run_id != run_id:
        raise ContractValidationError("run ID does not match the event ledger")
    event = RunEvent(
        run_id=run_id,
        sequence=previous.sequence + 1,
        timestamp_utc=timestamp_utc or utc_now_text(),
        status=status,
        event_type=event_type,
        reason_code=reason_code,
        details=dict(details or {}),
        previous_event_sha256=previous.event_sha256,
    )
    append_run_event(ledger, event)
    return event


def command_for_manifest(argv: Sequence[str]) -> tuple[str, ...]:
    """Return the exact Python entry command recorded in the run manifest."""
    return (sys.executable, "scripts/train_s_gen_v2.py", *argv)
