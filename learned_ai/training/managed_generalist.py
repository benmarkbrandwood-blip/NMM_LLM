"""Bounded, product-authorized supervision for local Generalist training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from learned_ai.training.checkpoint_envelope import load_checkpoint
from learned_ai.training.generalist_run_manifest import (
    RUN_EVENT_LEDGER_NAME,
    utc_now_text,
)
from learned_ai.training.run_contract import (
    RunEvent,
    append_run_event,
    canonical_json_bytes,
    canonical_sha256,
    load_run_events,
)


MANAGED_PLAN_SCHEMA = "nmm.managed-generalist-plan.v1"
MANAGED_AUTHORIZATION_SCHEMA = "nmm.managed-authorization.v1"
CONTROLLER_LEDGER_NAME = "controller-events.jsonl"
CONTROLLER_LOCK_NAME = "controller.lock"

_DYNAMIC_TRAINER_OPTIONS = frozenset(
    {
        "--launch",
        "--preflight",
        "--run-id",
        "--parent-run-id",
        "--start-mode",
        "--resume",
        "--out-dir",
        "--segment-games",
        "--managed-plan",
        "--managed-authorization",
    }
)


class ManagedContractError(RuntimeError):
    """Raised when a managed plan, authorization, or segment is unsafe."""


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManagedContractError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    text = _require_text(value, field=field).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ManagedContractError(f"{field} must be a SHA-256")
    return text


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManagedContractError(f"{field} must be a positive integer")
    return value


def _require_positive_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManagedContractError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ManagedContractError(f"{field} must be finite and positive")
    return number


def _require_utc(value: Any, *, field: str) -> str:
    text = _require_text(value, field=field)
    if not text.endswith("Z"):
        raise ManagedContractError(f"{field} must be an RFC 3339 UTC timestamp")
    try:
        datetime.fromisoformat(f"{text[:-1]}+00:00")
    except ValueError as exc:
        raise ManagedContractError(
            f"{field} must be an RFC 3339 UTC timestamp"
        ) from exc
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ManagedContractError(f"cannot hash required file: {path}") from exc
    return digest.hexdigest()


def _strict_json(path: Path) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ManagedContractError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ManagedContractError(f"cannot read managed contract: {path}") from exc
    if not isinstance(value, Mapping):
        raise ManagedContractError(f"managed contract must be a JSON object: {path}")
    return value


def _publish_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"managed contract already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    payload = canonical_json_bytes(value) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"managed contract already exists: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class ManagedPlan:
    """Immutable technical and resource envelope for one local training goal."""

    plan_id: str
    created_at_utc: str
    objective: str
    experiment_id: str
    git_commit: str
    control_dir: str
    paths_config: str
    paths_config_sha256: str
    resume_config_sha256: str
    max_games: int
    segment_games: int
    max_wall_hours: float
    common_trainer_args: tuple[str, ...]
    allow_safe_exact_resume: bool
    publication_allowed: bool
    promotion_allowed: bool

    _FIELDS: ClassVar[set[str]] = {
        "schema_version",
        "plan_sha256",
        "plan_id",
        "created_at_utc",
        "objective",
        "experiment_id",
        "git_commit",
        "control_dir",
        "paths_config",
        "paths_config_sha256",
        "resume_config_sha256",
        "max_games",
        "segment_games",
        "max_wall_hours",
        "common_trainer_args",
        "allow_safe_exact_resume",
        "publication_allowed",
        "promotion_allowed",
    }

    def __post_init__(self) -> None:
        for field in ("plan_id", "objective", "experiment_id", "git_commit"):
            _require_text(getattr(self, field), field=field)
        _require_utc(self.created_at_utc, field="created_at_utc")
        for field in ("control_dir", "paths_config"):
            path = Path(_require_text(getattr(self, field), field=field))
            if not path.is_absolute():
                raise ManagedContractError(f"{field} must be an absolute path")
        _require_sha256(self.paths_config_sha256, field="paths_config_sha256")
        _require_sha256(self.resume_config_sha256, field="resume_config_sha256")
        _require_positive_int(self.max_games, field="max_games")
        _require_positive_int(self.segment_games, field="segment_games")
        if self.segment_games > self.max_games:
            raise ManagedContractError("segment_games must not exceed max_games")
        _require_positive_number(self.max_wall_hours, field="max_wall_hours")
        args = tuple(self.common_trainer_args)
        if not args or any(not isinstance(item, str) or not item for item in args):
            raise ManagedContractError("common_trainer_args must contain strings")
        forbidden = sorted(set(args) & _DYNAMIC_TRAINER_OPTIONS)
        if forbidden:
            raise ManagedContractError(
                "common_trainer_args contains controller-owned options: "
                + ", ".join(forbidden)
            )
        object.__setattr__(self, "common_trainer_args", args)
        for field in (
            "allow_safe_exact_resume",
            "publication_allowed",
            "promotion_allowed",
        ):
            if not isinstance(getattr(self, field), bool):
                raise ManagedContractError(f"{field} must be a boolean")
        if self.publication_allowed or self.promotion_allowed:
            raise ManagedContractError(
                "managed training plans cannot pre-authorize publication or promotion"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": MANAGED_PLAN_SCHEMA,
            "plan_id": self.plan_id,
            "created_at_utc": self.created_at_utc,
            "objective": self.objective,
            "experiment_id": self.experiment_id,
            "git_commit": self.git_commit,
            "control_dir": self.control_dir,
            "paths_config": self.paths_config,
            "paths_config_sha256": self.paths_config_sha256,
            "resume_config_sha256": self.resume_config_sha256,
            "max_games": self.max_games,
            "segment_games": self.segment_games,
            "max_wall_hours": self.max_wall_hours,
            "common_trainer_args": list(self.common_trainer_args),
            "allow_safe_exact_resume": self.allow_safe_exact_resume,
            "publication_allowed": self.publication_allowed,
            "promotion_allowed": self.promotion_allowed,
        }

    @property
    def plan_sha256(self) -> str:
        return canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "plan_sha256": self.plan_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ManagedPlan:
        actual = set(value)
        if actual != cls._FIELDS:
            raise ManagedContractError(
                "managed plan fields differ; "
                f"unknown={sorted(actual - cls._FIELDS)}, "
                f"missing={sorted(cls._FIELDS - actual)}"
            )
        if value["schema_version"] != MANAGED_PLAN_SCHEMA:
            raise ManagedContractError("unsupported managed plan schema")
        plan = cls(
            **{
                key: value[key]
                for key in cls._FIELDS - {"schema_version", "plan_sha256"}
            }
        )
        if value["plan_sha256"] != plan.plan_sha256:
            raise ManagedContractError("managed plan hash does not match its content")
        return plan


@dataclass(frozen=True)
class ManagedAuthorization:
    """A product decision bound to one exact immutable plan."""

    plan_id: str
    plan_sha256: str
    authorized_at_utc: str
    authorized_by: str
    decision_note: str
    allow_safe_exact_resume: bool

    _FIELDS: ClassVar[set[str]] = {
        "schema_version",
        "plan_id",
        "plan_sha256",
        "authorized_at_utc",
        "authorized_by",
        "decision_note",
        "allow_safe_exact_resume",
    }

    def __post_init__(self) -> None:
        _require_text(self.plan_id, field="plan_id")
        _require_sha256(self.plan_sha256, field="plan_sha256")
        _require_utc(self.authorized_at_utc, field="authorized_at_utc")
        _require_text(self.authorized_by, field="authorized_by")
        _require_text(self.decision_note, field="decision_note")
        if not isinstance(self.allow_safe_exact_resume, bool):
            raise ManagedContractError("allow_safe_exact_resume must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANAGED_AUTHORIZATION_SCHEMA,
            "plan_id": self.plan_id,
            "plan_sha256": self.plan_sha256,
            "authorized_at_utc": self.authorized_at_utc,
            "authorized_by": self.authorized_by,
            "decision_note": self.decision_note,
            "allow_safe_exact_resume": self.allow_safe_exact_resume,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ManagedAuthorization:
        actual = set(value)
        if actual != cls._FIELDS:
            raise ManagedContractError("managed authorization fields differ")
        if value["schema_version"] != MANAGED_AUTHORIZATION_SCHEMA:
            raise ManagedContractError("unsupported managed authorization schema")
        return cls(
            **{key: value[key] for key in cls._FIELDS - {"schema_version"}}
        )


def publish_managed_plan(path: str | Path, plan: ManagedPlan) -> None:
    """Publish an immutable plan and initialize its append-only ledger."""
    target = Path(path)
    if target.resolve(strict=False).parent != Path(plan.control_dir).resolve(
        strict=False
    ):
        raise ManagedContractError("plan path must be inside its control directory")
    _publish_exclusive(target, plan.to_dict())
    event = RunEvent(
        run_id=plan.plan_id,
        sequence=0,
        timestamp_utc=plan.created_at_utc,
        status="planned",
        event_type="managed_plan_published",
        reason_code=None,
        details={"plan_sha256": plan.plan_sha256},
        previous_event_sha256=None,
    )
    append_run_event(target.parent / CONTROLLER_LEDGER_NAME, event)


def load_managed_plan(path: str | Path) -> ManagedPlan:
    return ManagedPlan.from_dict(_strict_json(Path(path)))


def load_managed_authorization(path: str | Path) -> ManagedAuthorization:
    return ManagedAuthorization.from_dict(_strict_json(Path(path)))


def _append_controller_event(
    plan: ManagedPlan,
    *,
    status: str,
    event_type: str,
    reason_code: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> RunEvent:
    ledger = Path(plan.control_dir) / CONTROLLER_LEDGER_NAME
    existing = load_run_events(ledger)
    previous = existing[-1] if existing else None
    event = RunEvent(
        run_id=plan.plan_id,
        sequence=0 if previous is None else previous.sequence + 1,
        timestamp_utc=utc_now_text(),
        status=status,
        event_type=event_type,
        reason_code=reason_code,
        details=dict(details or {}),
        previous_event_sha256=(
            None if previous is None else previous.event_sha256
        ),
    )
    append_run_event(ledger, event)
    return event


def authorize_plan(
    plan_path: str | Path,
    authorization_path: str | Path,
    *,
    authorized_by: str,
    decision_note: str,
    authorized_at_utc: str | None = None,
) -> ManagedAuthorization:
    """Publish a separate, immutable product authorization for an exact plan."""
    plan = load_managed_plan(plan_path)
    authorization = ManagedAuthorization(
        plan_id=plan.plan_id,
        plan_sha256=plan.plan_sha256,
        authorized_at_utc=authorized_at_utc or utc_now_text(),
        authorized_by=authorized_by,
        decision_note=decision_note,
        allow_safe_exact_resume=plan.allow_safe_exact_resume,
    )
    _publish_exclusive(Path(authorization_path), authorization.to_dict())
    _append_controller_event(
        plan,
        status="planned",
        event_type="product_authorization_recorded",
        details={
            "plan_sha256": plan.plan_sha256,
            "authorized_by": authorized_by,
        },
    )
    return authorization


def _verify_authorization(
    plan: ManagedPlan, authorization_path: str | Path
) -> ManagedAuthorization:
    authorization = load_managed_authorization(authorization_path)
    if authorization.plan_id != plan.plan_id:
        raise ManagedContractError("authorization names a different plan")
    if authorization.plan_sha256 != plan.plan_sha256:
        raise ManagedContractError("authorization does not bind the current plan hash")
    if authorization.allow_safe_exact_resume != plan.allow_safe_exact_resume:
        raise ManagedContractError("authorization changes the exact-resume policy")
    return authorization


def _completed_segment_events(plan: ManagedPlan) -> list[RunEvent]:
    ledger = Path(plan.control_dir) / CONTROLLER_LEDGER_NAME
    return [
        event
        for event in load_run_events(ledger)
        if event.event_type == "managed_segment_completed"
    ]


def managed_status(
    plan_path: str | Path, authorization_path: str | Path
) -> dict[str, Any]:
    """Return a small product view plus nested technical evidence."""
    plan = load_managed_plan(plan_path)
    completed = _completed_segment_events(plan)
    completed_games = (
        int(completed[-1].details["completed_games"]) if completed else 0
    )
    elapsed_seconds = sum(
        float(event.details.get("elapsed_seconds", 0.0)) for event in completed
    )
    authorization_error: str | None = None
    try:
        _verify_authorization(plan, authorization_path)
        authorized = True
    except ManagedContractError as exc:
        authorized = False
        authorization_error = str(exc)

    ledger_events = load_run_events(Path(plan.control_dir) / CONTROLLER_LEDGER_NAME)
    last = ledger_events[-1]
    needs_product_decision = False
    decision = None
    if completed_games >= plan.max_games or last.event_type == "managed_plan_completed":
        state = "completed"
        summary = "The authorized training plan reached its game bound."
    elif last.reason_code == "wall_time_limit":
        state = "resource_limit_reached"
        summary = "The authorized wall-time envelope is exhausted."
        needs_product_decision = True
        decision = "Authorize a new resource envelope or end the objective."
    elif last.status in {"failed", "quarantined", "interrupted"}:
        state = "stopped_for_agent_review"
        summary = "The Agent must diagnose a technical safety stop."
        decision = None
    elif not authorized:
        state = "awaiting_product_authorization"
        summary = "The technical plan exists, but training is not authorized."
        needs_product_decision = True
        decision = "Approve or reject the stated objective and resource envelope."
    elif last.status == "running":
        state = "running"
        summary = "An authorized training segment is running."
    else:
        state = "ready_to_run"
        summary = "The plan is authorized and the next safe segment may run."

    return {
        "state": state,
        "summary": summary,
        "needs_product_decision": needs_product_decision,
        "product_decision": decision,
        "progress": {
            "completed_games": completed_games,
            "max_games": plan.max_games,
            "completed_segments": len(completed),
            "elapsed_hours": round(elapsed_seconds / 3600.0, 4),
            "max_wall_hours": plan.max_wall_hours,
        },
        "technical": {
            "plan_id": plan.plan_id,
            "plan_sha256": plan.plan_sha256,
            "git_commit": plan.git_commit,
            "authorization_error": authorization_error,
            "last_event": last.to_dict(),
            "publication_allowed": plan.publication_allowed,
            "promotion_allowed": plan.promotion_allowed,
        },
    }


def _segment_run_id(plan: ManagedPlan, segment_index: int) -> str:
    return f"{plan.plan_id}-segment-{segment_index:04d}"


def _segment_output_dir(plan: ManagedPlan, segment_index: int) -> Path:
    return Path(plan.control_dir) / "segments" / f"segment-{segment_index:04d}"


def build_segment_command(
    plan: ManagedPlan,
    *,
    plan_path: str | Path,
    authorization_path: str | Path,
    segment_index: int,
    previous_checkpoint: Path | None,
    previous_run_id: str | None,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build one shell-free launch command owned by the supervisor."""
    _require_positive_int(segment_index, field="segment_index")
    run_id = _segment_run_id(plan, segment_index)
    output_dir = _segment_output_dir(plan, segment_index)
    command = [
        python_executable,
        "scripts/train_s_gen_v2.py",
        "--launch",
        "long-run",
        "--run-id",
        run_id,
        "--out-dir",
        str(output_dir),
        "--segment-games",
        str(plan.segment_games),
        "--managed-plan",
        str(Path(plan_path).resolve(strict=False)),
        "--managed-authorization",
        str(Path(authorization_path).resolve(strict=False)),
        *plan.common_trainer_args,
    ]
    if segment_index == 1:
        if previous_checkpoint is not None or previous_run_id is not None:
            raise ManagedContractError("the first segment must start fresh")
        command.extend(("--start-mode", "fresh"))
    else:
        if not plan.allow_safe_exact_resume:
            raise ManagedContractError("the plan does not authorize exact resume")
        if previous_checkpoint is None or previous_run_id is None:
            raise ManagedContractError("continuation requires an exact checkpoint")
        command.extend(
            (
                "--start-mode",
                "exact-resume",
                "--resume",
                str(previous_checkpoint.resolve(strict=False)),
                "--parent-run-id",
                previous_run_id,
            )
        )
    return command


def verify_managed_launch(
    plan_path: str | Path,
    authorization_path: str | Path,
    *,
    git_commit: str,
    resume_config_sha256: str,
    out_dir: str | Path,
    run_id: str,
    segment_games: int | None,
    start_mode: str,
    resume: str,
    parent_run_id: str | None,
    experiment_id: str,
) -> ManagedPlan:
    """Fail closed unless trainer arguments match one authorized segment."""
    plan = load_managed_plan(plan_path)
    _verify_authorization(plan, authorization_path)
    if git_commit != plan.git_commit:
        raise ManagedContractError("managed plan Git commit does not match")
    if resume_config_sha256 != plan.resume_config_sha256:
        raise ManagedContractError("managed plan training semantics do not match")
    if experiment_id != plan.experiment_id:
        raise ManagedContractError("managed experiment identity does not match")
    if segment_games != plan.segment_games:
        raise ManagedContractError("managed segment size does not match the plan")
    prefix = f"{plan.plan_id}-segment-"
    if not isinstance(run_id, str):
        raise ManagedContractError("managed run ID is required")
    if not run_id.startswith(prefix) or not run_id[len(prefix):].isdigit():
        raise ManagedContractError("managed run ID is outside the plan")
    segment_index = int(run_id[len(prefix):])
    expected_output = _segment_output_dir(plan, segment_index).resolve(strict=False)
    if Path(out_dir).resolve(strict=False) != expected_output:
        raise ManagedContractError("managed output directory is outside the plan")
    if segment_index == 1:
        if start_mode != "fresh" or resume or parent_run_id is not None:
            raise ManagedContractError("the first managed segment must start fresh")
    else:
        expected_previous_run = _segment_run_id(plan, segment_index - 1)
        expected_resume = (
            _segment_output_dir(plan, segment_index - 1) / "latest.pt"
        ).resolve(strict=False)
        if start_mode != "exact-resume":
            raise ManagedContractError("managed continuation must use exact resume")
        if Path(resume).resolve(strict=False) != expected_resume:
            raise ManagedContractError("managed continuation checkpoint differs")
        if parent_run_id != expected_previous_run:
            raise ManagedContractError("managed continuation parent differs")
    paths_config = Path(plan.paths_config)
    if _file_sha256(paths_config) != plan.paths_config_sha256:
        raise ManagedContractError("managed paths configuration has changed")
    return plan


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_state(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _inspect_completed_segment(
    plan: ManagedPlan,
    *,
    segment_index: int,
    previous_completed_games: int,
) -> tuple[int, Path]:
    output_dir = _segment_output_dir(plan, segment_index)
    events = load_run_events(output_dir / RUN_EVENT_LEDGER_NAME)
    if not events or events[-1].status != "completed":
        raise ManagedContractError("segment run ledger is not completed")
    checkpoint = output_dir / "latest.pt"
    envelope = load_checkpoint(checkpoint, map_location="cpu")
    descriptor = envelope.descriptor
    if descriptor.run_id != _segment_run_id(plan, segment_index):
        raise ManagedContractError("segment checkpoint run identity differs")
    if descriptor.experiment_id != plan.experiment_id:
        raise ManagedContractError("segment checkpoint experiment differs")
    if descriptor.config_sha256 != plan.resume_config_sha256:
        raise ManagedContractError("segment checkpoint semantics differ")
    completed_games = int(envelope.payload.trainer_state["game_count"])
    expected_games = min(
        previous_completed_games + plan.segment_games,
        plan.max_games,
    )
    if completed_games != expected_games:
        raise ManagedContractError(
            "segment checkpoint game count does not match the bounded schedule"
        )
    return completed_games, checkpoint


def run_next_segment(
    plan_path: str | Path,
    authorization_path: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Run exactly one authorized segment, then verify its durable evidence."""
    plan_path = Path(plan_path).resolve(strict=False)
    authorization_path = Path(authorization_path).resolve(strict=False)
    plan = load_managed_plan(plan_path)
    _verify_authorization(plan, authorization_path)

    completed_events = _completed_segment_events(plan)
    previous_completed_games = (
        int(completed_events[-1].details["completed_games"])
        if completed_events
        else 0
    )
    if previous_completed_games >= plan.max_games:
        return managed_status(plan_path, authorization_path)
    elapsed_seconds = sum(
        float(event.details.get("elapsed_seconds", 0.0))
        for event in completed_events
    )
    remaining_seconds = plan.max_wall_hours * 3600.0 - elapsed_seconds
    if remaining_seconds <= 0:
        _append_controller_event(
            plan,
            status="interrupted",
            event_type="managed_resource_limit_reached",
            reason_code="wall_time_limit",
        )
        raise ManagedContractError("managed wall-time resource limit is exhausted")

    root = _repository_root()
    commit, dirty = _git_state(root)
    if dirty:
        raise ManagedContractError("managed training requires a clean Git worktree")
    if commit != plan.git_commit:
        raise ManagedContractError("managed training Git commit has changed")
    if _file_sha256(Path(plan.paths_config)) != plan.paths_config_sha256:
        raise ManagedContractError("managed paths configuration has changed")

    segment_index = len(completed_events) + 1
    previous_checkpoint = None
    previous_run_id = None
    if completed_events:
        previous_checkpoint = Path(
            str(completed_events[-1].details["checkpoint"])
        )
        previous_run_id = str(completed_events[-1].details["run_id"])
        _inspect_completed_segment(
            plan,
            segment_index=segment_index - 1,
            previous_completed_games=(
                int(completed_events[-2].details["completed_games"])
                if len(completed_events) > 1
                else 0
            ),
        )
    output_dir = _segment_output_dir(plan, segment_index)
    if output_dir.exists():
        raise ManagedContractError("next managed segment output already exists")
    command = build_segment_command(
        plan,
        plan_path=plan_path,
        authorization_path=authorization_path,
        segment_index=segment_index,
        previous_checkpoint=previous_checkpoint,
        previous_run_id=previous_run_id,
        python_executable=python_executable,
    )

    lock = Path(plan.control_dir) / CONTROLLER_LOCK_NAME
    lock.parent.mkdir(parents=True, exist_ok=True)
    owns_lock = False
    try:
        try:
            with lock.open("x", encoding="ascii") as handle:
                handle.write(f"pid={os.getpid()}\n")
            owns_lock = True
        except FileExistsError as exc:
            raise ManagedContractError(
                "another supervisor owns the managed control lock"
            ) from exc
        _append_controller_event(
            plan,
            status="running",
            event_type="managed_segment_started",
            details={
                "segment_index": segment_index,
                "run_id": _segment_run_id(plan, segment_index),
            },
        )
        started = time.monotonic()
        try:
            result = runner(
                command,
                cwd=root,
                check=False,
                timeout=remaining_seconds,
            )
        except KeyboardInterrupt:
            elapsed = time.monotonic() - started
            _append_controller_event(
                plan,
                status="interrupted",
                event_type="managed_supervisor_interrupted",
                reason_code="operator_interrupt",
                details={
                    "segment_index": segment_index,
                    "elapsed_seconds": elapsed,
                },
            )
            raise
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            _append_controller_event(
                plan,
                status="interrupted",
                event_type="managed_segment_timed_out",
                reason_code="wall_time_limit",
                details={
                    "segment_index": segment_index,
                    "elapsed_seconds": elapsed,
                },
            )
            raise ManagedContractError("managed segment reached the wall-time limit") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            elapsed = time.monotonic() - started
            _append_controller_event(
                plan,
                status="failed",
                event_type="managed_supervisor_failed",
                reason_code="runner_exception",
                details={
                    "segment_index": segment_index,
                    "elapsed_seconds": elapsed,
                    "exception_type": type(exc).__name__,
                },
            )
            raise ManagedContractError("managed supervisor could not run trainer") from exc
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            _append_controller_event(
                plan,
                status="failed",
                event_type="managed_segment_failed",
                reason_code="trainer_exit_nonzero",
                details={
                    "segment_index": segment_index,
                    "returncode": result.returncode,
                    "elapsed_seconds": elapsed,
                },
            )
            raise ManagedContractError(
                f"managed trainer exited with code {result.returncode}"
            )
        try:
            completed_games, checkpoint = _inspect_completed_segment(
                plan,
                segment_index=segment_index,
                previous_completed_games=previous_completed_games,
            )
        except Exception as exc:
            _append_controller_event(
                plan,
                status="quarantined",
                event_type="managed_segment_quarantined",
                reason_code="evidence_validation_failed",
                details={
                    "segment_index": segment_index,
                    "elapsed_seconds": elapsed,
                    "exception_type": type(exc).__name__,
                },
            )
            raise ManagedContractError("managed segment evidence is invalid") from exc
        _append_controller_event(
            plan,
            status="completed",
            event_type="managed_segment_completed",
            details={
                "segment_index": segment_index,
                "run_id": _segment_run_id(plan, segment_index),
                "completed_games": completed_games,
                "checkpoint": str(checkpoint.resolve(strict=False)),
                "elapsed_seconds": elapsed,
            },
        )
        if completed_games >= plan.max_games:
            _append_controller_event(
                plan,
                status="completed",
                event_type="managed_plan_completed",
                details={"completed_games": completed_games},
            )
        return managed_status(plan_path, authorization_path)
    finally:
        if owns_lock and lock.exists():
            lock.unlink()


def run_authorized_plan(
    plan_path: str | Path,
    authorization_path: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Continue safe exact-resume segments until completion or a hard stop."""
    while True:
        status = run_next_segment(
            plan_path,
            authorization_path,
            runner=runner,
            python_executable=python_executable,
        )
        if status["state"] == "completed":
            return status
        if status["state"] != "ready_to_run":
            raise ManagedContractError(
                f"managed plan stopped in state {status['state']}"
            )
