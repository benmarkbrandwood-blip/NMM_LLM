"""Bounded, product-authorized supervision for local Generalist training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from learned_ai.training.checkpoint_envelope import (
    CheckpointPayload,
    load_checkpoint,
    save_checkpoint,
)
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
from learned_ai.training.training_identity import (
    TrainingIdentityError,
    experiment_digest,
    load_trainer_ruleset,
)


MANAGED_PLAN_SCHEMA = "nmm.managed-generalist-plan.v1"
MANAGED_AUTHORIZATION_SCHEMA = "nmm.managed-authorization.v1"
POLICY_HEALTH_GATE_SCHEMA = "nmm.managed-policy-health-gate.v1"
TECHNICAL_RECOVERY_EVIDENCE_SCHEMA = "nmm.managed-technical-recovery.v1"
CONTROLLER_LEDGER_NAME = "controller-events.jsonl"
CONTROLLER_LOCK_NAME = "controller.lock"

_RECOVERY_REASON_CODES = frozenset(
    {"host_reboot", "verified_implementation_repair"}
)

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
        "--segment-stop-game",
        "--managed-plan",
        "--managed-authorization",
    }
)


def _optional_positive_trainer_arg(
    args: tuple[str, ...], option: str
) -> int | None:
    """Return one optional positive integer from immutable trainer args."""
    matches = [index for index, value in enumerate(args) if value == option]
    if not matches:
        return None
    if len(matches) != 1 or matches[0] + 1 >= len(args):
        raise ManagedContractError(f"{option} must appear exactly once with a value")
    raw = args[matches[0] + 1]
    try:
        value = int(raw)
    except ValueError as exc:
        raise ManagedContractError(f"{option} must be a positive integer") from exc
    if value <= 0:
        raise ManagedContractError(f"{option} must be a positive integer")
    return value


def _optimizer_update_bound(plan: ManagedPlan) -> int | None:
    return _optional_positive_trainer_arg(
        plan.common_trainer_args, "--optimizer-update-bound"
    )


def _plan_completion_reached(
    plan: ManagedPlan,
    *,
    completed_games: int,
    completed_updates: int | None,
) -> bool:
    update_bound = _optimizer_update_bound(plan)
    if update_bound is None:
        return completed_games >= plan.game_bound
    return completed_updates is not None and completed_updates >= update_bound


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


def _require_finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManagedContractError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ManagedContractError(f"{field} must be finite")
    return number


def _require_probability(value: Any, *, field: str) -> float:
    number = _require_finite_number(value, field=field)
    if not 0.0 <= number <= 1.0:
        raise ManagedContractError(f"{field} must be between zero and one")
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
class PolicyHealthGate:
    """Immutable fixed-state acceptance gate for one managed plan."""

    corpus_path: str
    corpus_sha256: str
    audit_script_path: str
    audit_script_sha256: str
    exact_critical_states: int
    required_direct_preserving_rate: float
    min_candidate_preserving_rate: float
    min_candidate_logit_margin: float
    device: str = "auto"

    _FIELDS: ClassVar[set[str]] = {
        "schema_version",
        "corpus_path",
        "corpus_sha256",
        "audit_script_path",
        "audit_script_sha256",
        "exact_critical_states",
        "required_direct_preserving_rate",
        "min_candidate_preserving_rate",
        "min_candidate_logit_margin",
        "device",
    }

    def __post_init__(self) -> None:
        for field in ("corpus_path", "audit_script_path"):
            path = Path(_require_text(getattr(self, field), field=field))
            if not path.is_absolute():
                raise ManagedContractError(f"{field} must be an absolute path")
        _require_sha256(self.corpus_sha256, field="corpus_sha256")
        _require_sha256(self.audit_script_sha256, field="audit_script_sha256")
        _require_positive_int(
            self.exact_critical_states,
            field="exact_critical_states",
        )
        _require_probability(
            self.required_direct_preserving_rate,
            field="required_direct_preserving_rate",
        )
        _require_probability(
            self.min_candidate_preserving_rate,
            field="min_candidate_preserving_rate",
        )
        _require_finite_number(
            self.min_candidate_logit_margin,
            field="min_candidate_logit_margin",
        )
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ManagedContractError("policy-health device is unsupported")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_HEALTH_GATE_SCHEMA,
            "corpus_path": self.corpus_path,
            "corpus_sha256": self.corpus_sha256,
            "audit_script_path": self.audit_script_path,
            "audit_script_sha256": self.audit_script_sha256,
            "exact_critical_states": self.exact_critical_states,
            "required_direct_preserving_rate": (
                self.required_direct_preserving_rate
            ),
            "min_candidate_preserving_rate": self.min_candidate_preserving_rate,
            "min_candidate_logit_margin": self.min_candidate_logit_margin,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PolicyHealthGate:
        actual = set(value)
        if actual != cls._FIELDS:
            raise ManagedContractError(
                "policy-health gate fields differ; "
                f"unknown={sorted(actual - cls._FIELDS)}, "
                f"missing={sorted(cls._FIELDS - actual)}"
            )
        if value["schema_version"] != POLICY_HEALTH_GATE_SCHEMA:
            raise ManagedContractError("unsupported policy-health gate schema")
        return cls(
            **{
                key: value[key]
                for key in cls._FIELDS - {"schema_version"}
            }
        )


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
    policy_health: PolicyHealthGate | None = None
    completion_game_bound: int | None = None

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
        "policy_health",
        "completion_game_bound",
    }

    _OPTIONAL_FIELDS: ClassVar[set[str]] = {
        "policy_health",
        "completion_game_bound",
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
        if self.completion_game_bound is not None:
            _require_positive_int(
                self.completion_game_bound,
                field="completion_game_bound",
            )
            if self.completion_game_bound > self.max_games:
                raise ManagedContractError(
                    "completion_game_bound must not exceed max_games"
                )
        if self.segment_games > self.game_bound:
            raise ManagedContractError(
                "segment_games must not exceed the completion game bound"
            )
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
        _optional_positive_trainer_arg(args, "--optimizer-update-bound")
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
        if self.policy_health is not None and not isinstance(
            self.policy_health, PolicyHealthGate
        ):
            raise ManagedContractError(
                "policy_health must be a PolicyHealthGate or null"
            )

    @property
    def game_bound(self) -> int:
        """Return the authorized completion ceiling, separate from scheduling."""
        return self.completion_game_bound or self.max_games

    def _payload(self) -> dict[str, Any]:
        payload = {
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
        if self.policy_health is not None:
            payload["policy_health"] = self.policy_health.to_dict()
        if self.completion_game_bound is not None:
            payload["completion_game_bound"] = self.completion_game_bound
        return payload

    @property
    def plan_sha256(self) -> str:
        return canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "plan_sha256": self.plan_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ManagedPlan:
        actual = set(value)
        required = cls._FIELDS - cls._OPTIONAL_FIELDS
        if not required <= actual or actual - cls._FIELDS:
            raise ManagedContractError(
                "managed plan fields differ; "
                f"unknown={sorted(actual - cls._FIELDS)}, "
                f"missing={sorted(required - actual)}"
            )
        if value["schema_version"] != MANAGED_PLAN_SCHEMA:
            raise ManagedContractError("unsupported managed plan schema")
        fields = {
            key: value[key]
            for key in required - {"schema_version", "plan_sha256"}
        }
        if "policy_health" in value:
            raw_health = value["policy_health"]
            if not isinstance(raw_health, Mapping):
                raise ManagedContractError(
                    "managed plan policy_health must be an object"
                )
            fields["policy_health"] = PolicyHealthGate.from_dict(raw_health)
        if "completion_game_bound" in value:
            fields["completion_game_bound"] = value["completion_game_bound"]
        plan = cls(**fields)
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
    completed_updates = (
        int(completed[-1].details["completed_updates"])
        if completed and completed[-1].details.get("completed_updates") is not None
        else None
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
    if _plan_completion_reached(
        plan,
        completed_games=completed_games,
        completed_updates=completed_updates,
    ) or last.event_type == "managed_plan_completed":
        state = "completed"
        summary = (
            "The authorized training plan reached its optimizer-update bound."
            if _optimizer_update_bound(plan) is not None
            else "The authorized training plan reached its game bound."
        )
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

    progress: dict[str, Any] = {
        "completed_games": completed_games,
        "max_games": plan.game_bound,
        "schedule_max_games": plan.max_games,
        "completed_segments": len(completed),
        "elapsed_hours": round(elapsed_seconds / 3600.0, 4),
        "max_wall_hours": plan.max_wall_hours,
    }
    if _optimizer_update_bound(plan) is not None:
        progress.update(
            {
                "completed_optimizer_updates": completed_updates,
                "optimizer_update_bound": _optimizer_update_bound(plan),
            }
        )
    return {
        "state": state,
        "summary": summary,
        "needs_product_decision": needs_product_decision,
        "product_decision": decision,
        "progress": progress,
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


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _parse_lock_pid(lock: Path) -> int | None:
    if not lock.is_file():
        return None
    text = lock.read_text(encoding="ascii").strip()
    if not text.startswith("pid="):
        return None
    try:
        return int(text.removeprefix("pid=").strip())
    except ValueError:
        return None


def _clear_stale_controller_lock(plan: ManagedPlan) -> bool:
    """Remove a controller.lock only when its recorded PID is dead."""
    lock = Path(plan.control_dir) / CONTROLLER_LOCK_NAME
    if not lock.exists():
        return False
    pid = _parse_lock_pid(lock)
    if pid is None:
        raise ManagedContractError("managed control lock is malformed")
    if _pid_is_running(pid):
        raise ManagedContractError(
            "another supervisor owns the managed control lock"
        )
    lock.unlink()
    return True


def _specialist_db_path_for_plan(plan: ManagedPlan) -> Path:
    args = plan.common_trainer_args
    if "--specialist-db" in args:
        return Path(args[args.index("--specialist-db") + 1]).resolve(strict=True)
    paths = json.loads(Path(plan.paths_config).read_text(encoding="utf-8"))
    raw = paths.get("specialist_db_path")
    if not isinstance(raw, str) or not raw.strip():
        raise ManagedContractError("plan paths config lacks specialist_db_path")
    return Path(raw).resolve(strict=True)


def _live_specialist_identity(path: Path) -> dict[str, Any]:
    from learned_ai.data.specialist_db import SpecialistDB

    db = SpecialistDB(str(path))
    try:
        db.require_trusted_malom_labels()
        return db.checkpoint_identity()
    finally:
        db.close()


def _recovery_experiment_digest(
    plan: ManagedPlan,
    *,
    runtime_commit: str,
    asset_identities: Mapping[str, str],
) -> str:
    """Rebind a repaired continuation to its exact clean source commit."""
    immutable_assets = {
        "malom_tablebase": str(asset_identities.get("malom_tablebase", "")),
        "human_db": str(asset_identities.get("human_db", "")),
    }
    for name in ("opening_forcing_sources", "sanmill_training_runtime"):
        if name in asset_identities:
            immutable_assets[name] = str(asset_identities[name])
    if any(not value for value in immutable_assets.values()):
        raise ManagedContractError(
            "recovery checkpoint lacks an immutable experiment asset"
        )

    args = plan.common_trainer_args
    if "--ruleset-manifest" in args:
        ruleset_path = Path(args[args.index("--ruleset-manifest") + 1])
    else:
        ruleset_path = (
            _repository_root() / "data" / "rulesets" / "nmm-training-core@2.json"
        )
    try:
        ruleset = load_trainer_ruleset(ruleset_path.resolve(strict=True))
    except (OSError, TrainingIdentityError) as exc:
        raise ManagedContractError(
            "cannot verify the recovery ruleset identity"
        ) from exc
    return experiment_digest(
        experiment_id=plan.experiment_id,
        git_commit=runtime_commit,
        resume_config_sha256=plan.resume_config_sha256,
        immutable_assets=immutable_assets,
        ruleset=ruleset,
    )


def _write_recovery_checkpoint(
    source: Path,
    destination: Path,
    *,
    specialist_identity: Mapping[str, Any],
    plan: ManagedPlan | None = None,
    runtime_commit: str | None = None,
    recovery_reason: str = "host-reboot",
) -> Path:
    """Publish a recovery envelope whose SpecialistDB identity matches live state."""
    if (plan is None) != (runtime_commit is None):
        raise ManagedContractError(
            "recovery plan and runtime commit must be supplied together"
        )
    envelope = load_checkpoint(source, map_location="cpu")
    payload_dict = envelope.payload.to_dict()
    data_state = dict(payload_dict["data_state"])
    mutable_assets = dict(data_state["mutable_assets"])
    mutable_assets["specialist_db"] = dict(specialist_identity)
    data_state["mutable_assets"] = mutable_assets
    payload_dict["data_state"] = data_state
    assets = dict(envelope.descriptor.asset_identities)
    assets["specialist_db"] = str(specialist_identity["sha256"])
    implementation = dict(envelope.descriptor.implementation)
    if plan is not None and runtime_commit is not None:
        implementation["experiment_digest"] = _recovery_experiment_digest(
            plan,
            runtime_commit=runtime_commit,
            asset_identities=assets,
        )
    descriptor = replace(
        envelope.descriptor,
        checkpoint_id=(
            f"{envelope.descriptor.checkpoint_id}:{recovery_reason}-recovery"
        ),
        save_reason=f"interrupted-{recovery_reason}-recovery",
        created_at_utc=utc_now_text(),
        asset_identities=assets,
        implementation=implementation,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ManagedContractError("recovery checkpoint already exists")
    save_checkpoint(destination, descriptor, CheckpointPayload.from_dict(payload_dict), previous_copies=0)
    return destination.resolve(strict=True)


def _load_technical_recovery_evidence(
    path: str | Path,
    *,
    plan: ManagedPlan,
    failed_event: RunEvent,
    runtime_commit: str,
) -> dict[str, str]:
    """Verify a tracked diagnosis before reopening a failed segment."""
    source = Path(path).resolve(strict=True)
    root = _repository_root().resolve(strict=True)
    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise ManagedContractError(
            "technical recovery evidence must be inside the repository"
        ) from exc
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        raise ManagedContractError(
            "technical recovery evidence must be tracked at the current commit"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ManagedContractError(
                    f"technical recovery evidence repeats key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ManagedContractError(
            "technical recovery evidence is not valid JSON"
        ) from exc
    expected = {
        "schema_version",
        "plan_sha256",
        "failed_event_sha256",
        "failed_segment_index",
        "failure_code",
        "source_commit",
        "tested_repair_commit",
        "reproduction",
        "verification",
        "claim_boundary",
    }
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ManagedContractError(
            "technical recovery evidence fields differ; "
            f"unknown={sorted(actual - expected)}, "
            f"missing={sorted(expected - actual)}"
        )
    if value["schema_version"] != TECHNICAL_RECOVERY_EVIDENCE_SCHEMA:
        raise ManagedContractError("unsupported technical recovery evidence schema")
    if value["plan_sha256"] != plan.plan_sha256:
        raise ManagedContractError("technical recovery evidence names another plan")
    if value["failed_event_sha256"] != failed_event.event_sha256:
        raise ManagedContractError("technical recovery evidence names another failure")
    segment_index = value["failed_segment_index"]
    if (
        isinstance(segment_index, bool)
        or not isinstance(segment_index, int)
        or segment_index != int(failed_event.details["segment_index"])
    ):
        raise ManagedContractError("technical recovery segment identity differs")
    if value["source_commit"] != plan.git_commit:
        raise ManagedContractError("technical recovery source commit differs")
    repair_commit = _require_text(
        value["tested_repair_commit"], field="tested_repair_commit"
    )
    if not _git_is_ancestor(root, plan.git_commit, repair_commit):
        raise ManagedContractError("tested repair does not descend from the plan")
    if not _git_is_ancestor(root, repair_commit, runtime_commit):
        raise ManagedContractError("runtime commit does not contain the tested repair")
    failure_code = _require_text(value["failure_code"], field="failure_code")
    if not isinstance(value["reproduction"], Mapping) or not value["reproduction"]:
        raise ManagedContractError("technical recovery reproduction is empty")
    verification = value["verification"]
    if (
        not isinstance(verification, list)
        or not verification
        or any(not isinstance(item, str) or not item for item in verification)
    ):
        raise ManagedContractError("technical recovery verification is empty")
    _require_text(value["claim_boundary"], field="claim_boundary")
    return {
        "path": str(source),
        "sha256": _file_sha256(source),
        "failure_code": failure_code,
        "tested_repair_commit": repair_commit,
    }


def _pending_recovery_for_segment(
    plan: ManagedPlan, segment_index: int
) -> dict[str, Any] | None:
    """Return verified recovery details still pending for one segment index."""
    ledger = Path(plan.control_dir) / CONTROLLER_LEDGER_NAME
    if not ledger.exists():
        return None
    for event in reversed(load_run_events(ledger)):
        details = dict(event.details)
        if int(details.get("segment_index", -1)) != segment_index:
            continue
        if event.event_type == "managed_segment_completed":
            return None
        if (
            event.event_type == "managed_segment_interrupted"
            and event.reason_code in _RECOVERY_REASON_CODES
            and details.get("recovery_checkpoint")
        ):
            return details
    return None


def _plan_used_recovery(plan: ManagedPlan) -> bool:
    ledger = Path(plan.control_dir) / CONTROLLER_LEDGER_NAME
    if not ledger.exists():
        return False
    return any(
        event.event_type == "managed_segment_interrupted"
        and event.reason_code in _RECOVERY_REASON_CODES
        for event in load_run_events(ledger)
    )


def recover_interrupted_segment(
    plan_path: str | Path,
    authorization_path: str | Path,
    *,
    technical_evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """Quarantine an interrupted segment and publish a recovery checkpoint.

    Incomplete mid-segment work is not accepted as completed evidence. The next
    supervised segment exact-resumes from the interrupted latest.pt after the
    live SpecialistDB identity is rebound into a dedicated recovery envelope.
    Trainer failures additionally require tracked, commit-bound repair evidence.
    """
    plan_path = Path(plan_path).resolve(strict=False)
    authorization_path = Path(authorization_path).resolve(strict=False)
    plan = load_managed_plan(plan_path)
    _verify_authorization(plan, authorization_path)
    runtime_commit = _assert_managed_git_state(
        plan, allow_recovery_descendant=True
    )
    if _file_sha256(Path(plan.paths_config)) != plan.paths_config_sha256:
        raise ManagedContractError("managed paths configuration has changed")

    completed_events = _completed_segment_events(plan)
    pending_index = len(completed_events) + 1
    existing = _pending_recovery_for_segment(plan, pending_index)
    if existing is not None:
        return {
            "state": "stopped_for_agent_review",
            "summary": "Interrupted-segment recovery is already prepared.",
            "recovery": existing,
            "status": managed_status(plan_path, authorization_path),
        }

    _clear_stale_controller_lock(plan)

    ledger_events = load_run_events(Path(plan.control_dir) / CONTROLLER_LEDGER_NAME)
    if not ledger_events:
        raise ManagedContractError("managed controller ledger is empty")
    last = ledger_events[-1]
    technical_repair: dict[str, str] | None = None
    if last.event_type == "managed_segment_started" and last.status == "running":
        if technical_evidence_path is not None:
            raise ManagedContractError(
                "running-segment recovery does not accept repair evidence"
            )
        recovery_reason = "host_reboot"
        quarantine_kind = "interrupted"
        checkpoint_reason = "host-reboot"
    elif (
        last.event_type == "managed_segment_failed"
        and last.status == "failed"
        and last.reason_code == "trainer_exit_nonzero"
    ):
        if technical_evidence_path is None:
            raise ManagedContractError(
                "failed-segment recovery requires tracked repair evidence"
            )
        technical_repair = _load_technical_recovery_evidence(
            technical_evidence_path,
            plan=plan,
            failed_event=last,
            runtime_commit=runtime_commit,
        )
        recovery_reason = "verified_implementation_repair"
        quarantine_kind = "failed"
        checkpoint_reason = "verified-implementation-repair"
    else:
        raise ManagedContractError(
            "managed recovery requires host loss or a verified trainer repair"
        )

    segment_index = int(last.details["segment_index"])
    completed_events = _completed_segment_events(plan)
    if segment_index != len(completed_events) + 1:
        raise ManagedContractError("interrupted segment index does not follow completions")
    previous_completed_games = (
        int(completed_events[-1].details["completed_games"]) if completed_events else 0
    )
    expected_games = min(
        previous_completed_games + plan.segment_games,
        plan.game_bound,
    )
    incomplete = _segment_output_dir(plan, segment_index)
    preflight_only_failure = technical_repair is not None and not incomplete.exists()
    if not incomplete.exists():
        if technical_repair is None:
            _append_controller_event(
                plan,
                status="interrupted",
                event_type="managed_segment_interrupted",
                reason_code=recovery_reason,
                details={
                    "segment_index": segment_index,
                    "incomplete_output": None,
                    "recovery_checkpoint": None,
                },
            )
            return managed_status(plan_path, authorization_path)
        if not completed_events:
            raise ManagedContractError(
                "preflight-only failure has no completed parent checkpoint"
            )
        source_checkpoint = Path(
            str(completed_events[-1].details["checkpoint"])
        ).resolve(strict=True)
        expected_checkpoint_run = str(completed_events[-1].details["run_id"])
        checkpoint_origin = "previous_completed_boundary"
    else:
        latest = incomplete / "latest.pt"
        checkpoint_origin = "interrupted_latest"
        if latest.is_file():
            source_checkpoint = latest
            expected_checkpoint_run = _segment_run_id(plan, segment_index)
        else:
            if technical_repair is not None:
                raise ManagedContractError(
                    "failed segment has no latest.pt; refusing technical recovery"
                )
            if not completed_events:
                raise ManagedContractError(
                    "first-segment interruption has no checkpoint to recover"
                )
            source_checkpoint = Path(
                str(completed_events[-1].details["checkpoint"])
            ).resolve(strict=True)
            expected_checkpoint_run = str(completed_events[-1].details["run_id"])
            checkpoint_origin = "previous_completed_boundary"
    envelope = load_checkpoint(source_checkpoint, map_location="cpu")
    game_count = int(envelope.payload.trainer_state["game_count"])
    if envelope.descriptor.run_id != expected_checkpoint_run:
        raise ManagedContractError("interrupted checkpoint run identity differs")
    if checkpoint_origin == "interrupted_latest":
        if not previous_completed_games < game_count < expected_games:
            raise ManagedContractError(
                "interrupted checkpoint game_count is outside the recoverable window"
            )
    elif game_count != previous_completed_games:
        raise ManagedContractError(
            "completed parent checkpoint does not match its segment boundary"
        )

    stamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    quarantine: Path | None = None
    if incomplete.exists():
        quarantine = (
            Path(plan.control_dir)
            / "quarantine"
            / f"segment-{segment_index:04d}.{quarantine_kind}-{stamp}"
        )
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        if quarantine.exists():
            raise ManagedContractError("quarantine target already exists")
        incomplete.rename(quarantine)

    specialist_path = _specialist_db_path_for_plan(plan)
    backup_dir = Path(plan.control_dir) / "quarantine" / f"specialist-db-{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for suffix in ("", "-wal", "-shm"):
        src = Path(str(specialist_path) + suffix) if suffix else specialist_path
        if src.exists():
            shutil.copy2(src, backup_dir / src.name)
    specialist_identity = _live_specialist_identity(specialist_path)
    if preflight_only_failure:
        checkpoint_specialist = envelope.payload.data_state["mutable_assets"][
            "specialist_db"
        ]["sha256"]
        if specialist_identity["sha256"] != checkpoint_specialist:
            raise ManagedContractError(
                "preflight-only failure changed the SpecialistDB"
            )
    if checkpoint_origin == "interrupted_latest":
        if quarantine is None:
            raise ManagedContractError("interrupted checkpoint quarantine is missing")
        source_checkpoint = quarantine / "latest.pt"
    recovery_checkpoint = _write_recovery_checkpoint(
        source_checkpoint,
        Path(plan.control_dir) / "recovery" / f"segment-{segment_index:04d}.pt",
        specialist_identity=specialist_identity,
        plan=plan,
        runtime_commit=runtime_commit,
        recovery_reason=checkpoint_reason,
    )
    details = {
        "segment_index": segment_index,
        "incomplete_output": (
            None if quarantine is None else str(quarantine.resolve(strict=False))
        ),
        "recovery_checkpoint": str(recovery_checkpoint),
        "resume_game_count": game_count,
        "expected_segment_end": expected_games,
        "parent_run_id": (
            None
            if not completed_events
            else str(completed_events[-1].details["run_id"])
        ),
        "specialist_db_backup": str(backup_dir.resolve(strict=False)),
        "specialist_db_sha256": str(specialist_identity["sha256"]),
        "runtime_commit": runtime_commit,
        "checkpoint_recovery_reason": checkpoint_reason,
        "checkpoint_origin": checkpoint_origin,
    }
    if technical_repair is not None:
        details["technical_repair"] = technical_repair
    _append_controller_event(
        plan,
        status="interrupted",
        event_type="managed_segment_interrupted",
        reason_code=recovery_reason,
        details=details,
    )
    status = managed_status(plan_path, authorization_path)
    return {
        "state": status["state"],
        "summary": (
            "Verified recovery checkpoint is ready for exact-resume continuation."
        ),
        "recovery": details,
        "status": status,
    }


def recover_failed_segment(
    plan_path: str | Path,
    authorization_path: str | Path,
    *,
    technical_evidence_path: str | Path,
) -> dict[str, Any]:
    """Prepare exact resume only after a tracked implementation repair."""
    return recover_interrupted_segment(
        plan_path,
        authorization_path,
        technical_evidence_path=technical_evidence_path,
    )


def build_segment_command(
    plan: ManagedPlan,
    *,
    plan_path: str | Path,
    authorization_path: str | Path,
    segment_index: int,
    previous_checkpoint: Path | None,
    previous_run_id: str | None,
    previous_completed_games: int,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build one shell-free launch command owned by the supervisor."""
    _require_positive_int(segment_index, field="segment_index")
    if previous_completed_games < 0:
        raise ManagedContractError("previous_completed_games must be non-negative")
    expected_stop = min(
        previous_completed_games + plan.segment_games,
        plan.game_bound,
    )
    if expected_stop <= previous_completed_games:
        raise ManagedContractError("segment schedule has no remaining games")
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
        "--segment-stop-game",
        str(expected_stop),
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
    segment_stop_game: int | None = None,
) -> ManagedPlan:
    """Fail closed unless trainer arguments match one authorized segment."""
    plan = load_managed_plan(plan_path)
    _verify_authorization(plan, authorization_path)
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
    completed_events = _completed_segment_events(plan)
    if len(completed_events) != segment_index - 1:
        raise ManagedContractError("managed segment index does not follow completions")
    previous_completed_games = (
        int(completed_events[-1].details["completed_games"]) if completed_events else 0
    )
    expected_stop = min(
        previous_completed_games + plan.segment_games,
        plan.game_bound,
    )
    if segment_stop_game != expected_stop:
        raise ManagedContractError(
            "managed segment stop game does not match the schedule"
        )
    if git_commit != plan.git_commit:
        if not (
            _plan_used_recovery(plan)
            or _pending_recovery_for_segment(plan, segment_index) is not None
        ):
            raise ManagedContractError("managed plan Git commit does not match")
        root = _repository_root()
        if not _git_is_ancestor(root, plan.git_commit, git_commit):
            raise ManagedContractError("managed plan Git commit does not match")
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
        allowed_resumes = {expected_resume}
        recovery = _pending_recovery_for_segment(plan, segment_index)
        if recovery is not None and recovery.get("recovery_checkpoint"):
            allowed_resumes.add(
                Path(str(recovery["recovery_checkpoint"])).resolve(strict=False)
            )
        if start_mode != "exact-resume":
            raise ManagedContractError("managed continuation must use exact resume")
        if Path(resume).resolve(strict=False) not in allowed_resumes:
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


def _git_is_ancestor(root: Path, ancestor: str, commit: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, commit],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _assert_managed_git_state(
    plan: ManagedPlan,
    *,
    allow_recovery_descendant: bool = False,
) -> str:
    """Require a clean worktree on the frozen plan commit, or a recovery descendant."""
    root = _repository_root()
    commit, dirty = _git_state(root)
    if dirty:
        raise ManagedContractError("managed training requires a clean Git worktree")
    if commit == plan.git_commit:
        return commit
    if (
        allow_recovery_descendant
        and _git_is_ancestor(root, plan.git_commit, commit)
    ):
        return commit
    raise ManagedContractError("managed training Git commit has changed")


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
    completed_updates = int(envelope.payload.trainer_state["update_count"])
    expected_games = min(
        previous_completed_games + plan.segment_games,
        plan.game_bound,
    )
    update_bound = _optimizer_update_bound(plan)
    if update_bound is None:
        if completed_games != expected_games:
            raise ManagedContractError(
                "segment checkpoint game count does not match the bounded schedule"
            )
    else:
        if completed_updates > update_bound:
            raise ManagedContractError(
                "segment checkpoint exceeded the optimizer-update bound"
            )
        if completed_updates == update_bound:
            if not previous_completed_games < completed_games <= expected_games:
                raise ManagedContractError(
                    "optimizer-bounded segment game count is outside its safety ceiling"
                )
        elif completed_games != expected_games:
            raise ManagedContractError(
                "segment stopped before its game ceiling and optimizer-update bound"
            )
        elif completed_games >= plan.game_bound:
            raise ManagedContractError(
                "optimizer-update bound was not reached before the game ceiling"
            )
    return completed_games, checkpoint


def _trainer_arg_value(plan: ManagedPlan, option: str) -> str:
    matches = [
        index
        for index, value in enumerate(plan.common_trainer_args)
        if value == option
    ]
    if len(matches) != 1:
        raise ManagedContractError(
            f"managed trainer arguments must contain exactly one {option}"
        )
    index = matches[0]
    if index + 1 >= len(plan.common_trainer_args):
        raise ManagedContractError(f"managed trainer option {option} has no value")
    return plan.common_trainer_args[index + 1]


def _report_path(value: Any, *, field: str) -> Path:
    text = _require_text(value, field=field)
    path = Path(text)
    if not path.is_absolute():
        path = _repository_root() / path
    return path.resolve(strict=False)


def _report_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManagedContractError(f"policy-health {field} must be an object")
    return value


def _validate_policy_health_report(
    plan: ManagedPlan,
    *,
    segment_index: int,
    report_path: Path,
    checkpoint: Path,
    specialist_db: Path,
    completed_games: int,
    runtime_commit: str,
) -> dict[str, Any]:
    """Validate identities and frozen thresholds in one audit report."""
    gate = plan.policy_health
    if gate is None:
        raise ManagedContractError("policy-health gate is not enabled")
    report = _strict_json(report_path)
    if report.get("schema_version") != "nmm.generalist-policy-health.v1":
        raise ManagedContractError("policy-health report schema differs")
    evidence_id = _require_sha256(
        report.get("evidence_id"), field="policy-health evidence_id"
    )
    report_core = dict(report)
    del report_core["evidence_id"]
    if canonical_sha256(report_core) != evidence_id:
        raise ManagedContractError("policy-health evidence_id is invalid")

    identities = _report_mapping(
        report.get("identities"), field="identities"
    )
    expected_identity_values = {
        "git_commit": runtime_commit,
        "checkpoint_sha256": _file_sha256(checkpoint),
        "run_id": _segment_run_id(plan, segment_index),
        "experiment_id": plan.experiment_id,
        "corpus_sha256": gate.corpus_sha256,
        "paths_config_sha256": plan.paths_config_sha256,
        "specialist_db_sha256": _file_sha256(specialist_db),
    }
    for key, expected in expected_identity_values.items():
        if identities.get(key) != expected:
            raise ManagedContractError(
                f"policy-health report {key} identity differs"
            )
    expected_paths = {
        "checkpoint": checkpoint.resolve(strict=False),
        "corpus": Path(gate.corpus_path).resolve(strict=False),
        "specialist_db": specialist_db.resolve(strict=False),
    }
    for key, expected in expected_paths.items():
        if _report_path(identities.get(key), field=key) != expected:
            raise ManagedContractError(
                f"policy-health report {key} path differs"
            )

    checkpoint_state = _report_mapping(
        report.get("checkpoint_state"), field="checkpoint_state"
    )
    if checkpoint_state.get("game_count") != completed_games:
        raise ManagedContractError("policy-health report game count differs")
    diagnostic = _report_mapping(
        report.get("fixed_state_diagnostic"),
        field="fixed_state_diagnostic",
    )
    direct = _report_mapping(
        diagnostic.get("direct_lookahead_signal"),
        field="direct_lookahead_signal",
    )
    candidate = _report_mapping(
        diagnostic.get("candidate"), field="candidate"
    )
    metrics = _report_mapping(candidate.get("metrics"), field="candidate.metrics")
    all_metrics = _report_mapping(metrics.get("all"), field="candidate.metrics.all")

    direct_critical = direct.get("critical_states")
    candidate_critical = all_metrics.get("critical_states")
    if isinstance(direct_critical, bool) or not isinstance(direct_critical, int):
        raise ManagedContractError("policy-health direct critical count is invalid")
    if isinstance(candidate_critical, bool) or not isinstance(
        candidate_critical, int
    ):
        raise ManagedContractError("policy-health candidate critical count is invalid")
    direct_rate = _require_probability(
        direct.get("argmax_value_preserving_rate"),
        field="direct argmax preserving rate",
    )
    candidate_rate = _require_probability(
        all_metrics.get("critical_argmax_value_preserving_rate"),
        field="candidate argmax preserving rate",
    )
    candidate_margin = _require_finite_number(
        all_metrics.get(
            "critical_mean_preserving_minus_downgrading_logit"
        ),
        field="candidate preserving logit margin",
    )

    failures: list[str] = []
    if direct_critical != gate.exact_critical_states:
        failures.append("direct_critical_state_count")
    if candidate_critical != gate.exact_critical_states:
        failures.append("candidate_critical_state_count")
    if direct_rate < gate.required_direct_preserving_rate:
        failures.append("direct_value_preserving_rate")
    if candidate_rate < gate.min_candidate_preserving_rate:
        failures.append("candidate_value_preserving_rate")
    if candidate_margin < gate.min_candidate_logit_margin:
        failures.append("candidate_preserving_logit_margin")

    return {
        "passed": not failures,
        "failures": failures,
        "report": str(report_path.resolve(strict=False)),
        "report_sha256": _file_sha256(report_path),
        "evidence_id": evidence_id,
        "metrics": {
            "direct_critical_states": direct_critical,
            "candidate_critical_states": candidate_critical,
            "direct_value_preserving_rate": direct_rate,
            "candidate_value_preserving_rate": candidate_rate,
            "candidate_preserving_logit_margin": candidate_margin,
        },
        "thresholds": {
            "exact_critical_states": gate.exact_critical_states,
            "required_direct_preserving_rate": (
                gate.required_direct_preserving_rate
            ),
            "min_candidate_preserving_rate": (
                gate.min_candidate_preserving_rate
            ),
            "min_candidate_logit_margin": gate.min_candidate_logit_margin,
        },
    }


def _run_policy_health_gate(
    plan: ManagedPlan,
    *,
    segment_index: int,
    checkpoint: Path,
    completed_games: int,
    runtime_commit: str,
    python_executable: str,
    timeout_seconds: float,
    runner: Callable[..., subprocess.CompletedProcess],
) -> dict[str, Any] | None:
    gate = plan.policy_health
    if gate is None:
        return None
    corpus = Path(gate.corpus_path)
    audit_script = Path(gate.audit_script_path)
    if not corpus.is_file() or _file_sha256(corpus) != gate.corpus_sha256:
        raise ManagedContractError("policy-health corpus identity differs")
    if (
        not audit_script.is_file()
        or _file_sha256(audit_script) != gate.audit_script_sha256
    ):
        raise ManagedContractError("policy-health audit script identity differs")
    if timeout_seconds <= 0:
        raise ManagedContractError("no wall time remains for policy-health audit")

    report_path = (
        _segment_output_dir(plan, segment_index) / "policy-health.json"
    )
    if report_path.exists():
        raise ManagedContractError("policy-health report already exists")
    specialist_db = _specialist_db_path_for_plan(plan)
    command = [
        python_executable,
        str(audit_script),
        "--checkpoint",
        str(checkpoint),
        "--specialist-db",
        str(specialist_db),
        "--output",
        str(report_path),
        "--corpus",
        str(corpus),
        "--expected-corpus-sha256",
        gate.corpus_sha256,
        "--paths-config",
        plan.paths_config,
        "--expected-experiment-id",
        plan.experiment_id,
        "--expected-game-count",
        str(completed_games),
        "--seed",
        _trainer_arg_value(plan, "--seed"),
        "--schedule-max-games",
        str(plan.max_games),
        "--temp-start",
        _trainer_arg_value(plan, "--temp-start"),
        "--device",
        gate.device,
    ]
    try:
        result = runner(
            command,
            cwd=_repository_root(),
            check=False,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise ManagedContractError("policy-health audit timed out") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise ManagedContractError("policy-health audit could not run") from exc
    if result.returncode != 0:
        raise ManagedContractError(
            f"policy-health audit exited with code {result.returncode}"
        )
    if not report_path.is_file():
        raise ManagedContractError("policy-health audit did not publish a report")
    return _validate_policy_health_report(
        plan,
        segment_index=segment_index,
        report_path=report_path,
        checkpoint=checkpoint,
        specialist_db=specialist_db,
        completed_games=completed_games,
        runtime_commit=runtime_commit,
    )


def run_next_segment(
    plan_path: str | Path,
    authorization_path: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    health_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Run exactly one authorized segment, then verify its durable evidence."""
    plan_path = Path(plan_path).resolve(strict=False)
    authorization_path = Path(authorization_path).resolve(strict=False)
    plan = load_managed_plan(plan_path)
    _verify_authorization(plan, authorization_path)

    ledger_events = load_run_events(
        Path(plan.control_dir) / CONTROLLER_LEDGER_NAME
    )
    if ledger_events:
        last_event = ledger_events[-1]
        recovery_ready = (
            last_event.event_type == "managed_segment_interrupted"
            and last_event.reason_code in _RECOVERY_REASON_CODES
            and bool(last_event.details.get("recovery_checkpoint"))
        )
        if last_event.status in {"failed", "quarantined", "interrupted"} and not (
            recovery_ready
        ):
            raise ManagedContractError(
                "managed plan is stopped and requires Agent review"
            )

    completed_events = _completed_segment_events(plan)
    previous_completed_games = (
        int(completed_events[-1].details["completed_games"])
        if completed_events
        else 0
    )
    previous_completed_updates = (
        int(completed_events[-1].details["completed_updates"])
        if completed_events
        and completed_events[-1].details.get("completed_updates") is not None
        else None
    )
    if _plan_completion_reached(
        plan,
        completed_games=previous_completed_games,
        completed_updates=previous_completed_updates,
    ):
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

    completed_preview = _completed_segment_events(plan)
    pending_index = len(completed_preview) + 1
    allow_descendant = (
        _pending_recovery_for_segment(plan, pending_index) is not None
        or _plan_used_recovery(plan)
    )
    runtime_commit = _assert_managed_git_state(
        plan,
        allow_recovery_descendant=allow_descendant,
    )
    if _file_sha256(Path(plan.paths_config)) != plan.paths_config_sha256:
        raise ManagedContractError("managed paths configuration has changed")

    segment_index = pending_index
    previous_checkpoint = None
    previous_run_id = None
    recovery = _pending_recovery_for_segment(plan, segment_index)
    completed_events = completed_preview
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
    if recovery is not None:
        previous_checkpoint = Path(str(recovery["recovery_checkpoint"]))
        if previous_run_id is None and recovery.get("parent_run_id"):
            previous_run_id = str(recovery["parent_run_id"])
        # Live SpecialistDB may have advanced past the recovery envelope identity
        # during a failed mid-segment attempt; rebind before relaunch.
        specialist_identity = _live_specialist_identity(
            _specialist_db_path_for_plan(plan)
        )
        envelope = load_checkpoint(previous_checkpoint, map_location="cpu")
        recorded = envelope.payload.data_state["mutable_assets"]["specialist_db"][
            "sha256"
        ]
        if recorded != specialist_identity["sha256"]:
            refreshed = previous_checkpoint.with_name(
                f"{previous_checkpoint.stem}.refresh-{uuid4().hex}.pt"
            )
            _write_recovery_checkpoint(
                previous_checkpoint,
                refreshed,
                specialist_identity=specialist_identity,
                recovery_reason=str(
                    recovery.get("checkpoint_recovery_reason", "host-reboot")
                ),
            )
            os.replace(refreshed, previous_checkpoint)
    output_dir = _segment_output_dir(plan, segment_index)
    if output_dir.exists():
        if recovery is None:
            raise ManagedContractError(
                "next managed segment output already exists; run recover-interrupted "
                "if a host reboot left an incomplete segment"
            )
        # A previous recovery attempt may have overshot and been quarantined.
        # Keep the recovery checkpoint, but clear the failed output directory.
        stamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        failed = (
            Path(plan.control_dir)
            / "quarantine"
            / f"segment-{segment_index:04d}.failed-retry-{stamp}"
        )
        failed.parent.mkdir(parents=True, exist_ok=True)
        if failed.exists():
            raise ManagedContractError("failed-retry quarantine target already exists")
        output_dir.rename(failed)
    command = build_segment_command(
        plan,
        plan_path=plan_path,
        authorization_path=authorization_path,
        segment_index=segment_index,
        previous_checkpoint=previous_checkpoint,
        previous_run_id=previous_run_id,
        previous_completed_games=previous_completed_games,
        python_executable=python_executable,
    )

    root = _repository_root()
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
                "recovery": recovery is not None,
                "resume_checkpoint": (
                    None
                    if previous_checkpoint is None
                    else str(previous_checkpoint.resolve(strict=False))
                ),
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
        completed_updates = None
        if _optimizer_update_bound(plan) is not None:
            completed_updates = int(
                load_checkpoint(checkpoint, map_location="cpu").payload.trainer_state[
                    "update_count"
                ]
            )
        policy_health: dict[str, Any] | None = None
        if plan.policy_health is not None:
            try:
                policy_health = _run_policy_health_gate(
                    plan,
                    segment_index=segment_index,
                    checkpoint=checkpoint,
                    completed_games=completed_games,
                    runtime_commit=runtime_commit,
                    python_executable=python_executable,
                    timeout_seconds=remaining_seconds - elapsed,
                    runner=health_runner,
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                report = (
                    _segment_output_dir(plan, segment_index)
                    / "policy-health.json"
                )
                details: dict[str, Any] = {
                    "segment_index": segment_index,
                    "completed_games": completed_games,
                    "checkpoint": str(checkpoint.resolve(strict=False)),
                    "elapsed_seconds": elapsed,
                    "exception_type": type(exc).__name__,
                }
                if report.is_file():
                    details["report"] = str(report.resolve(strict=False))
                    details["report_sha256"] = _file_sha256(report)
                _append_controller_event(
                    plan,
                    status="quarantined",
                    event_type="managed_segment_policy_health_quarantined",
                    reason_code="policy_health_audit_failed",
                    details=details,
                )
                raise ManagedContractError(
                    "managed segment policy-health evidence is invalid"
                ) from exc
            if policy_health is None:
                raise ManagedContractError(
                    "enabled policy-health gate returned no result"
                )
            if not policy_health["passed"]:
                elapsed = time.monotonic() - started
                _append_controller_event(
                    plan,
                    status="quarantined",
                    event_type="managed_segment_policy_health_quarantined",
                    reason_code="policy_health_threshold_failed",
                    details={
                        "segment_index": segment_index,
                        "completed_games": completed_games,
                        "checkpoint": str(checkpoint.resolve(strict=False)),
                        "elapsed_seconds": elapsed,
                        "policy_health": policy_health,
                    },
                )
                raise ManagedContractError(
                    "managed segment failed the policy-health thresholds"
                )
        elapsed = time.monotonic() - started
        _append_controller_event(
            plan,
            status="completed",
            event_type="managed_segment_completed",
            details={
                "segment_index": segment_index,
                "run_id": _segment_run_id(plan, segment_index),
                "completed_games": completed_games,
                "completed_updates": completed_updates,
                "checkpoint": str(checkpoint.resolve(strict=False)),
                "elapsed_seconds": elapsed,
                "policy_health": policy_health,
            },
        )
        if _plan_completion_reached(
            plan,
            completed_games=completed_games,
            completed_updates=completed_updates,
        ):
            _append_controller_event(
                plan,
                status="completed",
                event_type="managed_plan_completed",
                details={
                    "completed_games": completed_games,
                    "completed_updates": completed_updates,
                },
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
    health_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Continue safe exact-resume segments until completion or a hard stop."""
    while True:
        status = run_next_segment(
            plan_path,
            authorization_path,
            runner=runner,
            health_runner=health_runner,
            python_executable=python_executable,
        )
        if status["state"] == "completed":
            return status
        if status["state"] != "ready_to_run":
            raise ManagedContractError(
                f"managed plan stopped in state {status['state']}"
            )
