"""Fail-closed continuation contract for the immutable Sanmill route probe."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from learned_ai.training.run_contract import canonical_sha256
from learned_ai.validation.sanmill_route_probe import (
    DEFAULT_PLAN_RELATIVE,
    ProbeGame,
    ProbePlan,
    SanmillRouteProbeError,
    _probe_game_record,
    _require_exact_keys,
    _strict_json,
    load_probe_plan,
    preflight_probe,
    tracked_plan_record,
)


CONTINUATION_PLAN_SCHEMA = "nmm.sanmill-no-update-route-continuation-plan.v1"
CONTINUATION_PREFLIGHT_SCHEMA = (
    "nmm.sanmill-no-update-route-continuation-preflight.v1"
)

_ROOT = Path(__file__).resolve().parents[2]
_PLAN_KEYS = {
    "schema_version",
    "status",
    "experiment_id",
    "claim_boundary",
    "parent_plan",
    "schedule_range",
    "bounded_work",
    "decision_rules",
    "plan_identity",
}
_PARENT_KEYS = {"path", "raw_sha256", "plan_identity"}
_RANGE_KEYS = {"start_scheduled_index", "end_scheduled_index_exclusive"}
_DECISION_RULES = {
    "diagnosis_only": True,
    "execution_requires_explicit_authority": True,
    "no_automatic_escalation": True,
    "no_retry": True,
    "preserve_parent_schedule_identity": True,
    "publish_success_or_failure_atomically": True,
    "refuse_output_overwrite": True,
    "training_launch": False,
}


@dataclass(frozen=True)
class ProbeContinuationPlan:
    path: Path
    raw_sha256: str
    identity: str
    experiment_id: str
    claim_boundary: str
    parent: ProbePlan
    start_scheduled_index: int
    end_scheduled_index_exclusive: int
    schedule: tuple[ProbeGame, ...]
    payload: Mapping[str, Any]


def _range_index(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SanmillRouteProbeError(f"{context} must be an integer")
    return value


def _bounded_work(parent: ProbePlan, schedule: tuple[ProbeGame, ...]) -> dict[str, int]:
    search_games = sum(game.opponent_kind == "sanmill" for game in schedule)
    return {
        "complete_games": len(schedule),
        "search_opponent_games": search_games,
        "frozen_target_games": len(schedule) - search_games,
        "maximum_logical_plies": len(schedule) * parent.max_ply,
        "maximum_search_calls": search_games * (parent.max_ply // 2),
        "maximum_requested_search_node_ceilings": sum(
            (game.node_budget or 0) * (parent.max_ply // 2) for game in schedule
        ),
    }


def load_probe_continuation_plan(path: str | Path) -> ProbeContinuationPlan:
    """Load a content-addressed contiguous slice of the immutable parent."""
    plan_path = Path(path)
    raw = plan_path.read_bytes()
    payload = _strict_json(plan_path)
    _require_exact_keys(payload, _PLAN_KEYS, context="probe continuation plan")
    if payload["schema_version"] != CONTINUATION_PLAN_SCHEMA:
        raise SanmillRouteProbeError("unsupported probe continuation plan schema")
    if payload["status"] != "prepared_unlaunched":
        raise SanmillRouteProbeError("probe continuation is not prepared/unlaunched")

    identity = payload["plan_identity"]
    if not isinstance(identity, str) or len(identity) != 64:
        raise SanmillRouteProbeError("probe continuation identity is invalid")
    identity_body = dict(payload)
    identity_body.pop("plan_identity")
    if canonical_sha256(identity_body) != identity:
        raise SanmillRouteProbeError("probe continuation identity mismatch")

    parent_record = payload["parent_plan"]
    if not isinstance(parent_record, Mapping):
        raise SanmillRouteProbeError("probe continuation parent must be an object")
    _require_exact_keys(parent_record, _PARENT_KEYS, context="probe continuation parent")
    if parent_record["path"] != DEFAULT_PLAN_RELATIVE.as_posix():
        raise SanmillRouteProbeError("probe continuation parent path drifted")
    parent = load_probe_plan(_ROOT / DEFAULT_PLAN_RELATIVE)
    if parent_record["raw_sha256"] != parent.raw_sha256:
        raise SanmillRouteProbeError("probe continuation parent bytes drifted")
    if parent_record["plan_identity"] != parent.identity:
        raise SanmillRouteProbeError("probe continuation parent identity drifted")

    range_record = payload["schedule_range"]
    if not isinstance(range_record, Mapping):
        raise SanmillRouteProbeError("probe continuation range must be an object")
    _require_exact_keys(range_record, _RANGE_KEYS, context="probe continuation range")
    start = _range_index(
        range_record["start_scheduled_index"], context="continuation start"
    )
    end = _range_index(
        range_record["end_scheduled_index_exclusive"], context="continuation end"
    )
    if start < 0 or end > len(parent.schedule) or start >= end:
        raise SanmillRouteProbeError("probe continuation range is invalid")
    schedule = tuple(parent.schedule[start:end])
    if [game.scheduled_index for game in schedule] != list(range(start, end)):
        raise SanmillRouteProbeError("probe continuation parent range drifted")

    bounded = _bounded_work(parent, schedule)
    if dict(payload["bounded_work"]) != bounded:
        raise SanmillRouteProbeError("probe continuation bounded work drifted")
    if dict(payload["decision_rules"]) != _DECISION_RULES:
        raise SanmillRouteProbeError("probe continuation decision boundary drifted")

    return ProbeContinuationPlan(
        path=plan_path,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        identity=identity,
        experiment_id=str(payload["experiment_id"]),
        claim_boundary=str(payload["claim_boundary"]),
        parent=parent,
        start_scheduled_index=start,
        end_scheduled_index_exclusive=end,
        schedule=schedule,
        payload=payload,
    )


def continuation_probe_plan(plan: ProbeContinuationPlan) -> ProbePlan:
    """Build the existing no-update execution view for the selected range."""
    payload = dict(plan.parent.payload)
    payload.update(
        {
            "schema_version": CONTINUATION_PLAN_SCHEMA,
            "status": "prepared_unlaunched",
            "experiment_id": plan.experiment_id,
            "claim_boundary": plan.claim_boundary,
            "schedule": [_probe_game_record(game) for game in plan.schedule],
            "bounded_work": dict(plan.payload["bounded_work"]),
            "decision_rules": dict(plan.payload["decision_rules"]),
            "plan_identity": plan.identity,
        }
    )
    node_budgets = tuple(
        sorted(
            {
                int(game.node_budget)
                for game in plan.schedule
                if game.node_budget is not None
            }
        )
    )
    return ProbePlan(
        path=plan.path,
        raw_sha256=plan.raw_sha256,
        identity=plan.identity,
        experiment_id=plan.experiment_id,
        claim_boundary=plan.claim_boundary,
        seed=plan.parent.seed,
        temperature=plan.parent.temperature,
        max_ply=plan.parent.max_ply,
        policy_hidden=plan.parent.policy_hidden,
        node_budgets=node_budgets,
        schedule=plan.schedule,
        payload=payload,
    )


def preflight_probe_continuation(
    plan_path: str | Path,
    paths_config: str | Path,
    *,
    require_published: bool = True,
    verify_malom_components: bool = True,
    perform_route_check: bool = True,
) -> dict[str, Any]:
    """Audit a continuation slice without consuming any scheduled game."""
    continuation = load_probe_continuation_plan(plan_path)
    parent_report = preflight_probe(
        continuation.parent.path,
        paths_config,
        require_published=require_published,
        verify_malom_components=verify_malom_components,
        perform_route_check=perform_route_check,
    )
    effective = continuation_probe_plan(continuation)
    return {
        **parent_report,
        "schema_version": CONTINUATION_PREFLIGHT_SCHEMA,
        "status": "ready_for_authorized_continuation_probe",
        "launch_authorized": False,
        "plan": tracked_plan_record(effective),
        "parent_probe_plan": parent_report["plan"],
        "schedule_range": {
            "start_scheduled_index": continuation.start_scheduled_index,
            "end_scheduled_index_exclusive": (
                continuation.end_scheduled_index_exclusive
            ),
        },
        "bounded_work": dict(continuation.payload["bounded_work"]),
        "next_gate": "explicit one-run continuation-probe authority",
    }
