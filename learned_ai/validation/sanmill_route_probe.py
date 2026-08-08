"""No-update integrated-route probe for the pinned Sanmill lineage.

The probe invokes the production Generalist rollout and strict Sanmill referee
without constructing an optimiser or permitting rollout/database persistence.
Loading a plan or running preflight does not authorise the bounded 36-game run.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import platform
import sqlite3
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from ai.human_db import HumanDB
from game.board import BoardState
from learned_ai.data.data_contract import (
    load_dataset_manifest,
    verify_dataset_snapshot,
)
from learned_ai.data.malom_label_provenance import CURRENT_MALOM_LABEL_VERSION
from learned_ai.data.specialist_db import SpecialistDB
from learned_ai.models.lookahead_advisor import LookaheadAdvisor
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.training.generalist_preflight import load_training_settings
from learned_ai.training.run_contract import canonical_sha256
from learned_ai.training.sanmill_referee import (
    TRAINING_REFEREE_PROFILE,
    TRAINING_REFEREE_SEMANTIC_DIGEST,
    TRAINING_SANMILL_BINARY_SHA256,
    TRAINING_SANMILL_BINARY_SIZE,
    TRAINING_SANMILL_COMMIT,
    TRAINING_SANMILL_TREE,
    SanmillBoardMirrorError,
    SanmillTrainingGame,
    SanmillTrainingOpponent,
    inspect_sanmill_training_installation,
    training_installation_record,
)
from learned_ai.training.training_identity import load_trainer_ruleset
from scripts import train_s_gen_v2 as trainer


PLAN_SCHEMA = "nmm.sanmill-no-update-route-probe-plan.v1"
DIAGNOSTIC_PLAN_SCHEMA = "nmm.sanmill-no-update-route-diagnostic-plan.v1"
PREFLIGHT_SCHEMA = "nmm.sanmill-no-update-route-probe-preflight.v1"
DIAGNOSTIC_PREFLIGHT_SCHEMA = (
    "nmm.sanmill-no-update-route-diagnostic-preflight.v1"
)
RESULT_SCHEMA = "nmm.sanmill-no-update-route-probe-result.v1"
FAILURE_SCHEMA = "nmm.sanmill-no-update-route-probe-failure.v1"
SCHEDULE_FAILURE_SCHEMA = "nmm.sanmill-route-probe-schedule-failure.v1"
DEFAULT_PLAN_RELATIVE = Path(
    "docs/experiments/sanmill-no-update-integrated-route-probe-v1.json"
)
DEFAULT_DIAGNOSTIC_PLAN_RELATIVE = Path(
    "docs/experiments/sanmill-no-update-integrated-route-diagnostic-v1.json"
)
DEFAULT_PATHS_RELATIVE = Path("data/training_paths.local.json")

_ROOT = Path(__file__).resolve().parents[2]
_PLAN_KEYS = {
    "schema_version",
    "status",
    "experiment_id",
    "claim_boundary",
    "sanmill_runtime",
    "model_route",
    "data_inputs",
    "schedule",
    "bounded_work",
    "decision_rules",
    "plan_identity",
}
_GAME_KEYS = {
    "scheduled_index",
    "game_id",
    "role",
    "opponent_kind",
    "node_budget",
    "learner_color",
    "route_depth",
    "sim_ply_depth",
    "torch_seed",
}
_DIAGNOSTIC_PLAN_KEYS = {
    "schema_version",
    "status",
    "experiment_id",
    "claim_boundary",
    "parent_plan",
    "selected_schedule_entry",
    "bounded_work",
    "decision_rules",
    "plan_identity",
}
_DIAGNOSTIC_PARENT_KEYS = {"path", "raw_sha256", "plan_identity"}
_RUN_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._-")


class SanmillRouteProbeError(ValueError):
    """Raised when the probe contract or evidence fails closed."""


class SanmillRouteProbeScheduleError(SanmillRouteProbeError):
    """Raised with the exact completed prefix and failed schedule entry."""

    def __init__(self, diagnostic: Mapping[str, Any]) -> None:
        super().__init__("probe schedule failed closed")
        self.diagnostic = dict(diagnostic)


class SanmillRouteProbeRunFailure(SanmillRouteProbeError):
    """Carries a complete quarantine report for atomic publication."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        super().__init__("probe run failed closed")
        self.report = dict(report)


@dataclass(frozen=True)
class ProbeGame:
    scheduled_index: int
    game_id: str
    role: str
    opponent_kind: str
    node_budget: int | None
    learner_color: str
    route_depth: str
    sim_ply_depth: int
    torch_seed: int


@dataclass(frozen=True)
class ProbePlan:
    path: Path
    raw_sha256: str
    identity: str
    experiment_id: str
    claim_boundary: str
    seed: int
    temperature: float
    max_ply: int
    policy_hidden: tuple[int, ...]
    node_budgets: tuple[int, ...]
    schedule: tuple[ProbeGame, ...]
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class ProbeDiagnosticPlan:
    path: Path
    raw_sha256: str
    identity: str
    experiment_id: str
    claim_boundary: str
    parent: ProbePlan
    selected: ProbeGame
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class ProbeInputs:
    human_db: Path
    specialist_db: Path
    malom_db: Path
    malom_manifest: Path
    ruleset_manifest: Path
    sanmill_checkout: Path


@dataclass
class ProbeRuntime:
    device: torch.device
    model: Any
    frozen_opponent: Any
    lookahead_advisor: Any
    human_db: HumanDB
    specialist_db: SpecialistDB
    malom_db: ExternalSolvedDB
    human_route: Any
    specialist_route: Any
    malom_route: Any
    installation: Any
    timing_sink: "_TimingSink"

    def close(self) -> None:
        self.human_db.close()
        self.specialist_db.close()
        self.malom_db.close()


def _probe_game_record(game: ProbeGame) -> dict[str, Any]:
    return {
        "scheduled_index": game.scheduled_index,
        "game_id": game.game_id,
        "role": game.role,
        "opponent_kind": game.opponent_kind,
        "node_budget": game.node_budget,
        "learner_color": game.learner_color,
        "route_depth": game.route_depth,
        "sim_ply_depth": game.sim_ply_depth,
        "torch_seed": game.torch_seed,
    }


def _completed_sample_identities(
    samples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "scheduled_index": int(sample["scheduled_index"]),
            "game_id": str(sample["game_id"]),
            "sample_identity": canonical_sha256(sample),
        }
        for sample in samples
    ]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise SanmillRouteProbeError(f"duplicate JSON key {key!r} in {path}")
            value[key] = item
        return value

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SanmillRouteProbeError(f"cannot read probe JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise SanmillRouteProbeError("probe JSON must be an object")
    return payload


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise SanmillRouteProbeError(
            f"{context} has wrong members: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_positive_int(value: Any, *, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SanmillRouteProbeError(f"{context} must be a positive integer")
    return value


def _require_finite_positive(value: Any, *, context: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise SanmillRouteProbeError(f"{context} must be finite and positive")
    return float(value)


def _runtime_contract() -> dict[str, Any]:
    return {
        "commit": TRAINING_SANMILL_COMMIT,
        "tree": TRAINING_SANMILL_TREE,
        "binary_sha256": TRAINING_SANMILL_BINARY_SHA256,
        "binary_size": TRAINING_SANMILL_BINARY_SIZE,
        "strict_referee_profile": TRAINING_REFEREE_PROFILE,
        "strict_referee_semantic_digest": TRAINING_REFEREE_SEMANTIC_DIGEST,
        "search_threads": 1,
        "shuffling": False,
        "fallback": "none",
    }


def load_probe_plan(path: str | Path) -> ProbePlan:
    """Load and fully validate the content-addressed 36-game plan."""
    plan_path = Path(path)
    raw = plan_path.read_bytes()
    payload = _strict_json(plan_path)
    _require_exact_keys(payload, _PLAN_KEYS, context="probe plan")
    if payload["schema_version"] != PLAN_SCHEMA:
        raise SanmillRouteProbeError("unsupported probe plan schema")
    if payload["status"] != "implemented_unlaunched":
        raise SanmillRouteProbeError("probe plan is not implemented/unlaunched")
    identity = payload["plan_identity"]
    if not isinstance(identity, str) or len(identity) != 64:
        raise SanmillRouteProbeError("probe plan identity is invalid")
    identity_body = dict(payload)
    identity_body.pop("plan_identity")
    if canonical_sha256(identity_body) != identity:
        raise SanmillRouteProbeError("probe plan identity mismatch")
    if dict(payload["sanmill_runtime"]) != _runtime_contract():
        raise SanmillRouteProbeError("probe Sanmill runtime pin drifted")

    model_route = payload["model_route"]
    expected_model_keys = {
        "seed",
        "start_mode",
        "checkpoint",
        "temperature",
        "max_ply",
        "policy_hidden",
        "sim_ply_depth_normal",
        "sim_ply_depth_deep",
        "cuda_required",
        "batch_games",
        "disabled_components",
        "optimizer",
        "rollout_persistence",
    }
    _require_exact_keys(model_route, expected_model_keys, context="model_route")
    if model_route["start_mode"] != "fresh" or model_route["checkpoint"] is not None:
        raise SanmillRouteProbeError("probe must start from fresh random weights")
    if model_route["optimizer"] is not None:
        raise SanmillRouteProbeError("probe must not construct an optimizer")
    if model_route["rollout_persistence"] is not False:
        raise SanmillRouteProbeError("probe rollout persistence must be false")
    if model_route["cuda_required"] is not True or model_route["batch_games"] != 1:
        raise SanmillRouteProbeError("probe requires CUDA and batch_games=1")
    if model_route["sim_ply_depth_normal"] != 5:
        raise SanmillRouteProbeError("normal probe route must use five simulated plies")
    if model_route["sim_ply_depth_deep"] != 12:
        raise SanmillRouteProbeError("deep probe route must use twelve simulated plies")
    disabled = set(model_route["disabled_components"])
    if disabled != {
        "sentinel",
        "value_net",
        "gap_net",
        "s1a_warmstart",
        "s1b_refresher",
        "imitation_mix",
        "opening_forcing",
        "recovery",
        "retry_rollouts",
        "branch_rollouts",
    }:
        raise SanmillRouteProbeError("probe disabled-component set drifted")

    seed = _require_positive_int(model_route["seed"], context="model seed")
    max_ply = _require_positive_int(model_route["max_ply"], context="max_ply")
    if max_ply != 120:
        raise SanmillRouteProbeError("probe max_ply must remain 120")
    temperature = _require_finite_positive(
        model_route["temperature"], context="temperature"
    )
    policy_hidden = tuple(model_route["policy_hidden"])
    if not policy_hidden or any(
        _require_positive_int(value, context="policy_hidden") <= 0
        for value in policy_hidden
    ):
        raise SanmillRouteProbeError("policy_hidden is invalid")

    schedule_values = payload["schedule"]
    if not isinstance(schedule_values, list):
        raise SanmillRouteProbeError("probe schedule must be an array")
    schedule: list[ProbeGame] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(schedule_values):
        if not isinstance(value, Mapping):
            raise SanmillRouteProbeError(f"schedule[{index}] must be an object")
        _require_exact_keys(value, _GAME_KEYS, context=f"schedule[{index}]")
        if value["scheduled_index"] != index:
            raise SanmillRouteProbeError("probe schedule index is not contiguous")
        role = value["role"]
        if not isinstance(role, str) or not role:
            raise SanmillRouteProbeError("probe role must be non-empty text")
        game_id, torch_seed = trainer._derive_game_identity(seed, index, role)
        if value["game_id"] != game_id or value["torch_seed"] != torch_seed:
            raise SanmillRouteProbeError("probe game identity derivation drifted")
        if game_id in seen_ids:
            raise SanmillRouteProbeError("probe schedule duplicates a game identity")
        seen_ids.add(game_id)
        opponent_kind = value["opponent_kind"]
        if opponent_kind not in {"sanmill", "frozen_target"}:
            raise SanmillRouteProbeError("probe opponent kind is invalid")
        budget = value["node_budget"]
        if opponent_kind == "sanmill":
            budget = _require_positive_int(budget, context="node_budget")
        elif budget is not None:
            raise SanmillRouteProbeError("frozen target must not have a node budget")
        learner_color = value["learner_color"]
        if learner_color not in {"W", "B"}:
            raise SanmillRouteProbeError("probe learner color is invalid")
        route_depth = value["route_depth"]
        expected_depth = {"normal": 5, "deep": 12}.get(route_depth)
        if expected_depth is None or value["sim_ply_depth"] != expected_depth:
            raise SanmillRouteProbeError("probe route depth is inconsistent")
        schedule.append(
            ProbeGame(
                scheduled_index=index,
                game_id=game_id,
                role=role,
                opponent_kind=opponent_kind,
                node_budget=budget,
                learner_color=learner_color,
                route_depth=route_depth,
                sim_ply_depth=expected_depth,
                torch_seed=torch_seed,
            )
        )

    budgets = tuple(
        sorted({game.node_budget for game in schedule if game.node_budget is not None})
    )
    if budgets != (1_000, 5_000, 25_000, 100_000, 500_000):
        raise SanmillRouteProbeError("probe node budget matrix drifted")
    if len(schedule) != 36:
        raise SanmillRouteProbeError("probe schedule must contain 36 games")
    for budget in budgets:
        games = [game for game in schedule if game.node_budget == budget]
        if len(games) != 6:
            raise SanmillRouteProbeError("each node budget must have six games")
        if sum(game.route_depth == "normal" for game in games) != 4:
            raise SanmillRouteProbeError("each budget needs four normal games")
        if sum(game.route_depth == "deep" for game in games) != 2:
            raise SanmillRouteProbeError("each budget needs two deep games")
        if sum(game.learner_color == "W" for game in games) != 3:
            raise SanmillRouteProbeError("each budget must be color balanced")
    controls = [game for game in schedule if game.opponent_kind == "frozen_target"]
    if (
        len(controls) != 6
        or sum(game.route_depth == "normal" for game in controls) != 4
    ):
        raise SanmillRouteProbeError("frozen-target control matrix drifted")
    if sum(game.route_depth == "deep" for game in controls) != 2:
        raise SanmillRouteProbeError("frozen-target deep controls drifted")
    if sum(game.learner_color == "W" for game in controls) != 3:
        raise SanmillRouteProbeError("frozen controls must be color balanced")

    search_games = sum(game.opponent_kind == "sanmill" for game in schedule)
    bounded = {
        "complete_games": len(schedule),
        "search_opponent_games": search_games,
        "frozen_target_games": len(schedule) - search_games,
        "maximum_logical_plies": len(schedule) * max_ply,
        "maximum_search_calls": search_games * (max_ply // 2),
        "maximum_requested_search_node_ceilings": sum(
            (game.node_budget or 0) * (max_ply // 2) for game in schedule
        ),
    }
    if dict(payload["bounded_work"]) != bounded:
        raise SanmillRouteProbeError("probe bounded-work calculation drifted")
    rules = payload["decision_rules"]
    if dict(rules) != {
        "auto_select_node_ladder": False,
        "strength_claim": False,
        "training_launch": False,
        "publish_only_complete_schedule": True,
        "refuse_output_overwrite": True,
    }:
        raise SanmillRouteProbeError("probe decision boundary drifted")

    return ProbePlan(
        path=plan_path,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        identity=identity,
        experiment_id=str(payload["experiment_id"]),
        claim_boundary=str(payload["claim_boundary"]),
        seed=seed,
        temperature=temperature,
        max_ply=max_ply,
        policy_hidden=policy_hidden,
        node_budgets=budgets,
        schedule=tuple(schedule),
        payload=payload,
    )


def load_probe_diagnostic_plan(path: str | Path) -> ProbeDiagnosticPlan:
    """Load the one-entry diagnostic without altering its parent plan."""
    plan_path = Path(path)
    raw = plan_path.read_bytes()
    payload = _strict_json(plan_path)
    _require_exact_keys(
        payload,
        _DIAGNOSTIC_PLAN_KEYS,
        context="probe diagnostic plan",
    )
    if payload["schema_version"] != DIAGNOSTIC_PLAN_SCHEMA:
        raise SanmillRouteProbeError("unsupported probe diagnostic plan schema")
    if payload["status"] != "prepared_unlaunched":
        raise SanmillRouteProbeError("probe diagnostic is not prepared/unlaunched")
    identity = payload["plan_identity"]
    if not isinstance(identity, str) or len(identity) != 64:
        raise SanmillRouteProbeError("probe diagnostic identity is invalid")
    identity_body = dict(payload)
    identity_body.pop("plan_identity")
    if canonical_sha256(identity_body) != identity:
        raise SanmillRouteProbeError("probe diagnostic identity mismatch")

    parent_record = payload["parent_plan"]
    if not isinstance(parent_record, Mapping):
        raise SanmillRouteProbeError("probe diagnostic parent must be an object")
    _require_exact_keys(
        parent_record,
        _DIAGNOSTIC_PARENT_KEYS,
        context="probe diagnostic parent",
    )
    if parent_record["path"] != DEFAULT_PLAN_RELATIVE.as_posix():
        raise SanmillRouteProbeError("probe diagnostic parent path drifted")
    parent = load_probe_plan(_ROOT / DEFAULT_PLAN_RELATIVE)
    if parent_record["raw_sha256"] != parent.raw_sha256:
        raise SanmillRouteProbeError("probe diagnostic parent bytes drifted")
    if parent_record["plan_identity"] != parent.identity:
        raise SanmillRouteProbeError("probe diagnostic parent identity drifted")

    selected_record = payload["selected_schedule_entry"]
    if not isinstance(selected_record, Mapping):
        raise SanmillRouteProbeError(
            "probe diagnostic selected schedule entry must be an object"
        )
    _require_exact_keys(
        selected_record,
        _GAME_KEYS,
        context="probe diagnostic selected schedule entry",
    )
    selected = parent.schedule[0]
    if dict(selected_record) != _probe_game_record(selected):
        raise SanmillRouteProbeError(
            "probe diagnostic must preserve parent schedule index zero exactly"
        )

    bounded = {
        "complete_games": 1,
        "search_opponent_games": 1,
        "frozen_target_games": 0,
        "maximum_logical_plies": parent.max_ply,
        "maximum_search_calls": parent.max_ply // 2,
        "maximum_requested_search_node_ceilings": (
            int(selected.node_budget or 0) * (parent.max_ply // 2)
        ),
    }
    if dict(payload["bounded_work"]) != bounded:
        raise SanmillRouteProbeError("probe diagnostic bounded work drifted")
    if dict(payload["decision_rules"]) != {
        "diagnosis_only": True,
        "execution_requires_explicit_authority": True,
        "no_automatic_escalation": True,
        "no_retry": True,
        "preserve_parent_schedule_identity": True,
        "publish_success_or_failure_atomically": True,
        "refuse_output_overwrite": True,
        "training_launch": False,
    }:
        raise SanmillRouteProbeError("probe diagnostic decision boundary drifted")

    return ProbeDiagnosticPlan(
        path=plan_path,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        identity=identity,
        experiment_id=str(payload["experiment_id"]),
        claim_boundary=str(payload["claim_boundary"]),
        parent=parent,
        selected=selected,
        payload=payload,
    )


def diagnostic_probe_plan(plan: ProbeDiagnosticPlan) -> ProbePlan:
    """Derive the one-entry execution view consumed by the existing route."""
    payload = dict(plan.parent.payload)
    payload.update(
        {
            "schema_version": DIAGNOSTIC_PLAN_SCHEMA,
            "status": "prepared_unlaunched",
            "experiment_id": plan.experiment_id,
            "claim_boundary": plan.claim_boundary,
            "schedule": [_probe_game_record(plan.selected)],
            "bounded_work": dict(plan.payload["bounded_work"]),
            "decision_rules": dict(plan.payload["decision_rules"]),
            "plan_identity": plan.identity,
        }
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
        node_budgets=(int(plan.selected.node_budget or 0),),
        schedule=(plan.selected,),
        payload=payload,
    )


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(_ROOT), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SanmillRouteProbeError(f"cannot inspect NMM_LLM Git state: {detail}")
    return result.stdout.strip()


def inspect_published_source(*, require_published: bool) -> dict[str, Any]:
    """Bind probe execution to a clean dev commit and its upstream."""
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise SanmillRouteProbeError("probe requires a clean tracked worktree")
    if _git("branch", "--show-current") != "dev":
        raise SanmillRouteProbeError("probe must run from dev")
    commit = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "@{upstream}")
    published = commit == upstream
    if require_published and not published:
        raise SanmillRouteProbeError("probe source commit must already be published")
    return {
        "branch": "dev",
        "commit": commit,
        "tree": _git("rev-parse", "HEAD^{tree}"),
        "upstream_commit": upstream,
        "tracked_worktree": "clean",
        "published": published,
    }


def tracked_plan_record(plan: ProbePlan) -> dict[str, str]:
    try:
        relative = plan.path.resolve().relative_to(_ROOT).as_posix()
    except ValueError as exc:
        raise SanmillRouteProbeError(
            "probe plan must be inside this repository"
        ) from exc
    _git("ls-files", "--error-unmatch", "--", relative)
    return {
        "relative_path": relative,
        "raw_sha256": plan.raw_sha256,
        "identity": plan.identity,
    }


def _resolve_setting(root: Path, value: Any, *, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SanmillRouteProbeError(f"local path {key} is not configured")
    path = Path(os.path.expandvars(os.path.expanduser(value.strip())))
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def resolve_probe_inputs(paths_config: str | Path) -> ProbeInputs:
    settings = load_training_settings(_ROOT, str(paths_config))
    required = {
        "human_db_route_probe_snapshot_path",
        "specialist_db_route_probe_snapshot_path",
        "malom_db_path",
        "sanmill_training_checkout",
    }
    missing = sorted(required - set(settings.values))
    if missing:
        raise SanmillRouteProbeError(
            "probe path registry is missing: " + ", ".join(missing)
        )
    return ProbeInputs(
        human_db=_resolve_setting(
            _ROOT,
            settings.values["human_db_route_probe_snapshot_path"],
            key="human_db_route_probe_snapshot_path",
        ),
        specialist_db=_resolve_setting(
            _ROOT,
            settings.values["specialist_db_route_probe_snapshot_path"],
            key="specialist_db_route_probe_snapshot_path",
        ),
        malom_db=_resolve_setting(
            _ROOT, settings.values["malom_db_path"], key="malom_db_path"
        ),
        malom_manifest=_ROOT / "data/manifests/malom-sector-corrected-v1.json",
        ruleset_manifest=_ROOT / "data/rulesets/nmm-training-core@2.json",
        sanmill_checkout=_resolve_setting(
            _ROOT,
            settings.values["sanmill_training_checkout"],
            key="sanmill_training_checkout",
        ),
    )


def _sidecar_inventory(path: Path) -> list[dict[str, Any]]:
    inventory = []
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists():
            inventory.append(
                {
                    "name": sidecar.name,
                    "size": sidecar.stat().st_size,
                    "sha256": _sha256_file(sidecar),
                }
            )
    return inventory


def _sqlite_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SanmillRouteProbeError(f"SQLite snapshot is missing: {path}")
    uri = f"file:{path.resolve().as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if quick_check != "ok":
            raise SanmillRouteProbeError(f"SQLite quick_check failed: {path}")
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        counts = {
            table: int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in sorted(tables)
        }
        metadata = (
            dict(connection.execute("SELECT key, value FROM meta"))
            if "meta" in tables
            else {}
        )
    finally:
        connection.close()
    return {
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
        "quick_check": quick_check,
        "counts": counts,
        "metadata": metadata,
        "sidecars": _sidecar_inventory(path),
    }


def verify_probe_inputs(
    plan: ProbePlan,
    inputs: ProbeInputs,
    *,
    verify_malom_components: bool = True,
) -> dict[str, Any]:
    """Verify immutable data and rules identities without writable opens."""
    expected = plan.payload["data_inputs"]
    human = _sqlite_record(inputs.human_db)
    specialist = _sqlite_record(inputs.specialist_db)
    if human != expected["human_db"]:
        raise SanmillRouteProbeError("HumanDB probe snapshot identity drifted")
    if specialist != expected["specialist_db"]:
        raise SanmillRouteProbeError("SpecialistDB probe snapshot identity drifted")
    if human["sidecars"] or specialist["sidecars"]:
        raise SanmillRouteProbeError("probe SQLite snapshots must be sidecar-free")
    if specialist["metadata"].get("malom_label_version") != (
        CURRENT_MALOM_LABEL_VERSION
    ):
        raise SanmillRouteProbeError("probe SpecialistDB label version drifted")
    if any(
        specialist["counts"].get(table) != 0
        for table in ("positions", "preferred_plays", "winning_lines")
    ):
        raise SanmillRouteProbeError("probe SpecialistDB must remain empty")

    manifest_sha256 = _sha256_file(inputs.malom_manifest)
    if manifest_sha256 != expected["malom"]["manifest_sha256"]:
        raise SanmillRouteProbeError("Malom manifest file identity drifted")
    manifest = load_dataset_manifest(inputs.malom_manifest)
    if manifest.manifest_sha256 != manifest_sha256:
        raise SanmillRouteProbeError("Malom manifest canonical identity drifted")
    if manifest.trust_level != CURRENT_MALOM_LABEL_VERSION:
        raise SanmillRouteProbeError("Malom manifest trust level drifted")
    anchor = next(
        (
            component
            for component in manifest.components
            if component.relative_path == "std.secval"
        ),
        None,
    )
    if anchor is None:
        raise SanmillRouteProbeError("Malom manifest omits std.secval")
    anchor_sha256 = _sha256_file(inputs.malom_db / anchor.relative_path)
    if anchor_sha256 != anchor.sha256:
        raise SanmillRouteProbeError("Malom std.secval anchor identity drifted")
    manifest_record = {
        "manifest_sha256": manifest_sha256,
        "trust_level": manifest.trust_level,
        "component_count": len(manifest.components),
        "anchor_relative_path": anchor.relative_path,
        "anchor_sha256": anchor_sha256,
    }
    if manifest_record != expected["malom"]:
        raise SanmillRouteProbeError("Malom manifest contract drifted")
    structural = None
    if verify_malom_components:
        structural = verify_dataset_snapshot(inputs.malom_db, manifest)
    malom = ExternalSolvedDB(str(inputs.malom_db), strict=True)
    try:
        if not malom.is_available():
            raise SanmillRouteProbeError("Malom database is unavailable")
    finally:
        malom.close()

    ruleset_file_sha256 = _sha256_file(inputs.ruleset_manifest)
    ruleset = load_trainer_ruleset(inputs.ruleset_manifest).to_dict()
    if {
        "file_sha256": ruleset_file_sha256,
        **ruleset,
    } != expected["ruleset"]:
        raise SanmillRouteProbeError("probe ruleset identity drifted")
    return {
        "human_db": human,
        "specialist_db": specialist,
        "malom": {
            **manifest_record,
            "verified_snapshot": structural,
        },
        "ruleset": {"file_sha256": ruleset_file_sha256, **ruleset},
    }


def model_state_sha256(model: Any) -> str:
    """Hash model tensor names, metadata, and bytes without serialization."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = value.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


class _TimingCollector:
    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)

    def observe(self, stage: str, seconds: float) -> None:
        if not isinstance(stage, str) or not stage:
            raise SanmillRouteProbeError("timing stage must be non-empty text")
        if not math.isfinite(seconds) or seconds < 0.0:
            raise SanmillRouteProbeError("probe observed non-finite timing")
        self.samples[stage].append(float(seconds))

    def record(self) -> dict[str, list[float]]:
        return {stage: list(values) for stage, values in sorted(self.samples.items())}


class _TimingSink:
    def __init__(self) -> None:
        self.current: _TimingCollector | None = None

    def observe(self, stage: str, seconds: float) -> None:
        if self.current is None:
            raise SanmillRouteProbeError("timing arrived outside a probe game")
        self.current.observe(stage, seconds)


class _TimedReadOnlyProxy:
    """Time selected read calls while delegating to the production adapter."""

    def __init__(
        self,
        target: Any,
        sink: _TimingSink,
        methods: Mapping[str, str],
    ) -> None:
        self._target = target
        self._sink = sink
        self._methods = dict(methods)

    @property
    def target(self) -> Any:
        return self._target

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._target, name)
        stage = self._methods.get(name)
        if stage is None or not callable(attribute):
            return attribute

        def timed(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return attribute(*args, **kwargs)
            finally:
                self._sink.observe(stage, time.perf_counter() - started)

        return timed


def _cuda_record(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SanmillRouteProbeError("the integrated probe requires CUDA")
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": str(device),
        "index": index,
        "name": properties.name,
        "total_memory": int(properties.total_memory),
        "capability": list(torch.cuda.get_device_capability(index)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _power_scheme() -> dict[str, Any]:
    result = subprocess.run(
        ["powercfg", "/getactivescheme"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _process_memory_record() -> dict[str, int | None]:
    if os.name != "nt":
        return {"working_set_bytes": None, "peak_working_set_bytes": None}
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        raise SanmillRouteProbeError("cannot inspect Windows process memory")
    return {
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
    }


def _host_record(device: torch.device) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "power_scheme": _power_scheme(),
        "cuda": _cuda_record(device),
        "process_memory": _process_memory_record(),
    }


def _build_runtime(plan: ProbePlan, inputs: ProbeInputs) -> ProbeRuntime:
    installation = inspect_sanmill_training_installation(inputs.sanmill_checkout)
    human = HumanDB(inputs.human_db, read_only=True, immutable=True)
    if not human.is_available():
        raise SanmillRouteProbeError("read-only HumanDB snapshot is unavailable")
    specialist = SpecialistDB(inputs.specialist_db, read_only=True)
    specialist.require_trusted_malom_labels()
    malom = ExternalSolvedDB(str(inputs.malom_db), strict=True)
    if not malom.is_available():
        human.close()
        specialist.close()
        raise SanmillRouteProbeError("read-only Malom adapter is unavailable")
    sink = _TimingSink()
    human_route = _TimedReadOnlyProxy(
        human, sink, {"query_all_frequencies": "human_db_query"}
    )
    specialist_route = _TimedReadOnlyProxy(
        specialist, sink, {"query_wdl": "specialist_db_query"}
    )
    malom_route = _TimedReadOnlyProxy(
        malom,
        sink,
        {
            "query": "malom_position_query",
            "query_move_quality": "malom_move_quality_inner",
        },
    )
    device = torch.device("cuda")
    _cuda_record(device)
    trainer._initialize_training_rngs(plan.seed)
    model, start_game, _best, difficulty, source = trainer._load_model(
        device,
        None,
        plan.policy_hidden,
        start_mode="fresh",
    )
    if (start_game, difficulty, source) != (0, trainer.DIFF_START, "scratch"):
        raise SanmillRouteProbeError("fresh model initialization contract drifted")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frozen = trainer.FrozenModelOpponent(model, device)
    for parameter in frozen._model.parameters():
        parameter.requires_grad_(False)
    advisor = LookaheadAdvisor(
        sentinel=None,
        evaluate_fn=trainer._simple_evaluate,
        value_net=None,
        gap_net=None,
        human_db=human_route,
        use_sentinel=True,
        ply_depth=12,
        sim_ply_depth=5,
        endgame_db=malom_route,
    )
    advisor.set_frozen_model(frozen._model, device=device)
    return ProbeRuntime(
        device=device,
        model=model,
        frozen_opponent=frozen,
        lookahead_advisor=advisor,
        human_db=human,
        specialist_db=specialist,
        malom_db=malom,
        human_route=human_route,
        specialist_route=specialist_route,
        malom_route=malom_route,
        installation=installation,
        timing_sink=sink,
    )


def _outcome_name(outcome: float) -> str:
    if outcome == trainer.WIN_REWARD:
        return "win"
    if outcome == trainer.LOSS_REWARD:
        return "loss"
    if outcome in {trainer.DRAW_SHORT, trainer.DRAW_LONG}:
        return "draw"
    raise SanmillRouteProbeError(f"unknown rollout outcome: {outcome!r}")


def _finite_sequence(values: Sequence[float], *, context: str) -> list[float]:
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise SanmillRouteProbeError(f"{context} contains invalid timing")
    return result


def execute_probe_schedule(
    plan: ProbePlan,
    runtime: ProbeRuntime,
    *,
    game_factory: Callable[..., Any] = SanmillTrainingGame,
    opponent_factory: Callable[..., Any] = SanmillTrainingOpponent,
    rollout_fn: Callable[..., Any] = trainer._rollout,
) -> list[dict[str, Any]]:
    """Execute exactly the frozen schedule through production route seams."""
    samples: list[dict[str, Any]] = []
    for scheduled in plan.schedule:
        collector = _TimingCollector()
        if runtime.timing_sink.current is not None:
            raise SanmillRouteProbeError("probe timing collector leaked between games")
        runtime.timing_sink.current = collector
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
            torch.cuda.reset_peak_memory_stats(runtime.device)
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        try:
            try:
                with game_factory(
                    runtime.installation,
                    seed=plan.seed,
                    timing_observer=runtime.timing_sink.observe,
                ) as game:
                    opponent = (
                        opponent_factory(
                            game,
                            node_budget=scheduled.node_budget,
                            depth=None,
                        )
                        if scheduled.opponent_kind == "sanmill"
                        else runtime.frozen_opponent
                    )
                    result = rollout_fn(
                        model=runtime.model,
                        device=runtime.device,
                        start_board=BoardState.new_game(),
                        learner_color=scheduled.learner_color,
                        opponent=opponent,
                        opp_color=("B" if scheduled.learner_color == "W" else "W"),
                        sentinel=None,
                        value_net=None,
                        temperature=plan.temperature,
                        max_ply=plan.max_ply,
                        record_branches=False,
                        branch_every=0,
                        retry_ply=0,
                        forced_placements=None,
                        lookahead_advisor=runtime.lookahead_advisor,
                        game_difficulty=1,
                        human_db=runtime.human_route,
                        specialist_db=runtime.specialist_route,
                        malom_db=runtime.malom_route,
                        deep_game=(scheduled.route_depth == "deep"),
                        torch_generator=trainer._game_torch_generator(
                            scheduled.torch_seed
                        ),
                        sanmill_game=game,
                        persist_rollout_evidence=False,
                        timing_observer=runtime.timing_sink.observe,
                    )
                    state = game.state
                    state_record = {
                        "fen": state.fen,
                        "history_sha256": state.history_sha256,
                        "logical_ply_count": state.logical_ply_count,
                        "terminal": state.terminal,
                        "winner": state.winner,
                        "outcome_reason_code": state.outcome_reason_code,
                    }
            finally:
                if runtime.device.type == "cuda":
                    torch.cuda.synchronize(runtime.device)
                wall_seconds = time.perf_counter() - wall_started
                cpu_seconds = time.process_time() - cpu_started
                runtime.timing_sink.current = None
        except Exception as exc:
            exception_record: dict[str, Any] = {
                "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
            }
            if isinstance(exc, SanmillBoardMirrorError):
                exception_record["bridge_diagnostic"] = dict(exc.diagnostic)
            raise SanmillRouteProbeScheduleError(
                {
                    "schema_version": SCHEDULE_FAILURE_SCHEMA,
                    "failed_schedule": _probe_game_record(scheduled),
                    "completed_sample_count": len(samples),
                    "completed_samples": _completed_sample_identities(samples),
                    "exception": exception_record,
                    "wall_seconds": wall_seconds,
                    "cpu_seconds": cpu_seconds,
                    "timing_samples_seconds": collector.record(),
                }
            ) from exc
        if not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
            raise SanmillRouteProbeError("probe game wall time is invalid")
        if not math.isfinite(cpu_seconds) or cpu_seconds < 0.0:
            raise SanmillRouteProbeError("probe game CPU time is invalid")
        if result.ply != state_record["logical_ply_count"]:
            raise SanmillRouteProbeError("rollout and Sanmill logical ply drifted")
        if result.ply > plan.max_ply:
            raise SanmillRouteProbeError("probe exceeded max_ply")
        if sum(result.phase_ply_counts.values()) != result.ply:
            raise SanmillRouteProbeError("probe phase counts do not match plies")
        search_observations = [
            dict(item) for item in result.opponent_search_observations
        ]
        if len(search_observations) != result.opponent_search_calls:
            raise SanmillRouteProbeError("probe search observation count drifted")
        if sum(item["nodes"] for item in search_observations) != (
            result.opponent_search_nodes
        ):
            raise SanmillRouteProbeError("probe search node total drifted")
        if scheduled.opponent_kind == "frozen_target" and search_observations:
            raise SanmillRouteProbeError("frozen target unexpectedly searched")
        if scheduled.opponent_kind == "sanmill" and any(
            item["nodes"] > int(scheduled.node_budget or 0)
            for item in search_observations
        ):
            raise SanmillRouteProbeError("Sanmill exceeded a fixed node ceiling")
        timing_samples = {
            stage: _finite_sequence(values, context=stage)
            for stage, values in collector.record().items()
        }
        sample = {
            "scheduled_index": scheduled.scheduled_index,
            "game_id": scheduled.game_id,
            "role": scheduled.role,
            "opponent_kind": scheduled.opponent_kind,
            "node_budget": scheduled.node_budget,
            "learner_color": scheduled.learner_color,
            "route_depth": scheduled.route_depth,
            "sim_ply_depth": scheduled.sim_ply_depth,
            "torch_seed": scheduled.torch_seed,
            "outcome": _outcome_name(result.outcome),
            "termination_reason": result.termination_reason,
            "logical_plies": result.ply,
            "learner_steps": len(result.trajectory),
            "phase_ply_counts": dict(result.phase_ply_counts),
            "compound_turn_count": result.compound_turn_count,
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "timing_samples_seconds": timing_samples,
            "opponent_search_observations": search_observations,
            "opponent_search_nodes": result.opponent_search_nodes,
            "opponent_search_calls": result.opponent_search_calls,
            "opponent_search_depth_sum": result.opponent_search_depth_sum,
            "sanmill_final_state": state_record,
            "process_memory": _process_memory_record(),
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(runtime.device))
                if runtime.device.type == "cuda"
                else None
            ),
            "cuda_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(runtime.device))
                if runtime.device.type == "cuda"
                else None
            ),
        }
        samples.append(sample)
    if [sample["game_id"] for sample in samples] != [
        game.game_id for game in plan.schedule
    ]:
        raise SanmillRouteProbeError("probe schedule was not completed exactly")
    return samples


def nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values or not 0.0 < percentile <= 1.0:
        raise SanmillRouteProbeError("nearest-rank input is invalid")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    checked = _finite_sequence(values, context="summary")
    ordered = sorted(checked)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median,
        "p90_nearest_rank": nearest_rank(ordered, 0.90),
        "max": ordered[-1],
    }


def summarize_probe(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = (
            sample["opponent_kind"],
            sample["node_budget"],
            sample["route_depth"],
            sample["learner_color"],
        )
        groups[key].append(sample)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        opponent_kind, node_budget, route_depth, learner_color = key
        result.append(
            {
                "opponent_kind": opponent_kind,
                "node_budget": node_budget,
                "route_depth": route_depth,
                "learner_color": learner_color,
                "games": len(group),
                "wall_seconds": _summary(
                    [float(sample["wall_seconds"]) for sample in group]
                ),
                "seconds_per_ply": _summary(
                    [
                        float(sample["wall_seconds"])
                        / max(1, int(sample["logical_plies"]))
                        for sample in group
                    ]
                ),
                "logical_plies": _summary(
                    [float(sample["logical_plies"]) for sample in group]
                ),
            }
        )
    return result


def preflight_probe(
    plan_path: str | Path,
    paths_config: str | Path,
    *,
    require_published: bool = True,
    verify_malom_components: bool = True,
    perform_route_check: bool = True,
) -> dict[str, Any]:
    """Read-only readiness audit; never consumes a scheduled probe game."""
    plan = load_probe_plan(plan_path)
    source = inspect_published_source(require_published=require_published)
    plan_record = tracked_plan_record(plan)
    inputs = resolve_probe_inputs(paths_config)
    data = verify_probe_inputs(
        plan, inputs, verify_malom_components=verify_malom_components
    )
    installation = inspect_sanmill_training_installation(inputs.sanmill_checkout)
    runtime = _build_runtime(plan, inputs)
    model_before = model_state_sha256(runtime.model)
    frozen_before = model_state_sha256(runtime.frozen_opponent._model)
    route_check = None
    try:
        host = _host_record(runtime.device)
        if perform_route_check:
            collector = _TimingCollector()
            runtime.timing_sink.current = collector
            try:
                with SanmillTrainingGame(
                    installation,
                    seed=plan.seed,
                    timing_observer=runtime.timing_sink.observe,
                ) as game:
                    result = trainer._rollout(
                        model=runtime.model,
                        device=runtime.device,
                        start_board=BoardState.new_game(),
                        learner_color="W",
                        opponent=runtime.frozen_opponent,
                        opp_color="B",
                        sentinel=None,
                        value_net=None,
                        temperature=plan.temperature,
                        max_ply=2,
                        record_branches=False,
                        branch_every=0,
                        retry_ply=0,
                        lookahead_advisor=runtime.lookahead_advisor,
                        human_db=runtime.human_route,
                        specialist_db=runtime.specialist_route,
                        malom_db=runtime.malom_route,
                        deep_game=False,
                        torch_generator=trainer._game_torch_generator(
                            trainer._derive_game_identity(
                                plan.seed, 1_000_000, "preflight-no-search"
                            )[1]
                        ),
                        sanmill_game=game,
                        persist_rollout_evidence=False,
                        timing_observer=runtime.timing_sink.observe,
                    )
                    route_check = {
                        "logical_plies": result.ply,
                        "termination_reason": result.termination_reason,
                        "opponent_search_calls": result.opponent_search_calls,
                        "sanmill_logical_plies": game.state.logical_ply_count,
                        "timing_stages": sorted(collector.samples),
                    }
            finally:
                runtime.timing_sink.current = None
            if route_check != {
                **route_check,
                "logical_plies": 2,
                "termination_reason": "max-ply-truncation",
                "opponent_search_calls": 0,
                "sanmill_logical_plies": 2,
            }:
                raise SanmillRouteProbeError("no-search route check drifted")
        if model_state_sha256(runtime.model) != model_before:
            raise SanmillRouteProbeError("preflight changed learner weights")
        if model_state_sha256(runtime.frozen_opponent._model) != frozen_before:
            raise SanmillRouteProbeError("preflight changed frozen-target weights")
    finally:
        runtime.close()
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "ready_for_authorized_probe",
        "launch_authorized": False,
        "plan": plan_record,
        "source": source,
        "sanmill": training_installation_record(installation, seed=plan.seed),
        "data": data,
        "model": {
            "start_mode": "fresh",
            "learner_sha256": model_before,
            "frozen_target_sha256": frozen_before,
            "requires_grad": False,
            "optimizer": None,
        },
        "host": host,
        "no_search_route_check": route_check,
        "bounded_work": dict(plan.payload["bounded_work"]),
        "next_gate": "explicit one-run probe launch authority",
    }


def preflight_probe_diagnostic(
    plan_path: str | Path,
    paths_config: str | Path,
    *,
    require_published: bool = True,
    verify_malom_components: bool = True,
    perform_route_check: bool = True,
) -> dict[str, Any]:
    """Audit the one-entry diagnostic without consuming its selected game."""
    diagnostic = load_probe_diagnostic_plan(plan_path)
    parent_report = preflight_probe(
        diagnostic.parent.path,
        paths_config,
        require_published=require_published,
        verify_malom_components=verify_malom_components,
        perform_route_check=perform_route_check,
    )
    effective = diagnostic_probe_plan(diagnostic)
    return {
        **parent_report,
        "schema_version": DIAGNOSTIC_PREFLIGHT_SCHEMA,
        "status": "ready_for_authorized_minimal_diagnostic",
        "launch_authorized": False,
        "plan": tracked_plan_record(effective),
        "parent_probe_plan": parent_report["plan"],
        "selected_schedule_entry": _probe_game_record(diagnostic.selected),
        "bounded_work": dict(diagnostic.payload["bounded_work"]),
        "next_gate": "explicit one-run minimal-diagnostic authority",
    }


def run_probe(
    plan: ProbePlan,
    inputs: ProbeInputs,
    *,
    source: Mapping[str, Any],
    run_id: str,
    invocation: Sequence[str],
) -> dict[str, Any]:
    if (
        not run_id
        or len(run_id) > 128
        or any(character not in _RUN_ID_CHARS for character in run_id)
    ):
        raise SanmillRouteProbeError("probe run_id is invalid")
    data_before = verify_probe_inputs(plan, inputs)
    runtime = _build_runtime(plan, inputs)
    learner_before = model_state_sha256(runtime.model)
    frozen_before = model_state_sha256(runtime.frozen_opponent._model)
    host_before = _host_record(runtime.device)
    started = _utc_now()
    wall_started = time.perf_counter()
    schedule_failure: SanmillRouteProbeScheduleError | None = None
    samples: list[dict[str, Any]] | None = None
    try:
        try:
            samples = execute_probe_schedule(plan, runtime)
        except SanmillRouteProbeScheduleError as exc:
            schedule_failure = exc
        learner_after = model_state_sha256(runtime.model)
        frozen_after = model_state_sha256(runtime.frozen_opponent._model)
        host_after = _host_record(runtime.device)
    finally:
        runtime.close()
    wall_seconds = time.perf_counter() - wall_started
    data_after = verify_probe_inputs(plan, inputs)
    source_after = inspect_published_source(require_published=True)
    sanmill_record = training_installation_record(
        runtime.installation,
        seed=plan.seed,
    )
    if schedule_failure is not None:
        failure_body = {
            "schema_version": FAILURE_SCHEMA,
            "status": "failed_closed",
            "claim_boundary": plan.claim_boundary,
            "run_id": run_id,
            "started_at": started,
            "failed_at": _utc_now(),
            "wall_seconds": wall_seconds,
            "invocation": list(invocation),
            "plan": tracked_plan_record(plan),
            "source_before": dict(source),
            "source_after": source_after,
            "source_unchanged": source_after == dict(source),
            "sanmill": sanmill_record,
            "data_before": data_before,
            "data_after": data_after,
            "data_unchanged": data_after == data_before,
            "model": {
                "learner_before_sha256": learner_before,
                "learner_after_sha256": learner_after,
                "learner_unchanged": learner_after == learner_before,
                "frozen_before_sha256": frozen_before,
                "frozen_after_sha256": frozen_after,
                "frozen_unchanged": frozen_after == frozen_before,
                "requires_grad": False,
                "optimizer_constructed": False,
                "backward_calls": 0,
                "checkpoint_writes": 0,
                "rollout_persistence": False,
            },
            "host_before": host_before,
            "host_after": host_after,
            "bounded_work": dict(plan.payload["bounded_work"]),
            "failure": schedule_failure.diagnostic,
            "interpretation": {
                "completed_measurement": False,
                "training_updates_measured": False,
                "strength_measured": False,
                "retry_authorized": False,
                "training_launch_authorized": False,
                "next_gate": "review failure evidence and authorize a minimal diagnostic",
            },
        }
        failure_report = {
            **failure_body,
            "report_identity": canonical_sha256(failure_body),
        }
        raise SanmillRouteProbeRunFailure(failure_report) from schedule_failure
    if samples is None:
        raise SanmillRouteProbeError("probe returned no samples")
    if learner_after != learner_before:
        raise SanmillRouteProbeError("probe changed learner model state")
    if frozen_after != frozen_before:
        raise SanmillRouteProbeError("probe changed frozen-target state")
    if data_after != data_before:
        raise SanmillRouteProbeError("probe input data identity changed")
    if source_after != dict(source):
        raise SanmillRouteProbeError("probe source identity changed during execution")
    body = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed_no_update_measurement",
        "claim_boundary": plan.claim_boundary,
        "run_id": run_id,
        "started_at": started,
        "completed_at": _utc_now(),
        "wall_seconds": wall_seconds,
        "invocation": list(invocation),
        "plan": tracked_plan_record(plan),
        "source": dict(source),
        "sanmill": sanmill_record,
        "data_before": data_before,
        "data_after": data_after,
        "model": {
            "learner_before_sha256": learner_before,
            "learner_after_sha256": learner_after,
            "frozen_before_sha256": frozen_before,
            "frozen_after_sha256": frozen_after,
            "requires_grad": False,
            "optimizer_constructed": False,
            "backward_calls": 0,
            "checkpoint_writes": 0,
            "rollout_persistence": False,
        },
        "host_before": host_before,
        "host_after": host_after,
        "bounded_work": dict(plan.payload["bounded_work"]),
        "samples": samples,
        "summary": summarize_probe(samples),
        "interpretation": {
            "training_updates_measured": False,
            "strength_measured": False,
            "node_ladder_auto_selected": False,
            "training_launch_authorized": False,
            "next_gate": "review route cost and freeze a separate training design",
        },
    }
    return {**body, "report_identity": canonical_sha256(body)}


def _diagnostic_report(
    report: Mapping[str, Any],
    diagnostic: ProbeDiagnosticPlan,
    *,
    outcome: str,
) -> dict[str, Any]:
    body = dict(report)
    body.pop("report_identity", None)
    body["diagnostic"] = {
        "schema_version": "nmm.sanmill-route-diagnostic-binding.v1",
        "outcome": outcome,
        "parent_probe_plan": tracked_plan_record(diagnostic.parent),
        "selected_schedule_entry": _probe_game_record(diagnostic.selected),
        "historical_failure_index_known": False,
    }
    interpretation = dict(body.get("interpretation", {}))
    interpretation.update(
        {
            "completed_measurement": False,
            "training_updates_measured": False,
            "strength_measured": False,
            "node_ladder_auto_selected": False,
            "retry_authorized": False,
            "training_launch_authorized": False,
            "next_gate": (
                "review the captured mirror diagnostic"
                if outcome == "failed_closed_with_diagnostic"
                else "record that parent schedule index zero did not reproduce"
            ),
        }
    )
    body["interpretation"] = interpretation
    return {**body, "report_identity": canonical_sha256(body)}


def run_probe_diagnostic(
    diagnostic: ProbeDiagnosticPlan,
    inputs: ProbeInputs,
    *,
    source: Mapping[str, Any],
    run_id: str,
    invocation: Sequence[str],
) -> dict[str, Any]:
    """Run exactly parent schedule index zero through the existing route."""
    effective = diagnostic_probe_plan(diagnostic)
    try:
        report = run_probe(
            effective,
            inputs,
            source=source,
            run_id=run_id,
            invocation=invocation,
        )
    except SanmillRouteProbeRunFailure as exc:
        raise SanmillRouteProbeRunFailure(
            _diagnostic_report(
                exc.report,
                diagnostic,
                outcome="failed_closed_with_diagnostic",
            )
        ) from exc
    return _diagnostic_report(
        report,
        diagnostic,
        outcome="selected_entry_completed_without_mirror_mismatch",
    )


def validate_probe_output(path: str | Path) -> Path:
    target = Path(path).resolve(strict=False)
    diagnostics = (_ROOT / "out" / "diagnostics").resolve(strict=False)
    try:
        target.relative_to(diagnostics)
    except ValueError as exc:
        raise SanmillRouteProbeError(
            "probe output must be under out/diagnostics"
        ) from exc
    if target.suffix.lower() != ".json":
        raise SanmillRouteProbeError("probe output must be a JSON file")
    if target.exists():
        raise FileExistsError(f"probe result already exists: {target}")
    return target


def probe_failure_output(path: str | Path) -> Path:
    completed = Path(path).resolve(strict=False)
    return completed.with_name(f"{completed.stem}.failure.json")


def _validate_completed_report(report: Mapping[str, Any], plan: ProbePlan) -> None:
    if report.get("schema_version") != RESULT_SCHEMA:
        raise SanmillRouteProbeError("probe result schema is invalid")
    if report.get("status") != "completed_no_update_measurement":
        raise SanmillRouteProbeError("probe result is not complete")
    samples = report.get("samples")
    if not isinstance(samples, list) or [
        sample.get("game_id") for sample in samples
    ] != [game.game_id for game in plan.schedule]:
        raise SanmillRouteProbeError("probe result schedule is incomplete")
    identity = report.get("report_identity")
    body = dict(report)
    body.pop("report_identity", None)
    if identity != canonical_sha256(body):
        raise SanmillRouteProbeError("probe result identity is invalid")


def _validate_failure_report(report: Mapping[str, Any], plan: ProbePlan) -> None:
    if report.get("schema_version") != FAILURE_SCHEMA:
        raise SanmillRouteProbeError("probe failure schema is invalid")
    if report.get("status") != "failed_closed":
        raise SanmillRouteProbeError("probe failure status is invalid")
    plan_record = report.get("plan")
    if not isinstance(plan_record, Mapping) or plan_record.get("identity") != (
        plan.identity
    ):
        raise SanmillRouteProbeError("probe failure plan identity is invalid")
    failure = report.get("failure")
    if (
        not isinstance(failure, Mapping)
        or failure.get("schema_version") != SCHEDULE_FAILURE_SCHEMA
    ):
        raise SanmillRouteProbeError("probe failure context is missing")
    completed = failure.get("completed_samples")
    completed_count = failure.get("completed_sample_count")
    if (
        not isinstance(completed, list)
        or not isinstance(completed_count, int)
        or isinstance(completed_count, bool)
        or completed_count != len(completed)
        or not 0 <= completed_count < len(plan.schedule)
    ):
        raise SanmillRouteProbeError("probe failure prefix is invalid")
    expected_prefix = [
        {
            "scheduled_index": game.scheduled_index,
            "game_id": game.game_id,
        }
        for game in plan.schedule[:completed_count]
    ]
    observed_prefix = [
        {
            "scheduled_index": item.get("scheduled_index"),
            "game_id": item.get("game_id"),
        }
        for item in completed
        if isinstance(item, Mapping)
    ]
    if observed_prefix != expected_prefix or any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("sample_identity"), str)
        or len(item["sample_identity"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in item["sample_identity"]
        )
        for item in completed
    ):
        raise SanmillRouteProbeError("probe failure prefix identity is invalid")
    if failure.get("failed_schedule") != _probe_game_record(
        plan.schedule[completed_count]
    ):
        raise SanmillRouteProbeError("probe failed schedule entry is invalid")
    identity = report.get("report_identity")
    body = dict(report)
    body.pop("report_identity", None)
    if identity != canonical_sha256(body):
        raise SanmillRouteProbeError("probe failure identity is invalid")


def _atomic_publish_report(path: Path, report: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"probe evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"probe evidence already exists: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_probe_result(
    path: str | Path,
    report: Mapping[str, Any],
    plan: ProbePlan,
) -> None:
    """Atomically publish a complete result without replacing evidence."""
    _validate_completed_report(report, plan)
    target = Path(path)
    _atomic_publish_report(target, report)


def publish_probe_failure(
    path: str | Path,
    report: Mapping[str, Any],
    plan: ProbePlan,
) -> None:
    """Atomically quarantine a failed run without creating a result."""
    _validate_failure_report(report, plan)
    _atomic_publish_report(Path(path), report)
