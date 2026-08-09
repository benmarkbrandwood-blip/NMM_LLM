"""Multi-seed no-update capture of production-shaped auxiliary batches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from game.board import BoardState
from learned_ai.training.run_contract import canonical_sha256
from learned_ai.training.sanmill_referee import (
    SanmillTrainingGame,
    SanmillTrainingOpponent,
    training_installation_record,
)
from learned_ai.training.scaffolded_a2c import ScaffoldedStep
from learned_ai.validation import sanmill_route_probe as route_probe
from learned_ai.validation.malom_policy_auxiliary_gradient_interaction import (
    measure_malom_policy_auxiliary_batch_gradients,
)
from scripts import train_s_gen_v2 as trainer


PLAN_SCHEMA = "nmm.sanmill-malom-policy-auxiliary-no-update-batch-capture-plan.v1"
PREFLIGHT_SCHEMA = (
    "nmm.sanmill-malom-policy-auxiliary-no-update-batch-capture-preflight.v1"
)
RESULT_SCHEMA = "nmm.sanmill-malom-policy-auxiliary-no-update-batch-capture-result.v1"
FAILURE_SCHEMA = "nmm.sanmill-malom-policy-auxiliary-no-update-batch-capture-failure.v1"
EXPECTED_PLAN_IDENTITY = (
    "a5c85ed13baecf3efed6780effdf590e97560a12e9ab197c5fc7bb4bf7341fab"
)
DEFAULT_PLAN_RELATIVE = Path(
    "docs/experiments/sanmill-malom-policy-auxiliary-no-update-batch-capture-v1.json"
)
DEFAULT_PATHS_RELATIVE = route_probe.DEFAULT_PATHS_RELATIVE

_ROOT = Path(__file__).resolve().parents[2]
_PLAN_KEYS = {
    "schema_version",
    "status",
    "experiment_id",
    "claim_boundary",
    "source_evidence",
    "parent_route_contract",
    "model_route",
    "schedule_contract",
    "batch_contract",
    "gradient_measurement",
    "bounded_work",
    "decision_rules",
    "plan_identity",
}
_PATTERN_KEYS = {"opponent_kind", "node_budget", "learner_color"}
_RUN_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._-")


class MalomPolicyAuxiliaryBatchCaptureError(ValueError):
    """Raised when a batch-capture contract or result fails closed."""


class MalomPolicyAuxiliaryBatchCaptureExecutionError(
    MalomPolicyAuxiliaryBatchCaptureError
):
    """Carries the exact completed prefix when execution fails."""

    def __init__(self, diagnostic: Mapping[str, Any]) -> None:
        super().__init__("batch capture failed closed")
        self.diagnostic = dict(diagnostic)


class MalomPolicyAuxiliaryBatchCaptureRunFailure(MalomPolicyAuxiliaryBatchCaptureError):
    """Carries a publishable failure report."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        super().__init__("batch capture run failed closed")
        self.report = dict(report)


@dataclass(frozen=True)
class CaptureGame:
    scheduled_index: int
    seed: int
    seed_game_index: int
    game_id: str
    role: str
    opponent_kind: str
    node_budget: int | None
    learner_color: str
    route_depth: str
    sim_ply_depth: int
    torch_seed: int


@dataclass(frozen=True)
class CapturePlan:
    path: Path
    raw_sha256: str
    identity: str
    experiment_id: str
    parent: route_probe.ProbePlan
    seeds: tuple[int, ...]
    schedule: tuple[CaptureGame, ...]
    temperature: float
    max_ply: int
    policy_hidden: tuple[int, ...]
    periodic_threshold: int
    final_flush_minimum: int
    target_ratios: tuple[float, ...]
    gamma: float
    entropy_coef: float
    value_coef: float
    denominator_floor: float
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class PendingContribution:
    game: Mapping[str, Any]
    steps: int


@dataclass(frozen=True)
class PendingBatch:
    reason: str
    steps: tuple[ScaffoldedStep, ...]
    contributions: tuple[PendingContribution, ...]


class ProductionBatchAccumulator:
    """Reproduce the trainer's whole-game update threshold exactly."""

    def __init__(self, *, threshold: int, final_minimum: int) -> None:
        if threshold <= 0 or final_minimum <= 0 or final_minimum > threshold:
            raise MalomPolicyAuxiliaryBatchCaptureError(
                "production batch thresholds are invalid"
            )
        self._threshold = threshold
        self._final_minimum = final_minimum
        self._steps: list[ScaffoldedStep] = []
        self._contributions: list[PendingContribution] = []

    @property
    def pending_steps(self) -> int:
        return len(self._steps)

    def append_game(
        self,
        game: Mapping[str, Any],
        steps: Sequence[ScaffoldedStep],
    ) -> PendingBatch | None:
        if not steps:
            return None
        self._steps.extend(steps)
        self._contributions.append(
            PendingContribution(game=dict(game), steps=len(steps))
        )
        if len(self._steps) < self._threshold:
            return None
        return self._take("periodic")

    def finish(self) -> tuple[PendingBatch | None, Mapping[str, Any] | None]:
        if not self._steps:
            return None, None
        if len(self._steps) >= self._final_minimum:
            return self._take("final_flush"), None
        excluded = {
            "reason": "below_final_flush_minimum",
            "steps": len(self._steps),
            "minimum_steps": self._final_minimum,
            "contributions": [
                _contribution_record(item) for item in self._contributions
            ],
        }
        self._steps.clear()
        self._contributions.clear()
        return None, excluded

    def _take(self, reason: str) -> PendingBatch:
        batch = PendingBatch(
            reason=reason,
            steps=tuple(self._steps),
            contributions=tuple(self._contributions),
        )
        self._steps.clear()
        self._contributions.clear()
        return batch


def _strict_json(path: Path) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MalomPolicyAuxiliaryBatchCaptureError(
                    f"duplicate JSON key {key!r} in {path}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            f"cannot read batch-capture JSON: {path}"
        ) from exc
    if not isinstance(value, Mapping):
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture JSON must be an object"
        )
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            f"{context} has wrong members: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, *, context: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MalomPolicyAuxiliaryBatchCaptureError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise MalomPolicyAuxiliaryBatchCaptureError(
            f"{context} must be finite" + (" and positive" if positive else "")
        )
    return result


def _positive_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            f"{context} must be a positive integer"
        )
    return value


def _game_record(game: CaptureGame) -> dict[str, Any]:
    return {
        "scheduled_index": game.scheduled_index,
        "seed": game.seed,
        "seed_game_index": game.seed_game_index,
        "game_id": game.game_id,
        "role": game.role,
        "opponent_kind": game.opponent_kind,
        "node_budget": game.node_budget,
        "learner_color": game.learner_color,
        "route_depth": game.route_depth,
        "sim_ply_depth": game.sim_ply_depth,
        "torch_seed": game.torch_seed,
    }


def _contribution_record(item: PendingContribution) -> dict[str, Any]:
    return {
        "game_id": item.game["game_id"],
        "opponent_kind": item.game["opponent_kind"],
        "learner_color": item.game["learner_color"],
        "termination_reason": item.game["termination_reason"],
        "steps": item.steps,
    }


def load_batch_capture_plan(path: str | Path) -> CapturePlan:
    """Load and fully validate the immutable multi-seed capture plan."""
    plan_path = Path(path)
    raw = plan_path.read_bytes()
    payload = _strict_json(plan_path)
    _require_exact_keys(payload, _PLAN_KEYS, context="batch-capture plan")
    if payload["schema_version"] != PLAN_SCHEMA:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "unsupported batch-capture plan schema"
        )
    if payload["status"] != "prepared_unlaunched":
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture plan is not prepared/unlaunched"
        )
    identity = payload["plan_identity"]
    identity_body = dict(payload)
    identity_body.pop("plan_identity")
    if (
        identity != EXPECTED_PLAN_IDENTITY
        or canonical_sha256(identity_body) != identity
    ):
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture plan identity mismatch"
        )

    parent_record = payload["parent_route_contract"]
    if not isinstance(parent_record, Mapping) or set(parent_record) != {
        "path",
        "raw_sha256",
        "plan_identity",
    }:
        raise MalomPolicyAuxiliaryBatchCaptureError("parent route contract is invalid")
    if parent_record["path"] != route_probe.DEFAULT_PLAN_RELATIVE.as_posix():
        raise MalomPolicyAuxiliaryBatchCaptureError("parent route path drifted")
    parent = route_probe.load_probe_plan(_ROOT / parent_record["path"])
    if (
        parent.raw_sha256 != parent_record["raw_sha256"]
        or parent.identity != parent_record["plan_identity"]
    ):
        raise MalomPolicyAuxiliaryBatchCaptureError("parent route identity drifted")

    model_route = payload["model_route"]
    expected_model_route = {
        "batch_games": 1,
        "checkpoint": None,
        "cuda_required": True,
        "disabled_components": list(
            parent.payload["model_route"]["disabled_components"]
        ),
        "malom_policy_label_capture_trigger": 1.0,
        "max_ply": 120,
        "mill_bonus_mode": "malom-preserving-only",
        "optimizer": None,
        "policy_hidden": [512, 256, 128],
        "rollout_persistence": False,
        "sim_ply_depth_deep": 12,
        "sim_ply_depth_normal": 5,
        "start_mode": "fresh",
        "temperature": 0.9,
    }
    if model_route != expected_model_route:
        raise MalomPolicyAuxiliaryBatchCaptureError("batch-capture model route drifted")

    schedule_contract = payload["schedule_contract"]
    if not isinstance(schedule_contract, Mapping) or set(schedule_contract) != {
        "seeds",
        "games_per_seed",
        "deep_local_index_by_seed",
        "pattern",
        "per_seed",
    }:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture schedule contract is invalid"
        )
    seeds = tuple(schedule_contract["seeds"])
    if seeds != (52, 53, 54):
        raise MalomPolicyAuxiliaryBatchCaptureError("batch-capture seeds drifted")
    pattern = schedule_contract["pattern"]
    if not isinstance(pattern, list) or len(pattern) != 20:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture pattern must contain 20 games"
        )
    if schedule_contract["games_per_seed"] != len(pattern):
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "games-per-seed differs from the pattern"
        )
    deep_by_seed = schedule_contract["deep_local_index_by_seed"]
    if deep_by_seed != {"52": 0, "53": 1, "54": 2}:
        raise MalomPolicyAuxiliaryBatchCaptureError("deep-route rotation drifted")

    schedule: list[CaptureGame] = []
    for seed in seeds:
        for local_index, entry in enumerate(pattern):
            if not isinstance(entry, Mapping):
                raise MalomPolicyAuxiliaryBatchCaptureError(
                    "batch-capture pattern entry must be an object"
                )
            _require_exact_keys(
                entry,
                _PATTERN_KEYS,
                context=f"schedule pattern {local_index}",
            )
            opponent_kind = entry["opponent_kind"]
            if opponent_kind not in {"sanmill", "frozen_target"}:
                raise MalomPolicyAuxiliaryBatchCaptureError(
                    "batch-capture opponent kind is invalid"
                )
            budget = entry["node_budget"]
            if opponent_kind == "sanmill":
                if budget != 1_000:
                    raise MalomPolicyAuxiliaryBatchCaptureError(
                        "batch capture must use the observed 1,000-node level"
                    )
            elif budget is not None:
                raise MalomPolicyAuxiliaryBatchCaptureError(
                    "frozen target must not have a node budget"
                )
            learner_color = entry["learner_color"]
            if learner_color not in {"W", "B"}:
                raise MalomPolicyAuxiliaryBatchCaptureError(
                    "batch-capture learner color is invalid"
                )
            route_depth = "deep" if local_index == deep_by_seed[str(seed)] else "normal"
            sim_ply_depth = 12 if route_depth == "deep" else 5
            budget_label = str(budget) if budget is not None else "none"
            role = (
                f"batch-capture-{opponent_kind}-{budget_label}-"
                f"{route_depth}-{learner_color}-{local_index:02d}"
            )
            game_id, torch_seed = trainer._derive_game_identity(
                seed,
                local_index,
                role,
            )
            schedule.append(
                CaptureGame(
                    scheduled_index=len(schedule),
                    seed=seed,
                    seed_game_index=local_index,
                    game_id=game_id,
                    role=role,
                    opponent_kind=opponent_kind,
                    node_budget=budget,
                    learner_color=learner_color,
                    route_depth=route_depth,
                    sim_ply_depth=sim_ply_depth,
                    torch_seed=torch_seed,
                )
            )

    expected_per_seed = {
        "deep_games": 1,
        "frozen_target_black": 6,
        "frozen_target_games": 12,
        "frozen_target_white": 6,
        "normal_games": 19,
        "sanmill_black": 4,
        "sanmill_games": 8,
        "sanmill_white": 4,
    }
    if schedule_contract["per_seed"] != expected_per_seed:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture per-seed matrix drifted"
        )
    for seed in seeds:
        selected = [game for game in schedule if game.seed == seed]
        observed = {
            "deep_games": sum(game.route_depth == "deep" for game in selected),
            "frozen_target_black": sum(
                game.opponent_kind == "frozen_target" and game.learner_color == "B"
                for game in selected
            ),
            "frozen_target_games": sum(
                game.opponent_kind == "frozen_target" for game in selected
            ),
            "frozen_target_white": sum(
                game.opponent_kind == "frozen_target" and game.learner_color == "W"
                for game in selected
            ),
            "normal_games": sum(game.route_depth == "normal" for game in selected),
            "sanmill_black": sum(
                game.opponent_kind == "sanmill" and game.learner_color == "B"
                for game in selected
            ),
            "sanmill_games": sum(game.opponent_kind == "sanmill" for game in selected),
            "sanmill_white": sum(
                game.opponent_kind == "sanmill" and game.learner_color == "W"
                for game in selected
            ),
        }
        if observed != expected_per_seed:
            raise MalomPolicyAuxiliaryBatchCaptureError(
                f"batch-capture seed {seed} matrix drifted"
            )

    batch_contract = payload["batch_contract"]
    expected_batch = {
        "draw_penalty_scale": 1.0,
        "final_flush_minimum_steps": 8,
        "preserve_complete_game_trajectories": True,
        "periodic_update_threshold_steps": 64,
        "production_cadence_note": (
            "append each complete primary trajectory, measure and clear after "
            "the accumulated batch reaches 64 steps, then measure a final "
            "residual only when it has at least 8 steps"
        ),
        "retroactive_rescore": True,
    }
    if batch_contract != expected_batch:
        raise MalomPolicyAuxiliaryBatchCaptureError("production batch contract drifted")

    measurement = payload["gradient_measurement"]
    if not isinstance(measurement, Mapping) or set(measurement) != {
        "denominator_floor",
        "entropy_coefficient",
        "gamma_td",
        "report_dimensions",
        "target_policy_head_ratios",
        "value_coefficient",
    }:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "gradient measurement contract is invalid"
        )
    target_ratios = tuple(
        _finite(value, context="target ratio", positive=True)
        for value in measurement["target_policy_head_ratios"]
    )
    if target_ratios != (0.25, 0.5, 1.0):
        raise MalomPolicyAuxiliaryBatchCaptureError("gradient target grid drifted")
    if measurement["report_dimensions"] != [
        "seed",
        "batch",
        "board phase",
        "opponent source",
        "learner colour",
        "termination reason",
    ]:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "gradient report dimensions drifted"
        )

    bounded = {
        "complete_games": len(schedule),
        "fresh_seeds": len(seeds),
        "frozen_target_games": sum(
            game.opponent_kind == "frozen_target" for game in schedule
        ),
        "maximum_active_seconds": 7_200,
        "maximum_gradient_batches": 33,
        "maximum_logical_plies": len(schedule) * 120,
        "maximum_requested_search_node_ceilings": sum(
            (game.node_budget or 0) * 60 for game in schedule
        ),
        "maximum_search_calls": sum(
            60 for game in schedule if game.opponent_kind == "sanmill"
        ),
        "sanmill_games": sum(game.opponent_kind == "sanmill" for game in schedule),
    }
    if payload["bounded_work"] != bounded:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture bounded work drifted"
        )
    if payload["decision_rules"] != {
        "automatic_extension": False,
        "automatic_retry": False,
        "execution_requires_explicit_authority": True,
        "publish_only_complete_schedule": True,
        "refuse_output_overwrite": True,
        "select_coefficient_or_target": False,
        "stop_all_seeds_on_any_failure": True,
        "training_launch": False,
    }:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture decision rules drifted"
        )
    if payload["claim_boundary"] != {
        "candidate_checkpoint_loaded": False,
        "coefficient_or_normalization_selected": False,
        "fresh_random_models": 3,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "strength_or_promotion_claim": False,
        "training_updates": 0,
    }:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture claim boundary drifted"
        )

    return CapturePlan(
        path=plan_path,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        identity=str(identity),
        experiment_id=str(payload["experiment_id"]),
        parent=parent,
        seeds=seeds,
        schedule=tuple(schedule),
        temperature=float(model_route["temperature"]),
        max_ply=int(model_route["max_ply"]),
        policy_hidden=tuple(model_route["policy_hidden"]),
        periodic_threshold=int(batch_contract["periodic_update_threshold_steps"]),
        final_flush_minimum=int(batch_contract["final_flush_minimum_steps"]),
        target_ratios=target_ratios,
        gamma=_finite(measurement["gamma_td"], context="gamma"),
        entropy_coef=_finite(
            measurement["entropy_coefficient"],
            context="entropy coefficient",
        ),
        value_coef=_finite(
            measurement["value_coefficient"],
            context="value coefficient",
        ),
        denominator_floor=_finite(
            measurement["denominator_floor"],
            context="denominator floor",
            positive=True,
        ),
        payload=payload,
    )


def _runtime_for_seed(
    plan: CapturePlan,
    inputs: route_probe.ProbeInputs,
    seed: int,
) -> route_probe.ProbeRuntime:
    parent_view = replace(plan.parent, seed=seed)
    return route_probe._build_runtime(parent_view, inputs)


def _outcome_name(outcome: float) -> str:
    if outcome == trainer.WIN_REWARD:
        return "win"
    if outcome == trainer.LOSS_REWARD:
        return "loss"
    if outcome in {trainer.DRAW_SHORT, trainer.DRAW_LONG}:
        return "draw"
    raise MalomPolicyAuxiliaryBatchCaptureError(f"unknown rollout outcome: {outcome!r}")


def _validate_labelled_trajectory(steps: Sequence[ScaffoldedStep]) -> None:
    for index, step in enumerate(steps):
        mask = getattr(step, "malom_preserving_mask", None)
        if mask is None:
            raise MalomPolicyAuxiliaryBatchCaptureError(
                f"trajectory step {index} is missing an exact Malom label"
            )
        value = np.asarray(mask)
        legal_count = len(np.asarray(step.move_features))
        if (
            value.dtype != np.bool_
            or value.ndim != 1
            or len(value) != legal_count
            or not bool(value.any())
        ):
            raise MalomPolicyAuxiliaryBatchCaptureError(
                f"trajectory step {index} has an invalid Malom label set"
            )


def _execute_game(
    plan: CapturePlan,
    scheduled: CaptureGame,
    runtime: route_probe.ProbeRuntime,
    *,
    game_factory: Callable[..., Any],
    opponent_factory: Callable[..., Any],
    rollout_fn: Callable[..., Any],
) -> tuple[list[ScaffoldedStep], dict[str, Any]]:
    wall_started = time.perf_counter()
    with game_factory(
        runtime.installation,
        seed=scheduled.seed,
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
            torch_generator=trainer._game_torch_generator(scheduled.torch_seed),
            sanmill_game=game,
            persist_rollout_evidence=False,
            mill_bonus_mode=plan.payload["model_route"]["mill_bonus_mode"],
            malom_policy_aux_coef=plan.payload["model_route"][
                "malom_policy_label_capture_trigger"
            ],
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
    wall_seconds = time.perf_counter() - wall_started
    if not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture game wall time is invalid"
        )
    if result.ply != state_record["logical_ply_count"]:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "rollout and Sanmill logical ply drifted"
        )
    if result.ply > plan.max_ply:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture game exceeded max_ply"
        )
    if sum(result.phase_ply_counts.values()) != result.ply:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture phase counts do not match logical plies"
        )
    observations = [dict(value) for value in result.opponent_search_observations]
    if len(observations) != result.opponent_search_calls:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture search observation count drifted"
        )
    if sum(value["nodes"] for value in observations) != result.opponent_search_nodes:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture search node total drifted"
        )
    if scheduled.opponent_kind == "frozen_target" and observations:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "frozen target unexpectedly used Sanmill search"
        )
    if scheduled.opponent_kind == "sanmill" and any(
        value["nodes"] > int(scheduled.node_budget or 0) for value in observations
    ):
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "Sanmill exceeded the frozen node ceiling"
        )

    trainer._retroactive_rescore(
        result.trajectory,
        result.step_diags,
        result.outcome,
        float(plan.payload["batch_contract"]["draw_penalty_scale"]),
    )
    _validate_labelled_trajectory(result.trajectory)
    informative_steps = sum(
        not bool(np.asarray(step.malom_preserving_mask).all())
        for step in result.trajectory
    )
    sample_body = {
        **_game_record(scheduled),
        "outcome": _outcome_name(result.outcome),
        "termination_reason": result.termination_reason,
        "logical_plies": result.ply,
        "learner_steps": len(result.trajectory),
        "malom_labelled_steps": len(result.trajectory),
        "malom_informative_steps": informative_steps,
        "phase_ply_counts": dict(result.phase_ply_counts),
        "compound_turn_count": result.compound_turn_count,
        "wall_seconds": wall_seconds,
        "opponent_search_observations": observations,
        "opponent_search_nodes": result.opponent_search_nodes,
        "opponent_search_calls": result.opponent_search_calls,
        "opponent_search_depth_sum": result.opponent_search_depth_sum,
        "sanmill_final_state": state_record,
    }
    sample = {
        **sample_body,
        "sample_identity": canonical_sha256(sample_body),
    }
    return list(result.trajectory), sample


def _dimension_support(
    contributions: Sequence[PendingContribution],
    key: str,
) -> list[dict[str, Any]]:
    games: Counter[str] = Counter()
    steps: Counter[str] = Counter()
    for item in contributions:
        value = str(item.game[key])
        games[value] += 1
        steps[value] += item.steps
    return [
        {
            "value": value,
            "games": games[value],
            "steps": steps[value],
        }
        for value in sorted(games)
    ]


def _measure_batch(
    plan: CapturePlan,
    runtime: route_probe.ProbeRuntime,
    pending: PendingBatch,
    *,
    seed: int,
    seed_batch_index: int,
    global_batch_index: int,
    measurement_fn: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    measurement = dict(
        measurement_fn(
            runtime.model,
            pending.steps,
            device=runtime.device,
            target_policy_head_ratios=plan.target_ratios,
            gamma=plan.gamma,
            entropy_coef=plan.entropy_coef,
            value_coef=plan.value_coef,
            denominator_floor=plan.denominator_floor,
        )
    )
    contributions = [_contribution_record(item) for item in pending.contributions]
    if sum(item["steps"] for item in contributions) != len(pending.steps):
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch contribution step count drifted"
        )
    body = {
        "global_batch_index": global_batch_index,
        "seed": seed,
        "seed_batch_index": seed_batch_index,
        "reason": pending.reason,
        "steps": len(pending.steps),
        "contributions": contributions,
        "support_by_opponent_source": _dimension_support(
            pending.contributions,
            "opponent_kind",
        ),
        "support_by_learner_color": _dimension_support(
            pending.contributions,
            "learner_color",
        ),
        "support_by_termination_reason": _dimension_support(
            pending.contributions,
            "termination_reason",
        ),
        "gradient_measurement": measurement,
    }
    return {**body, "batch_identity": canonical_sha256(body)}


def execute_batch_capture(
    plan: CapturePlan,
    inputs: route_probe.ProbeInputs,
    *,
    runtime_factory: Callable[
        [CapturePlan, route_probe.ProbeInputs, int],
        route_probe.ProbeRuntime,
    ] = _runtime_for_seed,
    game_factory: Callable[..., Any] = SanmillTrainingGame,
    opponent_factory: Callable[..., Any] = SanmillTrainingOpponent,
    rollout_fn: Callable[..., Any] = trainer._rollout,
    measurement_fn: Callable[..., Mapping[str, Any]] = (
        measure_malom_policy_auxiliary_batch_gradients
    ),
) -> dict[str, Any]:
    """Execute the frozen schedule without constructing an optimizer."""
    samples: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    excluded_residuals: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    wall_started = time.perf_counter()
    maximum_seconds = float(plan.payload["bounded_work"]["maximum_active_seconds"])

    for seed in plan.seeds:
        scheduled_games = [game for game in plan.schedule if game.seed == seed]
        runtime: route_probe.ProbeRuntime | None = None
        current: CaptureGame | None = None
        stage = "runtime_initialization"
        try:
            runtime = runtime_factory(plan, inputs, seed)
            learner_before = route_probe.model_state_sha256(runtime.model)
            frozen_before = route_probe.model_state_sha256(
                runtime.frozen_opponent._model
            )
            accumulator = ProductionBatchAccumulator(
                threshold=plan.periodic_threshold,
                final_minimum=plan.final_flush_minimum,
            )
            seed_batch_index = 0
            for current in scheduled_games:
                stage = "rollout"
                if time.perf_counter() - wall_started > maximum_seconds:
                    raise MalomPolicyAuxiliaryBatchCaptureError(
                        "batch-capture active-time envelope was reached"
                    )
                steps, sample = _execute_game(
                    plan,
                    current,
                    runtime,
                    game_factory=game_factory,
                    opponent_factory=opponent_factory,
                    rollout_fn=rollout_fn,
                )
                if time.perf_counter() - wall_started > maximum_seconds:
                    raise MalomPolicyAuxiliaryBatchCaptureError(
                        "batch-capture active-time envelope was reached"
                    )
                samples.append(sample)
                pending = accumulator.append_game(sample, steps)
                if pending is not None:
                    stage = "gradient_measurement"
                    batches.append(
                        _measure_batch(
                            plan,
                            runtime,
                            pending,
                            seed=seed,
                            seed_batch_index=seed_batch_index,
                            global_batch_index=len(batches),
                            measurement_fn=measurement_fn,
                        )
                    )
                    seed_batch_index += 1
                    if time.perf_counter() - wall_started > maximum_seconds:
                        raise MalomPolicyAuxiliaryBatchCaptureError(
                            "batch-capture active-time envelope was reached"
                        )

            stage = "final_flush_measurement"
            final_batch, excluded = accumulator.finish()
            if final_batch is not None:
                batches.append(
                    _measure_batch(
                        plan,
                        runtime,
                        final_batch,
                        seed=seed,
                        seed_batch_index=seed_batch_index,
                        global_batch_index=len(batches),
                        measurement_fn=measurement_fn,
                    )
                )
                if time.perf_counter() - wall_started > maximum_seconds:
                    raise MalomPolicyAuxiliaryBatchCaptureError(
                        "batch-capture active-time envelope was reached"
                    )
            if excluded is not None:
                excluded_body = {"seed": seed, **excluded}
                excluded_residuals.append(
                    {
                        **excluded_body,
                        "residual_identity": canonical_sha256(excluded_body),
                    }
                )

            learner_after = route_probe.model_state_sha256(runtime.model)
            frozen_after = route_probe.model_state_sha256(
                runtime.frozen_opponent._model
            )
            model_record = {
                "seed": seed,
                "learner_before_sha256": learner_before,
                "learner_after_sha256": learner_after,
                "learner_unchanged": learner_after == learner_before,
                "frozen_before_sha256": frozen_before,
                "frozen_after_sha256": frozen_after,
                "frozen_unchanged": frozen_after == frozen_before,
                "learner_matches_initial_frozen": learner_before == frozen_before,
                "learner_requires_grad_after": any(
                    parameter.requires_grad for parameter in runtime.model.parameters()
                ),
                "learner_gradients_populated": any(
                    parameter.grad is not None
                    for parameter in runtime.model.parameters()
                ),
            }
            if model_record != {
                **model_record,
                "learner_unchanged": True,
                "frozen_unchanged": True,
                "learner_matches_initial_frozen": True,
                "learner_requires_grad_after": False,
                "learner_gradients_populated": False,
            }:
                raise MalomPolicyAuxiliaryBatchCaptureError(
                    "batch-capture model mutation check failed"
                )
            model_records.append(model_record)
        except Exception as exc:
            failed_game = _game_record(current) if current is not None else None
            diagnostic = {
                "stage": stage,
                "seed": seed,
                "failed_game": failed_game,
                "completed_games": [
                    {
                        "scheduled_index": sample["scheduled_index"],
                        "game_id": sample["game_id"],
                        "sample_identity": sample["sample_identity"],
                    }
                    for sample in samples
                ],
                "completed_batches": [
                    {
                        "global_batch_index": batch["global_batch_index"],
                        "batch_identity": batch["batch_identity"],
                    }
                    for batch in batches
                ],
                "exception": {
                    "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "message": str(exc),
                },
            }
            raise MalomPolicyAuxiliaryBatchCaptureExecutionError(diagnostic) from exc
        finally:
            if runtime is not None:
                runtime.close()

    if [sample["game_id"] for sample in samples] != [
        game.game_id for game in plan.schedule
    ]:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture schedule was not completed exactly"
        )
    if not batches or len(batches) > int(
        plan.payload["bounded_work"]["maximum_gradient_batches"]
    ):
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture gradient-batch count is invalid"
        )
    return {
        "wall_seconds": time.perf_counter() - wall_started,
        "samples": samples,
        "batches": batches,
        "excluded_residuals": excluded_residuals,
        "models": model_records,
    }


def tracked_plan_record(plan: CapturePlan) -> dict[str, str]:
    try:
        relative = plan.path.resolve().relative_to(_ROOT).as_posix()
    except ValueError as exc:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture plan must be inside this repository"
        ) from exc
    route_probe._git("ls-files", "--error-unmatch", "--", relative)
    return {
        "relative_path": relative,
        "raw_sha256": plan.raw_sha256,
        "identity": plan.identity,
    }


def verify_source_evidence(plan: CapturePlan) -> dict[str, Any]:
    evidence = plan.payload["source_evidence"]
    expected_keys = {
        "calibration_plan_identity",
        "calibration_result_path",
        "calibration_result_identity",
        "calibration_result_sha256",
        "gradient_interaction_report_path",
        "gradient_interaction_audit_identity",
        "gradient_interaction_report_sha256",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_keys:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture source-evidence contract is invalid"
        )
    records: dict[str, Any] = {}
    for prefix, identity_key in (
        ("calibration_result", "result_identity"),
        ("gradient_interaction_report", "audit_identity"),
    ):
        relative = Path(str(evidence[f"{prefix}_path"]))
        path = (_ROOT / relative).resolve(strict=False)
        try:
            path.relative_to(_ROOT.resolve())
        except ValueError as exc:
            raise MalomPolicyAuxiliaryBatchCaptureError(
                f"{prefix} must stay inside the repository"
            ) from exc
        if not path.is_file():
            raise MalomPolicyAuxiliaryBatchCaptureError(
                f"{prefix} is missing: {relative.as_posix()}"
            )
        sha256 = _sha256_file(path)
        if sha256 != evidence[f"{prefix}_sha256"]:
            raise MalomPolicyAuxiliaryBatchCaptureError(f"{prefix} bytes drifted")
        payload = _strict_json(path)
        expected_identity = evidence[
            "calibration_result_identity"
            if prefix == "calibration_result"
            else "gradient_interaction_audit_identity"
        ]
        if payload.get(identity_key) != expected_identity:
            raise MalomPolicyAuxiliaryBatchCaptureError(
                f"{prefix} content identity drifted"
            )
        records[prefix] = {
            "path": relative.as_posix(),
            "sha256": sha256,
            identity_key: expected_identity,
        }
    records["calibration_plan_identity"] = evidence["calibration_plan_identity"]
    return records


def preflight_batch_capture(
    plan_path: str | Path,
    paths_config: str | Path,
    *,
    require_published: bool = True,
    verify_malom_components: bool = True,
    perform_route_check: bool = True,
) -> dict[str, Any]:
    """Verify identities and fresh initializations without consuming a game."""
    plan = load_batch_capture_plan(plan_path)
    parent_preflight = route_probe.preflight_probe(
        plan.parent.path,
        paths_config,
        require_published=require_published,
        verify_malom_components=verify_malom_components,
        perform_route_check=perform_route_check,
    )
    source = parent_preflight["source"]
    inputs = route_probe.resolve_probe_inputs(paths_config)
    data = route_probe.verify_probe_inputs(
        plan.parent,
        inputs,
        verify_malom_components=verify_malom_components,
    )
    source_evidence = verify_source_evidence(plan)
    initializations: list[dict[str, Any]] = []
    for seed in plan.seeds:
        runtime = _runtime_for_seed(plan, inputs, seed)
        try:
            learner = route_probe.model_state_sha256(runtime.model)
            frozen = route_probe.model_state_sha256(runtime.frozen_opponent._model)
            if learner != frozen:
                raise MalomPolicyAuxiliaryBatchCaptureError(
                    "fresh learner and frozen target identities differ"
                )
            if any(parameter.requires_grad for parameter in runtime.model.parameters()):
                raise MalomPolicyAuxiliaryBatchCaptureError(
                    "preflight learner unexpectedly requires gradients"
                )
            initializations.append(
                {
                    "seed": seed,
                    "learner_sha256": learner,
                    "frozen_target_sha256": frozen,
                }
            )
        finally:
            runtime.close()
    if len({item["learner_sha256"] for item in initializations}) != len(
        initializations
    ):
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "fresh seeds did not produce distinct model identities"
        )

    body = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "ready_for_explicit_one_run_authorization",
        "launch_authorized": False,
        "plan": tracked_plan_record(plan),
        "source": source,
        "parent_route_preflight": {
            "schema_version": parent_preflight["schema_version"],
            "status": parent_preflight["status"],
            "plan": parent_preflight["plan"],
            "sanmill": parent_preflight["sanmill"],
            "no_search_route_check": parent_preflight["no_search_route_check"],
        },
        "data": data,
        "source_evidence": source_evidence,
        "fresh_initializations": initializations,
        "gradient_contract": {
            "target_policy_head_ratios": list(plan.target_ratios),
            "denominator_floor": plan.denominator_floor,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
        },
        "bounded_work": dict(plan.payload["bounded_work"]),
        "claim_boundary": dict(plan.payload["claim_boundary"]),
        "next_gate": ("explicit one-run authority bound to this readiness identity"),
    }
    return {**body, "readiness_identity": canonical_sha256(body)}


def _validate_run_id(run_id: str) -> None:
    if (
        not run_id
        or len(run_id) > 128
        or any(character not in _RUN_ID_CHARS for character in run_id)
    ):
        raise MalomPolicyAuxiliaryBatchCaptureError("batch-capture run_id is invalid")


def validate_readiness(
    report: Mapping[str, Any],
    plan: CapturePlan,
    *,
    expected_identity: str,
    source: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "launch_authorized",
        "plan",
        "source",
        "parent_route_preflight",
        "data",
        "source_evidence",
        "fresh_initializations",
        "gradient_contract",
        "bounded_work",
        "claim_boundary",
        "next_gate",
        "readiness_identity",
    }
    if set(report) != expected_keys:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture readiness member set is invalid"
        )
    if report.get("schema_version") != PREFLIGHT_SCHEMA:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture readiness schema is invalid"
        )
    body = dict(report)
    identity = body.pop("readiness_identity", None)
    if (
        identity != expected_identity
        or identity != canonical_sha256(body)
        or report.get("status") != "ready_for_explicit_one_run_authorization"
    ):
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture readiness identity is invalid"
        )
    if report.get("launch_authorized") is not False:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "readiness must not claim launch authority"
        )
    plan_record = report.get("plan")
    if (
        not isinstance(plan_record, Mapping)
        or plan_record.get("identity") != plan.identity
        or plan_record.get("raw_sha256") != plan.raw_sha256
    ):
        raise MalomPolicyAuxiliaryBatchCaptureError("readiness plan identity drifted")
    if report.get("source") != dict(source):
        raise MalomPolicyAuxiliaryBatchCaptureError("readiness source commit drifted")
    if report.get("source_evidence") != verify_source_evidence(plan):
        raise MalomPolicyAuxiliaryBatchCaptureError("readiness source evidence drifted")
    if report.get("bounded_work") != plan.payload["bounded_work"]:
        raise MalomPolicyAuxiliaryBatchCaptureError("readiness bounded work drifted")
    if report.get("claim_boundary") != plan.payload["claim_boundary"]:
        raise MalomPolicyAuxiliaryBatchCaptureError("readiness claim boundary drifted")
    gradient_contract = report.get("gradient_contract")
    if gradient_contract != {
        "target_policy_head_ratios": list(plan.target_ratios),
        "denominator_floor": plan.denominator_floor,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "readiness gradient contract drifted"
        )
    initializations = report.get("fresh_initializations")
    if (
        not isinstance(initializations, list)
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"seed", "learner_sha256", "frozen_target_sha256"}
            or not isinstance(item.get("learner_sha256"), str)
            or len(item.get("learner_sha256", "")) != 64
            or any(
                character not in "0123456789abcdef"
                for character in item.get("learner_sha256", "")
            )
            or item.get("learner_sha256") != item.get("frozen_target_sha256")
            for item in initializations
        )
        or [item.get("seed") for item in initializations] != list(plan.seeds)
    ):
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "readiness fresh initializations drifted"
        )


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int]:
    checked = [float(value) for value in values]
    if not checked or any(not math.isfinite(value) for value in checked):
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture summary input is empty or non-finite"
        )
    ordered = sorted(checked)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median,
        "p90_nearest_rank": ordered[p90_index],
        "max": ordered[-1],
    }


def summarize_batch_capture(
    samples: Sequence[Mapping[str, Any]],
    batches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize observed support without making a selection or forecast."""
    by_source_color: Counter[tuple[str, str, str]] = Counter()
    terminations: Counter[str] = Counter()
    for sample in samples:
        by_source_color[
            (
                str(sample["opponent_kind"]),
                str(sample["learner_color"]),
                str(sample["outcome"]),
            )
        ] += 1
        terminations[str(sample["termination_reason"])] += 1

    per_seed: list[dict[str, Any]] = []
    for seed in sorted({int(sample["seed"]) for sample in samples}):
        seed_samples = [sample for sample in samples if sample["seed"] == seed]
        seed_batches = [batch for batch in batches if batch["seed"] == seed]
        per_seed.append(
            {
                "seed": seed,
                "games": len(seed_samples),
                "batches": len(seed_batches),
                "learner_steps": sum(
                    int(sample["learner_steps"]) for sample in seed_samples
                ),
                "informative_steps": sum(
                    int(sample["malom_informative_steps"]) for sample in seed_samples
                ),
            }
        )

    measured_batches = [
        batch
        for batch in batches
        if int(batch["gradient_measurement"]["support"]["informative_steps"]) > 0
    ]
    distributions = {
        "batch_steps": _numeric_summary([float(batch["steps"]) for batch in batches]),
        "informative_steps": _numeric_summary(
            [
                float(batch["gradient_measurement"]["support"]["informative_steps"])
                for batch in batches
            ]
        ),
        "ordinary_policy_head_gradient_l2": _numeric_summary(
            [
                float(batch["gradient_measurement"]["ordinary_policy_head_gradient_l2"])
                for batch in batches
            ]
        ),
        "raw_auxiliary_gradient_l2": _numeric_summary(
            [
                float(batch["gradient_measurement"]["raw_auxiliary_gradient_l2"])
                for batch in batches
            ]
        ),
    }
    if measured_batches:
        distributions["auxiliary_to_ordinary_policy_head_cosine"] = _numeric_summary(
            [
                float(
                    batch["gradient_measurement"][
                        "raw_auxiliary_to_ordinary_policy_head_cosine"
                    ]
                )
                for batch in measured_batches
            ]
        )

    candidate_distributions: list[dict[str, Any]] = []
    for target in sorted(
        {
            float(item["target_policy_head_ratio"])
            for batch in batches
            for item in batch["gradient_measurement"]["candidate_scales"]
        }
    ):
        selected = [
            item
            for batch in batches
            for item in batch["gradient_measurement"]["candidate_scales"]
            if float(item["target_policy_head_ratio"]) == target
        ]
        measured = [
            float(item["effective_coefficient"])
            for item in selected
            if item["status"] == "measured"
        ]
        candidate_distributions.append(
            {
                "target_policy_head_ratio": target,
                "status_counts": dict(
                    sorted(Counter(str(item["status"]) for item in selected).items())
                ),
                "effective_coefficient": (
                    _numeric_summary(measured) if measured else None
                ),
            }
        )

    labelled_by_phase: Counter[str] = Counter()
    informative_by_phase: Counter[str] = Counter()
    for batch in batches:
        support = batch["gradient_measurement"]["support"]
        labelled_by_phase.update(
            {
                str(key): int(value)
                for key, value in support["labelled_by_phase"].items()
            }
        )
        informative_by_phase.update(
            {
                str(key): int(value)
                for key, value in support["informative_by_phase"].items()
            }
        )

    return {
        "games": len(samples),
        "batches": len(batches),
        "batches_with_informative_steps": len(measured_batches),
        "per_seed": per_seed,
        "wdl_by_opponent_source_and_learner_color": [
            {
                "opponent_kind": key[0],
                "learner_color": key[1],
                "outcome": key[2],
                "games": value,
            }
            for key, value in sorted(by_source_color.items())
        ],
        "termination_reasons": dict(sorted(terminations.items())),
        "labelled_steps_by_phase": dict(sorted(labelled_by_phase.items())),
        "informative_steps_by_phase": dict(sorted(informative_by_phase.items())),
        "gradient_distributions": distributions,
        "candidate_scale_distributions": candidate_distributions,
        "selection_made": False,
    }


def run_batch_capture(
    plan: CapturePlan,
    inputs: route_probe.ProbeInputs,
    *,
    source: Mapping[str, Any],
    readiness_identity: str,
    run_id: str,
    invocation: Sequence[str],
) -> dict[str, Any]:
    """Run one authorized capture and return complete immutable evidence."""
    _validate_run_id(run_id)
    data_before = route_probe.verify_probe_inputs(plan.parent, inputs)
    evidence_before = verify_source_evidence(plan)
    source_before = dict(source)
    started_at = route_probe._utc_now()
    try:
        execution = execute_batch_capture(plan, inputs)
    except MalomPolicyAuxiliaryBatchCaptureExecutionError as exc:
        data_after = route_probe.verify_probe_inputs(plan.parent, inputs)
        source_after = route_probe.inspect_published_source(require_published=True)
        body = {
            "schema_version": FAILURE_SCHEMA,
            "status": "failed_closed",
            "run_id": run_id,
            "started_at": started_at,
            "failed_at": route_probe._utc_now(),
            "invocation": list(invocation),
            "readiness_identity": readiness_identity,
            "plan": tracked_plan_record(plan),
            "source_before": source_before,
            "source_after": source_after,
            "source_unchanged": source_after == source_before,
            "data_before": data_before,
            "data_after": data_after,
            "data_unchanged": data_after == data_before,
            "source_evidence_before": evidence_before,
            "source_evidence_after": verify_source_evidence(plan),
            "failure": exc.diagnostic,
            "claim_boundary": dict(plan.payload["claim_boundary"]),
            "retry_authorized": False,
            "training_launch_authorized": False,
        }
        report = {**body, "report_identity": canonical_sha256(body)}
        raise MalomPolicyAuxiliaryBatchCaptureRunFailure(report) from exc

    data_after = route_probe.verify_probe_inputs(plan.parent, inputs)
    evidence_after = verify_source_evidence(plan)
    source_after = route_probe.inspect_published_source(require_published=True)
    if data_after != data_before:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture input data changed during execution"
        )
    if source_after != source_before:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture source changed during execution"
        )
    if evidence_after != evidence_before:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture source evidence changed during execution"
        )
    installations = [
        training_installation_record(
            route_probe.inspect_sanmill_training_installation(inputs.sanmill_checkout),
            seed=seed,
        )
        for seed in plan.seeds
    ]
    body = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed_no_update_batch_capture",
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": route_probe._utc_now(),
        "invocation": list(invocation),
        "readiness_identity": readiness_identity,
        "plan": tracked_plan_record(plan),
        "source": source_before,
        "sanmill_by_seed": installations,
        "data_before": data_before,
        "data_after": data_after,
        "source_evidence_before": evidence_before,
        "source_evidence_after": evidence_after,
        "wall_seconds": execution["wall_seconds"],
        "samples": execution["samples"],
        "batches": execution["batches"],
        "excluded_residuals": execution["excluded_residuals"],
        "models": execution["models"],
        "summary": summarize_batch_capture(
            execution["samples"],
            execution["batches"],
        ),
        "bounded_work": dict(plan.payload["bounded_work"]),
        "claim_boundary": dict(plan.payload["claim_boundary"]),
        "interpretation": {
            "observed_fact": (
                "production-shaped rollout and gradient distributions from "
                "three fixed fresh models"
            ),
            "not_measured": [
                "optimizer response",
                "train or validation curve",
                "strength",
                "promotion",
                "normalization effectiveness",
            ],
            "next_gate": (
                "review the distribution before designing a bounded learning "
                "calibration"
            ),
        },
    }
    return {**body, "report_identity": canonical_sha256(body)}


def validate_output_path(path: str | Path) -> Path:
    target = Path(path).resolve(strict=False)
    diagnostics = (_ROOT / "out" / "diagnostics").resolve(strict=False)
    try:
        target.relative_to(diagnostics)
    except ValueError as exc:
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture output must be under out/diagnostics"
        ) from exc
    if target.suffix.lower() != ".json":
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture output must be a JSON file"
        )
    if target.exists():
        raise FileExistsError(f"batch-capture output already exists: {target}")
    return target


def failure_output_path(path: str | Path) -> Path:
    target = Path(path).resolve(strict=False)
    return target.with_name(f"{target.stem}.failure.json")


def publish_report(path: str | Path, report: Mapping[str, Any]) -> None:
    target = validate_output_path(path)
    schema_version = report.get("schema_version")
    identity_key = (
        "readiness_identity"
        if schema_version == PREFLIGHT_SCHEMA
        else "report_identity"
    )
    identity = report.get(identity_key)
    body = dict(report)
    body.pop(identity_key, None)
    if identity != canonical_sha256(body):
        raise MalomPolicyAuxiliaryBatchCaptureError(
            "batch-capture report identity is invalid"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
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
        if target.exists():
            raise FileExistsError(f"batch-capture output already exists: {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
