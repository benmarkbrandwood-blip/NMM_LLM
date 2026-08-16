"""Strict complete-game evaluation of frozen positional-safe guidance rules.

This module is intentionally an evaluation harness, not a policy training
surface.  Every candidate action is restricted to the corrected Malom
position-only safe set ``A_pos``.  The frozen human estimators are used only
to rank that set and are never fit or updated here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ai.malom_db import MalomDB
from game.board import BoardState
from game.rules import get_all_legal_moves, get_game_phase, terminal_wdl
from learned_ai.evaluation.human_f0h0_feasibility import (
    _oracle_inventory,
    canonical_sha256,
)
from learned_ai.evaluation.human_feature_deviation import PHASE_NAMES
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    NumericalContract,
)
from learned_ai.evaluation.human_feature_deviation_product_conversion import (
    _fold_parameters,
    _successor_response_risks,
)
from learned_ai.evaluation.sanmill_safe_inducement import (
    WDL_RANK,
    _label_response,
    _move_key,
)
from learned_ai.evaluation.sanmill_uci import (
    UciLogicalTurnResult,
    UciPositionState,
)
from learned_ai.training.run_contract import canonical_json_bytes
from learned_ai.training.sanmill_referee import (
    SanmillTrainingGame,
    nmm_move_actions,
)


PLAN_SCHEMA = "nmm.sanmill-safe-guidance-gameplay-plan.v1"
POOL_SCHEMA = "nmm.sanmill-safe-guidance-gameplay-start-pool.v1"
AUTHORIZATION_SCHEMA = "nmm.sanmill-safe-guidance-gameplay-authorization.v1"
PREFLIGHT_SCHEMA = "nmm.sanmill-safe-guidance-gameplay-preflight.v1"
GAME_SCHEMA = "nmm.sanmill-safe-guidance-gameplay-game.v1"
RESULT_SCHEMA = "nmm.sanmill-safe-guidance-gameplay-result.v1"
ATTEMPT_SPEC_SCHEMA = "nmm.sanmill-safe-guidance-gameplay-attempt-002.v1"
REHEARSAL_RESULT_SCHEMA = (
    "nmm.sanmill-safe-guidance-gameplay-attempt-002-rehearsal.v1"
)
RESOURCE_CHECKPOINT_SCHEMA = (
    "nmm.sanmill-safe-guidance-gameplay-resource-checkpoint.v1"
)

ARMS = ("random-safe", "full-guided", "geometry-guided")
PHASES = ("placement", "movement", "flying")
PRIMARY_NODE_BUDGET = 100_000
DECOMPOSITION_BUDGETS = (1_000, 100_000, 500_000)
STARTS_PER_PHASE = 85
EXPECTED_STARTS = STARTS_PER_PHASE * len(PHASES)
EXPECTED_GAMES = EXPECTED_STARTS * 2 * len(ARMS)
MAX_POST_START_LOGICAL_PLIES = 1536


class SafeGuidanceGameplayError(RuntimeError):
    """Raised when a frozen identity, semantic gate, or resource differs."""


class SafeGuidanceGameplayIncomplete(SafeGuidanceGameplayError):
    """Raised when an execution cannot yield the frozen complete result."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sealed(
    path: str | Path,
    *,
    schema: str,
    identity_field: str,
) -> tuple[dict[str, Any], str]:
    source = Path(path)
    try:
        raw = source.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafeGuidanceGameplayError(f"cannot load sealed JSON: {source}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise SafeGuidanceGameplayError(f"sealed schema differs: {source}")
    identity = value.get(identity_field)
    body = dict(value)
    body.pop(identity_field, None)
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise SafeGuidanceGameplayError(f"sealed identity differs: {source}")
    return value, hashlib.sha256(raw).hexdigest()


def load_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    plan, file_sha = load_sealed(
        path, schema=PLAN_SCHEMA, identity_field="plan_identity"
    )
    if plan.get("status") != "frozen_before_start_pool_or_gameplay":
        raise SafeGuidanceGameplayError("gameplay plan status differs")
    experiment = plan.get("experiment", {})
    if (
        experiment.get("arms") != list(ARMS)
        or experiment.get("starts_per_phase") != STARTS_PER_PHASE
        or experiment.get("starts") != EXPECTED_STARTS
        or experiment.get("games") != EXPECTED_GAMES
        or experiment.get("primary_node_budget") != PRIMARY_NODE_BUDGET
        or experiment.get("max_post_start_logical_plies")
        != MAX_POST_START_LOGICAL_PLIES
    ):
        raise SafeGuidanceGameplayError("gameplay plan experiment differs")
    boundary = plan.get("claim_boundary", {})
    if (
        boundary.get("safe_set") != "A_pos"
        or boundary.get("positional_only") is not True
        or boundary.get("A_allow_claim") is not False
        or boundary.get("human_trap_claim") is not False
    ):
        raise SafeGuidanceGameplayError("gameplay claim boundary differs")
    return plan, file_sha


def load_pool(path: str | Path) -> tuple[dict[str, Any], str]:
    pool, file_sha = load_sealed(
        path, schema=POOL_SCHEMA, identity_field="pool_identity"
    )
    if pool.get("status") != "frozen_before_any_gameplay_or_sanmill_observation":
        raise SafeGuidanceGameplayError("start pool status differs")
    states = pool.get("states")
    if not isinstance(states, list) or len(states) != EXPECTED_STARTS:
        raise SafeGuidanceGameplayError("start pool size differs")
    if len({row.get("state_id") for row in states}) != len(states):
        raise SafeGuidanceGameplayError("start IDs are duplicated")
    if len({row.get("session_id") for row in states}) != len(states):
        raise SafeGuidanceGameplayError("start pool reuses a source game")
    if Counter(str(row.get("phase")) for row in states) != Counter(
        {phase: STARTS_PER_PHASE for phase in PHASES}
    ):
        raise SafeGuidanceGameplayError("start pool phase allocation differs")
    for row in states:
        turns = row.get("logical_turns")
        if not isinstance(turns, list) or len(turns) != row.get("logical_ply"):
            raise SafeGuidanceGameplayError("start logical history differs")
        if not isinstance(row.get("oof_fold"), int) or not 0 <= row["oof_fold"] < 5:
            raise SafeGuidanceGameplayError("start OOF fold differs")
    return pool, file_sha


def load_authorization(path: str | Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_sealed(
        path, schema=AUTHORIZATION_SCHEMA, identity_field="authorization_identity"
    )
    if (
        value.get("operator") != "product-owner-direct"
        or value.get("grant_count") != 1
        or value.get("status") != "authorized_once_gameplay_unconsumed"
    ):
        raise SafeGuidanceGameplayError("gameplay authorization differs")
    return value, file_sha


def load_preflight(path: str | Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_sealed(
        path, schema=PREFLIGHT_SCHEMA, identity_field="preflight_identity"
    )
    if (
        value.get("status") != "ready_for_one_authorized_execution"
        or value.get("complete_games") != 0
        or value.get("determinism", {}).get("passed") is not True
        or value.get("guide_canary", {}).get("passed") is not True
    ):
        raise SafeGuidanceGameplayError("gameplay preflight differs")
    return value, file_sha


def load_attempt_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    value, file_sha = load_sealed(
        path, schema=ATTEMPT_SPEC_SCHEMA, identity_field="attempt_identity"
    )
    execution = value.get("formal_execution", {})
    rehearsal = value.get("rehearsal", {})
    if (
        value.get("status") != "frozen_before_attempt_002_rehearsal"
        or value.get("plan_identity")
        != "1d368c336db5f49493a2abf3c9e7d507c013d9fed3d14cd928ee988575969cc6"
        or value.get("start_pool_identity")
        != "385a376dd82953c23c232f34e3dd5a84e5887b978c60627657eccfa6821eb6e9"
        or value.get("start_pool_membership_identity")
        != "cb84ed8180b103d7c25d56a5051fb2476047788505ed0cb9f437c39c9048fb15"
        or len(execution.get("excluded_start_ids", [])) != 1
        or execution.get("starts") != 254
        or execution.get("games") != 1524
        or rehearsal.get("games") != 4
        or rehearsal.get("independent_starts") != 2
    ):
        raise SafeGuidanceGameplayError("attempt-002 frozen specification differs")
    return value, file_sha


def build_schedule(states: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(states) != EXPECTED_STARTS:
        raise SafeGuidanceGameplayError("schedule start count differs")
    schedule: list[dict[str, Any]] = []
    for start_index, state in enumerate(states):
        for color_index, candidate_color in enumerate(("W", "B")):
            unit_index = start_index * 2 + color_index
            for arm_index, arm in enumerate(ARMS):
                ordinal = unit_index * len(ARMS) + arm_index
                body = {
                    "namespace": "sanmill-safe-guidance-gameplay-game-v1",
                    "ordinal": ordinal,
                    "start_id": state["state_id"],
                    "candidate_color": candidate_color,
                    "arm": arm,
                }
                schedule.append(
                    {
                        "ordinal": ordinal,
                        "unit_index": unit_index,
                        "start_index": start_index,
                        "start_id": str(state["state_id"]),
                        "phase": str(state["phase"]),
                        "candidate_color": candidate_color,
                        "arm": arm,
                        "game_id": canonical_sha256(body),
                    }
                )
    if len(schedule) != EXPECTED_GAMES:
        raise SafeGuidanceGameplayError("schedule game count differs")
    return schedule


def select_schedule_excluding_starts(
    schedule: Sequence[Mapping[str, Any]],
    *,
    excluded_start_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Exclude only named starts while preserving every surviving game identity."""
    excluded = set(excluded_start_ids)
    if len(excluded) != len(excluded_start_ids):
        raise SafeGuidanceGameplayError("excluded start IDs are duplicated")
    available = {str(row["start_id"]) for row in schedule}
    if not excluded or not excluded <= available:
        raise SafeGuidanceGameplayError("excluded start ID is absent from schedule")
    selected = [dict(row) for row in schedule if str(row["start_id"]) not in excluded]
    expected = len(schedule) - len(excluded) * 2 * len(ARMS)
    if len(selected) != expected:
        raise SafeGuidanceGameplayError("excluded schedule size differs")
    surviving_ids = {
        str(row["game_id"])
        for row in schedule
        if str(row["start_id"]) not in excluded
    }
    if {str(row["game_id"]) for row in selected} != surviving_ids:
        raise SafeGuidanceGameplayError("surviving game identities changed")
    return selected


@dataclass
class ResourceLedger:
    """Fail-closed accounting across pool, preflight, and the once-only run."""

    engine_searches: int
    malom_queries: int
    active_seconds_before_run: float
    maximum_engine_searches: int
    maximum_malom_queries: int
    maximum_active_seconds: float
    started: float = 0.0

    def __post_init__(self) -> None:
        self.started = time.perf_counter()
        self.require_within()

    @property
    def active_seconds(self) -> float:
        return self.active_seconds_before_run + time.perf_counter() - self.started

    def add_engine(self, count: int = 1) -> None:
        self.engine_searches += count
        self.require_within()

    def add_malom(self, count: int) -> None:
        self.malom_queries += count
        self.require_within()

    def require_within(self) -> None:
        if self.engine_searches > self.maximum_engine_searches:
            raise SafeGuidanceGameplayIncomplete("engine-search ceiling exceeded")
        if self.malom_queries > self.maximum_malom_queries:
            raise SafeGuidanceGameplayIncomplete("Malom-query ceiling exceeded")
        if self.active_seconds > self.maximum_active_seconds:
            raise SafeGuidanceGameplayIncomplete("active-time ceiling exceeded")

    def record(self) -> dict[str, Any]:
        self.require_within()
        return {
            "engine_single_step_searches": self.engine_searches,
            "malom_read_only_queries": self.malom_queries,
            "active_seconds": self.active_seconds,
        }


def _resource_values(value: Mapping[str, Any]) -> dict[str, int | float]:
    required = {
        "engine_single_step_searches",
        "malom_read_only_queries",
        "active_seconds",
    }
    if not required <= value.keys():
        raise SafeGuidanceGameplayError("resource snapshot fields are absent")
    record = {
        "engine_single_step_searches": int(value["engine_single_step_searches"]),
        "malom_read_only_queries": int(value["malom_read_only_queries"]),
        "active_seconds": float(value["active_seconds"]),
    }
    if any(item < 0 for item in record.values()):
        raise SafeGuidanceGameplayError("resource snapshot is negative")
    return record


def append_resource_checkpoint(
    path: str | Path,
    *,
    completion_index: int,
    complete_games_before: int,
    game_record: Mapping[str, Any],
    resources_before: Mapping[str, Any],
    resources_after: Mapping[str, Any],
    previous_checkpoint_sha256: str | None,
) -> str:
    """Fsync one completed game's exact resource increment before its record."""
    if completion_index < 0 or complete_games_before < 0:
        raise SafeGuidanceGameplayError("resource checkpoint index is invalid")
    before = _resource_values(resources_before)
    after = _resource_values(resources_after)
    delta = {key: after[key] - before[key] for key in before}
    if any(value < 0 for value in delta.values()):
        raise SafeGuidanceGameplayError("resource checkpoint regressed")
    game_identity = canonical_sha256(game_record)
    body = {
        "schema_version": RESOURCE_CHECKPOINT_SCHEMA,
        "completion_index": completion_index,
        "schedule_ordinal": int(game_record["ordinal"]),
        "game_id": str(game_record["game_id"]),
        "game_record_identity": game_identity,
        "complete_games_before": complete_games_before + completion_index,
        "complete_games_after": complete_games_before + completion_index + 1,
        "resources_before": before,
        "resources_after": after,
        "resource_increment": delta,
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
    }
    checkpoint_sha = canonical_sha256(body)
    wrapper = {"checkpoint": body, "checkpoint_sha256": checkpoint_sha}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb" if completion_index == 0 else "ab") as handle:
        handle.write(canonical_json_bytes(wrapper) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return checkpoint_sha


def load_resource_checkpoints(
    path: str | Path,
    *,
    expected_baseline: Mapping[str, Any],
    complete_games_before: int,
) -> dict[str, Any]:
    """Recover and validate the exact fsynced checkpoint prefix after a crash."""
    target = Path(path)
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise SafeGuidanceGameplayError("resource checkpoint journal is absent") from exc
    if raw and not raw.endswith(b"\n"):
        raise SafeGuidanceGameplayError("resource checkpoint journal is partial")
    baseline = _resource_values(expected_baseline)
    previous_resources = baseline
    previous_sha: str | None = None
    rows: list[dict[str, Any]] = []
    for index, encoded in enumerate(raw.splitlines()):
        try:
            wrapper = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafeGuidanceGameplayError(
                "resource checkpoint line is malformed"
            ) from exc
        if not isinstance(wrapper, dict) or set(wrapper) != {
            "checkpoint",
            "checkpoint_sha256",
        }:
            raise SafeGuidanceGameplayError("resource checkpoint wrapper differs")
        row = wrapper["checkpoint"]
        if not isinstance(row, dict):
            raise SafeGuidanceGameplayError("resource checkpoint body differs")
        if (
            row.get("schema_version") != RESOURCE_CHECKPOINT_SCHEMA
            or row.get("completion_index") != index
            or row.get("complete_games_before") != complete_games_before + index
            or row.get("complete_games_after") != complete_games_before + index + 1
            or row.get("previous_checkpoint_sha256") != previous_sha
            or row.get("resources_before") != previous_resources
        ):
            raise SafeGuidanceGameplayError("resource checkpoint chain differs")
        observed_sha = wrapper["checkpoint_sha256"]
        if not isinstance(observed_sha, str) or canonical_sha256(row) != observed_sha:
            raise SafeGuidanceGameplayError("resource checkpoint identity differs")
        after = _resource_values(row.get("resources_after", {}))
        increment = _resource_values(row.get("resource_increment", {}))
        expected_increment = {
            key: after[key] - previous_resources[key] for key in previous_resources
        }
        if increment != expected_increment or any(
            value < 0 for value in expected_increment.values()
        ):
            raise SafeGuidanceGameplayError("resource checkpoint increment differs")
        rows.append(row)
        previous_resources = after
        previous_sha = observed_sha
    return {
        "checkpoints": rows,
        "checkpoint_count": len(rows),
        "tail_checkpoint_sha256": previous_sha,
        "last_resources": previous_resources,
        "complete_games_after": complete_games_before + len(rows),
        "file_sha256": sha256_file(target),
    }


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically publish a small progress or completion record."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.partial")
    with partial.open("wb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, target)


def _normal_move(move: Mapping[str, Any]) -> dict[str, str | None]:
    return {
        "from": move.get("from"),
        "to": move.get("to"),
        "capture": move.get("capture"),
    }


def _pooled_action_key(action: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the move key from a persisted A_pos action envelope."""
    move = action.get("move")
    if not isinstance(move, Mapping):
        raise SafeGuidanceGameplayError("persisted A_pos action move is absent")
    return _move_key(move)


def _assert_canary_selection(
    *,
    specification: str,
    actions: Sequence[Mapping[str, Any]],
    selected_index: int,
    selected_risk: float,
    expected: Mapping[str, Any],
) -> None:
    """Fail closed when a recomputed canary selection truly differs."""
    if not 0 <= selected_index < len(actions):
        raise SafeGuidanceGameplayError(
            f"{specification} guide canary selected index differs"
        )
    if _pooled_action_key(actions[selected_index]) != _move_key(
        expected["selected_action"]
    ):
        raise SafeGuidanceGameplayError(
            f"{specification} guide canary selected move differs"
        )
    if selected_risk != float(expected["maximum_risk"]):
        raise SafeGuidanceGameplayError(
            f"{specification} guide canary maximum risk differs"
        )


def _matching_move(board: BoardState, actions: Sequence[str]) -> Mapping[str, Any]:
    expected = tuple(actions)
    matches = [
        move for move in get_all_legal_moves(board) if nmm_move_actions(move) == expected
    ]
    if len(matches) != 1:
        raise SafeGuidanceGameplayError("frozen history does not select one move")
    return matches[0]


def replay_start(
    game: SanmillTrainingGame, state: Mapping[str, Any], ledger: ResourceLedger
) -> tuple[BoardState, dict[str, Any]]:
    board = BoardState.new_game()
    for actions in state["logical_turns"]:
        ledger.require_within()
        move = _matching_move(board, actions)
        game.apply_nmm_move(board, move)
        board = board.apply_move(move)
    game.assert_current_board(board)
    if board.to_fen_string() != state["fen"]:
        raise SafeGuidanceGameplayError("replayed start FEN differs")
    if list(game.history) != state["history_actions"]:
        raise SafeGuidanceGameplayError("replayed start action history differs")
    if game.state.logical_ply_count != state["logical_ply"]:
        raise SafeGuidanceGameplayError("replayed start logical ply differs")
    expected_strict_history = state.get("strict_history_sha256")
    if (
        expected_strict_history is not None
        and game.state.history_sha256 != expected_strict_history
    ):
        raise SafeGuidanceGameplayError("replayed strict history identity differs")
    if game.state.terminal:
        raise SafeGuidanceGameplayError("frozen start is strict-terminal")
    return board, dict(_checked_position_state(game.state))


class FrozenSafePolicy:
    """One no-fit A_pos policy with a deterministic per-game choice stream."""

    def __init__(
        self,
        *,
        arm: str,
        fold: int,
        readiness: Mapping[str, Any],
        database: MalomDB,
        random_seed: str,
        ledger: ResourceLedger,
    ) -> None:
        if arm not in ARMS:
            raise SafeGuidanceGameplayError("unknown gameplay arm")
        self.arm = arm
        self.database = database
        self.ledger = ledger
        parameters = _fold_parameters(readiness)
        if fold not in parameters:
            raise SafeGuidanceGameplayError("frozen OOF fold is absent")
        self.parameters = parameters[fold]
        self.contract = NumericalContract.from_plan(readiness["frozen_contract"])
        self.rng = random.Random(
            int.from_bytes(hashlib.sha256(random_seed.encode()).digest(), "big")
        )

    def choose(self, board: BoardState) -> tuple[Mapping[str, Any], dict[str, Any]]:
        parent, inventory, queries = _checked_oracle_inventory(board, self.database)
        self.ledger.add_malom(queries)
        best_rank = max(WDL_RANK[value.outcome] for _move, value in inventory)
        safe = sorted(
            [
                (dict(move), value)
                for move, value in inventory
                if WDL_RANK[value.outcome] == best_rank
            ],
            key=lambda row: _move_key(row[0]),
        )
        if not safe or any(value.outcome != parent for _move, value in safe):
            raise SafeGuidanceGameplayError("runtime A_pos construction differs")
        risks: list[float] | None = None
        response_sets = 0
        response_actions = 0
        if self.arm == "random-safe":
            selected = self.rng.randrange(len(safe))
        else:
            specification = "full" if self.arm == "full-guided" else "geometry"
            risks = []
            for move, _value in safe:
                successor = board.apply_move(move)
                full, geometry, _tier, extra, responses = _successor_response_risks(
                    successor,
                    learner_tier=parent,
                    parameters=self.parameters,
                    contract=self.contract,
                    database=self.database,
                )
                self.ledger.add_malom(extra)
                response_sets += int(responses > 0)
                response_actions += responses
                risks.append(float(full if specification == "full" else geometry))
            array = np.asarray(risks, dtype=np.float64)
            if not np.all(np.isfinite(array)):
                raise SafeGuidanceGameplayError("guide risk is nonfinite")
            selected = int(np.argmax(array))
        move = safe[selected][0]
        return move, {
            "parent_tier": parent,
            "a_pos_cardinality": len(safe),
            "selected_index": selected,
            "selected_move": _normal_move(move),
            "risks": risks,
            "argmax_tie_count": (
                None
                if risks is None
                else sum(value == risks[selected] for value in risks)
            ),
            "response_choice_sets": response_sets,
            "response_actions": response_actions,
            "safe_set": "A_pos",
            "positional_only": True,
        }


def _phase(board: BoardState, color: str) -> str:
    value = PHASE_NAMES.get(get_game_phase(board, color))
    if value not in PHASES:
        raise SafeGuidanceGameplayError("runtime phase differs")
    return value


def _final_positional_tier(board: BoardState, database: MalomDB) -> tuple[str, int]:
    rules = terminal_wdl(board)
    if rules is not None:
        return rules, 0
    value = database.query_value(board)
    if value is None or value.outcome not in WDL_RANK:
        raise SafeGuidanceGameplayError("final positional Malom tier is absent")
    return value.outcome, 1


def _checked_oracle_inventory(
    board: BoardState, database: MalomDB
) -> tuple[str, list[tuple[Mapping[str, Any], Any]], int]:
    """Validate the HumanDB/Malom inventory boundary at every live call."""
    parent, inventory, queries = _oracle_inventory(board, database)
    legal_keys = Counter(_move_key(move) for move in get_all_legal_moves(board))
    inventory_keys = Counter(_move_key(move) for move, _value in inventory)
    if parent not in WDL_RANK or queries < 1 or legal_keys != inventory_keys:
        raise SafeGuidanceGameplayError("oracle inventory contract differs")
    if any(value.outcome not in WDL_RANK for _move, value in inventory):
        raise SafeGuidanceGameplayError("oracle move-value contract differs")
    return parent, inventory, queries


def _checked_search_result(
    result: UciLogicalTurnResult,
    *,
    expected_node_budget: int,
) -> Mapping[str, Any]:
    """Validate the Sanmill logical-search dataclass and its portable record."""
    if not isinstance(result, UciLogicalTurnResult):
        raise SafeGuidanceGameplayError("Sanmill logical result type differs")
    semantic = result.semantic_record()
    if (
        result.status != "ok"
        or result.model_action is None
        or result.logical_ply_delta != 1
        or result.node_budget != expected_node_budget
        or semantic.get("model_action") != dict(result.model_action)
        or semantic.get("full_turn_actions") != list(result.full_turn_actions)
        or semantic.get("terminal") != result.terminal
        or semantic.get("winner") != result.winner
        or semantic.get("outcome_reason") != result.outcome_reason
        or result.total_nodes != result.primary_nodes + result.removal_nodes
    ):
        raise SafeGuidanceGameplayError("Sanmill logical result contract differs")
    if tuple(result.full_turn_actions) != nmm_move_actions(result.model_action):
        raise SafeGuidanceGameplayError("Sanmill logical move/action contract differs")
    return semantic


def _checked_position_state(state: UciPositionState) -> Mapping[str, Any]:
    """Validate direct state fields against the nested portable outcome."""
    if not isinstance(state, UciPositionState):
        raise SafeGuidanceGameplayError("Sanmill position-state type differs")
    portable = state.portable_record()
    outcome = portable.get("outcome")
    if not isinstance(outcome, Mapping):
        raise SafeGuidanceGameplayError("portable Sanmill outcome is absent")
    expected_outcome = {
        "terminal": state.terminal,
        "winner": state.winner,
        "winner_code": state.winner_code,
        "reason": state.outcome_reason,
        "reason_code": state.outcome_reason_code,
    }
    if dict(outcome) != expected_outcome:
        raise SafeGuidanceGameplayError("portable Sanmill outcome differs")
    if (
        portable.get("terminal") != state.terminal
        or portable.get("history_sha256") != state.history_sha256
        or portable.get("logical_ply_count") != state.logical_ply_count
        or portable.get("no_capture_count") != state.no_capture_count
        or portable.get("repetition_current_count")
        != state.repetition_current_count
    ):
        raise SafeGuidanceGameplayError("portable Sanmill state contract differs")
    return portable


def _strict_terminal_outcome(state: UciPositionState) -> tuple[str | None, str]:
    """Read terminal fields from the typed state and cross-check its payload."""
    _checked_position_state(state)
    if not state.terminal or state.outcome_reason == "ongoing":
        raise SafeGuidanceGameplayError("strict terminal state is absent")
    if state.winner not in {None, "white", "black"}:
        raise SafeGuidanceGameplayError("strict winner differs")
    winner = {None: None, "white": "W", "black": "B"}[state.winner]
    return winner, str(state.outcome_reason)


def validate_game_record(
    record: Mapping[str, Any],
    *,
    require_decomposition: bool = False,
) -> None:
    """Fail closed before journaling any cross-component gameplay payload."""
    required = {
        "schema_version",
        "ordinal",
        "game_id",
        "unit_index",
        "start_id",
        "phase",
        "arm",
        "candidate_color",
        "oof_fold",
        "strict_start",
        "post_start_logical_plies",
        "termination_class",
        "outcome_reason",
        "winner",
        "candidate_score",
        "final_state",
        "final_positional",
        "turns",
        "induced_events",
        "game_elapsed_seconds",
    }
    if not required <= record.keys() or record.get("schema_version") != GAME_SCHEMA:
        raise SafeGuidanceGameplayError("game record contract fields differ")
    if record["candidate_color"] not in {"W", "B"}:
        raise SafeGuidanceGameplayError("game record candidate color differs")
    rehearsal_only = record.get("rehearsal_only") is True
    if not rehearsal_only and record["arm"] not in ARMS:
        raise SafeGuidanceGameplayError("game record arm differs")
    if record["termination_class"] == "rules_terminal":
        if record["candidate_score"] not in {0.0, 0.5, 1.0}:
            raise SafeGuidanceGameplayError("terminal game score differs")
        outcome = record["final_state"].get("outcome")
        winner = record["winner"]
        expected_winner_name = {None: None, "W": "white", "B": "black"}.get(
            winner, object()
        )
        expected_score = (
            0.5
            if winner is None
            else float(winner == record["candidate_color"])
        )
        if (
            not isinstance(outcome, Mapping)
            or outcome.get("terminal") is not True
            or outcome.get("reason") != record["outcome_reason"]
            or outcome.get("winner") != expected_winner_name
            or record["candidate_score"] != expected_score
        ):
            raise SafeGuidanceGameplayError("terminal game outcome contract differs")
    elif record["termination_class"] == "safety_cap_incomplete":
        if record["candidate_score"] is not None or record["winner"] is not None:
            raise SafeGuidanceGameplayError("safety cap was converted to a result")
        outcome = record["final_state"].get("outcome")
        if not isinstance(outcome, Mapping) or outcome.get("terminal") is not False:
            raise SafeGuidanceGameplayError("safety-cap state contract differs")
    else:
        raise SafeGuidanceGameplayError("game termination class differs")
    if (
        not isinstance(record["turns"], list)
        or not record["turns"]
        or record["post_start_logical_plies"] != len(record["turns"])
    ):
        raise SafeGuidanceGameplayError("game turn contract differs")
    for expected_ply, turn in enumerate(record["turns"], start=1):
        if (
            turn.get("post_start_ply") != expected_ply
            or turn.get("mover_color") not in {"W", "B"}
            or not isinstance(turn.get("actions"), list)
            or not isinstance(turn.get("move"), Mapping)
            or "candidate_choice" not in turn
            or "engine_response" not in turn
        ):
            raise SafeGuidanceGameplayError("game turn payload differs")
    if not isinstance(record["induced_events"], list):
        raise SafeGuidanceGameplayError("induced-event collection differs")
    for event_index, event in enumerate(record["induced_events"]):
        required_event = {
            "event_index",
            "transition",
            "history_actions_before",
            "primary_semantic_search",
            "budget_flags",
            "budget_type",
        }
        if (
            not isinstance(event, Mapping)
            or not required_event <= event.keys()
            or event.get("event_index") != event_index
            or event.get("transition") not in {"W->D", "W->L", "D->L"}
            or not isinstance(event.get("history_actions_before"), list)
            or not isinstance(event.get("primary_semantic_search"), Mapping)
            or not isinstance(event.get("budget_flags"), Mapping)
        ):
            raise SafeGuidanceGameplayError("induced-event payload differs")
        if require_decomposition and (
            event.get("budget_type")
            not in {"budget-invariant", "budget-sensitive"}
            or set(event["budget_flags"]) != {"1000", "100000", "500000"}
            or any(
                not isinstance(event.get(f"semantic_search_{budget}"), Mapping)
                for budget in (1_000, 500_000)
            )
        ):
            raise SafeGuidanceGameplayError(
                "induced-event decomposition contract differs"
            )


def _finalize_game_record(
    *,
    schedule_item: Mapping[str, Any],
    start_state: Mapping[str, Any],
    candidate_color: str,
    strict_start: Mapping[str, Any],
    turns: list[dict[str, Any]],
    induced_events: list[dict[str, Any]],
    terminal_state: UciPositionState,
    board: BoardState,
    safety_cap: bool,
    database: MalomDB,
    ledger: ResourceLedger,
    game_started: float,
    rehearsal_only: bool = False,
) -> dict[str, Any]:
    final_state = _checked_position_state(terminal_state)
    if safety_cap:
        if terminal_state.terminal:
            raise SafeGuidanceGameplayError("terminal state reached at safety cap")
        winner = None
        score = None
        termination_class = "safety_cap_incomplete"
        outcome_reason = "safety_cap_incomplete"
    else:
        winner, outcome_reason = _strict_terminal_outcome(terminal_state)
        score = 0.5 if winner is None else float(winner == candidate_color)
        termination_class = "rules_terminal"
    final_tier, final_queries = _final_positional_tier(board, database)
    ledger.add_malom(final_queries)
    record = {
        "schema_version": GAME_SCHEMA,
        "ordinal": int(schedule_item["ordinal"]),
        "game_id": str(schedule_item["game_id"]),
        "unit_index": int(schedule_item["unit_index"]),
        "start_id": str(schedule_item["start_id"]),
        "phase": str(schedule_item["phase"]),
        "arm": str(schedule_item["arm"]),
        "candidate_color": candidate_color,
        "oof_fold": int(start_state["oof_fold"]),
        "strict_start": dict(strict_start),
        "post_start_logical_plies": len(turns),
        "termination_class": termination_class,
        "outcome_reason": outcome_reason,
        "winner": winner,
        "candidate_score": score,
        "final_state": dict(final_state),
        "final_positional": {
            "side_to_move": board.turn,
            "side_to_move_wdl": final_tier,
            "history_aware": False,
        },
        "turns": turns,
        "induced_events": induced_events,
        "game_elapsed_seconds": time.perf_counter() - game_started,
        "rehearsal_only": rehearsal_only,
    }
    validate_game_record(record)
    return record


def play_game(
    *,
    schedule_item: Mapping[str, Any],
    start_state: Mapping[str, Any],
    plan: Mapping[str, Any],
    readiness: Mapping[str, Any],
    database: MalomDB,
    installation: Any,
    ledger: ResourceLedger,
) -> dict[str, Any]:
    game_started = time.perf_counter()
    candidate_color = str(schedule_item["candidate_color"])
    policy = FrozenSafePolicy(
        arm=str(schedule_item["arm"]),
        fold=int(start_state["oof_fold"]),
        readiness=readiness,
        database=database,
        random_seed=(
            f"{plan['experiment']['random_safe_seed']}:{schedule_item['game_id']}"
        ),
        ledger=ledger,
    )
    turns: list[dict[str, Any]] = []
    induced_events: list[dict[str, Any]] = []
    candidate_has_acted = False
    safety_cap = False
    seed = int(plan["sanmill_contract"]["seed"])
    with SanmillTrainingGame(installation, seed=seed) as game:
        board, strict_start = replay_start(game, start_state, ledger)
        for post_start_ply in range(1, MAX_POST_START_LOGICAL_PLIES + 1):
            ledger.require_within()
            mover = board.turn
            phase = _phase(board, mover)
            before_history = game.state.history_sha256
            if mover == candidate_color:
                move, choice = policy.choose(board)
                applied = game.apply_nmm_move(board, move)
                actor = "candidate"
                candidate_has_acted = True
                engine_label = None
            else:
                actor = "sanmill"
                parent, inventory, inventory_queries = _checked_oracle_inventory(
                    board, database
                )
                ledger.add_malom(inventory_queries)
                result = game.session.search_logical_turn(PRIMARY_NODE_BUDGET)
                ledger.add_engine()
                semantic_search = _checked_search_result(
                    result,
                    expected_node_budget=PRIMARY_NODE_BUDGET,
                )
                assert result.model_action is not None
                move = result.model_action
                selected_values = [
                    value
                    for candidate, value in inventory
                    if _move_key(candidate) == _move_key(move)
                ]
                if len(selected_values) != 1:
                    raise SafeGuidanceGameplayError("Sanmill move is absent from inventory")
                after = selected_values[0].outcome
                transition = (
                    f"{parent}->{after}"
                    if WDL_RANK[after] < WDL_RANK[parent]
                    else None
                )
                if transition not in {None, "W->D", "W->L", "D->L"}:
                    raise SafeGuidanceGameplayError("unexpected downgrade transition")
                induced = bool(candidate_has_acted and transition)
                engine_label = {
                    "parent_tier": parent,
                    "after_tier": after,
                    "downgrade_transition": transition,
                    "induced_after_candidate_action": induced,
                    "primary_node_budget": PRIMARY_NODE_BUDGET,
                    "semantic_search": semantic_search,
                }
                applied = game.apply_nmm_move(board, move, search_result=result)
                choice = None
                if induced:
                    induced_events.append(
                        {
                            "event_index": len(induced_events),
                            "post_start_ply": post_start_ply,
                            "phase": phase,
                            "transition": transition,
                            "board_fen_before": board.to_fen_string(),
                            "history_actions_before": list(game.history[:-len(applied.actions)]),
                            "primary_move": _normal_move(move),
                            "primary_semantic_search": semantic_search,
                            "budget_flags": {"100000": True},
                            "budget_type": None,
                        }
                    )
                candidate_has_acted = False
            board = board.apply_move(applied.move)
            _checked_position_state(game.state)
            turns.append(
                {
                    "post_start_ply": post_start_ply,
                    "absolute_logical_ply": game.state.logical_ply_count,
                    "mover_color": mover,
                    "actor": actor,
                    "phase": phase,
                    "move": _normal_move(applied.move),
                    "actions": list(applied.actions),
                    "history_sha256_before": before_history,
                    "history_sha256_after": game.state.history_sha256,
                    "no_capture_count": game.state.no_capture_count,
                    "repetition_current_count": game.state.repetition_current_count,
                    "repetition_history_length": game.state.repetition_history_length,
                    "terminal": game.state.terminal,
                    "outcome_reason": game.state.outcome_reason,
                    "candidate_choice": choice,
                    "engine_response": engine_label,
                }
            )
            if game.state.terminal:
                break
        else:
            safety_cap = True
        terminal_state = game.state

    return _finalize_game_record(
        schedule_item=schedule_item,
        start_state=start_state,
        candidate_color=candidate_color,
        strict_start=strict_start,
        turns=turns,
        induced_events=induced_events,
        terminal_state=terminal_state,
        board=board,
        safety_cap=safety_cap,
        database=database,
        ledger=ledger,
        game_started=game_started,
    )


def replay_scripted_rehearsal_game(
    *,
    schedule_item: Mapping[str, Any],
    start_state: Mapping[str, Any],
    continuation_turns: Sequence[Sequence[str]],
    plan: Mapping[str, Any],
    database: MalomDB,
    installation: Any,
    ledger: ResourceLedger,
) -> dict[str, Any]:
    """Replay a known complete history through the exact packaging boundary."""
    if not continuation_turns:
        raise SafeGuidanceGameplayError("scripted rehearsal continuation is empty")
    game_started = time.perf_counter()
    candidate_color = str(schedule_item["candidate_color"])
    turns: list[dict[str, Any]] = []
    seed = int(plan["sanmill_contract"]["seed"])
    with SanmillTrainingGame(installation, seed=seed) as game:
        board, strict_start = replay_start(game, start_state, ledger)
        for post_start_ply, actions in enumerate(continuation_turns, start=1):
            if game.state.terminal:
                raise SafeGuidanceGameplayError(
                    "scripted rehearsal reached terminal before its final turn"
                )
            ledger.require_within()
            mover = board.turn
            phase = _phase(board, mover)
            before_history = game.state.history_sha256
            move = _matching_move(board, actions)
            applied = game.apply_nmm_move(board, move)
            board = board.apply_move(applied.move)
            _checked_position_state(game.state)
            turns.append(
                {
                    "post_start_ply": post_start_ply,
                    "absolute_logical_ply": game.state.logical_ply_count,
                    "mover_color": mover,
                    "actor": "scripted-rehearsal",
                    "phase": phase,
                    "move": _normal_move(applied.move),
                    "actions": list(applied.actions),
                    "history_sha256_before": before_history,
                    "history_sha256_after": game.state.history_sha256,
                    "no_capture_count": game.state.no_capture_count,
                    "repetition_current_count": game.state.repetition_current_count,
                    "repetition_history_length": game.state.repetition_history_length,
                    "terminal": game.state.terminal,
                    "outcome_reason": game.state.outcome_reason,
                    "candidate_choice": None,
                    "engine_response": None,
                }
            )
        if not game.state.terminal:
            raise SafeGuidanceGameplayError(
                "scripted rehearsal did not reach a strict rules terminal"
            )
        terminal_state = game.state
    return _finalize_game_record(
        schedule_item=schedule_item,
        start_state=start_state,
        candidate_color=candidate_color,
        strict_start=strict_start,
        turns=turns,
        induced_events=[],
        terminal_state=terminal_state,
        board=board,
        safety_cap=False,
        database=database,
        ledger=ledger,
        game_started=game_started,
        rehearsal_only=True,
    )


def analyze_rehearsal_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Confirm only technical rehearsal coverage, never an experiment effect."""
    if not 4 <= len(records) <= 8:
        raise SafeGuidanceGameplayError("rehearsal game count is outside 4..8")
    for record in records:
        validate_game_record(record)
        compact_game(record)
    rules = [row for row in records if row["termination_class"] == "rules_terminal"]
    draws = [row for row in rules if row["winner"] is None]
    decisive = [row for row in rules if row["winner"] is not None]
    rule_draws = [
        row
        for row in draws
        if str(row["outcome_reason"])
        in {"drawThreefoldRepetition", "drawFiftyMove", "drawFiftyMoveLegacy"}
    ]
    if len(rules) != len(records) or not draws or not decisive or not rule_draws:
        raise SafeGuidanceGameplayError("rehearsal terminal coverage is incomplete")
    return {
        "status": "technical_rehearsal_passed_non_evidence",
        "games": len(records),
        "rules_terminal_games": len(rules),
        "draws": len(draws),
        "decisive_games": len(decisive),
        "rule_draw_games": len(rule_draws),
        "termination_reasons": dict(
            sorted(Counter(str(row["outcome_reason"]) for row in records).items())
        ),
        "result_packaging_exercised": True,
        "compact_analysis_exercised": True,
        "experimental_effect_claim_allowed": False,
    }


def classify_induced_events(
    *,
    game_record: dict[str, Any],
    plan: Mapping[str, Any],
    database: MalomDB,
    installation: Any,
    ledger: ResourceLedger,
) -> None:
    """Classify only observed 100k downgrades with fresh 1k/500k roots."""
    from learned_ai.evaluation.sanmill_safe_inducement import _open_root

    seed = int(plan["sanmill_contract"]["seed"])
    protocol_timeout = float(plan["sanmill_contract"]["protocol_timeout_seconds"])
    search_timeout = float(plan["sanmill_contract"]["search_timeout_seconds"])
    for event in game_record["induced_events"]:
        board = BoardState.from_fen_string(str(event["board_fen_before"]))
        for budget in (1_000, 500_000):
            with _open_root(
                installation,
                seed=seed,
                history_actions=event["history_actions_before"],
                action_tokens=(),
                board=board,
                protocol_timeout=protocol_timeout,
                search_timeout=search_timeout,
            ) as session:
                result = session.search_logical_turn(budget)
            ledger.add_engine()
            semantic_search = _checked_search_result(
                result,
                expected_node_budget=budget,
            )
            assert result.model_action is not None
            move = result.model_action
            _before, _after, transition, queries = _label_response(
                board, move, database
            )
            ledger.add_malom(queries)
            event["budget_flags"][str(budget)] = transition is not None
            event[f"semantic_search_{budget}"] = semantic_search
        flags = [bool(event["budget_flags"][str(value)]) for value in DECOMPOSITION_BUDGETS]
        event["budget_type"] = "budget-invariant" if all(flags) else "budget-sensitive"


def run_guide_canary(
    *,
    main_pool: Mapping[str, Any],
    transfer_result: Mapping[str, Any],
    readiness: Mapping[str, Any],
    database: MalomDB,
    states_per_phase: int = 2,
) -> dict[str, Any]:
    """Reproduce persisted transfer selections without fitting any estimator."""
    pool_by_id = {str(row["state_id"]): row for row in main_pool["states"]}
    result_rows = transfer_result.get("analysis", {}).get("state_rows")
    if not isinstance(result_rows, list) or len(result_rows) != 360:
        raise SafeGuidanceGameplayError("transfer canary result rows differ")
    result_by_id = {str(row["state_id"]): row for row in result_rows}
    parameters = _fold_parameters(readiness)
    contract = NumericalContract.from_plan(readiness["frozen_contract"])
    fixtures: list[Mapping[str, Any]] = []
    for phase in PHASES:
        rows = sorted(
            (row for row in main_pool["states"] if row["phase"] == phase),
            key=lambda row: row["state_id"],
        )
        fixtures.extend(rows[:states_per_phase])
    query_count = 0
    observations = []
    for state in fixtures:
        expected = result_by_id.get(str(state["state_id"]))
        if expected is None or str(state["state_id"]) not in pool_by_id:
            raise SafeGuidanceGameplayError("guide canary fixture is absent")
        fold = int(expected["fold"])
        actions = sorted(state["a_pos"], key=_pooled_action_key)
        full_risks = []
        geometry_risks = []
        for action in actions:
            board = BoardState.from_fen_string(str(action["successor_fen"]))
            full, geometry, _tier, queries, _responses = _successor_response_risks(
                board,
                learner_tier=str(state["learner_parent_tier"]),
                parameters=parameters[fold],
                contract=contract,
                database=database,
            )
            query_count += queries
            full_risks.append(float(full))
            geometry_risks.append(float(geometry))
        full_index = int(np.argmax(np.asarray(full_risks, dtype=np.float64)))
        geometry_index = int(np.argmax(np.asarray(geometry_risks, dtype=np.float64)))
        _assert_canary_selection(
            specification="full",
            actions=actions,
            selected_index=full_index,
            selected_risk=full_risks[full_index],
            expected=expected["full"],
        )
        _assert_canary_selection(
            specification="geometry",
            actions=actions,
            selected_index=geometry_index,
            selected_risk=geometry_risks[geometry_index],
            expected=expected["geometry"],
        )
        observations.append(
            {
                "state_id": state["state_id"],
                "phase": state["phase"],
                "fold": fold,
                "passed": True,
                "full_selected_move": _normal_move(actions[full_index]["move"]),
                "geometry_selected_move": _normal_move(
                    actions[geometry_index]["move"]
                ),
                "full_risk": full_risks[full_index],
                "geometry_risk": geometry_risks[geometry_index],
            }
        )
    return {
        "passed": all(row["passed"] for row in observations),
        "states": len(observations),
        "states_per_phase": states_per_phase,
        "malom_queries": query_count,
        "estimator_refits": 0,
        "observations": observations,
    }


def append_game_record(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    previous_record_sha256: str | None,
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = {**record, "previous_record_sha256": previous_record_sha256}
    record_hash = canonical_sha256(body)
    wrapper = {"record": body, "record_sha256": record_hash}
    with target.open("xb" if previous_record_sha256 is None else "ab") as handle:
        handle.write(canonical_json_bytes(wrapper) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record_hash


def _mean_interval(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise SafeGuidanceGameplayError("primary values are empty")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(len(values))
    return {
        "support": len(values),
        "mean": mean,
        "sample_standard_deviation": sd,
        "standard_error": sd / math.sqrt(len(values)),
        "half_width": half,
        "interval": [mean - half, mean + half],
        "method": "start-clustered normal 1.96 standard-error interval",
    }


def _arm_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rules = [row for row in records if row["termination_class"] == "rules_terminal"]
    scores = [float(row["candidate_score"]) for row in rules]
    wins = sum(score == 1.0 for score in scores)
    draws = sum(score == 0.5 for score in scores)
    losses = sum(score == 0.0 for score in scores)
    induced_games = [row for row in rules if row["induced_events"]]
    induced_wins = sum(row["candidate_score"] == 1.0 for row in induced_games)
    transitions = Counter(
        event["transition"] for row in records for event in row["induced_events"]
    )
    budget_types = Counter(
        event["budget_type"] for row in records for event in row["induced_events"]
    )
    return {
        "games": len(records),
        "rules_terminal_games": len(rules),
        "safety_cap_incomplete_games": len(records) - len(rules),
        "strict_wdl": {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score_rate": statistics.fmean(scores) if scores else None,
        },
        "termination_reasons": dict(
            sorted(Counter(str(row["outcome_reason"]) for row in records).items())
        ),
        "inducement": {
            "events": sum(transitions.values()),
            "transitions": dict(sorted(transitions.items())),
            "budget_types": dict(sorted(budget_types.items())),
            "games_with_at_least_one_event": len(induced_games),
            "wins_after_at_least_one_event": induced_wins,
            "conversion_rate": (
                induced_wins / len(induced_games) if induced_games else None
            ),
        },
        "rule_draw_separation": {
            "draws_with_induced_event": sum(
                row["candidate_score"] == 0.5 and bool(row["induced_events"])
                for row in rules
            ),
            "draws_without_induced_event": sum(
                row["candidate_score"] == 0.5 and not row["induced_events"]
                for row in rules
            ),
            "games_without_induced_event": sum(not row["induced_events"] for row in rules),
        },
    }


def analyze_games(
    records: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    *,
    expected_start_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    expected_ids = (
        None if expected_start_ids is None else tuple(sorted(expected_start_ids))
    )
    expected_starts = EXPECTED_STARTS if expected_ids is None else len(expected_ids)
    expected_games = expected_starts * 2 * len(ARMS)
    if expected_ids is not None and len(set(expected_ids)) != len(expected_ids):
        raise SafeGuidanceGameplayError("expected analysis start IDs are duplicated")
    if len(records) != expected_games:
        raise SafeGuidanceGameplayIncomplete("complete game count differs")
    for record in records:
        validate_game_record(record, require_decomposition=True)
    if any(row["termination_class"] != "rules_terminal" for row in records):
        decision = "execution_incomplete_safety_cap"
    else:
        decision = None
    by_key = {
        (str(row["start_id"]), str(row["candidate_color"]), str(row["arm"])): row
        for row in records
    }
    if len(by_key) != expected_games:
        raise SafeGuidanceGameplayError("game pairing keys differ")
    starts = sorted({str(row["start_id"]) for row in records})
    if len(starts) != expected_starts or (
        expected_ids is not None and tuple(starts) != expected_ids
    ):
        raise SafeGuidanceGameplayError("analysis start membership differs")

    def differences(left: str, right: str) -> list[float]:
        values = []
        for start_id in starts:
            color_values = []
            for color in ("W", "B"):
                a = by_key[(start_id, color, left)]["candidate_score"]
                b = by_key[(start_id, color, right)]["candidate_score"]
                if a is None or b is None:
                    raise SafeGuidanceGameplayIncomplete(
                        "incomplete game entered paired score"
                    )
                color_values.append(float(a) - float(b))
            values.append(statistics.fmean(color_values))
        return values

    primary = _mean_interval(differences("full-guided", "random-safe"))
    maximum_half = float(plan["primary_decision"]["maximum_half_width"])
    primary["maximum_half_width"] = maximum_half
    primary["precision_adequate"] = primary["half_width"] <= maximum_half
    if decision is None:
        if not primary["precision_adequate"]:
            decision = "inconclusive_precision"
        elif primary["interval"][0] > 0.0:
            decision = "full_guidance_higher_fixed_runtime_score"
        elif primary["interval"][1] < 0.0:
            decision = "random_safe_higher_fixed_runtime_score"
        else:
            decision = "inconclusive_no_score_difference_resolved"
    geometry = _mean_interval(differences("full-guided", "geometry-guided"))
    by_arm = {
        arm: _arm_summary([row for row in records if row["arm"] == arm])
        for arm in ARMS
    }
    by_phase = {
        phase: {
            arm: _arm_summary(
                [row for row in records if row["phase"] == phase and row["arm"] == arm]
            )
            for arm in ARMS
        }
        for phase in PHASES
    }
    return {
        "decision": decision,
        "primary_full_minus_random_start_clustered_score": primary,
        "secondary_full_minus_geometry_start_clustered_score": {
            **geometry,
            "secondary_only": True,
            "cannot_flip_primary": True,
        },
        "by_arm": by_arm,
        "by_phase": by_phase,
        "completed_games": len(records),
        "completed_starts": len(starts),
        "all_rules_terminal": all(
            row["termination_class"] == "rules_terminal" for row in records
        ),
    }


def compact_game(record: Mapping[str, Any]) -> dict[str, Any]:
    """Strip bulky semantic search objects while preserving the move audit."""
    validate_game_record(record, require_decomposition=True)
    turns = []
    for row in record["turns"]:
        engine = row["engine_response"]
        if engine is not None:
            engine = {key: value for key, value in engine.items() if key != "semantic_search"}
            engine["semantic_search_sha256"] = canonical_sha256(
                row["engine_response"]["semantic_search"]
            )
        choice = row["candidate_choice"]
        if choice is not None and choice.get("risks") is not None:
            choice = dict(choice)
            choice["risks_identity"] = canonical_sha256(choice.pop("risks"))
        turns.append({**row, "candidate_choice": choice, "engine_response": engine})
    events = []
    for event in record["induced_events"]:
        value = dict(event)
        history = value.pop("history_actions_before")
        value["history_actions_before_count"] = len(history)
        value["history_actions_before_identity"] = canonical_sha256(history)
        value["primary_semantic_search_sha256"] = canonical_sha256(
            value.pop("primary_semantic_search")
        )
        for budget in (1_000, 500_000):
            semantic = value.pop(f"semantic_search_{budget}")
            value[f"semantic_search_{budget}_sha256"] = canonical_sha256(semantic)
        events.append(value)
    return {
        key: value
        for key, value in {
            **record,
            "turns": turns,
            "induced_events": events,
        }.items()
        if key != "previous_record_sha256"
    }


__all__ = [
    "ARMS",
    "ATTEMPT_SPEC_SCHEMA",
    "AUTHORIZATION_SCHEMA",
    "DECOMPOSITION_BUDGETS",
    "EXPECTED_GAMES",
    "EXPECTED_STARTS",
    "GAME_SCHEMA",
    "MAX_POST_START_LOGICAL_PLIES",
    "PHASES",
    "PLAN_SCHEMA",
    "POOL_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "PRIMARY_NODE_BUDGET",
    "RESULT_SCHEMA",
    "REHEARSAL_RESULT_SCHEMA",
    "RESOURCE_CHECKPOINT_SCHEMA",
    "ResourceLedger",
    "SafeGuidanceGameplayError",
    "SafeGuidanceGameplayIncomplete",
    "analyze_rehearsal_records",
    "analyze_games",
    "append_game_record",
    "append_resource_checkpoint",
    "build_schedule",
    "classify_induced_events",
    "compact_game",
    "load_authorization",
    "load_attempt_spec",
    "load_plan",
    "load_pool",
    "load_preflight",
    "load_resource_checkpoints",
    "load_sealed",
    "play_game",
    "replay_scripted_rehearsal_game",
    "run_guide_canary",
    "sha256_file",
    "select_schedule_excluding_starts",
    "validate_game_record",
    "write_json_atomic",
]
