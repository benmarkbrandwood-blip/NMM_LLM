"""Deterministic strength measurement for the frozen ``origin/main`` search.

The strict game/referee code is the current verified evaluation harness.  The
candidate policy is loaded from an exported ``origin/main`` source tree under
an isolated package name, together with a wheel built from that tree's Rust
extension.  This keeps the product search semantics separate from later
``dev`` changes without starting a second measurement process.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from ai.malom_db import MalomDB
from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.evaluation.human_f0h0_feasibility import canonical_sha256
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    ResourceLedger,
    SafeGuidanceGameplayError,
    _checked_oracle_inventory,
    _checked_position_state,
    _checked_search_result,
    _final_positional_tier,
    _matching_move,
    _move_key,
    _normal_move,
    _phase,
    _strict_terminal_outcome,
    replay_start,
    sha256_file,
)
from learned_ai.evaluation.sanmill_safe_inducement import WDL_RANK
from learned_ai.training.sanmill_referee import SanmillTrainingGame


CALIBRATION_PLAN_SCHEMA = "nmm.sanmill-classical-search-calibration-plan.v1"
CALIBRATION_RESULT_SCHEMA = "nmm.sanmill-classical-search-calibration-result.v1"
PLAN_SCHEMA = "nmm.sanmill-classical-search-strength-plan.v1"
AUTHORIZATION_SCHEMA = "nmm.sanmill-classical-search-strength-authorization.v1"
GAME_SCHEMA = "nmm.sanmill-classical-search-strength-game.v1"
RESULT_SCHEMA = "nmm.sanmill-classical-search-strength-result.v1"

PHASES = ("placement", "movement", "flying")
REFERENCE_ARMS = (
    "random-safe",
    "retained-v4-free",
    "retained-v4-a-pos",
    "active-specialists-free",
    "active-specialists-a-pos",
)
MAX_POST_START_LOGICAL_PLIES = 1536
SANMILL_NODE_BUDGET = 100_000


class ClassicalSearchStrengthError(SafeGuidanceGameplayError):
    """Raised when a frozen measurement contract differs."""


def move_key(move: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return a stable nullable move key."""

    return (
        str(move.get("from") or ""),
        str(move.get("to") or ""),
        str(move.get("capture") or ""),
    )


def board_from_state(state: Mapping[str, Any]) -> BoardState:
    """Replay one frozen source state without consulting outcome fields."""

    board = BoardState.new_game()
    for actions in state["logical_turns"]:
        board = board.apply_move(_matching_move(board, actions))
    if board.to_fen_string() != state["fen"]:
        raise ClassicalSearchStrengthError("frozen state replay differs")
    return board


def phase_balanced_membership(
    states: Sequence[Mapping[str, Any]],
    *,
    count: int,
    namespace: str,
    excluded_start_ids: Sequence[str] = (),
) -> list[str]:
    """Select a result-blind phase-balanced subset by namespaced SHA-256."""

    if count <= 0:
        raise ClassicalSearchStrengthError("membership count is not positive")
    excluded = set(excluded_start_ids)
    base, remainder = divmod(count, len(PHASES))
    quotas = {
        phase: base + (1 if index < remainder else 0)
        for index, phase in enumerate(PHASES)
    }
    selected: list[str] = []
    for phase in PHASES:
        candidates = [
            str(row["state_id"])
            for row in states
            if row["phase"] == phase and str(row["state_id"]) not in excluded
        ]
        candidates.sort(
            key=lambda state_id: hashlib.sha256(
                f"{namespace}|{phase}|{state_id}".encode("utf-8")
            ).hexdigest()
        )
        if len(candidates) < quotas[phase]:
            raise ClassicalSearchStrengthError("phase membership is undersized")
        selected.extend(candidates[: quotas[phase]])
    if len(selected) != count or len(set(selected)) != count:
        raise ClassicalSearchStrengthError("phase membership differs")
    return sorted(selected)


def calibration_membership(
    states: Sequence[Mapping[str, Any]],
    *,
    per_phase: int,
    namespace: str,
    excluded_start_ids: Sequence[str],
) -> list[str]:
    """Select cold-search calibration states before any product observation."""

    excluded = set(excluded_start_ids)
    selected: list[str] = []
    for phase in PHASES:
        candidates = []
        for row in states:
            if row["phase"] != phase or str(row["state_id"]) in excluded:
                continue
            if phase == "placement":
                board_field = str(row["fen"]).split("|", maxsplit=1)[0]
                if sum(value in "WB" for value in board_field) <= 4:
                    continue
            candidates.append(str(row["state_id"]))
        candidates.sort(
            key=lambda state_id: hashlib.sha256(
                f"{namespace}|{phase}|{state_id}".encode("utf-8")
            ).hexdigest()
        )
        if len(candidates) < per_phase:
            raise ClassicalSearchStrengthError("calibration phase is undersized")
        selected.extend(candidates[:per_phase])
    if len(selected) != per_phase * len(PHASES):
        raise ClassicalSearchStrengthError("calibration membership differs")
    return selected


def _load_alias_package(name: str, package_dir: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ClassicalSearchStrengthError("product package spec is absent")
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return package


@dataclass
class SearchObservation:
    """One product search observation used by calibration or a game turn."""

    move: dict[str, str | None]
    elapsed_seconds: float
    nodes: int
    completed_depth: int
    thinking: str
    bypassed_search: bool

    def record(self) -> dict[str, Any]:
        return {
            "move": self.move,
            "elapsed_seconds": self.elapsed_seconds,
            "nodes": self.nodes,
            "completed_depth": self.completed_depth,
            "thinking": self.thinking,
            "bypassed_search": self.bypassed_search,
        }


class ProductMainRuntime:
    """Load the exact frozen ``main`` search in the current process."""

    def __init__(
        self,
        *,
        product_root: Path,
        native_site: Path,
        resource_root: Path,
        expected: Mapping[str, Any],
    ) -> None:
        self.product_root = product_root.resolve()
        self.native_site = native_site.resolve()
        self.resource_root = resource_root.resolve()
        if "nmm_core" in sys.modules:
            raise ClassicalSearchStrengthError(
                "nmm_core was loaded before the frozen product runtime"
            )
        sys.path.insert(0, str(self.native_site))
        _load_alias_package("frozen_product_main_ai", self.product_root / "ai")
        self.game_ai_module = importlib.import_module(
            "frozen_product_main_ai.game_ai"
        )
        self.heuristics_module = importlib.import_module(
            "frozen_product_main_ai.heuristics"
        )
        self.fullgame_module = importlib.import_module(
            "frozen_product_main_ai.fullgame_db"
        )
        self.endgame_module = importlib.import_module(
            "frozen_product_main_ai.endgame_solved_db"
        )
        self.value_module = importlib.import_module(
            "frozen_product_main_ai.value_net"
        )
        native = importlib.import_module("nmm_core")
        observed_files = {
            "game_ai": sha256_file(Path(self.game_ai_module.__file__)),
            "heuristics": sha256_file(Path(self.heuristics_module.__file__)),
            "native_extension": sha256_file(
                next(
                    path
                    for path in (self.native_site / "nmm_core").iterdir()
                    if path.suffix == ".pyd"
                )
            ),
        }
        if observed_files != expected["implementation_sha256"]:
            raise ClassicalSearchStrengthError("product implementation hash differs")
        native_path = Path(native.__file__).resolve()
        if self.native_site not in native_path.parents:
            raise ClassicalSearchStrengthError("product native extension source differs")

        weights_path = self.resource_root / "data" / "weights" / "best.json"
        evolved = json.loads(weights_path.read_text(encoding="utf-8"))
        if sha256_file(weights_path) != expected["resource_sha256"][
            "evolved_weights"
        ]:
            raise ClassicalSearchStrengthError("product evolved weights differ")
        self.weights = self._canonical_weights(evolved)
        if self.weights.__dict__ != expected["resolved_weights"]:
            raise ClassicalSearchStrengthError("resolved product weights differ")

        fullgame_path = self.resource_root / "data" / "endgame" / "fullgame.bin"
        endgame_path = self.resource_root / "data" / "endgame"
        value_base = self.resource_root / "data" / "value_net_phase"
        gap_path = self.resource_root / "data" / "gap_net.npz"
        self.fullgame = self.fullgame_module.FullGameDB(fullgame_path)
        self.endgame = self.endgame_module.EndgameSolvedDB(endgame_path)
        self.value_net = self.value_module.PhaseValueNet.load_if_exists(value_base)
        self.gap_net = self.value_module.ValueNet.load_if_exists(gap_path)
        if (
            not self.fullgame.is_available()
            or not self.endgame.is_available()
            or self.value_net is None
            or self.gap_net is None
        ):
            raise ClassicalSearchStrengthError("product resource is unavailable")
        self._assert_resource_hashes(expected["resource_sha256"])

    def _assert_resource_hashes(self, expected: Mapping[str, str]) -> None:
        paths = {
            "fullgame_db": self.resource_root / "data/endgame/fullgame.bin",
            "value_place": self.resource_root / "data/value_net_phase_place.npz",
            "value_move": self.resource_root / "data/value_net_phase_move.npz",
            "value_fly": self.resource_root / "data/value_net_phase_fly.npz",
            "gap_net": self.resource_root / "data/gap_net.npz",
        }
        observed = {key: sha256_file(path) for key, path in paths.items()}
        observed["evolved_weights"] = expected["evolved_weights"]
        for path in sorted((self.resource_root / "data/endgame").glob("*.wdl")):
            observed[f"endgame/{path.name}"] = sha256_file(path)
        if observed != dict(expected):
            raise ClassicalSearchStrengthError("product resource hash differs")

    def _canonical_weights(self, evolved: Mapping[str, Any]) -> Any:
        def weight(key: str, default: int) -> int:
            return int(evolved.get(key, default))

        return self.heuristics_module.HeuristicWeights(
            close_mill=weight("close_mill", 500),
            cycling_mill=weight("cycling_mill", 300),
            block_opponent_mill=weight("block_opponent_mill", 400),
            stop_opponent_mills=weight("stop_opponent_mills", 450),
            feeder_diamond=weight("feeder_diamond", 200),
            mill_wrapping=weight("mill_wrapping", 150),
            cardinal_block=weight("cardinal_block", 400),
            scatter_placement=weight("scatter_placement", 100),
            setup_mill=weight("setup_mill", 150),
            mill_opening=weight("mill_opening", 200),
            long_term_position=weight("long_term_position", 100),
            mill_count_scale=weight("mill_count_scale", 100),
            mobility_scale=weight("mobility_scale", 100),
            blocked_scale=weight("blocked_scale", 100),
            make_mistakes=weight("make_mistakes", 0),
            opening_adherence=weight("opening_adherence", 50),
            value_net_blend=weight("value_net_blend", 80),
            humanlike_blend=weight("humanlike_blend", 0),
            cross_mill_cycling=weight("cross_mill_cycling", 300),
            move_variance_pct=weight("move_variance_pct", 0),
        )

    def new_ai(
        self,
        *,
        color: str,
        difficulty: int,
        node_budget: int | None,
        search_threads: int,
        max_depth: int,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if node_budget is not None:
            kwargs["override_node_budget"] = int(node_budget)
        ai = self.game_ai_module.GameAI(
            color=color,
            difficulty=difficulty,
            weights=self.weights,
            blunder_probability=0.0,
            fullgame_db=self.fullgame,
            endgame_solved_db=self.endgame,
            malom_db=None,
            value_net=self.value_net,
            gap_net=self.gap_net,
            **kwargs,
        )
        ai.max_search_depth = int(max_depth)
        ai.search_threads = int(search_threads)
        ai.use_extended_qsearch = True
        ai.suppress_fork_variety = False
        ai.star_square_mode = ""
        ai.use_ngram_search = False
        ai._ngram_model = None
        return ai

    @staticmethod
    def choose(ai: Any, board: BoardState) -> SearchObservation:
        started = time.perf_counter()
        move = ai.choose_move(board)
        elapsed = time.perf_counter() - started
        legal = {move_key(candidate) for candidate in get_all_legal_moves(board)}
        if move_key(move) not in legal:
            raise ClassicalSearchStrengthError("product search returned an illegal move")
        nodes = int(ai._nodes)
        depth = int(ai.last_depth_reached)
        return SearchObservation(
            move=_normal_move(move),
            elapsed_seconds=elapsed,
            nodes=nodes,
            completed_depth=depth,
            thinking=str(ai.last_thinking),
            bypassed_search=nodes == 0,
        )

    def close(self) -> None:
        for resource in (self.fullgame, self.endgame):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        gc.collect()


def calibration_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen node-budget mapping to timing-only observations."""

    by_difficulty: dict[str, Any] = {}
    for difficulty in (9, 10):
        rows = [row for row in records if int(row["difficulty"]) == difficulty]
        searched = [row for row in rows if int(row["nodes"]) > 0]
        if len(searched) < 3:
            raise ClassicalSearchStrengthError("too few timed search observations")
        nodes = [int(row["nodes"]) for row in searched]
        elapsed = [float(row["elapsed_seconds"]) for row in searched]
        raw_median = statistics.median(nodes)
        mapped = max(1_000, int(raw_median // 1_000) * 1_000)
        by_difficulty[str(difficulty)] = {
            "states": len(rows),
            "searched_states": len(searched),
            "bypassed_states": len(rows) - len(searched),
            "nodes": {
                "minimum": min(nodes),
                "median": raw_median,
                "maximum": max(nodes),
            },
            "elapsed_seconds": {
                "minimum": min(elapsed),
                "median": statistics.median(elapsed),
                "maximum": max(elapsed),
            },
            "mapped_node_budget": mapped,
            "completed_depths": {
                str(depth): count
                for depth, count in Counter(
                    int(row["completed_depth"]) for row in searched
                ).items()
            },
        }
    return by_difficulty


def paired_interval(values: Sequence[float]) -> dict[str, float | int]:
    """Start-clustered normal 95% interval."""

    if len(values) < 2:
        raise ClassicalSearchStrengthError("paired interval support is too small")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    half_width = 1.96 * standard_deviation / math.sqrt(len(values))
    return {
        "support": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "half_width": half_width,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def prior_scores_by_start(
    manifest: Mapping[str, Any],
    *,
    start_ids: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Recompute every old arm on an exact frozen start subset."""

    wanted = set(start_ids)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in manifest["machine_records"]["compact_records"]:
        arm = str(row["arm"])
        start_id = str(row["start_id"])
        if arm in REFERENCE_ARMS and start_id in wanted:
            grouped[arm][start_id].append(float(row["candidate_score"]))
    result: dict[str, dict[str, float]] = {}
    for arm in REFERENCE_ARMS:
        values = grouped.get(arm, {})
        if set(values) != wanted or any(len(items) != 2 for items in values.values()):
            raise ClassicalSearchStrengthError("prior arm subset coverage differs")
        result[arm] = {
            start_id: statistics.fmean(items) for start_id, items in values.items()
        }
    return result


def known_answer_fingerprint(record: Mapping[str, Any]) -> dict[str, Any]:
    """Exact per-game gate fields, including every logical turn and clock."""

    turns = record["turns"]
    return {
        "game_id": record["game_id"],
        "start_id": record["start_id"],
        "candidate_color": record["candidate_color"],
        "candidate_score": record["candidate_score"],
        "winner": record["winner"],
        "outcome_reason": record["outcome_reason"],
        "post_start_logical_plies": record["post_start_logical_plies"],
        "final_history_sha256": record["final_state"]["history_sha256"],
        "final_no_capture_count": record["final_state"]["no_capture_count"],
        "final_repetition_current_count": record["final_state"][
            "repetition_current_count"
        ],
        "turns": [
            {
                "actions": turn["actions"],
                "history_sha256_after": turn["history_sha256_after"],
                "no_capture_count": turn["no_capture_count"],
                "repetition_current_count": turn["repetition_current_count"],
                "outcome_reason": turn["outcome_reason"],
            }
            for turn in turns
        ],
    }


def exact_subset_gate(
    observed: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless the selected known-answer games match exactly."""

    observed_by_id = {
        str(row["game_id"]): known_answer_fingerprint(row) for row in observed
    }
    reference_by_id = {
        str(row["game_id"]): known_answer_fingerprint(row) for row in reference
    }
    differing = sorted(
        game_id
        for game_id in set(observed_by_id) | set(reference_by_id)
        if observed_by_id.get(game_id) != reference_by_id.get(game_id)
    )
    return {
        "passed": not differing and len(observed_by_id) == len(reference_by_id),
        "games": len(observed_by_id),
        "differing_game_ids": differing,
        "observed_identity": canonical_sha256(observed_by_id),
        "reference_identity": canonical_sha256(reference_by_id),
    }


def _classical_choice(
    *,
    board: BoardState,
    ai: Any,
    database: MalomDB,
    ledger: ResourceLedger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent, inventory, query_count = _checked_oracle_inventory(board, database)
    ledger.add_malom(query_count)
    inventory_by_key = {_move_key(move): value for move, value in inventory}
    observation = ProductMainRuntime.choose(ai, board)
    key = _move_key(observation.move)
    if key not in inventory_by_key:
        raise ClassicalSearchStrengthError("classical move is absent from Malom inventory")
    after = inventory_by_key[key].outcome
    transition = (
        f"{parent}->{after}" if WDL_RANK[after] < WDL_RANK[parent] else None
    )
    if transition not in {None, "W->D", "W->L", "D->L"}:
        raise ClassicalSearchStrengthError("classical downgrade transition differs")
    ledger.add_engine()
    return dict(observation.move), {
        "safety_mode": "observed-free",
        "safe_set": None,
        "positional_only": True,
        "parent_tier": parent,
        "selected_after_tier": after,
        "a_pos_cardinality": sum(value.outcome == parent for _, value in inventory),
        "self_downgrade_transition": transition,
        "search": observation.record(),
    }


def play_classical_game(
    *,
    schedule_item: Mapping[str, Any],
    start_state: Mapping[str, Any],
    product_runtime: ProductMainRuntime,
    product_contract: Mapping[str, Any],
    database: MalomDB,
    installation: Any,
    ledger: ResourceLedger,
) -> dict[str, Any]:
    """Play one strict complete game with one classical candidate color."""

    started = time.perf_counter()
    candidate_color = str(schedule_item["candidate_color"])
    ai = product_runtime.new_ai(
        color=candidate_color,
        difficulty=int(schedule_item["difficulty"]),
        node_budget=int(schedule_item["node_budget"]),
        search_threads=int(product_contract["deterministic_search_threads"]),
        max_depth=int(product_contract["max_depth"]),
    )
    turns: list[dict[str, Any]] = []
    safety_cap = False
    with SanmillTrainingGame(
        installation, seed=int(product_contract["sanmill_seed"])
    ) as game:
        board, strict_start = replay_start(game, start_state, ledger)
        for post_start_ply in range(1, MAX_POST_START_LOGICAL_PLIES + 1):
            ledger.require_within()
            mover = board.turn
            phase = _phase(board, mover)
            before_history = game.state.history_sha256
            if mover == candidate_color:
                move, choice = _classical_choice(
                    board=board, ai=ai, database=database, ledger=ledger
                )
                applied = game.apply_nmm_move(board, move)
                actor = "classical-search"
                engine = None
            else:
                ledger.add_engine()
                result = game.session.search_logical_turn(SANMILL_NODE_BUDGET)
                engine = _checked_search_result(
                    result, expected_node_budget=SANMILL_NODE_BUDGET
                )
                if result.model_action is None:
                    raise ClassicalSearchStrengthError("Sanmill returned no action")
                move = result.model_action
                applied = game.apply_nmm_move(board, move, search_result=result)
                actor = "sanmill"
                choice = None
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
                    "engine_search": engine,
                }
            )
            if game.state.terminal:
                break
        else:
            safety_cap = True
        terminal_state = game.state
    del ai
    gc.collect()
    final_state = dict(_checked_position_state(terminal_state))
    if safety_cap:
        winner = None
        score = None
        termination = "safety_cap_incomplete"
        reason = "safety_cap_incomplete"
    else:
        winner, reason = _strict_terminal_outcome(terminal_state)
        score = 0.5 if winner is None else float(winner == candidate_color)
        termination = "rules_terminal"
    final_tier, final_queries = _final_positional_tier(board, database)
    ledger.add_malom(final_queries)
    events = [
        {
            "post_start_ply": turn["post_start_ply"],
            "phase": turn["phase"],
            "transition": turn["candidate_choice"]["self_downgrade_transition"],
            "move": turn["move"],
        }
        for turn in turns
        if turn["actor"] == "classical-search"
        and turn["candidate_choice"]["self_downgrade_transition"] is not None
    ]
    return {
        "schema_version": GAME_SCHEMA,
        "ordinal": int(schedule_item["ordinal"]),
        "game_id": str(schedule_item["game_id"]),
        "unit_index": int(schedule_item["unit_index"]),
        "start_id": str(schedule_item["start_id"]),
        "phase": str(schedule_item["phase"]),
        "arm": str(schedule_item["arm"]),
        "difficulty": int(schedule_item["difficulty"]),
        "node_budget": int(schedule_item["node_budget"]),
        "candidate_color": candidate_color,
        "strict_start": strict_start,
        "post_start_logical_plies": len(turns),
        "termination_class": termination,
        "outcome_reason": reason,
        "winner": winner,
        "candidate_score": score,
        "final_state": final_state,
        "final_positional": {
            "side_to_move": board.turn,
            "side_to_move_wdl": final_tier,
            "history_aware": False,
        },
        "turns": turns,
        "self_downgrade_events": events,
        "game_elapsed_seconds": time.perf_counter() - started,
    }


def compact_game(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a tracked per-game record while raw turn ledgers remain local."""

    searches = [
        turn["candidate_choice"]["search"]
        for turn in record["turns"]
        if turn["actor"] == "classical-search"
    ]
    return {
        key: record[key]
        for key in (
            "schema_version",
            "ordinal",
            "game_id",
            "unit_index",
            "start_id",
            "phase",
            "arm",
            "difficulty",
            "node_budget",
            "candidate_color",
            "post_start_logical_plies",
            "termination_class",
            "outcome_reason",
            "winner",
            "candidate_score",
            "game_elapsed_seconds",
        )
    } | {
        "final_history_sha256": record["final_state"]["history_sha256"],
        "self_downgrade_events": len(record["self_downgrade_events"]),
        "self_downgrade_transitions": dict(
            Counter(event["transition"] for event in record["self_downgrade_events"])
        ),
        "candidate_searches": len(searches),
        "candidate_search_nodes": [int(row["nodes"]) for row in searches],
        "candidate_search_depths": [int(row["completed_depth"]) for row in searches],
        "candidate_search_seconds": [float(row["elapsed_seconds"]) for row in searches],
        "candidate_search_bypasses": sum(bool(row["bypassed_search"]) for row in searches),
    }


def analyze_games(
    records: Sequence[Mapping[str, Any]],
    *,
    prior_scores: Mapping[str, Mapping[str, float]],
    start_ids: Sequence[str],
    maximum_half_width: float,
) -> dict[str, Any]:
    """Analyze frozen budget points against every old arm on the same starts."""

    wanted = set(start_ids)
    by_arm: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for arm in sorted({str(row["arm"]) for row in records}):
        arm_rows = [row for row in records if row["arm"] == arm]
        if any(row["termination_class"] != "rules_terminal" for row in arm_rows):
            raise ClassicalSearchStrengthError("safety-cap game prevents WDL analysis")
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in arm_rows:
            grouped[str(row["start_id"])].append(float(row["candidate_score"]))
        if set(grouped) != wanted or any(len(values) != 2 for values in grouped.values()):
            raise ClassicalSearchStrengthError("classical start/color coverage differs")
        scores = {key: statistics.fmean(values) for key, values in grouped.items()}
        wins = sum(row["candidate_score"] == 1.0 for row in arm_rows)
        draws = sum(row["candidate_score"] == 0.5 for row in arm_rows)
        losses = sum(row["candidate_score"] == 0.0 for row in arm_rows)
        searches = [
            turn["candidate_choice"]["search"]
            for row in arm_rows
            for turn in row["turns"]
            if turn["actor"] == "classical-search"
        ]
        events = [event for row in arm_rows for event in row["self_downgrade_events"]]
        candidate_turns = len(searches)
        by_arm[arm] = {
            "games": len(arm_rows),
            "starts": len(scores),
            "strict_wdl": {
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "score_rate": statistics.fmean(scores.values()),
            },
            "terminal_reasons": dict(Counter(row["outcome_reason"] for row in arm_rows)),
            "self_downgrade": {
                "events": len(events),
                "candidate_turns": candidate_turns,
                "event_rate": len(events) / candidate_turns if candidate_turns else None,
                "transitions": dict(Counter(event["transition"] for event in events)),
                "by_phase": dict(Counter(event["phase"] for event in events)),
            },
            "work": {
                "candidate_searches": candidate_turns,
                "bypassed_searches": sum(bool(row["bypassed_search"]) for row in searches),
                "nodes": {
                    "minimum": min(int(row["nodes"]) for row in searches),
                    "median": statistics.median(int(row["nodes"]) for row in searches),
                    "maximum": max(int(row["nodes"]) for row in searches),
                },
                "completed_depths": {
                    str(depth): count
                    for depth, count in Counter(
                        int(row["completed_depth"]) for row in searches
                    ).items()
                },
                "elapsed_seconds": {
                    "median": statistics.median(float(row["elapsed_seconds"]) for row in searches),
                    "maximum": max(float(row["elapsed_seconds"]) for row in searches),
                },
            },
        }
        for reference_arm in REFERENCE_ARMS:
            differences = [
                scores[start_id] - float(prior_scores[reference_arm][start_id])
                for start_id in sorted(wanted)
            ]
            interval = paired_interval(differences)
            if interval["lower"] > 0:
                direction = "classical_higher"
            elif interval["upper"] < 0:
                direction = "classical_lower"
            else:
                direction = "direction_inconclusive"
            contrasts[f"{arm}_minus_{reference_arm}"] = {
                **interval,
                "direction": direction,
                "precision_adequate": interval["half_width"] <= maximum_half_width,
            }
    return {
        "by_arm": by_arm,
        "contrasts": contrasts,
        "maximum_half_width": maximum_half_width,
    }


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "CALIBRATION_PLAN_SCHEMA",
    "CALIBRATION_RESULT_SCHEMA",
    "ClassicalSearchStrengthError",
    "GAME_SCHEMA",
    "PLAN_SCHEMA",
    "ProductMainRuntime",
    "RESULT_SCHEMA",
    "analyze_games",
    "board_from_state",
    "calibration_membership",
    "calibration_summary",
    "compact_game",
    "exact_subset_gate",
    "known_answer_fingerprint",
    "paired_interval",
    "phase_balanced_membership",
    "play_classical_game",
    "prior_scores_by_start",
]
