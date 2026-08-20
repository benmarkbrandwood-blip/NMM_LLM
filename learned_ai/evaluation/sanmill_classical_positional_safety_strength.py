"""Paired strength measurement for the delivered classical ``A_pos`` gate.

The module deliberately reuses the verified strict-game harness and the real
``ProductPositionalSafetyGate``.  It does not reproduce the safety rule in an
evaluation-only policy.  The only glue supplied here is the same deterministic
fixed-depth restricted-root selector used by ``web.app`` after the gate has
proved the complete positional-safe move set.
"""

from __future__ import annotations

import gc
import json
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.endgame_solved_db import EndgameSolvedDB
from ai.fullgame_db import FullGameDB
from ai.game_ai import GameAI
from ai.heuristics import HeuristicWeights
from ai.malom_db import MalomDB
from ai.value_net import PhaseValueNet, ValueNet
from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.agents.positional_safety import ProductPositionalSafetyGate
from learned_ai.evaluation.human_f0h0_feasibility import canonical_sha256
from learned_ai.evaluation.sanmill_classical_search_strength import (
    SearchObservation,
    paired_interval,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    ResourceLedger,
    SafeGuidanceGameplayError,
    _checked_oracle_inventory,
    _checked_position_state,
    _checked_search_result,
    _final_positional_tier,
    _move_key,
    _normal_move,
    _phase,
    _strict_terminal_outcome,
    replay_start,
    sha256_file,
)
from learned_ai.evaluation.sanmill_safe_inducement import WDL_RANK
from learned_ai.training.sanmill_referee import SanmillTrainingGame


PLAN_SCHEMA = "nmm.sanmill-classical-positional-safety-strength-plan.v1"
AUTHORIZATION_SCHEMA = (
    "nmm.sanmill-classical-positional-safety-strength-authorization.v1"
)
GAME_SCHEMA = "nmm.sanmill-classical-positional-safety-strength-game.v1"
RESULT_SCHEMA = "nmm.sanmill-classical-positional-safety-strength-result.v1"

MAX_POST_START_LOGICAL_PLIES = 1536
SANMILL_NODE_BUDGET = 100_000
DIFFICULTIES = (9, 10)


class ClassicalPositionalSafetyStrengthError(SafeGuidanceGameplayError):
    """Raised when the frozen paired measurement contract differs."""


def _move_tuple(move: Mapping[str, Any]) -> tuple[str, str, str]:
    return tuple(str(move.get(field) or "") for field in ("from", "to", "capture"))


def restricted_root_select(
    ranked: Sequence[tuple[Mapping[str, Any], float]],
) -> dict[str, Any]:
    """Select the best restricted-root result with the product tie break."""

    if not ranked:
        raise ClassicalPositionalSafetyStrengthError(
            "restricted root research returned no move"
        )
    numeric = [(dict(move), float(score)) for move, score in ranked]
    best_score = max(score for _move, score in numeric)
    tied = [move for move, score in numeric if score == best_score]
    return min(tied, key=_move_tuple)


def _canonical_weights(evolved: Mapping[str, Any]) -> HeuristicWeights:
    """Construct the exact web product weights from the tracked JSON."""

    def weight(key: str, default: int) -> int:
        return int(evolved.get(key, default))

    return HeuristicWeights(
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
        cross_mill_cycling=weight("cross_mill_cycling", 300),
        move_variance_pct=weight("move_variance_pct", 0),
    )


@dataclass
class RerankObservation:
    """Work performed only when the delivered gate rewrites a move."""

    called: bool = False
    elapsed_seconds: float = 0.0
    nodes: int = 0
    completed_depth: int = 0
    candidate_moves: int = 0
    returned_moves: int = 0

    def record(self) -> dict[str, Any]:
        return {
            "called": self.called,
            "elapsed_seconds": self.elapsed_seconds,
            "nodes": self.nodes,
            "completed_depth": self.completed_depth,
            "candidate_moves": self.candidate_moves,
            "returned_moves": self.returned_moves,
        }


class ProductDevRuntime:
    """Load the current delivered ``dev`` classical coordinator directly."""

    def __init__(self, *, resource_root: Path, expected: Mapping[str, Any]) -> None:
        self.resource_root = resource_root.resolve()
        import nmm_core

        native_module = getattr(nmm_core, "nmm_core", None)
        native_path = Path(str(getattr(native_module, "__file__", "")))
        if not native_path.is_file():
            raise ClassicalPositionalSafetyStrengthError(
                "active dev native extension is unavailable"
            )
        observed = {
            "game_ai": sha256_file(self.resource_root / "ai/game_ai.py"),
            "heuristics": sha256_file(self.resource_root / "ai/heuristics.py"),
            "native_extension": sha256_file(native_path),
        }
        if observed != expected["implementation_sha256"]:
            raise ClassicalPositionalSafetyStrengthError(
                "dev product implementation hash differs"
            )

        weights_path = self.resource_root / "data/weights/best.json"
        evolved = json.loads(weights_path.read_text(encoding="utf-8"))
        self.weights = _canonical_weights(evolved)
        if dict(self.weights.__dict__) != dict(expected["resolved_weights"]):
            raise ClassicalPositionalSafetyStrengthError(
                "resolved dev product weights differ"
            )

        self.fullgame = FullGameDB(self.resource_root / "data/endgame/fullgame.bin")
        self.endgame = EndgameSolvedDB(self.resource_root / "data/endgame")
        self.value_net = PhaseValueNet.load_if_exists(
            self.resource_root / "data/value_net_phase"
        )
        self.gap_net = ValueNet.load_if_exists(
            self.resource_root / "data/gap_net.npz"
        )
        if (
            not self.fullgame.is_available()
            or not self.endgame.is_available()
            or self.value_net is None
            or self.gap_net is None
        ):
            raise ClassicalPositionalSafetyStrengthError(
                "dev product resource is unavailable"
            )
        self._assert_resource_hashes(expected["resource_sha256"])

    def _assert_resource_hashes(self, expected: Mapping[str, str]) -> None:
        paths = {
            "evolved_weights": self.resource_root / "data/weights/best.json",
            "fullgame_db": self.resource_root / "data/endgame/fullgame.bin",
            "value_place": self.resource_root / "data/value_net_phase_place.npz",
            "value_move": self.resource_root / "data/value_net_phase_move.npz",
            "value_fly": self.resource_root / "data/value_net_phase_fly.npz",
            "gap_net": self.resource_root / "data/gap_net.npz",
        }
        observed = {name: sha256_file(path) for name, path in paths.items()}
        for path in sorted((self.resource_root / "data/endgame").glob("*.wdl")):
            observed[f"endgame/{path.name}"] = sha256_file(path)
        if observed != dict(expected):
            raise ClassicalPositionalSafetyStrengthError(
                "dev product resource hash differs"
            )

    def new_ai(
        self,
        *,
        color: str,
        difficulty: int,
        node_budget: int,
        search_threads: int,
        max_depth: int,
        malom_adapter: Any,
    ) -> GameAI:
        ai = GameAI(
            color=color,
            difficulty=difficulty,
            weights=self.weights,
            blunder_probability=0.0,
            fullgame_db=self.fullgame,
            endgame_solved_db=self.endgame,
            malom_db=malom_adapter,
            value_net=self.value_net,
            gap_net=self.gap_net,
            override_node_budget=int(node_budget),
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
    def choose(ai: GameAI, board: BoardState) -> SearchObservation:
        started = time.perf_counter()
        move = ai.choose_move(board)
        elapsed = time.perf_counter() - started
        legal = {_move_tuple(candidate) for candidate in get_all_legal_moves(board)}
        if _move_tuple(move) not in legal:
            raise ClassicalPositionalSafetyStrengthError(
                "dev product search returned an illegal move"
            )
        return SearchObservation(
            move=_normal_move(move),
            elapsed_seconds=elapsed,
            nodes=int(ai._nodes),
            completed_depth=int(ai.last_depth_reached),
            thinking=str(ai.last_thinking),
            bypassed_search=int(ai._nodes) == 0,
        )

    def close(self) -> None:
        for resource in (self.fullgame, self.endgame):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        gc.collect()


def _unfiltered_choice(
    *,
    board: BoardState,
    ai: GameAI,
    database: MalomDB,
    ledger: ResourceLedger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent, inventory, query_count = _checked_oracle_inventory(board, database)
    ledger.add_malom(query_count)
    by_key = {_move_key(move): value for move, value in inventory}
    observation = ProductDevRuntime.choose(ai, board)
    key = _move_key(observation.move)
    if key not in by_key:
        raise ClassicalPositionalSafetyStrengthError(
            "unfiltered move is absent from Malom inventory"
        )
    after = by_key[key].outcome
    transition = f"{parent}->{after}" if WDL_RANK[after] < WDL_RANK[parent] else None
    if transition not in {None, "W->D", "W->L", "D->L"}:
        raise ClassicalPositionalSafetyStrengthError(
            "unfiltered downgrade transition differs"
        )
    ledger.add_engine()
    return dict(observation.move), {
        "safety_mode": "unfiltered",
        "positional_only": True,
        "parent_tier": parent,
        "original_after_tier": after,
        "selected_after_tier": after,
        "a_pos_cardinality": sum(value.outcome == parent for _, value in inventory),
        "self_downgrade_transition": transition,
        "intervened": False,
        "gate_decision": None,
        "search": observation.record(),
        "restricted_root_research": RerankObservation().record(),
    }


def _filtered_choice(
    *,
    board: BoardState,
    ai: GameAI,
    gate: ProductPositionalSafetyGate,
    difficulty: int,
    ledger: ResourceLedger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    observation = ProductDevRuntime.choose(ai, board)
    ledger.add_engine()
    rerank = RerankObservation()

    def safe_selector(safe_moves: list[dict[str, Any]]) -> Mapping[str, Any]:
        started = time.perf_counter()
        ranked = ai.score_root_moves(
            board,
            depth=2,
            time_budget=None,
            preserve_tt=False,
            candidate_moves=safe_moves,
        )
        rerank.called = True
        rerank.elapsed_seconds = time.perf_counter() - started
        rerank.nodes = int(ai._nodes)
        rerank.completed_depth = int(ai.last_depth_reached)
        rerank.candidate_moves = len(safe_moves)
        rerank.returned_moves = len(ranked)
        ledger.add_engine()
        return restricted_root_select(ranked)

    outcome = gate.constrain(
        board,
        observation.move,
        source="classical-coordinator",
        difficulty=int(difficulty),
        safe_selector=safe_selector,
        query_failure_move=observation.move,
    )
    decision = dict(outcome.decision)
    if decision.get("status") != "applied":
        raise ClassicalPositionalSafetyStrengthError(
            f"delivered A_pos gate was not applied: {decision.get('status')}"
        )
    query_count = int(decision.get("query_count", -1))
    if query_count <= 0:
        raise ClassicalPositionalSafetyStrengthError(
            "delivered A_pos query count is invalid"
        )
    # The product adapter is created with a query observer, so every physical
    # Malom lookup (including GameAI's own internal fast path) is counted at
    # the shared read-only choke.  Adding the logical inventory count here
    # would double-count the final gate.
    parent = str(decision["parent_tier"])
    original_after = str(decision["original_tier"])
    transition = (
        f"{parent}->{original_after}"
        if WDL_RANK[original_after] < WDL_RANK[parent]
        else None
    )
    if transition not in {None, "W->D", "W->L", "D->L"}:
        raise ClassicalPositionalSafetyStrengthError(
            "filtered original downgrade transition differs"
        )
    if bool(decision["intervened"]) != (
        _move_tuple(observation.move) != _move_tuple(outcome.move)
    ):
        raise ClassicalPositionalSafetyStrengthError(
            "delivered gate intervention record differs"
        )
    return dict(outcome.move), {
        "safety_mode": "product-A_pos",
        "positional_only": True,
        "parent_tier": parent,
        "original_after_tier": original_after,
        "selected_after_tier": parent,
        "a_pos_cardinality": int(decision["safe_move_count"]),
        "self_downgrade_transition": transition,
        "intervened": bool(decision["intervened"]),
        "gate_decision": decision,
        "search": observation.record(),
        "restricted_root_research": rerank.record(),
    }


def play_dev_classical_game(
    *,
    schedule_item: Mapping[str, Any],
    start_state: Mapping[str, Any],
    product_runtime: ProductDevRuntime,
    product_contract: Mapping[str, Any],
    database: MalomDB,
    product_malom_adapter: Any,
    gate: ProductPositionalSafetyGate | None,
    installation: Any,
    ledger: ResourceLedger,
) -> dict[str, Any]:
    """Play one strict complete game through current ``dev`` product code."""

    started = time.perf_counter()
    candidate_color = str(schedule_item["candidate_color"])
    filtered = bool(schedule_item["filtered"])
    if filtered != (gate is not None):
        raise ClassicalPositionalSafetyStrengthError("gate/arm binding differs")
    ai = product_runtime.new_ai(
        color=candidate_color,
        difficulty=int(schedule_item["difficulty"]),
        node_budget=int(schedule_item["node_budget"]),
        search_threads=int(product_contract["deterministic_search_threads"]),
        max_depth=int(product_contract["max_depth"]),
        malom_adapter=product_malom_adapter,
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
                if filtered:
                    assert gate is not None
                    move, choice = _filtered_choice(
                        board=board,
                        ai=ai,
                        gate=gate,
                        difficulty=int(schedule_item["difficulty"]),
                        ledger=ledger,
                    )
                else:
                    move, choice = _unfiltered_choice(
                        board=board,
                        ai=ai,
                        database=database,
                        ledger=ledger,
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
                    raise ClassicalPositionalSafetyStrengthError(
                        "Sanmill returned no action"
                    )
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
            "intervened": turn["candidate_choice"]["intervened"],
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
        "filtered": filtered,
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


def _semantic_turn(turn: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "actor": turn["actor"],
        "phase": turn["phase"],
        "move": turn["move"],
        "actions": turn["actions"],
        "history_sha256_after": turn["history_sha256_after"],
        "no_capture_count": turn["no_capture_count"],
        "repetition_current_count": turn["repetition_current_count"],
        "repetition_history_length": turn["repetition_history_length"],
        "terminal": turn["terminal"],
        "outcome_reason": turn["outcome_reason"],
    }
    if turn["actor"] == "classical-search":
        choice = turn["candidate_choice"]
        result["classical"] = {
            "parent_tier": choice["parent_tier"],
            "selected_after_tier": choice["selected_after_tier"],
            "self_downgrade_transition": choice["self_downgrade_transition"],
            "nodes": choice["search"]["nodes"],
            "completed_depth": choice["search"]["completed_depth"],
            "bypassed_search": choice["search"]["bypassed_search"],
        }
    return result


def _semantic_game(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "game_id": record["game_id"],
        "start_id": record["start_id"],
        "candidate_color": record["candidate_color"],
        "difficulty": record.get("difficulty"),
        "node_budget": record.get("node_budget"),
        "candidate_score": record["candidate_score"],
        "winner": record["winner"],
        "termination_class": record.get("termination_class"),
        "outcome_reason": record["outcome_reason"],
        "post_start_logical_plies": record["post_start_logical_plies"],
        "final_history_sha256": record["final_state"]["history_sha256"],
        "final_no_capture_count": record["final_state"]["no_capture_count"],
        "final_repetition_current_count": record["final_state"][
            "repetition_current_count"
        ],
        "turns": [_semantic_turn(turn) for turn in record["turns"]],
    }


def _difference_category(
    observed: Mapping[str, Any], reference: Mapping[str, Any]
) -> str:
    observed_turns = observed["turns"]
    reference_turns = reference["turns"]
    if [row.get("move") for row in observed_turns] != [
        row.get("move") for row in reference_turns
    ] or [row.get("actions") for row in observed_turns] != [
        row.get("actions") for row in reference_turns
    ]:
        return "move_sequence"
    terminal_fields = (
        "candidate_score",
        "winner",
        "outcome_reason",
        "post_start_logical_plies",
    )
    if any(observed.get(field) != reference.get(field) for field in terminal_fields):
        return "terminal_outcome"
    clock_fields = (
        "history_sha256_after",
        "no_capture_count",
        "repetition_current_count",
        "repetition_history_length",
    )
    if any(
        left.get(field) != right.get(field)
        for left, right in zip(observed_turns, reference_turns, strict=False)
        for field in clock_fields
    ):
        return "rule_clock_or_history"
    return "search_work_or_label"


def compare_classical_ledgers(
    observed: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare every dev unfiltered game to the v2 semantic record."""

    observed_by_id = {str(row["game_id"]): row for row in observed}
    reference_by_id = {str(row["game_id"]): row for row in reference}
    game_ids = sorted(set(observed_by_id) | set(reference_by_id))
    differing: list[str] = []
    categories: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for game_id in game_ids:
        left = observed_by_id.get(game_id)
        right = reference_by_id.get(game_id)
        if left is None or right is None:
            differing.append(game_id)
            categories["coverage"] += 1
            continue
        if _semantic_game(left) != _semantic_game(right):
            differing.append(game_id)
            category = _difference_category(left, right)
            categories[category] += 1
            if len(examples) < 12:
                examples.append(
                    {
                        "game_id": game_id,
                        "category": category,
                        "observed_semantic_identity": canonical_sha256(
                            _semantic_game(left)
                        ),
                        "reference_semantic_identity": canonical_sha256(
                            _semantic_game(right)
                        ),
                    }
                )
    return {
        "exact_match": not differing,
        "observed_games": len(observed_by_id),
        "reference_games": len(reference_by_id),
        "differing_games": len(differing),
        "differing_game_ids": differing,
        "difference_categories": dict(sorted(categories.items())),
        "examples": examples,
        "observed_semantic_identity": canonical_sha256(
            {game_id: _semantic_game(row) for game_id, row in observed_by_id.items()}
        ),
        "reference_semantic_identity": canonical_sha256(
            {game_id: _semantic_game(row) for game_id, row in reference_by_id.items()}
        ),
    }


def _arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if any(row["termination_class"] != "rules_terminal" for row in rows):
        raise ClassicalPositionalSafetyStrengthError(
            "safety-cap game prevents WDL analysis"
        )
    choices = [
        turn["candidate_choice"]
        for row in rows
        for turn in row["turns"]
        if turn["actor"] == "classical-search"
    ]
    searches = [choice["search"] for choice in choices]
    reranks = [
        choice["restricted_root_research"]
        for choice in choices
        if choice["restricted_root_research"]["called"]
    ]
    events = [event for row in rows for event in row["self_downgrade_events"]]
    interventions = [choice for choice in choices if choice["intervened"]]
    by_phase: dict[str, Any] = {}
    for phase in ("placement", "movement", "flying"):
        phase_rows = [row for row in rows if row["phase"] == phase]
        phase_choices = [
            turn["candidate_choice"]
            for row in rows
            for turn in row["turns"]
            if turn["actor"] == "classical-search" and turn["phase"] == phase
        ]
        by_phase[phase] = {
            "source_games": len(phase_rows),
            "wins": sum(row["candidate_score"] == 1.0 for row in phase_rows),
            "draws": sum(row["candidate_score"] == 0.5 for row in phase_rows),
            "losses": sum(row["candidate_score"] == 0.0 for row in phase_rows),
            "score_rate": (
                statistics.fmean(float(row["candidate_score"]) for row in phase_rows)
                if phase_rows
                else None
            ),
            "candidate_turns": len(phase_choices),
            "interventions": sum(bool(choice["intervened"]) for choice in phase_choices),
            "original_self_downgrades": sum(
                choice["self_downgrade_transition"] is not None
                for choice in phase_choices
            ),
        }
    return {
        "games": len(rows),
        "starts": len({str(row["start_id"]) for row in rows}),
        "strict_wdl": {
            "wins": sum(row["candidate_score"] == 1.0 for row in rows),
            "draws": sum(row["candidate_score"] == 0.5 for row in rows),
            "losses": sum(row["candidate_score"] == 0.0 for row in rows),
            "score_rate": statistics.fmean(float(row["candidate_score"]) for row in rows),
        },
        "terminal_reasons": dict(Counter(row["outcome_reason"] for row in rows)),
        "candidate_turns": len(choices),
        "interventions": {
            "count": len(interventions),
            "rate": len(interventions) / len(choices) if choices else None,
            "selection_rules": dict(
                Counter(
                    choice["gate_decision"]["selection_rule"]
                    for choice in choices
                    if choice["gate_decision"] is not None
                )
            ),
            "selection_failures": sum(
                bool(choice["gate_decision"].get("selection_error"))
                for choice in choices
                if choice["gate_decision"] is not None
            ),
        },
        "original_self_downgrade": {
            "events": len(events),
            "rate": len(events) / len(choices) if choices else None,
            "transitions": dict(Counter(event["transition"] for event in events)),
            "by_phase": dict(Counter(event["phase"] for event in events)),
        },
        "work": {
            "searches": len(searches),
            "bypassed_searches": sum(bool(row["bypassed_search"]) for row in searches),
            "nodes": {
                "minimum": min(int(row["nodes"]) for row in searches),
                "median": statistics.median(int(row["nodes"]) for row in searches),
                "maximum": max(int(row["nodes"]) for row in searches),
            },
            "elapsed_seconds": {
                "median": statistics.median(float(row["elapsed_seconds"]) for row in searches),
                "maximum": max(float(row["elapsed_seconds"]) for row in searches),
            },
            "restricted_root_researches": len(reranks),
            "restricted_root_nodes": sum(int(row["nodes"]) for row in reranks),
            "restricted_root_seconds": sum(
                float(row["elapsed_seconds"]) for row in reranks
            ),
        },
        "source_phase": by_phase,
    }


def analyze_filtered_contrasts(
    records: Sequence[Mapping[str, Any]],
    *,
    start_ids: Sequence[str],
    maximum_half_width: float,
) -> dict[str, Any]:
    """Compute the preregistered same-dev filtered-minus-unfiltered effects."""

    wanted = set(start_ids)
    by_arm: dict[str, Any] = {}
    start_scores: dict[str, dict[str, float]] = {}
    for arm in sorted({str(row["arm"]) for row in records}):
        rows = [row for row in records if str(row["arm"]) == arm]
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row["candidate_score"] is None:
                raise ClassicalPositionalSafetyStrengthError(
                    "incomplete game prevents paired analysis"
                )
            grouped[str(row["start_id"])].append(float(row["candidate_score"]))
        if set(grouped) != wanted or any(len(values) != 2 for values in grouped.values()):
            raise ClassicalPositionalSafetyStrengthError(
                "arm start/color coverage differs"
            )
        start_scores[arm] = {
            start_id: statistics.fmean(values)
            for start_id, values in grouped.items()
        }
        by_arm[arm] = _arm_summary(rows)

    primary: dict[str, Any] = {}
    for difficulty in DIFFICULTIES:
        unfiltered = f"classical-d{difficulty}-unfiltered"
        filtered = f"classical-d{difficulty}-a-pos"
        if unfiltered not in start_scores or filtered not in start_scores:
            raise ClassicalPositionalSafetyStrengthError(
                "required paired arm is absent"
            )
        values = [
            start_scores[filtered][start_id] - start_scores[unfiltered][start_id]
            for start_id in sorted(wanted)
        ]
        interval = paired_interval(values)
        if interval["lower"] > 0:
            decision = "filtered_higher"
        elif interval["upper"] < 0:
            decision = "filtered_lower"
        else:
            decision = "direction_inconclusive"
        primary[f"difficulty_{difficulty}_filtered_minus_unfiltered"] = {
            **interval,
            "decision": decision,
            "precision_adequate": interval["half_width"] <= maximum_half_width,
            "nonzero_start_differences": sum(value != 0.0 for value in values),
            "difference_distribution": dict(Counter(str(value) for value in values)),
        }
    return {
        "by_arm": by_arm,
        "primary": primary,
        "maximum_half_width": maximum_half_width,
    }


def compact_game(record: Mapping[str, Any]) -> dict[str, Any]:
    """Retain enough tracked detail to independently recompute every endpoint."""

    choices = [
        turn["candidate_choice"]
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
            "filtered",
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
        "candidate_turns": len(choices),
        "original_self_downgrades": sum(
            choice["self_downgrade_transition"] is not None for choice in choices
        ),
        "interventions": sum(bool(choice["intervened"]) for choice in choices),
        "intervention_phases": dict(
            Counter(
                turn["phase"]
                for turn in record["turns"]
                if turn["actor"] == "classical-search"
                and turn["candidate_choice"]["intervened"]
            )
        ),
        "selection_failures": sum(
            bool(choice["gate_decision"].get("selection_error"))
            for choice in choices
            if choice["gate_decision"] is not None
        ),
    }


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "ClassicalPositionalSafetyStrengthError",
    "GAME_SCHEMA",
    "PLAN_SCHEMA",
    "ProductDevRuntime",
    "RESULT_SCHEMA",
    "analyze_filtered_contrasts",
    "compact_game",
    "compare_classical_ledgers",
    "play_dev_classical_game",
    "restricted_root_select",
]
