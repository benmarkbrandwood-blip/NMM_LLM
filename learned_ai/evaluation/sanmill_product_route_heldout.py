"""Held-out comparison of the two delivered high-difficulty product routes.

The evaluator keeps the production decision sources intact: both routes run
the current ``GameAI`` classical coordinator first, the specialist-first route
then applies the current ``SpecialistRouter`` override, and both converge on
the current ``ProductPositionalSafetyGate``.  The gate is position-only
``A_pos``; it is never represented as history-aware ``A_allow``.
"""

from __future__ import annotations

import gc
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ai.human_db import HumanDB
from ai.malom_db import MalomDB
from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.agents.positional_safety import ProductPositionalSafetyGate
from learned_ai.agents.specialist_router import SpecialistRouter
from learned_ai.evaluation.human_f0h0_feasibility import canonical_sha256
from learned_ai.evaluation.retained_late_import_heldout_pool import (
    validate_retained_late_import_pool,
)
from learned_ai.evaluation.retained_phase_process_generalization import (
    replay_frozen_start,
)
from learned_ai.evaluation.sanmill_classical_positional_safety_strength import (
    ProductDevRuntime,
    RerankObservation,
    restricted_root_select,
)
from learned_ai.evaluation.sanmill_classical_search_strength import paired_interval
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    ResourceLedger,
    _checked_position_state,
    _checked_search_result,
    _final_positional_tier,
    _normal_move,
    _phase,
    _strict_terminal_outcome,
    sha256_file,
)
from learned_ai.training.run_contract import canonical_json_bytes
from learned_ai.training.sanmill_referee import SanmillTrainingGame


PLAN_SCHEMA = "nmm.sanmill-product-route-heldout-plan.v1"
AUTHORIZATION_SCHEMA = "nmm.sanmill-product-route-heldout-authorization.v1"
PREFLIGHT_SCHEMA = "nmm.sanmill-product-route-heldout-preflight.v1"
GAME_SCHEMA = "nmm.sanmill-product-route-heldout-game.v1"
RESULT_SCHEMA = "nmm.sanmill-product-route-heldout-result.v1"

POOL_IDENTITY = (
    "2eb04f542f88f8360f08f97e7657ca15646582a1532358dfeb04182ebad7d8f7"
)
POOL_RECORDS_IDENTITY = (
    "4e5f9ecf7508a995b74af6a36bcf966c89d9141940770ebb21c3629446830a31"
)
CONSUMED_PREFIX_IDENTITY = (
    "99951a691c106a86aa5e4affc16ced2b63866e2dd589379527d068b022003c7b"
)
EXPECTED_POOL_RECORDS = 361
CONSUMED_PREFIX_RECORDS = 253
EXPECTED_STARTS = 108
EXPECTED_GAMES = 864
MAX_POST_START_LOGICAL_PLIES = 1536
SANMILL_NODE_BUDGET = 100_000
MAXIMUM_HALF_WIDTH = 0.04
MATERIAL_LOWER_BOUND = 0.05
DIFFICULTY_BUDGETS = {9: 13_887_000, 10: 18_367_000}
ROUTES = ("specialist-first", "classical-first")
ARMS = tuple(
    f"d{difficulty}-{route}-a-pos"
    for difficulty in (9, 10)
    for route in ROUTES
)


class ProductRouteHeldoutError(RuntimeError):
    """Raised when the frozen route-comparison contract differs."""


def _move_tuple(move: Mapping[str, Any]) -> tuple[str, str, str]:
    return tuple(str(move.get(field) or "") for field in ("from", "to", "capture"))


def load_sealed(
    path: str | Path,
    *,
    schema: str,
    identity_field: str,
) -> tuple[dict[str, Any], str]:
    target = Path(path)
    raw = target.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != schema:
        raise ProductRouteHeldoutError(f"sealed schema differs: {target}")
    identity = payload.get(identity_field)
    body = dict(payload)
    body.pop(identity_field, None)
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise ProductRouteHeldoutError(f"sealed identity differs: {target}")
    return payload, sha256_file(target)


def membership_only_suffix(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return only ID-level membership for the unconsumed frozen suffix.

    This function deliberately does not access histories, FENs, phases,
    actions, outcomes, or any candidate observation.  It is the only corpus
    accessor used by the plan freezer before the protocol is immutable.
    """

    if (
        payload.get("pool_identity") != POOL_IDENTITY
        or payload.get("records_identity") != POOL_RECORDS_IDENTITY
    ):
        raise ProductRouteHeldoutError("held-out source-pool identity differs")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_POOL_RECORDS:
        raise ProductRouteHeldoutError("held-out source-pool size differs")
    memberships: list[dict[str, str]] = []
    all_record_ids: list[str] = []
    for row in records:
        if not isinstance(row, Mapping):
            raise ProductRouteHeldoutError("held-out membership row differs")
        start_id = row.get("start_id")
        record_identity = row.get("record_identity")
        if not isinstance(start_id, str) or not isinstance(record_identity, str):
            raise ProductRouteHeldoutError("held-out membership identity is absent")
        all_record_ids.append(record_identity)
        memberships.append(
            {"start_id": start_id, "record_identity": record_identity}
        )
    if (
        canonical_sha256(all_record_ids[:CONSUMED_PREFIX_RECORDS])
        != CONSUMED_PREFIX_IDENTITY
    ):
        raise ProductRouteHeldoutError("consumed 253-record prefix differs")
    suffix = memberships[CONSUMED_PREFIX_RECORDS:]
    if len(suffix) != EXPECTED_STARTS:
        raise ProductRouteHeldoutError("unconsumed suffix size differs")
    if len({row["start_id"] for row in suffix}) != EXPECTED_STARTS or len(
        {row["record_identity"] for row in suffix}
    ) != EXPECTED_STARTS:
        raise ProductRouteHeldoutError("unconsumed suffix membership is not unique")
    if {row["start_id"] for row in memberships[:CONSUMED_PREFIX_RECORDS]} & {
        row["start_id"] for row in suffix
    }:
        raise ProductRouteHeldoutError("consumed and unconsumed memberships overlap")
    return suffix


def validated_suffix_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the complete frozen pool after plan freeze and return the suffix."""

    records = validate_retained_late_import_pool(payload)
    memberships = membership_only_suffix(payload)
    suffix = [dict(row) for row in records[CONSUMED_PREFIX_RECORDS:]]
    observed = [
        {
            "start_id": str(row["start_id"]),
            "record_identity": str(row["record_identity"]),
        }
        for row in suffix
    ]
    if observed != memberships:
        raise ProductRouteHeldoutError("validated suffix order differs")
    return suffix


def build_schedule(
    memberships: Sequence[Mapping[str, str]],
    *,
    namespace: str,
) -> list[dict[str, Any]]:
    """Build the fixed start-major, difficulty, color, route execution order."""

    if len(memberships) != EXPECTED_STARTS:
        raise ProductRouteHeldoutError("formal membership must contain 108 starts")
    schedule: list[dict[str, Any]] = []
    for start_index, membership in enumerate(memberships):
        start_id = str(membership["start_id"])
        record_identity = str(membership["record_identity"])
        for difficulty in (9, 10):
            for color_index, candidate_color in enumerate(("W", "B")):
                unit_index = start_index * 2 + color_index
                for route in ROUTES:
                    arm = f"d{difficulty}-{route}-a-pos"
                    body = {
                        "namespace": namespace,
                        "start_id": start_id,
                        "start_record_identity": record_identity,
                        "difficulty": difficulty,
                        "candidate_color": candidate_color,
                        "route": route,
                    }
                    schedule.append(
                        {
                            "ordinal": len(schedule),
                            "start_index": start_index,
                            "unit_index": unit_index,
                            "start_id": start_id,
                            "start_record_identity": record_identity,
                            "difficulty": difficulty,
                            "node_budget": DIFFICULTY_BUDGETS[difficulty],
                            "candidate_color": candidate_color,
                            "route": route,
                            "arm": arm,
                            "game_id": "product-route-heldout-game:"
                            + canonical_sha256(body),
                        }
                    )
    if len(schedule) != EXPECTED_GAMES:
        raise ProductRouteHeldoutError("formal schedule size differs")
    if [row["ordinal"] for row in schedule] != list(range(EXPECTED_GAMES)):
        raise ProductRouteHeldoutError("formal schedule order differs")
    return schedule


class ProductRouteRuntime:
    """Own the real classical and specialist decision sources used by the web UI."""

    def __init__(
        self,
        *,
        classical: ProductDevRuntime,
        specialist: SpecialistRouter,
        human_db: HumanDB,
        specialist_db: Any | None,
        runtime_identity: str,
    ) -> None:
        self.classical = classical
        self.specialist = specialist
        self.human_db = human_db
        self.specialist_db = specialist_db
        self.runtime_identity = runtime_identity

    def close(self) -> None:
        close = getattr(self.specialist_db, "close", None)
        if callable(close):
            close()
        self.human_db.close()
        self.classical.close()

    def __enter__(self) -> "ProductRouteRuntime":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def choose_product_route_move(
    *,
    board: BoardState,
    ai: Any,
    route_runtime: ProductRouteRuntime,
    gate: ProductPositionalSafetyGate,
    route: str,
    difficulty: int,
    ledger: ResourceLedger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the delivered product order and apply the one final ``A_pos`` choke."""

    if route not in ROUTES or difficulty not in DIFFICULTY_BUDGETS:
        raise ProductRouteHeldoutError("product route binding differs")
    overall_started = time.perf_counter()
    classical = route_runtime.classical.choose(ai, board)
    ledger.add_engine()
    original = dict(classical.move)
    source = "classical-coordinator"
    candidate_scores: list[float] | None = None
    specialist_record = {
        "attempted": route == "specialist-first",
        "loaded": route_runtime.specialist.is_loaded(),
        "succeeded": False,
        "fell_back_to_classical": False,
        "phase_route": None,
        "legal_moves": 0,
        "elapsed_seconds": 0.0,
    }
    if route == "specialist-first":
        route_runtime.specialist.set_gameai(ai)
        legal = [dict(move) for move in get_all_legal_moves(board)]
        model, lookahead, phase_label = route_runtime.specialist._pick_specialist(
            board, board.turn
        )
        if model is None or lookahead is None:
            raise ProductRouteHeldoutError("preferred specialist route is absent")
        specialist_record["phase_route"] = phase_label
        specialist_record["legal_moves"] = len(legal)
        specialist_started = time.perf_counter()
        probabilities = route_runtime.specialist.score_moves(board, legal, board.turn)
        specialist_record["elapsed_seconds"] = (
            time.perf_counter() - specialist_started
        )
        if probabilities is None:
            specialist_record["fell_back_to_classical"] = True
        else:
            scores = [float(value) for value in probabilities]
            if (
                len(scores) != len(legal)
                or not scores
                or any(not math.isfinite(value) or value < 0.0 for value in scores)
                or sum(scores) <= 1e-12
            ):
                raise ProductRouteHeldoutError(
                    "specialist complete score inventory differs"
                )
            best = max(range(len(scores)), key=lambda index: scores[index])
            original = legal[best]
            candidate_scores = scores
            source = "specialist"
            specialist_record["succeeded"] = True

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
        original,
        source=source,
        difficulty=int(difficulty),
        candidate_scores=candidate_scores,
        safe_selector=safe_selector,
        query_failure_move=classical.move,
    )
    decision = dict(outcome.decision)
    if decision.get("status") != "applied":
        raise ProductRouteHeldoutError(
            f"final product A_pos gate was not applied: {decision.get('status')}"
        )
    selected = dict(outcome.move)
    if _move_tuple(selected) not in {
        _move_tuple(move) for move in get_all_legal_moves(board)
    }:
        raise ProductRouteHeldoutError("final product move is illegal")
    return selected, {
        "route": route,
        "product_source": source,
        "classical": classical.record(),
        "specialist": specialist_record,
        "final_gate": decision,
        "restricted_root_research": rerank.record(),
        "route_elapsed_seconds": time.perf_counter() - overall_started,
    }


def play_product_route_game(
    *,
    schedule_item: Mapping[str, Any],
    start_record: Mapping[str, Any],
    route_runtime: ProductRouteRuntime,
    product_contract: Mapping[str, Any],
    database: MalomDB,
    gate: ProductPositionalSafetyGate,
    installation: Any,
    ledger: ResourceLedger,
) -> dict[str, Any]:
    """Play one strict complete game from one frozen held-out history."""

    started = time.perf_counter()
    candidate_color = str(schedule_item["candidate_color"])
    difficulty = int(schedule_item["difficulty"])
    ai = route_runtime.classical.new_ai(
        color=candidate_color,
        difficulty=difficulty,
        node_budget=int(schedule_item["node_budget"]),
        search_threads=int(product_contract["deterministic_search_threads"]),
        max_depth=int(product_contract["max_depth"]),
        malom_adapter=product_contract["product_malom_adapter"],
    )
    turns: list[dict[str, Any]] = []
    safety_cap = False
    try:
        with SanmillTrainingGame(
            installation, seed=int(product_contract["sanmill_seed"])
        ) as game:
            board, strict_start = replay_frozen_start(game, start_record)
            if (
                strict_start["start_record_identity"]
                != schedule_item["start_record_identity"]
            ):
                raise ProductRouteHeldoutError("runtime start identity differs")
            for post_start_ply in range(1, MAX_POST_START_LOGICAL_PLIES + 1):
                ledger.require_within()
                mover = board.turn
                action_phase = _phase(board, mover)
                history_before = game.state.history_sha256
                if mover == candidate_color:
                    move, choice = choose_product_route_move(
                        board=board,
                        ai=ai,
                        route_runtime=route_runtime,
                        gate=gate,
                        route=str(schedule_item["route"]),
                        difficulty=difficulty,
                        ledger=ledger,
                    )
                    applied = game.apply_nmm_move(board, move)
                    actor = "product"
                    engine = None
                else:
                    ledger.add_engine()
                    search_result = game.session.search_logical_turn(
                        SANMILL_NODE_BUDGET
                    )
                    engine = _checked_search_result(
                        search_result, expected_node_budget=SANMILL_NODE_BUDGET
                    )
                    if search_result.model_action is None:
                        raise ProductRouteHeldoutError("Sanmill returned no action")
                    move = search_result.model_action
                    applied = game.apply_nmm_move(
                        board, move, search_result=search_result
                    )
                    actor = "sanmill"
                    choice = None
                board = board.apply_move(applied.move)
                state = dict(_checked_position_state(game.state))
                turns.append(
                    {
                        "post_start_ply": post_start_ply,
                        "absolute_logical_ply": game.state.logical_ply_count,
                        "mover_color": mover,
                        "actor": actor,
                        "phase": action_phase,
                        "move": _normal_move(applied.move),
                        "actions": list(applied.actions),
                        "history_sha256_before": history_before,
                        "history_sha256_after": game.state.history_sha256,
                        "no_capture_count": game.state.no_capture_count,
                        "repetition_current_count": game.state.repetition_current_count,
                        "repetition_history_length": game.state.repetition_history_length,
                        "terminal": game.state.terminal,
                        "outcome_reason": game.state.outcome_reason,
                        "product_choice": choice,
                        "sanmill_search": engine,
                        "state_identity": canonical_sha256(state),
                    }
                )
                if game.state.terminal:
                    break
            else:
                safety_cap = True
            terminal_state = game.state
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
        return {
            "schema_version": GAME_SCHEMA,
            "ordinal": int(schedule_item["ordinal"]),
            "game_id": str(schedule_item["game_id"]),
            "unit_index": int(schedule_item["unit_index"]),
            "start_id": str(schedule_item["start_id"]),
            "start_record_identity": str(schedule_item["start_record_identity"]),
            "start_phase": str(start_record["phase"]),
            "arm": str(schedule_item["arm"]),
            "route": str(schedule_item["route"]),
            "difficulty": difficulty,
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
            "game_elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        del ai
        gc.collect()


def append_game_record(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    previous_record_sha256: str | None,
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = {**record, "previous_record_sha256": previous_record_sha256}
    digest = canonical_sha256(body)
    wrapper = {"record": body, "record_sha256": digest}
    with target.open("xb" if previous_record_sha256 is None else "ab") as handle:
        handle.write(canonical_json_bytes(wrapper) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return digest


def load_game_records(
    path: str | Path,
    *,
    schedule: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target = Path(path)
    raw = target.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ProductRouteHeldoutError("game ledger is not LF-complete")
    records: list[dict[str, Any]] = []
    previous: str | None = None
    for encoded in raw.splitlines():
        wrapper = json.loads(encoded)
        if not isinstance(wrapper, dict) or set(wrapper) != {
            "record",
            "record_sha256",
        }:
            raise ProductRouteHeldoutError("game ledger wrapper differs")
        body = wrapper["record"]
        digest = wrapper["record_sha256"]
        if (
            not isinstance(body, dict)
            or body.get("previous_record_sha256") != previous
            or canonical_sha256(body) != digest
        ):
            raise ProductRouteHeldoutError("game ledger chain differs")
        ordinal = len(records)
        if ordinal >= len(schedule):
            raise ProductRouteHeldoutError("game ledger exceeds frozen schedule")
        expected = schedule[ordinal]
        for field in (
            "ordinal",
            "game_id",
            "start_id",
            "start_record_identity",
            "arm",
            "route",
            "difficulty",
            "node_budget",
            "candidate_color",
        ):
            if body.get(field) != expected.get(field):
                raise ProductRouteHeldoutError(f"game ledger {field} differs")
        if body.get("schema_version") != GAME_SCHEMA:
            raise ProductRouteHeldoutError("game ledger schema differs")
        record = dict(body)
        record.pop("previous_record_sha256")
        records.append(record)
        previous = str(digest)
    return {
        "records": records,
        "record_count": len(records),
        "tail_record_sha256": previous,
        "file_sha256": sha256_file(target),
    }


def _numeric(values: Sequence[int | float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    return {
        "support": len(numbers),
        "minimum": min(numbers) if numbers else None,
        "median": statistics.median(numbers) if numbers else None,
        "mean": statistics.fmean(numbers) if numbers else None,
        "maximum": max(numbers) if numbers else None,
    }


def _arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    product_choices = [
        turn["product_choice"]
        for row in rows
        for turn in row["turns"]
        if turn["actor"] == "product"
    ]
    scores = [float(row["candidate_score"]) for row in rows]
    specialist = [choice["specialist"] for choice in product_choices]
    gates = [choice["final_gate"] for choice in product_choices]
    classical = [choice["classical"] for choice in product_choices]
    route_latencies = [choice["route_elapsed_seconds"] for choice in product_choices]
    phase_summary: dict[str, Any] = {}
    for phase in ("placement", "movement", "flying"):
        phase_choices = [
            turn["product_choice"]
            for row in rows
            for turn in row["turns"]
            if turn["actor"] == "product" and turn["phase"] == phase
        ]
        phase_summary[phase] = {
            "turns": len(phase_choices),
            "specialist_successes": sum(
                bool(choice["specialist"]["succeeded"])
                for choice in phase_choices
            ),
            "specialist_fallbacks": sum(
                bool(choice["specialist"]["fell_back_to_classical"])
                for choice in phase_choices
            ),
            "final_A_pos_interventions": sum(
                bool(choice["final_gate"]["intervened"])
                for choice in phase_choices
            ),
            "internal_malom_or_db_bypasses": sum(
                bool(choice["classical"]["bypassed_search"])
                for choice in phase_choices
            ),
        }
    return {
        "games": len(rows),
        "starts": len({str(row["start_id"]) for row in rows}),
        "strict_wdl": {
            "wins": sum(score == 1.0 for score in scores),
            "draws": sum(score == 0.5 for score in scores),
            "losses": sum(score == 0.0 for score in scores),
            "score_rate": statistics.fmean(scores),
        },
        "terminal_reasons": dict(
            sorted(Counter(str(row["outcome_reason"]) for row in rows).items())
        ),
        "product_turns": len(product_choices),
        "specialist_coverage": {
            "attempts": sum(bool(row["attempted"]) for row in specialist),
            "successes": sum(bool(row["succeeded"]) for row in specialist),
            "fallbacks": sum(
                bool(row["fell_back_to_classical"]) for row in specialist
            ),
        },
        "final_A_pos": {
            "applied": sum(row.get("status") == "applied" for row in gates),
            "interventions": sum(bool(row["intervened"]) for row in gates),
            "selection_failures": sum(
                bool(row.get("selection_error")) for row in gates
            ),
            "selection_rules": dict(
                sorted(Counter(str(row["selection_rule"]) for row in gates).items())
            ),
        },
        "internal_malom_or_db_bypass": {
            "count": sum(bool(row["bypassed_search"]) for row in classical),
            "thinking": dict(
                sorted(Counter(str(row["thinking"]) for row in classical).items())
            ),
        },
        "work": {
            "nodes": _numeric([int(row["nodes"]) for row in classical]),
            "completed_depth": _numeric(
                [int(row["completed_depth"]) for row in classical]
            ),
            "classical_elapsed_seconds": _numeric(
                [float(row["elapsed_seconds"]) for row in classical]
            ),
            "specialist_elapsed_seconds": _numeric(
                [
                    float(row["elapsed_seconds"])
                    for row in specialist
                    if row["attempted"]
                ]
            ),
            "route_elapsed_seconds": _numeric(route_latencies),
            "game_elapsed_seconds": _numeric(
                [float(row["game_elapsed_seconds"]) for row in rows]
            ),
        },
        "by_action_phase": phase_summary,
    }


def analyze_records(
    records: Sequence[Mapping[str, Any]],
    *,
    start_ids: Sequence[str],
) -> dict[str, Any]:
    """Compute the frozen start-clustered route comparisons after all games."""

    if len(records) != EXPECTED_GAMES:
        raise ProductRouteHeldoutError("formal run is incomplete")
    if any(row["termination_class"] != "rules_terminal" for row in records):
        raise ProductRouteHeldoutError("safety-cap game prevents strict WDL analysis")
    if [int(row["ordinal"]) for row in records] != list(range(EXPECTED_GAMES)):
        raise ProductRouteHeldoutError("formal game order differs")
    wanted = set(start_ids)
    if len(wanted) != EXPECTED_STARTS or {
        str(row["start_id"]) for row in records
    } != wanted:
        raise ProductRouteHeldoutError("formal start coverage differs")

    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in records:
        grouped[
            (int(row["difficulty"]), str(row["route"]), str(row["start_id"]))
        ].append(float(row["candidate_score"]))
    if any(len(values) != 2 for values in grouped.values()):
        raise ProductRouteHeldoutError("both-color start unit is incomplete")

    primary: dict[str, Any] = {}
    for difficulty in (9, 10):
        differences = []
        for start_id in start_ids:
            classical = statistics.fmean(
                grouped[(difficulty, "classical-first", start_id)]
            )
            specialist = statistics.fmean(
                grouped[(difficulty, "specialist-first", start_id)]
            )
            differences.append(classical - specialist)
        interval = paired_interval(differences)
        if interval["half_width"] > MAXIMUM_HALF_WIDTH:
            decision = "precision_inadequate_stop"
        elif interval["lower"] >= MATERIAL_LOWER_BOUND:
            decision = "classical_first_material_route_candidate"
        else:
            decision = "no_classical_first_route_change_supported"
        primary[f"difficulty_{difficulty}_classical_minus_specialist"] = {
            **interval,
            "maximum_half_width": MAXIMUM_HALF_WIDTH,
            "minimum_material_lower_bound": MATERIAL_LOWER_BOUND,
            "precision_adequate": interval["half_width"] <= MAXIMUM_HALF_WIDTH,
            "decision": decision,
            "directional_note": (
                "specialist_first_higher" if interval["upper"] < 0.0 else None
            ),
            "difference_distribution": dict(
                sorted(Counter(str(value) for value in differences).items())
            ),
        }

    by_arm = {
        arm: _arm_summary([row for row in records if row["arm"] == arm])
        for arm in ARMS
    }
    start_phases = sorted({str(row["start_phase"]) for row in records})
    return {
        "status": "completed_once_heldout_product_route_comparison",
        "primary": primary,
        "by_arm": by_arm,
        "by_start_phase": {
            phase: {
                arm: _arm_summary(
                    [
                        row
                        for row in records
                        if row["arm"] == arm and row["start_phase"] == phase
                    ]
                )
                for arm in ARMS
            }
            for phase in start_phases
        },
    }


def compact_game(record: Mapping[str, Any]) -> dict[str, Any]:
    choices = [
        turn["product_choice"]
        for turn in record["turns"]
        if turn["actor"] == "product"
    ]
    return {
        key: record[key]
        for key in (
            "ordinal",
            "game_id",
            "start_id",
            "start_record_identity",
            "start_phase",
            "arm",
            "route",
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
        "product_turns": len(choices),
        "specialist_attempts": sum(
            bool(choice["specialist"]["attempted"]) for choice in choices
        ),
        "specialist_successes": sum(
            bool(choice["specialist"]["succeeded"]) for choice in choices
        ),
        "specialist_fallbacks": sum(
            bool(choice["specialist"]["fell_back_to_classical"])
            for choice in choices
        ),
        "final_A_pos_interventions": sum(
            bool(choice["final_gate"]["intervened"]) for choice in choices
        ),
        "final_A_pos_failures": sum(
            choice["final_gate"].get("status") != "applied" for choice in choices
        ),
        "final_A_pos_selection_failures": sum(
            bool(choice["final_gate"].get("selection_error")) for choice in choices
        ),
        "internal_malom_or_db_bypasses": sum(
            bool(choice["classical"]["bypassed_search"]) for choice in choices
        ),
        "candidate_nodes": sum(int(choice["classical"]["nodes"]) for choice in choices),
        "candidate_route_seconds": sum(
            float(choice["route_elapsed_seconds"]) for choice in choices
        ),
    }


__all__ = [
    "ARMS",
    "AUTHORIZATION_SCHEMA",
    "CONSUMED_PREFIX_IDENTITY",
    "CONSUMED_PREFIX_RECORDS",
    "DIFFICULTY_BUDGETS",
    "EXPECTED_GAMES",
    "EXPECTED_STARTS",
    "GAME_SCHEMA",
    "MATERIAL_LOWER_BOUND",
    "MAXIMUM_HALF_WIDTH",
    "PLAN_SCHEMA",
    "POOL_IDENTITY",
    "POOL_RECORDS_IDENTITY",
    "PREFLIGHT_SCHEMA",
    "ProductRouteHeldoutError",
    "ProductRouteRuntime",
    "RESULT_SCHEMA",
    "ROUTES",
    "analyze_records",
    "append_game_record",
    "build_schedule",
    "choose_product_route_move",
    "compact_game",
    "load_game_records",
    "load_sealed",
    "membership_only_suffix",
    "play_product_route_game",
    "validated_suffix_records",
]
