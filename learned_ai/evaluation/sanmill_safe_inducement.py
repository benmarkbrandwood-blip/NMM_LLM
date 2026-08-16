"""Positional-only safe-action inducement probe against pinned Sanmill.

The module is deliberately incapable of loading a policy model or playing a
complete game.  It freezes source-game-unique human positions selected without
engine outcomes, constructs ``A_pos`` with the corrected Malom tablebase, and
performs at most one deterministic Sanmill logical-turn search per frozen
state/action/budget cell.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ai.malom_db import MalomDB
from game.board import BoardState
from game.rules import get_game_phase, terminal_wdl
from learned_ai.evaluation.human_f0h0_feasibility import (
    F0D0Boundary,
    ReplayedDecision,
    _oracle_inventory,
    canonical_sha256,
)
from learned_ai.evaluation.human_feature_deviation import PHASE_NAMES
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    EstimatorAccess,
)
from learned_ai.evaluation.sanmill_uci import (
    EXPECTED_RULES_IDENTITY_SHA256,
    SanmillInstallation,
    SanmillUciSession,
    project_stable_sanmill_fen,
)
from learned_ai.training.sanmill_referee import (
    TRAINING_REFEREE_FORMAT,
    TRAINING_REFEREE_PROFILE,
    TRAINING_REFEREE_SEMANTIC_DIGEST,
    TRAINING_REPETITION_OBSERVATION,
    nmm_move_actions,
)


POOL_SCHEMA = "nmm.sanmill-safe-inducement-state-pool.v1"
PLAN_SCHEMA = "nmm.sanmill-safe-inducement-mechanism-plan.v1"
RESULT_SCHEMA = "nmm.sanmill-safe-inducement-preprobe-result.v1"
MAIN_POOL_SCHEMA = "nmm.sanmill-safe-inducement-main-state-pool.v2"
MAIN_PLAN_SCHEMA = "nmm.sanmill-safe-inducement-mechanism-plan.v2"
MAIN_AUTHORIZATION_SCHEMA = "nmm.sanmill-safe-inducement-authorization.v1"
MAIN_PREFLIGHT_SCHEMA = "nmm.sanmill-safe-inducement-preflight.v1"
MAIN_RESULT_SCHEMA = "nmm.sanmill-safe-inducement-main-result.v2"
POOL_SELECTION_SEED = "sanmill-safe-inducement-preprobe-state-v1-20260816"
MAIN_POOL_SELECTION_SEED = "sanmill-safe-inducement-main-state-v2-20260816"
PHASES = ("placement", "movement", "flying")
WDL_RANK = {"L": 0, "D": 1, "W": 2}
WDL_INVERSE = {"W": "L", "D": "D", "L": "W"}


class SafeInducementError(RuntimeError):
    """Raised when an immutable boundary or required observation differs."""


def _load_sealed(
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
        raise SafeInducementError(f"cannot load sealed JSON: {source}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise SafeInducementError(f"sealed JSON schema differs: {source}")
    identity = value.get(identity_field)
    if not isinstance(identity, str) or len(identity) != 64:
        raise SafeInducementError(f"sealed JSON identity absent: {source}")
    body = dict(value)
    body.pop(identity_field)
    if canonical_sha256(body) != identity:
        raise SafeInducementError(f"sealed JSON identity differs: {source}")
    return value, hashlib.sha256(raw).hexdigest()


def load_state_pool(
    path: str | Path,
    *,
    schema: str = POOL_SCHEMA,
) -> tuple[dict[str, Any], str]:
    pool, file_sha = _load_sealed(
        path,
        schema=schema,
        identity_field="pool_identity",
    )
    if pool.get("status") != "frozen_before_any_sanmill_query":
        raise SafeInducementError("state pool is not frozen before engine queries")
    rows = pool.get("states")
    if not isinstance(rows, list) or not rows:
        raise SafeInducementError("state pool is empty")
    if len({row.get("state_id") for row in rows}) != len(rows):
        raise SafeInducementError("state pool IDs are duplicated")
    if len({row.get("session_id") for row in rows}) != len(rows):
        raise SafeInducementError("state pool reuses a source game")
    expected = pool.get("selection_contract", {}).get("states_per_phase")
    counts = Counter(str(row.get("phase")) for row in rows)
    if not isinstance(expected, int) or counts != Counter(
        {phase: expected for phase in PHASES}
    ):
        raise SafeInducementError("state pool phase allocation differs")
    for row in rows:
        actions = row.get("a_pos")
        if not isinstance(actions, list) or not actions:
            raise SafeInducementError("state pool contains an empty A_pos")
        if row.get("a_pos_cardinality") != len(actions):
            raise SafeInducementError("state pool A_pos cardinality differs")
        engine_tiers = {action.get("engine_reply_state_tier") for action in actions}
        if len(engine_tiers) != 1 or next(iter(engine_tiers)) not in WDL_RANK:
            raise SafeInducementError("A_pos does not preserve one response tier")
    return pool, file_sha


def load_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    plan, file_sha = _load_sealed(
        path,
        schema=PLAN_SCHEMA,
        identity_field="plan_identity",
    )
    if plan.get("status") != "frozen_preprobe_authorized_main_unlaunched":
        raise SafeInducementError("mechanism plan status differs")
    boundary = plan.get("claim_boundary")
    if not isinstance(boundary, Mapping) or boundary.get("safe_set") != "A_pos":
        raise SafeInducementError("mechanism plan does not bind A_pos")
    if boundary.get("positional_only") is not True:
        raise SafeInducementError("mechanism plan is not positional-only")
    if boundary.get("human_trap_claim") is not False:
        raise SafeInducementError("mechanism plan permits a human-trap claim")
    budgets = plan.get("preprobe", {}).get("node_budgets")
    if budgets != [1_000, 10_000, 100_000, 500_000]:
        raise SafeInducementError("preprobe node ladder differs")
    resources = plan.get("preprobe", {}).get("resource_envelope", {})
    if (
        resources.get("maximum_engine_single_step_queries") != 100_000
        or resources.get("maximum_active_seconds") != 7_200
        or resources.get("maximum_concurrent_evaluators") != 1
        or resources.get("maximum_concurrent_sanmill_processes") != 1
    ):
        raise SafeInducementError("preprobe resource envelope differs")
    return plan, file_sha


def load_main_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load the sealed v2 main protocol and enforce its unchanged main gate."""
    plan, file_sha = _load_sealed(
        path,
        schema=MAIN_PLAN_SCHEMA,
        identity_field="plan_identity",
    )
    if plan.get("status") != "frozen_protocol_v2_execution_unlaunched":
        raise SafeInducementError("main protocol status differs")
    boundary = plan.get("claim_boundary", {})
    if (
        boundary.get("safe_set") != "A_pos"
        or boundary.get("positional_only") is not True
        or boundary.get("A_allow_claim") is not False
        or boundary.get("human_trap_claim") is not False
    ):
        raise SafeInducementError("main protocol claim boundary differs")
    main = plan.get("main_experiment", {})
    if main.get("node_budgets") != [1_000, 100_000, 500_000]:
        raise SafeInducementError("main protocol budget decomposition differs")
    if main.get("primary_node_budget") != 100_000:
        raise SafeInducementError("main protocol primary budget differs")
    gate = main.get("mechanism_success_gate", {})
    expected_gate = {
        "minimum_point_o_minus_b": 0.05,
        "minimum_lower_95_o_minus_b": 0.05,
        "minimum_evaluable_states": 330,
        "determinism_gate": True,
        "all_conditions_conjunctive": True,
    }
    if any(gate.get(key) != value for key, value in expected_gate.items()):
        raise SafeInducementError("main mechanism gate differs from v1")
    resources = main.get("resource_envelope", {})
    expected_resources = {
        "maximum_states": 360,
        "maximum_engine_single_step_queries": 40_000,
        "maximum_malom_queries": 250_000,
        "maximum_active_seconds": 14_400,
        "maximum_concurrent_evaluators": 1,
        "maximum_concurrent_sanmill_processes": 1,
        "maximum_complete_games": 0,
        "maximum_model_loads": 0,
        "maximum_training_updates": 0,
        "stop_at_any_limit": True,
        "automatic_retry_or_extension": False,
    }
    if any(resources.get(key) != value for key, value in expected_resources.items()):
        raise SafeInducementError("main resource envelope differs")
    return plan, file_sha


def load_main_authorization(path: str | Path) -> tuple[dict[str, Any], str]:
    authorization, file_sha = _load_sealed(
        path,
        schema=MAIN_AUTHORIZATION_SCHEMA,
        identity_field="authorization_identity",
    )
    if authorization.get("operator") != "product-owner-direct":
        raise SafeInducementError("main authorization operator differs")
    if authorization.get("grant_count") != 1:
        raise SafeInducementError("main authorization is not one-time")
    return authorization, file_sha


def load_main_preflight(path: str | Path) -> tuple[dict[str, Any], str]:
    preflight, file_sha = _load_sealed(
        path,
        schema=MAIN_PREFLIGHT_SCHEMA,
        identity_field="preflight_identity",
    )
    if preflight.get("status") != "ready_for_one_authorized_execution":
        raise SafeInducementError("main preflight is not ready")
    if preflight.get("measurement_searches") != 0:
        raise SafeInducementError("main preflight contains measurement searches")
    if preflight.get("determinism", {}).get("passed") is not True:
        raise SafeInducementError("main preflight determinism gate failed")
    return preflight, file_sha


def _move_key(move: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(move.get("from") or ""),
        str(move.get("to") or ""),
        str(move.get("capture") or ""),
    )


def _candidate_rank(
    session_id: str,
    logical_ply: int,
    phase: str,
    *,
    selection_seed: str = POOL_SELECTION_SEED,
) -> str:
    material = (
        f"{selection_seed}\0{session_id}\0{logical_ply}\0{phase}"
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _state_id(
    session_id: str,
    logical_ply: int,
    fen: str,
    history_actions: Sequence[str],
) -> str:
    return canonical_sha256(
        {
            "namespace": "sanmill-safe-inducement-state-v1",
            "session_id": session_id,
            "logical_ply": logical_ply,
            "fen": fen,
            "history_actions": list(history_actions),
        }
    )


def _history_before(
    decisions: Sequence[ReplayedDecision], logical_ply: int
) -> tuple[str, ...]:
    actions: list[str] = []
    for decision in decisions:
        if decision.logical_ply >= logical_ply:
            break
        actions.extend(nmm_move_actions(decision.move))
    return tuple(actions)


def _engine_reply_tier(
    board: BoardState,
    database: MalomDB,
) -> tuple[str, int]:
    rules = terminal_wdl(board)
    if rules is not None:
        return rules, 0
    value = database.query_value(board)
    if value is None or value.outcome not in WDL_RANK:
        raise SafeInducementError("required engine reply-state Malom value is absent")
    return value.outcome, 1


def build_state_pool(
    *,
    repository_root: Path,
    boundary: F0D0Boundary,
    official_membership: Mapping[str, Any],
    research_split: Mapping[str, Any],
    crossfit_structure: Mapping[str, Any],
    database: MalomDB,
    states_per_phase: int = 12,
    schema_version: str = POOL_SCHEMA,
    selection_seed: str = POOL_SELECTION_SEED,
    excluded_coordinates: frozenset[tuple[str, int]] = frozenset(),
) -> dict[str, Any]:
    """Build a source-game-unique pool without engine outcomes or predictions."""
    if states_per_phase <= 0:
        raise SafeInducementError("states_per_phase must be positive")
    construction_started = time.perf_counter()
    sample_rows = crossfit_structure.get("structure", {}).get("sample_games")
    if not isinstance(sample_rows, list) or len(sample_rows) != 6_400:
        raise SafeInducementError("frozen crossfit exploration sample differs")
    sample_ids = [str(row.get("session_id")) for row in sample_rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise SafeInducementError("crossfit exploration sample is duplicated")
    if canonical_sha256(sample_ids) != crossfit_structure.get("structure", {}).get(
        "sample_session_identity"
    ):
        raise SafeInducementError("crossfit sample identity differs")

    access = EstimatorAccess.from_memberships(
        official_membership,
        research_split,
        allowed_sessions=sample_ids,
    )
    records = {record.session_id: record for record in boundary.records}
    candidates: dict[str, list[tuple[str, str, int, str]]] = {
        phase: [] for phase in PHASES
    }
    candidate_identity_rows: dict[str, list[list[Any]]] = {
        phase: [] for phase in PHASES
    }
    replayed_decisions = 0
    for session_id in sorted(sample_ids):
        try:
            record = records[session_id]
        except KeyError as exc:
            raise SafeInducementError("sample game is absent from F0-D0") from exc
        decisions = access.load_decisions(repository_root, record, boundary)
        replayed_decisions += len(decisions)
        for decision in decisions:
            phase = PHASE_NAMES.get(get_game_phase(decision.board, decision.board.turn))
            if phase not in candidates:
                raise SafeInducementError("unexpected source state phase")
            fen = decision.board.to_fen_string()
            if (session_id, decision.logical_ply) in excluded_coordinates:
                continue
            candidates[phase].append(
                (
                    _candidate_rank(
                        session_id,
                        decision.logical_ply,
                        phase,
                        selection_seed=selection_seed,
                    ),
                    session_id,
                    decision.logical_ply,
                    fen,
                )
            )
            candidate_identity_rows[phase].append(
                [session_id, decision.logical_ply, fen]
            )
    if replayed_decisions != int(
        crossfit_structure.get("structure", {}).get("sample_decisions", -1)
    ):
        raise SafeInducementError("crossfit sample decision count differs")

    chosen: list[tuple[str, str, int, str]] = []
    used_games: set[str] = set()
    for phase in PHASES:
        phase_rows = sorted(candidates[phase])
        for _rank, session_id, logical_ply, fen in phase_rows:
            if session_id in used_games:
                continue
            chosen.append((phase, session_id, logical_ply, fen))
            used_games.add(session_id)
            if sum(1 for row in chosen if row[0] == phase) == states_per_phase:
                break
        if sum(1 for row in chosen if row[0] == phase) != states_per_phase:
            raise SafeInducementError(f"insufficient blind candidates for {phase}")

    selected_by_session = {row[1]: row for row in chosen}
    decision_cache: dict[str, Sequence[ReplayedDecision]] = {}
    for session_id in sorted(selected_by_session):
        decision_cache[session_id] = access.load_decisions(
            repository_root,
            records[session_id],
            boundary,
        )

    malom_queries = 0
    malom_started = time.perf_counter()
    states: list[dict[str, Any]] = []
    for phase, session_id, logical_ply, expected_fen in chosen:
        decisions = decision_cache[session_id]
        matching = [row for row in decisions if row.logical_ply == logical_ply]
        if len(matching) != 1 or matching[0].board.to_fen_string() != expected_fen:
            raise SafeInducementError("selected state replay differs")
        decision = matching[0]
        history = _history_before(decisions, logical_ply)
        parent_tier, inventory, query_count = _oracle_inventory(
            decision.board,
            database,
        )
        malom_queries += query_count
        best_rank = max(WDL_RANK[value.outcome] for _move, value in inventory)
        safe = sorted(
            (
                (dict(move), value)
                for move, value in inventory
                if WDL_RANK[value.outcome] == best_rank
            ),
            key=lambda item: _move_key(item[0]),
        )
        if not safe or any(value.outcome != parent_tier for _move, value in safe):
            raise SafeInducementError("A_pos construction differs from parent tier")
        a_pos: list[dict[str, Any]] = []
        response_tiers: set[str] = set()
        for move, _value in safe:
            successor = decision.board.apply_move(move)
            engine_tier, extra_queries = _engine_reply_tier(successor, database)
            malom_queries += extra_queries
            response_tiers.add(engine_tier)
            a_pos.append(
                {
                    "move": move,
                    "actions": list(nmm_move_actions(move)),
                    "successor_fen": successor.to_fen_string(),
                    "board_terminal": terminal_wdl(successor) is not None,
                    "engine_reply_state_tier": engine_tier,
                }
            )
        expected_engine_tier = WDL_INVERSE[parent_tier]
        if response_tiers != {expected_engine_tier}:
            raise SafeInducementError("A_pos successor perspective is inconsistent")
        state_id = _state_id(session_id, logical_ply, expected_fen, history)
        states.append(
            {
                "state_id": state_id,
                "session_id": session_id,
                "logical_ply": logical_ply,
                "phase": phase,
                "side_to_move": decision.board.turn,
                "fen": expected_fen,
                "history_actions": list(history),
                "history_actions_identity": canonical_sha256(list(history)),
                "learner_parent_tier": parent_tier,
                "engine_reply_state_tier": expected_engine_tier,
                "a_pos_cardinality": len(a_pos),
                "a_pos": a_pos,
            }
        )
    states.sort(key=lambda row: (PHASES.index(row["phase"]), row["state_id"]))
    malom_seconds = time.perf_counter() - malom_started
    return {
        "schema_version": schema_version,
        "status": "frozen_before_any_sanmill_query",
        "selection_contract": {
            "source_population": "frozen_6400_game_research_exploration_crossfit_sample",
            "seed": selection_seed,
            "unit": "one_human_decision_state",
            "rank": "SHA-256(seed NUL session_id NUL logical_ply NUL phase)",
            "phase_order": list(PHASES),
            "states_per_phase": states_per_phase,
            "source_game_reuse": False,
            "result_variables_or_estimator_predictions_allowed": False,
            "replacement_after_malom_or_engine_observation": False,
            "excluded_source_coordinates": len(excluded_coordinates),
            "excluded_source_coordinates_identity": canonical_sha256(
                sorted([session_id, ply] for session_id, ply in excluded_coordinates)
            ),
        },
        "source": {
            "crossfit_structure_identity": crossfit_structure["structure_identity"],
            "crossfit_sample_games": len(sample_ids),
            "crossfit_sample_decisions": replayed_decisions,
            "crossfit_sample_session_identity": crossfit_structure["structure"][
                "sample_session_identity"
            ],
            "candidate_counts": {
                phase: len(candidates[phase]) for phase in PHASES
            },
            "candidate_identities": {
                phase: canonical_sha256(sorted(candidate_identity_rows[phase]))
                for phase in PHASES
            },
            "claim": "realistic PlayOK-like source positions only",
            "human_behavior_or_human_trap_claim": False,
        },
        "oracle_construction": {
            "safe_set": "A_pos",
            "positional_only": True,
            "A_allow_claim": False,
            "trusted_label_version": "sector-corrected-v1",
            "queries": malom_queries,
            "elapsed_seconds": malom_seconds,
        },
        "states": states,
        "state_count": len(states),
        "state_membership_identity": canonical_sha256(
            [row["state_id"] for row in states]
        ),
        "resource_use": {
            "construction_active_seconds": time.perf_counter()
            - construction_started,
            "malom_queries": malom_queries,
            "engine_single_step_queries": 0,
        },
        "access_audit": {
            "research_exploration_raw_replays": sum(access.successful.values()),
            "research_confirmation_content_reads": 0,
            "official_selection_content_reads": 0,
            "official_confirmation_content_reads": 0,
            "official_final_test_content_reads": 0,
            "source_pool_2eb04f54_reads_or_consumption": 0,
            "human_estimator_prediction_reads": 0,
            "sanmill_processes_or_search_queries": 0,
            "database_writes": 0,
            "models_games_or_training": 0,
        },
    }


def count_source_phase_frequencies(
    *,
    repository_root: Path,
    boundary: F0D0Boundary,
    official_membership: Mapping[str, Any],
    research_split: Mapping[str, Any],
    crossfit_structure: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently replay the frozen source and count phase decisions only."""
    sample_rows = crossfit_structure.get("structure", {}).get("sample_games")
    if not isinstance(sample_rows, list) or len(sample_rows) != 6_400:
        raise SafeInducementError("frozen crossfit exploration sample differs")
    sample_ids = [str(row.get("session_id")) for row in sample_rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise SafeInducementError("crossfit exploration sample is duplicated")
    access = EstimatorAccess.from_memberships(
        official_membership,
        research_split,
        allowed_sessions=sample_ids,
    )
    records = {record.session_id: record for record in boundary.records}
    counts: Counter[str] = Counter()
    started = time.perf_counter()
    for session_id in sorted(sample_ids):
        try:
            record = records[session_id]
        except KeyError as exc:
            raise SafeInducementError("sample game is absent from F0-D0") from exc
        decisions = access.load_decisions(repository_root, record, boundary)
        for decision in decisions:
            phase = PHASE_NAMES.get(get_game_phase(decision.board, decision.board.turn))
            if phase not in PHASES:
                raise SafeInducementError("unexpected source state phase")
            counts[phase] += 1
    total = sum(counts.values())
    expected = int(crossfit_structure.get("structure", {}).get("sample_decisions", -1))
    if total != expected:
        raise SafeInducementError("crossfit sample decision count differs")
    return {
        "counts": {phase: counts[phase] for phase in PHASES},
        "total": total,
        "weights": {phase: counts[phase] / total for phase in PHASES},
        "count_identity": canonical_sha256(
            [[phase, counts[phase]] for phase in PHASES]
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "access_audit": {
            "research_exploration_raw_replays": sum(access.successful.values()),
            "research_confirmation_content_reads": 0,
            "official_selection_content_reads": 0,
            "official_confirmation_content_reads": 0,
            "official_final_test_content_reads": 0,
            "source_pool_2eb04f54_reads_or_consumption": 0,
            "human_estimator_prediction_reads": 0,
            "malom_queries": 0,
            "sanmill_search_queries": 0,
        },
    }


def _assert_state_matches(board: BoardState, state: Any) -> None:
    projected = project_stable_sanmill_fen(state.fen, terminal=state.terminal)
    if (
        projected.positions != board.positions
        or projected.pieces_placed != board.pieces_placed
        or (not state.terminal and projected.turn != board.turn)
    ):
        raise SafeInducementError("Sanmill state differs from frozen NMM successor")
    if state.rules_identity_sha256 != EXPECTED_RULES_IDENTITY_SHA256:
        raise SafeInducementError("Sanmill rules identity differs")
    identity = state.strict_referee_identity
    expected = {
        "format": TRAINING_REFEREE_FORMAT,
        "profile": TRAINING_REFEREE_PROFILE,
        "repetitionObservation": TRAINING_REPETITION_OBSERVATION,
        "originCounted": True,
        "semanticDigest": TRAINING_REFEREE_SEMANTIC_DIGEST,
    }
    if identity is None or identity.portable_record() != expected:
        raise SafeInducementError("Sanmill strict-referee identity differs")


def _label_response(
    board: BoardState,
    move: Mapping[str, Any],
    database: MalomDB,
) -> tuple[str, str, str | None, int]:
    parent = database.query_value(board)
    if parent is None or parent.outcome not in WDL_RANK:
        raise SafeInducementError("engine response parent Malom value is absent")
    child_board = board.apply_move(move)
    rules = terminal_wdl(child_board)
    queries = 1
    if rules is not None:
        value = database.terminal_move_value(parent, rules)
    else:
        child = database.query_value(child_board)
        queries += 1
        if child is None:
            raise SafeInducementError("engine response child Malom value is absent")
        value = database.move_value(parent, child)
    before = parent.outcome
    after = value.outcome
    if before not in WDL_RANK or after not in WDL_RANK:
        raise SafeInducementError("engine response Malom tier is invalid")
    transition = f"{before}->{after}" if WDL_RANK[after] < WDL_RANK[before] else None
    if transition not in {None, "W->D", "W->L", "D->L"}:
        raise SafeInducementError("unexpected engine positional downgrade")
    return before, after, transition, queries


def _open_root(
    installation: SanmillInstallation,
    *,
    seed: int,
    history_actions: Sequence[str],
    action_tokens: Sequence[str],
    board: BoardState,
    protocol_timeout: float,
    search_timeout: float,
) -> SanmillUciSession:
    session = SanmillUciSession(
        installation,
        seed=seed,
        protocol_timeout=protocol_timeout,
        search_timeout=search_timeout,
    )
    try:
        session.configure_strict_referee_profile(TRAINING_REFEREE_PROFILE)
        session.new_game()
        session.position_startpos([*history_actions, *action_tokens])
        state = session.state_json()
        _assert_state_matches(board, state)
        if state.action_token_count != len(history_actions) + len(action_tokens):
            raise SafeInducementError("Sanmill action history length differs")
        return session
    except BaseException:
        session.close()
        raise


def _search_once(
    installation: SanmillInstallation,
    *,
    seed: int,
    state_row: Mapping[str, Any],
    action_row: Mapping[str, Any],
    node_budget: int,
    protocol_timeout: float,
    search_timeout: float,
) -> dict[str, Any]:
    board = BoardState.from_fen_string(str(action_row["successor_fen"]))
    started = time.perf_counter()
    with _open_root(
        installation,
        seed=seed,
        history_actions=state_row["history_actions"],
        action_tokens=action_row["actions"],
        board=board,
        protocol_timeout=protocol_timeout,
        search_timeout=search_timeout,
    ) as session:
        root = session.state_json()
        if root.terminal:
            return {
                "searched": False,
                "strict_terminal": True,
                "strict_terminal_reason": root.outcome_reason,
                "semantic_search": None,
                "elapsed_seconds": time.perf_counter() - started,
                "search_elapsed_seconds": 0.0,
                "model_action": None,
            }
        result = session.search_logical_turn(node_budget)
        return {
            "searched": True,
            "strict_terminal": False,
            "strict_terminal_reason": None,
            "semantic_search": result.semantic_record(),
            "elapsed_seconds": time.perf_counter() - started,
            "search_elapsed_seconds": result.elapsed_seconds,
            "model_action": dict(result.model_action or {}),
        }


def _search_twice_same_process(
    installation: SanmillInstallation,
    *,
    seed: int,
    state_row: Mapping[str, Any],
    action_row: Mapping[str, Any],
    node_budget: int,
    protocol_timeout: float,
    search_timeout: float,
) -> list[dict[str, Any]]:
    board = BoardState.from_fen_string(str(action_row["successor_fen"]))
    with _open_root(
        installation,
        seed=seed,
        history_actions=state_row["history_actions"],
        action_tokens=action_row["actions"],
        board=board,
        protocol_timeout=protocol_timeout,
        search_timeout=search_timeout,
    ) as session:
        if session.state_json().terminal:
            raise SafeInducementError("determinism fixture became terminal")
        return [
            session.search_logical_turn(node_budget).semantic_record()
            for _ in range(2)
        ]


def run_determinism_gate(
    *,
    installation: SanmillInstallation,
    pool: Mapping[str, Any],
    plan: Mapping[str, Any],
    query_counter: Callable[[], None],
) -> dict[str, Any]:
    states = {row["state_id"]: row for row in pool["states"]}
    fixtures = plan["determinism_gate"]["fixtures"]
    budgets = plan["determinism_gate"].get("budgets")
    if budgets is None:
        budgets = plan["preprobe"]["node_budgets"]
    seed = int(plan["sanmill_contract"]["seed"])
    protocol_timeout = float(plan["sanmill_contract"]["protocol_timeout_seconds"])
    search_timeout = float(plan["sanmill_contract"]["search_timeout_seconds"])
    cells = [
        (fixture, int(budget))
        for fixture in fixtures
        for budget in budgets
    ]
    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for order_name, ordered in (
        ("forward", cells),
        ("reverse", list(reversed(cells))),
    ):
        for fixture, budget in ordered:
            state = states[str(fixture["state_id"])]
            action = state["a_pos"][int(fixture["a_pos_index"])]
            query_counter()
            value = _search_once(
                installation,
                seed=seed,
                state_row=state,
                action_row=action,
                node_budget=budget,
                protocol_timeout=protocol_timeout,
                search_timeout=search_timeout,
            )
            if not value["searched"]:
                raise SafeInducementError("determinism fixture has no engine response")
            key = f"{state['state_id']}:{fixture['a_pos_index']}:{budget}"
            observations[key].append(
                {"order": order_name, "semantic": value["semantic_search"]}
            )
    same_process: list[dict[str, Any]] = []
    for fixture, budget in cells:
        state = states[str(fixture["state_id"])]
        action = state["a_pos"][int(fixture["a_pos_index"])]
        query_counter()
        query_counter()
        pair = _search_twice_same_process(
            installation,
            seed=seed,
            state_row=state,
            action_row=action,
            node_budget=budget,
            protocol_timeout=protocol_timeout,
            search_timeout=search_timeout,
        )
        same_process.append(
            {
                "state_id": state["state_id"],
                "a_pos_index": fixture["a_pos_index"],
                "node_budget": budget,
                "identical": pair[0] == pair[1],
                "semantic_sha256": canonical_sha256(pair[0]),
            }
        )
    cross_process_identical = all(
        len(rows) == 2 and rows[0]["semantic"] == rows[1]["semantic"]
        for rows in observations.values()
    )
    same_process_identical = all(row["identical"] for row in same_process)
    return {
        "passed": cross_process_identical and same_process_identical,
        "semantic_fields": "UciLogicalTurnResult.semantic_record; timing and raw text excluded",
        "cross_process_and_opposite_order_identical": cross_process_identical,
        "same_process_repeat_identical": same_process_identical,
        "cells": len(cells),
        "cross_process_queries": 2 * len(cells),
        "same_process_queries": 2 * len(cells),
        "cross_process_observations": {
            key: [
                {
                    "order": row["order"],
                    "semantic_sha256": canonical_sha256(row["semantic"]),
                }
                for row in rows
            ]
            for key, rows in sorted(observations.items())
        },
        "same_process_observations": same_process,
    }


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_interval(
    values: Sequence[float], *, seed: str, repetitions: int
) -> dict[str, float] | None:
    if not values:
        return None
    if len(set(values)) == 1:
        point = float(values[0])
        return {"lower_95": point, "upper_95": point}
    rng = random.Random(int.from_bytes(hashlib.sha256(seed.encode()).digest(), "big"))
    means = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(repetitions)
    ]
    return {
        "lower_95": float(_percentile(means, 0.025)),
        "upper_95": float(_percentile(means, 0.975)),
    }


def _summarize_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: str,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state[str(row["state_id"])].append(row)
    state_rows: list[dict[str, Any]] = []
    for state_id, cells in sorted(by_state.items()):
        if any(cell.get("abstained") for cell in cells):
            continue
        cardinality = int(cells[0]["a_pos_cardinality"])
        if len(cells) != cardinality:
            raise SafeInducementError("measurement does not exhaust A_pos")
        downgrades = sum(bool(cell["downgrade_transition"]) for cell in cells)
        baseline = downgrades / cardinality
        oracle = float(downgrades > 0)
        state_rows.append(
            {
                "state_id": state_id,
                "b": baseline,
                "o": oracle,
                "o_minus_b": oracle - baseline,
                "downgrades": downgrades,
                "a_pos_cardinality": cardinality,
            }
        )
    metrics: dict[str, Any] = {}
    for field in ("b", "o", "o_minus_b"):
        values = [float(row[field]) for row in state_rows]
        metrics[field] = {
            "mean": statistics.fmean(values) if values else None,
            "state_bootstrap_percentile_95": _bootstrap_interval(
                values,
                seed=f"{bootstrap_seed}:{field}",
                repetitions=bootstrap_repetitions,
            ),
        }
    transitions = Counter(
        str(row["downgrade_transition"])
        for row in rows
        if row.get("downgrade_transition") is not None
    )
    timings = [float(row["search_elapsed_seconds"]) for row in rows if row["searched"]]
    return {
        "states_total": len(by_state),
        "states_evaluable": len(state_rows),
        "states_abstained": len(by_state) - len(state_rows),
        "actions": len(rows),
        "engine_queries": sum(bool(row["searched"]) for row in rows),
        "strict_terminal_no_response_actions": sum(
            bool(row["strict_terminal"]) for row in rows
        ),
        "downgrade_actions": sum(transitions.values()),
        "downgrade_transitions": {
            name: transitions.get(name, 0) for name in ("W->D", "W->L", "D->L")
        },
        "estimates": metrics,
        "search_timing_seconds": {
            "median": _percentile(timings, 0.5),
            "p90": _percentile(timings, 0.9),
            "maximum": max(timings) if timings else None,
            "sum": sum(timings),
        },
        "state_observations": state_rows,
    }


def summarize_measurements(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    bootstrap = plan["preprobe"]["uncertainty"]
    repetitions = int(bootstrap["bootstrap_repetitions"])
    seed = str(bootstrap["bootstrap_seed"])
    result: dict[str, Any] = {}
    for budget in plan["preprobe"]["node_budgets"]:
        selected = [row for row in rows if row["node_budget"] == budget]
        overall = _summarize_group(
            selected,
            bootstrap_seed=f"{seed}:{budget}:overall",
            bootstrap_repetitions=repetitions,
        )
        by_phase = {
            phase: _summarize_group(
                [row for row in selected if row["phase"] == phase],
                bootstrap_seed=f"{seed}:{budget}:phase:{phase}",
                bootstrap_repetitions=repetitions,
            )
            for phase in PHASES
        }
        by_tier = {
            tier: _summarize_group(
                [row for row in selected if row["engine_reply_state_tier"] == tier],
                bootstrap_seed=f"{seed}:{budget}:tier:{tier}",
                bootstrap_repetitions=repetitions,
            )
            for tier in ("W", "D", "L")
        }
        result[str(budget)] = {
            "overall": overall,
            "by_phase": by_phase,
            "by_engine_reply_state_tier": by_tier,
        }
    return result


def summarize_main_measurements(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize the v2 main cells without changing the v1 estimands."""
    interval = plan["main_experiment"]["interval"]
    repetitions = int(interval["repetitions"])
    seed = str(interval["seed"])
    result: dict[str, Any] = {}
    for raw_budget in plan["main_experiment"]["node_budgets"]:
        budget = int(raw_budget)
        selected = [row for row in rows if int(row["node_budget"]) == budget]
        result[str(budget)] = {
            "overall": _summarize_group(
                selected,
                bootstrap_seed=f"{seed}:{budget}:overall",
                bootstrap_repetitions=repetitions,
            ),
            "by_phase": {
                phase: _summarize_group(
                    [row for row in selected if row["phase"] == phase],
                    bootstrap_seed=f"{seed}:{budget}:phase:{phase}",
                    bootstrap_repetitions=repetitions,
                )
                for phase in PHASES
            },
            "by_engine_reply_state_tier": {
                tier: _summarize_group(
                    [
                        row
                        for row in selected
                        if row["engine_reply_state_tier"] == tier
                    ],
                    bootstrap_seed=f"{seed}:{budget}:tier:{tier}",
                    bootstrap_repetitions=repetitions,
                )
                for tier in ("W", "D", "L")
            },
        }
    return result


def decompose_budget_stability(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify safe actions as invariant, budget-sensitive, or never inducing."""
    budgets = tuple(int(value) for value in plan["main_experiment"]["node_budgets"])
    cells: dict[tuple[str, int], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    state_phase: dict[str, str] = {}
    for row in rows:
        state_id = str(row["state_id"])
        key = (state_id, int(row["a_pos_index"]))
        budget = int(row["node_budget"])
        if budget in cells[key]:
            raise SafeInducementError("duplicate main measurement cell")
        cells[key][budget] = row
        state_phase[state_id] = str(row["phase"])
    action_classes: Counter[str] = Counter()
    state_flags: dict[str, dict[str, bool]] = defaultdict(
        lambda: {"invariant": False, "sensitive": False}
    )
    for (state_id, _action_index), by_budget in cells.items():
        if tuple(sorted(by_budget)) != tuple(sorted(budgets)):
            raise SafeInducementError("budget decomposition cell is incomplete")
        if any(row.get("abstained") for row in by_budget.values()):
            raise SafeInducementError("budget decomposition contains abstention")
        outcomes = [
            bool(by_budget[budget].get("downgrade_transition")) for budget in budgets
        ]
        state_flags[state_id]
        if all(outcomes):
            action_class = "budget_invariant"
            state_flags[state_id]["invariant"] = True
        elif any(outcomes):
            action_class = "budget_sensitive"
            state_flags[state_id]["sensitive"] = True
        else:
            action_class = "never_inducing"
        action_classes[action_class] += 1
    state_rows = []
    for state_id, flags in sorted(state_flags.items()):
        invariant = bool(flags["invariant"])
        sensitive_only = not invariant and bool(flags["sensitive"])
        state_rows.append(
            {
                "state_id": state_id,
                "phase": state_phase[state_id],
                "o_inv": float(invariant),
                "o_sens": float(sensitive_only),
                "o_union": float(invariant or sensitive_only),
            }
        )

    def group(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(values)
        inv = sum(float(row["o_inv"]) for row in values)
        sens = sum(float(row["o_sens"]) for row in values)
        union = sum(float(row["o_union"]) for row in values)
        return {
            "states": count,
            "o_inv": inv / count if count else None,
            "o_sens": sens / count if count else None,
            "o_union": union / count if count else None,
            "invariant_share_of_induced_states": inv / union if union else None,
            "identity_check_o_union_equals_o_inv_plus_o_sens": (
                math.isclose(union, inv + sens)
            ),
        }

    overall = group(state_rows)
    threshold = float(
        plan["main_experiment"]["budget_decomposition"][
            "fixed_blind_spot_interpretation_threshold"
        ]
    )
    share = overall["invariant_share_of_induced_states"]
    interpretation = (
        "fixed_engine_evaluation_blind_spot"
        if share is not None and share >= threshold
        else "budget_sensitive_component_not_negligible"
    )
    return {
        "action_classes": {
            name: action_classes[name]
            for name in ("budget_invariant", "budget_sensitive", "never_inducing")
        },
        "overall": overall,
        "by_phase": {
            phase: group([row for row in state_rows if row["phase"] == phase])
            for phase in PHASES
        },
        "state_observations": state_rows,
        "interpretation_threshold": threshold,
        "interpretation": interpretation,
    }


def frequency_weighted_gain(
    summaries: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute the source-frequency weighted secondary 100k-node gain."""
    frequencies = plan["main_experiment"]["frequency_weighted_secondary"]
    weights = frequencies["weights"]
    primary_budget = str(plan["main_experiment"]["primary_node_budget"])
    phase_gains = {
        phase: summaries[primary_budget]["by_phase"][phase]["estimates"][
            "o_minus_b"
        ]["mean"]
        for phase in PHASES
    }
    if any(value is None for value in phase_gains.values()):
        raise SafeInducementError("frequency-weighted gain has an empty phase")
    estimate = sum(float(weights[phase]) * phase_gains[phase] for phase in PHASES)
    return {
        "node_budget": int(primary_budget),
        "phase_counts": frequencies["phase_counts"],
        "weights": weights,
        "phase_o_minus_b": phase_gains,
        "weighted_o_minus_b": estimate,
        "threshold": None,
        "can_flip_primary_decision": False,
        "population_boundary": "observed PlayOK-like source domain only",
    }


def classify_main(
    summaries: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    determinism_passed: bool,
) -> dict[str, Any]:
    """Apply only the frozen conjunctive v1 main gate at 100,000 nodes."""
    if not determinism_passed:
        return {
            "decision": "execution_incomplete",
            "failures": ["determinism_gate_failed"],
        }
    main = plan["main_experiment"]
    primary = summaries[str(main["primary_node_budget"])]["overall"]
    gate = main["mechanism_success_gate"]
    point = primary["estimates"]["o_minus_b"]["mean"]
    interval = primary["estimates"]["o_minus_b"][
        "state_bootstrap_percentile_95"
    ]
    failures: list[str] = []
    if primary["states_evaluable"] < int(gate["minimum_evaluable_states"]):
        failures.append("insufficient_evaluable_states")
    if point is None or point < float(gate["minimum_point_o_minus_b"]):
        failures.append("point_o_minus_b_below_5pp")
    if interval is None or interval["lower_95"] < float(
        gate["minimum_lower_95_o_minus_b"]
    ):
        failures.append("lower_95_o_minus_b_below_5pp")
    return {
        "decision": "mechanism_gate_passed" if not failures else "mechanism_gate_failed",
        "failures": failures,
        "primary_node_budget": int(main["primary_node_budget"]),
        "evaluable_states": primary["states_evaluable"],
        "point_o_minus_b": point,
        "lower_95_o_minus_b": interval["lower_95"] if interval else None,
        "gate": gate,
    }


def classify_preprobe(
    summaries: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    determinism_passed: bool,
) -> dict[str, Any]:
    if not determinism_passed:
        return {
            "conclusion": "C_design_invalid",
            "reason": "determinism_gate_failed",
            "recommended_main_node_budget": None,
        }
    gate = plan["preprobe"]["signal_gate"]
    eligible: list[int] = []
    reasons: dict[str, list[str]] = {}
    for raw_budget in plan["preprobe"]["node_budgets"]:
        budget = int(raw_budget)
        summary = summaries[str(budget)]["overall"]
        b = summary["estimates"]["b"]["mean"]
        gain = summary["estimates"]["o_minus_b"]["mean"]
        failures: list[str] = []
        if summary["states_evaluable"] < int(gate["minimum_evaluable_states"]):
            failures.append("insufficient_evaluable_states")
        if summary["downgrade_actions"] < int(gate["minimum_downgrade_actions"]):
            failures.append("insufficient_downgrade_actions")
        if b is None or not (
            float(gate["minimum_baseline_rate"])
            <= b
            <= float(gate["maximum_baseline_rate"])
        ):
            failures.append("baseline_outside_signal_zone")
        if gain is None or gain < float(gate["minimum_oracle_gain"]):
            failures.append("oracle_gain_below_minimum")
        reasons[str(budget)] = failures
        if not failures:
            eligible.append(budget)
    if not eligible:
        return {
            "conclusion": "B_no_signal_working_region",
            "reason": "no_node_budget_passed_the_frozen_signal_gate",
            "recommended_main_node_budget": None,
            "budget_failures": reasons,
        }
    chosen = max(eligible)
    return {
        "conclusion": "A_signal_region_main_experiment_worth_authorizing",
        "reason": "at_least_one_budget_passed_the_frozen_signal_gate",
        "eligible_node_budgets": eligible,
        "selection_rule": "highest passing budget",
        "recommended_main_node_budget": chosen,
        "budget_failures": reasons,
    }


def run_preprobe(
    *,
    installation: SanmillInstallation,
    database: MalomDB,
    pool: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the frozen determinism gate and exhaustive bounded preprobe."""
    envelope = plan["preprobe"]["resource_envelope"]
    maximum_queries = int(envelope["maximum_engine_single_step_queries"])
    maximum_seconds = float(envelope["maximum_active_seconds"])
    started = time.perf_counter()
    engine_queries = 0

    def count_query() -> None:
        nonlocal engine_queries
        engine_queries += 1
        if engine_queries > maximum_queries:
            raise SafeInducementError("engine query ceiling exceeded")
        if time.perf_counter() - started > maximum_seconds:
            raise SafeInducementError("active-time ceiling exceeded")

    determinism = run_determinism_gate(
        installation=installation,
        pool=pool,
        plan=plan,
        query_counter=count_query,
    )
    if not determinism["passed"]:
        return {
            "determinism": determinism,
            "measurements": [],
            "summaries": {},
            "decision": classify_preprobe(
                {}, plan=plan, determinism_passed=False
            ),
            "resource_use": {
                "engine_single_step_queries": engine_queries,
                "active_seconds": time.perf_counter() - started,
                "malom_queries": 0,
            },
        }

    cells: list[tuple[str, int, int]] = []
    states = {str(row["state_id"]): row for row in pool["states"]}
    for state in pool["states"]:
        for action_index in range(int(state["a_pos_cardinality"])):
            for budget in plan["preprobe"]["node_budgets"]:
                cells.append((str(state["state_id"]), action_index, int(budget)))
    order_seed = str(plan["preprobe"]["measurement_order_seed"])
    cells.sort(
        key=lambda row: (
            hashlib.sha256(f"{order_seed}\0{row[0]}\0{row[1]}\0{row[2]}".encode()).digest(),
            row,
        )
    )
    seed = int(plan["sanmill_contract"]["seed"])
    protocol_timeout = float(plan["sanmill_contract"]["protocol_timeout_seconds"])
    search_timeout = float(plan["sanmill_contract"]["search_timeout_seconds"])
    rows: list[dict[str, Any]] = []
    malom_queries = 0
    malom_seconds = 0.0
    for state_id, action_index, budget in cells:
        if engine_queries >= maximum_queries:
            raise SafeInducementError("engine query ceiling reached before completion")
        if time.perf_counter() - started >= maximum_seconds:
            raise SafeInducementError("active-time ceiling reached before completion")
        state = states[state_id]
        action = state["a_pos"][action_index]
        observed = _search_once(
            installation,
            seed=seed,
            state_row=state,
            action_row=action,
            node_budget=budget,
            protocol_timeout=protocol_timeout,
            search_timeout=search_timeout,
        )
        transition: str | None = None
        chosen_tier: str | None = None
        parent_tier = str(action["engine_reply_state_tier"])
        if observed["searched"]:
            count_query()
            malom_started = time.perf_counter()
            labelled_parent, chosen_tier, transition, queries = _label_response(
                BoardState.from_fen_string(str(action["successor_fen"])),
                observed["model_action"],
                database,
            )
            malom_seconds += time.perf_counter() - malom_started
            malom_queries += queries
            if labelled_parent != parent_tier:
                raise SafeInducementError("frozen and measured response tiers differ")
        rows.append(
            {
                "state_id": state_id,
                "session_id": state["session_id"],
                "phase": state["phase"],
                "engine_reply_state_tier": parent_tier,
                "a_pos_index": action_index,
                "a_pos_cardinality": state["a_pos_cardinality"],
                "node_budget": budget,
                "searched": observed["searched"],
                "strict_terminal": observed["strict_terminal"],
                "strict_terminal_reason": observed["strict_terminal_reason"],
                "engine_chosen_tier": chosen_tier,
                "downgrade_transition": transition,
                "search_elapsed_seconds": observed["search_elapsed_seconds"],
                "cell_elapsed_seconds": observed["elapsed_seconds"],
                "semantic_search": observed["semantic_search"],
                "abstained": False,
            }
        )
    summaries = summarize_measurements(rows, plan=plan)
    decision = classify_preprobe(
        summaries,
        plan=plan,
        determinism_passed=True,
    )
    return {
        "determinism": determinism,
        "measurements": rows,
        "summaries": summaries,
        "decision": decision,
        "resource_use": {
            "engine_single_step_queries": engine_queries,
            "active_seconds": time.perf_counter() - started,
            "malom_queries": malom_queries,
            "malom_elapsed_seconds": malom_seconds,
            "measurement_cells": len(rows),
            "maximum_engine_single_step_queries": maximum_queries,
            "maximum_active_seconds": maximum_seconds,
        },
    }


def run_main_experiment(
    *,
    installation: SanmillInstallation,
    database: MalomDB,
    pool: Mapping[str, Any],
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the frozen v2 measurement once without retry or resumption."""
    main = plan["main_experiment"]
    envelope = main["resource_envelope"]
    maximum_engine = int(envelope["maximum_engine_single_step_queries"])
    maximum_malom = int(envelope["maximum_malom_queries"])
    maximum_seconds = float(envelope["maximum_active_seconds"])
    if int(pool["state_count"]) > int(envelope["maximum_states"]):
        raise SafeInducementError("state ceiling exceeded before execution")
    prior = preflight["aggregate_resource_use_before_measurement"]
    engine_queries = int(prior["engine_single_step_queries"])
    malom_queries = int(prior["malom_queries"])
    prior_seconds = float(prior["active_seconds"])
    if (
        engine_queries >= maximum_engine
        or malom_queries >= maximum_malom
        or prior_seconds >= maximum_seconds
    ):
        raise SafeInducementError("resource ceiling reached before execution")

    started = time.perf_counter()

    def active_seconds() -> float:
        return prior_seconds + time.perf_counter() - started

    states = {str(row["state_id"]): row for row in pool["states"]}
    cells: list[tuple[str, int, int]] = []
    for state in pool["states"]:
        for action_index in range(int(state["a_pos_cardinality"])):
            for budget in main["node_budgets"]:
                cells.append((str(state["state_id"]), action_index, int(budget)))
    order_seed = str(main["measurement_order_seed"])
    cells.sort(
        key=lambda row: (
            hashlib.sha256(
                f"{order_seed}\0{row[0]}\0{row[1]}\0{row[2]}".encode()
            ).digest(),
            row,
        )
    )
    seed = int(plan["sanmill_contract"]["seed"])
    protocol_timeout = float(plan["sanmill_contract"]["protocol_timeout_seconds"])
    search_timeout = float(plan["sanmill_contract"]["search_timeout_seconds"])
    rows: list[dict[str, Any]] = []
    execution_malom_seconds = 0.0
    for state_id, action_index, budget in cells:
        if engine_queries >= maximum_engine:
            raise SafeInducementError("engine query ceiling reached before completion")
        if malom_queries >= maximum_malom:
            raise SafeInducementError("Malom query ceiling reached before completion")
        if active_seconds() >= maximum_seconds:
            raise SafeInducementError("active-time ceiling reached before completion")
        state = states[state_id]
        action = state["a_pos"][action_index]
        observed = _search_once(
            installation,
            seed=seed,
            state_row=state,
            action_row=action,
            node_budget=budget,
            protocol_timeout=protocol_timeout,
            search_timeout=search_timeout,
        )
        transition: str | None = None
        chosen_tier: str | None = None
        parent_tier = str(action["engine_reply_state_tier"])
        if observed["searched"]:
            engine_queries += 1
            if engine_queries > maximum_engine:
                raise SafeInducementError("engine query ceiling exceeded")
            if malom_queries + 2 > maximum_malom:
                raise SafeInducementError("Malom query ceiling would be exceeded")
            malom_started = time.perf_counter()
            labelled_parent, chosen_tier, transition, queries = _label_response(
                BoardState.from_fen_string(str(action["successor_fen"])),
                observed["model_action"],
                database,
            )
            execution_malom_seconds += time.perf_counter() - malom_started
            malom_queries += queries
            if labelled_parent != parent_tier:
                raise SafeInducementError("frozen and measured response tiers differ")
        if active_seconds() > maximum_seconds:
            raise SafeInducementError("active-time ceiling exceeded")
        rows.append(
            {
                "state_id": state_id,
                "session_id": state["session_id"],
                "logical_ply": state["logical_ply"],
                "phase": state["phase"],
                "engine_reply_state_tier": parent_tier,
                "a_pos_index": action_index,
                "a_pos_cardinality": state["a_pos_cardinality"],
                "safe_action": {
                    "move": action["move"],
                    "actions": action["actions"],
                    "successor_fen": action["successor_fen"],
                },
                "node_budget": budget,
                "searched": observed["searched"],
                "strict_terminal": observed["strict_terminal"],
                "strict_terminal_reason": observed["strict_terminal_reason"],
                "engine_model_action": observed["model_action"],
                "engine_chosen_tier": chosen_tier,
                "downgrade_transition": transition,
                "search_elapsed_seconds": observed["search_elapsed_seconds"],
                "cell_elapsed_seconds": observed["elapsed_seconds"],
                "semantic_search": observed["semantic_search"],
                "abstained": False,
            }
        )
    summaries = summarize_main_measurements(rows, plan=plan)
    decomposition = decompose_budget_stability(rows, plan=plan)
    weighted = frequency_weighted_gain(summaries, plan=plan)
    decision = classify_main(
        summaries,
        plan=plan,
        determinism_passed=bool(preflight["determinism"]["passed"]),
    )
    return {
        "measurements": rows,
        "summaries": summaries,
        "budget_decomposition": decomposition,
        "frequency_weighted_secondary": weighted,
        "decision": decision,
        "resource_use": {
            "pool_construction": preflight["resource_components"][
                "pool_construction"
            ],
            "preflight": preflight["resource_components"]["preflight"],
            "measurement": {
                "engine_single_step_queries": engine_queries
                - int(prior["engine_single_step_queries"]),
                "malom_queries": malom_queries - int(prior["malom_queries"]),
                "active_seconds": time.perf_counter() - started,
                "malom_elapsed_seconds": execution_malom_seconds,
                "cells": len(rows),
            },
            "aggregate": {
                "engine_single_step_queries": engine_queries,
                "malom_queries": malom_queries,
                "active_seconds": active_seconds(),
                "states": int(pool["state_count"]),
                "complete_games": 0,
                "model_loads": 0,
                "training_updates": 0,
                "database_writes": 0,
                "maximum_concurrent_evaluators": 1,
                "maximum_concurrent_sanmill_processes": 1,
            },
            "envelope": envelope,
        },
    }


__all__ = [
    "MAIN_AUTHORIZATION_SCHEMA",
    "MAIN_PLAN_SCHEMA",
    "MAIN_POOL_SCHEMA",
    "MAIN_POOL_SELECTION_SEED",
    "MAIN_PREFLIGHT_SCHEMA",
    "MAIN_RESULT_SCHEMA",
    "PLAN_SCHEMA",
    "POOL_SCHEMA",
    "RESULT_SCHEMA",
    "SafeInducementError",
    "build_state_pool",
    "classify_main",
    "classify_preprobe",
    "count_source_phase_frequencies",
    "decompose_budget_stability",
    "frequency_weighted_gain",
    "load_main_authorization",
    "load_main_plan",
    "load_main_preflight",
    "load_plan",
    "load_state_pool",
    "run_preprobe",
    "run_main_experiment",
    "summarize_main_measurements",
    "summarize_measurements",
]
