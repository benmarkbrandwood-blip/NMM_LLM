"""Zero-game complete-Malom-order audit for the retained diagnostic ledger.

The audit compares each recorded candidate choice only with alternatives that
preserve the exact parent coarse W/D/L.  Complete ``OracleMoveValue`` objects
are ordered with their validated parent-sector comparator.  The resulting
ordinal regret is positional ultra-strong alignment evidence; it is not a
distance-to-terminal value, a history-aware liveness proof, or strength.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.training.run_contract import canonical_sha256


REPORT_SCHEMA = "nmm.retained-oracle-order-audit-result.v1"
EXPECTED_CANDIDATES = (
    "retained-v3-refresh50",
    "retained-v4-no-refresh",
)
EXPECTED_GAMES = 256
EXPECTED_MATCHED_UNITS = 128
HORIZON_LOGICAL_PLY = 120
ENGINEERING_Z = 1.96
MAX_PRIMARY_HALF_WIDTH = 0.03
MIN_ORDERING_OPPORTUNITIES = 500
MIN_ORDERABLE_COVERAGE = 0.99

_WDL_LABEL = {"W": "win", "D": "draw", "L": "loss"}
_WDL_RANK = {"win": 2, "draw": 1, "loss": 0}


class RetainedOracleOrderAuditError(RuntimeError):
    """The frozen source, replay, or complete comparator invariant differs."""


def _normalised_move(move: Mapping[str, Any]) -> dict[str, str | None]:
    if any(field not in move for field in ("from", "to", "capture")):
        raise RetainedOracleOrderAuditError("move is not a complete atomic turn")
    return {
        "from": None if move["from"] is None else str(move["from"]),
        "to": str(move["to"]),
        "capture": None if move["capture"] is None else str(move["capture"]),
    }


def _move_key(move: Mapping[str, Any]) -> tuple[str | None, str, str | None]:
    value = _normalised_move(move)
    return value["from"], value["to"], value["capture"]


def _rate(numerator: int | float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _interval(values: Sequence[float]) -> dict[str, Any]:
    support = len(values)
    if not support:
        return {
            "support": 0,
            "mean": None,
            "sample_standard_deviation": None,
            "standard_error": None,
            "interval": [None, None],
            "half_width": None,
        }
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if support > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(support)
    half_width = ENGINEERING_Z * standard_error
    return {
        "support": support,
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "interval": [mean - half_width, mean + half_width],
        "half_width": half_width,
    }


def _oracle_context(value: Any) -> tuple[tuple[int, ...], int, str]:
    try:
        context = (
            tuple(int(item) for item in value.sector),
            int(value.sector_value),
            str(value.perspective),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RetainedOracleOrderAuditError(
            "complete Malom value context is malformed"
        ) from exc
    if len(context[0]) != 4 or context[2] not in {"W", "B"}:
        raise RetainedOracleOrderAuditError(
            "complete Malom value context is malformed"
        )
    return context


def _ordering_key(value: Any) -> tuple[int, int]:
    try:
        key = value.ordering_key()
    except (AttributeError, TypeError, ValueError) as exc:
        raise RetainedOracleOrderAuditError(
            "complete Malom ordering key is unavailable"
        ) from exc
    if (
        not isinstance(key, tuple)
        or len(key) != 2
        or any(not isinstance(item, int) for item in key)
    ):
        raise RetainedOracleOrderAuditError("complete Malom ordering key is invalid")
    return key


def classify_candidate_turn(
    *,
    board: BoardState,
    chosen_move: Mapping[str, Any],
    malom: Any,
    recorded_delta: float | None,
) -> dict[str, Any]:
    """Classify a chosen move inside the parent's coarse-WDL-preserving set."""
    chosen = _normalised_move(chosen_move)
    legal = [_normalised_move(move) for move in get_all_legal_moves(board)]
    legal_by_key = {_move_key(move): move for move in legal}
    chosen_key = _move_key(chosen)
    if len(legal_by_key) != len(legal) or chosen_key not in legal_by_key:
        raise RetainedOracleOrderAuditError("recorded candidate move is not legal")

    parent_wdl = malom.query_state(board)
    if parent_wdl is None:
        if recorded_delta is not None:
            raise RetainedOracleOrderAuditError(
                "recorded Malom delta exists while parent query is unavailable"
            )
        return {
            "parent_queryable": False,
            "all_legal_actions_queryable": False,
            "chosen_coarse_preserving": False,
            "preserving_actions": 0,
            "full_order_valued_preserving_actions": 0,
            "within_wdl_orderable": False,
            "full_order_choice_opportunity": False,
            "chosen_full_order_best": False,
            "missed_full_order_best": False,
            "normalised_ordinal_regret": None,
            "better_preserving_actions": None,
            "best_action_ties": None,
            "distinct_preserving_grades": None,
        }
    if parent_wdl not in _WDL_LABEL:
        raise RetainedOracleOrderAuditError("parent Malom W/D/L is invalid")
    parent_label = _WDL_LABEL[parent_wdl]

    rows = malom.query_all_moves(board, board.turn)
    rows_by_key: dict[tuple[str | None, str, str | None], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("move"), Mapping):
            raise RetainedOracleOrderAuditError("Malom move row is malformed")
        key = _move_key(row["move"])
        if key in rows_by_key:
            raise RetainedOracleOrderAuditError("Malom move rows contain a duplicate")
        rows_by_key[key] = row
    if set(rows_by_key) != set(legal_by_key):
        raise RetainedOracleOrderAuditError("Malom move rows differ from legal moves")

    unknown_rows = [row for row in rows_by_key.values() if row.get("wdl") == "unknown"]
    for row in rows_by_key.values():
        label = row.get("wdl")
        if label != "unknown" and label not in _WDL_RANK:
            raise RetainedOracleOrderAuditError("Malom move W/D/L is invalid")
        if label != "unknown" and _WDL_RANK[str(label)] > _WDL_RANK[parent_label]:
            raise RetainedOracleOrderAuditError(
                "Malom move improves beyond the exact parent value"
            )
    chosen_label = rows_by_key[chosen_key].get("wdl")
    expected_delta = (
        None
        if chosen_label == "unknown"
        else float(_WDL_RANK[str(chosen_label)] - _WDL_RANK[parent_label])
    )
    if expected_delta != recorded_delta:
        raise RetainedOracleOrderAuditError(
            "recorded chosen-move Malom delta does not replay"
        )

    preserving = {
        key: row for key, row in rows_by_key.items() if row.get("wdl") == parent_label
    }
    if not preserving and not unknown_rows:
        raise RetainedOracleOrderAuditError(
            "exact parent has no coarse-WDL-preserving action"
        )
    chosen_preserving = chosen_key in preserving
    valued = {
        key: row["oracle_value"]
        for key, row in preserving.items()
        if row.get("oracle_value") is not None
    }
    for key, value in valued.items():
        outcome = getattr(value, "outcome", None)
        if _WDL_LABEL.get(outcome) != preserving[key].get("wdl"):
            raise RetainedOracleOrderAuditError(
                "complete Malom outcome differs from coarse projection"
            )

    orderable = (
        chosen_preserving
        and not unknown_rows
        and len(valued) == len(preserving)
    )
    if not orderable:
        return {
            "parent_queryable": True,
            "all_legal_actions_queryable": not unknown_rows,
            "chosen_coarse_preserving": chosen_preserving,
            "preserving_actions": len(preserving),
            "full_order_valued_preserving_actions": len(valued),
            "within_wdl_orderable": False,
            "full_order_choice_opportunity": False,
            "chosen_full_order_best": False,
            "missed_full_order_best": False,
            "normalised_ordinal_regret": None,
            "better_preserving_actions": None,
            "best_action_ties": None,
            "distinct_preserving_grades": None,
        }

    contexts = {_oracle_context(value) for value in valued.values()}
    if len(contexts) != 1:
        raise RetainedOracleOrderAuditError(
            "complete Malom values use mixed parent contexts"
        )
    if next(iter(contexts))[2] != board.turn:
        raise RetainedOracleOrderAuditError(
            "complete Malom value uses the wrong parent perspective"
        )
    keys = {key: _ordering_key(value) for key, value in valued.items()}
    distinct = sorted(set(keys.values()), reverse=True)
    selected = keys[chosen_key]
    rank = distinct.index(selected)
    opportunity = len(distinct) > 1
    regret = rank / (len(distinct) - 1) if opportunity else 0.0
    best = distinct[0]
    return {
        "parent_queryable": True,
        "all_legal_actions_queryable": not unknown_rows,
        "chosen_coarse_preserving": True,
        "preserving_actions": len(preserving),
        "full_order_valued_preserving_actions": len(valued),
        "within_wdl_orderable": True,
        "full_order_choice_opportunity": opportunity,
        "chosen_full_order_best": selected == best,
        "missed_full_order_best": opportunity and selected != best,
        "normalised_ordinal_regret": regret,
        "better_preserving_actions": sum(value > selected for value in keys.values()),
        "best_action_ties": sum(value == best for value in keys.values()),
        "distinct_preserving_grades": len(distinct),
    }


_INTEGER_FIELDS = (
    "candidate_turns",
    "parent_queryable_turns",
    "all_legal_actions_queryable_turns",
    "chosen_coarse_preserving_turns",
    "within_wdl_orderable_turns",
    "full_order_choice_opportunity_turns",
    "chosen_full_order_best_turns",
    "missed_full_order_best_turns",
    "preserving_actions_total",
    "full_order_valued_preserving_actions_total",
    "better_preserving_actions_total",
)
_FLOAT_FIELDS = (
    "normalised_ordinal_regret_sum",
    "opportunity_normalised_ordinal_regret_sum",
)


def _empty_counts() -> dict[str, int | float]:
    return {
        **{field: 0 for field in _INTEGER_FIELDS},
        **{field: 0.0 for field in _FLOAT_FIELDS},
    }


def _increment(counts: dict[str, int | float], turn: Mapping[str, Any]) -> None:
    counts["candidate_turns"] += 1
    counts["parent_queryable_turns"] += int(bool(turn["parent_queryable"]))
    counts["all_legal_actions_queryable_turns"] += int(
        bool(turn["all_legal_actions_queryable"])
    )
    counts["chosen_coarse_preserving_turns"] += int(
        bool(turn["chosen_coarse_preserving"])
    )
    counts["within_wdl_orderable_turns"] += int(bool(turn["within_wdl_orderable"]))
    counts["full_order_choice_opportunity_turns"] += int(
        bool(turn["full_order_choice_opportunity"])
    )
    counts["chosen_full_order_best_turns"] += int(
        bool(turn["chosen_full_order_best"])
    )
    counts["missed_full_order_best_turns"] += int(
        bool(turn["missed_full_order_best"])
    )
    counts["preserving_actions_total"] += int(turn["preserving_actions"])
    counts["full_order_valued_preserving_actions_total"] += int(
        turn["full_order_valued_preserving_actions"]
    )
    regret = turn["normalised_ordinal_regret"]
    if regret is not None:
        counts["normalised_ordinal_regret_sum"] += float(regret)
        counts["better_preserving_actions_total"] += int(
            turn["better_preserving_actions"]
        )
        if turn["full_order_choice_opportunity"]:
            counts["opportunity_normalised_ordinal_regret_sum"] += float(regret)


def _summarise_counts(counts: Mapping[str, int | float]) -> dict[str, Any]:
    turns = int(counts["candidate_turns"])
    parent = int(counts["parent_queryable_turns"])
    preserving = int(counts["chosen_coarse_preserving_turns"])
    orderable = int(counts["within_wdl_orderable_turns"])
    opportunities = int(counts["full_order_choice_opportunity_turns"])
    best = int(counts["chosen_full_order_best_turns"])
    missed = int(counts["missed_full_order_best_turns"])
    result = {
        **{field: int(counts[field]) for field in _INTEGER_FIELDS},
        **{field: float(counts[field]) for field in _FLOAT_FIELDS},
    }
    result.update(
        {
            "parent_query_coverage": _rate(parent, turns),
            "all_action_query_coverage": _rate(
                int(counts["all_legal_actions_queryable_turns"]), turns
            ),
            "coarse_preserving_rate_given_parent_queryable": _rate(
                preserving, parent
            ),
            "within_wdl_orderable_coverage_per_candidate_turn": _rate(
                orderable, turns
            ),
            "within_wdl_orderable_coverage_given_coarse_preserving": _rate(
                orderable, preserving
            ),
            "full_order_choice_opportunity_rate_given_orderable": _rate(
                opportunities, orderable
            ),
            "chosen_full_order_best_rate_given_orderable": _rate(best, orderable),
            "chosen_full_order_best_rate_given_opportunity": _rate(
                opportunities - missed, opportunities
            ),
            "missed_full_order_best_rate_given_opportunity": _rate(
                missed, opportunities
            ),
            "mean_normalised_ordinal_regret_given_orderable": _rate(
                float(counts["normalised_ordinal_regret_sum"]), orderable
            ),
            "mean_normalised_ordinal_regret_given_opportunity": _rate(
                float(counts["opportunity_normalised_ordinal_regret_sum"]),
                opportunities,
            ),
            "mean_better_preserving_actions_given_orderable": _rate(
                int(counts["better_preserving_actions_total"]), orderable
            ),
        }
    )
    return result


def audit_game_record(record: Mapping[str, Any], malom: Any) -> dict[str, Any]:
    """Replay and summarise complete-order alignment for one source game."""
    try:
        board = BoardState.from_fen_string(record["prefix"]["final_nmm_fen"])
    except Exception as exc:
        raise RetainedOracleOrderAuditError("prefix board cannot be replayed") from exc
    all_counts = _empty_counts()
    after_horizon_counts = _empty_counts()
    turns = record.get("turns")
    if not isinstance(turns, list) or not turns:
        raise RetainedOracleOrderAuditError("source game turns are absent")
    for turn in turns:
        if not isinstance(turn, Mapping):
            raise RetainedOracleOrderAuditError("source turn is malformed")
        move = _normalised_move(turn["move"])
        if board.turn != turn.get("mover_color"):
            raise RetainedOracleOrderAuditError("source turn colour does not replay")
        if turn.get("actor") == "candidate":
            classified = classify_candidate_turn(
                board=board,
                chosen_move=move,
                malom=malom,
                recorded_delta=turn.get("candidate_malom_delta"),
            )
            _increment(all_counts, classified)
            if int(turn["absolute_logical_ply"]) > HORIZON_LOGICAL_PLY:
                _increment(after_horizon_counts, classified)
        try:
            board = board.apply_move(move)
        except Exception as exc:
            raise RetainedOracleOrderAuditError(
                "source move cannot be applied locally"
            ) from exc
        if board.to_fen_string() != turn.get("local_fen_after"):
            raise RetainedOracleOrderAuditError("source local FEN does not replay")
    return {
        "match_key": str(record["match_key"]),
        "candidate_id": str(record["candidate_id"]),
        "candidate_color": str(record["candidate_color"]),
        "stratum": str(record["stratum"]),
        "source_ordinal": int(record["ordinal"]),
        "source_game_id": str(record["game_id"]),
        "outcome_reason": str(record["outcome_reason"]),
        "all_candidate_turns": _summarise_counts(all_counts),
        "after_ply_120_candidate_turns": _summarise_counts(after_horizon_counts),
    }


def _aggregate_games(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = _empty_counts()
    after = _empty_counts()
    for game in games:
        for field in (*_INTEGER_FIELDS, *_FLOAT_FIELDS):
            counts[field] += game["all_candidate_turns"][field]
            after[field] += game["after_ply_120_candidate_turns"][field]
    return {
        "games": len(games),
        "all_candidate_turns": _summarise_counts(counts),
        "after_ply_120_candidate_turns": _summarise_counts(after),
        "outcome_reasons": dict(
            sorted(Counter(str(game["outcome_reason"]) for game in games).items())
        ),
    }


def _game_primary(game: Mapping[str, Any]) -> float | None:
    summary = game["all_candidate_turns"]
    return _rate(
        float(summary["normalised_ordinal_regret_sum"]),
        int(summary["within_wdl_orderable_turns"]),
    )


def _paired_summary(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_match: dict[str, dict[str, Mapping[str, Any]]] = {}
    for game in games:
        match = by_match.setdefault(str(game["match_key"]), {})
        candidate = str(game["candidate_id"])
        if candidate in match:
            raise RetainedOracleOrderAuditError("duplicate candidate in matched unit")
        match[candidate] = game
    if any(set(unit) != set(EXPECTED_CANDIDATES) for unit in by_match.values()):
        raise RetainedOracleOrderAuditError("oracle-order matched unit is incomplete")

    differences: list[float] = []
    for unit in by_match.values():
        v3 = _game_primary(unit[EXPECTED_CANDIDATES[0]])
        v4 = _game_primary(unit[EXPECTED_CANDIDATES[1]])
        if v3 is not None and v4 is not None:
            differences.append(v4 - v3)
    primary = _interval(differences)
    aggregate = {
        candidate: _aggregate_games(
            [game for game in games if game["candidate_id"] == candidate]
        )["all_candidate_turns"]
        for candidate in EXPECTED_CANDIDATES
    }
    coverage = {
        candidate: aggregate[candidate][
            "within_wdl_orderable_coverage_per_candidate_turn"
        ]
        for candidate in EXPECTED_CANDIDATES
    }
    opportunities = {
        candidate: aggregate[candidate]["full_order_choice_opportunity_turns"]
        for candidate in EXPECTED_CANDIDATES
    }
    complete = len(by_match) == EXPECTED_MATCHED_UNITS
    if not complete:
        decision = "pending"
    elif len(differences) != EXPECTED_MATCHED_UNITS or any(
        value is None or value < MIN_ORDERABLE_COVERAGE for value in coverage.values()
    ):
        decision = "insufficient_full_order_coverage"
    elif min(opportunities.values(), default=0) < MIN_ORDERING_OPPORTUNITIES:
        decision = "insufficient_ordering_opportunities"
    elif primary["half_width"] is None or primary["half_width"] > MAX_PRIMARY_HALF_WIDTH:
        decision = "inconclusive_precision"
    elif primary["interval"][0] > 0:
        decision = "v4_higher_full_order_regret"
    elif primary["interval"][1] < 0:
        decision = "v3_higher_full_order_regret"
    else:
        decision = "inconclusive"
    return {
        "matched_units_complete": len(by_match),
        "matched_units_expected": EXPECTED_MATCHED_UNITS,
        "matched_units_with_primary_support": len(differences),
        "within_wdl_orderable_coverage": coverage,
        "full_order_choice_opportunity_support": opportunities,
        "primary_mean_normalised_ordinal_regret_v4_minus_v3": {
            **primary,
            "decision": decision,
            "maximum_half_width": MAX_PRIMARY_HALF_WIDTH,
            "minimum_orderable_coverage": MIN_ORDERABLE_COVERAGE,
            "minimum_ordering_opportunities_per_candidate": (
                MIN_ORDERING_OPPORTUNITIES
            ),
            "interpretation": (
                "fixed-corpus paired interval; forced or oracle-tied preserving "
                "turns contribute zero, and full-orderable chosen-preserving "
                "turns are the per-game denominator"
            ),
        },
    }


def recompute_oracle_order_audit(
    *,
    source_spec: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    source_ledger_sha256: str,
    source_result_identity: str,
    safe_progress_result_identity: str,
    safe_progress_file_sha256: str,
    audit_plan_identity: str,
    implementation_commit: str,
    malom: Any,
) -> dict[str, Any]:
    """Recompute the complete zero-game full-order alignment audit."""
    if len(source_records) != EXPECTED_GAMES:
        raise RetainedOracleOrderAuditError("source ledger is not complete")
    games = [audit_game_record(record, malom) for record in source_records]
    candidates = {
        candidate: [game for game in games if game["candidate_id"] == candidate]
        for candidate in EXPECTED_CANDIDATES
    }
    if any(len(rows) != EXPECTED_GAMES // 2 for rows in candidates.values()):
        raise RetainedOracleOrderAuditError("source candidate support differs")
    body = {
        "schema_version": REPORT_SCHEMA,
        "audit_id": "sanmill-retained-v3-v4-oracle-order-audit-v1",
        "audit_plan_identity": audit_plan_identity,
        "implementation_commit": implementation_commit,
        "source": {
            "diagnostic_id": source_spec["diagnostic_id"],
            "spec_identity": source_spec["spec_identity"],
            "ledger_sha256": source_ledger_sha256,
            "result_identity": source_result_identity,
            "safe_progress_result_identity": safe_progress_result_identity,
            "safe_progress_file_sha256": safe_progress_file_sha256,
            "games": len(source_records),
            "new_games": 0,
        },
        "by_candidate": {
            candidate: _aggregate_games(candidates[candidate])
            for candidate in EXPECTED_CANDIDATES
        },
        "paired": _paired_summary(games),
        "by_candidate_color": {
            color: {
                candidate: _aggregate_games(
                    [
                        game
                        for game in candidates[candidate]
                        if game["candidate_color"] == color
                    ]
                )
                for candidate in EXPECTED_CANDIDATES
            }
            for color in ("W", "B")
        },
        "by_source_stratum": {
            stratum: {
                candidate: _aggregate_games(
                    [
                        game
                        for game in candidates[candidate]
                        if game["stratum"] == stratum
                    ]
                )
                for candidate in EXPECTED_CANDIDATES
            }
            for stratum in sorted({str(game["stratum"]) for game in games})
        },
        "per_game": games,
        "per_game_identity": canonical_sha256(games),
        "definitions": {
            "within_wdl_orderable": (
                "the recorded choice preserves parent coarse W/D/L and every "
                "coarse-preserving legal action has a complete comparable "
                "OracleMoveValue"
            ),
            "normalised_ordinal_regret": (
                "zero for the best complete Malom ordering grade; otherwise the "
                "zero-based chosen distinct-grade rank divided by the number of "
                "worse-than-best rank steps, in [0,1]"
            ),
            "full_order_choice_opportunity": (
                "the coarse-WDL-preserving legal set contains more than one "
                "distinct complete Malom ordering grade"
            ),
        },
        "claim_boundary": {
            "zero_game_reanalysis": True,
            "development_corpus_reused": True,
            "complete_malom_order_is_positional": True,
            "history_aware_liveness": False,
            "distance_to_terminal_claim": False,
            "passivity_causal_claim": False,
            "refresh_causal_claim": False,
            "playing_strength_claim": False,
            "automatic_training_setting_selection": False,
            "promotion_or_publication": False,
        },
    }
    return {**body, "result_identity": canonical_sha256(body)}
