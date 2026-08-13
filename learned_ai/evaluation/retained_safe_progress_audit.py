"""Zero-game safe-progress audit for the completed retained passivity ledger.

The audit never plays a game or loads a policy.  It replays the already
recorded complete moves, enumerates legal alternatives on the local rules
engine, and uses Malom only to decide whether an alternative preserves the
current mover's coarse W/D/L.  A preserving capture is a *safe progress
opportunity* because it resets the strict no-capture clock without lowering
the board-state W/D/L.  This is mechanism evidence, not a strength label.
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


REPORT_SCHEMA = "nmm.retained-safe-progress-audit-result.v1"
EXPECTED_CANDIDATES = (
    "retained-v3-refresh50",
    "retained-v4-no-refresh",
)
EXPECTED_GAMES = 256
EXPECTED_MATCHED_UNITS = 128
ENGINEERING_Z = 1.96
MAX_PRIMARY_HALF_WIDTH = 0.02
MIN_SAFE_CAPTURE_OPPORTUNITIES = 30
HORIZON_LOGICAL_PLY = 120

_WDL_LABEL = {"W": "win", "D": "draw", "L": "loss"}
_WDL_RANK = {"win": 2, "draw": 1, "loss": 0}


class RetainedSafeProgressAuditError(RuntimeError):
    """The immutable source or a replay/query invariant differs."""


def _normalised_move(move: Mapping[str, Any]) -> dict[str, str | None]:
    if any(field not in move for field in ("from", "to", "capture")):
        raise RetainedSafeProgressAuditError("move is not a complete atomic turn")
    return {
        "from": None if move["from"] is None else str(move["from"]),
        "to": str(move["to"]),
        "capture": None if move["capture"] is None else str(move["capture"]),
    }


def _move_key(move: Mapping[str, Any]) -> tuple[str | None, str, str | None]:
    value = _normalised_move(move)
    return value["from"], value["to"], value["capture"]


def _interval(values: Sequence[float]) -> dict[str, Any]:
    support = len(values)
    if support == 0:
        return {
            "support": 0,
            "mean": None,
            "sample_standard_deviation": None,
            "standard_error": None,
            "interval": [None, None],
            "half_width": None,
        }
    mean = statistics.fmean(values)
    if support == 1:
        standard_deviation = 0.0
        standard_error = 0.0
    else:
        standard_deviation = statistics.stdev(values)
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


def _empty_counts() -> Counter[str]:
    return Counter(
        {
            "candidate_turns": 0,
            "queryable_parent_turns": 0,
            "all_legal_actions_queryable_turns": 0,
            "chosen_capture_turns": 0,
            "chosen_preserving_capture_turns": 0,
            "legal_capture_action_turns": 0,
            "safe_capture_opportunity_turns": 0,
            "capture_opportunity_unknown_turns": 0,
            "missed_safe_capture_turns": 0,
            "chosen_board_revisit_turns": 0,
            "safe_novel_opportunity_turns": 0,
            "avoidable_board_revisit_turns": 0,
        }
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summarise_counts(counts: Mapping[str, int]) -> dict[str, Any]:
    turns = int(counts["candidate_turns"])
    opportunities = int(counts["safe_capture_opportunity_turns"])
    missed = int(counts["missed_safe_capture_turns"])
    revisits = int(counts["chosen_board_revisit_turns"])
    avoidable = int(counts["avoidable_board_revisit_turns"])
    return {
        **{key: int(value) for key, value in counts.items()},
        "parent_query_coverage": _rate(
            int(counts["queryable_parent_turns"]), turns
        ),
        "all_action_query_coverage": _rate(
            int(counts["all_legal_actions_queryable_turns"]), turns
        ),
        "chosen_capture_rate": _rate(int(counts["chosen_capture_turns"]), turns),
        "safe_capture_opportunity_rate": _rate(opportunities, turns),
        "safe_capture_selection_rate_given_opportunity": _rate(
            opportunities - missed, opportunities
        ),
        "missed_safe_capture_rate_given_opportunity": _rate(
            missed, opportunities
        ),
        "missed_safe_capture_share_per_candidate_turn": _rate(missed, turns),
        "chosen_board_revisit_rate": _rate(revisits, turns),
        "avoidable_board_revisit_rate_given_revisit": _rate(
            avoidable, revisits
        ),
        "avoidable_board_revisit_share_per_candidate_turn": _rate(
            avoidable, turns
        ),
    }


def classify_candidate_turn(
    *,
    board: BoardState,
    chosen_move: Mapping[str, Any],
    seen_fens: set[str],
    malom: Any,
    recorded_delta: float | None,
) -> dict[str, Any]:
    """Classify one recorded candidate turn against all legal alternatives."""
    chosen = _normalised_move(chosen_move)
    legal = [_normalised_move(move) for move in get_all_legal_moves(board)]
    legal_by_key = {_move_key(move): move for move in legal}
    if len(legal_by_key) != len(legal) or _move_key(chosen) not in legal_by_key:
        raise RetainedSafeProgressAuditError("recorded candidate move is not legal")

    chosen_after = board.apply_move(chosen).to_fen_string()
    chosen_revisit = chosen_after in seen_fens
    parent_wdl = malom.query_state(board)
    if parent_wdl is None:
        if recorded_delta is not None:
            raise RetainedSafeProgressAuditError(
                "recorded Malom delta exists while parent query is unavailable"
            )
        return {
            "parent_queryable": False,
            "all_legal_actions_queryable": False,
            "chosen_capture": chosen["capture"] is not None,
            "chosen_preserving_capture": False,
            "legal_capture_actions": sum(
                move["capture"] is not None for move in legal
            ),
            "safe_capture_opportunity": False,
            "capture_opportunity_unknown": any(
                move["capture"] is not None for move in legal
            ),
            "missed_safe_capture": False,
            "chosen_board_revisit": chosen_revisit,
            "safe_novel_opportunity": False,
            "avoidable_board_revisit": False,
        }
    if parent_wdl not in _WDL_LABEL:
        raise RetainedSafeProgressAuditError("parent Malom W/D/L is invalid")
    parent_label = _WDL_LABEL[parent_wdl]

    rows = malom.query_all_moves(board, board.turn)
    rows_by_key: dict[tuple[str | None, str, str | None], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("move"), Mapping):
            raise RetainedSafeProgressAuditError("Malom move row is malformed")
        key = _move_key(row["move"])
        if key in rows_by_key:
            raise RetainedSafeProgressAuditError("Malom move rows contain a duplicate")
        rows_by_key[key] = row
    if set(rows_by_key) != set(legal_by_key):
        raise RetainedSafeProgressAuditError("Malom move rows differ from legal moves")

    preserving_keys: set[tuple[str | None, str, str | None]] = set()
    unknown_keys: set[tuple[str | None, str, str | None]] = set()
    for key, row in rows_by_key.items():
        label = row.get("wdl")
        if label == "unknown":
            unknown_keys.add(key)
            continue
        if label not in _WDL_RANK:
            raise RetainedSafeProgressAuditError("Malom move W/D/L is invalid")
        if _WDL_RANK[label] > _WDL_RANK[parent_label]:
            raise RetainedSafeProgressAuditError(
                "Malom move improves beyond the exact parent value"
            )
        if label == parent_label:
            preserving_keys.add(key)

    chosen_key = _move_key(chosen)
    chosen_row = rows_by_key[chosen_key]
    chosen_label = chosen_row.get("wdl")
    expected_delta = (
        None
        if chosen_label == "unknown"
        else float(_WDL_RANK[str(chosen_label)] - _WDL_RANK[parent_label])
    )
    if expected_delta != recorded_delta:
        raise RetainedSafeProgressAuditError(
            "recorded chosen-move Malom delta does not replay"
        )

    capture_keys = {
        key for key, move in legal_by_key.items() if move["capture"] is not None
    }
    preserving_capture_keys = capture_keys & preserving_keys
    unknown_capture_keys = capture_keys & unknown_keys
    safe_capture_opportunity = bool(preserving_capture_keys)
    chosen_preserving_capture = (
        chosen_key in preserving_capture_keys
    )

    safe_novel_opportunity = False
    for key in preserving_keys:
        after_fen = board.apply_move(legal_by_key[key]).to_fen_string()
        if after_fen not in seen_fens:
            safe_novel_opportunity = True
            break

    return {
        "parent_queryable": True,
        "all_legal_actions_queryable": not unknown_keys,
        "chosen_capture": chosen["capture"] is not None,
        "chosen_preserving_capture": chosen_preserving_capture,
        "legal_capture_actions": len(capture_keys),
        "safe_capture_opportunity": safe_capture_opportunity,
        "capture_opportunity_unknown": (
            not safe_capture_opportunity and bool(unknown_capture_keys)
        ),
        "missed_safe_capture": (
            safe_capture_opportunity and not chosen_preserving_capture
        ),
        "chosen_board_revisit": chosen_revisit,
        "safe_novel_opportunity": safe_novel_opportunity,
        "avoidable_board_revisit": chosen_revisit and safe_novel_opportunity,
    }


def _increment(counts: Counter[str], turn: Mapping[str, Any]) -> None:
    counts["candidate_turns"] += 1
    counts["queryable_parent_turns"] += int(bool(turn["parent_queryable"]))
    counts["all_legal_actions_queryable_turns"] += int(
        bool(turn["all_legal_actions_queryable"])
    )
    counts["chosen_capture_turns"] += int(bool(turn["chosen_capture"]))
    counts["chosen_preserving_capture_turns"] += int(
        bool(turn["chosen_preserving_capture"])
    )
    counts["legal_capture_action_turns"] += int(
        int(turn["legal_capture_actions"]) > 0
    )
    counts["safe_capture_opportunity_turns"] += int(
        bool(turn["safe_capture_opportunity"])
    )
    counts["capture_opportunity_unknown_turns"] += int(
        bool(turn["capture_opportunity_unknown"])
    )
    counts["missed_safe_capture_turns"] += int(
        bool(turn["missed_safe_capture"])
    )
    counts["chosen_board_revisit_turns"] += int(
        bool(turn["chosen_board_revisit"])
    )
    counts["safe_novel_opportunity_turns"] += int(
        bool(turn["safe_novel_opportunity"])
    )
    counts["avoidable_board_revisit_turns"] += int(
        bool(turn["avoidable_board_revisit"])
    )


def audit_game_record(record: Mapping[str, Any], malom: Any) -> dict[str, Any]:
    """Replay and summarize one already completed diagnostic game."""
    try:
        board = BoardState.from_fen_string(record["prefix"]["final_nmm_fen"])
    except Exception as exc:
        raise RetainedSafeProgressAuditError("prefix board cannot be replayed") from exc
    seen_fens = {board.to_fen_string()}
    all_counts = _empty_counts()
    after_horizon_counts = _empty_counts()

    turns = record.get("turns")
    if not isinstance(turns, list) or not turns:
        raise RetainedSafeProgressAuditError("source game turns are absent")
    for turn in turns:
        if not isinstance(turn, Mapping):
            raise RetainedSafeProgressAuditError("source turn is malformed")
        move = _normalised_move(turn["move"])
        if board.turn != turn.get("mover_color"):
            raise RetainedSafeProgressAuditError("source turn colour does not replay")
        if turn.get("actor") == "candidate":
            classified = classify_candidate_turn(
                board=board,
                chosen_move=move,
                seen_fens=seen_fens,
                malom=malom,
                recorded_delta=turn.get("candidate_malom_delta"),
            )
            _increment(all_counts, classified)
            if int(turn["absolute_logical_ply"]) > HORIZON_LOGICAL_PLY:
                _increment(after_horizon_counts, classified)
        try:
            board = board.apply_move(move)
        except Exception as exc:
            raise RetainedSafeProgressAuditError(
                "source move cannot be applied locally"
            ) from exc
        if board.to_fen_string() != turn.get("local_fen_after"):
            raise RetainedSafeProgressAuditError("source local FEN does not replay")
        seen_fens.add(board.to_fen_string())

    return {
        "match_key": str(record["match_key"]),
        "candidate_id": str(record["candidate_id"]),
        "candidate_color": str(record["candidate_color"]),
        "stratum": str(record["stratum"]),
        "source_ordinal": int(record["ordinal"]),
        "source_game_id": str(record["game_id"]),
        "ongoing_after_total_logical_ply_120": bool(
            record["ongoing_after_total_logical_ply_120"]
        ),
        "outcome_reason": str(record["outcome_reason"]),
        "all_candidate_turns": _summarise_counts(all_counts),
        "after_ply_120_candidate_turns": _summarise_counts(after_horizon_counts),
    }


def _aggregate_games(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    integer_fields = tuple(_empty_counts())
    exact_all = Counter(
        {
            field: sum(int(game["all_candidate_turns"][field]) for game in games)
            for field in integer_fields
        }
    )
    exact_after = Counter(
        {
            field: sum(
                int(game["after_ply_120_candidate_turns"][field]) for game in games
            )
            for field in integer_fields
        }
    )
    return {
        "games": len(games),
        "all_candidate_turns": _summarise_counts(exact_all),
        "after_ply_120_candidate_turns": _summarise_counts(exact_after),
        "outcome_reasons": dict(
            sorted(Counter(str(game["outcome_reason"]) for game in games).items())
        ),
    }


def _game_rate(game: Mapping[str, Any], field: str) -> float:
    summary = game["all_candidate_turns"]
    turns = int(summary["candidate_turns"])
    if not turns:
        raise RetainedSafeProgressAuditError("candidate game has no candidate turn")
    return int(summary[field]) / turns


def _paired_summary(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_match: dict[str, dict[str, Mapping[str, Any]]] = {}
    for game in games:
        match = by_match.setdefault(str(game["match_key"]), {})
        candidate_id = str(game["candidate_id"])
        if candidate_id in match:
            raise RetainedSafeProgressAuditError("duplicate candidate in matched unit")
        match[candidate_id] = game
    complete = [
        match for match in by_match.values() if set(match) == set(EXPECTED_CANDIDATES)
    ]
    if len(complete) != len(by_match):
        raise RetainedSafeProgressAuditError("safe-progress matched unit is incomplete")

    def differences(field: str) -> list[float]:
        return [
            _game_rate(unit[EXPECTED_CANDIDATES[1]], field)
            - _game_rate(unit[EXPECTED_CANDIDATES[0]], field)
            for unit in complete
        ]

    primary = _interval(differences("missed_safe_capture_turns"))
    opportunity_support = {
        candidate: sum(
            int(game["all_candidate_turns"]["safe_capture_opportunity_turns"])
            for game in games
            if game["candidate_id"] == candidate
        )
        for candidate in EXPECTED_CANDIDATES
    }
    is_complete = len(complete) == EXPECTED_MATCHED_UNITS
    if not is_complete:
        decision = "pending"
    elif min(opportunity_support.values(), default=0) < MIN_SAFE_CAPTURE_OPPORTUNITIES:
        decision = "insufficient_safe_capture_opportunities"
    elif primary["half_width"] is None or primary["half_width"] > MAX_PRIMARY_HALF_WIDTH:
        decision = "inconclusive_precision"
    elif primary["interval"][0] > 0:
        decision = "v4_higher_missed_safe_capture_share"
    elif primary["interval"][1] < 0:
        decision = "v3_higher_missed_safe_capture_share"
    else:
        decision = "inconclusive"

    return {
        "matched_units_complete": len(complete),
        "matched_units_expected": EXPECTED_MATCHED_UNITS,
        "safe_capture_opportunity_support": opportunity_support,
        "primary_missed_safe_capture_share_v4_minus_v3": {
            **primary,
            "decision": decision,
            "maximum_half_width": MAX_PRIMARY_HALF_WIDTH,
            "minimum_opportunities_per_candidate": MIN_SAFE_CAPTURE_OPPORTUNITIES,
            "interpretation": (
                "fixed-corpus paired interval for each game's missed-safe-capture "
                "turns divided by all candidate turns; opportunity exposure and "
                "selection are intentionally combined, not population inference"
            ),
        },
        "chosen_capture_share_v4_minus_v3": _interval(
            differences("chosen_capture_turns")
        ),
        "safe_capture_opportunity_share_v4_minus_v3": _interval(
            differences("safe_capture_opportunity_turns")
        ),
        "avoidable_board_revisit_share_v4_minus_v3": _interval(
            differences("avoidable_board_revisit_turns")
        ),
    }


def recompute_safe_progress_audit(
    *,
    source_spec: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    source_ledger_sha256: str,
    source_result_identity: str,
    audit_plan_identity: str,
    implementation_commit: str,
    malom: Any,
) -> dict[str, Any]:
    """Recompute the complete zero-game safe-progress audit."""
    if len(source_records) != EXPECTED_GAMES:
        raise RetainedSafeProgressAuditError("source ledger is not complete")
    games = [audit_game_record(record, malom) for record in source_records]
    candidates = {
        candidate: [game for game in games if game["candidate_id"] == candidate]
        for candidate in EXPECTED_CANDIDATES
    }
    if any(len(rows) != EXPECTED_GAMES // 2 for rows in candidates.values()):
        raise RetainedSafeProgressAuditError("source candidate support differs")

    body = {
        "schema_version": REPORT_SCHEMA,
        "audit_id": "sanmill-retained-v3-v4-safe-progress-audit-v1",
        "audit_plan_identity": audit_plan_identity,
        "implementation_commit": implementation_commit,
        "source": {
            "diagnostic_id": source_spec["diagnostic_id"],
            "spec_identity": source_spec["spec_identity"],
            "ledger_sha256": source_ledger_sha256,
            "result_identity": source_result_identity,
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
                    [game for game in candidates[candidate] if game["candidate_color"] == color]
                )
                for candidate in EXPECTED_CANDIDATES
            }
            for color in ("W", "B")
        },
        "by_source_stratum": {
            stratum: {
                candidate: _aggregate_games(
                    [game for game in candidates[candidate] if game["stratum"] == stratum]
                )
                for candidate in EXPECTED_CANDIDATES
            }
            for stratum in sorted({str(game["stratum"]) for game in games})
        },
        "per_game": games,
        "per_game_identity": canonical_sha256(games),
        "definitions": {
            "safe_capture_opportunity": (
                "at least one complete legal capture preserves exact parent coarse W/D/L"
            ),
            "missed_safe_capture": (
                "a safe capture opportunity exists but the recorded choice is not a "
                "W/D/L-preserving capture"
            ),
            "board_revisit": (
                "chosen local BoardState FEN already appeared at the frozen prefix "
                "endpoint or in the audited post-prefix suffix; not a strict "
                "threefold adjudication"
            ),
            "safe_novel_opportunity": (
                "at least one W/D/L-preserving legal move reaches a local BoardState "
                "FEN not seen at the frozen prefix endpoint or in the audited "
                "post-prefix suffix"
            ),
        },
        "claim_boundary": {
            "zero_game_reanalysis": True,
            "development_corpus_reused": True,
            "malom_history_aware": False,
            "safe_capture_is_strength_label": False,
            "refresh_causal_claim": False,
            "playing_strength_claim": False,
            "automatic_training_setting_selection": False,
            "promotion_or_publication": False,
        },
    }
    return {**body, "result_identity": canonical_sha256(body)}
