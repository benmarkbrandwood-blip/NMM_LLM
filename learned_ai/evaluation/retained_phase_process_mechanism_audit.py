"""Zero-game mechanism audit for a completed phase-process ledger."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from game.board import BoardState
from learned_ai.evaluation.retained_oracle_order_audit import (
    _empty_counts as _empty_order_counts,
    _increment as _increment_order,
    _summarise_counts as _summarise_order_counts,
    classify_candidate_turn as classify_order_turn,
)
from learned_ai.evaluation.retained_phase_process_generalization import (
    EXPECTED_CANDIDATES,
    EXPECTED_GAMES,
    EXPECTED_STARTS,
    HORIZON_POST_START_LOGICAL_PLIES,
    _interval,
)
from learned_ai.evaluation.retained_safe_progress_audit import (
    _empty_counts as _empty_safe_counts,
    _increment as _increment_safe,
    _summarise_counts as _summarise_safe_counts,
    classify_candidate_turn as classify_safe_turn,
)
from learned_ai.training.run_contract import canonical_sha256


REPORT_SCHEMA = "nmm.retained-phase-process-mechanism-audit-result.v1"


class RetainedPhaseProcessMechanismError(RuntimeError):
    """Raised when ledger replay or mechanism evidence differs."""


def _normalised_move(move: Mapping[str, Any]) -> dict[str, str | None]:
    if any(field not in move for field in ("from", "to", "capture")):
        raise RetainedPhaseProcessMechanismError("recorded move is incomplete")
    return {
        "from": None if move["from"] is None else str(move["from"]),
        "to": str(move["to"]),
        "capture": None if move["capture"] is None else str(move["capture"]),
    }


def audit_game_record(
    record: Mapping[str, Any],
    malom: Any,
    *,
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Replay one completed game without selecting or applying a new move."""
    try:
        board = BoardState.from_fen_string(record["start"]["final_nmm_fen"])
    except Exception as exc:
        raise RetainedPhaseProcessMechanismError(
            "frozen start board cannot be replayed"
        ) from exc
    seen_fens = {board.to_fen_string()}
    safe_all = _empty_safe_counts()
    safe_after = _empty_safe_counts()
    order_all = _empty_order_counts()
    order_after = _empty_order_counts()
    turns = record.get("turns")
    if not isinstance(turns, list) or not turns:
        raise RetainedPhaseProcessMechanismError("source game turns are absent")

    for index, turn in enumerate(turns, 1):
        if not isinstance(turn, Mapping):
            raise RetainedPhaseProcessMechanismError("source turn is malformed")
        move = _normalised_move(turn["move"])
        if board.turn != turn.get("mover_color"):
            raise RetainedPhaseProcessMechanismError(
                "source turn colour does not replay"
            )
        if turn.get("actor") == "candidate":
            safe = classify_safe_turn(
                board=board,
                chosen_move=move,
                seen_fens=seen_fens,
                malom=malom,
                recorded_delta=turn.get("candidate_malom_delta"),
            )
            order = classify_order_turn(
                board=board,
                chosen_move=move,
                malom=malom,
                recorded_delta=turn.get("candidate_malom_delta"),
            )
            _increment_safe(safe_all, safe)
            _increment_order(order_all, order)
            if int(turn["post_start_logical_ply"]) > (
                HORIZON_POST_START_LOGICAL_PLIES
            ):
                _increment_safe(safe_after, safe)
                _increment_order(order_after, order)
        try:
            board = board.apply_move(move)
        except Exception as exc:
            raise RetainedPhaseProcessMechanismError(
                "source move cannot be applied locally"
            ) from exc
        if board.to_fen_string() != turn.get("local_fen_after"):
            raise RetainedPhaseProcessMechanismError(
                "source local FEN does not replay"
            )
        seen_fens.add(board.to_fen_string())
        if progress is not None:
            progress(index)

    return {
        "match_key": str(record["match_key"]),
        "start_id": str(record["start_id"]),
        "candidate_id": str(record["candidate_id"]),
        "candidate_color": str(record["candidate_color"]),
        "phase": str(record["phase"]),
        "source_ordinal": int(record["ordinal"]),
        "source_game_id": str(record["game_id"]),
        "outcome_reason": str(record["outcome_reason"]),
        "safe_progress": {
            "all_candidate_turns": _summarise_safe_counts(safe_all),
            "after_relative_horizon_candidate_turns": (
                _summarise_safe_counts(safe_after)
            ),
        },
        "complete_order": {
            "all_candidate_turns": _summarise_order_counts(order_all),
            "after_relative_horizon_candidate_turns": (
                _summarise_order_counts(order_after)
            ),
        },
    }


def _aggregate_safe(games: Sequence[Mapping[str, Any]], period: str) -> dict[str, Any]:
    counts = _empty_safe_counts()
    for game in games:
        summary = game["safe_progress"][period]
        for field in counts:
            counts[field] += int(summary[field])
    return _summarise_safe_counts(counts)


def _aggregate_order(
    games: Sequence[Mapping[str, Any]],
    period: str,
) -> dict[str, Any]:
    counts = _empty_order_counts()
    for game in games:
        summary = game["complete_order"][period]
        for field in counts:
            counts[field] += summary[field]
    return _summarise_order_counts(counts)


def _aggregate_games(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "games": len(games),
        "safe_progress": {
            "all_candidate_turns": _aggregate_safe(
                games,
                "all_candidate_turns",
            ),
            "after_relative_horizon_candidate_turns": _aggregate_safe(
                games,
                "after_relative_horizon_candidate_turns",
            ),
        },
        "complete_order": {
            "all_candidate_turns": _aggregate_order(
                games,
                "all_candidate_turns",
            ),
            "after_relative_horizon_candidate_turns": _aggregate_order(
                games,
                "after_relative_horizon_candidate_turns",
            ),
        },
        "outcome_reasons": dict(
            sorted(Counter(str(game["outcome_reason"]) for game in games).items())
        ),
    }


def _safe_rate(game: Mapping[str, Any], field: str) -> float | None:
    summary = game["safe_progress"]["all_candidate_turns"]
    turns = int(summary["candidate_turns"])
    return int(summary[field]) / turns if turns else None


def _order_regret(game: Mapping[str, Any]) -> float | None:
    summary = game["complete_order"]["all_candidate_turns"]
    orderable = int(summary["within_wdl_orderable_turns"])
    if not orderable:
        return None
    return float(summary["normalised_ordinal_regret_sum"]) / orderable


def _start_clustered_difference(
    games: Sequence[Mapping[str, Any]],
    metric: Callable[[Mapping[str, Any]], float | None],
) -> dict[str, Any]:
    by_match: dict[str, dict[str, Mapping[str, Any]]] = {}
    for game in games:
        match = by_match.setdefault(str(game["match_key"]), {})
        candidate = str(game["candidate_id"])
        if candidate in match:
            raise RetainedPhaseProcessMechanismError(
                "duplicate candidate in mechanism matched unit"
            )
        match[candidate] = game

    colour_values: dict[tuple[str, str], float] = {}
    for unit in by_match.values():
        if set(unit) != set(EXPECTED_CANDIDATES):
            continue
        v3 = metric(unit[EXPECTED_CANDIDATES[0]])
        v4 = metric(unit[EXPECTED_CANDIDATES[1]])
        if v3 is None or v4 is None:
            continue
        key = (
            str(unit[EXPECTED_CANDIDATES[0]]["start_id"]),
            str(unit[EXPECTED_CANDIDATES[0]]["candidate_color"]),
        )
        colour_values[key] = v4 - v3

    starts = sorted({key[0] for key in colour_values})
    values = [
        (colour_values[(start, "W")] + colour_values[(start, "B")]) / 2.0
        for start in starts
        if (start, "W") in colour_values and (start, "B") in colour_values
    ]
    interval = _interval(values)
    distribution = Counter(values)
    return {
        **interval,
        "start_units_expected": EXPECTED_STARTS,
        "matched_colour_units_with_metric": len(colour_values),
        "distribution": {
            str(value): distribution[value] for value in sorted(distribution)
        },
        "decision": "exploratory_no_directional_gate",
        "interpretation": (
            "start-clustered fixed-corpus engineering interval; exploratory "
            "mechanism evidence only, not strength or causality"
        ),
    }


def _paired_summary(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "start_clustered_missed_safe_capture_share_v4_minus_v3": (
            _start_clustered_difference(
                games,
                lambda game: _safe_rate(game, "missed_safe_capture_turns"),
            )
        ),
        "start_clustered_safe_capture_opportunity_share_v4_minus_v3": (
            _start_clustered_difference(
                games,
                lambda game: _safe_rate(
                    game,
                    "safe_capture_opportunity_turns",
                ),
            )
        ),
        "start_clustered_board_revisit_share_v4_minus_v3": (
            _start_clustered_difference(
                games,
                lambda game: _safe_rate(game, "chosen_board_revisit_turns"),
            )
        ),
        "start_clustered_mean_order_regret_v4_minus_v3": (
            _start_clustered_difference(games, _order_regret)
        ),
    }


def recompute_mechanism_audit(
    *,
    source_spec: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    source_ledger_sha256: str,
    source_result_identity: str,
    implementation_commit: str,
    malom: Any,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Recompute complete safe-progress and full-order process evidence."""
    if len(source_records) != EXPECTED_GAMES:
        raise RetainedPhaseProcessMechanismError("source ledger is not complete")
    games = []
    for game_index, record in enumerate(source_records):
        games.append(
            audit_game_record(
                record,
                malom,
                progress=(
                    None
                    if progress is None
                    else lambda turn, game_index=game_index: progress(
                        game_index,
                        turn,
                    )
                ),
            )
        )
    candidates = {
        candidate: [game for game in games if game["candidate_id"] == candidate]
        for candidate in EXPECTED_CANDIDATES
    }
    if any(len(rows) != EXPECTED_GAMES // 2 for rows in candidates.values()):
        raise RetainedPhaseProcessMechanismError(
            "mechanism candidate support differs"
        )
    phases = sorted({str(game["phase"]) for game in games})
    body = {
        "schema_version": REPORT_SCHEMA,
        "audit_id": "sanmill-retained-v3-v4-phase-process-mechanism-v1",
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
        "by_phase": {
            phase: {
                candidate: _aggregate_games(
                    [
                        game
                        for game in candidates[candidate]
                        if game["phase"] == phase
                    ]
                )
                for candidate in EXPECTED_CANDIDATES
            }
            for phase in phases
        },
        "per_game": games,
        "per_game_identity": canonical_sha256(games),
        "definitions": {
            "safe_capture_opportunity": (
                "at least one complete legal capture preserves exact parent "
                "coarse W/D/L and resets the strict no-capture clock"
            ),
            "board_revisit": (
                "chosen local BoardState FEN appeared at the frozen start or "
                "earlier in the recorded post-start suffix; not strict "
                "threefold adjudication"
            ),
            "within_wdl_orderable": (
                "the recorded choice preserves parent coarse W/D/L and every "
                "coarse-preserving legal action has a complete comparable "
                "OracleMoveValue"
            ),
            "normalised_ordinal_regret": (
                "zero for the best complete Malom ordering grade and one for "
                "the worst distinct preserving grade; positional, not a "
                "distance-to-terminal measure"
            ),
        },
        "claim_boundary": {
            "zero_new_games": True,
            "corpus_previously_project_visible": True,
            "exploratory_mechanism_metrics": True,
            "history_aware_liveness": False,
            "playing_strength_claim": False,
            "refresh_causal_claim": False,
            "promotion_or_publication": False,
        },
    }
    return {**body, "result_identity": canonical_sha256(body)}
