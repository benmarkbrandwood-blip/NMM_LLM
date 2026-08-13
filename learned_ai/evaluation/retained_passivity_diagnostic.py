"""Prospective process evidence for the retained-v3/v4 passivity diagnosis.

The diagnostic deliberately reuses the already inspected 64-start corpus and
therefore produces development evidence only.  Sanmill remains the sole rules
and history owner.  Malom observations are labelled theoretical state/move
diagnostics and never replace strict adjudication.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from game.board import BoardState
from learned_ai.evaluation.heldout_evaluation import (
    ActiveClock,
    replay_frozen_prefix,
)
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.training.sanmill_referee import (
    SanmillAppliedTurn,
    SanmillTrainingGame,
    nmm_move_actions,
)


PLAN_SCHEMA = "nmm.retained-passivity-diagnostic-plan.v1"
SPEC_SCHEMA = "nmm.retained-passivity-diagnostic-spec.v1"
GAME_SCHEMA = "nmm.retained-passivity-diagnostic-game.v1"
REPORT_SCHEMA = "nmm.retained-passivity-diagnostic-result.v1"

EXPECTED_CANDIDATES = ("retained-v3-refresh50", "retained-v4-no-refresh")
EXPECTED_STARTS = 64
EXPECTED_MATCHED_UNITS = 128
EXPECTED_GAMES = 256
HORIZON_LOGICAL_PLY = 120
PREFIX_LOGICAL_PLIES = 12
MAX_POST_PREFIX_LOGICAL_PLIES = 1536
SANMILL_NODE_CEILING = 500_000
ENGINEERING_Z = 1.96
MAX_PRIMARY_HALF_WIDTH = 0.10


class RetainedPassivityDiagnosticError(RuntimeError):
    """Raised when a diagnostic contract or evidence chain differs."""


class RetainedPassivityDiagnosticInvalid(RetainedPassivityDiagnosticError):
    """Raised for a semantic game failure that must stop the run."""


def build_schedule(
    records: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str] = EXPECTED_CANDIDATES,
) -> list[dict[str, Any]]:
    """Build adjacent v3/v4 games for each fixed start/colour unit."""
    if len(records) != EXPECTED_STARTS:
        raise RetainedPassivityDiagnosticError("diagnostic corpus must have 64 starts")
    if tuple(candidate_ids) != EXPECTED_CANDIDATES:
        raise RetainedPassivityDiagnosticError("diagnostic candidate order differs")
    source_ids = [str(record.get("source_core_id") or "") for record in records]
    if any(not value for value in source_ids) or len(set(source_ids)) != len(source_ids):
        raise RetainedPassivityDiagnosticError("diagnostic source IDs are not unique")

    schedule: list[dict[str, Any]] = []
    for start_index, record in enumerate(records):
        execution = record.get("execution_record")
        if not isinstance(execution, Mapping):
            raise RetainedPassivityDiagnosticError("corpus execution record is absent")
        final = execution.get("final")
        if not isinstance(final, Mapping):
            raise RetainedPassivityDiagnosticError("corpus final prefix is absent")
        prefix_identity = execution.get("prefix_identity")
        history_sha256 = final.get("history_sha256")
        if not isinstance(prefix_identity, str) or not isinstance(history_sha256, str):
            raise RetainedPassivityDiagnosticError("corpus prefix identities are absent")
        for color_index, candidate_color in enumerate(("W", "B")):
            unit_index = start_index * 2 + color_index
            match_key = f"{record['source_core_id']}:{candidate_color}"
            for candidate_index, candidate_id in enumerate(candidate_ids):
                ordinal = unit_index * 2 + candidate_index
                identity_body = {
                    "ordinal": ordinal,
                    "match_key": match_key,
                    "candidate_id": candidate_id,
                    "source_core_id": record["source_core_id"],
                    "candidate_color": candidate_color,
                    "prefix_identity": prefix_identity,
                }
                schedule.append(
                    {
                        "ordinal": ordinal,
                        "unit_index": unit_index,
                        "candidate_index": candidate_index,
                        "candidate_id": candidate_id,
                        "candidate_color": candidate_color,
                        "match_key": match_key,
                        "source_core_id": str(record["source_core_id"]),
                        "stratum": str(record.get("stratum") or "unknown"),
                        "prefix_identity": prefix_identity,
                        "expected_prefix_history_sha256": history_sha256,
                        "game_id": "passivity-game:" + canonical_sha256(identity_body),
                    }
                )
    if len(schedule) != EXPECTED_GAMES:
        raise RetainedPassivityDiagnosticError("diagnostic schedule size differs")
    return schedule


def _candidate_perspective_wdl(
    side_to_move_wdl: str,
    *,
    side_to_move: str,
    candidate_color: str,
) -> str:
    if side_to_move_wdl not in {"W", "D", "L"}:
        raise RetainedPassivityDiagnosticError("Malom W/D/L is invalid")
    if side_to_move not in {"W", "B"} or candidate_color not in {"W", "B"}:
        raise RetainedPassivityDiagnosticError("snapshot colour is invalid")
    if side_to_move == candidate_color:
        return side_to_move_wdl
    return {"W": "L", "D": "D", "L": "W"}[side_to_move_wdl]


def _snapshot_at_horizon(
    *,
    board: BoardState,
    state: Any,
    candidate_color: str,
    malom: Any,
) -> dict[str, Any]:
    if state.logical_ply_count != HORIZON_LOGICAL_PLY or state.terminal:
        raise RetainedPassivityDiagnosticError("horizon snapshot is not ongoing at 120")
    theoretical = malom.query_state(board)
    if theoretical is not None and theoretical not in {"W", "D", "L"}:
        raise RetainedPassivityDiagnosticError("Malom snapshot result is invalid")
    side_to_move = board.turn
    candidate_wdl = (
        None
        if theoretical is None
        else _candidate_perspective_wdl(
            theoretical,
            side_to_move=side_to_move,
            candidate_color=candidate_color,
        )
    )
    strict_state = state.portable_record()
    return {
        "absolute_logical_ply": HORIZON_LOGICAL_PLY,
        "post_prefix_logical_ply": HORIZON_LOGICAL_PLY - PREFIX_LOGICAL_PLIES,
        "local_fen": board.to_fen_string(),
        "history_sha256": state.history_sha256,
        "strict_referee_state": strict_state,
        "malom_theoretical": {
            "history_aware": False,
            "queryable": theoretical is not None,
            "side_to_move": side_to_move,
            "side_to_move_wdl": theoretical,
            "candidate_color": candidate_color,
            "candidate_perspective_wdl": candidate_wdl,
        },
    }


def _normalised_move(move: Mapping[str, Any]) -> dict[str, str | None]:
    return {
        "from": None if move.get("from") is None else str(move["from"]),
        "to": None if move.get("to") is None else str(move["to"]),
        "capture": None if move.get("capture") is None else str(move["capture"]),
    }


def _search_record(applied: SanmillAppliedTurn) -> dict[str, Any] | None:
    if applied.search is None:
        return None
    record = applied.search.semantic_record()
    if record.get("node_budget") != SANMILL_NODE_CEILING:
        raise RetainedPassivityDiagnosticError("Sanmill node ceiling differs")
    total_nodes = record.get("total_nodes")
    if not isinstance(total_nodes, int) or not 0 <= total_nodes <= SANMILL_NODE_CEILING:
        raise RetainedPassivityDiagnosticError("Sanmill node evidence is invalid")
    if not isinstance(record.get("search_calls"), int) or record["search_calls"] <= 0:
        raise RetainedPassivityDiagnosticError("Sanmill search evidence is absent")
    return record


def _turn_record(
    *,
    post_prefix_ply: int,
    mover_color: str,
    actor: str,
    board_after: BoardState,
    before_history: str,
    applied: SanmillAppliedTurn,
    candidate_malom_delta: float | None,
) -> dict[str, Any]:
    return {
        "post_prefix_logical_ply": post_prefix_ply,
        "absolute_logical_ply": PREFIX_LOGICAL_PLIES + post_prefix_ply,
        "mover_color": mover_color,
        "actor": actor,
        "move": _normalised_move(applied.move),
        "actions": list(applied.actions),
        "before_history_sha256": before_history,
        "after_history_sha256": applied.state.history_sha256,
        "local_fen_after": board_after.to_fen_string(),
        "sanmill_fen_after": applied.state.fen,
        "terminal": bool(applied.state.terminal),
        "outcome_reason": str(applied.state.outcome_reason),
        "candidate_malom_delta": candidate_malom_delta,
        "search": _search_record(applied),
    }


def _candidate_malom_summary(turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_turns = [turn for turn in turns if turn.get("actor") == "candidate"]
    values = [turn.get("candidate_malom_delta") for turn in candidate_turns]
    queryable = [float(value) for value in values if value is not None]
    if any(value not in (0.0, -1.0, -2.0) for value in queryable):
        raise RetainedPassivityDiagnosticError("candidate Malom delta is invalid")
    preserving = sum(value == 0.0 for value in queryable)
    one_step = sum(value == -1.0 for value in queryable)
    two_step = sum(value == -2.0 for value in queryable)
    return {
        "candidate_turns": len(candidate_turns),
        "queryable_turns": len(queryable),
        "unqueryable_turns": len(candidate_turns) - len(queryable),
        "preserving_turns": preserving,
        "one_step_downgrade_turns": one_step,
        "two_step_downgrade_turns": two_step,
        "query_coverage": len(queryable) / len(candidate_turns) if candidate_turns else None,
        "preserving_rate_given_queryable": (
            preserving / len(queryable) if queryable else None
        ),
        "downgrade_rate_given_queryable": (
            (one_step + two_step) / len(queryable) if queryable else None
        ),
    }


def _search_summary(turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    searches = [turn["search"] for turn in turns if turn.get("search") is not None]
    nodes = [int(search["total_nodes"]) for search in searches]
    depths = [
        int(search["completed_depth"])
        for search in searches
        if search.get("completed_depth") is not None
    ]
    return {
        "turns": len(searches),
        "node_ceiling_per_turn": SANMILL_NODE_CEILING,
        "total_observed_nodes": sum(nodes),
        "min_observed_nodes": min(nodes) if nodes else None,
        "max_observed_nodes": max(nodes) if nodes else None,
        "min_completed_depth": min(depths) if depths else None,
        "max_completed_depth": max(depths) if depths else None,
    }


def play_diagnostic_game(
    *,
    spec: Mapping[str, Any],
    schedule_item: Mapping[str, Any],
    corpus_record: Mapping[str, Any],
    policy: Any,
    installation: Any,
    previous_record_sha256: str | None,
    clock: ActiveClock,
    progress_callback: Callable[[str, int], None],
    game_factory: Callable[..., Any] = SanmillTrainingGame,
) -> dict[str, Any]:
    """Play one strict game while retaining prospective process evidence."""
    game_started = clock.elapsed()
    turns: list[dict[str, Any]] = []
    horizon_snapshot: dict[str, Any] | None = None
    horizon_survival = False
    safety_cap = False

    with game_factory(installation, seed=int(spec["runtime"]["seed"])) as game:

        def prefix_progress(ply: int) -> None:
            clock.require_within_budget()
            progress_callback("prefix", ply)

        board, prefix = replay_frozen_prefix(
            game,
            corpus_record,
            progress=prefix_progress,
        )
        if prefix["prefix_identity"] != schedule_item["prefix_identity"]:
            raise RetainedPassivityDiagnosticError("runtime prefix identity differs")
        if prefix["observed_history_sha256"] != schedule_item[
            "expected_prefix_history_sha256"
        ]:
            raise RetainedPassivityDiagnosticError("runtime prefix history differs")

        candidate_color = str(schedule_item["candidate_color"])
        max_plies = int(spec["protocol"]["max_post_prefix_logical_plies"])
        for post_prefix_ply in range(1, max_plies + 1):
            clock.require_within_budget()
            before_history = game.state.history_sha256
            mover = board.turn
            candidate_delta: float | None = None
            if mover == candidate_color:
                actor = "candidate"
                move = policy.choose_move(board)
                if not move:
                    raise RetainedPassivityDiagnosticInvalid(
                        "candidate returned no move in an ongoing state"
                    )
                raw_delta = policy.malom.query_move_quality(board, move)
                if raw_delta is not None:
                    candidate_delta = float(raw_delta)
                    if candidate_delta not in (0.0, -1.0, -2.0):
                        raise RetainedPassivityDiagnosticInvalid(
                            "candidate Malom move delta is outside exact W/D/L"
                        )
                applied = game.apply_nmm_move(board, move)
            else:
                actor = "sanmill"
                applied = game.search_and_apply(
                    board,
                    node_budget=SANMILL_NODE_CEILING,
                    depth=None,
                )
            board = board.apply_move(applied.move)
            turn = _turn_record(
                post_prefix_ply=post_prefix_ply,
                mover_color=mover,
                actor=actor,
                board_after=board,
                before_history=before_history,
                applied=applied,
                candidate_malom_delta=candidate_delta,
            )
            turns.append(turn)
            clock.require_within_budget()
            progress_callback("game", post_prefix_ply)

            if applied.state.logical_ply_count == HORIZON_LOGICAL_PLY:
                if not applied.state.terminal:
                    horizon_survival = True
                    horizon_snapshot = _snapshot_at_horizon(
                        board=board,
                        state=applied.state,
                        candidate_color=candidate_color,
                        malom=policy.malom,
                    )
            elif (
                applied.state.logical_ply_count > HORIZON_LOGICAL_PLY
                and horizon_snapshot is None
            ):
                raise RetainedPassivityDiagnosticError("ply-120 snapshot was skipped")

            if applied.state.terminal:
                break
        else:
            safety_cap = True

        final_state = game.state.portable_record()
        if safety_cap:
            if game.state.terminal:
                raise RetainedPassivityDiagnosticError("terminal state reached at safety cap")
            winner = None
            score = None
            outcome_reason = "safety_cap_incomplete"
            termination_class = "safety_cap_incomplete"
        else:
            if not game.state.terminal or game.state.outcome_reason == "ongoing":
                raise RetainedPassivityDiagnosticInvalid(
                    "game stopped without a rules terminal or safety cap"
                )
            winner_name = game.state.winner
            winner = {None: None, "white": "W", "black": "B"}.get(winner_name)
            if winner_name not in {None, "white", "black"}:
                raise RetainedPassivityDiagnosticError("Sanmill winner value is unknown")
            score = 0.5 if winner is None else (1.0 if winner == candidate_color else 0.0)
            outcome_reason = str(game.state.outcome_reason)
            termination_class = "rules_terminal"

        active = clock.require_within_budget()
        return {
            "schema_version": GAME_SCHEMA,
            "spec_identity": spec["spec_identity"],
            "ordinal": schedule_item["ordinal"],
            "unit_index": schedule_item["unit_index"],
            "game_id": schedule_item["game_id"],
            "match_key": schedule_item["match_key"],
            "candidate_id": schedule_item["candidate_id"],
            "candidate_color": candidate_color,
            "source_core_id": schedule_item["source_core_id"],
            "stratum": schedule_item["stratum"],
            "prefix": prefix,
            "ongoing_after_total_logical_ply_120": horizon_survival,
            "ply_120_snapshot": horizon_snapshot,
            "post_prefix_logical_plies": len(turns),
            "total_logical_plies": PREFIX_LOGICAL_PLIES + len(turns),
            "termination_class": termination_class,
            "outcome_reason": outcome_reason,
            "winner": winner,
            "candidate_score": score,
            "final_state": final_state,
            "turns": turns,
            "candidate_malom": _candidate_malom_summary(turns),
            "sanmill_search": _search_summary(turns),
            "game_elapsed_seconds": round(active - game_started, 6),
            "cumulative_active_seconds": round(active, 6),
            "complete_diagnostic": True,
            "previous_record_sha256": previous_record_sha256,
        }


def _validate_game_record(
    spec: Mapping[str, Any],
    record: Mapping[str, Any],
    ordinal: int,
    previous_hash: str | None,
) -> None:
    schedule = spec.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != EXPECTED_GAMES:
        raise RetainedPassivityDiagnosticError("runtime schedule differs")
    if ordinal >= len(schedule) or record.get("ordinal") != ordinal:
        raise RetainedPassivityDiagnosticError("diagnostic ordinal differs")
    expected = schedule[ordinal]
    for field in (
        "game_id",
        "match_key",
        "candidate_id",
        "candidate_color",
        "source_core_id",
        "stratum",
        "unit_index",
    ):
        if record.get(field) != expected.get(field):
            raise RetainedPassivityDiagnosticError(f"diagnostic {field} differs")
    if record.get("schema_version") != GAME_SCHEMA:
        raise RetainedPassivityDiagnosticError("diagnostic game schema differs")
    if record.get("spec_identity") != spec.get("spec_identity"):
        raise RetainedPassivityDiagnosticError("diagnostic spec identity differs")
    if record.get("previous_record_sha256") != previous_hash:
        raise RetainedPassivityDiagnosticError("diagnostic ledger chain differs")
    if record.get("complete_diagnostic") is not True:
        raise RetainedPassivityDiagnosticError("diagnostic game is not complete")
    plies = record.get("post_prefix_logical_plies")
    if not isinstance(plies, int) or not 1 <= plies <= MAX_POST_PREFIX_LOGICAL_PLIES:
        raise RetainedPassivityDiagnosticError("diagnostic game length is invalid")
    if record.get("total_logical_plies") != PREFIX_LOGICAL_PLIES + plies:
        raise RetainedPassivityDiagnosticError("diagnostic total game length differs")

    survival = record.get("ongoing_after_total_logical_ply_120")
    snapshot = record.get("ply_120_snapshot")
    if not isinstance(survival, bool):
        raise RetainedPassivityDiagnosticError("horizon survival flag is invalid")
    if survival:
        if not isinstance(snapshot, Mapping):
            raise RetainedPassivityDiagnosticError("surviving game lacks a snapshot")
        if snapshot.get("absolute_logical_ply") != HORIZON_LOGICAL_PLY:
            raise RetainedPassivityDiagnosticError("snapshot logical ply differs")
        if plies <= HORIZON_LOGICAL_PLY - PREFIX_LOGICAL_PLIES:
            raise RetainedPassivityDiagnosticError("snapshot exceeds game length")
    elif snapshot is not None:
        raise RetainedPassivityDiagnosticError("non-surviving game has a snapshot")
    if plies > HORIZON_LOGICAL_PLY - PREFIX_LOGICAL_PLIES and not survival:
        raise RetainedPassivityDiagnosticError("long game omitted horizon survival")

    turns = record.get("turns")
    if not isinstance(turns, list) or len(turns) != plies:
        raise RetainedPassivityDiagnosticError("diagnostic turn ledger differs")
    previous = record.get("prefix", {}).get("observed_history_sha256")
    if not isinstance(previous, str):
        raise RetainedPassivityDiagnosticError("diagnostic prefix history is absent")
    candidate_color = str(record["candidate_color"])
    for index, turn in enumerate(turns, 1):
        if not isinstance(turn, Mapping):
            raise RetainedPassivityDiagnosticError("diagnostic turn is invalid")
        mover = "W" if index % 2 == 1 else "B"
        expected_actor = "candidate" if mover == candidate_color else "sanmill"
        if turn.get("post_prefix_logical_ply") != index:
            raise RetainedPassivityDiagnosticError("diagnostic turn order differs")
        if turn.get("absolute_logical_ply") != PREFIX_LOGICAL_PLIES + index:
            raise RetainedPassivityDiagnosticError("diagnostic absolute ply differs")
        if turn.get("mover_color") != mover or turn.get("actor") != expected_actor:
            raise RetainedPassivityDiagnosticError("diagnostic actor order differs")
        if turn.get("before_history_sha256") != previous:
            raise RetainedPassivityDiagnosticError("diagnostic history chain differs")
        previous = turn.get("after_history_sha256")
        if not isinstance(previous, str) or len(previous) != 64:
            raise RetainedPassivityDiagnosticError("diagnostic history hash is invalid")
        actions = turn.get("actions")
        move = turn.get("move")
        if not isinstance(actions, list) or not isinstance(move, Mapping):
            raise RetainedPassivityDiagnosticError("diagnostic move evidence is invalid")
        if actions != list(nmm_move_actions(move)):
            raise RetainedPassivityDiagnosticError("diagnostic actions differ from move")
        delta = turn.get("candidate_malom_delta")
        if expected_actor == "candidate":
            if delta is not None and delta not in (0.0, -1.0, -2.0):
                raise RetainedPassivityDiagnosticError("candidate Malom delta differs")
            if turn.get("search") is not None:
                raise RetainedPassivityDiagnosticError("candidate turn has search evidence")
        elif delta is not None or not isinstance(turn.get("search"), Mapping):
            raise RetainedPassivityDiagnosticError("Sanmill turn evidence differs")
        is_last = index == plies
        if record.get("termination_class") == "rules_terminal":
            if bool(turn.get("terminal")) is not is_last:
                raise RetainedPassivityDiagnosticError("terminal turn placement differs")
        elif turn.get("terminal"):
            raise RetainedPassivityDiagnosticError("safety-cap game has terminal turn")

    final_state = record.get("final_state")
    if not isinstance(final_state, Mapping) or final_state.get("history_sha256") != previous:
        raise RetainedPassivityDiagnosticError("diagnostic final history differs")
    if record.get("candidate_malom") != _candidate_malom_summary(turns):
        raise RetainedPassivityDiagnosticError("candidate Malom summary differs")
    if record.get("sanmill_search") != _search_summary(turns):
        raise RetainedPassivityDiagnosticError("Sanmill search summary differs")

    termination = record.get("termination_class")
    if termination == "rules_terminal":
        if final_state.get("terminal") is not True or record.get("candidate_score") not in (
            0.0,
            0.5,
            1.0,
        ):
            raise RetainedPassivityDiagnosticError("rules-terminal result differs")
    elif termination == "safety_cap_incomplete":
        if (
            plies != MAX_POST_PREFIX_LOGICAL_PLIES
            or final_state.get("terminal") is not False
            or record.get("winner") is not None
            or record.get("candidate_score") is not None
        ):
            raise RetainedPassivityDiagnosticError("safety-cap disposition differs")
    else:
        raise RetainedPassivityDiagnosticError("diagnostic termination class differs")


def append_game_record(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    must_create: bool,
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record_hash = canonical_sha256(record)
    wrapper = {"record": dict(record), "record_sha256": record_hash}
    mode = "xb" if must_create else "ab"
    with target.open(mode) as handle:
        handle.write(canonical_json_bytes(wrapper) + b"\n")
        handle.flush()
        import os

        os.fsync(handle.fileno())
    return record_hash


def load_game_ledger(
    spec: Mapping[str, Any],
    path: str | Path,
) -> tuple[list[dict[str, Any]], str | None]:
    target = Path(path)
    if not target.exists():
        return [], None
    records: list[dict[str, Any]] = []
    previous: str | None = None
    with target.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith(b"\n") or raw.endswith(b"\r\n"):
                raise RetainedPassivityDiagnosticError(
                    f"diagnostic ledger line {line_number} is not LF-framed"
                )
            try:
                wrapper = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RetainedPassivityDiagnosticError(
                    f"diagnostic ledger line {line_number} is invalid JSON"
                ) from exc
            if canonical_json_bytes(wrapper) + b"\n" != raw:
                raise RetainedPassivityDiagnosticError(
                    f"diagnostic ledger line {line_number} is not canonical JSON"
                )
            if not isinstance(wrapper, dict) or set(wrapper) != {
                "record",
                "record_sha256",
            }:
                raise RetainedPassivityDiagnosticError("diagnostic ledger wrapper differs")
            record = wrapper["record"]
            if not isinstance(record, dict) or wrapper["record_sha256"] != canonical_sha256(
                record
            ):
                raise RetainedPassivityDiagnosticError("diagnostic ledger hash differs")
            _validate_game_record(spec, record, len(records), previous)
            previous = str(wrapper["record_sha256"])
            records.append(record)
    if len(records) > EXPECTED_GAMES:
        raise RetainedPassivityDiagnosticError("diagnostic ledger has too many games")
    return records, previous


def _interval(values: Sequence[float], z: float = ENGINEERING_Z) -> dict[str, Any]:
    if not values:
        return {
            "support": 0,
            "mean": None,
            "sample_standard_deviation": None,
            "standard_error": None,
            "interval": [None, None],
            "half_width": None,
        }
    mean = sum(values) / len(values)
    if len(values) == 1:
        deviation = 0.0
        error = 0.0
    else:
        deviation = statistics.stdev(values)
        error = deviation / math.sqrt(len(values))
    half_width = z * error
    return {
        "support": len(values),
        "mean": mean,
        "sample_standard_deviation": deviation,
        "standard_error": error,
        "interval": [mean - half_width, mean + half_width],
        "half_width": half_width,
    }


def _quantile(values: Sequence[int], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _candidate_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    games = len(records)
    survival = sum(bool(record["ongoing_after_total_logical_ply_120"]) for record in records)
    lengths = [int(record["total_logical_plies"]) for record in records]
    rules_records = [
        record for record in records if record["termination_class"] == "rules_terminal"
    ]
    scores = [float(record["candidate_score"]) for record in rules_records]
    candidate_malom = Counter()
    for record in records:
        summary = record["candidate_malom"]
        for key in (
            "candidate_turns",
            "queryable_turns",
            "unqueryable_turns",
            "preserving_turns",
            "one_step_downgrade_turns",
            "two_step_downgrade_turns",
        ):
            candidate_malom[key] += int(summary[key])
    queryable = candidate_malom["queryable_turns"]
    candidate_turns = candidate_malom["candidate_turns"]
    preserving = candidate_malom["preserving_turns"]
    downgrading = (
        candidate_malom["one_step_downgrade_turns"]
        + candidate_malom["two_step_downgrade_turns"]
    )
    snapshots = [
        record["ply_120_snapshot"]
        for record in records
        if record["ply_120_snapshot"] is not None
    ]
    malom_wdl = Counter(
        snapshot["malom_theoretical"]["candidate_perspective_wdl"] or "unqueryable"
        for snapshot in snapshots
    )
    return {
        "games": games,
        "horizon_120": {
            "survived": survival,
            "rules_terminal_on_or_before": games - survival,
            "survival_rate": survival / games if games else None,
        },
        "termination_classes": dict(
            sorted(Counter(record["termination_class"] for record in records).items())
        ),
        "outcome_reasons": dict(
            sorted(Counter(record["outcome_reason"] for record in records).items())
        ),
        "lengths": {
            "support": games,
            "min_total_logical_plies": min(lengths) if lengths else None,
            "median_total_logical_plies": _quantile(lengths, 0.5),
            "p90_total_logical_plies": _quantile(lengths, 0.9),
            "max_total_logical_plies": max(lengths) if lengths else None,
            "mean_total_logical_plies": sum(lengths) / games if games else None,
        },
        "malom_at_ply_120_candidate_perspective": {
            "snapshot_support": len(snapshots),
            "queryable": len(snapshots) - malom_wdl["unqueryable"],
            "unqueryable": malom_wdl["unqueryable"],
            "wins": malom_wdl["W"],
            "draws": malom_wdl["D"],
            "losses": malom_wdl["L"],
            "history_aware": False,
        },
        "candidate_malom_moves": {
            **dict(candidate_malom),
            "query_coverage": queryable / candidate_turns if candidate_turns else None,
            "preserving_rate_given_queryable": (
                preserving / queryable if queryable else None
            ),
            "downgrade_rate_given_queryable": (
                downgrading / queryable if queryable else None
            ),
        },
        "eventual_rules_wdl": {
            "support": len(rules_records),
            "wins": sum(score == 1.0 for score in scores),
            "draws": sum(score == 0.5 for score in scores),
            "losses": sum(score == 0.0 for score in scores),
            "score_rate": sum(scores) / len(scores) if scores else None,
            "safety_cap_excluded": games - len(rules_records),
            "strength_claim_allowed": False,
        },
    }


def _paired_comparison(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_unit: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        by_unit.setdefault(str(record["match_key"]), {})[str(record["candidate_id"])] = record
    complete_units = [
        unit for unit in by_unit.values() if set(unit) == set(EXPECTED_CANDIDATES)
    ]
    horizon_differences: list[float] = []
    length_differences: list[float] = []
    preserving_differences: list[float] = []
    for unit in complete_units:
        v3 = unit[EXPECTED_CANDIDATES[0]]
        v4 = unit[EXPECTED_CANDIDATES[1]]
        horizon_differences.append(
            float(v4["ongoing_after_total_logical_ply_120"])
            - float(v3["ongoing_after_total_logical_ply_120"])
        )
        length_differences.append(
            (int(v4["post_prefix_logical_plies"]) - int(v3["post_prefix_logical_plies"]))
            / MAX_POST_PREFIX_LOGICAL_PLIES
        )
        v3_rate = v3["candidate_malom"]["preserving_rate_given_queryable"]
        v4_rate = v4["candidate_malom"]["preserving_rate_given_queryable"]
        if v3_rate is not None and v4_rate is not None:
            preserving_differences.append(float(v4_rate) - float(v3_rate))

    primary = _interval(horizon_differences)
    complete = len(complete_units) == EXPECTED_MATCHED_UNITS
    if not complete:
        decision = "pending"
    elif primary["half_width"] is None or primary["half_width"] > MAX_PRIMARY_HALF_WIDTH:
        decision = "inconclusive_precision"
    elif primary["interval"][0] > 0:
        decision = "v4_higher_120_ply_survival"
    elif primary["interval"][1] < 0:
        decision = "v3_higher_120_ply_survival"
    else:
        decision = "inconclusive"
    return {
        "matched_units_complete": len(complete_units),
        "matched_units_expected": EXPECTED_MATCHED_UNITS,
        "primary_horizon_survival_v4_minus_v3": {
            **primary,
            "decision": decision,
            "maximum_half_width": MAX_PRIMARY_HALF_WIDTH,
            "precision_adequate": (
                primary["half_width"] is not None
                and primary["half_width"] <= MAX_PRIMARY_HALF_WIDTH
            ),
            "interpretation": "fixed-corpus engineering interval, not population inference",
        },
        "restricted_length_v4_minus_v3": _interval(length_differences),
        "per_game_preserving_rate_v4_minus_v3": _interval(
            preserving_differences
        ),
    }


def recompute_diagnostic(
    spec: Mapping[str, Any],
    ledger_path: str | Path,
) -> dict[str, Any]:
    """Recompute partial or complete web/report summaries from the ledger."""
    records, tail = load_game_ledger(spec, ledger_path)
    grouped = {
        candidate_id: [
            record for record in records if record["candidate_id"] == candidate_id
        ]
        for candidate_id in EXPECTED_CANDIDATES
    }
    body = {
        "schema_version": REPORT_SCHEMA,
        "diagnostic_id": spec["diagnostic_id"],
        "spec_identity": spec["spec_identity"],
        "status": "completed" if len(records) == EXPECTED_GAMES else "partial",
        "completed_games": len(records),
        "expected_games": EXPECTED_GAMES,
        "ledger_tail_record_sha256": tail,
        "by_candidate": {
            candidate_id: _candidate_summary(grouped[candidate_id])
            for candidate_id in EXPECTED_CANDIDATES
        },
        "paired": _paired_comparison(records),
        "by_candidate_color": {
            color: {
                candidate_id: _candidate_summary(
                    [
                        record
                        for record in grouped[candidate_id]
                        if record["candidate_color"] == color
                    ]
                )
                for candidate_id in EXPECTED_CANDIDATES
            }
            for color in ("W", "B")
        },
        "by_source_stratum": {
            stratum: {
                candidate_id: _candidate_summary(
                    [
                        record
                        for record in grouped[candidate_id]
                        if record["stratum"] == stratum
                    ]
                )
                for candidate_id in EXPECTED_CANDIDATES
            }
            for stratum in sorted({str(record["stratum"]) for record in records})
        },
        "claim_boundary": {
            "development_corpus_reused": True,
            "playing_strength_claim": False,
            "refresh_causal_claim": False,
            "promotion_or_publication": False,
            "malom_is_history_aware": False,
            "safety_cap_is_draw": False,
        },
    }
    return {**body, "result_identity": canonical_sha256(body)}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
