"""Relative-horizon process evaluation on frozen replayable phase histories."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.evaluation.heldout_evaluation import ActiveClock
from learned_ai.evaluation.retained_passivity_diagnostic import (
    _candidate_malom_summary,
    _candidate_perspective_wdl,
    _interval,
    _normalised_move,
    _quantile,
    _search_record,
    _search_summary,
)
from learned_ai.evaluation.retained_phase_process_corpus import (
    EXPECTED_ACCEPTED_COUNT,
    validate_retained_phase_process_corpus,
)
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.training.sanmill_referee import (
    SanmillAppliedTurn,
    SanmillTrainingGame,
    nmm_move_actions,
)


PLAN_SCHEMA = "nmm.retained-phase-process-generalization-plan.v1"
SPEC_SCHEMA = "nmm.retained-phase-process-generalization-spec.v1"
GAME_SCHEMA = "nmm.retained-phase-process-generalization-game.v1"
REPORT_SCHEMA = "nmm.retained-phase-process-generalization-result.v1"
EXPECTED_CANDIDATES = ("retained-v3-refresh50", "retained-v4-no-refresh")
EXPECTED_STARTS = EXPECTED_ACCEPTED_COUNT
EXPECTED_MATCHED_COLOUR_UNITS = EXPECTED_STARTS * 2
EXPECTED_GAMES = EXPECTED_STARTS * 4
HORIZON_POST_START_LOGICAL_PLIES = 108
MAX_POST_START_LOGICAL_PLIES = 1536
SANMILL_NODE_CEILING = 500_000
MAX_PRIMARY_HALF_WIDTH = 0.10


class RetainedPhaseProcessError(RuntimeError):
    """Raised when the phase-process contract or evidence differs."""


class RetainedPhaseProcessInvalid(RetainedPhaseProcessError):
    """Raised for a semantic game failure that must stop the evaluation."""


def build_schedule(
    records: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str] = EXPECTED_CANDIDATES,
) -> list[dict[str, Any]]:
    """Build adjacent v3/v4 games for both candidate colours at every start."""
    if len(records) != EXPECTED_STARTS:
        raise RetainedPhaseProcessError("phase-process corpus must have 39 starts")
    if tuple(candidate_ids) != EXPECTED_CANDIDATES:
        raise RetainedPhaseProcessError("phase-process candidate order differs")
    start_ids = [str(record.get("start_id") or "") for record in records]
    if any(not item for item in start_ids) or len(set(start_ids)) != len(start_ids):
        raise RetainedPhaseProcessError("phase-process start IDs are not unique")

    schedule = []
    for start_index, record in enumerate(records):
        strict_start = record.get("strict_start")
        if not isinstance(strict_start, Mapping):
            raise RetainedPhaseProcessError("phase-process strict start is absent")
        record_identity = record.get("record_identity")
        history_sha256 = strict_start.get("history_sha256")
        start_ply = strict_start.get("logical_ply_count")
        start_turn = record.get("turn")
        if (
            not isinstance(record_identity, str)
            or not isinstance(history_sha256, str)
            or not isinstance(start_ply, int)
            or start_turn not in {"W", "B"}
        ):
            raise RetainedPhaseProcessError("phase-process start identity differs")
        for colour_index, candidate_color in enumerate(("W", "B")):
            unit_index = start_index * 2 + colour_index
            match_key = f"{record['start_id']}:{candidate_color}"
            for candidate_index, candidate_id in enumerate(candidate_ids):
                ordinal = unit_index * 2 + candidate_index
                identity_body = {
                    "ordinal": ordinal,
                    "match_key": match_key,
                    "candidate_id": candidate_id,
                    "candidate_color": candidate_color,
                    "start_id": record["start_id"],
                    "start_record_identity": record_identity,
                }
                schedule.append(
                    {
                        "ordinal": ordinal,
                        "unit_index": unit_index,
                        "candidate_index": candidate_index,
                        "candidate_id": candidate_id,
                        "candidate_color": candidate_color,
                        "match_key": match_key,
                        "start_id": str(record["start_id"]),
                        "phase": str(record["phase"]),
                        "start_turn": str(start_turn),
                        "start_logical_ply": start_ply,
                        "start_record_identity": str(record_identity),
                        "expected_start_history_sha256": str(history_sha256),
                        "game_id": "phase-process-game:"
                        + canonical_sha256(identity_body),
                    }
                )
    if len(schedule) != EXPECTED_GAMES:
        raise RetainedPhaseProcessError("phase-process schedule size differs")
    return schedule


def _matching_move(board: BoardState, actions: Sequence[str]) -> Any:
    expected = tuple(str(action) for action in actions)
    matches = [
        move
        for move in get_all_legal_moves(board)
        if nmm_move_actions(move) == expected
    ]
    if len(matches) != 1:
        raise RetainedPhaseProcessError(
            f"phase-process history selects {len(matches)} legal moves"
        )
    return matches[0]


def replay_frozen_start(
    game: Any,
    record: Mapping[str, Any],
    *,
    progress: Callable[[int], None] | None = None,
) -> tuple[BoardState, dict[str, Any]]:
    """Replay one variable-length frozen history and verify its strict state."""
    turns = record.get("logical_turns")
    if not isinstance(turns, list) or not turns:
        raise RetainedPhaseProcessError("phase-process history is absent")
    board = BoardState.new_game()
    for index, actions in enumerate(turns, 1):
        if not isinstance(actions, list) or any(
            not isinstance(action, str) for action in actions
        ):
            raise RetainedPhaseProcessError("phase-process history action differs")
        if game.state.terminal:
            raise RetainedPhaseProcessError(
                "strict referee terminated before the frozen start"
            )
        move = _matching_move(board, actions)
        game.apply_nmm_move(board, move)
        board = board.apply_move(move)
        if progress is not None:
            progress(index)

    strict = record.get("strict_start")
    if not isinstance(strict, Mapping):
        raise RetainedPhaseProcessError("frozen strict-start evidence is absent")
    game.assert_current_board(board)
    expected = {
        "history_sha256": strict.get("history_sha256"),
        "sanmill_fen": strict.get("sanmill_fen"),
        "logical_ply_count": strict.get("logical_ply_count"),
        "action_token_count": strict.get("action_token_count"),
        "logical_plies_by_side": strict.get("logical_plies_by_side"),
        "no_capture_count": strict.get("no_capture_count"),
        "repetition_current_count": strict.get("repetition_current_count"),
    }
    observed = {
        "history_sha256": game.state.history_sha256,
        "sanmill_fen": game.state.fen,
        "logical_ply_count": game.state.logical_ply_count,
        "action_token_count": game.state.action_token_count,
        "logical_plies_by_side": list(game.state.logical_plies_by_side),
        "no_capture_count": game.state.no_capture_count,
        "repetition_current_count": game.state.repetition_current_count,
    }
    if expected != observed:
        raise RetainedPhaseProcessError("frozen strict-start state differs")
    if board.to_fen_string() != record.get("fen"):
        raise RetainedPhaseProcessError("frozen local start FEN differs")
    if game.state.terminal:
        raise RetainedPhaseProcessError("frozen phase-process start is terminal")
    return board, {
        "start_id": record.get("start_id"),
        "start_record_identity": record.get("record_identity"),
        "expected_history_sha256": expected["history_sha256"],
        "observed_history_sha256": observed["history_sha256"],
        "logical_ply_count": observed["logical_ply_count"],
        "action_token_count": observed["action_token_count"],
        "logical_plies_by_side": observed["logical_plies_by_side"],
        "no_capture_count": observed["no_capture_count"],
        "repetition_current_count": observed["repetition_current_count"],
        "repetition_history_length": game.state.repetition_history_length,
        "final_nmm_fen": board.to_fen_string(),
        "final_sanmill_fen": observed["sanmill_fen"],
    }


def _snapshot_at_horizon(
    *,
    board: BoardState,
    state: Any,
    candidate_color: str,
    malom: Any,
    start_logical_ply: int,
) -> dict[str, Any]:
    if (
        state.logical_ply_count != start_logical_ply + HORIZON_POST_START_LOGICAL_PLIES
        or state.terminal
    ):
        raise RetainedPhaseProcessError("relative-horizon snapshot differs")
    theoretical = malom.query_state(board)
    if theoretical is not None and theoretical not in {"W", "D", "L"}:
        raise RetainedPhaseProcessError("relative-horizon Malom value differs")
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
    return {
        "post_start_logical_ply": HORIZON_POST_START_LOGICAL_PLIES,
        "absolute_logical_ply": state.logical_ply_count,
        "local_fen": board.to_fen_string(),
        "history_sha256": state.history_sha256,
        "strict_referee_state": state.portable_record(),
        "malom_theoretical": {
            "history_aware": False,
            "queryable": theoretical is not None,
            "side_to_move": side_to_move,
            "side_to_move_wdl": theoretical,
            "candidate_color": candidate_color,
            "candidate_perspective_wdl": candidate_wdl,
        },
    }


def _turn_record(
    *,
    post_start_ply: int,
    start_logical_ply: int,
    mover_color: str,
    actor: str,
    board_after: BoardState,
    before_history: str,
    applied: SanmillAppliedTurn,
    candidate_malom_delta: float | None,
) -> dict[str, Any]:
    return {
        "post_start_logical_ply": post_start_ply,
        "absolute_logical_ply": start_logical_ply + post_start_ply,
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
        "no_capture_count": int(applied.state.no_capture_count),
        "repetition_current_count": int(applied.state.repetition_current_count),
        "repetition_history_length": int(applied.state.repetition_history_length),
        "candidate_malom_delta": candidate_malom_delta,
        "search": _search_record(applied),
    }


def _history_process(
    start: Mapping[str, Any],
    snapshot: Mapping[str, Any] | None,
    final_state: Mapping[str, Any],
) -> dict[str, Any]:
    horizon_state = (
        snapshot.get("strict_referee_state") if isinstance(snapshot, Mapping) else None
    )
    return {
        "start": {
            "no_capture_count": start.get("no_capture_count"),
            "repetition_current_count": start.get("repetition_current_count"),
            "repetition_history_length": start.get("repetition_history_length"),
        },
        "horizon": (
            None
            if not isinstance(horizon_state, Mapping)
            else {
                "no_capture_count": horizon_state.get("no_capture_count"),
                "repetition_current_count": horizon_state.get(
                    "repetition_current_count"
                ),
                "repetition_history_length": horizon_state.get(
                    "repetition_history_length"
                ),
            }
        ),
        "final": {
            "no_capture_count": final_state.get("no_capture_count"),
            "repetition_current_count": final_state.get("repetition_current_count"),
            "repetition_history_length": final_state.get("repetition_history_length"),
        },
    }


def play_phase_process_game(
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
    game_schema: str = GAME_SCHEMA,
) -> dict[str, Any]:
    """Play one strict game from a variable frozen history."""
    game_started = clock.elapsed()
    turns: list[dict[str, Any]] = []
    horizon_snapshot: dict[str, Any] | None = None
    horizon_survival = False
    safety_cap = False

    with game_factory(installation, seed=int(spec["runtime"]["seed"])) as game:

        def start_progress(ply: int) -> None:
            clock.require_within_budget()
            progress_callback("start", ply)

        board, start = replay_frozen_start(
            game,
            corpus_record,
            progress=start_progress,
        )
        if start["start_record_identity"] != schedule_item["start_record_identity"]:
            raise RetainedPhaseProcessError("runtime start identity differs")
        if (
            start["observed_history_sha256"]
            != schedule_item["expected_start_history_sha256"]
        ):
            raise RetainedPhaseProcessError("runtime start history differs")
        start_logical_ply = int(start["logical_ply_count"])
        candidate_color = str(schedule_item["candidate_color"])
        max_plies = int(spec["protocol"]["max_post_start_logical_plies"])

        for post_start_ply in range(1, max_plies + 1):
            clock.require_within_budget()
            before_history = game.state.history_sha256
            mover = board.turn
            candidate_delta: float | None = None
            if mover == candidate_color:
                actor = "candidate"
                move = policy.choose_move(board)
                if not move:
                    raise RetainedPhaseProcessInvalid(
                        "candidate returned no move in an ongoing state"
                    )
                raw_delta = policy.malom.query_move_quality(board, move)
                if raw_delta is not None:
                    candidate_delta = float(raw_delta)
                    if candidate_delta not in (0.0, -1.0, -2.0):
                        raise RetainedPhaseProcessInvalid(
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
            turns.append(
                _turn_record(
                    post_start_ply=post_start_ply,
                    start_logical_ply=start_logical_ply,
                    mover_color=mover,
                    actor=actor,
                    board_after=board,
                    before_history=before_history,
                    applied=applied,
                    candidate_malom_delta=candidate_delta,
                )
            )
            clock.require_within_budget()
            progress_callback("game", post_start_ply)

            if post_start_ply == HORIZON_POST_START_LOGICAL_PLIES:
                if not applied.state.terminal:
                    horizon_survival = True
                    horizon_snapshot = _snapshot_at_horizon(
                        board=board,
                        state=applied.state,
                        candidate_color=candidate_color,
                        malom=policy.malom,
                        start_logical_ply=start_logical_ply,
                    )
            elif (
                post_start_ply > HORIZON_POST_START_LOGICAL_PLIES
                and horizon_snapshot is None
            ):
                raise RetainedPhaseProcessError("relative-horizon snapshot was skipped")
            if applied.state.terminal:
                break
        else:
            safety_cap = True

        final_state = game.state.portable_record()
        if safety_cap:
            if game.state.terminal:
                raise RetainedPhaseProcessError("terminal state reached at safety cap")
            winner = None
            score = None
            outcome_reason = "safety_cap_incomplete"
            termination_class = "safety_cap_incomplete"
        else:
            if not game.state.terminal or game.state.outcome_reason == "ongoing":
                raise RetainedPhaseProcessInvalid(
                    "game stopped without a rules terminal or safety cap"
                )
            winner_name = game.state.winner
            winner = {None: None, "white": "W", "black": "B"}.get(winner_name)
            if winner_name not in {None, "white", "black"}:
                raise RetainedPhaseProcessError("Sanmill winner value is unknown")
            score = (
                0.5 if winner is None else (1.0 if winner == candidate_color else 0.0)
            )
            outcome_reason = str(game.state.outcome_reason)
            termination_class = "rules_terminal"

        active = clock.require_within_budget()
        return {
            "schema_version": game_schema,
            "spec_identity": spec["spec_identity"],
            "ordinal": schedule_item["ordinal"],
            "unit_index": schedule_item["unit_index"],
            "game_id": schedule_item["game_id"],
            "match_key": schedule_item["match_key"],
            "candidate_id": schedule_item["candidate_id"],
            "candidate_color": candidate_color,
            "start_id": schedule_item["start_id"],
            "phase": schedule_item["phase"],
            "start": start,
            "ongoing_after_post_start_logical_ply_108": horizon_survival,
            "post_start_ply_108_snapshot": horizon_snapshot,
            "post_start_logical_plies": len(turns),
            "total_logical_plies": start_logical_ply + len(turns),
            "termination_class": termination_class,
            "outcome_reason": outcome_reason,
            "winner": winner,
            "candidate_score": score,
            "final_state": final_state,
            "history_process": _history_process(start, horizon_snapshot, final_state),
            "turns": turns,
            "candidate_malom": _candidate_malom_summary(turns),
            "sanmill_search": _search_summary(turns),
            "game_elapsed_seconds": round(active - game_started, 6),
            "cumulative_active_seconds": round(active, 6),
            "complete_diagnostic": True,
            "previous_record_sha256": previous_record_sha256,
        }


def _expected_mover(start_turn: str, post_start_ply: int) -> str:
    if start_turn not in {"W", "B"}:
        raise RetainedPhaseProcessError("start turn differs")
    if post_start_ply % 2 == 1:
        return start_turn
    return "B" if start_turn == "W" else "W"


def _validate_game_record(
    spec: Mapping[str, Any],
    record: Mapping[str, Any],
    ordinal: int,
    previous_hash: str | None,
    *,
    expected_games: int = EXPECTED_GAMES,
    game_schema: str = GAME_SCHEMA,
) -> None:
    schedule = spec.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != expected_games:
        raise RetainedPhaseProcessError("runtime schedule differs")
    if ordinal >= len(schedule) or record.get("ordinal") != ordinal:
        raise RetainedPhaseProcessError("phase-process ordinal differs")
    expected = schedule[ordinal]
    for field in (
        "game_id",
        "match_key",
        "candidate_id",
        "candidate_color",
        "start_id",
        "phase",
        "unit_index",
    ):
        if record.get(field) != expected.get(field):
            raise RetainedPhaseProcessError(f"phase-process {field} differs")
    if record.get("schema_version") != game_schema:
        raise RetainedPhaseProcessError("phase-process game schema differs")
    if record.get("spec_identity") != spec.get("spec_identity"):
        raise RetainedPhaseProcessError("phase-process spec identity differs")
    if record.get("previous_record_sha256") != previous_hash:
        raise RetainedPhaseProcessError("phase-process ledger chain differs")
    if record.get("complete_diagnostic") is not True:
        raise RetainedPhaseProcessError("phase-process game is incomplete")

    start = record.get("start")
    if not isinstance(start, Mapping):
        raise RetainedPhaseProcessError("phase-process start evidence is absent")
    if (
        start.get("start_record_identity") != expected["start_record_identity"]
        or start.get("observed_history_sha256")
        != expected["expected_start_history_sha256"]
        or start.get("logical_ply_count") != expected["start_logical_ply"]
    ):
        raise RetainedPhaseProcessError("phase-process start binding differs")
    start_ply = int(start["logical_ply_count"])
    plies = record.get("post_start_logical_plies")
    if not isinstance(plies, int) or not 1 <= plies <= MAX_POST_START_LOGICAL_PLIES:
        raise RetainedPhaseProcessError("phase-process game length differs")
    if record.get("total_logical_plies") != start_ply + plies:
        raise RetainedPhaseProcessError("phase-process total length differs")

    survival = record.get("ongoing_after_post_start_logical_ply_108")
    snapshot = record.get("post_start_ply_108_snapshot")
    if not isinstance(survival, bool):
        raise RetainedPhaseProcessError("relative-horizon survival differs")
    if survival:
        if not isinstance(snapshot, Mapping):
            raise RetainedPhaseProcessError("surviving game lacks horizon snapshot")
        if (
            snapshot.get("post_start_logical_ply") != HORIZON_POST_START_LOGICAL_PLIES
            or snapshot.get("absolute_logical_ply")
            != start_ply + HORIZON_POST_START_LOGICAL_PLIES
            or plies <= HORIZON_POST_START_LOGICAL_PLIES
        ):
            raise RetainedPhaseProcessError("relative-horizon snapshot differs")
    elif snapshot is not None:
        raise RetainedPhaseProcessError("non-surviving game has horizon snapshot")
    if plies > HORIZON_POST_START_LOGICAL_PLIES and not survival:
        raise RetainedPhaseProcessError("long game omitted relative-horizon survival")

    turns = record.get("turns")
    if not isinstance(turns, list) or len(turns) != plies:
        raise RetainedPhaseProcessError("phase-process turn ledger differs")
    previous = start.get("observed_history_sha256")
    candidate_color = str(record["candidate_color"])
    start_turn = str(expected["start_turn"])
    for index, turn in enumerate(turns, 1):
        if not isinstance(turn, Mapping):
            raise RetainedPhaseProcessError("phase-process turn is invalid")
        mover = _expected_mover(start_turn, index)
        expected_actor = "candidate" if mover == candidate_color else "sanmill"
        if turn.get("post_start_logical_ply") != index:
            raise RetainedPhaseProcessError("phase-process turn order differs")
        if turn.get("absolute_logical_ply") != start_ply + index:
            raise RetainedPhaseProcessError("phase-process absolute ply differs")
        if turn.get("mover_color") != mover or turn.get("actor") != expected_actor:
            raise RetainedPhaseProcessError("phase-process actor order differs")
        if turn.get("before_history_sha256") != previous:
            raise RetainedPhaseProcessError("phase-process history chain differs")
        previous = turn.get("after_history_sha256")
        if not isinstance(previous, str) or len(previous) != 64:
            raise RetainedPhaseProcessError("phase-process history hash differs")
        actions = turn.get("actions")
        move = turn.get("move")
        if not isinstance(actions, list) or not isinstance(move, Mapping):
            raise RetainedPhaseProcessError("phase-process move evidence differs")
        if actions != list(nmm_move_actions(move)):
            raise RetainedPhaseProcessError("phase-process actions differ")
        delta = turn.get("candidate_malom_delta")
        if expected_actor == "candidate":
            if delta is not None and delta not in (0.0, -1.0, -2.0):
                raise RetainedPhaseProcessError("candidate Malom delta differs")
            if turn.get("search") is not None:
                raise RetainedPhaseProcessError("candidate turn has search evidence")
        elif delta is not None or not isinstance(turn.get("search"), Mapping):
            raise RetainedPhaseProcessError("Sanmill turn evidence differs")
        is_last = index == plies
        if record.get("termination_class") == "rules_terminal":
            if bool(turn.get("terminal")) is not is_last:
                raise RetainedPhaseProcessError("terminal turn placement differs")
        elif turn.get("terminal"):
            raise RetainedPhaseProcessError("safety-cap game has terminal turn")

    final_state = record.get("final_state")
    if (
        not isinstance(final_state, Mapping)
        or final_state.get("history_sha256") != previous
    ):
        raise RetainedPhaseProcessError("phase-process final history differs")
    if record.get("candidate_malom") != _candidate_malom_summary(turns):
        raise RetainedPhaseProcessError("candidate Malom summary differs")
    if record.get("sanmill_search") != _search_summary(turns):
        raise RetainedPhaseProcessError("Sanmill search summary differs")
    if record.get("history_process") != _history_process(start, snapshot, final_state):
        raise RetainedPhaseProcessError("history-process summary differs")

    termination = record.get("termination_class")
    if termination == "rules_terminal":
        if final_state.get("terminal") is not True or record.get(
            "candidate_score"
        ) not in (0.0, 0.5, 1.0):
            raise RetainedPhaseProcessError("rules-terminal result differs")
    elif termination == "safety_cap_incomplete":
        if (
            plies != MAX_POST_START_LOGICAL_PLIES
            or final_state.get("terminal") is not False
            or record.get("winner") is not None
            or record.get("candidate_score") is not None
        ):
            raise RetainedPhaseProcessError("safety-cap disposition differs")
    else:
        raise RetainedPhaseProcessError("phase-process termination differs")


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
    with target.open("xb" if must_create else "ab") as handle:
        handle.write(canonical_json_bytes(wrapper) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record_hash


def load_game_ledger(
    spec: Mapping[str, Any],
    path: str | Path,
    *,
    expected_games: int = EXPECTED_GAMES,
    game_schema: str = GAME_SCHEMA,
) -> tuple[list[dict[str, Any]], str | None]:
    target = Path(path)
    if not target.exists():
        return [], None
    records = []
    previous: str | None = None
    with target.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith(b"\n") or raw.endswith(b"\r\n"):
                raise RetainedPhaseProcessError(
                    f"phase-process ledger line {line_number} is not LF-framed"
                )
            try:
                wrapper = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RetainedPhaseProcessError(
                    f"phase-process ledger line {line_number} is invalid JSON"
                ) from exc
            if canonical_json_bytes(wrapper) + b"\n" != raw:
                raise RetainedPhaseProcessError(
                    f"phase-process ledger line {line_number} is not canonical"
                )
            if not isinstance(wrapper, dict) or set(wrapper) != {
                "record",
                "record_sha256",
            }:
                raise RetainedPhaseProcessError("phase-process ledger wrapper differs")
            record = wrapper["record"]
            if not isinstance(record, dict) or wrapper[
                "record_sha256"
            ] != canonical_sha256(record):
                raise RetainedPhaseProcessError("phase-process ledger hash differs")
            _validate_game_record(
                spec,
                record,
                len(records),
                previous,
                expected_games=expected_games,
                game_schema=game_schema,
            )
            previous = str(wrapper["record_sha256"])
            records.append(record)
    if len(records) > expected_games:
        raise RetainedPhaseProcessError("phase-process ledger has too many games")
    return records, previous


def _numeric_summary(values: Sequence[int | float]) -> dict[str, Any]:
    numbers = list(values)
    return {
        "support": len(numbers),
        "mean": sum(numbers) / len(numbers) if numbers else None,
        "median": _quantile(numbers, 0.5),
        "min": min(numbers) if numbers else None,
        "max": max(numbers) if numbers else None,
    }


def _candidate_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    games = len(records)
    survival = sum(
        bool(record["ongoing_after_post_start_logical_ply_108"]) for record in records
    )
    post_lengths = [int(record["post_start_logical_plies"]) for record in records]
    total_lengths = [int(record["total_logical_plies"]) for record in records]
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
        record["post_start_ply_108_snapshot"]
        for record in records
        if record["post_start_ply_108_snapshot"] is not None
    ]
    malom_wdl = Counter(
        snapshot["malom_theoretical"]["candidate_perspective_wdl"] or "unqueryable"
        for snapshot in snapshots
    )
    starts = [record["history_process"]["start"] for record in records]
    horizons = [
        record["history_process"]["horizon"]
        for record in records
        if record["history_process"]["horizon"] is not None
    ]
    finals = [record["history_process"]["final"] for record in records]
    return {
        "games": games,
        "horizon_108_post_start": {
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
            "post_start": _numeric_summary(post_lengths),
            "total": _numeric_summary(total_lengths),
            "p90_post_start": _quantile(post_lengths, 0.9),
            "p90_total": _quantile(total_lengths, 0.9),
        },
        "history_process": {
            "start_no_capture": _numeric_summary(
                [int(item["no_capture_count"]) for item in starts]
            ),
            "horizon_no_capture": _numeric_summary(
                [int(item["no_capture_count"]) for item in horizons]
            ),
            "horizon_minus_start_no_capture": _numeric_summary(
                [
                    int(record["history_process"]["horizon"]["no_capture_count"])
                    - int(record["history_process"]["start"]["no_capture_count"])
                    for record in records
                    if record["history_process"]["horizon"] is not None
                ]
            ),
            "final_no_capture": _numeric_summary(
                [int(item["no_capture_count"]) for item in finals]
            ),
            "start_repetition_current": _numeric_summary(
                [int(item["repetition_current_count"]) for item in starts]
            ),
            "horizon_repetition_current": _numeric_summary(
                [int(item["repetition_current_count"]) for item in horizons]
            ),
            "final_repetition_current": _numeric_summary(
                [int(item["repetition_current_count"]) for item in finals]
            ),
        },
        "malom_at_horizon_candidate_perspective": {
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
        by_unit.setdefault(str(record["match_key"]), {})[
            str(record["candidate_id"])
        ] = record
    complete_units = [
        unit for unit in by_unit.values() if set(unit) == set(EXPECTED_CANDIDATES)
    ]
    colour_differences: dict[tuple[str, str], float] = {}
    length_differences = []
    preserving_differences = []
    for unit in complete_units:
        v3 = unit[EXPECTED_CANDIDATES[0]]
        v4 = unit[EXPECTED_CANDIDATES[1]]
        key = (str(v3["start_id"]), str(v3["candidate_color"]))
        colour_differences[key] = float(
            v4["ongoing_after_post_start_logical_ply_108"]
        ) - float(v3["ongoing_after_post_start_logical_ply_108"])
        length_differences.append(
            (int(v4["post_start_logical_plies"]) - int(v3["post_start_logical_plies"]))
            / MAX_POST_START_LOGICAL_PLIES
        )
        v3_rate = v3["candidate_malom"]["preserving_rate_given_queryable"]
        v4_rate = v4["candidate_malom"]["preserving_rate_given_queryable"]
        if v3_rate is not None and v4_rate is not None:
            preserving_differences.append(float(v4_rate) - float(v3_rate))

    starts = sorted({key[0] for key in colour_differences})
    start_differences = [
        (colour_differences[(start, "W")] + colour_differences[(start, "B")]) / 2.0
        for start in starts
        if (start, "W") in colour_differences and (start, "B") in colour_differences
    ]
    primary = _interval(start_differences)
    complete = len(start_differences) == EXPECTED_STARTS
    if not complete:
        decision = "pending"
    elif primary["half_width"] is None or primary["half_width"] > (
        MAX_PRIMARY_HALF_WIDTH
    ):
        decision = "inconclusive_precision"
    elif primary["interval"][0] > 0:
        decision = "v4_higher_108_post_start_ply_survival"
    elif primary["interval"][1] < 0:
        decision = "v3_higher_108_post_start_ply_survival"
    else:
        decision = "inconclusive"
    distribution = Counter(start_differences)
    return {
        "matched_colour_units_complete": len(complete_units),
        "matched_colour_units_expected": EXPECTED_MATCHED_COLOUR_UNITS,
        "start_units_complete": len(start_differences),
        "start_units_expected": EXPECTED_STARTS,
        "primary_start_clustered_108_ply_survival_v4_minus_v3": {
            **primary,
            "decision": decision,
            "maximum_half_width": MAX_PRIMARY_HALF_WIDTH,
            "precision_adequate": (
                primary["half_width"] is not None
                and primary["half_width"] <= MAX_PRIMARY_HALF_WIDTH
            ),
            "distribution": {
                str(value): distribution[value] for value in sorted(distribution)
            },
            "interpretation": (
                "fixed project-visible phase-corpus engineering interval; "
                "process generalization only, not strength or causality"
            ),
        },
        "colour_unit_108_ply_survival_v4_minus_v3": _interval(
            list(colour_differences.values())
        ),
        "restricted_length_v4_minus_v3": _interval(length_differences),
        "per_game_preserving_rate_v4_minus_v3": _interval(preserving_differences),
    }


def summarize_records(
    spec: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    tail: str | None,
) -> dict[str, Any]:
    """Build the canonical partial or complete phase-process report."""
    grouped = {
        candidate_id: [
            record for record in records if record["candidate_id"] == candidate_id
        ]
        for candidate_id in EXPECTED_CANDIDATES
    }
    phases = sorted({str(record["phase"]) for record in records})
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
        "by_phase": {
            phase: {
                candidate_id: _candidate_summary(
                    [
                        record
                        for record in grouped[candidate_id]
                        if record["phase"] == phase
                    ]
                )
                for candidate_id in EXPECTED_CANDIDATES
            }
            for phase in phases
        },
        "claim_boundary": {
            "corpus_previously_project_visible": True,
            "held_out": False,
            "playing_strength_claim": False,
            "refresh_causal_claim": False,
            "promotion_or_publication": False,
            "survival_is_draw": False,
            "malom_is_history_aware": False,
            "safety_cap_is_draw": False,
        },
    }
    return {**body, "result_identity": canonical_sha256(body)}


def recompute_report(
    spec: Mapping[str, Any],
    ledger_path: str | Path,
) -> dict[str, Any]:
    records, tail = load_game_ledger(spec, ledger_path)
    return summarize_records(spec, records, tail)


def load_corpus_records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Validate and return the frozen ordered phase-process records."""
    validate_retained_phase_process_corpus(payload)
    records = payload.get("records")
    if not isinstance(records, list):
        raise RetainedPhaseProcessError("phase-process records are absent")
    return records
