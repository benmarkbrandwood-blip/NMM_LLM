"""Candidate-blind SpecialistDB coverage corpus construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from game.board import BoardState
from game.rules import get_all_legal_moves, get_game_phase
from learned_ai.training.run_contract import canonical_sha256
from learned_ai.training.sanmill_referee import nmm_move_actions


class SpecialistCoverageCorpusError(ValueError):
    """Raised when a frozen source history cannot be replayed exactly."""


def _phase_name(board: BoardState) -> str:
    return {
        "place": "placement",
        "move": "movement",
        "fly": "flying",
    }[get_game_phase(board, board.turn)]


def _matching_move(board: BoardState, actions: Sequence[str]) -> dict[str, Any]:
    matches = [
        dict(move)
        for move in get_all_legal_moves(board)
        if nmm_move_actions(move) == tuple(actions)
    ]
    if len(matches) != 1:
        raise SpecialistCoverageCorpusError(
            "source actions do not select exactly one legal logical turn"
        )
    return matches[0]


def replay_unique_prefix_states(
    records: Sequence[Mapping[str, Any]],
    *,
    required_logical_plies: int = 12,
) -> list[dict[str, Any]]:
    """Replay frozen histories and return exact-FEN-deduplicated pre-move states."""
    if not records:
        raise SpecialistCoverageCorpusError("source corpus has no records")
    states: dict[str, dict[str, Any]] = {}
    for record in records:
        corpus_id = record.get("corpus_id")
        stratum = record.get("stratum")
        execution = record.get("execution_record")
        if not isinstance(corpus_id, str) or not isinstance(stratum, str):
            raise SpecialistCoverageCorpusError("source record identity is invalid")
        if not isinstance(execution, Mapping):
            raise SpecialistCoverageCorpusError("source record lacks execution data")
        steps = execution.get("steps")
        if not isinstance(steps, list) or len(steps) != required_logical_plies:
            raise SpecialistCoverageCorpusError(
                "source record has the wrong logical-ply count"
            )
        flattened: list[str] = []
        board = BoardState.new_game()
        for logical_ply, step in enumerate(steps):
            if not isinstance(step, Mapping) or step.get("logical_ply") != logical_ply:
                raise SpecialistCoverageCorpusError("source step order is invalid")
            actions = step.get("action_tokens")
            if not isinstance(actions, list) or any(
                not isinstance(action, str) for action in actions
            ):
                raise SpecialistCoverageCorpusError("source action tokens are invalid")
            fen = board.to_fen_string()
            state = states.setdefault(
                fen,
                {
                    "board": board,
                    "fen": fen,
                    "phase": _phase_name(board),
                    "minimum_logical_ply": logical_ply,
                    "references": [],
                    "strata": set(),
                },
            )
            state["minimum_logical_ply"] = min(
                int(state["minimum_logical_ply"]), logical_ply
            )
            state["references"].append(
                {
                    "corpus_id": corpus_id,
                    "logical_ply": logical_ply,
                    "record_identity": record.get("record_identity"),
                    "source_history_id": record.get("source_history_id"),
                    "stratum": stratum,
                }
            )
            state["strata"].add(stratum)
            move = _matching_move(board, actions)
            flattened.extend(actions)
            board = board.apply_move(move)

        expected_actions = execution.get("action_tokens")
        if expected_actions != flattened:
            raise SpecialistCoverageCorpusError(
                "flattened source actions differ from the execution record"
            )
        final = execution.get("final")
        if not isinstance(final, Mapping) or board.to_fen_string() != final.get(
            "nmm_fen"
        ):
            raise SpecialistCoverageCorpusError("source final FEN differs after replay")

    result = list(states.values())
    for state in result:
        state["references"] = sorted(
            state["references"],
            key=lambda item: (
                str(item["corpus_id"]),
                int(item["logical_ply"]),
            ),
        )
        state["strata"] = sorted(state["strata"])
    result.sort(key=lambda item: (int(item["minimum_logical_ply"]), item["fen"]))
    return result


def build_empirical_coverage_corpus(
    records: Sequence[Mapping[str, Any]],
    specialist_db: Any,
    *,
    min_samples: int = 3,
    required_logical_plies: int = 12,
) -> dict[str, Any]:
    """Select every unique source state with an empirical successor hit."""
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    states = replay_unique_prefix_states(
        records,
        required_logical_plies=required_logical_plies,
    )
    entries: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    by_ply: dict[int, Counter[str]] = defaultdict(Counter)
    for state in states:
        board = state["board"]
        actions: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        legal_moves = get_all_legal_moves(board)
        counts["states"] = 1
        counts["legal_actions"] = len(legal_moves)
        for move in legal_moves:
            evidence = specialist_db.query_wdl_evidence(
                board.apply_move(move),
                min_samples=min_samples,
            )
            theoretical = bool(
                evidence is not None and evidence.theoretical_wdl is not None
            )
            empirical = bool(
                evidence is not None and evidence.empirical_distribution is not None
            )
            samples = (
                0
                if evidence is None
                else sum(int(value) for value in evidence.empirical_counts)
            )
            if evidence is not None:
                counts["rows_present"] += 1
            if theoretical:
                counts["theoretical_actions"] += 1
            if empirical:
                counts["empirical_actions"] += 1
            actions.append(
                {
                    "action_tokens": list(nmm_move_actions(move)),
                    "empirical_available": empirical,
                    "empirical_samples": samples,
                    "resulting_fen": board.apply_move(move).to_fen_string(),
                    "theoretical_available": theoretical,
                }
            )
        selected = counts["empirical_actions"] >= 1
        counts["selected_states"] = int(selected)
        totals.update(counts)
        by_ply[int(state["minimum_logical_ply"])].update(counts)
        if not selected:
            continue
        entry_core = {
            "fen": state["fen"],
            "phase": state["phase"],
            "source_minimum_logical_ply": int(state["minimum_logical_ply"]),
            "source_references": state["references"],
            "source_strata": state["strata"],
            "specialist_db_coverage": {
                "empirical_actions": counts["empirical_actions"],
                "legal_actions": counts["legal_actions"],
                "rows_present": counts["rows_present"],
                "theoretical_actions": counts["theoretical_actions"],
            },
            "actions": actions,
        }
        entry = dict(entry_core)
        entry["entry_identity"] = canonical_sha256(entry_core)
        entries.append(entry)

    for index, entry in enumerate(entries, start=1):
        entry["index"] = index
    portable_entries = [
        {key: value for key, value in entry.items() if key != "actions"}
        | {"actions": entry["actions"]}
        for entry in entries
    ]
    return {
        "selection_contract": {
            "candidate_loaded": False,
            "deduplication_key": "exact_nmm_fen",
            "minimum_empirical_actions": 1,
            "specialist_db_min_samples": min_samples,
            "state_boundary": "before_each_source_logical_turn",
            "tie_or_cap_rule": "none_keep_all_eligible",
        },
        "source_summary": {
            "source_record_count": len(records),
            "unique_replayed_state_count": len(states),
            "coverage": dict(sorted(totals.items())),
            "coverage_by_minimum_logical_ply": {
                str(ply): dict(sorted(counter.items()))
                for ply, counter in sorted(by_ply.items())
            },
        },
        "entries": portable_entries,
        "entries_identity": canonical_sha256(portable_entries),
    }
