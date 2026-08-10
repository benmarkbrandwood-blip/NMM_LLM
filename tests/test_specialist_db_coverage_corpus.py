"""Tests for candidate-blind SpecialistDB coverage-corpus construction."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.training.sanmill_referee import nmm_move_actions
from learned_ai.validation.specialist_db_coverage_corpus import (
    SpecialistCoverageCorpusError,
    build_empirical_coverage_corpus,
    replay_unique_prefix_states,
)


def _source_record(corpus_id: str) -> dict:
    board = BoardState.new_game()
    steps = []
    flattened = []
    for logical_ply in range(12):
        move = get_all_legal_moves(board)[0]
        actions = list(nmm_move_actions(move))
        steps.append({"logical_ply": logical_ply, "action_tokens": actions})
        flattened.extend(actions)
        board = board.apply_move(move)
    return {
        "corpus_id": corpus_id,
        "record_identity": f"record-{corpus_id}",
        "source_history_id": f"history-{corpus_id}",
        "stratum": "book",
        "execution_record": {
            "action_tokens": flattened,
            "steps": steps,
            "final": {"nmm_fen": board.to_fen_string()},
        },
    }


@dataclass
class _Evidence:
    empirical_counts: tuple[int, int, int]
    empirical_distribution: tuple[float, float, float] | None
    theoretical_wdl: object | None = None


@dataclass
class _Database:
    evidence: _Evidence | None

    def query_wdl_evidence(self, _board, min_samples: int):
        assert min_samples == 3
        return self.evidence


def test_replay_deduplicates_exact_source_states() -> None:
    states = replay_unique_prefix_states([_source_record("one"), _source_record("two")])

    assert len(states) == 12
    assert all(len(state["references"]) == 2 for state in states)
    assert states[0]["minimum_logical_ply"] == 0
    assert states[-1]["minimum_logical_ply"] == 11


def test_builder_keeps_all_states_with_empirical_successors() -> None:
    result = build_empirical_coverage_corpus(
        [_source_record("one")],
        _Database(_Evidence((3, 0, 0), (1.0, 0.0, 0.0))),
    )

    assert len(result["entries"]) == 12
    assert result["selection_contract"]["candidate_loaded"] is False
    assert result["selection_contract"]["tie_or_cap_rule"] == ("none_keep_all_eligible")
    assert all(
        entry["specialist_db_coverage"]["empirical_actions"] >= 1
        for entry in result["entries"]
    )


def test_builder_rejects_rows_without_empirical_support() -> None:
    result = build_empirical_coverage_corpus(
        [_source_record("one")],
        _Database(_Evidence((1, 0, 0), None)),
    )

    assert result["entries"] == []
    assert result["source_summary"]["coverage"]["selected_states"] == 0


def test_replay_fails_closed_on_final_fen_drift() -> None:
    record = _source_record("one")
    record["execution_record"]["final"]["nmm_fen"] = (
        BoardState.new_game().to_fen_string()
    )

    with pytest.raises(SpecialistCoverageCorpusError, match="final FEN"):
        replay_unique_prefix_states([record])
