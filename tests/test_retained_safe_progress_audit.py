from __future__ import annotations

from collections.abc import Mapping

import pytest

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.evaluation.retained_safe_progress_audit import (
    classify_candidate_turn,
)


class _AllDrawMalom:
    @staticmethod
    def query_state(_board: BoardState) -> str:
        return "D"

    @staticmethod
    def query_all_moves(board: BoardState, player: str) -> list[dict]:
        assert player == board.turn
        return [
            {"move": dict(move), "wdl": "draw", "dtm": None, "oracle_value": None}
            for move in get_all_legal_moves(board)
        ]


class _PartialMalom(_AllDrawMalom):
    @staticmethod
    def query_all_moves(board: BoardState, player: str) -> list[dict]:
        rows = _AllDrawMalom.query_all_moves(board, player)
        for row in rows:
            if row["move"]["capture"] is not None:
                row["wdl"] = "unknown"
        return rows


def _capture_board() -> BoardState:
    # This frozen diagnostic prefix gives White both capturing and non-capturing
    # complete legal actions.
    return BoardState.from_fen_string(".....B.BBW.WWBBB......WW|W|6|6")


def _pick(board: BoardState, *, capture: bool) -> Mapping[str, object]:
    return next(
        move
        for move in get_all_legal_moves(board)
        if (move["capture"] is not None) is capture
    )


def test_non_capture_is_a_missed_safe_capture_when_preserving_capture_exists() -> None:
    board = _capture_board()
    chosen = _pick(board, capture=False)
    result = classify_candidate_turn(
        board=board,
        chosen_move=chosen,
        seen_fens={board.to_fen_string()},
        malom=_AllDrawMalom(),
        recorded_delta=0.0,
    )
    assert result["safe_capture_opportunity"] is True
    assert result["chosen_capture"] is False
    assert result["missed_safe_capture"] is True
    assert result["capture_opportunity_unknown"] is False


def test_preserving_capture_selects_the_safe_progress_opportunity() -> None:
    board = _capture_board()
    chosen = _pick(board, capture=True)
    result = classify_candidate_turn(
        board=board,
        chosen_move=chosen,
        seen_fens={board.to_fen_string()},
        malom=_AllDrawMalom(),
        recorded_delta=0.0,
    )
    assert result["safe_capture_opportunity"] is True
    assert result["chosen_preserving_capture"] is True
    assert result["missed_safe_capture"] is False


def test_unknown_capture_values_do_not_invent_absent_opportunity() -> None:
    board = _capture_board()
    chosen = _pick(board, capture=False)
    result = classify_candidate_turn(
        board=board,
        chosen_move=chosen,
        seen_fens={board.to_fen_string()},
        malom=_PartialMalom(),
        recorded_delta=0.0,
    )
    assert result["safe_capture_opportunity"] is False
    assert result["capture_opportunity_unknown"] is True
    assert result["missed_safe_capture"] is False


def test_revisit_is_avoidable_when_a_preserving_novel_move_exists() -> None:
    board = _capture_board()
    chosen = _pick(board, capture=False)
    chosen_after = board.apply_move(dict(chosen)).to_fen_string()
    result = classify_candidate_turn(
        board=board,
        chosen_move=chosen,
        seen_fens={board.to_fen_string(), chosen_after},
        malom=_AllDrawMalom(),
        recorded_delta=0.0,
    )
    assert result["chosen_board_revisit"] is True
    assert result["safe_novel_opportunity"] is True
    assert result["avoidable_board_revisit"] is True


def test_recorded_malom_delta_must_replay() -> None:
    board = _capture_board()
    chosen = _pick(board, capture=False)
    with pytest.raises(RuntimeError, match="does not replay"):
        classify_candidate_turn(
            board=board,
            chosen_move=chosen,
            seen_fens={board.to_fen_string()},
            malom=_AllDrawMalom(),
            recorded_delta=-1.0,
        )
