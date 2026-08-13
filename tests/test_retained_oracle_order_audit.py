from __future__ import annotations

from collections.abc import Mapping

import pytest

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.evaluation.retained_oracle_order_audit import (
    EXPECTED_CANDIDATES,
    RetainedOracleOrderAuditError,
    _empty_counts,
    _paired_summary,
    _summarise_counts,
    classify_candidate_turn,
)
from scripts.run_retained_oracle_order_audit import DEFAULT_PLAN, load_audit_plan


def _key(move: Mapping[str, object]) -> tuple[object, object, object]:
    return move["from"], move["to"], move["capture"]


class _Value:
    def __init__(
        self,
        key: tuple[int, int],
        *,
        perspective: str,
        sector: tuple[int, int, int, int] = (6, 6, 6, 6),
    ) -> None:
        self._key = key
        self.sector = sector
        self.sector_value = 0
        self.perspective = perspective
        self.outcome = "D"

    def ordering_key(self) -> tuple[int, int]:
        return self._key


class _OrderMalom:
    def __init__(
        self,
        grades: Mapping[tuple[object, object, object], tuple[int, int]],
        *,
        missing: set[tuple[object, object, object]] | None = None,
        mixed_context: bool = False,
        labels: Mapping[tuple[object, object, object], str] | None = None,
    ) -> None:
        self.grades = grades
        self.missing = missing or set()
        self.mixed_context = mixed_context
        self.labels = labels or {}

    @staticmethod
    def query_state(_board: BoardState) -> str:
        return "D"

    def query_all_moves(self, board: BoardState, player: str) -> list[dict]:
        assert player == board.turn
        rows = []
        for index, move in enumerate(get_all_legal_moves(board)):
            key = _key(move)
            value = None
            if key not in self.missing:
                sector = (5, 5, 5, 5) if self.mixed_context and index == 0 else (6, 6, 6, 6)
                value = _Value(
                    self.grades.get(key, (0, 0)),
                    perspective=board.turn,
                    sector=sector,
                )
            rows.append(
                {
                    "move": dict(move),
                    "wdl": self.labels.get(key, "draw"),
                    "dtm": None,
                    "oracle_value": value,
                }
            )
        return rows


def _board() -> BoardState:
    return BoardState.from_fen_string(".....B.BBW.WWBBB......WW|W|6|6")


def _moves(board: BoardState) -> list[Mapping[str, object]]:
    return list(get_all_legal_moves(board))


def test_best_complete_grade_has_zero_regret() -> None:
    board = _board()
    chosen, inferior = _moves(board)[:2]
    result = classify_candidate_turn(
        board=board,
        chosen_move=chosen,
        malom=_OrderMalom({_key(chosen): (2, 0), _key(inferior): (1, 0)}),
        recorded_delta=0.0,
    )
    assert result["within_wdl_orderable"] is True
    assert result["full_order_choice_opportunity"] is True
    assert result["chosen_full_order_best"] is True
    assert result["normalised_ordinal_regret"] == 0.0


def test_worst_of_three_distinct_grades_has_unit_ordinal_regret() -> None:
    board = _board()
    chosen, middle, best = _moves(board)[:3]
    grades = {
        _key(chosen): (0, 0),
        _key(middle): (1, 0),
        _key(best): (2, 0),
    }
    result = classify_candidate_turn(
        board=board,
        chosen_move=chosen,
        malom=_OrderMalom(grades),
        recorded_delta=0.0,
    )
    assert result["chosen_full_order_best"] is False
    assert result["missed_full_order_best"] is True
    assert result["normalised_ordinal_regret"] == 1.0
    assert result["better_preserving_actions"] >= 2
    assert result["distinct_preserving_grades"] == 3


def test_middle_of_three_distinct_grades_has_half_ordinal_regret() -> None:
    board = _board()
    chosen, worse, best = _moves(board)[:3]
    grades = {
        _key(chosen): (1, 0),
        _key(worse): (0, 0),
        _key(best): (2, 0),
    }
    result = classify_candidate_turn(
        board=board,
        chosen_move=chosen,
        malom=_OrderMalom(grades),
        recorded_delta=0.0,
    )
    assert result["normalised_ordinal_regret"] == 0.5


def test_oracle_tie_is_not_an_ordering_opportunity() -> None:
    board = _board()
    chosen = _moves(board)[0]
    result = classify_candidate_turn(
        board=board,
        chosen_move=chosen,
        malom=_OrderMalom({}),
        recorded_delta=0.0,
    )
    assert result["within_wdl_orderable"] is True
    assert result["full_order_choice_opportunity"] is False
    assert result["chosen_full_order_best"] is True
    assert result["normalised_ordinal_regret"] == 0.0


def test_missing_complete_value_is_reported_as_not_orderable() -> None:
    board = _board()
    chosen, missing = _moves(board)[:2]
    result = classify_candidate_turn(
        board=board,
        chosen_move=chosen,
        malom=_OrderMalom({}, missing={_key(missing)}),
        recorded_delta=0.0,
    )
    assert result["chosen_coarse_preserving"] is True
    assert result["within_wdl_orderable"] is False
    assert result["normalised_ordinal_regret"] is None


def test_unknown_coarse_alternative_prevents_complete_ordering() -> None:
    board = _board()
    chosen, unknown = _moves(board)[:2]
    result = classify_candidate_turn(
        board=board,
        chosen_move=chosen,
        malom=_OrderMalom({}, labels={_key(unknown): "unknown"}),
        recorded_delta=0.0,
    )
    assert result["all_legal_actions_queryable"] is False
    assert result["within_wdl_orderable"] is False


def test_mixed_parent_contexts_fail_closed() -> None:
    board = _board()
    chosen = _moves(board)[0]
    with pytest.raises(
        RetainedOracleOrderAuditError, match="mixed parent contexts"
    ):
        classify_candidate_turn(
            board=board,
            chosen_move=chosen,
            malom=_OrderMalom({}, mixed_context=True),
            recorded_delta=0.0,
        )


def test_recorded_coarse_delta_must_replay() -> None:
    board = _board()
    chosen = _moves(board)[0]
    with pytest.raises(RetainedOracleOrderAuditError, match="does not replay"):
        classify_candidate_turn(
            board=board,
            chosen_move=chosen,
            malom=_OrderMalom({}),
            recorded_delta=-1.0,
        )


def test_coarse_downgrade_is_excluded_from_within_wdl_ordering() -> None:
    board = _board()
    chosen = _moves(board)[0]
    result = classify_candidate_turn(
        board=board,
        chosen_move=chosen,
        malom=_OrderMalom({}, labels={_key(chosen): "loss"}),
        recorded_delta=-1.0,
    )
    assert result["parent_queryable"] is True
    assert result["chosen_coarse_preserving"] is False
    assert result["within_wdl_orderable"] is False
    assert result["normalised_ordinal_regret"] is None


def _synthetic_summary(regret: float) -> dict:
    counts = _empty_counts()
    for field in (
        "candidate_turns",
        "parent_queryable_turns",
        "all_legal_actions_queryable_turns",
        "chosen_coarse_preserving_turns",
        "within_wdl_orderable_turns",
        "full_order_choice_opportunity_turns",
    ):
        counts[field] = 10
    counts["normalised_ordinal_regret_sum"] = regret * 10
    counts["opportunity_normalised_ordinal_regret_sum"] = regret * 10
    return _summarise_counts(counts)


def test_paired_decision_uses_per_game_normalised_regret_difference() -> None:
    games = []
    for unit in range(128):
        for candidate, regret in zip(EXPECTED_CANDIDATES, (0.1, 0.2), strict=True):
            summary = _synthetic_summary(regret)
            games.append(
                {
                    "match_key": f"unit-{unit}",
                    "candidate_id": candidate,
                    "outcome_reason": "draw_repetition",
                    "all_candidate_turns": summary,
                    "after_ply_120_candidate_turns": _summarise_counts(
                        _empty_counts()
                    ),
                }
            )
    paired = _paired_summary(games)
    primary = paired["primary_mean_normalised_ordinal_regret_v4_minus_v3"]
    assert primary["support"] == 128
    assert primary["mean"] == pytest.approx(0.1)
    assert primary["decision"] == "v4_higher_full_order_regret"


def test_frozen_plan_binds_prior_audit_and_zero_game_workload() -> None:
    plan = load_audit_plan(DEFAULT_PLAN)
    assert plan["plan_identity"] == (
        "95e1d5e6640765e14852b9dfc3f2793bf72ee583bc95fc0a3bd1512acb36d23d"
    )
    assert plan["safe_progress_source"]["result_identity"] == (
        "b60eaf6392d55e520b5a2a493ce7dd8961c05e811a7fd3cbb5375735fe312fea"
    )
    assert plan["source"]["files"]["ledger"]["sha256"] == (
        "c064f29d77cedd42a9ef405ec44dbbda045b47be31092e952568cecb5d49b562"
    )
    assert plan["workload"] == {
        "new_games": 0,
        "model_updates": 0,
        "database_writes": 0,
        "checkpoint_writes": 0,
    }
