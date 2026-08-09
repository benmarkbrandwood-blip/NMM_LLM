from __future__ import annotations

import pytest

from ai.malom_db import OracleMoveValue
from learned_ai.evaluation.heldout_evaluation import HeldoutEvaluationError
from learned_ai.evaluation.heldout_oracle_alternative_audit import (
    analyze_oracle_alternatives,
)


def _value(key1: int, key2: int, outcome: str) -> OracleMoveValue:
    return OracleMoveValue(
        key1=key1,
        key2=key2,
        sector_value=0,
        absolute_key1=key1,
        perspective="W",
        sector=(6, 6, 0, 0),
        outcome=outcome,
        source="test",
    )


def _row(move: dict, wdl: str, value: OracleMoveValue | None) -> dict:
    return {"move": move, "wdl": wdl, "dtm": None, "oracle_value": value}


def test_same_primary_preserving_capture_identifies_wrong_capture_target() -> None:
    chosen = {"from": "a7", "to": "d7", "capture": "b6"}
    better_capture = {"from": "a7", "to": "d7", "capture": "c5"}
    other = {"from": "a4", "to": "a7", "capture": None}

    result = analyze_oracle_alternatives(
        chosen_move=chosen,
        before_candidate_wdl="D",
        after_candidate_wdl="L",
        results=[
            _row(chosen, "loss", _value(-1, 4, "L")),
            _row(better_capture, "draw", _value(0, 8, "D")),
            _row(other, "draw", _value(0, 3, "D")),
        ],
    )

    assert result["classification"] == "wrong_capture_target"
    assert result["same_primary_preserving_count"] == 1
    assert result["preserving_alternatives_count"] == 2
    assert result["strictly_better_alternatives_count"] == 2
    assert result["chosen_is_full_oracle_best"] is False


def test_no_same_primary_preserver_identifies_primary_or_timing_error() -> None:
    chosen = {"from": None, "to": "d7", "capture": "b6"}
    other_capture = {"from": None, "to": "d7", "capture": "c5"}
    preserving = {"from": None, "to": "a7", "capture": None}

    result = analyze_oracle_alternatives(
        chosen_move=chosen,
        before_candidate_wdl="D",
        after_candidate_wdl="L",
        results=[
            _row(chosen, "loss", _value(-2, 4, "L")),
            _row(other_capture, "loss", _value(-1, 2, "L")),
            _row(preserving, "draw", _value(0, 7, "D")),
        ],
    )

    assert result["classification"] == "primary_action_or_mill_timing"
    assert result["same_primary_preserving_count"] == 0
    assert result["preserving_alternatives_count"] == 1


def test_non_capture_downgrade_identifies_primary_action() -> None:
    chosen = {"from": "a7", "to": "d7", "capture": None}
    preserving = {"from": "a4", "to": "a7", "capture": None}

    result = analyze_oracle_alternatives(
        chosen_move=chosen,
        before_candidate_wdl="W",
        after_candidate_wdl="D",
        results=[
            _row(chosen, "draw", _value(0, 4, "D")),
            _row(preserving, "win", _value(1, 2, "W")),
        ],
    )

    assert result["classification"] == "primary_action"
    assert result["full_oracle_best"][0]["move"] == preserving


def test_unknown_or_coarse_only_alternative_fails_closed() -> None:
    chosen = {"from": "a7", "to": "d7", "capture": None}

    with pytest.raises(
        HeldoutEvaluationError,
        match="lacks full Malom value",
    ):
        analyze_oracle_alternatives(
            chosen_move=chosen,
            before_candidate_wdl="D",
            after_candidate_wdl="L",
            results=[_row(chosen, "loss", None)],
        )


def test_coarse_and_full_outcome_mismatch_fails_closed() -> None:
    chosen = {"from": "a7", "to": "d7", "capture": None}

    with pytest.raises(
        HeldoutEvaluationError,
        match="coarse and full Malom outcomes differ",
    ):
        analyze_oracle_alternatives(
            chosen_move=chosen,
            before_candidate_wdl="D",
            after_candidate_wdl="L",
            results=[_row(chosen, "loss", _value(0, 1, "D"))],
        )
