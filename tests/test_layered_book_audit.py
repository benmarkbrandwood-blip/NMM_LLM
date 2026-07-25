from __future__ import annotations

from learned_ai.evaluation.layered_book_audit import (
    _expand_named_variation,
)


def test_named_variation_expands_omitted_capture_without_calling_fallback() -> None:
    opening = {
        "id": "two-capture-branches",
        "name": "Omitted capture branch",
        "family": "test",
        "lineMoves": [
            "a7",
            "a1",
            "d7",
            "d1",
            "g7",
            "g1",
            "a4",
            "b2",
            "d6",
            "b4",
            "f6",
            "f2",
        ],
    }

    result = _expand_named_variation(opening)

    assert result["status"] == "complete"
    assert result["failed_at_logical_ply"] is None
    histories = result["expanded_histories"]
    assert len(histories) == 2
    assert {history[4][1] for history, _board in histories} == {
        "xa1",
        "xd1",
    }
    assert all(len(history) == 12 for history, _board in histories)


def test_named_variation_reports_short_and_unreplayable_lines() -> None:
    short = _expand_named_variation(
        {
            "id": "short",
            "lineMoves": ["a7", "d7"],
        }
    )
    invalid = _expand_named_variation(
        {
            "id": "invalid",
            "lineMoves": ["a7"] * 12,
        }
    )

    assert short["status"] == "shorter_than_12"
    assert short["expanded_histories"] == ()
    assert invalid["status"] == "unreplayable"
    assert invalid["failed_at_logical_ply"] == 2
    assert invalid["expanded_histories"] == ()
