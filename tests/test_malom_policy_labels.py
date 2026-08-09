"""Fail-closed exact-WDL labels for the Generalist policy auxiliary."""

from __future__ import annotations

import numpy as np
import pytest

from learned_ai.training.malom_policy_labels import (
    MalomPolicyLabelError,
    label_malom_preserving_actions,
)


LEGAL_MOVES = [
    {"from": None, "to": "a1", "capture": None},
    {"from": None, "to": "d1", "capture": None},
    {"from": None, "to": "g1", "capture": None},
]


class FakeMalom:
    def __init__(self, root: str, rows: list[dict]) -> None:
        self.root = root
        self.rows = rows

    def is_available(self) -> bool:
        return True

    def query_state(self, board) -> str:
        return self.root

    def query_all_moves(self, board, player) -> list[dict]:
        return self.rows


def _rows(*wdl: str) -> list[dict]:
    return [
        {"move": dict(move), "wdl": outcome}
        for move, outcome in zip(LEGAL_MOVES, wdl, strict=True)
    ]


def test_labels_are_reordered_to_match_policy_actions() -> None:
    rows = _rows("draw", "loss", "draw")
    labels = label_malom_preserving_actions(
        FakeMalom("D", [rows[2], rows[0], rows[1]]),
        board=object(),
        player="W",
        legal_moves=LEGAL_MOVES,
    )

    assert labels.root_wdl == "draw"
    assert labels.qualities.tolist() == [0.0, -1.0, 0.0]
    assert labels.preserving_mask.dtype == np.bool_
    assert labels.preserving_mask.tolist() == [True, False, True]
    assert labels.preserving_count == 2
    assert labels.downgrading_count == 1


def test_complete_action_identity_includes_compulsory_capture() -> None:
    legal = [
        {"from": "d6", "to": "d5", "capture": "c3"},
        {"from": "d6", "to": "d5", "capture": "f2"},
    ]
    rows = [
        {"move": dict(legal[1]), "wdl": "loss"},
        {"move": dict(legal[0]), "wdl": "win"},
    ]

    labels = label_malom_preserving_actions(
        FakeMalom("W", rows),
        board=object(),
        player="W",
        legal_moves=legal,
    )

    assert labels.preserving_mask.tolist() == [True, False]
    assert labels.qualities.tolist() == [0.0, -2.0]


@pytest.mark.parametrize(
    ("malom", "legal_moves", "match"),
    [
        (FakeMalom("D", []), LEGAL_MOVES, "action set"),
        (
            FakeMalom("D", _rows("draw", "draw", "unknown")),
            LEGAL_MOVES,
            "unknown WDL",
        ),
        (
            FakeMalom("D", _rows("win", "draw", "loss")),
            LEGAL_MOVES,
            "positive WDL delta",
        ),
        (
            FakeMalom("W", _rows("draw", "draw", "loss")),
            LEGAL_MOVES,
            "no preserving action",
        ),
        (
            FakeMalom("D", _rows("draw", "draw", "loss")[:2] * 2),
            LEGAL_MOVES,
            "duplicate Malom action",
        ),
    ],
)
def test_invalid_or_incomplete_exact_labels_fail_closed(
    malom,
    legal_moves,
    match: str,
) -> None:
    with pytest.raises(MalomPolicyLabelError, match=match):
        label_malom_preserving_actions(
            malom,
            board=object(),
            player="W",
            legal_moves=legal_moves,
        )


def test_unavailable_teacher_fails_closed() -> None:
    malom = FakeMalom("D", _rows("draw", "draw", "loss"))
    malom.is_available = lambda: False

    with pytest.raises(MalomPolicyLabelError, match="not available"):
        label_malom_preserving_actions(
            malom,
            board=object(),
            player="W",
            legal_moves=LEGAL_MOVES,
        )
