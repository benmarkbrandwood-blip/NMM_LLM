"""Focused tests for compact six-arm follow-up calculations."""

from __future__ import annotations

import pytest

from learned_ai.evaluation.mill_bonus_ablation_followup import (
    MillBonusAblationFollowupError,
    _aggregate,
    _integer_from_rate,
    _two_proportion_independent_actions,
)


def _row(*, steps: int, downgrade: int, mill: int, mill_downgrade: int) -> dict:
    return {
        "steps": steps,
        "malom_known_move_rate": 1.0,
        "malom_downgrade_move_rate": downgrade / steps,
        "formed_mill_malom_known_place": mill,
        "formed_mill_malom_known_move": 0,
        "formed_mill_malom_known_fly": 0,
        "formed_mill_malom_downgrade_place": mill_downgrade,
        "formed_mill_malom_downgrade_move": 0,
        "formed_mill_malom_downgrade_fly": 0,
        "formed_mill_malom_downgrade_count": mill_downgrade,
        "formed_mill_count": mill,
        "mill_bonus_awarded_total": 0.0,
    }


def test_aggregate_reconstructs_exact_action_and_mill_support() -> None:
    summary = _aggregate(
        [
            _row(steps=10, downgrade=2, mill=1, mill_downgrade=1),
            _row(steps=20, downgrade=1, mill=2, mill_downgrade=0),
        ]
    )

    assert summary["learner_actions"] == 30
    assert summary["known_actions"] == 30
    assert summary["downgrade_actions"] == 3
    assert summary["all_action_downgrade_rate"] == pytest.approx(0.1)
    assert summary["known_mill_actions"] == 3
    assert summary["downgrade_mill_actions"] == 1
    assert summary["formed_mills"] == 3
    assert summary["mill_action_downgrade_rate"] == pytest.approx(1 / 3)
    assert summary["games_with_mill_downgrade"] == 1


def test_integer_reconstruction_fails_closed_on_rounded_rate() -> None:
    with pytest.raises(
        MillBonusAblationFollowupError,
        match="integer count",
    ):
        _integer_from_rate(0.333, 10, field="rate")


@pytest.mark.parametrize(
    ("control", "treatment", "expected"),
    [
        (0.09, 0.04, 381),
        (0.08, 0.06, 2554),
        (0.08, 0.05, 1059),
        (0.08, 0.04, 553),
    ],
)
def test_optimistic_power_scenarios_are_stable(
    control: float,
    treatment: float,
    expected: int,
) -> None:
    assert _two_proportion_independent_actions(control, treatment) == expected
