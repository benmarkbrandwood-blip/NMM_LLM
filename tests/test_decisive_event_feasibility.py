from __future__ import annotations

import pytest

from tools.estimate_decisive_event_feasibility import (
    build_report,
    minimum_games_for_event_yield,
    probability_at_least,
    wilson_interval,
)


def test_probability_at_least_matches_closed_forms() -> None:
    assert probability_at_least(
        sample_games=16, event_rate=0.02, target_events=1
    ) == pytest.approx(1.0 - 0.98**16)
    assert probability_at_least(
        sample_games=10, event_rate=0.5, target_events=10
    ) == pytest.approx(0.5**10)


def test_minimum_games_reaches_and_precedes_threshold() -> None:
    sample_games = minimum_games_for_event_yield(
        event_rate=22 / 1051,
        target_events=20,
        target_probability=0.90,
    )
    assert sample_games == 1234
    assert probability_at_least(
        sample_games=sample_games,
        event_rate=22 / 1051,
        target_events=20,
    ) >= 0.90
    assert probability_at_least(
        sample_games=sample_games - 1,
        event_rate=22 / 1051,
        target_events=20,
    ) < 0.90


def test_report_labels_plugin_estimates_as_non_power_diagnostics() -> None:
    report = build_report(
        label="v4-l5",
        observed_events=22,
        observed_games=1051,
        sample_sizes=[16, 128],
        target_event_counts=[20],
        confidence=0.95,
        target_probability=0.90,
    )
    assert report["scope"]["not_paired_effect_power"] is True
    assert report["candidateSamples"][0]["expectedEventsPlugIn"] == pytest.approx(
        16 * 22 / 1051
    )
    assert report["candidateSamples"][1]["probabilityZeroEventsPlugIn"] == (
        pytest.approx((1.0 - 22 / 1051) ** 128)
    )
    assert report["eventYieldThresholds"][0]["minimumGamesPlugIn"] == 1234


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = wilson_interval(22, 1051, 0.95)
    assert lower < 22 / 1051 < upper
