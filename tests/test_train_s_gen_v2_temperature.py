"""Regression tests for the Generalist v2 temperature schedule."""

from __future__ import annotations

import argparse

import pytest

from scripts.train_s_gen_v2 import (
    TEMP_END,
    TEMP_START,
    _compute_temperature,
    _finite_positive_float,
)


@pytest.mark.parametrize(
    ("game_count", "expected"),
    [
        (0, 1.10),
        (400, 0.65),
        (800, TEMP_END),
        (1_000, TEMP_END),
    ],
)
def test_temperature_schedule_honours_configured_start(game_count, expected):
    temperature = _compute_temperature(
        game_count=game_count,
        max_games=1_000,
        temp_start=1.10,
    )

    assert temperature == pytest.approx(expected)


def test_default_temperature_schedule_is_unchanged():
    assert _compute_temperature(0, 1_000, TEMP_START) == pytest.approx(0.90)
    assert _compute_temperature(400, 1_000, TEMP_START) == pytest.approx(0.55)
    assert _compute_temperature(800, 1_000, TEMP_START) == pytest.approx(TEMP_END)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.01", 0.01),
        ("1", 1.0),
        ("1e2", 100.0),
    ],
)
def test_temp_start_accepts_finite_positive_values(value, expected):
    assert _finite_positive_float(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    ["0", "-0.1", "nan", "inf", "-inf", "not-a-number"],
)
def test_temp_start_rejects_non_positive_or_non_finite_values(value):
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="finite positive number",
    ):
        _finite_positive_float(value)
