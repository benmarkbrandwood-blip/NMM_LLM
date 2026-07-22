"""Tests for move-level Sentinel quality labels and weights."""

from __future__ import annotations

import numpy as np
import pytest

from learned_ai.sentinel.feature_builder import FEATURE_DIM
from learned_ai.sentinel.labels import (
    BAD_MOVE_DTM_THRESHOLD,
    BAD_MOVE_WEIGHT,
    DRAW_WEIGHT,
    WEAK_LABEL_WEIGHT,
    MoveExample,
    dtm_quality,
    label_move,
    quality_from_wdl,
)


@pytest.mark.parametrize(
    ("wdl", "expected"),
    [
        ("win", 1.0),
        ("draw", 0.5),
        ("loss", 0.0),
        ("unknown", None),
        (None, None),
    ],
)
def test_quality_from_wdl_uses_mover_perspective(wdl, expected):
    assert quality_from_wdl(wdl) == expected


@pytest.mark.parametrize(
    ("wdl", "dtm", "expected"),
    [
        ("win", 1, 0.99),
        ("win", 100, 0.55),
        ("win", 1000, 0.55),
        ("loss", 1, 0.01),
        ("loss", 100, 0.45),
        ("loss", 1000, 0.45),
        ("draw", 25, 0.5),
    ],
)
def test_dtm_quality_is_bounded_and_directional(wdl, dtm, expected):
    assert dtm_quality(wdl, dtm) == pytest.approx(expected)


def test_draw_label_uses_reduced_solved_weight():
    quality, weight, source = label_move("draw")

    assert quality == 0.5
    assert weight == DRAW_WEIGHT
    assert source == "solved_db"


def test_dtm_label_preserves_solved_provenance():
    quality, weight, source = label_move("win", dtm=4)

    assert quality == pytest.approx(dtm_quality("win", 4))
    assert weight == 1.0
    assert source == "solved_db_dtm"


def test_imminent_solved_loss_receives_extra_weight():
    quality, weight, source = label_move(
        "loss",
        dtm=BAD_MOVE_DTM_THRESHOLD,
    )

    assert quality == pytest.approx(
        dtm_quality("loss", BAD_MOVE_DTM_THRESHOLD)
    )
    assert weight == BAD_MOVE_WEIGHT
    assert source == "solved_db_dtm"


def test_non_imminent_solved_loss_keeps_normal_weight():
    _, weight, source = label_move(
        "loss",
        dtm=BAD_MOVE_DTM_THRESHOLD + 1,
    )

    assert weight == 1.0
    assert source == "solved_db_dtm"


@pytest.mark.parametrize(
    ("heuristic_score", "expected"),
    [(-3.0, 0.0), (0.25, 0.25), (4.0, 1.0)],
)
def test_unknown_wdl_falls_back_to_clamped_weak_label(
    heuristic_score,
    expected,
):
    quality, weight, source = label_move(
        None,
        heuristic_score_norm=heuristic_score,
    )

    assert quality == expected
    assert weight == WEAK_LABEL_WEIGHT
    assert source == "heuristic_weak"


def test_move_example_exposes_scalar_target():
    example = MoveExample(
        features=np.zeros(FEATURE_DIM, dtype=np.float32),
        move_quality=0.75,
        training_weight=1.0,
        supervision_source="solved_db",
        ply=12,
        move_notation="a7-d7",
        position_key="example-position",
    )

    assert example.target() == 0.75
    assert isinstance(example.target(), float)
