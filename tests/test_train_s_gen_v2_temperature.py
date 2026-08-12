"""Regression tests for the Generalist v2 temperature schedule."""

from __future__ import annotations

import argparse

import pytest
import torch

from learned_ai.training.scaffolded_a2c import NonFiniteTrainingError
from scripts.train_s_gen_v2 import (
    TEMP_END,
    TEMP_START,
    _compute_post_fork_temperature,
    _compute_temperature,
    _compute_training_temperature,
    _finite_positive_float,
    _policy_distribution,
    _resolve_rollout_behaviour_temperature,
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


def test_post_fork_temperature_starts_at_global_fork_boundary():
    temperature = _compute_post_fork_temperature(
        post_fork_consumed_transitions=0,
        fork_game=50,
        max_games=5_000,
        temp_start=0.90,
        anneal_transitions=106_304,
    )

    assert temperature == pytest.approx(
        _compute_temperature(50, 5_000, 0.90)
    )


def test_post_fork_temperature_depends_on_transitions_not_game_count():
    first = _compute_training_temperature(
        schedule_axis="post-fork-transitions",
        game_count=300,
        max_games=5_000,
        temp_start=0.90,
        post_fork_consumed_transitions=8_192,
        fork_game=50,
        anneal_transitions=106_304,
    )
    second = _compute_training_temperature(
        schedule_axis="post-fork-transitions",
        game_count=500,
        max_games=5_000,
        temp_start=0.90,
        post_fork_consumed_transitions=8_192,
        fork_game=50,
        anneal_transitions=106_304,
    )

    assert first == pytest.approx(second)
    assert first == pytest.approx(0.8379808850090307)


def test_post_fork_temperature_can_preserve_a_mature_origin():
    first = _compute_training_temperature(
        schedule_axis="post-fork-transitions",
        game_count=327,
        max_games=5_000,
        temp_start=0.90,
        post_fork_consumed_transitions=0,
        fork_game=327,
        anneal_transitions=98_112,
        post_fork_temperature_origin=0.838,
    )
    second = _compute_training_temperature(
        schedule_axis="post-fork-transitions",
        game_count=518,
        max_games=5_000,
        temp_start=0.90,
        post_fork_consumed_transitions=4_096,
        fork_game=518,
        anneal_transitions=98_112,
        post_fork_temperature_origin=0.838,
    )

    assert first == pytest.approx(0.838)
    assert second < first
    assert second == pytest.approx(
        0.838 - (0.838 - TEMP_END) * (4_096 / 98_112)
    )


def test_rollout_temperature_schedule_uses_each_transition_ordinal():
    def schedule(index: int) -> float:
        return 0.9 - 0.01 * index

    assert _resolve_rollout_behaviour_temperature(
        default_temperature=0.5,
        learner_transition_index=0,
        schedule=schedule,
    ) == pytest.approx(0.9)
    assert _resolve_rollout_behaviour_temperature(
        default_temperature=0.5,
        learner_transition_index=7,
        schedule=schedule,
    ) == pytest.approx(0.83)


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_rollout_temperature_schedule_fails_closed(value):
    with pytest.raises(ValueError, match="finite and positive"):
        _resolve_rollout_behaviour_temperature(
            default_temperature=0.9,
            learner_transition_index=0,
            schedule=lambda _index: value,
        )


@pytest.mark.parametrize(
    ("consumed", "anneal"),
    [(-1, 106_304), (0, 0), (0, -1)],
)
def test_post_fork_temperature_rejects_invalid_transition_schedule(
    consumed, anneal
):
    with pytest.raises(ValueError, match="transition"):
        _compute_post_fork_temperature(
            post_fork_consumed_transitions=consumed,
            fork_game=50,
            max_games=5_000,
            temp_start=0.90,
            anneal_transitions=anneal,
        )


@pytest.mark.parametrize("origin", [0.0, -0.1, float("nan"), float("inf")])
def test_post_fork_temperature_rejects_invalid_explicit_origin(origin):
    with pytest.raises(ValueError, match="temperature origin"):
        _compute_post_fork_temperature(
            post_fork_consumed_transitions=0,
            fork_game=50,
            max_games=5_000,
            temp_start=0.90,
            anneal_transitions=106_304,
            temperature_origin=origin,
        )


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


def test_policy_distribution_uses_requested_temperature():
    log_probs, probs = _policy_distribution(torch.tensor([0.0, 2.0]), 0.5)

    expected = torch.log_softmax(torch.tensor([0.0, 4.0]), dim=-1)
    assert torch.equal(log_probs, expected)
    assert torch.equal(probs, expected.exp())


@pytest.mark.parametrize(
    "logits",
    [
        torch.tensor([0.0, float("nan")]),
        torch.tensor([0.0, float("inf")]),
    ],
)
def test_policy_distribution_fails_closed_on_non_finite_logits(logits):
    with pytest.raises(NonFiniteTrainingError, match="non-finite policy logits"):
        _policy_distribution(logits, 0.9)
