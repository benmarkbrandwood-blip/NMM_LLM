"""Focused tests for truthful Generalist update evidence."""

from __future__ import annotations

import pytest
import torch

from learned_ai.training.generalist_preflight import PreflightConfigurationError
from scripts import train_s_gen_v2 as trainer


def test_update_if_ready_does_not_claim_or_consume_small_batch() -> None:
    steps = [object()] * (trainer.MIN_UPDATE_STEPS - 1)

    def unexpected_update(*args, **kwargs):
        raise AssertionError("small batch must not call the optimizer")

    result = trainer._update_if_ready(
        update_fn=unexpected_update,
        model=object(),
        optimizer=object(),
        steps=steps,
        device=torch.device("cpu"),
        gamma=0.99,
        entropy_coef=0.01,
    )

    assert result is None
    assert len(steps) == trainer.MIN_UPDATE_STEPS - 1


def test_update_if_ready_reports_a_real_minimum_batch() -> None:
    steps = [object()] * trainer.MIN_UPDATE_STEPS
    calls = []

    def update(*args, **kwargs):
        calls.append((args, kwargs))
        return 1.0, 2.0, 3.0

    result = trainer._update_if_ready(
        update_fn=update,
        model=object(),
        optimizer=object(),
        steps=steps,
        device=torch.device("cpu"),
        gamma=0.99,
        entropy_coef=0.01,
    )

    assert result == (1.0, 2.0, 3.0)
    assert len(calls) == 1


def test_exact_transition_queue_consumes_64_and_retains_six() -> None:
    original = [bytes([index]) for index in range(70)]
    pending = list(original)

    batch = trainer._take_exact_transition_batch(pending, batch_size=64)

    assert batch == original[:64]
    assert pending == original[64:]
    assert pending[0] is original[64]


def test_exact_transition_queue_consumes_two_ordered_batches() -> None:
    original = list(range(140))
    pending = list(original)

    first = trainer._take_exact_transition_batch(pending, batch_size=64)
    second = trainer._take_exact_transition_batch(pending, batch_size=64)
    incomplete = trainer._take_exact_transition_batch(pending, batch_size=64)

    assert first == original[:64]
    assert second == original[64:128]
    assert incomplete is None
    assert pending == original[128:]


def test_exact_transition_queue_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        trainer._take_exact_transition_batch([], batch_size=0)


def test_configuration_rejects_update_cadence_below_real_batch(tmp_path) -> None:
    args = trainer._build_argument_parser().parse_args(
        [
            "--preflight",
            "smoke",
            "--out-dir",
            str(tmp_path / "out"),
            "--update-every",
            str(trainer.MIN_UPDATE_STEPS - 1),
        ]
    )

    with pytest.raises(PreflightConfigurationError, match="update_every must be at least"):
        trainer.validate_generalist_configuration(args)


def test_exact_transition_mode_requires_isolated_optimizer_path(tmp_path) -> None:
    args = trainer._build_argument_parser().parse_args(
        [
            "--preflight",
            "smoke",
            "--out-dir",
            str(tmp_path / "out"),
            "--exact-transition-batches",
        ]
    )

    with pytest.raises(
        PreflightConfigurationError,
        match="exact_transition_batches requires",
    ):
        trainer.validate_generalist_configuration(args)


def test_exact_transition_mode_accepts_frozen_diagnostic_controls(tmp_path) -> None:
    args = trainer._build_argument_parser().parse_args(
        [
            "--preflight",
            "smoke",
            "--out-dir",
            str(tmp_path / "out"),
            "--exact-transition-batches",
            "--no-s1a-warmstart",
            "--no-imitation-mix",
            "--no-s1b-refresher",
            "--no-recovery",
            "--no-opening-forcing",
            "--max-branches-per-game",
            "0",
        ]
    )

    trainer.validate_generalist_configuration(args)


@pytest.mark.parametrize(
    "outcome",
    [trainer.WIN_REWARD, trainer.LOSS_REWARD, trainer.DRAW_SHORT, trainer.DRAW_LONG],
)
def test_minimal_rollouts_keep_every_primary_outcome(outcome) -> None:
    assert trainer._keep_primary_trajectory(
        outcome,
        minimal_rollouts=True,
        confirmed=False,
    )


def test_nonminimal_loss_still_requires_confirmation() -> None:
    assert not trainer._keep_primary_trajectory(
        trainer.LOSS_REWARD,
        minimal_rollouts=False,
        confirmed=False,
    )
    assert trainer._keep_primary_trajectory(
        trainer.LOSS_REWARD,
        minimal_rollouts=False,
        confirmed=True,
    )
