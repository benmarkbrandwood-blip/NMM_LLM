"""Regression tests for Generalist v2 checkpoint creation and reporting."""

from __future__ import annotations

import pytest

from scripts import train_s_gen_v2 as trainer


@pytest.mark.parametrize(
    ("win_rate", "previous_best", "game_count", "expected"),
    [
        (1.0, 0.0, trainer.BEST_CHECKPOINT_MIN_GAMES - 1, False),
        (0.5, 0.5, trainer.BEST_CHECKPOINT_MIN_GAMES, False),
        (0.6, 0.5, trainer.BEST_CHECKPOINT_MIN_GAMES, True),
    ],
)
def test_best_checkpoint_eligibility(
    win_rate, previous_best, game_count, expected
):
    assert trainer._should_save_best_checkpoint(
        win_rate,
        previous_best,
        game_count,
    ) is expected


def test_final_report_does_not_claim_missing_best(tmp_path, capsys):
    latest_path = tmp_path / "latest.pt"
    latest_path.touch()

    trainer._report_final_checkpoints(tmp_path)

    output = capsys.readouterr().out
    assert f"Latest checkpoint: {latest_path}" in output
    assert "Best checkpoint: not created" in output
    assert "at least 10 heuristic games" in output


def test_final_report_names_available_best(tmp_path, capsys):
    latest_path = tmp_path / "latest.pt"
    best_path = tmp_path / "best.pt"
    latest_path.touch()
    best_path.touch()

    trainer._report_final_checkpoints(tmp_path)

    output = capsys.readouterr().out
    assert f"Latest checkpoint: {latest_path}" in output
    assert f"Best checkpoint available: {best_path}" in output
    assert "not created" not in output
