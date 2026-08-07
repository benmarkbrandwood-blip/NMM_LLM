"""Focused tests for Generalist curriculum evidence routing."""

from __future__ import annotations

from collections import deque

from scripts import train_s_gen_v2 as trainer


def _histories() -> tuple[deque[float], deque[float], deque[float]]:
    return deque(), deque(), deque()


def test_primary_reference_game_feeds_advancement_history() -> None:
    all_games, heuristic_games, advancement_games = _histories()

    added = trainer._record_curriculum_outcome(
        trainer.WIN_REWARD,
        win_history=all_games,
        win_history_heuristic=heuristic_games,
        level_heuristic_history=advancement_games,
        is_full_diff=True,
        is_advance_reference=True,
    )

    assert added is True
    assert list(all_games) == [1.0]
    assert list(heuristic_games) == [1.0]
    assert list(advancement_games) == [1.0]


def test_confirmation_or_retry_does_not_feed_advancement_history() -> None:
    all_games, heuristic_games, advancement_games = _histories()

    added = trainer._record_curriculum_outcome(
        trainer.DRAW_SHORT,
        win_history=all_games,
        win_history_heuristic=heuristic_games,
        level_heuristic_history=advancement_games,
        is_full_diff=True,
        is_advance_reference=False,
    )

    assert added is False
    assert list(all_games) == [0.5]
    assert list(heuristic_games) == [0.5]
    assert list(advancement_games) == []


def test_frozen_or_branch_game_only_feeds_all_game_history() -> None:
    all_games, heuristic_games, advancement_games = _histories()

    added = trainer._record_curriculum_outcome(
        trainer.LOSS_REWARD,
        win_history=all_games,
        win_history_heuristic=heuristic_games,
        level_heuristic_history=advancement_games,
        is_full_diff=False,
        is_advance_reference=False,
    )

    assert added is False
    assert list(all_games) == [0.0]
    assert list(heuristic_games) == []
    assert list(advancement_games) == []
