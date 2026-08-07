"""Focused evidence for shared history-dependent draw adjudication."""

from __future__ import annotations

from itertools import combinations

from game.board import POSITIONS, BoardState
from game.draw_rules import (
    NO_PROGRESS_DRAW,
    REPETITION_DRAW,
    StandardDrawState,
    StandardDrawTracker,
)
from game.game_engine import GameEngine


def _moving_board(*, turn: str = "W") -> BoardState:
    positions = {point: "" for point in POSITIONS}
    for point in ("d5", "a7", "b6", "g1"):
        positions[point] = "W"
    for point in ("e4", "g7", "b2", "a1"):
        positions[point] = "B"
    return BoardState.from_setup(positions, turn, "move")


def _cycle() -> list[dict[str, str | None]]:
    return [
        {"from": "d5", "to": "e5", "capture": None},
        {"from": "e4", "to": "e3", "capture": None},
        {"from": "e5", "to": "d5", "capture": None},
        {"from": "e3", "to": "e4", "capture": None},
    ]


def test_threefold_counts_the_initial_stable_moving_position() -> None:
    board = _moving_board()
    tracker = StandardDrawTracker(board)
    reasons = []

    for move in _cycle() * 2:
        after = board.apply_move(move)
        reasons.append(tracker.observe(board, move, after))
        board = after

    assert reasons[:-1] == [None] * 7
    assert reasons[-1] == REPETITION_DRAW


def test_game_engine_uses_the_shared_threefold_tracker() -> None:
    engine = GameEngine()
    engine.board = _moving_board()
    engine._draw_rules = StandardDrawTracker(engine.board)

    for move in _cycle() * 2:
        engine.apply_move(move)

    assert engine.finished
    assert engine.winner is None
    assert engine.draw_reason == REPETITION_DRAW
    assert engine.game_record["draw_reason"] == REPETITION_DRAW


def test_removal_resets_both_histories_then_observes_the_result() -> None:
    before = _moving_board()
    move = {"from": "d5", "to": "e5", "capture": "g7"}
    after = before.apply_move(move)
    target_key = StandardDrawTracker(after).snapshot().repetition_counts[0][0]
    tracker = StandardDrawTracker(
        before,
        state=StandardDrawState(((target_key, 2),), 99),
    )

    assert tracker.observe(before, move, after) is None
    snapshot = tracker.snapshot()
    assert snapshot.no_progress_plies == 0
    assert snapshot.repetition_counts == ((target_key, 1),)


def test_no_progress_draws_on_exactly_one_hundred_movement_plies() -> None:
    board = _moving_board()
    tracker = StandardDrawTracker(board)
    available = POSITIONS[:20]
    fixed_black = POSITIONS[20:24]

    reasons = []
    for index, white_points in enumerate(combinations(available, 4)):
        if index >= 100:
            break
        positions = {point: "" for point in POSITIONS}
        for point in white_points:
            positions[point] = "W"
        for point in fixed_black:
            positions[point] = "B"
        after = BoardState.from_setup(
            positions,
            "B" if board.turn == "W" else "W",
            "move",
        )
        move = {"from": "a7", "to": "d7", "capture": None}
        reasons.append(tracker.observe(board, move, after))
        board = after

    assert reasons[:99] == [None] * 99
    assert reasons[99] == NO_PROGRESS_DRAW


def test_repetition_has_priority_over_no_progress() -> None:
    before = _moving_board()
    move = _cycle()[0]
    after = before.apply_move(move)
    target_key = StandardDrawTracker(after).snapshot().repetition_counts[0][0]
    tracker = StandardDrawTracker(
        before,
        state=StandardDrawState(((target_key, 2),), 99),
    )

    assert tracker.observe(before, move, after) == REPETITION_DRAW
