"""Segment stop must not be overshot by confirm/retry/branch counts."""

from __future__ import annotations

from scripts import train_s_gen_v2 as trainer


def test_confirm_requires_two_remaining_slots() -> None:
    assert trainer._confirm_fits_in_segment(0) is False
    assert trainer._confirm_fits_in_segment(1) is False
    assert trainer._confirm_fits_in_segment(2) is True


def test_extra_rollout_blocked_at_segment_stop() -> None:
    assert trainer._extra_rollout_fits_in_segment(15, 16) is True
    assert trainer._extra_rollout_fits_in_segment(16, 16) is False
    assert trainer._segment_slots_remaining(16, 16) == 0


def test_confirm_then_primary_reaches_stop_without_retry() -> None:
    """Reproduce the smoke_v3 overshoot pattern under a one-slot remainder."""
    game_count = 14
    segment_stop = 16

    room = trainer._segment_slots_remaining(game_count, segment_stop)
    assert trainer._confirm_fits_in_segment(room)
    game_count += 1  # confirm
    game_count += 1  # primary
    assert game_count == segment_stop
    assert trainer._extra_rollout_fits_in_segment(game_count, segment_stop) is False


def test_single_remaining_slot_skips_confirm_keeps_primary_exact() -> None:
    game_count = 15
    segment_stop = 16

    room = trainer._segment_slots_remaining(game_count, segment_stop)
    assert trainer._confirm_fits_in_segment(room) is False
    game_count += 1  # primary only
    assert game_count == segment_stop
