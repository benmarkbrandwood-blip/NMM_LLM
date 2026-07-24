"""Regression tests for gen v2a curriculum + termination behaviour.

Covers the changes in learned_ai/training/termination.py and
scripts/train_s_gen_v2a.py.  None of these tests require a training run;
they exercise the pure helpers and dataclass contracts directly.
"""
from __future__ import annotations

from collections import deque

import pytest

from game.board import BoardState, POSITIONS
from learned_ai.training.termination import (
    INFRA_REASONS,
    TerminationReason,
    VALID_OUTCOMES,
    classify_terminal,
    rolling_percentages,
)


# ── Termination classification ────────────────────────────────────────────────

def _blank_board() -> BoardState:
    return BoardState(
        positions={p: "" for p in POSITIONS},
        turn="W",
        pieces_on_board={"W": 0, "B": 0},
        pieces_placed={"W": 0, "B": 0},
        pieces_captured={"W": 0, "B": 0},
    )


def test_classify_win_by_fewer_than_three_learner_wins():
    """Opponent placed 9 and dropped below 3 → learner wins by material."""
    b = _blank_board()
    b.pieces_placed = {"W": 9, "B": 9}
    b.pieces_on_board = {"W": 5, "B": 2}   # black is losing
    assert classify_terminal(b, learner_color="W") == TerminationReason.WIN_FEWER_THAN_THREE


def test_classify_loss_by_fewer_than_three_learner_loses():
    """Learner dropped below 3 after placing 9."""
    b = _blank_board()
    b.pieces_placed = {"W": 9, "B": 9}
    b.pieces_on_board = {"W": 2, "B": 5}
    assert classify_terminal(b, learner_color="W") == TerminationReason.LOSS_FEWER_THAN_THREE


def test_infra_reasons_never_in_valid_outcomes():
    """Contract: INFRA_* and VALID_OUTCOMES are disjoint sets."""
    assert INFRA_REASONS.isdisjoint(VALID_OUTCOMES)


# ── Rolling percentage helper ─────────────────────────────────────────────────

def test_rolling_percentages_sum_to_100_over_valid_outcomes():
    window = [
        TerminationReason.WIN_FEWER_THAN_THREE,
        TerminationReason.WIN_NO_LEGAL_MOVE,
        TerminationReason.LOSS_FEWER_THAN_THREE,
        TerminationReason.LOSS_NO_LEGAL_MOVE,
        TerminationReason.DRAW_MAX_PLY_TRUNCATED,
    ]
    pcts = rolling_percentages(window)
    valid_pct = sum(pcts[r.value] for r in VALID_OUTCOMES if r.value in pcts)
    assert valid_pct == pytest.approx(100.0, abs=1e-6)


def test_rolling_percentages_ignore_infra_from_denominator():
    """Infra failures must not shrink the valid-outcome denominator."""
    window = [
        TerminationReason.WIN_FEWER_THAN_THREE,
        TerminationReason.WIN_FEWER_THAN_THREE,
        TerminationReason.INFRA_LEARNER_FAILURE,
        TerminationReason.INFRA_OPPONENT_FAILURE,
    ]
    pcts = rolling_percentages(window)
    assert pcts["win_lt3"] == pytest.approx(100.0)
    # Infra returned as absolute counts, not percentages
    assert pcts["infra_learner"] == 1.0
    assert pcts["infra_opponent"] == 1.0


def test_rolling_percentages_empty_window_returns_zeros():
    pcts = rolling_percentages([])
    for r in VALID_OUTCOMES:
        assert pcts[r.value] == 0.0
    assert pcts["infra_learner"] == 0.0
    assert pcts["infra_opponent"] == 0.0


# ── _record_rollout_outcome helper (imported from training script) ────────────
# The helper is defined inside scripts/train_s_gen_v2a.py.  Import via source
# execution so we don't need the full module dependency chain at import time.

def _load_record_helper():
    """Import _record_rollout_outcome from the training script.

    Direct import triggers heavy dependencies (torch, sentinel, etc.).  Read the
    function source and exec into an isolated namespace with just the deque + enum
    imports it needs.
    """
    from pathlib import Path
    src = Path(__file__).parent.parent / "scripts" / "train_s_gen_v2a.py"
    text = src.read_text()
    # Extract the function definition block.  It starts after the marker.
    start_marker = "def _record_rollout_outcome("
    idx = text.find(start_marker)
    assert idx != -1, "helper marker missing — did the training script refactor?"
    # Grab until the next top-level def or blank marker
    end_marker = "\n\ndef _check_advance"
    end = text.find(end_marker, idx)
    src_block = text[idx:end]
    ns: dict = {
        "deque": deque,
        "TerminationReason": TerminationReason,
        "INFRA_REASONS": INFRA_REASONS,
    }
    exec(
        "_INFRA_REASON_VALUES = frozenset(r.value for r in INFRA_REASONS)\n"
        + src_block,
        ns,
    )
    return ns["_record_rollout_outcome"]


def test_record_infra_learner_excluded_from_all_win_histories():
    record = _load_record_helper()
    wh, whh, lhh, th = deque(), deque(), deque(), deque()
    record(
        TerminationReason.INFRA_LEARNER_FAILURE.value, 0.5,
        wh, whh, lhh, th,
        is_full_diff=True, advance_cooldown_batches=0,
    )
    assert list(wh) == []
    assert list(whh) == []
    assert list(lhh) == []
    # But logged for observability
    assert list(th) == [TerminationReason.INFRA_LEARNER_FAILURE.value]


def test_record_infra_opponent_excluded_even_though_outcome_is_win():
    """Opponent failure produces WIN_REWARD outcome but must not count."""
    record = _load_record_helper()
    wh, whh, lhh, th = deque(), deque(), deque(), deque()
    record(
        TerminationReason.INFRA_OPPONENT_FAILURE.value, 1.0,
        wh, whh, lhh, th,
        is_full_diff=True, advance_cooldown_batches=0,
    )
    assert list(wh) == []
    assert list(whh) == []
    assert list(lhh) == []
    assert list(th) == [TerminationReason.INFRA_OPPONENT_FAILURE.value]


def test_record_cooldown_diverts_from_level_history_only():
    """During cooldown, full-diff outcomes still enter win_history + heuristic;
    only level_heuristic_history is gated."""
    record = _load_record_helper()
    wh, whh, lhh, th = deque(), deque(), deque(), deque()
    record(
        TerminationReason.WIN_FEWER_THAN_THREE.value, 1.0,
        wh, whh, lhh, th,
        is_full_diff=True, advance_cooldown_batches=5,
    )
    assert list(wh) == [1.0]
    assert list(whh) == [1.0]
    assert list(lhh) == []           # gated by cooldown
    assert list(th) == [TerminationReason.WIN_FEWER_THAN_THREE.value]


def test_record_no_cooldown_writes_to_level_history():
    record = _load_record_helper()
    wh, whh, lhh, th = deque(), deque(), deque(), deque()
    record(
        TerminationReason.WIN_FEWER_THAN_THREE.value, 1.0,
        wh, whh, lhh, th,
        is_full_diff=True, advance_cooldown_batches=0,
    )
    assert list(lhh) == [1.0]


def test_record_non_full_diff_never_enters_level_history():
    """Branch or lower-diff opponent games are is_full_diff=False."""
    record = _load_record_helper()
    wh, whh, lhh, th = deque(), deque(), deque(), deque()
    record(
        TerminationReason.WIN_FEWER_THAN_THREE.value, 1.0,
        wh, whh, lhh, th,
        is_full_diff=False, advance_cooldown_batches=0,
    )
    assert list(wh) == [1.0]         # still counts toward overall
    assert list(whh) == []           # not a full-diff heuristic sample
    assert list(lhh) == []


# ── Advance-check + cooldown interaction ──────────────────────────────────────

def test_level_history_cap_is_bounded():
    """The uncapped-history advancement drag is prevented by a maxlen cap.

    Simulates 500 outcomes into a deque with the same maxlen convention the
    training script uses (4 × rolling_win).  Verifies the deque never grows
    beyond the cap and that the mean reflects the tail — not the full history.
    """
    rolling_win = 40
    cap = rolling_win * 4  # 160
    lhh: deque[float] = deque(maxlen=cap)
    # 400 losses then 100 wins — old poor games should age out
    for _ in range(400):
        lhh.append(0.0)
    for _ in range(100):
        lhh.append(1.0)
    assert len(lhh) == cap, "deque should be at maxlen after > cap appends"
    mean = sum(lhh) / len(lhh)
    # Tail is 60 losses + 100 wins → mean 100/160 = 0.625
    assert mean == pytest.approx(0.625, abs=1e-6), (
        f"level history mean should reflect the tail, got {mean}"
    )


def test_advance_stat_over_infra_flooded_window_is_stable():
    """A window flooded with infra failures produces the same rolling percentages
    over the valid subset as a window without any infra rows."""
    without = [TerminationReason.WIN_FEWER_THAN_THREE] * 20 + [TerminationReason.LOSS_FEWER_THAN_THREE] * 10
    with_infra = without + [TerminationReason.INFRA_LEARNER_FAILURE] * 25
    p1 = rolling_percentages(without)
    p2 = rolling_percentages(with_infra)
    # Same denominator (30) after infra exclusion → same percentages
    for r in VALID_OUTCOMES:
        assert p1[r.value] == pytest.approx(p2[r.value], abs=1e-6)
