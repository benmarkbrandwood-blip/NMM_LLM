"""ai/gap_net.py — GapNet: predicts human blunder density for a board position.

Identical architecture to ValueNet (79→128→64→1 tanh).
Trained on Malom WDL opportunity-gap targets (see scripts/build_gap_dataset.py).
Output in (-1, 1): near +1 = humans frequently blunder here; near -1 = humans play well.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ai.value_net import ValueNet, board_to_features, _INPUT_DIM  # reuse features

GapNet = ValueNet  # same architecture; distinguished only by training data and save path

_GAP_SCALE = 3000  # multiplier to convert gap_net output to heuristic integer units

def gap_net_correction(gap_net: ValueNet, board, color: str, e_v2: int,
                       phase: str, blend_place: int, blend_move: int, blend_fly: int) -> int:
    """Apply additive blunder-zone correction to an already-computed E_v2 score.
    Only call this when board.turn == AI's color.
    Returns e_v2 unchanged when gap_net is None or phase blend is 0.
    """
    _blend = {"place": blend_place, "move": blend_move, "fly": blend_fly}
    cap_pct = _blend.get(phase, 0)
    if cap_pct <= 0:
        return e_v2
    gamma = cap_pct / 100.0
    raw = gap_net.predict(board, color)  # tanh output in (-1, 1)
    gap = (raw + 1.0) / 2.0             # convert to [0, 1]
    if gap < 0.02:
        return e_v2
    bonus = int(gamma * gap * _GAP_SCALE)
    if phase == "fly":
        bonus = min(bonus, int(0.3 * gamma * _GAP_SCALE))
    return e_v2 + bonus
