"""learned_ai/models/lookahead_advisor.py — 5-ply heuristic-only lookahead.

For each legal move at the current position, simulates 5 half-plies using the
static heuristic for BOTH sides (no model calls, no recursion, no feedback loop).
At each depth, records 3 signals from the learner's perspective:

  h_norm   : (evaluate(board, learner_color) + 1) / 2  → [0, 1]
  vn_norm  : (value_net.predict(board, learner_color) + 1) / 2  (0.5 if no VN)
  sent_mean: mean sentinel score for current-player moves  (0.5 if disabled/unavailable)
             Flipped to 1 - mean when it is the opponent's turn, so the signal
             always expresses learner-perspective favourability.

The 5-depth × 3-signal = 15-float block is appended to the 62-float base features
by encode_position_with_lookahead(), producing the 77-float specialist input.

Sentinel calls inside lookahead are expensive (up to k × 5 per turn).  They are
disabled by default (use_sentinel=False fills sent_mean with 0.5); enable only
when you are willing to pay the latency.  The sentinel base signal is already
available in feature [58] of the base encoding.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal


def _static_best_move(board: BoardState, color: str, evaluate_fn) -> Optional[dict]:
    """Pick the move that maximises static heuristic eval for `color`."""
    moves = get_all_legal_moves(board)
    if not moves:
        return None
    best_score = -math.inf
    best_move  = moves[0]
    for mv in moves:
        try:
            after = board.apply_move(mv)
            score = float(evaluate_fn(after, color))
            if score > best_score:
                best_score = score
                best_move  = mv
        except Exception:
            pass
    return best_move


class LookaheadAdvisor:
    """N-ply heuristic lookahead scoring for each legal move.

    Returns a (k, ply_depth*3) ndarray — one row per candidate move —
    for use as the lookahead block in the specialist/overseer input.

    Parameters
    ----------
    sentinel      : SentinelAdvisor or None
    value_net     : value net with .predict(board, player) → float in [-1, 1], or None
    evaluate_fn   : callable(board, player) → float in [-1, 1]
    use_sentinel  : if False (default), sent_mean is always 0.5 (faster training)
    ply_depth     : number of half-plies to simulate (default 5; use 12 for Overseer)
    """

    def __init__(
        self,
        sentinel,
        value_net,
        evaluate_fn,
        use_sentinel: bool = False,
        endgame_db=None,
        ply_depth: int = 5,
    ) -> None:
        self._sentinel     = sentinel
        self._value_net    = value_net
        self._evaluate     = evaluate_fn
        self._use_sentinel = use_sentinel
        self._endgame_db   = endgame_db
        self._ply_depth    = ply_depth
        self.feat_dim      = ply_depth * 3   # total floats per move row

    def score_moves_matrix(
        self,
        board: BoardState,
        enc,
        learner_color: str,
    ) -> np.ndarray:
        """Return per-move lookahead feature block.  Shape: (k, ply_depth*3).

        Each row is the trajectory for one legal move.
        Returns a zero-filled matrix on any top-level error (neutral, no distortion).
        """
        opp_color = "B" if learner_color == "W" else "W"
        k = len(enc.legal_moves)
        if k == 0:
            return np.zeros((0, self.feat_dim), dtype=np.float32)

        rows = []
        for mv in enc.legal_moves:
            row = self._simulate_trajectory(board, mv, learner_color, opp_color)
            rows.append(row)

        return np.stack(rows).astype(np.float32)   # (k, ply_depth*3)

    # ── internal helpers ───────────────────────────────────────────────────────

    def _simulate_trajectory(
        self,
        board: BoardState,
        first_move: dict,
        learner_color: str,
        opp_color: str,
    ) -> np.ndarray:
        """Simulate self._ply_depth half-plies and return a (ply_depth*3,) float array.

        Half-plies alternate learner → opponent → learner → …
        Both sides always play the static-heuristic-best move.  first_move is
        the candidate move for half-ply 1 (the learner's choice being evaluated).
        On terminal or no-legal-moves, remaining depths are filled with the last
        valid signal (or the terminal score for all three channels).
        """
        # Layout: [h1, vn1, s1,  h2, vn2, s2, ...,  hN, vnN, sN]
        result = np.full(self._ply_depth * 3, 0.5, dtype=np.float32)
        try:
            b      = board
            actors = [learner_color if i % 2 == 0 else opp_color for i in range(self._ply_depth)]
            last_sig = (0.5, 0.5, 0.5)

            for depth_idx in range(self._ply_depth):
                actor = actors[depth_idx]

                # Apply the move for this half-ply
                if depth_idx == 0:
                    b = b.apply_move(first_move)
                else:
                    mv = _static_best_move(b, actor, self._evaluate)
                    if mv is None:
                        # No legal moves — propagate last signal to remaining depths
                        for fill in range(depth_idx, self._ply_depth):
                            result[fill * 3 : fill * 3 + 3] = last_sig
                        return result
                    b = b.apply_move(mv)

                # Check for terminal
                terminal, winner = is_terminal(b)
                if terminal:
                    val = 1.0 if winner == learner_color else (0.0 if winner else 0.5)
                    for fill in range(depth_idx, 5):
                        result[fill * 3 : fill * 3 + 3] = [val, val, val]
                    return result

                # Endgame DB probe — exact WDL terminates the trajectory early
                if self._endgame_db is not None:
                    try:
                        db_result = self._endgame_db.query(b)
                        if db_result is not None:
                            # db_result is from side-to-move's perspective ("W"/"L"/"D")
                            if b.turn == learner_color:
                                val = 1.0 if db_result == "W" else (0.0 if db_result == "L" else 0.5)
                            else:
                                val = 0.0 if db_result == "W" else (1.0 if db_result == "L" else 0.5)
                            for fill in range(depth_idx, self._ply_depth):
                                result[fill * 3 : fill * 3 + 3] = [val, val, val]
                            return result
                    except Exception:
                        pass

                # Record signals at this position
                sig = self._record_signals(b, learner_color)
                last_sig = sig
                result[depth_idx * 3 : depth_idx * 3 + 3] = sig

        except Exception:
            pass   # partial result already in `result`; remaining slots stay 0.5

        return result

    def _record_signals(
        self,
        board: BoardState,
        learner_color: str,
    ) -> tuple[float, float, float]:
        """Return (h_norm, vn_norm, sent_mean) from the learner's perspective.

        h_norm and vn_norm are always from the learner's view regardless of whose
        turn it is.  sent_mean is the mean sentinel quality of the current player's
        legal moves, flipped when the opponent is to move so the value still
        represents learner-favourability.
        """
        opp_color = "B" if learner_color == "W" else "W"
        current_player = board.turn   # always the OTHER player after apply_move

        # ── heuristic ─────────────────────────────────────────────────────────
        h_norm = 0.5
        try:
            h = float(self._evaluate(board, learner_color))
            h_norm = max(0.0, min(1.0, (h + 1.0) / 2.0))
        except Exception:
            pass

        # ── value net ─────────────────────────────────────────────────────────
        vn_norm = 0.5
        if self._value_net is not None:
            try:
                vn = float(self._value_net.predict(board, learner_color))
                vn_norm = max(0.0, min(1.0, (vn + 1.0) / 2.0))
            except Exception:
                pass

        # ── sentinel ──────────────────────────────────────────────────────────
        sent_mean = 0.5
        if self._use_sentinel and self._sentinel is not None:
            try:
                legal = get_all_legal_moves(board)
                if legal:
                    advice = self._sentinel.advise(board, legal, current_player)
                    m = float(sum(advice.move_scores) / len(advice.move_scores))
                    # Flip if it's the opponent's turn (high opponent score = bad for us)
                    sent_mean = max(0.0, min(1.0, m if current_player == learner_color else 1.0 - m))
            except Exception:
                pass

        return h_norm, vn_norm, sent_mean
