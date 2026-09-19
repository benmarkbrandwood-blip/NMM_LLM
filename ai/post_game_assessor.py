"""
ai/post_game_assessor.py — Post-game per-ply analysis.

PostGameAssessor replays a completed game record and produces a
PostGameAnnotation with per-ply heuristic scoring, score curves,
and turning-point detection.

Stage 1: heuristic signal only.
Stage 2 adds Sentinel; Stage 3 Malom; Stage 4 Trajectory + Policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from game.board import BoardState
from game.rules import get_all_legal_moves, get_game_phase
from ai.game_ai import GameAI
from ai.heuristics import evaluate_v2


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MoveAnnotation:
    ply: int
    color: str
    phase: str
    move_played: str       # notation string
    best_alt: Optional[str]

    # Heuristic signal
    heuristic_score_white: float    # absolute eval after move, White-normalised
    score_played: float             # normalised 0.0 (worst) – 1.0 (best)
    score_best: float               # always 1.0
    r_h: float                      # heuristic regret = 1.0 − score_played

    # Sentinel signal (Stage 2)
    sentinel_score_white: Optional[float] = None
    sentinel_played: Optional[float] = None
    sentinel_best: Optional[float] = None
    r_s: Optional[float] = None

    # Trajectory signal (Stage 4)
    traj_delta_played: Optional[float] = None
    traj_delta_best: Optional[float] = None
    r_t: Optional[float] = None
    traj_n: Optional[int] = None

    # Malom adjudication (Stage 3)
    wdl_before: Optional[str] = None   # "W"|"D"|"L" from mover's perspective
    wdl_after: Optional[str] = None    # "W"|"D"|"L" from mover's perspective
    oracle_source: str = "none"
    abstained_reason: Optional[str] = None
    quality: str = "clean"             # "confirmed_poor"|"poor_candidate"|"clean"

    # Human policy (Stage 4)
    policy_prob: Optional[float] = None
    policy_top_move: Optional[str] = None
    policy_top_prob: Optional[float] = None
    policy_prob_source: Optional[str] = None
    policy_support_n: Optional[int] = None
    is_unconventional: bool = False


@dataclass
class PostGameAnnotation:
    moves: list[MoveAnnotation]
    heuristic_curve: list[float]           # heuristic_score_white at each ply
    sentinel_curve: list[Optional[float]]  # None entries when Sentinel unavailable
    turning_point_ply: Optional[int]
    turning_point_quality: str             # e.g. "r_h:0.712" or "win_to_loss"
    turning_point_oracle: str              # "malom_full"|"retrograde_wdl"|"sentinel+heuristic"|"heuristic"
    opening_name: Optional[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _move_notation(move: dict) -> str:
    s = f"{move['from']}-{move['to']}" if move.get("from") else move.get("to", "")
    if move.get("capture"):
        s += f"x{move['capture']}"
    return s


# ── Assessor ──────────────────────────────────────────────────────────────────

class PostGameAssessor:
    """Replay a completed game record and produce a PostGameAnnotation.

    Parameters
    ----------
    difficulty:
        GameAI difficulty level. Has minor effect on move ordering heuristics.
    depth:
        Maximum search depth for per-ply scoring. Low values (3–4) keep
        assessment fast while preserving directional accuracy. Default 4.
    """

    def __init__(self, difficulty: int = 3, depth: int = 4) -> None:
        self._ai = GameAI(color="W", difficulty=difficulty)
        self._ai.max_search_depth = depth

    def assess(self, game_record: dict) -> PostGameAnnotation:
        """Replay `game_record` and return a fully annotated PostGameAnnotation."""
        moves_raw = game_record.get("moves", [])

        opening_name: Optional[str] = None
        for m in moves_raw:
            rec = m.get("opening_recognition") or {}
            if rec.get("status") in ("exact", "transposition") and rec.get("name"):
                opening_name = rec["name"]
                break

        annotations: list[MoveAnnotation] = []
        board = BoardState.new_game()

        for ply_idx, move_record in enumerate(moves_raw):
            played_move = {
                "from": move_record.get("from"),
                "to": move_record["to"],
                "capture": move_record.get("capture"),
            }
            color = move_record.get("color", board.turn)
            phase = get_game_phase(board, color)
            notation = move_record.get("notation") or _move_notation(played_move)

            scored = self._ai.assess_position(board)

            board_after = board.apply_move(played_move)
            heuristic_score_white = float(evaluate_v2(board_after, "W"))

            if scored:
                all_s = [s for _, s in scored]
                lo, hi = min(all_s), max(all_s)
                best_alt_move = scored[0][0]
                best_alt_notation = _move_notation(best_alt_move)

                move_key = (played_move.get("from"), played_move["to"], played_move.get("capture"))
                played_raw = next(
                    (s for m, s in scored
                     if (m.get("from"), m["to"], m.get("capture")) == move_key),
                    None,
                )

                if hi == lo:
                    score_played_norm = 1.0
                elif played_raw is None:
                    score_played_norm = 0.0
                else:
                    score_played_norm = (played_raw - lo) / (hi - lo)
                score_best_norm = 1.0
            else:
                score_played_norm = 1.0
                score_best_norm = 1.0
                best_alt_notation = None

            r_h = max(0.0, 1.0 - score_played_norm)

            annotations.append(MoveAnnotation(
                ply=ply_idx,
                color=color,
                phase=phase,
                move_played=notation,
                best_alt=best_alt_notation,
                heuristic_score_white=heuristic_score_white,
                score_played=score_played_norm,
                score_best=score_best_norm,
                r_h=r_h,
            ))
            board = board_after

        heuristic_curve = [a.heuristic_score_white for a in annotations]
        sentinel_curve: list[Optional[float]] = [a.sentinel_score_white for a in annotations]

        tp_ply, tp_quality, tp_oracle = self._detect_turning_point(annotations)

        return PostGameAnnotation(
            moves=annotations,
            heuristic_curve=heuristic_curve,
            sentinel_curve=sentinel_curve,
            turning_point_ply=tp_ply,
            turning_point_quality=tp_quality,
            turning_point_oracle=tp_oracle,
            opening_name=opening_name,
        )

    def _detect_turning_point(
        self, annotations: list[MoveAnnotation]
    ) -> tuple[Optional[int], str, str]:
        """Return (ply, quality_str, oracle_source) for the turning point.

        Stage 1: heuristic-only — ply where h(t) drops most steeply.
        """
        if len(annotations) < 2:
            return None, "", "heuristic"

        best_ply: Optional[int] = None
        best_drop = -1.0

        for i in range(1, len(annotations)):
            drop = (annotations[i - 1].heuristic_score_white
                    - annotations[i].heuristic_score_white)
            if drop > best_drop:
                best_drop = drop
                best_ply = i

        if best_ply is None or best_drop <= 0.0:
            return None, "", "heuristic"

        ann = annotations[best_ply]
        quality_str = f"r_h:{ann.r_h:.3f}"
        return best_ply, quality_str, "heuristic"
