"""
ai/post_game_assessor.py — Post-game per-ply analysis.

PostGameAssessor replays a completed game record and produces a
PostGameAnnotation with per-ply heuristic + sentinel scoring, score curves,
and turning-point detection.

Stage 1: heuristic signal.
Stage 2: Sentinel signal added.
Stage 3 adds Malom; Stage 4 Trajectory + Human Policy + Generalist AI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from learned_ai.sentinel.infer import SentinelAdvisor
    from ai.value_net import ValueNet  # GapNet shares this architecture
    from ai.malom_db import MalomDB
    from ai.trajectory_db import TrajectoryDB
    from ai.human_move_policy_advisor import HumanMovePolicyAdvisor
    from learned_ai.agents.specialist_router import GeneralistAgent

# Malom WDL transitions that mean the mover threw away a better outcome
_MALOM_CONFIRMED_POOR = frozenset({"win_to_draw", "win_to_loss", "draw_to_loss"})
_MALOM_CLEAN          = frozenset({"win_preserved", "draw_preserved", "all_losing"})

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
    sentinel_score_white: Optional[float] = None   # played quality, White-normalised (curve value)
    sentinel_played: Optional[float] = None        # raw Sentinel quality, mover's perspective
    sentinel_best: Optional[float] = None          # highest Sentinel score among candidates
    r_s: Optional[float] = None                    # sentinel regret = sentinel_best − sentinel_played

    # GapNet blunder-zone density (Stage 2)
    blunder_zone_score: Optional[float] = None     # (raw+1)/2 for board BEFORE move [0,1]

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

    # Generalist AI policy (Stage 4)
    generalist_policy_prob: Optional[float] = None   # P(generalist plays this move)
    generalist_top_move: Optional[str] = None        # generalist's preferred alternative
    generalist_value_after: Optional[float] = None   # value-head win prob, White-normalised
    generalist_self_assessed: bool = False            # True when generalist played this side


@dataclass
class PostGameAnnotation:
    moves: list[MoveAnnotation]
    heuristic_curve: list[float]           # heuristic_score_white at each ply
    sentinel_curve: list[Optional[float]]  # None entries when Sentinel unavailable
    turning_point_ply: Optional[int]
    turning_point_quality: str             # e.g. "r_h:0.712", "r_h+r_s:0.712", or "win_to_loss"
    turning_point_oracle: str              # "malom_full"|"retrograde_wdl"|"sentinel+heuristic"|"heuristic"
    opening_name: Optional[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _move_notation(move: dict) -> str:
    s = f"{move['from']}-{move['to']}" if move.get("from") else move.get("to", "")
    if move.get("capture"):
        s += f"x{move['capture']}"
    return s


def _find_played_idx(candidates: list[dict], played_move: dict) -> int:
    """Return index of played_move in candidates list, or 0 if not found."""
    key = (played_move.get("from"), played_move["to"], played_move.get("capture"))
    for i, m in enumerate(candidates):
        if (m.get("from"), m["to"], m.get("capture")) == key:
            return i
    return 0


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
    sentinel:
        Optional SentinelAdvisor. When provided, sentinel fields are populated
        per ply. Skipped gracefully when None.
    """

    def __init__(
        self,
        difficulty: int = 3,
        depth: int = 4,
        sentinel: Optional["SentinelAdvisor"] = None,
        gap_net: Optional["ValueNet"] = None,
        malom_db: Optional["MalomDB"] = None,
        trajectory_db: Optional["TrajectoryDB"] = None,
        policy_advisor: Optional["HumanMovePolicyAdvisor"] = None,
        policy_elo_band: str = "all",
        generalist: Optional["GeneralistAgent"] = None,
    ) -> None:
        self._ai = GameAI(color="W", difficulty=difficulty)
        self._ai.max_search_depth = depth
        self._sentinel = sentinel
        self._gap_net = gap_net
        self._malom_db = malom_db
        self._trajectory_db = trajectory_db
        self._policy_advisor = policy_advisor
        self._policy_elo_band = policy_elo_band
        self._generalist = generalist

    def assess(self, game_record: dict) -> PostGameAnnotation:
        """Replay `game_record` and return a fully annotated PostGameAnnotation."""
        moves_raw = game_record.get("moves", [])

        # Derive AI color for generalist_self_assessed flagging.
        human_color = game_record.get("human_color")
        ai_color: Optional[str] = (
            ("B" if human_color == "W" else "W") if human_color else None
        )

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

            # ── Heuristic fields ──────────────────────────────────────────────
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

            # Candidate list shared by Sentinel, Policy, and Generalist.
            candidates = [m for m, _ in scored]

            # ── Sentinel fields ───────────────────────────────────────────────
            sentinel_score_white: Optional[float] = None
            sentinel_played: Optional[float] = None
            sentinel_best: Optional[float] = None
            r_s: Optional[float] = None

            if self._sentinel is not None and candidates:
                played_idx = _find_played_idx(candidates, played_move)
                advice = self._sentinel.advise(board, candidates, color, played_idx)
                if advice is not None:
                    sentinel_played = advice.played_move_quality
                    sentinel_best = advice.best_available_quality
                    r_s = advice.opportunity_gap
                    # White-normalise: negate for Black plies
                    sentinel_score_white = (
                        sentinel_played if color == "W" else 1.0 - sentinel_played
                    )

            # ── GapNet blunder-zone density ───────────────────────────────────
            blunder_zone_score: Optional[float] = None

            if self._gap_net is not None:
                raw = self._gap_net.predict(board, color)   # tanh output in (-1, 1)
                blunder_zone_score = (raw + 1.0) / 2.0      # convert to [0, 1]

            # ── Malom adjudication ────────────────────────────────────────────
            wdl_before: Optional[str] = None
            wdl_after: Optional[str] = None
            oracle_source = "none"
            abstained_reason: Optional[str] = None
            quality = "clean"

            if self._malom_db is not None and self._malom_db.is_available():
                parent_val = self._malom_db.query_value(board)
                if parent_val is None:
                    abstained_reason = "parent_value_unavailable"
                else:
                    wdl_before = parent_val.outcome
                    if wdl_before == "L":
                        # Already losing — any move is equally bad; don't flag.
                        abstained_reason = "already_losing"
                    else:
                        result = self._malom_db.query_regret(board, played_move)
                        if not result.available:
                            wdl_before = None
                            abstained_reason = result.unavailable_reason
                        else:
                            wdl_after = result.omv.outcome
                            transition = result.wdl_transition
                            if transition in _MALOM_CONFIRMED_POOR:
                                oracle_source = "malom_full"
                                quality = "confirmed_poor"
                            elif transition in _MALOM_CLEAN:
                                oracle_source = "malom_full"
                                quality = "clean"
                            else:
                                # label_inconsistency or unexpected value — fail closed
                                wdl_before = None
                                wdl_after = None
                                abstained_reason = f"malom_{transition}"

            # ── Trajectory signal ─────────────────────────────────────────────
            traj_delta_played: Optional[float] = None
            traj_delta_best: Optional[float] = None
            r_t: Optional[float] = None

            if self._trajectory_db is not None and candidates:
                hints = self._trajectory_db.query(board, color)
                if hints:
                    traj_delta_best = max(hints.values())
                    traj_delta_played = hints.get(notation)
                    if traj_delta_played is not None:
                        r_t = traj_delta_best - traj_delta_played

            # ── Human policy signal ───────────────────────────────────────────
            policy_prob: Optional[float] = None
            policy_top_move: Optional[str] = None
            policy_top_prob: Optional[float] = None
            policy_prob_source: Optional[str] = None
            is_unconventional = False

            if self._policy_advisor is not None and candidates:
                probs = self._policy_advisor.probs(board, candidates,
                                                   self._policy_elo_band)
                if len(probs) > 0:
                    played_idx = _find_played_idx(candidates, played_move)
                    policy_prob = float(probs[played_idx])
                    top_idx = int(np.argmax(probs))
                    policy_top_move = _move_notation(candidates[top_idx])
                    policy_top_prob = float(probs[top_idx])
                    policy_prob_source = "learned"

            # ── Generalist AI policy signal ───────────────────────────────────
            generalist_policy_prob: Optional[float] = None
            generalist_top_move: Optional[str] = None
            generalist_self_assessed = (ai_color is not None and color == ai_color)

            if self._generalist is not None and candidates:
                g_scores = self._generalist.score_moves(board, candidates, color)
                if g_scores is not None and len(g_scores) == len(candidates):
                    played_idx = _find_played_idx(candidates, played_move)
                    generalist_policy_prob = float(g_scores[played_idx])
                    top_idx = int(max(range(len(g_scores)), key=lambda i: g_scores[i]))
                    generalist_top_move = _move_notation(candidates[top_idx])

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
                sentinel_score_white=sentinel_score_white,
                sentinel_played=sentinel_played,
                sentinel_best=sentinel_best,
                r_s=r_s,
                blunder_zone_score=blunder_zone_score,
                wdl_before=wdl_before,
                wdl_after=wdl_after,
                oracle_source=oracle_source,
                abstained_reason=abstained_reason,
                quality=quality,
                traj_delta_played=traj_delta_played,
                traj_delta_best=traj_delta_best,
                r_t=r_t,
                policy_prob=policy_prob,
                policy_top_move=policy_top_move,
                policy_top_prob=policy_top_prob,
                policy_prob_source=policy_prob_source,
                is_unconventional=is_unconventional,
                generalist_policy_prob=generalist_policy_prob,
                generalist_top_move=generalist_top_move,
                generalist_self_assessed=generalist_self_assessed,
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

        Stages 1–2: heuristic-only — ply where h(t) drops most steeply.
        Stage 5 upgrades this to the full Malom → Sentinel+Heuristic → Heuristic hierarchy.
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
