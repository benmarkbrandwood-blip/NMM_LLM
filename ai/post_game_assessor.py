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

from dataclasses import dataclass, field
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

# Stage 5 — turning-point severity and labels (keyed by (wdl_before, wdl_after))
_MALOM_SEVERITY: dict = {("W", "L"): 3, ("D", "L"): 2, ("W", "D"): 1}
_MALOM_TRANSITION_LABEL: dict = {
    ("W", "L"): "win_to_loss",
    ("D", "L"): "draw_to_loss",
    ("W", "D"): "win_to_draw",
}

# Poor-candidate regret thresholds (calibrate from validation data)
_R_H_POOR_THRESHOLD  = 0.30  # when sentinel also confirms
_R_H_SOLO_THRESHOLD  = 0.50  # when only heuristic is available
_R_S_POOR_THRESHOLD  = 0.20  # sentinel r_s confirmation floor

# Tier 3 turning-point detection: skip early placement plies where the full
# evaluator has high positional variance but the strategic consequence is low.
_TP_MIN_PLY = 6

from game.board import BoardState
from game.rules import get_all_legal_moves, get_game_phase
from ai.game_ai import GameAI
from ai.heuristics import evaluate, evaluate_v2  # evaluate used for rich curve (Fix 1)


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

    # Policy quality divergence (Stage 5b)
    policy_pref_delta: Optional[float] = None        # pref_prob − teacher_prob; negative = weak human choice

    # Horizon search delta (Stage 5c)
    horizon_delta: Optional[float] = None            # score_shallow − score_deep; positive = short-sighted
    horizon_shallow_score: Optional[float] = None    # normalised move score at shallow depth
    horizon_deep_score: Optional[float] = None       # normalised move score at deep depth (= score_played)

    # Suspicious-position deep re-score (Fix 3)
    r_h_deep: Optional[float] = None    # r_h at deep_depth search; replaces r_h for turning-point selection
    deep_scored: bool = False           # True when this ply was re-scored at deep_depth

    # Mobility signal
    legal_move_count: int = 0           # legal moves available to mover BEFORE this move


@dataclass
class PostGameAnnotation:
    moves: list[MoveAnnotation]
    heuristic_curve: list[float]           # heuristic_score_white at each ply
    sentinel_curve: list[Optional[float]]  # None entries when Sentinel unavailable
    turning_point_ply: Optional[int]
    turning_point_quality: str             # e.g. "r_h:0.712", "r_h+r_s:0.712", or "win_to_loss"
    turning_point_oracle: str              # "malom_full"|"retrograde_wdl"|"sentinel+heuristic"|"heuristic"
    opening_name: Optional[str]
    # Ranked list of (ply, quality_str, oracle_source) — includes the primary above.
    turning_points: list[tuple[int, str, str]] = field(default_factory=list)


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
        pref_advisor: Optional["HumanMovePolicyAdvisor"] = None,
        policy_elo_band: str = "all",
        generalist: Optional["GeneralistAgent"] = None,
        r_h_threshold: float = float("inf"),
        r_s_threshold: float = float("inf"),
        r_h_solo_threshold: float = float("inf"),
        shallow_depth: Optional[int] = None,
        deep_depth: Optional[int] = None,
        threat_delta_threshold: float = 200.0,
    ) -> None:
        self._ai = GameAI(color="W", difficulty=difficulty)
        self._ai.max_search_depth = depth
        if shallow_depth is not None:
            self._ai_shallow = GameAI(color="W", difficulty=difficulty)
            self._ai_shallow.max_search_depth = shallow_depth
        else:
            self._ai_shallow = None
        if deep_depth is not None:
            self._ai_deep = GameAI(color="W", difficulty=difficulty)
            self._ai_deep.max_search_depth = deep_depth
        else:
            self._ai_deep = None
        self._threat_delta_threshold = threat_delta_threshold
        self._sentinel = sentinel
        self._gap_net = gap_net
        self._malom_db = malom_db
        self._trajectory_db = trajectory_db
        self._policy_advisor = policy_advisor
        self._pref_advisor = pref_advisor
        self._policy_elo_band = policy_elo_band
        self._generalist = generalist
        self._r_h_threshold = r_h_threshold
        self._r_s_threshold = r_s_threshold
        self._r_h_solo_threshold = r_h_solo_threshold

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

            legal_move_count = len(get_all_legal_moves(board))
            scored = self._ai.assess_position(board)

            board_after = board.apply_move(played_move)
            # Fix 1: full evaluate() captures fork threats, two-piece configs, mobility
            # squeeze — patterns that precede captures by 1-3 plies.  evaluate_v2 (bare-
            # bones) only sees piece counts and stays flat until terminal captures occur.
            heuristic_score_white = float(evaluate(board_after, "W"))

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

            # ── Poor-candidate flagging (regret thresholds, Stage 5) ─────────
            if (quality == "clean" and oracle_source == "none"
                    and abstained_reason != "already_losing"):
                if r_s is not None:
                    if r_h > self._r_h_threshold and r_s > self._r_s_threshold:
                        quality = "poor_candidate"
                elif r_h > self._r_h_solo_threshold:
                    quality = "poor_candidate"

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

            # ── Policy quality divergence (Stage 5b) ─────────────────────────
            policy_pref_delta: Optional[float] = None

            if (self._pref_advisor is not None
                    and self._policy_advisor is not None
                    and candidates
                    and policy_prob is not None):
                pref_probs = self._pref_advisor.probs(board, candidates,
                                                      self._policy_elo_band)
                if len(pref_probs) > 0:
                    played_idx = _find_played_idx(candidates, played_move)
                    pref_prob_val = float(pref_probs[played_idx])
                    policy_pref_delta = pref_prob_val - policy_prob

            # ── Horizon search delta (Stage 5c) ──────────────────────────────
            horizon_delta: Optional[float] = None
            horizon_shallow_score: Optional[float] = None
            horizon_deep_score: Optional[float] = None

            if self._ai_shallow is not None:
                shallow_scored = self._ai_shallow.assess_position(board)
                if shallow_scored:
                    s_all = [s for _, s in shallow_scored]
                    s_lo, s_hi = min(s_all), max(s_all)
                    move_key = (played_move.get("from"), played_move["to"],
                                played_move.get("capture"))
                    shallow_raw = next(
                        (s for m, s in shallow_scored
                         if (m.get("from"), m["to"], m.get("capture")) == move_key),
                        None,
                    )
                    if s_hi == s_lo:
                        horizon_shallow_score = 1.0
                    elif shallow_raw is None:
                        horizon_shallow_score = 0.0
                    else:
                        horizon_shallow_score = (shallow_raw - s_lo) / (s_hi - s_lo)
                    horizon_deep_score = score_played_norm
                    horizon_delta = horizon_shallow_score - horizon_deep_score

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
                policy_pref_delta=policy_pref_delta,
                horizon_delta=horizon_delta,
                horizon_shallow_score=horizon_shallow_score,
                horizon_deep_score=horizon_deep_score,
                legal_move_count=legal_move_count,
            ))
            board = board_after

        heuristic_curve = [a.heuristic_score_white for a in annotations]
        sentinel_curve: list[Optional[float]] = [a.sentinel_score_white for a in annotations]

        # Fix 3: deep re-score suspicious positions before turning-point selection.
        if self._ai_deep is not None:
            self._deep_rescore_suspicious(annotations, moves_raw, heuristic_curve)

        tp_ply, tp_quality, tp_oracle, tp_list = self._detect_turning_point(annotations)

        return PostGameAnnotation(
            moves=annotations,
            heuristic_curve=heuristic_curve,
            sentinel_curve=sentinel_curve,
            turning_point_ply=tp_ply,
            turning_point_quality=tp_quality,
            turning_point_oracle=tp_oracle,
            opening_name=opening_name,
            turning_points=tp_list,
        )

    def _detect_turning_point(
        self, annotations: list[MoveAnnotation]
    ) -> tuple[Optional[int], str, str, list[tuple[int, str, str]]]:
        """Return (primary_ply, quality_str, oracle, ranked_list) for turning points.

        Ranked list holds up to 3 entries (ply, quality, oracle), ordered by impact.

        Three-tier hierarchy:
        1. Malom confirmed_poor — ranked by WDL severity, tie-break by r_h.
        2. Sentinel + Heuristic combined drop Δh + Δs (curve-drop).
        3. Heuristic-only: argmax(r_h_deep if available, else r_h) — Fix 2/3.
           Uses the worst decision rather than the steepest post-hoc curve collapse,
           so errors upstream of a capture are found before the terminal drop.
        """
        if len(annotations) < 2:
            return None, "", "heuristic", []

        # Tier 1 — Malom-confirmed poor moves
        malom_poor = [
            a for a in annotations
            if a.quality == "confirmed_poor"
            and a.oracle_source == "malom_full"
            and a.wdl_before is not None and a.wdl_after is not None
        ]
        if malom_poor:
            ranked = sorted(
                malom_poor,
                key=lambda a: (_MALOM_SEVERITY.get((a.wdl_before, a.wdl_after), 0), a.r_h),
                reverse=True,
            )[:3]
            top = ranked[0]
            label = _MALOM_TRANSITION_LABEL.get((top.wdl_before, top.wdl_after), "unknown")
            ranked_list = [
                (a.ply, _MALOM_TRANSITION_LABEL.get((a.wdl_before, a.wdl_after), "unknown"), "malom_full")
                for a in ranked
            ]
            return top.ply, label, "malom_full", ranked_list

        # Tier 2 — Sentinel + Heuristic combined curve drop
        has_sentinel = any(a.sentinel_score_white is not None for a in annotations)
        if has_sentinel:
            drops: list[tuple[float, int]] = []
            for i in range(1, len(annotations)):
                dh = (annotations[i - 1].heuristic_score_white
                      - annotations[i].heuristic_score_white)
                s_prev = annotations[i - 1].sentinel_score_white
                s_curr = annotations[i].sentinel_score_white
                ds = (s_prev - s_curr) if (s_prev is not None and s_curr is not None) else 0.0
                drops.append((dh + ds, i))
            drops.sort(reverse=True)
            top3 = [idx for (drop, idx) in drops[:3] if drop > 0.0]
            if not top3:
                return None, "", "sentinel+heuristic", []
            ranked_list = []
            for idx in top3:
                ann = annotations[idx]
                r_s_val = ann.r_s if ann.r_s is not None else 0.0
                ranked_list.append((ann.ply, f"r_h+r_s:{ann.r_h + r_s_val:.3f}", "sentinel+heuristic"))
            primary = ranked_list[0]
            return primary[0], primary[1], primary[2], ranked_list

        # Tier 3 — Heuristic-only: top-3 by effective r_h, skipping early placement.
        # Fix 2: use decision quality (r_h) rather than post-hoc curve collapse.
        # Fix 3: prefer r_h_deep (deep re-score) when available on suspicious plies.
        # _TP_MIN_PLY: skip early placement plies where the richer evaluator has
        # high positional variance but low actual game consequence.
        candidates: list[tuple[float, int]] = []
        for i, ann in enumerate(annotations):
            if i < _TP_MIN_PLY:
                continue
            val = ann.r_h_deep if ann.r_h_deep is not None else ann.r_h
            if val > 0.0:
                candidates.append((val, i))
        candidates.sort(reverse=True)
        top3 = candidates[:3]
        if not top3:
            return None, "", "heuristic", []
        ranked_list = []
        for (val, i) in top3:
            ann = annotations[i]
            if ann.r_h_deep is not None:
                ranked_list.append((ann.ply, f"r_h_deep:{ann.r_h_deep:.3f}", "heuristic"))
            else:
                ranked_list.append((ann.ply, f"r_h:{ann.r_h:.3f}", "heuristic"))
        primary = ranked_list[0]
        return primary[0], primary[1], primary[2], ranked_list

    # ── Fix 3: suspicious-position deep re-score ──────────────────────────────

    _DEEP_RESCORE_MAX = 12  # cap on plies re-scored per game

    def _deep_rescore_suspicious(
        self,
        annotations: list[MoveAnnotation],
        moves_raw: list[dict],
        heuristic_curve: list[float],
    ) -> None:
        """Identify suspicious plies and re-score them at self._ai_deep depth.

        Two flagging sources (union, then capped at _DEEP_RESCORE_MAX):
        - Capture-lookback: plies t-1 and t-2 before each capture event.
          The defensive failure is upstream of the capture itself.
        - Threat-delta: consecutive curve steps where |Δh| ≥ threat_delta_threshold,
          catching fork threats / mill setups before the piece is actually taken.

        Capture-lookback plies are included first so they survive the cap.
        """
        n = len(annotations)
        if n < 2:
            return

        # ── Build suspicious set ──────────────────────────────────────────────
        capture_lookback: set[int] = set()
        for i, m in enumerate(moves_raw):
            if m.get("capture"):
                for lookback in (1, 2):
                    candidate = i - lookback
                    if 0 <= candidate < n:
                        capture_lookback.add(candidate)

        threat_delta: set[int] = set()
        for i in range(1, len(heuristic_curve)):
            if abs(heuristic_curve[i] - heuristic_curve[i - 1]) >= self._threat_delta_threshold:
                threat_delta.add(i)

        # Merge: capture-lookback first, then threat-delta; cap total
        ordered = sorted(capture_lookback) + sorted(threat_delta - capture_lookback)
        suspicious = sorted(set(ordered[: self._DEEP_RESCORE_MAX]))

        if not suspicious:
            return

        suspicious_set = set(suspicious)

        # ── Replay game, deep-score at flagged plies ──────────────────────────
        board = BoardState.new_game()
        for ply_idx, move_record in enumerate(moves_raw):
            played_move = {
                "from": move_record.get("from"),
                "to": move_record["to"],
                "capture": move_record.get("capture"),
            }
            if ply_idx in suspicious_set:
                scored_deep = self._ai_deep.assess_position(board)
                if scored_deep:
                    all_s = [s for _, s in scored_deep]
                    lo, hi = min(all_s), max(all_s)
                    move_key = (
                        played_move.get("from"),
                        played_move["to"],
                        played_move.get("capture"),
                    )
                    played_raw = next(
                        (s for m, s in scored_deep
                         if (m.get("from"), m["to"], m.get("capture")) == move_key),
                        None,
                    )
                    if hi != lo:
                        score_deep = (
                            (played_raw - lo) / (hi - lo)
                            if played_raw is not None
                            else 0.0
                        )
                        annotations[ply_idx].r_h_deep = max(0.0, 1.0 - score_deep)
                        annotations[ply_idx].deep_scored = True
            board = board.apply_move(played_move)
