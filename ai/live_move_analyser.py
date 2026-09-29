"""ai/live_move_analyser.py — Fast per-move signal aggregation for live commentary.

Computes human-preference, policy, generalist, GapNet, and Sentinel signals for a
single move in ~10–15 ms (generalist on alternating plies: ~60–90 ms).
Results are consumed by the Coordinator to decide whether to emit live commentary
and to ground LLM prompts with hard facts rather than speculation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from game.board import BoardState
    from ai.human_pref_advisor import HumanPrefAdvisor
    from ai.human_move_policy_advisor import HumanMovePolicyAdvisor

# ── Thresholds ────────────────────────────────────────────────────────────────

UNCONVENTIONAL_THRESHOLD  = 0.05   # policy_prob below this → unusual move
WEAK_SCORE_THRESHOLD      = 0.35   # score_norm below this → weak (heuristic)
STRONG_SCORE_THRESHOLD    = 0.75   # score_norm above this → strong (heuristic)
WEAK_PREF_THRESHOLD       = -0.20  # pref_delta below this → weak (human preference)
STRONG_PREF_THRESHOLD     =  0.15  # pref_delta above this → strong (human preference)
WEAK_SENTINEL_THRESHOLD   =  0.40  # sentinel quality below this → weak
RISKY_BLUNDER_THRESHOLD   =  0.65  # blunder_zone above this → risky position
GEN_DIVERGE_THRESHOLD     =  0.15  # generalist prob gap above this → significant divergence


@dataclass
class LiveMoveSignals:
    """Per-move signal bundle produced by LiveMoveAnalyser."""

    move_played: str
    color:       str     # "W" / "B"
    ply:         int

    # Heuristic rank (0.0 = worst legal move, 1.0 = best)
    score_norm:  float

    # Human-preference signals (None when advisor unavailable)
    pref_delta:   Optional[float]  # played_rank − top1_rank; 0.0 = best, negative = worse
    policy_prob:  Optional[float]  # teacher probability of the played move

    # Phase 3 signals (None when advisor unavailable / not computed this ply)
    generalist_top:   Optional[str]    # notation of generalist's top move if it diverges
    blunder_zone:     Optional[float]  # GapNet [0,1]: high → humans frequently blunder here
    sentinel_quality: Optional[float]  # Sentinel [0,1] quality of played move

    # Derived flags
    is_unconventional: bool   # policy_prob < UNCONVENTIONAL_THRESHOLD
    closed_mill:       bool
    captured:          bool
    is_strong:         bool
    is_weak:           bool
    is_risky:          bool   # high blunder_zone (GapNet)

    def facts_block(self) -> str:
        """One-paragraph FACTS string for the LLM prompt."""
        lines = ["MOVE FACTS (do not contradict):"]
        color_name = "White" if self.color == "W" else "Black"
        lines.append(f"  Move: {self.move_played}  ({color_name}, ply {self.ply})")
        if self.closed_mill:
            lines.append("  Mill closed — capture made")
        if self.pref_delta is not None:
            direction = "matches or beats" if self.pref_delta >= 0 else "falls below"
            lines.append(
                f"  Human preference rating: {self.pref_delta:+.2f} "
                f"({direction} the typical top human choice)"
            )
        if self.policy_prob is not None:
            pct = f"{self.policy_prob:.0%}"
            lines.append(f"  Move frequency (teacher model): {pct} of human games")
            if self.is_unconventional:
                lines.append("  Unusual choice — rarely played in this position")
        if self.sentinel_quality is not None:
            label = "poor" if self.sentinel_quality < WEAK_SENTINEL_THRESHOLD else (
                "strong" if self.sentinel_quality > 0.65 else "acceptable"
            )
            lines.append(f"  Sentinel quality: {self.sentinel_quality:.2f} ({label})")
        if self.blunder_zone is not None and self.blunder_zone > 0.45:
            lines.append(
                f"  Blunder-zone risk: {self.blunder_zone:.2f} "
                f"({'high' if self.blunder_zone > RISKY_BLUNDER_THRESHOLD else 'moderate'})"
            )
        if self.generalist_top is not None:
            lines.append(f"  Generalist AI preferred: {self.generalist_top}")
        if self.is_strong:
            lines.append("  Overall assessment: strong move")
        elif self.is_weak:
            lines.append("  Overall assessment: weak move — a better option existed")
        elif self.is_risky:
            lines.append("  Overall assessment: risky position — high blunder density detected")
        return "\n".join(lines)

    def badge_tuple(self) -> tuple[str, str, str]:
        """Return (label, detail, severity) for the live signal badge.
        severity: 'warning' | 'caution' | 'good' | 'neutral'
        An empty label means no badge to show.
        """
        if self.closed_mill:
            return ("Mill", "closed", "good")
        if self.is_weak:
            if self.pref_delta is not None and self.pref_delta < WEAK_PREF_THRESHOLD:
                return ("Pref", f"δ{self.pref_delta:+.2f}", "warning")
            if self.sentinel_quality is not None and self.sentinel_quality < WEAK_SENTINEL_THRESHOLD:
                return ("Sentinel", f"{self.sentinel_quality:.2f}", "warning")
            return ("Score", f"{self.score_norm:.2f}", "warning")
        if self.is_risky:
            return ("Blunder zone", f"{self.blunder_zone:.2f}", "caution")
        if self.is_unconventional and self.policy_prob is not None:
            return ("Unusual", f"{self.policy_prob:.0%}", "caution")
        if self.generalist_top is not None:
            return ("AI preferred", self.generalist_top, "caution")
        if self.is_strong:
            if self.pref_delta is not None and self.pref_delta >= STRONG_PREF_THRESHOLD:
                return ("Pref", f"δ{self.pref_delta:+.2f}", "good")
            return ("Score", f"{self.score_norm:.2f}", "good")
        return ("", "", "neutral")


def _move_key(m: dict) -> tuple:
    return (m.get("from"), m["to"], m.get("capture"))


def _move_str(move: dict) -> str:
    if move.get("from"):
        s = f"{move['from']}-{move['to']}"
    else:
        s = move["to"]
    if move.get("capture"):
        s += f"x{move['capture']}"
    return s


class LiveMoveAnalyser:
    """Stateless per-move signal computer.

    Pass the advisors that are available; None advisors skip their signal
    silently.  All computation is synchronous and takes < 15 ms
    (generalist runs on alternating plies only: ~60–90 ms on those turns).
    """

    def __init__(
        self,
        policy_advisor:    Optional["HumanMovePolicyAdvisor"] = None,
        pref_advisor:      Optional["HumanPrefAdvisor"]       = None,
        generalist_advisor = None,   # GeneralistAgent | SpecialistRouter | None
        gap_net            = None,   # GapNet (ValueNet) | None
        sentinel_advisor   = None,   # SentinelAdvisor | None
    ) -> None:
        self._policy      = policy_advisor
        self._pref        = pref_advisor
        self._generalist  = generalist_advisor
        self._gap_net     = gap_net
        self._sentinel    = sentinel_advisor

    @property
    def has_signals(self) -> bool:
        return (
            self._policy is not None
            or self._pref is not None
            or self._generalist is not None
            or self._gap_net is not None
            or self._sentinel is not None
        )

    def analyse(
        self,
        board_before: "BoardState",
        move:         dict,
        legal_moves:  list[dict],
        score_norm:   float,
        color:        str,
        ply:          int,
    ) -> LiveMoveSignals:
        """Compute signals for the given move and return a LiveMoveSignals."""
        from ai.coordinator import _move_str as _coord_move_str  # avoid circular at module level

        move_str  = _coord_move_str(move)
        captured  = bool(move.get("capture"))
        closed_mill = captured  # captures only happen when a mill is closed in NMM

        pref_delta     : Optional[float] = None
        policy_prob    : Optional[float] = None
        generalist_top : Optional[str]   = None
        blunder_zone   : Optional[float] = None
        sentinel_quality: Optional[float] = None

        if legal_moves:
            key = _move_key(move)
            move_idx = next(
                (i for i, m in enumerate(legal_moves) if _move_key(m) == key),
                None,
            )

            # ── HumanPrefAdvisor (fast, always) ──────────────────────────────
            if self._pref is not None:
                try:
                    ranks = self._pref.rank(board_before, legal_moves)
                    if move_idx is not None and ranks:
                        best_rank    = max(ranks)
                        played_rank  = ranks[move_idx]
                        pref_delta   = played_rank - best_rank   # ≤ 0
                except Exception:
                    pass

            # ── HumanMovePolicyAdvisor (fast, always) ─────────────────────────
            if self._policy is not None:
                try:
                    probs = self._policy.probs(board_before, legal_moves, "all")
                    if move_idx is not None:
                        policy_prob = float(probs[move_idx])
                except Exception:
                    pass

            # ── GapNet blunder-zone (fast, always) ────────────────────────────
            if self._gap_net is not None:
                try:
                    raw = self._gap_net.predict(board_before, color)
                    blunder_zone = (raw + 1.0) / 2.0   # tanh (-1,1) → [0,1]
                except Exception:
                    pass

            # ── SentinelAdvisor (fast, always) ────────────────────────────────
            if self._sentinel is not None and getattr(self._sentinel, "is_loaded", lambda: False)():
                try:
                    advice = self._sentinel.advise(
                        board_before, legal_moves, color,
                        played_move_idx=move_idx if move_idx is not None else 0,
                    )
                    if advice is not None:
                        sentinel_quality = float(advice.played_move_quality)
                except Exception:
                    pass

            # ── Generalist divergence (heavy, alternating plies) ──────────────
            if self._generalist is not None and getattr(self._generalist, "is_loaded", lambda: False)():
                if ply % 2 == 1:   # odd plies only
                    try:
                        probs_gen = self._generalist.score_moves(board_before, legal_moves, color)
                        if probs_gen is not None and len(probs_gen) == len(legal_moves):
                            best_idx = int(max(range(len(probs_gen)), key=lambda i: probs_gen[i]))
                            play_prob = probs_gen[move_idx] if move_idx is not None else 0.0
                            if (
                                best_idx != move_idx
                                and probs_gen[best_idx] - play_prob > GEN_DIVERGE_THRESHOLD
                            ):
                                generalist_top = _move_str(legal_moves[best_idx])
                    except Exception:
                        pass

        is_unconventional = (
            policy_prob is not None and policy_prob < UNCONVENTIONAL_THRESHOLD
        )
        is_strong = (
            score_norm >= STRONG_SCORE_THRESHOLD
            or (pref_delta is not None and pref_delta >= STRONG_PREF_THRESHOLD)
            or (sentinel_quality is not None and sentinel_quality > 0.65)
        )
        is_weak = (
            score_norm < WEAK_SCORE_THRESHOLD
            or (pref_delta is not None and pref_delta < WEAK_PREF_THRESHOLD)
            or (sentinel_quality is not None and sentinel_quality < WEAK_SENTINEL_THRESHOLD)
        )
        is_risky = (
            blunder_zone is not None and blunder_zone > RISKY_BLUNDER_THRESHOLD
        )

        return LiveMoveSignals(
            move_played=move_str,
            color=color,
            ply=ply,
            score_norm=score_norm,
            pref_delta=pref_delta,
            policy_prob=policy_prob,
            generalist_top=generalist_top,
            blunder_zone=blunder_zone,
            sentinel_quality=sentinel_quality,
            is_unconventional=is_unconventional,
            closed_mill=closed_mill,
            captured=captured,
            is_strong=is_strong,
            is_weak=is_weak,
            is_risky=is_risky,
        )
