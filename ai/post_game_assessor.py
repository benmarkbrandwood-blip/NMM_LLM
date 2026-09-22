"""
ai/post_game_assessor.py — Post-game per-ply analysis.

PostGameAssessor replays a completed game record and produces a
PostGameAnnotation with per-ply heuristic + sentinel scoring, score curves,
and turning-point detection.

Three-pass streaming architecture (assess_streaming / _run_game_assessment):
  Pass 1 — heuristic + sentinel + GapNet + policy + pref + horizon (~37s)
  Pass 2 — generalist AI scoring (~12s, concurrent with Stage 1 LLM)
  Pass 3 — Malom oracle adjudication (~5s)

Single-shot assess() is kept for backward compatibility with tools and tests.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

log = logging.getLogger(__name__)

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
    malom_best_alt: Optional[str] = None  # best Malom-approved alternative notation

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
    # Base display ply for ply_idx=0: 1 when White moves first, 2 when Black moves first.
    # For setup games this is ply_offset+1 (or +2 if needed to preserve parity).
    ply_base: int = 1


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


# ── Internal assessment context ───────────────────────────────────────────────

@dataclass
class _AssessContext:
    """Shared state threaded through the three assessment passes."""
    moves_raw: list[dict]
    # board_seq[i]   = board state BEFORE ply i
    # board_seq[i+1] = board state AFTER  ply i  (= board_seq[i].apply_move(move_i))
    board_seq: list          # list[BoardState], length N+1
    ai_color: Optional[str]
    opening_name: Optional[str]
    # Base display ply: display_ply = ply_idx + ply_base.  Accounts for both
    # setup-game pre-placed pieces and whoever makes the first recorded move.
    ply_base: int = 1
    # Filled by _pass_heuristic; consumed by _pass_generalist
    candidates_per_ply: list = field(default_factory=list)   # list[list[dict]]
    played_idx_per_ply: list = field(default_factory=list)   # list[int]
    # Per-component timing accumulators (filled by pass methods)
    t_heuristic: float = 0.0
    t_sentinel: float = 0.0
    t_gapnet: float = 0.0
    t_malom: float = 0.0
    t_policy: float = 0.0
    t_pref: float = 0.0
    t_generalist: float = 0.0
    t_horizon: float = 0.0
    t_deep_rescore: float = 0.0


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

    # ── Public API ────────────────────────────────────────────────────────────

    def assess(self, game_record: dict) -> PostGameAnnotation:
        """Full single-shot assessment. Backward-compatible; used by tools and tests."""
        t_start = time.perf_counter()
        ctx = self._prepare(game_record)
        annotations = self._pass_heuristic(ctx)
        self._pass_generalist(ctx, annotations)
        self._pass_oracle(ctx, annotations)
        result = self._finalize(ctx, annotations)
        t_total = time.perf_counter() - t_start
        log.info(
            "Assessment timing (%d plies) — total: %.1fs | heuristic: %.1fs | "
            "sentinel: %.1fs | gapnet: %.1fs | malom: %.1fs | policy: %.1fs | "
            "pref: %.1fs | generalist: %.1fs | horizon: %.1fs | deep_rescore: %.1fs",
            len(annotations), t_total,
            ctx.t_heuristic, ctx.t_sentinel, ctx.t_gapnet, ctx.t_malom,
            ctx.t_policy, ctx.t_pref, ctx.t_generalist, ctx.t_horizon, ctx.t_deep_rescore,
        )
        return result

    def partial_annotation(
        self, ctx: _AssessContext, annotations: list[MoveAnnotation]
    ) -> PostGameAnnotation:
        """Build a PostGameAnnotation from the current pass state.

        Runs turning-point detection but skips deep-rescore; safe to call
        between passes for incremental stage messages.
        """
        heuristic_curve = [a.heuristic_score_white for a in annotations]
        sentinel_curve: list[Optional[float]] = [a.sentinel_score_white for a in annotations]
        tp_ply, tp_quality, tp_oracle, tp_list = self._detect_turning_point(annotations)
        return PostGameAnnotation(
            moves=annotations,
            heuristic_curve=heuristic_curve,
            sentinel_curve=sentinel_curve,
            turning_point_ply=tp_ply,
            turning_point_quality=tp_quality,
            turning_point_oracle=tp_oracle,
            opening_name=ctx.opening_name,
            turning_points=tp_list,
            ply_base=ctx.ply_base,
        )

    # ── Internal passes ───────────────────────────────────────────────────────

    def _prepare(self, game_record: dict) -> _AssessContext:
        """Build board sequence and extract metadata. O(N) board replays; no AI calls."""
        moves_raw = game_record.get("moves", [])

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

        setup_fen = game_record.get("setup_fen")
        if setup_fen:
            board = BoardState.from_fen_string(setup_fen)
            setup_offset = board.pieces_placed["W"] + board.pieces_placed["B"]
        else:
            board = BoardState.new_game()
            setup_offset = 0

        board_seq: list[BoardState] = [board]
        for move_record in moves_raw:
            played_move = {
                "from": move_record.get("from"),
                "to": move_record["to"],
                "capture": move_record.get("capture"),
            }
            board = board.apply_move(played_move)
            board_seq.append(board)

        # Compute ply_base: the display ply for ply_idx=0.
        # White moves must land on odd display plies; Black on even.
        # The nominal base after setup_offset is setup_offset+1, but we may need
        # to add 1 more if the first recorded move is Black and setup_offset is even
        # (or White and setup_offset is odd).
        first_color = moves_raw[0].get("color", "W") if moves_raw else "W"
        expected_parity = 1 if first_color == "W" else 0  # 1=odd for White, 0=even for Black
        ply_base = setup_offset + 1
        if ply_base % 2 != expected_parity:
            ply_base += 1

        return _AssessContext(
            moves_raw=moves_raw,
            board_seq=board_seq,
            ai_color=ai_color,
            opening_name=opening_name,
            ply_base=ply_base,
        )

    def _pass_heuristic(self, ctx: _AssessContext) -> list[MoveAnnotation]:
        """Pass 1: heuristic + sentinel + GapNet + policy + pref + horizon.

        Builds the full candidate list per ply and stores it in ctx for Pass 2.
        Sets quality='poor_candidate' based on heuristic/sentinel thresholds only
        (no Malom yet — Pass 3 will override with confirmed verdicts).
        """
        annotations: list[MoveAnnotation] = []

        for ply_idx, move_record in enumerate(ctx.moves_raw):
            board      = ctx.board_seq[ply_idx]
            board_after = ctx.board_seq[ply_idx + 1]

            played_move = {
                "from": move_record.get("from"),
                "to":   move_record["to"],
                "capture": move_record.get("capture"),
            }
            color    = move_record.get("color", board.turn)
            phase    = get_game_phase(board, color)
            notation = move_record.get("notation") or _move_notation(played_move)

            legal_move_count = len(get_all_legal_moves(board))

            # ── Heuristic ─────────────────────────────────────────────────────
            _t0 = time.perf_counter()
            scored = self._ai.assess_position(board)
            ctx.t_heuristic += time.perf_counter() - _t0

            # Fix 1: full evaluate() captures fork threats, two-piece configs, mobility
            # squeeze — patterns that precede captures by 1-3 plies.
            heuristic_score_white = float(evaluate(board_after, "W"))

            if scored:
                all_s = [s for _, s in scored]
                lo, hi = min(all_s), max(all_s)
                best_alt_move     = scored[0][0]
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
                score_best_norm   = 1.0
                best_alt_notation = None

            r_h = max(0.0, 1.0 - score_played_norm)
            candidates = [m for m, _ in scored]
            played_idx = _find_played_idx(candidates, played_move)

            ctx.candidates_per_ply.append(candidates)
            ctx.played_idx_per_ply.append(played_idx)

            # ── Sentinel ──────────────────────────────────────────────────────
            sentinel_score_white: Optional[float] = None
            sentinel_played:      Optional[float] = None
            sentinel_best:        Optional[float] = None
            r_s:                  Optional[float] = None

            if self._sentinel is not None and candidates:
                _t0 = time.perf_counter()
                advice = self._sentinel.advise(board, candidates, color, played_idx)
                ctx.t_sentinel += time.perf_counter() - _t0
                if advice is not None:
                    sentinel_played = advice.played_move_quality
                    sentinel_best   = advice.best_available_quality
                    r_s             = advice.opportunity_gap
                    sentinel_score_white = (
                        sentinel_played if color == "W" else 1.0 - sentinel_played
                    )

            # ── GapNet blunder-zone density ───────────────────────────────────
            blunder_zone_score: Optional[float] = None
            if self._gap_net is not None:
                _t0 = time.perf_counter()
                raw = self._gap_net.predict(board, color)
                ctx.t_gapnet += time.perf_counter() - _t0
                blunder_zone_score = (raw + 1.0) / 2.0

            # ── Human policy signal ───────────────────────────────────────────
            policy_prob:        Optional[float] = None
            policy_top_move:    Optional[str]   = None
            policy_top_prob:    Optional[float] = None
            policy_prob_source: Optional[str]   = None
            is_unconventional = False

            if self._policy_advisor is not None and candidates:
                _t0 = time.perf_counter()
                probs = self._policy_advisor.probs(board, candidates, self._policy_elo_band)
                ctx.t_policy += time.perf_counter() - _t0
                if len(probs) > 0:
                    policy_prob       = float(probs[played_idx])
                    top_idx           = int(np.argmax(probs))
                    policy_top_move   = _move_notation(candidates[top_idx])
                    policy_top_prob   = float(probs[top_idx])
                    policy_prob_source = "learned"
                    if policy_top_prob > 0 and (
                        policy_prob / policy_top_prob < 0.05 or policy_prob < 0.08
                    ):
                        is_unconventional = True

            # ── Policy quality divergence (pref vs teacher) ───────────────────
            policy_pref_delta: Optional[float] = None
            if (self._pref_advisor is not None
                    and self._policy_advisor is not None
                    and candidates
                    and policy_prob is not None):
                _t0 = time.perf_counter()
                pref_probs = self._pref_advisor.probs(board, candidates, self._policy_elo_band)
                ctx.t_pref += time.perf_counter() - _t0
                if len(pref_probs) > 0:
                    pref_prob_val     = float(pref_probs[played_idx])
                    policy_pref_delta = pref_prob_val - policy_prob

            # ── Horizon search delta ──────────────────────────────────────────
            horizon_delta:         Optional[float] = None
            horizon_shallow_score: Optional[float] = None
            horizon_deep_score:    Optional[float] = None

            if self._ai_shallow is not None:
                _t0 = time.perf_counter()
                shallow_scored = self._ai_shallow.assess_position(board)
                ctx.t_horizon += time.perf_counter() - _t0
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
                    horizon_delta      = horizon_shallow_score - horizon_deep_score

            # ── Poor-candidate flagging (heuristic-only; Malom overrides in Pass 3) ──
            quality = "clean"
            if r_s is not None:
                if r_h > self._r_h_threshold and r_s > self._r_s_threshold:
                    quality = "poor_candidate"
            elif r_h > self._r_h_solo_threshold:
                quality = "poor_candidate"

            generalist_self_assessed = (ctx.ai_color is not None and color == ctx.ai_color)

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
                policy_prob=policy_prob,
                policy_top_move=policy_top_move,
                policy_top_prob=policy_top_prob,
                policy_prob_source=policy_prob_source,
                is_unconventional=is_unconventional,
                policy_pref_delta=policy_pref_delta,
                horizon_delta=horizon_delta,
                horizon_shallow_score=horizon_shallow_score,
                horizon_deep_score=horizon_deep_score,
                legal_move_count=legal_move_count,
                quality=quality,
                generalist_self_assessed=generalist_self_assessed,
            ))

        return annotations

    def _pass_generalist(
        self, ctx: _AssessContext, annotations: list[MoveAnnotation]
    ) -> None:
        """Pass 2: generalist AI scoring. Mutates annotations in place."""
        if self._generalist is None:
            return
        for ply_idx, ann in enumerate(annotations):
            candidates = ctx.candidates_per_ply[ply_idx]
            if not candidates:
                continue
            board      = ctx.board_seq[ply_idx]
            played_idx = ctx.played_idx_per_ply[ply_idx]
            _t0 = time.perf_counter()
            g_scores = self._generalist.score_moves(
                board, candidates, ann.color, sim_ply_depth=5
            )
            ctx.t_generalist += time.perf_counter() - _t0
            if g_scores is not None and len(g_scores) == len(candidates):
                ann.generalist_policy_prob = float(g_scores[played_idx])
                top_idx = int(max(range(len(g_scores)), key=lambda i: g_scores[i]))
                ann.generalist_top_move    = _move_notation(candidates[top_idx])

    def _pass_oracle(
        self, ctx: _AssessContext, annotations: list[MoveAnnotation]
    ) -> None:
        """Pass 3: trajectory + Malom adjudication. Mutates annotations in place.

        Malom overrides quality when it has a verdict; on abstention the Pass 1
        poor_candidate flag is preserved unchanged.
        """
        for ply_idx, move_record in enumerate(ctx.moves_raw):
            board      = ctx.board_seq[ply_idx]
            ann        = annotations[ply_idx]
            played_move = {
                "from":    move_record.get("from"),
                "to":      move_record["to"],
                "capture": move_record.get("capture"),
            }

            # ── Trajectory signal ─────────────────────────────────────────────
            if self._trajectory_db is not None:
                candidates = ctx.candidates_per_ply[ply_idx]
                if candidates:
                    hints = self._trajectory_db.query(board, ann.color)
                    if hints:
                        ann.traj_delta_best   = max(hints.values())
                        ann.traj_delta_played = hints.get(ann.move_played)
                        if ann.traj_delta_played is not None:
                            ann.r_t = ann.traj_delta_best - ann.traj_delta_played

            # ── Malom adjudication ────────────────────────────────────────────
            if self._malom_db is None or not self._malom_db.is_available():
                continue

            _t0 = time.perf_counter()
            parent_val = self._malom_db.query_value(board)
            if parent_val is None:
                ann.abstained_reason = "parent_value_unavailable"
            else:
                wdl_before = parent_val.outcome
                ann.wdl_before = wdl_before
                if wdl_before == "L":
                    ann.abstained_reason = "already_losing"
                    ann.quality          = "clean"
                else:
                    result = self._malom_db.query_regret(board, played_move)
                    if not result.available:
                        ann.wdl_before       = None
                        ann.abstained_reason = result.unavailable_reason
                    else:
                        ann.wdl_after  = result.omv.outcome
                        transition     = result.wdl_transition
                        if transition in _MALOM_CONFIRMED_POOR:
                            ann.oracle_source = "malom_full"
                            ann.quality       = "confirmed_poor"
                            if (result.best_legal_move is not None
                                    and result.best_legal_move != played_move):
                                ann.malom_best_alt = _move_notation(result.best_legal_move)
                        elif transition in _MALOM_CLEAN:
                            ann.oracle_source = "malom_full"
                            ann.quality       = "clean"
                        else:
                            # label_inconsistency or unexpected — fail closed
                            ann.wdl_before       = None
                            ann.wdl_after        = None
                            ann.abstained_reason = f"malom_{transition}"
            ctx.t_malom += time.perf_counter() - _t0

    def _finalize(
        self, ctx: _AssessContext, annotations: list[MoveAnnotation]
    ) -> PostGameAnnotation:
        """Deep-rescore suspicious positions (if configured) and detect turning points."""
        heuristic_curve = [a.heuristic_score_white for a in annotations]
        sentinel_curve: list[Optional[float]] = [a.sentinel_score_white for a in annotations]

        _t0 = time.perf_counter()
        if self._ai_deep is not None:
            self._deep_rescore_suspicious(annotations, ctx.moves_raw, heuristic_curve)
        ctx.t_deep_rescore = time.perf_counter() - _t0

        tp_ply, tp_quality, tp_oracle, tp_list = self._detect_turning_point(annotations)
        return PostGameAnnotation(
            moves=annotations,
            heuristic_curve=heuristic_curve,
            sentinel_curve=sentinel_curve,
            turning_point_ply=tp_ply,
            turning_point_quality=tp_quality,
            turning_point_oracle=tp_oracle,
            opening_name=ctx.opening_name,
            turning_points=tp_list,
            ply_base=ctx.ply_base,
        )

    def _detect_turning_point(
        self, annotations: list[MoveAnnotation]
    ) -> tuple[Optional[int], str, str, list[tuple[int, str, str]]]:
        """Return (primary_ply, quality_str, oracle, ranked_list) for turning points.

        Ranked list merges Tier 1 (Malom) and Tier 2 (Sentinel+H) when both are
        available — up to 3 Malom entries followed by up to 3 Sentinel entries.
        The primary TP is always the top Malom entry when Malom has findings.

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

        malom_ranked: list[tuple[int, str, str]] = []
        if malom_poor:
            sorted_malom = sorted(
                malom_poor,
                key=lambda a: (_MALOM_SEVERITY.get((a.wdl_before, a.wdl_after), 0), a.r_h),
                reverse=True,
            )[:3]
            malom_ranked = [
                (a.ply, _MALOM_TRANSITION_LABEL.get((a.wdl_before, a.wdl_after), "unknown"), "malom_full")
                for a in sorted_malom
            ]

        # Tier 2 — Sentinel + Heuristic combined curve drop (always computed)
        sentinel_ranked: list[tuple[int, str, str]] = []
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
            for idx in top3:
                ann = annotations[idx]
                r_s_val = ann.r_s if ann.r_s is not None else 0.0
                sentinel_ranked.append((ann.ply, f"r_h+r_s:{ann.r_h + r_s_val:.3f}", "sentinel+heuristic"))

        # If Malom found TPs, return merged list with Malom first
        if malom_ranked:
            top = sorted_malom[0]
            label = _MALOM_TRANSITION_LABEL.get((top.wdl_before, top.wdl_after), "unknown")
            ranked_list = malom_ranked + sentinel_ranked
            return top.ply, label, "malom_full", ranked_list

        # Sentinel-only fallback
        if sentinel_ranked:
            primary = sentinel_ranked[0]
            return primary[0], primary[1], primary[2], sentinel_ranked

        # Tier 3 — Heuristic-only: top-3 by effective r_h, skipping early placement.
        # Fix 2: use decision quality (r_h) rather than post-hoc curve collapse.
        # Fix 3: prefer r_h_deep (deep re-score) when available on suspicious plies.
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

        ordered = sorted(capture_lookback) + sorted(threat_delta - capture_lookback)
        suspicious = sorted(set(ordered[: self._DEEP_RESCORE_MAX]))

        if not suspicious:
            return

        suspicious_set = set(suspicious)

        board = BoardState.new_game()
        for ply_idx, move_record in enumerate(moves_raw):
            played_move = {
                "from":    move_record.get("from"),
                "to":      move_record["to"],
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
                        annotations[ply_idx].r_h_deep  = max(0.0, 1.0 - score_deep)
                        annotations[ply_idx].deep_scored = True
            board = board.apply_move(played_move)
