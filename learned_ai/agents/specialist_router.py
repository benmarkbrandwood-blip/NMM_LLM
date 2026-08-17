"""learned_ai/agents/specialist_router.py — three-specialist inference router.

Loads the three v2 phase specialists (opening, midgame, endgame) and routes
each move choice to the appropriate specialist based on game phase.

Drop-in replacement for `OverseerAdvisor`: same interface (`is_loaded`,
`score_moves`, `set_db`, etc.), so the web/app.py wiring for the "Overseer
player" toggle drives it unchanged.

Routing rule (matches ScaffoldedAgent.choose_move_for_phase):
  * placement phase             → opening specialist
  * move/fly + ≤5 own or opp    → endgame specialist
  * else                        → midgame specialist

At inference each specialist sees:
  * feat_matrix (k, 134) — base 62 + 12-ply lookahead (h/vn/sent/gap)
  * The specialist's forward pass is instant; wall time is dominated by the
    LookaheadAdvisor's 12-ply sentinel calls.

Checkpoint search:
  learned_ai/checkpoints/scaffolded/{s_open_v2,s_mid_v2,s_end_v2}/best.pt

Returns None if all three specialists fail to load — caller falls back to
the classical coordinator.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from game.board import BoardState
from game.rules import get_game_phase
from learned_ai.agents.positional_safety import (
    PositionalSafetyError,
    PositionalSafetyFilter,
)
from learned_ai.training.checkpoint_envelope import (
    is_checkpoint_envelope,
    load_checkpoint,
)

log = logging.getLogger("nmm.specialist_router")


def _move_key(mv: dict) -> tuple:
    return (mv.get("from"), mv.get("to"), mv.get("capture"))


def _load_spec_model(path: Path):
    """Load a ScaffoldedPolicyNet checkpoint. Returns (model, cfg) or (None, {})."""
    if not path.exists():
        return None, {}
    try:
        from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
        if is_checkpoint_envelope(path):
            envelope = load_checkpoint(path)
            cfg = dict(envelope.payload.trainer_state["model_config"])
            state = envelope.payload.model_state
        else:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
            cfg = ckpt.get("model_config", {})
            state = ckpt.get("model") or ckpt
        model = ScaffoldedPolicyNet.from_config(cfg)
        model.load_state_dict(state)
        model.eval()
        return model, cfg
    except Exception as e:
        log.warning("Specialist load failed at %s: %s", path, e)
        return None, {}


class SpecialistRouter:
    """Phase-router over three v2 specialists.  API-compatible with OverseerAdvisor."""

    def __init__(
        self,
        spec_open,
        spec_mid,
        spec_end,
        ckpt_paths: dict[str, str],
        sentinel_advisor=None,
        db=None,
        value_net=None,
        gap_net=None,
        endgame_db=None,
        human_db=None,
        gameai=None,
        lookahead_advisor_open=None,
        lookahead_advisor_mid=None,
        lookahead_advisor_end=None,
        specialist_db=None,
        runtime_quarantine=None,
    ) -> None:
        self._spec_open = spec_open
        self._spec_mid  = spec_mid
        self._spec_end  = spec_end
        self._ckpt_paths = ckpt_paths
        self._sentinel  = sentinel_advisor
        self._db        = db          # Malom perfect DB (compat with Overseer set_db)
        self._value_net = value_net
        self._gap_net   = gap_net
        self._endgame_db = endgame_db
        self._human_db  = human_db
        self._gameai    = gameai
        self._la_open   = lookahead_advisor_open
        self._la_mid    = lookahead_advisor_mid
        self._la_end    = lookahead_advisor_end
        self._specialist_db = specialist_db
        self._runtime_quarantine = runtime_quarantine
        self._positional_safety_filter: PositionalSafetyFilter | None = None
        self._positional_safety_disabled_reason = "not configured"
        self._positional_safety_lock = threading.Lock()
        self._positional_safety_query_lock = threading.Lock()
        self._positional_safety_counters = {
            "requests": 0,
            "interventions": 0,
            "runtime_failures": 0,
            "model_score_failures": 0,
            "unavailable_requests": 0,
        }
        self._positional_safety_last_error = ""
        self._positional_safety_last_decision: dict | None = None

    # ── OverseerAdvisor-compatible surface ────────────────────────────────────

    def is_loaded(self) -> bool:
        """True when at least one specialist is loaded (router still usable)."""
        return any(m is not None for m in (self._spec_open, self._spec_mid, self._spec_end))

    def set_sentinel(self, sentinel_advisor) -> None:
        self._sentinel = sentinel_advisor
        for la in (self._la_open, self._la_mid, self._la_end):
            if la is not None:
                la._sentinel = sentinel_advisor  # type: ignore[attr-defined]

    def set_db(self, db) -> None:
        self._db = db

    def configure_positional_safety(
        self,
        oracle,
        *,
        label_version: str,
        manifest_sha256: str,
        content_sha256: str | None = None,
    ) -> None:
        """Enable the production-only, position-level Malom final filter."""
        safety_filter = PositionalSafetyFilter(
            oracle,
            label_version=label_version,
            manifest_sha256=manifest_sha256,
            content_sha256=content_sha256,
        )
        with self._positional_safety_lock:
            self._positional_safety_filter = safety_filter
            self._positional_safety_disabled_reason = ""
            self._positional_safety_last_error = ""
        log.info(
            "Specialist A_pos filter enabled label=%s manifest=%s content=%s",
            label_version,
            manifest_sha256,
            content_sha256 or "unrecorded",
        )

    def disable_positional_safety(self, reason: str) -> None:
        """Disable specialist filtering with a diagnostic reason."""
        reason = reason.strip() or "unspecified startup failure"
        with self._positional_safety_lock:
            self._positional_safety_filter = None
            self._positional_safety_disabled_reason = reason
            self._positional_safety_last_error = reason
        log.error("Specialist A_pos filter disabled: %s", reason)

    def positional_safety_status(self) -> dict:
        """Return JSON-safe runtime diagnostics for the product status API."""
        with self._positional_safety_lock:
            safety_filter = self._positional_safety_filter
            return {
                "configured": safety_filter is not None,
                "enabled": safety_filter is not None,
                "mode": "A_pos",
                "positional_only": True,
                "history_aware": False,
                "label_version": (
                    safety_filter.label_version if safety_filter is not None else None
                ),
                "manifest_sha256": (
                    safety_filter.manifest_sha256 if safety_filter is not None else None
                ),
                "content_sha256": (
                    safety_filter.content_sha256 if safety_filter is not None else None
                ),
                "disabled_reason": self._positional_safety_disabled_reason,
                **self._positional_safety_counters,
                "last_error": self._positional_safety_last_error,
                "last_decision": self._positional_safety_last_decision,
            }

    def set_value_net(self, value_net) -> None:
        self._value_net = value_net
        for la in (self._la_open, self._la_mid, self._la_end):
            if la is not None:
                la._value_net = value_net  # type: ignore[attr-defined]

    def set_human_db(self, human_db) -> None:
        self._human_db = human_db

    def set_gameai(self, gameai) -> None:
        self._gameai = gameai

    # ── continuous learning ───────────────────────────────────────────────────

    def record_game_result(self, game_record: dict) -> None:
        """Quarantine runtime evidence without mutating the trusted SpecialistDB."""
        if self._runtime_quarantine is None:
            log.warning("Runtime game was not recorded because quarantine is unavailable")
            return
        try:
            record_id = self._runtime_quarantine.append_game(
                game_record,
                source="web.specialist_router",
            )
            log.debug("Quarantined runtime game as %s", record_id)
        except Exception as e:
            log.warning("SpecialistRouter.record_game_result failed: %s", e)

    # ── routing ───────────────────────────────────────────────────────────────

    def _pick_specialist(self, board: BoardState, color: str):
        """Return (specialist_model, lookahead_advisor, phase_label)."""
        phase = get_game_phase(board, color)
        if phase == "place":
            return self._spec_open, self._la_open, "opening"
        own = int(board.pieces_on_board.get(color, 0))
        opp_color = "B" if color == "W" else "W"
        opp = int(board.pieces_on_board.get(opp_color, 0))
        if own <= 5 or opp <= 5:
            return self._spec_end, self._la_end, "endgame"
        return self._spec_mid, self._la_mid, "midgame"

    # ── inference ─────────────────────────────────────────────────────────────

    def score_moves(self, board: BoardState, candidates: list[dict], color: str) -> Optional[list[float]]:
        """Return per-candidate pick probabilities (sum to 1.0).

        Routes to the phase-appropriate specialist; falls back to whichever
        specialist is loaded if the preferred one is missing.

        v4: scores ALL legal moves via encode_position_with_lookahead (full-legal-moves
        mode).  The specialist is not limited to re-ranking a top-K subset.
        """
        if not candidates:
            return None
        try:
            from learned_ai.models.scaffolded_encoder import encode_position_with_lookahead

            spec, la, phase_label = self._pick_specialist(board, color)
            # Fallback ladder: preferred → any other loaded specialist
            if spec is None:
                for alt, alt_la in ((self._spec_mid, self._la_mid),
                                    (self._spec_end, self._la_end),
                                    (self._spec_open, self._la_open)):
                    if alt is not None:
                        spec, la = alt, alt_la
                        phase_label += "→fallback"
                        break
            if spec is None:
                return None

            enc = encode_position_with_lookahead(
                board, color,
                sentinel_advisor=self._sentinel,
                db=None,
                value_net=self._value_net,
                lookahead_advisor=la,
                specialist_db=self._specialist_db,
            )
            if enc is None or not enc.legal_moves:
                return None

            feat = torch.from_numpy(enc.feat_matrix).to(torch.float32)
            with torch.no_grad():
                probs = spec.policy_probs(feat)   # (k,)
            probs_np = probs.cpu().numpy()

            enc_key_to_idx = {_move_key(m): i for i, m in enumerate(enc.legal_moves)}
            result: list[float] = []
            for cand in candidates:
                idx = enc_key_to_idx.get(_move_key(cand))
                result.append(float(probs_np[idx]) if idx is not None and idx < len(probs_np) else 0.0)

            total = sum(result)
            if total > 1e-9:
                result = [v / total for v in result]
            return result

        except Exception as e:
            log.warning("SpecialistRouter.score_moves failed: %s", e, exc_info=True)
            return None

    def score_moves_positional_safe(
        self,
        board: BoardState,
        candidates: list[dict],
        color: str,
    ) -> Optional[list[float]]:
        """Score all legal moves, then retain only positional ``A_pos`` moves.

        Any unavailable or failed Malom query returns ``None`` after recording
        an error. The web caller may visibly fall back to its classical move,
        but this method never returns the unfiltered specialist argmax.
        """
        with self._positional_safety_lock:
            self._positional_safety_counters["requests"] += 1
            safety_filter = self._positional_safety_filter
            disabled_reason = self._positional_safety_disabled_reason
            if safety_filter is None:
                self._positional_safety_counters["unavailable_requests"] += 1
                self._positional_safety_last_error = disabled_reason
        if safety_filter is None:
            log.error(
                "Specialist A_pos request rejected because filter is disabled: %s",
                disabled_reason,
            )
            return None

        raw_scores = self.score_moves(board, candidates, color)
        if raw_scores is None:
            with self._positional_safety_lock:
                self._positional_safety_counters["model_score_failures"] += 1
                self._positional_safety_last_error = "specialist scoring failed"
            log.error("Specialist A_pos request rejected because scoring failed")
            return None

        try:
            with self._positional_safety_query_lock:
                filtered, decision = safety_filter.filter_scores(
                    board,
                    candidates,
                    raw_scores,
                )
        except PositionalSafetyError as exc:
            with self._positional_safety_lock:
                self._positional_safety_counters["runtime_failures"] += 1
                self._positional_safety_last_error = str(exc)
            log.error("Specialist A_pos failed closed: %s", exc, exc_info=True)
            return None
        except Exception as exc:
            detail = f"unexpected positional safety failure: {type(exc).__name__}"
            with self._positional_safety_lock:
                self._positional_safety_counters["runtime_failures"] += 1
                self._positional_safety_last_error = detail
            log.error("Specialist A_pos failed closed: %s", detail, exc_info=True)
            return None

        decision_record = asdict(decision)
        with self._positional_safety_lock:
            if decision.intervened:
                self._positional_safety_counters["interventions"] += 1
            self._positional_safety_last_error = ""
            self._positional_safety_last_decision = decision_record
        if decision.intervened:
            log.info(
                "Specialist A_pos intervention original=%s selected=%s "
                "tier=%s safe=%d/%d elapsed_ms=%.3f",
                decision.original_move,
                decision.selected_move,
                decision.parent_tier,
                decision.safe_move_count,
                decision.legal_move_count,
                decision.elapsed_ms,
            )
        return filtered


# ── generalist (single model, no phase routing) ───────────────────────────────

class GeneralistAgent:
    """Wraps a single s_gen_v2 ScaffoldedPolicyNet.  Same public interface as SpecialistRouter."""

    def __init__(self, model, la, sentinel_advisor=None, value_net=None, specialist_db=None):
        self._model    = model
        self._la       = la
        self._sentinel = sentinel_advisor
        self._value_net = value_net
        self._specialist_db = specialist_db
        self._gameai   = None
        self._db       = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def set_db(self, db) -> None:
        self._db = db

    def set_sentinel(self, sentinel_advisor) -> None:
        self._sentinel = sentinel_advisor
        if self._la is not None:
            self._la._sentinel = sentinel_advisor  # type: ignore[attr-defined]

    def set_value_net(self, value_net) -> None:
        self._value_net = value_net
        if self._la is not None:
            self._la._value_net = value_net  # type: ignore[attr-defined]

    def set_gameai(self, gameai) -> None:
        self._gameai = gameai

    def record_game_result(self, game_record: dict) -> None:
        pass  # generalist doesn't use specialist_db routing

    def score_moves(self, board: BoardState, candidates: list[dict], color: str) -> Optional[list[float]]:
        if not candidates or self._model is None:
            return None
        try:
            from learned_ai.models.scaffolded_encoder import encode_position_with_lookahead

            enc = encode_position_with_lookahead(
                board, color,
                sentinel_advisor=self._sentinel,
                db=None,
                value_net=self._value_net,
                lookahead_advisor=self._la,
                specialist_db=self._specialist_db,
            )
            if enc is None or not enc.legal_moves:
                return None

            feat = torch.from_numpy(enc.feat_matrix).to(torch.float32)
            with torch.no_grad():
                probs = self._model.policy_probs(feat)
            probs_np = probs.cpu().numpy()

            enc_key_to_idx = {_move_key(m): i for i, m in enumerate(enc.legal_moves)}
            result: list[float] = []
            for cand in candidates:
                idx = enc_key_to_idx.get(_move_key(cand))
                result.append(float(probs_np[idx]) if idx is not None and idx < len(probs_np) else 0.0)

            total = sum(result)
            if total > 1e-9:
                result = [v / total for v in result]
            return result
        except Exception as e:
            log.warning("GeneralistAgent.score_moves failed: %s", e, exc_info=True)
            return None


def load_generalist(
    ckpt_dir: Optional[Path] = None,
    sentinel_advisor=None,
    value_net=None,
    gap_net=None,
    human_db=None,
    specialist_db=None,
    ply_depth: int = 12,
) -> Optional[GeneralistAgent]:
    """Load the s_gen_v2 generalist checkpoint. Returns None if not found."""
    from learned_ai.models.lookahead_advisor import LookaheadAdvisor
    from learned_ai.agents.heuristic_agent import get_heuristic_evaluate

    root = Path(__file__).parent.parent.parent
    if ckpt_dir is None:
        ckpt_dir = root / "learned_ai" / "checkpoints" / "scaffolded"

    gen_path = ckpt_dir / "s_gen_v2" / "best.pt"
    m_gen, _ = _load_spec_model(gen_path)
    if m_gen is None:
        log.info("GeneralistAgent: no checkpoint at %s", gen_path)
        return None

    evaluate_fn = get_heuristic_evaluate()
    try:
        la = LookaheadAdvisor(
            sentinel=sentinel_advisor,
            evaluate_fn=evaluate_fn,
            value_net=value_net,
            gap_net=gap_net,
            human_db=human_db,
            use_sentinel=True,
            ply_depth=ply_depth,
        )
    except Exception as e:
        log.warning("GeneralistAgent LookaheadAdvisor init failed: %s", e)
        la = None

    log.info("GeneralistAgent loaded from %s ply_depth=%d", gen_path, ply_depth)
    return GeneralistAgent(
        model=m_gen,
        la=la,
        sentinel_advisor=sentinel_advisor,
        value_net=value_net,
        specialist_db=specialist_db,
    )


# ── loader ────────────────────────────────────────────────────────────────────

def load_specialist_router(
    ckpt_dir: Optional[Path] = None,
    sentinel_advisor=None,
    db=None,
    human_db=None,
    value_net=None,
    gap_net=None,
    specialist_db=None,
    runtime_quarantine=None,
    ply_depth: int = 12,
) -> Optional[SpecialistRouter]:
    """Load the three v2 specialists and their LookaheadAdvisors.

    Returns None only if ALL three specialist checkpoints fail to load.
    """
    from learned_ai.models.lookahead_advisor import LookaheadAdvisor
    from learned_ai.agents.heuristic_agent import get_heuristic_evaluate

    root = Path(__file__).parent.parent.parent
    if ckpt_dir is None:
        ckpt_dir = root / "learned_ai" / "checkpoints" / "scaffolded"

    open_path = ckpt_dir / "s_open_v2" / "best.pt"
    mid_path  = ckpt_dir / "s_mid_v2"  / "best.pt"
    end_path  = ckpt_dir / "s_end_v2"  / "best.pt"

    m_open, cfg_open = _load_spec_model(open_path)
    m_mid,  cfg_mid  = _load_spec_model(mid_path)
    m_end,  cfg_end  = _load_spec_model(end_path)

    if not any((m_open, m_mid, m_end)):
        log.info("SpecialistRouter: no v2 checkpoints found (searched %s, %s, %s)",
                 open_path, mid_path, end_path)
        return None

    evaluate_fn = get_heuristic_evaluate()

    def _mk_la(endgame_db_arg=None):
        try:
            return LookaheadAdvisor(
                sentinel=sentinel_advisor,
                evaluate_fn=evaluate_fn,
                value_net=value_net,
                gap_net=gap_net,
                human_db=human_db,
                use_sentinel=True,
                endgame_db=endgame_db_arg,
                ply_depth=ply_depth,
            )
        except Exception as e:
            log.warning("LookaheadAdvisor init failed: %s", e)
            return None

    la_open = _mk_la() if m_open is not None else None
    la_mid  = _mk_la() if m_mid  is not None else None
    la_end  = _mk_la(endgame_db_arg=db) if m_end is not None else None

    log.info("SpecialistRouter loaded — open=%s mid=%s end=%s ply_depth=%d",
             "OK" if m_open else "missing",
             "OK" if m_mid  else "missing",
             "OK" if m_end  else "missing",
             ply_depth)

    return SpecialistRouter(
        spec_open=m_open, spec_mid=m_mid, spec_end=m_end,
        ckpt_paths={
            "open": str(open_path) if m_open else "",
            "mid":  str(mid_path)  if m_mid  else "",
            "end":  str(end_path)  if m_end  else "",
        },
        sentinel_advisor=sentinel_advisor,
        db=db,
        value_net=value_net,
        gap_net=gap_net,
        endgame_db=db,
        human_db=human_db,
        specialist_db=specialist_db,
        runtime_quarantine=runtime_quarantine,
        lookahead_advisor_open=la_open,
        lookahead_advisor_mid=la_mid,
        lookahead_advisor_end=la_end,
    )
