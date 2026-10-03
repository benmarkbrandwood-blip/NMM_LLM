"""ai/coordinator.py — AI dialogue coordinator (GameAI ↔ MillsLLM)."""

from __future__ import annotations

import json
import math
import random
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.board import BoardState
    from ai.human_pref_advisor import HumanPrefAdvisor
    from ai.human_move_policy_advisor import HumanMovePolicyAdvisor

from ai.game_ai import GameAI
from ai.mills_llm import MillsLLM
from ai.memory_manager import MemoryManager
from ai.opening_book import Opening
from ai.opening_recognizer import OpeningRecognizer, INACTIVE_RESULT
from ai.endgame_recognizer import EndgameRecognizer, INACTIVE_ENDGAME
from ai.trajectory_db import TrajectoryDB
from ai.endgame_db import EndgameDB
from ai.board_symmetry import transform_notation as _transform_book_notation
from ai.move_guidance import (
    build_trajectory_hints,
    compute_force_book_early,
    format_trajectory_context,
    pick_target_opening,
    synthesize_opening_recognition,
)
from game.rules import get_all_legal_moves, get_game_phase


class Coordinator:
    def __init__(
        self,
        game_ai: GameAI,
        mills_llm: MillsLLM,
        memory: MemoryManager,
        think_time: float = 3.0,
        poor_move_threshold: float = 0.3,
        max_poor_move_comments: int = 5,
        opening_recognizer: OpeningRecognizer | None = None,
        endgame_recognizer: EndgameRecognizer | None = None,
        trajectory_db: TrajectoryDB | None = None,
        endgame_db: EndgameDB | None = None,
        vs_human: bool = True,
        human_color: str = "W",
        policy_advisor:        "HumanMovePolicyAdvisor | None" = None,
        pref_advisor:          "HumanPrefAdvisor | None"       = None,
        generalist_advisor     = None,   # GeneralistAgent | SpecialistRouter | None
        gap_net                = None,   # GapNet (ValueNet) | None
        sentinel_advisor       = None,   # SentinelAdvisor | None
        value_net              = None,   # ValueNet | PhaseValueNet | None
        llm_can_override_move: bool = True,
    ) -> None:
        self.game_ai = game_ai
        self.mills_llm = mills_llm
        self.memory = memory
        self.think_time = think_time
        self.poor_move_threshold = poor_move_threshold
        self.max_poor_move_comments = max_poor_move_comments
        self.opening_recognizer = opening_recognizer
        self.endgame_recognizer = endgame_recognizer
        self.trajectory_db = trajectory_db
        self.endgame_db = endgame_db
        self.vs_human = vs_human
        self.human_color = human_color

        from ai.live_move_analyser import LiveMoveAnalyser, HORIZON_THRESHOLD
        self._horizon_threshold = HORIZON_THRESHOLD
        # Shallow AI for horizon regret: 2-ply search, same weights as game_ai.
        _shallow = GameAI(color=game_ai.color, difficulty=game_ai.difficulty)
        _shallow.max_search_depth = 2
        self._live_analyser = LiveMoveAnalyser(
            policy_advisor=policy_advisor,
            pref_advisor=pref_advisor,
            generalist_advisor=generalist_advisor,
            gap_net=gap_net,
            sentinel_advisor=sentinel_advisor,
            shallow_ai=_shallow,
            value_net=value_net,
        )

        self.dialogue_log: list[dict] = []
        self._poor_move_count = 0
        self._general_comment_count = 0
        self._last_comment_turn = -2
        self._human_turn_num = 0
        self._turn_num = 0
        self._session_id = str(uuid.uuid4())
        self._game_moves: list[dict] = []
        self._endgame_state = INACTIVE_ENDGAME
        self._target_opening: Opening | None = None
        self._game_sym_idx: int = 0   # D4 symmetry applied to book moves this game
        self._last_novel_id: str | None = None   # set when an unnamed opening is saved
        self._dominant_turn_streak: int = 0
        self._session_signals: list = []   # rolling buffer of last 6 LiveMoveSignals (human moves)
        self.resignation_offered: bool = False
        self.last_thinking: str = ""   # plain-English label for the most recent AI move
        self.llm_can_override_move: bool = llm_can_override_move

    # ── Internal helpers ──────────────────────────────────────────────────────

    def emit(self, speaker: str, text: str, tag: str = "normal") -> None:
        if text:
            self.dialogue_log.append({"speaker": speaker, "text": text, "tag": tag})

    def flush_dialogue(self) -> list[dict]:
        entries = self.dialogue_log[:]
        self.dialogue_log.clear()
        return entries

    def _can_comment(self) -> bool:
        if self._poor_move_count >= self.max_poor_move_comments:
            return False
        if self._turn_num - self._last_comment_turn < 2:
            return False
        return True

    def _can_comment_general(self) -> bool:
        return self._turn_num - self._last_comment_turn >= 2

    def _emit_reasoning(self, board_before: "BoardState", move: dict, color: str, ply: int) -> None:
        """Emit heuristic breakdown of a move to the AI Discussion panel."""
        try:
            from ai.heuristics import tactical_move_bonus
            import re
            board_after = board_before.apply_move(move)
            bd = tactical_move_bonus(
                board_before, board_after, color,
                self.game_ai._weights,
                self.game_ai._opp_last_weak,
                return_breakdown=True,
            )
            if not isinstance(bd, dict):
                return
            top = bd.get("top_terms", [])
            total = bd.get("total", 0)
            color_name = "White" if color == "W" else "Black"
            move_str = _move_str(move)
            lines = [f"ply {ply} · {color_name}: {move_str}"]
            for label, val in top:
                clean = re.sub(r'\s*\([^)]+\)\s*$', '', label).strip()
                lines.append(f"  {clean}: {val:+d}")
            lines.append(f"  Δ total: {total:+d}")
            self.emit("GameAI", "\n".join(lines), tag="reasoning")
        except Exception:
            pass

    def _emit_signal_badge(self, signals: "LiveMoveSignals") -> None:
        """Emit a signal_badge entry for the live badge under the board."""
        import json
        label, detail, severity = signals.badge_tuple()
        self.emit("system", json.dumps({"label": label, "detail": detail, "severity": severity}), tag="signal_badge")

    # ── Game lifecycle ────────────────────────────────────────────────────────

    def on_game_start(self) -> None:
        self._poor_move_count = 0
        self._general_comment_count = 0
        self._last_comment_turn = -2
        self._human_turn_num = 0
        self._turn_num = 0
        self._game_moves = []
        self._session_id = str(uuid.uuid4())
        self._dominant_turn_streak = 0
        self._session_signals.clear()
        self.resignation_offered = False
        self.last_thinking = ""
        self.dialogue_log.clear()
        self.mills_llm.conversation_history.clear()
        self.mills_llm.narrative_memory = ""
        if self.opening_recognizer:
            self.opening_recognizer.reset()
        if self.endgame_recognizer:
            self.endgame_recognizer.reset()
        self._endgame_state = INACTIVE_ENDGAME

        # Pick an opening to target this game using UCB-scored selection.
        # select_opening() already filters by side, but double-check so a stale
        # 'both' entry from an unknown-outcome game is accepted for either colour.
        self._target_opening = None
        self._game_sym_idx = 0
        self._last_novel_id = None
        if self.opening_recognizer:
            candidate, sym_idx = pick_target_opening(
                self.opening_recognizer.book,
                self.game_ai.color,
            )
            if candidate is not None:
                self._target_opening = candidate
                self._game_sym_idx = sym_idx
                first_mv = ""
                if self._target_opening.line_moves:
                    raw = self._target_opening.line_moves[0]
                    first_mv = _transform_book_notation(raw, self._game_sym_idx) or raw
                    first_mv = f" → {first_mv}"
                self.emit(
                    "GameAI",
                    f"Targeting opening: {self._target_opening.name} "
                    f"(score {self._target_opening.opening_score(self.game_ai.color):.2f})"
                    f"{first_mv}",
                )

        recent = self.memory.load_recent_games(n=10)
        if recent:
            patterns = self.memory.analyse_patterns(recent)
            game_summaries = []
            for g in recent[:10]:
                notations = [m.get("notation", "") for m in g.get("moves", []) if m.get("notation")]
                move_str = " ".join(notations) if notations else "no moves recorded"
                game_summaries.append(
                    f"winner={g.get('winner','?')} opening={g.get('recognised_opening_name','unknown')} "
                    f"moves=({move_str})"
                )
            self.mills_llm.narrative_memory = (
                f"Recent pattern analysis: {json.dumps(patterns, indent=2)}\n\n"
                f"Last {len(recent)} games:\n" + "\n".join(game_summaries)
            )

    @staticmethod
    def _build_game_facts(game_record: dict) -> str:
        """Build an authoritative GAME FACTS block to ground the session summary LLM prompt.

        Derives piece counts and termination reason from the move list rather than
        relying on the LLM to infer them from an ASCII board (which can hallucinate).
        """
        moves = game_record.get("moves", [])
        total_half_moves = len(moves)
        winner = game_record.get("winner")

        # Compute final piece counts by replaying placement / capture events.
        # Seed from the first move's board_fen_before so resumed games (missing early placements)
        # start with the correct initial piece counts rather than 0.
        first_fen = moves[0].get("board_fen_before", "") if moves else ""
        board_part = first_fen.split("|")[0] if first_fen else ""
        on_board = {"W": board_part.count("W"), "B": board_part.count("B")}
        for m in moves:
            color = m.get("color", "")
            if m.get("type") == "place" and color in on_board:
                on_board[color] += 1
            cap = m.get("capture")
            if cap:
                # The captured piece belongs to the opponent of the mover
                opp = "B" if color == "W" else "W"
                on_board[opp] = max(0, on_board[opp] - 1)

        w_pieces = on_board["W"]
        b_pieces = on_board["B"]

        # Derive termination reason.
        # Check resignation flag first (set by coordinator before on_game_end).
        # Then infer from final piece counts: if the losing side has ≤ 2 pieces, piece-loss.
        # Otherwise assume no-legal-moves (blockade).
        if winner and game_record.get("result") == "ai_resignation":
            termination = "resignation"
        elif winner == "W" and b_pieces <= 2:
            termination = "piece-loss"
        elif winner == "B" and w_pieces <= 2:
            termination = "piece-loss"
        elif winner in ("W", "B"):
            termination = "no-legal-moves"
        else:
            termination = "draw-or-stalemate"

        winner_label = (
            "White" if winner == "W"
            else ("Black" if winner == "B" else "Draw")
        )

        return (
            "GAME FACTS (authoritative — do not contradict):\n"
            f"  Total half-moves: {total_half_moves}\n"
            f"  Termination: {termination}\n"
            f"  Final piece counts: White {w_pieces}, Black {b_pieces}\n"
            f"  Winner: {winner_label}"
        )

    @staticmethod
    def _arc_signals_block(signals_list: list) -> str:
        """Format a list of LiveMoveSignals as a compact RECENT SIGNALS block."""
        lines = [f"RECENT SIGNALS (last {len(signals_list)} human moves):"]
        for s in signals_list:
            parts = []
            if s.pref_delta is not None:
                parts.append(f"pref δ{s.pref_delta:+.2f}")
            if s.policy_prob is not None:
                parts.append(f"policy {s.policy_prob:.0%}")
            if s.sentinel_quality is not None:
                parts.append(f"sentinel {s.sentinel_quality:.2f}")
            if s.blunder_zone is not None:
                parts.append(f"risk {s.blunder_zone:.2f}")
            if s.is_unconventional:
                parts.append("unusual")
            if s.generalist_top:
                parts.append(f"gen:{s.generalist_top}")
            flags = []
            if s.is_weak:   flags.append("weak")
            if s.is_strong: flags.append("strong")
            if s.is_risky:  flags.append("risky")
            color_name = "W" if s.color == "W" else "B"
            signal_str = ", ".join(parts) if parts else "no signals"
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            lines.append(f"  ply {s.ply} {color_name} {s.move_played}: {signal_str}{flag_str}")
        return "\n".join(lines)

    def on_game_end(self, game_record: dict) -> None:
        self.memory.save_game_record(game_record)
        if self.trajectory_db is not None:
            self.trajectory_db.add_game(game_record)
        if self.endgame_db is not None:
            self.endgame_db.add_game(game_record)

        winner = game_record.get("winner")
        human_color = game_record.get("human_color", "W")

        if self.opening_recognizer:
            final = self.opening_recognizer.get_current_result()
            if final.status in ("novel", "inactive"):
                self._save_novel_opening(game_record)
            elif final.opening_id and final.status in ("exact", "probable", "transposition"):
                # Record this game's outcome against the recognised opening so
                # future UCB selection can learn which openings perform well.
                self.opening_recognizer.book.update_outcome_stats(
                    final.opening_id,
                    winner=winner or "D",
                    human_color=human_color,
                )

        facts_block = self._build_game_facts(game_record)
        summary = self.mills_llm.summarise_session([game_record], facts_block=facts_block)
        if summary:
            self.memory.save_session_narrative(summary)
            self.emit("MillsAI", summary)

    @staticmethod
    def _notation_to_move_dict(n: str) -> dict:
        cap, base = None, n
        if "x" in n:
            xi = n.index("x"); cap = n[xi + 1:]; base = n[:xi]
        if "-" in base:
            fr, to = base.split("-", 1)
            return {"from": fr, "to": to, "capture": cap}
        return {"from": None, "to": base, "capture": cap}

    @staticmethod
    def _compute_fen_signatures(placement_moves: list[str]) -> list[dict]:
        from game.board import BoardState
        board = BoardState.new_game()
        sigs = []
        seen: set[int] = set()
        n = len(placement_moves)
        for i, pos in enumerate(placement_moves):
            board = board.apply_move(Coordinator._notation_to_move_dict(pos))
            ply = i + 1
            if (ply in (4, 6, 8, 10, 12) or ply == n) and ply not in seen:
                seen.add(ply)
                sigs.append({"ply": ply, "fen": board.to_fen_string()})
        return sigs

    def _save_novel_opening(self, game_record: dict) -> None:
        from ai.opening_book import is_auto_named

        placement_moves = [
            m["to"] for m in game_record.get("moves", [])
            if m.get("type") == "place"
        ]
        if len(placement_moves) < 6:
            return

        book = self.opening_recognizer.book  # type: ignore[union-attr]
        winner = game_record.get("winner")

        # Check if an existing opening shares the same first 4+ moves.
        # If so, merge this game's outcome into it rather than creating a duplicate.
        similar = book.find_similar(placement_moves, min_common=4)
        if similar:
            canonical = max(
                similar,
                key=lambda o: sum(o.outcome_stats.get(k, 0) for k in ("W", "B", "D")),
            )
            if winner in ("W", "B", "D"):
                canonical.outcome_stats[winner] = canonical.outcome_stats.get(winner, 0) + 1
            # F: Always prompt the player to confirm/edit the name.
            # If LLM is available and the opening needs a name, ask LLM for a
            # suggestion but keep needs_llm_name=True so the frontend prompt fires.
            if is_auto_named(canonical.name) or canonical.needs_llm_name:
                llm_name = self.mills_llm.name_novel_opening(canonical.line_moves)
                if llm_name and not is_auto_named(llm_name):
                    # Store the LLM suggestion as the current name but leave
                    # needs_llm_name=True so the user is still prompted to confirm.
                    canonical.name = llm_name
                canonical.needs_llm_name = True
            book.save_opening(canonical)
            # Always surface the prompt so the player can confirm or rename.
            self._last_novel_id = canonical.opening_id
            return

        # No similar opening — create a new one.
        llm_available = self.mills_llm._client is not None
        name = self.mills_llm.name_novel_opening(placement_moves)
        sigs = self._compute_fen_signatures(placement_moves)
        # F: Always mark needs_llm_name=True and set _last_novel_id so the
        # frontend prompts the player to confirm/edit the name before it is saved.
        novel = book.save_novel_opening(
            placement_moves, sigs,
            outcome=winner,
            needs_llm_name=True,
        )
        novel.name = name  # LLM suggestion (or auto-name if LLM unavailable)
        book.save_opening(novel)
        self._last_novel_id = novel.opening_id

    # ── Tactical pre-screen ───────────────────────────────────────────────────

    def _tactical_situation(self, board: "BoardState") -> dict:
        """Classify the tactical urgency before move selection.

        Returns a dict with flags used both for logging and to inform the LLM
        about the immediate tactical context.
        """
        from ai.heuristics import (
            detect_double_mills, detect_feeder_mills,
            detect_diamonds, opponent_mills_in_n_moves,
        )
        from ai.heuristics import _closeable_mills, _fly_sacrifice_quality
        ai_color  = board.turn
        opp_color = "B" if ai_color == "W" else "W"

        can_close      = _closeable_mills(board, ai_color) > 0
        opp_can_close  = _closeable_mills(board, opp_color) > 0
        opp_doubles    = detect_double_mills(board, opp_color)
        ai_doubles     = detect_double_mills(board, ai_color)
        opp_diamonds   = detect_diamonds(board, opp_color)
        opp_threats_2  = opponent_mills_in_n_moves(board, opp_color, n=2)

        ai_pieces  = board.pieces_on_board[ai_color]
        opp_pieces = board.pieces_on_board[opp_color]
        sacrifice_viable = (
            ai_pieces == 6 and opp_pieces == 4
            and get_game_phase(board, ai_color) == "move"
            and _fly_sacrifice_quality(board, ai_color) > 0
        )

        return {
            "urgent":               can_close or opp_can_close or bool(opp_doubles),
            "can_close_mill":       can_close,
            "must_block_opponent":  opp_can_close,
            "opp_double_mills":     opp_doubles,
            "ai_double_mills":      ai_doubles,
            "opp_diamonds":         opp_diamonds,
            "opp_threats_in_2":    opp_threats_2,
            "6v4_sacrifice_viable": sacrifice_viable,
        }

    # ── AI deliberation ───────────────────────────────────────────────────────

    # How much the LLM recommendation must outScore GameAI's choice (after bonus)
    # before GameAI defers to it.
    LLM_BONUS = 0.15

    def deliberate(self, board: "BoardState") -> dict:
        self._turn_num += 1
        legal = get_all_legal_moves(board)
        if not legal:
            raise RuntimeError("No legal moves available")

        # 1. Get current opening recognition and endgame state
        recognition = (
            self.opening_recognizer.get_current_result()
            if self.opening_recognizer else INACTIVE_RESULT
        )

        # If recognition hasn't found an opening yet but we have a target,
        # synthesise a recognition hint from the target so the AI's opening
        # bonus steers it along the preferred line from the very first move.
        recognition = synthesize_opening_recognition(
            recognition,
            self._target_opening,
            board,
            self._game_moves,
            self._game_sym_idx,
        )
        if self.endgame_recognizer:
            self._endgame_state = self.endgame_recognizer.update(board)
            for msg in self.endgame_recognizer.transition_announcements():
                self.emit("MillsAI", msg)
        endgame_state = self._endgame_state

        # 2. Query trajectory DB for historical move-outcome hints
        trajectory_hints = build_trajectory_hints(
            self.trajectory_db,
            board,
            self._game_moves,
            self.game_ai,
            endgame_db=self.endgame_db,
            endgame_state=endgame_state,
        )
        trajectory_context = format_trajectory_context(trajectory_hints)

        # 3. Tactical pre-screen: log urgency level so the AI and LLM know the context
        tac = self._tactical_situation(board)
        if tac["can_close_mill"]:
            self.emit("GameAI", "Mill closure available — prioritising tactical completion")
        elif tac["must_block_opponent"]:
            self.emit("GameAI", "Opponent threatens a mill — defensive priority")
        elif tac["opp_double_mills"]:
            pivots = ", ".join(tac["opp_double_mills"][:2])
            self.emit("GameAI", f"Disrupting opponent cycling mill pivot at {pivots}")
        elif tac.get("6v4_sacrifice_viable"):
            self.emit("GameAI", "6v4 position — own fly nucleus ready, evaluating sacrifice path to winning endgame")

        # 4. GameAI picks its best move (with opening bonus, trajectory hints, endgame depth)
        # Force the book move for the AI's first 2 placements so opening variety is
        # visible regardless of adherence slider.  At 100% adherence the book move is
        # forced for any ply where recognition is active.
        force_book_early = compute_force_book_early(
            board, self._game_moves, self.game_ai.color,
        )
        ai_move = self.game_ai.choose_move(
            board,
            recognition=recognition,
            endgame_state=endgame_state,
            trajectory_hints=trajectory_hints,
            force_book_early=force_book_early,
            trajectory_db=self.trajectory_db,
        )
        self.last_thinking = self.game_ai.last_thinking
        ai_score = self.game_ai.score_move(board, ai_move)

        # 5. Expose score hint to MillsLLM for its prompt
        self.mills_llm._last_ai_score = ai_score

        # 5. Ask MillsLLM for a recommendation (with opening + endgame context).
        # At easy difficulty (≤4) the LLM call (~15–18 s) dominates a fast search,
        # so skip the opinion here and play the search move immediately.
        # LLM commentary still fires via react_to_human_move on the human's turns.
        _notations_so_far = [m.get("notation", "") for m in self._game_moves if m.get("notation")]
        _use_llm_opinion = self.game_ai.difficulty > 4 and not self.game_ai._force_stop
        if _use_llm_opinion:
            opinion, llm_notation = self.mills_llm.ask_for_move_opinion(
                board, legal, ai_move, recognition=recognition, endgame_state=endgame_state,
                audience="human" if self.vs_human else "ai",
                move_history=_notations_so_far,
                trajectory_context=trajectory_context,
            )
        else:
            opinion, llm_notation = None, None

        # 6. Try to adopt the LLM's recommendation if it scores well enough
        move = ai_move
        if llm_notation:
            from ai.mills_llm import _notation_to_move
            llm_move = _notation_to_move(llm_notation, legal)
            if llm_move and llm_move != ai_move:
                # Don't adopt if the LLM recommendation is a banned move
                _fen = board.to_fen_string()
                _banned = self.game_ai._pos_bans.get(_fen, set())
                if self.game_ai._move_notation(llm_move) in _banned:
                    llm_move = None
            if llm_move and llm_move != ai_move:
                llm_score = self.game_ai.score_move(board, llm_move)
                # Never let the LLM override the engine during tactical emergencies
                # (must-block or own-mill-closure), regardless of score delta.
                _tac_lock = tac["must_block_opponent"] or tac["can_close_mill"]
                if self.llm_can_override_move and llm_score + self.LLM_BONUS > ai_score and not _tac_lock:
                    move = llm_move
                    self.emit(
                        "GameAI",
                        f"Engine intended {_move_str(ai_move)} (score {ai_score:.2f}); "
                        f"MillsLLM recommends {llm_notation} (score {llm_score:.2f}) — adopting",
                    )
                else:
                    self.emit(
                        "GameAI",
                        f"MillsLLM suggests {llm_notation} "
                        f"(score {llm_score:.2f}), engine stays with {_move_str(ai_move)} "
                        f"(score {ai_score:.2f})",
                    )

        # 5. Log LLM's reasoning only when it successfully recommended a move
        if opinion and llm_notation:
            reason = _extract_reason(opinion)
            if reason:
                self.emit("MillsLLM", reason)

        move_str = _move_str(move)
        _ai_signals = self._live_analyser.analyse(
            board, move, legal, ai_score, self.game_ai.color, self._turn_num,
            is_ai_move=True,
        )
        self._emit_signal_badge(_ai_signals)
        self._emit_reasoning(board, move, self.game_ai.color, self._turn_num)
        self.emit("GameAI", f"Playing {move_str}")

        # Resignation check: if human has dominated for 3 consecutive AI turns
        if not self.resignation_offered:
            try:
                from ai.heuristics import evaluate, TANH_SCALE
                human_color = "B" if self.game_ai.color == "W" else "W"
                post = board.apply_move(move)
                raw  = evaluate(post, human_color)
                norm = math.tanh(raw / TANH_SCALE.get(get_game_phase(post, human_color), 180))
                if norm > 0.95:
                    self._dominant_turn_streak += 1
                else:
                    self._dominant_turn_streak = 0
                if self._dominant_turn_streak >= 5:
                    self.resignation_offered = True
                    farewell = random.choice([
                        "Your position is overwhelming — I concede. Well played.",
                        "I see no path forward. A masterful performance.",
                        "You've outplayed me completely. I yield.",
                        "My position is beyond recovery. Congratulations.",
                    ])
                    self.emit("MillsAI", farewell)
            except Exception:
                pass

        if self.game_ai.last_was_blunder and _use_llm_opinion:
            blunder_msg = self.mills_llm.announce_blunder(board, move, move_history=_notations_so_far)
            self.emit("MillsAI", blunder_msg if blunder_msg else
                      "I just made a mistake there — can you spot what I should have done instead?")

        # 8. Update recognizer with AI's move notation
        if self.opening_recognizer:
            self.opening_recognizer.update(move.get("to", ""), board)

        rec = recognition
        self._game_moves.append({
            "turn": self._turn_num,
            "color": board.turn,
            "type": get_game_phase(board, board.turn),
            "from": move.get("from"),
            "to": move.get("to"),
            "capture": move.get("capture"),
            "notation": move_str,
            "board_fen_before": board.to_fen_string(),
            "was_blunder": self.game_ai.last_was_blunder,
            "opening_recognition": {
                "status": rec.status,
                "name": rec.name,
                "confidence": rec.confidence,
            },
        })

        return move

    # ── Human move reaction ───────────────────────────────────────────────────

    def react_to_human_move(
        self,
        board_before: "BoardState",
        board_after: "BoardState",
        human_move: dict,
    ) -> None:
        self._turn_num += 1
        self._human_turn_num += 1

        # Update endgame state
        if self.endgame_recognizer:
            self._endgame_state = self.endgame_recognizer.update(board_after)
            for msg in self.endgame_recognizer.transition_announcements():
                self.emit("MillsAI", msg)

        # Update recognizer with human's move
        recognition = INACTIVE_RESULT
        if self.opening_recognizer:
            recognition = self.opening_recognizer.update(
                human_move.get("to", ""), board_after
            )
            if recognition.status == "exact" and recognition.name:
                self.emit("MillsAI", f"Opening recognised: {recognition.name}")
            elif recognition.status == "transposition" and recognition.name:
                self.emit("MillsAI", f"Transposition to: {recognition.name}")

        legal_moves  = get_all_legal_moves(board_before)
        score_before = self.game_ai.score_move(board_before, human_move)
        signals      = self._live_analyser.analyse(
            board_before, human_move, legal_moves,
            score_norm=score_before,
            color=board_before.turn,
            ply=self._turn_num,
        )

        self._emit_signal_badge(signals)
        self._session_signals.append(signals)
        if len(self._session_signals) > 6:
            self._session_signals.pop(0)

        self._game_moves.append({
            "turn": self._turn_num,
            "color": board_before.turn,
            "type": get_game_phase(board_before, board_before.turn),
            "from": human_move.get("from"),
            "to": human_move.get("to"),
            "capture": human_move.get("capture"),
            "notation": _move_str(human_move),
            "board_fen_before": board_before.to_fen_string(),
            "game_ai_score": score_before,
            "opening_recognition": {
                "status": recognition.status,
                "name": recognition.name,
                "confidence": recognition.confidence,
                "deviation": recognition.deviation_ply is not None,
            },
        })

        self._emit_reasoning(board_before, human_move, board_before.turn, self._turn_num)

        if not self._can_comment_general():
            return

        has_capture = bool(human_move.get("capture"))
        score_after = 1.0 - score_before
        _notations = [m.get("notation", "") for m in self._game_moves if m.get("notation")]

        # 1. Mill/capture commentary — always comment when human forms a mill
        if has_capture:
            comment = self.mills_llm.comment_on_mill(
                board_after, human_move,
                human_color=self.human_color, move_history=_notations,
            )
            if comment:
                self.emit("MillsAI", comment)
                self._general_comment_count += 1
                self._last_comment_turn = self._turn_num
                return

        # 2. Signal-grounded comment when live signals indicate a noteworthy move
        if self._live_analyser.has_signals and self._can_comment():
            _horizon_fires = (
                signals.horizon_delta is not None
                and signals.horizon_delta >= self._horizon_threshold
            )
            _blunder_notable = (
                signals.blunder_zone is not None and signals.blunder_zone > 0.50
            )
            _signal_fires = (
                signals.is_weak
                or signals.is_unconventional
                or signals.is_risky
                or _blunder_notable
                or signals.generalist_top is not None
                or _horizon_fires
            )
            if _signal_fires:
                comment = self.mills_llm.comment_with_live_signals(
                    board_before, human_move, signals,
                    human_color=self.human_color, move_history=_notations,
                )
                if comment:
                    tag = "warning" if signals.is_weak else "normal"
                    self.emit("MillsAI", comment, tag=tag)
                    self._poor_move_count += 1
                    self._last_comment_turn = self._turn_num
                    return

        # 2b. Poor-move warning fallback (no live signals, or signals didn't trigger)
        if self._can_comment():
            comment = self.mills_llm.evaluate_human_move(
                board_before=board_before,
                human_move=human_move,
                score_before=score_before,
                score_after=score_after,
                score_drop_threshold=self.poor_move_threshold,
                recognition=recognition,
                human_color=self.human_color,
                move_history=_notations,
            )
            if comment:
                self.emit("MillsAI", comment, tag="warning")
                self._poor_move_count += 1
                self._last_comment_turn = self._turn_num
                return

        # 3. Positive commentary on strong moves
        if score_before >= 0.75:
            comment = self.mills_llm.comment_on_good_move(
                board_after, human_move, score_before,
                human_color=self.human_color, move_history=_notations,
            )
            if comment:
                self.emit("MillsAI", comment)
                self._general_comment_count += 1
                self._last_comment_turn = self._turn_num
                return

        # 4. Periodic strategic question every 8 human moves
        if self._human_turn_num % 8 == 0:
            question = self.mills_llm.ask_strategic_question(
                board_after, human_color=self.human_color, move_history=_notations,
            )
            if question:
                self.emit("MillsAI", question)
                self._last_comment_turn = self._turn_num

        # 5. Phase 4: Arc comment every 4 human moves when buffer has enough data
        if (self._human_turn_num % 4 == 0
                and len(self._session_signals) >= 4
                and self._can_comment_general()):
            arc_block = self._arc_signals_block(self._session_signals[-4:])
            arc = self.mills_llm.comment_with_arc(
                board_after, arc_block, human_color=self.human_color,
            )
            if arc:
                self.emit("MillsAI", arc)
                self._last_comment_turn = self._turn_num

    # ── Export ────────────────────────────────────────────────────────────────

    def build_game_record(self, winner: str | None, human_color: str) -> dict:
        return {
            "session_id": self._session_id,
            "date": datetime.now().isoformat(),
            "human_color": human_color,
            "winner": winner,
            "moves": self._game_moves,
            "bad_moves_taught": [],
        }


def _move_str(move: dict) -> str:
    if move.get("from"):
        s = f"{move['from']}-{move['to']}"
    else:
        s = move["to"]
    if move.get("capture"):
        s += f"x{move['capture']}"
    return s


def _extract_reason(response: str) -> str:
    """Return the REASON line from a structured LLM response, stripping the MOVE line."""
    lines = []
    for line in response.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("MOVE:"):
            continue
        if stripped.upper().startswith("REASON:"):
            lines.append(stripped[7:].strip())
        elif lines:
            lines.append(stripped)
    return " ".join(lines).strip()
