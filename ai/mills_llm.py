"""ai/mills_llm.py — Ollama interface for LLM-assisted Nine Men's Morris."""

from __future__ import annotations

import pathlib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.board import BoardState
    from ai.memory_manager import MemoryManager
    from ai.post_game_assessor import PostGameAnnotation

_MAX_HISTORY = 16

# ── Phase strategy guide ───────────────────────────────────────────────────────
# Load docs/phase_strategy.md once at import time; injected into LLM prompts.
_PHASE_STRATEGY: dict[str, str] = {}
_phase_strategy_path = pathlib.Path(__file__).parent.parent / "docs" / "phase_strategy.md"
if _phase_strategy_path.exists():
    _raw = _phase_strategy_path.read_text(encoding="utf-8")
    for _sec in re.split(r'\n(?=## Phase )', _raw):
        _m = re.match(r'## Phase ([A-D])', _sec)
        if _m:
            _PHASE_STRATEGY[_m.group(1)] = _sec.strip()


def _phase_strategy_for(board: "BoardState") -> str:
    """Return the docs/phase_strategy.md section relevant to the current position."""
    if not _PHASE_STRATEGY:
        return ""
    from game.rules import get_game_phase
    player = board.turn
    phase = get_game_phase(board, player)
    placed = board.pieces_placed[player]
    on_board_w = board.pieces_on_board["W"]
    on_board_b = board.pieces_on_board["B"]
    if phase == "place":
        key = "A" if placed <= 6 else "B"
    elif min(on_board_w, on_board_b) <= 4:
        key = "D"
    else:
        key = "C"
    return _PHASE_STRATEGY.get(key, "")


# ── Mill names ─────────────────────────────────────────────────────────────────
# Keys are the exact 3-tuples from game.board.MILLS, in the same order.

MILL_NAMES: dict[tuple[str, str, str], str] = {
    # Outer ring
    ("a7", "d7", "g7"): "Outer top",
    ("g7", "g4", "g1"): "Outer right",
    ("g1", "d1", "a1"): "Outer bottom",
    ("a1", "a4", "a7"): "Outer left",
    # Middle ring
    ("b6", "d6", "f6"): "Middle top",
    ("f6", "f4", "f2"): "Middle right",
    ("f2", "d2", "b2"): "Middle bottom",
    ("b2", "b4", "b6"): "Middle left",
    # Inner ring
    ("c5", "d5", "e5"): "Inner top",
    ("e5", "e4", "e3"): "Inner right",
    ("e3", "d3", "c3"): "Inner bottom",
    ("c3", "c4", "c5"): "Inner left",
    # Cross-ring connecting lines
    ("d7", "d6", "d5"): "d-column top",
    ("g4", "f4", "e4"): "g-row",
    ("d1", "d2", "d3"): "d-col bottom",
    ("a4", "b4", "c4"): "a-row",
}


def _board_summary(board: "BoardState") -> str:
    """Return a structured POSITION SUMMARY block for LLM prompts.

    Includes: phase (per-side), piece counts (on board / in hand),
    closed mills by name, two-piece threats with their closing squares,
    and legal-move mobility counts for each side.
    """
    from game.board import MILLS
    from game.rules import get_game_phase

    lines: list[str] = ["POSITION SUMMARY:"]

    # Phase — per side, shown as single label when equal, else "W <x>, B <y>"
    phase_w = get_game_phase(board, "W")
    phase_b = get_game_phase(board, "B")
    if phase_w == phase_b:
        lines.append(f"Phase: {phase_w}")
    else:
        lines.append(f"Phase: W {phase_w}, B {phase_b}")

    # Piece counts: on board and in hand (pieces not yet placed)
    in_hand_w = max(0, 9 - board.pieces_placed["W"])
    in_hand_b = max(0, 9 - board.pieces_placed["B"])
    on_board_w = board.pieces_on_board["W"]
    on_board_b = board.pieces_on_board["B"]
    lines.append(
        f"White: {on_board_w} on board ({in_hand_w} in hand) | "
        f"Black: {on_board_b} on board ({in_hand_b} in hand)"
    )

    # Closed mills (all 3 squares owned by the same player)
    for color, label in (("W", "White"), ("B", "Black")):
        closed = [
            mill for mill in MILLS
            if all(board.positions[p] == color for p in mill)
        ]
        if closed:
            names = ", ".join(
                f"{MILL_NAMES.get(mill, '-'.join(mill))} ({'-'.join(mill)})"
                for mill in closed
            )
            lines.append(f"Closed mills — {label}: {names}")

    # Two-piece threats (2-configs) with closing square
    for color, label in (("W", "White"), ("B", "Black")):
        threats: list[str] = []
        for mill in MILLS:
            vals = [board.positions[p] for p in mill]
            if vals.count(color) == 2 and vals.count("") == 1:
                closing = next(p for p in mill if board.positions[p] == "")
                filled = [p for p in mill if board.positions[p] == color]
                name = MILL_NAMES.get(mill, '-'.join(mill))
                threats.append(
                    f"{name} ({'-'.join(filled)} — closes at {closing})"
                )
        if threats:
            lines.append(f"Threats — {label}: {'; '.join(threats)}")

    # Mobility: count legal moves for each side
    # For the side not to move we temporarily flip the turn.
    def _mobility_count(b: "BoardState", color: str) -> int:
        """Count distinct source→dest pairs, ignoring capture combinations."""
        phase = get_game_phase(b, color)
        if phase == "place":
            return len(b.legal_placements(color))
        return len(b.legal_moves(color))

    mob_w = _mobility_count(board, "W")
    mob_b = _mobility_count(board, "B")
    lines.append(f"Mobility: White {mob_w} moves | Black {mob_b} moves")

    return "\n".join(lines)

_BOARD_RULES = """\
You are MillsAI, an assistant for Nine Men's Morris.
Also called: Mills, Mühle, Merels.

This is NOT chess.
Never use chess terms, chess ideas, or chess piece names.

GAME FACTS:
- Board has exactly 24 valid nodes:
  Outer:  a7 d7 g7 g4 g1 d1 a1 a4
  Middle: b6 d6 f6 f4 f2 d2 b2 b4
  Inner:  c5 d5 e5 e4 e3 d3 c3 c4
- Players: White (W) and Black (B), 9 pieces each.
- Phases:
  1) place  -> place on any empty node
  2) move   -> slide along a connected line to an adjacent empty node
  3) fly    -> if a side has exactly 3 pieces, it may move to any empty node
- Mill: 3 own pieces in a row on a legal board line
- After forming a mill, remove one opponent piece
- Win by reducing the opponent to 2 pieces, or leaving them with no legal move

MOVE NOTATION:
- Place: d2
- Move: a4-a7
- Place + capture: d2xb6
- Move + capture: a4-a7xb6

STRICT VALIDITY:
- Only use node names from the 24-node list above
- Only choose from the provided LEGAL MOVES
- Never invent notation
"""

_DECISION_POLICY = """\
DECISION PRIORITIES FOR NINE MEN'S MORRIS:

Opening / placement priorities:
1. Prefer control of central/cardinal points and flexible positions
2. Avoid self-crowding and dead placements
3. Avoid making flashy early mills if they reduce future mobility
4. Block dangerous opponent mills, especially strong cardinal mills
5. Preserve the ability to create two future threats instead of one short-lived threat

Midgame priorities:
1. Keep or regain initiative
2. Preserve mobility and restrict opponent mobility
3. Prefer moves that maintain multiple future mill threats
4. Break or punish unstable opponent structures
5. Avoid moves that trap your own pieces

Endgame priorities:
1. Prevent immediate loss
2. Create or stop forced mills
3. Maximize mobility
4. In flying positions, value forcing threats and dual threats highly

GENERAL STYLE:
- Be concrete, not poetic
- Prefer safe, strong, legal moves over speculative ones
- If opening context is given, use it
- If endgame context is given, use it
- If strategic memory is given, treat it as a hint, not a rule
"""

_MOVE_SYSTEM = _BOARD_RULES + "\n" + _DECISION_POLICY + """

TASK:
Choose the single best move for the side to move from LEGAL MOVES.

YOUR RESPONSE MUST BEGIN WITH EXACTLY THIS FORMAT — NO EXCEPTIONS:
MOVE: <exact string from LEGAL MOVES>
REASON: <one sentence, max 18 words>

CRITICAL RULES:
- Line 1 MUST be "MOVE: " followed by one exact entry from LEGAL MOVES
- Line 2 MUST be "REASON: " followed by one sentence
- Do NOT write anything before the MOVE: line
- Do NOT add markdown, headers, or extra explanation
- If you are unsure, still pick the safest move from LEGAL MOVES
"""

_COMMENT_SYSTEM = _BOARD_RULES + """

TASK:
Comment briefly on the human's last move.

OUTPUT RULES:
- Write exactly one short sentence, max 18 words
- Focus on mobility, initiative, mill threat, blunder risk, or positional consequence
- Do not suggest a move
- Do not mention chess
- If the move is acceptable and not worth commenting on, reply exactly:
NO_COMMENT
"""

_QUESTION_SYSTEM = _BOARD_RULES + """

TASK:
Ask the human one brief useful question to learn their plan.

OUTPUT RULES:
- One sentence only
- Max 14 words
- No lecture
- No move suggestion
"""

_BLUNDER_SYSTEM = _BOARD_RULES + """

TASK:
You intentionally made a bad move for teaching.
Tell the human this was deliberate and invite them to find the better idea.

OUTPUT RULES:
- 1 or 2 short sentences
- Do not reveal the correct move
- Do not mention chess
"""

_SESSION_SYSTEM = _BOARD_RULES + """

TASK:
Write a compact markdown session summary.

FORMAT:
## Session
- Winner and result pattern (e.g. "White wins by piece loss, 8 vs 3")
- Opening phase style if notable (e.g. "central control vs flank spread")
- One strategic pattern that decided the game (describe in terms of piece advantage or mill threat — do NOT quote or paraphrase specific move notations)
- One lesson about mobility, initiative, or mill timing

STRICT CONSTRAINTS:
- Never invent, quote, or describe specific move notations (like "b6xb4" or "f2-d2")
- Only describe strategic patterns and piece-count outcomes
- Keep it concise (under 80 words)
"""

_DEBRIEF_GAME_SYSTEM = _BOARD_RULES + """

TASK:
Write a clear game debrief.

FORMAT:
## Result
## Opening
## Turning Point
## Mistakes
## Lessons

STYLE:
- concise
- concrete
- coaching tone
- no move invention
"""

_DEBRIEF_ANNOTATED_SYSTEM = _BOARD_RULES + """

TASK:
Write a concise post-game commentary in 3–5 sentences.

STRUCTURE:
1. One sentence on the game's overall character (who dominated, how balanced the game was).
2. One sentence focused on the turning point — the move that decided the game.
3. Optionally, one sentence on any other notable poor moves or recurring patterns.

STYLE:
- Coaching tone; be concrete and specific.
- Never invent move quality claims not present in the fact block below.
- Malom labels (W, D, L) are categorical outcomes; never paraphrase them as probabilities.
- Label any model probability or empirical win rate explicitly as such.
- Do not quote raw decimal scores; use the categorical descriptors already provided.
"""

_DEBRIEF_POSITION_SYSTEM = _BOARD_RULES + """

TASK:
Explain one replay position.

OUTPUT RULES:
- 2 short paragraphs max
- Explain why the played move mattered
- If a better move existed, explain the difference in plain language
- If this was a turning point, say why
"""

_OPENING_NAME_SYSTEM = _BOARD_RULES + """

TASK:
Invent a short traditional-sounding name for a novel opening sequence.

OUTPUT RULES:
- Reply with only the name
- 2 to 4 words
- No punctuation at the end
- Tone: memorable, serious, game-opening style
"""

_PLAYER_CHAT_SYSTEM = _BOARD_RULES + """

TASK:
Respond to the human player's message during a live game.

OUTPUT RULES:
- 1 to 3 sentences maximum
- Stay focused on Nine Men's Morris strategy or their question
- You may comment on the current position if relevant
- Do NOT suggest a specific move to play next
- Do NOT use chess terminology
- Be concise and coaching in tone
"""

_POSITIVE_COMMENT_SYSTEM = _BOARD_RULES + """

TASK:
Comment briefly on the human's strong move.

OUTPUT RULES:
- Write exactly one short sentence, max 18 words
- Focus on what makes this move strong: mobility gain, mill threat, positional control
- Be encouraging but concise
- Do not suggest another move
- Do not mention chess
- If the move is not worth commenting on, reply exactly: NO_COMMENT
"""

_MILL_COMMENT_SYSTEM = _BOARD_RULES + """

TASK:
Comment briefly on the human forming a mill and capturing a piece.

OUTPUT RULES:
- Write exactly one short sentence, max 18 words
- Note the tactical achievement and what it means for the position going forward
- Do not suggest a move
- Do not mention chess
"""

_POSITION_QUESTION_SYSTEM = _BOARD_RULES + """

TASK:
Ask the human one brief useful question about their strategic plan.

OUTPUT RULES:
- One sentence only
- Max 14 words
- No lecture, no move suggestion
- Ask something that invites them to think about mobility, threats, or mill formation
"""


def _move_history_block(notations: list[str], limit: int = 40) -> str:
    """Format recent move notations as a compact numbered move list."""
    if not notations:
        return ""
    recent = notations[-limit:]
    offset = len(notations) - len(recent)
    lines = []
    for i, n in enumerate(recent, start=offset + 1):
        lines.append(f"{i}. {n}")
    return "\n--- GAME MOVES SO FAR ---\n" + "  ".join(lines) + "\n---"


def _endgame_context_block(endgame_state) -> str:
    lines = [
        "",
        "--- ENDGAME CONTEXT ---",
        f"Phase: {endgame_state.phase}",
        f"Pieces: W={endgame_state.pieces_white} B={endgame_state.pieces_black} total={endgame_state.total_pieces}",
        f"Mobility: W={endgame_state.mobility_white} B={endgame_state.mobility_black}",
        f"Zugzwang risk: {'yes' if endgame_state.zugzwang_risk else 'no'}",
        f"Pattern: {endgame_state.pattern or 'none'}",
    ]
    if getattr(endgame_state, "pattern_notes", None):
        lines.append(f"Pattern notes: {endgame_state.pattern_notes}")
    lines.append("---")
    return "\n".join(lines)


def _opening_context_block(recognition) -> str:
    lines = [
        "",
        "--- OPENING CONTEXT ---",
        f"Name: {recognition.name or 'Unknown / Novel'}",
        f"Family: {recognition.family or '—'}",
        f"Status: {recognition.status}",
        f"Confidence: {recognition.confidence:.0%}",
        f"Book move now: {recognition.book_move or 'none / exhausted'}",
        f"Strategic idea: {recognition.strategic_notes or '—'}",
        f"Common blunders: {', '.join(recognition.common_blunders) if recognition.common_blunders else 'none recorded'}",
        "---",
    ]
    return "\n".join(lines)


def _move_to_notation(move: dict) -> str:
    if move.get("from"):
        s = f"{move['from']}-{move['to']}"
    else:
        s = move["to"]
    if move.get("capture"):
        s += f"x{move['capture']}"
    return s


def _notation_to_move(notation: str, legal: list[dict]) -> dict | None:
    notation = notation.strip().lower()
    for m in legal:
        if _move_to_notation(m) == notation:
            return m
    for m in legal:
        if m["to"] == notation and not m.get("capture"):
            return m
    return None


# ── Post-game annotation prompt helpers ───────────────────────────────────────

_WDL_ARROWS = {
    "win_to_loss": "Win→Loss",
    "win_to_draw": "Win→Draw",
    "draw_to_loss": "Draw→Loss",
}


def _wdl_quality_label(quality: str) -> str:
    """Convert a turning_point_quality string to a display label.

    Malom path: quality is a WDL class like 'win_to_loss' → 'Win→Loss'.
    Heuristic path: quality is already a regret label like 'r_h:0.712'.
    """
    return _WDL_ARROWS.get(quality, quality)


def _score_trend_words(
    heuristic_curve: "list[float]",
    turning_point_ply: "int | None",
) -> str:
    """Derive a categorical one-sentence description of the game arc.

    Uses sign of heuristic_score_white (positive = White ahead) to avoid
    quoting raw numbers in the LLM prompt.
    """
    if not heuristic_curve:
        return "No score data available."
    n = len(heuristic_curve)
    white_ahead = sum(1 for v in heuristic_curve if v > 0)
    frac = white_ahead / n
    if frac >= 0.70:
        trend = "White held the advantage for most of the game"
    elif frac <= 0.30:
        trend = "Black held the advantage for most of the game"
    elif frac >= 0.55:
        trend = "White held a slight edge overall"
    elif frac <= 0.45:
        trend = "Black held a slight edge overall"
    else:
        trend = "The position was closely contested throughout"
    if turning_point_ply is not None and 0 < turning_point_ply < n - 1:
        trend += f", with the key shift occurring at ply {turning_point_ply + 1}"
    return trend + "."


def _build_debrief_prompt(report, annotation: "PostGameAnnotation") -> str:
    """Build the structured user prompt for annotated game debrief.

    ``report`` is any duck-typed object with .winner, .loser, .opening_name.
    ``annotation`` is a PostGameAnnotation from PostGameAssessor.
    """
    lines: list[str] = []

    # GAME FACTS
    n_plies = len(annotation.moves)
    opening = annotation.opening_name or getattr(report, "opening_name", None) or "unknown"
    lines.append("GAME FACTS:")
    lines.append(f"  Winner: {report.winner}")
    lines.append(f"  Loser:  {report.loser}")
    lines.append(f"  Plies:  {n_plies}")
    lines.append(f"  Opening: {opening}")
    lines.append("")

    # SCORE TREND
    lines.append("SCORE TREND:")
    lines.append(
        "  " + _score_trend_words(annotation.heuristic_curve, annotation.turning_point_ply)
    )
    lines.append("")

    # TURNING POINT
    tp_ply = annotation.turning_point_ply
    if tp_ply is not None and 0 <= tp_ply < len(annotation.moves):
        tp = annotation.moves[tp_ply]
        oracle = annotation.turning_point_oracle
        quality_label = _wdl_quality_label(annotation.turning_point_quality)
        lines.append("TURNING POINT:")
        lines.append(f"  Ply:            {tp_ply + 1}")
        lines.append(f"  Move played:    {tp.move_played}")
        if tp.best_alt:
            lines.append(f"  Best available: {tp.best_alt}")
        if oracle in ("malom_full", "retrograde_wdl"):
            lines.append(f"  Malom outcome:  {quality_label}  (oracle: {oracle})")
        else:
            lines.append(f"  Regret signal:  {quality_label}  (oracle: {oracle})")
        lines.append("")

    # OTHER POOR MOVES (only when more than one confirmed_poor)
    confirmed_poor = [m for m in annotation.moves if m.quality == "confirmed_poor"]
    other_poor = [m for m in confirmed_poor if m.ply != tp_ply]
    if other_poor:
        lines.append("OTHER POOR MOVES:")
        for m in other_poor:
            if m.oracle_source in ("malom_full", "retrograde_wdl"):
                wdl_key = f"{m.wdl_before}→{m.wdl_after}" if m.wdl_before else "confirmed_poor"
                alt_note = f"  best: {m.malom_best_alt}" if m.malom_best_alt else ""
                lines.append(f"  Ply {m.ply + 1}: {m.move_played}  (Malom: {wdl_key}){alt_note}")
            else:
                lines.append(f"  Ply {m.ply + 1}: {m.move_played}  (r_h={m.r_h:.2f})")
        lines.append("")

    # SENTINEL TURNING POINTS (always shown when Sentinel available, even if Malom ran)
    sentinel_tps = [
        (ply, q, src) for ply, q, src in annotation.turning_points
        if src == "sentinel+heuristic"
    ]
    if sentinel_tps:
        lines.append("SENTINEL TURNING POINTS:")
        for ply, quality, _ in sentinel_tps:
            if 0 <= ply < len(annotation.moves):
                m = annotation.moves[ply]
                lines.append(f"  Ply {ply + 1}: {m.move_played}  ({quality})")
        lines.append("")

    # GENERALIST DIVERGENCE (moves where AI preferred a different move)
    gen_divs = [
        m for m in annotation.moves
        if m.generalist_top_move is not None
        and not m.generalist_self_assessed
        and m.generalist_top_move != m.move_played
    ]
    if gen_divs:
        lines.append("GENERALIST DIVERGENCE:")
        for m in sorted(gen_divs, key=lambda x: abs(x.r_h), reverse=True)[:4]:
            lines.append(
                f"  Ply {m.ply + 1} ({m.color}): played {m.move_played}, "
                f"AI preferred {m.generalist_top_move}"
            )
        lines.append("")

    # POLICY/PREF DIVERGENCE (pref_delta: human preference vs teacher policy)
    pref_moves = [
        m for m in annotation.moves
        if m.policy_pref_delta is not None and abs(m.policy_pref_delta) > 0.1
    ]
    if pref_moves:
        lines.append("POLICY DIVERGENCE (pref vs teacher):")
        for m in sorted(pref_moves, key=lambda x: x.policy_pref_delta)[:4]:
            direction = "weak" if m.policy_pref_delta < 0 else "strong"
            lines.append(
                f"  Ply {m.ply + 1}: {m.move_played}  delta={m.policy_pref_delta:+.2f} ({direction})"
            )
        lines.append("")

    # GAPNET RISK POSITIONS (pre-move blunder-zone density)
    gap_moves = [m for m in annotation.moves if m.blunder_zone_score is not None]
    if gap_moves:
        top_gap = sorted(gap_moves, key=lambda x: x.blunder_zone_score, reverse=True)[:3]
        lines.append("GAPNET RISK POSITIONS:")
        for m in top_gap:
            lines.append(
                f"  Ply {m.ply + 1} ({m.color}): {m.move_played}  "
                f"blunder-zone={m.blunder_zone_score:.2f}"
            )
        lines.append("")

    return "\n".join(lines)


class MillsLLM:
    def __init__(
        self,
        memory: "MemoryManager",
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
    ) -> None:
        self.model = model
        self._url = ollama_url
        self._memory = memory
        self.conversation_history: list[dict] = []
        self.narrative_memory: str = ""
        self.bad_moves_context: list[dict] = []
        self._client = self._make_client()

    def _make_client(self):
        try:
            import httpx
            import ollama
            # 5 s connect timeout, 30 s read timeout — prevents blocking forever
            # when Ollama is cold-loading a model or swapping between models.
            return ollama.Client(
                host=self._url,
                timeout=httpx.Timeout(30.0, connect=5.0),
            )
        except Exception:
            return None

    def _chat(self, system: str, user: str, keep_history: bool = False) -> str:
        if self._client is None:
            return ""
        messages = [{"role": "system", "content": system}]
        if keep_history:
            messages.extend(self.conversation_history[-_MAX_HISTORY:])
        messages.append({"role": "user", "content": user})
        try:
            response = self._client.chat(model=self.model, messages=messages)
            reply = response.message.content or ""
            if keep_history:
                self.conversation_history.append({"role": "user", "content": user})
                self.conversation_history.append({"role": "assistant", "content": reply})
                self.conversation_history = self.conversation_history[-_MAX_HISTORY:]
            return reply.strip()
        except Exception:
            return ""

    def _strategy_context(self, board_fen: str) -> str:
        snippets = self._memory.retrieve_strategy(board_fen, n=2)
        if not snippets:
            return ""
        return "\n".join(f"- {s[:140]}" for s in snippets)

    def ask_for_move_opinion(
        self,
        board: "BoardState",
        legal_moves: list[dict],
        game_ai_suggestion: dict,
        recognition=None,
        endgame_state=None,
        audience: str = "human",  # "human" or "ai"
        move_history: list[str] | None = None,
        trajectory_context: str = "",
    ) -> tuple[str, str | None]:
        notations = [_move_to_notation(m) for m in legal_moves]
        ai_notation = _move_to_notation(game_ai_suggestion)
        ai_score = getattr(self, "_last_ai_score", None)
        score_hint = f"{ai_score:+.2f}" if ai_score is not None else "n/a"
        strategy = self._strategy_context(board.to_fen_string())

        user_parts = [
            "LEGAL MOVES:",
            "\n".join(notations),
            "",
            f"SIDE TO MOVE: {board.turn}",
            f"ENGINE TOP CHOICE: {ai_notation}",
            f"ENGINE SCORE: {score_hint}",
            "",
            _board_summary(board),
            "",
            "BOARD:",
            board.to_display_grid(),
        ]

        if move_history:
            user_parts.append(_move_history_block(move_history))

        if recognition and recognition.status not in ("inactive", "novel"):
            user_parts.append(_opening_context_block(recognition))

        if endgame_state and endgame_state.active:
            user_parts.append(_endgame_context_block(endgame_state))

        if trajectory_context:
            user_parts.extend(["", "TRAJECTORY HISTORY:", trajectory_context])

        if strategy:
            user_parts.extend(["", "STRATEGIC MEMORY:", strategy])

        phase_guide = _phase_strategy_for(board)
        if phase_guide:
            user_parts.extend(["", "PHASE STRATEGY GUIDE:", phase_guide])

        user_parts.extend([
            "",
            "Choose the best move from LEGAL MOVES.",
            "Return exactly:",
            "MOVE: <exact legal move>",
            "REASON: <one short sentence>",
        ])

        audience_note = (
            "\nAUDIENCE: You are playing against a human player. Keep commentary engaging and accessible."
            if audience == "human"
            else "\nAUDIENCE: You are playing against another AI engine. Use precise technical language."
        )
        system = _MOVE_SYSTEM + audience_note
        user = "\n".join(user_parts)
        reply = self._chat(system, user, keep_history=False)
        notation = self._parse_move(reply, notations)

        if notation is None and self._client is not None:
            retry_system = _BOARD_RULES + """
Reply with exactly one line:
MOVE: <exact legal move>
No other text.
"""
            retry_user = "LEGAL MOVES:\n" + "\n".join(notations)
            retry_reply = self._chat(retry_system, retry_user, keep_history=False)
            notation = self._parse_move(retry_reply, notations)
            if notation:
                reply = retry_reply

        return reply, notation

    def _parse_move(self, response: str, legal_notations: list[str]) -> str | None:
        for line in response.splitlines():
            clean = re.sub(r"\*+", "", line).strip()
            if re.match(r"(?i)^move\s*:", clean):
                candidate = re.sub(r"(?i)^move\s*:", "", clean).strip()
                match = self._match_notation(candidate, legal_notations)
                if match:
                    return match

        for token in re.findall(r"[a-g][1-7](?:-[a-g][1-7])?(?:x[a-g][1-7])?", response.lower()):
            match = self._match_notation(token, legal_notations)
            if match:
                return match
        return None

    @staticmethod
    def _match_notation(candidate: str, legal_notations: list[str]) -> str | None:
        candidate = re.sub(r"[.\s)\]]+$", "", candidate.strip().lower())
        if candidate in legal_notations:
            return candidate
        base = re.split(r"x", candidate)[0]
        for legal in legal_notations:
            if legal == base or legal.startswith(base + "x"):
                return legal
        return None

    def evaluate_human_move(
        self,
        board_before: "BoardState",
        human_move: dict,
        score_before: float,
        score_after: float,
        score_drop_threshold: float = 0.3,
        recognition=None,
        human_color: str = "",
        move_history: list[str] | None = None,
    ) -> str | None:
        delta = score_after - score_before
        if delta > -score_drop_threshold:
            return None

        move_notation = _move_to_notation(human_move)
        color_ctx = f"HUMAN PLAYS AS: {'White' if human_color == 'W' else 'Black'}\n" if human_color else ""
        user_parts = [
            f"{color_ctx}HUMAN MOVE: {move_notation}",
            f"SCORE CHANGE: {delta:+.2f}",
            "",
            _board_summary(board_before),
            "",
            "BOARD AFTER MOVE:",
            board_before.to_display_grid(),
        ]
        if move_history:
            user_parts.append(_move_history_block(move_history))
        if recognition and recognition.status not in ("inactive", "novel"):
            user_parts.append(_opening_context_block(recognition))
        reply = self._chat(_COMMENT_SYSTEM, "\n".join(user_parts), keep_history=False)
        if not reply or reply == "NO_COMMENT":
            return None
        return reply.strip()

    def announce_blunder(
        self, board: "BoardState", move: dict, move_history: list[str] | None = None
    ) -> str:
        move_notation = _move_to_notation(move)
        history_block = _move_history_block(move_history) if move_history else ""
        user = f"Deliberate bad move played: {move_notation}{history_block}\n\nBOARD:\n{board.to_display_grid()}"
        return self._chat(_BLUNDER_SYSTEM, user, keep_history=False)

    def record_human_feedback(self, board: "BoardState", move: dict, reason: str) -> None:
        self._memory.store_bad_move(
            board_fen=board.to_fen_string(),
            move=move,
            reason=reason,
            full_board_ascii=board.to_display_grid(),
        )
        self.bad_moves_context = self._memory.retrieve_similar_positions(
            board.to_fen_string(), n_results=5
        )

    def comment_on_good_move(
        self, board: "BoardState", move: dict, score: float,
        human_color: str = "", move_history: list[str] | None = None,
    ) -> str | None:
        move_notation = _move_to_notation(move)
        color_ctx = f"HUMAN PLAYS AS: {'White' if human_color == 'W' else 'Black'}\n" if human_color else ""
        history_block = _move_history_block(move_history) if move_history else ""
        user = (
            f"{color_ctx}HUMAN MOVE: {move_notation}\n"
            f"MOVE QUALITY (0=worst, 1=best): {score:.2f}\n"
            f"{history_block}\n"
            f"{_board_summary(board)}\n"
            f"\nBOARD:\n{board.to_display_grid()}"
        )
        reply = self._chat(_POSITIVE_COMMENT_SYSTEM, user, keep_history=False)
        if not reply or reply.strip() == "NO_COMMENT":
            return None
        return reply.strip()

    def comment_on_mill(
        self, board: "BoardState", move: dict,
        human_color: str = "", move_history: list[str] | None = None,
    ) -> str | None:
        move_notation = _move_to_notation(move)
        color_ctx = f"HUMAN PLAYS AS: {'White' if human_color == 'W' else 'Black'}\n" if human_color else ""
        history_block = _move_history_block(move_history) if move_history else ""
        user = (
            f"{color_ctx}HUMAN MILL + CAPTURE: {move_notation}{history_block}\n"
            f"\n{_board_summary(board)}\n"
            f"\nBOARD:\n{board.to_display_grid()}"
        )
        reply = self._chat(_MILL_COMMENT_SYSTEM, user, keep_history=False)
        return reply.strip() if reply else None

    def ask_strategic_question(
        self, board: "BoardState", human_color: str = "", move_history: list[str] | None = None,
    ) -> str | None:
        color_ctx = f"HUMAN PLAYS AS: {'White' if human_color == 'W' else 'Black'}\n" if human_color else ""
        history_block = _move_history_block(move_history) if move_history else ""
        user = (
            f"{color_ctx}{history_block}\n"
            f"{_board_summary(board)}\n"
            f"\nBOARD:\n{board.to_display_grid()}"
        )
        reply = self._chat(_POSITION_QUESTION_SYSTEM, user, keep_history=False)
        return reply.strip() if reply.strip() else None

    def generate_question_for_human(self, board: "BoardState") -> str | None:
        user = f"{_board_summary(board)}\n\nBOARD:\n{board.to_display_grid()}"
        reply = self._chat(_QUESTION_SYSTEM, user, keep_history=False)
        return reply.strip() if reply.strip() else None

    def player_chat(
        self, message: str, board: "BoardState", move_history: list[str] | None = None,
    ) -> str:
        """Respond to an in-game message from the human player."""
        history_block = _move_history_block(move_history) if move_history else ""
        user = (
            f"Player: {message}{history_block}\n"
            f"\n{_board_summary(board)}\n"
            f"\nCURRENT BOARD:\n{board.to_display_grid()}"
        )
        reply = self._chat(_PLAYER_CHAT_SYSTEM, user, keep_history=True)
        return reply.strip() if reply else ""

    def summarise_session(self, game_records: list[dict], facts_block: str = "") -> str:
        if not game_records:
            return ""
        lines = []
        for rec in game_records:
            moves = rec.get("moves", [])
            notations = [m.get("notation", "") for m in moves if m.get("notation")]
            move_seq = " ".join(notations) if notations else "none"
            lines.append(
                f"- winner={rec.get('winner', '?')} "
                f"opening={rec.get('recognised_opening_name') or rec.get('opening_name', 'unknown')} "
                f"total_moves={len(moves)} "
                f"move_sequence: {move_seq}"
            )
        user_content = "\n".join(lines)
        if facts_block:
            user_content = facts_block + "\n\n" + user_content
        return self._chat(_SESSION_SYSTEM, user_content, keep_history=False)

    def name_novel_opening(self, move_sequence: list[str]) -> str:
        move_str = ", ".join(move_sequence)
        reply = self._chat(
            _OPENING_NAME_SYSTEM,
            f"Opening sequence: {move_str}",
            keep_history=False,
        )
        name = reply.strip().strip("\"'")
        if not name:
            first = move_sequence[0] if move_sequence else "?"
            name = f"Novel Opening ({first}...)"
        return name

    def debrief_game(self, report, annotation=None) -> str:
        if annotation is not None:
            user = _build_debrief_prompt(report, annotation)
            return self._chat(_DEBRIEF_ANNOTATED_SYSTEM, user, keep_history=False)
        user = (
            f"Winner: {report.winner}\n"
            f"Loser: {report.loser}\n"
            f"Opening: {report.opening_name or 'unknown'}\n"
            f"Moves: {len(report.game_record.get('moves', []))}"
        )
        return self._chat(_DEBRIEF_GAME_SYSTEM, user, keep_history=False)

    def debrief_position(
        self,
        board: "BoardState",
        ply: int,
        move_played: dict,
        best_move: dict,
        score_played: float,
        score_best: float,
        is_critical: bool,
        opening_name: str | None,
        context: str,
    ) -> str:
        user = (
            f"Ply: {ply}\n"
            f"Played: {_move_to_notation(move_played)} score={score_played:+.2f}\n"
            f"Best: {_move_to_notation(best_move)} score={score_best:+.2f}\n"
            f"Critical: {'yes' if is_critical else 'no'}\n"
            f"Opening: {opening_name or 'unknown'}\n"
            f"Context: {context}\n\n"
            f"BOARD:\n{board.to_display_grid()}"
        )
        return self._chat(_DEBRIEF_POSITION_SYSTEM, user, keep_history=False)
