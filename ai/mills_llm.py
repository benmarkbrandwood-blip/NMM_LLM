"""ai/mills_llm.py — Ollama interface for MillsLLM commentary."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.board import BoardState
    from ai.memory_manager import MemoryManager

_MAX_HISTORY = 20

_GAME_CONTEXT = """\
You are MillsAI, the AI for a Nine Men's Morris game (also called Mills, Mühle, or Merels).

THIS IS NOT CHESS. Do not think about chess pieces, chess rules, or chess strategy.

NINE MEN'S MORRIS — QUICK RULES:
- Board: 24 nodes on three concentric squares linked by lines.
- The 24 valid node names are EXACTLY:
    Outer ring: a7  d7  g7  g4  g1  d1  a1  a4
    Middle ring: b6  d6  f6  f4  f2  d2  b2  b4
    Inner ring:  c5  d5  e5  e4  e3  d3  c3  c4
- Each player has 9 pieces.
- Phase "place": players take turns placing a piece on any empty node.
- Phase "move":  players slide a piece to an adjacent connected node.
- Phase "fly":   player with only 3 pieces may jump to any empty node.
- MILL: 3 own pieces in a row along a board line → remove one enemy piece.
- WIN: enemy reduced to 2 pieces, or enemy has no legal move.

NOTATION (lower-case, no spaces):
  Placement:            d2          (place on node d2)
  Movement:             a4-a7       (slide from a4 to a7)
  Placement + capture:  d2xb6       (place on d2 then capture b6)
  Movement + capture:   a4-a7xb6    (move to a7 then capture b6)
"""

_MOVE_SYSTEM = _GAME_CONTEXT + """
TASK: Choose the best move for the current player from the LEGAL MOVES list.

OUTPUT FORMAT — follow this EXACTLY, two lines only:
MOVE: <exact notation from the legal moves list>
REASON: <one sentence>

RULES:
- Your MOVE must be copied EXACTLY from the LEGAL MOVES list — do not alter it.
- Do not invent node names. Only the 24 nodes listed above are valid.
- Do not output anything before the MOVE line.
"""

_COMMENT_SYSTEM = _GAME_CONTEXT + """
TASK: Comment on the human's last move in Nine Men's Morris.
- Write 1–2 short sentences about mill-formation risk or strategic impact.
- Do NOT suggest a move. Do NOT mention chess.
- If the move was fine, reply with exactly: NO_COMMENT
"""


def _endgame_context_block(endgame_state) -> str:
    lines = [
        "\n--- Endgame Context ---",
        f"Phase:         {endgame_state.phase}",
        f"Pieces:        W={endgame_state.pieces_white}  B={endgame_state.pieces_black}"
        f"  (total {endgame_state.total_pieces})",
        f"Mobility:      W={endgame_state.mobility_white} moves  "
        f"B={endgame_state.mobility_black} moves",
        f"Zugzwang risk: {'YES' if endgame_state.zugzwang_risk else 'no'}",
        f"Pattern:       {endgame_state.pattern or 'none'}"
        + (f" — {endgame_state.pattern_notes}" if endgame_state.pattern_notes else ""),
        "---\n",
    ]
    return "\n".join(lines)


def _opening_context_block(recognition) -> str:
    lines = [
        "\n--- Opening Context ---",
        f"Recognised opening: {recognition.name or 'Unknown / Novel'}",
        f"Family:             {recognition.family or '—'}",
        f"Status:             {recognition.status} (confidence {recognition.confidence:.0%})",
        f"Book move this ply: {recognition.book_move or 'none / exhausted'}",
        f"Strategic purpose:  {recognition.strategic_notes or '—'}",
        f"Common blunders:    {', '.join(recognition.common_blunders) if recognition.common_blunders else 'none on record'}",
        "---\n",
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
    # Partial match: LLM gave destination only (no capture), match first legal to that square
    for m in legal:
        if m["to"] == notation and not m.get("capture"):
            return m
    return None


class MillsLLM:
    def __init__(
        self,
        memory: "MemoryManager",
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2",
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
            import ollama
            return ollama.Client(host=self._url)
        except Exception:
            return None

    def _chat(self, system: str, user: str, keep_history: bool = False) -> str:
        if self._client is None:
            return ""
        messages = [{"role": "system", "content": system}]
        if keep_history:
            for turn in self.conversation_history[-_MAX_HISTORY:]:
                messages.append(turn)
        messages.append({"role": "user", "content": user})
        try:
            response = self._client.chat(model=self.model, messages=messages)
            reply = response.message.content or ""
            if keep_history:
                self.conversation_history.append({"role": "user", "content": user})
                self.conversation_history.append({"role": "assistant", "content": reply})
                if len(self.conversation_history) > _MAX_HISTORY * 2:
                    self.conversation_history = self.conversation_history[-_MAX_HISTORY * 2:]
            return reply
        except Exception:
            return ""

    def _strategy_context(self, board_fen: str) -> str:
        snippets = self._memory.retrieve_strategy(board_fen, n=2)
        if not snippets:
            return ""
        return "\n".join(f"- {s[:120]}" for s in snippets)

    # ── Move recommendation ───────────────────────────────────────────────────

    def ask_for_move_opinion(
        self,
        board: "BoardState",
        legal_moves: list[dict],
        game_ai_suggestion: dict,
        recognition=None,
        endgame_state=None,
    ) -> tuple[str, str | None]:
        """
        Returns (response_text, recommended_notation_or_None).
        recommended_notation is already validated as a legal move notation string.
        """
        notations = [_move_to_notation(m) for m in legal_moves]
        ai_notation = _move_to_notation(game_ai_suggestion)
        ai_score = getattr(self, "_last_ai_score", None)
        score_hint = f" (engine score: {ai_score:.2f})" if ai_score is not None else ""

        strategy = self._strategy_context(board.to_fen_string())

        # Legal moves go first so the LLM sees them before anything else.
        legal_block = "LEGAL MOVES (choose one of these EXACT strings):\n  " + \
                      "  ".join(notations)

        user = (
            f"{legal_block}\n\n"
            f"Engine top choice: {ai_notation}{score_hint}\n"
            f"Player to move:    {board.turn}\n\n"
            f"Board:\n{board.to_display_grid()}\n"
        )
        if recognition and recognition.status not in ("inactive", "novel"):
            user += _opening_context_block(recognition)
        if endgame_state and endgame_state.active:
            user += _endgame_context_block(endgame_state)
        if strategy:
            user += f"\nStrategy hints:\n{strategy}\n"
        user += (
            f"\nPick the single best move from the LEGAL MOVES list above.\n"
            f"Output format:\nMOVE: <exact notation>\nREASON: <one sentence>"
        )

        reply = self._chat(_MOVE_SYSTEM, user, keep_history=False)
        notation = self._parse_move(reply, notations)

        # Retry with a minimal prompt if the first pass failed to parse.
        if notation is None and self._client is not None:
            retry_system = (
                "You are playing Nine Men's Morris (not chess). "
                "Output ONLY one move notation from the list, nothing else."
            )
            retry_user = (
                f"Legal moves: {', '.join(notations)}\n"
                f"Respond with just one notation from that list, e.g.: {notations[0]}"
            )
            retry_reply = self._chat(retry_system, retry_user, keep_history=False)
            notation = self._parse_move(retry_reply, notations)
            if notation:
                reply = retry_reply  # use the cleaner reply for reasoning display

        return reply, notation

    def _parse_move(self, response: str, legal_notations: list[str]) -> str | None:
        # Pass 1: look for an explicit "MOVE:" line (handles markdown bold, mixed case)
        for line in response.splitlines():
            # Strip markdown bold/italic markers before checking
            clean = re.sub(r"\*+", "", line).strip()
            if re.match(r"(?i)move\s*:", clean):
                candidate = re.sub(r"(?i)move\s*:", "", clean).strip()
                match = self._match_notation(candidate, legal_notations)
                if match:
                    return match

        # Pass 2: scan every token in the response for a valid notation
        for token in re.findall(r"[a-g][1-7](?:-[a-g][1-7])?(?:x[a-g][1-7])?", response.lower()):
            match = self._match_notation(token, legal_notations)
            if match:
                return match

        return None

    @staticmethod
    def _match_notation(candidate: str, legal_notations: list[str]) -> str | None:
        # Strip trailing punctuation and whitespace
        candidate = re.sub(r"[.\s)\]]+$", "", candidate.strip().lower())
        if candidate in legal_notations:
            return candidate
        # Destination-only: match first legal move to that square (no capture)
        base = re.split(r"x", candidate)[0]
        for legal in legal_notations:
            if legal == base or legal.startswith(base + "x"):
                return legal
        return None

    # ── Human move commentary ─────────────────────────────────────────────────

    def evaluate_human_move(
        self,
        board_before: "BoardState",
        human_move: dict,
        score_before: float,
        score_after: float,
        score_drop_threshold: float = 0.3,
        recognition=None,
    ) -> str | None:
        delta = score_after - score_before
        if delta > -score_drop_threshold:
            return None

        move_notation = _move_to_notation(human_move)
        user = (
            f"Human played: {move_notation}  (score dropped {abs(delta):.2f})\n\n"
            f"Board after the move:\n{board_before.to_display_grid()}\n"
        )
        if recognition and recognition.status not in ("inactive", "novel"):
            user += _opening_context_block(recognition)
        user += "Comment on the strategic risk of this move."
        reply = self._chat(_COMMENT_SYSTEM, user, keep_history=False)
        if not reply.strip() or reply.strip() == "NO_COMMENT":
            return None
        return reply.strip()

    # ── Blunder announcement ──────────────────────────────────────────────────

    def announce_blunder(self, board: "BoardState", move: dict) -> str:
        move_notation = _move_to_notation(move)
        system = (
            _GAME_CONTEXT +
            "You just made a deliberate mistake to help the human learn Nine Men's Morris. "
            "In 1–2 sentences, tell them what you played and invite them to spot the better move. "
            "Do NOT reveal the correct move."
        )
        user = (
            f"I played {move_notation}. That was a deliberate mistake.\n"
            f"Board:\n{board.to_display_grid()}"
        )
        return self._chat(system, user, keep_history=False)

    # ── Feedback & memory ─────────────────────────────────────────────────────

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

    def generate_question_for_human(self, board: "BoardState") -> str | None:
        system = (
            "You are MillsAI. Ask the human ONE brief, curious question (max 20 words) "
            "about their strategy or plan. Do not lecture."
        )
        user = f"Board:\n{board.to_display_grid()}"
        reply = self._chat(system, user, keep_history=False)
        return reply.strip() if reply.strip() else None

    # ── Session summary ───────────────────────────────────────────────────────

    def summarise_session(self, game_records: list[dict]) -> str:
        if not game_records:
            return ""
        lines = []
        for rec in game_records:
            lines.append(
                f"- Winner: {rec.get('winner', '?')}, "
                f"Moves: {len(rec.get('moves', []))}"
            )
        system = (
            "You are MillsAI. Write a short markdown session summary (## Session header, "
            "3–5 bullet points: who won, key patterns, lessons)."
        )
        return self._chat(system, "\n".join(lines), keep_history=False)

    # ── Opening naming ────────────────────────────────────────────────────────

    def name_novel_opening(self, move_sequence: list[str]) -> str:
        """
        Ask the LLM to invent a creative name for a novel opening sequence.
        Returns a short name string (fallback to a positional label if offline).
        """
        system = (
            _GAME_CONTEXT +
            "You have discovered a novel Nine Men's Morris opening sequence. "
            "Invent a short, evocative name for it (2–4 words) in the style of "
            "traditional game opening names (e.g. 'Diagonal Thrust', "
            "'Corner Rush Defence', 'Central Mill Gambit'). "
            "Reply with ONLY the name — no explanation, no punctuation at the end."
        )
        move_str = ", ".join(move_sequence)
        user = (
            f"Opening move sequence ({len(move_sequence)} placements): {move_str}\n"
            "Name this opening."
        )
        reply = self._chat(system, user, keep_history=False)
        name = reply.strip().strip("\"'")
        if not name:
            first = move_sequence[0] if move_sequence else "?"
            name = f"Novel Opening ({first}...)"
        return name

    # ── Debrief ───────────────────────────────────────────────────────────────

    def debrief_game(self, report) -> str:
        system = (
            "You are MillsAI coaching a player through a completed game. "
            "Write a structured debrief with these sections:\n"
            "## Result\n## Key Turning Point\n## Where the loser went wrong\n## Lessons"
        )
        user = (
            f"Winner: {report.winner}  Loser: {report.loser}\n"
            f"Opening: {report.opening_name or 'unknown'}\n"
            f"Total moves: {len(report.game_record.get('moves', []))}\n"
            "Write the debrief."
        )
        return self._chat(system, user, keep_history=False)

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
        system = (
            "You are MillsAI coaching a player through a game replay. "
            "Comment on this position in 2–3 sentences. "
            "If it is a turning point, explain why clearly."
        )
        user = (
            f"Ply {ply}: {board.turn} played {_move_to_notation(move_played)} "
            f"(score {score_played:+.2f})\n"
            f"Best was {_move_to_notation(best_move)} (score {score_best:+.2f})\n"
            f"Turning point: {is_critical}\n"
            f"Board:\n{board.to_display_grid()}"
        )
        return self._chat(system, user, keep_history=False)
