"""
ai/ponder.py — B-75: Background search during the opponent's turn.

After the AI makes a move, predict the most likely opponent reply and start
a full-depth search from that position.  If the human plays the predicted
move, the result is used immediately (ponder hit); otherwise it is discarded
and a fresh search runs as normal.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.board import BoardState
    from ai.game_ai import GameAI

log = logging.getLogger(__name__)


class PonderManager:
    """Manages a single background search thread during the opponent's turn."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._ponder_ai: GameAI | None = None
        self._predicted_hash: int | None = None
        self._cached_move: dict | None = None
        self._completed_ponder_ai: GameAI | None = None  # B-94: retained after stop()
        self._lock = threading.Lock()

    def start(
        self,
        board: BoardState,         # board after AI's move — opponent to move here
        game_ai: GameAI,           # main AI (source of config: difficulty, weights, VN)
        game_notations: list[str], # full move-notation list up to and including AI's move
        trajectory_db=None,
        fullgame_db=None,
        endgame_state=None,
        ngram_model=None,          # SE-13: NGramOpponentModel | None
    ) -> None:
        """Predict the opponent's best reply and begin searching the response.

        Uses priority-based move ordering to predict the opponent move, with an
        optional value-network re-score of the top 3 candidates.
        """
        self.stop()  # cancel any previously running ponder

        from game.rules import get_all_legal_moves
        from ai.game_ai import GameAI, _order_moves

        opp_moves = get_all_legal_moves(board)
        if not opp_moves:
            return

        # Predict opponent move: priority order, optionally refined by value
        # network, trajectory-DB frequency, and fullgame-DB best move (B-93).
        ordered = _order_moves(board, opp_moves, None, None)
        predicted_move = ordered[0]

        if game_ai._value_net is not None and len(ordered) >= 2:
            candidates = ordered[:min(3, len(ordered))]
            best_vn: float | None = None
            for m in candidates:
                nb = board.apply_move(m)
                vn = game_ai._value_net.predict(nb, board.turn)  # from opponent's POV
                if best_vn is None or vn > best_vn:
                    best_vn = vn
                    predicted_move = m

        # B-93/SE-13: blend trajectory-DB frequency, fullgame-DB best move,
        # and n-gram opponent model into prediction scoring.
        # Each candidate gets a base score from its priority rank, then receives
        # additive boosts from whichever sources are available.
        if trajectory_db is not None or fullgame_db is not None or ngram_model is not None:
            freq_scores: dict[str, float] = {}
            if trajectory_db is not None:
                try:
                    freq_scores = trajectory_db.query_all_frequencies(board)
                except Exception:
                    pass

            fgdb_best: str | None = None
            if fullgame_db is not None:
                try:
                    fgdb_best = fullgame_db.best_move_validated(board)
                except Exception:
                    pass

            ngram_scores: dict[str, float] = {}
            if ngram_model is not None:
                try:
                    ngram_scores = ngram_model.predict(board.turn, game_notations)
                except Exception:
                    pass

            if freq_scores or fgdb_best or ngram_scores:
                best_score: float | None = None
                for i, m in enumerate(ordered):
                    notation = _move_notation(m)
                    score = float(-i)
                    score += freq_scores.get(notation, 0.0) * 5.0
                    if fgdb_best is not None and notation == fgdb_best:
                        score += 3.0
                    score += ngram_scores.get(notation, 0.0) * 4.0  # SE-13
                    if best_score is None or score > best_score:
                        best_score = score
                        predicted_move = m

        ponder_board = board.apply_move(predicted_move)
        predicted_hash = ponder_board.hash_key
        pred_notation = _move_notation(predicted_move)

        # Shadow AI: same config, fresh transposition table (avoids contaminating main).
        ponder_ai = GameAI(
            color=game_ai.color,
            difficulty=game_ai.difficulty,
            weights=game_ai._weights,
            value_net=game_ai._value_net,
            fullgame_db=fullgame_db,
            endgame_solved_db=game_ai._endgame_solved_db,
        )

        with self._lock:
            self._ponder_ai = ponder_ai
            self._predicted_hash = predicted_hash
            self._cached_move = None
            self._completed_ponder_ai = None

        ponder_notations = list(game_notations) + [pred_notation]

        def _run() -> None:
            try:
                move = ponder_ai.choose_move(
                    ponder_board,
                    endgame_state=endgame_state,
                    trajectory_db=trajectory_db,
                    game_notations=ponder_notations,
                    fullgame_db=fullgame_db,
                )
                with self._lock:
                    if self._predicted_hash == predicted_hash:
                        self._cached_move = move
                        self._completed_ponder_ai = ponder_ai  # B-94: expose for TT reuse
                        log.info(
                            "Ponder complete: opp %s → cached AI reply %s",
                            pred_notation,
                            _move_notation(move) if move else "None",
                        )
            except Exception as exc:
                log.debug("Ponder aborted: %s", exc)

        self._thread = threading.Thread(target=_run, daemon=True, name="ponder")
        self._thread.start()
        log.info("Ponder started: expecting opponent to play %s", pred_notation)

    def stop(self) -> None:
        """Interrupt the ponder search and wait up to 0.5 s for thread exit."""
        with self._lock:
            ai = self._ponder_ai
        if ai is not None:
            ai.force_stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None
        with self._lock:
            self._ponder_ai = None

    def get_result(self, board: BoardState) -> tuple[dict, GameAI | None] | None:
        """Return (cached_move, completed_ponder_ai) if board matches the predicted position.

        B-94: the completed_ponder_ai carries a pre-warmed TT; the caller may
        reset its _force_stop/_deadline and call _iterative_deepen() to deepen
        the search cheaply.  Returns None on miss or if ponder is incomplete.
        Must be called AFTER stop() so there is no concurrent write race.
        """
        with self._lock:
            if self._cached_move is None:
                return None
            if board.hash_key != self._predicted_hash:
                log.debug(
                    "Ponder miss: predicted hash %s, actual hash %s",
                    self._predicted_hash, board.hash_key,
                )
                return None
            log.info("Ponder hit — TT pre-warmed for deepening (B-94)")
            return self._cached_move, self._completed_ponder_ai

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


def _move_notation(move: dict) -> str:
    s = f"{move['from']}-{move['to']}" if move.get("from") else move.get("to", "")
    if move.get("capture"):
        s += f"x{move['capture']}"
    return s
