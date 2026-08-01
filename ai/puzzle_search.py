"""ai/puzzle_search.py — Endgame puzzle generation from WDL databases.

Generates Nine Men's Morris endgame puzzles by sampling positions from
loaded WDL tables and confirming forced wins via recursive minimax.

No DTM data required — win depth is computed by explicit game-tree search
with aggressive WDL pruning (only follow moves where the DB confirms the
winning side stays winning).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Optional

from game.board import BoardState, POSITIONS
from game.rules import get_all_legal_moves
from ai.endgame_solved_db import (
    EndgameSolvedDB,
    decode_position_id,
    get_wdl,
    WDL_WIN,
)

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Puzzle:
    puzzle_id: str
    source_db: str
    board_fen: str
    side_to_move: str
    winning_side: str
    target_win_in: int
    best_move: str
    solution_line: list[str]
    legal_moves: list[str]
    losing_moves: list[str]
    drawing_moves: list[str]
    hardness_score: float
    tags: list[str] = field(default_factory=list)

    @property
    def goal(self) -> str:
        side = "White" if self.winning_side == "W" else "Black"
        return f"{side} to move and win in {self.target_win_in}"

    def to_dict(self) -> dict:
        return {
            "id": self.puzzle_id,
            "source_db": self.source_db,
            "board_fen": self.board_fen,
            "side_to_move": self.side_to_move,
            "winning_side": self.winning_side,
            "goal": self.goal,
            "target_win_in": self.target_win_in,
            "best_move": self.best_move,
            "solution_line": self.solution_line,
            "legal_moves": self.legal_moves,
            "losing_moves": self.losing_moves,
            "drawing_moves": self.drawing_moves,
            "hardness_score": self.hardness_score,
            "tags": self.tags,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _notation(mv: dict) -> str:
    s = f"{mv['from']}-{mv['to']}" if mv.get("from") else mv.get("to", "")
    if mv.get("capture"):
        s += f"x{mv['capture']}"
    return s


def _board_from_pieces(
    white_pieces: list[str], black_pieces: list[str], turn: str
) -> BoardState:
    positions = {pos: "" for pos in POSITIONS}
    for p in white_pieces:
        positions[p] = "W"
    for p in black_pieces:
        positions[p] = "B"
    return BoardState.from_setup(positions, turn, "move")


def _is_terminal(board: BoardState, side_just_moved: str) -> bool:
    """True if the game has ended after side_just_moved played."""
    opponent = "B" if side_just_moved == "W" else "W"
    if board.pieces_on_board[opponent] < 3:
        return True
    if not get_all_legal_moves(board):
        return True
    return False


# ── Win-depth search (minimax over WDL) ──────────────────────────────────────


def find_win_depth(
    db: EndgameSolvedDB,
    board: BoardState,
    winning_side: str,
    max_depth: int,
    memo: dict | None = None,
) -> int | None:
    """Return minimum winning_side moves to force win, or None if > max_depth.

    `board.turn` may be either side.
    `max_depth` counts only winning_side moves (not opponent replies).
    """
    if memo is None:
        memo = {}
    return _search(db, board, winning_side, max_depth, memo)


def _search(db, board, ws, budget, memo):
    fen = board.to_fen_string()
    key = (fen, budget)
    if key in memo:
        return memo[key]
    result = _search_inner(db, board, ws, budget, memo)
    memo[key] = result
    return result


def _search_inner(db, board, ws, budget, memo):
    opp = "B" if ws == "W" else "W"
    moves = get_all_legal_moves(board)

    if board.turn == ws:
        if not moves or budget <= 0:
            return None
        best = None
        for mv in moves:
            child = board.apply_move(mv)
            # Terminal: opponent eliminated or blocked
            if child.pieces_on_board[opp] < 3 or not get_all_legal_moves(child):
                best = 1
                break  # can't do better than win-in-1
            # WDL prune: skip moves that don't maintain the forced win
            if db.query(child) != "L":
                continue
            d = _search(db, child, ws, budget - 1, memo)
            if d is not None:
                total = 1 + d
                if best is None or total < best:
                    best = total
        return best

    else:  # opponent's turn — worst-case (maximise depth for us)
        if not moves:
            return 0  # opponent has no legal moves; winning side already won
        worst = 0
        for mv in moves:
            child = board.apply_move(mv)
            if child.pieces_on_board[ws] < 3:
                return None  # winning side got eliminated
            if db.query(child) != "W":
                return None  # winning side no longer winning after this reply
            d = _search(db, child, ws, budget, memo)
            if d is None:
                return None
            worst = max(worst, d)
        return worst


# ── Principal variation extraction ───────────────────────────────────────────


def get_solution_line(
    db: EndgameSolvedDB,
    board: BoardState,
    winning_side: str,
    depth: int,
    memo: dict,
) -> list[str]:
    """Return PV: winning moves interleaved with opponent's best defense."""
    line: list[str] = []
    _pv(db, board, winning_side, depth, memo, line)
    return line


def _pv(db, board, ws, budget, memo, line):
    opp = "B" if ws == "W" else "W"
    moves = get_all_legal_moves(board)

    if board.turn == ws:
        if not moves or budget <= 0:
            return
        for mv in moves:
            child = board.apply_move(mv)
            note = _notation(mv)
            if child.pieces_on_board[opp] < 3 or not get_all_legal_moves(child):
                line.append(note)
                return
            if db.query(child) != "L":
                continue
            d = _search(db, child, ws, budget - 1, memo)
            if d is not None:
                line.append(note)
                _pv(db, child, ws, d, memo, line)
                return

    else:  # opponent picks hardest defense
        if not moves:
            return
        best_mv, best_child, best_d = None, None, -1
        for mv in moves:
            child = board.apply_move(mv)
            if child.pieces_on_board[ws] < 3:
                continue
            if db.query(child) != "W":
                continue
            d = _search(db, child, ws, budget, memo)
            if d is not None and d > best_d:
                best_d, best_mv, best_child = d, mv, child
        if best_mv is not None:
            line.append(_notation(best_mv))
            _pv(db, best_child, ws, best_d, memo, line)


# ── Hardness scoring ──────────────────────────────────────────────────────────


def score_hardness(
    db: EndgameSolvedDB,
    board: BoardState,
    winning_side: str,
    target_depth: int,
    memo: dict,
) -> tuple[float, list[str], list[str], list[str]]:
    """Return (hardness_score, winning_moves, drawing_moves, losing_moves)."""
    opp = "B" if winning_side == "W" else "W"
    winning, drawing, losing = [], [], []

    for mv in get_all_legal_moves(board):
        child = board.apply_move(mv)
        note = _notation(mv)

        if child.pieces_on_board[opp] < 3 or not get_all_legal_moves(child):
            winning.append(note)
            continue

        wdl = db.query(child)
        if wdl == "L":
            d = _search(db, child, winning_side, target_depth - 1, memo)
            if d is not None:
                winning.append(note)
            else:
                drawing.append(note)  # wins eventually, but past our depth budget
        elif wdl == "D":
            drawing.append(note)
        else:
            losing.append(note)  # None or "W" — opponent gains/keeps advantage

    n_winning = len(winning)
    n_legal   = n_winning + len(drawing) + len(losing)
    score = 0.0

    if n_winning == 1:
        score += 4.0   # single winning path — most difficult
    elif n_winning == 2:
        score += 2.5   # two paths — still hard

    if len(losing) > 0:
        score += 1.0   # wrong moves hand the game away
    if len(drawing) + len(losing) == n_legal - n_winning:
        score += 2.0   # every non-winning move fails
    if n_legal >= 6:
        score += 1.0   # high branching — harder to spot the right move

    # Bonus for a quiet (non-capture) winning move — harder to spot
    if n_winning <= 2:
        all_mvs = get_all_legal_moves(board)
        quiet_wins = [
            mv for mv in all_mvs
            if _notation(mv) in winning and not mv.get("capture")
        ]
        if quiet_wins:
            score += 1.0

    return score, winning, drawing, losing


# ── Puzzle generation ─────────────────────────────────────────────────────────


def _table_size(nW: int, nB: int) -> int:
    from math import comb
    return comb(24, nW) * comb(24 - nW, nB) * 2


def load_puzzle_db(db_dir, nW: int, nB: int, max_depth: int = 4) -> EndgameSolvedDB:
    """Load only the WDL tables needed to generate puzzles starting from (nW, nB).

    Winning side can make at most max_depth captures; opponent at most max_depth-1.
    Tables outside that range are never probed by find_win_depth.
    """
    from pathlib import Path
    from ai.endgame_solved_db import _expected_bytes
    import logging

    log = logging.getLogger(__name__)
    db = EndgameSolvedDB(None)   # empty DB

    db_path = Path(db_dir)
    # Load tables for (nW', nB') where pieces could be reached via captures
    for dW in range(max_depth):          # winning_side loses dW pieces to opponent
        for dB in range(max_depth + 1):  # opponent loses dB pieces to winning_side
            nW2, nB2 = nW - dW, nB - dB
            if nW2 < 3 or nB2 < 3:
                continue
            path = db_path / f"endgame_{nW2}_{nB2}.wdl"
            if not path.exists():
                continue
            expected = _expected_bytes(nW2, nB2)
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if len(data) != expected:
                log.warning("puzzle_search: %s wrong size — skipped", path.name)
                continue
            db._tables[(nW2, nB2)] = data
            log.info("puzzle_search: loaded %s", path.name)

    return db


def generate_puzzle(
    db: EndgameSolvedDB,
    nW: int,
    nB: int,
    winning_side: str,
    target_depth: int,
    max_attempts: int = 5000,
    max_winning_moves: int = 2,
) -> Puzzle | None:
    """Sample random positions from (nW, nB) table and return the first valid puzzle.

    winning_side:      "W" | "B" | "random"
    target_depth:      3 | 4 | 5 | 0=random (number of winning_side moves to force win)
    max_winning_moves: only accept positions with this many or fewer winning first moves
    """
    if winning_side == "random":
        winning_side = random.choice(["W", "B"])
    if target_depth == 0:
        target_depth = random.choice([3, 4, 5, 6, 7])

    table = db._tables.get((nW, nB))
    if table is None:
        return None

    size = _table_size(nW, nB)
    turn_bit = 0 if winning_side == "W" else 1
    half = size // 2

    candidates = random.sample(range(half), min(max_attempts, half))
    candidates = [i * 2 + turn_bit for i in candidates]

    memo: dict = {}

    for pos_id in candidates:
        if get_wdl(table, pos_id) != WDL_WIN:
            continue

        try:
            white_pieces, black_pieces, turn = decode_position_id(pos_id, nW, nB)
        except Exception:
            continue

        board = _board_from_pieces(white_pieces, black_pieces, turn)

        if db.query(board) != "W":
            continue

        # Confirm exact win depth: must equal target_depth, not less
        depth = find_win_depth(db, board, winning_side, target_depth, memo)
        if depth != target_depth:
            continue
        if target_depth > 1:
            if find_win_depth(db, board, winning_side, target_depth - 1, memo) is not None:
                continue

        h_score, winning_mvs, drawing_mvs, losing_mvs = score_hardness(
            db, board, winning_side, target_depth, memo
        )

        # Reject positions that are too easy (too many winning first moves)
        if len(winning_mvs) > max_winning_moves:
            continue

        solution = get_solution_line(db, board, winning_side, target_depth, memo)

        fen = board.to_fen_string()
        pid = "egdb_" + hashlib.md5(fen.encode()).hexdigest()[:8]

        all_legal = [_notation(m) for m in get_all_legal_moves(board)]
        tags = ["endgame", f"win-in-{target_depth}"]
        if len(winning_mvs) == 1:
            tags += ["unique-move", "bottleneck"]
        elif len(winning_mvs) == 2:
            tags.append("two-solutions")
        if len(all_legal) >= 6:
            tags.append("high-branching")

        return Puzzle(
            puzzle_id=pid,
            source_db=f"data/endgame/endgame_{nW}_{nB}.wdl",
            board_fen=fen,
            side_to_move=winning_side,
            winning_side=winning_side,
            target_win_in=target_depth,
            best_move=winning_mvs[0] if winning_mvs else "",
            solution_line=solution,
            legal_moves=all_legal,
            losing_moves=losing_mvs,
            drawing_moves=drawing_mvs,
            hardness_score=h_score,
            tags=tags,
        )

    return None
