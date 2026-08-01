"""ai/malom_puzzle_search.py — Malom-based puzzle generation for midgame positions.

Generates Nine Men's Morris movement-phase puzzles using the Malom ultra-strong
perfect database. Targets positions with a forced win in 4–7 moves for the
current mover, covering mid-game configurations (4–9 pieces per side).

Design notes
------------
Full minimax is too expensive in Python for midgame positions (branching ~15).
Instead we use a GREEDY solution extraction guided by Malom WDL:

  • Winning side: always play the move that minimises the opponent's remaining dtw
    (deepest threat first — good pedagogically and usually agrees with optimal).
  • Opponent side: try ALL moves; if any escape to non-losing, reject the position
    (only keep positions where every opponent reply keeps them losing).  Among
    all losing-for-opponent replies, pick the one with the highest dtw (hardest
    defense) as the representative response line shown to the solver.

The actual win depth is counted from the greedy line (not from dtw directly),
since Malom's dtw field semantics differ across midgame sectors ("relative
quality hint", per the DB documentation).  dtw is used only as a coarse
pre-filter.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Optional

from game.board import BoardState, POSITIONS, MILLS
from game.rules import get_all_legal_moves
from ai.malom_db import MalomDB, _get_hash_state


# ── Hash cache pre-warming ────────────────────────────────────────────────────
# HashState(W, B) init time scales with C(24, W): W=7 ≈ 0.77s, W=8 ≈ 1.74s.
# Pre-warming all 25 pairs for W,B ∈ [3,7] costs ≈ 6 s total (one-time).
# After warming, every db.query() for those piece counts is essentially free.

_PREWARM_DONE = False


def prewarm_hash_cache(max_pieces: int = 7) -> None:
    """Pre-initialise Malom hash states for (W, B) in 3..max_pieces.

    Safe to call multiple times; only runs if max_pieces exceeds the previous
    call.  After this, MalomDB.query() for positions in the covered range
    returns in microseconds (hash lookup, no re-initialisation cost).
    """
    global _PREWARM_DONE
    if _PREWARM_DONE and max_pieces <= 7:
        return
    for w in range(3, max_pieces + 1):
        for b in range(3, max_pieces + 1):
            _get_hash_state(w, b)
    _PREWARM_DONE = True


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class MalomPuzzle:
    puzzle_id: str
    board_fen: str
    side_to_move: str
    winning_side: str
    target_win_in: int
    best_move: str
    solution_line: list[str]
    legal_moves: list[str]
    winning_moves: list[str]
    losing_moves: list[str]
    drawing_moves: list[str]
    hardness_score: float
    source_db: str = "malom"
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
            "winning_moves": self.winning_moves,
            "losing_moves": self.losing_moves,
            "drawing_moves": self.drawing_moves,
            "hardness_score": self.hardness_score,
            "tags": self.tags,
        }


# ── Win-depth search (used by validate endpoint) ─────────────────────────────

def find_win_depth(
    db: MalomDB,
    board: BoardState,
    winning_side: str,
    max_depth: int,
) -> Optional[int]:
    """Return minimum winning_side moves to force a win, or None if > max_depth.

    Uses Malom WDL as oracle; does not recurse deeper than max_depth winning-side moves.
    """
    opp = "B" if winning_side == "W" else "W"

    def _search(cur: BoardState, budget: int) -> Optional[int]:
        moves = get_all_legal_moves(cur)
        if cur.turn == winning_side:
            if not moves or budget <= 0:
                return None
            best = None
            for mv in moves:
                child = cur.apply_move(mv)
                if child.pieces_on_board[opp] < 3 or not get_all_legal_moves(child):
                    return 1
                if _malom_wdl(db, child) != "L":
                    continue
                d = _search(child, budget - 1)
                if d is not None:
                    total = 1 + d
                    if best is None or total < best:
                        best = total
            return best
        else:
            if not moves:
                return 0
            worst = 0
            for mv in moves:
                child = cur.apply_move(mv)
                if child.pieces_on_board[winning_side] < 3:
                    return None
                if _malom_wdl(db, child) != "W":
                    return None
                d = _search(child, budget)
                if d is None:
                    return None
                worst = max(worst, d)
            return worst

    return _search(board, max_depth)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _notation(mv: dict) -> str:
    s = f"{mv['from']}-{mv['to']}" if mv.get("from") else mv.get("to", "")
    if mv.get("capture"):
        s += f"x{mv['capture']}"
    return s


def _malom_wdl(db: MalomDB, board: BoardState) -> Optional[str]:
    """Return 'W'/'L'/'D' for the current mover, or None if sector unavailable."""
    result = db.query(board)
    return result["outcome"] if result else None


def _malom_dtw(db: MalomDB, board: BoardState) -> int:
    """Return dtw from db.query, or a large default if unavailable."""
    result = db.query(board)
    return result.get("dtw", 0) if result else 0


def _has_mill(board: BoardState, color: str) -> bool:
    for mill in MILLS:
        if all(board.positions[p] == color for p in mill):
            return True
    return False


# ── Greedy solution extraction ────────────────────────────────────────────────

def extract_solution_line(
    db: MalomDB,
    board: BoardState,
    ws: str,
    max_depth: int = 9,
) -> Optional[tuple[list[str], int]]:
    """Extract a solution line greedily using Malom WDL + dtw for move selection.

    Returns (line, win_depth) where win_depth is the number of winning-side moves
    in the line, or None if no forced-win path is found within max_depth.

    Winning side picks the move that minimises the opponent's dtw (greediest
    threat).  Opponent tries ALL replies; if any reply leads to a position
    where the winning side is no longer winning (per Malom), None is returned
    immediately (position is not a forced win).  Among forced-losing replies,
    the one with the highest dtw (hardest defense) is chosen for the line.
    """
    opp = "B" if ws == "W" else "W"
    line: list[str] = []
    current = board
    ws_move_count = 0

    for _ in range(max_depth * 2 + 1):
        moves = get_all_legal_moves(current)

        if current.turn == ws:
            # Winning side's turn
            if not moves:
                return None  # winning side blocked — unexpected
            best_mv, best_child, best_dtw = None, None, 9999
            for mv in moves:
                child = current.apply_move(mv)
                # Immediate terminal: opponent eliminated or stuck
                if child.pieces_on_board[opp] < 3 or not get_all_legal_moves(child):
                    line.append(_notation(mv))
                    return (line, ws_move_count + 1)
                r = db.query(child)
                if r and r["outcome"] == "L":
                    dtw = r.get("dtw", 9999)
                    if dtw < best_dtw:
                        best_dtw = dtw
                        best_mv, best_child = mv, child
            if best_mv is None:
                return None  # no winning move found
            line.append(_notation(best_mv))
            current = best_child
            ws_move_count += 1
            if ws_move_count > max_depth:
                return None

        else:
            # Opponent's turn — check ALL replies
            if not moves:
                return (line, ws_move_count)  # opponent blocked: winning side has won
            best_mv, best_child, best_dtw = None, None, -1
            for mv in moves:
                child = current.apply_move(mv)
                if child.pieces_on_board[ws] < 3:
                    return None  # winning side unexpectedly eliminated
                r = db.query(child)
                if r is None:
                    return None  # sector unavailable — reject conservatively
                if r["outcome"] != "W":
                    return None  # opponent found an escape from the forced win
                dtw = r.get("dtw", -1)
                if dtw > best_dtw:
                    best_dtw = dtw
                    best_mv, best_child = mv, child
            if best_mv is None:
                return None
            line.append(_notation(best_mv))
            current = best_child

    return None  # exhausted max_depth without terminal — end of extract_solution_line


# ── Hardness scoring ──────────────────────────────────────────────────────────

def score_hardness(
    db: MalomDB,
    board: BoardState,
    winning_side: str,
) -> tuple[float, list[str], list[str], list[str]]:
    """Return (score, winning_moves, drawing_moves, losing_moves).

    Classifies first-level moves by Malom WDL outcome:
      winning  — opponent is losing (forced win maintained)
      drawing  — draw (winning side gave up the advantage)
      losing   — winning side is losing (blunder)
    """
    opp = "B" if winning_side == "W" else "W"
    winning, drawing, losing = [], [], []

    for mv in get_all_legal_moves(board):
        child = board.apply_move(mv)
        note = _notation(mv)

        if child.pieces_on_board[opp] < 3 or not get_all_legal_moves(child):
            winning.append(note)
            continue

        wdl = _malom_wdl(db, child)
        if wdl == "L":
            winning.append(note)
        elif wdl == "D":
            drawing.append(note)
        else:  # "W" or None
            losing.append(note)

    n_winning = len(winning)
    n_legal   = n_winning + len(drawing) + len(losing)
    score = 0.0

    if n_winning == 1:
        score += 4.0
    elif n_winning == 2:
        score += 2.5

    if len(losing) > 0:
        score += 1.0
    if n_winning < n_legal:
        score += 2.0  # not all moves win
    if n_legal >= 6:
        score += 1.0

    # Bonus for a quiet (non-capture) winning move
    if n_winning <= 2:
        quiet_wins = [
            mv for mv in get_all_legal_moves(board)
            if _notation(mv) in winning and not mv.get("capture")
        ]
        if quiet_wins:
            score += 1.0

    return score, winning, drawing, losing


# ── Reachability filter ───────────────────────────────────────────────────────

def _is_unreachable(board: BoardState) -> bool:
    """Cheap filter: 9v9 with mills on BOTH sides is impossible.

    The very first mill closure requires capturing the opponent, so no
    9v9 position can have simultaneous closed mills for both sides.
    """
    nW = board.pieces_on_board["W"]
    nB = board.pieces_on_board["B"]
    if nW == 9 and nB == 9 and _has_mill(board, "W") and _has_mill(board, "B"):
        return True
    return False


# ── Position sampling ─────────────────────────────────────────────────────────

def _random_movement_board(nW: int, nB: int, turn: str) -> BoardState:
    squares = random.sample(POSITIONS, nW + nB)
    positions = {p: "" for p in POSITIONS}
    for p in squares[:nW]:
        positions[p] = "W"
    for p in squares[nW:]:
        positions[p] = "B"
    return BoardState.from_setup(positions, turn, "move")


# ── Puzzle generation ─────────────────────────────────────────────────────────

# dtw pre-filter: keep positions where outcome=="W" and dtw is in a range
# that typically corresponds to win_in 4–7. Malom's midgame dtw semantics
# are approximate ("relative quality hint"), so we use a wide range [5, 17]
# and verify actual win depth from the greedy line.
_DTW_PREFILTER_MIN = 5
_DTW_PREFILTER_MAX = 17


def generate_malom_puzzle(
    db: MalomDB,
    winning_side: str = "random",
    target_win_in: int = 0,
    max_attempts: int = 3000,
    max_winning_moves: int = 2,
    min_pieces_per_side: int = 4,
    max_pieces_per_side: int = 7,
) -> Optional[MalomPuzzle]:
    """Sample random movement-phase positions and return the first valid puzzle.

    winning_side:          "W" | "B" | "random"
    target_win_in:         4–7, or 0 = any in range
    max_attempts:          positions sampled per call (Malom pre-filter is fast)
    max_winning_moves:     reject positions with more than N first-level wins
    max_pieces_per_side:   default 7; set higher for CLI (accepts slower hash init)
    """
    if not db.is_available():
        return None

    # One-time ~6 s pre-warm for (3..max_pieces)×(3..max_pieces) hash states.
    prewarm_hash_cache(max_pieces_per_side)

    for _ in range(max_attempts):
        ws = random.choice(["W", "B"]) if winning_side == "random" else winning_side
        nW = random.randint(min_pieces_per_side, max_pieces_per_side)
        nB = random.randint(min_pieces_per_side, max_pieces_per_side)

        board = _random_movement_board(nW, nB, ws)

        if _is_unreachable(board):
            continue

        result = db.query(board)
        if result is None:
            continue

        # Pre-filter: must be winning for current mover, dtw in rough range
        if result["outcome"] != "W":
            continue
        dtw = result.get("dtw", 0)
        if not (_DTW_PREFILTER_MIN <= dtw <= _DTW_PREFILTER_MAX):
            continue

        # Greedy solution extraction — determines actual win depth
        sol = extract_solution_line(db, board, ws, max_depth=9)
        if sol is None:
            continue
        line, actual_win_in = sol

        if not (4 <= actual_win_in <= 7):
            continue
        if target_win_in != 0 and actual_win_in != target_win_in:
            continue

        h_score, winning_mvs, drawing_mvs, losing_mvs = score_hardness(db, board, ws)

        if len(winning_mvs) > max_winning_moves:
            continue

        fen = board.to_fen_string()
        pid = "mlm_" + hashlib.md5(fen.encode()).hexdigest()[:8]
        all_legal = [_notation(m) for m in get_all_legal_moves(board)]

        tags = ["midgame", f"win-in-{actual_win_in}"]
        if len(winning_mvs) == 1:
            tags += ["unique-move", "bottleneck"]
        elif len(winning_mvs) == 2:
            tags.append("two-solutions")
        if len(all_legal) >= 6:
            tags.append("high-branching")
        if nW + nB >= 14:
            tags.append("early-game")
        elif nW + nB <= 8:
            tags.append("near-endgame")

        return MalomPuzzle(
            puzzle_id=pid,
            board_fen=fen,
            side_to_move=ws,
            winning_side=ws,
            target_win_in=actual_win_in,
            best_move=winning_mvs[0] if winning_mvs else (line[0] if line else ""),
            solution_line=line,
            legal_moves=all_legal,
            winning_moves=winning_mvs,
            losing_moves=losing_mvs,
            drawing_moves=drawing_mvs,
            hardness_score=h_score,
            tags=tags,
        )

    return None


# ── Placement puzzle generation ───────────────────────────────────────────────

def _random_placement_board_state(
    nW_on: int, nB_on: int, nW_unplaced: int, nB_unplaced: int, turn: str
) -> BoardState:
    """Create a random placement-phase board with no captures (nX_on == nX_placed)."""
    squares = random.sample(POSITIONS, nW_on + nB_on)
    positions = {p: "" for p in POSITIONS}
    for p in squares[:nW_on]:
        positions[p] = "W"
    for p in squares[nW_on:]:
        positions[p] = "B"
    board_str = "".join((positions[p] or ".") for p in POSITIONS)
    nW_placed = 9 - nW_unplaced
    nB_placed = 9 - nB_unplaced
    fen = f"{board_str}|{turn}|{nW_placed}|{nB_placed}"
    return BoardState.from_fen_string(fen)


_PLACEMENT_DTW_MIN = 3
_PLACEMENT_DTW_MAX = 25


def generate_malom_placement_puzzle(
    db: MalomDB,
    winning_side: str = "random",
    target_win_in: int = 0,
    max_attempts: int = 3000,
    max_winning_moves: int = 2,
) -> Optional[MalomPuzzle]:
    """Sample random placement-phase positions and return the first valid puzzle.

    winning_side:  "W" | "B" | "random"
    target_win_in: 4–7, or 0 = any in range

    Piece counts are kept in [3, 7] per side (nX_unplaced ∈ [2, 6]) so the
    positions stay within the pre-warmed Malom hash cache range.  The sampling
    assumes no captures yet (pieces_placed == pieces_on_board) so positions with
    any mill are discarded — a mill implies a prior capture, violating this invariant.
    """
    if not db.is_available():
        return None

    prewarm_hash_cache(7)

    for _ in range(max_attempts):
        ws = random.choice(["W", "B"]) if winning_side == "random" else winning_side

        # nW_unplaced in [2, 6] → nW_on in [3, 7] (within prewarm range)
        # Turn rule (no captures assumed, White places first):
        #   White's turn: nW_unplaced == nB_unplaced
        #   Black's turn: nB_unplaced == nW_unplaced + 1
        nW_unplaced = random.randint(2, 6)
        if ws == "W":
            nB_unplaced = nW_unplaced
        else:
            nB_unplaced = nW_unplaced + 1
            if nB_unplaced > 6:
                continue  # nB_on would be 2 < 3

        nW_on = 9 - nW_unplaced  # [3, 7]
        nB_on = 9 - nB_unplaced  # [3, 7]

        board = _random_placement_board_state(nW_on, nB_on, nW_unplaced, nB_unplaced, ws)

        # Reject positions with mills — any mill implies a capture happened,
        # contradicting our no-capture sampling (pieces_placed == pieces_on_board).
        if _has_mill(board, "W") or _has_mill(board, "B"):
            continue

        result = db.query(board)
        if result is None:
            continue

        if result["outcome"] != "W":
            continue
        dtw = result.get("dtw", 0)
        if not (_PLACEMENT_DTW_MIN <= dtw <= _PLACEMENT_DTW_MAX):
            continue

        sol = extract_solution_line(db, board, ws, max_depth=9)
        if sol is None:
            continue
        line, actual_win_in = sol

        if not (4 <= actual_win_in <= 7):
            continue
        if target_win_in != 0 and actual_win_in != target_win_in:
            continue

        h_score, winning_mvs, drawing_mvs, losing_mvs = score_hardness(db, board, ws)

        if len(winning_mvs) > max_winning_moves:
            continue

        fen = board.to_fen_string()
        pid = "plc_" + hashlib.md5(fen.encode()).hexdigest()[:8]
        all_legal = [_notation(m) for m in get_all_legal_moves(board)]

        tags = ["placement", f"win-in-{actual_win_in}"]
        if len(winning_mvs) == 1:
            tags += ["unique-move", "bottleneck"]
        elif len(winning_mvs) == 2:
            tags.append("two-solutions")
        if len(all_legal) >= 6:
            tags.append("high-branching")

        return MalomPuzzle(
            puzzle_id=pid,
            board_fen=fen,
            side_to_move=ws,
            winning_side=ws,
            target_win_in=actual_win_in,
            best_move=winning_mvs[0] if winning_mvs else (line[0] if line else ""),
            solution_line=line,
            legal_moves=all_legal,
            winning_moves=winning_mvs,
            losing_moves=losing_mvs,
            drawing_moves=drawing_mvs,
            hardness_score=h_score,
            source_db="malom_placement",
            tags=tags,
        )

    return None
