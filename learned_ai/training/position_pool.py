"""learned_ai/training/position_pool.py — phase-specific starting position pool.

Walks sequential move records from data/games, data/human_games, and
data/ai_games JSONL files to extract board states at the correct phase:

  Midgame: board state at movement turn `movement_turn` (default 10) from
           each game — i.e. 10 moves after the placement phase ends.
           Captures the natural mid-game structure without early opening noise.

  Endgame: any board state where total pieces < 12.

Public API
----------
  load_position_pool(root, phase, ...) -> list[BoardState]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("nmm.position_pool")


def _total_pieces(board) -> int:
    return board.pieces_on_board.get("W", 0) + board.pieces_on_board.get("B", 0)


def _iter_game_files(root: Path):
    for subdir in ("data/games", "data/human_games", "data/ai_games"):
        d = root / subdir
        if not d.exists():
            continue
        yield from d.glob("*.jsonl")


def _load_moves(fpath: Path) -> list[dict]:
    try:
        with open(fpath, encoding="utf-8") as f:
            game = json.load(f)
        return game.get("moves", [])
    except Exception:
        return []


def _load_midgame_pool(
    root: Path,
    movement_turn: int = 10,
    window: int = 2,
    max_positions: int = 50_000,
) -> list:
    """Extract board states at ~movement_turn moves into the movement phase.

    For each game, counts moves after placement ends and records boards
    in [movement_turn - window, movement_turn + window].  Stops scanning
    that game once past the window to avoid wasting time on later moves.
    """
    from game.board import BoardState

    seen: set[str] = set()
    pool: list = []
    lo = movement_turn - window
    hi = movement_turn + window

    for fpath in _iter_game_files(root):
        if len(pool) >= max_positions:
            break

        moves = _load_moves(fpath)
        movement_count = 0
        in_movement = False

        for mv in moves:
            if len(pool) >= max_positions:
                break
            fen = mv.get("board_fen_before")
            if not fen:
                continue
            try:
                board = BoardState.from_fen_string(fen)
            except Exception:
                continue

            if board.phase == "place":
                movement_count = 0
                in_movement = False
                continue

            if not in_movement:
                in_movement = True
            movement_count += 1

            if lo <= movement_count <= hi:
                if fen not in seen:
                    seen.add(fen)
                    pool.append(board)

            if movement_count > hi:
                break  # done with this game

    log.info(
        "position_pool: midgame turn=%d ±%d → %d unique positions",
        movement_turn, window, len(pool),
    )
    return pool


def _load_endgame_pool(
    root: Path,
    min_pieces: int = 4,
    max_pieces: int = 11,
    max_positions: int = 50_000,
) -> list:
    """Extract board states where total pieces is in [min_pieces, max_pieces]."""
    from game.board import BoardState

    seen: set[str] = set()
    pool: list = []

    for fpath in _iter_game_files(root):
        if len(pool) >= max_positions:
            break

        for mv in _load_moves(fpath):
            if len(pool) >= max_positions:
                break
            fen = mv.get("board_fen_before")
            if not fen or fen in seen:
                continue
            try:
                board = BoardState.from_fen_string(fen)
            except Exception:
                continue

            total = _total_pieces(board)
            if min_pieces <= total <= max_pieces:
                seen.add(fen)
                pool.append(board)

    log.info(
        "position_pool: endgame pieces=[%d,%d] → %d unique positions",
        min_pieces, max_pieces, len(pool),
    )
    return pool


def load_position_pool(
    root: Path,
    phase: str = "endgame",
    min_pieces: int = 4,
    max_pieces: int = 11,
    max_positions: int = 50_000,
    movement_turn: int = 10,
    window: int = 2,
) -> list:
    """Load a filtered pool of BoardState objects for phase-specific training.

    Args:
        root:          Project root directory.
        phase:         "midgame" or "endgame".
        min_pieces:    (endgame) Minimum total pieces on board.
        max_pieces:    (endgame) Maximum total pieces on board.
        max_positions: Cap on pool size.
        movement_turn: (midgame) Movement phase turn to sample around.
        window:        (midgame) ± turns around movement_turn to accept.

    Returns:
        List of BoardState objects.  Empty list if no data found.
    """
    if phase == "midgame":
        return _load_midgame_pool(root, movement_turn=movement_turn, window=window,
                                  max_positions=max_positions)
    else:
        return _load_endgame_pool(root, min_pieces=min_pieces, max_pieces=max_pieces,
                                  max_positions=max_positions)
