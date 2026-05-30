"""ai/native_core.py — Optional Rust acceleration adapter.

This module is the single integration seam between the Python engine and the
compiled `nmm_core` Rust extension. It is intentionally dependency-light (only
stdlib + game.board) so it imports cleanly even when heavier optional deps
(e.g. chromadb) are unavailable.

If `nmm_core` is not importable, `RUST_AVAILABLE` is False and every wrapper
falls back to the pure-Python engine, so the game runs identically with or
without the extension. Build the extension with `scripts/build_rust.sh`.

The board index order matches `game.board.POSITIONS` exactly; bitboards use
bit `i` for `POSITIONS[i]`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from game.board import POSITIONS, BoardState

# ── Extension probe ───────────────────────────────────────────────────────────

try:  # pragma: no cover - exercised by integration, not unit tests
    import nmm_core as _rust  # type: ignore

    RUST_AVAILABLE = True
except Exception:  # ImportError or ABI mismatch — fall back silently.
    _rust = None
    RUST_AVAILABLE = False


_IDX = {p: i for i, p in enumerate(POSITIONS)}


def is_available() -> bool:
    """True when the compiled Rust core is loaded and usable."""
    return RUST_AVAILABLE


# ── Conversions: BoardState <-> bitboard tuple ────────────────────────────────

def board_to_bits(board: BoardState) -> Tuple[int, int, int, int, int]:
    """(white, black, white_placed, black_placed, stm) where stm is 0=W,1=B."""
    white = 0
    black = 0
    for pos, v in board.positions.items():
        if v == "W":
            white |= 1 << _IDX[pos]
        elif v == "B":
            black |= 1 << _IDX[pos]
    stm = 0 if board.turn == "W" else 1
    return white, black, board.pieces_placed["W"], board.pieces_placed["B"], stm


def _idx_to_move_dict(
    frm: Optional[int], to: Optional[int], cap: Optional[int]
) -> Optional[dict]:
    if to is None:
        return None
    return {
        "from": None if frm is None else POSITIONS[frm],
        "to": POSITIONS[to],
        "capture": None if cap is None else POSITIONS[cap],
    }


# ── Engine wrappers (each falls back to Python when Rust absent) ──────────────

def legal_moves(board: BoardState) -> Optional[List[dict]]:
    """Rust-generated legal moves as Python move dicts, or None when unavailable.

    Returns None (not an empty list) when Rust is absent so callers can branch
    to the Python generator; an actual empty list means 'no legal moves'."""
    if not RUST_AVAILABLE:
        return None
    white, black, wp, bp, stm = board_to_bits(board)
    out: List[dict] = []
    for frm, to, cap in _rust.py_legal_moves(white, black, wp, bp, stm):
        md = _idx_to_move_dict(frm, to, cap)
        if md is not None:
            out.append(md)
    return out


def get_best_move(
    board: BoardState, max_depth: int = 6, time_limit_ms: int = 5000
) -> Optional[dict]:
    """Best move from the self-contained Rust search, or None when unavailable.

    NOTE: the Rust search uses an integer base evaluation and is NOT guaranteed
    to choose the same move as the Python AI — it returns legal, sane play only.
    The Python AI remains the default decision engine; this is opt-in."""
    if not RUST_AVAILABLE:
        return None
    white, black, wp, bp, stm = board_to_bits(board)
    frm, to, cap = _rust.py_get_best_move(
        white, black, wp, bp, stm, max_depth, time_limit_ms
    )
    return _idx_to_move_dict(frm, to, cap)


def evaluate_base(board: BoardState) -> Optional[int]:
    """Rust integer base evaluation from side-to-move's perspective, or None."""
    if not RUST_AVAILABLE:
        return None
    white, black, wp, bp, stm = board_to_bits(board)
    return _rust.py_evaluate(white, black, wp, bp, stm)


def fullgame_db_key(board: BoardState) -> Optional[bytes]:
    """Byte-identical FullGame DB key (canonical), or None when unavailable."""
    if not RUST_AVAILABLE:
        return None
    white, black, _wp, _bp, stm = board_to_bits(board)
    return bytes(
        _rust.py_db_key(
            white, black, stm, board.pieces_placed["W"], board.pieces_placed["B"]
        )
    )


def endgame_db_key(board: BoardState) -> Optional[str]:
    """Endgame DB string key '<canonical board24>|<turn>', or None."""
    if not RUST_AVAILABLE:
        return None
    white, black, _wp, _bp, stm = board_to_bits(board)
    return _rust.py_endgame_key(white, black, stm)


def canonical_board_str(board24: str) -> Optional[Tuple[str, int]]:
    """(canonical_str, sym_idx) via Rust, or None when unavailable."""
    if not RUST_AVAILABLE:
        return None
    return _rust.py_canonical_board_str(board24)


def opening_key(notations: List[str], depth: Optional[int] = None):
    """(canonical pipe-joined sequence, sym_idx) via Rust, or None."""
    if not RUST_AVAILABLE:
        return None
    return _rust.py_opening_key(list(notations), depth)
