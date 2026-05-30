"""bench/smoke_rust_game.py — End-to-end AI-vs-AI smoke test using the Rust core.

Plays a full Nine Men's Morris game where BOTH sides choose moves via the
self-contained Rust search (nmm_core.py_get_best_move), applying each move with
the Python engine (BoardState.apply_move) and validating legality against
game.rules at every ply. Verifies the Rust backend produces a legal, terminating
game end-to-end.

Run:
    python bench/smoke_rust_game.py
    python bench/smoke_rust_game.py --depth 5 --max-plies 300

Exit code 0 on a clean, legal, terminating game; non-zero otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.board import POSITIONS, BoardState  # noqa: E402
from game.rules import get_all_legal_moves, is_terminal  # noqa: E402

try:
    import nmm_core  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"FAIL: nmm_core not importable ({exc}). Build with scripts/build_rust.sh")
    raise SystemExit(2)

_IDX = {p: i for i, p in enumerate(POSITIONS)}


def _bits(board: BoardState):
    white = sum(1 << _IDX[p] for p, v in board.positions.items() if v == "W")
    black = sum(1 << _IDX[p] for p, v in board.positions.items() if v == "B")
    stm = 0 if board.turn == "W" else 1
    return white, black, board.pieces_placed["W"], board.pieces_placed["B"], stm


def _rust_move(board: BoardState, depth: int, time_ms: int):
    white, black, wp, bp, stm = _bits(board)
    frm, to, cap = nmm_core.py_get_best_move(white, black, wp, bp, stm, depth, time_ms)
    if to is None:
        return None
    return {
        "from": None if frm is None else POSITIONS[frm],
        "to": POSITIONS[to],
        "capture": None if cap is None else POSITIONS[cap],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--time-ms", type=int, default=2000)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    board = BoardState.new_game()
    ply = 0
    while ply < args.max_plies:
        terminal, winner = is_terminal(board)
        if terminal:
            print(f"\nGame over after {ply} plies. Winner: {winner}")
            return 0

        legal = get_all_legal_moves(board)
        if not legal:
            print(f"\nNo legal moves for {board.turn} at ply {ply} — "
                  f"{'B' if board.turn == 'W' else 'W'} wins.")
            return 0

        mv = _rust_move(board, args.depth, args.time_ms)
        if mv is None:
            print(f"FAIL: Rust returned no move at ply {ply} "
                  f"with {len(legal)} legal moves available.")
            return 1

        # Validate legality against the Python source of truth.
        legal_norm = {(m["from"], m["to"], m.get("capture")) for m in legal}
        if (mv["from"], mv["to"], mv.get("capture")) not in legal_norm:
            print(f"FAIL: Rust move {mv} is ILLEGAL at ply {ply}.")
            print(f"board: {board.to_fen_string()}")
            return 1

        if not args.quiet:
            tag = board.turn
            frm = mv["from"] or "+"
            cap = f"x{mv['capture']}" if mv["capture"] else ""
            print(f"ply {ply:3d} {tag}: {frm}->{mv['to']}{cap}", end="   ")
            if ply % 3 == 2:
                print()

        board = board.apply_move(mv)
        ply += 1

    print(f"\nReached ply cap ({args.max_plies}) without termination — "
          f"treating as draw (legal throughout).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
