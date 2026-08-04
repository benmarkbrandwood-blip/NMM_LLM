"""scripts/generate_malom_regret_golden.py — Generate the malom_regret_golden.json fixture.

Produces candidate entries covering every RegretResult category required by
§6.1 of gap_net_v3_plan.md.  Output is written to
tests/fixtures/malom_regret_golden.json.

Run this once, inspect the output, then commit the fixture.  The generation
script is never committed as a test helper; it is a one-shot tool.

Usage::

    .venv/bin/python scripts/generate_malom_regret_golden.py [--db-dir PATH]

If --db-dir is omitted the path is read from data/settings.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.malom_db import MalomDB, _REGRET_VERSION, _MALOM_LABEL_VERSION
from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal, terminal_wdl


def _load_db_dir() -> str:
    cfg = _ROOT / "data" / "settings.json"
    if cfg.exists():
        d = json.loads(cfg.read_text(encoding="utf-8"))
        p = d.get("malom_db_path")
        if p:
            return p
    raise RuntimeError("Cannot find malom_db_path in data/settings.json")


def _board_to_dict(board: BoardState) -> dict:
    return {
        "positions": {k: v for k, v in board.positions.items() if v},
        "turn": board.turn,
        "phase": getattr(board, "phase", None),
        "pieces_placed": dict(board.pieces_placed),
        "pieces_on_board": dict(board.pieces_on_board),
    }


def _board_from_dict(d: dict) -> BoardState:
    return BoardState.from_setup(
        positions=d["positions"],
        turn=d["turn"],
        phase=d.get("phase", "move"),
    )


def _find_move_of_category(
    db: MalomDB,
    board: BoardState,
    target_category: str,
) -> dict | None:
    """Return the first legal move from board whose regret has wdl_transition == target."""
    parent_val = db.query_value(board)
    if parent_val is None:
        return None
    for move in get_all_legal_moves(board):
        result = db.query_regret(board, move)
        if result.available and result.wdl_transition == target_category:
            return move
    return None


def _make_entry(
    db: MalomDB,
    board: BoardState,
    move: dict,
    label: str,
) -> dict | None:
    result = db.query_regret(board, move)
    if not result.available:
        print(f"  [{label}] SKIPPED: {result.unavailable_reason}")
        return None

    entry: dict = {
        "label": label,
        "board": _board_to_dict(board),
        "move": move,
        "expected": {
            "available": True,
            "wdl_transition": result.wdl_transition,
            "omv_outcome": result.omv.outcome if result.omv else None,
            "best_omv_outcome": result.best_omv.outcome if result.best_omv else None,
            "components": result.components,
            "regret_version": _REGRET_VERSION,
            "malom_label_version": _MALOM_LABEL_VERSION,
        },
    }
    print(f"  [{label}] transition={result.wdl_transition!r}  "
          f"omv={result.omv.outcome if result.omv else None}  "
          f"comp_a={result.components['class_downgrade_prob']}  "
          f"comp_b={result.components['wdl_utility_loss']}  "
          f"comp_c={result.components['ordinal_rank_loss']}")
    return entry


def _make_unavailable_entry(
    board: BoardState,
    move: dict | None,
    label: str,
    expected_reason: str,
    db: MalomDB,
) -> dict:
    if move is None:
        # Use a dummy sentinel move that is certainly not legal.
        move = {"from": None, "to": "a7", "capture": None}
    result = db.query_regret(board, move)
    print(f"  [{label}] available={result.available} reason={result.unavailable_reason!r}")
    return {
        "label": label,
        "board": _board_to_dict(board),
        "move": move,
        "expected": {
            "available": False,
            "unavailable_reason": result.unavailable_reason or expected_reason,
            "regret_version": _REGRET_VERSION,
            "malom_label_version": _MALOM_LABEL_VERSION,
        },
    }


# ── Known positions ────────────────────────────────────────────────────────────

# 3v3 fly-phase position from test_human_moves_audit_perspective.py.
_POS_3V3 = {
    "a7": "W", "d5": "W", "e4": "W",
    "g7": "B", "f4": "B", "c3": "B",
}

# 4v4 movement-phase positions (non-fly) for searching W→D and D categories.
# White has a mill on a4-a7-a1 (left outer column) plus one extra piece.
_POS_4V4_MILL = {
    "a4": "W", "a7": "W", "a1": "W", "d7": "W",
    "g7": "B", "g4": "B", "g1": "B", "d1": "B",
}

# An endgame-ish position for searching draw and loss categories.
_POS_DRAW_SEARCH = {
    "a7": "W", "d7": "W", "g7": "W", "g4": "W",
    "a1": "B", "d1": "B", "g1": "B", "a4": "B",
}


def _scan_for_category(
    db: MalomDB,
    seed_boards: list[BoardState],
    target_cat: str,
    max_depth: int = 2,
    max_nodes: int = 500,
) -> tuple[BoardState, dict] | None:
    """BFS over child boards to find any (board, move) yielding target_cat."""
    visited: set = set()
    frontier: list[BoardState] = list(seed_boards)
    nodes_checked = 0
    for _ in range(max_depth + 1):
        next_frontier: list[BoardState] = []
        for board in frontier:
            if nodes_checked >= max_nodes:
                return None
            key = (board.turn, tuple(sorted(board.positions.items())))
            if key in visited:
                continue
            visited.add(key)
            nodes_checked += 1
            pv = db.query_value(board)
            if pv is None:
                continue
            for move in get_all_legal_moves(board):
                result = db.query_regret(board, move)
                if result.available and result.wdl_transition == target_cat:
                    return board, move
                if not is_terminal(board.apply_move(move))[0]:
                    next_frontier.append(board.apply_move(move))
        frontier = next_frontier
    return None


def _build_corpus(db: MalomDB) -> list[dict]:
    corpus = []

    # ── Position 1: 3v3 movement phase ────────────────────────────────────────
    board_3v3_w = BoardState.from_setup(_POS_3V3, turn="W", phase="move")
    board_3v3_b = BoardState.from_setup(_POS_3V3, turn="B", phase="move")

    parent_w = db.query_value(board_3v3_w)
    parent_b = db.query_value(board_3v3_b)
    print(f"3v3 W-to-move: parent outcome = {parent_w.outcome if parent_w else None}")
    print(f"3v3 B-to-move: parent outcome = {parent_b.outcome if parent_b else None}")

    # Collect one example per category across all candidate boards.
    covered_categories: set[str] = set()
    all_categories = [
        "win_preserved", "win_to_draw", "win_to_loss",
        "draw_preserved", "draw_to_loss", "all_losing",
    ]

    board_4v4_w = BoardState.from_setup(_POS_4V4_MILL, turn="W", phase="move")
    board_4v4_b = BoardState.from_setup(_POS_4V4_MILL, turn="B", phase="move")
    board_draw_w = BoardState.from_setup(_POS_DRAW_SEARCH, turn="W", phase="move")
    board_draw_b = BoardState.from_setup(_POS_DRAW_SEARCH, turn="B", phase="move")

    candidate_boards = [
        (board_3v3_w, "3v3-W"),
        (board_3v3_b, "3v3-B"),
        (board_4v4_w, "4v4-W"),
        (board_4v4_b, "4v4-B"),
        (board_draw_w, "draw-W"),
        (board_draw_b, "draw-B"),
    ]
    for board, name in candidate_boards:
        pv = db.query_value(board)
        print(f"{name}: parent outcome = {pv.outcome if pv else None}")
        for cat in all_categories:
            if cat in covered_categories:
                continue
            move = _find_move_of_category(db, board, cat)
            if move is not None:
                entry = _make_entry(db, board, move, f"{name}:{cat}")
                if entry:
                    corpus.append(entry)
                    covered_categories.add(cat)
    print(f"  Categories covered so far: {covered_categories}")

    # ── Deep scan for remaining categories ────────────────────────────────────
    seed_boards = [b for b, _ in candidate_boards]
    for cat in all_categories:
        if cat in covered_categories:
            continue
        print(f"  Deep scanning for {cat!r} (depth=2)...")
        hit = _scan_for_category(db, seed_boards, cat, max_depth=2, max_nodes=2000)
        if hit is not None:
            board, move = hit
            entry = _make_entry(db, board, move, f"scan:{cat}")
            if entry:
                corpus.append(entry)
                covered_categories.add(cat)
        else:
            print(f"    not found within search budget")
    print(f"  Categories covered after deep scan: {covered_categories}")

    # ── Position 2: new-game (place phase, draw root) ─────────────────────────
    new_game = BoardState.new_game()
    parent_ng = db.query_value(new_game)
    print(f"\nnew-game: parent outcome = {parent_ng.outcome if parent_ng else None}")
    if parent_ng is not None:
        # Any legal place move from new-game
        for move in get_all_legal_moves(new_game):
            entry = _make_entry(db, new_game, move, "new-game:any")
            if entry:
                corpus.append(entry)
                break

    # ── available=False: move_not_legal ───────────────────────────────────────
    print("\n--- unavailable cases ---")
    # A placement move (from=None) is never legal in move/fly phase.
    bad_move = {"from": None, "to": "a1", "capture": None}
    entry = _make_unavailable_entry(board_3v3_w, bad_move, "unavail:move_not_legal",
                                    "move_not_legal", db)
    corpus.append(entry)

    # ── available=False: parent_terminal ──────────────────────────────────────
    # Construct a position where White has only 2 pieces → White already lost.
    # from_setup with phase="move" sets pieces_placed=9 for both; pieces_on_board
    # is derived from what's on the board, so 2 White pieces triggers terminal.
    term_board = BoardState.from_setup(
        {"d7": "W", "g7": "W",   # 2 white pieces → terminal (< 3)
         "a1": "B", "a4": "B", "b4": "B", "c3": "B"},
        turn="B",
        phase="move",
    )
    is_term, winner = is_terminal(term_board)
    print(f"terminal board: is_terminal={is_term}, winner={winner}")
    if is_term:
        legal = get_all_legal_moves(term_board)
        dummy_move = legal[0] if legal else {"from": None, "to": "a7", "capture": None}
        entry = _make_unavailable_entry(term_board, dummy_move,
                                        "unavail:parent_terminal",
                                        "parent_terminal", db)
        corpus.append(entry)

    return corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", default=None)
    args = parser.parse_args()

    db_dir = args.db_dir or _load_db_dir()
    print(f"Loading Malom DB from: {db_dir}")
    db = MalomDB(db_dir)
    if not db.is_available():
        print("ERROR: Malom DB not available", file=sys.stderr)
        sys.exit(1)

    print("\nBuilding corpus...")
    corpus = _build_corpus(db)
    db.close()

    out_path = _ROOT / "tests" / "fixtures" / "malom_regret_golden.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "regret_version": _REGRET_VERSION,
            "malom_label_version": _MALOM_LABEL_VERSION,
            "generator": "scripts/generate_malom_regret_golden.py",
            "note": "Human-inspected golden corpus for test_malom_regret_v1.py",
        },
        "cases": corpus,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {len(corpus)} cases to {out_path}")


if __name__ == "__main__":
    main()
