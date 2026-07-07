"""Benchmark: Python AI (GameAI) vs Rust search (py_search_stats) on mid-game positions.

Runs both engines on ~20 hand-crafted mid-game board positions with a 5-second budget,
then reports: move agreement rate, depth reached, nodes/sec ratio.
"""

import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game.board import BoardState, POSITIONS
import nmm_core as _rust
from ai.native_core import board_to_bits
from ai.game_ai import GameAI

# ── Test positions (hand-crafted mid-game boards) ────────────────────────────
# Each entry: (white_positions, black_positions, white_placed, black_placed, stm)
# Positions are indices into POSITIONS list.

def make_board(white_idx, black_idx, white_placed, black_placed, stm):
    pos_str = ["."] * 24
    for i in white_idx:
        pos_str[i] = "W"
    for i in black_idx:
        pos_str[i] = "B"
    fen = f"{''.join(pos_str)}|{stm}|{white_placed}|{black_placed}"
    return BoardState.from_fen_string(fen)


# POSITIONS order (from game/board.py):
# a7=0, d7=1, g7=2, b6=3, d6=4, f6=5, c5=6, d5=7, e5=8,
# a4=9, b4=10, c4=11, e4=12, f4=13, g4=14,
# c3=15, d3=16, e3=17, b2=18, d2=19, f2=20,
# a1=21, d1=22, g1=23

POSITIONS_MAP = {p: i for i, p in enumerate(POSITIONS)}

POSITIONS_LABELS = POSITIONS  # just to be clear

# 20 mid-game positions
TEST_POSITIONS = [
    # (name, white_idx, black_idx, wp, bp, stm)

    # --- Movement phase positions (wp=bp=9, mid-game) ---
    ("mid1_balanced",   [0,1,2,9,11,13], [5,6,8,14,15,17], 9, 9, "W"),
    ("mid2_white_mill", [0,1,2,9,10,11], [5,7,8,13,14,17], 9, 9, "W"),
    ("mid3_black_mill", [0,3,9,10,15,18], [5,6,8,12,13,14], 9, 9, "W"),
    ("mid4_crowded",    [0,1,3,9,10,18], [2,5,6,12,14,22], 9, 9, "B"),
    ("mid5_open",       [0,4,9,11,16,22], [2,6,14,17,19,23], 9, 9, "B"),

    # --- Late movement phase (7 vs 7 pieces) ---
    ("late1_even",      [0,1,9,11,15,19,22], [5,6,8,14,17,20,23], 9, 9, "W"),
    ("late2_white_adv", [0,1,2,9,11,15,22], [6,8,14,16,19,20,23], 9, 9, "W"),
    ("late3_black_adv", [0,3,9,11,15,18,22], [5,6,8,12,14,17,20], 9, 9, "B"),
    ("late4_tactical",  [0,1,4,9,11,19,22], [3,6,7,12,15,20,23], 9, 9, "W"),
    ("late5_near_end",  [0,1,9,11,22], [5,8,14,17,23], 9, 9, "W"),

    # --- 6 vs 6 ---
    ("six1",            [0,1,2,9,11,22], [5,6,8,14,17,23], 9, 9, "W"),
    ("six2",            [0,3,9,10,15,22], [6,8,12,14,17,23], 9, 9, "B"),
    ("six3_fork",       [0,1,9,10,15,22], [3,6,8,14,17,23], 9, 9, "W"),

    # --- Early movement (9 vs 9 with all pieces just placed) ---
    ("early_mv1",       [0,1,3,4,9,10,11,15,22], [2,6,7,8,12,14,17,19,23], 9, 9, "W"),
    ("early_mv2",       [0,2,4,9,10,11,15,19,22], [1,3,6,8,12,14,16,17,23], 9, 9, "B"),

    # --- Placement phase (partial pieces placed) ---
    ("place1",          [0,1,9], [2,6,14], 3, 3, "W"),
    ("place2",          [0,1,4,9,11], [2,6,8,14,17], 5, 5, "W"),
    ("place3_mill",     [0,1,2,9], [6,8,14,17], 4, 4, "W"),   # white about to form mill

    # --- Near-fly phase (one side with 4 pieces) ---
    ("near_fly1",       [0,1,9,22], [5,6,14,17], 9, 9, "W"),
    ("near_fly2",       [0,9,11,22], [6,8,14,23], 9, 9, "B"),
]


TIME_LIMIT_S = 3.0   # per engine per position
MAX_DEPTH = 12


def run_python_ai(board, time_limit=TIME_LIMIT_S):
    ai = GameAI(color=board.turn, difficulty=10, override_time_budget=time_limit)
    ai.blunder_probability = 0.0  # no randomness
    ai.max_search_depth = MAX_DEPTH
    t0 = time.time()
    move = ai.choose_move(board)
    elapsed = time.time() - t0
    return move, ai.last_depth_reached, elapsed


def run_rust_search(board, time_limit=TIME_LIMIT_S):
    white, black, wp, bp, stm = board_to_bits(board)
    t0 = time.time()
    frm, to, cap, nodes, depth = _rust.py_search_stats(
        white, black, wp, bp, stm, MAX_DEPTH, int(time_limit * 1000)
    )
    elapsed = time.time() - t0
    move = None
    if to is not None:
        move = {
            "from": None if frm is None else POSITIONS[frm],
            "to": POSITIONS[to],
            "capture": None if cap is None else POSITIONS[cap],
        }
    return move, depth, nodes, elapsed


def moves_agree(py_move, rs_move):
    if py_move is None or rs_move is None:
        return py_move is None and rs_move is None
    return py_move.get("to") == rs_move.get("to") and py_move.get("from") == rs_move.get("from")


def main():
    print(f"{'Name':<22} {'Phase':<8} {'PyDepth':>7} {'RsDepth':>7} {'Agree':>6} "
          f"{'PyMove':<14} {'RsMove':<14} {'RsNodes':>10} {'RsNps':>10} {'PyT':>6} {'RsT':>6}")
    print("-" * 115)

    agree_count = 0
    total = len(TEST_POSITIONS)
    py_depths = []
    rs_depths = []

    for name, white_idx, black_idx, wp, bp, stm in TEST_POSITIONS:
        board = make_board(white_idx, black_idx, wp, bp, stm)

        if wp < 9:
            phase = "place"
        elif len(white_idx) <= 3 or len(black_idx) <= 3:
            phase = "fly"
        else:
            phase = "move"

        # Python AI
        try:
            py_move, py_depth, py_t = run_python_ai(board)
        except Exception as e:
            py_move, py_depth, py_t = None, 0, 0.0
            print(f"  [Python error on {name}: {e}]")

        # Rust search
        try:
            rs_move, rs_depth, rs_nodes, rs_t = run_rust_search(board)
        except Exception as e:
            rs_move, rs_depth, rs_nodes, rs_t = None, 0, 0, 0.0
            print(f"  [Rust error on {name}: {e}]")

        agree = moves_agree(py_move, rs_move)
        if agree:
            agree_count += 1
        py_depths.append(py_depth)
        rs_depths.append(rs_depth)

        def fmt_move(m):
            if m is None:
                return "None"
            s = m.get("to", "?")
            if m.get("from"):
                s = f"{m['from']}->{s}"
            if m.get("capture"):
                s += f"x{m['capture']}"
            return s

        rs_nps = int(rs_nodes / rs_t) if rs_t > 0 else 0
        agree_str = "YES" if agree else "NO "
        print(
            f"{name:<22} {phase:<8} {py_depth:>7} {rs_depth:>7} {agree_str:>6} "
            f"{fmt_move(py_move):<14} {fmt_move(rs_move):<14} {rs_nodes:>10,} "
            f"{rs_nps:>10,} {py_t:>6.1f} {rs_t:>6.1f}"
        )

    print("-" * 115)
    print(f"\nAgreement: {agree_count}/{total} ({100*agree_count/total:.0f}%)")
    print(f"Python avg depth: {sum(py_depths)/total:.1f}   Rust avg depth: {sum(rs_depths)/total:.1f}")
    depth_gains = [r - p for r, p in zip(rs_depths, py_depths)]
    print(f"Rust depth gain per position: avg {sum(depth_gains)/total:+.1f}, "
          f"min {min(depth_gains):+d}, max {max(depth_gains):+d}")


if __name__ == "__main__":
    main()
