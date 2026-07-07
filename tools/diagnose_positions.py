#!/usr/bin/env python3
"""Diagnose heuristic score breakdowns for four disputed AI positions.

Usage: .venv/bin/python tools/diagnose_positions.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from ai.heuristics import tactical_move_bonus, DEFAULT_WEIGHTS, _placement_chain_scan, HeuristicWeights
from ai.game_ai import GameAI
from ai.value_net import ValueNet

_VN_PATH = _ROOT / "data" / "value_net.npz"
_value_net = ValueNet.load_if_exists(_VN_PATH)
print(f"ValueNet: {'loaded' if _value_net else 'NOT FOUND'}")


def _fmt_move(m: dict) -> str:
    frm = m.get("from")
    to = m.get("to", "?")
    cap = m.get("capture")
    s = f"{frm}->{to}" if frm else to
    if cap:
        s += f" x{cap}"
    return s


def ai_pick(board: BoardState, label: str = "", time_budget: float = 3.0, vn=None) -> None:
    """Run actual GameAI (difficulty 7) on board and print chosen move.

    vn=None → no value net (pure heuristic).  vn=_value_net → full blend.
    """
    hw = HeuristicWeights(value_net_blend=80 if vn is not None else 0)
    ai = GameAI(
        color=board.turn,
        difficulty=7,
        weights=hw,
        value_net=vn,
        override_time_budget=time_budget,
    )
    chosen = ai.choose_move(board)
    vn_label = "VN+heuristic" if vn is not None else "heuristic-only"
    print(f"\n  [AI pick — {label}  ({vn_label})]  chose: {_fmt_move(chosen)}")
    thinking = getattr(ai, "last_thinking", "")
    if thinking:
        for line in thinking.strip().splitlines()[:6]:
            print(f"    {line}")


def place(board: BoardState, pos: str, capture: str | None = None) -> BoardState:
    return board.apply_move({"from": None, "to": pos, "capture": capture})


def move(board: BoardState, frm: str, to: str, capture: str | None = None) -> BoardState:
    return board.apply_move({"from": frm, "to": to, "capture": capture})


def breakdown_diff(before: BoardState, sq: str, color: str, capture=None, label="") -> None:
    after = place(before, sq, capture)
    bd = tactical_move_bonus(before, after, color, return_breakdown=True)
    total = bd["total"]
    print(f"\n  → Place {color} at {sq}{f' x{capture}' if capture else ''} [{label}]: TOTAL={total:+d}")
    for name, val in bd["top_terms"]:
        print(f"      {name:<45} {val:+d}")


def chain_level(board: BoardState, color: str) -> int:
    return _placement_chain_scan(board, color)


# ─────────────────────────────────────────────────────────────────────────────
# Game 1: Black's 9th piece — a7 (suggested) vs e4 (AI chose)
# 1.d6 d2  2.f4 b4  3.f6 f2  4.b6xf2 f2  5.b2 d7  6.c3 g4  7.d1 d5  8.a4 c4
# Then White plays e5. Black to place 9th piece.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("GAME 1 — Black's 9th piece after White e5")
print("Position: White has d6,f4,f6,b6,b2,c3,d1,a4,e5 | Black has d2,b4,f2,d7,g4,d5,c4")

b = BoardState.new_game()
b = place(b, "d6")                   # W1
b = place(b, "d2")                   # B1
b = place(b, "f4")                   # W2
b = place(b, "b4")                   # B2
b = place(b, "f6")                   # W3
b = place(b, "f2")                   # B3
b = place(b, "b6", capture="f2")     # W4 closes b6-d6-f6, captures Black f2
b = place(b, "f2")                   # B4 re-places f2
b = place(b, "b2")                   # W5
b = place(b, "d7")                   # B5
b = place(b, "c3")                   # W6
b = place(b, "g4")                   # B6
b = place(b, "d1")                   # W7
b = place(b, "d5")                   # B7
b = place(b, "a4")                   # W8
b = place(b, "c4")                   # B8
b = place(b, "e5")                   # W9

print(f"\nIt is {b.turn}'s turn. Pieces placed: W={b.pieces_placed['W']} B={b.pieces_placed['B']}")
print(f"Black pieces on board: {[p for p in b.positions if b.positions[p]=='B']}")
print(f"White pieces on board: {[p for p in b.positions if b.positions[p]=='W']}")

# What mills does a7 give Black?
print("\n  2-configs if Black plays a7:")
after_a7 = place(b, "a7")
from ai.heuristics import _two_configs, _closed_mills, _cycling_mill_setup, _double_mills
print(f"    Black 2-configs: {_two_configs(after_a7,'B')}  mills: {_closed_mills(after_a7,'B')}  cycling: {_cycling_mill_setup(after_a7,'B')}  double: {_double_mills(after_a7,'B')}")

print("\n  Chain scan BEFORE Black's 9th move (placement_index=8):")
print(f"    a7: {chain_level(place(b,'a7'), 'B')}")
print(f"    e4: {chain_level(place(b,'e4'), 'B')}")

breakdown_diff(b, "a7", "B", label="SUGGESTED")
breakdown_diff(b, "e4", "B", label="AI CHOSE")
breakdown_diff(b, "d3", "B", label="HEURISTIC-ONLY PICK")
ai_pick(b, label="Game 1: Black 9th", vn=None)
ai_pick(b, label="Game 1: Black 9th", vn=_value_net)

# ─────────────────────────────────────────────────────────────────────────────
# Game 2: Check scatter — White's early placements are clustered
# 1.b2 f4  2.d2 f2  3.f6 c4  4.d1 d3
# At White's 5th placement (c3 vs something spread), show breakdown
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("GAME 2 — White's 5th piece clustering check")
print("Position after 1.b2 f4  2.d2 f2  3.f6 c4  4.d1 d3")

b2 = BoardState.new_game()
b2 = place(b2, "b2"); b2 = place(b2, "f4")
b2 = place(b2, "d2"); b2 = place(b2, "f2")
b2 = place(b2, "f6"); b2 = place(b2, "c4")
b2 = place(b2, "d1"); b2 = place(b2, "d3")

print(f"\nIt is {b2.turn}'s turn. Placement index={b2.pieces_placed['W']}")
print(f"White pieces: {[p for p in b2.positions if b2.positions[p]=='W']}")
print(f"Black pieces: {[p for p in b2.positions if b2.positions[p]=='B']}")

# c3 = clustered (adjacent to d3 which is White... wait, d3 is Black's here)
# Actually in game 2: 4.d1 d3 means White:d1 Black:d3
# White has b2,d2,f6,d1. 5th piece: user says White played c3.
breakdown_diff(b2, "c3", "W", label="White c3 (clustered — what AI does?)")
breakdown_diff(b2, "a7", "W", label="White a7 (spread alternative)")
breakdown_diff(b2, "g7", "W", label="White g7 (spread alternative)")

# Check scatter bonus specifically
print(f"\n  Scatter check (should fire for non-adjacent placements, index<6):")
for sq in ["c3", "a7", "g7", "g1", "a1"]:
    after = place(b2, sq)
    has_adj = any(b2.positions.get(nb) == "W" for nb in after.positions.keys()
                  if after.positions[nb] == "W" and sq != nb)
    # simpler: check adjacency
    from game.board import ADJACENCY
    adj_own = any(b2.positions[nb] == "W" for nb in ADJACENCY.get(sq, []))
    print(f"    {sq}: adjacent_to_own_piece={adj_own}")
breakdown_diff(b2, "e4", "W", label="AI CHOSE (e4)")
ai_pick(b2, label="Game 2: White 5th", vn=None)
ai_pick(b2, label="Game 2: White 5th", vn=_value_net)

# ─────────────────────────────────────────────────────────────────────────────
# Game 4: Keep-busy chain scan — Black's 7th piece
# 1.d6 f4  2.b4 b6  3.d3 g4  4.e4 c4  5.d7 d5  6.a7 g7  7.g1 ???
# Black's 7th: should be f2 (chain fork), AI goes elsewhere
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("GAME 4 — Black's 7th piece chain fork check")
print("After 1.d6 f4  2.b4 b6  3.d3 g4  4.e4 c4  5.d7 d5  6.a7 g7  7.g1")

b4 = BoardState.new_game()
b4 = place(b4, "d6"); b4 = place(b4, "f4")
b4 = place(b4, "b4"); b4 = place(b4, "b6")
b4 = place(b4, "d3"); b4 = place(b4, "g4")
b4 = place(b4, "e4"); b4 = place(b4, "c4")
b4 = place(b4, "d7"); b4 = place(b4, "d5")
b4 = place(b4, "a7"); b4 = place(b4, "g7")
b4 = place(b4, "g1")  # White's 7th

print(f"\nIt is {b4.turn}'s turn. Placement index={b4.pieces_placed['B']}")
print(f"White pieces: {[p for p in b4.positions if b4.positions[p]=='W']}")
print(f"Black pieces: {[p for p in b4.positions if b4.positions[p]=='B']}")

print("\n  Chain scan levels for Black's 7th placement candidates:")
for sq in ["f2", "f6", "c5", "c3", "a4", "e5", "e3"]:
    if b4.positions.get(sq, "") == "":
        after = place(b4, sq)
        lvl = chain_level(after, "B")
        tc = _two_configs(after, "B")
        print(f"    {sq}: chain_level={lvl}  2-configs={tc}")

from game.board import POSITIONS as ALL_POSITIONS

print("\n  Full tactical score ranking for ALL empty squares (Black's 7th):")
_all_scores = []
for sq in ALL_POSITIONS:
    if b4.positions.get(sq, "") == "":
        after_sq = place(b4, sq)
        sc = tactical_move_bonus(b4, after_sq, "B")
        _all_scores.append((sc, sq))
_all_scores.sort(reverse=True)
for sc, sq in _all_scores[:8]:
    print(f"    {sq}: {sc:+d}")

print("\n  Score breakdowns for Black's 7th (key candidates):")
for sq in ["c5", "d1", "f2"] + [s for _, s in _all_scores[:3]]:
    sq_list = ["c5", "d1", "f2"] + [s for _, s in _all_scores[:3]]
    if sq in sq_list and b4.positions.get(sq, "") == "":
        breakdown_diff(b4, sq, "B", label=sq)

ai_pick(b4, label="Game 4: Black 7th", vn=None)
ai_pick(b4, label="Game 4: Black 7th", vn=_value_net)

# ─────────────────────────────────────────────────────────────────────────────
# Game 3: Black's 9th piece — a1 (suggested) for herding
# 1.d6 d2  2.e4 f4  3.c5 c4  4.d5 d7  5.e5xd7 d7  6.e3xf4 f4  7.f6 b6  8.d3 c3  9.d1 ???
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("GAME 3 — Black's 9th piece (herding)")
print("After 1.d6 d2  2.e4 f4  3.c5 c4  4.d5 d7  5.e5xd7 d7  6.e3xf4 f4  7.f6 b6  8.d3 c3  9.d1")

b3 = BoardState.new_game()
b3 = place(b3, "d6"); b3 = place(b3, "d2")
b3 = place(b3, "e4"); b3 = place(b3, "f4")
b3 = place(b3, "c5"); b3 = place(b3, "c4")
b3 = place(b3, "d5"); b3 = place(b3, "d7")
b3 = place(b3, "e5", capture="d7"); b3 = place(b3, "d7")  # W closes e3-e4-e5? wait...
# Actually "5.e5xd7" means White plays e5 and closes a mill capturing d7
# e5 closes which mill? e5 is in: e5-e4-e3 (W has e4 and would need e3),
# c5-d5-e5 (W has c5,d5), or g7-f6-e5.
# W has c5,d5: placing e5 closes c5-d5-e5! captures d7 (Black)
# Then Black re-plays d7.
b3 = place(b3, "e3", capture="f4")  # 6.e3xf4: White places e3, closes e3-e4-e5? W has e4,e5 → closes e3-e4-e5, captures f4
b3 = place(b3, "f4")               # Black replaces f4
b3 = place(b3, "f6"); b3 = place(b3, "b6")
b3 = place(b3, "d3"); b3 = place(b3, "c3")
b3 = place(b3, "d1")               # White's 9th

print(f"\nIt is {b3.turn}'s turn. Placement index={b3.pieces_placed['B']}")
print(f"White pieces: {sorted([p for p in b3.positions if b3.positions[p]=='W'])}")
print(f"Black pieces: {sorted([p for p in b3.positions if b3.positions[p]=='B'])}")

print("\n  Score breakdowns for Black's 9th:")
for sq in ["a1", "g1", "a4", "b4", "b2", "a7"]:
    if b3.positions.get(sq, "") == "":
        breakdown_diff(b3, sq, "B", label=sq)

from ai.heuristics import evaluate as _evaluate
print("\n  Base evaluate() after each placement (from Black's POV):")
base = _evaluate(b3, "B")
print(f"    BEFORE placement: {base}")
for sq in ["a1", "g1", "a4", "b4", "b2", "a7"]:
    if b3.positions.get(sq, "") == "":
        after = place(b3, sq)
        ev = _evaluate(after, "B")
        print(f"    {sq}: evaluate={ev}  delta={ev - base:+d}")

ai_pick(b3, label="Game 3: Black 9th", vn=None)
ai_pick(b3, label="Game 3: Black 9th", vn=_value_net)

print("\nDone.")
