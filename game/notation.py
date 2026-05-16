"""
game/notation.py — Move encoding, decoding, and game export.

Notation format (matching the plan's export example):
  Placement:           d2
  Placement + capture: d2xf6
  Movement:            c5-c4
  Movement + capture:  g7-g4xd1
  Game end marker:     *
"""

from __future__ import annotations
from typing import List, Optional


# ── Single-move encoding ──────────────────────────────────────────────────────

def encode_move(move: dict, phase: str) -> str:
    """
    Encode a move dict to its notation string.
    'phase' is one of 'place', 'move', 'fly' (fly is treated identically to move).
    """
    if move.get("end"):
        return "*"

    result = ""
    if move["from"] is None:
        result = move["to"]
    else:
        result = f"{move['from']}-{move['to']}"

    if move.get("capture"):
        result += f"x{move['capture']}"

    return result


# ── Single-move decoding ──────────────────────────────────────────────────────

def parse_move_string(s: str) -> dict:
    """
    Parse a single notation token back to a move dict.
    Handles: placement, placement+capture, movement, movement+capture, and '*'.
    """
    s = s.strip()
    if s == "*":
        return {"from": None, "to": None, "capture": None, "end": True}

    capture: Optional[str] = None
    if "x" in s:
        s, capture = s.split("x", 1)

    if "-" in s:
        src, dest = s.split("-", 1)
        return {"from": src, "to": dest, "capture": capture, "end": False}
    else:
        return {"from": None, "to": s, "capture": capture, "end": False}


# ── Full game export ──────────────────────────────────────────────────────────

def export_pgn_style(game_record: dict) -> str:
    """
    Render a full game record in the two-column format:

        1. d2 d6
        2. f4 b4
        ...
        N. *

    game_record must be a dict with a 'moves' key whose value is a list of
    per-half-move dicts, each containing at least 'color' and 'notation'.
    """
    moves: List[dict] = game_record.get("moves", [])
    lines: List[str] = []
    i = 0
    turn_num = 1

    while i < len(moves):
        white_tok = ""
        black_tok = ""

        if i < len(moves) and moves[i]["color"] == "W":
            white_tok = moves[i]["notation"]
            i += 1

        if i < len(moves) and moves[i]["color"] == "B":
            black_tok = moves[i]["notation"]
            i += 1

        line = f"{turn_num}. {white_tok}"
        if black_tok:
            line += f" {black_tok}"
        lines.append(line.rstrip())
        turn_num += 1

    lines.append(f"{turn_num}. *")
    return "\n".join(lines)


def export_annotated(game_record: dict) -> str:
    """
    Like export_pgn_style but adds {?} after moves where a book-line deviation
    was recorded in the per-move opening_recognition dict.
    """
    moves: List[dict] = game_record.get("moves", [])
    lines: List[str] = []
    i = 0
    turn_num = 1

    def annotate(m: dict) -> str:
        tok = m["notation"]
        rec = m.get("opening_recognition") or {}
        if rec.get("deviation"):
            tok += "{?}"
        return tok

    while i < len(moves):
        white_tok = ""
        black_tok = ""

        if i < len(moves) and moves[i]["color"] == "W":
            white_tok = annotate(moves[i])
            i += 1

        if i < len(moves) and moves[i]["color"] == "B":
            black_tok = annotate(moves[i])
            i += 1

        line = f"{turn_num}. {white_tok}"
        if black_tok:
            line += f" {black_tok}"
        lines.append(line.rstrip())
        turn_num += 1

    lines.append(f"{turn_num}. *")
    return "\n".join(lines)
