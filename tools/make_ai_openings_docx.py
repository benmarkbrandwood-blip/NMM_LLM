#!/usr/bin/env python3
"""tools/make_ai_openings_docx.py

Generate two Word documents showing the Generalist v4 AI's opening preferences:
  docs/ai_opening_plays.docx       — 11 families with canonical 8-ply trunks
  docs/ai_opening_continuations.docx — 3×3 continuation tables per family

Data source : data/specialist_db_v4.sqlite  (AI self-play, winning lines)
Model       : learned_ai/checkpoints/scaffolded/s_gen_v4/BWDB/best.pt

Run with:
    .venv/bin/python tools/make_ai_openings_docx.py

Re-run when gen v4 finishes training at difficulty 20 to capture mature preferences.
"""
from __future__ import annotations

import collections
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Pt, RGBColor, Inches, Cm

from game.board import BoardState
from game.notation import encode_move, parse_move_string
from game.rules import get_all_legal_moves, get_game_phase
from learned_ai.models.scaffolded_encoder import encode_position_with_lookahead
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet

# ── paths ──────────────────────────────────────────────────────────────────────

DB_PATH   = "data/specialist_db_v4.sqlite"
CKPT_PATH = "learned_ai/checkpoints/scaffolded/s_gen_v4/BWDB/best.pt"
OUT_PLAYS  = "docs/ai_opening_plays.docx"
OUT_CONTS  = "docs/ai_opening_continuations.docx"

# ── family definitions ─────────────────────────────────────────────────────────
# Names hand-assigned based on the AI's preferred opening character.
FAMILY_NAMES = {
    ("d6", "f4"): "Corner Cross",
    ("d6", "d7"): "Diagonal Setup",
    ("f4", "b4"): "Counter Flank",
    ("f4", "d2"): "Center Column",
    ("d2", "b4"): "Column Hold",
    ("d6", "b4"): "Wing Anchor",
    ("d6", "d2"): "Double Column",
    ("f4", "b6"): "Column Strike",
    ("f4", "d5"): "Center Rush",
    ("b4", "d5"): "Reverse Center",
    ("d6", "a7"): "Corner Setup",
    ("b4", "d2"): "Column Mirror",
}

# ── helpers ────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("model_config", {})
    model = ScaffoldedPolicyNet.from_config(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    difficulty  = ckpt.get("difficulty", "?")
    game_count  = ckpt.get("game_count", 0)
    return model, device, difficulty, game_count


def replay_moves(move_strings: list[str]) -> BoardState:
    board = BoardState.new_game()
    for ms in move_strings:
        mv    = parse_move_string(ms)
        board = board.apply_move(mv)
    return board


def score_legal_moves(model, board: BoardState, device: str, top_n: int = 3) -> list[dict]:
    """Return top_n legal moves ranked by policy_logits, highest first."""
    enc = encode_position_with_lookahead(
        board, board.turn,
        sentinel_advisor=None, db=None, value_net=None, lookahead_advisor=None,
    )
    if enc is None or not enc.legal_moves:
        return []
    feat_t = torch.tensor(enc.feat_matrix, dtype=torch.float32)
    with torch.no_grad():
        logits = model.policy_logits(feat_t)
        probs  = F.softmax(logits, dim=-1)
    order = probs.argsort(descending=True)
    return [
        {**enc.legal_moves[i.item()], "_prob": float(probs[i].item())}
        for i in order[:top_n]
    ]


def mv_notation(mv: dict, color: str, ply_num: int, board: BoardState) -> str:
    """Format a move dict as '9.W:f4' or '10.B:b2-d2xf6'."""
    raw = encode_move(mv, board.phase)
    return f"{ply_num}.{color}:{raw}"


def format_trunk(moves: list[str]) -> str:
    """Format trunk as '1.W:d6  2.B:f4\n3.W:...' with 2 moves per line."""
    board = BoardState.new_game()
    parts = []
    for i, ms in enumerate(moves):
        mv    = parse_move_string(ms)
        color = board.turn
        num   = i + 1
        raw   = encode_move(mv, board.phase)
        parts.append(f"{num}.{color}:{raw}")
        board = board.apply_move(mv)
    # Pair up into "1.W:x  2.B:y"
    lines = []
    for j in range(0, len(parts), 2):
        chunk = parts[j:j+2]
        lines.append("  ".join(chunk))
    return "\n".join(lines)


def build_continuation_cells(model, trunk_moves: list[str], device: str,
                              trunk_ply: int) -> list[str]:
    """Return 9 cell strings for the 3×3 continuation table.

    Layout:
      Rows = W's 3 top moves at the trunk position
      Cols = B's 3 top responses to that W move
    Each cell shows: "W_ply.W:w  B_ply.B:b" + newline + "follow.W:f  ★ gen v4"
    """
    trunk_board = replay_moves(trunk_moves)
    side_a = trunk_board.turn   # side to move at end of trunk
    side_b = "B" if side_a == "W" else "W"
    ply_a  = trunk_ply + 1
    ply_b  = trunk_ply + 2
    ply_c  = trunk_ply + 3

    top_a = score_legal_moves(model, trunk_board, device, top_n=3)

    cells = []
    for mv_a in top_a:
        board_a = trunk_board.apply_move(mv_a)
        nota_a  = mv_notation(mv_a, side_a, ply_a, trunk_board)

        top_b = score_legal_moves(model, board_a, device, top_n=3)
        if not top_b:
            for _ in range(3):
                cells.append(f"{nota_a}\n(terminal)\n★ gen v4")
            continue

        for mv_b in top_b:
            board_b = board_a.apply_move(mv_b)
            nota_b  = mv_notation(mv_b, side_b, ply_b, board_a)

            # W's best follow-up
            top_c = score_legal_moves(model, board_b, device, top_n=1)
            if top_c:
                nota_c = mv_notation(top_c[0], side_a, ply_c, board_b)
                cells.append(f"{nota_a}  {nota_b}\n{nota_c}\n★ gen v4")
            else:
                cells.append(f"{nota_a}  {nota_b}\n★ gen v4")

    # Pad / trim to exactly 9
    while len(cells) < 9:
        cells.append("")
    return cells[:9]


# ── DB extraction ──────────────────────────────────────────────────────────────

def extract_families(db_path: str):
    """Return list of (code, name, games, trunk8, total_games) sorted by count."""
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()
    c.execute('SELECT move_seq FROM winning_lines WHERE result IN ("W", "draw")')
    rows = c.fetchall()
    conn.close()

    families: dict[tuple, list] = collections.defaultdict(list)
    for (seq,) in rows:
        moves = json.loads(seq)
        if len(moves) >= 2:
            key = tuple(moves[:2])
            families[key].append(moves)

    # Top 11
    ranked = sorted(families.items(), key=lambda x: -len(x[1]))[:11]
    result = []
    for rank, (key, games) in enumerate(ranked, 1):
        code = f"A{rank:02d}"
        name = FAMILY_NAMES.get(key, f"{key[0]}-{key[1]}")
        # Canonical 8-move trunk = most common 8-ply prefix
        prefix8: collections.Counter = collections.Counter()
        for g in games:
            prefix8[tuple(g[:min(8, len(g))])] += 1
        trunk8, _ = prefix8.most_common(1)[0]
        result.append((code, name, key, list(trunk8), len(games)))
    return result, len(rows)


# ── docx styling helpers ───────────────────────────────────────────────────────

def _set_cell_bg(cell, hex_colour: str) -> None:
    """Fill table cell background with a hex colour like 'DDEEFF'."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_colour)
    tcPr.append(shd)


def _heading(doc: Document, text: str, level: int) -> None:
    doc.add_heading(text, level=level)


def _normal(doc: Document, text: str, bold: bool = False) -> None:
    p = doc.add_paragraph(style="Normal")
    run = p.add_run(text)
    run.bold = bold


def _add_continuation_table(doc: Document, cells: list[str]) -> None:
    """Add a 3×3 table of continuation cells."""
    tbl = doc.add_table(rows=3, cols=3)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
    BG = "EEF4FF"
    for ri in range(3):
        for ci in range(3):
            idx  = ri * 3 + ci
            cell = tbl.cell(ri, ci)
            _set_cell_bg(cell, BG)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(cells[idx] if idx < len(cells) else "")
            run.font.size = Pt(9)
    doc.add_paragraph()


# ── document builders ──────────────────────────────────────────────────────────

def build_plays_doc(families, total_games: int, difficulty: int,
                    game_count: int, db_path: str, script_path: str) -> Document:
    doc = Document()
    doc.core_properties.title = "NMM AI Opening Plays (gen v4)"

    # Title + subtitle
    title_para = doc.add_heading("NMM AI Opening Plays — gen v4", level=0)

    doc.add_paragraph(
        f"Top 11 opening families from Generalist v4 self-play "
        f"(difficulty {difficulty}/20, {game_count:,} games, {total_games:,} winning/draw lines "
        f"from {db_path})."
    )
    doc.add_paragraph(
        f"Generated by: {script_path}\n"
        f"Re-run this script when gen v4 completes training at difficulty 20 "
        f"to capture its fully mature opening preferences."
    )
    doc.add_page_break()

    doc.add_heading("AI Opening Families (gen v4)", level=1)

    for code, name, key, trunk, game_count_fam in families:
        doc.add_heading(f"{code}  {name}", level=2)
        _normal(doc, f"8-ply  DB: {game_count_fam} games  (seed: {key[0]}-{key[1]})")
        doc.add_paragraph()  # blank line
        _normal(doc, f"Moves:  {format_trunk(trunk)}")
        doc.add_paragraph()

    return doc


def build_continuations_doc(families, model, device: str, difficulty: int,
                             script_path: str) -> Document:
    doc = Document()
    doc.core_properties.title = "NMM AI Opening Continuations (gen v4)"

    doc.add_heading("NMM AI Opening Continuations — gen v4", level=0)
    doc.add_paragraph(
        f"Top 9 continuations per family from ply 9 (3 W choices × 3 B responses). "
        f"★ gen v4 = Generalist v4 policy (difficulty {difficulty}/20).\n"
        f"Re-run {script_path} when gen v4 reaches difficulty 20."
    )

    for code, name, key, trunk, game_count_fam in families:
        doc.add_heading(f"{code}  {name}", level=2)
        _normal(doc, f"Opening (8-ply):  {format_trunk(trunk)}")
        _normal(doc, f"★ = Generalist v4 (difficulty {difficulty}/20, DB: {game_count_fam} games)")
        doc.add_paragraph()

        print(f"  Generating continuations for {code} {name} ...", flush=True)
        cells = build_continuation_cells(model, trunk, device, trunk_ply=8)
        _add_continuation_table(doc, cells)

    return doc


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    script_path = "tools/make_ai_openings_docx.py"
    os.makedirs("docs", exist_ok=True)

    print("Loading DB …")
    families, total_games = extract_families(DB_PATH)
    print(f"  {len(families)} families, {total_games} total games")

    print("Loading gen v4 model …")
    model, device, difficulty, game_count = load_model(CKPT_PATH)
    print(f"  difficulty={difficulty}, game_count={game_count:,}")

    print("Building ai_opening_plays.docx …")
    plays_doc = build_plays_doc(
        families, total_games, difficulty, game_count, DB_PATH, script_path
    )
    plays_doc.save(OUT_PLAYS)
    print(f"  Saved: {OUT_PLAYS}")

    print("Building ai_opening_continuations.docx …")
    conts_doc = build_continuations_doc(families, model, device, difficulty, script_path)
    conts_doc.save(OUT_CONTS)
    print(f"  Saved: {OUT_CONTS}")

    print("Done.")


if __name__ == "__main__":
    main()
