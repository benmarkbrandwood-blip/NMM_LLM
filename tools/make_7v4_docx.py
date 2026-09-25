#!/usr/bin/env python3
"""tools/make_7v4_docx.py

Generate '7 vs 4 arrangements.docx'.

Systematically enumerates all canonical 7-piece White formations that
guarantee a forced win when Black drops from 4 to 3 pieces (flying phase
transition).  Uses the Malom ultra-strong DB for verification.

Approach
--------
1. Enumerate every canonical 7W formation (under D4 symmetry) that contains
   at least one complete mill.
2. For each, check Malom against ALL C(17,3) = 680 possible Black=3-piece
   flying arrangements (B to move).  A formation is "guaranteed" if every one
   of the 680 positions is a Malom White win.
3. Group by FAMILY — defined by the canonical mill each formation uses as its
   "anchor" mill — with sub-grouping by number of complete mills present.
4. Render board images and assemble the docx.

Usage
-----
  .venv/bin/python tools/make_7v4_docx.py
  .venv/bin/python tools/make_7v4_docx.py --quick   # sample 50 B positions (fast approx)
  .venv/bin/python tools/make_7v4_docx.py --out "my_output.docx"
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import POSITIONS, MILLS, ADJACENCY, BoardState
from ai.board_symmetry import _BOARD_PERM, _POSITIONS as _BS_POSITIONS, _POS_IDX

# ── Load Malom DB ─────────────────────────────────────────────────────────────

def _load_malom():
    settings_path = _ROOT / "data" / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    malom_path = settings.get("malom_db_path", "")
    if not malom_path:
        print("ERROR: malom_db_path not set in data/settings.json", file=sys.stderr)
        sys.exit(1)
    from ai.malom_db import MalomDB
    db = MalomDB(malom_path)
    if not db.is_available():
        print(f"ERROR: Malom DB not available at {malom_path}", file=sys.stderr)
        sys.exit(1)
    return db

# ── D4 canonicalisation ───────────────────────────────────────────────────────

_POS_SET = list(POSITIONS)  # same order as game/board.py
_N = 24

def _apply_sym(pos_indices: frozenset[int], sym_idx: int) -> frozenset[int]:
    perm = _BOARD_PERM[sym_idx]
    return frozenset(perm[i] for i in pos_indices)

def _canonical_frozenset(pos_indices: frozenset[int]) -> frozenset[int]:
    """Return the lexicographically smallest D4 transform (as frozenset of indices)."""
    best = pos_indices
    for s in range(1, 8):
        t = _apply_sym(pos_indices, s)
        if t < best:
            best = t
    return best

def _pos_names_to_idx(positions: list[str]) -> frozenset[int]:
    return frozenset(_POS_IDX[p] for p in positions)

def _idx_to_pos_names(indices: frozenset[int]) -> list[str]:
    return sorted(_BS_POSITIONS[i] for i in indices)

# ── Mill helpers ──────────────────────────────────────────────────────────────

# Pre-index MILLS by frozenset of index triples for fast lookup
_MILL_IDX_SETS: list[frozenset[int]] = [
    frozenset(_POS_IDX[p] for p in mill) for mill in MILLS
]
_MILL_NAMES: list[tuple[str,str,str]] = list(MILLS)

def _count_mills(w_idx: frozenset[int]) -> int:
    return sum(1 for m in _MILL_IDX_SETS if m.issubset(w_idx))

def _find_mills(w_idx: frozenset[int]) -> list[tuple[str,str,str]]:
    return [_MILL_NAMES[i] for i, m in enumerate(_MILL_IDX_SETS) if m.issubset(w_idx)]

# ── Malom board query helper ──────────────────────────────────────────────────

def _make_board(w_positions: list[str], b_positions: list[str], turn: str = "B") -> BoardState:
    """Build a movement-phase BoardState (all pieces placed, B=3 flies)."""
    pos = {p: "" for p in POSITIONS}
    for p in w_positions:
        pos[p] = "W"
    for p in b_positions:
        pos[p] = "B"
    return BoardState.from_setup(pos, turn=turn, phase="move")

def _malom_b_wins(db, w_positions: list[str], b_positions: list[str]) -> bool:
    """True if this W=7,B=3(flying) position is a forced W win (B to move, B loses)."""
    board = _make_board(w_positions, b_positions, turn="B")
    result = db.query(board)
    if result is None:
        return False
    # outcome is from mover's (B's) perspective: "L" = B loses = W wins
    return result["outcome"] == "L"

# ── Formation enumeration ─────────────────────────────────────────────────────

def _enumerate_canonical_7w_with_mill() -> list[frozenset[int]]:
    """Return all D4-canonical frozensets of 7 W positions that include ≥1 mill."""
    seen: set[frozenset[int]] = set()
    result: list[frozenset[int]] = []

    all_pos_idx = list(range(_N))

    for mill_idx_set in _MILL_IDX_SETS:
        remaining = [i for i in all_pos_idx if i not in mill_idx_set]
        for extra4 in combinations(remaining, 4):
            w7 = mill_idx_set | frozenset(extra4)
            canon = _canonical_frozenset(w7)
            if canon not in seen:
                seen.add(canon)
                result.append(canon)

    return result

# ── Malom verification ────────────────────────────────────────────────────────

def _verify_formation(
    db,
    w_idx: frozenset[int],
    n_sample: Optional[int] = None,  # None = check all 680
) -> tuple[int, int]:
    """Return (n_wins, n_checked) for W=7 formation vs B=3 flying arrangements.

    If n_sample is set, randomly sample that many B arrangements instead of all.
    """
    w_pos = _idx_to_pos_names(w_idx)
    w_set = set(w_pos)
    remaining = [p for p in POSITIONS if p not in w_set]  # 17 positions

    if n_sample is not None and n_sample < len(list(combinations(remaining, 3))):
        b3_choices = random.sample(list(combinations(remaining, 3)), n_sample)
    else:
        b3_choices = list(combinations(remaining, 3))

    n_wins = 0
    for b3 in b3_choices:
        if _malom_b_wins(db, w_pos, list(b3)):
            n_wins += 1

    return n_wins, len(b3_choices)

# ── Mill family labelling ─────────────────────────────────────────────────────

# The 16 mills collapse to just 4 D4 orbit classes.
# We use a single representative from each orbit as the family label.
# Under D4, all 4 outer-ring side mills map to each other, etc.
# We assign family by the CANONICAL MILL present (or primary mill if multiple).

# Canonical representative of each mill orbit (one mill per D4 orbit):
_MILL_FAMILY = {
    # Outer ring sides → "outer"
    frozenset([_POS_IDX["a7"], _POS_IDX["d7"], _POS_IDX["g7"]]): "outer",
    frozenset([_POS_IDX["g7"], _POS_IDX["g4"], _POS_IDX["g1"]]): "outer",
    frozenset([_POS_IDX["g1"], _POS_IDX["d1"], _POS_IDX["a1"]]): "outer",
    frozenset([_POS_IDX["a1"], _POS_IDX["a4"], _POS_IDX["a7"]]): "outer",
    # Middle ring sides → "middle"
    frozenset([_POS_IDX["b6"], _POS_IDX["d6"], _POS_IDX["f6"]]): "middle",
    frozenset([_POS_IDX["f6"], _POS_IDX["f4"], _POS_IDX["f2"]]): "middle",
    frozenset([_POS_IDX["f2"], _POS_IDX["d2"], _POS_IDX["b2"]]): "middle",
    frozenset([_POS_IDX["b2"], _POS_IDX["b4"], _POS_IDX["b6"]]): "middle",
    # Inner ring sides → "inner"
    frozenset([_POS_IDX["c5"], _POS_IDX["d5"], _POS_IDX["e5"]]): "inner",
    frozenset([_POS_IDX["e5"], _POS_IDX["e4"], _POS_IDX["e3"]]): "inner",
    frozenset([_POS_IDX["e3"], _POS_IDX["d3"], _POS_IDX["c3"]]): "inner",
    frozenset([_POS_IDX["c3"], _POS_IDX["c4"], _POS_IDX["c5"]]): "inner",
    # Spokes (cross-ring) → "spoke"
    frozenset([_POS_IDX["d7"], _POS_IDX["d6"], _POS_IDX["d5"]]): "spoke",
    frozenset([_POS_IDX["g4"], _POS_IDX["f4"], _POS_IDX["e4"]]): "spoke",
    frozenset([_POS_IDX["d1"], _POS_IDX["d2"], _POS_IDX["d3"]]): "spoke",
    frozenset([_POS_IDX["a4"], _POS_IDX["b4"], _POS_IDX["c4"]]): "spoke",
}

_FAMILY_ORDER = ["outer", "middle", "spoke", "inner"]
_FAMILY_LABELS = {
    "outer": "Outer Ring Mill",
    "middle": "Middle Ring Mill",
    "spoke": "Cross-Ring (Spoke) Mill",
    "inner": "Inner Ring Mill",
}

def _primary_family(w_idx: frozenset[int]) -> str:
    """Return the family of the 'most prominent' (first in _FAMILY_ORDER) mill present."""
    mills_present = [m for m in _MILL_IDX_SETS if m.issubset(w_idx)]
    for fam in _FAMILY_ORDER:
        for m in mills_present:
            if _MILL_FAMILY.get(m) == fam:
                return fam
    return "other"

# ── Sub-grouping within a family ──────────────────────────────────────────────

def _sub_group_key(w_idx: frozenset[int], n_mills: int) -> str:
    """Sub-group label: how many mills + rough piece distribution."""
    w_names = _idx_to_pos_names(w_idx)
    outer = sum(1 for p in w_names if p in {"a7","d7","g7","g4","g1","d1","a1","a4"})
    middle = sum(1 for p in w_names if p in {"b6","d6","f6","f4","f2","d2","b2","b4"})
    inner = sum(1 for p in w_names if p in {"c5","d5","e5","e4","e3","d3","c3","c4"})
    if n_mills >= 3:
        return "3+ mills"
    if n_mills == 2:
        return "2 mills"
    # Single mill: characterise the 4 extra pieces by ring
    extra_label = f"outer={outer},mid={middle},inner={inner}"
    return extra_label

# ── Board rendering ────────────────────────────────────────────────────────────

_POS_XY: dict[str, tuple[float, float]] = {
    "a1": (0,0), "d1": (3,0), "g1": (6,0),
    "a4": (0,3), "b4": (1,3), "c4": (2,3),
    "e4": (4,3), "f4": (5,3), "g4": (6,3),
    "a7": (0,6), "d7": (3,6), "g7": (6,6),
    "b2": (1,1), "d2": (3,1), "f2": (5,1),
    "b6": (1,5), "d6": (3,5), "f6": (5,5),
    "c3": (2,2), "d3": (3,2), "e3": (4,2),
    "c5": (2,4), "d5": (3,4), "e5": (4,4),
}
_EDGES: list[tuple[str,str]] = [
    ("a7","d7"),("d7","g7"),("g7","g4"),("g4","g1"),
    ("g1","d1"),("d1","a1"),("a1","a4"),("a4","a7"),
    ("b6","d6"),("d6","f6"),("f6","f4"),("f4","f2"),
    ("f2","d2"),("d2","b2"),("b2","b4"),("b4","b6"),
    ("c5","d5"),("d5","e5"),("e5","e4"),("e4","e3"),
    ("e3","d3"),("d3","c3"),("c3","c4"),("c4","c5"),
    ("a4","b4"),("b4","c4"),
    ("d7","d6"),("d6","d5"),
    ("g4","f4"),("f4","e4"),
    ("d1","d2"),("d2","d3"),
]

_BOARD_BG    = "#D4920A"
_LINE_COLOR  = "black"
_PIECE_COLOR = "white"
_PIECE_EDGE  = "#555555"
_MILL_COLOR  = "#CC0000"

def _board_png(w_positions: list[str],
               mill_triples: list[tuple[str,str,str]],
               label: str,
               dpi: int = 110) -> bytes:
    fig, ax = plt.subplots(figsize=(2.2, 2.5))
    fig.patch.set_facecolor(_BOARD_BG)
    ax.set_facecolor(_BOARD_BG)
    for p1, p2 in _EDGES:
        x1,y1 = _POS_XY[p1]; x2,y2 = _POS_XY[p2]
        ax.plot([x1,x2],[y1,y2], color=_LINE_COLOR, linewidth=1.8, zorder=1)
    for pos,(x,y) in _POS_XY.items():
        ax.plot(x,y,".",color=_LINE_COLOR,markersize=3,zorder=2)
    mill_sq = {p for t in mill_triples for p in t}
    for pos in mill_sq:
        x,y = _POS_XY[pos]
        ax.add_patch(mpatches.Circle((x,y),0.18,color=_MILL_COLOR,zorder=3))
    for pos in w_positions:
        x,y = _POS_XY[pos]
        ax.add_patch(mpatches.Circle((x,y),0.42,facecolor=_PIECE_COLOR,
                                      edgecolor=_PIECE_EDGE,linewidth=1.2,zorder=4))
    ax.set_xlim(-0.85,6.85); ax.set_ylim(-1.1,7.1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(3,-0.8,label,ha="center",va="center",fontsize=8,
            fontweight="bold",color="black",transform=ax.transData)
    buf = io.BytesIO()
    fig.savefig(buf,format="png",dpi=dpi,bbox_inches="tight",facecolor=_BOARD_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ── docx helpers ──────────────────────────────────────────────────────────────

def _add_boards_grid(doc, entries, cols=6, img_width_in=1.3):
    rows_needed = (len(entries) + cols - 1) // cols
    padded = entries + [None] * (rows_needed * cols - len(entries))
    table = doc.add_table(rows=rows_needed, cols=cols)
    table.style = "Table Grid"
    for row in table.rows:
        for cell in row.cells:
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            tcBorders = OxmlElement("w:tcBorders")
            for side in ("top","left","bottom","right","insideH","insideV"):
                border = OxmlElement(f"w:{side}")
                border.set(qn("w:val"), "none")
                tcBorders.append(border)
            tcPr.append(tcBorders)
    idx = 0
    for r in range(rows_needed):
        for c in range(cols):
            entry = padded[idx]; idx += 1
            cell = table.cell(r,c)
            if entry is None:
                cell.paragraphs[0].clear(); continue
            png = _board_png(entry["w"], entry.get("mills",[]), entry["label"])
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run()
            run.add_picture(io.BytesIO(png), width=Inches(img_width_in))

def _para(doc, text, bold=False, size=10):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    return p

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate 7 vs 4 arrangements docx")
    ap.add_argument("--quick", action="store_true",
                    help="Sample 60 B positions per formation instead of all 680 (much faster)")
    ap.add_argument("--out", default=None,
                    help="Output path (default: '7 vs 4 arrangements.docx' in project root)")
    ap.add_argument("--json-out", default=None, metavar="PATH",
                    help="JSON export path (default: data/puzzles/7v4_formations.json)")
    ap.add_argument("--json-only", action="store_true",
                    help="Export JSON only, skip docx generation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    out_path = Path(args.out) if args.out else (_ROOT / "7 vs 4 arrangements.docx")
    json_out = Path(args.json_out) if args.json_out else (_ROOT / "data" / "puzzles" / "7v4_formations.json")
    n_sample = 60 if args.quick else None  # None = all 680

    print("Loading Malom DB …")
    db = _load_malom()
    print("Malom DB ready.")

    # ── Phase 1: Enumerate ────────────────────────────────────────────────────
    print("Enumerating all canonical 7W formations with ≥1 mill …")
    t0 = time.time()
    all_formations = _enumerate_canonical_7w_with_mill()
    print(f"  {len(all_formations):,} canonical formations in {time.time()-t0:.1f}s")

    # ── Phase 2: Two-pass Malom verification ─────────────────────────────────
    # Pass A: quick scan with n_sample random B positions to find candidates
    # Pass B: full 680-check only on candidates from pass A
    QUICK_SAMPLES = 60
    print(f"Malom pass A: scanning {len(all_formations):,} formations with "
          f"{QUICK_SAMPLES} random B arrangements …")
    t0 = time.time()

    candidates: list[frozenset] = []
    checked = 0
    interval = max(1, len(all_formations) // 20)

    # Sort: check multi-mill formations first (more likely to be guaranteed wins)
    all_formations.sort(key=lambda f: -_count_mills(f))

    for canon_idx in all_formations:
        n_wins, n_checked = _verify_formation(db, canon_idx, n_sample=QUICK_SAMPLES)
        checked += 1
        if n_wins == n_checked:
            candidates.append(canon_idx)
        if checked % interval == 0:
            elapsed = time.time() - t0
            rate = checked / elapsed if elapsed > 0 else 0
            eta = (len(all_formations) - checked) / rate if rate > 0 else 0
            print(f"  pass A  {checked:,}/{len(all_formations):,}  "
                  f"candidates so far: {len(candidates)}  "
                  f"ETA {eta:.0f}s", flush=True)

    elapsed_a = time.time() - t0
    print(f"Pass A done: {len(candidates)} candidates in {elapsed_a:.0f}s")

    mode_label = "full 680"
    if n_sample is not None:
        # --quick: skip pass B, use the quick-scan results directly
        print("--quick mode: skipping full verification (pass B)")
        guaranteed_idx = candidates
        mode_label = f"sampled {QUICK_SAMPLES}"
    else:
        # Pass B: full verify the candidates
        print(f"Malom pass B: full 680-check on {len(candidates)} candidates …")
        t0 = time.time()
        guaranteed_idx: list[frozenset] = []
        for i, canon_idx in enumerate(candidates):
            n_wins, n_checked = _verify_formation(db, canon_idx, n_sample=None)
            if n_wins == n_checked:
                guaranteed_idx.append(canon_idx)
            if (i + 1) % max(1, len(candidates) // 10) == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(candidates) - i - 1) / rate if rate > 0 else 0
                print(f"  pass B  {i+1}/{len(candidates)}  "
                      f"confirmed: {len(guaranteed_idx)}  ETA {eta:.0f}s", flush=True)
        elapsed_b = time.time() - t0
        print(f"Pass B done: {len(guaranteed_idx)} guaranteed wins in {elapsed_b:.0f}s")

    guaranteed: list[dict] = []
    for canon_idx in guaranteed_idx:
        w_pos = _idx_to_pos_names(canon_idx)
        mills = _find_mills(canon_idx)
        n_mills = len(mills)
        fam = _primary_family(canon_idx)
        sub = _sub_group_key(canon_idx, n_mills)
        guaranteed.append({
            "w": w_pos,
            "mills": mills,
            "n_mills": n_mills,
            "family": fam,
            "sub": sub,
            "canon_idx": canon_idx,
        })

    print(f"Total guaranteed-win formations: {len(guaranteed)}")

    if not guaranteed:
        print("No guaranteed-win formations found. Check Malom DB or try --quick.")
        return

    # ── JSON export (always) ──────────────────────────────────────────────────
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_data = {
        "version": 1,
        "count": len(guaranteed),
        "formations": [entry["w"] for entry in guaranteed],
    }
    json_out.write_text(json.dumps(json_data, separators=(",", ":")))
    print(f"Saved JSON: {json_out}")

    if args.json_only:
        print("Done (--json-only).")
        return

    # ── Phase 3: Group into families ──────────────────────────────────────────
    # Assign sequential labels per family / sub-group
    # Family order: outer, middle, spoke, inner
    by_family: dict[str, list[dict]] = defaultdict(list)
    for entry in guaranteed:
        by_family[entry["family"]].append(entry)

    # Within each family, sort: 3+ mills first, then 2 mills, then single mill
    # Within single-mill: sort by sub-group string (outer>mid>inner count)
    mill_sort = {"3+ mills": 0, "2 mills": 1}
    def _sort_key(e):
        sg = e["sub"]
        return (mill_sort.get(sg, 2), sg)

    for fam in by_family:
        by_family[fam].sort(key=_sort_key)

    # Label formations
    grand_num = 1
    for fam in _FAMILY_ORDER:
        fam_num = 1
        for entry in by_family.get(fam, []):
            entry["label"] = f"{fam[0].upper()}{fam_num}"
            fam_num += 1
            grand_num += 1

    # ── Phase 4: Stats ────────────────────────────────────────────────────────
    stats_by_family: dict[str, dict] = {}
    for fam in _FAMILY_ORDER:
        entries = by_family.get(fam, [])
        stats_by_family[fam] = {
            "total": len(entries),
            "two_plus_mills": sum(1 for e in entries if e["n_mills"] >= 2),
            "three_plus_mills": sum(1 for e in entries if e["n_mills"] >= 3),
        }

    total_guaranteed = len(guaranteed)
    two_mill_count = sum(1 for e in guaranteed if e["n_mills"] >= 2)
    three_mill_count = sum(1 for e in guaranteed if e["n_mills"] >= 3)

    print(f"\n=== RESULTS ===")
    print(f"Total guaranteed-win formations: {total_guaranteed}")
    for fam in _FAMILY_ORDER:
        s = stats_by_family[fam]
        print(f"  {_FAMILY_LABELS[fam]:30s}: {s['total']:4d}  "
              f"(2+ mills: {s['two_plus_mills']}, 3+ mills: {s['three_plus_mills']})")

    # ── Phase 5: Build docx ───────────────────────────────────────────────────
    print(f"\nBuilding {out_path} …")
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.45)
        section.bottom_margin = Inches(0.45)
        section.left_margin = Inches(0.45)
        section.right_margin = Inches(0.45)

    title = doc.add_heading("7 vs 4 Arrangements", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    intro = doc.add_paragraph(
        "White (7 pieces) vs Black (4 pieces → 3 pieces flying), Black to move. "
        "Malom ultra-strong database confirms all positions are forced White wins "
        "regardless of Black's 3-piece arrangement after transitioning to the flying phase.\n"
        f"Total guaranteed-win formations: {total_guaranteed} "
        f"({mode_label} B-arrangement check, "
        f"{'all confirmed exact' if not args.quick else 'approximate — re-run without --quick for full verification'}).\n"
        "Red dots mark active mill squares. "
        "Canonical forms under D4 symmetry (4 rotations × 4 reflections)."
    )
    intro.alignment = WD_ALIGN_PARAGRAPH.CENTER
    intro.paragraph_format.space_after = Pt(6)

    for fam in _FAMILY_ORDER:
        entries = by_family.get(fam, [])
        if not entries:
            continue
        s = stats_by_family[fam]
        heading_text = (
            f"{_FAMILY_LABELS[fam]} ({s['total']} formations — "
            f"2+ mills: {s['two_plus_mills']}, single mill: {s['total']-s['two_plus_mills']})"
        )
        h = doc.add_heading(heading_text, level=1)
        h.paragraph_format.space_before = Pt(10)

        # Sub-sections by mill count
        sub_groups: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            sub_groups[e["sub"]].append(e)

        for sg_key in sorted(sub_groups.keys(), key=lambda k: mill_sort.get(k, 2)):
            sg_entries = sub_groups[sg_key]
            _para(doc, f"  {sg_key}  ({len(sg_entries)} formations)", bold=True, size=9)
            # Show up to 48 boards per sub-group; cap to keep docx reasonable
            to_show = sg_entries[:48]
            _add_boards_grid(doc, to_show, cols=6, img_width_in=1.3)
            if len(sg_entries) > 48:
                _para(doc, f"  … and {len(sg_entries)-48} more (not shown)", size=8)
            doc.add_paragraph()

    # Summary stats section
    doc.add_heading("Malom Verification Summary", level=1)
    bullets = [
        (f"Total canonical 7W formations examined",
         f"{len(all_formations):,} (all with ≥1 mill, under D4 symmetry)"),
        (f"Guaranteed-win formations",
         f"{total_guaranteed} ({100*total_guaranteed/len(all_formations):.1f}% of all mill formations)"),
        ("Formations with 2+ active mills",
         f"{two_mill_count} ({100*two_mill_count/total_guaranteed:.0f}% of guaranteed wins)"),
        ("Formations with 3+ active mills",
         f"{three_mill_count} ({100*three_mill_count/total_guaranteed:.0f}% of guaranteed wins)"),
        ("Verification method",
         f"{'Full 680 B=3 arrangements per formation' if not args.quick else 'Sample of 60 random B=3 arrangements per formation (quick mode)'}"),
        ("Key insight",
         "These formations guarantee a White win at the moment Black transitions from 4 to 3 "
         "pieces and enters the flying phase, regardless of which 3 Black pieces remain. "
         "Families group by the anchor mill type (outer/middle/spoke/inner ring). "
         "Single-piece adjacency connects many formations within the same sub-group."),
    ]
    for heading_text, detail in bullets:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(heading_text + ": ")
        run.bold = True
        p.add_run(detail)

    doc.save(str(out_path))
    print(f"Saved: {out_path}")
    print(f"Done. {total_guaranteed} guaranteed-win formations across {len(_FAMILY_ORDER)} families.")


if __name__ == "__main__":
    main()
