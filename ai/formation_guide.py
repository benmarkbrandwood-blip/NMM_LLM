"""
ai/formation_guide.py

Formation Guide: given a board position where W has exactly 6 pieces,
find the best matching target formation from the 37 Malom-confirmed
winning configurations (+5 no-mill wins) and return piece assignment arrows.

Two modes:
  naive  — minimum total BFS distance across all 42×8 (D4) candidates.
           B-blocking penalty: +4 per B piece on a target square.
  malom  — same cost function, but after ranking by naive cost, the first
           arrow step is verified against the Malom DB and the formation
           whose next-move is a confirmed W-win is preferred.

Public API:
  best_formation(w_positions, b_positions, mode, db=None) → dict
"""
from __future__ import annotations

from collections import deque
from itertools import permutations
from typing import Optional

# ── Board geometry ─────────────────────────────────────────────────────────────

_POS_COORDS: dict[str, tuple[int, int]] = {
    "a7": (-3, 3), "d7": (0, 3), "g7": (3, 3),
    "a4": (-3, 0), "b4": (-2, 0), "c4": (-1, 0),
    "e4": (1, 0),  "f4": (2, 0),  "g4": (3, 0),
    "a1": (-3, -3), "d1": (0, -3), "g1": (3, -3),
    "b6": (-2, 2),  "d6": (0, 2),  "f6": (2, 2),
    "b2": (-2, -2), "d2": (0, -2), "f2": (2, -2),
    "c5": (-1, 1),  "d5": (0, 1),  "e5": (1, 1),
    "c3": (-1, -1), "d3": (0, -1), "e3": (1, -1),
}
_COORDS_POS: dict[tuple[int, int], str] = {v: k for k, v in _POS_COORDS.items()}

_ADJACENCY: dict[str, list[str]] = {p: [] for p in _POS_COORDS}
for _p1, _p2 in [
    ("a7","d7"), ("d7","g7"), ("g7","g4"), ("g4","g1"),
    ("g1","d1"), ("d1","a1"), ("a1","a4"), ("a4","a7"),
    ("b6","d6"), ("d6","f6"), ("f6","f4"), ("f4","f2"),
    ("f2","d2"), ("d2","b2"), ("b2","b4"), ("b4","b6"),
    ("c5","d5"), ("d5","e5"), ("e5","e4"), ("e4","e3"),
    ("e3","d3"), ("d3","c3"), ("c3","c4"), ("c4","c5"),
    ("a4","b4"), ("b4","c4"),
    ("d7","d6"), ("d6","d5"),
    ("g4","f4"), ("f4","e4"),
    ("d1","d2"), ("d2","d3"),
]:
    _ADJACENCY[_p1].append(_p2)
    _ADJACENCY[_p2].append(_p1)

# ── BFS distance matrix (precomputed at import, 24×24) ────────────────────────

def _bfs_from(src: str) -> dict[str, int]:
    dist: dict[str, int] = {src: 0}
    q = deque([src])
    while q:
        node = q.popleft()
        for nb in _ADJACENCY[node]:
            if nb not in dist:
                dist[nb] = dist[node] + 1
                q.append(nb)
    return dist

_DIST: dict[str, dict[str, int]] = {p: _bfs_from(p) for p in _POS_COORDS}

# ── D4 transforms ─────────────────────────────────────────────────────────────
# 8 symmetries of the board square, centred at d4=(0,0).
# (x, y) conventions: a→-3, d→0, g→+3; row1→-3, row4→0, row7→+3.

_D4: list[tuple[int, int, int, int]] = [
    # (xx, xy, yx, yy)  — new_x = xx*x + xy*y,  new_y = yx*x + yy*y
    ( 1,  0,  0,  1),   # identity
    ( 0,  1, -1,  0),   # rotate 90° CW
    (-1,  0,  0, -1),   # rotate 180°
    ( 0, -1,  1,  0),   # rotate 270° CW
    (-1,  0,  0,  1),   # reflect over vertical axis
    ( 1,  0,  0, -1),   # reflect over horizontal axis
    ( 0, -1, -1,  0),   # reflect over a7–g1 diagonal
    ( 0,  1,  1,  0),   # reflect over a1–g7 diagonal
]


def _transform_pos(pos: str, t: int) -> Optional[str]:
    if pos not in _POS_COORDS:
        return None
    x, y = _POS_COORDS[pos]
    xx, xy, yx, yy = _D4[t]
    nx, ny = xx * x + xy * y, yx * x + yy * y
    return _COORDS_POS.get((nx, ny))


def _transform_formation(w_pos: list[str], t: int) -> Optional[list[str]]:
    result = []
    for pos in w_pos:
        tp = _transform_pos(pos, t)
        if tp is None:
            return None
        result.append(tp)
    return result if len(set(result)) == len(result) else None

# ── Formation data ─────────────────────────────────────────────────────────────

FORMATIONS: list[dict] = [
    {"label": "1",  "w": ["a4","a7","d7","g7","g4","d1"],        "mills": [("a7","d7","g7")]},
    {"label": "2",  "w": ["a7","d7","g7","d1","a1","b4"],         "mills": [("a7","d7","g7")]},
    {"label": "3",  "w": ["a7","g1","d1","a1","b4","b6"],         "mills": [("g1","d1","a1")]},
    {"label": "4",  "w": ["a7","d7","g7","a1","b4","f6"],         "mills": [("a7","d7","g7")]},
    {"label": "5",  "w": ["a7","g7","g1","b6","d6","f6"],         "mills": [("b6","d6","f6")]},
    {"label": "6",  "w": ["a7","d7","g7","b6","f6","f2"],         "mills": [("a7","d7","g7")]},
    {"label": "7",  "w": ["a7","g4","b6","d6","f6","f2"],         "mills": [("b6","d6","f6")]},
    {"label": "8",  "w": ["g4","g1","b6","d6","f6","f2"],         "mills": [("b6","d6","f6")]},
    {"label": "9",  "w": ["d7","b4","b6","f6","f4","f2"],         "mills": [("f6","f4","f2")]},
    {"label": "10", "w": ["b4","b6","d6","f6","f4","d2"],         "mills": [("b6","d6","f6")]},
    {"label": "11", "w": ["g1","d1","a1","b4","b6","c4"],         "mills": [("g1","d1","a1")]},
    {"label": "12", "w": ["a4","d7","g7","b4","d6","c4"],         "mills": [("a4","b4","c4")]},
    {"label": "13", "w": ["a4","d7","g4","b4","d6","c4"],         "mills": [("a4","b4","c4")]},
    {"label": "14", "w": ["a4","d7","a1","b4","d6","c4"],         "mills": [("a4","b4","c4")]},
    {"label": "15", "w": ["a4","d7","g7","b4","f6","c4"],         "mills": [("a4","b4","c4")]},
    {"label": "16", "w": ["a7","d7","g7","b4","f6","c4"],         "mills": [("a7","d7","g7")]},
    {"label": "17", "w": ["a4","d7","a1","b4","f6","c4"],         "mills": [("a4","b4","c4")]},
    {"label": "18", "w": ["a7","d7","g7","b6","f6","c4"],         "mills": [("a7","d7","g7")]},
    {"label": "19", "w": ["a4","a7","a1","b6","f6","c4"],         "mills": [("a1","a4","a7")]},
    {"label": "20", "w": ["a4","d7","b4","d6","f6","c4"],         "mills": [("a4","b4","c4")]},
    {"label": "21", "w": ["a4","g7","b4","d6","f6","c4"],         "mills": [("a4","b4","c4")]},
    {"label": "22", "w": ["a4","g7","b6","d6","f6","c4"],         "mills": [("b6","d6","f6")]},
    {"label": "23", "w": ["a7","g7","b6","d6","f6","c4"],         "mills": [("b6","d6","f6")]},
    {"label": "24", "w": ["a4","a1","b6","d6","f6","c4"],         "mills": [("b6","d6","f6")]},
    {"label": "25", "w": ["a4","d7","b4","d6","f4","c4"],         "mills": [("a4","b4","c4")]},
    {"label": "26", "w": ["a4","g1","b4","b6","d2","c4"],         "mills": [("a4","b4","c4")]},
    {"label": "27", "w": ["a4","d1","b4","b6","d2","c4"],         "mills": [("a4","b4","c4")]},
    {"label": "28", "w": ["a7","g7","b4","b6","b2","c4"],         "mills": [("b2","b4","b6")]},
    {"label": "29", "w": ["g1","d1","a1","b4","b6","c5"],         "mills": [("g1","d1","a1")]},
    {"label": "30", "w": ["a7","d7","g7","b4","d6","c5"],         "mills": [("a7","d7","g7")]},
    {"label": "31", "w": ["a4","a7","a1","d6","f6","c5"],         "mills": [("a1","a4","a7")]},
    {"label": "32", "w": ["a4","d7","b6","d6","f6","c5"],         "mills": [("b6","d6","f6")]},
    {"label": "33", "w": ["a4","a1","b6","d6","f6","c5"],         "mills": [("b6","d6","f6")]},
    {"label": "34", "w": ["a7","d7","f6","f4","f2","c5"],         "mills": [("f6","f4","f2")]},
    {"label": "35", "w": ["a4","d7","g7","b4","c4","d5"],         "mills": [("a4","b4","c4")]},
    {"label": "36", "w": ["a4","d7","a1","b4","c4","d5"],         "mills": [("a4","b4","c4")]},
    {"label": "37", "w": ["a7","d7","g7","b6","c4","d5"],         "mills": [("a7","d7","g7")]},
]

NO_MILL_WINS: list[dict] = [
    {"label": "NM1", "w": ["g4","d1","a1","b6","f4","e5"], "mills": []},
    {"label": "NM2", "w": ["g7","g4","a1","f4","d2","b2"], "mills": []},
    {"label": "NM3", "w": ["a4","g1","b4","d2","e3","c3"], "mills": []},
    {"label": "NM4", "w": ["a4","b6","f6","f2","d5","e4"], "mills": []},
    {"label": "NM5", "w": ["g1","a1","f4","e5","e4","d3"], "mills": []},
]

ALL_FORMATIONS: list[dict] = FORMATIONS + NO_MILL_WINS

# ── BFS next-step toward target ────────────────────────────────────────────────

def _next_step(src: str, dst: str, occupied: set[str]) -> Optional[str]:
    """BFS first step from src toward dst, avoiding occupied squares (except src/dst)."""
    if src == dst:
        return dst
    visited = {src}
    q: deque[tuple[str, str]] = deque()
    for nb in _ADJACENCY[src]:
        if nb not in occupied or nb == dst:
            visited.add(nb)
            q.append((nb, nb))
    while q:
        node, first = q.popleft()
        if node == dst:
            return first
        for nb in _ADJACENCY[node]:
            if nb not in visited and (nb not in occupied or nb == dst):
                visited.add(nb)
                q.append((nb, first))
    return None  # path blocked

# ── Min-cost piece assignment ──────────────────────────────────────────────────

def _min_cost_assignment(
    w_current: list[str],
    target: list[str],
    b_set: set[str],
) -> tuple[int, list[tuple[str, str]]]:
    """Brute-force minimum total BFS-distance assignment of w_current → target.
    Adds +4 penalty per B piece sitting on a target square.
    Returns (total_cost, [(current_pos, target_pos), ...]).
    """
    n = len(w_current)
    b_penalty = sum(4 for t in target if t in b_set)

    best_cost = 10_000
    best_pairs: list[tuple[str, str]] = []

    for perm in permutations(range(len(target))):
        cost = b_penalty
        for i, j in enumerate(perm[:n]):
            cost += _DIST[w_current[i]][target[j]]
        if cost < best_cost:
            best_cost = cost
            best_pairs = [(w_current[i], target[perm[i]]) for i in range(n)]

    return best_cost, best_pairs


# ── Public API ─────────────────────────────────────────────────────────────────

def best_formation(
    w_positions: list[str],
    b_positions: list[str],
    mode: str = "naive",
    db=None,
) -> dict:
    """Find best target formation for W to aim for.

    w_positions: current positions of all W pieces (should be 6)
    b_positions: current positions of all B pieces
    mode: "naive" or "malom"
    db: MalomDB instance (required for mode="malom", else falls back to naive)

    Returns:
        formation_id: label of the matched formation (str)
        target_squares: the 6 target W positions after transform (list[str])
        arrows: [{from, to}, ...] for pieces that need to move
        stay: [str, ...] positions where W pieces are already on target
        total_distance: naive BFS cost (int)
        mode_used: "naive" or "malom"
    """
    if not w_positions:
        return _empty_result(mode)

    b_set = set(b_positions)
    w_list = list(w_positions)

    # ── Step 1: rank all 42×8 candidates by naive cost ────────────────────────
    candidates: list[tuple[int, dict]] = []
    for form in ALL_FORMATIONS:
        for t_idx in range(8):
            target = _transform_formation(form["w"], t_idx)
            if target is None:
                continue
            # Skip formations that overlap current B positions on ALL 6 squares
            # (would require capturing all B pieces — not a useful guide)
            b_overlap = sum(1 for t in target if t in b_set)
            if b_overlap == len(b_positions) and len(b_positions) >= 4:
                continue
            cost, pairs = _min_cost_assignment(w_list, target, b_set)
            candidates.append((cost, {
                "formation_id": form["label"],
                "target_squares": target,
                "pairs": pairs,
                "base_cost": cost,
            }))

    if not candidates:
        return _empty_result(mode)

    candidates.sort(key=lambda x: x[0])

    # ── Step 2: Malom verification (top-K candidates) ─────────────────────────
    mode_used = "naive"
    chosen = candidates[0][1]

    if mode == "malom" and db is not None:
        try:
            from game.board import BoardState
            # Build current board dict
            positions: dict[str, str] = {}
            for pos in w_list:
                positions[pos] = "W"
            for pos in b_positions:
                positions[pos] = "B"
            board = BoardState.from_setup(positions, turn="W", phase="move")

            occupied = set(w_list) | b_set

            K = min(5, len(candidates))
            for _, cand in candidates[:K]:
                pairs = cand["pairs"]
                # Check each moving piece: is the first BFS step a Malom W-win?
                all_good = True
                for w_cur, t in pairs:
                    if w_cur == t:
                        continue  # already at target — no move needed
                    step = _next_step(w_cur, t, occupied - {w_cur})
                    if step is None or step == w_cur:
                        continue
                    move = {"from": w_cur, "to": step, "capture": None}
                    try:
                        child = board.apply_move(move)  # child is B-to-move
                        val = db.query_value(child)
                        # "L" from child (B-to-move) means B loses = W wins = good.
                        # Anything else (W/D) means not a forced W-win → reject.
                        if val is not None and val.outcome != "L":
                            all_good = False
                            break
                    except Exception:
                        pass  # DB miss or error — treat as neutral
                if all_good:
                    chosen = cand
                    mode_used = "malom"
                    break
            else:
                # No fully-confirmed candidate — fall back to naive best
                chosen = candidates[0][1]
                mode_used = "malom"  # still label as malom (best effort)
        except Exception:
            pass  # BoardState unavailable or DB error — use naive result

    pairs = chosen["pairs"]
    stay = [w for w, t in pairs if w == t]
    arrows = [{"from": w, "to": t} for w, t in pairs if w != t]

    return {
        "formation_id": chosen["formation_id"],
        "target_squares": chosen["target_squares"],
        "arrows": arrows,
        "stay": stay,
        "total_distance": chosen["base_cost"],
        "mode_used": mode_used,
    }


def _empty_result(mode: str) -> dict:
    return {
        "formation_id": None,
        "target_squares": [],
        "arrows": [],
        "stay": [],
        "total_distance": 0,
        "mode_used": mode,
    }


# ── 7v4 formation guide ────────────────────────────────────────────────────────

_7V4_FORMATIONS: Optional[list[list[str]]] = None


def _load_7v4_formations() -> list[list[str]]:
    """Lazy-load 7v4 guaranteed-win formations from data/puzzles/7v4_formations.json.

    Returns [] if the file is not yet available (background scan still running).
    Does not cache a miss so the file is picked up as soon as it appears.
    """
    global _7V4_FORMATIONS
    if _7V4_FORMATIONS is not None:
        return _7V4_FORMATIONS
    try:
        import json as _json
        from pathlib import Path as _Path
        json_path = _Path(__file__).parent.parent / "data" / "puzzles" / "7v4_formations.json"
        if not json_path.exists():
            return []
        data = _json.loads(json_path.read_text())
        _7V4_FORMATIONS = data.get("formations", [])
        return _7V4_FORMATIONS
    except Exception:
        return []


def best_formation_7v4(
    w_positions: list[str],
    b_positions: list[str],
    mode: str = "naive",
    db=None,
) -> dict:
    """Find the best 7v4 target formation for W to aim for.

    w_positions: current W pieces (should be 7)
    b_positions: current B pieces (should be 4)
    Returns the same dict shape as best_formation().
    """
    formations = _load_7v4_formations()
    if not formations or not w_positions:
        return _empty_result(mode)

    w_set = set(w_positions)
    b_set = set(b_positions)
    w_list = list(w_positions)

    # Step 1: quick overlap ranking across all 5,176 × 8 D4 variants
    overlap_ranked: list[tuple[int, int, list[str], int]] = []
    for form_idx, form_w in enumerate(formations):
        for t_idx in range(8):
            target = _transform_formation(form_w, t_idx)
            if target is None or len(target) != 7:
                continue
            overlap = sum(1 for p in target if p in w_set)
            b_blocking = sum(1 for p in target if p in b_set)
            overlap_ranked.append((-overlap, b_blocking, target, form_idx))

    if not overlap_ranked:
        return _empty_result(mode)

    overlap_ranked.sort(key=lambda x: (x[0], x[1]))

    # Step 2: exact BFS assignment for candidates at max or max-1 overlap (cap 100)
    max_overlap = -overlap_ranked[0][0]
    cutoff = max_overlap - 1
    top_pool = []
    for entry in overlap_ranked:
        if -entry[0] < cutoff:
            break
        top_pool.append(entry)
        if len(top_pool) >= 100:
            break

    candidates: list[tuple[int, dict]] = []
    for neg_ov, b_block, target, form_idx in top_pool:
        cost, pairs = _min_cost_assignment(w_list, target, b_set)
        candidates.append((cost, {
            "formation_id": f"7v4:{form_idx}",
            "target_squares": target,
            "pairs": pairs,
            "base_cost": cost,
        }))

    candidates.sort(key=lambda x: x[0])

    # Step 3: optional Malom verification (same logic as best_formation)
    mode_used = "naive"
    chosen = candidates[0][1]

    if mode == "malom" and db is not None:
        try:
            from game.board import BoardState
            positions: dict[str, str] = {}
            for pos in w_list:
                positions[pos] = "W"
            for pos in b_positions:
                positions[pos] = "B"
            board = BoardState.from_setup(positions, turn="W", phase="move")
            occupied = w_set | b_set

            K = min(5, len(candidates))
            for _, cand in candidates[:K]:
                pairs = cand["pairs"]
                all_good = True
                for w_cur, t in pairs:
                    if w_cur == t:
                        continue
                    step = _next_step(w_cur, t, occupied - {w_cur})
                    if step is None or step == w_cur:
                        continue
                    move = {"from": w_cur, "to": step, "capture": None}
                    try:
                        child = board.apply_move(move)
                        val = db.query_value(child)
                        if val is not None and val.outcome != "L":
                            all_good = False
                            break
                    except Exception:
                        pass
                if all_good:
                    chosen = cand
                    mode_used = "malom"
                    break
            else:
                chosen = candidates[0][1]
                mode_used = "malom"
        except Exception:
            pass

    pairs = chosen["pairs"]
    stay = [w for w, t in pairs if w == t]
    arrows = [{"from": w, "to": t} for w, t in pairs if w != t]

    return {
        "formation_id": chosen["formation_id"],
        "target_squares": chosen["target_squares"],
        "arrows": arrows,
        "stay": stay,
        "total_distance": chosen["base_cost"],
        "mode_used": mode_used,
    }
