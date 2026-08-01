#!/usr/bin/env python3
"""tools/build_endgame_db_v2.py — Generalized retrograde endgame solver for NMM.

Usage
-----
    # Build a specific table (loads its sub-tables from disk automatically):
    python tools/build_endgame_db_v2.py --nW 4 --nB 3

    # Build all tables up to nW+nB ≤ <max-sum> in dependency order:
    python tools/build_endgame_db_v2.py --build-all [--max-sum 11]

    # Skip tables that are already complete on disk (content-verified):
    python tools/build_endgame_db_v2.py --build-all --skip-existing

    # Verify a specific table (size, checksum, random content sampling):
    python tools/build_endgame_db_v2.py --verify --nW 8 --nB 3 --out-dir /mnt/windows/NMM_DB

    # Verify all tables found in a directory:
    python tools/build_endgame_db_v2.py --verify-all --out-dir /mnt/windows/NMM_DB

    # Write missing .sha256 sidecars for tables that pass content verification:
    python tools/build_endgame_db_v2.py --fix-sidecars --out-dir /mnt/windows/NMM_DB

Output: data/endgame/endgame_{nW}_{nB}.wdl
Checksum sidecar: data/endgame/endgame_{nW}_{nB}.wdl.sha256

Algorithm
---------
All C(24,nW)×C(24-nW,nB)×2 positions are enumerated via the combinatorial index.

Pass 0: mark every position whose outcome is immediately determinable (blocked
move-phase mover → LOSS; any mill-closing move with a WIN capture → WIN).

Iterative forward passes propagate WIN and LOSS until fixed-point.
Remaining UNKNOWN positions → DRAW.

Tables are built in ascending (nW+nB) order so sub-tables are always fully
solved before they are consulted.  The 3v3 base case requires no sub-tables
because any mill capture there immediately reduces the opponent below 3 pieces.

Completion guarantee
--------------------
A fully-built table contains zero WDL_UNKNOWN values: the final pass promotes
all remaining unknowns to DRAW, then fills every non-canonical slot from its
canonical equivalent.  An interrupted pre-allocation is nearly 100% UNKNOWN
(the file is zeroed at creation time, and WDL_UNKNOWN == 0).  Verification
exploits this by sampling random positions and checking for UNKNOWN values.

Performance notes
-----------------
* _CT: precomputed Pascal's triangle avoids math.comb() function-call overhead.
* _RI: reusable 24-int buffer for the B-remapping step in _encode.
* Inner move loop uses bitmask occupancy checks and incremental mask updates
  rather than set/list allocations.
* new_mover list is only built when needed (mill-closing or non-capture encode).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util as _ilu
import logging
import mmap
import random
import re
import sys
import time
import types
from math import comb
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Load ai.endgame_solved_db without triggering ai/__init__.py heavy deps.
_ai_pkg = types.ModuleType("ai")
_ai_pkg.__path__ = [str(_ROOT / "ai")]
sys.modules.setdefault("ai", _ai_pkg)
_spec = _ilu.spec_from_file_location(
    "ai.endgame_solved_db", str(_ROOT / "ai" / "endgame_solved_db.py")
)
_esdb_mod = _ilu.module_from_spec(_spec)
sys.modules["ai.endgame_solved_db"] = _esdb_mod
_spec.loader.exec_module(_esdb_mod)

# Load ai.board_symmetry for D4 canonicalization helpers.
_bs_spec = _ilu.spec_from_file_location(
    "ai.board_symmetry", str(_ROOT / "ai" / "board_symmetry.py")
)
_bs_mod = _ilu.module_from_spec(_bs_spec)
sys.modules["ai.board_symmetry"] = _bs_mod
_bs_spec.loader.exec_module(_bs_mod)
_BPERM = _bs_mod._BOARD_PERM  # _BPERM[sym_idx][old_idx] = new_idx

from game.board import ADJACENCY, MILLS, POSITIONS

get_wdl = _esdb_mod.get_wdl
set_wdl = _esdb_mod.set_wdl
WDL_UNKNOWN = _esdb_mod.WDL_UNKNOWN
WDL_WIN = _esdb_mod.WDL_WIN
WDL_LOSS = _esdb_mod.WDL_LOSS
WDL_DRAW = _esdb_mod.WDL_DRAW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Fast combinatorial helpers ─────────────────────────────────────────────────
# Precomputed Pascal's triangle: _CT[n][k] = C(n, k) for 0 ≤ n,k ≤ 24.
# Avoids math.comb() function-call overhead in tight inner loops.
_CT: list[list[int]] = [[0] * 25 for _ in range(25)]
for _n in range(25):
    _CT[_n][0] = 1
    for _k in range(1, 25):
        if _k <= _n:
            _CT[_n][_k] = _CT[_n - 1][_k - 1] + _CT[_n - 1][_k]

# Reusable buffer: _RI[sq] = rank of square sq in the "remaining" list after
# removing white pieces.  Single-threaded use only.
_RI: list[int] = [0] * 24

# ── Board constants ────────────────────────────────────────────────────────────

_POS_TO_IDX: dict[str, int] = {pos: i for i, pos in enumerate(POSITIONS)}
_N = 24

# ── Adjacency as index lists ───────────────────────────────────────────────────

_ADJACENCY_IDX: list[list[int]] = [[] for _ in range(_N)]
for _pos_name, _neighbors in ADJACENCY.items():
    _ADJACENCY_IDX[_POS_TO_IDX[_pos_name]] = [_POS_TO_IDX[nb] for nb in _neighbors]

# ── Mill bitmasks per square ────────────────────────────────────────────────────

_MILL_MASKS_FOR: list[list[int]] = [[] for _ in range(_N)]
for _mill in MILLS:
    _mask = 0
    for _p in _mill:
        _mask |= 1 << _POS_TO_IDX[_p]
    for _p in _mill:
        _MILL_MASKS_FOR[_POS_TO_IDX[_p]].append(_mask)


# ── D4 canonicalization ────────────────────────────────────────────────────────
# Precomputed bitmask permutation pairs for the 7 non-identity D4 transforms.
# Each entry is a list of (old_bit, new_bit) pairs covering all 24 squares.
# Using bitmasks avoids list allocations in the inner canonicalization loop.
_BPERM_MASKS: list[list[tuple[int, int]]] = []
for _sym_idx in range(1, 8):
    _perm = _BPERM[_sym_idx]
    if _perm is None:
        continue
    _BPERM_MASKS.append([(1 << _old, 1 << _perm[_old]) for _old in range(_N)])


def _canonical_indices(w: list[int], b: list[int]) -> tuple[list[int], list[int]]:
    """Return the D4-canonical (w, b) index lists (bitmask-minimum over 8 transforms)."""
    w_mask = 0
    for i in w:
        w_mask |= 1 << i
    b_mask = 0
    for i in b:
        b_mask |= 1 << i
    best_w, best_b = w_mask, b_mask
    for pairs in _BPERM_MASKS:
        tw = tb = 0
        for old_bit, new_bit in pairs:
            if w_mask & old_bit:
                tw |= new_bit
            if b_mask & old_bit:
                tb |= new_bit
        if (tw, tb) < (best_w, best_b):
            best_w, best_b = tw, tb
    w_can = [i for i in range(_N) if (best_w >> i) & 1]
    b_can = [i for i in range(_N) if (best_b >> i) & 1]
    return w_can, b_can


def _is_canonical(w: list[int], b: list[int]) -> bool:
    w_can, b_can = _canonical_indices(w, b)
    return w_can == w and b_can == b


def _closes_mill(piece_mask: int, to_idx: int) -> bool:
    for mm in _MILL_MASKS_FOR[to_idx]:
        if (piece_mask & mm) == mm:
            return True
    return False


def _in_mill(piece_idx: int, piece_mask: int) -> bool:
    for mm in _MILL_MASKS_FOR[piece_idx]:
        if (piece_mask & mm) == mm:
            return True
    return False


# ── General encode / decode ────────────────────────────────────────────────────

def _table_size(nW: int, nB: int) -> int:
    return _CT[_N][nW] * _CT[_N - nW][nB] * 2


def _packed_table_bytes(nW: int, nB: int) -> int:
    return (_table_size(nW, nB) + 3) >> 2


def _encode(
    w_sorted: list[int], b_sorted: list[int], turn_bit: int, nC_b: int
) -> int:
    """Pack (W_indices, B_indices, turn_bit) into a table position-ID.

    nC_b = C(_N - nW, nB) must be pre-computed by the caller.
    Uses precomputed _CT and _RI buffer — no per-call heap allocations.
    W_indices are always passed first, regardless of who is to move.
    """
    nW = len(w_sorted)
    nB = len(b_sorted)
    # White rank: Σ C(w[i], i+1)
    wr = 0
    for i in range(nW):
        wr += _CT[w_sorted[i]][i + 1]
    # Fill _RI: for each non-white square, _RI[sq] = its rank in the remaining list.
    k = 0
    wp = 0
    for sq in range(_N):
        if wp < nW and w_sorted[wp] == sq:
            wp += 1
        else:
            _RI[sq] = k
            k += 1
    # Black rank: Σ C(_RI[b[i]], i+1)
    br = 0
    for i in range(nB):
        br += _CT[_RI[b_sorted[i]]][i + 1]
    return wr * nC_b * 2 + br * 2 + turn_bit


def _decode(
    pos_id: int, nW: int, nB: int, nC_b: int
) -> tuple[list[int], list[int], int]:
    """Unpack a position-ID into (w_sorted, b_sorted, turn_bit).

    Inlines combo_unrank using the _CT table for fast comb lookups.
    """
    turn_bit = pos_id & 1
    rem = pos_id >> 1
    br = rem % nC_b
    wr = rem // nC_b
    # Unrank white (inline combo_unrank(wr, nW, 24))
    w = []
    rr = wr
    up = _N - 1
    for i in range(nW, 0, -1):
        c = up
        while c >= i - 1 and _CT[c][i] > rr:
            c -= 1
        rr -= _CT[c][i]
        w.append(c)
        up = c - 1
    w.reverse()
    # Unrank black remapping (inline combo_unrank(br, nB, _N-nW))
    b_rem = []
    rr = br
    up = _N - nW - 1
    for i in range(nB, 0, -1):
        c = up
        while c >= i - 1 and _CT[c][i] > rr:
            c -= 1
        rr -= _CT[c][i]
        b_rem.append(c)
        up = c - 1
    b_rem.reverse()
    # Map remapped B indices back to actual squares
    k = 0
    wp = 0
    remaining = []
    for sq in range(_N):
        if wp < nW and w[wp] == sq:
            wp += 1
        else:
            remaining.append(sq)
    b = sorted(remaining[j] for j in b_rem)
    return w, b, turn_bit


# ── Capture helpers ────────────────────────────────────────────────────────────

def _valid_captures(other_list: list[int]) -> list[int]:
    """Non-mill opponent pieces; fall back to all if every piece is in a mill."""
    other_mask = 0
    for i in other_list:
        other_mask |= 1 << i
    non_mill = [i for i in other_list if not _in_mill(i, other_mask)]
    return non_mill if non_mill else list(other_list)


def _best_capture_wdl_for_mover(
    new_mover: list[int],
    other_list: list[int],
    turn_bit: int,
    nW: int,
    nB: int,
    sub_tables: dict[tuple[int, int], bytes],
) -> int:
    """WDL for the current mover after closing a mill, choosing the best capture.

    Returns WDL_WIN, WDL_DRAW, WDL_LOSS, or WDL_UNKNOWN.
    Cross-table convention:
      turn_bit == 0 (W moves): W captures B piece → sub-table (nW, nB-1), B next.
      turn_bit == 1 (B moves): B captures W piece → sub-table (nW-1, nB), W next.
    W_indices are always the first argument to _encode.
    """
    captures = _valid_captures(other_list)
    best = WDL_LOSS
    has_unknown = False

    for cap_idx in captures:
        new_other = sorted(i for i in other_list if i != cap_idx)
        n_new_other = len(new_other)

        if n_new_other < 3:
            return WDL_WIN  # opponent below 3 → immediate loss for them

        if turn_bit == 0:
            sub_key_nw, sub_key_nb = nW, n_new_other
            sub_nC_b = _CT[_N - sub_key_nw][sub_key_nb]
            w_c, b_c = _canonical_indices(new_mover, new_other)
            sub_key = _encode(w_c, b_c, 1, sub_nC_b)
        else:
            sub_key_nw, sub_key_nb = n_new_other, nB
            sub_nC_b = _CT[_N - sub_key_nw][sub_key_nb]
            w_c, b_c = _canonical_indices(new_other, new_mover)
            sub_key = _encode(w_c, b_c, 0, sub_nC_b)

        sub_tbl = sub_tables.get((sub_key_nw, sub_key_nb))
        if sub_tbl is None:
            has_unknown = True
            continue

        sub_val = get_wdl(sub_tbl, sub_key)
        if sub_val == WDL_LOSS:
            return WDL_WIN
        elif sub_val == WDL_WIN:
            pass
        elif sub_val == WDL_DRAW:
            if best != WDL_WIN:
                best = WDL_DRAW

    if has_unknown and best == WDL_LOSS:
        return WDL_UNKNOWN
    return best


# ── Core solver ────────────────────────────────────────────────────────────────

def _process_pos(
    w: list[int], b: list[int], turn_bit: int,
    table: bytearray | mmap.mmap,
    nW: int, nB: int, nC_b: int,
    sub_tables: dict,
) -> int:
    """Evaluate one position; return WDL_WIN/LOSS/UNKNOWN.

    Bitmask occupancy checks and incremental mask updates avoid Python object
    allocations in the hot path.  new_mover list is only built for mill-closing
    moves (rare) and non-capture moves that need encoding.
    """
    mover = w if turn_bit == 0 else b
    other = b if turn_bit == 0 else w
    n_mover = len(mover)
    fly_mover = n_mover <= 3
    next_bit = 1 - turn_bit

    mover_mask = 0
    for i in mover:
        mover_mask |= 1 << i
    occ_mask = mover_mask
    for i in other:
        occ_mask |= 1 << i

    all_opponent_win = True
    has_any_move = False

    for fi in range(n_mover):
        from_idx = mover[fi]
        from_bit = 1 << from_idx
        mover_no_fi = mover_mask ^ from_bit

        targets = range(_N) if fly_mover else _ADJACENCY_IDX[from_idx]

        for to_idx in targets:
            to_bit = 1 << to_idx
            if occ_mask & to_bit:
                continue
            has_any_move = True

            new_mover_mask = mover_no_fi | to_bit

            if _closes_mill(new_mover_mask, to_idx):
                new_mover = []
                for j in range(n_mover):
                    if j != fi:
                        new_mover.append(mover[j])
                new_mover.append(to_idx)
                new_mover.sort()

                outcome = _best_capture_wdl_for_mover(
                    new_mover, other, turn_bit, nW, nB, sub_tables
                )
                if outcome == WDL_WIN:
                    return WDL_WIN
                elif outcome in (WDL_DRAW, WDL_UNKNOWN):
                    all_opponent_win = False
            else:
                new_mover = []
                for j in range(n_mover):
                    if j != fi:
                        new_mover.append(mover[j])
                new_mover.append(to_idx)
                new_mover.sort()

                if turn_bit == 0:
                    w_c, b_c = _canonical_indices(new_mover, other)
                else:
                    w_c, b_c = _canonical_indices(other, new_mover)
                succ_id = _encode(w_c, b_c, next_bit, nC_b)
                sv = get_wdl(table, succ_id)
                if sv == WDL_LOSS:
                    return WDL_WIN
                elif sv != WDL_WIN:
                    all_opponent_win = False

    if not has_any_move:
        return WDL_LOSS
    if all_opponent_win:
        return WDL_LOSS
    return WDL_UNKNOWN


def solve_table(
    nW: int,
    nB: int,
    sub_tables: dict[tuple[int, int], bytes],
    out_path: Path,
    verbose: bool = True,
) -> None:
    """Solve all (nW, nB) positions and write the WDL file to *out_path*.

    sub_tables must contain fully-solved (nW, nB-1) and (nW-1, nB) tables.
    The 3v3 base case passes sub_tables={} because mill captures there
    always reduce the opponent to 2 pieces (immediate WIN).

    The file is pre-allocated as a sparse file (OS manages paging) so large
    tables (5v4, 6v5, …) never require the full bytes to be in RAM at once.
    """
    ts = _table_size(nW, nB)
    n_bytes = _packed_table_bytes(nW, nB)
    nC_b = _CT[_N - nW][nB]
    t0 = time.time()

    # Pre-allocate sparse file (all zeros = WDL_UNKNOWN = valid start state).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as _pre:
        _pre.seek(max(n_bytes, 1) - 1)
        _pre.write(b"\x00")
    _fh = open(out_path, "r+b")
    table = mmap.mmap(_fh.fileno(), n_bytes)

    # ── Precompute canonical position IDs (~ts/8) ─────────────────────────────
    canonical_ids: list[int] = []
    for pos_id in range(ts):
        w, b, _tb = _decode(pos_id, nW, nB, nC_b)
        if _is_canonical(w, b):
            canonical_ids.append(pos_id)

    if verbose:
        logger.info(
            "(%d,%d) Canonical positions: %d / %d (%.1f%%)",
            nW, nB, len(canonical_ids), ts, 100.0 * len(canonical_ids) / ts,
        )

    # ── Pass 0: mark terminals (canonical positions only) ─────────────────────
    n_pass0 = 0
    for pos_id in canonical_ids:
        w, b, turn_bit = _decode(pos_id, nW, nB, nC_b)
        v = _process_pos(w, b, turn_bit, table, nW, nB, nC_b, sub_tables)
        if v != WDL_UNKNOWN:
            set_wdl(table, pos_id, v)
            n_pass0 += 1

    if verbose:
        logger.info(
            "(%d,%d) Pass 0: %d resolved (%.1fs)", nW, nB, n_pass0, time.time() - t0
        )

    # ── Iterative forward passes (canonical positions only) ───────────────────
    for pass_num in range(1, 60):
        changed = 0
        tp = time.time()
        for pos_id in canonical_ids:
            if get_wdl(table, pos_id) != WDL_UNKNOWN:
                continue
            w, b, turn_bit = _decode(pos_id, nW, nB, nC_b)
            v = _process_pos(w, b, turn_bit, table, nW, nB, nC_b, sub_tables)
            if v != WDL_UNKNOWN:
                set_wdl(table, pos_id, v)
                changed += 1

        if verbose:
            logger.info(
                "(%d,%d) Pass %d: %d newly resolved (%.1fs)",
                nW, nB, pass_num, changed, time.time() - tp,
            )
        if changed == 0:
            break

    # ── Mark remaining canonical UNKNOWN as DRAW ──────────────────────────────
    n_draw = 0
    for pos_id in canonical_ids:
        if get_wdl(table, pos_id) == WDL_UNKNOWN:
            set_wdl(table, pos_id, WDL_DRAW)
            n_draw += 1

    # ── Fill non-canonical positions from their canonical equivalents ─────────
    try:
        for pos_id in range(ts):
            w, b, turn_bit = _decode(pos_id, nW, nB, nC_b)
            w_can, b_can = _canonical_indices(w, b)
            if w_can == w and b_can == b:
                continue  # canonical: already solved
            can_id = _encode(w_can, b_can, turn_bit, nC_b)
            set_wdl(table, pos_id, get_wdl(table, can_id))

        if verbose:
            n_win = sum(1 for i in range(ts) if get_wdl(table, i) == WDL_WIN)
            n_loss = sum(1 for i in range(ts) if get_wdl(table, i) == WDL_LOSS)
            logger.info(
                "(%d,%d) Solved: %d WIN  %d LOSS  %d DRAW  (total %d, %.1fs)",
                nW, nB, n_win, n_loss, n_draw, ts, time.time() - t0,
            )

        table.flush()
    finally:
        table.close()
        _fh.close()


def solve_3_3(out_dir: Path, verbose: bool = True) -> None:
    """Convenience wrapper: solve the (3,3) base case."""
    solve_table(3, 3, {}, _wdl_path(out_dir, 3, 3), verbose=verbose)


# ── Table file I/O ─────────────────────────────────────────────────────────────

def _wdl_path(out_dir: Path, nW: int, nB: int) -> Path:
    return out_dir / f"endgame_{nW}_{nB}.wdl"


def _sha256_path(out_dir: Path, nW: int, nB: int) -> Path:
    return out_dir / f"endgame_{nW}_{nB}.wdl.sha256"


def _compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_checksum_sidecar(out_dir: Path, nW: int, nB: int) -> str:
    wdl_path = _wdl_path(out_dir, nW, nB)
    sha_path = _sha256_path(out_dir, nW, nB)
    digest = _compute_file_sha256(wdl_path)
    sha_path.write_text(digest + "\n", encoding="ascii")
    return digest


def _table_status(out_dir: Path, nW: int, nB: int) -> tuple[bool, str]:
    """Return factual status of a table file (exists, size, checksum).

    Returns (ok, message).  ok=True means size is correct AND (either the
    sidecar matches OR no sidecar exists — size alone is not a completion
    guarantee; use _verify_table_content for that).
    """
    p = _wdl_path(out_dir, nW, nB)
    if not p.exists():
        return False, "missing .wdl"
    expected_bytes = _packed_table_bytes(nW, nB)
    actual_bytes = p.stat().st_size
    if actual_bytes != expected_bytes:
        return False, f"size mismatch ({actual_bytes} != {expected_bytes})"
    sha_path = _sha256_path(out_dir, nW, nB)
    if not sha_path.exists():
        return True, "size verified; no checksum sidecar"
    try:
        stored = sha_path.read_text(encoding="ascii").strip()
    except OSError as e:
        return False, f"checksum unreadable ({e})"
    current = _compute_file_sha256(p)
    if stored != current:
        return False, "checksum mismatch"
    return True, "size and checksum verified"


def _load_table(out_dir: Path, nW: int, nB: int) -> bytes | None:
    p = _wdl_path(out_dir, nW, nB)
    if not p.exists():
        return None
    expected_bytes = (_table_size(nW, nB) + 3) >> 2
    data = p.read_bytes()
    if len(data) != expected_bytes:
        logger.warning(
            "(%d,%d) Size mismatch: %s has %d bytes (expected %d) — skipping.",
            nW, nB, p, len(data), expected_bytes,
        )
        return None
    return data


# ── Content-based verification ─────────────────────────────────────────────────

def _verify_table_content(
    out_dir: Path,
    nW: int,
    nB: int,
    n_samples: int = 2000,
    seed: int = 0x4E4D4D,
) -> tuple[bool, int, int, list[str]]:
    """Sample random positions and check for WDL_UNKNOWN values.

    A completed table contains zero UNKNOWN values (final pass promotes all
    remaining unknowns to DRAW).  An interrupted pre-allocation is nearly
    100% UNKNOWN (file is zeroed at creation).

    Uses mmap so even multi-GB tables are not loaded into RAM.

    Returns (ok, n_unknown_found, n_sampled, issues).
    """
    issues: list[str] = []
    p = _wdl_path(out_dir, nW, nB)

    if not p.exists():
        return False, 0, 0, [f"missing .wdl: {p}"]

    ts = _table_size(nW, nB)
    expected_bytes = _packed_table_bytes(nW, nB)
    actual_bytes = p.stat().st_size

    if actual_bytes != expected_bytes:
        return False, 0, 0, [
            f"size mismatch: {actual_bytes} bytes on disk, expected {expected_bytes}"
        ]

    n_samples = min(n_samples, ts)
    rng = random.Random(seed)
    sample_ids = rng.sample(range(ts), n_samples)

    n_unknown = 0
    with open(p, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), expected_bytes, access=mmap.ACCESS_READ)
        try:
            for pos_id in sample_ids:
                if get_wdl(mm, pos_id) == WDL_UNKNOWN:
                    n_unknown += 1
        finally:
            mm.close()

    if n_unknown > 0:
        issues.append(
            f"{n_unknown}/{n_samples} sampled positions are UNKNOWN "
            "(interrupted or incomplete build)"
        )

    return n_unknown == 0, n_unknown, n_samples, issues


def _is_table_complete(
    out_dir: Path,
    nW: int,
    nB: int,
    fast_samples: int = 256,
) -> tuple[bool, str]:
    """True if the table is provably complete.

    Strategy:
      1. If a matching .sha256 sidecar exists → trust it (checksums don't lie).
      2. If size is wrong → definitely incomplete.
      3. If size is right but no sidecar → sample content for UNKNOWN values.
         (Catches interrupted pre-allocations that left the file at full size
         but with all bytes zeroed = WDL_UNKNOWN.)
    """
    p = _wdl_path(out_dir, nW, nB)
    if not p.exists():
        return False, "missing .wdl"

    expected_bytes = _packed_table_bytes(nW, nB)
    actual_bytes = p.stat().st_size
    if actual_bytes != expected_bytes:
        return False, f"size mismatch ({actual_bytes} != {expected_bytes})"

    sha_path = _sha256_path(out_dir, nW, nB)
    if sha_path.exists():
        try:
            stored = sha_path.read_text(encoding="ascii").strip()
        except OSError as e:
            return False, f"checksum unreadable ({e})"
        current = _compute_file_sha256(p)
        if stored != current:
            return False, "checksum mismatch (corrupted or interrupted)"
        return True, "checksum verified"

    # No sidecar: probe content for UNKNOWN values.
    ok, n_unk, n_samp, _ = _verify_table_content(
        out_dir, nW, nB, n_samples=fast_samples
    )
    if ok:
        return True, f"no sidecar; {n_samp} samples: 0 UNKNOWN (content looks complete)"
    return False, f"no sidecar; {n_unk}/{n_samp} UNKNOWN (interrupted build)"


def _find_tables_in_dir(out_dir: Path) -> list[tuple[int, int]]:
    """Discover all endgame_N_M.wdl files and return sorted (nW, nB) pairs."""
    pat = re.compile(r"^endgame_(\d+)_(\d+)\.wdl$")
    tables = []
    for p in sorted(out_dir.glob("endgame_*.wdl")):
        m = pat.match(p.name)
        if m:
            tables.append((int(m.group(1)), int(m.group(2))))
    return sorted(tables)


def _report_verify(
    out_dir: Path,
    nW: int,
    nB: int,
    n_samples: int,
) -> bool:
    """Run full verification for one table and print a report. Returns True if OK."""
    p = _wdl_path(out_dir, nW, nB)
    label = f"({nW},{nB})"

    if not p.exists():
        logger.error("%s MISSING: %s", label, p)
        return False

    expected_bytes = _packed_table_bytes(nW, nB)
    actual_bytes = p.stat().st_size
    size_mb = actual_bytes / 1024 / 1024

    if actual_bytes != expected_bytes:
        logger.error(
            "%s SIZE MISMATCH: %d bytes on disk, expected %d  [%.1f MB]",
            label, actual_bytes, expected_bytes, size_mb,
        )
        return False

    logger.info("%s size OK (%d bytes, %.1f MB)", label, actual_bytes, size_mb)

    # Checksum check.
    sha_path = _sha256_path(out_dir, nW, nB)
    if sha_path.exists():
        try:
            stored = sha_path.read_text(encoding="ascii").strip()
            current = _compute_file_sha256(p)
            if stored == current:
                logger.info("%s checksum OK (%s)", label, stored[:16] + "…")
            else:
                logger.error(
                    "%s CHECKSUM MISMATCH  stored=%s  actual=%s",
                    label, stored[:16] + "…", current[:16] + "…",
                )
                return False
        except OSError as e:
            logger.warning("%s checksum unreadable: %s", label, e)
    else:
        logger.info("%s no .sha256 sidecar (will rely on content sampling)", label)

    # Content sampling.
    ts = _table_size(nW, nB)
    n_samp = min(n_samples, ts)
    ok, n_unk, n_samp, issues = _verify_table_content(
        out_dir, nW, nB, n_samples=n_samp
    )

    # WDL distribution of sample.
    counts = {WDL_WIN: 0, WDL_DRAW: 0, WDL_LOSS: 0, WDL_UNKNOWN: 0}
    rng = random.Random(0x4E4D4D)
    sample_ids = rng.sample(range(ts), n_samp)
    with open(p, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), expected_bytes, access=mmap.ACCESS_READ)
        try:
            for pid in sample_ids:
                counts[get_wdl(mm, pid)] += 1
        finally:
            mm.close()

    logger.info(
        "%s sample %d/%d positions — WIN:%d  DRAW:%d  LOSS:%d  UNKNOWN:%d",
        label, n_samp, ts,
        counts[WDL_WIN], counts[WDL_DRAW], counts[WDL_LOSS], counts[WDL_UNKNOWN],
    )

    if issues:
        for msg in issues:
            logger.error("%s %s", label, msg)
        return False

    logger.info("%s PASS", label)
    return True


# ── Build schedule ─────────────────────────────────────────────────────────────

_ALL_TABLES: list[tuple[int, int]] = [
    (nW, nB)
    for s in range(6, 12)
    for nW in range(3, s)
    for nB in [s - nW]
    if nB >= 3
]


def _sub_tables_needed(nW: int, nB: int) -> list[tuple[int, int]]:
    deps = []
    if nB - 1 >= 3:
        deps.append((nW, nB - 1))
    if nW - 1 >= 3:
        deps.append((nW - 1, nB))
    return deps


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build and verify NMM retrograde endgame WDL tables."
    )
    ap.add_argument(
        "--out-dir", default="data/endgame",
        help="Directory containing endgame_*.wdl files (default: data/endgame)",
    )
    ap.add_argument("--nW", type=int, help="White piece count")
    ap.add_argument("--nB", type=int, help="Black piece count")

    # Build modes.
    ap.add_argument(
        "--build-all", action="store_true",
        help="Build all tables in dependency order",
    )
    ap.add_argument(
        "--max-sum", type=int, default=11,
        help="Maximum nW+nB to build when using --build-all (default: 11)",
    )
    ap.add_argument(
        "--skip-existing", action="store_true",
        help=(
            "Skip tables that are already complete. Completeness is determined by "
            "checksum sidecar if present, otherwise by content sampling "
            "(catches interrupted pre-allocations at correct file size)."
        ),
    )

    # Verify modes.
    ap.add_argument(
        "--verify", action="store_true",
        help="Verify a single table (use with --nW/--nB). Checks size, checksum, content.",
    )
    ap.add_argument(
        "--verify-all", action="store_true",
        help="Verify all endgame_*.wdl tables found in --out-dir.",
    )
    ap.add_argument(
        "--samples", type=int, default=2000,
        help="Random positions to sample per table for UNKNOWN check (default: 2000)",
    )

    # Sidecar repair.
    ap.add_argument(
        "--fix-sidecars", action="store_true",
        help=(
            "Write missing .sha256 sidecars for tables that pass content verification. "
            "Does not modify .wdl files."
        ),
    )

    ap.add_argument("--quiet", action="store_true", help="Suppress per-pass logging")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    verbose = not args.quiet

    # ── Verify modes ────────────────────────────────────────────────────────────
    if args.verify or args.verify_all:
        if args.verify:
            if args.nW is None or args.nB is None:
                ap.error("--verify requires --nW and --nB")
            tables_to_check = [(args.nW, args.nB)]
        else:
            tables_to_check = _find_tables_in_dir(out_dir)
            if not tables_to_check:
                logger.error("No endgame_*.wdl files found in %s", out_dir)
                sys.exit(1)
            logger.info("Found %d table(s) in %s", len(tables_to_check), out_dir)

        passed = []
        failed = []
        for nW, nB in tables_to_check:
            ok = _report_verify(out_dir, nW, nB, args.samples)
            (passed if ok else failed).append((nW, nB))

        print()
        print(f"Results: {len(passed)} PASS  {len(failed)} FAIL")
        if failed:
            print("Failed tables:", ", ".join(f"({nW},{nB})" for nW, nB in failed))
            sys.exit(1)
        sys.exit(0)

    # ── Fix-sidecars mode ───────────────────────────────────────────────────────
    if args.fix_sidecars:
        tables = _find_tables_in_dir(out_dir)
        if not tables:
            logger.error("No endgame_*.wdl files found in %s", out_dir)
            sys.exit(1)
        wrote = 0
        skipped = 0
        failed = 0
        for nW, nB in tables:
            sha_path = _sha256_path(out_dir, nW, nB)
            if sha_path.exists():
                logger.info("(%d,%d) sidecar already exists — skipping", nW, nB)
                skipped += 1
                continue
            ok, n_unk, n_samp, issues = _verify_table_content(
                out_dir, nW, nB, n_samples=args.samples
            )
            if not ok:
                logger.error(
                    "(%d,%d) content check failed (%s) — NOT writing sidecar",
                    nW, nB, "; ".join(issues),
                )
                failed += 1
                continue
            digest = _write_checksum_sidecar(out_dir, nW, nB)
            logger.info("(%d,%d) wrote sidecar: %s", nW, nB, digest)
            wrote += 1
        print(f"\nSidecars written: {wrote}  skipped: {skipped}  failed: {failed}")
        sys.exit(0 if failed == 0 else 1)

    # ── Build modes ─────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.build_all:
        schedule = [
            (nW, nB) for (nW, nB) in _ALL_TABLES if nW + nB <= args.max_sum
        ]
    elif args.nW is not None and args.nB is not None:
        if args.nW < 3 or args.nB < 3:
            ap.error("--nW and --nB must each be ≥ 3")
        schedule = [(args.nW, args.nB)]
    else:
        ap.error("Specify --build-all, --verify, --verify-all, --fix-sidecars, "
                 "or both --nW and --nB")

    loaded: dict[tuple[int, int], bytes] = {}

    for idx, (nW, nB) in enumerate(schedule):
        wdl_path = _wdl_path(out_dir, nW, nB)

        if args.skip_existing:
            complete, reason = _is_table_complete(out_dir, nW, nB, fast_samples=256)
            if complete:
                logger.info("(%d,%d) complete — skipping (%s).", nW, nB, reason)
                if not _sha256_path(out_dir, nW, nB).exists():
                    digest = _write_checksum_sidecar(out_dir, nW, nB)
                    logger.info("(%d,%d) wrote missing checksum sidecar: %s", nW, nB, digest)
                data = _load_table(out_dir, nW, nB)
                if data is not None:
                    loaded[(nW, nB)] = data
                continue
            else:
                logger.info("(%d,%d) not complete — rebuilding (%s).", nW, nB, reason)

        sub_tables: dict[tuple[int, int], bytes] = {}
        for dep_nw, dep_nb in _sub_tables_needed(nW, nB):
            if (dep_nw, dep_nb) in loaded:
                sub_tables[(dep_nw, dep_nb)] = loaded[(dep_nw, dep_nb)]
            else:
                dep_ok, dep_msg = _table_status(out_dir, dep_nw, dep_nb)
                if not dep_ok:
                    logger.warning(
                        "(%d,%d) Sub-table (%d,%d) incomplete or missing — %s. "
                        "Capture outcomes into that table will be treated as UNKNOWN.",
                        nW, nB, dep_nw, dep_nb, dep_msg,
                    )
                    continue
                data = _load_table(out_dir, dep_nw, dep_nb)
                if data is not None:
                    sub_tables[(dep_nw, dep_nb)] = data
                    loaded[(dep_nw, dep_nb)] = data
                    if not _sha256_path(out_dir, dep_nw, dep_nb).exists():
                        digest = _write_checksum_sidecar(out_dir, dep_nw, dep_nb)
                        logger.info(
                            "(%d,%d) Wrote missing checksum sidecar for dependency: %s",
                            dep_nw, dep_nb, digest,
                        )

        logger.info(
            "Building (%d,%d): %d positions, %.1f MB table",
            nW, nB,
            _table_size(nW, nB),
            _packed_table_bytes(nW, nB) / 1024 / 1024,
        )
        solve_table(nW, nB, sub_tables, wdl_path, verbose=verbose)
        digest = _write_checksum_sidecar(out_dir, nW, nB)
        logger.info("Wrote %s (%d bytes)", wdl_path, wdl_path.stat().st_size)
        logger.info("(%d,%d) SHA-256 %s", nW, nB, digest)
        data = _load_table(out_dir, nW, nB)
        if data is not None:
            loaded[(nW, nB)] = data

        remaining_schedule = set(schedule[idx + 1:])
        still_needed = set()
        for rnW, rnB in remaining_schedule:
            for dep in _sub_tables_needed(rnW, rnB):
                still_needed.add(dep)
        for key in list(loaded.keys()):
            if key not in remaining_schedule and key not in still_needed:
                del loaded[key]


if __name__ == "__main__":
    main()
