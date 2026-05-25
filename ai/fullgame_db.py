"""ai/fullgame_db.py — Read-only query interface for the full-game position DB.

Companion to ``tools/build_fullgame_db.py``.  The database is built offline
(potentially over many hours) and consulted at query time by the GameAI.

Design contract
---------------
The DB is *optional*.  When ``FullGameDB.is_available()`` returns False the
GameAI falls back to its normal negamax search.  When the DB IS available but
the queried position is not present, ``query()`` returns ``None`` (also a
fallback signal).  Only when an exact canonical hit is found does the DB
override the search — and even then GameAI may blend the result with the
heuristic via ``score_delta()``.

Two on-disk formats are supported
-----------------------------------
SQLite  (legacy / build intermediate):
    positions(key BLOB PK, outcome INT, depth INT, best_move TEXT,
              trajectories TEXT, samples INT)

Binary (preferred for production):
    32-byte file header + sorted 32-byte fixed-length records.
    Format auto-detected by magic bytes ``b"NMM_FGDB"`` at offset 0.
    Binary search gives O(log N) lookup; the whole file is mmap'd read-only.

`key` is a packed 9-byte canonical position id.  We rely on the existing
``ai.board_symmetry`` D4 helpers for canonicalization, so the DB only stores
one representative per equivalence class.

Public surface
--------------
    FullGameDB(path)
        .is_available()
        .query(board: BoardState) -> FullGameResult | None
        .score_delta(board, current_color) -> dict[str, float]
        .best_move(board) -> str | None        # actual-board notation
        .close()
"""

from __future__ import annotations

import logging
import mmap
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from game.board import BoardState, POSITIONS
from .board_symmetry import (
    SYM_INVERSE,
    canonical_board_str,
    transform_notation,
)

logger = logging.getLogger(__name__)

# ── Binary format constants ──────────────────────────────────────────────────

HEADER_MAGIC = b"NMM_FGDB"
FORMAT_VERSION = 1
HEADER_SIZE = 32    # bytes
RECORD_SIZE = 32    # bytes
KEY_SIZE = 9        # bytes

_HEADER_FMT = "<8sHI18x"    # magic(8) + version(2) + record_count(4) + pad(18)
_RECORD_FMT = "<9sBHIIIII"  # key(9) + outcome(1) + depth(2) + best_move(4) + 4×child(4)

assert struct.calcsize(_HEADER_FMT) == HEADER_SIZE
assert struct.calcsize(_RECORD_FMT) == RECORD_SIZE

# Position index tables for move packing
_POS_TO_IDX: dict[str, int] = {p: i for i, p in enumerate(POSITIONS)}
_IDX_TO_POS: list[str] = list(POSITIONS)

_NO_POS = 31            # sentinel: no from-square (placement) or no capture
_EMPTY_MOVE = 0xFFFFFFFF  # sentinel: empty child slot

# Packed move layout (32-bit uint):
#   bits  0-4 : from_idx  (0-23 = POSITIONS index; 31 = _NO_POS for placements)
#   bits  5-9 : to_idx    (0-23)
#   bits 10-14: cap_idx   (0-23; 31 = _NO_POS for non-captures)
#   bits 15-16: flag      (0=N, 1=W wins for STM, 2=L loses for STM)

_FLAG_ENCODE = {"N": 0, "W": 1, "L": 2}
_FLAG_DECODE = {0: "N", 1: "W", 2: "L"}

# Outcome byte encoding in binary records:
#   0 = unknown (NULL in SQLite)
#   1 = W win   (+1)
#   2 = B win   (-1)
#   3 = draw    (0)
_OUTCOME_ENCODE = {None: 0, 1: 1, -1: 2, 0: 3}
_OUTCOME_DECODE = {0: None, 1: 1, 2: -1, 3: 0}


# ── Binary move packing helpers ──────────────────────────────────────────────

def _pack_move(notation: Optional[str], flag: str = "N") -> int:
    """Pack a move notation + outcome flag into a 32-bit uint.

    Notation forms::

        None        → _EMPTY_MOVE (0xFFFFFFFF)
        "d2"        → placement, no capture
        "d2xa4"     → placement + capture
        "a7-a4"     → movement, no capture
        "a7-a4xb4"  → movement + capture
    """
    if notation is None:
        return _EMPTY_MOVE

    if "x" in notation:
        move_part, cap_str = notation.split("x", 1)
        cap_idx = _POS_TO_IDX[cap_str]
    else:
        move_part = notation
        cap_idx = _NO_POS

    if "-" in move_part:
        from_str, to_str = move_part.split("-", 1)
        from_idx = _POS_TO_IDX[from_str]
    else:
        to_str = move_part
        from_idx = _NO_POS

    to_idx = _POS_TO_IDX[to_str]
    flag_bits = _FLAG_ENCODE.get(flag, 0)
    return from_idx | (to_idx << 5) | (cap_idx << 10) | (flag_bits << 15)


def _unpack_move(packed: int) -> tuple[Optional[str], str]:
    """Unpack a 32-bit uint into (notation | None, flag_str)."""
    if packed == _EMPTY_MOVE:
        return None, "N"

    from_idx = packed & 0x1F
    to_idx = (packed >> 5) & 0x1F
    cap_idx = (packed >> 10) & 0x1F
    flag_bits = (packed >> 15) & 0x3

    to_pos = _IDX_TO_POS[to_idx]
    if from_idx == _NO_POS:
        notation = to_pos
    else:
        notation = f"{_IDX_TO_POS[from_idx]}-{to_pos}"
    if cap_idx != _NO_POS:
        notation += f"x{_IDX_TO_POS[cap_idx]}"

    return notation, _FLAG_DECODE.get(flag_bits, "N")


# ── Same key encoding as the builder ────────────────────────────────────────

_PIECE_BITS = {".": 0b00, "W": 0b01, "B": 0b10}


def _encode_canonical(board24: str, turn: str, placed_w: int, placed_b: int) -> bytes:
    val = 0
    for i, ch in enumerate(board24):
        val |= _PIECE_BITS[ch] << (i * 2)
    return val.to_bytes(6, "little") + bytes(
        (0 if turn == "W" else 1, placed_w & 0xFF, placed_b & 0xFF)
    )


def _unpack_trajectories(blob: str) -> list[tuple[str, bytes, str]]:
    if not blob:
        return []
    out = []
    for part in blob.split("|"):
        try:
            n, ck, f = part.rsplit(":", 2)
        except ValueError:
            continue
        try:
            out.append((n, bytes.fromhex(ck), f))
        except ValueError:
            continue
    return out


def export_to_binary(sqlite_path: str | Path, binary_path: str | Path) -> int:
    """Export a fullgame SQLite DB to the sorted binary format.

    Reads *sqlite_path* (must exist) in key-ascending order and writes
    *binary_path*.  The SQLite file is not modified.  Returns the number of
    records written.

    This is the only place the binary format is *written*; reading is done
    through ``FullGameDB``.
    """
    sqlite_path = Path(sqlite_path)
    binary_path = Path(binary_path)

    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT key, outcome, depth, best_move, trajectories "
        "FROM positions ORDER BY key ASC"
    ).fetchall()
    conn.close()

    records: list[bytes] = []
    for key, outcome, depth, best_move, traj_blob in rows:
        outcome_byte = _OUTCOME_ENCODE.get(outcome, 0)
        depth_val = 0xFFFF if depth is None else min(int(depth), 0xFFFE)
        bm_packed = _pack_move(best_move, "N")

        edges = _unpack_trajectories(traj_blob or "")
        informative = [e for e in edges if e[2] in ("W", "L")]
        neutral = [e for e in edges if e[2] == "N"]
        top4 = (informative + neutral)[:4]

        children = [_pack_move(n, f) for n, _, f in top4]
        while len(children) < 4:
            children.append(_EMPTY_MOVE)

        records.append(struct.pack(
            _RECORD_FMT,
            key, outcome_byte, depth_val, bm_packed,
            children[0], children[1], children[2], children[3],
        ))

    record_count = len(records)
    header = struct.pack(_HEADER_FMT, HEADER_MAGIC, FORMAT_VERSION, record_count)

    with open(binary_path, "wb") as fh:
        fh.write(header)
        for rec in records:
            fh.write(rec)

    logger.info("export_to_binary: wrote %d records → %s", record_count, binary_path)
    return record_count


@dataclass
class FullGameResult:
    """One row from the position table, with the symmetry index used to
    transform stored canonical move notations back into the actual board
    orientation the caller asked about."""

    outcome: Optional[int]          # 1=W win, -1=B win, 0=draw, None=unknown
    depth: Optional[int]            # plies to result, or None
    best_move_canonical: Optional[str]
    sym_idx: int                    # transform that maps actual board → canonical
    trajectories: list[tuple[str, bytes, str]]  # (canonical_notation, child_key, flag)


class FullGameDB:
    """Read-only wrapper around the position database (SQLite or binary)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: Optional[sqlite3.Connection] = None
        self._binary: bool = False
        self._mmap: Optional[mmap.mmap] = None
        self._file_handle = None
        self._record_count: int = 0

        if not self.path.exists():
            return

        try:
            with open(self.path, "rb") as fh:
                magic = fh.read(len(HEADER_MAGIC))
        except OSError as exc:
            logger.warning("FullGameDB: cannot read %s — %s", self.path, exc)
            return

        if magic == HEADER_MAGIC:
            self._open_binary()
        else:
            self._open_sqlite()

    def _open_binary(self) -> None:
        try:
            self._file_handle = open(self.path, "rb")
            raw = self._file_handle.read(HEADER_SIZE)
            if len(raw) < HEADER_SIZE:
                raise ValueError("header too short")
            magic, version, record_count = struct.unpack(_HEADER_FMT, raw)
            if magic != HEADER_MAGIC:
                raise ValueError(f"bad magic: {magic!r}")
            if version != FORMAT_VERSION:
                raise ValueError(f"unsupported version: {version}")
            self._record_count = record_count
            self._binary = True
            if record_count == 0:
                logger.warning("FullGameDB: binary %s has 0 records.", self.path)
                return  # available but empty; _mmap stays None
            self._mmap = mmap.mmap(
                self._file_handle.fileno(), 0, access=mmap.ACCESS_READ
            )
            logger.info(
                "FullGameDB: opened binary %s — %d records.", self.path, record_count
            )
        except Exception as exc:
            logger.warning("FullGameDB: could not open binary %s — %s", self.path, exc)
            if self._file_handle is not None:
                self._file_handle.close()
                self._file_handle = None
            self._binary = False

    def _open_sqlite(self) -> None:
        try:
            self._conn = sqlite3.connect(
                f"file:{self.path}?mode=ro", uri=True, check_same_thread=False,
            )
            self._conn.execute("SELECT key FROM positions LIMIT 1").fetchone()
            logger.info("FullGameDB: opened SQLite %s (read-only).", self.path)
        except sqlite3.Error as exc:
            logger.warning("FullGameDB: could not open SQLite %s — %s", self.path, exc)
            self._conn = None

    def is_available(self) -> bool:
        return self._conn is not None or self._binary

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
        self._binary = False
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Binary reader ────────────────────────────────────────────────────────

    def _query_binary(self, key: bytes) -> Optional[FullGameResult]:
        """Binary search the mmap'd sorted records for `key`."""
        if self._mmap is None or self._record_count == 0:
            return None

        lo, hi = 0, self._record_count - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            offset = HEADER_SIZE + mid * RECORD_SIZE
            rec_key = self._mmap[offset : offset + KEY_SIZE]
            if rec_key == key:
                return self._decode_record(offset)
            elif rec_key < key:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    def _decode_record(self, offset: int) -> FullGameResult:
        raw = self._mmap[offset : offset + RECORD_SIZE]
        _key, outcome_byte, depth_val, bm_packed, c0, c1, c2, c3 = struct.unpack(
            _RECORD_FMT, raw
        )

        outcome = _OUTCOME_DECODE.get(outcome_byte)
        depth = None if depth_val == 0xFFFF else depth_val
        bm_notation, _ = _unpack_move(bm_packed)

        trajectories: list[tuple[str, bytes, str]] = []
        for packed in (c0, c1, c2, c3):
            notation, flag = _unpack_move(packed)
            if notation is not None:
                trajectories.append((notation, b"", flag))

        return FullGameResult(
            outcome=outcome,
            depth=depth,
            best_move_canonical=bm_notation,
            sym_idx=0,  # caller fills in the real sym_idx after binary search
            trajectories=trajectories,
        )

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self, board: BoardState) -> Optional[FullGameResult]:
        """Look up the canonical position for `board`.  Returns None on miss."""
        fen = board.to_fen_string()
        board24, turn, pw, pb = fen.split("|")
        canon, sym = canonical_board_str(board24)
        key = _encode_canonical(canon, turn, int(pw), int(pb))

        if self._binary:
            result = self._query_binary(key)
            if result is not None:
                result.sym_idx = sym
            return result

        if self._conn is None:
            return None

        row = self._conn.execute(
            "SELECT outcome, depth, best_move, trajectories FROM positions WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        outcome, depth, best_move, traj_blob = row
        return FullGameResult(
            outcome=outcome,
            depth=depth,
            best_move_canonical=best_move,
            sym_idx=sym,
            trajectories=_unpack_trajectories(traj_blob or ""),
        )

    # ── Convenience helpers used by GameAI ──────────────────────────────────

    def best_move(self, board: BoardState) -> Optional[str]:
        """Return the best move notation in the actual board's orientation."""
        result = self.query(board)
        if result is None or not result.best_move_canonical:
            return None
        inv = SYM_INVERSE[result.sym_idx]
        return transform_notation(result.best_move_canonical, inv)

    def score_delta(self, board: BoardState, current_color: str) -> dict[str, float]:
        """Return a per-move score delta in [-0.5, +0.5] compatible with the
        TrajectoryDB / EndgameDB hint interface used by GameAI.

        Mapping:
            move flagged 'W' (winning for side-to-move) → +0.5
            move flagged 'L' (losing)                   → -0.5
            move flagged 'N' (neutral / unknown)        →  0.0

        Returns {} on miss so GameAI can fall back cleanly.
        """
        result = self.query(board)
        if result is None or not result.trajectories:
            return {}
        inv = SYM_INVERSE[result.sym_idx]
        out: dict[str, float] = {}
        for canon_notation, _child_key, flag in result.trajectories:
            actual = transform_notation(canon_notation, inv)
            if actual is None:
                continue
            if flag == "W":
                out[actual] = 0.5
            elif flag == "L":
                out[actual] = -0.5
            else:
                out[actual] = 0.0
        return out

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        if self._binary:
            return {"available": 1, "positions": self._record_count, "resolved": -1}
        if self._conn is None:
            return {"available": 0}
        total = self._conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        resolved = self._conn.execute(
            "SELECT COUNT(*) FROM positions WHERE outcome IS NOT NULL"
        ).fetchone()[0]
        return {"available": 1, "positions": total, "resolved": resolved}
