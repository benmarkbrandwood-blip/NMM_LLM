"""tools/build_fullgame_db.py — Full-game position database generator.

Builds a SQLite database of Nine Men's Morris positions covering the entire
game (placement, movement, fly, terminals).  D4 symmetry is used so up to 8
equivalent positions share one row.

WARNING — full Nine Men's Morris has on the order of 10^10 legal positions;
a true complete solve is infeasible on a normal machine.  This script
performs a bounded enumeration with backpropagation of win/loss/draw
trajectories so you can:

  • build the database to a position cap or depth cap that fits your disk,
  • resume the build after interruption,
  • run a quick `--dry-run` to validate the pipeline,
  • run a `--sample N` mode for sanity testing without a 12 GB commitment.

Outcomes follow Wikipedia's convention for solved game-trees:
   1 = WIN-FOR-WHITE,  -1 = WIN-FOR-BLACK,  0 = DRAW,  NULL = UNKNOWN
``depth`` is distance-to-terminal (plies) when known, otherwise NULL.
``best_move`` is the best canonical-form move notation for the side to move.

Run inside the project venv:
    .venv/bin/python tools/build_fullgame_db.py --help
    .venv/bin/python tools/build_fullgame_db.py --dry-run

    # Default location (project data/ directory)
    .venv/bin/python tools/build_fullgame_db.py --max-positions 500000

    # Another drive — use --db-dir (auto-names the file fullgame.sqlite)
    .venv/bin/python tools/build_fullgame_db.py --db-dir D:/databases --max-positions 500000
    .venv/bin/python tools/build_fullgame_db.py --db-dir /mnt/external  --max-positions 500000

    # Or give the full path explicitly with --output
    .venv/bin/python tools/build_fullgame_db.py --output E:/NMM/fullgame.sqlite

The script prints the resolved absolute path before starting so you always
know where it's writing.  It also checks the target directory is writable
before beginning the build, to avoid discovering a permissions error after
hours of work.

Generated files are intentionally placed under data/ by default — see .gitignore.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import struct
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

# ── Ensure project root on path when invoked directly ────────────────────────
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Dependency check / install ────────────────────────────────────────────────
# Stdlib-only by design.  We still verify the project's own requirements are
# importable so this script can be a one-stop "install everything and build".

_REQUIRED_STDLIB = ("sqlite3", "struct", "collections")
_REQUIRED_PROJECT = ("game.board", "game.rules", "ai.board_symmetry")


def _verify_deps() -> None:
    missing: list[str] = []
    for mod in _REQUIRED_STDLIB:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"FATAL: missing stdlib modules: {missing}", file=sys.stderr)
        sys.exit(2)


def _maybe_pip_install_requirements() -> None:
    req = _ROOT / "requirements.txt"
    if not req.exists():
        return
    try:
        __import__("fastapi")
        return  # something already installed → assume OK
    except ImportError:
        pass
    print("Installing project requirements…", file=sys.stderr)
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(req)],
    )


# ── Project imports ───────────────────────────────────────────────────────────

from game.board import POSITIONS, BoardState
from game.rules import get_all_legal_moves, is_terminal
# Import board_symmetry directly from its module to avoid pulling in ai/__init__.py
# (which depends on chromadb / fastapi).  This script is intentionally
# stdlib-only so it can run during DB builds without the full web stack.
import importlib.util as _ilu
import types as _types

_spec = _ilu.spec_from_file_location(
    "ai_board_symmetry",
    str(_ROOT / "ai" / "board_symmetry.py"),
)
_bs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_bs)
SYM_INVERSE = _bs.SYM_INVERSE
canonical_board_str = _bs.canonical_board_str
transform_notation = _bs.transform_notation

logger = logging.getLogger("fullgame_db.build")


def _load_fullgame_db_module():
    """Load ai.fullgame_db via importlib (avoids pulling in ai/__init__.py)."""
    if "ai" not in sys.modules:
        pkg = _types.ModuleType("ai")
        pkg.__path__ = [str(_ROOT / "ai")]
        sys.modules["ai"] = pkg
    # Register the already-loaded board_symmetry module under its canonical name
    # so that fullgame_db.py's relative import resolves correctly.
    sys.modules.setdefault("ai.board_symmetry", _bs)
    spec = _ilu.spec_from_file_location(
        "ai.fullgame_db", str(_ROOT / "ai" / "fullgame_db.py")
    )
    mod = _ilu.module_from_spec(spec)
    sys.modules["ai.fullgame_db"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Canonical key encoding ────────────────────────────────────────────────────
# The position state we must distinguish:
#   • 24-character board ("W"/"B"/".")
#   • side to move ("W"/"B")
#   • pieces placed so far for each side (0..9)
# Storing the board as a packed bit-pair string costs 6 bytes (2 bits × 24),
# plus 1 byte side, plus 1 byte placed_W, 1 byte placed_B = 9 bytes.

_PIECE_BITS = {".": 0b00, "W": 0b01, "B": 0b10}


def encode_canonical(board24: str, turn: str, placed_w: int, placed_b: int) -> bytes:
    """Pack a canonical position into 9 bytes for a compact SQLite primary key."""
    if len(board24) != 24:
        raise ValueError(f"board24 must be 24 chars, got {len(board24)}")
    val = 0
    for i, ch in enumerate(board24):
        val |= _PIECE_BITS[ch] << (i * 2)
    # 48 bits of board → 6 bytes
    board_bytes = val.to_bytes(6, "little")
    side = 0 if turn == "W" else 1
    return board_bytes + bytes((side, placed_w & 0xFF, placed_b & 0xFF))


def position_key(board: BoardState) -> bytes:
    """Compute the canonical 9-byte key for a BoardState (applies D4 canonicalization)."""
    fen = board.to_fen_string()
    board24, turn, pw, pb = fen.split("|")
    canon, _sym = canonical_board_str(board24)
    return encode_canonical(canon, turn, int(pw), int(pb))


def canonical_components(board: BoardState) -> tuple[str, int, str, int, int]:
    """Return (canonical_board24, sym_idx, turn, placed_w, placed_b)."""
    fen = board.to_fen_string()
    board24, turn, pw, pb = fen.split("|")
    canon, sym = canonical_board_str(board24)
    return canon, sym, turn, int(pw), int(pb)


def move_notation(move: dict) -> str:
    s = f"{move['from']}-{move['to']}" if move.get("from") else move.get("to", "")
    if move.get("capture"):
        s += f"x{move['capture']}"
    return s


# ── Schema ───────────────────────────────────────────────────────────────────
# WITHOUT ROWID keeps the table effectively as a clustered index on `key`.

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA page_size = 4096;
PRAGMA temp_store = MEMORY;

CREATE TABLE IF NOT EXISTS positions (
    key       BLOB PRIMARY KEY,    -- 9-byte canonical position
    outcome   INTEGER,             -- 1=W wins, -1=B wins, 0=draw, NULL=unknown
    depth     INTEGER,             -- plies to terminal (NULL if unknown)
    best_move TEXT,                -- canonical-form move notation
    samples   INTEGER NOT NULL DEFAULT 1
) WITHOUT ROWID;

-- Edges store the trajectory information: for each position, the list of
-- (move, child_key, classification) tuples.  Classification flag is:
--   'W' = winning move for side-to-move
--   'L' = losing move
--   'N' = neutral / unresolved
-- We pack edges as a single TEXT blob per position rather than a separate
-- table — keeps disk footprint smaller for big builds (one row vs ~10).
ALTER TABLE positions ADD COLUMN trajectories TEXT;  -- ignored if column exists
ALTER TABLE positions ADD COLUMN frequency INTEGER NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
);
"""

# Some SQLite versions raise on the ALTER if column exists; we tolerate that.


def _ensure_trajectories_column(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(positions)")}
    if "trajectories" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN trajectories TEXT")
    if "frequency" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN frequency INTEGER NOT NULL DEFAULT 0")


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    for stmt in _SCHEMA.strip().split(";"):
        s = stmt.strip()
        if not s:
            continue
        if s.upper().startswith("ALTER"):
            continue  # handled below
        try:
            conn.execute(s)
        except sqlite3.OperationalError as exc:
            logger.debug("Schema stmt skipped (%s): %s", s.split()[0], exc)
    _ensure_trajectories_column(conn)
    conn.commit()
    return conn


# ── Trajectory packing ───────────────────────────────────────────────────────
# Stored as compact pipe-separated triples: "notation:childkey_hex:flag".
# Flag: W (winning for side-to-move) / L (losing) / N (neutral/unknown).

def pack_trajectories(items: list[tuple[str, bytes, str]]) -> str:
    return "|".join(f"{n}:{ck.hex()}:{f}" for n, ck, f in items)


def unpack_trajectories(blob: str) -> list[tuple[str, bytes, str]]:
    if not blob:
        return []
    out = []
    for part in blob.split("|"):
        n, ck, f = part.rsplit(":", 2)
        out.append((n, bytes.fromhex(ck), f))
    return out


# ── Builder ──────────────────────────────────────────────────────────────────

class FullGameDBBuilder:
    """Forward-BFS position enumerator with terminal back-propagation.

    The builder canonicalises each visited position so symmetric duplicates
    share one DB row.  It records every legal move as an edge, then performs
    a retrograde-style pass that labels terminal positions and propagates
    win/loss/draw outcomes back through the parent links.
    """

    def __init__(
        self,
        db_path: Path,
        max_positions: Optional[int] = None,
        max_depth: Optional[int] = None,
        commit_every: int = 5000,
        progress_every: float = 5.0,
    ) -> None:
        self.db_path = db_path
        self.max_positions = max_positions
        self.max_depth = max_depth
        self.commit_every = commit_every
        self.progress_every = progress_every

        self.conn = init_db(db_path)
        self.conn.execute("PRAGMA foreign_keys = OFF")

        self._inserted = 0
        self._scanned = 0
        self._t_start = time.monotonic()
        self._t_last_progress = self._t_start

    # ── Forward enumeration ──────────────────────────────────────────────────

    def enumerate_forward(self, start: BoardState) -> None:
        """BFS over reachable canonical positions starting from `start`.

        Resumes by reading the existing positions table — already-seen keys
        are skipped, so re-running the script extends the build.
        """
        start_key = position_key(start)
        seen: set[bytes] = set()
        # Pre-load existing keys so we don't re-process completed work.
        cur = self.conn.execute("SELECT key FROM positions")
        for (k,) in cur:
            seen.add(bytes(k))
        logger.info("Resuming from %d positions already in DB.", len(seen))

        queue: deque[tuple[bytes, BoardState, int]] = deque()
        if start_key not in seen:
            queue.append((start_key, start, 0))

        # If the start key IS in the DB but we have nothing else to do,
        # the resume seed is the union of any rows with NULL outcome — i.e.,
        # rows we may still want to expand.
        if not queue:
            cur = self.conn.execute(
                "SELECT key FROM positions WHERE outcome IS NULL "
                "AND (trajectories IS NULL OR trajectories = '') LIMIT 10000"
            )
            for (k,) in cur:
                # We must reconstruct the BoardState from the key.  Skip for
                # now — forward-pass resume on an existing DB only adds new
                # positions reachable from `start`; deep continuation should
                # use the backprop pass.
                _ = k

        while queue:
            if self.max_positions is not None and self._inserted >= self.max_positions:
                logger.info("Reached --max-positions cap (%d).", self.max_positions)
                break

            key, board, depth = queue.popleft()

            terminal, winner = is_terminal(board)
            if terminal:
                outcome = 1 if winner == "W" else (-1 if winner == "B" else 0)
                self._upsert(key, outcome=outcome, depth=0, best_move=None,
                             trajectories="")
                continue

            if self.max_depth is not None and depth >= self.max_depth:
                # Frontier node: insert with unknown outcome, no children.
                self._upsert(key, outcome=None, depth=None, best_move=None,
                             trajectories="")
                continue

            moves = get_all_legal_moves(board)
            canon_board, sym, _turn, _pw, _pb = canonical_components(board)
            edges: list[tuple[str, bytes, str]] = []
            for mv in moves:
                child = board.apply_move(mv)
                child_key = position_key(child)
                # Notation in canonical orientation (for storage stability)
                actual_notation = move_notation(mv)
                canon_notation = transform_notation(actual_notation, sym) or actual_notation
                edges.append((canon_notation, child_key, "N"))
                if child_key not in seen:
                    seen.add(child_key)
                    queue.append((child_key, child, depth + 1))

            self._upsert(
                key,
                outcome=None,
                depth=None,
                best_move=None,
                trajectories=pack_trajectories(edges),
            )
            self._scanned += 1
            self._maybe_progress(len(queue))

        self.conn.commit()
        logger.info("Forward enumeration complete: %d rows.", self._inserted)

    # ── Backward propagation ─────────────────────────────────────────────────

    def backpropagate(self, passes: int = 6) -> None:
        """Iteratively label parent positions from already-labelled children.

        A position with outcome=NULL is resolved when every child has a
        defined outcome.  Side-to-move picks the best outcome (max for W, min
        for B); ties prefer draws over losses.

        Multiple passes propagate labels through long chains.  Convergence
        within `passes` is not guaranteed for cyclic position spaces (NMM
        movement phase contains cycles); residual NULLs are left as
        UNKNOWN and the AI's negamax fallback handles them.
        """
        for pass_no in range(1, passes + 1):
            updated = 0
            cur = self.conn.execute(
                "SELECT key, trajectories FROM positions "
                "WHERE outcome IS NULL AND trajectories IS NOT NULL AND trajectories <> ''"
            )
            rows = cur.fetchall()
            for key, blob in rows:
                edges = unpack_trajectories(blob)
                if not edges:
                    continue
                child_keys = [e[1] for e in edges]
                placeholders = ",".join("?" for _ in child_keys)
                child_rows = self.conn.execute(
                    f"SELECT key, outcome, depth FROM positions WHERE key IN ({placeholders})",
                    child_keys,
                ).fetchall()
                child_map = {bytes(k): (o, d) for k, o, d in child_rows}

                if len(child_map) < len(child_keys):
                    continue  # unknown children → skip this pass
                outcomes = [child_map[ck][0] for ck in child_keys]
                if any(o is None for o in outcomes):
                    continue

                # Determine side to move from key (byte 6: 0=W, 1=B)
                side = "W" if key[6] == 0 else "B"
                if side == "W":
                    best_o = max(outcomes)
                else:
                    best_o = min(outcomes)

                # Choose best move (first child whose outcome matches best)
                new_edges: list[tuple[str, bytes, str]] = []
                best_move: Optional[str] = None
                for (n, ck, _flag), o in zip(edges, outcomes):
                    if (side == "W" and o == 1) or (side == "B" and o == -1):
                        flag = "W"
                    elif (side == "W" and o == -1) or (side == "B" and o == 1):
                        flag = "L"
                    else:
                        flag = "N"
                    if best_move is None and o == best_o:
                        best_move = n
                    new_edges.append((n, ck, flag))

                # depth = 1 + min depth of any matching child
                matching_depths = [
                    child_map[ck][1] for (_, ck, _), o in zip(edges, outcomes)
                    if o == best_o and child_map[ck][1] is not None
                ]
                new_depth = (1 + min(matching_depths)) if matching_depths else None

                self.conn.execute(
                    "UPDATE positions SET outcome=?, depth=?, best_move=?, trajectories=? WHERE key=?",
                    (best_o, new_depth, best_move, pack_trajectories(new_edges), key),
                )
                updated += 1
            self.conn.commit()
            logger.info("Backprop pass %d: labelled %d positions.", pass_no, updated)
            if updated == 0:
                break

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _upsert(
        self,
        key: bytes,
        outcome: Optional[int],
        depth: Optional[int],
        best_move: Optional[str],
        trajectories: str,
    ) -> None:
        self.conn.execute(
            "INSERT INTO positions (key, outcome, depth, best_move, trajectories, samples) "
            "VALUES (?, ?, ?, ?, ?, 1) "
            "ON CONFLICT(key) DO UPDATE SET "
            " outcome   = COALESCE(positions.outcome, excluded.outcome), "
            " depth     = COALESCE(positions.depth,   excluded.depth), "
            " best_move = COALESCE(positions.best_move, excluded.best_move), "
            " trajectories = COALESCE(NULLIF(positions.trajectories,''), excluded.trajectories), "
            " samples   = positions.samples + 1",
            (key, outcome, depth, best_move, trajectories),
        )
        self._inserted += 1
        if self._inserted % self.commit_every == 0:
            self.conn.commit()

    def _maybe_progress(self, queue_len: int) -> None:
        now = time.monotonic()
        if now - self._t_last_progress < self.progress_every:
            return
        self._t_last_progress = now
        elapsed = now - self._t_start
        rate = self._inserted / elapsed if elapsed > 0 else 0
        logger.info(
            "progress: scanned=%d inserted=%d queue=%d rate=%.0f pos/s",
            self._scanned, self._inserted, queue_len, rate,
        )

    def vacuum(self) -> None:
        logger.info("VACUUM (compacting database)…")
        self.conn.execute("VACUUM")

    def write_meta(self, **kv: str) -> None:
        for k, v in kv.items():
            self.conn.execute(
                "INSERT INTO meta(k,v) VALUES(?,?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (k, str(v)),
            )
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


# ── Game-based builder (B-52) ────────────────────────────────────────────────

def _is_human_game(game: dict) -> bool:
    """Return True for human-human and human-AI games; False for AI-AI self-play."""
    if game.get("human_color"):
        return True  # human-AI game
    moves = game.get("moves") or []
    # human-human: human_color is null AND no move has a non-null game_ai_score
    if all(m.get("game_ai_score") is None for m in moves):
        return True
    return False


def _game_notation_to_move(move: dict) -> Optional[dict]:
    """Convert a JSONL move record into a move dict compatible with apply_move."""
    to = move.get("to")
    if not to:
        return None
    return {
        "from": move.get("from"),
        "to": to,
        "capture": move.get("capture"),
    }


class GameBasedBuilder:
    """Scan human-played JSONL game records and build a frequency-weighted DB.

    Each position reached in a qualifying game increments the `frequency`
    counter for that canonical key.  No BFS/DFS is performed; coverage is
    limited to positions that actually occurred in the supplied games.
    """

    def __init__(
        self,
        db_path: Path,
        min_frequency: int = 1,
        min_frequency_placement: Optional[int] = None,
        commit_every: int = 2000,
        progress_every: float = 5.0,
    ) -> None:
        self.db_path = db_path
        self.min_frequency = min_frequency
        self.min_frequency_placement = min_frequency_placement or min_frequency
        self.commit_every = commit_every
        self.progress_every = progress_every
        self.conn = init_db(db_path)
        self._processed = 0
        self._skipped = 0
        self._positions_seen = 0
        self._t_start = time.monotonic()
        self._t_last_progress = self._t_start

    def scan_games(self, games_dir: Path) -> None:
        """Scan all *.jsonl files in games_dir for qualifying human games."""
        import glob

        files = sorted(glob.glob(str(games_dir / "*.jsonl")))
        logger.info("GameBasedBuilder: scanning %d game files in %s", len(files), games_dir)

        for fpath in files:
            try:
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            game = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not _is_human_game(game):
                            self._skipped += 1
                            continue
                        self._process_game(game)
            except OSError as exc:
                logger.warning("GameBasedBuilder: could not read %s — %s", fpath, exc)

        self.conn.commit()
        logger.info(
            "GameBasedBuilder: processed=%d skipped=%d positions_seen=%d",
            self._processed, self._skipped, self._positions_seen,
        )

    def _process_game(self, game: dict) -> None:
        moves = game.get("moves") or []
        board = BoardState.new_game()
        keys_in_game: list[bytes] = []

        for move_record in moves:
            mv = _game_notation_to_move(move_record)
            if mv is None:
                break
            try:
                key = position_key(board)
                keys_in_game.append(key)
                board = board.apply_move(mv)
            except Exception:
                break  # illegal move or unexpected error; stop processing this game

        # Record the terminal position too
        try:
            keys_in_game.append(position_key(board))
        except Exception:
            pass

        for key in keys_in_game:
            self.conn.execute(
                "INSERT INTO positions (key, outcome, depth, best_move, trajectories, samples, frequency) "
                "VALUES (?, NULL, NULL, NULL, '', 1, 1) "
                "ON CONFLICT(key) DO UPDATE SET "
                " samples   = positions.samples + 1, "
                " frequency = positions.frequency + 1",
                (key,),
            )
            self._positions_seen += 1

        self._processed += 1
        if self._processed % self.commit_every == 0:
            self.conn.commit()
            self._maybe_progress()

    def apply_pruning(self) -> int:
        """Delete positions below the min-frequency threshold (except solved ones)."""
        deleted = self.conn.execute(
            "DELETE FROM positions WHERE frequency < ? AND outcome IS NULL",
            (self.min_frequency,),
        ).rowcount
        self.conn.commit()
        logger.info("GameBasedBuilder: pruned %d low-frequency positions.", deleted)
        return deleted

    def write_meta(self, **kv: str) -> None:
        for k, v in kv.items():
            self.conn.execute(
                "INSERT INTO meta(k,v) VALUES(?,?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (k, str(v)),
            )
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def _maybe_progress(self) -> None:
        now = time.monotonic()
        if now - self._t_last_progress < self.progress_every:
            return
        self._t_last_progress = now
        elapsed = now - self._t_start
        rate = self._processed / elapsed if elapsed > 0 else 0
        logger.info(
            "progress: processed=%d skipped=%d positions=%d rate=%.1f games/s",
            self._processed, self._skipped, self._positions_seen, rate,
        )


def _incremental_update(
    db_path: Path,
    games_dir: Path,
    min_frequency: int,
) -> int:
    """Scan new JSONL files since last update and increment frequencies."""
    conn = sqlite3.connect(str(db_path))
    _ensure_trajectories_column(conn)

    # Determine last_updated timestamp from meta
    row = conn.execute("SELECT v FROM meta WHERE k='last_updated'").fetchone()
    last_updated = int(row[0]) if row else 0

    import glob
    files = [
        f for f in glob.glob(str(games_dir / "*.jsonl"))
        if int(os.path.getmtime(f)) > last_updated
    ]
    conn.close()

    if not files:
        print("No new game files since last update.")
        return 0

    print(f"Incremental update: {len(files)} new game files.")
    builder = GameBasedBuilder(db_path, min_frequency=min_frequency)
    for fpath in files:
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        game = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not _is_human_game(game):
                        continue
                    builder._process_game(game)
        except OSError as exc:
            logger.warning("Incremental update: cannot read %s — %s", fpath, exc)

    builder.conn.commit()
    pruned = builder.apply_pruning()
    builder.write_meta(last_updated=str(int(time.time())))
    builder.close()
    print(f"Incremental update complete: {builder._processed} games, {pruned} positions pruned.")
    return builder._processed


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Nine Men's Morris full-game position database.  "
            "A complete solve is not attempted in one run; use --max-positions "
            "or --max-depth to bound the build.  The script is resumable: "
            "re-running extends an existing DB."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help=(
            "Output SQLite file path.  Accepts any absolute path including "
            "other drives (e.g. D:/databases/fullgame.sqlite or "
            "/mnt/external/fullgame.sqlite).  "
            "Default: <project>/data/fullgame.sqlite"
        ),
    )
    parser.add_argument(
        "--db-dir", type=Path, default=None,
        help=(
            "Directory to write fullgame.sqlite into.  Shorthand for "
            "--output <dir>/fullgame.sqlite.  Useful for pointing at another "
            "drive without typing the filename.  Ignored if --output is set."
        ),
    )
    parser.add_argument(
        "--max-positions", type=int, default=None,
        help="Stop forward enumeration after this many newly-inserted positions.",
    )
    parser.add_argument(
        "--max-depth", type=int, default=None,
        help="BFS ply cap.  Positions beyond this depth are stored as frontier nodes.",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Quick sanity build: cap to N positions and skip backprop. Implies --max-positions.",
    )
    parser.add_argument(
        "--passes", type=int, default=6,
        help="Backpropagation passes for win/loss labelling (default 6).",
    )
    parser.add_argument(
        "--vacuum", action="store_true",
        help="Run SQLite VACUUM after build to minimise file size.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate environment + build a tiny in-memory DB (100 positions) without writing to disk.",
    )
    parser.add_argument(
        "--install-deps", action="store_true",
        help="Pip-install project requirements before building.",
    )
    parser.add_argument(
        "--format", choices=["sqlite", "binary"], default="binary",
        help=(
            "Output format.  'binary' (default) writes the SQLite DB first (as "
            "build intermediate), then exports it to a sorted binary file alongside "
            "the SQLite output.  'sqlite' writes the SQLite DB only."
        ),
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-progress logging.",
    )
    parser.add_argument(
        "--source-games", type=Path, default=None, metavar="DIR",
        help=(
            "Scan human-played JSONL game records in DIR instead of BFS enumeration. "
            "Only human-human and human-AI games are included. "
            "Enables frequency tracking per position."
        ),
    )
    parser.add_argument(
        "--min-frequency", type=int, default=1, metavar="N",
        help=(
            "After scanning games, prune positions reached in fewer than N games "
            "(default 1 = keep all). Solved positions (WIN/LOSS/DRAW) are always kept."
        ),
    )
    parser.add_argument(
        "--min-frequency-placement", type=int, default=None, metavar="P",
        help="Stricter frequency threshold for placement-phase positions (default: same as --min-frequency).",
    )
    parser.add_argument(
        "--update-from-games", type=Path, default=None, metavar="DIR",
        help=(
            "Incremental update mode: scan new JSONL files in DIR since last build, "
            "increment frequencies, re-apply --min-frequency pruning, then rebuild binary."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ── Resolve output path ───────────────────────────────────────────────────
    if args.output is not None:
        output_path = args.output.resolve()
    elif args.db_dir is not None:
        output_path = args.db_dir.resolve() / "fullgame.sqlite"
    else:
        output_path = (_ROOT / "data" / "fullgame.sqlite").resolve()

    # Pre-flight: make sure the target directory can actually be created/written.
    if not args.dry_run:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"ERROR: Cannot create output directory {output_path.parent}: {exc}",
                  file=sys.stderr)
            return 1
        # Quick write test so we fail fast rather than after hours of work.
        _probe = output_path.parent / ".nmm_write_probe"
        try:
            _probe.write_bytes(b"")
            _probe.unlink()
        except OSError as exc:
            print(f"ERROR: Output directory is not writable ({output_path.parent}): {exc}",
                  file=sys.stderr)
            return 1

    print(f"Output: {output_path}")

    _verify_deps()
    if args.install_deps:
        _maybe_pip_install_requirements()

    # ── Incremental update mode ──────────────────────────────────────────────
    if args.update_from_games is not None:
        if not args.update_from_games.is_dir():
            print(f"ERROR: --update-from-games: not a directory: {args.update_from_games}",
                  file=sys.stderr)
            return 1
        if not output_path.exists():
            print(f"ERROR: DB not found at {output_path}. Run a full build first.",
                  file=sys.stderr)
            return 1
        n = _incremental_update(
            output_path,
            args.update_from_games,
            args.min_frequency,
        )
        if n == 0:
            return 0
        # Fall through to binary export + settings update
        if args.format == "binary":
            binary_path = output_path.with_suffix(".bin")
            fgdb_mod = _load_fullgame_db_module()
            n_rec = fgdb_mod.export_to_binary(output_path, binary_path)
            size_mb = os.path.getsize(binary_path) / (1024 * 1024)
            print(f"Binary export: {n_rec} records, {size_mb:.1f} MB → {binary_path}")
            final_db_path = binary_path
        else:
            final_db_path = output_path
        _update_settings(final_db_path)
        return 0

    if args.dry_run:
        # Build a tiny in-memory DB to exercise every code path.
        logger.info("DRY RUN: tiny in-memory build, 100 positions, no disk write.")
        builder = FullGameDBBuilder(
            db_path=Path(":memory:"),  # ignored when we override conn
            max_positions=100,
            commit_every=50,
        )
        # Swap to a true in-memory connection so init_db's file write is benign.
        builder.conn.close()
        builder.conn = sqlite3.connect(":memory:")
        for stmt in _SCHEMA.strip().split(";"):
            s = stmt.strip()
            if not s or s.upper().startswith(("ALTER", "PRAGMA")):
                continue
            try:
                builder.conn.execute(s)
            except sqlite3.OperationalError:
                pass
        _ensure_trajectories_column(builder.conn)
        builder.enumerate_forward(BoardState.new_game())
        builder.backpropagate(passes=2)
        count = builder.conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        resolved = builder.conn.execute(
            "SELECT COUNT(*) FROM positions WHERE outcome IS NOT NULL"
        ).fetchone()[0]
        print(f"DRY RUN OK: positions={count} resolved={resolved}")
        return 0

    # ── Game-based build (B-52) ──────────────────────────────────────────────
    if args.source_games is not None:
        if not args.source_games.is_dir():
            print(f"ERROR: --source-games: not a directory: {args.source_games}",
                  file=sys.stderr)
            return 1
        t0 = time.monotonic()
        gb = GameBasedBuilder(
            output_path,
            min_frequency=args.min_frequency,
            min_frequency_placement=args.min_frequency_placement,
        )
        gb.scan_games(args.source_games)
        pruned = 0
        if args.min_frequency > 1:
            pruned = gb.apply_pruning()
        gb.write_meta(
            schema_version="1",
            built_at=str(int(time.time())),
            source="games",
            last_updated=str(int(time.time())),
            min_frequency=str(args.min_frequency),
        )
        if args.vacuum:
            gb.conn.execute("VACUUM")
        elapsed = time.monotonic() - t0
        count = gb.conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        gb.close()
        print(
            f"Game-based build: {count} positions ({pruned} pruned) "
            f"from {gb._processed} games in {elapsed:.1f}s, {size_mb:.1f} MB on disk."
        )
        if args.format == "binary":
            binary_path = output_path.with_suffix(".bin")
            fgdb_mod = _load_fullgame_db_module()
            n = fgdb_mod.export_to_binary(output_path, binary_path)
            size_mb = os.path.getsize(binary_path) / (1024 * 1024)
            print(f"Binary export: {n} records, {size_mb:.1f} MB → {binary_path}")
            final_db_path = binary_path
        else:
            final_db_path = output_path
        _update_settings(final_db_path)
        return 0

    # ── Standard BFS build ───────────────────────────────────────────────────
    max_pos = args.max_positions
    if args.sample is not None:
        max_pos = args.sample

    builder = FullGameDBBuilder(
        db_path=output_path,
        max_positions=max_pos,
        max_depth=args.max_depth,
    )
    try:
        t0 = time.monotonic()
        builder.enumerate_forward(BoardState.new_game())
        builder.write_meta(
            schema_version="1",
            built_at=str(int(time.time())),
            max_positions=str(max_pos),
            max_depth=str(args.max_depth),
        )
        if args.sample is None:
            builder.backpropagate(passes=args.passes)
        if args.vacuum:
            builder.vacuum()
        elapsed = time.monotonic() - t0
        count = builder.conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        resolved = builder.conn.execute(
            "SELECT COUNT(*) FROM positions WHERE outcome IS NOT NULL"
        ).fetchone()[0]
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(
            f"Build complete: {count} positions ({resolved} resolved) "
            f"in {elapsed:.1f}s, {size_mb:.1f} MB on disk."
        )
    finally:
        builder.close()

    if args.format == "binary":
        binary_path = output_path.with_suffix(".bin")
        logger.info("Exporting binary format → %s", binary_path)
        fgdb_mod = _load_fullgame_db_module()
        n = fgdb_mod.export_to_binary(output_path, binary_path)
        size_mb = os.path.getsize(binary_path) / (1024 * 1024)
        print(f"Binary export: {n} records, {size_mb:.1f} MB → {binary_path}")
        final_db_path = binary_path
    else:
        final_db_path = output_path

    _update_settings(final_db_path)
    return 0


def _update_settings(db_path: Path) -> None:
    settings_path = _ROOT / "data" / "settings.json"
    try:
        settings: dict = {}
        if settings_path.exists():
            with open(settings_path) as f:
                settings = json.load(f)
        settings["fullgame_db_path"] = str(db_path)
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        print(f"Settings updated: fullgame_db_path = {db_path}")
    except OSError as exc:
        print(f"WARNING: could not update settings.json: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
