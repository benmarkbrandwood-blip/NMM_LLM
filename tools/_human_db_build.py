"""tools/_human_db_build.py — Shared HumanDB build pipeline.

Underlying implementation for both `tools/build_human_db.py` and
`tools/build_human_db_sha.py`.  The two entry-point scripts differ only
in whether they emit the `.sha256` sidecar file alongside the finished
database — everything else lives here so the pipelines cannot drift.

Reviewer requirement (docs/human_blunder_net_plan.md §2.2, Phase 2
Commit A):

- one shared implementation
- entry-point wrappers differ by a single flag (`emit_sha_sidecar`)
- schema v2 preserved verbatim in this commit; schema v3 (Elo bins etc.)
  lands in a follow-up commit that only touches this module
- INSERT OR REPLACE on the schema_version meta row (chosen deliberately
  so `--rebuild` refreshes the recorded version).

`--limit-files N` (advisor request) enables sub-second smoke tests
against a fixture; also drives the byte-content regression test in
`tests/test_human_db_builders_equivalent.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.trajectory_db import make_board_state_key, _norm
from ai.board_symmetry import transform_notation
from learned_ai.data.elo_binning import ELO_BIN_SIZE, elo_bin
from learned_ai.data.malom_label_provenance import (
    ensure_human_db_can_be_annotated,
    write_current_malom_label_version,
)
from game.board import BoardState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("_human_db_build")

SCHEMA_VERSION = "3"    # v3 adds Elo-bin sparse tables + provenance meta


# ── Schema ────────────────────────────────────────────────────────────────────
# v3 is ADDITIVE — the v2 tables (positions, moves, processed_files, meta) are
# unchanged in shape so existing readers (ai/human_db.py etc.) keep working
# against a v3 DB.  v3 adds:
#   positions_elo_bins    sparse per-(state_key, mover_elo_bin) game count
#   moves_elo_bins        sparse per-(state_key, notation, mover_elo_bin) event count
# and provenance rows in `meta` for full source identification.
_DDL = """
CREATE TABLE IF NOT EXISTS positions (
    state_key              TEXT PRIMARY KEY,
    total_games            INTEGER NOT NULL DEFAULT 0,
    -- Below three are HUMAN GAME OUTCOMES for the mover.  Do not confuse
    -- with Malom W/D/L labels (which live in malom_wdl / malom_wdl_after).
    wins                   INTEGER NOT NULL DEFAULT 0,
    losses                 INTEGER NOT NULL DEFAULT 0,
    draws                  INTEGER NOT NULL DEFAULT 0,
    malom_wdl              TEXT,
    malom_dtw              INTEGER,
    canonical_winning_move TEXT
);

CREATE TABLE IF NOT EXISTS moves (
    state_key        TEXT    NOT NULL,
    notation         TEXT    NOT NULL,
    -- HUMAN GAME OUTCOMES for the mover, NOT Malom W/D/L.
    wins             INTEGER NOT NULL DEFAULT 0,
    losses           INTEGER NOT NULL DEFAULT 0,
    draws            INTEGER NOT NULL DEFAULT 0,
    total            INTEGER NOT NULL DEFAULT 0,
    moves_to_end_sum REAL    NOT NULL DEFAULT 0.0,
    malom_wdl_after  TEXT,
    malom_dtw_after  INTEGER,
    PRIMARY KEY (state_key, notation)
);

CREATE INDEX IF NOT EXISTS idx_moves_state ON moves(state_key);

CREATE TABLE IF NOT EXISTS processed_files (
    file_path   TEXT PRIMARY KEY,
    mtime       REAL    NOT NULL,
    sha256      TEXT,
    games_found INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- v3 additions (Elo bin sparse tables) --------------------------------------
-- elo_bin is `int(mover_elo) // ELO_BIN_SIZE * ELO_BIN_SIZE` (see
-- learned_ai/data/elo_binning.py).  Rows only exist for observed bins.
-- Games where the mover Elo is None are counted in the v2 aggregate columns
-- above but NOT in the *_elo_bins tables.
CREATE TABLE IF NOT EXISTS positions_elo_bins (
    state_key   TEXT NOT NULL,
    elo_bin     INTEGER NOT NULL,
    total_games INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (state_key, elo_bin)
);

CREATE TABLE IF NOT EXISTS moves_elo_bins (
    state_key   TEXT NOT NULL,
    notation    TEXT NOT NULL,
    elo_bin     INTEGER NOT NULL,
    total       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (state_key, notation, elo_bin)
);

CREATE INDEX IF NOT EXISTS idx_moves_bins_state ON moves_elo_bins(state_key);
"""


# ── DB helpers ────────────────────────────────────────────────────────────────

def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)
    # INSERT OR REPLACE (deliberate) so --rebuild refreshes the version tag
    # after schema changes; the alternative (OR IGNORE) would silently
    # leave a stale version on a rebuild.
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    conn.commit()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(processed_files)")}
    if "sha256" not in cols:
        conn.execute("ALTER TABLE processed_files ADD COLUMN sha256 TEXT")
        conn.commit()


def _clear_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DELETE FROM positions;
        DELETE FROM moves;
        DELETE FROM positions_elo_bins;
        DELETE FROM moves_elo_bins;
        DELETE FROM processed_files;
        DELETE FROM meta WHERE key != 'schema_version';
        """
    )
    conn.commit()
    log.info("Cleared existing data (--rebuild).")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_output_checksum(output_path: Path) -> str:
    """Write `<output>.sha256` alongside the finished DB.  Called only when
    the entry-point script requested it (see `emit_sha_sidecar` on run())."""
    digest = _sha256_file(output_path)
    checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {output_path.name}\n", encoding="utf-8")
    return digest


# ── JSONL parsing ─────────────────────────────────────────────────────────────

def _parse_game(record: dict) -> Optional[list[dict]]:
    """Return a list of ply dicts or None if the game should be skipped."""
    if record.get("adaptive_softened"):
        return None
    source_type = record.get("source_type", "")
    if source_type not in ("human_vs_human", "human_involved", ""):
        return None
    if record.get("self_play") or (
        record.get("white_difficulty")
        and record.get("black_difficulty")
        and not record.get("human_color")
    ):
        return None
    moves = record.get("moves", [])
    if not moves:
        return None
    return moves


def _process_file(
    path: Path,
    pos_stats: dict,
    move_stats: dict,
    pos_boards: dict,
    move_boards: dict,
    pos_bins: dict,
    move_bins: dict,
) -> int:
    """Aggregate one JSONL file into the caller's stat dicts.  Returns games indexed.

    `pos_bins` / `move_bins` are (state_key)->{bin: count} and
    (state_key, notation)->{bin: count} respectively.  A per-ply mover
    Elo of None contributes to `pos_stats` / `move_stats` (v2 rollup)
    but NOT to the Elo-bin tables (sparse).
    """
    games_found = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        moves = _parse_game(record)
        if moves is None:
            continue

        winner = record.get("winner")
        w_elo = record.get("white_elo")
        b_elo = record.get("black_elo")
        games_found += 1
        total_plies = len(moves)
        parsed_plies: list[tuple] = []

        for i, move in enumerate(moves):
            notation = _norm(move.get("notation", ""))
            fen = move.get("board_fen_before", "")
            if not notation or not fen:
                continue
            try:
                board = BoardState.from_fen_string(fen)
            except Exception:
                continue
            state_key, sym_idx = make_board_state_key(board)
            canon_notation = transform_notation(notation, sym_idx)
            if canon_notation is None:
                continue

            next_board: Optional[BoardState] = None
            if i + 1 < len(moves):
                next_fen = moves[i + 1].get("board_fen_before", "")
                if next_fen:
                    try:
                        next_board = BoardState.from_fen_string(next_fen)
                    except Exception:
                        pass

            color = move.get("color", "W")
            mover_elo = w_elo if color == "W" else b_elo
            parsed_plies.append((state_key, canon_notation, board, next_board, color, mover_elo))

        for i, (state_key, canon_notation, board, next_board, color, mover_elo) in enumerate(parsed_plies):
            plies_remaining = total_plies - i

            if state_key not in pos_stats:
                pos_stats[state_key] = {"wins": 0, "losses": 0, "draws": 0, "total": 0}
            ps = pos_stats[state_key]
            ps["total"] += 1
            if winner == color:
                ps["wins"] += 1
            elif winner is not None and winner != color:
                ps["losses"] += 1
            else:
                ps["draws"] += 1

            if state_key not in pos_boards:
                pos_boards[state_key] = board

            key = (state_key, canon_notation)
            if key not in move_stats:
                move_stats[key] = {
                    "wins": 0, "losses": 0, "draws": 0, "total": 0, "mte_sum": 0.0,
                }
            ms = move_stats[key]
            ms["total"] += 1
            ms["mte_sum"] += plies_remaining
            if winner == color:
                ms["wins"] += 1
            elif winner is not None and winner != color:
                ms["losses"] += 1
            else:
                ms["draws"] += 1

            if next_board is not None and key not in move_boards:
                move_boards[key] = next_board

            # v3 Elo-bin tallies (sparse — no row for unknown-Elo events)
            bin_start = elo_bin(mover_elo)
            if bin_start is not None:
                if state_key not in pos_bins:
                    pos_bins[state_key] = {}
                pos_bins[state_key][bin_start] = pos_bins[state_key].get(bin_start, 0) + 1
                if key not in move_bins:
                    move_bins[key] = {}
                move_bins[key][bin_start] = move_bins[key].get(bin_start, 0) + 1
    return games_found


# ── Malom annotation ──────────────────────────────────────────────────────────

def _annotate_malom(
    pos_boards: dict,
    move_boards: dict,
    malom_path: str,
) -> tuple[dict, dict, bool]:
    """Query Malom for each unique board.  Returns (pos_malom, move_malom, completed)."""
    try:
        from ai.malom_db import MalomDB
        malom = MalomDB(malom_path)
    except Exception as exc:
        log.warning("Could not load MalomDB: %s — skipping annotation.", exc)
        return {}, {}, False

    if not malom.is_available():
        log.warning("Malom DB not available at %s — skipping annotation.", malom_path)
        return {}, {}, False

    log.info(
        "Malom DB ready. Annotating %d positions + %d next-positions …",
        len(pos_boards), len(move_boards),
    )
    pos_malom: dict = {}
    move_malom: dict = {}

    def _query(board):
        try:
            return malom.query(board)
        except Exception:
            return None

    total = len(pos_boards) + len(move_boards)
    done = 0
    log_every = max(1, total // 20) if total else 1
    for state_key, board in pos_boards.items():
        res = _query(board)
        if res:
            pos_malom[state_key] = {"wdl": res["outcome"], "dtw": res.get("dtw")}
        done += 1
        if done % log_every == 0:
            log.info(" Malom annotation: %d / %d (%.0f%%)", done, total, 100 * done / total)
    for key, board in move_boards.items():
        res = _query(board)
        if res:
            move_malom[key] = {"wdl": res["outcome"], "dtw": res.get("dtw")}
        done += 1
        if done % log_every == 0:
            log.info(" Malom annotation: %d / %d (%.0f%%)", done, total, 100 * done / total)
    log.info(
        "Malom annotation complete: %d position hits, %d move hits.",
        len(pos_malom), len(move_malom),
    )
    return pos_malom, move_malom, True


# ── Upserts ───────────────────────────────────────────────────────────────────

def _upsert_positions(conn: sqlite3.Connection, pos_stats: dict, pos_malom: dict) -> None:
    rows = []
    for state_key, s in pos_stats.items():
        pm = pos_malom.get(state_key, {})
        rows.append((
            state_key, s["total"], s["wins"], s["losses"], s["draws"],
            pm.get("wdl"), pm.get("dtw"),
        ))
    conn.executemany(
        """
        INSERT INTO positions(state_key, total_games, wins, losses, draws, malom_wdl, malom_dtw)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(state_key) DO UPDATE SET
            total_games = total_games + excluded.total_games,
            wins = wins + excluded.wins,
            losses = losses + excluded.losses,
            draws = draws + excluded.draws,
            malom_wdl = COALESCE(positions.malom_wdl, excluded.malom_wdl),
            malom_dtw = COALESCE(positions.malom_dtw, excluded.malom_dtw)
        """,
        rows,
    )


def _upsert_moves(conn: sqlite3.Connection, move_stats: dict, move_malom: dict) -> None:
    rows = []
    for (state_key, canon_notation), s in move_stats.items():
        mm = move_malom.get((state_key, canon_notation), {})
        rows.append((
            state_key, canon_notation,
            s["wins"], s["losses"], s["draws"], s["total"], s["mte_sum"],
            mm.get("wdl"), mm.get("dtw"),
        ))
    conn.executemany(
        """
        INSERT INTO moves(
            state_key, notation, wins, losses, draws, total, moves_to_end_sum,
            malom_wdl_after, malom_dtw_after
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(state_key, notation) DO UPDATE SET
            wins = wins + excluded.wins,
            losses = losses + excluded.losses,
            draws = draws + excluded.draws,
            total = total + excluded.total,
            moves_to_end_sum = moves_to_end_sum + excluded.moves_to_end_sum,
            malom_wdl_after = COALESCE(moves.malom_wdl_after, excluded.malom_wdl_after),
            malom_dtw_after = COALESCE(moves.malom_dtw_after, excluded.malom_dtw_after)
        """,
        rows,
    )


def _upsert_positions_elo_bins(conn: sqlite3.Connection, pos_bins: dict) -> None:
    rows = []
    for state_key, bins in pos_bins.items():
        for bin_start, n in bins.items():
            rows.append((state_key, int(bin_start), int(n)))
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO positions_elo_bins(state_key, elo_bin, total_games)
        VALUES (?, ?, ?)
        ON CONFLICT(state_key, elo_bin) DO UPDATE SET
            total_games = total_games + excluded.total_games
        """,
        rows,
    )


def _upsert_moves_elo_bins(conn: sqlite3.Connection, move_bins: dict) -> None:
    rows = []
    for (state_key, canon_notation), bins in move_bins.items():
        for bin_start, n in bins.items():
            rows.append((state_key, canon_notation, int(bin_start), int(n)))
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO moves_elo_bins(state_key, notation, elo_bin, total)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(state_key, notation, elo_bin) DO UPDATE SET
            total = total + excluded.total
        """,
        rows,
    )


def _recompute_canonical_winning_moves(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE positions
        SET canonical_winning_move = (
            SELECT notation
            FROM moves
            WHERE moves.state_key = positions.state_key
            ORDER BY wins DESC, total DESC
            LIMIT 1
        )
        """
    )


def _update_meta(
    conn: sqlite3.Connection,
    game_count: int,
    file_count: int,
    source_checksum_count: int,
    *,
    provenance: Optional[dict] = None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('build_date', ?)",
        (datetime.now().isoformat(timespec="seconds"),),
    )
    existing_games = conn.execute(
        "SELECT value FROM meta WHERE key = 'total_games'"
    ).fetchone()
    prev = int(existing_games[0]) if existing_games and existing_games[0] else 0
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('total_games', ?)",
        (str(prev + game_count),),
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('total_files', ?)",
        (str(file_count),),
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('source_sha256_files', ?)",
        (str(source_checksum_count),),
    )
    for key, value in (provenance or {}).items():
        if value is None:
            continue
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (key, str(value)),
        )


# ── Provenance helpers ────────────────────────────────────────────────────────

def _git_head() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return None


def _feature_canonicalisation_version() -> str:
    """Version tag for the state-key canonicalisation pipeline
    (make_board_state_key + transform_notation)."""
    return "d4-canonical-v1"


def _sources_manifest_sha256(file_infos: dict) -> str:
    """SHA-256 of the sorted (relative_path, sha256, mtime) tuples.

    Uses relative paths (relative to ROOT) so a candidate DB built on
    two different hosts under the same working tree produces the same
    manifest hash — reviewer's "logical source identity" requirement.
    """
    entries: list[tuple[str, float, str]] = []
    for fp, (mtime, _n, sha) in file_infos.items():
        try:
            rel = str(Path(fp).resolve().relative_to(ROOT.resolve()))
        except Exception:
            rel = str(Path(fp).name)
        entries.append((rel, mtime, sha))
    entries.sort()
    h = hashlib.sha256()
    for rel, mtime, sha in entries:
        h.update(rel.encode()); h.update(b"|")
        h.update(sha.encode()); h.update(b"|")
        h.update(f"{mtime:.0f}".encode()); h.update(b"\n")
    return h.hexdigest()


# ── Active-DB fail-closed guard ────────────────────────────────────────────────

def _resolve_active_db_paths() -> set[Path]:
    """Return every path we must refuse to overwrite: the reviewer's
    "active" HumanDB(s).  Resolved from `data/training_paths.local.json`
    (if present) plus the historical default `data/human_db.sqlite`."""
    paths: set[Path] = {(ROOT / "data" / "human_db.sqlite").resolve()}
    cfg = ROOT / "data" / "training_paths.local.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            p = data.get("human_db_path")
            if p:
                pp = Path(p)
                paths.add((pp if pp.is_absolute() else ROOT / pp).resolve())
        except Exception:
            pass
    return paths


def _guard_active_db(output_path: Path) -> None:
    """Fail closed if the resolved output path is the active HumanDB.
    The candidate DB must be written to a separate path."""
    resolved = output_path.resolve()
    active = _resolve_active_db_paths()
    if resolved in active:
        raise SystemExit(
            f"REFUSING to write v3 schema to the active HumanDB at {resolved}.\n"
            f"Pass --candidate-out with a distinct path (e.g. "
            f"data/human_db_candidate.sqlite).\n"
            f"Activation is a separate later step; see docs/human_blunder_net_plan.md §2.2."
        )


def _resolve_malom_path(cli_path: str, no_malom: bool) -> str:
    if no_malom:
        return ""
    if cli_path:
        return cli_path
    try:
        from learned_ai.sentinel.config import load_config as _lc
        return getattr(_lc(), "external_db_path", "") or ""
    except Exception:
        return ""


# ── Public argparser + run() ──────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    """Return the argparse parser used by BOTH entry-point scripts.  Keep
    argument set identical here so `--help` output matches."""
    ap = argparse.ArgumentParser(description="Build or update data/human_db.sqlite")
    ap.add_argument("--games-dir", default="data/human_games",
                    help="Directory containing human-vs-human *.jsonl files")
    ap.add_argument("--extra-dirs", nargs="*", default=[],
                    help="Additional game directories to include")
    ap.add_argument("--output", default="data/human_db.sqlite",
                    help="Output SQLite path")
    ap.add_argument("--malom-db", default="",
                    help="Path to Malom DB directory (e.g. .../Std_DD_89adjusted)")
    ap.add_argument("--no-malom", action="store_true",
                    help="Skip Malom annotation (malom_wdl/dtw columns stay NULL)")
    ap.add_argument("--update", action="store_true",
                    help="Only process files not yet in processed_files or whose SHA-256 changed")
    ap.add_argument("--rebuild", action="store_true",
                    help="Clear DB and reprocess all files from scratch")
    ap.add_argument("--limit-files", type=int, default=None,
                    help="Cap number of JSONL files scanned (fixture / smoke tests).")
    ap.add_argument("--candidate-out", type=str, default="",
                    help="Preferred flag for v3 candidate DB output.  Overrides "
                         "--output when both are given.  Required whenever "
                         "--output resolves to the active HumanDB path.")
    return ap


def run(args: argparse.Namespace, *, emit_sha_sidecar: bool = False) -> None:
    """Execute the build pipeline.  `emit_sha_sidecar=True` produces the
    `<output>.sha256` sidecar file — the sole difference between the two
    entry-point scripts."""
    # v3 schema is now produced unconditionally.  Fail closed against the
    # active DB path so a stray --output doesn't overwrite production data.
    chosen_out = getattr(args, "candidate_out", "") or args.output
    output_path = Path(chosen_out)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    _guard_active_db(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(output_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _init_db(conn)
    _migrate_schema(conn)

    if args.rebuild:
        _clear_db(conn)

    malom_path = _resolve_malom_path(args.malom_db, args.no_malom)
    if malom_path:
        ensure_human_db_can_be_annotated(conn, output_path)
    elif not args.no_malom:
        log.info(
            "No --malom-db specified and no config path found; "
            "skipping annotation."
        )

    all_dirs = [ROOT / args.games_dir] + [ROOT / d for d in args.extra_dirs]
    all_files: list[Path] = []
    for d in all_dirs:
        if d.exists():
            all_files.extend(sorted(d.rglob("*.jsonl")))
        else:
            log.warning("Directory not found: %s", d)

    if args.limit_files is not None:
        all_files = all_files[: args.limit_files]

    pending_hashes: dict[str, tuple[float, str]] = {}

    if args.update and not args.rebuild:
        already = {
            row[0]: {"mtime": row[1], "sha256": row[2]}
            for row in conn.execute("SELECT file_path, mtime, sha256 FROM processed_files")
        }
        new_files: list[Path] = []
        for p in all_files:
            mtime = p.stat().st_mtime
            sha256 = _sha256_file(p)
            prev = already.get(str(p))
            if prev is None or prev.get("sha256") != sha256:
                new_files.append(p)
                pending_hashes[str(p)] = (mtime, sha256)
        log.info("--update: %d / %d files need processing.", len(new_files), len(all_files))
        all_files = new_files
    else:
        for p in all_files:
            pending_hashes[str(p)] = (p.stat().st_mtime, _sha256_file(p))

    if not all_files:
        conn.close()
        if emit_sha_sidecar and output_path.exists():
            db_sha = _write_output_checksum(output_path)
            log.info("No files to process. DB is up to date. SQLite SHA-256: %s", db_sha)
        else:
            log.info("No files to process. DB is up to date.")
        return

    log.info("Processing %d files from %s…", len(all_files), args.games_dir)

    pos_stats: dict = {}
    move_stats: dict = {}
    pos_boards: dict = {}
    move_boards: dict = {}
    pos_bins:  dict = {}
    move_bins: dict = {}
    total_games = 0
    file_info_map: dict[str, tuple[float, int, str]] = {}

    t0 = time.time()
    for i, path in enumerate(all_files):
        mtime, sha256 = pending_hashes[str(path)]
        try:
            n = _process_file(path, pos_stats, move_stats,
                              pos_boards, move_boards,
                              pos_bins, move_bins)
        except Exception as exc:
            log.warning("Skipping %s — %s", path.name, exc)
            n = 0
        total_games += n
        file_info_map[str(path)] = (mtime, n, sha256)
        if (i + 1) % 500 == 0 or (i + 1) == len(all_files):
            elapsed = time.time() - t0
            log.info(
                " Parsed %d / %d files, %d games, %.1f s",
                i + 1, len(all_files), total_games, elapsed,
            )

    log.info(
        "Parsed %d games → %d unique positions, %d unique (position, move) pairs.",
        total_games, len(pos_stats), len(move_stats),
    )

    pos_malom: dict = {}
    move_malom: dict = {}
    malom_annotation_completed = False
    if malom_path:
        pos_malom, move_malom, malom_annotation_completed = _annotate_malom(
            pos_boards, move_boards, malom_path,
        )

    log.info("Writing to %s …", output_path)
    provenance = {
        "elo_bin_size":                     str(ELO_BIN_SIZE),
        "feature_canonicalisation_version": _feature_canonicalisation_version(),
        "builder_git_commit":               _git_head() or "",
        "source_manifest_sha256":           _sources_manifest_sha256(file_info_map),
        "built_at":                         datetime.now().isoformat(timespec="seconds"),
    }
    with conn:
        _upsert_positions(conn, pos_stats, pos_malom)
        _upsert_moves(conn, move_stats, move_malom)
        _upsert_positions_elo_bins(conn, pos_bins)
        _upsert_moves_elo_bins(conn, move_bins)
        _recompute_canonical_winning_moves(conn)
        _update_meta(conn, total_games, len(file_info_map), len(file_info_map),
                     provenance=provenance)
        if malom_annotation_completed:
            write_current_malom_label_version(conn)
        conn.executemany(
            "INSERT OR REPLACE INTO processed_files(file_path, mtime, sha256, games_found) VALUES (?, ?, ?, ?)",
            [(fp, mt, sha, gf) for fp, (mt, gf, sha) in file_info_map.items()],
        )

    elapsed = time.time() - t0
    pos_count = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    move_count = conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0]
    conn.close()

    if emit_sha_sidecar:
        db_sha = _write_output_checksum(output_path)
        log.info(
            "Done in %.1f s. DB: %d positions, %d moves, %d games. SQLite SHA-256: %s → %s",
            elapsed, pos_count, move_count, total_games, db_sha, output_path,
        )
    else:
        log.info(
            "Done in %.1f s. DB: %d positions, %d moves, %d games → %s",
            elapsed, pos_count, move_count, total_games, output_path,
        )
