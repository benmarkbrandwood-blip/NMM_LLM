"""learned_ai/data/specialist_db.py — Self-built experience database for specialist AIs.

Each specialist maintains a persistent SQLite database that accumulates per-position
WDL statistics from self-play games, Malom-validated labels for key positions (training
only), tagged winning move sequences, and promoted preferred plays.

Position keys use D4 (dihedral-8) symmetry so that all 8 rotationally and reflectionally
equivalent board positions share the same database entry, giving up to 8× data efficiency
— identical to the approach used by the endgame databases.

At inference, the DB populates counterfactual feature slots with WDL fractions from
self-play history, substituting for Malom (not required at inference time).
Ships pre-seeded from training; grows further from every game the user plays.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from learned_ai.data.malom_label_provenance import (
    CURRENT_MALOM_LABEL_VERSION,
    read_malom_label_version,
    write_current_malom_label_version,
)
from learned_ai.data.data_contract import TypedLabel

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS positions (
    pos_hash     TEXT    PRIMARY KEY,
    wins         INTEGER NOT NULL DEFAULT 0,
    draws        INTEGER NOT NULL DEFAULT 0,
    losses       INTEGER NOT NULL DEFAULT 0,
    malom_label  TEXT    DEFAULT NULL,
    last_seen    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS winning_lines (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    move_seq     TEXT    NOT NULL,
    phase        TEXT    NOT NULL,
    result       TEXT    NOT NULL,
    wins         INTEGER NOT NULL DEFAULT 1,
    times_played INTEGER NOT NULL DEFAULT 1,
    win_rate     REAL    NOT NULL DEFAULT 1.0,
    last_seen    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_wl_phase ON winning_lines(phase);
CREATE INDEX IF NOT EXISTS idx_wl_wr   ON winning_lines(win_rate);

CREATE TABLE IF NOT EXISTS preferred_plays (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    tag          TEXT    NOT NULL,
    pos_sequence TEXT    NOT NULL,
    win_rate     REAL    NOT NULL,
    times_played INTEGER NOT NULL DEFAULT 0,
    promoted     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_pp_promoted ON preferred_plays(promoted);

CREATE TABLE IF NOT EXISTS meta (
    key          TEXT PRIMARY KEY,
    value        TEXT
);
"""

_PROMOTE_MIN_PLAYED = 5
_PROMOTE_WIN_RATE   = 0.65
_DEMOTE_WIN_RATE    = 0.45
_DEMOTE_MIN_RECENT  = 20
_MIN_SAMPLES_QUERY  = 5
_RULES_VERSION = "nmm-project-rules-v1"


@dataclass(frozen=True)
class SpecialistWdlEvidence:
    """Physically separated theoretical and empirical position evidence."""

    perspective: str
    theoretical_wdl: Optional[TypedLabel]
    empirical_counts: tuple[int, int, int]
    empirical_distribution: Optional[Tuple[float, float, float]]


def _board_hash(board) -> str:
    """D4-canonical hash — all 8 symmetric equivalents map to the same key."""
    from ai.board_symmetry import canonical_board_str
    fen   = board.to_fen_string()
    parts = fen.split("|")          # [board_24, turn, W_placed, B_placed]
    canon, _ = canonical_board_str(parts[0])
    key = f"{canon}|{parts[1]}|{parts[2]}|{parts[3]}"
    return hashlib.sha1(key.encode()).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


class SpecialistDB:
    """Persistent self-play experience database for one specialist.

    Not thread-safe. Each training process opens its own instance.
    Grows indefinitely — even 100 000 games stay under 1 GB.
    """

    def __init__(self, db_path, *, read_only: bool = False) -> None:
        self._path = Path(db_path)
        self._read_only = bool(read_only)
        if self._read_only and not self._path.is_file():
            raise FileNotFoundError(f"read-only SpecialistDB does not exist: {self._path}")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._read_only:
            uri = f"file:{self._path.resolve().as_posix()}?mode=ro"
            self._conn = sqlite3.connect(
                uri, uri=True, check_same_thread=False
            )
        else:
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA_SQL)
        self._malom_label_count = self._conn.execute(
            "SELECT COUNT(*) FROM positions WHERE malom_label IS NOT NULL"
        ).fetchone()[0]
        self._malom_label_version = read_malom_label_version(self._conn)
        if self._malom_label_count == 0 and not self._read_only:
            # An empty label set is safe to adopt.  This covers both a new DB
            # and an existing self-play-only DB.
            write_current_malom_label_version(self._conn)
            self._malom_label_version = CURRENT_MALOM_LABEL_VERSION
        self._malom_labels_trusted = (
            self._malom_label_version == CURRENT_MALOM_LABEL_VERSION
        )
        self._checkpoint_identity_cache = None
        if not self._read_only:
            self._conn.commit()

    @property
    def malom_label_version(self) -> Optional[str]:
        return self._malom_label_version

    @property
    def malom_labels_trusted(self) -> bool:
        return self._malom_labels_trusted

    def require_trusted_malom_labels(self) -> None:
        """Fail before training can read or append incompatible Malom labels."""
        if self._malom_labels_trusted:
            return
        shown_version = self._malom_label_version or "<missing>"
        raise RuntimeError(
            f"SpecialistDB {self._path} contains {self._malom_label_count} "
            f"Malom labels with version {shown_version!r}; "
            f"{CURRENT_MALOM_LABEL_VERSION!r} is required. Use a new database "
            "path or rebuild the SpecialistDB before training."
        )

    def require_writable(self) -> None:
        """Reject writes through a SpecialistDB opened as a trusted snapshot."""
        if self._read_only:
            raise RuntimeError("SpecialistDB is read-only; runtime writes are quarantined")

    def checkpoint_identity(self) -> dict:
        """Flush WAL state and return a cached cryptographic database identity."""
        self.require_writable()
        change_count = self._conn.total_changes
        cached = self._checkpoint_identity_cache
        if cached is not None and cached[0] == change_count:
            return dict(cached[1])

        self._conn.commit()
        busy, log_pages, checkpointed_pages = self._conn.execute(
            "PRAGMA wal_checkpoint(FULL)"
        ).fetchone()
        if busy:
            raise RuntimeError(
                "SpecialistDB WAL checkpoint was busy; refusing checkpoint identity"
            )
        digest = hashlib.sha256()
        with self._path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        identity = {
            "sha256": digest.hexdigest(),
            "size": self._path.stat().st_size,
            "label_version": self._malom_label_version,
            "malom_label_count": int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM positions WHERE malom_label IS NOT NULL"
                ).fetchone()[0]
            ),
            "wal_log_pages": int(log_pages),
            "wal_checkpointed_pages": int(checkpointed_pages),
        }
        self._checkpoint_identity_cache = (change_count, identity)
        return dict(identity)

    # ── Position statistics ───────────────────────────────────────────────────

    def record_game(
        self,
        boards: List,
        result: str,
        move_seq: List[str],
        phase: str,
        learner_color: str = None,
    ) -> None:
        """Record all positions from a completed game and update winning lines.

        Parameters
        ----------
        boards        : list of BoardState objects (pre- and post-move boards)
        result        : 'W', 'D', or 'L' from the learner's perspective
        move_seq      : list of move notation strings (learner's moves only)
        phase         : 'open' | 'mid' | 'end'
        learner_color : 'W' or 'B' — when set, boards where board.turn != learner_color
                        are opponent-to-move; WDL is stored from the current player's
                        perspective, so W↔L are flipped for those boards.
        """
        self.require_writable()
        now = _now()
        with self._conn:
            for board in boards:
                h = _board_hash(board)
                # Flip W↔L for opponent-to-move boards so DB is always stored from
                # the current player's perspective (matched to encoder query path).
                if learner_color is not None and getattr(board, "turn", None) != learner_color:
                    eff = "L" if result == "W" else ("W" if result == "L" else "D")
                else:
                    eff = result
                self._conn.execute("""
                    INSERT INTO positions (pos_hash, wins, draws, losses, last_seen)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(pos_hash) DO UPDATE SET
                        wins      = wins   + excluded.wins,
                        draws     = draws  + excluded.draws,
                        losses    = losses + excluded.losses,
                        last_seen = excluded.last_seen
                """, (h,
                      1 if eff == "W" else 0,
                      1 if eff == "D" else 0,
                      1 if eff == "L" else 0,
                      now))

            if result in ("W", "D") and move_seq:
                seq_json = json.dumps(move_seq)
                row = self._conn.execute(
                    "SELECT id, times_played, wins FROM winning_lines WHERE move_seq=? AND phase=?",
                    (seq_json, phase)
                ).fetchone()
                if row:
                    wid, played, wins = row
                    new_played = played + 1
                    new_wins   = wins + (1 if result == "W" else 0)
                    self._conn.execute(
                        "UPDATE winning_lines SET times_played=?, wins=?, win_rate=?, last_seen=? WHERE id=?",
                        (new_played, new_wins, new_wins / new_played, now, wid)
                    )
                else:
                    self._conn.execute("""
                        INSERT INTO winning_lines (move_seq, phase, result, wins, times_played, win_rate, last_seen)
                        VALUES (?, ?, ?, ?, 1, 1.0, ?)
                    """, (seq_json, phase, result, 1 if result == "W" else 0, now))

            self._promote_lines(phase)

    def _promote_lines(self, phase: str) -> None:
        rows = self._conn.execute("""
            SELECT id, move_seq, times_played, win_rate
            FROM winning_lines
            WHERE phase=? AND times_played >= ? AND win_rate >= ?
        """, (phase, _PROMOTE_MIN_PLAYED, _PROMOTE_WIN_RATE)).fetchall()

        for wid, seq_json, played, wr in rows:
            existing = self._conn.execute(
                "SELECT id FROM preferred_plays WHERE pos_sequence=?", (seq_json,)
            ).fetchone()
            if not existing:
                self._conn.execute("""
                    INSERT INTO preferred_plays (tag, pos_sequence, win_rate, times_played, promoted)
                    VALUES (?, ?, ?, ?, 1)
                """, (f"{phase}_line_{wid}", seq_json, wr, played))
            else:
                self._conn.execute(
                    "UPDATE preferred_plays SET win_rate=?, times_played=?, promoted=1 WHERE id=?",
                    (wr, played, existing[0])
                )

        demote = self._conn.execute(
            "SELECT id, pos_sequence FROM preferred_plays WHERE promoted=1"
        ).fetchall()
        for pid, seq_json in demote:
            row = self._conn.execute(
                "SELECT times_played, win_rate FROM winning_lines WHERE move_seq=?", (seq_json,)
            ).fetchone()
            if row and row[0] >= _DEMOTE_MIN_RECENT and row[1] < _DEMOTE_WIN_RATE:
                self._conn.execute("UPDATE preferred_plays SET promoted=0 WHERE id=?", (pid,))

    # ── Malom validation (training-time only) ─────────────────────────────────

    def label_position_malom(self, board, wdl: str) -> None:
        """Store a Malom WDL label ('W'/'D'/'L') for a position (training time only)."""
        self.require_writable()
        self.require_trusted_malom_labels()
        if wdl not in ("W", "D", "L"):
            raise ValueError(f"Invalid Malom WDL label: {wdl!r}")
        h = _board_hash(board)
        existing = self._conn.execute(
            "SELECT malom_label FROM positions WHERE pos_hash=?", (h,)
        ).fetchone()
        if existing is not None and existing[0] is not None and existing[0] != wdl:
            raise RuntimeError(
                "conflicting trusted Malom label; refusing to overwrite evidence"
            )
        self._conn.execute("""
            INSERT INTO positions (pos_hash, wins, draws, losses, malom_label, last_seen)
            VALUES (?, 0, 0, 0, ?, ?)
            ON CONFLICT(pos_hash) DO UPDATE SET
                malom_label = COALESCE(positions.malom_label, excluded.malom_label)
        """, (h, wdl, _now()))
        self._conn.commit()

    # ── Inference query ───────────────────────────────────────────────────────

    def query_wdl_evidence(
        self, board, min_samples: int = _MIN_SAMPLES_QUERY
    ) -> Optional[SpecialistWdlEvidence]:
        """Return theoretical and empirical WDL evidence without blending them."""
        h = _board_hash(board)
        row = self._conn.execute(
            "SELECT wins, draws, losses, malom_label FROM positions WHERE pos_hash=?",
            (h,),
        ).fetchone()
        if row is None:
            return None
        wins, draws, losses, malom_label = row
        theoretical = None
        if self._malom_labels_trusted and malom_label:
            theoretical = TypedLabel(
                kind="theoretical_wdl",
                value=malom_label,
                perspective=board.turn,
                rules_version=_RULES_VERSION,
                history_identity=_board_hash(board),
                source_identity=(
                    f"specialist-db-malom:{self._malom_label_version}"
                ),
                validity_version=self._malom_label_version,
            )
        total = wins + draws + losses
        empirical = None
        if total >= min_samples:
            empirical = (wins / total, draws / total, losses / total)
        return SpecialistWdlEvidence(
            perspective=board.turn,
            theoretical_wdl=theoretical,
            empirical_counts=(wins, draws, losses),
            empirical_distribution=empirical,
        )

    def query_wdl(self, board, min_samples: int = _MIN_SAMPLES_QUERY) -> Optional[Tuple[float, float, float]]:
        """Return the legacy compatibility projection of separated WDL evidence.

        When a Malom label exists and self-play count is low, the Malom label
        provides a strong prior.  At inference without Malom, self-play statistics
        substitute once enough games have been played.
        """
        evidence = self.query_wdl_evidence(board, min_samples)
        if evidence is None:
            return None
        if (
            evidence.theoretical_wdl is not None
            and evidence.empirical_distribution is None
        ):
            if evidence.theoretical_wdl.value == "W":
                return (0.90, 0.05, 0.05)
            if evidence.theoretical_wdl.value == "D":
                return (0.05, 0.90, 0.05)
            if evidence.theoretical_wdl.value == "L":
                return (0.05, 0.05, 0.90)
        return evidence.empirical_distribution

    def query_win_prob(
        self, board, min_samples: int = _MIN_SAMPLES_QUERY
    ) -> Optional[float]:
        """Return P(win) + 0.5*P(draw), preserving unknown as None."""
        wdl = self.query_wdl(board, min_samples)
        if wdl is None:
            return None
        w, d, _ = wdl
        return w + 0.5 * d

    # ── Preferred plays ───────────────────────────────────────────────────────

    def get_promoted_plays(self, phase: str = "") -> List[Tuple[str, List[str], float]]:
        """Return [(tag, move_list, win_rate)] for all promoted plays."""
        rows = self._conn.execute(
            "SELECT tag, pos_sequence, win_rate FROM preferred_plays WHERE promoted=1 ORDER BY win_rate DESC"
        ).fetchall()
        result = []
        for tag, seq_json, wr in rows:
            try:
                result.append((tag, json.loads(seq_json), float(wr)))
            except Exception:
                pass
        return result

    # ── Stats / maintenance ───────────────────────────────────────────────────

    def stats(self) -> dict:
        pos  = self._conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        well = self._conn.execute(
            "SELECT COUNT(*) FROM positions WHERE wins+draws+losses >= ?", (_MIN_SAMPLES_QUERY,)
        ).fetchone()[0]
        malom = self._conn.execute(
            "SELECT COUNT(*) FROM positions WHERE malom_label IS NOT NULL"
        ).fetchone()[0]
        lines = self._conn.execute("SELECT COUNT(*) FROM winning_lines").fetchone()[0]
        prefs = self._conn.execute(
            "SELECT COUNT(*) FROM preferred_plays WHERE promoted=1"
        ).fetchone()[0]
        return {
            "positions": pos,
            "well_sampled": well,
            "malom_labeled": malom,
            "malom_label_version": self._malom_label_version,
            "malom_labels_trusted": self._malom_labels_trusted,
            "winning_lines": lines,
            "preferred_plays": prefs,
        }

    def close(self) -> None:
        self._conn.close()
