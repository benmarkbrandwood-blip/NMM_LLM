"""Immutable HumanDB snapshot and genuine twelve-ply history audit."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
from learned_ai.evaluation.layered_book_audit import (
    verify_layered_book_audit,
)
from learned_ai.evaluation.layered_opening_prefix import (
    LAYERED_PREFIX_SCHEMA,
    PREFIX_LOGICAL_PLIES_BY_SIDE_V2,
    PREFIX_LOGICAL_PLIES_V2,
)
from learned_ai.evaluation.oracle_corpus import ring16_canonical_fen
from learned_ai.evaluation.sanmill_uci import nmm_move_base
from learned_ai.training.run_contract import (
    canonical_json_bytes,
    canonical_sha256,
)


LAYERED_HUMAN_AUDIT_SCHEMA = "nmm.layered-human-source-audit.v1"
HUMAN_HISTORY_LEDGER_SCHEMA = "nmm.human-prefix-history-ledger.v1"
HUMAN_HISTORY_SCHEMA = "nmm.human-prefix-history.v1"
SQLITE_SNAPSHOT_SCHEMA = "nmm.sqlite-online-backup-snapshot.v1"


class LayeredHumanAuditError(ValueError):
    """Raised when HumanDB snapshot or source-history evidence is invalid."""


@dataclass(frozen=True)
class ExtractedHumanPrefix:
    relative_path: str
    file_size: int
    file_sha256: str
    session_sha256: str
    source: str
    source_type: str
    winner: str | None
    logical_turns: tuple[tuple[str, ...], ...]
    action_tokens: tuple[str, ...]
    final_nmm_fen: str
    final_ring16_fen: str


@dataclass
class _HistoryGroup:
    logical_turns: tuple[tuple[str, ...], ...]
    action_tokens: tuple[str, ...]
    final_nmm_fen: str
    final_ring16_fen: str
    occurrence_count: int = 0
    session_ids: set[str] = field(default_factory=set)
    results: Counter[str] = field(default_factory=Counter)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "present": False,
            "byte_length": 0,
            "sha256": None,
            "mtime_ns": None,
        }
    stat = path.stat()
    return {
        "present": True,
        "byte_length": stat.st_size,
        "sha256": _file_sha256(path),
        "mtime_ns": stat.st_mtime_ns,
    }


def _sidecars(database: Path) -> dict[str, Any]:
    return {
        "wal": _sidecar_record(Path(str(database) + "-wal")),
        "shm": _sidecar_record(Path(str(database) + "-shm")),
    }


def _sqlite_uri(path: Path, *, immutable: bool) -> str:
    suffix = "?mode=ro"
    if immutable:
        suffix += "&immutable=1"
    return path.resolve().as_uri() + suffix


def _sqlite_evidence(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(
        _sqlite_uri(path, immutable=True),
        uri=True,
    )
    try:
        quick_check = [
            str(row[0])
            for row in connection.execute("PRAGMA quick_check").fetchall()
        ]
        schema_rows = [
            {
                "type": str(row[0]),
                "name": str(row[1]),
                "table_name": str(row[2]),
                "sql": row[3],
            }
            for row in connection.execute(
                """
                SELECT type, name, tbl_name, sql
                FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """
            )
        ]
        tables = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            )
        }
        expected_tables = {"meta", "moves", "positions", "processed_files"}
        if tables != expected_tables:
            raise LayeredHumanAuditError(
                f"unexpected HumanDB tables: {sorted(tables)}"
            )
        row_counts = {
            table: int(
                connection.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0]
            )
            for table in sorted(tables)
        }
        meta = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT key, value FROM meta ORDER BY key"
            )
        }
        journal_mode = str(
            connection.execute("PRAGMA journal_mode").fetchone()[0]
        )
        user_version = int(
            connection.execute("PRAGMA user_version").fetchone()[0]
        )
    finally:
        connection.close()
    return {
        "quick_check": quick_check,
        "schema_sha256": canonical_sha256(schema_rows),
        "schema": schema_rows,
        "row_counts": row_counts,
        "meta": meta,
        "journal_mode": journal_mode,
        "user_version": user_version,
    }


def create_human_db_snapshot(
    source: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Create a point-in-time SQLite online backup without deleting sidecars."""
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if not source_path.is_file():
        raise LayeredHumanAuditError("active HumanDB is not a file")
    if destination_path.exists():
        raise LayeredHumanAuditError("HumanDB snapshot already exists")
    if destination_path in {
        source_path,
        Path(str(source_path) + "-wal"),
        Path(str(source_path) + "-shm"),
    }:
        raise LayeredHumanAuditError("snapshot destination overlaps active DB")
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    source_before = {
        "byte_length": source_path.stat().st_size,
        "mtime_ns": source_path.stat().st_mtime_ns,
        "sidecars": _sidecars(source_path),
    }
    source_connection = sqlite3.connect(
        _sqlite_uri(source_path, immutable=False),
        uri=True,
    )
    destination_connection = sqlite3.connect(destination_path)
    try:
        source_connection.backup(destination_connection, pages=4096)
        destination_connection.commit()
    except BaseException:
        destination_connection.close()
        source_connection.close()
        if destination_path.exists():
            destination_path.unlink()
        raise
    else:
        destination_connection.close()
        source_connection.close()

    source_after = {
        "byte_length": source_path.stat().st_size,
        "mtime_ns": source_path.stat().st_mtime_ns,
        "sidecars": _sidecars(source_path),
    }
    for suffix in ("-wal", "-shm"):
        if Path(str(destination_path) + suffix).exists():
            raise LayeredHumanAuditError(
                "snapshot unexpectedly retained a SQLite sidecar"
            )
    evidence = _sqlite_evidence(destination_path)
    if evidence["quick_check"] != ["ok"]:
        raise LayeredHumanAuditError("HumanDB snapshot quick_check failed")
    snapshot = {
        "schema_version": SQLITE_SNAPSHOT_SCHEMA,
        "source": {
            "path_lookup_key": "human_db_path",
            "before": source_before,
            "after": source_after,
            "sidecars_deleted": False,
        },
        "snapshot": {
            "path_lookup_key": "human_db_prefix12_snapshot_path",
            "byte_length": destination_path.stat().st_size,
            "sha256": _file_sha256(destination_path),
            **evidence,
        },
        "method": {
            "api": "python-sqlite3.Connection.backup",
            "source_mode": "read-only",
            "snapshot_query_mode": "read-only-immutable",
        },
    }
    snapshot["snapshot_identity"] = canonical_sha256(snapshot)
    return snapshot


def _move_notation(move: Mapping[str, Any]) -> str:
    notation = nmm_move_base(move)
    capture = move.get("capture")
    return notation if capture is None else f"{notation}x{capture}"


def _logical_turn(move: Mapping[str, Any]) -> tuple[str, ...]:
    primary = nmm_move_base(move)
    capture = move.get("capture")
    return (primary,) if capture is None else (primary, f"x{capture}")


def _extract_game_prefix_bytes(
    raw: bytes,
    *,
    relative_path: str,
) -> tuple[str, ExtractedHumanPrefix | None, str | None]:
    file_digest = hashlib.sha256(raw).hexdigest()
    file_size = len(raw)
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "invalid", None, "invalid_json"
    if not isinstance(record, Mapping):
        return "invalid", None, "record_not_object"
    moves = record.get("moves")
    if not isinstance(moves, list):
        return "invalid", None, "moves_not_array"
    if len(moves) < PREFIX_LOGICAL_PLIES_V2:
        return "short", None, None
    session_id = record.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return "invalid", None, "missing_session_id"

    board = BoardState.new_game()
    turns: list[tuple[str, ...]] = []
    for index, raw_move in enumerate(moves[:PREFIX_LOGICAL_PLIES_V2]):
        if not isinstance(raw_move, Mapping):
            return "invalid", None, f"move_{index}_not_object"
        if raw_move.get("board_fen_before") != board.to_fen_string():
            return "invalid", None, f"move_{index}_fen_mismatch"
        if raw_move.get("color") != board.turn:
            return "invalid", None, f"move_{index}_color_mismatch"
        expected = {
            "from": raw_move.get("from"),
            "to": raw_move.get("to"),
            "capture": raw_move.get("capture"),
        }
        matching = [
            move
            for move in get_all_legal_moves(board)
            if all(move.get(key) == value for key, value in expected.items())
        ]
        if len(matching) != 1:
            return "invalid", None, f"move_{index}_illegal_or_ambiguous"
        move = matching[0]
        if raw_move.get("notation") != _move_notation(move):
            return "invalid", None, f"move_{index}_notation_mismatch"
        turns.append(_logical_turn(move))
        board = board.apply_move(move)
        if index + 1 < PREFIX_LOGICAL_PLIES_V2 and is_terminal(board)[0]:
            return "invalid", None, f"move_{index}_early_terminal"

    winner = record.get("winner")
    if winner not in {"W", "B", None}:
        return "invalid", None, "unsupported_winner"
    action_tokens = tuple(token for turn in turns for token in turn)
    final_nmm_fen = board.to_fen_string()
    extracted = ExtractedHumanPrefix(
        relative_path=relative_path,
        file_size=file_size,
        file_sha256=file_digest,
        session_sha256=canonical_sha256(
            {"domain": "playok-session-id-v1", "session_id": session_id}
        ),
        source=str(record.get("source", "")),
        source_type=str(record.get("source_type", "")),
        winner=winner,
        logical_turns=tuple(turns),
        action_tokens=action_tokens,
        final_nmm_fen=final_nmm_fen,
        final_ring16_fen=ring16_canonical_fen(final_nmm_fen),
    )
    return "eligible", extracted, None


def _read_game(
    root: Path,
    path: Path,
) -> tuple[dict[str, Any], ExtractedHumanPrefix | None, str | None]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return (
            {
                "relative_path": path.relative_to(root).as_posix(),
                "status": "invalid",
                "error": f"read_error:{type(exc).__name__}",
            },
            None,
            "read_error",
        )
    relative = path.relative_to(root).as_posix()
    status, extracted, error = _extract_game_prefix_bytes(
        raw,
        relative_path=relative,
    )
    manifest = {
        "relative_path": relative,
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "status": status,
        "error": error,
    }
    return manifest, extracted, error


def _book_overlap_sets(
    book_audit_path: str | Path,
) -> dict[str, set[str]]:
    try:
        payload = json.loads(Path(book_audit_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LayeredHumanAuditError("cannot read Book audit evidence") from exc
    verify_layered_book_audit(payload)
    exact: set[str] = set()
    fens: set[str] = set()
    orbits: set[str] = set()
    for entry in payload["named_book_variations"]["entries"]:
        for record in entry["prefix_records"]:
            exact.add(record["exact_history_sha256"])
            fens.add(record["final"]["nmm_fen"])
            orbits.add(record["final"]["ring16_canonical_fen"])
    for record in payload["oracle_query_book"]["complete_histories"]:
        exact.add(canonical_sha256(record["action_tokens"]))
        fens.add(record["nmm_fen"])
        orbits.add(record["ring16_canonical_fen"])
    return {"exact": exact, "fen": fens, "orbit": orbits}


def _winner_key(winner: str | None) -> str:
    return {"W": "white_wins", "B": "black_wins", None: "draws"}[winner]


def _history_record(
    history_identity: str,
    group: _HistoryGroup,
    *,
    book: Mapping[str, set[str]],
) -> dict[str, Any]:
    return {
        "schema_version": HUMAN_HISTORY_SCHEMA,
        "history_identity": history_identity,
        "logical_ply_count": PREFIX_LOGICAL_PLIES_V2,
        "logical_plies_by_side": list(PREFIX_LOGICAL_PLIES_BY_SIDE_V2),
        "logical_turns": [list(turn) for turn in group.logical_turns],
        "action_tokens": list(group.action_tokens),
        "occurrence_count": group.occurrence_count,
        "distinct_game_count": len(group.session_ids),
        "side_roles": {
            "white": "first_player",
            "black": "second_player",
            "white_first_game_count": len(group.session_ids),
            "black_second_game_count": len(group.session_ids),
        },
        "results": {
            "white_wins": group.results["white_wins"],
            "black_wins": group.results["black_wins"],
            "draws": group.results["draws"],
        },
        "final": {
            "nmm_fen": group.final_nmm_fen,
            "ring16_canonical_fen": group.final_ring16_fen,
        },
        "overlap": {
            "book_exact_history": history_identity in book["exact"],
            "book_final_fen": group.final_nmm_fen in book["fen"],
            "book_ring16_orbit": group.final_ring16_fen in book["orbit"],
            "perfect_db": "pending",
        },
    }


def _concentration(
    ordered: Sequence[Mapping[str, Any]],
    total_games: int,
    sizes: Iterable[int],
) -> list[dict[str, Any]]:
    result = []
    for size in sizes:
        games = sum(
            int(record["distinct_game_count"])
            for record in ordered[:size]
        )
        result.append(
            {
                "top_history_count": size,
                "covered_games": games,
                "share_of_eligible_games": (
                    games / total_games if total_games else 0.0
                ),
            }
        )
    return result


def audit_human_prefix_histories(
    games_directory: str | Path,
    *,
    manifest_path: str | Path,
    ledger_path: str | Path,
    book_audit_path: str | Path,
    worker_count: int = 32,
) -> dict[str, Any]:
    """Audit every real recursive JSONL record and freeze a full local ledger."""
    root = Path(games_directory).resolve()
    if not root.is_dir():
        raise LayeredHumanAuditError("human game directory is unavailable")
    manifest_target = Path(manifest_path).resolve()
    ledger_target = Path(ledger_path).resolve()
    if manifest_target.exists() or ledger_target.exists():
        raise LayeredHumanAuditError("HumanDB audit output already exists")
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    ledger_target.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob("*.jsonl"))
    if not files:
        raise LayeredHumanAuditError("human game directory has no JSONL files")
    book = _book_overlap_sets(book_audit_path)

    status_counts: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()
    groups: dict[str, _HistoryGroup] = {}
    manifest_temporary = manifest_target.with_suffix(
        manifest_target.suffix + ".tmp"
    )
    if manifest_temporary.exists():
        raise LayeredHumanAuditError("HumanDB manifest temporary file exists")

    try:
        with manifest_temporary.open("xb") as manifest_handle:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = executor.map(
                    lambda path: _read_game(root, path),
                    files,
                    chunksize=64,
                )
                for manifest, extracted, error in results:
                    manifest_handle.write(canonical_json_bytes(manifest) + b"\n")
                    status_counts[manifest["status"]] += 1
                    if error:
                        invalid_reasons[error] += 1
                    if extracted is None:
                        continue
                    source_counts[extracted.source] += 1
                    source_type_counts[extracted.source_type] += 1
                    history_identity = canonical_sha256(
                        list(extracted.action_tokens)
                    )
                    group = groups.get(history_identity)
                    if group is None:
                        group = _HistoryGroup(
                            logical_turns=extracted.logical_turns,
                            action_tokens=extracted.action_tokens,
                            final_nmm_fen=extracted.final_nmm_fen,
                            final_ring16_fen=extracted.final_ring16_fen,
                        )
                        groups[history_identity] = group
                    elif (
                        group.logical_turns != extracted.logical_turns
                        or group.final_nmm_fen != extracted.final_nmm_fen
                        or group.final_ring16_fen
                        != extracted.final_ring16_fen
                    ):
                        raise LayeredHumanAuditError(
                            "same human history identity has different state"
                        )
                    group.occurrence_count += 1
                    group.session_ids.add(extracted.session_sha256)
                    group.results[_winner_key(extracted.winner)] += 1
            manifest_handle.flush()
            os.fsync(manifest_handle.fileno())
        os.replace(manifest_temporary, manifest_target)
    except BaseException:
        if manifest_temporary.exists():
            manifest_temporary.unlink()
        raise

    history_records = [
        _history_record(identity, group, book=book)
        for identity, group in groups.items()
    ]
    history_records.sort(
        key=lambda record: (
            -int(record["occurrence_count"]),
            str(record["history_identity"]),
        )
    )
    ledger_header = {
        "schema_version": HUMAN_HISTORY_LEDGER_SCHEMA,
        "logical_ply_count": PREFIX_LOGICAL_PLIES_V2,
        "logical_plies_by_side": list(PREFIX_LOGICAL_PLIES_BY_SIDE_V2),
        "record_count": len(history_records),
        "ordering": "occurrence_count_desc_then_history_identity",
    }
    ledger_temporary = ledger_target.with_suffix(ledger_target.suffix + ".tmp")
    if ledger_temporary.exists():
        raise LayeredHumanAuditError("HumanDB ledger temporary file exists")
    try:
        with ledger_temporary.open("xb") as ledger_handle:
            ledger_handle.write(canonical_json_bytes(ledger_header) + b"\n")
            for record in history_records:
                ledger_handle.write(canonical_json_bytes(record) + b"\n")
            ledger_handle.flush()
            os.fsync(ledger_handle.fileno())
        os.replace(ledger_temporary, ledger_target)
    except BaseException:
        if ledger_temporary.exists():
            ledger_temporary.unlink()
        raise

    frequency = Counter(
        int(record["occurrence_count"]) for record in history_records
    )
    eligible_games = status_counts["eligible"]
    repeated = [
        record
        for record in history_records
        if int(record["occurrence_count"]) >= 2
    ]
    distinct_sessions = {
        session
        for group in groups.values()
        for session in group.session_ids
    }
    book_overlap = {
        field: sum(
            bool(record["overlap"][field]) for record in history_records
        )
        for field in (
            "book_exact_history",
            "book_final_fen",
            "book_ring16_orbit",
        )
    }
    return {
        "source_file_count": len(files),
        "source_manifest": {
            "path_lookup_key": "human_db_prefix12_source_manifest_path",
            "byte_length": manifest_target.stat().st_size,
            "sha256": _file_sha256(manifest_target),
        },
        "history_ledger": {
            "path_lookup_key": "human_db_prefix12_history_ledger_path",
            "byte_length": ledger_target.stat().st_size,
            "sha256": _file_sha256(ledger_target),
            "schema_version": HUMAN_HISTORY_LEDGER_SCHEMA,
            "history_count": len(history_records),
        },
        "file_status_counts": [
            {"status": status, "count": status_counts[status]}
            for status in sorted(status_counts)
        ],
        "invalid_reason_counts": [
            {"reason": reason, "count": invalid_reasons[reason]}
            for reason in sorted(invalid_reasons)
        ],
        "source_counts": [
            {"source": source, "count": source_counts[source]}
            for source in sorted(source_counts)
        ],
        "source_type_counts": [
            {
                "source_type": source_type,
                "count": source_type_counts[source_type],
            }
            for source_type in sorted(source_type_counts)
        ],
        "eligible_record_count": eligible_games,
        "distinct_eligible_game_count": len(distinct_sessions),
        "unique_exact_history_count": len(history_records),
        "frequency_distribution": [
            {
                "occurrences": occurrences,
                "history_count": frequency[occurrences],
            }
            for occurrences in sorted(frequency)
        ],
        "maximum_history_frequency": (
            int(history_records[0]["occurrence_count"])
            if history_records
            else 0
        ),
        "repeated_history_count": len(repeated),
        "games_covered_by_repeated_histories": sum(
            int(record["distinct_game_count"]) for record in repeated
        ),
        "concentration": _concentration(
            history_records,
            eligible_games,
            (1, 10, 64, 100, 1000),
        ),
        "book_overlap_history_counts": book_overlap,
        "highest_frequency_histories": [
            {
                "history_identity": record["history_identity"],
                "action_tokens": record["action_tokens"],
                "occurrence_count": record["occurrence_count"],
                "distinct_game_count": record["distinct_game_count"],
                "results": record["results"],
                "final": record["final"],
                "overlap": record["overlap"],
            }
            for record in history_records[:20]
        ],
        "selection_status": (
            "frequency evidence only; no threshold or corpus membership frozen"
        ),
    }


def build_layered_human_audit(
    *,
    generator_commit: str,
    sqlite_snapshot: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    imported_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot_identity = sqlite_snapshot.get("snapshot_identity")
    if not isinstance(snapshot_identity, str):
        raise LayeredHumanAuditError("snapshot identity is missing")
    source_identity_body = {
        "kind": "human_db_snapshot",
        "path_lookup_key": "human_db_prefix12_snapshot_path",
        "snapshot_sha256": sqlite_snapshot["snapshot"]["sha256"],
        "snapshot_byte_length": sqlite_snapshot["snapshot"]["byte_length"],
        "schema_sha256": sqlite_snapshot["snapshot"]["schema_sha256"],
        "row_counts": sqlite_snapshot["snapshot"]["row_counts"],
        "meta": sqlite_snapshot["snapshot"]["meta"],
    }
    source_identity = {
        "kind": "human_db",
        "identity": source_identity_body,
        "identity_sha256": canonical_sha256(source_identity_body),
    }
    body = {
        "schema_version": LAYERED_HUMAN_AUDIT_SCHEMA,
        "status": "source-only-needs-decision",
        "candidate_loaded": False,
        "games_played": 0,
        "target": {
            "prefix_schema": LAYERED_PREFIX_SCHEMA,
            "logical_ply_count": PREFIX_LOGICAL_PLIES_V2,
            "logical_plies_by_side": list(
                PREFIX_LOGICAL_PLIES_BY_SIDE_V2
            ),
        },
        "generator": {
            "algorithm": "genuine-playok-history-audit-v1",
            "nmm_llm_commit": generator_commit,
        },
        "sqlite_snapshot": dict(sqlite_snapshot),
        "source_identity": source_identity,
        "raw_game_source": {
            **dict(source_audit),
            "imported_manifest": dict(imported_manifest),
            "scope": (
                "recursive current PlayOK JSONL sample at audit time; "
                "not a claim about all human players"
            ),
        },
        "label_policy": {
            "human_frequencies_and_outcomes": "usable",
            "unversioned_historical_malom_columns": "not_labels",
        },
        "decision": {
            "final_corpus_frozen": False,
            "human_frequency_threshold_frozen": False,
            "perfect_db_overlap": "pending",
            "synthetic_per_ply_chaining_allowed": False,
        },
    }
    return {**body, "audit_identity": canonical_sha256(body)}


def verify_layered_human_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "status",
        "candidate_loaded",
        "games_played",
        "target",
        "generator",
        "sqlite_snapshot",
        "source_identity",
        "raw_game_source",
        "label_policy",
        "decision",
        "audit_identity",
    }
    if set(payload) != expected:
        raise LayeredHumanAuditError("HumanDB audit top-level fields drifted")
    if (
        payload["schema_version"] != LAYERED_HUMAN_AUDIT_SCHEMA
        or payload["status"] != "source-only-needs-decision"
        or payload["candidate_loaded"] is not False
        or payload["games_played"] != 0
    ):
        raise LayeredHumanAuditError("HumanDB audit scope drifted")
    body = dict(payload)
    identity = body.pop("audit_identity")
    if canonical_sha256(body) != identity:
        raise LayeredHumanAuditError("HumanDB audit identity mismatch")
    snapshot = payload["sqlite_snapshot"]
    snapshot_body = dict(snapshot)
    snapshot_identity = snapshot_body.pop("snapshot_identity")
    if canonical_sha256(snapshot_body) != snapshot_identity:
        raise LayeredHumanAuditError("HumanDB snapshot identity mismatch")
    if (
        snapshot["snapshot"]["quick_check"] != ["ok"]
        or snapshot["source"]["sidecars_deleted"] is not False
    ):
        raise LayeredHumanAuditError("HumanDB snapshot safety evidence drifted")
    raw = payload["raw_game_source"]
    if raw["selection_status"] != (
        "frequency evidence only; no threshold or corpus membership frozen"
    ):
        raise LayeredHumanAuditError("HumanDB selection boundary drifted")
    if payload["decision"] != {
        "final_corpus_frozen": False,
        "human_frequency_threshold_frozen": False,
        "perfect_db_overlap": "pending",
        "synthetic_per_ply_chaining_allowed": False,
    }:
        raise LayeredHumanAuditError("HumanDB decision boundary drifted")
    if not math.isfinite(
        sum(
            float(item["share_of_eligible_games"])
            for item in raw["concentration"]
        )
    ):
        raise LayeredHumanAuditError("HumanDB concentration is non-finite")
    return {
        "snapshot_sha256": snapshot["snapshot"]["sha256"],
        "source_files": raw["source_file_count"],
        "eligible_games": raw["eligible_record_count"],
        "unique_histories": raw["unique_exact_history_count"],
        "repeated_histories": raw["repeated_history_count"],
        "maximum_frequency": raw["maximum_history_frequency"],
    }
