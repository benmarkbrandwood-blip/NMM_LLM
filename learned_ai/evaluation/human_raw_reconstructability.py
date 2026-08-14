"""Read-only F0-D0 audit for the raw human-game corpus.

The audit deliberately works from source JSONL records before consulting the
aggregate HumanDB. It never opens a model, invokes search, plays a game, or
writes a database. SQLite inputs are opened through immutable read-only URIs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from game.board import POSITIONS, BoardState
from game.draw_rules import StandardDrawTracker
from game.rules import get_all_legal_moves, get_game_phase, terminal_result


SCHEMA_VERSION = "nmm.f0-d0-human-raw-reconstructability.v1"
AUDIT_ID = "f0-d0-human-raw-reconstructability-v1"
RULESET_ID = "nmm-training-core@2"

_DIMENSIONS = ("history", "player", "source", "result", "condition")
_RECOVERABLE = "recoverable"
_PARTIAL = "partial"
_NOT_RECOVERABLE = "not_recoverable"
_RESULT_TO_OUTCOME = {
    "1-0": "W",
    "0-1": "B",
    "1/2-1/2": "D",
}


class F0D0AuditError(RuntimeError):
    """Raised when the source corpus cannot be audited deterministically."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's stable, finite JSON representation."""
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise F0D0AuditError("audit payload is not canonical JSON") from exc
    return rendered.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return SHA-256 over canonical JSON."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _player_key(source: str, identifier: str) -> str:
    payload = f"f0-d0-player-key-v1\0{source}\0{identifier}".encode()
    return hashlib.sha256(payload).hexdigest()


def _move_notation(move: Mapping[str, Any]) -> str:
    source = move.get("from")
    target = move.get("to")
    base = str(target) if source is None else f"{source}-{target}"
    capture = move.get("capture")
    return base if capture is None else f"{base}x{capture}"


def _position_key(board: BoardState) -> tuple[tuple[str, ...], str]:
    return tuple(board.positions[position] for position in POSITIONS), board.turn


def _failed_replay(code: str, replayed: int) -> dict[str, Any]:
    return {
        "status": "failed",
        "failure_code": code,
        "logical_plies_replayed": replayed,
        "history_identity": None,
        "final_fen_sha256": None,
        "final_no_progress_plies": None,
        "final_repetition_current_count": None,
        "maximum_repetition_count": None,
        "independent_outcome": None,
        "independent_reason": None,
    }


def _replay_moves(moves: Sequence[Any]) -> dict[str, Any]:
    board = BoardState.new_game()
    tracker = StandardDrawTracker(board)
    normalized: list[dict[str, Any]] = []
    independent_outcome: str | None = None
    independent_reason: str | None = None

    for index, raw_move in enumerate(moves):
        if independent_outcome is not None:
            return _failed_replay(
                f"history.move_{index}_after_rules_terminal",
                index,
            )
        if not isinstance(raw_move, Mapping):
            return _failed_replay(f"history.move_{index}_not_object", index)
        if raw_move.get("board_fen_before") != board.to_fen_string():
            return _failed_replay(f"history.move_{index}_fen_mismatch", index)
        if raw_move.get("color") != board.turn:
            return _failed_replay(f"history.move_{index}_color_mismatch", index)
        if raw_move.get("turn") != index // 2 + 1:
            return _failed_replay(f"history.move_{index}_turn_mismatch", index)

        expected_type = get_game_phase(board, board.turn)
        if raw_move.get("type") != expected_type:
            return _failed_replay(f"history.move_{index}_type_mismatch", index)
        expected = {
            "from": raw_move.get("from"),
            "to": raw_move.get("to"),
            "capture": raw_move.get("capture"),
        }
        matching = [
            move
            for move in get_all_legal_moves(board)
            if all(move.get(field) == value for field, value in expected.items())
        ]
        if len(matching) != 1:
            return _failed_replay(
                f"history.move_{index}_illegal_or_ambiguous",
                index,
            )
        move = matching[0]
        if raw_move.get("notation") != _move_notation(move):
            return _failed_replay(
                f"history.move_{index}_notation_mismatch",
                index,
            )

        after = board.apply_move(move)
        draw_reason = tracker.observe(board, move, after)
        is_terminal, winner, terminal_reason = terminal_result(after)
        normalized.append(
            {
                "from": move.get("from"),
                "to": move.get("to"),
                "capture": move.get("capture"),
            }
        )
        board = after

        if is_terminal:
            independent_outcome = winner
            independent_reason = f"rules_{terminal_reason}"
        elif draw_reason is not None:
            independent_outcome = "D"
            independent_reason = (
                "draw_threefold_repetition"
                if draw_reason == "repetition"
                else "draw_no_progress"
            )

    state = tracker.snapshot()
    repetition_counts = dict(state.repetition_counts)
    status = independent_reason or "ongoing"
    return {
        "status": status,
        "failure_code": None,
        "logical_plies_replayed": len(moves),
        "history_identity": canonical_sha256(normalized),
        "final_fen_sha256": hashlib.sha256(
            board.to_fen_string().encode("utf-8")
        ).hexdigest(),
        "final_no_progress_plies": state.no_progress_plies,
        "final_repetition_current_count": repetition_counts.get(
            _position_key(board),
            0,
        ),
        "maximum_repetition_count": max(repetition_counts.values(), default=0),
        "independent_outcome": independent_outcome,
        "independent_reason": independent_reason,
    }


def _present_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _optional_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _source_details(
    record: Mapping[str, Any],
    *,
    relative_path: str,
    imported_at: str | None,
) -> tuple[str, dict[str, bool]]:
    details = {
        "platform": _present_string(record.get("source")),
        "source_type": _present_string(record.get("source_type")),
        "current_raw_file": bool(relative_path),
        "import_timestamp": _present_string(imported_at),
        "exact_import_batch": _present_string(record.get("import_batch")),
        "upstream_file_identity": any(
            _present_string(record.get(field))
            for field in (
                "source_file_sha256",
                "original_file_sha256",
                "upstream_file_sha256",
            )
        ),
    }
    if all(details.values()):
        return _RECOVERABLE, details
    if all(
        details[field]
        for field in (
            "platform",
            "source_type",
            "current_raw_file",
            "import_timestamp",
        )
    ):
        return _PARTIAL, details
    return _NOT_RECOVERABLE, details


def _condition_details(record: Mapping[str, Any]) -> tuple[str, dict[str, bool]]:
    details = {
        "color_assignment": (
            _present_string(record.get("white_player"))
            and _present_string(record.get("black_player"))
        ),
        "ui_orientation": any(
            _present_string(record.get(field))
            for field in ("ui_orientation", "board_orientation", "orientation")
        ),
        "time_control": any(
            _present_string(record.get(field))
            for field in ("time_control", "clock", "clock_type")
        ),
        "exact_rules_variant": any(
            _present_string(record.get(field))
            for field in ("rules_variant", "game_type", "ruleset")
        ),
        "date": _present_string(record.get("date")),
        "white_elo": _optional_int(record.get("white_elo")),
        "black_elo": _optional_int(record.get("black_elo")),
    }
    required = (
        "color_assignment",
        "ui_orientation",
        "time_control",
        "exact_rules_variant",
    )
    if all(details[field] for field in required):
        return _RECOVERABLE, details
    if details["color_assignment"]:
        return _PARTIAL, details
    return _NOT_RECOVERABLE, details


def _recorded_outcome(record: Mapping[str, Any]) -> tuple[str | None, str | None]:
    result_raw = record.get("result_raw")
    outcome = _RESULT_TO_OUTCOME.get(result_raw)
    if outcome is None:
        return None, "result.unsupported_result_raw"
    winner = record.get("winner")
    expected_winner = None if outcome == "D" else outcome
    if winner != expected_winner:
        return None, "result.winner_result_raw_mismatch"
    return outcome, None


def audit_game_record(
    record: Mapping[str, Any],
    *,
    relative_path: str,
    imported_at: str | None,
) -> dict[str, Any]:
    """Audit one decoded raw game without reading external state."""
    session_id = record.get("session_id")
    if not _present_string(session_id):
        session_id = None
    source = str(record.get("source", ""))
    white_player = record.get("white_player")
    black_player = record.get("black_player")
    white_present = _present_string(white_player)
    black_present = _present_string(black_player)
    if white_present and black_present:
        player_status = _RECOVERABLE
    elif white_present or black_present:
        player_status = _PARTIAL
    else:
        player_status = _NOT_RECOVERABLE

    player_keys = []
    if white_present:
        player_keys.append(_player_key(source, str(white_player).strip()))
    if black_present:
        player_keys.append(_player_key(source, str(black_player).strip()))

    moves = record.get("moves")
    if not isinstance(moves, list):
        replay = _failed_replay("history.moves_not_array", 0)
        move_count = None
    else:
        replay = _replay_moves(moves)
        move_count = len(moves)
    history_status = (
        _RECOVERABLE if replay["status"] != "failed" else _NOT_RECOVERABLE
    )

    recorded_outcome, result_failure = _recorded_outcome(record)
    independent_outcome = replay.get("independent_outcome")
    source_basis = next(
        (
            str(record[field])
            for field in ("termination_reason", "result_reason", "draw_reason")
            if _present_string(record.get(field))
        ),
        None,
    )
    if result_failure is not None or history_status == _NOT_RECOVERABLE:
        result_status = _NOT_RECOVERABLE
        comparison = "unavailable"
    elif independent_outcome is not None:
        if recorded_outcome == independent_outcome:
            result_status = _RECOVERABLE
            comparison = "agree"
        else:
            result_status = _NOT_RECOVERABLE
            comparison = "disagree"
            result_failure = "result.independent_replay_disagrees"
    elif source_basis is not None:
        result_status = _PARTIAL
        comparison = "unverifiable_nonterminal"
    else:
        result_status = _PARTIAL
        comparison = "unverifiable_nonterminal"
        result_failure = "result.terminal_basis_missing"

    source_status, source_details = _source_details(
        record,
        relative_path=relative_path,
        imported_at=imported_at,
    )
    condition_status, condition_details = _condition_details(record)
    failure_codes: list[str] = []
    if replay.get("failure_code") is not None:
        failure_codes.append(str(replay["failure_code"]))
    if player_status != _RECOVERABLE:
        failure_codes.append("player.stable_pair_missing")
    if result_failure is not None:
        failure_codes.append(result_failure)

    dimensions = {
        "history": history_status,
        "player": player_status,
        "source": source_status,
        "result": result_status,
        "condition": condition_status,
    }
    behavior_eligible = (
        history_status == _RECOVERABLE
        and player_status == _RECOVERABLE
        and source_details["platform"]
        and condition_details["color_assignment"]
        and bool(move_count)
    )
    return {
        "session_id": session_id,
        "canonical_file": relative_path,
        "duplicate_files": [],
        "file_sha256": None,
        "imported_at": imported_at,
        "source": source,
        "date": record.get("date") if _present_string(record.get("date")) else None,
        "move_count": move_count,
        "recorded_outcome": recorded_outcome,
        "source_result_basis": source_basis,
        "player_keys": player_keys,
        "dimensions": dimensions,
        "source_fields": source_details,
        "condition_fields": condition_details,
        "replay": replay,
        "result_comparison": comparison,
        "behavior_replay_eligible": behavior_eligible,
        "outcome_analysis_eligible": (
            behavior_eligible and result_status == _RECOVERABLE
        ),
        "all_five_dimensions_recoverable": all(
            dimensions[field] == _RECOVERABLE for field in _DIMENSIONS
        ),
        "failure_codes": sorted(set(failure_codes)),
    }


def _filename_session_id(path: Path) -> str | None:
    if not path.stem.startswith("human_"):
        return None
    value = path.stem.removeprefix("human_")
    return value or None


def _invalid_game_audit(
    session_id: str | None,
    relative_path: str,
    imported_at: str | None,
    code: str,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "canonical_file": relative_path,
        "duplicate_files": [],
        "file_sha256": None,
        "imported_at": imported_at,
        "source": None,
        "date": None,
        "move_count": None,
        "recorded_outcome": None,
        "source_result_basis": None,
        "player_keys": [],
        "dimensions": {field: _NOT_RECOVERABLE for field in _DIMENSIONS},
        "source_fields": {},
        "condition_fields": {},
        "replay": _failed_replay(code, 0),
        "result_comparison": "unavailable",
        "behavior_replay_eligible": False,
        "outcome_analysis_eligible": False,
        "all_five_dimensions_recoverable": False,
        "failure_codes": [code],
    }


def _audit_file(
    path: Path,
    *,
    repository_root: Path,
    imported: Mapping[str, str],
) -> dict[str, Any]:
    relative_path = path.relative_to(repository_root).as_posix()
    expected_session = _filename_session_id(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        code = f"file.read_error_{type(exc).__name__}"
        return {
            "relative_path": relative_path,
            "byte_length": None,
            "sha256": None,
            "session_id": expected_session,
            "file_status": "invalid",
            "file_failure": code,
            "game_audit": _invalid_game_audit(
                expected_session,
                relative_path,
                imported.get(expected_session or ""),
                code,
            ),
        }

    file_sha256 = hashlib.sha256(raw).hexdigest()
    lines = [line for line in raw.splitlines() if line.strip()]
    code: str | None = None
    record: Any = None
    if len(lines) != 1:
        code = "file.record_count_not_one"
    else:
        try:
            record = json.loads(lines[0])
        except (UnicodeDecodeError, json.JSONDecodeError):
            code = "file.invalid_json"
    if code is None and not isinstance(record, Mapping):
        code = "file.record_not_object"

    record_session = record.get("session_id") if isinstance(record, Mapping) else None
    session_id = record_session if _present_string(record_session) else expected_session
    if code is None and record_session != expected_session:
        code = "file.session_filename_mismatch"

    imported_at = imported.get(str(session_id)) if session_id is not None else None
    if code is None:
        game_audit = audit_game_record(
            record,
            relative_path=relative_path,
            imported_at=imported_at,
        )
        file_status = "parsed"
    else:
        game_audit = _invalid_game_audit(
            str(session_id) if session_id is not None else None,
            relative_path,
            imported_at,
            code,
        )
        file_status = "invalid"
    game_audit["file_sha256"] = file_sha256
    return {
        "relative_path": relative_path,
        "byte_length": len(raw),
        "sha256": file_sha256,
        "session_id": session_id,
        "file_status": file_status,
        "file_failure": code,
        "game_audit": game_audit,
    }


def reconcile_file_audits(
    file_audits: Sequence[Mapping[str, Any]],
    *,
    imported: Mapping[str, str],
) -> dict[str, Any]:
    """Collapse byte-identical duplicate files into unique game sessions."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    unassigned: list[dict[str, Any]] = []
    input_files: list[dict[str, Any]] = []
    for file_audit in file_audits:
        input_files.append(
            {
                "relative_path": file_audit.get("relative_path"),
                "byte_length": file_audit.get("byte_length"),
                "sha256": file_audit.get("sha256"),
                "session_id": file_audit.get("session_id"),
                "status": file_audit.get("file_status", "parsed"),
                "failure": file_audit.get("file_failure"),
            }
        )
        session_id = file_audit.get("session_id")
        if not _present_string(session_id):
            unassigned.append(input_files[-1])
            continue
        grouped[str(session_id)].append(file_audit)

    game_records: list[dict[str, Any]] = []
    duplicate_sessions = 0
    duplicate_extra_files = 0
    for session_id in sorted(grouped):
        entries = sorted(grouped[session_id], key=lambda row: str(row["relative_path"]))
        hashes = {entry.get("sha256") for entry in entries}
        if len(hashes) != 1:
            raise F0D0AuditError(
                f"conflicting duplicate session bytes: {session_id}"
            )
        if len(entries) > 1:
            duplicate_sessions += 1
            duplicate_extra_files += len(entries) - 1
        canonical = entries[0]
        audit = copy.deepcopy(canonical["game_audit"])
        audit["canonical_file"] = canonical["relative_path"]
        audit["duplicate_files"] = [
            entry["relative_path"] for entry in entries[1:]
        ]
        audit["file_sha256"] = canonical["sha256"]
        audit["imported_at"] = imported.get(session_id)
        if session_id not in imported:
            audit["failure_codes"] = sorted(
                set(audit["failure_codes"] + ["source.import_manifest_missing"])
            )
            audit["dimensions"]["source"] = _NOT_RECOVERABLE
        game_records.append(audit)

    raw_ids = set(grouped)
    imported_ids = set(imported)
    return {
        "input_files": sorted(input_files, key=lambda row: str(row["relative_path"])),
        "game_records": game_records,
        "unassigned_files": unassigned,
        "file_occurrences": len(file_audits),
        "unique_sessions": len(grouped),
        "duplicate_sessions": duplicate_sessions,
        "duplicate_extra_files": duplicate_extra_files,
        "raw_session_ids_not_imported": sorted(raw_ids - imported_ids),
        "imported_ids_without_raw_session": sorted(imported_ids - raw_ids),
    }


def _relative_db_path(file_path: str) -> str | None:
    normalized = file_path.replace("\\", "/")
    marker = "/data/human_games/"
    if marker not in normalized:
        return None
    return "data/human_games/" + normalized.split(marker, 1)[1]


def _session_from_relative(relative_path: str | None) -> str | None:
    if relative_path is None:
        return None
    stem = PurePosixPath(relative_path).stem
    if not stem.startswith("human_"):
        return None
    return stem.removeprefix("human_") or None


def _path_record(path: Path, repository_root: Path, role: str) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(repository_root).as_posix(),
        "role": role,
        "byte_length": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def audit_human_db(
    database_path: str | Path,
    *,
    repository_root: str | Path,
    current_input_files: Sequence[Mapping[str, Any]],
    role: str,
) -> dict[str, Any]:
    """Inspect one HumanDB without writing it or consuming aggregate labels."""
    root = Path(repository_root).resolve()
    database = Path(database_path).resolve()
    if not database.is_file():
        raise F0D0AuditError(f"{role} HumanDB is unavailable")

    observed_paths = [database]
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(database) + suffix)
        if sidecar.exists():
            observed_paths.append(sidecar)
    before_files = [
        _path_record(path, root, role if path == database else f"{role}_sidecar")
        for path in observed_paths
    ]
    before_identity = canonical_sha256(before_files)

    uri = f"file:{database.as_posix()}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' ORDER BY name"
            )
        ]
        required = {"meta", "moves", "positions", "processed_files"}
        if not required.issubset(tables):
            raise F0D0AuditError(f"{role} HumanDB schema is incomplete")
        metadata = {
            str(key): str(value)
            for key, value in connection.execute(
                "SELECT key, value FROM meta ORDER BY key"
            )
        }
        table_counts = {
            table: int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in ("positions", "moves", "processed_files")
        }
        processed_rows_raw = connection.execute(
            "SELECT file_path, sha256, games_found FROM processed_files "
            "ORDER BY file_path"
        ).fetchall()
    finally:
        try:
            connection.close()
        except UnboundLocalError:
            pass

    current_by_path = {
        str(row["relative_path"]): row
        for row in current_input_files
        if row.get("relative_path") is not None
    }
    processed_rows: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    path_failures: list[dict[str, Any]] = []
    for stored_path, stored_sha256, games_found in processed_rows_raw:
        relative_path = _relative_db_path(str(stored_path))
        session_id = _session_from_relative(relative_path)
        portable = {
            "relative_path": relative_path,
            "stored_path_sha256": hashlib.sha256(
                str(stored_path).encode("utf-8")
            ).hexdigest(),
            "sha256": stored_sha256,
            "games_found": int(games_found),
            "session_id": session_id,
        }
        processed_rows.append(portable)
        if session_id is not None:
            groups[session_id].append(portable)
        current = current_by_path.get(str(relative_path))
        if relative_path is None:
            path_failures.append(
                {
                    "code": "processed_path_not_portable",
                    "stored_path_sha256": portable["stored_path_sha256"],
                }
            )
        elif current is not None and current.get("sha256") != stored_sha256:
            path_failures.append(
                {
                    "code": "processed_hash_differs_from_current_raw",
                    "relative_path": relative_path,
                    "session_id": session_id,
                }
            )

    current_sessions = {
        str(row["session_id"])
        for row in current_input_files
        if _present_string(row.get("session_id"))
    }
    processed_sessions = set(groups)
    duplicate_groups = [rows for rows in groups.values() if len(rows) > 1]
    games_found_sum = sum(int(row["games_found"]) for row in processed_rows)
    metadata_total = int(metadata["total_games"]) if "total_games" in metadata else None
    after_files = [
        _path_record(path, root, role if path == database else f"{role}_sidecar")
        for path in observed_paths
    ]
    after_identity = canonical_sha256(after_files)
    if before_identity != after_identity:
        raise F0D0AuditError(f"{role} HumanDB or sidecar changed during audit")

    return {
        "role": role,
        "files": before_files,
        "files_identity": before_identity,
        "open_mode": "sqlite-uri-mode-ro-immutable-1",
        "quick_check": quick_check,
        "tables": tables,
        "table_counts": table_counts,
        "metadata": metadata,
        "processed_rows_identity": canonical_sha256(processed_rows),
        "processed_games_found_sum": games_found_sum,
        "metadata_total_games_minus_processed_sum": (
            metadata_total - games_found_sum if metadata_total is not None else None
        ),
        "normalized_processed_session_count": len(processed_sessions),
        "processed_session_ids_with_any_game": sum(
            any(int(row["games_found"]) > 0 for row in rows)
            for rows in groups.values()
        ),
        "processed_session_ids_without_game": sum(
            not any(int(row["games_found"]) > 0 for row in rows)
            for rows in groups.values()
        ),
        "duplicate_processed_session_count": len(duplicate_groups),
        "duplicate_processed_extra_rows": sum(
            len(rows) - 1 for rows in duplicate_groups
        ),
        "duplicate_processed_hash_conflicts": sum(
            len({row["sha256"] for row in rows}) > 1
            for rows in duplicate_groups
        ),
        "current_session_ids_not_processed": sorted(
            current_sessions - processed_sessions
        ),
        "processed_session_ids_not_current": sorted(
            processed_sessions - current_sessions
        ),
        "path_or_hash_failures": path_failures,
    }


def _load_imported_manifest(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise F0D0AuditError("imported manifest is invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise F0D0AuditError("imported manifest is not an object")
    imported: dict[str, str] = {}
    invalid: list[str] = []
    for key, value in payload.items():
        if not _present_string(key) or not _present_string(value):
            invalid.append(str(key))
            continue
        try:
            datetime.fromisoformat(str(value))
        except ValueError:
            invalid.append(str(key))
            continue
        imported[str(key)] = str(value)
    if invalid:
        raise F0D0AuditError("imported manifest contains invalid entries")
    return imported, {
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "entry_count": len(imported),
        "entry_identity": canonical_sha256(imported),
        "minimum_timestamp": min(imported.values(), default=None),
        "maximum_timestamp": max(imported.values(), default=None),
        "explicit_batch_id_count": 0,
    }


def _dimension_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension in _DIMENSIONS:
        counts = Counter(str(row["dimensions"][dimension]) for row in records)
        result[dimension] = {
            status: int(counts.get(status, 0))
            for status in (_RECOVERABLE, _PARTIAL, _NOT_RECOVERABLE)
        }
    return result


def _numeric_summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
    }


def _selection_bias_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    dimension: str,
) -> dict[str, Any]:
    recovered = [
        row for row in records if row["dimensions"][dimension] == _RECOVERABLE
    ]
    excluded = [
        row for row in records if row["dimensions"][dimension] != _RECOVERABLE
    ]

    def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        dates = sorted(str(row["date"]) for row in rows if row.get("date") is not None)
        return {
            "games": len(rows),
            "result_counts": dict(
                sorted(Counter(str(row.get("recorded_outcome")) for row in rows).items())
            ),
            "move_count": _numeric_summary(
                [int(row["move_count"]) for row in rows if row.get("move_count") is not None]
            ),
            "date_range": [dates[0], dates[-1]] if dates else [None, None],
        }

    return {
        "recoverable_subset": _summary(recovered),
        "excluded_subset": _summary(excluded),
        "risk": (
            "no_selection_on_this_dimension"
            if not excluded
            else "selection_bias_possible_compare_recorded_strata"
        ),
    }


def _ruleset_input(path: Path, repository_root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise F0D0AuditError("ruleset input is invalid") from exc
    try:
        repetition_count = payload["draw"]["repetition"]["count"]
        no_progress_limit = payload["draw"]["noProgress"]["normalLimit"]
    except (KeyError, TypeError) as exc:
        raise F0D0AuditError("ruleset draw contract is absent") from exc
    if repetition_count != 3 or no_progress_limit != 100:
        raise F0D0AuditError("ruleset differs from StandardDrawTracker")
    return {
        "relative_path": path.relative_to(repository_root).as_posix(),
        "role": "strict_replay_ruleset",
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_identity": canonical_sha256(payload),
        "ruleset_id": RULESET_ID,
        "repetition_count": repetition_count,
        "no_progress_limit": no_progress_limit,
    }


def _count_field(records: Sequence[Mapping[str, Any]], field: str) -> int:
    return sum(bool(row[field]) for row in records)


def build_f0d0_manifest(
    *,
    repository_root: str | Path,
    games_directory: str | Path,
    imported_manifest_path: str | Path,
    active_human_db_path: str | Path,
    archived_human_db_path: str | Path,
    ruleset_path: str | Path,
    source_commit: str,
    worker_count: int = 16,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Build the complete in-memory F0-D0 result from read-only inputs."""
    root = Path(repository_root).resolve()
    games_root = Path(games_directory).resolve()
    imported_path = Path(imported_manifest_path).resolve()
    ruleset = Path(ruleset_path).resolve()
    if not games_root.is_dir():
        raise F0D0AuditError("human-game directory is unavailable")
    if not imported_path.is_file():
        raise F0D0AuditError("imported manifest is unavailable")
    if worker_count <= 0:
        raise F0D0AuditError("worker count must be positive")

    imported, imported_evidence = _load_imported_manifest(imported_path)
    files = sorted(games_root.rglob("*.jsonl"))
    if not files:
        raise F0D0AuditError("human-game directory has no JSONL inputs")
    audit_one = partial(
        _audit_file,
        repository_root=root,
        imported=imported,
    )
    file_audits: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for completed, audit in enumerate(executor.map(audit_one, files), start=1):
            file_audits.append(audit)
            if progress is not None and (
                completed == len(files) or completed % 5_000 == 0
            ):
                progress(completed, len(files))
    reconciled = reconcile_file_audits(file_audits, imported=imported)
    records = reconciled["game_records"]
    input_files = reconciled["input_files"]

    imported_file = {
        "relative_path": imported_path.relative_to(root).as_posix(),
        "role": "imported_manifest",
        "byte_length": imported_evidence["byte_length"],
        "sha256": imported_evidence["sha256"],
    }
    ruleset_input = _ruleset_input(ruleset, root)
    active_db = audit_human_db(
        active_human_db_path,
        repository_root=root,
        current_input_files=input_files,
        role="active_human_db",
    )
    archived_db = audit_human_db(
        archived_human_db_path,
        repository_root=root,
        current_input_files=input_files,
        role="archived_rebuilt_human_db_candidate",
    )

    raw_identity_rows = [
        {
            "relative_path": row["relative_path"],
            "byte_length": row["byte_length"],
            "sha256": row["sha256"],
        }
        for row in input_files
    ]
    session_identity_rows = [
        {
            "session_id": row["session_id"],
            "canonical_file": row["canonical_file"],
            "file_sha256": row["file_sha256"],
            "imported_at": row["imported_at"],
        }
        for row in records
    ]
    raw_files_identity = canonical_sha256(raw_identity_rows)
    session_source_identity = canonical_sha256(session_identity_rows)
    corpus_identity = canonical_sha256(
        {
            "schema_version": "nmm.human-raw-corpus.v1",
            "raw_files_identity": raw_files_identity,
            "session_source_identity": session_source_identity,
            "imported_manifest_sha256": imported_evidence["sha256"],
            "unique_sessions": len(records),
        }
    )

    dimension_counts = _dimension_counts(records)
    failure_counts = Counter(
        code for row in records for code in row.get("failure_codes", [])
    )
    player_keys = sorted(
        {str(key) for row in records for key in row.get("player_keys", [])}
    )
    global_omissions = []
    for code, field_group, field, dimension in (
        (
            "source.exact_import_batch_absent",
            "source_fields",
            "exact_import_batch",
            "source",
        ),
        (
            "source.upstream_file_identity_absent",
            "source_fields",
            "upstream_file_identity",
            "source",
        ),
        (
            "condition.ui_orientation_absent",
            "condition_fields",
            "ui_orientation",
            "condition",
        ),
        (
            "condition.time_control_absent",
            "condition_fields",
            "time_control",
            "condition",
        ),
        (
            "condition.exact_rules_variant_absent",
            "condition_fields",
            "exact_rules_variant",
            "condition",
        ),
    ):
        affected = sum(
            not bool(row.get(field_group, {}).get(field)) for row in records
        )
        global_omissions.append(
            {
                "code": code,
                "dimension": dimension,
                "affected_sessions": affected,
                "attribution": "field absent from each affected raw record",
            }
        )

    counts = {
        "raw_jsonl_file_occurrences": reconciled["file_occurrences"],
        "unique_sessions": reconciled["unique_sessions"],
        "duplicate_sessions": reconciled["duplicate_sessions"],
        "duplicate_extra_file_occurrences": reconciled["duplicate_extra_files"],
        "imported_manifest_entries": len(imported),
        "unassigned_raw_files": len(reconciled["unassigned_files"]),
        "behavior_replay_eligible_sessions": _count_field(
            records,
            "behavior_replay_eligible",
        ),
        "outcome_analysis_eligible_sessions": _count_field(
            records,
            "outcome_analysis_eligible",
        ),
        "all_five_dimensions_recoverable_sessions": _count_field(
            records,
            "all_five_dimensions_recoverable",
        ),
        "independent_result_agreements": sum(
            row["result_comparison"] == "agree" for row in records
        ),
        "independent_result_disagreements": sum(
            row["result_comparison"] == "disagree" for row in records
        ),
        "independent_result_unverifiable_nonterminal": sum(
            row["result_comparison"] == "unverifiable_nonterminal"
            for row in records
        ),
        "unique_player_keys": len(player_keys),
    }
    reconciliation = {
        "raw_session_ids_not_imported": reconciled[
            "raw_session_ids_not_imported"
        ],
        "imported_ids_without_raw_session": reconciled[
            "imported_ids_without_raw_session"
        ],
        "active_human_db": active_db,
        "archived_rebuilt_human_db_candidate": archived_db,
        "explained_counts": {
            "95389_raw_files": (
                "94540 unique sessions plus 849 byte-identical duplicate "
                "file occurrences"
            ),
            "94540_imported_ids": (
                "exact set equality with the 94540 unique raw session IDs"
            ),
            "94429_active_humandb": (
                "active meta.total_games; six above the exact sum of "
                "processed_files.games_found and not a unique-session count"
            ),
            "95221_historical_value": (
                "archived rebuilt candidate meta.total_games over a different "
                "95785-path input inventory containing 396 sessions absent "
                "from the current raw corpus"
            ),
        },
    }
    audit_result_identity = canonical_sha256(
        {
            "records": records,
            "dimension_counts": dimension_counts,
            "counts": counts,
            "reconciliation": reconciliation,
        }
    )
    data_input_files = (
        input_files
        + [imported_file, ruleset_input]
        + active_db["files"]
        + archived_db["files"]
    )
    inputs_identity = canonical_sha256(data_input_files)

    history_recoverable = dimension_counts["history"][_RECOVERABLE]
    player_recoverable = dimension_counts["player"][_RECOVERABLE]
    if history_recoverable == 0 or player_recoverable == 0:
        decision = "fatal_stop_raw_history_or_player_identity_inadequate"
    else:
        decision = "partial_recoverability_source_domain_only"

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "completed_read_only_f0_d0",
        "source_commit": source_commit,
        "decision": decision,
        "claim_boundary": {
            "f0_stage": "F0-D0_only",
            "new_games": 0,
            "search_batches": 0,
            "training_or_updates": False,
            "model_loaded": False,
            "database_write_migration_or_rebuild": False,
            "heldout_pool_2eb04f54_records_read": 0,
            "f0_h0_started": False,
        },
        "rules_contract": ruleset_input,
        "identities": {
            "raw_files_identity": raw_files_identity,
            "session_source_identity": session_source_identity,
            "corpus_identity": corpus_identity,
            "inputs_identity": inputs_identity,
            "audit_result_identity": audit_result_identity,
            "player_identifier_set_identity": canonical_sha256(player_keys),
        },
        "imported_manifest": imported_evidence,
        "counts": counts,
        "dimension_counts": dimension_counts,
        "failure_reason_counts": [
            {"code": code, "count": failure_counts[code]}
            for code in sorted(failure_counts)
        ],
        "global_failure_attribution": global_omissions,
        "selection_bias_risk": {
            "player": _selection_bias_summary(records, dimension="player"),
            "history": _selection_bias_summary(records, dimension="history"),
        },
        "reconciliation": reconciliation,
        "input_files": data_input_files,
        "game_records": records,
        "unassigned_files": reconciled["unassigned_files"],
        "result_disagreements": [
            {
                "session_id": row["session_id"],
                "canonical_file": row["canonical_file"],
                "recorded_outcome": row["recorded_outcome"],
                "independent_outcome": row["replay"]["independent_outcome"],
                "independent_reason": row["replay"]["independent_reason"],
            }
            for row in records
            if row["result_comparison"] == "disagree"
        ],
        "next_gate_impact": {
            "f0_h0_executed": False,
            "behavior_source_domain_support_count": counts[
                "behavior_replay_eligible_sessions"
            ],
            "strict_outcome_support_count": counts[
                "outcome_analysis_eligible_sessions"
            ],
            "transport_claim_available": False,
            "reason": (
                "UI orientation, time control, exact source rules variant, "
                "upstream file identity, and exact import batch are not fully "
                "preserved; F0-H0 must not treat them as observed."
            ),
        },
    }
    return seal_manifest(payload)


def seal_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add a self-verifying identity that excludes only that identity field."""
    sealed = dict(payload)
    sealed.pop("manifest_identity", None)
    sealed["manifest_identity"] = canonical_sha256(sealed)
    return sealed


def verify_manifest(payload: Mapping[str, Any]) -> None:
    """Verify the manifest and every non-circular nested identity."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise F0D0AuditError("manifest schema differs")
    if payload.get("audit_id") != AUDIT_ID:
        raise F0D0AuditError("manifest audit id differs")
    expected = payload.get("manifest_identity")
    if not isinstance(expected, str) or len(expected) != 64:
        raise F0D0AuditError("manifest identity is absent")
    body = dict(payload)
    body.pop("manifest_identity", None)
    if canonical_sha256(body) != expected:
        raise F0D0AuditError("manifest identity differs")

    if "game_records" not in payload:
        return
    records = payload["game_records"]
    input_files = payload["input_files"]
    identities = payload["identities"]
    raw_rows = [
        {
            "relative_path": row["relative_path"],
            "byte_length": row["byte_length"],
            "sha256": row["sha256"],
        }
        for row in input_files
        if row.get("role") is None
    ]
    if canonical_sha256(raw_rows) != identities["raw_files_identity"]:
        raise F0D0AuditError("raw file identity differs")
    session_rows = [
        {
            "session_id": row["session_id"],
            "canonical_file": row["canonical_file"],
            "file_sha256": row["file_sha256"],
            "imported_at": row["imported_at"],
        }
        for row in records
    ]
    if canonical_sha256(session_rows) != identities["session_source_identity"]:
        raise F0D0AuditError("session source identity differs")
    corpus_identity = canonical_sha256(
        {
            "schema_version": "nmm.human-raw-corpus.v1",
            "raw_files_identity": identities["raw_files_identity"],
            "session_source_identity": identities["session_source_identity"],
            "imported_manifest_sha256": payload["imported_manifest"]["sha256"],
            "unique_sessions": len(records),
        }
    )
    if corpus_identity != identities["corpus_identity"]:
        raise F0D0AuditError("corpus identity differs")
    if _dimension_counts(records) != payload["dimension_counts"]:
        raise F0D0AuditError("dimension counts differ")
    if canonical_sha256(input_files) != identities["inputs_identity"]:
        raise F0D0AuditError("input files identity differs")
    player_keys = sorted(
        {str(key) for row in records for key in row.get("player_keys", [])}
    )
    if canonical_sha256(player_keys) != identities["player_identifier_set_identity"]:
        raise F0D0AuditError("player identifier set identity differs")
    audit_result_identity = canonical_sha256(
        {
            "records": records,
            "dimension_counts": payload["dimension_counts"],
            "counts": payload["counts"],
            "reconciliation": payload["reconciliation"],
        }
    )
    if audit_result_identity != identities["audit_result_identity"]:
        raise F0D0AuditError("audit result identity differs")


def write_manifest(payload: Mapping[str, Any], output_path: str | Path) -> None:
    """Write one verified manifest without overwriting prior evidence."""
    verify_manifest(payload)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise F0D0AuditError("manifest output already exists") from exc
