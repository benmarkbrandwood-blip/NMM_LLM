from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.evaluation.layered_human_audit import (
    _extract_game_prefix_bytes,
    create_human_db_snapshot,
)


def _play(board: BoardState, notation: str) -> tuple[dict, BoardState]:
    matching = []
    for move in get_all_legal_moves(board):
        base = (
            str(move["to"])
            if move["from"] is None
            else f"{move['from']}-{move['to']}"
        )
        actual = (
            base
            if move["capture"] is None
            else f"{base}x{move['capture']}"
        )
        if actual == notation:
            matching.append(move)
    assert len(matching) == 1
    move = matching[0]
    record = {
        "turn": 1,
        "color": board.turn,
        "type": "place",
        "from": move["from"],
        "to": move["to"],
        "capture": move["capture"],
        "notation": notation,
        "board_fen_before": board.to_fen_string(),
    }
    return record, board.apply_move(move)


def _game_bytes() -> bytes:
    board = BoardState.new_game()
    moves = []
    for notation in (
        "c4",
        "b4",
        "d5",
        "d6",
        "c5",
        "f4",
        "e5xb4",
        "b4",
        "c3xf4",
        "d2",
        "f4",
        "e4",
    ):
        move, board = _play(board, notation)
        moves.append(move)
    return json.dumps(
        {
            "session_id": "game-1",
            "source": "playok",
            "source_type": "human",
            "winner": "W",
            "moves": moves,
        }
    ).encode("utf-8")


def test_extract_human_prefix_uses_one_real_complete_history() -> None:
    status, prefix, error = _extract_game_prefix_bytes(
        _game_bytes(),
        relative_path="human_game_1.jsonl",
    )

    assert status == "eligible"
    assert error is None
    assert prefix is not None
    assert len(prefix.logical_turns) == 12
    assert prefix.logical_turns[6] == ("e5", "xb4")
    assert prefix.logical_turns[8] == ("c3", "xf4")
    assert prefix.action_tokens[6:8] == ("e5", "xb4")
    assert prefix.winner == "W"


def test_extract_human_prefix_rejects_stitched_or_changed_history() -> None:
    payload = json.loads(_game_bytes())
    payload["moves"][5]["board_fen_before"] = BoardState.new_game().to_fen_string()

    status, prefix, error = _extract_game_prefix_bytes(
        json.dumps(payload).encode("utf-8"),
        relative_path="changed.jsonl",
    )

    assert status == "invalid"
    assert prefix is None
    assert error == "move_5_fen_mismatch"


def test_sqlite_online_backup_preserves_source_sidecar(
    tmp_path: Path,
) -> None:
    source = tmp_path / "active.sqlite"
    owner = sqlite3.connect(source)
    owner.execute("PRAGMA journal_mode=WAL")
    owner.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE positions (state_key TEXT PRIMARY KEY);
        CREATE TABLE moves (
            state_key TEXT,
            notation TEXT,
            PRIMARY KEY (state_key, notation)
        );
        CREATE TABLE processed_files (file_path TEXT PRIMARY KEY);
        INSERT INTO meta VALUES ('schema_version', '2');
        INSERT INTO positions VALUES ('p');
        INSERT INTO moves VALUES ('p', 'a7');
        INSERT INTO processed_files VALUES ('g.jsonl');
        """
    )
    owner.commit()
    shm = Path(str(source) + "-shm")
    assert shm.exists() and shm.stat().st_size > 0

    destination = tmp_path / "snapshot" / "human.sqlite"
    evidence = create_human_db_snapshot(source, destination)

    assert destination.exists()
    assert shm.exists() and shm.stat().st_size > 0
    assert evidence["source"]["sidecars_deleted"] is False
    assert evidence["snapshot"]["quick_check"] == ["ok"]
    assert evidence["snapshot"]["row_counts"] == {
        "meta": 1,
        "moves": 1,
        "positions": 1,
        "processed_files": 1,
    }
    owner.close()
