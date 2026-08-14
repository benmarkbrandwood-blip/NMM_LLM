from __future__ import annotations

import copy

import pytest

from game.board import BoardState
from learned_ai.evaluation.human_raw_reconstructability import (
    F0D0AuditError,
    audit_game_record,
    reconcile_file_audits,
    seal_manifest,
    verify_manifest,
)


def _record_from_moves(moves: list[dict]) -> dict:
    board = BoardState.new_game()
    rows: list[dict] = []
    for index, move in enumerate(moves):
        notation = (
            str(move["to"])
            if move["from"] is None
            else f'{move["from"]}-{move["to"]}'
        )
        if move.get("capture") is not None:
            notation += f'x{move["capture"]}'
        rows.append(
            {
                "turn": index // 2 + 1,
                "color": board.turn,
                "type": "place" if move["from"] is None else "move",
                "from": move["from"],
                "to": move["to"],
                "capture": move.get("capture"),
                "notation": notation,
                "board_fen_before": board.to_fen_string(),
            }
        )
        board = board.apply_move(move)
    return {
        "session_id": "ml-fixture-1",
        "source": "playok",
        "source_type": "human_vs_human",
        "date": "2026-01-02",
        "white_player": "player-white",
        "black_player": "player-black",
        "white_elo": 1200,
        "black_elo": 1300,
        "human_color": None,
        "winner": None,
        "draw_reason": None,
        "result_raw": "1/2-1/2",
        "moves": rows,
    }


def _short_record() -> dict:
    return _record_from_moves(
        [
            {"from": None, "to": "a7", "capture": None},
            {"from": None, "to": "d7", "capture": None},
            {"from": None, "to": "g7", "capture": None},
            {"from": None, "to": "a4", "capture": None},
            {"from": None, "to": "g4", "capture": None},
            {"from": None, "to": "g1", "capture": None},
        ]
    )


def test_continuous_record_reconstructs_rule_history() -> None:
    result = audit_game_record(
        _short_record(),
        relative_path="data/human_games/human_ml-fixture-1.jsonl",
        imported_at="2026-01-03T04:05:06",
    )

    assert result["dimensions"]["history"] == "recoverable"
    assert result["replay"]["status"] == "ongoing"
    assert result["replay"]["final_no_progress_plies"] == 0
    assert result["dimensions"]["player"] == "recoverable"
    assert result["dimensions"]["source"] == "partial"
    assert result["dimensions"]["result"] == "partial"
    assert result["dimensions"]["condition"] == "partial"


def test_recorded_fen_mismatch_fails_history_closed() -> None:
    record = _short_record()
    record["moves"][2]["board_fen_before"] = "bad-fen"

    result = audit_game_record(
        record,
        relative_path="data/human_games/human_ml-fixture-1.jsonl",
        imported_at="2026-01-03T04:05:06",
    )

    assert result["dimensions"]["history"] == "not_recoverable"
    assert result["replay"]["status"] == "failed"
    assert "history.move_2_fen_mismatch" in result["failure_codes"]


def test_zero_move_record_has_exact_initial_history_but_no_behavior_turn() -> None:
    record = _short_record()
    record["moves"] = []

    result = audit_game_record(
        record,
        relative_path="data/human_games/human_ml-fixture-1.jsonl",
        imported_at="2026-01-03T04:05:06",
    )

    assert result["dimensions"]["history"] == "recoverable"
    assert result["replay"]["logical_plies_replayed"] == 0
    assert result["replay"]["final_no_progress_plies"] == 0
    assert not result["behavior_replay_eligible"]


def test_conflicting_duplicate_session_fails_closed() -> None:
    first = {
        "relative_path": "data/human_games/human_ml-fixture-1.jsonl",
        "byte_length": 10,
        "sha256": "a" * 64,
        "session_id": "ml-fixture-1",
        "game_audit": {"session_id": "ml-fixture-1"},
    }
    second = {
        **first,
        "relative_path": "data/human_games/test_set/human_ml-fixture-1.jsonl",
        "sha256": "b" * 64,
    }

    with pytest.raises(F0D0AuditError, match="conflicting duplicate"):
        reconcile_file_audits(
            [first, second],
            imported={"ml-fixture-1": "2026-01-03T04:05:06"},
        )


def test_byte_identical_duplicate_session_collapses_with_both_paths() -> None:
    game_audit = audit_game_record(
        _short_record(),
        relative_path="data/human_games/human_ml-fixture-1.jsonl",
        imported_at="2026-01-03T04:05:06",
    )
    first = {
        "relative_path": "data/human_games/human_ml-fixture-1.jsonl",
        "byte_length": 10,
        "sha256": "a" * 64,
        "session_id": "ml-fixture-1",
        "game_audit": game_audit,
    }
    second = {
        **first,
        "relative_path": "data/human_games/test_set/human_ml-fixture-1.jsonl",
    }

    result = reconcile_file_audits(
        [first, second],
        imported={"ml-fixture-1": "2026-01-03T04:05:06"},
    )

    assert result["unique_sessions"] == 1
    assert result["duplicate_sessions"] == 1
    assert result["duplicate_extra_files"] == 1
    assert result["game_records"][0]["duplicate_files"] == [
        "data/human_games/test_set/human_ml-fixture-1.jsonl"
    ]


def test_manifest_identity_detects_tampering() -> None:
    payload = {
        "schema_version": "nmm.f0-d0-human-raw-reconstructability.v1",
        "audit_id": "f0-d0-human-raw-reconstructability-v1",
        "corpus_identity": "c" * 64,
        "counts": {"unique_sessions": 1},
    }
    sealed = seal_manifest(payload)

    verify_manifest(sealed)
    changed = copy.deepcopy(sealed)
    changed["counts"]["unique_sessions"] = 2
    with pytest.raises(F0D0AuditError, match="manifest identity"):
        verify_manifest(changed)
