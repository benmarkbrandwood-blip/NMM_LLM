"""Tests for learned_ai/sentinel/db_teacher.py (graceful external DB stub)."""

from __future__ import annotations

from game.board import BoardState
from game.rules import get_game_phase, terminal_wdl
from learned_ai.sentinel.config import SentinelConfig
from learned_ai.sentinel.db_teacher import ExternalSolvedDB, open_external_db


def _board():
    return BoardState.from_fen_string("BBW....B.W.W............|W|3|3")


class _FakeMalom:
    def __init__(self, outcomes=("D", "L")):
        self.outcomes = outcomes
        self.queried_boards = []

    def is_available(self):
        return True

    def query(self, board):
        index = len(self.queried_boards)
        self.queried_boards.append(board)
        return {"outcome": self.outcomes[index], "dtw": 0}

    def close(self):
        pass


def _available_fake_db(outcomes=("D", "L")):
    db = ExternalSolvedDB("")
    fake = _FakeMalom(outcomes)
    db._malom = fake
    return db, fake


def _placement_mill_board():
    return BoardState.from_setup(
        {"a7": "W", "d7": "W", "a4": "B", "b4": "B"},
        turn="W",
        phase="place",
    )


def _four_vs_four_mill_board():
    return BoardState.from_setup(
        {
            "a7": "W", "d7": "W", "g4": "W", "b4": "W",
            "a4": "B", "b6": "B", "d6": "B", "f6": "B",
        },
        turn="W",
        phase="move",
    )


def _three_vs_three_terminal_capture_board():
    return BoardState.from_setup(
        {
            "a7": "W", "d7": "W", "g4": "W",
            "a4": "B", "b6": "B", "d6": "B",
        },
        turn="W",
        phase="move",
    )


def _blocked_board():
    return BoardState.from_setup(
        {
            "a7": "W", "g7": "W", "g1": "W", "a1": "W",
            "d7": "B", "g4": "B", "d1": "B", "a4": "B",
        },
        turn="W",
        phase="move",
    )


def test_unavailable_returns_none_empty_path():
    db = ExternalSolvedDB(db_path="", enabled=True)
    assert db.is_available() is False
    assert db.query_state(_board()) is None
    assert db.query(_board()) is None
    assert db.query_move_quality(_board(), {"from": None, "to": "a4"}) is None


def test_is_available_false_when_no_path():
    db = ExternalSolvedDB("")
    assert db.is_available() is False


def test_no_crash_on_bad_path():
    # Must not raise on init even for a clearly nonexistent path.
    db = ExternalSolvedDB("/nonexistent/path/to/db")
    assert db.is_available() is False
    assert db.query_state(_board()) is None


def test_query_trajectory_length_matches_input():
    db = ExternalSolvedDB("")
    states = [_board(), _board(), _board()]
    result = db.query_trajectory(states)
    assert result == [None, None, None]
    assert len(result) == len(states)


def test_disabled_forces_unavailable(tmp_path):
    # Even if files exist, enabled=False keeps it unavailable.
    (tmp_path / "database.dat").write_bytes(b"\x00" * 16)
    (tmp_path / "preCalculatedVars.dat").write_bytes(b"\x01" * 16)
    db = ExternalSolvedDB(str(tmp_path), enabled=False)
    assert db.is_available() is False
    assert db.query_state(_board()) is None


def test_probe_records_format_metadata(tmp_path):
    # A directory with no .sec2 files: adapter probes but stays unavailable.
    (tmp_path / "database.dat").write_bytes(b"\x00" * 1024)
    (tmp_path / "preCalculatedVars.dat").write_bytes(b"ABCD" * 8)
    db = ExternalSolvedDB(str(tmp_path), enabled=True)
    assert db.is_available() is False
    assert "db_dir" in db.format_probe
    assert db.format_probe["available"] is False
    assert db.query_state(_board()) is None


def test_open_external_db_from_config():
    cfg = SentinelConfig(external_db_path="", external_db_enabled=False)
    db = open_external_db(cfg)
    assert isinstance(db, ExternalSolvedDB)
    assert db.is_available() is False


def test_close_is_noop():
    db = ExternalSolvedDB("")
    db.close()  # must not raise


def test_query_move_quality_accepts_explicit_non_capture_action():
    db, fake = _available_fake_db()
    board = _placement_mill_board()

    quality = db.query_move_quality(
        board, {"from": None, "to": "g4", "capture": None})

    assert quality == 1.0
    assert len(fake.queried_boards) == 2
    assert fake.queried_boards[1].positions["g4"] == "W"
    assert fake.queried_boards[1].turn == "B"


def test_query_move_quality_rejects_mill_without_capture_before_lookup():
    board = _placement_mill_board()
    incomplete_moves = (
        {"from": None, "to": "g7"},
        {"from": None, "to": "g7", "capture": None},
    )

    for move in incomplete_moves:
        db, fake = _available_fake_db()
        assert db.query_move_quality(board, move) is None
        assert fake.queried_boards == []


def test_query_move_quality_rejects_spurious_or_illegal_capture():
    board = _placement_mill_board()
    invalid_moves = (
        {"from": None, "to": "g4", "capture": "a4"},
        {"from": None, "to": "g7", "capture": "g4"},
    )

    for move in invalid_moves:
        db, fake = _available_fake_db()
        assert db.query_move_quality(board, move) is None
        assert fake.queried_boards == []


def test_query_move_quality_applies_complete_mill_and_capture_atomically():
    db, fake = _available_fake_db()
    board = _placement_mill_board()
    move = {"from": None, "to": "g7", "capture": "a4"}

    quality = db.query_move_quality(board, move)

    assert quality == 1.0
    assert len(fake.queried_boards) == 2
    settled = fake.queried_boards[1]
    assert settled.positions["g7"] == "W"
    assert settled.positions["a4"] == ""
    assert settled.pieces_on_board == {"W": 3, "B": 1}
    assert settled.turn == "B"


def test_query_move_quality_capture_covers_four_to_three_and_side_swap():
    db, fake = _available_fake_db()
    board = _four_vs_four_mill_board()
    move = {"from": "g4", "to": "g7", "capture": "a4"}

    quality = db.query_move_quality(board, move)

    assert quality == 1.0
    settled = fake.queried_boards[1]
    assert settled.pieces_on_board == {"W": 4, "B": 3}
    assert settled.turn == "B"
    assert get_game_phase(settled, "B") == "fly"


def test_db_teacher_move_enumerator_uses_atomic_rules_source():
    db, _ = _available_fake_db()
    board = _placement_mill_board()
    complete = {"from": None, "to": "g7", "capture": "a4"}
    incomplete = {"from": None, "to": "g7", "capture": None}

    moves = db._enumerate_legal_moves(board, "W")

    assert complete in moves
    assert incomplete not in moves
    assert db._enumerate_legal_moves(board, "B") == []


def test_terminal_wdl_covers_capture_blocking_and_placement_phase():
    capture_board = _three_vs_three_terminal_capture_board()
    settled = capture_board.apply_move(
        {"from": "g4", "to": "g7", "capture": "a4"})

    assert terminal_wdl(settled) == "L"
    assert terminal_wdl(settled, "W") == "W"
    assert terminal_wdl(_blocked_board()) == "L"
    assert terminal_wdl(_placement_mill_board()) is None


def test_query_state_resolves_rules_terminal_without_malom_probe():
    db, fake = _available_fake_db(("W",))
    settled = _three_vs_three_terminal_capture_board().apply_move(
        {"from": "g4", "to": "g7", "capture": "a4"})

    assert db.query_state(settled) == "L"
    assert fake.queried_boards == []


def test_query_move_quality_resolves_terminal_capture_without_child_probe():
    db, fake = _available_fake_db(("W",))
    board = _three_vs_three_terminal_capture_board()
    move = {"from": "g4", "to": "g7", "capture": "a4"}

    assert db.query_move_quality(board, move) == 0.0
    assert fake.queried_boards == [board]


def test_query_all_moves_labels_terminal_capture_without_child_probe():
    db, fake = _available_fake_db(("D",) * 100)
    board = _three_vs_three_terminal_capture_board()
    terminal_move = {"from": "g4", "to": "g7", "capture": "a4"}

    results = db.query_all_moves(board, "W")
    terminal_result = next(r for r in results if r["move"] == terminal_move)

    assert terminal_result["wdl"] == "win"
    assert terminal_result["dtm"] == 0
    assert all(terminal_wdl(queried) is None for queried in fake.queried_boards)
