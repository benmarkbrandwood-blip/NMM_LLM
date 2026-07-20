"""Safety tests for persisted Malom-label provenance."""

from __future__ import annotations

import importlib
import sqlite3
import sys
import types
from pathlib import Path

import pytest

# Import leaf modules without executing ai/__init__.py, whose public facade has
# optional chromadb-backed imports that are unrelated to these SQLite tests.
if "ai" not in sys.modules:
    _ai_package = types.ModuleType("ai")
    _ai_package.__path__ = [str(Path(__file__).resolve().parents[1] / "ai")]
    sys.modules["ai"] = _ai_package

from ai.board_symmetry import transform_notation
from ai.human_db import HumanDB
from learned_ai.data.malom_label_provenance import (
    CURRENT_MALOM_LABEL_VERSION,
    ensure_human_db_can_be_annotated,
    read_malom_label_version,
    require_current_human_db_malom_labels,
    write_current_malom_label_version,
)
from ai.trajectory_db import make_board_state_key
from game.board import BoardState
from learned_ai.data.specialist_db import SpecialistDB, _board_hash


_HUMAN_SCHEMA = """
CREATE TABLE positions (
    state_key TEXT PRIMARY KEY,
    total_games INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    draws INTEGER NOT NULL DEFAULT 0,
    malom_wdl TEXT,
    malom_dtw INTEGER,
    canonical_winning_move TEXT
);
CREATE TABLE moves (
    state_key TEXT NOT NULL,
    notation TEXT NOT NULL,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    draws INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0,
    moves_to_end_sum REAL NOT NULL DEFAULT 0.0,
    malom_wdl_after TEXT,
    malom_dtw_after INTEGER,
    PRIMARY KEY (state_key, notation)
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""


def _create_human_db(
    path: Path,
    *,
    label_version: str | None,
    with_labels: bool = True,
) -> BoardState:
    board = BoardState.new_game()
    state_key, sym_idx = make_board_state_key(board)
    notation = transform_notation("a1", sym_idx)
    assert notation is not None

    conn = sqlite3.connect(path)
    conn.executescript(_HUMAN_SCHEMA)
    conn.execute("INSERT INTO meta(key, value) VALUES ('total_games', '7')")
    if label_version is not None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            ("malom_label_version", label_version),
        )
    conn.execute(
        """
        INSERT INTO positions(
            state_key, total_games, wins, losses, draws,
            malom_wdl, malom_dtw, canonical_winning_move
        ) VALUES (?, 7, 4, 2, 1, ?, ?, ?)
        """,
        (
            state_key,
            "W" if with_labels else None,
            9 if with_labels else None,
            notation,
        ),
    )
    conn.execute(
        """
        INSERT INTO moves(
            state_key, notation, wins, losses, draws, total,
            moves_to_end_sum, malom_wdl_after, malom_dtw_after
        ) VALUES (?, ?, 4, 2, 1, 7, 21.0, ?, ?)
        """,
        (
            state_key,
            notation,
            "L" if with_labels else None,
            -8 if with_labels else None,
        ),
    )
    conn.commit()
    conn.close()
    return board


def test_new_specialist_db_stamps_and_uses_current_labels(tmp_path: Path) -> None:
    path = tmp_path / "specialist.sqlite"
    board = BoardState.new_game()

    db = SpecialistDB(path)
    try:
        assert db.malom_labels_trusted
        assert db.malom_label_version == CURRENT_MALOM_LABEL_VERSION
        db.label_position_malom(board, "D")
        evidence = db.query_wdl_evidence(board, min_samples=3)
        assert evidence is not None
        assert evidence.theoretical_wdl is not None
        assert evidence.theoretical_wdl.kind == "theoretical_wdl"
        assert evidence.theoretical_wdl.value == "D"
        assert evidence.empirical_distribution is None
        assert db.query_wdl(board, min_samples=3) == (0.05, 0.90, 0.05)
        with pytest.raises(RuntimeError, match="conflicting trusted Malom label"):
            db.label_position_malom(board, "W")
    finally:
        db.close()

    conn = sqlite3.connect(path)
    try:
        assert read_malom_label_version(conn) == CURRENT_MALOM_LABEL_VERSION
    finally:
        conn.close()


def test_legacy_specialist_labels_are_ignored_and_cannot_be_extended(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-specialist.sqlite"
    board = BoardState.new_game()

    seed = SpecialistDB(path)
    seed.close()
    conn = sqlite3.connect(path)
    conn.execute(
        """
        INSERT INTO positions(
            pos_hash, wins, draws, losses, malom_label, last_seen
        ) VALUES (?, 0, 0, 0, 'W', 'legacy')
        """,
        (_board_hash(board),),
    )
    conn.execute("DELETE FROM meta WHERE key = 'malom_label_version'")
    conn.commit()
    conn.close()

    db = SpecialistDB(path)
    try:
        assert not db.malom_labels_trusted
        assert db.malom_label_version is None
        assert db.query_wdl(board, min_samples=3) is None
        assert db.query_win_prob(board, min_samples=3) is None
        with pytest.raises(RuntimeError, match="new database path"):
            db.label_position_malom(board, "D")

        db._conn.execute(
            "UPDATE positions SET wins = 3 WHERE pos_hash = ?",
            (_board_hash(board),),
        )
        db._conn.commit()
        assert db.query_wdl(board, min_samples=3) == (1.0, 0.0, 0.0)
    finally:
        db.close()


def test_unlabelled_specialist_db_can_adopt_current_version(tmp_path: Path) -> None:
    path = tmp_path / "unlabelled-specialist.sqlite"
    seed = SpecialistDB(path)
    seed.close()

    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM meta WHERE key = 'malom_label_version'")
    conn.commit()
    conn.close()

    db = SpecialistDB(path)
    try:
        assert db.malom_labels_trusted
        assert db.malom_label_version == CURRENT_MALOM_LABEL_VERSION
    finally:
        db.close()


def test_human_db_hides_legacy_labels_but_keeps_game_statistics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-human.sqlite"
    board = _create_human_db(path, label_version=None)

    db = HumanDB(path)
    try:
        assert db.is_available()
        assert not db.malom_labels_trusted

        position = db.query_position(board)
        assert position is not None
        assert (position.total_games, position.wins) == (7, 4)
        assert position.malom_wdl is None
        assert position.malom_dtw is None

        moves = db.query_moves(board)
        assert len(moves) == 1
        assert moves[0].total == 7
        assert moves[0].malom_wdl_after is None
        assert moves[0].malom_dtw_after is None
        assert db.query_all_frequencies(board, min_samples=1) == {"a1": 1.0}
        assert db.get_malom_wdl(board) is None
    finally:
        db.close()


def test_human_db_exposes_current_labels(tmp_path: Path) -> None:
    path = tmp_path / "current-human.sqlite"
    board = _create_human_db(
        path,
        label_version=CURRENT_MALOM_LABEL_VERSION,
    )

    db = HumanDB(path)
    try:
        assert db.malom_labels_trusted
        position = db.query_position(board)
        assert position is not None
        assert (position.malom_wdl, position.malom_dtw) == ("W", 9)
        move = db.query_moves(board)[0]
        assert (move.malom_wdl_after, move.malom_dtw_after) == ("L", -8)
        assert db.get_malom_wdl(board) == {"outcome": "W", "dtw": 9}
    finally:
        db.close()


def test_builder_requires_rebuild_before_annotating_legacy_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "builder-human.sqlite"
    _create_human_db(path, label_version=None)
    conn = sqlite3.connect(path)
    try:
        with pytest.raises(RuntimeError, match="--rebuild"):
            ensure_human_db_can_be_annotated(conn, path)
        with pytest.raises(RuntimeError, match="corrected Malom decoder"):
            require_current_human_db_malom_labels(conn, path)

        write_current_malom_label_version(conn)
        ensure_human_db_can_be_annotated(conn, path)
        require_current_human_db_malom_labels(conn, path)
    finally:
        conn.close()


def test_builder_may_annotate_an_unlabelled_human_db(tmp_path: Path) -> None:
    path = tmp_path / "unlabelled-human.sqlite"
    _create_human_db(path, label_version=None, with_labels=False)
    conn = sqlite3.connect(path)
    try:
        ensure_human_db_can_be_annotated(conn, path)
    finally:
        conn.close()


@pytest.mark.parametrize(
    "module_name",
    ("tools.build_human_db", "tools.build_human_db_sha"),
)
def test_builders_check_legacy_labels_before_no_file_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)
    db_path = tmp_path / f"{module_name.rsplit('.', 1)[-1]}-legacy.sqlite"
    games_dir = tmp_path / "empty-games"
    games_dir.mkdir()
    _create_human_db(db_path, label_version=None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module_name,
            "--games-dir",
            str(games_dir),
            "--output",
            str(db_path),
            "--malom-db",
            "configured-malom-path",
            "--update",
        ],
    )

    with pytest.raises(RuntimeError, match="--rebuild"):
        module.main()


@pytest.mark.parametrize(
    "module_name",
    ("tools.build_human_db", "tools.build_human_db_sha"),
)
def test_builders_stamp_version_only_after_completed_annotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)
    db_path = tmp_path / f"{module_name.rsplit('.', 1)[-1]}-current.sqlite"
    games_dir = tmp_path / f"{module_name.rsplit('.', 1)[-1]}-games"
    games_dir.mkdir()
    (games_dir / "empty.jsonl").touch()
    monkeypatch.setattr(
        module,
        "_annotate_malom",
        lambda *_args: ({}, {}, True),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module_name,
            "--games-dir",
            str(games_dir),
            "--output",
            str(db_path),
            "--malom-db",
            "configured-malom-path",
        ],
    )

    module.main()

    conn = sqlite3.connect(db_path)
    try:
        assert read_malom_label_version(conn) == CURRENT_MALOM_LABEL_VERSION
    finally:
        conn.close()


def test_gap_dataset_rejects_unversioned_human_labels(tmp_path: Path) -> None:
    from scripts.build_gap_dataset import build_dataset

    path = tmp_path / "gap-human.sqlite"
    _create_human_db(path, label_version=None)

    with pytest.raises(RuntimeError, match=CURRENT_MALOM_LABEL_VERSION):
        build_dataset(
            path,
            tmp_path / "missing-sentinel.pt",
            tmp_path / "missing-value-net.npz",
            n_per_category=1,
            dtw_threshold=15,
        )


def test_vn_trajectory_rejects_unversioned_human_labels(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("scripts.train_vn_trajectory")
    path = tmp_path / "vn-human.sqlite"
    _create_human_db(path, label_version=None)
    conn = sqlite3.connect(path)
    try:
        with pytest.raises(RuntimeError, match=CURRENT_MALOM_LABEL_VERSION):
            module.build_dataset(
                conn,
                malom_db=None,
                rng=module.np.random.default_rng(42),
                sentinel_advisor=None,
                n_starts=0,
                verbose=False,
            )
    finally:
        conn.close()
