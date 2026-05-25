"""tests/test_fullgame_db.py — Sanity tests for the full-game database.

These tests stay tiny and fast (well under a second).  A full DB build is
expensive (potentially many GB) and is never attempted here.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Build a minimal `ai` namespace package containing ONLY the two leaf modules
# we need (board_symmetry, fullgame_db).  This avoids triggering the real
# ai/__init__.py which depends on chromadb / ollama / fastapi.
_ROOT = Path(__file__).resolve().parent.parent
import importlib.util as _ilu

_ai_pkg = types.ModuleType("ai")
_ai_pkg.__path__ = [str(_ROOT / "ai")]
# Mark as a regular package: relative imports inside ai/fullgame_db.py will
# now resolve against this empty package, not the real ai/__init__.py.
sys.modules["ai"] = _ai_pkg

def _load_leaf(name: str, file: Path):
    spec = _ilu.spec_from_file_location(f"ai.{name}", str(file))
    mod = _ilu.module_from_spec(spec)
    sys.modules[f"ai.{name}"] = mod
    spec.loader.exec_module(mod)
    setattr(_ai_pkg, name, mod)
    return mod

_load_leaf("board_symmetry", _ROOT / "ai" / "board_symmetry.py")
_fgdb = _load_leaf("fullgame_db", _ROOT / "ai" / "fullgame_db.py")
FullGameDB = _fgdb.FullGameDB
FullGameResult = _fgdb.FullGameResult
export_to_binary = _fgdb.export_to_binary
_pack_move = _fgdb._pack_move
_unpack_move = _fgdb._unpack_move
_EMPTY_MOVE = _fgdb._EMPTY_MOVE

from game.board import BoardState

# Load the builder as a script-style module.
_spec = _ilu.spec_from_file_location(
    "build_fullgame_db", str(_ROOT / "tools" / "build_fullgame_db.py"),
)
build_mod = _ilu.module_from_spec(_spec)
sys.modules["build_fullgame_db"] = build_mod
_spec.loader.exec_module(build_mod)


class TestCanonicalEncoding(unittest.TestCase):
    def test_encode_roundtrip_distinguishes_positions(self):
        b = BoardState.new_game()
        key1 = build_mod.position_key(b)
        b2 = BoardState(
            positions=dict(b.positions),
            turn="B",
            pieces_on_board=dict(b.pieces_on_board),
            pieces_placed=dict(b.pieces_placed),
            pieces_captured=dict(b.pieces_captured),
        )
        key2 = build_mod.position_key(b2)
        self.assertNotEqual(key1, key2)

    def test_d4_equivalence_keys_match(self):
        # Two boards that are D4-equivalent must yield identical keys.
        b = BoardState.new_game()
        m1 = {"from": None, "to": "a7", "capture": None}
        m2 = {"from": None, "to": "g7", "capture": None}  # 90° rotation of a7
        b1 = b.apply_move(m1)
        b2 = b.apply_move(m2)
        self.assertEqual(build_mod.position_key(b1), build_mod.position_key(b2))


class TestBuilderTinyRun(unittest.TestCase):
    def _build_tiny(self, path: Path, cap: int = 200) -> None:
        builder = build_mod.FullGameDBBuilder(
            db_path=path, max_positions=cap, commit_every=50,
        )
        builder.enumerate_forward(BoardState.new_game())
        builder.backpropagate(passes=2)
        builder.close()

    def test_tiny_build_and_query(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "tiny.sqlite"
            self._build_tiny(db_path, cap=300)
            self.assertTrue(db_path.exists())

            db = FullGameDB(db_path)
            self.assertTrue(db.is_available())
            stats = db.stats()
            self.assertGreater(stats["positions"], 0)

            # The opening board must be present.
            result = db.query(BoardState.new_game())
            self.assertIsNotNone(result)
            # And it must have trajectories (24 legal placements, but
            # canonicalisation may merge symmetric edges — at least 3 unique
            # by D4 equivalence classes of single placements on empty board).
            self.assertGreaterEqual(len(result.trajectories), 1)
            db.close()

    def test_score_delta_shape(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "tiny.sqlite"
            self._build_tiny(db_path, cap=200)
            db = FullGameDB(db_path)
            hints = db.score_delta(BoardState.new_game(), "W")
            self.assertIsInstance(hints, dict)
            for k, v in hints.items():
                self.assertIsInstance(k, str)
                self.assertGreaterEqual(v, -0.5)
                self.assertLessEqual(v, 0.5)
            db.close()


class TestMissingDBFallback(unittest.TestCase):
    def test_missing_file_is_unavailable(self):
        db = FullGameDB("/nonexistent/path/fullgame.sqlite")
        self.assertFalse(db.is_available())
        self.assertIsNone(db.query(BoardState.new_game()))
        self.assertEqual(db.score_delta(BoardState.new_game(), "W"), {})


class TestBinaryPackingHelpers(unittest.TestCase):
    def test_struct_record_size(self):
        import struct
        self.assertEqual(struct.calcsize("<9sBHIIIII"), 32)

    def test_empty_move_sentinel(self):
        self.assertEqual(_pack_move(None), _EMPTY_MOVE)
        notation, flag = _unpack_move(_EMPTY_MOVE)
        self.assertIsNone(notation)
        self.assertEqual(flag, "N")

    def test_placement_roundtrip(self):
        packed = _pack_move("a4", "W")
        notation, flag = _unpack_move(packed)
        self.assertEqual(notation, "a4")
        self.assertEqual(flag, "W")

    def test_movement_roundtrip(self):
        packed = _pack_move("a7-a4", "L")
        notation, flag = _unpack_move(packed)
        self.assertEqual(notation, "a7-a4")
        self.assertEqual(flag, "L")

    def test_capture_roundtrip(self):
        packed = _pack_move("a7-a4xb4", "N")
        notation, flag = _unpack_move(packed)
        self.assertEqual(notation, "a7-a4xb4")
        self.assertEqual(flag, "N")

    def test_placement_capture_roundtrip(self):
        packed = _pack_move("d2xa4", "W")
        notation, flag = _unpack_move(packed)
        self.assertEqual(notation, "d2xa4")
        self.assertEqual(flag, "W")

    def test_all_positions_roundtrip(self):
        from game.board import POSITIONS
        for pos in POSITIONS:
            packed = _pack_move(pos)
            notation, _ = _unpack_move(packed)
            self.assertEqual(notation, pos, f"Roundtrip failed for placement {pos!r}")


class TestBinaryRoundtrip(unittest.TestCase):
    def _build_tiny(self, path: Path, cap: int = 300) -> None:
        builder = build_mod.FullGameDBBuilder(
            db_path=path, max_positions=cap, commit_every=50,
        )
        builder.enumerate_forward(BoardState.new_game())
        builder.backpropagate(passes=2)
        builder.close()

    def test_binary_export_and_query(self):
        with tempfile.TemporaryDirectory() as td:
            sqlite_path = Path(td) / "tiny.sqlite"
            bin_path = Path(td) / "tiny.bin"

            self._build_tiny(sqlite_path)
            n = export_to_binary(sqlite_path, bin_path)
            self.assertGreater(n, 0)
            self.assertTrue(bin_path.exists())

            # File size must be header + n × 32 bytes.
            expected = 32 + n * 32
            self.assertEqual(bin_path.stat().st_size, expected)

            # Open via FullGameDB — must detect binary format.
            db = FullGameDB(bin_path)
            self.assertTrue(db.is_available())
            stats = db.stats()
            self.assertEqual(stats["positions"], n)

            # Opening board must hit.
            result = db.query(BoardState.new_game())
            self.assertIsNotNone(result)

            # Results from binary must be consistent with SQLite.
            db_sql = FullGameDB(sqlite_path)
            result_sql = db_sql.query(BoardState.new_game())
            self.assertIsNotNone(result_sql)
            self.assertEqual(result.outcome, result_sql.outcome)
            self.assertEqual(result.best_move_canonical, result_sql.best_move_canonical)

            db.close()
            db_sql.close()

    def test_binary_score_delta_shape(self):
        with tempfile.TemporaryDirectory() as td:
            sqlite_path = Path(td) / "tiny.sqlite"
            bin_path = Path(td) / "tiny.bin"
            self._build_tiny(sqlite_path, cap=200)
            export_to_binary(sqlite_path, bin_path)
            db = FullGameDB(bin_path)
            hints = db.score_delta(BoardState.new_game(), "W")
            self.assertIsInstance(hints, dict)
            for k, v in hints.items():
                self.assertIsInstance(k, str)
                self.assertGreaterEqual(v, -0.5)
                self.assertLessEqual(v, 0.5)
            db.close()


if __name__ == "__main__":
    unittest.main()
