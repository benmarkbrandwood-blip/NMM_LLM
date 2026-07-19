"""tests/test_malom_db.py — Tests for ai/malom_db.py.

Tests are grouped into:
  1. Unit tests that do NOT require the database files (fast, always run).
  2. Integration tests that require a configured Malom database (skipped if absent).

Fast tests cover:
  - Symmetry operations (rot90, swap, etc.)
  - Collapse / hash count formula
  - parse_secval (using the real secval file if present, else skipped)
  - board_to_wbf bit mapping
  - read_sector header parsing
  - decode_entry encoding

Integration tests cover:
  - MalomDB.is_available() for the real DB path
  - MalomDB.query() returns None gracefully (hash works, no crash)
  - For a known endgame position (3v3 on board) query returns W/L/D
"""

from __future__ import annotations

import json
import os
import sys
import types
import unittest
from math import comb
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import importlib.util as _ilu

# Load ai package
_ai_pkg = types.ModuleType("ai")
_ai_pkg.__path__ = [str(_ROOT / "ai")]
sys.modules.setdefault("ai", _ai_pkg)

def _load_leaf(name: str, file: Path):
    spec = _ilu.spec_from_file_location(f"ai.{name}", str(file))
    mod = _ilu.module_from_spec(spec)
    sys.modules[f"ai.{name}"] = mod
    spec.loader.exec_module(mod)
    setattr(_ai_pkg, name, mod)
    return mod

_mdb = _load_leaf("malom_db", _ROOT / "ai" / "malom_db.py")

# Imports from module under test
MalomDB = _mdb.MalomDB
OracleValue = _mdb.OracleValue
parse_secval = _mdb.parse_secval
board_to_wbf = _mdb.board_to_wbf
read_sector = _mdb.read_sector
decode_entry = _mdb.decode_entry
_decode_raw_entry = _mdb._decode_raw_entry
MALOM_BITS_TO_POS = _mdb.MALOM_BITS_TO_POS
_POS_TO_MALOM_BIT = _mdb._POS_TO_MALOM_BIT
_HashState = _mdb._HashState
_collapse = _mdb._collapse
_sym24_from_perm = _mdb._sym24_from_perm
_SYM_PERMS = _mdb._SYM_PERMS
_get_hash_state = _mdb._get_hash_state

# Load game.board for BoardState
from game.board import BoardState, POSITIONS


def _resolve_db_dir() -> Path:
    candidates: list[Path] = []
    if os.environ.get("NMM_MALOM_DB"):
        candidates.append(Path(os.environ["NMM_MALOM_DB"]))

    local_config = _ROOT / "data" / "training_paths.local.json"
    if local_config.exists():
        try:
            configured = json.loads(local_config.read_text(encoding="utf-8"))
            value = configured.get("malom_db_path")
            if value:
                path = Path(value)
                candidates.append(path if path.is_absolute() else _ROOT / path)
        except (OSError, ValueError, TypeError):
            pass

    candidates.append(Path("/mnt/windows/NMM_DB/strong"))
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "std.secval").is_file():
            return candidate
    return candidates[0]


_DB_DIR = _resolve_db_dir()
_SECVAL_PATH = _DB_DIR / "std.secval"
_DB_AVAILABLE = _DB_DIR.is_dir() and any(_DB_DIR.glob("std_*.sec2"))
_DB_SKIP_REASON = f"Malom DB not found at configured path: {_DB_DIR}"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bit mapping tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBitMapping(unittest.TestCase):
    def test_24_positions(self):
        self.assertEqual(len(MALOM_BITS_TO_POS), 24)

    def test_all_positions_unique(self):
        self.assertEqual(len(set(MALOM_BITS_TO_POS)), 24)

    def test_all_positions_in_board(self):
        for pos in MALOM_BITS_TO_POS:
            self.assertIn(pos, POSITIONS, f"{pos} not in POSITIONS")

    def test_reverse_mapping_consistent(self):
        for i, pos in enumerate(MALOM_BITS_TO_POS):
            self.assertEqual(_POS_TO_MALOM_BIT[pos], i)

    def test_millpos_top_side(self):
        # millpos[0] = 14 = bits {1,2,3} = a7,d7,g7 (top side)
        bits = [i for i in range(24) if (14 >> i) & 1]
        names = [MALOM_BITS_TO_POS[b] for b in bits]
        self.assertEqual(sorted(names), sorted(["a7", "d7", "g7"]))

    def test_millpos_right_side(self):
        # millpos[1] = 56 = bits {3,4,5} = g7,g4,g1 (right side)
        bits = [i for i in range(24) if (56 >> i) & 1]
        names = [MALOM_BITS_TO_POS[b] for b in bits]
        self.assertEqual(sorted(names), sorted(["g7", "g4", "g1"]))

    def test_millpos_left_side(self):
        # millpos[3] = 131 = bits {0,1,7} = a4,a7,a1 (left side)
        bits = [i for i in range(24) if (131 >> i) & 1]
        names = [MALOM_BITS_TO_POS[b] for b in bits]
        self.assertEqual(sorted(names), sorted(["a4", "a7", "a1"]))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Symmetry tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSymmetries(unittest.TestCase):
    def test_identity_is_last(self):
        # sym 15 = identity
        for bit in range(24):
            a = 1 << bit
            self.assertEqual(_sym24_from_perm(_SYM_PERMS[15], a), a)

    def test_rot90_four_times_is_identity(self):
        # 4 × rot90 = identity
        perm = _SYM_PERMS[0]  # rot90
        test_val = 0b10101010_11001100_01010101
        r = test_val
        for _ in range(4):
            r = _sym24_from_perm(perm, r)
        self.assertEqual(r, test_val & 0xFFFFFF)

    def test_16_symmetries_distinct_on_asymmetric_board(self):
        # Place one piece at a7 (bit 1) — asymmetric position
        a = 1 << 1  # a7
        images = set()
        for perm in _SYM_PERMS:
            images.add(_sym24_from_perm(perm, a))
        # Expect 8 distinct images (a7 has no swap-symmetry partner in outer ring)
        self.assertGreater(len(images), 1)

    def test_rot90_outer_ring_cycle(self):
        # rot90 should cycle: a4(0)->d7(2)->g4(4)->d1(6)->a4(0)
        perm = _SYM_PERMS[0]  # rot90
        bit = 0  # a4
        for expected_next in [2, 4, 6, 0]:
            bit = _SYM_PERMS[0].index(bit)
            # Actually apply to the bit: new bit = perm[old bit]
        # Simpler test: apply rot90 to bit-0 and get bit-2
        self.assertEqual(_sym24_from_perm(perm, 1 << 0), 1 << 2)  # a4 -> d7


# ─────────────────────────────────────────────────────────────────────────────
# 3. Collapse / hash count
# ─────────────────────────────────────────────────────────────────────────────

class TestCollapse(unittest.TestCase):
    def test_collapse_no_white(self):
        # No white pieces: collapse is identity (all 24 bits active)
        # With W=0: collapse(0, b_bits) should return b_bits unchanged
        b_bits = 0b101010
        self.assertEqual(_collapse(0, b_bits, 0), b_bits)

    def test_collapse_removes_white_slots(self):
        # White at bit 0, Black at bit 1: black is remapped to bit 0
        w = 0b001  # bit 0 occupied by white
        b = 0b010  # bit 1 occupied by black (not bit 0)
        # collapsed: bit 0 is white, so black at bit 1 maps to compressed position 0
        result = _collapse(w, b, 1)
        self.assertEqual(result, 1)  # bit 1 (black) remaps to compressed bit 0

    def test_collapse_round_trip_info_preserved(self):
        # collapse + manual expand should recover original black bits
        # Place white at bits 0,2; black at bit 3
        w = (1 << 0) | (1 << 2)
        b = (1 << 3)
        collapsed = _collapse(w, b, 2)
        # Compressed: 2 white bits removed, so 22 slots remain
        # Bit 3 original -> slot 2 in compressed (skipping bits 0 and 2)
        # slot 0 = bit 1, slot 1 = bit 3 -> wait, bit 2 is white
        # Unoccupied bits in order: 1, 3, 4, 5, ...
        # bit 3 is the 2nd unoccupied -> compressed bit 1
        self.assertEqual(collapsed, 1 << 1)


class TestHashCount(unittest.TestCase):
    def test_hash_count_3_3(self):
        """Hash count for sector (3,3) must match std_3_3_0_0.sec2 entry count."""
        hs = _get_hash_state(3, 3)
        self.assertEqual(hs.hash_count, 210140)

    def test_hash_count_formula(self):
        """hash_count = f_count * C(24-W, B)."""
        hs = _get_hash_state(3, 3)
        f_count = hs.hash_count // comb(21, 3)
        self.assertEqual(f_count, 158)
        self.assertEqual(hs.hash_count, 158 * comb(21, 3))

    def test_hash_count_0_0(self):
        """Sector (0,0): 1 entry (empty board)."""
        hs = _get_hash_state(0, 0)
        self.assertEqual(hs.hash_count, 1)

    def test_hash_range_valid(self):
        """All hashes for 3W/3B positions fit within hash_count."""
        hs = _get_hash_state(3, 3)
        # Sample: hash a specific 3W,3B board
        w_bits = (1 << 1) | (1 << 10) | (1 << 20)  # a7, d6, e4
        b_bits = (1 << 3) | (1 << 12) | (1 << 17)  # g7, f4, c5
        h = hs.hash(w_bits, b_bits)
        self.assertGreaterEqual(h, 0)
        self.assertLess(h, hs.hash_count)


# ─────────────────────────────────────────────────────────────────────────────
# 4. board_to_wbf
# ─────────────────────────────────────────────────────────────────────────────

class TestBoardToWbf(unittest.TestCase):
    def test_empty_board(self):
        board = BoardState.new_game()
        wb, bb, wf, bf = board_to_wbf(board)
        self.assertEqual(wb, 0)
        self.assertEqual(bb, 0)
        self.assertEqual(wf, 9)
        self.assertEqual(bf, 9)

    def test_single_white_piece_a7(self):
        """White piece at a7 should set bit 1 in wb_bits."""
        board = BoardState.new_game()
        positions = dict(board.positions)
        positions["a7"] = "W"
        from game.board import BoardState as BS
        b = BS(
            positions=positions,
            turn="B",
            pieces_on_board={"W": 1, "B": 0},
            pieces_placed={"W": 1, "B": 0},
            pieces_captured={"W": 0, "B": 0},
            hash_key=0,
        )
        wb, bb, wf, bf = board_to_wbf(b)
        self.assertEqual(wb, 1 << 1)   # a7 = bit 1
        self.assertEqual(bb, 0)
        self.assertEqual(wf, 8)        # 9 - 1 placed
        self.assertEqual(bf, 9)

    def test_single_black_piece_a4(self):
        """Black piece at a4 should set bit 0 in bb_bits."""
        board = BoardState.new_game()
        positions = dict(board.positions)
        positions["a4"] = "B"
        from game.board import BoardState as BS
        b = BS(
            positions=positions,
            turn="W",
            pieces_on_board={"W": 0, "B": 1},
            pieces_placed={"W": 0, "B": 1},
            pieces_captured={"W": 0, "B": 0},
            hash_key=0,
        )
        wb, bb, wf, bf = board_to_wbf(b)
        self.assertEqual(bb, 1 << 0)   # a4 = bit 0
        self.assertEqual(wb, 0)
        self.assertEqual(bf, 8)

    def test_move_phase_wf_zero(self):
        """In move phase (all placed), wf=bf=0."""
        board = BoardState.from_setup(
            {"a7": "W", "d7": "W", "g7": "W", "g4": "W", "g1": "W",
             "d1": "W", "a1": "W", "a4": "W", "b6": "W",
             "b2": "B", "d2": "B", "f2": "B", "f4": "B", "f6": "B",
             "d6": "B", "b4": "B", "c4": "B", "d5": "B"},
            turn="W", phase="move"
        )
        wb, bb, wf, bf = board_to_wbf(board)
        self.assertEqual(wf, 0)
        self.assertEqual(bf, 0)
        self.assertEqual(bin(wb).count("1"), 9)
        self.assertEqual(bin(bb).count("1"), 9)


# ─────────────────────────────────────────────────────────────────────────────
# 5. decode_entry
# ─────────────────────────────────────────────────────────────────────────────

class TestDecodeEntry(unittest.TestCase):
    VIRT_WIN = 299
    VIRT_LOSS = -299

    def _make_entry(self, key1: int, key2: int) -> bytes:
        """Pack key1 (12-bit signed) and key2 (12-bit signed) into 3 bytes."""
        k1 = key1 & 0xFFF
        k2 = key2 & 0xFFF
        raw = k1 | (k2 << 12)
        return bytes([raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0xFF])

    def test_win_entry(self):
        # Sanmill: raw 298 + sector 1 reaches the +299 virtual win value.
        data = self._make_entry(298, 5)
        result = decode_entry(data, 0, {}, 1, self.VIRT_WIN, self.VIRT_LOSS)
        self.assertEqual(result, "W")

    def test_loss_entry(self):
        data = self._make_entry(-298, 3)
        result = decode_entry(data, 0, {}, -1, self.VIRT_WIN, self.VIRT_LOSS)
        self.assertEqual(result, "L")

    def test_positive_raw_draw_grade(self):
        data = self._make_entry(18, 1)
        result = decode_entry(data, 0, {}, -18, self.VIRT_WIN, self.VIRT_LOSS)
        self.assertEqual(result, "D")

    def test_negative_raw_draw_grade(self):
        data = self._make_entry(-21, 2)
        result = decode_entry(data, 0, {}, 21, self.VIRT_WIN, self.VIRT_LOSS)
        self.assertEqual(result, "D")

    def test_raw_virtual_value_is_not_enough(self):
        data = self._make_entry(299, 5)
        result = decode_entry(data, 0, {}, -1, self.VIRT_WIN, self.VIRT_LOSS)
        self.assertEqual(result, "D")

    def test_zero_zero_count_projects_to_draw(self):
        # Sanmill classifies this as Count, then applies sector correction.
        data = self._make_entry(0, 0)
        result = decode_entry(data, 0, {}, 0, self.VIRT_WIN, self.VIRT_LOSS)
        self.assertEqual(result, "D")

    def test_positive_count_entry_projects_to_draw(self):
        data = self._make_entry(0, 5)
        result = decode_entry(data, 0, {}, 0, self.VIRT_WIN, self.VIRT_LOSS)
        self.assertEqual(result, "D")

    def test_count_entry_can_reach_virtual_loss(self):
        # Mirrors Sanmill's raw=(0,51), sector=-299 regression vector.
        data = self._make_entry(0, 51)
        result = decode_entry(
            data, 0, {}, -299, self.VIRT_WIN, self.VIRT_LOSS)
        self.assertEqual(result, "L")

    def test_spec_field2_without_emset_returns_none(self):
        # key2 = spec_field2 = -2048, no em_set entry → None
        data = self._make_entry(299, -2048)
        result = decode_entry(data, 0, {}, 0, self.VIRT_WIN, self.VIRT_LOSS)
        self.assertIsNone(result)

    def test_spec_field2_with_emset(self):
        # key2 = spec_field2 = -2048, em_set[0] = 7 → Win (key1=299, key2 from em_set)
        data = self._make_entry(299, -2048)
        result = decode_entry(data, 0, {0: 7}, 0, self.VIRT_WIN, self.VIRT_LOSS)
        self.assertEqual(result, "W")
        self.assertEqual(_decode_raw_entry(data, 0, {0: 7}).key2, 7)

    def test_multiple_entries(self):
        w = self._make_entry(299, 10)
        l = self._make_entry(-299, 2)
        d = self._make_entry(1, 0)
        data = w + l + d
        self.assertEqual(
            decode_entry(data, 0, {}, 0, self.VIRT_WIN, self.VIRT_LOSS), "W")
        self.assertEqual(
            decode_entry(data, 1, {}, 0, self.VIRT_WIN, self.VIRT_LOSS), "L")
        self.assertEqual(
            decode_entry(data, 2, {}, 0, self.VIRT_WIN, self.VIRT_LOSS), "D")

    def test_sym_entry_returns_tuple(self):
        # key1=0, key2=-1 → Sym, sym_op = -((-1)+1) = 0
        data = self._make_entry(0, -1)
        result = decode_entry(data, 0, {}, 0, self.VIRT_WIN, self.VIRT_LOSS)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], "SYM")
        self.assertEqual(result[1], 0)  # sym_op = -((-1)+1) = 0

    def test_sym_entry_sym_op_range(self):
        # key2 ranges from -1 to -16 → sym_op from 0 to 15
        for sym_op in range(16):
            key2 = -(sym_op + 1)
            data = self._make_entry(0, key2)
            result = decode_entry(data, 0, {}, 0, self.VIRT_WIN, self.VIRT_LOSS)
            self.assertIsInstance(result, tuple, f"Expected SYM tuple for sym_op={sym_op}")
            self.assertEqual(result[0], "SYM")
            self.assertEqual(result[1], sym_op)

    def test_count_not_sym(self):
        # key1=0, key2>0 → Count (not Sym)
        data = self._make_entry(0, 10)
        result = decode_entry(data, 0, {}, 0, self.VIRT_WIN, self.VIRT_LOSS)
        self.assertEqual(result, "D")

    def test_invalid_symmetry_operation_fails_closed(self):
        data = self._make_entry(0, -17)
        result = decode_entry(data, 0, {}, 0, self.VIRT_WIN, self.VIRT_LOSS)
        self.assertIsNone(result)

    def test_out_of_range_index_fails_closed(self):
        data = self._make_entry(299, 1)
        result = decode_entry(data, 1, {}, 0, self.VIRT_WIN, self.VIRT_LOSS)
        self.assertIsNone(result)


# ─────────────────────────────────────────────────────────────────────────────
# 6. parse_secval (requires secval file)
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_SECVAL_PATH.exists(), "std.secval not found")
class TestParseSecval(unittest.TestCase):
    def setUp(self):
        self.vw, self.vl, self.sv = parse_secval(_SECVAL_PATH)

    def test_virt_win(self):
        self.assertEqual(self.vw, 299)

    def test_virt_loss(self):
        self.assertEqual(self.vl, -299)

    def test_has_entries(self):
        self.assertEqual(len(self.sv), 498)

    def test_nonzero_sector_value_count_matches_audit(self):
        self.assertEqual(sum(value != 0 for value in self.sv.values()), 491)

    def test_sector_manifest_matches_files(self):
        expected = {
            f"std_{w}_{b}_{wf}_{bf}.sec2"
            for w, b, wf, bf in self.sv
        }
        actual = {path.name for path in _DB_DIR.glob("std_*.sec2")}
        self.assertEqual(actual, expected)

    def test_known_draw_sector(self):
        # (3,3,0,0) should be a draw (value 0) from the secval file
        self.assertIn((3, 3, 0, 0), self.sv)
        self.assertEqual(self.sv[(3, 3, 0, 0)], 0)

    def test_entry_format(self):
        for (W, B, WF, BF), sv in list(self.sv.items())[:5]:
            self.assertIsInstance(W, int)
            self.assertIsInstance(sv, int)


# ─────────────────────────────────────────────────────────────────────────────
# 7. read_sector (requires DB files)
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_DB_AVAILABLE, _DB_SKIP_REASON)
class TestReadSector(unittest.TestCase):
    def test_read_3_3_0_0(self):
        path = _DB_DIR / "std_3_3_0_0.sec2"
        data, hash_count, em_set, vw, vl = read_sector(path)
        self.assertEqual(hash_count, 210140)
        self.assertEqual(len(data), 210140 * 3)
        self.assertEqual(em_set, {})
        self.assertEqual(vw, 299)
        self.assertEqual(vl, -299)

    def test_read_0_0_9_9(self):
        path = _DB_DIR / "std_0_0_9_9.sec2"
        data, hash_count, em_set, _, _ = read_sector(path)
        self.assertEqual(hash_count, 1)  # single entry: empty board
        self.assertEqual(len(data), 3)

    def test_wdl_distribution_3_3(self):
        """Sector (3,3,0,0) should have a reasonable mix of W/L/D/Sym entries."""
        path = _DB_DIR / "std_3_3_0_0.sec2"
        data, hash_count, em_set, vw, vl = read_sector(path)
        outcomes = {"W": 0, "L": 0, "D": 0, "SYM": 0, "Count": 0}
        sample = min(10000, hash_count)
        for i in range(sample):
            raw = _decode_raw_entry(data, i, em_set)
            if raw is not None and raw.kind == "count":
                outcomes["Count"] += 1
            o = decode_entry(data, i, em_set, 0, vw, vl)
            if o is None:
                continue
            elif isinstance(o, tuple) and o[0] == "SYM":
                outcomes["SYM"] += 1
            else:
                outcomes[o] = outcomes.get(o, 0) + 1
        total = outcomes["W"] + outcomes["L"] + outcomes["D"]
        self.assertGreater(total, 0)
        # Count (retrograde in progress) entries should be <2% of sample
        count_frac = outcomes["Count"] / sample if sample > 0 else 0
        self.assertLess(count_frac, 0.02,
                        f"Too many Count entries in finished DB: {outcomes['Count']}/{sample}")
        # Sym entries (~19% of sector) should be detected, not zero
        self.assertGreater(outcomes["SYM"], 0,
                           "Expected some Sym entries in sector (3,3,0,0)")

    def test_all_sector_boundary_samples_follow_reference_projection(self):
        """Read first/middle/last entries from every declared std sector."""
        vw, vl, secvals = parse_secval(_SECVAL_PATH)
        seen = {"value": 0, "count": 0, "symmetry": 0}
        nonzero_raw_draws = 0
        sampled_entries = 0

        for sector, sector_value in sorted(secvals.items()):
            with self.subTest(sector=sector):
                path = _DB_DIR / ("std_%d_%d_%d_%d.sec2" % sector)
                data, hash_count, em_set, _, _ = read_sector(path, vw, vl)
                mapped_file = data.obj
                try:
                    indices = sorted({0, hash_count // 2, hash_count - 1})
                    sampled_entries += len(indices)
                    for idx in indices:
                        raw = _decode_raw_entry(data, idx, em_set)
                        self.assertIsNotNone(raw)
                        seen[raw.kind] += 1

                        actual = decode_entry(
                            data, idx, em_set, sector_value, vw, vl)
                        if raw.kind == "symmetry":
                            expected = ("SYM", raw.symmetry_operation)
                        else:
                            absolute = raw.key1 + sector_value
                            expected = (
                                "W" if absolute == vw
                                else "L" if absolute == vl
                                else "D"
                            )
                            if expected == "D" and raw.key1 != 0:
                                nonzero_raw_draws += 1
                        self.assertEqual(actual, expected)
                finally:
                    data.release()
                    mapped_file.close()

        self.assertEqual(sum(seen.values()), sampled_entries)
        self.assertGreaterEqual(sampled_entries, len(secvals))
        self.assertGreater(seen["count"], 0)
        self.assertGreater(seen["symmetry"], 0)
        self.assertGreater(nonzero_raw_draws, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 8. MalomDB integration (requires DB files)
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_DB_AVAILABLE, _DB_SKIP_REASON)
class TestMalomDB(unittest.TestCase):
    def setUp(self):
        self.db = MalomDB(str(_DB_DIR))

    def tearDown(self):
        self.db.close()

    def test_is_available(self):
        self.assertTrue(self.db.is_available())

    def test_repr_contains_dir(self):
        r = repr(self.db)
        self.assertIn(repr(str(_DB_DIR)), r)

    def test_empty_board_matches_sanmill_reference(self):
        """Golden vector from Sanmill database.rs::evaluates_empty_board."""
        board = BoardState.new_game()
        value = self.db.query_value(board)
        self.assertIsInstance(value, OracleValue)
        self.assertEqual(value.raw_key1, -21)
        self.assertEqual(value.sector_value, 21)
        self.assertEqual(value.absolute_key1, 0)
        self.assertEqual(value.key2, 2)
        self.assertEqual(value.entry_kind, "value")
        self.assertEqual(value.perspective, "W")
        self.assertEqual(value.sector, (0, 0, 9, 9))
        self.assertEqual(value.outcome, "D")
        self.assertEqual(self.db.query(board), {"outcome": "D", "dtw": 2})

    def test_black_to_move_after_a4_matches_sanmill_reference(self):
        """Golden vector from Sanmill's black-to-move side-swap test."""
        board = BoardState.new_game().apply_move(
            {"from": None, "to": "a4", "capture": None}
        )
        value = self.db.query_value(board)
        self.assertIsInstance(value, OracleValue)
        self.assertEqual(value.raw_key1, 18)
        self.assertEqual(value.sector_value, -18)
        self.assertEqual(value.absolute_key1, 0)
        self.assertEqual(value.key2, 1)
        self.assertEqual(value.perspective, "B")
        self.assertEqual(value.sector, (0, 1, 9, 8))
        self.assertEqual(value.outcome, "D")

    def test_symmetry_redirect_matches_sanmill_reference(self):
        """Golden vector from Sanmill's c3 symmetry-redirect test."""
        positions = {position: "" for position in POSITIONS}
        positions["c3"] = "B"
        board = BoardState(
            positions=positions,
            turn="W",
            pieces_on_board={"W": 0, "B": 1},
            pieces_placed={"W": 0, "B": 1},
            pieces_captured={"W": 0, "B": 0},
            hash_key=0,
        )
        value = self.db.query_value(board)
        self.assertIsInstance(value, OracleValue)
        self.assertEqual(value.raw_key1, 18)
        self.assertEqual(value.sector_value, -18)
        self.assertEqual(value.absolute_key1, 0)
        self.assertEqual(value.key2, 1)
        self.assertIsNotNone(value.symmetry_operation)
        self.assertEqual(value.outcome, "D")

    def test_query_endgame_position(self):
        """A fully-placed 3v3 position should return a valid WDL result."""
        # White: a7(1), d5(18), e4(20)
        # Black: g7(3), f4(12), c3(23)
        board = BoardState.from_setup(
            {"a7": "W", "d5": "W", "e4": "W",
             "g7": "B", "f4": "B", "c3": "B"},
            turn="W", phase="move"
        )
        result = self.db.query(board)
        self.assertIsNotNone(result, "Expected WDL result for endgame 3v3 position")
        self.assertIn(result["outcome"], ("W", "L", "D"))
        self.assertIn("dtw", result)

    def test_query_both_sides(self):
        """Same position queried for both sides should both work."""
        board_w = BoardState.from_setup(
            {"a7": "W", "d5": "W", "e4": "W",
             "g7": "B", "f4": "B", "c3": "B"},
            turn="W", phase="move"
        )
        board_b = BoardState.from_setup(
            {"a7": "W", "d5": "W", "e4": "W",
             "g7": "B", "f4": "B", "c3": "B"},
            turn="B", phase="move"
        )
        result_w = self.db.query(board_w)
        result_b = self.db.query(board_b)
        # Both should return a result (may differ — different hash after swap)
        self.assertIsNotNone(result_w)
        self.assertIsNotNone(result_b)
        self.assertIn(result_w["outcome"], ("W", "L", "D"))
        self.assertIn(result_b["outcome"], ("W", "L", "D"))

    def test_query_wrong_sector_returns_none(self):
        """A position whose sector file doesn't exist should return None."""
        # Build a position with unrealistic stone counts if needed.
        # Simplest: query a new game board for sector (0,0,9,9)
        board = BoardState.new_game()
        # Turn off the DB files temporarily by pointing to a bad path
        db2 = MalomDB("/nonexistent_path")
        result = db2.query(board)
        self.assertIsNone(result)

    def test_hash_deterministic(self):
        """Same position always produces the same hash."""
        board = BoardState.from_setup(
            {"a7": "W", "d5": "W", "e4": "W",
             "g7": "B", "f4": "B", "c3": "B"},
            turn="W", phase="move"
        )
        r1 = self.db.query(board)
        r2 = self.db.query(board)
        self.assertEqual(r1, r2)

    def test_close_then_query_graceful(self):
        """After close(), query() should still work (reloads cache) or return None."""
        board = BoardState.from_setup(
            {"a7": "W", "d5": "W", "e4": "W",
             "g7": "B", "f4": "B", "c3": "B"},
            turn="W", phase="move"
        )
        self.db.close()
        # After close, _available is still True (secval loaded), cache is empty
        result = self.db.query(board)
        # Should not raise; result may be valid
        self.assertIsInstance(result, (dict, type(None)))


# ─────────────────────────────────────────────────────────────────────────────
# 9. ExternalSolvedDB integration via db_teacher
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_DB_AVAILABLE, _DB_SKIP_REASON)
class TestExternalSolvedDB(unittest.TestCase):
    def _load_db_teacher(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "learned_ai.sentinel.db_teacher",
            str(_ROOT / "learned_ai" / "sentinel" / "db_teacher.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_is_available(self):
        mod = self._load_db_teacher()
        db = mod.ExternalSolvedDB(db_path=str(_DB_DIR), enabled=True)
        self.assertTrue(db.is_available())

    def test_query_state_endgame(self):
        mod = self._load_db_teacher()
        db = mod.ExternalSolvedDB(db_path=str(_DB_DIR), enabled=True)
        board = BoardState.from_setup(
            {"a7": "W", "d5": "W", "e4": "W",
             "g7": "B", "f4": "B", "c3": "B"},
            turn="W", phase="move"
        )
        result = db.query_state(board)
        self.assertIn(result, ("W", "L", "D"))

    def test_query_move_quality_uses_settled_four_to_three_capture(self):
        mod = self._load_db_teacher()
        db = mod.ExternalSolvedDB(db_path=str(_DB_DIR), enabled=True)
        board = BoardState.from_setup(
            {
                "a7": "W", "d7": "W", "g4": "W", "b4": "W",
                "a4": "B", "b6": "B", "d6": "B", "f6": "B",
            },
            turn="W",
            phase="move",
        )
        complete = {"from": "g4", "to": "g7", "capture": "a4"}
        incomplete = {"from": "g4", "to": "g7", "capture": None}
        spurious = {"from": "g4", "to": "g1", "capture": "a4"}

        self.assertIsInstance(db.query_move_quality(board, complete), float)
        self.assertIsNone(db.query_move_quality(board, incomplete))
        self.assertIsNone(db.query_move_quality(board, spurious))

        settled = board.apply_move(complete)
        self.assertEqual(settled.turn, "B")
        self.assertEqual(settled.pieces_on_board, {"W": 4, "B": 3})
        self.assertEqual(settled.positions["a4"], "")

    def test_disabled_not_available(self):
        mod = self._load_db_teacher()
        db = mod.ExternalSolvedDB(db_path=str(_DB_DIR), enabled=False)
        self.assertFalse(db.is_available())


if __name__ == "__main__":
    unittest.main()
