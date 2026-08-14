"""tests/test_gap_v3_owning_tier_rule.py — Stage D v2 owning-tier rule.

Batch 4 (docs/gap_net_v3_stage_e_rebuild_checklist.md §Stage D redo):
- For each state_key, the session with the smallest SHA-256 hash is the
  owning session; its split_tier is the state_key's owning tier.
- Aggregation uses events from the owning tier only.
- Events from other tiers are discarded and counted in disposition.
- Ties (impossible with cryptographic hashes but tested for stability):
  when hashes are equal, first-seen wins.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _load_extractor():
    spec = importlib.util.spec_from_file_location(
        "extract_gap_v3_dataset_v2",
        _ROOT / "tools" / "extract_gap_v3_dataset_v2.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def extractor():
    return _load_extractor()


def _session_hash(sid: str) -> str:
    return hashlib.sha256(sid.encode("utf-8")).hexdigest()


def _write_game(
    dir_path: Path, filename: str, session_id: str,
    white_elo: int, black_elo: int,
    moves: list[dict],
) -> None:
    rec = {
        "session_id": session_id,
        "white_elo":  white_elo,
        "black_elo":  black_elo,
        "moves":      moves,
    }
    (dir_path / filename).write_text(json.dumps(rec) + "\n", encoding="utf-8")


def _initial_fen() -> str:
    """FEN string of the initial NMM board (no pieces placed)."""
    from game.board import BoardState
    return BoardState.new_game().to_fen_string()


def _apply_move_fen(before_fen: str, placement_at: str) -> tuple[str, str]:
    """Return (before_fen, after_fen) for a placement move at `placement_at`."""
    from game.board import BoardState
    board = BoardState.from_fen_string(before_fen)
    board.apply_move({"from": None, "to": placement_at})
    return before_fen, board.to_fen_string()


def _session_meta_of(*sessions_and_tiers: tuple[str, str]) -> dict:
    """Build a session_meta dict {sid: {tier, session_hash, source_file}} from pairs."""
    out = {}
    for sid, tier in sessions_and_tiers:
        out[sid] = {
            "tier":         tier,
            "session_hash": _session_hash(sid),
            "source_file":  f"{sid}.jsonl",
        }
    return out


# ── Owning-tier assignment tests ────────────────────────────────────────────

def test_single_session_owns_state_key(extractor, tmp_path):
    """A state_key reached by only one session takes that session's tier."""
    session_meta = _session_meta_of(("s_train", "train"))
    fen_before, _ = _apply_move_fen(_initial_fen(), "a7")
    _write_game(tmp_path, "g1.jsonl", "s_train", 1200, 1200,
                moves=[{"board_fen_before": fen_before, "to": "a7", "color": "white"}])

    owning, stats = extractor._scan_pass1_owning_tier(tmp_path, session_meta)
    assert stats["n_events_total"] == 1
    assert len(owning) == 1
    (sk, entry), = owning.items()
    assert entry["tier"] == "train"
    assert entry["session_id"] == "s_train"


def test_multi_session_owning_smallest_hash(extractor, tmp_path):
    """When multiple sessions reach one state_key, owning tier = smallest-hash session's tier."""
    # Use session IDs whose hashes have a known ordering
    ids = ["session_alpha", "session_beta", "session_gamma"]
    hashes = {sid: _session_hash(sid) for sid in ids}
    min_id = min(ids, key=lambda s: hashes[s])
    other_ids = [s for s in ids if s != min_id]

    session_meta = _session_meta_of(
        (min_id, "val"),
        (other_ids[0], "train"),
        (other_ids[1], "test"),
    )
    fen_before, _ = _apply_move_fen(_initial_fen(), "a7")
    move = [{"board_fen_before": fen_before, "to": "a7", "color": "white"}]
    _write_game(tmp_path, "g_min.jsonl",   min_id,       1200, 1200, moves=move)
    _write_game(tmp_path, "g_other0.jsonl", other_ids[0], 1200, 1200, moves=move)
    _write_game(tmp_path, "g_other1.jsonl", other_ids[1], 1200, 1200, moves=move)

    owning, _ = extractor._scan_pass1_owning_tier(tmp_path, session_meta)
    assert len(owning) == 1
    (sk, entry), = owning.items()
    assert entry["session_id"] == min_id
    assert entry["session_hash"] == hashes[min_id]
    assert entry["tier"] == "val"


def test_tier_event_counts_across_all_tiers(extractor, tmp_path):
    """tier_event_counts on the owning entry records events from every tier that hit the state_key."""
    ids = ["a_train", "b_val", "c_test"]
    session_meta = _session_meta_of(
        ("a_train", "train"), ("b_val", "val"), ("c_test", "test"),
    )
    fen_before, _ = _apply_move_fen(_initial_fen(), "a7")
    move = [{"board_fen_before": fen_before, "to": "a7", "color": "white"}]
    _write_game(tmp_path, "g_a.jsonl", "a_train", 1200, 1200, moves=move)
    _write_game(tmp_path, "g_b.jsonl", "b_val",   1200, 1200, moves=move)
    _write_game(tmp_path, "g_c.jsonl", "c_test",  1200, 1200, moves=move)

    owning, _ = extractor._scan_pass1_owning_tier(tmp_path, session_meta)
    assert len(owning) == 1
    _, entry = next(iter(owning.items()))
    assert entry["tier_event_counts"] == {"train": 1, "val": 1, "test": 1}


def test_unknown_session_id_skipped(extractor, tmp_path):
    """An event whose session_id is not in the ledger is skipped, not aborted."""
    session_meta = _session_meta_of(("known", "train"))
    fen_before, _ = _apply_move_fen(_initial_fen(), "a7")
    move = [{"board_fen_before": fen_before, "to": "a7", "color": "white"}]
    _write_game(tmp_path, "g_known.jsonl",   "known",   1200, 1200, moves=move)
    _write_game(tmp_path, "g_unknown.jsonl", "unknown", 1200, 1200, moves=move)

    owning, stats = extractor._scan_pass1_owning_tier(tmp_path, session_meta)
    assert stats["n_events_total"] == 1
    assert len(owning) == 1


def test_owning_tier_deterministic_across_runs(extractor, tmp_path):
    """Identical inputs → identical owning assignments."""
    ids = ["a_train", "b_val", "c_test"]
    session_meta = _session_meta_of(
        ("a_train", "train"), ("b_val", "val"), ("c_test", "test"),
    )
    fen_before, _ = _apply_move_fen(_initial_fen(), "a7")
    move = [{"board_fen_before": fen_before, "to": "a7", "color": "white"}]
    _write_game(tmp_path, "g_a.jsonl", "a_train", 1200, 1200, moves=move)
    _write_game(tmp_path, "g_b.jsonl", "b_val",   1200, 1200, moves=move)
    _write_game(tmp_path, "g_c.jsonl", "c_test",  1200, 1200, moves=move)

    owning1, _ = extractor._scan_pass1_owning_tier(tmp_path, session_meta)
    owning2, _ = extractor._scan_pass1_owning_tier(tmp_path, session_meta)
    assert set(owning1.keys()) == set(owning2.keys())
    for sk in owning1:
        assert owning1[sk]["tier"]         == owning2[sk]["tier"]
        assert owning1[sk]["session_id"]   == owning2[sk]["session_id"]
        assert owning1[sk]["session_hash"] == owning2[sk]["session_hash"]
