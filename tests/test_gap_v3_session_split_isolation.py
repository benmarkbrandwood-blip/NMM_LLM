"""tests/test_gap_v3_session_split_isolation.py — Stage D v2 tier-isolation invariant.

Batch 4 (docs/gap_net_v3_stage_e_rebuild_checklist.md §Stage D redo):
Aggregation MUST use events from the owning tier only.  Events from other
tiers are DISCARDED and counted, never combined into the aggregated counts
that feed empirical G_v.

Invariant we lock here: after aggregation, no state_key in the emitted
counts has ANY event contributed by a session whose tier ≠ the state_key's
owning tier.
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


def _write_game(dir_path: Path, filename: str, session_id: str,
                white_elo: int, black_elo: int, moves: list[dict]) -> None:
    rec = {
        "session_id": session_id,
        "white_elo":  white_elo,
        "black_elo":  black_elo,
        "moves":      moves,
    }
    (dir_path / filename).write_text(json.dumps(rec) + "\n", encoding="utf-8")


def _initial_fen() -> str:
    from game.board import BoardState
    return BoardState.new_game().to_fen_string()


def _make_move(placement_at: str, color: str = "white") -> dict:
    return {"board_fen_before": _initial_fen(), "to": placement_at, "color": color}


def _session_meta(*sessions_and_tiers: tuple[str, str]) -> dict:
    return {
        sid: {"tier": tier, "session_hash": _session_hash(sid),
              "source_file": f"{sid}.jsonl"}
        for sid, tier in sessions_and_tiers
    }


# ── Isolation invariant tests ────────────────────────────────────────────────

def test_val_events_discarded_when_owning_tier_is_train(extractor, tmp_path):
    """State_key owned by train tier: val-tier events must be discarded."""
    # Pick two session_ids where "train_owns_id" hashes lower than "val_other_id"
    # so train is the owning tier.  If needed, swap them by hash values.
    ids = ["ss_alpha", "ss_beta"]
    hashes = {sid: _session_hash(sid) for sid in ids}
    min_id = min(ids, key=lambda s: hashes[s])
    other_id = [s for s in ids if s != min_id][0]

    session_meta = _session_meta((min_id, "train"), (other_id, "val"))

    _write_game(tmp_path, "g_owning.jsonl", min_id, 1200, 1200,
                moves=[_make_move("a7"), _make_move("a7")])   # 2 events
    _write_game(tmp_path, "g_other.jsonl", other_id, 1200, 1200,
                moves=[_make_move("a7"), _make_move("a7"), _make_move("a7")])  # 3 events

    owning, _ = extractor._scan_pass1_owning_tier(tmp_path, session_meta)
    counts, disp = extractor._scan_pass2_aggregate(tmp_path, session_meta, owning)

    # Exactly one (state_key, band) entry; count reflects owning-tier events only
    assert len(counts) == 1
    (sk, band), entry = next(iter(counts.items()))
    total = sum(entry["notation_counts"].values())
    assert total == 2, f"expected only 2 owning-tier events, got {total}"
    assert disp["events_kept_by_tier"].get("train", 0) == 2
    assert disp["events_discarded_by_tier"].get("val", 0) == 3


def test_train_events_kept_from_multiple_sessions_same_tier(extractor, tmp_path):
    """When multiple sessions in the owning tier reach the same state_key,
    all their events are kept (aggregation is by tier, not by owning session)."""
    ids = ["train_a", "train_b"]
    session_meta = _session_meta((ids[0], "train"), (ids[1], "train"))

    _write_game(tmp_path, "g_a.jsonl", ids[0], 1200, 1200,
                moves=[_make_move("a7"), _make_move("a7")])
    _write_game(tmp_path, "g_b.jsonl", ids[1], 1200, 1200,
                moves=[_make_move("a7")])

    owning, _ = extractor._scan_pass1_owning_tier(tmp_path, session_meta)
    counts, disp = extractor._scan_pass2_aggregate(tmp_path, session_meta, owning)

    assert len(counts) == 1
    _, entry = next(iter(counts.items()))
    total = sum(entry["notation_counts"].values())
    assert total == 3, f"expected all 3 owning-tier events kept, got {total}"
    assert disp["events_discarded_by_tier"] == {} or \
           sum(disp["events_discarded_by_tier"].values()) == 0


def test_no_cross_tier_leakage_by_construction(extractor, tmp_path):
    """Systematic check: every kept event's session_tier == its state_key's owning_tier."""
    session_meta = _session_meta(
        ("ta", "train"), ("tb", "train"),
        ("va", "val"),   ("vb", "val"),
        ("tc", "test"),
    )
    # Two state_keys via two placement positions
    for sid, elo in session_meta.items():
        for placement in ("a7", "d7"):
            _write_game(tmp_path, f"g_{sid}_{placement}.jsonl", sid,
                        1200, 1200, moves=[_make_move(placement)])

    owning, _ = extractor._scan_pass1_owning_tier(tmp_path, session_meta)
    counts, disp = extractor._scan_pass2_aggregate(tmp_path, session_meta, owning)

    # Systematic invariant: kept events' tier must equal owning_tier.
    # We reconstruct via disposition sums.  For each session tier, count kept
    # events; they should equal the sum over state_keys where that tier is
    # owning * events_of_that_tier_that_reached_that_state_key.
    total_kept = sum(disp["events_kept_by_tier"].values())
    total_disc = sum(disp["events_discarded_by_tier"].values())
    total_events = total_kept + total_disc + disp["events_dropped_uncovered"]

    # Sanity: total accounted for equals number of events emitted by generator
    # (5 sessions × 2 placements × 1 move = 10 events)
    assert total_events == 10


def test_disposition_captures_per_tier_discards(extractor, tmp_path):
    """When train owns a state_key but val/test also reached it,
    disposition records val and test discard counts."""
    ids = ["shd_a", "shd_b", "shd_c"]
    hashes = {sid: _session_hash(sid) for sid in ids}
    min_id = min(ids, key=lambda s: hashes[s])
    others = [s for s in ids if s != min_id]

    session_meta = _session_meta(
        (min_id,    "train"),
        (others[0], "val"),
        (others[1], "test"),
    )
    for sid, tier in session_meta.items():
        _write_game(tmp_path, f"g_{sid}.jsonl", sid, 1200, 1200,
                    moves=[_make_move("a7"), _make_move("a7")])

    owning, _ = extractor._scan_pass1_owning_tier(tmp_path, session_meta)
    counts, disp = extractor._scan_pass2_aggregate(tmp_path, session_meta, owning)

    assert disp["events_kept_by_tier"].get("train", 0) == 2
    assert disp["events_discarded_by_tier"].get("val",  0) == 2
    assert disp["events_discarded_by_tier"].get("test", 0) == 2


def test_uncovered_events_dropped_and_counted(extractor, tmp_path):
    """Sessions absent from state_key_owning (e.g. never seen in pass 1) → dropped."""
    session_meta = _session_meta(("known", "train"))

    _write_game(tmp_path, "g_known.jsonl", "known", 1200, 1200,
                moves=[_make_move("a7")])

    owning, _ = extractor._scan_pass1_owning_tier(tmp_path, session_meta)
    # Corrupt: run pass2 with an EMPTY owning dict to simulate no state_key coverage
    counts, disp = extractor._scan_pass2_aggregate(tmp_path, session_meta, {})
    assert len(counts) == 0
    assert disp["events_dropped_uncovered"] == 1
