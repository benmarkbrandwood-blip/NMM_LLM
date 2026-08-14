"""tests/test_gap_v3_extractor_v2_hardening.py — Batch 4 P1-B/P1-A hardening.

Batch 4 extractor (tools/extract_gap_v3_dataset_v2.py) consumes the session
ledger.  Codex-pattern hardening 2026-08-14:
- Ledger loaded via _verify_ledger_complete: partial/non-strict/malformed-
  tolerating ledgers refused unless allow_partial=True.
- JSONL files verified against ledger's file manifest (SHA-256 + presence).
  Drift is fatal; missing files fatal under full run, skipped under smoke.
- Malformed JSON lines fatal under strict=True (default).
- No-clobber: refuse to write into --out-dir with existing outputs unless
  force=True.
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


def _load_ledger_builder():
    spec = importlib.util.spec_from_file_location(
        "build_gap_v3_session_ledger",
        _ROOT / "tools" / "build_gap_v3_session_ledger.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def extractor():
    return _load_extractor()


@pytest.fixture(scope="module")
def ledger_builder():
    return _load_ledger_builder()


def _initial_fen() -> str:
    from game.board import BoardState
    return BoardState.new_game().to_fen_string()


def _write_game(dir_path: Path, filename: str, session_id: str) -> None:
    rec = {
        "session_id": session_id,
        "moves": [{"board_fen_before": _initial_fen(),
                   "to": "a7", "color": "white"}],
    }
    (dir_path / filename).write_text(json.dumps(rec) + "\n", encoding="utf-8")


# ── _load_session_ledger hardening ──────────────────────────────────────────

def test_load_session_ledger_accepts_clean(extractor, ledger_builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "g.jsonl", "s")

    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)

    info = extractor._load_session_ledger(ledger_path)
    assert "s" in info["session_meta"]
    assert info["ledger_file_shas"]         # non-empty
    assert info["provenance"]["is_partial"] is False


def test_load_session_ledger_rejects_partial(extractor, ledger_builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i in range(3):
        _write_game(games_dir, f"g_{i}.jsonl", f"s_{i}")

    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path, limit_files=1)   # partial

    with pytest.raises(RuntimeError, match="verification failed"):
        extractor._load_session_ledger(ledger_path)


def test_load_session_ledger_allow_partial_overrides(extractor, ledger_builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i in range(3):
        _write_game(games_dir, f"g_{i}.jsonl", f"s_{i}")

    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path, limit_files=1)

    info = extractor._load_session_ledger(ledger_path, allow_partial=True)
    assert info["provenance"]["is_partial"] is True


def test_load_session_ledger_rejects_non_strict(extractor, ledger_builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    good = json.dumps({"session_id": "s", "moves": []})
    (games_dir / "g.jsonl").write_text(f"{good}\nbad\n", encoding="utf-8")

    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path, strict=False)   # tolerated malformed

    with pytest.raises(RuntimeError, match="verification failed"):
        extractor._load_session_ledger(ledger_path)


# ── _iter_jsonl_events file-manifest verification ───────────────────────────

def test_iter_events_fails_on_sha_drift(extractor, ledger_builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "g.jsonl", "s")
    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)
    info = extractor._load_session_ledger(ledger_path)

    # Modify JSONL after ledger build → SHA drift
    _write_game(games_dir, "g.jsonl", "s")   # same session_id, different bytes? Actually same content.
    # Force different bytes via appended field:
    rec = {
        "session_id": "s", "extra": "drift",
        "moves": [{"board_fen_before": _initial_fen(),
                   "to": "a7", "color": "white"}],
    }
    (games_dir / "g.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 drift"):
        list(extractor._iter_jsonl_events(
            games_dir, info["session_meta"],
            ledger_file_shas=info["ledger_file_shas"],
        ))


def test_iter_events_fails_on_missing_file_in_full_run(extractor, ledger_builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "known.jsonl", "s")
    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)
    info = extractor._load_session_ledger(ledger_path)

    # Add a new file NOT in ledger manifest
    _write_game(games_dir, "orphan.jsonl", "s_new")

    with pytest.raises(RuntimeError, match="not in ledger file manifest"):
        list(extractor._iter_jsonl_events(
            games_dir, info["session_meta"],
            ledger_file_shas=info["ledger_file_shas"],
        ))


def test_iter_events_skips_missing_file_under_smoke_run(extractor, ledger_builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "known.jsonl", "s")
    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)
    info = extractor._load_session_ledger(ledger_path)

    _write_game(games_dir, "orphan.jsonl", "s_new")

    # limit_files=2 → smoke run → orphan skipped rather than fatal
    events = list(extractor._iter_jsonl_events(
        games_dir, info["session_meta"], limit_files=2,
        ledger_file_shas=info["ledger_file_shas"],
    ))
    # Only known-file events retained; s_new is skipped as unknown session too
    for _sk, sid, _b, _n, _p, _m in events:
        assert sid == "s"


def test_iter_events_fails_closed_on_malformed_json_strict(extractor, ledger_builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    good = json.dumps({"session_id": "s",
                       "moves": [{"board_fen_before": _initial_fen(),
                                  "to": "a7", "color": "white"}]})
    (games_dir / "g.jsonl").write_text(f"{good}\nnot json\n{good}\n", encoding="utf-8")
    ledger_path = tmp_path / "ledger.json"
    # Build ledger with non-strict so the ledger itself succeeds — then extractor
    # runs its own strict check.
    ledger_builder.build(games_dir, ledger_path, strict=False)

    # But extractor's _load_session_ledger will refuse the non-strict ledger.
    # Skip verification by constructing session_meta manually.
    session_meta = {"s": {"tier": "train", "session_hash": "abc",
                          "source_file": "g.jsonl"}}
    with pytest.raises(RuntimeError, match="malformed JSON"):
        list(extractor._iter_jsonl_events(
            games_dir, session_meta, strict=True,
        ))


def test_iter_events_tolerates_malformed_in_non_strict(extractor, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    good = json.dumps({"session_id": "s",
                       "white_elo": 1200, "black_elo": 1200,
                       "moves": [{"board_fen_before": _initial_fen(),
                                  "to": "a7", "color": "white"}]})
    (games_dir / "g.jsonl").write_text(f"{good}\nnot json\n{good}\n", encoding="utf-8")

    session_meta = {"s": {"tier": "train", "session_hash": "abc",
                          "source_file": "g.jsonl"}}
    events = list(extractor._iter_jsonl_events(
        games_dir, session_meta, strict=False,
    ))
    # Two valid records → 2 events at least
    assert len(events) >= 2


# ── run_extraction no-clobber ───────────────────────────────────────────────

def test_run_extraction_refuses_existing_outputs(extractor, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Simulate a pre-existing target file
    (out_dir / "parent_feats.f32.bin").write_bytes(b"stub")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        extractor.run_extraction(
            games_dir=tmp_path / "games",   # doesn't matter — guard fires first
            ledger_path=tmp_path / "ledger.json",
            teacher_net=tmp_path / "net.npz",
            malom_db_dir="/nonexistent",
            out_dir=out_dir,
        )


def test_run_extraction_empty_dir_ok(extractor, tmp_path):
    """A pre-existing empty out_dir must NOT trigger no-clobber."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Downstream fails on missing ledger; we only care no-clobber didn't fire.
    with pytest.raises((FileNotFoundError, RuntimeError)) as exc:
        extractor.run_extraction(
            games_dir=tmp_path / "games",
            ledger_path=tmp_path / "ledger.json",
            teacher_net=tmp_path / "net.npz",
            malom_db_dir="/nonexistent",
            out_dir=out_dir,
        )
    # The error must NOT be the no-clobber refusal
    assert "Refusing to overwrite" not in str(exc.value)
