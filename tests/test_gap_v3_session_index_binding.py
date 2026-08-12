"""tests/test_gap_v3_session_index_binding.py — session_index binds to ledger (P1-A).

Batch 3b Codex P1-A fix (2026-08-12):
tools/build_session_index.py must:
- Require --session-ledger PATH.
- Use ledger's session→split assignments (not recomputed game_level_split).
- Verify each scanned JSONL file appears in the ledger's file manifest with
  matching SHA-256.
- Record ledger identity (ledger_sha256, ledger_files_manifest_sha256,
  ledger_version, split_manifest_version, ledger_n_sessions) in the output
  provenance so downstream (HMPN extractor) can verify binding.
- Fail-closed on partial ledger unless --allow-partial-ledger.
- No-clobber existing output unless --force.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _load_index_builder():
    spec = importlib.util.spec_from_file_location(
        "build_session_index", _ROOT / "tools" / "build_session_index.py",
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
def index_builder():
    return _load_index_builder()


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


def _write_dataset_metadata(dir_path: Path, state_keys: list[str]) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    meta_path = dir_path / "metadata.npz"
    np.savez(
        str(meta_path),
        state_keys=np.array(state_keys, dtype=object),
        provenance=np.array("{}", dtype=object),
    )
    return meta_path


def _initial_state_key() -> str:
    from ai.trajectory_db import make_board_state_key
    from game.board import BoardState
    sk, _ = make_board_state_key(BoardState.new_game())
    return str(sk)


# ── Binding tests ────────────────────────────────────────────────────────────

def test_index_provenance_records_ledger_identity(
    index_builder, ledger_builder, tmp_path,
):
    """P1-A: session_index.provenance carries ledger_sha256 + files_manifest_sha256."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "g.jsonl", "s")

    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)

    dataset_dir = tmp_path / "hmpn_ds"
    _write_dataset_metadata(dataset_dir, [_initial_state_key()])

    idx_path = tmp_path / "session_index.npz"
    index_builder.build(
        dataset_dir, games_dir, ledger_path, idx_path,
    )

    idx = np.load(str(idx_path), allow_pickle=True)
    prov = json.loads(str(idx["provenance"]))
    assert prov["builder_version"] == "2"   # P1-A bumped
    assert prov["ledger_sha256"] == hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    assert prov["ledger_files_manifest_sha256"]
    ledger_data = json.loads(ledger_path.read_text())
    assert prov["ledger_files_manifest_sha256"] == ledger_data["files_manifest_sha256"]
    assert prov["ledger_version"] == "gap_v3_session_ledger_v1"


def test_index_uses_ledger_split_not_recomputed(
    index_builder, ledger_builder, tmp_path,
):
    """P1-A: the split reflected in game_split_mask comes from the ledger."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "g.jsonl", "some_session")

    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)
    ledger = json.loads(ledger_path.read_text())
    session_split = ledger["sessions"][0]["split"]

    dataset_dir = tmp_path / "hmpn_ds"
    _write_dataset_metadata(dataset_dir, [_initial_state_key()])

    idx_path = tmp_path / "session_index.npz"
    index_builder.build(dataset_dir, games_dir, ledger_path, idx_path)

    idx = np.load(str(idx_path), allow_pickle=True)
    mask = int(idx["game_split_mask"][0])
    expected = int(index_builder._SPLIT_TO_MASK[session_split])
    assert mask & expected, (
        f"state_key mask {bin(mask)} should have split={session_split} bit set"
    )


def test_index_fails_closed_on_file_sha_drift(
    index_builder, ledger_builder, tmp_path,
):
    """P1-A: if a JSONL file has changed since the ledger, the build must fail."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "g.jsonl", "s")

    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)

    # Modify the JSONL file after the ledger was built
    (games_dir / "g.jsonl").write_text(
        json.dumps({"session_id": "s", "moves": [], "extra": "drift"}) + "\n",
        encoding="utf-8",
    )

    dataset_dir = tmp_path / "hmpn_ds"
    _write_dataset_metadata(dataset_dir, [_initial_state_key()])
    idx_path = tmp_path / "session_index.npz"
    with pytest.raises(index_builder.SessionIndexBuildError, match="SHA-256 mismatch"):
        index_builder.build(dataset_dir, games_dir, ledger_path, idx_path)


def test_index_fails_closed_on_partial_ledger_by_default(
    index_builder, ledger_builder, tmp_path,
):
    """P1-A: partial ledger refused unless allow_partial_ledger=True."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i in range(3):
        _write_game(games_dir, f"g_{i}.jsonl", f"s_{i}")

    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path, limit_files=1)   # partial

    dataset_dir = tmp_path / "hmpn_ds"
    _write_dataset_metadata(dataset_dir, [_initial_state_key()])
    idx_path = tmp_path / "session_index.npz"
    with pytest.raises(index_builder.SessionIndexBuildError, match="Ledger verification failed"):
        index_builder.build(dataset_dir, games_dir, ledger_path, idx_path)


def test_index_accepts_partial_ledger_with_flag(
    index_builder, ledger_builder, tmp_path,
):
    """Smoke-test path: allow_partial_ledger permits building against a partial ledger."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i in range(3):
        _write_game(games_dir, f"g_{i}.jsonl", f"s_{i}")

    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path, limit_files=1)

    dataset_dir = tmp_path / "hmpn_ds"
    _write_dataset_metadata(dataset_dir, [_initial_state_key()])
    idx_path = tmp_path / "session_index.npz"
    # Under smoke run + partial-ledger, unmentioned files are skipped rather
    # than fatal, so this must succeed.
    index_builder.build(
        dataset_dir, games_dir, ledger_path, idx_path,
        limit_files=1, allow_partial_ledger=True,
    )
    assert idx_path.exists()


def test_index_no_clobber_refuses_existing_output(
    index_builder, ledger_builder, tmp_path,
):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "g.jsonl", "s")
    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)
    dataset_dir = tmp_path / "hmpn_ds"
    _write_dataset_metadata(dataset_dir, [_initial_state_key()])
    idx_path = tmp_path / "session_index.npz"
    index_builder.build(dataset_dir, games_dir, ledger_path, idx_path)

    with pytest.raises(index_builder.SessionIndexBuildError, match="Refusing to overwrite"):
        index_builder.build(dataset_dir, games_dir, ledger_path, idx_path)


def test_index_force_overrides_no_clobber(
    index_builder, ledger_builder, tmp_path,
):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "g.jsonl", "s")
    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)
    dataset_dir = tmp_path / "hmpn_ds"
    _write_dataset_metadata(dataset_dir, [_initial_state_key()])
    idx_path = tmp_path / "session_index.npz"
    index_builder.build(dataset_dir, games_dir, ledger_path, idx_path)
    # Should succeed with force
    index_builder.build(dataset_dir, games_dir, ledger_path, idx_path, force=True)


def test_index_counts_sessions_not_in_ledger(
    index_builder, ledger_builder, tmp_path,
):
    """A session_id absent from ledger is counted, not silently absorbed."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_game(games_dir, "g_known.jsonl", "known")

    ledger_path = tmp_path / "ledger.json"
    ledger_builder.build(games_dir, ledger_path)

    # Add another game AFTER ledger build but its SHA won't match — it's a
    # new file, not in ledger.  Under --limit-files we skip; under full it
    # fatals.  For this test, use --limit-files to reach the counting path.
    _write_game(games_dir, "g_extra.jsonl", "extra_session")

    dataset_dir = tmp_path / "hmpn_ds"
    _write_dataset_metadata(dataset_dir, [_initial_state_key()])
    idx_path = tmp_path / "session_index.npz"

    # limit_files=2 to include the new file, and allow_partial_ledger=False
    # because our ledger is full — but the extra file isn't in it.
    # The build must skip files missing from the ledger under limit_files;
    # sessions not in ledger get counted.
    prov = index_builder.build(
        dataset_dir, games_dir, ledger_path, idx_path,
        limit_files=2,
    )
    assert prov["n_files_missing_from_ledger"] >= 1
