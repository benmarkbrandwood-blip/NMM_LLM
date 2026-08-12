"""tests/test_gap_v3_session_ledger.py — session ledger determinism + contents.

Batch 3 Step 0 (docs/gap_net_v3_stage_e_rebuild_checklist.md):
- Every JSONL file's SHA-256, size, mtime, and game count recorded.
- Every session_id assigned via game_level_split(); recorded with session_hash.
- Deterministic: identical inputs → identical output (modulo built_at/elapsed).
- files_manifest_sha256 changes when file contents change.
- File-stem fallback when JSON record has no session_id.
- Duplicate session_ids: first occurrence wins (sorted iteration).
- Missing games_dir raises FileNotFoundError.
- Provenance completeness.
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

from learned_ai.data.human_db_split import game_level_split   # noqa: E402


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_gap_v3_session_ledger",
        _ROOT / "tools" / "build_gap_v3_session_ledger.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


def _write_jsonl_game(path: Path, session_id: str | None,
                      body_suffix: str = "") -> None:
    """Minimal JSONL: one game per file, optional session_id."""
    rec: dict = {"moves": [{"board_fen_before": f"9/9/9 w P 0 0{body_suffix}"}]}
    if session_id is not None:
        rec["session_id"] = session_id
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")


def test_ledger_output_shape(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i, sid in enumerate(["s_alpha", "s_beta", "s_gamma"]):
        _write_jsonl_game(games_dir / f"game_{i}.jsonl", sid)

    out = tmp_path / "ledger.json"
    builder.build(games_dir, out)

    ledger = json.loads(out.read_text())
    assert ledger["ledger_version"] == "gap_v3_session_ledger_v1"
    assert ledger["n_jsonl_files"] == 3
    assert ledger["n_sessions"] == 3
    assert len(ledger["files"]) == 3
    assert len(ledger["sessions"]) == 3


def test_session_hash_matches_sha256(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_jsonl_game(games_dir / "g.jsonl", "abc123")

    out = tmp_path / "ledger.json"
    builder.build(games_dir, out)

    ledger = json.loads(out.read_text())
    entry = ledger["sessions"][0]
    assert entry["session_hash"] == hashlib.sha256(b"abc123").hexdigest()


def test_split_matches_game_level_split(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i in range(20):
        _write_jsonl_game(games_dir / f"game_{i:02d}.jsonl", f"session_{i}")

    out = tmp_path / "ledger.json"
    builder.build(games_dir, out)

    ledger = json.loads(out.read_text())
    for entry in ledger["sessions"]:
        assert entry["split"] == game_level_split(entry["session_id"])


def test_n_by_split_sums_to_n_sessions(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i in range(30):
        _write_jsonl_game(games_dir / f"g_{i:02d}.jsonl", f"sess_{i}")

    out = tmp_path / "ledger.json"
    builder.build(games_dir, out)

    ledger = json.loads(out.read_text())
    counts = ledger["n_by_split"]
    assert counts["train"] + counts["val"] + counts["test"] == ledger["n_sessions"]


def test_deterministic_output(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i, sid in enumerate(["s1", "s2", "s3"]):
        _write_jsonl_game(games_dir / f"g_{i}.jsonl", sid)

    out1 = tmp_path / "ledger1.json"
    out2 = tmp_path / "ledger2.json"
    builder.build(games_dir, out1)
    builder.build(games_dir, out2)

    l1 = json.loads(out1.read_text())
    l2 = json.loads(out2.read_text())
    # Provenance may differ on built_at / elapsed_seconds; compare content parts
    assert l1["files_manifest_sha256"] == l2["files_manifest_sha256"]
    assert l1["files"] == l2["files"]
    assert l1["sessions"] == l2["sessions"]
    assert l1["n_by_split"] == l2["n_by_split"]


def test_manifest_hash_changes_on_content(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_jsonl_game(games_dir / "g.jsonl", "sess")
    out1 = tmp_path / "ledger1.json"
    builder.build(games_dir, out1)
    m1 = json.loads(out1.read_text())["files_manifest_sha256"]

    # Rewrite the same file with different body content → same session_id,
    # different SHA-256 of file → different manifest hash.
    _write_jsonl_game(games_dir / "g.jsonl", "sess", body_suffix="0")
    out2 = tmp_path / "ledger2.json"
    builder.build(games_dir, out2)
    m2 = json.loads(out2.read_text())["files_manifest_sha256"]

    assert m1 != m2


def test_missing_session_id_falls_back_to_file_stem(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_jsonl_game(games_dir / "unique_stem.jsonl", None)

    out = tmp_path / "ledger.json"
    builder.build(games_dir, out)

    ledger = json.loads(out.read_text())
    entry = ledger["sessions"][0]
    assert entry["session_id"] == "unique_stem"
    assert entry["session_source"] == "file_stem"
    assert ledger["n_sessions_from_stem"] == 1


def test_duplicate_session_ids_kept_first_occurrence(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    # Sorted iteration: "a_dupe.jsonl" comes before "z_dupe.jsonl"
    _write_jsonl_game(games_dir / "a_dupe.jsonl", "shared")
    _write_jsonl_game(games_dir / "z_dupe.jsonl", "shared")

    out = tmp_path / "ledger.json"
    builder.build(games_dir, out)

    ledger = json.loads(out.read_text())
    assert ledger["n_sessions"] == 1
    assert ledger["sessions"][0]["source_file"] == "a_dupe.jsonl"


def test_missing_games_dir_raises(builder, tmp_path):
    with pytest.raises(FileNotFoundError):
        builder.build(tmp_path / "nonexistent", tmp_path / "out.json")


def test_provenance_records_all_fields(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_jsonl_game(games_dir / "g.jsonl", "s")

    out = tmp_path / "ledger.json"
    builder.build(games_dir, out)

    ledger = json.loads(out.read_text())
    for key in ("ledger_version", "split_function", "split_manifest_version",
                "git_commit", "built_at", "games_dir", "n_jsonl_files",
                "n_sessions", "n_sessions_from_stem", "n_by_split",
                "files_manifest_sha256", "elapsed_seconds"):
        assert key in ledger, f"missing provenance field: {key}"


def test_file_entry_records_sha_size_mtime_count(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_jsonl_game(games_dir / "g.jsonl", "s1")

    out = tmp_path / "ledger.json"
    builder.build(games_dir, out)

    entry = json.loads(out.read_text())["files"][0]
    for key in ("rel_path", "sha256", "size_bytes", "mtime", "n_games"):
        assert key in entry
    assert entry["n_games"] == 1
    assert len(entry["sha256"]) == 64  # SHA-256 hex


def test_empty_games_dir_fails_closed(builder, tmp_path):
    """Codex P1-B: refuse to write an empty ledger for a directory with no JSONL files."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()

    out = tmp_path / "ledger.json"
    with pytest.raises(builder.LedgerBuildError, match="No .* files"):
        builder.build(games_dir, out)
    assert not out.exists()


# ── P1-B hardening regressions (Codex 2026-08-12) ────────────────────────────

def test_limit_files_marks_partial(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i in range(5):
        _write_jsonl_game(games_dir / f"g_{i}.jsonl", f"s_{i}")

    out = tmp_path / "ledger.json"
    builder.build(games_dir, out, limit_files=2)

    ledger = json.loads(out.read_text())
    assert ledger["is_partial"] is True
    assert ledger["limit_files_arg"] == 2


def test_full_scan_marks_not_partial(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_jsonl_game(games_dir / "g.jsonl", "s")
    out = tmp_path / "ledger.json"
    builder.build(games_dir, out)
    ledger = json.loads(out.read_text())
    assert ledger["is_partial"] is False
    assert ledger["limit_files_arg"] is None


def test_malformed_json_fails_closed_by_default(builder, tmp_path):
    """Codex P1-B: strict mode (default) refuses malformed JSON."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    (games_dir / "g.jsonl").write_text("this is not json\n", encoding="utf-8")

    out = tmp_path / "ledger.json"
    with pytest.raises(builder.LedgerBuildError, match="Malformed JSON"):
        builder.build(games_dir, out)


def test_malformed_json_tolerated_in_non_strict(builder, tmp_path):
    """strict=False records malformed count in provenance without aborting."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    good = json.dumps({"session_id": "s_good", "moves": []})
    (games_dir / "g.jsonl").write_text(
        f"{good}\nnot json\n{good}\n", encoding="utf-8",
    )
    out = tmp_path / "ledger.json"
    builder.build(games_dir, out, strict=False)
    ledger = json.loads(out.read_text())
    assert ledger["strict"] is False
    assert ledger["n_malformed_lines"] == 1


def test_no_valid_sessions_fails_closed(builder, tmp_path):
    """Codex P1-B: refuse to write a ledger with zero sessions."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    # File exists but contains only blank lines
    (games_dir / "empty.jsonl").write_text("\n\n\n", encoding="utf-8")

    out = tmp_path / "ledger.json"
    with pytest.raises(builder.LedgerBuildError, match="No valid sessions"):
        builder.build(games_dir, out)


def test_no_clobber_refuses_existing_output(builder, tmp_path):
    """Codex P1-B: refuse to overwrite existing output unless force=True."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_jsonl_game(games_dir / "g.jsonl", "s")
    out = tmp_path / "ledger.json"
    builder.build(games_dir, out)
    # Second build without force must refuse
    with pytest.raises(builder.LedgerBuildError, match="Refusing to overwrite"):
        builder.build(games_dir, out)


def test_force_overrides_no_clobber(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_jsonl_game(games_dir / "g.jsonl", "s")
    out = tmp_path / "ledger.json"
    builder.build(games_dir, out)
    original_size = out.stat().st_size
    # Add a second game and rebuild with force
    _write_jsonl_game(games_dir / "g2.jsonl", "s2")
    builder.build(games_dir, out, force=True)
    ledger = json.loads(out.read_text())
    assert ledger["n_sessions"] == 2


def test_atomic_publish_leaves_no_tmp_on_success(builder, tmp_path):
    """After a successful build, no leftover .tmp sibling exists."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_jsonl_game(games_dir / "g.jsonl", "s")
    out = tmp_path / "ledger.json"
    builder.build(games_dir, out)
    tmp_sibling = out.with_suffix(out.suffix + ".tmp")
    assert not tmp_sibling.exists()
    assert out.exists()


def test_provenance_includes_p1b_fields(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_jsonl_game(games_dir / "g.jsonl", "s")
    out = tmp_path / "ledger.json"
    builder.build(games_dir, out)
    ledger = json.loads(out.read_text())
    for key in ("is_partial", "limit_files_arg", "strict", "n_malformed_lines"):
        assert key in ledger


# ── _verify_ledger_complete tests ────────────────────────────────────────────

def test_verify_ledger_complete_accepts_clean_ledger(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_jsonl_game(games_dir / "g.jsonl", "s")
    out = tmp_path / "ledger.json"
    builder.build(games_dir, out)
    result = builder._verify_ledger_complete(out)
    assert result["n_sessions"] == 1


def test_verify_ledger_complete_rejects_partial(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i in range(5):
        _write_jsonl_game(games_dir / f"g_{i}.jsonl", f"s_{i}")
    out = tmp_path / "ledger.json"
    builder.build(games_dir, out, limit_files=2)
    with pytest.raises(builder.LedgerBuildError, match="partial"):
        builder._verify_ledger_complete(out)


def test_verify_ledger_complete_allow_partial_overrides(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i in range(5):
        _write_jsonl_game(games_dir / f"g_{i}.jsonl", f"s_{i}")
    out = tmp_path / "ledger.json"
    builder.build(games_dir, out, limit_files=2)
    result = builder._verify_ledger_complete(out, allow_partial=True)
    assert result["is_partial"] is True


def test_verify_ledger_complete_rejects_non_strict(builder, tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    good = json.dumps({"session_id": "s", "moves": []})
    (games_dir / "g.jsonl").write_text(f"{good}\nbad\n", encoding="utf-8")
    out = tmp_path / "ledger.json"
    builder.build(games_dir, out, strict=False)
    with pytest.raises(builder.LedgerBuildError, match="strict=False|malformed"):
        builder._verify_ledger_complete(out)
