"""tests/test_gap_v3_hmpn_trainer_ledger_guard.py — HMPN trainer safety guard.

Batch 3b + Codex P1-C hardening (2026-08-12):
The trainer's output-path guard now enforces:
  (a) No-clobber: refuse existing --output unless --force.
  (b) V2 filename refused when dataset is session-ledger-isolated.
  (c) V3 filename requires the dataset to have split_scheme=
      session_ledger_strict_single_tier (missing/legacy provenance refused).
  (d) When the dataset is session-ledger-isolated, --session-ledger PATH
      must be provided AND its SHA + files_manifest_sha256 must match the
      dataset's recorded values.
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


def _load_trainer_module():
    spec = importlib.util.spec_from_file_location(
        "train_human_move_policy_net",
        _ROOT / "tools" / "train_human_move_policy_net.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trainer():
    return _load_trainer_module()


def _write_metadata_with_provenance(dir_path: Path, provenance: dict) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(dir_path / "metadata.npz"),
        state_keys=np.array(["sk"], dtype=object),
        provenance=np.array(json.dumps(provenance), dtype=object),
    )


# ── _peek_dataset_provenance ─────────────────────────────────────────────────

def test_peek_returns_dataset_provenance(trainer, tmp_path):
    prov_in = {"split_scheme": "session_ledger_strict_single_tier",
               "session_ledger_sha256": "deadbeef" * 8}
    _write_metadata_with_provenance(tmp_path, prov_in)
    prov_out = trainer._peek_dataset_provenance(tmp_path)
    assert prov_out == prov_in


def test_peek_returns_empty_on_missing_metadata(trainer, tmp_path):
    assert trainer._peek_dataset_provenance(tmp_path / "nonexistent") == {}


def test_peek_returns_empty_when_provenance_absent(trainer, tmp_path):
    tmp_path.mkdir(exist_ok=True)
    np.savez(
        str(tmp_path / "metadata.npz"),
        state_keys=np.array(["sk"], dtype=object),   # no provenance key
    )
    assert trainer._peek_dataset_provenance(tmp_path) == {}


# ── _guard_output_path helpers ───────────────────────────────────────────────

def _write_synthetic_ledger(
    tmp_path: Path,
    filename: str = "ledger.json",
    files_manifest_sha256: str = "b" * 64,
    is_partial: bool = False,
    strict: bool = True,
    n_malformed_lines: int = 0,
    n_sessions: int = 1,
) -> Path:
    ledger = {
        "ledger_version":            "gap_v3_session_ledger_v1",
        "is_partial":                is_partial,
        "strict":                    strict,
        "n_malformed_lines":         n_malformed_lines,
        "n_sessions":                n_sessions,
        "files_manifest_sha256":     files_manifest_sha256,
        "sessions": [{"session_id": "s", "session_hash": "abc", "split": "train",
                      "source_file": "s.jsonl", "session_source": "record"}],
        "files": [],
    }
    path = tmp_path / filename
    path.write_text(json.dumps(ledger), encoding="utf-8")
    return path


def _session_ledger_prov(ledger_sha: str, manifest_sha: str = "b" * 64) -> dict:
    return {
        "split_scheme":                         "session_ledger_strict_single_tier",
        "session_ledger_sha256":                ledger_sha,
        "session_ledger_files_manifest_sha256": manifest_sha,
    }


# ── (b) V2 filename refused for session-ledger dataset ──────────────────────

def test_guard_blocks_default_v2_output_with_session_ledger(trainer, tmp_path):
    """Session-ledger dataset + v2 default output → refuse (rule b)."""
    v2_absolute = trainer._ROOT / trainer._V2_TEACHER_OUTPUT
    ledger_path = _write_synthetic_ledger(tmp_path)
    ledger_sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    prov = _session_ledger_prov(ledger_sha)
    # force=True bypasses the no-clobber (rule a); we're testing rule (b).
    with pytest.raises(SystemExit, match="v2 teacher"):
        trainer._guard_output_path(
            v2_absolute, prov,
            session_ledger_path=ledger_path, force=True,
        )


# ── (c) V3 filename requires session-ledger dataset ─────────────────────────

def test_guard_blocks_v3_output_with_non_session_ledger_dataset(trainer):
    """Rule (c): v3 filename with non-session-ledger dataset → refuse."""
    v3_absolute = trainer._ROOT / trainer._V3_TEACHER_OUTPUT
    prov = {"split_scheme": "state_key_three_way"}
    with pytest.raises(SystemExit, match="v3 teacher"):
        trainer._guard_output_path(v3_absolute, prov, force=True)


def test_guard_blocks_v3_output_when_provenance_absent(trainer):
    """Rule (c): v3 filename with empty provenance → refuse (missing/legacy)."""
    v3_absolute = trainer._ROOT / trainer._V3_TEACHER_OUTPUT
    with pytest.raises(SystemExit, match="v3 teacher"):
        trainer._guard_output_path(v3_absolute, {}, force=True)


def test_guard_allows_v3_output_with_session_ledger(trainer, tmp_path):
    """Rule (c) happy path: v3 filename + session-ledger + verified ledger → OK."""
    v3_absolute = trainer._ROOT / trainer._V3_TEACHER_OUTPUT
    ledger_path = _write_synthetic_ledger(tmp_path)
    ledger_sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    prov = _session_ledger_prov(ledger_sha)
    trainer._guard_output_path(
        v3_absolute, prov,
        session_ledger_path=ledger_path, force=True,
    )   # must not raise


# ── (d) Ledger identity verification ────────────────────────────────────────

def test_guard_requires_session_ledger_arg_for_session_ledger_dataset(trainer, tmp_path):
    """Rule (d): session-ledger dataset without --session-ledger arg → refuse."""
    prov = _session_ledger_prov(ledger_sha="a" * 64)
    with pytest.raises(SystemExit, match="--session-ledger PATH is required"):
        trainer._guard_output_path(
            tmp_path / "custom.npz", prov,
            session_ledger_path=None, force=True,
        )


def test_guard_rejects_missing_ledger_sha_in_provenance(trainer, tmp_path):
    """Rule (d): session-ledger dataset without session_ledger_sha256 → refuse."""
    ledger_path = _write_synthetic_ledger(tmp_path)
    prov = {"split_scheme": trainer._SESSION_LEDGER_SPLIT_SCHEME}   # missing sha
    with pytest.raises(SystemExit, match="no session_ledger_sha256"):
        trainer._guard_output_path(
            tmp_path / "custom.npz", prov,
            session_ledger_path=ledger_path, force=True,
        )


def test_guard_rejects_ledger_sha_mismatch(trainer, tmp_path):
    """Rule (d): --session-ledger sha ≠ dataset's recorded sha → refuse."""
    ledger_path = _write_synthetic_ledger(tmp_path)
    prov = _session_ledger_prov(ledger_sha="deadbeef" * 8)   # wrong sha
    with pytest.raises(SystemExit, match="Ledger SHA mismatch"):
        trainer._guard_output_path(
            tmp_path / "custom.npz", prov,
            session_ledger_path=ledger_path, force=True,
        )


def test_guard_rejects_manifest_sha_mismatch(trainer, tmp_path):
    """Rule (d): files_manifest_sha256 disagreement → refuse."""
    ledger_path = _write_synthetic_ledger(tmp_path, files_manifest_sha256="b" * 64)
    ledger_sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    prov = _session_ledger_prov(ledger_sha, manifest_sha="c" * 64)   # wrong manifest
    with pytest.raises(SystemExit, match="files_manifest_sha256 mismatch"):
        trainer._guard_output_path(
            tmp_path / "custom.npz", prov,
            session_ledger_path=ledger_path, force=True,
        )


def test_guard_rejects_partial_ledger(trainer, tmp_path):
    """Rule (d): --session-ledger built with --limit-files → refuse."""
    ledger_path = _write_synthetic_ledger(tmp_path, is_partial=True)
    ledger_sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    prov = _session_ledger_prov(ledger_sha)
    with pytest.raises(SystemExit, match="verification failed"):
        trainer._guard_output_path(
            tmp_path / "custom.npz", prov,
            session_ledger_path=ledger_path, force=True,
        )


def test_guard_accepts_custom_output_when_ledger_matches(trainer, tmp_path):
    """Rule (d) happy path: custom output + matching ledger → OK."""
    ledger_path = _write_synthetic_ledger(tmp_path, files_manifest_sha256="b" * 64)
    ledger_sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    prov = _session_ledger_prov(ledger_sha, manifest_sha="b" * 64)
    trainer._guard_output_path(
        tmp_path / "custom.npz", prov,
        session_ledger_path=ledger_path,
    )   # no force needed — file doesn't exist


# ── (a) No-clobber ──────────────────────────────────────────────────────────

def test_guard_no_clobber_refuses_existing_output(trainer, tmp_path):
    """Rule (a): existing --output refused unless force=True."""
    existing = tmp_path / "already.npz"
    existing.write_bytes(b"stub")
    with pytest.raises(SystemExit, match="Refusing to overwrite existing"):
        trainer._guard_output_path(existing, {})


def test_guard_force_overrides_no_clobber(trainer, tmp_path):
    """Rule (a) override: --force allows overwriting existing output."""
    existing = tmp_path / "already.npz"
    existing.write_bytes(b"stub")
    trainer._guard_output_path(existing, {}, force=True)   # no raise


# ── Non-session-ledger dataset behaviours ────────────────────────────────────

def test_guard_allows_default_v2_output_without_session_ledger(trainer):
    """State-key three-way dataset + v2 default output → OK (rule b noop, rule d skipped)."""
    v2_absolute = trainer._ROOT / trainer._V2_TEACHER_OUTPUT
    prov = {"split_scheme": "state_key_three_way"}
    trainer._guard_output_path(v2_absolute, prov, force=True)   # must not raise


def test_guard_allows_v2_output_when_scheme_absent(trainer):
    """No split_scheme in provenance (v1 dataset) → guard is a no-op past no-clobber."""
    v2_absolute = trainer._ROOT / trainer._V2_TEACHER_OUTPUT
    trainer._guard_output_path(v2_absolute, {}, force=True)   # must not raise


def test_v2_and_v3_output_paths_differ(trainer):
    """Sanity: the two protected filenames must not accidentally alias."""
    assert trainer._V2_TEACHER_OUTPUT != trainer._V3_TEACHER_OUTPUT
