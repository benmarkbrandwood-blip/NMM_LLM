"""tests/test_gap_v3_hmpn_trainer_ledger_guard.py — HMPN trainer safety guard.

Batch 3b (docs/gap_net_v3_stage_e_rebuild_checklist.md):
When the dataset was extracted with the session-ledger scheme, the trainer must
refuse to overwrite the v2 teacher artefact at
data/human_move_policy_net_v2_candidate.npz — the v3 retrain lands under a
separately-named file so v2 stays intact as exploratory comparison.
"""
from __future__ import annotations

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


# ── _guard_output_path — session-ledger dataset ──────────────────────────────

def test_guard_blocks_default_v2_output_with_session_ledger(trainer):
    """Session-ledger dataset + v2 default output → refuse."""
    v2_absolute = trainer._ROOT / trainer._V2_TEACHER_OUTPUT
    prov = {"split_scheme": trainer._SESSION_LEDGER_SPLIT_SCHEME}
    with pytest.raises(SystemExit, match="Refusing"):
        trainer._guard_output_path(v2_absolute, prov)


def test_guard_allows_v3_output_with_session_ledger(trainer):
    """Session-ledger dataset + v3 output → OK."""
    v3_absolute = trainer._ROOT / trainer._V3_TEACHER_OUTPUT
    prov = {"split_scheme": trainer._SESSION_LEDGER_SPLIT_SCHEME}
    trainer._guard_output_path(v3_absolute, prov)   # must not raise


def test_guard_allows_custom_output_with_session_ledger(trainer, tmp_path):
    """Session-ledger dataset + arbitrary custom path → OK."""
    prov = {"split_scheme": trainer._SESSION_LEDGER_SPLIT_SCHEME}
    trainer._guard_output_path(tmp_path / "custom.npz", prov)   # must not raise


# ── _guard_output_path — non-session-ledger dataset ──────────────────────────

def test_guard_allows_default_v2_output_without_session_ledger(trainer):
    """State-key three-way dataset + v2 default output → OK (default v2 behaviour)."""
    v2_absolute = trainer._ROOT / trainer._V2_TEACHER_OUTPUT
    prov = {"split_scheme": "state_key_three_way"}
    trainer._guard_output_path(v2_absolute, prov)   # must not raise


def test_guard_allows_v2_output_when_scheme_absent(trainer):
    """No split_scheme in provenance (e.g. v1 dataset) → guard is a no-op."""
    v2_absolute = trainer._ROOT / trainer._V2_TEACHER_OUTPUT
    trainer._guard_output_path(v2_absolute, {})   # must not raise


def test_v2_and_v3_output_paths_differ(trainer):
    """Sanity: the two protected filenames must not accidentally alias."""
    assert trainer._V2_TEACHER_OUTPUT != trainer._V3_TEACHER_OUTPUT
