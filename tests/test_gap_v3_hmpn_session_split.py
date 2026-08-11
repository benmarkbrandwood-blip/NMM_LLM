"""tests/test_gap_v3_hmpn_session_split.py — HMPN extractor --session-ledger flag.

Batch 3b (docs/gap_net_v3_stage_e_rebuild_checklist.md):
- Strict single-tier rule: state_key mask == 0b001 → train, 0b010 → val,
  0b100 → test.  Any other mask (mixed-tier or uncovered) is dropped from
  all splits so no train sample corresponds to a state_key also seen by a
  val/test session.
- _load_state_key_masks verifies the referenced metadata.npz SHA-256.
- _guard_output_dir refuses to overwrite the v2 dataset when --session-ledger
  is provided.
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


def _load_extractor():
    spec = importlib.util.spec_from_file_location(
        "extract_human_move_policy_dataset",
        _ROOT / "tools" / "extract_human_move_policy_dataset.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def extractor():
    return _load_extractor()


# ── _apply_session_ledger_split tests ────────────────────────────────────────

def test_apply_split_strict_single_tier(extractor):
    masks = {
        "sk_train":     0b001,
        "sk_val":       0b010,
        "sk_test":      0b100,
        "sk_mixed_tv":  0b011,   # train + val — drop
        "sk_mixed_tt":  0b101,   # train + test — drop
        "sk_mixed_vt":  0b110,   # val + test — drop
        "sk_mixed_all": 0b111,   # all three — drop
        "sk_uncovered": 0b000,   # no tier reached it — drop
    }
    kept, disp = extractor._apply_session_ledger_split(masks)
    assert kept == {"sk_train": "train", "sk_val": "val", "sk_test": "test"}
    assert disp == {
        "strict_train": 1, "strict_val": 1, "strict_test": 1,
        "mixed_tier": 4, "uncovered": 1,
    }


def test_apply_split_no_train_leakage(extractor):
    """A state_key touched by ANY val or test session must never end up in train."""
    masks = {
        "leak_1": 0b011,  # train + val
        "leak_2": 0b101,  # train + test
        "leak_3": 0b111,  # all
        "clean":  0b001,  # train only
    }
    kept, _ = extractor._apply_session_ledger_split(masks)
    for sk, split in kept.items():
        # Verify: if split == "train", the mask must be exactly 0b001
        if split == "train":
            assert masks[sk] == 0b001, f"leakage: {sk} in train with mask {bin(masks[sk])}"
    assert "clean" in kept and kept["clean"] == "train"
    assert "leak_1" not in kept
    assert "leak_2" not in kept
    assert "leak_3" not in kept


def test_apply_split_disposition_sums_to_input(extractor):
    masks = {f"sk_{i}": (i % 8) for i in range(100)}
    kept, disp = extractor._apply_session_ledger_split(masks)
    assert sum(disp.values()) == len(masks)
    assert (disp["strict_train"] + disp["strict_val"]
            + disp["strict_test"]) == len(kept)


# ── _load_state_key_masks tests ──────────────────────────────────────────────

def _write_synthetic_index(
    tmp_path: Path,
    state_keys: list[str],
    masks: list[int],
    referenced_meta_sha: str | None = None,
) -> tuple[Path, Path]:
    """Write a synthetic metadata.npz + session_index.npz pair.

    If referenced_meta_sha is None, uses the actual sha (agreement case).
    Otherwise writes the given sha into the index provenance (mismatch case).
    """
    meta_dir = tmp_path / "hmpn_ds"
    meta_dir.mkdir(exist_ok=True)
    meta_path = meta_dir / "metadata.npz"
    np.savez(
        str(meta_path),
        state_keys=np.array(state_keys, dtype=object),
        provenance=np.array("{}", dtype=object),
    )
    actual_sha = hashlib.sha256(meta_path.read_bytes()).hexdigest()

    idx_path = tmp_path / "session_index.npz"
    prov = {
        "dataset_dir":         str(meta_dir),
        "dataset_meta_sha256": referenced_meta_sha if referenced_meta_sha else actual_sha,
    }
    np.savez(
        str(idx_path),
        game_split_mask=np.array(masks, dtype=np.uint8),
        player_split_mask=np.zeros(len(masks), dtype=np.uint8),
        provenance=np.array(json.dumps(prov), dtype=object),
    )
    return idx_path, meta_path


def test_load_masks_returns_state_key_to_mask(extractor, tmp_path):
    idx_path, _ = _write_synthetic_index(
        tmp_path,
        state_keys=["sk1", "sk2", "sk3"],
        masks=[0b001, 0b010, 0b100],
    )
    lookup, prov = extractor._load_state_key_masks(idx_path)
    assert lookup == {"sk1": 0b001, "sk2": 0b010, "sk3": 0b100}
    assert "dataset_meta_sha256" in prov


def test_load_masks_rejects_sha_mismatch(extractor, tmp_path):
    idx_path, _ = _write_synthetic_index(
        tmp_path,
        state_keys=["sk1", "sk2"],
        masks=[0b001, 0b010],
        referenced_meta_sha="a" * 64,   # deliberately wrong
    )
    with pytest.raises(ValueError, match="sha"):
        extractor._load_state_key_masks(idx_path)


def test_load_masks_length_mismatch_rejected(extractor, tmp_path):
    """If mask length ≠ state_keys length, load must fail."""
    meta_dir = tmp_path / "hmpn_ds"
    meta_dir.mkdir()
    meta_path = meta_dir / "metadata.npz"
    np.savez(
        str(meta_path),
        state_keys=np.array(["a", "b", "c"], dtype=object),
        provenance=np.array("{}", dtype=object),
    )
    idx_path = tmp_path / "session_index.npz"
    np.savez(
        str(idx_path),
        game_split_mask=np.array([1, 2], dtype=np.uint8),   # length 2, not 3
        player_split_mask=np.zeros(2, dtype=np.uint8),
        provenance=np.array(json.dumps({
            "dataset_dir": str(meta_dir),
            "dataset_meta_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
        }), dtype=object),
    )
    with pytest.raises(ValueError, match="length"):
        extractor._load_state_key_masks(idx_path)


def test_load_masks_missing_file(extractor, tmp_path):
    with pytest.raises(FileNotFoundError):
        extractor._load_state_key_masks(tmp_path / "does_not_exist.npz")


# ── _guard_output_dir tests ──────────────────────────────────────────────────

def test_guard_blocks_default_output_dir_with_ledger(extractor, tmp_path):
    """Refuse overwriting v2 dataset when --session-ledger is set."""
    v2_absolute = extractor._ROOT / extractor._V2_OUTPUT_DIR
    with pytest.raises(SystemExit, match="Refusing"):
        extractor._guard_output_dir(v2_absolute, session_ledger_path=tmp_path / "any.json")


def test_guard_allows_default_output_dir_without_ledger(extractor):
    """No ledger → guard is a no-op even on the v2 default path."""
    v2_absolute = extractor._ROOT / extractor._V2_OUTPUT_DIR
    extractor._guard_output_dir(v2_absolute, session_ledger_path=None)  # must not raise


def test_guard_allows_custom_output_dir_with_ledger(extractor, tmp_path):
    """Custom output-dir + ledger is the intended v3 usage — must not raise."""
    extractor._guard_output_dir(
        tmp_path / "custom_v3_out",
        session_ledger_path=tmp_path / "any.json",
    )
