#!/usr/bin/env python3
"""tools/train_gap_net_v3.py — Train GapNet v3 3-head MLP (§11, gap_net_v3_plan.md).

Locked decisions (docs/gap_net_v3_stage_e_rebuild_checklist.md, 2026-08-06):
- 82 features = 79 board + 3-way Elo-band one-hot.
- 82 → 128 → 64 → 32 → 3 (Components A, B, C only; D dropped for regret_v1).
- D4 augmentation: opt-in via --d4-augmentation on; default off.
- Baseline reference on high-support rows: empirical G_v is the reference.
- Mean-predictor + uniform-P_h + teacher-fidelity all reported separately.
- Fail-closed load: non-NaN targets must be finite.

Requires the dataset produced by the rewritten Stage D extractor:
  data/gap_net_v3_dataset_v2/{parent_feats,targets,targets_uniform,targets_empirical}.f32.bin
  data/gap_net_v3_dataset_v2/metadata.npz (state_keys, band_idx, split, phase, mover_color,
                                           n_legal, ph_source, session_min_hash, provenance)

Split encoding in metadata.npz: 0=train, 1=val, 2=test (test never touched here).

Usage:
    .venv/bin/python tools/train_gap_net_v3.py --dataset-dir data/gap_net_v3_dataset_v2
    .venv/bin/python tools/train_gap_net_v3.py --d4-augmentation on --epochs 100
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ai.board_symmetry import _BOARD_PERM as _BD_PERM, SYM_INVERSE as _SYM_INV

# ── Constants ────────────────────────────────────────────────────────────────

_BOARD_DIM  = 79
_N_BANDS    = 3
_INPUT_DIM  = _BOARD_DIM + _N_BANDS      # 82
_H1, _H2, _H3, _N_HEADS = 128, 64, 32, 3
_BAND_NAMES = ("lower", "middle", "upper")
_COMP_NAMES = ("class_downgrade", "wdl_utility_loss", "ordinal_rank_loss")


# ── D4 board-block permutations (§11.2 ablation, opt-in) ─────────────────────

def _build_board_perms() -> list[np.ndarray]:
    """Permutation arrays for the 79-dim board block only.

    Band one-hot at indices 79–81 is invariant under board symmetry.
    perm[j] is the source index that provides the value for output index j
    within the 79-dim board block.
    """
    perms = []
    for sym_idx in range(8):
        perm = np.arange(_BOARD_DIM, dtype=np.int64)
        inv_bp = _BD_PERM[_SYM_INV[sym_idx]]
        if inv_bp is not None:
            for new_pos, old_pos in enumerate(inv_bp):
                for c in range(3):
                    perm[new_pos * 3 + c] = old_pos * 3 + c
        perms.append(perm)
    return perms

_BOARD_PERMS: list[np.ndarray] = _build_board_perms()


# ── Model ────────────────────────────────────────────────────────────────────

class GapNetV3(nn.Module):
    """82 → 128 → 64 → 32 → 3 MLP for regret_v1 (Components A, B, C)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_INPUT_DIM, _H1), nn.ReLU(),
            nn.Linear(_H1, _H2),        nn.ReLU(),
            nn.Linear(_H2, _H3),        nn.ReLU(),
            nn.Linear(_H3, _N_HEADS),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── NaN-masked MSE loss ──────────────────────────────────────────────────────

def _nan_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over non-NaN (pred, target) pairs; returns 0.0-connected scalar if all NaN."""
    mask = ~torch.isnan(target)
    if not mask.any():
        return pred.sum() * 0.0
    return ((pred[mask] - target[mask]) ** 2).mean()


# ── Feature construction ─────────────────────────────────────────────────────

def _build_features(board: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    """Concat 79-dim board features with 3-dim band one-hot → 82-dim input."""
    band_onehot = torch.zeros(
        band.shape[0], _N_BANDS, dtype=board.dtype, device=board.device,
    )
    band_onehot.scatter_(1, band.long().unsqueeze(1), 1.0)
    return torch.cat([board, band_onehot], dim=1)


# ── Dataset loading (fail-closed) ────────────────────────────────────────────

_DATASET_PROV_MALFORMED = "malformed_dataset_provenance"


def _validate_dataset_provenance_contract(prov: dict) -> None:
    """Codex 2026-08-15 P1: strict contract validation for a dataset's own
    provenance dict (embedded in metadata.npz under key `provenance`).

    Prior implementation parsed the provenance by truthiness — JSON string
    `"false"`, integer `1`, or a `production_ready=true` with a non-empty
    `non_ready_reasons` all silently coerced to a ready-and-clean state.
    Codex flagged this as a promotion-safety hole.

    Contract (fail-closed):
      - `production_ready` must be an exact Python bool (True or False).
        Missing, string, int, None → malformed → SystemExit.
      - `non_ready_reasons` must be a list.  Missing or wrong type → malformed.
      - `gate_status` must be a non-empty string.  Missing / wrong type → malformed.
      - Invariant A: `production_ready is True` implies `non_ready_reasons == []`.
      - Invariant B: `production_ready is True` implies `gate_status == "ok"`.
      - Invariant C: `production_ready is False` implies non-empty non_ready_reasons.
      - Invariant D: `gate_status != "ok"` implies `production_ready is False`.

    Any violation raises SystemExit — this cannot be bypassed by
    --allow-non-production-dataset, which is for legitimate non-ready
    datasets, not for silently-broken provenance.
    """
    def _bail(msg: str) -> None:
        raise SystemExit(
            f"[gap_net_v3] Dataset provenance is malformed / self-inconsistent: "
            f"{msg}.  Refusing to consume.  Re-run the extractor to rebuild "
            f"provenance."
        )

    if "production_ready" not in prov:
        _bail("missing `production_ready` field")
    pr = prov["production_ready"]
    if pr is not True and pr is not False:
        _bail(
            f"`production_ready` must be exact JSON boolean; got {pr!r} "
            f"of type {type(pr).__name__}"
        )
    if "non_ready_reasons" not in prov:
        _bail("missing `non_ready_reasons` field")
    reasons = prov["non_ready_reasons"]
    if not isinstance(reasons, list):
        _bail(
            f"`non_ready_reasons` must be a list; got {type(reasons).__name__}"
        )
    if "gate_status" not in prov:
        _bail("missing `gate_status` field")
    gate_status = prov["gate_status"]
    if not isinstance(gate_status, str) or not gate_status:
        _bail(f"`gate_status` must be a non-empty string; got {gate_status!r}")

    if pr is True and reasons:
        _bail(
            f"invariant A violated: production_ready=True but "
            f"non_ready_reasons={reasons!r}"
        )
    if pr is True and gate_status != "ok":
        _bail(
            f"invariant B violated: production_ready=True but "
            f"gate_status={gate_status!r} (expected 'ok')"
        )
    if pr is False and not reasons:
        _bail(
            "invariant C violated: production_ready=False but "
            "non_ready_reasons is empty"
        )
    if gate_status != "ok" and pr is True:
        _bail(
            f"invariant D violated: gate_status={gate_status!r} but "
            f"production_ready=True"
        )


def _verify_dataset_production_ready(
    dataset_dir: Path, allow_non_production: bool = False,
) -> dict:
    """Codex P1(4) 2026-08-14: Stage E must hard-reject datasets that came
    from a smoke, sub-floor, or hardening-bypassed extraction.  The extractor
    writes production_ready + non_ready_reasons + gate_status to metadata.npz
    provenance; we refuse to train against a non-ready dataset unless the
    user explicitly passes --allow-non-production-dataset.

    Codex 2026-08-15 (2nd pass) P1: dataset provenance is now validated as a
    strict contract (see `_validate_dataset_provenance_contract`).
    Malformed / self-inconsistent provenance stops the run and CANNOT be
    bypassed by --allow-non-production-dataset (which is for legitimate
    non-ready datasets, not for broken metadata).

    Returned dict (always with same keys):
        dataset_production_ready       bool  — parsed strict bool
        dataset_non_ready_reasons      list[str] — [] when ready
        non_production_override        bool  — True if user bypassed refusal
        dataset_provenance             dict  — raw extractor provenance
                                              ({} only when missing under
                                              --allow-non-production-dataset)

    NOTE: promotion_eligible is intentionally NOT in this dict — it is a
    downstream verdict computed by `_build_saved_provenance` from the
    combination of dataset readiness AND Stage E gate result AND frozen
    thresholds identifier.  This function only reports dataset state.
    """
    meta_path = dataset_dir / "metadata.npz"
    if not meta_path.exists():
        raise SystemExit(
            f"[gap_net_v3] metadata.npz not found in {dataset_dir}"
        )
    meta = np.load(meta_path, allow_pickle=True)
    if "provenance" not in meta.files:
        if allow_non_production:
            print("[gap_net_v3] WARNING: dataset has no provenance; "
                  "proceeding under --allow-non-production-dataset")
            return {
                "dataset_production_ready":  False,
                "dataset_non_ready_reasons": ["no_dataset_provenance"],
                "non_production_override":   True,
                "dataset_provenance":        {},
            }
        raise SystemExit(
            f"[gap_net_v3] Dataset at {dataset_dir} has no provenance; "
            f"cannot verify production_ready.  Pass --allow-non-production-dataset "
            f"to override (NOT for promotion evidence)."
        )
    prov_raw = meta["provenance"].item()
    if isinstance(prov_raw, bytes):
        prov_raw = prov_raw.decode("utf-8")
    try:
        prov = json.loads(prov_raw)
    except Exception as e:
        raise SystemExit(
            f"[gap_net_v3] Dataset provenance is not valid JSON: {e!s}"
        )
    if not isinstance(prov, dict):
        raise SystemExit(
            f"[gap_net_v3] Dataset provenance is not a JSON object; "
            f"got {type(prov).__name__}"
        )

    # Contract validation is unconditional — this cannot be overridden.
    _validate_dataset_provenance_contract(prov)

    production_ready = prov["production_ready"]   # already validated as bool
    reasons = list(prov["non_ready_reasons"])     # already validated as list

    if production_ready is False:
        if allow_non_production:
            print("[gap_net_v3] WARNING: training on non-production dataset")
            print(f"[gap_net_v3]   non_ready_reasons: {reasons}")
            print("[gap_net_v3]   this run is NOT eligible for promotion evidence.")
            return {
                "dataset_production_ready":  False,
                "dataset_non_ready_reasons": reasons,
                "non_production_override":   True,
                "dataset_provenance":        prov,
            }
        raise SystemExit(
            f"[gap_net_v3] Dataset at {dataset_dir} is NOT production_ready:\n"
            f"  reasons: {reasons}\n"
            f"Pass --allow-non-production-dataset to train against it anyway "
            f"(NOT eligible for promotion evidence)."
        )
    return {
        "dataset_production_ready":  True,
        "dataset_non_ready_reasons": [],
        "non_production_override":   False,
        "dataset_provenance":        prov,
    }


def _load_split(dataset_dir: Path, split_val: int) -> dict:
    """Load one split.  Returns dict of tensors + numpy arrays.

    Fail-closed: raises ValueError if any non-NaN target value is non-finite.
    """
    meta = np.load(str(dataset_dir / "metadata.npz"), allow_pickle=True)
    idx = np.where(meta["split"] == split_val)[0]
    n_total = len(meta["split"])

    def _mm(name: str, dim: int) -> np.ndarray:
        return np.memmap(
            str(dataset_dir / name),
            dtype="float32", mode="r", shape=(n_total, dim),
        )

    board_arr = _mm("parent_feats.f32.bin", _BOARD_DIM)[idx].copy()
    tgt_arr   = _mm("targets.f32.bin", _N_HEADS)[idx].copy()
    unif_arr  = _mm("targets_uniform.f32.bin", _N_HEADS)[idx].copy()
    emp_arr   = _mm("targets_empirical.f32.bin", _N_HEADS)[idx].copy()
    band_arr  = meta["band_idx"][idx].astype(np.int64)

    # Fail-closed A/B/C target discipline
    # (docs/gap_net_v3_stage_e_rebuild_checklist.md, Stage D redo):
    # model + uniform targets must be FULLY finite — no NaN, no ±inf.
    # Unavailable R_v abstains the row at extract time; it never emits NaN.
    # Only targets_empirical may contain NaN, as the 'support < min_support'
    # sentinel — its non-NaN entries must still be finite.
    for name, arr in (("targets", tgt_arr), ("targets_uniform", unif_arr)):
        if not np.isfinite(arr).all():
            raise ValueError(
                f"[gap_net_v3] non-finite value(s) (NaN or ±inf) found in "
                f"{name} (split={split_val}) — A/B/C targets must be fully "
                f"finite; unavailable R_v must abstain the row, never emit NaN"
            )
    emp_non_nan = ~np.isnan(emp_arr)
    if emp_non_nan.any() and not np.isfinite(emp_arr[emp_non_nan]).all():
        raise ValueError(
            f"[gap_net_v3] ±inf value(s) found in targets_empirical "
            f"(split={split_val}) — NaN is the only permitted non-finite "
            f"sentinel"
        )

    return {
        "board":   torch.from_numpy(board_arr),
        "band":    torch.from_numpy(band_arr),
        "y_model": torch.from_numpy(tgt_arr),
        "y_unif":  torch.from_numpy(unif_arr),
        "y_emp":   torch.from_numpy(emp_arr),
    }


# ── Chunked val forward ──────────────────────────────────────────────────────

def _forward_chunked(model: nn.Module, X: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
    parts = []
    for i in range(0, len(X), chunk):
        parts.append(model(X[i:i + chunk]))
    return torch.cat(parts, dim=0)


# ── Provenance helpers ───────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Stage E threshold registry (Codex 2026-08-15 3rd pass P1) ────────────────

_STAGE_E_THRESHOLDS_REGISTRY = (
    _ROOT / "configs" / "stage_e_thresholds.json"
)


def _validate_threshold_ranges(entry_or_args: dict) -> None:
    """Codex 2026-08-15 (3rd pass) P1: hard range checks on the four gate
    threshold values.  A deterministic probe at the previous commit showed
    a 2x-worse candidate PASSing Gate 1 with x_a=-2.0 (threshold 3.0).
    Applied to both frozen-registry entries AND runtime CLI values.

    Ranges:
      x_a in [0.0, 1.0]  — fractional improvement over uniform required.
      x_b in [0.0, 1.0]  — fractional MSE degradation vs teacher tolerated.
      min_high_support > 0 integer — cells with fewer high-support rows SKIP.
      min_denominator > 0 finite float — smaller denominators SKIP.
    """
    def _err(msg: str) -> None:
        raise SystemExit(f"[gap_net_v3] Stage E threshold out of range: {msg}")

    xa = entry_or_args.get("x_a") if isinstance(entry_or_args, dict) else getattr(entry_or_args, "stage_e_x_a", None)
    xb = entry_or_args.get("x_b") if isinstance(entry_or_args, dict) else getattr(entry_or_args, "stage_e_x_b", None)
    mhs = (entry_or_args.get("min_high_support") if isinstance(entry_or_args, dict)
           else getattr(entry_or_args, "stage_e_min_high_support", None))
    mden = (entry_or_args.get("min_denominator") if isinstance(entry_or_args, dict)
            else getattr(entry_or_args, "stage_e_min_denominator", None))

    for name, val in (("x_a", xa), ("x_b", xb), ("min_denominator", mden)):
        if val is None:
            _err(f"{name} missing")
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            _err(f"{name}={val!r} must be a real number")
        f = float(val)
        if not (f == f) or f in (float("inf"), float("-inf")):   # NaN / inf
            _err(f"{name}={val!r} is not finite")
    if not (0.0 <= float(xa) <= 1.0):
        _err(f"x_a={xa!r} outside allowed range [0.0, 1.0]")
    if not (0.0 <= float(xb) <= 1.0):
        _err(f"x_b={xb!r} outside allowed range [0.0, 1.0]")
    if not isinstance(mhs, int) or isinstance(mhs, bool) or mhs <= 0:
        _err(f"min_high_support={mhs!r} must be positive integer")
    if float(mden) <= 0.0:
        _err(f"min_denominator={mden!r} must be > 0")


def _load_frozen_thresholds(
    threshold_id: str,
    registry_path: Path = _STAGE_E_THRESHOLDS_REGISTRY,
) -> tuple[dict, str]:
    """Load a specific entry from the tracked threshold registry.

    Codex 2026-08-15 (3rd pass) P1: previously `--gate-thresholds-frozen-id`
    was an unverified self-assertion.  Now it must resolve to a registry
    entry with `frozen=true`; the entry's values are what the trainer must
    match against its CLI args.

    Returns (entry_dict, canonical_sha256).  Raises SystemExit on:
      - registry file missing / not JSON / wrong shape
      - id unknown
      - entry not frozen (draft)
      - entry values out of allowed ranges
    """
    if not registry_path.exists():
        raise SystemExit(
            f"[gap_net_v3] Stage E threshold registry not found: {registry_path}"
        )
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(
            f"[gap_net_v3] Stage E threshold registry unreadable: {e!s}"
        )
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise SystemExit(
            "[gap_net_v3] Stage E threshold registry malformed: "
            "missing 'entries' list"
        )
    matches = [e for e in entries if isinstance(e, dict)
               and e.get("id") == threshold_id]
    if not matches:
        known = sorted(e.get("id", "?") for e in entries if isinstance(e, dict))
        raise SystemExit(
            f"[gap_net_v3] Stage E threshold id {threshold_id!r} is not in "
            f"the registry.  Known ids: {known}"
        )
    if len(matches) > 1:
        raise SystemExit(
            f"[gap_net_v3] Stage E threshold id {threshold_id!r} appears "
            f"more than once in the registry"
        )
    entry = matches[0]
    if entry.get("frozen") is not True:
        raise SystemExit(
            f"[gap_net_v3] Stage E threshold id {threshold_id!r} is not "
            f"frozen (draft).  Only frozen entries grant Stage F eligibility."
        )

    _validate_threshold_ranges(entry)

    canonical_bytes = json.dumps(
        {k: entry[k] for k in sorted(entry)}, sort_keys=True,
    ).encode("utf-8")
    canonical_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    return entry, canonical_sha256


def _cli_matches_frozen_entry(args, entry: dict) -> Optional[str]:
    """Return None if CLI --stage-e-x-a/x-b/min-high-support/min-denominator
    match the frozen entry EXACTLY, else a mismatch description string.

    Exact equality is deliberate: a run using different values than the
    frozen set is not the run authority granted."""
    mismatches: list[str] = []
    checks = (
        ("x_a",              args.stage_e_x_a,             entry.get("x_a")),
        ("x_b",              args.stage_e_x_b,             entry.get("x_b")),
        ("min_high_support", args.stage_e_min_high_support, entry.get("min_high_support")),
        ("min_denominator",  args.stage_e_min_denominator, entry.get("min_denominator")),
    )
    for name, cli_val, entry_val in checks:
        if cli_val != entry_val:
            mismatches.append(f"{name}: CLI={cli_val!r} vs frozen={entry_val!r}")
    if mismatches:
        return "; ".join(mismatches)
    return None


# ── Saved-provenance builder (Codex 2026-08-15 2nd pass P1 + P2, 3rd pass P1) ─

def _build_saved_provenance(
    *,
    dataset_taint: dict,
    dataset_dir: Path,
    gate_results: list,
    gate_summary: dict,
    frozen_thresholds_entry: Optional[dict],
    frozen_thresholds_sha256: Optional[str],
    args,
    n_train: int,
    n_val: int,
    best_val_loss: float,
    epochs_trained: int,
    tr_means: list,
) -> tuple[dict, str]:
    """Single seam that constructs the provenance dict AND the model label.

    Codex 2026-08-15 (3rd pass) P1 + P2: prior versions labelled a Stage E
    PASS as promotion-eligible.  Under docs/gap_net_v3_plan.md §13/§16,
    Stage E only creates the CANDIDATE.  Full promotion requires an
    identity-bound Stage F artifact + every remaining plan gate + a
    separate live-promotion decision — none of which this trainer runs.

    New rule (fail-closed):
        stage_e_passed   = gate_summary["overall_verdict"] == "PASS"
        stage_f_eligible = dataset_eligibility
                           AND stage_e_passed
                           AND frozen_thresholds_entry present
                               (owner-reviewed registry hit with frozen=true)
        promotion_eligible = False  — always.  Only Stage F artifact +
                                       subsequent gates + live-promotion
                                       tooling can flip this.

    where:
        dataset_eligibility = dataset_production_ready AND NOT non_production_override

    Model label:
        stage_f_eligible True  → "gap_net_v3_stage_e_candidate"
                                 (candidate is eligible to enter Stage F)
        stage_f_eligible False → "gap_net_v3_stage_e_candidate_ineligible"
        In no path does this trainer emit the historical plain
        "gap_net_v3_candidate" label — that name is reserved for the
        Stage F artifact.

    `stage_f_ineligible_reasons` (list[str]) enumerates why the run is not
    Stage F eligible — always empty iff stage_f_eligible is True.
    `promotion_ineligible_reasons` always includes at least
    `"stage_f_artifact_not_produced_by_this_tool"` since only Stage F
    tooling can flip promotion_eligible.

    Codex 2026-08-15 P2: this is the SINGLE construction point for the
    saved provenance and model label.  main() is a thin wrapper and must
    not build provenance elsewhere.  Any regression that drops a field is
    caught by unit tests + a bounded integration test through main().
    """
    # Dataset side
    dataset_ready = dataset_taint["dataset_production_ready"] is True
    override      = dataset_taint["non_production_override"] is True
    dataset_eligibility = dataset_ready and not override

    # Stage E gate side
    gate_verdict = gate_summary.get("overall_verdict")
    if not isinstance(gate_verdict, str):
        gate_verdict = "MISSING"
    stage_e_passed = gate_verdict == "PASS"

    # Thresholds-frozen side — must be a real registry hit, not a self-assertion.
    gate_thresholds_frozen = frozen_thresholds_entry is not None
    frozen_id = (frozen_thresholds_entry or {}).get("id")

    # Stage F eligibility (this trainer can compute this):
    stage_f_ineligible_reasons: list[str] = []
    if not dataset_ready:
        stage_f_ineligible_reasons.append(
            f"dataset_not_production_ready:{dataset_taint['dataset_non_ready_reasons']!r}"
        )
    if override:
        stage_f_ineligible_reasons.append("non_production_override_set")
    if not stage_e_passed:
        stage_f_ineligible_reasons.append(
            f"stage_e_gate_verdict={gate_verdict!r} (must be 'PASS')"
        )
    if not gate_thresholds_frozen:
        stage_f_ineligible_reasons.append(
            "no_frozen_stage_e_thresholds "
            "(--gate-thresholds-frozen-id must resolve to a registry entry "
            "with frozen=true whose values equal the run's CLI values; "
            "see configs/stage_e_thresholds.json and checklist §Batch 5)"
        )
    stage_f_eligible = not stage_f_ineligible_reasons

    # Promotion eligibility (always False in this trainer — Codex 3rd pass P1)
    promotion_eligible = False
    promotion_ineligible_reasons = list(stage_f_ineligible_reasons) + [
        "stage_f_artifact_not_produced_by_this_tool "
        "(Stage E trainer creates a candidate only; promotion requires "
        "the Stage F single-use artifact + all remaining plan gates + "
        "live-promotion authority — see docs/gap_net_v3_plan.md §13/§16)"
    ]

    model_label = ("gap_net_v3_stage_e_candidate"
                   if stage_f_eligible
                   else "gap_net_v3_stage_e_candidate_ineligible")

    provenance = {
        "model":                     model_label,
        "architecture":              f"{_INPUT_DIM}→{_H1}→{_H2}→{_H3}→{_N_HEADS}",
        "input_layout":              "board[79] || band_onehot[3]",
        "components":                list(_COMP_NAMES),
        "bands":                     list(_BAND_NAMES),
        "dataset_dir":               str(dataset_dir),
        "dataset_feats_sha256":      _sha256(dataset_dir / "parent_feats.f32.bin"),
        "dataset_targets_sha256":    _sha256(dataset_dir / "targets.f32.bin"),
        "dataset_uniform_sha256":    _sha256(dataset_dir / "targets_uniform.f32.bin"),
        "dataset_emp_sha256":        _sha256(dataset_dir / "targets_empirical.f32.bin"),
        "dataset_metadata_sha256":   _sha256(dataset_dir / "metadata.npz"),
        # Codex 2026-08-15 P1 (2nd pass): dataset taint + Stage E gate.
        "dataset_production_ready":  dataset_taint["dataset_production_ready"],
        "dataset_non_ready_reasons": dataset_taint["dataset_non_ready_reasons"],
        "non_production_override":   dataset_taint["non_production_override"],
        "dataset_eligibility":       dataset_eligibility,
        "stage_e_gate_verdict":      gate_verdict,
        "stage_e_passed":            stage_e_passed,
        # Codex 2026-08-15 P1 (3rd pass): identity-bound frozen thresholds.
        "gate_thresholds_frozen":    gate_thresholds_frozen,
        "gate_thresholds_frozen_id": frozen_id,
        "gate_thresholds_frozen_sha256": frozen_thresholds_sha256,
        "gate_thresholds_frozen_entry":  frozen_thresholds_entry,
        # Codex 2026-08-15 P1 (3rd pass): Stage F eligibility + separate
        # promotion (always False from this tool).
        "stage_f_eligible":              stage_f_eligible,
        "stage_f_ineligible_reasons":    stage_f_ineligible_reasons,
        "promotion_eligible":            promotion_eligible,
        "promotion_ineligible_reasons":  promotion_ineligible_reasons,
        "n_train":                   n_train,
        "n_val":                     n_val,
        "best_val_loss":             best_val_loss,
        "epochs_trained":            epochs_trained,
        "lr":                        args.lr,
        "batch_size":                args.batch_size,
        "weight_decay":              args.weight_decay,
        "seed":                      args.seed,
        "d4_augmentation":           args.d4_augmentation,
        "tr_means":                  tr_means,
        "gate_results":              gate_results,
        "gate_summary":              gate_summary,
        "git_commit":                _git_commit(),
        "built_at":                  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return provenance, model_label


# ── Save ─────────────────────────────────────────────────────────────────────

def _save(model: GapNetV3, path: Path, provenance: dict) -> None:
    weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(
        str(path),
        **weights,
        provenance=np.array(json.dumps(provenance), dtype=object),
        architecture=np.array(
            json.dumps({
                "input":      _INPUT_DIM,
                "board_dim":  _BOARD_DIM,
                "n_bands":    _N_BANDS,
                "hidden":     [_H1, _H2, _H3],
                "heads":      _N_HEADS,
                "components": list(_COMP_NAMES),
                "bands":      list(_BAND_NAMES),
            }),
            dtype=object,
        ),
    )


# ── Gate reporting ───────────────────────────────────────────────────────────

def _mse_where(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.mean((pred[mask] - ref[mask]) ** 2))


def _evaluate_cell_verdict(
    candidate_mse: float,
    reference_mse: float,
    n_high_support: int,
    x: float,
    direction: str,             # "min_improvement" (gate 1) | "max_tolerance" (gate 2)
    min_n_high_support: int,
    min_denominator: float,
) -> tuple[str, dict]:
    """Return (verdict, detail) for one Gate × (band × component) cell.

    Verdicts (Codex P2-A 2026-08-12, hardened by P1 follow-up 2026-08-13):
      - "PASS"                              — gate satisfied.
      - "FAIL"                              — finite metrics, gate violated.
      - "FAIL_DIVERGED"                     — candidate or reference MSE is ±inf
                                              (training divergence).
      - "FAIL_UNEXPECTED_NAN"               — cell has adequate support
                                              (n_high_support ≥ min) but a
                                              gate MSE is NaN — a training bug
                                              or corrupt data, treated as
                                              failure (was SKIP_NON_FINITE).
      - "SKIP_INSUFFICIENT_SUPPORT"         — n_high_support < min_n_high_support.
      - "SKIP_DEGENERATE_DENOMINATOR"       — |reference_mse| < min_denominator.

    SKIP_NON_FINITE is no longer returned by this helper — supported cells
    with NaN MSE are FAIL_UNEXPECTED_NAN.  The name is retained as a defensive
    combine-helper fallback only (see _combine_gate_verdicts).
    """
    if n_high_support < min_n_high_support:
        return "SKIP_INSUFFICIENT_SUPPORT", {
            "reason": f"n_high_support {n_high_support} < min_n_high_support {min_n_high_support}",
        }
    # Codex P1 2026-08-13: supported cells with non-finite MSE fail closed.
    # NaN with adequate high-support means MSE computation broke (training
    # bug or corrupt data), not a data gap; ±inf indicates numerical
    # divergence.  Both are treated as gate failures, not skips.
    if not np.isfinite(candidate_mse) or not np.isfinite(reference_mse):
        if np.isinf(candidate_mse) or np.isinf(reference_mse):
            return "FAIL_DIVERGED", {
                "candidate_mse": candidate_mse, "reference_mse": reference_mse,
            }
        return "FAIL_UNEXPECTED_NAN", {
            "candidate_mse": candidate_mse, "reference_mse": reference_mse,
            "reason": (
                f"NaN MSE despite n_high_support={n_high_support} "
                f">= min_n_high_support={min_n_high_support} (training bug or "
                f"corrupt data)"
            ),
        }
    if abs(reference_mse) < min_denominator:
        return "SKIP_DEGENERATE_DENOMINATOR", {
            "reference_mse": reference_mse, "min_denominator": min_denominator,
        }

    if direction == "min_improvement":
        # Gate 1: candidate ≤ (1 − x) × reference
        threshold = (1.0 - x) * reference_mse
        passed = candidate_mse <= threshold
        margin_pct = (reference_mse - candidate_mse) / reference_mse * 100.0
    elif direction == "max_tolerance":
        # Gate 2: candidate ≤ (1 + x) × reference
        threshold = (1.0 + x) * reference_mse
        passed = candidate_mse <= threshold
        margin_pct = (reference_mse - candidate_mse) / reference_mse * 100.0
    else:
        raise ValueError(f"unknown direction: {direction!r}")

    return ("PASS" if passed else "FAIL"), {
        "threshold_mse":  float(threshold),
        "margin_pct":     float(margin_pct),
        "candidate_mse":  float(candidate_mse),
        "reference_mse":  float(reference_mse),
    }


def _report_gate(
    val_pred:  np.ndarray,  # (N, 3)
    y_model:   np.ndarray,  # (N, 3) model-P_h-derived G_v (training target)
    y_unif:    np.ndarray,  # (N, 3) uniform-P_h-derived G_v
    y_emp:     np.ndarray,  # (N, 3) empirical G_v (NaN where support insufficient)
    band_idx:  np.ndarray,  # (N,)   band 0=lower, 1=middle, 2=upper
    tr_means:  np.ndarray,  # (3,)   training-target mean per component
    x_a:                float = 0.30,
    x_b:                float = 0.20,
    min_n_high_support: int   = 100,
    min_denominator:    float = 1e-9,
) -> tuple[list[dict], dict]:
    """Compute Stage E gate metrics per component per band with executable verdicts.

    Codex P2-A 2026-08-12 + P1 follow-up 2026-08-13: the gate returns
    pass/fail per cell (band × component) alongside the raw MSE values.
    Under the "ALL_CELLS_MUST_PASS" coverage rule, overall PASS requires
    every one of `n_cells_total = _N_HEADS × _N_BANDS` cells to PASS.
    Anything less is FAIL (if any FAIL) or FAIL_INSUFFICIENT_COVERAGE
    (if all non-fail cells are SKIP).  See summary.overall_verdict.

    Reference framing (Decision 3A, 2026-08-06):
      - High-support rows (empirical present): empirical G_v is the reference.
        Report candidate / teacher / uniform MSE against empirical.
      - Model-only rows: candidate vs. teacher target is teacher-fidelity — not
        empirical validation.  Reported separately.
      - Mean-predictor baseline (predict train mean of teacher target) reported
        against teacher target across all valid rows.

    Gates (per band × per component):
      Gate 1 (beat uniform): candidate_vs_emp ≤ (1 − x_a) × uniform_vs_emp
      Gate 2 (track teacher): candidate_vs_emp ≤ (1 + x_b) × teacher_vs_emp

    Returns:
      results:  list[per-component dict with per_band verdict info]
      summary:  overall verdict + thresholds + failing/insufficient cell lists
    """
    results:       list[dict] = []
    failing_cells: list[dict] = []
    skipped_cells: list[dict] = []
    n_pass = n_fail = n_insufficient = n_degenerate = n_non_finite = 0

    for c in range(_N_HEADS):
        comp_result = {"component": _COMP_NAMES[c], "per_band": {}}
        for b in range(_N_BANDS):
            in_band = (band_idx == b)
            valid_model = in_band & ~np.isnan(y_model[:, c])
            valid_emp   = valid_model & ~np.isnan(y_emp[:, c])
            n_valid          = int(valid_model.sum())
            n_high_support   = int(valid_emp.sum())

            row: dict = {
                "n_valid":         n_valid,
                "n_high_support":  n_high_support,
                "candidate_teacher_fidelity_mse":
                    _mse_where(val_pred[:, c], y_model[:, c], valid_model),
                "mean_predictor_vs_teacher_mse":
                    _mse_where(
                        np.full(len(y_model), tr_means[c], dtype=np.float64),
                        y_model[:, c], valid_model,
                    ),
            }

            candidate_vs_emp = _mse_where(val_pred[:, c], y_emp[:, c], valid_emp)
            teacher_vs_emp   = _mse_where(y_model[:, c],  y_emp[:, c], valid_emp)
            uniform_vs_emp   = _mse_where(y_unif[:, c],   y_emp[:, c], valid_emp)
            row.update({
                "candidate_vs_empirical_mse": candidate_vs_emp,
                "teacher_vs_empirical_mse":   teacher_vs_emp,
                "uniform_vs_empirical_mse":   uniform_vs_emp,
            })

            gate1_verdict, gate1_detail = _evaluate_cell_verdict(
                candidate_vs_emp, uniform_vs_emp, n_high_support,
                x=x_a, direction="min_improvement",
                min_n_high_support=min_n_high_support,
                min_denominator=min_denominator,
            )
            gate2_verdict, gate2_detail = _evaluate_cell_verdict(
                candidate_vs_emp, teacher_vs_emp, n_high_support,
                x=x_b, direction="max_tolerance",
                min_n_high_support=min_n_high_support,
                min_denominator=min_denominator,
            )
            row["gate_1_verdict"] = gate1_verdict
            row["gate_1_detail"]  = gate1_detail
            row["gate_2_verdict"] = gate2_verdict
            row["gate_2_detail"]  = gate2_detail

            # Cell verdict: PASS iff both gates PASS; any FAIL / FAIL_DIVERGED /
            # FAIL_UNEXPECTED_NAN → cell FAIL (see _combine_gate_verdicts for
            # precedence).  Only when both gates are SKIP does the cell SKIP;
            # under ALL_CELLS_MUST_PASS a single SKIP blocks overall PASS.
            cell_verdict = _combine_gate_verdicts(gate1_verdict, gate2_verdict)
            row["cell_verdict"] = cell_verdict

            band_name = _BAND_NAMES[b]
            comp_name = _COMP_NAMES[c]
            if cell_verdict == "PASS":
                n_pass += 1
            elif cell_verdict.startswith("FAIL"):
                n_fail += 1
                failing_cells.append({
                    "band": band_name, "component": comp_name,
                    "gate_1": gate1_verdict, "gate_2": gate2_verdict,
                    "candidate_mse": candidate_vs_emp,
                    "teacher_mse":   teacher_vs_emp,
                    "uniform_mse":   uniform_vs_emp,
                    "n_high_support": n_high_support,
                })
            else:
                # SKIP_INSUFFICIENT_SUPPORT / SKIP_DEGENERATE_DENOMINATOR.
                # SKIP_NON_FINITE is a defensive combine-helper fallback and
                # should not be reachable via _evaluate_cell_verdict any more
                # (supported NaN is FAIL_UNEXPECTED_NAN); n_cells_skip_non_finite
                # is retained as a diagnostic counter in case that invariant
                # ever regresses.
                if "INSUFFICIENT" in cell_verdict:
                    n_insufficient += 1
                elif "DEGENERATE" in cell_verdict:
                    n_degenerate += 1
                elif "NON_FINITE" in cell_verdict:
                    n_non_finite += 1
                skipped_cells.append({
                    "band": band_name, "component": comp_name,
                    "cell_verdict": cell_verdict,
                    "n_high_support": n_high_support,
                })

            comp_result["per_band"][band_name] = row
        results.append(comp_result)

    total_cells = _N_HEADS * _N_BANDS
    # Codex P1 2026-08-13: promotion requires ALL n_cells_total cells to PASS.
    # A single passing cell surrounded by skips must not be reported as PASS.
    # If a different coverage rule is agreed with the user, freeze it here
    # (and update tests + §16 wording).
    if n_fail > 0:
        overall = "FAIL"
    elif n_pass == total_cells:
        overall = "PASS"
    else:
        overall = "FAIL_INSUFFICIENT_COVERAGE"
    summary = {
        "overall_verdict":              overall,
        "n_cells_total":                total_cells,
        "n_cells_pass":                 n_pass,
        "n_cells_fail":                 n_fail,
        "n_cells_skip_insufficient":    n_insufficient,
        "n_cells_skip_degenerate":      n_degenerate,
        "n_cells_skip_non_finite":      n_non_finite,
        "failing_cells":                failing_cells,
        "skipped_cells":                skipped_cells,
        "coverage_rule":                "ALL_CELLS_MUST_PASS",
        "thresholds": {
            "x_a":                x_a,
            "x_b":                x_b,
            "min_n_high_support": min_n_high_support,
            "min_denominator":    min_denominator,
            "gate_1_formula":     "MSE(candidate,emp) ≤ (1 − x_a) × MSE(uniform,emp)",
            "gate_2_formula":     "MSE(candidate,emp) ≤ (1 + x_b) × MSE(teacher,emp)",
        },
    }
    return results, summary


def _combine_gate_verdicts(g1: str, g2: str) -> str:
    """Combine two gate verdicts into a cell verdict.

    Precedence (highest first):
      FAIL_DIVERGED       → training numerical blow-up.
      FAIL_UNEXPECTED_NAN → supported cell with NaN MSE (Codex P1 2026-08-13).
      FAIL                → normal threshold miss.
      PASS                → both gates passed.
      SKIP_INSUFFICIENT_SUPPORT / SKIP_DEGENERATE_DENOMINATOR
                          → cell not evaluable; does NOT block on its own but
                            prevents overall PASS under the ALL_CELLS_MUST_PASS
                            coverage rule.
    """
    if "FAIL_DIVERGED" in (g1, g2):
        return "FAIL_DIVERGED"
    if "FAIL_UNEXPECTED_NAN" in (g1, g2):
        return "FAIL_UNEXPECTED_NAN"
    if "FAIL" in (g1, g2):
        return "FAIL"
    if g1 == "PASS" and g2 == "PASS":
        return "PASS"
    # Both are some SKIP_* variant.  Prefer more informative one.
    if any("INSUFFICIENT_SUPPORT" in v for v in (g1, g2)):
        return "SKIP_INSUFFICIENT_SUPPORT"
    return "SKIP_DEGENERATE_DENOMINATOR"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default=str(_ROOT / "data" / "gap_net_v3_dataset_v2"))
    # Default output name uses the Stage-E scope (Codex 2026-08-15 3rd pass P1):
    # the plain "gap_net_v3_candidate.npz" name is reserved for the Stage F
    # artifact — this trainer only produces the Stage E candidate.
    p.add_argument("--out",
                   default=str(_ROOT / "data" / "gap_net_v3_stage_e_candidate.npz"))
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--batch-size",   type=int,   default=4096)
    p.add_argument("--patience",     type=int,   default=15,
                   help="Early stop after N epochs without val improvement")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--d4-augmentation", choices=("on", "off"), default="off",
                   help="Random D4 board symmetry per batch sample (§11.2 ablation)")
    # ── Stage E gate thresholds (Codex P2-A, 2026-08-12) ────────────────────
    p.add_argument("--stage-e-x-a", type=float, default=0.30,
                   help="Gate 1 uniform-improvement threshold X_A (default 30 %%). "
                        "Draft per §16; user reviews before Stage E run.")
    p.add_argument("--stage-e-x-b", type=float, default=0.20,
                   help="Gate 2 teacher-tolerance threshold X_B (default 20 %%). "
                        "Draft per §16.")
    p.add_argument("--stage-e-min-high-support", type=int, default=100,
                   help="Minimum n_high_support per (band × component) cell for "
                        "gate evaluation.  Cells below → SKIP_INSUFFICIENT_SUPPORT.")
    p.add_argument("--stage-e-min-denominator", type=float, default=1e-9,
                   help="Minimum reference-MSE magnitude; below → SKIP_DEGENERATE_DENOMINATOR.")
    p.add_argument("--allow-non-production-dataset", action="store_true",
                   help="Codex P1(4) 2026-08-14: refuse to train against a dataset "
                        "whose provenance says production_ready=False (smoke, "
                        "sub-floor, or hardening-bypassed extractions).  Pass this "
                        "flag to override — the run will NOT be eligible for "
                        "promotion evidence.")
    # Codex 2026-08-15 (3rd pass) P1: --gate-thresholds-frozen-id must resolve
    # to a real entry in configs/stage_e_thresholds.json with frozen=true, AND
    # the runtime CLI values for x_a/x_b/min-support/min-denominator must EQUAL
    # that entry's values.  Anything else raises SystemExit.  Absent/empty →
    # Stage F ineligible (draft thresholds carry no authority).
    p.add_argument("--gate-thresholds-frozen-id", default=None,
                   help="Registry id (from configs/stage_e_thresholds.json) "
                        "of the owner-reviewed frozen Stage E threshold set.  "
                        "Required for stage_f_eligible=True.  Trainer rejects "
                        "unknown ids, draft (frozen=false) entries, and CLI "
                        "values that do not match the entry exactly.  "
                        "promotion_eligible is ALWAYS False from this trainer — "
                        "only Stage F tooling produces a promotable artifact.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gap_net_v3] device={device}  d4_augmentation={args.d4_augmentation}")

    dataset_dir = Path(args.dataset_dir)
    out_path    = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Codex P1(4) 2026-08-14 + 2026-08-15 taint propagation: verify dataset is
    # production-ready before load; capture taint fields for the saved model.
    dataset_taint = _verify_dataset_production_ready(
        dataset_dir, allow_non_production=args.allow_non_production_dataset,
    )

    # Codex 2026-08-15 (3rd pass) P1: range-validate CLI thresholds regardless
    # of registry state — negative x_a / non-finite / degenerate values must
    # not silently reach the gate math where they would flip legitimate FAILs
    # to PASS.
    _validate_threshold_ranges(args)

    # Codex 2026-08-15 (3rd pass) P1: resolve --gate-thresholds-frozen-id to a
    # registry entry with frozen=true whose values equal the CLI values EXACTLY.
    # Anything less is a self-assertion — no promotion authority.
    frozen_entry: Optional[dict] = None
    frozen_sha256: Optional[str] = None
    if args.gate_thresholds_frozen_id:
        frozen_entry, frozen_sha256 = _load_frozen_thresholds(
            args.gate_thresholds_frozen_id,
        )
        mismatch = _cli_matches_frozen_entry(args, frozen_entry)
        if mismatch is not None:
            raise SystemExit(
                f"[gap_net_v3] --gate-thresholds-frozen-id "
                f"{args.gate_thresholds_frozen_id!r} does not match runtime "
                f"CLI values: {mismatch}.  Either use the frozen values or "
                f"omit the id (which forfeits Stage F eligibility)."
            )
        print(f"[gap_net_v3] Frozen Stage E thresholds loaded: id="
              f"{args.gate_thresholds_frozen_id!r}  sha256={frozen_sha256[:16]}…")

    # ── Load train / val ─────────────────────────────────────────────────────
    print("[gap_net_v3] Loading train split …")
    t0 = time.time()
    tr = _load_split(dataset_dir, split_val=0)
    print(f"[gap_net_v3] Train: {len(tr['board']):,} rows  ({time.time()-t0:.1f}s)")

    print("[gap_net_v3] Loading val split …")
    t0 = time.time()
    va = _load_split(dataset_dir, split_val=1)
    print(f"[gap_net_v3] Val:   {len(va['board']):,} rows  ({time.time()-t0:.1f}s)")

    # ── Coverage per component per band ──────────────────────────────────────
    for c in range(_N_HEADS):
        n_valid = int((~torch.isnan(tr["y_model"][:, c])).sum())
        print(f"[gap_net_v3]   comp {_COMP_NAMES[c]}: "
              f"{n_valid:,}/{len(tr['board']):,} valid train targets")

    # ── Mean-predictor baseline (train target mean per component) ────────────
    tr_means_list: list[float] = []
    for c in range(_N_HEADS):
        mask = ~torch.isnan(tr["y_model"][:, c])
        tr_means_list.append(
            float(tr["y_model"][mask, c].mean()) if mask.any() else float("nan")
        )
    print(f"[gap_net_v3] Train component means (mean-predictor baseline): "
          f"{[f'{v:.6f}' for v in tr_means_list]}")

    # ── Build inputs ────────────────────────────────────────────────────────
    X_tr = _build_features(tr["board"], tr["band"])
    y_tr = tr["y_model"]

    tr_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    perm_t = torch.from_numpy(np.stack(_BOARD_PERMS)).long().to(device)  # (8, 79)

    X_va = _build_features(va["board"], va["band"]).to(device)
    y_va = va["y_model"].to(device)

    # ── Model + optimiser ────────────────────────────────────────────────────
    model = GapNetV3().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[gap_net_v3] Model: {n_params:,} parameters (input={_INPUT_DIM})")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    d4_on = args.d4_augmentation == "on"

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_steps = 0.0, 0
        t_ep = time.time()

        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            if d4_on:
                sym_choice   = torch.randint(0, 8, (xb.size(0),), device=device)
                board_gather = perm_t[sym_choice]                     # (B, 79)
                board_part   = torch.gather(xb[:, :_BOARD_DIM], 1, board_gather)
                xb           = torch.cat([board_part, xb[:, _BOARD_DIM:]], dim=1)
            pred = model(xb)
            loss = _nan_mse(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss  += loss.item()
            ep_steps += 1

        tr_loss = ep_loss / max(ep_steps, 1)

        model.eval()
        with torch.no_grad():
            val_pred = _forward_chunked(model, X_va)
            val_loss = _nan_mse(val_pred, y_va).item()

        elapsed = time.time() - t_ep
        print(f"[gap_net_v3] epoch {epoch:3d}/{args.epochs}  "
              f"tr={tr_loss:.6f}  val={val_loss:.6f}  ({elapsed:.1f}s)")

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"[gap_net_v3] Early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[gap_net_v3] Restored best checkpoint (val_loss={best_val_loss:.6f})")

    # ── Gate metrics per component per band ──────────────────────────────────
    model.eval()
    with torch.no_grad():
        val_pred_np = _forward_chunked(model, X_va).cpu().numpy()

    gate_results, gate_summary = _report_gate(
        val_pred_np,
        va["y_model"].numpy(),
        va["y_unif"].numpy(),
        va["y_emp"].numpy(),
        va["band"].numpy(),
        np.array(tr_means_list, dtype=np.float64),
        x_a=args.stage_e_x_a,
        x_b=args.stage_e_x_b,
        min_n_high_support=args.stage_e_min_high_support,
        min_denominator=args.stage_e_min_denominator,
    )
    print("[gap_net_v3] Stage E gate summary (§16 revised 2026-08-12):")
    for comp in gate_results:
        print(f"  Component {comp['component']}:")
        for band, row in comp["per_band"].items():
            base = (f"    {band}: n_valid={row['n_valid']:,}  "
                    f"n_high_support={row['n_high_support']:,}  "
                    f"teacher_fidelity={row['candidate_teacher_fidelity_mse']:.6f}  "
                    f"mean_pred={row['mean_predictor_vs_teacher_mse']:.6f}")
            print(base)
            print(f"      vs empirical  → candidate={row['candidate_vs_empirical_mse']}  "
                  f"teacher={row['teacher_vs_empirical_mse']}  "
                  f"uniform={row['uniform_vs_empirical_mse']}")
            print(f"      gate_1={row['gate_1_verdict']}  "
                  f"gate_2={row['gate_2_verdict']}  "
                  f"cell={row['cell_verdict']}")
    print(f"[gap_net_v3] OVERALL VERDICT: {gate_summary['overall_verdict']}  "
          f"pass={gate_summary['n_cells_pass']}  "
          f"fail={gate_summary['n_cells_fail']}  "
          f"skip_insufficient={gate_summary['n_cells_skip_insufficient']}  "
          f"skip_degenerate={gate_summary['n_cells_skip_degenerate']}  "
          f"skip_non_finite={gate_summary['n_cells_skip_non_finite']}")

    # Codex 2026-08-15 (2nd pass) P1 + P2: provenance + model label are built
    # in exactly ONE place — `_build_saved_provenance`.  main() is a strict
    # thin wrapper; do NOT construct provenance fields here.  Unit + bounded
    # integration tests exercise this seam directly.
    provenance, model_label = _build_saved_provenance(
        dataset_taint=dataset_taint,
        dataset_dir=dataset_dir,
        gate_results=gate_results,
        gate_summary=gate_summary,
        frozen_thresholds_entry=frozen_entry,
        frozen_thresholds_sha256=frozen_sha256,
        args=args,
        n_train=len(tr["board"]),
        n_val=len(va["board"]),
        best_val_loss=best_val_loss,
        epochs_trained=epoch,
        tr_means=tr_means_list,
    )
    print(f"[gap_net_v3] Model label: {model_label}")
    print(f"[gap_net_v3]   stage_e_passed:    {provenance['stage_e_passed']}")
    print(f"[gap_net_v3]   stage_f_eligible:  {provenance['stage_f_eligible']}")
    print(f"[gap_net_v3]   promotion_eligible:{provenance['promotion_eligible']} "
          "(always False from Stage E trainer)")
    if not provenance["stage_f_eligible"]:
        print("[gap_net_v3]   stage_f_ineligible_reasons:")
        for r in provenance["stage_f_ineligible_reasons"]:
            print(f"[gap_net_v3]     - {r}")

    model_cpu = model.cpu()
    _save(model_cpu, out_path, provenance)
    size_kb = round(out_path.stat().st_size / 1024, 1)
    print(f"[gap_net_v3] Saved → {out_path}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
