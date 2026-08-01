#!/usr/bin/env python3
"""tools/eval_human_move_policy_net_V2.py — Hardened Phase 4b evaluation.

Implements all items in docs/human_move_policy_eval_hardening_plan.md:

§1  Strict inference via probs_strict / raw_logits_strict.  Fail-closed skip
    policy with six counters.  Full-corpus feature-bank audit prerequisite noted.
§2  Two-tier degrade semantics: formal (Option 1, complete labels only) and
    diagnostic (Option 2, labelled-subset conditioning, marked diagnostic_only).
    Field names use wdl_severity, not regret.
§3  Matched macro and micro calibration pairs.  Fly phase distinguished.
    Per-band and per-phase degrade calibration.
§4  Single-pass strict logits → T=1 and T* probs from identical rows.
    Hard failure on optimizer error or boundary saturation.
§5  in_sample_empirical_reference (descriptive only, excluded from gates).
    Uniform top-k reports expected random coverage.  Model top-k renamed.
§6  OOD removed → split_integrity_check.  game_val_only →
    val_game_exclusive_stratum with exact condition documented.  Session-index
    incompatibility raises, does not warn.
§7  Provenance chain with DB hash, Malom label version, git commit, script SHA.
    DB opened read-only; PRAGMA quick_check run at startup.  provenance_ok flag.
§9  --run-test-set gated behind confirmation-slice warning; test is reserved.

Usage
-----
    # Standard val eval:
    .venv/bin/python tools/eval_human_move_policy_net_V2.py \\
        --dataset-dir data/human_move_policy_dataset \\
        --model       data/human_move_policy_net_v2_candidate.npz \\
        --candidate-db data/human_db_candidate.sqlite \\
        --output      data/gap_v3_prerequisite_eval_V2.json

    # With session index (enables val_game_exclusive_stratum):
    .venv/bin/python tools/eval_human_move_policy_net_V2.py \\
        --dataset-dir data/human_move_policy_dataset \\
        --model       data/human_move_policy_net_v2_candidate.npz \\
        --candidate-db data/human_db_candidate.sqlite \\
        --session-index data/human_move_policy_session_index.npz \\
        --output      data/gap_v3_prerequisite_eval_V2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.human_move_policy_advisor import HumanMovePolicyAdvisor   # noqa: E402
from game.board import BoardState                                   # noqa: E402
from game.rules import get_all_legal_moves, get_game_phase         # noqa: E402
from tools.train_value_net_v2 import board_from_state_key          # noqa: E402

sys.path.insert(0, str(_ROOT / "tools"))
import audit_human_moves as _ahb   # noqa: E402

_BAND_NAMES = ("lower", "middle", "upper")

_DEGRADE_CATS    = frozenset({"win_to_draw", "win_to_loss", "draw_to_loss"})
_WDL_SEV_WEIGHTS = {"win_to_loss": 1.0, "win_to_draw": 0.5, "draw_to_loss": 0.5}
_TRUSTED_CATS    = frozenset({
    "win_preserved", "win_to_draw", "win_to_loss",
    "draw_preserved", "draw_to_loss", "all_losing",
})
_REQUIRED_MALOM_VER = "sector-corrected-v1"
_T_BOUNDS = (0.1, 10.0)


# ── Metrics accumulator ─────────────────────────────────────────────────────

class Stratum:
    """Accumulate metrics for a slice of the evaluation set.

    Output field names follow the hardening plan:
      macro_multiclass_brier, human_event_topN_coverage, top_label_ece.
    """
    def __init__(self, name: str):
        self.name      = name
        self.n_events  = 0
        self.n_samples = 0
        self.nll_sum   = 0.0
        self.brier_sum = 0.0
        self.top1 = self.top3 = self.top5 = 0
        self.calib_bin_n     = np.zeros(10, dtype=np.int64)
        self.calib_bin_probs = np.zeros(10, dtype=np.float64)
        self.calib_bin_hits  = np.zeros(10, dtype=np.float64)

    def add_sample(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        notations: list[str],
        top1_notation: str,
        top3_notations: set[str],
        top5_notations: set[str],
    ) -> None:
        self.n_samples += 1
        total = int(targets.sum())
        if total == 0:
            return
        y = targets.astype(np.float64) / float(total)
        p = probs.astype(np.float64)
        self.brier_sum += float(np.sum((p - y) ** 2))
        top_conf = float(probs.max())
        for j, count in enumerate(targets):
            if count == 0:
                continue
            prob_j   = float(probs[j])
            was_top1 = (notations[j] == top1_notation)
            for _ in range(int(count)):
                self.n_events += 1
                self.nll_sum  += -np.log(max(prob_j, 1e-9))
                self.top1     += int(was_top1)
                self.top3     += int(notations[j] in top3_notations)
                self.top5     += int(notations[j] in top5_notations)
                bi = min(9, max(0, int(top_conf * 10)))
                self.calib_bin_n[bi]     += 1
                self.calib_bin_probs[bi] += top_conf
                self.calib_bin_hits[bi]  += int(was_top1)

    def finalize(self) -> dict:
        n  = max(self.n_events, 1)
        ns = max(self.n_samples, 1)
        ece = 0.0
        for i in range(10):
            if self.calib_bin_n[i] == 0:
                continue
            conf = self.calib_bin_probs[i] / self.calib_bin_n[i]
            acc  = self.calib_bin_hits[i]  / self.calib_bin_n[i]
            ece += (self.calib_bin_n[i] / n) * abs(conf - acc)
        return {
            "name":                        self.name,
            "n_samples":                   int(self.n_samples),
            "n_events":                    int(self.n_events),
            "event_nll":                   float(self.nll_sum / n),
            "macro_multiclass_brier":      float(self.brier_sum / ns),
            "human_event_top1_coverage":   float(self.top1 / n),
            "human_event_top3_coverage":   float(self.top3 / n),
            "human_event_top5_coverage":   float(self.top5 / n),
            "top_label_ece":               float(ece),
        }


# ── Degrading-move calibration ──────────────────────────────────────────────

class DegradeCal:
    """Macro and micro calibration of P(degrade) and E[WDL severity].

    Macro: equal weight per (state_key, band) sample.
    Micro: weighted by obs_total (human event count) per sample.

    ECE bins are computed on pred_degrade with both macro (sample-weighted)
    and micro (event-weighted) normalisation.
    """
    N = 10

    def __init__(self) -> None:
        self.n_pos = 0
        # Macro (per sample)
        self._m_pd = 0.0; self._m_od = 0.0
        self._m_pw = 0.0; self._m_ow = 0.0
        # Micro (event-weighted)
        self._u_pd = 0.0; self._u_od = 0; self._u_tot = 0
        self._u_pw = 0.0; self._u_ow = 0.0
        # ECE bins — macro (sample-weighted)
        self.mac_n = np.zeros(self.N); self.mac_p = np.zeros(self.N); self.mac_o = np.zeros(self.N)
        # ECE bins — micro (event-weighted)
        self.mic_w = np.zeros(self.N); self.mic_p = np.zeros(self.N); self.mic_o = np.zeros(self.N)

    def add(
        self,
        pred_degrade: float,
        pred_wdl_sev: float,
        obs_degrade: int,
        obs_wdl_sev: float,
        obs_total: int,
    ) -> None:
        obs_freq    = obs_degrade / max(obs_total, 1)
        obs_wdl_avg = obs_wdl_sev / max(obs_total, 1)
        self.n_pos  += 1
        self._m_pd  += pred_degrade;  self._m_od += obs_freq
        self._m_pw  += pred_wdl_sev;  self._m_ow += obs_wdl_avg
        self._u_pd  += pred_degrade * obs_total;  self._u_od += obs_degrade
        self._u_tot += obs_total
        self._u_pw  += pred_wdl_sev * obs_total;  self._u_ow += obs_wdl_sev
        bi = min(self.N - 1, int(pred_degrade * self.N))
        self.mac_n[bi] += 1; self.mac_p[bi] += pred_degrade; self.mac_o[bi] += obs_freq
        self.mic_w[bi] += obs_total
        self.mic_p[bi] += pred_degrade * obs_total
        self.mic_o[bi] += obs_degrade

    def finalize(self) -> dict:
        n  = max(self.n_pos, 1)
        ut = max(self._u_tot, 1)

        def _mac_ece():
            ece = 0.0
            for i in range(self.N):
                if self.mac_n[i] == 0: continue
                ece += (self.mac_n[i] / n) * abs(self.mac_p[i]/self.mac_n[i] - self.mac_o[i]/self.mac_n[i])
            return float(ece)

        def _mic_ece():
            ece = 0.0
            tw  = max(self._u_tot, 1)
            for i in range(self.N):
                if self.mic_w[i] == 0: continue
                ece += (self.mic_w[i] / tw) * abs(self.mic_p[i]/self.mic_w[i] - self.mic_o[i]/self.mic_w[i])
            return float(ece)

        return {
            "n_samples":                  int(self.n_pos),
            "macro_pred_degrade":         float(self._m_pd / n),
            "macro_obs_degrade":          float(self._m_od / n),
            "micro_pred_degrade":         float(self._u_pd / ut),
            "micro_obs_degrade":          float(self._u_od / ut),
            "macro_degrade_ece":          _mac_ece(),
            "micro_degrade_ece":          _mic_ece(),
            "macro_pred_wdl_severity":    float(self._m_pw / n),
            "macro_obs_wdl_severity":     float(self._m_ow / n),
            "micro_pred_wdl_severity":    float(self._u_pw / ut),
            "micro_obs_wdl_severity":     float(self._u_ow / ut),
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _move_notation(mv: dict) -> str:
    frm = mv.get("from"); to = mv.get("to") or ""; cap = mv.get("capture") or ""
    base = f"{frm}-{to}" if frm else to
    return f"{base}x{cap}" if cap else base


def _softmax_from_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / max(temperature, 1e-3)
    scaled = scaled - scaled.max()
    exp_s  = np.exp(scaled).astype(np.float32)
    return exp_s / float(exp_s.sum())


def _in_sample_empirical_probs(targets: np.ndarray) -> Optional[np.ndarray]:
    total = float(targets.sum())
    if total <= 0:
        return None
    return targets.astype(np.float32) / total


def _lmc_bin(n: int) -> str:
    if n <= 5:  return "lmc_2-5"
    if n <= 10: return "lmc_6-10"
    if n <= 20: return "lmc_11-20"
    return "lmc_21+"


def _sha256_file(p: Path) -> Optional[str]:
    if not p.exists(): return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info() -> tuple[Optional[str], Optional[bool]]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_ROOT, text=True
        ).strip())
        return commit, dirty
    except Exception:
        return None, None


# ── Provenance chain ─────────────────────────────────────────────────────────

def _compute_provenance_chain(
    model_path: Path,
    candidate_db: Path,
    dataset_dir: Path,
    script_path: Path,
    session_index_path: Optional[Path] = None,
) -> dict:
    """Hash all artefacts and check mutual compatibility.

    Returns a dict with provenance_ok: bool and compatibility_failures: list.
    """
    failures: list[str] = []

    model_sha   = _sha256_file(model_path)
    db_sha      = _sha256_file(candidate_db)
    script_sha  = _sha256_file(script_path)
    meta_sha    = _sha256_file(dataset_dir / "metadata.npz")
    git_commit, git_dirty = _git_info()

    # Load model provenance.
    model_prov: dict = {}
    try:
        d = np.load(str(model_path), allow_pickle=True)
        if "provenance_json" in d.files:
            model_prov = json.loads(str(d["provenance_json"].item()))
    except Exception as exc:
        failures.append(f"Could not read model provenance: {exc}")

    # Load dataset provenance.
    ds_prov: dict = {}
    try:
        d2 = np.load(str(dataset_dir / "metadata.npz"), allow_pickle=True)
        ds_prov = json.loads(str(d2["provenance"].item()))
    except Exception as exc:
        failures.append(f"Could not read dataset provenance: {exc}")

    # Open DB read-only and run quick_check.
    db_meta: dict = {}
    quick_check_result = "not_run"
    try:
        conn = sqlite3.connect(f"file:{candidate_db}?mode=ro", uri=True)
        try:
            for k, v in conn.execute("SELECT key, value FROM meta"):
                db_meta[k] = v
            quick_check_result = conn.execute("PRAGMA quick_check").fetchone()[0]
        finally:
            conn.close()
    except Exception as exc:
        failures.append(f"DB open/check failed: {exc}")

    if quick_check_result != "ok":
        failures.append(f"DB PRAGMA quick_check returned: {quick_check_result!r}")

    # Malom label version.
    malom_ver = db_meta.get("malom_label_version", "<missing>")
    if malom_ver != _REQUIRED_MALOM_VER:
        failures.append(
            f"Malom label version is {malom_ver!r}, expected {_REQUIRED_MALOM_VER!r}"
        )

    # DB hash: must match dataset provenance.
    ds_db_sha = ds_prov.get("candidate_db_sha256")
    if ds_db_sha and db_sha and ds_db_sha != db_sha:
        failures.append(
            f"DB SHA mismatch vs dataset provenance: "
            f"stored={ds_db_sha[:12]}… actual={db_sha[:12]}…"
        )

    # DB hash: must match model-embedded provenance (if present).
    mp_db_sha = (
        model_prov.get("candidate_db_sha256")
        or (model_prov.get("dataset_provenance") or {}).get("candidate_db_sha256")
    )
    if mp_db_sha and db_sha and mp_db_sha != db_sha:
        failures.append(
            f"DB SHA mismatch vs model provenance: "
            f"stored={mp_db_sha[:12]}… actual={db_sha[:12]}…"
        )

    # Session index compatibility.
    si_sha: Optional[str] = None
    si_prov: dict = {}
    if session_index_path is not None:
        si_sha = _sha256_file(session_index_path)
        try:
            si = np.load(str(session_index_path), allow_pickle=True)
            si_prov = json.loads(str(si["provenance"].item()))
            si_meta_sha = si_prov.get("dataset_metadata_sha256") or si_prov.get("metadata_sha256")
            if si_meta_sha and meta_sha and si_meta_sha != meta_sha:
                failures.append(
                    f"Session index embedded metadata SHA mismatch: "
                    f"stored={si_meta_sha[:12]}… actual={meta_sha[:12]}…"
                )
        except Exception as exc:
            failures.append(f"Session index provenance unreadable: {exc}")

    return {
        "model_sha256":             model_sha,
        "candidate_db_sha256":      db_sha,
        "malom_label_version":      malom_ver,
        "db_quick_check":           quick_check_result,
        "dataset_metadata_sha256":  meta_sha,
        "evaluator_script_sha256":  script_sha,
        "evaluator_git_commit":     git_commit,
        "evaluator_git_dirty":      git_dirty,
        "session_index_sha256":     si_sha,
        "model_provenance":         model_prov,
        "dataset_provenance":       ds_prov,
        "db_meta":                  db_meta,
        "compatibility_failures":   failures,
        "provenance_ok":            len(failures) == 0,
    }


# ── Session index ─────────────────────────────────────────────────────────────

def _load_session_index(
    path: Path, ds
) -> np.ndarray:
    """Load session index and fail-close on any incompatibility."""
    if not path.exists():
        raise RuntimeError(f"--session-index path not found: {path}")
    si = np.load(str(path), allow_pickle=True)
    mask: np.ndarray = si["game_split_mask"].astype(np.uint8)
    # Length check.
    n_state_keys = ds.state_keys.shape[0]
    if mask.shape[0] != n_state_keys:
        raise RuntimeError(
            f"Session index length {mask.shape[0]} != dataset n_state_keys {n_state_keys}"
        )
    return mask


# ── Temperature scaling (pass 1) ──────────────────────────────────────────────

def _find_temperature_strict(
    dataset_dir: Path,
    adv: HumanMovePolicyAdvisor,
    val_idx: np.ndarray,
    ds,
) -> float:
    """Find T* minimising val NLL using raw_logits_strict.

    Raises RuntimeError if:
      - scipy is not available
      - optimizer does not converge
      - T* saturates at a boundary (0.1 or 10.0)
    """
    try:
        from scipy.optimize import minimize_scalar
    except ImportError:
        raise RuntimeError(
            "scipy is required for temperature scaling. "
            "Install it: .venv/bin/pip install scipy  "
            "Or pass --skip-temperature to use T=1.0 explicitly."
        )

    logit_list:  list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    n_skipped = 0

    for sid in val_idx:
        state_idx   = int(ds.sample_state_idx[sid])
        band_idx    = int(ds.sample_band_idx[sid])
        band_name   = _BAND_NAMES[band_idx]
        state_key   = str(ds.state_keys[state_idx])
        legal_count = int(ds.state_succ_offsets[state_idx + 1] - ds.state_succ_offsets[state_idx])

        board = board_from_state_key(state_key)
        if board is None:
            n_skipped += 1; continue
        legal = get_all_legal_moves(board)
        if len(legal) != legal_count:
            n_skipped += 1; continue

        tgt_start = int(ds.sample_offsets[sid])
        tgt_end   = int(ds.sample_offsets[sid + 1])
        targets   = ds.sample_targets[tgt_start:tgt_end].astype(np.int64)
        if targets.sum() == 0:
            n_skipped += 1; continue

        try:
            logits = adv.raw_logits_strict(board, legal, band_name)
        except Exception:
            n_skipped += 1; continue

        logit_list.append(logits.astype(np.float64))
        target_list.append(targets)

    if not logit_list:
        raise RuntimeError("No valid samples collected for temperature scaling.")

    print(f"[eval/temp] Collected {len(logit_list):,} samples "
          f"(skipped {n_skipped:,}) for temperature fitting.")

    def _val_nll(T: float) -> float:
        T = max(T, 1e-3)
        nll = 0.0; events = 0
        for logits, targets in zip(logit_list, target_list):
            scaled = logits / T - logits.max() / T
            log_p  = scaled - np.log(np.exp(scaled).sum())
            for j, cnt in enumerate(targets):
                if cnt > 0:
                    nll += -float(cnt) * log_p[j]
                    events += int(cnt)
        return nll / max(events, 1)

    result = minimize_scalar(
        _val_nll, bounds=_T_BOUNDS, method="bounded",
        options={"xatol": 1e-5, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(
            f"Temperature optimizer did not converge: {result.message}"
        )
    T_star = float(result.x)
    if T_star <= _T_BOUNDS[0] + 1e-4 or T_star >= _T_BOUNDS[1] - 1e-4:
        raise RuntimeError(
            f"T* = {T_star:.4f} saturates at search boundary {_T_BOUNDS}. "
            "Inspect the model — temperature scaling may not be appropriate. "
            "Pass --skip-temperature to use T=1.0 explicitly."
        )
    nll_1  = _val_nll(1.0)
    print(f"[eval/temp] T*={T_star:.4f}  NLL(T=1)={nll_1:.5f}  NLL(T*)={result.fun:.5f}")
    return T_star


# ── Per-position Malom label processing ──────────────────────────────────────

def _process_malom_labels(
    probs: np.ndarray,
    notations: list[str],
    targets: np.ndarray,
    state_key: str,
    pos_wdl: dict,
    move_wdl: dict,
) -> dict:
    """Compute degrade stats for both formal and diagnostic tiers.

    Formal (Option 1): all legal moves must have trusted labels.
    Diagnostic (Option 2): conditioned on the labelled subset.

    Returns a dict with all needed fields for both tiers and the
    by_transition accumulation data.
    """
    n_legal  = len(notations)
    pre      = pos_wdl.get(state_key)
    obs_total = int(targets.sum())

    cats: list[str] = []
    for j in range(n_legal):
        aft = move_wdl.get((state_key, notations[j]))
        cats.append(_ahb._classify_transition(pre, aft))

    n_trusted = sum(1 for c in cats if c in _TRUSTED_CATS)
    has_all_trusted = (n_trusted == n_legal)

    # Formal (Option 1): accumulate only if all trusted.
    formal_pd = 0.0; formal_ps = 0.0
    formal_od = 0;   formal_os = 0.0
    if has_all_trusted:
        for j, cat in enumerate(cats):
            is_deg = cat in _DEGRADE_CATS
            w      = _WDL_SEV_WEIGHTS.get(cat, 0.0)
            formal_pd += float(probs[j]) * (1.0 if is_deg else 0.0)
            formal_ps += float(probs[j]) * w
            if targets[j] > 0:
                formal_od += int(is_deg) * int(targets[j])
                formal_os += w * int(targets[j])

    # Diagnostic (Option 2): condition on labelled subset.
    labelled_mass = sum(float(probs[j]) for j, c in enumerate(cats) if c in _TRUSTED_CATS)
    diag_pd = 0.0; diag_ps = 0.0
    diag_od = 0;   diag_os = 0.0
    diag_obs_labelled = 0
    for j, cat in enumerate(cats):
        if cat not in _TRUSTED_CATS:
            continue
        is_deg = cat in _DEGRADE_CATS
        w      = _WDL_SEV_WEIGHTS.get(cat, 0.0)
        diag_pd += float(probs[j]) * (1.0 if is_deg else 0.0)
        diag_ps += float(probs[j]) * w
        if targets[j] > 0:
            diag_od += int(is_deg) * int(targets[j])
            diag_os += w * int(targets[j])
            diag_obs_labelled += int(targets[j])

    # Normalise diagnostic quantities by labelled mass (conditional probability).
    if labelled_mass > 0:
        diag_pd_cond = diag_pd / labelled_mass
        diag_ps_cond = diag_ps / labelled_mass
    else:
        diag_pd_cond = diag_ps_cond = 0.0

    return {
        "cats":             cats,
        "n_legal":          n_legal,
        "n_trusted":        n_trusted,
        "has_all_trusted":  has_all_trusted,
        "labelled_mass":    labelled_mass,
        # Formal: raw sums for DegradeCal.add()
        "formal_pd":        formal_pd, "formal_ps": formal_ps,
        "formal_od":        formal_od, "formal_os": formal_os,
        "obs_total":        obs_total,
        # Diagnostic: conditional (normalised by labelled mass)
        "diag_pd":          diag_pd_cond, "diag_ps": diag_ps_cond,
        "diag_od":          diag_od,      "diag_os": diag_os,
        "diag_obs_labelled": diag_obs_labelled,
    }


# ── Core eval loop ────────────────────────────────────────────────────────────

def _run_eval_loop(
    ds,
    sample_idx: np.ndarray,
    adv: HumanMovePolicyAdvisor,
    T_star: float,
    pos_wdl: dict,
    move_wdl: dict,
    train_state_keys: set[str],
    game_split_mask: Optional[np.ndarray],
    min_support: int,
    *,
    label: str = "val",
) -> dict:
    t0 = time.time()
    MASK_TRAIN = np.uint8(0x01)
    MASK_VAL   = np.uint8(0x02)

    # Skip counters (§1).
    n_attempted = n_evaluated = 0
    n_skip_board = n_skip_lmc = n_skip_zero_tgt = n_skip_inf = 0

    # T* accumulators (primary).
    def _strata():
        return {
            "overall":  Stratum("overall"),
            "by_band":  {b: Stratum(f"band={b}") for b in _BAND_NAMES},
            "by_phase": {p: Stratum(f"phase={p}") for p in ("place", "move", "fly")},
            "by_lmc":   {k: Stratum(k) for k in ("lmc_2-5", "lmc_6-10", "lmc_11-20", "lmc_21+")},
            "by_trans": {},
            "abstention": Stratum("abstention"),
            "val_game_excl": Stratum("val_game_exclusive") if game_split_mask is not None else None,
            "uni_overall":  Stratum("uniform_overall"),
            "uni_by_band":  {b: Stratum(f"uniform_band={b}") for b in _BAND_NAMES},
            "emp_overall":  Stratum("in_sample_empirical_overall"),
            "emp_by_band":  {b: Stratum(f"in_sample_empirical_band={b}") for b in _BAND_NAMES},
        }

    S  = _strata()              # T* strata
    S1 = {"overall": Stratum("overall_t1")}   # T=1 overall only (for comparison)

    # Degrade calibration: formal and diagnostic, for T* and T=1.
    formal_cal_ts  = DegradeCal(); diag_cal_ts  = DegradeCal()
    formal_cal_t1  = DegradeCal(); diag_cal_t1  = DegradeCal()

    # Per-band and per-phase degrade cals (T* only, formal).
    formal_band_cal = {b: DegradeCal() for b in _BAND_NAMES}
    formal_phase_cal = {p: DegradeCal() for p in ("place", "move", "fly")}

    # KL and empirical tracking.
    kl_sum = 0.0; kl_n = 0
    n_skipped_partial_labels = 0

    # Uniform expected top-k accumulators.
    uni_topk_sum = {1: 0.0, 3: 0.0, 5: 0.0}; uni_topk_n = 0

    for i, sid in enumerate(sample_idx):
        n_attempted += 1
        state_idx   = int(ds.sample_state_idx[sid])
        band_idx    = int(ds.sample_band_idx[sid])
        band_name   = _BAND_NAMES[band_idx]
        state_key   = str(ds.state_keys[state_idx])
        mover       = str(ds.mover_colors[state_idx])
        legal_count = int(ds.state_succ_offsets[state_idx + 1] - ds.state_succ_offsets[state_idx])

        board = board_from_state_key(state_key)
        if board is None:
            n_skip_board += 1; continue

        legal = get_all_legal_moves(board)
        if len(legal) != legal_count:
            n_skip_lmc += 1; continue

        tgt_start = int(ds.sample_offsets[sid])
        tgt_end   = int(ds.sample_offsets[sid + 1])
        targets   = ds.sample_targets[tgt_start:tgt_end].astype(np.int64)
        if int(targets.sum()) == 0:
            n_skip_zero_tgt += 1; continue

        notations = [_move_notation(m) for m in legal]

        # Phase using get_game_phase (distinguishes fly).
        phase_str = get_game_phase(board, board.turn)  # "place", "move", or "fly"

        # Strict inference (§1) — single call, derive T=1 and T* from same logits.
        try:
            logits = adv.raw_logits_strict(board, legal, band_name)
        except Exception as exc:
            n_skip_inf += 1
            continue

        probs_ts = _softmax_from_logits(logits, T_star)
        probs_t1 = _softmax_from_logits(logits, 1.0)
        n_evaluated += 1

        # Check encoding failure (abstention flag) — logits are finite so encoding
        # succeeded; this stratum collects samples with any encoding error that
        # probs() (non-strict) would have silently zeroed. Since probs_strict raised
        # above on encoding failure, abstention stratum is vacuous in V2 strict mode
        # (kept for schema compatibility).
        has_fail = False  # strict path already rejects encoding failures

        top_order    = np.argsort(-probs_ts)
        top1_n       = notations[top_order[0]]
        top3_set     = {notations[j] for j in top_order[:3]}
        top5_set     = {notations[j] for j in top_order[:5]}
        top_order_t1 = np.argsort(-probs_t1)
        top1_n_t1    = notations[top_order_t1[0]]
        top3_set_t1  = {notations[j] for j in top_order_t1[:3]}
        top5_set_t1  = {notations[j] for j in top_order_t1[:5]}

        def _add_ts(st: Stratum) -> None:
            st.add_sample(probs_ts, targets, notations, top1_n, top3_set, top5_set)

        # T* primary strata.
        _add_ts(S["overall"])
        _add_ts(S["by_band"][band_name])
        ph_key = phase_str if phase_str in S["by_phase"] else "move"
        _add_ts(S["by_phase"][ph_key])
        _add_ts(S["by_lmc"][_lmc_bin(legal_count)])
        if has_fail:
            _add_ts(S["abstention"])
        if S["val_game_excl"] is not None and game_split_mask is not None:
            mask = game_split_mask[state_idx]
            if (mask & MASK_TRAIN) == 0 and (mask & MASK_VAL) != 0:
                _add_ts(S["val_game_excl"])

        # T=1 comparison (overall only).
        S1["overall"].add_sample(probs_t1, targets, notations, top1_n_t1, top3_set_t1, top5_set_t1)

        # Malom labels (§2, §3).
        ml = _process_malom_labels(probs_ts, notations, targets, state_key, pos_wdl, move_wdl)

        # By-transition strata (T* only).
        for j, cat in enumerate(ml["cats"]):
            if targets[j] > 0:
                st = S["by_trans"].setdefault(cat, Stratum(f"trans={cat}"))
                st_tgt = np.zeros(legal_count, dtype=np.int64)
                st_tgt[j] = targets[j]
                st.add_sample(probs_ts, st_tgt, notations, top1_n, top3_set, top5_set)

        # Formal degrade cal (T* and T=1).
        if ml["has_all_trusted"] and ml["obs_total"] > 0:
            formal_cal_ts.add(ml["formal_pd"], ml["formal_ps"],
                               ml["formal_od"], ml["formal_os"], ml["obs_total"])
            formal_band_cal[band_name].add(ml["formal_pd"], ml["formal_ps"],
                                            ml["formal_od"], ml["formal_os"], ml["obs_total"])
            formal_phase_cal[ph_key].add(ml["formal_pd"], ml["formal_ps"],
                                          ml["formal_od"], ml["formal_os"], ml["obs_total"])
            # T=1 formal degrade cal.
            ml1 = _process_malom_labels(probs_t1, notations, targets, state_key, pos_wdl, move_wdl)
            formal_cal_t1.add(ml1["formal_pd"], ml1["formal_ps"],
                               ml1["formal_od"], ml1["formal_os"], ml1["obs_total"])
        else:
            n_skipped_partial_labels += 1

        # Diagnostic degrade cal (T* and T=1) — always, conditioned on labelled subset.
        if ml["obs_total"] > 0 and ml["n_trusted"] > 0:
            diag_cal_ts.add(ml["diag_pd"], ml["diag_ps"],
                             ml["diag_od"], ml["diag_os"], ml["diag_obs_labelled"])
            ml1 = _process_malom_labels(probs_t1, notations, targets, state_key, pos_wdl, move_wdl)
            diag_cal_t1.add(ml1["diag_pd"], ml1["diag_ps"],
                             ml1["diag_od"], ml1["diag_os"], ml1["diag_obs_labelled"])

        # Uniform baseline (expected random top-k coverage).
        u    = np.full(legal_count, 1.0 / max(legal_count, 1), dtype=np.float32)
        u_t1 = notations[0]
        u_t3 = set(notations[:3])
        u_t5 = set(notations[:5])
        S["uni_overall"].add_sample(u, targets, notations, u_t1, u_t3, u_t5)
        S["uni_by_band"][band_name].add_sample(u, targets, notations, u_t1, u_t3, u_t5)
        uni_topk_sum[1] += 1.0 / max(legal_count, 1)
        uni_topk_sum[3] += min(3, legal_count) / max(legal_count, 1)
        uni_topk_sum[5] += min(5, legal_count) / max(legal_count, 1)
        uni_topk_n += 1

        # In-sample empirical reference (§5).
        total_events = int(targets.sum())
        if total_events >= min_support:
            emp = _in_sample_empirical_probs(targets)
            if emp is not None:
                e_ord = np.argsort(-emp)
                e_t1  = notations[e_ord[0]]
                e_t3  = {notations[j] for j in e_ord[:3]}
                e_t5  = {notations[j] for j in e_ord[:5]}
                S["emp_overall"].add_sample(emp, targets, notations, e_t1, e_t3, e_t5)
                S["emp_by_band"][band_name].add_sample(emp, targets, notations, e_t1, e_t3, e_t5)
                emp_f = targets.astype(np.float64) / float(total_events)
                eps   = 1e-9
                kl_sum += float(np.where(
                    emp_f > 0.0,
                    emp_f * (np.log(emp_f + eps) - np.log(probs_ts.astype(np.float64) + eps)),
                    0.0,
                ).sum())
                kl_n += 1

        if (i + 1) % 5000 == 0 or (i + 1) == len(sample_idx):
            elapsed = time.time() - t0
            print(f"[eval/{label}] {i + 1}/{len(sample_idx)}  ({elapsed:.1f}s)")

    # ── Split integrity check (§6: OOD replaced) ─────────────────────────────
    val_keys_in_train = sum(
        1 for sid in sample_idx
        if str(ds.state_keys[int(ds.sample_state_idx[sid])]) in train_state_keys
    )
    split_integrity = {
        "pass":               val_keys_in_train == 0,
        "n_val_keys_in_train": int(val_keys_in_train),
    }

    # ── Skip summary ─────────────────────────────────────────────────────────
    skip_counts = {
        "n_attempted_samples":          int(n_attempted),
        "n_evaluated_samples":          int(n_evaluated),
        "n_skipped_board_reconstruction": int(n_skip_board),
        "n_skipped_legal_count_mismatch": int(n_skip_lmc),
        "n_skipped_zero_target":         int(n_skip_zero_tgt),
        "n_skipped_inference_failure":   int(n_skip_inf),
        "n_skipped_partial_labels_formal": int(n_skipped_partial_labels),
    }
    inference_ok = (n_skip_inf == 0)
    board_skip_frac = (n_skip_board + n_skip_lmc) / max(n_attempted, 1)

    # Uniform expected top-k coverage (§5).
    uni_n = max(uni_topk_n, 1)
    expected_uniform_coverage = {
        "expected_top1_coverage": float(uni_topk_sum[1] / uni_n),
        "expected_top3_coverage": float(uni_topk_sum[3] / uni_n),
        "expected_top5_coverage": float(uni_topk_sum[5] / uni_n),
        "note": "Expected fraction of human events in model's random top-k set, "
                "averaged over samples by 1/n_legal per sample.",
    }

    model_ts = {
        "overall":       S["overall"].finalize(),
        "by_band":       {b: s.finalize() for b, s in S["by_band"].items()},
        "by_phase":      {p: s.finalize() for p, s in S["by_phase"].items()},
        "by_transition": {c: s.finalize() for c, s in S["by_trans"].items() if s.n_events > 0},
        "by_lmc":        {k: s.finalize() for k, s in S["by_lmc"].items()},
        "abstention":    S["abstention"].finalize(),
    }
    if S["val_game_excl"] is not None:
        model_ts["val_game_exclusive_stratum"] = S["val_game_excl"].finalize()
        model_ts["val_game_exclusive_condition"] = (
            "State key not reached by any train-game session "
            "AND reached by at least one val-game session, "
            "per game_split_mask bits MASK_TRAIN=0x01 MASK_VAL=0x02. "
            "Test-game membership not tracked in this session index."
        )

    result = {
        "n_samples_evaluated": int(n_evaluated),
        "elapsed_seconds":     round(time.time() - t0, 1),
        "skip_counts":         skip_counts,
        "formal_gate_skip_ok": inference_ok,
        "board_skip_fraction": float(board_skip_frac),
        "split_integrity_check": split_integrity,
        "model": model_ts,
        "t_1_comparison": {
            "overall": S1["overall"].finalize(),
            "formal_degrade_calibration": formal_cal_t1.finalize(),
            "diagnostic_degrade_calibration": {
                **diag_cal_t1.finalize(), "diagnostic_only": True,
            },
        },
        "degrade_calibration": {
            "formal": {
                **formal_cal_ts.finalize(),
                "n_skipped_partial_labels": int(n_skipped_partial_labels),
                "note": "Option 1: all legal moves must have trusted Malom labels.",
            },
            "formal_by_band":  {b: c.finalize() for b, c in formal_band_cal.items()},
            "formal_by_phase": {p: c.finalize() for p, c in formal_phase_cal.items()},
            "diagnostic": {
                **diag_cal_ts.finalize(),
                "diagnostic_only": True,
                "note": "Option 2: conditioned on labelled (trusted) subset only. "
                        "pred/obs normalised by labelled policy mass / labelled events.",
            },
        },
        "baseline_uniform": {
            "overall":  S["uni_overall"].finalize(),
            "by_band":  {b: s.finalize() for b, s in S["uni_by_band"].items()},
            "expected_random_coverage": expected_uniform_coverage,
        },
        "in_sample_empirical_reference": {
            "descriptive_only":  True,
            "note":              "Scores the sample's own observed distribution on itself. "
                                 "Non-predictive. Excluded from gates and promotion criteria.",
            "min_support":       int(min_support),
            "overall":           S["emp_overall"].finalize(),
            "by_band":           {b: s.finalize() for b, s in S["emp_by_band"].items()},
        },
        "observed_distribution_kl_to_model": {
            "min_support":  int(min_support),
            "n_samples":    int(kl_n),
            "mean_kl":      float(kl_sum / max(kl_n, 1)),
        },
    }
    return result


# ── Main evaluate() ──────────────────────────────────────────────────────────

def evaluate(
    dataset_dir: Path,
    model_path: Path,
    candidate_db: Path,
    session_index_path: Optional[Path] = None,
    min_support: int = 10,
    run_test_set: bool = False,
    skip_temperature: bool = False,
    ignore_provenance_failures: bool = False,
) -> dict:
    from tools.train_human_move_policy_net import MovePolicyDataset

    t0 = time.time()
    script_path = Path(__file__).resolve()

    # Provenance chain (§7).
    print("[eval] Computing provenance chain …")
    prov = _compute_provenance_chain(
        model_path, candidate_db, dataset_dir, script_path, session_index_path
    )
    if not prov["provenance_ok"]:
        msg = "Provenance chain FAILED:\n  " + "\n  ".join(prov["compatibility_failures"])
        if ignore_provenance_failures:
            print(f"[eval] WARNING: {msg}")
            print("[eval] Continuing because --ignore-provenance-failures was set.")
        else:
            raise RuntimeError(msg)

    ds = MovePolicyDataset(dataset_dir)
    val_idx  = ds.val_idx()
    test_idx = ds.test_idx() if run_test_set else np.array([], dtype=np.int64)
    print(f"[eval] val={len(val_idx):,}  test={len(test_idx):,}  (v2={ds._is_v2})")

    # Malom label preload (read-only DB already quick-checked in provenance).
    conn = sqlite3.connect(f"file:{candidate_db}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size = -262144")
    pos_wdl  = {sk: w for sk, w in conn.execute(
        "SELECT state_key, malom_wdl FROM positions WHERE malom_wdl IS NOT NULL"
    )}
    move_wdl = {(sk, nt): w for sk, nt, w in conn.execute(
        "SELECT state_key, notation, malom_wdl_after FROM moves WHERE malom_wdl_after IS NOT NULL"
    )}
    conn.close()
    print(f"[eval] Malom: {len(pos_wdl):,} pos · {len(move_wdl):,} moves")

    # Train state_key set for split integrity check (§6).
    train_idx = ds.train_idx()
    train_state_keys: set[str] = {
        str(ds.state_keys[int(ds.sample_state_idx[i])]) for i in train_idx
    }

    # Session index (§6: fail-closed).
    game_split_mask: Optional[np.ndarray] = None
    if session_index_path is not None:
        game_split_mask = _load_session_index(session_index_path, ds)
        print(f"[eval] Session index loaded ({game_split_mask.shape[0]:,} entries).")

    # Temperature scaling pass (§4).
    if skip_temperature:
        T_star = 1.0
        print("[eval] Skipping temperature scaling (--skip-temperature); T=1.0.")
    else:
        print("[eval] Pass 1: finding optimal temperature (strict logits) …")
        adv_t1 = HumanMovePolicyAdvisor(model_path, temperature=1.0)
        T_star = _find_temperature_strict(dataset_dir, adv_t1, val_idx, ds)

    # Main eval pass (single pass, both T=1 and T* derived from same logits).
    print(f"[eval] Pass 2: full eval with T*={T_star:.4f} and T=1 comparison …")
    adv = HumanMovePolicyAdvisor(model_path, temperature=T_star)

    val_report = _run_eval_loop(
        ds, val_idx, adv, T_star, pos_wdl, move_wdl,
        train_state_keys, game_split_mask, min_support, label="val",
    )

    test_report: Optional[dict] = None
    if run_test_set:
        print(
            "[eval] WARNING: --run-test-set consumes the reserved confirmation slice. "
            "This should only be done once, after the evaluator contract is frozen "
            "and thresholds are set from validation numbers only."
        )
        if len(test_idx) > 0:
            test_report = _run_eval_loop(
                ds, test_idx, adv, T_star, pos_wdl, move_wdl,
                train_state_keys, game_split_mask, min_support, label="test",
            )
        else:
            print("[eval] No test indices found (v1 dataset or empty partition).")

    report = {
        "meta": {
            "dataset_dir":        str(dataset_dir),
            "model":              str(model_path),
            "candidate_db":       str(candidate_db),
            "session_index":      str(session_index_path) if session_index_path else None,
            "dataset_is_v2":      bool(ds._is_v2),
            "min_support_for_kl": int(min_support),
            "temperature_star":   float(T_star),
            "skip_temperature":   bool(skip_temperature),
            "run_test_set":       bool(run_test_set),
            "elapsed_seconds":    round(time.time() - t0, 1),
            "evaluator_version":  "V2",
        },
        "val":  val_report,
        "provenance": prov,
    }
    if test_report is not None:
        report["test"] = test_report
    return report


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(report: dict) -> None:
    T = report["meta"]["temperature_star"]

    def _row(row: dict) -> str:
        nll    = row.get("event_nll", float("nan"))
        brier  = row.get("macro_multiclass_brier", float("nan"))
        top1   = row.get("human_event_top1_coverage", float("nan"))
        top3   = row.get("human_event_top3_coverage", float("nan"))
        top5   = row.get("human_event_top5_coverage", float("nan"))
        ece    = row.get("top_label_ece", float("nan"))
        n_ev   = row.get("n_events", 0)
        return (
            f"n_ev={n_ev:>10,}  nll={nll:.4f}  brier={brier:.4f}  "
            f"top1={top1*100:5.2f}%  top3={top3*100:5.2f}%  top5={top5*100:5.2f}%  "
            f"ece={ece:.3f}"
        )

    def _dc(dc: dict, label: str) -> None:
        mp = dc.get("macro_pred_degrade", float("nan"))
        mo = dc.get("macro_obs_degrade",  float("nan"))
        up = dc.get("micro_pred_degrade", float("nan"))
        uo = dc.get("micro_obs_degrade",  float("nan"))
        me = dc.get("macro_degrade_ece",  float("nan"))
        ue = dc.get("micro_degrade_ece",  float("nan"))
        n  = dc.get("n_samples", 0)
        print(f"  {label} (n={n:,}):")
        print(f"    P(degrade):  macro pred={mp:.4f} obs={mo:.4f} ece={me:.4f}  "
              f"micro pred={up:.4f} obs={uo:.4f} ece={ue:.4f}")
        mps = dc.get("macro_pred_wdl_severity", float("nan"))
        mos = dc.get("macro_obs_wdl_severity",  float("nan"))
        ups = dc.get("micro_pred_wdl_severity", float("nan"))
        uos = dc.get("micro_obs_wdl_severity",  float("nan"))
        print(f"    WDL sev:     macro pred={mps:.4f} obs={mos:.4f}  "
              f"micro pred={ups:.4f} obs={uos:.4f}")

    def _section(label: str, data: dict) -> None:
        m = data["model"]
        print(f"\n{'='*90}")
        print(f"{label} — MODEL (T*={T:.4f})")
        print(f"{'='*90}")
        print("OVERALL   " + _row(m["overall"]))
        print("\nT=1 comparison:")
        print("  OVERALL " + _row(data["t_1_comparison"]["overall"]))
        u = data["baseline_uniform"]
        print(f"\nUNIFORM   " + _row(u["overall"]))
        rc = u.get("expected_random_coverage", {})
        print(f"  expected random coverage: "
              f"top1={rc.get('expected_top1_coverage',0):.3f}  "
              f"top3={rc.get('expected_top3_coverage',0):.3f}  "
              f"top5={rc.get('expected_top5_coverage',0):.3f}")
        print("\nIn-sample empirical reference (descriptive only — not a gate):")
        print("  " + _row(data["in_sample_empirical_reference"]["overall"]))
        print("\nPer Elo band:")
        for b, row in m["by_band"].items():
            print(f"  {b:<8}" + _row(row))
        print("\nPer phase:")
        for p, row in m["by_phase"].items():
            print(f"  {p:<8}" + _row(row))
        print("\nPer legal-move count:")
        for k in ("lmc_2-5", "lmc_6-10", "lmc_11-20", "lmc_21+"):
            row = m["by_lmc"].get(k, {})
            if row: print(f"  {k:<12}" + _row(row))
        print("\nPer Malom transition:")
        for c, row in sorted(m["by_transition"].items()):
            print(f"  {c:<22}" + _row(row))
        if "val_game_exclusive_stratum" in m:
            print("\nVal-game-exclusive stratum:")
            print("  " + _row(m["val_game_exclusive_stratum"]))
        sc = data["skip_counts"]
        print(f"\nSkip counts: attempted={sc['n_attempted_samples']:,}  "
              f"evaluated={sc['n_evaluated_samples']:,}  "
              f"board={sc['n_skipped_board_reconstruction']}  "
              f"lmc={sc['n_skipped_legal_count_mismatch']}  "
              f"zero_tgt={sc['n_skipped_zero_target']}  "
              f"inference={sc['n_skipped_inference_failure']}")
        si = data["split_integrity_check"]
        print(f"Split integrity: pass={si['pass']}  n_val_keys_in_train={si['n_val_keys_in_train']}")
        dc = data["degrade_calibration"]
        print(f"\nDegrade calibration (skip_partial_labels={dc['formal'].get('n_skipped_partial_labels',0):,}):")
        _dc(dc["formal"], "formal (T*)")
        _dc(dc["t_1_comparison"]["formal_degrade_calibration"], "formal (T=1)")
        _dc(dc["diagnostic"], "diagnostic (T*, conditional on labelled)")
        kl = data["observed_distribution_kl_to_model"]
        print(f"\nObserved distribution KL to model (≥{kl['min_support']} events): "
              f"n={kl['n_samples']:,}  mean_kl={kl['mean_kl']:.4f}")

    prov = report["provenance"]
    print(f"\n[provenance] ok={prov['provenance_ok']}  "
          f"git={prov.get('evaluator_git_commit','?')[:12]}  "
          f"dirty={prov.get('evaluator_git_dirty','?')}  "
          f"malom_ver={prov.get('malom_label_version','?')}")
    if prov["compatibility_failures"]:
        print("[provenance] FAILURES:")
        for f in prov["compatibility_failures"]:
            print(f"  - {f}")

    _section("VAL", report["val"])
    if "test" in report:
        _section("TEST (confirmation slice — single use)", report["test"])


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir",    type=Path,
                   default=Path("data/human_move_policy_dataset"))
    p.add_argument("--model",          type=Path,
                   default=Path("data/human_move_policy_net_v2_candidate.npz"))
    p.add_argument("--candidate-db",   type=Path,
                   default=Path("data/human_db_candidate.sqlite"))
    p.add_argument("--session-index",  type=Path, default=None)
    p.add_argument("--min-support",    type=int, default=10)
    p.add_argument("--run-test-set",   action="store_true", default=False,
                   help="Consume the reserved confirmation slice. Run at most once.")
    p.add_argument("--skip-temperature", action="store_true", default=False,
                   help="Use T=1.0 explicitly (no scipy required).")
    p.add_argument("--ignore-provenance-failures", action="store_true", default=False,
                   help="Warn on provenance failures instead of raising.")
    p.add_argument("--output",         type=Path,
                   default=Path("data/gap_v3_prerequisite_eval_V2.json"))
    args = p.parse_args()

    report = evaluate(
        args.dataset_dir, args.model, args.candidate_db,
        session_index_path=args.session_index,
        min_support=args.min_support,
        run_test_set=args.run_test_set,
        skip_temperature=args.skip_temperature,
        ignore_provenance_failures=args.ignore_provenance_failures,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n[eval] Report written → {args.output}")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
