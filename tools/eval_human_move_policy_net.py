#!/usr/bin/env python3
"""tools/eval_human_move_policy_net.py — Phase 4b held-out evaluation for HumanMovePolicyNet.

Reports every metric required by GapNet v3 plan §4.1 / §16.

Primary metrics (every stratum):
    - event-weighted NLL
    - Brier score (multiclass: Σ_i (p_i − y_i)²  where y_i = count_i / Σcount)
    - top-1 / top-3 / top-5 accuracy vs the most-frequent human move
    - 10-bin reliability calibration + ECE

Strata:
    - overall
    - per Elo band (lower / middle / upper)
    - per phase (place / move)
    - per Malom transition category (win_preserved, draw_to_loss, …, unlabelled)
    - per legal-move-count bin (2-5, 6-10, 11-20, 21+)
    - OOD row: positions whose state_key was not in the training partition
    - abstention row: positions where any legal successor encoding fails

Baselines (reported alongside the model in every stratum):
    - uniform:    assigns 1/n_legal to every legal move
    - empirical:  uses HumanDB observed counts (positions with ≥ min_support only)

Temperature scaling:
    Two-pass eval.  Pass 1: collect raw logits via `adv.rank()` on the val
    set, find T* that minimises val NLL via `scipy.optimize.minimize_scalar`.
    Pass 2: reload with temperature=T*, run the full eval loop.

Game / player diagnostic:
    If `--session-index` is provided, also reports NLL + Brier for the
    game-val-only stratum (positions not reached by any game-train game).

Test set:
    By default only the val split is evaluated.  Pass `--run-test-set` to
    also evaluate the test partition (v2 datasets only).  Test-set results
    are saved under a separate key in the report so they cannot be
    confused with val numbers.

Usage
-----
    # Standard val eval (temperature scaling + all strata):
    .venv/bin/python tools/eval_human_move_policy_net.py \\
        --dataset-dir data/human_move_policy_dataset \\
        --model       data/human_move_policy_net_v2_candidate.npz \\
        --candidate-db data/human_db_candidate.sqlite \\
        --output      data/gap_v3_prerequisite_eval.json

    # Also run the untouched test set:
    .venv/bin/python tools/eval_human_move_policy_net.py \\
        --dataset-dir data/human_move_policy_dataset \\
        --model       data/human_move_policy_net_v2_candidate.npz \\
        --candidate-db data/human_db_candidate.sqlite \\
        --session-index data/human_move_policy_session_index.npz \\
        --run-test-set \\
        --output      data/gap_v3_prerequisite_eval.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ai.human_move_policy_advisor import HumanMovePolicyAdvisor   # noqa: E402
from ai.value_net import _INPUT_DIM, board_to_features            # noqa: E402
from game.board import BoardState                                  # noqa: E402
from game.rules import get_all_legal_moves                        # noqa: E402
from tools.train_value_net_v2 import board_from_state_key         # noqa: E402

sys.path.insert(0, str(_ROOT / "tools"))
import audit_human_moves as ahb   # noqa: E402


_BAND_NAMES = ("lower", "middle", "upper")

_DEGRADE_CATS = frozenset({"win_to_draw", "win_to_loss", "draw_to_loss"})
_REGRET_WEIGHTS: dict[str, float] = {
    "win_to_loss": 1.0,   # worst blunder
    "win_to_draw": 0.5,
    "draw_to_loss": 0.5,
}


# ── Move notation (matches extractor convention) ────────────────────────────

def _move_notation(mv: dict) -> str:
    frm = mv.get("from")
    to  = mv.get("to") or ""
    cap = mv.get("capture") or ""
    base = f"{frm}-{to}" if frm else to
    return f"{base}x{cap}" if cap else base


# ── Metrics accumulator ─────────────────────────────────────────────────────

class Stratum:
    """Accumulate stats for a slice of the eval set."""
    def __init__(self, name: str):
        self.name       = name
        self.n_events   = 0
        self.n_samples  = 0
        self.nll_sum    = 0.0
        self.brier_sum  = 0.0   # multiclass Brier: Σ_i (p_i − y_i)²
        self.top1       = 0
        self.top3       = 0
        self.top5       = 0
        # 10-bin reliability calibration.
        self.calib_bin_n     = np.zeros(10, dtype=np.int64)
        self.calib_bin_probs = np.zeros(10, dtype=np.float64)
        self.calib_bin_hits  = np.zeros(10, dtype=np.float64)

    def add_sample(
        self,
        probs: np.ndarray,       # (n_legal,) softmax distribution
        targets: np.ndarray,     # (n_legal,) int32 observed counts
        notations: list[str],
        top1_notation: str,
        top3_notations: set[str],
        top5_notations: set[str],
    ) -> None:
        self.n_samples += 1
        total_events = int(targets.sum())
        if total_events == 0:
            return
        # Multiclass Brier: treat empirical freq as target for this sample.
        y = targets.astype(np.float64) / float(total_events)
        p = probs.astype(np.float64)
        self.brier_sum += float(np.sum((p - y) ** 2))

        # Top-label confidence: model's probability for its own top-1 prediction.
        # All events from this position share the same top_conf, so they land
        # in the same ECE bin — this is standard top-label multiclass ECE.
        top_conf = float(probs.max())
        for j, count in enumerate(targets):
            if count == 0:
                continue
            notation   = notations[j]
            prob_j     = float(probs[j])
            was_top1   = (notation == top1_notation)
            was_top3   = (notation in top3_notations)
            was_top5   = (notation in top5_notations)
            for _ in range(int(count)):
                self.n_events += 1
                self.nll_sum  += -np.log(max(prob_j, 1e-9))
                self.top1     += int(was_top1)
                self.top3     += int(was_top3)
                self.top5     += int(was_top5)
                bin_idx = min(9, max(0, int(top_conf * 10)))
                self.calib_bin_n[bin_idx]     += 1
                self.calib_bin_probs[bin_idx] += top_conf
                self.calib_bin_hits[bin_idx]  += int(was_top1)

    def finalize(self) -> dict:
        n = max(self.n_events, 1)
        ns = max(self.n_samples, 1)
        ece = 0.0
        for i in range(10):
            if self.calib_bin_n[i] == 0:
                continue
            conf = self.calib_bin_probs[i] / self.calib_bin_n[i]
            acc  = self.calib_bin_hits[i]  / self.calib_bin_n[i]
            ece += (self.calib_bin_n[i] / n) * abs(conf - acc)
        return {
            "name":       self.name,
            "n_samples":  int(self.n_samples),
            "n_events":   int(self.n_events),
            "event_nll":  float(self.nll_sum / n),
            "brier":      float(self.brier_sum / ns),
            "top1":       float(self.top1 / n),
            "top3":       float(self.top3 / n),
            "top5":       float(self.top5 / n),
            "ece":        float(ece),
        }


# ── Degrading-move calibration ──────────────────────────────────────────────

class DegradeCal:
    """Per-position calibration of P(degrade) and E[regret] vs observed frequency.

    For each position with Malom labels we compute:
      pred_degrade = Σ_j probs[j]  for j ∈ {W→D, W→L, D→L}
      pred_regret  = Σ_j probs[j] * w_j   w: W→L=1, W→D=0.5, D→L=0.5
      obs_degrade  = count of observed degrading events / total events
      obs_regret   = Σ_j count[j]*w_j / total events

    Both are binned into 10 equal-width bins on [0, 1] and compared
    per-bin (ECE-style: weighted mean abs difference).
    """
    N_BINS = 10

    def __init__(self) -> None:
        self.n_pos              = 0
        self.total_pred_degrade = 0.0
        self.total_obs_degrade  = 0
        self.total_events       = 0
        self.total_pred_regret  = 0.0
        self.total_obs_regret   = 0.0
        self.pd_bin_n    = np.zeros(self.N_BINS, dtype=np.int64)
        self.pd_bin_pred = np.zeros(self.N_BINS, dtype=np.float64)
        self.pd_bin_obs  = np.zeros(self.N_BINS, dtype=np.float64)
        self.er_bin_n    = np.zeros(self.N_BINS, dtype=np.int64)
        self.er_bin_pred = np.zeros(self.N_BINS, dtype=np.float64)
        self.er_bin_obs  = np.zeros(self.N_BINS, dtype=np.float64)

    def add_position(
        self,
        prob_degrade: float,
        pred_regret: float,
        obs_degrade: int,
        obs_regret: float,
        obs_total: int,
    ) -> None:
        self.n_pos              += 1
        self.total_pred_degrade += prob_degrade
        self.total_obs_degrade  += obs_degrade
        self.total_events       += obs_total
        self.total_pred_regret  += pred_regret
        self.total_obs_regret   += obs_regret

        obs_freq    = obs_degrade / max(obs_total, 1)
        obs_reg_avg = obs_regret  / max(obs_total, 1)

        pd_idx = min(self.N_BINS - 1, int(prob_degrade * self.N_BINS))
        self.pd_bin_n[pd_idx]    += 1
        self.pd_bin_pred[pd_idx] += prob_degrade
        self.pd_bin_obs[pd_idx]  += obs_freq

        er_idx = min(self.N_BINS - 1, int(pred_regret * self.N_BINS))
        self.er_bin_n[er_idx]    += 1
        self.er_bin_pred[er_idx] += pred_regret
        self.er_bin_obs[er_idx]  += obs_reg_avg

    def finalize(self) -> dict:
        n = max(self.n_pos, 1)

        def _ece(bin_n, bin_pred, bin_obs) -> float:
            ece = 0.0
            for i in range(self.N_BINS):
                if bin_n[i] == 0:
                    continue
                avg_p = bin_pred[i] / bin_n[i]
                avg_o = bin_obs[i]  / bin_n[i]
                ece  += (bin_n[i] / n) * abs(avg_p - avg_o)
            return float(ece)

        return {
            "n_positions_labelled":   int(self.n_pos),
            "mean_pred_prob_degrade": float(self.total_pred_degrade / n),
            "obs_degrade_freq":       float(self.total_obs_degrade / max(self.total_events, 1)),
            "degrade_ece":            _ece(self.pd_bin_n, self.pd_bin_pred, self.pd_bin_obs),
            "mean_pred_regret":       float(self.total_pred_regret / n),
            "obs_regret":             float(self.total_obs_regret  / max(self.total_events, 1)),
            "regret_ece":             _ece(self.er_bin_n, self.er_bin_pred, self.er_bin_obs),
        }


# ── Baseline distributions ──────────────────────────────────────────────────

def _uniform_probs(n: int) -> np.ndarray:
    return np.full(n, 1.0 / max(n, 1), dtype=np.float32)


def _empirical_probs(targets: np.ndarray) -> Optional[np.ndarray]:
    """Return empirical distribution or None if sum is zero."""
    total = float(targets.sum())
    if total <= 0:
        return None
    return targets.astype(np.float32) / total


# ── Temperature scaling (pass 1) ────────────────────────────────────────────

def _find_temperature(
    dataset_dir: Path,
    model_path: Path,
    val_idx: np.ndarray,
) -> float:
    """Minimise val NLL w.r.t. temperature T using raw logits.

    Returns the optimal T* in (0.1, 10.0).
    """
    try:
        from scipy.optimize import minimize_scalar
    except ImportError:
        print("[eval] scipy not available — skipping temperature scaling, T=1.0")
        return 1.0

    adv = HumanMovePolicyAdvisor(model_path, temperature=1.0)
    from tools.train_human_move_policy_net import MovePolicyDataset
    ds = MovePolicyDataset(dataset_dir)

    # Collect (logit_vectors, target_vectors) on the val set.
    logit_list:  list[np.ndarray] = []
    target_list: list[np.ndarray] = []

    for sid in val_idx:
        state_idx = int(ds.sample_state_idx[sid])
        band_idx  = int(ds.sample_band_idx[sid])
        band_name = _BAND_NAMES[band_idx]
        state_key = str(ds.state_keys[state_idx])
        legal_count = int(ds.state_succ_offsets[state_idx + 1] - ds.state_succ_offsets[state_idx])

        board = board_from_state_key(state_key)
        if board is None:
            continue
        legal = get_all_legal_moves(board)
        if len(legal) != legal_count:
            continue

        tgt_start = int(ds.sample_offsets[sid])
        tgt_end   = int(ds.sample_offsets[sid + 1])
        targets   = ds.sample_targets[tgt_start:tgt_end].astype(np.int64)
        if targets.sum() == 0:
            continue

        logits = np.array(adv.rank(board, legal, band_name), dtype=np.float64)
        logit_list.append(logits)
        target_list.append(targets)

    if not logit_list:
        return 1.0

    def _val_nll(T: float) -> float:
        T = max(T, 1e-3)
        nll = 0.0
        events = 0
        for logits, targets in zip(logit_list, target_list):
            scaled = logits / T
            scaled -= scaled.max()
            log_p = scaled - np.log(np.exp(scaled).sum())
            for j, cnt in enumerate(targets):
                if cnt > 0:
                    nll += -float(cnt) * log_p[j]
                    events += int(cnt)
        return nll / max(events, 1)

    result = minimize_scalar(_val_nll, bounds=(0.1, 10.0), method="bounded")
    T_star = float(result.x)
    print(f"[eval] Temperature scaling: T*={T_star:.4f}  "
          f"val_nll(T=1)={_val_nll(1.0):.5f}  val_nll(T*)={result.fun:.5f}")
    return T_star


# ── Abstention check ─────────────────────────────────────────────────────────

def _has_encoding_failure(board: BoardState, legal: list[dict], mover: str) -> bool:
    """Return True if any legal successor fails board_to_features encoding."""
    for mv in legal:
        try:
            succ = board.apply_move(mv)
            board_to_features(succ, mover)
        except Exception:
            return True
    return False


# ── Legal-move-count stratum assignment ─────────────────────────────────────

def _lmc_bin(n: int) -> str:
    if n <= 5:
        return "lmc_2-5"
    if n <= 10:
        return "lmc_6-10"
    if n <= 20:
        return "lmc_11-20"
    return "lmc_21+"


# ── Core eval loop ──────────────────────────────────────────────────────────

def _run_eval_loop(
    ds,
    sample_idx: np.ndarray,
    adv: HumanMovePolicyAdvisor,
    pos_wdl: dict,
    move_wdl: dict,
    train_state_keys: Optional[set],
    game_split_mask: Optional[np.ndarray],
    min_support: int,
    *,
    label: str = "val",
) -> dict:
    """Run the full eval loop over `sample_idx`.

    Returns a dict with keys matching the report schema.
    """
    t0 = time.time()

    # Main model strata.
    overall  = Stratum("overall")
    by_band  = {b: Stratum(f"band={b}")  for b in _BAND_NAMES}
    by_phase = {p: Stratum(f"phase={p}") for p in ("place", "move")}
    by_trans: dict[str, Stratum] = {}
    for c in ("win_preserved", "win_to_draw", "win_to_loss",
              "draw_preserved", "draw_to_loss", "all_losing",
              "label_inconsistency", "unlabelled"):
        by_trans[c] = Stratum(f"trans={c}")
    by_lmc: dict[str, Stratum] = {k: Stratum(k)
                                   for k in ("lmc_2-5", "lmc_6-10", "lmc_11-20", "lmc_21+")}
    ood_stratum       = Stratum("ood")
    abstention_stratum = Stratum("abstention")
    game_val_only_stratum = Stratum("game_val_only") if game_split_mask is not None else None

    # Baseline (uniform) strata — parallel accumulators.
    uni_overall = Stratum("uniform_overall")
    uni_by_band = {b: Stratum(f"uniform_band={b}") for b in _BAND_NAMES}

    # Empirical baseline strata.
    emp_overall = Stratum("empirical_overall")
    emp_by_band = {b: Stratum(f"empirical_band={b}") for b in _BAND_NAMES}

    # KL (empirical || model) where support ≥ min_support.
    kl_sum = 0.0
    kl_n   = 0

    # Degrading-move calibration.
    degrade_cal = DegradeCal()

    MASK_TRAIN = np.uint8(0x01)
    MASK_VAL   = np.uint8(0x02)

    for i, sid in enumerate(sample_idx):
        state_idx   = int(ds.sample_state_idx[sid])
        band_idx    = int(ds.sample_band_idx[sid])
        band_name   = _BAND_NAMES[band_idx]
        state_key   = str(ds.state_keys[state_idx])
        mover       = str(ds.mover_colors[state_idx])
        legal_count = int(ds.state_succ_offsets[state_idx + 1] - ds.state_succ_offsets[state_idx])

        board = board_from_state_key(state_key)
        if board is None:
            continue
        legal = get_all_legal_moves(board)
        if len(legal) != legal_count:
            continue
        notations = [_move_notation(m) for m in legal]

        tgt_start = int(ds.sample_offsets[sid])
        tgt_end   = int(ds.sample_offsets[sid + 1])
        targets   = ds.sample_targets[tgt_start:tgt_end].astype(np.int64)
        total_events = int(targets.sum())
        if total_events == 0:
            continue

        # Phase.
        try:
            phase_str = "place" if board.phase == "place" else "move"
        except Exception:
            phase_str = "move"

        # OOD flag.
        is_ood = (train_state_keys is not None) and (state_key not in train_state_keys)

        # Abstention flag.
        has_fail = _has_encoding_failure(board, legal, mover)

        # Model distribution.
        probs = adv.probs(board, legal, band_name)
        top_order       = np.argsort(-probs)
        top1_notation   = notations[top_order[0]]
        top3_notations  = {notations[j] for j in top_order[:3]}
        top5_notations  = {notations[j] for j in top_order[:5]}

        # ── Model strata ──
        def _add(stratum: Stratum) -> None:
            stratum.add_sample(probs, targets, notations,
                               top1_notation, top3_notations, top5_notations)

        _add(overall)
        _add(by_band[band_name])
        _add(by_phase[phase_str])
        _add(by_lmc[_lmc_bin(legal_count)])
        if is_ood:
            _add(ood_stratum)
        if has_fail:
            _add(abstention_stratum)

        # Game-val-only stratum.
        if game_val_only_stratum is not None and game_split_mask is not None:
            mask = game_split_mask[state_idx]
            if (mask & MASK_TRAIN) == 0 and (mask & MASK_VAL) != 0:
                _add(game_val_only_stratum)

        # Malom transition strata + per-position degrading calibration.
        # Iterates ALL legal moves (not just observed) so prob_degrade
        # sums over the full distribution, not just observed moves.
        pre          = pos_wdl.get(state_key)
        prob_degrade = 0.0
        pred_regret  = 0.0
        obs_degrade  = 0
        obs_regret   = 0.0
        n_classified = 0

        for j, count in enumerate(targets):
            notation   = notations[j]
            aft        = move_wdl.get((state_key, notation))
            cat        = ahb._classify_transition(pre, aft)
            is_degrade = cat in _DEGRADE_CATS
            w          = _REGRET_WEIGHTS.get(cat, 0.0)

            if cat != "unlabelled":
                n_classified += 1
                prob_degrade += float(probs[j]) * (1.0 if is_degrade else 0.0)
                pred_regret  += float(probs[j]) * w

            if count > 0:
                if cat != "unlabelled":
                    obs_degrade += int(is_degrade) * int(count)
                    obs_regret  += w * int(count)
                st = by_trans.setdefault(cat, Stratum(f"trans={cat}"))
                singleton_targets = np.zeros(legal_count, dtype=np.int64)
                singleton_targets[j] = count
                st.add_sample(probs, singleton_targets, notations,
                              top1_notation, top3_notations, top5_notations)

        if n_classified > 0 and total_events > 0:
            degrade_cal.add_position(
                prob_degrade, pred_regret, obs_degrade, obs_regret, total_events
            )

        # ── Uniform baseline ──
        uni = _uniform_probs(legal_count)
        uni_t3 = {notations[j] for j in np.argsort(-uni)[:3]}
        uni_t5 = {notations[j] for j in np.argsort(-uni)[:5]}
        # For uniform, top-1 is ambiguous — take the first notation.
        uni_t1 = notations[0]
        uni_overall.add_sample(uni, targets, notations, uni_t1, uni_t3, uni_t5)
        uni_by_band[band_name].add_sample(uni, targets, notations, uni_t1, uni_t3, uni_t5)

        # ── Empirical baseline (min_support gate) ──
        if total_events >= min_support:
            emp = _empirical_probs(targets)
            if emp is not None:
                emp_t_ord = np.argsort(-emp)
                emp_t1    = notations[emp_t_ord[0]]
                emp_t3    = {notations[j] for j in emp_t_ord[:3]}
                emp_t5    = {notations[j] for j in emp_t_ord[:5]}
                emp_overall.add_sample(emp, targets, notations, emp_t1, emp_t3, emp_t5)
                emp_by_band[band_name].add_sample(emp, targets, notations, emp_t1, emp_t3, emp_t5)

            # KL (empirical || model)
            emp_f = targets.astype(np.float64) / float(total_events)
            eps = 1e-9
            kl_sum += float(np.where(
                emp_f > 0.0,
                emp_f * (np.log(emp_f + eps) - np.log(probs.astype(np.float64) + eps)),
                0.0,
            ).sum())
            kl_n += 1

        if (i + 1) % 5000 == 0 or (i + 1) == len(sample_idx):
            print(f"[eval/{label}] {i + 1}/{len(sample_idx)}  "
                  f"({(time.time()-t0):.1f}s)")

    result = {
        "n_samples_evaluated": int(len(sample_idx)),
        "elapsed_seconds": round(time.time() - t0, 1),
        "model": {
            "overall":          overall.finalize(),
            "by_band":          {b: s.finalize() for b, s in by_band.items()},
            "by_phase":         {p: s.finalize() for p, s in by_phase.items()},
            "by_transition":    {c: s.finalize() for c, s in by_trans.items()
                                 if s.n_events > 0},
            "by_lmc":           {k: s.finalize() for k, s in by_lmc.items()},
            "ood":              ood_stratum.finalize(),
            "abstention":       abstention_stratum.finalize(),
        },
        "degrade_calibration":  degrade_cal.finalize(),
        "baseline_uniform": {
            "overall":  uni_overall.finalize(),
            "by_band":  {b: s.finalize() for b, s in uni_by_band.items()},
        },
        "baseline_empirical": {
            "min_support":  int(min_support),
            "overall":      emp_overall.finalize(),
            "by_band":      {b: s.finalize() for b, s in emp_by_band.items()},
        },
        "empirical_kl_supported": {
            "min_support":  int(min_support),
            "n_samples":    int(kl_n),
            "mean_kl":      float(kl_sum / max(kl_n, 1)),
        },
    }
    if game_val_only_stratum is not None:
        result["model"]["game_val_only"] = game_val_only_stratum.finalize()
    return result


# ── Main evaluate() ─────────────────────────────────────────────────────────

def evaluate(
    dataset_dir: Path,
    model_path: Path,
    candidate_db: Path,
    session_index_path: Optional[Path] = None,
    min_support: int = 10,
    run_test_set: bool = False,
    skip_temperature: bool = False,
) -> dict:
    from tools.train_human_move_policy_net import MovePolicyDataset

    t0 = time.time()
    ds = MovePolicyDataset(dataset_dir)
    val_idx  = ds.val_idx()
    test_idx = ds.test_idx() if run_test_set else np.array([], dtype=np.int64)
    print(f"[eval] val={len(val_idx):,}  test={len(test_idx):,}  (v2={ds._is_v2})")

    # Preload Malom labels.
    conn = sqlite3.connect(str(candidate_db))
    pos_wdl  = {sk: w for sk, w in conn.execute(
        "SELECT state_key, malom_wdl FROM positions WHERE malom_wdl IS NOT NULL"
    )}
    move_wdl = {(sk, nt): w for sk, nt, w in conn.execute(
        "SELECT state_key, notation, malom_wdl_after FROM moves WHERE malom_wdl_after IS NOT NULL"
    )}
    conn.close()
    print(f"[eval] Malom preload: {len(pos_wdl):,} pos · {len(move_wdl):,} moves")

    # OOD: collect the set of state_keys in the train partition.
    train_idx = ds.train_idx()
    train_state_keys: set[str] = {
        str(ds.state_keys[int(ds.sample_state_idx[i])]) for i in train_idx
    }
    print(f"[eval] Train state_key set: {len(train_state_keys):,}")

    # Session index (optional).
    game_split_mask: Optional[np.ndarray] = None
    if session_index_path is not None and session_index_path.exists():
        si = np.load(str(session_index_path), allow_pickle=True)
        game_split_mask = si["game_split_mask"].astype(np.uint8)
        si_prov = json.loads(str(si["provenance"].item()))
        print(f"[eval] Session index loaded: {session_index_path}  "
              f"n_covered={si_prov.get('n_state_keys_covered', '?')}")
    elif session_index_path is not None:
        print(f"[eval] WARNING: session index not found at {session_index_path} — skipping")

    # ── Temperature scaling pass ──────────────────────────────────────────────
    if skip_temperature:
        T_star = 1.0
    else:
        print("[eval] Pass 1: finding optimal temperature …")
        T_star = _find_temperature(dataset_dir, model_path, val_idx)

    # ── Main eval pass ────────────────────────────────────────────────────────
    print(f"[eval] Pass 2: full eval with T={T_star:.4f} …")
    adv = HumanMovePolicyAdvisor(model_path, temperature=T_star)

    val_report = _run_eval_loop(
        ds, val_idx, adv, pos_wdl, move_wdl,
        train_state_keys, game_split_mask, min_support,
        label="val",
    )

    test_report: Optional[dict] = None
    if run_test_set and len(test_idx) > 0:
        print("[eval] Running test set (single-shot — do not repeat) …")
        test_report = _run_eval_loop(
            ds, test_idx, adv, pos_wdl, move_wdl,
            train_state_keys, game_split_mask, min_support,
            label="test",
        )
    elif run_test_set:
        print("[eval] WARNING: --run-test-set requested but no test indices found "
              "(v1 dataset or empty test partition).")

    report = {
        "meta": {
            "dataset_dir":          str(dataset_dir),
            "model":                str(model_path),
            "candidate_db":         str(candidate_db),
            "session_index":        str(session_index_path) if session_index_path else None,
            "dataset_is_v2":        bool(ds._is_v2),
            "min_support_for_kl":   int(min_support),
            "temperature_star":     float(T_star),
            "skip_temperature":     bool(skip_temperature),
            "run_test_set":         bool(run_test_set),
            "elapsed_seconds":      round(time.time() - t0, 1),
        },
        "val":  val_report,
        "provenance": adv.provenance,
    }
    if test_report is not None:
        report["test"] = test_report
    return report


# ── Summary printer ─────────────────────────────────────────────────────────

def _print_summary(report: dict) -> None:
    def _row(row: dict) -> str:
        return (f"n_ev={row['n_events']:>10,}  "
                f"nll={row['event_nll']:.4f}  "
                f"brier={row['brier']:.4f}  "
                f"top1={row['top1']*100:5.2f}%  "
                f"top3={row['top3']*100:5.2f}%  "
                f"top5={row['top5']*100:5.2f}%  "
                f"ece={row['ece']:.3f}")

    def _section(label: str, data: dict) -> None:
        m = data["model"]
        print(f"\n{'='*90}")
        print(f"{label} — MODEL (T={report['meta']['temperature_star']:.4f})")
        print(f"{'='*90}")
        print("OVERALL   " + _row(m["overall"]))
        u = data["baseline_uniform"]
        print(f"UNIFORM   " + _row(u["overall"]))
        e = data["baseline_empirical"]
        print(f"EMPIRICAL (≥{e['min_support']}) " + _row(e["overall"]))
        print()
        print("Per Elo band:")
        for b, row in m["by_band"].items():
            print(f"  {b:<8}" + _row(row))
        print()
        print("Per phase:")
        for p, row in m["by_phase"].items():
            print(f"  {p:<8}" + _row(row))
        print()
        print("Per legal-move count:")
        for k in ("lmc_2-5", "lmc_6-10", "lmc_11-20", "lmc_21+"):
            row = m["by_lmc"].get(k, {})
            if row:
                print(f"  {k:<12}" + _row(row))
        print()
        print("Per Malom transition:")
        for c, row in sorted(m["by_transition"].items()):
            print(f"  {c:<22}" + _row(row))
        print()
        print(f"OOD       " + _row(m["ood"]))
        print(f"ABSTAIN   " + _row(m["abstention"]))
        if "game_val_only" in m:
            print(f"GAME-VAL-ONLY " + _row(m["game_val_only"]))
        ek = data["empirical_kl_supported"]
        print(f"\nEmpirical KL (≥{ek['min_support']} events): "
              f"n={ek['n_samples']:,}  mean_kl={ek['mean_kl']:.4f}")
        dc = data["degrade_calibration"]
        print(f"\nDegrading-move calibration "
              f"(n_pos={dc['n_positions_labelled']:,}):")
        print(f"  P(degrade):  pred={dc['mean_pred_prob_degrade']:.4f}  "
              f"obs={dc['obs_degrade_freq']:.4f}  "
              f"ece={dc['degrade_ece']:.4f}")
        print(f"  E[regret]:   pred={dc['mean_pred_regret']:.4f}  "
              f"obs={dc['obs_regret']:.4f}  "
              f"ece={dc['regret_ece']:.4f}")

    _section("VAL", report["val"])
    if "test" in report:
        _section("TEST (untouched)", report["test"])


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir",    type=Path,
                   default=Path("data/human_move_policy_dataset"))
    p.add_argument("--model",          type=Path,
                   default=Path("data/human_move_policy_net_v2_candidate.npz"))
    p.add_argument("--candidate-db",   type=Path,
                   default=Path("data/human_db_candidate.sqlite"))
    p.add_argument("--session-index",  type=Path, default=None,
                   help="Path to session index .npz (from build_session_index.py).  "
                        "Enables game-val-only diagnostic stratum.")
    p.add_argument("--min-support",    type=int, default=10,
                   help="Min observed events for empirical baseline + KL metric.")
    p.add_argument("--run-test-set",   action="store_true", default=False,
                   help="Also evaluate the untouched test partition (v2 datasets only).  "
                        "Do not pass this flag during model selection — test leaks.")
    p.add_argument("--skip-temperature", action="store_true", default=False,
                   help="Skip temperature scaling pass (use T=1.0).  Faster smoke runs.")
    p.add_argument("--output",         type=Path,
                   default=Path("data/human_move_policy_eval.json"))
    args = p.parse_args()

    report = evaluate(
        args.dataset_dir, args.model, args.candidate_db,
        session_index_path=args.session_index,
        min_support=args.min_support,
        run_test_set=args.run_test_set,
        skip_temperature=args.skip_temperature,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n[eval] Report written → {args.output}")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
