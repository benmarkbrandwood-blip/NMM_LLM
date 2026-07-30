#!/usr/bin/env python3
"""tools/eval_human_blunder_net.py — Held-out evaluation for HumanBlunderNet.

Reports every metric flagged by the plan doc §Phase 4:

    - primary: event-weighted negative log-likelihood
    - top-1 / top-3 / top-5 accuracy vs the most-frequent human move
    - probability calibration (reliability bins + ECE)
    - per-Elo-band, per-phase strata
    - per-Malom-transition-category stratum (win_preserved,
      draw_to_loss, win_to_loss, all_losing, …) — reuses the
      audit's `_classify_transition`
    - comparison against empirical HumanDB frequencies for positions
      with ≥ min_support plays per band (per plan)

Only the held-out slice is reported (`sample_is_val == True`).  The
game-held-out diagnostic is deferred to a future revision — this
script uses the position-level split that HumanPrefNet + ValueNet
already share.

Usage
-----
    .venv/bin/python tools/eval_human_blunder_net.py \\
        --dataset-dir data/human_blunder_dataset \\
        --model       data/human_blunder_net.npz \\
        --candidate-db data/human_db_candidate.sqlite \\
        --output      data/human_blunder_eval.json
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

from ai.human_blunder_advisor import HumanBlunderAdvisor        # noqa: E402
from ai.value_net import _INPUT_DIM, board_to_features          # noqa: E402
from game.board import BoardState                               # noqa: E402
from game.rules import get_all_legal_moves                      # noqa: E402
from tools.train_value_net_v2 import board_from_state_key       # noqa: E402


# Must mirror tools/train_human_blunder_net.py and the extractor.
_BAND_NAMES = ("lower", "middle", "upper")


# ── Reuse the audit's transition classifier ─────────────────────────────────

sys.path.insert(0, str(_ROOT / "tools"))
import audit_human_blunders as ahb   # noqa: E402


# ── Move-notation helper (matches build_human_db_sha.py convention) ─────────

def _move_notation(mv: dict) -> str:
    frm = mv.get("from")
    to  = mv.get("to") or ""
    cap = mv.get("capture") or ""
    base = f"{frm}-{to}" if frm else to
    return f"{base}x{cap}" if cap else base


# ── Metrics accumulators ────────────────────────────────────────────────────

class Stratum:
    """Accumulate stats for a slice of the val set."""
    def __init__(self, name: str):
        self.name = name
        self.n_events = 0
        self.n_samples = 0
        self.nll_sum = 0.0
        self.top1 = 0
        self.top3 = 0
        self.top5 = 0
        # Reliability bins: 10 fixed bins over [0,1].
        self.calib_bin_n     = np.zeros(10, dtype=np.int64)
        self.calib_bin_probs = np.zeros(10, dtype=np.float64)
        self.calib_bin_hits  = np.zeros(10, dtype=np.float64)

    def add_event(self, prob_played: float, was_top1: bool,
                  was_top3: bool, was_top5: bool):
        self.n_events += 1
        # Guard against log(0).  Cap prob at 1e-9 for stable NLL.
        self.nll_sum += -np.log(max(prob_played, 1e-9))
        self.top1 += int(was_top1)
        self.top3 += int(was_top3)
        self.top5 += int(was_top5)
        bin_idx = min(9, max(0, int(prob_played * 10)))
        self.calib_bin_n[bin_idx]     += 1
        self.calib_bin_probs[bin_idx] += prob_played
        self.calib_bin_hits[bin_idx]  += int(was_top1)

    def finalize(self) -> dict:
        n = max(self.n_events, 1)
        # ECE = Σ |confidence − accuracy| weighted by bin count.
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
            "top1":       float(self.top1 / n),
            "top3":       float(self.top3 / n),
            "top5":       float(self.top5 / n),
            "ece":        float(ece),
        }


# ── Main eval loop ──────────────────────────────────────────────────────────

def evaluate(
    dataset_dir: Path,
    model_path: Path,
    candidate_db: Path,
    min_support: int = 10,
) -> dict:
    from tools.train_human_blunder_net import BlunderDataset

    t0 = time.time()
    ds = BlunderDataset(dataset_dir)
    val_idx = ds.val_idx()
    print(f"[eval] {len(val_idx):,} val samples")

    adv = HumanBlunderAdvisor(model_path, temperature=1.0)

    # Preload Malom labels for transition classification.
    conn = sqlite3.connect(str(candidate_db))
    pos_wdl   = {sk: w for sk, w in conn.execute(
        "SELECT state_key, malom_wdl FROM positions WHERE malom_wdl IS NOT NULL"
    )}
    move_wdl  = {(sk, nt): w for sk, nt, w in conn.execute(
        "SELECT state_key, notation, malom_wdl_after FROM moves WHERE malom_wdl_after IS NOT NULL"
    )}
    conn.close()
    print(f"[eval] Malom preload: {len(pos_wdl):,} pos · {len(move_wdl):,} moves")

    # Strata
    overall = Stratum("overall")
    by_band  = {b: Stratum(f"band={b}")   for b in _BAND_NAMES}
    by_phase = {p: Stratum(f"phase={p}")  for p in ("place", "move")}
    by_trans = defaultdict(lambda name=None: Stratum(f"trans={name}"))
    # Pre-materialise expected transition categories so the dict is stable
    for c in ("win_preserved", "win_to_draw", "win_to_loss",
              "draw_preserved", "draw_to_loss", "all_losing",
              "label_inconsistency", "unlabelled"):
        by_trans[c] = Stratum(f"trans={c}")

    # Support-thresholded empirical KL: for samples where the observed
    # count is ≥ min_support, compare the model's distribution to the
    # empirical one.  Report mean KL over these.
    kl_sum = 0.0
    kl_n   = 0

    # Iterate held-out samples.
    for i, sid in enumerate(val_idx):
        state_idx = int(ds.sample_state_idx[sid])
        band_idx  = int(ds.sample_band_idx[sid])
        band_name = _BAND_NAMES[band_idx]
        state_key = str(ds.state_keys[state_idx])
        mover     = str(ds.mover_colors[state_idx])
        legal_count = int(ds.state_succ_offsets[state_idx + 1] - ds.state_succ_offsets[state_idx])

        board = board_from_state_key(state_key)
        if board is None:
            continue
        legal = get_all_legal_moves(board)
        if len(legal) != legal_count:
            # Should not happen (extractor cached the count) — safety.
            continue
        notations = [_move_notation(m) for m in legal]

        # Predicted distribution
        probs = adv.probs(board, legal, band_name)
        pred_top1_idx = int(np.argmax(probs))
        top_order = np.argsort(-probs)
        top1_notation = notations[top_order[0]]
        top3_notations = {notations[i] for i in top_order[:3]}
        top5_notations = {notations[i] for i in top_order[:5]}

        # Target counts
        tgt_start = int(ds.sample_offsets[sid])
        tgt_end   = int(ds.sample_offsets[sid + 1])
        targets   = ds.sample_targets[tgt_start:tgt_end].astype(np.int64)
        total_events = int(targets.sum())
        if total_events == 0:
            continue

        # Phase strata: use board.phase at parent position.
        try:
            phase = board.phase if board.phase in ("place", "move") else "move"
        except Exception:
            phase = "move"
        phase_str = "place" if phase == "place" else "move"

        overall.n_samples += 1
        by_band[band_name].n_samples += 1
        by_phase[phase_str].n_samples += 1

        # Iterate each observed move as its own event (count-weighted).
        for j, count in enumerate(targets):
            if count == 0:
                continue
            notation = notations[j]
            prob_played = float(probs[j])
            was_top1 = (notation == top1_notation)
            was_top3 = (notation in top3_notations)
            was_top5 = (notation in top5_notations)

            for _ in range(int(count)):
                overall.add_event(prob_played, was_top1, was_top3, was_top5)
                by_band[band_name].add_event(prob_played, was_top1, was_top3, was_top5)
                by_phase[phase_str].add_event(prob_played, was_top1, was_top3, was_top5)

            # Malom-transition category for this observed move.
            pre   = pos_wdl.get(state_key)
            after = move_wdl.get((state_key, notation))
            cat   = ahb._classify_transition(pre, after)
            stratum = by_trans[cat]
            for _ in range(int(count)):
                stratum.add_event(prob_played, was_top1, was_top3, was_top5)
            stratum.n_samples += 1

        # Empirical KL where support is enough.
        if total_events >= min_support:
            emp = targets.astype(np.float64) / float(total_events)
            # KL(empirical || predicted): Σ p_i log(p_i / q_i) over p_i>0
            eps = 1e-9
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = np.where(
                    emp > 0.0,
                    emp * (np.log(emp + eps) - np.log(probs + eps)),
                    0.0,
                )
            kl_sum += float(terms.sum())
            kl_n   += 1

        if (i + 1) % 5000 == 0 or (i + 1) == len(val_idx):
            print(f"[eval] processed {i + 1}/{len(val_idx)} val samples "
                  f"({(time.time()-t0):.1f} s)")

    report = {
        "meta": {
            "dataset_dir":       str(dataset_dir),
            "model":             str(model_path),
            "candidate_db":      str(candidate_db),
            "min_support_for_kl": int(min_support),
            "elapsed_seconds":   round(time.time() - t0, 1),
        },
        "overall":               overall.finalize(),
        "by_band":               {b: s.finalize() for b, s in by_band.items()},
        "by_phase":              {p: s.finalize() for p, s in by_phase.items()},
        "by_transition":         {c: s.finalize() for c, s in by_trans.items()
                                  if s.n_events > 0},
        "empirical_kl_supported": {
            "min_support":  int(min_support),
            "n_samples":    int(kl_n),
            "mean_kl":      float(kl_sum / max(kl_n, 1)),
        },
        "provenance": adv.provenance,
    }
    return report


def _print_summary(report: dict) -> None:
    def _one(row: dict) -> str:
        return (f"n_ev={row['n_events']:>10,}   "
                f"nll={row['event_nll']:.4f}   "
                f"top1={row['top1']*100:5.2f}%   "
                f"top3={row['top3']*100:5.2f}%   "
                f"top5={row['top5']*100:5.2f}%   "
                f"ece={row['ece']:.3f}")
    print("=" * 80)
    print("OVERALL   " + _one(report["overall"]))
    print("=" * 80)
    print("Per Elo band:")
    for b, row in report["by_band"].items():
        print(f"  {b:<8}" + _one(row))
    print()
    print("Per phase:")
    for p, row in report["by_phase"].items():
        print(f"  {p:<8}" + _one(row))
    print()
    print("Per Malom transition:")
    for c, row in sorted(report["by_transition"].items()):
        print(f"  {c:<22}" + _one(row))
    print()
    ek = report["empirical_kl_supported"]
    print(f"Empirical KL (≥{ek['min_support']} events per sample): "
          f"n={ek['n_samples']:,}  mean_kl={ek['mean_kl']:.4f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, default=Path("data/human_blunder_dataset"))
    p.add_argument("--model",       type=Path, default=Path("data/human_blunder_net.npz"))
    p.add_argument("--candidate-db", type=Path, default=Path("data/human_db_candidate.sqlite"))
    p.add_argument("--min-support", type=int,  default=10,
                   help="Positions with < min_support observed events skip "
                        "the per-position empirical KL metric.")
    p.add_argument("--output",      type=Path, default=Path("data/human_blunder_eval.json"))
    args = p.parse_args()

    report = evaluate(args.dataset_dir, args.model, args.candidate_db,
                      min_support=args.min_support)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str),
                           encoding="utf-8")
    print(f"[eval] Report written → {args.output}")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
