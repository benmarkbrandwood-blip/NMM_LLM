# GapNet v3 — Post-Mortem

**Date:** 2026-09-18  
**Status:** Pipeline abandoned. Teacher retrain + direct opponent approach adopted.

---

## What GapNet v3 Was Trying To Do

GapNet v3 was a position-level scalar predictor of **G_v** — the expected
move-regret a human opponent will incur at a board position, conditioned on
their Elo band.  The intended use was to adjust the generalist AI's search
evaluation so that, among moves of similar minimax value, it would prefer
positions where the opponent is statistically likely to blunder.

The full pipeline:

```
Human games (HumanDB)
  → Teacher MLP: learns P(move | position, Elo_band)
    → Extractor: computes G_v = Σ_m  P_teacher(m) × Malom_regret(m)
      → GapNet MLP: approximates G_v at inference speed
        → Search: adds G_v signal to position evaluation
```

Each step added abstraction and approximation error on top of the last.

---

## The Stage E Gate

Stage E was a promotion gate with 9 cells — one per (band × component).
The three components of G_v were `class_downgrade`, `wdl_utility_loss`, and
`ordinal_rank_loss`, all derived from the `regret_v1` formula.

**Gate 1** (the binding gate):  
`candidate_MSE ≤ (1 − 0.30) × uniform_MSE`

The candidate had to beat a uniform-distribution baseline by at least 30%
on the subset of positions where real empirical G_v data existed (n ≥ 5
human visits, ~0.78% of the dataset, referred to as "hybrid" rows).  These
are the opening and common midgame positions where humans play most often.

**Gate 2:**  
`candidate_MSE ≤ (1 + 0.20) × teacher_MSE`

The candidate had to be within 20% of the teacher's own G_v estimates.

---

## What Actually Happened

Gate 2 **passed** for every evaluable cell in the stress test.  The model
architecture and training procedure were sound.

Gate 1 **failed** for all six evaluable cells.  The gate was structurally
unachievable given the teacher's quality on hybrid positions:

| Component | Band | Teacher vs uniform |
|---|---|---|
| class_downgrade | middle | teacher +1.7% better |
| class_downgrade | upper | ~0% (effectively equal) |
| wdl_utility_loss | middle | teacher **−15%** (worse) |
| wdl_utility_loss | upper | teacher **−21%** (worse) |
| ordinal_rank_loss | middle | teacher +13% better |
| ordinal_rank_loss | upper | teacher +21% better |

Gate 1 demanded a 30% improvement over uniform.  For `wdl_utility_loss`, the
teacher was actively worse than uniform.  No amount of GapNet training could
achieve Gate 1 when the target labels (teacher G_v) carry no signal over
uniform on exactly the positions being evaluated.

Stress test result (trained on 19,186 hybrid rows only, empirical targets,
best val_loss = 0.053226):

```
class_downgrade:   candidate=0.0556  uniform=0.0535  gate1_threshold=0.0375  [FAIL]
wdl_utility_loss:  candidate=0.0703  uniform=0.0548  gate1_threshold=0.0384  [FAIL]
ordinal_rank_loss: candidate=0.0338  uniform=0.0413  gate1_threshold=0.0289  [FAIL]
```

Even with direct empirical targets, the candidate could not beat uniform by
30% because the positions with empirical data are high-branching positions
where the human preference signal is genuinely diffuse.

---

## Root Cause

**The teacher is near-uniform on the positions that matter most.**

The teacher MLP learns P(move | position, Elo_band) from one observed move
per game visit.  In opening and common midgame positions there are 15–18
legal moves, many of which are objectively reasonable.  The sparse per-move
label count means the model cannot learn a sharp preference in those positions
— it assigns near-equal probability to everything, reproducing the uniform
baseline.

This is not a fixable hyperparameter problem.  It is a fundamental
data-density issue: the observed move at a high-branching position does not
tell the model why the other legal moves were rejected, and with few visits
per (position, move) pair the model never accumulates enough signal to
discriminate.

The downstream consequence: G_v = Σ P_teacher(m) × Malom_regret(m) ≈
Σ (1/K) × Malom_regret(m) = uniform G_v on frequent positions.  GapNet is
then trying to learn to replicate a signal that is itself uninformative.

---

## What We Are Doing Instead

The GapNet pipeline is abandoned.  The human signal that exists in the
database is real — it just cannot be expressed through this many
indirection layers.

**New direction:**

1. Retrain the teacher on the full `human_db_candidate_new.sqlite` (larger,
   more recent) using all human moves regardless of game outcome.  Add
   loss weighting by legal-move count at each position so high-branching
   positions receive proportionally more gradient — these are precisely
   the positions where the current teacher is blind, and where the most
   value lies.

2. Use the retrained teacher directly as a **training opponent** for the
   generalist AI, constituting ~30% of training games.  This replaces the
   existing 5% humanlike-blend slot and removes the GapNet approximation
   chain entirely.  The generalist learns a value function implicitly
   calibrated to human-simulated opponents without any explicit G_v
   computation.

3. Blend with the existing opening book for the placement phase, where
   human preferences are most structured and the teacher's coverage is
   strongest.

This approach answers the original question directly — "how do I beat human
opponents?" — by training against a frozen, empirically-grounded human
simulator rather than approximating a scalar correction to a general-purpose
search.
