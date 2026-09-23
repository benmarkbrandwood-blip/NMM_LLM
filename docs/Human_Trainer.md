# Human Teacher v4 — Training Plan

**Date:** 2026-09-18  
**Status:** Complete — both parts done; evaluation and overlay wiring finalised 2026-09-19.

---

## Background

GapNet v3 is abandoned (see `gap_net_v3_fail.md`).  The human signal in the
database is real but cannot survive a three-stage approximation chain.  The
new approach uses the teacher directly as a training opponent for the
generalist AI, replacing the GapNet correction pathway entirely.

---

## Part 1 — Retrain the Teacher

### Dataset

`data/human_move_policy_dataset_v2_new` — already extracted from
`human_db_candidate_new.sqlite` (103,278 source files, 2,308,662 positions,
all game outcomes included, no session-ledger filtering).

| Split | Samples |
|---|---|
| Train | 2,143,486 |
| Val | 402,409 |
| Test (held out) | 133,775 |

The test split is held out entirely; it is only used for the post-training
evaluation described below.

### Trainer change — branching-factor loss weighting

Current trainer weights each position by event count only.  A position with
16 legal moves and 1 observed event contributes as much per-event as a
position with 3 legal moves — the model cannot learn preferences in
high-branching positions.

Add `--branch-weight {none, linear, log}` flag (default `log`):

- `none` — current behaviour, no extra weight.
- `linear` — multiply each sample's loss by `n_legal_moves`.
- `log` — multiply by `log(1 + n_legal_moves)`.  Softer; avoids extreme
  up-weighting of rare 18-move positions over common 12-move ones.

`n_legal_moves` is the number of rows in the successor feature matrix for
that sample (already available in `sample_slice`).

### Test evaluation — top-K move accuracy

After training, `--eval-test` runs the model on the held-out test split and
reports how often the actual human move appears in the model's top-K
predictions.  Metrics:

- Top-1, Top-3, Top-5 accuracy
- Stratified by Elo band: lower / middle / upper
- Stratified by legal-move-count bucket: ≤4 / 5–8 / 9–12 / 13+

Comparison: both old model and new model evaluated against the same test
split so improvement is directly measurable.

### Output

`data/human_move_policy_net_v4_branching.npz`

### Command

```
.venv/bin/python tools/train_human_move_policy_net.py \
    --dataset-dir data/human_move_policy_dataset_v2_new \
    --output data/human_move_policy_net_v4_branching.npz \
    --branch-weight log \
    --eval-test \
    --epochs 60 \
    --patience 8 \
    --force
```

---

## Part 2 — Wire Teacher into the Generalist

### Interface fix

`game_ai._apply_humanlike_adjust` calls `advisor.probs(board, moves)` with
two arguments.  `HumanMovePolicyAdvisor.probs` currently requires a third
`elo_band` argument.

Fix: add `elo_band: str = "all"` as a default parameter to
`HumanMovePolicyAdvisor.probs()`.  This makes the teacher duck-type
compatible with `HumanPrefAdvisor` at the call site; `"all"` averages the
three Elo-band distributions, which is the right default for an unknown
opponent.

No other changes to `game_ai.py` or `heuristics.py`.

### New opponent slot

Load the retrained teacher via `HumanMovePolicyAdvisor`.  Pass it as
`human_pref_net` to the heuristic agent with `humanlike_blend=50` (same
blend weight as the existing HumanPrefNet slot).

New CLI flags for `scripts/train_s_gen_v3.py`:

- `--human-teacher-net PATH` — path to teacher .npz (default
  `data/human_move_policy_net_v4_branching.npz`)
- `--human-teacher-blend INT` — blend weight 0–100 (default 50)
- `--no-human-teacher` — disable the slot entirely

### Revised opponent schedule

| Slot | Old % | New % | Notes |
|---|---|---|---|
| vs_heuristic_hard | 10 | 10 | unchanged |
| vs_heuristic_easy | 20 | 15 | −5% |
| vs_heuristic_blunder | 10 | 10 | unchanged |
| vs_heuristic_blend | 10 | 5 | −5% |
| vs_heuristic_humanlike (HumanPrefNet) | 5 | 5 | unchanged |
| vs_heuristic_teacher_blended (new) | — | 25 | new slot |
| standard (self-play / current-diff heuristic) | 45 | 30 | −15% |

### Advancement gate

The advancement gate (`level_heuristic_history`) feeds only from
`is_advance_reference` games — i.e. `vs_heuristic` at exactly the current
difficulty.  Teacher-blended games are not reference games and do not affect
advancement timing.

### Prerequisites

- Part 1 complete and test evaluation passes.
- `data/human_move_policy_net_v4_branching.npz` present.

---

## Success Criteria

**Teacher (Part 1):** ✓ Met
- Top-3 accuracy on 13+ legal-move positions improved by +10–16% across all
  Elo bands vs HumanPrefNet (the prior best model).
- High-branching positions (openings, early midgame) are now well-predicted.

**Generalist (Part 2):** ✓ Met
- Teacher slot wired into generalist at 25% blend (humanlike_blend=50).
- No gate regressions observed at difficulty 11.

---

## Part 3 — Evaluation: Teacher vs HumanPrefNet

### Accuracy comparison (2026-09-19)

`tools/compare_human_nets.py` evaluated both models on the 133,775-sample
held-out test split (upper band for teacher).

| Position type | Winner | Margin (top-1) |
|---|---|---|
| 13+ moves (openings, early midgame) | **Teacher** | +10–16% |
| 9–12 moves | HumanPrefNet mostly | +1–3% |
| 5–8 moves | **HumanPrefNet** | +1–2% |
| 1–4 moves (endgame) | Tied | <1% |

Teacher wins decisively on high-branching positions — the branching-weight
fix worked.  HumanPrefNet retains an edge in constrained late-game positions,
where its pairwise ranking loss gives sharper resolution.

### Round-robin tournament (2026-09-19)

`tools/human_nets_tournament.py` — 6 configs × 40 games/pair = 600 games,
diff=5, 0.1s/move budget.

| Config | Pts | % | Notes |
|---|---|---|---|
| HP-25 | 112.5 | 56.2% | **Best overall** |
| T-50 | 112.0 | 56.0% | Tied for best |
| T-25 | 109.0 | 54.5% | |
| HP-50 | 108.5 | 54.2% | |
| T-pure | 86.0 | 43.0% | |
| HP-pure | 72.0 | 36.0% | Worst |

Both pure-human configs play poorly; a heuristic scaffold is essential.
HumanPrefNet needs more heuristic (25% human) to peak; teacher peaks at
50% human — consistent with its stronger raw move predictions.

### Final wiring decisions

- **Pred overlay** (`web/app.py`): `human_move_policy_net_v4_branching.npz`
  (teacher, v4).  Teacher is far better in high-branching positions where the
  overlay gives the most guidance value.
- **Humanlike-play slider**: `human_pref_net.npz` (HumanPrefNet).  Wins the
  head-to-head tournament at HP-25 and is the established play-style option.
- **Candidate DB**: `human_db_candidate_new.sqlite` wired for both overlay
  trajectory data and game writes (full swap).
