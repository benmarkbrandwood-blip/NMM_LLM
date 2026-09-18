# Human Teacher v4 — Training Plan

**Date:** 2026-09-18  
**Status:** Part 1 in progress.

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

**Teacher (Part 1):**
- Top-3 accuracy on test split improves over current teacher across all
  three Elo bands.
- Top-3 accuracy on 13+ legal-move positions (high-branching) improves
  meaningfully — this is the bucket where the current teacher is near-uniform.

**Generalist (Part 2):**
- No regression in win rate vs heuristic opponent at the current difficulty
  after adding the 25% teacher slot.
- Qualitative improvement in play style against human opponents (evaluated
  via game review, not automated gate).
