# Sentinel Stage 1 Rollback — Heuristic-labelled Training from Human Games

Small preservation doc for the *original* Stage 1 recipe that the v2b
pipeline currently skips.  Kept here so it can be run again as a rollback
or as a future experiment.

## Why it was removed

Stage 1 trained on **weak heuristic labels** derived on-the-fly from replayed
game files.  It reached very healthy numbers — val loss ≈ 0.338, per-class
accuracy ≈ 98% for W / D / L — but the trunk it produced was so committed to
the heuristic function that downstream stages could not move it onto Malom
labels: Stage 2's val loss stalled identical to Stage 1's for 10 consecutive
epochs (`0.3382 → 0.3382 → …`) before early-stop.

The v2b default therefore skips Stage 1 (see `scripts/train_sentinel_v2_step0.sh`
and `docs/retrain_v2_plan.md`) and enters the chain at Stage 2 on Malom labels
from random init.  This doc keeps the old recipe alive as an experiment path.

## When it might be worth reviving

- **Pre-training on unlimited weak labels** could still be valuable if paired
  with a *high-LR* Stage 2 that can actually move the trunk (bump Stage 2 LR
  above the old `3e-4`), or with sample-weighting that up-weights positions
  where heuristic and Malom disagree.
- **Standalone heuristic sentinel** — useful as a lightweight, low-latency
  advisor that captures human-heuristic patterns without needing Malom at
  inference.  Nothing about the current Stage 1 output stops it being used
  directly as `learned_ai/sentinel/checkpoints/best.pt` if you want a
  "heuristic-flavoured" sentinel deployed alongside (or instead of) a v2
  Malom-trained one.
- **A/B baseline** — run this alongside a v2 run to measure how much of the
  v2 gain over v1 is actually structural vs. label-source-driven.

## Recipe A — Original Stage 1 (JSONL replay from both AI and human games)

Identical to what shipped with `configs/sentinel_stage1.yaml` before the skip:

```bash
.venv/bin/python scripts/train_sentinel.py \
  --config configs/sentinel_stage1.yaml \
  --game-dir data/games \
  --human-game-dir data/human_games \
  --drop-db-features \
  --out-dir learned_ai/sentinel/checkpoints/rollback_stage1 \
  --device cuda
```

Or, equivalent one-liner via the wrapper:

```bash
RUN_STAGE1=1 DEVICE=cuda ./scripts/train_sentinel_v2_step0.sh
```

The wrapper will run Stage 1 → Stage 2 → Stage 4 in sequence.  If you want
Stage 1 output only, kill the process after the Stage 1 completion banner
and copy `v2_stage1/best.pt` where you want it.

## Recipe B — Human-only variant (drop AI self-play games)

Trains purely on human JSONL games.  Useful if you want the sentinel's
"style" to mirror human play rather than what the AI already plays.

```bash
.venv/bin/python scripts/train_sentinel.py \
  --config configs/sentinel_stage1.yaml \
  --game-dir data/human_games \
  --drop-db-features \
  --out-dir learned_ai/sentinel/checkpoints/rollback_stage1_human_only \
  --device cuda
```

Two knobs to consider tuning if you go this route:

- `--decisive-only` — skip drawn games; forces the sentinel to learn from games
  with a real winner, which sharpens the heuristic signal.
- `--trajectory-weight` — up-weight the actually-played move in decisive games
  by the outcome-boost factors in `learned_ai/sentinel/labels.py`.

## Success criteria that made the old Stage 1 look good

- Val loss stabilises around **0.338** on `configs/sentinel_stage1.yaml`.
- Per-class accuracy on the validation split is ~98% W / ~100% D / ~100% L
  — this reflects the near-trivial heuristic-label task, not real strength.
- **Do not** interpret the ~98% number as sentinel quality; it's the model
  matching the heuristic function, which is exactly why it plateaued Stage 2.
  For real quality use `scripts/eval_sentinel_db.py` — see Step 6a of the
  plan.

## Chain-in options

- **Standalone** — copy `rollback_stage1/best.pt` to
  `learned_ai/sentinel/checkpoints/best.pt` to deploy as production.  Fast,
  small model with heuristic-shaped scoring.
- **Warm-start for future v2** — pass `--resume rollback_stage1/best.pt` to a
  future Stage 2 run that has a bumped LR (e.g. add `lr: 0.003` to
  `configs/sentinel_stage2.yaml` for that experiment) and/or a
  disagreement-weighted loss.  This is the "hybrid" avenue mentioned in
  `docs/retrain_v2_plan.md`'s Reconsider-later section.
- **Comparison run** — evaluate `rollback_stage1/best.pt` with
  `scripts/eval_sentinel_db.py` to see how the heuristic-trained trunk
  actually correlates with Malom ground truth; a low score there but high
  training accuracy is the diagnostic that motivated the skip.

## Files touched (for cleanup / audit)

- `configs/sentinel_stage1.yaml` — unchanged; still describes this stage.
- `scripts/train_sentinel_v2_step0.sh` — `RUN_STAGE1=1` env var restores it.
- `scripts/train_sentinel.py` — trainer itself is unchanged; understands
  both paths.
