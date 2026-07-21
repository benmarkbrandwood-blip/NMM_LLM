# Retrain v2 Plan — Sentinel, Value Net, Gap Net

## Motivation

The v1 sentinel, value net, and gap net were trained before the Malom sector-offset decoder bug was
fixed.  Specifically:

- **Specialist DB** had 364,262 Malom labels written by the old decoder.  Many wins/losses were
  misclassified as draws.  All labels have been cleared (`sector-corrected-v1` provenance now
  stamped); correct labels will accumulate as training runs.
- **Human DB** had Malom WDL suppressed (missing version tag).  A full `--rebuild` with the
  corrected decoder completed 2026-07-21; all 2,167,498 positions now have correct labels
  (100% Malom coverage, 0 unknowns).

The gap net training dataset (`data/gap_net_training.npz`) reads `malom_wdl_after` directly from
`human_db.sqlite`, so it was built on corrupt labels and must be regenerated.  The gap dataset
builder also uses the **sentinel** to compute composite quality scores — so the gap dataset must be
rebuilt *after* the sentinel v2 checkpoint is available, to get the best possible signal.

The sentinel's dataset reads from the Malom DB at feature-build time; retraining from scratch means
all examples are now correctly labelled.

The value net trains from game-outcome JSONL files (final win/loss outcome, no Malom labels), so
it does not directly benefit from the DB fix.  It is retrained as v2 to establish a clean baseline
under the same naming convention, and to optionally incorporate a richer game-file mix.

---

## Naming conventions

During training the v2 artifacts are kept under distinct names to avoid clobbering production.
Once confirmed better (Step 5), they are renamed in place so all existing code paths continue to
work without any changes.

| Artifact | During training (v2 name) | After promotion (production name) |
|---|---|---|
| Sentinel checkpoint | `learned_ai/sentinel/checkpoints/v2/best.pt` | `learned_ai/sentinel/checkpoints/best.pt` |
| Sentinel stage dirs | `checkpoints/v2_stage{1,2,4,5}/` | *(staging only, not referenced by runtime)* |
| Value net | `data/value_net_v2.npz` | `data/value_net.npz` |
| Gap net | `data/gap_net_v2.npz` | `data/gap_net.npz` |
| Gap net training data | `data/gap_net_training_v2.npz` | *(build artefact, not referenced at runtime)* |

Avoid `value_net_human_v2.npz` — that name is already taken by an earlier experiment.  Use
`value_net_v2.npz` for the new production net.

**Do not overwrite production files** (`best.pt`, `value_net.npz`, `gap_net.npz`) during training
or testing.  Renaming into production is a deliberate final step described in Step 6.

---

## Step 0 — Train sentinel v2 (Stages 1 → 2 → 4 → 5)

**Sentinel must be trained first** so the gap net dataset can be built with the correct sentinel
signal in Step 1.

Stage 3 is permanently archived (feature leakage); do not run it.

The `v2/` checkpoint directory already exists from a prior partial run.  Use `v2_stage{N}`
sub-directories to avoid overwriting anything already there.  If a usable Stage 5 checkpoint
already exists at `checkpoints/v2/best.pt`, evaluate it first (Step 4) before retraining.

### Stage 1 — Structural foundation

No DB labels.  `--drop-db-features` zeroes oracle slots so the model learns pure structural
patterns.

```bash
.venv/bin/python scripts/train_sentinel.py \
  --config configs/sentinel_stage1.yaml \
  --game-dir data/games \
  --human-game-dir data/human_games \
  --drop-db-features \
  --out-dir learned_ai/sentinel/checkpoints/v2_stage1 \
  --device cpu
```

### Stage 2 — DB calibration

Resume from Stage 1.  DB provides strong WDL + DTM labels; `--drop-db-features` still zeroes
oracle indicator slots so weights update toward ground truth without memorising the oracle path.

```bash
.venv/bin/python scripts/train_sentinel.py \
  --config configs/sentinel_stage2.yaml \
  --game-dir data/games \
  --human-game-dir data/human_games \
  --db-path /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted \
  --drop-db-features \
  --resume learned_ai/sentinel/checkpoints/v2_stage1/best.pt \
  --out-dir learned_ai/sentinel/checkpoints/v2_stage2 \
  --device cpu
```

### Stage 4 — Corrected full training

Resume from Stage 2.  DB labels with `--drop-db-features`.  Human game data included.

```bash
.venv/bin/python scripts/train_sentinel.py \
  --config configs/sentinel_stage4.yaml \
  --game-dir data/games \
  --human-game-dir data/human_games \
  --db-path /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted \
  --drop-db-features \
  --resume learned_ai/sentinel/checkpoints/v2_stage2/best.pt \
  --out-dir learned_ai/sentinel/checkpoints/v2_stage4 \
  --device cpu
```

### Stage 5 — DB feature fine-tune

Resume from Stage 4.  DB feature slots now **visible** (no `--drop-db-features`).  Very low LR.
Set `--epochs` to Stage 4's best epoch + 8 (check the Stage 4 checkpoint or log).

```bash
.venv/bin/python scripts/train_sentinel.py \
  --config configs/sentinel_stage5.yaml \
  --game-dir data/games \
  --human-game-dir data/human_games \
  --db-path /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted \
  --resume learned_ai/sentinel/checkpoints/v2_stage4/best.pt \
  --out-dir learned_ai/sentinel/checkpoints/v2_stage5 \
  --epochs <stage4_best_epoch + 8> \
  --device cpu
```

Stage 6 (AIDB + contrastive) is out of scope for this retrain cycle.

After Stage 5 completes, copy the best checkpoint to the v2 slot but **do not** overwrite the
production `best.pt` yet:

```bash
cp learned_ai/sentinel/checkpoints/v2_stage5/best.pt \
   learned_ai/sentinel/checkpoints/v2/best.pt
```

---

## Step 1 — Regenerate gap net training dataset

Now that sentinel v2 exists, rebuild the gap dataset with the corrected Malom labels **and** the
better sentinel signal:

```bash
.venv/bin/python scripts/build_gap_dataset.py \
  --sentinel-ckpt learned_ai/sentinel/checkpoints/v2/best.pt \
  --out data/gap_net_training_v2.npz \
  --samples-per-category 20000
```

(Check whether `build_gap_dataset.py` accepts a `--sentinel-ckpt` argument; if not, temporarily
point the hard-coded sentinel path at `v2/best.pt`, or add the flag before running.)

---

## Step 2 — Train value net v2

The value net trains from game-outcome JSONL records; include both AI self-play and human game
directories for the widest coverage.

```bash
.venv/bin/python tools/train_value_net.py \
  --games-dir data/games data/human_games \
  --output data/value_net_v2.npz \
  --decisive-only \
  --epochs 30
```

---

## Step 3 — Train gap net v2

Use the regenerated v2 training dataset (Step 1):

```bash
.venv/bin/python tools/train_gap_net.py \
  --data data/gap_net_training_v2.npz \
  --out  data/gap_net_v2.npz \
  --epochs 80
```

---

## Step 4 — Testing protocol

### 4a. Malom DB correlation eval (sentinel)

`eval_sentinel.py` iterates JSONL game files.  The user's requirement is to evaluate sentinel
against **DB-sampled positions from specialist_db (AI-vs-AI)** and **human_db (human-vs-human)**.
This requires a new evaluation script — call it `scripts/eval_sentinel_db.py` — that:

1. Samples N positions from each SQLite DB (e.g. 1,000 from each, stratified by phase).
2. Reconstructs each board from the stored `state_key` (canonical FEN, reversible for human_db;
   position-hash only for specialist_db — use the raw FEN column if present, or reconstruct from
   the stored board representation).
3. Queries Malom DB for per-move WDL + DTM for all legal moves at each position.
4. Runs sentinel (DB slots zeroed, matching live inference) and scores all legal moves.
5. Reports the same metrics as `eval_sentinel.py`: `win_acc`, `loss_acc`, `top1_win_rate`,
   `spearman_r`, `dtm_pearson_r`, and phase breakdown.

Run for both v1 and v2 sentinel to produce a direct comparison table.

**Minimum sample size:** ≥500 positions per source (specialist_db and human_db) to get stable
Spearman r.

### 4b. Game benchmark — sentinel v2 vs v1 vs baseline

Use the existing `tools/bench_sentinel_v2.py`.  It already tests Base, OldS20, OldS30 vs NewS20,
NewS30 in a round-robin — point `--new-ckpt` at the new v2 checkpoint:

```bash
.venv/bin/python tools/bench_sentinel_v2.py \
  --diff 5 \
  --budget 3.0 \
  --games-per-pair 40 \
  --old-ckpt learned_ai/sentinel/checkpoints/best.pt \
  --new-ckpt learned_ai/sentinel/checkpoints/v2/best.pt \
  --out eval_sentinel_v2_roundrobin.json
```

This tests sentinel correcting the heuristic at 20% and 30% minimum gap thresholds.  40 games/pair
(20 per colour) gives enough statistical separation for a clear ±5pp verdict.

### 4c. Game benchmark — value net v2 and gap net v2

Use `bench_trajectory_value_net.py` which supports `--blends 10 30`:

```bash
# Value net comparison (old vs v2, blends 10% and 30%)
.venv/bin/python scripts/bench_trajectory_value_net.py \
  --vn-path data/value_net_v2.npz \
  --blends 10 30 \
  --games 40 \
  --difficulty 5
```

For gap net, `bench_sentinel.py` has `--white-gap-net` / `--black-gap-net` (on/off, no blend
percentage).  Run a head-to-head of old gap net vs new gap net vs baseline:

```bash
# Old gap net vs baseline
.venv/bin/python scripts/bench_sentinel.py \
  --games 40 --difficulty 5 --white-gap-net

# New gap net vs baseline (point GameAI at gap_net_v2.npz — requires --gap-net-path flag
# or temporarily swap the file path; add --gap-net-path to bench_sentinel.py if needed)
```

Note: `bench_sentinel.py` hard-codes `data/gap_net.npz`.  Add a `--gap-net-path` argument before
running the gap net v2 comparison.

---

## Step 5 — Success criteria for promotion

Promote a v2 model to replace v1 production only if **all three** conditions hold:

| Model | Condition |
|---|---|
| Sentinel | Malom correlation eval: `win_acc` improves ≥3pp AND `top1_win_rate` improves ≥3pp vs v1 |
| Sentinel | Game bench: NewS20 vs Base ≥50% win rate, and NewS20 beats or matches OldS20 within noise (±3pp) |
| Value net | Game bench: V2-VN10 or V2-VN30 beats Baseline AND is not beaten by OldVN at same blend |
| Gap net | Game bench: new gap net vs baseline ≥50% win rate AND ≥OldGap vs baseline within noise |

If any model fails its criteria, investigate before promoting.  A v2 that performs identically to
v1 is not worth the disruption — only promote if there is a real improvement.

---

## Step 6 — Promotion

When v2 is confirmed better, the v2 files are **renamed into the standard production paths** so
that all existing code, training scripts, and web server continue to work without any changes.

```bash
# Back up v1 production files
cp learned_ai/sentinel/checkpoints/best.pt \
   learned_ai/sentinel/checkpoints/best-v1-$(date +%Y%m%d).pt
cp data/value_net.npz  data/value_net_v1_$(date +%Y%m%d).npz
cp data/gap_net.npz    data/gap_net_v1_$(date +%Y%m%d).npz

# Rename v2 into production (overwrites previous production files)
mv learned_ai/sentinel/checkpoints/v2/best.pt \
   learned_ai/sentinel/checkpoints/best.pt
mv data/value_net_v2.npz  data/value_net.npz
mv data/gap_net_v2.npz    data/gap_net.npz
```

After renaming:
- The Flask server picks up `best.pt` on restart (no config change).
- `ai/game_ai.py`, training scripts, and all benchmark scripts reference the standard paths and
  automatically use the new weights.
- `data/training_paths.local.json` requires no changes (paths are unchanged).

Restart the Flask server to pick up the new sentinel checkpoint.

---

## Step 7 — Rollback

If production regressions appear, restore the dated backups:

```bash
mv learned_ai/sentinel/checkpoints/best-v1-YYYYMMDD.pt \
   learned_ai/sentinel/checkpoints/best.pt
mv data/value_net_v1_YYYYMMDD.npz  data/value_net.npz
mv data/gap_net_v1_YYYYMMDD.npz    data/gap_net.npz
```

The `v2_stage{N}/` checkpoint directories and the v2 training datasets remain for continued
iteration.

---

## Known issues / caveats

- `specialist_db.sqlite` positions use `pos_hash` (SHA-1 of canonical FEN, not reversible).  Check
  whether the DB also stores a raw FEN or board representation that `eval_sentinel_db.py` can use.
  If not, the specialist_db correlation eval must be done via JSONL game replay (same as
  `eval_sentinel.py`) rather than direct DB sampling.
- The `v2/` checkpoint directory already contains a partial earlier attempt.  Inspect before
  starting Stage 1 training; if it contains a usable Stage 5 checkpoint, evaluate it first rather
  than retraining from scratch.
- `bench_sentinel.py` hard-codes `data/gap_net.npz`.  Add a `--gap-net-path` CLI argument before
  running the gap net v2 comparison (Step 4c).
- Sentinel Stage 5's `--epochs` value depends on Stage 4's best epoch.  Check the Stage 4 log or
  checkpoint metadata before running Stage 5.
- `build_gap_dataset.py` may not yet accept `--sentinel-ckpt` as a CLI argument.  Verify and add
  the flag if needed before running Step 1.
