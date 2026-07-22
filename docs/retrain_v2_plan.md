# Retrain v2 Plan — Sentinel, Value Net, Gap Net

## Integration Status — Proposal, Not Launch Contract

This maintainer-authored plan was imported from `main` on 22 July 2026. It is
valuable design input, but no retraining run or promotion is authorized. The
staged rebuilt databases remain under `../Mills`; neither has replaced the
active database selected by `data/training_paths.local.json`.

A read-only audit verified both staged SQLite files, their
`sector-corrected-v1` metadata, and 30 deterministic HumanDB labels against the
current corrected Malom adapter. The audit does not establish the lineage of
the seven updated model checkpoints or freeze the three proposed model
contracts. Exact hashes, counts, and open questions are in
[`docs/evidence/main-integration-audit-2026-07-22.md`](evidence/main-integration-audit-2026-07-22.md).

Before implementation, distinguish local engineering prerequisites from
maintainer decisions:

| Type | Unresolved item |
| --- | --- |
| Local | Activate a reviewed copy rather than replacing the active HumanDB or fresh-baseline SpecialistDB implicitly |
| Local | Build a game-level split from source games; aggregate HumanDB rows do not retain game membership |
| Local | Make required Sentinel/data loads fail closed and use the actual `--sentinel` gap-builder flag |
| Local | Replace raw-rate 40-game promotion claims with a frozen paired protocol and uncertainty rule |
| Maintainer | Confirm whether Sentinel must be DB-free at runtime; the proposed Stage 5 exposes oracle slots while evaluation masks them |
| Maintainer | Confirm whether ValueNet v2 is an outcome value or a human next-move preference ranker |
| Maintainer | Confirm whether GapNet predicts future human-blunder propensity or the current position-quality gap target |

Do not begin Stage 0 until the applicable maintainer contract is answered and a
new experiment document records inputs, output isolation, fixed work, tests,
checkpoint roles, and authorization.

## Motivation

The v1 sentinel, value net, and gap net were trained before the Malom sector-offset decoder bug was
fixed.  Specifically:

- **Legacy Specialist DB** had 364,262 unversioned Malom labels written by the old decoder. The
  staged rebuilt candidate has `sector-corrected-v1`, 2,112,951 retained empirical positions,
  and zero Malom labels. It is not the empty fresh-baseline database and is not active.
- **Active Human DB** still has its historical unversioned label columns masked. The staged
  21 July rebuild has `sector-corrected-v1`; all 2,167,498 positions have labels and sampled
  position/successor labels match the corrected adapter. It remains a candidate until activation
  and lineage are recorded.

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

## Step 0 — Train sentinel v2 (Stages 1 → 2 → 4; Stage 5 blocked)

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
  --db-path <malom_db_path-from-local-registry> \
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
  --db-path <malom_db_path-from-local-registry> \
  --drop-db-features \
  --resume learned_ai/sentinel/checkpoints/v2_stage2/best.pt \
  --out-dir learned_ai/sentinel/checkpoints/v2_stage4 \
  --device cpu
```

### Stage 5 — Blocked runtime-contract decision

Do not run the former DB-feature fine-tune command yet. The stated Sentinel
goal is a compact live approximation of Malom, and the proposed evaluation
zeroes DB feature slots to match inference. Training Stage 5 with those oracle
slots visible would introduce a train/inference mismatch and could teach the
network to depend on the database it is meant to replace.

The maintainer must first confirm the runtime contract. If Sentinel is DB-free
at runtime, retain `--drop-db-features` in every training stage and use Malom
only as the supervised teacher. If live inference intentionally supplies the
same oracle slots, document that dependency and storage goal explicitly before
restoring a Stage 5 command.

Stage 6 (AIDB + contrastive) is out of scope for this retrain cycle.

After the final approved Sentinel stage passes its own validation, record its
source hash and copy it into the v2 staging slot. No source stage or promotion
command is frozen while the Stage 5 contract remains unresolved. Do not
overwrite production `best.pt`.

---

## Step 1 — Regenerate gap net training dataset

Now that sentinel v2 exists, rebuild the gap dataset with the corrected Malom labels **and** the
better sentinel signal:

```bash
.venv/bin/python scripts/build_gap_dataset.py \
  --sentinel learned_ai/sentinel/checkpoints/v2/best.pt \
  --out data/gap_net_training_v2.npz \
  --samples-per-category 20000
```

The current builder accepts `--sentinel`. It presently falls back to
heuristics if the required checkpoint fails to load; the retraining workflow
must change that path to fail closed before this command is approved.

---

## Step 2 — Train value net v2

The v2 value net is trained on a **next-move prediction** task using human game positions from
`human_db.sqlite`, rather than the v1 approach of predicting final game outcome from JSONL records.

**Training approach:**
- For each unique board state in the human DB, apply a **per-position quality filter** to avoid
  training the value net to prefer drawn successors when winning successors were available.  Group
  all move records by `state_key`; for each group:
  - If **any** move recorded from this position has `malom_wdl_after = 'L'` (a winning move was
    played at least once from this state in the DB), only include records from this position where
    the human's chosen move **also** has `malom_wdl_after = 'L'`.
  - If no winning move appears for this `state_key` in the DB, accept records where the human's
    chosen move has `malom_wdl_after = 'D'` (draw was the best observed option from this state).
  - Skip any record where `malom_wdl_after = 'W'` (human played a losing move), regardless of
    what else is available.
- For each qualifying (position, chosen-move) pair, enumerate all legal moves from that position,
  apply each to get a set of successor board states, and extract 79-float features for each.
- Train with a **pairwise sigmoid BCE (Bradley-Terry) ranking loss**: for each (positive, negative)
  successor pair, minimise −log σ(v(human_successor) − v(other_successor)).  A margin ranking loss
  max(0, margin − (v(human) − v(other))) is an acceptable alternative, but pairwise sigmoid BCE
  gives smoother gradients and is preferred.
- All unique qualifying positions form the training set.  Repeat across epochs; stop via early
  stopping when validation next-move accuracy plateaus (see `--patience` below).

**New training script required:** `tools/train_value_net_v2.py` (or `scripts/train_value_net_human.py`).
The existing `tools/train_value_net.py` reads JSONL records with final-outcome labels and uses a
**numpy MLP with L2 loss — it cannot support a pairwise ranking loss without a significant
rewrite**.  The new script should use PyTorch (or rewrite the MLP with a pairwise loss function)
and must:

1. Query `human_db.sqlite`, grouping move records by `state_key` and applying the per-position
   filter above.
2. For each qualifying position, reconstruct the board from `state_key` (canonical FEN, reversible).
3. Enumerate all legal moves, apply each to get successor boards, extract 79-float features for
   each successor using `board_to_features()`.
4. Train with pairwise sigmoid BCE ranking loss to push the value of the human's chosen successor
   above all alternatives.
5. Track **validation next-move accuracy** (% of positions where the net ranks the human's
   successor highest) on a held-out **game-level** split; stop training when accuracy does not
   improve by ≥0.5pp for `--patience` consecutive epochs.

The aggregate `positions` and `moves` tables do not retain game membership, so
the script cannot create this game-level split by querying HumanDB alone. Split
the source game files first, replay each split, and join canonical state keys
to the versioned HumanDB labels, or build a new intermediate dataset that
retains a stable game identifier. A position-level split is not an acceptable
fallback.

Note: the ranking loss trains only relative ordering, not absolute scalar magnitudes.  The v2 VN
output scale may differ from the v1 VN (which was trained with L2 regression to a [−1, 1] final
outcome).  This does not make v2 unusable for VN blending, but the blend % may need re-calibration
— verify with the game bench in Step 4c before settling on a blend %.

```bash
.venv/bin/python tools/train_value_net_v2.py \
  --db data/human_db.sqlite \
  --output data/value_net_v2.npz \
  --patience 10   # stops if val next-move accuracy does not improve ≥0.5pp for 10 epochs
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

This tests sentinel correcting the heuristic at 20% and 30% minimum gap
thresholds. Forty games per pair is a diagnostic only; it does not by itself
give a promotion-grade uncertainty bound. Freeze a paired, colour-swapped
workload and confidence-interval decision rule before treating the result as
accept/reject evidence.

### 4c. Value net eval — next-move accuracy on human DB positions

**New eval script required:** `scripts/eval_value_net_human.py`.

For each unique position in `human_db.sqlite` where `malom_wdl_after IN ('L', 'D')` (the same
set used for v2 training, or a held-out test split):

1. Reconstruct the board from `state_key`.
2. Enumerate all legal moves; apply each to get successor boards and their 79-float features.
3. Run the value net over all successors; record whether the human's actual move produces the
   highest-valued successor.
4. Report **next-move accuracy** (% correct) broken down by game phase.

Run identically for v1 (`data/value_net.npz`) and v2 (`data/value_net_v2.npz`).  The winner is
the net with higher next-move accuracy — this is the primary value net success criterion.

```bash
# Evaluate old value net
.venv/bin/python scripts/eval_value_net_human.py \
  --db data/human_db.sqlite \
  --net data/value_net.npz \
  --out eval_vn_v1_nextmove.json

# Evaluate new value net
.venv/bin/python scripts/eval_value_net_human.py \
  --db data/human_db.sqlite \
  --net data/value_net_v2.npz \
  --out eval_vn_v2_nextmove.json
```

Use a dedicated test split for a fair comparison.  The split **must be at the game level** — hold
out whole games, not individual positions.  Position-level splits leak board states that appear in
multiple games and will inflate reported next-move accuracy.

**Sanity check — v2 VN game bench:** Because the ranking loss may shift the VN output scale
relative to v1 (which used L2 regression), run a brief game bench to confirm the v2 VN is still
useful when blended with the heuristic:

```bash
.venv/bin/python tools/bench_trajectory_value_net.py \
  --vn-path data/value_net_v2.npz \
  --blends 30 60 \
  --games 40
```

The v2 VN30 agent should achieve a win rate ≥50% vs the base heuristic.  If not, the output scale
has shifted adversely — re-scale or re-normalise v2 VN outputs before using it in Step 4d or
promoting to production.

### 4d. Gap net eval — beats heuristic + high VN blend

The gap net's job is to correct a heuristic AI that is leaning heavily on the value net (and
therefore susceptible to Malom-detectable blunders in the blender's shadow).  The test opponent
is **heuristic AI using `value_net_v2.npz` at VN blend 60–80%** — this is a stronger, more
relevant baseline than v1 VN, and ensures the gap net is not being compared against an already-
outdated opponent.

Run each gap net (old v1 and new v2) against this opponent:

```bash
# Old gap net vs heuristic+VN80 (v2 value net opponent)
# (requires --vn-path flag in bench_sentinel.py — add before running)
.venv/bin/python scripts/bench_sentinel.py \
  --games 40 --difficulty 5 \
  --white-gap-net \
  --black-value-net --vn-path data/value_net_v2.npz --vn-blend 80

# New gap net vs heuristic+VN80 (v2 value net opponent)
# (requires --gap-net-path and --vn-path flags in bench_sentinel.py — add before running)
.venv/bin/python scripts/bench_sentinel.py \
  --games 40 --difficulty 5 \
  --white-gap-net --gap-net-path data/gap_net_v2.npz \
  --black-value-net --vn-path data/value_net_v2.npz --vn-blend 80
```

The winning gap net is the one that achieves a higher win rate against the VN-blend opponent.
That winner is then promoted (Step 6).

Note: `bench_sentinel.py` hard-codes `data/gap_net.npz`.  Add a `--gap-net-path` CLI argument
before running the v2 comparison.

---

## Step 5 — Draft success criteria, pending frozen evaluation

The thresholds below are maintainer proposals. Do not promote until they are
embedded in a preregistered paired protocol with fixed positions, work per
move, sample size, confidence rule, candidate identities, and an inconclusive
outcome. After that freeze, promote a v2 model only if all applicable
conditions hold:

| Model | Condition |
|---|---|
| Sentinel | Malom correlation eval (4a): `win_acc` improves ≥3pp AND `top1_win_rate` improves ≥3pp vs v1 |
| Sentinel | Game bench (4b): NewS20 vs Base ≥50% win rate, and NewS20 beats or matches OldS20 within noise (±3pp) |
| Value net | Next-move accuracy eval (4c): v2 next-move accuracy ≥ v1 + 2pp on the game-level test split |
| Value net | Game bench (4c sanity check): v2 VN30 win rate vs base heuristic ≥ 50% |
| Gap net | Gap net vs VN80 bench (4d): new gap net win rate vs VN80 opponent ≥ old gap net win rate vs same |

If any model fails its criteria, investigate before promoting.  A v2 that performs identically to
v1 is not worth the disruption — only promote if there is a real improvement.

---

## Step 6 — Promotion (not authorized)

Only after a separately recorded acceptance and product authorization may the
v2 files be moved into the standard production paths. No executable promotion
command is recorded while the candidates and acceptance protocol are
undefined. Before any replacement, bind source and destination hashes, make a
recoverable backup, prove rollback, and use the repository's destructive-action
review rather than ad-hoc copy or rename commands.

After renaming:
- The Flask server picks up `best.pt` on restart (no config change).
- `ai/game_ai.py`, training scripts, and all benchmark scripts reference the standard paths and
  automatically use the new weights.
- `data/training_paths.local.json` requires no changes (paths are unchanged).

Restart the Flask server to pick up the new sentinel checkpoint.

---

## Step 7 — Rollback

Before promotion, create a rollback manifest containing the production and
candidate identities, backup locations, and verification command. If a
post-promotion regression appears, restore only through that reviewed manifest.
There is no generic rollback command because the exact artifact identities and
backup names do not yet exist.

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
  running the gap net v2 comparison (Step 4d).
- Two new scripts are required before testing can begin: `tools/train_value_net_v2.py` (Step 2)
  and `scripts/eval_value_net_human.py` (Step 4c).  Neither exists yet.
- Sentinel Stage 5 is blocked on the runtime oracle-feature contract; no epoch
  value should be selected before that decision.
- `build_gap_dataset.py` accepts `--sentinel`, but its required-sentinel path
  still needs to fail closed for this experiment.
- **Pre-flight: malom_wdl_after semantics.** The Step 2 training filter depends on 'L' meaning the
  next player loses (i.e. the human's move was winning) and 'W' meaning the human played a losing
  move. Fifteen staged successor rows matched the corrected Malom adapter for
  W/D/L and DTW. Preserve this convention in focused dataset tests rather than
  relying only on the audit sample.
- **Pre-flight: bench_sentinel.py VN path.** Step 4d requires the opponent to use
  `value_net_v2.npz`.  Add a `--vn-path` CLI argument to `bench_sentinel.py` alongside
  `--gap-net-path` before running Step 4d.
