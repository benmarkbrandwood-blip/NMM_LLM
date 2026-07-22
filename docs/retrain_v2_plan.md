# Retrain v2 Plan — Sentinel, Value Net, Gap Net

## Integration Status — Proposal, Not Launch Contract

This maintainer-authored plan was imported from `main` on 22 July 2026. It is
valuable design input, but no retraining run or promotion is authorized. The
staged rebuilt databases remain under `../Mills`; neither has replaced the
active database selected by `data/training_paths.local.json`.

A read-only audit verified both staged SQLite files, their
`sector-corrected-v1` metadata, and 30 deterministic HumanDB labels against the
current corrected Malom adapter. The audit does not establish the lineage of
the seven updated model checkpoints or authorize a retraining run. Exact
hashes, counts, and evidence boundaries are in
[`docs/evidence/main-integration-audit-2026-07-22.md`](evidence/main-integration-audit-2026-07-22.md).

Current code and the written plan are sufficient to resolve the apparent model
contract ambiguities without asking the maintainer to restate them:

| Type | Disposition |
| --- | --- |
| Local | Activate a reviewed copy rather than replacing the active HumanDB or fresh-baseline SpecialistDB implicitly |
| Local | Build a game-level split from source games; aggregate HumanDB rows do not retain game membership |
| Local | Make required Sentinel/data loads fail closed and use the actual `--sentinel` gap-builder flag |
| Local | Replace raw-rate 40-game promotion claims with a frozen paired protocol and uncertainty rule |
| Resolved contract | Sentinel is a DB-free board/move model; Malom is a label teacher only, and oracle-derived input slots stay masked in every training and inference path |
| Resolved contract | The existing ValueNet remains an outcome-value estimator; the imported next-move ranking proposal is a separate HumanPolicy research path, not a replacement ValueNet checkpoint |
| Resolved contract | GapNet retains its implemented current-position target: best composite move quality minus frequency-weighted human-play quality |
| Conditional evidence | The seven imported checkpoints need additional maintainer lineage only if a future experiment proposes to adopt or describe them as corrected inputs |
| External review | Complete: the owner excluded original Oracle position 101 and accepted the remaining 106; the resulting package still gates only the separate Stage-0 diagnostic freeze |

Do not begin any retraining stage until a new experiment document records
inputs, output isolation, fixed work, tests, checkpoint roles, and explicit
product authorization. The contract dispositions above do not themselves
authorize training. The Oracle review belongs to the separate Stage-0
evaluation proposal and does not block local retraining safety work.

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

The existing value net trains from game-outcome JSONL files (final win/loss
outcome, no Malom labels), so it does not directly benefit from the DB fix.
The imported proposal later changes the target to human next-move ranking.
That is a different behavioural model and is retained below only as historical
design input; it must not be saved or promoted as `value_net_v2.npz`.

---

## Naming conventions

During training the v2 artifacts are kept under distinct names to avoid clobbering production.
Once confirmed better (Step 5), they are renamed in place so all existing code paths continue to
work without any changes.

| Artifact | During training (v2 name) | After promotion (production name) |
|---|---|---|
| Sentinel checkpoint | `learned_ai/sentinel/checkpoints/v2/best.pt` | `learned_ai/sentinel/checkpoints/best.pt` |
| Sentinel stage dirs | `checkpoints/v2_stage{1,2,4}/` | *(staging only, not referenced by runtime)* |
| HumanPolicy research candidate | Not frozen by this plan | No production mapping |
| Gap net | `data/gap_net_v2.npz` | `data/gap_net.npz` |
| Gap net training data | `data/gap_net_training_v2.npz` | *(build artefact, not referenced at runtime)* |

Do not create or promote `data/value_net_v2.npz` from the next-move ranking
proposal. A future HumanPolicy experiment must choose its own versioned name,
schema, raw-game split manifest, calibration contract, and runtime adapter in
the experiment that owns it.

**Do not overwrite production files** (`best.pt`, `value_net.npz`, `gap_net.npz`) during training
or testing.  Renaming into production is a deliberate final step described in Step 6.

---

## Step 0 — Train sentinel v2 (Stages 1 → 2 → 4; Stage 5 blocked)

**Sentinel must be trained first** so the gap net dataset can be built with the correct sentinel
signal in Step 1.

Stage 3 is permanently archived (feature leakage); do not run it.

The `v2/` checkpoint directory already exists from a prior partial run. Use
`v2_stage{N}` sub-directories to avoid overwriting anything already there. Any
existing Stage 5 checkpoint has oracle-exposed lineage and may be inventoried
only as a historical ablation; it is not a candidate for the DB-free Sentinel.

The shared Sentinel feature contract now makes DB-only slots unavailable in
both training entry points and in offline evaluation. `--drop-db-features` is
retained in the commands as an explicit provenance marker, but it is no longer
an opt-in safety switch: omitting it does not expose the oracle slots.

### Stage 1 — Structural foundation

No DB labels. The mandatory feature contract zeroes oracle slots so the model
learns pure structural patterns.

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

Resume from Stage 1. DB provides strong WDL + DTM labels; the mandatory mask
still zeroes oracle indicator slots so weights update toward ground truth
without memorising the oracle path.

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

Resume from Stage 2. DB supplies labels while the mandatory feature contract
keeps its oracle input slots masked. Human game data is included.

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

### Stage 5 — Rejected for the DB-free runtime contract

Do not run the former DB-feature fine-tune command. The stated Sentinel
goal is a compact live approximation of Malom, and the proposed evaluation
zeroes DB feature slots to match inference. Training Stage 5 with those oracle
slots visible would introduce a train/inference mismatch and could teach the
network to depend on the database it is meant to replace.

The current inference wrapper is explicitly DB-free, and a shared contract now
enforces the same oracle-slot mask in both trainers and evaluation. Use Malom
only as the supervised teacher. A future model that intentionally consumes live
oracle fields would be a separately named architecture and experiment, not a
restored Stage 5.

Stage 6 (AIDB + contrastive) is out of scope for this retrain cycle.

After the final approved Sentinel stage passes its own validation, record its
source hash and copy it into the v2 staging slot. No source stage or promotion
command is frozen by this proposal. Do not overwrite production `best.pt`.

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

The builder accepts `--sentinel` and now fails closed before database sampling
when that checkpoint is missing, incompatible, unavailable after load, or
returns malformed/non-finite scores. It also fails closed if Sentinel inference
breaks while labels are being generated. This clears the loader behaviour only;
the command remains unauthorized until its experiment inputs are frozen.

---

## Step 2 — Archived next-move ranking proposal

The imported Step 2 changes the supervision unit from final board outcome to
observed human move preference. That is not ValueNet retraining: it is a
HumanPolicy problem. Reusing the ValueNet class, filename, blend percentage,
or promotion path would silently change the meaning and calibration of an
existing production interface.

Do not implement the aggregate-table pairwise proposal as
`tools/train_value_net_v2.py`, and do not create `data/value_net_v2.npz` from
it. The aggregate `positions` and `moves` tables have already discarded game
and player membership and cannot support the required leakage-safe split or a
complete observed choice-set likelihood.

Any future HumanPolicy implementation is owned by the data and model contract
in [`docs/v5-specialist-plan.md`](v5-specialist-plan.md). It must start from
the raw JSONL records, preserve stable game and available player identity,
split before aggregation, represent the complete legal choice set for each
observed decision, and define calibration, OOD, and abstention evidence. That
work requires a separate experiment and authorization; it is not a prerequisite
for Sentinel or GapNet safety fixes.

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

### 4c. HumanPolicy evaluation — outside this retrain cycle

Do not compare the existing outcome ValueNet and a human-choice model under a
single “next-move accuracy” promotion rule. A future HumanPolicy evaluation
must use a frozen raw-game split with player/game isolation, multiclass choice
likelihood and calibration, support/OOD slices, and a separately named
checkpoint. The v5 plan owns those requirements.

### 4d. Gap net eval — beats heuristic + high VN blend

GapNet v2 retains the implemented current-position target: the difference
between the best composite move quality and the frequency-weighted quality of
observed human moves. “Anticipating blunders” describes the intended use of
that current-position risk signal; it does not create a future-event label.

Evaluate target calibration and ranking on a leakage-safe held-out source
before any game benchmark. A later game benchmark must compare old and new
GapNet against the same frozen, explicitly named opponent and fixed work. It
must not depend on the nonexistent `value_net_v2.npz`. Forty games remain a
diagnostic only, and no promotion command is frozen here.

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
| Gap net | Improve frozen held-out target metrics and pass a separately frozen paired game benchmark against the same named opponent; thresholds remain to be preregistered |

If any model fails its criteria, investigate before promoting.  A v2 that performs identically to
v1 is not worth the disruption — only promote if there is a real improvement.

---

## Step 6 — Promotion (not authorized)

Only after a separately recorded acceptance and product authorization may an
accepted Sentinel or GapNet v2 file be moved into its standard production
path. HumanPolicy has no production mapping in this plan. No executable
promotion command is recorded while the candidates and acceptance protocol
are undefined. Before any replacement, bind source and destination hashes,
make a recoverable backup, prove rollback, and use the repository's
destructive-action review rather than ad-hoc copy or rename commands.

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
- The `v2/` checkpoint directory already contains a partial earlier attempt.
  Inventory it before starting Stage 1, but do not adopt a Stage 5 checkpoint
  whose inputs exposed oracle fields.
- `bench_sentinel.py` hard-codes `data/gap_net.npz`.  Add a `--gap-net-path` CLI argument before
  running the gap net v2 comparison (Step 4d).
- The imported `tools/train_value_net_v2.py` and
  `scripts/eval_value_net_human.py` ideas are not approved implementation
  tasks. HumanPolicy work belongs to the v5 raw-game data contract.
- Sentinel Stage 5 is rejected for the DB-free runtime contract. Do not select
  an epoch or checkpoint from that lineage for v2 promotion.
- Both Sentinel trainers and the offline evaluator now share and enforce the
  same DB-only feature-slot mask. The legacy `--drop-db-features` flag is a
  provenance marker rather than an opt-in safety boundary.
- `build_gap_dataset.py` now treats `--sentinel` as required and fails closed
  both at load/probe time and during per-position scoring.
- **Pre-flight: malom_wdl_after semantics.** The GapNet category filter depends
  on 'L' meaning the next player loses (i.e. the human's move was winning) and
  'W' meaning the human played a losing move. Fifteen staged successor rows
  matched the corrected Malom adapter for
  W/D/L and DTW. Preserve this convention in focused dataset tests rather than
  relying only on the audit sample.
- **Pre-flight: frozen opponent identity.** A future GapNet game benchmark
  must name and hash the same opponent for old/new comparisons. It must not
  infer an opponent from a default production path.
