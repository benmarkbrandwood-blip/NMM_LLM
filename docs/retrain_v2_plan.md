# Retrain v2 Plan — Sentinel, ValueNet, HumanPrefNet, GapNet

## Motivation

The v1 sentinel, value net, and gap net were trained before the Malom sector-offset decoder
bug was fixed. That fix lands correct WDL labels across both DBs:

- **Specialist DB** — 364,262 stale Malom labels cleared, `sector-corrected-v1` stamped; correct
  labels accumulate as training runs proceed.
- **Human DB** — full `--rebuild` with the corrected decoder completed 2026-07-21; all
  2,167,498 positions carry correct Malom WDL labels (100% coverage, 0 unknowns).

The **gap net** training dataset (`data/gap_net_training.npz`) reads `malom_wdl_after` directly
from `human_db`, so it was built on corrupt labels and must be regenerated. The gap dataset
builder also uses the sentinel to compute composite quality scores — so the gap dataset must be
rebuilt *after* sentinel v2 exists, to get the best possible signal.

The **sentinel** dataset reads Malom at feature-build time; retraining from scratch means all
examples are now correctly labelled.

The **value net** has underperformed historically. Root cause: the v1 label is *final game
outcome* — every position in a game gets the same {+1, 0, −1}, so a single tactical error 40 plies
before the loss trains the VN to score the losing position as +1. This retrain replaces the label
with per-position Malom WDL, which is now available for both DBs.

A **new HumanPrefNet** is added: a human-move ranker trained on `human_db` positions using a
pairwise ranking loss. Distinct from ValueNet — different semantics, different loss, different
production path. It has two uses:

1. **GapNet training signal + eval opponent**: positions where HumanPrefNet's top move disagrees
   with Malom-optimal are exactly the blunder-prone positions GapNet exists to exploit. Using
   HumanPrefNet as the eval opponent tests what GapNet is *for*, unlike the previous plan's
   "beat heuristic+VN80" test which measured GapNet against a non-human opponent.
2. **Human-like play mode**: an opt-in inference setting that blends HumanPrefNet ranking into
   the heuristic's move ordering. We accept weaker absolute play in exchange for more
   recognisably human moves — this is the point of the mode.

---

## Naming conventions

During training, v2 artifacts are kept under distinct names to avoid clobbering production. Once
confirmed better (Step 8), they are renamed in place. HumanPrefNet is a **new artifact** that
does not overwrite anything from v1.

| Artifact | During training (v2 name) | After promotion (production name) |
|---|---|---|
| Sentinel checkpoint | `learned_ai/sentinel/checkpoints/v2/best.pt` | `learned_ai/sentinel/checkpoints/best.pt` |
| Sentinel stage dirs | `checkpoints/v2_stage{1,2,4}/` | *(staging only)* |
| ValueNet | `data/value_net_v2.npz` | `data/value_net.npz` |
| HumanPrefNet | `data/human_pref_net.npz` | `data/human_pref_net.npz` *(new file, no rename)* |
| GapNet | `data/gap_net_v2.npz` | `data/gap_net.npz` |
| Gap training data | `data/gap_net_training_v2.npz` | *(build artefact)* |

**Do not overwrite production files** (`best.pt`, `value_net.npz`, `gap_net.npz`) during training.
Renaming to production is a deliberate final step (Step 8).

**Ordering constraint (important)**: the plan trains models in a specific order because each
depends on the previous:

```
Sentinel v2  →  ValueNet v2  →  HumanPrefNet  →  GapNet v2
```

- Sentinel must be trained first because the gap dataset builder uses it.
- ValueNet is next because its Malom-WDL retraining is independent of HumanPrefNet.
- HumanPrefNet is next because both the gap dataset build (Step 4) and the GapNet eval
  (Step 6d) consume its signal.
- GapNet is last because its dataset and eval both depend on all three predecessors.

---

## Step 0 — Train Sentinel v2 (Stages 1 → 2 → 4)

A shell wrapper `scripts/train_sentinel_v2_step0.sh` runs all three stages in
sequence with the correct resume chain and promotes the final checkpoint into
`checkpoints/v2/best.pt`.  It is idempotent — delete any `v2_stage{N}/`
directory to force that stage to rerun.  Environment overrides
(`MALOM_DB`, `DEVICE`, `GAME_DIR`, `HUMAN_GAME_DIR`) let you retarget without
editing the script.

```bash
./scripts/train_sentinel_v2_step0.sh                # cpu, default paths
DEVICE=cuda ./scripts/train_sentinel_v2_step0.sh    # on a gpu box
```

The individual commands below run the same three stages if you prefer to launch
them one at a time.


**Stage 3 is permanently archived** (feature leakage); do not run it.

**Stage 5 is dropped from this retrain cycle.** Live sentinel inference always zeroes the oracle
(DB-derived) input slots to match runtime realism (see `eval_sentinel_db.py` in Step 6a). Stage 5
trains with those slots visible, which teaches the model to trust signals that will always be
zeroed in production — a real train/serve skew. If a future offline-analysis pipeline ever runs
sentinel with DB slots populated, a Stage 5 fine-tune can be added then under its own checkpoint
name; do not add it now.

The `v2/` checkpoint directory already exists from a prior partial run. Use `v2_stage{N}/`
sub-directories. If a usable Stage 4 checkpoint already exists at `checkpoints/v2/best.pt`,
evaluate it first (Step 6a) before retraining.

### Stage 1 — Structural foundation

No DB labels. `--drop-db-features` zeroes oracle slots so the model learns structural patterns.

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

Resume from Stage 1. DB provides strong WDL + DTM labels; `--drop-db-features` still zeroes
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

Resume from Stage 2. DB labels with `--drop-db-features`. Human game data included.

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

Stage 6 (AIDB + contrastive) is out of scope for this retrain cycle.

After Stage 4 completes, copy the best checkpoint to the v2 slot but **do not** overwrite the
production `best.pt` yet:

```bash
cp learned_ai/sentinel/checkpoints/v2_stage4/best.pt \
   learned_ai/sentinel/checkpoints/v2/best.pt
```

---

## Step 1 — Retrain ValueNet v2 on Malom WDL labels

The v1 ValueNet regresses final game outcome ({+1, 0, −1} of the entire game) against every
position in that game. This is a very noisy per-position signal — a single tactical error 40 plies
before a loss trains the net to think the losing position is `+1`. This is the load-bearing cause
of v1 VN underperformance.

**New training target: Malom WDL of the position itself.**

Both `specialist_db.sqlite` and `human_db.sqlite` now carry correct Malom WDL labels
(sector-corrected-v1). Every position with a WDL label becomes a training example:

```
label = +1 if malom_wdl == 'W' (side to move wins)
        0 if malom_wdl == 'D'
       −1 if malom_wdl == 'L'
```

This preserves the ValueNet semantic contract (scalar position value in `[−1, 1]`), so all
existing blender code (`ai/game_ai.py:1617-1619`, difficulty settings, VN blend %) continues to
work unchanged. Only the label semantics change; the output range does not.

**Training**:

- Loss: **MSE regression** against the {+1, 0, −1} Malom target.
- Optional aux head: **DTM regression** (distance-to-mate) to provide gradient between W-vs-W and
  L-vs-L positions where WDL alone is uninformative. Skip if it complicates the numpy MLP —
  Malom WDL alone is the primary signal.
- Split: game-level 80/20 (hold out whole games, not positions).
- Early stopping on val MSE with patience 10 epochs.

**New script**: `tools/train_value_net_v2.py`. The existing `train_value_net.py` reads JSONL
game records with final-outcome labels; the new script queries `human_db.sqlite` and
`specialist_db.sqlite` for `(state_key, malom_wdl)` pairs, reconstructs boards, extracts
79-float features, and regresses to the {+1, 0, −1} target.

```bash
.venv/bin/python tools/train_value_net_v2.py \
  --human-db data/human_db.sqlite \
  --specialist-db data/specialist_db.sqlite \
  --output data/value_net_v2.npz \
  --epochs 200 --patience 10
```

Note: `scripts/train_vn_trajectory.py` already trains a *phase-specific compound-label* VN. That
experiment is separate from this plan and its checkpoints are not touched.

---

## Step 2 — Train HumanPrefNet

**Purpose**: rank moves by human likelihood, given a position. Distinct model from ValueNet.

**Training data**: `human_db.sqlite`, grouped by `state_key`. For each unique position:

- Apply a per-position quality filter to avoid training on blunder positions:
  - If any recorded human move from this `state_key` has `malom_wdl_after = 'L'` (winning move
    played at least once), include only records where the human's chosen move is also `'L'`.
  - If no winning move recorded, accept records where the chosen move is `'D'` (draw was best
    available).
  - Skip any record where `malom_wdl_after = 'W'` (human played a losing move).

For each qualifying `(position, chosen_move)` pair, enumerate all legal moves from the position,
apply each to get successors, and extract 79-float features per successor.

**Loss**: **pairwise sigmoid BCE (Bradley-Terry) ranking**:

```
loss = − Σ log σ ( h_θ(chosen_successor) − h_θ(other_successor) )
```

for each `(chosen, other)` legal-move pair. The net outputs a scalar per successor; only relative
ordering is trained — absolute magnitudes are unconstrained.

**New script**: `tools/train_human_pref_net.py`. Uses PyTorch (or a rewritten numpy MLP with
pairwise loss).

```bash
.venv/bin/python tools/train_human_pref_net.py \
  --db data/human_db.sqlite \
  --output data/human_pref_net.npz \
  --patience 10
```

**Pre-flight**: spot-check `malom_wdl_after` semantics before writing the script — confirm 'L'
means the *next player loses* (i.e. the human's move was winning) and 'W' means the human's move
was losing. Convention has flipped in the codebase before.

---

## Step 3 — Wire HumanPrefNet into game inference

Both Step 4 (gap dataset build) and Step 6d (GapNet human-proxy eval) need HumanPrefNet reachable
at inference time. Do this before regenerating the gap dataset.

**Two integration points:**

### 3a. Advisor class

Create `ai/human_pref_advisor.py` (or add to `ai/value_net.py`) exposing:

```python
class HumanPrefAdvisor:
    def __init__(self, npz_path: str): ...
    def rank(self, board: BoardState, legal_moves: list[dict]) -> list[float]:
        """Return one scalar per move; larger = more human-likely."""
    def probs(self, board: BoardState, legal_moves: list[dict]) -> np.ndarray:
        """Softmax over rank() outputs — for sampling in human-play mode."""
```

Because HumanPrefNet outputs an unbounded scalar, `probs()` should use a temperature-scaled
softmax; expose the temperature as a param on the advisor.

### 3b. Human-like play mode in `GameAI`

Add to `AIWeights` (`ai/heuristics.py`):

```python
humanlike_blend: int = 0        # 0 = pure heuristic, 100 = pure human_pref sampling
humanlike_temperature: float = 1.0  # softmax temperature for probs()
```

Add to `GameAI.__init__`:

```python
human_pref_net = None    # HumanPrefAdvisor | None
```

Add a leaf-level blend in `_apply_value_net_score`-style pattern:

```python
def _apply_humanlike_adjust(self, scored, board):
    if self._human_pref_net is None or self._weights.humanlike_blend == 0:
        return scored
    α = self._weights.humanlike_blend / 100.0
    probs = self._human_pref_net.probs(board, [m for m, _ in scored])
    # Map probs to the same integer scale as heuristic scores
    hp_scores = [int(p * _VN_SCALE) for p in probs]
    return [(m, int((1 - α) * raw + α * hp)) for (m, raw), hp in zip(scored, hp_scores)]
```

Apply after opening adjustments and before VN blending. **Do not** clip terminal scores — a mate
move should remain mate regardless of humanlike setting.

**We accept weaker play from this mode.** The heuristic AI's job here is to *look like a human*,
not to play optimally. The primary success criterion is move-distribution match (Step 6c), not
game win rate.

**UI exposure (optional)**: expose `humanlike_blend` as a slider in the web UI difficulty panel
(“Human-like play — 0–100%”). Low difficulties can default `humanlike_blend` above zero if you
want difficulty-3 games to feel more human.

---

## Step 4 — Regenerate GapNet training dataset

Now that Sentinel v2 and HumanPrefNet both exist, rebuild the gap dataset with corrected Malom
labels, the better sentinel signal, and an additional human-preference signal.

**Label enrichment**: at each qualifying position, compute

```
hp_disagreement = | hp_top_move_score − malom_optimal_move_score |
```

as an auxiliary target. Positions where HumanPrefNet's top move diverges strongly from Malom's
optimal move are exactly the blunder-prone positions GapNet exists to exploit — direct signal,
better than aggregate `best_q − freq-weighted_human_q` alone.

**Feature enrichment (recommended)**: add mill-density, mill-threat count, phase indicator, and
recent-mill-count features to the 79-float input (or a small pre-appended vector). Cheap, and
directly encodes patterns humans miss.

```bash
.venv/bin/python scripts/build_gap_dataset.py \
  --sentinel-ckpt learned_ai/sentinel/checkpoints/v2/best.pt \
  --human-pref-ckpt data/human_pref_net.npz \
  --out data/gap_net_training_v2.npz \
  --samples-per-category 20000
```

Verify `build_gap_dataset.py` accepts `--sentinel-ckpt` and `--human-pref-ckpt`; add the flags
before running.

---

## Step 5 — Train GapNet v2

```bash
.venv/bin/python tools/train_gap_net.py \
  --data data/gap_net_training_v2.npz \
  --out  data/gap_net_v2.npz \
  --epochs 80
```

---

## Step 6 — Testing protocol

### 6a. Sentinel — Malom DB correlation eval

New script: `scripts/eval_sentinel_db.py`. Sample ≥500 positions from each of `specialist_db` and
`human_db` (stratified by phase). For each: reconstruct board, query Malom for per-move WDL+DTM,
run sentinel with **DB slots zeroed** (matches live inference), score all legal moves. Report
`win_acc`, `loss_acc`, `top1_win_rate`, `spearman_r`, `dtm_pearson_r`, phase breakdown. Run for
v1 and v2 to compare.

### 6b. Sentinel — game bench (v2 vs v1 vs baseline)

```bash
.venv/bin/python tools/bench_sentinel_v2.py \
  --diff 5 --budget 3.0 --games-per-pair 40 \
  --old-ckpt learned_ai/sentinel/checkpoints/best.pt \
  --new-ckpt learned_ai/sentinel/checkpoints/v2/best.pt \
  --out eval_sentinel_v2_roundrobin.json
```

### 6c. ValueNet — held-out MSE + game bench + blunder rate

**Held-out MSE (primary):** hold out a game-level 20% split; measure MSE vs Malom WDL and
sign-accuracy (`sign(vn.predict) == sign(malom_wdl)`). Compare v1 vs v2 on the same held-out
set.

**Game bench at multiple blends**: bench v2 VN at blends **30, 60, 80** against the base
heuristic. A v2 that wins at blend=30 but regresses at blend=80 is a calibration failure, not a
real win. All three blends must be non-losing for promotion.

```bash
.venv/bin/python tools/bench_trajectory_value_net.py \
  --vn-path data/value_net_v2.npz \
  --blends 30 60 80 \
  --games 40
```

**Blunder rate**: run VN-only self-play (blend=100) for 100 games; count Malom-losing moves per
100 moves. v2 must be lower than v1.

### 6d. HumanPrefNet — assessment

Held-out game-level 20% split.

- **Top-1 / top-3 / top-5 next-move accuracy**: primary metric.
- **Spearman r**: for positions with multiple recorded human moves, correlate predicted ranking
  with actual play frequency.
- **Elo-strata check**: report top-1 accuracy on high-Elo games separately from bulk. If accuracy
  drops sharply on high-Elo, HumanPrefNet is capturing average-noise, not human skill.
- **Sanity: AI-move-pruner bench**: an AI that drops from its consideration any move HumanPrefNet
  scores below the 10th percentile (never played by humans) should not lose more than 5pp win
  rate vs unpruned. Confirms the model captures reasonable play, not just a narrow subset.

```bash
.venv/bin/python scripts/eval_human_pref_net.py \
  --db data/human_db.sqlite \
  --net data/human_pref_net.npz \
  --out eval_human_pref_net.json
```

### 6e. GapNet — human-proxy opponent bench

**This replaces the v1 plan's "beat heuristic+VN80" test.** VN80 is not a human, so it did not
test what GapNet is for.

Configure a **human-proxy opponent**: a `GameAI` with `humanlike_blend=100` — its move selection
samples from HumanPrefNet's softmax over legal moves. AI-with-GapNet and AI-without-GapNet each
play 40 games (20 per colour) against this proxy. Difficulty 5, VN blend 60 (v2 VN).

```bash
# GapNet v1 vs human proxy
.venv/bin/python scripts/bench_sentinel.py \
  --games 40 --difficulty 5 \
  --white-gap-net \
  --black-humanlike --humanlike-blend 100

# GapNet v2 vs human proxy
.venv/bin/python scripts/bench_sentinel.py \
  --games 40 --difficulty 5 \
  --white-gap-net --gap-net-path data/gap_net_v2.npz \
  --black-humanlike --humanlike-blend 100
```

Winner: higher win rate vs the human proxy. `bench_sentinel.py` needs `--gap-net-path`,
`--humanlike-blend`, and `--vn-path` CLI flags added before this can run.

---

## Step 7 — Success criteria for promotion

Promote v2 to replace v1 only if **all** conditions hold:

| Model | Condition |
|---|---|
| Sentinel | 6a: `win_acc` ≥ v1 + 3pp AND `top1_win_rate` ≥ v1 + 3pp |
| Sentinel | 6b: NewS20 vs Base ≥ 50% win rate AND NewS20 vs OldS20 within ±3pp |
| ValueNet | 6c held-out: sign accuracy ≥ v1 + 3pp on Malom WDL test set |
| ValueNet | 6c game bench: non-losing at blends 30 / 60 / 80 (each ≥ 50% vs base) |
| ValueNet | 6c blunder rate: v2 Malom-losing moves ≤ v1 |
| HumanPrefNet | 6d: top-1 accuracy ≥ 40% (top-3 ≥ 65%) — set floor; no v1 to compare |
| HumanPrefNet | 6d Spearman r ≥ 0.5 on multi-move positions |
| HumanPrefNet | 6d Elo-strata: top-1 accuracy on high-Elo games within 3pp of bulk |
| HumanPrefNet | 6d prune bench: pruned AI within 5pp of unpruned |
| GapNet | 6e: v2 win rate vs human proxy ≥ v1 win rate vs same |

Any model failing its criteria: investigate before promoting. A v2 that performs identically to
v1 is not worth the disruption — only promote on real improvement.

---

## Step 8 — Promotion

```bash
# Back up v1 production files
cp learned_ai/sentinel/checkpoints/best.pt \
   learned_ai/sentinel/checkpoints/best-v1-$(date +%Y%m%d).pt
cp data/value_net.npz  data/value_net_v1_$(date +%Y%m%d).npz
cp data/gap_net.npz    data/gap_net_v1_$(date +%Y%m%d).npz

# Rename v2 into production
mv learned_ai/sentinel/checkpoints/v2/best.pt \
   learned_ai/sentinel/checkpoints/best.pt
mv data/value_net_v2.npz  data/value_net.npz
mv data/gap_net_v2.npz    data/gap_net.npz
# HumanPrefNet is a new artifact; no rename needed — data/human_pref_net.npz is its final path
```

Restart the Flask server to pick up the new sentinel checkpoint. No config change needed —
paths in `data/training_paths.local.json` are unchanged.

---

## Step 9 — Rollback

```bash
mv learned_ai/sentinel/checkpoints/best-v1-YYYYMMDD.pt \
   learned_ai/sentinel/checkpoints/best.pt
mv data/value_net_v1_YYYYMMDD.npz  data/value_net.npz
mv data/gap_net_v1_YYYYMMDD.npz    data/gap_net.npz
```

HumanPrefNet is additive — leave `data/human_pref_net.npz` in place. Setting
`humanlike_blend=0` disables it at inference.

---

## Known issues / caveats

- `specialist_db.sqlite` positions use `pos_hash` (SHA-1 of canonical FEN, not reversible). If
  no raw FEN column is present, ValueNet training must fall back to replaying JSONL games to
  reconstruct positions and cross-reference the Malom label. Check schema before writing
  `train_value_net_v2.py`.
- The `v2/` checkpoint directory already contains a partial earlier attempt. Inspect before
  starting Stage 1; if it holds a usable Stage 4 checkpoint, evaluate first before retraining.
- `bench_sentinel.py` hard-codes `data/gap_net.npz` and does not support HumanPrefNet as an
  opponent. Add `--gap-net-path`, `--vn-path`, `--humanlike-blend`, and a `--black-humanlike`
  side flag before running Step 6e.
- New scripts required: `tools/train_value_net_v2.py` (Step 1), `tools/train_human_pref_net.py`
  (Step 2), `ai/human_pref_advisor.py` (Step 3), `scripts/eval_sentinel_db.py` (Step 6a),
  `scripts/eval_human_pref_net.py` (Step 6d).
- **Pre-flight — `malom_wdl_after` semantics**: confirm 'L' = next player loses (human's move was
  winning) and 'W' = human played a losing move, by spot-checking a sample from `human_db`
  before writing `train_human_pref_net.py`.
- **HumanPrefNet scale**: pairwise ranking loss does not constrain absolute output magnitude.
  The `HumanPrefAdvisor.probs()` softmax temperature will need tuning — expose it as a param.
- **Human-play mode is intentionally weaker.** Do not treat lower absolute win rate as a
  regression when `humanlike_blend > 0`.

---

## Reconsider later

Some things were built **before** the first end-to-end retrain run.  Revisit them
after real training data is in, and change the design if that data justifies it:

- **`bench_sentinel.py` humanlike / gap / vn flag surface (Step 6e).**  Landed
  ahead of any real v2 checkpoints so the human-proxy bench command matches
  the plan.  When actual gap_net_v2.npz + human_pref_net.npz + value_net_v2.npz
  land, sanity-check that:
  1. The flag names + defaults still make sense as an operator-facing surface.
  2. `humanlike_blend=100` produces the intended "sampled human proxy"
     behaviour or whether a lower default (e.g. 60) reads more like a plausible
     human at difficulty 5.
  3. Whether the proxy should also carry `--vn-blend` for a more realistic
     opponent (humans use engines too), or stay purely HumanPrefNet-driven.
  4. Whether a `--suite` preset for the Step 6e round-robin is worth adding
     alongside the current single-matchup mode.
- **HumanPrefNet softmax temperature.**  Currently a plain float arg
  (`humanlike_temperature`, default 1.0).  After the first real HP checkpoint,
  inspect the raw score distribution and either (a) auto-set a sensible default
  based on the training-time score variance, or (b) publish a recommended value
  per game phase.
- **HumanPrefNet weights loader location.**  `HumanPrefLoader` in
  `scripts/build_gap_dataset.py` is kept as a backwards-compat subclass of
  `HumanPrefAdvisor`.  Once every downstream caller has been migrated to the
  `ai/` module, delete the shim.
- **Malom labels on synthetic-fallback gap-dataset positions.**  Positions
  generated by `_generate_synthetic_positions` get NaN `y_hp` because they
  have no HumanPrefNet-vs-Malom pairing signal by construction.  If synthetic
  samples become a large fraction of the v2 dataset, consider stripping them
  or scoring them against HumanPrefNet directly instead of leaving NaN.
