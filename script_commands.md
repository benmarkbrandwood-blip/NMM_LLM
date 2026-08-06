# NMM Script Command Reference

## Board Layout

```
a7───────────d7───────────g7
│            │            │
│   b6───────d6──────f6  │
│   │         │         │  │
│   │   c5───d5───e5   │  │
│   │   │           │  │  │
a4──b4──c4       e4──f4───g4
│   │   │           │  │  │
│   │   c3───d3───e3   │  │
│   │         │         │  │
│   b2───────d2──────f2  │
│            │            │
a1───────────d1───────────g1
```

## Rebuild Rust Engine

```
bash scripts/build_rust.sh
```

Run whenever `native/nmm_core/src/` files change.

## Self-Play Game Generation

```
python tools/self_play.py --games 500 --no-llm --white 7 --black 3 --parallel 4  \
  --game-dir data/games/self_play --random-difficulty
```

| Flag | Default | Description |
| - | - | - |
| `--games N` | 10 | Number of games to play |
| `--white D` | 5 | White difficulty (1–10) |
| `--black D` | 5 | Black difficulty (1–10) |
| `--parallel N` | 1 | Parallel workers |
| `--game-dir PATH` | — | Output directory for game JSONL files |
| `--random-difficulty` | off | Randomise difficulty each game within range |
| `--min-difficulty D` | 1 | Lower bound for random difficulty |
| `--max-difficulty D` | 9 | Upper bound for random difficulty |
| `--blunder P` | 0.0 | Probability of blunder move |
| `--no-llm` | off | Skip LLM commentary |
| `--swap` | off | Alternate which side plays first |
| `--personalities LIST` | — | Comma-separated personality names |
| `--white-personality NAME` | — | Specific personality for White |
| `--black-personality NAME` | — | Specific personality for Black |
| `--verbose` | off | Per-game status lines |


## HumanPrefNet — Train (train_human_pref_net.py)

Implements Step 2 of `docs/retrain_v2_plan.md`. Trains a move-ranking net on
human_db moves using pairwise Bradley-Terry (sigmoid BCE) loss:
`L = − log σ(h(chosen_successor) − h(other_successor))`.

**Per-position filter** (avoids training on positions humans blundered): keep
records with `malom_wdl_after='L'` (human's move was winning) when any exist
for this state; else `='D'` records; always skip `='W'` records.

Output: PyTorch weights saved as `data/human_pref_net.npz` — layers stored as
`w{i}, b{i}` plus `input_dim` / `layer_count` metadata so a lightweight numpy
loader can drive inference without a torch dependency.

**Full run:**

```
.venv/bin/python tools/train_human_pref_net.py \
  --db data/human_db.sqlite \
  --output data/human_pref_net.npz \
  --patience 10 --batch-size 4096
```

**Smoke test (500 positions, 3 epochs — CPU or GPU):**

```
.venv/bin/python tools/train_human_pref_net.py \
  --limit 500 --epochs 3 --output /tmp/hp_smoke.npz
```

| Flag | Default | Description |
| --- | --- | --- |
| `--db PATH` | `data/human_db.sqlite` | Source database (`moves` table joined against `positions`). |
| `--output PATH` | `data/human_pref_net.npz` | Output NPZ. |
| `--epochs N` | 100 | Max epochs (early stopping usually terminates sooner). |
| `--patience N` | 10 | Early-stop patience — epochs without val-loss improvement. |
| `--lr F` | 3e-4 | Learning rate. |
| `--batch-size N` | 512 | Mini-batch size. See note below on picking a larger value for full-DB runs. |
| `--val-fraction F` | 0.20 | Held-out fraction for val + early stop. |
| `--pairs-per-position N` | 4 | Cap on (chosen, other) pairs sampled per qualifying position. |
| `--limit N` | — | Cap positions loaded (for smoke tests). |
| `--seed N` | 42 | RNG seed. |

**Memory model**: the trainer keeps the full pair arrays on CPU and moves
only the current batch to GPU each step.  Peak GPU memory scales with
`--batch-size`, not dataset size — an 8 GB GPU handles the full 2M-position
run at any reasonable batch size.  Default 512 is safe on any hardware;
bump to 4096 or 8192 on a modern GPU to cut per-epoch iteration count 8–16×
(per-batch payload at 8192 is still ~2.6 MB per side).

Ranking accuracy (val) is the primary metric: fraction of `(chosen, other)`
pairs where `h(chosen) > h(other)`. The smoke test above reaches ~0.68 on 500
positions in 3 epochs; a full run should climb well past that.

Not usable at inference until Step 3 wires `ai/human_pref_advisor.py` into
`GameAI` (both for "human-like play" mode and as the human-proxy opponent in
the Step 4 GapNet dataset build).


## Retrain v2 — Testing Protocol (Step 6)

Four evaluation scripts implement the plan's testing matrix. None of them
touch production files; they all read from `data/human_db.sqlite` and (for
Sentinel) the Malom DB.

### 6a — Sentinel DB correlation (eval_sentinel_db.py)

Draws an **independent** JSONL + Malom sample with the same 60/40 composition
the training builder uses (`scripts/build_sentinel_dataset_v2.py`), but with
a different default RNG seed (99999 vs the builder's 42) so overlap with the
training set is minimised.  Reconstructs each board, queries Malom per move,
runs the sentinel with DB slots ZEROED (matches live inference), and reports:

- `win_acc`, `loss_acc`, `top1_win_rate`, `spearman_r`, `dtm_pearson_r`
- `phase_breakdown` — same four metrics split by place / move / fly
- `source_breakdown` — same four metrics split into **Malom-source** and
  **JSONL-source** sub-populations.  This is the contamination fence: the
  Malom-source rows may overlap the training pool at the state-key level,
  whereas the JSONL-source rows come from random game files that are unlikely
  to have been fully consumed during training.  A large gap between the two
  suggests memorisation; near-identical numbers suggest genuine generalisation.

specialist_db is deliberately skipped (irreversible `pos_hash`).

```bash
# v1 vs v2 comparison
.venv/bin/python scripts/eval_sentinel_db.py \
  --checkpoint learned_ai/sentinel/checkpoints/best.pt \
  --output eval_sentinel_db_v1.json --n-samples 1000

.venv/bin/python scripts/eval_sentinel_db.py \
  --checkpoint learned_ai/sentinel/checkpoints/v2/best.pt \
  --output eval_sentinel_db_v2.json --n-samples 1000

# Repeat 2-3 times with different --seed to gauge sampling variance.
.venv/bin/python scripts/eval_sentinel_db.py \
  --checkpoint learned_ai/sentinel/checkpoints/v2/best.pt \
  --output eval_sentinel_db_v2_seed2.json --n-samples 1000 --seed 12345
```

| Flag | Default | Description |
| --- | --- | --- |
| `--checkpoint PATH` | *required* | Sentinel `.pt` to evaluate |
| `--human-db PATH` | `data/human_db.sqlite` | State-key source for the Malom-sample portion |
| `--malom-db PATH` | `/mnt/windows/.../Std_DD_89adjusted` | Malom DB directory (per-move WDL+DTM ground truth) |
| `--n-samples N` | 1000 | Total positions across both sources |
| `--jsonl-fraction F` | 0.40 | Fraction of the sample drawn from JSONL replay (matches training builder) |
| `--game-dir PATH` | `data/games` | AI self-play JSONL for the JSONL-source portion |
| `--human-game-dir PATH` | `data/human_games` | Human JSONL for the JSONL-source portion |
| `--output PATH` | — | Optional JSON summary path |
| `--seed N` | 99999 | RNG seed; deliberately different from the training builder's 42 |

### 6b — Sentinel game bench (bench_sentinel_v2.py)

Existing script — round-robin between old and new sentinel checkpoints at
20% / 30% gap thresholds vs base heuristic.

```bash
.venv/bin/python tools/bench_sentinel_v2.py \
  --diff 5 --budget 3.0 --games-per-pair 40 \
  --old-ckpt learned_ai/sentinel/checkpoints/best.pt \
  --new-ckpt learned_ai/sentinel/checkpoints/v2/best.pt \
  --out eval_sentinel_v2_roundrobin.json
```

### 6c — ValueNet held-out MSE + sign accuracy (eval_value_net_v2.py)

Deterministic SHA-256 hash of the state_key drives a shared 20% held-out
split; the same split is used across all `--net` entries so v1 vs v2
metrics are directly comparable.

```bash
.venv/bin/python scripts/eval_value_net_v2.py \
  --net data/value_net.npz    --net-name v1 \
  --net data/value_net_v2.npz --net-name v2 \
  --output eval_vn_v2_holdout.json
```

| Flag | Default | Description |
| --- | --- | --- |
| `--human-db PATH` | `data/human_db.sqlite` | Position source |
| `--net PATH` | *required, repeatable* | ValueNet `.npz` to evaluate |
| `--net-name NAME` | *matches `--net` order* | Optional label per net |
| `--val-fraction F` | 0.20 | Held-out fraction (matching v2 training split rule) |
| `--limit N` | — | Cap positions read (smoke tests) |
| `--output PATH` | — | Optional JSON summary path |

For the game-bench sanity check (blends 30 / 60 / 80), use
`tools/bench_trajectory_value_net.py`:

```bash
.venv/bin/python tools/bench_trajectory_value_net.py \
  --vn-path data/value_net_v2.npz --blends 30 60 80 --games 40
```

### 6d — HumanPrefNet held-out top-K + Spearman (eval_human_pref_net.py)

Same per-position filter as the training script — keeps records with
`malom_wdl_after='L'` when any exist for a state_key, else `='D'`, always
skips `='W'`.  Split is deterministic hash by state_key so it's stable
across runs.

```bash
.venv/bin/python scripts/eval_human_pref_net.py \
  --net data/human_pref_net.npz \
  --output eval_human_pref_net.json
```

Reports:

- `top1_acc` / `top3_acc` / `top5_acc` — does the human's actual highest-
  frequency move fall in the top-K HumanPrefNet ranking?
- `spearman_multi` — for positions with more than one recorded human move,
  Spearman r between the HP-net ranking and observed play frequency.

Elo-strata top-1 and the AI-move-pruner bench remain in the plan as follow-
ups; they need per-game Elo tracking (would require a `human_db` rebuild)
and a game-bench runner respectively.

### 6e — GapNet human-proxy opponent bench

Both prerequisites have landed: Step 3 wires `HumanPrefAdvisor` into
`GameAI`, and `scripts/bench_sentinel.py` now exposes `--gap-net-path`,
`--vn-path`, `--white-humanlike`, `--black-humanlike`, `--humanlike-blend`,
`--humanlike-temperature`, and `--human-pref-path` flags.  See the
"Benchmarking — Sentinel / GAP Net / Tournament" section below for the
full flag table and the Step 6e command.

The plan's Reconsider-later section (`docs/retrain_v2_plan.md`) tracks the
follow-up items to revisit once real v2 checkpoints exist — flag defaults,
softmax temperature calibration, and whether a `--suite` preset is worth
adding for this round-robin.


## Retrain v2 — Step 3: HumanPrefNet at inference

Two pieces of plumbing land in this step (see docs/retrain_v2_plan.md § 3):

**Advisor class** — `ai/human_pref_advisor.py`

```python
from ai.human_pref_advisor import HumanPrefAdvisor, try_load

adv = HumanPrefAdvisor("data/human_pref_net.npz", temperature=1.0)
scores = adv.rank(board, legal_moves)        # unbounded scalars, higher = more human-likely
probs  = adv.probs(board, legal_moves)       # softmax; use for sampling in humanlike mode

# Graceful "off" if the file is absent:
adv = try_load("data/human_pref_net.npz")    # returns None on missing / bad file
```

Pure-numpy forward pass — no torch import at inference.  Reads the same NPZ
layout emitted by `tools/train_human_pref_net.py` (`w0/b0…w{n-1}/b{n-1}`
plus `input_dim` / `layer_count` metadata).

**HeuristicWeights fields** — `ai/heuristics.py`

```python
humanlike_blend: int = 0        # 0 = pure heuristic; 100 = pure HumanPrefNet
humanlike_temperature: float = 1.0
```

**GameAI wiring** — `ai/game_ai.py`

- New `human_pref_net=` parameter on `GameAI.__init__`.  Pass `try_load(...)`
  from your loader; leave as `None` to disable.
- New `_apply_humanlike_adjust(scored, board)` method blends
  `hp_prob × _VN_SCALE` into each move's leaf score with weight
  `humanlike_blend / 100`.  Called between opening adjustments and VN
  blending.  Mate / terminal scores (|raw| ≥ 5,000,000) are preserved
  untouched.

**Human-like play mode is intentionally weaker.**  The whole point is
move-distribution match; do not treat a lower absolute win rate as a
regression when `humanlike_blend > 0`.

**Web GUI exposure.**  The Settings modal exposes a "Human-like play"
checkbox and a "Strength %" slider directly below the "Gap exploitation"
group.  Toggling the checkbox sends `use_humanlike` + `humanlike_blend`
on every WebSocket play request; the server overrides
`HeuristicWeights.humanlike_blend` from those fields before constructing
`GameAI`.  Disabling the checkbox forces `humanlike_blend = 0` regardless
of the slider value.  Slider defaults to 50 %; unchecked by default.
The AI Tuning panel no longer carries `humanlike_blend` — the toggle
lives in Settings because it changes the AI's *style*, not its
tuning constants.





## Value Net v2 — Malom WDL Training (train_value_net_v2.py)

Implements Step 1 of `docs/retrain_v2_plan.md`. Trains ValueNet on per-position
Malom WDL labels ({+1, 0, −1}) from `human_db.sqlite` rather than the v1
final-outcome-per-game approach. Output range stays `[−1, 1]` — existing
blender code, VN blend %, and difficulty settings work unchanged.

`specialist_db.sqlite` is **not** used (positions keyed by irreversible
`pos_hash`). The `--specialist-db` flag is accepted for compatibility with the
plan command but has no effect.

**Full run:**

```
.venv/bin/python tools/train_value_net_v2.py \
  --human-db data/human_db.sqlite \
  --specialist-db data/specialist_db.sqlite \
  --output data/value_net_v2.npz \
  --epochs 200 --patience 10
```

**Smoke test (5k positions, 3 epochs):**

```
.venv/bin/python tools/train_value_net_v2.py \
  --limit 5000 --epochs 3 --output /tmp/vn_v2_smoke.npz
```

| Flag | Default | Description |
| --- | --- | --- |
| `--human-db PATH` | `data/human_db.sqlite` | Positions + `malom_wdl` label source. Query is `WHERE malom_wdl IS NOT NULL`. |
| `--specialist-db PATH` | — | Accepted for API compat; skipped with a warning. |
| `--output PATH` | `data/value_net_v2.npz` | Output `.npz` checkpoint. |
| `--epochs N` | 200 | Max epochs (early stopping usually terminates sooner). |
| `--patience N` | 10 | Early-stop patience — epochs without val-loss improvement. |
| `--lr F` | 1e-3 | Learning rate. |
| `--batch-size N` | 256 | Mini-batch size. |
| `--val-fraction F` | 0.20 | Held-out fraction for validation + early stop. |
| `--limit N` | — | Cap positions loaded (for smoke tests). |
| `--seed N` | 42 | RNG seed. |

**Not** promoted to production. Compare against `data/value_net.npz` per plan
Step 4c (held-out MSE + game bench at blends 30 / 60 / 80) before Step 8
promotion.


## Value Net — Basic Training (train_value_net.py)

```
.venv/bin/python tools/train_value_net.py  \
  --games-dir data/games --games-dir data/human_games  \
  --decisive-only --epochs 30 --output data/value_net.npz
```

| Flag | Default | Description |
| - | - | - |
| `--games-dir PATH` | — | Game directory (repeatable for multiple dirs) |
| `--output PATH` | — | Output .npz path |
| `--epochs N` | 30 | Training epochs |
| `--lr F` | 0.001 | Learning rate |
| `--batch-size N` | 256 | Mini-batch size |
| `--decisive-only` | off | Skip drawn games |


## Value Net — Human-Filtered V3 (train_value_net_filtered.py)

```
.venv/bin/python tools/train_value_net_filtered.py
```

| Flag | Default | Description |
| - | - | - |
| `--games-dir PATH` | `data/human_games` | Source game directory |
| `--output PATH` | auto | Output .npz path |
| `--epochs N` | 100 | Training epochs |
| `--lr F` | 3e-4 | Learning rate |
| `--batch-size N` | 256 | Mini-batch size |
| `--val-frac F` | 0.1 | Validation fraction |
| `--patience N` | 10 | Early-stop patience (epochs) |
| `--weight-decay F` | 1e-4 | L2 regularisation |
| `--placement-blend F` | 0.35 | Weight of placement-phase positions |
| `--heuristic-scale F` | auto | Scale for heuristic target labels |
| `--min-elo N` | 0 | Minimum player ELO filter |
| `--decisive-only` | off | Skip drawn games |


**Benchmark all nets once V3 is trained:**

```
.venv/bin/python tools/bench_vn_filtered.py --diff 4 --budget 3.0 --games-per-pair 10

# Longer run (200 games)
.venv/bin/python tools/bench_vn_filtered.py --diff 5 --budget 3.0 --games-per-pair 20
```

| Flag | Default | Description |
| - | - | - |
| `--diff D` | 4 | AI difficulty for benchmark games |
| `--budget F` | 3.0 | Per-move time budget (seconds) |
| `--games-per-pair N` | 10 | Games per config pair (must be even) |
| `--out PATH` | — | JSON results output path |


## Value Net — Phase Trajectory Training (train_vn_trajectory.py)

Trains THREE phase-specific value nets (placement / movement / fly) saved as `data/value_net_phase_place.npz`, `data/value_net_phase_move.npz`, `data/value_net_phase_fly.npz`. At inference the web app loads all three as a `PhaseValueNet` and dispatches `predict()` to the correct sub-net based on the game phase.

Reward per position: `malom_sign × best_composite_quality` where composite = 0.6 × sentinel + 0.4 × heuristic (same signals as GAP net, computed live per trajectory position). Winner moves are randomly sampled from ALL winning successors (not just best DTW).

**Full training run — all three phases (default settings):**

```
.venv/bin/python scripts/train_vn_trajectory.py
```

Defaults: 5000 starts · 40 epochs · traj depth 40 · sentinel loaded automatically · bench 2000 accuracy.

**Train only one phase:**

```
.venv/bin/python scripts/train_vn_trajectory.py --phase move
```

**Smaller run to check quality first:**

```
.venv/bin/python scripts/train_vn_trajectory.py --n-starts 500 --epochs 20
```

**Fine-tune from existing phase nets:**

```
.venv/bin/python scripts/train_vn_trajectory.py  \
  --continue-from data/value_net_phase  \
  --n-starts 2000 --epochs 20
```

**Benchmark only (no training):**

```
.venv/bin/python scripts/train_vn_trajectory.py  \
  --epochs 0 --bench-accuracy 2000 --bench-traj 300 --bench-games 50
```

| Flag | Default | Description |
| - | - | - |
| `--db PATH` | `data/human_db.sqlite` | Human DB for starting positions |
| `--malom-db PATH` | `.../Std_DD_89adjusted` | Malom DB directory |
| `--sentinel PATH` | `learned_ai/sentinel/checkpoints/best.pt` | Sentinel checkpoint for composite quality (falls back to heuristic-only if missing) |
| `--out PATH` | `data/value_net_phase` | Base output path (no extension); creates _place/_move/_fly.npz |
| `--phase PHASE` | all | Which phase(s) to train: `place`, `move`, `fly`, or `all` |
| `--n-starts N` | 5000 | Starting positions to sample from human_db |
| `--traj-depth N` | 40 | Max plies per trajectory |
| `--min-placed N` | 7 | Min total pieces placed in start position |
| `--bucket-cap N` | none | Max starts per placement-stage bucket |
| `--seed N` | 42 | RNG seed for winner-move random sampling |
| `--use-heuristic` | off | Use heuristic AI for loser moves (slower) |
| `--heuristic-difficulty D` | 4 | Heuristic AI difficulty (only with `--use-heuristic`) |
| `--heuristic-time F` | 0.05 | Time budget per AI move in seconds (only with `--use-heuristic`) |
| `--epochs N` | 40 | Training epochs |
| `--lr F` | 8e-4 | Learning rate |
| `--batch N` | 512 | Mini-batch size |
| `--continue-from PATH` | — | Base path of existing phase nets to fine-tune |
| `--bench-accuracy N` | 2000 | Positions for accuracy test (move-phase net) |
| `--bench-traj N` | 300 | Positions for trajectory-follow test |
| `--bench-games N` | 0 | Full games vs raw heuristic (0 = skip) |
| `--bench-gap` | off | Also benchmark GAP net for comparison |
| `--difficulty D` | 6 | Difficulty for full-game benchmark |
| `--vn-blend N` | 80 | VN blend % for full-game benchmark |
| `--time-budget F` | 0.5 | Per-move budget for full-game benchmark |


## Trajectory Value Net — Round-Robin Benchmark (bench_trajectory_value_net.py)

Tests five configs against each other: Baseline, TrajVN at 10/30/60% blend, and GapNet.

**Full round-robin (10 pairs):**

```
.venv/bin/python scripts/bench_trajectory_value_net.py --games 20 --difficulty 4
```

**Single matchup:**

```
.venv/bin/python scripts/bench_trajectory_value_net.py --matchup TrajVN-30% Baseline --games 40
```

**Custom blend percentages:**

```
.venv/bin/python scripts/bench_trajectory_value_net.py --blends 20 40 80 --games 20
```

| Flag | Default | Description |
| - | - | - |
| `--games N` | 20 | Games per matchup pair (must be even) |
| `--difficulty D` | 4 | AI difficulty for all configs |
| `--time-limit F` | — | Per-move time limit in seconds |
| `--vn-path PATH` | `data/value_net_trajectory.npz` | Trajectory value net path |
| `--gap-path PATH` | `data/gap_net.npz` | Gap net path |
| `--blends PCT…` | `10 30 60` | Value-net blend percentages to test |
| `--matchup A B` | — | Single matchup between two named configs instead of round-robin |


Configs: `Baseline`, `TrajVN-10%`, `TrajVN-30%`, `TrajVN-60%`, `GapNet`

## GAP Net — Build Dataset + Train

**Step 1 — Build training dataset:**

```
.venv/bin/python scripts/build_gap_dataset.py
```

For the v2 rebuild in `docs/retrain_v2_plan.md` (Step 4 — sentinel v2 signal
plus HumanPrefNet disagreement as an auxiliary label):

```
.venv/bin/python scripts/build_gap_dataset.py \
  --sentinel-ckpt learned_ai/sentinel/checkpoints/v2/best.pt \
  --human-pref-ckpt data/human_pref_net.npz \
  --out data/gap_net_training_v2.npz \
  --samples-per-category 20000
```

The output NPZ carries `X`, `y` (gap target), and (v2) `y_hp` — a per-position
`malom_top_q − malom_q_of_hp_top` value in `[0, 1]`.  Positions with no HP
signal (older datasets or synthetic-fallback samples) get NaN in `y_hp`.

| Flag | Default | Description |
| - | - | - |
| `--db PATH` | `data/human_db.sqlite` | Human DB source |
| `--sentinel PATH` | `learned_ai/sentinel/checkpoints/best.pt` | Sentinel checkpoint |
| `--sentinel-ckpt PATH` | — | Alias of `--sentinel` (matches retrain_v2_plan.md wording) |
| `--human-pref-ckpt PATH` | — | HumanPrefNet .npz — enables `y_hp` auxiliary label |
| `--value-net PATH` | `data/value_net.npz` | Value net checkpoint (synthetic-fallback opponent) |
| `--out PATH` | `data/gap_net_training.npz` | Output training data |
| `--samples-per-category N` | 15000 | Samples per WDL category |
| `--dtw-threshold N` | 15 | DTW threshold for gap labelling |


**Step 2 — Train the GAP net:**

```
# v1 backwards compat (uses only y):
.venv/bin/python tools/train_gap_net.py --epochs 80

# v2 (retrain_v2_plan.md Step 5 — blends y_hp into target):
.venv/bin/python tools/train_gap_net.py \
  --data data/gap_net_training_v2.npz \
  --out  data/gap_net_v2.npz \
  --epochs 80 --hp-blend 0.3
```

| Flag | Default | Description |
| - | - | - |
| `--epochs N` | 80 | Training epochs |
| `--lr F` | 0.001 | Learning rate |
| `--data PATH` | `data/gap_net_training.npz` | Training data |
| `--out PATH` | `data/gap_net.npz` | Output net |
| `--hp-blend F` | 0.0 | Weight added: `y ← clip(y + hp_blend * y_hp, -1, 1)`. 0 = ignore y_hp. Positions with NaN y_hp use plain y. |


## Benchmarking — Sentinel / GAP Net / Tournament

**Base vs base (sanity check):**

```
.venv/bin/python scripts/bench_sentinel.py --games 200 --difficulty 4
```

**Sentinel (20% gap) vs base:**

```
.venv/bin/python scripts/bench_sentinel.py --games 200 --difficulty 4  \
  --white-sentinel score_adjust
```

**GAP net vs base:**

```
.venv/bin/python scripts/bench_sentinel.py --games 200 --difficulty 4 --white-gap-net
```

**Sentinel + GAP net vs base:**

```
.venv/bin/python scripts/bench_sentinel.py --games 200 --difficulty 4  \
  --white-sentinel score_adjust --white-gap-net
```

| Flag | Default | Description |
| - | - | - |
| `--games N` | 4 | Number of games |
| `--difficulty D` | 4 | AI difficulty (1–10) |
| `--white-sentinel MODE` | — | Sentinel mode for White: `score_adjust` etc. |
| `--black-sentinel MODE` | — | Sentinel mode for Black |
| `--sentinel-path PATH` | best.pt | Sentinel checkpoint |
| `--sentinel-scale F` | 0.20 | Min gap fraction for sentinel intervention |
| `--white-value-net` | off | Enable value net for White |
| `--black-value-net` | off | Enable value net for Black |
| `--vn-blend N` | 0 | Value net blend % |
| `--vn-path PATH` | `data/value_net.npz` | Override value_net checkpoint (v2 comparisons) |
| `--white-gap-net` | off | Enable GAP net for White |
| `--black-gap-net` | off | Enable GAP net for Black |
| `--gap-net-path PATH` | `data/gap_net.npz` | Override gap_net checkpoint (v2 comparisons) |
| `--white-humanlike` | off | Blend HumanPrefNet into White's leaf scoring (Step 6e human-proxy) |
| `--black-humanlike` | off | Blend HumanPrefNet into Black's leaf scoring |
| `--humanlike-blend N` | 100 | humanlike_blend percentage; 100 = pure HumanPrefNet |
| `--humanlike-temperature F` | 1.0 | Softmax temperature for `HumanPrefAdvisor.probs()` |
| `--human-pref-path PATH` | `data/human_pref_net.npz` | HumanPrefNet checkpoint |
| `--time-budget F` | — | Per-move time budget override |
| `--suite` | off | Run preset benchmark suite |
| `--round-robin` | off | Round-robin all configurations |

**Step 6e — GapNet vs human-proxy opponent** (once v2 checkpoints are trained):

```
.venv/bin/python scripts/bench_sentinel.py \
  --games 40 --difficulty 5 \
  --white-gap-net --gap-net-path data/gap_net_v2.npz \
  --black-humanlike --humanlike-blend 100 \
  --human-pref-path data/human_pref_net.npz \
  --vn-path data/value_net_v2.npz
```

Repeat with `--gap-net-path data/gap_net.npz` for the v1 baseline and compare
win rates against the same proxy.  Human-proxy games are intentionally weaker
in absolute strength — the goal is measuring GapNet's ability to exploit
human-style play, not to beat a strong engine.


**Full round-robin tournament (S0/S10/S20/S30 + VN blends):**

```
.venv/bin/python tools/bench_tournament.py --diff 4 --budget 3.0 --games-per-pair 10
```

| Flag | Default | Description |
| - | - | - |
| `--diff D` | 4 | AI difficulty |
| `--budget F` | 3.0 | Per-move time budget (seconds) |
| `--games-per-pair N` | 10 | Games per config pair (must be even) |
| `--out PATH` | `eval_results.json` | JSON results output |


**Sentinel v2 benchmark (after training v2/best.pt):**

```
.venv/bin/python tools/bench_sentinel_v2.py --diff 4 --budget 3.0 --games-per-pair 10 --gap 20
```

| Flag | Default | Description |
| - | - | - |
| `--diff D` | 4 | AI difficulty |
| `--budget F` | 3.0 | Per-move time budget (seconds) |
| `--games-per-pair N` | 10 | Games per config pair |
| `--old-ckpt PATH` | — | Old sentinel checkpoint for comparison |
| `--new-ckpt PATH` | — | New sentinel checkpoint to test |
| `--out PATH` | — | JSON results output |


## Opening Audit

```
.venv/bin/python scripts/audit_openings.py --games 5 --diff 4
```

| Flag | Default | Description |
| - | - | - |
| `--games N` | 0 | Games to simulate per opening (0 = eval only) |
| `--diff D` | 3 | Heuristic difficulty for simulation |
| `--threshold F` | 0.06 | Eval margin for W/B vs equal classification |
| `--sim-margin F` | 0.08 | Win-rate margin for simulation classification |
| `--only-id ID` | — | Audit a single opening by ID |
| `--dry-run` | off | Print report without writing files |
| `--seed N` | 42 | Random seed |


## Human DB — Import PlayOK Games

```
python tools/import_playok.py  \
  --archive ~/playok_archive/games  \
  --output data/human_games
```

| Flag | Default | Description |
| - | - | - |
| `--archive PATH` | `~/playok_archive/games` | Input archive directory |
| `--output PATH` | `data/human_games` | Output directory for JSONL files |
| `--dry-run` | off | Count games without writing |
| `--validate-only` | off | Check legality without writing |
| `--limit N` | — | Stop after N new games |
| `--verbose` | off | Per-game status lines |


## Human DB — Rebuild from Scratch

> The two entry points (`build_human_db.py` and `build_human_db_sha.py`)
> share a single library `tools/_human_db_build.py`; they differ only
> in whether they emit the `.sha256` sidecar (the `_sha` variant does).
> Both write **schema v3** (adds `positions_elo_bins` / `moves_elo_bins`
> tables + provenance meta).  A fail-closed guard refuses to write to
> the active HumanDB at `data/human_db.sqlite` or the
> `human_db_path` value in `data/training_paths.local.json` — use
> `--candidate-out` for any new build.

```
.venv/bin/python tools/build_human_db.py  \
  --games-dir data/human_games  \
  --candidate-out data/human_db_candidate.sqlite  \
  --rebuild
```

| Flag | Default | Description |
| - | - | - |
| `--games-dir PATH` | `data/human_games` | Primary game directory |
| `--extra-dirs PATH…` | — | Additional directories |
| `--output PATH` | `data/human_db.sqlite` | Output path (blocked when it resolves to the active DB) |
| `--candidate-out PATH` | — | Preferred output path for v3 candidate builds; overrides `--output` |
| `--malom-db PATH` | auto-resolved | Malom DB directory (see resolution order below) |
| `--no-malom` | off | Skip Malom annotation entirely |
| `--rebuild` | off | Clear DB and reprocess everything from scratch |
| `--update` | off | Only process new / changed files (skips ones in `processed_files` whose SHA-256 matches) |
| `--limit-files N` | — | Cap number of JSONL files scanned (fixture / smoke tests) |

Malom path resolution (highest priority first): `--no-malom` → `--malom-db` → `NMM_MALOM_DB` env var → `malom_db_path` in `data/training_paths.local.json` → sentinel config default (empty).

> **Note:** Do not use both `--rebuild` and `--update`. Use neither to append without checking.

## Human DB — Incremental Update (SHA-tracked, emits .sha256 sidecar)

```
.venv/bin/python tools/build_human_db_sha.py  \
  --update  \
  --games-dir data/human_games  \
  --candidate-out data/human_db_candidate.sqlite
```

Same argparse as `build_human_db.py`; differs only in writing
`<output>.sha256` for download-verification pipelines.  Malom path
auto-resolves from local config or `NMM_MALOM_DB`.

## Human DB — Validate Candidate

```
.venv/bin/python tools/validate_human_db_candidate.py  \
  --candidate data/human_db_candidate.sqlite
```

Reports SQLite `PRAGMA quick_check`, per-key reconciliation of
`positions_elo_bins` totals vs `positions.total_games` and
`moves_elo_bins.total` vs `moves.total`, the new-game Malom semantic
probe (`outcome='D'`), every required provenance meta row, and the
candidate's SHA-256.  Emits `<candidate>.validation.json`.  Exits 0
only when every check passes; activation of the candidate over
`data/human_db.sqlite` is a separate manual step.


## HumanMovePolicyNet — Phase 1 audit

```
.venv/bin/python tools/audit_human_moves.py  \
  --db data/human_db.sqlite  \
  --games-dir data/human_games  \
  --output data/human_moves_audit_optA.json
```

Classifies every recorded human move by (Elo band × Malom WDL
transition × phase).  Reports the full sample-flow gate counts,
per-cell `n_moves` + `n_positions`, coverage share thresholds
(≥1/5/10/25/100 plays), Elo histogram + percentiles, and player
concentration (top 10 movers).  Also captures full provenance: DB
SHA-256, `malom_label_version`, source-manifest hash, git HEAD.
Output is used as the audit baseline in
`docs/human_moves_audit_phase1.md`.

Uses bin-aligned Option A boundaries (`lower ≤ 1149 · middle 1150-1249
· upper ≥ 1250`).  Boundaries live in `learned_ai/data/elo_binning.py`
so the audit and the v3 builder agree on every bin.

| Flag | Default | Description |
| - | - | - |
| `--db PATH` | `data/human_db.sqlite` | HumanDB to read Malom labels from (v2 active or v3 candidate) |
| `--games-dir PATH` | `data/human_games` | JSONL games to replay for Elo attribution |
| `--output PATH` | `data/human_moves_audit.json` | Report path |
| `--top-players N` | 20 | How many top movers to list |
| `--limit-files N` | — | Cap number of JSONL files scanned (smoke) |


## HumanMovePolicyNet — Dataset extraction (v2)

```
.venv/bin/python tools/extract_human_move_policy_dataset.py  \
  --db data/human_db_candidate.sqlite  \
  --output-dir data/human_move_policy_dataset
```

v2 extractor (EXTRACT_VERSION="2").  Reads the v3 candidate DB,
enumerates every legal move at each unique `state_key`, encodes
successor features from the **original mover's perspective** via
`board_to_features(succ, mover_color)`, and joins with `moves_elo_bins`
under Option A band membership.  Successor features are dedup'd by
`state_key`.

Emits two files under `--output-dir`:
- `succ_feats.f32.bin` — numpy float32 memmap of the successor feature bank
- `metadata.npz` — offsets, per-sample records, Option-A band index,
  `sample_split` int8 (0=train/1=val/2=test), backward-compat `sample_is_val`,
  provenance (source DB sha256, feature version, git commit, split counts)

**v2 split**: `three_way_split(state_key)` — 5 % test (buckets 0–4),
15 % val (buckets 5–19), 80 % train (buckets 20–99).  `in_val_bucket`
callers (ValueNet, HumanPrefNet, Sentinel) are unaffected — they never
see `sample_split`.

Smoke run:
```
.venv/bin/python tools/extract_human_move_policy_dataset.py  \
  --db data/human_db_candidate.sqlite  \
  --output-dir data/human_move_policy_dataset  \
  --limit-state-keys 500
```

| Flag | Default | Description |
| - | - | - |
| `--db PATH` | `data/human_db_candidate.sqlite` | v3 candidate DB |
| `--output-dir PATH` | `data/human_move_policy_dataset` | Output directory (memmap + metadata.npz) |
| `--limit-state-keys N` | — | Cap number of state_keys processed (smoke) |
| `--val-fraction FLOAT` | 0.20 | Legacy param kept for compat; v2 ignores it (uses three_way_split) |


## HumanMovePolicyNet — Session index build

```
.venv/bin/python tools/build_session_index.py  \
  --dataset-dir data/human_move_policy_dataset  \
  --games-dir   data/human_games  \
  --output      data/human_move_policy_session_index.npz
```

Scans all 97 138 JSONL game files in `--games-dir`, maps each game to
a split tier via `game_level_split(session_id)`, and for each move
maps the board position to its row in the dataset `state_keys` array.
Builds two uint8 bitmask arrays indexed by dataset row:

- `game_split_mask`: bit 0=appeared in game-train, bit 1=appeared in
  game-val, bit 2=appeared in game-test game.
- `player_split_mask`: same but keyed by the mover's player identity.

Positions with `(mask & 0x01)==0` and `(mask & 0x02)!=0` are
**game-val-only** — not reached by any training game.  The eval script
uses this to report the `game_val_only` diagnostic stratum.

Saves `data/human_move_policy_session_index.npz.provenance.json`
alongside the `.npz`.

Smoke run (first 500 files, ~5 s):
```
.venv/bin/python tools/build_session_index.py --limit-files 500
```

| Flag | Default | Description |
| - | - | - |
| `--dataset-dir PATH` | `data/human_move_policy_dataset` | Extractor output directory (needs metadata.npz) |
| `--games-dir PATH` | `data/human_games` | JSONL game files directory |
| `--output PATH` | `data/human_move_policy_session_index.npz` | Output path |
| `--limit-files N` | — | Cap number of JSONL files scanned (smoke) |


## HumanMovePolicyNet — Train (v2 three-way split)

```
.venv/bin/python tools/train_human_move_policy_net.py  \
  --dataset-dir data/human_move_policy_dataset  \
  --output data/human_move_policy_net_v2_candidate.npz  \
  --epochs 200  \
  --patience 10
```

MLP 82 → 128 → 64 → 32 → 1 (79 board features + 3 Elo band one-hot).
**Ordinary count-weighted cross-entropy** over observed events;
softmax over every legal move at each position (unobserved legal
moves get target 0).

On v2 datasets (`sample_split` present), train/val indices come from
the three-way split (train=80 %, val=15 %, test=5 %).  On v1 datasets
falls back to `sample_is_val` (no test tier).

Default output changed from `human_move_policy_net.npz` to
`human_move_policy_net_v2_candidate.npz` to protect the first-run
model.  Saves a companion `<output>.provenance.json`.

First-run result (v1, 22 epochs, early stop): val NLL=1.5953, ~11.6 h.

| Flag | Default | Description |
| - | - | - |
| `--dataset-dir PATH` | `data/human_move_policy_dataset` | Extractor output directory |
| `--output PATH` | `data/human_move_policy_net_v2_candidate.npz` | `.npz` output |
| `--epochs N` | 40 | Max epochs (increase for v2 run: 200 recommended) |
| `--patience N` | 6 | Early-stop patience (0 disables; 10 recommended for v2) |
| `--lr FLOAT` | 3e-4 | Adam learning rate |
| `--dropout FLOAT` | 0.2 | Dropout on hidden layers |
| `--batch-positions N` | 512 | (position, band) samples per gradient step |
| `--grad-clip FLOAT` | 1.0 | Gradient-norm clip |
| `--seed N` | 42 | RNG seed |


## GapNet v3 — Stage D Dataset Extraction

Smoke test (500 state_keys, ~2 min):
```
.venv/bin/python tools/extract_gap_v3_dataset.py \
  --limit 500 \
  --out-dir /tmp/gap_v3_smoke
```

Full run (writes to `data/gap_net_v3_dataset/`, checkpoint every 50 k emitted rows):
```
PYTHONUNBUFFERED=1 nohup .venv/bin/python -u tools/extract_gap_v3_dataset.py \
  --min-support 1 > data/logs/gap_v3_extract_full.log 2>&1 &
```

Resume an interrupted full run:
```
PYTHONUNBUFFERED=1 nohup .venv/bin/python -u tools/extract_gap_v3_dataset.py \
  --min-support 1 --resume > data/logs/gap_v3_extract_full.log 2>&1 &
```

| Flag | Default | Description |
| - | - | - |
| `--human-db PATH` | `data/human_db_candidate.sqlite` | Human game DB |
| `--malom-db-dir PATH` | from `data/settings.json` | Malom DB directory |
| `--model PATH` | `data/human_move_policy_net_v2_candidate.npz` | P_h policy model |
| `--out-dir PATH` | `data/gap_net_v3_dataset/` | Output directory |
| `--min-support N` | 1 | Min total plays per (state_key, band) to include |
| `--min-empirical-support N` | 25 | Min support for empirical P_h (below → model-only) |
| `--temperature FLOAT` | 0.7674 | Model temperature (T* from Phase 4b eval) |
| `--limit N` | off | Process first N state_keys only (smoke test) |
| `--resume` | off | Continue from `checkpoint.json` in `--out-dir` |

Output files in `data/gap_net_v3_dataset/`:
- `parent_feats.f32.bin` — (N, 79) float32, row-major
- `targets.f32.bin` — (N, 4) float32 `[g_v_A, g_v_B, g_v_C, NaN]`
- `targets_empirical.f32.bin` — (N, 4) float32 (empirical G_v, NaN where absent)
- `metadata.npz` — state_keys, band_idx, split, phase, mover_color, n_legal, ph_source
- `provenance.json` — full provenance record
- `abstained.jsonl` — one line per abstained (state_key, band) with reason
- `checkpoint.json` + `metadata_checkpoint.npz` — resume state


## HumanMovePolicyNet — Phase 4b Evaluate (full)

Standard val eval (temperature scaling, all strata, all baselines):
```
.venv/bin/python tools/eval_human_move_policy_net.py  \
  --dataset-dir data/human_move_policy_dataset  \
  --model data/human_move_policy_net_v2_candidate.npz  \
  --candidate-db data/human_db_candidate.sqlite  \
  --session-index data/human_move_policy_session_index.npz  \
  --output data/gap_v3_prerequisite_eval.json
```

Single-shot test set (run **once only** after model is finalised):
```
.venv/bin/python tools/eval_human_move_policy_net.py  \
  --dataset-dir data/human_move_policy_dataset  \
  --model data/human_move_policy_net_v2_candidate.npz  \
  --candidate-db data/human_db_candidate.sqlite  \
  --session-index data/human_move_policy_session_index.npz  \
  --run-test-set  \
  --output data/gap_v3_prerequisite_eval.json
```

Phase 4b reports (per §4.1 GapNet v3 plan):
- **Temperature scaling**: pass-1 finds T* on val NLL; pass-2 uses T*.
- **Model strata**: overall, per Elo band, per phase, per Malom
  transition, per legal-move-count bin (2-5 / 6-10 / 11-20 / 21+),
  OOD (positions not in train), abstention (encoding failures),
  game-val-only (with `--session-index`).
- **Baseline strata**: uniform and empirical (≥ `--min-support`).
- **Metrics per stratum**: event NLL, Brier, top-1/3/5, ECE.
- **Empirical KL** (model || empirical, positions ≥ `--min-support`).

Output key `val` = val set results; `test` = test set (only if
`--run-test-set`).  Report also carries model provenance (dataset
lineage, hparams, git commit, best val NLL).

Smoke run (skip temperature, fast):
```
.venv/bin/python tools/eval_human_move_policy_net.py  \
  --dataset-dir data/human_move_policy_dataset  \
  --model data/human_move_policy_net_v2_candidate.npz  \
  --candidate-db data/human_db_candidate.sqlite  \
  --skip-temperature  \
  --output data/human_move_policy_eval_smoke.json
```

| Flag | Default | Description |
| - | - | - |
| `--dataset-dir PATH` | `data/human_move_policy_dataset` | Extractor output directory |
| `--model PATH` | `data/human_move_policy_net_v2_candidate.npz` | Trained model `.npz` |
| `--candidate-db PATH` | `data/human_db_candidate.sqlite` | v3 candidate DB (Malom transition labels) |
| `--session-index PATH` | — | Session index `.npz` (enables game-val-only stratum) |
| `--min-support N` | 10 | Min observed events for empirical baseline + KL |
| `--run-test-set` | off | Also evaluate untouched test partition (single-shot only) |
| `--skip-temperature` | off | Skip T* search (use T=1.0); faster smoke runs |
| `--output PATH` | `data/human_move_policy_eval.json` | Report path |


## Full Game DB — Build

```
python tools/build_fullgame_db.py  \
  --expand-from-games data/games  \
  --min-seed-frequency 3  \
  --early-expand-depth 4  \
  --expand-depth 6  \
  --output /mnt/windows/NMM_DB/fullgame.bin  \
  --temp-db /mnt/windows/NMM_DB/  \
  --max-db-gb 40
```

| Flag | Default | Description |
| - | - | - |
| `--expand-from-games DIR` | `data/games` | Human game records to seed from |
| `--output PATH` | `data/fullgame.bin` | Output binary file |
| `--db-dir DIR` | — | Shorthand for `--output <dir>/fullgame.bin` |
| `--temp-db PATH` | alongside output | Temporary SQLite build DB (use large drive) |
| `--max-db-gb GB` | 10.0 | Stop BFS when temp DB exceeds this size |
| `--max-gb GB` | 6.0 | Abort when process RSS exceeds this (GB) |
| `--min-seed-frequency N` | 2 | Min human-game visits to seed a BFS position |
| `--expand-depth D` | 4 | BFS depth for late-game positions |
| `--early-expand-depth D` | 2× expand-depth | BFS depth for early-game positions |
| `--max-expand-positions N` | unlimited | Hard cap on BFS positions |
| `--passes N` | 6 | Backpropagation passes for W/L labelling |
| `--dry-run` | off | Synthetic build (no disk write) |
| `--quiet` | off | Suppress progress logging |


## Endgame DB — Build

```
python tools/build_endgame_db.py --build-all --skip-existing  \
  --out-dir /mnt/windows/NMM_DB
```

| Flag | Default | Description |
| - | - | - |
| `--out-dir PATH` | `data/endgame` | Directory for `endgame_*.wdl` files |
| `--build-all` | off | Build all tables in dependency order |
| `--max-sum N` | 11 | Max nW+nB when using `--build-all` |
| `--nW N` | — | White piece count (single table build) |
| `--nB N` | — | Black piece count (single table build) |
| `--skip-existing` | off | Skip tables whose .wdl already exists |
| `--quiet` | off | Suppress per-pass logging |


## Sentinel v2 — Step 0 Wrapper (Stages 1 → 2 → 4)

Orchestrates the three-stage Sentinel v2 retrain from `docs/retrain_v2_plan.md`
(Stage 3 archived, Stage 5 dropped). Idempotent — delete any `v2_stage{N}/`
directory to force that stage to rerun. Promotes the final checkpoint to
`learned_ai/sentinel/checkpoints/v2/best.pt` without touching production
`best.pt`.

**Combined JSONL + Malom dataset (v2b default).** The wrapper now builds
a single training dataset up front via `scripts/build_sentinel_dataset_v2.py`
and feeds it to all three stages via `train_sentinel.py --dataset PATH`:

- 60 % Malom-sampled positions (equal parts placement / midgame / fly),
  each position guaranteed to contain at least one W-or-L legal move
  (not all-draw) so the model isn't dominated by flat-draw states.
- 40 % classic JSONL replay from `data/games` + `data/human_games`.
- Total size defaults to 4M examples (~2× the JSONL-only baseline);
  override via `DATASET_TOTAL_EXAMPLES=...`.

**Best-checkpoint fallback**: the trainer restores `best_val` from the
resume checkpoint, so a stage that never dips below that inherited value
never writes a fresh `best.pt`. The wrapper handles this by copying the
resume checkpoint (or `latest.pt` as second fallback) into the stage's
`best.pt` with a clear log message, so the chain proceeds even when a
stage plateaus.

```bash
./scripts/train_sentinel_v2_step0.sh                # cpu, default paths
DEVICE=cuda ./scripts/train_sentinel_v2_step0.sh    # gpu run
MALOM_DB=/path/to/Std_DD_89adjusted ./scripts/train_sentinel_v2_step0.sh
REBUILD_DATASET=1 ./scripts/train_sentinel_v2_step0.sh   # force dataset rebuild
RUN_STAGE1=1 ./scripts/train_sentinel_v2_step0.sh        # A/B include Stage 1 (⚠︎ not equivalent to Recipe A of the rollback doc)
```

**Extended epoch budget (§S1)**: `configs/sentinel_stage{1,2,4}.yaml` ship
with `epochs: 100 / 150 / 150` (5× the previous limits) so the trainer has
room to fit the combined dataset.  Every stage carries `patience: 10`, so
early-stop still shortens the run when val plateaus.

**Position-level split marker (§S1)**: `scripts/build_sentinel_dataset_v2.py`
now emits an `is_val` boolean array in the `.npz`.  Every candidate move
from the same `position_key` (state_key for Malom-sampled rows, board FEN
for JSONL rows) lands in the same slice, so no ply leaks between train and
val.  `scripts/train_sentinel.py` picks up the marker automatically; falls
back to a random per-example split with a warning if a legacy `.npz` has
no marker.

| Env var | Default | Description |
| --- | --- | --- |
| `DEVICE` | `cpu` | Passed as `--device` to `train_sentinel.py` |
| `MALOM_DB` | `malom_db_path` from `data/training_paths.local.json` (else `/mnt/windows/...`) | Malom DB directory used for Malom sampling AND for Stages 2 / 4 label queries |
| `GAME_DIR` | `data/games` | AI self-play game directory |
| `HUMAN_GAME_DIR` | `data/human_games` | Human game directory |
| `PATIENCE` | *from config* | Override early-stop patience. All three stages ship `patience: 10` in their YAML. Set `PATIENCE=0` to disable, or any positive int to override. |
| `DATASET_PATH` | `learned_ai/sentinel/datasets/v2_combined.npz` | Where the combined dataset is written / read |
| `DATASET_TOTAL_EXAMPLES` | `4000000` | Combined-dataset target size |
| `DATASET_MALOM_FRACTION` | `0.60` | Fraction of examples drawn from Malom sampling |
| `REBUILD_DATASET` | *(unset)* | Set to `1` to rebuild `DATASET_PATH` even when the file already exists |
| `RUN_STAGE1` | *(unset)* | Set to `1` to include the Stage 1 heuristic-label warm-start. **Skipped by default** — it anchors the trunk on the heuristic function and Stage 2 then plateaus without learning Malom labels. Stage 2 is the entry point in the default flow. |

**Why Stage 1 is skipped**: an earlier run showed Stage 2's val loss identical
to Stage 1's for 10 straight epochs (0.3382 → 0.3382 → …). Stage 1 imprints a
strong heuristic prior; at Stage 2's original `lr: 0.0003` Adam can't move the
trunk far enough to learn Malom labels. The fix is (a) skip Stage 1 so there's
no prior anchor, and (b) bump Stage 2's LR to `0.001` so it can learn from
random init. Both are default in v2b.

**Standalone dataset builder**: `scripts/build_sentinel_dataset_v2.py` can
be run outside the wrapper for offline dataset production or debugging.
Its output is a plain `.npz` that `train_sentinel.py --dataset` consumes.

```bash
.venv/bin/python scripts/build_sentinel_dataset_v2.py \
  --out learned_ai/sentinel/datasets/v2_combined.npz \
  --total-examples 4000000 --malom-fraction 0.60
```

Small `--total-examples` (< 250k) auto-caps the JSONL file glob so smoke
runs finish in seconds instead of minutes.

Early stopping (`patience`) is now built into `scripts/train_sentinel.py`: the
epoch loop breaks after N consecutive epochs without a new best val loss.
`best.pt` remains the epoch with the lowest val loss; `latest.pt` is the last
epoch run. On resume, the trainer starts a fresh `stall` counter.


## Sentinel v2 — Train

```
.venv/bin/python scripts/train_sentinel.py  \
  --game-dir data/games  \
  --human-game-dir data/human_games  \
  --ai-game-dir data/ai_games  \
  --db-path /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted  \
  --drop-db-features  \
  --aux-wdl --lambda-wdl 0.4  \
  --contrastive --lambda-contrastive 0.4  \
  --curriculum  \
  --epochs 50 --epochs-phase1 10  \
  --lr-phase1 5e-3 --lr-phase2 5e-3  \
  --out-dir learned_ai/sentinel/checkpoints/v2  \
  --device cuda
```

| Flag | Default | Description |
| - | - | - |
| `--game-dir PATH` | `data/games` | AI self-play game directory |
| `--human-game-dir PATH` | — | Human game directory |
| `--ai-game-dir PATH` | — | Additional AI game directory |
| `--db-path PATH` | — | Malom DB directory |
| `--dataset PATH` | — | Preprocessed .npz (skips replay) |
| `--drop-db-features` | off | Zero out DB-derived input features |
| `--aux-wdl` | off | Add auxiliary WDL prediction head |
| `--lambda-wdl F` | 0.3 | Weight for WDL auxiliary loss |
| `--contrastive` | off | Enable contrastive loss |
| `--lambda-contrastive F` | 0.3 | Weight for contrastive loss |
| `--curriculum` | off | Phase 1 → Phase 2 curriculum schedule |
| `--epochs N` | — | Total training epochs |
| `--epochs-phase1 N` | — | Epochs in phase 1 |
| `--lr-phase1 F` | — | Learning rate for phase 1 |
| `--lr-phase2 F` | 1e-4 | Learning rate for phase 2 |
| `--out-dir PATH` | — | Checkpoint output directory |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--resume PATH` | — | Resume from checkpoint |
| `--decisive-only` | off | Skip drawn games |
| `--trajectory-weight` | off | Weight samples by trajectory quality |
| `--limit N` | — | Max game files to load |
| `--config PATH` | — | JSON config file (overrides flags) |
| `--patience N` | — | Override `config.patience`. 0 disables early stopping. |


## Learned AI — Imitation Data Generation

Run these before training specialist networks.

**Step 1 — AI self-play imitation data (~10h):**

```
.venv/bin/python scripts/gen_imitation_data.py  \
  --games 1000 --diff 7  \
  --sentinel learned_ai/sentinel/checkpoints/best.pt
```

| Flag | Default | Description |
| - | - | - |
| `--games N` | 2000 | Games to generate |
| `--diff D` | 3 | AI difficulty |
| `--sentinel PATH` | — | Sentinel checkpoint |
| `--malom PATH` | — | Malom DB path |
| `--value-net PATH` | `data/value_net.npz` | Value net checkpoint |
| `--out PATH` | auto | Output .npz |
| `--max-ply N` | 300 | Max plies per game |
| `--seed N` | 42 | Random seed |


**Step 2 — Human game imitation data (62-float, legacy):**

```
.venv/bin/python scripts/gen_human_imitation_data.py
```

| Flag | Default | Description |
| - | - | - |
| `--games-dir PATH` | `data/games` | Source game directory |
| `--out PATH` | `learned_ai/data/human_imitation.npz` | Output .npz |
| `--sentinel PATH` | `learned_ai/sentinel/checkpoints/best.pt` | Sentinel checkpoint |
| `--malom PATH` | — | Malom DB path |
| `--value-net PATH` | `data/value_net.npz` | Value net |
| `--won-weight F` | 1.0 | Sample weight for winner positions |
| `--draw-weight F` | 0.3 | Sample weight for draw positions |
| `--loser-weight F` | 0.5 | Sample weight for loser positions from human-won games |


**Step 2b — Human game imitation data v2 (122-float, for v2 specialists; ~6–8h with sentinel):**

Uses `encode_position_with_lookahead` with 15-ply LookaheadAdvisor + GapNet. Cap `--max-moves 120` prevents stalling on very long game files.

```
.venv/bin/python scripts/gen_human_imitation_data_v2.py  \
  --gap-net data/gap_net.npz --max-moves 120
```

Output: `learned_ai/data/human_imitation2.npz` — 13,040 positions, 122-float features.

| Flag | Default | Description |
| - | - | - |
| `--games-dir PATH` | `data/human_games` | Source game directory |
| `--out PATH` | `learned_ai/data/human_imitation2.npz` | Output .npz |
| `--sentinel PATH` | `best.pt` | Sentinel checkpoint |
| `--value-net PATH` | `data/value_net.npz` | Value net |
| `--gap-net PATH` | `data/gap_net.npz` | GapNet (blunder density) |
| `--max-moves N` | 120 | Cap moves per game to avoid stalls |


## Learned AI — Specialist Training v3 (Opening / Midgame / Endgame)

Three independent phase specialists (opening / midgame / endgame). Each one sits on top of the classical engine's alpha-beta search and re-ranks its top-K candidates.

**Per-move features (126 floats each row):**
- **62 base** — sentinel score, heuristic + VN blended eval and delta, counterfactual block, `is_engine_top1` flag, and the 58-float sentinel/board context.
- **60 lookahead** — 15 half-plies × 4 signals (heuristic + VN + sentinel + gap). Training simulates only `--sim-ply-depth` half-plies (default 5) and pads to full width; inference always runs full 15.
- **4 top-K extras** — `ab_score_norm`, `ab_rank_norm`, `human_freq`, `human_rank`.

**Value input (80 floats):** 23 encoder base + 9 history (last 3 moves' from/to/capture as normalised indices) + 48 raw-board one-hot (24 positions × 2 colours).

**Model:** `ScaffoldedPolicyNet` — policy MLP `126 → 512 → 256 → 128 → 1`, value MLP `80 → 256 → 128 → 64 → 1`. ~289 k params.

**Difficulty:** 20 levels. Log-scale per-move opponent budget: L1 ≈ 1 ms → L15+ caps at 2 s (mid/end) or 1 s (opening). **Advancement:** Sanmill superiority-probability gate — `P(true score > target) ≥ 0.70` on the last 50 games (v2a); target ramps 55% (L1) → 60% (L20) with time-of-flight relaxation to a 51% floor after 1000+ stalled games. In v2a the check now fires every batch once `games_at_level ≥ 20` (the old `games_at_level % 10 == 0` gate silently skipped windows under batched increments).

### Prerequisites

Generate the 122-float human imitation warm-start dataset once (`human_imitation2.npz`, ~6-8 h wall):

```
.venv/bin/python scripts/gen_human_imitation_data_v2.py \
  --gap-net data/gap_net.npz --max-moves 120
```

### Fresh training runs (recommended flags)

Speed flags: `--sim-ply-depth 5` (~3× lookahead speed-up during training; inference stays at 15) and `--minimal-rollouts` (one primary rollout per game, no confirm / retry). Launch each specialist in its own terminal / tmux pane — they train independently and in parallel.

**Opening specialist — fresh:**

```
.venv/bin/python scripts/train_s_open_v2.py \
  --max-games 30000 --batch-games 10 \
  --sim-ply-depth 5 --minimal-rollouts \
  --self-play-ratio 0.05
```

**Midgame specialist — fresh:**

```
.venv/bin/python scripts/train_s_mid_v2.py \
  --max-games 30000 --batch-games 10 \
  --sim-ply-depth 5 --minimal-rollouts \
  --self-play-ratio 0.05
```

**Endgame specialist — fresh:**

```
.venv/bin/python scripts/train_s_end_v2.py \
  --max-games 30000 --batch-games 10 \
  --sim-ply-depth 5 --minimal-rollouts \
  --self-play-ratio 0.05 \
  --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted
```

### Resume from best checkpoint

```
.venv/bin/python scripts/train_s_open_v2.py --auto-resume-best --max-games 30000 --batch-games 10 --sim-ply-depth 5 --minimal-rollouts
.venv/bin/python scripts/train_s_mid_v2.py  --auto-resume-best --max-games 30000 --batch-games 10 --sim-ply-depth 5 --minimal-rollouts
.venv/bin/python scripts/train_s_end_v2.py  --auto-resume-best --max-games 30000 --batch-games 10 --sim-ply-depth 5 --minimal-rollouts --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted
```

### Smoke test (2-5 games each, no warm-start)

```
.venv/bin/python scripts/train_s_open_v2.py --max-games 5 --no-s1a-warmstart --sim-ply-depth 5 --minimal-rollouts
.venv/bin/python scripts/train_s_mid_v2.py  --max-games 5 --no-s1a-warmstart --sim-ply-depth 5 --minimal-rollouts
.venv/bin/python scripts/train_s_end_v2.py  --max-games 5 --no-s1a-warmstart --sim-ply-depth 5 --minimal-rollouts --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted
```

### Common flags (all three v2/v3 specialists)

| Flag | Default | Description |
| - | - | - |
| `--sentinel PATH` | `best.pt` | Sentinel checkpoint |
| `--value-net PATH` | `data/value_net.npz` | Trajectory value net |
| `--gap-net PATH` | `data/gap_net.npz` | Gap net (blunder density) |
| `--out-dir PATH` | `learned_ai/checkpoints/scaffolded/s_*_v2` | Checkpoint output |
| `--s1a-data PATH` | `data/human_imitation2.npz` | Pre-RL imitation warm-start data |
| `--no-s1a-warmstart` | off | Skip s1a warm-start (start RL from scratch) |
| `--batch-games N` | 1 | Parallel primary rollouts via ThreadPoolExecutor. 10 recommended on 16+ cores; diminishing returns beyond 24. |
| `--sim-ply-depth N` | 5 | LookaheadAdvisor simulated depth during training. Feature width still 60 floats via padding; inference runs full 15. |
| `--minimal-rollouts` | off | Skip retry + confirm rollouts (branches already off by default). ~3× training throughput at the cost of sample efficiency. |
| `--max-games N` | 5000 | Games (soft cap; specialist stops early on hitting max difficulty). |
| `--diff-max N` | 20 | Maximum difficulty level. |
| `--diff-start N` | 1 | Starting difficulty level. |
| `--time-budget F` | -1 (auto per-level) | Override per-move budget for the opponent's α-β search. |
| `--self-play-ratio F` | 0.5 | Fraction of games vs frozen model. 0.05 recommended once RL is stable. |
| `--lr F` | 1e-4 | Learning rate. |
| `--entropy-coef F` | 0.01 | Entropy regularisation coefficient. |
| `--update-every N` | 16 | Policy update interval (steps). |
| `--rolling-win N` | 50 | Rolling window for the Sanmill advance test. |
| `--resume PATH` | — | Explicit checkpoint to resume from. |
| `--auto-resume-best` | off | Auto-resume from `s_*_v2/best.pt`. |
| `--ppo` | off | Use PPO instead of A2C. |
| `--seed N` | 42 | Random seed. |
| `--malom PATH` | — | (endgame only) Malom perfect DB directory for endgame reward + lookahead early-exit. |

### Notes for overnight runs

- Each specialist runs independently. Launch all three in parallel, one per terminal / tmux pane.
- At `--batch-games 10` on a 16-core CUDA box, expect **~300-1100 games/hour at diff 1** with the v3 speed flags applied.
- Watch `htop` — if all CPU cores are pegged, raise `--batch-games` cautiously (10 → 16 → 24). Beyond 24 you'll see diminishing returns from Python GIL + memory pressure.
- Advance-check log line format: `[s_open_v2] advance-check @ diff 3: P=0.982 ≥ 0.95 (target=0.545, score=0.760)`. When the P-value stays < 0.5 for 5000+ games at a level, more games won't help — the model has plateaued.


## Learned AI — Generalist v2a (train_s_gen_v2a.py, legacy)

The v2a script is retained at the pre-`4809b33` state as a stable reference
for the original generalist behaviour (temp/entropy boost, two-stage
recovery, rehearsal games, auto-resume flags).  It does **not** carry the
new termination classification, advancement cooldown, no-reload-on-advance,
or `level_heuristic_history` cap — those live in v2b.

Use v2a if you specifically want to reproduce an older run; use v2b for all
new training.  Checkpoints from either script are cross-compatible: v2b's
`_load_model` accepts the `s_gen_v2a` stage tag as valid lineage, so a v2a
`latest.pt` resumes cleanly with `--auto-resume-latest`.


## Learned AI — Generalist v2b (train_s_gen_v2b.py)

Full-game generalist that plays from `new_game()` through placement, midgame, and endgame in a single model. Uses a diversified opponent schedule to prevent overfitting to any one difficulty.

**Discussion-plan fixes landed (§A/§D/§I/§L/§O/§R/§T/§Δ)** — see
`docs/discussion_plan.md` for the full write-up.  Short summary of the
behavioural changes:

- **§T** — Temperature schedule cools with `games_at_level` (per-level
  clock, resets on advancement) rather than global `game_count`.
  `--temp-floor` (default 0.30) sets a persistent minimum effective
  temperature.  Boost decays per **primary game**, not per outer batch, so
  batch size no longer changes curriculum timing.  `temp_boost`,
  `entropy_boost`, and `temp_boost_last_game` round-trip through
  `latest.pt`.
- **§D** — Rules-based draws detected during rollout:
  `DRAW_REPETITION` (threefold identical board+turn) and `DRAW_50_MOVE`
  (100 half-plies without a capture) are distinct termination reasons from
  `DRAW_MAX_PLY_TRUNCATED`.
- **§R** — `latest.pt` persists optimizer state, all four RNG streams
  (Python / NumPy / torch / torch.cuda), and every periodic counter
  (`games_since_target_update`, `_last_log_game`,
  `_last_advance_print_game`, `_last_deep_game`).  Resume is
  trajectory-equivalent modulo GPU determinism.
- **§L** — LR-adaptation now smooths its win-rate signal via an EMA
  (`lr_win_rate_ema`, alpha 0.30, persisted in `latest.pt`), so a single
  bad log tick can't slam LR to the boundary.
- **§O** — Console log line uses explicit opponent-class tags: `heur`,
  `hard`, `easy`, `blnd`, `bldd`, `self`, `?xyz`.  JSONL row still
  carries the full `game_type` string for downstream analysis.
- **§A** — Only `vs_heuristic` games at the **current difficulty**
  populate `level_heuristic_history` (the advancement gate).
  `is_advance_reference` is a new `_GameConfig` field set exclusively for
  those games; hard / easy / blend / blunder / frozen games still feed
  `win_history_heuristic` for display + recovery but do NOT drive
  advancement.
- **§I** — INFRA rollouts (learner encoder failure, opponent exception)
  are excluded from `_retroactive_rescore`, from `ep_steps` extension,
  from SpecialistDB writes, and from every W/D/L history.  A single
  broken opponent no longer contaminates the RL update.
- **§Δ** — `GameDiag.opponent_search_depth_mean` (float): realised
  opponent search depth per move, averaged over the rollout.  Lets the
  plot distinguish "harder opponent" from "same opponent + more search
  time" at high difficulty levels.

New CLI flags: `--temp-floor F`, `--temp-anneal-games N`.  Old flags
unchanged.  A v2b `latest.pt` produced by any older revision is still
loadable — missing new fields fall back to safe defaults.

**Opponent schedule (per batch iteration):**

| Slot | Normal | Rehearsal | Description |
| --- | --- | --- | --- |
| 10% | 10% | 10% | Next-higher-difficulty (anti-overfit) |
| 20% | `_lower_diff_hi` | expanded | Random lower difficulty |
| 10% | 10% | 10% | Blundering heuristic (25% blunder rate) |
| 10% | 10% | 10% | Blended (VN 10% + gap 30% + sentinel 20%) |
| 50% | remainder | compressed | Standard (self-play or current-diff heuristic) |

**Recommended run:**

```
.venv/bin/python scripts/train_s_gen_v2b.py \
  --max-games 50000 \
  --temp-start 1.2 \
  --sim-ply-depth 12 \
  --self-play-ratio 0.05 \
  --ppo \
  --advance-temp-boost-frac 0.5 \
  --advance-entropy-boost-frac 0.5 \
  --advance-rehearsal-games 50 \
  --hot-explore-games 75 \
  --auto-resume-best
```

**Resume from checkpoint:**

```
.venv/bin/python scripts/train_s_gen_v2b.py --auto-resume-best \
  --max-games 50000 --temp-start 1.2 --sim-ply-depth 12 \
  --self-play-ratio 0.05 --ppo \
  --advance-temp-boost-frac 0.5 --advance-entropy-boost-frac 0.5 \
  --advance-rehearsal-games 50 --hot-explore-games 75
```

**Smoke test (3 games, no warm-start, no nets):**

```
.venv/bin/python scripts/train_s_gen_v2b.py --max-games 3 \
  --no-s1a-warmstart --no-sentinel --no-value-net --no-gap-net --no-imitation-mix
```

### Flags table

| Flag | Default | Description |
| --- | --- | --- |
| `--max-games N` | 5000 | Total games to train |
| `--temp-start F` | 0.90 | Starting temperature; anneals to 0.20 over 80% of run |
| `--sim-ply-depth N` | 5 | LookaheadAdvisor simulation depth during training (12 recommended; inference stays at 15-ply) |
| `--self-play-ratio F` | 0.5 | Fraction of standard-slot games vs frozen model (0.05 recommended — mostly vs heuristic) |
| `--ppo` | off | Use PPO update; otherwise A2C |
| `--auto-resume-best` | off | Resume from `<out-dir>/<run-name>/best.pt` if present, else `<out-dir>/best.pt` |
| `--auto-resume-latest` | off | Resume from `<out-dir>/<run-name>/latest.pt` if present, else `<out-dir>/latest.pt` |
| `--resume PATH` | — | Resume from explicit checkpoint path |
| `--out-dir PATH` | `learned_ai/checkpoints/scaffolded/s_gen_v2b` | Checkpoint output directory (parent) |
| `--run-name STR` | "" | Optional subfolder under `--out-dir` for parallel experiments |
| `--diff-start N` | 1 | Override starting difficulty |
| `--diff-max N` | 20 | Maximum difficulty level |
| `--lr F` | 1e-4 | Base learning rate (scaled by win-rate LR adaptation each log interval) |
| `--entropy-coef F` | 0.01 | Base entropy regularisation coefficient (boosted during advancement/hot-explore) |
| `--update-every N` | 64 | Policy update batch size (steps) |
| `--rolling-win N` | 40 | Rolling window for win/loss history |
| `--log-every N` | 50 | Games between log + checkpoint + recovery check |
| `--update-target-every N` | 50 | Games between frozen opponent refreshes |
| `--sentinel PATH` | `sentinel/best.pt` | Sentinel checkpoint |
| `--value-net PATH` | `data/value_net.npz` | Value net checkpoint |
| `--gap-net PATH` | `data/gap_net.npz` | Gap net checkpoint |
| `--malom PATH` | — | Malom DB directory |
| `--no-sentinel` | off | Disable sentinel |
| `--no-value-net` | off | Disable value net |
| `--no-gap-net` | off | Disable gap net |
| `--batch-games N` | 1 | Parallel game rollouts per iteration |
| `--max-ply N` | 60 | Max plies per primary game |
| `--max-ply-branch N` | 60 | Max plies per branch game |
| `--minimal-rollouts` | off | Skip retry + confirm rollouts (~3× throughput, lower sample efficiency) |
| `--no-s1a-warmstart` | off | Skip pre-RL imitation warm-start |
| `--no-imitation-mix` | off | Disable AlphaZero-style imitation mini-steps during RL |
| `--heuristic-node-budget N` | — | Fixed node budget per heuristic move (deterministic; overrides time budget) |
| `--segment-games N` | — | Stop after N games from resume point (bounded-run support) |
| `--seed N` | 42 | Random seed |

### Advancement and recovery flags (new in v2a)

| Flag | Default | Description |
| --- | --- | --- |
| `--advance-temp-boost-frac F` | 0.5 | On difficulty advancement (or Stage-1 recovery trigger), add `F × (temp_start − scheduled_temp)` as a decaying temperature bonus. Decays at 0.97/iter. |
| `--advance-entropy-boost-frac F` | 0.5 | On advancement/hot-explore, add `F × entropy_coef` as a decaying entropy bonus to encourage exploration at the new difficulty. Decays at 0.97/iter. |
| `--advance-rehearsal-games N` | 0 | After each difficulty advancement, expand the lower-diff opponent slot from 20% to `--advance-rehearsal-prob` for N iterations. 0 = disabled. 50 recommended. |
| `--advance-rehearsal-prob F` | 0.45 | Lower-diff opponent probability during rehearsal window (vs standard 0.20). |
| `--advance-cooldown-tail-batches N` | 8 | Extra batches beyond rehearsal before another advance can fire. Prevents chained advances driven by the temporarily easier rehearsal opponent mix. Persisted in `latest.pt`. |
| `--hot-explore-games N` | 75 | **Stage-1 recovery:** run N elevated-temperature PRIMARY games before reloading a checkpoint. Counter decrements by `batch_games` per batch (so ~ `N / batch_games` batches). While `hot_explore_remaining > 0` the scheduled temperature is floored at `temp_start × 1.3`. Set 0 to skip Stage 1. |

### Two-stage recovery

Recovery triggers at each `--log-every` checkpoint when the chess-style score
`win + 0.5×draw < 0.35` **and the model is actively degrading** (second-half
score of the rolling window is at least 5pp below the first half). Suppressed
entirely while `_current_recovery_state == "grace"` so a Stage 2 restore cannot
immediately re-trigger Stage 1.

1. **Stage 1 (hot-explore):** Boosts temperature and entropy coefficient, runs `--hot-explore-games` PRIMARY games at elevated temperature without loading any checkpoint. If the model recovers naturally, flags are cleared.
2. **Stage 2 (checkpoint restore):** If still failing after Stage 1 completes, loads `best{difficulty}.pt`, resets the optimiser, clears histories, and suppresses the draw penalty for 100 games. All boosts are cleared.

Setting `--hot-explore-games 0` skips Stage 1 entirely (original behaviour).

### Difficulty advancement (v2a, updated)

Uses the Sanmill superiority-probability gate on `level_heuristic_history`.
Checked **every batch** once `games_at_level ≥ 20` (was `% 10 == 0` before —
that modulo gate silently skipped whole windows when batch increments crossed
multiples of 10).

`level_heuristic_history` is capped at `4 × --rolling-win` (default 160)
outcomes so poor early games at a level can age out as the model improves.
Before the cap, it was uncapped — under long-tenure runs at diff 1 the
historical mean could be permanently dragged below target even after recent
performance recovered.

When the check fires:

- **No checkpoint reload.** The in-memory model just achieved the gate, so it
  continues at the new level with those same weights. `best{prev_diff}.pt` is
  still saved as a milestone.
- **Advancement cooldown** starts: `advance_rehearsal_games + advance_cooldown_tail_batches`
  batches. While the cooldown is active, rehearsal outcomes flow into
  `win_history_heuristic` (recovery + display) but not into
  `level_heuristic_history`. When the cooldown expires, `level_heuristic_history`
  is cleared so the next advance decision is based only on fresh post-cooldown
  samples.
- Cooldown, `games_at_level`, `level_heuristic_history`, and
  `termination_history` are persisted in `latest.pt`, so a restart during
  rehearsal cannot bypass the gate.

### Termination classification (v2a)

Every rollout tags a termination reason from
`learned_ai/training/termination.py`:

| Reason | Meaning |
| --- | --- |
| `win_lt3` / `loss_lt3` | opponent (or learner) placed 9 and dropped below 3 |
| `win_blocked` / `loss_blocked` | opponent (or learner) has no legal move |
| `draw_trunc` | rollout hit `--max-ply` without a terminal state |
| `draw_rep` / `draw_50` | reserved for repetition + 50-move rule (not yet detected) |
| `infra_learner` | learner encoder/policy raised or produced no action |
| `infra_opponent` | opponent raised or produced no action on a non-terminal position |

Infra reasons **never** enter `win_history`, `win_history_heuristic`, or
`level_heuristic_history` — so a broken opponent cannot inflate advancement
statistics. A 50-game rolling termination-mix is logged in each `GameDiag`
row and rendered on Row 6 of the v2a plot script.

### Deep-batch selector (v2a)

`_is_deep_game` (1-in-20 batches use full 12-ply sim) now uses a
threshold-based counter instead of `game_count % 20 == 0`. Under
`batch_games=6` the old modulo gate fired only 1-in-30 batches or worse.


## Learned AI — v2a Training Dashboard (plot_specialist_training_2a.py)

Live 7-row dashboard for one or more scaffolded checkpoint folders.  Reads
`train_log.jsonl` and refreshes on an interval.

**Default (all four v2 specialists):**

```
.venv/bin/python tools/plot_specialist_training_2a.py
```

**Single v2a run:**

```
.venv/bin/python tools/plot_specialist_training_2a.py \
  learned_ai/checkpoints/scaffolded/s_gen_v2a/hot-recover-fixed-model-fail-assessment
```

**Single render, no live loop:**

```
.venv/bin/python tools/plot_specialist_training_2a.py \
  s_gen_v2a/hot-recover-fixed-model-fail-assessment --no-loop
```

Rows:
- Row 0 — entropy + chosen prob
- Row 1 — Malom win-move + heuristic top-1 + policy top-1 %
- Row 2 — best_win_rate + win_rate_200 + draw rate; ply on rhs
- Row 3 — sentinel chosen vs mean + gap shaded
- Row 4 — reward breakdown (sentinel + heuristic) + LR on rhs
- Row 5 — retro (outcome) reward
- **Row 6 (v2a-only) — 50-game rolling termination-reason mix, stacked area.
  Infra failure counts drawn as red x's (learner) and pink +'s (opponent).**
  Distinguishes truncation draws (`draw_trunc`) from rules-based draws so a
  high draw rate can't be mistaken for genuine drawn play.

Vertical markers on all rows:
- black dashed — Stage-1 hot-explore trigger
- green dashed — Stage-2 checkpoint restore
- green solid  — post-grace resurrection
- blue  dashed — difficulty advance (label + level annotation on top row)

The resolver prefers folders containing `train_log.jsonl` over empty stubs at
the same relative path.

| Flag | Default | Description |
| --- | --- | --- |
| `--interval N` | 20 | Refresh interval in minutes |
| `--no-loop` | off | Single render then exit (useful for saving a snapshot) |

The original `tools/plot_specialist_training.py` (6-row layout, no termination
mix) remains available for pre-v2a runs.


## Learned AI — Specialist Benchmark (bench_scaffolded.py)

Runs the **v2 SpecialistRouter** (opening + midgame + endgame specialists routed by phase) as one player, versus a matrix of heuristic-opponent configurations at multiple difficulties. Colours alternate every game. Streams results to `data/bench/scaffolded_v2_<timestamp>.jsonl` (one row per matchup) — safe for overnight runs.

**Opponent configurations** (all share `value_net_blend=20`, sentinel `_sentinel_activation_prob=0.20`):

| Config | Description |
| --- | --- |
| `raw` | GameAI only (no sentinel / vn / gap) |
| `sentinel` | GameAI + sentinel score_adjust (20% intervention) |
| `vn` | GameAI + value_net (blend 20%) |
| `gap` | GameAI + gap_net (blunder-zone exploitation) |
| `sv` | GameAI + sentinel + value_net |
| `full` | GameAI + sentinel + value_net + gap_net |
| `deep` | Full stack + max_search_depth=25 (extended tactical search) |

**Heuristic time budget**: by default (`--time-budget -1`), each opponent move uses the SAME per-difficulty cap the game applies in real play:

| Diff | Cap | Notes |
| --- | --- | --- |
| 1–5 | 15 s | Reduced to 3 s (first 2 moves) / 10 s (≤4 pieces on board) automatically |
| 6 | 30 s | Same early-placement reductions |
| 7 | 45 s | Same early-placement reductions |
| 8–10 | 60 s | Same early-placement reductions |

Pass `--time-budget SECONDS` (positive value) to override with a flat cap — useful for fast smoke tests since the game-native caps are slow (they mirror real interactive play).

**Quick smoke test (10 games, diff 5, two configs, capped at 2 s/move):**

```
.venv/bin/python scripts/bench_scaffolded.py --games 10 --difficulties 5 \
    --opponents raw,full --time-budget 2.0
```

**Overnight sweep at game-native per-difficulty budgets:**

```
.venv/bin/python scripts/bench_scaffolded.py --games 40 --difficulties 3,5,7,9
```

**Deeper specialist lookahead (25 plies) at game-native budgets:**

```
.venv/bin/python scripts/bench_scaffolded.py --games 40 --difficulties 5,7,9 \
    --specialist-ply-depth 25
```

| Flag | Default | Description |
| --- | --- | --- |
| `--games N` | 40 | Games per matchup (alternating colours) |
| `--difficulties LIST` | `3,5,7,9` | Comma-separated GameAI difficulties (1–10) |
| `--opponents LIST` | `raw,sentinel,vn,gap,sv,full,deep` | Which configs to test (comma-separated) |
| `--time-budget F` | `-1` (game-native) | Per-move heuristic budget. ≤ 0 → use game's per-difficulty caps (15/30/45/60 s + early reductions). Positive → flat override. |
| `--specialist-ply-depth N` | 15 | LookaheadAdvisor ply depth used by the specialists |
| `--max-plies N` | 400 | Max plies per game before draw |
| `--sentinel-path PATH` | `learned_ai/sentinel/checkpoints/best.pt` | Sentinel checkpoint |
| `--value-net-path PATH` | `data/value_net.npz` | Value net checkpoint |
| `--gap-net-path PATH` | `data/gap_net.npz` | Gap net checkpoint |
| `--malom-path PATH` | `/mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted` | Malom perfect DB directory |
| `--out-dir PATH` | `data/bench` | Output directory for the JSONL stream |
| `--quiet` | off | Suppress per-game outcome dots |

**Output**: `data/bench/scaffolded_v2_<YYYYMMDD_HHMMSS>.jsonl`, one row per matchup with fields `config, difficulty, games, wins, draws, losses, win_rate, draw_rate, score, elapsed_s, avg_s_per_game, time_budget_s, time_budget_mode, specialist_ply_depth, timestamp`. `time_budget_mode` records whether that row used `game_native_per_diff` or `flat_override`. Score = `(wins + 0.5 × draws) / games`. A final markdown table is printed to stdout.

**Prerequisite**: v2 specialist checkpoints must exist at `learned_ai/checkpoints/scaffolded/{s_open_v2,s_mid_v2,s_end_v2}/best.pt`.


## Learned AI — Extended Tactical Search Benchmark

Head-to-head: extended tactical search (fast_eval=False) vs no-extended (fast_eval=True).

**100-game result at 1s/move (completed 2026-07-10):** Extended 51 — No-Extended 10 — Draws 39 (51% vs 10%, 39% draws). Extended wins 5:1. Result is decisive.

**Run this bench yourself:**

```
.venv/bin/python /tmp/bench_ext_vs_noext.py
```

Variables at top of script: `BUDGET` (seconds/move), `N_GAMES` (total games).

## Learned AI — Overseer Training (RETIRED)

The overseer meta-layer has been removed. Specialists now act directly for their own phase (place → opening, move ≥6 pieces → midgame, move/fly ≤5 pieces → endgame). The scripts below still exist but are not used in the v2 pipeline.

```
.venv/bin/python scripts/train_scaffolded_overseer_parallel.py  \
  --midgame-ckpt  learned_ai/checkpoints/scaffolded/s_mid/best.pt  \
  --endgame-ckpt  learned_ai/checkpoints/scaffolded/s_end/best.pt  \
  --opening-ckpt  learned_ai/checkpoints/scaffolded/s_open/best.pt  \
  --max-games 10000 --max-ply 140  \
  --malom /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted  \
  --workers 8
```

## Puzzle Generators

### Opening / Placement Puzzles

Uses Malom DB path from `data/settings.json`.

```
.venv/bin/python tools/placement_puzzle_generator.py  \
  --depth random --max-winning-moves 2 --side random
```

| Flag | Default | Description |
| - | - | - |
| `--side W\|B\|random` | random | Which side has the winning move |
| `--depth 0\|4\|5\|6\|7` | 0 | Target win depth in winner moves (0 = random) |
| `--max-winning-moves N` | 2 | Reject positions with more than N winning first moves |
| `--count N` | 0 | Puzzles to generate (0 = run forever) |
| `--attempts N` | 3000 | Positions sampled per puzzle attempt |
| `--out PATH` | `data/puzzles/` | Output directory |
| `--print` | off | Print each puzzle JSON to stdout |


### Midgame Puzzles (Malom DB)

```
.venv/bin/python tools/malom_puzzle_generator.py  \
  --depth 6 --max-winning-moves 2 --side random  \
  --min-pieces 4 --max-pieces 16
```

| Flag | Default | Description |
| - | - | - |
| `--side W\|B\|random` | random | Which side has the winning move |
| `--depth 0\|4\|5\|6\|7` | 0 | Target win depth (0 = random) |
| `--max-winning-moves N` | 2 | Reject positions with more than N winning first moves |
| `--min-pieces N` | 4 | Minimum pieces per side |
| `--max-pieces N` | 7 | Maximum pieces per side (raise for richer midgame) |
| `--count N` | 0 | Puzzles to generate (0 = run forever) |
| `--attempts N` | 3000 | Positions sampled per puzzle attempt |
| `--out PATH` | `data/puzzles/` | Output directory |
| `--print` | off | Print each puzzle JSON to stdout |


### Endgame Puzzles (Retrograde DB)

```
.venv/bin/python tools/puzzle_generator.py  \
  --depth random --max-winning-moves 2 --side random --random-db
```

| Flag | Default | Description |
| - | - | - |
| `--side W\|B\|random` | random | Which side has the winning move |
| `--depth 3\|4\|5\|6\|7\|random` | random | Target win depth in winner moves |
| `--max-winning-moves N` | 2 | Reject positions with more than N winning first moves |
| `--db FILE\|random` | random | Specific endgame .wdl file from `data/endgame/` |
| `--random-db` | off | Pick a new random DB file for every attempt (cross-table) |
| `--count N` | 0 | Puzzles to generate (0 = run forever) |
| `--attempts N` | 5000 | Positions sampled per puzzle attempt |
| `--out PATH` | `data/puzzles/` | Output directory |
| `--print` | off | Print each puzzle JSON to stdout |

---

### Unified Puzzle Generator (recommended — all types, minimax-verified)

`tools/unified_puzzle_generator.py` replaces the three legacy scripts above.
Puzzles are minimax-verified (exact depth, opponent plays hardest defense) and
require a single forced-win path (`--max-winning-moves 1` default).
Outputs to `data/puzzles/endgame/`, `data/puzzles/malom/`, `data/puzzles/placement/`.

**Single type:**
```
# 200 midgame puzzles, any depth 4-10, 6 parallel workers
.venv/bin/python tools/unified_puzzle_generator.py \
  --type midgame --depth 0 --count 200 --workers 6

# 200 endgame puzzles, any depth 3-10
.venv/bin/python tools/unified_puzzle_generator.py \
  --type endgame --depth 0 --count 200 --workers 6

# 200 placement puzzles, any depth 4-10
.venv/bin/python tools/unified_puzzle_generator.py \
  --type placement --depth 0 --count 200 --workers 6

# Specific depth / side
.venv/bin/python tools/unified_puzzle_generator.py \
  --type midgame --depth 5 --side W --count 50 --workers 4
```

**Batch mode** (edit `data/puzzles/batch_config.json` first, then run once):
```
.venv/bin/python tools/unified_puzzle_generator.py \
  --batch data/puzzles/batch_config.json --workers 6
```
Batch mode is resume-aware: already-met quotas are skipped.

**`data/puzzles/batch_config.json` format:**
```json
{
  "cells": [
    {"type": "midgame",   "side": "random", "depth": 0, "count": 200},
    {"type": "placement", "side": "random", "depth": 0, "count": 200},
    {"type": "endgame",   "side": "random", "depth": 0, "count": 200}
  ],
  "max_winning_moves": 1,
  "min_hardness": 3.0
}
```

| Flag | Default | Description |
| - | - | - |
| `--type endgame\|midgame\|placement` | — | Puzzle type (required unless `--batch`) |
| `--depth 0\|3–10` | 0 | Target win depth; 0 = random within type range |
| `--side W\|B\|random` | random | Which side has the winning move |
| `--max-winning-moves N` | 1 | Max first moves that win within depth budget |
| `--min-hardness F` | 3.0 | Minimum hardness score to accept a puzzle |
| `--count N` | 0 | Puzzles to generate (0 = run forever) |
| `--attempts N` | 5000/3000 | Positions sampled per attempt (endgame/malom) |
| `--workers N` | cpu_count−1 | Parallel worker processes |
| `--batch PATH` | — | Batch config JSON; overrides `--type/--depth/--side` |
| `--out PATH` | type-specific | Override output directory |

**Tags written to puzzle JSON:**

| Tag | Meaning |
| - | - |
| `endgame` / `midgame` / `placement` | Game phase |
| `win-in-N` | Exact verified forced-win depth (3–10) |
| `unique-move` | Exactly one winning first move within budget |
| `bottleneck` | Same as `unique-move` |
| `two-solutions` | Two winning first moves within budget |
| `high-branching` | ≥ 6 legal moves at the root |
| `near-endgame` | Midgame with ≤ 8 total pieces |
| `early-game` | Midgame with ≥ 14 total pieces |


