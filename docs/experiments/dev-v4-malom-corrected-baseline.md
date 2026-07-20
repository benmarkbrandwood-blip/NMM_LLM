# Dev v4 Malom-Corrected Fresh Baseline

## Status and Claim Boundary

Experiment ID: `dev-v4-malom-corrected-fresh-v1`

The one-game integration smoke completed on 20 July 2026 with status
`passed_with_observation`; the checkpoint-reporting observation has since been
resolved on `dev`. The long run has not started. A complete recommended
configuration is recorded below, but the owner has not yet accepted or revised
it. Two parts of that proposal also cannot be expressed safely by the current
CLI: disabling ongoing imitation mixing and selecting measured fixed search
work instead of a wall-clock opponent cutoff.

This is a fresh-initialised, Malom-corrected **v4-style Generalist baseline**.
It is not the v5 `reference_safe_baseline`, a release candidate, or evidence of
playing strength. The author's continuing `main` run remains a separately
labelled legacy comparison.

## Fixed Lineage and Component Boundaries

| Component | First-run decision |
| --- | --- |
| Code | Run from a clean `dev`; record `git rev-parse HEAD` at launch |
| Execution | One Windows process on one CUDA device; no distributed or C++ trainer |
| Model checkpoint | Fresh initialisation; omit both `--resume` and `--auto-resume-best` |
| Author `main` bundle | Diagnostic reference only; never an input, resume source, or `dev` acceptance artefact |
| Output | Use a new, dedicated directory with no historical checkpoints or logs |
| Malom | Enable the machine-local `malom_db_path` through `data/training_paths.local.json` and the corrected decoder |
| SpecialistDB | Start with an empty DB carrying `malom_label_version=sector-corrected-v1` |
| HumanDB | Human frequencies and outcomes may be used; historical unversioned Malom columns remain masked |
| Sentinel | Disabled with `--no-sentinel` |
| ValueNet | Disabled with `--no-value-net` |
| GapNet | Disabled with `--no-gap-net` |
| Temperature | Recovery does not reheat; the smoke uses the current `0.90` default |
| Checkpoint roles | `latest.pt` is the continuation snapshot; `best.pt` is an optional, evaluation-gated model-selection snapshot |
| Interruption | Never resume automatically; only a separately preflighted explicit `exact-resume` from a compatible v2 envelope may continue the baseline |

The intended long-run paths are:

- output: `learned_ai/checkpoints/scaffolded/s_gen_v2_sector_corrected`;
- SpecialistDB: `data/specialist_db.sector_corrected.sqlite`.

On 20 July 2026, the SpecialistDB above was verified read-only with zero rows
in `positions`, `winning_lines`, and `preferred_plays`, and metadata
`malom_label_version=sector-corrected-v1`. Verify those facts again immediately
before the long run. Do not use the active baseline DB for the smoke.

## Recommended Long-Run Definition - Not Yet Frozen

The following is the recommended corrected v4-style baseline. Recording it
does not approve a smoke or long run, and it does not silently convert the
proposal into an owner decision.

| Choice | Recommended value |
| --- | --- |
| Update algorithm | A2C; omit `--ppo` |
| Imitation | No S1A warm-start and no imitation mini-step during RL |
| Opponents | 50% frozen target and 50% heuristic; refresh target every 50 games; adaptive difficulty 1 through 20 |
| Rollout | Complete rollout, `sim_ply_depth=5`, no branch rollouts, `max_ply=60` |
| Search work | Fixed measured work per move; no wall-clock search cutoff |
| Temperature | Start `0.90`; linearly reach `0.20` at 80% total progress |
| Game budget | 5,000 |
| Seed | 42 |
| Concurrency | `batch_games=1` |
| Process segments | End every 250 games and continue only by explicit `exact-resume` |
| Checkpoints | Save `latest.pt` every 50 games through `--log-every=50` |
| Monitoring | Record every 50 games; audit integrity and resources at each 250-game boundary |
| Ordinary stopping | Do not stop for an intermediate win-rate result |
| Quarantine stopping | Stop for non-finite values, Malom/DB identity change, wrong label version, checkpoint corruption, CUDA failure, or broken evidence chain |

The author-`main` bundle does not alter any row in this table. Its model and
logs are not warm-start material, a target model, or a formal baseline.

### Launch-control gaps

The proposal is not yet smoke-ready for two independent reasons:

1. `--no-s1a-warmstart` disables only the pre-RL warm-start. The trainer still
   loads `human_imitation2.npz` when present and applies an imitation mini-step
   after RL updates. Add `--no-imitation-mix`, make the disabled state visible
   in preflight/log/contract evidence, and test launch plus exact-resume
   compatibility. That file is absent from the current Windows checkout, but
   absence is not an experiment control and must not substitute for the flag.
2. A negative `--time-budget` currently selects the trainer's automatic
   per-difficulty wall-clock seconds. It does not mean unlimited or fixed-work
   search. Add a deterministic work-budget control and log the effective work,
   or explicitly revise this proposed row before readiness review.

PPO is not a launch blocker for this A2C baseline, but it is quarantined for a
separate reason: sampled old log probabilities use temperature-scaled logits,
while the current PPO update recomputes unscaled logits. Do not enable `--ppo`
until a deterministic ratio-at-collection regression test and reviewed fix
exist. See
[`docs/evidence/author-main-generalist-audit-2026-07-20.md`](../evidence/author-main-generalist-audit-2026-07-20.md).

### Formal evaluation proposal - also not frozen

Formal evaluation is not required to implement the launch controls or run a
disposable infrastructure smoke. It must be frozen before candidate results
are inspected or used for promotion. The current proposal is:

- a compatible immutable baseline bundle, still to be selected;
- 64 reviewed, training-disjoint starts spanning placement, movement, and
  flying phases;
- 256 colour-swapped pairs, for 512 games total;
- fixed work per move and `max_ply=200`, with overflow scored as a draw;
- accept only when the 95% confidence-interval lower bound is above zero,
  reject only when its upper bound is below zero, otherwise report
  `inconclusive`.

The existing random smoke bundle and three infrastructure positions are not a
formal baseline or evaluation corpus.

## Required Preflight Evidence

Before either smoke or long training, record:

1. `git rev-parse HEAD` and `git status --short --branch`;
2. the exact command and random seed;
3. resolved logical path keys, without copying host-specific absolute values
   into tracked files;
4. output-directory existence and contents;
5. SpecialistDB label version and row counts;
6. the focused Generalist and 102-test Malom/provenance results.

The launch log must show all of the following:

- `No checkpoint found` and a scratch source checkpoint;
- Sentinel, ValueNet, and GapNet disabled by CLI;
- the Malom database available;
- a trusted `sector-corrected-v1` SpecialistDB;
- the selected device, output path, and actual temperature.

Any mismatch stops the run. A missing legacy component is not equivalent to an
explicitly disabled component for this experiment.

## One-Game Integration Smoke

The following is the exact historical command that produced the first smoke.
It is retained as evidence, not as a reusable current command. The hardened CLI
now also requires a launch mode and run ID; the reviewed command must state its
start mode explicitly. The proposed pure RL baseline still needs the new
imitation-mix control. Any future smoke must use new disposable paths and a
freshly reviewed command.

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --paths-config data\training_paths.local.json `
  --out-dir learned_ai\checkpoints\smoke\s_gen_v2_v4_malom_corrected_fresh_v1 `
  --specialist-db data\specialist_db.smoke.v4_malom_corrected_fresh_v1.sqlite `
  --no-sentinel `
  --no-value-net `
  --no-gap-net `
  --temp-start 0.90 `
  --seed 42 `
  --max-games 1 `
  --batch-games 1 `
  --max-ply 40 `
  --sim-ply-depth 2 `
  --minimal-rollouts `
  --no-s1a-warmstart
```

The absence of `--resume` and `--auto-resume-best` is part of the baseline
lineage decision. The smoke also omits `--ppo` and the archived mixed-opponent
changes to keep the integration check bounded. It predates the recommended
long-run definition and did not prove that RL imitation mixing was disabled;
the one-game run did not reach a periodic RL update. The smoke proves only that
the selected components, writers, and corrected label boundary initialised and
completed one bounded game. It does not approve a long run or establish
strength.

## Smoke Result - 20 July 2026

The smoke ran from clean `dev` commit
`80f4a1fe525d98706b1b0913083f2c2067f8bf66`. Preflight reconfirmed that the
disposable paths did not exist, the active baseline output directory did not
exist, and the active SpecialistDB was an empty `sector-corrected-v1` database.
Immediately before launch, the focused trainer tests reported `9 passed`; the
Malom/provenance suite reported `102 passed, 498 subtests passed`.

| Check | Result |
| --- | --- |
| Process | Exit code `0` after approximately 24.4 seconds |
| Device | CUDA |
| Lineage | `source_checkpoint=scratch`; no resume option |
| Legacy learned inputs | Sentinel, ValueNet, and GapNet disabled by CLI |
| Live data inputs | Malom and HumanDB loaded |
| Game | One `vs_frozen` game, learner Black, 33 ply, outcome `-1.0` |
| Schedule | Temperature `0.9` |
| Training log | One row in `train_log.jsonl` |
| Final checkpoint | `latest.pt`, stage `s_gen_v2`, game count `1` |
| Smoke SpecialistDB | 32 positions, 10 Malom-labelled, no winning or preferred lines |
| Active baseline DB | Unchanged: all three data tables and Malom-labelled count remain zero |
| SQLite integrity | `ok` for both smoke and active baseline databases |

The single-game outcome and diagnostic rates are integration observations only;
they have no strength or acceptance authority. HumanDB emitted the expected
warning that its historical Malom labels were masked because its label version
is missing, while its human statistics remained available.

The ignored local evidence is under:

- `learned_ai/checkpoints/smoke/s_gen_v2_v4_malom_corrected_fresh_v1`;
- `data/specialist_db.smoke.v4_malom_corrected_fresh_v1.sqlite`.

That output directory contains `train_log.jsonl`, `latest.pt`, and the local
`smoke_manifest.json` with the exact command and result. These generated files
remain ignored and are not part of this documentation commit.

### Checkpoint observation and superseding recovery contract

The trainer's final console message named the output path `best.pt`, but the
one-game smoke produced only `latest.pt`; `best.pt` was absent. The checkpoint
that exists is readable and records stage `s_gen_v2`, game count `1`,
`source_checkpoint=scratch`, and temperature `0.9`.

This discrepancy did not invalidate the bounded initialisation smoke. Commit
`bf9472c` subsequently changed the final report to always name `latest.pt` and
to name `best.pt` only when that file exists. Otherwise it reports that no best
checkpoint was created. Regression tests also lock the actual best-checkpoint
gate: it is evaluated at a logging checkpoint, requires at least 10 heuristic
games, and requires a win rate strictly above the prior best at that
difficulty. A one-game run is therefore not expected to create `best.pt`.

The hardened experiment uses this conservative recovery policy:

- its initial launch omits both `--resume` and `--auto-resume-best`;
- `best.pt` is optional model-selection evidence, not an operational recovery
  requirement;
- after an interruption, do not continue automatically; inspect and verify
  `latest.pt`, then start a separately recorded `exact-resume` segment with an
  explicit `--resume` path;
- exact resume restores model, optimiser, counters, rolling histories,
  curriculum, target state, Python/NumPy/PyTorch/CUDA RNG state, component RNG
  state, data cursor, and SpecialistDB identity;
- historical pre-v2 checkpoints remain explicit `weights-only` imports and
  reset optimiser, RNG, counters, and curriculum state.

The bounded parity audit on 20 July 2026 proved that a continuous two-game
CUDA run and a one-game run followed by one exact-resume game produced equal
model, optimiser, scheduler/scaler, RNG, trainer, data, log, and SpecialistDB
semantic state. The full evidence is recorded in
[`docs/evidence/v4-infrastructure-hardening-2026-07-20.md`](../evidence/v4-infrastructure-hardening-2026-07-20.md).

The original one-game run also did not create `update_log.jsonl`, so it did not
exercise periodic update-log or best-checkpoint cadence. The reporting fix is
covered by focused tests; a new post-fix smoke has not been launched without a
separate readiness gate and launch authorisation.

## Long-Run Launch Gate

The bounded initialisation smoke passed, and its checkpoint observation is
resolved in code and tests. Before another smoke, the owner must accept or
revise the recommended definition, the explicit imitation-mix disable control
must exist, and the fixed-work search row must be implemented or deliberately
changed. The new smoke must use the hardened launch contract, disposable paths,
and enough work to exercise an RL update while proving that imitation stayed
disabled.

Before the long run, rerun readiness on the intended clean launch commit, then
re-check the empty corrected SpecialistDB, new output directory, resolved work
budget, component manifest, and quarantine rules immediately before launch. Do
not reuse the original smoke DB, smoke checkpoint, or any author-`main`
checkpoint.
