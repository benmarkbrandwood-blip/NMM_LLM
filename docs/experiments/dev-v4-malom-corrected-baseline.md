# Dev v4 Malom-Corrected Fresh Baseline

## Status and Claim Boundary

Experiment ID: `dev-v4-malom-corrected-fresh-v1`

The one-game integration smoke completed on 20 July 2026 with status
`passed_with_observation`; the checkpoint-reporting observation has since been
resolved on `dev`. The long run has not started. Its update algorithm, opponent
schedule, temperature start, game budget, concurrency, monitoring cadence, and
stop criteria still need to be frozen.

This is a fresh-initialised, Malom-corrected **v4-style Generalist baseline**.
It is not the v5 `reference_safe_baseline`, a release candidate, or evidence of
playing strength. The author's continuing `main` run remains a separately
labelled legacy comparison.

## Fixed Lineage and Component Boundaries

| Component | First-run decision |
| --- | --- |
| Code | Run from a clean `dev`; record `git rev-parse HEAD` at launch |
| Model checkpoint | Fresh initialisation; omit both `--resume` and `--auto-resume-best` |
| Output | Use a new, dedicated directory with no historical checkpoints or logs |
| Malom | Enable the machine-local `malom_db_path` through `data/training_paths.local.json` and the corrected decoder |
| SpecialistDB | Start with an empty DB carrying `malom_label_version=sector-corrected-v1` |
| HumanDB | Human frequencies and outcomes may be used; historical unversioned Malom columns remain masked |
| Sentinel | Disabled with `--no-sentinel` |
| ValueNet | Disabled with `--no-value-net` |
| GapNet | Disabled with `--no-gap-net` |
| Temperature | Recovery does not reheat; the smoke uses the current `0.90` default |
| Checkpoint roles | `latest.pt` is the continuation snapshot; `best.pt` is an optional, evaluation-gated model-selection snapshot |
| Interruption | Never resume automatically; inspect the run and use a recorded explicit `--resume` only if weight continuation is accepted |

The intended long-run paths are:

- output: `learned_ai/checkpoints/scaffolded/s_gen_v2_sector_corrected`;
- SpecialistDB: `data/specialist_db.sector_corrected.sqlite`.

On 20 July 2026, the SpecialistDB above was verified read-only with zero rows
in `positions`, `winning_lines`, and `preferred_plays`, and metadata
`malom_label_version=sector-corrected-v1`. Verify those facts again immediately
before the long run. Do not use the active baseline DB for the smoke.

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

Use new disposable paths. If either path already contains a prior run, choose
a new name rather than mixing evidence.

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
changes to keep the integration check bounded; this does not yet freeze the
long-run update algorithm or opponent schedule. The smoke proves only that the
selected components, writers, and corrected label boundary initialise and
complete one bounded game. It does not approve a long run or establish
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

### Checkpoint observation and resolution

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

The first experiment uses this conservative recovery policy:

- its initial launch omits both `--resume` and `--auto-resume-best`;
- `best.pt` is optional model-selection evidence, not an operational recovery
  requirement;
- after an interruption, do not continue automatically; inspect `latest.pt`
  and, if continuation is accepted, start a separately recorded segment with
  an explicit `--resume` path;
- current resume restores weights and selected scalar counters, but not the
  optimiser, rolling histories, difficulty-local counters, target age, or RNG
  state. It is a weight continuation, not exact trainer-state recovery. If an
  exact continuation is required, stop and extend the checkpoint schema and
  tests before launch.

The original one-game run also did not create `update_log.jsonl`, so it did not
exercise periodic update-log or best-checkpoint cadence. The reporting fix is
covered by focused tests; a new post-fix smoke has not been launched without a
separate readiness gate and launch authorisation.

## Long-Run Launch Gate

The bounded initialisation smoke passed, and its checkpoint observation is
resolved in code and tests. Before the long run, freeze the update algorithm,
opponent schedule, temperature start, game budget, seed, concurrency,
checkpoint cadence, monitor interval, and stop criteria. Re-run the readiness
preflight and a newly authorised bounded smoke against disposable paths on the
intended launch commit, then re-check the active empty SpecialistDB and
dedicated long-run output immediately before launch. Do not reuse the original
smoke DB or smoke checkpoint.
