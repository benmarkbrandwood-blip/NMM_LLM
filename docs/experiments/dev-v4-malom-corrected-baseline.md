# Dev v4 Malom-Corrected Fresh Baseline

## Status and Claim Boundary

Historical smoke experiment ID: `dev-v4-malom-corrected-fresh-v1`

Managed long-run experiment ID: `dev-v4-managed-baseline-v1`

The one-game integration smoke completed on 20 July 2026 with status
`passed_with_observation`; the checkpoint-reporting observation has since been
resolved on `dev`. The authorized managed long run
`managed-v4-baseline-v1` reached `managed_plan_completed` with
`completed_games=5000` / `completed_segments=20` on 21 July 2026 (UTC). Training
completion is not playing-strength evidence and does not authorize promotion.
The paired-runner repair is complete. The owner reviewed the 107 generated
candidates, removed original position 101, and accepted the resulting
106-start package. The next evaluation proposal has a `needs_decision`
readiness verdict: its technical gates pass, while a new explicit product
authorization remains open; see
[`dev-v4-formal-paired-eval-v1.md`](dev-v4-formal-paired-eval-v1.md). A
complete recommended training configuration remains recorded below for
lineage.

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

The intended long-run path boundary is:

- control/output base: a new ignored directory under
  `learned_ai/checkpoints/scaffolded/s_gen_v2_sector_corrected`, with one
  isolated `segments/segment-NNNN` directory per process;
- SpecialistDB: `data/specialist_db.sector_corrected.sqlite`.

On 20 July 2026, the SpecialistDB above was verified read-only with zero rows
in `positions`, `winning_lines`, and `preferred_plays`, and metadata
`malom_label_version=sector-corrected-v1`. That was the historical pre-launch
state, not its current state. The completed managed run populated the database;
on 22 July it had 132,182 positions, 41,904 current-version Malom labels, 916
winning lines, no preferred plays, and lineage root
`managed-v4-baseline-v1-segment-0001`. Do not reuse it as the empty input to a
new fresh experiment, and do not replace it with the staged maintainer DB.

## Agent-Managed Long-Run Definition

The following is the Agent-selected corrected v4-style default. An actual run
freezes these values, the exact Git commit, local path-config identity, and
resource bounds in `plan.json`. Recording this default does not approve a
smoke or long run. Launch authorization remains a separate product decision in
`authorization.json`.

| Choice | Recommended value |
| --- | --- |
| Update algorithm | A2C; omit `--ppo` |
| Imitation | No S1A warm-start and no imitation mini-step during RL |
| Opponents | 50% frozen target and 50% heuristic; refresh target every 50 games; adaptive difficulty 1 through 20 |
| Rollout | Complete rollout, `sim_ply_depth=5`, no branch rollouts, `max_ply=60` |
| Search work | 500,000 native search nodes per heuristic move; single-threaded and no wall-clock search cutoff |
| Temperature | Start `0.90`; linearly reach `0.20` at 80% total progress |
| Game budget | 5,000 |
| Seed | 42 |
| Concurrency | `batch_games=1` |
| Process segments | End every 250 games; the supervisor may continue only from the verified preceding `latest.pt` by explicit `exact-resume` |
| Checkpoints | Save `latest.pt` every 50 games through `--log-every=50` |
| Monitoring | Record every 50 games; audit integrity and resources at each 250-game boundary |
| Ordinary stopping | Do not stop for an intermediate win-rate result |
| Quarantine stopping | Stop for non-finite values, Malom/DB identity change, wrong label version, checkpoint corruption, CUDA failure, or broken evidence chain |

The author-`main` bundle does not alter any row in this table. Its model and
logs are not warm-start material, a target model, or a formal baseline.

### Launch-control closure

The two previously identified launch-control gaps are closed in code and
focused tests:

1. `--no-imitation-mix` is independent of `--no-s1a-warmstart`. Disabled
   mixing never reads the dataset. Enabled mixing fails closed when its
   required dataset is missing, corrupt, empty, or inconsistent. Both controls
   are visible in preflight, run manifests, and exact-resume semantics.
2. `--heuristic-node-budget` selects a deterministic per-move native search
   cap and is mutually exclusive with an explicit time budget. Fixed-work mode
   requires one Rust search thread and fails closed if the native backend
   cannot honor the contract. Per-game logs record the configured node budget,
   search-call count, and actual cumulative nodes.

The 500,000-node default is an Agent-owned starting choice, not a strength
claim. Fixed-node heuristic search must follow
[`docs/fixed-node-heuristic-search.md`](../fixed-node-heuristic-search.md)
(Sanmill-aligned stand-pat leaves, same candidate set in and out). The
immutable plan, not this calibration note, is authoritative for an actual run.

PPO is not a launch blocker for this A2C baseline, but it is quarantined for a
separate reason: sampled old log probabilities use temperature-scaled logits,
while the current PPO update recomputes unscaled logits. Do not enable `--ppo`
until a deterministic ratio-at-collection regression test and reviewed fix
exist. See
[`docs/evidence/author-main-generalist-audit-2026-07-20.md`](../evidence/author-main-generalist-audit-2026-07-20.md).

### Formal evaluation status — needs decision

Formal evaluation remains separate from training completion. The candidate and
architecture-matched scratch-init bundles have been exported and CPU-verified,
but expert review invalidated the original 64-start / 256-pair freeze proposal
and found draw-lifecycle and partial-ledger defects in the paired runner. Those
runner prerequisites are now repaired and covered by focused tests. The owner
reviewed all 107 generated candidates, excluded original position 101, and
accepted the other 106. Deterministic start reuse is rejected, and the runtime
contract now binds the clean code, device, environment, route, components, and
feature ablation. A clean read-only audit reverified the corpus, both bundles,
isolated targets, and an in-memory specification. Freeze and execution remain
unauthorized only because a new explicit product authorization is outstanding;
promotion and publication are outside this Stage-0 contract.
The controlling record is
[`dev-v4-formal-paired-eval-v1.md`](dev-v4-formal-paired-eval-v1.md), with
the evidence and decision rationale in
[`dev-v4-formal-paired-eval-v1-decision-brief.md`](dev-v4-formal-paired-eval-v1-decision-brief.md)
and the readiness result in
[`dev-v4-stage0-readiness-2026-07-22.md`](../evidence/dev-v4-stage0-readiness-2026-07-22.md).

After a new product authorization and a repeated clean-state/output check, the
next proposed experiment is only a Stage-0 training-signal diagnostic:

- 106 owner-accepted unique playable stable positions selected from 107 FENs
  projected from 108 Sanmill `action=p` keys, with original review position 101
  excluded and two pending-removal keys retained only as successor provenance;
- placement 106 / movement 0 / flying 0;
- exactly 106 colour-swapped pairs / 212 games, with no deterministic start
  reuse;
- `policy-argmax-v1` versus the verified scratch-init control;
- the normal interval described only as variation across the fixed convenience
  corpus.

The Oracle is separately engine-derived rather than a direct training-book
export, but 28 of 106 selected positions overlap named-line trajectories and
the full set is early placement. It is therefore not demonstrated held-out or
training-disjoint. The evaluation also zeroes the 72-feature lookahead block
used during training, so even an accepted result is ablation evidence, not a
training-route-aligned strength or promotion result.

The existing random smoke bundle, the three infrastructure positions, and the
rejected 64-position draft are not formal evaluation corpora.

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
resolved in code and tests. The authorized managed long run
`managed-v4-baseline-v1` later completed at 5000 games. Further training still
requires a **new** immutable plan and a **new** product authorization; the
completed plan remains immutable and does not extend itself.

Before another smoke or long run, use the managed supervisor to freeze a new
immutable plan on the intended clean commit. The plan must bind the resolved
training semantics, local path-config hash, game and segment bounds, wall-time
envelope, component exclusions, and fixed-node work. The product owner then
approves only the objective and resource envelope through the separate
authorization contract. Plan creation alone never authorizes launch.

Immediately before launch, rerun the training-readiness workflow and re-check
the corrected SpecialistDB identity, new output directory, resolved work
budget, component manifest, native fixed-work probe, and quarantine rules. A
new disposable smoke must reach at least one RL update and prove that imitation
stayed disabled. Do not reuse the original smoke DB, smoke checkpoint, or any
author-`main` checkpoint.
