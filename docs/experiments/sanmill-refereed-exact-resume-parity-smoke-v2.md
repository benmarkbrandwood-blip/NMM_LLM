# Sanmill-Refereed Exact-Resume Parity Smoke v2

## Status and authority

Status: `prepared_unlaunched`

Experiment: `dev-v4-sanmill-refereed-fresh-v1`

This is the only authorised retry of the stopped v1 parity comparison. The
[v1 failure record](../evidence/sanmill-refereed-exact-resume-parity-smoke-v1-failure-2026-08-08.md)
proves that its second segment never launched and that its completed outputs
and databases are quarantined. V2 uses new run IDs, output paths, and empty
SpecialistDB files. It must not read or resume any v1 artefact.

The delegated technical authority is staged. After this contract and a new
readiness record are published, the continuous reference and first segment may
run once if both final preflights pass. The second segment may run once only
after segment 0001 completes and its exact-resume preflight passes from the
same clean published source commit. A failure consumes the affected authority
and stops v2 without automatic retry.

## Frozen question, semantics, and bounds

V2 changes no gameplay or learning semantics from the v1 contract. It asks
whether a real process boundary followed by integrity-verified `exact-resume`
produces the same future state as an uninterrupted route after finite A2C
updates.

Both two-game routes freeze:

- fresh random seed 42 and `batch_games=1`;
- A2C with `update_every=8` so every game reaches a normal periodic update
  before a segment boundary;
- Sanmill as referee for every complete logical turn and as the non-self-play
  opponent;
- one 1,000-node fixed-work level, no curriculum advancement, and no wall-clock
  search limit;
- `self_play_ratio=0.60`, yielding one Sanmill-opponent and one frozen-target
  primary game in each route;
- `max_ply=120`, simulation depth 5, minimal rollouts, and no branches; and
- temperature start 0.90 with recovery, Sentinel, ValueNet, GapNet, S1A,
  S1B, imitation mixing, and opening forcing disabled.

The total ceiling is four primary games, 480 logical plies, two Sanmill-search
opponent games at 1,000 nodes per search, and two frozen-target games. No v2
checkpoint may become retained-run lineage material.

## Isolated inputs and outputs

The continuous route uses:

```text
out/sanmill-refereed-exact-resume-parity-v2/continuous
data/specialist_db.sanmill_refereed_exact_resume_parity_v2.continuous.sqlite
```

The segmented route uses two output directories and one database:

```text
out/sanmill-refereed-exact-resume-parity-v2/segment-0001
out/sanmill-refereed-exact-resume-parity-v2/segment-0002
data/specialist_db.sanmill_refereed_exact_resume_parity_v2.segmented.sqlite
```

Both databases must be newly created by the current SpecialistDB
implementation, empty, unbound, `quick_check=ok`, and stamped
`malom_label_version=sector-corrected-v1` before readiness is published.

## Exact commands

All commands run from the repository root. A read-only preflight replaces
`--launch smoke` with `--preflight smoke` and omits `--run-id`. There must be
no Git change between the final fresh preflights and the parity comparison.

### Uninterrupted reference

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --launch smoke `
  --run-id sanmill-refereed-exact-resume-parity-v2-continuous `
  --paths-config data\training_paths.local.json `
  --experiment-id dev-v4-sanmill-refereed-fresh-v1 --start-mode fresh `
  --out-dir out\sanmill-refereed-exact-resume-parity-v2\continuous `
  --specialist-db data\specialist_db.sanmill_refereed_exact_resume_parity_v2.continuous.sqlite `
  --referee-engine sanmill --opponent-engine sanmill `
  --sanmill-node-ladder 1000 --curriculum-advance-policy disabled `
  --diff-start 1 --diff-max 1 --self-play-ratio 0.60 --seed 42 `
  --max-games 2 --segment-games 2 --segment-stop-game 2 `
  --max-ply 120 --batch-games 1 --sim-ply-depth 5 `
  --temp-start 0.90 --update-every 8 --log-every 1 `
  --max-branches-per-game 0 --minimal-rollouts --no-recovery `
  --no-sentinel --no-value-net --no-gap-net `
  --no-s1a-warmstart --no-s1b-refresher `
  --no-imitation-mix --no-opening-forcing
```

### First segmented process

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --launch smoke `
  --run-id sanmill-refereed-exact-resume-parity-v2-segment-0001 `
  --paths-config data\training_paths.local.json `
  --experiment-id dev-v4-sanmill-refereed-fresh-v1 --start-mode fresh `
  --out-dir out\sanmill-refereed-exact-resume-parity-v2\segment-0001 `
  --specialist-db data\specialist_db.sanmill_refereed_exact_resume_parity_v2.segmented.sqlite `
  --referee-engine sanmill --opponent-engine sanmill `
  --sanmill-node-ladder 1000 --curriculum-advance-policy disabled `
  --diff-start 1 --diff-max 1 --self-play-ratio 0.60 --seed 42 `
  --max-games 2 --segment-games 1 --segment-stop-game 1 `
  --max-ply 120 --batch-games 1 --sim-ply-depth 5 `
  --temp-start 0.90 --update-every 8 --log-every 1 `
  --max-branches-per-game 0 --minimal-rollouts --no-recovery `
  --no-sentinel --no-value-net --no-gap-net `
  --no-s1a-warmstart --no-s1b-refresher `
  --no-imitation-mix --no-opening-forcing
```

### Exact-resume process

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --launch smoke `
  --run-id sanmill-refereed-exact-resume-parity-v2-segment-0002 `
  --parent-run-id sanmill-refereed-exact-resume-parity-v2-segment-0001 `
  --paths-config data\training_paths.local.json `
  --experiment-id dev-v4-sanmill-refereed-fresh-v1 `
  --start-mode exact-resume `
  --resume out\sanmill-refereed-exact-resume-parity-v2\segment-0001\latest.pt `
  --out-dir out\sanmill-refereed-exact-resume-parity-v2\segment-0002 `
  --specialist-db data\specialist_db.sanmill_refereed_exact_resume_parity_v2.segmented.sqlite `
  --referee-engine sanmill --opponent-engine sanmill `
  --sanmill-node-ladder 1000 --curriculum-advance-policy disabled `
  --diff-start 1 --diff-max 1 --self-play-ratio 0.60 --seed 42 `
  --max-games 2 --segment-games 1 --segment-stop-game 2 `
  --max-ply 120 --batch-games 1 --sim-ply-depth 5 `
  --temp-start 0.90 --update-every 8 --log-every 1 `
  --max-branches-per-game 0 --minimal-rollouts --no-recovery `
  --no-sentinel --no-value-net --no-gap-net `
  --no-s1a-warmstart --no-s1b-refresher `
  --no-imitation-mix --no-opening-forcing
```

### Exact semantic comparison

```powershell
.\.venv\Scripts\python.exe scripts\verify_resume_parity.py `
  --continuous-checkpoint out\sanmill-refereed-exact-resume-parity-v2\continuous\latest.pt `
  --resumed-checkpoint out\sanmill-refereed-exact-resume-parity-v2\segment-0002\latest.pt `
  --continuous-log out\sanmill-refereed-exact-resume-parity-v2\continuous\train_log.jsonl `
  --resumed-log out\sanmill-refereed-exact-resume-parity-v2\segment-0001\train_log.jsonl `
  --resumed-log out\sanmill-refereed-exact-resume-parity-v2\segment-0002\train_log.jsonl `
  --continuous-database data\specialist_db.sanmill_refereed_exact_resume_parity_v2.continuous.sqlite `
  --resumed-database data\specialist_db.sanmill_refereed_exact_resume_parity_v2.segmented.sqlite
```

## Acceptance and stop rules

V2 passes only if all three preflights pass on one clean published source,
all lifecycle ledgers finish `completed`, both routes use only finite periodic
updates, and the existing verifier reports exact equality for model,
optimizer, scheduler, scaler, RNG, normalized trainer/data state, combined
training logs, and semantic SpecialistDB rows.

Any protocol failure, non-finite value, final-flush-only update, checkpoint or
database identity failure, schedule difference, or semantic comparison
difference is `fatal_stop`. Success remains continuation/infrastructure
evidence, not strength evidence or long-run authorization.

