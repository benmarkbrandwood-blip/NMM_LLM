# Sanmill Corrected Exact-Resume Parity v3

## Status and scope

Status: `ready_for_three_staged_smoke_launches`

Experiment ID: `dev-v4-sanmill-corrected-resume-parity-v3`

This current-source parity smoke is required because commit
`4b0420755428d73581108f6e93cd95407b1b72dc` added the persisted
`bootstrap_perspective` field to pending A2C/PPO steps, and commit
`0fbc9510400c88a493b6e2efdcf7c9e92ae8b150` changed frozen-opponent
feature construction. The earlier v2 parity result predates both changes.

The owner has delegated the staged technical launch decision. Each command
may run once only after its own read-only preflight passes from the same clean
published commit. A failure stops the sequence. None of these checkpoints or
databases may seed a retained run.

## Frozen comparison

Compare one uninterrupted two-game process with the same two games split at a
real process boundary. Both routes use:

- seed 42, `batch_games=1`, A2C, and `update_every=8`;
- Sanmill as authoritative referee and search opponent;
- one 1,000-node level with advancement disabled;
- `self_play_ratio=0.60`, `max_ply=120`, and simulation depth 5;
- minimal rollouts and no branches; and
- no recovery, Sentinel, ValueNet, GapNet, warm-start, imitation, refresher,
  or opening forcing.

The two empty databases both have initial SHA-256
`5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d`:

```text
data/specialist_db.sanmill_corrected_exact_resume_parity_v3.continuous.sqlite
data/specialist_db.sanmill_corrected_exact_resume_parity_v3.segmented.sqlite
```

All outputs are isolated below:

```text
out/sanmill-corrected-exact-resume-parity-v3/
```

## Exact commands

For each launch, first replace `--launch smoke --run-id ...` with
`--preflight smoke` and run the remaining command read-only. The preflight
must return `ready_for_smoke`, `errors=[]`, and
`unresolved_decisions=[]`.

### Continuous reference

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --launch smoke `
  --run-id sanmill-corrected-exact-resume-parity-v3-continuous `
  --paths-config data\training_paths.local.json `
  --experiment-id dev-v4-sanmill-corrected-resume-parity-v3 `
  --start-mode fresh `
  --out-dir out\sanmill-corrected-exact-resume-parity-v3\continuous `
  --specialist-db data\specialist_db.sanmill_corrected_exact_resume_parity_v3.continuous.sqlite `
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
  --run-id sanmill-corrected-exact-resume-parity-v3-segment-0001 `
  --paths-config data\training_paths.local.json `
  --experiment-id dev-v4-sanmill-corrected-resume-parity-v3 `
  --start-mode fresh `
  --out-dir out\sanmill-corrected-exact-resume-parity-v3\segment-0001 `
  --specialist-db data\specialist_db.sanmill_corrected_exact_resume_parity_v3.segmented.sqlite `
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
  --run-id sanmill-corrected-exact-resume-parity-v3-segment-0002 `
  --parent-run-id sanmill-corrected-exact-resume-parity-v3-segment-0001 `
  --paths-config data\training_paths.local.json `
  --experiment-id dev-v4-sanmill-corrected-resume-parity-v3 `
  --start-mode exact-resume `
  --resume out\sanmill-corrected-exact-resume-parity-v3\segment-0001\latest.pt `
  --out-dir out\sanmill-corrected-exact-resume-parity-v3\segment-0002 `
  --specialist-db data\specialist_db.sanmill_corrected_exact_resume_parity_v3.segmented.sqlite `
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

### Semantic verifier

```powershell
.\.venv\Scripts\python.exe scripts\verify_resume_parity.py `
  --continuous-checkpoint out\sanmill-corrected-exact-resume-parity-v3\continuous\latest.pt `
  --resumed-checkpoint out\sanmill-corrected-exact-resume-parity-v3\segment-0002\latest.pt `
  --continuous-log out\sanmill-corrected-exact-resume-parity-v3\continuous\train_log.jsonl `
  --resumed-log out\sanmill-corrected-exact-resume-parity-v3\segment-0001\train_log.jsonl `
  --resumed-log out\sanmill-corrected-exact-resume-parity-v3\segment-0002\train_log.jsonl `
  --continuous-database data\specialist_db.sanmill_corrected_exact_resume_parity_v3.continuous.sqlite `
  --resumed-database data\specialist_db.sanmill_corrected_exact_resume_parity_v3.segmented.sqlite
```

## Acceptance

The existing verifier must report exact equality for model and optimiser
state, scheduler/scaler state, all persisted RNG state, normalized trainer and
data state including pending-step bootstrap perspective, combined game logs,
and semantic SpecialistDB tables. Both games must use finite periodic updates;
a final-flush-only comparison is invalid.

Passing remains continuation evidence only. It is not strength evidence and
does not authorize a retained run until automatic segment-boundary policy
health quarantine is also implemented and tested.
