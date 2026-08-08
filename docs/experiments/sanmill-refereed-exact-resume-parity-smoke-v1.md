# Sanmill-Refereed Exact-Resume Parity Smoke v1

## Status and authority

Status: `prepared_unlaunched`

Experiment: `dev-v4-sanmill-refereed-fresh-v1`

This contract defines one bounded comparison between an uninterrupted
two-game Sanmill-refereed training run and the same two-game schedule split
across a fresh process and an exact-resume process. It is a technical
continuation test, not a strength evaluation, node-ladder decision, retained
training plan, or authority to reuse any resulting checkpoint.

The delegated technical authority applies only after this document is
published, both input databases are proven empty and isolated, the continuous
and first-segment exact-command preflights pass from one clean published source
commit, and a staged readiness record is separately published. The continuous
and first-segment invocations may then be launched once. The exact-resume
preflight necessarily waits for the first segment to create its checkpoint;
it must run from the same published source commit, without an intervening Git
change, and must pass before the second segment may be launched once. Any
failure consumes the affected invocation and stops the comparison; there is
no automatic retry.

## Question under test

After real finite A2C updates, does a process boundary followed by
`exact-resume` produce exactly the same future training state and semantic
evidence as an uninterrupted run?

The comparison uses the repository's existing exact parity verifier. It
requires equality of:

- model and Adam optimiser state;
- scheduler and scaler state;
- Python, NumPy, PyTorch, CUDA, and component random-number state;
- normalized trainer state, including counters, histories, the frozen target,
  and any pending rollout steps;
- normalized data state;
- the combined per-game training log; and
- the semantic rows of all four SpecialistDB tables.

Only path- or segmentation-specific identities already excluded by the
verifier may differ. Checkpoint payload bytes and run manifests are not
expected to match because their run, path, parent, and checkpoint-chain
identities intentionally differ.

## Frozen training semantics

Both routes use the same published NMM_LLM source, pinned Sanmill runtime,
ruleset, data assets, and trajectory-affecting arguments:

- fresh random model with seed 42;
- CUDA when selected by the existing trainer runtime;
- A2C, `batch_games=1`, and `update_every=8`;
- two total games with `max_ply=120` and simulation depth 5;
- Sanmill as authoritative referee for every complete logical turn;
- one 1,000-node Sanmill opponent level with curriculum advancement disabled;
- frozen-target ratio 0.60, which deterministically schedules one Sanmill
  opponent and one frozen-target game at seed 42;
- minimal rollouts and no branches;
- temperature start 0.90 with the existing two-game global schedule; and
- Sentinel, ValueNet, GapNet, recovery, S1A warm-start, S1B refresher,
  imitation mixing, and trainer-side opening forcing disabled.

`update_every=8` is deliberate. A primary game has at least eight learner
steps in the selected deterministic schedule, so each game reaches the normal
periodic update before a process can end. The first split segment therefore
does not use the final-flush path to perform an update earlier than the
uninterrupted route.

The two routes use separate empty SpecialistDB files, each created by the
current `SpecialistDB` implementation and stamped
`malom_label_version=sector-corrected-v1`. The segmented route alone reuses
its database between its first and second processes, as required by the
checkpoint's mutable-asset identity.

## Resource ceiling

The comparison permits exactly:

- four completed primary games across both routes;
- at most 480 logical plies in aggregate;
- two Sanmill-search opponent games at a 1,000-node per-search ceiling;
- two frozen-target games, still refereed by Sanmill;
- no branch, confirmation, or retry rollout; and
- no retained-run continuation after the parity verdict.

## Frozen commands

All commands run from the repository root. A read-only preflight replaces
`--launch smoke` with `--preflight smoke` and omits `--run-id`. The staged
readiness record covers the two fresh commands directly and makes the
exact-resume invocation conditional on its later exact-command preflight.
There must be no source commit between the first and second segmented
processes because the checkpoint's experiment identity binds the Git commit.

### Uninterrupted reference

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --launch smoke `
  --run-id sanmill-refereed-exact-resume-parity-v1-continuous `
  --paths-config data\training_paths.local.json `
  --experiment-id dev-v4-sanmill-refereed-fresh-v1 `
  --start-mode fresh `
  --out-dir out\sanmill-refereed-exact-resume-parity-v1\continuous `
  --specialist-db data\specialist_db.sanmill_refereed_exact_resume_parity_v1.continuous.sqlite `
  --referee-engine sanmill --opponent-engine sanmill `
  --sanmill-node-ladder 1000 `
  --curriculum-advance-policy disabled `
  --diff-start 1 --diff-max 1 `
  --self-play-ratio 0.60 --seed 42 `
  --max-games 2 --segment-games 2 --segment-stop-game 2 `
  --max-ply 120 --batch-games 1 --sim-ply-depth 5 `
  --temp-start 0.90 --update-every 8 --log-every 1 `
  --max-branches-per-game 0 --minimal-rollouts --no-recovery `
  --no-sentinel --no-value-net --no-gap-net `
  --no-s1a-warmstart --no-s1b-refresher `
  --no-imitation-mix --no-opening-forcing
```

### Segmented first process

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --launch smoke `
  --run-id sanmill-refereed-exact-resume-parity-v1-segment-0001 `
  --paths-config data\training_paths.local.json `
  --experiment-id dev-v4-sanmill-refereed-fresh-v1 `
  --start-mode fresh `
  --out-dir out\sanmill-refereed-exact-resume-parity-v1\segment-0001 `
  --specialist-db data\specialist_db.sanmill_refereed_exact_resume_parity_v1.segmented.sqlite `
  --referee-engine sanmill --opponent-engine sanmill `
  --sanmill-node-ladder 1000 `
  --curriculum-advance-policy disabled `
  --diff-start 1 --diff-max 1 `
  --self-play-ratio 0.60 --seed 42 `
  --max-games 2 --segment-games 1 --segment-stop-game 1 `
  --max-ply 120 --batch-games 1 --sim-ply-depth 5 `
  --temp-start 0.90 --update-every 8 --log-every 1 `
  --max-branches-per-game 0 --minimal-rollouts --no-recovery `
  --no-sentinel --no-value-net --no-gap-net `
  --no-s1a-warmstart --no-s1b-refresher `
  --no-imitation-mix --no-opening-forcing
```

### Segmented exact-resume process

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --launch smoke `
  --run-id sanmill-refereed-exact-resume-parity-v1-segment-0002 `
  --parent-run-id sanmill-refereed-exact-resume-parity-v1-segment-0001 `
  --paths-config data\training_paths.local.json `
  --experiment-id dev-v4-sanmill-refereed-fresh-v1 `
  --start-mode exact-resume `
  --resume out\sanmill-refereed-exact-resume-parity-v1\segment-0001\latest.pt `
  --out-dir out\sanmill-refereed-exact-resume-parity-v1\segment-0002 `
  --specialist-db data\specialist_db.sanmill_refereed_exact_resume_parity_v1.segmented.sqlite `
  --referee-engine sanmill --opponent-engine sanmill `
  --sanmill-node-ladder 1000 `
  --curriculum-advance-policy disabled `
  --diff-start 1 --diff-max 1 `
  --self-play-ratio 0.60 --seed 42 `
  --max-games 2 --segment-games 1 --segment-stop-game 2 `
  --max-ply 120 --batch-games 1 --sim-ply-depth 5 `
  --temp-start 0.90 --update-every 8 --log-every 1 `
  --max-branches-per-game 0 --minimal-rollouts --no-recovery `
  --no-sentinel --no-value-net --no-gap-net `
  --no-s1a-warmstart --no-s1b-refresher `
  --no-imitation-mix --no-opening-forcing
```

### Semantic comparison

```powershell
.\.venv\Scripts\python.exe scripts\verify_resume_parity.py `
  --continuous-checkpoint out\sanmill-refereed-exact-resume-parity-v1\continuous\latest.pt `
  --resumed-checkpoint out\sanmill-refereed-exact-resume-parity-v1\segment-0002\latest.pt `
  --continuous-log out\sanmill-refereed-exact-resume-parity-v1\continuous\train_log.jsonl `
  --resumed-log out\sanmill-refereed-exact-resume-parity-v1\segment-0001\train_log.jsonl `
  --resumed-log out\sanmill-refereed-exact-resume-parity-v1\segment-0002\train_log.jsonl `
  --continuous-database data\specialist_db.sanmill_refereed_exact_resume_parity_v1.continuous.sqlite `
  --resumed-database data\specialist_db.sanmill_refereed_exact_resume_parity_v1.segmented.sqlite
```

## Acceptance and stop rules

The smoke passes only if:

1. all three final preflights pass from the same clean published commit;
2. all three lifecycle ledgers finish in `completed` without quarantine;
3. both routes complete games 0 and 1 with the same schedule and terminal
   records;
4. each route performs finite periodic A2C updates and never relies on an
   update caused only by a process-final flush;
5. the exact semantic comparison returns `status=passed`; and
6. all final checkpoint envelopes and both SpecialistDB files pass their
   integrity checks.

Any protocol error, state mismatch, non-finite update, checkpoint failure,
database identity change, unexpected final-flush update, or parity difference
is `fatal_stop`. The resulting artefacts are quarantined evidence and cannot
be retried or used as a long-run source without a new diagnosis and contract.
