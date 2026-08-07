# Rules-Corrected Successor Smoke Result — 7 August 2026

## Decision

`passed`

The authorized two-game smoke `successor-rules-v2-smoke-001` completed from
scratch on clean, remotely published `dev` commit
`5cb44b1213d991da20d00db571eb9137493d0f89`. It performed one real optimizer
step, preserved a verified version-2 checkpoint, and wrote only to its isolated
output directory and SpecialistDB.

This is training-route and evidence-chain proof only. It is not playing-
strength evidence, does not promote the checkpoint, and does not authorize a
long run or an exact resume.

## Frozen Smoke Contract

| Field | Value |
| --- | --- |
| Experiment | `dev-v4-rules-corrected-successor-v1` |
| Run | `successor-rules-v2-smoke-001` |
| Git | `5cb44b1213d991da20d00db571eb9137493d0f89`; clean; `dev == origin/dev` |
| Start | `fresh`; no checkpoint and no automatic resume |
| Algorithm | A2C; PPO disabled |
| Components | Sentinel, ValueNet, GapNet, S1A warm-start, RL imitation mix, S1B refresher, and opening forcing explicitly disabled |
| Work | one CUDA process, `batch_games=1`, 500,000 heuristic nodes per move, no wall-clock search limit |
| Rollout | two counted games, depth-5 simulation, no branch rollouts, `max_ply=120` |
| Schedule | seed 42; temperature 0.90; 50% frozen / 50% heuristic schedule |
| Rules | `nmm-training-core@2`, semantic digest `sha256:52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a` |
| Protocol evidence | immutable `mif-suite-1.0`, Suite JCS `sha256:81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f` |
| External safety bound | two hours; not reached |

The exact command was:

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --launch smoke `
  --run-id successor-rules-v2-smoke-001 `
  --experiment-id dev-v4-rules-corrected-successor-v1 `
  --start-mode fresh `
  --paths-config data\training_paths.local.json `
  --out-dir learned_ai\checkpoints\smoke\s_gen_v2_successor_readiness_v1 `
  --specialist-db data\specialist_db.successor_readiness_v1.sqlite `
  --ruleset-manifest data\rulesets\nmm-training-core@2.json `
  --no-sentinel --no-value-net --no-gap-net `
  --no-s1a-warmstart --no-imitation-mix --no-s1b-refresher `
  --no-opening-forcing `
  --max-games 2 --segment-games 2 --batch-games 1 `
  --max-ply 120 --max-ply-branch 120 `
  --max-branches-per-game 0 --sim-ply-depth 5 `
  --heuristic-node-budget 500000 `
  --seed 42 --temp-start 0.90 `
  --self-play-ratio 0.50 --update-target-every 50 --log-every 50
```

## Verification Before Launch

The first attempted focused-test invocation encountered Windows pytest
temporary-root setup errors. Eighty-two tests had passed and no assertion had
failed, but that invocation is not acceptance evidence. The identical
selection was repeated with a fresh isolated `--basetemp` and reported:

```text
144 passed
```

The mandatory label-boundary group reported:

```text
103 passed, 498 subtests passed
```

The final exact-command preflight then returned `ready_for_smoke`, zero errors,
no checkpoint, an absent output directory, and SpecialistDB counts 0/0/0.
The database had `malom_label_version=sector-corrected-v1` and no lineage.

## Runtime Result

The process exited with code zero. Its external wall time was about 27 seconds;
the lifecycle interval from `training_started` to `training_completed` was
about 21.74 seconds. The two counted games consisted of one primary rollout
and its confirmation rollout. This is a functional observation, not a
throughput benchmark for a long run.

The primary game was learner Black against the difficulty-1 heuristic. It
ended after 29 logical plies by no legal move. Both counted outcomes were
losses. The result has no strength meaning at this sample size.

The final flush consumed 14 primary learner steps and recorded one real Adam
update:

| Field | Value |
| --- | ---: |
| `update_count` | 1 |
| Batch steps | 14 |
| Policy loss | 0.5545236468 |
| Value loss | 0.4305941463 |
| Entropy | 2.3191523552 |
| Learning rate | 0.0001 |
| Pending steps after save | 0 |

All values are finite. The checkpoint contains optimizer state for 14 parameter
entries. The runtime manifest and console both record imitation mixing as
disabled, and the focused no-imitation regression proves that this control
does not read its dataset.

## Persisted Evidence

The lifecycle ledger contains the ordered, hash-linked states
`preflight_passed`, `running`, and `completed`. Its chain reloads successfully.
The canonical run-manifest identity is
`5ec9b441b01f5031ece5a05d9817127dd12527396a3cbbfc64a8e2b1536c8339`.

| Ignored local artefact | SHA-256 |
| --- | --- |
| `run-manifest.json` | `ee3d276e2fda6eca5954854b504183da827369202e7a7321cabdcb90121542a1` |
| `run-events.jsonl` | `3ae1a542692ece521bb079800a95003e3cc80cbea6f6d672fd3c02a716ea33cb` |
| `train_log.jsonl` | `cfd0fa462c263fdd16b47db48696d39d85c943c02e06ecd82b75ca58dbf15912` |
| `update_log.jsonl` | `f83b786b7784dad6de75daa63705d1732733d81886ce0de57c806a63c8318820` |
| `latest.pt` file | `2fdbc37492ae07cad859d9e95e8ad8bc852f8f34c3cf8150858a738f01ecae40` |
| final SpecialistDB | `dd35645301d8ec2fa82719bf480388f79a3c40dc7d9ca8cf0397d5052caeb19b` |

The checkpoint tool verified payload SHA-256
`ad19e815e5a3ff9c7e410f911f37e8d61254dfd97054fedc7bf03dedb8c171b2`.
Its descriptor records `role=latest`, `save_reason=final`, `game_count=2`,
`update_count=1`, `source_checkpoint=scratch`, and the expected MIF, rules,
Malom, HumanDB, and SpecialistDB identities.

The final SpecialistDB passes `quick_check`, retains
`malom_label_version=sector-corrected-v1`, and binds lineage root
`successor-rules-v2-smoke-001`. It contains 43 positions, 17 Malom labels,
zero winning lines, and zero preferred plays.

## Remaining Gate

The current long-run verdict remains `needs_decision`. Before a long run:

1. run the current complete test suite and report any unavailable historical
   integration assets separately;
2. use a new output directory and a different new empty SpecialistDB rather
   than continuing this smoke;
3. freeze the objective, game and wall-time envelope, segment size,
   `max_ply`, monitoring, and quarantine criteria in a new immutable plan;
4. obtain a separate product authorization.

The two-game elapsed time is too small and structurally unrepresentative to
justify a 12-hour bound for 5,000 games. A larger non-training performance
estimate or a conservative plan envelope is required before authorization.
