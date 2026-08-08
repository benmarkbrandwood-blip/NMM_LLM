# Sanmill Fixed-Resource Curriculum Smoke v1

## Status and authority

Status: `prepared_unlaunched`

Parent experiment: `dev-v4-sanmill-refereed-fresh-v1`

Parent decision:
[`sanmill-fixed-resource-curriculum-v1.md`](sanmill-fixed-resource-curriculum-v1.md)

The product owner delegated the remaining technical gates through entry into
the retained run on 8 August 2026. This contract authorizes one launch only
after a separate readiness record is committed and the exact command returns
`ready_for_smoke` again from that clean published commit. A failed launch
consumes this authority and requires a new isolated contract; it must not be
retried in place.

## Question and claim boundary

The smoke asks whether the production trainer can route five consecutive
update-capable games through the deterministic Sanmill resource schedule and
record the expected level and node ceiling at every game. It does not measure
or compare opponent strength.

The five games use these one-game stages:

```text
game index:   0      1      2        3        4
level:        1      2      3        4        5
node ceiling: 1,000  5,000  25,000   100,000  500,000
```

`self_play_ratio=0` ensures that every scheduled game invokes the Sanmill
search opponent; Sanmill remains the authoritative referee for both sides.
This differs deliberately from the retained run's 0.60 frozen-target share
and is limited to route coverage. Exact-resume parity for the eventual mixed
route was already proved separately.

## Isolation and initial database identity

Run ID:

```text
sanmill-fixed-resource-curriculum-smoke-v1-20260808-001
```

Output:

```text
out/sanmill-fixed-resource-curriculum-smoke-v1
```

SpecialistDB:

```text
data/specialist_db.sanmill_fixed_resource_curriculum_smoke_v1.sqlite
```

Before readiness, the output did not exist. The database was newly created by
the current `SpecialistDB` implementation and had:

- SHA-256
  `5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d`;
- size 45,056 bytes and `quick_check=ok`;
- `malom_label_version=sector-corrected-v1`;
- zero positions, Malom labels, winning lines, and preferred plays; and
- no training-lineage binding.

This database and output are disposable smoke assets and cannot seed or resume
the retained run.

## Exact command

The read-only readiness command replaces `--launch smoke` with
`--preflight smoke` and omits `--run-id`:

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --launch smoke `
  --run-id sanmill-fixed-resource-curriculum-smoke-v1-20260808-001 `
  --paths-config data\training_paths.local.json `
  --experiment-id dev-v4-sanmill-refereed-fresh-v1 `
  --start-mode fresh `
  --out-dir out\sanmill-fixed-resource-curriculum-smoke-v1 `
  --specialist-db data\specialist_db.sanmill_fixed_resource_curriculum_smoke_v1.sqlite `
  --referee-engine sanmill --opponent-engine sanmill `
  --sanmill-node-ladder 1000,5000,25000,100000,500000 `
  --sanmill-stage-games 1,1,1,1,1 `
  --curriculum-advance-policy fixed-resource `
  --diff-start 1 --diff-max 5 --self-play-ratio 0 --seed 42 `
  --max-games 5 --segment-games 5 --segment-stop-game 5 `
  --max-ply 120 --batch-games 1 --sim-ply-depth 5 `
  --temp-start 0.90 --update-every 8 --log-every 1 `
  --max-branches-per-game 0 --minimal-rollouts --no-recovery `
  --no-sentinel --no-value-net --no-gap-net `
  --no-s1a-warmstart --no-s1b-refresher `
  --no-imitation-mix --no-opening-forcing
```

The ceiling is five primary games, at most 600 logical plies, and one process.
No optional model or historical checkpoint is loaded.

## Acceptance and stop rules

The smoke passes only if:

1. readiness and launch both bind one clean published NMM_LLM commit and the
   pinned Sanmill training runtime;
2. five completed `vs_sanmill` games are logged with difficulty sequence
   `1,2,3,4,5` and node budgets
   `1000,5000,25000,100000,500000`;
3. every authoritative turn passes legal-action and state parity;
4. every optimizer result and checkpoint tensor is finite;
5. the final checkpoint is a valid envelope at game count five;
6. the lifecycle ledger ends `completed`; and
7. the final SpecialistDB remains trusted, internally consistent, and bound
   only to this smoke lineage.

Any protocol or search failure, mirror/rule drift, non-finite value,
unexpected opponent stratum, schedule mismatch, checkpoint failure, or data
identity failure is `fatal_stop`. A pass authorizes preparation of the
managed retained plan, not a strength claim.
