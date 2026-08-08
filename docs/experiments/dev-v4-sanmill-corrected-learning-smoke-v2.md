# Sanmill-Refereed Corrected Learning Smoke v2

## Status and authority

Status: `ready_for_local_plan_and_read_only_preflight`

Experiment ID: `dev-v4-sanmill-corrected-learning-smoke-v2`

This is one fresh 500-game update smoke. The owner has delegated routine
technical launch decisions through long-run readiness, but every evidence and
integrity gate in this document remains mandatory. The smoke authorizes one
`run-next` segment only. It does not authorize a second segment, a 5,000-game
continuation, promotion, or publication.

The plan retains `max_games=5000` solely so the temperature and fixed-resource
curriculum are evaluated at the same global progress as the proposed retained
run. Setting `max_games=500` would prematurely cool from 0.90 to 0.20 and
would not test the intended training regime.

## Purpose

The completed `managed-sanmill-v4-fresh-v1` run is infrastructure-complete but
learning-invalid. Its fixed-state report is recorded in the
[root-cause evidence](../evidence/generalist-policy-health-root-cause-2026-08-08.md).
This smoke asks a narrower question: after correcting opponent-perspective
value bootstrapping and frozen-opponent feature construction, can a fresh
policy complete many real updates without immediately learning the same
inverse-Malom direction?

This is not a strength test. Fresh policies are expected to lose heavily to
Sanmill and no W/D/L or rolling-score threshold is used for acceptance.

## Immutable lineage and data boundary

The first process starts from seed-42 random weights. It must not load
`--resume`, `--auto-resume-best`, the failed 5,000-game checkpoint, a
maintainer checkpoint, or any historical model.

The dedicated SpecialistDB is:

```text
data/specialist_db.sanmill_corrected_learning_smoke_v2.sqlite
```

Its pre-plan state is empty, unbound, `quick_check=ok`, and declares
`malom_label_version=sector-corrected-v1`. Its initial file SHA-256 is:

```text
5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d
```

The output/control directory must not exist before plan generation:

```text
out/managed-sanmill-corrected-learning-smoke-v2
```

The imported HumanDB contributes frequencies and outcomes only. Its
historical unversioned Malom columns remain disabled. Malom queries use the
`sector-corrected-v1` dataset and manifest.

## Frozen training route

- A2C, learning rate 0.0001, update interval 64 steps, seed 42, and
  `batch_games=1`;
- Sanmill as authoritative referee for every logical turn and as the search
  opponent for the non-self-play stratum;
- frozen-target opponents on 60% of games, with deterministic argmax over the
  same lookahead and SpecialistDB feature route as the learner;
- five Sanmill node levels of 1,000, 5,000, 25,000, 100,000, and 500,000;
- fixed global stage durations of 500, 500, 500, 1,000, and 2,500 games;
- one process segment ending at game 500, so this smoke exercises level 1 and
  stops exactly at the first curriculum boundary;
- `max_ply=120`, simulation depth 5, and `minimal_rollouts`;
- temperature 0.90 on the 5,000-game schedule, reaching approximately 0.8125
  at this smoke boundary;
- no branches, recovery, Sentinel, ValueNet, GapNet, S1A warm-start,
  imitation mixing, S1B refresher, or opening forcing; and
- no local GameAI opponent, local rules authority, wall-clock search budget,
  database move substitution, or random failure fallback.

## Managed plan preparation

After this document is committed and `dev == origin/dev` with a clean tracked
worktree, prepare the local ignored plan with:

```powershell
.\.venv\Scripts\python.exe scripts\manage_generalist_run.py prepare `
  --control-dir out\managed-sanmill-corrected-learning-smoke-v2 `
  --plan-id managed-sanmill-corrected-learning-smoke-v2 `
  --max-wall-hours 1 `
  --objective "fresh corrected-learning smoke after value and feature-route fixes" `
  --experiment-id dev-v4-sanmill-corrected-learning-smoke-v2 `
  --max-games 5000 `
  --segment-games 500 `
  --max-ply 120 `
  --engine-profile sanmill-fixed-resource `
  --self-play-ratio 0.60 `
  --sanmill-node-ladder 1000,5000,25000,100000,500000 `
  --sanmill-stage-games 500,500,500,1000,2500 `
  --specialist-db data\specialist_db.sanmill_corrected_learning_smoke_v2.sqlite
```

The generated authorization must bind the exact plan SHA-256 and state that
only one 500-game `run-next` is authorized. Use `run-next`, never
`run-authorized`, for this smoke.

## Pre-launch gates

Before the one launch:

1. `HEAD == origin/dev == plan.git_commit`, and the tracked worktree is clean.
2. The output/control directory was absent before plan generation and contains
   only the newly generated plan and authorization before launch.
3. The SpecialistDB still has its initial identity, no rows, no lineage, and
   no non-empty WAL.
4. The configured HumanDB, corrected Malom manifest, ruleset, MIF Suite, and
   isolated Sanmill runtime match their frozen identities.
5. The mandatory 103 Malom/teacher/provenance tests and focused trainer,
   checkpoint, exact-resume, frozen-opponent, and policy-health tests pass.
6. The exact first-segment `--preflight long-run` generated by the manager
   returns `ready_for_long_run`, `errors=[]`, and
   `unresolved_decisions=[]`.
7. No training process or supervisor owns the target output or database.

Any mismatch consumes the attempted launch and leaves the assets quarantined.
Do not delete or recycle them into a retry.

## Post-smoke health gate

After game 500, close the trainer and run the committed health audit against
`segment-0001/latest.pt`, the dedicated SpecialistDB, and the fixed corpus with
SHA-256:

```text
cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e
```

The smoke passes the learning-health gate only if all of the following hold:

- exactly 500 completed games and at least one optimiser update;
- no non-finite loss, logit, reward, checkpoint, database, referee, CUDA, or
  identity error;
- exactly 29 Malom-critical fixed positions;
- direct lookahead argmax value-preserving rate remains 1.0;
- candidate critical argmax value-preserving rate is at least 0.50; and
- candidate mean best-preserving minus best-downgrading logit is at least
  -0.10.

The last two limits are anti-collapse quarantine boundaries, not evidence that
the model is strong or better than scratch. The old failed final policy scored
0.069 and -0.730 respectively; the seed-42 scratch reference scored 0.690 and
approximately zero.

## Consequence

If the smoke passes, record its raw logs, checkpoint and database identities,
fixed-state report, and exact result in a separate evidence commit. Then rerun
process-boundary exact-resume equivalence on the corrected source and add an
automatic segment-boundary health quarantine to the retained supervisor.
Only those later results can produce `ready_for_long_run` for a new 5,000-game
lineage. The smoke itself must never be promoted into that lineage.
