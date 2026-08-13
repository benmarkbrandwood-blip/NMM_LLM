# Sanmill no-refresh retained long v4

Status: `designed_unlaunched_attempt_003`

Experiment ID: `dev-v4-sanmill-no-refresh-retained-v4-seed70-attempt-003`

Plan ID: `managed-sanmill-no-refresh-retained-v4-seed70-attempt-003`

This document freezes a fresh 5,000-game research baseline that suppresses
all periodic frozen-target refreshes. It does not authorize launch, held-out
evaluation, promotion, or publication.

## Preparation attempts

Preparation attempt 001 is the preserved machine-local directory
`managed-sanmill-no-refresh-retained-v4-seed70`. It froze plan identity
`2421fa96ab471af19b4134a1782953770e729ea40df894b62e2b345af9211a25`
at source `f1a8974a`, but it never received an authorization, never created a
segment directory, and completed zero games. Its readiness JSON was not
produced by version-controlled code and did not preserve the raw preflight
report or command artifact. It is therefore marked
`invalidated_unlaunched_never_authorize` in its ignored disposition record.
Its existing plan, ledger, readiness narrative, database and disposition must
remain byte-for-byte historical evidence; they must not be authorized,
launched, resumed, overwritten, relabelled, or reused.

Attempt 002 used the same unused seed 70 with new experiment, plan, control and
database identities. It was authorized once under plan identity `2a59a93f`
and readiness identity `a6cd2cd1`, then failed closed in segment 0001 before an
accepted checkpoint or segment. A legacy full-queue A2C batch was cleared
before its behaviour-temperature evidence was calculated, causing `min()` to
receive an empty iterable. Its database contains partial first-rollout writes,
its authorization is consumed, and the entire namespace must remain preserved
and must never be retried, resumed, overwritten or reused. See the
[attempt-002 failure evidence](../evidence/sanmill-no-refresh-retained-v4-attempt-002-failure-2026-08-13.md).

Any successor must use a tested correction plus new experiment, plan, control,
database, readiness and authorization identities. Reusing seed 70 for a fresh
successor does not resume attempt 002 because no checkpoint was retained, but
it still requires a new lineage and a new product decision.

Attempt 003 is that fresh successor. Commit `cde6a5e` snapshots the legacy
non-exact transition batch before clearing the pending queue and adds a focused
regression that preserves all batch temperatures after the original queue is
cleared. Attempt 003 retains the scientific configuration and seed 70 but uses
new experiment, plan, control, template, writable database, readiness and
authorization identities. It must be prepared only from the final clean
published source; this document still grants no launch authority.

## Objective and evidence boundary

The completed seed-58 preserving-retained v3 baseline refreshed its frozen
target every 50 games and ended with 199 draws in its final 200 training
games. Subsequent paired development evidence found a material harmful effect
from the first game-50 refresh. The independent mature-fork cohorts then
pointed in opposite directions and pooled to only `+0.0225694` for
`refresh-mature minus stale-control`, below the frozen `1/12` gate. The
replication result is therefore `no_replicated_material_effect` with no
selected cadence.

This v4 plan does not relabel that null selection. It makes a new research
choice: test the simplest schedule that avoids the demonstrated harmful early
intervention by keeping the initial frozen target for the whole run. The
trainer setting is `--target-refresh-every 5001`, strictly beyond the
5,000-game schedule. It is a permanent no-refresh hypothesis test, not a claim
that stale targets are generally beneficial.

The comparison baseline is the completed seed-58 v3 lineage. Seed 70 is new,
so any between-run difference is descriptive and seed-confounded; it is not a
causal cadence estimate. It is also source-confounded. Retained v3 ran at
source `3f400135`, before the current explicit SpecialistDB read-mode and
learning-rate-mode interfaces and before later manager, preflight and trainer
hardening. Focused tests show that `full` and
`adaptive-search-opponent-win-rate` preserve their intended historical paths,
and current-source stale-control arms exercise the new route, but there is no
byte-for-byte trajectory-parity proof against the v3 executable.

Attempt 002 permits no scientific conclusion about no-refresh training because
it produced no accepted checkpoint or segment. For a separately prepared
successor, the allowed conclusion remains limited to the health and outcome of
one fresh no-refresh baseline under the attempt-003 source, plus later
held-out performance of its frozen candidate if separately authorized. A
difference from v3 cannot be attributed solely to target refresh, seed, or any
other single factor. Training W/D/L, the 29-state policy-health gate,
repetition, max-ply truncation, and late-window curves remain development
diagnostics. Any candidate requires a separately frozen held-out evaluation
before a strength or promotion claim.

## Frozen lineage

- fresh random model and optimizer state;
- fresh seed `70`, not used by an earlier tracked training experiment;
- no imported or historical checkpoint;
- `--start-mode fresh` for segment 0001;
- no `--resume` or `--auto-resume-best` for segment 0001;
- one isolated, initially empty SpecialistDB;
- managed exact-resume only between accepted 250-game segments;
- no observation-based checkpoint resurrection or automatic retry; and
- no checkpoint, database, or output from retained v2, retained v3, the target
  diagnostics, maintainer `main`, or another experiment may seed this lineage.

The run is bound to the final clean published `dev` commit used to generate
its managed plan. Any later tracked commit makes the preparation stale and
requires a new, still-unlaunched preparation; it does not authorize rebasing
or editing the plan in place.

## Frozen training configuration

- A2C; PPO disabled;
- learning rate `0.0001` with the historical
  `adaptive-search-opponent-win-rate` rule;
- temperature `0.90` to `0.20` on the 5,000-game global schedule;
- single-game batches and the trainer's 64-transition A2C update cadence;
- `malom-preserving-only` mill shaping;
- no generic Malom downgrade penalty;
- Malom policy auxiliary coefficient `0.0` in fixed mode;
- SpecialistDB read mode `full`, matching retained v3;
- 60% games against the single initial frozen target and 40% against Sanmill;
- target refresh interval `5001`, so no refresh can occur in games 1--5,000;
- Sanmill fixed-node ladder `1,000, 5,000, 25,000, 100,000, 500,000`;
- stage durations `500, 500, 500, 1,000, 2,500` games;
- fixed-resource curriculum, difficulty 1 through 5;
- strict Sanmill referee and opponent, one thread and shuffling disabled;
- `max_ply=120`, recorded as truncation rather than a rules draw;
- lookahead simulation depth 5 and minimal rollouts;
- no Sentinel, ValueNet, GapNet, imitation warm-start, imitation mixing,
  refresher, opening forcing, branches, or recovery; and
- log and checkpoint cadence 50 games, with one managed process segment every
  250 completed games.

`target-refresh-every=5001` is the only intended learning-schedule change in
the frozen configuration relative to retained v3. It is not the only source
code difference. The explicit read-mode and learning-rate-mode arguments
freeze current names for historical behavior; they do not establish binary or
trajectory parity with the older source.

## Rules, data, and runtime identities

The plan must bind:

- MIF Suite `mif-suite-1.0`, release commit
  `a0a0f21cff5d6fbde045cd1482e416b92e0dc45a`, suite JCS SHA-256
  `81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f`,
  and release-manifest SHA-256
  `dde89416bf5251cdc445ebdb9b92a899f58ec3930d1d8077ae26f1cb1a084499`;
- training ruleset `data/rulesets/nmm-training-core@2.json`, semantic digest
  `52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a`
  and document digest
  `1dfdf5777f36866a53a942c1addd21857d3b72eede8ea2bf4fe1beedfbe878f2`;
- corrected Malom manifest identity
  `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747`;
- HumanDB identity
  `8662e3331210893495aef38c0cb774bd387e508ac8b859261a78b43b74184d31`;
- Sanmill runtime commit
  `a6623f88959f7453594df274fbe1f128af7ff55e`, tree
  `17b9b0fd51ee8dac54c0454a6935978a47d19e0c`, binary SHA-256
  `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`,
  and strict-referee semantic digest
  `1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94`;
  and
- seed-70 operational Sanmill identity
  `bd7016e0b53ba2e93e042a931cffb1ffcdf6e7b70108fd355ff2962d11d78796`.

HumanDB contributes empirical frequencies and outcomes only. Its historical
unversioned Malom columns remain masked. Required identity drift is a stop;
it is not permission to rebuild or substitute an input.

## Isolated outputs and database

Control and output directory:

```text
learned_ai/checkpoints/scaffolded/s_gen_v2_sanmill_refereed/
managed-sanmill-no-refresh-retained-v4-seed70-attempt-003
```

SpecialistDB:

```text
data/specialist_db.sanmill_no_refresh_retained_v4.seed70.attempt_003.sqlite
```

Both targets must be absent before preparation. Copy the database once from
the closed snapshot
`data/specialist_db.no_refresh_retained_v4.attempt_003.template.sqlite`.
The snapshot is a byte-for-byte copy of the closed attempt-002 template, which
was copied from the already closed attempt-001 runtime database. It was not
opened through a writable SQLite connection and has no
WAL, SHM or rollback-journal sidecar. The snapshot and fresh copy must be
45,056 bytes with SHA-256
`5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d`,
`malom_label_version=sector-corrected-v1`, `quick_check=ok`, zero positions,
zero winning lines, zero preferred plays, and no WAL, SHM, or rollback-journal
sidecar.

The older
`data/specialist_db.mill_bonus_ablation_v1.template.sqlite` currently has an
empty WAL and a non-empty SHM sidecar. It is preserved untouched as historical
local state and is not an attempt-003 input. Its sidecars must not be deleted
to make a readiness check pass.

These machine-local artefacts remain ignored. They must not be committed,
overwritten, shared with another lineage, or replaced after plan generation.

## Resource and monitoring envelope

- exactly one fresh lineage;
- at most 5,000 completed games;
- at most 12 active supervisor hours;
- 250 completed games per process segment;
- one local machine and one CUDA device; and
- no held-out games inside the training authorization.

The fixed 29-state policy-health audit runs before every completed segment is
accepted as a resume parent. It requires exactly 29 direct and candidate
states, direct value preservation `1.0`, candidate preservation at least
`0.50`, and candidate preserving-minus-downgrading mean logit margin at least
`-0.10`. The corpus is
`docs/experiments/dev-v4-phase-covered-corpus-v1.json`, SHA-256
`cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e`.

The controller records complete raw train and update logs, segment boundaries,
checkpoint and database identities, target age, opponent and colour strata,
termination reasons, max-ply truncations, repetition, entropy, chosen
probability, Malom preservation, learning rate, temperature, and requested
Sanmill nodes. Rolling curves never replace raw counts.

## Stop and prohibited conditions

Stop fail closed on any non-finite value, failed policy-health gate, target
refresh before completion, schedule drift, rules/data/runtime identity drift,
database or sidecar mismatch, broken event chain, malformed checkpoint,
unsafe exact-resume state, Sanmill failure, CUDA failure, competing owner, or
resource exhaustion.

There is no automatic retry, recovery, extension, held-out evaluation, model
promotion, model publication, release, or Git history rewrite. A failed or
interrupted segment is diagnosed autonomously but remains stopped unless a
separately proven and explicitly covered semantics-identical recovery applies.

## Current-source execution evidence

No additional counted smoke is required merely to exercise the no-refresh
switch. The attempt-002 failure was an evidence-buffer alias, and the focused
regression now proves that clearing the legacy pending queue cannot erase the
snapshot used for behaviour-temperature logging. At the same underlying
trainer route used by this successor, replication
attempt 002 completed three `stale-control` arms with no subsequent target
refresh. Each consumed exactly 8,192 transitions in 128 finite A2C updates and
passed the identical policy-health gate. The paired plan also proved that the
temperature and learning-rate exposure matched its refresh arm byte-for-byte.

The completed retained-v3 run separately proves the fresh start, 20 managed
segment boundaries, strict Sanmill curriculum, database lineage, and
policy-health route over the full 5,000-game schedule. The v4 preflight and
focused tests must still bind the final source and exact new paths. These two
evidence sources are complementary; neither is relabelled as held-out strength
or as a multi-seed long-run effect.

## Managed-plan preparation

Run only from the final clean published source commit, after verifying that
both isolated targets are absent and creating the exact empty database:

```powershell
.\.venv\Scripts\python.exe scripts\manage_generalist_run.py prepare `
  --control-dir learned_ai\checkpoints\scaffolded\s_gen_v2_sanmill_refereed\managed-sanmill-no-refresh-retained-v4-seed70-attempt-003 `
  --plan-id managed-sanmill-no-refresh-retained-v4-seed70-attempt-003 `
  --max-wall-hours 12 `
  --objective "fresh no-refresh retained research baseline after the pooled mature-refresh null result" `
  --experiment-id dev-v4-sanmill-no-refresh-retained-v4-seed70-attempt-003 `
  --seed 70 `
  --max-games 5000 --segment-games 250 --max-ply 120 `
  --engine-profile sanmill-fixed-resource --self-play-ratio 0.60 `
  --sanmill-node-ladder 1000,5000,25000,100000,500000 `
  --sanmill-stage-games 500,500,500,1000,2500 `
  --target-refresh-every 5001 `
  --lr-adaptation-mode adaptive-search-opponent-win-rate `
  --mill-bonus-mode malom-preserving-only `
  --malom-policy-aux-mode fixed --malom-policy-aux-coef 0.0 `
  --specialist-read-mode full `
  --specialist-db data\specialist_db.sanmill_no_refresh_retained_v4.seed70.attempt_003.sqlite `
  --policy-health-gate --policy-health-device auto
```

The generated plan and any later authorization remain ignored. The plan must
freeze the exact source commit, objective, fresh lineage, 5,000-game and
12-hour ceilings, isolated paths, component switches, target interval, and
policy-health gate. It must keep publication and promotion disabled.

## Final technical readiness gate and evidence bundle

Before reporting that technical gates passed with verdict `needs_decision`:

1. fetch and prove `HEAD == origin/dev == plan.git_commit`, active branch
   `dev`, and a clean tracked worktree;
2. confirm reviewed `origin/main` remains
   `40da3ddfced972c418541665ec739b3752edcd1f` or stop for a new read-only
   source review;
3. verify the plan, local path registry, this document, empty database,
   policy corpus, audit script, ruleset, Malom, HumanDB, MIF release, and seed-70
   Sanmill identities;
4. verify the first segment is fresh, its output directory is absent, the
   database is still empty and closed, and no resume or auto-resume flag can be
   selected;
5. run the mandatory Malom/DB/provenance tests plus focused manager, preflight,
   trainer, checkpoint, exact-resume, Sanmill, reward-mode, target-refresh, and
   policy-health tests;
6. run the exact first-segment preflight and require no technical errors and
   no unresolved decision other than the separate product authorization; and
7. prove that no trainer, supervisor, stale lock, or competing process owns
   the output, database, runtime, or CUDA resource.

The version-controlled generic readiness generator constructs the real first
segment from `plan.json`; no command is copied from prose. It persists the
command array, raw preflight JSON and raw-report SHA-256, plan, document,
database and dependency identities, and a canonical readiness identity:

```powershell
.\.venv\Scripts\python.exe scripts\generate_managed_generalist_readiness.py generate `
  --plan learned_ai\checkpoints\scaffolded\s_gen_v2_sanmill_refereed\managed-sanmill-no-refresh-retained-v4-seed70-attempt-003\plan.json `
  --experiment-document docs\experiments\sanmill-no-refresh-retained-long-v4.md `
  --reviewed-main 40da3ddfced972c418541665ec739b3752edcd1f
```

The ignored bundle can be independently recalculated and checked without
launching training:

```powershell
.\.venv\Scripts\python.exe scripts\generate_managed_generalist_readiness.py verify `
  --readiness learned_ai\checkpoints\scaffolded\s_gen_v2_sanmill_refereed\managed-sanmill-no-refresh-retained-v4-seed70-attempt-003\technical-readiness.json
```

Technical readiness and launch authority are reported separately. This
document permits plan generation and read-only preflight only. A later direct
long-run request must bind the exact generated plan and readiness identity and
may create its one ordinary plan authorization without introducing
per-segment approval prompts.

## Reviewed launch route

After the exact plan is technically ready and separately authorized, the only
reviewed supervisor route is:

```powershell
.\.venv\Scripts\python.exe scripts\manage_generalist_run.py run-authorized `
  --plan learned_ai\checkpoints\scaffolded\s_gen_v2_sanmill_refereed\managed-sanmill-no-refresh-retained-v4-seed70-attempt-003\plan.json `
  --authorization learned_ai\checkpoints\scaffolded\s_gen_v2_sanmill_refereed\managed-sanmill-no-refresh-retained-v4-seed70-attempt-003\authorization.json
```

Do not launch this command from the document alone.
