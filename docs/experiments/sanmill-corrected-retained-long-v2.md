# Sanmill-Corrected Retained Long Run v2

## Status and authority

Status: `ready_for_local_plan_and_final_preflight`

Experiment ID: `dev-v4-sanmill-corrected-retained-v2`

Plan ID: `managed-sanmill-corrected-retained-v2`

The product owner delegated routine technical decisions through entry into
long training and then requested careful repair followed by retraining. This
document freezes the resulting successor. It authorizes creation of the local
ignored plan and authorization, final read-only readiness checks, and launch
only if every gate below passes. It does not authorize publication, model
promotion, a larger resource envelope, or weakening an isolation boundary.

## Why this is a new lineage

The completed `managed-sanmill-v4-fresh-v1` run is learning-invalid. Its
policy learned the inverse direction on fixed Malom-critical positions. Two
independent implementation defects have since been corrected:

1. an A2C/PPO bootstrap from an opponent-to-move successor was used without
   negating the successor value; and
2. frozen-target opponents selected actions without the learner's lookahead
   and SpecialistDB feature route.

A separate fresh 500-game smoke completed 110 finite updates and passed the
prespecified fixed-state anti-collapse limits. Current-source two-game
continuous-versus-segmented parity then passed with exact update, checkpoint
state and database-table equality. Finally, the managed supervisor gained a
real-checkpoint-tested policy-health gate that runs before every segment is
accepted as an exact-resume parent.

None of those historical models or databases is an input to this experiment.
The first process reconstructs seed-42 scratch weights and starts at game
zero. No `--resume`, `--auto-resume-best`, maintainer checkpoint, old
Generalist, smoke checkpoint, or parity checkpoint is permitted.

## Frozen training configuration

- A2C, learning rate 0.0001, default update interval 64, seed 42, and
  `batch_games=1`;
- Sanmill as complete-history rules authority for every logical turn and as
  the search opponent for the non-frozen stratum;
- the isolated strict runtime pinned to Sanmill commit
  `a6623f88959f7453594df274fbe1f128af7ff55e`, with
  `StrictFailurePolicy=true`,
  `StrictRefereeProfile=mif-stable-moving-v1`, one thread, fixed seed,
  disabled shuffling and no search fallback;
- frozen-target opponents on 60 percent of games, using deterministic argmax
  over the same lookahead and SpecialistDB feature route as the learner;
- Sanmill-search opponents on 40 percent of games;
- fixed node ceilings `1,000,5,000,25,000,100,000,500,000` for global game
  intervals `500,500,500,1,000,2,500`;
- target refresh and checkpoint/log cadence every 50 games;
- temperature 0.90, linearly reaching 0.20 at 80 percent of 5,000 games;
- `sim_ply_depth=5`, `max_ply=120`, minimal primary rollouts, no branches,
  and no wall-clock search limit;
- no score-based advancement, lower-level random blending, recovery,
  resurrection, PPO, Sentinel, ValueNet, GapNet, S1A warm-start, S1B
  refresher, imitation mixing, or trainer-side opening forcing; and
- HumanDB frequencies and outcomes available while its unversioned
  historical Malom fields remain masked.

Node transitions are curriculum exposure, not claims that the learner has
beaten a lower level. W/D/L, rolling score, entropy, loss and graph trends are
diagnostics and never authorize early promotion or an automatic
configuration change.

## Resource envelope

- at most 5,000 completed games;
- at most 12 active supervisor hours, including boundary audits;
- 250 games per process segment; and
- one local machine and one CUDA device.

The controller may advance through the already authorized segments only by
validated exact resume. Wall-time exhaustion stops and requires a new product
decision; it must not silently shorten the experiment or alter the schedule.

## Fresh isolated assets

Control and output directory:

```text
learned_ai/checkpoints/scaffolded/s_gen_v2_sanmill_refereed/
managed-sanmill-corrected-retained-v2
```

SpecialistDB:

```text
data/specialist_db.sanmill_corrected_retained_v2.sqlite
```

Before plan generation, the output directory must be absent. The database
must be a newly created, empty, unbound current-schema SpecialistDB with
`malom_label_version=sector-corrected-v1`, `quick_check=ok`, zero positions,
zero winning lines, zero preferred plays, and no non-empty WAL. Its initial
file hash and metadata are readiness evidence. No smoke, parity, failed-long,
maintainer, active corrected, or legacy database may be copied into it.

## Mandatory segment-boundary policy gate

The immutable managed plan must enable the committed fixed-state audit after
every 250-game trainer process, before the segment is recorded as completed.
It binds the exact checkpoint, live closed SpecialistDB, source commit,
experiment, game count, paths configuration, audit script and fixed corpus
whose SHA-256 is:

```text
cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e
```

Each segment must satisfy:

- exactly 29 Malom-critical positions in both direct and policy summaries;
- direct lookahead argmax value-preserving rate exactly 1.0;
- candidate critical argmax value-preserving rate at least 0.50; and
- candidate mean best-preserving minus best-downgrading logit at least
  -0.10.

Audit execution failure, missing or malformed output, identity drift,
critical-state drift, a non-finite metric or threshold failure quarantines
the segment. It is not a completed parent, automatic continuation stops, and
the Agent must diagnose the lineage. These limits detect catastrophic
direction failure; passing them is not strength evidence.

## Managed plan command

Run only from the clean published commit containing this document and the
gate readiness evidence:

```powershell
.\.venv\Scripts\python.exe scripts\manage_generalist_run.py prepare `
  --control-dir learned_ai\checkpoints\scaffolded\s_gen_v2_sanmill_refereed\managed-sanmill-corrected-retained-v2 `
  --plan-id managed-sanmill-corrected-retained-v2 `
  --max-wall-hours 12 `
  --objective "fresh retained Sanmill-refereed baseline after corrected value bootstrap and feature alignment" `
  --experiment-id dev-v4-sanmill-corrected-retained-v2 `
  --max-games 5000 --segment-games 250 --max-ply 120 `
  --engine-profile sanmill-fixed-resource --self-play-ratio 0.60 `
  --sanmill-node-ladder 1000,5000,25000,100000,500000 `
  --sanmill-stage-games 500,500,500,1000,2500 `
  --specialist-db data\specialist_db.sanmill_corrected_retained_v2.sqlite `
  --policy-health-gate --policy-health-device auto
```

The separately published authorization must bind the exact plan SHA-256,
name `product-owner` as authorizer, repeat the resource and isolation
boundary, and record the owner's delegated technical launch authority.

## Final launch gates

Before starting `run-authorized`:

1. `HEAD == origin/dev == plan.git_commit`, with a clean tracked worktree and
   no unreviewed `origin/main` change selected for this experiment;
2. plan, authorization, path-config, audit-script and corpus hashes match;
3. the output contains only the newly published plan, authorization and
   controller ledger, and the fresh SpecialistDB retains its recorded empty
   identity;
4. the mandatory Malom/provenance suite and all focused manager, trainer,
   checkpoint, exact-resume, Sanmill-referee and policy-health tests pass;
5. the exact first-segment trainer command returns
   `ready_for_long_run`, `errors=[]` and `unresolved_decisions=[]` in
   read-only preflight mode;
6. the resolved resume configuration equals the managed plan pin;
7. the MIF Suite, training ruleset, HumanDB, corrected Malom manifest and
   isolated Sanmill runtime match their frozen identities; and
8. no trainer, supervisor, stale lock or competing process owns the output,
   database, runtime or CUDA resource.

Any mismatch stops before training. No timeout, default, checkpoint discovery
or fallback may substitute for a failed gate.

## Claim boundary

Successful launch means only that a fresh, bounded, rules-corrected retained
training lineage has started. Segment completion is learning-health and
infrastructure evidence. Completion of all 5,000 games is still not a strength
or promotion result. The retained candidate requires a later separately
frozen held-out evaluation before any playing-strength claim.
