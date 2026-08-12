# Mature target-refresh fork replication v1, attempt 002

Status: `designed_unlaunched_needs_publication`

Plan identity:
`85ad0b99093bc7e81ac6057b92abd8a38cafdc03893b6b46c760dffd3fa5acca`

The authoritative machine-readable contract is
[`sanmill-target-refresh-mature-fork-replication-v1-attempt-002.json`](sanmill-target-refresh-mature-fork-replication-v1-attempt-002.json).
This document does not authorize launch.

## Why attempt 002 exists

The first replication contract was validly prepared at published commit
`1e88081e850bcea7f6fe56ab73f77df8015376a7`. Its readiness identity was
`a4d9fd63d77098b5b58d95712d38b9dc4b4b2a290a67773e7889ec8363c646d5`
and its readiness file SHA-256 was
`6995cd6ac179c715dae206c3d74c01c534163e8804013a368a4712dff33d161d`.
It contained no parent or child authorization and no training segment.

A later tracked handoff-only commit advanced `dev`. The parent sequence and
the managed trainer both require the exact clean published plan commit, so
that otherwise valid preparation became stale. Loosening only the parent
check would leave the trainer guaranteed to fail at launch. Attempt 002
therefore keeps every scientific, resource, measurement, stop-rule,
source-data and prohibited-operation decision unchanged, while assigning
fresh plan, database, common-fork, authorization, ledger and result paths at
the final published source.

The complete first preparation remains preserved under
`out/target-refresh-mature-fork-replication-v1/`. It must not be authorized,
launched, deleted, overwritten or relabelled.

## Frozen scientific design

The prior seeds 67–69 mature cohort had a paired
`refresh-mature minus stale-control` effect of `-0.0763889`, below the
predeclared absolute material boundary of `1/12`; only seed 67 materially
supported stale control. Attempt 002 uses disjoint seeds 64–66, whose
`no-refresh` checkpoints are already at exactly 8,192 post-game-50 consumed
transitions.

For each seed, preparation creates one normalized untreated mature fork and
two payload-equivalent branches:

1. `refresh-mature` copies the mature learner into the frozen opponent once;
2. `stale-control` retains the stale game-50 frozen opponent.

Each arm then consumes exactly 8,192 additional transitions in
64-transition A2C batches. Learning rate, transition-indexed temperature,
fixed 1,000-node Sanmill work, policy-health gate, components and data rules
match the first mature cohort. The immutable single-process order is seed 64
refresh/control, seed 65 refresh/control, then seed 66 refresh/control.

Seed 64 uses the previously frozen byte-identical closed database snapshot.
Its historical zero-byte WAL and 32,768-byte SHM sidecar remain untouched.
Seeds 65 and 66 use their already closed source databases. Every source is
bound to `sector-corrected-v1` metadata and exact checkpoint/database
identities.

## Frozen pooled decision

The replication cohort must first be material under the original direct rule:
absolute aggregate paired effect at least `1/12`, at least two supporting
replication seeds, the original opposite-seed guard and at most 25%
truncation.

A cadence input is selected only when the replication direction also has:

- pooled six-seed aggregate effect of at least `1/12`;
- at least three of six seeds supporting by at least `1/12`;
- no more than one opposite seed at or beyond `-1/12`;
- pooled truncation no higher than 25%.

The result explicitly records `automatic_long_run_selection=false`.
Training W/D/L and policy-distribution movement remain context and mechanism
evidence, not substitutes for the paired direct outcome.

## Resources and authority boundary

The immutable aggregate ceiling is 3,600 training games, 49,152 consumed
transitions, four active hours, 172,800,000 requested Sanmill node ceilings
and 288 no-update development games. Each arm is capped at 600 games and 0.6
active hours.

There is no automatic retry, recovery, resume, extension, held-out
evaluation, promotion, publication, retained run or long-training fallback.
Any identity drift, cohort overlap, treatment contamination, non-finite
state, database/sidecar drift, Sanmill mismatch, policy-health failure,
resource exhaustion or malformed result closes the sequence.

After this contract and the handoff update are published together, the
authorization-free preparation must be generated once at that exact commit.
No later tracked commit may be added before the parent decision and any
authorized execution complete. The plan itself grants zero launch authority.

