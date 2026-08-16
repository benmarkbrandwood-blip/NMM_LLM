# Safe-Guidance Gameplay Execution Failure — 16 August 2026

## Disposition

The explicitly authorized second and final zero-game preflight passed.  Its
identity is
`2af758a0d51c5eb7e0e84a6e552f8759cbc424f4df0481df2fb9dec542509401`.
The negative canary regression rejected a genuinely mismatched move, all ten
focused gameplay tests passed, the required Malom/DB-teacher/provenance group
passed with 103 tests and 498 subtests, Ruff passed, and all runtime,
determinism, protected-data, and strict-history gates passed.

The once-only execution then failed closed on the first scheduled game.  The
game-zero marker exists, so authorization
`806e7b674c96ca3f5dd98067a09b6c76bda3db2cca12c75d92ba3cc5f7b495e2`
is consumed.  No retry, resume, continuation, batching, or second execution
was attempted or is authorized.

The sealed failure identity is
`7aa4f771c3871cfd48c6935167c5f8c1ad40534281bcf2069c2ad3d36a2dc55f`.

## Failure

The first schedule item was the random-safe arm as White from placement start
`00092c974cab...`.  The strict referee reached a rules terminal.  While
packaging that result, the evaluator raised `KeyError: 'winner'` before it
could write the first game record.

`UciPositionState.portable_record()` stores the terminal fields under
`final_state["outcome"]`.  The new evaluator instead read
`final_state["winner"]` and would next have read
`final_state["outcome_reason"]`.  Existing held-out and retained diagnostic
evaluators use `game.state.winner` and `game.state.outcome_reason`, which
confirms the interface mismatch.  This is an evaluator result-packaging
defect, not a Sanmill or Malom component failure.

Exactly one rules-terminal game was reached, because the failing branch is
entered only after the loop breaks on `game.state.terminal`.  It is not an
accepted result: no raw ledger, progress record, completion marker, or tracked
result manifest exists.

## Resource accounting

The last sealed exact baseline before measurement was cumulative:

- 72 engine single-step searches;
- 12,638 read-only Malom queries;
- 116.4832933 active seconds; and
- zero complete games.

The failed first game used additional engine searches, Malom queries, and
active time, but those exact increments are not recoverable.  The runner kept
its resource ledger only in memory and planned to write the first progress
record after terminal packaging and serialization.  The observed runner
process wall time was 12.4696033 seconds, but that is not substituted for the
missing active-time increment.  The evidence therefore records the last exact
baseline, one rules-terminal game reached, and unknown execution increments;
it does not manufacture estimates.

The absence of a durable per-turn resource journal is a second execution
harness defect.  A future design would need to fix both the terminal-state
interface and resource durability before seeking new authority.  Neither fix
nor any execution retry was performed here.

## Boundaries

All official selection, confirmation, final-test, and research-confirmation
content remained unopened.  The remaining 108 records in source pool
`2eb04f54` were not read or consumed.  There were zero model loads, fits,
training updates, or database writes.

No score difference, conversion rate, human-trap ability, product value,
promotion, deployment, publication, release, or training conclusion can be
drawn.  The machine-readable evidence is
`docs/evidence/sanmill-safe-guidance-gameplay-execution-failure-2026-08-16.json`.
Any repair, preflight, restart, replay, or continuation requires a new explicit
product-owner decision.
