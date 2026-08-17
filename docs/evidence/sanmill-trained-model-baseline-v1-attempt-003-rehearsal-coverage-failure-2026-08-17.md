# Baseline-v1 attempt-003 rehearsal coverage failure

Date: 2026-08-17

## Outcome

Attempt-003 stopped before preflight and before the formal measurement
marker.  All 27 authorized non-evidence rehearsal games completed and their
durable ledgers align, but the frozen registry required 68 real code-object
return events and observed only 67.  The missing boundary is
`solved-db.close`, or `ExternalSolvedDB.close`.

The machine failure identity is
`f28adaccaaacefed0bcb2c2a4cc98feff5061c9b24e2d9f113b14ee8a7c7d17b`.
The actual coverage-ledger identity is
`1957ea999654f39610f7c9611ec6943a30af69bb59a81149dd665cb5400166f2`.
No unexpected boundary event occurred.

Under the product owner's defense-against-regression rule, this is the second
structural-contract stop.  No registry edit, point repair, new validation
layer, rehearsal retry, preflight, or formal execution will be attempted
without a new explicit product decision.

## What was empirically covered

The 27 games covered all four trained-model arms, both candidate colors, all
three source phases, free and positional-only `A_pos`-constrained selection,
and strict threefold, fifty-move, and decisive terminals.  All 27 reached a
strict rules terminal: seven draws and 20 decisive results.  The terminal
reasons were five threefold draws, two fifty-move draws, ten
`loseFewerThanThree`, and ten `loseNoLegalMoves`.

All other 67 frozen rehearsal boundaries produced successful, shape-valid
events from their real Python code objects.  This includes every Malom query
surface, the single query-counting throat, candidate scoring and selection,
manual Sanmill search/apply, strict referee state checks, game packaging, and
per-game durable resource recovery.

The output namespace contains 27 hash-chained game records and 27 fsynced
resource checkpoints.  Recovery ends at cumulative resources of 496 Sanmill
searches, 13,561 Malom queries, 482.0538405000116 active seconds, and 54
non-evidence games across all failed attempts.  Attempt-003 itself added 248
searches, 6,769 queries, 367.73950110000444 seconds, and 27 games.  Every
resource remains below the unchanged envelope.

## Remaining concrete risk

Read-only inspection explains the absent event.  `TrainingAlignedPolicy.close`
closes its SpecialistDB and HumanDB, but it does not call
`ExternalSolvedDB.close`.  The frozen formal route therefore would not call
that method either.  The registry classified a cleanup boundary as required
even though the formal route does not execute it.

This is not evidence of an untested move-selection or strict-referee path.
The remaining risk is lifecycle cleanup: the retained-v4 route's read-only
Malom adapter can keep mapped sector/cache resources open until process
teardown.  For a single bounded process, this is unlikely to alter a move or
game result, because the adapter remains intentionally live throughout the
run and its query throat was covered.  It could still affect handle or memory
retention and shutdown behavior.  That risk was not dynamically discharged,
so the frozen acceptance contract cannot be declared passed.

## Decision boundary

The product owner now has two choices:

1. explicitly lower the acceptance contract by waiving this cleanup-boundary
   event for the already completed rehearsal, accepting the lifecycle risk;
   or
2. abandon this measurement.

Choosing the first option would require a new, narrow authorization that says
how the frozen failed gate may be waived.  It must not be described as a
passed 68-of-68 rehearsal, and it does not authorize another rehearsal.  No
implementation or protocol change is being proposed here.

The official selection, confirmation, final-test, and
research-confirmation segments were not opened.  The remaining 108 records
in source pool `2eb04f54` were not read or consumed.  No training, fitting,
weight update, checkpoint change, database write, model claim, promotion,
deployment, publication, or release occurred.
