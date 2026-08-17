# Baseline-v1 classification-correction readiness stop

Date: 2026-08-17

## Outcome

The classification correction was completed, but the measurement was
abandoned before the zero-game preflight.  No preflight process, candidate
load, formal namespace, measurement marker, engine search, Malom query, or
game was started in this correction round.

The new authorization identity is
`89457fc102a58b2e8c12ef7541643fcca0e6865c2da6f896db06921d4cfc1805`.
It binds the old registry, immutable old coverage ledger, corrected registry,
and the one-to-one event mapping.  It was not consumed by a formal marker,
but it is operationally closed and must not be reused after this mandated
abandonment.

## Classification correction

All 68 old rehearsal-required rows were re-derived from the frozen three-part
principle, and the other 20 rows were audited for false negatives.  Four rows
changed from rehearsal-required to static-audit-only:

- `solved-db.close`;
- `solved-db.query-all-moves`;
- `solved-db.query-move-quality`;
- `solved-db.query-trajectory`.

No row changed in the opposite direction.  Registry v2 identity
`7ef7b1b17d1f0baa94159c0161a6c5082f39f8571941e2c7e47e4d97f98405e7`
contains 64 rehearsal-required, six preflight-required, ten
static-audit-only, and eight not-required-with-reason rows.  Its static audit
passes.  All 64 dynamic requirements are represented in the 67 immutable
old events.

This correction was made after the missing event was observed.  Its result is
nonetheless statically derivable from the frozen call graph and classification
principle.  The old coverage ledger `1957ea99...` remains byte-for-byte
unchanged, and failure `f28adacc...` remains a 67-of-68 failure.  The old
rehearsal is not 68/68 and is not passed.

## Lifecycle finding

`load_training_aligned_policy` creates `ExternalSolvedDB`, and
`TrainingAlignedPolicy` owns the resulting reference.  There is no shared
external owner.  Its `close` method closes SpecialistDB and HumanDB but not
the Malom adapter.  This is a cleanup omission that leaves the read-only
adapter's mappings and handles live until process teardown, not intentional
shared ownership.  No fix was made.

The authorized formal risk control would have recorded evaluator and Sanmill
peak RSS and open-handle counts, with resource exhaustion failing closed.
Because no formal evaluator process started, there is no lifecycle monitor or
resource observation to report.

## Why preflight could not start

The corrected attempt and authorization require registry v2.  The unmodified
function `audit_instrumentation_surface` in
`learned_ai/evaluation/sanmill_trained_model_baseline.py` has no registry
parameter and hard-codes the v1 registry path at line 490.  The preflight
loads the requested registry separately, calls that hard-coded audit at line
774, and at line 777 requires the audit identity to equal the requested
registry identity.  It raises `instrumentation surface audit failed` at line
780 when they differ.

Direct evaluation confirms that the audit returns v1 identity
`1ceb23f5...`, not required v2 identity `7ef7b1b1...`.  Therefore preflight
cannot pass under the frozen implementation.  Changing this behavior would
be a tooling change, while this round explicitly prohibited tooling changes
and required abandonment after another tooling or contract-consistency stop.
The predictable failing preflight was not launched merely to create partial
output.

No repair proposal is included.  This measurement is abandoned.

## Verification and boundaries

The existing focused suite passed 23 tests.  The mandatory Malom,
DB-teacher, and label-provenance group passed 103 tests and 498 subtests.
Task-scope Ruff passed.  Pytest emitted only its existing cache-write warning;
repository-local test basetemps were used.

The cumulative sunk resources remain exactly 496 engine searches, 13,561
Malom queries, 54 non-evidence games, and 482.0538405000116 active seconds.
This correction round added zero to every counter.

Official selection, confirmation, final-test, and research-confirmation
content remained unopened.  Source-pool `2eb04f54` remained unread and its
108 records remain unconsumed.  There was no training, model or checkpoint
change, database write, baseline comparison, strength or equivalence claim,
promotion, deployment, publication, or release.
