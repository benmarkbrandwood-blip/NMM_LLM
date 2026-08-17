# Baseline-v1 boundary-registry v2 classification correction

Date: 2026-08-17

## Disposition

The attempt-003 rehearsal remains failed under registry v1.  It observed 67
of 68 v1-required events, and `solved-db.close` was absent.  Its coverage
ledger identity remains
`1957ea999654f39610f7c9611ec6943a30af69bb59a81149dd665cb5400166f2`,
and failure identity
`f28adaccaaacefed0bcb2c2a4cc98feff5061c9b24e2d9f113b14ee8a7c7d17b`
is not withdrawn.  The old rehearsal is not 68/68 and is not passed.

This correction was made after the missing event was observed.  The result
prompted a full review, but the classification predicate is static: a boundary
is rehearsal-required if and only if it lies on formal gameplay or durability,
is reachable without weakening a guard, and omitting it leaves a formal path
untested.  No observed score, move, result, or coverage absence is an input to
that predicate.

Registry v2 has identity
`7ef7b1b17d1f0baa94159c0161a6c5082f39f8571941e2c7e47e4d97f98405e7`
and file SHA-256
`23a2ef1066d27efa55ef77784cbf30060a7a9e3147b3cc4dc83b713dea5cfa91`.
The machine correction record has identity
`1f51e4f92e51301e0e958b710503f0691b83e4b510da2db24d1ae9c3bff69477`.

## Full rederivation

All 68 v1 rehearsal-required rows were re-evaluated.  Sixty-four satisfy all
three frozen predicates and remain rehearsal-required.  Four do not lie on
the formal route and become static-audit-only:

- `solved-db.close`: `TrainingAlignedPolicy.close` closes SpecialistDB and
  HumanDB but never calls `ExternalSolvedDB.close`; no other frozen formal
  caller exists.
- `solved-db.query-all-moves`: the training-aligned encoder receives
  `db=None`; its lookahead advisor uses `ExternalSolvedDB.query` instead.
- `solved-db.query-move-quality`: only the all-surface rehearsal canary calls
  this method.
- `solved-db.query-trajectory`: only the all-surface rehearsal canary calls
  this method.

The other 20 registry rows were also checked for false negatives.  None needs
to become rehearsal-required.  The corrected totals are 64 rehearsal-required,
6 preflight-required, 10 static-audit-only, and 8
not-required-with-reason.  The complete row-by-row predicate result is in the
machine correction record.

The 67 immutable v1 events were re-sealed one-to-one under the v2 registry
identity without executing any Python function again.  Each re-sealed event
embeds its source event SHA-256.  The source ledger remains byte-for-byte
unchanged.  The derived ledger identity is
`2b0ef88576a805b71bb694e73ebc86090a8433358ebcab57aed32724ee6abc1a`.
It contains all 64 v2-required events plus the three canary-only events that
v2 now treats as static.  This is a classification-identity mapping, not a
new rehearsal and not a claim that v1 passed.

## Malom adapter ownership

`load_training_aligned_policy` constructs `ExternalSolvedDB` and passes it to
`TrainingAlignedPolicy`, which stores it as `self.malom`.  No shared external
owner is supplied.  Therefore the missing close is a resource-lifecycle
cleanup omission, not intentional shared ownership.  This round does not fix
it.

The formal contract accepts a bounded lifecycle risk only: the retained-v4
Malom adapter remains open until evaluator-process teardown, when the operating
system releases its mappings and handles.  Formal operation must record peak
RSS and peak open-handle counts for the evaluator and Sanmill processes.
Any resource-exhaustion signal fails closed.

## Scope

No gameplay, candidate loading, model, checkpoint, database, threshold, start
pool, arm, metric, or implementation code changed.  No rehearsal was rerun.
Official selection, confirmation, final-test, and research-confirmation
content remained unopened.  Source-pool `2eb04f54` remained unread and its
108 records remain unconsumed.
