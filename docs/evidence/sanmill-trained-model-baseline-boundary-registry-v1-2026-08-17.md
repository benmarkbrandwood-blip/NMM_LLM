# Baseline-v1 boundary registry v1

Date: 2026-08-17

## Outcome

The attempt-002 parallel instrumentation inventories have been replaced by
one frozen boundary registry.  Static signature audit, public-surface audit,
stage evidence inventory, rehearsal coverage contract, and result-shape
checks are all derived from that registry.

The registry identity is
`1ceb23f52ffd722919940f0471ec32e4e5e63199549a675ce86b25ac4f7055f7`.
Its file SHA-256 is
`2a11dcc5464174e8cc8feb42ced29c955dfe08e0e0deee0898f93298f89286df`.
It contains 88 unique real-callable rows: 68 require a real Python return
event in the non-evidence rehearsal, six require explicit fail-closed
preflight canary evidence, six are static-audit-only, and eight are not
dynamically required for a row-specific recorded reason.

The rehearsal coverage-contract identity is
`164aa632b05ab6faad889937ac0afbec1333d0d048f96e9d384a4f865634fd07`.
This is an expectation identity, not a claim that the new rehearsal has run.
The actual append-only coverage-ledger identity can only be produced by the
authorization-bound attempt-003 rehearsal.

## Independent classification

Rows were classified from the corrected product-owner principle before
comparing them with the four missing attempt-002 methods.  A row is
`rehearsal-required` only when it is on formal gameplay or its durability
path, is safely reachable during rehearsal, and leaving it unexecuted would
leave a formal path untested.

`SanmillTrainingGame.search_and_apply` is therefore
`not-required-with-reason`.  The frozen experiment intentionally composes
`search_logical_turn` and `apply_nmm_move` so that it can validate and retain
the semantic search report.  The separate move-path audit proves equivalent
successful strict-state transitions, and the helper is not called by formal
execution.

`EstimatorAccess.load_decisions`, the protected access constructor and denial
methods, `_PoisonGameAI.__getattribute__`, and
`audit_specialist_gameai_dependency` are `preflight-required`.  Executing
them during rehearsal would either require a protected-access negative
canary or would test a later candidate-route gate rather than formal gameplay.

This independently derived result matches the corrected disposition of all
four attempt-002 missing methods.  The match is a cross-check, not the source
of the classification.

## Runtime observation design

The rehearsal observer uses `sys.setprofile` in a strictly non-evidence
scope.  It maps a frame's real `f_code` to the code object frozen in the
registry.  It does not wrap, proxy, replace, or change the signature of any
production callable.  The first successful return from each registered code
object is appended and fsynced as a hash-chained JSONL event.  Acceptance
recovers that ledger and rejects a missing required ID, an unknown ID, an
incorrect code identity, a failed result-shape validator, a duplicate event,
or a broken hash chain.

The hook is forbidden during formal execution.  Preflight negative canaries
use explicit evidence derived from the registry's preflight rows.  A new
preflight row without a canary proof fails closed rather than being silently
treated as covered.

## Static and durability gates

The registry freezes full reflected signatures, including positional and
keyword-only parameters, defaults, annotations, and return annotations.  It
also freezes Python source and bytecode identities for dynamic rows and every
public method or property on the registered boundary owners.  The static
audit rejects transparent `__getattr__` proxies, post-construction Malom
delegate rebinding, an unknown attribute interceptor, signature drift, code
drift, and an unregistered public method.

Malom counting remains at the single real `MalomDB.query_value` observer
throat.  No query proxy was reintroduced.  Per-game resource checkpoints are
still appended and fsynced before the corresponding game record.

## Focused verification

The structural suite proves that:

- signature drift and an unregistered public method are rejected;
- a newly required but unexecuted rehearsal row fails the dynamic gate;
- a preflight-only row cannot satisfy rehearsal coverage;
- an observed event maps to the registered real code object; and
- a simulated crash preserves the exact completed-game resource account.

The poison canary also has a direct negative regression: a real forbidden
attribute read must raise.  At the implementation freeze gate, the focused
baseline, registry, and route-policy suites reported 23 passing tests.  The
mandatory Malom, DB-teacher, and label-provenance group reported 103 passing
tests and 498 passing subtests.  Task-scope Ruff passed.

## Boundaries

No formal marker, formal game, training, fitting, checkpoint mutation,
database write, or protected-content read occurred during this structural
work.  Attempt-001 and attempt-002 records and output namespaces remain
unchanged.  The registry does not authorize execution by itself; attempt-003
must separately bind the registry and coverage contract before rehearsal.
