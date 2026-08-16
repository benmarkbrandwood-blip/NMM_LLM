# Baseline-v1 attempt-002 tooling structural analysis

Date: 2026-08-16

## Outcome

Attempt-002 is stopped before preflight.  The 27-game non-evidence run
completed, but it does not satisfy its frozen dynamic-coverage contract.  Its
sealed machine record says `passed_non_evidence_technical_rehearsal`; that
status is not accepted by the post-run acceptance audit.

The acceptance-failure identity is
`57f90937b336acb16ad09afae40fef3cafd9c806a001c428890f2f9ff426428a`.
No formal output namespace or measurement marker exists.  No further point
repair, rehearsal retry, corrected preflight, or formal execution has been
started.

## What completed correctly

The run completed 27 strict-referee games: 24 live games covering three source
phases, all four arms, and both candidate colors, plus real strict replays for
threefold, fifty-move, and decisive terminals.  Every game has an fsynced
resource checkpoint before its game record, and recovery aligns all 27
records.  Free and `A_pos`-constrained choices both executed.

All Malom surfaces exercised by the canary returned the expected shape and
exact query delta.  The old failed namespace and authorization remained
unchanged.  Protected content reads, source-pool reads, database writes,
training updates, and checkpoint changes were all zero.

The cumulative cost, including attempt-001 sunk cost, is 248 Sanmill searches,
6,792 Malom queries, 27 non-evidence games, and 114.31433940000716 active
seconds.  This remains inside the unchanged resource ceiling.

## Decisive acceptance defect

The frozen supplement required every method named by the instrumentation
inventory to execute at least once in the rehearsal.  Four named methods did
not execute:

1. `SanmillTrainingGame.search_and_apply`.  The baseline path intentionally
   calls `search_logical_turn` and `apply_nmm_move` separately so it can retain
   the engine semantic report.
2. `EstimatorAccess.load_decisions`.  The guard canary called `derive`, which
   called `assert_allowed` and rejected the protected session before any
   producer ran, but it did not enter `load_decisions` itself.
3. `_PoisonGameAI.__getattribute__`.  This canary exists only in the later
   zero-game preflight.
4. `audit_specialist_gameai_dependency`.  The rehearsal called the generic
   instrumentation audit, but not this separate dependency audit.

The rehearsal acceptance code verified phase, arm, color, safety mode,
terminal reason, recovery, and Malom-canary coverage.  It did not compare a
runtime call trace against the frozen interception inventory.  Consequently,
its machine `passed` status proves the checks it implemented, not the stronger
coverage requirement that was frozen.

Changing the inventory after seeing this result would be a post hoc contract
change.  Calling the missing methods now would be a second rehearsal suffix
without authorization.  Both are prohibited.  The only compliant disposition
is fail closed.

## Why contract failures repeated

The two failures have the same structural origin even though their immediate
symptoms differ.

Attempt-001 had a transparent proxy whose hand-written `query()` signature and
resource call diverged from the real delegate.  Attempt-002 removed that proxy
correctly, but retained several parallel, manually maintained descriptions of
the experiment:

- the callable surface and signature audit;
- the prose and JSON inventory of intercepted methods;
- the live rehearsal schedule;
- the hand-written coverage booleans; and
- the later preflight-only canaries.

There is no machine-enforced relationship between those descriptions.  A
method can be added to the inventory without becoming a required runtime event,
and a runtime path can be covered without producing an identity tied to the
inventory.  The term `intercepted` also conflates three different classes:

- methods on the actual formal gameplay path;
- infrastructure methods exercised to prove durability or access denial; and
- preflight-only negative canaries that are not part of gameplay.

That ambiguity made the frozen requirement stricter than the executable
acceptance gate.  The current layering is therefore itself a defect source;
another local call or boolean would leave the parallel-contract problem in
place.

## One structural remedy

Any future authorization should first replace the parallel descriptions with
one frozen boundary registry.  Each registry row should contain:

- a stable boundary ID and owning module;
- the real callable and its complete inspected signature;
- its role: resource throat, formal gameplay, durability, protected guard, or
  canary;
- its required stage: static audit, rehearsal, preflight, or formal run;
- whether dynamic execution is mandatory in that stage; and
- its expected result-shape validator and resource semantics.

The static audit, dynamic coverage gate, evidence inventory, and tests should
all be generated from that registry.  Adding a method would then necessarily
change the frozen registry identity and the expected dynamic event set.

Dynamic coverage should observe real Python function-entry events by registered
code object, for example with a scoped profiling hook.  This avoids another
delegating proxy and does not alter production signatures.  The hook should be
active only in the non-evidence rehearsal and should emit an append-only event
ledger.  Acceptance should require exact coverage of every registry row tagged
`rehearsal-required`; a missing row must fail even if all games finish.

Preflight-only rows such as the poison canary should be tagged and verified in
preflight, not ambiguously listed as rehearsal-required.  Formal gameplay
should continue using the single Malom query throat and explicit durable
resource checkpoints.  The registry and coverage-ledger identities must be
bound by any future authorization.

At minimum, the replacement needs tests that:

- reject signature drift and an unregistered public method;
- add a rehearsal-required registry row without executing it and prove the
  dynamic gate fails;
- prove a preflight-only row cannot satisfy a rehearsal requirement;
- prove each observed event maps to the registered real code object; and
- preserve exact completed-game resources through a simulated crash.

This is a proposed architecture, not an authorized implementation.  It would
change the tooling structure and frozen attempt contract, so it requires a new
explicit product-owner decision before any code change or execution.

## Preserved records and boundaries

The attempt-002 rehearsal result and its six output files are retained byte for
byte as non-evidence execution history.  The earlier attempt-001 failure and
old authorization also remain unchanged.  Neither machine `passed` label is a
formal candidate result.

The official selection, confirmation, final-test, and research-confirmation
segments remain unopened.  The remaining 108 records in source pool
`2eb04f54` remain unread and unconsumed.  The result makes no strength,
equivalence, human-trap, training-value, promotion, deployment, publication,
or release claim.
