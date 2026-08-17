# Baseline-v1 final authorization static-conflict stop

Date: 2026-08-17

## Outcome

The final authorization stopped during its required pre-change static walk.
No registry reference was changed, no new attempt or machine authorization was
frozen, and no preflight, model load, Sanmill process, query, search, or game
started.  Under the authorization's final ceiling, the trained-model baseline
measurement is abandoned.

## Complete forward reference inventory

The code and test search found five hard-coded v1 registry paths and no
hard-coded old registry or coverage-contract identity in the toolchain:

| File and line | Role |
| --- | --- |
| `learned_ai/evaluation/sanmill_trained_model_baseline.py:490` | shared instrumentation-surface audit |
| `scripts/preflight_sanmill_trained_model_baseline.py:596` | zero-game preflight default |
| `scripts/rehearse_sanmill_trained_model_baseline_attempt_002.py:448` | rehearsal acceptance default |
| `scripts/run_sanmill_trained_model_baseline.py:262` | formal execution and result-manifest default |
| `tests/test_sanmill_trained_model_boundary_registry.py:30` | registry contract test fixture |

Each row could mechanically replace the v1 filename with the frozen v2
filename without changing a predicate.  No replacement was applied because
the same pre-change walk found two independent authorization conflicts.

Frozen v1 plans, authorizations, ledgers, failures, and correction records are
historical provenance rather than forward pointers.  They were deliberately
excluded from alignment.  The registry module's schema name ending in `.v1`
is also a data-schema version, not a registry-instance pointer, and remains
unchanged.

## Fatal conflict 1: required passed record

The preflight binds the rehearsal to the current attempt and authorization,
then requires status `passed_non_evidence_technical_rehearsal` at line 627.
The existing attempt-003 record is and must remain
`failed_closed_after_registry_dynamic_coverage_gate`, under failure identity
`f28adacc...` and coverage ledger `1957ea99...`.

There is no same-attempt, same-authorization passed record.  Creating one from
the existing 27 games would describe the existing rehearsal as passed, which
this authorization explicitly forbids.  Changing the status predicate would
change validation logic, and rerunning the rehearsal is also forbidden.
Therefore no admissible preflight input exists even after path alignment.

## Fatal conflict 2: lifecycle evidence has no runtime path

The frozen corrected attempt requires evaluator and Sanmill peak RSS and open
handle counts, with resource exhaustion failing closed.  The formal runner
contains no peak-RSS or handle collector and writes no lifecycle-monitor
record.  Adding such instrumentation, including an ad-hoc external monitor,
would exceed the authorization's reference-only code-change boundary.

This is independent of the passed-record conflict.  Either conflict prevents
the complete preflight-to-evidence chain.

## Recorded technical debt

Two independent items remain recorded without implementation:

- registry version paths are repeated in five places and need a separately
  authorized single-source design;
- `TrainingAlignedPolicy` owns but does not close its `ExternalSolvedDB`
  adapter, leaving its mappings and handles live until process teardown.

No parser, registry mechanism, validation layer, lifecycle fix, or workaround
was built in this round.

## Preservation and resource boundary

Registry v2 `7ef7b1b1...` was not modified.  Registry v1, failure
`f28adacc...`, and coverage ledger `1957ea99...` remain unchanged; the old
rehearsal is neither 68/68 nor passed.

Cumulative consumption remains exactly 496 engine searches, 13,561 Malom
queries, 54 non-evidence games, and 482.0538405000116 active seconds.  This
round added zero to every counter.  Official selection, confirmation,
final-test, research-confirmation, and source pool `2eb04f54` remained
unopened.  No candidate result or baseline comparison exists, so there is no
strength, equivalence, product, promotion, deployment, publication, release,
or training conclusion.
