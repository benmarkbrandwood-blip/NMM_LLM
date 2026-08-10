# Sanmill target-refresh common-anchor diagnostic v1 attempt 002

Status: `designed_unlaunched_needs_publication`. The machine-readable source
of truth is
[`sanmill-target-refresh-common-anchor-diagnostic-v1-attempt-002.json`](sanmill-target-refresh-common-anchor-diagnostic-v1-attempt-002.json).
This document and its tooling do not authorize training.

## Why this is a new attempt

Attempt 001 stopped at game 50 before producing a measurement anchor. The
trainer used the dedicated role `development_measurement_anchor`, but the
checkpoint envelope had not admitted that role. The failure occurred before
an accepted segment, candidate checkpoint, result, or later arm existed. The
failure and its artefact identities are frozen in
[`target-refresh-common-anchor-diagnostic-attempt-001-failure-2026-08-10.md`](../evidence/target-refresh-common-anchor-diagnostic-attempt-001-failure-2026-08-10.md).

Commit `e02aca46364280674fb564ba68a536fce45292c7` adds the two dedicated
development-measurement roles while preserving fail-closed rejection of
unknown roles. Focused checkpoint, measurement, manager, preflight, Malom and
label-provenance tests passed after the correction.

Attempt 001 is not resumed or retried. Its four grants are consumed. Attempt
002 therefore has a new content identity, plan IDs, experiment IDs, control
directories and four fresh SpecialistDB paths. None of the attempt-001 files
may be reused, relabelled, moved into the successor, or treated as result
evidence.

## Unchanged scientific contract

The hypothesis, two seeds, four-arm order, target-refresh treatment, fixed
learning rate, common game-50 anchor, optimizer-update matching, no-update
measurements, policy-health gate, resource ceilings and stop rules remain
unchanged from v1. This is an infrastructure-corrected successor, not a
post-result redesign. The primary contrast remains `no-refresh minus refresh`
against the common frozen model anchor; fixed-node Sanmill remains a separate
measurement stratum.

The claim boundary also remains unchanged: this can produce development-only
mechanism evidence. It cannot select a retained setting, authorize held-out
evaluation, establish playing strength, promote or publish a model, or start
long training.

## Launch boundary

The source commit containing this contract must first be published to
`origin/dev`. Preparation may then create only four new immutable plans,
preflights and closed fresh `sector-corrected-v1` SpecialistDB copies. A new
readiness identity and a new explicit product authorization are required
before any attempt-002 arm can start. Any arm anomaly again stops the entire
sequence without automatic retry, continuation, extension or resume.
