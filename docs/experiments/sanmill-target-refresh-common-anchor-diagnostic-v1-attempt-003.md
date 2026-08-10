# Sanmill target-refresh common-anchor diagnostic v1 attempt 003

Status: `designed_unlaunched_needs_publication`. The machine-readable source
of truth is
[`sanmill-target-refresh-common-anchor-diagnostic-v1-attempt-003.json`](sanmill-target-refresh-common-anchor-diagnostic-v1-attempt-003.json).
This document and its tooling do not authorize training.

## Why this is a new attempt

Attempt 002 completed both seed-64 arms at their exact 34-update bounds. They
passed policy health and shared byte-identical first-50 game rows and an
identical game-50 model anchor. The result analyser then rejected their valid
122- and 92-game completion counts because a reused fixed-game helper required
the 150-game safety ceiling. The sequence stopped before seed 65, produced no
result and selected no training setting. The failure and its artefact
identities are frozen in
[`target-refresh-common-anchor-diagnostic-attempt-002-failure-2026-08-10.md`](../evidence/target-refresh-common-anchor-diagnostic-attempt-002-failure-2026-08-10.md).

Commit `873e1265fc98636cefc7a561e3d139f8fce621e5` changes result validation
only. It accepts a controller-validated optimizer-bounded completion count,
preserves the fixed-game default and rejects invalid or over-ceiling counts.
Focused checkpoint, measurement, manager, preflight, result-analysis, Malom
and label-provenance tests passed after the correction.

Attempts 001 and 002 are not resumed or retried. Their grants and databases
are consumed. Attempt 003 therefore has a new content identity, plan IDs,
experiment IDs, control directories and four fresh SpecialistDB paths. None
of the earlier attempt files may be reused, relabelled, moved into the
successor, or treated as result evidence.

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
before any attempt-003 arm can start. Any arm anomaly again stops the entire
sequence without automatic retry, continuation, extension or resume.
