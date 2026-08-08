# Managed Policy-Health Gate Readiness — 8 August 2026

## Decision

`passed_for_fresh_retained_plan`

Commit `c070739a9c94938528e76083cf3ef69f997c7a5a` adds an optional,
hash-bound fixed-state policy-health gate to the managed Generalist
supervisor. Legacy plans omit the optional member and retain their original
canonical plan content and hash.

When a new plan enables the gate, a trainer process returning zero is no
longer sufficient to publish a completed segment. The supervisor first
validates the run ledger and checkpoint envelope, then invokes the committed
read-only policy audit. A segment becomes an exact-resume parent only after
the report passes all identity and policy-direction checks.

## Fail-closed boundary

The immutable gate binds:

- the audit script path and SHA-256;
- the fixed corpus path and SHA-256;
- the repository commit, experiment, segment run and completed game count;
- the exact checkpoint path and file SHA-256;
- the live closed SpecialistDB path and file SHA-256; and
- the paths configuration identity.

It requires exactly 29 Malom-critical states, direct-lookahead argmax
value-preserving rate 1.0, candidate argmax value-preserving rate at least
0.50, and candidate mean best-preserving minus best-downgrading logit at
least -0.10.

A non-zero or timed-out audit, missing or malformed report, identity drift,
critical-state-count drift, or threshold failure appends a durable
`managed_segment_policy_health_quarantined` event. The failed segment is not
recorded as completed and cannot become the next segment's parent. A direct
retry is rejected pending Agent diagnosis. These are anti-collapse limits,
not strength or promotion criteria.

## Verification

The changed management and training paths passed:

- 205 focused manager, preflight, launch, checkpoint, exact-resume,
  Generalist-update, frozen-opponent, Sanmill-referee, and policy-health
  tests;
- 103 mandatory Malom, DB-teacher, and label-provenance tests, including 498
  parameterized subtests;
- Ruff on all changed Python files; and
- `git diff --check`.

The complete `tests` invocation did not finish inside its 600-second bound.
It reached approximately 15 percent with no reported failure, matching the
repository's previously recorded full-suite timeout behaviour. This record
therefore does not claim a complete-suite pass.

Focused tests explicitly cover legacy-plan compatibility, gate success,
threshold quarantine, blocked retry, and missing or malformed audit output.
No test was skipped, deleted, or weakened to obtain these results.

## Real checkpoint exercise

On the clean published gate commit, the controller's real audit route was
run read-only against the completed corrected-learning smoke checkpoint and
its closed SpecialistDB. It produced the ignored local report:

```text
out/policy-health-gate-e2e-c070739/segments/segment-0001/policy-health.json
```

The report SHA-256 is
`3a43cfc24594b86ac72d2b53da05691fcee541b870ee960473306e48b7f79b69`
and its evidence ID is
`72297292dc9c31220ed3166322a0d11323f2073f3b56f30736a4e1da264d3cdc`.
The controller accepted:

| Metric | Required | Observed |
| --- | ---: | ---: |
| Direct critical states | exactly 29 | 29 |
| Candidate critical states | exactly 29 | 29 |
| Direct value-preserving rate | 1.0 | 1.0 |
| Candidate value-preserving rate | at least 0.50 | 0.965517 |
| Candidate preserving logit margin | at least -0.10 | +0.004366 |

This exercise reuses an old smoke only as read-only gate input. Its model,
checkpoint, database and report remain excluded from the retained lineage.
No training process was started by this verification.

## Consequence

The supervisor gate is ready to be required by a new immutable 5,000-game
plan. The retained plan must still use fresh random initialization, a new
empty trusted SpecialistDB and an absent output directory. It must pass a
new exact read-only preflight on its final published source before launch.
