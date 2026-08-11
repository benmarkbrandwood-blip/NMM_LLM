# Target-refresh direct cross-play v1 attempt 002

Status: `designed_unlaunched_needs_authorization`

This is a fresh one-shot successor to
[attempt 001](sanmill-target-refresh-direct-crossplay-v1.md). Attempt 001
[failed closed](../evidence/target-refresh-direct-crossplay-attempt-001-failure-2026-08-12.md)
before schedule ordinal zero because the runner requested abbreviated policy
seed fields. Its authorization and output directory are consumed and remain
immutable.

## Unchanged scientific contract

Attempt 002 retains the complete attempt-001 scientific design:

- the same seed-67, seed-68 and seed-69 checkpoint pairs at exactly 8,192
  post-fork consumed transitions;
- the same per-seed game-50 frozen model as the dynamic lookahead anchor;
- the same twelve audited histories, with four placement, four movement and
  four flying starts;
- four replicates per start, colour swapping and common per-colour random
  streams;
- training-policy sampling at temperature `0.2`;
- Sanmill as strict portable referee only, never as a move selector;
- a 120-post-start-logical-ply development truncation cap;
- 144 pairs and 288 CPU no-update games; and
- the same aggregate, per-seed, opposite-direction and truncation decision
  thresholds.

It remains development evidence only. It cannot authorize held-out evaluation,
model selection, promotion, publication or long training.

## Corrected implementation boundary

Implementation commit
`2db2f945becbf2da0c208547ae624f53778274dd` explicitly maps White and Black
to the closed schedule fields `policy_seed_white` and
`policy_seed_black`. The focused regression constructs the real deterministic
schedule, verifies both generator streams against their frozen integer seeds,
and fails closed when an abbreviated field is supplied. The relevant focused,
Generalist-preflight, schedule, replay, referee, Malom and provenance gate
reports `200 passed, 498 subtests passed`; Ruff and `git diff --check` pass.

No gameplay, policy sampling, checkpoint, corpus, HumanDB, Malom, referee,
resource or decision rule changed.

## Isolation and authority

The JSON plan is authoritative under identity
`4b0a873020b1ee157f01578d2565b9ff27917ee2f15fc00128e0ae91a6ead56f`.
Every attempt-002 output is isolated under
`out/target-refresh-direct-crossplay-v1-attempt-002`. Preparation must occur
only from clean published `dev`; readiness must reproduce byte-for-byte before
launch. The attempt requires one new explicit product authorization bound to
that readiness identity. An anomaly consumes the launch and fails closed;
there is no automatic retry, recovery, extension, held-out evaluation,
promotion, publication or long-training fallback.
