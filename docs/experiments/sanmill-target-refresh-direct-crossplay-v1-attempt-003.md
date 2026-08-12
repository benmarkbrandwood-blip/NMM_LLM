# Target-refresh direct cross-play v1 attempt 003

Status: `designed_unlaunched_needs_readiness`

This is a fresh one-shot successor to
[attempt 002](sanmill-target-refresh-direct-crossplay-v1-attempt-002.md).
Attempt-002 was never authorized or launched. Its stored readiness became
non-reproducible after an unrelated SQLite reader changed only the active
HumanDB `-shm` mtime. The deterministic reproduction and correction boundary
are recorded in the
[attempt-002 readiness-drift evidence](../evidence/target-refresh-direct-crossplay-attempt-002-readiness-drift-2026-08-12.md).
The attempt-002 directory remains immutable.

## Unchanged scientific contract

Attempt-003 retains the complete attempt-002 scientific and resource design:

- the same seed-67, seed-68 and seed-69 checkpoint pairs at exactly 8,192
  post-fork consumed transitions;
- the same per-seed game-50 frozen model as the dynamic lookahead anchor;
- the same twelve audited histories, split equally across placement,
  movement and flying;
- four replicates per start, colour swapping and common per-colour random
  streams;
- training-policy sampling at temperature `0.2`;
- Sanmill as strict portable referee only, never as a move selector;
- a 120-post-start-logical-ply development truncation cap;
- exactly 144 pairs and 288 CPU no-update games within two active hours; and
- the same aggregate, per-seed, opposite-direction and truncation decision
  thresholds.

It remains development evidence only. It cannot authorize held-out
evaluation, model promotion, publication or long training.

## Corrected execution-integrity boundary

Implementation commit
`5254b10937c650dcd389edd74ac39123a7f24a1c` preserves the immutable
HumanDB main-file route and excludes volatile WAL/SHM metadata from readiness
identity. The runner records sidecar snapshots as telemetry but bases drift
decisions only on the files that belong to the experiment view.

The following remain fail closed:

- HumanDB main-file identity, `quick_check`, size and mtime;
- masked historical Malom-label policy;
- Malom manifest identity and `std.secval` hash, size and mtime;
- all checkpoint, corpus, source, schedule and referee identities; and
- every existing no-write, legality, terminal and resource gate.

The focused and neighbouring launch gates report `207 passed, 498 subtests
passed`. Ruff, Python compilation and `git diff --check` pass. The regression
proves that sidecar-only timestamp changes preserve the stable observation
identity while HumanDB-main or Malom changes do not.

## Isolation and authority

The authoritative JSON plan has identity
`2f1665e59aaa7af96af345381338689a7edd51c55401a13a8c9fd4c8a58535ff`
and SHA-256
`17c43b513602479d00eed5b36a5b5c02a779c31111894f72ab7acf6bced20c25`.
Every attempt-003 output is isolated below
`out/target-refresh-direct-crossplay-v1-attempt-003`.

Preparation may occur only after this plan is committed, ordinarily pushed,
and `dev == origin/dev` with a clean tracked worktree. The resulting readiness
must rebuild exactly before launch. A launch still requires one explicit
product authorization bound to that new readiness identity. An anomaly
consumes the launch and fails closed; there is no automatic retry, recovery,
extension, held-out evaluation, promotion, publication or long-training
fallback.
