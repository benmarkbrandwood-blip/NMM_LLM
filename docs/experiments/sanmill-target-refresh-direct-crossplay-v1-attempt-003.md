# Target-refresh direct cross-play v1 attempt 003

Status: `completed_once_material_no_refresh_direct_effect`

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

## Completed result

The product owner authorized the unchanged three-seed, 288-game CPU scope on
12 August 2026. The supplied readiness identity referred to consumed
attempt-001 rather than current attempt-003. Because the requested objective,
workload and boundaries were otherwise exact, the operator recorded the
identity correction in a fresh authorization instead of changing the
scientific contract. Attempt-003 then completed once under readiness identity
`9fd354a793711ff925239a5f548b415134373d74d468ddf5961aead029cf3265`
and authorization identity
`3175570e24a063d222a4d2f83ce8f145adb6f246163d6740f32b1754de8b1f30`.

All 288 no-update games completed in about 0.049 active hours. The no-refresh
condition produced `178 W / 12 D / 98 L`, with a paired mean score effect of
`+0.2777778` over refresh-once. All three seeds and all three phases supported
the no-refresh direction. The truncation rate was `0.03125`, below the frozen
invalidation limit. The result was therefore classified
`material_no_refresh_direct_effect`.

The result does not select permanent no-refresh or authorize long training.
The next discriminating test is a mature-boundary fork: refresh the target
once from each seed's mature no-refresh checkpoint in the treatment arm,
leave it stale in the control arm, and hold all post-fork optimizer exposure
and measurement work equal. The frozen evidence and complete interpretation
are recorded in
[the attempt-003 result](../evidence/target-refresh-direct-crossplay-attempt-003-result-2026-08-12.md).
