# Mature target-refresh analysis recovery v2

## Status and purpose

Status: `completed_once_no_material_direct_effect`

The machine-readable contract is
[`sanmill-target-refresh-mature-fork-analysis-recovery-v2.json`](sanmill-target-refresh-mature-fork-analysis-recovery-v2.json).
Its plan identity is
`32158846cb3e3903589663465d6217ed546442eee617e0aa5fe94defe45feb25`.

The one-shot recovery completed on 13 August 2026 under readiness identity
`13e25cd5fb0552dd171b3f736e42841c1222bb627f08447abbb8a77bb7031fab`.
It produced 288 no-update development games, result identity
`5e7bb7bf0505d1f3a2b43f50572e1ed9de8861114a25193490e34533f4dafd61`,
and classification `no_material_direct_effect`. It performed no training,
optimizer, database, or checkpoint write. Neither condition is selected, and
the authorization is consumed. Preserve the full
[result evidence](../evidence/target-refresh-mature-fork-analysis-recovery-v2-result-2026-08-13.md).

The remaining sections preserve the exact prelaunch contract and are not a
new launch instruction.

Recovery v1 consumed its one-shot authorization and failed closed before the
first development game. It incorrectly required transient mature-fork capture
metadata to remain in ordinary post-fork checkpoint implementation identity.
The failure, exact control identities and zero-game boundary are frozen in the
[v1 failure record](../evidence/target-refresh-mature-fork-analysis-recovery-v1-failure-2026-08-13.md).

This successor changes only evidence ingestion and preflight. It does not
retry or resume training and does not alter any model, optimizer, database,
checkpoint, schedule, policy corpus, replay corpus, rule, seed or scientific
threshold.

## Frozen scope

- Inputs: the same six completed seed 67/68/69 `refresh-mature` and
  `stale-control` branches from mature target-refresh attempt 002.
- Policy analysis: the same 64 placement/movement/flying states at 4,096 and
  8,192 post-fork transitions.
- Outcome analysis: the same 288 CPU no-update games, twelve audited starts,
  four replicates, colour swaps, common random streams, 0.20 sampling
  temperature, strict Sanmill referee and 120-logical-ply development ceiling.
- Additional training games and optimizer updates: zero.
- Database and checkpoint writes: zero.
- Maximum additional active time: 3.5 hours.
- Claim boundary: development mechanism evidence only; not held-out strength,
  retained-setting selection, promotion, publication or long training.

## Corrected evidence gate

Ordinary checkpoints retain the stable managed-run implementation mapping.
The initial branch envelope alone carries `mature_target_refresh_*` capture
metadata and `target_refresh_branch_*` treatment metadata. Durable fork
semantics remain independently bound by the checkpoint experiment/config,
trainer recovery state, treatment, fork game, transition origin, consumed
transition count, source-checkpoint path and immutable asset identities.

Before readiness may be published, preflight loads and validates all 12
candidate checkpoints:

- seeds 67, 68 and 69;
- transition boundaries 4,096 and 8,192; and
- both `refresh-mature` and `stale-control` conditions.

The candidate audit identity and exact pair count become part of readiness.
The post-fix local diagnostic audit covered all 12 checkpoints and produced
identity
`d3c7e0dd5b611a9bec4086355035ed462f3fd11fbc006d49781b1f47b35340e0`.
That local result is implementation verification, not launch readiness; final
readiness must be regenerated after this contract and its exact source are
published.

## One-shot preparation

After ordinary fast-forward publication, run from clean `dev == origin/dev`:

```powershell
.\.venv\Scripts\python.exe `
  scripts\run_target_refresh_mature_fork_analysis_recovery.py `
  --preflight `
  --plan docs\experiments\sanmill-target-refresh-mature-fork-analysis-recovery-v2.json `
  --readiness out\target-refresh-mature-fork-diagnostic-v1-attempt-002-analysis-recovery-v2\readiness.json `
  --authorization out\target-refresh-mature-fork-diagnostic-v1-attempt-002-analysis-recovery-v2\authorization.json
```

The preflight must find the entire v2 output namespace absent and return
`ready_for_product_authorization`. A new explicit product decision must name
the resulting readiness identity and the exact 288-game/3.5-hour zero-training
scope. Neither the v1 authorization nor a passing preflight authorizes launch.

If separately authorized, record that decision and invoke `--launch once`
with the same explicit plan, readiness and authorization paths plus a unique
run ID. The exclusive launch marker consumes the grant before analysis starts.
Any mismatch is terminal. There is no automatic retry, recovery, extension,
held-out evaluation, promotion, publication or long-training continuation.
