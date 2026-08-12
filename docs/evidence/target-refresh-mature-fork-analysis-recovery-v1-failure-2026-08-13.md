# Mature target-refresh analysis recovery v1 failure — 13 August 2026

## Decision and claim boundary

Status: `failed_closed_before_development_games`

The product owner authorized exactly one analysis-only recovery against
readiness identity
`3e6a9db846cac65bd8d1989a67935a2fea1460424936006729b91ac67aa09e63`.
The grant allowed 288 CPU no-update development games and at most 3.5 active
hours. It allowed no training, optimizer update, database or checkpoint write,
retry, resume, extension, held-out evaluation, promotion, publication, or
long-training launch. The one-shot launch consumed that grant.

The process failed closed before its first development game. This record is
diagnostic and lineage evidence only. It is not a result for the target-refresh
hypothesis and does not authorize a retry or successor launch.

## Frozen identities

| Item | Identity |
| --- | --- |
| Analysis source at launch | `d7bf147d477ac09d50de3da124340746ad7ae8b2` |
| Recovery plan | `70fb522b863ceb583b393697a11894540ce3ab5c5764b6aa8e892ebb7cc451e6` |
| Readiness file SHA-256 | `15c9459b747113ca3ebd016c86ebd4f879b1474ff9778367bfc816feab5baa53` |
| Authorization identity | `cec9dccacefa280d5b07688da1b0de6b86eb8ad5cd21f491f4b2aff8ca557fbf` |
| Authorization file SHA-256 | `a2e32b519e2eb2192379e48290c868013b34cabf9e3c87751d99f974509e4742` |
| Launch identity | `32195dff61da940a6dc1074e788aff3b31394cda13cd8a930dbb46e2baad79ee` |
| Launch file SHA-256 | `f5078bd67316fb8dfc85d7a734c465af756c8b0a2f90948d86154af587f27604` |
| Failure identity | `c4e73033a2d5a8aee41a6af49be0475f9aa2d28c6b7dd4330585c305468d3201` |
| Failure file SHA-256 | `8c8ca699909ed5093fda1ddbc334a6aa38608d5d41a6018b2f4ffc410660d27b` |
| Reporter stderr SHA-256 | `c33aaf8ace33f242b06de69f7349e7b3ef02dcd467a5ac26ca61723e5c70fe62` |
| Corrective implementation | `763f20ca18deeb9de4b7b92e4949b206f0073357` |

The ignored control directory contains the exact readiness, authorization,
launch, failure, stdout and stderr bytes. It contains no development ledger,
result or completion record.

## Observed facts / 观察事实

- The reporter exited with code 2 after about `0.010135` active hours.
- Its terminal diagnostic was
  `candidate semantics differ: seed 67 refresh-mature 4096`.
- The failure occurred while loading the first candidate checkpoint, before
  the direct-game loop. No development game was played.
- The process performed zero training games and zero optimizer updates. The
  recovery route contains no database or checkpoint writer.
- All six completed training arms, their 49,152 post-fork transitions and
  their existing checkpoint bytes remained unchanged.
- The HumanDB warning only confirms the intended masking of its unversioned
  historical Malom columns. It is not the failure cause.

## Hypothesis / 假设

The reporter compared an ordinary post-fork training checkpoint with its
initial branch envelope using an invalid implementation-identity invariant.
It removed `target_refresh_branch_*` one-shot metadata but incorrectly required
`mature_target_refresh_*` capture metadata to remain in every later ordinary
checkpoint.

## Supporting evidence / 支持证据

- The initial branch envelope deliberately carries both one-shot metadata
  families. Its trainer recovery state and source-checkpoint chain preserve the
  durable fork semantics after loading.
- Every inspected transition checkpoint uses the stable managed-run
  implementation mapping and omits both one-shot metadata families.
- The stable implementation fields, experiment identity, configuration,
  treatment state, transition origin, consumed-transition count, pending-step
  bound and source-checkpoint path all remain available for strict validation.
- After correction, a read-only audit loaded all 12 candidate checkpoints at
  the 4,096 and 8,192 boundaries across seeds 67, 68 and 69. All passed with
  audit identity
  `d3c7e0dd5b611a9bec4086355035ed462f3fd11fbc006d49781b1f47b35340e0`.
- Focused reporter and recovery tests reported `21 passed`; Ruff and
  `git diff --check` passed for the changed scope.

## Counterevidence / 反证

- This failure does not show checkpoint corruption, model divergence, changed
  rules/data identities, or a training defect.
- Passing the 12-checkpoint semantic audit does not prove that the later policy
  comparison or 288-game measurement will complete.
- No target-refresh result exists, so the failure supplies no evidence for
  either refresh condition.

## Next validation experiment / 下一步验证实验

Freeze a successor analysis-only recovery with a new plan identity, isolated
control/output paths and a new one-shot authorization. Its preflight must run
the full 12-checkpoint semantic audit before authorization. The scientific
measurement, seeds, checkpoints, 288-game schedule, resource ceiling, stop
rules and claim boundary must remain unchanged. Any successor launch requires
a fresh explicit product authorization and remains prohibited from automatic
retry, held-out evaluation, promotion, publication or long training.

