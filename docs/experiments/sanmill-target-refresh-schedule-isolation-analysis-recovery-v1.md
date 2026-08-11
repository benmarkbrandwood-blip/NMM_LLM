# Target-refresh schedule-isolation analysis recovery v1

The machine-readable contract is
[`sanmill-target-refresh-schedule-isolation-analysis-recovery-v1.json`](sanmill-target-refresh-schedule-isolation-analysis-recovery-v1.json).
Its immutable plan identity is
`5525c54ea58e860a82a78eebbc6541de6e42eaf77da405f966b4eb724590aa0c`.

## Purpose

The original schedule-isolation v2 launch completed all training work and then
failed before the first development game because its result reader rejected
uniform Windows CRLF JSONL. This plan does not repeat, resume, or extend that
training. It permits only the missing analysis over the already completed
checkpoints.

The owning failure evidence is
[`target-refresh-schedule-isolation-diagnostic-v2-attempt-001-failure-2026-08-11.md`](../evidence/target-refresh-schedule-isolation-diagnostic-v2-attempt-001-failure-2026-08-11.md).

## Frozen scope

- Device: CPU.
- Candidate checkpoints: the six completed attempt-001 arms only.
- Feature analysis: the already frozen 64-position phase corpus at the four
  transition boundaries.
- Outcome analysis: exactly 288 candidate-blind development games using the
  already frozen replay grid, colors, common random numbers, game-50 anchors,
  fixed Sanmill node work, and 120-logical-ply ceiling.
- Training games: zero.
- Optimizer updates: zero.
- Database writes: zero.
- Checkpoint writes: zero.
- Maximum additional active time: 5.5 hours. Together with the measured 0.4824
  training hours this remains below the original six-hour family ceiling.

The analysis may publish only the development outcome ledger, the result, and
the local recovery control evidence. It is not held-out evaluation and cannot
authorize retry, extension, promotion, model publication, or long training.

## Source separation

The checkpoints were trained at commit
`49defb8a79d07a19c035ca6c1f23f266ae5ed2b2`. The recovery controller requires
the current `dev` commit to be clean, published, and descended from that source.
Every intervening tracked path must be on the reporter's exact analysis-only
allowlist. The result records both the contract-frozen publisher and the
executed publisher hash.

The minimum recovery implementation is commit
`d88503426e7423bed9503b3f859188f28ebce739`. The plan pins the publisher and
controller byte identities, all parent contract/readiness/authorization/launch
and failure hashes, and the completed game/transition/resource accounting.

## One-shot workflow

1. Run `--preflight`. It revalidates every parent artifact, the completed
   training resource audit, paired temperature/node schedules, clean published
   analysis source, implementation hashes, and absent result targets.
2. Bind the product owner's explicit one-shot decision to the resulting
   readiness identity in a local authorization file.
3. Run `--launch once` with that identity and a unique run ID. The launch marker
   is exclusive and consumes the recovery authorization before the first game.
4. On any error, publish a fail-closed recovery failure. No retry is permitted.

Current state: `designed_unlaunched_needs_authorization`. Creating this plan or
running preflight does not itself start analysis.

