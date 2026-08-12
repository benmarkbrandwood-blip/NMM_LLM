# Mature target-refresh attempt-002 failure evidence

## Verdict

All six bounded training arms completed once and passed their policy-health
gates, but the parent sequence failed closed before the first CPU development
game. The failure is in the result publisher's input-framing policy, not a
demonstrated trainer, checkpoint, database, referee, or numerical failure.
The original launch and all child authorizations are consumed and must not be
retried or resumed.

## Observed facts

- Training source: `e7ff859923c067251c3feb08fe7aee6fff59116d`.
- Parent plan identity:
  `442c170177b5a8b867b14db31e62b16219fc3ee65ae1fac804842e493c35089d`.
- Readiness identity:
  `d2860ae09980d6eb9aa87d0f81d0a93b634f3f7725220b7392ee8c6eca7282b7`.
- Authorization identity:
  `181a8e883d70102e516c2319ea0329372e27d6b087b2bc7c86aad3570ec3e375`.
- Run ID: `mature-target-refresh-attempt-002-20260812T045901Z`.
- Launch identity:
  `34dd77f553fcf4ec82ba1365e95f74cdb9520a6d23ea4139cd5d5c71d111d749`.
- Failure identity:
  `d4e13fbaaef4df8ea6002e4c783208f017887226b86dfb4c7a8e13816995620e`.
- Failure record SHA-256:
  `f40336ddc4a84125467def28532543e4b8ed823e4b13dcddcb2bb95898761c45`.
- The sequence completed 2,529 new training games, 49,152 optimizer-consumed
  transitions, and 768 A2C updates in about 0.403 active hours.
- Every arm reached exactly 8,192 post-mature-fork transitions as 128 batches
  of 64, produced the required checkpoints, retained finite update logs, and
  passed the controller's policy-health gate.
- No development ledger, result, or completion record exists. Therefore zero
  of the planned 288 CPU no-update development games ran.
- The parent failure record explicitly sets
  `retry_or_recovery_authorized=false`.

| Order | Seed | Condition | Source games | Final games | New games | Transitions | Updates | Active hours |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 67 | refresh-mature | 439 | 952 | 513 | 8,192 | 128 | 0.0726 |
| 2 | 67 | stale-control | 439 | 768 | 329 | 8,192 | 128 | 0.0673 |
| 3 | 68 | refresh-mature | 327 | 802 | 475 | 8,192 | 128 | 0.0630 |
| 4 | 68 | stale-control | 327 | 598 | 271 | 8,192 | 128 | 0.0629 |
| 5 | 69 | refresh-mature | 518 | 1,022 | 504 | 8,192 | 128 | 0.0707 |
| 6 | 69 | stale-control | 518 | 955 | 437 | 8,192 | 128 | 0.0665 |

The six final checkpoint SHA-256 values, in the table order above, are:

1. `978b5eec51fac7c5ec0f00dbc17031ade5d4c21e0d9547ddf52a281f77feae30`
2. `7fa009fa45a27bcafdaa881c73199f6fbaa9a55e155665547a4686104e049a72`
3. `8a5a88d8a4b0cf8fe69600bc90c6a341b08fcbb21d6842f74e58f94172430c04`
4. `c0c845da4682744ba0010d3836cb8aedfac73e91678dd1e480f42dab23e807c7`
5. `820b21b241f638fdaa54bb9862dffe67e87fea835ec0a602a9b5c52aa9d4575c`
6. `d7f58cfba629df1e4e079cdd8b0a8b8105c42fdcf4da4f71e23f31c8b3dcf776`

## Failure mechanism

The publisher first verified the frozen raw SHA-256 of the policy corpus and
then incorrectly required the same file to use canonical minified JSON
framing. The frozen corpus is valid, LF-terminated, pretty-printed JSON. Its
raw bytes are already the contract identity and its contents have a dedicated
semantic validator. The two other frozen reference inputs use the same
intentional presentation format, so changing only the first call would merely
move the failure to the replay corpus and then its Sanmill audit.

The affected immutable reference files are:

| Role | Path | Frozen SHA-256 |
|---|---|---|
| Policy corpus | `docs/experiments/dev-v4-phase-covered-corpus-v1.json` | `cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e` |
| Replay corpus | `docs/experiments/dev-v4-phase-replay-development-corpus-v1.json` | `9637efaae21074eefb4fab9e22550f5729999b30d03ed469dc88cf75aae07c2f` |
| Replay audit | `docs/evidence/phase-replay-development-corpus-sanmill-audit-2026-08-11.json` | `4634ba61a4e43c0b6d80a80c882aea5ca985b9bc8923e7895b39bf8ad557e42e` |

## Hypothesis

For immutable reference inputs, exact raw SHA-256 plus the existing semantic
validator is the correct fail-closed boundary. Canonical minified framing
should remain mandatory for controller-generated authority and result files,
whose byte identity is owned by the current pipeline.

## Supporting evidence

- All three reference inputs match the exact hashes frozen in the parent
  contract.
- All three parse as JSON objects, contain no carriage returns, and end with
  LF. They differ from canonical serialization only because they are
  deliberately pretty printed.
- The failure occurred before candidate models were loaded for result
  analysis and before any direct-crossplay game was scheduled.
- The six training arms independently passed their checkpoint and
  policy-health acceptance paths before the publisher was entered.

## Counterevidence and remaining gap

- The 288 development outcomes do not exist, so this evidence cannot decide
  whether a mature target refresh helps, harms, or has no material effect.
- The final pairwise policy distributions have not yet been produced by the
  corrected publisher.
- Passing training health does not prove playing-strength improvement.
- The consumed authorization forbids automatic recovery even though no
  analysis game ran.

## Next validation experiment

First repair the publisher with focused tests that preserve strict canonical
handling for generated authority evidence while accepting only exact-hash,
semantically valid reference inputs. Then freeze a separate analysis-only
plan over the six existing checkpoints: zero training games, zero optimizer
updates, zero database or checkpoint writes, and exactly 288 CPU development
games in a new output namespace. That new one-shot plan requires its own
readiness identity and explicit launch authority. It cannot retry, run
held-out evaluation, promote or publish a model, or start long training.
