# Target-refresh direct cross-play attempt-001 failure

Status: `failed_closed_before_first_game`

## Frozen launch identity

- source commit:
  `b25fe338c037f88098ee0f3cea504f8727e40567`;
- plan identity:
  `c7a03214c8efdc7d55f36dab658da8fc57e71fe5b99038806753e7e851fc4e05`;
- readiness identity:
  `6da0b4ac40b699e2fced1d58be65b82b512611dabe7ee9ae4ca8c228201faf9d`;
- readiness file SHA-256:
  `48028d292fc84eb2b978d2d5f824059aa9a6b7c59461055d52d964616b5c6bd9`;
- authorization identity:
  `5485baf75f5e81f559b2d2d2d2e933a6973c35be50bdab3be184eccd4c90d76b`;
- authorization file SHA-256:
  `6ec4422b05881aa2d78e1ca8443a08414e2df6ae5241eb1bf72a0d8fbed433ec`;
- launch identity:
  `b2042f77239cfcd8124ac1bbd59579d6529fd4e5d14bcdb0d7f103fe119646dc`;
- launch file SHA-256:
  `b89ea3612576556d1ca3e40326c3fe36b46caa135234939ae72a693bd6c754d7`;
- failure identity:
  `403e4f637af48626e0693904a4a482e60558758343b6118110ed5dcc39f7c5e4`;
- failure file SHA-256:
  `3272027e25898cd7902a75397f994e084a6254dd1b0c9ace82cff526cdb1c099`.

The one-shot launch began under run ID
`target-refresh-direct-crossplay-v1-2026-08-12-attempt-001`. Its authorization
is consumed. It does not authorize a retry or recovery.

## Observed failure

The process failed after approximately 0.79 seconds while constructing the
per-colour policy random generators for schedule ordinal zero. The frozen
schedule correctly stores the fields `policy_seed_white` and
`policy_seed_black`. The game runner incorrectly derived and requested the
abbreviated keys `policy_seed_w` and `policy_seed_b`; the first lookup raised
`KeyError: 'policy_seed_w'`.

This occurred before `SanmillTrainingGame` was entered, before the first replay
history was loaded into a referee process, and before either policy selected
an action. The HumanDB warning only restated the frozen
`masked_historical_labels` policy and did not cause the failure.

## Resource and data boundary

- the game ledger exists as the canonical empty file, SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- zero direct cross-play games and zero logical plies were recorded;
- no result or completion record exists;
- zero training games, optimizer updates, database writes and checkpoint
  writes occurred;
- no Python or Sanmill referee process remained after the failure; and
- a complete readiness rebuild after failure remained byte-equivalent to the
  stored readiness, including checkpoint records, referee identity and
  HumanDB/Malom read-only observations.

This is infrastructure-failure evidence only. It contains no policy outcome,
strength, target-refresh, phase, colour or termination evidence.

## Required successor boundary

The runner must map `W` and `B` explicitly to the closed schedule fields
`policy_seed_white` and `policy_seed_black`, with a focused regression that
uses a schedule produced by the frozen builder. The consumed output directory
must remain immutable. Any attempt-002 requires a new plan identity, isolated
output paths, clean published source, new readiness identity and a new explicit
product authorization. It must retain the same scientific design and may not
be presented as a resume or automatic retry.
