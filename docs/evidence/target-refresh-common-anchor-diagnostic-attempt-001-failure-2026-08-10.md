# Target-refresh common-anchor diagnostic attempt 001 failure

## Status and claim boundary

The first authorised execution of
`sanmill-target-refresh-common-anchor-diagnostic-v1` stopped fail closed in
its first arm on 10 August 2026. The remaining three arms were not started.
The four-arm sequence is aborted, and none of its four authorisation files may
be used to continue or restart it.

This is implementation-failure evidence only. The 50 simulated training games
are not a completed arm, do not enter a target-refresh comparison, and provide
no held-out, strength, promotion, publication, or long-training evidence.

## Frozen execution identities

- NMM_LLM source: `05c4cbf48c8364686c1717d3f3d2a3558e6f8f5f`
- contract plan identity:
  `8e3982338f517472a73603b6b34080622433dedbc8d6ed3afdc5ce0c969609a4`
- readiness identity:
  `d6ed98beebb01d9c3482b9dcc7547656dace88df8afe5ad3a8550f7b86c24547`
- readiness file SHA-256:
  `6b4c7aff0318e17ff308201b01feffe84aa3fe5bf4fa96b2fe8333c245fd11de`
- first-arm plan identity:
  `4d59eeb1dd1d76d0b7f493216c32c7520d7e4f518d11b150c35ad985585724d4`
- first-arm plan file SHA-256:
  `d19626e22840e472de17bac6b8824abf414bee1b8936850b0968f0de84e31644`
- first-arm authorisation file SHA-256:
  `3ac8c6faff43011d47474cb2bd991143c0f21eb2e4ab76763151e80d5a575130`
- first-arm controller ledger SHA-256:
  `83ddb550138e416cac49330875faeb34f9c1cb4be405578ff8c04e82ecd00c10`

The exact ignored evidence remains under
`out/target-refresh-common-anchor-diagnostic-v1`. No file from that directory
is a reusable plan or training input.

## Observed facts

The frozen launch order began with `seed64-refresh`. Its long-run preflight
returned `ready_for_long_run`, and the trainer started fresh with the expected
Sanmill, MIF, ruleset, HumanDB masking, Malom and empty
`sector-corrected-v1` SpecialistDB identities. Sentinel, ValueNet, GapNet,
imitation, PPO, recovery, opening forcing and the policy auxiliary were
explicitly disabled.

The arm reached game 50 and completed the required 18 A2C optimiser updates.
Its `update_log.jsonl` contains 18 rows, from game 3 through game 49, and has
SHA-256
`387821581801265b388f41b92081eb5d69c01385feb1c5fd4cbecd8a574a32bc`.
The trainer then attempted to save the fixed game-50 development measurement
anchor and raised:

```text
CheckpointFormatError: unsupported checkpoint role:
'development_measurement_anchor'
```

The segment event ledger records `preflight_passed`, `training_started`, then
`training_failed` with exception type `CheckpointFormatError`. It has SHA-256
`6887748b767fb076f452b17fa8c5c0266f8dab7475ba9891270df76601e9335a`.
The run manifest has SHA-256
`fcd0145b11213ad64e8b38795440ad0a96935bf9b2ce5774417793d312dbd65c`.

No measurement anchor, measurement candidate, `latest.pt`, accepted segment,
completed game ledger, policy-health result, or experiment result was
published. Managed status therefore reports zero accepted games and
`stopped_for_agent_review`.

The first-arm SpecialistDB remains internally valid but is now contaminated
by the failed attempt and must never seed another fresh arm. Its main-file
SHA-256 is
`0ccae5e158f15ff298802b1dbb7e914608493007cc758a42d4669c03bb8457ed`;
`quick_check=ok`; it contains 2,903 positions, 481 trusted Malom labels,
32 winning lines, no preferred plays, and lineage root
`managed-target-refresh-anchor-v1-s64-refresh-segment-0001`. No WAL or SHM
sidecar exists.

The other three arms have no segment directory and their SpecialistDB files
remain byte-identical to the empty template, SHA-256
`5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d`.
Although their individual managed status still says `ready_to_run`, the
four-arm decision note and frozen stop rule require the entire sequence to
stop after any arm anomaly. Their authorisation files are retained as consumed
sequence evidence, not standing permission:

| Arm | Authorisation SHA-256 |
| --- | --- |
| `seed64-no-refresh` | `a010f094f5a5695a79e10fe428d58792f8c155ed81e552e8aca1eb736d97a091` |
| `seed65-refresh` | `10f1acbd3d7643b9c664c96167de91924ddc9d697cd27d96677584d490578e56` |
| `seed65-no-refresh` | `da8878b3c989704f412371d34dc582f26fec12e4fa2bc2a2b1c8bd5747cbd024` |

## Root cause and correction evidence

The trainer deliberately emits two non-lineage evidence snapshots:
`development_measurement_anchor` at game 50 and
`development_measurement_candidate` at each later measurement update. The
result analyser requires those exact roles. The version-2 checkpoint envelope
still accepted only `latest`, `best_train`, `candidate`, and `accepted`, so the
first new role failed before any anchor bytes were written. Reusing a generic
role would erase the evidence distinction and is not acceptable.

A focused regression reproduced both missing roles as two failures while an
unknown-role control still passed. Correction commit
`e02aca46364280674fb564ba68a536fce45292c7` extends only the closed role
vocabulary and adds round-trip persistence tests for both roles. It does not
change gameplay, rollout, optimiser, resume, database, Sanmill, or measurement
semantics.

Post-correction verification reports:

- three role-focused tests passed;
- 103 checkpoint, measurement, manager and preflight tests passed;
- 103 mandatory Malom, DB-teacher and provenance tests passed, with 498
  parameterised subtests; and
- Ruff passes the changed production module and changed test scope after
  excluding the test file's pre-existing unrelated `F841` baseline.

## Required successor boundary

Attempt 001 must not be resumed or retried. A successor requires all of the
following before another product decision:

1. ordinary publication of the correction commit;
2. a new attempt identity, four new plan IDs and four new control directories;
3. four new arm-specific SpecialistDB files copied from the pristine template;
4. fresh preflights from a clean `dev == origin/dev` source;
5. verification that none of the attempt-001 authorisations, segment files,
   database bytes, optimiser state, or game observations are inputs; and
6. a new explicit product authorisation against the successor readiness
   identity.

No automatic retry, held-out evaluation, promotion, publication, or long
training follows from this failure or its correction.
