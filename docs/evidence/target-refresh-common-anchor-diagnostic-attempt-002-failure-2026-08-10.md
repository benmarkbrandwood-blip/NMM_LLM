# Target-refresh common-anchor diagnostic attempt 002 failure

## Status and claim boundary

The second authorised execution of the common-anchor target-refresh
diagnostic stopped fail closed on 10 August 2026 after the two seed-64 arms
completed. The required result analyser then rejected their legitimate
optimizer-bounded completion counts. The seed-65 arms were not started.

The four-arm sequence is aborted. All four authorisation files are consumed
sequence evidence and may not be used to start, continue or restart any arm.
No experiment result was published.

This is infrastructure-failure evidence only. It is not target-refresh,
held-out, strength, promotion, publication or long-training evidence.

## Frozen execution identities

- NMM_LLM training source:
  `6d4f12879df4f503996d5afa980476e636391f17`
- Sanmill training runtime source:
  `a6623f88959f7453594df274fbe1f128af7ff55e`
- MIF Suite 1.0 release source:
  `a0a0f21cff5d6fbde045cd1482e416b92e0dc45a`
- contract plan identity:
  `7b22663bca04c8bc380a7b7688cfbc0ae714c74a62884b98fc307f02c22cb36b`
- contract file SHA-256:
  `20c193d14f019b234d6d805839dbf45e38df40840baf0b0da815e1d5c7690f47`
- readiness identity:
  `bcbb625d6d903ce8257a550a2489cf302bbb0fae50558f999c1a62b08a53165c`
- readiness file SHA-256:
  `f0cd2f43a500d062b62dc02f0198661007cf83a57e6701d8d80b6b0de2ca5b34`
- NMM training rules semantic digest:
  `sha256:52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a`
- portable Sanmill referee semantic digest:
  `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94`

The exact ignored runtime evidence remains under
`out/target-refresh-common-anchor-diagnostic-v1-attempt-002`.

| Arm | Plan SHA-256 | Authorisation SHA-256 |
| --- | --- | --- |
| `seed64-refresh` | `605055cb179b20c320a5b43e7cb59f8b7f94b1239bfe17a5a8fd48820fee4bea` | `c9930f97a6b3b88968ab85eecf6b691c6ee202b1ead50abbb66f6f85e6b50267` |
| `seed64-no-refresh` | `0946e5b560f415564916605bd86d24c38346df87873fcbd9c3b344268c2187e3` | `51c93862aa6f503c3dd96dac8569a16fd904a58ad6fe317e14ae44fc3e2edb0b` |
| `seed65-refresh` | `21050211d809928bac4b59771de6eab13e94e12308b59d3dd230506b1be69460` | `b44c4ac12b8db092beb0d635313ac95f422d5142ad7d86c1bd235dbf4d6521cb` |
| `seed65-no-refresh` | `7915d7743036ca228fe9bdf38cc004212995ea2c285a7be4ce07ac70d144d680` | `75741b463d6dec3d8f80da8310b49c586f0ab01d39527008ee71062da4ff8fff` |

## Observed facts

The authorised envelope allowed at most 600 training games, 256 no-update
measurement games and two active hours. Before the analyser anomaly, the two
completed arms used 214 training games, 128 no-update measurement games and
275.4760069 seconds of trainer time in total. Neither arm requested a retry or
resume.

| Evidence | `seed64-refresh` | `seed64-no-refresh` |
| --- | ---: | ---: |
| accepted training games | 122 | 92 |
| optimiser updates | 34 | 34 |
| post-anchor optimiser updates | 16 | 16 |
| no-update measurement games | 64 | 64 |
| measurement checkpoints | 4 | 4 |
| policy-health result | passed | passed |
| trainer seconds | 137.8306488 | 137.6453581 |

The controller ledgers end with `managed_plan_completed`. Their SHA-256
values are respectively
`dd6b5443868c4f13d6a93b1b48f191ce83fa42d27da590b453d6bde45f23b156`
and
`0eff9c936955d654152967602ee0a88460ba7f97e1b12e288f20a6e19672c9fd`.

The completed policy-health reports passed at their actual accepted game
counts. The refresh arm reported a candidate preserving rate of `1.0`, direct
preserving rate of `1.0`, and candidate logit margin
`0.0002730156506756963`; its report SHA-256 is
`1211a27cd6731d10d68e237c9571c55954a610828bb988f2a2dc9ca2129ca7d6`.
The no-refresh arm reported the same two preserving rates and margin
`0.0002830934516106058`; its report SHA-256 is
`183e69d50db217ac97d51ed5ee3911f01b115ddebaaef558816566056d115cc9`.

The seed-64 arms have byte-identical canonical training rows through game 50
and identical game-50 anchor model tensors. The anchor model-state SHA-256 is
`94aed99fee3d94b9f27a645df56a333bc939797e396ee2a882f5b02f7a45b4cd`.
This proves the intended same-seed pairing boundary, but one seed cannot
support the preregistered two-seed decision.

The principal completed artefact hashes are:

| Artefact | `seed64-refresh` | `seed64-no-refresh` |
| --- | --- | --- |
| training ledger | `67f26195a6d1e9cb22607b65c5370db8f138ad4e06f43077a1b05b3bbab3d0d8` | `aa4f5ae9d428e459c92aab99acf46297f9a35172cafae6d073695f2f979e27ec` |
| update ledger | `f1f96fa0efef38bf43ac20668c2388d886303ad0286ae13707da1f1d564b57ab` | `cba77afd4da2bf52c13c45d35bbd3ebb19c4fd691514c74d34990dd1eff426b8` |
| measurement ledger | `ced59dce4c2cc918069264ba462741ddf3733f3fdba9a84f6b91bbfad3420f00` | `5ccd6c3391bab90f680582060cc0b445325ddc0bbe76da37a7104bf5811da72f` |
| final checkpoint | `617e6e559d563b880b06528aff234e4ee7f943d6f6400cfc5c990832510d4cf2` | `17c225fd6664431247c8aec04489be9bed9f719431549abe64a5e606f775fe91` |

Both completed writable SpecialistDB files pass `quick_check`, retain
`sector-corrected-v1`, and have distinct lineage roots. The refresh database
has SHA-256
`fff1340dbf5a2d35ae2a0f7f4e072270a46806f5f3ede3fa228bfda64f34f41f`,
4,962 positions, 1,177 trusted Malom labels, 33 winning lines and no
preferred plays. The no-refresh database has SHA-256
`21a4d63e6bf7130e86257715bc4e22b5dcc7a0d0a5b1bac0cb2bdb7311e071`,
5,177 positions, 880 trusted labels, 54 winning lines and no preferred plays.

The two seed-65 controls contain plan, preflight, authorisation and controller
records only; neither has a segment directory. Their database main files
remain byte-identical to the pristine empty template, SHA-256
`5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d`.
They are nevertheless consumed attempt-002 evidence and must not seed a later
attempt.

An explicit post-stop SQLite `quick_check` audit opened all four databases in
read-only mode but caused SQLite to create a zero-byte `-wal` and a 32,768-byte
`-shm` sidecar beside each main file. The sidecars were not deleted and the
main-file hashes above did not change. Preserve them as audit side effects;
do not reinterpret them as training data or remove them implicitly.

## Failure and root cause

The frozen result analyser first validates that an arm reached its exact
optimiser-update bound and that the accepted game count is above the game-50
anchor and no higher than the 150-game safety ceiling. It then reused the
fixed-game mill-bonus policy-health helper. That helper unconditionally
revalidated policy health at `plan.game_bound`, or 150 games, even though an
optimizer-bounded arm legitimately stops as soon as its required update count
is reached.

Consequently the first completed arm was rejected with:

```text
ManagedContractError: policy-health report game count differs
```

The actual reports were internally consistent at games 122 and 92. This was
therefore an analysis-contract defect, not evidence of a bad checkpoint,
failed policy-health gate, training divergence, corrupt database or pairing
failure.

Correction commit
`873e1265fc98636cefc7a561e3d139f8fce621e5` allows the shared helper to
accept an explicitly validated completion count, retains the fixed-game
default, rejects invalid or over-ceiling counts, and passes the controller's
accepted count from the common-anchor analyser. It changes result validation
only; gameplay, rollout, optimiser, measurement and database semantics are
unchanged.

A deterministic regression failed before the correction with an unexpected
`completed_games` keyword and passed afterwards. Post-correction verification
reports:

- 118 checkpoint, measurement, manager, preflight and result-analysis tests
  passed;
- 103 mandatory Malom, DB-teacher and provenance tests passed, with 498
  parameterised subtests;
- Ruff passed all changed production and test files; and
- read-only analysis of both completed seed-64 arms accepted their actual
  game counts, policy-health reports and common anchor.

## Interpretation discipline

### Observed facts

Both seed-64 arms completed the frozen optimiser exposure and measurement
work, passed policy health, and shared the required pre-intervention history
and model anchor. The result pipeline then failed for a deterministic count
validation reason, and seed 65 did not run.

### Hypothesis

The experiment hypothesis remains that target refresh caused the earlier
endogenous-opponent contrast rather than a changing measurement denominator
or unequal optimiser exposure.

### Supporting evidence

The completed seed-64 pair proves that the new common-anchor and
optimizer-matched execution design can preserve its pairing invariant.

### Counterevidence and missing evidence

Only one of two required seeds exists. The frozen decision rule requires both
seeds and all four arms, so neither the seed-64 contrast nor any raw training
curve may select refresh or no-refresh. No result identity exists.

### Next validation experiment

Attempt 002 must not be resumed. A successor may be considered only after the
correction and this evidence are published, and it requires a new contract
identity, plan IDs, control directories, four pristine database copies,
preflights, readiness identity and explicit product authorisation. No
automatic retry, held-out evaluation, promotion, publication or long training
is authorised by this record.

Readiness verdict for attempt 002: `fatal_stop`.
