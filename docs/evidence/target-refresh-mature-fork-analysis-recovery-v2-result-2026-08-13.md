# Mature target-refresh analysis recovery v2 result — 13 August 2026

Status: `completed_once_no_selection`
Training-readiness verdict: `needs_decision`

The isolated recovery-v2 analysis completed exactly once. It loaded the six
already completed mature target-refresh branches read-only, compared all 12
candidate checkpoints on the frozen 64-state policy corpus, and completed the
frozen 288-game direct cross-play grid. It performed no training, optimizer,
database, or checkpoint write.

The predeclared classifier returned `no_material_direct_effect`. Neither
`refresh-mature` nor `stale-control` is selected for retained or long training.
This is development mechanism evidence, not held-out strength, promotion, or
publication evidence.

## Evidence identities

| Item | Identity or SHA-256 |
| --- | --- |
| Analysis source | `77e5d7c74be17fb7c512c00d5b023e4ee089e530` |
| Recovery plan identity | `32158846cb3e3903589663465d6217ed546442eee617e0aa5fe94defe45feb25` |
| Readiness identity | `13e25cd5fb0552dd171b3f736e42841c1222bb627f08447abbb8a77bb7031fab` |
| Readiness file SHA-256 | `91e6266de5acbc1584e4389bd2b9d9b270b27ebdbbdb216345aae06b1b6c7732` |
| Authorization identity | `c02f2ffdc0e630dbffb7b4c9e30e67c3ac1735bd424c381a5d8ff53c2f4c1dc5` |
| Authorization file SHA-256 | `0bcf7430cc8428e4e6f6f0f3115062ab83460a678ccb67133ee65acb3d945a5e` |
| Launch identity | `988c6bd142c2e3605ff84e08eb86c5ec51afbcb294629b81dcd61c85e3163667` |
| Launch file SHA-256 | `b410d222c69f6547b6e9f1b1954c9f05b843bdfd979d67b903922942f7145d60` |
| Completion identity | `38363fd3efcccc6609e2039bc142a6b6db61449682de488a35c2b316e7c2a1a9` |
| Completion file SHA-256 | `dd301a5ca19430a792cb039cfff7bf5df888be4cd96af44d8c0a19a49539eaf4` |
| Result identity | `5e7bb7bf0505d1f3a2b43f50572e1ed9de8861114a25193490e34533f4dafd61` |
| Result file SHA-256 | `f03c4244260bf719ddc269a4ba1dc8ae0779023c7cadc91f54e0e0035c08bd61` |
| Ledger rows / SHA-256 | `288` / `f0d354179f0934c9b259ca23f8daaeb4efa80506eb8a3651a708d8c71abf25b7` |
| Candidate audit identity | `d3c7e0dd5b611a9bec4086355035ed462f3fd11fbc006d49781b1f47b35340e0` |
| Completed-artifact identity | `c186b7b02e2b012e6e899fe48429a388def6f94a640118faac273fb3a60fe49d` |
| Training-audit identity | `23a37e39ba2ff33733d2964d5cdf3bf74893b2d5990d2735fe6704fc82239e47` |

The run ID is
`mature-target-refresh-analysis-recovery-v2-20260812T225430Z`. The ignored
control directory contains the exact readiness, authorization, launch,
ledger, result, completion, stdout, and stderr bytes. It contains no failure
record.

## Gate review

| Gate | Observed | Expected | Result |
| --- | --- | --- | --- |
| Published source | `dev == origin/dev == 77e5d7c`, tracked clean at launch | Exact clean published analysis source | Pass |
| Recovery scope | 288 CPU no-update games in 0.047169 active hours | At most 288 games and 3.5 hours | Pass |
| Mutation boundary | 0 training games, optimizer updates, database writes, and checkpoint writes | All zero | Pass |
| Candidate inputs | All 12 checkpoints revalidated after completion | Six seed/condition pairs at 4,096 and 8,192 transitions | Pass |
| Ledger | 288 strict canonical rows; schedule and pair identities recomputed | 144 complete colour-swapped pairs | Pass |
| Read-only data | HumanDB main/WAL/SHM and Malom observations equal before and after | No drift | Pass |
| Direct-effect gate | Aggregate `-0.076389`; one stale-supporting seed | Magnitude at least `0.083333` and two supporting seeds | No selection |
| Policy-persistence gate | Only seed 67 retained a material trigger at both boundaries | At least two persistent seeds | No selection |

Before launch, 38 focused recovery, report, diagnostic, result, and sequence
tests passed. The mandatory Malom, DB-teacher, and provenance gate separately
passed 103 tests plus 498 parameterized subtests. After completion, the result
and every control identity were recomputed, the direct summary and policy
classifier were independently rerun from the stored artefacts, and all 12
candidate checkpoints were loaded and audited again.

The reviewed launch command was:

```powershell
.\.venv\Scripts\python.exe `
  scripts\run_target_refresh_mature_fork_analysis_recovery.py `
  --launch once `
  --plan docs\experiments\sanmill-target-refresh-mature-fork-analysis-recovery-v2.json `
  --readiness out\target-refresh-mature-fork-diagnostic-v1-attempt-002-analysis-recovery-v2\readiness.json `
  --authorization out\target-refresh-mature-fork-diagnostic-v1-attempt-002-analysis-recovery-v2\authorization.json `
  --expected-readiness-identity 13e25cd5fb0552dd171b3f736e42841c1222bb627f08447abbb8a77bb7031fab `
  --run-id mature-target-refresh-analysis-recovery-v2-20260812T225430Z
```

## Observed facts / 观察事实

### Frozen training evidence

The earlier six-arm attempt remains the only training source. Its six branches
used seeds 67, 68, and 69, with paired `refresh-mature` and `stale-control`
conditions. Each branch consumed exactly 8,192 post-mature-fork transitions as
128 batches of 64. The total was 2,529 new games, 49,152 transitions, 768 A2C
updates, and 0.403 managed active hours. Paired temperature and learning-rate
exposure was byte-equal, Sanmill work stayed fixed at 1,000 nodes, and every
policy-health gate passed.

The last complete 50-game windows are reported below. They occur at different
game indices because the experiment equalized transitions rather than game
counts, so they are descriptive online-training curves, not a common baseline
or supervised validation result.

| Seed | Condition | Window | Score | Mean entropy | Malom-preserving rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 67 | refresh mature | 901–950 | 0.02 | 2.1858 | 0.9136 |
| 67 | stale control | 701–750 | 0.34 | 1.9617 | 0.8888 |
| 68 | refresh mature | 751–800 | 0.07 | 2.0872 | 0.9385 |
| 68 | stale control | 501–550 | 0.32 | 2.0462 | 0.9291 |
| 69 | refresh mature | 951–1,000 | 0.00 | 2.2036 | 0.9237 |
| 69 | stale control | 901–950 | 0.11 | 2.1345 | 0.9381 |

Those training scores use different frozen opponents by treatment and cannot
decide the direct effect. No supervised validation split or validation-loss
curve exists; the frozen policy corpus and paired direct games are the named
development measures.

### Policy-distribution evidence

At temperature 0.2, the 64-state full-action comparisons were:

| Seed | Transitions | Mean JS | Mean TV | Top-1 disagreement | Max phase absolute Malom-mass delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 67 | 4,096 | 0.02904 | 0.13071 | 1.56% | 0.12651 |
| 67 | 8,192 | 0.02434 | 0.12594 | 3.12% | 0.02959 |
| 68 | 4,096 | 0.00544 | 0.04271 | 0.00% | 0.02939 |
| 68 | 8,192 | 0.00369 | 0.05724 | 0.00% | 0.00004 |
| 69 | 4,096 | 0.00056 | 0.02256 | 0.00% | 0.00502 |
| 69 | 8,192 | 0.00225 | 0.04476 | 3.12% | 0.00122 |

Seed 67 had persistent material JS and TV triggers. Seed 68 crossed the JS
threshold at 4,096 and the TV threshold at 8,192, but retained no common
trigger across both boundaries. Seed 69 crossed none. The frozen two-seed
persistence rule therefore failed, with policy decision identity
`38710fb582542adbef68ebe557e8121f349e24dc14c2af4e1d09833cddf5f94f`.

### Direct cross-play evidence

The refresh-mature candidate scored `89 W / 88 D / 111 L`, or `0.461806`,
over all 288 games. The primary paired contrast was frozen as
`refresh-mature minus stale-control`:

| Stratum | Paired mean score effect |
| --- | ---: |
| Aggregate | -0.076389 |
| Seed 67 | -0.229167 |
| Seed 68 | +0.010417 |
| Seed 69 | -0.010417 |
| Placement | -0.093750 |
| Movement | -0.062500 |
| Flying | -0.072917 |

Of 144 pairs, 126 were equal, 17 favoured stale control, and one favoured
refresh mature. The aggregate effect fell just short of the predeclared
`-0.083333` material threshold, and only seed 67 crossed the per-seed stale
threshold. Two supporting seeds were required. The direct-summary identity is
`94215bd3ac2fcac3510d7c61e82c5be14590d710296e35ab7cb69878b8a4d57c`.

The refresh candidate showed a large colour asymmetry: score `0.618056` as
Black and `0.305556` as White. Colour swapping protects the paired primary
contrast, but the raw score must not be interpreted as general strength.

Termination counts were 198 `lose_fewer_than_three`, two
`lose_no_legal_moves`, 55 threefold-repetition draws, five fifty-move draws,
and 28 max-ply truncations. The truncation rate was `0.097222`, below the
frozen `0.25` invalidation threshold. Truncations remain development draws,
not rules draws.

## Hypotheses / 假设

1. A single mature target refresh may harm the seed-67 trajectory, but that
   effect is seed-dependent rather than a repeatable treatment effect.
2. The negative aggregate effect may be real but smaller than the frozen
   material threshold; the current three-seed sample cannot distinguish that
   from one outlying seed.
3. Colour sensitivity and the finite development corpus may obscure a smaller
   direct effect, although colour-swapped pairing removes first-order colour
   imbalance from the primary contrast.

## Supporting evidence / 支持证据

- Seed 67 has both persistent policy separation and a large negative paired
  outcome effect.
- Placement, movement, and flying phase contrasts all point toward stale
  control.
- The aggregate `-0.076389` effect is close to the frozen `-0.083333` gate.
- Same-seed arms share one mature fork, exact update exposure, common random
  streams, fixed temperature/LR schedules, and fixed Sanmill work.

## Counterevidence / 反证

- Seeds 68 and 69 have near-zero outcome effects of opposite signs.
- Only one of three seeds has persistent material policy divergence.
- The required two-seed support rule fails, so selecting stale control after
  observing seed 67 would move the decision boundary post hoc.
- Training W/D/L compares treatment-specific frozen targets and is not a
  common validation baseline.
- The 64-state corpus and 288 games are development evidence already used in
  this research chain, not held-out strength evidence.
- The large colour asymmetry and 28 development truncations limit broader
  interpretation even though the preregistered paired safety gate passes.

## Next validation experiments / 下一步验证实验

No automatic successor is justified. Preserve both conditions and stop this
sequence. If resolving mature refresh cadence remains a product objective, the
smallest scientifically useful successor is a separately frozen replication
with new independent seeds under the same common-fork, equal-transition,
fixed-schedule contract and the same candidate-vs-candidate paired outcome
measure. Seed count, aggregate resources, acceptance threshold, and the role
of any larger replay corpus must be fixed before observing new outcomes.

Adding more games only to these three already observed seeds could refine
their within-seed estimates but would not establish cross-seed persistence.
Any new training, development extension, held-out evaluation, promotion,
publication, or long run remains outside this consumed authorization and must
be covered by one new aggregate product decision rather than repeated
per-arm approvals.
