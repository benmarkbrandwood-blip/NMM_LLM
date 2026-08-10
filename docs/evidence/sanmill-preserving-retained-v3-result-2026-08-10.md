# Sanmill-Preserving Retained v3 Completion Evidence — 10 August 2026

## Result

The authorized `managed-sanmill-preserving-retained-v3-seed58` plan reached
its bounded terminal condition:

- controller state: `completed`;
- completed games: 5,000 of 5,000;
- accepted process segments: 20 of 20;
- recorded active segment time: 6,238.321384 seconds, or 1.732867 hours,
  below the 12-hour limit;
- final segment-completion event: sequence 47,
  `338420a0380e453b858bf49fb15f44bf7b7cec579c27e278b774258d358f345d`;
  and
- final plan-completion event: sequence 48,
  `41e48ecde45a4a1cd52eeaa05d580e3c0d241f08b73c05f3211680387c679f1e`.

All 20 mandatory fixed-state policy-health gates passed before their segment
became an accepted resume parent. No controller lock remains. The result is a
retained research baseline, not playing-strength, promotion or publication
evidence.

## Frozen authority and implementation lineage

| Item | Identity |
| --- | --- |
| Plan ID | `managed-sanmill-preserving-retained-v3-seed58` |
| Plan semantic SHA-256 | `e7fccb1ad174e59b4ccdd71b34b11f180e77b23c985f50716f2acabfeb88ed65` |
| Plan file SHA-256 | `9712b5edb4b1ea0ab94c9da721018bf71f6d648e6e09a799abff50c055cc361d` |
| Authorization file SHA-256 | `b19420464d08574c17dd64d16e251e666a810ae8e4404fe464f8adf8722752cd` |
| Plan source commit | `3f400135c833f49de6700f1ab9c246cca9ab25f7` |
| Final runtime commit | `af0334567a40cb8d15cb50b1fc2fa4c628d303c1` |
| Experiment ID | `dev-v4-sanmill-preserving-retained-v3-seed58` |
| Experiment digest | `sha256:adb0fc6dd34721d2ad1441173baec27c1723c343aa82553ce91bbeab791b369c` |
| Resume-config SHA-256 | `51a774bd78c23a2fe1a3cdfdff192d42cedf6f4007a87ab02d392e6c114748b4` |
| Training ruleset semantic digest | `sha256:52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a` |
| Sanmill source commit | `a6623f88959f7453594df274fbe1f128af7ff55e` |
| Sanmill runtime identity | `5d436ac3eff3d7a7f186a4a7fb1c656739bafc93baeb5bb4e5b1dbf905dbaf04` |
| MIF release | `mif-suite-1.0` at `a0a0f21cff5d6fbde045cd1482e416b92e0dc45a` |
| MIF Suite JCS digest | `sha256:81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f` |

The immutable plan retained its original semantic identity. The run admitted
two published, focused implementation repairs after fail-closed preflight
stops:

1. segment 2 ran from commit `268785f1dc6506dd48a485502c53b29c8d266396`
   after the trainer was made to close the writable SpecialistDB at process
   shutdown; and
2. segments 3 through 20 ran from commit
   `af0334567a40cb8d15cb50b1fc2fa4c628d303c1` after immutable-mode policy
   audits were prevented from creating SQLite sidecars.

The two repair records are
[`sanmill-preserving-retained-v3-segment-boundary-repair-2026-08-10.json`](sanmill-preserving-retained-v3-segment-boundary-repair-2026-08-10.json),
SHA-256
`ecc9c4a14179f6107923091bec06952df1dfa7c0a0cdbd3207ecb0afa842f94c`,
and
[`sanmill-preserving-retained-v3-policy-audit-sidecar-repair-2026-08-10.json`](sanmill-preserving-retained-v3-policy-audit-sidecar-repair-2026-08-10.json),
SHA-256
`fca520489f857825775eb7142086d333c1398bb7d54e7cd8da97b6e01b17e9da`.

Both failed attempts stopped before producing a new segment log, checkpoint or
counted game. The active 20 segment logs therefore contain the complete final
lineage without a quarantined game prefix:

- exactly 5,000 rows;
- game numbers exactly 1 through 5,000;
- no missing or duplicate game number; and
- 5,000 unique `game_id` values.

The controller ledger SHA-256 is
`7ae247e9e84cdbc8669019a7df6ba07cd4079c242c6046f586fab0501d57e288`.

## Observed facts

### Training outcomes

The trainer records learner wins as `1.5`, losses as `-1.0`, rules draws as
`-0.15`, and 120-ply truncations as `-0.25`. The score rate below uses ordinary
match scoring, one point for a win and half a point for a draw; it is not the
training reward.

| Opponent / learner colour | Games | Wins | Draws | Losses | Score rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 5,000 | 36 | 2,514 | 2,450 | 25.86% |
| Frozen target, all | 3,053 | 11 | 1,532 | 1,510 | 25.45% |
| Frozen target, learner White | 1,535 | 6 | 785 | 744 | 25.96% |
| Frozen target, learner Black | 1,518 | 5 | 747 | 766 | 24.93% |
| Sanmill search, all | 1,947 | 25 | 982 | 940 | 26.50% |
| Sanmill search, learner White | 971 | 9 | 486 | 476 | 25.95% |
| Sanmill search, learner Black | 976 | 16 | 496 | 464 | 27.05% |

For the final 200 chronological games, the aggregate was 0 wins, 199 draws
and 1 loss, a 49.75% score. Frozen-target games were 0/122/0 and Sanmill games
were 0/77/1. The near-50% score is therefore almost entirely draw mass and is
not evidence of reliable win conversion.

Sanmill-search outcomes by curriculum resource level were:

| Level | Node ceiling | Games | Wins | Draws | Losses | Score rate | Mean completed depth |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,000 | 196 | 0 | 3 | 193 | 0.77% | 3.08 |
| 2 | 5,000 | 184 | 0 | 0 | 184 | 0.00% | 4.29 |
| 3 | 25,000 | 193 | 0 | 0 | 193 | 0.00% | 5.64 |
| 4 | 100,000 | 370 | 1 | 15 | 354 | 2.30% | 7.10 |
| 5 | 500,000 | 1,004 | 24 | 964 | 16 | 50.40% | 12.59 |

These levels are sequential training stages, not randomized opponents against
one frozen learner. Their rows are confounded with model age, target age,
temperature, SpecialistDB contents and learning updates. They cannot establish
that a larger node ceiling is easier, or that the final candidate has beaten
any earlier level.

Termination reasons were:

| Termination reason | All | Frozen target | Sanmill search |
| --- | ---: | ---: | ---: |
| Loss by fewer than three pieces | 1,391 | 1,300 | 91 |
| Loss by no legal moves | 1,095 | 221 | 874 |
| Threefold-repetition draw | 1,728 | 1,424 | 304 |
| Fifty-move draw | 225 | 0 | 225 |
| 120-ply truncation | 561 | 108 | 453 |

The 561 truncations are not rules-based draws and remain disaggregated.

### Curves, updates and policy-health classes

Ordinary A2C training has no supervised validation split or validation-loss
curve. This run produced training losses only. It used one preregistered seed,
58, so it supplies no multi-seed variance estimate. All 1,612 update records
were present, and every recorded policy loss, value loss, entropy, reward,
probability, learning rate and temperature value was finite.

The raw 200-game snapshots show the transition without presenting a fitted
forecast:

| End game | W / D / L | Score | Mean entropy | Mean chosen probability | Mean Malom-preserving rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0 / 1 / 199 | 0.25% | 2.290 | 0.170 | 0.920 |
| 1,000 | 0 / 0 / 200 | 0.00% | 2.310 | 0.160 | 0.910 |
| 1,500 | 0 / 0 / 200 | 0.00% | 2.310 | 0.160 | 0.920 |
| 2,000 | 0 / 0 / 200 | 0.00% | 2.290 | 0.160 | 0.920 |
| 2,500 | 1 / 47 / 152 | 12.25% | 1.790 | 0.290 | 0.950 |
| 3,000 | 1 / 198 / 1 | 50.00% | 1.170 | 0.520 | 1.000 |
| 3,500 | 1 / 196 / 3 | 49.50% | 0.740 | 0.680 | 1.000 |
| 4,000 | 1 / 198 / 1 | 50.00% | 0.470 | 0.800 | 1.000 |
| 4,500 | 3 / 197 / 0 | 50.75% | 0.390 | 0.830 | 1.000 |
| 5,000 | 0 / 199 / 1 | 49.75% | 0.380 | 0.830 | 1.000 |

All 20 fixed-corpus gates passed. Candidate value preservation was at least
0.758621 and the preserving-minus-downgrading logit margin was always
positive. At the final boundary, direct and candidate preservation were both
1.0 and the mean margin was +4.799480. The final policy-health report SHA-256
is `d976fcbe3707887c6d50f6afa3248ef8755c9a363a93f752f4b07b6ffc0f4635`.

Final candidate metrics by phase on the fixed development corpus were:

| Phase | States | Critical states | Preserving argmax | Mean logit margin | Scheduled entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Placement | 22 | 15 | 1.0 | +4.599357 | 0.562793 |
| Movement | 21 | 8 | 1.0 | +5.404615 | 0.280446 |
| Flying | 21 | 6 | 1.0 | +4.492941 | 0.535961 |

The 29 critical positions are inspected development data and were checked at
every boundary. These metrics reject a catastrophic value-direction collapse
on that corpus; they are not held-out validation or playing-strength evidence.

### Final checkpoint and SpecialistDB

`checkpoint_tool.py verify` accepted:

- checkpoint
  `learned_ai/checkpoints/scaffolded/s_gen_v2_sanmill_refereed/managed-sanmill-preserving-retained-v3-seed58/segments/segment-0020/latest.pt`;
- checkpoint ID
  `managed-sanmill-preserving-retained-v3-seed58-segment-0020:checkpoint:00000006`;
- payload SHA-256
  `1c14955e2c7ca69824c5369f7501713788c4d8650f63dd4d6cf4992fab037ac8`;
  and
- file SHA-256
  `28e8af274f4fc9dd7e00ce4f7be884c855354218c796888f1c1ab81a4cdc9fa7`.

The final checkpoint records game count 5,000, update count 1,612, target age
50 and temperature 0.20. Its SpecialistDB identity equals the closed database
file SHA-256
`82d7fbcd897be2493ee40b40a44aa7cd941c95ff538b4f9bf21e2977cd4a8abe`.
No WAL, SHM or rollback-journal sidecar exists. Immutable read-only SQLite
verification reported `quick_check=ok` and:

- 164,431 positions;
- 10,871 positions with at least three empirical samples, the trainer's
  feature threshold;
- 4,577 positions with at least five empirical samples;
- 35,197 trusted `sector-corrected-v1` Malom labels;
- 2,666 positions with both a trusted Malom label and at least three empirical
  samples;
- 2,272 winning lines and zero promoted preferred plays;
- empirical totals of 39,975 wins, 189,170 draws and 39,975 losses; and
- lineage root
  `managed-sanmill-preserving-retained-v3-seed58-segment-0001`.

## Hypotheses

1. The large late draw mass may reflect useful loss avoidance, but may also
   reflect an over-decisive policy that has learned preservation and repetition
   more readily than win conversion.
2. The cumulative SpecialistDB may influence that transition because one
   unstratified empirical distribution pools games from both opponent sources,
   all five node stages and every earlier checkpoint.
3. Temperature reaching 0.20, frozen-target feedback, and the stronger
   late-stage Sanmill search are competing explanations. A SpecialistDB effect
   must not be inferred from the curve alone.

## Supporting evidence

- The last 200 games contain 199 draws but no wins.
- Mean chosen probability rose from about 0.16 near the early plateaus to 0.83
  at the end while mean entropy fell to 0.38.
- The final fixed-state policy strongly preserves Malom value in placement,
  movement and flying positions.
- The SpecialistDB contains 10,871 empirically queryable positions and 2,666
  positions where empirical evidence can replace a trusted theoretical prior
  in the current legacy projection.

## Counterevidence and confounders

- The late curve coincides with the 500,000-node stage, a much older learner,
  lower temperature and a changed frozen target; it is not a controlled
  SpecialistDB comparison.
- A draw against the learner's current frozen target is not an independent
  strength result.
- The run has one seed and no validation curve.
- The fixed 29-state gate was used during development and cannot test
  generalization or tactical trap creation.
- The prior retained-v2 run, reward ablations and auxiliary calibrations used
  different seeds, reward semantics, database lineages or resource envelopes.
  They provide context, not a clean baseline comparison for this candidate.

## Next validation experiments

The smallest next experiment is a no-update, read-only mechanism audit on the
fixed development corpus. It must hold the checkpoint, target, histories,
Malom, HumanDB and encoder route fixed while projecting the final SpecialistDB
as full evidence, trusted Malom-only evidence, empirical-only evidence and
disabled evidence. It must record per-action database coverage, empirical
sample counts, theoretical/empirical disagreement and policy changes without
writing a checkpoint or database.

Only if that audit crosses a preregistered material-sensitivity threshold
should a three-seed, single-factor training calibration be frozen. No new
training, held-out evaluation, promotion or publication is authorized by this
completion record.

## Claim boundary

This evidence establishes bounded completion, contiguous lineage accounting,
checkpoint/database integrity, finite updates, recovery traceability and
survival of the fixed anti-collapse gate. It does not establish playing
strength, superiority to either opponent, successful win conversion, a causal
SpecialistDB effect, promotion, publication readiness or MIF full conformance.
