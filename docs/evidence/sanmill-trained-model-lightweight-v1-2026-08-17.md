# Lightweight trained-model measurement against Malom-safe random

Date: 2026-08-17

Decision: `mixed_arm_results`; every frozen primary contrast is
`inconclusive_precision`

## Executive result

The fresh lightweight measurement completed its one planned execution.  It
first reproduced the attempt-002 random-safe known answer exactly, including
all 508 action sequences, and only then loaded either trained route.  It then
completed all 2,032 candidate games.  Every one of the 2,540 games reached a
strict rules terminal; no safety-cap record entered the score.

The observed result is mixed, not "all trained models below the zero-training
baseline":

| Arm | W / D / L | Strict score | Minus random-safe | 95% interval | Half-width |
| --- | --- | ---: | ---: | ---: | ---: |
| random-safe reproduction | 21 / 414 / 73 | 44.8819% | -- | -- | -- |
| retained-v4 free | 128 / 310 / 70 | 55.7087% | +10.8268 pp | [+8.6029, +13.0506] pp | 2.2238 pp |
| retained-v4 `A_pos` | 128 / 310 / 70 | 55.7087% | +10.8268 pp | [+8.6029, +13.0506] pp | 2.2238 pp |
| active specialists free | 68 / 160 / 280 | 29.1339% | -15.7480 pp | [-19.2870, -12.2090] pp | 3.5390 pp |
| active specialists `A_pos` | 97 / 346 / 65 | 53.1496% | +8.2677 pp | [+6.3717, +10.1638] pp | 1.8961 pp |

The independent unit is one start after averaging the two candidate colors.
Each interval is the frozen start-clustered normal 95-percent engineering
interval over 254 starts.  All four observed intervals exclude zero, but the
frozen decision function first requires a half-width no greater than 1.5
percentage points.  None met that precision target.  The only permitted
primary decision for each arm is therefore `inconclusive_precision`.

This precision result must not be softened into a passed directional gate.
Conversely, it must not be paraphrased as all trained routes losing: three
point estimates were above random-safe and one was below it.  These are
internal engineering intervals on the reused frozen start set, not population
inference.

## Frozen identities

- plan:
  `cd6461a0a867c8d6b3e6220ba55521df2318d57914ee1b4252d8a56eef33cbee`;
- authorization:
  `d076d9e1d502064433df6c9285ba11364029e4eba40921bac2d9211debdafce8`;
- result:
  `548b3cac544e81f52cdbe15f980dddd61b4963fe1378f01dba5e878ffc248f83`;
- result-file SHA-256:
  `ea42014fa5997480c91d9d69bb54f31cd945deaccb9440442e50cc95abcfc703`;
- start pool:
  `385a376dd82953c23c232f34e3dd5a84e5887b978c60627657eccfa6821eb6e9`;
- formal 254-start membership:
  `610f62e74b4a70500adfcaa3e0c19769dd178b480ef9765115ae6ab9a5af13d2`;
- Sanmill runtime:
  `705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea`;
- Malom content:
  `c414fe88778f8d1d95cd3015532b43cad59f09e8398d8e46c42188b6829f3544`.

The measurement source commit was
`a8285caf2095358125bbec32f0015124ed372979`, with source tree
`f3201cb66276696bb1041edce4ae1e667a2dfdac`.  The result identity was
independently recomputed after execution and matched.

## Exact known-answer reproduction

The reproduction used the original attempt-002 random-safe seed and schedule
for the same 254 eligible starts and both colors.  Candidate model loading and
candidate games were unavailable until this gate passed.

| Check | Expected | Observed |
| --- | ---: | ---: |
| games | 508 | 508 |
| wins | 21 | 21 |
| draws | 414 | 414 |
| losses | 73 | 73 |
| strict score | 44.8819% | 44.8819% |
| threefold draws | 109 | 109 |
| fifty-move draws | 305 | 305 |
| per-game mismatches | 0 | 0 |

The compared per-game surface was stronger than the aggregate requirement:
game/start/color identity, W/D/L, terminal reason, logical plies, final
history, rule counters, and the complete action sequence all matched.  The
expected and observed fingerprint was
`08d87b1293f408029385bbd08960dc9e5539645f97bccb744f32585031ea9a36`.

The following comparability identities also matched attempt-002 before the
known-answer execution:

- Sanmill commit `a6623f88959f7453594df274fbe1f128af7ff55e`;
- Sanmill binary SHA-256
  `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`;
- strict referee digest
  `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94`;
- fixed Sanmill work of 100,000 nodes; and
- the start-pool and formal-membership identities above.

This exact reproduction is the end-to-end evidence for the strict game loop,
rules clocks, repetition history, colors, start handling, `A_pos`, and score
encoding.  No baseline-v1 registry, coverage ledger, rehearsal pass, or old
authorization was used as acceptance evidence.

## Candidate loading and route identity

Retained-v4 was loaded only through its full frozen `TrainingAlignedPolicy`
route, never through the product `GeneralistAgent`.  The bundle identity was
`817d2e36fbd0b614c5c48737ee987f684b99eb6ff697591618123ec7307a2d0f`;
the checkpoint SHA-256 was
`295b268e697255908f9c7517f4697ca251a10ec0f13d922cbcbab2260fb5105d`;
and the frozen SpecialistDB SHA-256 was
`3d69d1acb007dbd26a48ae1c6acec4bb29f905ffedd21c816ad1771a6cf942ed`.
Its label version was `sector-corrected-v1` and the feature width was 134.

The active-specialist runtime identity was
`2199c5b8e98881885b8d2c2377d062e658d4d05cfc3c3459b90725fc2bb3137f`.
Its open/mid/end checkpoint SHA-256 values were, respectively,
`d020e1442676e16cdced6c91dac958817c3a22a283cc293d6e19930a87703701`,
`a587ab995224a1d43c99fd2f42e4bff9c060ac6da55edcddb43a39fc07ef26d2`,
and `5de51a1afd5794374d4394cce2950957a23f02504b5c5952a062d91414b94be8`.

Three sampled placement/movement/flying states were scored twice by both
routes.  All six repeated score vectors and selected moves were bit-exact.
The retained-v4 scorer was the same frozen harness scorer used by the prior
v3/v4 evaluation.

The specialist measurement did not run the product's 30/60-second alpha-beta
presearch.  Static inspection found that the successful specialist score path
does not read `SpecialistRouter._gameai`, so no successful-argmax bias was
identified.  Failure behavior still differs: this measurement fails closed,
while the product silently falls back to alpha-beta.  The measured route is
therefore not claimed to reproduce the entire product failure surface.

## Why retained-v4 free and constrained are identical

The retained-v4 free and `A_pos` arms were identical in all 508 paired games:
every action sequence, strict outcome, terminal reason, and final history
matched.  This is not explained by an absence of unsafe alternatives.

Across the 13,222 retained-v4 free candidate turns, 9,605 turns had an
`A_pos` set that was a proper subset of all legal moves.  The unrestricted
argmax nevertheless selected an `A_pos` move every time, producing zero
positional downgrade events.  Adding the final `A_pos` restriction therefore
changed none of the observed actions.

The narrow supported statement is that this complete frozen route naturally
selected position-preserving moves on this exact start set and Sanmill
runtime.  It is not a universal safety guarantee.  It cannot be attributed to
the checkpoint alone because the route's lookahead has Malom terminal early
exits and a frozen target network.  It also remains only positional safety:
the Malom query does not carry repetition or no-progress history.

## Positional downgrade diagnostics

| Arm | Candidate turns | W to D | W to L | D to L | All events | Event rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| retained-v4 free | 13,222 | 0 | 0 | 0 | 0 | 0.0000% |
| retained-v4 `A_pos` | 13,222 | 0 | 0 | 0 | 0 | 0.0000% |
| active specialists free | 9,360 | 192 | 209 | 331 | 732 | 7.8205% |
| active specialists `A_pos` | 14,853 | 0 | 0 | 0 | 0 | 0.0000% |

For the free specialist arm, the 732 events occurred in the actual action
phases as follows: 159 in placement, 192 in movement, and 381 in flying.  The
`A_pos` restriction raised its score by 24.0157 points, with a descriptive
paired interval of [+20.9619, +27.0696] points.

This supports the mechanism hypothesis that positional self-downgrades are a
major weakness of the free specialist route.  It does not identify the causal
effect of any single downgrade: constraining an early choice changes every
later state and opponent reply.

## Comparison with attempt-002 full guidance

The two constrained trained arms remove the action-safety asymmetry and were
also compared, as frozen secondary endpoints, with attempt-002 full guidance
at 44.2913 percent:

| Constrained arm | Minus full guidance | 95% interval |
| --- | ---: | ---: |
| retained-v4 `A_pos` | +11.4173 pp | [+9.2217, +13.6129] pp |
| active specialists `A_pos` | +8.8583 pp | [+7.0646, +10.6520] pp |

These are secondary engineering contrasts on the same reused starts.  They
cannot flip the frozen primary precision decision and are not a promotion or
population claim.

## Source-phase diagnostics

These phase results are descriptive and have no separate multiplicity-adjusted
decision.

| Source phase | Arm | Games | W / D / L | Score |
| --- | --- | ---: | --- | ---: |
| placement | random-safe | 168 | 1 / 153 / 14 | 46.1310% |
| placement | retained-v4 free / `A_pos` | 168 each | 30 / 124 / 14 | 54.7619% |
| placement | active specialists free | 168 | 6 / 7 / 155 | 5.6548% |
| placement | active specialists `A_pos` | 168 | 18 / 137 / 13 | 51.4881% |
| movement | random-safe | 170 | 4 / 132 / 34 | 41.1765% |
| movement | retained-v4 free / `A_pos` | 170 each | 27 / 111 / 32 | 48.5294% |
| movement | active specialists free | 170 | 17 / 72 / 81 | 31.1765% |
| movement | active specialists `A_pos` | 170 | 23 / 115 / 32 | 47.3529% |
| flying | random-safe | 170 | 16 / 129 / 25 | 47.3529% |
| flying | retained-v4 free / `A_pos` | 170 each | 71 / 75 / 24 | 63.8235% |
| flying | active specialists free | 170 | 45 / 81 / 44 | 50.2941% |
| flying | active specialists `A_pos` | 170 | 56 / 94 / 20 | 60.5882% |

The retained-v4 point advantage is largest in the flying-source stratum.  The
free specialist route's aggregate loss is dominated by placement starts.
Because the source phases were deliberately near-balanced, these figures are
not weighted to a product traffic distribution.

## Strict terminal diagnostics

All games reached a strict terminal and none reached the 1,536-ply safety cap.

| Arm | Threefold | Fifty-move | Fewer-than-three | No-legal-moves |
| --- | ---: | ---: | ---: | ---: |
| random-safe | 109 | 305 | 47 | 47 |
| retained-v4 free | 201 | 109 | 152 | 46 |
| retained-v4 `A_pos` | 201 | 109 | 152 | 46 |
| active specialists free | 75 | 85 | 126 | 222 |
| active specialists `A_pos` | 168 | 178 | 108 | 54 |

Retained-v4 shifted draws strongly from the fifty-move rule toward threefold
repetition relative to random-safe.  The specialist constraint both increased
draws and sharply reduced no-legal-move terminals.  These are process
differences under the exact protocol, not evidence of history-aware safety.

## Precision shortfall

The planned 254 starts did not achieve the frozen 1.5-point half-width.  Using
the observed start-level standard deviations only as a post-run planning
diagnostic, a fixed-width calculation would require approximately 559 starts
for either retained-v4 arm, 406 for constrained specialists, and 1,414 for
free specialists.  These values are not a new frozen plan and no additional
games are authorized by this report.

## Resource, ledger, and access audit

The one execution used:

- 2,540 complete games: 508 reproduction and 2,032 candidate games;
- 12,418.08 of 21,600 active seconds;
- 72,184 Sanmill single-step searches;
- 2,777,047 Malom read-only queries;
- zero training updates, model fits, or database writes.

The raw reproduction ledger contains 508 hash-chained records.  Its file
SHA-256 is
`5c75349096c26296673c2fef105283a3aede8171e976a3ae383e33f864272a4a`
and its chain tail is
`bd026ecace9691c4708fd7751b9be0e0083cad1cf606dbceef59d9f06bce4aaa`.
The raw candidate ledger contains 2,032 hash-chained records.  Its file
SHA-256 is
`4d5292bc8748832f01a79541cd0babef007c750b8af92900f33bc98ba83825c2`
and its chain tail is
`df4a9fd2b10517edb3c00a12a9e21a16ff953b2e95733ecfab1c8faf060dce67`.
Both chains and every canonical row hash were independently recomputed after
completion.  Compact per-game evidence is also embedded in the tracked result
manifest.

Before/after byte size, mtime, and SHA-256 were unchanged for HumanDB and the
retained-v4 SpecialistDB snapshot.  No journal was created.  The pre-existing
HumanDB WAL/SHM state was unchanged.  Access counters were zero for official
selection, confirmation, final-test, research-confirmation, and source pool
`2eb04f54`.  No checkpoint or alias changed.

## Implementation changes

This was a fresh lightweight harness, not a baseline-v1 repair or recovery.
The following files were added before any outcome was observed:

- `learned_ai/evaluation/sanmill_trained_model_lightweight.py` for frozen-plan
  validation, reproduction scheduling, the exact gate, and read-only audits;
- `scripts/freeze_sanmill_trained_model_lightweight.py` for the immutable plan;
- `scripts/run_sanmill_trained_model_lightweight.py` for the one execution,
  candidate loading, durable ledgers, analysis, and sealing;
- `tests/test_sanmill_trained_model_lightweight.py` for focused negative and
  positive contracts;
- the frozen plan and its separate authorization record.

The harness reused the already tested strict game loop and candidate scorers.
It did not change gameplay, models, checkpoints, databases, aliases, or any
old frozen record.  The implementation commit was `ebea0b5`; authorization was
committed separately as `a8285ca`.

## Evidence assessment and next validation

Observed facts are the exact reproduction, strict terminal records, score
contrasts, positional transition labels, and route identities above.

The strongest supported mechanism hypothesis is that retained-v4's complete
frozen route already ranks position-preserving moves first on this corpus,
whereas the free specialist route often ranks a positional downgrade first.
The 9,605 proper-subset opportunities and the specialist's 732 observed
downgrades support that explanation.

Counterevidence and limits are material:

- retained-v4's route itself consults Malom during lookahead, so its result is
  not evidence for checkpoint-only safety;
- `A_pos` ignores repetition and no-progress history;
- the start pool is reused development evidence, not a new held-out or
  population sample;
- the active-specialist lineage remains untraceable beyond its frozen payloads;
- every primary contrast missed the preregistered precision target; and
- no comparison supports another engine, runtime, product UI, or human user.

If another internal validation is ever authorized, the shortest discriminating
checks are a genuinely new candidate-blind start pool large enough for the
chosen precision target and an ablation that separates retained-v4 weights
from its Malom-enabled lookahead.  This report does not authorize either.

## Claim boundary

This is an internal directional measurement on one exact fixed Sanmill
runtime, one reused frozen start set, and positional-only safety where
constrained.  It is not an `A_allow` result, human-trap result, equivalence
claim, public claim, promotion decision, deployment decision, release
decision, model-replacement authorization, or training authorization.
