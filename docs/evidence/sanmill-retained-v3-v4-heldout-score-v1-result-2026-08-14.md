# Retained-v3/v4 high-precision held-out score result

Date: 2026-08-14

Status: `completed_v4_higher_named_route_fixed_heldout_score`

The preregistered 253-start / 1,012-game fixed-width evaluation completed once.
Its primary decision is `v4_higher_fixed_heldout_score`: retained-v4 scored
`+1.6798pp` above retained-v3 on this frozen held-out prefix, with a 95%
start-clustered engineering interval of `[+0.6195pp, +2.7402pp]`. The
`1.0604pp` half-width passed the frozen `1.5pp` precision gate.

This is a directional relation between two named routes on one fixed corpus.
It is not an equivalence, population, Elo, universal playing-strength,
target-refresh causal, promotion, publication or release result.

## Identity and completion chain

| Artifact | Identity |
| --- | --- |
| Plan | `6620821e879f53058d15990cd0e8c884ae62fec213b3d96200e8894c20e19714` |
| Stable source readiness | `f233c991aa66a8699fac8952fd0c758a5fabb09de7a0e66ba3043635934b2b08` |
| Product authorization | `816cc390b6850b02cf3cb36afdc78daf154a4d4e982d6798fb633376ea6f2503` |
| Launch readiness | `765c0829e272cf53f3fc6178cd8f83621c8de2fc3653be298df6ff70683b2ef4` |
| Runtime specification | `cb736759790989ee6a4bd6bd1f6f965e4341c67729da6249e2d91d585b4fa943` |
| Launch | `b4505be8f06c217a3e57994b3ae6ddb30957ac0b8bd56a7a6ead49f1f1c6a4a2` |
| Result | `8d7a4a0aefdd9b0716cccfa3a8d9ace44493c870cc9c9eed885bc7fd35c74730` |
| Completion | `8949a8fdc0c38772b40e348d7e645ec70aed3cd76663d426320154b0e708ac7c` |

The canonical ledger contains exactly 1,012 records. Its SHA-256 is
`7dd72447a4e918e5ee816604f18b5e3497a002a2969d1ca22119043a8df8fc45`,
which equals the completion binding. A fresh `recompute` from that ledger was
identity-equal to the stored report. All 1,012 games reached strict rules
terminals, no game hit the 1,536-post-start safety cap, no `failure.json`
exists, and evaluator-active time was 1,749.805795 seconds (29.16 minutes).

Two launch-preparation checks failed closed before the counted run. The first
rejected a non-canonical authorization-file newline. A later elevated
`Start-Process` context could not read the deliberately read-only input
snapshots. Both stopped before a runtime spec, launch or game existed and did
not consume the grant. The one canonical launch above ran in the verified
ordinary workspace context and completed without retry, resume, semantic
recovery, expansion, training or update.

## Preregistered primary result

| Measure | Result |
| --- | ---: |
| Independent starts | 253 / 253 |
| Matched colour units | 506 / 506 |
| Mean score difference, v4 minus v3 | +1.6798pp |
| 95% engineering interval | [+0.6195pp, +2.7402pp] |
| Half-width | 1.0604pp |
| Frozen maximum half-width | 1.5000pp |
| Decision | `v4_higher_fixed_heldout_score` |

The start-level distribution was 24 starts favouring v4, seven favouring v3,
and 222 tied after averaging candidate White and Black. The colour-unit
estimate has the same mean but is not the independent-unit primary.

## Strict terminal results

| Candidate | Games | Wins | Draws | Losses | Score rate | Cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| retained-v3 refresh-50 | 506 | 62 | 292 | 152 | 41.1067% | 0 |
| retained-v4 no-refresh | 506 | 80 | 273 | 153 | 42.7866% | 0 |

V4's aggregate advantage came from 18 additional wins while losses were
essentially unchanged (153 versus 152), not from converting losses into
long-running incompletes.

The following strata are descriptive because no separate directional
interval was preregistered for them:

| Candidate role | v3 score | v4 score | Difference |
| --- | ---: | ---: | ---: |
| Candidate White, 253 games each | 35.9684% | 38.3399% | +2.3715pp |
| Candidate Black, 253 games each | 46.2451% | 47.2332% | +0.9881pp |

| Frozen start phase | Games per candidate | v3 score | v4 score | Difference |
| --- | ---: | ---: | ---: | ---: |
| placement | 198 | 39.1414% | 41.4141% | +2.2727pp |
| movement | 196 | 40.3061% | 40.8163% | +0.5102pp |
| flying | 112 | 45.9821% | 48.6607% | +2.6786pp |

## Secondary process evidence

The start-clustered 108-post-start-ply survival difference was `+0.5929pp`,
interval `[-1.7668pp, +2.9526pp]`. It is inconclusive and does not reproduce
the earlier reused-development `+7.8125pp` survival direction. Mean post-start
length was 38.5079 plies for v3 and 38.2036 for v4; the preregistered
restricted-length difference was `-0.0198pp`, interval `[-0.2043pp,
+0.1647pp]` on its normalized scale. Neither process endpoint distinguishes
the routes.

Malom query coverage was 100% for both routes. Every queryable selected move
preserved the current coarse W/D/L: 9,708 / 9,708 v3 candidate turns and
9,634 / 9,634 v4 candidate turns. This common ceiling does not explain the
score difference. Malom remains history-free and is not a strict result
adjudicator.

## Observed facts / 观察事实

The fixed held-out primary passed both its direction and precision gates in
favour of the named v4 route. The eventual W/D/L, start-level direction count,
colour strata and phase strata are all arithmetically consistent with that
primary. Every result was strict-terminal and the stored report reproduces
exactly from the identity-bound ledger.

## Hypotheses / 假设

1. The named v4 route converts more of these frozen starts into wins than the
   named v3 route while leaving its loss exposure nearly unchanged.
2. The advantage may be larger when the candidate is White and in placement
   or flying starts, but these subgroup directions are descriptive candidates
   for future validation, not established moderators.
3. The earlier apparent v4 passivity is corpus-specific rather than a stable
   mediator of the held-out score advantage.

## Supporting evidence / 支持证据

Hypothesis 1 is supported by 80 versus 62 wins, 153 versus 152 losses, and 24
versus seven start-level directional units. Hypothesis 2 is supported by the
positive descriptive score differences in both colours and all three phases,
with the smallest difference in movement. Hypothesis 3 is supported by the
held-out survival interval crossing zero, near-equal length, and the earlier
phase-process failure to reproduce the development survival direction.

## Counterevidence / 反证

The absolute effect is small, 222 / 253 start units tie, and subgroup intervals
were not preregistered. The routes differ in training seed, source commit,
target age and accumulated SpecialistDB, so no observed difference can be
assigned to refresh policy. One fixed source family cannot guarantee a
population or deployment-wide advantage. Common 100% Malom-preserving move
rates leave the behavioural mechanism unresolved.

## Next validation experiments / 下一步验证实验

The preregistered named-route question is answered; no extension or repeat on
the consumed 253-start prefix is justified. The product owner may separately
decide whether the result is sufficient to nominate v4 as the preferred
research candidate. Any promotion, release or broader strength claim requires
its own frozen acceptance contract, genuinely unconsumed evaluation data and
explicit authority.

If the objective instead becomes target-refresh causality, the smallest valid
design remains same-source, same-seed, equal-transition refresh/no-refresh
pairs at comparable late maturity over multiple seeds. That is a new training
objective and is not authorized by this completion.
