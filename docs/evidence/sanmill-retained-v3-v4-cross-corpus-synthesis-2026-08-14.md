# Retained-v3/v4 cross-corpus zero-game synthesis — 14 August 2026

Status: `completed_zero_new_game_cross_corpus_synthesis`

Decision status: `needs_decision`

This record compares two already completed fixed-corpus diagnostics. It loaded
their existing ledgers read-only and requested no candidate move, game,
training step, optimizer update, database or checkpoint write. It is not a
held-out result, population inference, target-refresh causal experiment,
equivalence result, or authority for another evaluation or training run.

## Bound source evidence

| Evidence | Identity / scope |
| --- | --- |
| Reused-development diagnostic | plan `035c68f80b94dddb8d139d56c38c86c4fde29fa13de5e19db1f4e1fe484c318e`; result `d250f03d72b535c0249bdf0ada7d5a75d91f7fcc44e8926c4f6dfba35d2e63d0`; 64 starts / 256 games |
| Phase-process diagnostic | plan `4c85ff3362927db9b63014e0c91022a5d169d19efa4aa85b3a643febd0ce3256`; result `6007af186b9a7ce908416f4578ebc31c0c19fc27733c32ed44751bb39cc3c812`; 39 starts / 156 games |
| Phase-process completion | `48ac2ad4c6abc79b69c7de597ad46a5197949b4dcdee1f962e621d1be2fc57c8` |

Both diagnostics used the same two route-bundle identities, checkpoint file
and payload hashes, SpecialistDB file hashes, deterministic CPU float32
argmax, seed 42 runtime, strict complete-history Sanmill referee, 500,000-node
per-turn ceiling, and 1,536-post-start invalid safety-cap semantics. Every
development prefix was exactly 12 logical plies, so its absolute-ply-120
event is exactly 108 post-prefix plies and is comparable to the phase
diagnostic's 108-post-start event.

The intentional difference is the start/history corpus. The development
corpus reused 64 already inspected 12-ply openings. The phase corpus used 39
project-visible variable histories across placement, movement and flying.
Neither corpus is held out.

## Start-clustered survival comparison

Both colours were first averaged inside each start. The development result's
original preregistered interval treated 128 start/colour units as support; the
table below instead applies the common one-start analysis unit to both
corpora. This changes the engineering interval slightly, not the mean.

| Fixed corpus | Independent starts | Mean v4 minus v3 | 95% engineering interval | Start-difference distribution |
| --- | ---: | ---: | ---: | --- |
| Reused development | 64 | +7.8125pp | [+0.5019pp, +15.1231pp] | -0.5 × 6; 0 × 43; +0.5 × 14; +1 × 1 |
| Phase histories | 39 | -2.5641pp | [-6.0707pp, +0.9425pp] | -0.5 × 2; 0 × 37 |

The earlier positive v4-survival direction did not reproduce on the phase
corpus. The phase point estimate is opposite, but its preregistered interval
crosses zero and does not establish higher v3 survival.

As a post-hoc fixed-corpus heterogeneity description, phase minus development
is `-10.3766pp`. Combining the two independent start-set standard errors gives
an engineering interval of `[-18.4847pp, -2.2685pp]`. This line was not a
preregistered directional gate and is not population inference. It supports
closing the specific claim that the development-corpus direction generalized;
it does not support a universal passivity ordering.

Descriptive strata reinforce the lack of a stable direction:

| Fixed stratum | Starts | Mean v4 minus v3 survival | 95% engineering interval |
| --- | ---: | ---: | ---: |
| Development Book | 22 | +6.8182pp | [-6.5455pp, +20.1818pp] |
| Development HumanDB | 21 | +2.3810pp | [-8.2607pp, +13.0226pp] |
| Development PerfectDB | 21 | +14.2857pp | [+0.5210pp, +28.0504pp] |
| Phase placement | 18 | -2.7778pp | [-8.2222pp, +2.6667pp] |
| Phase movement | 14 | -3.5714pp | [-10.5714pp, +3.4286pp] |
| Phase flying | 7 | 0.0000pp | [0.0000pp, 0.0000pp] |

These strata are post-hoc and small. In particular, the development effect is
concentrated in its PerfectDB stratum; this is a hypothesis about corpus
composition, not a tested mediator.

## Mechanism stability

- The immediate safe-capture hypothesis is unsupported in both corpora.
  Development v3/v4 selected 330/331 and 309/309 observed preserving captures;
  phase v3/v4 selected 156/156 and 155/155.
- The terminal-reason direction flipped. Development v4 had more fifty-move
  terminals and fewer threefold terminals than v3 (66 versus 56; 52 versus
  63). Phase v4 had fewer fifty-move terminals and more threefold terminals
  (14 versus 18; 30 versus 27).
- Start-clustered complete-order regret was `-0.5619pp` on development,
  interval `[-2.4350pp, +1.3111pp]`, and `+1.7401pp` on phase, interval
  `[-0.9551pp, +4.4353pp]`. Both are inconclusive and their direction is not
  stable.
- Phase board-revisit share was exploratory: v4 minus v3 `+2.1487pp`, interval
  `[+0.2565pp, +4.0408pp]`. It had no preregistered directional gate, and v4
  survival was not higher on that corpus. Revisit exposure is therefore a
  possible separately testable mechanism, not a sufficient causal
  explanation.

## Start-clustered score precision and planning

Eventual score remains descriptive because both corpora were project-visible.
It cannot choose a stronger route.

| Fixed corpus | Starts | Mean v4 minus v3 score | Start SD | 95% engineering interval |
| --- | ---: | ---: | ---: | ---: |
| Reused development | 64 | +1.1719pp | 8.2492pp | [-0.8492pp, +3.1929pp] |
| Phase histories | 39 | +0.6410pp | 12.1493pp | [-3.1720pp, +4.4541pp] |

The post-hoc phase-minus-development score-effect contrast is `-0.5308pp`,
interval `[-4.8464pp, +3.7847pp]`. There is no useful cross-corpus score
separation.

Using the larger observed start-level score SD, 12.1493pp, as a conservative
fixed-width planning input gives:

| Target 95% half-width | Starts | Total games |
| ---: | ---: | ---: |
| 3.0pp | 64 | 256 |
| 2.0pp | 142 | 568 |
| 1.5pp | 253 | 1,012 |
| 1.0pp | 568 | 2,272 |

Each start costs four games: two candidates and both candidate colours. These
are engineering estimates from two fixed, visible corpora, not a population
variance guarantee or authorization. A genuine held-out contract must first
freeze paired score as primary, an independently exposure-audited corpus, and
one owner-chosen fixed-width, minimum-effect or equivalence target. “Not
distinguished at the chosen width” must not be relabelled “equivalent” unless
an equivalence margin is prospectively frozen and the whole interval lies
inside it.

## Decision

The process-generalization branch is complete: the earlier positive survival
direction did not generalize, and adding games to the observed endpoint would
be a prohibited post-result extension. Do not repeat or extend either consumed
diagnostic.

The next product decision is one of three different objectives:

1. For route selection, plan a genuinely held-out paired-score evaluation and
   choose its precision/effect/equivalence target before choosing the budget.
2. For target-refresh causality, authorize new same-source, same-seed,
   equal-transition, multiple-seed training pairs at the late maturity of
   interest. The retained routes cannot answer causality.
3. For mechanism research, separately preregister a new-corpus history/revisit
   endpoint. It has lower direct model-selection value and cannot promote the
   current exploratory interval.

Verdict: `needs_decision`. No launch command or unconsumed authority exists.
