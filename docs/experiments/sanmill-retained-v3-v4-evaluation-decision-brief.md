# Sanmill retained-v3 versus no-refresh-v4 evaluation decision brief

Status: `preferred_research_candidate_selected`

Decision status: `retained_v4_preferred_for_research_no_promotion`

This is the current planning record after the 256-game reused-development
diagnostic, its zero-game mechanism audits, the 156-game phase-process
generalization diagnostic, a zero-new-game cross-corpus synthesis, and a
zero-candidate-game freeze of a new training-disjoint source pool. Every prior
game authorization is consumed. The selected 253-start / 1,012-game /
target-1.5pp fixed-width held-out successor has now also completed once and its
authorization is consumed. This record authorizes no further evaluation,
training, update, promotion, publication or release.

## Product disposition

On 14 August 2026, the product owner accepted retained-v4 as the preferred
research candidate. The decision is bound to candidate ID
`retained-v4-no-refresh`, route-bundle identity `817d2e36...`, checkpoint file
SHA-256 `295b268e...` and checkpoint payload SHA-256 `ed7932bc...`.

This resolves which of the two existing named routes should be preferred in
future research planning. Retained-v3 remains a frozen comparator. The
selection does not copy or rename a checkpoint, change an active model alias,
authorize another game or training update, or constitute promotion,
deployment, publication or release. Preserve the full
[research-candidate decision](sanmill-retained-v4-preferred-research-candidate-v1.md).

## Completed evidence

### High-precision held-out score result

Plan `6620821e...` completed all 1,012 games over 253 independent held-out
starts with result `8d7a4a0a...` and completion `8949a8fd...`. Every game
reached a strict rules terminal and zero hit the safety cap. The preregistered
start-clustered v4-minus-v3 score difference was `+1.6798pp`, interval
`[+0.6195pp, +2.7402pp]`, half-width `1.0604pp`, decision
`v4_higher_fixed_heldout_score`.

V4 recorded 80 wins, 273 draws and 153 losses versus v3's 62 / 292 / 152.
Twenty-four start units favoured v4, seven favoured v3 and 222 tied. The
secondary 108-post-start-ply survival difference was `+0.5929pp`, interval
`[-1.7668pp, +2.9526pp]`, so the earlier development-corpus passivity direction
did not reproduce on held-out data. Preserve the
[completion evidence](../evidence/sanmill-retained-v3-v4-heldout-score-v1-result-2026-08-14.md).

### Reused-development process result

Plan identity
`035c68f80b94dddb8d139d56c38c86c4fde29fa13de5e19db1f4e1fe484c318e`
completed once with result identity
`d250f03d72b535c0249bdf0ada7d5a75d91f7fcc44e8926c4f6dfba35d2e63d0`.
The 64-start corpus has `development_corpus_reused=true`; it is not held out.

V4 survived beyond absolute logical ply 120 in 62/128 games versus 52/128
for v3. The preregistered start/colour-unit difference was `+7.8125pp`, with
engineering interval `[+0.4051pp, +15.2199pp]`, and obtained the fixed-corpus
direction `v4_higher_120_ply_survival`. Survival is a process event, not a
draw or a directionally valid strength metric.

All 256 games reached strict rules terminals and no game reached the
1,536-post-prefix safety cap. Eventual score was 49.6094% for v3 and 50.7812%
for v4, but this secondary visible-corpus endpoint was neither held out nor
powered for strength. Immediate preserving-capture selection was essentially
complete, and the preregistered complete-order regret difference was
inconclusive.

See the
[development completion evidence](../evidence/sanmill-retained-v3-v4-passivity-diagnostic-v1-result-2026-08-13.md).

### Phase-process generalization result

Plan identity
`4c85ff3362927db9b63014e0c91022a5d169d19efa4aa85b3a643febd0ce3256`
completed once under direct product authorization, with primary result
identity
`6007af186b9a7ce908416f4578ebc31c0c19fc27733c32ed44751bb39cc3c812`
and completion identity
`48ac2ad4c6abc79b69c7de597ad46a5197949b4dcdee1f962e621d1be2fc57c8`.
The authorization is consumed.

All 156 games across 39 project-visible variable-history starts completed in
399.619311 evaluator-active seconds. Every game reached a strict rules
terminal and none reached the safety cap. The preregistered start-clustered
v4-minus-v3 survival difference at 108 post-start plies was `-2.5641pp`,
interval `[-6.0707pp, +0.9425pp]`, decision `inconclusive`. It does not
reproduce the earlier positive v4-survival direction and does not establish
the opposite direction.

See the
[phase completion evidence](../evidence/sanmill-retained-v3-v4-phase-process-generalization-v1-result-2026-08-14.md).

## Cross-corpus conclusion

The two completed protocols used identical named candidate route identities,
checkpoint and SpecialistDB hashes, deterministic runtime, strict referee,
Sanmill node work and safety-cap semantics. Every development start had a
12-ply prefix, so absolute ply 120 is exactly 108 post-start plies and is
comparable to the phase endpoint.

For a common analysis unit, both colours were averaged inside each start:

| Fixed corpus | Independent starts | Survival difference v4 minus v3 | 95% engineering interval |
| --- | ---: | ---: | ---: |
| Reused development | 64 | +7.8125pp | [+0.5019pp, +15.1231pp] |
| Phase histories | 39 | -2.5641pp | [-6.0707pp, +0.9425pp] |

The phase-minus-development effect contrast is `-10.3766pp`, engineering
interval `[-18.4847pp, -2.2685pp]`. It is a post-hoc description of two fixed,
independent start sets, not a preregistered directional gate or population
inference. The justified decision is that the development-corpus passivity
direction did not generalize. There is no justified universal ordering of the
routes' passivity.

The accompanying mechanisms are also unstable across corpora. The v4-minus-v3
fifty-move/threefold direction flips, and complete-order regret changes sign
while remaining inconclusive in both. Phase board-revisit share points upward,
but it was exploratory, unregistered and not accompanied by higher v4
survival. No observed mechanism supports a causal refresh claim.

The full calculation and strata are in the
[cross-corpus synthesis](../evidence/sanmill-retained-v3-v4-cross-corpus-synthesis-2026-08-14.md).

## Paired-score planning

Playing-strength planning must use one start as the independent unit, averaging
both colours inside that start. Both completed corpora remain visible and
therefore provide variance estimates only:

| Fixed corpus | Starts | Score difference v4 minus v3 | Start SD | 95% engineering interval |
| --- | ---: | ---: | ---: | ---: |
| Reused development | 64 | +1.1719pp | 8.2492pp | [-0.8492pp, +3.1929pp] |
| Phase histories | 39 | +0.6410pp | 12.1493pp | [-3.1720pp, +4.4541pp] |

Using the larger observed SD, 12.1493pp, as a conservative fixed-width
planning input gives:

| Target 95% half-width | Starts | Total games |
| ---: | ---: | ---: |
| 3.0pp | 64 | 256 |
| 2.0pp | 142 | 568 |
| 1.5pp | 253 | 1,012 |
| 1.0pp | 568 | 2,272 |

These are not authorization, a population variance guarantee, or an
equivalence margin. The earlier 264-game estimate for a 2pp half-width used
only the lower development-corpus SD and is superseded for conservative
planning by 568 games. A held-out plan must freeze its objective and decision
threshold before the number of starts is selected.

## Candidate-blind held-out source pool

The previously missing Option A input is now frozen before either candidate
policy is loaded. Pool identity
`2eb04f542f88f8360f08f97e7657ca15646582a1532358dfeb04182ebad7d8f7`
contains 361 independent complete-history starts, one per source game and one
per unique `ring16` orbit. The phase mix is 153 placement, 152 movement and 56
flying.

The sources are the 406 PlayOK games present in the import manifest but absent
from the active HumanDB used by both retained routes. Of those, 395 replayed
legally; 11 failed closed. A read-only scan then rejected any candidate state
with D4 exposure in that HumanDB or either candidate SpecialistDB, or exact /
`ring16` overlap with eight prior experiment corpora. It left 6,663 states from
361 games. All 361 selected histories passed two fresh-process strict Sanmill
replays; no start was already terminal. Source results were not read for
selection, candidate policies were not loaded, and zero candidate games were
played.

The master order was frozen to support nested, phase-covered prefixes:

| Conservative planning target | Starts | Total candidate games | P / M / F | Pool support |
| ---: | ---: | ---: | ---: | --- |
| 3.0pp half-width | 64 | 256 | 22 / 21 / 21 | available |
| 2.0pp half-width | 142 | 568 | 48 / 47 / 47 | available |
| 1.5pp half-width | 253 | 1,012 | 99 / 98 / 56 | available |
| 1.0pp half-width | 568 | 2,272 | pool total 153 / 152 / 56 | unavailable |

The product owner subsequently selected the 253-record prefix and 1.5pp
fixed-width target. That selection is now bound by the immutable successor
plan described below, and the selected prefix has completed one evaluation.
The remaining pool is not an evaluated result, the observed variance is not a
population guarantee, and completion authorizes no additional game. Preserve the
[source-pool readiness evidence](../evidence/sanmill-retained-v3-v4-late-import-heldout-pool-readiness-2026-08-14.md).

## Remaining hypotheses and confounders

1. Start/history composition may materially alter process behavior. The
   positive development effect was concentrated in its PerfectDB stratum, but
   this is a post-hoc mediator hypothesis.
2. The named v4 route has a small positive score relation on the completed
   frozen held-out prefix; its mechanism and population generality remain
   unresolved.
3. V3 and v4 differ in seed, source commit, target age and accumulated
   SpecialistDB. Their comparison cannot identify a target-refresh effect.
4. The existing mature-fork cohort is much earlier than the late v4 behavior
   of interest and should not be used as a maturity-mismatched causal proxy.

## Completed Option A successor

The product owner chose fixed-width description with a target 1.5pp 95%
engineering half-width. Machine plan identity
`6620821e879f53058d15990cd0e8c884ae62fec213b3d96200e8894c20e19714`
freezes the first 253 records of pool `2eb04f54`, both candidate colours, both
candidates and 1,012 games. The ordered prefix identity is `99951a69`; its
phase support is 99 placement / 98 movement / 56 flying.

The independent unit is one start after averaging the black and white
v4-minus-v3 score differences. Strict W/D/L score is primary. Survival,
no-capture, repetition, phase, length, termination and Malom diagnostics are
secondary and may not replace score after outcomes are observed.

Two zero-game preflights passed every technical gate with stable source
readiness `f233c991...`. Direct authorization `816cc390...`, launch
`b4505be8...`, result `8d7a4a0a...` and completion `8949a8fd...` are now
consumed. The observed interval was wholly positive and its half-width passed
the 1.5pp precision gate. The target width was not an equivalence margin. See
the
[frozen plan](sanmill-retained-v3-v4-heldout-score-v1.md) and
[completion evidence](../evidence/sanmill-retained-v3-v4-heldout-score-v1-result-2026-08-14.md).

## Other unselected objectives

### Option B: target-refresh causality

- Train same-source, same-seed, equal-transition refresh/no-refresh pairs over
  multiple seeds.
- Reach a preregistered late maturity comparable to the behavior of interest.
- Pair schedule exposure, opponent work, data routes and evaluation protocol.

This is materially larger and requires separate training and evaluation
authority. Neither retained final route can answer it.

### Option C: history/revisit mechanism

- Use a new corpus and preregister one history-aware revisit/no-progress
  endpoint plus its denominator and directional gate.
- Treat score and survival as secondary unless a separate powered claim is
  frozen.

This has lower immediate model-selection value. The current exploratory
revisit interval cannot be promoted into the new decision.

## Current gates

| Gate | State |
| --- | --- |
| Development process result | completed; authorization consumed |
| Phase-process generalization | completed; authorization consumed; `inconclusive` |
| Cross-corpus zero-game synthesis | completed; prior positive direction not reproduced |
| Independent analysis unit | one start, averaging both colours |
| New objective | Option A fixed-width held-out score completed |
| Candidate-blind held-out source pool | identity `2eb04f54`; selected prefix now consumed as evaluation evidence |
| Evaluation corpus subset | first 253 records completed; prefix `99951a69`; 99 / 98 / 56 phases |
| Primary threshold / equivalence margin | target half-width 1.5pp; no equivalence margin |
| New game, time and node envelope | 1,012 games; 4 active hours; frozen node ceiling |
| Immutable plan and runtime | plan `6620821e`; launch `b4505be8`; completion `8949a8fd` |
| Primary decision | `v4_higher_fixed_heldout_score`; +1.6798pp; [+0.6195pp, +2.7402pp] |
| Research candidate disposition | retained-v4 selected for research; retained-v3 remains comparator |
| Further launch, promotion or training authority | absent |

Verdict: the named v4 route scored higher on the frozen held-out prefix under
the preregistered precision rule and is now the product-selected preferred
research candidate. Promotion readiness remains `needs_decision`; no
refresh-causal, population-strength, Elo, deployment or release conclusion
follows.
