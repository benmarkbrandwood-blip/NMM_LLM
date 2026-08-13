# Sanmill retained-v3 versus no-refresh-v4 evaluation decision brief

Status: `process_generalization_completed_next_objective_unselected`

Decision status: `needs_decision`

This is the current planning record after the 256-game reused-development
diagnostic, its zero-game mechanism audits, the 156-game phase-process
generalization diagnostic, and a zero-new-game cross-corpus synthesis. Every
associated game authorization is consumed. This record authorizes no new
evaluation, training, update, promotion, publication or release.

## Completed evidence

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

## Remaining hypotheses and confounders

1. Start/history composition may materially alter process behavior. The
   positive development effect was concentrated in its PerfectDB stratum, but
   this is a post-hoc mediator hypothesis.
2. The named candidates may have a small paired-score difference that neither
   visible corpus can establish as a held-out strength relation.
3. V3 and v4 differ in seed, source commit, target age and accumulated
   SpecialistDB. Their comparison cannot identify a target-refresh effect.
4. The existing mature-fork cohort is much earlier than the late v4 behavior
   of interest and should not be used as a maturity-mismatched causal proxy.

## Next owner choice

Freeze exactly one claim family before preparing another immutable plan.

### Option A: genuinely held-out route relation

- Use a new, exposure-audited corpus and the same black/white pairing.
- Make within-start averaged v4-minus-v3 score the primary endpoint.
- Choose one decision framework prospectively: fixed-width description,
  directional minimum effect, or equivalence with a meaningful margin.
- Use the selected target and a conservative variance assumption to freeze
  starts, games, active time and node work.
- Keep survival, no-capture, repetition, phase and termination reason
  explanatory; none may replace paired score after seeing the result.

This is the shortest route to a model-selection answer, but it still requires
a product choice of precision/effect/equivalence target and explicit launch
authority.

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
| New objective | not selected |
| Held-out corpus | absent |
| Primary threshold / equivalence margin | absent |
| New game, time and node envelope | absent |
| New immutable plan and readiness | absent |
| New launch or training authority | absent |

Verdict: `needs_decision`. No exact launch command has been reviewed because
no successor plan or authority exists.
