# Sanmill retained-v3 versus no-refresh-v4 evaluation decision brief

Status: `process_generalization_source_ready_runner_and_authority_absent`

This is the current planning record after completion of the 256-game
development diagnostic and two zero-game mechanism audits. It authorizes no
new development or held-out game, model update, database or checkpoint write,
training, promotion, publication or release.

## Observed facts

### Completed development evidence

Plan identity
`035c68f80b94dddb8d139d56c38c86c4fde29fa13de5e19db1f4e1fe484c318e`
completed once under product-authorized source readiness
`eb7e75fa8f52f2f4a8c3e09b92c5be802ea20f513ac1f6fb4eb071d4cfb4a8ec`.
Its result identity is
`d250f03d72b535c0249bdf0ada7d5a75d91f7fcc44e8926c4f6dfba35d2e63d0`.
The 64-start corpus was already inspected and has
`development_corpus_reused=true`; it is not held out.
The full authority, identity and result chain is preserved in the
[completion evidence](../evidence/sanmill-retained-v3-v4-passivity-diagnostic-v1-result-2026-08-13.md).

The predeclared process endpoint obtained a direction on this fixed corpus:

| Candidate | Games surviving beyond total logical ply 120 | Rate |
| --- | ---: | ---: |
| retained-v3, refresh every 50 | 52 / 128 | 40.625% |
| retained-v4, no refresh | 62 / 128 | 48.438% |

The matched start/colour v4-minus-v3 difference was `+7.8125pp`, with
engineering interval `[+0.4051pp, +15.2199pp]`. This confirms that the named
v4 route prolonged more often under the development protocol. Survival is a
process event, not a draw or a directionally valid strength metric.

All 256 games reached rules terminals after the horizon; none reached the
1,536-post-prefix safety cap. V3 ended 4 W / 119 D / 5 L, score 49.6094%; v4
ended 6 W / 118 D / 4 L, score 50.7812%. V4 had more no-progress terminals
(66 versus 56) and fewer threefold repetitions (52 versus 63). These outcome
counts were secondary and were not powered or held out.

### Completed zero-game mechanism evidence

The safe-progress audit, result identity
`b60eaf6392d55e520b5a2a493ce7dd8961c05e811a7fd3cbb5375735fe312fea`,
found v3 selected 330/331 and v4 309/309 immediately available
W/D/L-preserving captures. Both selected every observed such capture after
ply 120. Its predeclared missed-capture difference was inconclusive with
half-width 0.0547pp, well below its 2pp maximum.

The complete Malom order audit, result identity
`e0576747c7cc6e7b3a4295b3ae31fe9a377adb5d2cd9a2c997df6f70d9bffa00`,
found a predeclared normalized-regret v4-minus-v3 difference of `-0.5619pp`,
interval `[-2.4350pp, +1.3111pp]`, also inconclusive. Conditional-on-
opportunity regret points in the opposite direction but has no separately
preregistered paired test. V4 encountered fewer distinct complete-order
choice opportunities, particularly after ply 120; this is a mediator
hypothesis, not proof that all legal actions were equivalent or that winning
paths were absent.

### Start-clustered paired-score precision

For future playing-strength planning, the independent unit must be one start,
not one start/colour unit. The two colour-specific score differences were
therefore averaged within each start before calculating the interval.

| Statistic | 128 start/colour units | 64 independent starts |
| --- | ---: | ---: |
| Difference distribution | -0.5 × 2; 0 × 121; +0.5 × 5 | -0.25 × 2; 0 × 57; +0.25 × 5 |
| Mean v4 minus v3 | +1.171875pp | +1.171875pp |
| Sample standard deviation | 11.6795pp | 8.2492pp |
| 95% engineering half-width | 2.0234pp | 2.0211pp |
| 95% engineering interval | [-0.8515pp, +3.1952pp] | [-0.8492pp, +3.1929pp] |

The cluster correction did not widen this particular interval because the
seven discordant colour units occurred in seven different starts; observed
within-start colour correlation was zero. Future contracts must nevertheless
use starts as the unit and must not assume zero clustering in a new corpus.

The fixed-corpus interval is not population inference. It neither distinguishes
the candidates nor proves equivalence. Equivalence would require a
prospectively chosen margin and the entire interval inside that margin.

Using the observed start-level standard deviation only as a fixed-width
planning input gives:

| Target 95% half-width | Starts | Total games |
| ---: | ---: | ---: |
| 2.0pp | 66 | 264 |
| 1.5pp | 117 | 468 |
| 1.171875pp | 191 | 764 |
| 1.0pp | 262 | 1,048 |

Each start costs four games: two candidates and a black/white swap. The
1.171875pp row only asks for a fixed width equal to the noisy pilot estimate.
Power estimates aimed at that same observed effect, approximately 389 starts
/ 1,556 games at 80% and 521 starts / 2,084 games at 90%, are illustrative,
post-hoc values and must not choose the contract.

### Source-only preparation for Option A

The recommended process-generalization route now has a frozen, candidate-blind
source corpus. It retains all 39 current-referee-valid histories remaining
after excluding the 12-start phase-development corpus and three histories that
strictly terminate before their requested source state. Its identity is
`3be3d76c34511e0f78d0f5bfe4a338c415c393306a955538bb85823e9d62c080`.

All 39 starts are disjoint from the completed diagnostic's 64 openings by
exact FEN and `ring16`, and have zero D4 hits in HumanDB and both candidate-
owned SpecialistDBs. The source has 18 placement, 14 movement and seven flying
starts. No candidate was loaded and no game was played.

The start-clustered horizon differences in the existing ledger have standard
deviation `29.8392pp`. A 39-start / 156-game successor therefore has an
estimated 95% half-width of `9.3651pp`, below the prior 10pp planning target,
but the new corpus can still fail the actual precision gate. Because the
histories begin at different absolute plies, the successor endpoint must be
survival through 108 additional post-start plies, not absolute ply 120.

## Hypotheses

1. The named no-refresh-v4 route's longer survival may generalize to a new
   opening corpus and may be mediated by no-capture-clock trajectories or
   different exposure to complete-order choices.
2. The named candidates may have a small paired-score difference that the
   reused 64-start development corpus cannot distinguish.
3. Any causal effect of target refresh requires new same-source, same-seed,
   equal-transition training pairs at a maturity that includes the late
   behavior of interest.

## Supporting evidence

- Ply-120 survival produced a directional fixed-corpus decision at the frozen
  precision gate.
- All games finished under the larger safety ceiling, so strict termination
  reason and no-capture history can be measured prospectively.
- The start-clustered score variance and exact discordance distribution are
  now known without spending another game.
- Colour swapping, identical frozen Sanmill work and start-level pairing remain
  the correct controls for both process and score questions.

## Counterevidence and confounders

- V3 and v4 use different seeds, source commits, target ages and accumulated
  SpecialistDBs. Their comparison cannot identify a refresh effect.
- The completed corpus is development data. Its intervals summarize only this
  fixed corpus and deterministic route.
- Survival can mean greater resistance, passivity, or both. It cannot replace
  paired score when the claim is playing strength.
- Only seven of 128 colour-specific score pairs were discordant. The observed
  standard deviation is therefore uncertain as a population planning value.
- The existing mature-fork checkpoints are much earlier than the late v4
  behavior under study. A new multi-seed evaluation of those checkpoints could
  miss the phenomenon for maturity reasons and should not be launched.

## Next validation choices

Freeze exactly one claim before choosing games. Do not combine these endpoints
after observing a new result.

### Option A: new-corpus process generalization — source selected

- Keep the two named frozen final routes and strict 500,000-node Sanmill
  protocol.
- Use the frozen 39-start phase corpus, disjoint from the completed diagnostic
  and the two candidate database routes at the audited D4 start-state level.
- Make start-clustered 108-post-start-ply survival the primary process
  estimand; retain
  no-capture count, repetition count, termination reason and complete-order
  opportunity exposure as secondary endpoints.
- Continue to strict rule terminal or an explicitly invalid safety cap.
- Freeze a start-level fixed-width target and maximum start/game/time/node
  envelope before launch.
- W/D/L remains descriptive; this option cannot answer which model is
  stronger.

### Option B: held-out playing-strength relation

- Use a newly exposure-audited opening corpus and the same black/white pairing.
- Make the within-start averaged v4-minus-v3 score difference primary.
- The product owner must first choose a practically meaningful fixed-width
  target, directional minimum effect, or equivalence margin. Use that choice,
  not the observed +1.171875pp pilot value, to set the number of starts.
- Keep process endpoints explanatory and preserve invalid-cap semantics.
- Report by colour, opening stratum and termination reason, but do not change
  the primary decision after seeing those strata.

### Option C: refresh-cadence causality

- Train same-source, same-seed, equal-transition refresh/no-refresh pairs
  across multiple seeds.
- Reach a preregistered late maturity comparable to the onset being studied;
  the existing early mature-fork cohort is not a substitute.
- Hold schedule exposure, opponent work, data routes and evaluation protocol
  paired. This is a materially larger training experiment with a separate
  authority decision.

## Current launch gates

| Gate | State |
| --- | --- |
| Completed development result and audits | verified and identity-bound |
| Independent analysis unit | one start, averaging both colours |
| Objective: process, strength or causality | Option A continued for technical preparation; no launch grant |
| Primary estimand and material threshold | draft: 108-post-start survival; 10pp maximum half-width |
| New corpus membership and exposure audit | 39 starts frozen; identity `3be3d76c...` |
| Maximum game/time/node envelope | draft: 156 games, 2 active hours, 59.904B summed node ceiling |
| Successor machine-readable plan and readiness | absent |
| Separate launch or training authority | absent |

Verdict: `not_ready`.
