# Retained-v3/v4 passivity diagnostic v1 result — 13 August 2026

Status: `completed_development_diagnostic_not_held_out`

Training-readiness verdict: `needs_decision`

This is the immutable completion record for the named retained-v3 and
no-refresh-v4 final-route diagnostic on a reused development corpus. It is not
held-out evidence, a playing-strength result, a refresh-cadence causal
experiment, an equivalence result, or authority for another game, training,
promotion, publication or release.

## Authority, execution and identities

| Item | Recorded value |
| --- | --- |
| Frozen plan identity | `035c68f80b94dddb8d139d56c38c86c4fde29fa13de5e19db1f4e1fe484c318e` |
| Frozen plan file SHA-256 | `e4394d015490d1e337554589c339db19a20ae45f2968bb9cbceee2ba207cd5b3` |
| Product-authorized source readiness | `eb7e75fa8f52f2f4a8c3e09b92c5be802ea20f513ac1f6fb4eb071d4cfb4a8ec` |
| Post-authorization launch readiness | `a591af4a45f2162380c5169e463c9e12a6ae142a17638769cff6a9d9a3a409c2` |
| Authorization identity | `d1b1286f6a37f14e7034e084271c991a6c36378b79b8684ac120025592bbcc4f` (`product-owner-direct`) |
| Launch identity | `4a43f7807de9f5c4dbfd76b8e07ba762b20cc3c4f48ea7b2b21e510a63e5169d` |
| Runtime spec identity | `d47c94e29b0acd1c32f2515b90d82716a59f6c5e767b17c2f30ff829e87e5a9b` |
| Result identity | `d250f03d72b535c0249bdf0ada7d5a75d91f7fcc44e8926c4f6dfba35d2e63d0` |
| Completion identity | `fe1f243c26f0a9f987e3e24997e23ab296e5c8c6c7a967b5cf57e6f77e89ac64` |
| Ledger SHA-256 | `c064f29d77cedd42a9ef405ec44dbbda045b47be31092e952568cecb5d49b562` |
| Report file SHA-256 | `54def8962cc32898e1810c9abd9b97d9db0d06ee1c30ed8633f5eccd44646322` |
| Completion file SHA-256 | `ceb98b0c67eaac40bde470a2bc60965628aacd9625bf5ae16dde1e3735fd877e` |

The frozen plan binds evaluator anchor `361d99a43a9ca549b6f4594d8cb5c26a23d5dd54`.
The final launch ran from clean descendant source
`ff35dfad0606c4405c4fddd703bd0d7104125a9d`, after the sidecar-free resource
bindings and launch gates were published. It started at
`2026-08-13T11:41:10.243544Z` and completed at
`2026-08-13T11:57:31.005995Z`.

All 256 planned games completed once, covering 64 starts, both candidate
colours, and both candidates. Evaluator active time was 984.057494 seconds
(0.2733 hours), below the two-hour ceiling. There was no automatic retry,
semantic recovery, resource extension, training, optimizer update, database
write, checkpoint write, held-out claim, promotion, publication or release.
The authorization is consumed.

## Predeclared primary result

The primary process endpoint was strict-referee survival beyond total logical
ply 120. It is not a draw and does not predict the eventual result.

| Candidate | Survived / games | Survival rate |
| --- | ---: | ---: |
| retained-v3, refresh every 50 | 52 / 128 | 40.6250% |
| retained-v4, no refresh | 62 / 128 | 48.4375% |

The matched start/colour mean difference, v4 minus v3, was `+0.078125`
(+7.8125 percentage points). Its predeclared fixed-corpus engineering interval
was `[+0.0040505, +0.1521995]`; half-width `0.0740745` was below the frozen
0.10 maximum. The predeclared decision is therefore
`v4_higher_120_ply_survival` on this reused fixed development corpus.

This is the only directional process decision. It confirms that the named v4
route was more likely to remain ongoing at ply 120 under this protocol. It
does not say that v4 is stronger, that longer survival is good, or that target
refresh caused the difference.

## Rules terminal outcomes

All 256 games subsequently reached a strict rules terminal; none reached the
1,536-post-prefix safety cap. The longest games were 254 total logical plies
for v3 and 259 for v4. This shows that games under this particular evaluator
can finish when allowed beyond ply 120. It cannot recover the counterfactual
outcomes of the 605 separately observed attempt-003 training truncations.

| Candidate | W / D / L | Score | Fifty-move / no-progress | Threefold repetition | Fewer than three | No legal moves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| retained-v3 | 4 / 119 / 5 | 49.6094% | 56 | 63 | 4 | 5 |
| retained-v4 | 6 / 118 / 4 | 50.7812% | 66 | 52 | 5 | 5 |

The v4 route ended more often through the no-capture clock and less often
through threefold repetition. That is a candidate history-aware mechanism,
not a confirmed explanation and not a strength result.

At ply 120, history-free Malom classified the 52 surviving v3 positions as
5 theoretical wins and 47 draws, and the 62 surviving v4 positions as one
theoretical win and 61 draws; neither set contained a theoretical loss. The
survivor sets are selected by different route behavior and have very small win
counts. This comparison is hypothesis-generating only.

## Zero-game mechanism audits

The completed ledger was subsequently reanalysed twice without a new game,
model load, update or database write.

The safe-progress audit is bound to plan identity
`3338ba5979db20d89d81bf4408d2fa1eeef098eefb6d854ef56d707ad268fb73`,
result identity
`b60eaf6392d55e520b5a2a493ce7dd8961c05e811a7fd3cbb5375735fe312fea`,
and report SHA-256
`d68b8279cc65a429a900388396efde50da00cd668fcf5249b94177cb940a12b1`.
It found:

- v3 selected a preserving capture on 330 of 331 safe-capture opportunity
  turns; v4 selected one on all 309 of 309;
- after ply 120 both selected every observed safe preserving capture, 12/12
  and 10/10 respectively;
- the predeclared missed-safe-capture share difference was `-0.0279pp`, with
  interval `[-0.0826pp, +0.0268pp]`, decision `inconclusive`; and
- post-ply-120 local-board revisits were 27/845 for v3 and 14/1,060 for v4.

The complete-order audit is bound to plan identity
`95e1d5e6640765e14852b9dfc3f2793bf72ee583bc95fc0a3bd1512acb36d23d`,
result identity
`e0576747c7cc6e7b3a4295b3ae31fe9a377adb5d2cd9a2c997df6f70d9bffa00`,
and report SHA-256
`65d3e0a8959feeee7e0510f8c1f4491b61a7e8d7890d10f41e6171f246be3701`.
Its predeclared per-game normalized ordinal-regret difference was `-0.5619pp`,
with interval `[-2.4350pp, +1.3111pp]`, decision `inconclusive`.

The descriptive conditional direction is different from the predeclared
all-orderable denominator: when a complete-order choice opportunity existed,
mean regret was 36.83% for v3 and 38.86% for v4; after ply 120 it was 38.57%
and 45.67%. Those conditional values have no separately preregistered paired
test and must not be promoted into a post-hoc decision. V4 also encountered
fewer complete-order choice-opportunity turns, 201/1,060 after ply 120 versus
256/845 for v3. That is a possible opportunity-exposure mediator. It does not
mean all legal actions were equal or that no winning path existed.

Together, the audits rule out a large missed-immediate-safe-capture mechanism
on this corpus and leave complete-order selection inconclusive. They do not
identify the cause of longer v4 survival.

## Start-clustered horizon precision

The predeclared horizon interval used 128 start/colour units. For planning the
new-corpus process confirmation, the two colour-specific v4-minus-v3 survival
differences were also averaged inside each of the 64 independent starts.

The start-level differences were `-0.5` six times, `0` 43 times, `+0.5`
14 times and `+1` once. Their mean was still `+7.8125pp`; sample standard
deviation was `29.8392pp`; the 95% engineering half-width was `7.3106pp`, for
interval `[+0.5019pp, +15.1231pp]`. This start-clustered result remains a
fixed reused-development-corpus description, not population inference.

Using that observed standard deviation only as a planning input gives 35
starts / 140 games for an estimated 10pp half-width, 61 / 244 for 7.5pp, and
137 / 548 for 5pp. The subsequently audited 39-start phase corpus therefore
has an estimated 9.3651pp half-width at 156 games. Its actual variance may be
larger, so the successor must permit `inconclusive_precision` and must not add
starts after seeing results.

## Start-clustered score precision

The original report's paired engineering intervals treat 128 start/colour
units as the analysis units. For planning a future strength evaluation, the
two colours from one opening must instead be clustered: first average the two
colour-specific v4-minus-v3 score differences within each start, then compute
variation across 64 starts.

The 128 colour-specific score differences were `-0.5` twice, `0` 121 times,
and `+0.5` five times. After averaging colours, the 64 start-level differences
were `-0.25` twice, `0` 57 times, and `+0.25` five times. No start had two
discordant colours, so the observed within-start colour correlation was zero.

| Statistic | 128 start/colour units | 64 start units |
| --- | ---: | ---: |
| Mean v4 minus v3 score | +1.171875pp | +1.171875pp |
| Sample standard deviation | 11.6795pp | 8.2492pp |
| 95% engineering half-width | 2.0234pp | 2.0211pp |
| 95% engineering interval | [-0.8515pp, +3.1952pp] | [-0.8492pp, +3.1929pp] |

Thus the requested clustering correction does not inflate this fixed-corpus
interval. It is still the correct unit for every future design; the sparse
seven discordances do not justify assuming zero cluster effect elsewhere.
This interval is development-corpus engineering evidence, not population
inference. Because it crosses zero, it does not distinguish strength. Because
no equivalence margin was frozen, it also does not establish equivalence.

Using the observed start-level standard deviation only as a fixed-width
planning basis gives:

| Target 95% half-width | Required starts | Total games at 4 games/start |
| ---: | ---: | ---: |
| 2.0pp | 66 | 264 |
| 1.5pp | 117 | 468 |
| 1.171875pp | 191 | 764 |
| 1.0pp | 262 | 1,048 |

The 1.171875pp row merely sets fixed width equal to the noisy pilot point
estimate. Conventional power calculations that target that same observed
effect, about 389 starts / 1,556 games for 80% and 521 starts / 2,084 games
for 90%, are illustrative only and must not choose the budget. A new contract
must instead freeze a product-relevant minimum effect, fixed-width target, or
equivalence margin before inspecting held-out outcomes.

## Decision and next validation

Preserve the completed diagnostic and both zero-game reports under their exact
identities. Do not rerun, extend, relabel the development corpus as held out,
or reinterpret longer survival as strength.

The next owner decision is the claim to buy:

1. For **process generalization**, the source-only 39-start phase corpus is
   now frozen under identity `3be3d76c...`. Because its histories begin at
   logical plies 7–178, use survival through 108 additional plies rather than
   an absolute ply-120 snapshot. The runner, machine plan, readiness and
   authority remain absent. W/D/L remains descriptive.
2. For **playing-strength relation**, use the start-clustered paired score as
   the primary endpoint, select a practically meaningful fixed-width or
   effect threshold first, and then bind the required starts, four games per
   start, active-time ceiling and invalid-cap rule. Process metrics remain
   explanatory secondary endpoints.
3. For a **causal refresh claim**, neither route suffices. Train same-source,
   same-seed, equal-transition refresh/no-refresh pairs across multiple seeds
   to a maturity that includes the late behavior of interest.

Do not launch the existing early mature-fork checkpoints as a shortcut for
the third question. Their exposure is substantially earlier than the late
retained-v4 transition, so absence of the target behavior would be
uninformative. No new game or training run is authorized by this evidence.
