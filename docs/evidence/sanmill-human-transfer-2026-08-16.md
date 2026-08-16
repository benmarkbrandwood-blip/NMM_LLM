# Sanmill-to-Human-Risk Transfer Test — 16 August 2026

## Decision

Decision: `A_substantive_transfer_exists`

On the frozen 360-state pool, the full out-of-fold human choice model selected
a 100,000-node Sanmill-inducing action in 45 states, or 12.500%.  The matched
uniform-safe-action baseline was 2.822% and the measured oracle ceiling was
17.500%.  The resulting gain was 9.678 percentage points, with a state
bootstrap 95% interval of [6.785, 12.777] points.  The transfer ratio was
65.936%, with interval [52.131%, 78.823%].

The frozen existence condition required the lower bound of `A - b` to exceed
zero.  The independently frozen substantive condition required the lower
bound of transfer to be at least 25%.  Both conditions passed.  The threshold
and the sole primary estimator were sealed in commit `420248d` before any
human-risk value was calculated for this state pool.

This supports only a shared one-step positional difficulty structure between
the frozen OOF human choice model and the exact pinned Sanmill runtime at
100,000 nodes.  It does not restore the rejected human-specialization route,
identify a causal human response, establish repeated-play reachability or
redemption, or show human-trap ability, playing strength, or product value.

## Frozen identities

| Item | Identity or SHA-256 |
| --- | --- |
| Coverage audit | `401006c5aafc1e09d86498bd1f0a78726f5064a962c3e3efa8e2e1879599d3d8` |
| Coverage audit file | `35c000e01f028929299f88311960ff6359f2226f9dfe60c6a9a588b59d131886` |
| Transfer plan | `51b477b576a9d29c602ac70b35aa175dd40e4c1a953494e113bea633cebe80ba` |
| Transfer plan file | `cd2d20da0e3368b56546fc18b5596428a1d917ab26d0e31cc0fae4c89c07ef48` |
| Transfer result | `c6dce5690a138361238ddd4661cce78251e67fb0f9d8003a0473a0b12c1a2700` |
| Transfer result file | `0c9ec12ebc57fdead9ce77942edc68e9389564a146395d5ba1badd023b103298` |
| Main state pool | `261bd980acda6018f6c7fe4c268b1e00f190e33ae5c055ceb466d43c77dcc53c` |
| Main result | `fa3e5d6afa6a0816022909e287c0ab51d05311e3ad66ceac69c4441e9a981d3f` |
| Main correction | `85b04c9681ce8cf9fbd340885944c8d6da69831e00adfad410ebfef7018a199b` |
| Cross-fit structure | `b2ab654856a13d17fbc5256b6395c078e6cd13db9114da390daf720d952e6ae4` |
| Readiness result | `0df4a8bcfab8636048c8b005945a1d4bd719b23f377c06d25a6d6e5b745d0ec2` |
| Product-conversion result | `3da605d1d92d1a53b00dc9dabda1ac95c2e4624ec53354bddc0f8a7f53301d5f` |

The raw main result was always read together with its analysis correction.
The correction preserves the primary 100,000-node values and supplies the
corrected invariant/sensitive state decomposition.

## Availability audit before estimator freeze

All 360 states were checked individually.  Each source session is present in
the frozen 6,400-game cross-fit sample, its acting player maps to the session's
held-out fold, its other player is in that same fold, all five fold parameter
sets exist, every `A_pos` action has a successor FEN, and every state is present
in the sealed main result.

Coverage was 360/360, with 120 states per phase.  Fold counts were 98, 54, 74,
57, and 77.  There is no unavailable subset, so the requested subset-bias
comparison is not applicable.  The previously reported 59.18% cross-fold
discard occurred before the 6,400-game same-fold sample was frozen.  The main
pool was selected inside that retained sample; the earlier discard therefore
does not create partial coverage in this test.

The product-conversion manifest preserves aggregate results and query counts,
but not per-successor risk rows.  Exact recomputation was therefore necessary.
It reused the sealed fold means, scales, full and geometry coefficients, the
same ten-feature encoder, the same numerical softmax contract, and the same
tier-loss routine.  No estimator was refit.  The product-conversion record's
sealed OOF reproduction remains the bit-level source evidence: 10,416 D-to-L
events, 199,234 parent-D decisions, full log loss 1.9867105406467653,
geometry log loss 2.146902903833993, and improvement 0.1601923631872277.

## Frozen estimator

For each safe successor, the model predicts the probability that the
opponent chooses a response with a lower positional WDL tier.  W-tier response
states count W-to-D and W-to-L, and D-tier response states count D-to-L.  This
all-tier definition matches the main engine endpoint; it does not replace the
earlier D-only human research question.

The full ten-feature model is the only primary estimator.  Its parameters are
from the source acting player's held-out fold.  Actions are ordered by
`(from, to, capture)` and an exact risk tie selects the first canonical action.
The state-uniform estimands are:

- `A`: downgrade rate of the model's maximum-risk safe action;
- `b`: mean within-state uniform safe-action downgrade rate;
- `o`: share of states having at least one inducing safe action; and
- `transfer = (A - b) / (o - b)`.

The nested three-feature geometry model repeats these quantities as a frozen
secondary control and cannot change the primary decision.  Intervals use
20,000 state-level percentile bootstrap replicates.  No pseudo-event or prior
is used.

## Primary result and geometry control

| Specification | `A` | `b` | `o` | `A-b` (95% interval) | Transfer (95% interval) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full ten-feature | 12.500% | 2.822% | 17.500% | 9.678pp [6.785, 12.777] | 65.936% [52.131, 78.823] |
| Geometry control | 9.444% | 2.822% | 17.500% | 6.623pp [4.006, 9.405] | 45.119% [30.329, 59.724] |

The full model selected 11 more inducing actions than geometry, a paired
difference of 3.056 percentage points with interval [1.111, 5.000].  Thus the
alignment is not explained entirely by the nested geometry panel.  This is a
secondary comparison of frozen predictors, not causal attribution to an
individual feature.

Among the 60 states containing both inducing and non-inducing actions, mean
within-state AUC was 0.879 [0.817, 0.933] for the full model and 0.807 [0.735,
0.871] for geometry.  The other 300 states correctly abstain from AUC because
all actions have the same engine label.

## Phase and budget-type decomposition

The phase results are secondary and cannot reverse the primary decision.

| Phase | `A` | `b` | `o` | `A-b` (95% interval) | Transfer (95% interval) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Placement | 11.667% | 3.427% | 18.333% | 8.240pp [3.690, 13.217] | 55.277% [30.082, 79.341] |
| Movement | 1.667% | 1.369% | 4.167% | 0.298pp [-1.786, 2.679] | 10.638% [-100.000, 100.000] |
| Flying | 24.167% | 3.669% | 30.000% | 20.497pp [13.689, 27.636] | 77.846% [61.755, 91.757] |

There is no resolved movement-stage transfer.  Placement and flying account
for the overall result.  This limits any claim of phase-general behavior.

Budget-type rows classify an action as invariant when it induces at all three
frozen budgets, or sensitive when it induces at a nonempty strict subset.

| Full-model action class | Selected | Uniform | Oracle | Gain | Transfer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Invariant | 9.722% | 1.711% | 12.500% | 8.011pp [5.359, 10.955] | 74.253% [58.906, 87.671] |
| Sensitive | 9.722% | 4.074% | 15.556% | 5.648pp [3.314, 8.113] | 49.192% [31.962, 65.683] |

The model identifies both classes, so alignment is not confined to a fixed
blind spot.  It is nevertheless stronger for the invariant class.  The
geometry control transfers 61.379% [44.857, 76.989] for invariant actions and
22.579% [4.644, 40.081] for sensitive actions.  The full feature increment is
therefore most practically visible on the budget-sensitive class, but this
remains an associational secondary decomposition.

## `A_pos` cardinality and ties

The 32 singleton states have no action-selection headroom.  For cardinality
2, the full-model transfer point was -100%; for 3--4 it was 55.556% with an
interval spanning -33.333% to 100%; for 5--8 it was 100% among 49 states; and
for 9 or more it was 67.727% [53.671, 81.054] among 191 states.  Sparse bins
are unstable and are not separate decisions.

The full model had exact maximum-risk ties in 146 states.  Of these, 129 had
maximum predicted risk zero and also had no inducing engine action, so their
canonical tie resolution cannot affect `A`, `b`, or `o`.  Only 7 of the other
tied states had any inducing action, and the frozen tie rule selected an
inducing action in 4.  The 214 unique-maximum states had `A=19.159%`,
`b=4.260%`, and transfer 68.006%.  These tie diagnostics are exploratory
sensitivity context; the canonical rule was frozen before calculation.

The selected full-model transitions were 32 D-to-L, 11 W-to-L, and 2 W-to-D.
The result is therefore not merely a consequence of one transition label,
although D-to-L supplies most selected events.

## Resources, access, and verification boundary

Exact recomputation used 112,920 corrected Malom queries and 71.63 active
seconds, below the frozen ceilings of 500,000 and 3,600.  It reconstructed
6,176 nonterminal response choice sets and 113,086 response actions; seven
safe successors were terminal.  It made zero Sanmill queries, played zero
games, loaded zero policy models, performed zero estimator fits or training
updates, and wrote no database.

Official selection, confirmation, final-test, research-confirmation, and the
remaining 108 source-pool records all had zero content reads.  The guard test
raises before a protected producer is called.  Every Malom label comes from
the trusted `sector-corrected-v1` snapshot and is positional-only `A_pos`, not
full-rules `A_allow`.

The historical `sanmill_checkout` route group produced 63 passes and seven
fail-closed local integration failures.  Each failure reports the already
known moving-checkout change inside the pinned bridge source scope.  This
analysis did not invoke that route, change a machine path, or weaken a test to
hide it.

## Claim boundary and remaining evidence gap

The result is limited to source-game-unique states from the observed
PlayOK-like domain, the exact frozen OOF estimators, the exact pinned Sanmill
runtime at 100,000 nodes, one response, and positional WDL.  UI orientation,
time control, and exact source rules are not reconstructable.  F0-D0's
nonrandom history attrition remains: 1,751 excluded games contain only 35
draws, versus 26,157 draws among 92,789 retained games.  Another 54,923 games
lack an independently verifiable terminal basis.

F0-H0 remains `stop_condition_triggered`, estimator readiness remains
`B_not_ready_fail_closed`, and product conversion remains
`C_conversion_not_established`.  The earlier Sanmill mechanism result remains
`mechanism_gate_passed`.  To turn the shared ranking into human-trap evidence
would still require action-specific human counterfactual identification,
policy-induced multi-step visitation, and demonstrated conversion of an
opponent downgrade into product-level gain.  None is supplied here.
