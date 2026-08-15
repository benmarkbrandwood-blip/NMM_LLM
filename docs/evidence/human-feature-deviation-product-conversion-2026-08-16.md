# Human feature-deviation product conversion derivation

Date: 16 August 2026

Status: `C_conversion_not_established`

## Outcome first

The requested conversion from predictive statistics to product-scale safe
human-trap value cannot be established from the existing evidence.  The
frozen estimator can rank positional D-to-L risk, and its predictions imply
large within-state risk contrasts across `A_pos` successors.  Those
contrasts are observational prediction contrasts.  They do not identify
what the same opponent would do after the learner deliberately selected a
different action, how the new policy would change state visitation, or how
often a learner would convert a positional advantage into a win.

This is conclusion C, not B.  The obstacle is not that the 487-player
confirmation arm is too imprecise for a known product estimand.  The product
estimand is not identified, so no product tier is decidable at 487 players
or at any larger number of players drawn under the same observational
design.

The previous `B_not_ready_fail_closed` decision remains effective and is not
reopened, relaxed, or reinterpreted.  This derivation creates no new
research question and grants no authority to open research confirmation,
official holdouts, or any later gate.

## Frozen identity and correction history

The initial derivation contract was frozen and pushed before any new
hypothetical-successor query:

- v1 derivation identity:
  `5f45c0296d86553b7fa67e1aea45269690da15ccf00d840f0c3709479c575e1b`;
- v1 file SHA-256:
  `0203ef221c67d3a0d71d395910b54e835f757cec2f2bb9c6c78173e8c0204ef3`;
  and
- freeze commit: `a59354c`.

An independent exact-value check then found that three narrative-rounded
log-loss values had been entered as `1e-12` reproduction targets.  They
differed from the sealed machine result by about `1e-11`.  This was a
transcription defect, not a scientific result.  It was detected before any
new successor query or protected-content read.

V1 remains immutable.  V2 corrects only those three exact values and inherits
every method, bound, threshold, access rule, and claim boundary:

- v2 derivation identity:
  `d4ead92b6ff1b8be07d5a16ab1041547e973ee50d4ae794e3ab5fc1a06d20444`;
- v2 file SHA-256:
  `697675b868bafe6bf9ff23ee198fa781eae3f077f71c85c540dcbd737215bf70`;
  and
- correction commit: `da8a48d`.

The execution implementation was committed and pushed at `d39598b` before
the frozen sample was replayed.  The completed result has identity
`3da605d1d92d1a53b00dc9dabda1ac95c2e4624ec53354bddc0f8a7f53301d5f`
and file SHA-256
`885000ea45507ae8a4e64a0aa114c2425b02b5f5e5c4d35570c58e526a2d3d0a`.
The result file retains the contract-start date in its name; the bounded
calculation crossed local midnight and completed on 16 August.

## What was and was not executed

The exact 6,400 frozen research-exploration games were replayed twice in one
bounded process: once to reproduce the original out-of-fold inventory and
once to evaluate hypothetical safe successors.  No estimator was refitted.

The original sealed results reproduced at machine precision:

| Quantity | Reproduced value |
| --- | ---: |
| Covered decisions | 292,192 |
| Parent-D decisions | 199,234 |
| Observed D-to-L events | 10,416 |
| Geometry log loss | 2.146902903833993 |
| Full-panel log loss | 1.986710540646765 |
| Paired improvement | 0.160192363187228 nat |

The run made 3,869,797 original-inventory and 40,477,298 hypothetical-
successor Malom queries, or 44,347,095 total.  It evaluated 2,520,495
opponent response choice sets and 38,382,497 response actions; 4,510 safe
successors were terminal.  Elapsed time was 11,828.51 seconds.  Both the
80-million-query and four-hour hard limits were respected.

Malom access was read-only and bound to `sector-corrected-v1`.  There were
zero reads of research confirmation, official selection, official
confirmation, official final-test, HumanDB, or source pool `2eb04f54`; zero
games, searches, strategy-model loads, training, or updates; and zero
database writes.

## The six-link conversion chain

| Link | Frozen estimand or evidence | Current status |
| --- | --- | --- |
| 1. Identification | Out-of-fold choice prediction, calibration, and D-to-L discrimination | Estimable as prediction on existing exploration data |
| 2. Safe steerability | Spread of predicted opponent risk across positional-safe successors | Predictive plug-in only; action-specific causal response is absent |
| 3. Single-step inducement | Change in the same opponent's D-to-L probability caused by selecting an action | Not identified from observational human-human paths |
| 4. Multi-step accumulation | Policy visitation, repeated opportunities, dependence, and adaptation | Not identified; no policy rollout or game evidence |
| 5. Redemption | Learner win conversion after opponent positional D-to-L | Not identified; requires learner-versus-human game evidence |
| 6. Product effect | Complete-game score or win-rate change | Not identified because links 3 through 5 are missing |

The task proposed links 2 and 3 as a direct measure.  That is too strong.
The calculation can ask the fixed human model for the risk assigned to each
successor board.  It cannot observe the counterfactual response of the same
human after the learner changes the predecessor action.  Calibration of
observed paths does not repair that missing causal comparison.

All safety statements are positional-only and use `A_pos`.  Malom's board
query omits repetition and no-progress history, so none of these actions is
called `A_allow`.

## Predictive safe-successor uplift

For each learner decision, the frozen full model scores the opponent's next
choice set after every action in `A_pos`.  The primary risk is opponent
positional D-to-L loss when the learner parent is D.  Other parent tiers are
assigned zero for this primary estimand; W-tier losses are retained only in
the separate secondary endpoint.

The three references mean:

- `uniform_A_pos`: no preference among positional-safe actions;
- `geometry_A_pos`: the frozen geometry-only human-choice distribution,
  restricted and renormalized to `A_pos`; and
- `human_frequency_A_pos`: the frozen full-panel human-choice distribution,
  restricted and renormalized to `A_pos`.

The table reports equal-player means on parent-D decisions, with median and
90th percentile in parentheses.  “Corrected” applies a fold-external
calibration map after selecting the raw-risk argmax.

| Reference | Raw uplift | Corrected uplift | Retained |
| --- | ---: | ---: | ---: |
| Uniform | 0.15484 (0.10753, 0.41085) | 0.08295 (0.04609, 0.23473) | 53.57% |
| Geometry | 0.14209 (0.08773, 0.38572) | 0.07693 (0.03902, 0.21688) | 54.14% |
| Human frequency | 0.13365 (0.08398, 0.37020) | 0.07255 (0.03680, 0.20722) | 54.28% |

When all 292,192 decisions remain in the denominator and non-D primary
uplift is zero, the corrected equal-player means are 0.05222, 0.04840, and
0.04579 respectively.  Their player-bootstrap intervals are
`[0.05042, 0.05409]`, `[0.04672, 0.05003]`, and
`[0.04415, 0.04741]`.  These intervals quantify predictive-sample
uncertainty; they are not causal confidence intervals.

Exactly 33,369 decisions, 11.4202%, have `|A_pos| = 1`.  They remain in every
applicable denominator with exactly zero raw and corrected uplift.  Under the
human-frequency reference, corrected all-decision uplift grows with
cardinality:

| `|A_pos|` | Decisions | Mean | Median | P90 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 33,369 | 0 | 0 | 0 |
| 2 | 22,579 | 0.01943 | 0 | 0.05850 |
| 3-4 | 57,663 | 0.03629 | 0.00184 | 0.11603 |
| 5-8 | 81,961 | 0.04996 | 0 | 0.15666 |
| 9+ | 96,620 | 0.07109 | 0.03097 | 0.21563 |

The same human-frequency endpoint is highest in placement:

| Phase | Decisions | Mean | Median | P90 | Zero share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Placement | 113,923 | 0.06594 | 0.02782 | 0.20502 | 33.30% |
| Movement | 167,807 | 0.02679 | 0 | 0.08777 | 55.37% |
| Flying | 10,462 | 0.04486 | 0 | 0.14630 | 66.23% |

The all-tier raw endpoint is reported separately in the manifest.  Its
equal-player means are 0.13918, 0.12725, and 0.11756 for uniform, geometry,
and human-frequency references.  It is not substituted for D-to-L.

## Calibration and winner correction

On 199,234 observed parent-D out-of-fold decisions, the uncorrected model has
calibration intercept -1.23560, slope 0.80859, Brier score 0.11022, and ECE
0.12480.  Exact zero risk contains zero events and remains exact zero; no
prior manufactures an effect.

The equal-player reliability bins show substantial overprediction:

| Bin | Decisions | Mean predicted | Observed D-to-L |
| ---: | ---: | ---: | ---: |
| 2 | 60,086 | 0.00153 | 0.00067 |
| 3 | 22,390 | 0.05733 | 0.04407 |
| 4 | 20,859 | 0.12086 | 0.03755 |
| 5 | 21,287 | 0.19168 | 0.08512 |
| 6 | 19,968 | 0.29829 | 0.11411 |
| 7 | 19,300 | 0.41105 | 0.13843 |
| 8 | 18,318 | 0.55924 | 0.25130 |
| 9 | 17,026 | 0.78861 | 0.51120 |

Bins 0 and 1 are empty because repeated exact-zero risks create duplicate
weighted-quantile boundaries.  All five fold-external calibration slopes are
positive, from 0.77130 to 0.85598, and all fits converge in six iterations.

The correction shrinks values after the raw-risk winner is selected.  It
therefore addresses global overconfidence but not the winner's curse caused
by residual, action-specific ranking error among hypothetical successors.
Calling the corrected values unbiased inducement effects would be false.

## Policy shift and support

Relative to each reference, the selected argmax shifts probability mass
toward the two highest observed risk bins.  Jensen-Shannon divergence is
0.05019 nats for uniform, 0.04200 for geometry, and 0.03736 for human
frequency.  Neither selected nor reference policies place mass in the two
low-support bins.  Selected mass above the observed scalar risk range is
only 0.000545%; reference mass above it is at most 0.000222%.

This rules out one narrow explanation: the large predictive contrast is not
mostly created by low-support reliability bins or extrapolation beyond the
observed scalar risk range.  It does not establish counterfactual support at
the state-action level, and it cannot quantify how a deployed policy would
change future state visitation.

## Log loss is not an argmax-value metric

On 199,234 parent-D decisions, the full-risk and geometry-risk argmax actions
agree for 81.74% of equal-player mass.  The full-risk regret of taking the
geometry argmax is 0.01614.  Player-level association between paired log-loss
improvement and argmax regret is effectively zero: Pearson -0.02559 and
Spearman -0.01422.

Mean log-loss improvement is 0.13316 where the argmax differs and 0.16789
where it agrees.  Thus the observed 0.160192-nat log-loss improvement has no
defensible proportional mapping to safe-action argmax value.  D-to-L
top-versus-bottom discrimination is also not the same estimand.  Both
“equivalent threshold” fields are deliberately left null.

This directly explains why the old 0.01-nat product interpretation was
unsupported: predictive likelihood and the value of the selected safe action
answer different questions.

## Redemption and the perfect-redemption ceiling

Redemption cannot be estimated from human-versus-human records.  It requires
learner-versus-human evidence after a positional D-to-L transition.  The
retained candidate's 42.7866% score against fixed-node Sanmill is neither a
human redemption rate nor a transportable substitute and is not used in the
calculation.

The manifest reports only a deliberately favorable ceiling: every induced
D-to-L event is worth the full 0.5-point draw-to-loss swing and is converted
perfectly.  Even Malom theoretical distance is not used as a rate proxy.  Its
stored distance is sector-relative and cannot supply a globally calibrated
probability that this learner defeats this human under full rules.

Applying `0.5 * mean(min(1, sum uplift))` to observed paths produces corrected
numbers of 39.10, 38.08, and 36.75 score points per 100 games for the three
references.  Their size is evidence that this is an extremely loose ceiling,
not evidence of a plausible product gain.  It assumes away causal
inducement, policy-induced visitation, dependence, adaptation, and imperfect
redemption.  It is explicitly neither a rollout nor a product estimate.

## Reverse product thresholds

The frozen product tiers are +0.5, +1.0, and +2.0 score points per 100 games.
The exploratory paths contain 15.5691 parent-D opportunities per game and
color.  Under perfect redemption, a necessary single-step uplift is

`2 * score_gain_per_game / 15.5691`.

| Product tier | Necessary single-step uplift | D discrimination equivalent | Log-loss equivalent |
| ---: | ---: | --- | --- |
| +0.5 points / 100 | 0.000642 | Not identified | Not identified |
| +1.0 point / 100 | 0.001285 | Not identified | Not identified |
| +2.0 points / 100 | 0.002569 | Not identified | Not identified |

These are necessary, not sufficient, thresholds under an upper-bound
assumption.  Any imperfect redemption, fewer policy-visited opportunities,
adaptation, or dependence makes the required causal uplift larger.  Because
the causal uplift itself is not identified, comparing these small thresholds
with the 0.04579 to 0.05222 predictive plug-in means would be invalid.

The 487-player arm therefore cannot adjudicate any tier.  This is not a power
calculation: increasing observational player count cannot identify the
missing counterfactual, multi-step, or redemption mappings.

## What evidence would unblock the chain

The missing links require evidence of three distinct kinds:

1. action-specific counterfactual response evidence for safe actions, with a
   design capable of separating action choice from successor-state risk;
2. policy-level visitation evidence that measures repeated opportunities,
   dependence, and opponent adaptation; and
3. learner-versus-human evidence measuring conversion after positional
   D-to-L under full rules.

Those needs involve games or intervention.  This round neither designs nor
authorizes them.  It also does not authorize a new research question based
on the numerical thresholds above.

## Bias and claim boundary

The retained source is not a random sample.  F0-D0 excluded 1,751 games with
only 35 draws while retaining 92,789 games with 26,157 draws.  Another
54,923 games lack independently verifiable terminal basis.  UI orientation,
time control, and exact source rules variant are unrecoverable.  Conclusions
are restricted to the observed PlayOK-like source and do not transport to
the product UI or a new population.

State novelty is not claimed.  All contamination and generalization
boundaries remain player- and game-based.  F0-H0 remains stopped, the prior
readiness result remains `B_not_ready_fail_closed`, and no confirmation,
E0, F0-H1, T0, game, training, reward, promotion, deployment, publication, or
release authority follows from this derivation.

## Verification

Task-scope Ruff passes.  The complete feature-deviation group passes 53
tests.  The B2 freeze file passes all eight tests, including two guards that
raise before any final-test raw record, decision, or feature producer can be
called.  The mandatory Malom, DB-teacher, and label-provenance group passes
103 tests plus 498 parameterized subtests.

## Machine record

The complete distributions, calibration fits, bootstrap intervals, support
mass, query ledger, input hashes, claim boundary, and unique C reason are in
[the machine manifest](human-feature-deviation-product-conversion-manifest-2026-08-15.json).
Independent record tests recompute both plan and result identities and pin
the decision, zero-access ledger, singleton handling, calibration direction,
null equivalences, and resource limits.
