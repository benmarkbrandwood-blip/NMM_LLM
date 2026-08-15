# Featureized human deviation: design review and exploration

Date: 15 August 2026

## Outcome first

The proposed question is worth retaining only after a material narrowing.
The defensible question is not whether a feature model discovers a causal
human trap.  It is whether a fixed, low-dimensional conditional-choice model
has out-of-player predictive value for complete human actions and positional
tier loss, followed by a separate one-ply positional steerability necessary
condition.

This is a new estimator and does not reopen F0-H0.  The corrected F0-H0 result
identity remains
`8bd2da62785e9c8cda0a055e98213959cbdf8f88aa860384171f00f4f39c6bdc`,
and its `stop_condition_triggered` decision remains binding.  In particular,
the failed exact-state and `ring16` frequency route is not rerun, relaxed, or
reinterpreted here.

The narrowed question was frozen before player membership or feature
statistics under plan identity
`04177a73ca5b9a1aa8cc8352477f2050759e6a742cee049f1191d3064ae5d662`.
The plan file SHA-256 is
`5909cb86c5c21638bb6361b3035e5993e75d5edd26d41af24606dd9842e591c7`.

A fixed 128-game exploration-only screen did not immediately reject the
feature direction.  It did not fit a model and is not confirmatory evidence.
No research-confirmation or official selection, confirmation, or final-test
content was opened.  Confirmation is deliberately not executed in this
round.

## Facts independently checked

The design and implementation were checked against the frozen documents,
current code, and manifest identities rather than accepting prose alone.

- The official B2 membership remains 36,949 train, 887 selection, 386
  confirmation, and 847 final-test games under membership identity
  `06c49903baf76ee7787af8333058e164cb54ea7a27035a1371747d6000d07b0b`.
- F0-H0 failed because supported `ring16` coverage was 20.16%, player Gini was
  0.804, and Kish effective player support was 177.7.  Positional choice space
  was not the failure: 88.69% of its sampled decisions had `|A_pos| > 1`.
- `ai/malom_db.py` exposes a position-only tablebase query.  The full
  comparator used here is the existing complete-action inventory in
  `learned_ai/evaluation/human_f0h0_b2_train_screen.py`; it preserves capture
  identity and labels each legal atomic action.  Neither path carries
  repetition multiplicity or the no-progress clock, so the safe set remains
  `A_pos`, never `A_allow`.
- `learned_ai/sentinel/db_teacher.py` provides useful comparator and
  provenance patterns, but its DB-teacher target is not a human conditional
  choice model.
- `learned_ai/models/scaffolded_encoder.py` mixes visible board features with
  Sentinel, heuristic, value, GapNet, lookahead, and human-frequency signals.
  It also has neutral fallbacks for optional components.  Reusing that 134- or
  138-wide representation would destroy the intended low-dimensional,
  human-visible estimand and could leak the rejected aggregate frequencies.
- The existing GapNet builder predicts a state-level gap between the best
  Sentinel-plus-heuristic composite quality and aggregate
  human-frequency-weighted quality.  It deduplicates by state, has no
  player-isolated membership, and is not a model of a complete legal choice
  set.  Its target and existing artifact therefore cannot be reused for this
  question.

The reusable pieces are raw strict replay, complete legal atomic choices,
current Malom provenance checks, the full positional comparator, and selected
named visible feature logic.  A dedicated conditional-choice representation
is required.

## Revised formal problem

The frozen question is:

> Within the frozen B2 train source domain, can a fixed ten-term,
> actor-normalized conditional-choice model improve average-unique-player
> prediction of complete human action choices on unseen source-domain players,
> and can its predicted positional tier-loss mass discriminate immediate
> D-to-L choices?  Separately, does a one-ply `A_pos`-preserving predecessor
> choice have enough positional-only leverage to change that predicted
> exposure?

This wording is preferable to “identify human mistakes” for four reasons.

1. Prediction on unseen players is observable; causal inducement is not.
2. Complete legal choices, including capture identity, are the correct choice
   units.  Unchosen moves are alternatives, not independent negative rows.
3. A positional W/D/L loss is objectively a position-tier loss, while
   within-tier regret is a different difficulty measure.
4. One-ply steerability is only a structural necessary condition.  It does
   not establish repeated-play reachability, full-rule safety, product-user
   transport, or conversion to game score.

The new problem is aligned with the product objective as a rejection-oriented
precursor: it can show whether visible human tendencies and safe local leverage
exist.  Passing it would still not authorize E0, F0-H1, T0, a reward change,
training, or a product claim.

## Review of the three proposals

### Proposal 1: sparsity curves -- reject as proposed, retain a narrower curve

Recomputing exact-state or `ring16` support and extrapolating to the failed 80%
F0-H0 threshold would reopen the estimator that has already been rejected.
That is not allowed and is not scientifically useful here.

If a fixed feature model later passes its basic support screen, a model-specific
learning curve can be useful.  It must keep two sampling meanings separate:

- player-uniform sampling estimates how performance changes as genuinely new
  independent source-domain players are added;
- whole-game sampling estimates more traffic from the existing activity mix
  and can add many decisions while barely increasing independent-player
  information.

Kish effective support and predictive loss must also remain separate curves.
No single functional form is trustworthy for a saturation extrapolation.  The
frozen guard therefore requires power-law, Michaelis-Menten, and logarithmic
forms to backtest, prohibits extrapolation beyond twice observed player
support, and reports the requirement as unidentified if the projected scales
differ by more than twofold.  Coverage and effective-player growth must never
be forced into one curve.

### Proposal 2: two-factor heuristic deviation -- modify

The intuition is sound: a heuristic matters only when its recommendation can
conflict with `A_pos` and humans sometimes follow the conflicting choice.  The
literal product
`P(disagreement) * P(follow | disagreement)`, however, becomes ambiguous when
several moves tie for a heuristic maximum and the two probabilities use
different conditioning sets.

For each named score the frozen design instead defines `H_max` as all legal
actions tied for the maximum and reports:

- possible conflict: at least one member of `H_max` lies outside `A_pos`;
- strict conflict: every member of `H_max` lies outside `A_pos`;
- mixed conflict: `H_max` contains both safe and unsafe actions;
- direct joint unsafe follow: the observed action is both in `H_max` and
  outside `A_pos`.

These rates are kept separate rather than multiplied.  The direct joint rate
is the closest descriptive analogue to the proposed product, without silently
changing denominators.

The original three heuristics are not an adequate or clean basis by
themselves:

- closing a mill and blocking an immediate mill are retained;
- “do not let my board-piece count fall below the opponent” is not a distinct
  current-action preference in this atomic representation.  A legal action
  does not remove the actor's own piece.  Within a choice set, material balance
  changes mainly when closing a mill triggers a capture, making it structurally
  close to the mill-closing feature;
- double-mill creation, latent mill creation, high-connectivity occupation,
  own mobility, and capture-target threat are important omitted candidates;
- flying is treated as a phase stratum rather than assumed to share the same
  behavior as placement and movement.

The frozen ten-term dictionary therefore contains three geometry controls
(`source_degree`, `destination_degree`, `capture_degree`) and seven tactical
terms (`closes_mill`, `blocks_immediate_mill`, `creates_double_mill`,
`new_own_potential_mills`, `own_mobility_delta`, `material_balance_after`, and
`captured_opponent_threat_lines`).  It is actor- and color-normalized.  Phase,
color, and current W/D/L tier are reported as strata.

The material term is retained in v1 as a frozen observable diagnostic, not as
an independently interpretable causal factor.  The pilot's identical support
counts for `closes_mill` and `material_balance_after` reinforce the expected
collinearity.  Ridge regularization keeps prediction well-defined, but their
individual coefficients must not be interpreted independently.

### Proposal 3: safe steerability -- accept only as a necessary condition

The tablebase alone cannot prove that a learned policy can repeatedly reach a
target state or cause a human response.  It can still support a useful local
screen without starting games:

1. at an observed learner predecessor, enumerate every complete learner action
   in `A_pos`;
2. settle each action, including capture;
3. enumerate the next human choice set;
4. use the already frozen human model to sum predicted mass on position-tier
   losing human actions; and
5. measure the range of that unsafe mass across learner-safe successors.

This is a one-ply positional leverage bound.  It abstains for phase support
gaps and feature values more than four exploration standard deviations from
the fitted domain.  It may run only after every predictive confirmation gate
passes.  A positive result is neither causal inducement nor `A_allow` safety.

## Answers to the six open questions

### 1. Player-clustered uncertainty is mandatory

Decisions from one player are not independent.  Point estimates may be
decision-weighted for the natural-traffic estimand, but uncertainty must be
clustered by source-domain player.  The confirmatory primary estimand first
averages loss within each actor and then gives each actor equal weight; a
2,000-replicate player bootstrap supplies intervals.

Kish 177.7 is not a universal sample size for every new statistic, but it
shows why millions of plies do not imply millions of independent observations.
For a bounded player-level rate, 178 equally informative players have a
worst-case normal 95% half-width of roughly 7.3 percentage points.  The frozen
hash assignment places 551 keys on the research-confirmation side, but only
290 of them occur in surviving same-arm games.  Before further evaluability
loss, that gives an optimistic worst-case player-level half-width of about 5.8
points.  Actual precision must come from the observed across-player
distribution, not from a ply-level binomial interval.

### 2. Reweighting cannot recover an unknown product population

Equal-player weighting is worthwhile as the primary source-domain estimand
because it prevents a few heavy users from defining the answer.  It also
upweights sparse players and does not transform PlayOK-like users into product
users.  Natural-game weighting is therefore reported secondarily.

No product-user activity, skill, UI, or time-control target distribution is
available, so product-population importance weights are not identifiable.
The direction of product shift cannot be inferred from these data.  In the
pilot, 137 observed players had a decision-weighted Kish support of only 72.85,
illustrating the information loss caused by activity concentration.  Future
reports must publish both weightings and their effective support, not choose
the more favorable result.

### 3. Deviation, mistake, and trap value are different

An action outside `A_pos` is a position-tier loss under the trusted Malom
labels.  W-to-D, W-to-L, and D-to-L events must remain separate.  It is still
not automatically a strict-rule loss because position queries omit repetition
and no-progress history.

Within-tier comparator regret can alter practical difficulty and may matter
for traps in theoretically drawn positions.  It must remain a secondary
continuous difficulty or steerability measure.  Combining it with W/D/L loss
would let distance conventions manufacture an “error” without a tier loss.

### 4. A fixed heuristic panel can miss real patterns

Hypothesis-driven terms reduce fishing but impose omission risk.  The valid
compromise is a two-track process: the frozen panel supplies confirmation;
residual pattern discovery is allowed only inside research-exploration and
must use player-fold stability rather than state identities.  Any newly
discovered feature requires a v2 preregistration and a new untouched
player-isolated confirmation resource.  It cannot be confirmed on the data
that suggested it.

This v1 round does not perform residual discovery or alter the ten terms after
seeing the pilot.

### 5. The fastest rejection screen is opportunity before model fitting

Before fitting a conditional-choice model, ask whether named features vary
inside legal choice sets, whether their maxima ever conflict with `A_pos`, and
whether enough independent players both encounter and follow such conflicts.
Also check that D-to-L events exist across enough players.  A feature that is
constant inside choices, never conflicts, or is supported by only a few
people cannot support the product mechanism regardless of model sophistication.

The 128-game pilot implements this screen.  It is more directly decisive for
the new estimand than repeating F0-H0 coverage curves.

### 6. GapNet cannot supply the new target

GapNet estimates a board-level opportunity gap built from Sentinel and a
handwritten heuristic against aggregate human frequencies.  It neither
predicts which complete action a new player will choose nor isolates players,
and its aggregate HumanDB tables do not retain per-game membership.  It also
mixes the behavior signal being studied into the target.

Directly reusing GapNet would therefore answer the old aggregate-state
question under another name.  The current GapNet artifact and target are not
used.  Only generic model plumbing, replay utilities, and current comparator
provenance patterns may be reused after separate review.

## Frozen exploration/confirmation separation

Only official B2 train membership was partitioned.  A player key is assigned
by the first byte of
`SHA-256("human-feature-deviation-player-split-v1-2026-08-15" NUL player_key)`:
bytes 0 through 63 select research-confirmation and the remainder select
research-exploration.  Games whose two players fall on opposite sides are
discarded.

The resulting split identity is
`fa74650c1afdffeb0d30f334b2b7859538f81b0e502c17a64092bfdcd99a06dd`;
its file SHA-256 is
`931e55f1d49ceef715783146dd94b29d972f5cc70cbd6d43d142a59e8338b62c`.

| Partition | Games | Logical plies | Assigned keys | Keys in its games |
| --- | ---: | ---: | ---: | ---: |
| research-exploration | 19,257 | 905,648 | 1,665 | 1,416 |
| research-confirmation | 2,751 | 129,697 | 551 | 290 |
| cross-player discard | 14,941 | 707,071 | n/a | 1,302 |

The two assigned player sets have zero overlap.  Assigned-key counts include
players whose only games cross the split and are therefore discarded; the
smaller participating-key counts are the relevant upper bounds for model
support.  They were recomputed from F0-D0 manifest metadata without opening
game content.  The 40.44% game discard is the cost of genuine player isolation
in a two-player corpus and must remain visible in any transport claim.  A
fixed 128-game pilot was selected from research-exploration by a second
session-ID hash before any feature statistic.  Its session identity is
`d63d6e2ccf898e1167fa731ac94d3f8195a0a9e23467c7edacdcaec69cdb0fc4`.

Research-confirmation content remains unopened.  Official selection,
confirmation, and final-test remain unopened as well.

## Frozen confirmatory contract

The future primary model is an L2-regularized conditional multinomial logit
over each complete legal choice set.  The nested contrast is the three-term
geometry control against the full ten-term model.  Malom is applied after
prediction to label tier loss; it is never a predictor.

The main independent gates are:

- at least 200 evaluable confirmation players, 10,000 decisions, 50 D-to-L
  events, and 50 players with D-to-L events;
- at least three tactical terms varying in at least 1% of choices and supported
  by at least 100 players each; at least one tactical possible-conflict rate of
  2%; and 50 players with a direct unsafe follow;
- average-unique-player choice log-loss improvement of at least 0.01 nats,
  with its player-bootstrap 95% lower bound above zero;
- positive Brier improvement for D-to-L prediction and a top-versus-bottom
  risk-quintile D-to-L difference of at least two percentage points, with the
  lower bound meeting that floor;
- positive point improvements in placement and movement, no color-specific
  adverse log-loss change above 0.005 nats, and descriptive-only flying results
  when fewer than 50 flying players are evaluable; and
- at least 99% Malom decision coverage, with missing labels abstained rather
  than imputed.

The later one-ply necessary condition requires a predicted unsafe-mass range
of at least two points, a player-clustered lower bound showing that at least 1%
of eligible predecessors meet that range, and support from at least 100 players
and 500 games.

The 0.01-nat prediction floor is approximately a 1% per-choice likelihood
gain.  The two-point discrimination floor is independently chosen for this
new estimand and is not the failed F0-H0 one-point product bound.  Failure of
any gate rejects this feature direction.  Passing every gate authorizes only
a later design decision.

## Exploration-only result

Result identity:
`c489dca91c00569491d2b50a879bd014081e0109e0899afcc2bf2f13d584d7d6`.
File SHA-256:
`9b4cb41bf9fdb174d3556e021117da2143820da6964b8a5422e6b4afdfba611d`.

The pilot opened 128 research-exploration games containing 6,140 decisions
from 137 player keys.  All 6,140 decisions were covered.  It issued 81,463
Malom queries in 53.6013 seconds, or 1,519.8 queries/second.  There were no
abstentions.  `|A_pos| > 1` held for 5,427 decisions (88.39%).  Observed tier
losses were 128 W-to-D, 12 W-to-L, and 218 D-to-L; these are unclustered pilot
counts, not confirmatory rates.

The following table is exploratory.  Variation is a share of all 6,140
covered choices.  Conflict is a share of choices where that feature varied.

| Feature | Varies | Possible conflict | Direct unsafe follow | Players |
| --- | ---: | ---: | ---: | ---: |
| blocks immediate mill | 23.49% | 17.06% | 32 | 27 |
| capture degree | 10.29% | 40.19% | 8 | 8 |
| captured threat lines | 9.01% | 50.99% | 2 | 2 |
| closes mill | 10.29% | 54.75% | 20 | 16 |
| creates double mill | 0.00% | n/a | 0 | 0 |
| destination degree | 93.00% | 36.80% | 131 | 66 |
| material balance after | 10.29% | 54.75% | 20 | 16 |
| new own potential mills | 73.21% | 47.41% | 84 | 50 |
| own mobility delta | 62.43% | 31.62% | 62 | 39 |
| source degree | 59.27% | 54.99% | 71 | 49 |

This pilot rules out the simplest global objection that every named feature is
constant or never conflicts with positional safety.  It does not establish
out-of-player prediction, clustered precision, or steerability.  It also
identifies two cautions: double-mill creation had zero variation in this
sample, and mill closure and post-action material balance are structurally
redundant in many atomic choice sets.  Zero events remain point estimate zero;
no prior manufactures an effect.

## Why confirmation was not executed

The frozen plan explicitly sets confirmation execution to false.  More
importantly, this round has not implemented or independently verified the
conditional-logit optimizer, player-weighted loss, player bootstrap,
calibration endpoint, or one-time confirmation access ledger.  Opening the
confirmation arm (551 assigned keys, 290 present in surviving games) before
those components and their artifact hashes are frozen would contaminate the
only player-isolated confirmatory resource.

The correct current disposition is therefore:

- the narrowed research question survives the exploration-only fast-reject
  screen;
- it has no confirmatory result and no positive product claim;
- F0-H0 remains stopped;
- a later execution requires a separate immutable implementation and access
  freeze; and
- no E0, F0-H1, T0, reward, game, training, promotion, or release work is
  authorized.

## Claim boundary and inherited bias

All labels are `A_pos` and positional-only.  The evidence is limited to the
observed PlayOK-like source domain.  UI orientation, time control, and exact
rules variant cannot be recovered.  The F0-D0 history filter is non-random:
1,751 excluded games contain only 35 draws, while 92,789 retained games contain
26,157 draws.  Another 54,923 games lack independently verifiable terminal
basis.  No result is transported to product UI users.

The access ledger records zero reads from research-confirmation, official
selection, official confirmation, final-test, HumanDB, or source pool
`2eb04f54`; zero games, searches, neural model loads, training updates, and
database writes also occurred.

## Verification

Task-scope Ruff check and format check pass for the new evaluation module,
freeze and exploration runners, and focused tests.  The focused feature tests,
reused F0-H0 comparator tests, and mandatory Malom, DB-teacher, and
label-provenance group pass 124 tests and 498 parameterized subtests.  The
tracked-artifact test independently recomputes the plan, split, and result
identities and asserts zero protected-partition and source-pool reads.
