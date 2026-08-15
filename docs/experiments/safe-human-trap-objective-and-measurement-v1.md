# Safe human-trap objective and measurement design v1

Date: 2026-08-14

Updated: 2026-08-15

Status: `f0_h0_stop_condition_triggered_no_later_gate_authority`

Authority: `product-owner-direct`

This document defines the product objective that replaces fixed-node Sanmill
score as the ultimate purpose of the training program. It is a design record,
not an experiment authorization. It permits no new game, search batch,
training update, database write, checkpoint change, promotion, deployment,
publication, or release.

## Product objective

Within an independently verified theory-preserving action domain, maximize the
probability that a real target human opponent makes a consequential error, and
then convert that error into factual match utility. This is **safe human-trap
ability**.

Absolute playing strength remains a necessary constraint. It is not the
project's differentiating objective: Malom already supplies perfect positional
information, and deeper Sanmill search supplies generic tactical strength.
The project-specific opportunity is to learn which theory-safe positions and
continuations are difficult for actual humans.

F0-D0 now freezes the current raw-corpus identity as
`4c54d55209543e70edaeb33cb1dea25d2707312c3781580ba326ae35882dea29`.
It reconciles 95,389 human-game JSONL occurrences to 94,540 unique imported
sessions and identifies 92,226 behavior-replay-eligible sessions. The active
HumanDB's 94,429 is stale, duplicate-path-inflated metadata with an
unattributed six-game drift. The previously quoted 95,221 belongs exactly to
a different archived 95,785-path inventory and is not a current-corpus
acceptance denominator. Preserve the
[F0-D0 evidence](../evidence/f0-d0-human-raw-reconstructability-2026-08-14.md)
and its machine-readable manifest.

## Governing definitions

This design adopts the existing v5 definitions in
`docs/v5/oracle-and-rules-spec.md`,
`docs/v5/human-data-and-statistics-plan.md`, and
`docs/v5/training-research-plan.md`.

Let `S` be a complete `DecisionState`, including strict history required for
repetition and no-progress rules.

- `A_legal(S)` is the full legal settled-action set.
- `A_pos(S)` preserves the best positional W/D/L tier under the complete
  Malom comparator.
- `A_allow(S)` is the full-rule permitted set after history, W-liveness, and
  the separately named L-mode survival contract are applied.
- `q_ref` is the accepted non-human-targeted reference policy inside the same
  permitted set.

Malom's board-only query does not by itself construct `A_allow`; it does not
carry repetition or no-progress history. Until the full-rule proof/history
gate is accepted, a study may claim only positional safety and must use
`A_pos`, not silently relabel it `A_allow`.

For an opponent reply state `T(S,a)` and target condition `c`, define a
theoretical error as an opponent action that lowers the best available
theoretical tier. W-to-D, W-to-L, and D-to-L events are reported separately.
Within-tier comparator regret is a secondary difficulty measure, not silently
merged with tier loss.

A candidate action is a qualified safe inducement only when all of the
following hold:

1. the learner action is in the declared permitted set;
2. the learner does not make the first theoretical downgrade;
3. at least one correct opponent reply remains;
4. independent-player and support floors pass and the state is not OOD;
5. a simultaneous lower confidence bound on opponent error probability is at
   least a preregistered positive `p_min`;
6. the lower confidence bound improves over `q_ref` by at least a
   preregistered `delta_induce`; and
7. conditional conversion and factual loss-rate gates pass.

Entering a theoretically inferior state and hoping for a later human error is
not a safe trap. A nonzero model softmax, a rare reply, or a one-step
`Trap_1` value is not enough.

## Measurement hierarchy

Safety is a hard gate, not a reward tradeoff. Among candidates that pass it:

1. the primary product endpoint is conservative factual match score against a
   frozen, independently evaluated target-human policy under a frozen horizon
   and runtime;
2. the main mechanism endpoint is safe-inducement lift,
   `P(opponent first theory downgrade | candidate) - P(... | q_ref)`;
3. conditional conversion after the opponent's first downgrade is a required
   gate and separately reported endpoint; and
4. first-error timing, reply-set size, exact comparator regret, termination
   reason, and action entropy are diagnostics.

This hierarchy prevents a model from maximizing harmless or unconvertible
mistakes while losing actual product utility.

Every human-behavior split is made at complete-game and, where recoverable,
player level. Raw UI orientation and color are retained. Train, selection,
one-time confirmation, and untouched final-test membership are frozen before
candidate inspection. Unsupported conditions abstain rather than inherit a
global probability.

## Low-cost complexity proxy

A low-node/high-node Sanmill comparison is a useful first proxy, but it is not
human-trap evidence and cannot select a release model by itself.

The proposed diagnostic freezes one state set and, for every candidate in the
declared safe set, evaluates the opponent reply state with both a low-node and
high-node deterministic Sanmill configuration. It reports:

- low/high selected-reply disagreement;
- whether the low-node reply downgrades while the high-node reply preserves;
- full Malom comparator regret of each selected reply;
- normalized low/high score and rank gaps; and
- coverage, phase, color, legal-reply count, and abstention.

The falsifiable proxy hypothesis is: compared with `q_ref`, a candidate that
manufactures decision complexity increases the rate at which low-node search
chooses a theoretically worse reply while high-node search remains safe,
without any learner-side theoretical downgrade.

A positive result means only “engine-budget-sensitive complexity.” A negative
result rejects this proxy for the tested candidates. Neither result estimates
human error probability.

## Required feasibility order

No trap reward or training modification is allowed before these gates:

1. `F0-D0`: replay raw records and establish complete-history, player, source,
   result, and condition recoverability.
2. `F0-H0`: run a read-only rejection screen for independent support,
   modifiable-state reach, concentration, and an upper bound on available
   product effect.
3. `E0`: accept deterministic rules, complete comparator, and the exact
   positional/full-history claim boundary.
4. `F0-H1`: freeze one minimal HumanPolicy and quantify supported headroom
   inside the accepted permit set.
5. `T0-H-pilot`: run a separately authorized non-promotable direct comparison
   only if the feasibility screens permit it.

## F0-D0 completion -- 14 August 2026

F0-D0 completed read-only under manifest identity
`bf7404d1f090073a1b36635b89d329e7011140d48e4fb3b3076efd7e55b5bca7`.
Of 94,540 unique sessions, 92,789 have strict recoverable histories, all have
both source-supplied player identifiers, 92,226 contain at least one eligible
human decision, and 37,866 have a recorded result that agrees with an
independently replayed strict terminal. There are zero result disagreements.

This is partial source-domain recoverability, not full five-dimension
recovery. Explicit import batch, upstream source-file identity, UI
orientation, time control, exact source rules variant, and every source-side
termination basis are absent. The 1,751 strict legality failures and 54,923
nonterminal results remain individually attributed in the manifest without
invented causes.

## F0-H0 v1 supersession and corrected split retest -- 15 August 2026

F0-H0 v1 froze plan identity
`95a802625867906ab453ed7a52bbba1e0202b08473b10f897ba81c87fb59d530`
and result identity
`714627f8be20bc45a267c97752171644040fc1273a24f82a570a7cb83512fe82`.
Its zero-cut stop decision is now
`superseded_by_corrected_split_design`.  Requiring every complete game and
every game of each player to stay in one partition algebraically forces every
connected component into one partition.  That rule tests only whether a
zero-cut split exists and does not measure corpus-specific split scale.  The
v1 plan, membership, result, and narrative remain unchanged as historical
records.  See the
[supersession correction](../evidence/f0-h0-v1-supersession-correction-2026-08-15.md).

The corrected measurement was frozen under plan identity
`e1d2241cc23da1227fde7a3f84d2ff4c43a4167c2020d521abbc9f3eee1f833c`
before any corrected statistic was calculated.  Result identity
`cbfa6d43fa31e9644bae169e6b6d42232aa008e54921c96a46fbdddb73a95931`
confirms one 4,994-player component and measures three alternatives without
selecting one: player cuts with discarded cross-boundary games, calendar
holdouts, and player-owned decisions with explicit trajectory and `ring16`
leakage.  See the
[corrected split evidence](../evidence/f0-h0-corrected-split-feasibility-2026-08-15.md).

This is `completed_measurement_only_no_split_selection`.  It makes no
feasible/infeasible or continue/stop decision.  The four F0-H0 scientific
dimensions remain unrun, and final-test membership has not been selected or
opened as an analysis population.  No later gate, game, search, model load,
Malom query, training change, database rebuild, alternative-data
substitution, or source-pool access is authorized by either F0 result.

## F0-H0 Design B supplement -- 15 August 2026

The Design B support and second-level split supplement was frozen under plan
identity
`889ccfcc407def9b7c2b4f3058611566e1bcb541976c42ed286d449dc67d633a`
and pushed before its result was calculated.  Result identity
`a45fbfa0c472f86f03596b0618c799c4e0fb522bcfaa9b431efc904e838301a2`
records the completed read-only measurement.

The March strong pool contains 4,577 games, 1,245 source player keys, 207,044
decisions, and 1,973 strict-outcome games.  Its induced graph has 31
components, but the giant contains 1,178 players and 4,465 games.  One
measured 50%/25%/25% player subdivision discards 737 cross-role games.  The
three frozen time-pair candidates produce strong selection, confirmation, and
final role scales of 887/386/847, 1,686/773/22, and 2,535/469/58 games.

A same-size random whole-game control measured decision-weighted `ring16`
overlap of 37.55% and 57.31%, compared with 34.32% and 53.60% for the coarse
March Design B split.  This shows that recurring game states create a large
baseline overlap.  It does not make the prior four-role Design C result
equivalent to the two-role random control; partition count, support, and
same-game trajectory exposure differ.  See the
[Design B supplement evidence](../evidence/f0-h0-design-b-supplement-2026-08-15.md).

This is
`completed_measurement_only_no_final_split_selection`.  The role names in the
candidate measurements are hypothetical only.  No final membership is
frozen, no support threshold or split is selected, and no F0-H0 scientific
dimension has run.  The supplement authorizes no later gate or state-changing
operation.

## Binding B2 split and state-novelty addendum -- 15 August 2026

The product owner selected Design B2 with cuts at 1 April and 1 May 2026.
The freeze contract was sealed before membership generation under plan
identity
`a4dc271d00a36394d4e5b61751f7536cf3e869cb90136fbe7bedd6016c6acb30`.
The F0-D0 manifest then reproduced exactly 36,949 train, 887 selection, 386
confirmation, and 847 final-test sessions without opening any raw game.
Official membership identity
`06c49903baf76ee7787af8333058e164cb54ea7a27035a1371747d6000d07b0b`
freezes those four session-ID sets.  The three test-segment player-key sets
have 295, 160, and 322 keys and zero pairwise intersections.  Final-test
content remains sealed behind a fail-closed accessor until a separate
one-time authorization.

This addendum does not run or pass F0-H0.  Independent support, modifiable
state reach, concentration, and the product-effect upper bound remain
uncomputed.  Only train, selection, and confirmation may be characterized in
the freeze round; final-test may expose only its count and membership hash.

The permitted nonfinal characterization subsequently completed under result
identity
`183a39ab29ddfbec76a7188606b0a1297ffbdb845346a05753807f2c609b65e6`.
It strictly replayed 38,222 train, selection, and confirmation games and kept
all final-test access counters at zero.  The preregistered Malom cost rule
selected the already frozen 10,000-game fallback sample, identity
`d43ee042514d9dea389849e943a5fb9d0f2d6218f6e226a980afc9354e9c8cd4`.
This cost branch is not an F0-H0 result and cannot be changed after observing
the benchmark without a new frozen contract.  See the
[freeze evidence](../evidence/f0-h0-design-b2-freeze-characterization-2026-08-15.md).

The earlier same-metric comparison measured decision-weighted `ring16`
overlap of 34.32% and 53.60% for Design B, below the 37.55% and 57.31%
same-size random disjoint-game baseline.  The source is the
[Design B supplement evidence](../evidence/f0-h0-design-b-supplement-2026-08-15.md).
The substantial shared-state rate therefore reflects intrinsic Nine Men's
Morris state convergence, not a defect uniquely introduced by B2.  At the
coverage required by this corpus, state-level novelty is not an attainable
acceptance boundary.  This is binding on later contracts: contamination,
acceptance, and generalization claims must be defined at game and player
levels.  They must not require that a canonical or `ring16` state was never
seen.  State overlap may still be measured and reported as a diagnostic, but
it cannot veto a game/player-isolated split or be relabelled as player or
game leakage.

The complexity proxy may be designed alongside F0, but execution still needs
its own immutable search/evaluation contract and resource authorization.

## F0-H0 B2 train rejection decision -- 15 August 2026

The train-only rejection contract was frozen before screening statistics
under plan identity
`dd87175dc950cbcde4b0b44cd5d4a8da0b039dcbd3cacaf198ba43ec00de0bdc`.
The immutable 10,000-game cost sample intersects official B2 train membership
in 9,113 games and 429,523 decisions.  Selection, confirmation, and final-test
content remained sealed.

The first execution exposed a technical estimator defect: unequal Jeffreys
pseudo-count denominators could create positive action lift for a transition
with zero observed events.  The original plan and result remain unchanged.
A corrected technical replay was frozen under plan identity
`a6972c3dae62ec249ccf6ea7bc7bf46132288a15db41b1c33b347b75615a9d0c`.
It uses exactly the v1 sample and threshold object and changes only zero-event
handling.  Corrected result identity
`8bd2da62785e9c8cda0a055e98213959cbdf8f88aa860384171f00f4f39c6bdc`
is the decision source.  Preserve the
[screen evidence](../evidence/f0-h0-b2-train-rejection-screen-2026-08-15.md)
and both historical and corrected manifests.

F0-H0 triggers the stop condition.  Positional choice itself is abundant:
88.69% of sampled decisions have `A_pos` cardinality greater than one, and
the state-level `k=20`, `m=5` estimability gate passes.  The corpus fails the
independent-support and concentration contract instead.  Only 20.16% of
decisions belong to a `ring16` class supported by at least five players and
ten games, against the frozen 80% floor.  Player Gini is 0.804 against 0.75,
and Kish effective support is 177.7 against 500.  Seven of 23 conjunctive
gates fail.

The stop condition for overly concentrated independent support is binding.
E0, F0-H1, T0-H-pilot, reward changes, training, and substitution of another
data source remain closed.  This is positional-only `A_pos` evidence in the
observed PlayOK-like source domain.  It is not an `A_allow` proof, a causal
inducement claim, or evidence about product UI users or a new population.

## Current evidence and tension

Retained-v3's final 200 chronological training games were 0 wins, 199 draws,
and 1 loss. This shows that the existing preserving objective can converge to
extreme draw behavior. It does not prove that the preserving reward caused the
behavior, and it does not show active trap creation.

Theory preservation is compatible with safe traps because several permitted
actions may remain. It is not sufficient: the policy must learn how permitted
actions change human error probability and conversion. Any future reward or
teacher change therefore needs an ablation that distinguishes safety from
active inducement; “first learn strength, then traps appear automatically” is
not an accepted mechanism.

## Current candidate disposition

- retained-v4: preferred research candidate, no promotion;
- retained-v3: frozen comparator;
- current active three specialists: product-serving legacy artifacts with
  untraceable corrected-label training lineage;
- Sanmill: external strength/reference option, not the project objective and
  not selected for deployment here.

The route-compatibility and specialist-lineage evidence is recorded in
`docs/evidence/retained-v4-product-route-and-specialist-lineage-audit-2026-08-14.md`.

## Stop conditions and prohibited interpretation

Stop optional human-target specialization if raw histories are inadequate,
independent support is too concentrated, conservative headroom is below the
signed product effect, or safety/conversion gates cannot be met.

Do not:

- treat `Trap_1` as a reward, safety proof, or multi-step trap measure;
- allow a theoretical concession in the name of inducement;
- substitute low/high Sanmill disagreement for human evidence;
- use fixed-node Sanmill score alone as the value case for a candidate;
- consume the remaining 108 held-out source-pool records without a new frozen
  plan and explicit authority; or
- start training while the metric, target population, split, reference,
  `p_min`, `delta_induce`, horizon, and conversion gate remain unfrozen.

## Featureized-deviation research addendum -- 15 August 2026

F0-H0 remains stopped under corrected result identity
`8bd2da62785e9c8cda0a055e98213959cbdf8f88aa860384171f00f4f39c6bdc`.
Its exact-state and `ring16` empirical-frequency estimator is not reopened,
relaxed, or reinterpreted by this addendum.

A distinct rejection-oriented question was frozen under plan identity
`04177a73ca5b9a1aa8cc8352477f2050759e6a742cee049f1191d3064ae5d662`.
It asks whether a fixed ten-term conditional-choice model predicts complete
human actions and positional D-to-L loss on unseen source-domain players, then
separately tests a one-ply `A_pos` steerability necessary condition.  It does
not treat predictive association as causal human-error induction, and it does
not establish `A_allow`, repeated-play reachability, product-user transport,
or game-score conversion.

The train-internal player split is frozen under identity
`fa74650c1afdffeb0d30f334b2b7859538f81b0e502c17a64092bfdcd99a06dd`.
The hash rule assigns 1,665/551 disjoint keys to research-exploration and
research-confirmation.  After discarding 14,941 cross-arm games, their
19,257/2,751 surviving games involve 1,416/290 player keys.  Assigned and
participating counts must not be conflated.  Official selection, confirmation,
and final-test content remain sealed.

The fixed 128-game exploration-only result has identity
`c489dca91c00569491d2b50a879bd014081e0109e0899afcc2bf2f13d584d7d6`.
It covered all 6,140 decisions, made 81,463 read-only Malom queries, fit no
model, and opened no protected content.  Several named features vary and
conflict with positional safety often enough to avoid an immediate global
rejection, while double-mill creation had zero pilot variation and mill closure
is structurally close to post-action material balance.  These are exploratory
observations only.

No confirmatory model or one-ply steerability result exists.  The
[design review](../evidence/human-feature-deviation-design-review-2026-08-15.md)
records the independent critique, frozen thresholds, pilot evidence, and
claim boundary.  This addendum grants no authority for confirmation, E0,
F0-H1, T0, reward changes, games, training, promotion, publication, or release.

## Featureized-deviation precision addendum -- 15 August 2026

The follow-up precision and design round does not alter the F0-H0 stop.  Its
[evidence](../evidence/human-feature-deviation-precision-rebalance-and-extension-2026-08-15.md)
records an outcome-blind split rebalance and a larger exploration-only feature
screen.

The selected future research split has identity
`8187ffa06cc73f4e052b7481f06dc3629a23feace63e086c7075c74c17940028`.
Its disjoint player arms are assigned 1,108 keys each.  Surviving same-arm
games contain 980 exploration and 487 confirmation player keys.  The
confirmation-side decision-weighted Kish diagnostic is 58.91, up from 46.78
under v1, but this does not certify the frozen precision gates.  Research
confirmation and all official holdouts remain unopened.

The 1,024-game exploration extension has result identity
`53e010f473a88d4a384b264906a4a8d1826b92fd5f48e4b386b57356ee78c61a`.
It opened only the frozen exploration membership and fit no model.  The old
post-action material term is exactly affine with mill closure inside every
one of 48,855 complete choice sets and is removed from future conditional
choice models.  Simultaneous double-mill closure remains a real but absent
event in this sample; it is replaced by the distinct, observable future
mill-fork proxy.  These are exploratory design decisions, not predictive or
causal findings.

Revised feature plan identity
`5919b9666d66c568898797e3b2089a71a71bc289696291d29c7aec6dd91e0935`
freezes the new split, ten-term dictionary, unchanged substantive effect
floors, and a new pre-execution precision gate.  That gate is not met because
no permitted exploration-only conditional model or player-level variance
calibration exists.  Confirmation therefore remains prohibited.  A later
round may implement and calibrate the frozen estimator using research
exploration only; it may not relax thresholds after that calibration or open
protected content without satisfying the frozen gate.
