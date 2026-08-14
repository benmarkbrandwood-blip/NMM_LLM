# Safe human-trap objective and measurement design v1

Date: 2026-08-14

Status: `product_objective_defined_measurement_draft_no_launch_authority`

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

The available corpus is valuable but its exact usable identity is not yet
frozen. Current machine evidence distinguishes 95,389 human-game JSONL files,
94,540 imported IDs, and 94,429 games in the active HumanDB. The previously
quoted 95,221 is not a current manifest-backed identity and must not appear in
an acceptance denominator until a replayed dataset manifest establishes it.

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

The complexity proxy may be designed alongside F0, but execution still needs
its own immutable search/evaluation contract and resource authorization.

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
