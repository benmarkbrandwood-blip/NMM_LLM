# v5 Human Data and Statistical Analysis Plan

Status: governing human-data, estimand, power, and human-evaluation
specification.

This document does not treat HumanDB size, neural accuracy, proxy games, or
training curves as evidence that a product improved against people.

## Data Authority

The imported HumanDB has two distinct information classes:

- observed human frequencies, choices, game results, player/time fields that
  survive the raw-record rebuild; and
- historical Malom-derived fields.

The first class may support behaviour modelling after replay, deduplication,
and split checks. Unversioned historical Malom columns are not label authority.
Candidate counterfactual outcomes come only from a versioned oracle/teacher or
from prospective data with a known action propensity.

The complete observed decision is the supervision unit for HumanPolicy:

```text
DecisionState
legal choice set
observed action
actor/opponent conditions
game/player/time/source identity
```

An unchosen candidate is missing observational data, not a negative example.

## Legal, Licence, and Privacy Gate

Before a HumanDB rebuild, new human collection, model distribution using human
records, or redistribution of a derived Malom pack, record an explicit
decision for:

- source-platform permission for training use;
- player-identity hashing and cross-version linkage;
- retention and deletion policy;
- consent and notice for newly collected games;
- commercial and model-distribution rights;
- Malom/tablebase licence and derived-pack redistribution;
- access controls for raw and linked data;
- responsible owner and unresolved legal questions.

An inventory field saying `licence: unknown` is evidence of an open question,
not a pass. An unresolved right required by the planned use is a
`release_block`; raw-data collection or processing that itself lacks authority
is an `experiment_block`.

## Target Population and Estimand

Every training, proxy, and human endpoint declares which population measure it
estimates:

| ID | Sampling interpretation |
| --- | --- |
| `average_unique_player` | A player receives equal target weight |
| `average_natural_game` | A naturally occurring game receives equal target weight |
| `traffic_weighted_decision` | A product decision receives traffic-proportional weight |
| `prospective_trial_population` | A randomised participant under the trial's eligibility and assignment rules |

Player/game normalisation is not assumed to be correct for every objective.
Its effect on the target measure, effective sample size, and player
concentration is reported. The product owner freezes the target measure before
candidate results are examined.

## HumanDB Support Audit

F0-H is a read-only, rejection-capable audit. By target condition and coarse
phase/structure, report:

- games, observed decisions, independent players, and time coverage;
- player/game concentration and maximum information share;
- player-cluster effective sample size;
- legal choice-set and action support;
- D-state mass with `|A_allow| > 1`;
- optimistic HumanPolicy calibration width;
- expected OOD/abstention mass;
- optimistic distinguishable-pair mass;
- maximum whole-game effect implied by natural visit mass.

Millions of decisions from a small or concentrated player set do not become
millions of independent observations. If the optimistic effect bound is below
the product target, stop specialisation before building HumanDB v3 or a
multi-step teacher.

## Rebuild and Splitting

If F0-H permits continuation, rebuild from raw complete games and retain the
fields needed for the target estimand. Do not reconstruct missing
player/time/Elo information from an aggregate table.

Deduplicate and split in this order:

1. replay and reject illegal/incomplete records;
2. group exact, same-game, symmetry, and accepted near duplicates;
3. construct player-game components;
4. assess whether a player/time-isolated split is feasible with adequate
   effective sample size;
5. freeze train, selection, one-time confirmation, and final-test membership;
6. generate out-of-fold predictions only inside train.

Every variant of the same game or duplicate component stays on one side.
Confirmation is accessed once by one selected recipe. Final test is not used
for architecture, calibration, thresholds, or curriculum.

## HumanPolicy Evidence Level

The first HumanPolicy supplies coarse observed-choice probabilities and
abstains outside support. It fixes one behaviour member or latent style for a
whole rollout, as required by the
[training research plan](training-research-plan.md).

`label_generation_hp` and a different architecture trained on disjoint folds
of the same HumanDB remain same-source models. A separately isolated
`proxy_eval_hp` can detect overfitting to one model, but its proper label is
`independent_model_same_source_proxy`. It cannot establish transport to new
humans, a new platform, or a new historical period.

True external confirmation requires prospectively collected people or a
materially independent source with its own provenance and target match.

## Primary Product Effects

### Reference product

A reference candidate is compared with the current product on:

- match score, `win + 0.5 * draw`, as the default primary playing result;
- loss-rate non-inferiority;
- W conversion and rules-draw behaviour;
- runtime availability and resources.

Win rate remains a key secondary endpoint. A product may instead make it
primary only by freezing that decision and a coherent sample-size analysis
before results.

### D-only specialisation

A D-only treatment is compared with the same-stack compact reference on:

- one primary natural-match product effect, normally match score;
- loss-rate non-inferiority;
- W-conversion non-inferiority;
- runtime risk/availability non-inferiority.

It is not required to show W-conversion superiority because it does not train
the W policy. If W specialisation is later enabled, W conversion may become a
superiority endpoint under a new objective and power plan.

## Margin and Power Semantics

Keep three quantities separate:

- `m`: the release superiority or non-inferiority margin;
- `d_plan`: the assumed true effect used for planning;
- `MDE(n)`: the effect detectable with the specified design and sample size.

If promotion requires:

```text
LCB(effect) > m
```

then power is computed under `d_plan > m`, using the distance
`d_plan - m`. A calculation that detects a difference from zero does not power
a positive-margin claim. The same rule applies to action-level
`delta_specialize`, ordering margins, natural match effects, and W conversion.

Pilot data may estimate nuisance quantities—control rate, draw rate,
correlation, attrition, player ICC, W-event frequency, and variance—while
blinded to treatment effect. It must not move the product margin to whatever
the available sample can pass.

When no feasible sample exists under the recruitment/resource ceiling, the
answer is `not_feasible` or `inconclusive`, not a smaller post hoc margin.

## W-Start Conversion Suite

The W-start suite represents states naturally reached by the frozen controls,
not a convenient puzzle collection. Freeze a mixture weighted by:

- certified-W entry states from the current product;
- certified-W entry states from the compact reference;
- phase, proof rank, rules-history slack, and pack/prover path;
- natural visit mass.

Deduplicate same-game, symmetry, and near-identical states. A participant must
not repeatedly receive highly similar positions in a way that trains them
during the trial. Report separately:

```text
positional W
full-history certW
valid assigned W start
terminal win/draw/loss
runtime unavailable or shortfall
```

Artificial coverage strata remain stress evidence and do not silently receive
natural-population weight.

## Proxy Evaluation

Proxy opponents include:

- the label HumanPolicy as an overfitting diagnostic;
- `independent_model_same_source_proxy`;
- independently implemented search/AB families;
- reference-best-reply stress;
- old checkpoints as robustness opponents when their lineage is labelled.

Report each separately. Agreement across proxy models supports robustness to
those models only. It does not replace a human trial.

## Two Human Trials, Two Estimands

Strategy efficacy and real product effect require different timing contracts:

1. `latency_matched_strategy_trial`: both arms reveal moves on the same visible
   schedule, isolating strategic quality.
2. `real_latency_product_trial`: each arm exposes its actual product latency,
   timeout, and unavailable behaviour, measuring the delivered experience.

The first cannot be interpreted as complete product effect. The second cannot
attribute a difference solely to strategy. If resources permit only one, its
claim is limited to its declared estimand.

Both use randomised frozen versions, one intent-to-treat result per assignment,
player clustering, colour/start balancing, actual assignment probabilities,
and prespecified handling of disconnections and runtime failure.

## Style and “Human-Like” Evidence

Human-like play is optional and separately evaluated. A blinded style study
may measure:

- expert/player naturalness ratings;
- patience versus premature Mill closure;
- repetition and purposeless-delay tendency;
- strategic explanation fidelity;
- identifiable model quirks or implausible openings.

Style does not override legality, safety, match result, or loss-rate gates.
HumanPolicy top-1 accuracy and win rate are not substitutes for a blinded style
assessment.

## Diagnostic Metrics

HumanPolicy calibration, safe/allow mass, first-error probability,
`P(certW)`, proof rank, squeeze, IW effective sample size, rank flips, and
style tags remain diagnostics or research-stage gates. They become release
gates only when a separate product requirement names the estimand, margin,
support domain, and power.

## Data-Collection Escalation

New human data is a separate project. It begins only after a support or trial
audit identifies the missing evidence and an approved plan freezes platform,
purpose, budget, consent, privacy, retention, player linkage, target
population, and stopping rule.

Ordinary observational games train behaviour only. Counterfactual estimation
requires an approved safe-set randomisation with the true per-decision
propensity recorded. Players used for training or selection do not count as
unseen confirmation players for the same version.
