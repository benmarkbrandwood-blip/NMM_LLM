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

HumanPolicy observes what the player could perceive, not the server's complete
rules state. Its supervision unit is:

```text
HumanObservationState
legal choice set
observed action
actor/opponent conditions
game/player/time/source identity
```

An unchosen candidate is missing observational data, not a negative example.

`HumanObservationState` is a separate versioned type containing only
player-visible information:

```text
visible board and side/phase cues
visible move history and draw/claim indicators
clock and time-control information
UI orientation, prompts, and assistance
available player/session context and prior-game exposure
explicit missingness for fields not preserved by the source
```

The full `DecisionState` remains the input to rules, Oracle, and proof layers.
HumanPolicy must not read hidden repetition multiplicity, no-progress counters,
proof/pack status, `A_allow`, or other server-only fields unless the source UI
actually displayed equivalent information. An offline labeler may join both
types by an opaque decision ID without copying hidden fields into the behaviour
model.

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

Freeze a small robustness panel with the primary estimand:

- the main target Elo interval and one adjacent interval on each side where
  support permits;
- first exposure and repeat exposure;
- a small number of predeclared behaviour clusters;
- source platform and target product platform.

Use a prespecified worst-stratum non-inferiority rule, CVaR, or maximum subgroup
regret for harms that the product decision considers material. Do not create
dozens of underpowered release gates. Report unsupported strata as unknown and
control multiplicity for every confirmatory subgroup decision.

## Planning Data and Untouched Evidence

Before any state/action headroom analysis, freeze:

- a content-addressed `planning_snapshot` that F0-H0, architecture work, and
  model design may inspect; and
- an untouched source that planning cannot access, defined by a pre-existing
  access-controlled snapshot or, preferably, future players/time collected
  after the design freeze.

The current aggregate HumanDB and historical corpus have already informed
project planning. They are development/planning data and cannot be
retroactively renamed a pristine final test merely by choosing rows after
their aggregate properties were examined. A player/time-isolated historical
split may still support development confirmation under an explicit limitation;
the strongest transport claim requires prospective or demonstrably untouched
data.

Hashes, membership rules, permitted users/jobs, and access counts are frozen
before F0-H0. Planning, train, and selection jobs cannot read the untouched
source.

## F0-D0 Raw Reconstructability

Before using aggregate positions for full-rule reasoning, replay raw records
and report:

- games with a complete continuous move sequence;
- games that replay unambiguously from the initial state;
- recoverable player, Elo, time, source, and rules-variant fields;
- recoverable terminal and draw reasons;
- interrupted, missing-ply, duplicate-export, malformed, and illegal rates;
- decisions for which repetition/no-progress/claim history can be recovered;
- independent players and games in each usable support class.

The current HumanDB v2 tables and four-field board FEN do not contain complete
history, player, Elo, or time state. Their board/action frequencies may support
a coarse visible-board behaviour audit. They cannot supply formal `A_allow`,
`certW`, W-liveness, or full-rule teacher labels.

## F0-H0 Coarse Support Screen

F0-H0 is read-only and rejection-only. On the planning snapshot, by target
condition and coarse phase/structure, report:

- recoverable games, decisions, independent players, and time coverage;
- player/game concentration and maximum information share;
- player-cluster effective sample size where identity is recoverable;
- legal choice-set and observed-action support;
- coarse positional-D and multi-legal-action mass;
- natural probability that a game reaches at least one potentially modifiable
  state.

A conservative coarse whole-game score bound may use:

```text
P(reference game reaches at least one modifiable coarse state)
* maximum possible per-game score swing
```

This deliberately avoids adding repeated decision effects as if they were
independent. A tighter first-divergence or closed-loop occupancy calculation is
permitted only when its assumptions are verified. Any quantity that is not
proved to upper-bound the effect is named `optimistic_headroom_heuristic` and
cannot veto the route alone.

Millions of decisions from a small or concentrated player set do not become
millions of independent observations. A valid conservative bound below the
product target rejects specialisation before HumanDB v3 or teacher work.
F0-H0 cannot approve T1.

## Rebuild, Splitting, and F0-H1

If F0-H0 permits continuation, rebuild the development corpus from raw
complete games and retain the fields needed for the target estimand. Do not
reconstruct missing player/time/Elo information from an aggregate table.

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

After E0 and the split freeze, F0-H1 may use accepted `A_allow`, the complete
comparator, and one approved minimal HumanPolicy on planning/train data. It
reports formal D eligibility, HumanPolicy support/abstention, equivalence-band
candidate mass, and exact-headroom uncertainty. The separate T0-H-pilot owns
any same-state or short-rollout comparison. Neither stage reads confirmation or
final data.

## HumanPolicy Evidence Level

The first HumanPolicy supplies coarse observed-choice probabilities and
abstains outside support. It fixes one behaviour member or latent style for a
whole rollout, as required by the
[training research plan](training-research-plan.md).

That fixed member models a consistent but static player, not learning or
adaptation. Unless repeat-exposure evidence is available, its claim is limited
to first or isolated exposure. Before a repeat-play product claim, report:

- first exposure and the second through fourth similar exposures;
- within-player change across consecutive games;
- conditioning on available prior-game/session history; and
- an adaptive stress opponent that searches for a public frozen policy's
  repeated pattern.

`label_generation_hp` and a different architecture trained on disjoint folds
of the same HumanDB remain same-source models. A separately isolated
`proxy_eval_hp` can detect overfitting to one model, but its proper label is
`independent_model_same_source_proxy`. It cannot establish transport to new
humans, a new platform, or a new historical period.

True external confirmation requires prospectively collected people or a
materially independent source with its own provenance and target match.

When HumanDB comes from a different UI or platform, run a small
`transport_pilot` before T1 promotion evidence:

- present aligned decisions through both source-like and target-product views;
- compare action distributions, decision time, and orientation effects;
- separate game-structure features from UI/source features; and
- retain the source model as a source-only proxy when transport fails.

The pilot is not required for a coarse source-domain F0-H1 audit, but it is
required before calling that model representative of the target product.

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

Player-level parallel assignment is the default because seeing one strategy
can change later responses to the other. A crossover design must separately
report first-period effects, treatment order, period-by-treatment interaction,
and learning/carryover; the aggregate crossover contrast is not sufficient.

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
