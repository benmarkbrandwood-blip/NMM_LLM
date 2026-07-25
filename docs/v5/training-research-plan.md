# v5 Training Research Plan

Status: governing research and training specification.

This document owns the falsifiable training hypotheses, baseline ladder,
minimal target-specialisation experiment, and escalation rules. It does not
authorise a run. Every smoke, training, or evaluation still needs its own
frozen experiment contract and explicit launch approval.

## Two Active Lanes

The project retains two non-interchangeable lanes:

| Lane | Purpose | Promotion boundary |
| --- | --- | --- |
| Corrected v4 exploration | Low-cost observation, trainer hardening, clean from-scratch controls, and candidate generation | Must pass the same frozen evaluation and runtime gates as any other source |
| v5 reference/specialisation research | Establish deterministic reference semantics, test compact architectures, then conditionally test human-targeted D ranking | Optional modules open only after feasibility and reference gates pass |

Maintainer-`main` v2a/v2b, Sentinel v2, HumanPrefNet, hot recovery, and related
graphs may continue as clearly labelled exploratory work. Their observations
can generate hypotheses. They cannot initialise or resume the retained `dev`
lineage unless their data, code, checkpoint, optimiser, and run state become a
separately accepted input.

## Corrected-v4 Boundary

Legacy contaminated artifacts and clean corrected-v4 runs are different
categories:

- A checkpoint, optimiser, replay, SpecialistDB, or label snapshot that may
  contain pre-correction Malom influence is shadow-only for release lineage.
- A from-scratch run using verified corrected labels, isolated state,
  versioned inputs, and exact-resume evidence is a legitimate control or
  candidate source.
- A contamination ablation can explain old behaviour but cannot repair
  lineage or turn an old checkpoint into a corrected release source.

The completed managed corrected-v4 baseline is clean category-two lineage. Its
5,000-game completion and accepted Stage-0 diagnostic show reproducible
training signal only; they do not establish strength, phase coverage, or
promotion.

## F0 Feasibility Before Complex Training

No full HumanPolicy v3, multi-step FrozenTeacher, GapNet/SelfRiskNet, DAgger,
adaptive curriculum, or PPO work begins before these bounded reports:

### F0-H: Specialisation headroom

On the intended natural traffic measure, report:

- mass of D decisions;
- mass with `|A_allow| > 1`;
- exact and near ties under the complete ultra-strong comparator;
- HumanPolicy-supported mass and an optimistic uncertainty floor;
- optimistic fraction of distinguishable candidate pairs;
- natural visit mass of those states;
- an upper bound on the whole-game effect even if every eligible decision
  improves.

If the global upper bound is below the signed product effect, target
specialisation stops before HumanDB v3 or full teacher construction.

Before full teacher construction, run one same-state, same-budget comparison
over:

- coarse W/D/L with a neutral tie policy;
- the complete ultra-strong reference;
- one-step HumanPolicy allow-mass ranking; and
- a very short, support-qualified target rollout.

If neither human-conditioned baseline shows a stable advantage over the
ultra-strong reference, do not build a larger `T_full` merely because the
ultra-strong design was expected to leave headroom.

### F0-P: proof and history feasibility

Use the requirements in
[the oracle/rules specification](oracle-and-rules-spec.md) to measure
W-liveness and runtime proof size. Do not bind reference training to a proof
system that has not demonstrated a viable support domain.

### F0-A: architecture comparison

Compare ordinary always-move, verified compact, compact plus pack, optional
local full DB, server oracle, and hybrid designs before freezing the release
form. Include total engineering cost, licensing/redistribution, target-device
latency/resources, availability, and user experience.

## T0 Reference Baseline Ladder

Use the smallest baseline capable of falsifying the next hypothesis:

`q_ref` is the non-human-targeted reference policy produced by the accepted
rules/oracle layer: full-history liveness/proof rank in W, the complete
ultra-strong comparator in D, and the verified survival/reference ordering in
L. Its exact tie policy is versioned.

1. direct `q_ref` from the accepted oracle/rules gate;
2. a small policy distilled from `q_ref`;
3. the same policy plus shallow strategic search constrained to the
   applicable authorised set;
4. the same policy plus measured exact-pack variants;
5. a clean corrected-v4/generalist source under the same runtime and
   evaluation contract.

The shallow-search baseline is mandatory before adding more predictive heads.
Proof search establishes authorisation; it is not assumed to repair strategic
ranking. A small AB/MCTS-style search inside the authorised set may provide
more value than a new GapNet or SelfRiskNet and must be measured.

If a reference configuration already meets the signed product need, stop at
the reference candidate. Building human-targeted complexity is not a required
definition of project success.

## HumanPolicy v1 Scope

The first behaviour model is deliberately coarse:

- broad Elo condition rather than sparse fine-grained cells;
- observed single-decision choice distribution;
- one-step unsafe/allow mass and coarse structural calibration;
- very short rollout horizons only where support is adequate;
- explicit OOD/abstention.

Player/game concentration, time span, and state sparsity prohibit unsupported
claims of precise fine-state probabilities. High-dimensional uncertainty
sets, complex partial identification, and long multi-step traps remain
conditional research.

### Consistency within a rollout

The simulated opponent must not become a different average person on every
move. Each rollout samples one frozen behaviour member or latent style:

```text
z ~ P(style | target condition)
a_t ~ HumanPolicy(a_t | DecisionState_t, target condition, z)
```

The same `z` or ensemble member remains fixed for that rollout. A memoryless
population-average baseline may be retained as an explicit ablation.

### Symmetry

Oracle and StrategyPolicy may enforce proven D4 equivariance. HumanPolicy
must retain raw UI orientation and colour metadata. D4 augmentation or colour
exchange is an ablation because human visual and opening preferences need not
be symmetric even when game value is.

## Target Population and Training Weights

Training weights must name the estimand they approximate:

- `average_unique_player`;
- `average_natural_game`;
- `traffic_weighted_decision`; or
- another explicitly defined product measure.

Player/game normalisation is not an automatically neutral correction. It
changes the target population and therefore must be justified against the
product estimand. See the
[human-data/statistics plan](human-data-and-statistics-plan.md).

## T1 Minimal D-Mode Specialisation

The first specialisation experiment is limited to:

- D states;
- high HumanPolicy support;
- `|A_allow| > 1`;
- non-flying states;
- short, explicitly bounded horizons;
- direct teacher;
- one HumanPolicy;
- one policy-only StrategyPolicy;
- no GapNet, SelfRiskNet, DAgger, PPO, or adaptive curriculum.

The direct teacher must first beat `q_ref` on a confirmation set not used to
choose labels. A minimum student must retain a preregistered share of that
gain. The effect must remain directionally stable under a different
same-source model family, independent search/AB stress, a reference-best-reply
branch, and at least one deployment-relevant self continuation.

Same-source model agreement tests model overfitting only. It is not an
independent human confirmation.

### Qualified inducements

A state/candidate may be called a target-human inducement only when:

- it preserves the applicable D/full-rule boundary;
- at least one correct opponent reply remains;
- the simultaneous lower confidence bound on target-human incorrect-reply
  mass is at least a frozen positive `p_min`;
- independent-player/support floors pass and the mass is not an OOD
  extrapolation;
- the incorrect reply genuinely creates the specified theoretical event; and
- the reference-best-reply branch does not degrade the AI.

A softmax's nonzero probability is not evidence. A popular but unsupported
action and a theoretically forced win are not target-human inducements.

## D Teacher Objective

The D teacher preserves D as a hard constraint. Its primary event remains
entry into a full-history certified W after an opponent error, but two D
candidates with similar entry probability must not be treated as equivalent
when their W states have materially different product value.

Use a lexicographic, conversion-aware target:

1. conservative lower bound on entering certified W within `H_D`;
2. among statistically indistinguishable first-tier candidates, conservative
   terminal conversion under the frozen W policy and runtime contract;
3. if conversion cannot be estimated reliably, use proof rank, measured
   W-conversion calibration, future proof availability, pack closure, and
   runtime unavailability risk as secondary fields;
4. retain `q_ref` or abstain whenever uncertainty can reverse the order.

The deployment-aware fields may break ties only within an oracle-equivalent or
explicitly allowed near-equivalent tier. They cannot buy a theoretical
downgrade in verified mode.

## W and L Scope

The first D-only experiment does not change the W policy. W uses the reference
liveness/conversion policy, and its conversion is a non-inferiority endpoint
for D specialisation. A W-specific superiority requirement is activated only
with an explicitly trained W objective.

L research uses the ordering in the oracle/rules specification: reach a
minimum survival threshold, then prioritise rescue probability, then survival
distance. A model that merely prolongs a lost game without improving rescue or
experience is not a successful result.

## Teacher Labels and Ranking

Start with the simplest representation of supported evidence:

- exact reference tie groups;
- top-group classification;
- pairwise/listwise order only for simultaneously distinguishable candidates;
- no within-group order for interval-unknown candidates.

Do not interpret a lower confidence bound as a probability mass. A
softmax/group-mass mapping may be tested only after held-out calibration shows
that it improves teacher regret over the simple target.

If a sampled dominance graph contains a cycle:

1. first test deterministic arithmetic, shared-input, comparator, and interval
   construction for a reproducible contradiction;
2. a deterministic algebraic contradiction is a job failure;
3. otherwise merge the strongly connected component into an unknown/tie group
   or abstain for the state.

Finite-sample inconsistency alone is not a fatal safety fault.

## Sampling and Importance Weights

The governing plan does not freeze intuitive percentage tables. Experiment
configuration may declare natural replay floors, safety-boundary coverage,
single-source caps, and stress caps. Exact bucket ratios are selected only by
bounded train/selection experiments and frozen before confirmation.

Importance weighting may use a preregistered stabilisation method:

- stratified resampling;
- self-normalised weights;
- truncation with a reported bias bound;
- control variates or a doubly robust estimator.

Every method reports the target measure, sampling probability, effective
sample size, maximum information share, variance/design effect, and estimated
stabilisation bias. “Unbiased but unusable variance” is not automatically
preferred.

## Deployment-Aware Training

An oracle policy is not deployed by merely removing oracle inputs. E1
regenerates labels under the exact named runtime contract:

- pack and per-candidate hit semantics;
- `X_rt`;
- proof certificate and scheduler;
- deadline and early-stop rule;
- actual authorised pool;
- ordinary or verified product mode;
- runtime unavailable/shortfall handling.

If a verified pool can contain oracle-denied horizon-proved actions, train a
separate runtime fallback policy over those candidates. Otherwise verified
mode selects only from `A_allow ∩ A_runtime` and reports unavailable when the
intersection is empty.

Among oracle-equivalent candidates, runtime training may optimise proof cost,
future availability, pack closure, support closure, and measured risk. It must
not silently redefine an oracle safety claim.

## Optional Escalation

Open one optional component only when the preceding report identifies its
specific bottleneck:

| Evidence | Permitted next experiment |
| --- | --- |
| Direct teacher is too slow but valid | One calibrated amortiser |
| Student loses direct-teacher gain on reached states | One bounded DAgger round, then reassess ROI |
| Graph representation is the bottleneck | Matched graph-versus-phase-aware-MLP ablation |
| Reference and student are strong but independent outcomes plateau | Stronger teacher/search comparison before PPO |
| HumanPolicy support is inadequate | New independent human data under a separately approved collection plan |

There is no requirement to train GapNet, SelfRiskNet, a shared encoder, or PPO.
DAgger has no standing round count; each round must demonstrate incremental
value before another is proposed.

## Search Opponents and Elo

Search depth, node budget, style mixture, and claimed Elo are experimental
configuration, not a governing mapping. Calibrate search opponents on held-out
human games and complete sequences, including correlated errors, phase
transitions, repetition/no-progress behaviour, and time conditions. Log actual
completed depth and nodes.

Relative ratings produced by round-robin AI matches are useful for ordering
frozen configurations. They are not human Elo until anchored by games against
an adequately identified human population. Any fixed depth/Elo mixture remains
a versioned ablation rather than a default law of play.

## Numerical and Resume Gates

Every retained training path defines:

- target range and normalisation for each head;
- NaN/Inf detection before optimiser steps and checkpoint writes;
- per-head loss and gradient norms;
- sample-weight and mask distributions;
- clipping/mixed-precision policy;
- tests for empty masks, singleton batches, extreme weights, and large choice
  sets;
- complete optimiser/scheduler/scaler and RNG state;
- data cursor, curriculum, opponent, proof/cache, and logging state;
- same-batch output equality before and after exact resume.

Losses near `10^20`, policy losses with incompatible old/new log-probability
semantics, or weights-only restart cannot be accepted as exact continuation.

## Stop Conditions

Stop target specialisation when:

- F0-H cannot support the signed global effect;
- direct teacher does not stably beat `q_ref`;
- the minimum student cannot retain the gain;
- independent opponent/search stress reverses the direction;
- runtime-aware closure removes the gain;
- numerical or resume integrity fails.

Preserve the diagnostic artifacts and reference candidate. A stopped optional
branch is not a failed reference project.
