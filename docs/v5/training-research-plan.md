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

## Feasibility Before Complex Training

No full HumanPolicy, multi-step FrozenTeacher, GapNet/SelfRiskNet, DAgger,
adaptive curriculum, or PPO work begins before the applicable stages below.

### F0-D0: data reconstructability and planning freeze

Before inspecting state/action headroom, freeze one planning snapshot and an
untouched or prospective confirmation source as specified by the
[human-data plan](human-data-and-statistics-plan.md). Produce the raw-game
reconstructability report before treating any aggregate HumanDB row as a
full-history decision.

### F0-H0: pre-E0 coarse upper screen

F0-H0 uses only the planning snapshot, `A_legal`, trusted coarse positional
W/D/L where available, raw observed-choice frequencies, and coarse natural
visit estimates. It may report:

- mass of coarse D decisions and states with more than one legal action;
- observed action support and concentration;
- a conservative reachability upper bound on whole-game score headroom; and
- clearly named optimistic heuristics that are not mathematical bounds.

F0-H0 cannot use formal `A_allow`, `certW`, a trained HumanPolicy, or the
complete comparator because E0 has not accepted them. A valid upper bound below
the signed product effect rejects specialisation. A heuristic estimate cannot
approve T1 or veto it by itself.

### F0-P0: pre-E0 proof reconnaissance

Inventory history-state dimensions, positional W incidence, rough graph
branching, and representative proof-resource scale. These measurements may
reject an obviously infeasible design but cannot establish soundness,
W-liveness, or runtime authorisation.

### F0-A0: architecture inventory

Under the known product constraints, inventory ordinary local search, compact
policy/search, exact-policy compression, pack, optional local full DB, server
Oracle, and hybrid designs. Unknown deployment constraints remain explicit;
F0-A0 cannot select a final architecture while they are open.

### E0: deterministic foundation

Accept one rules/history state, complete comparator, atomic-action surface, and
strict Oracle/proof error boundary before exact headroom or proof feasibility.

### F0-H1: post-E0 exact headroom

Using only the planning/train domains, report:

- D-state mass with `|A_allow| > 1`;
- exact and equivalence-band ties under the complete comparator;
- support and uncertainty from one approved minimal HumanPolicy;
- distinguishable candidate-pair and natural visit mass; and
- a conservative whole-game bound or an explicitly non-binding optimistic
  headroom heuristic.

### T0-H-pilot: non-promotable direct comparison

Under its own bounded experiment card, compare:

- coarse W/D/L with a neutral tie policy;
- the complete ultra-strong reference;
- one-step HumanPolicy allow-mass ranking; and
- a very short, support-qualified target rollout.

The pilot cannot access confirmation/final data and cannot promote a candidate.
If neither human-conditioned baseline shows a stable advantage over the
ultra-strong reference, do not build a larger teacher merely because the
design was expected to leave headroom.

### F0-P1: exact proof and viability feasibility

Use the accepted E0 semantics to measure full-history W-liveness, runtime proof
size, and recursive viability for any whole-game verified claim. Do not bind a
reference or product to a proof system that has not closed its support domain.

### F0-A1: final lane selection

After the product's deployment constraints and the applicable H1/P1 evidence
are frozen, select one ordinary, theory-verified, positional-exact,
bounded-survival, or Oracle-service lane. Subsequent T0 and E1 work follow that
lane; they do not default back to compact proof merely because it appears in
the research plan.

## T0 Reference Baseline Ladder

Use the smallest baseline capable of falsifying the next hypothesis:

`q_ref` is the non-human-targeted reference policy produced by the accepted
rules/oracle layer: full-history liveness/proof rank in W; full-rule D
viability followed by versioned history-slack/cycle-risk handling and then the
complete positional comparator in D; and the verified survival/reference
ordering in L. If only the positional D comparator is available, name the
control `positional_ultra_strong_control` rather than implying complete
history-aware D strategy.

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
- `HumanObservationState` inputs only, with hidden rule/proof state excluded;
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
a_t ~ HumanPolicy(a_t | HumanObservationState_t, target condition, z)
```

The same `z` or ensemble member remains fixed for that rollout. A memoryless
population-average baseline may be retained as an explicit ablation. This is a
static-player model; repeat-play claims require the adaptation and carryover
tests in the human-data plan.

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

## T1 Minimal Supported-Error-Pattern Reranking

The first specialisation experiment safely reorders already supported human
error patterns. It does not claim to invent novel traps or unfamiliar
dilemmas. It is limited to:

- D states;
- high HumanPolicy support;
- `|A_allow| > 1`;
- non-flying states;
- short, explicitly bounded horizons;
- direct teacher;
- one HumanPolicy;
- one policy-only StrategyPolicy;
- no GapNet, SelfRiskNet, DAgger, PPO, or adaptive curriculum.

The direct teacher must first beat `q_ref` on a planning-domain
`teacher_validation` holdout not used to choose labels. This is not the
one-time confirmation or final test. A minimum student must retain a
preregistered share of that gain. The effect must remain directionally stable
under a different same-source model family, independent search/AB stress, a
reference-best-reply branch, and a prespecified set of deployment-relevant
self continuations.

Same-source model agreement tests model overfitting only. It is not an
independent human confirmation.

### Qualified inducements

A state/candidate may be called a target-human inducement only when:

- it preserves the applicable D/full-rule boundary;
- at least one correct opponent reply remains;
- the simultaneous lower confidence bound on target-human incorrect-reply
  mass is at least a frozen positive `p_min`;
- the simultaneous lower bound on incorrect-reply probability or terminal
  match score improves over the frozen `q_ref` action by at least the
  preregistered `delta_induce`;
- independent-player/support floors pass and the mass is not an OOD
  extrapolation;
- the incorrect reply genuinely creates the specified theoretical event; and
- the reference-best-reply branch does not degrade the AI.

A softmax's nonzero probability is not evidence. A popular but unsupported
action and a theoretically forced win are not target-human inducements.
Without the relative improvement requirement, the accurate description is
“enters a supported human-error-prone state,” not “induces an error.”

## D Teacher Objective

The D teacher preserves the applicable full-rule D boundary as a hard
constraint. Within that permitted set, its primary utility is the conservative
terminal product score under frozen human, self-continuation, horizon, and
runtime identities:

```text
Q_D(S, a; pi_self, pi_h, H, runtime)
  = LCB E[1 * win + 0.5 * draw + 0 * loss
          | S, a, pi_self, pi_h, H, runtime]
```

The loss-rate constraint remains independently enforced; expected score cannot
purchase a forbidden theory downgrade or an unacceptable loss increase.
`P(certW)`, conditional W conversion, proof rank, pack closure, and runtime
availability are mechanism diagnostics and, where the product contract
requires them, secondary constraints. They do not outrank terminal utility
merely because the current prover finds one line easier to certify.

A secondary field may break a first-tier tie only when the confidence interval
for the primary-utility difference is wholly inside the preregistered
equivalence band `[-epsilon_D, +epsilon_D]`. Failure to reject zero difference
is not equivalence. If equivalence cannot be established, obtain more data or
retain `q_ref`/abstain.

### Continuation-dependent labels

Teacher values are versioned functions of `pi_self`, `pi_h`, horizon, and the
runtime contract; they are not timeless state-action labels. T1 therefore uses
a conservative policy-improvement loop:

1. label with frozen `pi_k` and a prespecified continuation set;
2. constrain the student update by a small KL bound or a bounded state scope;
3. evaluate `pi_(k+1)` in closed loop under the same product estimand;
4. relabel high-impact reached states with `pi_(k+1)`; and
5. continue only when whole-policy gain and every hard constraint remain.

The primary report includes the conservative result across the continuation
set, not only the checkpoint under which the teacher looked best. DAgger may
later improve reached-state coverage, but it does not by itself cure a drifting
continuation-dependent objective.

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
- ordinary-best-effort, theory-preserving, positional-exact, or
  bounded-survival product mode;
- runtime unavailable/shortfall handling.

If a bounded-survival pool can contain Oracle-denied horizon-proved actions,
train a separate bounded policy over those candidates. A
theory-preserving policy selects only from its recursively viable
`A_allow ∩ A_runtime ∩ K_theory` pool. Every named mode reports unavailable
when its exact pool is empty.

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

- a valid F0-H0 bound cannot support the signed global effect;
- F0-H1 or T0-H-pilot cannot establish adequate supported headroom;
- direct teacher does not stably beat `q_ref`;
- the minimum student cannot retain the gain;
- independent opponent/search stress reverses the direction;
- runtime-aware closure removes the gain;
- numerical or resume integrity fails.

Preserve the diagnostic artifacts and reference candidate. A stopped optional
branch is not a failed reference project.
