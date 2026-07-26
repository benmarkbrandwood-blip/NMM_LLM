# Mill Training Plan v5

Status: modular governing entry point, revised 25 July 2026.

This document defines the objective, current evidence boundary, decision path,
and authority of the v5 subdocuments. It is deliberately short. Detailed
rules, experiments, statistics, runtime design, release gates, and engineering
governance have separate owners under [`docs/v5/`](v5/).

The prior 1,744-line plan is retained only as a historical design snapshot in
[`v5-specialist-plan-legacy-2026-07-25.md`](v5-specialist-plan-legacy-2026-07-25.md).
It must not be used as an implementation contract when it conflicts with this
entry point, a modular specification, current code, or accepted evidence.

## Product Scope and Objective

The first v5 product is the separately named top-strength/research opponent
defined by the [product decision](v5/product-decision.md). It does not replace
the current ten difficulty levels, personality presets, adaptive difficulty,
tournament roster, or LLM coaching surfaces. Those modes retain their existing
contracts unless a later persona-specific decision explicitly adopts a v5
component.

v5 has three physically separate objectives, in this order:

1. Preserve legal play and every safety or evidence boundary claimed by the
   selected product mode.
2. Improve the preregistered practical match outcome against a defined target
   population without increasing the loss rate beyond its non-inferiority
   margin.
3. Measure human-facing style—patience, constriction, inducement, naturalness,
   and interpretability—only as a secondary objective that cannot override the
   first two.

This ordering governs the top-strength/research product. Style, teaching value,
and calibrated weakening may be primary objectives for a future coach,
personality, or adaptive product, but such a mode needs its own contract and
must not inherit the top-strength or verified claims.

The project does not claim to reproduce human cognition. A model may be
described as human-targeted only when it was trained against a declared human
distribution. It may be described as human-like only after a separate blinded
style evaluation supports that wording.

## Authority and Document Map

When two artifacts disagree, use the following order:

1. the signed product decision, rules variant, and primary normative rules
   sources named by their owning contract;
2. the owning modular specification below;
3. independently derived reference results and reviewed boundary examples;
4. executable conformance tests;
5. the engineering implementation;
6. experimental and operational evidence;
7. this entry point for navigation; and
8. the archived monolithic snapshot.

Tests are executable evidence that an implementation follows a frozen
contract; they cannot make an incorrect rule normative merely by passing.
Raw inputs, commands, hashes, and results remain immutable evidence. Their
interpretation and acceptance status remain corrigible when stronger evidence
shows that a specification, expected result, or test was wrong.

| Owner | Scope |
| --- | --- |
| [Product decision](v5/product-decision.md) | Initial v5 role, retained existing modes, persona objectives, and product constraints that must precede final architecture selection |
| [Oracle and rules](v5/oracle-and-rules-spec.md) | Decision state, atomic actions, Malom values, safe sets, history, W liveness, proof semantics, and deterministic acceptance |
| [Training research](v5/training-research-plan.md) | Corrected-v4/reference baselines, feasibility-first research, HumanPolicy/StrategyPolicy roles, minimal D specialisation, and optional escalation |
| [Human data and statistics](v5/human-data-and-statistics-plan.md) | Target population, HumanDB use, support and privacy gates, estimands, power, and human evaluation |
| [Compact runtime](v5/compact-runtime-design.md) | Product modes, architecture comparison, pack/prover candidate pools, runtime-aware training, availability, and target hardware |
| [Release acceptance](v5/release-acceptance-plan.md) | Candidate names, compressed release gates, claims, product trials, and promotion decisions |
| [Engineering governance](v5/engineering-governance.md) | Risk-proportionate planning, contracts, review, automatic-repair boundaries, and failure severity |
| [External-review disposition](v5/external-review-disposition-2026-07-25.md) | Evidence-based disposition of all 46 review findings and the resulting changes |

The current machine/run state remains owned by the
[Windows training handover](handoff/windows-training-2026-07-20.md). Local
asset relationships and machine-specific lookup keys remain owned by the
[local training layout](local-training-layout.md). Frozen experiments remain
under [`docs/experiments/`](experiments/); generated evidence remains under
[`docs/evidence/`](evidence/).

## Current Evidence Snapshot

The following statements are current as of this revision and supersede stale
implementation claims in the archived plan:

- The sector-corrected Malom decoder is implemented. Current code computes
  `absolute_key1 = raw_key1 + sector_value` and maps only the virtual extrema
  to W/L. The correction chain and focused Malom/provenance tests are accepted.
  A complete, independently accepted ultra-strong comparator and full-history
  RuleAwareSafetyGate are still future work.
- The in-repository compact `GameEngine` still does not implement general
  position-multiplicity repetition and does not correctly model the complete
  no-capture history required by the intended standard rules. It is not the
  formal referee.
- The corrected-v4 managed baseline was fresh-initialised, used a separate
  `sector-corrected-v1` SpecialistDB, disabled legacy Sentinel/ValueNet/GapNet,
  completed 5,000 games in 20 exact-resume segments, and passed its authorised
  Stage-0 training-signal diagnostic. That result is infrastructure and
  ablation evidence, not formal strength or promotion evidence.
- Legacy checkpoints, legacy SpecialistDB labels, and maintainer-`main`
  checkpoints with incomplete corrected-data lineage remain shadow or
  exploratory artifacts. This does not contaminate the fresh corrected-v4
  lineage and does not prohibit a new from-scratch corrected-v4 control.
- The pinned Sanmill strict logical-turn bridge has passed rule/history,
  deterministic fixed-work, error-policy, and local performance probes. No
  candidate-versus-baseline formal evaluation has been run through it.
- Maintainer `main` experiments involving Sentinel v2, HumanPrefNet, v2a/v2b
  recovery, and training graphs remain useful hypotheses and shadow evidence.
  They do not change the `dev` promotion path without lineage, frozen controls,
  and independent evaluation.
- No additional smoke, long training run, or candidate-versus-baseline run is
  authorised by this document revision.

## Product Modes Are a Decision, Not an Assumption

The product role is frozen, but its architecture is not. The feasibility phase
must compare at least these architectures before a release form is frozen:

| Mode | Move behaviour | Permitted claim |
| --- | --- | --- |
| `ordinary_always_move` | Always returns a legal move through a bounded policy/search path; may use best effort when proof is unavailable | Playing strength and measured risk only; no per-move safety claim |
| `verified_compact` | Selects only an action authorised by an exact pack or completed runtime proof; returns `runtime_unavailable` when none is authorised in budget | Only the exact per-move guarantee recorded by the authorisation artifact |
| `oracle_optional` | Uses a separately installed local tablebase or a versioned server oracle | Claims are limited to the mounted service, rules/history coverage, and verified interface |

These are distinct product contracts. The first is not a hidden fallback for
the second, and the second must not silently take over the first. A hybrid may
be evaluated, but each decision must record which authority selected the move.
No mode is the release default until target device, deployment form, costs,
availability, user experience, and redistribution constraints are compared.

## Feasibility-First Decision Path

The identifiers below replace the archived document's overlapping
Section/Stage/P-numbering. They describe dependencies, not calendar promises.

```text
P-1 product brief
└─ role is frozen; deployment, offline, device, cost, and resource limits
   must close before final architecture selection

F0 pre-E0 read-only feasibility
├─ F0-R0: reconcile current code, assets, rules, and accepted evidence
├─ F0-D0: raw-game reconstructability plus planning/untouched data freeze
├─ F0-H0: coarse specialisation upper screen; rejection only
├─ F0-P0: proof/history scale reconnaissance; rejection only
└─ F0-A0: architecture inventory under known product constraints

E0 deterministic foundation
└─ oracle comparator + authoritative rules/history + atomic actions

Post-E0 exact feasibility
├─ F0-H1: formal A_allow + approved minimal HumanPolicy headroom
├─ T0-H-pilot: non-promotable same-state/short-rollout comparison
├─ F0-P1: full-history proof and recursive-viability feasibility
└─ F0-A1: select one product lane under frozen deployment constraints

Selected product lane
├─ ordinary: strong policy/search, no online proof claim
├─ theory verified: recursive full-rule viability required
├─ positional exact: positional claim only
├─ bounded survival: finite-horizon claim only
└─ Oracle service: mounted local/server authority and cache

T0 reference baselines in the selected lane
└─ exact/classic baselines → direct q_ref where applicable → compact student
   → authorised search/pack variants → clean corrected-v4 control

T1 optional supported-error-pattern reranking
└─ only if F0-H1, T0-H-pilot, and T0 justify it; high-support D states only

E1 deployment closure
└─ train/evaluate against the exact runtime pack, prover, pool, deadline,
   scheduler, and product mode

R0 frozen acceptance
└─ correctness + risk/availability + match result + one product effect +
   target-device resources and claims
```

Pre-E0 F0 is allowed before building a comprehensive governance framework
because it is read-only, bounded, and cannot produce a release artifact. It may
produce reports and recommended thresholds, but it may not generate training
labels, modify databases, train models, or launch evaluation games. F0-H0 and
F0-P0 may reject a route; they cannot approve T1 or a verified product.

Post-E0 F0-H1/F0-P1 and T0-H-pilot use accepted deterministic semantics and
their own bounded experiment cards. They still cannot promote a candidate, and
an evaluation pilot still needs explicit launch authority.

The expensive HumanPolicy v3, full multi-step teacher, GapNet/SelfRiskNet,
DAgger, adaptive curriculum, and PPO stack remains closed until all applicable
F0-H1, F0-P1, T0-H-pilot, and T0 stop/go decisions pass. A negative
feasibility result is a valid project result and preserves the reference path.

## Required Early Decisions

### Candidate-pool alignment

The oracle training policy and a verified runtime policy must not silently rank
different games:

- `oracle_policy` is trained and normalised inside full-rule `A_allow`.
- `verified_runtime_policy` is trained on the exact pool produced by the
  deployed pack/prover/deadline contract.
- If the verified runtime contract permits an oracle-denied but
  horizon-proved action, that action requires runtime-policy supervision and
  must be reported as a weaker finite-horizon decision, not as theory-safe.
- If one policy is retained, verified mode may select only from
  `A_allow ∩ A_runtime`; an empty intersection is `runtime_unavailable`.
- Ordinary always-move mode may choose outside that intersection only under
  its separately named best-effort contract and without a safety claim.

Training horizon `X` and runtime horizon `X_rt` are different versioned
quantities unless explicitly proved equal. No model trained for one may be
presented as calibrated for the other without a matching evaluation.

### D-only specialisation and W conversion

The first target-specialisation experiment changes D decisions only.
Therefore:

- compare the compact reference with the current product for reference-path
  strength and W conversion;
- compare D-specialised treatment with the same-stack compact reference for a
  primary natural-match effect;
- require loss-rate and W-conversion non-inferiority for D-only
  specialisation;
- require W-conversion superiority only when W specialisation is explicitly
  enabled and powered.

The default natural-match primary endpoint is match score
`win + 0.5 * draw`; loss rate remains an independent non-inferiority gate.
Win rate is a key secondary endpoint unless a later product decision gives it
primary status and provides a coherent power calculation.

### Statistical margins

A superiority margin and an assumed true planning effect are not the same
quantity. If release requires `LCB(effect) > margin`, power must be computed
under a planning alternative strictly greater than that margin. A sample-size
example for testing against zero cannot justify a positive-margin gate.
`delta_specialize`, `delta_order`, human match effects, and W conversion all
follow this rule.

### W conversion and deployment feasibility

A positional W pack entry does not establish full-history W liveness. Before W
conversion becomes a foundational dependency, F0-P1 must identify the minimal
sufficient history state and measure proof graph size, cache reuse, proof
artifact size, and cold/warm completion across natural W, cyclic W, and
near-threshold histories. Until then, a pack W result may order candidates but
cannot alone claim a forced full-rules win.

## Baseline and Contamination Boundary

There are two different v4 categories:

1. legacy artifacts whose labels, trajectories, optimiser state, replay, or
   SpecialistDB may predate the decoder correction; these remain shadow-only
   for release lineage; and
2. from-scratch corrected-v4 runs built from verified inputs with isolated
   state and recorded resume lineage; these are valid controls or candidate
   sources once they pass the same evaluation gates as any other source.

Contamination ablation may explain old results but cannot repair lineage.
Promotion must never reuse contaminated optimiser state, replay, labels, or
checkpoints. The completed fresh corrected-v4 baseline belongs to category 2,
although its Stage-0 result remains too narrow for promotion.

## Stop Rules

- Stop optional specialisation when a valid F0-H0 bound is below the signed
  product effect, F0-H1 HumanPolicy support is inadequate, or the smallest
  student cannot retain a direct-teacher gain.
- Stop a verified runtime architecture when its proof/availability contract
  cannot close on target hardware; this does not prohibit an explicitly
  ordinary always-move product.
- Stop promotion on any false authorisation, unauthorised move, rule/oracle
  contradiction, corrupted checkpoint, data-identity drift, or invalid
  statistical design.
- Do not weaken a test, narrow a frozen confirmation domain, change a margin,
  or relabel a failure after observing candidate results.
- Do not start training or evaluation merely because a design document is
  complete. The owning experiment contract and explicit launch approval remain
  separate requirements.
