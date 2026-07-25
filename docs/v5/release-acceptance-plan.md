# v5 Release Acceptance Plan

Status: governing promotion, claims, and frozen-acceptance specification.

This document compresses release decisions into a small set of product gates.
Detailed model and teacher metrics remain available for diagnosis, but a
candidate is not forced to pass dozens of correlated exploratory metrics.

## Evidence Layers

| Layer | What it can establish |
| --- | --- |
| Deterministic conformance | Rules, action atomicity, oracle fields, proof soundness, route authority, and replay correctness within the stated domain |
| Offline model/runtime evaluation | Teacher following, calibration, downgrade/availability/resource rates, and runtime closure on frozen data |
| Frozen proxy matches | Performance against the named HumanPolicy/search/checkpoint opponents only |
| Prospective human trial | Effect for the preregistered participant population and product mode |

No layer inherits the claim of a later layer. A proxy result is not a human
result; `0/n` is not a proof; a training graph is not a strength evaluation.

## Candidate Names

| Name | Required evidence |
| --- | --- |
| `oracle_reference` | Accepted rules/oracle/proof path in the mounted workbench domain |
| `compact_reference_candidate` | Reference source completed E1 runtime closure in a named product mode |
| `compact_proxy_specialised_candidate` | D-specialised source completed E1 and passed frozen proxy/mechanism gates |
| `human_validated_product` | Complete product passed the applicable prospective human product-effect gate |
| `ordinary_measured_product` | Always-move product passed measured strength/risk/resource gates without a per-move safety claim |

“Safe,” “certified,” “human-like,” and “improved against humans” are not
candidate names and cannot be inferred from model provenance.

## Five Release Gate Families

### 1. Implementation correctness

Zero tolerance applies to:

- illegal or incomplete atomic action;
- wrong rule/history transition or terminal result;
- oracle/comparator contradiction in the stated domain;
- false pack/proof authorisation;
- selecting outside the named product pool;
- silent authority takeover or random failure fallback;
- schema/hash/replay corruption;
- product wording that asserts a guarantee the runtime did not produce.

These are deterministic failures, not rates to trade against playing strength.

### 2. Risk and availability

For the named product mode, freeze before confirmation:

- theory-downgrade estimand and cap;
- `runtime_unavailable`, shortfall, support-escape, and incomplete-pool
  definitions and caps;
- proof-censoring audit;
- start/game cluster and interval method;
- handling of unavoidable states and unknown audit results.

Ordinary always-move and verified compact products have different gates and
claims. Verified mode requires replayable authorisation for every move.
Ordinary mode reports measured downgrade/failure rates and never upgrades them
to a guarantee.

### 3. Match-result non-inferiority

Against the frozen current product and, where appropriate, the same-stack
compact reference:

- match score is non-inferior or superior according to the signed objective;
- loss rate is non-inferior;
- colours, starts, rules, work units, and opponent streams are paired or
  otherwise controlled;
- intervals use game/start/player/checkpoint clusters as applicable;
- the best seed is not selected as the estimate.

### 4. One primary product effect

A specialised product has one named primary product effect per experiment.
For the first D-only experiment, default to natural-match-score superiority
over the same-stack compact reference. W conversion is non-inferior, not
co-primary superiority.

If W specialisation is later enabled, a new objective may make conversion
superiority primary. If the product prioritises win rate rather than match
score, freeze that estimand and its power before results.

Human-like style requires a separate blinded style endpoint; it is not
inferred from match performance.

### 5. Resources, delivery, and claims

The final installed host path meets frozen target-device limits for:

- bundle/download size;
- startup RSS and time;
- cold/warm p50/p95/p99 move latency;
- energy/thermal limits when applicable;
- optional server/network availability;
- UI/API/error presentation.

The shipped SBOM/provenance inventory binds code, model, data, rule/oracle,
pack, configuration, host route, and user-visible claims.

## Diagnostic Metrics

The following remain diagnostic unless a separate product requirement promotes
one before results:

- HumanPolicy top-k, NLL, Brier, and calibration;
- safe/allow mass and squeeze;
- `P(certW)` and first-error rate;
- teacher CE/regret and group entropy;
- proof rank and proof cost;
- IW ESS and maximum weight share;
- style tags and scenario accuracy;
- training win/draw curves and policy top-1;
- rank flips and proxy-model gaps.

Diagnostics localise a failed gate and decide whether another research
experiment is worth its cost. They do not independently block a release that
passes its governing product gates, and they do not rescue a release that
fails one.

## Reference Release

A `compact_reference_candidate` may become a reference product without human
specialisation when it passes:

- implementation correctness;
- its named risk/availability contract;
- current-product match-result non-inferiority;
- target-device resources and claims.

It does not need a human-specialisation effect and cannot be described as
human-targeted merely because HumanDB was used for a diagnostic.

## D-Only Specialised Release

The first D-only treatment additionally requires:

- F0-H and direct-teacher headroom passed;
- same-stack compact reference as the attribution control;
- primary natural-match product effect passed;
- loss-rate non-inferiority;
- W-conversion non-inferiority;
- no material runtime risk/availability regression;
- stable direction under the frozen independent-model/search proxy panel.

The prospective human product-effect trial is required for an
`human_validated_product` name. Without it, the strongest result is
`compact_proxy_specialised_candidate`.

## W Conversion Evaluation

The W-start suite follows the natural-entry weighting in the
[human-data/statistics plan](human-data-and-statistics-plan.md). It reports
positional W, certW eligibility, pack/prover route, assigned state, and final
result separately.

A D-only candidate can change which W states are reached in natural games, so:

- controlled W-start conversion measures the frozen W/runtime subsystem;
- conversion after natural D→W entry is a transport/joint-chain diagnostic;
- neither is silently substituted for the other.

## Timing and Product Trials

Use the two trial estimands defined in the human-data/statistics plan:

- latency-matched strategy efficacy;
- real-latency product effect.

An unavailable event, timeout, or shortfall remains in the intent-to-treat
denominator according to the pre-signed product rule. Record one match result
per assignment and separate availability endpoints; never double count a
technical event as an additional synthetic loss.

## Statistical Decision

Every confirmatory endpoint has:

- target population and sampling unit;
- control and treatment identity;
- margin and planning alternative;
- power and maximum sample;
- clustering and multiplicity method;
- invalidity/attrition handling;
- one-time or group-sequential access rule.

Promotion requires the confidence bound to cross the signed margin. A result
is:

- `pass` when every applicable release gate passes;
- `fail` on a deterministic violation, prespecified harm, or futility rule;
- `inconclusive` when power, events, support, or interval separation is
  insufficient.

An inconclusive result retains the current product. It does not authorise a
smaller post hoc margin, another look at the same confirmation set, or removal
of difficult states.

## Evidence Ledger

For each candidate, retain an append-only ledger containing:

- candidate and parent lineage;
- code/data/rule/oracle/runtime/host hashes;
- endpoint and gate family;
- deterministic or statistical method;
- data access count and randomisation identity;
- raw denominator, clusters, effect, interval, and decision;
- failed, skipped, and unknown gates;
- artifact hashes and recomputation command.

The verifier recomputes decisions from raw immutable evidence. A handwritten
`passed: true`, screenshot, plot, or narrative is supporting context only.

## Release Claims

The final UI, API, README, status output, and error messages are checked against
the actual mode and evidence:

- ordinary mode says best effort/measured risk, not safe;
- pack mode names the exact positional relation actually checked;
- proof mode names the actual `X_rt` bound;
- unavailable and shortfall are visible rather than replaced by a move;
- human improvement is claimed only for the validated target population;
- human-like wording requires its own blinded style evidence.

A footer disclaimer cannot cure an overclaim in the primary product surface.
