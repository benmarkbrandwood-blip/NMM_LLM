# v5 Compact Runtime Design

Status: governing product-runtime and deployment-closure specification.

This document owns runtime product modes, candidate-pool construction,
pack/prover interaction, availability, target-device benchmarking, and
runtime-aware training. It does not assume in advance that a compact local
bundle is cheaper or better than an optional full DB or server oracle.

## Architecture Decision Comes First

F0-A compares at least:

1. ordinary local always-move AI;
2. verified compact AI without a pack;
3. verified compact AI with measured pack variants;
4. optional locally installed full tablebase;
5. server oracle;
6. local policy with optional oracle escalation.

For each, report:

- target users and offline/online operation;
- rule/history ownership;
- playing-strength and safety claim;
- unavailable/failure behaviour;
- local size, RSS, startup, latency, energy, and server/network costs;
- tablebase/pack licensing and redistribution;
- implementation and maintenance cost;
- privacy and availability implications.

No release architecture or size budget is governing until this comparison and
the target deployment device/form are frozen.

## Product Modes

### Ordinary always-move

`ordinary_always_move` always returns a legal move within a fixed work budget.
It may combine a compact policy with bounded search and a deterministic
best-effort failure route. It makes no per-move theory-safe or finite-horizon
guarantee. Its offline downgrade, strength, and failure rates are measured and
disclosed.

“Always move” does not mean random silent fallback. The legal fallback,
ordering, seed, and failure reason are explicit and deterministic under the
frozen contract. A search failure that violates the declared contract is still
an error.

### Verified compact

`verified_compact` selects only from:

```text
A_runtime = A_pack_exact union A_runtime_proved
```

subject to any stricter intersection with full-rule `A_allow` required by its
claim. If no eligible action is authorised within budget, it returns
`runtime_unavailable`. It never calls an ordinary best-effort path while
retaining a safety label.

### Oracle optional

`oracle_optional` uses a separately installed local tablebase or a pinned
server service. Its guarantees are limited by the service's variant,
rules-history handling, action atomicity, availability, and verified
comparator. Network or mount failure follows the product's separately stated
behaviour and cannot masquerade as an oracle decision.

## Rule and History Owner

The runtime owner must preserve:

- complete logical moves, including removal;
- repetition-equivalence history;
- no-capture/no-progress state and resets;
- terminal and claim semantics;
- exact replay of the returned action.

The current preferred bridge evidence uses pinned Sanmill strict logical turns
and `statejson`. It has passed infrastructure probes only. Any production or
evaluation use pins its source and binary identities, replays every returned
logical action in NMM_LLM, and compares the resulting state/history/terminal
identity. A later Sanmill revision is not adopted implicitly.

## Per-Candidate Pack and Prover Union

Pack use is not discarded because one unrelated successor misses:

1. query the current state exactly;
2. for each legal atomic candidate, query its settled successor;
3. add a candidate with sufficient exact fields to `A_pack_exact`;
4. send every unresolved candidate to the prover;
5. union the independently authorised candidates;
6. record `pack_exact`, proof tier, or both for every candidate.

For W/D, the pack establishes only the verified positional relation. W
liveness needs a full-history liveness or bounded-conversion certificate. For
L, the pack orders positional tiers but does not establish survival.

## Oracle and Runtime Policy Alignment

Maintain separate policy identities when the candidate games differ:

- `oracle_policy`: trained within `A_allow`;
- `verified_runtime_policy`: trained on the exact deployed authorised pool;
- optional `ordinary_policy`: trained/evaluated for the always-move contract.

If verified runtime can authorise an oracle-denied but horizon-surviving move,
the runtime policy must receive explicit supervision for comparing such moves,
and reports must call the choice finite-horizon rather than theory-safe.

If a single policy is used, verified mode is restricted to
`A_allow ∩ A_runtime`. Empty intersection means `runtime_unavailable`. A
runtime pool must not force a policy to rank candidates that training
systematically pushed to zero without a matching runtime objective.
At runtime, membership in the intersection must itself be established by a
deployable exact artifact or completed certificate; an unavailable oracle
lookup cannot be assumed true from an offline label.

`X` and `X_rt` have independent units, support, and versions. A safety prior
trained for one horizon may order proof work for the other only after
calibration; it does not authorise.

## Deployment-Aware Secondary Objective

Among actions tied under the applicable primary oracle/runtime tier, the
deployment teacher may prefer:

- lower proof cost;
- higher next-decision proof availability;
- pack closure;
- lower `runtime_unavailable` risk;
- lower support-escape risk;
- better W conversion under the frozen runtime.

These are lexicographic secondary fields. They cannot compensate for a lower
oracle tier in verified theory-preserving mode. Ordinary mode may optimise a
different measured utility, but it has a different name and claims.

## Proof Scheduler

The scheduler records legal, proved, refuted, and unresolved candidates.
Candidate order may use policy mass, a safety/proof prior, pack proximity, and
admissible bounds.

Early stop must not depend only on the current policy's raw mass. It requires:

- at least one authorised action;
- no unresolved action whose independent utility/safety upper bound can still
  displace the selected tier;
- the policy's highest-ranked unresolved action status;
- an independent proof-cost/safety prior;
- conservative bounds on unresolved candidates;
- a frozen random audit fraction of low-policy-mass unresolved actions.

The audit estimates how often policy-guided censoring hides a better
candidate. Thresholds do not relax under load. Partial proof, timeout, and “no
counterexample found” are `unknown`, never authorisation.

## Runtime Outcomes

Use distinct states:

| Outcome | Meaning |
| --- | --- |
| `authorised_pack` | Selected candidate has its own exact pack artifact for the stated positional claim |
| `authorised_proof` | Selected candidate has a completed proof for the stated `X_rt` property |
| `proof_pool_incomplete` | At least one action is authorised but other candidates remain unresolved |
| `survival_shortfall` | Complete analysis proves no candidate meets `X_rt`; optional best effort follows only a signed ordinary/shortfall rule |
| `runtime_unavailable` | No action is authorised before the deadline and some candidates remain unknown |
| `runtime_fault` | Schema, rule, proof, replay, or authority invariant failed |

Only one actual match result is recorded per game assignment. Availability and
fault events remain separate endpoints; they do not fabricate a second loss.

## Target Device and Budgets

Before setting model, pack, RSS, startup, or latency limits, freeze:

- deployment form and supported operating systems;
- representative minimum/median hardware;
- offline/online requirement;
- thread and work-accounting rules;
- energy or thermal constraints;
- acceptable visible latency;
- storage/download policy.

Then measure cold and warm p50/p95/p99 for the complete host path, not only a
standalone prover. Report proof completion and unavailable rates by phase,
history complexity, candidate count, pack path, and certificate tier.

Planning numbers such as 50 MB or 512 MB have no authority before this target
is fixed.

## Runtime Closure

E1 uses the exact production graph:

```text
authoritative state adapter
→ legal atomic candidates
→ policy/proof scheduling
→ per-candidate pack and prover
→ named product-mode selection
→ action replay and state/history verification
→ immutable event log
```

Legacy GameAI adjustments, Sentinel, specialist/generalist routers, direct
Malom branches, HumanDB, patches, traps, and database write-back cannot
silently co-decide or replace a failed authority. A component may participate
only when its role is named in the frozen mode and covered by route tests.

Closed-loop training-owned runs measure:

- support escape and OOD;
- pack/prover pool composition;
- theory downgrade under offline oracle replay;
- W conversion and L rescue;
- policy-guided proof censoring;
- unavailable/shortfall/fault;
- complete latency/resource distribution.

Confirmation or final states cannot be removed after observing an adverse
event.

## Claims

Per-move wording must identify the source:

- `pack-exact positional` for the exact positional relation actually checked;
- `runtime bound to X_rt` for a completed finite-horizon proof;
- `best effort` for ordinary or signed shortfall selection;
- `unavailable` when verified mode cannot authorise in budget.

`offline-verified` is aggregate evaluation evidence, not an online
certificate. A compact policy, safety prior, neural W/D/L head, or low observed
error rate must not be called perfect or theory-safe.
