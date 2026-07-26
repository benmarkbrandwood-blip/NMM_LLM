# v5 Compact Runtime Design

Status: governing product-runtime and deployment-closure specification.

This document owns runtime product modes, candidate-pool construction,
pack/prover interaction, availability, target-device benchmarking, and
runtime-aware training. It does not assume in advance that a compact local
bundle is cheaper or better than an optional full DB or server oracle.

## Architecture Decision Comes First

F0-A0 inventories and F0-A1 selects among at least:

1. ordinary local best-effort AI;
2. theory-preserving compact AI with a recursively closed strategy;
3. positional-exact and bounded-survival compact modes;
4. optional locally installed full tablebase;
5. server Oracle;
6. local policy with optional Oracle escalation.

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

### Ordinary best effort

`ordinary_best_effort` always returns a legal move within a fixed work budget.
It may combine a compact policy with bounded search and a deterministic
best-effort failure route. It makes no per-move theory-safe or finite-horizon
guarantee. Its offline downgrade, strength, and failure rates are measured and
disclosed.

“Always move” does not mean random silent fallback. The legal fallback,
ordering, seed, and failure reason are explicit and deterministic under the
frozen contract. A search failure that violates the declared contract is still
an error.

### Theory-preserving verified

`theory_preserving_verified` is a whole-game contract. It selects only an
action that is currently authorised and preserves membership in a recursively
closed full-rule viability set. It cannot be implemented by treating
next-decision proof availability as a preference.

Let `K` contain acceptable terminal states and states satisfying:

```text
S in K iff there exists an authorised action a such that
for every legal opponent reply b,
succ(S, a, b) is in K or is an acceptable terminal state
```

The exact quantifier structure follows atomic logical turns and the signed
rules variant. Membership, action coverage, opponent branches, and the
terminal/invariant base bind to a verifiable artifact. If recursive closure
cannot be constructed or conservatively established for the support domain,
this mode does not exist for that domain.

### Positional exact

`positional_exact` selects an action with exact pack fields sufficient for the
stated positional relation. It makes no full-history liveness, recursive
availability, or whole-game claim. Missing exact coverage is
`runtime_unavailable` for this mode.

### Bounded survival

`bounded_survival` selects an action only after a completed proof of the stated
`X_rt` property. The certificate applies to that move and horizon. It does not
promise that the next decision has an authorised action. A later unavailable
state does not retroactively falsify the prior bound, but it does prevent the
product from presenting the game as continuously verified.

### Oracle service

`oracle_service` uses a separately installed local tablebase or a pinned
server service. Its guarantees are limited by the service's variant,
rules-history handling, action atomicity, availability, and verified
comparator. Network or mount failure follows the product's separately stated
behaviour and cannot masquerade as an oracle decision.

`compact_verified_family` is an internal architecture label for experiments
that compare these mechanisms. It is not a product mode, UI badge, or safety
claim.

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

The union is a scheduling/data structure, not a uniform guarantee. A
positional pack candidate and an `X_rt`-proved candidate remain in different
claim classes. They may be compared by a hybrid experiment only when the
selected action records and displays its actual class. Neither class enters
`theory_preserving_verified` without a recursive full-rule viability artifact.

## Oracle and Runtime Policy Alignment

Maintain separate policy identities when the candidate games differ:

- `oracle_policy`: trained within `A_allow`;
- `theory_policy`: trained within the recursively viable full-rule pool;
- `positional_policy`: trained on exact positional candidates;
- `bounded_survival_policy`: trained on the exact deployed proof pool;
- optional `ordinary_policy`: trained/evaluated for best effort.

If bounded-survival runtime can authorise an Oracle-denied but
horizon-surviving move, that policy must receive explicit supervision for
comparing such moves, and reports must call the choice finite-horizon rather
than theory-preserving.

If a single policy is used, each named mode is restricted to its exact pool.
For a theory-preserving mode that includes runtime proof, this is at least
`A_allow ∩ A_runtime ∩ K`. Empty intersection means
`runtime_unavailable`. A runtime pool must not force a policy to rank
candidates that training systematically pushed to zero without a matching
runtime objective.
At runtime, membership in the intersection must itself be established by a
deployable exact artifact or completed certificate; an unavailable oracle
lookup cannot be assumed true from an offline label.

`X` and `X_rt` have independent units, support, and versions. A safety prior
trained for one horizon may order proof work for the other only after
calibration; it does not authorise.

## Deployment-Aware Secondary Objective

Among actions tied under the applicable primary Oracle/runtime tier, the
deployment teacher may prefer:

- lower proof cost;
- higher next-decision proof availability in positional-exact or
  bounded-survival modes;
- pack closure;
- lower `runtime_unavailable` risk;
- lower support-escape risk;
- better W conversion under the frozen runtime.

These are lexicographic secondary fields. In theory-preserving mode, recursive
membership is a hard condition, not a tie-break. Secondary fields cannot
compensate for a lower Oracle tier. Ordinary mode may optimise a different
measured utility, but it has a different name and claims.

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
| `authorised_theory` | Selected candidate has a recursive full-rule viability artifact for the stated whole-game property |
| `authorised_positional` | Selected candidate has its own exact pack artifact for the stated positional claim |
| `authorised_bounded` | Selected candidate has a completed proof for the stated `X_rt` property |
| `proof_pool_incomplete` | At least one action is authorised but other candidates remain unresolved |
| `survival_shortfall` | Complete analysis proves no candidate meets `X_rt`; optional best effort follows only a signed ordinary/shortfall rule |
| `runtime_unavailable` | The selected named mode has no action authorised before the deadline or outside its support domain |
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

- `theory-preserving within <domain>` only for an action retaining membership
  in the recursively verified full-rule set;
- `pack-exact positional` for the exact positional relation actually checked;
- `runtime bound to X_rt` for a completed finite-horizon proof;
- `best effort` for ordinary or signed shortfall selection;
- `unavailable` when the named proof-bearing mode cannot authorise in budget.

The UI must not replace those names with one undifferentiated “verified”
indicator. A game may claim continuous theory-preserving play only when every
decision remains in the declared recursive support domain.

`offline-verified` is aggregate evaluation evidence, not an online
certificate. A compact policy, safety prior, neural W/D/L head, or low observed
error rate must not be called perfect or theory-safe.
