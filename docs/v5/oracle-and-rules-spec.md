# v5 Oracle and Rules Specification

Status: governing deterministic-semantics specification.

This document owns the meanings of game state, atomic action, Malom value,
safe/authorised candidate sets, history-aware proof, and deterministic
acceptance. It does not select a neural architecture, human target, product
default, or release threshold.

## Current Boundary

The current repository has already corrected sector-relative Malom decoding:
`absolute_key1 = raw_key1 + sector_value`, with only the virtual extrema
projected to W/L. Existing tests and sampled reference comparisons support
that correction.

The following work is not yet accepted merely because coarse W/D/L is now
correct:

- a complete field-preserving ultra-strong candidate comparator;
- one authoritative NMM_LLM rule/history state;
- general repetition multiplicity and correct no-capture progress/reset
  semantics in the formal path;
- full-history W-liveness proof and its feasibility envelope;
- one independently verified soundness interface shared by offline and
  runtime proof artifacts.

The pinned Sanmill strict logical-turn bridge is accepted infrastructure and an
independent rules/history reference at its recorded revision. It is not
permission to replace project semantics by an unreviewed later Sanmill state,
and it has not evaluated an NMM_LLM candidate.

## Formal State and Action

A formal decision consumes a versioned `DecisionState`, not a board-only FEN.
Its logically required information is:

```text
board occupancy
side to move
on-board and in-hand counts for both sides
rules variant
phase/transition state
repetition-equivalence history sufficient for adjudication
no-capture/no-progress state sufficient for adjudication
claim and terminal state
```

The exact minimal sufficient history representation is an F0-P1 deliverable.
Until it is proved, an implementation may use a conservative complete
representation, but it must not silently omit history to improve cache reuse.

An `AtomicAction` is a complete player decision:

- placement, movement, or flying move; and
- the required removal when that move closes a Mill.

An intermediate pending-removal state is valid inside a rules protocol, but is
not independently queried as a stable Malom position, policy candidate, or
training example. The action is settled, counts and phase are validated, the
side changes, and only then is the destination sector selected and queried.

## Oracle Value

The oracle interface returns a structured `OracleValue` rather than only a
scalar or `(wdl, steps)`:

```text
raw_key1
sector_value
absolute_key1
key2 and every comparator field required by the reference
entry kind and symmetry redirect
source and destination sector
original and actor perspective
query status
oracle/asset/schema/variant identity
```

The comparator must reproduce the independently pinned reference behaviour,
including perspective conversion, sector correction, `undo_negate`, and every
lexicographic field. A coarse W/D/L interface is insufficient for
ultra-strong tie ordering.

For every supported settled legal state, missing data, an unexpected entry
kind, a perspective contradiction, a sector mismatch, or an impossible
actor-created minimax upgrade is an implementation error. It is not a neutral
value and does not authorise fallback.

## Candidate Sets

The following sets have different meanings and must retain different names:

| Set | Meaning |
| --- | --- |
| `A_legal(S)` | Every legal settled atomic action from the rules engine |
| `A_pos(S)` | Actions preserving the best coarse positional W/D/L tier under the complete oracle comparator |
| `A_allow(S)` | Full-rule actions permitted after history, W-liveness, and L-mode horizon handling |
| `A_pack_exact(S)` | Individual candidates whose current state and successor have exact, verified pack values sufficient for that candidate's positional claim |
| `A_runtime_proved(S, X_rt)` | Individual candidates with a completed runtime proof of the declared finite-horizon property |
| `A_runtime(S)` | The pool a named product contract may actually select from |
| `K_theory` | States in a recursively closed full-rule viability set for one explicitly stated whole-game property |

For W and D, `A_pos` is non-empty when action enumeration and oracle semantics
are correct. An empty set is an adapter or enumeration defect, not evidence
that no safe move exists.

For W, `A_allow` contains only actions covered by a full-history winning
strategy. For D, it contains D-preserving actions after historical-rule
adjudication. For L, no action is “theory-safe” in the ordinary non-loss sense;
the contract first applies a finite-horizon survival threshold and then
separates rescue probability from mere delay.

`certW` means positional W plus a valid full-history winning proof for the
exact `DecisionState` and rules version. Positional W without that proof is
reported as `positional_W`; it is not silently counted as `certW`.

Keep three epistemic/deployment layers separate:

- `full_rule_value`: an exact game value only when an independently accepted
  complete method establishes it for the stated rules and support domain;
- `offline_certifiable`: the property can be proved under the frozen offline
  proof budget; and
- `runtime_authorisable`: the deployed pack/prover can establish the exact
  declared property within its runtime contract.

When no accepted complete method exists, use `offline_verified_W`,
`estimated_full_rule_value`, or another evidence-accurate name; do not call a
high-budget estimate `game_true_W`. Improving a prover may expand the latter
two sets without changing the game semantics. An ordinary strategic policy
therefore optimises terminal product utility subject to its rule constraints,
not whatever the current prover happens to certify cheaply. A verified product
may make offline or runtime certifiability an explicit constraint.

Within D, `q_ref` first requires full-rule D viability. Its versioned reference
order then considers remaining history slack and cycle/claim risk before the
complete positional comparator and final tie policy. A baseline that has only
the latter comparator is named `positional_ultra_strong_control`.

The three human-behaviour masses are also distinct:

- `positional_safe_mass`: HumanPolicy mass on `A_pos`;
- `full_rule_allow_mass`: HumanPolicy mass on `A_allow`;
- `runtime_authorisable_mass(X_rt)`: HumanPolicy mass on the exact runtime
  pool.

An L-tier quantity must not be named safe mass without an explicit
finite-horizon qualifier.

## Recursive Viability

A one-step exact value or finite-horizon proof does not establish that a later
decision remains authorisable. A whole-game `theory_preserving_verified` mode
requires a versioned invariant or viability kernel:

```text
K_theory = acceptable terminal states union
  {S: exists authorised a such that every legal opponent reply
      reaches K_theory or an acceptable terminal state}
```

The artifact binds the exact state/action semantics, rules and Oracle versions,
support domain, terminal property, covered opponent replies, and independent
verifier. The selected action must prove the quantified successor condition,
not merely prefer a state with historically high next-proof availability.

If the project can establish only a positional relation or an `X_rt` survival
bound, it uses the separate `positional_exact` or `bounded_survival` name.
Those valid local certificates do not imply recursive feasibility.

## Full-History W Liveness

A positional W value does not prove that the player can avoid a repetition or
no-progress draw. A W certificate requires an augmented-history AND/OR proof:

- node identity includes the history state sufficient for all applicable
  draw/claim rules;
- AI nodes need one proved lower-rank successor;
- opponent nodes require every legal reply, including a draw claim, to remain
  in the winning attractor;
- rank is generated by a monotone reachability proof, not assumed from raw
  `key2`;
- proof artifacts bind the root, variant, rules/oracle versions, explored
  actions, rank, and verifier.

Before W liveness becomes a reference-baseline dependency, F0-P1 must:

1. prove or conservatively define the minimal sufficient history key;
2. measure graph nodes, edges, strongly connected components, history depth,
   cache reuse, proof size, and memory;
3. cover natural W, cyclic W, capture/reset boundaries, and histories close to
   repetition/no-progress thresholds;
4. report offline and target-hardware cold/warm completion distributions.

If the pilot does not close, a positional W result remains a positional
ordering signal. It cannot be called a full-rules forced win.

## L-State Survival and Rescue

Let `D_survive(S, a)` be a proved lower bound on the number of plies before the
strongest legal opponent can force terminal loss after action `a`.

The default L ordering is:

1. reject any action that fails a product-required minimum survival threshold
   when another candidate satisfies it;
2. among candidates satisfying the threshold, prioritise conservative
   draw-rescue or comeback probability under a qualified teacher;
3. use additional survival distance and the complete Malom L ordering as
   secondary criteria;
4. when no candidate reaches the threshold, maximise the proved survival
   bound and report `survival_shortfall`.

This avoids treating purposeless delay as a primary playing objective.
`survival_shortfall` is not theoretical safety and cannot be reported as a
draw opportunity without separate evidence.

## Exact-Pack Semantics

Pack use is per candidate, not all-or-nothing:

1. the current state must have an exact verified pack value;
2. each successor with an exact verified value is independently considered
   for `A_pack_exact`;
3. candidates without a sufficient exact successor enter the runtime prover;
4. an experiment may schedule over the union of independently authorised pack
   and proof candidates, but each retains its own claim class;
5. every selected action records its own authorisation source.

For W/D, a pack candidate may establish only the positional claim its fields
support. For W, full-history liveness still requires a liveness certificate or
a bounded conversion certificate sufficient for the named claim. For L, pack
ordering alone never establishes a survival bound. An exact value for one L
successor also does not prove that it belongs to the best available L tier;
that comparative claim requires exact values or sound bounds for every
candidate capable of outranking it.

An implementation must not discard safe exact candidates merely because a
different successor misses the pack. Conversely, one exact successor must not
be used to infer values for candidates that were not queried.

## Independent Soundness Interface

Offline liveness, runtime survival proofs, and fast certificates share a small
soundness interface:

```text
root DecisionState hash
AtomicAction
claimed property and horizon
variant/rules/oracle identity
complete premises and covered opponent branches
proved/refuted/unknown status
independent verifier identity
```

`unknown`, timeout, partial expansion, a neural prior, policy confidence, and
“no counterexample yet” never authorise an action in a
theory-preserving, positional-exact, or bounded-survival mode. They may
influence search order only.

## Deterministic Acceptance

E0 acceptance requires:

- fieldwise agreement among the project adapter and independently pinned
  reference paths for all `OracleValue` fields and candidate order;
- exhaustive or property-based rule tests for placement, movement, flying,
  Mill/removal, terminal precedence, repetition, no-progress, and capture
  resets;
- colour, side-to-move, D4 action, sector, and perspective metamorphic tests;
- different histories for the same board producing the correct distinct
  result;
- proof soundness and root-binding fault injection;
- zero unsupported intermediate states entering trainable data;
- zero silent fallback on the formal path.

Finite random samples supplement these invariants; `0/n` is not a global
proof. Every accepted artifact records its support domain and all uncovered
domains.
