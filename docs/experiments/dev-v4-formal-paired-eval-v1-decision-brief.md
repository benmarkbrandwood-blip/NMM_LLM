# Formal Paired Evaluation v1 — Expert Decision Record

Date: 21 July 2026

Evaluation ID: `dev-v4-formal-paired-eval-v1`

Audience: product owner, evaluation reviewer, and domain expert

Status: **fatal stop; freeze and run are not authorized**

Related:

- [evaluation contract](dev-v4-formal-paired-eval-v1.md)
- [corpus review record](dev-v4-formal-paired-eval-v1-corpus-review.md)
- [training experiment](dev-v4-malom-corrected-baseline.md)

## Decision

The original 64-start proposal and the later one-endpoint-per-named-line
proposal are both rejected. The current runner also has a draw-lifecycle defect
that can abort a valid game and leave a non-restartable partial ledger.
Therefore no current corpus may be frozen and no paired evaluation may run.

After the runner defect and retry semantics are fixed and tested, the next
reviewable experiment may use the 109 unique, playable positions obtained by
projecting the 110 raw Sanmill move-oracle keys into NMM FEN. That experiment
is a **Stage-0 placement-opening training-signal diagnostic**, not a formal
playing-strength or promotion gate.

| Decision item | Recorded decision |
| --- | --- |
| A — start definition | Use the **109 unique playable projected Oracle positions** for the proposed repaired Stage-0 diagnostic. Reject one synthetic endpoint per named line. |
| B — source scope | Oracle-only for Stage 0. Named lines remain audit evidence, not the frozen start set. |
| C — phase coverage | All 109 starts are placement phase. There are no movement- or flying-phase starts. This is an explicit non-claim, not adequate release coverage. |
| D — workload | Exactly **109 colour-swapped pairs / 218 games**: one pair per unique start. No modulo reuse. |
| E — inference route | `policy-argmax-v1` is allowed only as a Stage-0 **lookahead-feature ablation** diagnostic. It is not training-route-aligned strength evidence. |
| F — opponent | Keep the architecture-matched scratch-init bundle as a training-gain control. It is not a product-strength baseline. |
| Freeze + run | **Deferred under fatal stop.** A new explicit product authorization is required after every prerequisite below is complete. |

## Purpose and claim boundary

The managed corrected-v4 run `managed-v4-baseline-v1` completed 5,000 games
and 20 segments. Its metrics prove neither playing strength nor promotion
readiness.

The narrow question that the proposed Stage-0 diagnostic may answer is:

> Under a deterministic zero-lookahead policy-argmax ablation, does the
> trained candidate score above an architecture-matched scratch
> initialization on this fixed, placement-only, source-overlapping convenience
> corpus?

Even an accepted result would not establish:

- strength under the training-time LookaheadAdvisor route;
- strength under fixed-node search, Sanmill search, or browser routing;
- movement- or flying-phase strength;
- generalization to held-out opening positions;
- release, publication, or promotion readiness.

## Independent corpus audit

### Named lines are not a well-defined one-line/one-position corpus

Sanmill contains 107 named lines. Their `lineMoves` omit the removal choice
after a mill-forming placement. Exhaustive legal replay found:

- **49 of 107** lines have multiple legal endpoints;
- those lines have between **2 and 42** legal endpoints;
- `novel-25964b79` cannot be replayed successfully;
- a successful endpoint of `novel-17e922b1` is already terminal and has no
  legal decision.

The draft generator resolves missing removals with a depth-first search that
prefers no-removal continuations and then sorts removal squares. Its selected
endpoint is therefore a constructive implementation choice, not a position
specified by the opening book. This invalidates the proposed A1a rule.
Enumerating every endpoint would instead overweight ambiguous lines and would
still produce synthetic endpoints.

The 107 lines also are not held-out training data:

- their source labels comprise 11 curated `book` records and 96 `learned`
  records; the latter split into 41 `book-*` and 55 `novel-*` identifiers;
- the local training books contain 143 records and 133 unique move sequences;
- **100 of 107** Sanmill named sequences exactly match the local training
  opening pool;
- all **11 of 11** curated lines match by identifier and exact sequence;
- training samples an opening line for 50% of games and forces the learner's
  first four placements from it;
- the training log does not record the sampled opening identifier, so exact
  per-line exposure cannot be reconstructed.

### The 64-position draft is rejected

The historical draft contains 64 exact FENs but only **63** distinct 16-way
ring-symmetry orbits. Its phase mix is placement 52 / movement 12 / flying 0.
It was a deterministic approximately five-percent slice of the available
book-derived orbit pool. It must not be relabelled or frozen as the final
corpus.

### Oracle projection is 109, not 110

Sanmill provides 110 raw move-oracle keys. Two keys differ only in a trailing
counter that NMM FEN does not encode, so projection produces exactly **109**
unique NMM FENs and 109 symmetry orbits. All 109 are playable.

All 109 starts are placement phase, with total placed-piece count from 0
through 16. There are **zero movement starts and zero flying starts**.
Games may later enter those phases, but that does not replace controlled
movement- or flying-start coverage.

Sanmill documentation describes the Oracle as its own engine-derived table,
not as a direct export of the NMM_LLM training opening book. It would therefore
be inaccurate to call every Oracle key the same artifact as the training book.
However, the overlap audit found:

- **28 of 109** Oracle orbits on a named-line trajectory;
- **23 of 109** Oracle orbits on the first eight plies of a named line.

Combined with the early-placement-only distribution and the named-line source
overlap, this corpus has not been demonstrated to be held out or
training-disjoint. The correct description is a
**source-overlapping, in-distribution-adjacent placement diagnostic**.

## Runner and statistical audit

### Fatal draw-lifecycle defect

The paired runner exits a game only when `winner is not None`. The game engine
can end through repetition or the 50-move rule with
`finished=True, winner=None`. The runner's board-only terminal check does not
observe that engine state. It then attempts another move and raises
`ValueError: Game is already over.`

A deterministic no-capture replay reaches this state at ply 117, within the
proposed `max_ply=200`. The failure is loud, but the runner opens the games
ledger with exclusive creation. A crash can therefore leave a partial ledger
that blocks a same-path retry. The fix must cover both game lifecycle and
partial-ledger recovery semantics, with focused regression tests.

### Repeated starts add no information

The route selects every move with pure `argmax`. Start selection is
`start_positions[pair % len(start_positions)]`, and the recorded seed is not
used for move selection. Repeating a start therefore reproduces the same
colour-swapped pair.

The old 64-start / 256-pair proposal copied each deterministic result four
times. Treating those copies as 256 observations would artificially narrow the
reported interval. For this route, `pairs` must equal the number of unique
starts: 109 pairs and 218 games.

The normal interval is an engineering summary of variation across the fixed
start corpus. It must not be described as a population confidence interval
unless its sampling assumptions are separately justified. The frozen report
must state this interpretation.

### Evaluation is a feature ablation

Training encodes policy inputs with a real `LookaheadAdvisor`. The current
paired route passes `lookahead_advisor=None`, which zeroes the 72-feature
lookahead block. Both bundles are treated symmetrically, but the candidate is
evaluated off its training input route. An inconclusive or rejected result
cannot distinguish weak learning from sensitivity to this ablation.

For that reason `policy-argmax-v1` is Stage-0 diagnostic evidence only. A
formal strength or promotion gate requires a separately implemented and frozen
training-route-aligned evaluator.

## Required prerequisites before a new freeze decision

All of the following are mandatory:

1. Fix engine-level draw handling in the paired runner.
2. Define recoverable or explicitly restartable partial-ledger behavior.
3. Add focused tests for repetition draw, 50-move draw, and retry behavior.
4. Generate a new Oracle-only list containing exactly 109 unique playable NMM
   FENs; do not reuse the rejected 64-position draft.
5. Record conversion provenance, overlap results, phase counts, orbit
   uniqueness, playability validation, and a new
   `start_positions_sha256`.
6. Have the owner review the exact 109-position artifact.
7. Freeze from a clean, tracked commit. At review time the branch was ahead of
   `origin/dev` and the experiment documents and draft artifacts were
   uncommitted; the live state must be rechecked rather than inferred from this
   historical observation.
8. Obtain a new explicit product decision authorizing freeze and run.

Until all eight are complete, no exact freeze or run command is approved.

## Inconclusive-result governance

An inconclusive Stage-0 result remains inconclusive. It may be followed by a
separately preregistered and independently frozen v2 experiment with additional
unique starts or a route-aligned evaluator. That is not prohibited peeking
provided:

- v2 is specified before any v2 outcomes are observed;
- v1 and v2 observations are not combined as one nominal sample;
- v1 is not rerun or selectively reinterpreted;
- v2 has its own corpus hash, work budget, interval interpretation, and claim
  boundary.

## Final authorization state

Candidate and scratch bundles have passed CPU verification, but that does not
clear the fatal stop. Freeze, paired execution, promotion, and publication
remain forbidden pending the prerequisites and a new product authorization.
