# Formal Paired Evaluation v1 — Expert Decision Record

Date: 21 July 2026; owner review updated 22 July 2026

Evaluation ID: `dev-v4-formal-paired-eval-v1`

Audience: product owner, evaluation reviewer, and domain expert

Status: **fatal stop; freeze and run are not authorized**. Runner repair and
owner corpus review are complete; a clean tracked freeze state, repeated
readiness evidence, and new authorization remain open.

Related:

- [evaluation contract](dev-v4-formal-paired-eval-v1.md)
- [corpus review record](dev-v4-formal-paired-eval-v1-corpus-review.md)
- [training experiment](dev-v4-malom-corrected-baseline.md)

## Decision

The original 64-start proposal and the later one-endpoint-per-named-line
proposal are both rejected. At review time, the runner also had a
draw-lifecycle defect that could abort a valid game and leave a
non-restartable partial ledger. Those runner defects and their focused
regressions are now repaired. The owner reviewed the corrected 107-candidate
package, requested removal of original review position 101, and accepted the
other 106. The owner-reviewed package is generated, but no corpus may be frozen
and no paired evaluation may run until the remaining freeze prerequisites are
complete.

The next reviewable experiment may use 106 unique, playable stable positions
selected from the 107 FENs obtained from the 108 Sanmill `action=p`
move-oracle keys. Original review position 101 is retained only as excluded
provenance. The other two of 110 raw keys are pending removals and are retained
only as successor provenance. That experiment is a **Stage-0
placement-opening training-signal diagnostic**, not a formal playing-strength
or promotion gate.

| Decision item | Recorded decision |
| --- | --- |
| A — start definition | Use the **106 owner-accepted unique playable stable Oracle positions** for the proposed repaired Stage-0 diagnostic. Exclude original review position 101; reject pending-removal keys and one synthetic endpoint per named line as direct starts. |
| B — source scope | Oracle-only for Stage 0. Named lines remain audit evidence, not the frozen start set. |
| C — phase coverage | All 106 starts are placement phase. There are no movement- or flying-phase starts. This is an explicit non-claim, not adequate release coverage. |
| D — workload | Exactly **106 colour-swapped pairs / 212 games**: one pair per unique start. No modulo reuse. |
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

### Oracle projection yields 107 candidates; owner review selects 106

Sanmill provides 110 raw move-oracle keys: 108 stable placement
(`action=p`) keys and two pending-removal (`action=r`) keys. NMM compact FEN
does not represent a pending staged removal, so treating all 110 as direct
starts loses game state. The two removals were instead applied for provenance;
their successors duplicate selected starts exactly or under ring16.

The 108 stable keys produce exactly **107** unique NMM FENs because one pair
differs only in a trailing counter that NMM FEN does not encode. The 107 FENs
also occupy 107 ring16 orbits and are all playable. The owner excluded original
review position 101 and accepted the other 106, which remain exact/ring16
unique.

All 106 selected starts are placement phase, with total placed-piece count from
0 through 16. There are **zero movement starts and zero flying starts**.
Games may later enter those phases, but that does not replace controlled
movement- or flying-start coverage.

Of 439 source move recommendations, 438 match NMM legal moves. One record
recommends occupied `c3`; the corresponding source candidate itself has 17
legal moves but is original review position 101. It is excluded from the
selected corpus and retained only as audit provenance. The defect prevents a
blanket claim that all source Oracle recommendations are valid.

Sanmill documentation describes the Oracle as its own engine-derived table,
not as a direct export of the NMM_LLM training opening book. It would therefore
be inaccurate to call every Oracle key the same artifact as the training book.
However, the overlap audit found:

- **28 of 106** selected Oracle orbits on a named-line trajectory;
- **23 of 106** selected Oracle orbits on the first eight plies of a named line.

Combined with the early-placement-only distribution and the named-line source
overlap, this corpus has not been demonstrated to be held out or
training-disjoint. The correct description is a
**source-overlapping, in-distribution-adjacent placement diagnostic**.

## Runner and statistical audit

### Resolved draw and partial-ledger defects

At review time, the paired runner exited a game only when `winner is not
None`. The game engine can end through repetition or the 50-move rule with
`finished=True, winner=None`. The runner's board-only terminal check did not
observe that engine state, so it attempted another move and raised
`ValueError: Game is already over.`

A deterministic no-capture replay reaches this state at ply 117, within the
proposed `max_ply=200`. The old runner also opened the final games ledger with
exclusive creation, so a crash could leave a partial final ledger that blocked
a same-path retry.

The runner now stops whenever `engine.finished` is true, preserves a draw as
`winner=None`, and records the engine's `draw_reason`. It writes incomplete
evidence to `<output>.partial`; a retry accepts only a same-spec, ordered,
hash-valid completed prefix and plays only missing games. A malformed or
mismatched prefix fails closed and is left untouched. A complete ledger is
recomputed before atomic publication to the immutable final output. The
focused evaluation suite reports `7 passed`, including repetition, 50-move,
valid retry, atomic publication, and malformed-prefix rejection.

### Repeated starts add no information

The route selects every move with pure `argmax`. Start selection is
`start_positions[pair % len(start_positions)]`, and the recorded seed is not
used for move selection. Repeating a start therefore reproduces the same
colour-swapped pair.

The old 64-start / 256-pair proposal copied each deterministic result four
times. Treating those copies as 256 observations would artificially narrow the
reported interval. For this route, `pairs` must equal the number of unique
starts: 106 pairs and 212 games.

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

Runner draw handling, recoverable valid-prefix resume, fail-closed malformed
evidence, and their focused tests are complete in the current change. The
following remain mandatory:

1. Freeze from a clean, tracked commit. At review time the branch was ahead of
   `origin/dev` and the experiment documents and draft artifacts were
   uncommitted; the live state must be rechecked rather than inferred from this
   historical observation.
2. Repeat the focused evaluation/readiness verification from that commit.
3. Obtain a new explicit product decision authorizing freeze and run.

Until all three are complete, no exact freeze or run command is approved.

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

Candidate and scratch bundles have passed CPU verification, and the runner
repair prerequisites are now complete. That does not clear the fatal stop.
Freeze, paired execution, promotion, and publication remain forbidden pending
the clean freeze state, repeated readiness evidence, and a new product
authorization.
