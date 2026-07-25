# Training-Aligned Paired Evaluation v1 — Product Decision Brief

Date: 23 July 2026

Status: **the next baseline direction is recorded, but no formal evaluation
specification or candidate-versus-baseline execution is authorized**.

Related:

- [completed Stage-0 result](../evidence/dev-v4-stage0-result-2026-07-23.md)
- [phase-corpus review record](dev-v4-phase-covered-corpus-v1-review.md)
- [managed training experiment](dev-v4-malom-corrected-baseline.md)
- [current Sanmill bridge contract](sanmill-strict-uci-bridge-smoke-v2.md)
- [current Sanmill bridge result](../evidence/sanmill-strict-uci-bridge-smoke-v2-2026-07-25.md)
- [complete Sanmill book-path contract](sanmill-book-path-corpus-v1.md)
- [complete Sanmill book-path artifact](sanmill-book-path-corpus-v1.json)
- [Sanmill prefix-diversity audit](../evidence/sanmill-prefix-diversity-audit-2026-07-25.md)

## What is now locally resolved

Stage 0 established a training signal but not playing strength. Its candidate
beat random initialization on 106 placement starts while both models received
zeroes for the 72 lookahead features used during training.

The candidate training input route can now be reconstructed without guessing:

| Item | Frozen fact |
| --- | --- |
| Route bundle | `316c345e918575b11efab3e0cfd618c1ab0e8411ff4a78c06512781af8764aed` |
| Final policy weights | `3a47c372187fbdeba797bd12bcb0e8304c94074c6957421272359b770d2faef4` |
| Frozen target weights | `12f6c0349576294a7ae5ca55c5e06cabc255bc007e4a35e3abf410aa0ba9d321` |
| Target age at final checkpoint | 6 games |
| Feature route | 134 floats: 62 base plus 12×6 lookahead |
| Simulated depth | 5 plies, padded to the 12-ply feature width |
| Learner continuation | Frozen-target argmax with zero-lookahead recursion break |
| Opponent continuation | HumanDB top frequency, then historical heuristic fallback |
| Terminal order | Project rules first, then corrected Malom |
| Enabled data | Bound HumanDB, final SpecialistDB, corrected Malom |
| Disabled networks | Sentinel, ValueNet, GapNet |

The loader verifies both model canaries and every bound identity, opens the two
SQLite databases read-only, and propagates dependency, decoder, feature, and
non-finite failures. A real local load reproduced all resource identities.

The historical rollout evaluator checks empty squares against `None`, whereas
`BoardState` stores an empty square as `""`. Its mobility terms are therefore
zero and its blocked-opponent term is inflated. The aligned route preserves
that observed training behavior deliberately. Correcting it inside this
evaluation would change the candidate's input distribution and create a
different experiment; a corrected evaluator may be studied later under a new
route name.

The 64-position phase-covered corpus is also generated and mechanically
audited. The Mill expert completed a quick first pass over every panel and
supplied a plausible move for each. The product disposition is not yet frozen.

## Recorded baseline direction

### 1. Competent baseline

The current `GameAI` is deferred as the formal baseline. Its search can be
made deterministic, but the surrounding compact position and game lifecycle
do not yet carry the full repetition and no-capture history required of the
formal referee.

The current strict logical-turn bridge passed against pinned Sanmill commit
`db65eb3e73189d934d615d0f47519d395193c646`. Sanmill owns the action history,
standard-rule lifecycle, and terminal outcome. The bridge disables shuffling,
uses one thread, a fixed seed, and one fixed-node ceiling per complete logical
turn. `StrictFailurePolicy` and `go logical` prevent Perfect DB, patch/trap,
depth-4, or random failure recovery from substituting a move. Machine-readable
`statejson` supplies rule identity, action and logical counts, no-capture and
repetition counters, terminal reason, and history SHA-256. Sanmill's
non-developer phase-depth policy remains active through
`DrawOnHumanExperience=true`; normal turns send no positive explicit depth.
HumanDB, the perfect database, patches, traps, and opening-book search were
disabled for this smoke.

The NMM opening-book source is now corrected on Sanmill `master`. The
pinned asset contains 109 entries and 437 unique recommendations; authoritative
replay found zero illegal and zero duplicate recommendations. The bridge still
leaves book play disabled. Sanmill now exposes fail-closed data-query
interfaces for the corrected book, HumanDB, and Perfect DB, so the
provider-interface blocker is closed. NMM_LLM now has a strict JSONL client
and a deterministic paired-prefix sampler. The remaining gate is a frozen
diversity and book-miss policy plus formal-runner integration, not an
unresolved provider or book-legality defect.

The bridge established rule consistency, one-budget compound-turn handling,
semantic replay reproducibility, and representative fixed-node performance. It
did not load a candidate or establish playing strength. The formal node budget
is deliberately not yet selected.

For a later infrastructure smoke, the provisional opening policy is 75%
corrected-book-derived prefixes and 25% StrictSteps perfect-database tied-best
prefixes. Perfect-database sampling covers exactly eight logical player moves
in total: four by each side, or four full rounds, not eight rounds. A
mill-forming move and its required staged removal count as one logical move,
even though UCI emits two action tokens. The sampler must use a frozen seed per
pair and replay the same prefix in both colour-swapped games. MTD(f) then
resumes with engine `Shuffling=false`. This ratio and prefix length are smoke
proposals only and are not yet a formal evaluation decision.

The implemented sampler does not contain that ratio as a default. It accepts
explicit integer source weights and an explicit candidate policy, uses
versioned SHA-256 draws, and records one prefix for both games of the pair.
Focused Sanmill bridge/query/prefix tests report `60 passed`; the complete
repository suite at `d6ea9f5` reports `1022 passed, 498 subtests passed`.

The corrected book is a sparse position-to-candidate source rather than a
complete depth-eight tree for every locally sampled branch. A fixed diagnostic
`pair-12` generated a byte-identical eight-ply prefix in two fresh processes,
while `pair-0` failed closed with `book_miss` before its sixth logical move.
The complete-path infrastructure option is now built and frozen separately.
It contains all 192 exact eight-logical-ply histories and records 508 shorter
`book_miss` leaves without fallback. The 192 histories end in 84 distinct
FENs but only seven distinct `ring16` endpoint orbits. Selecting uniformly
over histories or exact FENs therefore does not create broad structural
diversity.

A separate pre-result source audit generated 64 deterministic StrictSteps
prefixes twice in fresh processes. Both canonical record sets matched; all 64
histories, exact FENs, and `ring16` orbits were unique, with zero overlap
against the seven book orbits. Therefore the provisional 75% book / 25%
Perfect DB ratio is no longer recommended when the stated objective is
structural diversity. The current diversity-first draft is one representative
from each of the seven book orbits plus 57 orbit-unique Perfect DB prefixes.
This 7/57 draft is evidence-based but not yet a frozen product decision. A
runtime fallback from book miss to Perfect DB remains forbidden.

The configured Perfect DB passed a read-only initial-position source probe
with 24 StrictSteps ties and complete standard-sector coverage. The configured
HumanDB did not pass its source probe because a non-empty SQLite `-shm`
sidecar made the database non-immutable; it remains outside the prefix policy.

Not recommended:

- scratch initialization, because Stage 0 already answered that narrow
  training-signal question;
- a maintainer-`main` weights-only checkpoint, because its corrected-data
  lineage, full trainer state, and inference route are not established;
- a wall-clock baseline, because host load changes the effective work;
- the current `GameAI` as formal referee until its historical-rule state is
  either replaced or independently corrected and verified.

### 2. Corpus review and freeze

The Mill-domain expert completed a quick first pass over all 64 rendered
starts, gave a move he would choose for every position, identified several
unlikely or poor states, and described the spread as useful overall. He also
suggested adding positions where closing a Mill competes with blocking an
opponent Mill, preserving an approaching piece, or enabling a chain Mill.

This is meaningful domain feedback, but it is not an automatic corpus freeze.
The product owner must decide whether to accept the draft unchanged, replace
specified outliers, or add a separately identified tactical stratum. Apply
only pre-result, explicitly justified changes; regenerate and audit new
identities if the exact start list changes.

The corpus is legal, playable, phase-balanced, absent by exact lookup from the
bound HumanDB and final SpecialistDB, and labelled by corrected Malom. Its
source is nevertheless seeded rules replay rather than expert play. Human
review is therefore the remaining evidence for whether extreme or unnatural
states are acceptable for the intended strength claim.

### 3. Fixed workload and interpretation

Recommended initial contract, subject to corpus exclusions:

| Field | Recommended value |
| --- | --- |
| Pairs | One colour-role-swapped pair per accepted unique start |
| Current draft size | 64 pairs / 128 games |
| Candidate route | Exact `s-gen-v2-training-aligned-v1`, policy argmax |
| Baseline | Not frozen; strict fixed-node Sanmill logical-turn bridge v2 passed book-off validation |
| Maximum length | Not frozen; 60 complete turns is smoke-only and is not a rules draw |
| Random seed | 42 base seed; search is deterministic and only the frozen prefix sampler may choose among approved opening alternatives |
| Result summary | Pair-score difference and a fixed-corpus engineering interval |
| Decision rule | Lower bound `> 0`: accept; upper bound `< 0`: reject; otherwise inconclusive |
| Stopping | Infrastructure or evidence-integrity failure only; no result-based early stop |

The interval describes variation across this fixed reviewed corpus. It is not
automatically a population confidence interval. Repeated deterministic starts
must not be counted as additional observations.

### 4. Launch authority

The authorized strict Sanmill logical-turn bridge and its rule,
reproducibility, and performance report are complete. Safe next work is
limited to selecting a pre-result prefix distribution from the audited
sources, recording the phase-corpus disposition, and then implementing the
still-unfrozen formal runner. The paired-prefix client, sampler, and complete
book-path inventory are implemented. The remaining product choices include
whether HumanDB frequencies participate in prefix selection, the
book/Perfect DB objective and exact stratum construction, node budget,
history-bearing start representation, accepted phase corpus, game count, and
rules-compliant termination contract. If diversity is the objective, the
source audit recommends the orbit-stratified 7/57 draft rather than 75/25.
Starting candidate-versus-baseline games requires a separate explicit
instruction against a later frozen specification.

## Current stop conditions

No original-maintainer technical clarification is currently required for the
bridge: code, checkpoint, database, and fixture evidence resolve the route
facts above. Formal evaluation remains stopped at the paired-prefix policy,
phase-corpus disposition, workload, runner-audit, and launch gates. The
complete book-path inventory removes one implementation uncertainty but does
not make those product decisions automatically.

Until those choices are recorded:

- do not freeze a formal evaluation specification;
- do not run a benchmark that records candidate-versus-baseline outcomes;
- do not start another evaluation or training run;
- do not reinterpret the Stage-0 `accepted` decision as promotion evidence.
