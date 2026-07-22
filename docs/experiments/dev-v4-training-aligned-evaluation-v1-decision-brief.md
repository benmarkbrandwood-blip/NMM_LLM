# Training-Aligned Paired Evaluation v1 — Product Decision Brief

Date: 23 July 2026

Status: **the next baseline direction is recorded, but no formal evaluation
specification or candidate-versus-baseline execution is authorized**.

Related:

- [completed Stage-0 result](../evidence/dev-v4-stage0-result-2026-07-23.md)
- [phase-corpus review record](dev-v4-phase-covered-corpus-v1-review.md)
- [managed training experiment](dev-v4-malom-corrected-baseline.md)
- [authorized Sanmill bridge smoke](sanmill-strict-uci-bridge-smoke-v1.md)

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
audited. It is not yet domain-approved or frozen.

## Recorded baseline direction

### 1. Competent baseline

The current `GameAI` is deferred as the formal baseline. Its search can be
made deterministic, but the surrounding compact position and game lifecycle
do not yet carry the full repetition and no-capture history required of the
formal referee.

The selected next step is a strict bridge to pinned Sanmill commit
`da922965946ab87b3b3f9eed5b170f3e01d6c473`. Sanmill owns the action history,
standard-rule lifecycle, and terminal outcome. The bridge disables shuffling,
uses one thread, a fixed seed, and a fixed-node command, and fails rather than
using Sanmill's release-mode random recovery path. Optional databases,
patches, traps, and non-rule draw heuristics are disabled. The exact contract
and bounded authorization are recorded in the linked bridge-smoke document.

The bridge report must establish rule consistency, semantic replay
reproducibility, and representative fixed-node performance before Sanmill can
be proposed as the formal baseline. The node budget is deliberately not yet
selected.

Not recommended:

- scratch initialization, because Stage 0 already answered that narrow
  training-signal question;
- a maintainer-`main` weights-only checkpoint, because its corrected-data
  lineage, full trainer state, and inference route are not established;
- a wall-clock baseline, because host load changes the effective work;
- the current `GameAI` as formal referee until its historical-rule state is
  either replaced or independently corrected and verified.

### 2. Corpus review and freeze

Recommended: send all 64 rendered starts for Mill-domain review, apply only
explicitly justified exclusions, regenerate if needed, and then freeze one
pair per accepted ring16-unique start.

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
| Baseline | Not frozen; strict fixed-node Sanmill bridge under validation |
| Maximum length | Not frozen; 60 complete turns is smoke-only and is not a rules draw |
| Random seed | 42 for provenance; neither policy may use random move choice |
| Result summary | Pair-score difference and a fixed-corpus engineering interval |
| Decision rule | Lower bound `> 0`: accept; upper bound `< 0`: reject; otherwise inconclusive |
| Stopping | Infrastructure or evidence-integrity failure only; no result-based early stop |

The interval describes variation across this fixed reviewed corpus. It is not
automatically a population confidence interval. Repeated deterministic starts
must not be counted as additional observations.

### 4. Launch authority

The currently authorized work is only the strict Sanmill bridge and its rule,
reproducibility, and performance report. After that evidence is reviewed, the
remaining product choices include the formal node budget, history-bearing
start representation, accepted corpus, game count, and rules-compliant match
termination contract. Starting candidate-versus-baseline games requires a
separate explicit instruction against a later frozen specification.

## Current stop conditions

No original-maintainer technical clarification is currently required: code,
checkpoint, database, and fixture evidence resolve the route facts above. The
bounded Sanmill bridge validation may proceed, but formal evaluation remains
stopped at the later product gates.

Until those choices are recorded:

- do not freeze a formal evaluation specification;
- do not run a benchmark that records candidate-versus-baseline outcomes;
- do not start another evaluation or training run;
- do not reinterpret the Stage-0 `accepted` decision as promotion evidence.
