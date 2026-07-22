# Training-Aligned Paired Evaluation v1 — Product Decision Brief

Date: 23 July 2026

Status: **technical preparation complete up to the product choices; no
specification is frozen and no execution is authorized**.

Related:

- [completed Stage-0 result](../evidence/dev-v4-stage0-result-2026-07-23.md)
- [phase-corpus review record](dev-v4-phase-covered-corpus-v1-review.md)
- [managed training experiment](dev-v4-malom-corrected-baseline.md)

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

## Product decisions now required

### 1. Competent baseline

Recommended: use the current corrected-rules `GameAI` with deterministic
alpha-beta/PVS search, `use_mcts=false`, difficulty 10, no optional database or
learned-network components, and an explicit 500,000-node ceiling per move.

The frozen constructor and call contract should also state the currently
implicit defaults: `blunder_probability=0`, `search_threads=1`, `top_n=1`,
default `HeuristicWeights`, and no opening recognition, forced book move,
trajectory hint, game notation, endgame-state hint, position ban, star-square
mode, fork-variety mode, or n-gram search. In particular, the runner must call
`choose_move` without a recognition object. The default weights have
`opening_adherence=50`, but opening-book bonuses are inactive only when no
recognition or trajectory context is supplied.

Reasons for the recommendation:

- it is a competent non-random opponent rather than another training-gain
  control;
- a node ceiling is reproducible across machines, unlike wall-clock search;
- 500,000 nodes is the fixed budget used for the managed run's heuristic
  opponents, so the comparison has an existing operational reference;
- the native search disables its clock deadline when a node limit is present,
  and any unavailable or failed native backend is a hard failure rather than a
  Python fallback;
- it avoids assigning corrected-data lineage to a legacy maintainer
  checkpoint whose training state and route are incomplete.

The tradeoff is runtime. Before freeze, the implementation should benchmark
this exact baseline on representative placement, movement, and flying starts
and report the projected 128-game envelope. If that envelope is too large,
choose a smaller fixed node count before observing evaluation outcomes.

Not recommended:

- scratch initialization, because Stage 0 already answered that narrow
  training-signal question;
- a maintainer-`main` weights-only checkpoint, because its corrected-data
  lineage, full trainer state, and inference route are not established;
- a wall-clock baseline, because host load changes the effective work.

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
| Baseline | Fixed-node GameAI contract selected above |
| Maximum length | 200 ply; overflow is a draw |
| Random seed | 42 for provenance; neither policy may use random move choice |
| Result summary | Pair-score difference and a fixed-corpus engineering interval |
| Decision rule | Lower bound `> 0`: accept; upper bound `< 0`: reject; otherwise inconclusive |
| Stopping | Infrastructure or evidence-integrity failure only; no result-based early stop |

The interval describes variation across this fixed reviewed corpus. It is not
automatically a population confidence interval. Repeated deterministic starts
must not be counted as additional observations.

### 4. Launch authority

Do not grant launch authority as part of a vague approval. After the three
choices above, the remaining local work is to implement and test the new
paired runner, benchmark the fixed workload, freeze the reviewed corpus and
runtime contract, and publish a read-only readiness report. Starting the run
then requires a separate explicit instruction against that exact frozen spec.

## Current stop conditions

No original-maintainer technical clarification is required: code, checkpoint,
database, and fixture evidence resolve the route facts above. Work is paused
at product choices because changing the baseline, corpus-review requirement,
or workload changes the question being answered.

Until those choices are recorded:

- do not freeze a new evaluation specification;
- do not run a benchmark that records candidate-versus-baseline outcomes;
- do not start another evaluation or training run;
- do not reinterpret the Stage-0 `accepted` decision as promotion evidence.
