# v4 Infrastructure Hardening Plan

## Status and Purpose

This document is the English, persisted version of the infrastructure plan
derived from reviews of Stockfish, fishtest, WDL_model, nnue-pytorch, lc0,
lczero-training, and the Lightvector repositories. It converts reusable ideas
from those projects into a prioritized plan for this repository.

The primary target is the current, defect-corrected v4-style Generalist path,
especially `scripts/train_s_gen_v2.py`. The broader v5 document remains a
design and risk library. This plan does **not** require complete v5 execution,
and it does not make optional v5 research a prerequisite for a corrected v4
long run.

The present operating boundary is one Windows machine and one CUDA GPU.
Distributed training, multi-host orchestration, and a C++ data loader are not
part of this plan.

Implementation status on 20 July 2026: Slices 1 through 6 are implemented and
validated. Slice 7 completed its baseline probe, but its optimization trigger
did not activate because the current online serial rollout has no persistent
loader. P3 remains optional and was not executed. A long run remains subject
to an immutable Agent-selected technical plan, a separate product decision on
the objective and resource envelope, and a newly authorized final smoke. The
managed authority boundary is documented in
[`docs/managed-training-operations.md`](managed-training-operations.md); see
[`docs/evidence/v4-infrastructure-hardening-2026-07-20.md`](evidence/v4-infrastructure-hardening-2026-07-20.md).

## Executive Priority Order

| Priority | Outcome | When it is required |
| --- | --- | --- |
| **P0** | Make a v4 run reproducible, recoverable, provenance-safe, and fail-closed | Before any long corrected v4 run |
| **P1** | Make candidates portable, independently evaluable, and promotable through frozen evidence | Before calling any model accepted or release-worthy |
| **P2** | Improve the local data path only when profiling proves that it limits throughput | Conditional on measured bottlenecks or the introduction of persistent replay/offline data |
| **P3** | Explore calibration, compression, new heads, or later v5 ideas through isolated ablations | Optional and separately authorized |

The shortest execution path is therefore:

1. harden the current v4 run contract;
2. run a bounded integration smoke and exact-resume checks;
3. freeze the Agent-selected technical choices and obtain separate product
   authorization for the objective and resource envelope;
4. run the corrected v4 experiment on one machine and one GPU;
5. evaluate saved candidates with a frozen local protocol;
6. optimize the loader or activate later research only when evidence justifies
   the additional complexity.

## 1. Fixed Scope and Decision Boundary

### 1.1 Current training target

The first target remains the experiment already defined in
[`docs/experiments/dev-v4-malom-corrected-baseline.md`](experiments/dev-v4-malom-corrected-baseline.md):

- fresh/random model initialization;
- the corrected v4-style Generalist path;
- no unscoped automatic resume; managed continuation is explicit
  `exact-resume` from the verified preceding segment;
- Sentinel disabled;
- ValueNet disabled;
- GapNet disabled;
- PPO disabled;
- the trusted, isolated `sector-corrected-v1` SpecialistDB;
- HumanDB human frequencies and empirical outcomes allowed, while its
  historical unversioned Malom columns remain masked;
- all historical model artifacts retained as lineage-labeled, exploratory
  comparisons rather than described as corrected retraining products.

Checkpoint envelope v2 now supports complete trainer-state recovery for
checkpoints created by the hardened Generalist path. Historical checkpoints
remain weights-continuation artifacts and may enter only through the explicit
`weights-only` path.

### 1.2 Engineering boundary

The default implementation language remains Python/PyTorch. The repository
already has a Rust/PyO3 native boundary under `native/nmm_core`; it may be used
only for a hot path that is first identified by end-to-end profiling and has a
Python reference implementation.

This plan introduces no C++, CMake, C++ loader, or second native build system.
It also introduces no distributed trainer, worker fleet, coordinator service,
lease server, or remote result database. Local CPU preprocessing is allowed,
but CUDA training and CUDA evaluation must be serialized on the single GPU.

### 1.3 Algorithm boundary

P0 and P1 are infrastructure work. They must not silently change reward
semantics, model architecture, curriculum, opponent mixture, temperature
schedule, search budget, or promotion thresholds. Any such change is a named
experiment with a frozen baseline and an isolated ablation.

Before a long run, the Agent must conservatively freeze and record:

- update algorithm;
- opponent schedule;
- temperature start;
- game budget;
- seed;
- `batch_games`/local concurrency;
- checkpoint cadence;
- monitoring cadence;
- stop and quarantine criteria.

The product owner is not expected to choose these ML or infrastructure
parameters. Infrastructure validates and records the Agent's choices, and the
Agent must not revise them after results are visible. The product owner decides
only the objective, total resource envelope, launch, later resource expansion,
and publication or promotion.

## 2. What to Reuse from the Reference Projects

The reference projects are sources of patterns, not architectural mandates.
The aim is to copy the discipline of their contracts and verification while
implementing the minimum mechanisms appropriate for this repository.

| Reference | Ideas worth adapting | Ideas intentionally not imported now |
| --- | --- | --- |
| **Stockfish** | Small deterministic correctness corpora analogous to perft; fixed-work benchmark signatures; reproducible search positions; explicit network validation; separate correctness and performance measurements | Chess-specific engine structure, NNUE implementation details, C++ build layout, or thread architecture |
| **fishtest** | Paired games with colors/roles swapped; immutable raw game evidence; recomputation of results; explicit bad-result quarantine; confidence-aware sequential testing after simulation validates its operating characteristics | Distributed worker/server infrastructure, contribution identities, leases, heartbeats, remote task scheduling, and fleet administration |
| **WDL_model** | Phase-conditioned WDL calibration for interpretation and reporting; explicit fitted-parameter provenance | Treating a fitted empirical curve as theoretical truth or as a replacement for Malom/rules evidence |
| **nnue-pytorch** | Strict experiment configuration; separation of model import from exact resume; hashed serialization; cross-implementation checks; end-to-end smoke tests | C++ training or inference integration, NNUE-specific sparse features, and a new native loader by default |
| **lc0** | A self-describing model package; explicit input/head semantics; `describe`, `verify`, and backend benchmark operations; stable model identity | Reproducing the entire network format, plugin/backend ecosystem, or C++ runtime |
| **lczero-training** | Typed root configuration; explicit checkpoint migration with dry-run; staged local loading; loader probes that expose shapes, rates, memory, and wait time | Distributed training, framework migration for its own sake, Protobuf as a default, or a persistent data daemon |
| **KataGo** | Strict configuration, atomic checkpoints and backups, interruption-safe state, data freshness budgets, canary positions, symmetry checks, and careful lifecycle separation between training, self-play evidence, and accepted networks | Its distributed self-play system, C++ engine, giant configuration surface, or a simple imitation of its gating/window policy without adapting statistics to this project |
| **goscorer** | A small shared corpus of input positions and exact expected outputs that can be consumed by multiple implementations | Go scoring semantics themselves |
| **GoNN** | Dataset splitting by game identity rather than adjacent positions; an explicit failed-experiment and negative-result ledger | Legacy TensorFlow training code or Go-specific architectures |
| **leela-analysis** | Cache identity that includes search/model budget and all semantics capable of changing the result | Its engine protocol and Go-specific analysis format |
| **ArimaaSharp / fireflower** | Historical examples of independent game-engine tests and quality harnesses | No direct implementation import is currently justified |

### 2.1 Adaptation rule

Every adapted mechanism must answer four questions:

1. Which current v4 failure mode does it prevent or expose?
2. What is the smallest public contract needed here?
3. What acceptance test proves that the mechanism works in this repository?
4. What complexity is deliberately left out until a measured trigger occurs?

The reference repositories must not become runtime dependencies merely because
they inspired a mechanism.

## 3. P0 — Required Before a Long Corrected v4 Run

P0 is the minimum infrastructure closure. A long run must not begin merely
because the one-game integration path launches; it must be possible to explain
exactly what ran, recover it without guesswork, and reject contaminated or
ambiguous inputs before side effects occur.

### P0-A. Canonical experiment configuration and run contract

Create one canonical resolved configuration for each run. CLI flags and the
machine-local path file may remain convenient inputs, but the trainer must
resolve them into a typed, immutable snapshot before opening databases,
creating an output directory, or initializing CUDA state.

The configuration loader must reject:

- unknown keys;
- duplicate assignments that do not use an explicitly documented precedence
  rule;
- invalid types;
- non-finite numeric values;
- out-of-range values;
- mutually incompatible options;
- missing required assets;
- output-directory reuse that is not an explicit resume;
- a resume checkpoint whose declared run, schema, or asset lineage conflicts
  with the requested mode.

Legacy CLI import may be supported, but it must produce the same canonical
configuration and record how every final value was obtained. It must not form
a second, less strict execution path.

Provide preflight modes for at least:

- `smoke`;
- `long-run`;
- `weights-only` import;
- `exact-resume`.

Preflight must complete without mutating training databases or model output.
It should report every resolved logical path, existence check, schema/version,
asset identity, component enable/disable state, checkpoint mode, and claim
boundary. Host-specific absolute paths stay in ignored local configuration;
tracked manifests use logical keys and relative artifact identities where
possible.

#### Run manifest

Each run receives an immutable `RunManifest` containing at least:

- manifest schema version;
- run ID and experiment ID;
- parent run ID, when applicable;
- Git commit and dirty-diff identity;
- exact command and canonical resolved configuration;
- configuration hash;
- Python, PyTorch, CUDA, driver, OS, and GPU identity;
- global and component-specific seeds;
- input asset names, hashes or frozen manifest identities, schemas, and
  intended use;
- model architecture and feature schema;
- explicitly enabled and disabled components;
- output, database, checkpoint, and logging policy;
- checkpoint mode and lineage;
- permitted evidence claims;
- lifecycle status and timestamps.

Publish the manifest atomically. Append lifecycle events to a structured event
ledger rather than rewriting history in place. Expected statuses include
`planned`, `preflight_passed`, `running`, `interrupted`, `failed`, `completed`,
and `quarantined`. A status must never hide the underlying terminal reason.

#### P0-A exit criteria

- Two equivalent invocations produce byte-identical canonical configuration
  JSON apart from explicitly excluded run-instance fields.
- Invalid configuration fails before any output directory, database row, or
  checkpoint is created.
- A run can be reconstructed from its manifest without consulting console
  recollection or machine-local notes.
- The manifest clearly states that the current experiment is corrected v4,
  not a completed v5 implementation.

#### Managed authority gate

Long-run readiness also requires an immutable managed plan and a separate
authorization file bound to the exact plan SHA-256. The plan records the Git
commit, path-config identity, resume-semantics hash, game and segment bounds,
wall-time limit, safe exact-resume policy, and the prohibition on publication
or promotion. Plan creation has no launch authority.

The supervisor owns fresh first-segment construction, exact-resume lineage,
subprocess time bounds, single-controller locking, run-ledger verification,
checkpoint validation, and product-readable status. Technical failures route
to Agent review. Only launch and resource-scope changes route to the product
owner.

### P0-B. Checkpoint envelope v2 and explicit recovery semantics

Introduce a versioned checkpoint envelope. It must save enough state for exact
resume, not only model weights.

The envelope should include:

- checkpoint schema version;
- run, experiment, parent, and checkpoint IDs;
- checkpoint role and save reason;
- model state;
- optimizer state;
- scheduler and gradient-scaler state, when present;
- Python RNG state;
- NumPy RNG state;
- PyTorch CPU RNG state;
- all relevant CUDA RNG states;
- completed game, batch, and update counters;
- difficulty and temperature state;
- rolling metrics and recovery-window state;
- curriculum state;
- target-network age/state, where applicable;
- data cursor and snapshot-consumption state;
- cache/bucket state only when its omission would change the future sample
  sequence;
- canonical configuration hash;
- feature, label, database, and model schema versions;
- input asset manifest identities;
- creation time, software identity, and integrity hash.

Checkpoint publication must use this sequence:

1. write to a temporary file in the destination filesystem;
2. flush and close it;
3. reload or structurally verify it;
4. compute and record SHA-256;
5. rotate a bounded set of previous known-good files, such as `prev0` through
   `prev3`;
6. publish with `os.replace`.

A corrupted newest checkpoint must produce an explicit failure or quarantine
result. The loader must not silently fall back to a previous file and continue
under an incorrect lineage claim.

#### Three different start modes

The user-facing contract must distinguish:

- `fresh`: initialize all model and trainer state from the resolved
  configuration;
- `weights-only`: import compatible weights, intentionally reset optimizer,
  RNG, counters, rolling state, and curriculum, and record the new lineage;
- `exact-resume`: restore the complete checkpointed trainer state and reject
  any incompatible or missing required state.

Historical v4 checkpoints are `weights-only`. They must not be relabeled as
exactly resumable.

#### Checkpoint roles

Keep these identities separate:

- `latest`: newest continuation state;
- `best_train`: optional training-metric snapshot, never formal strength
  evidence by itself;
- `candidate`: immutable artifact submitted to frozen validation/evaluation;
- `accepted`: immutable artifact that passed the declared evaluation contract.

No save operation should overwrite an already accepted artifact.

#### Inspection and migration

Provide an inspection/migration command that can:

- describe an old or new checkpoint without loading it into a trainer;
- verify integrity and schema;
- show a complete compatibility diff against a target configuration;
- dry-run a migration;
- write a new file without overwriting the source;
- run model canaries before declaring migration successful.

Migration rules must be explicit. Shape coincidence is not proof of semantic
compatibility.

#### P0-B exit criteria

- Kill or interrupt the process during checkpoint publication; the last
  published checkpoint remains readable.
- Corruption and truncation are detected.
- `fresh`, `weights-only`, and `exact-resume` cannot be confused.
- A one-game run, exact resume, and one more game match the supported
  continuous two-game reference within the declared determinism tolerance.

### P0-C. Deterministic game identity, scheduling, and resume

Each game must receive a stable ID and seed derived from immutable inputs such
as:

- run seed;
- game index;
- player/role assignment;
- curriculum or experiment branch.

Use a stable hash, not Python's process-randomized hash. Decisions inside a
game must use a game-local RNG or explicit substream. Shared global RNG calls
must not make results depend on thread scheduling.

When `batch_games > 1`:

- assign game IDs, seeds, roles, and branches before submission;
- preserve a fixed merge/update order;
- do not allow task completion order to change the training sequence;
- record incomplete and retried games without silently reusing an identity for
  different evidence.

The strict reference begins with `batch_games=1`, because that is the simplest
way to prove resume parity. Batched determinism should then be tested
separately; it must not be assumed from the serial result.

PyTorch/CUDA operations known to be nondeterministic must be identified. If a
fully deterministic implementation is unavailable or unacceptably slow, the
manifest must record the exception and the acceptance test must use a frozen,
quantified tolerance rather than vague similarity.

#### P0-C exit criteria

- The same fresh serial run produces the same game IDs, branches, actions, and
  supported numeric outputs.
- Exact resume preserves the future sequence.
- Increasing local preprocessing concurrency does not reassign games or
  change merge order.
- Any residual CUDA variation is measured and documented.

### P0-D. Data, label, provenance, and cache firewall

#### Asset manifests

Every authoritative or training-relevant asset needs a manifest containing:

- logical asset name and role;
- source/provenance;
- schema/version;
- size and stable identity;
- creation or import process;
- trust level;
- allowed consumers;
- validation performed;
- known exclusions or contamination boundaries.

The large Malom asset should be represented by a frozen manifest with its
validated component inventory and hashes. Do not recompute a complete
approximately 83.6 GB hash tree on every launch; verify the frozen manifest
identity plus cheap structural checks during routine preflight, and reserve a
full audit for explicit validation.

#### Physical label separation

Do not overload a generic `value` field. Store or expose distinct typed labels
for:

- `theoretical_wdl`;
- `empirical_outcome`;
- `human_observation`;
- `teacher_score`;
- `model_prediction`.

Each label must carry perspective, rules/history requirements, source identity,
and validity/version. Transformations between perspectives must be explicit
and testable.

#### Malom and SpecialistDB rules

- Persisted Malom labels are trusted only at
  `malom_label_version=sector-corrected-v1`.
- A trusted label must not be overwritten by a lower-authority observation.
- A missing counterfactual remains missing; it must not be synthesized as a
  neutral value.
- Mill formation plus capture remains one atomic action for oracle queries and
  cache identity.
- Terminal resolution and perspective must be explicit.
- Database writes must be atomic and return structured error reasons.
- Each experiment uses an isolated SpecialistDB unless an explicitly frozen,
  read-only snapshot has been approved.

Historical SpecialistDB and model artifacts remain shadow inputs or
comparisons. They do not enter the corrected trusted store through a relabeling
shortcut.

#### Human and runtime data

HumanDB empirical frequencies and outcomes may remain useful under their
existing provenance boundary. This plan does not require a HumanDB v3
migration before corrected v4 training.

Runtime/user-generated data must enter telemetry or quarantine storage first.
It must not flow directly into the trusted training database. Training consumes
only explicit, frozen snapshots whose lineage is recorded by the run manifest.

#### Cache identity

A cache key must include every factor that can change semantics, including:

- canonical state and relevant history;
- rules mode;
- perspective;
- pending capture/atomic-action state;
- search or rollout budget;
- model/bundle identity;
- feature schema;
- input asset manifest identities;
- configuration hash or the result-affecting subset of it.

A cache hit under a different search budget or model is a correctness bug, not
an optimization.

#### P0-D exit criteria

- Legacy or unversioned Malom fields cannot become trusted labels.
- Swapping perspective twice is an identity and one swap reverses the expected
  W/L semantics.
- Atomic capture, terminal, and counterfactual cases have regression tests.
- Reusing a run database or mismatched snapshot fails preflight.
- Cache tests prove that every result-affecting dimension changes the key.

### P0-E. Gold corpus, differential tests, and deterministic signatures

Build a compact, reviewed corpus of rule and oracle boundary cases. It should
cover at least:

- placement, movement, and flying transitions;
- mill formation and mandatory capture;
- the rule for capturing from mills;
- terminal states;
- repetition/history-sensitive situations supported by the project;
- side-to-move and perspective inversions;
- legal-move enumeration and canonical ordering;
- Malom sector boundaries and corrected value decoding;
- known historical defects and every future escaped defect.

Each case stores:

- a stable case ID;
- canonical state and required history;
- expected legal atomic actions;
- expected child states;
- expected terminal outcome, if any;
- expected theoretical label, if applicable;
- the independently justified source of the expected result.

Where a reference implementation is used, comparison must be field-by-field
and recorded at a pinned source revision. The project rules and independently
tested repository semantics remain authoritative; a third-party engine is a
differential reference, not permission to bypass local analysis.

#### Fixed search signature

Create a small Stockfish-style signature command using:

- a frozen position list;
- one thread;
- fixed node/work budgets rather than wall-clock stopping;
- fixed seed and canonical configuration;
- stable aggregation order.

The signature detects behavioral drift. It is not an Elo estimate.

#### Model canaries

For a frozen model/bundle, preserve expected:

- raw head outputs/logits;
- legal-action mask;
- top-k actions and ordering;
- value/WDL outputs;
- symmetry transformations;
- single-example and batch behavior.

Compare relevant paths such as:

- PyTorch CPU;
- PyTorch CUDA;
- NumPy reference, where available;
- Rust/PyO3 implementation, where used;
- exported/package-loaded model.

FP32 reference paths should normally use exact equality where the same
implementation supports it, or a maximum absolute error no greater than
`1e-6` when backend arithmetic differs. FP16, quantized, or alternative
backends require measured, baseline-frozen thresholds. Thresholds must not be
chosen after observing a candidate.

Golden outputs must never update automatically. A golden change requires an
explicit semantic explanation, review, and a separately visible diff.

#### P0-E exit criteria

- Every corrected historical defect has a permanent case.
- The fixed signature is stable on the supported reference environment.
- Model canaries detect changed weights, feature ordering, masks, head
  semantics, and bundle tampering.
- A golden update cannot occur as a side effect of running tests.

### P0-F. Numerical and error-policy closure

The trainer must fail or quarantine on:

- non-finite total or component loss;
- non-finite gradients;
- non-finite model parameters;
- non-finite labels or probabilities;
- impossible probability normalization;
- impossible Malom deltas or label ranges;
- corruption of critical counters or checkpoint state.

When a failure occurs, preserve a bounded diagnostic artifact containing the
game/batch identity, configuration and checkpoint IDs, tensor metadata,
relevant statistics, and stable error code. Sensitive or enormous raw data
need not be duplicated, but the evidence must be sufficient to reproduce the
failure from its frozen snapshot.

Do not clear a rolling metric and continue as though nothing happened. Do not
replace an unavailable teacher or failed oracle query with a neutral score. Do
not silently disable Sentinel, ValueNet, GapNet, Malom, or another configured
component. The permitted response is explicit failure, explicit quarantine,
or an explicitly configured and separately named degraded mode.

#### P0-F exit criteria

- Fault injection produces stable error codes and preserved diagnostics.
- Non-finite values cannot reach a published candidate.
- Missing dependencies cannot silently alter the experiment definition.
- Console text, manifest state, and process exit status agree.

## 4. P1 — Candidate Delivery, Frozen Evaluation, and Promotion

P1 begins after the P0 run contract is reliable. It prevents a locally saved
weight file or favorable rolling metric from being mistaken for an accepted
model.

### P1-A. Self-describing model bundle

Define a versioned bundle containing at least:

- `bundle.json`;
- model weights;
- feature/input schema;
- head/output schema;
- model canaries;
- optional calibration data;
- applicable license and notice material.

`bundle.json` should declare:

- bundle schema and version;
- immutable bundle/model identity and SHA-256 values;
- architecture name and parameters;
- weight serialization format;
- input feature names, order, dtype, shape, normalization, perspective, and
  history/rules assumptions;
- head names, semantics, shapes, transforms, and legal-action masking;
- compatible rules and feature schema versions;
- producer run/checkpoint identity;
- training-data and asset lineage summary;
- supported runtime/backends and precision modes;
- canary identity and tolerances;
- resource expectations;
- permitted evidence claims.

Provide operations equivalent to:

- `describe`;
- `verify`;
- `export`;
- `compare`.

The loader must reject an unknown mandatory field, unsupported version,
integrity mismatch, incompatible feature schema, or failed canary. It must not
infer semantics merely from tensor shapes.

### P1-B. Frozen local evaluation protocol

Create an immutable `EvaluationSpec` before running games. It includes:

- candidate and baseline bundle identities;
- start-position corpus and hash;
- number of pairs or prevalidated sequential stopping rule;
- pair and game seeds;
- color/role swap policy;
- fixed search/work budget;
- maximum ply and adjudication rules;
- rules/history version;
- evaluation configuration and hash;
- GPU/runtime identity;
- primary statistics and confidence method;
- acceptance, rejection, and inconclusive thresholds;
- permitted exception and quarantine rules.

On one GPU, load and run candidate and baseline serially or use a controlled
process design that cannot create GPU contention. Do not run training and
formal CUDA evaluation concurrently.

Playing-strength comparisons use fixed work—nodes, visits, rollouts, or another
stable work unit—not equal wall-clock cutoffs. Runtime performance is a
separate benchmark with latency, throughput, memory, and energy/resource
conditions reported independently.

Every game writes an immutable JSONL record. Aggregate results must be
recomputable from raw records. A recomputation command should detect duplicate,
missing, malformed, incomplete, and wrong-spec games.

For the first corrected v4 evaluation, default to a fixed-N paired design with
intervals. Introduce SPRT only after simulation demonstrates suitable Type I
and Type II behavior under the project's high-draw, paired setting and after
its boundaries are frozen before candidate results are inspected.

Formal promotion must not use:

- `win_rate > 50%` without an uncertainty contract;
- a rule that treats candidate wins as decisive while ties hide uncertainty;
- the maximum value of a repeatedly inspected rolling training metric;
- a changing opponent or start distribution;
- wall-clock equality as a strength budget.

### P1-C. Explicit candidate lifecycle

Use a small local state machine:

```text
candidate -> validating -> evaluating -> accepted
                                |
                                +-> rejected
                                +-> quarantined
```

An inconclusive result remains inconclusive; it is not rounded into acceptance
or rejection. Validation failure and infrastructure failure are distinct from
strength rejection.

Promotion must be atomic and append evidence rather than overwriting the
candidate. The accepted record contains the bundle hash, evaluation spec hash,
result hash, code/runtime identity, decision rule, and decision time.

### P1-D. Structured observability

Training and evaluation JSONL events should carry stable IDs and include, as
applicable:

- run, checkpoint, candidate, evaluation, pair, and game IDs;
- seed and player/role assignment;
- opponent and difficulty;
- branch/curriculum choice;
- temperature;
- W/D/L and terminal reason;
- total and component losses;
- learning rate and gradient statistics;
- wall, CPU, GPU, and loader-wait timing;
- GPU utilization and memory observations where available;
- Malom query hit, miss, unknown, and error counts;
- label source, perspective, phase, and distribution;
- checkpoint and data lineage.

Dashboards remain operational health tools. They do not replace the immutable
records or frozen evaluation results.

### P1-E. Windows CPU and controlled CUDA gates

The regular Windows/CPU validation path should cover:

- configuration and manifest contracts;
- provenance and trust gates;
- rules and gold-corpus tests;
- Python/Rust parity where applicable;
- checkpoint and bundle inspection;
- deterministic signatures that do not require CUDA.

A controlled local CUDA gate should cover:

- CUDA availability and device identity;
- model forward and backward pass;
- CPU/CUDA canary comparison;
- bounded one-game training smoke;
- exact-resume parity;
- bundle export/reload round trip;
- a fixed performance probe.

CUDA validation should be an explicit local gate rather than a claim that
ordinary CPU CI has tested GPU behavior.

## 5. P2 — Conditional Local Data and Throughput Work

P2 is activated only when the current online loop acquires a meaningful
persistent replay/offline-data path, or profiling proves that input preparation
materially limits end-to-end training throughput.

### P2-A. Staged local data pipeline

If persistent data is introduced, use explicit local stages such as:

```text
immutable source snapshot
        -> validated sample index
        -> bounded Python queue/prefetch
        -> batch collation
        -> single-GPU training
```

Each boundary carries schema version, snapshot identity, sample/game ID, and
perspective/label metadata. Bounded queues provide backpressure and prevent
unbounded RAM growth.

Do not force this pipeline onto the existing online game loop merely to copy a
reference architecture. If games are generated and immediately consumed in a
simple serial path, retain the simpler contract until measurements justify a
stage boundary.

### P2-B. Freshness and consumption ledger

For replay/offline training, record:

- data snapshot ID;
- first and last included game/sample ID;
- creation time and producer identity;
- sampling weights;
- how many optimizer steps consumed the snapshot;
- whether a sample may repeat and under what policy;
- maximum training work permitted per unit of newly generated data.

This prevents accidental infinite reuse of a stale snapshot and permits exact
resume of the sampling sequence. It does not imply a rolling acceptance set:
the fixed gold and evaluation corpora remain immutable.

### P2-C. Loader probe

Before optimizing, provide a non-mutating probe that reports:

- sample/batch shapes and dtypes;
- ranges and non-finite counts;
- perspective and label-source distribution;
- parse, transform, collate, transfer, and wait timings;
- samples/games per second;
- RAM and VRAM high-water observations;
- queue occupancy and data-wait ratio;
- stable snapshot and configuration IDs.

An optional bounded diagnostic NPZ may be written to an explicit disposable
location, but the probe must not mutate training state, advance a production
cursor, or write to the trusted database.

### P2-D. Optimization ladder and trigger

Use an initial trigger such as a sustained `data_wait_ratio > 10%` under the
frozen representative workload, confirmed across repeated measurements. The
exact threshold may be revised before the first profile is inspected, but it
must be frozen for that decision.

Apply optimizations in this order, remeasuring after each step:

1. vectorize Python/PyTorch transformations;
2. eliminate repeated conversion, parsing, and feature recomputation;
3. batch and prefetch with a bounded queue;
4. use pinned memory when measurement shows transfer benefit;
5. use a bounded number of `DataLoader` workers;
6. use persistent workers only when lifecycle and resume tests remain clean;
7. consider mmap or a streaming-friendly local format;
8. materialize expensive stable features when the storage/lineage tradeoff is
   favorable;
9. move only the remaining proven hot path to Rust/PyO3.

This project should first use a bounded Python queue. There is no planned C++
loader. A native path is considered only after end-to-end measurements prove
the bottleneck remains.

### P2-E. Rust/PyO3 boundary, if triggered

A Rust optimization must have:

- a profiled hot-path justification;
- a maintained Python reference implementation;
- the same input/output schema and error semantics;
- deterministic or tolerance-bounded parity tests;
- a Python fallback for diagnosis;
- implementation identity recorded in the run and bundle;
- end-to-end evidence that throughput or latency improved rather than merely a
  faster isolated microbenchmark.

Do not create a second serialized data format solely for the native path.

### P2-F. Single-machine scheduling

CPU threads/processes may prepare data while the GPU trains, subject to bounded
RAM use and deterministic merge order. The single CUDA GPU remains an exclusive
resource for formal stages:

- training;
- model export/load canaries;
- candidate evaluation;
- CUDA performance probes.

These stages run serially. State remains inside the run/evaluation directories;
there is no daemon, network protocol, remote worker, or cross-machine lease.

## 6. P3 — Optional Research and Productization

P3 items are useful possibilities, not default commitments.

### P3-A. WDL calibration

Fit phase-conditioned empirical WDL curves only for interpretation, reporting,
or a separately defined calibration consumer. Record the dataset, split,
phase definition, fitted parameters, uncertainty, and calibration error.

Calibrated WDL is not theoretical Malom WDL and must not cross the data-layer
boundary into a field named or trusted as `theoretical_wdl`.

### P3-B. Sparse, incremental, or quantized inference

Consider these only after a stable bundle and baseline exist. Each proposal
requires:

- a representative profile;
- reference implementation;
- output-error distribution, not only mean error;
- canary thresholds;
- latency, throughput, RSS/VRAM, and bundle-size measurements;
- frozen paired strength regression.

An optimization that passes a tensor-difference threshold but loses playing
strength is not automatically acceptable.

### P3-C. New heads and uncertainty signals

Auxiliary WDL, phase, uncertainty, mistake-risk, or human-policy heads may be
tested one at a time. Each requires a named label source, loss weight, consumer,
ablation, and removal criterion. No new head should quietly change the v4
baseline while being described as infrastructure.

### P3-D. Tuning and later v5 branches

- Use SPSA or another automated tuner only after the evaluation harness is
  stable and its noise behavior is understood.
- PPO remains a separate experiment.
- HumanPolicy, DAgger, proof/safety layers, complex teacher chains, and the
  complete v5 path require their own activation decision and acceptance plan.
- Trap curricula and mixed-opponent proposals remain hypotheses until a frozen
  ablation demonstrates benefit.

## 7. Shared Public Contracts

Implement a small family of versioned contracts instead of unrelated ad hoc
dictionaries:

| Contract | Purpose |
| --- | --- |
| `RunManifest` | Defines what a training run is and what it may claim |
| `CheckpointEnvelope` | Defines resumable trainer state and lineage |
| `DatasetManifest` | Defines immutable data/assets, trust, and allowed use |
| `ModelBundleManifest` | Defines portable model semantics and integrity |
| `EvaluationSpec` | Freezes an evaluation before results exist |
| `EvaluationResult` | Binds raw records, statistics, and decision to the spec |
| `ProbeReport` | Records non-mutating performance/data diagnostics |

The first implementation should use typed Python dataclasses or an equivalent
strict Python model plus canonical JSON and, where useful, JSON Schema. Every
contract has:

- `schema_version`;
- canonical serialization rules;
- SHA-256 identity;
- `describe`/inspection support;
- strict validation and fail-closed unknown-version behavior.

Protobuf is not required initially. It becomes relevant only if an actual
cross-language or long-term binary compatibility need exceeds what canonical
JSON can provide.

## 8. Implementation Slices

Each slice should be independently reviewable and should avoid bundling an
algorithm experiment with infrastructure changes.

### Slice 1 — Configuration and run manifest

- canonical configuration;
- strict validation;
- preflight modes;
- atomic run manifest;
- event ledger;
- regression tests for no-side-effect failure.

### Slice 2 — Checkpoint envelope v2

- complete trainer/RNG state;
- atomic save, bounded previous copies, and integrity verification;
- explicit fresh/weights-only/exact-resume modes;
- inspection and dry-run migration;
- serial exact-resume parity test.

### Slice 3 — Data and label firewall

- asset and dataset manifests;
- typed label sources;
- trusted Malom/SpecialistDB enforcement;
- explicit perspective and atomic-action semantics;
- cache-key contract;
- quarantine path for runtime/user data.

### Slice 4 — Gold corpus, differential checks, and canaries

- rules/oracle fixtures;
- fixed search signature;
- model/bundle canaries;
- CPU/CUDA and Python/Rust parity where applicable;
- controlled golden-update procedure.

### Slice 5 — Bounded training closure

- one-game smoke under the final manifest/checkpoint contracts;
- `1 game + exact resume + 1 game` versus continuous `2 games`;
- fault injection for interruption, corruption, missing assets, and non-finite
  values;
- Agent freeze of technical choices and product authorization of the objective
  and resource envelope.

### Slice 6 — Bundle and local evaluation

- model bundle and inspection commands;
- frozen paired evaluation;
- raw JSONL recomputation;
- candidate lifecycle and atomic promotion;
- Windows CPU and controlled CUDA gates.

### Slice 7 — Conditional performance work

- loader probe and representative baseline;
- only the steps in the P2 optimization ladder justified by the measurements;
- Rust/PyO3 work only if the Python path remains the end-to-end bottleneck.

P1 work may be designed while P0 is under development, but no model should be
formally accepted until P0 provenance and checkpoint semantics are in place.

## 9. Acceptance Test Matrix

### 9.1 Configuration and manifest

- unknown, duplicate, conflicting, non-finite, and out-of-range values fail;
- invalid preflight has no side effects;
- canonical JSON and hashes are stable;
- dirty Git state is visible rather than silently described as clean;
- component disablement is explicit.

### 9.2 Checkpoint and resume

- interruption at every publication stage leaves a valid last checkpoint;
- truncated, tampered, and wrong-hash files fail verification;
- exact resume rejects missing optimizer/RNG/cursor state;
- old checkpoints can enter only through explicit weights-only import;
- resume with a changed config, feature schema, asset, or DB fails or requires
  an explicit, reviewed migration;
- serial continuous and resumed results match within the declared contract.

### 9.3 Provenance and label safety

- unversioned and wrong-version Malom labels remain untrusted;
- HumanDB historical Malom columns remain masked;
- corrected labels cannot be appended to a legacy labeled store;
- experiment database isolation is enforced;
- theoretical, empirical, human, teacher, and model labels cannot be confused;
- perspective and counterfactual-missing behavior are covered.

### 9.4 Rules, oracle, and cache

- all gold-corpus legal actions and children match expectations;
- atomic capture and terminal behavior are stable;
- differential checks report field-level discrepancies;
- cache identity changes with rules, history, perspective, budget, model,
  schema, asset, and configuration;
- fixed signatures detect semantic drift.

### 9.5 Model and bundle

- single/batch and CPU/CUDA outputs pass canaries;
- Python/Rust outputs pass parity when Rust is used;
- modified weights, manifest, schema, or calibration fail integrity checks;
- bundle reload preserves input/head semantics;
- unknown versions and unsupported precision/backends fail clearly.

### 9.6 Evaluation and promotion

- paired colors/roles and seeds are complete;
- duplicate, missing, incomplete, and wrong-spec games are detected;
- aggregate results recompute exactly from immutable JSONL;
- re-running recomputation is idempotent;
- acceptance/rejection/inconclusive outcomes follow the frozen rule;
- infrastructure failure cannot appear as strength rejection or acceptance;
- promotion is atomic and cannot overwrite an accepted bundle.

### 9.7 Performance

- strength benchmarks use fixed work;
- performance benchmarks use frozen hardware/runtime conditions;
- warm-up, synchronization, precision, batch size, and cache state are recorded;
- queue/worker changes are compared end to end;
- any Rust path demonstrates real throughput/latency benefit and preserves
  parity.

## 10. Explicit Non-Goals

The following are deliberately excluded from the current plan:

- fishtest-style distributed workers or server;
- multi-machine or multi-GPU training;
- DDP, leases, heartbeats, remote task queues, or network result databases;
- a C++ or CMake subsystem;
- a C++ data loader;
- Protobuf merely because a reference project uses it;
- making full v5 implementation a prerequisite for corrected v4;
- turning the fixed gold/evaluation corpus into an automatically rolling set;
- automatically updating golden outputs;
- a simple gatekeeper that declares every tie acceptable;
- using equal wall-clock limits as equal playing-strength work;
- implicit schema or checkpoint migration;
- importing runtime/user data directly into a trusted training database;
- using rolling training metrics as formal candidate selection evidence;
- neutral-value fallbacks for failed or unavailable authoritative components;
- assuming this repository's AGPL license automatically authorizes copying from
  every reference repository.

## 11. License and Attribution Boundary

This repository is licensed under AGPL-3.0-or-later. That makes many forms of
open-source combination possible, but it does **not** remove the need to inspect
the exact source license before directly copying code.

For every direct code or asset import, record:

- source repository and revision;
- exact source file or component;
- copyright holder and notices;
- SPDX license identity and version/options;
- third-party or per-file license exceptions;
- modification history;
- required notice, source, patent, and redistribution obligations;
- the compatibility conclusion for this AGPL-3.0-or-later project.

MIT/Expat, BSD-2-Clause, BSD-3-Clause, Apache-2.0, GPLv3-compatible, and AGPLv3
sources can often be combined with an AGPLv3 work when their conditions are
followed, but this is not an automatic blanket decision. Apache-2.0 patent and
notice terms, BSD/MIT notices, GPL version clauses, AGPL network-source duties,
and embedded third-party files must all be preserved as applicable. GPLv3 and
AGPLv3 combination requires careful preservation of each license's scope and
the relationship permitted by AGPL/GPL Section 13; it must not be summarized as
relicensing the original GPL component without analysis.

Do not directly copy material that has:

- no license grant;
- GPLv2-only terms without a separately established compatibility path;
- a special/non-standard license that has not been reviewed;
- ambiguous provenance;
- incompatible third-party restrictions.

Ideas, public interfaces, test strategies, and architectural patterns should
normally be reimplemented in this repository's own code and terminology.
Whenever code is copied rather than independently implemented, include the
required notices and an auditable attribution record. This section is an
engineering compliance rule, not a substitute for legal review when a license
case is unclear.

## 12. Default Decisions Carried by This Plan

Unless the owner explicitly changes them before implementation or experiment
freeze, the defaults are:

- current target: defect-corrected v4-style Generalist;
- v5: optional design/risk library, not a mandatory full build;
- first corrected run: fresh/random initialization;
- Sentinel, ValueNet, GapNet, and PPO: disabled;
- trusted SpecialistDB: isolated `sector-corrected-v1` database;
- host: one Windows machine;
- accelerator: one CUDA GPU;
- distributed training: out of scope;
- C++: out of scope;
- data pipeline: bounded Python queue first, and only when a staged pipeline is
  actually needed;
- native optimization: existing Rust/PyO3 boundary only after profiling;
- historical checkpoints: weights-only;
- checkpoint promotion: training metrics may create a candidate, but frozen
  evaluation creates an accepted artifact;
- P0 plus an Agent-frozen plan and product launch authorization: required
  before a long run;
- P1: required before an accepted/release claim;
- P2/P3: evidence-triggered and optional.

## 13. Definition of Completion

This plan is successfully executed for the corrected v4 path when all of the
following are true:

1. A clean preflight produces a canonical configuration and immutable run
   manifest without touching training state.
2. The chosen assets, corrected label boundary, disabled legacy components,
   output isolation, seed, and checkpoint mode are machine-verifiable.
3. The trainer publishes integrity-checked checkpoints and supports a proven
   exact-resume path for new-format checkpoints.
4. Historical checkpoints can be inspected and imported only as explicitly
   lineage-changing weights-only inputs.
5. Rules, Malom boundaries, cache semantics, and model paths are protected by
   gold, differential, signature, and canary tests.
6. Non-finite values, corrupt state, provenance conflicts, and unavailable
   required components fail closed with preserved diagnostics.
7. A bounded smoke and resume-parity test pass on the intended Windows/CUDA
   environment.
8. Agent-selected technical choices are frozen in an immutable plan and the
   product owner separately authorizes the objective and resource envelope.
9. Saved candidates are packaged with self-describing semantics and evaluated
   against frozen baselines using immutable, recomputable paired evidence.
10. No distributed, C++, or optional v5 complexity has been introduced without
    a separate need and decision.

At that point, the repository has a defensible single-machine v4 training and
acceptance foundation. It can then support measured iteration toward selected
v5 ideas without making the current experiment wait for an unnecessarily large
research platform.
