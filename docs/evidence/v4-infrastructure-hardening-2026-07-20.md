# v4 Infrastructure Hardening Evidence — 20 July 2026

## Outcome and Scope

The infrastructure plan for the corrected v4-style Generalist path is
implemented through Slice 6. Slice 7 was measured and stopped at its declared
condition: the current trainer has an online serial rollout path, not a
persistent replay/offline loader, so loader optimization was not activated.
No C++, distributed training, multi-GPU scheduling, or new native data format
was introduced.

This evidence establishes reproducibility, recovery, provenance, bounded
determinism, portable bundles, frozen local evaluation, and evidence-backed
candidate promotion. It does not establish playing strength, approve a long
training run, or claim completion of v5.

The project and exported bundles use `AGPL-3.0-or-later`. Adapted ideas from
reference projects were reimplemented as local contracts; no reference
repository became a runtime dependency.

## Implemented Contracts

### Run and recovery

- canonical resolved run configuration and immutable run manifest;
- read-only preflight before output/database mutation;
- append-only hashed lifecycle events;
- checkpoint envelope v2 with complete trainer, optimizer, data, and RNG
  state;
- explicit `fresh`, `weights-only`, and `exact-resume` modes;
- atomic checkpoint publication with bounded verified previous copies;
- corruption, truncation, incompatible lineage, and non-finite state rejection;
- read-only checkpoint describe/verify/compare and default-dry-run legacy
  migration with a required model canary;
- deterministic game identities, independently derived RNG substreams, and
  bounded run segments.

### Data and correctness

- typed label and cache-key contracts;
- component-level dataset inventories;
- a frozen corrected Malom inventory covering 512 files and 83,582,223,577
  bytes;
- routine structural Malom verification against manifest identity
  `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747`;
- SpecialistDB lineage isolation and exact checkpoint binding;
- read-only web access to SpecialistDB and a hash-chained quarantine ledger for
  runtime game evidence;
- corrected v4 rules corpus signature
  `b2acd29e816ef89b52d70dde61cbe12a9a94ca0dd34ce03019c1768709e90437`;
- deterministic model output canaries on CPU and CUDA.

### Delivery and evaluation

- immutable self-describing model bundles with explicit ordered 134-element
  move input and 80-element value input schemas, policy/value head semantics,
  weight and license hashes, lineage, runtime compatibility, and canaries;
- `export`, `describe`, `verify`, and `compare` bundle operations;
- an initial frozen three-position corrected-v4 evaluation corpus;
- fixed-N paired evaluation with role/color swap, fixed work, frozen seeds,
  maximum ply, adjudication, confidence method, and decision thresholds;
- append-only hash-chained game JSONL and independent result recomputation that
  rejects duplicate, missing, malformed, incomplete, or wrong-spec games;
- candidate lifecycle states with distinct validation, evaluation,
  inconclusive, rejection, quarantine, and atomic acceptance outcomes;
- immutable acceptance records binding bundle, spec, raw-record, and result
  identities.

## Bounded CUDA Evidence

All generated evidence below remains in ignored local smoke directories. It
is retained for diagnosis but is not committed as a model or training-data
artifact.

### Exact-resume parity

The final comparison used seed 42, one CUDA GPU, `batch_games=1`, maximum 40
plies, simulation depth 2, minimal rollouts, no branches, and Sentinel,
ValueNet, and GapNet disabled.

- continuous run: `hardening-continuous-20260720-c`;
- first segment: `hardening-segment-1-20260720-c`;
- exact-resume segment: `hardening-segment-2-20260720-c`;
- final lifecycle status for every segment: `completed`;
- parity result: exact equality for model, optimizer, scheduler, scaler, RNG,
  normalized trainer state, normalized data state, two combined training-log
  records, and the semantic rows of all four SpecialistDB tables.

The audit exposed and fixed two real defects before passing: checkpoint
`map_location=cuda` moved CPU RNG bytes to CUDA, and the trainer did not seed
Python's global `random` generator. Earlier failed/superseded smoke outputs are
preserved under their original ignored names rather than relabeled as passing.

### Bundle and evaluation

- bundle identity:
  `8af76ace72820cdc477f3ccb957cb89bf06ac29f3adcf191356267e658f1248f`;
- model identity:
  `93537d0a68f450ea5cbb7cd29e330a2b2253e00330ffee9cff49aa787eba025d`;
- CUDA bundle canary maximum absolute differences: policy approximately
  `3.55e-14`, value approximately `2.50e-16`;
- paired smoke spec identity:
  `e08fbbd8ed56aa4b62bf23a2b619e704af01b27e877c186b1de60e3b945fc1b4`;
- raw-record hash:
  `94bf24a769e273e111f9cfaa8f94482a8555449358d7d53b4339e3e37aa885d3`;
- recomputed result identity:
  `70c5a9b160d3204bcb0514a47fd5a50af1f3b177255e7cb50266ccfdfdc29e33`;
- result: two short self-comparison games, both draws, correctly classified as
  `inconclusive` rather than accepted.

### Performance trigger

The fixed probe used 24 legal-action rows and 100 policy/value forwards:

| Device | Mean latency | Forwards/s | Peak CUDA memory |
| --- | ---: | ---: | ---: |
| CPU | approximately 0.107 ms | approximately 9,313 | not applicable |
| RTX 4090 CUDA | approximately 0.411 ms | approximately 2,435 | 10,138,624 bytes |

The tiny fixed workload is faster on CPU because CUDA launch overhead
dominates. This is not a training-throughput claim. The loader-wait ratio is
not applicable because no persistent loader exists; therefore the frozen 10%
loader optimization trigger is not active. A bounded Python queue, native
loader, or Rust/PyO3 hot path must not be introduced from this evidence.

## Verification Results

The final focused Windows validation reported:

- `223 passed, 498 subtests passed` across run contracts, preflight, checkpoint
  and migration, resume, data/provenance, Malom, gold corpus, canaries, bundle,
  paired evaluation, candidate lifecycle, and performance-probe tests;
- the repository-required Malom/DB/provenance group is included in that result;
- real CUDA training, exact resume, bundle reload, paired evaluation, and
  performance paths completed successfully.

Full-suite collection is not a clean baseline. It collected 832 tests and
reproduced four pre-existing stale internal-interface errors:

- missing `learned_ai.models.action_encoder`;
- missing `learned_ai.models.state_encoder`;
- missing `DEFAULT_BACKWARD_DECAY` export;
- missing `SentinelOutput` export.

These errors are outside this hardening scope and must not be reported as
regressions fixed by this work. A repository-wide `compileall` attempt also
could not replace several existing `__pycache__` files because other processes
held them; all new modules were nevertheless imported and executed by focused
pytest and real CLI/CUDA paths.

## Remaining Owner Decisions Before a Long Run

Infrastructure is ready for another read-only readiness audit, but the long
run remains blocked by experiment decisions rather than missing mechanisms.
The owner must freeze, before viewing long-run results:

- update algorithm;
- opponent schedule;
- temperature start;
- total game budget;
- seed;
- `batch_games` (the deterministic supported baseline is currently 1);
- checkpoint cadence;
- monitoring cadence;
- stop and quarantine criteria;
- formal evaluation baseline bundle, pair count, start corpus, margins, and
  maximum ply.

After those decisions are recorded, rerun the training-readiness skill against
the exact clean commit, dedicated output directory, isolated empty
`sector-corrected-v1` SpecialistDB, and exact launch command. Do not launch a
long run from the smoke bundle, smoke databases, or any superseded smoke
segment.
