# Sanmill Node-Throughput Calibration v1

## Status and authority

Status: `designed_unlaunched`

Plan identity:
`2dff4e6d37f36af90d9e90943dad8f4bcccbec802615f7eedc1103af32a51290`.
Raw plan SHA-256:
`2fc47ac72737c30e507eaee47554a70ee7c745ba247d871b57f506a6494e74ca`.

This document and its
[machine-readable plan](sanmill-node-throughput-calibration-v1.json) define a
bounded, engine-only calibration for the fresh Sanmill-refereed training
lineage. They do not authorise the calibration, another training smoke, or a
long run. No result exists yet.

The calibration is intentionally separate from the completed
[smoke-002 result](../evidence/sanmill-refereed-fresh-v1-smoke-002-result-2026-08-08.md).
That two-game smoke proved integration and evidence-chain behaviour, but its
single 1,000-node Sanmill game cannot establish a retained-run resource
envelope.

## Question being measured

The calibration asks how the pinned, strict, single-threaded Sanmill process
converts a requested logical-turn node ceiling into:

- actual primary and compulsory-removal nodes;
- completed and effective search depth;
- search calls and selected complete logical turn;
- median and tail search latency; and
- cold-process and persistent-process throughput on this Windows host.

It does not ask whether a node ceiling is strong enough. It does not load a
candidate model, run the trainer, update an optimiser, access a checkpoint or
training database, adjudicate candidate-versus-baseline games, or select a
curriculum automatically.

## Pinned runtime

The plan uses the same isolated runtime as smoke-002:

- Sanmill commit `a6623f88959f7453594df274fbe1f128af7ff55e`;
- tree `17b9b0fd51ee8dac54c0454a6935978a47d19e0c`;
- release binary SHA-256
  `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`;
- `StrictFailurePolicy=true`;
- `StrictRefereeProfile=mif-stable-moving-v1`;
- one thread, lazy SMP and shuffling disabled, seed 42;
- no wall-clock search limit; and
- Perfect DB, Opening Book, HumanDB move substitution, patches, traps, shallow
  fallback, and random fallback disabled.

The ignored local path is resolved only through
`data/training_paths.local.json`; it is not embedded in the plan or evidence.
Execution fails closed if the source checkout, tree, tracked worktree, binary,
licence, rules identity, referee identity, options, or any root state changes.

## Fixed roots

Eight stable complete-turn boundaries are replayed from one previously
recorded legal game. The plan stores the full action history once and freezes
the exact prefix length, FEN, history SHA-256, logical counts, rule counters,
legal-action-set SHA-256, and state identity for each root.

| Root | Role | Important distinction |
| --- | --- | --- |
| `placement-empty` | placement | empty board and opening-depth behaviour |
| `placement-mid` | placement | thirteen legal placements |
| `placement-last` | placement | final placement before movement |
| `moving-initial` | movement | first origin-counted repetition observation |
| `moving-mid` | movement | full material and eleven no-capture plies |
| `moving-reduced` | movement | post-capture reset of rule-history windows |
| `flying-black` | flying | three-piece side with 39 legal actions |
| `compound-capable` | movement/removal | root historically selecting a Mill plus compulsory removal |

The last root is not forced to return a compound turn at every budget. The
report records whether it does. This avoids changing the fixed-work question
by adding an artificial depth limit or requiring a particular best move.

## Work matrix

The immutable survey grid is:

- node ceilings: `1,000`, `5,000`, `25,000`, `100,000`, and `500,000`;
- nine repetitions per mode/root/ceiling cell;
- no independent depth ceiling;
- eight roots; and
- two modes.

The maximum requested work is 90,864,000 node ceilings over 720 timed
searches and at most 405 Sanmill process launches. Actual nodes may be lower
when a position completes a depth before exhausting the ceiling or when
`DrawOnHumanExperience` selects the opening-depth path. The report must retain
both requested and actual nodes; requested nodes alone are not a throughput
measurement.

`cold_process` starts a new process, applies strict options, clears the hash,
replays one root, and searches once. It records process startup, replay/setup,
and search time separately.

`warm_sequence` starts one process for a given repetition and node ceiling,
clears the hash once, then visits all roots in chronological order without
clearing the hash between them. Each repetition still starts a fresh process.
This is a controlled proxy for the persistent Sanmill process used within one
training game; it is not an end-to-end training benchmark.

Budget order rotates between repetitions to reduce a simple thermal or
background-load bias. Root order remains chronological in the warm mode.
Every cell must produce one semantic result identity across its nine fresh
repetitions. Search must leave the authoritative root state unchanged.

## Statistics and evidence

The report preserves every raw sample. Each mode/root/ceiling cell records:

- minimum, median, nearest-rank p90, maximum, and median absolute deviation for
  search seconds, actual nodes, node utilisation, and nodes per second;
- the complete semantic logical-turn response without timing text;
- primary, removal, and total nodes, search calls, depth, score, move, terminal
  outcome, and whether a compulsory removal was selected; and
- process-start and position-replay/setup times.

It also records the exact clean, published NMM_LLM commit, plan identity and
raw SHA-256, pinned Sanmill installation record, host/CPU description, and the
active Windows power scheme before and after the run. A power-scheme change,
semantic nondeterminism, state mutation, illegal root, timeout, protocol
error, or identity drift aborts without publishing a result. Publication is
atomic and refuses to overwrite an existing file.

CPU frequency, thermal state, and unrelated host load cannot be made identical
by this tool. Median, p90, and dispersion are therefore retained instead of
presenting one timing as exact. A materially noisy result should be repeated
under a newly authorised run ID rather than edited.

## Interpretation gate

The result can support a later proposal for a small node ladder, per-level
wall-time estimates, search timeouts, and segment sizing. It cannot establish:

- playing strength or an advancement threshold;
- candidate quality;
- learner/frozen/Sanmill opponent proportions;
- complete games per hour;
- policy-feature, GPU, optimiser, checkpoint, logging, or SQLite cost; or
- authority to start training.

After reviewing utilisation, latency tails, depth, compulsory-removal costs,
and cold/warm effects, a separate no-update integrated route probe may be
proposed to measure the remaining Python policy/referee/logging overhead. The
product owner must explicitly approve both any node-ladder decision and that
probe. There is no timeout or automatic default decision.

## Commands

After this design is committed and published, the read-only preflight is:

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_sanmill_nodes.py `
  --preflight `
  --plan docs\experiments\sanmill-node-throughput-calibration-v1.json `
  --paths-config data\training_paths.local.json
```

Preflight validates the clean published source, runtime, and all eight replay
fixtures. It performs no timed search and reports
`launch_authorized=false`.

Only after a separate authorization may `--launch calibration`, a new run ID,
and an absent ignored output path replace `--preflight`. The output should be
placed under `out/diagnostics/`; it must not be committed until its claim
boundary, hashes, and result interpretation have been reviewed.

## Design verification

The focused plan, runner, publisher, pinned-runtime fixture, and training
referee group reported 13 passed. The actual isolated training runtime
replayed all eight roots under the expected current rule and referee
identities. Ruff and `git diff --check` pass.

An intentionally broader run including the historical strict-v2 bridge group
reported 52 passed and four fail-closed checks. All four stop at the already
documented protected-source-path gate for the moving historical
`sanmill_checkout`; none reaches the isolated `sanmill_training_checkout` used
by this calibration. No test was skipped, weakened, or reclassified to obtain
the focused green result.
