# Sanmill No-Update Integrated-Route Probe v1

## Status and authority

Status: `implemented_unlaunched_needs_published_preflight`

This document defines a bounded measurement for the fresh Sanmill-refereed
lineage. The content-addressed
[machine-readable plan](sanmill-no-update-integrated-route-probe-v1.json),
production-route controls, strict preflight, runner, and atomic publisher are
implemented. This document is not a readiness record, training run, strength
evaluation, or authority to execute the probe.

The owner accepted the five probe-only ceilings in the
[node-ladder decision brief](sanmill-node-ladder-v1-decision-brief.md) by
requesting implementation. A clean published implementation commit, a passing
readiness audit, and explicit one-run authority remain separate gates. No
timeout or lack of response may choose a default.

## Question being measured

The completed engine-only calibration measured isolated fixed-position search.
This probe asks how much wall time and host resource the complete production
rollout route consumes when it also includes:

- fresh Generalist feature construction and policy inference on the intended
  CUDA device;
- normal depth-5 and occasional depth-12 lookahead simulation;
- read-only HumanDB frequency and corrected Malom queries;
- authoritative Sanmill application, replay, state comparison, rule history,
  and terminal adjudication after both players' complete turns;
- a persistent Sanmill process within each game; and
- natural complete-game length variation under a 120-logical-ply experiment
  cap.

It does not ask which node ceiling is stronger, whether the learner improves,
or when a curriculum should advance. Game outcomes are retained only to
explain game length and termination; they are not a candidate evaluation.

## Production route that must be preserved

The implementation invokes the production `_rollout` route and the
production `SanmillTrainingGame` and `SanmillTrainingOpponent` classes. It
does not copy gameplay, projection, replay, search, feature, reward, or rule
logic into a second probe implementation.

The runtime contract remains the one already verified for this lineage:

- Sanmill commit `a6623f88959f7453594df274fbe1f128af7ff55e`;
- Git tree `17b9b0fd51ee8dac54c0454a6935978a47d19e0c`;
- release binary SHA-256
  `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`;
- `StrictFailurePolicy=true` and
  `StrictRefereeProfile=mif-stable-moving-v1`;
- one search thread, no lazy SMP, no shuffling, and seed 42;
- `go logical nodes N` with no wall-clock or independent depth limit; and
- no database, opening-book, patch, trap, shallow-search, or random fallback
  inside Sanmill search.

One Sanmill process is started, configured, and retained for the full history
of one game, then closed. Reusing one process across unrelated games would no
longer match the production training architecture.

## Frozen learner and data boundary

The probe uses the current fresh-lineage architecture from random weights with
seed 42 and no checkpoint. The model's production sampling route and fixed
temperature `0.90` are retained. Sentinel, ValueNet, GapNet, S1A warm-start,
S1B refresher, imitation mixing, trainer-side opening forcing, recovery,
retries, and branch rollouts remain disabled.

No optimiser may be constructed. All model parameters are marked as not
requiring gradients, and the canonical model-state digest is recorded before
and after the complete probe. The frozen-target copy receives its own before
and after digest. Any byte or tensor difference is a fatal result, not a
measurement warning.

The probe must exercise data reads without permitting training side effects:

- HumanDB is captured through SQLite's supported online-backup mechanism,
  converted to a closed sidecar-free snapshot, and opened with
  `HumanDB(..., read_only=True)`; the active database's WAL or SHM is never
  deleted or moved;
- a dedicated empty `sector-corrected-v1` SpecialistDB with no lineage binding
  is created before plan publication, closed, and reopened with
  `SpecialistDB(..., read_only=True)` so empty-database feature lookups remain
  on the route;
- corrected Malom data is opened through its normal read-only query adapter;
  and
- all source identities, schemas, row counts, `quick_check` results, main-file
  hashes, and any sidecar inventory are frozen before launch and compared
  again afterward.

The implemented plan reuses the already closed, sidecar-free online-backup
HumanDB snapshot indexed by `human_db_route_probe_snapshot_path`; its SHA-256
is `97be7152573815180df6950b6150c667b1e5c2c8b1b21748b3ed9cf020b6f93c`.
It binds a new probe-only empty SpecialistDB indexed by
`specialist_db_route_probe_snapshot_path`, with SHA-256
`b4d522d23720ab86013bbadd6b3414fba1e205ab988f3949c6ed417dca486b7f`.
The HumanDB adapter uses SQLite `immutable=1` for this closed snapshot so even
a read does not create WAL or SHM sidecars. Neither registry key redirects the
active training databases.

The production rollout currently persists completed evidence whenever a
SpecialistDB object is present. Implementation therefore requires an explicit
probe-only `persist_rollout_evidence=False` control whose default remains
`True`. It may suppress only the final `record_game` and Malom-label writes;
it must not bypass SpecialistDB feature reads, Malom reward queries, rollout
construction, authoritative replay, or termination handling.

## Proposed bounded work matrix

The candidate node ceilings are:

```text
1,000; 5,000; 25,000; 100,000; 500,000
```

For each ceiling, run six complete Sanmill-opponent games:

- four normal depth-5 games: two fixed schedule seeds with learner White and
  learner Black for each seed; and
- two forced depth-12 games: one fixed schedule seed with learner White and
  learner Black.

Add six Sanmill-refereed frozen-target control games with no search opponent:
four normal depth-5 games and two forced depth-12 games under the same color
balance. This produces 36 games: 30 search-opponent games and six controls.

The deep route is deliberately oversampled. Production training currently
selects it once per 20 scheduled games, but measuring one natural occurrence
would not separate its cost from one game's length. Reports must therefore
keep normal and deep strata separate. Any projection to a retained run may
apply the intended 19:1 weighting only after showing the raw strata.

With `max_ply=120`, the hard workload bounds are:

| Bound | Value |
| --- | ---: |
| Complete games | 36 |
| Maximum logical plies | 4,320 |
| Search-opponent games | 30 |
| Maximum search calls | 1,800 |
| Maximum requested search-node ceilings | 227,160,000 |
| Batch size | 1 |

The exact schedule order, game identities, learner colors, Torch generators,
and normal/deep flags must be stored in an immutable machine-readable plan.
They must not be generated after partial results are visible. The 120-ply cap
is an experiment truncation, not a rules draw; Sanmill rules terminals remain
separate termination reasons.

## Measurements

The raw report must preserve one record per game and enough per-turn samples
to recompute every aggregate. At minimum it records:

- total game wall time and logical plies;
- phase counts, learner color, normal/deep route, outcome, rule-terminal reason,
  and max-ply truncation separately;
- Sanmill process startup/configuration time;
- policy feature/lookahead plus inference time on learner turns;
- HumanDB, SpecialistDB, and Malom query totals where they can be observed
  without changing query semantics;
- authoritative apply/replay/state-comparison time for learner, frozen-target,
  and search-opponent turns;
- requested and actual opponent nodes, search calls, completed-depth
  distribution, compound-turn count, and search time;
- CPU process time, peak resident memory, CUDA allocated/reserved peak memory,
  device identity, and the active Windows power scheme; and
- model and database before/after identities.

Each node ceiling, route depth, learner color, and opponent kind is reported
as a separate stratum. A single pooled games-per-hour number is prohibited.
Median, nearest-rank P90, range, and raw sample count must accompany any timing
summary. A 19:1 weighted planning estimate may be added, but it cannot replace
the raw normal/deep results.

Instrumentation may add monotonic timers and no-op-by-default observation
hooks around production operations. Tests must prove that enabling the hooks
does not change selected actions, authoritative states, rewards, termination,
or semantic game identities.

## Fail-closed invariants

The run aborts without publishing a completed report if any of the following
occurs:

- source, plan, runtime, rules, referee, model, or data identity drift;
- an optimiser, backward pass, checkpoint save, training update log, or
  writable database connection is created;
- model or frozen-target state changes;
- any database file or sidecar changes;
- Sanmill protocol failure, timeout, illegal action, fallback, or history/state
  mismatch;
- a non-finite feature, probability, reward, timer, or resource value;
- a scheduled game is missing, duplicated, or run with the wrong color,
  ceiling, route depth, or seed; or
- the output path already exists.

The result is written atomically under ignored `out/diagnostics/`. Failure
evidence may be written to a different quarantine path, but it must never be
relabeled as a completed result.

## Implementation and verification gates

Implementation is split into independently reviewable commits:

1. the default-on rollout-persistence control, timing hooks, and
   semantic-parity tests; and
2. the dedicated no-update runner, immutable plan schema, publisher, and
   focused tests.

A third evidence commit may publish a readiness record only after the exact
command passes from a clean implementation commit already present on
`origin/dev`. The implemented read-only command is:

```powershell
.\.venv\Scripts\python.exe scripts\probe_sanmill_integrated_route.py `
  --preflight `
  --plan docs\experiments\sanmill-no-update-integrated-route-probe-v1.json `
  --paths-config data\training_paths.local.json
```

It includes one unscheduled two-ply frozen-target route check and consumes none
of the 36 planned games.

Focused tests must at least prove:

- the probe calls the production rollout/referee/opponent route;
- an attempted optimiser construction, `backward`, `step`, checkpoint write,
  or SQLite write fails the test;
- the read-only SpecialistDB remains visible to encoder queries while rollout
  persistence is suppressed;
- normal and forced-deep games restore lookahead depth correctly even after an
  exception;
- the schedule and hard work totals are exact;
- semantic results match an equivalent production rollout with observation
  hooks disabled; and
- atomic publication refuses overwrite and incomplete schedules.

The relevant trainer, Sanmill-referee, Malom/provenance, lint, and
`git diff --check` groups must pass. The readiness audit must additionally
verify the intended CUDA device and perform a short no-search route check; it
must not consume any of the 36 planned games.

No MIF or Sanmill repository change is currently required. The pinned Sanmill
CLI already exposes the strict logical-turn search and authoritative state
needed by this probe.

## Interpretation and next decision

A completed probe can support a no-update route games-per-hour envelope,
GPU/CPU balance, memory headroom, a provisional segment duration, and whether
all five ceilings are operationally distinct enough to retain. Because the
probe deliberately excludes optimiser, database-write, checkpoint, and
training-log cost, it is not a final full-training throughput measurement. It
cannot support a strength ranking, model promotion, an advancement threshold,
or a training launch.

After the evidence is reviewed, a separate decision must freeze or reject:

- the retained node levels;
- a deterministic hard-stage or adjacent-level blended exposure schedule;
- the frozen-target/Sanmill-opponent ratio;
- game, wall-time, segment, checkpoint, and max-ply limits; and
- whether a new update-enabled training smoke is required before a long run.

The current readiness verdict is `not_ready`: implementation exists, but its
commit is not yet the clean published source of a passing preflight. Probe and
training launch authority are both absent.
