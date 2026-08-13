# Sanmill retained-v3 versus no-refresh-v4 passivity diagnostic v1

Status: `frozen_awaiting_product_authorization`

This is a no-update development diagnosis. It is not held-out evidence, a
playing-strength test, a refresh-cadence causal experiment, or authority to
play any game. The machine-readable
[plan](sanmill-retained-v3-v4-passivity-diagnostic-v1.json) is frozen at
identity
`035c68f80b94dddb8d139d56c38c86c4fde29fa13de5e19db1f4e1fe484c318e`,
file SHA-256
`e4394d015490d1e337554589c339db19a20ae45f2968bb9cbceee2ba207cd5b3`,
and evaluator implementation commit `361d99a43a9ca549b6f4594d8cb5c26a23d5dd54`.
A separate product-owner grant must bind that exact plan identity, file and
resource envelope before a development-corpus game may start.

## Observed facts / 观察事实

The two frozen candidates are the final `latest.pt` checkpoints from the
completed retained runs:

| Candidate | Training seed | Final checkpoint file SHA-256 | Payload SHA-256 | Final target age |
| --- | ---: | --- | --- | ---: |
| retained-v3, refresh every 50 games | 58 | `28e8af274f4fc9dd7e00ce4f7be884c855354218c796888f1c1ab81a4cdc9fa7` | `1c14955e2c7ca69824c5369f7501713788c4d8650f63dd4d6cf4992fab037ac8` | 50 |
| retained-v4, no refresh | 70 | `295b268e697255908f9c7517f4697ca251a10ec0f13d922cbcbab2260fb5105d` | `ed7932bc7c11b1aa41274ea0de7bd08902812b1188ca4739b6d0d8dc15e46727` | 5,000 |

The exact CPU-verifiable training-route bundles have been exported into the
ignored diagnostic namespace. Their identities are:

- retained-v3:
  `b6d7ecf62ea9aeba893eff51e794d9307c444f361f54c9e1e832ac5b5d7bc5a0`;
- retained-v4:
  `817d2e36fbd0b614c5c48737ee987f684b99eb6ff697591618123ec7307a2d0f`.

Each route uses its lineage-owned frozen target and read-only SpecialistDB.
HumanDB and corrected Malom are common. Sentinel, ValueNet and GapNet remain
disabled. The retained-v3 SpecialistDB identity is `82d7fbcd...`, and the
retained-v4 identity is `3d69d1ac...`; replacing either with a common database
would no longer evaluate the frozen training route.

The first full source preflight at published commit `bebad52` failed closed
under readiness identity `1ad48701...`: both the separate authorization and
the v3 immutable data route were absent/invalid. The latter failure was caused
by the preserved empty WAL plus 32,768-byte SHM beside the original v3 main
file. The main file remained byte-identical. The plan now binds the already
documented sidecar-free v3 audit snapshot and a newly created sidecar-free v4
main-file snapshot in the ignored diagnostic namespace. Both snapshots are
byte-identical to their lineage-owned main files, report `quick_check=ok`, and
remain free of WAL, SHM and journal after strict route loading. No sidecar was
deleted and no training database was changed.

In sequential 500,000-node training rows, the Sanmill arm reached the
120-logical-ply limit in 43.9% of retained-v3 games and 57.6% of retained-v4
games. Those observations differ in seed, source, model age and trajectory.
They motivate this diagnosis but do not identify a refresh effect.

The existing 64-record, twelve-logical-ply opening corpus has already been
inspected and used. Reusing it is therefore explicitly labelled development,
not held out. Its executable identity is
`417d74ebe01734c43e48531cab81ba742bc89e455f1c834ea7e31006b886f8b9`.

## Hypotheses / 假设

1. The retained-v4 final route has a higher probability than retained-v3 of
   remaining rules-ongoing after total logical ply 120 against the same frozen
   500,000-node Sanmill opponent.
2. If that difference exists, it may be associated with longer rules-terminal
   games, different no-capture/repetition state at ply 120, or a different
   rate of Malom-queryable value downgrades by the candidate.
3. Eventual W/D/L may remain almost all draws. It is therefore explanatory
   secondary evidence and cannot replace the process estimand.

## Supporting evidence / 支持证据

- The training-stage 120-ply truncation shares differ by 13.7 percentage
  points, while the final-stage scores are almost identical.
- The earlier 128-game retained-v2 strict evaluation needed only 483.7 active
  seconds and showed that the pinned 500,000-node route, full history,
  colour-swap schedule and per-turn node evidence are operational.
- Malom can classify a settled board and exact complete move from the current
  mover's theoretical perspective. It can therefore supply a separately
  labelled process diagnostic without adjudicating repetition or no-progress
  history.

## Counterevidence / 反证

- The candidates have different training seeds, plan source commits, runtime
  repair histories, frozen-target ages and accumulated SpecialistDBs. This
  study can describe their final-route difference only; it cannot attribute
  that difference to target refresh.
- Deterministic argmax development games may have a different length and
  decisive-rate distribution from sampled training games.
- Malom has no threefold-repetition or no-capture history. Its W/D/L at ply 120
  is theoretical board-state context, never a strict game result.
- The fixed-corpus engineering interval describes variation across these 128
  matched start/colour game units. It is not a population confidence interval.

## Next validation experiment / 下一步验证实验

### Frozen design

For each of 64 source records and each candidate colour, run retained-v3 and
retained-v4 in adjacent order against a fresh strict Sanmill process. Replay
the same complete twelve-ply history for every matched unit. This gives:

- 64 starts;
- two candidate colours per start;
- two frozen candidates per start/colour unit;
- 128 matched units and 256 games total; and
- no repeated deterministic game.

Candidate inference is deterministic CPU float32 policy argmax over its exact
`s-gen-v2-training-aligned-v1` route. Sanmill uses one thread, MTD(f), IDS,
shuffling off, seed 42, no wall-clock move limit and at most 500,000 nodes per
complete logical turn. Sanmill is the sole complete-history referee.

### Primary metric and decision rule

The primary per-game value is `ongoing_after_total_logical_ply_120`:

- `1` only when the strict referee is non-terminal immediately after total
  logical ply 120; and
- `0` when a rules terminal occurs on or before logical ply 120.

It is a horizon-survival indicator, not a draw. For each matched start/colour
unit, compute retained-v4 minus retained-v3. Report the mean and the existing
two-sided 95% normal engineering interval (`z=1.96`).

- lower bound above zero: `v4_higher_120_ply_survival`;
- upper bound below zero: `v3_higher_120_ply_survival`;
- otherwise: `inconclusive`.

The interval half-width must also be reported against a preregistered maximum
of 0.10. Failing that precision check does not change observed counts, but the
directional decision remains `inconclusive_precision`.

### Secondary process metrics

For every game, retain support counts and report:

- total and post-prefix logical plies to rules terminal or safety cap;
- termination reason, with rules draws separated from the safety cap;
- candidate move counts whose exact Malom downgrade is queryable, preserving,
  one-step downgraded or two-step downgraded;
- at total logical ply 120, if still ongoing: local FEN, strict referee state,
  history SHA-256, no-capture count, repetition counters, and Malom theoretical
  W/D/L from both side-to-move and candidate perspectives;
- matched restricted-length difference, normalized by the 1,536-post-prefix
  safety ceiling; and
- eventual rules-terminal W/D/L as descriptive secondary evidence only.

The web report must display each metric with its denominator and a short help
explanation. In particular it must state that ply-120 survival is not a draw,
Malom is history-free, preserving rate is conditional on query coverage, and
eventual W/D/L has no strength or promotion claim in this reused corpus.

### Resource and failure boundary

- at most 256 games;
- at most two active evaluator hours;
- one evaluator process and one strict Sanmill process at a time;
- at most 1,536 post-prefix logical plies per game;
- at most 196,608 Sanmill search turns and a summed per-turn node ceiling of
  98,304,000,000 nodes across the full theoretical safety envelope;
- no model update, database write, checkpoint write, corpus resampling,
  result-based early stop, retry, extension, promotion, publication or release.

Reaching the 1,536-post-prefix safety ceiling is recorded as
`safety_cap_incomplete`, never as a draw. It may contribute the already-known
ply-120 survival value and censored length, but not eventual W/D/L. Any illegal
action, state/history divergence, missing required database, non-finite model
output, Sanmill failure, ledger integrity error or identity drift stops the
run fail closed.

An explicitly authorized same-spec resume may continue only the missing suffix
after validating the immutable spec and hash-chained completed-game prefix. It
may not replay a completed game. There is no automatic retry or recovery.

### Claim boundary

Completion may support only a statement about the process difference between
these two named final routes on this reused fixed development corpus. A causal
refresh claim still requires same-source, same-seed, equal-transition
refresh/no-refresh pairs across multiple seeds. A playing-strength claim still
requires a separately powered, newly exposure-audited held-out contract.

## Current launch gates

| Gate | Current state |
| --- | --- |
| Objective and claim boundary | frozen |
| Candidate checkpoint and route identities | verified |
| Primary metric and precision rule | frozen |
| Corpus and schedule | frozen as non-held-out development |
| Resource and failure envelope | frozen |
| Prospective process evaluator | implemented and unit tested |
| Live web report and metric help | implemented and JavaScript-checked |
| Focused and mandatory provenance tests | 55 focused passed; 103 tests / 498 subtests provenance passed |
| Clean published implementation source | implementation `361d99a` awaits ordinary publication with this amended plan |
| Machine-readable plan identity | `035c68f8...` |
| Separate product authorization | absent |

Verdict: `needs_decision` until the frozen plan is itself published, a final
read-only preflight passes, and an exact plan-bound product grant exists.
