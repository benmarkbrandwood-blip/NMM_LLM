# Sanmill Safe-Inducement Preprobe — 16 August 2026

## Decision

Status: `A_signal_region_main_experiment_worth_authorizing`

The bounded preprobe found a predeclared signal region at 10,000 and 100,000
nodes.  The frozen selection rule chooses the highest passing budget, so the
recommended budget for a separately authorised main mechanism experiment is
100,000 nodes.

This decision says only that positional-safe action choice can change whether
the exact pinned, fixed-node Sanmill opponent makes a one-step positional WDL
error on this small frozen state pool.  It is not evidence of human-trap
ability, playing strength, product value, refresh causality, promotion,
deployment, publication, or release.  It does not reopen or modify the F0-H0
`stop_condition_triggered`, estimator `B_not_ready_fail_closed`, or conversion
`C_conversion_not_established` decisions.

## Frozen identities

| Item | Identity |
| --- | --- |
| Preprobe and unlaunched-main plan | `ad36a467e87bda62cd57e9e55bf862c3e3d8c9678960b85307f8a5a72964aaec` |
| Plan file SHA-256 | `42125217bde6eb3505f70b0be606d907834ab38f1bf522bbef8db009f8881d63` |
| State pool | `4fe23672a43cb424f1f1d5854ecc7382a423105d5af4234313eccb4fceb73952` |
| State membership | `609211cdf6f5c5e8a7684315c50592ef9af9e64cdaef470dc835700feb3a7f9b` |
| State-pool file SHA-256 | `a0e681c87f322b4ac674411c6feb66e458c50cd1c2e07f9dcf74c7235c0f6a79` |
| Execution source commit | `5cbab128e4de72c2eff25ea71f8c18315eb877aa` |
| Execution source tree | `38c3c470ea79f96a0429ff13f0378b5dc93aea30` |
| Result | `14aa42310611a033bca9e22da829e2f8c8cead5f422a285f273ddd30c6aa7155` |
| Result file SHA-256 | `c456263a6b63bb8fd6407fbbd7f55cda9bbb28d097c7c3d664809ee2a4259324` |

The 2,343,046-byte machine-readable
[manifest](sanmill-safe-inducement-preprobe-manifest-2026-08-16.json)
contains every state/action/budget cell, semantic search report, transition
label, timing, resource counter, identity, and access audit.

## State-pool blindness and oracle boundary

The source population is the previously frozen 6,400-game
research-exploration crossfit sample.  It contributes positions only as
realistic PlayOK-like source states; no human-behaviour conclusion transfers
to this experiment.

The selection algorithm ranked decisions by
`SHA-256(seed NUL session_id NUL logical_ply NUL phase)`, selected 12 states
per phase, and allowed each source game at most once.  It did not read an
estimator prediction or Sanmill outcome, and it did not replace a state after
Malom or engine observation.  The final pool contains 36 states and 540
complete `A_pos` actions:

| Phase | States | `A_pos` actions |
| --- | ---: | ---: |
| Placement | 12 | 180 |
| Movement | 12 | 103 |
| Flying | 12 | 257 |

The minimum, median, and maximum `A_pos` cardinalities are 2, 10, and 51.
The engine reply-state positional tiers are 2 W, 30 D, and 4 L states.  Pool
construction made 1,523 corrected Malom queries in 12.08 seconds.  All labels
are `sector-corrected-v1`, positional-only `A_pos`; none is `A_allow`.

## Runtime and determinism gate

The execution independently verified the exact isolated runtime:

| Field | Value |
| --- | --- |
| Sanmill commit | `a6623f88959f7453594df274fbe1f128af7ff55e` |
| Sanmill tree | `17b9b0fd51ee8dac54c0454a6935978a47d19e0c` |
| Binary SHA-256 | `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619` |
| Binary size | 5,641,216 bytes |
| Licence | `AGPL-3.0-or-later`; file SHA-256 `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0` |
| Rules identity | `3e62cb93a1e0afe4534ce4824d233344816050b547bb8761dd7fe985d8ad399f` |
| Referee profile | `mif-stable-moving-v1` |
| Runtime identity | `705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea` |

The contract fixed one thread, `UseLazySmp=false`, `Shuffling=false`, seed 42,
`MoveTimeMs=0`, and `go logical nodes N` without an explicit depth.  Perfect
DB, trap patches, opening-book search, and random failure fallback were off.
`DrawOnHumanExperience=true` retains Sanmill's phase-aware depth policy, so a
requested node ceiling is not a promise that every placement root consumes
all nodes.

Six source-blind fixtures at four budgets formed 24 determinism cells.  Each
cell produced the exact same `UciLogicalTurnResult.semantic_record` in two
fresh processes run in forward and reverse order, and in two repeated queries
inside one process.  All 96 determinism searches matched.  Timing and raw
protocol text were deliberately excluded from semantic equality.

## Estimands

For state `i`, let `k_i` be the number of its complete `A_pos` actions after
which Sanmill chose a positional-tier-losing reply, and let `m_i` be
`|A_pos(S_i)|`.  The frozen estimands are:

- `b_i = k_i / m_i`, then `b = mean_i(b_i)`;
- `o_i = 1[k_i > 0]`, then `o = mean_i(o_i)`; and
- `o - b = mean_i(o_i - b_i)`.

This first averages uniformly over safe actions inside a state, then gives
each source-game-unique state equal weight.  It does not pool all actions and
therefore does not let high-cardinality flying positions dominate merely by
having more actions.  All 36 states were evaluable at all budgets, no safe
successor was strict-terminal, and there were no abstentions.

Intervals below are frozen-state, nonparametric percentile bootstrap
intervals with 10,000 resamples.  They are engineering uncertainty summaries,
not population intervals.  Zero events receive no prior or pseudo-event.

## Preprobe results

| Nodes | Downgrades / actions | `b` | `o` | `o - b` | Frozen gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1,000 | 7 / 540 | 0.845% `[0.163, 1.668]` | 13.889% `[2.778, 25.000]` | 13.043% `[2.696, 23.773]` | Fail: `b < 1%` |
| 10,000 | 8 / 540 | 1.540% `[0.327, 3.287]` | 16.667% `[5.556, 30.556]` | 15.127% `[5.114, 27.535]` | Pass |
| 100,000 | 6 / 540 | 1.048% `[0.082, 2.682]` | 11.111% `[2.778, 22.222]` | 10.063% `[2.083, 20.316]` | Pass |
| 500,000 | 6 / 540 | 0.568% `[0.082, 1.186]` | 11.111% `[2.778, 22.222]` | 10.543% `[2.564, 21.137]` | Fail: `b < 1%` |

The preregistered working-zone gate required at least 30 evaluable states, at
least two downgrade actions, `1% <= b <= 30%`, and point estimate
`o - b >= 5%`.  The preprobe gate intentionally uses point estimates; the
unlaunched main experiment has the stronger requirement that the 95% lower
bound of `o - b` is at least 5%.  This preprobe therefore does not establish
the main mechanism claim.

The result is not monotone in requested nodes.  Placement search often stops
below the requested ceiling under the retained phase-depth policy.  The data
support choosing a fixed working configuration, not treating node count as a
continuous strength scale.

### Phase concentration

| Nodes | Placement `b / o-b` | Movement `b / o-b` | Flying `b / o-b` |
| ---: | ---: | ---: | ---: |
| 1,000 | 1.474% / 15.192% | 0 / 0 | 1.062% / 23.938% |
| 10,000 | 3.558% / 21.442% | 0 / 0 | 1.062% / 23.938% |
| 100,000 | 2.083% / 6.250% | 0 / 0 | 1.062% / 23.938% |
| 500,000 | 0.641% / 7.692% | 0 / 0 | 1.062% / 23.938% |

All movement cells had zero downgrade events.  Flying produced five events at
every budget, while placement produced 2, 3, 1, and 1.  This concentration is
why the main protocol keeps equal phase allocation and mandatory phase
reporting.  It must not be summarized as a phase-general mechanism.

The separated downgrade counts were:

| Nodes | W to D | W to L | D to L |
| ---: | ---: | ---: | ---: |
| 1,000 | 0 | 1 | 6 |
| 10,000 | 1 | 0 | 7 |
| 100,000 | 0 | 0 | 6 |
| 500,000 | 1 | 0 | 5 |

### Timing and resource use

| Nodes | Search median | Search p90 | Full-cell median | Full-cell p90 |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.292 ms | 0.380 ms | 61.12 ms | 71.16 ms |
| 10,000 | 0.879 ms | 1.322 ms | 61.50 ms | 75.37 ms |
| 100,000 | 5.054 ms | 8.981 ms | 65.99 ms | 76.94 ms |
| 500,000 | 22.422 ms | 42.022 ms | 83.52 ms | 108.51 ms |

Fresh process startup and complete-history replay dominate the lower budgets.
The run completed 2,160 measurement searches plus 96 determinism searches:
2,256 engine single-step queries in 195.57 active seconds.  Response labeling
used 4,320 Malom queries in 16.07 seconds.  It stayed below 100,000 engine
queries and 7,200 active seconds, with at most one evaluator and one Sanmill
process at a time.

## Unlaunched main protocol

The frozen plan contains a separate 360-state protocol, 120 states per phase,
with one state per source game and a new blind hash namespace.  It excludes
the preprobe states, exhausts `A_pos` at the selected 100,000-node setting, and
uses the state as the independent unit.  Its conjunctive mechanism gate is:

- at least 330 evaluable states;
- deterministic search;
- point `o - b >= 5%`; and
- state-bootstrap 95% lower bound `o - b >= 5%`.

Its hard envelope is 360 states, 40,000 engine single-step searches, 250,000
Malom queries, 14,400 active seconds, one evaluator, one Sanmill process, zero
complete games, zero model loads, and zero training updates.  Any limit stops
the run; automatic retry or extension is forbidden.

The main experiment has not been launched and is not authorised by the
preprobe.  It requires a separately frozen exact child state-pool identity and
explicit product-owner authority.  Even a successful main result would remain
an engine-inducement mechanism result, not a human-trap or product claim.

## Verification and historical bridge reconciliation

The current-route focused group passed 28 tests.  The node-calibration group
passed 7 tests.  The mandatory Malom, DB-teacher, and label-provenance group
passed 103 tests and 498 subtests.  Task-scope Ruff and `git diff --check`
passed.

The experiment intentionally uses the exact current training/referee runtime
`a6623f8`, not the historical bridge-smoke-v2 binary at `db65eb3`.  The current
runtime preserves the audited single-threaded strict options, rules identity,
logical-turn protocol, fail-closed behavior, and AGPL licence, while adding the
complete-history strict-referee profile required here.  Its different source,
tree, and binary identities are recorded above rather than described as
byte-identical to the old smoke.

As an extra non-acceptance check, the historical `test_sanmill_uci.py` group
reported 41 passes and 4 fail-closed local integration failures.  Those four
tests resolve the ignored `sanmill_checkout` key to the moving
`D:\Repo\Sanmill` reference checkout, which has changed source inside the old
`db65eb3` pinned bridge scope.  They do not use the exact
`sanmill_training_checkout` exercised here.  No test was weakened or skipped,
and the machine registry was not rewritten to conceal that unrelated historic
route drift.

## Protected access

Official selection, official confirmation, official final-test, and research
confirmation content reads were all zero.  Source pool `2eb04f54` reads and
consumption were zero; its remaining 108 records are unchanged.  The run made
no HumanDB or Malom writes, loaded no policy model, played no complete game,
and performed no training or weight update.
