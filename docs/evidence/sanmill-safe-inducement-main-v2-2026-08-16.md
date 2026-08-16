# Sanmill Safe-Inducement Main Experiment v2 — 16 August 2026

## Decision

Status: `mechanism_gate_passed`

At the frozen primary budget of 100,000 nodes, all 360 states were evaluable.
The state-uniform oracle increment `o - b` was 14.678%, with a frozen-state
bootstrap 95% interval of [11.363%, 18.178%].  The point estimate and lower
bound both exceed the unchanged 5% thresholds, the minimum of 330 evaluable
states is met, and the runtime determinism gate passed.

This establishes only that directed selection among complete positional-safe
`A_pos` actions can change the probability that the exact pinned Sanmill
runtime makes a one-step positional WDL downgrade.  It is not human-trap
ability, playing strength, product value, refresh causality, equivalence,
promotion, deployment, publication, release, or training authority.  The
F0-H0 `stop_condition_triggered`, estimator `B_not_ready_fail_closed`, and
conversion `C_conversion_not_established` decisions remain unchanged.

## Identities

| Item | Identity or SHA-256 |
| --- | --- |
| Protocol v2 | `22f79951080afe01f11e0d9f2bbd16e8421fd92f2e96841da3627405271b89bf` |
| Protocol v2 file | `887b637010240ed9aae99732692a5bd1c9c127e5757479c85e9224f7ee41c134` |
| Main state pool | `261bd980acda6018f6c7fe4c268b1e00f190e33ae5c055ceb466d43c77dcc53c` |
| Main membership | `69246e3ea5f44a16be9950f1d55cdddaec5bb14ba828666124b25d0edce76448` |
| Main pool file | `ec025de56f0089a5ff3dc05483daca935bef653aebfe14b2a1c102b40d2205d9` |
| Authorization | `0f5e1f87871ff2b0592454916fc1dc3959f77214d447f8c3129c9f8d1e2bbb12` |
| Authorization file | `5dfb5ce3ca513546f890b9d69e06c56fd5faf6cb185a6fa7288e9e6cce8d618e` |
| Initial preflight | `530a189589c4f8a454602b3dc63209422b9484cd64b39a085ee2b46c241e0df3` |
| Corrected preflight | `a720a94e8d5733102c6bafe325880a9999b2d0463ec19ee145caeaeb17e596a5` |
| Corrected preflight file | `c7dd1319c94a17ede264f1f07f7d4478bc3c34b652d5558e09a94958d1eb7930` |
| Raw main result | `fa3e5d6afa6a0816022909e287c0ab51d05311e3ad66ceac69c4441e9a981d3f` |
| Raw main result file | `db995778cbc020782f832e12c0199318a9f9381789c456dc90ef9b51a8725fb0` |
| Analysis correction | `85b04c9681ce8cf9fbd340885944c8d6da69831e00adfad410ebfef7018a199b` |
| Analysis correction file | `175ee0be1e61b92e8e37269fb86a541a6c69da810a28e7f376f19d398d3422ae` |

The 23,181,923-byte raw
[manifest](sanmill-safe-inducement-main-v2-manifest-2026-08-16.json)
contains all 18,549 state/action/budget cells, semantic search records, Malom
transition labels, timing, resources, and access counters.  Preserve it with
the smaller
[analysis correction](sanmill-safe-inducement-main-v2-analysis-correction-2026-08-16.json).
The correction changes no raw observation and performs no new query.

## Protocol v2 and state pool

Protocol v1 remains unchanged.  Protocol v2 added exactly two secondary
analyses before main-pool construction:

- the same state/action cells were frozen at 1,000, 100,000, and 500,000
  nodes, while the unchanged primary gate remained exclusively at 100,000;
- a source-frequency-weighted secondary `o - b` was added without a threshold
  or authority to reverse the equal-phase primary decision.

The source phase frequencies were independently replayed before any main-pool
Malom query: 113,923 placement, 167,807 movement, and 10,462 flying decisions,
for 292,192 total.  Their weights are 38.989%, 57.430%, and 3.581%.
This weighting describes only the observed PlayOK-like source domain.

The new blind namespace selected 360 states, 120 per phase, with 360 unique
source games.  It excluded all 36 preprobe source coordinates and had zero
overlap.  No estimator prediction or Sanmill outcome participated in
selection, and no state was replaced after Malom or engine observation.  The
pool has 6,183 complete `A_pos` actions: 1,469 placement, 616 movement, and
4,098 flying.  Pool construction used 15,392 read-only, corrected Malom
queries.  Every label is `sector-corrected-v1`, positional-only `A_pos`; none
is an `A_allow` proof.

## Authorization and preflight

The tracked `authorization.json` records `product-owner-direct`, one execution,
and the exact 360-state, 40,000-search, 250,000-Malom-query, 14,400-second
envelope.  It forbids retry, resume, recovery, extension, complete games,
model loads, training, database writes, promotion, deployment, publication,
and release.

The initial preflight passed 18 determinism cells: two source-blind fixtures
per phase at all three budgets.  Forward and reverse fresh-process searches
and two searches in the same process matched exactly, for 72 searches total.
Timing and raw protocol text were excluded from semantic equality.

The first launch attempt stopped before writing the measurement-consumption
marker or making a measurement search.  Windows `tasklist` returned access
denied inside the host sandbox, and the fail-closed process check treated that
as a stop.  A focused reproduction replaced it with a fail-closed PowerShell
`Get-Process` count.  Twelve tests covered the zero-count and malformed-count
paths.  Corrected preflight `a720a94e...` reused the already-passed
determinism evidence with zero additional determinism or measurement searches,
verified the unchanged runtime, and bound a new untouched run namespace.
The direct authorization was still unconsumed under its frozen first-
measurement-search rule.

The exact execution runtime remained:

| Field | Value |
| --- | --- |
| Sanmill commit | `a6623f88959f7453594df274fbe1f128af7ff55e` |
| Sanmill tree | `17b9b0fd51ee8dac54c0454a6935978a47d19e0c` |
| Binary SHA-256 | `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619` |
| Runtime identity | `705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea` |
| Rules identity | `3e62cb93a1e0afe4534ce4824d233344816050b547bb8761dd7fe985d8ad399f` |
| Licence | `AGPL-3.0-or-later` |

## Main results

Each budget exhausts the same 6,183 actions and all 360 states.  Eight actions
per budget were already strict-terminal after the safe action and correctly
remain known zero-response indicators in the denominator.  There are no
abstentions.

| Nodes | Downgrades / actions | `b` | `o` | `o - b` (95% interval) |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 258 / 6,183 | 5.222% | 21.944% | 16.722% [13.233%, 20.309%] |
| 100,000 | 157 / 6,183 | 2.822% | 17.500% | 14.678% [11.363%, 18.178%] |
| 500,000 | 135 / 6,183 | 2.079% | 15.000% | 12.921% [9.791%, 16.276%] |

At the primary 100,000-node budget:

| Phase | Downgrades / actions | `b` | `o` | `o - b` (95% interval) |
| --- | ---: | ---: | ---: | ---: |
| Placement | 36 / 1,469 | 3.427% | 18.333% | 14.907% [9.280%, 20.834%] |
| Movement | 5 / 616 | 1.369% | 4.167% | 2.798% [0.625%, 5.565%] |
| Flying | 116 / 4,098 | 3.669% | 30.000% | 26.331% [18.993%, 33.914%] |

Movement is no longer a zero-event phase, but its 2.798% increment is much
smaller than placement and flying and is below the 5% mechanism threshold if
read as a separate phase.  The equal-phase overall result therefore does not
establish a phase-general mechanism of comparable magnitude.

The downgrade transitions remain separate:

| Nodes | W to D | W to L | D to L |
| ---: | ---: | ---: | ---: |
| 1,000 | 15 | 69 | 174 |
| 100,000 | 3 | 34 | 120 |
| 500,000 | 3 | 28 | 104 |

## Budget-invariant and budget-sensitive decomposition

The first sealed result normalized `o_inv`, `o_sens`, and `o_union` over only
the 85 states having an event at some budget.  That was a technical analysis
defect: 275 never-inducing states were absent from the state-flag map.  The
raw cells, per-budget summaries, primary decision, invariant share among
induced states, weighted secondary metric, and resource/access ledgers were
unaffected.  The original manifest remains immutable; correction identity
`85b04c96...` is the decision source for this section.

The corrected decomposition is:

| Scope | `o_inv` | `o_sens` | `o_union` | Invariant share of induced states |
| --- | ---: | ---: | ---: | ---: |
| Overall | 12.500% | 11.111% | 23.611% | 52.941% |
| Placement | 7.500% | 20.000% | 27.500% | 27.273% |
| Movement | 1.667% | 10.833% | 12.500% | 13.333% |
| Flying | 28.333% | 2.500% | 30.833% | 91.892% |

The action counts are 121 budget-invariant, 161 budget-sensitive, and 5,901
never-inducing.  `o_union = o_inv + o_sens` holds overall and in every phase.

Overall, the invariant share is 52.941%, below the frozen 80% interpretation
threshold.  The overall mechanism is therefore not described as almost
entirely a fixed evaluation blind spot; a budget-sensitive component is
non-negligible.  Flying alone is 91.892% invariant and is explicitly described
as a fixed-engine evaluation-blind-spot concentration.  Placement and movement
are predominantly budget-sensitive.  None of these one-step classifications
proves repeated-play reachability or complexity-manufacturing trap ability.

## Frequency-weighted secondary result

At 100,000 nodes, source-frequency weighting produces `o - b = 8.361%`, from
phase increments of 14.907% placement, 2.798% movement, and 26.331% flying.
The value is lower than the equal-phase 14.678% because movement supplies
57.430% of observed source decisions and has the smallest phase increment.
This secondary metric had no gate and did not change the primary decision.

## Resources and access

| Component | Engine searches | Malom queries | Active seconds |
| --- | ---: | ---: | ---: |
| Pool construction | 0 | 15,392 | 44.64 |
| Preflight | 72 | 0 | 8.32 |
| Main measurement | 18,525 | 33,171 | 1,707.02 |
| Aggregate | 18,597 | 48,563 | 1,759.97 |
| Authorized ceiling | 40,000 | 250,000 | 14,400 |

The run used at most one evaluator and one Sanmill process.  It played zero
complete games, loaded zero policy models, performed zero training or weight
updates, and made zero database writes.  Official selection, confirmation,
final-test, research-confirmation, and source pool `2eb04f54` content reads
were all zero; the remaining 108 source-pool records remain unconsumed.

## Verification and historical drift

- the focused safe-inducement and strict-referee group passed 22 tests;
- the mandatory Malom, DB-teacher, and label-provenance group passed 103 tests
  and 498 parameterized subtests;
- task-scope Ruff and `git diff --check` passed; and
- the fail-closed protected-access test raises before its producer is called.

The historical `sanmill_checkout` test group again reported 41 passes and four
fail-closed local integration failures.  Its moving checkout changes files
inside the old `db65eb3` bridge scope.  This is the previously recorded drift,
not the exact `sanmill_training_checkout` used here.  No path, test, or machine
registry was changed to conceal it.

## Final boundary

The unique decision is `mechanism_gate_passed`.  It is a result for this exact
fixed runtime, these source-game-unique positions, these three requested node
budgets, and positional-only `A_pos`.  It grants no authority for a second
execution, recovery, complete game, model training, reward change, promotion,
deployment, publication, or release.
