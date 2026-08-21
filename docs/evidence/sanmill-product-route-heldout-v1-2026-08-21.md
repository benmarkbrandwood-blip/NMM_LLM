# Sanmill Product-Route Held-Out Result — 21 August 2026

## Status and decision

The once-only held-out product-route comparison completed all 864 planned
games without retry, recovery, resume, extension, or result-contingent
changes. Both difficulty-specific primary endpoints produced the frozen
decision:

`classical_first_material_route_candidate`

This is a candidate classification for a separate product-owner decision. It
does not change the default route, model aliases, deployment, or release.

| Difficulty | Classical minus specialist | 95% interval | Half-width | Decision |
|---|---:|---:|---:|---|
| D9 | +14.5833pp | [+11.4834pp, +17.6832pp] | 3.0999pp | `classical_first_material_route_candidate` |
| D10 | +14.5833pp | [+11.4834pp, +17.6832pp] | 3.0999pp | `classical_first_material_route_candidate` |

The preregistered requirements were a half-width no greater than 4.0pp and a
lower endpoint of at least +5.0pp. Both endpoints satisfy both requirements.
Each estimate uses 108 independent starts after averaging the two candidate
colors within a start. The start-level difference distribution was one at
-0.25, 52 at zero, 46 at +0.25, and nine at +0.50.

## Frozen design and identities

The product owner directly authorized the exact final, never-consumed suffix
of source pool
`2eb04f542f88f8360f08f97e7657ca15646582a1532358dfeb04182ebad7d8f7`.
The plan froze before any suffix candidate move or result was opened.

- plan identity:
  `bae8e6ad8d23ba42f6ac68e5a3b8dcb8e9d53a98670e53637486f97989d1b0e1`;
- authorization identity:
  `9b28bdda1357081a72a18abf28736d18194134d57c16c32a12a771302158741c`;
- schedule identity:
  `e7a347463e29e797b408cbd1ad025ed58249d6f6a0f5fd2d89f1d02c837460d6`;
- suffix membership identity:
  `985329117fe8a1f2eb171de3d3a671482006a2e2b42f27847ac55114b98d860a`;
- suffix start-ID identity:
  `9dca4fa7af21e21d4656875c7190ef39d797203513fe0c16fd1a4d7f19421893`;
- execution commit:
  `de6da9020edc2ca14a7bdf54fd88772ccfbedf3f`;
- execution tree:
  `a8c11dce4b6fed8f4da311359912a761c4a42f8a`;
- result identity:
  `89d24d3abc99811ee644a00e99d300c5721c8d1410a7c1458370ff15ce80f28d`.

The 108 starts were source-pool records 254 through 361 in the existing
frozen order. They had zero overlap with the consumed 253-record prefix and
zero overlap with the earlier 48 development starts. No record was reordered
or replaced. The sample contains 54 placement starts and 54 movement starts;
it contains no flying-origin start. Games can and did enter the flying phase,
but this result does not independently cover flying-origin positions.

The four arms were kept distinct:

1. D9 specialist-first plus the final product `A_pos` gate;
2. D9 classical-first plus the same final product `A_pos` gate;
3. D10 specialist-first plus the final product `A_pos` gate; and
4. D10 classical-first plus the same final product `A_pos` gate.

Each arm used all 108 starts with candidate colors swapped, for 216 games per
arm. Execution followed master start order, D9 before D10, white before black,
and specialist-first before classical-first. No result was opened until all
four arms had completed.

## Exact product and referee contract

The specialist-first route ran the current `dev` classical coordinator,
applied the real `SpecialistRouter` override, and converged on the delivered
`ProductPositionalSafetyGate`. The classical-first route ran the same current
coordinator without the specialist override and converged on the same final
gate. Every game constructed a fresh product AI; the Rust transposition table
was retained only within that game.

D9 used 13,887,000 deterministic product nodes and D10 used 18,367,000.
Sanmill used runtime identity
`705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea`,
one thread, seed 42, and 100,000 nodes per response. The strict referee digest
was
`sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94`.

Malom was read-only, had label version `sector-corrected-v1`, and had content
identity
`c414fe88778f8d1d95cd3015532b43cad59f09e8398d8e46c42188b6829f3544`.
The constraint is position-only `A_pos`. It is not history-aware `A_allow` and
does not prove safety against repetition or no-progress history.

## Absolute strict W/D/L

| Difficulty | Route | W | D | L | Score rate |
|---|---|---:|---:|---:|---:|
| D9 | specialist-first + `A_pos` | 25 | 137 | 54 | 43.2870% |
| D9 | classical-first + `A_pos` | 83 | 84 | 49 | 57.8704% |
| D10 | specialist-first + `A_pos` | 25 | 137 | 54 | 43.2870% |
| D10 | classical-first + `A_pos` | 83 | 84 | 49 | 57.8704% |

The equal aggregate D9 and D10 values are measured, not an arm collapse. The
specialist-first D9 and D10 move sequences were equal in 216 of 216 games.
The classical-first sequences differed in nine of 216 games. Two of those
changed candidate score in opposite directions, so their aggregate W/D/L and
the route contrast cancelled exactly.

By start phase, the classical and specialist score rates were respectively
61.5741% and 45.3704% on the 54 placement starts, and 54.1667% and 41.2037%
on the 54 movement starts. These are descriptive secondary slices and carry
no separate decision threshold.

## Position-only gate and route behavior

All product decisions reported final gate status `applied`; selection failures
were zero.

| Difficulty and route | Product turns | Final interventions | Specialist success/fallback |
|---|---:|---:|---:|
| D9 specialist-first | 5,135 | 637 (12.41%) | 5,135 / 0 |
| D9 classical-first | 5,260 | 0 | not attempted |
| D10 specialist-first | 5,135 | 637 (12.41%) | 5,135 / 0 |
| D10 classical-first | 5,290 | 0 | not attempted |

For each specialist arm, interventions were 58 of 166 placement turns,
578 of 4,310 movement turns, and one of 659 flying turns. The classical route
already returned an `A_pos` move on every observed turn, so its identical final
gate recorded no rewrite. This is evidence about the exact delivered route,
not a claim that unrestricted classical search is universally safe.

The ledger classified internal Malom or solved-DB bypasses on 4,755 of 5,135
specialist turns in each difficulty, 4,912 of 5,260 D9 classical turns, and
4,931 of 5,290 D10 classical turns. These are descriptive implementation-path
counts; they do not change the primary route estimand.

## Terminal reasons

| Arm | Fifty-move | Threefold | Fewer than three | No legal moves |
|---|---:|---:|---:|---:|
| D9 specialist-first | 50 | 87 | 17 | 62 |
| D9 classical-first | 55 | 29 | 35 | 97 |
| D10 specialist-first | 50 | 87 | 17 | 62 |
| D10 classical-first | 56 | 28 | 35 | 97 |

Strict referee outcomes, not Malom labels, determined every terminal result.
The large route difference in threefold repetition is secondary evidence and
does not identify a causal mechanism by itself.

## Work and latency

| Arm | Route median/mean per product turn | Mean game time | Mean nodes/turn |
|---|---:|---:|---:|
| D9 specialist-first | 201.18 / 377.44 ms | 8.36 s | 151,060 |
| D9 classical-first | 4.53 / 98.17 ms | 2.76 s | 145,453 |
| D10 specialist-first | 57.16 / 244.14 ms | 6.21 s | 163,450 |
| D10 classical-first | 4.24 / 110.61 ms | 3.07 s | 171,366 |

Median node count was zero in every arm because the current coordinator often
used an internal Malom or solved-DB path. Maximum nodes matched the frozen D9
or D10 budget. Latency is descriptive only: the fixed schedule creates cache
and order effects, and the routes reach different trajectories. It is not a
randomized causal latency comparison.

## Preflight, resources, and access audit

The zero-game preflight independently established that the suffix was the
unconsumed 108-record tail, all histories replayed under the same strict
referee, both real product-route canaries passed, the product code and runtime
identities were clean and published, the three specialist checkpoints and
read-only databases resolved, and no protected segment was opened.

The completed execution consumed:

- 864 complete games of the 864-game maximum;
- 4,593.496 active seconds of the 18,000-second maximum;
- 41,673 engine single-step searches; and
- 404,865 read-only Malom queries.

All resource limits passed. No measurement or Sanmill process remained after
completion. Database writes, model fits, training updates, official selection
reads, official confirmation reads, official final-test reads, and research
confirmation reads were all zero. The 253-record prefix contributed no new
evidence, and the old 48 development starts did not enter the main sample.

Final focused, product-route, strict-dependency, and mandatory Malom
provenance verification passed 152 tests and 498 subtests. Pytest emitted only
the known host permission warning for its optional `.pytest_cache`; the local
`--basetemp` test work completed normally. Task-scope Ruff checks passed.

The once-only use consumed all 108 suffix records. The frozen source pool now
has zero unconsumed records and must not be reused as fresh held-out evidence.

## Independent recomputation

The independent recomputer parsed all 864 raw JSONL records, verified the
previous-record hash chain, and did not call the main analysis function. It
reproduced both primary endpoint objects without a difference.

- raw ledger SHA-256:
  `b42d9af4ef2694e6882d6a535b4543ce167f8f3809c903baee634cd1854fa1ca`;
- ledger tail identity:
  `043f8ed17ffa11ef689102bf0187c8b804bf1572ea8f6012867685dbc159e76e`;
- recomputation identity:
  `7d4685acc5b5c8c6e01bd79f864d4455476e72a8dde14ca1f96fe105b0e1baf0`;
- comparison differences: none.

The tracked machine result is
[sanmill-product-route-heldout-v1-manifest-2026-08-21.json](sanmill-product-route-heldout-v1-manifest-2026-08-21.json),
and the independent record is
[sanmill-product-route-heldout-v1-independent-recompute-2026-08-21.json](sanmill-product-route-heldout-v1-independent-recompute-2026-08-21.json).
The frozen plan is
[sanmill-product-route-heldout-v1.json](../experiments/sanmill-product-route-heldout-v1.json).

## Claim boundary

This result is limited to the exact final 108-start held-out suffix, the
current published `dev` implementation, the named checkpoints and read-only
databases, the exact Sanmill runtime, and position-only `A_pos`. It is not an
overall-strength, human-opponent, causal-mechanism, equivalence, or
history-aware safety result. It does not authorize a default-route change,
promotion, deployment, release, training, or external publication.
