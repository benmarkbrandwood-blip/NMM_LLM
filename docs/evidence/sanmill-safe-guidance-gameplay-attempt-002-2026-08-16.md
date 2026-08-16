# Sanmill safe-guidance gameplay attempt-002

Date: 2026-08-16

Decision: `inconclusive_no_score_difference_resolved`

## Executive result

The one authorized formal execution completed all 1,524 games from all 254
eligible starts.  Every game reached a strict rules terminal.  There were no
safety-cap incomplete games, retries, resumptions, parameter changes, or
result-based stops.

The frozen primary comparison was full-feature guidance minus random-safe,
with both colors averaged within each start.  The observed difference was
-0.5906 percentage points.  Its start-clustered 95-percent engineering
interval was [-1.6468, +0.4657] percentage points.  The half-width was 1.0562
percentage points, within the frozen 1.5-point precision limit, but the
interval crossed zero.  The only valid primary decision is therefore
`inconclusive_no_score_difference_resolved`.

This is not an equivalence result.  It also is not evidence that guidance is
better.  Under the exact tested runtime and start set, the interval excludes a
resolved positive advantage of about 0.5 percentage point or larger, while
remaining compatible with both a small advantage and a larger disadvantage.

## Frozen identities

- protocol:
  `1d368c336db5f49493a2abf3c9e7d507c013d9fed3d14cd928ee988575969cc6`;
- original start pool:
  `385a376dd82953c23c232f34e3dd5a84e5887b978c60627657eccfa6821eb6e9`;
- original membership:
  `cb84ed8180b103d7c25d56a5051fb2476047788505ed0cb9f437c39c9048fb15`;
- attempt-002 specification:
  `1610b74f48bd0b65c164b184e154376979fce274f0c58cbbc28c8322c3cc80e4`;
- non-evidence rehearsal:
  `1bfc156f3346b17bc738541fc27d45063049b9d1e1ea9e3ca2c1fa2f2ef93a03`;
- attempt-002 authorization:
  `4ac976a9fc1942b2a06fd39e25bf7af35fb34a5f28968ff5f8a381842839c3b2`;
- zero-measurement preflight:
  `87f0cb63d340a412c7562dd3430702dabdd66596b2cc4006ca2d898fd078e1a1`;
- result:
  `f7abf95ca688860fa3b871757f42397b351a3c00701ba27363028f800a9222f0`;
- result file SHA-256:
  `71ce03fecbe2c76347ca9d66e2aeb3082eca7d91ceeba48bcab2f958af52a10c`.

The formal execution used source commit
`87b2590a46c48fb9bd3de4febb4da63b9d923da3` and source tree
`80f0d3a93689a98f8b9369b49dac0106bc153639`.

## Attempt-002 eligibility and preconditions

The failed attempt-001 start
`00092c974cabf05874f066b8948e791f9fdc82d84a65759da1ba78f212a643b0`
was excluded.  Its outcome was not persisted, but a terminal state existed in
the failed process and absolute observer visibility could not be disproved.
The disposition therefore remained fail closed.

The original 1,530-game schedule was reconstructed first.  Only the six rows
for that start were removed.  All surviving ordinals, game IDs, colors, arms,
and random-safe seeds remained unchanged.  The resulting phase allocation was
84 placement, 85 movement, and 85 flying starts.

Before formal measurement, one four-game rehearsal used two starts outside
the frozen pool.  It covered a decisive terminal, three rules draws, result
packaging, per-game durable accounting, completion, and analysis.  It was
explicitly marked non-evidence.  The zero-measurement preflight then passed:

- seven negative contract, crash-recovery, and protected-data regressions;
- all 18 focused gameplay tests;
- 103 Malom, DB-teacher, and label-provenance tests plus 498 subtests;
- task-scope Ruff;
- the frozen six-state guide canary with exactly 1,000 Malom queries;
- 72 deterministic Sanmill search checks; and
- strict full-history replay of all 254 formal starts, excluding the failed
  attempt-001 start.

No formal measurement marker existed until these gates passed.

## Primary result

| Arm | Games | Wins | Draws | Losses | Strict score |
| --- | ---: | ---: | ---: | ---: | ---: |
| random-safe | 508 | 21 | 414 | 73 | 44.8819% |
| full-guided | 508 | 6 | 438 | 64 | 44.2913% |
| geometry-guided | 508 | 7 | 436 | 65 | 44.2913% |

Full-guided minus random-safe was -0.5906 points, with interval
[-1.6468, +0.4657] and half-width 1.0562 points.  The full-guided minus
geometry-guided secondary difference was exactly 0.0000 points, with interval
[-0.3866, +0.3866].  The geometry result is secondary and cannot alter the
primary decision; its narrow interval is not a preregistered equivalence test.

The 508 paired color cells had the following anatomy:

| Full score | Random score | Cells |
| ---: | ---: | ---: |
| 0.0 | 0.0 | 64 |
| 0.5 | 0.0 | 9 |
| 0.5 | 0.5 | 411 |
| 0.5 | 1.0 | 18 |
| 1.0 | 0.5 | 3 |
| 1.0 | 1.0 | 3 |

After averaging the two colors within each start, 224 of 254 start differences
were zero, 12 were +0.25 for full guidance, and 18 were -0.25.  This exactly
reproduces the frozen mean and interval.

## Inducement and conversion diagnostics

These endpoints are secondary.  In particular, conditioning on a game having
an induced event creates a selected subset, so the conversion proportions are
descriptive and are not a causal arm comparison.

| Arm | Events | Games with event | Wins in those games | Conversion |
| --- | ---: | ---: | ---: | ---: |
| random-safe | 131 | 131 | 13 | 9.9237% |
| full-guided | 128 | 126 | 2 | 1.5873% |
| geometry-guided | 127 | 126 | 3 | 2.3810% |

The full-feature guide did not produce more observed downgrade events than
random-safe.  Its observed event-to-win conversion was also lower.  Combined
with the unresolved negative primary point estimate, the experiment provides
no complete-game evidence that the previously observed one-step transfer
advantage converts into higher strict score.  It does not prove that guidance
causes harm, and it does not establish equivalence.

The transition counts were:

| Arm | D to L | W to D | W to L |
| --- | ---: | ---: | ---: |
| random-safe | 130 | 0 | 1 |
| full-guided | 118 | 5 | 5 |
| geometry-guided | 118 | 5 | 4 |

## Budget-type decomposition

| Arm | Budget-invariant | Budget-sensitive | Invariant share |
| --- | ---: | ---: | ---: |
| random-safe | 94 | 37 | 71.76% |
| full-guided | 90 | 38 | 70.31% |
| geometry-guided | 86 | 41 | 67.72% |

Most observed events were invariant across the frozen 1k, 100k, and 500k
node budgets.  The result therefore remains dominated by fixed evaluation
blind spots of this exact Sanmill runtime, not by a demonstrated general
complexity-manufacturing ability.  This decomposition cannot be generalized
to another engine or engine version.

For full guidance, the invariant/sensitive split by phase was 25/29 in
placement, 12/7 in movement, and 53/2 in flying.  The flying signal was thus
almost entirely budget invariant.

## Phase diagnostics

| Phase | Arm | Games | W / D / L | Score |
| --- | --- | ---: | --- | ---: |
| placement | random-safe | 168 | 1 / 153 / 14 | 46.1310% |
| placement | full-guided | 168 | 3 / 152 / 13 | 47.0238% |
| movement | random-safe | 170 | 4 / 132 / 34 | 41.1765% |
| movement | full-guided | 170 | 0 / 140 / 30 | 41.1765% |
| flying | random-safe | 170 | 16 / 129 / 25 | 47.3529% |
| flying | full-guided | 170 | 3 / 146 / 21 | 44.7059% |

The descriptive full-minus-random score differences were +0.8929 points in
placement, 0.0000 in movement, and -2.6471 in flying.  These phase results were
not separate primary claims and have no multiplicity-adjusted directional
decision.  They show where the aggregate point estimate arose, not a new
frozen conclusion.

## Rules terminal and draw diagnostics

All 1,524 formal games reached a strict rules terminal, so no safety-cap record
entered the score as a draw.

| Arm | Threefold draws | Fifty-move draws | All draws |
| --- | ---: | ---: | ---: |
| random-safe | 109 | 305 | 414 |
| full-guided | 292 | 146 | 438 |
| geometry-guided | 307 | 129 | 436 |

Full and geometry guidance changed the route to a draw: they produced many
more threefold repetitions and fewer no-progress draws than random-safe.  This
is a real process difference under the tested protocol, but it did not become
a higher strict score.  It also illustrates the known limitation of `A_pos`:
position safety does not include the repetition or no-progress history and is
not `A_allow`.

## Resource and durability audit

Attempt-002 used 1,528 complete games and 256 distinct starts when the four
non-evidence rehearsal games are included.  The final cumulative resource use
was:

- 50,404 of 80,000 Sanmill single-step searches;
- 6,257,569 of 20,000,000 Malom read-only queries;
- 2,254.03 of 21,600 active seconds;
- 1,528 of 1,536 complete games; and
- 256 of 256 distinct starts.

The 72 searches and 12,638 Malom queries from failed attempt-001 remain
separately disclosed sunk cost and are not included above.  The unknown
attempt-001 first-game increment remains unknown; no estimate replaced it.

The formal raw game ledger contains 1,524 records.  Its file SHA-256 is
`70734756f12eddd1dbf0095fec55c5ba7893ea401239b56ea00d92d3b7e6aa74`,
and its chain tail is
`24c42e0be6c360495a4299bb1ae8880fe058c53217feb115e44ecadf15ca8aff`.
The resource journal contains exactly 1,524 checkpoints.  Its file SHA-256 is
`6fc0ddedb992bb293848e79cf22d8569559692b2ed6be29cdccf2614fa9185ce`,
and its chain tail is
`01c86e9e977afdc22daedaba2d0b8146f60e432647e1c561cb1ddb4afa9e988e`.

An independent post-run load verified both chains, every canonical row hash,
the exact 4-game baseline, all 1,524 resource-to-game identity alignments, and
the final cumulative totals.

## Access and claim boundary

The run loaded no playing model, fitted no estimator, performed no training or
weight update, and wrote no database.  Official selection, confirmation,
final-test, research-confirmation, and the remaining 108 records of source
pool `2eb04f54` all had zero reads or consumption.

This is an exact fixed-runtime, fixed-start, positional-only complete-game
result.  It is not human-trap ability, human generalization, product value,
playing-strength, refresh-causality, or equivalence evidence.  It does not
authorize training, attempt-003, promotion, deployment, publication, release,
or any further execution.
