# F0-H0 corrected split-feasibility measurement -- 15 August 2026

Status: `completed_measurement_only_no_split_selection`

## Scope and non-decision

This work measures the scale of three candidate corpus splits after the F0-H0
v1 zero-cut constraint was found to be structurally defective.  It does not
select or recommend a split.  It makes no feasible/infeasible, continue/stop,
scientific, product, equivalence, or release decision.

The four F0-H0 screening dimensions were not run.  There were no Malom
queries, games, searches, model loads, training updates, HumanDB reads, or
database writes.  E0, F0-H1, and T0-H-pilot remain outside this measurement.

The machine-readable result is the
[corrected measurement manifest](f0-h0-corrected-split-feasibility-manifest-2026-08-15.json).
Its identities are:

- result identity:
  `cbfa6d43fa31e9644bae169e6b6d42232aa008e54921c96a46fbdddb73a95931`;
- result file SHA-256:
  `eb0ed05a458b282a88b6bce12824a9744780238601609f446d7772b886dba77a`;
- frozen plan identity:
  `e1d2241cc23da1227fde7a3f84d2ff4c43a4167c2020d521abbc9f3eee1f833c`;
- preregistration commit:
  `71815fa03c3b4ddea06d87f94aa5ad7db0a10954`.

The plan was committed and pushed before any corrected graph, time, split, or
state-overlap statistic was calculated.  The repository base frozen by that
plan is `61f0d97d199fe81e7c45c7b4290e8c4700638187`.

The frozen plan's `frozen_at_utc` value is
`2026-08-15T00:00:00Z`.  That field mistakenly labelled the local date
boundary as UTC and must not be used as the chronology clock.  The immutable
file is preserved rather than silently rewritten.  The preregistration commit
time is 15 August 2026 at 01:54:02 +08:00
(`2026-08-14T17:54:02Z`), and the result file was created at 02:12:20 +08:00.
The pushed Git commit therefore establishes the required before/after order
despite the metadata-label defect.

## Input boundary and raw replay

Before measurement, the runner independently recomputed all mandatory F0-D0
identities:

| Item | Identity |
| --- | --- |
| Corpus | `4c54d55209543e70edaeb33cb1dea25d2707312c3781580ba326ae35882dea29` |
| F0-D0 manifest | `bf7404d1f090073a1b36635b89d329e7011140d48e4fb3b3076efd7e55b5bca7` |
| F0-D0 manifest file | `0ab20955d551351ac25885b54d59a9f63fb6b2708e3292404d71dab2ff7dace6` |

The behavior boundary is exactly 92,226 games, 4,394,220 logical plies, and
4,994 source-scoped player keys.  These values, not an estimate or HumanDB
aggregate, are every denominator below.

The F0-D0 manifest is sufficient for graph, player, game-count, and calendar
measurements.  It does not contain each pre-decision board state.  Exact
Design C `ring16` overlap therefore required opening the 92,226 raw game files
already named and hashed by F0-D0.  Before use, the runner verified every file
size, SHA-256, session ID, JSONL framing, FEN, actor color, turn number, phase,
legal move, notation, and terminal replay.  It read 734,090,532 bytes and
strictly replayed all 4,394,220 decisions.  It did not open any record from the
remaining `2eb04f54` source pool.

## Supersession boundary

The exact v1 defect and preserved identities are recorded in the
[v1 supersession correction](f0-h0-v1-supersession-correction-2026-08-15.md).
The v1 plan, split, result, and evidence remain byte-for-byte unchanged.  The
replacement scope is split-scale measurement only.

## Graph structure

The statement that the behavior graph has one giant component is confirmed:

| Item | Measured value |
| --- | ---: |
| Player vertices | 4,994 |
| Game edges, including repeated pairs | 92,226 |
| Unique opponent pairs | 24,127 |
| Self-games | 0 |
| Connected components | 1 |
| Giant-component players | 4,994 |
| Giant-component games | 92,226 |
| Non-giant components | none |

Connectivity explains why a zero-cut split collapses.  It does not determine
the scale available after controlled edge discards.

### Degree distribution

Quantiles use the preregistered nearest-rank convention.

| Quantile | Distinct opponents | Games per player |
| --- | ---: | ---: |
| min | 1 | 1 |
| p01 | 1 | 1 |
| p05 | 1 | 1 |
| p10 | 1 | 1 |
| p25 | 1 | 1 |
| p50 | 1 | 3 |
| p75 | 4 | 8 |
| p90 | 12 | 33 |
| p95 | 40 | 131 |
| p99 | 198 | 896 |
| max | 386 | 3,907 |

The measured tails are:

| Threshold | Players with at least that many opponents | Players with at least that many games |
| ---: | ---: | ---: |
| 5 | 1,042 | -- |
| 10 | 601 | 1,131 |
| 25 | 322 | 600 |
| 50 | 221 | 383 |
| 100 | 140 | 269 |
| 250 | 21 | 169 |
| 500 | -- | 109 |
| 1,000 | -- | 38 |

### Community structure

Weighted NetworkX Louvain with resolution 1.0 and seed 20260815 found 19
communities.  Game count per unique opponent pair was the edge weight.  The
modularity is `0.27306094974065637`.

Community player sizes, largest first, are:

```text
1553, 855, 622, 398, 387, 284, 264, 193, 183, 110,
66, 45, 15, 6, 4, 3, 2, 2, 2
```

The player-size nearest-rank quantiles are min 2, p25 4, p50 110, p75 387,
p90 855, p95 1,553, and max 1,553.  Per-community internal games, boundary
games, unique-pair counts, and identities remain in the machine manifest.

## Time structure

The observed dates span 11 December 2025 through 19 July 2026.  ISO weeks
contain the following exact number of games:

| Week start | Games | Player keys |
| --- | ---: | ---: |
| 2025-12-08 | 239 | 75 |
| 2025-12-15 | 804 | 182 |
| 2025-12-22 | 1,758 | 286 |
| 2025-12-29 | 2,655 | 378 |
| 2026-01-05 | 3,259 | 404 |
| 2026-01-12 | 4,114 | 435 |
| 2026-01-19 | 4,088 | 453 |
| 2026-01-26 | 3,884 | 427 |
| 2026-02-02 | 4,429 | 441 |
| 2026-02-09 | 4,443 | 451 |
| 2026-02-16 | 3,881 | 416 |
| 2026-02-23 | 4,057 | 447 |
| 2026-03-02 | 3,862 | 403 |
| 2026-03-09 | 3,539 | 444 |
| 2026-03-16 | 3,625 | 408 |
| 2026-03-23 | 4,200 | 461 |
| 2026-03-30 | 3,689 | 411 |
| 2026-04-06 | 3,714 | 399 |
| 2026-04-13 | 3,126 | 374 |
| 2026-04-20 | 3,877 | 430 |
| 2026-04-27 | 3,482 | 372 |
| 2026-05-04 | 2,856 | 355 |
| 2026-05-11 | 3,530 | 398 |
| 2026-05-18 | 3,473 | 376 |
| 2026-05-25 | 2,689 | 316 |
| 2026-06-01 | 3,623 | 360 |
| 2026-06-08 | 2,508 | 343 |
| 2026-06-15 | 1,411 | 259 |
| 2026-06-22 | 710 | 180 |
| 2026-06-29 | 318 | 140 |
| 2026-07-06 | 305 | 105 |
| 2026-07-13 | 78 | 42 |

The weekly game counts sum to 92,226.

For each frozen cut, pre means strictly before the date and post begins on the
date.  Player classes are mutually exclusive.  The three game-touch columns
count games touching at least one member of that player class and therefore
can overlap with one another.

| Cut | Pre games | Post games | Pre-only players / touching games | Post-only players / touching games | Spanning players / touching games |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-03-01 | 36,949 | 55,277 | 1,818 / 16,591 | 2,778 / 27,227 | 398 / 85,331 |
| 2026-04-01 | 54,095 | 38,131 | 2,683 / 27,995 | 1,902 / 14,836 | 409 / 85,998 |
| 2026-05-01 | 69,284 | 22,942 | 3,449 / 37,364 | 1,170 / 8,850 | 375 / 84,199 |
| 2026-06-01 | 83,273 | 8,953 | 4,210 / 53,418 | 489 / 2,488 | 295 / 78,590 |
| 2026-07-01 | 91,652 | 574 | 4,814 / 79,660 | 57 / 109 | 123 / 52,875 |

Player-game incidence counts for each class, which count both endpoints, are
also retained in the manifest.

## Design A scale: player cut with discarded games

Each target was measured independently.  The frozen algorithm used Louvain
communities for initialization, fixed-cardinality weighted Kernighan--Lin
refinement, eight deterministic restarts, and selection by the fewest cut
games followed by membership identity.  It did not seek representative or
high-volume holdout players; its preregistered objective was minimum cut.

| Target | Holdout players | Holdout internal games | Train internal games | Cross-cut games discarded | Discard share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | 250 | 106 | 91,882 | 238 | 0.2581% |
| 10% | 499 | 109 | 91,632 | 485 | 0.5259% |
| 20% | 999 | 104 | 91,136 | 986 | 1.0691% |

The non-monotone holdout-game count is not a transcription error.  The three
targets are separate optimized cuts, and low-volume player vertices can make
a large player count coexist with very few internal holdout games.  The
result reports scale only; it does not claim that any row is a suitable final
split.

## Design B scale: calendar holdout

Time-only retains every post-cut game, including returning players.  The
strong column retains only post-cut games for which both player keys were
absent from every pre-cut game.

| Cut | Time-only post games | Time-only post players | Both-new post games | Players in both-new games |
| --- | ---: | ---: | ---: | ---: |
| 2026-03-01 | 55,277 | 3,176 | 4,577 | 1,245 |
| 2026-04-01 | 38,131 | 2,311 | 1,319 | 523 |
| 2026-05-01 | 22,942 | 1,545 | 847 | 322 |
| 2026-06-01 | 8,953 | 784 | 58 | 44 |
| 2026-07-01 | 574 | 180 | 0 | 0 |

The strong membership identity for every row is frozen in the manifest.  The
counts show the trade between calendar recency and unseen-player volume but
do not choose a cut.

## Design C scale: player-owned decisions

The frozen hash split assigns every player key to exactly one partition using
70%, 15%, 7.5%, and 7.5% target ratios.  Each logical ply belongs only to its
actor.  A game is counted for every partition whose player owns at least one
decision in it.

| Partition | Player keys | Decisions | Games with a decision |
| --- | ---: | ---: | ---: |
| train | 3,496 | 3,304,646 | 86,494 |
| selection | 749 | 498,841 | 19,138 |
| one-time-confirmation | 375 | 268,615 | 10,992 |
| final-test | 374 | 322,118 | 12,843 |

There are 54,717 games whose two actors are in the same partition and 37,509
games whose actors are in different partitions.  The four decision counts
sum exactly to 4,394,220.  The player-membership identity is
`a524cdd63bf5ff9aa7a66c5c10af7521044eaac58b3d9481b28d69205ae67e25`.
This is a measured candidate membership, not a selected final split.

### `ring16` state overlap

The repository's `ring16` canonical pre-decision FEN defines state identity.
There are 1,885,533 unique orbits across all four partitions.

“Unique overlap” is the share of a partition's unique orbits also seen in any
other partition.  “Decision-weighted overlap” is the share of that
partition's decisions whose orbit also occurs in another partition.

| Partition | Unique orbits | Shared orbits | Unique overlap | Decisions on shared orbits | Weighted overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 1,506,893 | 203,060 | 13.48% | 1,698,016 | 51.38% |
| selection | 287,839 | 128,720 | 44.72% | 332,656 | 66.69% |
| one-time-confirmation | 179,145 | 77,556 | 43.29% | 163,965 | 61.04% |
| final-test | 202,142 | 92,022 | 45.52% | 207,997 | 64.57% |

Pairwise exact-state overlap is:

| Pair | Shared orbits | Jaccard | Share of left | Share of right | Seen in same game | Cross-game only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train / selection | 122,498 | 7.33% | 8.13% | 42.56% | 0 | 122,498 |
| train / confirmation | 73,006 | 4.53% | 4.84% | 40.75% | 0 | 73,006 |
| train / final-test | 86,889 | 5.36% | 5.77% | 42.98% | 0 | 86,889 |
| selection / confirmation | 36,082 | 8.37% | 12.54% | 20.14% | 0 | 36,082 |
| selection / final-test | 42,416 | 9.48% | 14.74% | 20.98% | 0 | 42,416 |
| confirmation / final-test | 27,621 | 7.81% | 15.42% | 13.66% | 0 | 27,621 |

An exact pre-decision state includes the side to move.  In one two-player
game, a repetition of that state returns to the same actor, so exact shared
orbits across different actor partitions are zero within the same game.
This does not remove trajectory leakage: adjacent and other nearby states
from the same game are still split between the two actors' partitions.

### Same-game trajectory leakage by ply distance

The following counts include every cross-partition decision pair from the
same game, whether or not its two states have the same `ring16` identity.  A
game belongs to exactly one pair of actor partitions.  Distance 2 is zero
because actors alternate and the same actor owns decisions two plies apart.

| Pair | Cross-partition games | d=1 | d=2 | d=3--4 | d=5--8 | d=9--16 | d=17+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train / selection | 15,235 | 720,433 | 0 | 690,284 | 1,290,350 | 2,221,256 | 6,876,310 |
| train / confirmation | 8,613 | 395,479 | 0 | 378,380 | 705,570 | 1,207,051 | 3,563,274 |
| train / final-test | 10,135 | 476,968 | 0 | 456,807 | 853,253 | 1,465,540 | 4,413,595 |
| selection / confirmation | 1,197 | 56,115 | 0 | 53,735 | 100,369 | 172,455 | 521,834 |
| selection / final-test | 1,507 | 75,452 | 0 | 72,444 | 135,891 | 235,961 | 773,337 |
| confirmation / final-test | 822 | 38,388 | 0 | 36,751 | 68,606 | 117,628 | 360,591 |

## What each design can and cannot estimate

This section analyzes claim boundaries.  It is not a recommendation.

### Design A

After discarding cross-cut games, Design A can support development measurement
on held-out source player keys and a source-domain unseen-account claim for
those keys.  It cannot by itself support future-time generalization, real
person identity beyond the recovered account key, or transport to the product
UI.

Its mechanism estimates can target held-out actors.  Complete-game product
endpoints exclude every cross-cut game, changing the natural-game estimand.
Source, calendar, opponent-network, and repeated-position dependence can
remain even after the cross-cut games are removed.

### Design B

The time-only population supports a future-period source-traffic claim but
not unseen-player generalization when players return after the cut.  The
strong both-new subset supports a future-period unseen-account claim for that
subset.  Neither transports to the product UI.

Mechanism endpoints can be defined on either named population.  Factual
product endpoints on the both-new subset no longer represent all later
traffic.  Returning-player identity and learning history cross the time-only
cut; removing them changes the post-cut traffic population.

### Design C

Design C supports actor-choice estimates on held-out decision-maker keys.  It
does not support independent complete-game outcomes, future-time
generalization, an untouched trajectory or state distribution, or transport
to the product UI.

Actor-level mechanism labels remain uniquely owned, but uncertainty must be
clustered by game and player.  A complete-game product endpoint cannot be
assigned without a separate game-level rule.  One game can contribute nearby
trajectory states to multiple partitions, and `ring16`-equivalent states can
recur across different games and partitions at the measured rates above.

## Access and operation audit

| Operation | Measured count |
| --- | ---: |
| F0-D0 manifest reads | 1 |
| Manifest-bound raw files opened | 92,226 |
| Manifest-bound raw bytes read | 734,090,532 |
| HumanDB reads | 0 |
| Malom queries | 0 |
| Database writes | 0 |
| Source-pool `2eb04f54` artefact reads | 0 |
| Source-pool records consumed | 0 |
| Games started | 0 |
| Search batches started | 0 |
| Models loaded | 0 |
| Training updates | 0 |
| F0-H0 scientific dimensions run | 0 |
| Final split selections | 0 |

## Measurement-only disposition

The three designs now have exact measured scale and named leakage boundaries.
No design is selected.  No design is described as feasible or infeasible.
The corrected result does not itself reopen or complete F0-H0 and does not
authorize the four scientific dimensions or any later stage.
