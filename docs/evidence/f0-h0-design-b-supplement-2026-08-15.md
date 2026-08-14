# F0-H0 Design B supplement -- 15 August 2026

Status: `completed_measurement_only_no_final_split_selection`

## Scope and non-decision

This read-only supplement measures the support of the March and May Design B
holdouts, the second-level scale of player and calendar subdivisions inside
the March pool, and a same-metric `ring16` comparator for Design B, the prior
Design C result, and a random whole-game baseline.

It does not select or recommend a final split.  It makes no feasible or
infeasible, continue or stop, scientific, product, equivalence, promotion, or
release decision.  The four F0-H0 scientific dimensions were not run.  E0,
F0-H1, and T0-H-pilot remain outside this measurement.

The labels `selection`, `one-time-confirmation`, and `final-test` below name
hypothetical measurement roles only.  No candidate membership is frozen as
the eventual untouched final test.  No human-error, trap, Malom, policy, or
product endpoint was calculated on any candidate role.

The machine-readable result is the
[supplement manifest](f0-h0-design-b-supplement-manifest-2026-08-15.json).
Its identities are:

- result identity:
  `a45fbfa0c472f86f03596b0618c799c4e0fb522bcfaa9b431efc904e838301a2`;
- result file SHA-256:
  `2bb06f06f55a86a14bdb30808dd68e617995acf4d663c1a1fefa99186866d850`;
- frozen plan identity:
  `889ccfcc407def9b7c2b4f3058611566e1bcb541976c42ed286d449dc67d633a`;
- frozen plan file SHA-256:
  `d96cc6cc13ce3cf44f7394d364db083159423923882f159902ef41a19ddb97e3`;
- preregistration commit:
  `1593dbf2f4a106910a0b036722e59f01cb79881f`.

The plan was committed and pushed at `2026-08-14T19:22:44Z`, before the
result file was created at `2026-08-14T19:27:50Z`.  No statistic in this
document was used to choose a cut pair, graph fraction, restart count,
sampling seed, baseline size, or interpretation threshold.

## Input boundary and method

The runner independently verified the mandatory F0-D0 and prior-result
identities before measurement:

| Item | Identity or file SHA-256 |
| --- | --- |
| F0-D0 corpus | `4c54d55209543e70edaeb33cb1dea25d2707312c3781580ba326ae35882dea29` |
| F0-D0 manifest | `bf7404d1f090073a1b36635b89d329e7011140d48e4fb3b3076efd7e55b5bca7` |
| F0-D0 manifest file | `0ab20955d551351ac25885b54d59a9f63fb6b2708e3292404d71dab2ff7dace6` |
| Prior split result | `cbfa6d43fa31e9644bae169e6b6d42232aa008e54921c96a46fbdddb73a95931` |
| Prior split result file | `eb0ed05a458b282a88b6bce12824a9744780238601609f446d7772b886dba77a` |

The behavior denominator remains exactly 92,226 games, 4,394,220 logical
plies, and 4,994 source-scoped player keys.  Outcome support remains a
separate F0-D0 flag with a corpus-wide denominator of 37,866 games.  A
recorded source result was never substituted for that flag.

Graph, date, game, player, move-count, and outcome-support measurements use
only the F0-D0 manifest.  Per-phase decisions and pre-decision `ring16` states
are absent from that manifest, so the frozen plan permitted opening only the
manifest-bound raw games needed for those measurements.  The runner verified
each opened file's path, size, SHA-256, session ID, framing, FEN, actor, turn,
phase, legal move, notation, and strict terminal status.

The union contained 80,719 raw files, 640,865,505 bytes, 3,835,847 decisions,
and 33,268 strict terminals.  There were no HumanDB reads, database writes,
Malom queries, games, searches, model loads, training updates, or reads from
the remaining `2eb04f54` source pool.

## March and May support

“Strong post” means that both player keys in a post-cut game are absent from
every behavior game before the cut.  Decisions are exact logical plies.
Phases are measured on the predecessor board with repository rules.

| Cut and side | Games | Players | Decisions | Placement | Movement | Flying | Strict-outcome games | Outcome players |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| March train | 36,949 | 2,216 | 1,742,416 | 658,641 | 1,025,077 | 58,698 | 15,135 | 1,539 |
| March strong post | 4,577 | 1,245 | 207,044 | 81,151 | 118,192 | 7,701 | 1,973 | 787 |
| May train | 69,284 | 3,824 | 3,288,902 | 1,234,196 | 1,949,750 | 104,956 | 28,684 | 2,650 |
| May strong post | 847 | 322 | 37,353 | 14,967 | 21,478 | 908 | 357 | 203 |

The strict-outcome shares are 43.11% for the March strong pool and 42.15%
for the May strong pool.  The March strong phase mix is 39.20% placement,
57.09% movement, and 3.72% flying.  The May mix is 40.07%, 57.50%, and
2.43%, respectively.

### Games per strong-pool player

Counts include both endpoints of every game.  Quantiles use the frozen
nearest-rank convention.

| Cut | min | p01 | p05 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| March | 1 | 1 | 1 | 1 | 1 | 2 | 5 | 12 | 21 | 118 | 410 |
| May | 1 | 1 | 1 | 1 | 1 | 2 | 5 | 11 | 18 | 39 | 122 |

| At least this many games | March players | May players |
| ---: | ---: | ---: |
| 1 | 1,245 | 322 |
| 2 | 787 | 199 |
| 5 | 364 | 89 |
| 10 | 161 | 41 |
| 20 | 68 | 13 |
| 50 | 27 | 3 |
| 100 | 14 | 2 |
| 250 | 3 | 0 |
| 500 | 0 | 0 |

These are source-account keys, not verified real-person identities.

## B1: player subdivision inside the March pool

The induced March graph is not a single connected component:

| Item | Measured value |
| --- | ---: |
| Player vertices | 1,245 |
| Games | 4,577 |
| Connected components | 31 |
| Giant-component players | 1,178 (94.62%) |
| Giant-component games | 4,465 (97.55%) |
| Non-giant players | 67 |
| Non-giant games | 112 |

The 30 non-giant components consist of one four-player component, five
three-player components, and 24 two-player components.  Their per-component
games and identities are all present in the manifest.  Weighted Louvain found
54 communities with modularity `0.679724271345805`.

The earlier whole-corpus zero-cut collapse does not repeat literally: small
components can be assigned without a cut.  It does repeat for the 94.62% of
players inside the giant component.  A three-role player split at useful
scale must either keep that giant component intact in one role or discard
games crossing a cut through it.

### Independent edge-cut scales

Each target is optimized independently, so internal-game counts need not be
monotone with the target player fraction.

| Target players | Measured players | Holdout games | Train games | Discarded cross games | Discard share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25% | 311 | 325 | 4,043 | 209 | 4.57% |
| 33.33% | 415 | 196 | 4,045 | 336 | 7.34% |
| 50% | 622 | 2,178 | 1,872 | 527 | 11.51% |

The 25% and 33.33% cuts again show that minimizing crossing edges can select
many low-activity players while preserving few internal games.

### Simultaneous three-way edge cut

The frozen 50%/25%/25% measurement produced:

| Hypothetical role | Players | Internal games |
| --- | ---: | ---: |
| selection | 623 | 1,912 |
| one-time-confirmation | 311 | 715 |
| final-test | 311 | 1,213 |

It discards 737 cross-role games, or 16.10% of the March pool.  The measured
membership identity is
`57749ebf5e60beb4a4a9dca599dcb15e2909f66c41967552715db4f7782a1a9f`.
That identity records one deterministic scale measurement; it is not a
selected or frozen final split.

## B2: calendar subdivision inside the March pool

The 4,577-game March pool was first divided by calendar.  “Pool games” are
the March pool games falling in that segment.  “Strong games” reapply the
strict rule at each segment boundary: both players must be absent from every
behavior game before that segment starts.  Thus the three pool-game counts
always sum to 4,577, while later strong counts can be smaller.

| Frozen candidate | Role | Pool games / players | Strong games / players | Strict outcomes / players | Strong decisions (P/M/F) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 Apr, 1 May | selection | 887 / 295 | 887 / 295 | 399 / 191 | 41,130 (15,780/23,553/1,797) |
| 1 Apr, 1 May | confirmation | 1,648 / 444 | 386 / 160 | 160 / 108 | 17,889 (6,864/10,463/562) |
| 1 Apr, 1 May | final | 2,042 / 606 | 847 / 322 | 357 / 203 | 37,353 (14,967/21,478/908) |
| 17 Apr, 3 Jun | selection | 1,686 / 489 | 1,686 / 489 | 755 / 313 | 77,884 (29,965/44,131/3,788) |
| 17 Apr, 3 Jun | confirmation | 2,245 / 638 | 773 / 333 | 325 / 220 | 33,704 (13,708/19,015/981) |
| 17 Apr, 3 Jun | final | 646 / 217 | 22 / 25 | 8 / 10 | 713 (378/328/7) |
| 1 May, 1 Jun | selection | 2,535 / 697 | 2,535 / 697 | 1,112 / 442 | 115,534 (45,032/65,333/5,169) |
| 1 May, 1 Jun | confirmation | 1,253 / 396 | 469 / 201 | 200 / 134 | 20,470 (8,337/11,642/491) |
| 1 May, 1 Jun | final | 789 / 252 | 58 / 44 | 26 / 23 | 1,854 (994/850/10) |

These rows are the requested scale measurements.  In particular, the
late-segment strong support ranges from 22 to 847 games across the three
frozen candidate pairs.  This document does not define how much support is
enough and therefore does not convert that range into a design decision.

## Same-metric `ring16` comparison

State identity is the repository's `ring16` canonical pre-decision FEN.
For each partition, decision-weighted overlap is the fraction of its
decisions whose orbit occurs in at least one other partition in the same
profile.  Every new profile uses whole-game membership, strict replay, and
the same orbit implementation as the prior Design C measurement.

### Coarse Design B and random whole-game baseline

The random baseline is a preregistered, deterministic draw of two disjoint
whole-game sets from all 92,226 behavior games.  Its set sizes exactly match
March train and March strong post: 36,949 and 4,577 games.  It applies no
player or time isolation.

| Profile | Large side | Small side |
| --- | ---: | ---: |
| Design B March train/test | 34.32% | 53.60% |
| Random whole-game left/right | 37.55% | 57.31% |
| B minus random | -3.23pp | -3.72pp |

The exact-orbit Jaccard is 4.85% for coarse Design B and 5.87% for random.
This control establishes that a large absolute overlap is substantially a
property of recurring Nine Men's Morris states, not automatically a split
defect.  Time plus unseen-player isolation lowers the measured overlap by
about 3--4 percentage points in this one frozen comparison.

### Four-role profiles

| Profile | Train | Selection | Confirmation | Final |
| --- | ---: | ---: | ---: | ---: |
| B2, 1 Apr / 1 May | 28.11% | 49.55% | 60.79% | 53.15% |
| B2, 17 Apr / 3 Jun | 28.70% | 51.09% | 50.69% | 47.97% |
| B2, 1 May / 1 Jun | 30.52% | 52.33% | 51.30% | 55.45% |
| Prior Design C | 51.38% | 66.69% | 61.04% | 64.57% |

The prior Design C values are numerically higher than the measured B2 values
for most roles, while the random comparator confirms a high game-intrinsic
baseline.  These rows do not isolate one cause.  Design C has four
player-owned decision sets with different sizes and allows the two actors in
one game to occupy different roles.  The random control has two whole-game
sets, and the B2 profiles retain only segment-specific unseen-player games.
Because the number of comparison partitions, support, and exposure differ,
the table is a structural comparator, not an effect estimate.

In the prior Design C result, zero exact `ring16` orbits were shared between
roles within one game; all exact-orbit intersections were cross-game.  That
does not remove its separately measured same-game trajectory leakage: nearby,
non-identical states from one two-player game are still assigned to the two
actors' roles.  Whole-game Design B avoids that trajectory split by
construction.

## Inherited bias and claim boundary

F0-D0 history recovery is not random.  The excluded 1,751 games contain only
35 recorded draws, while the retained 92,789 contain 26,157.  All support in
this supplement inherits that selection.  The 54,923 nonterminal histories
outside strict outcome support still have no independently verifiable source
termination basis.

Player keys are stable source accounts only.  UI orientation, time control,
exact source rule variant, and upstream file or import-batch identity remain
unrecoverable.  Results are limited to the observed PlayOK-like source domain
and cannot be transported to the product UI, other time controls, new
populations, or real-person identities.

No final membership has been chosen, and no support threshold was registered
for this supplement.  A later product choice must use these scale numbers
without relabelling them a scientific pass.  Any eventual frozen split must
receive a new identity before the four F0-H0 dimensions begin.  This work
grants no authority for Malom queries, E0, F0-H1, T0-H-pilot, games, search,
model loading, training, database work, source-pool use, promotion,
publication, or release.

## Verification

The focused synthetic suite covers support and outcome separation, March-pool
calendar segmentation, segment-specific player novelty, deterministic
disjoint baseline sampling, three-way graph accounting, decision-weighted
overlap, and exact raw replay with phase accounting.  Evidence-consistency
tests additionally bind the sealed plan, result, narrative, and prohibited
operation counters.

The sealed-result verifier returned identity
`a45fbfa0c472f86f03596b0618c799c4e0fb522bcfaa9b431efc904e838301a2`
and file SHA-256
`2bb06f06f55a86a14bdb30808dd68e617995acf4d663c1a1fefa99186866d850`.
The two focused test files passed 11 tests.  Ruff passed on the measurement
module, runner, and both focused test files.  `git diff --check`, final Git
state, and remote synchronization are verified at publication.
