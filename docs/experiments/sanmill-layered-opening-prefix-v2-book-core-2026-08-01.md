# Twelve-ply layered-prefix Book core

Status: `book_membership_frozen_other_strata_pending`

Decision date: 2026-08-01

The accepted 22-place Book stratum now has deterministic source-only
membership. The machine-readable list is
[stored beside this document](sanmill-layered-opening-prefix-v2-book-core-2026-08-01.json)
with membership identity
`27563913709e56646f0646d74e6b4cd15c94b458250bd1c9dbb741e32239fa98`.

No candidate model was loaded and no game was played. This decision does not
freeze HumanDB or Perfect DB membership, the final 64-prefix list, or an
evaluation or training launch.

## Allocation

| Book subtype | Prefixes | Coverage rule |
| --- | ---: | --- |
| Expert-curated plays | 15 | One primary for each of 14 audited eight-ply parent orbits, then the first structurally new P03 extended-family priority |
| Sanmill named variations | 7 | One structurally new representative for every declared family |
| **Total** | **22** | Exact history, final FEN, and D4/`ring16` are all unique |

The expert-curated allocation preserves breadth before depth. The first 14
members are the breadth primaries from the reviewed shortlist. The fifteenth
is `expert-book-play-008`, the first entry in the pre-existing P03 extra order
that remains a distinct frozen coverage pattern.

For Sanmill named variations, complete entries retain the asset's source
order. Families are encountered in that same order. Within each family, the
first audit-defined representative that does not duplicate a previously
selected exact history, final FEN, or `ring16` orbit is chosen. A variation's
audit representative is not replaced by another capture expansion merely to
make it fit.

This is a technical arrangement under the expert's explicit delegation. D4
normalisation and the source-order tiebreak are project policies, not quoted
expert terminology.

## Frozen members

| ID | Subtype | Family/parent | Source member | Source name |
| --- | --- | --- | --- | --- |
| `book-core-001` | Expert | P01 | `expert-book-play-001-better-d1` | Better continuation |
| `book-core-002` | Expert | P02 | `expert-book-play-002` | Source line |
| `book-core-003` | Expert | P03 | `expert-book-play-003` | Source line; expert primary Black response |
| `book-core-004` | Expert | P04 | `expert-book-play-005` | Source line |
| `book-core-005` | Expert | P05 | `expert-book-play-007` | Source line |
| `book-core-006` | Expert | P06 | `expert-book-play-012` | Source line |
| `book-core-007` | Expert | P07 | `expert-book-play-013` | Source line |
| `book-core-008` | Expert | P08 | `expert-book-play-025` | Source line |
| `book-core-009` | Expert | P09 | `expert-book-play-027` | Source line |
| `book-core-010` | Expert | P10 | `expert-book-play-029` | Explicit continuation of row 28 |
| `book-core-011` | Expert | P11 | `expert-book-play-030` | Source line |
| `book-core-012` | Expert | P12 | `expert-book-play-031` | Source line |
| `book-core-013` | Expert | P13 | `expert-book-play-032` | P13-A representative |
| `book-core-014` | Expert | P14 | `expert-book-play-034` | Interrupted Knight source line |
| `book-core-015` | Expert | P03 | `expert-book-play-008` | Perpendicular Mill Rush priority |
| `book-core-016` | Sanmill | Early Game | `book-03-572229` | Game Three |
| `book-core-017` | Sanmill | Man-to-Man Marking | `book-22-7cd89e` | Man-to-Man Marking |
| `book-core-018` | Sanmill | Black Diamond | `book-23-5f1cc8` | Black Diamond |
| `book-core-019` | Sanmill | Mill Rush | `book-25-b14595` | Mill Rush — Parallel Lines |
| `book-core-020` | Sanmill | Battle Lines | `book-37-e4eb5c` | Battle Lines — Black Loss |
| `book-core-021` | Sanmill | Z Mill | `book-40-658b81` | Open Z Mill |
| `book-core-022` | Sanmill | novel | `novel-003932a3` | Z mill c4 — B line 1 |

The labels “Black Loss” and similar source names are retained as provenance;
they are not newly asserted outcome labels. Paired evaluation design, if later
authorised, must still assign each tested system to both colours on every
frozen history.

## Relationship to the full expert catalogue

The separate
[33-pattern Expert Book catalogue](sanmill-layered-expert-book-coverage-decision-2026-08-01.md)
is unchanged. Its remaining patterns continue to belong to the expert
diagnostic layer even though they do not all fit inside the balanced core's 22
Book places.

The subsequent
[HumanDB core decision](sanmill-layered-opening-prefix-v2-human-core-2026-08-01.md)
selects 21 genuine histories in frozen frequency order while excluding these
22 structures. Perfect DB membership remains the next source-only step.
