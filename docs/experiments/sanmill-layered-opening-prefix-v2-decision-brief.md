# Twelve-ply layered opening-prefix corpus decision brief

Status: `needs_decision`

Initial decision date: 2026-07-25

Evidence update: 2026-07-26

This brief compares the completed source-only Sanmill Book, expert-curated
Book, HumanDB, and Perfect DB audits and proposes, but does not freeze, a
64-prefix composition. It does not load a candidate model, play games, replace
the movement/flying corpus, or authorize an evaluation launch.

## Frozen evidence inputs

| Stratum | Audit identity | Essential source identity |
| --- | --- | --- |
| Sanmill Book subtype | `b0d3bd526b7ab951c7dbd10552c77e8f6db5f1e69510829bdd642049bff72c48` | corrected 109-position/437-recommendation asset SHA-256 `cdc4768bc461c22177634985a4cc1d92452774e2992515b937fed8812eb076f5` |
| Expert Book subtype | `29293523ba211775827bfd903ef8dd0a0220bcd958a46974e37809a55dd51ff4` | source identity `9c0c3ae58aaec67cb752899121cec465d19dd2bf332a869fddb0216e36a1463f` |
| HumanDB | `15aa81528a93d0777cc6479a285ff41058ae5860383b24d2629038cd779478fa` | immutable snapshot SHA-256 `97be7152573815180df6950b6150c667b1e5c2c8b1b21748b3ed9cf020b6f93c` |
| Perfect DB | `fcac7e8e864e345669a497c600e4a901eff1f9203e3b83baf6895a89c2b0be56` | portable identity `d6a85d25e87e28cc33e1408539585dadd83349da1cb63aa3f1a0f32307087508` |

The detailed records are:

- [Book source audit](../evidence/sanmill-layered-book-source-audit-2026-07-25.md);
- [expert Book play audit](../evidence/sanmill-layered-expert-book-source-audit-2026-07-26.md);
- [HumanDB source audit](../evidence/sanmill-layered-human-source-audit-2026-07-25.md);
  and
- [Perfect DB source audit](../evidence/sanmill-layered-perfect-source-audit-2026-07-25.md).

## Evidence summary

### Book

The state-indexed Oracle query graph has no pure Book route through logical ply
12. All 1,472 histories entering the final ply fail closed with `book_miss`.
It therefore contributes no `oracle_query_book` candidate to this corpus.

The author-defined named lines are a different representation. Of 107
variations:

- 84 reach twelve legal logical plies;
- 22 are shorter than twelve;
- one is unreplayable at the required boundary;
- the 84 complete variations expand to 112 capture-resolved records; and
- those records contain 110 unique exact histories, final FENs, and ring16
  orbits.

The complete variations span seven declared families. One representative per
selected variation is consistent with the Mill expert's advice, but selecting
all 84 would leave no room for the other required strata.

The later expert-curated document contributes a second Book subtype:

- 35 source rows contain 36 explicit candidates;
- all 36 replay legally through project rules and two fresh pinned Sanmill
  processes;
- they reduce to 34 exact histories, 33 final FENs, and 32 final `ring16`
  orbits;
- their eight-ply parents reduce to 15 exact histories and 14 `ring16` orbits;
  and
- one parent supplies 16 of the 36 twelve-ply children.

Rows 14/20 and 18/19 are exact duplicates. Six expert histories have exact
support in 29 distinct games in the frozen PlayOK sample, with a maximum of
nine games for one history. This corroborates that some lines occur in human
play, but it does not make the whole delivery a HumanDB-frequency source.

The expert subtype has no exact-history overlap with the Sanmill named lines.
It shares eight unique final FENs and eight unique final `ring16` orbits with
them, so concatenating both Book pools would still duplicate structures.
Row 11's final `c5` was read unambiguously from its embedded move-list
screenshot; the visual evidence basis remains recorded but is not a blocker.

### HumanDB

The immutable snapshot and recursive PlayOK source audit yield:

- 92,939 distinct eligible games;
- 83,002 unique exact twelve-ply histories;
- 77,828 singleton histories;
- 5,174 histories supported by at least two games;
- 15,111 games covered by those repeated histories, or 16.259% of eligible
  games; and
- a maximum exact-history frequency of 76 games.

The top 21 histories contain 572 distinct-game observations. Their minimum
support is 18 games before cross-source deduplication. The top 64 histories
cover 1,165 games, or 1.254% of the whole eligible sample. HumanDB can
therefore supply a well-supported small stratum, but no small list should be
described as representing most human openings.

### Perfect DB

The fixed pre-result audit pool contains 128 deterministic StrictSteps routes.
All 128 have distinct exact histories, final FENs, and ring16 orbits. Across
1,536 logical-ply choices, every selected outcome is a database draw with WDL
0; 1,450 choices have multiple tied-best candidates and 86 have one best
candidate. Candidate pools range from one to 24.

The two fresh Sanmill processes produced byte-identical canonical evidence.
There was no Book or HumanDB overlap at exact-history, final-FEN, or ring16
level. This demonstrates enough independent Perfect DB structure for any
plausible share of a 64-prefix corpus; it does not make those routes
human-common.

## Cross-source structure

The source inventories have these unique-set sizes:

| Source | Exact histories | Final FENs | Ring16 orbits |
| --- | ---: | ---: | ---: |
| Book named-line records | 110 | 110 | 110 |
| Expert Book records | 34 | 33 | 32 |
| HumanDB | 83,002 | 64,468 | 38,237 |
| Perfect DB audit pool | 128 | 128 | 128 |

Unique Book/HumanDB intersections contain three exact histories, nine final
FENs, and 22 ring16 orbits. Because many human histories can reach one
structure, 36 HumanDB histories end at a Book FEN and 528 end in a Book ring16
orbit. The audited Perfect DB pool has zero overlap with either source on all
three measures.

The expert Book pool intersects HumanDB at six exact histories, 12 unique
final FENs, and 17 unique final `ring16` orbits. It has zero overlap with the
fixed Perfect DB pool on all three measures. Against the Sanmill Book subtype
it has zero exact-history, eight final-FEN, and eight `ring16` intersections.

Final selection should remove cross-stratum endpoint and ring16 duplicates
unless a later contract gives a specific reason to retain the same structure
as a controlled source comparison.

## Legal-history and colour boundary

Every accepted record replays from `startpos` to exactly twelve complete
logical plies with final per-side counts `[6, 6]`. A Mill-forming action and
its mandatory removal remain one logical ply and may occupy two action tokens.
No accepted boundary has a pending removal.

An even twelve-ply prefix always leaves White to move. Corpus selection cannot
make side-to-move half White and half Black without violating the fixed-length
contract. Agent colour balance must therefore come from a paired evaluation
that assigns each tested system to both colours on the same immutable history.

## Recommended provisional composition

The recommended default is:

| Stratum | Prefixes |
| --- | ---: |
| Book | 22 |
| HumanDB | 21 |
| Perfect DB | 21 |
| **Total** | **64** |

This near-equal split is recommended because all three sources now have enough
eligible candidates and no evidence establishes a justified priority weight.
It also gives each stratum a meaningful separately reported sample. Book
receives the indivisible extra slot; that is an accounting choice, not a claim
that Book evidence is stronger.

The proposed later selection rules are:

1. **Book:** treat the Sanmill named lines and expert-curated plays as
   separately identified subtypes inside the 22-slot Book stratum. Cover
   expert eight-ply parents before adding a second child from one parent, and
   cover every Sanmill declared family before adding another member of a
   covered family. A later expert review must determine whether the 14
   structural parent groups map onto the seven author-named families; the two
   concepts are not assumed equivalent. Retain row 11's visual `c5`
   provenance, use no more than one record per exact history, and maximise
   exact-FEN/`ring16` diversity under a predeclared deterministic tiebreak.
2. **HumanDB:** rank complete histories by distinct-game count, then occurrence
   count, then history identity; skip cross-stratum endpoint/ring16 duplicates
   and take the first 21 remaining histories. Preserve their empirical outcome
   distributions and describe them only as frequent in the current PlayOK
   sample.
3. **Perfect DB:** take 21 routes from the already fixed 128-route pool by a
   predeclared route-order/diversity rule after the same cross-source
   deduplication. Do not regenerate routes in response to the observed
   candidate.

These rules are a proposal, not an executable freeze. The 22 Book slots are
not yet divided between the two subtypes. The exact parent/family mapping,
subtype allocation, selected identities, list hash, and review images must be
generated only after the product owner accepts the composition and the expert
reviews the parent-group interpretation.

## Inputs that remain excluded

The 15 additions in the maintainer's delivered `learned_openings.json` remain
an independent low-confidence candidate pool. They are not HumanDB frequency
evidence, the corrected Sanmill Book, or Perfect DB routes, and they are not
included in the proposed 64.

The expert Book delivery is no longer excluded as an unaudited input, but none
of its rows has automatic membership. Its uncontrolled app-play observations
remain contextual and cannot rank candidates or change source weights.

The existing movement/flying phase-coverage corpus also remains separate.
Placement-prefix selection cannot replace its later-phase coverage.

## Decision requested

The recommended product decision remains to accept the provisional
`22 Book / 21 HumanDB / 21 Perfect DB` composition. The newly measured expert
source does not justify changing that overall split.

Before freezing the 22 Book members, the Mill expert should review the 14
eight-ply structural parent groups against the human opening-family concepts
used in the delivered repertoire. The
[expert parent-and-child review guide](sanmill-layered-expert-book-parent-review.md)
turns that gate into four concrete questions:

1. which human opening family or strategic plan describes each parent;
2. which structural groups should be combined or split at the human-strategy
   level;
3. which resulting parents are core, useful, or optional/niche coverage; and
4. which continuations from the same exact parent are strategically distinct
   enough to supplement its primary twelve-ply child.

The guide defines the fourth point by strategic purpose rather than move
difference and provides all 14 parent histories plus every multi-child
continuation. The review remains semantic: it is not a request to recheck
legality, hashes, HumanDB frequency, row 11, Sanmill replay, or objective
strength.

Until that decision is recorded, status remains `needs_decision`; no final
prefix list, final corpus image-review package, candidate-versus-baseline run,
or launch authorization exists.
