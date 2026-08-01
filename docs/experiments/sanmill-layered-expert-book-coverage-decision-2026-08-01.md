# Expert Book unique-pattern coverage decision

Status: `expert_coverage_frozen_core_membership_not_frozen`

Decision date: 2026-08-01

Machine-readable decision:
[`sanmill-layered-expert-book-coverage-decision-2026-08-01.json`](sanmill-layered-expert-book-coverage-decision-2026-08-01.json)

This decision closes the Mill-expert selection gate for the reviewed
twelve-ply Book delivery. It freezes a complete expert coverage catalogue of
33 unique placement patterns. It does not freeze the balanced 64-prefix core,
an executable diagnostic protocol, or an evaluation launch.

## Authority and evidence boundary

The expert clarified two points after reading the concrete shortlist:

1. every unique placement pattern should be kept because apparently similar
   early parents can produce materially different counterplay; and
2. a different route ending in the same placement does not need a separate
   place, and the project may arrange the representatives.

The
[semantic evidence](../evidence/maintainer-book-opening-plays-semantic-review-2026-07-26.md)
records that statement and the strategic examples behind it. It does not
attribute D4 normalization to the expert. The repository adopts D4
normalization below as a technical coverage policy under the delegated
arrangement authority.

The decision is bound to reviewed source-audit identity
`1f6f9ceb8df36150ea401145e16c88cc25550622c1ad85a1b54a067183b9978d`.
The raw source and its reviewed correction remain immutable.

## Placement-pattern identity

One coverage pattern is one value of
`prefix_record.final.ring16_canonical_fen` after exactly twelve complete
logical plies. This key preserves side to move and per-side placement counts
while normalizing D4 rotations and reflections.

The audited reduction is:

| Boundary | Count |
| --- | ---: |
| Raw source records | 36 |
| Unique exact histories | 35 |
| Unique exact final FENs | 34 |
| Unique D4/`ring16` placement patterns | 33 |

All 36 records remain source provenance. Freezing 33 representatives does not
delete, rewrite, or relabel an alternate history.

The ordered catalogue identity is
`44519385ebfcb0e15207a23ec3914abb353c1638e5df6a52c4ffe3c53dba4617`.

[![Reviewed expert parent overview](assets/sanmill-layered-expert-book-parent-review-reviewed-source-2026-07-26/parent-overview.png)](assets/sanmill-layered-expert-book-parent-review-reviewed-source-2026-07-26/parent-overview.png)

## Frozen coverage catalogue

The machine-readable decision contains every coverage ID, source row,
variation ID, review alias, and canonical endpoint. The compact inventory is:

| Review family | Retained source rows or variants | Pattern count |
| --- | --- | ---: |
| P01 | row 1 better-`d1`; row 1 trap-`c5` | 2 |
| P02 | row 2 | 1 |
| P03 | rows 3, 8-11, 15, 17-19, 21-23, 28 | 13 |
| P04 | rows 4-5 | 2 |
| P05 | rows 6-7 | 2 |
| P06 | row 12 | 1 |
| P07 | row 13 | 1 |
| P08 | rows 24-25 | 2 |
| P09 | rows 26-27 | 2 |
| P10 | row 29 | 1 |
| P11 | row 30 | 1 |
| P12 | row 31 | 1 |
| P13-A | rows 32-33 | 2 |
| P14 | row 34 | 1 |
| P13-B | row 35 | 1 |
| **Total** | | **33** |

P13-A and P13-B share a D4-normalized eight-ply parent, but their twelve-ply
canonical endpoints are distinct. They therefore remain separate coverage
patterns.

## Collapsed P03 routes

Only one final `ring16` pattern contains more than one audited record:

| P03 child | Source row | Disposition |
| ---: | ---: | --- |
| 001 | 3 | Coverage representative; explicitly selected by the expert |
| 006 | 14 | Alternate transposition; same exact final FEN |
| 008 | 16 | Different exact history and exact FEN; D4-equivalent endpoint |
| 012 | 20 | Exact-history duplicate of row 14; same exact final FEN |

Rows 14, 16, and 20 remain linked provenance behind
`expert-pattern-004`; they do not consume three additional coverage slots.

[![Reviewed P03 children](assets/sanmill-layered-expert-book-parent-review-reviewed-source-2026-07-26/child-overviews/P03.png)](assets/sanmill-layered-expert-book-parent-review-reviewed-source-2026-07-26/child-overviews/P03.png)

## Deterministic representative policy

If a later reviewed source adds another route to an existing placement
pattern, choose its representative in this order:

1. an explicit expert primary;
2. higher exact-history support in the frozen HumanDB sample;
3. typed source evidence before a visual interpretation; and
4. source row, then variation ID.

HumanDB support is occurrence evidence only. It is not a strength label. A
different representative requires a new versioned decision; it must not
silently change this catalogue.

## Core benchmark and diagnostic separation

The 33-pattern catalogue serves two different products:

- The **Expert Book diagnostic suite** contains all 33 representatives and
  reports them separately. Its prefix membership is frozen here, but its
  opponent, workload, pairing, termination, result rule, and launch contract
  are not.
- The **balanced 64-prefix core** may use a documented subset of these
  patterns inside its Book stratum while retaining separately reported
  HumanDB and Perfect DB strata. Its provisional composition remains
  `22 Book / 21 HumanDB / 21 Perfect DB`; neither that composition nor any
  member is frozen here.

This separation makes `must keep` mean preserved and testable without forcing
all expert patterns into the primary source-balanced statistic. A result from
the diagnostic suite must not be pooled into the core result unless a future
contract prespecifies such a statistic.

## Remaining product work

No further Mill-expert tier or primary-row answer is required. The remaining
source-only work is to:

1. decide whether to accept the provisional `22/21/21` core composition;
2. select the internal 22-slot Book subset across Sanmill named lines and this
   expert catalogue with cross-source `ring16` deduplication;
3. freeze the resulting core identities and render its final review package;
   and
4. define any diagnostic execution protocol separately.

No candidate model was loaded, no game was played, and no evaluation or
training launch is authorized by this decision.
