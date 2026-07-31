# Expert Book Semantic Review Follow-Up

Status: `concrete_proposal_ready_for_expert_correction`

Initial request: 2026-07-26

Latest response: 2026-07-31

The Mill expert has already provided substantial family and child-line notes.
This follow-up does not ask him to repeat legality checks, move transcription,
board review, HumanDB counts, or the material he has already supplied.

The complete disposition is in the
[semantic review evidence](../evidence/maintainer-book-opening-plays-semantic-review-2026-07-26.md).
The corrected technical source is the
[reviewed-source audit](../evidence/sanmill-layered-expert-book-reviewed-source-audit-2026-07-26.md).

The current rendered package is bound to reviewed audit identity
`1f6f9ceb8df36150ea401145e16c88cc25550622c1ad85a1b54a067183b9978d`
and manifest identity
`1349107cc616b44a7017af2db1734f04a267c286be5155f449fed93c552bf568`.
It preserves all 14 parent groups and corrects P03 row 19 to `c4 d5 e3 d1`.

[![Reviewed parent overview](assets/sanmill-layered-expert-book-parent-review-reviewed-source-2026-07-26/parent-overview.png)](assets/sanmill-layered-expert-book-parent-review-reviewed-source-2026-07-26/parent-overview.png)

## Remaining expert inputs

### 1. P14

Provide a short family or plan description for P14, or state that it has no
recognized family name.

```text
P14: ...
```

### 2. P03 extended-family partition

The expert has already established that P03's sixteen children should not all
be treated as one family merely because they share the same first eight plies.
What remains is the grouping itself. Source-row identifiers are sufficient.

```text
P03 family A: rows ...
P03 family B: rows ...
...
```

[![Reviewed P03 children](assets/sanmill-layered-expert-book-parent-review-reviewed-source-2026-07-26/child-overviews/P03.png)](assets/sanmill-layered-expert-book-parent-review-reviewed-source-2026-07-26/child-overviews/P03.png)

### 3. Coverage priorities

Assign each resulting family one tier. This is a repertoire-coverage judgment,
not a claim that the line is winning.

```text
Core: ...
Useful: ...
Optional/niche: ...
```

### 4. Primary and additional children

For each multi-child exact parent, identify one primary representative and any
additional child that teaches a genuinely different plan. P01 already marks
the `d1` continuation as better and the `c5` continuation as a trap. The
remaining groups are P03, P04, P05, P08, P09, and P13-A.

```text
Pxx primary: row ...
Pxx additional distinct: rows ... because ...
Pxx same-plan/redundant: rows ...
```

The existing annotations already support a strategic difference between the
two P04 children and between the two P05 children, but they do not select a
primary.

## Response disposition

The expert supplied `Interrupted Knight` for P14 and a child-level P03
classification. A later clarification identifies P03 child 003 as the outer
Parallel Mill Rush variant and child 004 as the inner variant. He explicitly
selects child 001, source row 3, as the primary Black response.

The frozen source audit establishes a stronger technical relationship for the
ambiguous child 006 note. Child 001 uses `b2 c5 c4 e5` at plies 9-12, while
child 006 uses `c4 e5 b2 c5`. They have different exact histories but the same
final FEN and `ring16` orbit. Child 012 is an exact-history duplicate of child
006. Children 006 and 012 are therefore retained as provenance but proposed as
same-plan redundant continuations behind primary child 001.

The response does not assign coverage tiers and does not select primary rows
for P04, P05, P08, P09, or P13-A. Rather than repeat the abstract questions,
the
[breadth-first shortlist](sanmill-layered-expert-book-shortlist-proposal-2026-07-31.md)
presents a concrete grouping and explicit primary choices for correction. The
expert's difficulty distinguishing lines from their common eight-ply parent is
supporting evidence for retaining the twelve-ply child boundary, not a reason
to collapse the v2 contract back to eight plies.

## Product decision after expert input

Even after this semantic review closes, the product owner must still approve:

- the provisional `22 Book / 21 HumanDB / 21 Perfect DB` composition; and
- how the 22 Book places are divided between the corrected Sanmill named-line
  subtype and the expert-curated subtype.

Only after both decisions may the repository generate a proposed immutable
64-prefix list and final review images. No candidate evaluation or training is
authorized by this follow-up.
